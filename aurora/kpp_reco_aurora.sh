#!/bin/bash
#PBS -N m2fs_kpp
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_kpp.out
# Aurora: CRV reco checks on real KPP data (run 124155, CrvRecoTableDumper reference) on Intel PVC:
#   V-R2 (reco-vr2jax: JAX pulse fit vs stock pulses) and V-R3 (reco-vr3 --jax: JAX coincidence finder vs stock
#   clusters), on one tile, with the same commands on the node CPU as a control.
# The reference file is Mu2e-internal: it is NOT staged on Flare. The submitter streams it into the job over ssh
# (the job waits for $INBOX/ready); it lives only in node tmpfs (crvfs_kpp_<pid>) on the compute node and is deleted
# on exit. Submit and stream with core/aurora/kpp_stream.sh (run on a GPVM, through the Aurora ssh master).
# Site settings (site.env.example): submit with qsub -A "$ALCF_PROJECT" -v M2FS_FLARE=<work area on Flare>,...
F=${M2FS_FLARE:?set M2FS_FLARE (work area on Flare; qsub -v M2FS_FLARE=...)}
FL=$F; V=$FL/venv/bin/python; CODE=$FL/code; NAME=${NAME:-crvreco_data.124155_000001.root}
umask 077   # outputs describe Mu2e data: private to the user on the shared project file system
O=$FL/bench/kpp_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O; chmod 700 $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git log -1 --format="umbrella %h %s"; git submodule status
export LD_LIBRARY_PATH=$FL/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT
L=${TMPDIR:-/tmp}/crvfs_kpp_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
echo "$(hostname) $L $O" > $FL/bench/kpp_inbox_${PBS_JOBID%%.*}   # read by kpp_stream.sh
echo "waiting for $L/ready (up to 20 min)"
for i in $(seq 240); do [ -f $L/ready ] && break; sleep 5; done
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
ls -la $L | grep -v ready; echo "md5 $(md5sum $L/$NAME | cut -c1-32)"
IN=$L/$NAME
sec() { echo; echo "==== $*"; }
sec "V-R3 coincidences (JAX) on tile 0 | V-R2 pulses (JAX) on tile 1 | CPU controls"
ZE_AFFINITY_MASK=0 taskset -c 1-12 $V -m crvfastsim reco-vr3 $IN --jax --outdir $O/vr3_xpu > $O/vr3_xpu.txt 2>&1 & a=$!
ZE_AFFINITY_MASK=1 taskset -c 13-24 $V -m crvfastsim reco-vr2jax $IN --length 36 --rounds 3 --outdir $O/vr2_xpu > $O/vr2_xpu.txt 2>&1 & b=$!
JAX_PLATFORMS=cpu taskset -c 25-51 $V -m crvfastsim reco-vr3 $IN --jax --outdir $O/vr3_cpu > $O/vr3_cpu.txt 2>&1 & c=$!
JAX_PLATFORMS=cpu taskset -c 53-103 $V -m crvfastsim reco-vr2jax $IN --length 36 --rounds 3 --outdir $O/vr2_cpu > $O/vr2_cpu.txt 2>&1 & d=$!
wait $a $b $c $d
for r in vr3_xpu vr3_cpu vr2_xpu vr2_cpu; do
  echo "-- $r"; j=$(ls $O/$r/*_stats.json 2>/dev/null | head -1)
  if [ -n "$j" ]; then $V - "$j" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
drop = ("unmatched_fast_keys", "unmatched_ref_keys", "examples", "timers")
print(json.dumps({k: v for k, v in j.items() if k not in drop})[:1800])
EOF
  else grep -h -m2 -E "Fatal|Error" $O/$r.txt | cut -c1-300; fi
done
echo "outdir $O"; echo exit=0
