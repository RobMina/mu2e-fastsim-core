#!/bin/bash
#PBS -N m2fs_vr1
#PBS -A GenesisHackathonOct26
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_vr1.out
# Aurora: V-R1 (JAX waveform synthesis vs stock CrvWaveforms/CrvDigi) on Intel PVC, MC 001400_00014516:
#   noise off: crvreco_mcnonoise (every stock digi sample-identical); noise on: crvreco_mc vs the noise-free one.
# Both references are Mu2e-internal: streamed into tmpfs by kpp_stream.sh (NAMES = space-separated basenames), deleted
# on exit, never on Flare. Same commands on the node CPU as a control. BRANCH: a crv branch to test (optional).
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv/bin/python; CODE=$FL/code
umask 077
O=$FL/bench/vr1_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O; chmod 700 $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
CRV=$CODE/crv
if [ -n "${BRANCH:-}" ]; then   # test a crv branch without moving the umbrella pins (first on PYTHONPATH)
  CRV=$FL/crv_$BRANCH
  if [ -d $CRV/.git ]; then git -C $CRV fetch -q && git -C $CRV reset -q --hard origin/$BRANCH
  else git clone -q -b $BRANCH https://github.com/RobMina/mu2e-crv-fastsim.git $CRV; fi
fi
export PYTHONPATH=$CRV:$CODE/core${PYTHONPATH:+:$PYTHONPATH}
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git log -1 --format="umbrella %h %s"; git -C $CRV log -1 --format="crv %h %s"
export LD_LIBRARY_PATH=$FL/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT
L=/tmp/crvfs_vr1_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
echo "$(hostname) $L $O" > $FL/bench/kpp_inbox_${PBS_JOBID%%.*}
echo "waiting for $L/ready (up to 30 min)"
for i in $(seq 360); do [ -f $L/ready ] && break; sleep 5; done
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
ls -la $L | grep -v ready
NN=$L/crvreco_mcnonoise.001400_00014516.root; NZ=$L/crvreco_mc.001400_00014516.root
echo; echo "==== V-R1 noise off: JAX on tile 0 | CPU control"
ZE_AFFINITY_MASK=0 taskset -c 1-12 $V -m crvfastsim reco-vr1 $NN --jax --outdir $O/vr1_xpu > $O/vr1_xpu.txt 2>&1 & a=$!
JAX_PLATFORMS=cpu taskset -c 13-51 $V -m crvfastsim reco-vr1 $NN --jax --outdir $O/vr1_cpu > $O/vr1_cpu.txt 2>&1 & b=$!
echo "==== V-R1 noise on: JAX on tile 1 | CPU control"
ZE_AFFINITY_MASK=1 taskset -c 53-64 $V -m crvfastsim reco-vr1 $NZ --noise --jax --nonoise-ref $NN --outdir $O/vr1n_xpu > $O/vr1n_xpu.txt 2>&1 & c=$!
JAX_PLATFORMS=cpu taskset -c 65-103 $V -m crvfastsim reco-vr1 $NZ --noise --jax --nonoise-ref $NN --outdir $O/vr1n_cpu > $O/vr1n_cpu.txt 2>&1 & d=$!
wait $a $b $c $d
for r in vr1_xpu vr1_cpu vr1n_xpu vr1n_cpu; do
  echo "-- $r"; j=$(ls $O/$r/*_stats.json 2>/dev/null | head -1)
  if [ -n "$j" ]; then $V - "$j" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
def short(v):
    return v if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 200 else "<%d>" % len(v)
print(json.dumps({k: short(v) for k, v in j.items()})[:2000])
PY
  else grep -h -m2 -E "Fatal|Error" $O/$r.txt | cut -c1-300; fi
done
echo "outdir $O"; echo exit=0
