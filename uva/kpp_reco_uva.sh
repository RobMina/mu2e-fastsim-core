#!/bin/bash
#SBATCH -J m2fs_kpp
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
# UVA: CRV reco checks on real KPP data (run 124155, CrvRecoTableDumper reference) inside the SIF on one A100:
#   V-R3 (reco-vr3 --jax: JAX coincidence finder vs stock clusters) on GPU and on CPU as a control;
#   V-R2 (reco-vr2jax: JAX pulse fit vs stock pulses) on GPU.
# The reference file is Mu2e-internal: it is streamed into the job by core/uva/kpp_stream_uva.sh (srun --overlap into
# node tmpfs, crvfs_kpp_j<job>_<pid>), never written to the cluster scratch, and deleted on exit. Outputs (stats only) go to an unsynced
# dir that the stream script copies back to FNAL and removes. Submit only through kpp_stream_uva.sh.
# Site settings come from the environment (site.env.example): FASTSIMKIT_SCRATCH (work area on the cluster), the
# account at submission (sbatch -A "$FASTSIMKIT_SLURM_ACCOUNT"), and the log path (sbatch -o "$FASTSIMKIT_SCRATCH/kpp/slurm-%j.out").
F=${M2FS_UVA:-${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH (work area on the cluster scratch; sbatch --export=ALL,FASTSIMKIT_SCRATCH=...)}}
SIF=${SIF:-$F/mu2efastsim.sif}; NAME=${NAME:-crvreco_data.124155_000001.root}
umask 077
O=$F/kpp/kpp_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
module load apptainer >/dev/null 2>&1
L=${TMPDIR:-/tmp}/crvfs_kpp_j${SLURM_JOB_ID:-0}_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "$(date -Is) $(hostname) deleted $L" >> $O/cleanup.txt; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
for d in ${TMPDIR:-/tmp}/crvfs_*_j*; do [ -d "$d" ] && [ "$d" != "$L" ] && [ -O "$d" ] || continue; j=${d##*_j}; j=${j%%_*}; squeue -h -j "$j" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done   # only dirs of finished jobs: several jobs share a node
OR=$(readlink -f $O)
A() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} -B $L -B $OR --pwd /opt/mu2e-fastsim "$SIF" "$@"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$O sif=$SIF"; A grep -E "umbrella|crv" /opt/mu2e-fastsim/VERSION
A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != 'gpu')" 2>/dev/null \
  || { echo "JAX is not on the GPU: abort"; exit 5; }
echo "$(hostname) $L $O" > $F/kpp/inbox_${SLURM_JOB_ID}   # read by kpp_stream_uva.sh
echo "waiting for $L/ready (up to 20 min)"
for i in $(seq 240); do [ -f $L/ready ] && break; sleep 5; done
rm -f $F/kpp/inbox_${SLURM_JOB_ID}
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
ls -la $L | grep -v ready; echo "md5 $(md5sum $L/$NAME | cut -c1-32)"
IN=$L/$NAME
sec() { echo; echo "==== $*"; }
sec "V-R3 coincidences: JAX on the A100 | JAX CPU control in parallel"
A python -m crvfastsim reco-vr3 $IN --jax --outdir $OR/vr3_gpu > $O/vr3_gpu.txt 2>&1 & a=$!
A env JAX_PLATFORMS=cpu python -m crvfastsim reco-vr3 $IN --jax --outdir $OR/vr3_cpu > $O/vr3_cpu.txt 2>&1 & c=$!
wait $a
sec "V-R2 pulses: JAX on the A100"
A python -m crvfastsim reco-vr2jax $IN --length 36 --rounds 3 --outdir $OR/vr2_gpu > $O/vr2_gpu.txt 2>&1
wait $c
for r in vr3_gpu vr3_cpu vr2_gpu; do
  echo "-- $r"; j=$(ls $O/$r/*_stats.json 2>/dev/null | head -1)
  if [ -n "$j" ]; then A python - "$(readlink -f $j)" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
drop = ("unmatched_fast_keys", "unmatched_ref_keys", "examples", "timers")
print(json.dumps({k: v for k, v in j.items() if k not in drop})[:1800])
EOF
  else grep -h -m2 -E "Fatal|Error" $O/$r.txt | cut -c1-300; fi
done
echo "outdir $O elapsed $SECONDS s"; echo exit=0
