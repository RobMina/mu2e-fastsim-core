#!/bin/bash
#SBATCH -J m2fs_p2
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/%u/mu2efastsim/kpp/slurm-%j.out
# UVA: P2 of the CRV fused device response (photons -> SiPM charges) on ONE whole A100-80 (SXM4, as Polaris; not a
# MIG slice), on real G4 CrvSteps and on the synthetic sample of Polaris PBS 7764070 for a direct comparison.
# The G4 step shards and the CRVCalib/CRVPhoton CSVs are Mu2e-internal: streamed into job tmpfs by kpp_stream_uva.sh
# (srun --overlap), never on /scratch, deleted on exit with a record in cleanup.txt; the v6_0 lookup tables come from
# CVMFS into the same tmpfs. Code: a fresh clone of the public umbrella at $REF (the SIF predates the batching fixes),
# run with the SIF's Python stack. Outputs (stats only) go to an unsynced dir the stream script copies back to FNAL.
# usage (laptop, UVA network): JOB=p2_uva.sh kpp_stream_uva.sh mu2egpvm02 mu2e/fastsim_uva <shards and CSVs on the GPVM>
F=${M2FS_UVA:-/scratch/$USER/mu2efastsim}
SIF=${SIF:-$F/mu2efastsim.sif}; REF=${REF:-main}; NEV=${NEV:-20000}; REPEAT=${REPEAT:-3}
umask 077
O=$F/kpp/p2_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
module load apptainer >/dev/null 2>&1
L=${TMPDIR:-/tmp}/crvfs_p2_j${SLURM_JOB_ID:-0}_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "$(date -Is) $(hostname) deleted $L" >> $O/cleanup.txt; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
for d in ${TMPDIR:-/tmp}/crvfs_*_j*; do [ -d "$d" ] && [ "$d" != "$L" ] && [ -O "$d" ] || continue; j=${d##*_j}; j=${j%%_*}; squeue -h -j "$j" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done
OR=$(readlink -f $O)
CODE=$F/p2_code
if [ -d $CODE/.git ]; then git -C $CODE fetch -q origin && git -C $CODE checkout -q -f $REF && git -C $CODE reset -q --hard origin/$REF
else git clone -q https://github.com/RobMina/mu2e-fastsim.git $CODE && git -C $CODE checkout -q $REF; fi
git -C $CODE submodule update -q --init --recursive
CR=$(readlink -f $CODE)
A() { apptainer exec --nv --cleanenv --env "PYTHONPATH=$CR/crv:$CR/core:$CR/tracker,CRVFS_LOOKUP_DIR=$L,XLA_PYTHON_CLIENT_PREALLOCATE=false,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
      -B $L -B $OR -B $CR --pwd $CR "$SIF" "$@"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$O sif=$SIF ref=$REF"
git -C $CODE log -1 --format="umbrella %h %s" | cut -c1-100; git -C $CODE submodule status | cut -c1-60
nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm,mig.mode.current --format=csv,noheader; nvidia-smi -L
A python -c "import jax, sys, crvfastsim, os; print(jax.devices(), os.path.dirname(crvfastsim.__file__)); sys.exit(jax.default_backend() != 'gpu')" \
  || { echo "JAX not on the GPU or code not importable: abort"; exit 5; }
A python -c "from crvfastsim.response import device; device.photon_budget" || { echo "code at $REF lacks memory-sized batching"; exit 5; }
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  A python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f >/dev/null || { echo "fetch $f failed"; exit 4; }
done
echo "$(hostname) $L $O" > $F/kpp/inbox_${SLURM_JOB_ID}   # read by kpp_stream_uva.sh
echo "waiting for $L/ready (up to 20 min)"
for i in $(seq 240); do [ -f $L/ready ] && break; sleep 5; done
rm -f $F/kpp/inbox_${SLURM_JOB_ID}
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
CAL=$(ls $L/crvcalib*.csv | head -1); DEV=$(ls $L/crvphoton*.csv | head -1)
ls -la $L | grep -v ready | awk 'NR>1{print $5, $9}'
rc=0
for S in $L/crvsteps.*.root; do
  n=$(basename $S .root)
  A python -m crvfastsim v4sipm $S --device --no-stock --seed 1 --repeat $REPEAT --calib $CAL --deviation $DEV \
    --outdir $OR/$n > $O/$n.txt 2>&1 || { echo "FAILED $n"; rc=6; }
  echo "== $n"; tail -9 $O/$n.txt | cut -c1-300
done
A python -m crvfastsim.response.synthsteps --events $NEV --out $L/synth_steps.root > $O/synth.txt 2>&1 &&
  A python -m crvfastsim v4sipm $L/synth_steps.root --device --no-stock --seed 1 --repeat $REPEAT --outdir $OR/synth >> $O/synth.txt 2>&1 \
  || { echo "FAILED synthetic"; rc=7; }
echo "== synthetic"; tail -9 $O/synth.txt | cut -c1-300
echo "outdir $O elapsed $SECONDS s"; echo exit=$rc; exit $rc
