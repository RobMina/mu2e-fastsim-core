#!/bin/bash
#SBATCH -J m2fs_valid
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/%u/mu2efastsim/probe/slurm-%j.out
# UVA: the fastsim inside the SIF on one A100, with the REAL CRVConditions v6_0 tables (copy-compute-delete: CVMFS
# Stratum-1 -> tmpfs /tmp, deleted on exit, never on /scratch). Image checks, pytest, R5 GPU vs CPU, V4 photons and
# SiPM (JAX twin and fused device response vs the numpy ports, synthetic muon CrvSteps), real-table response bench.
# usage: sbatch [--export=ALL,SIF=...,NEV=events] validate_uva.sh
F=${M2FS_UVA:-/scratch/$USER/mu2efastsim}
SIF=${SIF:-$F/mu2efastsim.sif}; NEV=${NEV:-20000}
umask 077
O=$F/probe/valid_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
module load apptainer >/dev/null 2>&1
L=${TMPDIR:-/tmp}/crvfs_valid_j${SLURM_JOB_ID:-0}_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "$(date -Is) $(hostname) deleted $L" >> $O/cleanup.txt; echo "deleted $L"' EXIT   # file record: tee dies with the job
trap 'exit 143' TERM INT
OR=$(readlink -f $O)
A() { apptainer exec --nv --cleanenv --env CRVFS_LOOKUP_DIR=$L,CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} -B $L -B $OR --pwd /opt/mu2e-fastsim "$SIF" "$@"; }
sec() { echo; echo "==== $*"; }
for d in ${TMPDIR:-/tmp}/crvfs_*_j*; do [ -d "$d" ] && [ "$d" != "$L" ] && [ -O "$d" ] || continue; j=${d##*_j}; j=${j%%_*}; squeue -h -j "$j" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done   # only dirs of finished jobs: several jobs share a node
sec "node, image"; echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$O sif=$SIF $(apptainer --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
t0=$SECONDS; A python -c "import jax; print('jax', jax.__version__, jax.devices())"; echo "cold start $((SECONDS-t0)) s"
A python -c "import jax, sys; sys.exit(jax.default_backend() != 'gpu')" 2>/dev/null || { echo "JAX is not on the GPU in $SIF: abort (no GPU-rate SUs on CPU work)"; exit 5; }
A cat /opt/mu2e-fastsim/VERSION; A python --version
sec "fetch real tables into $L (tmpfs)"
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  A python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done
df -h $L | tail -1; ls -la $L
A python -m crvfastsim.response.synthsteps --events $NEV --out $L/steps.root || exit 4
sec "pytest (GPU, in the image)"
A python -m pytest -q -rfE -p no:cacheprovider crv/tests/test_reco_jax.py crv/tests/test_photons.py core/tests 2>&1 | tail -6
sec "R5/P6 repro: GPU per seed, then CPU against GPU"
A python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $OR/repro_gpu > $O/repro_gpu.txt 2>&1
RX=$(ls $O/repro_gpu/repro_*.npz 2>/dev/null | head -1); RX=${RX:+$(readlink -f $RX)}
A env JAX_PLATFORMS=cpu python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $OR/repro_cpu \
  ${RX:+--against $RX} > $O/repro_cpu.txt 2>&1
for r in gpu cpu; do echo "-- repro_$r"; grep -h -oE '"(backend|pass_bit_identical|pass_statistical|same_seed_identical)": [^,}]*' \
  $O/repro_$r/repro_stats.json 2>/dev/null | tr '\n' ' '; echo; done
sec "V4 photons (JAX twin vs numpy port), V4 SiPM (JAX twin, fused device response), real v6_0 tables"
A python -m crvfastsim v4photons $L/steps.root --jax --no-stock --seed 1 --outdir $OR/v4ph > $O/v4ph.txt 2>&1; echo "v4photons exit=$?"
A python -m crvfastsim v4sipm $L/steps.root --jax --no-stock --seed 1 --outdir $OR/v4si_jax > $O/v4si_jax.txt 2>&1; echo "v4sipm jax exit=$?"
A python -m crvfastsim v4sipm $L/steps.root --device --no-stock --seed 1 --outdir $OR/v4si_dev > $O/v4si_dev.txt 2>&1; echo "v4sipm device exit=$?"
for f in v4ph v4si_jax v4si_dev; do echo "-- $f"; tail -8 $O/$f.txt | cut -c1-300; done
sec "response bench, real table 3200_0 and real photon map, one GPU"
B="python -m crvfastsim.response.bench --repeat 3"
A $B --table 3200_0 --rows 1000000 --numpy-rows 0 --seed 20261021 --outdir $OR/ph_real > $O/ph_real.txt 2>&1; echo "photons $(tail -1 $O/ph_real.txt)"
A $B --stage sipm --photon-map $L/photonMap.root --sipms 500000 --numpy-sipms 0 --seed 20261121 --outdir $OR/si_real > $O/si_real.txt 2>&1
echo "sipm $(tail -1 $O/si_real.txt)"
sec "reco chain bench in the image (GPU), and JAX CPU on ${SLURM_CPUS_ON_NODE:-?} cores"
A python -m crvfastsim reco-bench synthetic --stage chain --max-events 200000 --length 12 --batch 65536 --repeat 3 \
  --outdir $OR/gpu_chain > $O/gpu_chain.txt 2>&1; echo "chain gpu exit=$?"
grep -hoE '"(first_call_s|events_per_s)": [^,}]*' $O/gpu_chain/*.json 2>/dev/null | tr '\n' ' '; echo
timeout 900 apptainer exec --cleanenv --env JAX_PLATFORMS=cpu -B $OR --pwd /opt/mu2e-fastsim "$SIF" python -m crvfastsim reco-bench synthetic \
  --stage chain --max-events 20000 --length 12 --repeat 2 --outdir $OR/cpu_chain > $O/cpu_chain.txt 2>&1; echo "chain cpu exit=$?"
grep -hoE '"(first_call_s|events_per_s)": [^,}]*' $O/cpu_chain/*.json 2>/dev/null | tr '\n' ' '; echo
find $O -name '*.npz' -delete   # repro arrays derive from synthetic inputs, but keep the run dir small
sec "done"; echo "outdir $O elapsed $SECONDS s"; echo exit=0
