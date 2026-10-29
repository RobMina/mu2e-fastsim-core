#!/bin/bash
#SBATCH -J m2fs_chain
#SBATCH -A ldmxuva
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80gb
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --mem=900G
#SBATCH -t 01:00:00
#SBATCH -o /scratch/%u/mu2efastsim/kpp/slurm-%j.out
# UVA production template: the fast CRV chain (runchain: transport -> tier 1b -> photons -> SiPM -> waveforms ->
# pulses -> coincidences) on a whole 8 x A100-80 node, one process per GPU, response AND reco on the GPU (unlike
# Aurora, where reco ran on the CPU), 16 cores per process. Code: the SIF's Python stack with a fresh git clone of
# the umbrella at $REF (the SIF may predate runchain --emit), PYTHONPATH pointing at it.
# Inputs (CRY gen tables, tier-1b library, CRVCalib/CRVPhoton CSVs, GDML) are streamed by kpp_stream_uva.sh into
# node tmpfs; the v6_0 lookup tables come from CVMFS into tmpfs; all deleted on exit (cleanup.txt). Per-shard JSONs
# and the merge go back to the GPVM. usage (laptop, UVA network):
#   JOB=chain_node_uva.sh EXPORTS=NPROC=8,RUNS=4 kpp_stream_uva.sh mu2egpvm03 mu2e/fastsim_uva <files...>
# Tier 1b (crvfastsim.tier1b_assets): T1B=strict_v3 (default; stream library.root, the topology sidecar and
# templates_v3.npz, each with its .json), T1B=strict_v2 (templates_v2.npz instead) or T1B=legacy (stream
# lib_prod_as_20files.root; runs with --library, the sampling before 2026-10-28).
F=${M2FS_UVA:-/scratch/$USER/mu2efastsim}
SIF=${SIF:-$F/mu2efastsim.sif}; NP=${NPROC:-8}; MAXEV=${MAXEV:-}; REF=${REF:-main}
umask 077
O=$F/kpp/chain_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
module load apptainer >/dev/null 2>&1
L=${TMPDIR:-/tmp}/crvfs_chain_j${SLURM_JOB_ID:-0}_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "$(date -Is) $(hostname) deleted $L" >> $O/cleanup.txt; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
for d in ${TMPDIR:-/tmp}/crvfs_*_j*; do [ -d "$d" ] && [ "$d" != "$L" ] && [ -O "$d" ] || continue; j=${d##*_j}; j=${j%%_*}; squeue -h -j "$j" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done   # only dirs of finished jobs: several jobs share a node
OR=$(readlink -f $O)
CODE=${CODE_DIR:-$F/chain_code}
if [ "${CODE_READY:-0}" != 1 ]; then   # array runs prepare one checkout on the login node and set CODE_READY=1
  if [ -d $CODE/.git ]; then git -C $CODE fetch -q origin && git -C $CODE checkout -q $REF && git -C $CODE reset -q --hard origin/$REF 2>/dev/null || git -C $CODE checkout -q $REF
  else git clone -q https://github.com/RobMina/mu2e-fastsim.git $CODE && git -C $CODE checkout -q $REF; fi
  git -C $CODE submodule update -q --init --recursive
fi
CR=$(readlink -f $CODE)
A() { apptainer exec --nv --cleanenv --env "PYTHONPATH=$CR/crv:$CR/core:$CR/tracker,CRVFS_LOOKUP_DIR=$L,CRVFS_DATA=$L,$1" \
      -B $L -B $OR -B $CR --pwd $CR "$SIF" "${@:2}"; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} out=$O nproc=$NP maxev=${MAXEV:-all} sif=$SIF"
git -C $CODE log -1 --format="umbrella %h %s" | cut -c1-120; git -C $CODE submodule status | cut -c1-60
nvidia-smi -L | head -8
A CUDA_VISIBLE_DEVICES=0 python -c "import jax, sys, crvfastsim, os; print(jax.devices(), os.path.dirname(crvfastsim.__file__)); sys.exit(jax.default_backend() != 'gpu')" \
  || { echo "JAX not on the GPU or code not importable: abort"; exit 5; }
A CUDA_VISIBLE_DEVICES= python -m crvfastsim runchain-merge --help >/dev/null 2>&1 || { echo "runchain-merge missing in $REF"; exit 5; }
echo "$(hostname) $L $O" > $F/kpp/inbox_${SLURM_JOB_ID}
echo "waiting for $L/ready (up to 30 min)"; tw=$SECONDS
for i in $(seq 360); do [ -f $L/ready ] && break; sleep 5; done
echo "input wait $((SECONDS-tw)) s (charged; streaming time)"
rm -f $F/kpp/inbox_${SLURM_JOB_ID}
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  A CUDA_VISIBLE_DEVICES= python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f >/dev/null || { echo "fetch $f failed"; exit 4; }
done
ls -la $L | grep -v ready | awk 'NR>1{print $5, $9}'
GEN=($(ls $L/gen.*.root)); LIB=$(ls $L/lib*.root | head -1); CAL=$(ls $L/crvcalib*.csv | head -1)
DEV=$(ls $L/crvphoton*.csv 2>/dev/null | head -1); GD=$(ls $L/*.gdml | head -1)
echo "gen files ${#GEN[@]}; library $(basename $LIB); calib $(basename $CAL); gdml $(basename $GD)"
T1B=${T1B:-strict_v3}
if [ "$T1B" = legacy ]; then TIER1B="--library $LIB"
else TV=${T1B#strict_}; for f in library.root topology_sidecar_strict_v2_v2.npz topology_sidecar_strict_v2_v2.npz.json templates_$TV.npz templates_$TV.npz.json; do
  [ -f $L/$f ] || { echo "tier-1b set $T1B: $f not streamed"; exit 3; }; done; TIER1B="--tier1b $T1B --tier1b-dir $L"; fi
echo "tier 1b: $TIER1B"
CPP=$(( ${SLURM_CPUS_ON_NODE:-128} / NP ))
for FPP in ${RUNS:-1}; do
  R=$O/fpp$FPP; mkdir -p $R
  t0=$(date +%s.%N); pids=()
  for i in $(seq 0 $((NP-1))); do
    fl=(); for k in $(seq 0 $((FPP-1))); do fl+=(${GEN[$(((i*FPP+k) % ${#GEN[@]}))]}); done
    PIN=""   # one process: the Slurm cgroup already holds the job's cores (a fixed list fails on a shared node)
    [ $NP -gt 1 ] && PIN="taskset -c $((i*CPP))-$((i*CPP+CPP-1))"   # whole-node jobs only: split the node's cores
    ( A CUDA_VISIBLE_DEVICES=$i,XLA_PYTHON_CLIENT_PREALLOCATE=false,JAX_ENABLE_X64=1 \
        $PIN python -m crvfastsim runchain "${fl[@]}" $TIER1B --calib $CAL \
        ${DEV:+--deviation $DEV} --gdml $GD --jax --seed $((${SEED_BASE:-20261021}+i)) ${MAXEV:+--max-events $MAXEV} \
        --emit $R/shard_$i.json --outdir $R/p$i ) > $R/p$i.txt 2>&1 & pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=$((rc+1)); done
  WALL=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')
  echo "== pass files/process=$FPP: processes $NP, failed $rc, wall ${WALL} s"
  for i in $(seq 0 $((NP-1))); do [ -f $R/shard_$i.json ] || { echo "-- p$i failed:"; grep -m3 -E "Error|Fatal|die:|Traceback" $R/p$i.txt | cut -c1-250; }; done
  A CUDA_VISIBLE_DEVICES= python -m crvfastsim runchain-merge "$R/shard_*.json" --wall $WALL --outdir $R/merge 2>&1 | \
    grep -E '"(shards|generated_events|events_with_steps|wall_s|slowest_shard_wall_s|accepted_events_per_node_hour)"'
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
echo "outdir $O elapsed $SECONDS s"; echo exit=0
