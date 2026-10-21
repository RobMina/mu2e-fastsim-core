#!/bin/bash
#PBS -N m2fs_chain
#PBS -A GenesisHackathonOct26
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_chain.out
# Aurora production template: the fast CRV chain (runchain: transport -> tier 1b -> photons -> SiPM -> waveforms ->
# pulses -> coincidences) on a full node, one process per PVC tile (12), response on the tile, reco on the node CPU
# (--reco-cpu), 8 cores per process (0 and 52 reserved). Inputs: CRY gen tables (one per process, cycled), the tier-1b
# library, CRVCalib/CRVPhoton CSVs, GDML: all streamed by kpp_stream.sh into tmpfs and deleted on exit (the lookup
# tables come from CVMFS, also into tmpfs). Never on Flare; per-shard JSONs and the merge go back to the GPVM.
# usage (from the laptop): EXTRA_V=NPROC=12,RUNS=1 kpp_stream.sh mu2egpvm03 chain_node_aurora.sh <files...>
# (RUNS: space-separated files-per-process passes; qsub -v cannot carry spaces or commas, so set RUNSFILE instead)
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv/bin/python; CODE=$FL/code; NP=${NPROC:-12}; MAXEV=${MAXEV:-}
[ -n "${RUNSFILE:-}" ] && RUNS=$(cat $RUNSFILE)
umask 077
O=$FL/bench/chain_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O; chmod 700 $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
CRV=$CODE/crv
if [ -n "${BRANCH:-}" ]; then
  CRV=$FL/crv_$BRANCH
  if [ -d $CRV/.git ]; then git -C $CRV fetch -q && git -C $CRV reset -q --hard origin/$BRANCH
  else git clone -q -b $BRANCH https://github.com/RobMina/mu2e-crv-fastsim.git $CRV; fi
fi
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O nproc=$NP maxev=${MAXEV:-all}"
git log -1 --format="umbrella %h"; git -C $CRV log -1 --format="crv %h %s"
export PYTHONPATH=$CRV:$CODE/core LD_LIBRARY_PATH=$FL/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT
L=/tmp/crvfs_chain_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
echo "$(hostname) $L $O" > $FL/bench/kpp_inbox_${PBS_JOBID%%.*}
echo "waiting for $L/ready (up to 30 min)"
for i in $(seq 360); do [ -f $L/ready ] && break; sleep 5; done
[ -f $L/ready ] || { echo "no input arrived"; exit 3; }
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  $V -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f >/dev/null || { echo "fetch $f failed"; exit 4; }
done
export CRVFS_LOOKUP_DIR=$L CRVFS_DATA=$L
ls -la $L | grep -v ready | awk '{print $5, $9}'
GEN=($(ls $L/gen.*.root)); LIB=$(ls $L/lib*.root | head -1); CAL=$(ls $L/crvcalib*.csv | head -1)
DEV=$(ls $L/crvphoton*.csv 2>/dev/null | head -1); GD=$(ls $L/*.gdml | head -1)
echo "gen files ${#GEN[@]}; library $(basename $LIB); calib $(basename $CAL); gdml $(basename $GD)"
cores() { local i=$1; [ $i -lt 6 ] && echo "$((1+8*i))-$((8+8*i))" || echo "$((53+8*(i-6)))-$((60+8*(i-6)))"; }
# RUNS: files per process for each pass (e.g. "1 4"); each pass is a separate node-wide measurement on the same inputs
for FPP in ${RUNS:-1}; do
  R=$O/fpp$FPP; mkdir -p $R
  t0=$(date +%s.%N); pids=()
  for i in $(seq 0 $((NP-1))); do
    fl=(); for k in $(seq 0 $((FPP-1))); do fl+=(${GEN[$(((i*FPP+k) % ${#GEN[@]}))]}); done
    ZE_AFFINITY_MASK=$i taskset -c $(cores $i) $V -m crvfastsim runchain "${fl[@]}" --library $LIB --calib $CAL \
      ${DEV:+--deviation $DEV} --gdml $GD --jax --reco-cpu --seed $((20261021+i)) ${MAXEV:+--max-events $MAXEV} \
      --emit $R/shard_$i.json --outdir $R/p$i > $R/p$i.txt 2>&1 & pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=$((rc+1)); done
  WALL=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')
  echo "== pass files/process=$FPP: processes $NP, failed $rc, wall ${WALL} s"
  for i in $(seq 0 $((NP-1))); do [ -f $R/shard_$i.json ] || { echo "-- p$i failed:"; grep -m3 -E "Error|Fatal|die:" $R/p$i.txt | cut -c1-250; }; done
  $V -m crvfastsim runchain-merge "$R/shard_*.json" --wall $WALL --outdir $R/merge 2>&1 | \
    grep -E '"(shards|generated_events|events_with_steps|wall_s|slowest_shard_wall_s|accepted_events_per_node_hour)"'
done
echo "outdir $O"; echo exit=0
