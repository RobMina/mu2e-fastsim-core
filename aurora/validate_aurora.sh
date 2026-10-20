#!/bin/bash
#PBS -N m2fs_valid
#PBS -A GenesisHackathonOct26
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_valid.out
# Aurora physics validation of the JAX kernels on Intel PVC against the numpy ports (no Offline on Aurora):
#  V4 photons (JAX twin vs numpy port), V4 SiPM (JAX twin and fused device response vs numpy port), on synthetic muon
#  CrvSteps (response.synthsteps) with the REAL v6_0 tables (CVMFS -> tmpfs /tmp, deleted on exit);
#  reco pytest (JAX vs numpy pulses, waveforms, coincidences incl. the oneAPI tier skip);
#  R5/P6 reproducibility: reco-repro twice on PVC (bit-identical per seed) and CPU vs PVC (statistical).
# usage: qsub [-v M2FS_FLARE=...,NEV=events,ONLY=reco] validate_aurora.sh   (ONLY=reco: skip the V4 response arms)
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv/bin/python; CODE=$FL/code; NEV=${NEV:-20000}
O=$FL/bench/valid_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git log -1 --format="umbrella %h %s"; git submodule status
export LD_LIBRARY_PATH=$FL/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT
L=/tmp/crvfs_valid_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  $V -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done
export CRVFS_LOOKUP_DIR=$L
$V -m crvfastsim.response.synthsteps --events $NEV --out $L/steps.root || exit 4
sec() { echo; echo "==== $*"; }
# tiles: 0 photons, 1 sipm jax, 2 sipm device, 3-4 repro; CPU arms pinned away from them
p1=; p2=; p3=
if [ "${ONLY:-}" != reco ]; then
sec "V4 photons, SiPM (jax), SiPM (device) in parallel on tiles 0-2"
ZE_AFFINITY_MASK=0 taskset -c 1-16 $V -m crvfastsim v4photons $L/steps.root --jax --no-stock --seed 1 --outdir $O/v4ph > $O/v4ph.txt 2>&1 & p1=$!
ZE_AFFINITY_MASK=1 taskset -c 17-32 $V -m crvfastsim v4sipm $L/steps.root --jax --no-stock --seed 1 --outdir $O/v4si_jax > $O/v4si_jax.txt 2>&1 & p2=$!
ZE_AFFINITY_MASK=2 taskset -c 33-48 $V -m crvfastsim v4sipm $L/steps.root --device --no-stock --seed 1 --outdir $O/v4si_dev > $O/v4si_dev.txt 2>&1 & p3=$!
fi
sec "R5/P6 repro: PVC twice per seed (tile 3), then CPU against PVC"
ZE_AFFINITY_MASK=3 taskset -c 53-68 $V -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $O/repro_xpu > $O/repro_xpu.txt 2>&1 & p4=$!
sec "reco pytest on tile 4"
ZE_AFFINITY_MASK=4 taskset -c 69-84 $V -m pytest -q -rfE -p no:cacheprovider crv/tests/test_reco_jax.py crv/tests/test_photons.py core/tests > $O/pytest.txt 2>&1 & p5=$!
wait $p5; grep -E "passed|failed|FAILED|ERROR|Fatal" $O/pytest.txt | tail -15
wait $p4; RX=$(ls $O/repro_xpu/repro_*.npz 2>/dev/null | head -1); echo "xpu npz: $RX"
JAX_PLATFORMS=cpu taskset -c 85-103 $V -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $O/repro_cpu \
  ${RX:+--against $RX} > $O/repro_cpu.txt 2>&1
for r in xpu cpu; do echo "-- repro_$r"; grep -h -E '"(backend|pass_bit_identical|same_seed_identical|identical|sample_identical_frac|identical_frac|max_rel_pe_diff|max_abs_time_diff_ns|max_abs_adc_diff|clusters|per_event_type_counts_equal|d|p)"' \
  $O/repro_$r/repro_stats.json | tr -s ' ' | tr -d '\n' | cut -c1-1500; echo; done
if [ -n "$p1" ]; then wait $p1 $p2 $p3
  for f in v4ph v4si_jax v4si_dev; do echo "-- $f"; tail -8 $O/$f.txt | cut -c1-300; done; fi
echo "outdir $O"; echo exit=0
