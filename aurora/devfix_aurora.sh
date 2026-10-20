#!/bin/bash
#PBS -N m2fs_devfix
#PBS -A GenesisHackathonOct26
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_devfix.out
# Aurora: test a crv branch (default oneapi-device-fix) on PVC without touching the umbrella pins: the branch is cloned
# to $FL/crv_<branch> and put first on PYTHONPATH. Runs the fused device response V4 (v4sipm --device --no-stock) on
# synthetic muon steps with the real v6_0 tables (CVMFS -> tmpfs, deleted), on a tile and on the CPU, plus pytest.
# usage: qsub [-v M2FS_FLARE=...,BRANCH=...,NEV=...] devfix_aurora.sh
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv/bin/python; CODE=$FL/code; BR=${BRANCH:-oneapi-device-fix}; NEV=${NEV:-20000}
B=$FL/crv_$BR
O=$FL/bench/devfix_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
if [ -d $B/.git ]; then git -C $B fetch -q && git -C $B reset -q --hard origin/$BR
else git clone -q -b $BR https://github.com/RobMina/mu2e-crv-fastsim.git $B; fi
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git -C $B log -1 --format="crv branch $BR %h %s"
export PYTHONPATH=$B:$CODE/core LD_LIBRARY_PATH=$FL/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT
$V -c "import crvfastsim, sys; print('crvfastsim from', crvfastsim.__file__)"
L=/tmp/crvfs_devfix_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  $V -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done
export CRVFS_LOOKUP_DIR=$L
$V -m crvfastsim.response.synthsteps --events $NEV --out $L/steps.root || exit 4
echo; echo "==== v4sipm --device on tile 0 | same on CPU (fused kernel) | pytest on tile 1"
ZE_AFFINITY_MASK=0 taskset -c 1-24 $V -X faulthandler -m crvfastsim v4sipm $L/steps.root --device --no-stock --seed 1 ${MAXSTEPS:+--max-steps $MAXSTEPS} --outdir $O/dev_xpu > $O/dev_xpu.txt 2>&1 & a=$!
[ -n "${XPU_ONLY:-}" ] || JAX_PLATFORMS=cpu taskset -c 53-103 $V -m crvfastsim v4sipm $L/steps.root --device --no-stock --seed 1 --outdir $O/dev_cpu > $O/dev_cpu.txt 2>&1 & b=$!
(cd $B && ZE_AFFINITY_MASK=1 taskset -c 25-51 $V -m pytest -q -rfEs -p no:cacheprovider tests/test_device_split.py tests/test_photons.py > $O/pytest.txt 2>&1) & c=$!
wait $c; grep -E "passed|failed|skipped|SKIP|FAILED|Fatal" $O/pytest.txt | tail -6
wait $a ${b:-}
for r in dev_xpu dev_cpu; do [ -f $O/$r.txt ] || continue; echo "-- $r"; grep -v "^\s*$" $O/$r.txt | grep -vE "^\s+@|experimental" | grep -vE "^\s*$" | tail -40 | cut -c1-300; done
echo "outdir $O"; echo exit=0
