#!/bin/bash
#PBS -N m2fs_vr1n
#PBS -A GenesisHackathonOct26
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_vr1n.out
# Aurora: isolate the V-R1 noise-on runtime abort with vr1noise_probe.py on synthetic charges (no Mu2e input), on one
# tile. BRANCH (optional) tests a crv branch first on PYTHONPATH, as devfix_aurora.sh.
# usage: qsub [-v M2FS_FLARE=...,EV=events,CASES=one_case,CASEFILE=file_with_a,b,c,BRANCH=...] vr1noise_aurora.sh (qsub -v cannot carry commas)
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv/bin/python; CODE=$FL/code
O=$FL/bench/vr1n_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
CRV=$CODE/crv
if [ -n "${BRANCH:-}" ]; then
  B=$FL/crv_$BRANCH
  if [ -d $B/.git ]; then git -C $B fetch -q && git -C $B reset -q --hard origin/$BRANCH
  else git clone -q -b $BRANCH https://github.com/RobMina/mu2e-crv-fastsim.git $B; fi
  CRV=$B
fi
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git log -1 --format="umbrella %h"; git -C $CRV log -1 --format="crv %h %s"
export PYTHONPATH=$CRV:$CODE/core LD_LIBRARY_PATH=$FL/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
taskset -c 1-51 $V $CODE/core/aurora/vr1noise_probe.py --events ${EV:-25000} --timeout 500 ${CASES:+--cases $CASES} ${CASEFILE:+--cases $(cat $CASEFILE)} --out $O/probe.json
echo "outdir $O"; echo exit=0
