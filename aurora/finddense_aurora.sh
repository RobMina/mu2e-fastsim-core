#!/bin/bash
#PBS -N m2fs_fdense
#PBS -A GenesisHackathonOct26
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_fdense.out
# Aurora: coinc_jax.find_dense compile/run per tier and under several XLA_FLAGS on one PVC tile (finddense_probe.py).
# Synthetic pulses only. usage: qsub [-v M2FS_FLARE=...,EV=events] finddense_aurora.sh
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv; CODE=$FL/code
O=$FL/bench/finddense_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git log -1 --format="umbrella %h %s"; git submodule status
export LD_LIBRARY_PATH=$V/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$FL/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
$V/bin/python core/aurora/finddense_probe.py --events ${EV:-20000} --timeout 400 --out $O/finddense.json
echo "outdir $O"; echo exit=0
