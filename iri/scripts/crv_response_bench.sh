#!/bin/bash
# CRV fast response on one Polaris node with the REAL photon lookup table and SiPM photon map: fetched from the Mu2e
# CVMFS repository over HTTP (Stratum-1 via the site proxy) into node-local scratch, used, and deleted (Mu2e-internal,
# never written to Eagle). Photon kernel and SiPM-charge stage, one process per A100.
# usage: crv_response_bench.sh [rows_per_gpu] [sipms_per_gpu]
E=${M2FS_EAGLE:?}; V=$E/venv/bin/python; N=${1:-1000000}; NS=${2:-500000}
cd $E/code || exit 2
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
git pull -q --ff-only && git submodule update -q --init --recursive; git log -1 --format="code %h %s"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
O=$E/bench/response_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
L=/local/scratch/crvfs_lookup_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
for f in LookupTable_3200_0 photonMap.root; do
  $V -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done
ls -la $L
export CRVFS_LOOKUP_DIR=$L
CRVFS_LOOKUP_DIR=$L $V -m pytest -q -p no:cacheprovider crv/tests/test_photons.py 2>&1 | tail -2
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim.response.bench --table 3200_0 --rows $N --numpy-rows 0 --repeat 3 \
    --seed $((20261020+g)) --outdir $O/ph$g > $O/ph$g.txt 2>&1 &
done; wait
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim.response.bench --stage sipm --photon-map $L/photonMap.root --sipms $NS \
    --numpy-sipms 0 --repeat 3 --seed $((20261120+g)) --outdir $O/si$g > $O/si$g.txt 2>&1 &
done; wait
CUDA_VISIBLE_DEVICES=0 $V -m crvfastsim.response.bench --table 3200_0 --rows 20000 --numpy-rows 20000 --no-jax > $O/ph_cpu.txt 2>&1
CUDA_VISIBLE_DEVICES=0 $V -m crvfastsim.response.bench --stage sipm --photon-map $L/photonMap.root --sipms 20000 --numpy-sipms 20000 --no-jax > $O/si_cpu.txt 2>&1
for f in ph0 ph1 ph2 ph3 si0 si1 si2 si3 ph_cpu si_cpu; do echo "$f $(tail -1 $O/$f.txt)"; done
echo "outdir $O"; echo exit=0
