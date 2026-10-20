#!/bin/bash
# V4 of the fused device response (photons -> SiPM charges on the GPU) against the numpy ports, on one Polaris node.
# Steps: muon-like synthetic CrvSteps written on the node (crvfastsim.response.synthsteps, no Mu2e input); lookup
# tables and photon map fetched from CVMFS to node-local scratch and deleted on exit. Warm timed passes (--repeat) give
# the device throughput; the job fails if either seed fails.
E=${M2FS_EAGLE:?}; V=$E/venv/bin/python
cd $E/code || exit 2
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
git pull -q --ff-only && git submodule update -q --init --recursive; git log -1 --format="code %h %s"; git -C crv log -1 --format="crv %h"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
O=$E/bench/devv4_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
L=/local/scratch/crvfs_lookup_$$; mkdir -p $L; chmod 700 $L; trap 'rm -rf $L; echo "deleted $L"' EXIT
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  $V -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || exit 3; done
export MU2E_SEARCH_PATH=$E/offline_data   # crv_counters_extracted_v04.txt (public, fetched by the reco bench job)
export CRVFS_LOOKUP_DIR=$L
$V -m crvfastsim.response.synthsteps --events ${NEV:-20000} --out $L/steps.root || exit 4
pids=()
for sd in 1 2; do
  CUDA_VISIBLE_DEVICES=$((sd-1)) $V -m crvfastsim v4sipm $L/steps.root --device --no-stock --seed $sd --repeat ${REPEAT:-2} \
    --outdir $O/v4_seed$sd > $O/v4_seed$sd.txt 2>&1 & pids+=($!)
done
rc=0; for i in 0 1; do wait ${pids[$i]} || { echo "seed $((i+1)) failed"; rc=5; }; done
for sd in 1 2; do echo "== seed $sd"; tail -9 $O/v4_seed$sd.txt; done
echo "outdir $O"; echo exit=$rc; exit $rc
