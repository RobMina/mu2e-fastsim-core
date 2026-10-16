#!/bin/bash
# GPU throughput of the CRV fast pulse reco (crvfastsim reco-bench) on one Polaris node: GPU tests, then one
# process per A100 on synthetic KPP-like waveforms. usage: crv_reco_bench.sh [n_events_per_gpu] [rounds]
# (defaults 200000, kernel default rounds); then the coincidence finder on NC (100000) synthetic events per GPU. Pulls the umbrella checkout first.
E=${M2FS_EAGLE:?}; V=$E/venv/bin/python; N=${1:-200000}; R=${2:-}
cd $E/code || exit 2
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
git pull -q --ff-only && git submodule update -q --init --recursive; git log -1 --format="code %h %s"; git submodule status
# public Offline data files the fast chain reads (no CVMFS on Polaris): fetched once into $E/offline_data
D=$E/offline_data/Offline/CRVResponse/data; mkdir -p $D
[ -s $D/singlePEWaveform_v3.txt ] || curl -s -o $D/singlePEWaveform_v3.txt \
  https://raw.githubusercontent.com/Mu2e/Offline/main/CRVResponse/data/singlePEWaveform_v3.txt
G=$E/offline_data/Offline/Mu2eG4/geom; mkdir -p $G
[ -s $G/crv_counters_extracted_v04.txt ] || curl -s -o $G/crv_counters_extracted_v04.txt \
  https://raw.githubusercontent.com/Mu2e/Offline/main/Mu2eG4/geom/crv_counters_extracted_v04.txt
export MU2E_SEARCH_PATH=$E/offline_data${MU2E_SEARCH_PATH:+:$MU2E_SEARCH_PATH}
wc -l $D/singlePEWaveform_v3.txt
export XLA_PYTHON_CLIENT_PREALLOCATE=false
O=$E/bench/$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
$V -c "import jax; print('devices', jax.devices())"
$V -m pytest -q -p no:cacheprovider crv/tests/test_reco_jax.py core/tests 2>&1 | tail -3
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim reco-bench synthetic --max-events $N --length 12 --batch 65536 --repeat 3 \
    --seed $((20261016+g)) ${R:+--rounds $R} --outdir $O/gpu$g > $O/gpu$g.txt 2>&1 &
done; wait
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim reco-bench synthetic --stage waveforms --max-events 200000 --batch 65536 \
    --repeat 3 --seed $((20261116+g)) --outdir $O/wf$g > $O/wf$g.txt 2>&1 &
done; wait
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim reco-bench synthetic --stage coinc --max-events ${NC:-100000} --repeat 3 \
    --seed $((20261216+g)) --outdir $O/co$g > $O/co$g.txt 2>&1 &
done; wait
for g in 0 1 2 3; do grep -h -E '"(device|events|clusters|fallback_events|first_call_s|kernel_s|pack_s|fallback_s|events_per_s)"' \
  $O/co$g/bench_coinc_stats.json | tr -d '\n'; echo; done
for g in 0 1 2 3; do grep -h -E '"(stage|events_per_s|compile_s|us_per_event)"' $O/wf$g/bench_waveforms_stats.json | tr -d '\n'; echo; done
for g in 0 1 2 3; do grep -h -E '"(device|events_per_s|compile_s|us_per_event|rounds|rounds_exhausted_first_batch)"' $O/gpu$g/bench_stats.json | tr -d '\n'; echo; done
echo "outdir $O"; echo exit=0
