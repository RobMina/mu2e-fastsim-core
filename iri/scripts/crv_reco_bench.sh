#!/bin/bash
# GPU throughput of the CRV fast pulse reco (crvfastsim reco-bench) on one Polaris node: GPU tests, then one
# process per A100 on synthetic KPP-like waveforms. usage: crv_reco_bench.sh [n_events_per_gpu] [rounds]
# (defaults 200000, kernel default rounds). Pulls the umbrella checkout first.
E=${M2FS_EAGLE:?}; V=$E/venv/bin/python; N=${1:-200000}; R=${2:-}
cd $E/code || exit 2
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
git pull -q --ff-only && git submodule update -q --init --recursive; git log -1 --format="code %h %s"; git submodule status
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
for g in 0 1 2 3; do grep -h -E '"(stage|events_per_s|compile_s|us_per_event)"' $O/wf$g/bench_waveforms_stats.json | tr -d '\n'; echo; done
for g in 0 1 2 3; do grep -h -E '"(device|events_per_s|compile_s|us_per_event|rounds|rounds_exhausted_first_batch)"' $O/gpu$g/bench_stats.json | tr -d '\n'; echo; done
echo "outdir $O"; echo exit=0
