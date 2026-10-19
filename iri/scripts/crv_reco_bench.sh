#!/bin/bash
# GPU throughput of the CRV fast pulse reco (crvfastsim reco-bench) on one Polaris node: GPU tests, then one
# process per A100 on synthetic KPP-like waveforms. usage: crv_reco_bench.sh [n_events_per_gpu] [rounds]
# (defaults 200000, kernel default rounds); then the coincidence finder on NC (100000) and the pipelined pulses -> coincidences chain on NCH (200000) synthetic events per GPU. Pulls the umbrella checkout first.
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
# R5: GPU twice (bit-identical per seed), CPU on the same node, then GPU vs CPU (statistical)
CUDA_VISIBLE_DEVICES=0 $V -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $O/repro_gpu > $O/repro_gpu.txt 2>&1
JAX_PLATFORMS=cpu $V -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $O/repro_cpu \
  --against $O/repro_gpu/repro_gpu.npz > $O/repro_cpu.txt 2>&1
for r in gpu cpu; do grep -h -E '"(backend|pass_bit_identical|pass_statistical|sample_identical_frac|identical_frac|max_rel_pe_diff|max_abs_time_diff_ns|clusters|per_event_type_counts_equal|p)"' \
  $O/repro_$r/repro_stats.json | tr -d '\n'; echo; done
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
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim reco-bench synthetic --stage chain --max-events ${NCH:-200000} --repeat 3 \
    --length 12 --batch 65536 --chunks ${CHUNKS:-8} --rounds ${R:-3} --seed $((20261316+g)) --outdir $O/ch$g > $O/ch$g.txt 2>&1 &
done; wait
for g in 0 1 2 3; do grep -h -E '"(device|events|waveforms|clusters|coinc_fallback_events|events_per_s|us_per_event|pulse_dispatch_s|pulse_collect_s|coinc_async_s|coinc_finish_s|wall_s)"' \
  $O/ch$g/bench_chain_stats.json | tr -d '\n'; echo; done
for g in 0 1 2 3; do grep -h -E '"(stage|events_per_s|compile_s|us_per_event)"' $O/wf$g/bench_waveforms_stats.json | tr -d '\n'; echo; done
for g in 0 1 2 3; do grep -h -E '"(device|events_per_s|compile_s|us_per_event|rounds|rounds_exhausted_first_batch)"' $O/gpu$g/bench_stats.json | tr -d '\n'; echo; done
echo "outdir $O"; echo exit=0
