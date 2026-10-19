#!/bin/bash
# GPU throughput of the CRV photon substep kernel (crvfastsim.response.bench) on one Polaris node: tests, then one process
# per A100 on synthetic tables and rows (no Mu2e-internal input). usage: crv_photon_bench.sh [rows_per_gpu] [prob]
E=${M2FS_EAGLE:?}; V=$E/venv/bin/python; N=${1:-1000000}; P=${2:-0.0015}
cd $E/code || exit 2
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
git pull -q --ff-only && git submodule update -q --init --recursive; git log -1 --format="code %h %s"; git submodule status
export XLA_PYTHON_CLIENT_PREALLOCATE=false
O=$E/bench/photons_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
$V -c "import jax; print('devices', jax.devices())"
$V -m pytest -q -p no:cacheprovider crv/tests/test_photons.py 2>&1 | tail -3
CUDA_VISIBLE_DEVICES=0 $V -m crvfastsim.response.bench --rows 20000 --numpy-rows 20000 --no-jax --outdir $O/cpu > $O/cpu.txt 2>&1
cat $O/cpu.txt
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim.response.bench --rows $N --prob $P --numpy-rows 0 --repeat 3 --seed $((20261019+g)) \
    --outdir $O/gpu$g > $O/gpu$g.txt 2>&1 &
done; wait
for g in 0 1 2 3; do tail -1 $O/gpu$g.txt; done
echo "outdir $O"; echo exit=0
