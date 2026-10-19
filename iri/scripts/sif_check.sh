#!/bin/bash
# R4 check on a Polaris compute node: find apptainer, time a cold start of the mu2e-fastsim image (first exec with an
# empty cache), run the GPU tests and the chain bench inside it. usage: sif_check.sh <image.sif>
S=${1:?usage: sif_check.sh <image.sif>}
E=${M2FS_EAGLE:?}; O=$E/bench/sif_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
A=$(command -v apptainer || true)
[ -z "$A" ] && for m in apptainer spack-pe-base/0.10.1; do module load $m 2>/dev/null; done; A=$(command -v apptainer || command -v singularity || true)
echo "apptainer: ${A:-none} $($A --version 2>/dev/null)"; [ -n "$A" ] || { echo exit=3; exit 3; }
ls -la $S
export APPTAINER_CACHEDIR=/local/scratch/$USER/apptainer_cache; mkdir -p $APPTAINER_CACHEDIR
L=/local/scratch/$USER/sif; mkdir -p $L
t0=$(date +%s.%N); cp $S $L/img.sif; t1=$(date +%s.%N)
$A exec --nv --cleanenv $L/img.sif python -c "import jax, crvfastsim.reco.chain_jax; print('devices', jax.devices())"; t2=$(date +%s.%N)
echo "copy_to_local_s $(echo "$t1-$t0" | bc) cold_first_exec_s $(echo "$t2-$t1" | bc) cold_start_total_s $(echo "$t2-$t0" | bc)"
$A exec --nv --cleanenv $L/img.sif cat /opt/mu2e-fastsim/VERSION
$A exec --nv --cleanenv $L/img.sif python -m pytest -q -p no:cacheprovider /opt/mu2e-fastsim/crv/tests/test_reco_jax.py 2>&1 | tail -2
CUDA_VISIBLE_DEVICES=0 $A exec --nv --cleanenv -B $O $L/img.sif crvfastsim reco-bench synthetic --stage chain \
  --max-events 200000 --repeat 3 --length 12 --batch 65536 --chunks 8 --rounds 3 --outdir $O/chain > $O/chain.txt 2>&1
grep -h -E '"(device|events|clusters|events_per_s|us_per_event)"' $O/chain/bench_chain_stats.json | tr -d '\n'; echo
rm -rf $L
echo "outdir $O"; echo exit=0
