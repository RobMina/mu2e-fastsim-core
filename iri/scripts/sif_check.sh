#!/bin/bash
# R4 check on a Polaris compute node: find apptainer, build the mu2e-fastsim image from the Eagle checkout if it is
# missing (the IRI upload endpoint is not implemented), time a cold start (copy to node-local disk + first exec with
# an empty cache), run the GPU tests and the chain bench inside it. usage: sif_check.sh [image.sif]
E=${M2FS_EAGLE:?}; S=${1:-$E/mu2efastsim.sif}; O=$E/bench/sif_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
A=$(command -v apptainer || true)
[ -z "$A" ] && for m in apptainer spack-pe-base/0.10.1; do module load $m 2>/dev/null; done; A=$(command -v apptainer || command -v singularity || true)
echo "apptainer: ${A:-none} $($A --version 2>/dev/null)"; [ -n "$A" ] || { echo exit=3; exit 3; }
if [ ! -s $S ]; then
  cd $E/code && git pull -q --ff-only && git submodule update -q --init --recursive && git log -1 --format="code %h"
  export APPTAINER_TMPDIR=/local/scratch/$USER/apptainer_tmp
  tb=$(date +%s); bash container/build.sh /local/scratch/$USER/build.sif > $O/build.txt 2>&1; rc=$?; te=$(date +%s)
  tail -5 $O/build.txt; echo "build_exit $rc build_s $((te-tb))"; [ $rc -eq 0 ] || { echo exit=4; exit 4; }
  cp /local/scratch/$USER/build.sif $S && rm -f /local/scratch/$USER/build.sif
fi
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
