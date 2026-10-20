#!/bin/bash
# R4 check on a Polaris compute node: find apptainer, build mu2e-fastsim images from the Eagle checkout (the IRI
# upload endpoint is not implemented), and for each: time a cold start (copy to node-local disk + first exec), run the
# GPU tests and the pulses -> coincidences chain bench inside it.
# usage: [REBUILD=1] [VARIANTS="full trim"] sif_check.sh. Variants: full (all CUDA wheels), trim (build.sh default:
# without cuDNN/NCCL/NVSHMEM/cuFFT), cpu. Images: $M2FS_EAGLE/mu2efastsim_<variant>.sif.
E=${M2FS_EAGLE:?}; O=$E/bench/sif_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
# apptainer on compute nodes: lmod, then the spack base 0.8.1 module tree (apptainer 1.3.2; probed 2026-10-19, PBS 7762807)
source /opt/cray/pe/lmod/lmod/init/bash; export MODULEPATH=/soft/modulefiles
module use /soft/spack/base/0.8.1/install/modulefiles/Core; module load apptainer 2>/dev/null
A=$(command -v apptainer || true)
echo "apptainer: ${A:-none} $($A --version 2>/dev/null)"; [ -n "$A" ] || { echo exit=3; exit 3; }
cd $E/code && git pull -q --ff-only && git submodule update -q --init --recursive && git log -1 --format="code %h"
export APPTAINER_TMPDIR=/local/scratch/$USER/apptainer_tmp APPTAINER_CACHEDIR=/local/scratch/$USER/apptainer_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
for V in ${VARIANTS:-full trim}; do
  S=$E/mu2efastsim_$V.sif
  echo "=== variant $V"
  [ "${REBUILD:-0}" = 1 ] && rm -f $S
  if [ ! -s $S ]; then
    case $V in full) env="VARIANT=gpu TRIM=" ;; trim) env="VARIANT=gpu" ;; cpu) env="VARIANT=cpu" ;; esac
    tb=$(date +%s); env $env bash container/build.sh /local/scratch/$USER/build.sif > $O/build_$V.txt 2>&1; rc=$?
    te=$(date +%s); echo "build_exit $rc build_s $((te-tb))"; [ $rc -eq 0 ] || { tail -5 $O/build_$V.txt; continue; }
    cp /local/scratch/$USER/build.sif $S && rm -f /local/scratch/$USER/build.sif
  fi
  echo "image_bytes $(stat -c %s $S)"
  L=/local/scratch/$USER/sif; rm -rf $L $APPTAINER_CACHEDIR/*; mkdir -p $L
  NV=--nv; [ $V = cpu ] && NV=
  t0=$(date +%s.%N); cp $S $L/img.sif; t1=$(date +%s.%N)
  $A exec $NV --cleanenv $L/img.sif python -c "import jax, crvfastsim.cli, crvfastsim.reco.chain_jax; print('devices', jax.devices())" 2>&1 | grep -v "^INFO" | tail -3
  if [ $V != cpu ] && ! $A exec --nv --cleanenv $L/img.sif python -c "import jax, sys; sys.exit(jax.default_backend() != 'gpu')" 2>/dev/null; then
    echo "variant $V: JAX did not start on the GPU (falls back to CPU): not a usable GPU image"; rm -rf $L; continue; fi
  t2=$(date +%s.%N)
  echo "copy_to_local_s $(echo "$t1-$t0" | bc) cold_first_exec_s $(echo "$t2-$t1" | bc) cold_start_total_s $(echo "$t2-$t0" | bc)"
  $A exec $NV --cleanenv $L/img.sif tail -1 /opt/mu2e-fastsim/VERSION
  $A exec $NV --cleanenv $L/img.sif python -m pytest -q -p no:cacheprovider /opt/mu2e-fastsim/crv/tests/test_reco_jax.py 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES=0 $A exec $NV --cleanenv -B $O $L/img.sif crvfastsim reco-bench synthetic --stage chain \
    --max-events ${NCH:-200000} --repeat 3 --length 12 --batch 65536 --chunks 8 --rounds 3 --outdir $O/chain_$V > $O/chain_$V.txt 2>&1
  grep -h -E '"(backend|device|events|clusters|events_per_s|us_per_event)"' $O/chain_$V/bench_chain_stats.json 2>/dev/null | tr -d '\n'; echo
  [ -s $O/chain_$V/bench_chain_stats.json ] || grep -v "^INFO" $O/chain_$V.txt | tail -5
  rm -rf $L
done
echo "outdir $O"; echo exit=0
