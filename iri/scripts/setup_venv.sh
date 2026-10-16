#!/bin/bash
# Create the project venv on Eagle for Polaris GPU jobs: ALCF conda Python 3.13 + pip JAX (bundled CUDA 12 libs).
# The conda module does not load under the IRI job wrapper, so the base interpreter is used directly.
set -e
BASEPY=/soft/applications/conda/2026-10-01/mconda3/bin/python3
VENV=${M2FS_EAGLE:?}/venv
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export PIP_CACHE_DIR=/local/scratch/pipcache
echo "base: $($BASEPY --version)"
[ -x $VENV/bin/python ] || $BASEPY -m venv $VENV
$VENV/bin/python -m pip install -q --upgrade pip
$VENV/bin/python -m pip install -q "jax[cuda12]" numpy h5py pyarrow
$VENV/bin/python -m pip freeze | grep -i -E "^(jax|jaxlib|jax-cuda|numpy|h5py|pyarrow)" 
$VENV/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
echo exit=0
