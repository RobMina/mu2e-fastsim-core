#!/bin/bash
# Create or update the project venv on Eagle for Polaris GPU jobs: ALCF conda Python 3.13 + pip JAX (bundled CUDA 12
# libs) + the umbrella's packages (mu2efastsim, crvfastsim, trkfastsim) editable from $M2FS_EAGLE/code.
# The conda module does not load under the IRI job wrapper, so the base interpreter is used directly.
set -e
BASEPY=/soft/applications/conda/2026-10-01/mconda3/bin/python3
VENV=${M2FS_EAGLE:?}/venv
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export PIP_CACHE_DIR=/local/scratch/pipcache
echo "base: $($BASEPY --version)"
[ -x $VENV/bin/python ] || $BASEPY -m venv $VENV
$VENV/bin/python -m pip install -q --upgrade pip
$VENV/bin/python -m pip install -q "jax[cuda12]" numpy h5py pyarrow pandas awkward uproot matplotlib pytest
# the umbrella checkout's packages, editable (git_sync.sh first)
CODE=${M2FS_EAGLE}/code
for pkg in core crv tracker; do
  [ -f "$CODE/$pkg/pyproject.toml" ] && $VENV/bin/python -m pip install -q --no-deps -e "$CODE/$pkg"
done
$VENV/bin/python -m pip freeze | grep -i -E "^(jax|jaxlib|jax-cuda|numpy|h5py|pyarrow|awkward|uproot|-e)" 
$VENV/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
echo exit=0
