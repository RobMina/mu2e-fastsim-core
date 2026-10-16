#!/bin/bash
# Probe the conda module failure under IRI: print the modulefile, check its dependencies, load verbosely.
exec bash -l <<'INNER'
module use /soft/modulefiles
cat /soft/modulefiles/conda/2026-10-01.lua | grep -v "^\s*--" | head -60
echo "---- dependency availability"
module -t avail cray-hdf5-parallel gcc-native PrgEnv-gnu 2>&1 | head
echo "---- verbose load"
module load PrgEnv-gnu gcc-native/14 2>&1 | tail -2
module load cray-hdf5-parallel/1.14.3.9 2>&1 | tail -3
module -D load conda/2026-10-01 2>&1 | grep -i -E "error|fail|cannot|unable" | head -8
module load conda/2026-10-01 2>&1 | tail -3
echo "loaded: $(module -t list 2>&1 | tr '\n' ' ')"
echo "python=$(which python3)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tail -1
echo exit=0
INNER
