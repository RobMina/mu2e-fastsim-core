#!/bin/bash
# traccc Kalman tests on a Polaris compute node (no build: needs the build of README.md section 2).
# The upstream wire-chamber Kalman fit test (CPU; pulls of d0, z0, phi, theta, qop checked by the test itself) and the
# CUDA Kalman test (telescope geometry; there is no CUDA wire-chamber test in v1.6.0). The examples (simulate +
# truth_fitting_example) were dropped: in v1.6.0 the fitting example throws map::at on the simulated wire chamber.
T=${TRACCC_AREA:-/eagle/GenesisHackathonOct26/ram2aq/traccc}
B=$T/build-cuda-fp64/bin
R=$T/runs/$(date +%Y%m%d_%H%M%S); mkdir -p $R; cd $R || exit 1
export TRACCC_TEST_DATA_DIR=${TRACCC_TEST_DATA_DIR:-$T/traccc/data}
ulimit -c 0
echo "host $(hostname) job ${PBS_JOBID:-none} out $R"; nvidia-smi -L
[ -x $B/traccc_test_cpu ] || { echo "no build at $B"; exit 2; }

run() { local log="$R/$1.log"; shift; echo "== $*"; local t0=$SECONDS; "$@" > "$log" 2>&1; local rc=$?; echo "exit $rc ($((SECONDS - t0)) s)"; }   # no /usr/bin/time on Polaris compute nodes

run cpu_wire $B/traccc_test_cpu --gtest_filter='*KalmanFittingWireChamber*'
run cuda_kalman $B/traccc_test_cuda --gtest_filter='*Kalman*'
for f in cpu_wire cuda_kalman; do echo "--- $f"; grep -E '^\[ *(RUN|  *OK|  FAILED|  PASSED|==========) ' $R/$f.log | tail -20; done
ls -la $R
