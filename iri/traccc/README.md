# traccc on Polaris: build and the wire-chamber Kalman test

Purpose (roadmap decision R9): ACTS traccc is the reference and benchmark for the tracker fast reco's own JAX
Kalman core. This first test answers four questions on its drift-chamber example:

1. Does it fit wire (line-surface) measurements as a drift **time** with a fitted t0, or only a signed drift radius?
2. How does it resolve the left/right ambiguity?
3. Does the `cuda-fp64` preset build and run (we need float64)?
4. Fit throughput on one A100 (tracks/s), CPU vs CUDA, for the benchmark table.

The user runs the build (rule G6). Nothing here touches Mu2e-internal data.

## 1. Login node (user)

```bash
ssh polaris.alcf.anl.gov          # a polaris-login-0N node; build there, no compute allocation needed
```

The build compiles for the A100 (`sm_80`) without a GPU present, so a login node is enough. Login nodes are shared:
keep `-j` at 8 and run the build inside `screen` or `tmux` so a dropped ssh does not kill it. Only the test run in
section 3 needs a GPU.

## 2. Build (user, on the login node; copy-paste, there is no build script by rule G6)

Module names below are from the ALCF docs and `core/iri/scripts/probe_env.sh` (`gcc-native/14` exists); I have not
checked `cudatoolkit-standalone`, `cmake` or Boost on today's Polaris, so step 2a prints what is there first.

```bash
# 2a. environment (check the printed versions: g++ >= 12 for C++20, nvcc >= 12, cmake >= 3.25, Boost present)
module use /soft/modulefiles
module load PrgEnv-gnu gcc-native/14 cudatoolkit-standalone spack-pe-base cmake boost 2>&1 | tail -5
module -t list 2>&1 | tr '\n' ' '; echo
g++ --version | head -1; nvcc --version | tail -2; cmake --version | head -1
ls -d ${BOOST_ROOT:-/nonexistent} 2>/dev/null || echo "BOOST_ROOT unset: look with 'module avail boost'"
export CUDAHOSTCXX=$(which g++)

# 2b. source and data (traccc v1.6.0, 2026-07-10)
T=/eagle/GenesisHackathonOct26/ram2aq/traccc; mkdir -p $T && cd $T
[ -d traccc ] || git clone --branch v1.6.0 --depth 1 https://github.com/acts-project/traccc.git
./traccc/data/traccc_data_get_files.sh

# 2c. configure: float64 CUDA preset, A100 (sm_80), warnings not fatal, examples on
cmake --preset cuda-fp64 -S traccc -B build-cuda-fp64 \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DTRACCC_FAIL_ON_WARNINGS=FALSE -DTRACCC_BUILD_EXAMPLES=TRUE 2>&1 | tail -20

# 2d. build
cmake --build build-cuda-fp64 -j 8 2>&1 | tee $T/build.log | tail -30
ls build-cuda-fp64/bin | grep -E 'simulate_wire_chamber|truth_fitting_example'
```

Expect 1-2 h for 2d at `-j 8` (detray, vecmem, ACTS headers, CUDA kernels). If the session drops, reattach (`screen -r`)
or redo 2a and rerun 2d: it resumes.

Known traps:

- traccc needs C++20 and CUDA >= 12 (its CI: gcc 13.3 + CUDA 12.6). If nvcc rejects gcc 14, try
  `module swap gcc-native/14 gcc-native/13` (or 12) and set `CUDAHOSTCXX` again before 2c, in a fresh build dir.
- The presets set `TRACCC_FAIL_ON_WARNINGS=TRUE`; 2c turns it off, because a newer compiler warns where CI does not.
- Login nodes reach GitHub directly (the ALCF proxy is needed only on compute nodes). CMake's FetchContent pulls
  vecmem, detray, ACTS and googletest during 2c.
- If Boost is not found, 2c fails at `program_options`: tell me what `module avail boost` lists.

## 3. Run (needs one GPU: a short batch job, submitted from the login node)

```bash
qsub -A GenesisHackathonOct26 -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle \
  -o ~/traccc_wire.log -j oe -- /bin/bash -lc \
  'module use /soft/modulefiles; module load PrgEnv-gnu gcc-native/14 cudatoolkit-standalone; \
   bash /eagle/GenesisHackathonOct26/ram2aq/crvfastsim/code/core/iri/traccc/run_wire_chamber.sh'
```

Use the same modules as the build (2a). If `debug` sits in the queue, `-q preemptable` started within a minute on
2026-10-19. The script simulates the detray wire chamber (2 GeV/c, 100 particles x 10 events; then 100 GeV/c, 2000
particles x 10 events) and runs the truth fitting example on CPU and on CUDA with `--check-performance`, writing to
`/eagle/GenesisHackathonOct26/ram2aq/traccc/runs/<date>/`. Please send me `~/traccc_wire.log` and the tail of
`$T/build.log`.
