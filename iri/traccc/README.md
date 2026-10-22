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

Module versions checked on polaris-login-02 on 2026-10-22: `PrgEnv-gnu` gives g++ 14.3; `spack-pe-base` + `cmake` give
cmake 3.30.5. `cudatoolkit-standalone` 11.8 to 13.0.1 exist; 12.9.1 is used because it accepts gcc 14 (the CI's 12.6
needs gcc <= 13, and only `gcc-native/14` exists). There is no `boost` module, so 2a checks the Spack stack and 2a' builds
Boost.ProgramOptions alone if needed. Order matters: `spack-pe-base` must be loaded before `cmake`.

```bash
# 2a. environment (one module per line, so a failure is visible)
module use /soft/modulefiles
module swap PrgEnv-nvidia PrgEnv-gnu
module load spack-pe-base
module load cmake
module load cudatoolkit-standalone/12.9.1
g++ --version | head -1; nvcc --version | tail -1; cmake --version | head -1     # expect 14.3, 12.9, 3.30.5
export CUDAHOSTCXX=$(which g++) CC=$(which gcc) CXX=$(which g++)
T=/eagle/GenesisHackathonOct26/ram2aq/traccc; mkdir -p $T
BOOST=$(ls -d /soft/spack/pe/*/base/install/*/*/boost-* 2>/dev/null | tail -1); echo "spack boost: ${BOOST:-none}"
ls $BOOST/lib*/libboost_program_options* 2>/dev/null | head -1
```

If that prints a `libboost_program_options` path, set `export BOOST_ROOT=$BOOST` and skip 2a'.

```bash
# 2a'. only if 2a found no Boost with program_options: Boost.ProgramOptions alone into $T/boost (~5 min)
cd $T && curl -sLO https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz && tar xzf boost_1_86_0.tar.gz
cd boost_1_86_0 && ./bootstrap.sh --with-libraries=program_options --prefix=$T/boost && ./b2 -j 8 install
export BOOST_ROOT=$T/boost
```

Every new shell needs 2a (and the `export BOOST_ROOT` line) again before 2c/2d.

```bash
# 2b. source and data (traccc v1.6.0, 2026-07-10)
cd $T
[ -d traccc ] || git clone --branch v1.6.0 --depth 1 https://github.com/acts-project/traccc.git
./traccc/data/traccc_data_get_files.sh

# 2c. configure: float64 CUDA preset, A100 (sm_80), warnings not fatal, examples on
cmake --preset cuda-fp64 -S traccc -B build-cuda-fp64 -DBoost_ROOT=$BOOST_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DTRACCC_FAIL_ON_WARNINGS=FALSE -DTRACCC_BUILD_EXAMPLES=TRUE 2>&1 | tail -20

# 2d. build
cmake --build build-cuda-fp64 -j 8 2>&1 | tee $T/build.log | tail -30
ls build-cuda-fp64/bin | grep -E 'simulate_wire_chamber|truth_fitting_example'
```

Expect 1-2 h for 2d at `-j 8` (detray, vecmem, ACTS headers, CUDA kernels). If the session drops, reattach (`screen -r`)
or redo 2a and rerun 2d: it resumes.

Known traps:

- traccc needs C++20 and CUDA >= 12 (its CI: gcc 13.3 + CUDA 12.6). If nvcc 12.9 still rejects gcc 14, add
  `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` in 2c (fresh build dir).
- The presets set `TRACCC_FAIL_ON_WARNINGS=TRUE`; 2c turns it off, because a newer compiler warns where CI does not.
- Login nodes reach GitHub directly (the ALCF proxy is needed only on compute nodes). CMake's FetchContent pulls
  vecmem, detray, ACTS and googletest during 2c.
- If CMake does not find Boost, 2c fails at `program_options`: check `echo $BOOST_ROOT` and that 2a' installed
  `$T/boost/lib/libboost_program_options.so`.

## 3. Run (needs one GPU: a short batch job, submitted from the login node)

```bash
qsub -A GenesisHackathonOct26 -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle \
  -o ~/traccc_wire.log -j oe -- /bin/bash -lc \
  'module use /soft/modulefiles; module swap PrgEnv-nvidia PrgEnv-gnu; module load cudatoolkit-standalone/12.9.1; \
   export LD_LIBRARY_PATH=/eagle/GenesisHackathonOct26/ram2aq/traccc/boost/lib:$LD_LIBRARY_PATH; \
   bash /eagle/GenesisHackathonOct26/ram2aq/crvfastsim/code/core/iri/traccc/run_wire_chamber.sh'
```

Same compiler and CUDA as the build (2a); the Boost path matters only if 2a' built it. If `debug` sits in the queue, `-q preemptable` started within a minute on
2026-10-19. The script simulates the detray wire chamber (2 GeV/c, 100 particles x 10 events; then 100 GeV/c, 2000
particles x 10 events) and runs the truth fitting example on CPU and on CUDA with `--check-performance`, writing to
`/eagle/GenesisHackathonOct26/ram2aq/traccc/runs/<date>/`. Please send me `~/traccc_wire.log` and the tail of
`$T/build.log`.
