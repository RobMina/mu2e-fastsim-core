#!/bin/bash
# traccc wire-chamber Kalman test on a Polaris compute node (no build: needs the build of README.md section 2).
# Simulates the detray wire chamber, then runs the truth fitting example on CPU and CUDA with performance output.
T=${TRACCC_AREA:-/eagle/GenesisHackathonOct26/ram2aq/traccc}
B=$T/build-cuda-fp64/bin
R=$T/runs/$(date +%Y%m%d_%H%M%S); mkdir -p $R; cd $R || exit 1
export TRACCC_TEST_DATA_DIR=${TRACCC_TEST_DATA_DIR:-$T/traccc/data}
echo "host $(hostname) job ${PBS_JOBID:-none} out $R"; nvidia-smi -L
[ -x $B/traccc_simulate_wire_chamber ] || { echo "no build at $B"; exit 2; }

run() { local log="$R/$1.log"; shift; echo "== $*"; local t0=$SECONDS; "$@" > "$log" 2>&1; local rc=$?; echo "exit $rc ($((SECONDS - t0)) s)"; }   # no /usr/bin/time on Polaris compute nodes

for cfg in "p2:2:100" "p100:100:2000"; do
  IFS=: read tag p n <<< "$cfg"
  sim=detray_simulation/wire_chamber/${tag}
  ( cd $B && run sim_$tag ./traccc_simulate_wire_chamber --gen-vertex-xyz-mm=0:0:0 --gen-vertex-xyz-std-mm=0:0:0 \
      --gen-mom-gev=$p:$p --gen-phi-degree=0:360 --gen-events=10 --gen-nparticles=$n --output-directory=$sim/ \
      --gen-eta=-1:1 --constraint-step-size-mm=1 --search-window 3:3 )
  D=$TRACCC_TEST_DATA_DIR/$sim
  geo=$(ls $D/*geometry*.json 2>/dev/null | head -1); mat=$(ls $D/*material*.json 2>/dev/null | head -1)
  grid=$(ls $D/*grid*.json 2>/dev/null | head -1)
  echo "detector files: $geo $mat $grid"
  for exe in traccc_truth_fitting_example traccc_truth_fitting_example_cuda; do
    [ -x $B/$exe ] || { echo "missing $exe"; continue; }
    ( cd $R && run ${exe#traccc_}_$tag $B/$exe --input-directory=$sim/ --check-performance --detector-file=$geo \
        --material-file=$mat --grid-file=$grid --input-events=10 --constraint-step-size-mm=1 --search-window 3:3 )
  done
done
ls -la $R
grep -h -i -E 'fitted|tracks|throughput|time|chi|pull' $R/*truth_fitting*.log | head -40
