#!/bin/bash
# P2 of the CRV fused device response on real G4 CrvSteps, on a Fermilab EAF A100 session (run in the EAF terminal).
# Inputs are read in place from /exp/mu2e/data (nothing leaves FNAL); outputs go to $D/bench/eaf_p2_<time>.
# Also runs the synthetic sample of Polaris PBS 7764070 for a direct A100-PCIe vs A100-SXM4 comparison.
# usage: bash core/eaf/p2_real_steps.sh [crvsteps.root ...]   (default: G4 shards 11-13)
# env: VENV (venv dir), OUT, REPEAT (warm passes, 2), NEV (synthetic events, 20000), MAXSTEPS, CPU=1 (test without GPU)
M2FS_CODE=${M2FS_CODE:-/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim}
D=${CRVFS_DATA:-/exp/mu2e/data/users/ram2aq/crv_fastsim}
VENV=${VENV:-$D/venv_eaf}
BASE=/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python   # Mu2e ana env: numpy, pandas, uproot, awkward, scipy
JAXPKG="jax[cuda12]==0.11.2"; [ -n "${CPU:-}" ] && JAXPKG="jax==0.11.2"
[ -x "$VENV/bin/python" ] || "$BASE" -m venv --system-site-packages "$VENV" || exit 2
"$VENV/bin/python" -c "import jax" 2>/dev/null || "$VENV/bin/python" -m pip install -q "$JAXPKG" || exit 3
PY=$VENV/bin/python
export PYTHONPATH=$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker
export CRVFS_LOOKUP_DIR=$D/lookup_v6_0 XLA_PYTHON_CLIENT_PREALLOCATE=false
G=$D/geometry
O=${OUT:-$D/bench/eaf_p2_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$O" || exit 4
{ hostname; git -C "$M2FS_CODE" log -1 --format="umbrella %h"; git -C "$M2FS_CODE/crv" log -1 --format="crv %h"
  nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
  "$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"; } | tee "$O/env.txt"
if [ -z "${CPU:-}" ] && ! grep -q -i cuda "$O/env.txt"; then echo "JAX sees no GPU: start the session with the A100 profile"; exit 5; fi
shards=("$@"); [ ${#shards[@]} -eq 0 ] && shards=("$D"/g4ref_v1_tables/crvsteps.001400_000000{11,12,13}.root)
rc=0
for f in "${shards[@]}"; do
  n=$(basename "$f" .root)
  "$PY" -m crvfastsim v4sipm "$f" --device --no-stock --seed 1 --repeat "${REPEAT:-2}" ${MAXSTEPS:+--max-steps $MAXSTEPS} \
    --calib "$G/crvcalib_sim_best_v1_5_run1400.csv" --deviation "$G/crvphoton_sim_best_v1_5_run1400.csv" \
    --outdir "$O/$n" > "$O/$n.txt" 2>&1 || { echo "FAILED $n (see $O/$n.txt)"; rc=6; }
  echo "== $n"; tail -9 "$O/$n.txt"
done
S=$O/synth_steps.root   # synthetic, no Mu2e input: same sample as PBS 7764070
"$PY" -m crvfastsim.response.synthsteps --events "${NEV:-20000}" --out "$S" > "$O/synth.txt" 2>&1 &&
  "$PY" -m crvfastsim v4sipm "$S" --device --no-stock --seed 1 --repeat "${REPEAT:-2}" ${MAXSTEPS:+--max-steps $MAXSTEPS} \
    --outdir "$O/synth" >> "$O/synth.txt" 2>&1 || { echo "FAILED synthetic (see $O/synth.txt)"; rc=7; }
echo "== synthetic"; tail -9 "$O/synth.txt"; rm -f "$S"
echo "outdir $O"; echo "exit=$rc"; exit $rc
