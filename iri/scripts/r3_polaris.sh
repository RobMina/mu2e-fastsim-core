#!/bin/bash
# R3 on one Polaris node: cached MDC2025as CrvSteps (Eagle r3_inputs, staged by stage_r3_eagle.sh) replayed through the
# fused device response, waveforms, reco and the crvresponse objective, one reco-r3 process per A100, then the node merge.
# usage (make_inline_job.py --args): r3_polaris.sh [TARGET] [extra reco-r3 args ...]   (e.g. --v05 --knob cross_talk=0.06)
#   TARGET: events with CRV steps for the node (default 1000000, split over the 4 GPUs); extras default "--v05".
# Lookup tables come from CVMFS into /local/scratch and are deleted on exit. Outputs: $M2FS_EAGLE/r3/<time>_<jobid>/.
# R3_KNOBFILE (env, a file on Eagle): one line per GPU of reco-r3 --knob values (empty line = nominal); each GPU then
# runs its own point on all step tables with TARGET events, and the node merge is skipped (points must not be summed).
# R3_TABLES (env): space-separated step-table sequencers to use (e.g. "00000990 00001023"), default all staged tables.
T_JOB=$(date +%s)
E=${M2FS_EAGLE:?}; V=$E/venv/bin/python; I=$E/r3_inputs
TARGET=${1:-1000000}; shift || true; EXTRA=("$@"); [ ${#EXTRA[@]} -gt 0 ] || EXTRA=(--v05)
cd $E/code || exit 2
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
git pull -q --ff-only && git submodule update -q --init --recursive
git log -1 --format="code %h %s"; git -C crv log -1 --format="crv %h"; git -C core log -1 --format="core %h"
[ -s $I/MANIFEST.txt ] || { echo "no $I/MANIFEST.txt: run stage_r3_eagle.sh"; exit 3; }
(cd $I && grep -v '^crvextrap' MANIFEST.txt | awk '{print $1"  "$2}' | md5sum -c --quiet) || { echo "r3_inputs md5 check failed"; exit 3; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$E/code/core:$E/code/crv${PYTHONPATH:+:$PYTHONPATH}
export MU2E_SEARCH_PATH=$E/offline_data
O=$E/r3/$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
L=/local/scratch/crvfs_lookup_$$; mkdir -p $L; chmod 700 $L; trap 'rm -rf $L; echo "deleted $L"' EXIT
for f in LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root; do
  $V -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || exit 4; done
export CRVFS_LOOKUP_DIR=$L
$V -c "import crvfastsim.objective.crvresponse, crvfastsim.reco.r3" || exit 5
if [ -n "${R3_TABLES:-}" ]; then FILES=(); for q in $R3_TABLES; do FILES+=($I/steps/crvsteps.001400_$q.root); done
else FILES=($I/steps/crvsteps.*.root); fi
for f in "${FILES[@]}"; do [ -s "$f" ] || { echo "missing $f"; exit 3; }; done
NG=4; PER=$(( (TARGET + NG - 1) / NG ))
echo "target $TARGET ($PER per GPU), ${#FILES[@]} step tables, extras ${EXTRA[*]}, setup $(( $(date +%s) - T_JOB )) s"
KN=(); if [ -n "${R3_KNOBFILE:-}" ]; then mapfile -t KN < "$R3_KNOBFILE"; PER=$TARGET; cp "$R3_KNOBFILE" $O/knobs.txt; fi
pids=()
for g in $(seq 0 $((NG-1))); do
  if [ -n "${R3_KNOBFILE:-}" ]; then mine=("${FILES[@]}"); KA=(); [ -n "${KN[$g]:-}" ] && KA=(--knob ${KN[$g]})
  else mine=(); KA=(); for k in "${!FILES[@]}"; do [ $((k % NG)) -eq $g ] && mine+=("${FILES[$k]}"); done; fi
  echo "gpu $g: ${#mine[@]} tables, target $PER, knobs: ${KN[$g]:-nominal}"
  CUDA_VISIBLE_DEVICES=$g $V -m crvfastsim reco-r3 "${mine[@]}" --calib $I/cond/crvcalib.csv \
    --deviation $I/cond/crvphoton.csv --jax --objective --target $PER --seed $((1000 + g)) "${EXTRA[@]}" "${KA[@]}" \
    --emit $O/r3.gpu$g.json > $O/r3.gpu$g.txt 2>&1 & pids+=($!)
done
rc=0; for g in $(seq 0 $((NG-1))); do wait ${pids[$g]} || { echo "gpu $g failed"; rc=6; }; done
WALL=$(( $(date +%s) - T_JOB ))
for g in $(seq 0 $((NG-1))); do echo "== gpu $g"; tail -3 $O/r3.gpu$g.txt; done
if [ -z "${R3_KNOBFILE:-}" ]; then
  $V -m crvfastsim reco-r3 "$O/r3.gpu*.json" --merge --wall $WALL --outdir $O > $O/r3_merge.txt 2>&1 || rc=7
  cat $O/r3_merge.txt | head -30
fi
echo "job wall $WALL s (queue excluded); outdir $O"; echo exit=$rc; exit $rc
