"""fastsimkit: the Mu2eBO evaluator-contract kit for the fast chain on a compute endpoint.

Contract (Mu2eBO core/contract.py): submit(name, params, files, inputs) -> {"handle": name};
status(handle) -> {"state", "message", "poll_ms", "progress"}; results(handle) -> {"metrics", "files", "metadata"};
check, describe, cancel. One BO point = one job: the kit renders a job body from a task template and the point's
params, submits it through an endpoint (``mu2efastsim.endpoints``), and on completion fetches ``out/`` and reads
``out/metrics.json`` (a flat {name: number} map written by the task). Tasks are templates in TASKS; a task names its
resources, the params it accepts and the metrics it returns.

Run as an MCP stdio server (``python -m mu2efastsim.endpoints.kit``) for a kits.toml entry, or use FastsimKit directly.
Environment: FASTSIMKIT_ENDPOINT (default uva; eaf), FASTSIMKIT_STATE_DIR (local job records and fetched outputs),
FASTSIMKIT_MAX_SU_PER_JOB (default 500), FASTSIMKIT_SIF (image path on the site; default
<FASTSIMKIT_SCRATCH>/mu2efastsim.sif, needed only by image-based endpoints). Endpoint settings: see
mu2efastsim.site and each endpoint module.
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import EndpointError, Resources, get
from mu2efastsim import site
from .parameters import schema, validate as validate_parameters
from . import capabilities, geometry as geometry_artifacts
from .artifacts import digest
from .submission import read_json, durable_json, submission_lock

VERSION = "1"
POLL_MS = 60_000


@dataclass(frozen=True)
class Task:
    params: Tuple[str, ...]
    metrics: Tuple[str, ...]
    resources: Resources
    body: str           # bash; @sif@ and @p_<param>@ are substituted (shell-quoted); writes $M2FS_OUT/metrics.json
    defaults: Dict[str, Any] = field(default_factory=dict)   # optional params (absent from a call: this value)


_COMMON = """module load apptainer >/dev/null 2>&1
M2FS_OUT=$(readlink -f "$M2FS_OUT")   # the scratch area may be a symlink (UVA): bind the real path
@runtime_a@echo "host $(hostname) job ${SLURM_JOB_ID:-none} out $M2FS_OUT"; nvidia-smi -L
A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != 'gpu')" || { echo "JAX not on the GPU"; exit 5; }
"""

# Copy-compute-delete (E9): the CRVConditions tables are Mu2e-internal. They are fetched per job from the CVMFS
# Stratum-1 into node tmpfs (mode 700), used through CRVFS_LOOKUP_DIR and deleted by an EXIT trap; the deletion is
# recorded in out/cleanup.txt (the kit refuses results without it). Never on shared or scratch storage.
_TABLES = """L=${TMPDIR:-/tmp}/crvfs_kit_j${SLURM_JOB_ID:-0}_$$; mkdir -p "$L"; chmod 700 "$L"
trap 'rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
trap 'exit 143' TERM INT
for d in ${TMPDIR:-/tmp}/crvfs_*_j*; do [ -d "$d" ] && [ "$d" != "$L" ] && [ -O "$d" ] || continue; j=${d##*_j}; j=${j%%_*}; squeue -h -j "$j" -o %i 2>/dev/null | grep -q . || rm -rf "$d"; done
@runtime_t@
for f in @tables@; do
  T python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f "$L/$f" >/dev/null || { echo "fetch $f failed"; exit 3; }
done
ls -la "$L"
"""

# json -> metrics.json helper run inside the image: flattens chosen keys of the task's JSON outputs
_METRICS = """A python - "$M2FS_OUT" <<'EOF'
import json, os, sys
o = sys.argv[1]; m = {}
def load(p):
    p = os.path.join(o, p)
    return json.load(open(p)) if os.path.exists(p) else None
@metrics_py@
json.dump({k: float(v) for k, v in m.items()}, open(os.path.join(o, "metrics.json"), "w"))
EOF
"""

TASKS: Dict[str, Task] = {
    # synthetic CRV reco chain throughput: a smoke task for the endpoint (no Mu2e-internal input)
    "reco_bench": Task(
        params=("events", "seed"), metrics=("events_per_s", "first_call_s", "events"),
        resources=Resources(cores=8, mem_gb=48, gpus=1, gpu_type="a100", minutes=30),
        body=_COMMON + """A python -m crvfastsim reco-bench synthetic --stage chain --max-events @p_events@ --length 12 \\
  --batch 65536 --repeat 3 --seed @p_seed@ --outdir "$M2FS_OUT/bench" > "$M2FS_OUT/bench.txt" 2>&1 || exit 6
A python - "$M2FS_OUT" <<'EOF'
import glob, json, sys
j = json.load(open(sorted(glob.glob(sys.argv[1] + "/bench/*.json"))[-1]))
def find(d, k):
    if isinstance(d, dict):
        if k in d and isinstance(d[k], (int, float)): return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None: return r
    return None
m = {k: float(find(j, k)) for k in ("events_per_s", "first_call_s", "events") if find(j, k) is not None}
json.dump(m, open(sys.argv[1] + "/metrics.json", "w"))
EOF
"""),
    # CRV response throughput with the REAL v6_0 lookup table and photon map (copy-compute-delete)
    "response_bench": Task(
        params=("rows", "sipms", "seed"),
        metrics=("photon_rows_per_s", "photons_per_row", "sipms_per_s", "charges_per_photon"),
        resources=Resources(cores=8, mem_gb=64, gpus=1, gpu_type="a100", minutes=30),
        body=_COMMON + _TABLES.replace("@tables@", "LookupTable_3200_0 photonMap.root") + """B="python -m crvfastsim.response.bench --repeat 3"
T $B --table 3200_0 --rows @p_rows@ --numpy-rows 0 --seed @p_seed@ --outdir "$M2FS_OUT/ph" > "$M2FS_OUT/ph.txt" 2>&1 || exit 6
T $B --stage sipm --photon-map "$L/photonMap.root" --sipms @p_sipms@ --numpy-sipms 0 --seed @p_seed@ --outdir "$M2FS_OUT/si" > "$M2FS_OUT/si.txt" 2>&1 || exit 6
""" + _METRICS.replace("@metrics_py@", """ph, si = load("ph/photon_bench.json"), load("si/sipm_bench.json")
m.update(photon_rows_per_s=ph["jax_rows_per_s"], photons_per_row=ph["jax_photons_per_row"],
         sipms_per_s=si["jax_sipms_per_s"], charges_per_photon=si["jax_charges_per_photon"])""")),
    # V4: JAX photon and SiPM response vs the numpy ports on synthetic muon CrvSteps, real tables; pass flags per
    # check and the largest |relative PE deviation| over the four sector groups
    "v4_response": Task(
        params=("events", "seed"),
        metrics=("v4_photons_pass", "v4_sipm_pass", "v4_photons_max_abs_pe_rel", "v4_sipm_max_abs_pe_rel",
                 "v4_photons_jax_s", "v4_sipm_jax_s"),
        resources=Resources(cores=16, mem_gb=96, gpus=1, gpu_type="a100", minutes=45),
        body=_COMMON + _TABLES.replace("@tables@", "LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root")
        + """T python -m crvfastsim.response.synthsteps --events @p_events@ --seed @p_seed@ --out "$L/steps.root" > "$M2FS_OUT/steps.txt" 2>&1 || exit 6
T python -m crvfastsim v4photons "$L/steps.root" --jax --no-stock --seed @p_seed@ --outdir "$M2FS_OUT/v4ph" > "$M2FS_OUT/v4ph.txt" 2>&1
T python -m crvfastsim v4sipm "$L/steps.root" --jax --no-stock --seed @p_seed@ --outdir "$M2FS_OUT/v4si" > "$M2FS_OUT/v4si.txt" 2>&1
""" + _METRICS.replace("@metrics_py@", """for key, f in (("v4_photons", "v4ph/v4photons_stats.json"), ("v4_sipm", "v4si/v4sipm_stats.json")):
    j = load(f)
    if j is None:
        m[key + "_pass"] = 0.0
        continue
    m[key + "_pass"] = 1.0 if j.get("pass") else 0.0
    m[key + "_max_abs_pe_rel"] = max(abs(g["pe_per_sipm_rel"]) for g in j["groups"].values())
    m[key + "_jax_s"] = j.get("time_s_fast", 0.0)""")),
}

# default runtime: the project image under Apptainer (Slurm sites); an endpoint with a runtime() method supplies its own
# A/T definitions instead (EAF: a venv on shared FNAL storage, since EAF starts its own images)
APPTAINER_RUNTIME = {"a": 'A() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} -B "$M2FS_OUT" --pwd /opt/mu2e-fastsim @sif@ "$@"; }\n',
                     "t": 'T() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},CRVFS_LOOKUP_DIR=$L -B "$L" -B "$M2FS_OUT" --pwd /opt/mu2e-fastsim @sif@ "$@"; }'}

# Tasks on Mu2e-internal inputs read in place (G4 gen shards, the tier-1b asset set, CRVCalib/CRVPhoton CSVs under
# @fnal_data@): only on an endpoint that mounts that storage (EAF; `endpoint.fnal_data`). Elsewhere the kit refuses them
# at check and submit: those inputs may not be copied off site except copy-compute-delete, which these tasks do not do.
# Tier 1b: the default asset set strict_v4 (library + topology/v3 template sidecars; chainpoint.TIER1B_SETS).
_FNAL_INPUTS = """D=@fnal_data@
G=$D/geometry; T1B=$D/tier1b_strict_v2; GDML=$G/mu2e_extracted_v04.gdml
CAL=$G/crvcalib_sim_best_v1_5_run1400.csv; DEVF=$G/crvphoton_sim_best_v1_5_run1400.csv
for f in "$T1B/library.root" "$T1B/topology_sidecar_strict_v2_v2.npz" "$T1B/topology_sidecar_strict_v2_v2.npz.json" \\
  "$T1B/templates_v3.npz" "$T1B/templates_v3.npz.json" "$CAL" "$DEVF" "$GDML"; do [ -r "$f" ] || { echo "missing $f"; exit 3; }; done
"""

TASKS.update({
    # the whole fast chain on real CRY gen shards (runchain: transport -> tier 1b -> photons -> SiPM -> waveforms ->
    # pulses -> coincidences), JAX on the GPU; clusters per generated event per sector type vs the stock chain's
    "chain": Task(
        params=("shards", "max_events", "seed"),
        metrics=("events_with_steps_per_s", "ms_per_event_with_steps", "generated_events", "events_with_steps",
                 "clusters_per_generated_EX", "clusters_per_generated_T1", "clusters_per_generated_T2",
                 "clusters_per_generated_M", "max_abs_cluster_rate_rel"),
        resources=Resources(cores=8, mem_gb=32, gpus=1, gpu_type="a100", minutes=60),
        body=_COMMON + _TABLES.replace("@tables@", "LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root")
        + _FNAL_INPUTS + """GEN=(); REF=()
N=@p_shards@; [ "$N" -ge 1 ] && [ "$N" -le 10 ] || { echo "shards must be 1..10 (held-out G4 shards 11-20)"; exit 2; }
for i in $(seq 11 $((10 + N))); do   # stock chain (G4 + digi + reco) on the same shards: the V2 reference tables
  GEN+=("$D/g4ref_v1_tables/gen.001400_000000$i.root"); r="$D/v2_fastC_v1/ref.001400_000000$i.root"
  [ -r "$r" ] && REF+=("$r"); done
T python -m crvfastsim runchain "${GEN[@]}" --tier1b strict_v4 --tier1b-dir "$T1B" --calib "$CAL" --deviation "$DEVF" --gdml "$GDML" --jax \
  --max-events @p_max_events@ --seed @p_seed@ ${REF:+--compare "${REF[@]}"} --outdir "$M2FS_OUT/chain" > "$M2FS_OUT/chain.txt" 2>&1 || exit 6
""" + _METRICS.replace("@metrics_py@", """j = load("chain/runchain_stats.json")
T = j["timers"]; nev = max(T.get("events_with_steps", 1), 1)
work = sum(v for k, v in T.items() if k.endswith("_s") and k != "setup_s")
m.update(events_with_steps_per_s=nev / max(work, 1e-9), ms_per_event_with_steps=1e3 * work / nev,
         generated_events=j["generated_events"], events_with_steps=nev)
for k, v in j.get("clusters_per_generated", {}).items():
    m["clusters_per_generated_" + k] = v
ref = j.get("clusters_per_generated_ref")
if ref:
    m["max_abs_cluster_rate_rel"] = max(abs(j["clusters_per_generated"].get(k, 0.0) / v - 1) for k, v in ref.items() if v)""")),
    # regression set on the endpoint (veto plan E-P6): unit tests, V4 photons and SiPM (JAX and fused device), the
    # reco reproducibility check (R5/P6: same seed bit-identical, CPU vs GPU statistical) and the chain on one shard;
    # regression_pass = 1 only if every part passes
    "regression": Task(
        params=("events", "seed"),
        metrics=("pytest_failed", "v4_photons_pass", "v4_sipm_jax_pass", "v4_sipm_device_pass", "repro_bit_identical",
                 "repro_cpu_gpu_statistical", "chain_ok", "regression_pass"),
        resources=Resources(cores=8, mem_gb=32, gpus=1, gpu_type="a100", minutes=90),
        body=_COMMON + _TABLES.replace("@tables@", "LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root")
        + _FNAL_INPUTS + """T python -m pytest -q -p no:cacheprovider crv/tests core/tests > "$M2FS_OUT/pytest.txt" 2>&1
T python -m crvfastsim.response.synthsteps --events @p_events@ --seed @p_seed@ --out "$L/steps.root" > "$M2FS_OUT/steps.txt" 2>&1 || exit 6
T python -m crvfastsim v4photons "$L/steps.root" --jax --no-stock --seed @p_seed@ --outdir "$M2FS_OUT/v4ph" > "$M2FS_OUT/v4ph.txt" 2>&1
T python -m crvfastsim v4sipm "$L/steps.root" --jax --no-stock --seed @p_seed@ --outdir "$M2FS_OUT/v4si_jax" > "$M2FS_OUT/v4si_jax.txt" 2>&1
T python -m crvfastsim v4sipm "$D/g4ref_v1_tables/crvsteps.001400_00000012.root" --device --no-stock --seed @p_seed@ \
  --calib "$CAL" --deviation "$DEVF" --outdir "$M2FS_OUT/v4si_dev" > "$M2FS_OUT/v4si_dev.txt" 2>&1
A python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir "$M2FS_OUT/repro_gpu" > "$M2FS_OUT/repro_gpu.txt" 2>&1
RX=$(ls "$M2FS_OUT"/repro_gpu/repro_*.npz 2>/dev/null | head -1)
JAX_PLATFORMS=cpu A python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir "$M2FS_OUT/repro_cpu" ${RX:+--against "$RX"} > "$M2FS_OUT/repro_cpu.txt" 2>&1
T python -m crvfastsim runchain "$D/g4ref_v1_tables/gen.001400_00000011.root" --tier1b strict_v4 --tier1b-dir "$T1B" --calib "$CAL" --deviation "$DEVF" --gdml "$GDML" \
  --jax --max-events 10000 --seed @p_seed@ --outdir "$M2FS_OUT/chain" > "$M2FS_OUT/chain.txt" 2>&1
find "$M2FS_OUT" -name '*.npz' -delete
""" + _METRICS.replace("@metrics_py@", """import re
txt = open(os.path.join(o, "pytest.txt")).read() if os.path.exists(os.path.join(o, "pytest.txt")) else ""
fm = re.search(r"(\\d+) failed", txt); pm = re.search(r"(\\d+) passed", txt)
m["pytest_failed"] = float(fm.group(1)) if fm else (0.0 if pm else 1.0)
for key, f in (("v4_photons_pass", "v4ph/v4photons_stats.json"), ("v4_sipm_jax_pass", "v4si_jax/v4sipm_stats.json"),
               ("v4_sipm_device_pass", "v4si_dev/v4sipm_stats.json")):
    j = load(f); m[key] = 1.0 if j and j.get("pass") else 0.0
g, c = load("repro_gpu/repro_stats.json"), load("repro_cpu/repro_stats.json")
m["repro_bit_identical"] = 1.0 if g and g.get("pass_bit_identical") else 0.0
m["repro_cpu_gpu_statistical"] = 1.0 if c and ((c.get("against") or {}).get("comparison") or {}).get("pass_statistical") else 0.0
ch = load("chain/runchain_stats.json")
m["chain_ok"] = 1.0 if ch and ch.get("clusters_per_generated") else 0.0
m["regression_pass"] = float(m["pytest_failed"] == 0 and all(m[k] == 1.0 for k in (
    "v4_photons_pass", "v4_sipm_jax_pass", "v4_sipm_device_pass", "repro_bit_identical", "repro_cpu_gpu_statistical", "chain_ok")))""")),
})

# Tier C4 of the veto optimisation (crvfastsim.veto): a veto rule applied to a cached selection of cosmic tracks and
# coincidences (inefficiency; made by crvfastsim.veto.cosmic_ana under @fnal_data@/veto_ve1) and, if the selection has
# a deadtime sample, to the selected signal tracks of CE + beam frames (deadtime, methods S and F of
# crvfastsim.veto.deadtime; @fnal_data@/deadtime_run1a). CPU only, seconds per point. Gates < 0 mean off.
# gate_mode: 0 = an out-of-gate coincidence removes the event (mu2e_cosmic_ana), 1 = it only cannot veto (58017).
_VETO_C4 = r"""D=@fnal_data@
case @p_selection@ in
  run1a_dev) SEL=$D/veto_ve1/run1a_dev_sel.parquet; DT=$D/deadtime_run1a/ce_mix1bb_dev_sel.parquet;;
  *) echo "unknown selection @p_selection@"; exit 2;;
esac
[ -r "$SEL" ] || { echo "missing $SEL"; exit 3; }
G() { awk -v g="$1" 'BEGIN { if (g + 0 < 0) print "null"; else print g }'; }
GM=$(awk -v g=@p_gate_mode@ 'BEGIN { print (g + 0 >= 0.5) ? "coinc" : "event" }')
printf '{"name": "kit", "dt_lo": %s, "dt_hi": %s, "min_pe": %s, "min_layers": %s, "gate_start_lo": %s, "gate_end_hi": %s, "gate_mode": "%s"}\n' \
  @p_dt_lo@ @p_dt_hi@ @p_min_pe@ @p_min_layers@ "$(G @p_gate_start_lo@)" "$(G @p_gate_end_hi@)" "$GM" > "$M2FS_OUT/card.json"
cat "$M2FS_OUT/card.json"
( cd "$M2FS_CODE" && PYTHONPATH="$M2FS_CODE/crv:$M2FS_CODE/core" "$M2FS_VENV/bin/python" -m crvfastsim.veto "$SEL" \
  --card "$M2FS_OUT/card.json" --pmin @p_pmin@ --pmax @p_pmax@ --outdir "$M2FS_OUT/veto" ) > "$M2FS_OUT/veto.txt" 2>&1 \
  || { cat "$M2FS_OUT/veto.txt"; exit 6; }
tail -1 "$M2FS_OUT/veto.txt"
if [ -r "$DT" ]; then
  ( cd "$M2FS_CODE" && PYTHONPATH="$M2FS_CODE/crv:$M2FS_CODE/core" "$M2FS_VENV/bin/python" -m crvfastsim.veto.deadtime "$DT" \
    --card "$M2FS_OUT/card.json" --pmin @p_pmin@ --pmax @p_pmax@ --outdir "$M2FS_OUT/deadtime" ) > "$M2FS_OUT/deadtime.txt" 2>&1 \
    || { cat "$M2FS_OUT/deadtime.txt"; exit 7; }
  tail -3 "$M2FS_OUT/deadtime.txt"
fi
"$M2FS_VENV/bin/python" - "$M2FS_OUT" <<'PY'
import json, os, sys
o = sys.argv[1]; j = json.load(open(os.path.join(o, "veto", "veto_stats.json"))); v = j["vetoed"]
m = {"vetoed_frac": v["frac"], "inefficiency": 1.0 - v["frac"], "vetoed_lo": v["lo"], "vetoed_hi": v["hi"],
     "vetoed": v["k"], "selected": j["selected"], "passing_gate": j["passing_gate"]}
dp = os.path.join(o, "deadtime", "deadtime_stats.json")
if os.path.exists(dp):
    d = json.load(open(dp))
    m.update({"deadtime": d["F"]["frac"], "deadtime_err": d["F"]["err"], "deadtime_full": d["S"]["frac"],
              "deadtime_full_lo": d["S"]["lo"], "deadtime_full_hi": d["S"]["hi"], "deadtime_selected": d["selected"]})
else:
    m.update({k: float("nan") for k in ("deadtime", "deadtime_err", "deadtime_full", "deadtime_full_lo",
                                        "deadtime_full_hi", "deadtime_selected")})
json.dump({k: float(x) for k, x in m.items()}, open(os.path.join(o, "metrics.json"), "w"))
PY
"""
TASKS["veto_c4"] = Task(
    params=("selection", "dt_lo", "dt_hi", "min_pe", "min_layers", "gate_start_lo", "gate_end_hi", "gate_mode", "pmin",
            "pmax"),
    metrics=("inefficiency", "vetoed_frac", "vetoed_lo", "vetoed_hi", "vetoed", "selected", "passing_gate",
             "deadtime", "deadtime_err", "deadtime_full", "deadtime_full_lo", "deadtime_full_hi", "deadtime_selected"),
    resources=Resources(cores=2, mem_gb=8, gpus=0, gpu_type="", minutes=15),
    body='@runtime_a@' + _VETO_C4, defaults={"gate_mode": 0})

FNAL_TASKS = {"chain", "regression", "veto_c4"}

# Delegated tasks: a task whose job needs more than a rendered body on some endpoints (inputs streamed into the running
# job) names a factory here; DELEGATES[task](kit) returns a handler with check/submit/status/results/cancel, used when
# its `applies(endpoint)` is true (otherwise the task renders as a plain body, e.g. on EAF).
DELEGATES: Dict[str, Any] = {}

# Beam stage (umbrella beam/, package beamfastsim): task beam_point, stopping-target metrics for one target geometry.
# Registered when the umbrella's beam/ is on PYTHONPATH; FNAL endpoints only (reads the MuBeamCat-derived table in place).
try:
    from beamfastsim import kit_task as _beam_task
except ImportError:
    _beam_task = None
if _beam_task is not None:
    _beam_task.register(TASKS, Task, Resources, FNAL_TASKS)
    if hasattr(_beam_task, "register_delegate"):
        _beam_task.register_delegate(DELEGATES)

TABLE_TASKS = {"response_bench", "v4_response", "chain", "regression"}   # stage Mu2e-internal tables: cleanup record required

# CRV response tuning (crv package, crvfastsim.kit_task): task crv_response, one point of every chain.KNOBS knob scored
# against the KPP data reference. Registered when crvfastsim is importable: inputs read in place on FNAL endpoints (EAF),
# streamed into the job on Slurm-over-SSH endpoints (UVA; delegate crvfastsim.kit_uva).
try:
    from crvfastsim import kit_task as _crv_task
except ImportError:
    _crv_task = None
if _crv_task is not None:
    _crv_task.register(TASKS, Task, Resources, FNAL_TASKS, TABLE_TASKS, common=_COMMON, tables=_TABLES,
                       delegates=DELEGATES)


# Optional fixed primary-only tracker pipeline; registration does not deploy the endpoint.
try:
    from trkfastsim import kit_task as _tracker_task
except ImportError:
    _tracker_task = None
if _tracker_task is not None:
    _tracker_task.register(TASKS, Task, Resources, FNAL_TASKS)
    _tracker_task.register_delegate(DELEGATES)
    TABLE_TASKS.add("tracker_point")  # scratch EventNtuple cleanup is mandatory


def _arg(v) -> str:
    """A param as a command-line word; Mu2eBO sends every "number" as a float, so 200000.0 becomes 200000."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def admit_resources(endpoint, resources):
    """Run an endpoint's pure resource policy before preparing code or recording work."""
    if hasattr(endpoint, "admit_resources"):
        try:
            endpoint.admit_resources(resources)
        except EndpointError as exc:
            raise ValueError(str(exc)) from exc


class FastsimKit:
    def __init__(self, endpoint=None, state_dir: str | Path | None = None, sif: str | None = None):
        root = Path(state_dir or os.environ.get("FASTSIMKIT_STATE_DIR", "~/.mu2efastsim/kit")).expanduser()
        self.out_root = root / "out"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._sif = sif
        self.ep = endpoint or get(os.environ.get("FASTSIMKIT_ENDPOINT", "uva"), state_dir=root / "jobs",
                                  max_su_per_job=float(os.environ.get("FASTSIMKIT_MAX_SU_PER_JOB", "500")))
        self._cp = None
        self._delegates: Dict[str, Any] = {}

    def _delegate(self, params=None, handle=None):
        """Handler for a delegated task on this endpoint, or None (plain rendered task)."""
        task = params.get("task") if params is not None else self._task_of(handle)
        factory = DELEGATES.get(task)
        if factory is None or not factory.applies(self.ep):
            return None
        if task not in self._delegates:
            self._delegates[task] = factory.make(self)
        return self._delegates[task]

    @property
    def chain_point(self):
        """The fan-out task (chainpoint.py): one point = N one-GPU jobs with inputs streamed in; built on first use."""
        if self._cp is None:
            from .chainpoint import ChainPoint
            self._cp = ChainPoint(self)
        return self._cp

    def _is_cp(self, params=None, handle=None) -> bool:
        if params is not None:
            return params.get("task") == "chain_point"
        return self._task_of(handle) == "chain_point"

    @staticmethod
    def _task(params: Dict[str, Any]) -> Task:
        if not isinstance(params, dict) or any(not isinstance(k, str) for k in params):
            raise ValueError("fastsimkit: params must be an object with string keys")
        name = params.get("task")
        if not isinstance(name, str) or name not in TASKS:
            raise ValueError(f"fastsimkit: unknown task {name!r}; choose from {sorted(TASKS)}")
        t = TASKS[name]
        extra = set(params) - set(t.params) - {"task"}
        missing = set(t.params) - set(params) - set(t.defaults)
        if extra or missing:
            raise ValueError(f"fastsimkit: task {name!r} takes {list(t.params)}; extra {sorted(extra)}, "
                             f"missing {sorted(missing)}")
        return t

    def _parameters(self, params):
        t = self._task(params)
        return {"task": params["task"], **validate_parameters(params["task"], params, t.params, t.defaults)}

    @property
    def sif(self) -> str:
        """The project image: the constructor argument, $FASTSIMKIT_SIF or <FASTSIMKIT_SCRATCH>/mu2efastsim.sif."""
        return self._sif or site.setting("FASTSIMKIT_SIF") or site.scratch() + "/mu2efastsim.sif"

    @sif.setter
    def sif(self, value):
        self._sif = value

    def render(self, params: Dict[str, Any], geometry=None) -> Tuple[str, Resources]:
        params = self._parameters(params)
        t = self._task(params)
        fnal = getattr(self.ep, "fnal_data", None)
        if params["task"] in FNAL_TASKS and not fnal:
            raise ValueError(f"fastsimkit: task {params['task']!r} reads Mu2e-internal inputs in place and needs an "
                             f"endpoint that mounts FNAL storage (eaf); {getattr(self.ep, 'name', '?')!r} does not")
        rt = self.ep.runtime() if hasattr(self.ep, "runtime") else APPTAINER_RUNTIME
        body = t.body.replace("@runtime_a@", rt["a"]).replace("@runtime_t@", rt["t"])
        if geometry is not None:
            if params["task"] != "beam_point" or "@geometry_setup@" not in body:
                raise ValueError("fastsimkit: task does not support geometry handoff")
            setup = geometry_artifacts.stage_shell(geometry) + 'GEOM="$M2FS_GEOMETRY"\n'
        else:
            setup = ""
        subs = {**({"sif": shlex.quote(self.sif)} if "@sif@" in body else {}),
                **{f"p_{k}": shlex.quote(_arg(params.get(k, t.defaults.get(k)))) for k in t.params}}
        if fnal:
            subs["fnal_data"] = shlex.quote(fnal)
        for k, v in subs.items():
            body = body.replace(f"@{k}@", v)
        if any(s in body for s in ("@p_", "@sif@", "@tables@", "@metrics_py@", "@runtime_", "@fnal_data@")):
            raise ValueError(f"fastsimkit: unfilled placeholder in task {params['task']!r}")
        body = body.replace("@geometry_setup@", setup)
        admit_resources(self.ep, t.resources)
        return body, t.resources

    # -- contract ------------------------------------------------------------------------------------------------
    def _admit(self, name, params, files, inputs, requested_metrics=None):
        import re
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
            raise ValueError(f"fastsimkit: bad handle {name!r}")
        if not isinstance(params, dict) or any(not isinstance(k, str) for k in params):
            raise ValueError("fastsimkit: params must be an object with string keys")
        ref = geometry_artifacts.reference(files, inputs, params)
        if ref is not None:
            params = {**params, "geom": "nominal"}
        if self._is_cp(params):
            handler = self.chain_point
            from .chainpoint import validate
            params = {"task": "chain_point", **validate(params)}
        else:
            params = self._parameters(params)
            handler = self._delegate(params=params)
        known = self._task_of(name)
        if known and known != params["task"]:
            raise ValueError(f"fastsimkit: {name!r} already belongs to task {known!r}")
        rendered = None
        old = self._request(name)
        if old is None:
            if handler is not None:
                result = handler.check(name, params)
                if not result["ok"]:
                    raise ValueError(result["message"])
            else:
                rendered = self.render(params)
        from . import chainpoint
        if old is None and known and (ref is not None or requested_metrics is not None):
            raise ValueError("fastsimkit: cannot attach a new contract to an already submitted legacy handle")
        cap = old["capabilities"] if old else capabilities.resolve(params["task"], params, self.ep, TASKS, chainpoint)
        names = capabilities.require_metrics(requested_metrics, cap)
        if old is not None:
            self._resolve_request(name, params, ref, names, old)
            if handler is not None and hasattr(handler, "check_binding"):
                handler.check_binding(name, params, old["geometry"])
        return params, handler, rendered, ref, cap, names

    def _request_path(self, name):
        return self.out_root.parent / "requests" / f"{name}.json"

    def _request(self, name):
        path = self._request_path(name)
        record = read_json(path)
        if record is None:
            if path.exists():
                raise ValueError("fastsimkit: corrupt request record")
            return None
        if (not isinstance(record, dict) or record.get("schema_version") != 1
                or record.get("request_sha") != digest({k: v for k, v in record.items() if k != "request_sha"})):
            raise ValueError("fastsimkit: corrupt request record")
        return record

    def _target(self):
        return {"endpoint": self.ep.name, **{k: str(getattr(self.ep, k, "")) for k in
                ("host", "account", "partition", "workdir", "fnal_data")}}

    def _resolve_request(self, name, params, ref, names, old):
        if old is not None:
            if old["params"] != params or old["requested_metrics"] != names or old["target"] != self._target():
                raise ValueError("fastsimkit: request differs from the bound parameters, metrics or endpoint")
            if geometry_artifacts.identity(ref) != geometry_artifacts.identity(old["geometry"]):
                raise ValueError("fastsimkit: geometry differs from the bound request")
            return geometry_artifacts.resolve(ref, saved=old["geometry"])
        return geometry_artifacts.resolve(ref)

    def check(self, name, params, files, inputs, requested_metrics=None) -> Dict[str, Any]:
        try:
            p, handler, rendered, ref, cap, names = self._admit(name, params, files, inputs, requested_metrics)
            old = self._request(name)
            artifact = self._resolve_request(name, p, ref, names, old)
            if artifact is not None and handler is None and old is None:
                self.render(p, geometry=artifact)
        except (ValueError, EndpointError) as e:
            return {"ok": False, "message": str(e)}
        result = {"ok": True, "message": ""}
        if requested_metrics is not None or ref is not None:
            result.update(capabilities=old["capabilities"] if old else cap,
                          geometry=geometry_artifacts.identity(artifact))
        return result

    def _task_of(self, handle) -> str:
        p = self.out_root.parent / "tasks" / f"{handle}.txt"
        return p.read_text().strip() if p.exists() else ""

    def submit(self, name, params, files, inputs, requested_metrics=None) -> Dict[str, Any]:
        from dataclasses import asdict
        p, handler, rendered, ref, cap, names = self._admit(name, params, files, inputs, requested_metrics)
        with submission_lock(self.out_root.parent / "requests", name):
            old = self._request(name)
            artifact = self._resolve_request(name, p, ref, names, old)
            record = old
            if record is None and (requested_metrics is not None or artifact is not None):
                if self._task_of(name):
                    raise ValueError("fastsimkit: cannot attach a new contract to an already submitted legacy handle")
                if handler is None:
                    rendered = self.render(p, geometry=artifact)
                record = {"schema_version": 1, "params": p, "requested_metrics": names,
                          "geometry": artifact, "capabilities": cap, "target": self._target(),
                          "body": rendered[0] if handler is None else None,
                          "resources": asdict(rendered[1]) if handler is None else None}
                record["request_sha"] = digest(record)
                durable_json(self._request_path(name), record)
            if handler is not None:
                self._record_task(name, p["task"])
                if artifact is not None:
                    return handler.submit(name, p, geometry=artifact)
                return handler.submit(name, p)
            script, res = rendered if record is None else (record["body"], Resources(**record["resources"]))
            try:
                self.ep.submit(name, script, res)
            except EndpointError as e:
                raise ValueError(str(e)) from e
            self._record_task(name, p["task"])
        return {"handle": name}

    def _record_task(self, name, task):
        tdir = self.out_root.parent / "tasks"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{name}.txt").write_text(task)

    def status(self, handle) -> Dict[str, Any]:
        if self._is_cp(handle=handle):
            return self.chain_point.status(handle)
        d = self._delegate(handle=handle)
        if d is not None:
            return d.status(handle)
        s = self.ep.status(handle)
        return {"state": s.state, "message": f"{s.site_state} {s.message}".strip(),
                "poll_ms": POLL_MS if s.state == "working" else 0, "progress": None}

    def results(self, handle) -> Dict[str, Any]:
        result = self._results(handle)
        record = self._request(handle)
        if record is not None:
            result["metrics"], unavailable = capabilities.contract_results(
                result["metrics"], record["requested_metrics"], record["capabilities"])
            result["metadata"]["unavailable_metrics"] = unavailable
            result["metadata"].update(requested_metrics=record["requested_metrics"],
                                       capabilities=record["capabilities"])
            if record["geometry"] is not None:
                ref = geometry_artifacts.verify_result(self.out_root / handle / "out", record["geometry"])
                result["metadata"]["diagnostic_files"] = result["files"]
                result["files"] = [ref]
                result["metadata"]["geometry"] = geometry_artifacts.identity(record["geometry"])
        return result

    def _results(self, handle) -> Dict[str, Any]:
        if self._is_cp(handle=handle):
            return self.chain_point.results(handle)
        d = self._delegate(handle=handle)
        if d is not None:
            return d.results(handle)
        s = self.ep.status(handle)
        if s.state != "completed":
            raise ValueError(f"fastsimkit: {handle!r} is {s.state}, not completed")
        files = self.ep.fetch(handle, self.out_root)
        mpath = self.out_root / handle / "out" / "metrics.json"
        if not mpath.exists():
            raise ValueError(f"fastsimkit: {handle!r} wrote no out/metrics.json")
        if self._task_of(handle) in TABLE_TASKS and not (self.out_root / handle / "out" / "cleanup.txt").exists():
            raise ValueError(f"fastsimkit: {handle!r} staged Mu2e-internal tables but wrote no cleanup record")
        raw_metrics = json.loads(mpath.read_text())
        if self._task_of(handle) == "tracker_point":
            _tracker_task.validate_metrics(raw_metrics)
            if not (mpath.parent / "provenance.json").is_file():
                raise ValueError("fastsimkit: tracker_point wrote no provenance.json")
        metrics = {k: float(v) for k, v in raw_metrics.items()}
        if hasattr(self.ep, "after_results"):      # EAF: launch the next queued point, or stop an idle server
            self.ep.after_results(handle)
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state}}

    def cancel(self, handle) -> Dict[str, Any]:
        if self._is_cp(handle=handle):
            return self.chain_point.cancel(handle)
        d = self._delegate(handle=handle)
        if d is not None:
            return d.cancel(handle)
        return {"state": self.ep.cancel(handle)}

    @staticmethod
    def describe() -> Dict[str, Any]:
        from . import chainpoint
        params = sorted({"task", *[p for t in TASKS.values() for p in t.params], *chainpoint.PARAMS})
        metrics = sorted({m for t in TASKS.values() for m in t.metrics} | set(chainpoint.METRICS))
        tasks = {}
        for name, task in TASKS.items():
            routes = ["eaf"] if name in FNAL_TASKS else ["eaf", "uva"]
            if name in DELEGATES:
                routes = ["eaf", "uva"]
            tasks[name] = {"params": {k: v.describe() for k, v in schema(name, task.params).items()},
                           "defaults": dict(task.defaults), "required": sorted(set(task.params) - set(task.defaults)),
                           "endpoints": routes, "file_kinds": [], "inputs_supported": False,
                           "validation_scope": "admission only; not physics certification"}
        chain_roles = {"shards": "execution", "files_per_job": "fidelity", "shard_offset": "fidelity",
                       "max_events": "fidelity", "seed": "stochastic_stream", "wall_minutes": "execution",
                       "gen_set": "scenario", "run_config": "scenario", "chain_flags": "numerical",
                       "tier1b": "scenario", "gen_events": "fidelity"}
        tasks["chain_point"] = {
            "params": {k: {"type": "integer", "minimum": lo, "maximum": hi,
                           "unit": "min" if k == "wall_minutes" else "1", "role": chain_roles[k]}
                       for k, (lo, hi) in chainpoint.LIMITS.items()},
            "defaults": dict(chainpoint.OPTIONAL),
            "required": sorted(set(chainpoint.PARAMS) - set(chainpoint.OPTIONAL)),
            "endpoints": ["uva"], "file_kinds": [], "inputs_supported": False,
            "validation_scope": "admission only; not physics certification"}
        for name, extra in capabilities.describe_tasks(TASKS, chainpoint).items():
            tasks[name].update(extra)
            if name == "beam_point":
                tasks[name].update(inputs_supported=True, geometry_contract={
                    "format": geometry_artifacts.FORMAT, "schema_version": 1,
                    "base_include": geometry_artifacts.BASE, "max_bytes": geometry_artifacts.MAX_BYTES,
                    "geom_parameter": "required without FileRef; mutually exclusive with FileRef"})
        metrics = sorted(set(metrics) | {m for spec in tasks.values() for m in spec["metrics"]})
        return {"params": params, "metrics": metrics, "accepts_lists": False,
                "parameter_schema_version": 1, "capabilities_schema_version": 1,
                "requested_metrics_supported": True, "tasks": tasks}


def make_server(kit: FastsimKit):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="fastsimkit", version=VERSION)

    @server.tool(structured_output=True)
    def submit(name: str, params: dict[str, Any], files: list[dict[str, Any]],
               inputs: list[dict[str, Any]], requested_metrics: list[str] | None = None) -> dict[str, Any]:
        return kit.submit(name, params, files, inputs, requested_metrics)

    @server.tool(structured_output=True)
    def status(handle: str) -> dict[str, Any]:
        return kit.status(handle)

    @server.tool(structured_output=True)
    def results(handle: str) -> dict[str, Any]:
        return kit.results(handle)

    @server.tool(structured_output=True)
    def check(name: str, params: dict[str, Any], files: list[dict[str, Any]],
              inputs: list[dict[str, Any]], requested_metrics: list[str] | None = None) -> dict[str, Any]:
        return kit.check(name, params, files, inputs, requested_metrics)

    @server.tool(structured_output=True)
    def describe() -> dict[str, Any]:
        return kit.describe()

    @server.tool(structured_output=True)
    def cancel(handle: str) -> dict[str, Any]:
        return kit.cancel(handle)

    return server


def main() -> int:
    make_server(FastsimKit()).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
