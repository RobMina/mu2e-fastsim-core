"""fastsimkit: the Mu2eBO evaluator-contract kit for the fast chain on a compute endpoint.

Contract (Mu2eBO core/contract.py): submit(name, params, files, inputs) -> {"handle": name};
status(handle) -> {"state", "message", "poll_ms", "progress"}; results(handle) -> {"metrics", "files", "metadata"};
check, describe, cancel. One BO point = one job: the kit renders a job body from a task template and the point's
params, submits it through an endpoint (``mu2efastsim.endpoints``), and on completion fetches ``out/`` and reads
``out/metrics.json`` (a flat {name: number} map written by the task). Tasks are templates in TASKS; a task names its
resources, the params it accepts and the metrics it returns.

Run as an MCP stdio server (``python -m mu2efastsim.endpoints.kit``) for a kits.toml entry, or use FastsimKit directly.
Environment: FASTSIMKIT_ENDPOINT (default uva; eaf), FASTSIMKIT_STATE_DIR (local job records and fetched outputs),
FASTSIMKIT_MAX_SU_PER_JOB (default 500), FASTSIMKIT_SIF (image path on the site).
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import EndpointError, Resources, get

VERSION = "1"
POLL_MS = 60_000


@dataclass(frozen=True)
class Task:
    params: Tuple[str, ...]
    metrics: Tuple[str, ...]
    resources: Resources
    body: str           # bash; @sif@ and @p_<param>@ are substituted (shell-quoted); writes $M2FS_OUT/metrics.json


_COMMON = """module load apptainer >/dev/null 2>&1
M2FS_OUT=$(readlink -f "$M2FS_OUT")   # /scratch is a symlink on UVA: bind the real path
@runtime_a@echo "host $(hostname) job ${SLURM_JOB_ID:-none} out $M2FS_OUT"; nvidia-smi -L
A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != 'gpu')" || { echo "JAX not on the GPU"; exit 5; }
"""

# Copy-compute-delete (E9): the CRVConditions tables are Mu2e-internal. They are fetched per job from the CVMFS
# Stratum-1 into node tmpfs (mode 700), used through CRVFS_LOOKUP_DIR and deleted by an EXIT trap; the deletion is
# recorded in out/cleanup.txt (the kit refuses results without it). Never on shared or scratch storage.
_TABLES = """L=${TMPDIR:-/tmp}/crvfs_kit_$$; mkdir -p "$L"; chmod 700 "$L"
trap 'rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
trap 'exit 143' TERM INT
find ${TMPDIR:-/tmp} -maxdepth 1 -name 'crvfs_*' -user "$USER" ! -path "$L" -exec rm -rf {} + 2>/dev/null
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
# A/T definitions instead (EAF: a venv on /exp, since EAF starts its own images)
APPTAINER_RUNTIME = {"a": 'A() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} -B "$M2FS_OUT" --pwd /opt/mu2e-fastsim @sif@ "$@"; }\n',
                     "t": 'T() { apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},CRVFS_LOOKUP_DIR=$L -B "$L" -B "$M2FS_OUT" --pwd /opt/mu2e-fastsim @sif@ "$@"; }'}

TABLE_TASKS = {"response_bench", "v4_response"}   # tasks that stage Mu2e-internal tables: cleanup record required


def _arg(v) -> str:
    """A param as a command-line word; Mu2eBO sends every "number" as a float, so 200000.0 becomes 200000."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


class FastsimKit:
    def __init__(self, endpoint=None, state_dir: str | Path | None = None, sif: str | None = None):
        root = Path(state_dir or os.environ.get("FASTSIMKIT_STATE_DIR", "~/.mu2efastsim/kit")).expanduser()
        self.out_root = root / "out"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.sif = sif or os.environ.get("FASTSIMKIT_SIF", "/scratch/ram2aq/mu2efastsim/mu2efastsim.sif")
        self.ep = endpoint or get(os.environ.get("FASTSIMKIT_ENDPOINT", "uva"), state_dir=root / "jobs",
                                  max_su_per_job=float(os.environ.get("FASTSIMKIT_MAX_SU_PER_JOB", "500")))

    @staticmethod
    def _task(params: Dict[str, Any]) -> Task:
        name = params.get("task")
        if name not in TASKS:
            raise ValueError(f"fastsimkit: unknown task {name!r}; choose from {sorted(TASKS)}")
        t = TASKS[name]
        extra = set(params) - set(t.params) - {"task"}
        missing = set(t.params) - set(params)
        if extra or missing:
            raise ValueError(f"fastsimkit: task {name!r} takes {list(t.params)}; extra {sorted(extra)}, "
                             f"missing {sorted(missing)}")
        return t

    def render(self, params: Dict[str, Any]) -> Tuple[str, Resources]:
        t = self._task(params)
        rt = self.ep.runtime() if hasattr(self.ep, "runtime") else APPTAINER_RUNTIME
        body = t.body.replace("@runtime_a@", rt["a"]).replace("@runtime_t@", rt["t"])
        subs = {"sif": shlex.quote(self.sif), **{f"p_{k}": shlex.quote(_arg(params[k])) for k in t.params}}
        for k, v in subs.items():
            body = body.replace(f"@{k}@", v)
        if any(s in body for s in ("@p_", "@sif@", "@tables@", "@metrics_py@", "@runtime_")):
            raise ValueError(f"fastsimkit: unfilled placeholder in task {params['task']!r}")
        return body, t.resources

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params, files, inputs) -> Dict[str, Any]:
        try:
            self.render(params)
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def _task_of(self, handle) -> str:
        p = self.out_root.parent / "tasks" / f"{handle}.txt"
        return p.read_text().strip() if p.exists() else ""

    def submit(self, name, params, files, inputs) -> Dict[str, Any]:
        if files or inputs:
            raise ValueError("fastsimkit: files and inputs are not supported yet (tasks take params only)")
        script, res = self.render(params)
        try:
            self.ep.submit(name, script, res)
        except EndpointError as e:
            raise ValueError(str(e))
        tdir = self.out_root.parent / "tasks"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{name}.txt").write_text(params["task"])
        return {"handle": name}

    def status(self, handle) -> Dict[str, Any]:
        s = self.ep.status(handle)
        return {"state": s.state, "message": f"{s.site_state} {s.message}".strip(),
                "poll_ms": POLL_MS if s.state == "working" else 0, "progress": None}

    def results(self, handle) -> Dict[str, Any]:
        s = self.ep.status(handle)
        if s.state != "completed":
            raise ValueError(f"fastsimkit: {handle!r} is {s.state}, not completed")
        files = self.ep.fetch(handle, self.out_root)
        mpath = self.out_root / handle / "out" / "metrics.json"
        if not mpath.exists():
            raise ValueError(f"fastsimkit: {handle!r} wrote no out/metrics.json")
        if self._task_of(handle) in TABLE_TASKS and not (self.out_root / handle / "out" / "cleanup.txt").exists():
            raise ValueError(f"fastsimkit: {handle!r} staged Mu2e-internal tables but wrote no cleanup record")
        metrics = {k: float(v) for k, v in json.loads(mpath.read_text()).items()}
        return {"metrics": metrics,
                "files": [{"name": p.name, "uri": p.resolve().as_uri(), "kind": "text"} for p in files
                          if p.suffix in (".json", ".txt", ".out", ".log")],
                "metadata": {"endpoint": self.ep.name, "job_id": s.job_id, "site_state": s.site_state}}

    def cancel(self, handle) -> Dict[str, Any]:
        return {"state": self.ep.cancel(handle)}

    @staticmethod
    def describe() -> Dict[str, Any]:
        params = sorted({"task", *[p for t in TASKS.values() for p in t.params]})
        metrics = sorted({m for t in TASKS.values() for m in t.metrics})
        return {"params": params, "metrics": metrics, "accepts_lists": False}


def make_server(kit: FastsimKit):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="fastsimkit", version=VERSION)

    @server.tool(structured_output=True)
    def submit(name: str, params: dict[str, Any], files: list[dict[str, Any]],
               inputs: list[dict[str, Any]]) -> dict[str, Any]:
        return kit.submit(name, params, files, inputs)

    @server.tool(structured_output=True)
    def status(handle: str) -> dict[str, Any]:
        return kit.status(handle)

    @server.tool(structured_output=True)
    def results(handle: str) -> dict[str, Any]:
        return kit.results(handle)

    @server.tool(structured_output=True)
    def check(name: str, params: dict[str, Any], files: list[dict[str, Any]],
              inputs: list[dict[str, Any]]) -> dict[str, Any]:
        return kit.check(name, params, files, inputs)

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
