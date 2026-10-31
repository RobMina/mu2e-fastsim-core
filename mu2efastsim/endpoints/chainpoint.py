"""chain_point: one Mu2eBO point of the whole fast CRV chain as N independent one-GPU jobs (fastsimkit task).

A whole node waits hours in the UVA queue while one-GPU jobs start in minutes, so a point fans out into N sub-jobs
(``<name>.j<k>``), each running ``runchain`` on its own gen files. The Mu2e-internal inputs (gen tables, tier-1b library,
CRVCalib/CRVPhoton CSVs, GDML) are never stored on the endpoint: when a sub-job starts it writes ``<job dir>/inbox``
(its node and tmpfs dir), and the kit pipes each file from the FNAL host through the kit host into that tmpfs
(``ssh <fnal> cat`` -> ``srun --overlap ... cat >``), checks its md5 and touches ``ready``. The tables come from CVMFS.
The job's EXIT trap deletes the tmpfs dir and writes ``out/cleanup.txt``; ``results`` refuses a sub-job without it.
Streaming runs in background threads of the kit process (or step by step from ``status`` when ``background=False``),
so the kit host must stay up and on the endpoint's network for the point's lifetime.

Params: shards (sub-jobs, 1..32), files_per_job (gen files each, 1..25), shard_offset (first gen file in the set),
max_events (per file, 0 = all), seed, wall_minutes (per sub-job), chain_flags (optional bitmask: 1 --prefetch, 2
--fused-response), gen_set (0 g4ref_v1, 1 crygen_v1, 2 crygen_run1a_v1,
3 crygen_run2_v1, 4 native CPU CRY, 5 device JAX CRY), gen_events (native accepted events per sub-job, 1..2**20-1), run_config (optional, 0 extracted_v04 = default, 1 run1a_v01, 2 run1b_v01, 3 run2_v10: crvfastsim
runconfig; it picks the GDML world, the lookup tables and the response/cuts; other configurations than extracted run
with nominal CRVCalib constants, no CSVs streamed). A gen set belongs to one CRY configuration: extracted gen sets
(0, 1) only with run_config 0, the default-CRY sets (2, 3) only with the others. tier1b (optional; crvfastsim
tier1b_assets): 0 = the configuration's default asset set (extracted: strict_v4, the strict library with its topology
and v3 template sidecars, placement options, chord placement and hard-muon draws; named configurations: legacy, they
have no library of their own), 1 = legacy (tier1b_v7 library, no placement; the default before 2026-10-28), 2 =
strict_v2 (v2 templates, no chord placement; the extracted default 2026-10-28 to 2026-10-29, so records with tier1b 0
from those days ran it), 3 = strict_v3 (strict_v4 without the hard-muon draws; the extracted default 2026-10-29 to
2026-10-30). Records written before this parameter existed ran legacy. The set's files are streamed once per node into the shared library dir.

Generated gen_sets 4/5 require run_config=0, files_per_job=1, shard_offset=max_events=0 and no prefetch flag.
It removes gen-table streaming only; geometry, conditions, tier-1b and optics are still needed. Fresh sources use
seed+1000*k and subrun k+1 within each point; seeds across separate points are the caller's responsibility.

Input wait: a sub-job waits for its inputs until a deadline sized from the bytes the whole point streams (every sub-job
can queue behind the others on the kit host's link): ``wait_budget_s`` = 600 s + point bytes / FASTSIMKIT_STREAM_MBPS
(default 4 MB/s; measured 5-6.6 MB/s aggregate in kitshm02/kitcg03), at least 1800 s. ``check`` and ``submit`` refuse a point whose
wait budget plus the estimated chain time (``chain_estimate_s``) does not fit in wall_minutes. Metrics: accepted and generated events, accepted per
GPU-hour (charged and chain-only), accepted per estimated SU, campaign time, input wait, clusters per generated event
per sector type.
"""
from __future__ import annotations

import json
from dataclasses import asdict
import math
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import EndpointError, Resources
from mu2efastsim import site
from .slurm_ssh import su_estimate
from . import artifacts
from .snapshots import prepare_code
from .submission import submission_lock

PARAMS = ("shards", "files_per_job", "shard_offset", "max_events", "seed", "wall_minutes", "gen_set", "run_config",
          "chain_flags", "tier1b", "gen_events")
OPTIONAL = {"gen_set": 0, "run_config": 0, "chain_flags": 0, "tier1b": 0, "gen_events": 0}
# chain_flags bits -> runchain options (speed only; 1 is bit-identical, 2 statistically equal, V4)
CHAIN_FLAGS = {1: "--prefetch", 2: "--fused-response"}
METRICS = ("accepted_events", "generated_events", "jobs_ok", "gpu_hours", "est_su", "accepted_per_gpu_hour",
           "accepted_per_est_su", "accepted_per_gpu_hour_chain", "input_wait_s_mean", "campaign_s",
           "clusters_per_generated_EX", "clusters_per_generated_T1", "clusters_per_generated_T2",
           "clusters_per_generated_M", "library_reused_jobs")
SECTOR = {1: "EX", 2: "T1", 3: "T2", 4: "M"}
# gen sets (CRY primaries of every generated event): number -> (file pattern, subruns that exist); datasets registry
CRYGEN_V1_MISSING = (2011, 2028, 2031, 2033, 2034, 2037)    # held at the 3 h grid lifetime and removed, not rerun
GEN_SETS = {0: ("g4ref_v1_tables/gen.001400_{:08d}.root", tuple(range(1, 21))),     # 20 x 50 k, with a G4 reference
            1: ("crygen_v1_tables/gen.001400_{:08d}.root",                          # 94 x 1 M, CRY only (no G4 reference)
                tuple(s for s in range(2001, 2101) if s not in CRYGEN_V1_MISSING)),
            2: ("crygen_run1a_v1_tables/gen.001400_{:08d}.root", tuple(range(3001, 3021))),   # default CRY, Run 1A world
            3: ("crygen_run2_v1_tables/gen.001400_{:08d}.root", tuple(range(4001, 4021)))}    # default CRY, Run 2 world
# sizes for the input-wait budget (bytes, 2026-10-23, FNAL storage): per gen file and generated events per file, per set
GEN_FILE = {0: (2.82e6, 50_000), 1: (5.60e7, 1_000_000), 2: (5.67e7, 1_000_000), 3: (5.66e7, 1_000_000)}
SMALL_BYTES = 3.9e6                                         # CSVs + GDML
WAIT_FLOOR_S, WAIT_SLACK_S, STREAM_MBPS = 1800, 600, 4.0
# chain time per sub-job on one A100 (kitcg03: ~470 s for 4 M generated events, 22 s setup): setup + per M generated,
# with a 1.5x margin on the per-event part
CHAIN_SETUP_S, CHAIN_S_PER_MEVENT = 60.0, 180.0
# tier-1b asset sets (crvfastsim.tier1b_assets.SETS; crv tests keep the two equal): directory under the FNAL data root,
# files (library first), bytes of the whole set
_V3_FILES = ("library.root", "topology_sidecar_strict_v2_v2.npz", "topology_sidecar_strict_v2_v2.npz.json",
             "templates_v3.npz", "templates_v3.npz.json")
TIER1B_SETS = {"strict_v4": ("tier1b_strict_v2", _V3_FILES, 1.1699e9),      # strict_v3's files, hard-muon draws on
               "strict_v3": ("tier1b_strict_v2", ("library.root", "topology_sidecar_strict_v2_v2.npz",
                                                  "topology_sidecar_strict_v2_v2.npz.json", "templates_v3.npz",
                                                  "templates_v3.npz.json"), 1.1699e9),
               "strict_v2": ("tier1b_strict_v2", ("library.root", "topology_sidecar_strict_v2_v2.npz",
                                                  "topology_sidecar_strict_v2_v2.npz.json", "templates_v2.npz",
                                                  "templates_v2.npz.json"), 1.0834e9),
               "legacy": ("tier1b_v7", ("lib_prod_as_20files.root",), 1.007e9)}
LIBRARY = "tier1b_v7/lib_prod_as_20files.root"              # the legacy library
CALIB = "geometry/crvcalib_sim_best_v1_5_run1400.csv"
DEVIATION = "geometry/crvphoton_sim_best_v1_5_run1400.csv"
GDML = "geometry/mu2e_extracted_v04.gdml"
# run configurations (crvfastsim.runconfig): name, GDML under the FNAL data root, lookup tables, extracted CRY?
RUN_CONFIGS = {0: ("extracted_v04", GDML, ("6000_0", "2370_1", "3200_0"), True),
               1: ("run1a_v01", "geometry/mu2e_run1a_v01.gdml", ("6000_0",), False),
               2: ("run1b_v01", "geometry/mu2e_run1a_v01.gdml", ("6000_0",), False),
               3: ("run2_v10", "geometry/mu2e_run2_v10.gdml",
                   ("4550_0", "1045_1", "3040_0", "3200_0", "6000_1", "6000_0", "5000_1", "6900_1", "5700_0", "2370_1",
                    "2100_1", "1550_1"), False)}
NATIVE_GEN_SET = 4
DEVICE_GEN_SET = 5
GENERATED_SETS = (NATIVE_GEN_SET, DEVICE_GEN_SET)
EXTRACTED_GEN_SETS = (0, 1)
# Conservative admission heuristic includes CPU generation; not a runtime prediction or throughput claim.
NATIVE_GENERATION_S_PER_MEVENT = 900.0
LIMITS = {"shards": (1, 32), "files_per_job": (1, 25), "shard_offset": (0, 99), "max_events": (0, 10 ** 7),
          "seed": (0, 2 ** 31 - 1), "wall_minutes": (10, 180), "gen_set": (0, DEVICE_GEN_SET),
          "run_config": (0, max(RUN_CONFIGS)), "chain_flags": (0, sum(CHAIN_FLAGS)), "tier1b": (0, 3),
          "gen_events": (0, 2**20 - 1)}

BODY = """@common@@tables@CODE=@code@
C() { apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},CRVFS_LOOKUP_DIR=$L,CRVFS_DATA=$L,CRVFS_OFFLINE_CACHE=$L/offline,PYTHONPATH=$CODE/crv:$CODE/core:$CODE/tracker" -B "$L" -B /dev/shm -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }
C python -c "import crvfastsim.analyses.runchain as r; assert hasattr(r, 'merge')" || { echo "runchain --emit missing in $CODE"; exit 5; }
@native_probe@# library once per node: /dev/shm/crvfs_lib_<user>_<md5>, one holder file per job, deleted by the last holder (flock);
# holders of jobs that left the queue are dropped first, so a killed job cannot keep the copy alive
S=/dev/shm/crvfs_lib_${USER}_@libmd5@; LOCK=/dev/shm/crvfs_lib_${USER}.lock; J=${SLURM_JOB_ID:-0}
lib_hold() { ( flock 9; mkdir -p "$S/holders"; chmod 700 "$S"
  for h in "$S"/holders/*; do [ -e "$h" ] || continue; squeue -h -j "${h##*/}" -o %i 2>/dev/null | grep -q . || rm -f "$h"; done
  touch "$S/holders/$J" ) 9>"$LOCK"; }
lib_release() { ( flock 9; rm -f "$S/holders/$J"; [ -z "$(ls -A "$S/holders" 2>/dev/null)" ] && { rm -rf "$S"; echo "$(date -Is) $(hostname) deleted node library $S" >> "$M2FS_OUT/cleanup.txt"; } ) 9>"$LOCK"; }
trap 'lib_release; rm -rf "$L"; echo "$(date -Is) $(hostname) deleted $L" >> "$M2FS_OUT/cleanup.txt"' EXIT
lib_hold
LIBSTATE=$( [ -f "$S/ready" ] && echo have || echo need )
echo "$(hostname) $L $S $LIBSTATE" > "$(dirname "$M2FS_OUT")/inbox"      # read by the kit, which then streams the inputs in
tw=$SECONDS; while [ $((SECONDS-tw)) -lt @waits@ ]; do [ -f "$L/ready" ] && [ -f "$S/ready" ] && break; sleep 5; done
[ -f "$L/ready" ] && [ -f "$S/ready" ] || { echo "no input arrived within @waits@ s"; exit 3; }
echo "{\\"input_wait_s\\": $((SECONDS-tw)), \\"wait_budget_s\\": @waits@, \\"library\\": \\"$LIBSTATE\\"}" > "$M2FS_OUT/wait.json"
@gen_setup@
C python -m crvfastsim runchain "${GEN[@]}" @genopts@ @tier1b@ @conds@ \\
  --gdml "$L/@gdml@" --jax --seed @seed@ @maxev@ @runcfg@ --emit "$M2FS_OUT/shard.json" --outdir "$M2FS_OUT/chain" > "$M2FS_OUT/chain.txt" 2>&1 || exit 6
echo '{}' > "$M2FS_OUT/metrics.json"
"""


class FnalSource:
    """Where the Mu2e-internal inputs live: read with `ssh <host> cat`, never copied to the kit host's disk.
    ``host`` defaults to $FASTSIMKIT_FNAL_HOST and ``root`` to mu2efastsim.site.crv_data()."""

    def __init__(self, host: str | None = None, root: str | None = None, runner=None):
        self._host, self.root = host, (root or site.crv_data()).rstrip("/")
        self.runner = runner

    @property
    def host(self) -> str:
        return self._host or site.fnal_host()

    @host.setter
    def host(self, value: str) -> None:     # explicit retarget: overrides $FASTSIMKIT_FNAL_HOST for this source
        self._host = value

    def path(self, rel: str) -> str:
        return f"{self.root}/{rel}"

    def argv(self, rel: str) -> List[str]:
        return ["ssh", "-o", "BatchMode=yes", self.host, "cat " + shlex.quote(self.path(rel))]

    def md5(self, rel: str) -> str:
        import subprocess
        p = (self.runner or subprocess.run)(["ssh", "-o", "BatchMode=yes", self.host, "md5sum " + shlex.quote(self.path(rel))],
                                            capture_output=True, timeout=600)
        if p.returncode:
            raise EndpointError(f"chain_point: md5sum of {rel} on {self.host} failed: {p.stderr[-200:]!r}")
        return p.stdout.decode().split()[0]

    def fingerprint(self, rel: str) -> dict:
        import subprocess
        cmd = artifacts.fingerprint_command(self.path(rel))
        p = (self.runner or subprocess.run)(["ssh", "-o", "BatchMode=yes", self.host, cmd],
                                            capture_output=True, timeout=900)
        if p.returncode:
            raise EndpointError(f"artifact fingerprint failed for {rel}: {p.stderr[-200:]!r}")
        return artifacts.parse_fingerprint(p.stdout.decode())


def validate(params: Dict[str, Any]) -> Dict[str, int]:
    params = {**OPTIONAL, **params}                       # optional: older studies name no gen set or run config
    missing = [k for k in PARAMS if k not in params]
    extra = sorted(set(params) - set(PARAMS) - {"task"})
    if missing or extra:
        raise ValueError(f"fastsimkit: task 'chain_point' takes {list(PARAMS)}; missing {missing}, extra {extra}")
    out = {}
    for k in PARAMS:
        v = params[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            raise ValueError(f"fastsimkit: chain_point {k} must be a whole number, got {v!r}")
        lo, hi = LIMITS[k]
        if not lo <= v <= hi:
            raise ValueError(f"fastsimkit: chain_point {k} must be in [{lo}, {hi}], got {v!r}")
        out[k] = int(v)
    if out["gen_set"] in GENERATED_SETS:
        if out["run_config"] != 0:
            raise ValueError("fastsimkit: generated CRY supports only run_config=0 (extracted_v04)")
        if out["files_per_job"] != 1 or out["shard_offset"] != 0 or out["max_events"] != 0:
            raise ValueError("fastsimkit: native CRY needs files_per_job=1, shard_offset=0, max_events=0; use gen_events")
        if not out["gen_events"]:
            raise ValueError("fastsimkit: native CRY requires positive gen_events per sub-job")
        if out["chain_flags"] & 1:
            raise ValueError("fastsimkit: native CRY does not support chain_flags prefetch")
        return out
    if out["gen_events"]:
        raise ValueError("fastsimkit: gen_events requires generated gen_set=4 or 5")
    del out["gen_events"]
    if (out["gen_set"] in EXTRACTED_GEN_SETS) != RUN_CONFIGS[out["run_config"]][3]:
        raise ValueError(f"fastsimkit: chain_point gen_set {out['gen_set']} is not a CRY sample of run_config "
                         f"{out['run_config']} ({RUN_CONFIGS[out['run_config']][0]})")
    n = len(GEN_SETS[out["gen_set"]][1])
    if out["shard_offset"] >= n:
        raise ValueError(f"fastsimkit: chain_point shard_offset must be < {n} for gen_set {out['gen_set']}")
    return out


def tier1b_set(p: Dict[str, int]) -> str:
    """Asset set of a point: for extracted strict_v4 with tier1b 0, strict_v2 with 2 and strict_v3 with 3; legacy
    otherwise (and for records made before the tier1b parameter)."""
    if "tier1b" not in p or p["tier1b"] == 1 or p.get("run_config", 0) != 0:
        return "legacy"
    return {2: "strict_v2", 3: "strict_v3"}.get(p["tier1b"], "strict_v4")


def library_files(p: Dict[str, int]) -> List[str]:
    """The point's tier-1b files relative to the FNAL data root, library first."""
    d, files, _ = TIER1B_SETS[tier1b_set(p)]
    return [f"{d}/{f}" for f in files]


def point_bytes(p: Dict[str, int]) -> float:
    """Bytes the kit streams for the whole point, worst case (every sub-job streams its own library copy)."""
    gen_bytes = 0 if p.get("gen_set", 0) in GENERATED_SETS else p["files_per_job"] * GEN_FILE[p.get("gen_set", 0)][0]
    return p["shards"] * (TIER1B_SETS[tier1b_set(p)][2] + SMALL_BYTES * (1.6 if p.get("run_config", 0) == 3 else 1.0)
                          + gen_bytes)


def small_inputs(p: Dict[str, int]) -> List[str]:
    """Per sub-job inputs besides the library and the gen tables: the CRVCalib/CRVPhoton CSVs (extracted only; other
    configurations run with nominal constants) and the configuration's GDML world."""
    rc = RUN_CONFIGS[p.get("run_config", 0)]
    return ([CALIB, DEVIATION] if rc[3] else []) + [rc[1]]


def wait_budget_s(p: Dict[str, int], mbps: Optional[float] = None) -> int:
    """Job-side input deadline [s]: slack + point bytes at the assumed kit-host link rate, at least WAIT_FLOOR_S."""
    import os
    if mbps is None:
        mbps = os.environ.get("FASTSIMKIT_STREAM_MBPS", STREAM_MBPS)
    try:
        if isinstance(mbps, bool):
            raise ValueError
        mbps = float(mbps)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("fastsimkit: FASTSIMKIT_STREAM_MBPS must be a finite positive number") from None
    if not math.isfinite(mbps) or mbps <= 0:
        raise ValueError("fastsimkit: FASTSIMKIT_STREAM_MBPS must be a finite positive number")
    wait = WAIT_SLACK_S + point_bytes(p) / (mbps * 1e6)
    if not math.isfinite(wait):
        raise ValueError("fastsimkit: FASTSIMKIT_STREAM_MBPS is too small for a finite input wait budget")
    return int(max(WAIT_FLOOR_S, wait))


def chain_estimate_s(p: Dict[str, int]) -> float:
    """Estimated chain time of one sub-job [s] (setup + per million generated events, with margin)."""
    if p.get("gen_set", 0) in GENERATED_SETS:
        return CHAIN_SETUP_S + (CHAIN_S_PER_MEVENT + NATIVE_GENERATION_S_PER_MEVENT) * p["gen_events"] / 1e6
    per_file = GEN_FILE[p.get("gen_set", 0)][1]
    if p["max_events"]:
        per_file = min(per_file, p["max_events"])
    return CHAIN_SETUP_S + CHAIN_S_PER_MEVENT * p["files_per_job"] * per_file / 1e6


def gen_files(p: Dict[str, int], k: int) -> List[str]:
    """Gen tables of sub-job k: files_per_job consecutive shards starting at shard_offset + k * files_per_job (cycled)."""
    if p.get("gen_set", 0) in GENERATED_SETS:
        return []
    f, o = p["files_per_job"], p["shard_offset"]
    pat, subruns = GEN_SETS[p.get("gen_set", 0)]
    return [pat.format(subruns[(o + k * f + i) % len(subruns)]) for i in range(f)]


class ChainPoint:
    def __init__(self, kit, source: Optional[FnalSource] = None, code: Optional[str] = None, code_ref: str = "main",
                 background: bool = True, poll_s: float = 20.0, streams: int = 2):
        import os
        self.kit, self.ep = kit, kit.ep
        self.source = source or FnalSource(site.fnal_host(), site.crv_data())
        self.code = code or site.setting("FASTSIMKIT_CODE") or site.scratch() + "/chain_code_kit"
        self.code_ref = os.environ.get("FASTSIMKIT_CODE_REF", code_ref)
        self.background, self.poll_s = background, poll_s
        # parallel file streams per sub-job. Each stream is an ssh channel to the FNAL host and one to the endpoint,
        # and both ride one ControlMaster connection per host, which sshd caps at MaxSessions (10): 4 streams x 2
        # sub-jobs + polling failed with ssh rc 255 (kitcp02). Keep streams x concurrent sub-jobs well under 10.
        self.streams = int(os.environ.get("FASTSIMKIT_STREAMS", streams))
        self._chan = threading.BoundedSemaphore(int(os.environ.get("FASTSIMKIT_MAX_CHANNELS", 6)))   # across sub-jobs
        self.dir = Path(kit.out_root).parent / "chain_point"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._threads: Dict[str, threading.Thread] = {}
        self._node_lib: Dict[str, threading.Lock] = {}     # one library stream per node at a time
        self._lock = threading.Lock()

    # -- records -------------------------------------------------------------------------------------------------
    def _rec_path(self, name):
        return self.dir / f"{name}.json"

    def record(self, name) -> Optional[dict]:
        p = self._rec_path(name)
        return artifacts.checked_record(p, allow_legacy=True)

    def subs(self, name) -> List[str]:
        r = self.record(name)
        return [f"{name}.j{k}" for k in range(r["params"]["shards"])] if r else []

    def _mark(self, sub, what, text=""):
        (self.dir / f"{sub}.{what}").write_text(text)

    def _has(self, sub, what) -> bool:
        return (self.dir / f"{sub}.{what}").exists()

    # -- rendering -----------------------------------------------------------------------------------------------
    def _require_streaming(self):
        if not hasattr(self.ep, "stream_in"):
            raise ValueError(f"fastsimkit: chain_point streams its inputs into the job and needs an endpoint that can "
                             f"(slurm_ssh); {getattr(self.ep, 'name', '?')!r} cannot")

    def render(self, p: Dict[str, int], k: int, manifest=None):
        from .kit import _COMMON, _TABLES, APPTAINER_RUNTIME
        self._require_streaming()
        sif = shlex.quote(manifest["image"]["path"] if manifest else self.kit.sif)
        rc = RUN_CONFIGS[p.get("run_config", 0)]
        tables = _TABLES.replace("@tables@", " ".join(f"LookupTable_{t}" for t in rc[2]) + " photonMap.root")
        body = BODY.replace("@common@", _COMMON).replace("@tables@", tables)
        body = body.replace("@runtime_a@", APPTAINER_RUNTIME["a"]).replace("@runtime_t@", APPTAINER_RUNTIME["t"])
        conds = (f'--calib "$L/{Path(CALIB).name}" --deviation "$L/{Path(DEVIATION).name}"' if rc[3]
                 else "--calib nominal")
        lib = library_files(p)
        t1b = (f'--library "$S/{Path(lib[0]).name}"' if tier1b_set(p) == "legacy"
               else f'--tier1b {tier1b_set(p)} --tier1b-dir "$S"')
        if manifest:
            shas = [next(f["sha256"] for f in manifest["inputs"]["files"] if f["src"] == src) for src in lib]
            libkey = shas[0] if len(shas) == 1 else artifacts.digest(shas)   # legacy keeps its node cache name
        else:
            libkey = "preview"
        native = p.get("gen_set", 0) in GENERATED_SETS
        backend = "device" if p.get("gen_set") == DEVICE_GEN_SET else "native"
        native_seed, native_subrun = p["seed"] + 1000 * k, k + 1
        subs = {"gen_setup": 'GEN=(native)' if native else 'GEN=($(ls "$L"/gen.*.root))',
                "genopts": (f'--gen {backend} --gen-events {p["gen_events"]} --gen-seed {native_seed} '
                            f'--gen-subrun {native_subrun}' if native else ""),
                "sif": sif, "code": shlex.quote(manifest["code"]["path"] if manifest else self.code),
                "tier1b": t1b, "conds": conds,
                "gdml": Path(rc[1]).name, "seed": str(p["seed"] + 1000 * k),
                "native_probe": ((
                    'C python -c "import jax; from mu2efastsim.cry_device import device_tables; '
                    'assert jax.default_backend() == \'gpu\'; device_tables()" || exit 5\n'
                    if backend == "device" else
                    'C python -c "from mu2efastsim.cry_native import CrySource; s=CrySource(seed=0); s.close()" || exit 5\n')
                    if native else ""),
                "runcfg": " ".join(([] if p.get("run_config", 0) == 0 else [f"--run-config {rc[0]}"])
                                   + [o for b, o in CHAIN_FLAGS.items() if p.get("chain_flags", 0) & b]),
                "maxev": f"--max-events {p['max_events']}" if p["max_events"] else "",
                "libmd5": libkey, "waits": str(wait_budget_s(p))}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        import re
        if re.search(r"@(sif|code|tier1b|conds|gdml|seed|maxev|runcfg|libmd5|waits|common|tables|runtime_a|runtime_t|gen_setup|genopts|native_probe)@", body):
            raise ValueError("fastsimkit: chain_point: unfilled placeholder")
        if manifest is not None:
            files = {f["src"]: f for f in manifest["inputs"]["files"]}
            guard = artifacts.consumption_guard('"$S"', [files[src] for src in lib])
            guard += artifacts.consumption_guard('"$L"', [files[src] for src in small_inputs(p) + gen_files(p, k)])
            body = body.replace(subs["gen_setup"], guard + subs["gen_setup"], 1)
        res = Resources(cores=16, mem_gb=110, gpus=1, gpu_type="a100", minutes=p["wall_minutes"])
        return (artifacts.runtime_prefix(manifest) if manifest else "") + body, res

    def lib_md5(self, p: Optional[Dict[str, int]] = None) -> str:
        return self.source.md5(library_files(p or {**OPTIONAL})[0])

    # -- contract ------------------------------------------------------------------------------------------------
    def _admit(self, params) -> Dict[str, int]:
        """Validate the point and its resource budgets without local or remote side effects."""
        p = validate(params)
        self._require_streaming()
        su = p["shards"] * su_estimate(Resources(cores=16, gpus=1, minutes=p["wall_minutes"]),
                                      *getattr(self.ep, "su_rates", (53.0, 0.5)))
        cap = getattr(self.ep, "max_su_per_job", float("inf")) * p["shards"]
        if su > cap:
            raise ValueError(f"fastsimkit: chain_point worst case {su:.0f} SU exceeds {cap:.0f}")
        wait, chain = wait_budget_s(p), chain_estimate_s(p)
        need = wait + chain
        if need > 60 * p["wall_minutes"]:
            raise ValueError(f"fastsimkit: chain_point needs ~{need / 60:.0f} min per sub-job (input wait budget "
                             f"{wait / 60:.0f} min for {point_bytes(p) / 1e9:.1f} GB streamed + chain "
                             f"~{chain / 60:.0f} min) > wall_minutes {p['wall_minutes']}: use fewer "
                             f"files_per_job or shards, or more wall_minutes")
        return p

    def check(self, name, params) -> Dict[str, Any]:
        try:
            p = self._admit(params)
            self.render(p, 0)
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def ensure_code(self):
        return prepare_code(self.ep, self.code, self.code_ref)

    def submit(self, name, params) -> Dict[str, Any]:
        p = self._admit(params)

        def prepare():
            paths = set(small_inputs(p) + library_files(p))
            for k in range(p["shards"]):
                paths.update(gen_files(p, k))
            inputs = artifacts.resolve_inputs(self.source, {src: Path(src).name for src in paths})
            if p.get("gen_set", 0) in GENERATED_SETS:
                from ..cry_receipts import EXTRACTED_V04_GDML_SHA256
                geometry = next(f for f in inputs["files"] if f["src"] == GDML)
                if geometry["sha256"] != EXTRACTED_V04_GDML_SHA256:
                    raise ValueError("native chain_point requires the verified extracted-v04 GDML")
            code = self.ensure_code()
            image = artifacts.resolve_image(self.ep, self.kit.sif)
            manifest = artifacts.make_manifest("chain_point", code, image, inputs, p)
            rendered = [self.render(p, k, manifest) for k in range(p["shards"])]
            return {"submitted_at": time.time(), "code": code["path"], "code_ref": self.code_ref,
                    "manifest": manifest, "body": [b for b, r in rendered],
                    "resources": [asdict(r) for b, r in rendered]}

        rec = artifacts.binding(self.dir, name, p, artifacts.target_identity(self.ep, self.source), prepare)
        for k, sub in enumerate(self.subs(name)):
            try:
                self.ep.submit(sub, rec["body"][k], Resources(**rec["resources"][k]))
            except EndpointError as e:
                raise ValueError(str(e))
        self.ensure_streaming(name)
        return {"handle": name}

    def ensure_streaming(self, name):
        if "manifest" not in (self.record(name) or {}):
            return
        for k, sub in enumerate(self.subs(name)):
            if self._has(sub, "streamed") or self._has(sub, "streamerr"):
                continue
            if not self.background:
                self.stream_step(name, k)
                continue
            with self._lock:
                t = self._threads.get(sub)
                if t is None or not t.is_alive():
                    t = threading.Thread(target=self._stream_loop, args=(name, k), daemon=True)
                    self._threads[sub] = t
                    t.start()

    def _stream_loop(self, name, k):
        sub = self.subs(name)[k]
        while not (self._has(sub, "streamed") or self._has(sub, "streamerr")):
            if self.stream_step(name, k) == "gone":
                return
            time.sleep(self.poll_s)

    def stream_step(self, name, k) -> str:
        """One attempt for sub-job k: 'waiting' (not started yet), 'gone' (left the queue), 'streamed', 'error'."""
        sub = self.subs(name)[k]
        st = self.ep.status(sub)
        if st.state != "working":
            return "gone"
        inbox = self.ep.read_file(sub, "inbox").split()
        if len(inbox) < 2:
            return "waiting"
        host, tmp = inbox[0], inbox[1]
        shared, libstate = (inbox[2], inbox[3]) if len(inbox) >= 4 else (None, "need")
        rec = self.record(name)
        p = rec["params"]
        entries = {f["src"]: f for f in rec["manifest"]["inputs"]["files"]}
        lib = library_files(p)
        from concurrent.futures import ThreadPoolExecutor

        def one(rel):   # the library is the long pole; the small files and gen tables go alongside it
            entry = entries[rel]
            want = entry["md5"]
            with self._chan:
                got = self.ep.stream_in(sub, host, f"{tmp}/{Path(rel).name}", entry["argv"])
            if got != want:
                raise EndpointError(f"md5 mismatch for {rel}: {got} != {want}")
        def library():   # once per node: the first sub-job there streams it, later ones find it (or wait for it)
            if shared is None:
                for rel in lib:
                    one(rel)
                return
            cache_key = artifacts.digest([getattr(self.ep, "host", ""), host, shared])
            with self._node_lib.setdefault(host, threading.Lock()), submission_lock(self.dir / ".cache-writers", cache_key):
                have = self.ep.run_on_node(sub, host, f"test -f {shlex.quote(shared)}/ready && echo have || true").strip()
                if have == "have":
                    if not artifacts.verify_cache(self.ep, sub, host, shared, [entries[src] for src in lib]):
                        raise EndpointError("cached library differs from pinned manifest")
                    self._mark(sub, "libreused", host)
                    return
                for rel in lib:
                    want = entries[rel]["md5"]
                    with self._chan:
                        got = self.ep.stream_in(sub, host, f"{shared}/{Path(rel).name}", entries[rel]["argv"])
                    if got != want:
                        raise EndpointError(f"md5 mismatch for {rel}: {got} != {want}")
                if not artifacts.verify_cache(self.ep, sub, host, shared, [entries[src] for src in lib]):
                    raise EndpointError("streamed library differs from pinned manifest")
                self.ep.run_on_node(sub, host, f"touch {shlex.quote(shared)}/ready")
                self._mark(sub, "libstreamed", host)

        try:
            with submission_lock(self.dir / ".stage-writers", sub):
                used = small_inputs(p) + gen_files(p, k) + (lib if shared is None else [])
                have = self.ep.run_on_node(sub, host, f"test -f {shlex.quote(tmp)}/ready && echo have || true").strip()
                if have == "have":
                    if not artifacts.verify_cache(self.ep, sub, host, tmp, [entries[src] for src in used]):
                        raise EndpointError("ready shard inputs differ from pinned manifest; refusing overwrite")
                    if shared is not None and not artifacts.verify_cache(self.ep, sub, host, shared, [entries[src] for src in lib]):
                        raise EndpointError("ready shared library differs from pinned manifest")
                else:
                    with ThreadPoolExecutor(max_workers=self.streams) as ex:
                        futs = [ex.submit(library)] + [ex.submit(one, rel) for rel in small_inputs(p) + gen_files(p, k)]
                        for f in futs:
                            f.result()
                    if not artifacts.verify_cache(self.ep, sub, host, tmp, [entries[src] for src in used]):
                        raise EndpointError("streamed inputs differ from pinned manifest")
                    self.ep.run_on_node(sub, host, f"touch {shlex.quote(tmp)}/ready")
        except EndpointError as e:
            self._mark(sub, "streamerr", str(e))
            self.ep.cancel(sub)
            return "error"
        self._mark(sub, "streamed", time.strftime("%Y-%m-%dT%H:%M:%S"))
        return "streamed"

    def status(self, name) -> Dict[str, Any]:
        from .kit import POLL_MS
        subs = self.subs(name)
        sts = [self.ep.status(s) for s in subs]
        self.ensure_streaming(name)
        done = sum(s.state != "working" for s in sts)
        ok = sum(s.state == "completed" for s in sts)
        errs = [s for s in subs if self._has(s, "streamerr")]
        prog = {"done": done, "total": len(subs), "ok": ok}
        if done < len(subs):
            run = sum(s.site_state == "RUNNING" for s in sts)
            return {"state": "working", "message": f"{ok}/{len(subs)} done, {run} running", "poll_ms": POLL_MS,
                    "progress": prog}
        if ok == len(subs):
            return {"state": "completed", "message": f"{ok}/{len(subs)} sub-jobs completed", "poll_ms": 0, "progress": prog}
        state = "cancelled" if all(s.state == "cancelled" for s in sts) else "failed"
        bad = [f"{sub}: {s.message}" for sub, s in zip(subs, sts) if s.state != "completed"]
        msg = "; ".join(bad[:4]) + (f"; stream errors: {len(errs)}" if errs else "")
        return {"state": state, "message": msg[:500], "poll_ms": 0, "progress": prog}

    def results(self, name) -> Dict[str, Any]:
        st = self.status(name)
        if st["state"] != "completed":
            raise ValueError(f"fastsimkit: {name!r} is {st['state']}, not completed")
        rec = self.record(name)
        acc = gen = 0
        chain_s = gpu_s = wait = su = 0.0
        cl: Dict[int, int] = {}
        files, ends, ids = [], [], []
        reused = 0
        native_shards = []
        for index, sub in enumerate(self.subs(name)):
            fs = self.ep.fetch(sub, self.kit.out_root)
            out = Path(self.kit.out_root) / sub / "out"
            if not (out / "cleanup.txt").exists():
                raise ValueError(f"fastsimkit: {sub!r} staged Mu2e-internal inputs but wrote no cleanup record")
            if "manifest" in rec:
                artifacts.verify_result(out, rec["manifest"])
            s = json.loads((out / "shard.json").read_text())
            if rec["params"].get("gen_set", 0) in GENERATED_SETS:
                from ..cry_receipts import merge_receipts, EXTRACTED_V04_GDML_SHA256
                if "native_generator" not in s:
                    raise ValueError("native shard has no generator receipt")
                merge_receipts([s])
                receipt = s["native_generator"]
                expected_range = {"run": 1400, "subrun": index + 1, "first_event": 1,
                                  "last_event": rec["params"]["gen_events"]}
                expected_generator = "jax-cry-1.7" if rec["params"]["gen_set"] == DEVICE_GEN_SET else "native-cry-1.7"
                if (receipt.get("generator") != expected_generator or receipt.get("seed") != rec["params"]["seed"] + 1000 * index
                        or receipt.get("event_range") != expected_range
                        or receipt.get("cosmic_config", {}).get("geometry_sha256") != EXTRACTED_V04_GDML_SHA256):
                    raise ValueError("native shard receipt differs from bound request")
                native_shards.append(s)
            elif "native_generator" in s:
                raise ValueError("table chain_point returned a native generator receipt")
            T = s["timers"]
            acc += int(T.get("events_with_steps", 0))
            gen += int(s["generated_events"])
            chain_s += float(s["wall_s"])
            for k, v in s.get("cl_type_counts", {}).items():
                cl[int(k)] = cl.get(int(k), 0) + int(v)
            w = out / "wait.json"
            wj = json.loads(w.read_text()) if w.exists() else {}
            wait += wj.get("input_wait_s", 0)
            reused += self._has(sub, "libreused")
            t = self.ep.times(sub)
            gpu_s += t["elapsed_s"]
            ends.append(t["end"])
            su += su_estimate(Resources(cores=16, gpus=1, minutes=t["elapsed_s"] / 60.0))
            ids.append(t.get("job_id", ""))
            files += [x for x in fs if x.suffix in (".json", ".txt", ".out")]
        n = len(self.subs(name))
        m = {"accepted_events": acc, "generated_events": gen, "jobs_ok": n, "gpu_hours": gpu_s / 3600.0, "est_su": su,
             "accepted_per_gpu_hour": 3600.0 * acc / gpu_s if gpu_s else 0.0,
             "accepted_per_est_su": acc / su if su else 0.0,
             "accepted_per_gpu_hour_chain": 3600.0 * acc / chain_s if chain_s else 0.0,
             "input_wait_s_mean": wait / n, "campaign_s": max(ends) - rec["submitted_at"] if ends else 0.0,
             "library_reused_jobs": reused}
        if rec["params"].get("run_config", 0) == 0:
            for k, lab in SECTOR.items():
                m[f"clusters_per_generated_{lab}"] = cl.get(k, 0) / gen if gen else 0.0
        else:   # sector types of the configuration (counters file), by number; the extracted names do not apply
            for lab in SECTOR.values():
                m[f"clusters_per_generated_{lab}"] = float("nan")
            for k, v in sorted(cl.items()):
                m[f"clusters_per_generated_type{k}"] = v / gen if gen else 0.0
        generator_metadata = {}
        if native_shards:
            from ..cry_receipts import merge_receipts
            generator_metadata = merge_receipts(native_shards)
            live = generator_metadata["live_seconds"]
            generator_metadata["clusters_per_live_second"] = {SECTOR.get(k, str(k)): v / live for k, v in cl.items()}
            generator_metadata["generation_s"] = sum(s["timers"].get("generation_s", 0.0) for s in native_shards)
        return {"metrics": {k: float(v) for k, v in m.items()},
                "files": [{"name": str(p.relative_to(self.kit.out_root)), "uri": p.resolve().as_uri(), "kind": "text"}
                          for p in files],
                "metadata": {"endpoint": self.ep.name, "job_ids": ",".join(ids), "code_ref": rec["code_ref"],
                             **({"native_generation": generator_metadata} if generator_metadata else {}),
                             "artifact_manifest": rec.get("manifest"), "manifest_sha": rec.get("manifest", {}).get("manifest_sha"),
                             "artifact_binding": "verified" if "manifest" in rec else "legacy-unverified",
                             "su_note": "est_su from Slurm elapsed time and the measured UVA rates; MAM posts the charge later"}}

    def cancel(self, name) -> Dict[str, Any]:
        states = [self.ep.cancel(s) for s in self.subs(name)]
        return {"state": "cancelled" if any(s == "cancelled" for s in states) else (states[0] if states else "cancelled")}
