"""chain_point: one Mu2eBO point of the whole fast CRV chain as N independent one-GPU jobs (fastsimkit task).

A whole node waits hours in the UVA queue while one-GPU jobs start in minutes, so a point fans out into N sub-jobs
(``<name>.j<k>``), each running ``runchain`` on its own gen files. The Mu2e-internal inputs (gen tables, tier-1b library,
CRVCalib/CRVPhoton CSVs, GDML) are never stored on the endpoint: when a sub-job starts it writes ``<job dir>/inbox``
(its node and tmpfs dir), and the kit pipes each file from the FNAL host through the kit host into that tmpfs
(``ssh <fnal> cat`` -> ``srun --overlap ... cat >``), checks its md5 and touches ``ready``. The tables come from CVMFS.
The job's EXIT trap deletes the tmpfs dir and writes ``out/cleanup.txt``; ``results`` refuses a sub-job without it.
Streaming runs in background threads of the kit process (or step by step from ``status`` when ``background=False``),
so the kit host must stay up and on the endpoint's network for the point's lifetime.

Params: shards (sub-jobs, 1..32), files_per_job (gen files each, 1..20), shard_offset (first gen file, 0..19),
max_events (per file, 0 = all), seed, wall_minutes (per sub-job). Metrics: accepted and generated events, accepted per
GPU-hour (charged and chain-only), accepted per estimated SU, campaign time, input wait, clusters per generated event
per sector type.
"""
from __future__ import annotations

import json
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import EndpointError, Resources
from .slurm_ssh import su_estimate

PARAMS = ("shards", "files_per_job", "shard_offset", "max_events", "seed", "wall_minutes")
METRICS = ("accepted_events", "generated_events", "jobs_ok", "gpu_hours", "est_su", "accepted_per_gpu_hour",
           "accepted_per_est_su", "accepted_per_gpu_hour_chain", "input_wait_s_mean", "campaign_s",
           "clusters_per_generated_EX", "clusters_per_generated_T1", "clusters_per_generated_T2",
           "clusters_per_generated_M", "library_reused_jobs")
SECTOR = {1: "EX", 2: "T1", 3: "T2", 4: "M"}
N_GEN = 20                                                  # g4ref_v1 gen shards 1..20
LIBRARY = "tier1b_v7/lib_prod_as_20files.root"
CALIB = "geometry/crvcalib_sim_best_v1_5_run1400.csv"
DEVIATION = "geometry/crvphoton_sim_best_v1_5_run1400.csv"
GDML = "geometry/mu2e_extracted_v04.gdml"
GEN = "g4ref_v1_tables/gen.001400_{:08d}.root"
LIMITS = {"shards": (1, 32), "files_per_job": (1, 20), "shard_offset": (0, N_GEN - 1), "max_events": (0, 10 ** 7),
          "seed": (0, 2 ** 31 - 1), "wall_minutes": (10, 180)}

BODY = """@common@@tables@CODE=@code@
C() { apptainer exec --nv --cleanenv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},CRVFS_LOOKUP_DIR=$L,CRVFS_DATA=$L,PYTHONPATH=$CODE/crv:$CODE/core:$CODE/tracker" -B "$L" -B /dev/shm -B "$M2FS_OUT" -B "$CODE" --pwd "$CODE" @sif@ "$@"; }
C python -c "import crvfastsim.analyses.runchain as r; assert hasattr(r, 'merge')" || { echo "runchain --emit missing in $CODE"; exit 5; }
# library once per node: /dev/shm/crvfs_lib_<user>_<md5>, one holder file per job, deleted by the last holder (flock);
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
tw=$SECONDS; for i in $(seq 360); do [ -f "$L/ready" ] && [ -f "$S/ready" ] && break; sleep 5; done
[ -f "$L/ready" ] && [ -f "$S/ready" ] || { echo "no input arrived"; exit 3; }
echo "{\\"input_wait_s\\": $((SECONDS-tw)), \\"library\\": \\"$LIBSTATE\\"}" > "$M2FS_OUT/wait.json"
GEN=($(ls "$L"/gen.*.root))
C python -m crvfastsim runchain "${GEN[@]}" --library "$S/@lib@" --calib "$L/@calib@" --deviation "$L/@dev@" \\
  --gdml "$L/@gdml@" --jax --seed @seed@ @maxev@ --emit "$M2FS_OUT/shard.json" --outdir "$M2FS_OUT/chain" > "$M2FS_OUT/chain.txt" 2>&1 || exit 6
echo '{}' > "$M2FS_OUT/metrics.json"
"""


class FnalSource:
    """Where the Mu2e-internal inputs live: read with `ssh <host> cat`, never copied to the kit host's disk."""

    def __init__(self, host: str = "mu2egpvm03", root: str = "/exp/mu2e/data/users/ram2aq/crv_fastsim", runner=None):
        self.host, self.root = host, root.rstrip("/")
        self.runner = runner

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


def validate(params: Dict[str, Any]) -> Dict[str, int]:
    missing = [k for k in PARAMS if k not in params]
    extra = sorted(set(params) - set(PARAMS) - {"task"})
    if missing or extra:
        raise ValueError(f"fastsimkit: task 'chain_point' takes {list(PARAMS)}; missing {missing}, extra {extra}")
    out = {}
    for k in PARAMS:
        v = params[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
            raise ValueError(f"fastsimkit: chain_point {k} must be a whole number, got {v!r}")
        lo, hi = LIMITS[k]
        if not lo <= int(v) <= hi:
            raise ValueError(f"fastsimkit: chain_point {k} must be in [{lo}, {hi}], got {v!r}")
        out[k] = int(v)
    return out


def gen_files(p: Dict[str, int], k: int) -> List[str]:
    """Gen tables of sub-job k: files_per_job consecutive shards starting at shard_offset + k * files_per_job (cycled)."""
    f, o = p["files_per_job"], p["shard_offset"]
    return [GEN.format((o + k * f + i) % N_GEN + 1) for i in range(f)]


class ChainPoint:
    def __init__(self, kit, source: Optional[FnalSource] = None, code: Optional[str] = None, code_ref: str = "main",
                 background: bool = True, poll_s: float = 20.0, streams: int = 2):
        import os
        self.kit, self.ep = kit, kit.ep
        self.source = source or FnalSource(os.environ.get("FASTSIMKIT_FNAL_HOST", "mu2egpvm03"),
                                           os.environ.get("FASTSIMKIT_FNAL_DATA", "/exp/mu2e/data/users/ram2aq/crv_fastsim"))
        self.code = code or os.environ.get("FASTSIMKIT_CODE", "/scratch/ram2aq/mu2efastsim/chain_code_kit")
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
        self._libmd5: Optional[str] = None
        self._node_lib: Dict[str, threading.Lock] = {}     # one library stream per node at a time
        self._lock = threading.Lock()

    # -- records -------------------------------------------------------------------------------------------------
    def _rec_path(self, name):
        return self.dir / f"{name}.json"

    def record(self, name) -> Optional[dict]:
        p = self._rec_path(name)
        return json.loads(p.read_text()) if p.exists() else None

    def subs(self, name) -> List[str]:
        r = self.record(name)
        return [f"{name}.j{k}" for k in range(r["params"]["shards"])] if r else []

    def _mark(self, sub, what, text=""):
        (self.dir / f"{sub}.{what}").write_text(text)

    def _has(self, sub, what) -> bool:
        return (self.dir / f"{sub}.{what}").exists()

    # -- rendering -----------------------------------------------------------------------------------------------
    def render(self, p: Dict[str, int], k: int):
        from .kit import _COMMON, _TABLES, APPTAINER_RUNTIME
        if not hasattr(self.ep, "stream_in"):
            raise ValueError(f"fastsimkit: chain_point streams its inputs into the job and needs an endpoint that can "
                             f"(slurm_ssh); {getattr(self.ep, 'name', '?')!r} cannot")
        sif = shlex.quote(self.kit.sif)
        tables = _TABLES.replace("@tables@", "LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root")
        body = BODY.replace("@common@", _COMMON).replace("@tables@", tables)
        body = body.replace("@runtime_a@", APPTAINER_RUNTIME["a"]).replace("@runtime_t@", APPTAINER_RUNTIME["t"])
        subs = {"sif": sif, "code": shlex.quote(self.code), "lib": Path(LIBRARY).name, "calib": Path(CALIB).name,
                "dev": Path(DEVIATION).name, "gdml": Path(GDML).name, "seed": str(p["seed"] + 1000 * k),
                "maxev": f"--max-events {p['max_events']}" if p["max_events"] else "",
                "libmd5": self.lib_md5()[:12]}
        for a, b in subs.items():
            body = body.replace(f"@{a}@", b)
        import re
        if re.search(r"@(sif|code|lib|calib|dev|gdml|seed|maxev|libmd5|common|tables|runtime_a|runtime_t)@", body):
            raise ValueError("fastsimkit: chain_point: unfilled placeholder")
        res = Resources(cores=16, mem_gb=110, gpus=1, gpu_type="a100", minutes=p["wall_minutes"])
        return body, res

    def lib_md5(self) -> str:
        if self._libmd5 is None:
            self._libmd5 = self.source.md5(LIBRARY)
        return self._libmd5

    # -- contract ------------------------------------------------------------------------------------------------
    def check(self, name, params) -> Dict[str, Any]:
        try:
            p = validate(params)
            self.render(p, 0)
            su = p["shards"] * su_estimate(Resources(cores=16, gpus=1, minutes=p["wall_minutes"]))
            cap = getattr(self.ep, "max_su_per_job", float("inf")) * p["shards"]
            if su > cap:
                raise ValueError(f"fastsimkit: chain_point worst case {su:.0f} SU exceeds {cap:.0f}")
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        return {"ok": True, "message": ""}

    def ensure_code(self):
        q = shlex.quote
        self.ep._sh(f"set -e; C={q(self.code)}; R={q(self.code_ref)}; "
                    f"if [ -d $C/.git ]; then git -C $C fetch -q origin && git -C $C checkout -q $R && "
                    f"git -C $C reset -q --hard origin/$R; else git clone -q https://github.com/RobMina/mu2e-fastsim.git $C "
                    f"&& git -C $C checkout -q $R; fi; git -C $C submodule update -q --init --recursive", timeout=900)

    def submit(self, name, params) -> Dict[str, Any]:
        p = validate(params)
        old = self.record(name)
        if old is not None and old["params"] != p:
            raise ValueError(f"fastsimkit: {name!r} was already submitted with different params")
        if old is None:
            self.ensure_code()
            self._rec_path(name).write_text(json.dumps({"params": p, "submitted_at": time.time(),
                                                        "code": self.code, "code_ref": self.code_ref}))
        for k, sub in enumerate(self.subs(name)):
            body, res = self.render(p, k)
            try:
                self.ep.submit(sub, body, res)
            except EndpointError as e:
                raise ValueError(str(e))
        self.ensure_streaming(name)
        return {"handle": name}

    def ensure_streaming(self, name):
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
        p = self.record(name)["params"]
        from concurrent.futures import ThreadPoolExecutor

        def one(rel):   # the library is the long pole; the small files and gen tables go alongside it
            want = self.source.md5(rel)
            with self._chan:
                got = self.ep.stream_in(sub, host, f"{tmp}/{Path(rel).name}", self.source.argv(rel))
            if got != want:
                raise EndpointError(f"md5 mismatch for {rel}: {got} != {want}")
        def library():   # once per node: the first sub-job there streams it, later ones find it (or wait for it)
            if shared is None:
                one(LIBRARY)
                return
            with self._node_lib.setdefault(host, threading.Lock()):
                have = self.ep.run_on_node(sub, host, f"test -f {shlex.quote(shared)}/ready && echo have || true").strip()
                if have == "have":
                    self._mark(sub, "libreused", host)
                    return
                want = self.lib_md5()
                with self._chan:
                    got = self.ep.stream_in(sub, host, f"{shared}/{Path(LIBRARY).name}", self.source.argv(LIBRARY))
                if got != want:
                    raise EndpointError(f"md5 mismatch for {LIBRARY}: {got} != {want}")
                self.ep.run_on_node(sub, host, f"touch {shlex.quote(shared)}/ready")
                self._mark(sub, "libstreamed", host)

        try:
            with ThreadPoolExecutor(max_workers=self.streams) as ex:
                futs = [ex.submit(library)] + [ex.submit(one, rel) for rel in [CALIB, DEVIATION, GDML] + gen_files(p, k)]
                for f in futs:
                    f.result()
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
        for sub in self.subs(name):
            fs = self.ep.fetch(sub, self.kit.out_root)
            out = Path(self.kit.out_root) / sub / "out"
            if not (out / "cleanup.txt").exists():
                raise ValueError(f"fastsimkit: {sub!r} staged Mu2e-internal inputs but wrote no cleanup record")
            s = json.loads((out / "shard.json").read_text())
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
        for k, lab in SECTOR.items():
            m[f"clusters_per_generated_{lab}"] = cl.get(k, 0) / gen if gen else 0.0
        return {"metrics": {k: float(v) for k, v in m.items()},
                "files": [{"name": str(p.relative_to(self.kit.out_root)), "uri": p.resolve().as_uri(), "kind": "text"}
                          for p in files],
                "metadata": {"endpoint": self.ep.name, "job_ids": ",".join(ids), "code_ref": rec["code_ref"],
                             "su_note": "est_su from Slurm elapsed time and the measured UVA rates; MAM posts the charge later"}}

    def cancel(self, name) -> Dict[str, Any]:
        states = [self.ep.cancel(s) for s in self.subs(name)]
        return {"state": "cancelled" if any(s == "cancelled" for s in states) else (states[0] if states else "cancelled")}
