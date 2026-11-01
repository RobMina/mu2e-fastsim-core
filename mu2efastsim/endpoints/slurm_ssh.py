"""Slurm over SSH: an endpoint for a Slurm site reached by ``ssh <host>`` with a key (UVA Rivanna/Afton).

Site settings come from arguments or the environment, never from the package: ``host`` ($FASTSIMKIT_SLURM_HOST),
``account`` ($FASTSIMKIT_SLURM_ACCOUNT) and ``workdir`` ($FASTSIMKIT_WORKDIR, else <FASTSIMKIT_SCRATCH>/kit); a
missing one raises mu2efastsim.site.SiteConfigError naming the variable when it is first needed.

Every remote action is one ssh command through ``runner`` (injectable for tests). Job layout on the site:
``<workdir>/<handle>/{job.sh, out/, slurm.out}``; the job script writes its results under ``out/``. The job id is kept
in a local state dir, so a restarted caller finds its jobs. ``status`` asks ``squeue`` first and ``sacct`` once the job
has left the queue; a final state is cached locally. Budget guard: a submit whose estimated charge exceeds
``max_su_per_job`` is refused (UVA's published per-type SU rates, GPU_TYPES). ``gpu_type`` picks the card of GPU jobs:
"auto" (default) puts a one-GPU job whose memory request fits the free RTX PRO 6000 MIG slice (``1g.24gb`` on
``gpu-mig``, at most 32 GB and 4 cores) on the slice and leaves any larger request on the card the task asked for; an
explicit type (``a40``, ``a100``, ``1g.24gb``, ...) applies to every GPU job; None keeps every task's own request. The
choice and the original request go into the job script header and the job record (``gpu_choice``).
Durable intents and per-handle locks fence each dispatch attempt; an unrecorded intent is reconciled by exact
scheduler comment, never blindly resubmitted. See docs/slurm-recovery.md for storage and single-cluster limits.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from mu2efastsim import site

from . import EndpointError, JobRecord, JobStatus, Resources, StateDir, spec_sha
from .submission import IntentStore, durable_json, handle_key, read_json, request_digest, submission_lock

WORKING = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED", "STAGE_OUT",
           "SIGNALING", "REQUEUE_HOLD", "REQUEUE_FED", "PREEMPTED"}
FAILED = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
Runner = Callable[[Sequence[str], Optional[bytes], float], subprocess.CompletedProcess]


def ssh_runner(host: str, connect_timeout: int = 15) -> Runner:
    def run(cmd: Sequence[str], stdin: Optional[bytes], timeout: float) -> subprocess.CompletedProcess:
        remote = " ".join(shlex.quote(c) for c in cmd)
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}", host, remote],
                              input=stdin, capture_output=True, timeout=timeout)
    return run


# UVA GPU types: partition, SU per GPU-hour / core-hour / GB-hour (rc-website data/service.yaml and the Slurm billing
# weights / 10, read 2026-10-30), and the largest (cores, GB) a job of that type may ask for
GPU_TYPES = {
    "a100": ("gpu", 50.89, 0.403, 0.043, None),
    "a40": ("gpu", 18.669, 0.34, 0.02, None),
    "a6000": ("gpu", 14.273, 0.383, 0.062, None),
    "v100": ("gpu", 2.096, 0.058, 0.005, None),
    "b200": ("gpu", 127.42, 0.437, 0.044, None),
    "rtx_pro_6000": ("gpu", 33.43, 0.427, 0.049, None),
    "1g.24gb": ("gpu-mig", 0.0, 0.0, 0.0, (4, 32)),
}
CPU_RATES = (0.177, 0.0143)   # standard (Rivanna) per core-hour, per GB-hour
AUTO, SLICE = "auto", "1g.24gb"   # default gpu_type: the slice for requests that fit it, else the task's card


def su_estimate(r: Resources, su_per_gpu_h: Optional[float] = None, su_per_core_h: Optional[float] = None) -> float:
    """Charge estimate: explicit flat rates when given, else UVA's rates for r.gpu_type (A100 when unknown)."""
    if su_per_gpu_h is not None or su_per_core_h is not None:
        return r.minutes / 60.0 * (r.gpus * (su_per_gpu_h or 0.0) + r.cores * (su_per_core_h or 0.0))
    if not r.gpus:
        return r.minutes / 60.0 * (r.cores * CPU_RATES[0] + r.mem_gb * CPU_RATES[1])
    _, gpu, core, gb, _ = GPU_TYPES.get(r.gpu_type, GPU_TYPES["a100"])
    return r.minutes / 60.0 * (r.gpus * gpu + r.cores * core + r.mem_gb * gb)


def map_state(site_state: str, exit_code: str = "0:0") -> str:
    s = site_state.split()[0].rstrip("+") if site_state else ""
    if s in WORKING:
        return "working"
    if s == "COMPLETED":
        return "completed" if exit_code.split(":")[0] == "0" else "failed"
    if s.startswith("CANCELLED"):
        return "cancelled"
    if s in FAILED:
        return "failed"
    raise EndpointError(f"unknown Slurm state {site_state!r}")


class SlurmSSHEndpoint:
    def __init__(self, host: str | None = None, account: str | None = None, partition: str = "gpu",
                 workdir: str | None = None, state_dir: str | Path = "~/.mu2efastsim/uva",
                 max_su_per_job: float = 500.0, runner: Optional[Runner] = None, name: str = "uva",
                 su_per_gpu_h: Optional[float] = None, su_per_core_h: Optional[float] = None, timeout_s: float = 120.0,
                 stream_runner=None, gpu_type: Optional[str] = AUTO):
        if gpu_type not in (None, AUTO) and gpu_type not in GPU_TYPES:
            raise EndpointError(f"unknown gpu_type {gpu_type!r}; known: {AUTO!r}, None, {sorted(GPU_TYPES)}")
        self.name, self._host, self._account, self.partition = name, host, account, partition
        self.gpu_type = gpu_type
        self.workdir = (workdir or site.setting("FASTSIMKIT_WORKDIR") or site.scratch() + "/kit").rstrip("/")
        self.state = StateDir(Path(state_dir).expanduser())
        self.max_su_per_job = max_su_per_job
        self.su_rates = (su_per_gpu_h, su_per_core_h)
        self.run = runner or (lambda cmd, stdin, timeout: ssh_runner(self.host)(cmd, stdin, timeout))
        self.timeout_s = timeout_s
        self.stream_runner = stream_runner   # tests: (source_argv, host, remote_cmd, timeout) -> node md5 output

    @property
    def host(self) -> str:
        return self._host or site.slurm_host()

    @host.setter
    def host(self, value: str) -> None:     # explicit retarget: overrides $FASTSIMKIT_SLURM_HOST for this endpoint
        self._host = value

    @property
    def account(self) -> str:
        return self._account or site.slurm_account()

    @account.setter
    def account(self, value: str) -> None:  # overrides $FASTSIMKIT_SLURM_ACCOUNT for this endpoint
        self._account = value

    # -- helpers -------------------------------------------------------------------------------------------------
    def _sh(self, script: str, stdin: Optional[bytes] = None, timeout: Optional[float] = None) -> str:
        return self._raw(script, stdin, timeout).decode(errors="replace")

    def _raw(self, script: str, stdin: Optional[bytes] = None, timeout: Optional[float] = None) -> bytes:
        try:
            p = self.run(["bash", "-c", script], stdin, timeout or self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise EndpointError(f"{self.name}: ssh timed out after {e.timeout} s (UVA needs the UVA network or VPN)")
        if p.returncode != 0:
            err = (p.stderr or b"").decode(errors="replace").strip()[-400:]
            raise EndpointError(f"{self.name}: remote command failed (rc {p.returncode}): {err}")
        return p.stdout or b""

    def _dir(self, handle: str) -> str:
        return f"{self.workdir}/{handle}"

    def gpu_choice(self, r: Resources):
        """(the job actually submitted, why) for request r; why is None when r passes unchanged by policy None."""
        if not r.gpus or self.gpu_type is None:
            return r, None
        requested = dict(r.__dict__)
        if self.gpu_type == AUTO:
            cores, mem = GPU_TYPES[SLICE][4]
            if r.gpus == 1 and r.mem_gb <= mem:
                out = Resources(cores=min(r.cores, cores), mem_gb=r.mem_gb, gpus=1, gpu_type=SLICE, minutes=r.minutes)
                why = f"{r.mem_gb} GB fits the {SLICE} slice (<= {mem} GB, cores capped at {cores})"
            else:
                out = r
                why = f"{r.gpus} GPU(s), {r.mem_gb} GB do not fit one {SLICE} slice ({mem} GB): the task's card"
            return out, {"policy": AUTO, "chosen": out.gpu_type, "requested": requested, "why": why}
        cap = GPU_TYPES[self.gpu_type][4]
        cores, mem = (min(r.cores, cap[0]), min(r.mem_gb, cap[1])) if cap else (r.cores, r.mem_gb)
        out = Resources(cores=cores, mem_gb=mem, gpus=r.gpus, gpu_type=self.gpu_type, minutes=r.minutes)
        return out, {"policy": "explicit", "chosen": self.gpu_type, "requested": requested,
                     "why": f"gpu_type={self.gpu_type} set on the endpoint"}

    def shape(self, r: Resources) -> Resources:
        """The job actually submitted for request r (gpu_choice without the reason)."""
        return self.gpu_choice(r)[0]

    def _partition(self, r: Resources) -> str:
        if not r.gpus:
            return "standard"
        return GPU_TYPES[r.gpu_type][0] if r.gpu_type in GPU_TYPES and self.partition == "gpu" else self.partition

    def header(self, handle: str, r: Resources, choice: Optional[dict] = None) -> str:
        gres = f"gpu:{r.gpu_type}:{r.gpus}" if r.gpu_type else f"gpu:{r.gpus}"
        lines = ["#!/bin/bash", f"#SBATCH -J m2fs_{handle}"[:60], f"#SBATCH -A {self.account}",
                 f"#SBATCH -p {self._partition(r)}", f"#SBATCH -c {r.cores}",
                 f"#SBATCH --mem={r.mem_gb}G", f"#SBATCH -t {r.minutes}", f"#SBATCH -o {self._dir(handle)}/slurm.out"]
        if r.gpus:
            lines.insert(4, f"#SBATCH --gres={gres}")
        if choice:
            lines.append("# m2fs gpu_choice " + json.dumps(choice, sort_keys=True))
        return "\n".join(lines) + f"\nexport M2FS_JOB_DIR={self._dir(handle)} M2FS_OUT={self._dir(handle)}/out\n"

    def admit_resources(self, resources: Resources) -> None:
        """Apply the configured per-job budget without contacting Slurm."""
        su = su_estimate(self.shape(resources), *self.su_rates)
        if su > self.max_su_per_job:
            raise EndpointError(f"estimated {su:.0f} SU exceeds max_su_per_job {self.max_su_per_job:.0f}")

    def _target(self, resources):
        return {"host": self.host, "account": self.account, "workdir": self.workdir,
                "partition": self._partition(resources)}

    def _load_record(self, handle):
        handle_key(handle)
        path = self.state._path(handle)
        data = read_json(path)
        if data is None and not path.exists():
            return None
        try:
            rec = JobRecord(**data)
            if (rec.handle != handle or not re.fullmatch(r"[1-9][0-9]*", rec.job_id)
                    or not re.fullmatch(r"[0-9a-f]{16}", rec.spec_sha) or not isinstance(rec.remote_dir, str)
                    or not isinstance(rec.resources, dict)):
                raise ValueError("invalid job record identity")
            intent = IntentStore(self.state.root).get(handle)
            if intent is not None:
                resources = Resources(**rec.resources)
                if (intent["target"] != self._target(resources) or intent["spec_sha"] != rec.spec_sha
                        or intent["resources"] != rec.resources or intent["remote_dir"] != rec.remote_dir
                        or intent["created_at"] != rec.submitted_at or intent.get("unsupported_ack")):
                    raise ValueError("record/intent identity or endpoint target mismatch")
            return rec
        except (ValueError, TypeError, KeyError) as exc:
            raise EndpointError(f"{handle!r}: invalid job record or target: {exc}") from exc

    def _save_record(self, rec):
        durable_json(self.state._path(rec.handle), rec.__dict__)

    def _adopt(self, intent, job_id):
        rec = JobRecord(intent["handle"], intent["spec_sha"], job_id, intent["remote_dir"],
                        intent["created_at"], intent["resources"])
        self._save_record(rec)
        return rec

    @staticmethod
    def _recovery_ids(text, token):
        ids = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.strip().split("|", 1)
            if len(fields) != 2:
                raise EndpointError("uninterpretable scheduler recovery row")
            job_id, comment = (field.strip() for field in fields)
            if comment == token:
                if not re.fullmatch(r"[1-9][0-9]*", job_id):
                    raise EndpointError(f"unsupported matching allocation ID {job_id!r}")
                ids.add(job_id)
        return ids

    def _reconcile(self, intent):
        if intent.get("unsupported_ack"):
            raise EndpointError(f"{intent['handle']!r}: nonlocal or compound sbatch acknowledgement; "
                                "operator investigation required; intent retained, no resubmission")
        token = intent["token"]
        start = dt.datetime.fromtimestamp(max(0, intent["created_at"] - 86400), dt.timezone.utc)
        since = start.strftime("%Y-%m-%dT%H:%M:%S")
        q = self._sh("unset SLURM_CLUSTERS; squeue --local -h --me -o '%i|%.128k'")
        a = self._sh(f"unset SLURM_CLUSTERS; TZ=UTC sacct --local -n -X -P --duplicates --user=\"$(id -un)\" "
                     f"--starttime={since} -o JobIDRaw,Comment%128")
        ids = self._recovery_ids(q, token) | self._recovery_ids(a, token)
        if len(ids) != 1:
            detail = "no visible matching allocation" if not ids else f"multiple matching allocations {sorted(ids)}"
            raise EndpointError(f"{intent['handle']!r}: unresolved submission ({detail}); retry the same request "
                                "to reconcile only; do not delete the intent or submit a new handle without investigation")
        return self._adopt(intent, ids.pop())

    # -- interface -----------------------------------------------------------------------------------------------
    def submit(self, handle: str, script: str, resources: Resources) -> JobRecord:
        with submission_lock(self.state.root, handle):
            old = self._load_record(handle)
            intent = IntentStore(self.state.root).get(handle)
            if ((old is not None and old.spec_sha == spec_sha(script, resources))
                    or (intent is not None and intent["resources"] == resources.__dict__
                        and intent["request_sha"] == self._request(handle, script, resources)[2])):
                return self._submit_locked(handle, script, resources)   # recorded as requested, before gpu_type
            return self._submit_locked(handle, script, *self.gpu_choice(resources))

    def _request(self, handle, script, resources, choice=None):
        """(job script body, target, request digest) of one submission."""
        target = self._target(resources)
        body = self.header(handle, resources, choice) + "umask 077\nmkdir -p \"$M2FS_OUT\"\n" + script
        return body, target, request_digest(body, resources.__dict__, target)

    def _submit_locked(self, handle, script, resources, choice=None):
        sha = spec_sha(script, resources)
        old = self._load_record(handle)
        if old is not None:
            if old.spec_sha != sha:
                raise EndpointError(f"{handle!r} was already submitted with a different script or resources")
            return old
        d = self._dir(handle)
        body, target, request = self._request(handle, script, resources, choice)
        intents = IntentStore(self.state.root)
        intent = intents.get(handle)
        if intent is not None:
            if (intent["request_sha"] != request or intent["spec_sha"] != sha or intent["target"] != target
                    or intent["remote_dir"] != d or intent["resources"] != resources.__dict__):
                raise EndpointError(f"{handle!r}: pending submission has a different request or target")
            return self._reconcile(intent)
        self.admit_resources(resources)
        intent = intents.create(handle, sha, request, target, resources.__dict__.copy(), d, time.time())
        try:
            out = self._sh(f"set -e; unset SLURM_CLUSTERS SBATCH_CLUSTERS; mkdir -p {shlex.quote(d)}/out; "
                           f"cat > {shlex.quote(d)}/job.sh; cd {shlex.quote(d)} && "
                           f"sbatch --parsable --comment={shlex.quote(intent['token'])} job.sh", stdin=body.encode())
            job_id = out.strip()
            if re.fullmatch(r"[1-9][0-9]*(?:;[^\n]+|[+_][^\n]+)", job_id):
                intent["unsupported_ack"] = job_id
                durable_json(intents._path(handle), intent)
            if not re.fullmatch(r"[1-9][0-9]*", job_id):
                raise EndpointError(f"unsupported or missing local sbatch job ID: {out!r}")
            rec = self._adopt(intent, job_id)
            if choice:
                rec.gpu_choice = choice
                self._save_record(rec)
            return rec
        except (EndpointError, OSError) as exc:
            raise EndpointError(f"{handle!r}: submission outcome uncertain; intent retained; retry the same request "
                                f"to reconcile only (no automatic resubmission): {exc}") from exc

    def _record(self, handle: str) -> JobRecord:
        rec = self._load_record(handle)
        if rec is None:
            raise EndpointError(f"{self.name}: no job {handle!r}")
        return rec

    def status(self, handle: str) -> JobStatus:
        with submission_lock(self.state.root, handle):
            return self._status_locked(handle)

    def _status_locked(self, handle):
        rec = self._record(handle)
        if rec.final:
            return JobStatus(**rec.final)
        q = self._sh(f"squeue -h -j {rec.job_id} -o '%T|%r' 2>/dev/null || true").strip()
        if q:
            st, reason = (q.splitlines()[0].split("|") + [""])[:2]
            return JobStatus("working", reason if reason not in ("None", "") else "", st, rec.job_id)
        a = self._sh(f"sacct -n -X -P -j {rec.job_id} -o State,ExitCode").strip()
        if not a:
            return JobStatus("working", "not yet in squeue or sacct", "", rec.job_id)
        st, code = (a.splitlines()[0].split("|") + ["0:0"])[:2]
        js = JobStatus(map_state(st, code), f"slurm {st} exit {code}", st, rec.job_id)
        rec.final = js.__dict__.copy()
        self._save_record(rec)
        return js

    def fetch(self, handle: str, dest: Path) -> List[Path]:
        """Copy <remote_dir>/out and slurm.out into dest/<handle>/ (tar over ssh); returns the local files."""
        rec = self._record(handle)
        d = shlex.quote(rec.remote_dir)
        raw = self._raw(f"cd {d} && if [ -f slurm.out ]; then tar -cf - out slurm.out; else tar -cf - out; fi",
                        timeout=600)
        target = Path(dest) / handle
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            t.extractall(target, filter="data")
        return sorted(p for p in target.rglob("*") if p.is_file())

    # -- extras for kit tasks that stream inputs into a running job (chain_point) ----------------------------------
    def read_file(self, handle: str, rel: str) -> str:
        """Contents of <remote_dir>/<rel>, or "" if it does not exist yet."""
        rec = self._record(handle)
        return self._sh(f"cat {shlex.quote(rec.remote_dir + '/' + rel)} 2>/dev/null || true")

    def _srun(self, rec: JobRecord, node: str) -> str:
        return f"srun --jobid={rec.job_id} --overlap -N1 -n1 --mem=0 -w {shlex.quote(node)}"

    def run_on_node(self, handle: str, node: str, cmd: str, timeout: float = 120) -> str:
        """Run `cmd` as a step of the running job on its node (compute nodes take no ssh)."""
        rec = self._record(handle)
        return self._sh(f"{self._srun(rec, node)} sh -c {shlex.quote(cmd)}", timeout=timeout)

    def stream_in(self, handle: str, node: str, dest: str, source_argv: Sequence[str], timeout: float = 3600) -> str:
        """Pipe the stdout of `source_argv` (run on this host, e.g. `ssh <fnal> cat <file>`) into `dest` on the job's
        node, without touching this host's disk; returns the md5 the node computed."""
        rec = self._record(handle)
        remote = f"{self._srun(rec, node)} sh -c {shlex.quote(f'cat > {shlex.quote(dest)} && md5sum {shlex.quote(dest)}')}"
        if self.stream_runner is not None:
            out = self.stream_runner(list(source_argv), self.host, remote, timeout)
        else:
            src = subprocess.Popen(list(source_argv), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                p = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, remote], stdin=src.stdout,
                                   capture_output=True, timeout=timeout)
            finally:
                src.stdout.close()
                src.wait(timeout=60)
            if p.returncode or src.returncode:
                raise EndpointError(f"{self.name}: streaming into {dest} failed (rc {p.returncode}/{src.returncode}): "
                                    f"{p.stderr.decode(errors='replace')[-300:]}")
            out = p.stdout.decode()
        return out.split()[0] if out.split() else ""

    def times(self, handle: str) -> dict:
        """Slurm start/end (epoch s) and elapsed seconds of a finished job."""
        import datetime as dt
        rec = self._record(handle)
        a = self._sh(f"sacct -n -X -P -j {rec.job_id} -o Start,End,ElapsedRaw").strip().splitlines()
        s, e, el = (a[0].split("|") + ["", "", "0"])[:3] if a else ("", "", "0")
        ts = lambda x: dt.datetime.fromisoformat(x).timestamp() if x and x[0].isdigit() else time.time()
        return {"start": ts(s), "end": ts(e), "elapsed_s": float(el or 0), "job_id": rec.job_id}

    def cancel(self, handle: str) -> str:
        rec = self._record(handle)
        st = self.status(handle)
        if st.state != "working":
            return st.state
        self._sh(f"scancel {rec.job_id}")
        return "cancelled"
