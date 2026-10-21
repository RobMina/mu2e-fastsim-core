"""Fermilab EAF (Elastic Analysis Facility): an endpoint that runs jobs in the user's JupyterHub server on an A100 MIG
slice, driven from a GPVM with a hub API token.

EAF has no batch queue and no ssh: one server per user, holding one MIG slice (1g.10gb, 2g.20gb or 4g.40gb). A job is
a script started detached in a server terminal (websocket); it reads its inputs in place from /exp and writes
``<workdir>/<handle>/{job.sh, job.log, job.pid, job.done, out/}`` on /exp, which the GPVM reads directly: no transfer,
nothing Mu2e-internal leaves FNAL. The adapter

* starts the server when none runs, trying slices in ``slices`` order (largest first by default; a slice that does not
  become ready within ``start_timeout_s`` counts as unavailable and its spawn is cancelled), and reports "no capacity"
  when none starts;
* runs one job at a time (one slice per user): a submit while another job runs is refused;
* maps states from the files: ``job.done`` holds the exit code; no ``job.done`` and a live server is ``working``;
  no ``job.done`` and no server, or a job past its wall time, is ``failed``;
* cancels by killing the job's process group (``job.pid``), not the server; ``stop_idle`` stops the server once no
  job runs (the kit calls it after results, or a cron).

Configuration: ``token_file`` (default $EAF_TOKEN_FILE), ``hub`` (default https://analytics-hub.fnal.gov), ``workdir``
(on /exp), ``state_dir`` (local job records). ``hub_client`` is injectable for tests (see FakeHub in the tests).
"""
from __future__ import annotations

import json
import os
import shlex
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import EndpointError, JobRecord, JobStatus, Resources, StateDir, spec_sha

PROFILE = "fife-sl7-unpriv-gpu-a100"
IMAGE = "fife-al9-unpriv-htcondor-gpu-a100-{gb}gb"
SLICE_GB = {"4g.40gb": 40, "2g.20gb": 20, "1g.10gb": 10}
CODE = "/exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim"
VENV = "/exp/mu2e/data/users/ram2aq/crv_fastsim/venv_eaf"
ANA_PY = "/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python"
JAX_PKG = "jax[cuda12]==0.11.2"


class HubClient:
    """The JupyterHub REST calls the endpoint needs (token auth), plus terminal stdin over a websocket."""

    def __init__(self, hub: str, token_file: str, timeout_s: float = 60.0):
        self.hub, self.token_file, self.timeout_s = hub.rstrip("/"), token_file, timeout_s

    def _token(self) -> str:
        if not self.token_file or not os.path.exists(self.token_file):
            raise EndpointError("eaf: no hub token file (EAF_TOKEN_FILE); make one at <hub>/hub/token")
        if os.stat(self.token_file).st_mode & 0o077:
            raise EndpointError(f"eaf: {self.token_file} is readable by others; chmod 600 it")
        return open(self.token_file).read().strip()

    def call(self, method: str, path: str, body: Optional[dict] = None):
        req = urllib.request.Request(self.hub + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "token " + self._token(),
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=ssl.create_default_context()) as r:
                data = r.read()
                return r.status, (json.loads(data) if data else None)
        except urllib.error.HTTPError as e:
            raise EndpointError(f"eaf: {method} {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError) as e:
            raise EndpointError(f"eaf: {method} {path} failed: {e} (EAF is reachable only from the FNAL network)")

    def user(self) -> dict:
        return self.call("GET", "/hub/api/user")[1]

    def server(self) -> Optional[dict]:
        return (self.user().get("servers") or {}).get("")

    def start(self, options: dict) -> None:
        self.call("POST", f"/hub/api/users/{self.user()['name']}/server", options)

    def stop(self) -> None:
        self.call("DELETE", f"/hub/api/users/{self.user()['name']}/server")

    def send(self, line: str) -> None:
        """Type one line into a new server terminal."""
        import websocket
        s = self.server()
        if not (s and s.get("ready")):
            raise EndpointError("eaf: no running server")
        _, term = self.call("POST", s["url"] + "api/terminals", {})
        url = self.hub.replace("https://", "wss://") + s["url"] + f"terminals/websocket/{term['name']}"
        ws = websocket.create_connection(url, header=["Authorization: token " + self._token()], timeout=30)
        try:
            ws.send(json.dumps(["stdin", line + "\r"]))
            time.sleep(2)
        finally:
            ws.close()


def slice_of(options: Optional[dict]) -> str:
    img = (options or {}).get("imagegpu", "")
    for name, gb in SLICE_GB.items():
        if img.endswith(f"-{gb}gb"):
            return name
    return ""


class EAFEndpoint:
    def __init__(self, workdir: str = "/exp/mu2e/data/users/ram2aq/crv_fastsim/eafkit", code: str = CODE,
                 venv: str = VENV, shared_prefix: str = "/exp/",
                 state_dir: str | Path = "~/.mu2efastsim/eaf", token_file: Optional[str] = None,
                 hub: str = "https://analytics-hub.fnal.gov", slices: Sequence[str] = ("4g.40gb", "2g.20gb", "1g.10gb"),
                 start_timeout_s: float = 420.0, poll_s: float = 10.0, hub_client: Optional[HubClient] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 name: str = "eaf", **_ignored):
        self.name = name
        self.workdir, self.code, self.venv = workdir.rstrip("/"), code.rstrip("/"), venv.rstrip("/")
        if not self.workdir.startswith(shared_prefix):
            raise EndpointError(f"eaf: workdir must be under {shared_prefix} (shared by EAF and the GPVM)")
        bad = [s for s in slices if s not in SLICE_GB]
        if bad or not slices:
            raise EndpointError(f"eaf: unknown slices {bad}; choose from {list(SLICE_GB)}")
        self.slices = tuple(slices)
        self.state = StateDir(Path(state_dir).expanduser())
        self.hub = hub_client or HubClient(hub, token_file or os.environ.get("EAF_TOKEN_FILE", ""))
        self.start_timeout_s, self.poll_s = start_timeout_s, poll_s
        self.clock, self.sleep = clock, sleep

    # -- files on /exp -------------------------------------------------------------------------------------------
    def _dir(self, handle: str) -> Path:
        return Path(self.workdir) / handle

    def _read(self, handle: str, name: str) -> Optional[str]:
        p = self._dir(handle) / name
        return p.read_text() if p.exists() else None

    # -- server --------------------------------------------------------------------------------------------------
    def _ready_server(self, resources: Resources) -> dict:
        """The running server, or start one: slices in order, skipping those too small for resources.mem_gb."""
        s = self.hub.server()
        if s and s.get("ready"):
            return s
        if s and s.get("pending"):
            raise EndpointError("eaf: a server is starting or stopping; retry shortly")
        tried = []
        for sl in self.slices:
            self.hub.start({"profile": PROFILE, "imagegpu": IMAGE.format(gb=SLICE_GB[sl])})
            t0 = self.clock()
            while self.clock() - t0 < self.start_timeout_s:
                s = self.hub.server()
                if s and s.get("ready"):
                    return s
                self.sleep(self.poll_s)
            tried.append(sl)
            self.hub.stop()                     # cancel the spawn that did not get a slice
            self.sleep(self.poll_s)
        raise EndpointError(f"eaf: no capacity: no slice of {tried or list(self.slices)} started within "
                            f"{self.start_timeout_s:.0f} s")

    def _running_job(self) -> Optional[str]:
        """Handle of a job that still runs here (one slice per user), or None."""
        for p in sorted(self.state.root.glob("*.json")):
            rec = JobRecord(**json.loads(p.read_text()))
            if rec.final is None and self.status(rec.handle).state == "working":
                return rec.handle
        return None

    def launch_line(self, d: Path, resources: Resources) -> str:
        """The line typed into a server terminal: one detached process group (setsid), its leader's pid in job.pid
        for cancel, a watchdog for the wall time, and the exit code in job.done (143 on TERM, with a marker telling
        a cancel from a timeout)."""
        q = {k: shlex.quote(str(d / k)) for k in ("job.sh", "job.log", "job.pid", "job.done", "timedout")}
        wall = max(1, int(resources.minutes)) * 60
        inner = (f"trap 'echo exit=143 > {q['job.done']}; exit 143' TERM; echo $$ > {q['job.pid']}; "
                 f"( sleep {wall}; echo 1 > {q['timedout']}; kill -TERM -- -$$ ) & w=$!; "
                 f"bash {q['job.sh']}; rc=$?; kill $w 2>/dev/null; echo exit=$rc > {q['job.done']}")
        return f"setsid nohup bash -c {shlex.quote(inner)} > {q['job.log']} 2>&1 < /dev/null &"

    def runtime(self) -> Dict[str, str]:
        """Shell definitions of A (run a command in the fast-chain environment) and T (the same with the staged
        tables in CRVFS_LOOKUP_DIR) for the kit's task bodies: the venv on /exp (built on first use from the Mu2e
        ana env plus pip JAX) instead of an Apptainer image, which EAF cannot start."""
        v, c = shlex.quote(self.venv), shlex.quote(self.code)
        a = (f'M2FS_VENV={v}; M2FS_CODE={c}\n'
             f'[ -x "$M2FS_VENV/bin/python" ] || {ANA_PY} -m venv --system-site-packages "$M2FS_VENV" || exit 2\n'
             f'"$M2FS_VENV/bin/python" -c "import jax" 2>/dev/null || "$M2FS_VENV/bin/python" -m pip install -q '
             f'"{JAX_PKG}" || exit 2\n'
             'git -C "$M2FS_CODE" log -1 --format="umbrella %h" 2>/dev/null; git -C "$M2FS_CODE/crv" log -1 --format="crv %h" 2>/dev/null\n'
             'A() { local c=$1; shift; [ "$c" = python ] && c="$M2FS_VENV/bin/python"; ( cd "$M2FS_CODE" && '
             'PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker" XLA_PYTHON_CLIENT_PREALLOCATE=false '
             '"$c" "$@" ); }\n')
        t = 'T() { ( export CRVFS_LOOKUP_DIR="$L"; A "$@" ); }'
        return {"a": a, "t": t}

    # -- interface -----------------------------------------------------------------------------------------------
    def submit(self, handle: str, script: str, resources: Resources) -> JobRecord:
        sha = spec_sha(script, resources)
        old = self.state.get(handle)
        if old is not None:
            if old.spec_sha != sha:
                raise EndpointError(f"{handle!r} was already submitted with a different script or resources")
            return old
        busy = self._running_job()
        if busy:
            raise EndpointError(f"eaf: job {busy!r} still runs (one MIG slice per user); retry after it finishes")
        if resources.gpus > 1:
            raise EndpointError("eaf: one MIG slice per user; gpus must be 0 or 1")
        s = self._ready_server(resources)
        d = self._dir(handle)
        (d / "out").mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        q = {k: shlex.quote(str(d / k)) for k in ("job.sh", "job.log", "job.pid", "job.done")}
        header = (f"#!/bin/bash\nexport M2FS_JOB_DIR={shlex.quote(str(d))} M2FS_OUT={shlex.quote(str(d / 'out'))}\n"
                  "umask 077\nnvidia-smi -L\n")
        (d / "job.sh").write_text(header + script)
        self.hub.send(self.launch_line(d, resources))
        rec = JobRecord(handle, sha, "", str(d), self.clock(), dict(resources.__dict__, slice=slice_of(s.get("user_options"))))
        self.state.put(rec)
        return rec

    def _record(self, handle: str) -> JobRecord:
        rec = self.state.get(handle)
        if rec is None:
            raise EndpointError(f"{self.name}: no job {handle!r}")
        return rec

    def status(self, handle: str) -> JobStatus:
        rec = self._record(handle)
        if rec.final:
            return JobStatus(**rec.final)
        slice_ = rec.resources.get("slice", "")
        done = self._read(handle, "job.done")
        if done is not None:
            code = done.strip().split("=")[-1]
            cancelled, timedout = self._read(handle, "cancelled") is not None, self._read(handle, "timedout") is not None
            state = "completed" if code == "0" else ("cancelled" if cancelled and not timedout else "failed")
            msg = "timed out (wall time)" if timedout else f"exit {code}"
            js = JobStatus(state, msg, slice_, rec.job_id)
        elif self.clock() - rec.submitted_at > 60 * rec.resources.get("minutes", 60) + 120:
            js = JobStatus("failed", "no exit record after the wall time (server culled or stopped?)", slice_, rec.job_id)
        elif self._read(handle, "job.pid") is None and self.clock() - rec.submitted_at > 120:
            js = JobStatus("failed", "the job never started in the server terminal", slice_, rec.job_id)
        else:
            s = self.hub.server()
            if not (s and s.get("ready")):
                js = JobStatus("failed", "the EAF server stopped before the job finished", slice_, rec.job_id)
            else:
                return JobStatus("working", "", slice_, rec.job_id)
        rec.final = js.__dict__.copy()
        self.state.put(rec)
        return js

    def fetch(self, handle: str, dest: Path) -> List[Path]:
        """Copy out/ and job.log into dest/<handle>/ (both are on /exp already)."""
        import shutil
        rec = self._record(handle)
        src, target = Path(rec.remote_dir), Path(dest) / handle
        if (src / "out").is_dir():
            shutil.copytree(src / "out", target / "out", dirs_exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        if (src / "job.log").exists():
            shutil.copy2(src / "job.log", target / "job.log")
        return sorted(p for p in target.rglob("*") if p.is_file())

    def cancel(self, handle: str) -> str:
        rec = self._record(handle)
        st = self.status(handle)
        if st.state != "working":
            return st.state
        pid = (self._read(handle, "job.pid") or "").strip()
        if not pid.isdigit():
            raise EndpointError(f"eaf: {handle!r} has no pid yet; retry")
        (Path(rec.remote_dir) / "cancelled").write_text(str(self.clock()))
        self.hub.send(f"kill -TERM -- -{pid} 2>/dev/null || kill -TERM {pid}")
        return "cancelled"

    def stop_idle(self) -> bool:
        """Stop the server when no job runs; True if a stop was sent."""
        if self._running_job():
            return False
        s = self.hub.server()
        if s and (s.get("ready") or s.get("pending")):
            self.hub.stop()
            return True
        return False


def main(argv=None) -> int:
    """python -m mu2efastsim.endpoints.eaf {server|jobs|stop-idle} [--state-dir D] (needs EAF_TOKEN_FILE)."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("cmd", choices=("server", "jobs", "stop-idle"))
    ap.add_argument("--state-dir", default=os.environ.get("FASTSIMKIT_STATE_DIR", "~/.mu2efastsim/kit") + "/jobs")
    a = ap.parse_args(argv)
    ep = EAFEndpoint(state_dir=a.state_dir)
    if a.cmd == "server":
        s = ep.hub.server()
        print(json.dumps(s and {k: s.get(k) for k in ("ready", "pending", "started", "user_options")}, indent=1))
    elif a.cmd == "jobs":
        for p in sorted(ep.state.root.glob("*.json")):
            h = p.stem
            print(h, ep.status(h).__dict__)
    else:
        print("stopped" if ep.stop_idle() else "not stopped (a job runs, or no server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
