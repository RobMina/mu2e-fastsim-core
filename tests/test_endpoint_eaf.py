"""EAF endpoint against a fake JupyterHub; the launch line runs for real in a local bash (no EAF, no network)."""
import json
import shutil
import subprocess
import time

import pytest

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.eaf import EAFEndpoint, slice_of
from mu2efastsim.endpoints.kit import FastsimKit


class FakeHub:
    """Server life cycle in memory; `send` runs the typed line in a local bash, as the EAF terminal would.
    available: slices that start; others never become ready."""

    def __init__(self, available=("2g.20gb",), run=True):
        self.available, self.run = set(available), run
        self.srv, self.starts, self.stops, self.sent = None, [], 0, []

    def server(self):
        return self.srv

    def start(self, options):
        self.starts.append(slice_of(options))
        ok = slice_of(options) in self.available
        self.srv = {"ready": ok, "pending": None if ok else "spawn", "url": "/user/u/", "user_options": options}

    def stop(self):
        self.stops += 1
        self.srv = None

    def send(self, line):
        self.sent.append(line)
        if self.run:
            subprocess.run(["bash", "-c", line], check=True)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(tmp_path, hub, clock=None, **kw):
    clock = clock or Clock()
    return EAFEndpoint(workdir=str(tmp_path / "work"), state_dir=tmp_path / "jobs", hub_client=hub, clock=clock,
                       sleep=clock.sleep, start_timeout_s=60, poll_s=10, shared_prefix=str(tmp_path), **kw), clock


needs_setsid = pytest.mark.skipif(shutil.which("setsid") is None, reason="setsid (util-linux) not available")


def wait_pid(ep, h, timeout=10):
    t0 = time.time()
    while not (ep._dir(h) / "job.pid").exists() and time.time() - t0 < timeout:
        time.sleep(0.1)


def wait_done(ep, h, timeout=20):
    t0 = time.time()
    while (ep._dir(h) / "job.done").exists() is False and time.time() - t0 < timeout:
        time.sleep(0.1)


def test_slice_fallback_and_no_capacity(tmp_path):
    hub = FakeHub(available=("1g.10gb",), run=False)
    ep, _ = make(tmp_path, hub)
    ep.submit("p1", "true\n", Resources(minutes=5))
    assert hub.starts == ["4g.40gb", "2g.20gb", "1g.10gb"] and hub.stops == 2      # the two stuck spawns cancelled
    assert ep.state.get("p1").resources["slice"] == "1g.10gb"
    hub2 = FakeHub(available=(), run=False)
    ep2, _ = make(tmp_path / "b", hub2)
    with pytest.raises(EndpointError, match="no capacity"):
        ep2.submit("p2", "true\n", Resources(minutes=5))


@needs_setsid
def test_run_complete_fetch_and_idempotent(tmp_path):
    hub = FakeHub()
    ep, clock = make(tmp_path, hub)
    script = 'echo hello; echo \'{"x": 1.5}\' > "$M2FS_OUT/metrics.json"\n'
    rec = ep.submit("p1", script, Resources(minutes=5))
    wait_done(ep, "p1")
    assert ep.status("p1").state == "completed"
    files = ep.fetch("p1", tmp_path / "fetched")
    assert any(p.name == "metrics.json" for p in files) and "hello" in (tmp_path / "fetched/p1/job.log").read_text()
    assert ep.submit("p1", script, Resources(minutes=5)).spec_sha == rec.spec_sha                     # same job back
    with pytest.raises(EndpointError, match="different"):
        ep.submit("p1", "echo other\n", Resources(minutes=5))


@needs_setsid
def test_failure_one_job_at_a_time_and_cancel(tmp_path):
    hub = FakeHub()
    ep, clock = make(tmp_path, hub)
    ep.submit("bad", "exit 3\n", Resources(minutes=5))
    wait_done(ep, "bad")
    assert ep.status("bad").state == "failed" and "exit 3" in ep.status("bad").message
    ep.submit("long", "sleep 30\n", Resources(minutes=5))
    wait_pid(ep, "long")
    with pytest.raises(EndpointError, match="still runs"):
        ep.submit("other", "true\n", Resources(minutes=5))
    assert ep.cancel("long") == "cancelled"
    wait_done(ep, "long")
    assert ep.status("long").state == "cancelled"
    assert ep.stop_idle() and hub.srv is None                          # nothing runs: the server is stopped
    assert not ep.stop_idle()                                          # and not stopped twice


@needs_setsid
def test_wall_time_watchdog(tmp_path):
    hub = FakeHub()
    ep, _ = make(tmp_path, hub)
    line = ep.launch_line(ep._dir("x"), Resources(minutes=1))
    assert "sleep 60" in line and "kill -TERM -- -$$" in line and "timedout" in line
    # the watchdog for real, with a 2 s wall time: the job is killed, marked timed out and reported failed
    d = ep._dir("slow")
    (d / "out").mkdir(parents=True)
    (d / "job.sh").write_text("sleep 30\n")
    subprocess.run(["bash", "-c", line.replace("sleep 60", "sleep 2").replace(str(ep._dir("x")), str(d))], check=True)
    from mu2efastsim.endpoints import JobRecord
    ep.state.put(JobRecord("slow", "sha", "", str(d), ep.clock(), {"minutes": 1, "slice": "2g.20gb"}))
    wait_done(ep, "slow", timeout=15)
    st = ep.status("slow")
    assert st.state == "failed" and "timed out" in st.message


def test_status_without_exit_record(tmp_path):
    hub = FakeHub(run=False)                       # the typed line never runs: no pid, no done
    ep, clock = make(tmp_path, hub)
    ep.submit("lost", "true\n", Resources(minutes=5))
    assert ep.status("lost").state == "working"
    clock.t += 200
    assert ep.status("lost").state == "failed" and "never started" in ep.status("lost").message


def test_kit_renders_venv_runtime_for_eaf(tmp_path):
    hub = FakeHub(run=False)
    ep, _ = make(tmp_path, hub)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    body, res = kit.render({"task": "v4_response", "events": 1000, "seed": 1})
    assert "apptainer exec" not in body and "venv" in body and "CRVFS_LOOKUP_DIR" in body and "@p_" not in body
    subprocess.run(["bash", "-n"], input=body.encode(), check=True)
    uva = FastsimKit(endpoint=type("U", (), {"name": "u"})(), state_dir=tmp_path / "kit2", sif="/x.sif")
    assert "apptainer exec" in uva.render({"task": "v4_response", "events": 1000, "seed": 1})[0]


def test_workdir_must_be_shared(tmp_path):
    with pytest.raises(EndpointError, match="shared"):
        EAFEndpoint(workdir="/tmp/x", state_dir=tmp_path, hub_client=FakeHub())
