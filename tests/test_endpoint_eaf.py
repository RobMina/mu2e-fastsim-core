"""EAF endpoint against a fake JupyterHub; the launch line runs for real in a local bash (no EAF, no network)."""
import json
import re
import shutil
import subprocess
import time

import pytest

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.eaf import EAFEndpoint, slice_of
from mu2efastsim.endpoints.kit import FastsimKit

pytestmark = pytest.mark.usefixtures("site_env")   # placeholder site variables (tests/conftest.py)


class FakeHub:
    """Server life cycle in memory; `send` runs the typed line in a local bash, as the EAF terminal would.
    available: slices that start; others never become ready."""

    def __init__(self, available=("2g.20gb",), run=True):
        self.available, self.run = set(available), run
        self.srv, self.starts, self.stops, self.sent, self.pings = None, [], 0, [], 0

    def server(self):
        return self.srv

    def start(self, options):
        self.starts.append(slice_of(options))
        ok = slice_of(options) in self.available
        self.srv = {"ready": ok, "pending": None if ok else "spawn", "url": "/user/u/", "user_options": options}

    def stop(self):
        self.stops += 1
        self.srv = None

    def user_name(self):
        return "u"

    def call(self, method, path, body=None):
        if path.endswith("/activity"):
            self.pings += 1
        return 200, None

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
    ep.submit("other", "true\n", Resources(minutes=5))                 # slice busy: queued, not refused
    st = ep.status("other")
    assert st.state == "working" and st.message.startswith("queued")
    assert ep.cancel("long") == "cancelled"
    wait_done(ep, "long")
    assert ep.status("long").state == "cancelled"                      # this call also launches "other"
    wait_done(ep, "other")
    assert ep.status("other").state == "completed"
    assert ep.stop_idle() and hub.srv is None                          # nothing runs or waits: the server is stopped
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
        EAFEndpoint(workdir="/elsewhere/x", state_dir=tmp_path, hub_client=FakeHub())


def test_queue_order_keepalive_and_linger_stop(tmp_path):
    hub = FakeHub(available=("4g.40gb",), run=False)   # first slice starts; jobs never run: files written by hand
    ep, clock = make(tmp_path, hub, linger_s=120)
    for h in ("a", "b", "c"):
        ep.submit(h, f"echo {h}\n", Resources(minutes=5))
        clock.t += 1
    assert [r.handle for r in ep._queue()] == ["b", "c"]
    assert ep.status("c").message == "queued (1 ahead)"
    (ep._dir("a") / "job.pid").write_text("1")
    assert ep.status("a").state == "working" and hub.pings == 1       # keepalive posted while working
    (ep._dir("a") / "job.done").write_text("exit=0\n")
    assert ep.status("a").state == "completed"                        # and "b" is launched in the same call
    assert ep.state.get("b").resources.get("launched_at") and not ep.state.get("c").resources.get("launched_at")
    assert ep.cancel("c") == "cancelled" and ep.status("c").state == "cancelled"
    (ep._dir("b") / "job.pid").write_text("2"); (ep._dir("b") / "job.done").write_text("exit=0\n")
    assert ep.status("b").state == "completed"
    assert hub.srv is not None                                        # within linger: the server stays
    clock.t += 121
    ep.after_results("b")
    assert hub.srv is None and hub.stops == 1                         # past linger, nothing queued: stopped


def test_server_lifetime_cap_and_cull_report(tmp_path):
    hub = FakeHub(run=False)
    ep, clock = make(tmp_path, hub, max_age_s=3600)
    with pytest.raises(EndpointError, match="lifetime"):
        ep.submit("long", "true\n", Resources(minutes=90))
    ep.submit("j", "true\n", Resources(minutes=5))
    (ep._dir("j") / "job.pid").write_text("1"); (ep._dir("j") / "job.alive").write_text(str(int(clock.t)))
    clock.t += 300
    hub.srv = None                                                    # the hub culled the server
    st = ep.status("j")
    assert st.state == "failed" and "server stopped" in st.message and "last heartbeat 300 s ago" in st.message


def test_kit_fnal_tasks_only_on_fnal_endpoints(tmp_path):
    hub = FakeHub(run=False)
    ep, _ = make(tmp_path, hub)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    for task, p in (("chain", {"shards": 2, "max_events": 20000.0, "seed": 1}), ("regression", {"events": 5000, "seed": 1})):
        body, _ = kit.render({"task": task, **p})
        assert not re.search(r"@[a-z_]+@", body)                       # no unfilled kit placeholder
        assert ep.fnal_data in body
        if task == "chain":
            assert "--max-events 20000 " in body and "v2_fastC_v1/ref." in body
        subprocess.run(["bash", "-n"], input=body.encode(), check=True)
    uva = FastsimKit(endpoint=type("U", (), {"name": "uva"})(), state_dir=tmp_path / "kit2", sif="/x.sif")
    c = uva.check("n", {"task": "chain", "shards": 1, "max_events": 10, "seed": 1}, [], [])
    assert not c["ok"] and "FNAL storage" in c["message"]


def test_concurrent_submits_start_one_server(tmp_path):
    """Mu2eBO submits independent steps from threads: one server start, the second job queued."""
    import threading
    hub = FakeHub(available=("4g.40gb",), run=False)
    lock = threading.Lock()
    orig = hub.start

    def start(options):                        # a second start while one exists is the hub's HTTP 500
        with lock:
            assert hub.srv is None, "second server start"
            orig(options)
    hub.start = start
    ep, _ = make(tmp_path, hub)
    errs = []

    def go(h):
        try:
            ep.submit(h, f"echo {h}\n", Resources(minutes=5))
        except Exception as e:                 # noqa: BLE001
            errs.append(e)
    th = [threading.Thread(target=go, args=(h,)) for h in ("s1", "s2")]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs and len(hub.starts) == 1
    launched = [h for h in ("s1", "s2") if ep.state.get(h).resources.get("launched_at")]
    assert len(launched) == 1 and len(ep._queue()) == 1


def test_watcher_stops_after_linger(tmp_path):
    hub = FakeHub(available=("4g.40gb",), run=False)
    ep, clock = make(tmp_path, hub, linger_s=120)
    ep.submit("w", "true\n", Resources(minutes=5))
    (ep._dir("w") / "job.pid").write_text("1"); (ep._dir("w") / "job.done").write_text("exit=0\n")
    assert ep.status("w").state == "completed" and hub.srv is not None          # inside the linger: kept
    spawned = []
    ep._spawn_watcher = lambda: spawned.append(1)
    ep.after_results("w")
    assert spawned == [1]                                                       # nobody may call again: a watcher
    assert ep.wait_and_stop() and hub.srv is None                               # sleeps out the linger, then stops
    assert clock.t >= ep.state.get("w").resources["ended_at"] + 120


def test_kit_beam_point_renders_on_eaf(tmp_path):
    """beam_point (umbrella beam/, beamfastsim.kit_task) is registered when beam/ is importable: FNAL-only, valid bash."""
    from mu2efastsim.endpoints import kit as kitmod
    if "beam_point" not in kitmod.TASKS:
        pytest.skip("beamfastsim not on PYTHONPATH")
    hub = FakeHub(run=False)
    ep, _ = make(tmp_path, hub)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    body, res = kit.render({"task": "beam_point", "geom": "move_extras12.txt", "muons": 105350.0, "seed": 1.0,
                            "repeat": 0.0})
    assert not re.search(r"@[a-z_]+@", body) and ep.fnal_data in body and "--n 105350 " in body
    assert res.gpus == 1 and "beam_point" in kitmod.FNAL_TASKS
    subprocess.run(["bash", "-n"], input=body.encode(), check=True)


def test_kit_veto_c4_deadtime_and_gate_mode_default(tmp_path):
    """veto_c4: valid bash on EAF with and without gate_mode (default 0 = event gate); deadtime metrics declared."""
    from mu2efastsim.endpoints import kit as kitmod
    hub = FakeHub(run=False)
    ep, _ = make(tmp_path, hub)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    p = {"task": "veto_c4", "selection": "run1a_dev", "dt_lo": 0.0, "dt_hi": 150.0, "min_pe": 0.0, "min_layers": 0,
         "gate_start_lo": 420.0, "gate_end_hi": 1700.0, "pmin": 85.0, "pmax": 200.0}
    for extra in ({}, {"gate_mode": 1}):
        body, res = kit.render({**p, **extra})
        assert not re.search(r"@[a-z_]+@", body) and ep.fnal_data in body and res.gpus == 0
        assert "crvfastsim.veto.deadtime" in body and "deadtime_run1a/ce_mix1bb_dev_sel.parquet" in body
        assert f"awk -v g={extra.get('gate_mode', 0)} " in body
        subprocess.run(["bash", "-n"], input=body.encode(), check=True)
    assert {"deadtime", "deadtime_err", "deadtime_full"} <= set(kitmod.TASKS["veto_c4"].metrics)
    with pytest.raises(ValueError):
        kit.render({**p, "bogus": 1})


def test_kit_crv_response_renders_on_eaf(tmp_path):
    """crv_response (crv package, crvfastsim.kit_task): registered when crvfastsim is importable; FNAL-only, valid bash,
    every chain.KNOBS knob a param (absent = nominal), knob values reach reco-r3 --knob, unknown params refused."""
    from mu2efastsim.endpoints import kit as kitmod
    if "crv_response" not in kitmod.TASKS:
        pytest.skip("crvfastsim not on PYTHONPATH")
    from crvfastsim import chain
    hub = FakeHub(run=False)
    ep, _ = make(tmp_path, hub)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    assert set(chain.KNOBS) <= set(kitmod.TASKS["crv_response"].params) and "crv_response" in kitmod.FNAL_TASKS
    nominal, res = kit.render({"task": "crv_response", "seed": 1.0})
    assert not re.search(r"@[a-z_0-9]+@", nominal) and ep.fnal_data in nominal and res.gpus == 1
    assert "corr_P4_v2mod.json" in nominal and "crvsteps.001400_$q.root" in nominal and "CRVFS_LOOKUP_DIR" in nominal
    subprocess.run(["bash", "-n"], input=nominal.encode(), check=True)
    body, _ = kit.render({"task": "crv_response", "seed": 2.0, "speed_EX": 1.08, "atten_T10": 0.55, "period": "P3"})
    assert 'KN+=("speed_EX="1.08)' in body and 'KN+=("atten_T10="0.55)' in body
    subprocess.run(["bash", "-n"], input=body.encode(), check=True)
    with pytest.raises(ValueError):
        kit.render({"task": "crv_response", "seed": 1.0, "speed_EX9": 1.0})
    uva = FastsimKit(endpoint=type("U", (), {"name": "u"})(), state_dir=tmp_path / "kit2", sif="/x.sif")
    with pytest.raises(ValueError):
        uva.render({"task": "crv_response", "seed": 1.0})


def test_kit_crv_response_knob_lines_run(tmp_path):
    """The rendered knob block builds the --knob words: set knobs only, in KNOBS order, values passed through."""
    from mu2efastsim.endpoints import kit as kitmod
    if "crv_response" not in kitmod.TASKS:
        pytest.skip("crvfastsim not on PYTHONPATH")
    hub = FakeHub(run=False)
    ep, _ = make(tmp_path, hub)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    body, _ = kit.render({"task": "crv_response", "seed": 1.0, "speed_EX": 1.08, "atten_T1": 0.58})
    block = body[body.index("KN=()"):body.index("KA=()")]
    out = subprocess.run(["bash", "-c", block + 'printf "%s\\n" "${KN[@]}"'], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["atten_T1=0.58", "speed_EX=1.08"]
