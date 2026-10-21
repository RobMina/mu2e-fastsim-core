"""Endpoint adapter and fastsimkit against a fake Slurm site (no ssh, no cluster)."""
import io
import json
import shutil
import subprocess
import tarfile

import pytest

from mu2efastsim.endpoints import EndpointError, Resources
from mu2efastsim.endpoints.kit import FastsimKit
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint, map_state, su_estimate


class FakeSite:
    """Answers the shell snippets SlurmSSHEndpoint sends, from an in-memory queue."""

    def __init__(self):
        self.next_id, self.queue, self.acct, self.scripts, self.calls = 1000, {}, {}, {}, []

    def __call__(self, cmd, stdin, timeout):
        s = cmd[-1]
        self.calls.append(s)
        out = b""
        if "sbatch --parsable" in s:
            self.next_id += 1
            jid = str(self.next_id)
            self.queue[jid] = "PENDING"
            self.scripts[jid] = stdin.decode()
            out = f"{jid}\n".encode()
        elif s.startswith("squeue"):
            jid = s.split("-j ")[1].split()[0]
            out = f"{self.queue[jid]}|None\n".encode() if jid in self.queue else b""
        elif s.startswith("sacct"):
            jid = s.split("-j ")[1].split()[0]
            out = (self.acct[jid] + "\n").encode() if jid in self.acct else b""
        elif s.startswith("scancel"):
            jid = s.split()[1]
            self.queue.pop(jid, None)
            self.acct[jid] = "CANCELLED by 1|0:0"
        elif "tar -cf" in s:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in (("out/metrics.json", json.dumps({"events_per_s": 20020.0, "first_call_s": 58.3,
                                                                    "events": 200000}).encode()),
                                   ("slurm.out", b"done\n")):
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        return subprocess.CompletedProcess(cmd, 0, out, b"")

    def finish(self, jid, state="COMPLETED", code="0:0"):
        self.queue.pop(jid, None)
        self.acct[jid] = f"{state}|{code}"


@pytest.fixture
def site_ep(tmp_path):
    site = FakeSite()
    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/scratch/u/kit", max_su_per_job=100)
    return site, ep


def test_map_state():
    assert map_state("PENDING") == "working"
    assert map_state("COMPLETED", "0:0") == "completed"
    assert map_state("COMPLETED", "3:0") == "failed"
    assert map_state("CANCELLED by 123") == "cancelled"
    assert map_state("TIMEOUT") == "failed"
    with pytest.raises(EndpointError):
        map_state("WEIRD")


def test_su_estimate_matches_measured_jobs():
    assert su_estimate(Resources(cores=8, gpus=1, minutes=20)) == pytest.approx(19, abs=2)    # job 21703670: 19 SU
    assert su_estimate(Resources(cores=16, gpus=1, minutes=24)) == pytest.approx(25, abs=2)   # job 21714859: 25 SU


def test_submit_status_fetch(site_ep, tmp_path):
    site, ep = site_ep
    rec = ep.submit("p1.bench", "echo hi\n", Resources(minutes=30))
    script = site.scripts[rec.job_id]
    assert "#SBATCH -A ldmxuva" in script and "#SBATCH --gres=gpu:a100:1" in script and script.endswith("echo hi\n")
    assert ep.status("p1.bench").state == "working"
    site.finish(rec.job_id)
    st = ep.status("p1.bench")
    assert st.state == "completed" and st.job_id == rec.job_id
    n = len(site.calls)
    assert ep.status("p1.bench").state == "completed" and len(site.calls) == n   # final state cached, no ssh
    files = ep.fetch("p1.bench", tmp_path / "f")
    assert {p.name for p in files} == {"metrics.json", "slurm.out"}


def test_submit_idempotent_and_refusals(site_ep):
    site, ep = site_ep
    a = ep.submit("p2.bench", "x\n", Resources(minutes=30))
    assert ep.submit("p2.bench", "x\n", Resources(minutes=30)).job_id == a.job_id and len(site.scripts) == 1
    with pytest.raises(EndpointError, match="different"):
        ep.submit("p2.bench", "y\n", Resources(minutes=30))
    with pytest.raises(EndpointError, match="exceeds max_su_per_job"):
        ep.submit("big", "x\n", Resources(gpus=8, minutes=600))
    with pytest.raises(EndpointError, match="bad handle"):
        ep.submit("../evil", "x\n", Resources(minutes=5))


def test_cancel_and_failure(site_ep):
    site, ep = site_ep
    a = ep.submit("p3", "x\n", Resources(minutes=10))
    assert ep.cancel("p3") == "cancelled" and ep.status("p3").state == "cancelled"
    b = ep.submit("p4", "x\n", Resources(minutes=10))
    site.finish(b.job_id, "OUT_OF_MEMORY", "0:125")
    assert ep.status("p4").state == "failed"


def test_kit_contract_roundtrip(site_ep, tmp_path):
    site, ep = site_ep
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/scratch/u/m.sif")
    p = {"task": "reco_bench", "events": 200000, "seed": 7}
    assert kit.check("c1.bench", p, [], [])["ok"]
    assert not kit.check("c1.bench", {"task": "nope"}, [], [])["ok"]
    assert not kit.check("c1.bench", {"task": "reco_bench", "events": 1}, [], [])["ok"]
    assert kit.submit("c1.bench", p, [], []) == {"handle": "c1.bench"}
    jid = ep.state.get("c1.bench").job_id
    assert "--max-events 200000" in site.scripts[jid] and "/scratch/u/m.sif" in site.scripts[jid]
    s = kit.status("c1.bench")
    assert s["state"] == "working" and s["poll_ms"] > 0 and set(s) == {"state", "message", "poll_ms", "progress"}
    with pytest.raises(ValueError):
        kit.results("c1.bench")
    site.finish(jid)
    r = kit.results("c1.bench")
    assert r["metrics"] == {"events_per_s": 20020.0, "first_call_s": 58.3, "events": 200000.0}
    assert r["metadata"]["job_id"] == jid and all(f["uri"].startswith("file://") for f in r["files"])
    d = kit.describe()
    assert "task" in d["params"] and "events_per_s" in d["metrics"] and d["accepts_lists"] is False


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_rendered_jobs_are_valid_bash(site_ep, tmp_path):
    from mu2efastsim.endpoints.kit import TASKS
    _, ep = site_ep
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    for name, task in TASKS.items():
        params = {"task": name, **{p: 1 for p in task.params}}
        body, res = kit.render(params)
        assert "{{" not in body and "}}" not in body and "@" not in body.replace("$@", "")
        script = ep.header("t", res) + body
        r = subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True)
        assert r.returncode == 0, (name, r.stderr.decode())


def test_table_tasks_stage_and_delete(site_ep, tmp_path):
    from mu2efastsim.endpoints.kit import TABLE_TASKS, TASKS
    _, ep = site_ep
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    for name in TABLE_TASKS:
        params = {"task": name, **{p: 1 for p in TASKS[name].params}}
        body, _ = kit.render(params)
        assert "cvmfsget" in body and "trap 'rm -rf" in body and "cleanup.txt" in body
        code = "\n".join(line.split("#")[0] for line in body.splitlines())
        assert "chmod 700" in code and "/scratch" not in code   # tables never on scratch


def test_results_require_cleanup_record_for_table_tasks(site_ep, tmp_path):
    site, ep = site_ep
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    kit.submit("r1.resp", {"task": "response_bench", "rows": 10, "sipms": 10, "seed": 1}, [], [])
    site.finish(ep.state.get("r1.resp").job_id)
    with pytest.raises(ValueError, match="cleanup record"):
        kit.results("r1.resp")       # the fake site's tar has no out/cleanup.txt
