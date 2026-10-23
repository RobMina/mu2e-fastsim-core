"""chain_point fan-out against a fake Slurm site: sub-jobs, inbox, streaming with md5 checks, merge, refusals."""
import io
import json
import subprocess
import tarfile

import pytest

from mu2efastsim.endpoints import chainpoint as CP
from mu2efastsim.endpoints.kit import FastsimKit
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint


class Site:
    """Fake UVA: sbatch, squeue/sacct, cat of job files, srun --overlap steps, tar of out/."""

    def __init__(self):
        self.next_id, self.state, self.files, self.node_files, self.cmds = 5000, {}, {}, {}, []
        self.dirs = {}

    def __call__(self, cmd, stdin, timeout):
        s = cmd[-1]
        self.cmds.append(s)
        out = b""
        if "sbatch --parsable" in s:
            self.next_id += 1
            j = str(self.next_id)
            self.state[j] = "PENDING"
            self.dirs[j] = s.split("cd ")[1].split(" &&")[0].strip("'")
            out = f"{j}\n".encode()
        elif s.startswith("squeue"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|None\n".encode() if self.state.get(j) in ("PENDING", "RUNNING") else b""
        elif s.startswith("sacct") and "ElapsedRaw" in s:
            out = b"2026-10-21T18:00:00|2026-10-21T18:06:00|360\n"
        elif s.startswith("sacct"):
            j = s.split("-j ")[1].split()[0]
            out = f"{self.state[j]}|0:0\n".encode()
        elif s.startswith("cat "):
            path = s.split()[1].strip("'")
            out = self.files.get(path, "").encode()
        elif s.startswith("srun") and "touch" in s:
            self.node_files[s.split("touch ")[1].split()[0].strip("'\"")] = "ready"
        elif s.startswith("srun") and "test -f" in s:
            path = s.split("test -f ")[1].split()[0].strip("'\"")
            out = b"have\n" if path in self.node_files else b""
        elif s.startswith("scancel"):
            self.state[s.split()[1]] = "CANCELLED"
        elif "tar -cf" in s:
            d = s.split("cd ")[1].split(" &&")[0].strip("'")
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                for name, data in self.out_files(d).items():
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            out = buf.getvalue()
        elif s.startswith("set -e; C="):   # ensure_code
            pass
        return subprocess.CompletedProcess(cmd, 0, out, b"")

    def out_files(self, d):
        k = int(d.rsplit(".j", 1)[1])
        shard = {"generated_events": 50000 * 2, "wall_s": 120.0, "timers": {"events_with_steps": 4400 + k, "setup_s": 22.0},
                 "cl_type_counts": {"1": 3900, "2": 680, "3": 940, "4": 360}}
        return {"out/shard.json": json.dumps(shard).encode(), "out/wait.json": b'{"input_wait_s": 100}',
                "out/cleanup.txt": b"deleted\n", "out/metrics.json": b"{}", "slurm.out": b"ok\n"}

    def start(self, jid, ep_dir, node="udc-x"):
        self.state[jid] = "RUNNING"
        lib = f"/dev/shm/crvfs_lib_u_{node}"
        have = "have" if f"{lib}/ready" in self.node_files else "need"
        self.files[f"{ep_dir}/inbox"] = f"{node} /tmp/crvfs_kit_j{jid}_1 {lib} {have}\n"


class Source(CP.FnalSource):
    def __init__(self):
        super().__init__("gpvm", "/exp/data")
        self.data = {}

    def md5(self, rel):
        return "md5:" + rel


def make(tmp_path, cap=500.0):
    site = Site()
    streamed = []

    def stream_runner(src, host, remote, timeout):
        rel = src[-1].split("/exp/data/", 1)[1].strip("'")
        dest = remote.split("cat > ")[1].split()[0].strip("'\"")
        streamed.append((rel, dest))
        return "md5:" + rel + " " + dest

    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/scratch/u/kit", max_su_per_job=cap,
                          stream_runner=stream_runner)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    from mu2efastsim.endpoints.chainpoint import ChainPoint
    kit._cp = ChainPoint(kit, source=Source(), code="/scratch/u/code", background=False)
    return site, ep, kit, streamed


P = {"task": "chain_point", "shards": 3, "files_per_job": 2, "shard_offset": 18, "max_events": 0, "seed": 7,
     "wall_minutes": 45}


def test_validate_and_gen_files():
    p = CP.validate(P)
    assert CP.gen_files(p, 0) == ["g4ref_v1_tables/gen.001400_00000019.root", "g4ref_v1_tables/gen.001400_00000020.root"]
    assert CP.gen_files(p, 1)[0].endswith("_00000001.root")          # cycles over the 20 shards
    for bad in ({**P, "shards": 0}, {**P, "files_per_job": 2.5}, {**P, "seed": True}, {k: v for k, v in P.items() if k != "seed"}):
        with pytest.raises(ValueError):
            CP.validate(bad)


def test_check_and_budget(tmp_path):
    _, _, kit, _ = make(tmp_path)
    assert kit.check("c", P, [], [])["ok"]
    assert not kit.check("c", {**P, "shards": 99}, [], [])["ok"]
    _, _, kit2, _ = make(tmp_path / "b", cap=10.0)
    r = kit2.check("c", {**P, "wall_minutes": 120}, [], [])
    assert not r["ok"] and "SU" in r["message"]


def test_fanout_stream_merge(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    assert kit.submit("pt1", P, [], []) == {"handle": "pt1"}
    jobs = [ep.state.get(f"pt1.j{k}").job_id for k in range(3)]
    assert len(set(jobs)) == 3
    bodies = [site.dirs[j] for j in jobs]
    s = kit.status("pt1")
    assert s["state"] == "working" and s["progress"] == {"done": 0, "total": 3, "ok": 0} and not streamed
    site.start(jobs[0], ep.state.get("pt1.j0").remote_dir)
    kit.status("pt1")                               # streams sub-job 0 now that its inbox exists
    names = sorted(d.rsplit("/", 1)[1] for _, d in streamed)   # streamed in parallel: order not fixed
    assert [d for r, d in streamed if r.endswith("lib_prod_as_20files.root")] == ["/dev/shm/crvfs_lib_u_udc-x/lib_prod_as_20files.root"]
    assert names == sorted(["lib_prod_as_20files.root", "crvcalib_sim_best_v1_5_run1400.csv",
                     "crvphoton_sim_best_v1_5_run1400.csv", "mu2e_extracted_v04.gdml",
                     "gen.001400_00000019.root", "gen.001400_00000020.root"])
    assert all(d.startswith(f"/tmp/crvfs_kit_j{jobs[0]}_1/") for r, d in streamed if "lib_prod" not in r)
    assert any("touch" in c and "/ready" in c for c in site.cmds)
    for j, k in zip(jobs[1:], (1, 2)):           # same node: the library is already there
        site.start(j, ep.state.get(f"pt1.j{k}").remote_dir)
    kit.status("pt1")
    assert len(streamed) == 6 + 2 * 5             # sub-jobs 1 and 2 get CSVs, GDML and gen files, not the library
    for j in jobs:
        site.state[j] = "COMPLETED"
    s = kit.status("pt1")
    assert s["state"] == "completed" and s["progress"]["ok"] == 3
    r = kit.results("pt1")
    m = r["metrics"]
    assert m["accepted_events"] == 4400 * 3 + 3 and m["generated_events"] == 300000 and m["jobs_ok"] == 3
    assert m["accepted_per_gpu_hour"] == pytest.approx(m["accepted_events"] / (3 * 360 / 3600))
    assert m["accepted_per_gpu_hour_chain"] == pytest.approx(m["accepted_events"] / (3 * 120 / 3600))
    assert m["input_wait_s_mean"] == 100 and m["library_reused_jobs"] == 2 and m["clusters_per_generated_EX"] == pytest.approx(3 * 3900 / 300000)
    assert r["metadata"]["job_ids"] == ",".join(jobs)
    assert {f["name"] for f in r["files"]} >= {"pt1.j0/out/shard.json", "pt1.j2/out/cleanup.txt", "pt1.j1/slurm.out"}
    assert kit.submit("pt1", P, [], []) == {"handle": "pt1"} and len(site.dirs) == 3      # idempotent
    with pytest.raises(ValueError, match="different params"):
        kit.submit("pt1", {**P, "seed": 8}, [], [])


def test_md5_mismatch_cancels_the_subjob(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    ep.stream_runner = lambda src, host, remote, timeout: "bad"
    kit.submit("pt2", {**P, "shards": 1}, [], [])
    j = ep.state.get("pt2.j0").job_id
    site.start(j, ep.state.get("pt2.j0").remote_dir)
    kit.status("pt2")
    assert site.state[j] == "CANCELLED"
    s = kit.status("pt2")
    assert s["state"] == "cancelled" and "stream errors: 1" in s["message"]


def test_results_refused_without_cleanup(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    kit.submit("pt3", {**P, "shards": 1}, [], [])
    j = ep.state.get("pt3.j0").job_id
    site.start(j, ep.state.get("pt3.j0").remote_dir)
    kit.status("pt3")
    site.state[j] = "COMPLETED"
    orig = site.out_files
    site.out_files = lambda d: {k: v for k, v in orig(d).items() if "cleanup" not in k}
    with pytest.raises(ValueError, match="cleanup record"):
        kit.results("pt3")


def test_body_is_valid_bash_and_keeps_inputs_off_disk(tmp_path):
    _, ep, kit, _ = make(tmp_path)
    body, res = kit.chain_point.render(CP.validate(P), 1)
    script = ep.header("t", res) + body
    r = subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    code = "\n".join(line.split("#")[0] for line in body.splitlines())
    assert "/scratch" not in code.replace("/scratch/u/code", "") and "--seed 1007" in body
    assert res.gpus == 1 and res.minutes == 45


def test_library_streamed_once_per_node(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    kit.submit("pt4", {**P, "shards": 3, "files_per_job": 1}, [], [])
    jobs = [ep.state.get(f"pt4.j{k}").job_id for k in range(3)]
    for j, k, node in zip(jobs, range(3), ("nA", "nA", "nB")):
        site.start(j, ep.state.get(f"pt4.j{k}").remote_dir, node=node)
        kit.status("pt4")
    libs = sorted(d for r, d in streamed if r.endswith("lib_prod_as_20files.root"))
    assert libs == ["/dev/shm/crvfs_lib_u_nA/lib_prod_as_20files.root", "/dev/shm/crvfs_lib_u_nB/lib_prod_as_20files.root"]


def test_body_shares_and_releases_the_node_library(tmp_path):
    _, ep, kit, _ = make(tmp_path)
    body, _ = kit.chain_point.render(CP.validate(P), 0)
    assert "/dev/shm/crvfs_lib_${USER}_" in body and "flock 9" in body and "lib_release" in body
    assert body.count("trap '") >= 1 and "deleted node library" in body
    assert '--library "$S/lib_prod_as_20files.root"' in body


def test_gen_sets():
    p = CP.validate({**P, "gen_set": 1, "shard_offset": 92, "files_per_job": 3})
    assert CP.gen_files(p, 0) == ["crygen_v1_tables/gen.001400_00002099.root", "crygen_v1_tables/gen.001400_00002100.root",
                                  "crygen_v1_tables/gen.001400_00002001.root"]
    p = CP.validate({**P, "gen_set": 1, "shard_offset": 9, "files_per_job": 2})     # 2011 is missing: skipped
    assert CP.gen_files(p, 0) == ["crygen_v1_tables/gen.001400_00002010.root", "crygen_v1_tables/gen.001400_00002012.root"]
    every = {f for k in range(94) for f in CP.gen_files({**p, "files_per_job": 1, "shard_offset": 0}, k)}
    assert len(every) == 94 and not any(f"_0000{s}." in f for f in every for s in CP.CRYGEN_V1_MISSING)
    with pytest.raises(ValueError):
        CP.validate({**P, "gen_set": 1, "shard_offset": 94})
    assert CP.validate(P)["gen_set"] == 0                       # default: g4ref_v1
    with pytest.raises(ValueError):
        CP.validate({**P, "gen_set": 0, "shard_offset": 20})
    with pytest.raises(ValueError):
        CP.validate({**P, "gen_set": 2})


def test_wait_budget_scales_with_bytes(tmp_path):
    small = CP.validate(P)                                                   # 3 x 2 g4ref files: the floor
    assert CP.wait_budget_s(small, mbps=4.0) == CP.WAIT_FLOOR_S
    cg03 = CP.validate({**P, "gen_set": 1, "shards": 4, "files_per_job": 4, "shard_offset": 0})
    cg02 = CP.validate({**cg03, "files_per_job": 20})
    assert CP.wait_budget_s(cg03, mbps=4.0) == pytest.approx(600 + 4 * (1.007e9 + 3.9e6 + 4 * 5.6e7) / 4e6, abs=1)
    assert CP.wait_budget_s(cg02, mbps=4.0) > 2700 > CP.wait_budget_s(cg03, mbps=4.0) > CP.WAIT_FLOOR_S
    _, _, kit, _ = make(tmp_path)
    body, _ = kit.chain_point.render(cg02, 0)
    w = CP.wait_budget_s(cg02)
    assert f"-lt {w} ]" in body and f"no input arrived within {w} s" in body and "seq 360" not in body
    assert f'\\"wait_budget_s\\": {w}' in body


def test_check_refuses_wall_too_short_for_wait_and_chain(tmp_path):
    _, _, kit, _ = make(tmp_path)
    cg = {**P, "gen_set": 1, "shards": 4, "shard_offset": 0}
    assert kit.check("a", {**cg, "files_per_job": 4, "wall_minutes": 45}, [], [])["ok"]      # kitcg03
    r = kit.check("b", {**cg, "files_per_job": 20, "wall_minutes": 45}, [], [])             # kitcg02 at 45 min
    assert not r["ok"] and "wall_minutes" in r["message"] and "GB streamed" in r["message"]
    assert kit.check("c", {**cg, "files_per_job": 20, "wall_minutes": 120}, [], [])["ok"]
    assert kit.check("d", {**P, "shards": 4, "files_per_job": 20, "shard_offset": 0}, [], [])["ok"]   # fcp01
    small = {**cg, "files_per_job": 20, "wall_minutes": 45, "max_events": 10_000}
    assert CP.chain_estimate_s(CP.validate(small)) < CP.chain_estimate_s(CP.validate({**small, "max_events": 0}))
