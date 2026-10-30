"""chain_point fan-out against a fake Slurm site: sub-jobs, inbox, streaming with md5 checks, merge, refusals."""
import io
import hashlib
import json
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mu2efastsim.endpoints import chainpoint as CP
from mu2efastsim.endpoints.kit import FastsimKit
from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint


class Site:
    """Fake UVA: sbatch, squeue/sacct, cat of job files, srun --overlap steps, tar of out/."""

    def __init__(self):
        self.next_id, self.state, self.files, self.node_files, self.cmds = 5000, {}, {}, {}, []
        self.dirs = {}
        self.kit = None

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
        elif s.startswith("python3 -c"):
            out = json.dumps({"md5": "a" * 32, "sha256": "b" * 64, "size": 1}).encode()
        elif s.startswith("srun") and "VERIFIED" in s:
            out = b"VERIFIED\n"
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
        return {"out/artifact-manifest.json": json.dumps(self.kit.chain_point.record(d.rsplit("/", 1)[1].rsplit(".j", 1)[0])["manifest"]).encode(),
                "out/shard.json": json.dumps(shard).encode(), "out/wait.json": b'{"input_wait_s": 100}',
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
        return hashlib.md5(rel.encode()).hexdigest()


    def fingerprint(self, rel):
        return {"md5": self.md5(rel), "sha256": hashlib.sha256(rel.encode()).hexdigest(), "size": len(rel)}


@pytest.fixture(autouse=True)
def fake_code(monkeypatch):
    from mu2efastsim.endpoints import snapshots
    monkeypatch.setattr(CP, "prepare_code", lambda ep, base, ref: {"path": base + ".snapshot", "umbrella_commit": "a" * 40})
    monkeypatch.setattr(snapshots, "code_guard", lambda _: "true\n")


def make(tmp_path, cap=500.0):
    site = Site()
    streamed = []

    def stream_runner(src, host, remote, timeout):
        rel = src[-1].split("/exp/data/", 1)[1].strip("'")
        dest = remote.split("cat > ")[1].split()[0].strip("'\"")
        streamed.append((rel, dest))
        return hashlib.md5(rel.encode()).hexdigest() + " " + dest

    ep = SlurmSSHEndpoint(runner=site, state_dir=tmp_path / "jobs", workdir="/scratch/u/kit", max_su_per_job=cap,
                          stream_runner=stream_runner)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit", sif="/x/m.sif")
    from mu2efastsim.endpoints.chainpoint import ChainPoint
    kit._cp = ChainPoint(kit, source=Source(), code="/scratch/u/code", background=False)
    site.kit = kit
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
    shared = sorted(d for r, d in streamed if r.startswith("tier1b_"))      # the default set: library + sidecars
    assert shared == sorted(f"/dev/shm/crvfs_lib_u_udc-x/{f}" for f in CP.TIER1B_SETS["strict_v4"][1])
    assert names == sorted(list(CP.TIER1B_SETS["strict_v4"][1]) + ["crvcalib_sim_best_v1_5_run1400.csv",
                     "crvphoton_sim_best_v1_5_run1400.csv", "mu2e_extracted_v04.gdml",
                     "gen.001400_00000019.root", "gen.001400_00000020.root"])
    assert all(d.startswith(f"/tmp/crvfs_kit_j{jobs[0]}_1/") for r, d in streamed if not r.startswith("tier1b_"))
    assert any("touch" in c and "/ready" in c for c in site.cmds)
    for j, k in zip(jobs[1:], (1, 2)):           # same node: the library is already there
        site.start(j, ep.state.get(f"pt1.j{k}").remote_dir)
    kit.status("pt1")
    assert len(streamed) == 10 + 2 * 5            # sub-jobs 1 and 2 get CSVs, GDML and gen files, not the library
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


@pytest.mark.parametrize("tier1b,lib", [(0, "tier1b_strict_v2/library.root"), (1, "tier1b_v7/lib_prod_as_20files.root")])
def test_library_streamed_once_per_node(tmp_path, tier1b, lib):
    site, ep, kit, streamed = make(tmp_path)
    kit.submit("pt4", {**P, "shards": 3, "files_per_job": 1, "tier1b": tier1b}, [], [])
    jobs = [ep.state.get(f"pt4.j{k}").job_id for k in range(3)]
    for j, k, node in zip(jobs, range(3), ("nA", "nA", "nB")):
        site.start(j, ep.state.get(f"pt4.j{k}").remote_dir, node=node)
        kit.status("pt4")
    name = lib.rsplit("/", 1)[1]
    libs = sorted(d for r, d in streamed if r == lib)
    assert libs == [f"/dev/shm/crvfs_lib_u_nA/{name}", f"/dev/shm/crvfs_lib_u_nB/{name}"]
    assert sum(r.startswith("tier1b_") for r, _ in streamed) == 2 * len(CP.library_files(CP.validate({**P, "tier1b": tier1b})))


def test_body_shares_and_releases_the_node_library(tmp_path):
    _, ep, kit, _ = make(tmp_path)
    body, _ = kit.chain_point.render(CP.validate(P), 0)
    assert "/dev/shm/crvfs_lib_${USER}_" in body and "flock 9" in body and "lib_release" in body
    assert body.count("trap '") >= 1 and "deleted node library" in body
    assert '--tier1b strict_v4 --tier1b-dir "$S"' in body and "--library" not in body
    v3, _ = kit.chain_point.render(CP.validate({**P, "tier1b": 3}), 0)
    assert '--tier1b strict_v3 --tier1b-dir "$S"' in v3
    old, _ = kit.chain_point.render(CP.validate({**P, "tier1b": 2}), 0)
    assert '--tier1b strict_v2 --tier1b-dir "$S"' in old
    legacy, _ = kit.chain_point.render(CP.validate({**P, "tier1b": 1}), 0)
    assert '--library "$S/lib_prod_as_20files.root"' in legacy and "--tier1b" not in legacy
    named, _ = kit.chain_point.render(CP.validate({**P, "gen_set": 3, "run_config": 3, "shard_offset": 0}), 0)
    assert '--library "$S/lib_prod_as_20files.root"' in named          # named configurations: legacy only


def test_tier1b_sets():
    assert CP.tier1b_set(CP.validate(P)) == "strict_v4" and CP.tier1b_set(CP.validate({**P, "tier1b": 1})) == "legacy"
    assert CP.tier1b_set(CP.validate({**P, "tier1b": 3})) == "strict_v3"
    assert CP.TIER1B_SETS["strict_v4"][1] == CP.TIER1B_SETS["strict_v3"][1]
    assert CP.tier1b_set(CP.validate({**P, "tier1b": 2})) == "strict_v2"
    assert CP.library_files(CP.validate(P))[3] == "tier1b_strict_v2/templates_v3.npz"
    assert CP.library_files(CP.validate({**P, "tier1b": 2}))[3] == "tier1b_strict_v2/templates_v2.npz"
    assert CP.tier1b_set({k: v for k, v in CP.validate(P).items() if k != "tier1b"}) == "legacy"   # older records
    assert CP.library_files(CP.validate({**P, "tier1b": 1})) == [CP.LIBRARY]
    assert CP.library_files(CP.validate(P))[0] == "tier1b_strict_v2/library.root"
    with pytest.raises(ValueError):
        CP.validate({**P, "tier1b": 4})


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
    assert CP.wait_budget_s(cg03, mbps=4.0) == pytest.approx(600 + 4 * (1.1699e9 + 3.9e6 + 4 * 5.6e7) / 4e6, abs=1)
    assert CP.wait_budget_s({**cg03, "tier1b": 2}, mbps=4.0) == pytest.approx(600 + 4 * (1.0834e9 + 3.9e6 + 4 * 5.6e7) / 4e6, abs=1)
    assert CP.wait_budget_s({**cg03, "tier1b": 1}, mbps=4.0) == pytest.approx(600 + 4 * (1.007e9 + 3.9e6 + 4 * 5.6e7) / 4e6, abs=1)
    assert CP.wait_budget_s(cg02, mbps=4.0) > 2700 > CP.wait_budget_s(cg03, mbps=4.0) > CP.WAIT_FLOOR_S
    _, _, kit, _ = make(tmp_path)
    body, _ = kit.chain_point.render(cg02, 0)
    w = CP.wait_budget_s(cg02)
    assert f"-lt {w} ]" in body and f"no input arrived within {w} s" in body and "seq 360" not in body
    assert f'\\"wait_budget_s\\": {w}' in body


def test_check_refuses_wall_too_short_for_wait_and_chain(tmp_path):
    _, _, kit, _ = make(tmp_path)
    cg = {**P, "gen_set": 1, "shards": 4, "shard_offset": 0}
    assert kit.check("a", {**cg, "files_per_job": 4, "wall_minutes": 50}, [], [])["ok"]      # kitcg03
    # the strict_v3 set (v3 templates, +86 MB) needs ~46.3 min of wait + chain for this point; strict_v2 ~44.9
    assert not kit.check("a2", {**cg, "files_per_job": 4, "wall_minutes": 45}, [], [])["ok"]
    assert kit.check("a3", {**cg, "files_per_job": 4, "wall_minutes": 45, "tier1b": 2}, [], [])["ok"]
    r = kit.check("b", {**cg, "files_per_job": 20, "wall_minutes": 45}, [], [])             # kitcg02 at 45 min
    assert not r["ok"] and "wall_minutes" in r["message"] and "GB streamed" in r["message"]
    assert kit.check("c", {**cg, "files_per_job": 20, "wall_minutes": 120}, [], [])["ok"]
    assert kit.check("d", {**P, "shards": 4, "files_per_job": 20, "shard_offset": 0}, [], [])["ok"]   # fcp01
    small = {**cg, "files_per_job": 20, "wall_minutes": 45, "max_events": 10_000}
    assert CP.chain_estimate_s(CP.validate(small)) < CP.chain_estimate_s(CP.validate({**small, "max_events": 0}))


def test_run_configs(tmp_path):
    """run_config picks the GDML, lookup tables and conditions; a gen set belongs to one CRY configuration."""
    _, ep, kit, _ = make(tmp_path)
    assert CP.validate(P)["run_config"] == 0
    with pytest.raises(ValueError):
        CP.validate({**P, "run_config": 3})                         # extracted g4ref primaries with the Run 2 CRV
    with pytest.raises(ValueError):
        CP.validate({**P, "gen_set": 3, "shard_offset": 0})        # default-CRY primaries with the extracted CRV
    p = CP.validate({**P, "gen_set": 3, "run_config": 3, "shard_offset": 0})
    assert CP.gen_files(p, 0)[0] == "crygen_run2_v1_tables/gen.001400_00004001.root"
    assert CP.small_inputs(p) == ["geometry/mu2e_run2_v10.gdml"]
    body, _ = kit.chain_point.render(p, 0)
    assert "--run-config run2_v10" in body and "--calib nominal" in body and "mu2e_run2_v10.gdml" in body
    assert "LookupTable_4550_0" in body and "LookupTable_1550_1" in body and "crvcalib" not in body
    r = subprocess.run(["bash", "-n"], input=(ep.header("t", _) + body).encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    b0, _ = kit.chain_point.render(CP.validate(P), 0)
    assert "--run-config" not in b0 and "crvcalib_sim_best_v1_5_run1400.csv" in b0 and "LookupTable_4550_0" not in b0
    p1 = CP.validate({**P, "gen_set": 2, "run_config": 2, "shard_offset": 0})
    assert CP.small_inputs(p1) == ["geometry/mu2e_run1a_v01.gdml"]


def test_chain_flags(tmp_path):
    _, _, kit, _ = make(tmp_path)
    b, _ = kit.chain_point.render(CP.validate({**P, "chain_flags": 3}), 0)
    assert "--prefetch --fused-response" in b
    b0, _ = kit.chain_point.render(CP.validate(P), 0)
    assert "--prefetch" not in b0 and "--fused-response" not in b0
    with pytest.raises(ValueError):
        CP.validate({**P, "chain_flags": 4})


@pytest.mark.parametrize("key", CP.PARAMS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10 ** 400, -(10 ** 400)],
                         ids=["nan", "inf", "negative-inf", "huge-int", "negative-huge-int"])
def test_validate_rejects_nonfinite_and_huge_numbers(key, value):
    with pytest.raises(ValueError, match=key):
        CP.validate({**P, key: value})


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), -float("inf"), "invalid", "", True,
                                  10 ** 400, 5e-324],
                         ids=["zero", "negative", "nan", "inf", "negative-inf", "text", "empty", "bool",
                              "huge-int", "underflow-rate"])
def test_wait_budget_rejects_invalid_stream_rate(monkeypatch, rate):
    monkeypatch.setenv("FASTSIMKIT_STREAM_MBPS", "4")
    with pytest.raises(ValueError, match="FASTSIMKIT_STREAM_MBPS"):
        CP.wait_budget_s(CP.validate(P), mbps=rate)
    monkeypatch.setenv("FASTSIMKIT_STREAM_MBPS", str(rate))
    with pytest.raises(ValueError, match="FASTSIMKIT_STREAM_MBPS"):
        CP.wait_budget_s(CP.validate(P))


def test_wait_budget_default_environment_and_override(monkeypatch):
    p = CP.validate({**P, "shards": 12})
    monkeypatch.delenv("FASTSIMKIT_STREAM_MBPS", raising=False)
    assert CP.wait_budget_s(p) == CP.wait_budget_s(p, mbps=CP.STREAM_MBPS)
    monkeypatch.setenv("FASTSIMKIT_STREAM_MBPS", "2")
    assert CP.wait_budget_s(p) == CP.wait_budget_s(p, mbps=2)
    assert CP.wait_budget_s(p) > CP.wait_budget_s(p, mbps=4)
    monkeypatch.setenv("FASTSIMKIT_STREAM_MBPS", "invalid")
    assert CP.wait_budget_s(p, mbps=4) > CP.WAIT_FLOOR_S


@pytest.mark.parametrize("changes,cap,rate,streaming,message", [
    ({"wall_minutes": 10}, 500, "4", True, "wall_minutes"),
    ({"gen_set": 1, "shards": 4, "files_per_job": 20, "shard_offset": 0}, 500, "4", True, "GB streamed"),
    ({}, 10, "4", True, "SU"),
    ({"seed": float("nan")}, 500, "4", True, "seed"),
    ({"seed": float("inf")}, 500, "4", True, "seed"),
    ({"seed": -float("inf")}, 500, "4", True, "seed"),
    ({"seed": 10 ** 400}, 500, "4", True, "seed"),
    ({}, 500, "0", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "-1", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "nan", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "inf", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "-inf", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "1e400", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "5e-324", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "invalid", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "", True, "FASTSIMKIT_STREAM_MBPS"),
    ({}, 500, "4", False, "needs an endpoint"),
], ids=["short-wall", "large-point", "su-cap", "nan", "inf", "negative-inf", "huge-int", "zero-rate",
        "negative-rate", "nan-rate", "inf-rate", "negative-inf-rate", "overflow-rate", "underflow-rate",
        "text-rate", "empty-rate", "unsupported-endpoint"])
def test_check_and_submit_share_admission_without_side_effects(tmp_path, monkeypatch, changes, cap, rate,
                                                               streaming, message):
    monkeypatch.setenv("FASTSIMKIT_STREAM_MBPS", rate)
    ep = Mock(spec=["name", "max_su_per_job", "submit", "_sh"] + (["stream_in"] if streaming else []))
    ep.name, ep.max_su_per_job = "fake", cap
    source = Mock(spec=CP.FnalSource)
    cp = CP.ChainPoint(SimpleNamespace(ep=ep, out_root=tmp_path / "out", sif="/fake.sif"), source=source)
    render, ensure_code, ensure_streaming = Mock(), Mock(), Mock()
    monkeypatch.setattr(cp, "render", render)
    monkeypatch.setattr(cp, "ensure_code", ensure_code)
    monkeypatch.setattr(cp, "ensure_streaming", ensure_streaming)
    before = sorted(tmp_path.rglob("*"))
    params = {**P, **changes}
    checked = cp.check("rejected", params)
    assert not checked["ok"] and message in checked["message"]
    with pytest.raises(ValueError) as exc:
        cp.submit("rejected", params)
    assert str(exc.value) == checked["message"]
    assert sorted(tmp_path.rglob("*")) == before
    assert cp.record("rejected") is None and cp._threads == {}
    assert ep.mock_calls == source.mock_calls == []
    render.assert_not_called()
    ensure_code.assert_not_called()
    ensure_streaming.assert_not_called()


@pytest.mark.parametrize("rates,accepted", [((10, 0.5), True), ((100, 0.5), False)])
def test_admission_uses_endpoint_su_rates(tmp_path, rates, accepted):
    site, ep, kit, streamed = make(tmp_path, cap=40)
    ep.su_rates = rates
    params = {**P, "shards": 1, "wall_minutes": 60}
    result = kit.check("rates", params, [], [])
    assert result["ok"] is accepted
    if accepted:
        assert kit.submit("rates", params, [], []) == {"handle": "rates"}
        assert len(site.state) == 1
    else:
        with pytest.raises(ValueError, match="SU"):
            kit.submit("rates", params, [], [])
        assert site.cmds == [] and streamed == []


def test_retry_preserves_resolved_manifest_and_body(tmp_path, monkeypatch):
    site, ep, kit, streamed = make(tmp_path)
    kit.submit("pinned", {**P, "shards": 1}, [], [])
    original = kit.chain_point.record("pinned")
    kit.chain_point.code_ref = "new-branch"
    kit.sif = "/changed.sif"
    monkeypatch.setattr(kit.chain_point.source, "fingerprint", Mock(side_effect=AssertionError("must not resolve")))
    monkeypatch.setattr(kit.chain_point, "ensure_code", Mock(side_effect=AssertionError("must not update")))
    kit.submit("pinned", {**P, "shards": 1}, [], [])
    assert kit.chain_point.record("pinned") == original
    assert len(site.state) == 1


def test_pinned_input_change_before_stream_is_refused(tmp_path, monkeypatch):
    site, ep, kit, streamed = make(tmp_path)
    kit.submit("changed", {**P, "shards": 1}, [], [])
    monkeypatch.setattr(kit.chain_point.source, "md5", lambda _: "0" * 32)
    ep.stream_runner = lambda *args: "0" * 32
    j = ep.state.get("changed.j0").job_id
    site.start(j, ep.state.get("changed.j0").remote_dir)
    kit.status("changed")
    assert site.state[j] == "CANCELLED"
    assert not kit.chain_point._has("changed.j0", "streamed")


def test_ready_cache_must_verify_content(tmp_path, monkeypatch):
    site, ep, kit, streamed = make(tmp_path)
    kit.submit("corrupt", {**P, "shards": 1}, [], [])
    j = ep.state.get("corrupt.j0").job_id
    site.start(j, ep.state.get("corrupt.j0").remote_dir)
    site.node_files["/dev/shm/crvfs_lib_u_udc-x/ready"] = "ready"
    monkeypatch.setattr(CP.artifacts, "verify_cache", lambda *args: False)
    kit.status("corrupt")
    assert site.state[j] == "CANCELLED"
    assert not kit.chain_point._has("corrupt.j0", "libreused")


def test_same_handle_controllers_never_rewrite_ready_inputs(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    site, ep, kit, streamed = make(tmp_path)
    cp = kit.chain_point
    kit.submit("race", {**P, "shards": 1}, [], [])
    j = ep.state.get("race.j0").job_id
    site.start(j, ep.state.get("race.j0").remote_dir)
    other = CP.ChainPoint(kit, source=Source(), code="/different/code", background=False)
    entered, release = threading.Event(), threading.Event()
    original = ep.stream_runner

    def slow(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    ep.stream_runner = slow
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cp.stream_step, "race", 0)
        assert entered.wait(5)
        second = pool.submit(other.stream_step, "race", 0)
        release.set()
        assert first.result() == second.result() == "streamed"
    assert len(streamed) == 10
    cp._rec_path("race").with_suffix(".j0.streamed").unlink(missing_ok=True)
    assert other.stream_step("race", 0) == "streamed"
    assert len(streamed) == 10


NATIVE = {**P, "gen_set": CP.NATIVE_GEN_SET, "gen_events": 50000,
          "files_per_job": 1, "shard_offset": 0, "max_events": 0}


def test_native_admission_bytes_budget_and_descriptor(tmp_path):
    _, _, kit, _ = make(tmp_path)
    p = CP.validate(NATIVE)
    assert CP.gen_files(p, 0) == CP.gen_files(p, 31) == []
    assert CP.point_bytes(p) == p["shards"] * (CP.TIER1B_SETS["strict_v3"][2] + CP.SMALL_BYTES)
    assert CP.chain_estimate_s(p) == 60 + (180 + 900) * .05
    assert kit.check("native", {"task": "chain_point", **p}, [], [])["ok"]
    assert "gen_events" not in CP.validate(P)
    desc = kit.describe()["tasks"]["chain_point"]
    assert desc["params"]["gen_events"]["maximum"] == 2**20 - 1
    assert desc["configurations"]["gen_set"]["4"]["files"] == 0


@pytest.mark.parametrize("changes,message", [({"gen_events": 0}, "gen_events"),
    ({"gen_events": 2**20}, "gen_events"), ({"files_per_job": 2}, "files_per_job"),
    ({"shard_offset": 1}, "shard_offset"), ({"max_events": 1}, "max_events"),
    ({"run_config": 3}, "run_config"), ({"chain_flags": 1}, "prefetch"),
    ({"chain_flags": 3}, "prefetch"), ({"gen_set": 0}, "gen_events"),
    ({"gen_events": 1000000, "wall_minutes": 45}, "wall_minutes")])
def test_native_check_submit_refuse_before_resolution(tmp_path, changes, message):
    site, _, kit, streamed = make(tmp_path)
    params = {**NATIVE, **changes}
    result = kit.check("badnative", params, [], [])
    assert not result["ok"] and message in result["message"]
    with pytest.raises(ValueError, match=message):
        kit.submit("badnative", params, [], [])
    assert not site.cmds and not streamed and kit.chain_point.record("badnative") is None


def native_source(kit):
    from mu2efastsim.cry_receipts import EXTRACTED_V04_GDML_SHA256
    original = kit.chain_point.source.fingerprint
    kit.chain_point.source.fingerprint = lambda rel: {
        **original(rel), **({"sha256": EXTRACTED_V04_GDML_SHA256} if rel == CP.GDML else {})}


def test_native_fanout_pins_only_detector_inputs_and_preserves_guards(tmp_path):
    site, ep, kit, streamed = make(tmp_path)
    native_source(kit)
    kit.submit("native", NATIVE, [], [])
    rec = kit.chain_point.record("native")
    assert {f["src"] for f in rec["manifest"]["inputs"]["files"]} == set(CP.small_inputs(rec["params"]) + CP.library_files(rec["params"]))
    for k, body in enumerate(rec["body"]):
        assert "GEN=(native)" in body and 'ls "$L"/gen.' not in body
        assert f'--gen native --gen-events 50000 --gen-seed {7 + 1000*k} --gen-subrun {k+1}' in body
        assert "staged artifact digest mismatch" in body and "runtime image changed" in body
        assert "artifact-manifest.json" in body and "lib_release" in body and "cleanup.txt" in body
        assert "CrySource(seed=0)" in body
        assert subprocess.run(["bash", "-n"], input=body.encode(), capture_output=True).returncode == 0
        job = ep.state.get(f"native.j{k}").job_id
        site.start(job, ep.state.get(f"native.j{k}").remote_dir)
        kit.status("native")
    assert not any("/gen." in rel for rel, _ in streamed)
    assert len(streamed) == len(CP.library_files(rec["params"])) + 3 * len(CP.small_inputs(rec["params"]))
    kit.submit("native", NATIVE, [], [])
    assert len(site.state) == 3


def test_native_rejects_unknown_geometry_before_code_and_dispatch(tmp_path, monkeypatch):
    site, _, kit, _ = make(tmp_path)
    monkeypatch.setattr(kit.chain_point, "ensure_code", lambda: pytest.fail("code preparation reached"))
    with pytest.raises(ValueError, match="verified extracted-v04 GDML"):
        kit.submit("wronggeom", NATIVE, [], [])
    assert not site.state


@pytest.mark.parametrize("corrupt", [False, True, "missing"])
@pytest.mark.parametrize("backend", ["native", "device"])
def test_native_results_check_bound_receipts(tmp_path, corrupt, backend):
    from mu2efastsim.cry_receipts import EXTRACTED_V04_GDML_SHA256, GENERATOR
    site, ep, kit, _ = make(tmp_path)
    native_source(kit)
    kit.submit("native", {**NATIVE, "gen_set": CP.DEVICE_GEN_SET if backend == "device" else CP.NATIVE_GEN_SET}, [], [])
    original = site.out_files

    def output(directory):
        files = original(directory)
        k = int(directory.rsplit(".j", 1)[1])
        shard = json.loads(files["out/shard.json"])
        shard.update(generated_events=50000, live_seconds=0.8)
        shard["native_generator"] = {"generator": GENERATOR, "seed": 7 + 1000*k + int(corrupt is True),
            "accepted_events": 50000, "live_seconds": 0.8, "raw_batch_size": 100000,
            "cosmic_config": {"geometry_sha256": EXTRACTED_V04_GDML_SHA256},
            "cosmic_time_offset": "fixture", "stock_overenergy_accumulation": True,
            "event_range": {"run": 1400, "subrun": k+1, "first_event": 1, "last_event": 50000}}
        if backend == "device":
            from mu2efastsim.cry_constants import SEA_LEVEL_SHA256
            shard["native_generator"].update(generator="jax-cry-1.7", sampling={
                "table_sha256": SEA_LEVEL_SHA256, "subbox_m": 300., "date": "2021-06-21",
                "latitude": 41.8, "lifetime_s": 5e-8}, rng="jax threefry2x32", device="cuda:0",
                particle_capacity=400000, selected_capacity=8192)
        if corrupt == "missing":
            del shard["native_generator"]
        files["out/shard.json"] = json.dumps(shard).encode()
        return files

    site.out_files = output
    for k in range(3):
        site.state[ep.state.get(f"native.j{k}").job_id] = "COMPLETED"
    if corrupt:
        with pytest.raises(ValueError, match="bound request|no generator receipt"):
            kit.results("native")
    else:
        result = kit.results("native")
        metadata = result["metadata"]["native_generation"]
        assert metadata["live_seconds"] == pytest.approx(2.4)
        assert len(metadata["native_generators"]) == 3
        assert metadata["clusters_per_live_second"]["EX"] == pytest.approx(3*3900/2.4)
        assert result["metrics"]["generated_events"] == 150000


def test_table_normalization_preserves_existing_request_identity():
    expected = {k: v for k, v in P.items() if k != "task"}
    expected.update(gen_set=0, run_config=0, chain_flags=0, tier1b=0)
    assert CP.validate(P) == CP.validate({**P, "gen_events": 0}) == expected
    assert CP.validate({**NATIVE, "gen_events": 50000.0})["gen_events"] == 50000


def test_native_fused_flag_remains_opt_in(tmp_path):
    _, _, kit, _ = make(tmp_path)
    body, _ = kit.chain_point.render(CP.validate({**NATIVE, "chain_flags": 2}), 1)
    assert "--fused-response" in body and "--prefetch" not in body
    assert "--gen-seed 1007 --gen-subrun 2" in body


def test_native_retry_uses_original_image_code_manifest_and_body(tmp_path, monkeypatch):
    site, _, kit, _ = make(tmp_path)
    native_source(kit)
    kit.submit("nativepin", NATIVE, [], [])
    original = kit.chain_point.record("nativepin")
    kit.chain_point.code_ref, kit.sif = "new-code", "/new-image.sif"
    monkeypatch.setattr(kit.chain_point.source, "fingerprint", lambda _: pytest.fail("new resolution"))
    monkeypatch.setattr(kit.chain_point, "ensure_code", lambda: pytest.fail("new code preparation"))
    kit.submit("nativepin", NATIVE, [], [])
    assert kit.chain_point.record("nativepin") == original
    assert len(site.state) == 3


def test_native_ready_cache_corruption_cancels_without_regeneration(tmp_path, monkeypatch):
    site, ep, kit, streamed = make(tmp_path)
    native_source(kit)
    kit.submit("nativecache", {**NATIVE, "shards": 1}, [], [])
    job = ep.state.get("nativecache.j0").job_id
    site.start(job, ep.state.get("nativecache.j0").remote_dir)
    site.node_files[f"/tmp/crvfs_kit_j{job}_1/ready"] = "ready"
    monkeypatch.setattr(CP.artifacts, "verify_cache", lambda *a: False)
    kit.status("nativecache")
    assert site.state[job] == "CANCELLED" and not streamed


def test_device_gen_set_admission_and_render(tmp_path):
    site, ep, kit, _ = make(tmp_path)
    params = {**NATIVE, "gen_set": CP.DEVICE_GEN_SET}
    p = CP.validate(params)
    assert CP.gen_files(p, 0) == [] and CP.point_bytes(p) == CP.point_bytes(CP.validate(NATIVE))
    assert kit.check("device", params, [], [])["ok"]
    body, _ = kit.chain_point.render(p, 1)
    assert "--gen device --gen-events 50000 --gen-seed 1007 --gen-subrun 2" in body
    assert "cry_device import device_tables" in body and "CrySource" not in body
    assert subprocess.run(["bash", "-n"], input=body.encode(), capture_output=True).returncode == 0
    desc = kit.describe()["tasks"]["chain_point"]["configurations"]["gen_set"]["5"]
    assert desc["generator"] == "jax-cry-1.7" and desc["files"] == 0
    native_source(kit)
    kit.submit("device", params, [], [])
    assert len(site.state) == 3
    assert kit.chain_point.record("device")["manifest"]["resolved_params"]["gen_set"] == CP.DEVICE_GEN_SET
