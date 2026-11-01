"""Render a grid of endpoint submissions to JSON, to compare two versions of mu2efastsim.endpoints byte for byte.

    python ci/endpoint_identity.py OUT.json            # render with the mu2efastsim on PYTHONPATH
    python ci/endpoint_identity.py --compare A.json B.json [--known]   # --known: A is core main (KNOWN_TEXT)
    python ci/endpoint_identity.py --digests MAIN.json OUT.json      # per-case SHA-256 of a main rendering

Each case submits through the real endpoint classes against the fakes of the test suites (no ssh, no hub, no grid)
and records what reaches the site (every remote command and its stdin: job scripts, headers, sbatch arguments), the
shaped resources, the GPU choice, budget estimates and refusals, and every file the endpoints and the kit leave in
their state directories (records, intents, kit task records). Temporary paths, submission tokens and wall-clock
timestamps are normalized. Streamed tasks (chain_point, crv_response, beam_point) need crv and the umbrella's beam
on PYTHONPATH; their cases are left out when those packages are missing. Renderings are written to a scratch area,
never committed: tests/test_endpoint_identity.py renders in-process and compares per-case SHA-256 digests with
tests/data/endpoint_identity_main.sha256.json, the digests of core main's rendering (with KNOWN_TEXT applied).
"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent
SITE_PLACEHOLDERS = {
    "M2FS_DATA": "/path/to/fnal/data", "M2FS_CODE": "/path/to/fnal/code", "FASTSIMKIT_SCRATCH": "/path/to/scratch",
    "FASTSIMKIT_FNAL_HOST": "fnal-host.example", "FASTSIMKIT_SLURM_HOST": "slurm-host.example",
    "FASTSIMKIT_SLURM_ACCOUNT": "test_account", "FASTSIMKIT_EAF_SHARED_PREFIX": "/path/to/fnal/",
    "FASTSIMKIT_GEOMETRY_PREFIX": "/path/to/fnal/",
}
SITE_OPTIONAL = ("M2FS_INPUTS", "FASTSIMKIT_FNAL_DATA", "CRVFS_DATA", "FASTSIMKIT_WORKDIR", "FASTSIMKIT_CODE",
                 "FASTSIMKIT_SIF", "FASTSIMKIT_EAF_WORKDIR", "FASTSIMKIT_EAF_CODE", "FASTSIMKIT_EAF_VENV")
TOKEN = re.compile(r"m2fs-v1-[0-9a-f]{8,}")
EPOCH = re.compile(r"\b1[7-9]\d{8}(?:\.\d+)?\b")
CAMPAIGN = re.compile(r'(\\?"campaign_s\\?": )-?[0-9.e+-]+')   # wall clock now minus the fake sacct start
ISO = re.compile(r"\b20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d\b")


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _autouse(mod, monkeypatch, tmp: Path) -> None:
    """Apply a test module's autouse fixtures (the suites' own fakes for snapshots, images, slots, site variables)."""
    for value in vars(mod).values():
        marker = getattr(value, "_fixture_function_marker", None)
        if marker is not None and marker.autouse:
            fn = value._get_wrapped_function()
            args = {"monkeypatch": monkeypatch, "tmp_path": tmp}
            fn(**{n: args[n] for n in inspect.signature(fn).parameters})


class Recorder:
    """Wraps an endpoint's runner and keeps every remote command with its stdin."""

    def __init__(self, runner, log):
        self.runner, self.log = runner, log

    def __call__(self, cmd, stdin, timeout):
        self.log.append({"cmd": list(cmd), "stdin": stdin.decode() if stdin else None})
        return self.runner(cmd, stdin, timeout)


class Case:
    def __init__(self, tmp: Path):
        self.tmp, self.log, self.data = tmp, [], {}

    def norm(self, value):
        text = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
        for root in (self.tmp.resolve(), self.tmp):    # the EAF test workdir gets a name the site-path scan accepts
            text = text.replace(str(root / "work"), "<TMP>/eaf-jobs").replace(str(root), "<TMP>")
        return CAMPAIGN.sub(r"\1<T>", ISO.sub("<ISO>", EPOCH.sub("<T>", TOKEN.sub("<TOKEN>", text))))

    def files(self):
        out = {}
        for p in sorted(self.tmp.rglob("*")):
            if p.is_file() and "test_slots" not in p.parts:
                try:
                    out[str(p.relative_to(self.tmp))] = self.norm(p.read_text())
                except UnicodeDecodeError:
                    out[str(p.relative_to(self.tmp))] = f"<{p.stat().st_size} bytes>"
        return out

    def result(self):
        return {"data": {k: self.norm(v) for k, v in self.data.items()},
                "traffic": [self.norm(e) for e in self.log], "files": self.files()}


def attempt(fn):
    try:
        return {"ok": fn()}
    except Exception as exc:    # the refusal text is part of the behaviour under comparison
        return {"error": f"{type(exc).__name__}: {exc}"}


def slurm_cases(cases):
    from mu2efastsim.endpoints import Resources
    from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint, su_estimate
    tests = _module(CORE / "tests" / "test_endpoints.py", "_identity_test_endpoints")
    R = Resources
    requests = {
        "cpu": R(cores=2, mem_gb=8, gpus=0, gpu_type="", minutes=15), "default": R(),
        "g16": R(cores=8, mem_gb=16, gpus=1, gpu_type="a100", minutes=60),
        "g32": R(cores=8, mem_gb=32, gpus=1, gpu_type="a100", minutes=60),
        "g33": R(cores=8, mem_gb=33, gpus=1, gpu_type="a100", minutes=60),
        "g96": R(cores=16, mem_gb=96, gpus=1, gpu_type="a100", minutes=45),
        "any16": R(cores=4, mem_gb=16, gpus=1, gpu_type="", minutes=30),
        "v100_64": R(cores=8, mem_gb=64, gpus=1, gpu_type="v100", minutes=30),
        "multi2_16": R(cores=8, mem_gb=16, gpus=2, gpu_type="a100", minutes=60),
        "multi4_200": R(cores=32, mem_gb=200, gpus=4, gpu_type="a100", minutes=120),
    }
    configs = {"auto": {}, "a100": {"gpu_type": "a100"}, "a40": {"gpu_type": "a40"},
               "slice": {"gpu_type": "1g.24gb"}, "none": {"gpu_type": None}, "partition": {"partition": "gpu-x"},
               "flat_rates": {"su_per_gpu_h": 53.0, "su_per_core_h": 0.5}, "cap10": {"max_su_per_job": 10.0}}
    for cname, kw in configs.items():
        with tempfile.TemporaryDirectory() as d:
            c = Case(Path(d))
            fake = tests.FakeSite()
            ep = SlurmSSHEndpoint(runner=fake, state_dir=c.tmp / "jobs", workdir="/path/to/scratch/kit", **kw)
            ep.run = Recorder(ep.run, c.log)
            c.data["config"] = {"partition": ep.partition, "gpu_type": ep.gpu_type, "su_rates": list(ep.su_rates),
                                "max_su_per_job": ep.max_su_per_job, "name": ep.name}
            for rname, r in requests.items():
                shaped, choice = ep.gpu_choice(r)
                c.data[f"{rname}.shape"] = [type(shaped).__name__, dataclasses.asdict(shaped)]
                c.data[f"{rname}.choice"] = choice
                c.data[f"{rname}.header"] = ep.header(rname, shaped, choice)
                c.data[f"{rname}.su"] = attempt(lambda: round(su_estimate(shaped, *ep.su_rates), 9))
                c.data[f"{rname}.admit"] = attempt(lambda: ep.admit_resources(r))
                rec = attempt(lambda: dataclasses.asdict(ep.submit(rname, f"echo {rname}\n", r)))
                c.data[f"{rname}.submit"] = rec
                if "ok" in rec:
                    c.data[f"{rname}.resubmit"] = attempt(lambda: ep.submit(rname, f"echo {rname}\n", r).job_id)
                    c.data[f"{rname}.changed"] = attempt(lambda: ep.submit(rname, "echo other\n", r).job_id)
                    c.data[f"{rname}.status"] = attempt(lambda: ep.status(rname).__dict__)
                    fake.finish(rec["ok"]["job_id"])
                    c.data[f"{rname}.final"] = attempt(lambda: ep.status(rname).__dict__)
            cases[f"slurm/{cname}"] = c.result()
    with tempfile.TemporaryDirectory() as d:
        c = Case(Path(d))
        for rname, r in requests.items():
            c.data[f"su_estimate/{rname}"] = attempt(lambda: round(su_estimate(r), 9))
            c.data[f"su_estimate_flat/{rname}"] = attempt(lambda: round(su_estimate(r, 53.0, 0.5), 9))
        c.data["unknown_gpu_type"] = attempt(lambda: SlurmSSHEndpoint(runner=lambda *a: None, state_dir=c.tmp,
                                                                      workdir="/w", gpu_type="h100"))
        cases["slurm/module"] = c.result()


KIT_PARAMS = {
    "reco_bench": {"events": 200000.0, "seed": 20261021.0},
    "v4_response": {"events": 1000, "seed": 1},
    "chain": {"shards": 1, "max_events": 0, "seed": 1},
    "veto_c4": {"selection": "run1a_dev", "dt_lo": 0.0, "dt_hi": 150.0, "min_pe": 0.0, "min_layers": 0,
                "gate_start_lo": 420.0, "gate_end_hi": 1700.0, "pmin": 85.0, "pmax": 200.0},
    "crv_response": {"seed": 1, "atten_T1": 0.58},
    "beam_point": {"geom": "move_extras12.txt", "muons": 105350.0, "seed": 1.0, "repeat": 0.0},
}


def kit_cases(cases):
    from mu2efastsim.endpoints import Resources
    from mu2efastsim.endpoints.kit import TASKS, FastsimKit
    from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint
    tests = _module(CORE / "tests" / "test_endpoints.py", "_identity_test_endpoints")
    eaf_tests = _module(CORE / "tests" / "test_endpoint_eaf.py", "_identity_test_endpoint_eaf")
    names = sorted(TASKS)
    with tempfile.TemporaryDirectory() as d:     # Slurm: what each task renders and sends (no delegates)
        c = Case(Path(d))
        fake = tests.FakeSite()
        ep = SlurmSSHEndpoint(runner=fake, state_dir=c.tmp / "jobs", workdir="/path/to/scratch/kit")
        ep.run = Recorder(ep.run, c.log)
        kit = FastsimKit(endpoint=ep, state_dir=c.tmp / "kit", sif="/x/m.sif")
        kit._delegate = lambda **kw: None
        c.data["tasks"] = names
        for name in names:
            p = {"task": name, **KIT_PARAMS.get(name, {})}
            c.data[f"{name}.render"] = attempt(lambda: [kit.render(p)[0], dataclasses.asdict(kit.render(p)[1])])
            c.data[f"{name}.check"] = attempt(lambda: kit.check(f"k_{name}", p, [], []))
            c.data[f"{name}.submit"] = attempt(lambda: kit.submit(f"k_{name}", p, [], []))
        cases["kit/slurm"] = c.result()
    with tempfile.TemporaryDirectory() as d:     # EAF: the same tasks through the EAF endpoint and a fake hub
        c = Case(Path(d))
        hub = eaf_tests.FakeHub(run=False)
        ep, clock = eaf_tests.make(c.tmp, hub)
        kit = FastsimKit(endpoint=ep, state_dir=c.tmp / "kit")
        for name in names:
            p = {"task": name, **KIT_PARAMS.get(name, {})}
            c.data[f"{name}.render"] = attempt(lambda: [kit.render(p)[0], dataclasses.asdict(kit.render(p)[1])])
            c.data[f"{name}.check"] = attempt(lambda: kit.check(f"k_{name}", p, [], []))
        rec = attempt(lambda: dataclasses.asdict(ep.submit("e1", "true\n", Resources(minutes=5))))
        c.data["e1.submit"] = rec
        c.data["e1.queued"] = attempt(lambda: dataclasses.asdict(ep.submit("e2", "true\n", Resources(minutes=5))))
        c.data["e1.status"] = attempt(lambda: ep.status("e1").__dict__)
        c.data["hub"] = {"starts": hub.starts, "sent": hub.sent, "stops": hub.stops}
        cases["eaf"] = c.result()


def streamed_cases(cases):
    import pytest
    suites = [("chain_point", CORE / "tests" / "test_chainpoint.py")]
    for pkg, label in (("crvfastsim", "crv_response"), ("beamfastsim", "beam_point")):
        try:
            mod = __import__(pkg)
        except ImportError:
            continue
        suites.append((label, Path(mod.__file__).resolve().parent.parent / "tests" / "test_kit_uva.py"))
    for label, test_file in suites:
        if not test_file.exists():
            continue
        t = _module(test_file, f"_identity_{label}")
        with tempfile.TemporaryDirectory() as d, pytest.MonkeyPatch.context() as mp:
            c = Case(Path(d))
            _autouse(t, mp, c.tmp)
            site, ep, kit, streamed = t.make(c.tmp)
            ep.run = Recorder(ep.run, c.log)
            p = dict(t.P, shards=2) if label == "chain_point" else {"task": label, **KIT_PARAMS[label]}
            c.data["check"] = attempt(lambda: kit.check("pt", p, [], []))
            c.data["submit"] = attempt(lambda: kit.submit("pt", p, [], []))
            jobs = sorted(site.state)
            for j in jobs:
                if label == "chain_point":
                    site.start(j, site.dirs[j])
                else:
                    site.state[j] = "RUNNING"
                    site.files[f"{site.dirs[j]}/inbox"] = ("udc-x /dev/shm/crvfs_resp_u_abc/in need\n"
                                                           if label == "crv_response"
                                                           else "udc-a1 /dev/shm/beamfs_kit_j7001_9\n")
            c.data["jobs"] = len(jobs)
            c.data["status1"] = attempt(lambda: kit.status("pt"))
            for j in jobs:
                site.state[j] = "COMPLETED"
            c.data["status2"] = attempt(lambda: kit.status("pt"))
            c.data["results"] = attempt(lambda: kit.results("pt"))
            c.data["streamed"] = sorted(streamed)   # files stream in parallel threads: order is not behaviour
            cases[f"streamed/{label}"] = c.result()


def render() -> dict:
    """All cases, keyed by name; runs with placeholder site variables and restores the environment."""
    saved = dict(os.environ)
    try:
        for name in SITE_OPTIONAL:
            os.environ.pop(name, None)
        os.environ.update(SITE_PLACEHOLDERS)
        cases = {}
        slurm_cases(cases)
        kit_cases(cases)
        streamed_cases(cases)
        return cases
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _esc(text: str) -> str:
    return json.dumps(text)[1:-1]


# Intentional differences between core main and the move to hep-endpoints, as (old, new) text in the rendered
# cases. The snapshot check embedded in streamed job bodies is hependpoints.snapshots' generic form: the modules a
# branch override may move come from the request ("branch_modules", Mu2e's ("core", "crv")) instead of a literal, and
# "submodule update" is skipped when no module is selected (Mu2e always selects some). Same behaviour for Mu2e.
KNOWN_TEXT = [
    ('override = not relative and name in ("core", "crv") and snapshot["policy"]["branch"]',
     'override = (not relative and name in request.get("branch_modules", [])\n'
     '                                and snapshot["policy"]["branch"])'),
    ('\n    git(temporary, "submodule", "update", "--quiet", "--init", "--recursive", "--", *modules)',
     '\n    if modules:                     # with no paths, "submodule update" would initialize every submodule\n'
     '        git(temporary, "submodule", "update", "--quiet", "--init", "--recursive", "--", *modules)'),
    ('if branch and name in ("core", "crv"):', 'if branch and name in request.get("branch_modules", []):'),
]
VERIFY = re.compile(r'(\{\\"action\\": \\"verify\\", \\"snapshot\\": .*?\\"policy\\": \{[^{}]*\}\})\}')
DIGEST = re.compile(r'(\\?"(?:spec_sha|request_sha|binding_sha)\\?": \\?")[0-9a-f]{16,64}')


def apply_known(case: dict) -> dict:
    """A rendering of core main with the KNOWN_TEXT changes applied; digests of changed job bodies are masked."""
    def fix(text):
        new = text
        for old, rep in KNOWN_TEXT:
            new = new.replace(_esc(old), _esc(rep))
        new = VERIFY.sub(lambda m: m.group(1) + ', \\"branch_modules\\": [\\"core\\", \\"crv\\"]}', new)
        return new
    changed = any(fix(v) != v for v in [*case["traffic"], *case["files"].values()])
    mask = (lambda t: DIGEST.sub(r"\1<SHA>", t)) if changed else (lambda t: t)
    return {"data": {k: mask(fix(v)) for k, v in case["data"].items()},
            "traffic": [mask(fix(v)) for v in case["traffic"]],
            "files": {k: mask(fix(v)) for k, v in case["files"].items()}, "changed": changed}


def masked(case: dict) -> dict:
    """The case with the digests of job bodies masked (for cases whose bodies carry KNOWN_TEXT)."""
    return {"data": {n: DIGEST.sub(r"\1<SHA>", v) for n, v in case["data"].items()},
            "traffic": [DIGEST.sub(r"\1<SHA>", v) for v in case["traffic"]],
            "files": {n: DIGEST.sub(r"\1<SHA>", v) for n, v in case["files"].items()}}


def case_digest(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def main_digests(main: dict) -> dict:
    """Per-case digests of a rendering of core main with KNOWN_TEXT applied, and the cases where it applied."""
    known = {k: apply_known(v) for k, v in main.items()}
    changed = sorted(k for k, v in known.items() if v.pop("changed"))
    return {"changed": changed, "cases": {k: case_digest(v) for k, v in sorted(known.items())}}


def compare(a: dict, b: dict, known: bool = False) -> list:
    """Unified diffs of the cases present in both renderings, and the names present in only one. With known, the
    first rendering is core main's: KNOWN_TEXT is applied to it and digests are masked on both sides where it applied."""
    out = [f"only in first: {k}" for k in sorted(set(a) - set(b))]
    out += [f"only in second: {k}" for k in sorted(set(b) - set(a))]
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if known:
            x = apply_known(x)
            if x.pop("changed"):
                y = masked(y)
        if x != y:
            xs = json.dumps(x, indent=1, sort_keys=True).splitlines()
            ys = json.dumps(y, indent=1, sort_keys=True).splitlines()
            out.append("\n".join(difflib.unified_diff(xs, ys, f"first/{k}", f"second/{k}", lineterm="")))
    return out


def main(argv):
    if argv[:1] == ["--compare"]:
        known = "--known" in argv
        a, b = (json.loads(Path(p).read_text()) for p in [x for x in argv[1:] if x != "--known"][:2])
        diffs = compare(a, b, known)
        print("\n".join(diffs) if diffs else f"identical: {len(set(a) & set(b))} cases")
        return 1 if diffs else 0
    if argv[:1] == ["--digests"]:
        Path(argv[2]).write_text(json.dumps(main_digests(json.loads(Path(argv[1]).read_text())), indent=1) + "\n")
        return 0
    Path(argv[0]).write_text(json.dumps(render(), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
