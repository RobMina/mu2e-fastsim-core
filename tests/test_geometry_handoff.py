"""Synthetic target FileRefs, local staging and fake endpoints; no physics or network."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from mu2efastsim.endpoints import geometry as G
from mu2efastsim.endpoints.kit import FastsimKit


P = {"task": "beam_point", "muons": 10, "seed": 1, "repeat": 0}


def file_ref(path, dz=25):
    data = f'#include "{G.BASE}"\n// target fixture\ndouble stoppingTarget.deltaZ = {dz};\n'.encode()
    path.write_bytes(data)
    return {"name": "geom", "kind": "geom", "uri": path.resolve().as_uri(),
            "sha256": hashlib.sha256(data).hexdigest(), "metadata": {
                "format": "offline_simpleconfig", "schema_version": 1, "renderer": "offline_simpleconfig",
                "renderer_sha256": "b" * 64}}


class Endpoint:
    name = "eaf"
    fnal_data = "/exp/mu2e/data/test/crv_fastsim"

    def __init__(self):
        self.submissions = []

    def runtime(self):
        return {"a": 'M2FS_CODE=/fake/code; M2FS_VENV=/fake/venv\nA() { return 0; }\n', "t": "T() { return 0; }\n"}

    def submit(self, *args):
        self.submissions.append(args)


@pytest.fixture
def kit(tmp_path):
    return FastsimKit(endpoint=Endpoint(), state_dir=tmp_path / "kit")


def refused(kit, params, files, inputs=(), metrics=None):
    checked = kit.check("p", params, files, list(inputs), metrics)
    assert not checked["ok"]
    with pytest.raises(ValueError) as exc:
        kit.submit("p", params, files, list(inputs), metrics)
    assert str(exc.value) == checked["message"]
    assert not kit.ep.submissions and not (kit.out_root.parent / "tasks").exists()


def test_bare_legacy_fileref_is_explicit_refusal(kit):
    refused(kit, P, [{"name": "geom", "kind": "geom", "uri": "file:///unknown"}])


@pytest.mark.parametrize("change", [{"kind": "gdml"}, {"name": "other"}, {"sha256": "a" * 63},
                                   {"uri": "root://host/path"}, {"uri": "file://remote/path"},
                                   {"uri": "file:///pnfs/mu2e/data"}, {"uri": "file:///tmp/%00"},
                                   {"uri": "file:///tmp/../bad"}, {"metadata": {}}, {"unknown": 1}])
def test_descriptor_refusals_precede_allocation(tmp_path, kit, change):
    ref = {**file_ref(tmp_path / "a.txt"), **change}
    refused(kit, P, [ref])


def test_conflicting_param_multiple_files_other_tasks(tmp_path, kit):
    ref = file_ref(tmp_path / "a.txt")
    refused(kit, {**P, "geom": "nominal"}, [ref])
    refused(kit, P, [ref], [ref])
    refused(kit, {"task": "chain_point"}, [ref])


@pytest.mark.parametrize("text", [
    '#include "other.txt"\n', '#include "Offline/Mu2eG4/geom/geom_common.txt"\n#include "extra.txt"\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble mu2e.solenoidOffset = 0;\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble stoppingTarget.deltaZ = NaN;\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble stoppingTarget.deltaZ = 1e999;\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\nunsupported stuff\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble stoppingTarget.deltaZ = 0;\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\nvector<double> stoppingTarget.radii = {};\n',
    '#include "Offline/Mu2eG4/geom/geom_common.txt"\ndouble stoppingTarget.deltaZ = 1; double stoppingTarget.deltaZ = 2;\n',
])
def test_unsupported_overlay_fails_before_submission(tmp_path, kit, text):
    path = tmp_path / "a.txt"
    ref = file_ref(path)
    path.write_text(text)
    ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    refused(kit, P, [ref])


def test_digest_regular_file_size_guards(tmp_path, kit):
    path = tmp_path / "a.txt"
    ref = file_ref(path)
    path.write_text("replacement")
    refused(kit, P, [ref])
    path.unlink()
    target = tmp_path / "b.txt"
    original = file_ref(target)
    path.symlink_to(target)
    refused(kit, P, [{**original, "uri": path.as_uri()}])
    path.unlink()
    path.write_bytes(b" " * (G.MAX_BYTES + 1))
    refused(kit, P, [ref])


def test_two_candidates_bound_on_eaf_and_unchanged_after_source_mutation(tmp_path, kit):
    a, b = file_ref(tmp_path / "a.txt", 25), file_ref(tmp_path / "b.txt", 30)
    for name, ref in (("a", a), ("b", b)):
        checked = kit.check(name, P, [ref], [], ["stop_frac"])
        assert checked["ok"] and checked["geometry"]["sha256"] == ref["sha256"]
        kit.submit(name, P, [ref], [], ["stop_frac"])
        request = kit._request(name)
        assert request["geometry"]["reference"]["sha256"] == ref["sha256"]
        assert 'GEOM="$M2FS_GEOMETRY"' in request["body"]
        assert subprocess.run(["bash", "-n"], input=request["body"], text=True, capture_output=True).returncode == 0
    before = kit.ep.submissions[0][1]
    (tmp_path / "a.txt").unlink()
    kit.submit("a", P, [a], [], ["stop_frac"])
    assert kit.ep.submissions[-1][1] == before
    with pytest.raises(ValueError, match="geometry differs"):
        kit.submit("a", P, [b], [], ["stop_frac"])
    assert G.identity(a) != G.identity(b)
    assert G.identity(a)["metadata"] == G.identity(b)["metadata"]


def test_local_staging_and_result_fileref_roundtrip(tmp_path, kit, monkeypatch):
    ref = file_ref(tmp_path / "a.txt")
    kit.submit("one", P, [ref], [], ["stop_frac"])
    artifact = kit._request("one")["geometry"]
    out = kit.out_root / "one" / "out"
    out.mkdir(parents=True)
    result = subprocess.run(["bash", "-c", G.stage_shell(artifact)],
                            env={"PATH": __import__("os").environ["PATH"], "M2FS_OUT": str(out)}, capture_output=True)
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(kit, "_results", lambda _: {"metrics": {"stop_frac": 0.2}, "metadata": {}, "files": []})
    answer = kit.results("one")
    downstream = answer["files"][0]
    assert downstream["sha256"] == ref["sha256"] and downstream["metadata"] == ref["metadata"]
    assert kit.check("next", P, [], [downstream], ["stop_frac"])["ok"]
    kit.submit("next", P, [], [downstream], ["stop_frac"])
    assert G.identity(kit._request("next")["geometry"]) == G.identity(artifact)
    (out / "geometry.txt").chmod(0o600)
    (out / "geometry.txt").write_text("corrupted")
    with pytest.raises(ValueError, match="geometry differs"):
        kit.results("one")


def test_requested_metric_rejected_before_dispatch(kit):
    refused(kit, {**P, "geom": "nominal"}, [], metrics=["muons_per_s_warm"])
    refused(kit, {**P, "geom": "nominal"}, [], metrics=["input_wait_s"])
    refused(kit, {**P, "geom": "nominal"}, [], metrics=["not_a_metric"])


def test_requested_conditional_metric_checked_on_results(tmp_path, kit, monkeypatch):
    p = {**P, "geom": "nominal"}
    kit.submit("p", p, [], [], ["stop_r_mean"])
    monkeypatch.setattr(kit, "_results", lambda _: {"metrics": {"stop_r_mean": -1, "stops": 0}, "files": [], "metadata": {}})
    with pytest.raises(ValueError, match="sentinel|stops"):
        kit.results("p")
    with pytest.raises(ValueError, match="request differs"):
        kit.submit("p", p, [], [], ["stop_frac"])


def test_refused_requested_metric_does_not_resolve_file(tmp_path, kit, monkeypatch):
    ref = file_ref(tmp_path / "a.txt")
    monkeypatch.setattr(G, "resolve", lambda *a, **k: pytest.fail("must fail before file read"))
    refused(kit, P, [ref], metrics=["muons_per_s_warm"])


def test_lost_submission_reply_reuses_geometry_and_original_rendering(tmp_path, kit, monkeypatch):
    from mu2efastsim.endpoints import EndpointError
    ref = file_ref(tmp_path / "a.txt")
    calls = []
    def fail(*args):
        calls.append(args)
        assert kit._request("p")["geometry"]["reference"] == ref
        raise EndpointError("lost reply")
    monkeypatch.setattr(kit.ep, "submit", fail)
    with pytest.raises(ValueError, match="lost reply"):
        kit.submit("p", P, [ref], [], ["stop_frac"])
    (tmp_path / "a.txt").unlink()
    monkeypatch.setattr(kit.ep, "runtime", lambda: {"a": "NEW RUNTIME", "t": ""})
    monkeypatch.setattr(kit.ep, "submit", lambda *args: calls.append(args))
    kit.submit("p", P, [ref], [], ["stop_frac"])
    assert calls[0] == calls[1]


def test_retry_ignores_changed_current_template(tmp_path, kit, monkeypatch):
    from dataclasses import replace
    from mu2efastsim.endpoints.kit import TASKS
    ref = file_ref(tmp_path / "a.txt")
    kit.submit("p", P, [ref], [], ["stop_frac"])
    original = kit.ep.submissions[-1]
    (tmp_path / "a.txt").unlink()
    monkeypatch.setitem(TASKS, "beam_point", replace(TASKS["beam_point"], body="@p_UNKNOWN@"))
    assert kit.check("p", P, [ref], [], ["stop_frac"])["ok"]
    kit.submit("p", P, [ref], [], ["stop_frac"])
    assert kit.ep.submissions[-1] == original


def test_staging_canonicalizes_site_output_alias(tmp_path):
    actual = tmp_path / "physical"
    actual.mkdir()
    alias = tmp_path / "scratch"
    alias.symlink_to(actual, target_is_directory=True)
    ref = file_ref(tmp_path / "a.txt")
    artifact = G.resolve(ref)
    body = f'M2FS_OUT={alias}\n' + G.stage_shell(artifact) + 'printf "%s" "$M2FS_GEOMETRY"'
    result = subprocess.run(["bash", "-c", body], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(actual / "geometry.txt")


def test_eaf_lifetime_refused_before_request_binding(tmp_path):
    from mu2efastsim.endpoints.eaf import EAFEndpoint
    ep = EAFEndpoint(workdir=str(tmp_path / "work"), shared_prefix=str(tmp_path),
                     state_dir=tmp_path / "jobs", hub_client=SimpleNamespace(), max_age_s=60)
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    ref = file_ref(tmp_path / "a.txt")
    check = kit.check("p", P, [ref], [], ["stop_frac"])
    assert not check["ok"] and "lifetime" in check["message"]
    with pytest.raises(ValueError, match="lifetime"):
        kit.submit("p", P, [ref], [], ["stop_frac"])
    assert kit._request("p") is None and not list(ep.state.root.glob("*.json"))


def test_comment_placeholders_preserve_exact_candidate_bytes(tmp_path, kit):
    path = tmp_path / "a.txt"
    ref = file_ref(path)
    path.write_text(path.read_text() + "// @p_muons@ @sif@ @fnal_data@\n")
    ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert kit.check("comment", P, [ref], [], ["stop_frac"])["ok"]
    kit.submit("comment", P, [ref], [], ["stop_frac"])
    body = kit.ep.submissions[-1][1]
    assert "// @p_muons@ @sif@ @fnal_data@" in body
