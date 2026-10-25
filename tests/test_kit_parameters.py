"""Evaluator-boundary tests use fake endpoints and render-only adversarial sentinels."""
import shlex
from types import SimpleNamespace

import pytest

from mu2efastsim.endpoints.kit import FastsimKit, TASKS
from mu2efastsim.endpoints.parameters import validate


class Endpoint:
    name = "eaf"
    fnal_data = "/exp/mu2e/data/test/crv_fastsim"

    def __init__(self):
        self.submissions = []

    def submit(self, *args):
        self.submissions.append(args)


@pytest.fixture
def kit(tmp_path):
    return FastsimKit(endpoint=Endpoint(), state_dir=tmp_path)


def refused(kit, params, *, name="point", files=None, inputs=None):
    files = [] if files is None else files
    inputs = [] if inputs is None else inputs
    result = kit.check(name, params, files, inputs)
    assert result["ok"] is False
    with pytest.raises(ValueError) as exc:
        kit.submit(name, params, files, inputs)
    assert str(exc.value) == result["message"]
    assert kit.ep.submissions == []
    assert not (kit.out_root.parent / "tasks").exists()


@pytest.mark.parametrize("value", [True, False, 1.5, -1, "12", "1e2", "1..2", "${AUDIT_MARKER}",
                                  "$(AUDIT_MARKER)", None, [], {}, float("nan"), float("inf"), -float("inf")])
def test_counts_rejected_identically(kit, value):
    refused(kit, {"task": "reco_bench", "events": value, "seed": 1})


@pytest.mark.parametrize("params", [None, [], {"task": []}, {"task": {}}, {"task": "missing"},
                                   {"task": "reco_bench", 1: 2}])
def test_invalid_task_objects(kit, params):
    refused(kit, params)


@pytest.mark.parametrize("name", ["../escape", ".hidden", "x/y", "x\n#SBATCH", "x;echo", "", None])
def test_bad_handles_precede_task_dispatch(kit, name):
    refused(kit, {"task": "reco_bench", "events": 1, "seed": 1}, name=name)


@pytest.mark.parametrize("task", ["reco_bench", "chain_point", "crv_response", "beam_point"])
@pytest.mark.parametrize("field", ["files", "inputs"])
def test_artifacts_rejected_before_handlers(kit, monkeypatch, task, field):
    def unexpected(*args, **kwargs):
        pytest.fail("artifact rejection must precede handler construction")
    monkeypatch.setattr(kit, "_delegate", unexpected)
    refused(kit, {"task": task}, **{field: [{"name": "geom", "uri": "file:///candidate", "kind": "geom"}]})


@pytest.mark.parametrize("task,kwargs", [
    ("veto_c4", {"selection": "other"}), ("veto_c4", {"gate_mode": 0.5}),
    ("veto_c4", {"min_layers": 5}), ("veto_c4", {"pmax": 0}),
    ("veto_c4", {"dt_hi": -1}), ("veto_c4", {"min_pe": -1}),
    ("crv_response", {"period": "P9"}), ("crv_response", {"tables": "other"}),
    ("crv_response", {"tables": "all", "events": 0}),
    ("crv_response", {"background": True}), ("crv_response", {"background": 0.5}),
    ("crv_response", {"speed_EX": 0}), ("crv_response", {"atten_T10": -1}),
    ("crv_response", {"cross_talk": 1.01}), ("crv_response", {"wf_noise": -1}),
    ("beam_point", {"repeat": 0.5}), ("beam_point", {"muons": float("inf")}),
    ("beam_point", {"geom": "/exp/mu2e/../elsewhere/a.txt"}),
    ("beam_point", {"geom": "${AUDIT_MARKER}.txt"}), ("beam_point", {"geom": "/tmp/a.txt"}),
])
def test_task_domains(kit, task, kwargs):
    if task not in TASKS:
        pytest.skip(f"{task} package not importable")
    defaults = {
        "veto_c4": {"selection": "run1a_dev", "dt_lo": 0, "dt_hi": 150, "min_pe": 0, "min_layers": 0,
                    "gate_start_lo": -1, "gate_end_hi": -1, "pmin": 0, "pmax": 100000},
        "crv_response": {"seed": 1}, "beam_point": {"geom": "nominal", "muons": 10, "seed": 1, "repeat": 0},
    }
    refused(kit, {"task": task, **defaults[task], **kwargs})


@pytest.mark.parametrize("value", ["1.08", "1e", "1..2", "${AUDIT_MARKER}", "$(AUDIT_MARKER)",
                                  True, None, float("nan"), float("inf"), 10 ** 400])
def test_crv_plain_and_delegate_share_value_refusals(kit, value):
    ku = pytest.importorskip("crvfastsim.kit_uva")
    params = {"task": "crv_response", "seed": 1, "speed_EX": value}
    refused(kit, params)
    with pytest.raises(ValueError) as exc:
        ku.validate(params)
    assert str(exc.value) == kit.check("point", params, [], [])["message"]


def test_integral_floats_defaults_and_valid_submission(kit):
    p = {"task": "reco_bench", "events": 12.0, "seed": 3.0}
    assert kit.check("p.bench", p, [], []) == {"ok": True, "message": ""}
    assert kit.submit("p.bench", p, [], []) == {"handle": "p.bench"}
    assert len(kit.ep.submissions) == 1
    body = kit.ep.submissions[0][1]
    assert "--max-events 12 " in body and "--seed 3 " in body


def test_delegated_refusal_has_no_submit_or_task_record(kit, monkeypatch):
    delegate = SimpleNamespace(check=lambda *args: {"ok": False, "message": "delegate admission refused"},
                               submit=lambda *args: pytest.fail("must not submit"))
    monkeypatch.setattr(kit, "_delegate", lambda **kwargs: delegate)
    refused(kit, {"task": "reco_bench", "events": 12, "seed": 1})


def test_chain_budget_rejected_before_outer_record(kit):
    kit.ep.stream_in = lambda *args: pytest.fail("must not stream")
    refused(kit, {"task": "chain_point", "shards": 1, "files_per_job": 1, "shard_offset": 0,
                  "max_events": 1, "seed": 1, "wall_minutes": 10})


def test_schema_matches_registered_parameters_and_preserves_legacy_describe():
    description = FastsimKit.describe()
    assert description["accepts_lists"] is False
    assert description["parameter_schema_version"] == 1
    assert "task" in description["params"] and "events_per_s" in description["metrics"]
    for name, task in TASKS.items():
        spec = description["tasks"][name]
        assert set(spec["params"]) == set(task.params)
        assert spec["file_kinds"] == []
        assert all("unit" in p and "role" in p for p in spec["params"].values())
        assert set(spec["required"]) == set(task.params) - set(task.defaults)
    assert description["tasks"]["veto_c4"]["params"]["dt_lo"]["unit"] == "ns"


def test_knob_shell_tokens_quote_outside_double_quotes():
    kt = pytest.importorskip("crvfastsim.kit_task")
    template = next(line for line in kt._knob_lines().splitlines() if "speed_EX=" in line)
    sentinel = "${AUDIT_MARKER}"
    rendered = template.replace("@p_speed_EX@", shlex.quote(sentinel))
    assert rendered == '[ -n \'${AUDIT_MARKER}\' ] && KN+=("speed_EX="\'${AUDIT_MARKER}\')'


def test_beam_delegate_and_plain_normalize_identically():
    ku = pytest.importorskip("beamfastsim.kit_uva")
    task = TASKS["beam_point"]
    params = {"task": "beam_point", "geom": "move_extras12.txt", "muons": 10.0, "seed": 3.0, "repeat": 0.0}
    assert ku.validate(params) == validate("beam_point", params, task.params, task.defaults)


@pytest.mark.parametrize("task,params", [
    ("reco_bench", {"events": 1, "seed": 1}),
    ("crv_response", {"seed": 1}),
    ("beam_point", {"geom": "nominal", "muons": 10, "seed": 1, "repeat": 0}),
])
def test_slurm_budget_rejected_before_code_or_records(tmp_path, task, params):
    from mu2efastsim.endpoints.slurm_ssh import SlurmSSHEndpoint
    if task not in TASKS:
        pytest.skip(f"{task} package not importable")
    ep = SlurmSSHEndpoint(state_dir=tmp_path / "jobs", max_su_per_job=1,
                          runner=lambda *args: pytest.fail("no remote work on refused point"))
    ep.submissions = []
    kit = FastsimKit(endpoint=ep, state_dir=tmp_path / "kit")
    refused(kit, {"task": task, **params})
    assert "max_su_per_job" in kit.check("point", {"task": task, **params}, [], [])["message"]
    assert not list((tmp_path / "jobs").glob("*.json"))
