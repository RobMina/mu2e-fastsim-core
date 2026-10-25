"""Capability contracts use fake endpoints and local files, never physics dependencies."""
from copy import deepcopy
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mu2efastsim.endpoints import capabilities as C, chainpoint
from mu2efastsim.endpoints.kit import TASKS


STREAM = SimpleNamespace(name="uva", stream_in=lambda *a: pytest.fail("remote I/O"))
MOUNTED = SimpleNamespace(name="eaf", fnal_data="/not-mounted-on-this-host/crv_fastsim")


def resolved(task, params=None, endpoint=STREAM, tasks=TASKS):
    return C.resolve(task, params or {}, endpoint, tasks, chainpoint)


def status(result, name):
    return result["metrics"][name]["availability"]


def test_declarative_schema_serializable_and_fresh():
    description = C.describe_tasks(TASKS, chainpoint)
    assert set(description) == set(TASKS) | {"chain_point"}
    json.dumps(description, allow_nan=False)
    for task, descriptor in description.items():
        assert descriptor["file_kinds"] == (["geom"] if task == "beam_point" else [])
        assert descriptor["validation_scope"]["physics_commissioned"] is False
        assert descriptor["route_requirements"]
        for metric in descriptor["metrics"].values():
            assert metric["availability"] in ("available", "conditional", "unavailable")
            assert metric["unit"] and metric["reason"]
        for rule in descriptor["metric_rules"]:
            assert all(isinstance(values, list) for values in rule["when"].values())
            assert set(rule) == {"when", "unavailable", "conditional", "reason"}
    description["chain_point"]["metrics"]["jobs_ok"]["unit"] = "damaged"
    assert C.describe_tasks(TASKS, chainpoint)["chain_point"]["metrics"]["jobs_ok"]["unit"] == "count"


def test_beam_repeat_route_population_and_uncertainty():
    warm = resolved("beam_point", {"repeat": 1})
    cold = resolved("beam_point", {"repeat": 0})
    assert status(warm, "muons_per_s_warm") == "available"
    assert status(cold, "muons_per_s_warm") == "unavailable"
    assert status(warm, "input_wait_s") == "available"
    assert status(resolved("beam_point", {"repeat": 1}, MOUNTED), "input_wait_s") == "unavailable"
    assert status(warm, "stop_frac_mu_minus") == "conditional"
    for name in ("stops_foil_mean", "stop_r_mean", "stop_frac_upstream_half"):
        assert status(warm, name) == "conditional"
        with pytest.raises(ValueError, match="sentinel"):
            C.validate_results({name: -1, "stops": 0}, [name], warm)
        with pytest.raises(ValueError, match="stops > 0"):
            C.validate_results({name: 0.3, "stops": 0}, [name], warm)
        with pytest.raises(ValueError, match="stops > 0"):
            C.validate_results({name: 0.3}, [name], warm)
        C.validate_results({name: 0.3, "stops": 1}, [name], warm)
    with pytest.raises(ValueError, match="unavailable"):
        C.require_metrics(["muons_per_s_warm"], cold)
    C.validate_results({"stop_frac_mu_minus": 0.2}, ["stop_frac_mu_minus"], warm)
    with pytest.raises(ValueError):
        C.validate_results({"stop_frac_mu_minus": -1}, ["stop_frac_mu_minus"], warm)
    error = warm["metrics"]["stop_frac_err"]["uncertainty"]
    assert error["kind"] == "binomial_standard_error"
    assert error["gp_observation_noise"] is False


@pytest.mark.parametrize("run_config", [1, 2, 3])
def test_chain_point_nonextracted_dynamic_not_census(run_config):
    result = resolved("chain_point", {"run_config": run_config, "gen_set": 2})
    for sector in ("EX", "T1", "T2", "M"):
        assert status(result, f"clusters_per_generated_{sector}") == "unavailable"
    assert not any("type" in key for key in result["metrics"])
    assert len(result["metric_patterns"]) == 1
    assert result["metric_patterns"][0]["availability"] == "conditional"
    assert C.require_metrics(["clusters_per_generated_type999"], result) == ["clusters_per_generated_type999"]
    for bad in ("type0", "type01", "type-1", "type1x", "type1\n"):
        with pytest.raises(ValueError):
            C.require_metrics(["clusters_per_generated_" + bad], result)
    with pytest.raises(ValueError, match="missing"):
        C.validate_results({}, ["clusters_per_generated_type5"], result)
    values = {"clusters_per_generated_type5": 0.1, "clusters_per_generated_EX": float("nan")}
    C.validate_results(values, ["clusters_per_generated_type5"], result)
    assert "clusters_per_generated_EX" in values  # old producer's NaN is not silently removed


def test_chain_point_extracted_and_common_metrics():
    result = resolved("chain_point", {"run_config": 0, "gen_set": 0})
    assert result["metric_patterns"] == []
    assert all(status(result, name) == "available" for name in chainpoint.METRICS)
    with pytest.raises(ValueError, match="unknown"):
        C.require_metrics(["clusters_per_generated_type1"], result)
    assert result["metrics"]["input_wait_s_mean"]["unit"] == "s"
    assert result["metrics"]["accepted_per_est_su"]["unit"] == "events/SU"
    assert result["configurations"]["run_config"]["3"]["lookup_table_ids"] == list(chainpoint.RUN_CONFIGS[3][2])
    assert result["route"]["name"] == "streaming"
    assert resolved("chain_point", {"run_config": 0}, MOUNTED)["route"]["supported"] is False


def test_crv_period_terms_streaming_and_fixed_configuration():
    task = importlib.import_module("crvfastsim.kit_task")
    result = resolved("crv_response", {"period": "P3", "tables": "vr5"})
    assert result["configurations"]["required_terms"] == list(task.TERMS)
    for term in task.TERMS:
        assert status(result, "term_" + term) == ("conditional" if term == "dt_cluster_T2" else "available")
    assert status(result, "objective_all") == "conditional"
    assert status(result, "objective") == "available"
    assert status(result, "input_wait_s") == status(result, "inputs_reused") == "available"
    assert result["metrics"]["objective_sigma"]["uncertainty"]["gp_observation_noise"] is False
    for period in ("P4", "raw"):
        other = resolved("crv_response", {"period": period, "tables": "vr5"}, MOUNTED)
        assert status(other, "term_dt_cluster_T2") == status(other, "objective_all") == "available"
        assert status(other, "input_wait_s") == status(other, "inputs_reused") == "unavailable"
    # Simulate the published optional parameter even when the shared checkout predates it.
    definition = TASKS["crv_response"]
    tasks = {"crv_response": SimpleNamespace(metrics=definition.metrics,
                                            params=tuple(definition.params) + ("response_config",))}
    config = {"schema": 1, "scenario": "extracted_v04"}
    result = resolved("crv_response", {"period": "P4", "tables": "vr5", "response_config": json.dumps(config)}, tasks=tasks)
    assert result["configuration"]["response_config"] == config
    fixed = result["configurations"]["response_config"]
    assert set(fixed["lookup_table_ids"]) == {"6000_0", "2370_1", "3200_0"}
    assert fixed["lookup_tables_fixed"] is True
    assert fixed["asset_overrides"] is False
    assert resolved("crv_response", {"tables": "all", "period": "P4"})["route"]["supported"] is False
    assert resolved("crv_response", {"tables": "all", "period": "P4"}, MOUNTED)["route"]["supported"] is True


def test_missing_crv_manifest_does_not_invent_guaranteed_terms(monkeypatch):
    def missing(name):
        raise ImportError(name)
    monkeypatch.setattr(C.importlib, "import_module", missing)
    result = resolved("crv_response", {"period": "P4", "tables": "vr5"})
    assert result["configurations"]["required_terms"] is None
    assert status(result, "term_dt_cluster_T2") == "conditional"


def test_veto_deadtime_local_presence_and_absence(tmp_path):
    endpoint = SimpleNamespace(name="eaf", fnal_data=str(tmp_path))
    result = resolved("veto_c4", {"selection": "run1a_dev"}, endpoint)
    assert status(result, "deadtime") == "unavailable"
    with pytest.raises(ValueError, match="absent"):
        C.require_metrics(["deadtime"], result)
    path = tmp_path / "deadtime_run1a" / "ce_mix1bb_dev_sel.parquet"
    path.parent.mkdir()
    path.write_text("not read as data")
    result = resolved("veto_c4", {"selection": "run1a_dev"}, endpoint)
    for name in ("deadtime", "deadtime_err", "deadtime_full", "deadtime_full_lo", "deadtime_full_hi", "deadtime_selected"):
        assert status(result, name) == "conditional"
    assert C.require_metrics(["deadtime"], result) == ["deadtime"]
    with pytest.raises(ValueError, match="finite"):
        C.validate_results({"deadtime": float("nan")}, ["deadtime"], result)
    assert status(resolved("veto_c4", {}, MOUNTED), "deadtime") == "conditional"


@pytest.mark.parametrize("root", ["/pnfs/mu2e/data", "/some/mount/pnfs/mu2e", "relative/path"])
def test_refuse_pnfs_or_relative_probes(root, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("must not stat PNFS/relative paths")
    monkeypatch.setattr(C.os, "lstat", forbidden)
    endpoint = SimpleNamespace(name="eaf", fnal_data=root)
    assert status(resolved("veto_c4", {}, endpoint), "deadtime") == "conditional"
    assert status(resolved("chain", {"shards": 2}, endpoint), "max_abs_cluster_rate_rel") == "conditional"


def test_no_symlink_traversal_into_pnfs(tmp_path, monkeypatch):
    link = tmp_path / "linked"
    link.symlink_to("/pnfs/mu2e/data")
    original = C.os.lstat
    def checked(path, *a, **k):
        assert not str(path).startswith("/pnfs/")
        return original(path, *a, **k)
    monkeypatch.setattr(C.os, "lstat", checked)
    result = resolved("veto_c4", {}, SimpleNamespace(name="eaf", fnal_data=str(link)))
    assert status(result, "deadtime") == "conditional"
    assert "Symlink" in result["metrics"]["deadtime"]["reason"]


def test_chain_exact_reference_filenames(tmp_path):
    endpoint = SimpleNamespace(name="eaf", fnal_data=str(tmp_path))
    assert status(resolved("chain", {"shards": 2}, endpoint), "max_abs_cluster_rate_rel") == "unavailable"
    path = tmp_path / "v2_fastC_v1" / "ref.001400_00000012.root"
    path.parent.mkdir()
    path.write_text("not opened")
    assert status(resolved("chain", {"shards": 1}, endpoint), "max_abs_cluster_rate_rel") == "unavailable"
    result = resolved("chain", {"shards": 2}, endpoint)
    assert status(result, "max_abs_cluster_rate_rel") == "conditional"
    assert status(result, "clusters_per_generated_EX") == "conditional"
    assert status(result, "generated_events") == "available"


@pytest.mark.parametrize("requested", ["events", {}, ["events", "events"], [True], [None], [""],
                                        [" events"], ["events\n"], ["x.y"], ["unknown"]])
def test_invalid_requests(requested):
    with pytest.raises(ValueError):
        C.require_metrics(requested, resolved("reco_bench"))


def test_sorted_requests_no_mutation_and_empty_compatibility():
    result = resolved("reco_bench")
    before = deepcopy(result)
    assert C.require_metrics(["first_call_s", "events"], result) == ["events", "first_call_s"]
    assert C.require_metrics(None, result) == C.require_metrics([], result) == []
    assert result == before
    metrics = {"events": 1, "first_call_s": 0.4, "extra": -1}
    C.validate_results(metrics, ["events"], result)
    assert metrics == {"events": 1, "first_call_s": 0.4, "extra": -1}
    C.validate_results({"events": float("nan")}, [], result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", None, [], 10 ** 500])
def test_requested_results_must_be_finite_numbers(value):
    with pytest.raises(ValueError, match="finite"):
        C.validate_results({"events": value}, ["events"], resolved("reco_bench"))


def test_unavailable_still_refused_with_plausible_result_and_unknown_task():
    with pytest.raises(ValueError, match="unavailable"):
        C.validate_results({"muons_per_s_warm": 10}, ["muons_per_s_warm"], resolved("beam_point", {"repeat": 0}))
    with pytest.raises(ValueError, match="unknown capability task"):
        resolved("bogus")
    for task in ("chain", "regression", "veto_c4"):
        result = resolved(task)
        assert result["route"]["supported"] is False
        assert all(m["availability"] == "unavailable" for m in result["metrics"].values())


def test_units_flags_and_conditional_diagnostics():
    result = C.describe_tasks(TASKS, chainpoint)
    assert result["regression"]["metrics"]["regression_pass"]["unit"] == "1"
    assert result["crv_response"]["metrics"]["inputs_reused"]["unit"] == "1"
    assert result["beam_point"]["metrics"]["stop_r_mean"]["unit"] == "mm"
    assert result["response_bench"]["metrics"]["photon_rows_per_s"]["unit"] == "rows/s"
    assert result["v4_response"]["metrics"]["v4_photons_pass"]["availability"] == "available"
    assert result["v4_response"]["metrics"]["v4_photons_jax_s"]["availability"] == "conditional"


def test_dependency_light_import_and_describe():
    code = '''
import sys
class NoPhysicsImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'jax', 'awkward', 'scipy', 'pandas', 'pyarrow'}:
            raise AssertionError('physics dependency imported: ' + fullname)
sys.meta_path.insert(0, NoPhysicsImports())
from mu2efastsim.endpoints import capabilities, chainpoint
from mu2efastsim.endpoints.kit import TASKS
capabilities.describe_tasks(TASKS, chainpoint)
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_opt_in_results_omit_unavailable_placeholders_with_explicit_reasons():
    result = resolved("chain_point", {"run_config": 1, "gen_set": 2})
    raw = {"accepted_events": 10, "clusters_per_generated_EX": float("nan"),
           "clusters_per_generated_type3": 0.2}
    finite, absent = C.contract_results(raw, ["accepted_events"], result)
    assert finite == {"accepted_events": 10, "clusters_per_generated_type3": 0.2}
    assert "clusters_per_generated_EX" in absent
    json.dumps(finite, allow_nan=False)
    assert "clusters_per_generated_EX" in raw
    with pytest.raises(ValueError, match="unavailable"):
        C.contract_results(raw, ["clusters_per_generated_EX"], result)


def test_opt_in_conditional_failure_not_silently_coerced_to_zero():
    result = resolved("beam_point", {"repeat": 0})
    raw = {"stop_frac": 0, "stops": 0, "stops_foil_mean": -1, "muons_per_s_warm": -1,
           "stop_frac_mu_minus": float("nan")}
    finite, absent = C.contract_results(raw, ["stop_frac"], result)
    assert finite == {"stop_frac": 0, "stops": 0}
    assert {"stops_foil_mean", "muons_per_s_warm", "stop_frac_mu_minus"} <= absent.keys()
    with pytest.raises(ValueError):
        C.contract_results(raw, ["stop_frac_mu_minus"], result)
    with pytest.raises(ValueError):
        C.contract_results({"stop_frac": float("nan")}, [], result)
