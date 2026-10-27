"""Optional tracker registration: stdlib-only descriptors and admission, no live calls."""
from types import SimpleNamespace

import pytest

from mu2efastsim.endpoints import capabilities as C
from mu2efastsim.endpoints import chainpoint
from mu2efastsim.endpoints.kit import FastsimKit, TASKS
from mu2efastsim.endpoints.parameters import schema, validate


def test_tracker_schema_is_dependency_free():
    names = ("sample", "shard", "seed")
    specs = schema("tracker_point", names)
    assert specs["sample"].choices == ("extracted_v04_primary_v1",)
    assert specs["shard"].minimum == 1 and specs["shard"].maximum == 20
    assert specs["seed"].maximum == 2**31 - 1
    assert validate("tracker_point", {"sample": "extracted_v04_primary_v1", "shard": 1., "seed": 0.}, names)["shard"] == 1


@pytest.mark.skipif("tracker_point" not in TASKS, reason="tracker package not installed")
def test_tracker_routes_labels_and_uncertainty(tmp_path):
    desc = FastsimKit.describe()["tasks"]["tracker_point"]
    assert desc["configurations"]["fit"] == "hybrid F2"
    assert desc["configurations"]["F1_supported"] is False
    assert desc["configurations"]["secondary_library"] is False
    assert desc["configurations"]["fit_material"] is False
    assert desc["configurations"]["momentum_MeV_c"] == 2000
    assert desc["metrics"]["digis"]["unit"] == "count"
    assert desc["metrics"]["digitization_s"]["unit"] == "s"
    assert not desc["metrics"]["fit_event_fraction_se"]["uncertainty"]["gp_observation_noise"]
    params = {"task": "tracker_point", **TASKS["tracker_point"].defaults}
    for ep, wanted in [(SimpleNamespace(name="eaf", fnal_data="/exp/root/crv_fastsim"), "mounted"),
                       (SimpleNamespace(name="uva", stream_in=lambda: None), "streaming"),
                       (SimpleNamespace(name="unknown"), "unavailable")]:
        cap = C.resolve("tracker_point", params, ep, TASKS, chainpoint)
        assert cap["route"]["name"] == wanted
        assert cap["route"]["supported"] == (wanted != "unavailable")
        assert set(cap["metrics"]) == set(TASKS["tracker_point"].metrics)
