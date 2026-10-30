"""Native shard aggregation is independent-sample exposure, never paired table identity."""
import copy

import pytest

from mu2efastsim.cry_receipts import GENERATOR, merge_receipts


def record(seed=1, subrun=1):
    return {"generated_events": 10, "live_seconds": 0.25,
            "native_generator": {"generator": GENERATOR, "seed": seed, "accepted_events": 10,
                                 "live_seconds": 0.25, "raw_batch_size": 100000,
                                 "cosmic_config": {"geometry_sha256": "fixture"},
                                 "cosmic_time_offset": "fixture", "stock_overenergy_accumulation": True,
                                 "event_range": {"run": 1400, "subrun": subrun,
                                                 "first_event": 1, "last_event": 10}}}


def test_native_exposure_sum_and_table_compatibility():
    first, second = record(), record(2, 2)
    merged = merge_receipts([first, second])
    assert merged["live_seconds"] == 0.5
    assert merged["native_generators"] == [first["native_generator"], second["native_generator"]]
    assert merge_receipts([{}, {}]) == {}


@pytest.mark.parametrize("change", ["mixed", "seed", "range", "config", "count", "clock", "nan", "bool",
                                    "missing-range", "missing-config", "generator", "null", "batch", "packed-collision", "huge-live"])
def test_corrupt_mixed_duplicate_receipts_refused(change):
    first, second = record(), copy.deepcopy(record(2, 2))
    receipt = second["native_generator"]
    if change == "mixed":
        del second["native_generator"]
    elif change == "seed":
        receipt["seed"] = 1
    elif change == "range":
        receipt["event_range"]["subrun"] = 1
    elif change == "config":
        receipt["cosmic_config"]["geometry_sha256"] = "other"
    elif change == "count":
        second["generated_events"] = 11
    elif change == "clock":
        second["live_seconds"] = 0.5
    elif change in ("nan", "bool"):
        second["live_seconds"] = receipt["live_seconds"] = float("nan") if change == "nan" else True
    elif change == "missing-range":
        del receipt["event_range"]
    elif change == "missing-config":
        del receipt["cosmic_config"]
    elif change == "generator":
        receipt["generator"] = "other"
    elif change == "packed-collision":
        receipt["event_range"].update(run=1401, subrun=1)
    elif change == "huge-live":
        second["live_seconds"] = receipt["live_seconds"] = 10**400
    elif change == "batch":
        receipt["raw_batch_size"] = 0
    else:
        second["native_generator"] = None
    with pytest.raises(ValueError, match="native generator merge"):
        merge_receipts([first, second])


def test_device_receipts_do_not_merge_with_cpu_generator():
    from mu2efastsim.cry_receipts import DEVICE_GENERATOR
    first, second = record(), record(2, 2)
    first["native_generator"]["generator"] = DEVICE_GENERATOR
    first["native_generator"].update(sampling={"table_sha256": "b5227d8c1d38ab249a3b238456a67a60c9a9a5b6f91f748562f4a517390dca75",
                "subbox_m": 300., "date": "2021-06-21", "latitude": 41.8, "lifetime_s": 5e-8},
                rng="jax threefry2x32", device="cuda:0", particle_capacity=400000, selected_capacity=8192)
    assert merge_receipts([first])["generator"] == DEVICE_GENERATOR
    with pytest.raises(ValueError, match="incompatible"):
        merge_receipts([first, second])

    del first["native_generator"]["sampling"]
    with pytest.raises(ValueError, match="sampling provenance"):
        merge_receipts([first])
