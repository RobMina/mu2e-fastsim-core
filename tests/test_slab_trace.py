"""Passive slab-loss records describe material columns, not geometrical stop positions."""
import copy

import numpy as np
import pytest

from mu2efastsim import eloss as E
from mu2efastsim import materials as M
from mu2efastsim import transport as T


def momentum(tk):
    tk = np.asarray(tk, float)
    return np.sqrt(tk * (tk + 2 * E.MMU))


def compare_trace(p, length, callback=None):
    rng_default = np.random.default_rng(90210)
    rng_trace = np.random.default_rng(90210)
    records = []
    expected = T.slab_loss(rng_default, M.CONCRETE_MARS, p, length)

    def collect(record):
        records.append(copy.deepcopy(record))
        if callback is not None:
            callback(record)

    actual = T.slab_loss(rng_trace, M.CONCRETE_MARS, p, length, trace=collect)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert rng_default.bit_generator.state == rng_trace.bit_generator.state
    assert rng_default.random(32).tobytes() == rng_trace.random(32).tobytes()
    assert [records[0]["event"], records[-1]["event"]] == ["init", "final"]
    assert records[-1]["returned_loss_MeV"].tobytes() == actual.tobytes()
    keys = records[0].keys()
    for record in records:
        assert record.keys() == keys
        for key, value in record.items():
            if key not in ("event", "slice_index"):
                assert isinstance(value, np.ndarray)
                assert value.shape == record["lane_index"].shape
                assert value.flags.owndata
    return actual, records


@pytest.mark.parametrize("seed", [1, 17, 123])
def test_real_sampler_exact_output_and_rng(seed):
    inputs = np.random.default_rng(seed)
    p = np.r_[0.0, 10.0, 5000.0, inputs.uniform(20, 6000, 60)]
    length = np.r_[200.0, 0.0, 0.0, inputs.uniform(0, 2500, 60)]
    _, records = compare_trace(p, length)
    assert any(record["event"] == "slice" for record in records)
    np.testing.assert_array_equal(records[0]["lane_index"], np.arange(p.size))
    np.testing.assert_array_equal(records[0]["requested_material_length_mm"], length)
    np.testing.assert_array_equal(records[0]["processed_material_length_after_mm"], 0)
    processed = np.zeros(p.size)
    accumulated = np.zeros(p.size)
    for record in records[1:-1]:
        lanes = record["lane_index"]
        np.testing.assert_array_equal(record["loss_accumulator_before_MeV"], accumulated[lanes])
        np.testing.assert_array_equal(record["processed_material_length_before_mm"], processed[lanes])
        accumulated[lanes] += record["effective_loss_MeV"]
        processed[lanes] = record["processed_material_length_after_mm"]
        np.testing.assert_array_equal(record["loss_accumulator_after_MeV"], accumulated[lanes])
    np.testing.assert_array_equal(records[-1]["loss_accumulator_after_MeV"], accumulated)
    np.testing.assert_array_equal(records[-1]["processed_material_length_after_mm"], processed)


def test_records_can_be_mutated_without_affecting_calculation():
    def corrupt(record):
        for value in record.values():
            if isinstance(value, np.ndarray):
                value[...] = 0
        record.clear()

    compare_trace(np.array([10.0, 300.0, 1000.0, 5000.0]), np.array([2., 2000., 2000., 1000.]), corrupt)


def test_retained_records_do_not_alias_inputs_outputs_or_each_other():
    p, length = momentum([100.0, 1000.0]), np.array([2.0, 100.0])
    records = []
    result = T.slab_loss(np.random.default_rng(1), M.CONCRETE_MARS, p, length, trace=records.append)
    snapshots = copy.deepcopy(records)
    p[:] = 0
    length[:] = 0
    result[:] = 0
    for record, snapshot in zip(records, snapshots):
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, snapshot[key])
    records[0]["requested_material_length_mm"][:] = -1
    np.testing.assert_array_equal(records[-1]["requested_material_length_mm"], [2., 100.])
    records[-1]["loss_accumulator_after_MeV"][:] = -1
    np.testing.assert_array_equal(records[-1]["loss_accumulator_before_MeV"],
                                  snapshots[-1]["loss_accumulator_before_MeV"])


def test_stopped_input_and_zero_length():
    p = momentum([0.0, 0.5, 0.5, 100.0])
    tk0 = np.sqrt(p ** 2 + E.MMU ** 2) - E.MMU
    result, records = compare_trace(p, np.array([8.0, 8.0, 0.0, 0.0]))
    assert [record["event"] for record in records] == ["init", "final"]
    np.testing.assert_array_equal(result, [tk0[0], tk0[1], tk0[2], 0.0])
    final = records[-1]
    np.testing.assert_array_equal(final["model_stop"], [True, True, True, False])
    np.testing.assert_array_equal(final["return_adjusted"], [False, True, True, False])
    np.testing.assert_array_equal(final["loss_accumulator_after_MeV"], 0.0)
    np.testing.assert_array_equal(final["return_adjustment_MeV"], result)
    np.testing.assert_array_equal(final["untraversed_material_length_mm"], [8., 8., 0., 0.])
    np.testing.assert_array_equal(final["processed_material_length_after_mm"], 0.0)


def test_empty():
    result, records = compare_trace(np.array([]), np.array([]))
    assert result.size == 0
    assert [record["event"] for record in records] == ["init", "final"]
    assert all(record["lane_index"].size == 0 for record in records)


def test_tail_overshoot_uses_mean_not_raw_sample(monkeypatch):
    monkeypatch.setattr(E, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 2.0))

    def tail(rng, mat, p, length, cut):
        rng.random(len(p))
        return np.full_like(p, 1e8)

    monkeypatch.setattr(E, "sample_loss", tail)
    result, records = compare_trace(momentum([100.0]), np.array([1.0]))
    assert [record["event"] for record in records] == ["init", "slice", "final"]
    sampled = records[1]
    np.testing.assert_array_equal(sampled["raw_sample_loss_MeV"], [1e8])
    np.testing.assert_array_equal(sampled["effective_loss_MeV"], [2.0])
    np.testing.assert_array_equal(sampled["mean_replaced"], [True])
    np.testing.assert_array_equal(sampled["model_stop"], [False])
    np.testing.assert_array_equal(result, [2.0])
    np.testing.assert_array_equal(records[-1]["untraversed_material_length_mm"], [0.0])


@pytest.mark.parametrize("length", [1.0, 3.0])
def test_late_below_cutoff_closure_adds_energy_not_length(monkeypatch, length):
    monkeypatch.setattr(E, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 0.1))
    monkeypatch.setattr(E, "sample_loss", lambda rng, mat, p, step, cut: np.full_like(p, 1.25))
    p = momentum([2.0])
    tk0 = np.sqrt(p ** 2 + E.MMU ** 2) - E.MMU
    result, records = compare_trace(p, np.array([length]))
    assert [record["event"] for record in records] == ["init", "slice", "final"]
    sampled, final = records[1:]
    assert sampled["model_stop"].all()
    assert not sampled["mean_replaced"].any()
    np.testing.assert_array_equal(result, tk0)
    np.testing.assert_array_equal(final["loss_accumulator_after_MeV"], [1.25])
    np.testing.assert_array_equal(final["return_adjustment_MeV"], tk0 - 1.25)
    assert final["return_adjusted"].all()
    np.testing.assert_array_equal(final["processed_material_length_after_mm"],
                                  sampled["step_material_length_mm"])
    np.testing.assert_array_equal(final["untraversed_material_length_mm"],
                                  length - sampled["step_material_length_mm"])
    np.testing.assert_array_equal(final["step_material_length_mm"], 0)


@pytest.mark.parametrize("cap", [0, 1])
def test_iteration_cap_mean_remainder_and_clamp(monkeypatch, cap):
    monkeypatch.setattr(T, "N_SLICES_MAX", cap)
    monkeypatch.setattr(E, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 2.0))
    monkeypatch.setattr(E, "sample_loss", lambda rng, mat, p, step, cut: 2.0 * step)
    p = momentum([100.0, 100.0, 0.5, 100.0])
    result, records = compare_trace(p, np.array([10.0, 100.0, 100.0, 0.0]))
    remainder = next(record for record in records if record["event"] == "mean_remainder")
    final = records[-1]
    np.testing.assert_array_equal(remainder["lane_index"], [0, 1])
    assert remainder["iteration_cap"].all()
    assert not remainder["mean_replaced"].any()
    assert np.isnan(remainder["raw_sample_loss_MeV"]).all()
    np.testing.assert_array_equal(remainder["effective_loss_MeV"],
                                  2.0 * remainder["step_material_length_mm"])
    np.testing.assert_array_equal(remainder["model_stop"], [False, True])
    np.testing.assert_array_equal(final["iteration_cap"], [True, True, False, False])
    np.testing.assert_array_equal(final["processed_material_length_after_mm"], [10., 100., 0., 0.])
    np.testing.assert_array_equal(final["untraversed_material_length_mm"], [0., 0., 100., 0.])
    np.testing.assert_array_equal(final["loss_accumulator_after_MeV"], [20., 200., 0., 0.])
    assert result[1] < final["loss_accumulator_after_MeV"][1]
    assert final["return_adjustment_MeV"][1] < 0
    assert final["kinetic_energy_after_MeV"][1] < 0


def test_trace_is_keyword_only_and_exceptions_propagate():
    with pytest.raises(TypeError):
        T.slab_loss(np.random.default_rng(1), M.CONCRETE_MARS, np.array([]), np.array([]), None)

    def fail(record):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        T.slab_loss(np.random.default_rng(1), M.CONCRETE_MARS, np.array([]), np.array([]), trace=fail)
