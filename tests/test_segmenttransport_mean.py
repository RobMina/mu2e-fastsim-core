"""Deterministic policy checks for EXPERIMENTAL transport, not new loss physics."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from mu2efastsim import eloss, materials, segmenttransport as segment


MAT = materials.POLYSTYRENE


class ForbiddenRNG:
    def __getattribute__(self, name):
        pytest.fail(f"mean mode must not access RNG attribute {name}")


def forbidden(*args, **kwargs):
    pytest.fail("mean mode must not call the sampler")


def run(ke, length, *, rng=None, **kwargs):
    return segment.transport_segment(rng, MAT, ke, length, tcut_MeV=0.1,
                                     loss_model="mean", **kwargs)


def patch_mean(monkeypatch, dedx=2.0):
    calls = []

    def mean(mat, p, cut):
        calls.append((p.copy(), cut))
        return np.full(p.shape, dedx)

    monkeypatch.setattr(eloss, "dedx_restricted", mean)
    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    return calls


def assert_bytes_equal(left, right):
    assert left.keys() == right.keys()
    for group in left:
        assert left[group].keys() == right[group].keys()
        for name, actual in left[group].items():
            expected = right[group][name]
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes(), (group, name)


@pytest.mark.parametrize("stopping", ["mean-replace", "sampled"])
def test_constant_dedx_matches_analytic_energy_and_step_records(monkeypatch, stopping):
    calls = patch_mean(monkeypatch)
    initial = np.array([10.0, 100.0, 0.5, 0.0, 100.0])
    length = np.array([1.0, 2.5, 4.0, 2.0, 0.0])
    result = run(initial, length, rng=ForbiddenRNG(), stopping=stopping)
    rows, final = result["slices"], result["final"]
    np.testing.assert_allclose(final["final_ke_MeV"], [8, 95, 0.5, 0, 100])
    np.testing.assert_allclose(final["traveled_mm"], [1, 2.5, 0, 0, 0])
    np.testing.assert_array_equal(final["status"],
                                  ["transmitted", "transmitted", "model_stop", "exhausted", "transmitted"])
    np.testing.assert_array_equal(rows["lane_index"][:4], [2, 3, 0, 1])
    np.testing.assert_allclose(rows["sampled_loss_MeV"], 2 * (rows["s1_mm"] - rows["s0_mm"]))
    np.testing.assert_allclose(rows["applied_loss_MeV"], rows["sampled_loss_MeV"])
    np.testing.assert_allclose(rows["ke_out_MeV"], initial[rows["lane_index"]] - 2 * rows["s1_mm"])
    np.testing.assert_array_equal(rows["s1_mm"][2:4], [0.25, 1])
    assert calls and all(cut == 0.1 for _, cut in calls)
    for lane in range(initial.size):
        selected = rows["lane_index"] == lane
        np.testing.assert_array_equal(rows["s0_mm"][selected][1:], rows["s1_mm"][selected][:-1])
        np.testing.assert_array_equal(rows["ke_in_MeV"][selected][1:], rows["ke_out_MeV"][selected][:-1])


def test_real_restricted_mean_recomputed_at_each_step_without_rng(monkeypatch):
    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    reference = run([3.0, 10.0, 100.0], [0.1, 1.0, 2.5], rng=ForbiddenRNG())
    assert_bytes_equal(reference, run([3.0, 10.0, 100.0], [0.1, 1.0, 2.5]))
    rows = reference["slices"]
    before = rows["ke_in_MeV"]
    p = np.sqrt(before) * np.sqrt(before + 2 * eloss.MMU)
    dedx = eloss.dedx_restricted(MAT, p, 0.1)
    np.testing.assert_allclose(rows["sampled_loss_MeV"], dedx * (rows["s1_mm"] - rows["s0_mm"]),
                               rtol=2e-14, atol=1e-15)
    for seed in (5, 912):
        rng = np.random.default_rng(seed)
        state = deepcopy(rng.bit_generator.state)
        assert_bytes_equal(reference, run([3.0, 10.0, 100.0], [0.1, 1.0, 2.5], rng=rng))
        assert rng.bit_generator.state == state


@pytest.mark.parametrize("stopping", ["mean-replace", "sampled"])
@pytest.mark.parametrize("length", [1.0, 10.0])
def test_mean_threshold_stops_retain_residual_and_terminal_precedes_exit(monkeypatch, stopping, length):
    patch_mean(monkeypatch)
    result = run([2.5, 3.0, 2.0], length, stopping=stopping, fractional_loss=1)
    rows, final = result["slices"], result["final"]
    np.testing.assert_array_equal(final["final_ke_MeV"], [0.5, 1, 0])
    np.testing.assert_array_equal(final["traveled_mm"], [1, 1, 1])
    np.testing.assert_array_equal(final["status"], ["model_stop", "model_stop", "exhausted"])
    np.testing.assert_array_equal(rows["sampled_loss_MeV"], [2, 2, 2])
    np.testing.assert_array_equal(rows["applied_loss_MeV"], [2, 2, 2])
    np.testing.assert_array_equal(rows["terminal"], final["status"])


def test_initial_stops_zero_length_and_empty_avoid_loss_evaluation(monkeypatch):
    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    monkeypatch.setattr(eloss, "dedx_restricted", forbidden)
    result = run([0, 0.25, 1, 20], [5, 5, 0, 0], rng=ForbiddenRNG())
    np.testing.assert_array_equal(result["final"]["final_ke_MeV"], [0, 0.25, 1, 20])
    np.testing.assert_array_equal(result["final"]["traveled_mm"], [0, 0, 0, 0])
    np.testing.assert_array_equal(result["final"]["status"],
                                  ["exhausted", "model_stop", "model_stop", "transmitted"])
    np.testing.assert_array_equal(result["slices"]["lane_index"], [0, 1, 2])
    np.testing.assert_array_equal(result["slices"]["sampled_loss_MeV"], [0, 0, 0])
    np.testing.assert_array_equal(result["slices"]["applied_loss_MeV"], [0, 0, 0])
    for length in (0, 1, []):
        empty = run([], length, rng=ForbiddenRNG())
        assert all(array.shape == (0,) for group in empty.values() for array in group.values())
        assert empty["slices"]["lane_index"].dtype.kind in "iu"


def test_zero_dedx_and_iteration_cap_do_not_consume_rng(monkeypatch):
    calls = patch_mean(monkeypatch, dedx=0)
    rng = np.random.default_rng(77)
    state = deepcopy(rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="max_steps=2"):
        run([10, 100], 3, max_steps=2, rng=rng)
    assert len(calls) == 2
    assert rng.bit_generator.state == state
    with np.errstate(all="raise"):
        result = run([10, 100], 2, max_steps=2, rng=ForbiddenRNG())
    np.testing.assert_array_equal(result["final"]["final_ke_MeV"], [10, 100])
    np.testing.assert_array_equal(result["final"]["traveled_mm"], [2, 2])
    np.testing.assert_array_equal(result["slices"]["sampled_loss_MeV"], [0, 0, 0, 0])


@pytest.mark.parametrize("bad", [None, "", "Mean", "unknown", False, 1, [], {}, np.array(["mean"])])
@pytest.mark.parametrize("ke,length", [([], []), ([0.5], 1), ([10], 1)])
def test_invalid_loss_model_before_loss_evaluation(monkeypatch, bad, ke, length):
    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    monkeypatch.setattr(eloss, "dedx_restricted", forbidden)
    rng = np.random.default_rng(3)
    state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="loss_model"):
        segment.transport_segment(rng, MAT, ke, length, tcut_MeV=0.1, loss_model=bad)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("settings", [
    {"stopping": "unknown"}, {"fractional_loss": 1.1}, {"max_step_mm": 0},
    {"stop_ke_MeV": 0}, {"max_steps": True},
])
def test_mean_preserves_settings_validation(monkeypatch, settings):
    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    monkeypatch.setattr(eloss, "dedx_restricted", forbidden)
    with pytest.raises(ValueError):
        run([], [], **settings)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_mean_rejects_invalid_dedx_without_sampling(monkeypatch, bad):
    patch_mean(monkeypatch, dedx=bad)
    with pytest.raises(ValueError, match="dedx"):
        run([10], 1, rng=ForbiddenRNG())


def test_mean_does_not_mutate_or_alias_inputs_or_material(monkeypatch):
    patch_mean(monkeypatch)
    ke, length = np.array([10.0, 20.0]), np.array([1.0, 2.0])
    initial, requested, material = ke.copy(), length.copy(), asdict(MAT)
    ke.setflags(write=False)
    length.setflags(write=False)
    result = run(ke, length)
    for group in result.values():
        for array in group.values():
            assert not np.shares_memory(array, ke)
            assert not np.shares_memory(array, length)
    result["final"]["final_ke_MeV"][:] = 0
    result["final"]["traveled_mm"][:] = 0
    result["slices"]["ke_in_MeV"][:] = 0
    np.testing.assert_array_equal(ke, initial)
    np.testing.assert_array_equal(length, requested)
    assert asdict(MAT) == material


@pytest.mark.parametrize("stopping", ["mean-replace", "sampled"])
@pytest.mark.parametrize("force_overshoot", [False, True])
def test_default_and_explicit_sampled_are_byte_identical_with_same_following_rng(monkeypatch, stopping, force_overshoot):
    if force_overshoot:
        monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full(p.shape, 2.0))
        monkeypatch.setattr(eloss, "sample_loss", lambda rng, mat, p, step, cut: 50 + rng.random(p.shape))
    implicit_rng, explicit_rng = np.random.default_rng(731), np.random.default_rng(731)
    args = (MAT, [0, 0.5, 1, 1.01, 3, 10, 100, 1000], [2, 2, 2, 100, 100, 100, 4, 0])
    implicit = segment.transport_segment(implicit_rng, *args, tcut_MeV=0.1, stopping=stopping)
    explicit = segment.transport_segment(explicit_rng, *args, tcut_MeV=0.1, stopping=stopping,
                                         loss_model="sampled")
    assert_bytes_equal(implicit, explicit)
    assert implicit_rng.bit_generator.state == explicit_rng.bit_generator.state
    assert implicit_rng.bit_generator.random_raw(32).tobytes() == explicit_rng.bit_generator.random_raw(32).tobytes()
