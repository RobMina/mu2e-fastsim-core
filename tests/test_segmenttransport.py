"""Policy tests for isolated EXPERIMENTAL transport, not Geant4 validation."""
from dataclasses import replace

import numpy as np
import pytest

from mu2efastsim import eloss, materials, segmenttransport as segment


MAT = materials.POLYSTYRENE


def run(ke, length, **kwargs):
    return segment.transport_segment(np.random.default_rng(731), MAT, ke, length,
                                     tcut_MeV=0.1, **kwargs)


def patch_model(monkeypatch, dedx=2.0, raw=0.25):
    calls = []
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full(p.shape, dedx))

    def sample(rng, mat, p, step, cut):
        calls.append((p.copy(), step.copy(), cut))
        return np.full(p.shape, raw) if np.isscalar(raw) else raw(p, step)

    monkeypatch.setattr(eloss, "sample_loss", sample)
    return calls


def assert_equal(a, b):
    for group in a:
        assert a[group].keys() == b[group].keys()
        for name in a[group]:
            np.testing.assert_array_equal(a[group][name], b[group][name])


def assert_conservation(result, initial, requested):
    rows, final = result["slices"], result["final"]
    np.testing.assert_allclose(rows["ke_in_MeV"] - rows["ke_out_MeV"], rows["applied_loss_MeV"])
    assert np.all(rows["s1_mm"] >= rows["s0_mm"])
    assert np.all(rows["ke_out_MeV"] >= 0)
    assert np.all(rows["sampled_loss_MeV"] >= 0)
    assert np.all(final["traveled_mm"] <= requested)
    total = np.bincount(rows["lane_index"], weights=rows["applied_loss_MeV"], minlength=len(initial))
    np.testing.assert_allclose(total + final["final_ke_MeV"], initial)
    for lane in range(len(initial)):
        select = rows["lane_index"] == lane
        s0, s1 = rows["s0_mm"][select], rows["s1_mm"][select]
        before, after = rows["ke_in_MeV"][select], rows["ke_out_MeV"][select]
        if s0.size:
            assert s0[0] == 0
            np.testing.assert_array_equal(s0[1:], s1[:-1])
            np.testing.assert_array_equal(before[1:], after[:-1])
            assert s1[-1] == final["traveled_mm"][lane]
            assert after[-1] == final["final_ke_MeV"][lane]


def test_sampled_partial_stopping_preserves_raw(monkeypatch):
    calls = patch_model(monkeypatch, dedx=2, raw=40)
    result = run([10.0], 100, stopping="sampled")
    rows = result["slices"]
    np.testing.assert_array_equal(calls[0][1], [0.25])
    np.testing.assert_allclose(calls[0][0], np.sqrt([10.0]) * np.sqrt(10 + 2 * eloss.MMU))
    np.testing.assert_array_equal(rows["s1_mm"], [0.0625])
    np.testing.assert_array_equal(rows["sampled_loss_MeV"], [40])
    np.testing.assert_array_equal(rows["applied_loss_MeV"], [10])
    np.testing.assert_array_equal(rows["terminal"], ["exhausted"])
    assert result["final"]["status"][0] == "exhausted"
    assert_conservation(result, [10], [100])


@pytest.mark.parametrize("raw", [10, 40])
def test_mean_replacement_only_for_overshoots(monkeypatch, raw):
    patch_model(monkeypatch, dedx=2, raw=raw)
    mean = run([10], 0.25)
    sampled = run([10], 0.25, stopping="sampled")
    assert mean["slices"]["sampled_loss_MeV"][0] == raw
    assert mean["slices"]["applied_loss_MeV"][0] == 0.5
    assert mean["final"]["final_ke_MeV"][0] == 9.5
    assert mean["final"]["traveled_mm"][0] == 0.25
    assert mean["final"]["status"][0] == "transmitted"
    assert sampled["final"]["final_ke_MeV"][0] == 0
    assert sampled["final"]["traveled_mm"][0] == 0.25 * 10 / raw


def test_nonovershoot_arms_identical_and_cutoff_not_deposited(monkeypatch):
    patch_model(monkeypatch, dedx=1, raw=9.5)
    mean = run([10], 0.5)
    assert_equal(mean, run([10], 0.5, stopping="sampled"))
    assert mean["final"]["final_ke_MeV"][0] == 0.5
    assert mean["final"]["status"][0] == "model_stop"
    assert mean["slices"]["terminal"][0] == "model_stop"
    assert mean["slices"]["applied_loss_MeV"][0] == 9.5
    assert_conservation(mean, [10], [0.5])


def test_mean_proposal_exactly_exhausts(monkeypatch):
    patch_model(monkeypatch, dedx=10, raw=30)
    result = run([10], 10, fractional_loss=1)
    assert result["final"]["final_ke_MeV"][0] == 0
    assert result["final"]["traveled_mm"][0] == 1
    assert result["final"]["status"][0] == "exhausted"


def test_zero_stopped_and_empty_do_not_call_loss(monkeypatch):
    def forbidden(*args):
        pytest.fail("zero/stopped/empty lanes must not call loss functions")

    monkeypatch.setattr(eloss, "sample_loss", forbidden)
    monkeypatch.setattr(eloss, "dedx_restricted", forbidden)
    result = run([0, 0.25, 1, 30], [12, 12, 0, 0])
    np.testing.assert_array_equal(result["slices"]["lane_index"], [0, 1, 2])
    np.testing.assert_array_equal(result["slices"]["sampled_loss_MeV"], [0, 0, 0])
    np.testing.assert_array_equal(result["final"]["final_ke_MeV"], [0, 0.25, 1, 30])
    np.testing.assert_array_equal(result["final"]["status"], ["exhausted", "model_stop", "model_stop", "transmitted"])
    assert_conservation(result, [0, 0.25, 1, 30], [12, 12, 0, 0])
    for length in (0, 10, []):
        empty = run([], length)
        assert all(value.shape == (0,) for group in empty.values() for value in group.values())
        assert empty["slices"]["lane_index"].dtype.kind in "iu"


def test_vectorized_steps_and_scalar_references(monkeypatch):
    calls = patch_model(monkeypatch, dedx=2, raw=lambda p, step: step * 2)
    initial = np.array([10, 100, 0.5, 100])
    requested = np.array([1, 2.5, 4, 0])
    batch = run(initial, requested)
    np.testing.assert_array_equal(calls[0][1], [0.25, 1])
    np.testing.assert_array_equal(batch["slices"]["lane_index"][:5], [2, 0, 1, 0, 1])
    assert calls[0][2] == 0.1
    assert_conservation(batch, initial, requested)
    for lane in range(len(initial)):
        scalar = run(initial[lane:lane + 1], requested[lane])
        for key in scalar["final"]:
            np.testing.assert_array_equal(batch["final"][key][lane:lane + 1], scalar["final"][key])
        for key in scalar["slices"]:
            if key != "lane_index":
                np.testing.assert_array_equal(batch["slices"][key][batch["slices"]["lane_index"] == lane],
                                              scalar["slices"][key])


def test_zero_dedx_safe_and_length_broadcast(monkeypatch):
    calls = patch_model(monkeypatch, dedx=0, raw=0)
    with np.errstate(all="raise"):
        result = run([10, 100], 2.5)
    assert len(calls) == 3
    np.testing.assert_array_equal(result["final"]["traveled_mm"], [2.5, 2.5])
    np.testing.assert_array_equal(result["final"]["final_ke_MeV"], [10, 100])
    np.testing.assert_array_equal(calls[-1][1], [0.5, 0.5])


@pytest.mark.parametrize("stopping", ["mean-replace", "sampled"])
def test_real_sampler_seed_and_conservation(stopping):
    initial = np.array([0, 0.5, 1, 1.01, 3, 10, 100, 1000])
    requested = np.array([2, 2, 2, 100, 100, 100, 4, 0])
    result = run(initial, requested, stopping=stopping)
    assert_equal(result, run(initial, requested, stopping=stopping))
    assert_conservation(result, initial, requested)
    assert "model_stop" in result["final"]["status"]
    assert "transmitted" in result["final"]["status"]


def test_cap_raises_without_remainder(monkeypatch):
    calls = patch_model(monkeypatch, dedx=0, raw=0)
    with pytest.raises(RuntimeError, match="max_steps=2"):
        run([100], 3, max_steps=2)
    assert len(calls) == 2
    assert run([100], 2, max_steps=2)["final"]["status"][0] == "transmitted"


@pytest.mark.parametrize("ke,length,settings", [
    (1, 1, {}), ([[1]], 1, {}), ([2], [1, 2], {}), ([2], [[1]], {}),
    ([-1], 1, {}), ([np.nan], 1, {}), ([np.inf], 1, {}), ([2], -1, {}),
    ([2], np.nan, {}), ([2], np.inf, {}), ([], -1, {}),
    ([2], 1, {"tcut_MeV": 0}), ([2], 1, {"tcut_MeV": np.inf}),
    ([2], 1, {"tcut_MeV": [1]}), ([2], 1, {"max_step_mm": 0}),
    ([2], 1, {"max_step_mm": np.nan}), ([2], 1, {"fractional_loss": 0}),
    ([2], 1, {"fractional_loss": 1.1}), ([2], 1, {"stop_ke_MeV": 0}),
    ([2], 1, {"stop_ke_MeV": -1}), ([2], 1, {"max_steps": 0}),
    ([2], 1, {"max_steps": 1.5}), ([2], 1, {"max_steps": True}),
    ([2], 1, {"stopping": "unknown"}), ([2], 1, {"stopping": []}),
    ([1e308], 1, {}), ([1e-12], 1, {"stop_ke_MeV": 1e-15}),
])
def test_invalid_inputs_before_sampling(monkeypatch, ke, length, settings):
    calls = patch_model(monkeypatch)
    kwargs = {"tcut_MeV": 0.1, **settings}
    rng = np.random.default_rng(41)
    state = rng.bit_generator.state
    with pytest.raises(ValueError):
        segment.transport_segment(rng, MAT, ke, length, **kwargs)
    assert calls == []
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("mat", [
    None, "G4_POLYSTYRENE", replace(MAT, rho=0), replace(MAT, rho=np.inf),
    replace(MAT, I_eV=-1), replace(MAT, fractions={}),
    replace(MAT, fractions={"Unknown": 1}), replace(MAT, fractions={"C": -1}),
    replace(MAT, fractions={"C": np.nan}), replace(MAT, fractions={"C": 0.2}),
    replace(MAT, sternheimer=(1, 2)), replace(MAT, sternheimer=(1, 2, 1, 1, 1, 1)),
])
def test_invalid_material_even_for_empty_input(monkeypatch, mat):
    calls = patch_model(monkeypatch)
    with pytest.raises(ValueError):
        segment.transport_segment(np.random.default_rng(1), mat, [], [], tcut_MeV=0.1)
    assert calls == []


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_invalid_raw_loss(monkeypatch, bad):
    patch_model(monkeypatch, raw=bad)
    with pytest.raises(ValueError, match="sampled loss"):
        run([10], 1)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_invalid_mean_before_sampling(monkeypatch, bad):
    calls = patch_model(monkeypatch, dedx=bad)
    with pytest.raises(ValueError, match="dedx"):
        run([10], 1)
    assert calls == []


def test_invalid_sampler_shape(monkeypatch):
    patch_model(monkeypatch, raw=lambda p, step: np.array(1.0))
    with pytest.raises(ValueError, match="shape"):
        run([10], 1)


def test_underflow_step_rejected_before_sampling(monkeypatch):
    calls = patch_model(monkeypatch, dedx=1e308)
    with pytest.raises(RuntimeError, match="advance"):
        run([10], 1, fractional_loss=np.nextafter(0.0, 1.0))
    assert calls == []


def test_partial_stop_underflow_rejected(monkeypatch):
    patch_model(monkeypatch, dedx=0, raw=1e308)
    with pytest.raises(RuntimeError, match="advance"):
        run([10], np.nextafter(0.0, 1.0), stopping="sampled")


def test_coordinate_stagnation_rejected(monkeypatch):
    calls = []

    def dedx(mat, p, cut):
        calls.append(1)
        return np.full(p.shape, 0 if len(calls) == 1 else 1e308)

    patch_model(monkeypatch, raw=0)
    monkeypatch.setattr(eloss, "dedx_restricted", dedx)
    with pytest.raises(RuntimeError, match="advance"):
        run([10], 2)


def test_tiny_positive_kinetic_residual_is_not_floored():
    result = run([1e-300], 10)
    assert result["final"]["final_ke_MeV"][0] == 1e-300
    assert result["slices"]["applied_loss_MeV"][0] == 0


def test_inputs_are_not_mutated():
    ke, length = np.array([5.0, 20.0]), np.array([1.0, 2.0])
    original_ke, original_length = ke.copy(), length.copy()
    result = run(ke, length)
    np.testing.assert_array_equal(ke, original_ke)
    np.testing.assert_array_equal(length, original_length)
    result["final"]["final_ke_MeV"][:] = 0
    np.testing.assert_array_equal(ke, original_ke)


def test_experimental_defaults_frozen_seed_regression():
    result = run([0.5, 3, 10, 100], [1, 2, 1, 2.5])
    explicit = run([0.5, 3, 10, 100], [1, 2, 1, 2.5], stopping="mean-replace",
                   max_step_mm=1.0, fractional_loss=0.05, stop_ke_MeV=1.0, max_steps=10000)
    assert_equal(result, explicit)
    np.testing.assert_allclose(result["final"]["final_ke_MeV"],
                               [0.5, 0.9837781554227573, 9.167951087821367, 99.51091875915436],
                               rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(result["final"]["traveled_mm"],
                               [0, 0.6996631309996734, 1, 2.5], rtol=2e-13, atol=1e-14)
    np.testing.assert_array_equal(result["final"]["status"],
                                 ["model_stop", "model_stop", "transmitted", "transmitted"])
    np.testing.assert_array_equal(result["slices"]["lane_index"],
                                 [0, 1, 2, 3, 1, 2, 3, 1, 3] + [1] * 19)


def test_sub_ulp_positive_debit_is_rejected(monkeypatch):
    from mu2efastsim import eloss, materials, segmenttransport
    monkeypatch.setattr(eloss, "dedx_restricted", lambda m, p, c: np.ones_like(p))
    monkeypatch.setattr(eloss, "sample_loss", lambda r, m, p, length, c: np.ones_like(p))
    with pytest.raises(RuntimeError, match="precision"):
        segmenttransport.transport_segment(None, materials.ALUMINIUM, [1e16], 1., tcut_MeV=1.)
