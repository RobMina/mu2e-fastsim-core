"""Synthetic energy-ledger regressions; no detector files or Geant4 reference needed."""
import numpy as np
import pytest

from mu2efastsim import eloss, materials, transport as T


def _piece(name, bottom, top):
    planes = np.array([[1, 0, 0, 10], [-1, 0, 0, 10], [0, 1, 0, top],
                       [0, -1, 0, -bottom], [0, 0, 1, 10], [0, 0, -1, 10]], float)
    return name, planes, np.array([-10., bottom, -10.]), np.array([10., top, 10.])


def _remaining_p(p, loss):
    kinetic = np.maximum(np.hypot(p, eloss.MMU) - eloss.MMU - loss, 0.)
    return np.sqrt(kinetic * (kinetic + 2 * eloss.MMU))


@pytest.mark.parametrize("banded", [False, True])
@pytest.mark.parametrize("edges", [[3., 0.], [3., 2., 1., 0.]])
def test_world_slabs_and_air_receive_remaining_momentum(monkeypatch, banded, edges):
    p = np.array([100., 300., 3000.])
    o = np.tile([0., 3., 0.], (len(p), 1))
    d = np.tile([0., -1., 0.], (len(p), 1))
    pieces = [_piece(materials.ALUMINIUM.name, 2., 3.),
              _piece(materials.CONCRETE_MARS.name, 1., 2.)]
    world = T.BandedWorld(pieces, edges) if banded else pieces
    slabs, air = [], []

    def slab(rng, mat, momentum, length):
        slabs.append((mat, momentum.copy()))
        return np.full_like(momentum, 20. if mat == materials.ALUMINIUM else 5.)

    def dedx(mat, momentum, cut):
        assert mat == materials.AIR
        air.append(momentum.copy())
        return np.full_like(momentum, 2.)

    monkeypatch.setattr(T, "slab_loss", slab)
    monkeypatch.setattr(eloss, "dedx_restricted", dedx)
    out = T.world_loss(np.random.default_rng(1), o, d, p, world, edges,
                       return_band_p=True, return_tail=True, return_band_len=True)
    np.testing.assert_allclose(slabs[0][1], p)
    first_loss = 20. if slabs[0][0] == materials.ALUMINIUM else 5.
    np.testing.assert_allclose(slabs[1][1], _remaining_p(p, first_loss))
    np.testing.assert_allclose(air[-1], _remaining_p(p, 25.))
    np.testing.assert_allclose(out[0], _remaining_p(p, 27.))
    assert out[2].all()
    assert all(np.isfinite(value).all() for value in out)
    np.testing.assert_allclose(out[-1].sum(axis=1), 3.)
    losses_in = [0.] if len(edges) == 2 else [0., 20., 25.]
    losses_out = [27.] if len(edges) == 2 else [20., 25., 27.]
    expected = np.column_stack([.5 * (_remaining_p(p, a) + _remaining_p(p, b))
                                for a, b in zip(losses_in, losses_out)])
    np.testing.assert_allclose(out[5], expected)


def test_world_stopped_lane_never_reenters_a_slab_with_positive_momentum(monkeypatch):
    p = np.array([100., 300.])
    pieces = [_piece(materials.ALUMINIUM.name, 1., 2.),
              _piece(materials.CONCRETE_MARS.name, 0., 1.)]
    seen = []

    def slab(rng, mat, momentum, length):
        seen.append(momentum.copy())
        return np.hypot(momentum, eloss.MMU) - eloss.MMU

    monkeypatch.setattr(T, "slab_loss", slab)
    out = T.world_loss(np.random.default_rng(1), np.tile([0., 2., 0.], (2, 1)),
                       np.tile([0., -1., 0.], (2, 1)), p, pieces, [2., 0.])
    np.testing.assert_allclose(seen[1], 0., atol=1e-6)
    np.testing.assert_array_equal(out[0], 0.)
    assert not out[2].any()


def test_world_air_overloss_returns_zero_momentum(monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 1000.))
    out = T.world_loss(np.random.default_rng(1), np.array([[0., 1., 0.]]),
                       np.array([[0., -1., 0.]]), np.array([100.]), [], [1., 0.], return_band_p=True)
    np.testing.assert_array_equal(out[0], 0.)
    assert not out[2].any()
    np.testing.assert_allclose(out[5], [[50.]])


def test_world_empty_input():
    out = T.world_loss(np.random.default_rng(1), np.empty((0, 3)), np.empty((0, 3)),
                       np.empty(0), [], [1., 0.], return_band_p=True, return_tail=True, return_band_len=True)
    assert all(value.shape[0] == 0 for value in out)


def test_p_after_energy_identity_and_stop_floor():
    p = np.array([0., 1., 100., 300., 3000.])
    kinetic = np.hypot(p, eloss.MMU) - eloss.MMU
    for loss in (np.zeros_like(p), .5 * kinetic, kinetic, kinetic + 1000.):
        np.testing.assert_allclose(T._p_after(p, loss), _remaining_p(p, loss), atol=1e-9)
