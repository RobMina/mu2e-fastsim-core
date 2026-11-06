"""Per-step bremsstrahlung emission records, not annihilation photons, showers or exit flux."""
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
jax = pytest.importorskip("jax")
jnp = jax.numpy

from mu2efastsim import eslab_jax as SJ  # noqa: E402
from mu2efastsim import materials as M  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    layers = [(M.BY_NAME["G4_W"], 20.0, 0.0814923, 0.00770222, 0.5),
              (M.BY_NAME["G4_Al"], 20.0, 0.03411, 0.00099, 0.5)]
    return {(positron, msc): SJ.lane_tables(layers, positron=positron, msc=msc)
            for positron in (False, True) for msc in (False, True)}


def _assert_bytes_equal(a, b):
    assert a.keys() == b.keys()
    for name in a:
        av, bv = np.asarray(a[name]), np.asarray(b[name])
        assert av.dtype == bv.dtype and av.shape == bv.shape, name
        assert av.tobytes() == bv.tobytes(), name


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("msc,field", [(False, False), (True, False), (False, True), (True, True)])
def test_brems_records_match_state_only_steps(tables, dtype, positron, msc, field):
    n, steps = 64, 12
    lt = tables[positron, msc]
    mi = jnp.arange(n) % 2
    T = np.full(n, 105.0, dtype)
    T[:8] = 0.0
    d = np.tile(np.array([0.0, 0.0, 1.0], dtype), (n, 1))
    z0 = np.full(n, 10.0, dtype)
    z0[8:16] = 20.0
    st = SJ.lane_state(T, d, z0, lt.length[np.asarray(mi)], positron=positron)
    st["path"] = jnp.full(n, 3.0, dtype)
    st["nph"] = jnp.full(n, 2, jnp.int32)
    st["eph"] = jnp.full(n, 0.25, dtype)
    B = jnp.tile(jnp.array([0.7, 0.2, 0.0], dtype), (n, 1)) if field else None
    kw = dict(B=B, positron=positron, msc=msc)

    @jax.jit
    def plain(key, state):
        def step(s, _):
            sn = SJ.lane_step(key, lt, mi, s, **kw)
            return sn, sn
        return jax.lax.scan(step, state, None, length=steps)

    @jax.jit
    def recorded(key, state):
        def step(s, _):
            sn, photon = SJ.lane_step_photons(key, lt, mi, s, **kw)
            return sn, (sn, photon)
        return jax.lax.scan(step, state, None, length=steps)

    for seed in (4, 19):
        key = jax.random.PRNGKey(seed)
        final, history = plain(key, st)
        rfinal, (rhistory, photons) = recorded(key, st)
        _assert_bytes_equal(final, rfinal)
        _assert_bytes_equal(history, rhistory)
        p = jax.tree_util.tree_map(np.asarray, photons)
        h = jax.tree_util.tree_map(np.asarray, rhistory)
        assert isinstance(photons, SJ.BremsPhotonRecord)
        assert p.valid.shape == (steps, n) and p.valid.dtype == np.bool_
        assert p.energy.shape == p.path.shape == (steps, n)
        assert p.position.shape == p.direction.shape == (steps, n, 3)
        for values in (p.energy, p.position, p.direction, p.path):
            assert values.dtype == dtype
            assert np.isfinite(values).all()
            assert (values[~p.valid] == 0).all()
        assert not p.valid[:, :16].any()
        assert p.valid.any(), "small sample must exercise brems emission"
        previous_n = np.concatenate([np.asarray(st["nph"])[None], h["nph"][:-1]])
        previous_e = np.concatenate([np.asarray(st["eph"])[None], h["eph"][:-1]])
        previous_T = np.concatenate([T[None], h["T"][:-1]])
        previous_finished = np.concatenate([(np.asarray(st["stopped"]) | np.asarray(st["exited"]))[None],
                                             (h["stopped"] | h["exited"])[:-1]])
        np.testing.assert_array_equal(h["nph"] - previous_n, p.valid)
        tol = 3e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(h["eph"] - previous_e, p.energy, atol=tol, rtol=tol)
        assert not p.valid[previous_finished].any()
        assert (p.energy >= 0).all()
        cuts = lt.kcut[np.asarray(mi)]
        assert (p.energy[p.valid] >= np.broadcast_to(cuts, p.valid.shape)[p.valid] * (1 - tol)).all()
        assert (p.energy[p.valid] <= previous_T[p.valid] * (1 + tol)).all()
        np.testing.assert_allclose(np.linalg.norm(p.direction[p.valid], axis=-1), 1.0, atol=tol)
        np.testing.assert_array_equal(p.position[p.valid], h["x"][p.valid])
        np.testing.assert_array_equal(p.path[p.valid], h["path"][p.valid])
        if not msc and not field:
            previous_d = np.concatenate([d[None], h["d"][:-1]])
            previous_c = np.concatenate([np.asarray(st["cont"])[None], h["cont"][:-1]])
            Tp = previous_T - (h["cont"] - previous_c)
            mom = np.sqrt(np.maximum(Tp * (Tp + 2 * SJ.ME), 0.0))
            recoil = mom[p.valid, None] * previous_d[p.valid] - p.energy[p.valid, None] * p.direction[p.valid]
            recoil /= np.linalg.norm(recoil, axis=-1, keepdims=True)
            np.testing.assert_allclose(h["d"][p.valid], recoil, atol=tol, rtol=tol)


def test_annihilation_is_not_a_brems_record(tables):
    lt = tables[True, False]
    tab = lt.tab._replace(sig_b=np.zeros_like(lt.tab.sig_b), sig_d=np.zeros_like(lt.tab.sig_d),
                          sig_a=np.full_like(lt.tab.sig_a, 1e12), dedx=np.zeros_like(lt.tab.dedx))
    lt = lt._replace(tab=tab)
    n = 64
    mi = jnp.zeros(n, jnp.int32)
    st = SJ.lane_state(np.ones(n), np.tile([0.0, 0.0, 1.0], (n, 1)), np.full(n, 10.0), 20.0, positron=True)
    key = jax.random.PRNGKey(7)
    kw = dict(positron=True, msc=False)
    plain = jax.jit(partial(SJ.lane_step, key, lt, mi, **kw))(st)
    result, photons = jax.jit(partial(SJ.lane_step_photons, key, lt, mi, **kw))(st)
    _assert_bytes_equal(plain, result)
    assert np.asarray(result["annih"]).all()
    assert (np.asarray(result["nan"]) == 2).all()
    assert not np.asarray(photons.valid).any()
    for values in photons:
        assert (np.asarray(values) == 0).all()


def test_empty_and_finished_lanes_have_safe_zero_records(tables):
    lt = tables[True, False]
    for n in (0, 8):
        mi = jnp.zeros(n, jnp.int32)
        st = SJ.lane_state(np.zeros(n, np.float32), np.zeros((n, 3), np.float32), np.zeros(n), 20.0,
                           positron=True)
        result, photons = jax.jit(partial(SJ.lane_step_photons, jax.random.PRNGKey(1), lt, mi,
                                         positron=True, msc=False))(st)
        assert photons.energy.shape == photons.path.shape == photons.valid.shape == (n,)
        assert photons.position.shape == photons.direction.shape == (n, 3)
        for values in photons:
            assert np.isfinite(values).all() and (np.asarray(values) == 0).all()
        assert not np.asarray(result["nph"]).any()


@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("msc,field", [(False, False), (True, False), (False, True), (True, True)])
def test_finite_state_only_matches_records_with_physical_tables(tables, positron, msc, field):
    from mu2efastsim.geometry import annulus_jax as A

    n, steps = 32, 6
    lt = tables[positron, msc]
    mi = jnp.arange(n) % 2
    T = np.full(n, 105.0)
    T[:4] = 0.0
    st = SJ.lane_state(T, np.tile([0.6, 0.0, 0.8], (n, 1)), np.full(n, 10.0), 20.0,
                       positron=positron)
    st["x"] = st["x"].at[:, 0].set(jnp.where(jnp.arange(n) < 16, 4.99, 2.0))
    B = jnp.tile(jnp.array([0.0, 0.0, 2.0]), (n, 1)) if field else None

    def supported_or_inactive(supported, proposed):
        assert np.all(np.asarray(supported) | (np.asarray(proposed) == 0))

    def boundary(x, d, p, q, B, smax):
        crossing = A.crossing(x, d, p, q, B, smax, 10.0, 10.0, 1.0, 5.0)
        jax.debug.callback(supported_or_inactive, crossing.supported, smax, ordered=True)
        return jnp.where(smax > 0, crossing.distance, 0.0)

    kw = dict(B=B, positron=positron, msc=msc, boundary=boundary,
              safety=lambda x: A.safety(x, 10.0, 10.0, 1.0, 5.0))

    @jax.jit
    def plain(key, state):
        def step(s, _):
            sn = SJ.lane_step(key, lt, mi, s, **kw)
            return sn, sn
        return jax.lax.scan(step, state, None, length=steps)

    @jax.jit
    def recorded(key, state):
        def step(s, _):
            sn, photon = SJ.lane_step_photons(key, lt, mi, s, **kw)
            return sn, (sn, photon)
        return jax.lax.scan(step, state, None, length=steps)

    for seed in (4, 19):
        key = jax.random.PRNGKey(seed)
        final, history = plain(key, st)
        rfinal, (rhistory, photons) = recorded(key, st)
        for left, right in ((final, rfinal), (history, rhistory)):
            assert left.keys() == right.keys()
            for name in left:
                av, bv = np.asarray(left[name]), np.asarray(right[name])
                assert av.dtype == bv.dtype and av.shape == bv.shape, name
                if np.issubdtype(av.dtype, np.inexact):
                    np.testing.assert_allclose(av, bv, rtol=2e-12, atol=2e-12, err_msg=name)
                    assert np.isfinite(av).all(), name
                else:
                    np.testing.assert_array_equal(av, bv, err_msg=name)
        assert np.asarray(final["exited"])[4:16].all()
        assert np.asarray(photons.valid).any()
        np.testing.assert_array_equal(np.asarray(rhistory["nph"])[-1], np.asarray(photons.valid).sum(axis=0))
        assert not np.asarray(photons.valid)[:, :4].any()
