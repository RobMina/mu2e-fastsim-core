"""Synthetic extra-exit hook checks, not a finite-foil geometry implementation."""
from functools import partial
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from mu2efastsim import eslab_jax as SJ  # noqa: E402
from mu2efastsim import materials as M  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    layers = [(M.BY_NAME["G4_Al"], 20.0, 0.03411, 0.00099, 1.0)]
    return {p: SJ.lane_tables(layers, positron=p, msc=True) for p in (False, True)}


def _rates(lt, **rates):
    return lt._replace(tab=lt.tab._replace(**{k: np.full_like(getattr(lt.tab, k), v)
                                             for k, v in rates.items()}))


def _synthetic(lt):
    return _rates(lt, sig_b=0, sig_d=0, sig_a=0, dedx=21, rng=100, lam1=100,
                  hard=0, soft=0.01, soft_all=0.01)


def _state(n, dtype, positron=False):
    return SJ.lane_state(np.full(n, 105, dtype), np.tile(np.array([1, 0, 0], dtype), (n, 1)),
                         np.full(n, 10, dtype), 20.0, positron=positron)


def _mean_loss(key, mc, T, length, tcut, mean, *args, **kwargs):
    return mean


def _bytes_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        av, bv = np.asarray(a[k]), np.asarray(b[k])
        assert av.dtype == bv.dtype and av.shape == bv.shape, k
        assert av.tobytes() == bv.tobytes(), k


def _plane(boundary):
    def first_exit(xx, uu, pm, q, B, smax):
        if B is None:
            g = (boundary - xx[:, 0]) / jnp.where(uu[:, 0] > 0, uu[:, 0], 1)
            return jnp.where((uu[:, 0] > 0) & (g >= 0) & (g <= smax), g, jnp.inf)
        # Cyclic rotation maps the transverse x plane to the existing helix z-plane oracle.
        return SJ._to_plane(xx[:, [1, 2, 0]], uu[:, [1, 2, 0]], pm, q, B[:, [1, 2, 0]], boundary, smax)
    return first_exit


def _private(key, lt, mi, st, B, positron, msc, first_exit, finite_lanes):
    return SJ._lane_step(key, lt, mi, st, B, positron, msc, True, None, False, False,
                          first_exit=first_exit, finite_lanes=finite_lanes)


def _full_em(key, lt, mi, st, B, positron, msc, first_exit, finite_lanes):
    dt, n = SJ.dprec.float_of(st["T"]), st["T"].shape[0]
    tab = SJ.Tables(*(v if isinstance(v, float) else SJ._Lanes(jnp.asarray(v, dt), mi) for v in lt.tab))
    mc = SJ.eloss_jax.MatConsts(*(jnp.asarray(v, dt)[mi] for v in lt.mc))
    L, tcut, kcut, dsmax = (jnp.asarray(v, dt)[mi] for v in (lt.length, lt.tcut, lt.kcut, lt.dsmax))
    interp = lambda v, T: SJ._interp_lanes(v, tab.lnt0, tab.dlnt, T)
    body = SJ._slab_body(key, tab, mc, interp, n, dt, L, tcut, kcut, dsmax, positron, msc,
                         positron, SJ.emsc.MOTT, B, return_emissions=True,
                         first_exit=first_exit, finite_lanes=finite_lanes)
    return body(st)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("msc,field", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_legs_boundary_flush_and_replay(tables, monkeypatch, dtype, positron, msc, field):
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", _mean_loss)
    n, key = 8, jax.random.PRNGKey(17)
    lt, mi = _synthetic(tables[positron]), jnp.zeros(n, jnp.int32)
    st = _state(n, dtype, positron)
    st["pend"], st["path"] = jnp.full(n, 0.1, dtype), jnp.full(n, 3, dtype)
    B = jnp.tile(jnp.array([0, 0, 0.7], dtype), (n, 1)) if field else None
    kw = dict(B=B, positron=positron, msc=msc)
    plan = jnp.full(n, 0.9, dtype)
    ks = jax.random.split(jax.random.fold_in(key, st["it"]), 24)
    tau = jax.random.uniform(ks[3], (n,), dtype)
    leg1 = tau * plan
    pm, q = jnp.sqrt(st["T"] * (st["T"] + 2 * SJ.ME)), 1.0 if positron else -1.0
    if field:
        p1, d1 = SJ._helix(st["x"], st["d"], pm, q, B, leg1)
    else:
        p1, d1 = st["x"] + leg1[:, None] * st["d"], st["d"]
    expo = -jnp.log(jax.random.uniform(ks[4], (n,), dtype, minval=SJ.dprec.tiny_of(dtype)))
    u = jnp.minimum(expo * -jnp.expm1(-plan * 0.01), SJ.emsc.U_MAX)
    dn = SJ._rotate_uz(d1, 1 - u, 2 * jnp.pi * jax.random.uniform(ks[5], (n,), dtype))
    if field:
        end, _ = SJ._helix(p1, dn, pm, q, B, plan - leg1)
    else:
        end = p1 + (plan - leg1)[:, None] * dn
    boundary = jnp.where(jnp.arange(n) % 2 == 0, p1[:, 0] / 2, (p1[:, 0] + end[:, 0]) / 2) if msc else plan / 2
    hook = _plane(boundary)
    finite = jnp.ones(n, bool)
    sn, record = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, first_exit=hook,
                                finite_lanes=finite, **kw))(st)
    primary = jax.jit(lambda s: _private(key, lt, mi, s, B, positron, msc, hook, finite))(st)
    replay_state, replay_record = jax.jit(lambda s: _full_em(key, lt, mi, jax.lax.optimization_barrier(s),
                                                           B, positron, msc, hook, finite))(st)
    _bytes_equal(primary, sn)
    tol = 1e-5 if dtype == np.float32 else 2e-12
    for k in sn:
        np.testing.assert_allclose(sn[k], replay_state[k], atol=tol, rtol=tol)
    for a, b in zip(record, replay_record):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(sn["x"][:, 0], boundary, atol=tol, rtol=tol)
    assert np.asarray(sn["exited"]).all() and not np.asarray(sn["stopped"]).any()
    assert (np.asarray(sn["pend"]) == 0).all()
    np.testing.assert_allclose(sn["cont"], 21 * (sn["path"] - 3 + 0.1), atol=5 * tol, rtol=tol)
    assert not np.asarray(record.valid).any()
    for k in ("nph", "nde", "nan", "annih"):
        assert not np.asarray(sn[k]).any()
    if msc:
        actual_step = np.asarray(sn["path"] - 3)
        assert (actual_step[::2] < np.asarray(leg1)[::2]).all()
        assert (actual_step[1::2] > np.asarray(leg1)[1::2]).all()


@pytest.mark.parametrize("kind", ["brem", "delta", "annih"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hard_event_before_after_and_at_boundary(tables, monkeypatch, kind, dtype):
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", _mean_loss)
    n, key = 4, jax.random.PRNGKey(29)
    lt = _synthetic(tables[True])
    lt = _rates(lt, **{"sig_b" if kind == "brem" else "sig_d" if kind == "delta" else "sig_a": 10})
    st, mi = _state(n, dtype, True), jnp.zeros(n, jnp.int32)
    ks = jax.random.split(jax.random.fold_in(key, st["it"]), 24)
    free = -jnp.log(jax.random.uniform(ks[1 if kind == "annih" else 0], (n,), dtype,
                                      minval=SJ.dprec.tiny_of(dtype))) / 10
    boundary = free * jnp.array([0.5, 2, 1, 0.5], dtype)
    plane = _plane(boundary)

    def hook(xx, uu, pm, q, B, smax):
        # An exact plan-distance tie avoids host/device rounding in the independent free-path oracle.
        return jnp.where(jnp.arange(n) == 2, smax, plane(xx, uu, pm, q, B, smax))

    sn, p = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, positron=True, msc=False,
                            first_exit=hook))(st)
    raw, rp = jax.jit(lambda s: _full_em(key, lt, mi, jax.lax.optimization_barrier(s),
                                       None, True, False, hook, None))(st)
    tol = 2e-5 if dtype == np.float32 else 2e-12
    for k in sn:
        np.testing.assert_allclose(sn[k], raw[k], atol=tol, rtol=tol)
    for a, b in zip(p, rp):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sn["exited"], [True, False, True, True])
    count = np.asarray(sn[{"brem": "nph", "delta": "nde", "annih": "nan"}[kind]])
    np.testing.assert_array_equal(count, [0, 2 if kind == "annih" else 1, 0, 0])
    expected = np.zeros((n, 3), bool)
    if kind != "delta":
        expected[1, 1:] = kind == "annih"
        expected[1, 0] = kind == "brem"
    np.testing.assert_array_equal(p.valid, expected)
    assert (np.asarray(p.position)[:, :, 0] <= np.asarray(boundary)[:, None] + 1e-6).all()
    assert not np.asarray(sn["annih"])[[0, 2, 3]].any()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_range_out_at_boundary_retains_rest_births(tables, monkeypatch, dtype):
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", lambda key, mc, T, *a, **kw: 2 * T)
    n, key = 4, jax.random.PRNGKey(7)
    st, mi = _state(n, dtype, True), jnp.zeros(n, jnp.int32)
    boundary = jnp.full(n, 0.02, dtype)
    sn, p = jax.jit(partial(SJ.lane_step_emissions, key, _synthetic(tables[True]), mi,
                            positron=True, msc=False, first_exit=_plane(boundary)))(st)
    assert np.asarray(sn["stopped"]).all() and not np.asarray(sn["exited"]).any()
    assert np.asarray(p.valid)[:, 1:].all() and not np.asarray(p.valid)[:, 0].any()
    np.testing.assert_array_equal(p.mechanism[:, 1:], 3)
    np.testing.assert_allclose(p.position[:, 1:, 0], 0.02)
    np.testing.assert_allclose(p.energy[:, 1:], SJ.ME)
    np.testing.assert_array_equal(sn["nan"], 2)
    again, p2 = jax.jit(partial(SJ.lane_step_emissions, key, _synthetic(tables[True]), mi,
                               positron=True, msc=False, first_exit=_plane(boundary)))(sn)
    assert not np.asarray(p2.valid).any()
    for k in sn.keys() - {"it"}:
        np.testing.assert_array_equal(again[k], sn[k])


def test_finite_trapped_mask_default_and_inactive(tables, monkeypatch):
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", _mean_loss)
    n, key = 6, jax.random.PRNGKey(2)
    lt = _rates(_synthetic(tables[False]), rng=0.01, soft=0, soft_all=0, lam1=100)
    lt = lt._replace(dsmax=np.full_like(lt.dsmax, 0.2))
    st, mi = _state(n, np.float64), jnp.zeros(n, jnp.int32)
    st["stopped"] = st["stopped"].at[4].set(True)
    st["exited"] = st["exited"].at[5].set(True)
    finite = jnp.array([True, False, True, False, True, True])
    hook = _plane(jnp.full(n, 0.5))
    kw = dict(msc=True, first_exit=hook)
    sn, p = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, finite_lanes=finite, **kw))(st)
    np.testing.assert_allclose(sn["path"][:4], [0.2, 1, 0.2, 1])
    np.testing.assert_allclose(sn["pend"][:4], [0.2, 0, 0.2, 0])
    assert not np.asarray(p.valid).any()
    for k in st.keys() - {"it"}:
        np.testing.assert_array_equal(sn[k][4:], st[k][4:])
    all_sn, _ = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, **kw))(st)
    np.testing.assert_allclose(all_sn["path"][:4], 0.2)
    off, _ = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, finite_lanes=False, **kw))(st)
    plain = jax.jit(partial(SJ.lane_step, key, lt, mi, msc=True))(st)
    _bytes_equal(off, plain)


@pytest.mark.parametrize("field", [False, True])
def test_existing_z_faces_still_win(tables, monkeypatch, field):
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", _mean_loss)
    n, key = 4, jax.random.PRNGKey(11)
    st, mi = _state(n, np.float64), jnp.zeros(n, jnp.int32)
    st["d"] = jnp.tile(jnp.array([0.8, 0, 0.6]), (n, 1))
    st["x"] = st["x"].at[:, 2].set(19.94)
    B = jnp.tile(jnp.array([0, 0, 0.7]), (n, 1)) if field else None
    sn, p = jax.jit(partial(SJ.lane_step_emissions, key, _synthetic(tables[False]), mi,
                            B=B, msc=False, first_exit=_plane(jnp.full(n, 0.5))))(st)
    assert np.asarray(sn["exited"]).all() and not np.asarray(p.valid).any()
    np.testing.assert_allclose(sn["x"][:, 2], 20, atol=1e-12)
    np.testing.assert_allclose(sn["path"], 0.1, atol=1e-12)


@pytest.fixture(scope="module")
def old_module():
    path = os.environ.get("ESLAB_BOUNDARY_BASELINE")
    if not path:
        pytest.skip("set ESLAB_BOUNDARY_BASELINE to an external pre-edit module snapshot")
    spec = importlib.util.spec_from_file_location("mu2efastsim._eslab_boundary_baseline", Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("msc,field", [(False, False), (True, False), (False, True), (True, True)])
def test_default_bytes_external_preedit_snapshot(old_module, tables, dtype, positron, msc, field):
    n, key = 8, jax.random.PRNGKey(19)
    st, lt, mi = _state(n, dtype, positron), tables[positron], jnp.zeros(n, jnp.int32)
    st["T"] = jnp.asarray(np.geomspace(0.02, 105, n).astype(dtype))
    st["d"] = st["d"].at[:, 2].set(0.6).at[:, 0].set(0.8)
    st["x"] = st["x"].at[0, 2].set(19.999)
    B = jnp.tile(jnp.array([0.7, 0.2, 0], dtype), (n, 1)) if field else None
    kw = dict(B=B, positron=positron, msc=msc)
    for name in ("lane_step", "lane_step_photons", "lane_step_emissions"):
        def run(module, state):
            def step(s, _):
                out = getattr(module, name)(key, lt, mi, s, **kw)
                sn, record = (out, ()) if name == "lane_step" else out
                return sn, (sn, record)
            return jax.lax.scan(step, state, None, length=3)
        a = jax.jit(partial(run, SJ))(st)
        b = jax.jit(partial(run, old_module))(st)
        for av, bv in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            assert np.asarray(av).dtype == np.asarray(bv).dtype
            assert np.asarray(av).tobytes() == np.asarray(bv).tobytes(), name
