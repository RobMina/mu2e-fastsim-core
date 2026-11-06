"""Finite-material per-leg hooks; synthetic rates isolate transport from fluctuation statistics."""
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from mu2efastsim import eslab_jax as SJ
from mu2efastsim import materials as M


N = 8
KEY = jax.random.PRNGKey(19)


def _tables(n=N, **changes):
    def grid(value):
        return np.broadcast_to(np.asarray(value)[:, None] if np.ndim(value) else value, (n, SJ.NT)).copy()

    values = {name: grid(0.0) for name in SJ.Tables._fields}
    values.update(lnt0=float(SJ._grid()[0][0]), dlnt=float(np.diff(SJ._grid()[0])[0]), ff_tmin=0.0,
                  dedx=grid(0.5), lam1=grid(1e6), rng=grid(1e6), uc=grid(0.1), soft=grid(0.5),
                  soft_all=grid(0.9), ue=grid(1.0), hard_cum=np.ones((n, 2, SJ.NT)),
                  scr=np.full((n, 1, SJ.NT), 0.01), env=np.ones((n, 1, SJ.NT)),
                  brem_cum=np.ones((n, 1, SJ.NT)), brem_x=np.full((n, SJ.NT, SJ.NX), 0.25),
                  z=np.full((n, 1), 13.0), ffa=np.zeros((n, 1)), mott=np.zeros((n, 1, 5, 6)))
    values.update({name: grid(value) for name, value in changes.items()})
    mc = SJ.eloss_jax.consts(M.BY_NAME["G4_Al"])
    mc = SJ.eloss_jax.MatConsts(*(np.full(n, v) for v in mc))
    return SJ.LaneTables(SJ.Tables(**values), mc, np.full(n, 20.0), np.full(n, 0.05),
                         np.full(n, 0.01), np.ones(n))


def _state(n=N, leg=2, dtype=np.float64, positron=False):
    d = np.tile([1.0, 0.0, 0.0] if leg == 1 else [0.0, 0.0, 1.0], (n, 1)).astype(dtype)
    st = SJ.lane_state(np.full(n, 100.0, dtype), d, np.full(n, 10.0, dtype), 20.0, positron=positron)
    st.update(pend=jnp.full(n, 0.3, dtype), path=jnp.full(n, 4.0, dtype))
    return st


def _field(field, n=N, dtype=np.float64):
    return jnp.tile(jnp.array([0.0, 0.0, 2.0], dtype), (n, 1)) if field else None


def _keys(st):
    return jax.random.split(jax.random.fold_in(KEY, st["it"]), 24)


def _expo(key, n=N, dtype=jnp.float64):
    return -jnp.log(jax.random.uniform(key, (n,), dtype, minval=SJ.dprec.tiny_of(dtype)))


def _advance(x, d, p, q, B, s):
    return (x + s[:, None] * d, d) if B is None else SJ.helix(x, d, p, q, B, s)


def _legs(st, B, positron=False, plan=None, soft=0.5):
    n, dt = st["T"].size, st["T"].dtype
    plan = jnp.ones(n, dt) if plan is None else plan
    ks = _keys(st)
    p = jnp.sqrt(st["T"] * (st["T"] + 2 * SJ.ME))
    q = 1.0 if positron else -1.0
    s1 = jax.random.uniform(ks[3], (n,), dt) * plan
    x1, d1 = _advance(st["x"], st["d"], p, q, B, s1)
    u = jnp.minimum(_expo(ks[4], n, dt) * -jnp.expm1(-plan * soft), SJ.emsc.U_MAX)
    d2 = SJ._rotate_uz(d1, 1 - u, 2 * jnp.pi * jax.random.uniform(ks[5], (n,), dt))
    return p, q, s1, x1, d2, plan - s1


def _wall_boundary(wall, side, calls=None):
    """Complete exit for z in [0,20] and side*x <= side*wall, on these short monotone test arcs."""
    def boundary(x, d, p, q, B, smax):
        if calls is not None:
            jax.debug.callback(lambda *args: calls.append(tuple(np.asarray(a) for a in args)),
                               x, d, p, q, smax, ordered=True)
        endpoint, _ = _advance(x, d, p, q, B, smax)
        outside = side * (x[:, 0] - wall) >= 0
        cross = side * (endpoint[:, 0] - wall) >= 0

        def bisect(_, bounds):
            lo, hi = bounds
            mid = (lo + hi) * 0.5
            xm, _ = _advance(x, d, p, q, B, mid)
            beyond = side * (xm[:, 0] - wall) >= 0
            return jnp.where(beyond, lo, mid), jnp.where(beyond, mid, hi)

        lo, hi = jax.lax.fori_loop(0, 48, bisect, (jnp.zeros_like(smax), smax))
        lateral = jnp.where(outside, 0.0, jnp.where(cross, (lo + hi) * 0.5, jnp.inf))
        dz = d[:, 2]
        axial = jnp.where(dz > 0, (20 - x[:, 2]) / jnp.where(dz > 0, dz, 1.0),
                          jnp.where(dz < 0, -x[:, 2] / jnp.where(dz < 0, dz, -1.0), jnp.inf))
        axial = jnp.where((axial >= 0) & (axial <= smax), axial, jnp.inf)
        return jnp.minimum(lateral, axial)
    return boundary


def _exit_case(st, B, leg, positron=False, plan=None, calls=None):
    p, q, s1, x1, d2, s2 = _legs(st, B, positron, plan)
    if leg == 1:
        distance = 0.4 * s1
        xn, dn = _advance(st["x"], st["d"], p, q, B, distance)
    else:
        distance = s1 + 0.4 * s2
        xn, dn = _advance(x1, d2, p, q, B, 0.4 * s2)
    wall = xn[:, 0]
    side = jnp.where(wall >= 0, 1.0, -1.0)
    safety = lambda x: jnp.maximum(jnp.minimum(side * (wall - x[:, 0]), jnp.minimum(x[:, 2], 20 - x[:, 2])), 0.0)
    return _wall_boundary(wall, side, calls), safety, distance, xn, dn, (p, q, s1, x1, d2, s2)


@pytest.fixture
def mean_loss(monkeypatch):
    def loss(key, mc, T, length, tcut, mean, positron, **kwargs):
        return mean
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", loss)


def _step(lt, st, **kwargs):
    fn = partial(SJ.lane_step_photons, KEY, lt, jnp.arange(st["T"].size), mott=False, **kwargs)
    return jax.jit(fn)(st)


def _no_records(photons):
    for value in photons:
        assert np.isfinite(value).all()
        assert not np.asarray(value).any()


def _bytes(a, b):
    for av, bv in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        av, bv = np.asarray(av), np.asarray(bv)
        assert av.shape == bv.shape and av.dtype == bv.dtype
        assert av.tobytes() == bv.tobytes()


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("leg", [1, 2])
def test_forced_msc_exit_uses_actual_leg_and_flushes_pending(mean_loss, field, positron, leg):
    st, B, calls = _state(leg=leg, positron=positron), _field(field), []
    boundary, safety, distance, xn, dn, legs = _exit_case(st, B, leg, positron, calls=calls)
    result, photons = _step(_tables(), st, B=B, positron=positron, boundary=boundary, safety=safety)
    jax.block_until_ready(result)
    assert len(calls) == 2
    p, q, s1, x1, d2, s2 = legs
    np.testing.assert_allclose(calls[0][0], st["x"], atol=1e-13)
    np.testing.assert_allclose(calls[0][1], st["d"], atol=1e-13)
    np.testing.assert_allclose(calls[0][2], p, atol=1e-13)
    assert calls[0][3] == q
    np.testing.assert_allclose(calls[0][4], s1, atol=1e-13)
    np.testing.assert_allclose(calls[1][0], xn if leg == 1 else x1, atol=1e-12)
    np.testing.assert_allclose(calls[1][1], dn if leg == 1 else d2, atol=1e-12)
    np.testing.assert_allclose(calls[1][2], p, atol=1e-13)
    np.testing.assert_allclose(calls[1][4], 0.0 if leg == 1 else s2, atol=1e-13)
    np.testing.assert_allclose(result["x"], xn, atol=1e-12)
    np.testing.assert_allclose(result["d"], dn, atol=1e-12)
    np.testing.assert_allclose(result["path"] - st["path"], distance, atol=1e-12)
    np.testing.assert_allclose(result["cont"], 0.5 * (st["pend"] + distance), atol=1e-12)
    np.testing.assert_allclose(result["T"] + result["cont"], st["T"], atol=1e-12)
    assert np.asarray(result["exited"]).all()
    assert not np.asarray(result["stopped"]).any()
    assert not np.asarray(result["pend"]).any()
    _no_records(photons)


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("leg", [1, 2])
@pytest.mark.parametrize("interaction", ["brems", "delta", "hard", "annih"])
def test_exit_suppresses_endpoint_interactions(mean_loss, field, leg, interaction):
    st, B = _state(leg=leg, positron=True), _field(field)
    index, name = {"brems": (0, "sig_b"), "delta": (0, "sig_d"),
                   "hard": (2, "hard"), "annih": (1, "sig_a")}[interaction]
    lt = _tables(**{name: np.asarray(_expo(_keys(st)[index]))})
    lt = lt._replace(dsmax=np.full(N, 2.0))
    boundary, safety, distance, xn, dn, _ = _exit_case(st, B, leg, True)
    result, photons = _step(lt, st, B=B, positron=True, boundary=boundary, safety=safety)
    control, cp = _step(lt, st, B=B, positron=True)
    if interaction == "hard":
        p, q, _, x1, d2, s2 = _legs(st, B, True)
        _, no_hard = _advance(x1, d2, p, q, B, s2)
        assert np.max(np.abs(np.asarray(control["d"] - no_hard))) > 1e-3
    else:
        assert np.asarray(control[{"brems": "nph", "delta": "nde", "annih": "nan"}[interaction]]).all()
    if interaction == "brems":
        assert np.asarray(cp.valid).all()
    for name in ("nph", "eph", "nde", "ede", "nan", "ean", "annih"):
        assert not np.asarray(result[name]).any(), name
    assert np.asarray(result["exited"]).all()
    np.testing.assert_allclose(result["d"], dn, atol=1e-12)
    np.testing.assert_allclose(result["x"], xn, atol=1e-12)
    np.testing.assert_allclose(result["path"], st["path"] + distance, atol=1e-12)
    np.testing.assert_allclose(result["cont"], 0.5 * (st["pend"] + distance), atol=1e-12)
    assert not np.asarray(result["pend"]).any()
    _no_records(photons)


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("leg", [1, 2])
def test_stopping_precedes_boundary_exit(monkeypatch, field, leg):
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", lambda key, mc, T, *args, **kwargs: T)
    st, B = _state(leg=leg, positron=True), _field(field)
    boundary, safety, distance, _, _, _ = _exit_case(st, B, leg, True)
    result, photons = _step(_tables(), st, B=B, positron=True, boundary=boundary, safety=safety)
    assert np.asarray(result["stopped"]).all()
    assert not np.asarray(result["exited"]).any()
    assert not np.asarray(result["annih"]).any()
    np.testing.assert_array_equal(result["T"], 0.0)
    np.testing.assert_array_equal(result["cont"], st["T"])
    np.testing.assert_array_equal(result["nan"], 2)
    np.testing.assert_allclose(result["ean"], 2 * SJ.ME)
    np.testing.assert_allclose(result["path"], st["path"] + distance, atol=1e-12)
    _no_records(photons)


@pytest.mark.parametrize("field", [False, True])
def test_lateral_safety_prevents_planar_range_trapping(mean_loss, field):
    st, B, calls = _state(leg=1), _field(field), []
    lt = _tables(rng=0.2)
    boundary = _wall_boundary(jnp.full(N, 0.1), jnp.ones(N), calls)
    safety = lambda x: jnp.maximum(jnp.minimum(0.1 - x[:, 0], jnp.minimum(x[:, 2], 20 - x[:, 2])), 0.0)
    result, _ = _step(lt, st, B=B, boundary=boundary, safety=safety)
    jax.block_until_ready(result)
    expected = jax.random.uniform(_keys(st)[3], (N,), jnp.float64)
    np.testing.assert_allclose(calls[0][4], expected, atol=1e-13)
    planar_calls = []
    _step(lt, st, B=B, boundary=_wall_boundary(jnp.full(N, 0.1), jnp.ones(N), planar_calls),
          safety=lambda x: jnp.minimum(x[:, 2], 20 - x[:, 2]))[0]["T"].block_until_ready()
    rem = max(SJ.eslab.STEP_FRACTION * 100 / 0.5, SJ.eslab.FINAL_RANGE) - 0.3
    np.testing.assert_allclose(planar_calls[0][4], expected * rem, atol=1e-12)
    assert np.max(np.abs(calls[0][4] - planar_calls[0][4])) > 1.0


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("msc", [False, True])
def test_inactive_rows_receive_zero_proposals_and_remain_inert(mean_loss, field, msc):
    st, B, calls = _state(), _field(field), []
    st["stopped"] = st["stopped"].at[0].set(True)
    st["exited"] = st["exited"].at[1].set(True)
    st["pend"] = st["pend"].at[:2].set(1000.0)

    def boundary(x, d, p, q, B, smax):
        jax.debug.callback(lambda s: calls.append(np.asarray(s)), smax, ordered=True)
        return jnp.zeros_like(smax)

    result, photons = _step(_tables(), st, B=B, msc=msc, boundary=boundary,
                            safety=lambda x: jnp.zeros(x.shape[0], x.dtype))
    jax.block_until_ready(result)
    assert len(calls) == (2 if msc else 1)
    assert all((s >= 0).all() and (s[:2] == 0).all() for s in calls)
    if msc:
        np.testing.assert_array_equal(calls[1], 0.0)
    for name in st:
        if name != "it":
            _bytes(st[name][:2], result[name][:2])
    np.testing.assert_array_equal(result["path"], st["path"])
    _no_records(photons)


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("msc", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_null_hooks_and_far_boundary_preserve_exact_state_and_records(field, msc, dtype):
    st, B, lt = _state(dtype=dtype), _field(field, dtype=dtype), _tables(sig_b=3.0)
    omitted = _step(lt, st, B=B, msc=msc)
    explicit = _step(lt, st, B=B, msc=msc, boundary=None, safety=None)
    far = _step(lt, st, B=B, msc=msc,
                boundary=lambda x, d, p, q, B, smax: jnp.full_like(smax, jnp.inf),
                safety=lambda x: jnp.minimum(x[:, 2], 20 - x[:, 2]))
    _bytes(omitted, explicit)
    _bytes(omitted, far)
    plain = jax.jit(partial(SJ.lane_step, KEY, lt, jnp.arange(N), B=B, msc=msc, mott=False))(st)
    disabled = jax.jit(partial(SJ.lane_step, KEY, lt, jnp.arange(N), B=B, msc=msc, mott=False,
                               boundary=None, safety=None))(st)
    _bytes(plain, disabled)
    _state_close(plain, omitted[0])
    assert np.asarray(omitted[1].valid).any()


@pytest.mark.parametrize("step", [SJ.lane_step, SJ.lane_step_photons])
@pytest.mark.parametrize("hook", ["boundary", "safety"])
def test_hooks_must_be_paired_before_any_state_or_table_tracing(step, hook):
    with pytest.raises(ValueError, match="boundary and safety must be supplied together"):
        step(None, None, None, None, **{hook: lambda *args: None})


@pytest.mark.parametrize("field", [False, True])
def test_unscattered_exit_replaces_legacy_faces(mean_loss, monkeypatch, field):
    def forbidden(*args, **kwargs):
        raise AssertionError("opt-in transport must not call the legacy helical face query")

    monkeypatch.setattr(SJ, "_to_plane", forbidden)
    st, B = _state(leg=1), _field(field)
    p = jnp.sqrt(st["T"] * (st["T"] + 2 * SJ.ME))
    distance = jnp.full(N, 0.25)
    xn, dn = _advance(st["x"], st["d"], p, -1.0, B, distance)
    boundary = _wall_boundary(xn[:, 0], jnp.ones(N))
    result, photons = _step(_tables(), st, B=B, msc=False, boundary=boundary,
                            safety=lambda x: jnp.maximum(xn[:, 0] - x[:, 0], 0.0))
    np.testing.assert_allclose(result["x"], xn, atol=1e-12)
    np.testing.assert_allclose(result["d"], dn, atol=1e-12)
    np.testing.assert_allclose(result["path"], st["path"] + distance, atol=1e-12)
    np.testing.assert_allclose(result["cont"], 0.5 * (st["pend"] + distance), atol=1e-12)
    assert np.asarray(result["exited"]).all()
    _no_records(photons)


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("msc", [False, True])
def test_complete_boundary_still_handles_axial_exit(mean_loss, field, msc):
    st, B = _state(leg=2), _field(field)
    st["x"] = st["x"].at[:, 2].set(20 - 1e-6)
    boundary = _wall_boundary(jnp.full(N, 1e6), jnp.ones(N))
    result, photons = _step(_tables(), st, B=B, msc=msc, boundary=boundary,
                            safety=lambda x: jnp.minimum(x[:, 2], 20 - x[:, 2]))
    np.testing.assert_allclose(result["x"][:, 2], 20.0, atol=1e-13)
    np.testing.assert_allclose(result["path"] - st["path"], 1e-6, atol=1e-13)
    assert np.asarray(result["exited"]).all()
    _no_records(photons)


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("msc", [False, True])
def test_overdue_pending_loss_cannot_propose_negative_path(mean_loss, field, msc):
    st, B, calls = _state(), _field(field), []
    st["pend"] = jnp.full(N, 1000.0)

    def boundary(x, d, p, q, B, smax):
        jax.debug.callback(lambda s: calls.append(np.asarray(s)), smax, ordered=True)
        return jnp.full_like(smax, jnp.inf)

    result, photons = _step(_tables(), st, B=B, msc=msc, boundary=boundary,
                            safety=lambda x: jnp.zeros(x.shape[0], x.dtype))
    jax.block_until_ready(result)
    assert calls and all((s == 0).all() for s in calls)
    np.testing.assert_array_equal(result["x"], st["x"])
    np.testing.assert_array_equal(result["path"], st["path"])
    assert np.asarray(result["stopped"]).all()
    assert not np.asarray(result["exited"]).any()
    _no_records(photons)


def _state_close(a, b):
    assert a.keys() == b.keys()
    for name in a:
        av, bv = np.asarray(a[name]), np.asarray(b[name])
        assert av.shape == bv.shape and av.dtype == bv.dtype, name
        if np.issubdtype(av.dtype, np.inexact):
            tol = 8 * np.finfo(av.dtype).eps
            np.testing.assert_allclose(av, bv, rtol=tol, atol=tol, err_msg=name)
        else:
            np.testing.assert_array_equal(av, bv, err_msg=name)


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("leg", [1, 2])
def test_state_only_finite_hooks_match_recorded_msc_exit(mean_loss, field, positron, leg):
    st, B, lt = _state(leg=leg, positron=positron), _field(field), _tables()
    boundary, safety, distance, _, _, _ = _exit_case(st, B, leg, positron)
    kw = dict(B=B, positron=positron, boundary=boundary, safety=safety, mott=False)
    plain = jax.jit(partial(SJ.lane_step, KEY, lt, jnp.arange(N), **kw))(st)
    recorded, photons = _step(lt, st, B=B, positron=positron, boundary=boundary, safety=safety)
    _state_close(plain, recorded)
    np.testing.assert_allclose(plain["path"], st["path"] + distance, atol=2e-12)
    assert np.asarray(plain["exited"]).all()
    _no_records(photons)


@pytest.mark.parametrize("hooks", [False, True])
def test_state_only_does_not_construct_photon_records(monkeypatch, mean_loss, hooks):
    def forbidden(*args, **kwargs):
        raise AssertionError("state-only steps must not construct a photon record")

    st, lt = _state(), _tables(sig_b=3.0)
    kw = {}
    if hooks:
        boundary, safety, *_ = _exit_case(st, None, 2)
        kw = dict(boundary=boundary, safety=safety)
    monkeypatch.setattr(SJ, "BremsPhotonRecord", forbidden)
    result = jax.jit(partial(SJ.lane_step, KEY, lt, jnp.arange(N), mott=False, **kw))(st)
    assert isinstance(result, dict)
    assert result["it"] == st["it"] + 1
