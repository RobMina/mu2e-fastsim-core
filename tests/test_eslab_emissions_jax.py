"""Full EM photon births inherit the NumPy annihilation convention, not new Geant4 validation."""
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from mu2efastsim import eannihil, eslab_jax as SJ  # noqa: E402
from mu2efastsim import materials as M  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    layers = [(M.BY_NAME["G4_W"], 20.0, 0.0814923, 0.00770222, 0.5),
              (M.BY_NAME["G4_Al"], 20.0, 0.03411, 0.00099, 0.5)]
    return {(positron, msc): SJ.lane_tables(layers, positron=positron, msc=msc)
            for positron in (False, True) for msc in (False, True)}


def _bytes_equal(a, b):
    assert a.keys() == b.keys()
    for name in a:
        av, bv = np.asarray(a[name]), np.asarray(b[name])
        assert av.dtype == bv.dtype and av.shape == bv.shape, name
        assert av.tobytes() == bv.tobytes(), name


def _record(p, n, dtype):
    assert isinstance(p, SJ.EMPhotonRecord)
    assert p.energy.shape == p.path.shape == p.valid.shape == p.mechanism.shape == (n, 3)
    assert p.position.shape == p.direction.shape == (n, 3, 3)
    assert p.valid.dtype == np.bool_ and p.mechanism.dtype == np.int32
    for values in (p.energy, p.position, p.direction, p.path):
        assert values.dtype == dtype
        assert np.isfinite(values).all()
        assert (values[~p.valid] == 0).all()
    assert (p.mechanism[~p.valid] == 0).all()
    assert np.isin(p.mechanism[p.valid], [1, 2, 3]).all()
    tol = 4e-5 if dtype == np.float32 else 2e-11
    np.testing.assert_allclose(np.linalg.norm(p.direction[p.valid], axis=-1), 1.0, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("positron", [False, True])
@pytest.mark.parametrize("msc,field", [(False, False), (True, False), (False, True), (True, True)])
def test_primary_bytes_and_brems_interface(tables, dtype, positron, msc, field):
    n, steps = 24, 5
    lt = tables[positron, msc]
    mi = jnp.arange(n) % 2
    T = np.full(n, 105.0, dtype)
    T[:3] = 0.0
    z = np.full(n, 10.0, dtype)
    z[3:6] = 20.0
    d = np.tile(np.array([0.0, 0.0, 1.0], dtype), (n, 1))
    st = SJ.lane_state(T, d, z, 20.0, positron=positron)
    st["path"] = jnp.full(n, 2.5, dtype)
    B = jnp.tile(jnp.array([0.7, 0.2, 0.0], dtype), (n, 1)) if field else None
    kw = dict(positron=positron, msc=msc, B=B)

    def run(fn, key, state):
        def step(s, _):
            sn, p = fn(key, lt, mi, s, **kw)
            return sn, (sn, p)
        return jax.lax.scan(step, state, None, length=steps)

    key = jax.random.PRNGKey(19)
    plain = jax.jit(lambda s: run(lambda *a, **k: (SJ.lane_step(*a, **k), ()), key, s))(st)
    old = jax.jit(lambda s: run(SJ.lane_step_photons, key, s))(st)
    new = jax.jit(lambda s: run(SJ.lane_step_emissions, key, s))(st)
    _bytes_equal(plain[0], new[0])
    _bytes_equal(plain[1][0], new[1][0])
    _bytes_equal(old[0], new[0])
    _bytes_equal(old[1][0], new[1][0])
    p = jax.tree_util.tree_map(np.asarray, new[1][1])
    b = jax.tree_util.tree_map(np.asarray, old[1][1])
    tol = 5e-6 if dtype == np.float32 else 2e-13
    for name in b._fields:
        actual, expected = getattr(p, name)[:, :, 0], getattr(b, name)
        if name == "valid":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    assert not p.valid[:, :6].any()
    assert p.valid[:, :, 0].any()
    for i in range(steps):
        _record(SJ.EMPhotonRecord(*(v[i] for v in p)), n, dtype)
    h = jax.tree_util.tree_map(np.asarray, new[1][0])
    previous_nan = np.concatenate([np.asarray(st["nan"])[None], h["nan"][:-1]])
    previous_ean = np.concatenate([np.asarray(st["ean"])[None], h["ean"][:-1]])
    np.testing.assert_array_equal(h["nan"] - previous_nan, p.valid[:, :, 1:].sum(axis=-1))
    tol = 5e-5 if dtype == np.float32 else 2e-11
    np.testing.assert_allclose(h["ean"] - previous_ean, p.energy[:, :, 1:].sum(axis=-1), atol=tol, rtol=tol)


def _rates(lt, **rates):
    return lt._replace(tab=lt.tab._replace(**{name: np.full_like(getattr(lt.tab, name), value)
                                             for name, value in rates.items()}))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_flight_energy_momentum_and_no_repeat(tables, dtype):
    lt = _rates(tables[True, False], sig_b=0, sig_d=0, sig_a=1e9, dedx=0)
    n = 32
    T = np.geomspace(0.02, 105.0, n).astype(dtype)
    rng = np.random.default_rng(6)
    d = rng.normal(size=(n, 3)).astype(dtype)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:2] = [[0, 0, 1], [0, 0, -1]]
    st = SJ.lane_state(T, d, np.full(n, 10, dtype), 20.0, positron=True)
    st["path"] = jnp.full(n, 4.0, dtype)
    mi, key = jnp.arange(n) % 2, jax.random.PRNGKey(7)
    step = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, positron=True, msc=False))
    sn, photon = step(st)
    _bytes_equal(jax.jit(partial(SJ.lane_step, key, lt, mi, positron=True, msc=False))(st), sn)
    p = jax.tree_util.tree_map(np.asarray, photon)
    _record(p, n, dtype)
    assert np.asarray(sn["annih"]).all() and (np.asarray(sn["T"]) == 0).all()
    assert not p.valid[:, 0].any() and p.valid[:, 1:].all()
    assert (p.mechanism[:, 1:] == 2).all()
    tol = 5e-5 if dtype == np.float32 else 3e-11
    np.testing.assert_allclose(p.energy.sum(axis=1), T + 2 * SJ.ME, rtol=tol, atol=tol)
    momentum = (p.energy[:, :, None] * p.direction).sum(axis=1)
    np.testing.assert_allclose(momentum, np.sqrt(T * (T + 2 * SJ.ME))[:, None] * d, rtol=tol, atol=tol)
    np.testing.assert_array_equal(p.position[:, 1], sn["x"])
    np.testing.assert_array_equal(p.position[:, 2], sn["x"])
    np.testing.assert_array_equal(p.path[:, 1], sn["path"])
    sn2, p2 = step(sn)
    assert not np.asarray(p2.valid).any()
    for values in p2:
        assert (np.asarray(values) == 0).all()
    for name in st.keys() - {"it"}:
        np.testing.assert_array_equal(sn2[name], sn[name])


@pytest.mark.parametrize("stop_kind", ["out", "low"])
def test_new_rest_births_use_step_endpoint(tables, monkeypatch, stop_kind):
    n = 8
    lt = _rates(tables[True, False], sig_b=0, sig_d=0 if stop_kind == "out" else 1e9, sig_a=0, dedx=1)
    if stop_kind == "out":
        monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", lambda key, mc, T, *a, **kw: 2 * T)
    else:
        monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", lambda key, mc, T, *a, **kw: jnp.zeros_like(T))
        monkeypatch.setattr(SJ, "_sample_delta", lambda key, T, *a, **kw: T - SJ.eslab.T_STOP / 2)
    st = SJ.lane_state(np.ones(n), np.tile([0.0, 0.0, 1.0], (n, 1)), np.full(n, 10.0), 20.0, positron=True)
    mi, key = jnp.zeros(n, jnp.int32), jax.random.PRNGKey(31)
    kw = dict(positron=True, msc=False)
    sn, photons = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, **kw))(st)
    _bytes_equal(jax.jit(partial(SJ.lane_step, key, lt, mi, **kw))(st), sn)
    p = jax.tree_util.tree_map(np.asarray, photons)
    _record(p, n, np.float64)
    assert p.valid[:, 1:].all() and not p.valid[:, 0].any()
    assert (p.mechanism[:, 1:] == 3).all()
    np.testing.assert_array_equal(p.energy[:, 1:], np.full((n, 2), SJ.ME))
    np.testing.assert_array_equal(p.direction[:, 1], -p.direction[:, 2])
    np.testing.assert_array_equal(p.position[:, 1], sn["x"])
    np.testing.assert_array_equal(p.path[:, 1], sn["path"])
    assert (np.asarray(sn["nan"]) == 2).all() and not np.asarray(sn["annih"]).any()
    sn2, photons2 = jax.jit(partial(SJ.lane_step_emissions, key, lt, mi, **kw))(sn)
    assert not np.asarray(photons2.valid).any()
    np.testing.assert_array_equal(sn2["nan"], sn["nan"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_initial_birth_helper_selection(tables, dtype):
    n = 8
    st = SJ.lane_state(np.zeros(n, dtype), np.zeros((n, 3), dtype), np.full(n, 3, dtype), 20.0, positron=True)
    st["path"] = jnp.arange(n, dtype=dtype)
    st["nan"] = st["nan"].at[1].set(0).at[2].set(4)
    st["ean"] = st["ean"].at[3].set(0)
    st["annih"] = st["annih"].at[4].set(True)
    st["stopped"] = st["stopped"].at[5].set(False)
    original = {k: np.asarray(v).copy() for k, v in st.items()}
    key = jax.random.PRNGKey(9)
    p = jax.tree_util.tree_map(np.asarray, jax.jit(SJ.initial_annihilation_photons)(key, st))
    _bytes_equal(original, st)
    _record(p, n, dtype)
    expected = np.zeros((n, 3), bool)
    expected[[0, 6, 7], 1:] = True
    np.testing.assert_array_equal(p.valid, expected)
    assert (p.mechanism[p.valid] == 3).all()
    np.testing.assert_array_equal(p.energy[p.valid], np.full(6, SJ.ME, dtype))
    np.testing.assert_array_equal(p.direction[:, 1], -p.direction[:, 2])
    np.testing.assert_array_equal(p.path[[0, 6, 7], 1], np.asarray(st["path"])[[0, 6, 7]])
    for positron, annihilation in ((False, True), (True, False)):
        off = SJ.lane_state(np.zeros(n, dtype), np.zeros((n, 3), dtype), np.zeros(n, dtype), 20.0,
                            positron=positron, annihilation=annihilation)
        assert not np.asarray(SJ.initial_annihilation_photons(key, off).valid).any()


@pytest.mark.parametrize("n", [0, 4])
def test_empty_or_finished_zero_records(tables, n):
    lt = tables[True, False]
    st = SJ.lane_state(np.zeros(n, np.float32), np.zeros((n, 3), np.float32), np.zeros(n, np.float32), 20.0,
                       positron=True)
    key = jax.random.PRNGKey(1)
    sn, p = jax.jit(partial(SJ.lane_step_emissions, key, lt, jnp.zeros(n, jnp.int32),
                            positron=True, msc=False))(st)
    p = jax.tree_util.tree_map(np.asarray, p)
    _record(p, n, np.float32)
    for values in p:
        assert (values == 0).all()
    assert not np.asarray(sn["nph"]).any()
    initial = jax.jit(SJ.initial_annihilation_photons)(key, st)
    _record(jax.tree_util.tree_map(np.asarray, initial), n, np.float32)


def test_annihilation_disabled(tables):
    lt = _rates(tables[True, False], sig_a=1e9)
    n = 8
    st = SJ.lane_state(np.ones(n), np.tile([0.0, 0.0, 1.0], (n, 1)), np.full(n, 10.0), 20.0,
                       positron=True, annihilation=False)
    sn, p = jax.jit(partial(SJ.lane_step_emissions, jax.random.PRNGKey(8), lt, jnp.zeros(n, jnp.int32),
                            positron=True, msc=False, annihilation=False))(st)
    assert not np.asarray(p.valid[:, 1:]).any()
    assert not np.asarray(sn["nan"]).any() and not np.asarray(sn["ean"]).any()


def test_pair_matches_numpy_inherited_convention():
    n, key = 12, jax.random.PRNGKey(22)
    ks = jax.random.split(key, 4)
    T = np.geomspace(0.1, 105.0, n)
    d = np.random.default_rng(4).normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    class Uniforms:
        def __init__(self, keys):
            self.keys = iter(keys)

        def random(self, size):
            return np.asarray(jax.random.uniform(next(self.keys), (size,), jnp.float64))

    e, u = SJ._annihilation_pair(ks, jnp.asarray(T), jnp.asarray(d), jnp.ones(n, bool))
    expected_e, expected_u = eannihil.in_flight(Uniforms(ks[:2]), T, d)
    np.testing.assert_allclose(e, expected_e, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(u, expected_u, rtol=3e-13, atol=3e-13)
    e, u = SJ._annihilation_pair(ks, jnp.asarray(T), jnp.asarray(d), jnp.zeros(n, bool))
    expected_e, expected_u = eannihil.at_rest(Uniforms(ks[2:]), n)
    np.testing.assert_array_equal(e, expected_e)
    np.testing.assert_allclose(u, expected_u, rtol=1e-14, atol=1e-14)
