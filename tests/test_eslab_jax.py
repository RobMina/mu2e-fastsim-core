"""JAX twin of the electron slab (mu2efastsim.eslab_jax) and the electron Urban sampler against the NumPy models.

Exact: the host tables equal the NumPy functions at the grid nodes; energy balance per track; muon sample_loss
outputs unchanged by the electron options (SHA-256 of a fixed grid, float64 and float32). Statistical (small samples
here; the 200k-event parity per case is a slow test and a script): loss, exit angle, fates and yields against eslab.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
jax = pytest.importorskip("jax")

from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import eioni as I                                       # noqa: E402
from mu2efastsim import eloss_jax as EJ                                  # noqa: E402
from mu2efastsim import emsc as E                                        # noqa: E402
from mu2efastsim import eslab as S                                       # noqa: E402
from mu2efastsim import eslab_jax as SJ                                  # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

AL, W = M.BY_NAME["G4_Al"], M.BY_NAME["G4_W"]
CUTS = {"G4_Al": (0.03411, 0.00099), "G4_W": (0.0814923, 0.00770222)}

# eloss_jax.sample_loss for muons over the grid of _muon_hashes, computed with the sampler before the electron
# options were added (JAX 0.11.2 CPU); float64 results depend on the CPU's math kernels, so one hash per platform seen.
# float32: the sampler now returns the input dtype. The earlier pinned value (e41dc54f...) hashed the float64 arrays
# that float32 inputs used to return; the values are unchanged up to that final rounding (the new float32 outputs
# equal the old float64 outputs rounded to float32 over this whole grid), so the hash below is of the same samples.
MUON_SHA256 = {"float64": {"5a43f3ec712cf4a397270126c9aad3c900e60b16de04546cdcf2d898d9ab6ff2",    # arm64
                           "31ac62093ec331e1c5058b8b1612b4812ea02ec683dca0654f0d11d0842c6e99"},   # x86-64
               "float32": {"4915c2e46046d85676b1f63354fe26cbba12d1c4884ef8063238a5701be97496"}}


def _muon_hashes():
    out = {}
    for dt in (np.float64, np.float32):
        h = hashlib.sha256()
        for name in ("G4_Al", "G4_W", "G4_POLYSTYRENE", "G4_MYLAR", "CONCRETE_MARS"):
            mat = M.BY_NAME[name]
            for i, p in enumerate((50.0, 105.0, 300.0, 1000.0, 5000.0)):
                for L in (0.01, 1.0, 20.0):
                    for tcut in (0.03, 1.0):
                        x = np.asarray(EJ.sample_loss(jax.random.PRNGKey(i), mat, np.full(64, p, dt),
                                                      np.asarray(L, dt), np.asarray(tcut, dt)))
                        h.update(x.tobytes())
        out[np.dtype(dt).name] = h.hexdigest()
    return out


def test_muon_sampler_is_bit_identical_to_before_the_electron_options():
    if jax.__version__ != "0.11.2" or jax.default_backend() != "cpu":
        pytest.skip("reference hashes recorded with JAX 0.11.2 on CPU")
    h = _muon_hashes()
    assert h["float32"] in MUON_SHA256["float32"]
    if h["float64"] not in MUON_SHA256["float64"]:
        pytest.skip(f"no float64 reference for this CPU: {h['float64']}")


def test_electron_options_are_inert_for_muons():
    # explicit muon tmax and gaussian=True reproduce the default call bit for bit
    for dt in (np.float64, np.float32):
        p = np.geomspace(30.0, 5000.0, 64).astype(dt)
        for mat in (W,):   # high Z; the options are material-independent (Al runs in the nightly sampler tests)
            c = EJ.consts(mat)
            mass = np.asarray(EJ.MMU, dt)
            tmax = EJ.kinematics(p, mass)[4]
            a = EJ._sample(jax.random.PRNGKey(7), c, p, np.asarray(1.0, dt), np.asarray(0.05, dt), mass)
            b = EJ._sample(jax.random.PRNGKey(7), c, p, np.asarray(1.0, dt), np.asarray(0.05, dt), mass, tmax=tmax,
                           gaussian=True)
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_electron_urban_sampler_matches_numpy():
    n = 200_000
    for mat, T, L in ((AL, 105.0, 0.1), (W, 2.0, 0.01), (AL, 0.5, 0.05)):
        tc, kc = CUTS[mat.name]
        mean = ((I.dedx_restricted(mat, T, tc) + B.dedx_restricted(mat, T, kc)) * L)[0]
        a = I.continuous_loss(np.random.default_rng(1), mat, np.full(n, T), L, tc, kc)
        b = np.asarray(EJ.sample_loss_electron(jax.random.PRNGKey(2), mat, np.full(n, T), L, tc, np.full(n, mean)))
        assert b.mean() == pytest.approx(a.mean(), rel=0.01)
        assert np.quantile(b, [0.05, 0.5, 0.95]) == pytest.approx(np.quantile(a, [0.05, 0.5, 0.95]), rel=0.02)


@pytest.mark.parametrize("positron", [False, True])
def test_tables_equal_the_numpy_models_at_the_nodes(positron):
    tc, kc = CUTS["G4_W"]
    tab = SJ.build_tables(W, tc, kc, positron, 0.05)
    T = SJ._grid()[1]
    sb, sd = S._sigmas(W, T, tc, kc, positron)
    assert tab.sig_b == pytest.approx(sb) and tab.sig_d == pytest.approx(sd)
    assert tab.lam1 == pytest.approx(E.lambda1(W, T, tc, positron))
    assert tab.uc == pytest.approx(E.u_cut(W, T, tc, positron, 0.05))
    assert tab.rng == pytest.approx(S.csda_range(W, T, tc, kc, positron))
    assert np.all(np.diff(tab.brem_x, axis=1) >= 0) and tab.brem_x.min() >= 0 and tab.brem_x.max() <= 1


@pytest.mark.pr_params("True-True")
@pytest.mark.parametrize("msc", [False, True])
@pytest.mark.parametrize("positron", [False, True])
def test_energy_balance_faces_and_dtype(positron, msc):
    n, T0, L = 400, 105.0, 2.0
    tc, kc = CUTS["G4_W"]
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    r = jax.tree_util.tree_map(np.asarray, SJ.electron_slab(jax.random.PRNGKey(3), W, np.full(n, T0), d, L, tc, kc,
                                                            positron, msc=msc))
    out = r.continuous + r.photon_energy + r.delta_energy + r.annihilation_energy - SJ.ME * r.n_annihilation
    assert T0 - r.T == pytest.approx(out, abs=1e-9)
    left = ~r.stopped
    z = r.position[:, 2]
    assert np.allclose(z[left & r.reflected], 0.0, atol=1e-9) and np.allclose(z[left & ~r.reflected], L, atol=1e-9)
    assert (r.T[r.stopped] == 0).all() and r.T.dtype == np.float64
    if positron:
        assert (r.n_annihilation[r.stopped] == 2).all()
    r32 = SJ.electron_slab(jax.random.PRNGKey(3), W, np.full(8, T0, np.float32), d[:8].astype(np.float32), L, tc, kc,
                           positron, msc=msc)
    assert r32.T.dtype == np.float32 and r32.position.dtype == np.float32


@pytest.mark.slow
def test_small_sample_parity_with_numpy():
    # 1 mm Al, 105 MeV e- with msc: medians of loss and exit angle, photon and delta yields within statistics
    n = 20_000
    tc, kc = CUTS["G4_Al"]
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    a = S.electron_slab(np.random.default_rng(4), AL, np.full(n, 105.0), d, 1.0, tc, kc, msc=True)
    b = jax.tree_util.tree_map(np.asarray, SJ.electron_slab(jax.random.PRNGKey(4), AL, np.full(n, 105.0), d, 1.0, tc,
                                                            kc, msc=True))
    th = lambda dd: np.arctan2(np.hypot(dd[:, 0], dd[:, 1]), dd[:, 2])
    assert np.median(105.0 - b.T) == pytest.approx(np.median(105.0 - a.T), rel=0.02)
    assert np.median(th(b.direction)) == pytest.approx(np.median(th(a.direction)), rel=0.03)
    assert b.n_deltas.mean() == pytest.approx(a.deltas.track.size / n, abs=5 * np.sqrt(0.6 / n))
    assert b.n_photons.mean() == pytest.approx(a.photons.track.size / n, abs=5 * np.sqrt(0.15 / n))


@pytest.mark.weekly   # ~29 min on a CI runner (case 2 alone ~22 min): the weekly run, not the nightly
@pytest.mark.parametrize("case", [("G4_Al", 105.0, 0.1), ("G4_Al", 20.0, 1.0), ("G4_W", 105.0, 3.5)])
def test_parity_200k(case):
    name, T0, L = case
    tc, kc = CUTS[name]
    r = SJ.compare_numpy(M.BY_NAME[name], T0, L, tc, kc, n=200_000)
    for k in ("loss", "theta", "r"):
        assert r[k]["chi2_ndf"] < 2.0, (k, r[k])


def test_helix_and_plane_crossing_match_ehelix():
    from mu2efastsim import ehelix
    jnp = jax.numpy
    rng = np.random.default_rng(6)
    n = 300
    x = rng.normal(0, 5, (n, 3))
    u = rng.normal(0, 1, (n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = rng.uniform(0.5, 105.0, n)
    B = rng.normal(0, 1, (n, 3))
    s = rng.uniform(0, 30, n)
    xn, un = ehelix.helix(x, u, p, -1.0, B, s)
    xj, uj = SJ._helix(jnp.asarray(x), jnp.asarray(u), jnp.asarray(p), -1.0, jnp.asarray(B), jnp.asarray(s))
    assert np.allclose(xj, xn, atol=1e-9) and np.allclose(uj, un, atol=1e-12)
    zp = x[:, 2] + rng.uniform(-3, 3, n)
    sn = ehelix.to_plane(x, u, p, -1.0, B, zp, 6.0)
    sj = np.asarray(SJ._to_plane(jnp.asarray(x), jnp.asarray(u), jnp.asarray(p), -1.0, jnp.asarray(B),
                                 jnp.asarray(zp), jnp.asarray(np.full(n, 6.0))))
    fin = np.isfinite(sn)
    assert (np.isfinite(sj) == fin).all() and np.allclose(sj[fin], sn[fin], atol=1e-9)


@pytest.mark.slow
def test_start_depth_field_and_compaction_parity_with_numpy():
    # 0.5 mm Al from mid-depth in a 1 T transverse field, 105 MeV e- both ways: fates and loss medians as in NumPy;
    # the compacted run keeps the energy balance and matches NumPy within statistics (uncompacted runs:
    # test_small_sample_parity_with_numpy, test_boundary_starts_exit_outward_and_transport_inward)
    n, L, T0 = 2000, 0.5, 105.0
    tc, kc = CUTS["G4_Al"]
    rng = np.random.default_rng(7)
    ct = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    d = np.stack([np.sqrt(1 - ct ** 2) * np.cos(ph), np.sqrt(1 - ct ** 2) * np.sin(ph), ct], axis=1)
    Bf = np.array([1.0, 0.0, 0.0])
    a = S.electron_slab(np.random.default_rng(8), AL, np.full(n, T0), d, L, tc, kc, msc=True, z0=0.25, field=Bf)
    kw = dict(msc=True, z0=0.25, field=Bf)
    r = SJ.electron_slab(jax.random.PRNGKey(9), AL, np.full(n, T0), d, L, tc, kc, chunk=16, min_bucket=64, **kw)
    assert r.reflected.mean() == pytest.approx(a.reflected.mean(), abs=5 * np.sqrt(0.25 / n))
    assert np.median(T0 - r.T[~r.stopped]) == pytest.approx(np.median(T0 - a.T[~a.stopped]), rel=0.05)
    assert np.median(r.path) == pytest.approx(np.median(a.path), rel=0.03)
    out = r.continuous + r.photon_energy + r.delta_energy
    assert T0 - r.T == pytest.approx(out, abs=1e-8)
    z = r.position[:, 2]
    assert np.allclose(z[~r.stopped & r.reflected], 0.0, atol=1e-8)
    assert np.allclose(z[~r.stopped & ~r.reflected], L, atol=1e-8)


@pytest.mark.parametrize("chunk", [0, -1, 0.5, 1.5, np.float64(2.0), False, True, np.bool_(True)])
def test_compaction_rejects_nonpositive_or_noninteger_chunks(chunk, monkeypatch):
    def unexpected_compaction(*args):
        pytest.fail("invalid chunk reached compaction")

    monkeypatch.setattr(SJ, "_compacted", unexpected_compaction)
    with pytest.raises(ValueError, match="chunk must be a positive integer"):
        SJ.electron_slab(jax.random.PRNGKey(42), AL, np.ones(1), np.array([[0.0, 0.0, 1.0]]), 1.0,
                         *CUTS["G4_Al"], chunk=chunk)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_plane_crossings_over_many_revolutions(dtype):
    from mu2efastsim import ehelix
    jnp = jax.numpy
    R = 1.0 / ehelix.K
    x = jnp.zeros((4, 3), dtype)
    d = jnp.asarray(np.tile([0.0, 0.0, 1.0], (4, 1)), dtype)
    p = jnp.ones(4, dtype)
    field = jnp.asarray(np.tile([1.0, 0.0, 0.0], (4, 1)), dtype)
    zp = jnp.asarray([0.5 * R, 2 * R, -0.5 * R, 0.5 * R], dtype)
    smax = jnp.asarray([32 * np.pi * R] * 3 + [0.1 * R], dtype)
    s = np.asarray(jax.jit(SJ._to_plane)(x, d, p, -1.0, field, zp, smax))
    expected = R * np.array([np.arcsin(0.5), np.pi + np.arcsin(0.5)])
    assert s[[0, 2]] == pytest.approx(expected, rel=2e-6)
    assert np.isinf(s[[1, 3]]).all()


@pytest.mark.pr_params("float64")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_slab_leaves_at_first_face_before_curling_back(dtype):
    from mu2efastsim import ehelix
    air, T0, cut = M.BY_NAME["G4_AIR"], 1.0, 1000.0
    p = np.sqrt(T0 * (T0 + 2 * B.ME))
    dedx = (I.dedx_restricted(air, T0, cut) + B.dedx_restricted(air, T0, cut))[0]
    plan = max(S.STEP_FRACTION * T0 / dedx, S.FINAL_RANGE)
    R = plan / (32 * np.pi)
    L = 0.5 * R
    r = SJ.electron_slab(jax.random.PRNGKey(42), air, np.array([T0], dtype),
                         np.array([[0.0, 0.0, 1.0]], dtype), L, cut, cut, field=[p / (ehelix.K * R), 0.0, 0.0])
    assert not r.stopped[0] and not r.reflected[0]
    assert float(r.path[0]) == pytest.approx(R * np.arcsin(0.5), rel=2e-6)
    assert float(r.position[0, 2]) == pytest.approx(L, rel=2e-6)
    assert T0 - np.asarray(r.T) == pytest.approx(np.asarray(r.continuous), abs=1e-6 if dtype == np.float32 else 1e-12)
    assert not np.asarray(r.n_photons + r.n_deltas).any()


@pytest.mark.pr_params("True-chunk1-float64")   # the per-PR check of lane compaction (chunk)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("msc,chunk", [(False, None), (True, np.int64(1))])
def test_boundary_starts_exit_outward_and_transport_inward(dtype, msc, chunk):
    T = np.ones(4, dtype)
    d = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype)
    z0 = np.array([0.0, 1.0, 0.0, 1.0])
    for field in (None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]):
        r = jax.tree_util.tree_map(np.asarray, SJ.electron_slab(
            jax.random.PRNGKey(42), M.BY_NAME["G4_AIR"], T, d, 1.0, 1000.0, 1000.0,
            msc=msc, z0=z0, field=field, chunk=chunk, min_bucket=1))
        assert np.array_equal(r.path[:2], [0.0, 0.0])
        assert np.array_equal(r.T[:2], T[:2]) and np.array_equal(r.continuous[:2], [0.0, 0.0])
        assert np.array_equal(r.position[:2, 2], z0[:2]) and np.array_equal(r.direction[:2], d[:2])
        assert np.array_equal(r.reflected[:2], [True, False]) and not r.stopped[:2].any()
        assert not (r.n_photons[:2] + r.n_deltas[:2] + r.n_annihilation[:2]).any()
        assert (r.path[2:] > 0).all()
        assert ((r.position[:, 2] >= -1e-6) & (r.position[:, 2] <= 1.0 + 1e-6)).all()


@pytest.mark.pr_params("False-field2-float64")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("msc,field", [(True, None), (True, [0.4, 0.0, 1.0]), (False, [0.0, 1.0, 0.0])])
def test_lane_step_with_mixed_layer_types_equals_one_slab_call_per_type(dtype, msc, field):
    # Al foil, polyethylene and W lanes interleaved in one loop of lane_step: each lane ends as the lane in the same
    # slot of electron_slab run on all lanes with that lane's material (same key): discrete outcomes identical,
    # continuous ones to rounding (XLA fuses the two graphs differently)
    jnp = jax.numpy
    pe = M.BY_NAME["G4_POLYETHYLENE"]
    layers = [(AL, 0.1056, *CUTS["G4_Al"]), (pe, 0.5, 0.015, 0.00099), (W, 0.35, *CUTS["G4_W"])]
    n = 96
    rng = np.random.default_rng(5)
    ct = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    d = np.stack([np.sqrt(1 - ct ** 2) * np.cos(ph), np.sqrt(1 - ct ** 2) * np.sin(ph), ct], axis=1).astype(dtype)
    T = rng.uniform(20.0, 105.0, n).astype(dtype)
    mi = np.arange(n) % 3
    z0 = np.array([0.5 * layers[m][1] for m in mi])
    lt = SJ.lane_tables(layers, msc=msc)
    Bf = None if field is None else np.broadcast_to(np.asarray(field, float), (n, 3)).astype(dtype)
    key = jax.random.PRNGKey(11)

    @jax.jit
    def loop(st):
        def cond(s):
            return jnp.any(~s["stopped"] & ~s["exited"])
        return jax.lax.while_loop(cond, lambda s: SJ.lane_step(key, lt, jnp.asarray(mi), s, Bf, msc=msc), st)
    st = jax.tree_util.tree_map(np.asarray, loop(SJ.lane_state(T, d, z0, lt.length[mi])))
    for m, (mat, L, tc, kc) in enumerate(layers):
        r = jax.tree_util.tree_map(np.asarray, SJ.electron_slab(key, mat, T, d, L, tc, kc, msc=msc,
                                                                z0=np.full(n, 0.5 * L), field=field))
        j = mi == m
        for a, b in ((st["stopped"], r.stopped), (st["exited"] & (st["d"][:, 2] < 0), r.reflected),
                     (st["nph"], r.n_photons), (st["nde"], r.n_deltas)):
            assert np.array_equal(a[j], b[j])
        tol = 1e-11 if dtype == np.float64 else 2e-5
        for a, b in ((st["T"], r.T), (st["d"], r.direction), (st["x"], r.position), (st["path"], r.path),
                     (st["eph"], r.photon_energy), (st["ede"], r.delta_energy)):
            assert np.allclose(a[j], b[j], rtol=tol, atol=tol)


@pytest.mark.slow   # unroll identity per PR: test_eloss_jax::test_unrolled_sampler_same_draws
def test_lane_step_unroll_gives_the_same_draws():
    # unroll (opt-in) unrolls the fixed-trip loops of the step: same keys and arithmetic, so the same outcomes
    jnp = jax.numpy
    layers = [(AL, 0.1056, *CUTS["G4_Al"]), (W, 0.35, *CUTS["G4_W"])]
    n = 64
    rng = np.random.default_rng(6)
    ct = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    d = np.stack([np.sqrt(1 - ct ** 2) * np.cos(ph), np.sqrt(1 - ct ** 2) * np.sin(ph), ct], axis=1)
    T = rng.uniform(5.0, 105.0, n)
    mi = jnp.asarray(np.arange(n) % 2)
    lt = SJ.lane_tables(layers)
    Bf = np.tile([0.3, 0.0, 1.0], (n, 1))
    key = jax.random.PRNGKey(12)
    outs = []
    for unroll in (False, True):
        @jax.jit
        def loop(st):
            return jax.lax.while_loop(lambda s: jnp.any(~s["stopped"] & ~s["exited"]),
                                      lambda s: SJ.lane_step(key, lt, mi, s, Bf, unroll=unroll), st)
        outs.append(jax.tree_util.tree_map(np.asarray, loop(SJ.lane_state(T, d, np.full(n, 0.05), lt.length[mi]))))
    a, b = outs
    for k in ("stopped", "exited", "nph", "nde", "it"):
        assert np.array_equal(a[k], b[k])
    for k in ("T", "d", "x", "path", "eph", "ede"):
        assert np.allclose(a[k], b[k], rtol=1e-11, atol=1e-11)
