"""JAX twin of the Urban energy-loss sampler against the numpy eloss.py: exact dE/dx, statistical sample_loss.

Criteria per point (N = 4e6 samples each side): mean within 0.5 %; P(loss == 0) equal within 5 binomial sigma; quantiles
q05..q99 of the non-zero losses within 2 %; binned ratio (stats.ratio_band, 20 bins over the numpy q05-q95) max
|ratio - 1| minus 2 sigma <= 1 %. N = 4e6 because at 1e6 numpy-vs-numpy (two seeds) already reaches 1.1-1.3 % on the
band metric (tail bins have ~2.5 % stat error; max over 20 bins of |z| - 2 is often > 0). Script: prints the table.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
jax = pytest.importorskip("jax")
import jax.numpy as jnp                                                   # noqa: E402

from mu2efastsim import eloss as E                                        # noqa: E402
from mu2efastsim import eloss_jax as EJ                                   # noqa: E402
from mu2efastsim import materials as M                                    # noqa: E402
from mu2efastsim import stats                                             # noqa: E402

GAS = M.Material("StrawGas", 0.001696, 158.62097145526,
                 {"Ar": 0.784057035890315, "O": 0.15700963928136, "C": 0.0589333248283256})
N = 4_000_000
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _tk_step(mat, p, frac=0.05):
    """Step [mm] losing frac of the kinetic energy on average (the slab_loss slicing near range-out)."""
    tk = np.sqrt(p * p + E.MMU ** 2) - E.MMU
    return frac * tk / float(E.dedx_restricted(mat, p, 1e9))


# (label, material, p [MeV/c] (scalar or (lo, hi) log-uniform), length [mm] (scalar or (lo, hi) uniform), tcut [MeV])
POINTS = [
    ("gas 990eV 1mm 3GeV", GAS, 3000.0, 1.0, 990e-6),
    ("gas 990eV 5mm 3GeV", GAS, 3000.0, 5.0, 990e-6),
    ("gas 990eV 10mm 300MeV", GAS, 300.0, 10.0, 990e-6),
    ("gas 990eV 1-10mm 0.2-10GeV", GAS, (200.0, 1e4), (1.0, 10.0), 990e-6),
    ("PS 87.2keV 20mm 3GeV", M.POLYSTYRENE, 3000.0, 20.0, 0.0872),
    ("PS 87.2keV 20mm 300MeV", M.POLYSTYRENE, 300.0, 20.0, 0.0872),
    ("concrete 1e9 100mm 200MeV (gauss)", M.CONCRETE_MARS, 200.0, 100.0, 1e9),
    ("concrete 1e9 500mm 1GeV (urban, mean/cut 2.5)", M.CONCRETE_MARS, 1000.0, 500.0, 1e9),
    ("concrete 1e9 50mm 3GeV (urban)", M.CONCRETE_MARS, 3000.0, 50.0, 1e9),
    ("concrete 1e9 5%tk 100MeV (range-out)", M.CONCRETE_MARS, 100.0, _tk_step(M.CONCRETE_MARS, 100.0), 1e9),
    ("concrete 1e9 5%tk 40MeV (range-out)", M.CONCRETE_MARS, 40.0, _tk_step(M.CONCRETE_MARS, 40.0), 1e9),
    ("PS 87.2keV 5%tk 60MeV (range-out)", M.POLYSTYRENE, 60.0, _tk_step(M.POLYSTYRENE, 60.0), 0.0872),
    ("steel 1e9 2mm 50MeV (gauss)", M.MILD_STEEL, 50.0, 2.0, 1e9),
    # Gamma branch (Bohr regime, mean/sigma < 2) only exists at p ~ 0.9-3 MeV/c (last um of range): 0.1 um step
    ("steel 1e9 0.1um 1.85MeV (gamma)", M.MILD_STEEL, 1.85, 1e-4, 1e9),
]


def _inputs(p, L, n, seed=7):
    rng = np.random.default_rng(seed)
    pp = np.exp(rng.uniform(np.log(p[0]), np.log(p[1]), n)) if isinstance(p, tuple) else np.full(n, p)
    ll = rng.uniform(L[0], L[1], n) if isinstance(L, tuple) else np.full(n, L)
    return pp, ll


def branches(mat, p, L, tcut):
    """Fractions of steps in the Gauss / Gamma / Urban branches (as eloss.sample_loss decides)."""
    tk, b2, bg2, g, tmax = E.kinematics(p)
    cut = np.minimum(tcut, tmax)
    mean = E.dedx_restricted(mat, p, tcut) * L
    bohr = (mean >= E._BOHR_MIN * cut) & (tmax <= 2 * cut)
    siga = np.sqrt(np.clip((tmax / b2 - 0.5 * cut) * E.TWOPI_MC2_RCL2 * L * mat.electron_density_per_mm3, 0, None))
    sn = mean / np.where(siga > 0, siga, 1)
    return float((bohr & (sn >= 2)).mean()), float((bohr & (sn < 2)).mean()), float((~bohr).mean())


def compare(mat, p, L, tcut, n=N, seed=0):
    pp, ll = _inputs(p, L, n)
    ref = E.sample_loss(np.random.default_rng(seed), mat, pp, ll, tcut)
    fast = np.asarray(EJ.sample_loss(jax.random.key(seed), mat, jnp.asarray(pp), jnp.asarray(ll), tcut))
    z0r, z0f = (ref == 0).mean(), (fast == 0).mean()
    qr, qf = np.quantile(ref[ref > 0], QS), np.quantile(fast[fast > 0], QS)   # zero spike compared separately
    rb = stats.ratio_band(fast, ref)
    return {"mean_rel": fast.mean() / ref.mean() - 1, "q_rel": qf / qr - 1, "band": rb["max_abs_dev"],
            "p0": (z0r, z0f), "p0_sig": np.sqrt(2 * max(z0r, 1 / n) * (1 - z0r) / n),
            "band_m2s": rb["max_abs_dev_minus_2sigma"], "mean_ref": ref.mean(), "branches": branches(mat, pp, ll, tcut),
            "finite": bool(np.isfinite(fast).all() and (fast >= 0).all())}


def test_dedx_exact():
    p = np.geomspace(1.0, 1e5, 2001)
    for mat in (GAS, M.POLYSTYRENE, M.CONCRETE_MARS, M.ALUMINIUM, M.COPPER, M.TUNGSTEN, M.MILD_STEEL, M.AIR):
        for tc in (990e-6, 0.0872, 1e9):
            a = E.dedx_restricted(mat, p, tc)
            b = np.asarray(EJ.dedx_restricted(mat, p, tc))
            assert np.allclose(b, a, rtol=1e-12, atol=0), (mat.name, tc, np.max(np.abs(b / a - 1)))


def test_dedx_jits_with_array_constants():
    c = EJ.stack_consts([GAS, M.POLYSTYRENE], np.array([0, 1, 1, 0]))
    p = jnp.array([300.0, 300.0, 3000.0, 3000.0])
    d = np.asarray(jax.jit(lambda c, p: EJ._dedx(c, p, 0.0872, EJ.MMU))(c, p))
    ref = [E.dedx_restricted(m, x, 0.0872) for m, x in ((GAS, 300.0), (M.POLYSTYRENE, 300.0), (M.POLYSTYRENE, 3000.0),
                                                       (GAS, 3000.0))]
    assert np.allclose(d, ref, rtol=1e-12)


@pytest.mark.parametrize("pt", POINTS, ids=[x[0] for x in POINTS])
def test_sample_loss_matches_numpy(pt):
    _, mat, p, L, tc = pt
    r = compare(mat, p, L, tc)
    assert r["finite"]
    assert abs(r["mean_rel"]) < 0.005, r
    assert abs(r["p0"][1] - r["p0"][0]) <= 5 * r["p0_sig"], r
    assert np.all(np.abs(r["q_rel"]) < 0.02), r
    assert r["band_m2s"] is not None and r["band_m2s"] <= 0.01, r


def test_branch_coverage():
    """The points exercise every branch (Gauss, Gamma, Urban)."""
    fr = np.array([branches(m, *_inputs(p, L, 1000), tc) for _, m, p, L, tc in POINTS])
    assert (fr > 0.99).any(axis=0).all()


def test_bit_identical_and_fixed_shape():
    p = jnp.asarray(np.geomspace(30.0, 1e4, 4096))
    a = EJ.sample_loss(jax.random.key(3), GAS, p, 5.0, 990e-6)
    b = EJ.sample_loss(jax.random.key(3), GAS, p, 5.0, 990e-6)
    c = EJ.sample_loss(jax.random.key(4), GAS, p, 5.0, 990e-6)
    assert np.array_equal(np.asarray(a), np.asarray(b)) and not np.array_equal(np.asarray(a), np.asarray(c))
    assert a.shape == p.shape and a.dtype == jnp.float64
    # same compiled program for any values: no retrace when inputs change but shapes do not
    f = jax.jit(lambda k, p, L: EJ.sample_loss(k, M.POLYSTYRENE, p, L, 0.0872))
    f(jax.random.key(0), p, jnp.full(p.shape, 20.0))
    n0 = f._cache_size()
    f(jax.random.key(1), p * 0.01, jnp.full(p.shape, 1e3))
    assert f._cache_size() == n0


def test_poisson_means_below_truncation():
    """All Poisson means fed to the fixed-K loop are <= 8 (so K = 32 truncates with P < 1e-10), incl. w1 -> 1."""
    from scipy.stats import poisson
    assert poisson.sf(EJ.K_POISSON, 8.0) < 1e-10
    a3 = jnp.asarray(np.geomspace(8.0001, 1e12, 600))[:, None]
    w1 = jnp.asarray(1 + np.geomspace(1e-15, 1e8, 600))[None, :]
    alfa, alfa1, namean, p3 = (np.broadcast_to(np.asarray(x), (600, 600)) for x in EJ._ion_split(a3, w1))
    assert p3.max() <= 8.0 and (p3 >= 0).all() and (namean >= 0).all()
    # same split as G4/eloss.py where its form is well conditioned (w1 - 1 >= 1e-3)
    a, w = np.broadcast_arrays(np.asarray(a3), np.asarray(w1))
    al = w * (8 + a) / (w * 8 + a)
    nm = a * w * (al - 1) / ((w - 1) * al)
    good = np.broadcast_to(w - 1 >= 1e-3, nm.shape)
    assert np.allclose(namean[good], nm[good], rtol=1e-9) and np.allclose(alfa, al, rtol=1e-14)
    assert np.allclose(alfa1[good], (al * np.log(al) / (al - 1))[good], rtol=1e-9)
    assert np.isfinite(alfa1).all() and (alfa1 >= 1).all()


def test_gamma_sampler():
    from scipy.stats import kstest
    for a in (0.3, 1.0, 3.5):
        x = np.asarray(EJ._gamma_fixed(jax.random.key(5), jnp.full(200_000, a), EJ.GAMMA_ROUNDS))
        assert kstest(x, "gamma", args=(a,)).pvalue > 1e-3


def test_truncated_gauss():
    """Inverse-CDF truncated Gauss vs the numpy rejection sampler."""
    eav = np.full(400_000, 1.0)
    for s2 in (0.01, 0.5, 4.0, 100.0):
        a = E._sample_gauss_clipped(np.random.default_rng(1), eav, np.full_like(eav, s2))
        b = np.asarray(EJ._gauss_clipped(jax.random.uniform(jax.random.key(1), eav.shape, jnp.float64), eav, s2))
        assert b.min() >= 0 and b.max() <= 2.0
        assert abs(a.mean() - b.mean()) < 4 * a.std() / np.sqrt(eav.size) + 1e-12
        assert np.allclose(np.quantile(a, QS), np.quantile(b, QS), rtol=0.01, atol=2e-3)


if __name__ == "__main__":
    print(f"| point | mean ref [MeV] | branches G/Gm/U | mean rel | max abs q05..q99 rel | band | band-2sig |")
    print("|---|---|---|---|---|---|---|")
    for lab, mat, p, L, tc in POINTS:
        r = compare(mat, p, L, tc)
        br = "/".join(f"{x:.2f}" for x in r["branches"])
        print(f"| {lab} | {r['mean_ref']:.4g} | {br} | {100 * r['mean_rel']:+.3f} % | "
              f"{100 * np.abs(r['q_rel']).max():.3f} % | {100 * r['band']:.2f} % | {100 * r['band_m2s']:.2f} % |")


def test_supplied_mean():
    """mean=None is the default dE/dx * length; a supplied mean reproduces the default when equal, scales the sample
    mean otherwise, and keeps the stream bit-identical for the same key."""
    p = jnp.asarray(np.geomspace(15.0, 80.0, 200_000))
    L = jnp.full(p.shape, 0.02)
    k = jax.random.key(11)
    m0 = EJ.dedx_restricted(M.ALUMINIUM, p, 1.0) * L
    a = np.asarray(EJ.sample_loss(k, M.ALUMINIUM, p, L, 1.0))
    b = np.asarray(EJ.sample_loss(k, M.ALUMINIUM, p, L, 1.0, mean=m0))
    assert np.allclose(a, b, rtol=1e-12, atol=0)
    c = np.asarray(EJ.sample_loss(k, M.ALUMINIUM, p, L, 1.0, mean=1.1 * m0))
    assert abs(c.mean() / a.mean() - 1.1) < 0.005
    assert np.isfinite(c).all() and (c >= 0).all()
