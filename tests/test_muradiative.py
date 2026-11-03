"""Unit tests: muon radiative losses (bremsstrahlung, pair production, photonuclear) and the slab_loss opt-in."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import materials as M                                  # noqa: E402
from mu2efastsim import muradiative as R                                # noqa: E402
from mu2efastsim import transport as T                                  # noqa: E402


def _b(mat, tk, processes=("brems", "pair", "nuclear")):
    """b = (1/E) dE/dx [cm2/g] by direct integration of d Sigma / d v."""
    E = tk + R.MMU
    lv = np.linspace(np.log(1e-7), np.log(1 - 1e-9), 4000)
    v = np.exp(lv)
    return np.trapezoid(v * v * R.dsigma_dv_per_mm(mat, E, v, processes=processes), lv) * 10.0 / mat.rho


def test_b_against_published_ranges():
    # radiative b at 100 GeV, 1e-6 cm2/g: published muon tables give Cu ~6, Al ~3.5, Pb ~14
    assert 5.0e-6 < _b(M.COPPER, 1e5) < 7.5e-6
    assert 2.8e-6 < _b(M.ALUMINIUM, 1e5) < 4.2e-6
    assert 11e-6 < _b(M.LEAD, 1e5) < 17e-6
    # pair production dominates bremsstrahlung at 100 GeV in copper; photonuclear is the smallest
    bb, bp, bn = (_b(M.COPPER, 1e5, (k,)) for k in ("brems", "pair", "nuclear"))
    assert bp > bb > bn > 0


def test_cross_sections_physical():
    E = 1e5 + R.MMU
    eps = np.geomspace(1.0, 0.99 * E, 50)
    for Z, A in ((13, 26.98), (29, 63.546)):
        b = R.brems_dxs(E, eps, Z, A)
        assert (b >= 0).all() and (b[:10] * eps[:10]).std() / (b[:10] * eps[:10]).mean() < 0.05   # ~1/eps at small v
        assert (R.pair_dxs(E, eps, Z) >= 0).all()
        assert (R.nuclear_dxs(E, eps, A)[eps < R.NUC_EPS_MIN] == 0).all()
    assert R.brems_dxs(E, E, 29, 63.5) == 0 and R.pair_dxs(E, 1.0, 29) == 0


def test_sampled_mean_matches_table_and_zero_below_threshold():
    rng = np.random.default_rng(5)
    mat = M.STAINLESS_STEEL
    t = R.table(mat)
    p = np.full(200_000, 50_000.0)
    L = 100.0
    s = R.sample_loss(rng, mat, p, np.full(p.size, L))
    E = np.sqrt(50_000.0 ** 2 + R.MMU ** 2)
    expect = np.interp(np.log(E), t.lnE, t.b) * E * L
    assert abs(s.mean() / expect - 1) < 0.05
    assert (s <= E - R.MMU).all()
    assert (R.sample_loss(rng, mat, np.full(100, 500.0), np.full(100, L)) == 0).all()


def test_slab_loss_default_unchanged_and_opt_in_adds():
    p = np.full(20_000, 20_000.0)
    L = np.full(p.size, 50.0)
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = T.slab_loss(r1, M.COPPER, p, L)
    b = T.slab_loss(r2, M.COPPER, p, L, radiative=False)
    assert np.array_equal(a, b) and r1.random() == r2.random()     # same values, same draws
    r3 = np.random.default_rng(7)
    c = T.slab_loss(r3, M.COPPER, p, L, radiative=True)
    d = c - a                                                        # same ionisation draws first
    assert (d >= -1e-9).all() and 0.7 < d.mean() / (_b(M.COPPER, 20_000.0 - R.MMU) * 20_000.0 * 50.0 * M.COPPER.rho / 10) < 1.3
