"""Unit tests: multiple scattering to a plane and sliced slab loss (no data files)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import eloss as E                                      # noqa: E402
from mu2efastsim import materials as M                                  # noqa: E402
from mu2efastsim import transport as T                                  # noqa: E402


def test_msc_lands_on_plane_and_angle_width():
    rng = np.random.default_rng(3)
    n = 100_000
    o = np.tile([0.0, 10000.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, 3000.0)
    x0 = E.radiation_length_mm(M.AIR)
    end, dn, _ = T.msc_to_plane(rng, o, d, p, 5000.0, x0)
    assert np.allclose(end[:, 1], 5000.0)
    th0 = E.highland_theta0(3000.0, 5000.0 / x0)
    assert abs(np.std(np.arctan2(dn[:, 0], -dn[:, 1])) / th0 - 1) < 0.02


def test_slab_loss_mean_and_range_out():
    rng = np.random.default_rng(4)
    p = np.full(50_000, 5000.0)
    loss = T.slab_loss(rng, M.CONCRETE_MARS, p, np.full(p.size, 1000.0))  # length is per track
    assert 400.0 < loss.mean() < 600.0          # ~2 GeV/m in concrete for a few-GeV muon
    tk0 = np.sqrt(300.0 ** 2 + E.MMU ** 2) - E.MMU
    stopped = T.slab_loss(rng, M.CONCRETE_MARS, np.full(1000, 300.0), np.full(1000, 2000.0))
    assert np.allclose(stopped, tk0)


def test_world_loss_slab_and_air_bands():
    # one 1 m concrete slab between y = 2000 and 3000 mm, plane at y = 0, vertical 5 GeV/c muons
    cube = np.array([[1, 0, 0, 5000], [-1, 0, 0, 5000], [0, 1, 0, 3000], [0, -1, 0, -2000], [0, 0, 1, 5000],
                     [0, 0, -1, 5000]], float)
    world = [("CONCRETE_MARS", cube, np.array([-5000.0, 2000, -5000]), np.array([5000.0, 3000, 5000]))]
    rng = np.random.default_rng(7)
    n = 20_000
    o = np.tile([0.0, 10000.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, 5000.0)
    pp, xx0, alive, bxx0, bh = T.world_loss(rng, o, d, p, world, [10000.0, 3000.0, 2000.0, 0.0])
    assert alive.all()
    x0c = E.radiation_length_mm(M.CONCRETE_MARS)
    assert np.allclose(bxx0[:, 1], 1000.0 / x0c) and np.allclose(bh[:, 1], 2500.0)
    assert 400.0 < (p - pp).mean() < 650.0


def test_new_materials_minimum_ionisation_matches_pdg():
    # PDG minimum ionisation [MeV cm2/g]: Cu 1.403, Si 1.664; CuW1090 against the mass-weighted W/Cu value, loosely
    # (a compound I and density effect move its minimum by ~2 %)
    p = np.geomspace(150.0, 5000.0, 400)
    for m, ref, tol in ((M.COPPER, 1.403, 0.005), (M.SILICON, 1.664, 0.005), (M.CUW1090, 0.9 * 1.145 + 0.1 * 1.403, 0.03)):
        d = (E.dedx_restricted(m, p, 1e9) * 10 / m.rho).min()
        assert abs(d / ref - 1) < tol, (m.name, d)



def test_sliced_msc_matches_single_band_and_grows_for_soft_muons():
    rng = np.random.default_rng(11)
    n = 40_000
    o = np.tile([0.0, 3000.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    x0 = E.radiation_length_mm(M.CONCRETE_MARS)
    bxx0 = np.tile([1000.0 / x0, 1000.0 / x0], (n, 1))
    bh = np.tile([2000.0, 500.0], (n, 1))
    # constant |p|: the sliced form reduces to the single-|p| form
    p = np.full(n, 3000.0)
    _, d1, _ = T.msc_to_plane(rng, o, d, p, 0.0, x0, bxx0.sum(1), bxx0, bh)
    _, d2, _ = T.msc_to_plane(rng, o, d, p, 0.0, x0, bxx0.sum(1), bxx0, bh, band_p=np.tile([3000.0, 3000.0], (n, 1)))
    s1, s2 = np.std(d1[:, 0]), np.std(d2[:, 0])
    assert abs(s2 / s1 - 1) < 0.02
    # a muon at 800 then 400 MeV/c in two equal bands scatters by sqrt((1/800^2 + 1/400^2)/2) * pg relative to one
    # band at the geometric-mean |p| pg = sqrt(1000 * 300) (beta ~ 1 to 4 %; tolerance 3 %)
    pg = np.full(n, np.sqrt(1000.0 * 300.0))
    _, d3, _ = T.msc_to_plane(rng, o, d, pg, 0.0, x0, bxx0.sum(1), bxx0, bh)
    _, d4, _ = T.msc_to_plane(rng, o, d, pg, 0.0, x0, bxx0.sum(1), bxx0, bh, band_p=np.tile([800.0, 400.0], (n, 1)))
    expect = np.sqrt((1 / 800.0 ** 2 + 1 / 400.0 ** 2) / 2) * pg[0]
    assert abs(np.std(d4[:, 0]) / np.std(d3[:, 0]) / expect - 1) < 0.03


def test_world_loss_band_p_decreases_downwards():
    cube = np.array([[1, 0, 0, 5000], [-1, 0, 0, 5000], [0, 1, 0, 3000], [0, -1, 0, -2000], [0, 0, 1, 5000],
                     [0, 0, -1, 5000]], float)
    world = [("CONCRETE_MARS", cube, np.array([-5000.0, 2000, -5000]), np.array([5000.0, 3000, 5000]))]
    rng = np.random.default_rng(8)
    o = np.tile([0.0, 10000.0, 0.0], (500, 1))
    d = np.tile([0.0, -1.0, 0.0], (500, 1))
    out = T.world_loss(rng, o, d, np.full(500, 2000.0), world, [10000.0, 3000.0, 2000.0, 0.0], return_band_p=True)
    bp = out[5]
    assert (np.diff(bp, axis=1) <= 1e-9).all() and (bp[:, 1] < bp[:, 0] - 100).all()



def test_slab_loss_range_out_follows_the_mean_range():
    # survival through concrete brackets the CSDA range of the same dE/dx (numerical integral of 1/dE/dx from TK_STOP
    # to the initial kinetic energy): > 90 % survive 0.9 R, < 10 % survive 1.1 R, for 0.5 and 1 GeV/c muons
    rng = np.random.default_rng(12)
    n = 3000
    for p0 in (500.0, 1000.0):
        tk0 = np.sqrt(p0 ** 2 + E.MMU ** 2) - E.MMU
        tk = np.linspace(T.TK_STOP, tk0, 4000)
        pk = np.sqrt((tk + E.MMU) ** 2 - E.MMU ** 2)
        R = np.trapezoid(1.0 / E.dedx_restricted(M.CONCRETE_MARS, pk, 1e9), tk)
        s09 = np.mean(T.slab_loss(rng, M.CONCRETE_MARS, np.full(n, p0), np.full(n, 0.9 * R)) < tk0)
        s11 = np.mean(T.slab_loss(rng, M.CONCRETE_MARS, np.full(n, p0), np.full(n, 1.1 * R)) < tk0)
        assert s09 > 0.9 and s11 < 0.1, (p0, R, s09, s11)


def test_highland_increments_sum_to_the_stack_width():
    p = np.full(3, 1000.0)
    x = np.linspace(0, 0.05, 11)
    v = sum(T.highland_var_increment(p, a, b) for a, b in zip(x[:-1], x[1:]))
    assert np.allclose(np.sqrt(v), E.highland_theta0(p, 0.05), rtol=1e-9)


def test_single_scatter_tail_rate_and_spectrum():
    rng = np.random.default_rng(21)
    n = 200_000
    chic2 = np.full(n, 1e-5)
    # theta_n far above the angles: bare 1/theta^3 (up to THETA_TAIL_MAX)
    th = T.single_scatter_tail(rng, chic2, np.full(n, 0.01), np.full(n, 1e6))
    k = (th != 0).any(1)
    lam = 1e-5 * (1 / 0.01 ** 2 - 1 / T.THETA_TAIL_MAX ** 2)
    assert abs(k.mean() / (1 - np.exp(-lam)) - 1) < 0.03
    sp = np.hypot(th[k, 0], th[k, 1])
    assert abs((sp > 0.02).mean() - 0.25) < 0.02          # P(theta > 2 theta_min) = 1/4 for 1/theta^3
    # form factor: rate suppressed by the integral of F^2 = (1 + t^2/tn^2)^-4 over the 1/theta^3 spectrum
    tn = 0.02
    t2 = T.single_scatter_tail(rng, chic2, np.full(n, 0.01), np.full(n, tn))
    x = np.linspace(0.01, T.THETA_TAIL_MAX, 200_001)
    f = 2e-5 / x ** 3 * (1 + (x / tn) ** 2) ** -4
    lam2 = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x))
    assert abs((t2 != 0).any(1).mean() / (1 - np.exp(-lam2)) - 1) < 0.05


def test_tail_threshold_crossing_and_fixed_k():
    th0 = np.array([1e-3])
    assert np.allclose(T.tail_threshold(None, th0, 3.0), 3e-3)
    c = np.array([1e-8])
    u = T.tail_threshold(c, th0)[0] / th0[0]
    assert abs(u ** 4 * np.exp(-u * u / 2) - 2 * c[0] / th0[0] ** 2) < 1e-10 and u > 2


def test_chic2_matches_lynch_dahl_aluminium():
    # Lynch-Dahl: chi_c^2 = 0.157 Z(Z+1)/A rho t / (p beta)^2; 1 cm Al at 1 GeV/c
    p = 1000.0
    pb = p * p / np.sqrt(p * p + E.MMU ** 2)
    ref = 0.157 * 13 * 14 / 26.981538 * 2.699 * 1.0 / pb ** 2
    assert np.isclose(T.chi_c2(M.ALUMINIUM, 10.0, p), ref)
    assert np.isclose(T.chi_c2(M.ALUMINIUM, 10.0, p, electrons=False), ref * 13 / 14)
    # sqrt(20) hbar c / (p 1.2 fm A^(1/3)), A^(1/3) = 3.0 for Al
    assert np.isclose(T.theta_nuclear(M.ALUMINIUM, p), np.sqrt(20) * 197.3269804 / (p * 1.2 * 26.981538 ** (1 / 3)))


def test_msc_to_plane_tail_widens_only_the_tails():
    rng = np.random.default_rng(5)
    n = 200_000
    o = np.tile([0.0, 3000.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    x0 = E.radiation_length_mm(M.CONCRETE_MARS)
    bxx0 = np.tile([1000.0 / x0], (n, 1))
    bh = np.tile([1500.0], (n, 1))
    p = np.full(n, 3000.0)
    chic2 = T.chi_c2(M.CONCRETE_MARS, np.full((n, 1), 1000.0), 3000.0, electrons=False)
    thn = T.theta_nuclear(M.CONCRETE_MARS, np.full((n, 1), 3000.0))
    _, d0, _ = T.msc_to_plane(rng, o, d, p, 0.0, x0, bxx0.sum(1), bxx0, bh)
    _, d1, _ = T.msc_to_plane(rng, o, d, p, 0.0, x0, bxx0.sum(1), bxx0, bh, band_chic2=chic2, band_thn=thn)
    a0, a1 = np.arctan2(d0[:, 0], -d0[:, 1]), np.arctan2(d1[:, 0], -d1[:, 1])
    th0 = E.highland_theta0(3000.0, 1000.0 / x0)
    # core unchanged (the tail starts where its density crosses the Gaussian, ~3.6 theta0 here); beyond 4 theta0 the
    # Gaussian leaves ~13 of 2e5; the tail adds some tens (the form factor cuts it near theta_n ~ 6 theta0), and with
    # no form factor (theta_n huge) many more
    assert abs(np.median(np.abs(a1)) / np.median(np.abs(a0)) - 1) < 0.02
    n0, n1 = (np.abs(a0) > 4 * th0).sum(), (np.abs(a1) > 4 * th0).sum()
    _, d2, _ = T.msc_to_plane(rng, o, d, p, 0.0, x0, bxx0.sum(1), bxx0, bh, band_chic2=chic2, band_thn=thn * 1e6)
    n2 = (np.abs(np.arctan2(d2[:, 0], -d2[:, 1])) > 4 * th0).sum()
    assert n1 > 2 * max(n0, 1) and n2 > 5 * n1


def test_electron_tail_ends_at_the_electron_kinematic_limit():
    rng = np.random.default_rng(31)
    n = 400_000
    th = T.single_scatter_tail(rng, np.full(n, 1e-8), np.full(n, 1e-3), None, T.ME_OVER_MMU)
    sp = np.hypot(th[:, 0], th[:, 1])
    k = sp > 0
    lam = 1e-8 * (1 / 1e-3 ** 2 - 1 / T.ME_OVER_MMU ** 2)    # ~0.01: almost always one scatter
    assert abs(k.mean() / (1 - np.exp(-lam)) - 1) < 0.05
    assert np.mean(sp[k] > T.ME_OVER_MMU * 1.0001) < 0.01     # only the rare sums of two pass the limit
    # electron part = chi_c^2 (Z+1) minus the nuclear Z^2 part: Al 1 cm at 1 GeV/c
    assert np.isclose(T.chi_c2_electron(M.ALUMINIUM, 10.0, 1000.0), T.chi_c2(M.ALUMINIUM, 10.0, 1000.0) / 14)
    # tungsten: X0 3.5 mm (PDG), the wire material
    assert abs(E.radiation_length_mm(M.TUNGSTEN) / 3.504 - 1) < 0.02


def test_mpv_fit_is_stable_where_argmax_is_not():
    from mu2efastsim import stats
    rng = np.random.default_rng(41)
    tc = 990e-6
    gas = M.Material("StrawGas", 0.001696, 158.62097145526, {"Ar": 0.784057035890315, "O": 0.15700963928136,
                                                              "C": 0.0589333248283256})
    fits = []
    for _ in range(4):
        L = rng.uniform(1, 8, 40_000)
        fits.append(stats.mpv_fit(1e3 * E.sample_loss(rng, gas, np.full(L.size, 3000.0), L, tc), 0, 4, 200))
    mu = np.array([f[0] for f in fits])
    err = np.array([f[1] for f in fits])
    # the same distribution four times: fitted MPVs agree within their errors, and errors are ~2-4 %
    assert np.all(np.isfinite(mu)) and np.std(mu) < 3 * err.mean() and np.all(err / mu < 0.05)


def test_band_len_spreads_displacement_and_lowers_correlation():
    rng = np.random.default_rng(3)
    n = 200000
    o = np.tile([0.0, 10000.0, 0.0], (n, 1))
    d = np.tile([0.0, -1.0, 0.0], (n, 1))
    p = np.full(n, 1000.0)
    bxx0 = np.full((n, 1), 5.0)
    bh = np.full((n, 1), 5000.0)
    blen = np.full((n, 1), 10000.0)
    _, d0, _ = T.msc_to_plane(np.random.default_rng(3), o, d, p, 0.0, 1e9, 5.0, bxx0, bh)
    e1, d1, _ = T.msc_to_plane(np.random.default_rng(3), o, d, p, 0.0, 1e9, 5.0, bxx0, bh, band_len=blen)
    assert np.allclose(d0, d1)                                     # angles unchanged
    y, t = e1[:, 0], np.arctan(d1[:, 0] / -d1[:, 1])
    rho = np.corrcoef(y, t)[0, 1]
    assert abs(rho - np.sqrt(3) / 2) < 0.01                        # one band spanning the whole path = uniform slab
    assert abs(y.std() / (t.std() * 10000.0) - 1 / np.sqrt(3)) < 0.01


def test_all_primary_muons_keys():
    import pandas as pd
    from mu2efastsim import tables as T
    g = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1, 1, 2, 3], "idx": [1, 0, 2, 0, 0],
                      "pdg": [13, -13, 22, 13, 2112], "e": 1.0})
    m = T.all_primary_muons(g)
    assert m[["event", "idx", "mukey"]].values.tolist() == [[1, 0, 1], [1, 1, 2], [2, 0, 1]]
    first = T.primary_muons(g.sort_values(["event", "idx"]))
    assert first.idx.tolist() == m[m.mukey == 1].idx.tolist()
