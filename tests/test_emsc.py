"""Unit tests: electron multiple scattering (mu2efastsim.emsc) and positron annihilation (mu2efastsim.eannihil)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import eannihil as A                                    # noqa: E402
from mu2efastsim import emsc as E                                        # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

AL, W = M.BY_NAME["G4_Al"], M.BY_NAME["G4_W"]
TCUT = 0.0341


def test_screening_is_twice_moliere_at_high_energy():
    # light-particle screening of G4WentzelOKandVIxSection: 2 A_Moliere (the 3.76 term times sqrt(tau/(tau+Z^2/3)))
    T, z = 1000.0, 13
    p2 = T * (T + 2 * E.ME)
    a_mol = 0.25 * (E.ALPHA * E.ME * z ** (1 / 3) / 0.88534) ** 2 / p2 * (1.13 + 3.76 * (E.ALPHA * z) ** 2)
    assert E.screening([z], [T])[0, 0] == pytest.approx(2 * a_mol, rel=2e-3)


def test_transport_xsec_splits_into_soft_and_hard():
    T = np.array([0.5, 5.0, 105.0])
    uc = E.u_cut(AL, T, TCUT)
    full = E.transport_xsec(AL, T, 2.0, TCUT)
    soft = E.transport_xsec(AL, T, uc, TCUT)
    assert np.all((0 < soft) & (soft < full))
    # hard (Rutherford) rate at u_c is Sigma_1 / C1 by construction
    z, n, a = E.elements(AL)
    s = E.screening(z, T)[0]
    k = n[0] * E.COEFF * z[0] * (T + E.ME) ** 2 / (T * (T + 2 * E.ME)) ** 2
    ue = E.u_electron(T, TCUT)
    rate = k * z[0] * (1 / (uc + s) - 1 / (2 + s)) + np.where(ue > uc, k * (1 / (uc + s) - 1 / (ue + s)), 0)
    assert rate == pytest.approx(full / E.C1, rel=0.02)


def test_lambda1_against_highland_scale():
    # Lewis: <theta^2> ~ 2 L / lambda_1 exceeds the Highland core (13.6 / p)^2 x / X0 by the single-scattering tail
    L, p = 1.0, np.sqrt(105.0 * (105.0 + 2 * E.ME))
    th2 = 2 * L / E.lambda1(AL, 105.0, TCUT)[0]
    hl = 2 * (13.6 / p) ** 2 * (L / 88.97) * (1 + 0.038 * np.log(L / 88.97)) ** 2
    assert 1.0 < th2 / hl < 2.0


def test_mean_cosine_follows_lewis():
    # fixed energy, no loss: soft steps plus Poisson hard collisions give <cos theta> = exp(-s Sigma_1) up to Mott
    rng = np.random.default_rng(3)
    n, T, h, nstep = 100_000, 20.0, 0.05, 20
    uc = E.u_cut(AL, np.full(n, T), TCUT)
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    rate = E.hard_xsec(AL, np.full(n, T), uc, TCUT)
    for _ in range(nstep):
        d = E.deflect(rng, d, E.sample_soft(rng, AL, np.full(n, T), h, uc, TCUT))
        k = rng.poisson(rate * h)
        for j in range(1, k.max() + 1):
            m = np.flatnonzero(k >= j)
            u, acc = E.sample_hard(rng, AL, np.full(m.size, T), uc[m], TCUT)
            d[m[acc]] = E.deflect(rng, d[m[acc]], u[acc])
    # Sigma_1 with the Mott and form factors on the hard part, integrated numerically above u_c
    z, na, a = E.elements(AL)
    s, ue = E.screening(z, [T])[0, 0], E.u_electron(T, TCUT)
    p2, invb2, _ = E.kinematics(T)
    k = na[0] * E.COEFF * z[0] * invb2 / p2
    u = np.geomspace(uc[0], 2.0, 20001)
    dens = k * (z[0] / (1 + E.FORM_FACTOR * a[0] ** 0.54 * p2 * u) ** 2 + (u < ue)) / (u + s) ** 2
    dens *= E.mott_factor(z[0], u, invb2)
    f = u * dens
    hard = (0.5 * (f[1:] + f[:-1]) * np.diff(u)).sum()
    expect = np.exp(-h * nstep * (E.transport_xsec(AL, T, uc[0], TCUT)[0] + hard))
    assert d[:, 2].mean() == pytest.approx(expect, abs=4 * d[:, 2].std() / np.sqrt(n))


def test_hard_collisions_follow_screened_rutherford():
    rng = np.random.default_rng(5)
    n, T = 400_000, 105.0
    uc = E.u_cut(W, np.full(n, T), 0.0815)
    u, acc = E.sample_hard(rng, W, np.full(n, T), uc, 0.0815)
    u = u[acc]
    assert acc.mean() > 0.5 and (u >= uc[0]).all()
    s = E.screening([74], [T])[0, 0]
    edges = np.geomspace(uc[0], 2.0, 21)
    w = 1 / (edges[:-1] + s) - 1 / (edges[1:] + s)       # Rutherford; Mott and form factor are small below u ~ 0.02
    h = np.histogram(u, edges)[0]
    m = edges[1:] < 0.02
    ratio = h[m] / w[m]
    assert ratio / ratio[0] == pytest.approx(1.0, rel=0.06)


def test_heitler_high_energy_limit():
    T = 105.0
    gam = T / A.ME + 1
    approx = A.PI_RE2 / gam * (np.log(2 * gam) - 1)
    assert A.xsec_per_electron(T) == pytest.approx(approx, rel=0.02)


def test_annihilation_photons_conserve_four_momentum():
    rng = np.random.default_rng(7)
    T = np.array([0.01, 1.0, 105.0, 50.0])
    d = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0.6, 0.8], [0, 0, -1.0]])
    e, u = A.in_flight(rng, T, d)
    assert e.sum(axis=1) == pytest.approx(T + 2 * A.ME)
    p = np.sqrt(T * (T + 2 * A.ME))
    assert (e[:, :, None] * u).sum(axis=1) == pytest.approx(p[:, None] * d, abs=1e-9)
    assert np.linalg.norm(u, axis=2) == pytest.approx(1.0)
    e0, u0 = A.at_rest(rng, 3)
    assert e0 == pytest.approx(A.ME) and (u0[:, 0] + u0[:, 1]) == pytest.approx(0.0, abs=1e-12)


def test_tabulated_mott_ratio_limits_and_envelope():
    rng = np.random.default_rng(9)
    c = E.mott_coefficients()
    assert c.shape == (93, 5, 6) and c[1, 0, 0] == 1.0
    # forward limit is Rutherford; the electron enhancement at intermediate angles grows with Z
    b = np.sqrt(1 - (E.ME / (105.0 + E.ME)) ** 2)
    assert E.mott_ratio_table(74, 1e-8, b) == pytest.approx(1.0, abs=5e-3)
    assert E.mott_ratio_table(74, 0.5, b) > E.mott_ratio_table(13, 0.5, b) + 0.5
    for z in (1, 6, 8, 13, 26, 74, 82):
        T = np.geomspace(1e-3, 1e3, 400)
        env = E.mott_envelope([z], T)[0]
        u = rng.uniform(0, 2, (2000, T.size))
        r = E.mott_factor(np.full(u.shape, z), u, (T + E.ME) ** 2 / (T * (T + 2 * E.ME)))
        assert (r <= env[None, :] + 1e-12).all() and (r >= 0).all()


def test_form_factor_threshold_accepts_more_large_angle_collisions_below_it(monkeypatch):
    W = M.BY_NAME["G4_W"]
    n, T, uc = 200_000, 50.0, 0.5
    assert E.FF_TMIN == 100.0                       # Geant4 option 4: form factor only in the WentzelVI range
    acc = {}
    for tmin in (0.0, 100.0):
        monkeypatch.setattr(E, "FF_TMIN", tmin)
        u, a = E.sample_hard(np.random.default_rng(3), W, np.full(n, T), uc, TCUT)
        acc[tmin] = a.mean()
    assert acc[100.0] > acc[0.0] * 1.1            # no nuclear form factor below 100 MeV
    monkeypatch.setattr(E, "FF_TMIN", 0.0)
    u0, a0 = E.sample_hard(np.random.default_rng(3), W, np.full(n, 150.0), uc, TCUT)
    monkeypatch.setattr(E, "FF_TMIN", 100.0)
    u1, a1 = E.sample_hard(np.random.default_rng(3), W, np.full(n, 150.0), uc, TCUT)
    assert np.array_equal(a0, a1) and np.array_equal(u0, u1)   # above the threshold: unchanged
