"""Unit tests: Seltzer-Berger electron bremsstrahlung (mu2efastsim.ebrem) against NIST ESTAR and its own cross section.

ESTAR radiative stopping powers [MeV cm2/g] (physics.nist.gov/cgi-bin/Star/e_table-t.pl, read 2026-10-30) at
T = 1, 10, 50, 105, 500 MeV; ESTAR's radiative loss comes from the same Seltzer-Berger cross sections, without the
dielectric suppression Geant4 applies, so agreement is at the per-mille to half-percent level.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

T_ESTAR = np.array([1.0, 10.0, 50.0, 105.0, 500.0])
ESTAR_RLOSS = {
    "G4_Al": [0.02119, 0.2858, 1.761, 3.912, 19.92],
    "G4_W": [0.1159, 1.132, 6.477, 14.26, 71.78],
    "G4_MYLAR": [0.01138, 0.1626, 1.031, 2.309, 11.88],
    "G4_POLYSTYRENE": [0.01011, 0.1464, 0.9334, 2.095, 10.81],
}
KCUT = 1e-3  # MeV: the 990 eV gamma floor that the 0.01 mm range cut of Mu2e primary jobs gives in every material


def test_table_covers_the_elements_and_is_hashed():
    row, kappa, lnT, chi = B.tables()
    assert set(row) == {z for z, _ in M.ELEMENTS.values()}
    assert kappa.size == 32 and lnT.size == 57 and chi.shape == (len(row), 57, 32)
    assert kappa[0] == 1e-12 and kappa[-1] == 1.0
    assert lnT[0] == pytest.approx(np.log(1e-3), abs=1e-4) and lnT[-1] == pytest.approx(np.log(1e4), abs=1e-4)
    d = np.load(B.DATA)
    assert d["sha256"].size == len(row) and all(len(h) == 64 for h in d["sha256"])


def test_bilinear_hits_the_nodes_and_clamps():
    row, kappa, lnT, chi = B.tables()
    i, j = 7, 30
    assert B.chi(13, kappa[i], lnT[j]) == pytest.approx(chi[row[13], j, i])
    assert B.chi(13, 2.0, 20.0) == pytest.approx(chi[row[13], -1, -1])
    mid = B.chi(13, 0.5 * (kappa[i] + kappa[i + 1]), lnT[j])
    assert mid == pytest.approx(0.5 * (chi[row[13], j, i] + chi[row[13], j, i + 1]))


@pytest.mark.parametrize("name", sorted(ESTAR_RLOSS))
def test_total_radiative_loss_matches_estar(name):
    mat = M.BY_NAME[name]
    rloss = B.dedx_restricted(mat, T_ESTAR, 1e9) * 10.0 / mat.rho      # MeV/mm -> MeV cm2/g
    assert rloss == pytest.approx(ESTAR_RLOSS[name], rel=0.01)


def test_restricted_loss_grows_with_the_cut_to_the_total():
    T = np.array([105.0])
    cuts = [1e-3, 1e-2, 1.0, 50.0, 105.0, 1e9]
    d = np.array([B.dedx_restricted(M.BY_NAME["G4_Al"], T, c)[0] for c in cuts])
    assert np.all(np.diff(d) >= 0) and d[-2] == pytest.approx(d[-1])
    assert d[0] < 1e-4 * d[-1]


def test_sampled_spectrum_follows_the_cross_section():
    rng = np.random.default_rng(3)
    mat, T = M.BY_NAME["G4_Al"], 105.0
    k = B.sample_k(rng, mat, np.full(400_000, T), KCUT)
    assert (k > KCUT).all() and (k <= T).all()
    edges = np.geomspace(KCUT, T, 21)
    obs = np.histogram(k, edges)[0]
    kp2 = B.MIGDAL * mat.electron_density_per_mm3 * (T + B.ME) ** 2
    tail = np.array([B._xsec_atom(13, np.array([T]), np.array([e]), False, np.array([kp2]))[0] for e in edges[:-1]])
    prob = (tail - np.append(tail[1:], 0.0)) / tail[0]
    exp = prob * k.size
    chi2 = ((obs - exp) ** 2 / exp).sum()
    assert chi2 / (edges.size - 2) < 2.0


@pytest.mark.slow
def test_discrete_plus_continuous_loss_is_the_total():
    rng = np.random.default_rng(5)
    mat, T, kc = M.BY_NAME["G4_MYLAR"], 105.0, 0.1
    k = B.sample_k(rng, mat, np.full(300_000, T), kc)
    above = B.xsec_per_mm(mat, T, kc)[0] * k.mean()
    total = B.dedx_restricted(mat, T, 1e9)[0]
    assert B.dedx_restricted(mat, T, kc)[0] + above == pytest.approx(total, rel=0.01)


@pytest.mark.slow
def test_multi_element_selection_follows_the_partial_cross_sections():
    rng = np.random.default_rng(7)
    mat, T = M.BY_NAME["G4_MYLAR"], np.full(200_000, 50.0)
    part = B._partial_xsec(mat, T[:1], KCUT, False)[:, 0]
    iz = B._select_element(rng, mat, T, KCUT, False)
    frac = np.bincount(iz, minlength=part.size) / iz.size
    assert frac == pytest.approx(part / part.sum(), abs=0.005)


def test_positron_loses_less_at_low_energy_and_the_same_at_high():
    al = M.BY_NAME["G4_Al"]
    e = B.dedx_restricted(al, np.array([0.05, 105.0]), 1e9)
    p = B.dedx_restricted(al, np.array([0.05, 105.0]), 1e9, positron=True)
    assert p[0] < 0.9 * e[0]
    assert p[1] == pytest.approx(e[1], rel=0.01)


def test_photon_angle_follows_the_2bs_density():
    rng = np.random.default_rng(9)
    T, k, z = 105.0, 20.0, 13
    cost = B.sample_cos_theta(rng, np.full(400_000, T), np.full(400_000, k), np.full(400_000, z))
    assert (np.abs(cost) <= 1).all()
    e = T + B.ME
    gam = e / B.ME
    beta = np.sqrt((gam - 1) * (gam + 1)) / gam
    ymax = 2 * beta * (1 + beta) * gam * gam
    y = (1 - cost) * ymax / 2
    r = (e - k) / e
    fz = 0.00008116224 * np.cbrt(z) * np.cbrt(z + 1)

    def dens(yy):    # proposal 1/(1+y)^2 times the 2BS rejection function
        y2 = (1 + yy) ** 2
        x = 4 * yy * r / y2
        return (4 * x - (1 + r) ** 2 - (1 + r * r - x) * np.log(fz / y2)) / y2

    edges = np.concatenate([[0.0], np.geomspace(0.01, 100.0, 24)])
    obs = np.histogram(y, edges)[0]
    grid = np.concatenate([[0.0], np.geomspace(1e-6, ymax, 400_001)])
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens(grid[1:]) + dens(grid[:-1])) * np.diff(grid))])
    exp = np.diff(np.interp(edges, grid, cdf)) / cdf[-1] * y.size
    assert ((obs - exp) ** 2 / exp).sum() / (edges.size - 1) < 2.0
    assert 0.3 < np.median(np.sqrt(y)) < 1.5      # theta ~ 1/gamma


def test_rotate_uz_and_emission_momentum():
    rng = np.random.default_rng(11)
    d = rng.normal(size=(1000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    z = np.tile([0.0, 0.0, 1.0], (1000, 1))
    assert B.rotate_uz(d, z) == pytest.approx(d)
    T = np.full(1000, 105.0)
    k = np.full(1000, 30.0)
    dg, de = B.emit(rng, T, d, k, np.full(1000, 13))
    assert np.linalg.norm(dg, axis=1) == pytest.approx(1.0) and np.linalg.norm(de, axis=1) == pytest.approx(1.0)
    rest = np.sqrt(T * (T + 2 * B.ME))[:, None] * d - k[:, None] * dg      # momentum left to the parent
    assert np.abs(np.cross(rest, de)).max() < 1e-9 * np.linalg.norm(rest, axis=1).max()
    assert np.einsum("ij,ij->i", dg, d).min() > 0.0


def test_sample_along_thin_slab_counts_and_energy():
    rng = np.random.default_rng(13)
    mat, T0, L = M.BY_NAME["G4_Al"], 105.0, 0.1
    T = np.full(200_000, T0)
    idx, s, k, z, Tend = B.sample_along(rng, mat, T, L, KCUT)
    n = np.bincount(idx, minlength=T.size)
    assert n.mean() == pytest.approx(B.xsec_per_mm(mat, T0, KCUT)[0] * L, rel=0.02)
    assert ((s >= 0) & (s < L)).all() and (z == 13).all()
    assert T0 - Tend == pytest.approx(np.bincount(idx, weights=k, minlength=T.size))
    loss = T0 - Tend
    expect = (B.dedx_restricted(mat, T0, 1e9) - B.dedx_restricted(mat, T0, KCUT))[0] * L
    assert abs(loss.mean() - expect) < 4 * loss.std() / np.sqrt(loss.size)    # hard photons dominate the mean


def test_above_one_gev_is_refused():
    with pytest.raises(ValueError, match="1 GeV"):
        B.dedx_restricted(M.BY_NAME["G4_Al"], 2000.0, 1e9)
