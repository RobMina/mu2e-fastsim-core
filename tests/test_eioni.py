"""Unit tests: Moller / Bhabha ionisation (mu2efastsim.eioni) against NIST ESTAR and its own cross section.

ESTAR collision stopping powers [MeV cm2/g] (physics.nist.gov/cgi-bin/Star/e_table-t.pl, read 2026-10-30) at
T = 0.1, 1, 10, 50, 105, 500 MeV for electrons.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import eioni as I                                       # noqa: E402
from mu2efastsim import eloss as E                                       # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

T_ESTAR = np.array([0.1, 1.0, 10.0, 50.0, 105.0, 500.0])
ESTAR_CLOSS = {
    "G4_Al": [3.177, 1.465, 1.636, 1.789, 1.851, 1.969],
    "G4_W": [2.047, 1.016, 1.203, 1.362, 1.422, 1.532],
    "G4_MYLAR": [3.832, 1.71, 1.831, 1.986, 2.047, 2.173],
    "G4_POLYSTYRENE": [4.034, 1.794, 1.916, 2.073, 2.136, 2.266],
}


@pytest.mark.parametrize("name", sorted(ESTAR_CLOSS))
def test_unrestricted_collision_loss_matches_estar(name):
    mat = M.BY_NAME[name]
    closs = I.dedx_restricted(mat, T_ESTAR, 1e9) * 10.0 / mat.rho
    assert closs == pytest.approx(ESTAR_CLOSS[name], rel=0.015)


def test_positron_differs_little_from_electron_at_high_energy():
    al = M.BY_NAME["G4_Al"]
    e, p = I.dedx_restricted(al, 105.0, 1e9), I.dedx_restricted(al, 105.0, 1e9, positron=True)
    assert abs(p / e - 1) < 0.05
    assert I.max_secondary(10.0) == 5.0 and I.max_secondary(10.0, positron=True) == 10.0


@pytest.mark.parametrize("positron", [False, True])
def test_restricted_plus_deltas_is_the_total(positron):
    rng = np.random.default_rng(21)
    mat, T, tc = M.BY_NAME["G4_Al"], 105.0, 0.05
    w = I.sample_delta(rng, np.full(400_000, T), tc, positron)
    assert (w >= tc).all() and (w <= I.max_secondary(T, positron)).all()
    total = I.dedx_restricted(mat, T, 1e9, positron)[0]
    rest = I.dedx_restricted(mat, T, tc, positron)[0] + I.xsec_per_mm(mat, T, tc, positron)[0] * w.mean()
    assert rest == pytest.approx(total, rel=0.01)


def test_moller_spectrum_follows_the_cross_section():
    rng = np.random.default_rng(23)
    mat, T, tc = M.BY_NAME["G4_MYLAR"], 50.0, 0.01
    w = I.sample_delta(rng, np.full(300_000, T), tc)
    edges = np.geomspace(tc, T / 2, 16)
    obs = np.histogram(w, edges)[0]
    tail = np.array([I.xsec_per_mm(mat, T, e)[0] for e in edges[:-1]])
    exp = (tail - np.append(tail[1:], 0.0)) / tail[0] * w.size
    assert ((obs - exp) ** 2 / exp).sum() / (edges.size - 2) < 2.0


def test_delta_kinematics_conserve_momentum():
    rng = np.random.default_rng(25)
    n = 2000
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    T, w = np.full(n, 105.0), np.geomspace(0.01, 50.0, n)
    dd, de = I.delta_direction(rng, T, d, w)
    p, pd = np.sqrt(T * (T + 2 * E.ME)), np.sqrt(w * (w + 2 * E.ME))
    cost = np.einsum("ij,ij->i", dd, d)
    assert cost == pytest.approx(np.minimum(w * (T + 2 * E.ME) / (pd * p), 1.0))
    rest = p[:, None] * d - pd[:, None] * dd
    assert np.abs(np.cross(rest, de)).max() < 1e-9 * p.max()


def test_continuous_loss_mean_and_no_gaussian_branch():
    rng = np.random.default_rng(27)
    mat, T, L, tc, kc = M.BY_NAME["G4_Al"], np.full(200_000, 105.0), 0.2, 0.03, 0.001
    loss = I.continuous_loss(rng, mat, T, L, tc, kc)
    mean = (I.dedx_restricted(mat, 105.0, tc) + B.dedx_restricted(mat, 105.0, kc))[0] * L
    assert loss.mean() == pytest.approx(mean, rel=0.01)
    assert (loss >= 0).all() and loss.std() > 0.05 * mean


def test_fluctuate_refactor_keeps_muon_sample_loss():
    p, L, tc = np.geomspace(50.0, 5e4, 5000), 10.0, 0.0872
    a = E.sample_loss(np.random.default_rng(31), M.POLYSTYRENE, p, L, tc)
    tkin, beta2, bg2, gam, tmax = E.kinematics(p)
    cut = np.minimum(tc, tmax)
    mean = E.dedx_restricted(M.POLYSTYRENE, p, tc) * L
    b = E.fluctuate(np.random.default_rng(31), M.POLYSTYRENE, mean, np.full(p.size, L), cut, tmax, beta2, gam)
    assert np.array_equal(a, b)
