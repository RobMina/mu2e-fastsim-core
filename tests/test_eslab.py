"""Unit tests: straight-line electron slab crossing (mu2efastsim.eslab), energy bookkeeping and thin-slab limits."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import eioni as I                                       # noqa: E402
from mu2efastsim import eslab as S                                       # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

TCUT, KCUT = 0.03, 1e-3


def _run(name, T0, L, n, seed, positron=False):
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    return S.electron_slab(np.random.default_rng(seed), M.BY_NAME[name], np.full(n, T0), d, L, TCUT, KCUT, positron)


@pytest.mark.parametrize("positron", [False, True])
def test_energy_is_conserved_per_track(positron):
    n, T0 = 5000, 105.0
    r = _run("G4_W", T0, 2.0, n, 1, positron)
    out = r.continuous + np.bincount(r.photons.track, r.photons.energy, n) + np.bincount(r.deltas.track,
                                                                                         r.deltas.energy, n)
    assert T0 - r.T == pytest.approx(out, abs=1e-9)
    assert ((r.photons.s >= 0) & (r.photons.s <= 2.0)).all()


def test_thin_slab_mean_loss_and_photon_count():
    n, T0, L = 200_000, 105.0, 0.1
    mat = M.BY_NAME["G4_Al"]
    r = _run("G4_Al", T0, L, n, 3)
    lost = T0 - r.T
    expect = (I.dedx_restricted(mat, T0, 1e9) + B.dedx_restricted(mat, T0, 1e9))[0] * L
    assert abs(lost.mean() - expect) < 4 * lost.std() / np.sqrt(n)
    lam = B.xsec_per_mm(mat, T0, KCUT)[0] * L
    assert r.photons.track.size / n == pytest.approx(lam, rel=0.05)
    assert not r.stopped.any()


def test_slow_electron_stops_and_deposits_the_rest():
    n = 5000
    r = _run("G4_Al", 1.0, 5.0, n, 5)
    assert r.stopped.all() and (r.T == 0).all()
    sec = np.bincount(r.photons.track, r.photons.energy, n) + np.bincount(r.deltas.track, r.deltas.energy, n)
    assert r.continuous == pytest.approx(1.0 - sec, abs=1e-9)


def test_recoil_keeps_directions_unit_and_forward():
    r = _run("G4_W", 105.0, 1.0, 5000, 7)
    assert np.linalg.norm(r.direction, axis=1) == pytest.approx(1.0)
    assert np.median(r.direction[:, 2]) > 0.9999


def test_tracks_below_both_cuts_range_out():
    # W cuts of the 0.01 mm range cut (gamma 7.7 keV, e- 81.5 keV): a few-keV electron has no discrete process left
    n, T0 = 2000, 0.005
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    r = S.electron_slab(np.random.default_rng(17), M.BY_NAME["G4_W"], np.full(n, T0), d, 1.0, 0.0814923, 0.00770222)
    assert r.stopped.all() and r.photons.track.size == 0 and r.deltas.track.size == 0
    assert r.continuous == pytest.approx(T0)
