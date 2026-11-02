"""Unit tests: straight-line electron slab crossing (mu2efastsim.eslab), energy bookkeeping and thin-slab limits."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import eioni as I                                       # noqa: E402
from mu2efastsim import eannihil as A                                    # noqa: E402
from mu2efastsim import eslab as S                                       # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

TCUT, KCUT = 0.03, 1e-3


def _run(name, T0, L, n, seed, positron=False, msc=False):
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    return S.electron_slab(np.random.default_rng(seed), M.BY_NAME[name], np.full(n, T0), d, L, TCUT, KCUT, positron,
                           msc=msc)


def _balance(r, n):
    """Energy carried off per track: continuous + photons + deltas + annihilation photons - 2 m_e per annihilation."""
    a = r.annihilation
    return (r.continuous + np.bincount(r.photons.track, r.photons.energy, n)
            + np.bincount(r.deltas.track, r.deltas.energy, n) + np.bincount(a.track, a.energy - B.ME, n))


@pytest.mark.parametrize("msc", [False, True])
@pytest.mark.parametrize("positron", [False, True])
def test_energy_is_conserved_per_track(positron, msc):
    n, T0 = (600 if msc else 3000), 105.0
    r = _run("G4_W", T0, 2.0, n, 1, positron, msc)
    assert T0 - r.T == pytest.approx(_balance(r, n), abs=1e-9)
    assert ((r.photons.s >= 0) & (r.photons.s <= r.path[r.photons.track] + 1e-12)).all()
    z = r.position[:, 2]
    assert ((z >= -1e-9) & (z <= 2.0 + 1e-9)).all()
    out = ~r.stopped
    assert np.allclose(z[out & r.reflected], 0.0, atol=1e-9) and np.allclose(z[out & ~r.reflected], 2.0, atol=1e-9)
    if positron:
        assert np.bincount(r.annihilation.track, minlength=n)[r.stopped].min() == 2


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


def test_msc_off_moves_only_by_recoil():
    r = _run("G4_Al", 105.0, 1.0, 2000, 11)
    assert np.median(r.path) == pytest.approx(1.0, rel=1e-5) and (r.path >= 1.0 - 1e-12).all() and not r.reflected.any()
    assert np.median(np.hypot(r.position[:, 0], r.position[:, 1])) < 1e-4


def test_msc_scatters_and_lengthens_path():
    n = 1500
    r = _run("G4_W", 105.0, 3.5, n, 13, msc=True)
    tr = ~r.stopped & ~r.reflected
    assert r.path[tr].mean() > 3.5 * 1.01
    assert 0.1 < r.stopped.mean() < 0.3 and r.reflected.mean() < 0.02
    assert np.median(np.hypot(r.position[tr, 0], r.position[tr, 1])) > 0.1


def test_positron_annihilation_in_flight_rate():
    n, L = 200_000, 1.0
    mat = M.BY_NAME["G4_Al"]
    r = _run("G4_Al", 105.0, L, n, 19, positron=True)
    lam = A.xsec_per_mm(mat, 105.0) * L
    assert r.annihilated.sum() == pytest.approx(lam * n, rel=5 / np.sqrt(lam * n))
    assert r.annihilation.track.size == 2 * r.stopped.sum()


def test_slow_electrons_stop_or_reflect_with_msc():
    # 0.5 MeV in W: restricted CSDA range below 0.5 mm, so nothing crosses 1 mm or stops beyond 0.5 mm
    n = 100
    W = M.BY_NAME["G4_W"]
    assert S.csda_range(W, 0.5, TCUT, KCUT) < 0.5
    assert np.all(np.diff(S.csda_range(W, np.geomspace(0.002, 100, 50), TCUT, KCUT)) > 0)
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    r = S.electron_slab(np.random.default_rng(23), W, np.full(n, 0.5), d, 1.0, TCUT, KCUT, msc=True)
    assert (r.stopped | r.reflected).all() and r.reflected.any()
    assert (r.position[r.stopped, 2] < 0.5).all()


def test_start_depth_default_is_identical_and_inside_start_exits_either_face():
    n, T0, L = 4000, 105.0, 0.5
    mat = M.BY_NAME["G4_Al"]
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    a = S.electron_slab(np.random.default_rng(7), mat, np.full(n, T0), d, L, TCUT, KCUT, msc=True)
    b = S.electron_slab(np.random.default_rng(7), mat, np.full(n, T0), d, L, TCUT, KCUT, msc=True, z0=0.0)
    assert np.array_equal(a.T, b.T) and np.array_equal(a.position, b.position)
    half = S.electron_slab(np.random.default_rng(8), mat, np.full(n, T0), d, L, TCUT, KCUT, z0=0.5 * L)
    full = S.electron_slab(np.random.default_rng(9), mat, np.full(n, T0), d, 0.5 * L, TCUT, KCUT)
    da, db = T0 - half.T, T0 - full.T
    assert abs(da.mean() - db.mean()) < 4 * np.hypot(da.std(), db.std()) / np.sqrt(n)
    back = S.electron_slab(np.random.default_rng(10), mat, np.full(n, T0), -d, L, TCUT, KCUT, z0=0.25 * L)
    assert back.reflected.all() and np.allclose(back.position[:, 2], 0.0, atol=1e-9)
    with pytest.raises(ValueError):
        S.electron_slab(np.random.default_rng(1), mat, np.full(2, T0), d[:2], L, TCUT, KCUT, z0=2 * L)
