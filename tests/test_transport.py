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
