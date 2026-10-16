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
