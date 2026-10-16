"""Detector-independent pieces of the straight-line muon transport (from mu2e-crv-fastsim transport.py, f451d7b).

* msc_to_plane: carry tracks to a horizontal plane through a uniform slab, or through height bands of material,
  with correlated (displacement, angle) Highland multiple scattering in two transverse planes (PDG eq. 34.20);
* slab_loss: energy loss through a thick layer integrated in slices so dE/dx follows the slowing muon.

Detector packages own the geometry-specific parts (which plane, which footprint, which world bands).
"""
import numpy as np

from . import eloss


def msc_to_plane(rng, o, d, p, y_plane, x0_mm, x_over_x0=None, band_xx0=None, band_h=None):
    """Move each track from o to the plane y = y_plane through a uniform slab, with PDG (eq. 34.20) correlated MSC.

    Two independent transverse planes per track: y_plane displacement and angle from (z1, z2) unit Gaussians,
    theta_plane = z2 * theta0, y_plane = z1 * L * theta0 / sqrt(12) + z2 * L * theta0 / 2. Returns new (origin,
    direction, path length L). Tracks are assumed downward (d_y < 0).
    """
    L = (o[:, 1] - y_plane) / -d[:, 1]
    th0 = eloss.highland_theta0(p, L / x0_mm if x_over_x0 is None else x_over_x0)
    if band_xx0 is not None:
        # scattering spread over path bands at distance band_h from the plane: exact Gaussian covariance
        w = band_xx0 / np.maximum(band_xx0.sum(1, keepdims=True), 1e-30)
        s_tt = th0 ** 2
        s_yt = s_tt * (w * band_h).sum(1)
        s_yy = s_tt * (w * band_h ** 2).sum(1)
        a = np.sqrt(s_tt)
        b = np.sqrt(np.clip(s_yy - s_yt ** 2 / np.maximum(s_tt, 1e-30), 0, None))
        z_t = rng.standard_normal((len(p), 2))
        z_y = rng.standard_normal((len(p), 2))
        dth = z_t * a[:, None]
        dy = z_t * (s_yt / np.maximum(a, 1e-30))[:, None] + z_y * b[:, None]
        return _apply_msc(o, d, L, dth, dy, y_plane)
    z1 = rng.standard_normal((len(p), 2))
    z2 = rng.standard_normal((len(p), 2))
    dy = z1 * (L * th0 / np.sqrt(12))[:, None] + z2 * (L * th0 / 2)[:, None]
    dth = z2 * th0[:, None]
    return _apply_msc(o, d, L, dth, dy, y_plane)


def _apply_msc(o, d, L, dth, dy, y_plane):
    """Displace and turn each track at the end of its path L (two transverse planes), then put it on y = y_plane."""
    ref = np.where(np.abs(d[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    end = o + d * L[:, None] + u * dy[:, [0]] + v * dy[:, [1]]
    dn = d + u * np.tan(dth[:, [0]]) + v * np.tan(dth[:, [1]])
    dn /= np.linalg.norm(dn, axis=1)[:, None]
    # the transverse shift has a y component: slide back along the new direction onto the plane, so no track starts
    # inside or below the topmost bars
    back = (end[:, 1] - y_plane) / dn[:, 1]
    end = end - dn * back[:, None]
    return end, dn, L - back


N_SLICES_MAX = 40
SLICE_FRAC = 0.05   # max fractional kinetic-energy loss per slice


def slab_loss(rng, mat, p, length):
    """Energy loss [MeV] through `length` [mm] of `mat`, integrated in slices so dE/dx follows the slowing muon.

    Each slice loses at most ~SLICE_FRAC of the current kinetic energy (mean); fluctuations are sampled per slice.
    A muon whose loss reaches its kinetic energy stops (loss = initial kinetic energy).
    """
    p = np.asarray(p, float).copy()
    tk0 = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU
    left = np.asarray(length, float).copy()
    lost = np.zeros_like(p)
    for _ in range(N_SLICES_MAX):
        go = (left > 0) & (lost < tk0)
        if not go.any():
            break
        tk = tk0[go] - lost[go]
        pg = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 1e-6, None))
        step = np.minimum(left[go], SLICE_FRAC * tk / eloss.dedx_restricted(mat, pg, 1e9))
        last = np.flatnonzero(go)
        step = np.where(left[go] <= step * 1.0001, left[go], step)
        lost[last] += eloss.sample_loss(rng, mat, pg, step, 1e9)
        left[last] -= step
    rest = (left > 0) & (lost < tk0)   # still inside after N_SLICES_MAX: finish with one mean step
    if rest.any():
        tk = tk0[rest] - lost[rest]
        pg = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 1e-6, None))
        lost[rest] += eloss.dedx_restricted(mat, pg, 1e9) * left[rest]
    return np.minimum(lost, tk0)
