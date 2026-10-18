"""Detector-independent pieces of the straight-line muon transport (from mu2e-crv-fastsim transport.py, f451d7b).

* msc_to_plane: carry tracks to a horizontal plane through a uniform slab, or through height bands of material,
  with correlated (displacement, angle) Highland multiple scattering in two transverse planes (PDG eq. 34.20);
* slab_loss: energy loss through a thick layer integrated in slices so dE/dx follows the slowing muon;
* world_loss: material column of the GDML world between a track origin and a horizontal plane, in height bands
  (energy loss per material, x/X0 and lever arm per band for msc_to_plane).

Detector packages own the geometry-specific parts (which plane, which footprint, which world bands).
"""
import numpy as np

from . import eloss, materials
from . import world as world_mod


def msc_to_plane(rng, o, d, p, y_plane, x0_mm, x_over_x0=None, band_xx0=None, band_h=None, band_p=None):
    """Move each track from o to the plane y = y_plane through a uniform slab, with PDG (eq. 34.20) correlated MSC.

    Two independent transverse planes per track: y_plane displacement and angle from (z1, z2) unit Gaussians,
    theta_plane = z2 * theta0, y_plane = z1 * L * theta0 / sqrt(12) + z2 * L * theta0 / 2. Returns new (origin,
    direction, path length L). Tracks are assumed downward (d_y < 0).

    band_p [N, nbands] (sliced scattering): the muon |p| in each band (world_loss returns it). Each band then
    scatters with its own 1/(beta p)^2: theta0^2 = sum_k (13.6/(beta_k p_k))^2 x_k (1 + 0.038 ln(X/beta^2))^2, the
    Highland log term on the total x/X0 (Lynch-Dahl / PDG 34.16 summed over layers), so soft muons that lose most of
    their energy in the concrete scatter most near the plane. Without band_p one |p| (`p`) is used for the column.
    """
    L = (o[:, 1] - y_plane) / -d[:, 1]
    th0 = eloss.highland_theta0(p, L / x0_mm if x_over_x0 is None else x_over_x0)
    if band_xx0 is not None:
        # scattering spread over path bands at distance band_h from the plane: exact Gaussian covariance
        if band_p is not None:
            xt = np.maximum(band_xx0.sum(1), 1e-12)
            bp = np.maximum(band_p, 1.0)
            beta2 = bp ** 2 / (bp ** 2 + eloss.MMU ** 2)
            # per-band variance with that band's momentum; the log term uses the column's total x/X0 and a
            # variance-weighted beta (beta ~ 1 except for the softest muons)
            v_k = (13.6 / (np.sqrt(beta2) * bp)) ** 2 * band_xx0
            beta2_eff = (v_k * beta2).sum(1) / np.maximum(v_k.sum(1), 1e-30)
            corr = (1 + 0.038 * np.log(xt / np.maximum(beta2_eff, 1e-6))) ** 2
            v_k = v_k * corr[:, None]
            w = v_k / np.maximum(v_k.sum(1, keepdims=True), 1e-30)
            th0 = np.sqrt(v_k.sum(1))
        else:
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


def world_loss(rng, o, d, p, world, band_edges_y, return_band_p=False):
    """Material between o and the plane y = band_edges_y[-1] (tracks going down).

    band_edges_y: decreasing heights [mm]; the last is the plane. Returns (|p| at the plane, total x/X0, survived?,
    x/X0 per band, path distance from each band's centre to the plane); with return_band_p=True also the mean of the
    entry and exit |p| of each band [N, nbands] (for sliced scattering in msc_to_plane). Loss per material is integrated in slices
    (slab_loss); materials without constants are treated as concrete. The band lever arms let msc_to_plane place the
    scattering where it happens (material close above the plane scatters with a short arm, distant air with a long one).
    """
    edges = np.asarray(band_edges_y, float)
    y_plane = edges[-1]
    L = (o[:, 1] - y_plane) / -d[:, 1]
    nb = len(edges) - 1
    band_xx0 = np.zeros((len(p), nb))
    band_h = np.zeros((len(p), nb))
    loss = np.zeros(len(p))
    band_p = np.zeros((len(p), nb))
    x0_air = eloss.radiation_length_mm(materials.AIR)
    for k in range(nb):
        y_top, y_bot = edges[k], edges[k + 1]
        loss_in = loss.copy()
        s_top = np.clip((o[:, 1] - y_top) / -d[:, 1], 0, L)
        s_bot = np.clip((o[:, 1] - y_bot) / -d[:, 1], 0, L)
        seg = s_bot - s_top
        if not (seg > 0).any():
            continue
        band_pieces = [q for q in world if q[3][1] >= y_bot and q[2][1] <= y_top]
        act = np.flatnonzero(seg > 0)
        col = {} if not band_pieces else {
            m: np.bincount(act, weights=v, minlength=len(p))
            for m, v in world_mod.column(band_pieces, (o + d * s_top[:, None])[act], d[act], seg[act]).items()}
        solid = np.zeros(len(p))
        for name, c in col.items():
            if not c.any():
                continue
            mat = materials.BY_NAME.get(name, materials.CONCRETE_MARS)
            hit = np.flatnonzero(c > 0)
            loss[hit] += slab_loss(rng, mat, p[hit] - loss[hit], c[hit])
            band_xx0[:, k] += c / eloss.radiation_length_mm(mat)
            solid += c
        air = np.clip(seg - solid, 0, None)
        loss += eloss.dedx_restricted(materials.AIR, np.maximum(p - loss, 1.0), 1e9) * air
        band_xx0[:, k] += air / x0_air
        band_h[:, k] = L - 0.5 * (s_top + s_bot)
        band_p[:, k] = 0.5 * (_p_after(p, loss_in) + _p_after(p, loss))
    tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - loss
    p_plane = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 0, None))
    if return_band_p:
        return p_plane, band_xx0.sum(1), tk > 0.5, band_xx0, band_h, band_p
    return p_plane, band_xx0.sum(1), tk > 0.5, band_xx0, band_h


def _p_after(p, loss):
    """|p| after losing kinetic energy `loss` (MeV), floored at 0."""
    tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - loss
    return np.sqrt(np.clip((np.maximum(tk, 0) + eloss.MMU) ** 2 - eloss.MMU ** 2, 0, None))
