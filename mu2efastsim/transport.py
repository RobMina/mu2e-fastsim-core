"""Detector-independent pieces of the straight-line muon transport (from mu2e-crv-fastsim transport.py, f451d7b).

* msc_to_plane: carry tracks to a horizontal plane through a uniform slab, or through height bands of material,
  with correlated (displacement, angle) Highland multiple scattering in two transverse planes (PDG eq. 34.20);
* slab_loss: energy loss through a thick layer integrated in slices so dE/dx follows the slowing muon;
* world_loss: material column of the GDML world between a track origin and a horizontal plane, in height bands
  (energy loss per material, x/X0 and lever arm per band for msc_to_plane);
* scattering pieces for detector packages: highland_var_increment (Highland per layer of a stack), chi_c2,
  theta_nuclear, tail_threshold and single_scatter_tail (the Rutherford tail above the Gaussian core).

Detector packages own the geometry-specific parts (which plane, which footprint, which world bands).
"""
import numpy as np

from . import eloss, materials
from . import world as world_mod


def msc_to_plane(rng, o, d, p, y_plane, x0_mm, x_over_x0=None, band_xx0=None, band_h=None, band_p=None,
                 band_chic2=None, band_thn=None, tail_k=None, band_len=None, core_scale=1.0):
    """Move each track from o to the plane y = y_plane through a uniform slab, with PDG (eq. 34.20) correlated MSC.

    Two independent transverse planes per track: y_plane displacement and angle from (z1, z2) unit Gaussians,
    theta_plane = z2 * theta0, y_plane = z1 * L * theta0 / sqrt(12) + z2 * L * theta0 / 2. Returns new (origin,
    direction, path length L). Tracks are assumed downward (d_y < 0).

    band_p [N, nbands] (sliced scattering): the muon |p| in each band (world_loss returns it). Each band then
    scatters with its own 1/(beta p)^2: theta0^2 = sum_k (13.6/(beta_k p_k))^2 x_k (1 + 0.038 ln(X/beta^2))^2, the
    Highland log term on the total x/X0 (Lynch-Dahl / PDG 34.16 summed over layers), so soft muons that lose most of
    their energy in the concrete scatter most near the plane. Without band_p one |p| (`p`) is used for the column.

    band_chic2, band_thn [N, nbands] (world_loss return_tail=True): add the single-scattering tail
    (single_scatter_tail) above the column's Gaussian core, each tail scatter at its band's lever arm. tail_k: the
    tail threshold in units of the column theta0 (None = where the tail density crosses the Gaussian, tail_threshold).

    band_len [N, nbands] (opt-in, world_loss return_band_len=True): path length of each band; each band's scattering is
    then spread uniformly along it (adds w_k len_k^2 / 12 to the displacement variance) instead of sitting at its centre.

    core_scale (opt-in, default 1): factor on the Gaussian core width (angle and displacement). Highland fits the central
    98 % of the full distribution, so with a single-scattering tail added on top the core is counted twice; the slab test
    against Geant4 (trkfastsim slabcheck) fixes the factor.
    """
    L = (o[:, 1] - y_plane) / -d[:, 1]
    th0 = eloss.highland_theta0(p, L / x0_mm if x_over_x0 is None else x_over_x0) * core_scale
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
            th0 = np.sqrt(v_k.sum(1)) * core_scale
        else:
            w = band_xx0 / np.maximum(band_xx0.sum(1, keepdims=True), 1e-30)
        s_tt = th0 ** 2
        s_yt = s_tt * (w * band_h).sum(1)
        s_yy = s_tt * (w * band_h ** 2).sum(1)
        if band_len is not None:   # within-band spread of the scattering points (uniform along the band path)
            s_yy = s_yy + s_tt * (w * band_len ** 2).sum(1) / 12.0
        a = np.sqrt(s_tt)
        b = np.sqrt(np.clip(s_yy - s_yt ** 2 / np.maximum(s_tt, 1e-30), 0, None))
        z_t = rng.standard_normal((len(p), 2))
        z_y = rng.standard_normal((len(p), 2))
        dth = z_t * a[:, None]
        dy = z_t * (s_yt / np.maximum(a, 1e-30))[:, None] + z_y * b[:, None]
        if band_chic2 is not None:
            th_min = tail_threshold(band_chic2.sum(1), th0, tail_k)
            for k in range(band_chic2.shape[1]):
                tt = single_scatter_tail(rng, band_chic2[:, k], th_min, band_thn[:, k])
                dth += tt
                dy += tt * band_h[:, [k]]
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


HBARC_MEV_FM = 197.3269804
R_NUCLEUS_FM = 1.2           # nuclear radius R = 1.2 fm A^(1/3) (form factor of single scattering)
THETA_TAIL_MAX = 1.0         # rad: cap of the single-scattering tail (the 1/theta^3 form is a small-angle approximation)


ME_OVER_MMU = eloss.ME / eloss.MMU     # rad: largest muon deflection on a free electron


def _zz_over_a(mat, electrons):
    from .materials import ELEMENTS
    return sum(w * ELEMENTS[e][0] * (ELEMENTS[e][0] + (1 if electrons else 0)) / ELEMENTS[e][1]
               for e, w in mat.fractions.items())


def chi_c2(mat, length_mm, p, mass=eloss.MMU, electrons=True):
    """Single-scattering scale chi_c^2 [rad^2] of a unit-charge particle through `length_mm` of `mat`.

    chi_c^2 = 0.157 sum_i w_i Z_i (Z_i + 1) / A_i  rho t / (p beta)^2, t in cm, p beta in MeV (Moliere / Lynch-Dahl
    NIM B58 (1991) 6; Z+1 counts scattering on atomic electrons). The mean number of single scatters with space angle
    above theta (theta_core << theta << theta_nuclear) is chi_c^2 / theta^2. electrons=False: Z^2 only, the nuclear
    part, for the tail (a muon is deflected by at most m_e / m_mu = 4.8 mrad on an electron, ME_OVER_MMU).
    """
    p = np.maximum(np.asarray(p, float), 1e-3)
    pb = p * p / np.sqrt(p * p + mass * mass)
    return 0.157 * _zz_over_a(mat, electrons) * mat.rho * 0.1 * np.asarray(length_mm, float) / pb ** 2


def chi_c2_electron(mat, length_mm, p, mass=eloss.MMU):
    """Electron part of chi_c^2 [rad^2]: 0.157 sum_i w_i Z_i / A_i rho t / (p beta)^2 (the Z of Z(Z+1)). A muon is
    deflected by at most ME_OVER_MMU on an electron, so this part ends there (single_scatter_tail theta_max)."""
    return chi_c2(mat, length_mm, p, mass) - chi_c2(mat, length_mm, p, mass, electrons=False)


def theta_nuclear(mat, p):
    """Form-factor angle [rad] of single scattering: theta_n = sqrt(20) hbar c / (p R), R = 1.2 fm A_eff^(1/3).

    The nucleus is an exponential charge distribution with the rms radius of a uniform sphere of radius R (rms^2 =
    3/5 R^2 = 12 a^2), form factor F(q) = (1 + q^2 R^2 / 20)^-2, so F^2 = (1 + theta^2 / theta_n^2)^-4 with q = p theta.
    A_eff^(1/3) is the mass-weighted A^(1/3) of the material."""
    from .materials import ELEMENTS
    a13 = sum(w * ELEMENTS[e][1] ** (1 / 3) for e, w in mat.fractions.items())
    return np.sqrt(20.0) * HBARC_MEV_FM / (np.maximum(np.asarray(p, float), 1e-3) * R_NUCLEUS_FM * a13)


def highland_var_increment(p, x_prev, x_new, mass=eloss.MMU):
    """Plane-projected variance [rad^2] that a layer adds when the traversed x/X0 grows from x_prev to x_new.

    Highland with the log term on the cumulative thickness (Lynch-Dahl): (13.6/(beta p))^2 [g(x_new) - g(x_prev)],
    g(x) = x (1 + 0.038 ln(x / beta^2))^2, so a stack of thin layers sums to the Highland width of the stack instead of
    the (too small) quadrature sum of thin-layer widths. g is increasing for x / beta^2 > 4e-12."""
    p = np.maximum(np.asarray(p, float), 1e-3)
    beta2 = p * p / (p * p + mass * mass)

    def g(x):
        x = np.maximum(np.asarray(x, float), 0.0)
        return np.where(x > 0, x * (1 + 0.038 * np.log(np.maximum(x, 1e-30) / beta2)) ** 2, 0.0)

    return (13.6 ** 2 / (beta2 * p * p)) * np.maximum(g(x_new) - g(x_prev), 0.0)


def tail_threshold(chic2, theta0, k=None):
    """Space angle [rad] above which single scatters are added to a Gaussian core of plane width theta0.

    k given: k * theta0. k None: the larger angle where the single-scattering density 2 chi_c^2 / theta^3 equals the
    Gaussian space-angle density theta / theta0^2 exp(-theta^2 / (2 theta0^2)), i.e. u^4 exp(-u^2/2) = 2 chi_c^2 /
    theta0^2 with u = theta / theta0 (Newton from u = 6; at least u = 2, the maximum of the left side)."""
    th0 = np.maximum(np.asarray(theta0, float), 1e-12)
    if k is not None:
        return k * th0
    r = np.maximum(np.asarray(chic2, float) / th0 ** 2, 1e-300)
    u = np.full(np.shape(r), 6.0)
    for _ in range(30):
        h = 4 * np.log(u) - 0.5 * u * u - np.log(2 * r)
        u = np.maximum(u - h / (4 / u - u), 2.0)
    return u * th0


def single_scatter_tail(rng, chic2, theta_min, theta_n=None, theta_max=THETA_TAIL_MAX):
    """Projected angles [N, 2] of the single scatters above space angle theta_min (screened Rutherford tail).

    Rate per track dN/dtheta = 2 chi_c^2 / theta^3 F^2(theta), F^2 = (1 + theta^2 / theta_n^2)^-4 (theta_nuclear;
    None = no form factor, as for electrons), up to theta_max (scalar or [N]; ME_OVER_MMU for the electron part).
    Sampled by thinning: Poisson(chi_c^2 (1/theta_min^2 - 1/theta_max^2)) proposals from the bare 1/theta^3 spectrum,
    each kept with probability F^2 (exact for a Poisson process); uniform azimuth; the kept scatters of a track add.
    Zero where theta_min >= theta_max."""
    chic2 = np.asarray(chic2, float)
    n = len(chic2)
    out = np.zeros((n, 2))
    a = 1.0 / np.maximum(theta_min, 1e-12) ** 2
    b = np.broadcast_to(1.0 / np.asarray(theta_max, float) ** 2, (n,))
    lam = np.clip(chic2 * (a - b), 0, None)
    k = rng.poisson(lam)
    if not k.any():
        return out
    idx = np.repeat(np.arange(n), k)
    th = 1.0 / np.sqrt(a[idx] - rng.random(len(idx)) * (a[idx] - b[idx]))
    if theta_n is None:
        keep = np.ones(len(idx), bool)
    else:
        tn = np.broadcast_to(np.asarray(theta_n, float), (n,))[idx]
        keep = rng.random(len(idx)) < (1 + (th / tn) ** 2) ** -4
    idx, th = idx[keep], th[keep]
    ph = rng.random(len(idx)) * 2 * np.pi
    np.add.at(out, idx, np.c_[th * np.cos(ph), th * np.sin(ph)])
    return out


N_SLICES_MAX = 200
SLICE_FRAC = 0.05   # max fractional kinetic-energy loss per slice
TK_STOP = 1.0       # MeV: kinetic energy below which a muon is stopped (the remainder is deposited)
MAX_SLICES_PER_SLAB = 4   # slices for a muon with > 3x the energy the remaining slab takes (flat dE/dx regime)


def slab_loss(rng, mat, p, length):
    """Energy loss [MeV] through `length` [mm] of `mat`, integrated in slices so dE/dx follows the slowing muon.

    Each slice loses at most ~SLICE_FRAC of the current kinetic energy (mean); fluctuations are sampled per slice.
    Range-out follows the mean range (CSDA), not a single fluctuation: if a slice's sampled loss would exceed the
    remaining kinetic energy, that slice takes its mean loss instead (the large-loss tail of a slice is delta-ray
    energy that leaves the muon, not a reason to stop it); the muon stops only when its kinetic energy falls below
    TK_STOP. At SLICE_FRAC = 5 % a muon needs about 60-90 slices to range out, inside N_SLICES_MAX; beyond the cap the
    remainder is one mean step, limited to the remaining kinetic energy.
    """
    p = np.asarray(p, float).copy()
    tk0 = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU
    left = np.asarray(length, float).copy()
    lost = np.zeros_like(p)
    for _ in range(N_SLICES_MAX):
        go = (left > 0) & (tk0 - lost > TK_STOP)
        if not go.any():
            break
        tk = tk0[go] - lost[go]
        pg = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 1e-6, None))
        dedx = eloss.dedx_restricted(mat, pg, 1e9)
        # slice by energy: SLICE_FRAC of the kinetic energy, but never finer than needed to cross what is left of the
        # slab in MAX_SLICES_PER_SLAB pieces when the muon has energy to spare (dE/dx is flat above ~1 GeV/c, so a
        # fast muon needs few slices; slow muons are still sliced at SLICE_FRAC down to the stop)
        need = dedx * left[go]
        coarse = tk > 3.0 * need
        step = np.where(coarse, left[go] / MAX_SLICES_PER_SLAB, SLICE_FRAC * tk / dedx)
        step = np.minimum(left[go], np.maximum(step, SLICE_FRAC * tk / dedx))
        last = np.flatnonzero(go)
        step = np.where(left[go] <= step * 1.0001, left[go], step)
        dl = eloss.sample_loss(rng, mat, pg, step, 1e9)
        dl = np.where(dl >= tk, dedx * step, dl)          # tail sample beyond the remaining energy: mean instead
        lost[last] += dl
        left[last] -= step
    rest = (left > 0) & (tk0 - lost > TK_STOP)   # still inside after N_SLICES_MAX: finish with one mean step
    if rest.any():
        tk = tk0[rest] - lost[rest]
        pg = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 1e-6, None))
        lost[rest] += eloss.dedx_restricted(mat, pg, 1e9) * left[rest]
    stopped = tk0 - lost <= TK_STOP
    return np.where(stopped, tk0, np.minimum(lost, tk0))


def world_loss(rng, o, d, p, world, band_edges_y, return_band_p=False, return_tail=False, return_band_len=False):
    """Material between o and the plane y = band_edges_y[-1] (tracks going down).

    band_edges_y: decreasing heights [mm]; the last is the plane. Returns (|p| at the plane, total x/X0, survived?,
    x/X0 per band, path distance from each band's centre to the plane); with return_band_p=True also the mean of the
    entry and exit |p| of each band [N, nbands] (for sliced scattering in msc_to_plane). Loss per material is integrated in slices
    (slab_loss); materials without constants are treated as concrete. The band lever arms let msc_to_plane place the
    scattering where it happens (material close above the plane scatters with a short arm, distant air with a long one).
    return_tail=True appends the single-scattering inputs per band: nuclear chi_c^2 [N, nbands] at the band's mean |p| and the
    chi_c^2-weighted form-factor angle theta_nuclear (msc_to_plane band_chic2, band_thn).
    return_band_len=True appends the path length in each band [N, nbands] (msc_to_plane band_len), after the tail outputs.
    """
    edges = np.asarray(band_edges_y, float)
    y_plane = edges[-1]
    L = (o[:, 1] - y_plane) / -d[:, 1]
    nb = len(edges) - 1
    band_xx0 = np.zeros((len(p), nb))
    band_h = np.zeros((len(p), nb))
    loss = np.zeros(len(p))
    band_p = np.zeros((len(p), nb))
    band_chic2 = np.zeros((len(p), nb))
    band_thw = np.zeros((len(p), nb))
    band_len = np.zeros((len(p), nb))
    x0_air = eloss.radiation_length_mm(materials.AIR)
    for k in range(nb):
        y_top, y_bot = edges[k], edges[k + 1]
        loss_in = loss.copy()
        s_top = np.clip((o[:, 1] - y_top) / -d[:, 1], 0, L)
        s_bot = np.clip((o[:, 1] - y_bot) / -d[:, 1], 0, L)
        seg = s_bot - s_top
        if not (seg > 0).any():
            continue
        band_pieces = world.band(k) if hasattr(world, "band") else [q for q in world if q[3][1] >= y_bot and q[2][1] <= y_top]
        act = np.flatnonzero(seg > 0)
        col = {} if not len(band_pieces) else {
            m: np.bincount(act, weights=v, minlength=len(p))
            for m, v in world_mod.column(band_pieces, (o + d * s_top[:, None])[act], d[act], seg[act]).items()}
        solid = np.zeros(len(p))
        cols = []
        for name, c in col.items():
            if not c.any():
                continue
            mat = materials.BY_NAME.get(name, materials.CONCRETE_MARS)
            hit = np.flatnonzero(c > 0)
            loss[hit] += slab_loss(rng, mat, p[hit] - loss[hit], c[hit])
            band_xx0[:, k] += c / eloss.radiation_length_mm(mat)
            solid += c
            cols.append((mat, c))
        air = np.clip(seg - solid, 0, None)
        loss += eloss.dedx_restricted(materials.AIR, np.maximum(p - loss, 1.0), 1e9) * air
        band_xx0[:, k] += air / x0_air
        band_h[:, k] = L - 0.5 * (s_top + s_bot)
        band_len[:, k] = seg
        band_p[:, k] = 0.5 * (_p_after(p, loss_in) + _p_after(p, loss))
        if return_tail:
            pm = np.maximum(band_p[:, k], 1.0)
            for mat, c in cols + [(materials.AIR, air)]:
                x2 = chi_c2(mat, c, pm, electrons=False)
                band_chic2[:, k] += x2
                band_thw[:, k] += x2 * theta_nuclear(mat, pm)
    tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - loss
    p_plane = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 0, None))
    out = (p_plane, band_xx0.sum(1), tk > 0.5, band_xx0, band_h)
    if return_band_p:
        out += (band_p,)
    if return_tail:
        out += (band_chic2, np.where(band_chic2 > 0, band_thw / np.maximum(band_chic2, 1e-300), 1.0))
    if return_band_len:
        out += (band_len,)
    return out


def _p_after(p, loss):
    """|p| after losing kinetic energy `loss` (MeV), floored at 0."""
    tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - loss
    return np.sqrt(np.clip((np.maximum(tk, 0) + eloss.MMU) ** 2 - eloss.MMU ** 2, 0, None))


class BandedWorld:
    """World pieces pre-split into the height bands of world_loss, each band packed as a world.PieceSet."""

    def __init__(self, pieces, band_edges_y, cell=500.0):
        self._pieces, self._cell = pieces, cell
        self.edges = np.asarray(band_edges_y, float)
        self.bands = [world_mod.PieceSet([q for q in pieces if q[3][1] >= self.edges[k + 1] and q[2][1] <= self.edges[k]])
                      .build_grid(cell) for k in range(len(self.edges) - 1)]

    def band(self, k):
        return self.bands[k]

    def pieces(self):
        """All pieces as one PieceSet (for whole-column queries outside the band loop)."""
        if not hasattr(self, "_all"):
            self._all = world_mod.PieceSet(self._pieces).build_grid(self._cell)
        return self._all
