"""JAX twin of eslab.electron_slab (e-/e+ through a slab with msc, annihilation and optional field and compaction).

Same physics and step logic as eslab; how the NumPy constructs map to JAX:
  * every cross section, stopping power, range, soft/hard msc rate and u_c is a host table on a uniform ln T grid
    (build_tables, float64 from the NumPy models), interpolated linearly in ln T in the kernel;
  * the Seltzer-Berger photon energy comes from an inverse-CDF table in x = ln(k / kcut) / ln(T / kcut) per ln T node,
    its nodes crowded towards the hard end
    (the node picked with the linear-interpolation weight) instead of rejection sampling;
  * rejection samplers (Moller/Bhabha transfer, 2BS photon angle) run a fixed number of rounds;
  * the step loop is a lax.while_loop over all tracks, finished tracks masked; photons, deltas and annihilation photons
    are returned as per-track counts and energy sums, not as lists (lane_step_photons optionally records each step's
    bremsstrahlung emission only; lane_step_emissions also records annihilation births, without photon transport);
  * hard elastic collisions need no thinning: the energy does not change between loss points, so the rate at the
    pre-step energy is exact; annihilation is thinned against its value at half the pre-step energy as in eslab.
The kernel computes in the dtype of T (shared kernel, mu2efastsim.precision.float_of): float64 from host code,
the device float inside a dprec.jit stage. Energies in MeV, lengths in mm.
"""
from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from . import eannihil, ebrem, eioni, eloss_jax, emsc, eslab
from . import precision as dprec
from .eloss import ME

NT = 513                      # ln T nodes from eslab.T_STOP to 1 GeV
NX = 257                      # inverse-CDF nodes of the brem photon energy
REJECT_ROUNDS = 24            # fixed rounds of the rejection samplers
C_POWER = 3.0                 # brem CDF nodes C = 1 - (1 - u)^C_POWER, u uniform on [0, 1]
C_NODES = 1.0 - (1.0 - np.linspace(0.0, 1.0, NX)) ** C_POWER
_CACHE = {}
SCAN_UNROLL = 4               # with unroll: sub-arcs of _to_plane taken before its scan loop
UNROLL = 8                    # with unroll=True: loop iterations per trip of the fixed-trip loops


def _unroll(on):
    """fori_loop keyword for unroll: off (False / 0), True (UNROLL per trip) or an integer factor."""
    return {"unroll": UNROLL if on is True else int(on)} if on else {}


class Tables(NamedTuple):
    lnt0: object
    dlnt: object
    sig_b: object     # (NT,) brem above kcut [mm-1]
    sig_d: object     # (NT,) delta above tcut
    dedx: object      # (NT,) restricted continuous loss [MeV/mm]
    sig_a: object     # (NT,) annihilation in flight
    lam1: object      # (NT,) first transport mean free path [mm]
    rng: object       # (NT,) restricted CSDA range [mm]
    uc: object        # (NT,) soft/hard boundary for the slab's hard path
    soft: object      # (NT,) soft transport below uc, with Mott
    soft_all: object  # (NT,) all collisions as soft (range-trapped tracks)
    hard: object      # (NT,) hard sampling (envelope) rate above uc
    hard_cum: object  # (2 nel, NT) cumulative target fractions: nuclei of each element, then their electrons
    scr: object       # (nel, NT) screening s
    ue: object        # (NT,) largest u on atomic electrons
    env: object       # (nel, NT) Mott envelope
    brem_cum: object  # (nel, NT) cumulative element fractions of brem
    brem_x: object    # (NT, NX) inverse CDF: x at the CDF values C_NODES
    z: object         # (nel,)
    ffa: object       # (nel,) nuclear form factor constant FORM_FACTOR A^0.54 [MeV-2]
    mott: object      # (nel, 5, 6) Mott fit coefficients
    ff_tmin: object   # nuclear form factor on hard collisions above this T (emsc.FF_TMIN)


def _grid():
    lnt = np.linspace(np.log(eslab.T_STOP), np.log(ebrem.T_MAX), NT)
    return lnt, np.clip(np.exp(lnt), eslab.T_STOP, ebrem.T_MAX)


def build_tables(mat, tcut, kcut, positron=False, max_hard_path=np.inf):
    """Host float64 tables of `mat` for the cuts, particle and the msc hard path (eslab: max_step / hard_per_step)."""
    key = (mat.name, float(tcut), float(kcut), bool(positron), float(max_hard_path), emsc.MOTT, float(emsc.FF_TMIN))
    if key in _CACHE:
        return _CACHE[key]
    lnt, T = _grid()
    sb, sd = eslab._sigmas(mat, T, tcut, kcut, positron)
    dedx = eioni.dedx_restricted(mat, T, tcut, positron) + ebrem.dedx_restricted(mat, T, kcut, positron)
    sa = eannihil.xsec_per_mm(mat, T) if positron else np.zeros_like(T)
    lam1 = emsc.lambda1(mat, T, tcut, positron)
    rng = eslab.csda_range(mat, T, tcut, kcut, positron)
    uc = emsc.u_cut(mat, T, tcut, positron, max_hard_path)
    soft = emsc.soft_transport_xsec(mat, T, uc, tcut, positron)
    soft_all = emsc.soft_transport_xsec(mat, T, emsc.U_MAX, tcut, positron)
    nuc, ele, s, ue = emsc._hard_rates(mat, T, uc, tcut, positron)
    rates = np.concatenate([nuc, ele])
    hard = rates.sum(axis=0)
    hard_cum = np.cumsum(rates, axis=0) / np.where(hard > 0, hard, 1.0)
    z, n, a = emsc.elements(mat)
    env = emsc.mott_envelope(z, T, positron)
    on = T > kcut
    part = np.zeros((z.size, T.size))
    if on.any():
        part[:, on] = ebrem._partial_xsec(mat, T[on], kcut, positron)
    tot = part.sum(axis=0)
    brem_cum = np.cumsum(part, axis=0) / np.where(tot > 0, tot, 1.0)
    brem_x = _brem_inverse_cdf(mat, T, kcut, positron)
    mott = emsc.mott_coefficients()[z.astype(int)]
    out = Tables(lnt[0], lnt[1] - lnt[0], sb, sd, dedx, sa, lam1, rng, uc, soft, soft_all, hard, hard_cum, s, ue, env,
                 brem_cum, brem_x, z, emsc.FORM_FACTOR * a ** 0.54, mott, float(emsc.FF_TMIN))
    _CACHE[key] = out
    return out


def _brem_inverse_cdf(mat, T, kcut, positron):
    """x at the CDF values C_NODES for each T node: photon density in x = ln(k / kcut) / ln(T / kcut) is proportional
    to k dsigma/dk with the Ter-Mikaelian factor (ebrem._kdsdk_material). C_NODES crowd towards 1, where the
    Seltzer-Berger density falls steeply at k -> T."""
    out = np.tile(C_NODES, (T.size, 1))
    on = np.flatnonzero(T > kcut * 1.0001)
    if on.size == 0:
        return out
    xf = 1.0 - (1.0 - np.linspace(0.0, 1.0, 8 * NX)) ** 3
    Tj = T[on][:, None]
    k = kcut * (Tj / kcut) ** xf[None, :]
    dens = ebrem._kdsdk_material(mat, k.ravel(), np.broadcast_to(Tj, k.shape).ravel(), positron).reshape(k.shape)
    cdf = np.concatenate([np.zeros((on.size, 1)), np.cumsum(0.5 * (dens[:, 1:] + dens[:, :-1]) * np.diff(xf),
                                                              axis=1)], axis=1)
    cdf /= np.where(cdf[:, -1:] > 0, cdf[:, -1:], 1.0)
    for r, j in enumerate(on):
        out[j] = np.interp(C_NODES, cdf[r], xf)
    return out


def _interp(tab, lnt0, dlnt, T):
    """Linear interpolation in ln T of a (NT,) or (m, NT) table at T (clamped to the grid)."""
    f = jnp.clip((jnp.log(jnp.maximum(T, 1e-30)) - lnt0) / dlnt, 0.0, NT - 1.000001)
    i = f.astype(jnp.int32)
    w = f - i
    return tab[..., i] * (1 - w) + tab[..., i + 1] * w


def _rotate_uz(d, cost, phi):
    """Directions at polar cos `cost` and azimuth phi about the unit vectors d (CLHEP rotateUz)."""
    sint = jnp.sqrt(jnp.clip((1 - cost) * (1 + cost), 0.0, None))
    x, y, zl = sint * jnp.cos(phi), sint * jnp.sin(phi), cost
    u1, u2, u3 = d[:, 0], d[:, 1], d[:, 2]
    up = u1 * u1 + u2 * u2
    s = jnp.sqrt(jnp.where(up > 0, up, 1.0))
    o0 = jnp.where(up > 0, (u1 * u3 * x - u2 * y) / s + u1 * zl, jnp.where(u3 > 0, x, -x))
    o1 = jnp.where(up > 0, (u2 * u3 * x + u1 * y) / s + u2 * zl, jnp.where(u3 > 0, y, -y))
    o2 = jnp.where(up > 0, -s * x + u3 * zl, jnp.where(u3 > 0, zl, -zl))
    o = jnp.stack([o0, o1, o2], axis=1)
    return o / jnp.linalg.norm(o, axis=1, keepdims=True)


K_FIELD = 0.299792458         # MeV/c per (T mm), as ehelix.K


def _hframe(u, B):
    """(b_hat, |B|, u_par, u_perp, w = u_perp x b_hat) per row, b_hat = z where B = 0 (ehelix._frame)."""
    bm = jnp.linalg.norm(B, axis=1)
    pos = bm > 0
    bh = jnp.where(pos[:, None], B / jnp.where(pos, bm, 1.0)[:, None], jnp.array([0.0, 0.0, 1.0], B.dtype))
    upar = (u * bh).sum(axis=1)
    uperp = u - upar[:, None] * bh
    return bh, bm, upar, uperp, jnp.cross(uperp, bh)


def _sc(ks):
    """(sin(ks)/ks, (1 - cos(ks))/ks) with series near 0."""
    small = jnp.abs(ks) < 1e-6
    k2 = ks * ks
    ksn = jnp.where(small, 1.0, ks)
    return (jnp.where(small, 1 - k2 / 6, jnp.sin(ksn) / ksn),
            jnp.where(small, ks / 2 - ks * k2 / 24, (1 - jnp.cos(ksn)) / ksn))


def _helix(x, u, p, q, B, s):
    """ehelix.helix: exact helix over path s in the per-row uniform field B."""
    bh, bm, upar, uperp, w = _hframe(u, B)
    ks = K_FIELD * q * bm / p * s
    s1, c1 = _sc(ks)
    xn = x + (upar * s)[:, None] * bh + (s * s1)[:, None] * uperp + (s * c1)[:, None] * w
    un = upar[:, None] * bh + jnp.cos(ks)[:, None] * uperp + jnp.sin(ks)[:, None] * w
    return xn, un / jnp.linalg.norm(un, axis=1, keepdims=True)


def _to_plane(x, u, p, q, B, zp, smax, unroll=False):
    """Scan sub-arcs of at most 0.5 rad for the first s in (0, smax] reaching z = zp, inf where none does. unroll
    (opt-in): the first SCAN_UNROLL sub-arcs and the bisection are unrolled (same arithmetic, fewer device loops)."""
    bh, bm, upar, uperp, w = _hframe(u, B)
    k = K_FIELD * q * bm / p
    smax = jnp.where(jnp.isfinite(smax), smax, 0.0)

    def z_of(s):
        s1, c1 = _sc(k * s)
        return x[:, 2] + upar * s * bh[:, 2] + s * s1 * uperp[:, 2] + s * c1 * w[:, 2] - zp

    def dz_of(s):
        return upar * bh[:, 2] + jnp.cos(k * s) * uperp[:, 2] + jnp.sin(k * s) * w[:, 2]

    nseg = jnp.maximum(jnp.ceil(jnp.abs(k) * smax / 0.5), 1)
    lo = jnp.full_like(smax, 1e-12)
    f0 = z_of(lo)

    def scanning(c):
        j, _, _, _, found = c
        return jnp.any((~found) & (j <= nseg))

    def scan(c):
        j, lo, f0, hi, found = c
        on = (~found) & (j <= nseg)
        sj = smax * j / nseg
        fj = z_of(sj)
        cross = on & ((jnp.sign(fj) != jnp.sign(f0)) | (fj == 0)) & (f0 != 0)
        return (j + 1, jnp.where(on & ~cross, sj, lo), jnp.where(on & ~cross, fj, f0), jnp.where(cross, sj, hi),
                found | cross)
    c = (jnp.int32(1), lo, f0, lo, jnp.zeros(smax.shape, bool))
    for _ in range(SCAN_UNROLL if unroll else 0):
        c = scan(c)
    _, lo, f0, hi, found = lax.while_loop(scanning, scan, c)
    hi = jnp.where(found, hi, lo)

    def bis(_, c):
        lo, flo, hi = c
        mid = 0.5 * (lo + hi)
        fm = z_of(mid)
        same = jnp.sign(fm) == jnp.sign(flo)
        return jnp.where(same, mid, lo), jnp.where(same, fm, flo), jnp.where(same, hi, mid)
    lo, _, hi = lax.fori_loop(0, 40, bis, (lo, z_of(lo), hi), **_unroll(unroll))
    s = 0.5 * (lo + hi)
    for _ in range(2):
        d = dz_of(s)
        big = jnp.abs(d) > 1e-12
        s = jnp.where(big, s - z_of(s) / jnp.where(big, d, 1.0), s)
    return jnp.where(found, jnp.clip(s, 0.0, smax), jnp.inf)


helix = _helix            # device helix step and plane crossing for callers' own loops (ehelix twins)
to_plane = _to_plane


def _pick(cum, u):
    """Row index of cumulative fractions cum (m, n) for uniforms u (n,)."""
    return jnp.minimum((u[None, :] > cum).sum(axis=0), cum.shape[0] - 1)


def _sample_delta(key, T, tcut, positron, dt, unroll=False):
    """G4MollerBhabhaModel transfer with REJECT_ROUNDS rejection rounds (the last trial if none is accepted)."""
    tmax = T if positron else 0.5 * T
    ok = tcut < tmax
    Ts = jnp.where(ok, T, 4 * tcut + 1.0)
    tmax = Ts if positron else 0.5 * Ts
    xmin, xmax = tcut / Ts, tmax / Ts
    gam = (Ts + ME) / ME
    beta2 = 1.0 - 1.0 / (gam * gam)
    if positron:
        y = 1.0 / (1.0 + gam)
        y2 = y * y
        y12 = 1.0 - 2.0 * y
        b1, b2 = 2.0 - y2, y12 * (3.0 + y2)
        b4 = y12 ** 3
        b3 = b4 + y12 * y12
        yy = xmax * xmax
        grej = 1.0 + (yy * yy * b4 - xmin ** 3 * b3 + yy * b2 - xmin * b1) * beta2

        def zfun(x):
            xx = x * x
            return 1.0 + (xx * xx * b4 - x * xx * b3 + xx * b2 - x * b1) * beta2
    else:
        gg = (2.0 * gam - 1.0) / (gam * gam)
        yv = 1.0 - xmax
        grej = 1.0 - gg * xmax + xmax * xmax * (1.0 - gg + (1.0 - gg * yv) / (yv * yv))

        def zfun(x):
            yv = 1.0 - x
            return 1.0 - gg * x + x * x * (1.0 - gg + (1.0 - gg * yv) / (yv * yv))

    def body(i, carry):
        val, done = carry
        u = jax.random.uniform(jax.random.fold_in(key, i), (2,) + T.shape, dt)
        x = xmin * xmax / (xmin * (1.0 - u[0]) + xmax * u[0])
        acc = grej * u[1] <= zfun(x)
        return jnp.where(done, val, x), done | acc

    x, _ = lax.fori_loop(0, REJECT_ROUNDS, body, (xmin, jnp.zeros(T.shape, bool)), **_unroll(unroll))
    return x * Ts


def _cos_2bs(key, T, k, z, dt, unroll=False):
    """G4Generator2BS photon polar cosine with REJECT_ROUNDS rounds (ebrem.sample_cos_theta)."""
    energy = T + ME
    ratio = (energy - k) / energy
    ratio1 = (1 + ratio) ** 2
    ratio2 = 1 + ratio * ratio
    gam = energy / ME
    beta = jnp.sqrt((gam - 1) * (gam + 1)) / gam
    fz = 0.00008116224 * jnp.cbrt(z) * jnp.cbrt(z + 1)
    ymax = 2 * beta * (1 + beta) * gam * gam

    def rej(y):
        y2 = (1 + y) ** 2
        x = 4 * y * ratio / y2
        return 4 * x - ratio1 - (ratio2 - x) * jnp.log(fz / y2)

    gmax = jnp.maximum(rej(jnp.zeros_like(T)), rej(ymax))

    def body(i, carry):
        val, done = carry
        u = jax.random.uniform(jax.random.fold_in(key, i), (2,) + T.shape, dt)
        yy = u[0] * ymax / (1 + ymax * (1 - u[0]))
        acc = (u[1] * gmax <= rej(yy)) & (yy <= ymax)
        return jnp.where(done, val, yy), done | acc

    y, _ = lax.fori_loop(0, REJECT_ROUNDS, body, (jnp.zeros_like(T), jnp.zeros(T.shape, bool)), **_unroll(unroll))
    return 1 - 2 * y / ymax


def _mott(z, u, beta, coef, positron, mott):
    if mott == "none":
        return jnp.ones_like(u)
    if mott == "table" and not positron:
        b0 = beta - emsc.MOTT_BETA_SHIFT
        bk = jnp.stack([b0 ** k for k in range(6)], axis=-1)
        aj = (coef * bk[:, None, :]).sum(axis=-1)
        su = jnp.sqrt(u)
        return jnp.maximum(sum(aj[:, j] * su ** j for j in range(5)), 0.0)
    sign = -1.0 if positron else 1.0
    b2 = beta * beta
    return 1 - 0.5 * b2 * u + sign * 0.5 * jnp.pi * emsc.ALPHA * z * jnp.sqrt(0.5 * b2 * u) * (2 - u)


class SlabOut(NamedTuple):
    T: object
    direction: object
    position: object
    path: object
    continuous: object
    stopped: object
    reflected: object
    annihilated: object
    n_photons: object
    photon_energy: object
    n_deltas: object
    delta_energy: object
    n_annihilation: object
    annihilation_energy: object
    steps: object        # loop iterations used (all tracks)


class BremsPhotonRecord(NamedTuple):
    """At most one bremsstrahlung emission per lane per step, not a propagated photon or flux-exit record.

    energy (n,) [MeV], position (n, 3) [mm] and direction (n, 3) are at emission in the slab frame;
    path (n,) [mm] is the parent's accumulated path at emission. valid (n,) marks emissions; all other
    entries are zero (including direction). Excludes annihilation photons, delta cascades and showers.
    """
    energy: object
    position: object
    direction: object
    path: object
    valid: object


class EMPhotonRecord(NamedTuple):
    """Fixed three-slot photon births per parent, not photon transport or exit flux.

    Slot 0 is bremsstrahlung; slots 1 and 2 are an annihilation pair. energy (n, 3) is in MeV,
    position and direction (n, 3, 3) are in the slab frame (position in mm), path (n, 3) is the
    parent's accumulated path in mm. valid (n, 3) is boolean; mechanism (n, 3) is int32:
    1 = bremsstrahlung, 2 = in-flight annihilation, 3 = at-rest annihilation. Every invalid value
    is zero, including direction and mechanism. Floating fields follow the parent's T dtype.
    No delta cascades or showers are recorded.
    Annihilation directions inherit eannihil's isotropic COM boost / isotropic back-to-back rest
    convention; this interface does not constitute new Geant4 validation of that convention.
    """
    energy: object
    position: object
    direction: object
    path: object
    valid: object
    mechanism: object


def _annihilation_pair(ks, T, d, flight):
    """eannihil's inherited two-photon convention, using only the four supplied secondary keys."""
    n, dt = T.shape[0], T.dtype

    def isotropic(kc, kp):
        c = 2 * jax.random.uniform(kc, (n,), dt) - 1
        s = jnp.sqrt(jnp.maximum((1 - c) * (1 + c), 0.0))
        phi = 2 * jnp.pi * jax.random.uniform(kp, (n,), dt)
        return jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), c], axis=1)

    Ts = jnp.where(flight, T, 1.0)
    ds = jnp.where(flight[:, None], d, jnp.array([0.0, 0.0, 1.0], dt))
    etot = Ts + 2 * ME
    p = jnp.sqrt(Ts * (Ts + 2 * ME))
    half = 0.5 * jnp.sqrt(etot * etot - p * p)
    u = isotropic(ks[0], ks[1])
    beta, gam = p / etot, etot / (2 * half)
    e1 = gam * half * (1 + beta * u[:, 2])
    x, y, z = half * u[:, 0], half * u[:, 1], gam * half * (u[:, 2] + beta)
    u1, u2, u3 = ds[:, 0], ds[:, 1], ds[:, 2]
    up = u1 * u1 + u2 * u2
    s = jnp.sqrt(jnp.where(up > 0, up, 1.0))
    k1 = jnp.stack([jnp.where(up > 0, (u1 * u3 * x - u2 * y) / s + u1 * z, jnp.where(u3 > 0, x, -x)),
                    jnp.where(up > 0, (u2 * u3 * x + u1 * y) / s + u2 * z, jnp.where(u3 > 0, y, -y)),
                    jnp.where(up > 0, -s * x + u3 * z, jnp.where(u3 > 0, z, -z))], axis=1)
    k2 = p[:, None] * ds - k1
    e2 = etot - e1
    dirs = jnp.stack([k1 / e1[:, None], k2 / jnp.linalg.norm(k2, axis=1, keepdims=True)], axis=1)
    rest = isotropic(ks[2], ks[3])
    return (jnp.where(flight[:, None], jnp.stack([e1, e2], axis=1), jnp.full((n, 2), ME, dt)),
            jnp.where(flight[:, None, None], dirs, jnp.stack([rest, -rest], axis=1)))


def _emission_record(energy, x, direction, path, valid, mechanism):
    """Mask every field, including inactive directions, to exact zeros."""
    return EMPhotonRecord(jnp.where(valid, energy, 0.0), jnp.where(valid[:, :, None], x[:, None, :], 0.0),
                          jnp.where(valid[:, :, None], direction, 0.0), jnp.where(valid, path[:, None], 0.0),
                          valid, jnp.where(valid, mechanism, 0).astype(jnp.int32))


def initial_annihilation_photons(key, st):
    """Initial stopped-positron births as an EMPhotonRecord; consume once, immediately after lane_state.

    Selects only stopped lanes with nan == 2, ean == 2*ME and ~annih, as initialized when positron
    annihilation is enabled. Slot 0 is invalid; slots 1 and 2 are isotropic back-to-back ME photons
    at st['x']/st['path']. This stateless helper neither advances nor marks the state: the caller must
    consume it once, not after steps or on each navigation iteration. lane_step_emissions never re-emits
    already-stopped lanes. Directions inherit eannihil's convention, not new Geant4 validation.
    """
    dt = dprec.float_of(st["T"])
    n = st["T"].shape[0]
    on = st["stopped"] & (st["nan"] == 2) & (st["ean"] == jnp.asarray(2 * ME, dt)) & ~st["annih"]
    zero = jnp.zeros(n, dt)
    ks = jax.random.split(jax.random.fold_in(key, st["it"]), 24)[20:24]
    energy, direction = _annihilation_pair(ks, zero, st["d"], jnp.zeros(n, bool))
    valid = jnp.stack([jnp.zeros(n, bool), on, on], axis=1)
    return _emission_record(jnp.concatenate([zero[:, None], energy], axis=1), st["x"],
                            jnp.concatenate([jnp.zeros((n, 1, 3), dt), direction], axis=1), st["path"], valid,
                            jnp.broadcast_to(jnp.array([0, 3, 3], jnp.int32), (n, 3)))


def _slab_body(key, tab, mc, I, n, dt, L, tcut, kcut, dsmax, positron, msc, anni, mott, B, unroll=False,
               return_photons=False, return_emissions=False, *, first_exit=None, finite_lanes=None):
    """One step of the slab loop as a function of the state dict (used by _run and by callers that embed the slab
    step in their own loop, e.g. with per-lane materials: tab / I from lane_tables). B: per-track fields or None.
    unroll (opt-in): the fixed-trip loops inside the step are unrolled (same draws, to rounding).
    return_photons is a static Python flag: False returns the existing state only; True returns
    (next_state, BremsPhotonRecord), using the same draws and pre-recoil photon direction. No extra RNG draws.
    return_emissions adds annihilation births as EMPhotonRecord, drawing only from unused ks[20:24]."""
    field = B is not None
    nel = tab.z.shape[0]
    if first_exit is not None:
        finite_lanes = jnp.asarray(True if finite_lanes is None else finite_lanes, bool)

        def extra_face(xx, uu, pm, q, smax):
            return jnp.where(finite_lanes, first_exit(xx, uu, pm, q, B, smax), jnp.inf)

    def to_face(z, dz):
        return jnp.where(dz > 0, (L - z) / jnp.where(dz > 0, dz, 1.0), jnp.where(dz < 0, -z / jnp.where(dz < 0, dz, -1.0),
                                                                                 jnp.inf))

    def expo(k):
        return -jnp.log(jax.random.uniform(k, (n,), dt, minval=dprec.tiny_of(dt)))

    def body(st):
        live = ~st["stopped"] & ~st["exited"]
        ks = jax.random.split(jax.random.fold_in(key, st["it"]), 24)
        T = jnp.where(live, st["T"], 1.0)
        sb, sd = I(tab.sig_b, T), I(tab.sig_d, T)
        sig = sb + sd
        free = jnp.where(sig > 0, expo(ks[0]) / jnp.where(sig > 0, sig, 1.0), jnp.inf)
        dedx = I(tab.dedx, T)
        rem = jnp.maximum(eslab.STEP_FRACTION * T / jnp.where(dedx > 0, dedx, jnp.inf), eslab.FINAL_RANGE) - st["pend"]
        if anni:
            sa_max = I(tab.sig_a, eslab.SIGMA_MAX_FRACTION * T)
            free_a = jnp.where(sa_max > 0, expo(ks[1]) / jnp.where(sa_max > 0, sa_max, 1.0), jnp.inf)
        else:
            free_a = jnp.full(n, jnp.inf, dt)
        x, d = st["x"], st["d"]
        if msc:
            zl = x[:, 2]
            trapped = I(tab.rng, T) < jnp.minimum(zl, L - zl)
            if first_exit is not None:
                trapped = trapped & ~finite_lanes
            mcap = jnp.where(trapped, jnp.inf, jnp.minimum(jnp.maximum(emsc.STEP_LAMBDA1 * I(tab.lam1, T),
                                                                        eslab.MSC_MIN_STEP), dsmax))
            sh = jnp.where(trapped, 0.0, I(tab.hard, T))
            free_h = jnp.where(sh > 0, expo(ks[2]) / jnp.where(sh > 0, sh, 1.0), jnp.inf)
        else:
            mcap = free_h = jnp.full(n, jnp.inf, dt)
        limit = jnp.minimum(rem, mcap)
        plan = jnp.minimum(jnp.minimum(free, limit), jnp.minimum(free_a, free_h))
        if field or first_exit is not None:
            pm = jnp.sqrt(T * (T + 2 * ME))
            q = 1.0 if positron else -1.0
        if field:

            def hface(xx, uu, smax):
                g = jnp.minimum(_to_plane(xx, uu, pm, q, B, L, smax, unroll),
                                _to_plane(xx, uu, pm, q, B, 0.0, smax, unroll))
                if first_exit is not None:
                    g = jnp.minimum(g, extra_face(xx, uu, pm, q, smax))
                return g
        if msc and field:
            tau = jax.random.uniform(ks[3], (n,), dt)
            leg1 = tau * plan
            g1 = hface(x, d, leg1)
            out1 = leg1 >= g1
            p1, d1 = _helix(x, d, pm, q, B, jnp.where(out1, g1, leg1))
            tmid = jnp.maximum(T - dedx * (st["pend"] + 0.5 * plan), 0.5 * T)
            s1 = jnp.where(trapped, I(tab.soft_all, tmid), I(tab.soft, tmid))
            m1 = -jnp.expm1(-plan * s1)
            u = jnp.minimum(expo(ks[4]) * m1, emsc.U_MAX)
            dn = jnp.where(out1[:, None], d1, _rotate_uz(d1, 1 - u, 2 * jnp.pi * jax.random.uniform(ks[5], (n,), dt)))
            leg2 = plan - leg1
            g2 = hface(p1, dn, leg2)
            out2 = ~out1 & (leg2 >= g2)
            step = jnp.where(out1, g1, jnp.where(out2, leg1 + g2, plan))
            xn, dn = _helix(p1, dn, pm, q, B, jnp.where(out1, 0.0, jnp.where(out2, g2, leg2)))
            left = out1 | out2
            other = jnp.minimum(jnp.minimum(free, rem), free_a)
            sub = (free_h < jnp.minimum(jnp.minimum(free, limit), free_a)) | (mcap < jnp.minimum(other, free_h))
            apply = left | ~sub
        elif msc:
            tau = jax.random.uniform(ks[3], (n,), dt)
            leg1 = tau * plan
            g1 = to_face(x[:, 2], d[:, 2])
            if first_exit is not None:
                g1 = jnp.minimum(g1, extra_face(x, d, pm, q, leg1))
            out1 = leg1 >= g1
            p1 = x + jnp.where(out1, g1, leg1)[:, None] * d
            tmid = jnp.maximum(T - dedx * (st["pend"] + 0.5 * plan), 0.5 * T)
            s1 = jnp.where(trapped, I(tab.soft_all, tmid), I(tab.soft, tmid))
            m1 = -jnp.expm1(-plan * s1)
            u = jnp.minimum(expo(ks[4]) * m1, emsc.U_MAX)
            dn = jnp.where(out1[:, None], d, _rotate_uz(d, 1 - u, 2 * jnp.pi * jax.random.uniform(ks[5], (n,), dt)))
            leg2 = plan - leg1
            g2 = to_face(p1[:, 2], dn[:, 2])
            if first_exit is not None:
                g2 = jnp.minimum(g2, extra_face(p1, dn, pm, q, leg2))
            out2 = ~out1 & (leg2 >= g2)
            step = jnp.where(out1, g1, jnp.where(out2, leg1 + g2, plan))
            xn = p1 + jnp.where(out1, 0.0, jnp.where(out2, g2, leg2))[:, None] * dn
            left = out1 | out2
            other = jnp.minimum(jnp.minimum(free, rem), free_a)
            sub = (free_h < jnp.minimum(jnp.minimum(free, limit), free_a)) | (mcap < jnp.minimum(other, free_h))
            apply = left | ~sub
        elif field:
            g = hface(x, d, plan)
            left = g <= plan
            step = jnp.where(left, g, plan)
            xn, dn = _helix(x, d, pm, q, B, step)
            apply = jnp.ones(n, bool)
        else:
            g = to_face(x[:, 2], d[:, 2])
            if first_exit is not None:
                g = jnp.minimum(g, extra_face(x, d, pm, q, plan))
            left = g <= plan
            step = jnp.where(left, g, plan)
            xn, dn = x + step[:, None] * d, d
            apply = jnp.ones(n, bool)
        step = jnp.where(live, step, 0.0)
        xn = jnp.where(live[:, None], xn, x)
        dn = jnp.where(live[:, None], dn, d)
        pend = st["pend"] + step
        path = st["path"] + step
        app = live & apply
        loss = eloss_jax.sample_loss_electron(ks[6], mc, T, jnp.where(app, pend, 1e-6), tcut,
                                              jnp.where(app, dedx * pend, 0.0), positron,
                                              **({"unroll": True} if unroll else {})).astype(dt)
        out = app & (loss >= T - eslab.T_STOP)
        cont = st["cont"] + jnp.where(app, jnp.where(out, T, loss), 0.0)
        Tn = jnp.where(app, jnp.where(out, 0.0, T - loss), st["T"])
        pend = jnp.where(app, 0.0, pend)
        stopped = st["stopped"] | out
        exited = st["exited"] | (live & left & ~out)
        act = live & ~left & ~out
        nan, ean = st["nan"], st["ean"]
        if anni:
            nan = jnp.where(out, nan + 2, nan)
            ean = jnp.where(out, ean + 2 * ME, ean)
        # brem or delta at the step end (integral approach: sigma(post) / sigma(pre))
        hit = act & (free <= jnp.minimum(jnp.minimum(limit, free_a), free_h))
        Tp = jnp.where(hit, Tn, 1.0)
        sbp, sdp = I(tab.sig_b, Tp), I(tab.sig_d, Tp)
        r = jax.random.uniform(ks[7], (n,), dt) * sig
        brem = hit & (r < sbp)
        ioni = hit & ~brem & (r < sbp + sdp)
        # brem photon: node picked by the interpolation weight, x from the inverse CDF, element for the 2BS angle
        f = jnp.clip((jnp.log(Tp) - tab.lnt0) / tab.dlnt, 0.0, NT - 1.000001)
        j0 = f.astype(jnp.int32)
        jj = j0 + (jax.random.uniform(ks[8], (n,), dt) < f - j0)
        uc_ = jax.random.uniform(ks[9], (n,), dt)
        c = (1.0 - (1.0 - uc_) ** (1.0 / C_POWER)) * (NX - 1)          # node coordinate of the CDF value
        ci = jnp.minimum(c.astype(jnp.int32), NX - 2)
        xb = tab.brem_x[jj, ci] * (1 - (c - ci)) + tab.brem_x[jj, ci + 1] * (c - ci)
        kph = jnp.where(brem, kcut * jnp.exp(xb * jnp.log(jnp.maximum(Tp, kcut) / kcut)), 0.0)
        kph = jnp.minimum(kph, Tp)
        zb = tab.z[_pick(I(tab.brem_cum, Tp), jax.random.uniform(ks[10], (n,), dt))]
        cg = _cos_2bs(ks[11], Tp, jnp.where(brem, kph, 0.5 * Tp), zb, dt, unroll)
        dg = _rotate_uz(dn, cg, 2 * jnp.pi * jax.random.uniform(ks[12], (n,), dt))
        pp = jnp.sqrt(Tp * (Tp + 2 * ME))
        db = pp[:, None] * dn - kph[:, None] * dg
        db = db / jnp.linalg.norm(db, axis=1, keepdims=True)
        # delta ray
        w = jnp.where(ioni, _sample_delta(ks[13], Tp, tcut, positron, dt, unroll), 0.0)
        pd = jnp.sqrt(w * (w + 2 * ME))
        cd = jnp.minimum(w * (Tp + 2 * ME) / jnp.where(ioni, pd * pp, 1.0), 1.0)
        ddl = _rotate_uz(dn, cd, 2 * jnp.pi * jax.random.uniform(ks[14], (n,), dt))
        de = pp[:, None] * dn - pd[:, None] * ddl
        de = de / jnp.linalg.norm(de, axis=1, keepdims=True)
        dn = jnp.where(brem[:, None], db, jnp.where(ioni[:, None], de, dn))
        Tn = Tn - kph - w
        nph = st["nph"] + brem
        eph = st["eph"] + kph
        nde = st["nde"] + ioni
        ede = st["ede"] + w
        annih = st["annih"]
        if anni:
            ah = act & (free_a < jnp.minimum(jnp.minimum(limit, free), free_h))
            acc = ah & (jax.random.uniform(ks[15], (n,), dt) * sa_max <= I(tab.sig_a, jnp.where(ah, Tn, 1.0)))
            if return_emissions:
                annih_flight = acc
                annih_T = jnp.where(acc, Tn, 1.0)
                annih_d = jnp.where(acc[:, None], dn, jnp.array([0.0, 0.0, 1.0], dt))
            nan = jnp.where(acc, nan + 2, nan)
            ean = jnp.where(acc, ean + Tn + 2 * ME, ean)
            Tn = jnp.where(acc, 0.0, Tn)
            annih = annih | acc
            stopped = stopped | acc
        if msc:
            hh = act & (free_h < jnp.minimum(jnp.minimum(limit, free), free_a))
            Th = jnp.where(hh, Tn, 1.0)
            cum = I(tab.hard_cum, Th)
            pick = _pick(cum, jax.random.uniform(ks[16], (n,), dt))
            iel = pick % nel
            on_nuc = pick < nel
            sc = I(tab.scr, Th)[iel, jnp.arange(n)]
            ucl = I(tab.uc, Th)
            umax = jnp.where(on_nuc, emsc.U_MAX, I(tab.ue, Th))
            w1, w2 = 1 / (ucl + sc), 1 / (umax + sc)
            uh = jnp.clip(1 / (w1 - jax.random.uniform(ks[17], (n,), dt) * (w1 - w2)) - sc, ucl, umax)
            mom2 = Th * (Th + 2 * ME)
            beta = jnp.sqrt(mom2) / (Th + ME)
            ff = jnp.where(on_nuc & (Th > tab.ff_tmin), 1 / (1 + tab.ffa[iel] * mom2 * uh) ** 2, 1.0)
            envl = I(tab.env, Th)[iel, jnp.arange(n)]
            acc = hh & (jax.random.uniform(ks[18], (n,), dt) * envl
                        <= _mott(tab.z[iel], uh, beta, tab.mott[iel], positron, mott) * ff)
            dh = _rotate_uz(dn, 1 - uh, 2 * jnp.pi * jax.random.uniform(ks[19], (n,), dt))
            dn = jnp.where(acc[:, None], dh, dn)
        low = ~stopped & ~exited & (Tn <= eslab.T_STOP)
        cont = cont + jnp.where(low, Tn, 0.0)
        if anni:
            nan = jnp.where(low, nan + 2, nan)
            ean = jnp.where(low, ean + 2 * ME, ean)
        Tn = jnp.where(low, 0.0, Tn)
        stopped = stopped | low
        next_state = dict(T=Tn, d=dn, x=xn, path=path, pend=pend, cont=cont, stopped=stopped, exited=exited, annih=annih,
                          nph=nph, eph=eph, nde=nde, ede=ede, nan=nan, ean=ean, it=st["it"] + 1)
        if return_emissions:
            if anni:
                pair = annih_flight | out | low
                pair_e, pair_d = _annihilation_pair(ks[20:24], annih_T, annih_d, annih_flight)
                code = jnp.where(annih_flight, 2, 3)
            else:
                pair = jnp.zeros(n, bool)
                pair_e, pair_d = jnp.zeros((n, 2), dt), jnp.zeros((n, 2, 3), dt)
                code = jnp.zeros(n, jnp.int32)
            valid = jnp.stack([brem, pair, pair], axis=1)
            brems_e = jnp.where(brem, kph, jnp.zeros(n, dt))
            brems_d = jnp.where(brem[:, None], dg, jnp.zeros((n, 3), dt))
            photons = _emission_record(jnp.concatenate([brems_e[:, None], pair_e], axis=1), xn,
                                       jnp.concatenate([brems_d[:, None, :], pair_d], axis=1), path, valid,
                                       jnp.stack([jnp.ones(n, jnp.int32), code, code], axis=1))
            return next_state, photons
        if return_photons:
            zero = jnp.zeros((n,), dt)
            zero3 = jnp.zeros((n, 3), dt)
            photons = BremsPhotonRecord(jnp.where(brem, kph, zero), jnp.where(brem[:, None], xn, zero3),
                                        jnp.where(brem[:, None], dg, zero3), jnp.where(brem, path, zero), brem)
            return next_state, photons
        return next_state
    return body


@partial(jax.jit, static_argnames=("positron", "msc", "annihilation", "max_steps", "mott", "return_state"))
def _run(key, tab, mc, T0, d0, L, tcut, kcut, dsmax, positron, msc, annihilation, max_steps, mott, x0=None, B=None,
         st_in=None, it_end=None, return_state=False):
    """The step loop. Optional: start positions x0, per-track constant fields B (helix steps, faces along the
    helix), a state st_in to resume from (lane compaction) and a traced iteration bound it_end."""
    dt = dprec.float_of(T0)
    tab = Tables(*(jnp.asarray(v, dt) for v in tab))
    mc = eloss_jax.MatConsts(*(jnp.asarray(v, dt) for v in mc))
    n = T0.shape[0]
    L, tcut, kcut, dsmax = (jnp.asarray(v, dt) for v in (L, tcut, kcut, dsmax))
    anni = positron and annihilation
    field = B is not None
    if field:
        B = jnp.asarray(B, dt)
    if st_in is None:
        f32z = jnp.zeros(n, dt)
        st0 = dict(T=jnp.asarray(T0, dt), d=jnp.asarray(d0, dt), x=jnp.zeros((n, 3), dt), path=f32z, pend=f32z,
                   cont=f32z, stopped=T0 <= eslab.T_STOP, exited=jnp.zeros(n, bool), annih=jnp.zeros(n, bool),
                   nph=jnp.zeros(n, jnp.int32), eph=f32z, nde=jnp.zeros(n, jnp.int32), ede=f32z,
                   nan=jnp.zeros(n, jnp.int32), ean=f32z, it=jnp.int32(0))
        if x0 is not None:
            st0["x"] = jnp.asarray(x0, dt)
        z, dz = st0["x"][:, 2], st0["d"][:, 2]
        st0["exited"] = ~st0["stopped"] & (((z == 0) & (dz < 0)) | ((z == L) & (dz > 0)))
        st0["cont"] = jnp.where(st0["stopped"], st0["T"], 0.0)
        if anni:
            st0["nan"] = jnp.where(st0["stopped"], 2, 0)
            st0["ean"] = jnp.where(st0["stopped"], 2 * ME, 0.0)
        st0["T"] = jnp.where(st0["stopped"], 0.0, st0["T"])
    else:
        st0 = st_in
    stop_at = max_steps if it_end is None else jnp.minimum(it_end, max_steps)
    I = lambda tabv, T: _interp(tabv, tab.lnt0, tab.dlnt, T)
    body = _slab_body(key, tab, mc, I, n, dt, L, tcut, kcut, dsmax, positron, msc, anni, mott, B)

    def cond(st):
        return jnp.any(~st["stopped"] & ~st["exited"]) & (st["it"] < stop_at)

    st = lax.while_loop(cond, body, st0)
    if return_state:
        return st
    refl = st["exited"] & (st["d"][:, 2] < 0)
    return SlabOut(st["T"], st["d"], st["x"], st["path"], st["cont"], st["stopped"], refl, st["annih"], st["nph"],
                   st["eph"], st["nde"], st["ede"], st["nan"], st["ean"], st["it"])


def electron_slab(key, mat, T, d, length, tcut, kcut, positron=False, msc=False, annihilation=True, max_step=None,
                  hard_per_step=eslab.HARD_PER_STEP, max_steps=200_000, z0=None, field=None, chunk=None,
                  min_bucket=256):
    """eslab.electron_slab on the device: same arguments with a jax.random key; returns SlabOut (per-track
    secondary counts and energy sums). Computes in the dtype of T. Opt-in, as in eslab: z0 (start depth) and field
    (a 3-vector or (n, 3) per-track constant fields [T]: helix steps, faces along the helix). Tracks starting on a
    face and pointing outward exit without transport. chunk (positive integer): run the loop in rounds of `chunk`
    iterations and keep only the live lanes between rounds (padded to a power of two, at least
    min_bucket); statistically equivalent, not identical per seed (a lane's random numbers depend on its slot)."""
    if chunk is not None and (not isinstance(chunk, (int, np.integer)) or isinstance(chunk, (bool, np.bool_)) or chunk <= 0):
        raise ValueError("electron_slab: chunk must be a positive integer")
    dt = dprec.float_of(T)
    T = jnp.asarray(T, dt)
    n = T.shape[0]
    dsmax = length / eslab.MSC_SLAB_STEPS if max_step is None else float(max_step)
    tab = build_tables(mat, tcut, kcut, positron, dsmax / hard_per_step if msc else np.inf)
    mc = eloss_jax.consts(mat)
    x0 = None
    if z0 is not None:
        z0 = np.broadcast_to(np.asarray(z0, float), (n,))
        if ((z0 < 0) | (z0 > length)).any():
            raise ValueError("electron_slab: z0 outside [0, length]")
        x0 = jnp.asarray(np.stack([np.zeros(n), np.zeros(n), z0], axis=1), dt)
    B = None if field is None else jnp.asarray(np.broadcast_to(np.asarray(field, float), (n, 3)), dt)
    args = (tab, mc, T, jnp.asarray(d, dt), length, tcut, kcut, dsmax, bool(positron), bool(msc), bool(annihilation),
            int(max_steps), emsc.MOTT)
    if chunk is None:
        if x0 is None and B is None:
            return _run(key, *args)
        return _run(key, *args, x0=x0, B=B)
    return _compacted(key, args, x0, B, int(chunk), int(min_bucket))


def _compacted(key, args, x0, B, chunk, min_bucket):
    """Rounds of `chunk` iterations; between rounds the live lanes are gathered into a power-of-two bucket (dead
    padding lanes are marked stopped) and finished lanes are written back on the host."""
    tab, mc, T, d, *rest = args
    n = T.shape[0]
    max_steps = rest[-2]
    st = _run(key, tab, mc, T, d, *rest, x0=x0, B=B, it_end=jnp.int32(chunk), return_state=True)
    final = {k: np.array(v) for k, v in st.items() if k != "it"}
    idx = np.arange(n)
    Bh = None if B is None else np.asarray(B)
    rounds = 1
    while True:
        it = int(st["it"])
        stv = {k: np.asarray(v) for k, v in st.items() if k != "it"}
        real = idx >= 0
        for k in final:
            final[k][idx[real]] = stv[k][real]
        live = real & ~stv["stopped"] & ~stv["exited"]
        if not live.any() or it >= max_steps:
            break
        keep = np.flatnonzero(live)
        m = max(min_bucket, 1 << int(np.ceil(np.log2(keep.size))))
        m = min(m, max(n, min_bucket))
        sel = np.concatenate([keep, np.full(m - keep.size, keep[0])])
        nst = {k: v[sel] for k, v in stv.items()}
        nst["stopped"][keep.size:] = True             # padding lanes: copies of a live lane, inert
        idx = np.concatenate([idx[keep], np.full(m - keep.size, -1)])
        Bc = None if Bh is None else jnp.asarray(Bh[np.maximum(idx, 0)])
        nst = {k: jnp.asarray(v) for k, v in nst.items()}
        nst["it"] = jnp.int32(it)
        st = _run(key, tab, mc, nst["T"], nst["d"], *rest, B=Bc, st_in=nst, it_end=jnp.int32(it + chunk),
                  return_state=True)
        rounds += 1
    refl = final["exited"] & (final["d"][:, 2] < 0)
    return SlabOut(final["T"], final["d"], final["x"], final["path"], final["cont"], final["stopped"], refl,
                   final["annih"], final["nph"], final["eph"], final["nde"], final["ede"], final["nan"], final["ean"],
                   np.int32(it))


class LaneTables(NamedTuple):
    """Slab tables of several layer types stacked on a leading axis (lane_tables), for per-lane materials."""
    tab: Tables       # (M, ...) arrays; elements padded to the largest count (zero-width picks), common ln T grid
    mc: object        # eloss_jax.MatConsts of (M,) arrays
    length: object    # (M,) slab thickness [mm]
    tcut: object      # (M,)
    kcut: object      # (M,)
    dsmax: object     # (M,) msc step cap


def lane_tables(layers, positron=False, msc=True, hard_per_step=eslab.HARD_PER_STEP):
    """LaneTables of layer types given as (material, length, tcut, kcut) or (..., max_step), with the tables and step
    caps electron_slab would build for each of them on its own."""
    tabs, mcs, rows = [], [], []
    for spec in layers:
        mat, length, tcut, kcut = spec[:4]
        dsmax = length / eslab.MSC_SLAB_STEPS if len(spec) < 5 or spec[4] is None else float(spec[4])
        tabs.append(build_tables(mat, tcut, kcut, positron, dsmax / hard_per_step if msc else np.inf))
        mcs.append(eloss_jax.consts(mat))
        rows.append((length, tcut, kcut, dsmax))
    nmax = max(t.z.size for t in tabs)

    def pad_el(a, fill):                       # (nel, ...) -> (nmax, ...), extra rows `fill` or a copy of the last
        extra = nmax - a.shape[0]
        if extra == 0:
            return np.asarray(a, float)
        rep = np.repeat(a[-1:], extra, axis=0) if fill is None else np.full((extra,) + a.shape[1:], fill, float)
        return np.concatenate([a, rep], axis=0)
    out = {}
    for f in Tables._fields:
        vals = [getattr(t, f) for t in tabs]
        if f in ("lnt0", "dlnt", "ff_tmin"):
            if len(set(float(v) for v in vals)) != 1:
                raise ValueError(f"lane_tables: {f} differs between layer types")
            out[f] = float(vals[0])
        elif f == "hard_cum":                  # nuclei block, then electrons block: pad each with its last row
            out[f] = np.stack([np.concatenate([pad_el(v[:v.shape[0] // 2], None), pad_el(v[v.shape[0] // 2:], None)])
                               for v in vals])
        elif f in ("scr", "env", "z"):
            out[f] = np.stack([pad_el(np.asarray(v, float), 1.0) for v in vals])
        elif f == "brem_cum":
            out[f] = np.stack([pad_el(v, None) for v in vals])
        elif f == "ffa":
            out[f] = np.stack([pad_el(np.asarray(v, float), 0.0) for v in vals])
        elif f == "mott":
            out[f] = np.stack([pad_el(v, None) for v in vals])
        else:
            out[f] = np.stack([np.asarray(v, float) for v in vals])
    mc = eloss_jax.MatConsts(*(np.array([getattr(c, f) for c in mcs], float) for f in eloss_jax.MatConsts._fields))
    return LaneTables(Tables(**out), mc, *(np.array(c, float) for c in zip(*rows)))


class _Lanes:
    """A stacked table seen through the per-lane layer index: a[i] -> stack[mi, i] (shape without the layer axis)."""

    def __init__(self, a, mi):
        self.a, self.mi = a, mi
        self.shape = a.shape[1:]

    def __getitem__(self, idx):
        return self.a[(self.mi,) + (idx if isinstance(idx, tuple) else (idx,))]


def _interp_lanes(tab, lnt0, dlnt, T):
    """_interp of a _Lanes table: lane j reads its own layer's row."""
    f = jnp.clip((jnp.log(jnp.maximum(T, 1e-30)) - lnt0) / dlnt, 0.0, NT - 1.000001)
    i = f.astype(jnp.int32)
    w = f - i
    a, mi = tab.a, tab.mi
    if a.ndim == 2:
        return a[mi, i] * (1 - w) + a[mi, i + 1] * w
    return (a[mi, :, i] * (1 - w)[:, None] + a[mi, :, i + 1] * w[:, None]).T


def lane_state(T, d, z0, length, positron=False, annihilation=True):
    """Slab loop state for lanes starting at depth z0 in direction d (local frame) with kinetic energy T, as
    electron_slab starts them (below T_STOP: stopped; on a face pointing outward: exited). Device arrays."""
    dt = dprec.float_of(T)
    T = jnp.asarray(T, dt)
    n = T.shape[0]
    zero = jnp.zeros(n, dt)
    x = jnp.zeros((n, 3), dt).at[:, 2].set(jnp.asarray(z0, dt))
    d = jnp.asarray(d, dt)
    stopped = T <= eslab.T_STOP
    z, dz = x[:, 2], d[:, 2]
    anni = positron and annihilation
    return dict(T=jnp.where(stopped, 0.0, T), d=d, x=x, path=zero, pend=zero, cont=jnp.where(stopped, T, 0.0),
                stopped=stopped, exited=~stopped & (((z == 0) & (dz < 0)) | ((z == length) & (dz > 0))),
                annih=jnp.zeros(n, bool), nph=jnp.zeros(n, jnp.int32), eph=zero, nde=jnp.zeros(n, jnp.int32), ede=zero,
                nan=jnp.where(stopped, 2, 0).astype(jnp.int32) if anni else jnp.zeros(n, jnp.int32),
                ean=jnp.where(stopped, 2 * ME, 0.0).astype(dt) if anni else zero, it=jnp.int32(0))


def lane_step(key, lt, mi, st, B=None, positron=False, msc=True, annihilation=True, mott=None, unroll=False):
    """One slab step for every lane of the state dict st (lane_state), lane j in layer type mi[j] of the LaneTables
    lt, with per-lane constant fields B (n, 3) [T] in the slab frame or None; finished lanes (stopped or exited) are
    left as they are. Traceable (call inside a jitted loop); the random stream is key folded with st["it"]. unroll
    (opt-in): fixed-trip loops unrolled (fewer device launches per step on a GPU)."""
    return _lane_step(key, lt, mi, st, B, positron, msc, annihilation, mott, unroll, False)


def lane_step_photons(key, lt, mi, st, B=None, positron=False, msc=True, annihilation=True, mott=None, unroll=False):
    """lane_step with a fixed-shape bremsstrahlung-only emission record: (next_state, BremsPhotonRecord).

    Traceable inside jitted loops, with the same arguments, random keys and next state as lane_step. The record
    contains at most one photon per lane: its energy, emission position, pre-recoil direction, parent path and
    valid mask. Finished lanes and non-emitting steps have invalid, zero records (direction is also zero).
    This does not record annihilation photons or delta cascades, propagate a shower, or report flux at an exit.
    """
    return _lane_step(key, lt, mi, st, B, positron, msc, annihilation, mott, unroll, True)


def lane_step_emissions(key, lt, mi, st, B=None, positron=False, msc=True, annihilation=True, mott=None, unroll=False,
                        *, first_exit=None, finite_lanes=None):
    """lane_step with full photon births: (next_state, EMPhotonRecord), with identical primary draws/state.

    Same arguments as lane_step; traceable inside jitted loops. Slot 0 matches lane_step_photons to rounding;
    slots 1 and 2 use the existing accepted in-flight / newly-stopped annihilation masks. In flight,
    energy and direction are captured before the primary energy is zeroed. At rest the pair has total
    energy 2*ME at the step-end position/path, including range-out and low-energy stops. Only unused
    ks[20:24] generate secondary directions, inheriting eannihil's isotropic COM boost and isotropic
    back-to-back rest convention (not new Geant4 validation). Finished lanes never emit again.
    Already-stopped lane_state positrons are initialization births, not new steps: consume
    initial_annihilation_photons once immediately after initialization if those births are needed.
    This records births only, not photon transport, delta cascades, showers or exit flux.
    first_exit (optional static, traceable closure) takes (xx, uu, pm, q, B, smax) for each actual leg
    in local slab coordinates and returns per-lane path to the first EXTRA material exit, or inf.
    B is None for straight legs; pm and q are the actual momentum magnitude and charge sign.
    The caller guarantees nonnegative distances and externally flags unresolved geometry searches.
    Existing z faces still compete; boundary exits flush pending loss and suppress endpoint hard events,
    but do not override range-out. finite_lanes selects extra-boundary lanes and disables their z-only
    trapped shortcut (defaults to all lanes with a hook); other lanes retain slab behavior.
    This hook is not finite-foil navigation or lateral state integration and adds no return fields.
    The recorder replays the shared step behind a compiler barrier to keep primary arithmetic byte-identical;
    this costs extra computation, not extra primary random draws. Defaults do not replay the step.
    """
    return _lane_step(key, lt, mi, st, B, positron, msc, annihilation, mott, unroll, False, True,
                      first_exit=first_exit, finite_lanes=finite_lanes)


def _lane_step(key, lt, mi, st, B, positron, msc, annihilation, mott, unroll, return_photons,
               return_emissions=False, *, first_exit=None, finite_lanes=None):
    """Shared lane setup for the state-only and photon-recording forms of the same slab step."""
    dt = dprec.float_of(st["T"])
    tab = Tables(*(v if isinstance(v, float) else _Lanes(jnp.asarray(v, dt), mi) for v in lt.tab))
    mc = eloss_jax.MatConsts(*(jnp.asarray(v, dt)[mi] for v in lt.mc))
    n = st["T"].shape[0]
    L, tcut, kcut, dsmax = (jnp.asarray(v, dt)[mi] for v in (lt.length, lt.tcut, lt.kcut, lt.dsmax))
    I = lambda tabv, T: _interp_lanes(tabv, tab.lnt0, tab.dlnt, T)
    body = _slab_body(key, tab, mc, I, n, dt, L, tcut, kcut, dsmax, bool(positron), bool(msc),
                      bool(positron and annihilation), emsc.MOTT if mott is None else mott,
                      None if B is None else jnp.asarray(B, dt), unroll, return_photons, return_emissions,
                      first_exit=first_exit, finite_lanes=finite_lanes)
    if return_emissions:
        # Replay only the optional recorder so secondary outputs cannot alter primary compiler fusion.
        primary = _slab_body(key, tab, mc, I, n, dt, L, tcut, kcut, dsmax, bool(positron), bool(msc),
                             bool(positron and annihilation), emsc.MOTT if mott is None else mott,
                             None if B is None else jnp.asarray(B, dt), unroll,
                             first_exit=first_exit, finite_lanes=finite_lanes)
        next_state = primary(st)
        _, record = body(lax.optimization_barrier(st))
        return next_state, record
    return body(st)


def compare_numpy(mat, T0, length, tcut, kcut, positron=False, msc=True, n=100_000, seed=1, z0=None, field=None,
                  chunk=None, isotropic=False):
    """Statistical parity with eslab.electron_slab: fates, loss of transmitted primaries, exit angle, displacement and
    secondary yields; two-sample chi2 (eslab_g4.chi2) on equal-probability bins of the NumPy sample. z0, field and
    chunk as in electron_slab; isotropic: directions uniform on the sphere (with z0) instead of along +z."""
    import time
    from . import eslab_g4
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    if isotropic:
        g = np.random.default_rng(seed + 1000)
        ct, ph = g.uniform(-1, 1, n), g.uniform(0, 2 * np.pi, n)
        d = np.stack([np.sqrt(1 - ct ** 2) * np.cos(ph), np.sqrt(1 - ct ** 2) * np.sin(ph), ct], axis=1)
    t0 = time.time()
    r = eslab.electron_slab(np.random.default_rng(seed), mat, np.full(n, T0), d, length, tcut, kcut, positron,
                            msc=msc, z0=z0, field=field)
    t1 = time.time()
    fd = dprec.FLOAT             # the device float of this process (MU2EFS_DEVICE_FLOAT)
    j = electron_slab(jax.random.PRNGKey(seed), mat, np.full(n, T0, fd), d.astype(fd), length, tcut, kcut, positron,
                      msc=msc, z0=z0, field=field, chunk=chunk)
    j = jax.tree_util.tree_map(np.asarray, j)
    t2 = time.time()
    th = lambda dd: np.arctan2(np.hypot(dd[:, 0], dd[:, 1]), dd[:, 2])
    tn, tj = ~r.stopped & ~r.reflected, ~j.stopped & ~j.reflected
    out = dict(material=mat.name, T=T0, length=length, positron=positron, msc=msc, n=n, seconds=[t1 - t0, t2 - t1],
               steps=int(j.steps), dtype=str(j.T.dtype))
    for k, a, b in (("stopped", r.stopped, j.stopped), ("reflected", r.reflected, j.reflected),
                    ("annihilated", r.annihilated, j.annihilated)):
        out[k] = [float(a.mean()), float(b.mean())]
    for k, a, b in (("loss", T0 - r.T[tn], T0 - j.T[tj]), ("theta", th(r.direction)[tn], th(j.direction)[tj]),
                    ("r", np.hypot(*r.position[tn, :2].T), np.hypot(*j.position[tj, :2].T))):
        e = eslab_g4.quantile_edges(a, 40, 0.0, max(a.max(), b.max()) * 1.0001)
        c, nb = eslab_g4.chi2(a, b, e)
        out[k] = dict(chi2_ndf=c / max(nb, 1), mean=[float(a.mean()), float(b.mean())],
                      err=float(a.std() / np.sqrt(a.size)))
    out["photons_per_event"] = [r.photons.track.size / n, float(j.n_photons.mean())]
    out["deltas_per_event"] = [r.deltas.track.size / n, float(j.n_deltas.mean())]
    out["photon_energy_per_event"] = [float(r.photons.energy.sum() / n), float(j.photon_energy.mean())]
    return out


def bench(mat, T0, length, tcut, kcut, n, positron=False, msc=True, reps=2, chunk=None, field=None):
    """Device throughput: seconds per call (after a compile call) for n primaries; float of the process setting;
    chunk (lane compaction) and field (constant 3-vector [T]) as in electron_slab."""
    import time
    fd = dprec.FLOAT
    d = np.tile(np.array([0.0, 0.0, 1.0], fd), (n, 1))
    T = np.full(n, T0, fd)
    kw = dict(msc=msc, chunk=chunk, field=field)
    t0 = time.time()
    out = electron_slab(jax.random.PRNGKey(0), mat, T, d, length, tcut, kcut, positron, **kw)
    jax.block_until_ready(out.T)
    first = time.time() - t0
    times = []
    for r in range(reps):
        t0 = time.time()
        out = electron_slab(jax.random.PRNGKey(r + 1), mat, T, d, length, tcut, kcut, positron, **kw)
        jax.block_until_ready(out.T)
        times.append(time.time() - t0)
    return dict(material=mat.name, T=T0, length=length, n=n, dtype=str(np.dtype(fd)), backend=jax.default_backend(),
                device=str(jax.devices()[0].device_kind), first_call_s=first, call_s=times, steps=int(out.steps),
                per_primary_us=1e6 * min(times) / n, chunk=chunk, field=None if field is None else list(field),
                stopped=float(np.mean(np.asarray(out.stopped))), reflected=float(np.mean(np.asarray(out.reflected))))


if __name__ == "__main__":
    import argparse
    import json
    from . import materials
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.eslab_jax", description="device vs NumPy slab parity")
    ap.add_argument("material")
    ap.add_argument("T", type=float)
    ap.add_argument("length", type=float)
    ap.add_argument("tcut", type=float)
    ap.add_argument("kcut", type=float)
    ap.add_argument("--positron", action="store_true")
    ap.add_argument("--no-msc", action="store_true")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--bench", action="store_true", help="device timing only (no NumPy arm)")
    ap.add_argument("--chunk", type=int, help="lane compaction every CHUNK iterations")
    ap.add_argument("--field", type=float, nargs=3, help="constant field [T]")
    ap.add_argument("--z0", type=float, help="start depth [mm] (parity only)")
    ap.add_argument("--isotropic", action="store_true", help="isotropic start directions (parity only)")
    a = ap.parse_args()
    if a.bench:
        print(json.dumps(bench(materials.BY_NAME[a.material], a.T, a.length, a.tcut, a.kcut, a.n, a.positron,
                               not a.no_msc, chunk=a.chunk, field=a.field), indent=1))
        raise SystemExit
    print(json.dumps(compare_numpy(materials.BY_NAME[a.material], a.T, a.length, a.tcut, a.kcut, a.positron,
                                   not a.no_msc, a.n, z0=a.z0, field=a.field, chunk=a.chunk,
                                   isotropic=a.isotropic), indent=1))
