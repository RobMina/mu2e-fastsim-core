"""e-/e+ crossing of a uniform slab with Geant4 EM option 4 physics, optionally in a magnetic field.

The slab is 0 <= z <= length, unbounded in x and y; tracks start at the origin with directions d (d_z > 0). Each step
is the shortest of the free path to the next discrete interaction (Seltzer-Berger photon above kcut or Moller / Bhabha
delta above tcut, from the cross sections at the pre-step energy; for e+ also annihilation in flight; with msc also a
hard elastic collision), the step limit of option 4's e+- step function (mean continuous loss at most STEP_FRACTION of
the kinetic energy, not below FINAL_RANGE), with msc a fraction of the transport mean free path and of the slab
thickness, and the distance to the slab face. The fluctuated continuous loss (eioni.continuous_loss) is taken over the
path since the last loss point: at every step without msc; with msc not at the msc sub-steps and hard collisions,
whose many short segments would narrow the Urban straggling. At a brem/delta point the interaction happens with
probability sigma(post) / sigma(pre) (G4VEnergyLossProcess integral approach); annihilation and hard elastic
collisions, whose cross sections grow as the energy falls, are sampled against their value at half the pre-step energy
and thinned. Photons and deltas are returned, not transported; the parent's direction changes by their recoil. With
msc=True (emsc) the soft deflection of a step is applied at a uniformly random point along it (random hinge), hard
collisions at the step end; a track whose restricted CSDA range is shorter than its distance to both faces cannot
leave, so it steps by energy loss only and scatters with the soft part alone (u_c = 2), its secondaries still made. A positron that annihilates in flight or stops gives two photons (eannihil). Energies in MeV,
lengths in mm.
"""
from typing import NamedTuple

import numpy as np

from . import eannihil, ebrem, ehelix, eioni, emsc

STEP_FRACTION = 0.2      # G4EmStandardPhysics_option4 SetStepFunction(0.2, 10 um) for e+-
FINAL_RANGE = 0.01       # mm
T_STOP = 1e-3            # MeV: below this the electron deposits what it has left
MSC_MIN_STEP = 1e-5      # mm: floor of the msc step limit
MSC_SLAB_STEPS = 10      # with msc, steps at most length / MSC_SLAB_STEPS (PENELOPE's DSMAX)
HARD_PER_STEP = 2        # with msc, at least this many hard collisions per max_step (soft part stays Gaussian)
MAX_STEPS = 100_000
SIGMA_MAX_FRACTION = 0.5  # annihilation and hard elastic: majorant cross section at this fraction of the energy


class Secondaries(NamedTuple):
    track: np.ndarray     # index of the parent
    s: np.ndarray         # path length of the parent from the slab entry [mm]
    energy: np.ndarray    # kinetic energy [MeV]
    direction: np.ndarray
    position: np.ndarray  # (n, 3) emission point [mm]


class SlabResult(NamedTuple):
    T: np.ndarray           # kinetic energy where the track leaves the slab (0 if stopped or annihilated)
    direction: np.ndarray   # direction there
    continuous: np.ndarray  # continuous loss deposited along the path [MeV]
    stopped: np.ndarray     # stopped inside (ranged out or annihilated)
    photons: Secondaries
    deltas: Secondaries
    position: np.ndarray    # (n, 3) exit or stopping point
    path: np.ndarray        # path length in the slab [mm]
    reflected: np.ndarray   # left through the entry face (z = 0)
    annihilated: np.ndarray  # e+ annihilated in flight
    annihilation: Secondaries  # annihilation photons (in flight and at rest), two rows per positron


def _empty():
    e = np.empty(0)
    return Secondaries(e.astype(int), e, e, np.empty((0, 3)), np.empty((0, 3)))


def _cat(parts):
    if not parts:
        return _empty()
    return Secondaries(*(np.concatenate(c) for c in zip(*parts)))


def _sigmas(mat, T, tcut, kcut, positron):
    sb = np.where(T > kcut, ebrem.xsec_per_mm(mat, T, kcut, positron), 0.0)
    sd = np.where(eioni.max_secondary(T, positron) > tcut, eioni.xsec_per_mm(mat, T, tcut, positron), 0.0)
    return sb, sd


def _free(rng, sig):
    """Exponential free paths for macroscopic cross sections sig (infinite where sig = 0)."""
    free = rng.exponential(1.0, sig.size) / np.where(sig > 0, sig, 1.0)
    return np.where(sig > 0, free, np.inf)


def _to_face(z, dz, length):
    """Distance along the direction to the face the track is heading to."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dz > 0, (length - z) / dz, np.where(dz < 0, -z / dz, np.inf))


_RANGE = {}


def _range_table(mat, tcut, kcut, positron):
    """(ln T grid, restricted CSDA range [mm]) from T_STOP to 1 GeV, cached by material name and cuts."""
    key = (mat.name, tcut, kcut, positron)
    if key not in _RANGE:
        _RANGE[key] = _build_range(mat, tcut, kcut, positron)
    return _RANGE[key]


def _build_range(mat, tcut, kcut, positron):
    T = np.geomspace(T_STOP, ebrem.T_MAX, 400)
    dedx = eioni.dedx_restricted(mat, T, tcut, positron) + ebrem.dedx_restricted(mat, T, kcut, positron)
    f = T / dedx
    lt = np.log(T)
    rng = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lt))])
    return lt, rng


def csda_range(mat, T, tcut, kcut, positron=False):
    """Restricted CSDA range [mm] above T_STOP (continuous loss only: an upper bound on the path length)."""
    lt, r = _range_table(mat, float(tcut), float(kcut), bool(positron))
    return np.interp(np.log(np.maximum(np.asarray(T, float), T_STOP)), lt, r)


def electron_slab(rng, mat, T, d, length, tcut, kcut, positron=False, msc=False, annihilation=True, max_step=None,
                  hard_per_step=HARD_PER_STEP, field=None):
    """Cross a slab of thickness `length` [mm] of `mat` from z = 0 along directions d (n, 3) from kinetic energies
    T [MeV]; msc=True adds multiple scattering (emsc), annihilation=False switches off e+ annihilation, max_step [mm]
    caps the msc step (default length / MSC_SLAB_STEPS: a hinge beyond the face would leave the track unscattered) and
    hard_per_step sets the least number of hard elastic collisions per max_step; field (a 3-vector [T] or a callable
    B(x)) bends every step on the helix of the field at its start (ehelix), the faces found along the helix."""
    T = np.array(np.atleast_1d(T), float)
    n = T.size
    d = np.array(d, float).reshape(n, 3)
    L = float(length)
    pos = np.zeros((n, 3))
    path = np.zeros(n)
    pend = np.zeros(n)       # path since the last continuous-loss point
    cont = np.zeros(n)
    stopped = np.zeros(n, bool)
    exited = np.zeros(n, bool)
    annihilated = np.zeros(n, bool)
    anni = positron and annihilation
    dsmax = L / MSC_SLAB_STEPS if max_step is None else float(max_step)
    charge = 1.0 if positron else -1.0
    photons, deltas, gammas = [], [], []

    def stop_at_rest(idx):
        if anni and idx.size:
            e, u = eannihil.at_rest(rng, idx.size)
            gammas.append((np.repeat(idx, 2), np.repeat(path[idx], 2), e.ravel(), u.reshape(-1, 3),
                           np.repeat(pos[idx], 2, axis=0)))

    live = np.flatnonzero(T > T_STOP)
    low0 = np.flatnonzero(T <= T_STOP)
    cont[low0] += T[low0]
    T[low0] = 0.0
    stopped[low0] = True
    stop_at_rest(low0)
    for _ in range(MAX_STEPS):
        if live.size == 0:
            break
        Tl = T[live]
        sb, sd = _sigmas(mat, Tl, tcut, kcut, positron)
        sig = sb + sd
        free = _free(rng, sig)
        dedx = eioni.dedx_restricted(mat, Tl, tcut, positron) + ebrem.dedx_restricted(mat, Tl, kcut, positron)
        pl = pend[live]
        rem = np.maximum(STEP_FRACTION * Tl / np.where(dedx > 0, dedx, np.inf), FINAL_RANGE) - pl
        mcap = np.full(live.size, np.inf)
        free_a = free_h = mcap
        if anni:
            sa_max = eannihil.xsec_per_mm(mat, SIGMA_MAX_FRACTION * Tl)
            free_a = _free(rng, sa_max)
        if msc:
            zl = pos[live, 2]
            trapped = csda_range(mat, Tl, tcut, kcut, positron) < np.minimum(zl, L - zl)
            cap = np.where(trapped, np.inf, dsmax)
            uc = np.where(trapped, emsc.U_MAX, emsc.u_cut(mat, Tl, tcut, positron, cap / hard_per_step))
            mcap = np.where(trapped, np.inf, np.minimum(
                np.maximum(emsc.STEP_LAMBDA1 * emsc.lambda1(mat, Tl, tcut, positron), MSC_MIN_STEP), dsmax))
            sh_max = emsc.hard_xsec(mat, SIGMA_MAX_FRACTION * Tl, uc, tcut, positron)
            free_h = _free(rng, sh_max)
        limit = np.minimum(rem, mcap)
        plan = np.minimum(np.minimum(free, limit), np.minimum(free_a, free_h))
        dl = d[live]
        xl = pos[live]
        if field is not None:
            pmom = np.sqrt(Tl * (Tl + 2 * eioni.ME))

        def face(x, u, smax):
            if field is None:
                return _to_face(x[:, 2], u[:, 2], L)
            B = ehelix.field_at(field, x)
            return np.minimum(ehelix.to_plane(x, u, pmom, charge, B, L, smax),
                              ehelix.to_plane(x, u, pmom, charge, B, 0.0, smax))

        def move(x, u, s):
            if field is None:
                return x + s[:, None] * u, u
            return ehelix.helix(x, u, pmom, charge, ehelix.field_at(field, x), s)

        if msc:
            tau = rng.random(live.size)
            leg1 = tau * plan
            g1 = face(xl, dl, leg1)
            out1 = leg1 >= g1
            p1, d1 = move(xl, dl, np.where(out1, g1, leg1))
            tmid = np.maximum(Tl - dedx * (pl + 0.5 * plan), 0.5 * Tl)
            u = emsc.sample_soft(rng, mat, tmid, plan, uc, tcut, positron)
            dn = np.where(out1[:, None], d1, emsc.deflect(rng, d1, u))
            leg2 = plan - leg1
            g2 = face(p1, dn, leg2)
            out2 = (~out1) & (leg2 >= g2)
            step = np.where(out1, g1, np.where(out2, leg1 + g2, plan))
            pos[live], d[live] = move(p1, dn, np.where(out1, 0.0, np.where(out2, g2, leg2)))
            left = out1 | out2
            other = np.minimum(np.minimum(free, rem), free_a)
            sub = (free_h < np.minimum(np.minimum(free, limit), free_a)) | (mcap < np.minimum(other, free_h))
            apply = left | ~sub
        else:
            g = face(xl, dl, plan)
            left = g <= plan
            step = np.where(left, g, plan)
            pos[live], d[live] = move(xl, dl, step)
            apply = np.ones(live.size, bool)
        pend[live] += step
        path[live] += step
        out = np.zeros(live.size, bool)
        ia = np.flatnonzero(apply)
        ja = live[ia]
        loss = eioni.continuous_loss(rng, mat, T[ja], pend[ja], tcut, kcut, positron)
        oa = loss >= T[ja] - T_STOP
        out[ia] = oa
        cont[ja] += np.where(oa, T[ja], loss)
        T[ja] = np.where(oa, 0.0, T[ja] - loss)
        pend[ja] = 0.0
        stopped[ja[oa]] = True
        stop_at_rest(ja[oa])
        ex = left & ~out
        exited[live[ex]] = True
        act = ~left & ~out
        hit = act & (free <= np.minimum(np.minimum(limit, free_a), free_h))
        if hit.any():
            idx, Tp = live[hit], T[live[hit]]
            sbp, sdp = _sigmas(mat, Tp, tcut, kcut, positron)
            r = rng.random(idx.size) * sig[hit]
            brem = r < sbp
            ioni = (~brem) & (r < sbp + sdp)
            if brem.any():
                j = idx[brem]
                k, zz = ebrem.sample_k(rng, mat, T[j], kcut, positron, return_z=True)
                dg, d[j] = ebrem.emit(rng, T[j], d[j], k, zz)
                photons.append((j, path[j].copy(), k, dg, pos[j].copy()))
                T[j] -= k
            if ioni.any():
                j = idx[ioni]
                w = eioni.sample_delta(rng, T[j], tcut, positron)
                dd, d[j] = eioni.delta_direction(rng, T[j], d[j], w)
                deltas.append((j, path[j].copy(), w, dd, pos[j].copy()))
                T[j] -= w
        ah = act & (free_a < np.minimum(np.minimum(limit, free), free_h))
        if ah.any():
            m = np.flatnonzero(ah)
            idx = live[m]
            acc = rng.random(idx.size) * sa_max[m] <= eannihil.xsec_per_mm(mat, T[idx])
            j = idx[acc]
            if j.size:
                e, u2 = eannihil.in_flight(rng, T[j], d[j])
                gammas.append((np.repeat(j, 2), np.repeat(path[j], 2), e.ravel(), u2.reshape(-1, 3),
                               np.repeat(pos[j], 2, axis=0)))
                T[j] = 0.0
                annihilated[j] = True
                stopped[j] = True
        if msc:
            hh = act & (free_h < np.minimum(np.minimum(limit, free), free_a))
            if hh.any():
                m = np.flatnonzero(hh)
                idx = live[m]
                ucm = uc[m]
                keep = rng.random(idx.size) * sh_max[m] <= emsc.hard_xsec(mat, T[idx], ucm, tcut, positron)
                uh, acc = emsc.sample_hard(rng, mat, T[idx], ucm, tcut, positron)
                j = idx[keep & acc]
                if j.size:
                    d[j] = emsc.deflect(rng, d[j], uh[keep & acc])
        cur = live[~stopped[live] & ~exited[live]]
        low = cur[T[cur] <= T_STOP]
        cont[low] += T[low]
        T[low] = 0.0
        stopped[low] = True
        stop_at_rest(low)
        live = live[~stopped[live] & ~exited[live]]
    if live.size:
        raise RuntimeError(f"electron_slab: {live.size} tracks still inside after {MAX_STEPS} steps; first: "
                           f"T {T[live[:5]]} MeV, at z {pos[live[:5], 2]} of {L} mm")
    reflected = exited & (d[:, 2] < 0)
    return SlabResult(T, d, cont, stopped, _cat(photons), _cat(deltas), pos, path, reflected, annihilated,
                      _cat(gammas))
