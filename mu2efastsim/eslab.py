"""Straight-line e-/e+ crossing of a uniform slab with Geant4 EM option 4 energy-loss physics (no msc, no field).

Each step is the shortest of the free path to the next discrete interaction (Seltzer-Berger photon above kcut or
Moller / Bhabha delta above tcut, from the cross sections at the pre-step energy), the step limit of option 4's
e+- step function (mean continuous loss at most STEP_FRACTION of the kinetic energy, not below FINAL_RANGE) and
what is left of the slab. The step takes the fluctuated continuous loss (eioni.continuous_loss); at an interaction
point the interaction happens with probability sigma(post) / sigma(pre) (G4VEnergyLossProcess integral approach)
and the process is chosen in proportion to its cross section. Photons and deltas are returned, not transported;
the parent's direction changes by their recoil only. Multiple scattering is left to the caller, and positron
annihilation is not included. Energies in MeV, lengths in mm.
"""
from typing import NamedTuple

import numpy as np

from . import ebrem, eioni

STEP_FRACTION = 0.2      # G4EmStandardPhysics_option4 SetStepFunction(0.2, 10 um) for e+-
FINAL_RANGE = 0.01       # mm
T_STOP = 1e-3            # MeV: below this the electron deposits what it has left
MAX_STEPS = 10_000


class Secondaries(NamedTuple):
    track: np.ndarray    # index of the parent
    s: np.ndarray        # distance from the slab entry [mm]
    energy: np.ndarray   # kinetic energy [MeV]
    direction: np.ndarray


class SlabResult(NamedTuple):
    T: np.ndarray          # kinetic energy at the exit (0 if stopped)
    direction: np.ndarray  # direction at the exit (recoil from photons and deltas only)
    continuous: np.ndarray  # continuous loss deposited along the path [MeV]
    stopped: np.ndarray
    photons: Secondaries
    deltas: Secondaries


def _empty():
    e = np.empty(0)
    return Secondaries(e.astype(int), e, e, np.empty((0, 3)))


def _cat(parts):
    if not parts:
        return _empty()
    return Secondaries(*(np.concatenate(c) for c in zip(*parts)))


def _sigmas(mat, T, tcut, kcut, positron):
    sb = np.where(T > kcut, ebrem.xsec_per_mm(mat, T, kcut, positron), 0.0)
    sd = np.where(eioni.max_secondary(T, positron) > tcut, eioni.xsec_per_mm(mat, T, tcut, positron), 0.0)
    return sb, sd


def electron_slab(rng, mat, T, d, length, tcut, kcut, positron=False):
    """Cross `length` [mm] of `mat` along directions d (n, 3) from kinetic energies T [MeV]."""
    T = np.array(np.atleast_1d(T), float)
    d = np.array(d, float).reshape(T.size, 3)
    full = np.broadcast_to(np.asarray(length, float), T.shape)
    done = np.zeros_like(T)
    cont = np.zeros_like(T)
    stopped = np.zeros(T.size, bool)
    photons, deltas = [], []
    live = np.flatnonzero((done < full) & (T > T_STOP))
    for _ in range(MAX_STEPS):
        if live.size == 0:
            break
        Tl = T[live]
        sb, sd = _sigmas(mat, Tl, tcut, kcut, positron)
        sig = sb + sd
        free = rng.exponential(1.0, live.size) / np.where(sig > 0, sig, 1.0)
        free = np.where(sig > 0, free, np.inf)    # below both cuts: no discrete interaction left
        dedx = eioni.dedx_restricted(mat, Tl, tcut, positron) + ebrem.dedx_restricted(mat, Tl, kcut, positron)
        limit = np.maximum(STEP_FRACTION * Tl / np.where(dedx > 0, dedx, np.inf), FINAL_RANGE)
        left = full[live] - done[live]
        step = np.minimum(np.minimum(free, limit), left)
        loss = eioni.continuous_loss(rng, mat, Tl, step, tcut, kcut, positron)
        out = loss >= Tl - T_STOP
        cont[live] += np.where(out, Tl, loss)
        T[live] = np.where(out, 0.0, Tl - loss)
        stopped[live[out]] = True
        done[live] += step
        hit = (~out) & (free <= np.minimum(limit, left))
        if hit.any():
            idx, Tp = live[hit], T[live[hit]]
            sbp, sdp = _sigmas(mat, Tp, tcut, kcut, positron)
            u = rng.random(idx.size) * sig[hit]
            brem = u < sbp
            ioni = (~brem) & (u < sbp + sdp)
            if brem.any():
                j = idx[brem]
                k, z = ebrem.sample_k(rng, mat, T[j], kcut, positron, return_z=True)
                dg, d[j] = ebrem.emit(rng, T[j], d[j], k, z)
                photons.append((j, done[j].copy(), k, dg))
                T[j] -= k
            if ioni.any():
                j = idx[ioni]
                w = eioni.sample_delta(rng, T[j], tcut, positron)
                dd, d[j] = eioni.delta_direction(rng, T[j], d[j], w)
                deltas.append((j, done[j].copy(), w, dd))
                T[j] -= w
        low = (T[live] <= T_STOP) & ~stopped[live]
        cont[live[low]] += T[live[low]]
        T[live[low]] = 0.0
        stopped[live[low]] = True
        live = live[(done[live] < full[live]) & ~stopped[live]]
    if live.size:
        raise RuntimeError(f"electron_slab: {live.size} tracks still inside after {MAX_STEPS} steps; first: "
                           f"T {T[live[:5]]} MeV, at {done[live[:5]]} of {full[live[:5]]} mm")
    return SlabResult(T, d, cont, stopped, _cat(photons), _cat(deltas))
