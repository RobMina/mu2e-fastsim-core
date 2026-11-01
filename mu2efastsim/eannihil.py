"""Positron annihilation as in Geant4 11.3.2 EM option 4 (G4eplusAnnihilation with its defaults).

In flight: G4eeToTwoGammaModel, the Heitler cross section per electron and two photons isotropic in the centre-of-mass
frame, boosted along the positron. At rest: G4SimplePositronAtRestModel (the default positron-at-rest model), two
back-to-back photons of m_e c^2 in an isotropic direction. Energies in MeV, lengths in mm.
"""
import numpy as np

from . import ebrem
from .eloss import ME

PI_RE2 = np.pi * ebrem.RE_MM ** 2          # mm2


def xsec_per_electron(T):
    """Heitler two-photon annihilation cross section per atomic electron [mm2] (ComputeCrossSectionPerElectron)."""
    T = np.maximum(np.asarray(T, float), 1e-6)
    gam = T / ME + 1.0
    bg2 = (gam - 1.0) * (gam + 1.0)
    bg = np.sqrt(bg2)
    return PI_RE2 * ((gam * gam + 4 * gam + 1.0) * np.log(gam + bg) - (gam + 3.0) * bg) / (bg2 * (gam + 1.0))


def xsec_per_mm(mat, T):
    """Macroscopic in-flight annihilation cross section [mm-1]."""
    return mat.electron_density_per_mm3 * xsec_per_electron(T)


def _isotropic(rng, n):
    cost = 2 * rng.random(n) - 1
    sint = np.sqrt((1 - cost) * (1 + cost))
    phi = 2 * np.pi * rng.random(n)
    return np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1)


def in_flight(rng, T, d):
    """Two photons of a positron of kinetic energy T and direction d annihilating on a free electron at rest:
    (energies (n, 2), directions (n, 2, 3))."""
    T = np.atleast_1d(np.asarray(T, float))
    d = np.asarray(d, float).reshape(T.size, 3)
    etot = T + 2 * ME
    p = np.sqrt(T * (T + 2 * ME))
    half = 0.5 * np.sqrt(etot * etot - p * p)          # photon energy in the centre-of-mass frame
    n = _isotropic(rng, T.size)
    beta = p / etot
    gam = etot / (2 * half)
    e1 = gam * half * (1 + beta * n[:, 2])
    k1 = np.stack([half * n[:, 0], half * n[:, 1], gam * half * (n[:, 2] + beta)], axis=1)
    k1 = ebrem.rotate_uz(d, k1)
    k2 = p[:, None] * d - k1
    e2 = etot - e1
    dirs = np.stack([k1 / e1[:, None], k2 / np.linalg.norm(k2, axis=1, keepdims=True)], axis=1)
    return np.stack([e1, e2], axis=1), dirs


def at_rest(rng, n):
    """Two back-to-back photons of m_e c^2 per stopped positron: (energies (n, 2), directions (n, 2, 3))."""
    u = _isotropic(rng, n)
    return np.full((n, 2), ME), np.stack([u, -u], axis=1)
