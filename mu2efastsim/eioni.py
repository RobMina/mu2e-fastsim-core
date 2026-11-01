"""Electron and positron ionisation, matched to Geant4 11.3.2 G4MollerBhabhaModel (G4eIonisation above 100 keV).

Restricted mean loss (ComputeDEDXPerVolume, with its low-energy cap th = 0.25 sqrt(Zeff) keV), the cross section for
delta rays above the cut (CrossSectionPerVolume) and the Moller / Bhabha energy-transfer sampler (SampleSecondaries).
The largest transfer is T/2 for e- (identical particles) and T for e+. EM option 4 uses G4PenelopeIonisationModel
below 100 keV, which is not implemented: these functions keep the Moller-Bhabha formulae there. The delta direction
is the free-electron two-body angle; option 4 draws it with G4DeltaAngle (bound shells), which differs mostly for
low-energy deltas. `continuous_loss` adds the restricted Seltzer-Berger loss and fluctuates the sum with the Urban
model, as G4VEnergyLossProcess does with one table of all continuous losses. Energies in MeV, lengths in mm.
"""
import numpy as np

from . import ebrem
from .eloss import ME, TWOPI_MC2_RCL2, density_correction, fluctuate


def max_secondary(T, positron=False):
    """Largest delta kinetic energy [MeV]: T/2 (Moller) or T (Bhabha)."""
    T = np.asarray(T, float)
    return T if positron else 0.5 * T


def _kin(T):
    tau = T / ME
    gam = tau + 1.0
    bg2 = tau * (tau + 2)
    return tau, gam, bg2, bg2 / (gam * gam)


def dedx_restricted(mat, T, tcut, positron=False):
    """Restricted collision loss [MeV/mm] for delta electrons below tcut (G4MollerBhabhaModel::ComputeDEDXPerVolume)."""
    T = np.atleast_1d(np.asarray(T, float))
    th = 0.25 * np.sqrt(mat.zeff) * 1e-3
    tkin = np.maximum(T, th)
    tau, gam, bg2, beta2 = _kin(tkin)
    eexc2 = (mat.I_eV * 1e-6 / ME) ** 2
    d = np.minimum(tcut, max_secondary(tkin, positron)) / ME
    if positron:
        d2 = d * d * 0.5
        d3 = d2 * d / 1.5
        d4 = d3 * d * 0.75
        y = 1.0 / (1.0 + gam)
        dedx = (np.log(2.0 * (tau + 2.0) / eexc2) + np.log(tau * d)
                - beta2 * (tau + 2.0 * d - y * (3.0 * d2 + y * (d - d3 + y * (d2 - tau * d3 + d4)))) / tau)
    else:
        dedx = (np.log(2.0 * (tau + 2.0) / eexc2) - 1.0 - beta2 + np.log((tau - d) * d) + tau / (tau - d)
                + (0.5 * d * d + (2.0 * tau + 1.0) * np.log(1.0 - d / tau)) / (gam * gam))
    dedx = dedx - density_correction(mat, bg2)
    dedx = np.maximum(dedx * TWOPI_MC2_RCL2 * mat.electron_density_per_mm3 / beta2, 0.0)
    x = T / th
    low = np.where(x > 0.25, 1 / np.sqrt(np.maximum(x, 1e-300)), 1.4 * np.sqrt(x) / (0.1 + x))
    return np.where(T < th, dedx * low, dedx)


def xsec_per_mm(mat, T, tcut, positron=False):
    """Macroscopic cross section [mm-1] for a delta above tcut (ComputeCrossSectionPerElectron times n_e)."""
    T = np.atleast_1d(np.asarray(T, float))
    tmax = max_secondary(T, positron)
    ok = tcut < tmax
    Ts = np.where(ok, T, 1.0)
    xmin = tcut / Ts
    xmax = np.where(ok, tmax, 2 * tcut) / Ts
    tau, gam, bg2, beta2 = _kin(Ts)
    if positron:
        y = 1.0 / (1.0 + gam)
        y2 = y * y
        y12 = 1.0 - 2.0 * y
        b1, b2 = 2.0 - y2, y12 * (3.0 + y2)
        b4 = y12 ** 3
        b3 = b4 + y12 * y12
        cross = ((xmax - xmin) * (1.0 / (beta2 * xmin * xmax) + b2 - 0.5 * b3 * (xmin + xmax)
                                  + b4 * (xmin * xmin + xmin * xmax + xmax * xmax) / 3.0)
                 - b1 * np.log(xmax / xmin))
    else:
        gg = (2.0 * gam - 1.0) / (gam * gam)
        cross = ((xmax - xmin) * (1.0 - gg + 1.0 / (xmin * xmax) + 1.0 / ((1.0 - xmin) * (1.0 - xmax)))
                 - gg * np.log(xmax * (1.0 - xmin) / (xmin * (1.0 - xmax)))) / beta2
    return np.where(ok, cross * TWOPI_MC2_RCL2 / Ts * mat.electron_density_per_mm3, 0.0)


def sample_delta(rng, T, tcut, positron=False):
    """Delta kinetic energies [MeV] above tcut (G4MollerBhabhaModel::SampleSecondaries rejection); needs T > tcut,
    and T > 2 tcut for e-."""
    T = np.atleast_1d(np.asarray(T, float))
    tmax = max_secondary(T, positron)
    if np.any(tcut >= tmax):
        raise ValueError("sample_delta needs tcut below the largest transfer")
    xmin, xmax = tcut / T, tmax / T
    gam = (T + ME) / ME
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

        def zfun(x, m):
            xx = x * x
            return 1.0 + (xx * xx * b4[m] - x * xx * b3[m] + xx * b2[m] - x * b1[m]) * beta2[m]
    else:
        gg = (2.0 * gam - 1.0) / (gam * gam)
        y = 1.0 - xmax
        grej = 1.0 - gg * xmax + xmax * xmax * (1.0 - gg + (1.0 - gg * y) / (y * y))

        def zfun(x, m):
            yv = 1.0 - x
            return 1.0 - gg[m] * x + x * x * (1.0 - gg[m] + (1.0 - gg[m] * yv) / (yv * yv))
    out = np.empty_like(T)
    todo = np.ones(T.size, bool)
    while todo.any():
        idx = np.flatnonzero(todo)
        r0, r1 = rng.random(idx.size), rng.random(idx.size)
        x = xmin[idx] * xmax[idx] / (xmin[idx] * (1.0 - r0) + xmax[idx] * r0)
        acc = grej[idx] * r1 <= zfun(x, idx)
        out[idx[acc]] = x[acc] * T[idx[acc]]
        todo[idx[acc]] = False
    return out


def delta_direction(rng, T, d, w):
    """Free-electron two-body directions: (delta direction, parent's new direction) for a delta of energy w."""
    T, w = np.asarray(T, float), np.asarray(w, float)
    energy = T + ME
    pd = np.sqrt(w * (w + 2 * ME))
    p = np.sqrt(T * (T + 2 * ME))
    cost = np.minimum(w * (energy + ME) / (pd * p), 1.0)
    sint = np.sqrt((1 - cost) * (1 + cost))
    phi = 2 * np.pi * rng.random(T.size)
    dd = ebrem.rotate_uz(d, np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1))
    rest = p[:, None] * d - pd[:, None] * dd
    return dd, rest / np.linalg.norm(rest, axis=1, keepdims=True)


def continuous_loss(rng, mat, T, length, tcut, kcut, positron=False):
    """Sampled continuous loss [MeV] over `length` [mm]: restricted Moller-Bhabha (below tcut) plus restricted
    Seltzer-Berger (below kcut) mean, Urban fluctuations without the heavy-particle Gaussian regime."""
    T = np.atleast_1d(np.asarray(T, float))
    length = np.asarray(length, float) * np.ones_like(T)
    mean = (dedx_restricted(mat, T, tcut, positron) + ebrem.dedx_restricted(mat, T, kcut, positron)) * length
    tmax = max_secondary(T, positron)
    tau, gam, bg2, beta2 = _kin(T)
    return fluctuate(rng, mat, mean, length, np.minimum(tcut, tmax), tmax, beta2, gam, gaussian=False)
