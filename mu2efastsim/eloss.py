"""Muon energy loss in thin layers, matched to Geant4 11.3.2 (MuBetheBloch above 0.2 MeV, Urban fluctuations).

Restricted mean loss (G4MuBetheBlochModel::ComputeDEDXPerVolume without shell, high-order and radiative corrections,
which are below 1 % for 0.1-100 GeV muons in polystyrene), the Urban fluctuation sampler (G4UniversalFluctuation +
G4UrbanFluctuation::SampleGlandz) and Birks saturation (G4EmSaturation). Array functions: numpy in, numpy out.
"""
import numpy as np

from .materials import ELEMENTS as ELEMENT_ZA

ME = 0.51099895          # MeV
MMU = 105.6583755        # MeV
TWOPI_MC2_RCL2 = 2 * np.pi * ME * (2.8179403262e-12) ** 2   # MeV mm2
TWOLN10 = 2 * np.log(10.0)


P_FLOOR = 1e-3  # MeV/c: momenta below this are clamped (stopped particles carried through array code)


def kinematics(p, mass=MMU):
    """|p| [MeV/c] -> (kinetic energy, beta^2, beta*gamma squared, gamma, Tmax); p is clamped at P_FLOOR."""
    p = np.maximum(np.asarray(p, float), P_FLOOR)
    e = np.sqrt(p * p + mass * mass)
    gam = e / mass
    bg2 = (p / mass) ** 2
    beta2 = bg2 / (gam * gam)
    ratio = ME / mass
    tmax = 2 * ME * bg2 / (1 + 2 * gam * ratio + ratio * ratio)
    return e - mass, beta2, bg2, gam, tmax


def density_correction(mat, bg2):
    """Sternheimer delta with G4DensityEffectData parameters (G4IonisParamMat::GetDensityCorrection)."""
    C, X0, X1, a, m, d0 = mat.density_params()
    x = np.log(bg2) / TWOLN10
    y = np.where(x >= X1, TWOLN10 * x - C, TWOLN10 * x - C + a * np.clip(X1 - x, 0, None) ** m)
    return np.where(x < X0, d0 * np.exp(TWOLN10 * (x - X0)) if d0 > 0 else 0.0, y)


def dedx_restricted(mat, p, tcut, mass=MMU):
    """Restricted mean energy loss [MeV/mm] for delta electrons below tcut [MeV]."""
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    cut = np.minimum(tcut, tmax)
    I = mat.I_eV * 1e-6
    d = np.log(2 * ME * bg2 * cut / (I * I)) - (1 + cut / tmax) * beta2
    tot = tkin + mass
    d = d + (0.5 * cut / tot) ** 2
    d = d - density_correction(mat, bg2)
    return np.maximum(d * TWOPI_MC2_RCL2 * mat.electron_density_per_mm3 / beta2, 0.0)


# G4UniversalFluctuation model constants
_RATE, _FW, _A0, _NMAXCONT, _BOHR_MIN, _MINLOSS = 0.56, 4.0, 42.0, 8.0, 10.0, 10e-6


def _sample_gauss_clipped(rng, eav, sig2):
    """G4UniversalFluctuation::SampleGauss: uniform in [0, 2 eav] when eav < sig/4, else Gauss truncated to [0, 2 eav]."""
    sig = np.sqrt(sig2)
    out = np.where(eav < 0.25 * sig, eav + (2 * rng.random(eav.shape) - 1) * eav, 0.0)
    g = eav >= 0.25 * sig
    todo = g & (sig2 > 0)
    x = np.zeros_like(eav)
    while todo.any():
        x[todo] = rng.normal(eav[todo], sig[todo])
        todo &= (x < 0) | (x > 2 * eav)
    return np.where(g, np.where(sig2 > 0, x, 0.0), out)


def _add_excitation(rng, ax, ex, eav, loss, sig2):
    """G4UniversalFluctuation::AddExcitation, vectorised; returns the updated (eav, loss, sig2)."""
    cont = ax > _NMAXCONT
    eav = eav + np.where(cont, ax * ex, 0.0)
    sig2 = sig2 + np.where(cont, ax * ex * ex, 0.0)
    k = rng.poisson(np.where(cont | (ax <= 0), 0.0, ax))
    loss = loss + np.where(k > 0, ((k + 1) - 2 * rng.random(ax.shape)) * ex, 0.0)
    return eav, loss, sig2


def sample_loss(rng, mat, p, length, tcut, mass=MMU):
    """Sampled restricted energy loss [MeV] over `length` [mm] (arrays), Geant4 Urban fluctuation model."""
    p = np.asarray(p, float)
    length = np.asarray(length, float) * np.ones_like(p)
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    cut = np.minimum(tcut, tmax)
    mean = dedx_restricted(mat, p, tcut, mass) * length
    ne = mat.electron_density_per_mm3
    out = mean.copy()
    # Gaussian/Gamma regime: many collisions, cut close to tmax
    bohr = (mean >= _BOHR_MIN * cut) & (tmax <= 2 * cut)
    siga = np.sqrt(np.clip((tmax / beta2 - 0.5 * cut) * TWOPI_MC2_RCL2 * length * ne, 0, None))
    if bohr.any():
        sn = mean / np.where(siga > 0, siga, 1)
        gs = bohr & (sn >= 2)
        if gs.any():
            out[gs] = _sample_gauss_clipped(rng, mean[gs], siga[gs] ** 2)
        gm = bohr & (sn < 2)
        if gm.any():
            neff = sn[gm] ** 2
            out[gm] = mean[gm] * rng.gamma(neff, 1.0) / neff
    # Urban model for the rest
    f = mat.fluct_params()
    e0, I = f["e0"], f["I"]
    u = ~bohr & (mean >= _MINLOSS) & (cut > e0)
    if u.any():
        scaling = np.minimum(1 + 0.5e-3 / cut[u], 1.5)
        ml = mean[u] / scaling
        tc = cut[u]
        w2 = np.where(tc > I, np.log(2 * ME * beta2[u] * gam[u] ** 2) - beta2[u], 0.0)
        a1 = np.zeros_like(ml)
        a2 = np.zeros_like(ml)
        e1 = np.full_like(ml, f["e1"])
        e2 = f["e2"]
        lnI, lne1, lne2 = np.log(I), np.log(f["e1"]), np.log(e2)
        ok = (tc > I) & (w2 > lnI)
        full = ok & (w2 > lne2)
        C = ml * (1 - _RATE) / np.where(w2 > lnI, w2 - lnI, 1)
        a1 = np.where(full, C * f["f1"] * (w2 - lne1) / f["e1"], np.where(ok, ml * (1 - _RATE) / e1, 0.0))
        a2 = np.where(full, C * f["f2"] * (w2 - lne2) / e2, 0.0)
        small = ok & (a1 < _A0)
        fwnow = np.where(small, 0.5 + (_FW - 0.5) * np.sqrt(np.clip(a1, 0, None) / _A0), _FW)
        a1 = np.where(ok, a1 / fwnow, a1)
        e1 = np.where(ok, e1 * fwnow, e1)
        w1 = tc / e0
        a3 = _RATE * ml * (tc - e0) / (e0 * tc * np.log(w1))
        a3 = np.where(a1 + a2 <= 0, a3 / _RATE, a3)
        loss = np.zeros_like(ml)
        eav = np.zeros_like(ml)
        s2 = np.zeros_like(ml)
        eav, loss, s2 = _add_excitation(rng, a1, e1, eav, loss, s2)
        eav, loss, s2 = _add_excitation(rng, a2, np.full_like(ml, e2), eav, loss, s2)
        loss = loss + np.where(s2 > 0, _sample_gauss_clipped(rng, eav, s2), 0.0)
        # ionisation part
        alfa = np.ones_like(ml)
        p3 = a3.copy()
        cont = a3 > _NMAXCONT
        alfa = np.where(cont, w1 * (_NMAXCONT + a3) / (w1 * _NMAXCONT + a3), 1.0)
        safe = np.where(cont, alfa, 2.0)
        alfa1 = np.where(cont, safe * np.log(safe) / (safe - 1), 0.0)
        namean = np.where(cont, a3 * w1 * (alfa - 1) / ((w1 - 1) * np.where(cont, alfa, 1)), 0.0)
        eav3 = namean * e0 * alfa1
        s23 = e0 * e0 * namean * (alfa - alfa1 * alfa1)
        p3 = p3 - namean
        w3 = alfa * e0
        has = tc > w3
        w = np.where(has, (tc - w3) / tc, 0.0)
        nnb = rng.poisson(np.where(has, np.clip(p3, 0, None), 0.0))
        idx = np.repeat(np.arange(len(ml)), nnb)
        r = rng.random(idx.size)
        loss = loss + np.bincount(idx, weights=w3[idx] / (1 - w[idx] * r), minlength=len(ml))
        loss = loss + np.where(s23 > 0, _sample_gauss_clipped(rng, eav3, np.where(s23 > 0, s23, 0.0)), 0.0)
        out[u] = loss * scaling
    return out


def birks(mat, edep, length):
    """Visible energy for a charged track depositing edep [MeV] over length [mm] (G4EmSaturation, no NIEL)."""
    kb = mat.birks_mm_per_MeV
    if kb <= 0:
        return edep
    return np.where(length > 0, edep / (1 + kb * edep / np.where(length > 0, length, 1)), edep)


def delta_rate_per_mm(mat, p, tcut, mass=MMU):
    """Mean number of delta electrons with T in [tcut, tmax] per mm (spin-1/2 cross-section integral)."""
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    tot = tkin + mass
    ok = tmax > tcut
    t1, t2 = tcut, np.where(ok, tmax, 2 * tcut)
    n = (1 / t1 - 1 / t2) - beta2 / tmax * np.log(t2 / t1) + (t2 - t1) / (2 * tot * tot)
    return np.where(ok, TWOPI_MC2_RCL2 * mat.electron_density_per_mm3 / beta2 * n, 0.0)


def sample_deltas(rng, mat, p, length, tcut, mass=MMU):
    """Delta electrons above tcut along `length` [mm]: returns (track index, kinetic energy [MeV]) of each delta.

    Energies from 1/T^2 sampling with the spin-1/2 rejection weight (1 - beta^2 T/Tmax + T^2/(2E^2)), as in
    G4MuBetheBlochModel::SampleSecondaries (radiative correction neglected, below 1 % here).
    """
    p = np.asarray(p, float)
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    tot = tkin + mass
    lam = delta_rate_per_mm(mat, p, tcut, mass) * np.asarray(length, float)
    n = rng.poisson(lam)
    idx = np.repeat(np.arange(p.size), n)
    T = np.empty(idx.size)
    todo = np.ones(idx.size, bool)
    while todo.any():
        k = np.flatnonzero(todo)
        tm = tmax[idx[k]]
        r = rng.random(k.size)
        t = tcut * tm / (tm * (1 - r) + tcut * r)
        g = 1 - beta2[idx[k]] * t / tm + t * t / (2 * tot[idx[k]] ** 2)
        acc = rng.random(k.size) * (1 + tm * tm / (2 * tot[idx[k]] ** 2)) <= g
        T[k[acc]] = t[acc]
        todo[k[acc]] = False
    return idx, T


def radiation_length_mm(mat):
    """X0 [mm] from the composition (Dahl's per-element formula, as in the PDG review), rho in g/cm3."""
    inv = 0.0
    for e, w in mat.fractions.items():
        z, a = ELEMENT_ZA[e]
        lrad, lradp = (np.log(184.15 * z ** (-1 / 3)), np.log(1194.0 * z ** (-2 / 3))) if z > 4 else _LRAD_LIGHT[z]
        az = (ALPHA * z) ** 2
        fz = az * (1 / (1 + az) + 0.20206 - 0.0369 * az + 0.0083 * az ** 2 - 0.002 * az ** 3)
        x0 = 716.408 * a / (z * z * (lrad - fz) + z * lradp)   # g/cm2
        inv += w / x0
    return 10.0 / (inv * mat.rho)


def highland_theta0(p, x_over_x0, mass=MMU):
    """Highland/Lynch-Dahl plane-projected RMS scattering angle [rad] for a muon of |p| [MeV/c]."""
    e = np.sqrt(p * p + mass * mass)
    beta = p / e
    xx = np.maximum(x_over_x0, 1e-12)
    return 13.6 / (beta * p) * np.sqrt(xx) * (1 + 0.038 * np.log(xx / beta ** 2))


ALPHA = 1 / 137.035999
_LRAD_LIGHT = {1: (5.31, 6.144), 2: (4.79, 5.621), 3: (4.74, 5.805), 4: (4.71, 5.924)}
