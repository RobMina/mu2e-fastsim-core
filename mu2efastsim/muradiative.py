"""Radiative energy loss of muons: bremsstrahlung, e+e- pair production and photonuclear interaction (opt-in).

Differential cross sections per atom, as in Geant4 11.3:
  * bremsstrahlung: G4MuBremsstrahlungModel::ComputeDMicroscopicCrossSection (Kelner, Kokoulin, Petrukhin; nuclear
    and atomic-electron screening logarithms, nuclear size dn = 1.54 A^0.27);
  * pair production: G4MuPairProductionModel::ComputeDMicroscopicCrossSection (Kokoulin; 8-point Gauss integration
    over the pair asymmetry in ln(1 - rho), electron and muon terms, Z (Z + zeta) with the atomic-electron zeta);
  * photonuclear: G4KokoulinMuonNuclearXS::ComputeDDMicroscopicCrossSection (Borog-Petrukhin, shadowing
    A_eff = 0.22 A + 0.78 A^0.89), transfers above 0.2 GeV only, as Geant4.

`RadiativeTable` integrates them per material on a kinetic-energy grid (1 GeV to 100 TeV): the mean loss per mm with
fractional transfer v = eps / E below V_CUT (applied as a continuous loss), the rate per mm above it and the inverse
cumulative distribution of v there. `sample_loss` draws the discrete transfers (Poisson count, v by inverse CDF at an
energy node chosen with the linear interpolation weight in ln E) and adds the continuous part. Below 1 GeV kinetic the
loss is zero (Geant4's lowest model energies are 1 GeV for bremsstrahlung and 0.85 GeV for pairs; radiative losses
are below 0.1 % of ionisation there). Nothing here is used unless asked for (transport.slab_loss radiative=True).
"""
import numpy as np

from .eloss import ME, MMU
from .materials import ELEMENTS, N_A

ALPHA = 1 / 137.035999
RE_MM = 2.8179403262e-12           # classical electron radius [mm]
MICROBARN_MM2 = 1e-28
MP = 938.27208816                   # proton mass [MeV]
SQRTE = float(np.sqrt(np.e))
TK_MIN = 1000.0                     # MeV: no radiative loss below this kinetic energy
V_CUT = 1e-4                        # fractional transfer below which the loss is continuous (mean)
NUC_EPS_MIN = 200.0                 # MeV: G4KokoulinMuonNuclearXS CutFixed
_GX, _GW = np.polynomial.legendre.leggauss(8)
_GX, _GW = 0.5 * (_GX + 1.0), 0.5 * _GW   # Gauss-Legendre on [0, 1], as xgi/wgi of the pair model


def brems_dxs(E, eps, Z, A, mass=MMU):
    """d sigma / d eps [mm2 / MeV] per atom for a muon of total energy E [MeV] emitting a photon of energy eps."""
    E, eps = np.broadcast_arrays(np.asarray(E, float), np.asarray(eps, float))
    rmass = mass / ME
    coeff = 16.0 * ALPHA * (RE_MM / rmass) ** 2 / 3.0
    iz = int(round(Z))
    z13 = 1.0 / iz ** (1 / 3)                      # Z^-1/3
    dn = 1.54 * A ** 0.27
    dnstar = dn if iz == 1 else dn / dn ** (1.0 / iz)
    b, b1 = (202.4, 446.0) if iz == 1 else (183.0, 1429.0)
    ok = (eps < E - mass) & (eps > 0)
    v = np.where(ok, eps / E, 0.5)
    delta = 0.5 * mass * mass * v / np.where(ok, E - eps, 1.0)
    rab0 = delta * SQRTE
    rab1 = b * z13
    fn = np.log(rab1 / (dnstar * (ME + rab0 * rab1)) * (mass + delta * (dnstar * SQRTE - 2.0)))
    fn = np.maximum(fn, 0.0)
    epmax1 = E / (1.0 + 0.5 * mass * rmass / E)
    rab2 = b1 * z13 * z13
    fe = np.log(rab2 * mass / ((1.0 + delta * rmass / (ME * SQRTE)) * (ME + rab0 * rab2)))
    fe = np.where(eps < epmax1, np.maximum(fe, 0.0), 0.0)
    x = 1.0 - v + 0.75 * v * v
    return np.where(ok, coeff * x * Z * (Z * fn + fe) / np.where(ok, eps, 1.0), 0.0)


def pair_dxs(E, eps, Z, mass=MMU):
    """d sigma / d eps [mm2 / MeV] per atom for a muon of total energy E [MeV] producing an e+e- pair of energy eps."""
    E, eps = np.broadcast_arrays(np.asarray(E, float), np.asarray(eps, float))
    z13 = Z ** (1 / 3)
    z23 = z13 * z13
    resid = E - eps
    ok = (eps > 4.0 * ME) & (resid > 0.75 * SQRTE * z13 * mass)
    eps_ = np.where(ok, eps, 1.0)
    resid_ = np.where(ok, resid, 1.0)
    a0 = 1.0 / (E * resid_)
    alf = 4.0 * ME / eps_
    rt = np.sqrt(np.clip(1.0 - alf, 0.0, None))
    delta = 6.0 * mass * mass * a0
    tmnexp = alf / (1.0 + rt) + delta * rt
    ok &= tmnexp < 1.0
    tmn = np.log(np.where(ok, tmnexp, 0.5))
    mr2 = (mass / ME) ** 2
    if Z < 1.5:
        bbb, g1, g2 = 202.4, 4.4e-5, 4.8e-5
    else:
        bbb, g1, g2 = 183.0, 1.95e-5, 5.3e-5
    z1exp = E / (mass + g1 * z23 * E)
    z2exp = E / (mass + g2 * z13 * E)
    zeta = np.where(z1exp > 35.221047195922, (0.073 * np.log(z1exp) - 0.26) / (0.058 * np.log(z2exp) - 0.14), 0.0)
    z2 = Z * (Z + zeta)
    screen0 = 2.0 * ME * SQRTE * bbb / (z13 * eps_)
    beta = 0.5 * eps_ * eps_ * a0
    xi0 = 0.5 * mr2 * beta
    total = np.zeros_like(E)
    for xg, wg in zip(_GX, _GW):
        rho = np.exp(tmn * xg) - 1.0
        rho2 = rho * rho
        xi = xi0 * (1.0 - rho2)
        xi = np.maximum(xi, 1e-300)
        xi1 = 1.0 + xi
        xii = 1.0 / xi
        b40, b62 = 4.0 * beta, 6.0 * beta + 2.0
        ye1 = 1.0 + ((b40 + 5.0) + (b40 - 1.0) * rho2) / (b62 * np.log(3.0 + xii) + (2.0 * beta - 1.0) * rho2 - b40)
        ym1 = 1.0 + (b62 * (1.0 + rho2) + 6.0) / ((b40 + 3.0) * (1.0 + rho2) * np.log(3.0 + xi) + 2.0 - 3.0 * rho2)
        be = np.where(xi <= 1000.0,
                      ((2.0 + rho2) * (1.0 + beta) + xi * (3.0 + rho2)) * np.log1p(xii)
                      + (1.0 - rho2 - beta) / xi1 - (3.0 + rho2),
                      0.5 * (3.0 - rho2 + 2.0 * beta * (1.0 + rho2)) * xii)
        a10 = (1.0 + 2.0 * beta) * (1.0 - rho2)
        bm = np.where(xi >= 0.001,
                      ((1.0 + rho2) * (1.0 + 1.5 * beta) + a10 * xii) * np.log(xi1) + xi * (1.0 - rho2 - beta) / xi1 + a10,
                      0.5 * (5.0 - rho2 + beta * (3.0 + rho2)) * xi)
        screen = screen0 * xi1 / (1.0 - rho2)
        ale = np.log(bbb / z13 * np.sqrt(xi1 * ye1) / (1.0 + screen * ye1))
        cre = 0.5 * np.log(1.0 + 2.25 * z23 * xi1 * ye1 / mr2)
        fe = np.maximum((ale - cre) * be, 0.0)
        fm = np.maximum(np.log(bbb * (mass / ME) / (1.5 * z23 * (1.0 + screen * ym1))), 0.0) * bm / mr2
        total = total + wg * (1.0 + rho) * (fe + fm)
    factor = 4.0 * ALPHA ** 2 * RE_MM ** 2 / (3.0 * np.pi)
    out = -tmn * total * factor * z2 * resid_ / (E * eps_)
    return np.where(ok, np.maximum(out, 0.0), 0.0)


def nuclear_dxs(E, eps, A, mass=MMU):
    """d sigma / d eps [mm2 / MeV] per nucleus of mass number A for photonuclear transfer eps (> 0.2 GeV)."""
    E, eps = np.broadcast_arrays(np.asarray(E, float), np.asarray(eps, float))
    alam2, alam = 0.400e6, 632.456          # MeV^2, MeV
    ok = (eps > NUC_EPS_MIN) & (eps < E - 0.5 * MP)
    e_ = np.where(ok, eps, 1e3)
    ep = e_ / 1e3
    aeff = 0.22 * A + 0.78 * A ** 0.89
    sigph = (49.2 + 11.1 * np.log(ep) + 151.8 / np.sqrt(ep)) * MICROBARN_MM2
    v = e_ / E
    v1 = np.where(ok, 1.0 - v, 0.5)
    v2 = v * v
    m2 = mass * mass
    up = E * E * v1 / m2 * (1.0 + m2 * v2 / (alam2 * v1))
    down = 1.0 + e_ / alam * (1.0 + alam / (2.0 * MP) + e_ / alam)
    d = ALPHA / np.pi * aeff * sigph / e_ * (-v1 + (v1 + 0.5 * v2 * (1.0 + 2.0 * m2 / alam2)) * np.log(up / down))
    return np.where(ok, np.maximum(d, 0.0), 0.0)


def atoms_per_mm3(mat):
    """{element: atoms per mm3} of a mu2efastsim Material."""
    return {e: mat.rho * w / ELEMENTS[e][1] * N_A * 1e-3 for e, w in mat.fractions.items()}


def dsigma_dv_per_mm(mat, E, v, mass=MMU, processes=("brems", "pair", "nuclear")):
    """Macroscopic d Sigma / d v [1/mm] (v = eps / E) of the material, summed over elements and the processes."""
    E = np.asarray(E, float)
    eps = v * E
    out = np.zeros(np.broadcast(E, v).shape)
    for e, n in atoms_per_mm3(mat).items():
        Z, A = ELEMENTS[e]
        if "brems" in processes:
            out = out + n * brems_dxs(E, eps, Z, A, mass) * E
        if "pair" in processes:
            out = out + n * pair_dxs(E, eps, Z, mass) * E
        if "nuclear" in processes:
            out = out + n * nuclear_dxs(E, eps, A, mass) * E
    return out


class RadiativeTable:
    """Per-material table on a kinetic-energy grid: continuous mean loss below V_CUT [MeV/mm], rate above [1/mm],
    inverse CDF of v above V_CUT on N_U quantiles."""
    N_U = 257

    def __init__(self, mat, mass=MMU, tk_grid=None, n_v=400, processes=("brems", "pair", "nuclear")):
        self.mass = mass
        self.tk = np.geomspace(TK_MIN, 1e8, 51) if tk_grid is None else np.asarray(tk_grid, float)
        self.lnE = np.log(self.tk + mass)
        nE = len(self.tk)
        self.cont = np.zeros(nE)
        self.rate = np.zeros(nE)
        self.b = np.zeros(nE)                      # total mean loss / E [1/mm]
        self.vq = np.zeros((nE, self.N_U))
        u = np.linspace(0.0, 1.0, self.N_U)
        for i, tk in enumerate(self.tk):
            E = tk + mass
            lv = np.linspace(np.log(1e-7), np.log(1.0 - 1e-9), n_v)
            v = np.exp(lv)
            s = dsigma_dv_per_mm(mat, E, v, mass, processes)
            vs = v * s                             # dSigma / d ln v
            lo = v < V_CUT
            # continuous: integral of eps dSigma = E * integral of v^2 s d ln v below the cut
            self.cont[i] = E * np.trapezoid((v * vs)[lo], lv[lo]) if lo.sum() > 1 else 0.0
            self.b[i] = np.trapezoid(v * vs, lv)
            hi = ~lo
            c = np.concatenate([[0.0], np.cumsum(0.5 * (vs[hi][1:] + vs[hi][:-1]) * np.diff(lv[hi]))])
            self.rate[i] = c[-1]
            if c[-1] > 0:
                self.vq[i] = np.exp(np.interp(u * c[-1], c, lv[hi]))
            else:
                self.vq[i] = V_CUT

    def sample(self, rng, p, length):
        """Radiative loss [MeV] of muons of |p| [MeV/c] over `length` [mm] (arrays), at the entry energy."""
        p = np.asarray(p, float)
        length = np.asarray(length, float) * np.ones_like(p)
        E = np.sqrt(p * p + self.mass * self.mass)
        tk = E - self.mass
        on = tk >= self.tk[0]
        lnE = np.log(np.clip(E, np.exp(self.lnE[0]), np.exp(self.lnE[-1])))
        j = np.clip(np.searchsorted(self.lnE, lnE) - 1, 0, len(self.lnE) - 2)
        w = (lnE - self.lnE[j]) / (self.lnE[j + 1] - self.lnE[j])
        cont = np.where(on, ((1 - w) * self.cont[j] + w * self.cont[j + 1]) * length, 0.0)
        lam = np.where(on, ((1 - w) * self.rate[j] + w * self.rate[j + 1]) * length, 0.0)
        n = rng.poisson(lam)
        loss = cont
        if n.any():
            idx = np.repeat(np.arange(p.size), n)
            node = j[idx] + (rng.random(idx.size) < w[idx])
            uq = rng.random(idx.size) * (self.N_U - 1)
            k = np.minimum(uq.astype(np.int64), self.N_U - 2)
            f = uq - k
            v = np.exp((1 - f) * np.log(self.vq[node, k]) + f * np.log(self.vq[node, k + 1]))
            loss = loss + np.bincount(idx, weights=v * E[idx], minlength=p.size)
        return np.minimum(loss, tk)


_TABLES = {}


def table(mat, mass=MMU):
    """Cached RadiativeTable of a material (by name and mass)."""
    k = (mat.name, float(mass))
    if k not in _TABLES:
        _TABLES[k] = RadiativeTable(mat, mass)
    return _TABLES[k]


def sample_loss(rng, mat, p, length, mass=MMU):
    """Sampled radiative loss [MeV] (brems + pair + photonuclear) of muons of |p| over `length` [mm] of `mat`."""
    return table(mat, mass).sample(rng, p, length)
