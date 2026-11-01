"""Electron and positron multiple scattering for the slab stepper: a mixed (class II) scheme on the screened-Rutherford
(Wentzel) elastic cross section that Geant4 11.3.2 uses in EM option 4.

Option 4 scatters e+- with G4WentzelVIModel plus single Coulomb scattering above 100 MeV and with
G4GoudsmitSaundersonMscModel (Mott-corrected screened Rutherford) below. Both rest on the same single-scattering
cross section, which this module samples directly instead of through their condensed-history tables:

  dsigma/du = 2 pi (m_e r_e)^2 / (beta^2 p^2) [Z^2 F_N(u) + Z F_e(u)] R(u) / (u + s)^2,  u = 1 - cos(theta),

with the screening s of G4WentzelOKandVIxSection::SetupTarget for light particles, its exponential nuclear form factor
F_N = 1 / (1 + f_A p^2 u)^2, scattering on atomic electrons up to the angle that transfers the delta cut
(ComputeMaxElectronScattering, larger transfers are Moller/Bhabha deltas), and a Mott factor R(u) on hard
collisions: for e- the fitted Mott/Rutherford ratio of G4ScreeningMottCrossSection (G4MottData coefficients, bundled
in data/mott_G4MottData_11.3.2.npz), for e+ the McKinley-Feshbach form R(u) = 1 - beta^2 u / 2 - (pi alpha Z / 2)
beta sqrt(u / 2) (2 - u) of its single-scattering sampler.

Mixed scheme (as PENELOPE): collisions with u > u_c are hard and sampled one by one; u_c is chosen so that the hard mean
free path is C1 times the first transport mean free path lambda_1. The soft collisions of a step of length L add up
to one deflection with <u> = 1 - exp(-L Sigma_1,soft), u drawn from an exponential (a two-dimensional Gaussian in
the small-angle limit); the slab stepper applies it at a uniformly random point of the step (random hinge).
Energies in MeV, lengths in mm.
"""
import argparse
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import ebrem
from .eloss import ME
from .materials import ELEMENTS

ALPHA = 1 / 137.035999
COEFF = 2 * np.pi * (ME * ebrem.RE_MM) ** 2      # MeV2 mm2
SCREEN_A0 = 0.5 * ALPHA ** 2 * (ME / 0.88534) ** 2   # ScreenRSquareElec / Z^(2/3) [MeV2]
FORM_FACTOR = 6.937e-6                           # MeV-2, times A^0.54 (G4WentzelOKandVIxSection::InitialiseA)
MF_MAX = 0.7698003589195010                      # max of sqrt(u/2)(2 - u) on [0, 2] (u = 2/3)
C1 = 0.05                                        # hard mean free path / lambda_1
STEP_LAMBDA1 = 0.1                               # step limit in units of lambda_1
U_MAX = 2.0
MOTT = "table"         # e- hard collisions: "table" (G4ScreeningMottCrossSection fit), "mckinley", "none" (diagnostic)
MOTT_DATA = Path(__file__).resolve().parent / "data" / "mott_G4MottData_11.3.2.npz"
MOTT_BETA_SHIFT = 0.7181228                       # G4ScreeningMottCrossSection::RatioMottRutherfordCosT
U_FLOOR = 1e-14                                  # u_c when every elastic collision is hard
_TGRID = np.geomspace(1e-3, 1e3, 241)
_LNRATE = np.linspace(-12.0, 20.0, 257)          # ln(hard collisions per mm) grid of the u_c table


_CACHE = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def elements(mat):
    """(Z, atoms per mm3, A) arrays of the material's elements."""
    return _cached(("elements", mat.name), lambda: _elements(mat))


def _elements(mat):
    rows = [(ELEMENTS[e][0], mat.rho * 6.02214076e23 * w / ELEMENTS[e][1] * 1e-3, ELEMENTS[e][1])
            for e, w in mat.fractions.items()]
    z, n, a = (np.array(c, float) for c in zip(*rows))
    return z, n, a


def kinematics(T):
    """(p^2 [MeV2], 1/beta^2, tau) of e+- with kinetic energy T."""
    T = np.asarray(T, float)
    mom2 = T * (T + 2 * ME)
    return mom2, (T + ME) ** 2 / mom2, T / ME


def screening(z, T):
    """Screening s (in u = 1 - cos theta) per element (rows) and energy (columns), light-particle branch of
    G4WentzelOKandVIxSection::SetupTarget."""
    mom2, invb2, tau = kinematics(T)
    z = np.asarray(z, float)[:, None]
    z23 = z ** (2.0 / 3.0)
    f = np.minimum(1.13 * z, 1.13 + 3.76 * z * z * invb2 * ALPHA ** 2 * np.sqrt(tau / (tau + z23)))
    return f * SCREEN_A0 * z23 / mom2


def u_electron(T, tcut, positron=False):
    """Largest u of scattering on atomic electrons: the angle at which the transfer reaches min(tcut, tmax)
    (ComputeMaxElectronScattering, light particles)."""
    T = np.asarray(T, float)
    mom2 = T * (T + 2 * ME)
    t = np.minimum(tcut, T if positron else 0.5 * T)
    mom21 = t * (t + 2 * ME)
    t1 = T - t
    mom22 = t1 * (t1 + 2 * ME)
    ctm = (mom2 + mom22 - mom21) * 0.5 / np.sqrt(np.maximum(mom2 * mom22, 1e-300))
    ctm = np.where(ctm < 1.0, ctm, 1.0)
    if not positron:
        ctm = np.maximum(ctm, 0.0)
    return np.clip(1.0 - ctm, 0.0, U_MAX)


def _g(x):
    """integral_0^x v / (1 + v)^2 dv = ln(1 + x) - x / (1 + x)."""
    x = np.asarray(x, float)
    small = x < 1e-3
    xs = np.where(small, x, 0.0)
    return np.where(small, xs * xs * (0.5 - xs * (2.0 / 3.0 - 0.75 * xs)), np.log1p(x) - x / (1 + x))


def _parts(mat, T, tcut, positron):
    z, n, a = elements(mat)
    mom2, invb2, _ = kinematics(T)
    s = screening(z, T)
    k = n[:, None] * COEFF * z[:, None] * invb2 / mom2          # per element: n K_i / Z_i ... times Z or 1 below
    return z, a, s, k, u_electron(T, tcut, positron)


def transport_xsec(mat, T, uc, tcut, positron=False):
    """Sigma_1 [mm-1] of the collisions with u < uc (uc >= 2: the full first transport cross section)."""
    T = np.atleast_1d(np.asarray(T, float))
    z, a, s, k, ue = _parts(mat, T, tcut, positron)
    uc = np.minimum(np.broadcast_to(uc, T.shape), U_MAX)
    nuc = z[:, None] * _g(uc / s)
    ele = _g(np.minimum(uc, ue) / s)
    return (k * (nuc + ele)).sum(axis=0)


def _hard_rates(mat, T, uc, tcut, positron):
    """Envelope rates [mm-1] of hard collisions (u > uc) on nuclei and on electrons, per element: (nuc, ele, s, ue)."""
    z, a, s, k, ue = _parts(mat, T, tcut, positron)
    uc = np.broadcast_to(uc, T.shape)
    nuc = k * z[:, None] * (1 / (uc + s) - 1 / (U_MAX + s))
    ele = np.where(ue > uc, k * (1 / (uc + s) - 1 / (ue + s)), 0.0)
    env = mott_envelope(z, T, positron)
    return nuc * env, ele * env, s, ue


def hard_xsec(mat, T, uc, tcut, positron=False):
    """Sampling rate [mm-1] of hard collisions above uc: screened Rutherford times the Mott envelope; accepted
    collisions (sample_hard) occur at the physical rate."""
    T = np.atleast_1d(np.asarray(T, float))
    nuc, ele, _, _ = _hard_rates(mat, T, uc, tcut, positron)
    return (nuc + ele).sum(axis=0)


def convert_mott(header, out):
    """Write the npz of the electron Mott/Rutherford fit coefficients fMottCoef[93][5][6] from a Geant4
    G4MottData.hh (values unchanged; SHA-256 of the header inside)."""
    text = Path(header).read_text()
    block = text[text.index("fMottCoef[93][5][6]"):]
    block = block[block.index("{"):block.index("};") + 1]
    vals = np.array([float(v) for v in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", block)])
    if vals.size != 93 * 5 * 6:
        raise ValueError(f"{header}: {vals.size} numbers in fMottCoef, expected {93 * 5 * 6}")
    np.savez_compressed(out, coef=vals.reshape(93, 5, 6), sha256=np.array(hashlib.sha256(text.encode()).hexdigest()),
                        source=np.array("Geant4 11.3.2 G4MottData.hh fMottCoef"))


@lru_cache(maxsize=None)
def mott_coefficients(path=MOTT_DATA):
    """fMottCoef[Z][j][k] (Z = 0..92) of the bundled npz."""
    return np.load(path)["coef"]


def mott_ratio_table(z, u, beta):
    """Mott/Rutherford ratio of electrons, sum_j sqrt(u)^j sum_k a[Z, j, k] (beta - 0.7181228)^k
    (G4ScreeningMottCrossSection::RatioMottRutherfordCosT, Boschini et al. / Lijian et al. fit)."""
    a = mott_coefficients()[np.asarray(z, int)]                 # (..., 5, 6)
    b0 = np.asarray(beta, float) - MOTT_BETA_SHIFT
    bk = np.stack([b0 ** k for k in range(6)], axis=-1)        # (..., 6)
    aj = (a * bk[..., None, :]).sum(axis=-1)                   # (..., 5)
    su = np.sqrt(np.asarray(u, float))
    return sum(aj[..., j] * su ** j for j in range(5))


_BETA_GRID = np.linspace(0.0, 1.0, 1001)


@lru_cache(maxsize=None)
def _mott_env_table(z):
    """Max over u in [0, 2] of the tabulated ratio for element z on _BETA_GRID (each bin's max with its neighbours)."""
    su = np.linspace(0.0, np.sqrt(U_MAX), 129)
    r = mott_ratio_table(np.full((_BETA_GRID.size, su.size), z), (su ** 2)[None, :], _BETA_GRID[:, None])
    m = r.max(axis=1)
    return np.maximum(np.maximum(m, np.roll(m, 1)), np.roll(m, -1))


def mott_envelope(z, T, positron=False):
    """Upper bound of the Mott factor R(u) per element and energy (rows, columns)."""
    _, invb2, _ = kinematics(T)
    z = np.asarray(z, float)[:, None]
    if MOTT == "none" or positron:
        return np.ones((z.shape[0], np.size(T)))       # e+: the McKinley-Feshbach factor is below 1
    if MOTT == "mckinley":
        return 1.0 + 0.5 * np.pi * ALPHA * z * np.sqrt(1 / invb2) * MF_MAX
    beta = np.sqrt(1 / np.atleast_1d(invb2))
    return 1.02 * np.array([np.interp(beta, _BETA_GRID, _mott_env_table(int(zz))) for zz in z[:, 0]])


def mott_factor(z, u, invb2, positron=False):
    """Ratio to Rutherford of hard collisions: the tabulated electron fit (MOTT = "table"), or the McKinley-Feshbach
    form of G4WentzelOKandVIxSection::SampleSingleScattering (e+ always; sign of the alpha Z term by charge)."""
    if MOTT == "none":
        return np.ones_like(np.asarray(u, float))
    b2 = 1 / invb2
    if MOTT == "table" and not positron:
        return np.maximum(mott_ratio_table(z, u, np.sqrt(b2)), 0.0)
    sign = -1.0 if positron else 1.0
    return 1 - 0.5 * b2 * u + sign * 0.5 * np.pi * ALPHA * z * np.sqrt(0.5 * b2 * u) * (2 - u)


def _uc_table(mat, tcut, positron):
    return _cached(("uc", mat.name, tcut, positron), lambda: _build_uc_table(mat, tcut, positron))


def _build_uc_table(mat, tcut, positron):
    """ln(u_c) on (_TGRID, _LNRATE): the u_c at which the Rutherford rate of collisions above it is exp(lnrate)
    per mm, floored at U_FLOOR where every collision is needed; also ln Sigma_1 on _TGRID."""
    T = _TGRID
    z, a, s, k, ue = _parts(mat, T, tcut, positron)
    target = np.exp(_LNRATE)[:, None]

    def rate(uc):
        nuc = k[:, None] * z[:, None, None] * (1 / (uc + s[:, None]) - 1 / (U_MAX + s[:, None]))
        ele = np.where(ue > uc, k[:, None] * (1 / (uc + s[:, None]) - 1 / (ue + s[:, None])), 0.0)
        return (nuc + ele).sum(axis=0)

    shape = (_LNRATE.size, T.size)
    lo, hi = np.full(shape, np.log(U_FLOOR)), np.full(shape, np.log(U_MAX))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        big = rate(np.exp(mid)) > target
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    return 0.5 * (lo + hi), np.log(transport_xsec(mat, T, U_MAX, tcut, positron))


def u_cut(mat, T, tcut, positron=False, max_hard_path=np.inf):
    """Soft/hard boundary u_c at kinetic energy T: hard collisions every C1 lambda_1, or every max_hard_path [mm]
    if that is shorter (bilinear in ln T and ln rate)."""
    tab, lsig1 = _uc_table(mat, float(tcut), bool(positron))
    lt = np.log(np.clip(np.atleast_1d(np.asarray(T, float)), _TGRID[0], _TGRID[-1]))
    lgrid = np.log(_TGRID)
    lr = np.maximum(np.interp(lt, lgrid, lsig1) - np.log(C1), -np.log(max_hard_path))
    lr = np.clip(lr, _LNRATE[0], _LNRATE[-1])
    fi = (lt - lgrid[0]) / (lgrid[1] - lgrid[0])
    fj = (lr - _LNRATE[0]) / (_LNRATE[1] - _LNRATE[0])
    i = np.clip(fi.astype(int), 0, lgrid.size - 2)
    j = np.clip(fj.astype(int), 0, _LNRATE.size - 2)
    x, y = fi - i, fj - j
    v = ((1 - y) * ((1 - x) * tab[j, i] + x * tab[j, i + 1]) + y * ((1 - x) * tab[j + 1, i] + x * tab[j + 1, i + 1]))
    return np.exp(v)


def lambda1(mat, T, tcut, positron=False):
    """First transport mean free path [mm]."""
    return 1.0 / transport_xsec(mat, T, U_MAX, tcut, positron)


def sample_soft(rng, mat, T, length, uc, tcut, positron=False):
    """u = 1 - cos(theta) of the summed soft collisions over `length` (exponential with mean 1 - exp(-L Sigma_1s))."""
    T = np.atleast_1d(np.asarray(T, float))
    m1 = -np.expm1(-np.asarray(length, float) * transport_xsec(mat, T, uc, tcut, positron))
    return np.minimum(rng.exponential(1.0, T.size) * m1, U_MAX)


def sample_hard(rng, mat, T, uc, tcut, positron=False):
    """One hard-collision trial per track at energy T with boundary uc: (u, accepted). The trial picks element and
    target (nucleus or electrons) by envelope rate, u from 1 / (u + s)^2 above uc, then accepts with
    R(u) F_N(u) / envelope."""
    T = np.atleast_1d(np.asarray(T, float))
    uc = np.broadcast_to(uc, T.shape)
    nuc, ele, s, ue = _hard_rates(mat, T, uc, tcut, positron)
    rates = np.concatenate([nuc, ele])                      # rows: nuclei then electrons of each element
    cum = np.cumsum(rates, axis=0)
    pick = np.minimum((rng.random(T.size) * cum[-1] > cum).sum(axis=0), rates.shape[0] - 1)
    nel = nuc.shape[0]
    iel = pick % nel
    on_nuc = pick < nel
    cols = np.arange(T.size)
    si = s[iel, cols]
    umax = np.where(on_nuc, U_MAX, ue)
    w1, w2 = 1 / (uc + si), 1 / (umax + si)
    u = 1 / (w1 - rng.random(T.size) * (w1 - w2)) - si
    u = np.clip(u, uc, umax)
    z, n, a = elements(mat)
    mom2, invb2, _ = kinematics(T)
    ff = np.where(on_nuc, 1 / (1 + FORM_FACTOR * a[iel] ** 0.54 * mom2 * u) ** 2, 1.0)
    env = mott_envelope(z, T, positron)[iel, cols]
    acc = rng.random(T.size) * env <= mott_factor(z[iel], u, invb2, positron) * ff
    return u, acc


def deflect(rng, d, u):
    """New unit directions after polar deflection u = 1 - cos(theta) with a uniform azimuth."""
    cost = 1.0 - u
    sint = np.sqrt(np.clip((1 - cost) * (1 + cost), 0.0, None))
    phi = 2 * np.pi * rng.random(np.shape(u)[0])
    out = ebrem.rotate_uz(d, np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1))
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.emsc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert-mott", help="rebuild the bundled Mott npz from a Geant4 G4MottData.hh")
    c.add_argument("header")
    c.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "convert-mott":
        convert_mott(a.header, a.out)


if __name__ == "__main__":
    _main()
