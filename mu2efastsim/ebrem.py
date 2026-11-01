"""Electron and positron bremsstrahlung below 1 GeV, matched to Geant4 11.3.2 G4SeltzerBergerModel (EM option 4).

Mu2e runs primary-particle jobs (CeEndpoint and the other stopped-muon products) with ShieldingM_EMZ, whose eBrem is
G4SeltzerBergerModel up to 1 GeV with the G4Generator2BS photon angles. The model reads the Seltzer-Berger scaled
cross section chi(Z, kappa, ln T) = (beta^2 / Z^2) k dsigma/dk [mb] from G4EMLOW brem_SB/br<Z>, with kappa = k / T
the photon energy over the kinetic energy, interpolated bilinearly (G4Physics2DVector). This module uses the same
tables, Ter-Mikaelian (dielectric) suppression 1 / (1 + k_p^2 / k^2), the positron factor, Geant4's Gauss-Legendre
integration of the restricted loss and cross section, its rejection sampler for the photon energy and the 2BS photon
angle. Array functions: numpy in, numpy out; energies in MeV, lengths in mm.

The tables are a compact copy of G4EMLOW 8.6.1 (data/brem_sb_G4EMLOW8.6.1.npz, the elements of materials.ELEMENTS);
`python3 -m mu2efastsim.ebrem convert <G4EMLOW>/brem_SB OUT.npz` rebuilds it and records each file's SHA-256.
"""
import argparse
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .materials import ELEMENTS, N_A

ME = 0.51099895                     # MeV
RE_MM = 2.8179403262e-12            # classical electron radius
LAMBDAE_MM = 3.8615926796e-10       # reduced electron Compton wavelength (CLHEP electron_Compton_length)
MB_MM2 = 1e-25                      # millibarn
ALPHA = 1 / 137.035999
MIGDAL = 4 * np.pi * RE_MM * LAMBDAE_MM ** 2      # mm3: k_p^2 = MIGDAL n_e E_tot^2
T_MIN = 1e-3                        # G4SeltzerBergerModel fLowestKinEnergy
T_MAX = 1e3                         # its high-energy limit in option 4; eBremsstrahlungRelModel takes over above
DATA = Path(__file__).resolve().parent / "data" / "brem_sb_G4EMLOW8.6.1.npz"

_XGL = np.array([1.98550718e-02, 1.01666761e-01, 2.37233795e-01, 4.08282679e-01,
                 5.91717321e-01, 7.62766205e-01, 8.98333239e-01, 9.80144928e-01])
_WGL = np.array([5.06142681e-02, 1.11190517e-01, 1.56853323e-01, 1.81341892e-01,
                 1.81341892e-01, 1.56853323e-01, 1.11190517e-01, 5.06142681e-02])
_NCOUNT_MAX = 100                   # SampleEnergyTransfer: accept the last trial after this many


def read_g4emlow(path):
    """One brem_SB/br<Z> file (G4Physics2DVector::Retrieve) -> (kappa nodes, ln T nodes, chi[lnT, kappa])."""
    tok = Path(path).read_text().split()
    nx, ny = int(tok[1]), int(tok[2])
    v = np.array(tok[3:], float)
    return v[:nx], v[nx:nx + ny], v[nx + ny:].reshape(ny, nx)


def convert(src_dir, out, zs=None):
    """Write the npz for elements zs (default: those of materials.ELEMENTS) from a G4EMLOW brem_SB directory."""
    zs = sorted({z for z, _ in ELEMENTS.values()}) if zs is None else sorted(zs)
    grids, chis, shas = None, [], []
    for z in zs:
        f = Path(src_dir) / f"br{z}"
        x, y, c = read_g4emlow(f)
        if grids is None:
            grids = (x, y)
        elif not (np.array_equal(grids[0], x) and np.array_equal(grids[1], y)):
            raise ValueError(f"{f}: grid differs from br{zs[0]}")
        chis.append(c)
        shas.append(hashlib.sha256(f.read_bytes()).hexdigest())
    np.savez_compressed(out, z=np.array(zs), kappa=grids[0], lnT=grids[1], chi=np.array(chis),
                        sha256=np.array(shas), source=np.array("G4EMLOW8.6.1/brem_SB"))


@lru_cache(maxsize=None)
def tables(path=DATA):
    """(z -> row, kappa nodes, ln T nodes, chi[row, lnT, kappa]) from the bundled npz."""
    d = np.load(path)
    return {int(z): i for i, z in enumerate(d["z"])}, d["kappa"], d["lnT"], d["chi"]


def _bin(nodes, v):
    """G4Physics2DVector bin: lower node index in [0, n - 2], v already clamped to the node range."""
    return np.clip(np.searchsorted(nodes, v, side="right") - 1, 0, nodes.size - 2)


def chi(z, kappa, lnT):
    """Bilinear chi [mb] of element z, clamped to the table as G4Physics2DVector::Value (no bicubic)."""
    row, xs, ys, c = tables()
    t = c[row[int(z)]]
    x = np.clip(np.asarray(kappa, float), xs[0], xs[-1])
    y = np.clip(np.asarray(lnT, float), ys[0], ys[-1])
    i, j = _bin(xs, x), _bin(ys, y)
    x1, x2, y1, y2 = xs[i], xs[i + 1], ys[j], ys[j + 1]
    return ((y2 - y) * (t[j, i] * (x2 - x) + t[j, i + 1] * (x - x1))
            + (y - y1) * (t[j + 1, i] * (x2 - x) + t[j + 1, i + 1] * (x - x1))) / ((x2 - x1) * (y2 - y1))


def atom_densities(mat):
    """[(Z, atoms per mm3)] of the material's elements."""
    return [(ELEMENTS[e][0], mat.rho * N_A * w / ELEMENTS[e][1] * 1e-3) for e, w in mat.fractions.items()]


def _check_energy(T):
    if np.any(T > T_MAX):
        raise ValueError("Seltzer-Berger brem is used up to 1 GeV (option 4); above it Geant4 switches to "
                         "G4eBremsstrahlungRelModel, which is not implemented here")


def _kdsdk(z, k, T, positron):
    """k dsigma/dk per atom [mm2] without the Ter-Mikaelian factor (ComputeDXSectionPerAtom times the constant)."""
    lnT = np.log(T)
    pt2 = T * (T + 2 * ME)
    invb2 = (T + ME) ** 2 / pt2
    v = z * z * chi(z, k / T, lnT) * invb2 * MB_MM2
    if positron:
        e2 = T - k
        invbeta1 = np.sqrt(invb2)
        invbeta2 = (e2 + ME) / np.sqrt(np.where(e2 > 0, e2 * (e2 + 2 * ME), 1.0))
        dum0 = 2 * np.pi * ALPHA * z * (invbeta1 - invbeta2)
        v = np.where((e2 > 0) & (dum0 >= -12.0), v * np.exp(np.maximum(dum0, -12.0)), 0.0)
    return v


def _kdsdk_material(mat, k, T, positron):
    """sum_i N_i k dsigma_i/dk / (1 + k_p^2 / k^2) [mm-1]."""
    kp2 = MIGDAL * mat.electron_density_per_mm3 * (T + ME) ** 2
    s = sum(n * _kdsdk(z, k, T, positron) for z, n in atom_densities(mat))
    return s / (1 + kp2 / (k * k))


def dedx_restricted(mat, T, kcut, positron=False):
    """Radiative energy loss [MeV/mm] to photons below kcut (ComputeDEDXPerVolume; kcut >= T gives the total)."""
    T = np.atleast_1d(np.asarray(T, float))
    _check_energy(T)
    etot = T + ME
    tmax = np.minimum(kcut, T)
    amax = tmax / etot
    nsub = (20 * amax).astype(int) + 3
    delta = amax / nsub
    out = np.zeros_like(T)
    live = T > T_MIN
    for sub in range(int(nsub.max(initial=0))):
        on = live & (sub < nsub)
        if not on.any():
            continue
        for x, w in zip(_XGL, _WGL):
            k = (sub + x) * delta[on] * etot[on]
            out[on] += w * _kdsdk_material(mat, k, T[on], positron)
    return np.maximum(out * delta * etot, 0.0)


def _xsec_atom(z, T, kmin, positron, kp2):
    """ComputeXSectionPerAtom(kmin) times the constant: integral of dsigma/dk per atom from kmin to T [mm2], with
    the Ter-Mikaelian factor of the material (kp2)."""
    etot = T + ME
    amin, amax = np.log(kmin / etot), np.log(T / etot)
    nsub = (0.45 * (amax - amin)).astype(int) + 4
    delta = (amax - amin) / nsub
    out = np.zeros_like(T)
    for sub in range(int(nsub.max(initial=0))):
        on = sub < nsub
        for x, w in zip(_XGL, _WGL):
            k = np.exp(amin[on] + (sub + x) * delta[on]) * etot[on]
            out[on] += w * _kdsdk(z, k, T[on], positron) / (1 + kp2[on] / (k * k))
    return np.maximum(out * delta, 0.0)


def _partial_xsec(mat, T, kcut, positron):
    """N_i sigma_i [mm-1] per element (rows) for T > kcut."""
    kp2 = MIGDAL * mat.electron_density_per_mm3 * (T + ME) ** 2
    kmin = np.full(T.shape, float(kcut))
    return np.array([n * _xsec_atom(z, T, kmin, positron, kp2) for z, n in atom_densities(mat)])


def xsec_per_mm(mat, T, kcut, positron=False):
    """Macroscopic cross section [mm-1] for a photon above kcut (ComputeCrossSectionPerAtom, kmax = T)."""
    T = np.atleast_1d(np.asarray(T, float))
    _check_energy(T)
    out = np.zeros_like(T)
    on = (T > T_MIN) & (T > kcut)
    if on.any():
        out[on] = _partial_xsec(mat, T[on], kcut, positron).sum(axis=0)
    return out


def _select_element(rng, mat, T, kcut, positron):
    """SelectTargetAtom: element index per track with probability N_i sigma_i(T, kcut)."""
    if len(mat.fractions) == 1:
        return np.zeros(T.size, int)
    cum = np.cumsum(_partial_xsec(mat, T, kcut, positron), axis=0)
    u = rng.random(T.size) * cum[-1]
    return np.minimum((u[None, :] > cum).sum(axis=0), cum.shape[0] - 1)


def sample_k(rng, mat, T, kcut, positron=False, return_z=False):
    """Photon energies [MeV] above kcut for tracks of kinetic energy T (SelectTargetAtom + SampleEnergyTransfer)."""
    T = np.atleast_1d(np.asarray(T, float))
    _check_energy(T)
    if np.any(T <= kcut):
        raise ValueError("sample_k needs T > kcut")
    zs = np.array([z for z, _ in atom_densities(mat)])
    z = zs[_select_element(rng, mat, T, kcut, positron)]
    lnT = np.log(T)
    kp2 = MIGDAL * mat.electron_density_per_mm3 * (T + ME) ** 2
    xmin = np.log(kcut * kcut + kp2)
    xrange = np.log(T * T + kp2) - xmin
    x0 = kcut / T
    vmax = np.empty_like(T)
    for zz in np.unique(z):
        m = z == zz
        vmax[m] = chi(zz, x0[m], lnT[m]) * 1.02
        lim = (not positron) & (x0[m] < 0.97) & ((T[m] > 300.0) | (T[m] < 0.02))
        if lim.any():
            ylim = np.minimum(chi(zz, 0.97, 4 * np.log(10.0)), 1.1 * chi(zz, 0.97, lnT[m]))
            vmax[m] = np.where(lim, np.maximum(vmax[m], ylim), vmax[m])
    vmax = np.where(x0 < 0.05, vmax * 1.2, vmax)
    if positron:
        e1 = T - kcut
        invbeta1 = (e1 + ME) / np.sqrt(e1 * (e1 + 2 * ME))
    k = np.empty_like(T)
    todo = np.ones(T.size, bool)
    for _ in range(_NCOUNT_MAX):
        idx = np.flatnonzero(todo)
        if idx.size == 0:
            break
        r0, r1 = rng.random(idx.size), rng.random(idx.size)
        g = np.sqrt(np.maximum(np.exp(xmin[idx] + r0 * xrange[idx]) - kp2[idx], 0.0))
        v = np.empty(idx.size)
        for zz in np.unique(z[idx]):
            m = z[idx] == zz
            v[m] = chi(zz, g[m] / T[idx][m], lnT[idx][m])
        if positron:
            e2 = T[idx] - g
            invbeta2 = (e2 + ME) / np.sqrt(np.where(e2 > 0, e2 * (e2 + 2 * ME), 1.0))
            dum0 = 2 * np.pi * ALPHA * z[idx] * (invbeta1[idx] - invbeta2)
            v = np.where((e2 > 0) & (dum0 >= -12.0), v * np.exp(np.maximum(dum0, -12.0)), 0.0)
        k[idx] = g
        acc = v >= vmax[idx] * r1
        todo[idx[acc]] = False
    return (k, z) if return_z else k


def sample_cos_theta(rng, T, k, z):
    """G4Generator2BS: photon polar angle cosine relative to the parent direction (Koch-Motz 2BS, PIRS-0203)."""
    T, k, z = (np.atleast_1d(np.asarray(a, float)) for a in (T, k, z))
    energy = T + ME
    ratio = (energy - k) / energy
    ratio1 = (1 + ratio) ** 2
    ratio2 = 1 + ratio * ratio
    gam = energy / ME
    beta = np.sqrt((gam - 1) * (gam + 1)) / gam
    fz = 0.00008116224 * np.cbrt(z) * np.cbrt(z + 1)
    ymax = 2 * beta * (1 + beta) * gam * gam

    def rej(y, m):
        y2 = (1 + y) ** 2
        x = 4 * y * ratio[m] / y2
        return 4 * x - ratio1[m] - (ratio2[m] - x) * np.log(fz[m] / y2)

    allm = np.ones(T.size, bool)
    gmax = np.maximum(rej(np.zeros(T.size), allm), rej(ymax, allm))
    y = np.empty_like(T)
    todo = np.ones(T.size, bool)
    while todo.any():
        idx = np.flatnonzero(todo)
        q = rng.random(idx.size)
        yy = q * ymax[idx] / (1 + ymax[idx] * (1 - q))
        acc = (rng.random(idx.size) * gmax[idx] <= rej(yy, idx)) & (yy <= ymax[idx])
        y[idx[acc]] = yy[acc]
        todo[idx[acc]] = False
    return 1 - 2 * y / ymax


def rotate_uz(d, local):
    """CLHEP Hep3Vector::rotateUz: local directions (n, 3) in the frame whose z axis is the unit vector d (n, 3)."""
    d = np.asarray(d, float)
    u1, u2, u3 = d[:, 0], d[:, 1], d[:, 2]
    up = u1 * u1 + u2 * u2
    x, y, zl = local[:, 0], local[:, 1], local[:, 2]
    s = np.sqrt(np.where(up > 0, up, 1.0))
    out = np.empty_like(local)
    out[:, 0] = np.where(up > 0, (u1 * u3 * x - u2 * y) / s + u1 * zl, np.where(u3 > 0, x, -x))
    out[:, 1] = np.where(up > 0, (u2 * u3 * x + u1 * y) / s + u2 * zl, np.where(u3 > 0, y, -y))
    out[:, 2] = np.where(up > 0, -s * x + u3 * zl, np.where(u3 > 0, zl, -zl))
    return out


def emit(rng, T, d, k, z):
    """One emission per row: photon directions (2BS) and the parent's new direction (p d - k d_gamma, normalised)."""
    T = np.asarray(T, float)
    cost = sample_cos_theta(rng, T, k, z)
    sint = np.sqrt((1 - cost) * (1 + cost))
    phi = 2 * np.pi * rng.random(T.size)
    dg = rotate_uz(d, np.stack([sint * np.cos(phi), sint * np.sin(phi), cost], axis=1))
    p = np.sqrt(T * (T + 2 * ME))
    de = p[:, None] * d - np.asarray(k, float)[:, None] * dg
    return dg, de / np.linalg.norm(de, axis=1, keepdims=True)


def sample_along(rng, mat, T, length, kcut, positron=False):
    """Discrete photons above kcut along `length` [mm] with no other loss: each free path from sigma at the current
    energy, the energy reduced after each photon. Returns (track index, distance from the start [mm], k [MeV],
    element Z) of every photon and the kinetic energy at the end."""
    T = np.array(np.atleast_1d(T), float)
    full = np.broadcast_to(np.asarray(length, float), T.shape)
    done = np.zeros_like(T)
    out = []
    live = np.flatnonzero((T > kcut) & (full > 0))
    while live.size:
        sig = xsec_per_mm(mat, T[live], kcut, positron)
        s = done[live] + np.where(sig > 0, rng.exponential(1.0, live.size) / np.where(sig > 0, sig, 1.0), np.inf)
        hit = s < full[live]
        idx = live[hit]
        if idx.size:
            k, z = sample_k(rng, mat, T[idx], kcut, positron, return_z=True)
            out.append((idx, s[hit], k, z))
            T[idx] -= k
            done[idx] = s[hit]
        live = idx[T[idx] > kcut]
    if not out:
        e = np.empty(0)
        return e.astype(int), e, e, e.astype(int), T
    idx, s, k, z = (np.concatenate(c) for c in zip(*out))
    return idx, s, k, z, T


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.ebrem")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="rebuild the bundled npz from a G4EMLOW brem_SB directory")
    c.add_argument("src")
    c.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "convert":
        convert(args.src, args.out)


if __name__ == "__main__":
    _main()
