"""Mean muon energy loss of the fast model against reference formulae, for one material: python3 -m mu2efastsim.dedxtable.

Columns per |p| [MeV/c], in MeV cm2/g:
  restricted  eloss.dedx_restricted with no cut (tcut = 1e9): what slab_loss integrates (mean of sample_loss)
  bethe_pdg   PDG eq. 34.5 (Bethe with Tmax, density effect from the same Sternheimer parameters), as a cross-check of
              the restricted formula at tcut = Tmax
  b_rad       radiative loss b(E) E with b = b_brems + b_pair + b_photonuc for standard rock scaled by the
              material's Z^2/A (Groom, Mokhov, Striganov, ADNDT 78 (2001) 183, table II: b ~ 3.5e-6 g-1 cm2 for rock
              at 10-100 GeV; rising as ln E below). Only an estimate, to size the missing piece
  total       bethe_pdg + b_rad
and the ratios restricted / total, so a missing fraction of the total loss is visible at a glance.
"""
import argparse

import numpy as np

from . import eloss, materials

K_PDG = 0.307075        # MeV mol-1 cm2


def _z_over_a(mat):
    return sum(w * materials.ELEMENTS[e][0] / materials.ELEMENTS[e][1] for e, w in mat.fractions.items())


def _z2_over_a(mat):
    return sum(w * materials.ELEMENTS[e][0] ** 2 / materials.ELEMENTS[e][1] for e, w in mat.fractions.items())


def bethe_pdg(mat, p, mass=eloss.MMU):
    """PDG mean dE/dx [MeV cm2/g] (eq. 34.5), unrestricted, density effect as eloss.density_correction."""
    tkin, beta2, bg2, gam, tmax = eloss.kinematics(p, mass)
    I = mat.I_eV * 1e-6
    arg = 2 * eloss.ME * bg2 * tmax / (I * I)
    return K_PDG * _z_over_a(mat) / beta2 * (0.5 * np.log(arg) - beta2 - 0.5 * eloss.density_correction(mat, bg2))


def b_radiative(mat, e):
    """Radiative energy-loss coefficient b(E) [g-1 cm2] estimate: standard rock (Z=11, A=22, Z^2/A = 5.5) values of
    Groom et al. 2001, interpolated in ln E between 1 and 100 GeV, scaled by Z^2/A of the material."""
    e_gev = np.asarray(e, float) / 1e3
    pts_e = np.array([1.0, 3.0, 10.0, 30.0, 100.0])
    pts_b = np.array([1.0e-6, 1.6e-6, 2.4e-6, 3.0e-6, 3.5e-6])         # rock, all radiative processes
    b = np.interp(np.log(np.clip(e_gev, 1e-3, None)), np.log(pts_e), pts_b, left=pts_b[0] * 0.0 + 0.0, right=pts_b[-1])
    b = np.where(e_gev < 1.0, pts_b[0] * np.clip(e_gev, 0, None), b)    # fades below 1 GeV
    return b * _z2_over_a(mat) / 5.5


def table(mat, p):
    e = np.sqrt(p * p + eloss.MMU ** 2)
    rho = mat.rho
    restricted = eloss.dedx_restricted(mat, p, 1e9) * 10.0 / rho      # MeV/mm -> MeV cm2/g
    bethe = bethe_pdg(mat, p)
    rad = b_radiative(mat, e) * e
    return {"p": p, "restricted": restricted, "bethe_pdg": bethe, "b_rad": rad, "total": bethe + rad}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="CONCRETE_MARS", help="materials.BY_NAME key")
    ap.add_argument("--p", default="300,500,1000,2000,3000,5000,10000,20000,50000", help="|p| list [MeV/c]")
    a = ap.parse_args(argv)
    mat = materials.BY_NAME[a.material]
    p = np.array([float(x) for x in a.p.split(",")])
    t = table(mat, p)
    print(f"{mat.name}: rho {mat.rho} g/cm3, I {mat.I_eV:.1f} eV, Z/A {_z_over_a(mat):.4f}, Z2/A {_z2_over_a(mat):.3f}, "
          f"density params {tuple(round(x, 4) for x in mat.density_params())}")
    print(f"{'p MeV/c':>9s} {'restricted':>11s} {'bethe_pdg':>10s} {'b_rad*E':>8s} {'total':>7s} {'restr/bethe':>11s} {'restr/total':>11s}")
    for k in range(len(p)):
        print(f"{p[k]:9.0f} {t['restricted'][k]:11.4f} {t['bethe_pdg'][k]:10.4f} {t['b_rad'][k]:8.4f} {t['total'][k]:7.4f}"
              f" {t['restricted'][k] / t['bethe_pdg'][k]:11.4f} {t['restricted'][k] / t['total'][k]:11.4f}")


if __name__ == "__main__":
    main()
