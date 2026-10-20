"""Material constants for the muon transport, matched to what Geant4 11.3.2 uses for the extracted world.

Sources (read 2026-10-16):
  * composition, density and mean excitation energy I: the v04 GDML export (`<material>` blocks);
  * Sternheimer density-effect parameters: G4DensityEffectData.cc (NIST materials by name);
  * Birks constant: G4EmSaturation::InitialiseG4materials (only G4_POLYSTYRENE has one: 0.07943 mm/MeV; the Mu2e fcl
    BirksConsts override is unset, Mu2eG4/fcl/prolog.fcl);
  * Urban fluctuation parameters: G4IonisParamMat::ComputeFluctModel (from Zeff and I).
Electron densities are computed here from the composition (electrons per volume = rho N_A sum_i w_i Z_i / A_i).
"""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

N_A = 6.02214076e23
# Z, A [g/mol] of the elements in the materials below
ELEMENTS = {"H": (1, 1.00794), "C": (6, 12.0107), "N": (7, 14.0067), "O": (8, 15.9994), "Na": (11, 22.98977),
            "Al": (13, 26.981538), "Si": (14, 28.0855), "Ar": (18, 39.948), "K": (19, 39.0983), "Ca": (20, 40.078),
            "Fe": (26, 55.845), "Mn": (25, 54.938049), "Cu": (29, 63.546), "W": (74, 183.84), "Mg": (12, 24.305),
            "B": (5, 10.811), "Ti": (22, 47.867), "F": (9, 18.9984032)}


@dataclass(frozen=True)
class Material:
    name: str
    rho: float                  # g/cm3
    I_eV: float                 # mean excitation energy
    fractions: dict             # element -> mass fraction
    sternheimer: tuple = ()     # (C, X0, X1, a, m, delta0) of G4DensityEffectData, empty = no table
    birks_mm_per_MeV: float = 0.0
    cuts_e_MeV: dict = field(default_factory=dict)   # production threshold for delta electrons, by region tag

    @cached_property
    def zeff(self):
        return sum(w * ELEMENTS[e][0] for e, w in self.fractions.items())

    @cached_property
    def electron_density_per_mm3(self):
        z_over_a = sum(w * ELEMENTS[e][0] / ELEMENTS[e][1] for e, w in self.fractions.items())
        return self.rho * N_A * z_over_a * 1e-3  # cm-3 -> mm-3

    def density_params(self):
        return self._density_params

    @cached_property
    def _density_params(self):
        """(C, X0, X1, a, m, delta0): the NIST table when given, else Sternheimer 1971 for condensed materials
        (G4IonisParamMat::ComputeDensityEffectParameters, solid branch; every non-NIST material here is a solid)."""
        if self.sternheimer:
            return self.sternheimer
        hbarc_mm = 197.3269804e-12            # MeV mm
        re_mm = 2.8179403262e-12
        e_plasma = np.sqrt(4 * np.pi * hbarc_mm ** 2 * re_mm * self.electron_density_per_mm3)   # MeV
        C = 1 + 2 * np.log(self.I_eV * 1e-6 / e_plasma)
        icase = 0 if self.I_eV < 100.0 else 1
        x0 = 0.2 if C < (3.681, 5.215)[icase] else 0.326 * C - (1.0, 1.5)[icase]
        x1 = (2.0, 3.0)[icase]
        m = 3.0
        a = 2 * np.log(10) * (C / (2 * np.log(10)) - x0) / (x1 - x0) ** m
        return (C, x0, x1, a, m, 0.0)

    def fluct_params(self):
        return self._fluct_params

    @cached_property
    def _fluct_params(self):
        """G4IonisParamMat::ComputeFluctModel: (f1, f2, e1, e2) in MeV, plus e0 = 10 eV and log I."""
        I = self.I_eV * 1e-6
        z = self.zeff
        if z > 2.1:
            f2 = 2.0 / z
            f1 = 1.0 - f2
            e2 = 10.0 * z * z * 1e-6
            loge1 = (np.log(I) - f2 * np.log(e2)) / f1
        elif z > 1.1:
            f2, f1, e2, loge1 = 0.0, 1.0, 40e-6, np.log(I)
        else:
            f2, f1, e2, loge1 = 0.0, 1.0, 10e-6, np.log(I)
        return {"f1": f1, "f2": f2, "e1": float(np.exp(loge1)), "e2": e2, "e0": 10e-6, "I": I}


POLYSTYRENE = Material("G4_POLYSTYRENE", 1.06, 68.7, {"C": 0.922577329294378, "H": 0.0774226707056217},
                       sternheimer=(3.2999, 0.1647, 2.5031, 0.16454, 3.2224, 0.0), birks_mm_per_MeV=0.07943,
                       cuts_e_MeV={"crv": 0.0872, "tagger": 2.2296})
ALUMINIUM = Material("G4_Al", 2.699, 166.0, {"Al": 1.0}, sternheimer=(4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12))
AIR = Material("G4_AIR", 0.00120479, 85.7,
               {"C": 0.000124000124000124, "N": 0.755267755267755, "O": 0.231781231781232, "Ar": 0.0128270128270128},
               sternheimer=(10.5961, 1.7418, 4.2759, 0.10914, 3.3994, 0.0))
CONCRETE_MARS = Material("CONCRETE_MARS", 2.35, 125.203942404978,
                         {"H": 0.006, "C": 0.03, "O": 0.5, "Na": 0.01, "Al": 0.03, "Si": 0.2, "K": 0.01, "Ca": 0.2})
MB_OVERBURDEN = Material("MBOverburden", 2.15, 125.453468818982,
                         {"O": 0.518324115382827, "Si": 0.279959130320566, "Al": 0.201716754296607})

MILD_STEEL = Material("MildSteel", 7.86, 284.511116716604, {"Mn": 0.01, "Si": 0.0025, "C": 0.0025, "Fe": 0.985})
SHIELDING_CONCRETE = Material("ShieldingConcrete", 2.5, 129.122501214703,
                              {"O": 0.52, "Si": 0.325, "Ca": 0.06, "Na": 0.015, "Fe": 0.04, "Al": 0.04})

# tracker-area electronics and services above the KPP tracker (v04 GDML, read 2026-10-16); Sternheimer tables for the
# NIST elements from G4DensityEffectData, mixtures use the condensed-material formula (density_params)
ALUMINIUM_6061 = Material("Al6061", 2.70, 166.0, {"Al": 0.98, "Mg": 0.01, "Si": 0.006, "Cu": 0.004})
ALUMINIUM_7075 = Material("Al7075", 2.81, 166.0, {"Al": 0.90, "Cu": 0.016, "Mg": 0.025, "Ti": 0.059})
COPPER = Material("G4_Cu", 8.96, 322.0, {"Cu": 1.0}, sternheimer=(4.4190, -0.0254, 3.2792, 0.14339, 2.9044, 0.08))
SILICON = Material("G4_Si", 2.33, 173.0, {"Si": 1.0}, sternheimer=(4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14))
ELECTRONICS2 = Material("Electronics2", 4.52, 229.423798310015,
                        {"Si": 0.0393766268325182, "O": 0.0937139019072929, "Ca": 0.021183442718492,
                         "Al": 0.0107331999698117, "Mg": 0.00235184116252291, "B": 0.0036336834583261,
                         "Ti": 0.000747978346220884, "Na": 0.00115729823161608, "K": 0.00129503062544011,
                         "Fe": 0.000545552317870526, "F": 0.001092, "C": 0.0715765662980504, "H": 0.0089116569351508,
                         "N": 0.00368122119668749, "Cu": 0.74})
CUW1090 = Material("CuW1090", 17.3, 663.689239365592, {"W": 0.9, "Cu": 0.1})

# tracker interior (Mu2eG4 ConstructMaterials; compositions approximate where the GDML holds a mixture: the
# energy loss depends on Z/A and I, both within 1-2 % of these for the alloys)
BRONZE_C938 = Material("BronzeC938", 9.4, 310.0, {"Cu": 0.75, "Mn": 0.10, "Fe": 0.15})
TRACKER_MANIFOLD = Material("TrackerManifold", 1.95, 120.0, {"C": 0.13, "H": 0.017, "Al": 0.645, "O": 0.208})
G10 = Material("G10", 1.70, 100.0, {"Si": 0.28, "O": 0.47, "C": 0.20, "H": 0.05})
ELECTRONICS = Material("Electronics", 0.58, 150.0, {"Si": 0.30, "O": 0.40, "C": 0.15, "Cu": 0.15})
MYLAR = Material("G4_MYLAR", 1.40, 78.7, {"C": 0.625010832340889, "H": 0.0419607170679432, "O": 0.333028450591168})
POLYETHYLENE = Material("G4_POLYETHYLENE", 0.94, 57.4, {"C": 0.856, "H": 0.144})
SILICA = Material("G4_SILICON_DIOXIDE", 2.32, 139.2, {"Si": 0.4674, "O": 0.5326})

BY_NAME = {m.name: m for m in (POLYSTYRENE, ALUMINIUM, AIR, CONCRETE_MARS, MB_OVERBURDEN, MILD_STEEL,
                               SHIELDING_CONCRETE, COPPER, SILICON, ELECTRONICS2, CUW1090, ALUMINIUM_6061,
                               ALUMINIUM_7075, BRONZE_C938, TRACKER_MANIFOLD, G10, ELECTRONICS, POLYETHYLENE, SILICA,
                               MYLAR)}
