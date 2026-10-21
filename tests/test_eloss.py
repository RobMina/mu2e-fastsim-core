"""Unit tests: muon energy loss in polystyrene against Geant4-derived numbers (no data files).

Reference values: PDG restricted dE/dx behaviour, and the g4ref_v1 primary-muon CrvSteps in EX/T1/T2 (2026-10-16,
8 shards, 97,177 steps with |p| > 300 MeV/c): visible energy mean 3.583 MeV over their own path lengths. The
per-crossing distribution check against G4 lives in the V1 comparison; these tests pin the formulas.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import eloss as E                                       # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

TCUT = 0.0872  # MeV, the 0.1 mm range cut in CRSmother_CRV_* regions (measured from the delta spectrum)


def test_minimum_ionising_unrestricted():
    # PDG (2023 tables): polystyrene minimum ionisation 1.936 MeV cm2/g = 2.052 MeV/cm, near p ~ 0.4 GeV/c
    p = np.geomspace(200.0, 2000.0, 400)
    d = E.dedx_restricted(M.POLYSTYRENE, p, 1e9)
    assert abs(d.min() - 0.2052) / 0.2052 < 0.005
    assert 250.0 < p[d.argmin()] < 500.0


def test_restricted_below_unrestricted_and_plateaus():
    p = np.array([1e3, 1e4, 1e5])
    r = E.dedx_restricted(M.POLYSTYRENE, p, TCUT)
    u = E.dedx_restricted(M.POLYSTYRENE, p, 1e9)
    assert (r < u).all()
    assert abs(r[2] / r[1] - 1) < 0.005  # Fermi plateau of the restricted loss


def test_sampled_mean_equals_restricted_mean():
    rng = np.random.default_rng(1)
    p = np.full(200_000, 3000.0)
    loss = E.sample_loss(rng, M.POLYSTYRENE, p, 20.0, TCUT)
    expect = E.dedx_restricted(M.POLYSTYRENE, 3000.0, TCUT) * 20.0
    assert abs(loss.mean() / expect - 1) < 0.002


def test_birks_saturation_small_for_mips():
    vis = E.birks(M.POLYSTYRENE, np.array([3.3]), np.array([20.0]))
    assert np.isclose(vis[0], 3.3 / (1 + 0.07943 * 3.3 / 20.0))


def test_delta_rate():
    # ~2 deltas above 87 keV per 20 mm of polystyrene for relativistic muons
    n = E.delta_rate_per_mm(M.POLYSTYRENE, np.array([3000.0]), TCUT)[0] * 20.0
    assert 1.9 < n < 2.1
    rng = np.random.default_rng(2)
    idx, T = E.sample_deltas(rng, M.POLYSTYRENE, np.full(50_000, 3000.0), 20.0, TCUT)
    assert T.min() >= TCUT and abs(len(T) / 50_000 / n - 1) < 0.02


def test_material_mass_fractions_sum_to_one():
    """Every material's composition is complete (a dropped element lowers Z/A and the electron density, so dE/dx)."""
    from mu2efastsim import materials as MM
    bad = {n: round(sum(m.fractions.values()), 6) for n, m in MM.BY_NAME.items() if abs(sum(m.fractions.values()) - 1) > 1e-6}
    assert not bad, bad


def test_tracker_materials_match_the_gdml_export():
    """Spot values of the v04 GDML for the tracker-interior alloys (I and the heavy constituents drive dE/dx and X0)."""
    from mu2efastsim import materials as MM
    assert abs(MM.BRONZE_C938.I_eV - 374.5128) < 1e-3 and abs(MM.BRONZE_C938.fractions["Pb"] - 0.145) < 1e-9
    assert abs(MM.ELECTRONICS.I_eV - 226.4011) < 1e-3 and abs(MM.ELECTRONICS.fractions["Au"] - 0.034) < 1e-9
    assert abs(MM.ALUMINIUM_7075.fractions["Zn"] - 0.056) < 1e-9
    assert abs(MM.CONCRETE_MARS.fractions["Fe"] - 0.014) < 1e-9
