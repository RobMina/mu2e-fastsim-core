"""JAX twin of the electron slab (mu2efastsim.eslab_jax) and the electron Urban sampler against the NumPy models.

Exact: the host tables equal the NumPy functions at the grid nodes; energy balance per track; muon sample_loss
outputs unchanged by the electron options (SHA-256 of a fixed grid, float64 and float32). Statistical (small samples
here; the 200k-event parity per case is a slow test and a script): loss, exit angle, fates and yields against eslab.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
jax = pytest.importorskip("jax")

from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import eioni as I                                       # noqa: E402
from mu2efastsim import eloss_jax as EJ                                  # noqa: E402
from mu2efastsim import emsc as E                                        # noqa: E402
from mu2efastsim import eslab as S                                       # noqa: E402
from mu2efastsim import eslab_jax as SJ                                  # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402

AL, W = M.BY_NAME["G4_Al"], M.BY_NAME["G4_W"]
CUTS = {"G4_Al": (0.03411, 0.00099), "G4_W": (0.0814923, 0.00770222)}

# eloss_jax.sample_loss for muons over the grid of _muon_hashes, computed with the sampler before the electron
# options were added (JAX 0.11.2 CPU); float64 results depend on the CPU's math kernels, so one hash per platform seen
MUON_SHA256 = {"float64": {"5a43f3ec712cf4a397270126c9aad3c900e60b16de04546cdcf2d898d9ab6ff2",    # arm64
                           "31ac62093ec331e1c5058b8b1612b4812ea02ec683dca0654f0d11d0842c6e99"},   # x86-64
               "float32": {"e41dc54f335924dbc4d600c6d88c3637f302c5676f9b91f3ddd39835150ecb33"}}


def _muon_hashes():
    out = {}
    for dt in (np.float64, np.float32):
        h = hashlib.sha256()
        for name in ("G4_Al", "G4_W", "G4_POLYSTYRENE", "G4_MYLAR", "CONCRETE_MARS"):
            mat = M.BY_NAME[name]
            for i, p in enumerate((50.0, 105.0, 300.0, 1000.0, 5000.0)):
                for L in (0.01, 1.0, 20.0):
                    for tcut in (0.03, 1.0):
                        x = np.asarray(EJ.sample_loss(jax.random.PRNGKey(i), mat, np.full(64, p, dt),
                                                      np.asarray(L, dt), np.asarray(tcut, dt)))
                        h.update(x.tobytes())
        out[np.dtype(dt).name] = h.hexdigest()
    return out


def test_muon_sampler_is_bit_identical_to_before_the_electron_options():
    if jax.__version__ != "0.11.2" or jax.default_backend() != "cpu":
        pytest.skip("reference hashes recorded with JAX 0.11.2 on CPU")
    h = _muon_hashes()
    assert h["float32"] in MUON_SHA256["float32"]
    if h["float64"] not in MUON_SHA256["float64"]:
        pytest.skip(f"no float64 reference for this CPU: {h['float64']}")


def test_electron_options_are_inert_for_muons():
    # explicit muon tmax and gaussian=True reproduce the default call bit for bit
    for dt in (np.float64, np.float32):
        p = np.geomspace(30.0, 5000.0, 64).astype(dt)
        for mat in (AL, W):
            c = EJ.consts(mat)
            mass = np.asarray(EJ.MMU, dt)
            tmax = EJ.kinematics(p, mass)[4]
            a = EJ._sample(jax.random.PRNGKey(7), c, p, np.asarray(1.0, dt), np.asarray(0.05, dt), mass)
            b = EJ._sample(jax.random.PRNGKey(7), c, p, np.asarray(1.0, dt), np.asarray(0.05, dt), mass, tmax=tmax,
                           gaussian=True)
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_electron_urban_sampler_matches_numpy():
    n = 200_000
    for mat, T, L in ((AL, 105.0, 0.1), (W, 2.0, 0.01), (AL, 0.5, 0.05)):
        tc, kc = CUTS[mat.name]
        mean = ((I.dedx_restricted(mat, T, tc) + B.dedx_restricted(mat, T, kc)) * L)[0]
        a = I.continuous_loss(np.random.default_rng(1), mat, np.full(n, T), L, tc, kc)
        b = np.asarray(EJ.sample_loss_electron(jax.random.PRNGKey(2), mat, np.full(n, T), L, tc, np.full(n, mean)))
        assert b.mean() == pytest.approx(a.mean(), rel=0.01)
        assert np.quantile(b, [0.05, 0.5, 0.95]) == pytest.approx(np.quantile(a, [0.05, 0.5, 0.95]), rel=0.02)


@pytest.mark.parametrize("positron", [False, True])
def test_tables_equal_the_numpy_models_at_the_nodes(positron):
    tc, kc = CUTS["G4_W"]
    tab = SJ.build_tables(W, tc, kc, positron, 0.05)
    T = SJ._grid()[1]
    sb, sd = S._sigmas(W, T, tc, kc, positron)
    assert tab.sig_b == pytest.approx(sb) and tab.sig_d == pytest.approx(sd)
    assert tab.lam1 == pytest.approx(E.lambda1(W, T, tc, positron))
    assert tab.uc == pytest.approx(E.u_cut(W, T, tc, positron, 0.05))
    assert tab.rng == pytest.approx(S.csda_range(W, T, tc, kc, positron))
    assert np.all(np.diff(tab.brem_x, axis=1) >= 0) and tab.brem_x.min() >= 0 and tab.brem_x.max() <= 1


@pytest.mark.parametrize("msc", [False, True])
@pytest.mark.parametrize("positron", [False, True])
def test_energy_balance_faces_and_dtype(positron, msc):
    n, T0, L = 400, 105.0, 2.0
    tc, kc = CUTS["G4_W"]
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    r = jax.tree_util.tree_map(np.asarray, SJ.electron_slab(jax.random.PRNGKey(3), W, np.full(n, T0), d, L, tc, kc,
                                                            positron, msc=msc))
    out = r.continuous + r.photon_energy + r.delta_energy + r.annihilation_energy - SJ.ME * r.n_annihilation
    assert T0 - r.T == pytest.approx(out, abs=1e-9)
    left = ~r.stopped
    z = r.position[:, 2]
    assert np.allclose(z[left & r.reflected], 0.0, atol=1e-9) and np.allclose(z[left & ~r.reflected], L, atol=1e-9)
    assert (r.T[r.stopped] == 0).all() and r.T.dtype == np.float64
    if positron:
        assert (r.n_annihilation[r.stopped] == 2).all()
    r32 = SJ.electron_slab(jax.random.PRNGKey(3), W, np.full(8, T0, np.float32), d[:8].astype(np.float32), L, tc, kc,
                           positron, msc=msc)
    assert r32.T.dtype == np.float32 and r32.position.dtype == np.float32


def test_small_sample_parity_with_numpy():
    # 1 mm Al, 105 MeV e- with msc: medians of loss and exit angle, photon and delta yields within statistics
    n = 20_000
    tc, kc = CUTS["G4_Al"]
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    a = S.electron_slab(np.random.default_rng(4), AL, np.full(n, 105.0), d, 1.0, tc, kc, msc=True)
    b = jax.tree_util.tree_map(np.asarray, SJ.electron_slab(jax.random.PRNGKey(4), AL, np.full(n, 105.0), d, 1.0, tc,
                                                            kc, msc=True))
    th = lambda dd: np.arctan2(np.hypot(dd[:, 0], dd[:, 1]), dd[:, 2])
    assert np.median(105.0 - b.T) == pytest.approx(np.median(105.0 - a.T), rel=0.02)
    assert np.median(th(b.direction)) == pytest.approx(np.median(th(a.direction)), rel=0.03)
    assert b.n_deltas.mean() == pytest.approx(a.deltas.track.size / n, abs=5 * np.sqrt(0.6 / n))
    assert b.n_photons.mean() == pytest.approx(a.photons.track.size / n, abs=5 * np.sqrt(0.15 / n))


@pytest.mark.slow
@pytest.mark.parametrize("case", [("G4_Al", 105.0, 0.1), ("G4_Al", 20.0, 1.0), ("G4_W", 105.0, 3.5)])
def test_parity_200k(case):
    name, T0, L = case
    tc, kc = CUTS[name]
    r = SJ.compare_numpy(M.BY_NAME[name], T0, L, tc, kc, n=200_000)
    for k in ("loss", "theta", "r"):
        assert r[k]["chi2_ndf"] < 2.0, (k, r[k])
