"""JAX twin of eloss.dedx_restricted and eloss.sample_loss (Geant4 11.3.2 Urban fluctuations), jit-able, float64.

Same physics branches as eloss.py; the numpy module stays the default. How the eloss.py constructs map to JAX
(see docs/gpu_port.md; branch structure as in Celeritas EnergyLossUrbanDistribution/Gaussian/Gamma):
  * numpy Generator -> explicit jax.random keys, one subkey per random quantity; same key -> bit-identical output;
  * boolean-mask assignment -> every branch evaluated on the full array, selected with jnp.where (fixed shapes);
  * truncated-Gauss rejection loop -> exact inverse-CDF sampling of the Gauss truncated to [0, 2 eav] (ndtri);
  * gamma (Bohr regime with mean/sigma < 2) -> Marsaglia-Tsang with a fixed number of rounds (+ alpha<1 boost);
  * Poisson counts + np.repeat/bincount -> inverse-CDF Poisson in a fixed K-round fori_loop that also sums the
    ionisation terms of the first k rounds. All Poisson means are <= 8 (excitations: Gaussian above _NMAXCONT; the
    ionisation count after the continuous part is 8 a3 / (8 + a3) < 8, computed in closed form), so K = 32
    truncates with P < 1e-10.
Materials: a Material (closed over, constants folded) or MatConsts (pytree of scalars or per-step arrays).
CLI: python3 -m mu2efastsim.eloss_jax --bench [--n N ...]  (default 1e5 and 1e6 steps)
"""
from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import ndtr, ndtri

from .eloss import ME, MMU, P_FLOOR, TWOLN10, TWOPI_MC2_RCL2, _A0, _BOHR_MIN, _FW, _MINLOSS, _NMAXCONT, _RATE

jax.config.update("jax_enable_x64", True)

K_POISSON = 32      # fixed Poisson rounds (all means <= 8: truncation probability < 1e-10)
GAMMA_ROUNDS = 10   # Marsaglia-Tsang rounds (acceptance >= 0.95 per round: unaccepted < 1e-12)


class MatConsts(NamedTuple):
    """Material constants used by the sampler (MeV, mm); fields may be scalars or arrays broadcastable to the steps."""
    I: object
    ne: object
    C: object
    X0: object
    X1: object
    a: object
    m: object
    d0: object
    f1: object
    f2: object
    e1: object
    e2: object
    e0: object


def consts(mat):
    """Material -> MatConsts (float64 numpy scalars); MatConsts pass through."""
    if isinstance(mat, MatConsts):
        return mat
    C, X0, X1, a, m, d0 = mat.density_params()
    f = mat.fluct_params()
    return MatConsts(*(np.float64(v) for v in (mat.I_eV * 1e-6, mat.electron_density_per_mm3, C, X0, X1, a, m, d0,
                                                 f["f1"], f["f2"], f["e1"], f["e2"], f["e0"])))


def stack_consts(mats, index):
    """Per-step MatConsts from a list of materials and an integer material index per step (mixed-material batches)."""
    cs = [consts(m) for m in mats]
    return MatConsts(*(jnp.asarray(np.array([getattr(c, f) for c in cs]))[jnp.asarray(index)] for f in MatConsts._fields))


def kinematics(p, mass=MMU):
    """|p| [MeV/c] -> (kinetic energy, beta^2, beta*gamma squared, gamma, Tmax); p is clamped at P_FLOOR."""
    p = jnp.maximum(jnp.asarray(p, jnp.float64), P_FLOOR)
    e = jnp.sqrt(p * p + mass * mass)
    gam = e / mass
    bg2 = (p / mass) ** 2
    beta2 = bg2 / (gam * gam)
    ratio = ME / mass
    tmax = 2 * ME * bg2 / (1 + 2 * gam * ratio + ratio * ratio)
    return e - mass, beta2, bg2, gam, tmax


def _density_correction(c, bg2):
    x = jnp.log(bg2) / TWOLN10
    y = jnp.where(x >= c.X1, TWOLN10 * x - c.C, TWOLN10 * x - c.C + c.a * jnp.clip(c.X1 - x, 0, None) ** c.m)
    low = jnp.where(c.d0 > 0, c.d0 * jnp.exp(TWOLN10 * (x - c.X0)), 0.0)
    return jnp.where(x < c.X0, low, y)


def _dedx(c, p, tcut, mass):
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    cut = jnp.minimum(tcut, tmax)
    d = jnp.log(2 * ME * bg2 * cut / (c.I * c.I)) - (1 + cut / tmax) * beta2
    d = d + (0.5 * cut / (tkin + mass)) ** 2
    d = d - _density_correction(c, bg2)
    return jnp.maximum(d * TWOPI_MC2_RCL2 * c.ne / beta2, 0.0)


_dedx_jit = jax.jit(_dedx)


def dedx_restricted(mat, p, tcut, mass=MMU):
    """Restricted mean energy loss [MeV/mm] for delta electrons below tcut [MeV] (jit-able; jnp array out)."""
    return _dedx_jit(consts(mat), jnp.asarray(p, jnp.float64), jnp.asarray(tcut, jnp.float64),
                     jnp.asarray(mass, jnp.float64))


def _gauss_clipped(u, eav, sig2):
    """G4UniversalFluctuation::SampleGauss from one uniform: uniform in [0, 2 eav] if eav < sig/4, else exact
    inverse-CDF Gauss truncated to [0, 2 eav] (replaces the rejection loop)."""
    sig = jnp.sqrt(jnp.maximum(sig2, 0.0))
    flat = eav < 0.25 * sig
    b = jnp.where(sig2 > 0, eav / jnp.where(sig2 > 0, sig, 1.0), 1.0)
    pa = ndtr(-b)
    z = jnp.clip(ndtri(pa + u * (1 - 2 * pa)), -b, b)
    return jnp.where(flat, eav + (2 * u - 1) * eav, jnp.where(sig2 > 0, eav + sig * z, 0.0))


def _gamma_fixed(key, alpha, rounds):
    """Gamma(alpha, 1) by Marsaglia-Tsang with a fixed number of rounds; alpha < 1 via the U^(1/alpha) boost."""
    kb, kr = jax.random.split(key)
    a1 = jnp.where(alpha < 1, alpha + 1, alpha)
    d = a1 - 1.0 / 3.0
    cc = 1.0 / jnp.sqrt(9 * d)

    def body(i, carry):
        val, done = carry
        kn, ku = jax.random.split(jax.random.fold_in(kr, i))
        x = jax.random.normal(kn, alpha.shape, jnp.float64)
        v = (1 + cc * x) ** 3
        lu = jnp.log(jax.random.uniform(ku, alpha.shape, jnp.float64, minval=1e-300))
        ok = (v > 0) & (lu < 0.5 * x * x + d - d * v + d * jnp.log(jnp.where(v > 0, v, 1.0)))
        take = ok & ~done
        return jnp.where(take, d * v, val), done | ok

    val, _ = lax.fori_loop(0, rounds, body, (d, jnp.zeros(alpha.shape, bool)))
    ub = jax.random.uniform(kb, alpha.shape, jnp.float64, minval=1e-300)
    return jnp.where(alpha < 1, val * ub ** (1 / jnp.where(alpha > 0, alpha, 1.0)), val)


def _ion_split(a3, w1):
    """Ionisation continuous/Poisson split (G4UniversalFluctuation, a3 > _NMAXCONT): (alfa, alfa1, namean, p3).

    G4 computes namean = a3 w1 (alfa-1) / ((w1-1) alfa) and p3 = a3 - namean, which is algebraically
    namean = a3^2 / (_NMAXCONT + a3), p3 = _NMAXCONT a3 / (_NMAXCONT + a3) < _NMAXCONT; the G4 form cancels
    catastrophically for w1 -> 1 (p3 up to ~30 at w1 - 1 = 1e-12), the closed form is used so p3 < 8 holds exactly.
    """
    cont = a3 > _NMAXCONT
    alfa = jnp.where(cont, w1 * (_NMAXCONT + a3) / (w1 * _NMAXCONT + a3), 1.0)
    am1 = jnp.where(cont, a3 * (w1 - 1) / (w1 * _NMAXCONT + a3), 1.0)  # alfa - 1 without cancellation
    am1 = jnp.where(am1 > 0, am1, 1.0)
    alfa1 = jnp.where(cont, jnp.where(am1 < 1e-8, 1 + 0.5 * am1, alfa * jnp.log1p(am1) / am1), 0.0)
    p3 = jnp.where(cont, _NMAXCONT * a3 / (_NMAXCONT + a3), a3)
    namean = jnp.where(cont, a3 - p3, 0.0)
    return alfa, alfa1, namean, p3


@partial(jax.jit, static_argnames=("K", "gamma_rounds"))
def _sample(key, c, p, length, tcut, mass, K=K_POISSON, gamma_rounds=GAMMA_ROUNDS, mean=None):
    p = jnp.asarray(p, jnp.float64)
    p, length, tcut = jnp.broadcast_arrays(p, jnp.asarray(length, jnp.float64), jnp.asarray(tcut, jnp.float64))
    c = MatConsts(*(jnp.broadcast_to(jnp.asarray(v, jnp.float64), p.shape) for v in c))
    sh = p.shape
    ks = jax.random.split(key, 9)
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    cut = jnp.minimum(tcut, tmax)
    # mean: caller-supplied mean loss (e.g. range-integrated, as G4VEnergyLossProcess for long steps)
    mean = _dedx(c, p, tcut, mass) * length if mean is None else jnp.broadcast_to(jnp.asarray(mean, jnp.float64), sh)
    # Gaussian/Gamma regime: many collisions, cut close to tmax
    bohr = (mean >= _BOHR_MIN * cut) & (tmax <= 2 * cut)
    siga = jnp.sqrt(jnp.clip((tmax / beta2 - 0.5 * cut) * TWOPI_MC2_RCL2 * length * c.ne, 0, None))
    sn = mean / jnp.where(siga > 0, siga, 1)
    gs = bohr & (sn >= 2)
    gm = bohr & (sn < 2)
    x_gs = _gauss_clipped(jax.random.uniform(ks[0], sh, jnp.float64), mean, siga ** 2)
    neff = jnp.where(gm, sn * sn, 1.0)
    x_gm = mean * _gamma_fixed(ks[1], neff, gamma_rounds) / neff
    # Urban model for the rest (inputs made safe outside u so every lane stays finite)
    u = ~bohr & (mean >= _MINLOSS) & (cut > c.e0)
    tc = jnp.where(u, cut, 1.0)
    scaling = jnp.minimum(1 + 0.5e-3 / tc, 1.5)
    ml = jnp.where(u, mean, 0.0) / scaling
    w2 = jnp.where(tc > c.I, jnp.log(2 * ME * beta2 * gam ** 2) - beta2, 0.0)
    lnI, lne1, lne2 = jnp.log(c.I), jnp.log(c.e1), jnp.log(c.e2)
    ok = (tc > c.I) & (w2 > lnI)
    full = ok & (w2 > lne2)
    C = ml * (1 - _RATE) / jnp.where(w2 > lnI, w2 - lnI, 1)
    a1 = jnp.where(full, C * c.f1 * (w2 - lne1) / c.e1, jnp.where(ok, ml * (1 - _RATE) / c.e1, 0.0))
    a2 = jnp.where(full, C * c.f2 * (w2 - lne2) / c.e2, 0.0)
    small = ok & (a1 < _A0)
    fwnow = jnp.where(small, 0.5 + (_FW - 0.5) * jnp.sqrt(jnp.clip(a1, 0, None) / _A0), _FW)
    e1 = jnp.where(ok, c.e1 * fwnow, c.e1)
    a1 = jnp.where(ok, a1 / fwnow, a1)
    e2 = c.e2
    w1 = tc / c.e0
    lw1 = jnp.log(jnp.where(w1 > 1, w1, 2.0))
    a3 = _RATE * ml * (tc - c.e0) / (c.e0 * tc * lw1)
    a3 = jnp.where(a1 + a2 <= 0, a3 / _RATE, a3)
    # excitation: Gaussian (continuous) part above _NMAXCONT, Poisson below
    cont1, cont2 = a1 > _NMAXCONT, a2 > _NMAXCONT
    eav = jnp.where(cont1, a1 * e1, 0.0) + jnp.where(cont2, a2 * e2, 0.0)
    s2 = jnp.where(cont1, a1 * e1 * e1, 0.0) + jnp.where(cont2, a2 * e2 * e2, 0.0)
    lam1 = jnp.where(cont1 | (a1 <= 0), 0.0, a1)
    lam2 = jnp.where(cont2 | (a2 <= 0), 0.0, a2)
    # ionisation: continuous part above _NMAXCONT, the rest as Poisson count of 1/E^2 terms
    cont = a3 > _NMAXCONT
    alfa, alfa1, namean, p3 = _ion_split(a3, w1)
    eav3 = namean * c.e0 * alfa1
    s23 = c.e0 * c.e0 * namean * (alfa - alfa1 * alfa1)
    w3 = alfa * c.e0
    has = tc > w3
    w = jnp.where(has, (tc - w3) / tc, 0.0)
    lam3 = jnp.where(has, jnp.clip(p3, 0, None), 0.0)
    # one fixed-trip loop: inverse-CDF Poisson counts for lam1, lam2, lam3 and the sum of the first n3 ionisation terms
    uu = jax.random.uniform(ks[2], (3,) + sh, jnp.float64)
    kr = ks[3]
    lams = jnp.stack([lam1, lam2, lam3])

    def body(j, carry):
        pmf, cdf, k, ion = carry
        above = uu > cdf
        r = jax.random.uniform(jax.random.fold_in(kr, j), sh, jnp.float64)
        ion = ion + jnp.where(above[2], w3 / (1 - w * r), 0.0)
        pmf = pmf * lams / (j + 1)
        return pmf, cdf + pmf, k + above, ion

    pmf0 = jnp.exp(-lams)
    _, _, kk, ion = lax.fori_loop(0, K, body, (pmf0, pmf0, jnp.zeros((3,) + sh, jnp.int32), jnp.zeros(sh)))
    rx = jax.random.uniform(ks[4], (2,) + sh, jnp.float64)
    loss = jnp.where(kk[0] > 0, ((kk[0] + 1) - 2 * rx[0]) * e1, 0.0)
    loss = loss + jnp.where(kk[1] > 0, ((kk[1] + 1) - 2 * rx[1]) * e2, 0.0)
    loss = loss + jnp.where(s2 > 0, _gauss_clipped(jax.random.uniform(ks[5], sh, jnp.float64), eav, s2), 0.0)
    loss = loss + ion
    loss = loss + jnp.where(s23 > 0, _gauss_clipped(jax.random.uniform(ks[6], sh, jnp.float64), eav3,
                                                    jnp.where(s23 > 0, s23, 0.0)), 0.0)
    out = jnp.where(gs, x_gs, jnp.where(gm, x_gm, mean))
    return jnp.where(u, loss * scaling, out)


def sample_loss(key, mat, p, length, tcut, mass=MMU, K=K_POISSON, mean=None):
    """Sampled restricted energy loss [MeV] over `length` [mm] (arrays broadcast to p's shape), G4 Urban model.

    key: jax.random key; mat: Material or MatConsts. jit-able (close over a Material, or pass MatConsts as arrays).
    mean: optional mean loss [MeV] replacing dE/dx(p) * length (the fluctuation width still uses p, length, tcut).
    """
    return _sample(key, consts(mat), p, length, tcut, jnp.asarray(mass, jnp.float64), K=K, mean=mean)


def _bench(n, reps=3):
    """Samples/s of numpy eloss.sample_loss vs this module on the default JAX device, for three typical cases."""
    import time
    from . import eloss, materials as M
    gas = M.Material("StrawGas", 0.001696, 158.62097145526,
                     {"Ar": 0.784057035890315, "O": 0.15700963928136, "C": 0.0589333248283256})
    rng = np.random.default_rng(1)
    cases = [("straw gas 990 eV, 1-10 mm, 3 GeV/c", gas, np.full(n, 3000.0), rng.uniform(1, 10, n), 990e-6),
             ("polystyrene 87.2 keV, 20 mm, 0.3-10 GeV/c", M.POLYSTYRENE, np.geomspace(300, 1e4, n), 20.0, 0.0872),
             ("concrete 1e9 cut, 5-50 mm, 0.2-3 GeV/c", M.CONCRETE_MARS, np.geomspace(200, 3e3, n),
              rng.uniform(5, 50, n), 1e9)]
    print(f"device: {jax.devices()[0]}, n = {n}")
    for name, mat, p, L, tc in cases:
        t = time.perf_counter()
        for _ in range(reps):
            eloss.sample_loss(rng, mat, p, L, tc)
        tn = (time.perf_counter() - t) / reps
        pj, Lj = jnp.asarray(p), jnp.asarray(np.broadcast_to(L, p.shape))
        t = time.perf_counter()
        sample_loss(jax.random.key(0), mat, pj, Lj, tc).block_until_ready()
        tcmp = time.perf_counter() - t
        t = time.perf_counter()
        for i in range(reps):
            sample_loss(jax.random.key(i + 1), mat, pj, Lj, tc).block_until_ready()
        tj = (time.perf_counter() - t) / reps
        print(f"{name:45s} numpy {n / tn:10.3e}/s  jax {n / tj:10.3e}/s  (first call incl. compile {tcmp:.2f} s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--n", type=int, nargs="+", default=[100_000, 1_000_000])
    a = ap.parse_args()
    if a.bench:
        for n in a.n:
            _bench(n)
    else:
        ap.print_help()
