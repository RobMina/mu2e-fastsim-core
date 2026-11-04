"""Device-resident sliced slab loss: transport.slab_loss with one per-lane slice loop inside a single kernel.

transport.slab_loss slices a (material, |p|, length) pair so dE/dx follows the slowing muon: one slice for a fast
muon, 60-150 for one that ranges out. Written as an XLA loop over a batch, every iteration runs the slice on every
lane and the host reads the loop predicate back, so a batch costs (slices of its slowest lane) x (launches + one
device-to-host copy). Here a Pallas (Triton) kernel runs the loop per block of lanes on the device: one launch per
call, and a slow lane costs only its block.

  * raw-key threefry (`tf2x32`, `fold_in`, `uniform`, `normal`): bit for bit jax.random with the partitionable
    threefry (the JAX default), as uint32 arithmetic that also lowers inside a Pallas kernel;
  * `sample`: eloss_jax._sample on raw keys (Gaussian/Gamma/Urban regimes, mean and tmax from |p|), the same draws;
  * `slab_loss(keys, consts, p, length)`: transport.slab_loss per lane with the slice keys fold_in(key, slice), as the
    tracker's device slab sampler; `path_loss`: a sequence of slabs and mean-loss segments (a world column) in one call.
    With `rad=` (muradiative_jax tables) both add transport.slab_loss's default radiative losses after the kernel.
    impl "xla" (a vmapped while loop: the reference, and the path on CPU and non-NVIDIA GPUs), "pallas" (the Triton
    kernel, NVIDIA CUDA only) or "interpret" (the kernel run by the Pallas interpreter, for CPU tests of its code);
    `default_impl()` picks the kernel on a CUDA backend and the XLA path elsewhere (MU2EFS_SLAB_KERNEL=auto, default;
    xla or pallas force one), `provenance()` records the choice.

Arithmetic is that of eloss_jax._sample; only erf/erfc inside the kernel come from a Cephes rational approximation
(Pallas has no erf), so kernel and XLA results agree to rounding, not bit for bit. Precision: the dtype of p
(float64, or float32 when the caller runs in float32).
"""
from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .eloss import ME, MMU, TWOPI_MC2_RCL2, _A0, _BOHR_MIN, _FW, _MINLOSS, _NMAXCONT, _RATE
from .eloss_jax import GAMMA_ROUNDS, K_POISSON, MatConsts, _dedx, _ion_split, kinematics
from . import precision as dprec
from . import transport as tr

jax.config.update("jax_enable_x64", True)

BLOCK = 128         # lanes per kernel program (Pallas block); power of two
IMPLS = ("pallas", "xla", "interpret")
_ROT = ((13, 15, 26, 6), (17, 29, 16, 24))
_SQRT2 = float(np.sqrt(2.0))


def cuda_backend():
    """True when the default JAX device is an NVIDIA GPU under CUDA (the only platform the Triton kernel targets)."""
    try:
        d = jax.devices()[0]
        return d.platform == "gpu" and "cuda" in str(getattr(d.client, "platform_version", "")).lower()
    except Exception:
        return False


_PALLAS = {}


def pallas_available():
    """Whether a small float64 Triton kernel with a per-block loop compiles and runs on this device (checked once;
    the failure, e.g. a GPU Triton does not support, is kept for provenance)."""
    if "ok" not in _PALLAS:
        try:
            from jax.experimental import pallas as pl

            def probe(x_ref, o_ref):
                x = x_ref[...]
                o_ref[...] = lax.while_loop(lambda v: jnp.max((v > 1.0).astype(jnp.int32)) > 0,
                                            lambda v: jnp.where(v > 1.0, 0.5 * v + jnp.log1p(v) * 0.0, v), x)

            x = jnp.linspace(0.5, 64.0, 32, dtype=jnp.float64)
            out = pl.pallas_call(probe, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
            _PALLAS["ok"] = bool(jnp.all(out <= 1.0))
            _PALLAS["error"] = None if _PALLAS["ok"] else "probe kernel gave wrong values"
        except Exception as e:      # noqa: BLE001 (any lowering or compile failure means: use the XLA path)
            _PALLAS["ok"], _PALLAS["error"] = False, f"{type(e).__name__}: {str(e)[:200]}"
    return _PALLAS["ok"]


def default_impl():
    """Implementation for callers that do not force one: MU2EFS_SLAB_KERNEL=auto (default: pallas on a CUDA backend
    where the kernel compiles, xla elsewhere), xla or pallas."""
    v = os.environ.get("MU2EFS_SLAB_KERNEL", "auto").strip().lower() or "auto"
    if v == "auto":
        return "pallas" if cuda_backend() and pallas_available() else "xla"
    if v not in ("xla", "pallas"):
        raise ValueError(f"MU2EFS_SLAB_KERNEL must be auto, xla or pallas, not {v!r}")
    return v


def provenance():
    """Slab-loss implementation record: the setting, the resolved implementation, whether the backend is CUDA and,
    if the kernel was probed, why it is unavailable."""
    impl = default_impl()
    return {"setting": os.environ.get("MU2EFS_SLAB_KERNEL", "auto"), "impl": impl, "cuda": cuda_backend(),
            "pallas_error": _PALLAS.get("error"), "block": BLOCK}


# --------------------------------------------------------------------------- raw-key threefry (jax.random bits)

def tf2x32(k0, k1, x0, x1):
    """Threefry-2x32, 20 rounds (jax's threefry2x32 primitive) on uint32 arrays."""
    u = jnp.uint32
    ks = (k0, k1, k0 ^ k1 ^ u(0x1BD11BDA))
    x0, x1 = x0 + ks[0], x1 + ks[1]
    for r in range(5):
        for rot in _ROT[r % 2]:
            x0 = x0 + x1
            x1 = (x1 << u(rot)) | (x1 >> u(32 - rot))
            x1 = x0 ^ x1
        x0 = x0 + ks[(r + 1) % 3]
        x1 = x1 + ks[(r + 2) % 3] + u(r + 1)
    return x0, x1


def raw(key):
    """Typed jax.random key (any batch shape) -> raw key (k0, k1) of uint32 arrays."""
    kd = jax.random.key_data(key)
    return kd[..., 0], kd[..., 1]


def fold_in(k, d):
    """jax.random.fold_in(key, d) (also jax.random.split(key, n)[d]) on a raw key."""
    k0, k1 = k
    d = jnp.asarray(d).astype(jnp.uint32)
    return tf2x32(k0, k1, jnp.zeros_like(d), d)


def _bits(k, index, dtype):
    k0, k1 = k
    i = jnp.broadcast_to(jnp.asarray(index, jnp.uint32), k0.shape)
    b0, b1 = tf2x32(k0, k1, jnp.zeros_like(i), i)
    if np.dtype(dtype) == np.float64:
        return (b0.astype(jnp.uint64) << jnp.uint64(32)) | b1.astype(jnp.uint64)
    return b0 ^ b1


def uniform(k, dtype, index=0, minval=0.0, maxval=1.0):
    """Element `index` of jax.random.uniform(key, shape, dtype, minval, maxval) (flat index; 0 for shape ())."""
    dtype = np.dtype(dtype)
    bits = _bits(k, index, dtype)
    if dtype == np.float64:
        fb = (bits >> jnp.uint64(12)) | jnp.uint64(0x3FF0000000000000)
        f = lax.bitcast_convert_type(fb, jnp.float64) - jnp.float64(1.0)
    else:
        fb = (bits >> jnp.uint32(9)) | jnp.uint32(0x3F800000)
        f = lax.bitcast_convert_type(fb, jnp.float32) - jnp.float32(1.0)
    lo, hi = jnp.asarray(minval, dtype), jnp.asarray(maxval, dtype)
    return jnp.maximum(lo, f * (hi - lo) + lo)


def normal(k, dtype, index=0):
    """Element `index` of jax.random.normal(key, shape, dtype)."""
    dtype = np.dtype(dtype)
    u = uniform(k, dtype, index, np.nextafter(np.array(-1.0, dtype), np.array(0.0, dtype)), 1.0)
    return jnp.asarray(_SQRT2, dtype) * lax.erf_inv(u)


# --------------------------------------------------------------------------- erf / erfc / ndtr

# Cephes ndtr.c coefficients (erf for |x| < 1; erfc for 1 <= x < 8 and x >= 8)
_T = (9.60497373987051638749e0, 9.00260197203842689217e1, 2.23200534594684319226e3, 7.00332514112805075473e3,
      5.55923013010394962768e4)
_U = (3.35617141647503099647e1, 5.21357949780152679795e2, 4.59432382970980127987e3, 2.26290000613890934246e4,
      4.92673942608635921086e4)
_P = (2.46196981473530512524e-10, 5.64189564831068821977e-1, 7.46321056442269912687e0, 4.86371970985681366614e1,
      1.96520832956077098242e2, 5.26445194995477358631e2, 9.34528527171957607540e2, 1.02755188689515710272e3,
      5.57535335369399327526e2)
_Q = (1.32281951154744992508e1, 8.67072140885989742329e1, 3.54937778887819891062e2, 9.75708501743205489753e2,
      1.82390916687909736289e3, 2.24633760818710981792e3, 1.65666309194161350182e3, 5.57535340817727675546e2)
_R = (5.64189583547755073984e-1, 1.27536670759978104416e0, 5.01905042251180477414e0, 6.16021097993053585195e0,
      7.40974269950448939160e0, 2.97886665372100240670e0)
_S = (2.26052863220117276590e0, 9.39603524938001434673e0, 1.20489539808096656605e1, 1.70814450747565897222e1,
      9.60896809063285878198e0, 3.36907645100081516050e0)


def _polevl(x, c):
    y = jnp.full_like(x, c[0])
    for a in c[1:]:
        y = y * x + a
    return y


def _p1evl(x, c):
    y = x + c[0]
    for a in c[1:]:
        y = y * x + a
    return y


def erf_cephes(x):
    """erf(x) for |x| < 1 (Cephes); outside, the value is not used."""
    z = x * x
    return x * _polevl(z, _T) / _p1evl(z, _U)


def erfc_cephes(a):
    """erfc(a) for a >= 0 (Cephes: 1 - erf below 1, rational approximations above)."""
    lo = a < 1.0
    b = jnp.where(lo, 1.0, a)
    z = jnp.exp(-b * b)
    small = b < 8.0
    p = jnp.where(small, _polevl(b, _P), _polevl(b, _R))
    q = jnp.where(small, _p1evl(b, _Q), _p1evl(b, _S))
    return jnp.where(lo, 1.0 - erf_cephes(jnp.where(lo, a, 0.0)), z * p / q)


def ndtr(x, erf=lax.erf, erfc=lax.erfc):
    """jax.scipy.special.ndtr with a pluggable erf pair (lax: bit for bit jax's; Cephes: inside the kernel)."""
    h = x.dtype.type(0.5) * np.sqrt(2.0, dtype=x.dtype)
    w = x * h
    z = jnp.abs(w)
    y = jnp.where(z < h, 1.0 + erf(jnp.where(z < h, w, 0.0)), jnp.where(w > 0, 2.0 - erfc(z), erfc(z)))
    return 0.5 * y


_NP0 = (-5.99633501014107895267e1, 9.80010754185999661536e1, -5.66762857469070293439e1, 1.39312609387279679503e1,
        -1.23916583867381258016e0)
_NQ0 = (1.0, 1.95448858338141759834e0, 4.67627912898881538453e0, 8.63602421390890590575e1,
        -2.25462687854119370527e2, 2.00260212380060660359e2, -8.20372256168333339912e1, 1.59056225126211695515e1,
        -1.18331621121330003142e0)
_NP1 = (4.05544892305962419923e0, 3.15251094599893866154e1, 5.71628192246421288162e1, 4.40805073893200834700e1,
        1.46849561928858024014e1, 2.18663306850790267539e0, -1.40256079171354495875e-1, -3.50424626827848203418e-2,
        -8.57456785154685413611e-4)
_NQ1 = (1.0, 1.57799883256466749731e1, 4.53907635128879210584e1, 4.13172038254672030440e1, 1.50425385692907503408e1,
        2.50464946208309415979e0, -1.42182922854787788574e-1, -3.80806407691578277194e-2,
        -9.33259480895457427372e-4)
_NP2 = (3.23774891776946035970e0, 6.91522889068984211695e0, 3.93881025292474443415e0, 1.33303460815807542389e0,
        2.01485389549179081538e-1, 1.23716634817820021358e-2, 3.01581553508235416007e-4, 2.65806974686737550832e-6,
        6.23974539184983293730e-9)
_NQ2 = (1.0, 6.02427039364742014255e0, 3.67983563856160859403e0, 1.37702099489081330271e0, 2.16236993594496635890e-1,
        1.34204006088543189037e-2, 3.28014464682127739104e-4, 2.89247864745380683936e-6, 6.79019408009981274425e-9)


def _polyval(c, x):
    y = jnp.zeros_like(x)
    for a in c:
        y = y * x + x.dtype.type(a)
    return y


def ndtri(p):
    """jax.scipy.special.ndtri with scalar coefficients (a Pallas kernel cannot capture arrays): same arithmetic."""
    t = p.dtype.type
    mcp = jnp.where(p > t(-np.expm1(-2.0)), t(1.0) - p, p)
    mcp = jnp.where(mcp == t(0.0), t(0.5), mcp)
    w = mcp - t(0.5)
    ww = w * w
    big = (w + w * ww * (_polyval(_NP0, ww) / _polyval(_NQ0, ww))) * -t(np.sqrt(2.0 * np.pi))
    z = jnp.sqrt(t(-2.0) * jnp.log(mcp))
    first = z - jnp.log(z) / z
    small = first - _polyval(_NP2, 1 / z) / _polyval(_NQ2, 1 / z) / z
    other = first - _polyval(_NP1, 1 / z) / _polyval(_NQ1, 1 / z) / z
    x = jnp.where(mcp > t(np.exp(-2.0)), big, jnp.where(z >= t(8.0), small, other))
    x = jnp.where(p > t(1.0 - np.exp(-2.0)), x, -x)
    return jnp.where(p == t(0.0), -t(np.inf), jnp.where(p == t(1.0), t(np.inf), x))


def _gauss_clipped(u, eav, sig2, erfs):
    """eloss_jax._gauss_clipped with the given erf pair."""
    sig = jnp.sqrt(jnp.maximum(sig2, 0.0))
    flat = eav < 0.25 * sig
    b = jnp.where(sig2 > 0, eav / jnp.where(sig2 > 0, sig, 1.0), 1.0)
    pa = ndtr(-b, *erfs)
    z = jnp.clip(ndtri(pa + u * (1 - 2 * pa)), -b, b)
    return jnp.where(flat, eav + (2 * u - 1) * eav, jnp.where(sig2 > 0, eav + sig * z, 0.0))


# --------------------------------------------------------------------------- loss sampler on raw keys

def _gamma(k, alpha, rounds, need=None, any_=None):
    """eloss_jax._gamma_fixed on a raw key (same draws); rounds stop once every lane in `need` has accepted (the
    later rounds change only the other lanes)."""
    dt = alpha.dtype
    kb, kr = fold_in(k, 0), fold_in(k, 1)
    a1 = jnp.where(alpha < 1, alpha + 1, alpha)
    d = a1 - 1.0 / 3.0
    cc = 1.0 / jnp.sqrt(9 * d)
    tiny = dprec.tiny_of(dt)

    def body(i, carry):
        val, done = carry
        ki = fold_in(kr, i)
        x = normal(fold_in(ki, 0), dt)
        v = (1 + cc * x) ** 3
        lu = jnp.log(uniform(fold_in(ki, 1), dt, 0, tiny))
        ok = (v > 0) & (lu < 0.5 * x * x + d - d * v + d * jnp.log(jnp.where(v > 0, v, 1.0)))
        take = ok & ~done
        return jnp.where(take, d * v, val), done | ok

    if any_ is None:
        val, _ = lax.fori_loop(0, rounds, body, (d, jnp.zeros(alpha.shape, bool)))
    else:
        _, val, _ = lax.while_loop(lambda s: (s[0] < rounds) & any_(need & ~s[2]),
                                   lambda s: (s[0] + 1,) + body(s[0], s[1:]), (0, d, jnp.zeros(alpha.shape, bool)))
    ub = uniform(kb, dt, 0, tiny)
    return jnp.where(alpha < 1, val * ub ** (1 / jnp.where(alpha > 0, alpha, 1.0)), val)


def sample(k, c, p, length, tcut, mass=MMU, K=K_POISSON, gamma_rounds=GAMMA_ROUNDS, erfs=(lax.erf, lax.erfc),
           any_=None):
    """eloss_jax._sample(key, c, p, length, tcut, mass) (gaussian=True, mean and tmax from p) on a raw key k with the
    same draws; arrays broadcast to p's shape, c fields scalars or arrays of that shape. With any_ (a reduction to one
    flag, inside a kernel: per block) a regime no lane is in is skipped and the Poisson and Gamma rounds stop when no
    lane can change; the values of the lanes that use them are unchanged. Without it every branch and round runs (as
    eloss_jax._sample; the form for vmap)."""
    dt = p.dtype
    length = jnp.broadcast_to(jnp.asarray(length, dt), p.shape)
    tcut = jnp.broadcast_to(jnp.asarray(tcut, dt), p.shape)
    c = MatConsts(*(jnp.broadcast_to(jnp.asarray(v, dt), p.shape) for v in c))
    ks = [fold_in(k, j) for j in range(7)]
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    cut = jnp.minimum(tcut, tmax)
    mean = _dedx(c, p, tcut, mass) * length
    bohr = (mean >= _BOHR_MIN * cut) & (tmax <= 2 * cut)
    siga = jnp.sqrt(jnp.clip((tmax / beta2 - 0.5 * cut) * TWOPI_MC2_RCL2 * length * c.ne, 0, None))
    sn = mean / jnp.where(siga > 0, siga, 1)
    gs = bohr & (sn >= 2)
    gm = bohr & (sn < 2)
    zero = jnp.zeros_like(p)
    lazy = (lambda m, f: lax.cond(any_(m), f, lambda: zero)) if any_ is not None else (lambda m, f: f())
    x_gs = lazy(gs, lambda: _gauss_clipped(uniform(ks[0], dt), mean, siga ** 2, erfs))
    neff = jnp.where(gm, sn * sn, 1.0)
    x_gm = lazy(gm, lambda: mean * _gamma(ks[1], neff, gamma_rounds, gm, any_) / neff)
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
    cont1, cont2 = a1 > _NMAXCONT, a2 > _NMAXCONT
    eav = jnp.where(cont1, a1 * e1, 0.0) + jnp.where(cont2, a2 * e2, 0.0)
    s2 = jnp.where(cont1, a1 * e1 * e1, 0.0) + jnp.where(cont2, a2 * e2 * e2, 0.0)
    lam = (jnp.where(cont1 | (a1 <= 0), 0.0, a1), jnp.where(cont2 | (a2 <= 0), 0.0, a2))
    alfa, alfa1, namean, p3 = _ion_split(a3, w1)
    eav3 = namean * c.e0 * alfa1
    s23 = c.e0 * c.e0 * namean * (alfa - alfa1 * alfa1)
    w3 = alfa * c.e0
    has = tc > w3
    w = jnp.where(has, (tc - w3) / tc, 0.0)
    lam = lam + (jnp.where(has, jnp.clip(p3, 0, None), 0.0),)
    uu = tuple(uniform(ks[2], dt, q) for q in range(3))

    def body(j, carry):
        pmf, cdf, kk, ion = carry
        above = tuple(a > b for a, b in zip(uu, cdf))
        r = uniform(fold_in(ks[3], j), dt)
        ion = ion + jnp.where(above[2], w3 / (1 - w * r), 0.0)
        pmf = tuple(f * lm / (j + 1) for f, lm in zip(pmf, lam))
        return pmf, tuple(a + b for a, b in zip(cdf, pmf)), tuple(n + a for n, a in zip(kk, above)), ion

    pmf0 = tuple(jnp.exp(-lm) for lm in lam)
    zi = jnp.zeros(p.shape, jnp.int32)
    def more(st):
        j, pmf, cdf = st[0], st[1], st[2]
        return (j < K) & any_(u & ((uu[0] > cdf[0]) | (uu[1] > cdf[1]) | (uu[2] > cdf[2])))

    init = (pmf0, pmf0, (zi, zi, zi), jnp.zeros_like(p))
    if any_ is None:
        _, _, kk, ion = lax.fori_loop(0, K, body, init)
    else:
        _, _, _, kk, ion = lax.while_loop(more, lambda st: (st[0] + 1,) + body(st[0], st[1:]), (0,) + init)
    rx0, rx1 = uniform(ks[4], dt, 0), uniform(ks[4], dt, 1)
    loss = jnp.where(kk[0] > 0, ((kk[0] + 1) - 2 * rx0) * e1, 0.0)
    loss = loss + jnp.where(kk[1] > 0, ((kk[1] + 1) - 2 * rx1) * e2, 0.0)
    g2 = lazy(u & (s2 > 0), lambda: _gauss_clipped(uniform(ks[5], dt), eav, s2, erfs))
    loss = loss + jnp.where(s2 > 0, g2, 0.0)
    loss = loss + ion
    g3 = lazy(u & (s23 > 0), lambda: _gauss_clipped(uniform(ks[6], dt), eav3, jnp.where(s23 > 0, s23, 0.0), erfs))
    loss = loss + jnp.where(s23 > 0, g3, 0.0)
    out = jnp.where(gs, x_gs, jnp.where(gm, x_gm, mean))
    return jnp.where(u, loss * scaling, out)


# --------------------------------------------------------------------------- transport.slab_loss per lane

def _pending(tk0, s):
    i, left, lost = s
    return (i < tr.N_SLICES_MAX) & (left > 0) & (tk0 - lost > tr.TK_STOP)


def _slice(k, c, tk0, s, erfs, any_=None, lazy=False):
    """One transport.slab_loss slice for the pending lanes (slice keys fold_in(k, i))."""
    i, left, lost = s
    go = _pending(tk0, s)
    tk = tk0 - lost
    pg = jnp.sqrt(jnp.clip((tk + MMU) ** 2 - MMU ** 2, 1e-6, None))
    dedx = _dedx(c, pg, 1e9, MMU)
    dedx_s = jnp.where(dedx > 0, dedx, 1.0)
    need = dedx * left
    coarse = tk > 3.0 * need
    step = jnp.where(coarse, left / tr.MAX_SLICES_PER_SLAB, tr.SLICE_FRAC * tk / dedx_s)
    step = jnp.minimum(left, jnp.maximum(step, tr.SLICE_FRAC * tk / dedx_s))
    step = jnp.where(left <= step * 1.0001, left, step)
    dl = sample(fold_in(k, i), c, pg, jnp.where(go, step, 0.0), 1e9, MMU, erfs=erfs, any_=any_ if lazy else None)
    dl = jnp.where(dl >= tk, dedx * step, dl)
    return i + go, jnp.where(go, left - step, left), jnp.where(go, lost + dl, lost)


def _finish(c, tk0, s):
    """Mean step beyond N_SLICES_MAX, the stop and the cap at the kinetic energy (transport.slab_loss's tail)."""
    _, left, lost = s
    rest = (left > 0) & (tk0 - lost > tr.TK_STOP)
    tk = tk0 - lost
    pg = jnp.sqrt(jnp.clip((tk + MMU) ** 2 - MMU ** 2, 1e-6, None))
    lost = jnp.where(rest, lost + _dedx(c, pg, 1e9, MMU) * left, lost)
    return jnp.where(tk0 - lost <= tr.TK_STOP, tk0, jnp.minimum(lost, tk0))


def _lanes(k, c, p, length, erfs, any_):
    tk0 = jnp.sqrt(p * p + MMU * MMU) - MMU
    s = (jnp.zeros(p.shape, jnp.int32), length, jnp.zeros_like(p))
    lazy = any_ is not jnp.any    # the kernel's per-block reduction: skip unused regimes and rounds
    s = lax.while_loop(lambda s: any_(_pending(tk0, s)), lambda s: _slice(k, c, tk0, s, erfs, any_, lazy), s)
    return _finish(c, tk0, s), s[0]


def _kernel(k0, k1, p, length, *refs):
    *cref, loss_ref, n_ref = refs
    c = MatConsts(*(r[...] for r in cref))
    loss, n = _lanes((k0[...], k1[...]), c, p[...], length[...], (erf_cephes, erfc_cephes),
                     lambda x: jnp.max(x.astype(jnp.int32)) > 0)
    loss_ref[...] = loss
    n_ref[...] = n


def work_order(c, p, length):
    """Lane order by expected slice count (fewest first): tk0 / (dE/dx L) ascending is the order of most slices, so
    the kernel's blocks hold lanes of similar loop length."""
    tk0 = jnp.sqrt(p * p + MMU * MMU) - MMU
    need = _dedx(c, jnp.maximum(p, 1e-3), 1e9, MMU) * length
    r = jnp.where(length > 0, tk0 / jnp.maximum(need, 1e-30), jnp.inf)
    return jnp.argsort(-r, stable=True)


RAD_FOLD = 0x52414401   # radiative stream of a slab: fold_in(slab key, RAD_FOLD), apart from the slice keys


def radiative(k, rad, p, length, loss, mat=None):
    """transport.slab_loss's radiative step after the ionisation slices (muradiative_jax.after_ionisation): raw keys
    k [N], RadTable rad (muradiative_jax.table, or stack with per-lane index mat), slab-entry |p|, length, ionisation
    loss [N]. The draw uses the typed key fold_in(k, RAD_FOLD) per lane."""
    from . import muradiative_jax as RJ
    keys = jax.random.wrap_key_data(jnp.stack(fold_in(k, RAD_FOLD), -1))
    return RJ.after_ionisation(keys, rad, p, length, loss, mat)


@partial(jax.jit, static_argnames=("impl", "block", "sort"))
def slab_loss(k, c, p, length, impl="pallas", block=BLOCK, sort=True, rad=None, rad_mat=None):
    """transport.slab_loss per lane: raw keys k = (k0, k1) [N] (slice keys fold_in(k, slice)), MatConsts c (fields
    scalars or [N]), |p| and length [N]. Returns (loss [N], slices taken [N]); lanes with length 0 lose nothing.
    sort (kernel only): run the lanes in work_order (each lane's result does not depend on its position).
    rad: RadTable of the material(s) (rad_mat: per-lane index into a stacked table) for transport.slab_loss's default
    radiative losses, drawn after the kernel (radiative()); None gives the ionisation-only loss (radiative=False)."""
    if rad is not None:
        loss, slices = slab_loss(k, c, p, length, impl, block, sort)
        return radiative(k, rad, p, jnp.asarray(length, p.dtype), loss, rad_mat), slices
    dt = p.dtype
    length = jnp.asarray(length, dt)
    c = MatConsts(*(jnp.broadcast_to(jnp.asarray(v, dt), p.shape) for v in c))
    if impl != "xla" and sort:
        order = work_order(c, p, length)
        g = lambda x: x[order]
        loss, slices = slab_loss((g(k[0]), g(k[1])), MatConsts(*map(g, c)), g(p), g(length), impl, block, False)
        inv = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0]))
        return loss[inv], slices[inv]
    if impl == "xla":
        one = lambda kk0, kk1, cc, pp, ll: _lanes((kk0, kk1), cc, pp, ll, (lax.erf, lax.erfc), jnp.any)
        return jax.vmap(one)(k[0], k[1], c, p, length)
    if impl not in IMPLS:
        raise ValueError(f"slab_loss impl must be one of {IMPLS}")
    from jax.experimental import pallas as pl
    n = p.shape[0]
    npad = -(-max(n, 1) // block) * block
    pad = lambda x, v=0: jnp.pad(x, (0, npad - n), constant_values=v)
    args = [pad(k[0]), pad(k[1]), pad(p, 1000.0), pad(length)] + [pad(f, 1.0) for f in c]
    spec = pl.BlockSpec((block,), lambda i: (i,))
    loss, slices = pl.pallas_call(
        _kernel, grid=(npad // block,), in_specs=[spec] * len(args), out_specs=[spec, spec],
        out_shape=[jax.ShapeDtypeStruct((npad,), dt), jax.ShapeDtypeStruct((npad,), jnp.int32)],
        interpret=impl == "interpret")(*args)
    return loss[:n], slices[:n]


# --------------------------------------------------------------------------- a path of segments in one kernel

def p_after(p, loss):
    """|p| after losing `loss` of kinetic energy (0 below rest)."""
    tk = jnp.sqrt(p * p + MMU * MMU) - MMU - loss
    return jnp.sqrt(jnp.maximum((jnp.maximum(tk, 0) + MMU) ** 2 - MMU ** 2, 0))


def _segment(k, p, lost, c, f1, f2, mean, L, erfs, any_):
    """One path segment: mean loss at max(|p|, 1) (mean) or slab_loss with pair key fold_in(fold_in(k, f1), f2)."""
    pp = p_after(p, lost)
    hit = (L > 0) & ~mean
    dl, _ = _lanes(fold_in(fold_in(k, f1), f2), c, pp, jnp.where(hit, L, 0.0), erfs, any_)
    dm = _dedx(c, jnp.maximum(pp, 1.0), 1e9, MMU) * L
    return lost + jnp.where(hit, dl, 0.0) + jnp.where(mean, dm, 0.0)


def _path_kernel(k0, k1, p, L, f1, f2, mean, *refs):
    *cref, out_ref = refs
    kk, pp = (k0[...], k1[...]), p[...]
    any_ = lambda x: jnp.max(x.astype(jnp.int32)) > 0

    def seg(j, lost):
        c = MatConsts(*(r[j] for r in cref))
        lost = _segment(kk, pp, lost, c, f1[j], f2[j], mean[j] > 0, L[j, :], (erf_cephes, erfc_cephes), any_)
        out_ref[j, :] = lost
        return lost

    lax.fori_loop(0, L.shape[0], seg, jnp.zeros_like(pp))


def path_radiative(k, p, cum, L, f1, f2, mean, rad, rad_mat):
    """Radiative losses added to a path_loss result: per segment j in order, the draw of slab j (pair key
    fold_in(fold_in(k, f1[j]), f2[j]), radiative()) at the |p| entering it (ionisation and radiative losses of the
    segments before), for muons still above TK_STOP; a slab's ionisation plus radiative loss is capped at the kinetic
    energy entering it. The ionisation of later segments keeps the |p| without the earlier radiative losses (a change
    of their dE/dx below 1e-4)."""
    J = L.shape[0]
    tk0 = jnp.sqrt(p * p + MMU * MMU) - MMU
    extra, out, prev = jnp.zeros_like(p), [], jnp.zeros_like(p)
    for j in range(J):
        ion_j = cum[j] - prev
        prev = cum[j]
        entry = cum[j] - ion_j + extra
        pin = p_after(p, entry)
        hit = (L[j] > 0) & ~mean[j]
        ion_c = jnp.minimum(ion_j, jnp.maximum(tk0 - entry, 0.0))
        kj = fold_in(fold_in(k, f1[j]), f2[j])
        tot = radiative(kj, rad, pin, jnp.where(hit, L[j], 0.0), ion_c, jnp.full(p.shape, rad_mat[j], jnp.int32))
        extra = extra + jnp.where(hit, tot - ion_c, 0.0)
        out.append(cum[j] + extra)        # as the ionisation-only path: mean segments are not capped at the stop
    return jnp.stack(out)


@partial(jax.jit, static_argnames=("impl", "block"))
def path_loss(k, p, L, c, f1, f2, mean, impl="pallas", block=32, rad=None, rad_mat=None):
    """Kinetic energy lost along a path of J segments crossed in order (the momentum carried from one to the next):
    lanes k = (k0, k1) raw keys [N], |p| at the start [N], lengths L [J, N]; per segment MatConsts c (fields [J]),
    key folds f1, f2 [J] (slab pair key fold_in(fold_in(k, f1), f2)) and `mean` [J] (mean dE/dx at max(|p|, 1) instead
    of a sliced, sampled slab). Returns the cumulative loss after each segment [J, N]. Segment by segment this is
    slab_loss / a mean step; one kernel runs the whole path (no host step between segments).
    rad, rad_mat: a stacked RadTable and the material index per segment [J] for transport.slab_loss's default
    radiative losses (path_radiative, after the kernel); None gives the ionisation-only path."""
    if rad is not None:
        cum = path_loss(k, p, L, c, f1, f2, mean, impl, block)
        return path_radiative(k, p, cum, jnp.asarray(L, p.dtype), jnp.asarray(f1, jnp.uint32),
                              jnp.asarray(f2, jnp.uint32), jnp.asarray(mean, bool), rad, jnp.asarray(rad_mat))
    dt = p.dtype
    J, n = L.shape
    L = jnp.asarray(L, dt)
    c = MatConsts(*(jnp.asarray(v, dt) for v in c))
    f1, f2 = jnp.asarray(f1, jnp.uint32), jnp.asarray(f2, jnp.uint32)
    mean = jnp.asarray(mean, bool)
    if impl == "xla":
        def one(kk0, kk1, pp, ll):
            out, lost = [], jnp.zeros_like(pp)
            for j in range(J):
                lost = _segment((kk0, kk1), pp, lost, MatConsts(*(f[j] for f in c)), f1[j], f2[j], mean[j], ll[j],
                                (lax.erf, lax.erfc), jnp.any)
                out.append(lost)
            return jnp.stack(out)
        return jax.vmap(one, in_axes=(0, 0, 0, 1), out_axes=1)(k[0], k[1], p, L)
    if impl not in IMPLS:
        raise ValueError(f"path_loss impl must be one of {IMPLS}")
    from jax.experimental import pallas as pl
    npad = -(-max(n, 1) // block) * block
    jpad = 1 << max(J - 1, 0).bit_length()
    lane = lambda x, v=0: jnp.pad(x, (0, npad - n), constant_values=v)
    seg = lambda x, v=0: jnp.pad(x, (0, jpad - J), constant_values=v)
    args = [lane(k[0]), lane(k[1]), lane(p, 1000.0), jnp.pad(L, ((0, jpad - J), (0, npad - n))), seg(f1), seg(f2),
            seg(mean.astype(jnp.int32))] + [seg(f, 1.0) for f in c]
    ls, rs, ss = pl.BlockSpec((block,), lambda i: (i,)), pl.BlockSpec((jpad, block), lambda i: (0, i)), \
        pl.BlockSpec((jpad,), lambda i: (0,))
    out = pl.pallas_call(
        _path_kernel,
        grid=(npad // block,), in_specs=[ls, ls, ls, rs] + [ss] * (3 + len(c)), out_specs=rs,
        out_shape=jax.ShapeDtypeStruct((jpad, npad), dt), interpret=impl == "interpret")(*args)
    return out[:J, :n]
