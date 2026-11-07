"""Shared frozen-field helix evaluation and padded, uncertified enclosures."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

K_FIELD = 0.299792458
_PAD = 512 * 2.220446049250313e-16


class HelixBounds(NamedTuple):
    """Lane boxes (n,3), surface/derivative intervals (n,4), and validity (n,)."""
    x_lo: object
    x_hi: object
    u_lo: object
    u_hi: object
    f_lo: object
    f_hi: object
    df_lo: object
    df_hi: object
    supported: object


class HelixEvaluation(NamedTuple):
    """Position x and raw derivative u (n,3), with support status (n,)."""
    x: object
    u: object
    supported: object


class _Helix(NamedTuple):
    x: object
    u: object
    parallel: object
    perpendicular: object
    w: object
    omega: object
    slo: object
    shi: object
    supported: object


def _array(value, shape):
    """Reject nonnumeric and non-float64 floating inputs without silent promotion."""
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    try:
        value = jnp.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return jnp.zeros(shape, dtype=dtype), False
    good = (value.dtype == jnp.float64 or value.dtype.kind in 'iu') and jax.config.x64_enabled
    value = value.astype(dtype) if good else jnp.zeros(value.shape, dtype=dtype)
    return jnp.broadcast_to(value, shape), good


def _norm(v):
    return jnp.hypot(jnp.hypot(v[:, 0], v[:, 1]), v[:, 2])


def _magnitude_bits(v):
    # The barrier prevents JIT from folding bit guards into flush-to-zero float comparisons.
    dtype = jnp.uint64 if v.dtype == jnp.float64 else jnp.uint32
    mask = 0x7fffffffffffffff if v.dtype == jnp.float64 else 0x7fffffff
    bits = jax.lax.optimization_barrier(jax.lax.bitcast_convert_type(v, dtype))
    return bits & dtype(mask)


def _zero(v):
    return _magnitude_bits(v) == 0


def _normal_or_zero(v):
    bits = _magnitude_bits(v)
    minimum = 0x0010000000000000 if v.dtype == jnp.float64 else 0x00800000
    return (bits == 0) | (bits >= jnp.asarray(minimum, dtype=bits.dtype))


def _product(al, ah, bl, bh):
    products = jnp.stack((al * bl, al * bh, ah * bl, ah * bh))
    return jnp.min(products, axis=0), jnp.max(products, axis=0)


def _prepare(x, u, p, q, B, slo, shi):
    """Validate and sanitize the common trajectory domain for both public APIs."""
    shape = jnp.shape(x)
    if len(shape) != 2 or shape[1:] != (3,) or jnp.shape(u) != shape:
        raise ValueError('x and u must have identical (n,3) shapes')
    n = shape[0]
    x, xok = _array(x, shape)
    u, uok = _array(u, shape)
    B, bok = _array(jnp.zeros_like(x) if B is None else B, shape)
    converted = [_array(v, (n,)) for v in (p, q, slo, shi)]
    p, q, slo, shi = [v[0] for v in converted]
    scalars = jnp.stack((p, q, slo, shi))
    bm, speed = _norm(B), _norm(u)
    valid = (xok & uok & bok & all(v[1] for v in converted)
             & jnp.all(jnp.isfinite(x) & jnp.isfinite(u) & jnp.isfinite(B), axis=1)
             & jnp.all(jnp.isfinite(scalars), axis=0)
             & jnp.all(jnp.abs(x) <= 1e8, axis=1) & (jnp.abs(speed - 1.) <= 1e-12)
             & (p >= 1e-6) & (p <= 1e9) & (jnp.abs(q) <= 100.)
             & (_zero(q) | (jnp.abs(q) >= 1e-100))
             & (bm <= 1e4) & (jnp.all(_zero(B), axis=1) | (bm >= 1e-100))
             & _normal_or_zero(slo) & _normal_or_zero(shi)
             & (slo >= 0) & (shi >= slo) & (shi <= 1e8))
    # Sanitize invalid lanes before arithmetic; they remain explicitly unsupported.
    x, u, B = [jnp.where(valid[:, None], v, 0.) for v in (x, u, B)]
    p = jnp.where(valid, p, 1.)
    q, slo, shi = [jnp.where(valid, v, 0.) for v in (q, slo, shi)]
    bm = _norm(B)
    bh = jnp.where((bm > 0)[:, None], B / jnp.where(bm > 0, bm, 1.)[:, None],
                   jnp.array([0., 0., 1.]))
    parallel = jnp.sum(u * bh, axis=1)[:, None] * bh
    perpendicular, w = u - parallel, jnp.cross(u, bh)
    omega = K_FIELD * q * bm / p
    valid = valid & (jnp.abs(omega) * shi <= 1e4)
    vectors = [jnp.where(valid[:, None], v, 0.) for v in (x, u, parallel, perpendicular, w)]
    scalars = [jnp.where(valid, v, 0.) for v in (omega, slo, shi)]
    return _Helix(*vectors, *scalars, valid)


def _evaluate_at(helix, s):
    """Use one stable phase expression for point evaluation and enclosure centers."""
    phase = helix.omega * s
    denominator = jnp.where(phase != 0, phase, 1.)
    sinc = jnp.where(jnp.abs(phase) < 1e-4, 1 - phase**2 / 6 + phase**4 / 120,
                     jnp.sin(phase) / denominator)
    cosc = 2 * jnp.sin(phase * .5)**2 / denominator
    x = helix.x + s[:, None] * (helix.parallel + sinc[:, None] * helix.perpendicular
                               + cosc[:, None] * helix.w)
    u = helix.parallel + jnp.cos(phase)[:, None] * helix.perpendicular + jnp.sin(phase)[:, None] * helix.w
    valid = helix.supported & jnp.all(jnp.isfinite(x) & jnp.isfinite(u), axis=1)
    return HelixEvaluation(jnp.where(valid[:, None], x, jnp.nan),
                           jnp.where(valid[:, None], u, jnp.nan), valid)


def evaluate(x, u, p, q, B, s):
    """Return frozen-helix position, raw dx/ds, and lane support status.

    x/u have shape (n,3); B broadcasts there (None means zero); p/q/s broadcast
    to (n,). This uses enclose's trajectory domain and stable phase arithmetic,
    not eslab_jax._helix's normalized direction or cancellation-prone cosc.
    Unsupported lanes return NaN x/u; malformed shapes raise ValueError.
    """
    helix = _prepare(x, u, p, q, B, s, s)
    return _evaluate_at(helix, helix.slo)


def enclose(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x=0., axis_y=0.):
    """Enclose a frozen helix and four inward-positive annular surface functions.

    x/u have shape (n,3); B broadcasts there (None means zero); other arguments
    broadcast to (n,). Surfaces are lower face, upper face, outer rim, inner hole.
    u and surface derivatives are raw derivatives with respect to s, not time.
    Unsupported lanes return all-NaN numeric outputs; shape errors raise ValueError.
    See docs/helix-bounds.md for the restricted domain and uncertified padding.
    """
    helix = _prepare(x, u, p, q, B, slo, shi)
    x, u, w, omega, slo, shi = helix.x, helix.u, helix.w, helix.omega, helix.slo, helix.shi
    converted = [_array(v, (x.shape[0],)) for v in (z, half, inner, outer, axis_x, axis_y)]
    z, half, inner, outer, axis_x, axis_y = [v[0] for v in converted]
    geometry = jnp.stack((z, half, inner, outer, axis_x, axis_y))
    valid = (helix.supported & all(v[1] for v in converted)
             & jnp.all(jnp.isfinite(geometry) & _normal_or_zero(geometry)
                       & (jnp.abs(geometry) <= 1e8), axis=0)
             & (half > 0) & (inner >= 0) & (outer > inner))
    z, half, inner, outer, axis_x, axis_y = [
        jnp.where(valid, v, 0.) for v in (z, half, inner, outer, axis_x, axis_y)]
    phase_max = jnp.abs(omega) * shi
    mid, h = slo + (shi - slo) * .5, (shi - slo) * .5
    center = _evaluate_at(helix, mid)
    xm, um = center.x, center.u
    valid = valid & center.supported
    speed = _norm(u) * (1 + _PAD)
    kappa = jnp.abs(omega) * (_norm(w) + _PAD * speed) * (1 + _PAD)
    # In exact arithmetic, Taylor and arc-length boxes contain the entire interval.
    xr = jnp.minimum(jnp.abs(um) * h[:, None] + (.5 * kappa * h**2)[:, None],
                     (speed * h)[:, None])
    ur = jnp.minimum(kappa * h, 2 * speed)[:, None]
    xp = _PAD * (1 + jnp.abs(x) + shi[:, None]) * (1 + phase_max[:, None])
    up = _PAD * (1 + phase_max[:, None])
    xl, xh, ul, uh = xm - xr - xp, xm + xr + xp, um - ur - up, um + ur + up
    axis = jnp.stack((axis_x, axis_y), axis=1)
    rp = _PAD * (1 + jnp.abs(axis) + jnp.maximum(jnp.abs(xl[:, :2]), jnp.abs(xh[:, :2])))
    rl, rh = xl[:, :2] - axis - rp, xh[:, :2] - axis + rp
    r2l = jnp.sum(jnp.where((rl <= 0) & (rh >= 0), 0., jnp.minimum(rl**2, rh**2)), axis=1)
    r2h = jnp.sum(jnp.maximum(rl**2, rh**2), axis=1)
    dl, dh = _product(rl, rh, ul[:, :2], uh[:, :2])
    dl, dh = 2 * jnp.sum(dl, axis=1), 2 * jnp.sum(dh, axis=1)
    fl = jnp.stack((xl[:, 2] - z + half, z + half - xh[:, 2], outer**2 - r2h, r2l - inner**2), axis=1)
    fh = jnp.stack((xh[:, 2] - z + half, z + half - xl[:, 2], outer**2 - r2l, r2h - inner**2), axis=1)
    dfl = jnp.stack((ul[:, 2], -uh[:, 2], -dh, dl), axis=1)
    dfh = jnp.stack((uh[:, 2], -ul[:, 2], -dl, dh), axis=1)
    ap = _PAD * (1 + jnp.abs(z) + half + jnp.maximum(jnp.abs(xl[:, 2]), jnp.abs(xh[:, 2])))
    fp = _PAD * (1 + r2h + inner**2 + outer**2)
    dp = _PAD * (1 + jnp.maximum(jnp.abs(dl), jnp.abs(dh)))
    fpad = jnp.stack((ap, ap, fp, fp), axis=1)
    dpad = jnp.stack((_PAD + up[:, 0], _PAD + up[:, 0], dp, dp), axis=1)
    arrays = (xl, xh, ul, uh, fl - fpad, fh + fpad, dfl - dpad, dfh + dpad)
    valid = valid & jnp.all(jnp.stack([jnp.all(jnp.isfinite(a), axis=1) for a in arrays]), axis=0)
    return HelixBounds(*(jnp.where(valid[:, None], a, jnp.nan) for a in arrays), valid)
