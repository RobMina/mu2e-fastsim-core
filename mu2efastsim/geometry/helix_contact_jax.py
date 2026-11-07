"""DEVELOPMENT only: chronological surface-zero candidates using uncertified boxes."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mu2efastsim.geometry import helix_bounds_jax as HB

CANDIDATE = 0
TRANSVERSE_BRACKET = 1
NO_CANDIDATE = 2
UNSUPPORTED = 3
PRECISION_UNRESOLVED = 4
WORK_LIMIT_UNRESOLVED = 5
START_UNRESOLVED = 6
_ACTIVE = -1


class ContactSearch(NamedTuple):
    """Status, interval endpoints, four-surface masks, and visited-node count."""
    status: object
    slo: object
    shi: object
    surfaces: object
    monotone: object
    work: object


class _State(NamedTuple):
    left: object
    right: object
    depth: object
    size: object
    result: ContactSearch


def _lane(value, n):
    # Match HB's scalar dtype policy without promoting unsupported float inputs.
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    try:
        value = jnp.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return jnp.zeros(n, dtype), False
    good = jax.config.x64_enabled and (value.dtype == jnp.float64 or value.dtype.kind in 'iu')
    value = value.astype(dtype) if good else jnp.zeros(value.shape, dtype)
    return jnp.broadcast_to(value, (n,)), good


def isolate(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x=0., axis_y=0.,
            *, atol=1e-6, max_work=512, max_depth=48):
    """Find the first unexcluded surface-zero interval, NOT a navigation distance.

    Inputs/domain follow HB.enclose; atol broadcasts to (n,) and must be positive
    finite float64 (or integer). max_work/max_depth are static Python integers.
    Earlier intervals must be excluded before any candidate is returned. Surface
    order is lower face, upper face, outer cylinder, inner cylinder (disabled at
    inner=0). Infinite planes/cylinders are NOT clipped to the annular volume.
    A candidate need not contain a zero; TRANSVERSE_BRACKET additionally has one
    possible surface, strict endpoint sign change, and a derivative excluding zero
    under the engineering boxes. No status is certified physics clearance.
    See docs/helix-contact.md for output and exhaustion semantics.
    """
    if type(max_work) is not int or max_work < 0:
        raise ValueError('max_work must be a static nonnegative Python integer')
    if type(max_depth) is not int or not 0 <= max_depth <= 64:
        raise ValueError('max_depth must be a static Python integer in [0,64]')
    domain = HB.enclose(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x, axis_y)
    n = domain.supported.shape[0]
    converted = [_lane(v, n) for v in (slo, shi, z, half, inner, outer, axis_x, axis_y, atol)]
    slo, shi, z, half, inner, outer, axis_x, axis_y, atol = [v[0] for v in converted]
    valid = domain.supported & all(v[1] for v in converted) & jnp.isfinite(atol) & (atol > 0)
    enabled = jnp.broadcast_to(jnp.arange(4) < 3, (n, 4)) | ((inner > 0)[:, None] & (jnp.arange(4) == 3))

    def box(lo, hi):
        return HB.enclose(x, u, p, q, B, lo, hi, z, half, inner, outer, axis_x, axis_y)

    def possible(bounds):
        return enabled & (bounds.f_lo <= 0) & (bounds.f_hi >= 0)

    start = box(slo, slo)
    start_on = possible(start)
    valid = valid & start.supported
    uncertain = valid & jnp.any(start_on, axis=1)
    status = jnp.where(valid, jnp.where(uncertain, START_UNRESOLVED, _ACTIVE), UNSUPPORTED)
    result = ContactSearch(status, jnp.where(uncertain, slo, jnp.nan),
                           jnp.where(uncertain, slo, jnp.nan), start_on & uncertain[:, None],
                           jnp.zeros((n, 4), bool), jnp.zeros(n, jnp.int32))
    shape = (n, max_depth + 1)
    left = jnp.zeros(shape, slo.dtype).at[:, 0].set(jnp.where(valid, slo, 0.))
    right = jnp.zeros(shape, slo.dtype).at[:, 0].set(jnp.where(valid, shi, 0.))
    state = _State(left, right, jnp.zeros(shape, jnp.int32), jnp.ones(n, jnp.int32), result)
    rows = jnp.arange(n)

    def top(st):
        index = jnp.maximum(st.size - 1, 0)
        return index, st.left[rows, index], st.right[rows, index], st.depth[rows, index]

    def condition(carry):
        iteration, st = carry
        return (iteration < max_work) & jnp.any(st.result.status == _ACTIVE)

    def visit(carry):
        iteration, st = carry
        out = st.result
        active = out.status == _ACTIVE
        index, lo, hi, depth = top(st)
        bounds = box(lo, hi)
        faces = possible(bounds)
        excluded = ~jnp.any(faces, axis=1)
        mid = lo + (hi - lo) * .5
        narrow = hi - lo <= atol
        stalled = (mid <= lo) | (mid >= hi)
        exhausted = depth >= max_depth
        split = active & bounds.supported & ~excluded & ~narrow & ~stalled & ~exhausted
        pop = active & bounds.supported & excluded
        done = active & ~split & ~pop
        finished = pop & (st.size == 1)
        code = jnp.where(~bounds.supported, UNSUPPORTED,
                         jnp.where(narrow, CANDIDATE,
                                   jnp.where(stalled, PRECISION_UNRESOLVED, WORK_LIMIT_UNRESOLVED)))
        status = jnp.where(done, code, jnp.where(finished, NO_CANDIDATE, out.status))
        record = done & bounds.supported
        out = ContactSearch(status, jnp.where(record, lo, out.slo), jnp.where(record, hi, out.shi),
                            jnp.where(record[:, None], faces, out.surfaces), out.monotone,
                            out.work + active.astype(jnp.int32))
        # Pop excluded nodes; replace split nodes by right then left, so left is next.
        left = st.left.at[rows, index].set(jnp.where(split, mid, st.left[rows, index]))
        right = st.right
        depths = st.depth.at[rows, index].set(jnp.where(split, depth + 1, st.depth[rows, index]))
        next_index = jnp.minimum(index + 1, max_depth)
        left = left.at[rows, next_index].set(jnp.where(split, lo, left[rows, next_index]))
        right = right.at[rows, next_index].set(jnp.where(split, mid, right[rows, next_index]))
        depths = depths.at[rows, next_index].set(jnp.where(split, depth + 1, depths[rows, next_index]))
        size = st.size + split.astype(jnp.int32) - pop.astype(jnp.int32)
        return iteration + 1, _State(left, right, depths, size, out)

    _, state = jax.lax.while_loop(condition, visit, (0, state))
    out = state.result
    waiting = out.status == _ACTIVE
    _, lo, hi, _ = top(state)
    out = out._replace(status=jnp.where(waiting, WORK_LIMIT_UNRESOLVED, out.status),
                       slo=jnp.where(waiting, lo, out.slo), shi=jnp.where(waiting, hi, out.shi))
    # Only final intervals need endpoint evaluation and singleton sign tests.
    finite = jnp.isfinite(out.slo) & jnp.isfinite(out.shi)
    lo, hi = jnp.where(finite, out.slo, 0.), jnp.where(finite, out.shi, 0.)
    bounds, low, high = box(lo, hi), box(lo, lo), box(hi, hi)
    a, b = HB.evaluate(x, u, p, q, B, lo), HB.evaluate(x, u, p, q, B, hi)

    def surface_values(point):
        radius2 = (point.x[:, 0] - axis_x)**2 + (point.x[:, 1] - axis_y)**2
        return jnp.stack((point.x[:, 2] - z + half, z + half - point.x[:, 2],
                          outer**2 - radius2, radius2 - inner**2), axis=1)

    fa, fb = surface_values(a), surface_values(b)
    # Retain both deciding and final possibilities; fused evaluations can differ by ulps.
    faces = (out.surfaces | possible(bounds)) & finite[:, None]
    increasing, decreasing = bounds.df_lo > 0, bounds.df_hi < 0
    monotone = faces & (increasing | decreasing)
    rising = (low.f_hi < 0) & (high.f_lo > 0) & (fa < 0) & (fb > 0) & increasing
    falling = (low.f_lo > 0) & (high.f_hi < 0) & (fa > 0) & (fb < 0) & decreasing
    checked = bounds.supported & low.supported & high.supported & a.supported & b.supported
    transverse = ((jnp.sum(faces, axis=1) == 1)
                  & jnp.any(faces & possible(bounds) & (rising | falling), axis=1))
    status = jnp.where((out.status == CANDIDATE) & transverse, TRANSVERSE_BRACKET, out.status)
    status = jnp.where(finite & ~checked, UNSUPPORTED, status)
    keep = finite & (status != UNSUPPORTED)
    return ContactSearch(status, jnp.where(keep, out.slo, jnp.nan), jnp.where(keep, out.shi, jnp.nan),
                         faces & keep[:, None], monotone & keep[:, None], out.work)
