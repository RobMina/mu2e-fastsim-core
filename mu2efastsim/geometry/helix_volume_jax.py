"""DEVELOPMENT only: clipped helix boundary candidates, never transport authority."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mu2efastsim.geometry import helix_bounds_jax as HB
from mu2efastsim.geometry import helix_contact_jax as HC

CANDIDATE = 0
ENTRY_BRACKET = 1
EXIT_BRACKET = 2
NO_CANDIDATE = 3
UNSUPPORTED = 4
PRECISION_UNRESOLVED = 5
WORK_LIMIT_UNRESOLVED = 6
START_UNRESOLVED = 7
_ACTIVE = -1

REGION_UNCERTAIN = 0
INTERIOR = 1
EXTERIOR = 2
REGION_UNSUPPORTED = 3


class VolumeBounds(NamedTuple):
    """HB bounds, strict whole-interval region, and clipped possible surfaces."""
    bounds: HB.HelixBounds
    region: object
    surfaces: object


class VolumeSearch(NamedTuple):
    """Candidate interval and diagnostics, not a root distance or navigator result."""
    status: object
    slo: object
    shi: object
    surfaces: object
    monotone: object
    work: object
    initial_region: object
    domain_region: object


def _enabled(inner):
    return (jnp.arange(4) < 3) | ((inner > 0)[:, None] & (jnp.arange(4) == 3))


def _classify(bounds, enabled):
    interior = jnp.all(~enabled | (bounds.f_lo > 0), axis=1)
    exterior = jnp.any(enabled & (bounds.f_hi < 0), axis=1)
    region = jnp.where(bounds.supported,
                       jnp.where(interior, INTERIOR, jnp.where(exterior, EXTERIOR, REGION_UNCERTAIN)),
                       REGION_UNSUPPORTED)
    faces = (enabled & (bounds.f_lo <= 0) & (bounds.f_hi >= 0)
             & ~exterior[:, None] & bounds.supported[:, None])
    return VolumeBounds(bounds, region, faces)


def enclose(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x=0., axis_y=0.):
    """Classify strict volume interior/exterior using HB's uncertified enclosures."""
    bounds = HB.enclose(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x, axis_y)
    inner, _ = HB._array(inner, (bounds.supported.shape[0],))
    return _classify(bounds, _enabled(inner))


def isolate(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x=0., axis_y=0.,
            *, atol=1e-6, max_work=512, max_depth=48):
    """Search left-first for clipped boundary candidates with bounded JAX stacks.

    Inputs follow HB.enclose; static limits follow HC.isolate. A unique monotone
    surface with strict opposite endpoint signs and every other enabled constraint
    strictly inside throughout may receive ENTRY_BRACKET or EXIT_BRACKET. All such
    claims are conditional on uncertified boxes. No midpoint/root distance, nudge,
    fallback, one-sided start convention, or transport authorization is provided.
    See docs/helix-volume.md for the complete development-only contract.
    """
    if type(max_work) is not int or max_work < 0:
        raise ValueError('max_work must be a static nonnegative Python integer')
    if type(max_depth) is not int or not 0 <= max_depth <= 64:
        raise ValueError('max_depth must be a static Python integer in [0,64]')
    domain = HB.enclose(x, u, p, q, B, slo, shi, z, half, inner, outer, axis_x, axis_y)
    n = domain.supported.shape[0]
    converted = [HB._array(v, (n,)) for v in (slo, shi, z, half, inner, outer, axis_x, axis_y, atol)]
    slo, shi, z, half, inner, outer, axis_x, axis_y, atol = [v[0] for v in converted]
    valid = (domain.supported & all(v[1] for v in converted) & jnp.isfinite(atol)
             & HB._normal_or_zero(atol) & (atol > 0))
    enabled = _enabled(inner)

    def box(lo, hi):
        return HB.enclose(x, u, p, q, B, lo, hi, z, half, inner, outer, axis_x, axis_y)

    start = _classify(box(slo, slo), enabled)
    valid = valid & start.bounds.supported
    uncertain = valid & jnp.any(start.surfaces, axis=1)
    status = jnp.where(valid, jnp.where(uncertain, START_UNRESOLVED, _ACTIVE), UNSUPPORTED)
    result = HC.ContactSearch(status, jnp.where(uncertain, slo, jnp.nan),
                              jnp.where(uncertain, slo, jnp.nan), start.surfaces & uncertain[:, None],
                              jnp.zeros((n, 4), bool), jnp.zeros(n, jnp.int32))
    shape = (n, max_depth + 1)
    left = jnp.zeros(shape, slo.dtype).at[:, 0].set(jnp.where(valid, slo, 0.))
    right = jnp.zeros(shape, slo.dtype).at[:, 0].set(jnp.where(valid, shi, 0.))
    state = HC._State(left, right, jnp.zeros(shape, jnp.int32), jnp.ones(n, jnp.int32), result)
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
        volume = _classify(box(lo, hi), enabled)
        supported = volume.bounds.supported
        excluded = (volume.region == INTERIOR) | (volume.region == EXTERIOR)
        mid = lo + (hi - lo) * .5
        narrow = hi - lo <= atol
        stalled = (mid <= lo) | (mid >= hi)
        exhausted = depth >= max_depth
        split = active & supported & ~excluded & ~narrow & ~stalled & ~exhausted
        pop = active & supported & excluded
        done = active & ~split & ~pop
        finished = pop & (st.size == 1)
        code = jnp.where(~supported, UNSUPPORTED,
                         jnp.where(narrow, CANDIDATE,
                                   jnp.where(stalled, PRECISION_UNRESOLVED, WORK_LIMIT_UNRESOLVED)))
        status = jnp.where(done, code, jnp.where(finished, NO_CANDIDATE, out.status))
        record = done & supported
        out = HC.ContactSearch(status, jnp.where(record, lo, out.slo), jnp.where(record, hi, out.shi),
                               jnp.where(record[:, None], volume.surfaces, out.surfaces), out.monotone,
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
        return iteration + 1, HC._State(left, right, depths, size, out)

    _, state = jax.lax.while_loop(condition, visit, (0, state))
    out = state.result
    waiting = out.status == _ACTIVE
    _, lo, hi, _ = top(state)
    out = out._replace(status=jnp.where(waiting, WORK_LIMIT_UNRESOLVED, out.status),
                       slo=jnp.where(waiting, lo, out.slo), shi=jnp.where(waiting, hi, out.shi))
    finite = jnp.isfinite(out.slo) & jnp.isfinite(out.shi)
    lo, hi = jnp.where(finite, out.slo, 0.), jnp.where(finite, out.shi, 0.)
    bounds, low, high = box(lo, hi), box(lo, lo), box(hi, hi)
    a, b = HB.evaluate(x, u, p, q, B, lo), HB.evaluate(x, u, p, q, B, hi)

    def surface_values(point):
        radius2 = (point.x[:, 0] - axis_x)**2 + (point.x[:, 1] - axis_y)**2
        return jnp.stack((point.x[:, 2] - z + half, z + half - point.x[:, 2],
                          outer**2 - radius2, radius2 - inner**2), axis=1)

    fa, fb = surface_values(a), surface_values(b)
    final = _classify(bounds, enabled)
    # Preserve deciding possibilities even if an independent final program differs by ulps.
    faces = (out.surfaces | final.surfaces) & finite[:, None]
    increasing, decreasing = bounds.df_lo > 0, bounds.df_hi < 0
    monotone = faces & (increasing | decreasing)
    rising = (low.f_hi < 0) & (high.f_lo > 0) & (fa < 0) & (fb > 0) & increasing
    falling = (low.f_lo > 0) & (high.f_hi < 0) & (fa > 0) & (fb < 0) & decreasing
    others_inside = jnp.all(~enabled | faces | (bounds.f_lo > 0), axis=1)
    unique = (jnp.sum(faces, axis=1) == 1) & others_inside
    entry = unique & jnp.any(faces & final.surfaces & rising, axis=1)
    exit = unique & jnp.any(faces & final.surfaces & falling, axis=1)
    status = jnp.where((out.status == CANDIDATE) & entry, ENTRY_BRACKET,
                       jnp.where((out.status == CANDIDATE) & exit, EXIT_BRACKET, out.status))
    checked = bounds.supported & low.supported & high.supported & a.supported & b.supported
    status = jnp.where(finite & ~checked, UNSUPPORTED, status)
    keep = finite & (status != UNSUPPORTED)
    initial_region = jnp.where(valid, start.region, REGION_UNSUPPORTED)
    domain_region = jnp.where(valid, _classify(domain, enabled).region, REGION_UNSUPPORTED)
    return VolumeSearch(status, jnp.where(keep, out.slo, jnp.nan), jnp.where(keep, out.shi, jnp.nan),
                        faces & keep[:, None], monotone & keep[:, None], out.work,
                        initial_region, domain_region)
