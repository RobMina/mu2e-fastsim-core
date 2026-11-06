"""Finite annular boundary queries for straight and axial-field charged legs.

This detector-independent development primitive is not a material navigator. It
uses exact plane/quadratic intersections, including the two radial phases of a
circular helix. Tilted fields are unsupported, never treated as clear paths.
Positions are mm, momentum MeV/c, magnetic field tesla; directions are unit length.
"""
from typing import NamedTuple

import jax.numpy as jnp

from mu2efastsim import eslab_jax as SJ


class Crossing(NamedTuple):
    """Distance is inf for clear, NaN for unsupported; surfaces: 0 none, 1/2 faces, 3 rim, 4 hole."""
    distance: object
    surface: object
    supported: object


def _quadratic(a, b, c):
    """Roots of a*x*x + 2*b*x + c, with a stable near root and linear limit."""
    disc = b * b - a * c
    real = disc >= 0
    root = jnp.sqrt(jnp.maximum(disc, 0.))
    q = -b - jnp.copysign(root, b)
    nonzero = q != 0
    first = q / jnp.where(a != 0, a, 1.)
    second = c / jnp.where(nonzero, q, 1.)
    repeated = -b / jnp.where(a != 0, a, 1.)
    first = jnp.where(nonzero, first, repeated)
    second = jnp.where(nonzero, second, repeated)
    linear = -c / jnp.where(b != 0, 2 * b, 1.)
    first = jnp.where(a != 0, first, linear)
    second = jnp.where(a != 0, second, jnp.inf)
    valid = real & ((a != 0) | (b != 0))
    return jnp.stack((jnp.where(valid, first, jnp.inf), jnp.where(valid, second, jnp.inf)), axis=1)


def safety(x, z, half, inner, outer, axis_x=0.):
    """Nonnegative distance to the nearest annular material boundary (inside starts)."""
    x = jnp.asarray(x)
    r = jnp.hypot(x[:, 0] - axis_x, x[:, 1])
    axial = half - jnp.abs(x[:, 2] - z)
    hole = jnp.where(jnp.asarray(inner) > 0, r - inner, jnp.inf)
    return jnp.maximum(jnp.minimum(axial, jnp.minimum(outer - r, hole)), 0.)


def crossing(x, u, p, q, B, smax, z, half, inner, outer, axis_x=0., *, entering=False):
    """Earliest material exit or vacuum entry in [0,smax], float64 lane arrays.

    x/u are (n,3); other quantities broadcast to (n,), except B (n,3) or None.
    Only Bx=By=0 is supported. Invalid inputs or an interior vacuum-entry start
    are unsupported. Boundary starts use one-sided derivatives, not nudges.
    A trajectory on a flat face/cylinder is in the closed material volume; isolated
    tangent touches are not transitions. At the starting boundary an outward
    tangent to the outer rim exits at zero, while an inner-hole tangent does not.
    No-hit is inf with surface 0. Unsupported/unresolved is NaN with surface 0.
    Callers must inspect supported before transport. Nearly repeated roots whose
    nonzero discriminant is within the roundoff bound are unresolved; exact
    repeated roots use the tangent convention. Nonzero curvature must be in
    [1e-150,1e100]/mm with proposed phase <=1e6 rad; other cases are unsupported.
    This API cannot silently supply a core callback for unrestricted fields.
    Finite path limits are required. This remains a development solver, not a
    certified arbitrary-magnitude geometry or field-map navigation interface.
    """
    x, u = jnp.asarray(x), jnp.asarray(u)
    if x.ndim != 2 or x.shape[1:] != (3,) or u.shape != x.shape:
        raise ValueError("x and u must have identical (n,3) shapes")
    if x.dtype != jnp.float64 or u.dtype != jnp.float64:
        raise ValueError("annular crossing requires float64 positions and directions")
    n = x.shape[0]
    p, q, smax, z, half, inner, outer, axis_x = (
        jnp.broadcast_to(jnp.asarray(v, x.dtype), (n,)) for v in (p, q, smax, z, half, inner, outer, axis_x))
    B = jnp.zeros_like(x) if B is None else jnp.broadcast_to(jnp.asarray(B, x.dtype), x.shape)
    valid = (jnp.all(jnp.isfinite(x) & jnp.isfinite(u) & jnp.isfinite(B), axis=1)
             & jnp.all(jnp.isfinite(jnp.stack((p, q, smax, z, half, inner, outer, axis_x))), axis=0)
             & (p > 0) & (smax >= 0) & (half > 0) & (inner >= 0) & (outer > inner)
             & (jnp.abs(jnp.sum(u * u, axis=1) - 1.) < 1e-12) & (B[:, 0] == 0) & (B[:, 1] == 0))
    xx, yy, zz = x[:, 0] - axis_x, x[:, 1], x[:, 2] - z
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    k = SJ.K_FIELD * q * B[:, 2] / jnp.where(p > 0, p, 1.)
    curved = k != 0
    ka = jnp.abs(k)
    valid = valid & (~curved | ((ka >= 1e-150) & (ka <= 1e100))) & (ka * smax <= 1e6)
    radial2 = xx * xx + yy * yy
    dot = xx * ux + yy * uy
    transverse2 = ux * ux + uy * uy
    cross = xx * uy - yy * ux
    za = (-half - zz) / jnp.where(uz != 0, uz, 1.)
    zb = (half - zz) / jnp.where(uz != 0, uz, 1.)
    axial_start = jnp.where(uz != 0, jnp.maximum(0., jnp.minimum(za, zb)), 0.)
    period = 2 * jnp.pi / jnp.where(curved, ka, 1.)

    def radial_roots(radius):
        c0 = (jnp.sqrt(radial2) - radius) * (jnp.sqrt(radial2) + radius)
        disc = transverse2 * radius * radius - cross * cross
        root = jnp.sqrt(jnp.maximum(disc, 0.))
        denom = jnp.where(transverse2 > 0, transverse2, 1.)
        far = -dot - jnp.copysign(root, dot)
        near = c0 / jnp.where(far != 0, far, 1.)
        first = far / denom
        near = jnp.where((far != 0) & (disc != 0), near, -dot / denom)
        straight = jnp.stack((first, near), axis=1)
        straight = jnp.where(((disc >= 0) & (transverse2 > 0))[:, None], straight, jnp.inf)
        exact_c0 = radial2 - radius * radius
        a = exact_c0 * k * k + 4 * transverse2 + 4 * k * cross
        b = 2 * ka * dot
        c = exact_c0 * k * k
        w = _quadratic(a, b, c)
        phase = jnp.mod(2 * jnp.arctan(w), 2 * jnp.pi)
        phase = jnp.where(jnp.isfinite(w), phase, jnp.inf)
        phase = phase.at[:, 1].set(jnp.where((a == 0) & (b != 0), jnp.pi, phase[:, 1]))
        roots = phase / jnp.where(curved, ka, 1.)[:, None]
        cycles = jnp.maximum(jnp.ceil((axial_start[:, None] - roots) / period[:, None]), 0.)
        roots = roots + cycles * period[:, None]
        curved_disc = b * b - a * c
        error = 64 * jnp.finfo(x.dtype).eps
        shallow_straight = (disc != 0) & (jnp.abs(disc) <= error * (transverse2 * radius * radius + cross * cross))
        cscale = (radial2 + radius * radius) * k * k
        ascale = cscale + 4 * transverse2 + 4 * jnp.abs(k * cross)
        discscale = b * b + jnp.abs(a * c) + jnp.abs(c) * ascale + jnp.abs(a) * cscale
        shallow_curved = (curved_disc != 0) & (jnp.abs(curved_disc) <= error * discscale)
        finite = jnp.isfinite(disc) & jnp.isfinite(curved_disc) & jnp.isfinite(a) & jnp.isfinite(c)
        resolved = finite & ~jnp.where(curved, shallow_curved, shallow_straight)
        return jnp.where(curved[:, None], roots, straight), resolved

    outer_roots, outer_ok = radial_roots(outer)
    inner_roots, inner_ok = radial_roots(inner)
    valid = valid & outer_ok & (inner_ok | (inner == 0))
    candidates = jnp.concatenate((jnp.zeros((n, 1), x.dtype),
                                  jnp.where((uz != 0)[:, None], jnp.stack((za, zb), axis=1), jnp.inf),
                                  outer_roots, jnp.where((inner > 0)[:, None], inner_roots, jnp.inf)), axis=1)
    usable = jnp.isfinite(candidates) & (candidates >= 0) & (candidates <= smax[:, None])
    s = jnp.where(usable, candidates, 0.)
    angle = k[:, None] * s
    sinc, cosc = SJ._sc(angle)
    px = xx[:, None] + s * (sinc * ux[:, None] + cosc * uy[:, None])
    py = yy[:, None] + s * (sinc * uy[:, None] - cosc * ux[:, None])
    pz = zz[:, None] + s * uz[:, None]
    dx = jnp.cos(angle) * ux[:, None] + jnp.sin(angle) * uy[:, None]
    dy = jnp.cos(angle) * uy[:, None] - jnp.sin(angle) * ux[:, None]
    r2 = px * px + py * py
    dr2 = 2 * (px * dx + py * dy)
    ddr2 = 2 * (transverse2[:, None] + k[:, None] * (px * dy - py * dx))
    f = jnp.stack((pz + half[:, None], half[:, None] - pz,
                   outer[:, None] ** 2 - r2, r2 - inner[:, None] ** 2), axis=2)
    df = jnp.stack((jnp.broadcast_to(uz[:, None], s.shape), jnp.broadcast_to(-uz[:, None], s.shape),
                    -dr2, dr2), axis=2)
    ddf = jnp.stack((jnp.zeros_like(s), jnp.zeros_like(s), -ddr2, ddr2), axis=2)
    tol = 128 * jnp.finfo(x.dtype).eps
    scale = jnp.stack((jnp.maximum(half, 1.), jnp.maximum(half, 1.),
                       jnp.maximum(outer * outer, 1.), jnp.maximum(inner * inner, 1.)), axis=1)
    coordinate_scale = jnp.abs(xx[:, None]) + jnp.abs(yy[:, None]) + jnp.abs(s) * jnp.sqrt(transverse2[:, None])
    radial_scale = (jnp.abs(px) + jnp.abs(py)) * coordinate_scale
    axial_scale = jnp.abs(zz[:, None]) + jnp.abs(s * uz[:, None])
    evalscale = jnp.maximum(scale[:, None, :], jnp.stack((axial_scale, axial_scale, radial_scale, radial_scale), axis=2))
    on = jnp.abs(f) <= tol * evalscale
    rootsurface = jnp.array([0, 1, 2, 3, 3, 4, 4])
    on = on | ((jnp.arange(1, 5)[None, None, :] == rootsurface[None, :, None]) & usable[:, :, None])
    dscale = jnp.stack((jnp.ones(n), jnp.ones(n), jnp.maximum(2 * outer, 1.),
                        jnp.maximum(2 * outer, 1.)), axis=1)
    moving = jnp.abs(df) > tol * dscale[:, None, :]
    moving = moving.at[:, :, :2].set(df[:, :, :2] != 0)
    ddscale = 2 * (transverse2[:, None] + ka[:, None] * (jnp.abs(px * dy) + jnp.abs(py * dx)))
    acceleration = jnp.where(jnp.abs(ddf) <= tol * jnp.maximum(ddscale[:, :, None], 1.), 0., ddf)

    def side(sign):
        near = jnp.where(moving, sign * df >= 0, acceleration >= 0)
        inside = jnp.where(on, near, f >= 0)
        inside = inside.at[:, :, 3].set(inside[:, :, 3] | (inner == 0)[:, None])
        return jnp.all(inside, axis=2)

    before, after = side(-1), side(1)
    closed = jnp.all((f[:, 0, :] >= -tol * scale).at[:, 3].set(
        (f[:, 0, 3] >= -tol * scale[:, 3]) | (inner == 0)), axis=1)
    active_on = on.at[:, :, 3].set(on[:, :, 3] & (inner > 0)[:, None])
    at_boundary = jnp.any(active_on[:, 0, :], axis=1)
    if entering:
        valid = valid & (~closed | at_boundary)
        transition = ~before & after
        transition = transition.at[:, 0].set(closed & at_boundary & after[:, 0])
    else:
        valid = valid & closed
        transition = before & ~after
        transition = transition.at[:, 0].set(closed & ~after[:, 0])
    paths = jnp.where(usable & transition, candidates, jnp.inf)
    selected = jnp.argmin(paths, axis=1)
    distance = jnp.min(paths, axis=1)
    hit = jnp.isfinite(distance)
    outward = jnp.where(moving, df < 0, acceleration < 0)
    inward = jnp.where(moving, df > 0, acceleration > 0)
    changing = active_on & (inward if entering else outward)
    changing = jnp.where(jnp.any(changing, axis=2, keepdims=True), changing, active_on)
    faces = jnp.argmax(changing, axis=2) + 1
    face = jnp.take_along_axis(faces, selected[:, None], axis=1)[:, 0]
    return Crossing(jnp.where(valid, distance, jnp.nan), jnp.where(valid & hit, face, 0), valid)
