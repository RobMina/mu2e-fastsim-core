"""Frozen NumPy straight-ray test oracle copied from beamfastsim.flash_exit.foil_exit."""
import numpy as np

INVALID, Z_MINUS, Z_PLUS, OUTER_RIM, INNER_HOLE = range(5)


def foil_exit(position, direction, foil_z, foil_ht, foil_rin, foil_rout, axis_x):
    """Return the first outward exit of straight rays born inside the foil.

    position and direction must have identical (..., 3) shapes; each foil
    parameter broadcasts to the leading shape. Nonzero directions are normalized,
    so distance is physical path length and returned direction is a unit vector.
    The closed birth volume is rin <= hypot(x-axis_x,y) <= rout and
    abs(z-foil_z) <= ht. Outside/nonfinite births, nonfinite/zero directions, or
    parameters not satisfying finite ht > 0 and 0 <= rin < rout are invalid.

    Exact surface births are accepted: outward motion exits at distance zero,
    inward motion continues through material. An outer-rim transverse tangent
    exits immediately; an inner-hole tangent merely touches and is not an exit.
    Motion confined to a flat face continues to another outward boundary.

    Returns a dict with distance (...), position/direction (...,3), valid (...),
    and integer face (...): 0 invalid, 1 lower-z, 2 upper-z, 3 outer rim, 4 hole.
    Exact corner ties prefer that face-code order. Invalid lanes have NaN numeric
    outputs and face 0. Incompatible array shapes raise ValueError. Calculations
    use float64; unrepresentable exit coordinates/distances are marked invalid.
    """
    p, d = np.asarray(position, dtype=float), np.asarray(direction, dtype=float)
    if p.ndim < 1 or p.shape[-1] != 3 or d.shape != p.shape:
        raise ValueError("position and direction must have identical (..., 3) shapes")
    z, ht, rin, rout, ax = [np.broadcast_to(np.asarray(a, dtype=float), p.shape[:-1])
                           for a in (foil_z, foil_ht, foil_rin, foil_rout, axis_x)]
    # Mask invalid lanes and normalize directions by scale before computing exits.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        scale = np.max(np.abs(d), axis=-1)
        valid = (np.all(np.isfinite(p), axis=-1) & np.all(np.isfinite(d), axis=-1)
                 & (scale > 0) & np.isfinite(z) & np.isfinite(ht) & (ht > 0)
                 & np.isfinite(rin) & np.isfinite(rout) & np.isfinite(ax)
                 & (rin >= 0) & (rout > rin))
        x, y, zz = p[..., 0] - ax, p[..., 1], p[..., 2] - z
        radius = np.hypot(x, y)
        valid &= (np.isfinite(radius) & np.isfinite(zz) & (radius >= rin)
                  & (radius <= rout) & (np.abs(zz) <= ht))
        u = np.where(valid[..., None], d, 0.) / np.where(valid, scale, 1.)[..., None]
        norm = np.sqrt(np.sum(u * u, axis=-1))
        u /= np.where(valid, norm, 1.)[..., None]
        x, y, zz, radius = [np.where(valid, a, 0.) for a in (x, y, zz, radius)]
        ht, rin, rout = [np.where(valid, a, 1.) for a in (ht, rin, rout)]
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        lower = np.where(uz < 0, (ht + zz) / -uz, np.inf)
        upper = np.where(uz > 0, (ht - zz) / uz, np.inf)

        transverse = np.hypot(ux, uy)
        vx = np.divide(ux, transverse, out=np.zeros_like(ux), where=transverse > 0)
        vy = np.divide(uy, transverse, out=np.zeros_like(uy), where=transverse > 0)
        # Work in outer-radius units to avoid squaring dimensional coordinates.
        xr, yr, rr, ri = x / rout, y / rout, radius / rout, rin / rout
        b = xr * vx + yr * vy
        cross = np.abs(xr * vy - yr * vx)
        root = np.sqrt(np.maximum((1. - cross) * (1. + cross), 0.))
        # Rationalized roots avoid cancellation for births close to a surface.
        outer_path = np.where(b >= 0, (rout - radius) * (1. + rr) / (root + b),
                              rout * (root - b))
        outer_path = np.where((radius == rout) & (b >= 0), 0., outer_path)
        outer = np.where(transverse > 0, outer_path / transverse, np.inf)
        disc = (ri - cross) * (ri + cross)
        hole_path = (radius - rin) * (rr + ri) / (-b + np.sqrt(np.maximum(disc, 0.)))
        hole = np.where((rin > 0) & (b < 0) & (disc > 0) & (transverse > 0),
                        hole_path / transverse, np.inf)
        candidates = np.stack((lower, upper, outer, hole), axis=-1)
        candidates = np.where(candidates >= 0, candidates, np.inf)
        index = np.argmin(candidates, axis=-1)
        distance = np.take_along_axis(candidates, index[..., None], axis=-1)[..., 0]
        endpoint = p + distance[..., None] * u
        valid &= np.isfinite(distance) & np.all(np.isfinite(endpoint), axis=-1)
    return {"distance": np.where(valid, distance, np.nan),
            "position": np.where(valid[..., None], endpoint, np.nan),
            "direction": np.where(valid[..., None], u, np.nan),
            "valid": valid, "face": np.where(valid, index + 1, INVALID)}
