"""Charged-particle stepping in a magnetic field for the electron transport (NumPy).

A field is either a constant 3-vector [T] or a callable B(x) -> (n, 3) [T] of positions (n, 3) [mm], e.g. a wrapped
field-map lookup. In a step of path length s the track follows the exact helix of the field at the step start
(`helix`), which is exact in a uniform field; `rk4` is the classical fourth-order step for a field that varies along
the step. The direction obeys du/ds = K q / p (u x B), K = 0.299792458 (mm, MeV/c, T); q in units of e.
`to_plane` gives the path length along the helix to a plane z = const. Lengths in mm, momenta in MeV/c.
"""
import numpy as np

K = 0.299792458        # MeV/c per (T mm)
_SMALL = 1e-6          # |k s| below which the series forms are used


def field_at(field, x):
    """B [T] at positions x (n, 3) for a constant vector or a callable field."""
    x = np.asarray(x, float)
    if callable(field):
        return np.asarray(field(x), float).reshape(x.shape)
    return np.broadcast_to(np.asarray(field, float), x.shape)


def _frame(u, B):
    """(b_hat, |B|, u_par along b, u_perp, w = u_perp x b_hat) per row; b_hat = z where B = 0."""
    bm = np.linalg.norm(B, axis=1)
    bh = np.where(bm[:, None] > 0, B / np.where(bm > 0, bm, 1.0)[:, None], np.array([0.0, 0.0, 1.0]))
    upar = (u * bh).sum(axis=1)
    uperp = u - upar[:, None] * bh
    return bh, bm, upar, uperp, np.cross(uperp, bh)


def _sc(ks):
    """(sin(ks)/ks, (1 - cos(ks))/ks) with series near 0."""
    small = np.abs(ks) < _SMALL
    k2 = ks * ks
    ksn = np.where(small, 1.0, ks)
    s1 = np.where(small, 1 - k2 / 6, np.sin(ksn) / ksn)
    c1 = np.where(small, ks / 2 - ks * k2 / 24, (1 - np.cos(ksn)) / ksn)
    return s1, c1


def curvature(p, q, bm):
    """Signed k = K q |B| / p [1/mm]: the direction turns by k s about b_hat."""
    return K * q * bm / np.asarray(p, float)


def helix(x, u, p, q, B, s):
    """Exact helix over path s in the uniform field B (n, 3): new positions and unit directions."""
    x, u = np.asarray(x, float), np.asarray(u, float)
    s = np.broadcast_to(np.asarray(s, float), (x.shape[0],))
    bh, bm, upar, uperp, w = _frame(u, np.asarray(B, float))
    k = curvature(p, q, bm)
    ks = k * s
    s1, c1 = _sc(ks)
    xn = x + (upar * s)[:, None] * bh + (s * s1)[:, None] * uperp + (s * c1)[:, None] * w
    un = upar[:, None] * bh + np.cos(ks)[:, None] * uperp + np.sin(ks)[:, None] * w
    return xn, un / np.linalg.norm(un, axis=1, keepdims=True)


def rk4(x, u, p, q, field, h):
    """One classical RK4 step of length h along (x, u) in a position-dependent field."""
    x, u = np.asarray(x, float), np.asarray(u, float)
    h = np.broadcast_to(np.asarray(h, float), (x.shape[0],))[:, None]
    c = (K * q / np.asarray(p, float))
    c = np.broadcast_to(c, (x.shape[0],))[:, None]

    def du(xx, uu):
        return c * np.cross(uu, field_at(field, xx))

    k1x, k1u = u, du(x, u)
    k2x, k2u = u + 0.5 * h * k1u, du(x + 0.5 * h * k1x, u + 0.5 * h * k1u)
    k3x, k3u = u + 0.5 * h * k2u, du(x + 0.5 * h * k2x, u + 0.5 * h * k2u)
    k4x, k4u = u + h * k3u, du(x + h * k3x, u + h * k3u)
    xn = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    un = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
    return xn, un / np.linalg.norm(un, axis=1, keepdims=True)


def to_plane(x, u, p, q, B, z_plane, s_max):
    """Smallest path length in (0, s_max] at which the helix through (x, u) in the uniform field B reaches z = z_plane
    (np.inf where it does not): scan in sub-arcs of at most 0.5 rad, then bisection and Newton on z(s)."""
    x, u = np.asarray(x, float), np.asarray(u, float)
    n = x.shape[0]
    zp = np.broadcast_to(np.asarray(z_plane, float), (n,))
    smax = np.broadcast_to(np.asarray(s_max, float), (n,)).copy()
    bh, bm, upar, uperp, w = _frame(u, np.asarray(B, float))
    k = curvature(p, q, bm)

    def z_of(s):
        s1, c1 = _sc(k * s)
        return x[:, 2] + upar * s * bh[:, 2] + s * s1 * uperp[:, 2] + s * c1 * w[:, 2] - zp

    def dz_of(s):
        return upar * bh[:, 2] + np.cos(k * s) * uperp[:, 2] + np.sin(k * s) * w[:, 2]

    nseg = np.clip(np.ceil(np.abs(k) * np.where(np.isfinite(smax), smax, 0.0) / 0.5), 1, 4096).astype(int)
    lo = np.full(n, 1e-12)            # a track on the plane at s = 0 does not count as reaching it
    f0 = z_of(lo)
    hi = np.full(n, np.nan)
    found = np.zeros(n, bool)
    for j in range(1, int(nseg.max()) + 1):
        on = (~found) & (j <= nseg)
        if not on.any():
            break
        sj = smax * j / nseg
        fj = z_of(sj)
        cross = on & ((np.sign(fj) != np.sign(f0)) | (fj == 0)) & (f0 != 0)
        hi = np.where(cross, sj, hi)
        lo = np.where(on & ~cross, sj, lo)
        f0 = np.where(on & ~cross, fj, f0)
        found |= cross
    hi = np.where(found, hi, lo)
    flo = z_of(lo)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fm = z_of(mid)
        same = np.sign(fm) == np.sign(flo)
        lo, flo = np.where(same, mid, lo), np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    s = 0.5 * (lo + hi)
    for _ in range(2):
        d = dz_of(s)
        s = np.where(np.abs(d) > 1e-12, s - z_of(s) / np.where(np.abs(d) > 1e-12, d, 1.0), s)
    s = np.clip(s, 0.0, smax)
    return np.where(found, s, np.inf)


def read_g4bl(header_path):
    """(lo, d, n, B[nx, ny, nz, 3]) of a G4BL-format Mu2e field map (.header + .bin: 0xDEADBEEF marker, then
    (Bx, By, Bz) doubles in T at index ix*ny*nz + iy*nz + iz), as Offline BFieldManagerMaker::readG4BLBinary."""
    import re
    text = open(header_path).read()
    g = dict((k, float(v)) for k, v in re.findall(r"(\w+)=([-\d.eE+]+)", text.split("grid", 1)[1].split("\n")[0]))
    lo = np.array([g["X0"], g["Y0"], g["Z0"]])
    d = np.array([g["dX"], g["dY"], g["dZ"]])
    n = np.array([int(g["nX"]), int(g["nY"]), int(g["nZ"])])
    binf = header_path[: -len(".header")] + ".bin"
    if np.fromfile(binf, dtype=np.uint32, count=1)[0] != 0xDEADBEEF:
        raise ValueError(f"{binf}: bad endian marker")
    return lo, d, n, np.fromfile(binf, dtype=np.float64, offset=4).reshape(n[0], n[1], n[2], 3)


def grid_field(lo, d, n, Bgrid):
    """Callable B(x) [T]: trilinear interpolation on a regular grid (Offline BFGridMap::interpolateTriLinear),
    zero outside. The arguments are those of a G4BL map (read_g4bl, or the beam package's FieldMap fields)."""
    lo, d, n, Bgrid = (np.asarray(a) for a in (lo, d, n, Bgrid))

    def field(x):
        uu = (np.asarray(x, float) - lo) / d
        i = np.floor(uu).astype(int)
        inside = np.all((i >= 0) & (i < n - 1), axis=1)
        i = np.clip(i, 0, n - 2)
        f = uu - i
        out = np.zeros(uu.shape)
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    out += (wx * wy * wz)[:, None] * Bgrid[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return np.where(inside[:, None], out, 0.0)
    return field


def propagate(x, u, p, q, field, h, z_stop, s_max, method="helix"):
    """Vacuum transport in steps of h [mm] until z reaches z_stop (the last step ends on the plane) or the path
    reaches s_max: (positions, directions, path lengths, reached). method "helix" (each step on the helix of the
    field at its midpoint) or "rk4"; the final partial step to the plane is a helix."""
    x, u = np.array(x, float), np.array(u, float)
    n = x.shape[0]
    p = np.broadcast_to(np.asarray(p, float), (n,))
    s = np.zeros(n)
    reached = np.zeros(n, bool)
    live = np.arange(n)
    while live.size:
        hl = np.minimum(h, s_max - s[live])
        B = field_at(field, x[live])
        if callable(field):           # field at the helix midpoint of the step: second order in the field gradient
            B = field_at(field, helix(x[live], u[live], p[live], q, B, 0.5 * hl)[0])
        sp = to_plane(x[live], u[live], p[live], q, B, z_stop, hl)
        hit = np.isfinite(sp)
        step = np.where(hit, sp, hl)
        if method == "rk4":
            xr, ur = rk4(x[live], u[live], p[live], q, field, step)
            xh, uh = helix(x[live], u[live], p[live], q, B, step)
            x[live] = np.where(hit[:, None], xh, xr)
            u[live] = np.where(hit[:, None], uh, ur)
        else:
            x[live], u[live] = helix(x[live], u[live], p[live], q, B, step)
        s[live] += step
        reached[live[hit]] = True
        live = live[~hit & (s[live] < s_max - 1e-12)]
    return x, u, s, reached


def map_check(header, start, z_stop, n=2000, p=105.5, q=-1.0, h=10.0, h_ref=1.0, cos_range=(0.4, 0.8), seed=1):
    """Self-consistency in a real field map: isotropic tracks in a forward cone from `start` [mm] propagated to the
    plane z = z_stop with helix steps of h and with RK4 steps of h_ref; returns the position and path differences."""
    import time
    f = grid_field(*read_g4bl(header))
    rng = np.random.default_rng(seed)
    ct = rng.uniform(*cos_range, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u0 = np.stack([np.sqrt(1 - ct ** 2) * np.cos(phi), np.sqrt(1 - ct ** 2) * np.sin(phi), ct], axis=1)
    x0 = np.tile(np.asarray(start, float), (n, 1))
    smax = 4 * abs(z_stop - start[2]) / cos_range[0]
    t0 = time.time()
    xh, uh, sh, rh = propagate(x0, u0, p, q, f, h, z_stop, smax)
    t1 = time.time()
    xr, ur, sr, rr = propagate(x0, u0, p, q, f, h_ref, z_stop, smax, method="rk4")
    t2 = time.time()
    both = rh & rr
    dr = np.linalg.norm((xh - xr)[both], axis=1)
    rad = np.hypot(*(xh[both, :2] - np.asarray(start[:2], float)).T)
    return dict(n=n, reached=[int(rh.sum()), int(rr.sum())], dr_mm=dict(median=float(np.median(dr)),
                q99=float(np.quantile(dr, 0.99)), max=float(dr.max())),
                ds_mm_max=float(np.abs(sh - sr)[both].max()), path_mm_median=float(np.median(sh[both])),
                exit_radius_mm=dict(median=float(np.median(rad)), max=float(rad.max())),
                seconds=[t1 - t0, t2 - t1], steps=[h, h_ref])


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.ehelix", description="field-map stepping check")
    ap.add_argument("header")
    ap.add_argument("--start", type=float, nargs=3, required=True)
    ap.add_argument("--z-stop", type=float, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--ref-step", type=float, default=1.0)
    a = ap.parse_args()
    print(json.dumps(map_check(a.header, a.start, a.z_stop, a.n, h=a.step, h_ref=a.ref_step), indent=1))
