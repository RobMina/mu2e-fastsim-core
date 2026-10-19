"""Passive world material for the muon transport, from the Geant4 GDML export (stage B/C).

The material a cosmic crosses above the extracted CRV is a few hundred convex-or-simple prisms placed in HallAir and
the world: `box` and `xtru` (extruded polygons, mostly rotated by (90, 0, 90)) of concrete and overburden, plus the
worldDirt boxes. This module flattens them into global-frame convex pieces and ray-traces straight segments through
them, returning the material column a track crosses (per material, in mm).

Approximations, by design and documented: subtraction/union solids are taken as their first constituent's convex hull
pieces (2 subtraction slabs above the CRV), tubes/polycones/tori (pipes, cryostats) are skipped (they lie below the
CRV), and overlapping daughters are not subtracted from their mother (mothers above the CRV are air: HallAir, World).
"""
import re
import xml.etree.ElementTree as ET

import numpy as np

_STRIP = re.compile(r"0x[0-9a-f]+$")


def _strip(n):
    return _STRIP.sub("", n)


def _rot(r):
    """GDML rotation element -> 3x3 matrix applied to daughter-frame points (Geant4 placement uses the inverse)."""
    if r is None:
        return np.eye(3)
    scale = np.pi / 180 if r.get("unit", "deg") == "deg" else 1.0
    ax, ay, az = (float(r.get(a, 0)) * scale for a in "xyz")
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


def _convex_pieces_2d(v):
    """Split a simple polygon (vertex list) into convex pieces by ear clipping into triangles."""
    v = np.asarray(v, float)
    if len(v) < 3:
        return []
    area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
    if area < 0:
        v = v[::-1]
    idx = list(range(len(v)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            a, b, c = v[idx[k - 1]], v[idx[k]], v[idx[(k + 1) % len(idx)]]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            others = [v[j] for j in idx if j not in (idx[k - 1], idx[k], idx[(k + 1) % len(idx)])]
            if any(_in_tri(p, a, b, c) for p in others):
                continue
            tris.append(np.array([a, b, c]))
            idx.pop(k)
            break
    if len(idx) == 3:
        tris.append(v[idx])
    return tris


def _in_tri(p, a, b, c):
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def load(gdml_path, keep_materials=None, y_min=None):
    """Global-frame convex pieces: list of (material name, planes N x 4 [nx, ny, nz, d]).

    keep_materials: names to keep (default: everything that is not air or vacuum). y_min: drop pieces whose global
    bounding box lies entirely below this height (mm).
    """
    root = ET.parse(gdml_path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    world = root.find("setup/world").get("ref")
    pieces = []

    def mat_of(lv):
        m = vols[lv].find("materialref")
        return _strip(m.get("ref")) if m is not None else ""

    def emit(solid, R, t, mat):
        tag = solid.tag
        if tag in ("subtraction", "union", "intersection"):
            emit(solids[solid.find("first").get("ref")], R, t, mat)
            return
        if tag == "box":
            h = 0.5 * np.array([float(solid.get(a)) for a in "xyz"])
            tri_sets = [np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]])]
            z0, z1 = -h[2], h[2]
        elif tag == "xtru":
            sec = solid.findall("section")
            if len(sec) != 2 or any(float(s.get("scalingFactor", 1)) != 1 for s in sec):
                return
            v = [[float(p.get("x")), float(p.get("y"))] for p in solid.findall("twoDimVertex")]
            tri_sets = _convex_pieces_2d(v)
            z0, z1 = float(sec[0].get("zPosition")), float(sec[1].get("zPosition"))
        else:
            return
        for poly in tri_sets:
            planes = []
            n_edges = len(poly)
            area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
            pp = poly if area > 0 else poly[::-1]
            for i in range(n_edges):
                a, b = pp[i], pp[(i + 1) % n_edges]
                n = np.array([b[1] - a[1], -(b[0] - a[0]), 0.0])
                n /= np.linalg.norm(n)
                planes.append((n, n @ np.array([a[0], a[1], 0.0])))
            planes += [(np.array([0, 0, 1.0]), z1), (np.array([0, 0, -1.0]), -z0)]
            P = np.array([[*(R @ n), (R @ n) @ t + d] for n, d in planes])
            corners = np.array([[x, y, z] for x, y in pp for z in (z0, z1)]) @ R.T + t
            if y_min is not None and corners[:, 1].max() < y_min:
                continue
            pieces.append((mat, P, corners.min(0), corners.max(0)))

    def walk(lv, R, t):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            p = pv.find("position")
            pos = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            Rl = _rot(pv.find("rotation"))
            R2, t2 = R @ Rl, R @ pos + t
            mat = mat_of(child)
            if (keep_materials is None and not re.search(r"AIR|Vacuum|VACUUM|Galactic", mat)) or \
               (keep_materials is not None and mat in keep_materials):
                emit(solids[vols[child].find("solidref").get("ref")], R2, t2, mat)
            walk(child, R2, t2)

    walk(world, np.eye(3), np.zeros(3))
    return pieces


def column(pieces, o, d, t_max, chunk=4096):
    """Path length [mm] of each segment o + s d, s in [0, t_max], inside each material: dict name -> [N] array.

    Convex-piece slab clipping: for each piece, entry = max over planes of the entering intersections, exit = min over
    the leaving ones. Pieces are grouped by material; overlaps within a material are summed (geometry overlaps are rare
    and small above the CRV).
    """
    o = np.asarray(o, float)
    d = np.asarray(d, float)
    t_max = np.asarray(t_max, float) * np.ones(len(o))
    out = {}
    by_mat = {}
    for mat, P, lo, hi in pieces:
        by_mat.setdefault(mat, []).append((P, lo, hi))
    for mat, plist in by_mat.items():
        acc = np.zeros(len(o))
        for P, lo, hi in plist:
            # cheap bounding-box reject over all segments at once
            inv = 1.0 / np.where(d == 0, 1e-300, d)
            t1 = (lo - o) * inv
            t2 = (hi - o) * inv
            tin = np.maximum(np.max(np.minimum(t1, t2), 1), 0.0)
            tout = np.minimum(np.min(np.maximum(t1, t2), 1), t_max)
            cand = np.flatnonzero(tout > tin)
            if cand.size == 0:
                continue
            n, dd = P[:, :3], P[:, 3]
            den = d[cand] @ n.T                     # [M, planes]
            num = dd[None, :] - o[cand] @ n.T       # n.(o + s d) <= dd  <=>  s den <= num
            with np.errstate(divide="ignore", invalid="ignore"):
                s = num / den
            enter = np.where(den < 0, s, -np.inf).max(1)
            leave = np.where(den > 0, s, np.inf).min(1)
            outside = ((den == 0) & (num < 0)).any(1)
            a = np.maximum(enter, 0.0)
            b = np.minimum(leave, t_max[cand])
            L = np.where(outside, 0.0, np.clip(b - a, 0.0, None))
            acc[cand] += L
        out[mat] = acc
    return out


def load_tubes(gdml_path, keep_materials=None, bbox=None):
    """Tube and polycone solids of the GDML world as cylindrical shells: list of (material, R [3x3], t [3], rmin, rmax,
    half z, phi0, dphi) in the global frame (R maps local to global). Polycones become one shell per z section with the
    section's smaller rmin and larger rmax. Boolean solids use their first constituent (as `load`).

    keep_materials: names to keep (default: everything that is not air or vacuum). bbox: (lo, hi) global corners; keep
    only shells whose global bounding box overlaps it (the tracker region holds thousands of support pieces).
    """
    root = ET.parse(gdml_path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    world = root.find("setup/world").get("ref")
    out = []

    def keep(mat):
        if keep_materials is not None:
            return mat in keep_materials
        return not re.search(r"AIR|Vacuum|VACUUM|Galactic", mat)

    def shells(solid):
        while solid.tag in ("subtraction", "union", "intersection"):
            solid = solids[solid.find("first").get("ref")]
        ang = np.pi / 180 if solid.get("aunit", "deg") == "deg" else 1.0
        phi0 = float(solid.get("startphi", 0)) * ang
        dphi = float(solid.get("deltaphi", 360)) * ang
        if solid.tag in ("tube", "cutTube"):
            return [(float(solid.get("rmin", 0)), float(solid.get("rmax")), 0.0, 0.5 * float(solid.get("z")), phi0, dphi)]
        if solid.tag == "polycone":
            zp = sorted((float(z.get("z")), float(z.get("rmin", 0)), float(z.get("rmax"))) for z in solid.findall("zplane"))
            res = []
            for (z0, a0, b0), (z1, a1, b1) in zip(zp[:-1], zp[1:]):
                if z1 > z0:
                    res.append((min(a0, a1), max(b0, b1), 0.5 * (z0 + z1), 0.5 * (z1 - z0), phi0, dphi))
            return res
        return []

    def walk(lv, R, t):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            p = pv.find("position")
            pos = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            Rl = _rot(pv.find("rotation"))
            R2, t2 = R @ Rl, R @ pos + t
            mat = _strip(vols[child].find("materialref").get("ref"))
            if keep(mat):
                for rmin, rmax, zc, hz, phi0, dphi in shells(solids[vols[child].find("solidref").get("ref")]):
                    tc = t2 + R2 @ np.array([0.0, 0.0, zc])
                    corners = np.array([[sx * rmax, sy * rmax, sz * hz] for sx in (-1, 1) for sy in (-1, 1)
                                        for sz in (-1, 1)]) @ R2.T + tc
                    lo, hi = corners.min(0), corners.max(0)
                    if bbox is not None and (np.any(hi < bbox[0]) or np.any(lo > bbox[1])):
                        continue
                    out.append((mat, R2, tc, rmin, rmax, hz, phi0, dphi))
            walk(child, R2, t2)

    walk(world, np.eye(3), np.zeros(3))
    return out


def column_tubes(tubes, o, d, t_max, nsub=16):
    """Path length [mm] of each segment o + s d, s in [0, t_max], inside the cylindrical shells: dict material -> [N].

    Per shell, in its local frame: the chord inside r <= rmax minus the chord inside r < rmin, clipped to |z| <= hz;
    the phi range of a sector is applied by subdividing each inside interval into `nsub` pieces and keeping those whose
    midpoint lies in [phi0, phi0 + dphi] (exact for full tubes; nsub sets the accuracy at sector edges).
    """
    o = np.asarray(o, float)
    d = np.asarray(d, float)
    t_max = np.asarray(t_max, float) * np.ones(len(o))
    out = {}
    for mat, R, tc, rmin, rmax, hz, phi0, dphi in tubes:
        ol = (o - tc) @ R          # global -> local: R^T (x - t), written as row vectors
        dl = d @ R
        A = dl[:, 0] ** 2 + dl[:, 1] ** 2
        B = 2 * (ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1])
        r2 = ol[:, 0] ** 2 + ol[:, 1] ** 2

        def chord(r):
            C = r2 - r * r
            disc = B * B - 4 * A * C
            par = A < 1e-15
            ok = np.where(par, C <= 0, disc > 0)
            sq = np.sqrt(np.where(ok & ~par, disc, 0.0))
            a_ = np.where(par, 1.0, A)
            return (np.where(par, -np.inf, (-B - sq) / (2 * a_)), np.where(par, np.inf, (-B + sq) / (2 * a_)), ok)

        with np.errstate(divide="ignore", invalid="ignore"):
            z0 = np.where(np.abs(dl[:, 2]) > 1e-15, (-hz - ol[:, 2]) / dl[:, 2], -np.inf)
            z1 = np.where(np.abs(dl[:, 2]) > 1e-15, (hz - ol[:, 2]) / dl[:, 2], np.inf)
        zin = (np.abs(dl[:, 2]) > 1e-15) | (np.abs(ol[:, 2]) <= hz)
        zlo, zhi = np.minimum(z0, z1), np.maximum(z0, z1)
        a0, a1, aok = chord(rmax)
        lo = np.maximum(np.maximum(a0, zlo), 0.0)
        hi = np.minimum(np.minimum(a1, zhi), t_max)
        cand = np.flatnonzero(aok & zin & (hi > lo))
        if not len(cand):
            continue
        full = dphi >= 2 * np.pi - 1e-9
        # inside the shell: [lo, hi] minus the rmin hole [b0, b1]; split into up to two intervals
        ivals = []
        if rmin > 0:
            b0, b1, bok = chord(rmin)
            hole = bok[cand]
            b0c = np.where(hole, b0[cand], hi[cand])         # no hole crossing: the first interval is all of [lo, hi]
            b1c = np.where(hole, b1[cand], hi[cand])
            ivals.append((lo[cand], np.minimum(hi[cand], np.maximum(b0c, lo[cand]))))
            ivals.append((np.maximum(lo[cand], np.minimum(b1c, hi[cand])), hi[cand]))
        else:
            ivals.append((lo[cand], hi[cand]))
        acc = np.zeros(len(cand))
        for s0, s1 in ivals:
            ln = np.clip(s1 - s0, 0, None)
            if full:
                acc += ln
                continue
            k = (np.arange(nsub) + 0.5) / nsub
            sm = s0[:, None] + ln[:, None] * k[None, :]
            x = ol[cand, 0][:, None] + dl[cand, 0][:, None] * sm
            y = ol[cand, 1][:, None] + dl[cand, 1][:, None] * sm
            ph = np.mod(np.arctan2(y, x) - phi0, 2 * np.pi)
            acc += ln * np.mean(ph <= dphi, axis=1)
        v = out.setdefault(mat, np.zeros(len(o)))
        v[cand] += acc
    return out
