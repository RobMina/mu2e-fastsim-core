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


def _pieces_of(solid, solids, R, t, mat, y_min=None, clip=None):
    """Global-frame convex pieces (mat, planes, lo, hi) of a box or xtru solid (booleans: first constituent).

    clip: optional extra half-space planes [K, 4] (n . x <= d, global frame) added to every piece (lo/hi unchanged)."""
    out = []
    tag = solid.tag
    if tag in ("subtraction", "union", "intersection"):
        return _pieces_of(solids[solid.find("first").get("ref")], solids, R, t, mat, y_min, clip)
    if tag == "box":
        h = 0.5 * np.array([float(solid.get(a)) for a in "xyz"])
        tri_sets = [np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]])]
        z0, z1 = -h[2], h[2]
    elif tag == "xtru":
        sec = solid.findall("section")
        if len(sec) != 2 or any(float(s.get("scalingFactor", 1)) != 1 for s in sec):
            return out
        v = [[float(p.get("x")), float(p.get("y"))] for p in solid.findall("twoDimVertex")]
        tri_sets = _convex_pieces_2d(v)
        z0, z1 = float(sec[0].get("zPosition")), float(sec[1].get("zPosition"))
    else:
        return out
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
        if clip is not None:
            P = np.vstack([P, np.asarray(clip, float).reshape(-1, 4)])
        out.append((mat, P, corners.min(0), corners.max(0)))
    return out


def load(gdml_path, keep_materials=None, y_min=None, names=None):
    """Global-frame convex pieces: list of (material name, planes N x 4 [nx, ny, nz, d]).

    keep_materials: names to keep (default: everything that is not air or vacuum). y_min: drop pieces whose global
    bounding box lies entirely below this height (mm). names: optional list; the placed volume's name (pointer suffix
    stripped) of each returned piece is appended (overrides and diagnostics).
    """
    root = ET.parse(gdml_path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    world = root.find("setup/world").get("ref")
    pieces = []

    def mat_of(lv):
        m = vols[lv].find("materialref")
        return _strip(m.get("ref")) if m is not None else ""

    def emit(solid, R, t, mat, vname=""):
        new = _pieces_of(solid, solids, R, t, mat, y_min)
        pieces.extend(new)
        if names is not None:
            names.extend([vname] * len(new))

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
                emit(solids[vols[child].find("solidref").get("ref")], R2, t2, mat, _strip(child))
            walk(child, R2, t2)

    walk(world, np.eye(3), np.zeros(3))
    return pieces


class PieceSet:
    """The convex pieces of `load` packed into arrays for vectorised columns (same result as `column`).

    Bounding boxes [P, 3] and planes padded to the largest plane count [P, Kmax, 4] (padding planes never cut).
    `column` tests every (track, piece) pair's bounding box in one array operation per chunk of tracks, then clips
    only the surviving pairs against their planes, so there is no Python loop over pieces.
    """

    def __init__(self, pieces):
        self.mats = sorted({q[0] for q in pieces})
        mi = {m: i for i, m in enumerate(self.mats)}
        self.mat = np.array([mi[q[0]] for q in pieces], np.int64)
        self.lo = np.array([q[2] for q in pieces], float).reshape(-1, 3)
        self.hi = np.array([q[3] for q in pieces], float).reshape(-1, 3)
        kmax = max((len(q[1]) for q in pieces), default=1)
        P = np.zeros((len(pieces), kmax, 4))
        P[:, :, 3] = 1e30                       # padding plane 0 . x <= 1e30: always inside, never enters or leaves
        for i, q in enumerate(pieces):
            P[i, :len(q[1])] = q[1]
        self.P = P

    def __len__(self):
        return len(self.mat)

    def build_grid(self, cell=500.0):
        """Uniform x-z grid of `cell` mm over the pieces' footprint: for each cell, the pieces whose box overlaps it.

        column then tests only the pieces in the cells a segment's x-z bounding box touches (pieces are mostly
        compact slabs and prisms, and segments in a height band are short in x-z), instead of every piece."""
        if not len(self.mat):
            self.grid = None
            return self
        self.g0 = np.array([self.lo[:, 0].min(), self.lo[:, 2].min()])
        g1 = np.array([self.hi[:, 0].max(), self.hi[:, 2].max()])
        self.cell = float(cell)
        self.ng = np.maximum(np.ceil((g1 - self.g0) / cell).astype(int), 1)
        ix0 = np.clip(((self.lo[:, 0] - self.g0[0]) // cell).astype(int), 0, self.ng[0] - 1)
        ix1 = np.clip(((self.hi[:, 0] - self.g0[0]) // cell).astype(int), 0, self.ng[0] - 1)
        iz0 = np.clip(((self.lo[:, 2] - self.g0[1]) // cell).astype(int), 0, self.ng[1] - 1)
        iz1 = np.clip(((self.hi[:, 2] - self.g0[1]) // cell).astype(int), 0, self.ng[1] - 1)
        nx, nz = ix1 - ix0 + 1, iz1 - iz0 + 1
        cnt = nx * nz
        ip = np.repeat(np.arange(len(self.mat)), cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cid = (ix0[ip] + k // nz[ip]) * self.ng[1] + iz0[ip] + k % nz[ip]
        order = np.lexsort((ip, cid))                       # by cell, pieces ascending within a cell
        self.cell_items = ip[order].astype(np.int64)
        self.cell_start = np.r_[0, np.cumsum(np.bincount(cid, minlength=self.ng[0] * self.ng[1]))]
        self.grid = True
        return self

    def _candidates(self, o, d, t_max):
        """(track, piece) candidate pairs from the grid: pieces in the cells of each segment's x-z bounding box."""
        a, b = o, o + d * t_max[:, None]
        x0, x1 = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
        z0, z1 = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
        c = self.cell
        ix0 = np.clip(((x0 - self.g0[0]) // c).astype(int), 0, self.ng[0] - 1)
        ix1 = np.clip(((x1 - self.g0[0]) // c).astype(int), 0, self.ng[0] - 1)
        iz0 = np.clip(((z0 - self.g0[1]) // c).astype(int), 0, self.ng[1] - 1)
        iz1 = np.clip(((z1 - self.g0[1]) // c).astype(int), 0, self.ng[1] - 1)
        outside = (x1 < self.g0[0]) | (z1 < self.g0[1]) | (x0 > self.g0[0] + self.ng[0] * c) | (z0 > self.g0[1] + self.ng[1] * c)
        nx, nz = ix1 - ix0 + 1, iz1 - iz0 + 1
        ncell = np.where(outside, 0, nx * nz)
        it = np.repeat(np.arange(len(o)), ncell)
        k = np.arange(ncell.sum()) - np.repeat(np.cumsum(ncell) - ncell, ncell)
        cx = ix0[it] + k // nz[it]
        cz = iz0[it] + k % nz[it]
        cid = cx * self.ng[1] + cz
        cnt = self.cell_start[cid + 1] - self.cell_start[cid]
        it2 = np.repeat(it, cnt)
        j = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ip = self.cell_items[np.repeat(self.cell_start[cid], cnt) + j]
        if not len(it2):
            return it2, ip
        key = np.unique(it2 * len(self.mat) + ip)          # a piece spanning several cells counts once per track
        return key // len(self.mat), key % len(self.mat)

    def column(self, o, d, t_max, chunk=2048, moments=False):
        """Path length per material; moments=True: (length, int s ds, int s^2 ds) per material."""
        o = np.asarray(o, float)
        d = np.asarray(d, float)
        t_max = np.asarray(t_max, float) * np.ones(len(o))
        acc = np.zeros((len(self.mats), len(o)))
        if moments:
            acc1, acc2 = np.zeros_like(acc), np.zeros_like(acc)
        if not len(self.mat):
            return {}
        inv = 1.0 / np.where(d == 0, 1e-300, d)
        for c0 in range(0, len(o), chunk):
            sl = slice(c0, c0 + chunk)
            oc, ic, tm = o[sl], inv[sl], t_max[sl]
            if getattr(self, "grid", None):
                it, ip = self._candidates(oc, d[sl], tm)
                if not len(it):
                    continue
                t1 = (self.lo[ip] - oc[it]) * ic[it]
                t2 = (self.hi[ip] - oc[it]) * ic[it]
                tin = np.maximum(np.minimum(t1, t2).max(1), 0.0)
                tout = np.minimum(np.maximum(t1, t2).min(1), tm[it])
                ok = tout > tin
                it, ip = it[ok], ip[ok]
            else:
                t1 = (self.lo[None] - oc[:, None]) * ic[:, None]      # [n, P, 3]
                t2 = (self.hi[None] - oc[:, None]) * ic[:, None]
                tin = np.maximum(np.minimum(t1, t2).max(2), 0.0)
                tout = np.minimum(np.maximum(t1, t2).min(2), tm[:, None])
                it, ip = np.nonzero(tout > tin)
            if not len(it):
                continue
            Pp = self.P[ip]                                         # [M, K, 4]
            den = np.einsum("mj,mkj->mk", d[c0 + it], Pp[:, :, :3])
            num = Pp[:, :, 3] - np.einsum("mj,mkj->mk", o[c0 + it], Pp[:, :, :3])
            with np.errstate(divide="ignore", invalid="ignore"):
                s = num / den
            enter = np.where(den < 0, s, -np.inf).max(1)
            leave = np.where(den > 0, s, np.inf).min(1)
            outside = ((den == 0) & (num < 0)).any(1)
            a = np.maximum(enter, 0.0)
            b = np.minimum(leave, tm[it])
            L = np.where(outside, 0.0, np.clip(b - a, 0.0, None))
            np.add.at(acc, (self.mat[ip], c0 + it), L)
            if moments:
                e1 = a + L
                np.add.at(acc1, (self.mat[ip], c0 + it), 0.5 * (e1 ** 2 - a ** 2))
                np.add.at(acc2, (self.mat[ip], c0 + it), (e1 ** 3 - a ** 3) / 3.0)
        if moments:
            return {m: (acc[i], acc1[i], acc2[i]) for i, m in enumerate(self.mats)}
        return {m: acc[i] for i, m in enumerate(self.mats)}


def column(pieces, o, d, t_max, chunk=4096):
    """Path length [mm] of each segment o + s d, s in [0, t_max], inside each material: dict name -> [N] array.

    Convex-piece slab clipping: for each piece, entry = max over planes of the entering intersections, exit = min over
    the leaving ones. Pieces are grouped by material; overlaps within a material are summed (geometry overlaps are rare
    and small above the CRV). `pieces` is a list from `load` or a PieceSet (vectorised, same result, much faster).
    """
    if isinstance(pieces, PieceSet):
        return pieces.column(o, d, t_max)
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


def load_tubes(gdml_path, keep_materials=None, bbox=None, skip_names=(), names=None):
    """Tube and polycone solids of the GDML world as cylindrical shells: list of (material, R [3x3], t [3], rmin, rmax,
    half z, phi0, dphi) in the global frame (R maps local to global). Polycones become one shell per z section with the
    section's smaller rmin and larger rmax. Boolean solids use their first constituent (as `load`).

    keep_materials: names to keep (default: everything that is not air or vacuum). bbox: (lo, hi) global corners; keep
    only shells whose global bounding box overlaps it (the tracker region holds thousands of support pieces).
    skip_names: volume-name prefixes to drop with their daughters (e.g. the straws, which a detector package handles).
    names: optional list; the logical-volume path (mother/.../volume) of each returned shell is appended (diagnostics).
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

    def walk(lv, R, t, path):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            p = pv.find("position")
            pos = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            Rl = _rot(pv.find("rotation"))
            R2, t2 = R @ Rl, R @ pos + t
            if skip_names and _strip(child).startswith(tuple(skip_names)):
                continue
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
                    if names is not None:
                        names.append(path + _strip(child))
            walk(child, R2, t2, path + _strip(child) + "/")

    walk(world, np.eye(3), np.zeros(3), "")
    return out


def _add(dct, m, n, idx, val):
    v = dct.setdefault(m, np.zeros(n))
    np.add.at(v, idx, val)


def column_tubes(tubes, o, d, t_max, nsub=16, moments=False):
    """Path length [mm] of each segment o + s d, s in [0, t_max], inside the cylindrical shells: dict material -> [N].
    moments=True: also the first and second moments of s over the material, int s ds and int s^2 ds (same dicts, as
    (length, m1, m2); sector shells by the same nsub sampling).

    Per shell, in its local frame: the chord inside r <= rmax minus the chord inside r < rmin, clipped to |z| <= hz;
    the phi range of a sector is applied by subdividing each inside interval into `nsub` pieces and keeping those whose
    midpoint lies in [phi0, phi0 + dphi] (exact for full tubes; nsub sets the accuracy at sector edges).
    """
    o = np.asarray(o, float)
    d = np.asarray(d, float)
    t_max = np.asarray(t_max, float) * np.ones(len(o))
    out, o1, o2 = {}, {}, {}
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
        acc, a1, a2 = np.zeros(len(cand)), np.zeros(len(cand)), np.zeros(len(cand))
        for s0, s1 in ivals:
            ln = np.clip(s1 - s0, 0, None)
            if full:
                acc += ln
                if moments:
                    e1 = s0 + ln
                    a1 += 0.5 * (e1 ** 2 - s0 ** 2)
                    a2 += (e1 ** 3 - s0 ** 3) / 3.0
                continue
            k = (np.arange(nsub) + 0.5) / nsub
            sm = s0[:, None] + ln[:, None] * k[None, :]
            x = ol[cand, 0][:, None] + dl[cand, 0][:, None] * sm
            y = ol[cand, 1][:, None] + dl[cand, 1][:, None] * sm
            ph = np.mod(np.arctan2(y, x) - phi0, 2 * np.pi)
            ins = ph <= dphi
            acc += ln * np.mean(ins, axis=1)
            if moments:
                a1 += ln * np.mean(ins * sm, axis=1)
                a2 += ln * np.mean(ins * sm * sm, axis=1)
        v = out.setdefault(mat, np.zeros(len(o)))
        v[cand] += acc
        if moments:
            _add(o1, mat, len(o), cand, a1)
            _add(o2, mat, len(o), cand, a2)
    if moments:
        return {m: (out[m], o1.get(m, np.zeros(len(o))), o2.get(m, np.zeros(len(o)))) for m in out}
    return out



class CoaxialShells:
    """Cylindrical shells sharing one axis (the tracker's), organised for fast straight-line columns.

    Shells are grouped by (rmin, rmax). For a track, the radial chord through each group's annulus needs one quadratic
    per distinct radius; each annulus entry/exit interval is then split at the group's shell z boundaries and phi
    sectors by sampling (nsub points per interval), so the cost is O(distinct radii x tracks) plus a lookup, not
    O(shells x tracks). Non-coaxial shells are kept separately and summed with column_tubes.

    Built from load_tubes output: shells whose axis is parallel to `axis_dir` and pass through `axis_point` (within
    tol) are coaxial; the rest are 'other'.
    """

    def __init__(self, tubes, axis_point, axis_dir=(0.0, 0.0, 1.0), tol=1.0):
        self.c = np.asarray(axis_point, float)
        a = np.asarray(axis_dir, float)
        self.a = a / np.linalg.norm(a)
        ref = np.array([1.0, 0.0, 0.0]) if abs(self.a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.e1 = np.cross(self.a, ref)
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.a, self.e1)
        groups, other = {}, []
        for mat, R, tc, rmin, rmax, hz, phi0, dphi in tubes:
            ax = R @ np.array([0.0, 0.0, 1.0])
            off = tc - self.c
            perp = off - (off @ self.a) * self.a
            if abs(abs(ax @ self.a) - 1) < 1e-9 and np.linalg.norm(perp) < tol:
                # phi measured in the shell's local frame: rotate it into this object's (e1, e2) frame
                lx = R @ np.array([1.0, 0.0, 0.0])
                rot = np.arctan2(lx @ self.e2, lx @ self.e1)
                sign = 1.0 if ax @ self.a > 0 else -1.0
                p0 = (rot + sign * phi0) if sign > 0 else (rot - phi0 - dphi)
                zc = (tc - self.c) @ self.a
                groups.setdefault((round(rmin, 4), round(rmax, 4)), []).append(
                    (mat, zc - hz, zc + hz, np.mod(p0, 2 * np.pi), dphi))
            else:
                other.append((mat, R, tc, rmin, rmax, hz, phi0, dphi))
        self.other = other
        self.groups = []
        for (rmin, rmax), sh in groups.items():
            self.groups.append({"rmin": rmin, "rmax": rmax, "mat": np.array([s[0] for s in sh]),
                                "z0": np.array([s[1] for s in sh]), "z1": np.array([s[2] for s in sh]),
                                "p0": np.array([s[3] for s in sh]), "dp": np.array([s[4] for s in sh])})
        self.n_coaxial = sum(len(g["z0"]) for g in self.groups)

    def __len__(self):
        return self.n_coaxial + len(self.other)

    def column(self, o, d, t_max, nsub=16, moments=False):
        """Path length [mm] per material of segments o + s d, s in [0, t_max] (as column/column_tubes); moments=True
        returns (length, int s ds, int s^2 ds) per material (sectors by the nsub sampling).

        Per (rmin, rmax) group: the annulus intervals of each track (two quadratics), then for every shell of the group
        that the interval's z range overlaps, the exact z-clipped sub-interval; phi sectors by sampling that sub-interval
        at nsub points (exact for full rings). Candidate shells are found by sorting the group's shells in z, so the cost
        per track is the number of shells its z range actually overlaps.
        """
        o = np.asarray(o, float)
        d = np.asarray(d, float)
        t_max = np.asarray(t_max, float) * np.ones(len(o))
        w = o - self.c
        ox, oy, oz = w @ self.e1, w @ self.e2, w @ self.a
        dx, dy, dz = d @ self.e1, d @ self.e2, d @ self.a
        A = dx * dx + dy * dy
        B = 2 * (ox * dx + oy * dy)
        r2 = ox * ox + oy * oy
        out, o1, o2 = {}, {}, {}
        if self.other:
            res = column_tubes(self.other, o, d, t_max, nsub, moments=moments)
            for m, v in res.items():
                if moments:
                    out[m], o1[m], o2[m] = v[0].copy(), v[1].copy(), v[2].copy()
                else:
                    out[m] = v
        k = (np.arange(nsub) + 0.5) / nsub

        def chord(r):
            C = r2 - r * r
            disc = B * B - 4 * A * C
            par = A < 1e-15
            ok = np.where(par, C <= 0, disc > 0)
            sq = np.sqrt(np.where(ok & ~par, disc, 0.0))
            a_ = np.where(par, 1.0, A)
            return np.where(par, -np.inf, (-B - sq) / (2 * a_)), np.where(par, np.inf, (-B + sq) / (2 * a_)), ok

        for g in self.groups:
            a0, a1, aok = chord(g["rmax"])
            lo, hi = np.maximum(a0, 0.0), np.minimum(a1, t_max)
            cand = np.flatnonzero(aok & (hi > lo))
            if not len(cand):
                continue
            ivals = []
            if g["rmin"] > 0:
                b0, b1, bok = chord(g["rmin"])
                hole = bok[cand]
                b0c = np.where(hole, b0[cand], hi[cand])
                b1c = np.where(hole, b1[cand], hi[cand])
                ivals.append((lo[cand], np.minimum(hi[cand], np.maximum(b0c, lo[cand]))))
                ivals.append((np.maximum(lo[cand], np.minimum(b1c, hi[cand])), hi[cand]))
            else:
                ivals.append((lo[cand], hi[cand]))
            order = np.argsort(g["z0"])
            z0s, z1s, p0s, dps, mats = (g[c][order] for c in ("z0", "z1", "p0", "dp", "mat"))
            zmax_run = np.maximum.accumulate(z1s)
            full = dps >= 2 * np.pi - 1e-9
            for s0, s1 in ivals:
                live = s1 > s0
                if not live.any():
                    continue
                ci, s0, s1 = cand[live], s0[live], s1[live]
                za, zb = oz[ci] + dz[ci] * s0, oz[ci] + dz[ci] * s1
                zlo, zhi = np.minimum(za, zb), np.maximum(za, zb)
                # shells with z0 < zhi and z1 > zlo: z0 sorted, so scan j = last with z0 < zhi downwards
                j = np.searchsorted(z0s, zhi, "left") - 1
                active = j >= 0
                while active.any():
                    jj = np.where(active, j, 0)
                    over = active & (z1s[jj] > zlo)
                    if over.any():
                        sel = np.flatnonzero(over)
                        js = jj[sel]
                        c = ci[sel]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            t0 = np.where(np.abs(dz[c]) > 1e-15, (z0s[js] - oz[c]) / dz[c], -np.inf)
                            t1 = np.where(np.abs(dz[c]) > 1e-15, (z1s[js] - oz[c]) / dz[c], np.inf)
                        u0 = np.maximum(s0[sel], np.minimum(t0, t1))
                        u1 = np.minimum(s1[sel], np.maximum(t0, t1))
                        ln = np.clip(u1 - u0, 0, None)
                        frac = np.ones(len(sel))
                        sec = ~full[js] & (ln > 0)
                        if moments:
                            e1 = u0 + ln
                            m1 = 0.5 * (e1 ** 2 - u0 ** 2)
                            m2 = (e1 ** 3 - u0 ** 3) / 3.0
                        if sec.any():
                            sm = u0[sec, None] + ln[sec, None] * k[None, :]
                            cs = c[sec]
                            ph = np.mod(np.arctan2(oy[cs, None] + dy[cs, None] * sm, ox[cs, None] + dx[cs, None] * sm)
                                        - p0s[js[sec], None], 2 * np.pi)
                            ins = ph <= dps[js[sec], None]
                            frac[sec] = np.mean(ins, axis=1)
                            if moments:
                                m1[sec] = ln[sec] * np.mean(ins * sm, axis=1)
                                m2[sec] = ln[sec] * np.mean(ins * sm * sm, axis=1)
                        val = ln * frac
                        for m in np.unique(mats[js]):
                            mm = (mats[js] == m) & (val > 0)
                            if mm.any():
                                v = out.setdefault(m, np.zeros(len(o)))
                                np.add.at(v, c[mm], val[mm])
                                if moments:
                                    _add(o1, m, len(o), c[mm], m1[mm])
                                    _add(o2, m, len(o), c[mm], m2[mm])
                    j = j - 1
                    active = active & (j >= 0) & (zmax_run[np.maximum(j, 0)] > zlo)
        if moments:
            z = np.zeros(len(o))
            return {m: (out[m], o1.get(m, z), o2.get(m, z)) for m in out}
        return out


# ---- signed convex pieces: booleans and mother-minus-daughter, exactly (up to the n-gon of round solids) ----
NGON = 128          # polygon sides of a full circle for round solids (equal-area; sagitta 0.2 mm at r = 700 mm)


def _prism(poly, z0, z1):
    """Local planes [K, 4] (n . x <= d) and bounding box (lo, hi) of a prism over a convex polygon, z in [z0, z1]."""
    pp = np.asarray(poly, float)
    if 0.5 * np.sum(pp[:, 0] * np.roll(pp[:, 1], -1) - np.roll(pp[:, 0], -1) * pp[:, 1]) < 0:
        pp = pp[::-1]
    a, b = pp, np.roll(pp, -1, axis=0)
    n = np.c_[b[:, 1] - a[:, 1], -(b[:, 0] - a[:, 0]), np.zeros(len(pp))]
    L = np.linalg.norm(n, axis=1)
    keep = L > 1e-12
    n = n[keep] / L[keep, None]
    d = np.einsum("ij,ij->i", n[:, :2], a[keep])
    P = np.vstack([np.c_[n, d], [[0, 0, 1.0, z1], [0, 0, -1.0, -z0]]])
    return P, np.r_[pp.min(0), z0], np.r_[pp.max(0), z1]


def _annulus_sector(rmin, rmax, phi0, dphi, z0, z1, ngon=NGON):
    """Signed prisms [(sign, P, lo, hi)] of an annulus sector: per wedge of <= 90 deg, + wedge(rmax) - wedge(rmin)."""
    out = []
    nw = max(1, int(np.ceil(dphi / (np.pi / 2) - 1e-9)))
    for k in range(nw):
        a0 = phi0 + k * dphi / nw
        w = dphi / nw
        m = max(2, int(np.ceil(ngon * w / (2 * np.pi))))
        ang = a0 + w * np.arange(m + 1) / m
        f = np.sqrt((w / m) / np.sin(w / m))                       # equal-area polygon radius factor
        for sign, r in ((1, rmax), (-1, rmin)):
            if r > 0:
                poly = np.r_[[[0.0, 0.0]], np.c_[np.cos(ang), np.sin(ang)] * r * f]
                out.append((sign, *_prism(poly, z0, z1)))
    return out


def _intersect(pa, pb):
    """Convex intersection of two signed local pieces (sign product); None if their boxes do not overlap."""
    lo, hi = np.maximum(pa[2], pb[2]), np.minimum(pa[3], pb[3])
    if np.any(lo >= hi):
        return None
    return (pa[0] * pb[0], np.vstack([pa[1], pb[1]]), lo, hi)


def _move(pieces, R, t):
    """Signed local pieces moved by x -> R x + t (planes and boxes)."""
    out = []
    for s, P, lo, hi in pieces:
        n = P[:, :3] @ R.T
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ R.T + t
        out.append((s, np.c_[n, P[:, 3] + n @ t], corners.min(0), corners.max(0)))
    return out


def signed_local(solid, solids, ngon=NGON):
    """Signed convex pieces [(sign, P local, lo, hi)] of a GDML solid; None if a part is not supported.

    box, xtru (two unscaled sections), tube/cutTube (cut planes ignored), polycone (each z section as a tube from its
    smaller rmin to its larger rmax), and subtraction / union / intersection of these (second solid with an inline
    position and rotation, the placement convention of _rot). Subtraction A - B = A - (A n B), union A + B - (A n B)."""
    tag = solid.tag
    ang = np.pi / 180 if solid.get("aunit", "deg") == "deg" else 1.0
    if tag == "box":
        h = 0.5 * np.array([float(solid.get(a)) for a in "xyz"])
        return [(1, *_prism([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]], -h[2], h[2]))]
    if tag == "xtru":
        sec = solid.findall("section")
        if len(sec) != 2 or any(float(s.get("scalingFactor", 1)) != 1 for s in sec):
            return None
        v = [[float(p.get("x")), float(p.get("y"))] for p in solid.findall("twoDimVertex")]
        z0, z1 = float(sec[0].get("zPosition")), float(sec[1].get("zPosition"))
        return [(1, *_prism(tri, z0, z1)) for tri in _convex_pieces_2d(v)]
    if tag in ("tube", "cutTube"):
        hz = 0.5 * float(solid.get("z"))
        return _annulus_sector(float(solid.get("rmin", 0)), float(solid.get("rmax")), float(solid.get("startphi", 0)) * ang,
                               float(solid.get("deltaphi", 360)) * ang, -hz, hz, ngon)
    if tag == "polycone":
        zp = sorted((float(z.get("z")), float(z.get("rmin", 0)), float(z.get("rmax"))) for z in solid.findall("zplane"))
        out = []
        for (z0, a0, b0), (z1, a1, b1) in zip(zp[:-1], zp[1:]):
            if z1 > z0:
                out += _annulus_sector(min(a0, a1), max(b0, b1), float(solid.get("startphi", 0)) * ang,
                                       float(solid.get("deltaphi", 360)) * ang, z0, z1, ngon)
        return out
    if tag in ("subtraction", "union", "intersection"):
        A = signed_local(solids[solid.find("first").get("ref")], solids, ngon)
        B = signed_local(solids[solid.find("second").get("ref")], solids, ngon)
        if A is None or B is None or solid.find("rotationref") is not None or solid.find("positionref") is not None:
            return None
        p = solid.find("position")
        if p is not None or solid.find("rotation") is not None:
            t = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            B = _move(B, _rot(solid.find("rotation")), t)
        AB = [q for q in (_intersect(a, b) for a in A for b in B) if q is not None]
        if tag == "intersection":
            return AB
        neg = [(-s, P, lo, hi) for s, P, lo, hi in AB]
        return A + neg if tag == "subtraction" else A + B + neg
    return None


def signed_subtree(gdml_path, mother, bbox=None, labels=False, ngon=NGON):
    """Signed global convex pieces [(sign, material, P, lo, hi)] of every non-vacuum volume below `mother` (logical
    volume name without its pointer suffix), and the skipped placements {(name, solid tag, material): count}.

    Each volume adds its solid in its material, and its solid with the opposite sign in its mother's material when the
    mother is not vacuum, so mother minus daughters is exact. bbox: (lo, hi) global corners; keep only pieces whose box
    overlaps it. labels: material keys 'volume|material' (volume = leaf name without copy indices)."""
    root = ET.parse(gdml_path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    vac = re.compile(r"AIR|Vacuum|VACUUM|Galactic")
    out, skipped = [], {}

    def emit(local, R, t, key, sign):
        for s, P, lo, hi in _move(local, R, t):
            if bbox is not None and (np.any(hi < bbox[0]) or np.any(lo > bbox[1])):
                continue
            out.append((s * sign, key, P, lo, hi))

    def walk(lv, R, t, inside, mkey):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            n = _strip(child)
            p = pv.find("position")
            pos = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            R2, t2 = R @ _rot(pv.find("rotation")), R @ pos + t
            ins = inside or n == mother
            key = None
            if ins and n != mother:
                mat = _strip(vols[child].find("materialref").get("ref"))
                solid = solids[vols[child].find("solidref").get("ref")]
                local = signed_local(solid, solids, ngon)
                key = None if vac.search(mat) else (f"{re.sub(r'(_[0-9]+)+$', '', n)}|{mat}" if labels else mat)
                if local is None:
                    if key is not None or mkey is not None:
                        k = (re.sub(r"(_[0-9]+)+$", "", n), solid.tag, mat)
                        skipped[k] = skipped.get(k, 0) + 1
                else:
                    if key is not None:
                        emit(local, R2, t2, key, 1)
                    if mkey is not None:
                        emit(local, R2, t2, mkey, -1)
            walk(child, R2, t2, ins, key)

    walk(root.find("setup/world").get("ref"), np.eye(3), np.zeros(3), False, None)
    return out, skipped


class SignedPieceSet:
    """Column per material of signed convex pieces (signed_subtree): sum of + pieces minus sum of - pieces, clipped
    at 0. Pieces are packed into PieceSets by sign and plane count (so wide polygons do not pad every box)."""

    BUCKETS = (8, 16, 64, 1 << 30)

    def __init__(self, pieces, cell=500.0):
        self.sets = []
        self.mats = sorted({q[1] for q in pieces})
        for sign in (1, -1):
            lo_k = 0
            for hi_k in self.BUCKETS:
                sel = [(m, P, lo, hi) for s, m, P, lo, hi in pieces if s == sign and lo_k < len(P) <= hi_k]
                if sel:
                    self.sets.append((sign, PieceSet(sel).build_grid(cell)))
                lo_k = hi_k
        self.n = len(pieces)

    def __len__(self):
        return self.n

    def column(self, o, d, t_max, nsub=None, moments=False):
        """Column per material (clipped at 0); moments=True: (length, int s ds, int s^2 ds), each signed sum, the
        moments zeroed where the length clips to 0."""
        out = {}
        for sign, ps in self.sets:
            for m, c in ps.column(o, d, t_max, moments=moments).items():
                v = out.setdefault(m, np.zeros((3, len(o))) if moments else np.zeros(len(o)))
                v += sign * (np.asarray(c) if moments else c)
        if moments:
            res = {}
            for m, v in out.items():
                ok = v[0] > 1e-9
                res[m] = (np.where(ok, v[0], 0.0), np.where(ok, v[1], 0.0), np.where(ok, v[2], 0.0))
            return res
        return {m: np.clip(v, 0.0, None) for m, v in out.items()}
