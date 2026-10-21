"""Which GDML volumes the fast world keeps: python3 -m mu2efastsim.worldcensus GDML [--y-min MM] [--top N].

world.load turns box and xtru solids into convex pieces and ignores every other solid type; world.load_tubes adds
tubes and polycones (used inside the tracker). This walks the same placements and reports, per solid type and per
material, the number of placed volumes and their mass (solid volume x density, booleans by their first constituent as
world.load does), split into kept by world.load, kept by load_tubes only, and skipped by both. Only volumes whose
global bounding box reaches above y_min are counted (the stage C world above the tracker plane).
"""
import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from .world import _rot, _strip

BOX_LIKE = ("box", "xtru")
TUBE_LIKE = ("tube", "cutTube", "polycone")


def _first(solid, solids):
    while solid.tag in ("subtraction", "union", "intersection"):
        solid = solids[solid.find("first").get("ref")]
    return solid


def _volume_mm3(s):
    """Solid volume [mm3] for the common GDML solids (approximate for xtru: prism of the polygon); None if unknown."""
    g = lambda k, d=0.0: float(s.get(k, d))
    ang = np.pi / 180 if s.get("aunit", "deg") == "deg" else 1.0
    if s.tag == "box":
        return g("x") * g("y") * g("z")
    if s.tag in ("tube", "cutTube"):
        return 0.5 * g("deltaphi", 360) * ang * (g("rmax") ** 2 - g("rmin") ** 2) * g("z")
    if s.tag == "cone":
        dphi = g("deltaphi", 360) * ang
        f = lambda a, b: (a * a + a * b + b * b) / 3
        return 0.5 * dphi * (f(g("rmax1"), g("rmax2")) - f(g("rmin1"), g("rmin2"))) * g("z")
    if s.tag == "polycone":
        zp = sorted((float(z.get("z")), float(z.get("rmin", 0)), float(z.get("rmax"))) for z in s.findall("zplane"))
        dphi = g("deltaphi", 360) * ang
        f = lambda a, b: (a * a + a * b + b * b) / 3
        return sum(0.5 * dphi * (f(a1, b1) - f(a0, b0)) * (z1 - z0) for (z0, a0, a1), (z1, b0, b1) in zip(zp[:-1], zp[1:]))
    if s.tag == "sphere":
        return (4 / 3) * np.pi * (g("rmax") ** 3 - g("rmin") ** 3) * (g("deltaphi", 360) / 360) * \
            (np.cos(g("starttheta") * ang) - np.cos((g("starttheta") + g("deltatheta", 180)) * ang)) / 2
    if s.tag == "trd":
        return g("z") * (g("x1") * g("y1") + g("x2") * g("y2") + np.sqrt(g("x1") * g("y1") * g("x2") * g("y2"))) / 3
    if s.tag == "xtru":
        v = np.array([[float(p.get("x")), float(p.get("y"))] for p in s.findall("twoDimVertex")])
        sec = [float(x.get("zPosition")) for x in s.findall("section")]
        area = 0.5 * abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))
        return area * (max(sec) - min(sec))
    return None


def census(gdml_path, y_min=None):
    root = ET.parse(gdml_path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    rho = {}
    for m in root.find("materials").findall("material"):
        d = m.find("D")
        if d is not None:
            rho[_strip(m.get("name"))] = float(d.get("value"))
    world = root.find("setup/world").get("ref")
    out = defaultdict(lambda: {"n": 0, "kg": 0.0, "unknown_volume": 0})

    def walk(lv, R, t):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            p = pv.find("position")
            pos = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
            R2, t2 = R @ _rot(pv.find("rotation")), R @ pos + t
            mref = vols[child].find("materialref")
            mat = _strip(mref.get("ref")) if mref is not None else ""
            if not re.search(r"AIR|Vacuum|VACUUM|Galactic", mat):
                s = _first(solids[vols[child].find("solidref").get("ref")], solids)
                if y_min is None or t2[1] + _extent(s) >= y_min:
                    kept = "load" if s.tag in BOX_LIKE else ("load_tubes" if s.tag in TUBE_LIKE else "skipped")
                    v = _volume_mm3(s)
                    r = out[(kept, s.tag, mat)]
                    r["n"] += 1
                    if v is None:
                        r["unknown_volume"] += 1
                    else:
                        r["kg"] += v * 1e-3 * rho.get(mat, 0.0) * 1e-3
            walk(child, R2, t2)

    walk(world, np.eye(3), np.zeros(3))
    return dict(out)


def _extent(s):
    """Generous half-size [mm] of a solid (for the y_min test on the placement centre)."""
    vals = [float(s.get(k)) for k in ("x", "y", "z", "rmax", "rmax1", "rmax2", "x1", "x2", "y1", "y2") if s.get(k)]
    for z in s.findall("zplane"):
        vals += [abs(float(z.get("z"))), float(z.get("rmax"))]
    for p in s.findall("twoDimVertex"):
        vals += [abs(float(p.get("x"))), abs(float(p.get("y")))]
    return max(vals) if vals else 1e9


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gdml")
    ap.add_argument("--y-min", type=float, default=None, help="only volumes reaching above this height [mm]")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)
    c = census(a.gdml, a.y_min)
    tot = defaultdict(lambda: [0, 0.0])
    for (k, tag, mat), r in c.items():
        tot[k][0] += r["n"]
        tot[k][1] += r["kg"]
    print(f"volumes reaching above y = {a.y_min} mm (not air/vacuum):")
    for k, (n, kg) in sorted(tot.items()):
        print(f"  {k:10s} {n:7d} volumes  {kg / 1e3:12.1f} t")
    print(f"largest {a.top} (kept_by, solid, material) by mass:")
    for (k, tag, mat), r in sorted(c.items(), key=lambda kv: -kv[1]["kg"])[:a.top]:
        print(f"  {k:10s} {tag:12s} {mat:24s} n {r['n']:6d}  {r['kg'] / 1e3:10.2f} t" +
              (f"  ({r['unknown_volume']} without a volume formula)" if r["unknown_volume"] else ""))


if __name__ == "__main__":
    main()
