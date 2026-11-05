"""Unit tests: convex-piece ray clipping and 2D ear clipping in mu2efastsim.world (synthetic geometry)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import world as W                                       # noqa: E402

CUBE = np.array([[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1], [0, 0, 1, 1], [0, 0, -1, 1]], float)
PIECES = [("X", CUBE, np.array([-1.0, -1, -1]), np.array([1.0, 1, 1]))]


def test_vertical_and_miss_and_partial():
    o = np.array([[0, 5, 0], [0.5, 5, 0.5], [3, 5, 0]], float)
    d = np.array([[0, -1, 0]] * 3, float)
    assert np.allclose(W.column(PIECES, o, d, 100.0)["X"], [2, 2, 0])
    assert np.allclose(W.column(PIECES, o, d, 4.5)["X"], [0.5, 0.5, 0])


def test_slanted_through_centre():
    th = 0.3
    d = np.array([[np.sin(th), -np.cos(th), 0]])
    o = -d * 10.0  # passes through the origin
    assert np.isclose(W.column(PIECES, o, d, 100.0)["X"][0], 2 / np.cos(th))


def test_ear_clipping_area_preserved():
    # an L-shaped (non-convex) polygon
    v = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [0, 3]]
    tris = W._convex_pieces_2d(v)
    area = sum(0.5 * abs((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1]))
               for t in tris)
    assert len(tris) == 4 and np.isclose(area, 4 * 1 + 1 * 2)



def test_tube_shell_chords():
    # full shell rmin 1, rmax 2, |z| <= 5 along z: a vertical (y) line through the axis crosses 2 x 1 mm of wall
    tube = [("X", np.eye(3), np.zeros(3), 1.0, 2.0, 5.0, 0.0, 2 * np.pi)]
    o = np.array([[0.0, 10.0, 0.0], [1.5, 10.0, 0.0], [0.0, 10.0, 7.0]])
    d = np.array([[0.0, -1.0, 0.0]] * 3)
    c = W.column_tubes(tube, o, d, 100.0)["X"]
    assert np.isclose(c[0], 2.0) and np.isclose(c[1], 2 * np.sqrt(4 - 2.25)) and c[2] == 0


def test_tube_sector_keeps_only_its_phi_range():
    # upper half shell (phi 0..180 deg): the vertical line through the axis crosses only the top wall (1 mm)
    tube = [("X", np.eye(3), np.zeros(3), 1.0, 2.0, 5.0, 0.0, np.pi)]
    c = W.column_tubes(tube, np.array([[0.0, 10.0, 0.0]]), np.array([[0.0, -1.0, 0.0]]), 100.0, nsub=64)["X"]
    assert abs(c[0] - 1.0) < 0.05



def test_coaxial_shells_match_column_tubes():
    rng = np.random.default_rng(3)
    tubes = []
    for i in range(40):
        rmin, rmax = (700.0, 819.0) if i % 2 else (810.0, 850.0)
        zc = rng.uniform(-1500, 1500)
        phi0, dphi = (0.0, 2 * np.pi) if i % 3 == 0 else (rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 2.0))
        tubes.append(("A" if i % 2 else "B", np.eye(3), np.array([10.0, 20.0, zc]), rmin, rmax, rng.uniform(5, 40), phi0, dphi))
    n = 3000
    o = np.c_[rng.uniform(-900, 900, n) + 10, np.full(n, 1200.0), rng.uniform(-1600, 1600, n)]
    d = np.c_[rng.normal(0, .4, n), -np.ones(n), rng.normal(0, .4, n)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    ref = W.column_tubes(tubes, o, d, 4000.0, nsub=64)
    cs = W.CoaxialShells(tubes, axis_point=[10.0, 20.0, 0.0])
    assert cs.n_coaxial == 40 and not cs.other
    got = cs.column(o, d, 4000.0, nsub=64)
    for m in ("A", "B"):
        assert np.allclose(got.get(m, 0), ref.get(m, 0), atol=0.2), m       # phi sampling differs at sector edges



def test_pieceset_matches_column_loop():
    rng = np.random.default_rng(4)
    pieces = []
    for _ in range(60):
        c = rng.uniform(-50, 50, 3)
        h = rng.uniform(1, 8, 3)
        P = np.array([[1, 0, 0, c[0] + h[0]], [-1, 0, 0, -(c[0] - h[0])], [0, 1, 0, c[1] + h[1]],
                      [0, -1, 0, -(c[1] - h[1])], [0, 0, 1, c[2] + h[2]], [0, 0, -1, -(c[2] - h[2])]], float)
        if rng.random() < 0.3:
            P = P[:5]                                      # a piece with fewer planes (padding path)
            P = np.vstack([P, [0, 0, -1, -(c[2] - h[2])]])
        pieces.append(("A" if rng.random() < 0.5 else "B", P, c - h, c + h))
    n = 3000
    o = np.c_[rng.uniform(-60, 60, n), np.full(n, 100.0), rng.uniform(-60, 60, n)]
    d = np.c_[rng.normal(0, .3, n), -np.ones(n), rng.normal(0, .3, n)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    ref = W.column(pieces, o, d, 250.0)
    got = W.column(W.PieceSet(pieces), o, d, 250.0)
    grid = W.column(W.PieceSet(pieces).build_grid(cell=7.0), o, d, 250.0)
    for m in ref:
        assert np.allclose(got[m], ref[m], atol=1e-9) and np.allclose(grid[m], ref[m], atol=1e-9)


def _signed_column(local, o, d, L):
    ps = W.SignedPieceSet([(s, "X", P, lo, hi) for s, P, lo, hi in local], cell=50.0)
    return ps.column(o, d, L).get("X", np.zeros(len(o)))


def test_signed_annulus_chords_match_tube():
    local = W._annulus_sector(1.0, 2.0, 0.0, 2 * np.pi, -5.0, 5.0, ngon=256)
    o = np.array([[0.0, 10, 0], [1.5, 10, 0], [0.3, 10, 2]])
    d = np.array([[0.0, -1, 0]] * 3)
    tube = [("X", np.eye(3), np.zeros(3), 1.0, 2.0, 5.0, 0.0, 2 * np.pi)]
    ref = W.column_tubes(tube, o, d, 100.0)["X"]
    assert np.allclose(_signed_column(local, o, d, np.full(3, 100.0)), ref, atol=2e-3)


def test_signed_subtraction_box_minus_box():
    import xml.etree.ElementTree as ET
    xml = """<solids><box name="A" x="10" y="10" z="10"/><box name="B" x="4" y="20" z="4"/>
             <subtraction name="S"><first ref="A"/><second ref="B"/><position name="p" x="2" y="0" z="0"/></subtraction></solids>"""
    sol = {s.get("name"): s for s in ET.fromstring(xml)}
    local = W.signed_local(sol["S"], sol)
    o = np.array([[2.0, 20, 0], [-3.0, 20, 0], [4.5, 20, 0]])   # through the hole, beside it, at its edge region
    d = np.array([[0.0, -1, 0]] * 3)
    assert np.allclose(_signed_column(local, o, d, np.full(3, 100.0)), [0.0, 10.0, 10.0])


def test_moments_coaxial_and_pieces():
    # a full ring r 1-2 about z, |z| < 5; vertical line x = 0 from y = 10 down: material at s in [8, 9] and [11, 12]
    tube = [("X", np.eye(3), np.zeros(3), 1.0, 2.0, 5.0, 0.0, 2 * np.pi)]
    o, d, L = np.array([[0.0, 10, 0]]), np.array([[0.0, -1, 0]]), np.array([20.0])
    m1 = (9 ** 2 - 8 ** 2 + 12 ** 2 - 11 ** 2) / 2
    m2 = (9 ** 3 - 8 ** 3 + 12 ** 3 - 11 ** 3) / 3
    for res in (W.column_tubes(tube, o, d, L, moments=True)["X"],
                W.CoaxialShells(tube, axis_point=(0, 0, 0)).column(o, d, L, moments=True)["X"]):
        assert np.allclose([res[0][0], res[1][0], res[2][0]], [2.0, m1, m2])
    # the cube of PIECES (|x|, |y|, |z| <= 1) from y = 5 down: s in [4, 6]
    r = W.PieceSet(PIECES).column(np.array([[0.0, 5, 0]]), d, np.array([100.0]), moments=True)["X"]
    assert np.allclose([r[0][0], r[1][0], r[2][0]], [2.0, (36 - 16) / 2, (216 - 64) / 3])
    sp = W.SignedPieceSet([(1, "X", P, lo, hi) for _, P, lo, hi in PIECES])
    r = sp.column(np.array([[0.0, 5, 0]]), d, np.array([100.0]), moments=True)["X"]
    assert np.allclose([r[0][0], r[1][0], r[2][0]], [2.0, 10.0, 152 / 3])


def test_build_grid_cells_match_reference_loop():
    rng = np.random.default_rng(3)
    lo = rng.uniform(0, 4000, (300, 3))
    pieces = [("X", CUBE, l, l + rng.uniform(10, 900, 3)) for l in lo]
    ps = W.PieceSet(pieces).build_grid(250.0)
    g0, c, ng = ps.g0, ps.cell, ps.ng
    cells = [[] for _ in range(ng[0] * ng[1])]
    for ip in range(len(pieces)):
        ix0, ix1 = [int(np.clip((v - g0[0]) // c, 0, ng[0] - 1)) for v in (ps.lo[ip, 0], ps.hi[ip, 0])]
        iz0, iz1 = [int(np.clip((v - g0[1]) // c, 0, ng[1] - 1)) for v in (ps.lo[ip, 2], ps.hi[ip, 2])]
        for ix in range(ix0, ix1 + 1):
            for iz in range(iz0, iz1 + 1):
                cells[ix * ng[1] + iz].append(ip)
    assert np.array_equal(ps.cell_start, np.r_[0, np.cumsum([len(x) for x in cells])])
    assert np.array_equal(ps.cell_items, [i for x in cells for i in x])


def test_signed_cone_chords_match_frustum():
    import xml.etree.ElementTree as ET
    xml = """<solids><cone name="C" rmin1="20" rmax1="100" rmin2="10" rmax2="50" z="200" startphi="0" deltaphi="360"
             aunit="deg" lunit="mm"/></solids>"""
    sol = {s.get("name"): s for s in ET.fromstring(xml)}
    assert W.signed_local(sol["C"], sol) is None                       # opt-in: off by default
    local = W.signed_local(sol["C"], sol, ngon=512, cones=True)
    z = np.array([-50.0, 0.0, 60.0])
    rmax, rmin = 75.0 - 0.25 * z, 15.0 - 0.05 * z                      # linear radii at those z
    o = np.c_[np.zeros(3), np.full(3, 300.0), z]
    d = np.array([[0.0, -1, 0]] * 3)
    got = _signed_column(local, o, d, np.full(3, 1000.0))
    assert np.allclose(got, 2 * (rmax - rmin), rtol=2e-3)
    o2 = np.array([[30.0, 0.0, -150.0]])                                # along z at r = 30: inside only where rmin < 30 < rmax
    got2 = _signed_column(local, o2, np.array([[0.0, 0, 1]]), np.array([300.0]))
    assert np.isclose(got2[0], 200.0, rtol=1e-3)


def test_signed_subtree_skip_drops_daughters(tmp_path):
    g = """<gdml><materials/><solids><box name="W" x="1000" y="1000" z="1000"/><box name="A" x="100" y="100" z="100"/>
           <box name="B" x="10" y="10" z="10"/></solids><structure>
           <volume name="Bv"><materialref ref="Fe"/><solidref ref="B"/></volume>
           <volume name="Av"><materialref ref="Al"/><solidref ref="A"/><physvol><volumeref ref="Bv"/></physvol></volume>
           <volume name="Hv"><materialref ref="G4_AIR"/><solidref ref="W"/><physvol><volumeref ref="Av"/></physvol></volume>
           <volume name="Wv"><materialref ref="G4_AIR"/><solidref ref="W"/><physvol><volumeref ref="Hv"/></physvol></volume>
           </structure><setup name="Default" version="1.0"><world ref="Wv"/></setup></gdml>"""
    f = tmp_path / "t.gdml"
    f.write_text(g)
    full, _ = W.signed_subtree(str(f), "Hv")
    cut, _ = W.signed_subtree(str(f), "Hv", skip=("Bv",))
    assert {p[1] for p in full} == {"Al", "Fe"} and {p[1] for p in cut} == {"Al"}
    ps = W.SignedPieceSet(cut, cell=50.0)
    col = ps.column(np.array([[0.0, 200, 0]]), np.array([[0.0, -1, 0]]), np.array([400.0]))
    assert np.isclose(col["Al"][0], 90.0)                              # mother minus the skipped daughter


def test_signed_polyhedra_flat_sides():
    import xml.etree.ElementTree as ET
    xml = """<solids><polyhedra name="P" numsides="4" startphi="-45" deltaphi="360" aunit="deg" lunit="mm">
             <zplane z="-10" rmin="5" rmax="20"/><zplane z="10" rmin="5" rmax="20"/></polyhedra></solids>"""
    sol = {s.get("name"): s for s in ET.fromstring(xml)}
    local = W.signed_local(sol["P"], sol, cones=True)            # a square tube: outer half-width 20, inner 5
    o = np.array([[0.0, 100.0, 0.0], [12.0, 100.0, 0.0], [19.0, 100.0, 9.0]])
    got = _signed_column(local, o, np.array([[0.0, -1, 0]] * 3), np.full(3, 300.0))
    assert np.allclose(got, [30.0, 40.0, 40.0], atol=1e-6)
