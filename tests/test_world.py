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
