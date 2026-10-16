"""Unit tests: generic ray-box crossings (synthetic boxes; numpy and JAX backends)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import crossings as cx                                 # noqa: E402

LO = np.array([[-1.0, -1, -1], [-1, -4, -1]])
HI = np.array([[1.0, 1, 1], [1, -2, 1]])


def test_vertical_track_crosses_both_boxes():
    c = cx.crossings(np.array([[0.0, 5, 0]]), np.array([[0.0, -1, 0]]), LO, HI, backend="numpy")
    assert len(c) == 2 and np.allclose(sorted(c[:, 4]), [2, 2])


def test_slanted_path_length():
    th = 0.4
    d = np.array([[np.sin(th), -np.cos(th), 0.0]])
    c = cx.crossings(-d * 10.0, d, LO[:1], HI[:1], backend="numpy")
    assert len(c) == 1 and np.isclose(c[0, 4], 2 / np.cos(th))


def test_miss():
    c = cx.crossings(np.array([[5.0, 5, 0]]), np.array([[0.0, -1, 0]]), LO, HI, backend="numpy")
    assert len(c) == 0


def test_numpy_and_jax_backends_agree():
    if not cx.HAVE_JAX:
        pytest.skip("no jax")
    rng = np.random.default_rng(5)
    o = np.c_[rng.uniform(-2, 2, 2000), np.full(2000, 6.0), rng.uniform(-2, 2, 2000)]
    d = np.c_[rng.normal(0, 0.3, 2000), -np.ones(2000), rng.normal(0, 0.3, 2000)]
    d /= np.linalg.norm(d, axis=1)[:, None]
    a = cx.crossings(o, d, LO, HI, backend="numpy")
    b = cx.crossings(o, d, LO, HI, backend="jax")
    assert a.shape == b.shape and np.array_equal(a[:, :2], b[:, :2])
    assert np.allclose(a[:, 2:], b[:, 2:], rtol=0, atol=1e-9)
