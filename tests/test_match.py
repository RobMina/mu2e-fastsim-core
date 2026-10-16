"""Unit tests of mu2efastsim.match: multi-column joins, lookups and the KS test."""
import numpy as np

from mu2efastsim import match


def test_join_multicolumn():
    a = np.array([[1, 2, 3], [1, 2, 4], [5, 0, 0]])
    b = np.array([[5, 0, 0], [1, 2, 3]])
    ia, ib = match.join(a, b)
    assert {tuple(a[i]) for i in ia} == {(1, 2, 3), (5, 0, 0)}
    assert all((a[i] == b[j]).all() for i, j in zip(ia, ib))


def test_lookup_many_to_one():
    out = match.lookup(np.array([[1, 2], [1, 2], [3, 4], [9, 9]]), np.array([[3, 4], [1, 2]]), np.array([7.0, 5.0]))
    assert out[:3].tolist() == [5.0, 5.0, 7.0] and np.isnan(out[3])


def test_ks():
    rng = np.random.default_rng(1)
    assert match.ks_2samp(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000))[1] > 0.01
    assert match.ks_2samp(rng.normal(0, 1, 5000), rng.normal(0.2, 1, 5000))[1] < 1e-6
