"""Unit tests: the JAX twin of the radiative muon loss against the NumPy sampler (same tables, same distribution)."""
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import materials as M                                  # noqa: E402
from mu2efastsim import muradiative as R                                # noqa: E402
from mu2efastsim import muradiative_jax as RJ                           # noqa: E402


@pytest.mark.parametrize("mat,p,L", [(M.COPPER, 100_000.0, 200.0), (M.CONCRETE_MARS, 20_000.0, 2000.0),
                                     (M.STAINLESS_STEEL, 1_000_000.0, 500.0)])
def test_device_matches_numpy_distribution(mat, p, L):
    n = 200_000
    a = R.sample_loss(np.random.default_rng(3), mat, np.full(n, p), np.full(n, L))
    b = np.asarray(RJ.sample_loss(jax.random.PRNGKey(3), RJ.table(mat), jax.numpy.full(n, p), jax.numpy.full(n, L)))
    se = np.sqrt(a.var() / n + b.var() / n)
    assert abs(b.mean() - a.mean()) < 4 * se
    assert abs((b > 0).mean() - (a > 0).mean()) < 4 * np.sqrt((a > 0).mean() / n) + 1e-4
    for q in (0.9, 0.99):
        qa, qb = np.quantile(a, q), np.quantile(b, q)
        assert abs(qb - qa) <= 0.1 * max(qa, 1.0)


def test_device_zero_below_threshold_and_capped():
    p = jax.numpy.array([300.0, 900.0, 2000.0, 5e6])
    L = jax.numpy.full(4, 1e5)
    s = np.asarray(RJ.sample_loss(jax.random.PRNGKey(1), RJ.table(M.LEAD), p, L))
    tk = np.sqrt(np.asarray(p) ** 2 + R.MMU ** 2) - R.MMU
    assert s[0] == 0 and s[1] == 0 and (s <= tk).all()
