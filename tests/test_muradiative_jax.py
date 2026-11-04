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


def test_stacked_per_lane_matches_single_tables():
    mats = [M.COPPER, M.CONCRETE_MARS, M.ALUMINIUM_6061]
    tab = RJ.stack(mats)
    n = 60_000
    mat = jax.numpy.asarray(np.repeat(np.arange(3), n // 3))
    p = jax.numpy.full(n, 50_000.0)
    L = jax.numpy.full(n, 1000.0)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(5), i))(jax.numpy.arange(n))
    s = np.asarray(RJ.sample_loss(keys, tab, p, L, mat))
    for i, m in enumerate(mats):
        ref = np.asarray(RJ.sample_loss(jax.random.PRNGKey(9), RJ.table(m), p[: n // 3], L[: n // 3]))
        a = s[i * n // 3:(i + 1) * n // 3]
        assert abs(a.mean() - ref.mean()) < 4 * np.sqrt(a.var() / a.size + ref.var() / ref.size)


def test_after_ionisation_follows_slab_loss_rules():
    from mu2efastsim import transport as T
    tab = RJ.table(M.CONCRETE_MARS)
    p = jax.numpy.array([500.0, 20_000.0, 20_000.0, 20_000.0])
    L = jax.numpy.array([1000.0, 1000.0, 0.0, 1000.0])
    tk0 = np.sqrt(np.asarray(p) ** 2 + R.MMU ** 2) - R.MMU
    ion = jax.numpy.array([tk0[0], 500.0, 0.0, 500.0])
    out = np.asarray(RJ.after_ionisation(jax.random.PRNGKey(2), tab, p, L, ion))
    assert out[0] == tk0[0] and out[2] == 0.0 and out[1] >= 500.0 and out[3] >= 500.0 and (out <= tk0).all()
    assert T.TK_STOP > 0
