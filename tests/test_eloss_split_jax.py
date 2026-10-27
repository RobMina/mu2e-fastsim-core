"""Synthetic restricted-soft/explicit-hard contracts and spin-1/2 moment checks on CPU.

Statistical gates use analytic compound-Poisson errors, including the fourth
cumulant for the variance uncertainty, rather than tolerances tuned to one seed.
No detector reference data or beam-stage integration is required here.
"""
import numpy as np
import pytest
from scipy.integrate import quad

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mu2efastsim import eloss as E
from mu2efastsim import eloss_jax as EJ
from mu2efastsim import eloss_split_jax as S
from mu2efastsim import materials as M

MAT = M.POLYSTYRENE


def _quadrature(mat, p, cut, order, mass=E.MMU):
    kinetic, beta2, _, _, top = E.kinematics(p, mass)
    if cut >= top:
        return 0.0
    scale = E.TWOPI_MC2_RCL2 * mat.electron_density_per_mm3 / beta2
    total = kinetic + mass

    def integrand(log_t):
        t = np.exp(log_t)
        return scale * t ** (order - 1) * (1 - beta2 * t / top + 0.5 * (t / total) ** 2)

    return quad(integrand, np.log(cut), np.log(top), epsabs=1e-13, epsrel=2e-12)[0]


@pytest.mark.parametrize("mat", [MAT, M.ALUMINIUM, M.MILD_STEEL])
def test_analytic_moments_match_quadrature_and_numpy_rate(mat):
    p = np.array([1.85, 15.0, 40.0, 100.0, 300.0, 3000.0, 10000.0])
    top = E.kinematics(p)[-1]
    cut = top * np.array([0.2, 0.8, 0.1, 0.5, 0.01, 1e-4, 0.7])
    actual = S.hard_moments_per_mm(mat, p, cut)
    expected = [[_quadrature(mat, pp, tc, k) for pp, tc in zip(p, cut)] for k in range(3)]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    np.testing.assert_allclose(actual.rate, E.delta_rate_per_mm(mat, p, cut), rtol=2e-12)
    for helper, value in zip((S.hard_rate_per_mm, S.hard_mean_per_mm, S.hard_variance_per_mm), actual):
        np.testing.assert_array_equal(helper(mat, p, cut), value)
    assert all(value.dtype == jnp.float64 for value in actual)


def test_hard_mean_is_inclusive_minus_restricted_bethe_where_unclamped():
    p = jnp.array([40.0, 100.0, 300.0, 3000.0, 10000.0])
    cut = EJ.kinematics(p)[-1] * 0.2
    difference = EJ.dedx_restricted(MAT, p, 1e9) - EJ.dedx_restricted(MAT, p, cut)
    np.testing.assert_allclose(S.hard_mean_per_mm(MAT, p, cut), difference, rtol=2e-12)


def test_moments_near_endpoint_and_at_nonmuon_mass():
    p = np.array([15.0, 3000.0, 1e6])
    top = E.kinematics(p)[-1]
    cut = top * (1 - 1e-10)
    moments = S.hard_moments_per_mm(MAT, p, cut)
    beta2 = E.kinematics(p)[1]
    total = np.hypot(p, E.MMU)
    density = E.TWOPI_MC2_RCL2 * MAT.electron_density_per_mm3 / beta2
    midpoint = 0.5 * (cut + top)
    rate = density * (top - cut) * (1 / midpoint ** 2 - beta2 / (top * midpoint) + 0.5 / total ** 2)
    for order, actual in enumerate(moments):
        np.testing.assert_allclose(actual, rate * midpoint ** order, rtol=2e-6, atol=0)
        assert np.all(np.asarray(actual) > 0)
    mass = 938.27208816
    p, cut = 300.0, 0.01
    np.testing.assert_allclose(S.hard_moments_per_mm(MAT, p, cut, mass),
                               [_quadrature(MAT, p, cut, k, mass) for k in range(3)], rtol=2e-12)


def test_restricted_mean_is_mandatory_and_not_inferred():
    args = (jax.random.key(1), jax.random.key(2), MAT, 100.0, 0.1, 0.01)
    with pytest.raises(TypeError, match="restricted_mean"):
        S.sample_loss_split(*args)
    with pytest.raises(ValueError, match="restricted_mean"):
        S.sample_loss_split(*args, restricted_mean=None)
    tiny_mean = 2e-6
    out = S.sample_loss_split(*args, restricted_mean=tiny_mean)
    assert float(out.soft_requested) == tiny_mean


def test_reproducibility_separate_keys_and_existing_soft_sampler():
    p = jnp.full(1024, 300.0)
    length, cut = 4.0, 0.0872
    mean = EJ.dedx_restricted(MAT, p, cut) * length
    sk, hk = jax.random.key(12), jax.random.key(13)

    def sample(soft_key=sk, hard_key=hk, restricted_mean=mean):
        return S.sample_loss_split(soft_key, hard_key, MAT, p, length, cut, restricted_mean=restricted_mean)

    a, repeat = sample(), sample()
    assert isinstance(a, S.SplitLoss)
    for x, y in zip(a, repeat):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.soft_requested, EJ.sample_loss(sk, MAT, p, length, cut, mean=mean))
    changed_hard = sample(hard_key=jax.random.key(14))
    np.testing.assert_array_equal(a.soft_requested, changed_hard.soft_requested)
    assert not np.array_equal(a.hard_requested, changed_hard.hard_requested)
    changed_soft = sample(soft_key=jax.random.key(15))
    assert not np.array_equal(a.soft_requested, changed_soft.soft_requested)
    np.testing.assert_array_equal(a.hard_requested, changed_soft.hard_requested)
    np.testing.assert_array_equal(a.hard_count, changed_soft.hard_count)
    changed_mean = sample(restricted_mean=mean * 1.4)
    np.testing.assert_array_equal(a.hard_requested, changed_mean.hard_requested)
    np.testing.assert_array_equal(a.hard_count, changed_mean.hard_count)
    assert not np.array_equal(a.soft_requested, changed_mean.soft_requested)
    count_key, _ = jax.random.split(hk)
    expected_counts = jax.random.poisson(count_key, S.hard_rate_per_mm(MAT, p, cut) * length,
                                         shape=p.shape, dtype=jnp.int64)
    np.testing.assert_array_equal(a.hard_count, expected_counts)


def test_zero_length_and_cut_at_or_above_tmax():
    p = jnp.array([0.0, 1.85, 15.0, 40.0, 300.0, 3000.0])
    top = EJ.kinematics(p)[-1]
    zero = S.sample_loss_split(jax.random.key(20), jax.random.key(21), MAT, p, 0.0, top * 0.2,
                                restricted_mean=0.0)
    for value in zero:
        np.testing.assert_array_equal(value, np.zeros(p.shape))
    for cut in (top, top * 2, jnp.full(p.shape, 1e9)):
        out = S.sample_loss_split(jax.random.key(20), jax.random.key(21), MAT, p, 0.02, cut,
                                   restricted_mean=EJ.dedx_restricted(MAT, p, cut) * 0.02)
        np.testing.assert_array_equal(out.hard_requested, np.zeros(p.shape))
        np.testing.assert_array_equal(out.hard_count, np.zeros(p.shape))
        for value in S.hard_moments_per_mm(MAT, p, cut):
            np.testing.assert_array_equal(value, np.zeros(p.shape))
        assert np.isfinite(out.soft_requested).all()
        np.testing.assert_array_equal(
            out.soft_requested,
            EJ.sample_loss(jax.random.key(20), MAT, p, 0.02, cut,
                           mean=EJ.dedx_restricted(MAT, p, cut) * 0.02))


@pytest.mark.parametrize("signal_to_sigma", [0.5, 4.0], ids=["gamma", "gaussian"])
def test_branch_forced_cut_sweep_rejects_only_overlapping_bohr(signal_to_sigma):
    p = 40.0
    _, beta2, _, _, top = map(float, EJ.kinematics(p))
    half = top / 2
    cut = jnp.array([0.1 * top, 0.4 * top, np.nextafter(half, 0), half,
                     np.nextafter(half, np.inf), 0.8 * top,
                     np.nextafter(top, 0), top, np.nextafter(top, np.inf), 1.2 * top])
    effective_cut = jnp.minimum(cut, top)
    # Force the mean criterion everywhere and each inherited Bohr subbranch
    # by constructing length from its actual (not analytically restricted) width.
    mean = 1.2 * E._BOHR_MIN * effective_cut
    variance_per_mm = (E.TWOPI_MC2_RCL2 * MAT.electron_density_per_mm3
                       * (top / beta2 - effective_cut / 2))
    length = (mean / signal_to_sigma) ** 2 / variance_per_mm
    sk, hk = jax.random.key(80), jax.random.key(81)
    out = S.sample_loss_split(sk, hk, MAT, p, length, cut, restricted_mean=mean)
    unsupported = np.array([False, False, False, True, True, True, True, False, False, False])
    np.testing.assert_array_equal(np.isnan(out.soft_requested), unsupported)
    legacy = EJ.sample_loss(sk, MAT, jnp.full(cut.shape, p), length, cut, mean=mean)
    assert np.isfinite(legacy).all()
    np.testing.assert_array_equal(out.soft_requested[~unsupported], legacy[~unsupported])
    # Reject the combined request, not the independently defined hard channel.
    hard_only = S.sample_loss_split(sk, hk, MAT, p, length, cut, restricted_mean=0.0)
    np.testing.assert_array_equal(out.hard_requested, hard_only.hard_requested)
    np.testing.assert_array_equal(out.hard_count, hard_only.hard_count)
    assert np.isfinite(out.hard_requested).all()
    np.testing.assert_array_equal(out.hard_count[cut >= top], 0)
    np.testing.assert_array_equal(out.hard_requested[cut >= top], 0.0)


def test_overlapping_bohr_mean_and_positive_path_boundaries_under_jit():
    p = 40.0
    cut = float(EJ.kinematics(p)[-1]) * 0.75
    threshold = E._BOHR_MIN * cut
    mean = jnp.array([0.0, np.nextafter(threshold, 0), threshold,
                      np.nextafter(threshold, np.inf)])
    length = jnp.array([[0.0], [1e-12], [0.1]])
    # Respect the caller contract: zero path has zero restricted step mean.
    mean = jnp.where(length > 0, mean, 0.0)
    sk, hk = jax.random.key(82), jax.random.key(83)
    sample = jax.jit(lambda path, soft_mean: S.sample_loss_split(
        sk, hk, MAT, p, path, cut, restricted_mean=soft_mean))
    out = sample(length, mean)
    unsupported = np.array([[False, False, False, False],
                            [False, False, True, True],
                            [False, False, True, True]])
    np.testing.assert_array_equal(np.isnan(out.soft_requested), unsupported)
    legacy = EJ.sample_loss(sk, MAT, jnp.full(mean.shape, p), length, cut, mean=mean)
    np.testing.assert_array_equal(out.soft_requested[~unsupported], legacy[~unsupported])
    for value in out:
        np.testing.assert_array_equal(value[0], 0.0)


def test_five_percent_soft_energy_limiter_at_five_mev_stays_admitted():
    kinetic = 5.0
    p = np.sqrt(kinetic * (kinetic + 2 * E.MMU))
    top = float(EJ.kinematics(p)[-1])
    cut = jnp.array([0.4, 0.5, 0.75, 0.9, 1.0, 1.1]) * top
    mean = 0.05 * kinetic
    assert np.all(mean < E._BOHR_MIN * np.minimum(cut, top))
    length = mean / EJ.dedx_restricted(MAT, p, cut)
    sk, hk = jax.random.key(84), jax.random.key(85)
    out = S.sample_loss_split(sk, hk, MAT, p, length, cut, restricted_mean=mean)
    assert np.isfinite(out.soft_requested).all()
    # Scalar/vector compilation can differ by the final floating-point bit.
    np.testing.assert_allclose(
        out.soft_requested,
        EJ.sample_loss(sk, MAT, jnp.full(cut.shape, p), length, cut, mean=mean),
        rtol=1e-14, atol=0)


@pytest.mark.parametrize("p,fraction,lam", [(15.0, 0.15, 0.2), (40.0, 0.4, 3.0), (100.0, 0.2, 12.0),
                                          (300.0, 0.05, 1.5), (3000.0, 0.15, 4.0), (10000.0, 0.6, 2.0)])
def test_compound_poisson_mean_variance_and_count(p, fraction, lam):
    n = 60000
    cut = float(E.kinematics(p)[-1]) * fraction
    rate, mean_rate, variance_rate = map(float, S.hard_moments_per_mm(MAT, p, cut))
    length = lam / rate
    out = S.sample_loss_split(jax.random.key(31), jax.random.key(32), MAT, jnp.full(n, p), length, cut,
                               restricted_mean=0.0)
    counts, energies = np.asarray(out.hard_count), np.asarray(out.hard_requested)
    expected_mean, expected_var = length * mean_rate, length * variance_rate
    kappa4 = length * _quadrature(MAT, p, cut, 4)
    assert abs(counts.mean() - lam) < 6 * np.sqrt(lam / n)
    assert abs(counts.var(ddof=1) - lam) < 6 * np.sqrt((lam + 2 * lam * lam) / n)
    assert abs(energies.mean() - expected_mean) < 6 * np.sqrt(expected_var / n)
    assert abs(energies.var(ddof=1) - expected_var) < 6 * np.sqrt((kappa4 + 2 * expected_var ** 2) / n)
    p0 = np.exp(-lam)
    assert abs(np.mean(counts == 0) - p0) < 6 * np.sqrt(p0 * (1 - p0) / n) + 1 / n
    assert np.all(energies >= counts * cut * (1 - 1e-14))
    assert np.all(energies <= counts * E.kinematics(p)[-1] * (1 + 1e-14))
    assert np.isfinite(energies).all()
    assert np.issubdtype(counts.dtype, np.integer)
    np.testing.assert_array_equal(energies[counts == 0], 0.0)
    np.testing.assert_array_equal(out.soft_requested, 0.0)


def test_fixed_low_cut_relativistic_tail_matches_numpy_sampler():
    n, p, length, cut = 120000, 3000.0, 20.0, 0.0872
    pp = np.full(n, p)
    out = S.sample_loss_split(jax.random.key(40), jax.random.key(41), MAT, pp, length, cut, restricted_mean=0.0)
    idx, transfer = E.sample_deltas(np.random.default_rng(42), MAT, pp, length, cut)
    reference = np.bincount(idx, weights=transfer, minlength=n)
    moments = S.hard_moments_per_mm(MAT, p, cut)
    variance = float(moments.variance) * length
    kappa4 = _quadrature(MAT, p, cut, 4) * length
    for energies in (np.asarray(out.hard_requested), reference):
        assert abs(energies.mean() - float(moments.mean) * length) < 6 * np.sqrt(variance / n)
        assert abs(energies.var(ddof=1) - variance) < 6 * np.sqrt((kappa4 + 2 * variance ** 2) / n)
    assert abs(out.hard_requested.mean() - reference.mean()) < 6 * np.sqrt(2 * variance / n)
    for threshold in (0.1, 0.3, 1.0, 10.0):
        a = np.mean(np.asarray(out.hard_requested) > threshold)
        b = np.mean(reference > threshold)
        assert abs(a - b) < 6 * np.sqrt((a * (1 - a) + b * (1 - b)) / n) + 1 / n


def test_no_hard_count_cap_or_kinetic_clipping():
    p, n, lam = 15.0, 512, 160.0
    kinetic, _, _, _, top = E.kinematics(p)
    # A huge supplied mean must not select the unsupported overlapping Bohr
    # branch: cut < Tmax/2 keeps this an admitted Urban request.
    cut = float(top) * 0.4
    length = lam / float(S.hard_rate_per_mm(MAT, p, cut))
    out = S.sample_loss_split(jax.random.key(50), jax.random.key(51), MAT, jnp.full(n, p), length, cut,
                               restricted_mean=5 * kinetic)
    assert np.min(out.hard_count) > 64
    assert np.all(np.asarray(out.hard_requested) > kinetic)
    assert np.all(np.asarray(out.soft_requested) > kinetic)
    assert abs(np.mean(out.hard_count) - lam) < 6 * np.sqrt(lam / n)
    assert np.all(np.asarray(out.hard_requested) >= np.asarray(out.hard_count) * cut)


def test_scalar_broadcast_material_arrays_and_jit():
    sk, hk = jax.random.key(60), jax.random.key(61)
    scalar = S.sample_loss_split(sk, hk, MAT, 40.0, 0.1, 0.01, restricted_mean=0.1)
    assert all(x.shape == () for x in scalar)
    c = EJ.stack_consts([MAT, M.ALUMINIUM], np.array([[0], [1]]))
    p, length, cut = jnp.array([15.0, 100.0, 3000.0]), jnp.array([[0.0], [0.1]]), 0.01
    mean = EJ.dedx_restricted(c, p, cut) * length
    f = jax.jit(lambda sk, hk, c, p, length, mean: S.sample_loss_split(sk, hk, c, p, length, cut,
                                                                   restricted_mean=mean))
    out = f(sk, hk, c, p, length, mean)
    assert all(x.shape == (2, 3) for x in out)
    for value in out:
        assert np.isfinite(value).all()
        np.testing.assert_array_equal(value[0], 0.0)
    cache_size = f._cache_size()
    f(sk, hk, c, p * 0.9, length * 2, mean * 2)
    assert f._cache_size() == cache_size
    by_mean = S.sample_loss_split(sk, hk, MAT, 100.0, 0.1, cut, restricted_mean=jnp.full((2, 3), 0.2))
    assert all(x.shape == (2, 3) for x in by_mean)
    by_material = S.sample_loss_split(sk, hk, c, 100.0, 0.1, cut, restricted_mean=0.2)
    assert all(x.shape == (2, 1) for x in by_material)
    masses = jnp.array([E.MMU, 938.27208816])
    by_mass = S.sample_loss_split(sk, hk, MAT, 100.0, 0.1, cut, mass=masses, restricted_mean=0.2)
    assert all(x.shape == (2,) for x in by_mass)
    moments = jax.jit(S.hard_moments_per_mm)(c, p, cut)
    assert all(x.shape == (2, 3) for x in moments)
    empty = S.sample_loss_split(sk, hk, MAT, jnp.empty((0,)), 0.1, cut, restricted_mean=0.0)
    assert all(x.shape == (0,) for x in empty)


def test_jit_vmap_matches_independent_scalar_calls():
    keys = jax.random.split(jax.random.key(70), 8)
    sk, hk = keys[:4], keys[4:]
    p = jnp.array([15.0, 40.0, 300.0, 3000.0])
    length = jnp.array([0.0, 0.2, 2.0, 10.0])
    cut = jnp.array([0.01, 0.01, 1.0, 0.0872])
    mean = EJ.dedx_restricted(MAT, p, cut) * length

    def scalar(skey, hkey, momentum, path, threshold, restricted):
        return S.sample_loss_split(skey, hkey, MAT, momentum, path, threshold, restricted_mean=restricted)

    batched = jax.jit(jax.vmap(scalar))(sk, hk, p, length, cut, mean)
    for i in range(4):
        expected = scalar(sk[i], hk[i], p[i], length[i], cut[i], mean[i])
        for actual, reference in zip(batched, expected):
            np.testing.assert_allclose(actual[i], reference, rtol=1e-13, atol=0)
    analytic = jax.jit(jax.vmap(lambda momentum, threshold: S.hard_moments_per_mm(MAT, momentum, threshold)))(p, cut)
    for actual, expected in zip(analytic, S.hard_moments_per_mm(MAT, p, cut)):
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
