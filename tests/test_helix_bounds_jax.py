"""Shared evaluator and sampled boxes, not certification or navigation validation."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')

from mu2efastsim.geometry import helix_bounds_jax as H

K = 0.299792458


def reference(x, u, p, q, B, s):
    """NumPy Rodrigues solution with direct trig and stable integrated sine."""
    bm = np.linalg.norm(B, axis=1)
    b = B / np.where(bm > 0, bm, 1.)[:, None]
    parallel = np.sum(u * b, axis=1)[:, None] * b
    perpendicular, cross = u - parallel, np.cross(u, b)
    omega = K * q * bm / p
    angle = omega[:, None] * s
    velocity = (parallel[:, None, :] + np.cos(angle)[..., None] * perpendicular[:, None, :]
                + np.sin(angle)[..., None] * cross[:, None, :])
    integral_sin = np.divide(2 * np.sin(angle / 2)**2, omega[:, None],
                             out=np.zeros_like(s), where=omega[:, None] != 0)
    position = (x[:, None, :] + s[..., None] * parallel[:, None, :]
                + (s * np.sinc(angle / np.pi))[..., None] * perpendicular[:, None, :]
                + integral_sin[..., None] * cross[:, None, :])
    return position, velocity


def inputs(n=1):
    return dict(x=np.tile([3., 1., -.5], (n, 1)), u=np.tile([.6, 0., .8], (n, 1)),
                p=np.full(n, 3.), q=np.ones(n), B=np.tile([1., -2., 3.], (n, 1)),
                slo=np.zeros(n), shi=np.full(n, 10.), z=np.zeros(n), half=np.full(n, 2.),
                inner=np.full(n, 1.), outer=np.full(n, 5.), axis_x=np.zeros(n), axis_y=np.zeros(n))


def check_samples(data, bounds, count=1025):
    assert np.asarray(bounds.supported).all()
    n = len(data['x'])
    bcast = lambda name: np.broadcast_to(data[name], (n,))
    B = np.zeros((n, 3)) if data['B'] is None else np.broadcast_to(data['B'], (n, 3))
    s = bcast('slo')[:, None] + (bcast('shi') - bcast('slo'))[:, None] * np.linspace(0., 1., count)
    x, u = reference(data['x'], data['u'], bcast('p'), bcast('q'), B, s)
    r = x[..., :2] - np.stack((bcast('axis_x'), bcast('axis_y')), axis=1)[:, None, :]
    r2, dr2 = np.sum(r * r, axis=2), 2 * np.sum(r * u[..., :2], axis=2)
    z, half = bcast('z')[:, None], bcast('half')[:, None]
    f = np.stack((x[..., 2] - z + half, z + half - x[..., 2],
                  bcast('outer')[:, None]**2 - r2, r2 - bcast('inner')[:, None]**2), axis=2)
    df = np.stack((u[..., 2], -u[..., 2], -dr2, dr2), axis=2)
    for sample, lo, hi in ((x, bounds.x_lo, bounds.x_hi), (u, bounds.u_lo, bounds.u_hi),
                            (f, bounds.f_lo, bounds.f_hi), (df, bounds.df_lo, bounds.df_hi)):
        assert np.all(sample >= np.asarray(lo)[:, None, :])
        assert np.all(sample <= np.asarray(hi)[:, None, :])
        assert np.all(np.asarray(lo) <= np.asarray(hi))
    expected_speed = np.broadcast_to(np.linalg.norm(data['u'], axis=1)[:, None], (n, count))
    np.testing.assert_allclose(np.linalg.norm(u, axis=2), expected_speed, atol=2e-15, rtol=0.)


def point_inputs(data, s):
    return {**{key: data[key] for key in ('x', 'u', 'p', 'q', 'B')}, 's': s}


def check_evaluations(data, bounds, compiled, count=17):
    """Direct public evaluator outputs and derived surfaces must fit the boxes."""
    n = len(data['x'])
    bcast = lambda name: np.broadcast_to(data[name], (n,))
    distances = bcast('slo')[:, None] + (bcast('shi') - bcast('slo'))[:, None] * np.linspace(0., 1., count)
    B = np.zeros((n, 3)) if data['B'] is None else np.broadcast_to(data['B'], (n, 3))
    points = dict(x=np.repeat(data['x'], count, axis=0), u=np.repeat(data['u'], count, axis=0),
                  p=np.repeat(bcast('p'), count), q=np.repeat(bcast('q'), count),
                  B=np.repeat(B, count, axis=0), s=distances.ravel())
    result = (jax.jit(H.evaluate) if compiled else H.evaluate)(**points)
    assert result.supported.shape == (n * count,)
    assert result.x.shape == result.u.shape == (n * count, 3)
    assert np.asarray(result.supported).all()
    x, u = np.asarray(result.x).reshape(n, count, 3), np.asarray(result.u).reshape(n, count, 3)
    expected_x, expected_u = reference(data['x'], data['u'], bcast('p'), bcast('q'), B, distances)
    np.testing.assert_allclose(x, expected_x, atol=5e-8, rtol=3e-12)
    np.testing.assert_allclose(u, expected_u, atol=3e-12, rtol=3e-12)
    r = x[..., :2] - np.stack((bcast('axis_x'), bcast('axis_y')), axis=1)[:, None, :]
    r2, dr2 = np.sum(r * r, axis=2), 2 * np.sum(r * u[..., :2], axis=2)
    z, half = bcast('z')[:, None], bcast('half')[:, None]
    f = np.stack((x[..., 2] - z + half, z + half - x[..., 2],
                  bcast('outer')[:, None]**2 - r2, r2 - bcast('inner')[:, None]**2), axis=2)
    df = np.stack((u[..., 2], -u[..., 2], -dr2, dr2), axis=2)
    for samples, lo, hi in ((x, bounds.x_lo, bounds.x_hi), (u, bounds.u_lo, bounds.u_hi),
                             (f, bounds.f_lo, bounds.f_hi), (df, bounds.df_lo, bounds.df_hi)):
        assert np.all(samples >= np.asarray(lo)[:, None, :])
        assert np.all(samples <= np.asarray(hi)[:, None, :])


def test_random_tilted_both_charges_translations_batch_and_jit():
    rng = np.random.default_rng(713)
    n = 128
    data = inputs(n)
    data['x'] = rng.uniform(-1e5, 1e5, (n, 3))
    data['u'] = rng.normal(size=(n, 3))
    data['u'] /= np.linalg.norm(data['u'], axis=1)[:, None]
    data['B'] = rng.uniform(-5., 5., (n, 3))
    data['q'] = rng.choice([-1., 1.], n)
    data['p'] = rng.uniform(.1, 100., n)
    data['slo'] = rng.uniform(0., 100., n)
    data['shi'] = data['slo'] + rng.uniform(0., 100., n)
    data['axis_x'], data['axis_y'], data['z'] = data['x'].T
    eager, compiled = H.enclose(**data), jax.jit(H.enclose)(**data)
    check_samples(data, eager)
    check_samples(data, compiled)
    for bounds in (eager, compiled):
        for use_jit in (False, True):
            check_evaluations(data, bounds, use_jit)
    for a, b in zip(eager[:-1], compiled[:-1]):
        np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-10)
    again = jax.jit(H.enclose)(**data)
    for a, b in zip(compiled, again):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('field,charge', [(None, 1.), ([0., 0., 0.], -1.),
                                          ([1e-90, -2e-90, 3e-90], 1.),
                                          ([1., -2., 3.], 0.), ([0., 0., 3.], -1.)])
def test_straight_tiny_zero_neutral_parallel(field, charge):
    data = inputs()
    data.update(B=field, q=charge, u=np.array([[0., 0., 1.]]), slo=2., shi=9.)
    check_samples(data, jax.jit(H.enclose)(**data))


@pytest.mark.parametrize('start,stop', [(0., 0.), (7., 7.), (0., np.pi),
                                       (0., 2 * np.pi), (np.pi, 4 * np.pi)])
@pytest.mark.parametrize('charge', [-1., 1.])
def test_zero_half_full_turn_intervals(start, stop, charge):
    data = inputs()
    data.update(x=np.zeros((1, 3)), u=np.array([[1., 0., 0.]]), p=K, q=charge,
                B=[0., 0., 1.], slo=start, shi=stop, inner=0., outer=1.)
    result = jax.jit(H.enclose)(**data)
    check_samples(data, result)
    check_evaluations(data, result, compiled=False)
    check_evaluations(data, result, compiled=True)
    if start == 0. and stop == 2 * np.pi:
        x, _ = reference(data['x'], data['u'], np.array([K]), np.array([charge]),
                         np.array([[0., 0., 1.]]), np.array([[0., np.pi, 2 * np.pi]]))
        np.testing.assert_allclose(x[:, 0], x[:, -1], atol=1e-15)
        assert np.sum(x[0, 1, :2]**2) > data['outer']**2
        assert result.f_lo[0, 2] < 0 < result.f_hi[0, 2]


@pytest.mark.parametrize('key,value', [
    ('p', 0.), ('p', -1.), ('p', 1e-7), ('p', 1e10), ('p', np.nan),
    ('q', np.inf), ('q', 101.), ('q', 1e-200), ('B', [1e-200, 0., 0.]),
    ('B', [1e5, 0., 0.]), ('B', [np.nan, 0., 0.]), ('shi', np.inf),
    ('shi', 1e9), ('slo', -1.), ('slo', 11.), ('half', 0.), ('inner', -1.),
    ('outer', 1.), ('axis_x', 1e9), ('p', 'bad'), ('q', 1j),
    ('p', np.float32(3.)), ('B', np.ones(3, dtype=np.float32)),
    ('x', np.ones((1, 3), dtype=np.float32)), ('u', np.ones((1, 3))),
    ('x', np.full((1, 3), 1e100)), ('u', np.array([['bad', 'bad', 'bad']])),
    ('shi', 1e7), ('q', np.nextafter(0., 1.)),
    ('B', [np.nextafter(0., 1.), 0., 0.]),
])
def test_explicit_unsupported(key, value):
    data = inputs()
    data[key] = value
    result = H.enclose(**data)
    assert not np.asarray(result.supported).any()
    for array in result[:-1]:
        assert np.isnan(array).all()
    if key in ('x', 'u', 'p', 'q', 'B', 'slo', 'shi') and not (key == 'slo' and value == 11.):
        point = H.evaluate(**point_inputs(data, data[key] if key in ('slo', 'shi') else data['shi']))
        assert not np.asarray(point.supported).any()
        assert np.isnan(point.x).all() and np.isnan(point.u).all()


def test_invalid_lane_does_not_contaminate_valid_lane():
    data = inputs(3)
    data['p'][1] = np.nan
    data['u'][2] *= 2.
    result = jax.jit(H.enclose)(**data)
    np.testing.assert_array_equal(result.supported, [True, False, False])
    for array in result[:-1]:
        assert np.isfinite(array[0]).all() and np.isnan(array[1:]).all()


def test_near_domain_limits_singletons_and_long_intervals():
    data = inputs(6)
    data.update(x=np.tile([1e8, -1e8, 1e8], (6, 1)), p=np.ones(6),
                B=np.tile([0., 0., 1e-4 / K], (6, 1)), q=np.array([1., -1., 1., -1., 1., -1.]),
                slo=np.array([0., 1e8, 1e8 - 1., 0., 1e-10, 1e4]),
                shi=np.array([0., 1e8, 1e8, 1e8, 1e-10, 1e4 + 1e-8]),
                axis_x=1e8, axis_y=-1e8, z=1e8)
    bounds = jax.jit(H.enclose)(**data)
    check_samples(data, bounds)
    check_evaluations(data, bounds, compiled=False)
    check_evaluations(data, bounds, compiled=True)


def test_shape_errors_are_programming_errors():
    data = inputs()
    data['x'] = np.zeros(3)
    with pytest.raises(ValueError, match='identical'):
        H.enclose(**data)
    with pytest.raises(ValueError, match='identical'):
        H.evaluate(**point_inputs(data, 1.))
    data = inputs()
    data['B'] = np.zeros((2, 3))
    with pytest.raises(ValueError):
        H.enclose(**data)
    with pytest.raises(ValueError):
        H.evaluate(**point_inputs(data, 1.))


def test_x64_disabled_is_explicitly_unsupported():
    previous = jax.config.x64_enabled
    try:
        jax.config.update('jax_enable_x64', False)
        result = H.enclose(**inputs())
        assert not result.supported.any()
        assert all(np.isnan(a).all() for a in result[:-1])
        for evaluate in (H.evaluate, jax.jit(H.evaluate)):
            point = evaluate(**point_inputs(inputs(), 1.))
            assert not point.supported.any()
            assert np.isnan(point.x).all() and np.isnan(point.u).all()
    finally:
        jax.config.update('jax_enable_x64', previous)


def test_random_tilted_singletons_and_narrow_intervals():
    rng = np.random.default_rng(714)
    data = inputs(64)
    data['u'] = rng.normal(size=(64, 3))
    data['u'] /= np.linalg.norm(data['u'], axis=1)[:, None]
    data['B'] = rng.uniform(-5., 5., (64, 3))
    data['q'] = rng.choice([-1., 1.], 64)
    data['slo'] = rng.uniform(0., 1e3, 64)
    data['shi'] = data['slo'] + rng.choice([0., 1e-10, 1e-7], 64)
    check_samples(data, jax.jit(H.enclose)(**data), count=17)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('charge', [-1., 1.])
@pytest.mark.parametrize('phase', [0.999999e-6, 1e-6, 1.000001e-6, 0.999999e-4, 1.000001e-4])
def test_small_phase_singleton_uses_analytic_curve(phase, charge, compiled):
    data = inputs()
    data.update(x=np.zeros((1, 3)), u=np.array([[1., 0., 0.]]),
                p=K, q=charge, B=np.array([0., 0., phase]), slo=1., shi=1.)
    bounds = (jax.jit(H.enclose) if compiled else H.enclose)(**data)
    check_samples(data, bounds, count=1)
    check_evaluations(data, bounds, compiled=False, count=1)
    check_evaluations(data, bounds, compiled=True, count=1)
    point = (jax.jit(H.evaluate) if compiled else H.evaluate)(**point_inputs(data, 1.))
    np.testing.assert_allclose(point.x[0, 1], -charge * 2 * np.sin(phase / 2)**2 / phase,
                               atol=1e-20, rtol=2e-15)
    data.update(slo=.5, shi=1.5)
    bounds = (jax.jit(H.enclose) if compiled else H.enclose)(**data)
    check_evaluations(data, bounds, compiled)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('offset', [-9e-13, 9e-13])
def test_singleton_derivative_is_not_renormalized(offset, compiled):
    data = inputs()
    data.update(u=np.array([[1. + offset, 0., 0.]]), B=None, slo=0., shi=0.)
    bounds = (jax.jit(H.enclose) if compiled else H.enclose)(**data)
    check_samples(data, bounds, count=1)
    check_evaluations(data, bounds, compiled, count=1)
    point = (jax.jit(H.evaluate) if compiled else H.evaluate)(**point_inputs(data, 0.))
    np.testing.assert_array_equal(point.u, data['u'])
    assert np.linalg.norm(point.u) != 1.


def test_parallel_tilted_and_near_unit_input_not_renormalized():
    data = inputs(2)
    data['u'] = data['B'] / np.linalg.norm(data['B'], axis=1)[:, None]
    data['u'][1] *= 1 + 5e-13
    data['slo'], data['shi'] = 100., 100.
    bounds = jax.jit(H.enclose)(**data)
    assert np.all(bounds.supported)
    position = data['x'] + 100 * data['u']
    assert np.all(position >= bounds.x_lo) and np.all(position <= bounds.x_hi)
    assert np.all(data['u'] >= bounds.u_lo) and np.all(data['u'] <= bounds.u_hi)
    check_evaluations(data, bounds, compiled=False, count=1)
    check_evaluations(data, bounds, compiled=True, count=1)


@pytest.mark.parametrize('compiled', [False, True])
def test_direct_evaluator_nonunit_tilted_singleton_narrow_and_turn_intervals(compiled):
    rng = np.random.default_rng(715)
    data = inputs(48)
    data['u'] = rng.normal(size=(48, 3))
    data['u'] /= np.linalg.norm(data['u'], axis=1)[:, None]
    data['u'] *= rng.choice([1 - 9e-13, 1 + 9e-13], (48, 1))
    data['B'] = rng.uniform(-5., 5., (48, 3))
    data['q'] = rng.choice([-1., 1.], 48)
    data['slo'] = rng.uniform(0., 100., 48)
    data['shi'] = data['slo'] + rng.choice([0., 1e-10, 30.], 48)
    bounds = (jax.jit(H.enclose) if compiled else H.enclose)(**data)
    check_samples(data, bounds, count=33)
    check_evaluations(data, bounds, compiled, count=33)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('field,charge', [(None, 1.), ([0., -0., 0.], -1.),
                                          ([1e-90, -2e-90, 3e-90], -1.),
                                          ([1., -2., 3.], 0.), ([1., -2., 3.], -0.)])
def test_direct_evaluator_zero_and_tiny_field_or_charge(compiled, field, charge):
    data = inputs(2)
    data.update(B=field, q=charge, slo=np.array([7., 2.]), shi=np.array([7., 9.]))
    bounds = (jax.jit(H.enclose) if compiled else H.enclose)(**data)
    check_evaluations(data, bounds, compiled)
    point = (jax.jit(H.evaluate) if compiled else H.evaluate)(**point_inputs(data, data['shi']))
    assert point.supported.all()
    np.testing.assert_allclose(point.x, data['x'] + data['shi'][:, None] * data['u'], atol=2e-15, rtol=0.)
    np.testing.assert_allclose(point.u, data['u'], atol=2e-15, rtol=0.)


@pytest.mark.parametrize('compiled', [False, True])
def test_shared_domain_subnormal_zero_and_invalid_lanes(compiled):
    tiny = np.nextafter(0., 1.)
    data = inputs(16)
    data['q'][:4] = [0., -0., tiny, -tiny]
    data['B'][4:9] = [[tiny, 0., 0.], [-tiny, 0., 0.], [0., -0., 0.],
                      [1e-100, 0., 0.], [0., 0., 1e4]]
    data['p'][9] = np.nan
    data['u'][10] *= 2.
    data['slo'] = data['shi'] = np.array([1.] * 11 + [-1., 1e9, 1., 1., 1.])
    data['q'][13:] = [1e-100, 0., tiny]
    data['B'][14:] = [[tiny, 0., 0.], [0., 0., 0.]]
    enclose = jax.jit(H.enclose) if compiled else H.enclose
    evaluate = jax.jit(H.evaluate) if compiled else H.evaluate
    bounds = enclose(**data)
    point = evaluate(**point_inputs(data, data['shi']))
    expected = [True, True, False, False, False, False, True, True, True,
                False, False, False, False, True, False, False]
    for result in (bounds, point):
        np.testing.assert_array_equal(result.supported, expected)
        for a in result[:-1]:
            assert np.isfinite(np.asarray(a)[expected]).all()
            assert np.isnan(np.asarray(a)[np.logical_not(expected)]).all()
    for key, value in [('p', np.float32(3.)), ('q', True), ('s', np.inf), ('s', 1e7)]:
        args = point_inputs(inputs(), 1.)
        args[key] = value
        result = evaluate(**args)
        assert not result.supported.any()
        assert np.isnan(result.x).all() and np.isnan(result.u).all()


def test_geometry_and_interval_validation_are_enclosure_only():
    data = inputs()
    data.update(slo=11., shi=10., half=-1.)
    assert not H.enclose(**data).supported.any()
    assert H.evaluate(**point_inputs(data, 11.)).supported.all()


def test_evaluator_integer_broadcast_and_jit_repeatability():
    data = inputs(3)
    args = point_inputs(data, np.array([0, 1, 2]))
    args.update(p=3, q=-1, B=np.array([0, 0, 1]))
    eager = H.evaluate(**args)
    evaluate = jax.jit(H.evaluate)
    compiled = evaluate(**args)
    assert eager.supported.all() and compiled.supported.all()
    for a, b in zip(eager[:-1], compiled[:-1]):
        np.testing.assert_allclose(a, b, atol=2e-15, rtol=2e-15)
    for a, b in zip(compiled, evaluate(**args)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('key', ['slo', 'shi', 's', 'z', 'half', 'inner', 'outer', 'axis_x', 'axis_y'])
def test_distance_and_geometry_subnormal_rejection_signed_zeros_and_normal_lanes(key, compiled):
    tiny = np.nextafter(0., 1.)
    normal = np.finfo(np.float64).smallest_normal
    largest_subnormal = np.nextafter(normal, 0.)
    values = np.array([tiny, -tiny, largest_subnormal, -largest_subnormal,
                       0., -0., normal, -normal, 1.])
    data = inputs(len(values))
    data['inner'] = 0.
    enclose = jax.jit(H.enclose) if compiled else H.enclose
    evaluate = jax.jit(H.evaluate) if compiled else H.evaluate
    expected = [False, False, False, False, True, True, True, False, True]
    if key in ('z', 'axis_x', 'axis_y'):
        expected[7] = True
    if key in ('half', 'outer'):
        expected[4:6] = [False, False]
    if key == 's':
        results = [evaluate(**point_inputs(data, values))]
    else:
        data[key] = values
        results = [enclose(**data)]
        if key in ('slo', 'shi'):
            results.append(evaluate(**point_inputs(data, values)))
    for result in results:
        np.testing.assert_array_equal(result.supported, expected)
        for array in result[:-1]:
            assert np.isfinite(np.asarray(array)[expected]).all()
            assert np.isnan(np.asarray(array)[np.logical_not(expected)]).all()


@pytest.mark.parametrize('compiled', [False, True])
def test_distance_and_geometry_integer_and_float_dtype_contract(compiled):
    data = inputs(2)
    data.update(slo=0, shi=1, z=0, half=2, inner=0, outer=5, axis_x=0, axis_y=0)
    enclose = jax.jit(H.enclose) if compiled else H.enclose
    evaluate = jax.jit(H.evaluate) if compiled else H.evaluate
    assert enclose(**data).supported.all()
    assert evaluate(**point_inputs(data, 1)).supported.all()
    for key in ('slo', 'shi', 'z', 'half', 'inner', 'outer', 'axis_x', 'axis_y'):
        result = enclose(**{**data, key: np.float32(data[key])})
        assert not result.supported.any()
        assert all(np.isnan(a).all() for a in result[:-1])
    point = evaluate(**point_inputs(data, np.float32(1.)))
    assert not point.supported.any()
    assert np.isnan(point.x).all() and np.isnan(point.u).all()


@pytest.mark.parametrize('compiled', [False, True])
def test_singleton_axial_padding_floor_is_unresolved_by_subdivision(compiled):
    data = inputs()
    data.update(x=np.zeros((1, 3)), u=np.array([[1., 0., 0.]]), p=1., q=1.,
                B=np.array([0., 0., 1e-4 / H.K_FIELD]), slo=1e8, shi=1e8,
                z=0., half=1e-4, inner=0., outer=1e8)
    enclose = jax.jit(H.enclose) if compiled else H.enclose
    evaluate = jax.jit(H.evaluate) if compiled else H.evaluate
    bounds = enclose(**data)
    point = evaluate(**point_inputs(data, data['slo']))
    assert bounds.supported.all() and point.supported.all()
    np.testing.assert_array_equal(point.x[:, 2], [0.])
    true_faces = np.stack((point.x[:, 2] - data['z'] + data['half'],
                           data['z'] + data['half'] - point.x[:, 2]), axis=1)
    np.testing.assert_array_equal(true_faces, [[1e-4, 1e-4]])
    assert np.all(bounds.f_lo[:, :2] < 0.)
    assert np.all(bounds.f_hi[:, :2] > true_faces)
    midpoint = data['slo'] + (data['shi'] - data['slo']) * .5
    for lo, hi in ((data['slo'], midpoint), (midpoint, data['shi'])):
        assert lo == hi == 1e8
        child = enclose(**{**data, 'slo': lo, 'shi': hi})
        assert child.supported.all()
        np.testing.assert_array_equal(child.f_lo, bounds.f_lo)
        np.testing.assert_array_equal(child.f_hi, bounds.f_hi)
