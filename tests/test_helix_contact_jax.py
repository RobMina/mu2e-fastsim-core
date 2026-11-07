"""Development surface isolation regressions, not certified navigation validation."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')

from mu2efastsim.geometry import helix_bounds_jax as HB
from mu2efastsim.geometry import helix_contact_jax as HC

K = 0.299792458
COMPILED = jax.jit(HC.isolate, static_argnames=('max_work', 'max_depth'))


def inputs(n=1):
    return dict(x=np.tile([3., 0., 0.], (n, 1)), u=np.tile([1., 0., 0.], (n, 1)),
                p=np.ones(n), q=np.ones(n), B=np.zeros((n, 3)), slo=np.zeros(n),
                shi=np.full(n, 7.), z=np.zeros(n), half=np.full(n, 2.),
                inner=np.ones(n), outer=np.full(n, 5.), axis_x=np.zeros(n), axis_y=np.zeros(n))


def assert_bracket(result, expected, face, atol=1e-6):
    assert np.all(np.isin(result.status, [HC.CANDIDATE, HC.TRANSVERSE_BRACKET]))
    assert np.all(np.asarray(result.slo) <= expected)
    assert np.all(np.asarray(result.shi) >= expected)
    assert np.all(np.asarray(result.shi) - np.asarray(result.slo) <= atol)
    assert np.all(np.asarray(result.surfaces)[:, face])
    assert not np.isinf(result.slo).any() and not np.isinf(result.shi).any()


def assert_prefix_excluded(data, result):
    # Replay the dyadic left siblings with HB, not endpoint equality or sampling.
    for lane, stop in enumerate(np.asarray(result.slo)):
        one = {key: np.asarray(value)[lane:lane + 1] for key, value in data.items()}
        left, right = float(one['slo'][0]), float(one['shi'][0])
        for _ in range(65):
            if left == stop:
                break
            mid = left + (right - left) * .5
            if stop >= mid:
                bounds = HB.enclose(**{**one, 'slo': left, 'shi': mid})
                enabled = np.array([True, True, True, one['inner'][0] > 0])
                assert bounds.supported[0]
                assert not np.any(enabled & (bounds.f_lo[0] <= 0) & (bounds.f_hi[0] >= 0))
                left = mid
            else:
                right = mid
        assert left == stop


@pytest.mark.parametrize('compiled', [False, True])
def test_analytic_straight_faces_hole_and_outer(compiled):
    data = inputs(4)
    data['u'][:] = [[0., 0., -1.], [0., 0., 1.], [1., 0., 0.], [-1., 0., 0.]]
    data['shi'][:] = 7.1
    result = (COMPILED if compiled else HC.isolate)(**data)
    assert np.all(result.status == HC.TRANSVERSE_BRACKET)
    assert np.all(result.slo < 2.) and np.all(result.shi > 2.)
    np.testing.assert_array_equal(result.surfaces, np.eye(4, dtype=bool))
    np.testing.assert_array_equal(result.monotone, result.surfaces)
    assert_prefix_excluded(data, result)


@pytest.mark.parametrize('compiled', [False, True])
def test_both_charges_full_turn_equal_endpoints_leave_return(compiled):
    data = inputs(2)
    data.update(x=np.zeros((2, 3)), u=np.tile([1., 0., 0.], (2, 1)), p=np.full(2, K),
                q=np.array([-1., 1.]), B=np.tile([0., 0., 1.], (2, 1)),
                shi=np.full(2, 2 * np.pi), inner=np.zeros(2), outer=np.ones(2))
    point = HB.evaluate(**{key: data[key] for key in ('x', 'u', 'p', 'q', 'B')}, s=data['shi'])
    np.testing.assert_allclose(point.x, data['x'], atol=1e-15)
    result = (COMPILED if compiled else HC.isolate)(**data)
    assert_bracket(result, np.pi / 3, 2)
    assert np.all(result.status == HC.TRANSVERSE_BRACKET)
    assert_prefix_excluded(data, result)


@pytest.mark.parametrize('compiled', [False, True])
def test_tilted_field_both_charges_analytic_face_and_parallel_motion(compiled):
    data = inputs(4)
    field = np.array([1., 0., 1.]) / np.sqrt(2.)
    data.update(x=np.zeros((4, 3)), u=np.tile([0., 1., 0.], (4, 1)), p=np.full(4, K),
                q=np.array([-1., 1., -1., 1.]), B=np.tile(field, (4, 1)),
                shi=np.full(4, 2.7), half=np.full(4, .4), inner=np.zeros(4), outer=np.full(4, 10.))
    data['u'][2:] = field
    result = (COMPILED if compiled else HC.isolate)(**data)
    expected = np.array([np.arccos(1 - .4 * np.sqrt(2.))] * 2 + [.4 * np.sqrt(2.)] * 2)
    assert np.all(result.status == HC.TRANSVERSE_BRACKET)
    assert np.all(result.slo < expected) and np.all(result.shi > expected)
    np.testing.assert_array_equal(result.surfaces, np.eye(4, dtype=bool)[[1, 0, 1, 1]])
    assert_prefix_excluded(data, result)


def test_nonzero_search_start_and_translated_axes():
    data = inputs()
    data.update(x=np.array([[1003., -200., 30.]]), z=np.array([30.]), axis_x=np.array([1000.]),
                axis_y=np.array([-200.]), slo=np.array([.3]), shi=np.array([6.2]))
    result = COMPILED(**data)
    assert_bracket(result, 2., 2)
    assert_prefix_excluded(data, result)


def test_zero_inner_disables_algebraic_axis_zero():
    data = inputs(2)
    data.update(x=np.array([[0., 0., 0.], [-1., 0., 0.]]), inner=np.zeros(2),
                outer=np.full(2, 10.), shi=np.full(2, 2.))
    result = COMPILED(**data)
    np.testing.assert_array_equal(result.status, [HC.NO_CANDIDATE, HC.NO_CANDIDATE])
    assert np.isnan(result.slo).all() and np.isnan(result.shi).all()
    assert not result.surfaces.any()


def test_surface_candidates_are_not_volume_entry_or_exit():
    data = inputs()
    data['x'][:] = [20., 0., -3.]
    data['u'][:] = [0., 0., 1.]
    result = COMPILED(**data)
    assert_bracket(result, 1., 0)
    assert data['x'][0, 0] > data['outer'][0]


@pytest.mark.parametrize('compiled', [False, True])
def test_tangent_and_near_tangent_are_not_transverse_or_clear(compiled):
    data = inputs(3)
    data.update(x=np.array([[-3., 1., 0.], [-3., 1. + 1e-14, 0.], [-3., 1. - 1e-14, 0.]]),
                outer=np.full(3, 5.), shi=np.full(3, 6.3))
    result = (COMPILED if compiled else HC.isolate)(**data)
    assert np.all(result.status == HC.CANDIDATE)
    assert np.all(result.surfaces[:, 3])
    assert np.all(result.slo < 3.)
    assert np.all(3. - np.asarray(result.shi) < 1e-4)
    assert_prefix_excluded(data, result)


def test_tangent_blocks_later_simple_root():
    data = inputs()
    data.update(x=np.array([[-3., 1., 0.]]), shi=np.array([10.]))
    result = COMPILED(**data)
    assert result.status[0] == HC.CANDIDATE
    assert result.shi[0] < 3. + np.sqrt(24.)
    assert result.surfaces[0, 3] and not result.surfaces[0, 2]


@pytest.mark.parametrize('compiled', [False, True])
def test_start_surface_and_padding_uncertainty_are_explicit(compiled):
    data = inputs(5)
    data['x'][:] = [[5., 0., 0.], [5. - 1e-14, 0., 0.], [1., 0., 0.],
                   [3., 0., 2.], [3., 0., -2.]]
    data['u'][:] = [[-1., 0., 0.], [1., 0., 0.], [0., 0., 1.],
                   [1., 0., 0.], [0., 0., 1.]]
    result = (COMPILED if compiled else HC.isolate)(**data)
    assert np.all(result.status == HC.START_UNRESOLVED)
    assert np.all(result.slo == 0.) and np.all(result.shi == 0.) and np.all(result.work == 0)
    assert np.all(result.surfaces.any(axis=1))


def test_too_small_tolerance_and_work_depth_exhaustion():
    data = inputs()
    data['shi'][:] = 7.1
    precision = COMPILED(**data, atol=1e-30, max_depth=64)
    assert precision.status[0] == HC.PRECISION_UNRESOLVED
    assert np.isfinite(precision.slo).all() and np.isfinite(precision.shi).all()
    # Near the padding floor, separate eager/JIT box programs can differ by ulps.
    assert precision.shi[0] <= 2.
    for limits in ({'max_work': 0}, {'max_work': 1}, {'max_work': 5}, {'max_depth': 0}, {'max_depth': 2}):
        result = COMPILED(**data, **limits)
        assert result.status[0] == HC.WORK_LIMIT_UNRESOLVED
        assert np.isfinite(result.slo).all() and np.isfinite(result.shi).all()
        assert result.work[0] <= limits.get('max_work', 512)
        assert_prefix_excluded(data, result)


def test_mixed_lanes_eager_jit_repeatability_and_independence():
    data = inputs(6)
    data['shi'][:] = [7.1, .25, 7., 7., 7., 7.1]
    data['x'][2, 0] = 5.
    data['p'][3] = np.nan
    data['x'][4] = [-3., 1., 0.]
    atol = np.array([1e-6] * 5 + [1e-30])
    eager = HC.isolate(**data, atol=atol, max_depth=64)
    compiled = COMPILED(**data, atol=atol, max_depth=64)
    expected = [HC.TRANSVERSE_BRACKET, HC.NO_CANDIDATE, HC.START_UNRESOLVED,
                HC.UNSUPPORTED, HC.CANDIDATE, HC.PRECISION_UNRESOLVED]
    np.testing.assert_array_equal(compiled.status, expected)
    for a, b in zip(eager, compiled):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(compiled, COMPILED(**data, atol=atol, max_depth=64)):
        np.testing.assert_array_equal(a, b)
    for lane in range(6):
        one = {key: value[lane:lane + 1] for key, value in data.items()}
        single = COMPILED(**one, atol=atol[lane:lane + 1], max_depth=64)
        for a, b in zip(single, compiled):
            np.testing.assert_array_equal(a[0], b[lane])


@pytest.mark.parametrize('key,value', [('p', 0.), ('q', 101.), ('shi', np.inf), ('slo', -1.),
                                       ('slo', 8.), ('half', 0.), ('inner', -1.), ('outer', 1.),
                                       ('p', np.float32(1.)), ('atol', 0.), ('atol', np.nan),
                                       ('atol', np.float32(1e-6)), ('B', [1e-200, 0., 0.])])
def test_unsupported_is_not_no_candidate(key, value):
    data = inputs()
    data[key] = value
    result = COMPILED(**data)
    assert result.status[0] == HC.UNSUPPORTED
    assert np.isnan(result.slo).all() and np.isnan(result.shi).all()
    assert not result.surfaces.any() and not result.monotone.any()


def test_singleton_endpoint_and_simultaneous_surfaces():
    data = inputs()
    data['shi'][:] = 0.
    assert COMPILED(**data).status[0] == HC.NO_CANDIDATE
    data['shi'][:] = 2.
    result = COMPILED(**data)
    assert_bracket(result, 2., 2)
    assert result.status[0] == HC.CANDIDATE
    data['u'][:] = [1 / np.sqrt(2.), 0., 1 / np.sqrt(2.)]
    data['shi'][:] = 5.
    result = COMPILED(**data)
    assert_bracket(result, 2 * np.sqrt(2.), 1)
    assert result.surfaces[0, 2] and result.status[0] == HC.CANDIDATE


def test_static_limits_shapes_and_empty_lanes():
    for limits in ({'max_work': -1}, {'max_work': 1.5}, {'max_depth': -1}, {'max_depth': 65}):
        with pytest.raises(ValueError, match='static'):
            HC.isolate(**inputs(), **limits)
    data = inputs()
    data['x'] = np.zeros(3)
    with pytest.raises(ValueError, match='identical'):
        HC.isolate(**data)
    result = COMPILED(**inputs(0))
    assert result.status.shape == result.slo.shape == result.shi.shape == (0,)
    assert result.surfaces.shape == result.monotone.shape == (0, 4)


def test_curved_tangent_and_no_candidate_under_engineering_boxes():
    data = inputs(4)
    data.update(x=np.zeros((4, 3)), u=np.tile([1., 0., 0.], (4, 1)), p=np.full(4, K),
                q=np.array([-1., 1., -1., 1.]), B=np.tile([0., 0., 1.], (4, 1)),
                shi=np.full(4, 2 * np.pi), inner=np.zeros(4), outer=np.array([2., 2., 3., 3.]))
    result = COMPILED(**data)
    np.testing.assert_array_equal(result.status, [HC.CANDIDATE, HC.CANDIDATE,
                                                  HC.NO_CANDIDATE, HC.NO_CANDIDATE])
    assert np.all(result.surfaces[:2, 2])
    assert np.all(result.slo[:2] < np.pi)
    assert np.all(np.pi - np.asarray(result.shi[:2]) < 1e-4)
    assert np.isnan(result.slo[2:]).all() and np.isnan(result.shi[2:]).all()


def test_mixed_work_budget_and_neutral_tilted_field():
    data = inputs(5)
    data['B'][:] = [1., 2., 3.]
    data['q'][:] = 0.
    data['slo'][0], data['shi'][0] = 2. - 2e-7, 2. + 2e-7
    data['shi'][1] = .1
    data['x'][2, 0] = 5.
    data['p'][3] = 0.
    result = COMPILED(**data, max_work=1)
    np.testing.assert_array_equal(result.status, [HC.TRANSVERSE_BRACKET, HC.NO_CANDIDATE,
                                                  HC.START_UNRESOLVED, HC.UNSUPPORTED,
                                                  HC.WORK_LIMIT_UNRESOLVED])
    np.testing.assert_array_equal(result.work, [1, 1, 0, 0, 1])
    assert result.slo[4] == 0. and result.shi[4] == 3.5
    for a, b in zip(result, HC.isolate(**data, max_work=1)):
        np.testing.assert_array_equal(a, b)


def test_eager_nonnumeric_rejection_and_disabled_x64():
    for key in ('atol', 'p', 'half'):
        result = HC.isolate(**{**inputs(), key: 'bad'})
        assert result.status[0] == HC.UNSUPPORTED
        assert np.isnan(result.slo).all() and not result.surfaces.any()
    previous = jax.config.x64_enabled
    try:
        jax.config.update('jax_enable_x64', False)
        result = HC.isolate(**inputs())
        assert result.status[0] == HC.UNSUPPORTED
        assert np.isnan(result.slo).all() and not result.surfaces.any()
    finally:
        jax.config.update('jax_enable_x64', previous)
