"""Development volume search regressions, not certified navigation validation."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')

from mu2efastsim.geometry import helix_bounds_jax as HB
from mu2efastsim.geometry import helix_contact_jax as HC
from mu2efastsim.geometry import helix_volume_jax as HV

K = 0.299792458
COMPILED = jax.jit(HV.isolate, static_argnames=('max_work', 'max_depth'))


def inputs(n=1):
    return dict(x=np.tile([3., 0., 0.], (n, 1)), u=np.tile([1., 0., 0.], (n, 1)),
                p=np.ones(n), q=np.ones(n), B=np.zeros((n, 3)), slo=np.zeros(n),
                shi=np.full(n, 7.1), z=np.zeros(n), half=np.full(n, 2.),
                inner=np.ones(n), outer=np.full(n, 5.), axis_x=np.zeros(n), axis_y=np.zeros(n))


def assert_bracket(result, expected, faces, status):
    np.testing.assert_array_equal(result.status, status)
    assert np.all(np.asarray(result.slo) <= expected)
    assert np.all(np.asarray(result.shi) >= expected)
    assert np.all(np.asarray(result.shi) - np.asarray(result.slo) <= 1e-6)
    np.testing.assert_array_equal(result.surfaces, np.eye(4, dtype=bool)[faces])
    assert not np.isinf(result.slo).any() and not np.isinf(result.shi).any()


def assert_prefix_excluded(data, result):
    # Replay chronological dyadic siblings using strict volume boxes, never samples.
    for lane, stop in enumerate(np.asarray(result.slo)):
        one = {key: np.asarray(value)[lane:lane + 1] for key, value in data.items()}
        left, right = float(one['slo'][0]), float(one['shi'][0])
        for _ in range(65):
            if left == stop:
                break
            mid = left + (right - left) * .5
            if stop >= mid:
                volume = HV.enclose(**{**one, 'slo': left, 'shi': mid})
                assert volume.region[0] in (HV.INTERIOR, HV.EXTERIOR)
                assert not volume.surfaces.any()
                left = mid
            else:
                right = mid
        assert left == stop


@pytest.mark.parametrize('compiled', [False, True])
def test_analytic_entry_exit_all_surfaces(compiled):
    data = inputs(8)
    data['x'][4:] = [[3., 0., -3.], [3., 0., 3.], [7., 0., 0.], [0., 0., 0.]]
    data['u'][:] = [[0., 0., -1.], [0., 0., 1.], [1., 0., 0.], [-1., 0., 0.],
                    [0., 0., 1.], [0., 0., -1.], [-1., 0., 0.], [1., 0., 0.]]
    result = (COMPILED if compiled else HV.isolate)(**data)
    assert_bracket(result, np.array([2., 2., 2., 2., 1., 1., 2., 1.]), [0, 1, 2, 3] * 2,
                   [HV.EXIT_BRACKET] * 4 + [HV.ENTRY_BRACKET] * 4)
    np.testing.assert_array_equal(result.initial_region, [HV.INTERIOR] * 4 + [HV.EXTERIOR] * 4)
    np.testing.assert_array_equal(result.monotone, result.surfaces)
    assert_prefix_excluded(data, result)


@pytest.mark.parametrize('compiled', [False, True])
def test_earlier_extraneous_plane_cylinder_hole_zeros_do_not_block_entry(compiled):
    data = inputs(6)
    data['x'][:] = [[8., 0., -3.], [0., 0., -3.], [8., 0., -7.], [0., 0., -6.],
                    [8., 0., -2.], [5., 0., -3.]]
    data['u'][:] = [[-.8, 0., .6], [.6, 0., .8], [-.6, 0., .8], [.6, 0., .8],
                    [-.8, 0., .6], [-.6, 0., .8]]
    data['inner'][1] = 2.
    data['shi'][:] = 8.7
    result = (COMPILED if compiled else HV.isolate)(**data)
    assert_bracket(result, np.array([3.75, 10 / 3, 6.25, 5., 3.75, 1.25]), [2, 3, 0, 0, 2, 0],
                   [HV.ENTRY_BRACKET] * 6)
    np.testing.assert_array_equal(result.initial_region, [HV.EXTERIOR] * 6)
    assert_prefix_excluded(data, result)
    legacy = jax.jit(HC.isolate)(**data)
    assert np.all(np.asarray(legacy.shi) < np.asarray(result.slo))
    np.testing.assert_array_equal(legacy.status[4:], [HC.START_UNRESOLVED] * 2)


@pytest.mark.parametrize('compiled', [False, True])
def test_strict_enclosure_classification_and_clipping(compiled):
    data = inputs(7)
    data['shi'][:] = [.1, 8., 2., 2., 0., 0., 0.]
    data['x'][:] = [[3., 0., 0.], [20., 0., -3.], [0., 0., -9.], [0., 0., 0.],
                    [5., 0., 0.], [0., 0., 0.], [3., 0., 0.]]
    data['u'][1] = [0., 0., 1.]
    data['inner'][3] = data['inner'][5] = 0.
    data['p'][6] = 0.
    volume = (jax.jit(HV.enclose) if compiled else HV.enclose)(**data)
    np.testing.assert_array_equal(volume.region, [HV.INTERIOR, HV.EXTERIOR, HV.EXTERIOR,
                                                  HV.INTERIOR, HV.REGION_UNCERTAIN,
                                                  HV.INTERIOR, HV.REGION_UNSUPPORTED])
    assert volume.surfaces[4, 2]
    assert not volume.surfaces[:4].any() and not volume.surfaces[5:].any()
    assert np.isnan(volume.bounds.f_lo[6]).all()


def test_inner_zero_disabled_axis_start_and_axis_passage():
    data = inputs(3)
    data.update(x=np.array([[0., 0., 0.], [-1., 0., 0.], [0., 0., -3.]]),
                inner=np.zeros(3), outer=np.full(3, 10.), shi=np.full(3, 2.7))
    data['u'][2] = [0., 0., 1.]
    result = COMPILED(**data)
    np.testing.assert_array_equal(result.status, [HV.NO_CANDIDATE, HV.NO_CANDIDATE, HV.ENTRY_BRACKET])
    assert not result.surfaces[:, 3].any()
    assert np.isnan(result.slo[:2]).all() and np.isnan(result.shi[:2]).all()
    assert result.slo[2] < 1. < result.shi[2]


@pytest.mark.parametrize('compiled', [False, True])
def test_full_turn_equal_endpoints_both_charges_exit_and_entry(compiled):
    data = inputs(4)
    data.update(x=np.zeros((4, 3)), p=np.full(4, K), q=np.array([-1., 1., -1., 1.]),
                B=np.tile([0., 0., 1.], (4, 1)), shi=np.full(4, 2 * np.pi),
                inner=np.array([0., 0., 1., 1.]), outer=np.array([1., 1., 3., 3.]))
    point = HB.evaluate(**{key: data[key] for key in ('x', 'u', 'p', 'q', 'B')}, s=data['shi'])
    np.testing.assert_allclose(point.x, data['x'], atol=1e-15)
    result = (COMPILED if compiled else HV.isolate)(**data)
    assert_bracket(result, np.pi / 3, [2, 2, 3, 3],
                   [HV.EXIT_BRACKET] * 2 + [HV.ENTRY_BRACKET] * 2)
    assert_prefix_excluded(data, result)


@pytest.mark.parametrize('compiled', [False, True])
def test_tilted_field_curved_faces_and_parallel_motion(compiled):
    data = inputs(4)
    field = np.array([1., 0., 1.]) / np.sqrt(2.)
    data.update(x=np.zeros((4, 3)), u=np.tile([0., 1., 0.], (4, 1)), p=np.full(4, K),
                q=np.array([-1., 1., -1., 1.]), B=np.tile(field, (4, 1)),
                shi=np.full(4, 2.7), half=np.full(4, .4), inner=np.zeros(4), outer=np.full(4, 10.))
    data['u'][2:] = field
    result = (COMPILED if compiled else HV.isolate)(**data)
    expected = np.array([np.arccos(1 - .4 * np.sqrt(2.))] * 2 + [.4 * np.sqrt(2.)] * 2)
    assert_bracket(result, expected, [1, 0, 1, 1], [HV.EXIT_BRACKET] * 4)
    assert_prefix_excluded(data, result)


@pytest.mark.parametrize('compiled', [False, True])
def test_grazing_near_grazing_and_later_crossing_remain_candidates(compiled):
    data = inputs(4)
    data.update(x=np.array([[-3., 1., 0.], [-3., 1. + 1e-14, 0.], [-3., 1. - 1e-14, 0.],
                            [-3., 5., 0.]]), shi=np.full(4, 10.))
    result = (COMPILED if compiled else HV.isolate)(**data)
    np.testing.assert_array_equal(result.status, [HV.CANDIDATE] * 4)
    assert np.all(result.slo < 3.) and np.all(3. - np.asarray(result.shi) < 1e-4)
    np.testing.assert_array_equal(result.surfaces, np.eye(4, dtype=bool)[[3, 3, 3, 2]])
    assert np.all(np.asarray(result.shi[:3]) < 3. + np.sqrt(24.))


def test_curved_grazing_and_full_turn_without_boundary():
    data = inputs(4)
    data.update(x=np.zeros((4, 3)), p=np.full(4, K), q=np.array([-1., 1., -1., 1.]),
                B=np.tile([0., 0., 1.], (4, 1)), shi=np.full(4, 2 * np.pi), inner=np.zeros(4),
                outer=np.array([2., 2., 3., 3.]))
    result = COMPILED(**data)
    np.testing.assert_array_equal(result.status, [HV.CANDIDATE] * 2 + [HV.NO_CANDIDATE] * 2)
    assert np.all(result.slo[:2] < np.pi) and np.all(np.pi - np.asarray(result.shi[:2]) < 1e-4)
    assert not result.surfaces[2:].any()


@pytest.mark.parametrize('compiled', [False, True])
def test_boundary_starts_one_sided_and_padding_uncertainty(compiled):
    data = inputs(6)
    data['x'][:] = [[5., 0., 0.], [5., 0., 0.], [1., 0., 0.], [3., 0., 2.],
                    [3., 0., -2.], [5. - 1e-14, 0., 0.]]
    data['u'][:] = [[-1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.],
                    [0., 0., 1.], [1., 0., 0.]]
    result = (COMPILED if compiled else HV.isolate)(**data)
    np.testing.assert_array_equal(result.status, [HV.START_UNRESOLVED] * 6)
    assert np.all(result.slo == 0.) and np.all(result.shi == 0.) and np.all(result.work == 0)
    assert np.all(result.initial_region == HV.REGION_UNCERTAIN)
    assert np.all(result.surfaces.any(axis=1))


def test_ties_are_not_entry_exit_and_endpoint_contact_is_not_bracket():
    data = inputs(3)
    data['u'][0] = [1 / np.sqrt(2.), 0., 1 / np.sqrt(2.)]
    data['x'][1] = [7., 0., 4.]
    data['u'][1] = [-1 / np.sqrt(2.), 0., -1 / np.sqrt(2.)]
    data['shi'][2] = 2.
    result = COMPILED(**data)
    np.testing.assert_array_equal(result.status, [HV.CANDIDATE] * 3)
    assert np.all(result.surfaces[:2, 1:3]) and not result.surfaces[:2, [0, 3]].any()
    assert np.all(result.slo[:2] <= 2 * np.sqrt(2.)) and np.all(result.shi[:2] >= 2 * np.sqrt(2.))
    assert result.slo[2] < 2. and result.shi[2] == 2. and result.surfaces[2, 2]


def test_translated_axes_nonzero_start_and_neutral_tilted_field():
    data = inputs()
    data.update(x=np.array([[1003., -200., 30.]]), z=np.array([30.]), axis_x=np.array([1000.]),
                axis_y=np.array([-200.]), slo=np.array([.3]), shi=np.array([6.2]),
                q=np.zeros(1), B=np.array([[1., 2., 3.]]))
    result = COMPILED(**data)
    assert_bracket(result, 2., [2], [HV.EXIT_BRACKET])
    assert_prefix_excluded(data, result)


def test_mixed_lane_independence_eager_jit_and_precision_floor():
    data = inputs(7)
    data['shi'][1] = .25
    data['x'][2, 0] = 5.
    data['p'][3] = np.nan
    data['x'][4] = [-3., 1., 0.]
    data['x'][6] = [20., 0., -3.]
    data['u'][6] = [0., 0., 1.]
    atol = np.array([1e-6] * 5 + [1e-30, 1e-6])
    eager = HV.isolate(**data, atol=atol, max_depth=64)
    compiled = COMPILED(**data, atol=atol, max_depth=64)
    expected = [HV.EXIT_BRACKET, HV.NO_CANDIDATE, HV.START_UNRESOLVED, HV.UNSUPPORTED,
                HV.CANDIDATE, HV.PRECISION_UNRESOLVED, HV.NO_CANDIDATE]
    np.testing.assert_array_equal(compiled.status, expected)
    for a, b in zip(eager, compiled):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(compiled, COMPILED(**data, atol=atol, max_depth=64)):
        np.testing.assert_array_equal(a, b)
    for lane in range(7):
        one = {key: value[lane:lane + 1] for key, value in data.items()}
        single = COMPILED(**one, atol=atol[lane:lane + 1], max_depth=64)
        for a, b in zip(single, compiled):
            np.testing.assert_array_equal(a[0], b[lane])
    assert np.isfinite(compiled.slo[5]) and np.isfinite(compiled.shi[5])


def test_work_depth_limits_and_completed_lanes_do_not_advance():
    data = inputs(5)
    data['slo'][0], data['shi'][0] = 2. - 2e-7, 2. + 2e-7
    data['shi'][1] = .1
    data['x'][2, 0] = 5.
    data['p'][3] = 0.
    result = COMPILED(**data, max_work=1)
    np.testing.assert_array_equal(result.status, [HV.EXIT_BRACKET, HV.NO_CANDIDATE,
                                                  HV.START_UNRESOLVED, HV.UNSUPPORTED,
                                                  HV.WORK_LIMIT_UNRESOLVED])
    np.testing.assert_array_equal(result.work, [1, 1, 0, 0, 1])
    assert result.slo[4] == 0. and result.shi[4] == 3.55
    for limits in ({'max_work': 0}, {'max_work': 5}, {'max_depth': 0}, {'max_depth': 2}):
        result = COMPILED(**inputs(), **limits)
        assert result.status[0] == HV.WORK_LIMIT_UNRESOLVED
        assert np.isfinite(result.slo).all() and np.isfinite(result.shi).all()
        assert result.work[0] <= limits.get('max_work', 512)
        assert_prefix_excluded(inputs(), result)


@pytest.mark.parametrize('key,value', [('p', 0.), ('q', 101.), ('shi', np.inf), ('slo', -1.),
                                       ('slo', 8.), ('half', 0.), ('inner', -1.), ('outer', 1.),
                                       ('p', np.float32(1.)), ('atol', 0.), ('atol', np.nan),
                                       ('atol', np.float32(1e-6)), ('B', [1e-200, 0., 0.]),
                                       ('atol', np.nextafter(0., 1.)), ('half', np.nextafter(0., 1.))])
def test_invalid_is_never_no_candidate(key, value):
    result = COMPILED(**{**inputs(), key: value})
    assert result.status[0] == HV.UNSUPPORTED
    assert np.isnan(result.slo).all() and np.isnan(result.shi).all()
    assert not result.surfaces.any() and not result.monotone.any()
    assert result.initial_region[0] == result.domain_region[0] == HV.REGION_UNSUPPORTED


def test_shapes_static_limits_empty_lanes_and_singletons():
    for limits in ({'max_work': -1}, {'max_work': True}, {'max_depth': -1}, {'max_depth': 65}):
        with pytest.raises(ValueError, match='static'):
            HV.isolate(**inputs(), **limits)
    with pytest.raises(ValueError, match='identical'):
        HV.isolate(**{**inputs(), 'x': np.zeros(3)})
    with pytest.raises(ValueError):
        HV.isolate(**{**inputs(), 'B': np.zeros((2, 3))})
    result = COMPILED(**inputs(0))
    assert result.status.shape == result.initial_region.shape == result.domain_region.shape == (0,)
    assert result.surfaces.shape == result.monotone.shape == (0, 4)
    result = COMPILED(**{**inputs(), 'shi': 0.})
    assert result.status[0] == HV.NO_CANDIDATE


def test_eager_nonnumeric_and_disabled_x64_rejection():
    for key in ('atol', 'p', 'half'):
        result = HV.isolate(**{**inputs(), key: 'bad'})
        assert result.status[0] == HV.UNSUPPORTED
        assert np.isnan(result.slo).all() and not result.surfaces.any()
    previous = jax.config.x64_enabled
    try:
        jax.config.update('jax_enable_x64', False)
        result = HV.isolate(**inputs())
        assert result.status[0] == HV.UNSUPPORTED
        assert np.isnan(result.slo).all() and not result.surfaces.any()
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize('compiled', [False, True])
def test_extraneous_hole_tangency_before_physical_face_entry(compiled):
    data = inputs()
    data.update(x=np.array([[-3., 1., -7.]]), u=np.array([[.6, 0., .8]]), shi=np.array([8.7]))
    result = (COMPILED if compiled else HV.isolate)(**data)
    assert_bracket(result, 6.25, [0], [HV.ENTRY_BRACKET])
    assert_prefix_excluded(data, result)
    legacy = jax.jit(HC.isolate)(**data)
    assert legacy.status[0] == HC.CANDIDATE and legacy.shi[0] < 5.0001


@pytest.mark.parametrize('compiled', [False, True])
def test_outside_endpoints_do_not_hide_interior_entry(compiled):
    data = inputs()
    data.update(x=np.array([[-7., 0., 0.]]), shi=np.array([14.1]))
    result = (COMPILED if compiled else HV.isolate)(**data)
    assert_bracket(result, 2., [2], [HV.ENTRY_BRACKET])
    assert result.initial_region[0] == HV.EXTERIOR
    assert result.domain_region[0] == HV.REGION_UNCERTAIN
    assert_prefix_excluded(data, result)


@pytest.mark.parametrize('compiled', [False, True])
def test_padding_floor_is_uncertain_not_interior_or_a_bracket(compiled):
    data = inputs()
    data.update(x=np.zeros((1, 3)), p=1., q=1., B=np.array([0., 0., 1e-4 / K]),
                slo=1e8, shi=1e8, z=0., half=1e-4, inner=0., outer=1e8)
    volume = (jax.jit(HV.enclose) if compiled else HV.enclose)(**data)
    assert volume.region[0] == HV.REGION_UNCERTAIN
    assert volume.surfaces[0, :2].all()
    result = (COMPILED if compiled else HV.isolate)(**data)
    assert result.status[0] == HV.START_UNRESOLVED
    assert result.slo[0] == result.shi[0] == 1e8 and result.work[0] == 0
    assert result.surfaces[0, :2].all()


@pytest.mark.parametrize('compiled', [False, True])
def test_invalid_batch_domain_and_valid_integer_broadcast(compiled):
    data = inputs(6)
    data.update(p=1, q=0, B=None, z=0, half=2, inner=1, outer=5)
    data['u'][1] *= 2.
    data['x'][2, 0] = np.inf
    data['slo'][3] = np.nextafter(0., 1.)
    data['shi'][4] = 1e9
    result = (COMPILED if compiled else HV.isolate)(**data)
    np.testing.assert_array_equal(result.status, [HV.EXIT_BRACKET] + [HV.UNSUPPORTED] * 4
                                 + [HV.EXIT_BRACKET])
    for key, value in [('q', True), ('q', 1j), ('x', np.ones((1, 3), dtype=np.float32)),
                       ('outer', np.float32(5.))]:
        result = (COMPILED if compiled else HV.isolate)(**{**inputs(), key: value})
        assert result.status[0] == HV.UNSUPPORTED
        assert np.isnan(result.slo).all() and not result.surfaces.any()


def test_unvisited_exterior_node_does_not_convert_work_exhaustion_to_clear():
    data = inputs()
    data.update(x=np.array([[8., 0., -3.]]), u=np.array([[-.8, 0., .6]]), shi=np.array([5.1]))
    result = COMPILED(**data, max_work=1)
    assert result.status[0] == HV.WORK_LIMIT_UNRESOLVED and result.work[0] == 1
    assert result.slo[0] == 0. and result.shi[0] == 2.55
    assert not result.surfaces.any() and not result.monotone.any()
    assert result.domain_region[0] == HV.REGION_UNCERTAIN
    final = HV.enclose(**{**data, 'slo': result.slo, 'shi': result.shi})
    assert final.region[0] == HV.EXTERIOR


@pytest.mark.parametrize('compiled', [False, True])
def test_entry_exit_labels_replay_all_box_conditions(compiled):
    data = inputs(4)
    data['x'][:] = [[3., 0., 0.], [7., 0., 0.], [3., 0., 0.], [0., 0., 0.]]
    data['u'][:] = [[1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.], [1., 0., 0.]]
    result = (COMPILED if compiled else HV.isolate)(**data)
    np.testing.assert_array_equal(result.status, [HV.EXIT_BRACKET, HV.ENTRY_BRACKET] * 2)
    bounds = HB.enclose(**{**data, 'slo': result.slo, 'shi': result.shi})
    low = HB.enclose(**{**data, 'slo': result.slo, 'shi': result.slo})
    high = HB.enclose(**{**data, 'slo': result.shi, 'shi': result.shi})
    mask = np.asarray(result.surfaces)
    assert np.all(mask.sum(axis=1) == 1)
    assert np.all(np.asarray(bounds.f_lo)[~mask] > 0.)
    rising = (np.asarray(bounds.df_lo)[mask] > 0) & (np.asarray(low.f_hi)[mask] < 0)
    rising &= np.asarray(high.f_lo)[mask] > 0
    falling = (np.asarray(bounds.df_hi)[mask] < 0) & (np.asarray(low.f_lo)[mask] > 0)
    falling &= np.asarray(high.f_hi)[mask] < 0
    np.testing.assert_array_equal(rising, result.status == HV.ENTRY_BRACKET)
    np.testing.assert_array_equal(falling, result.status == HV.EXIT_BRACKET)
