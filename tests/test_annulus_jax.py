"""Analytic annular boundary development tests; no full navigator or G4 validation."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from mu2efastsim.geometry import annulus_jax as A
from _annulus_reference import foil_exit
from mu2efastsim import eslab_jax as SJ


# One compiled program per (lane count, entering): every scalar is passed as a traced array (as in production, where
# they come from lane state) and field=None as the zero field crossing() substitutes for None.
_CROSSING = jax.jit(A.crossing, static_argnames=("entering",))


def query(x, u, *, field=None, p=1., q=1., limit=100., z=0., half=2., inner=1., outer=5.,
          axis=0., entering=False):
    x, u = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(u, float))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return traced(x, u, field=field, p=p, q=q, limit=limit, z=z, half=half, inner=inner, outer=outer, axis=axis,
                  entering=entering)


def traced(x, u, *, field, p, q, limit, z=0., half=2., inner=1., outer=5., axis=0., entering=False):
    n = len(x)
    lane = lambda v: jnp.broadcast_to(jnp.asarray(v, jnp.float64), (n,))
    B = jnp.zeros((n, 3), jnp.float64) if field is None else jnp.broadcast_to(jnp.asarray(field, jnp.float64), (n, 3))
    return _CROSSING(jnp.asarray(x), jnp.asarray(u), lane(p), lane(q), B, lane(limit), lane(z), lane(half),
                     lane(inner), lane(outer), lane(axis), entering=entering)


@pytest.mark.pr_params("x0-u0-2.0-2")
@pytest.mark.parametrize("x,u,expected,face", [
    ([3., 0., 0.], [0., 0., 1.], 2., 2),
    ([3., 0., 0.], [0., 0., -1.], 2., 1),
    ([3., 0., 0.], [1., 0., 0.], 2., 3),
    ([3., 0., 0.], [-1., 0., 0.], 2., 4),
    ([5., 0., 0.], [1., 0., 0.], 0., 3),
    ([5., 0., 0.], [0., 1., 0.], 0., 3),
    ([5., 0., 0.], [-1., 0., 0.], 4., 4),
    ([1., 0., 0.], [-1., 0., 0.], 0., 4),
    ([1., 0., 0.], [0., 1., 0.], np.sqrt(24.), 3),
    ([3., 0., 2.], [1., 0., 0.], 2., 3),
    ([3., 0., 2.], [0., 0., -1.], 4., 1),
    ([3., 0., 2.], [0., 0., 1.], 0., 2),
])
def test_straight_faces_and_boundary_starts(x, u, expected, face):
    got = query(x, u)
    assert got.supported[0]
    np.testing.assert_allclose(got.distance, [expected], atol=1e-12)
    assert got.surface[0] == face


def test_straight_batch_matches_independent_photon_exit():
    rng = np.random.default_rng(123)
    n = 1024
    theta = rng.uniform(0., 2 * np.pi, n)
    inner, outer = rng.uniform(0., 3., n), rng.uniform(4., 10., n)
    radius = rng.uniform(inner, outer)
    center, half = rng.uniform(5000., 6000., n), rng.uniform(.002, 1., n)
    x = np.column_stack((-3904. + radius * np.cos(theta), radius * np.sin(theta),
                         center + rng.uniform(-half, half)))
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    expected = foil_exit(x, u, center, half, inner, outer, -3904.)
    got = query(x, u, z=center, half=half, inner=inner, outer=outer, axis=-3904., limit=1e6)
    assert np.all(got.supported) and expected['valid'].all()
    np.testing.assert_allclose(got.distance, expected['distance'], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(got.surface, expected['face'])


def test_no_hole_and_zero_transverse_motion():
    got = query([0., 0., 0.], [1., 0., 0.], inner=0.)
    np.testing.assert_allclose(got.distance, [5.])
    assert got.surface[0] == 3
    got = query([0., 0., 0.], [0., 0., 1.], inner=0., field=[0., 0., 10.])
    np.testing.assert_allclose(got.distance, [2.])


@pytest.mark.pr_params("x0-u0-1.0-4")
@pytest.mark.parametrize("x,u,expected,face", [
    ([0., 0., .7], [1., 0., 0.], 1., 4),
    ([6., 0., -.7], [-1., 0., 0.], 1., 3),
    ([3., 0., -3.], [0., 0., 1.], 1., 1),
    ([1., 0., .7], [1., 0., 0.], 0., 4),
    ([1., 0., .7], [-1., 0., 0.], 2., 4),
    ([5., 0., .7], [-1., 0., 0.], 0., 3),
])
def test_lateral_and_face_entry_at_actual_depth(x, u, expected, face):
    got = query(x, u, entering=True)
    assert got.supported[0]
    np.testing.assert_allclose(got.distance, [expected], atol=1e-12)
    assert got.surface[0] == face


def test_vacuum_chord_across_hole_has_entry_with_equal_endpoint_radii():
    got = query([6., 0., .7], [-1., 0., 0.], limit=12., entering=True)
    np.testing.assert_allclose(got.distance, [1.])
    assert got.surface[0] == 3


@pytest.mark.pr_params("-1.0")
@pytest.mark.parametrize("charge", [-1., 1.])
def test_curved_first_exit_even_after_full_turn_return(charge):
    k = .2
    radius = 1 / k
    limit = 2 * np.pi / k
    field = [0., 0., k / SJ.K_FIELD]
    got = query([0., 0., 0.], [1., 0., 0.], q=charge, field=field, inner=0., outer=4., limit=limit)
    expected = 2 * np.arcsin(4. / (2 * radius)) / k
    np.testing.assert_allclose(got.distance, [expected], atol=1e-11)
    assert got.surface[0] == 3
    endpoint = SJ.helix(jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]), jnp.ones(1), charge,
                        jnp.array([field]), jnp.array([limit]))[0]
    np.testing.assert_allclose(endpoint, 0., atol=1e-12)


@pytest.mark.pr_params("-1.0")
@pytest.mark.parametrize("charge", [-1., 1.])
def test_curved_hole_entry_after_exit(charge):
    field = [0., 0., 1. / SJ.K_FIELD]
    x = [1., 0., .3]
    u = [-.5, charge * np.sqrt(.75), 0.]
    first = query(x, u, field=field, q=charge, inner=1., outer=5.)
    assert first.distance[0] == 0. and first.surface[0] == 4
    entry = query(x, u, field=field, q=charge, inner=1., outer=5., entering=True)
    assert entry.supported[0] and 0. < entry.distance[0] < 2 * np.pi
    s = float(entry.distance[0])
    xx, uu = jnp.array([x]), jnp.array([u])
    before = SJ.helix(xx, uu, jnp.ones(1), charge, jnp.array([field]), jnp.array([s - 1e-5]))[0]
    after = SJ.helix(xx, uu, jnp.ones(1), charge, jnp.array([field]), jnp.array([s + 1e-5]))[0]
    assert np.linalg.norm(before[0, :2]) < 1. < np.linalg.norm(after[0, :2])


def test_curved_radial_root_repeats_after_axial_admission():
    k = 1.
    uz = .1
    u = [np.sqrt(1 - uz * uz), 0., uz]
    field = [0., 0., k / SJ.K_FIELD]
    got = query([0., 0., -2.], u, field=field, inner=1., outer=1.5,
                half=.1, entering=True, limit=30.)
    assert got.supported[0] and np.isfinite(got.distance[0])
    s = float(got.distance[0])
    at = SJ.helix(jnp.array([[0., 0., -2.]]), jnp.array([u]), jnp.ones(1), 1.,
                  jnp.array([field]), jnp.array([s]))[0][0]
    assert abs(at[2]) <= .1 + 1e-12
    assert 1. - 1e-12 <= np.linalg.norm(at[:2]) <= 1.5 + 1e-12
    assert s > 3 * 2 * np.pi


def test_tangent_touch_interior_does_not_exit():
    got = query([-3., 1., 0.], [1., 0., 0.], inner=1.)
    np.testing.assert_allclose(got.distance, [3. + np.sqrt(24.)])
    assert got.surface[0] == 3


def test_curved_outer_touch_does_not_exit():
    got = query([0., 0., 0.], [1., 0., 0.], field=[0., 0., 1. / SJ.K_FIELD], inner=0., outer=2., limit=7.)
    assert got.supported[0] and np.isinf(got.distance[0]) and got.surface[0] == 0


def test_circular_motion_on_surface_is_not_an_exit():
    got = query([2., 0., 0.], [0., -1., 0.], field=[0., 0., .5 / SJ.K_FIELD],
                inner=0., outer=2., limit=30.)
    assert got.supported[0] and np.isinf(got.distance[0])


def test_curved_root_at_pi_phase():
    got = query([0., 0., 0.], [1., 0., 0.], field=[0., 0., 1. / SJ.K_FIELD],
                inner=0., outer=2., limit=np.pi)
    assert np.isinf(got.distance[0])
    got = query([1., 0., 0.], [0., 1., 0.], field=[0., 0., 1. / SJ.K_FIELD],
                inner=0., outer=2., limit=7.)
    assert got.supported[0] and np.isfinite(got.distance[0])


def test_limit_and_unsupported_are_distinct():
    clear = query([3., 0., 0.], [1., 0., 0.], limit=1.)
    assert clear.supported[0] and np.isinf(clear.distance[0]) and clear.surface[0] == 0
    hit = query([3., 0., 0.], [1., 0., 0.], limit=2.)
    assert hit.distance[0] == 2.
    for kwargs in ({'field': [1e-16, 0., 1.]}, {'p': 0.}, {'limit': np.inf}, {'inner': 6.}):
        bad = query([3., 0., 0.], [1., 0., 0.], **kwargs)
        assert not bad.supported[0] and np.isnan(bad.distance[0]) and bad.surface[0] == 0
    bad = query([3., 0., 0.], [1., 0., 0.], entering=True)
    assert not bad.supported[0]
    bad = query([6., 0., 0.], [1., 0., 0.])
    assert not bad.supported[0]


def test_safety_includes_rim_hole_and_faces():
    x = jnp.array([[4.9, 0., 0.], [1.1, 0., 0.], [3., 0., 1.9], [0., 0., 0.]])
    np.testing.assert_allclose(A.safety(x, 0., 2., jnp.array([1., 1., 1., 0.]), 5.), [.1, .1, .1, 2.])


@pytest.mark.parametrize("charge", [-1., 1.])
def test_random_axial_field_exits_against_circle_intersections(charge):
    rng = np.random.default_rng(320)
    n = 1024
    inner, outer = rng.uniform(.1, 2., n), rng.uniform(3., 8., n)
    radius, phi = rng.uniform(inner, outer), rng.uniform(-np.pi, np.pi, n)
    x = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.zeros(n)))
    u = rng.normal(size=(n, 3))
    u[:, 2] = 0.
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    k = charge * rng.uniform(.1, 3., n)
    cx, cy = x[:, 0] + u[:, 1] / k, x[:, 1] - u[:, 0] / k
    orbit = 1 / np.abs(k)
    amplitude = 2 * np.hypot(cx, cy) * orbit
    cosine = 2 * (cx * (-u[:, 1] / k) + cy * (u[:, 0] / k))
    sine = 2 * (cx * (u[:, 0] / k) + cy * (u[:, 1] / k))
    phase = np.arctan2(sine * np.sign(k), cosine)
    constant = cx * cx + cy * cy + orbit * orbit
    roots = []
    for r in (inner, outer):
        level = (r * r - constant) / amplitude
        delta = np.arccos(np.clip(level, -1., 1.))
        times = np.mod(phase[:, None] + np.stack((delta, -delta), axis=1), 2 * np.pi) / np.abs(k[:, None])
        roots.append(np.where((np.abs(level) < 1.)[:, None], times, np.inf))
    expected = np.min(np.concatenate(roots, axis=1), axis=1)
    field = np.column_stack((np.zeros(n), np.zeros(n), np.abs(k) / SJ.K_FIELD))
    got = query(x, u, field=field, q=charge, inner=inner, outer=outer, limit=100.)
    assert np.all(got.supported)
    np.testing.assert_allclose(got.distance, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.pr_params("0.0")
@pytest.mark.parametrize("field", [0., 1e-18, 1e-12, 1e-6])
def test_weak_curvature_limit_near_rim(field):
    got = query([5. - 1e-9, 0., 0.], [1., 0., 0.], field=[0., 0., field])
    assert got.supported[0]
    np.testing.assert_allclose(got.distance, [1e-9], atol=1e-15, rtol=1e-6)


def test_integer_coordinate_hole_tangent_is_not_an_exit():
    got = query([-5., 1., 0.], [1., 0., 0.], outer=6.)
    np.testing.assert_allclose(got.distance, [5. + np.sqrt(35.)], atol=1e-12)
    assert got.surface[0] == 3


def test_distant_straight_entry_root_is_not_lost_to_reconstruction_roundoff():
    got = query([1000., 1., 0.], [-1., 0., 0.], limit=1001., entering=True)
    assert got.supported[0] and got.surface[0] == 3
    np.testing.assert_allclose(got.distance, [1000. - np.sqrt(24.)], atol=1e-10)


def test_many_turns_before_axial_admission():
    ut = np.sqrt(.99)
    s = 2 * np.arcsin(1 / (2 * ut)) + 100 * np.pi
    got = query([0., 0., -.1 * s], [ut, 0., .1], field=[0., 0., 1 / SJ.K_FIELD],
                inner=1., outer=1.5, half=.01, limit=s + 1., entering=True)
    assert got.supported[0] and got.surface[0] == 4
    np.testing.assert_allclose(got.distance, [s], atol=1e-11)


def test_surface_following_entry_reports_touched_surface():
    got = query([3., 0., 2.], [1., 0., 0.], entering=True)
    assert got.supported[0] and got.distance[0] == 0. and got.surface[0] == 2
    got = query([2., 0., 0.], [0., -1., 0.], field=[0., 0., .5 / SJ.K_FIELD],
                inner=0., outer=2., entering=True)
    assert got.supported[0] and got.distance[0] == 0. and got.surface[0] == 3


def test_rotated_surface_following_circle():
    phi = np.linspace(0., 2 * np.pi, 257)
    x = np.column_stack((2 * np.cos(phi), 2 * np.sin(phi), np.zeros(phi.size)))
    u = np.column_stack((np.sin(phi), -np.cos(phi), np.zeros(phi.size)))
    got = query(x, u, field=[0., 0., .5 / SJ.K_FIELD], inner=0., outer=2., limit=30.)
    assert np.all(np.isinf(np.asarray(got.distance)[np.asarray(got.supported)]))
    assert np.all(np.isnan(np.asarray(got.distance)[~np.asarray(got.supported)]))
    assert not np.any(got.surface)


def test_tiny_axial_velocity_is_not_classified_as_stationary():
    got = query([2., 0., 0.], [0., -1., 1e-14], field=None, half=.001,
                inner=0., outer=1e13, limit=2e11)
    assert got.supported[0] and got.surface[0] == 2
    np.testing.assert_allclose(got.distance, [1e11])


@pytest.mark.pr_params("-2.0-True-False", "-2.0-True-True")
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('entering', [False, True])
@pytest.mark.parametrize('k', [0., -.2, .2, -2., 2.])
def test_near_axial_radial_boundary_starts_resolve_or_fail_closed(compiled, entering, k):
    radius, radial_sign, axial_sign, transverse = np.meshgrid(
        [1., 5.], [-1., 1.], [-1., 1.], [1e-10, 1e-12, 1e-13, 1e-14, 1e-15], indexing='ij')
    radius, radial_sign, axial_sign, transverse = (
        a.ravel() for a in (radius, radial_sign, axial_sign, transverse))
    x = np.column_stack((radius, np.zeros_like(radius), np.zeros_like(radius)))
    u = np.column_stack((radial_sign * transverse, np.zeros_like(radius), axial_sign))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    field = None if k == 0 else [0., 0., -100. * k / SJ.K_FIELD]
    if compiled:
        got = traced(x, u, field=field, p=100., q=-1., limit=4., entering=entering)
    else:
        got = A.crossing(jnp.asarray(x), jnp.asarray(u), 100., -1., field, 4., 0., 2., 1., 5., entering=entering)
    material_directed = (radius == 1.) == (radial_sign > 0)
    radial_face = np.where(radius == 1., 4, 3)
    axial_face = np.where(axial_sign > 0, 2, 1)
    axial_exit = 2. / np.abs(u[:, 2])
    # The second boundary root is analytic even when its displacement rounds away.
    returning = (np.full_like(radius, np.inf) if k == 0 else
                 np.mod(2 * np.arctan(-abs(k) * radius / u[:, 0]), 2 * np.pi) / abs(k))
    if entering:
        distance = np.where(material_directed, 0., np.where(returning <= axial_exit, returning, np.inf))
        surface = np.where(np.isfinite(distance), radial_face, 0)
    else:
        distance = np.where(material_directed, np.minimum(returning, axial_exit), 0.)
        surface = np.where(material_directed & (axial_exit < returning), axial_face, radial_face)
    supported = np.asarray(got.supported)
    np.testing.assert_allclose(np.asarray(got.distance)[supported], distance[supported], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(got.surface)[supported], surface[supported])
    assert np.all(np.isnan(np.asarray(got.distance)[~supported]))
    assert np.all(np.asarray(got.surface)[~supported] == 0)
    assert np.all(supported[transverse == 1e-10])


@pytest.mark.pr_params("1e-08-1.0")
@pytest.mark.parametrize('radius', [1., 5.])
@pytest.mark.parametrize('transverse', [1e-8, 2. ** -26])
def test_small_transverse_boundary_tangent_resolves_or_fails_closed(radius, transverse):
    got = query([radius, 0., 0.], [0., transverse, 1.], half=10000., limit=20000.)
    if got.supported[0]:
        expected = 0. if radius == 5. else 10000. * np.sqrt(1. + transverse ** 2)
        np.testing.assert_allclose(got.distance, [expected], atol=1e-12, rtol=1e-12)
        assert got.surface[0] == (3 if radius == 5. else 2)
    else:
        assert np.isnan(got.distance[0]) and got.surface[0] == 0


@pytest.mark.pr_params("0.0-False")
@pytest.mark.parametrize('entering', [False, True])
@pytest.mark.parametrize('k', [0., -.2, .2])
def test_exact_axial_radial_boundary_starts_stay_supported(entering, k):
    x = [[1., 0., 0.], [1., 0., 0.], [5., 0., 0.], [5., 0., 0.]]
    u = [[0., 0., -1.], [0., 0., 1.], [0., 0., -1.], [0., 0., 1.]]
    field = None if k == 0 else [0., 0., k / SJ.K_FIELD]
    got = query(x, u, field=field, entering=entering, limit=4.)
    assert np.all(got.supported)
    np.testing.assert_array_equal(got.distance, np.full(4, 0. if entering else 2.))
    np.testing.assert_array_equal(got.surface, [4, 4, 3, 3] if entering else [1, 2, 1, 2])


def test_numerically_extreme_curvature_and_phase_are_unsupported():
    for field, limit in ((1e-160 / SJ.K_FIELD, 100.), (1 / SJ.K_FIELD, 2e11)):
        got = query([3., 0., 0.], [1., 0., 0.], field=[0., 0., field], limit=limit)
        assert not got.supported[0] and np.isnan(got.distance[0])


@pytest.mark.pr_params("x0-1.0-0.0-6.0-False")
@pytest.mark.parametrize('x,k,inner,outer,entering', [
    ([3., 5., 0.], 1., 0., 6., False),
    ([3., 3., 0.], -1., 4., 7., False),
    ([3., 3., 0.], -1., 0., 4., True),
    ([3., 5., 0.], 1., 6., 7., True),
])
def test_offset_circle_exact_tangency_is_not_a_transition(x, k, inner, outer, entering):
    got = query(x, [1., 0., 0.], field=[0., 0., k / SJ.K_FIELD], inner=inner, outer=outer,
                entering=entering, limit=10.)
    assert got.supported[0] and np.isinf(got.distance[0]) and got.surface[0] == 0


@pytest.mark.pr_params("-1e-14")
@pytest.mark.parametrize('shift', [-1e-14, 1e-14])
def test_unresolved_near_tangencies_are_not_reported_as_clear(shift):
    straight = query([-5., 1. + shift, 0.], [1., 0., 0.], outer=6.)
    assert not straight.supported[0] and np.isnan(straight.distance[0])
    curved = query([3., 5., 0.], [1., 0., 0.], field=[0., 0., 1 / SJ.K_FIELD],
                    inner=0., outer=6. + shift, limit=10.)
    assert not curved.supported[0] and np.isnan(curved.distance[0])


def test_shape_dtype_and_empty_lanes():
    with pytest.raises(ValueError, match='float64'):
        A.crossing(jnp.ones((1, 3), jnp.float32), jnp.ones((1, 3), jnp.float32), 1., 1., None, 1., 0., 1., 0., 2.)
    with pytest.raises(ValueError, match='identical'):
        A.crossing(jnp.ones((1, 2)), jnp.ones((1, 3)), 1., 1., None, 1., 0., 1., 0., 2.)
    got = A.crossing(jnp.zeros((0, 3)), jnp.zeros((0, 3)), 1., 1., None, 1., 0., 1., 0., 2.)
    assert got.distance.shape == got.surface.shape == got.supported.shape == (0,)
