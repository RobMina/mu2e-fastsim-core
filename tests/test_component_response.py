"""Software invariants of local bundle replay; synthetic fixtures, not Geant4 validation."""
from dataclasses import replace

import numpy as np
import pytest

from mu2efastsim.component_response import (C_MM_NS, ENTRANCE, EXITS, INCOMING, PARTICLES, STEPS,
    ComponentType, Library, Placement, UnsupportedEntrance, next_in_vacuum, replay)


def columns(schema, rows):
    return {key: np.asarray([r[key] for r in rows], dtype=bool if key == "exited" else
            np.int64 if key in ("pdg", "event", "track", "parent", "particle", "volume") else float
            ).reshape(len(rows), *tail) for key, tail in schema.items()}


def component():
    return ComponentType("finite-segmented-A", "field-free-cuts-A", (2., 3., 1.), 2)


def placement(instance=7, angle=0., translation=(0., 0., 0.), volumes=(100, 101)):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return Placement(component(), instance, rotation, np.asarray(translation), np.asarray(volumes))


def data():
    entry = dict(pdg=22, position=[0., 0., -1.], momentum=[0., 0., 10.], energy=10.)
    entrances = columns(ENTRANCE, [entry, entry])
    particles = columns(PARTICLES, [dict(pdg=22, parent=-1, exited=True),
                                    dict(pdg=22, parent=0, exited=True),
                                    dict(pdg=22, parent=1, exited=False),
                                    dict(pdg=22, parent=-1, exited=True)])
    steps = columns(STEPS, [dict(particle=0, volume=0, x0=[0., 0., -1.], x1=[0., 0., 0.],
                p0=[0., 0., 10.], p1=[0., 0., 7.], t0=0., t1=.004, length=1., edep=1.),
                           dict(particle=2, volume=1, x0=[0., 0., 0.], x1=[.5, 0., .1],
                p0=[1., 0., 0.], p1=[0., 0., 0.], t0=.006, t1=.007, length=.6, edep=1.)])
    exits = columns(EXITS, [dict(particle=0, position=[0., 0., 1.], momentum=[0., 0., 7.], energy=7., time=.008),
                           dict(particle=1, position=[2., 0., .5], momentum=[1., 0., 0.], energy=1., time=.02),
                           dict(particle=0, position=[0., 0., 1.], momentum=[0., 0., 10.], energy=10., time=.007)])
    return dict(component=component(), entrances=entrances, particles=particles, particle_offsets=[0, 3, 4],
                steps=steps, step_offsets=[0, 2, 2], exits=exits, exit_offsets=[0, 2, 3])


def bank(**changes):
    args = data()
    args.update(changes)
    return Library(**args)


def incoming(p=None, n=1, energy=10.):
    p = p or placement()
    local = np.tile([0., 0., -1.], (n, 1))
    momentum = np.tile([0., 0., energy], (n, 1))
    return dict(event=np.tile([1, 0, 1], (n, 1)), track=np.arange(1, n + 1), parent=np.full(n, -1),
                pdg=np.full(n, 22), position=p.global_points(local), momentum=p.global_vectors(momentum),
                energy=np.full(n, energy), time=np.arange(n, dtype=float) + 20.)


class FixedRandom:
    def __init__(self, draw):
        self.draw = draw

    def random(self, n):
        return np.full(n, self.draw)


def test_rigid_replay_covariance_and_volume_mapping():
    lib, a = bank(), placement()
    b = placement(23, .73, (100., -30., 17.), volumes=(9001, 5003))
    x = replay(lib, a, incoming(a), [4], FixedRandom(.1), next_track=10)
    y = replay(lib, b, incoming(b), [4], FixedRandom(.1), next_track=10)
    for key in ("x0", "x1"):
        np.testing.assert_allclose(y.steps[key], b.global_points(x.steps[key]))
    for key in ("p0", "p1"):
        np.testing.assert_allclose(y.steps[key], b.global_vectors(x.steps[key]))
    for key in ("position", "momentum"):
        transform = b.global_points if key == "position" else b.global_vectors
        np.testing.assert_allclose(y.exits[key], transform(x.exits[key]))
    for key in ("t0", "t1", "edep", "length", "track", "parent", "event"):
        np.testing.assert_array_equal(x.steps[key], y.steps[key])
    np.testing.assert_array_equal(y.steps["volume"], [9001, 5003])
    assert np.all(y.steps["instance"] == 23)
    assert np.all(y.exits["instance"] == 23)


def test_frame_roundtrip_nontrivial_three_axis_rotation():
    q, _ = np.linalg.qr(np.random.default_rng(10).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    p = replace(placement(), rotation=q, translation=[7., 8., 9.])
    x = np.random.default_rng(2).normal(size=(15, 3))
    np.testing.assert_allclose(p.local_points(p.global_points(x)), x, atol=1e-14)
    np.testing.assert_allclose(p.local_vectors(p.global_vectors(x)), x, atol=1e-14)


def test_particle_ancestry_repeated_donors_and_large_event_keys():
    inc = incoming(n=3)
    inc["event"] = np.array([[2**53 + 13, 7, 4], [2**53 + 13, 7, 4], [2**53 + 14, 7, 4]])
    inc["parent"] = np.array([5, 5, -1])
    out = replay(bank(), placement(), inc, [0, 1, 0], FixedRandom(.1), next_track=2**53 + 20)
    np.testing.assert_array_equal(out.particles["track"][::3], [1, 2, 3])
    np.testing.assert_array_equal(out.particles["parent"][::3], [5, 5, -1])
    np.testing.assert_array_equal(out.particles["parent"][1::3], [1, 2, 3])
    np.testing.assert_array_equal(out.particles["parent"][2::3], out.particles["track"][1::3])
    assert len(np.unique(out.particles["track"])) == 9
    assert out.next_track == 2**53 + 26
    np.testing.assert_array_equal(out.exits["event"], np.repeat(inc["event"], 2, axis=0))
    assert out.exits["event"].dtype == np.int64


def test_same_track_later_visit_has_distinct_interaction_and_children():
    inc = incoming(n=2)
    inc["track"][:] = 1
    inc["time"][:] = [10., 100.]
    out = replay(bank(), placement(), inc, [4, 5], FixedRandom(.1), next_track=9)
    np.testing.assert_array_equal(out.exits["track"][::2], [1, 1])
    np.testing.assert_array_equal(out.exits["interaction"][::2], [4, 5])
    assert out.exits["time"][2] - out.exits["time"][0] == pytest.approx(90.)
    assert out.particles["track"][1] != out.particles["track"][4]


def test_zero_deposit_transmission_is_an_outcome_not_missing_support():
    out = replay(bank(), placement(), incoming(), [0], FixedRandom(.9), next_track=2)
    assert len(out.steps["edep"]) == 0
    assert len(out.exits["track"]) == 1 and out.exits["track"][0] == 1
    assert out.exits["energy"][0] == 10. and out.next_track == 2


def test_terminal_absorption_keeps_trial_with_no_exits():
    args = data()
    args["particles"]["exited"][:] = False
    args["exits"] = columns(EXITS, [])
    args["exit_offsets"] = [0, 0, 0]
    out = replay(Library(**args), placement(), incoming(), [0], FixedRandom(.1), next_track=2)
    assert len(out.exits["track"]) == 0 and len(out.steps["edep"]) == 2
    assert not out.particles["exited"].any()


def test_complete_trial_draw_preserves_correlated_zero_and_two_step_outcomes():
    n = 10000
    out = replay(bank(), placement(), incoming(n=n), np.arange(n), np.random.default_rng(5), next_track=n + 1)
    counts = np.bincount(out.steps["interaction"], minlength=n)
    np.testing.assert_array_equal(counts, np.where(out.trials == 0, 2, 0))
    assert abs(np.mean(out.trials == 0) - .5) < .02
    assert np.all(np.bincount(out.exits["interaction"], minlength=n) == np.where(out.trials == 0, 2, 1))


def test_empty_incoming_batch_keeps_typed_arrays():
    out = replay(bank(), placement(), incoming(n=0), np.empty(0, np.int64), np.random.default_rng(1), next_track=1)
    assert out.next_track == 1 and out.steps["x0"].shape == (0, 3)
    assert out.exits["event"].shape == (0, 3) and out.exits["track"].dtype == np.int64
    assert len(out.trials) == 0


@pytest.mark.parametrize("change", ["energy", "position", "azimuth", "species"])
def test_unsupported_entrance_never_rescales_or_shifts_and_does_not_consume_rng(change):
    inc = incoming()
    if change == "energy":
        inc["energy"][:] = 11.
    elif change == "position":
        inc["position"][0, 0] += .01
    elif change == "azimuth":
        inc["momentum"][0] = [1., 0., 9.]
    else:
        inc["pdg"][:] = -11
    rng, other = np.random.default_rng(7), np.random.default_rng(7)
    with pytest.raises(UnsupportedEntrance) as err:
        replay(bank(), placement(), inc, [0], rng, next_track=2)
    np.testing.assert_array_equal(err.value.rows, [0])
    np.testing.assert_array_equal(rng.random(8), other.random(8))


def test_signed_species_not_pooled():
    args = data()
    args["entrances"]["pdg"][:] = 11
    args["particles"]["pdg"][[0, 3]] = 11
    inc = incoming()
    inc["pdg"][:] = -11
    with pytest.raises(UnsupportedEntrance):
        replay(Library(**args), placement(), inc, [0], np.random.default_rng(1), next_track=2)


def test_component_geometry_and_physics_must_match():
    for c in (replace(component(), geometry="different"), replace(component(), physics="different"),
              replace(component(), half_size=(2., 3., 2.))):
        with pytest.raises(ValueError, match="incompatible"):
            replay(bank(), replace(placement(), component=c), incoming(), [0], FixedRandom(.1), next_track=2)


@pytest.mark.parametrize("bad", ["reflection", "scale", "nan"])
def test_proper_rotation_only(bad):
    r = np.eye(3)
    r[0, 0] = {"reflection": -1., "scale": 2., "nan": np.nan}[bad]
    with pytest.raises(ValueError):
        replace(placement(), rotation=r)


@pytest.mark.parametrize("bad", ["offsets", "parent", "root", "step_outside", "exit_inside", "exit_inward",
                                  "entry_outward", "exit_duplicate", "late_step", "negative_edep", "length"])
def test_invalid_interaction_records_rejected(bad):
    args = data()
    if bad == "offsets":
        args["step_offsets"] = [0, 1, 1]
    elif bad == "parent":
        args["particles"]["parent"][1] = 1
    elif bad == "root":
        args["particles"]["pdg"][0] = 11
    elif bad == "step_outside":
        args["steps"]["x1"][0, 0] = 3.
    elif bad == "exit_inside":
        args["exits"]["position"][0, 2] = 0.
    elif bad == "exit_inward":
        args["exits"]["momentum"][0, 2] *= -1
    elif bad == "entry_outward":
        args["entrances"]["momentum"][0, 2] *= -1
    elif bad == "exit_duplicate":
        args["exits"]["particle"][1] = 0
    elif bad == "late_step":
        args["steps"]["t1"][0] = .1
    elif bad == "negative_edep":
        args["steps"]["edep"][0] = -.1
    else:
        args["steps"]["length"][0] = .1
    with pytest.raises(ValueError):
        Library(**args)


def test_bank_and_placement_copy_inputs_and_replay_leaves_inputs_untouched():
    args = data()
    lib = Library(**args)
    args["steps"]["edep"][0] = 99.
    assert lib.steps["edep"][0] == 1.
    with pytest.raises(ValueError):
        lib.steps["edep"][0] = 99.
    inc = incoming()
    copy = {k: v.copy() for k, v in inc.items()}
    replay(lib, placement(), inc, [0], FixedRandom(.1), next_track=2)
    for k in inc:
        np.testing.assert_array_equal(inc[k], copy[k])


def test_identity_and_allocator_validation():
    inc = incoming(n=2)
    for interaction, cursor in (([0, 0], 10), ([0, 1], 2), ([-1, 1], 10)):
        with pytest.raises(ValueError):
            replay(bank(), placement(), inc, interaction, FixedRandom(.1), next_track=cursor)
    inc["event"] = inc["event"].astype(float)
    with pytest.raises(ValueError, match="integers"):
        replay(bank(), placement(), inc, [0, 1], FixedRandom(.1), next_track=10)


def test_vacuum_navigation_recomputes_intersection_for_moved_module():
    inc = incoming()
    inc["position"][:, 2] = -5.
    far, close = placement(1, translation=(0., 0., 10.)), placement(2, translation=(0., 0., 0.))
    rows, which, entry = next_in_vacuum(inc, [far, close])
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(which, [1])
    np.testing.assert_allclose(entry["position"], [[0., 0., -1.]])
    assert entry["time"][0] == pytest.approx(20. + 4. / C_MM_NS)
    rows, which, entry = next_in_vacuum(inc, [placement(3, translation=(10., 0., 0.))])
    assert len(rows) == 0 and entry["position"].shape == (0, 3)


def test_vacuum_flight_time_uses_energy_over_momentum():
    inc = incoming()
    inc["position"][:, 2] = -3.
    inc["energy"][:] = 20.
    _, _, entry = next_in_vacuum(inc, [placement()])
    assert entry["time"][0] == pytest.approx(20. + 4. / C_MM_NS)


def test_vacuum_rotated_side_face_entry():
    p = placement(angle=np.pi / 2, translation=(100., 30., 8.))
    inc = incoming(p)
    inc["position"] -= p.global_vectors([[0., 0., 5.]])
    _, _, entry = next_in_vacuum(inc, [p])
    np.testing.assert_allclose(p.local_points(entry["position"]), [[0., 0., -1.]], atol=1e-12)
    out = replay(bank(), p, entry, [0], FixedRandom(.9), next_track=2)
    np.testing.assert_allclose(p.local_points(out.exits["position"]), [[0., 0., 1.]], atol=1e-12)


def test_two_component_chain_preserves_survivor_and_does_not_repeat_outward_surface():
    first, second = placement(1), placement(2, translation=(0., 0., 2.), volumes=(200, 201))
    out = replay(bank(), first, incoming(), [0], FixedRandom(.9), next_track=10)
    state = {key: out.exits[key] for key in INCOMING}
    rows, which, entry = next_in_vacuum(state, [first, second])
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(which, [1])
    assert entry["track"][0] == 1 and entry["time"][0] == out.exits["time"][0]
    again = replay(bank(), second, entry, [1], FixedRandom(.1), next_track=out.next_track)
    assert again.exits["track"][0] == 1 and again.exits["track"][1] == 10
    assert again.next_track == 12
    assert len(next_in_vacuum({k: again.exits[k] for k in INCOMING}, [first, second])[0]) == 0


def test_secondary_boundary_handoff_to_another_component():
    first, receiver = placement(1), placement(2, np.pi / 2, translation=(3., 0., .5))
    out = replay(bank(), first, incoming(), [0], FixedRandom(.1), next_track=10)
    state = {key: out.exits[key] for key in INCOMING}
    rows, which, entry = next_in_vacuum(state, [first, receiver])
    np.testing.assert_array_equal(rows, [1])
    assert entry["track"][0] == 10 and entry["parent"][0] == 1
    np.testing.assert_allclose(receiver.local_points(entry["position"]), [[0., 0., -1.]], atol=1e-12)
    np.testing.assert_allclose(receiver.local_vectors(entry["momentum"]), [[0., 0., 1.]], atol=1e-12)


def test_vacuum_miss_parallel_tangent_and_legitimate_reentry():
    inc = incoming(n=3)
    inc["position"][:] = [[3., 0., -5.], [2., 0., -5.], [0., 0., 1.]]
    inc["momentum"][2, 2] = -10.
    rows, _, entry = next_in_vacuum(inc, [placement()])
    np.testing.assert_array_equal(rows, [2])
    np.testing.assert_allclose(entry["position"], [[0., 0., 1.]])


def test_vacuum_rejects_inside_and_ambiguous_ownership():
    inc = incoming()
    inc["position"][:] = 0.
    with pytest.raises(ValueError, match="inside"):
        next_in_vacuum(inc, [placement()])
    inc["position"][:, 2] = -2.
    with pytest.raises(ValueError, match="ambiguous"):
        next_in_vacuum(inc, [placement(1), placement(2)])


@pytest.mark.parametrize("angle", [0., .1, .2, .73, np.pi / 2])
def test_vacuum_tangent_classification_is_rigidly_invariant(angle):
    p = placement(angle=angle, translation=(100., -30., 17.))
    inc = incoming(p, n=4)
    inc["position"] = p.global_points([[2., 0., -5.], [2., 0., -1.], [1.9999, 0., -5.], [2., 0., -1.]])
    inc["momentum"][3] = p.global_vectors([[-1., 0., 9.]])[0]
    rows, _, entry = next_in_vacuum(inc, [p])
    np.testing.assert_array_equal(rows, [2, 3])
    assert p.component.on_boundary(p.local_points(entry["position"]), p.local_vectors(entry["momentum"]), entering=True).all()


@pytest.mark.parametrize("pz", [-1e-10, 0., 1e-14])
def test_replay_rejects_outward_or_grazing_entrance_despite_state_roundoff(pz):
    args = data()
    args["entrances"]["momentum"][:] = [1., 0., 1e-10]
    lib = Library(**args)
    inc = incoming()
    inc["momentum"][:] = [1., 0., pz]
    rng, original = np.random.default_rng(3), np.random.default_rng(3)
    with pytest.raises(UnsupportedEntrance):
        replay(lib, placement(), inc, [0], rng, next_track=2)
    np.testing.assert_array_equal(rng.random(5), original.random(5))


def test_unsigned_int64_compatible_ids_and_overflow():
    inc = incoming()
    inc["event"] = np.array([[2**53 + 9, 7, 4]], np.uint64)
    out = replay(bank(), placement(), inc, np.array([0], np.uint64), FixedRandom(.1), next_track=2)
    assert out.exits["event"][0, 0] == 2**53 + 9
    inc["event"][0, 0] = 2**63
    with pytest.raises(ValueError, match="int64"):
        replay(bank(), placement(), inc, [0], FixedRandom(.1), next_track=2)


def test_vacuum_empty_particles_and_empty_geometry():
    assert len(next_in_vacuum(incoming(n=0), [placement()])[0]) == 0
    assert len(next_in_vacuum(incoming(), [])[0]) == 0
