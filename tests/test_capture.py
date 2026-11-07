"""Small, local, synthetic schema-1 capture smoke regressions (no transport/physics certification)."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_compare

BIG = 2**53
BOOL = set("complete bankAdmissible recordComplete recordBankAdmissible capturePresent statusPresent eventIDsMatch "
           "statusClean overflowSimParticles entered ended preInside postInside firstExit killed".split())
STR = set("geometryTag physicsTag envelope errors validationErrors creator stopProcess name logical motherLogical "
          "material solid placement process".split())
FLOAT = set(capture.FLOATS + "deposit nonIonizingDeposit length density cpuTime realTime".split())


def empty(name, size):
    fields = [*capture.EVENT, *capture.FIELDS[name].split(),
              *(prefix + field for prefix in capture.STATES.get(name, ()) for field in capture.STATE)]
    table = {}
    for field in fields:
        leaf = field.split("_")[-1]
        dtype = (np.int64 if field in capture.INT64 or leaf == "volume" else np.bool_ if field in BOOL
                 else object if leaf in STR else np.float64 if leaf in FLOAT else np.int32)
        table[field] = np.full(size, "" if dtype == object else 0, dtype=dtype)
    for field, width in (("translation", 3), ("rotation", 9), ("halfSize", 3)):
        if field in table:
            table[field] = np.zeros((size, width))
    return table


def copy_state(dest, prefix, source, source_prefix, rows):
    for field in capture.STATE:
        dest[prefix + field][:] = source[source_prefix + field][rows]


@pytest.fixture
def tables():
    """Two distinct large triplets: zero-deposit transmission, then a stopping parent and child."""
    result = {"events": empty("events", 2), "tracks": empty("tracks", 3), "steps": empty("steps", 5),
              "classifications": empty("classifications", 3), "volumes": empty("volumes", 4)}
    event_rows = {"events": [0, 1], "tracks": [0, 1, 1], "steps": [0, 0, 1, 1, 1],
                  "classifications": [0, 1, 1], "volumes": [0, 0, 1, 1]}
    for name, table in result.items():
        for field in capture.EVENT:
            table[field][:] = np.array([BIG + 1, BIG + 2], dtype=np.int64)[event_rows[name]]
    e, t, s, c, v = (result[name] for name in capture.TABLES)
    for field in "complete recordComplete capturePresent statusPresent eventIDsMatch statusClean".split():
        e[field][:] = True
    for field in capture.EVENT:
        e["record" + field.capitalize()][:] = e[field]
    e["schema"][:] = 1
    e["geometryTag"][:] = "synthetic-box"
    e["physicsTag"][:] = "synthetic-not-physics"
    e["envelope"][:] = "box"
    e["nTracks"][:], e["nG4Tracks"][:], e["nClassifications"][:] = [1, 2], [1, 2], [1, 2]
    e["nSteps"][:], e["nVolumes"][:] = [2, 3], [2, 2]
    e["halfSize"][:] = 1
    e["rotation"][:] = [np.eye(3).ravel(), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).ravel()]
    e["translation"][:] = [[0, 0, 0], [11, 13, 17]]
    t["index"][:] = [0, 0, 1]
    t["track"][:] = [BIG + 1, BIG + 1, BIG + 2]
    t["parent"][:] = [0, 0, BIG + 1]
    t["pdg"][:] = [13, 13, 11]
    t["steps"][:], t["secondaries"][:] = [2, 2, 1], [0, 1, 0]
    t["reason"][:], t["status"][:] = [2, 1, 1], [0, 2, 2]
    t["entered"][:], t["ended"][:] = True, True
    s["index"][:] = [0, 1, 0, 1, 2]
    s["track"][:] = t["track"][[0, 0, 1, 1, 2]]
    s["number"][:], s["secondaries"][:] = [1, 2, 1, 2, 1], [0, 0, 0, 1, 0]
    s["preInside"][:], s["postInside"][:] = [0, 1, 0, 1, 1], [1, 0, 1, 1, 1]
    s["firstExit"][:], s["trackStatus"][:] = [0, 1, 0, 0, 0], [0, 0, 0, 2, 2]
    s["deposit"][:], s["nonIonizingDeposit"][:] = [0, 0, 0, 2, 1], [0, 0, 0, 0.25, 0]
    s["length"][:] = [0.001, 2, 0.001, 1, 0.4]
    s["pre_lz"][:], s["post_lz"][:] = [-1.001, -1, -1.001, -1, 0], [-1, 1, -1, 0, 0.4]
    s["pre_time"][:], s["post_time"][:] = [0, 1, 0, 1, 2], [1, 3, 1, 2, 3]
    s["pre_lpz"][:], s["post_lpz"][:] = 1, [1, 1, 1, 0, 0]
    er = np.asarray(event_rows["steps"])
    for prefix in ("pre_", "post_"):
        for local, world, shift in (("l", "", e["translation"][er]), ("lp", "p", 0)):
            transformed = np.einsum("nij,nj->ni", e["rotation"][er].reshape(-1, 3, 3),
                                    capture.vectors(s, prefix + local, "xyz")) + shift
            for axis, field in enumerate("xyz"):
                s[prefix + world + field][:] = transformed[:, axis]
        s[prefix + "weight"][:] = 1
        s[prefix + "volume"][:] = BIG + 1
        s[prefix + "material"][:] = "G4_Galactic" if prefix == "pre_" else "material"
    s["post_stepStatus"][:] = [1, 1, 1, 0, 0]
    copy_state(t, "begin_", s, "pre_", [0, 2, 4])
    copy_state(t, "end_", s, "post_", [1, 3, 4])
    copy_state(t, "entrance_", s, "post_", [0, 2, 3])
    for field in capture.STATE:
        t["entrance_" + field][2] = t["begin_" + field][2]
    for field in ("index", "track", "parent", "pdg"):
        c[field][:] = t[field]
    copy_state(c, "state_", t, "begin_", slice(None))
    v["index"][:], v["id"][:] = [0, 1, 0, 1], [BIG + 1, BIG + 2, BIG + 1, BIG + 2]
    v["name"][:] = ["box", "world", "box", "world"]
    v["solid"][:] = "G4Box"
    v["translation"][:] = e["translation"][event_rows["volumes"]]
    v["rotation"][:] = e["rotation"][event_rows["volumes"]]
    return result


def test_distribution_diagnostic_preserves_empty_outcomes(tables):
    values = capture_compare.observables(tables)
    np.testing.assert_array_equal(values["component_deposit_MeV"], [0, 3])
    report = capture_compare.compare(tables, tables, 2)
    assert not report["geometry_equivalence_established"] and not report["bank_admissible"]
    assert report["metrics"]["component_deposit_MeV"]["difference"] == 0
    assert report["metrics"]["component_deposit_MeV"]["reference_mean"] == 1.5
    assert report["metrics"]["component_deposit_MeV"]["reference_mean_se"] == pytest.approx(1.5)


def test_comparison_rejects_partial_or_different_physics(tables):
    candidate = {name: {key: value.copy() for key, value in table.items()} for name, table in tables.items()}
    candidate["events"]["complete"][0] = False
    with pytest.raises(ValueError, match="complete"):
        capture_compare.compare(tables, candidate, 2)
    candidate["events"]["complete"][0] = True
    candidate["events"]["physicsTag"][:] = "other"
    with pytest.raises(ValueError, match="physicsTag"):
        capture_compare.compare(tables, candidate, 2)


def test_comparison_aggregates_by_full_event_identity(tables):
    expected = capture_compare.observables(tables)
    for name in capture.TABLES:
        tables[name] = {field: value[::-1].copy() for field, value in tables[name].items()}
    actual = capture_compare.observables(tables)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key][::-1])


def test_approach_deposit_is_preserved_and_separate(tables):
    tables["steps"]["deposit"][0] = 2.7808513550537834e-29
    report = capture.audit(tables, 2)
    assert report["approach_deposit"] == tables["steps"]["deposit"][0]
    assert report["component_deposit"] == 3.0
    assert tables["steps"]["deposit"][0] > 0


def test_diagnose_does_not_accept_failed_event(tables, capsys):
    tables["events"]["complete"][0] = False
    tables["events"]["errors"][0] = "nontrivial_outside_transport"
    report = capture.diagnose(tables)
    assert report["diagnostic_only"] and not report["bank_admissible"]
    assert "nontrivial_outside_transport" in capsys.readouterr().out
    with pytest.raises(ValueError, match="complete"):
        capture.audit(tables, 2)


@pytest.mark.parametrize("status", [0, 1, 2])
def test_zero_step_natural_child_is_retained(tables, status):
    tables["steps"] = {field: value[:4] for field, value in tables["steps"].items()}
    tables["events"]["nSteps"][1] = 2
    t = tables["tracks"]
    t["steps"][2] = 0
    tables["classifications"]["status"][2] = status
    for field in capture.STATE:
        t["end_" + field][2] = t["begin_" + field][2]
    assert capture.audit(tables, 2)["natural_stops"] == 2


@pytest.mark.parametrize("status", [-1, 3, 4, 5, 6, 99])
def test_unsupported_classification_status(tables, status):
    tables["classifications"]["status"][2] = status
    with pytest.raises(ValueError, match="unsupported classification status"):
        capture.audit(tables, 2)


@pytest.mark.parametrize("status", [-1, 2, 3, 4, 5, 6, 99])
def test_nonfinal_step_must_continue(tables, status):
    tables["steps"]["trackStatus"][2] = status
    with pytest.raises(ValueError, match="unsupported intermediate track status"):
        capture.audit(tables, 2)


@pytest.mark.parametrize("row", [0, 2])
def test_terminal_classification_cannot_have_steps(tables, row):
    tables["classifications"]["status"][row] = 2
    with pytest.raises(ValueError, match="terminal classification"):
        capture.audit(tables, 2)


def test_stop_but_alive_intermediate_keeps_at_rest_path(tables):
    tables["steps"]["trackStatus"][2] = 1
    assert capture.audit(tables, 2)["events"] == 2


@pytest.mark.parametrize("inset,accepted", [(0., False), (5e-8, False), (2e-7, True), (-5e-8, False)])
@pytest.mark.parametrize("atol", [1e-7, 1e-8])
def test_secondary_birth_has_conservative_audit_margin(tables, inset, accepted, atol):
    tables["steps"] = {field: value[:4] for field, value in tables["steps"].items()}
    tables["events"]["nSteps"][1] = 2
    t, e, c = tables["tracks"], tables["events"], tables["classifications"]
    t["steps"][2] = 0
    local = np.array([0., 0., 1. - inset])
    world = e["rotation"][1].reshape(3, 3) @ local + e["translation"][1]
    for field in capture.STATE:
        t["end_" + field][2] = t["begin_" + field][2]
    for table, prefix in ((t, "begin_"), (t, "end_"), (t, "entrance_"), (c, "state_")):
        for axis, field in enumerate("xyz"):
            table[prefix + "l" + field][2] = local[axis]
            table[prefix + field][2] = world[axis]
    if accepted or inset > atol:
        assert capture.audit(tables, 2, atol=atol)["events"] == 2
    else:
        with pytest.raises(ValueError, match="secondary not born strictly inside"):
            capture.audit(tables, 2, atol=atol)


def test_physical_row_order_does_not_replace_record_indices(tables):
    for name in capture.TABLES:
        tables[name] = {field: value[::-1].copy() for field, value in tables[name].items()}
    assert capture.audit(tables, 2)["events"] == 2


def test_transmission_stop_census_and_zero_deposit(tables):
    result = capture.audit(tables, 2)
    assert result == {"events": 2, "rows": {"events": 2, "tracks": 3, "steps": 5, "classifications": 3, "volumes": 4},
                      "bank_admissible": False, "experimental": True, "first_exits": 1, "natural_stops": 2,
                      "secondaries": 1, "zero_deposit_steps": 3, "raw_deposit": 3.0, "raw_nonionizing_deposit": 0.25,
                      "component_deposit": 3.0, "approach_deposit": 0.0}


@pytest.mark.parametrize("name,field,row,value,message", [
    ("events", "complete", 0, False, "complete"),
    ("events", "statusPresent", 0, False, "statusPresent"),
    ("events", "g4Status", 0, 1, "g4Status"),
    ("events", "bankAdmissible", 0, True, "bankAdmissible"),
    ("events", "recordEvent", 0, BIG + 2, "identity"),
    ("events", "nTracks", 0, 2, "census"),
    ("events", "nSteps", 0, 3, "census"),
    ("events", "nClassifications", 0, 2, "census"),
    ("events", "nVolumes", 0, 3, "census"),
    ("tracks", "parent", 2, BIG + 3, "parent"),
    ("tracks", "parent", 2, BIG + 2, "ancestry"),
    ("tracks", "secondaries", 1, 0, "secondary"),
    ("tracks", "steps", 0, 3, "step census"),
    ("tracks", "reason", 0, 3, "reason"),
    ("tracks", "status", 1, 0, "status"),
    ("steps", "number", 1, 3, "sequence"),
    ("steps", "track", 4, BIG + 3, "step"),
    ("steps", "secondaries", 3, 0, "secondary"),
    ("steps", "firstExit", 1, False, "exit"),
    ("steps", "postInside", 1, True, "exit"),
    ("steps", "post_stepStatus", 1, 0, "geometry boundary"),
    ("steps", "post_lz", 0, -0.5, "state continuity|covariance|boundary"),
    ("tracks", "entrance_x", 1, 9.0, "state continuity|covariance"),
    ("classifications", "state_px", 2, 9.0, "state continuity|covariance"),
    ("classifications", "killed", 2, True, "acceptance"),
    ("classifications", "track", 2, BIG + 1, "duplicate classification"),
    ("classifications", "parent", 2, BIG + 2, "parent"),
    ("volumes", "id", 2, BIG + 3, "state volume"),
])
def test_rejects_corrupt_records(tables, name, field, row, value, message):
    tables[name][field][row] = value
    with pytest.raises(ValueError, match=message):
        capture.audit(tables, 2)


@pytest.mark.parametrize("field,width", [("translation", 3), ("rotation", 9), ("halfSize", 3)])
def test_ragged_metadata_has_explicit_rejection(tables, field, width, tmp_path, monkeypatch):
    tables["events"][field] = ak.unflatten(tables["events"][field][0].copy(), [width, 0])
    before = ak.copy(tables["events"][field])
    message = f"^events: malformed {field}$"
    with pytest.raises(ValueError, match=message):
        capture.audit(tables, 2)
    monkeypatch.setattr(capture, "read", lambda *args: tables)
    output = tmp_path / "audit.json"
    with pytest.raises(ValueError, match=message):
        capture.main(["unused.root", "--expected-events", "2", "--output", str(output)])
    assert not output.exists()
    assert ak.almost_equal(tables["events"][field], before)


def test_missing_step_even_with_corrected_event_census(tables):
    tables["steps"] = {field: value[[0, 2, 3, 4]] for field, value in tables["steps"].items()}
    tables["events"]["nSteps"][0] = 1
    with pytest.raises(ValueError, match="track step census"):
        capture.audit(tables, 2)


def test_expected_events_empty_and_missing_tables(tables):
    with pytest.raises(ValueError, match="expected event"):
        capture.audit(tables, 3)
    empty_tables = {name: {field: value[:0] for field, value in table.items()} for name, table in tables.items()}
    with pytest.raises(ValueError, match="expected event"):
        capture.audit(empty_tables, 1)
    with pytest.raises(ValueError, match="positive expected"):
        capture.audit(empty_tables, 0)
    del tables["steps"]
    with pytest.raises(ValueError, match="missing table"):
        capture.audit(tables, 2)


def test_empty_event_is_not_success(tables):
    for name in capture.TABLES[1:]:
        tables[name] = {field: value[:0] for field, value in tables[name].items()}
        tables["events"]["n" + name.capitalize()][:] = 0
    with pytest.raises(ValueError, match="empty capture"):
        capture.audit(tables, 2)


@pytest.mark.parametrize("name,field", [("events", "event"), ("tracks", "track"), ("tracks", "parent"),
                                        ("steps", "track"), ("volumes", "id"), ("steps", "pre_volume")])
def test_no_float_identity_roundtrip(tables, name, field):
    tables[name][field] = tables[name][field].astype(float)
    with pytest.raises(ValueError, match="int64"):
        capture.audit(tables, 2)


def test_exit_must_be_last(tables):
    s = tables["steps"]
    tables["steps"] = {field: value[[0, 1, 1, 2, 3, 4]].copy() for field, value in s.items()}
    s = tables["steps"]
    s["index"][2], s["number"][2] = 2, 3
    s["firstExit"][2], s["preInside"][2] = False, False
    tables["events"]["nSteps"][0], tables["tracks"]["steps"][0] = 3, 3
    with pytest.raises(ValueError, match="after first exit"):
        capture.audit(tables, 2)


def test_boundary_not_just_consistent_transforms(tables):
    # Move the entire event-1 exit consistently, but off its actual face.
    tables["steps"]["post_lz"][1] = tables["steps"]["post_z"][1] = 1.1
    tables["tracks"]["end_lz"][0] = tables["tracks"]["end_z"][0] = 1.1
    with pytest.raises(ValueError, match="exact box boundary"):
        capture.audit(tables, 2)


@pytest.mark.parametrize("field", ["x", "px"])
def test_covariance_not_just_matching_states(tables, field):
    # Keep every linked begin/pre/classification state equal; violate only R*x+t or R*p.
    tables["steps"]["pre_" + field][2] += 0.1
    tables["tracks"]["begin_" + field][1] += 0.1
    tables["classifications"]["state_" + field][1] += 0.1
    with pytest.raises(ValueError, match="covariance"):
        capture.audit(tables, 2)


def test_full_triplet_not_event_number_only(tables):
    tables["steps"]["run"][0] = BIG + 2
    with pytest.raises(ValueError, match="unmatched steps identity"):
        capture.audit(tables, 2)


def test_natural_stop_cannot_have_exit(tables):
    tables["steps"]["firstExit"][3] = True
    tables["steps"]["postInside"][3] = False
    with pytest.raises(ValueError, match="first exit census"):
        capture.audit(tables, 2)


def test_bad_rotation(tables):
    tables["events"]["rotation"][1, 0] = 1
    with pytest.raises(ValueError, match="rotation"):
        capture.audit(tables, 2)


def test_root_highest_cycles_and_cli(tmp_path, tables, capsys):
    path, output = tmp_path / "capture.root", tmp_path / "audit.json"
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree("dump/" + name, table)
        bad = {field: value.copy() for field, value in tables["events"].items()}
        bad["complete"][:] = False
        # Old cycles must never mask a partial checkpoint in a newer cycle.
        root.mktree("dump/events", bad)
    with pytest.raises(ValueError, match="complete"):
        capture.audit(capture.read(path), 2)
    with uproot.update(path) as root:
        root.mktree("dump/events", tables["events"])
    loaded = capture.read(path)
    assert loaded["tracks"]["track"].dtype == np.dtype("int64")
    np.testing.assert_array_equal(loaded["tracks"]["track"], tables["tracks"]["track"])
    capture.main([str(path), "--expected-events", "2", "--output", str(output)])
    assert json.loads(output.read_text())["bank_admissible"] is False
    assert json.loads(capsys.readouterr().out)["events"] == 2
    with pytest.raises(SystemExit):
        capture.main([str(path), "--output", str(output)])
