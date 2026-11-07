"""Synthetic callback-table regressions only; no executed C++ callbacks or physics transport."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_callbacks as callbacks
from test_capture import BIG, copy_state, empty


def set_state(table, prefix, z, pz, energy, time, volume, status=-1):
    for field in capture.FLOATS:
        table[prefix + field][:] = 0.
    for field, value in (("z", z), ("lz", z), ("pz", pz), ("lpz", pz), ("energy", energy),
                         ("kinetic", energy), ("time", time), ("weight", 1.), ("volume", volume),
                         ("stepStatus", status)):
        table[prefix + field][:] = value
    table[prefix + "material"][:] = np.where(volume < 0, "", np.where(volume == 0, "G4_Galactic", "G4_POLYSTYRENE"))
    table[prefix + "placement"][:] = np.where(volume < 0, "", np.where(volume == 0, "/0:0", "/0:0/1:0"))


def fixture(run=11, subrun=0):
    events = np.arange(1, 14, dtype=np.int64)
    present = events != 12
    result = {}
    for name, count in (("events", 1), ("tracks", 2), ("steps", 3), ("classifications", 2), ("volumes", 2)):
        counts = np.ones(13, dtype=np.int64) if name == "events" else np.where(present, count, 0)
        if name == "classifications":
            counts[10] = 3
        table = empty(name, counts.sum())
        table["run"][:] = run
        table["subrun"][:] = subrun
        table["event"][:] = np.repeat(events, counts)
        if name != "events":
            table["index"][:] = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        result[name] = table
    e, t, s, c, v = (result[name] for name in capture.TABLES)
    e["schema"][:] = 1
    e["recordComplete"][:] = np.isin(events, [5, 6, 13])
    e["capturePresent"][:] = e["eventIDsMatch"][:] = present
    for field in capture.EVENT:
        e["record" + field.capitalize()][:] = np.where(present, e[field], 0)
    e["g4Status"][:] = np.iinfo(np.int32).max
    for field, value in (("geometryTag", "diagnostic-constructed-objects-v1"),
                         ("physicsTag", "NO-PHYSICS-TRANSPORT-v1"), ("envelope", "CallbackFixtureEnvelope")):
        e[field][:] = np.where(present, value, "")
    for field, row in (("translation", np.zeros(3)), ("halfSize", np.full(3, 50.)), ("rotation", np.eye(3).ravel())):
        e[field] = ak.unflatten(np.tile(row, 12), np.where(present, len(row), 0))
    e["nTracks"][:] = e["nVolumes"][:] = np.where(present, 2, 0)
    e["nSteps"][:] = np.where(present, 3, 0)
    e["nClassifications"][:] = np.where(present, np.where(events == 11, 3, 2), 0)
    e["errors"][:] = np.array([
        "boundary_born_secondary_ownership_unresolved", "boundary_born_secondary_ownership_unresolved",
        "secondary_born_outside_envelope\nboundary_born_secondary_ownership_unresolved\ntrack_never_entered_component",
        "secondary_born_outside_envelope\nboundary_born_secondary_ownership_unresolved\ntrack_never_entered_component",
        "", "", "nonterminal_or_kill_secondaries_status", "nonterminal_or_kill_secondaries_status",
        "nonterminal_or_kill_secondaries_status", "nonterminal_or_kill_secondaries_status",
        "reclassification_unsupported\nclassification_tracking_census_mismatch", "", "",
    ], dtype=object)
    e["nErrors"][:] = [1, 1, 3, 3, 0, 0, 1, 1, 1, 1, 2, 0, 0]
    e["validationErrors"][:] = "Missing StatusG4: fixture\nCapture record is incomplete\nStatusG4 is not clean\nCapture record contains errors\nStatusG4/captured track census mismatch"
    e["validationErrors"][e["recordComplete"]] = "Missing StatusG4: fixture\nStatusG4 is not clean\nStatusG4/captured track census mismatch"
    e["validationErrors"][~present] = "Missing ComponentCapture: fixture:componentCapture\nMissing StatusG4: fixture\nCapture record is incomplete\nStatusG4 is not clean"

    world = v["index"] == 0
    v["id"][:] = v["index"]
    v["name"][:] = np.where(world, "CallbackFixtureWorld", "CallbackFixtureEnvelope")
    v["logical"][:] = np.where(world, "CallbackFixtureWorldLogical", "CallbackFixtureEnvelopeLogical")
    v["motherLogical"][:] = np.where(world, "", "CallbackFixtureWorldLogical")
    v["material"][:] = np.where(world, "G4_Galactic", "G4_POLYSTYRENE")
    v["density"][:] = np.where(world, 1e-25, 1.06)
    v["solid"][:] = "G4Box"
    v["rotation"][:] = np.eye(3).ravel()

    n = s["index"]
    s["number"][:] = n + 1
    s["track"][:] = 1
    s["secondaries"][:] = n == np.where(s["event"] <= 4, 2, 1)
    s["preInside"][:] = n > 0
    s["postInside"][:] = n < 2
    s["firstExit"][:] = n == 2
    s["length"][:] = np.array([0.001, 50., 50.])[n]
    set_state(s, "pre_", np.array([-50.001, -50., 0.])[n], 10., 10., n.astype(float), np.where(n == 0, 0, 1), 7)
    set_state(s, "post_", np.array([-50., 0., 50.])[n], 10., 10., n + 1., np.where(n == 2, 0, 1), np.where(n == 1, 4, 1))

    child = t["index"] == 1
    outside = child & np.isin(t["event"], [3, 4])
    invalid_status = child & np.isin(t["event"], [7, 8, 9, 10])
    t["track"][:] = t["index"] + 1
    t["parent"][:] = child
    t["pdg"][:] = 22
    t["steps"][:] = np.where(child, 0, 3)
    t["secondaries"][:] = ~child
    t["status"][:] = np.where(child, np.where(invalid_status, t["event"] - 4, 2), 0)
    t["reason"][:] = np.where(child, np.where(invalid_status, 3, 1), 2)
    t["entered"][:] = ~outside
    t["ended"][:] = True
    z = np.where(t["event"] <= 4, 50., 0.)
    time = np.where(t["event"] <= 4, 3., 2.)
    pz = np.where(child, np.where(np.isin(t["event"], [1, 3]), -1., 1.), 10.)
    energy = np.where(child, 1., 10.)
    volume = np.where(~child | outside, 0, 1)
    set_state(t, "begin_", np.where(child, z, -50.001), pz, energy, np.where(child, time, 0.), volume)
    set_state(t, "end_", np.where(child, z, 50.), pz, energy, np.where(child, time, 3.), volume, np.where(child, -1, 1))
    set_state(t, "entrance_", np.where(outside, 0., np.where(child, z, -50.)), np.where(outside, 0., pz),
              np.where(outside, 0., energy), np.where(outside, 0., np.where(child, time, 1.)),
              np.where(outside, -1, 1), np.where(child, -1, 1))

    child = c["index"] > 0
    c["track"][:] = np.where(child, 2, 1)
    c["parent"][:] = child
    c["pdg"][:] = 22
    c["status"][:] = np.where(child & (c["event"] == 6), 2, 0)
    tr = 2 * (c["event"] - 1 - (c["event"] > 12)) + child
    copy_state(c, "state_", t, "begin_", tr)
    return result


def test_all_thirteen_cases_are_diagnostic_only():
    tables = fixture()
    result = callbacks.report(tables, 11)
    assert result["diagnostic_only"] and result["callback_fixture_passed"]
    assert not result["bank_admissible"] and not result["physical_transport"]
    assert result["rows"] == dict(events=13, tracks=24, steps=36, classifications=25, volumes=24)
    assert result["events"] == len(result["cases"]) == 13
    cases = result["cases"]
    assert [row["event"] for row in cases] == list(range(1, 14))
    assert len({row["case"] for row in cases}) == 13
    assert np.array_equal(np.flatnonzero(tables["events"]["recordComplete"]) + 1, [5, 6, 13])
    assert cases[11]["capturePresent"] is False and cases[11]["nTracks"] == 0
    assert cases[12]["recordComplete"] and cases[12]["errors"] == ""
    assert result["strict_audit_rejection"] == "events: malformed translation"
    with pytest.raises(ValueError, match="^events: malformed translation$"):
        capture.audit(tables, 13)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("run,subrun", [(BIG + 1, BIG + 3), (2**63 - 1, 2**63 - 2)])
def test_large_int64_and_unordered_rows(run, subrun):
    tables = fixture(run, subrun)
    tables = {name: {field: value[::-1] for field, value in table.items()} for name, table in tables.items()}
    result = json.loads(json.dumps(callbacks.report(tables, run, subrun)))
    assert result["cases"][0]["run"] == run and result["cases"][-1]["subrun"] == subrun
    with pytest.raises(ValueError, match="census"):
        callbacks.report(tables, run - 1, subrun)


@pytest.mark.parametrize("field,value", [("run", -1), ("run", 11.), ("run", True), ("run", 2**63),
                                         ("subrun", -1), ("subrun", False), ("subrun", 0.)])
def test_invalid_arguments(field, value):
    args = dict(run=11, subrun=0)
    args[field] = value
    with pytest.raises(ValueError):
        callbacks.report(fixture(), **args)


@pytest.mark.parametrize("field", ["complete", "bankAdmissible", "recordBankAdmissible", "statusPresent", "statusClean",
                                   "overflowSimParticles"])
def test_no_admission_or_fake_status(field):
    tables = fixture()
    tables["events"][field][4] = True
    with pytest.raises(ValueError, match=field):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("event", range(1, 14))
def test_record_completeness_is_case_specific(event):
    tables = fixture()
    tables["events"]["recordComplete"][event - 1] ^= True
    with pytest.raises(ValueError, match="recordComplete"):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("field,value", [("g4Status", 0), ("nG4Tracks", 2), ("nKilledStepLimit", 1),
    ("nKilledByFieldPropagator", 1), ("cpuTime", 1.), ("realTime", 1.), ("nTracks", 1), ("nSteps", 2),
    ("nClassifications", 1), ("nVolumes", 1), ("nErrors", 0), ("recordRun", 12), ("recordSubrun", 1),
    ("recordEvent", 2), ("schema", 2), ("geometryTag", "physical"), ("physicsTag", "transport"),
    ("envelope", "wrong"), ("capturePresent", False), ("eventIDsMatch", False)])
def test_event_pollution(field, value):
    tables = fixture()
    tables["events"][field][0] = value
    with pytest.raises(ValueError):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("event", [1, 3, 5, 7, 11, 12, 13])
@pytest.mark.parametrize("pollution", ["extra", "duplicate", "missing", "trailing-newline"])
def test_exact_errors_even_with_updated_census(event, pollution):
    tables = fixture()
    errors = tables["events"]["errors"][event - 1]
    errors = {"extra": errors + "\nunrelated_error", "duplicate": errors + "\n" + errors,
              "missing": "different_error", "trailing-newline": errors + "\n"}[pollution]
    tables["events"]["errors"][event - 1] = errors
    tables["events"]["nErrors"][event - 1] = errors.count("\n") + bool(errors)
    with pytest.raises(ValueError, match="errors"):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("event", [1, 5, 11, 12, 13])
def test_validation_message_order_and_source(event):
    tables = fixture()
    tables["events"]["validationErrors"][event - 1] = "\n".join(reversed(tables["events"]["validationErrors"][event - 1].split("\n")))
    with pytest.raises(ValueError, match="validationErrors"):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("name,field,event,index,value", [
    ("tracks", "begin_pz", 1, 1, 1.), ("tracks", "begin_lpz", 3, 1, 1.),
    ("tracks", "begin_volume", 3, 1, 1), ("tracks", "begin_volume", 1, 1, 0),
    ("tracks", "begin_material", 4, 1, "G4_POLYSTYRENE"), ("tracks", "begin_z", 2, 1, 49.99),
    ("tracks", "begin_time", 5, 1, 3.), ("tracks", "begin_energy", 5, 1, 10.),
    ("tracks", "begin_mass", 5, 1, 1.), ("tracks", "begin_charge", 5, 1, 1.),
    ("tracks", "begin_weight", 5, 1, 2.), ("tracks", "end_z", 5, 0, 51.),
    ("tracks", "end_stepStatus", 5, 0, -1), ("tracks", "entrance_z", 3, 1, 50.),
    ("tracks", "entrance_volume", 3, 1, 0), ("tracks", "entrance_weight", 4, 1, 0.),
    ("tracks", "entered", 3, 1, True), ("tracks", "ended", 5, 1, False),
    ("tracks", "status", 7, 1, 2), ("tracks", "status", 8, 1, 3),
    ("tracks", "status", 9, 1, 4), ("tracks", "status", 10, 1, 5),
    ("tracks", "reason", 7, 1, 1), ("tracks", "status", 5, 0, 2),
    ("tracks", "steps", 5, 1, 1), ("tracks", "secondaries", 5, 0, 0),
    ("tracks", "track", 5, 1, 1), ("tracks", "parent", 5, 1, 0),
    ("tracks", "pdg", 5, 1, 11), ("tracks", "creator", 5, 1, "fakeProcess"),
    ("tracks", "stopProcess", 5, 1, "fakeProcess"),
    ("steps", "post_z", 1, 0, -49.), ("steps", "post_lz", 2, 2, 49.),
    ("steps", "post_stepStatus", 3, 0, 4), ("steps", "post_stepStatus", 4, 1, 1),
    ("steps", "pre_stepStatus", 5, 1, 1), ("steps", "trackStatus", 6, 2, 2),
    ("steps", "preInside", 7, 0, True), ("steps", "postInside", 8, 2, True),
    ("steps", "firstExit", 9, 1, True), ("steps", "secondaries", 1, 1, 1),
    ("steps", "number", 5, 2, 2), ("steps", "length", 5, 1, 49.),
    ("steps", "deposit", 5, 1, 1.), ("steps", "nonIonizingDeposit", 5, 1, 1.),
    ("steps", "pre_material", 5, 0, "G4_POLYSTYRENE"), ("steps", "post_volume", 5, 2, 1),
    ("steps", "post_placement", 5, 2, "/0:0/1:0"),
    ("classifications", "track", 11, 2, 1), ("classifications", "status", 6, 1, 0),
    ("classifications", "status", 5, 1, 2), ("classifications", "killed", 5, 1, True),
    ("classifications", "state_pz", 1, 1, 1.), ("classifications", "state_stepStatus", 5, 1, 7),
    ("volumes", "name", 5, 1, "wrong"), ("volumes", "material", 5, 0, "G4_AIR"),
    ("volumes", "id", 5, 1, 0), ("volumes", "copy", 5, 1, 1),
    ("volumes", "solid", 5, 1, "G4Tubs"), ("volumes", "motherLogical", 5, 1, "wrong"),
    ("volumes", "logical", 5, 1, "wrong"), ("volumes", "density", 5, 1, 0.),
])
def test_corrupt_callback_rows(name, field, event, index, value):
    tables = fixture()
    table = tables[name]
    table[field][(table["event"] == event) & (table["index"] == index)] = value
    with pytest.raises(ValueError):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("name", capture.TABLES[1:])
@pytest.mark.parametrize("event", [5, 11, 13])
def test_missing_child_rows_even_with_updated_counts(name, event):
    tables = fixture()
    table = tables[name]
    keep = ~((table["event"] == event) & (table["index"] == 1))
    tables[name] = {field: value[keep] for field, value in table.items()}
    tables["events"]["n" + name.capitalize()][event - 1] -= 1
    with pytest.raises(ValueError, match="census"):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("name", capture.TABLES[1:])
def test_discard_has_no_stale_rows_even_with_updated_counts(name):
    tables = fixture()
    table = tables[name]
    table["event"][table["event"] == 13] = 12
    tables["events"]["n" + name.capitalize()][11:13] = tables["events"]["n" + name.capitalize()][12:10:-1]
    with pytest.raises(ValueError, match="census"):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("field,value", [("recordRun", 11), ("recordSubrun", 1), ("recordEvent", 12),
    ("capturePresent", True), ("eventIDsMatch", True), ("geometryTag", "stale"), ("physicsTag", "stale"),
    ("envelope", "stale"), ("nErrors", 1), ("recordComplete", True)])
def test_discard_has_no_stale_header(field, value):
    tables = fixture()
    tables["events"][field][11] = value
    with pytest.raises(ValueError):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("name", capture.TABLES)
@pytest.mark.parametrize("field", capture.EVENT)
def test_exact_triplet_matching(name, field):
    tables = fixture(BIG + 1, BIG + 3)
    tables[name][field][0] += 1
    with pytest.raises(ValueError):
        callbacks.report(tables, BIG + 1, BIG + 3)


@pytest.mark.parametrize("name,field,dtype", [
    ("events", "run", float), ("events", "subrun", float), ("events", "event", float),
    ("events", "nTracks", np.int32), ("events", "nErrors", float), ("events", "recordRun", float),
    ("events", "capturePresent", np.int64), ("events", "nG4Tracks", float),
    ("tracks", "index", float), ("tracks", "track", np.int32), ("tracks", "parent", float),
    ("tracks", "steps", float), ("tracks", "begin_volume", float), ("tracks", "ended", np.int32),
    ("steps", "number", float), ("steps", "secondaries", float), ("classifications", "index", float),
    ("volumes", "id", float), ("volumes", "copy", float),
])
def test_strict_dtypes(name, field, dtype):
    tables = fixture(BIG + 1, BIG + 3)
    tables[name][field] = tables[name][field].astype(dtype)
    with pytest.raises(ValueError, match="must be"):
        callbacks.report(tables, BIG + 1, BIG + 3)


@pytest.mark.parametrize("field", ["translation", "rotation", "halfSize"])
@pytest.mark.parametrize("mode", ["stale-missing", "missing-present", "wrong-width", "integer", "nonfinite", "wrong-value"])
def test_mixed_metadata_validation(field, mode):
    tables = fixture()
    values = tables["events"][field]
    width = callbacks.VECTORS[field]
    flat = ak.to_numpy(ak.flatten(values))
    counts = ak.to_numpy(ak.num(values))
    if mode == "stale-missing":
        tables["events"][field] = np.zeros((13, width), dtype=float)
    elif mode == "missing-present":
        counts[12] = 0
        tables["events"][field] = ak.unflatten(flat[:-width], counts)
    elif mode == "wrong-width":
        counts[0] -= 1
        tables["events"][field] = ak.unflatten(flat[1:], counts)
    elif mode == "integer":
        tables["events"][field] = ak.values_astype(values, np.int64)
    else:
        flat[0] = np.nan if mode == "nonfinite" else 2.
        tables["events"][field] = ak.unflatten(flat, counts)
    with pytest.raises(ValueError, match=field):
        callbacks.report(tables, 11)


@pytest.mark.parametrize("mode", ["missing-table", "missing-field", "short-column", "matrix-event", "empty", "missing-event", "extra-event", "nonstring"])
def test_malformed_schema_or_census(mode):
    tables = fixture()
    if mode == "missing-table":
        del tables["steps"]
    elif mode == "missing-field":
        del tables["tracks"]["begin_z"]
    elif mode == "short-column":
        tables["events"]["run"] = tables["events"]["run"][:-1]
    elif mode == "matrix-event":
        tables["events"]["event"] = tables["events"]["event"][:, None]
    elif mode == "empty":
        tables = {name: {field: value[:0] for field, value in table.items()} for name, table in tables.items()}
    elif mode == "missing-event":
        tables["events"] = {field: value[:-1] for field, value in tables["events"].items()}
    elif mode == "extra-event":
        tables["events"]["event"][-1] = 14
    else:
        tables["events"]["errors"][0] = None
    with pytest.raises(ValueError):
        callbacks.report(tables, 11)


def test_strict_audit_must_reject(monkeypatch):
    monkeypatch.setattr(capture, "audit", lambda *args: {})
    with pytest.raises(ValueError, match="unexpectedly accepted"):
        callbacks.report(fixture(), 11)


def test_report_does_not_change_inputs():
    tables = fixture()
    before = {name: {field: ak.copy(value) if isinstance(value, ak.Array) else value.copy()
                     for field, value in table.items()} for name, table in tables.items()}
    callbacks.report(tables, 11)
    for name in capture.TABLES:
        for field in tables[name]:
            if isinstance(tables[name][field], ak.Array):
                assert ak.almost_equal(tables[name][field], before[name][field])
            else:
                np.testing.assert_array_equal(tables[name][field], before[name][field])


def test_cli_no_output_on_failure_and_required_run(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    tables = fixture()
    monkeypatch.setattr(capture, "read", lambda path: tables)
    args = [str(tmp_path / "synthetic.root"), "--run", "11", "--output", str(output)]
    callbacks.main(args)
    original = output.read_bytes()
    result = json.loads(original)
    assert result == json.loads(capsys.readouterr().out)
    assert result["callback_fixture_passed"] and not result["physical_transport"]
    tables["tracks"]["status"][-1] = 3
    with pytest.raises(ValueError):
        callbacks.main(args)
    assert output.read_bytes() == original
    output.unlink()
    with pytest.raises(ValueError):
        callbacks.main(args)
    assert not output.exists() and capsys.readouterr().out == ""
    with pytest.raises(SystemExit):
        callbacks.main([str(tmp_path / "synthetic.root"), "--output", str(output)])


def test_cli_root_highest_cycles_and_mixed_metadata(tmp_path, capsys):
    path, output = tmp_path / "capture.root", tmp_path / "report.json"
    tables = fixture(BIG + 1, BIG + 3)
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree("dump/" + name, table)
    args = [str(path), "--run", str(BIG + 1), "--subrun", str(BIG + 3), "--output", str(output)]
    callbacks.main(args)
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["cases"][-1]["run"] == BIG + 1
    original = output.read_bytes()
    tables["events"]["capturePresent"][-1] = False
    with uproot.update(path) as root:
        root.mktree("dump/events", tables["events"])
    with pytest.raises(ValueError, match="capturePresent"):
        callbacks.main(args)
    assert output.read_bytes() == original
    output.unlink()
    with pytest.raises(ValueError):
        callbacks.main(args)
    assert not output.exists() and capsys.readouterr().out == ""
