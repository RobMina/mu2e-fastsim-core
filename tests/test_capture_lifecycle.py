"""Synthetic negative lifecycle diagnostics, never physics or bank-admission fixtures."""
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_lifecycle as lifecycle
from test_capture import BIG, empty, tables as physical_tables


def fixture(kind, run=11, subrun=0, first_event=1, size=3):
    sizes = {"events": size, "tracks": 0, "steps": 0,
             "classifications": size if kind == "stacking-rejection" else 0,
             "volumes": 3 * size if kind == "stacking-rejection" else 0}
    result = {name: empty(name, count) for name, count in sizes.items()}
    for name, table in result.items():
        table["run"][:] = run
        table["subrun"][:] = subrun
        repeats = 3 if name == "volumes" else 1
        table["event"][:] = np.repeat(np.arange(size, dtype=np.int64) + first_event, repeats)[:sizes[name]]
    e, t, s, c, v = (result[name] for name in capture.TABLES)
    e["schema"][:] = 1
    if kind == "stacking-rejection":
        for field in "capturePresent statusPresent eventIDsMatch statusClean".split():
            e[field][:] = True
        for field in capture.EVENT:
            e["record" + field.capitalize()][:] = e[field]
        e["geometryTag"][:], e["physicsTag"][:], e["envelope"][:] = "synthetic-box", "not-physics", "box"
        e["rotation"][:] = np.eye(3).ravel()
        e["halfSize"][:] = 1
        e["nClassifications"][:], e["nVolumes"][:], e["nErrors"][:] = 1, 3, 3
        e["errors"][:] = "\n".join(lifecycle.ERRORS)
        e["validationErrors"][:] = "Capture record is incomplete\nCapture record contains errors"
        c["track"][:], c["killed"][:] = 1, True
        v["index"][:] = v["id"][:] = np.tile(np.arange(3, dtype=np.int64), size)
    else:
        for field in lifecycle.VECTORS:
            e[field] = np.empty((size, 0))
        e["g4Status"][:] = np.iinfo(np.int32).max
        e["validationErrors"][:] = ("Missing ComponentCapture: g4run:componentCapture\nMissing StatusG4: g4run\n"
                                    "Capture record is incomplete\nStatusG4 is not clean")
    return result


@pytest.mark.parametrize("kind", lifecycle.KINDS)
def test_success_is_only_negative_diagnostic(kind):
    tables = fixture(kind)
    result = lifecycle.report(tables, kind, 11, 0, 1, 3)
    assert result["diagnostic_only"] and result["negative_fixture_passed"]
    assert not result["bank_admissible"]
    assert result["events"] == 3 and len(result["event_rows"]) == 3
    assert result["rows"] == {name: len(table["event"]) for name, table in tables.items()}
    assert "complete" in result["strict_audit_rejection"]
    assert "deposit" not in json.dumps(result, allow_nan=False)
    assert result["event_rows"][0]["g4Status"] == (0 if kind == "stacking-rejection" else np.iinfo(np.int32).max)
    with pytest.raises(ValueError):
        capture.audit(tables, 3)


@pytest.mark.parametrize("kind", lifecycle.KINDS)
@pytest.mark.parametrize("first", [BIG + 1, np.iinfo(np.int64).max - 2])
def test_full_int64_triplet_and_unordered_rows(kind, first):
    tables = fixture(kind, BIG + 1, BIG + 3, first)
    for name in capture.TABLES:
        tables[name] = {field: value[::-1].copy() for field, value in tables[name].items()}
    result = json.loads(json.dumps(lifecycle.report(tables, kind, BIG + 1, BIG + 3, first, 3), allow_nan=False))
    assert result["event_rows"][0]["run"] == BIG + 1
    assert result["event_rows"][0]["subrun"] == BIG + 3
    assert result["event_rows"][0]["event"] == first
    assert result["event_rows"][2]["event"] == first + 2
    with pytest.raises(ValueError, match="census"):
        lifecycle.report(tables, kind, BIG, BIG + 3, first, 3)


@pytest.mark.parametrize("argument,value", [("run", 12), ("subrun", 1), ("first_event", 2),
                                             ("expected_events", 2), ("expected_events", 4),
                                             ("expected_events", 0), ("expected_events", -1),
                                             ("expected_events", True), ("expected_events", 3.0),
                                             ("run", 11.0), ("run", -1), ("subrun", True),
                                             ("first_event", 1.0), ("first_event", 2**63 - 1),
                                             ("run", 2**63), ("kind", "physical-zero")])
def test_independent_census_arguments(argument, value):
    args = dict(kind="stacking-rejection", run=11, subrun=0, first_event=1, expected_events=3)
    args[argument] = value
    with pytest.raises(ValueError):
        lifecycle.report(fixture("stacking-rejection"), **args)


@pytest.mark.parametrize("kind", lifecycle.KINDS)
@pytest.mark.parametrize("field,value", [("run", 12), ("subrun", 1), ("event", 7), ("event", 2)])
def test_wrong_or_duplicate_event_census(kind, field, value):
    tables = fixture(kind)
    tables["events"][field][0] = value
    with pytest.raises(ValueError, match="census|duplicate"):
        lifecycle.report(tables, kind, 11, 0, 1, 3)


@pytest.mark.parametrize("kind", lifecycle.KINDS)
@pytest.mark.parametrize("field", ["complete", "recordComplete", "bankAdmissible", "recordBankAdmissible"])
def test_all_admission_flags_must_be_false(kind, field):
    tables = fixture(kind)
    tables["events"][field][1] = True
    with pytest.raises(ValueError, match=field):
        lifecycle.report(tables, kind, 11, 0, 1, 3)


@pytest.mark.parametrize("name,field,value", [
    ("events", "capturePresent", False), ("events", "statusPresent", False),
    ("events", "eventIDsMatch", False), ("events", "statusClean", False),
    ("events", "g4Status", 1), ("events", "nG4Tracks", 1),
    ("events", "nTracks", 1), ("events", "nSteps", 1), ("events", "nClassifications", 2),
    ("events", "nVolumes", 4), ("events", "nErrors", 2), ("events", "recordRun", 12),
    ("events", "recordSubrun", 1), ("events", "recordEvent", 2),
    ("events", "overflowSimParticles", True), ("events", "nKilledStepLimit", 1),
    ("events", "nKilledByFieldPropagator", 1), ("events", "schema", 2),
    ("events", "validationErrors", ""), ("events", "geometryTag", ""),
    ("classifications", "index", 1), ("classifications", "track", 2),
    ("classifications", "parent", 1), ("classifications", "killed", False),
    ("classifications", "run", 12), ("classifications", "subrun", 1),
    ("volumes", "id", 1), ("volumes", "index", 1), ("volumes", "id", 3),
    ("volumes", "event", 9),
])
def test_stacking_stale_headers_and_rows(name, field, value):
    tables = fixture("stacking-rejection")
    tables[name][field][0] = value
    with pytest.raises(ValueError):
        lifecycle.report(tables, "stacking-rejection", 11, 0, 1, 3)


@pytest.mark.parametrize("field,value", [
    ("capturePresent", True), ("statusPresent", True), ("eventIDsMatch", True), ("statusClean", True),
    ("recordRun", 11), ("recordSubrun", 1), ("recordEvent", 1), ("nErrors", 1),
    ("errors", "classification_rejection"), ("geometryTag", "stale"), ("physicsTag", "stale"),
    ("envelope", "stale"), ("nTracks", 1), ("nSteps", 1), ("nClassifications", 1), ("nVolumes", 1),
    ("validationErrors", "Missing ComponentCapture: g4run"), ("nG4Tracks", 1),
])
def test_internal_filter_stale_headers(field, value):
    tables = fixture("internal-filter")
    tables["events"][field][0] = value
    with pytest.raises(ValueError):
        lifecycle.report(tables, "internal-filter", 11, 0, 1, 3)


@pytest.mark.parametrize("name", capture.TABLES[1:])
def test_internal_filter_cannot_keep_rows_even_with_updated_counts(name):
    tables = fixture("internal-filter")
    tables[name] = empty(name, 1)
    tables[name]["run"][:], tables[name]["event"][:] = 11, 1
    tables["events"]["n" + name.capitalize()][0] = 1
    with pytest.raises(ValueError, match="stale.*rows"):
        lifecycle.report(tables, "internal-filter", 11, 0, 1, 3)


@pytest.mark.parametrize("field,width", lifecycle.VECTORS.items())
def test_internal_filter_has_no_geometry_metadata(field, width):
    tables = fixture("internal-filter")
    tables["events"][field] = np.zeros((3, width))
    with pytest.raises(ValueError, match=field):
        lifecycle.report(tables, "internal-filter", 11, 0, 1, 3)


@pytest.mark.parametrize("errors", ["", "\n".join(lifecycle.ERRORS[:2]),
    "\n".join((*lifecycle.ERRORS, "reclassification_unsupported")),
    "\n".join((lifecycle.ERRORS[0], lifecycle.ERRORS[0], lifecycle.ERRORS[2])),
    "\n".join((*lifecycle.ERRORS[:2], "unrelated_error")), "\n".join(lifecycle.ERRORS) + "\n"])
def test_error_pollution_even_with_updated_count(errors):
    tables = fixture("stacking-rejection")
    tables["events"]["errors"][0] = errors
    tables["events"]["nErrors"][0] = len(errors.split("\n")) if errors else 0
    with pytest.raises(ValueError, match="recorder errors"):
        lifecycle.report(tables, "stacking-rejection", 11, 0, 1, 3)


def test_error_order_does_not_matter():
    tables = fixture("stacking-rejection")
    tables["events"]["errors"][:] = "\n".join(reversed(lifecycle.ERRORS))
    assert lifecycle.report(tables, "stacking-rejection", 11, 0, 1, 3)["negative_fixture_passed"]


@pytest.mark.parametrize("update_index", [False, True])
def test_duplicate_classification_even_with_updated_count(update_index):
    tables = fixture("stacking-rejection")
    tables["classifications"] = {field: value[[0, 0, 1, 2]].copy()
                                 for field, value in tables["classifications"].items()}
    tables["events"]["nClassifications"][0] = 2
    if update_index:
        tables["classifications"]["index"][1] = 1
    with pytest.raises(ValueError, match="duplicate|census"):
        lifecycle.report(tables, "stacking-rejection", 11, 0, 1, 3)


@pytest.mark.parametrize("name,field,dtype", [
    ("events", "run", float), ("events", "subrun", float), ("events", "event", float),
    ("events", "recordEvent", float), ("events", "nErrors", float), ("events", "nTracks", np.int32),
    ("events", "complete", np.int64), ("events", "nG4Tracks", float),
    ("classifications", "index", float), ("classifications", "track", float),
    ("classifications", "parent", float), ("classifications", "killed", np.int64),
    ("volumes", "id", float), ("steps", "index", float), ("tracks", "track", float),
])
def test_relevant_dtypes(name, field, dtype):
    tables = fixture("stacking-rejection", BIG + 1, BIG + 2, BIG + 3)
    tables[name][field] = tables[name][field].astype(dtype)
    with pytest.raises(ValueError, match="must be"):
        lifecycle.report(tables, "stacking-rejection", BIG + 1, BIG + 2, BIG + 3, 3)


@pytest.mark.parametrize("mode", ["missing_table", "missing_field", "short_column", "matrix_identity",
                                  "nonfinite", "nonstrings", "missing_event", "empty"])
def test_schema_and_missing_census(mode):
    tables = fixture("stacking-rejection")
    if mode == "missing_table":
        del tables["steps"]
    elif mode == "missing_field":
        del tables["volumes"]["index"]
    elif mode == "short_column":
        tables["events"]["run"] = tables["events"]["run"][:2]
    elif mode == "matrix_identity":
        tables["events"]["run"] = tables["events"]["run"][:, None]
    elif mode == "nonfinite":
        tables["events"]["cpuTime"][0] = np.nan
    elif mode == "nonstrings":
        tables["events"]["errors"][0] = None
    elif mode == "missing_event":
        tables["events"] = {field: value[:2] for field, value in tables["events"].items()}
    else:
        tables = {name: {field: value[:0] for field, value in table.items()} for name, table in tables.items()}
    with pytest.raises(ValueError):
        lifecycle.report(tables, "stacking-rejection", 11, 0, 1, 3)


@pytest.mark.parametrize("kind", lifecycle.KINDS)
def test_physical_zero_is_not_a_negative_fixture(physical_tables, kind):
    tables = {name: {field: value[table["event"] == BIG + 1].copy() for field, value in table.items()}
              for name, table in physical_tables.items()}
    assert capture.audit(tables, 1)["component_deposit"] == 0
    with pytest.raises(ValueError):
        lifecycle.report(tables, kind, BIG + 1, BIG + 1, BIG + 1, 1)


def test_strict_audit_must_actually_reject(monkeypatch):
    monkeypatch.setattr(capture, "audit", lambda tables, expected_events: {"accepted": True})
    with pytest.raises(ValueError, match="unexpectedly accepted"):
        lifecycle.report(fixture("stacking-rejection"), "stacking-rejection", 11, 0, 1, 3)


@pytest.mark.parametrize("kind", lifecycle.KINDS)
def test_cli_highest_cycles_and_no_report_on_failure(tmp_path, capsys, kind):
    path, output = tmp_path / "capture.root", tmp_path / "report.json"
    tables = fixture(kind)
    for field in lifecycle.VECTORS:
        tables["events"][field] = ak.from_regular(ak.Array(tables["events"][field]))
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree("dump/" + name, table)
    args = [str(path), "--kind", kind, "--run", "11", "--expected-events", "3", "--output", str(output)]
    lifecycle.main(args)
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["negative_fixture_passed"] and not result["bank_admissible"]
    original = output.read_bytes()
    tables["events"]["complete"][0] = True
    with uproot.update(path) as root:
        root.mktree("dump/events", tables["events"])
    with pytest.raises(ValueError, match="complete"):
        lifecycle.main(args)
    assert output.read_bytes() == original
    output.unlink()
    with pytest.raises(ValueError, match="complete"):
        lifecycle.main(args)
    assert not output.exists() and capsys.readouterr().out == ""
    with pytest.raises(SystemExit):
        lifecycle.main([str(path), "--kind", kind, "--run", "11", "--output", str(output)])


@pytest.mark.parametrize("name", ["classifications", "volumes"])
@pytest.mark.parametrize("field", capture.EVENT)
def test_large_child_identity_must_match_full_triplet(name, field):
    tables = fixture("stacking-rejection", BIG + 1, BIG + 1, BIG + 1)
    tables[name][field][0] = BIG
    with pytest.raises(ValueError, match="unmatched"):
        lifecycle.report(tables, "stacking-rejection", BIG + 1, BIG + 1, BIG + 1, 3)


@pytest.mark.parametrize("field,value", [("g4Status", 0), ("nG4Tracks", -1), ("overflowSimParticles", True),
    ("nKilledStepLimit", 1), ("nKilledByFieldPropagator", 1), ("cpuTime", 1.), ("realTime", 1.)])
def test_missing_status_uses_only_pinned_default_values(field, value):
    tables = fixture("internal-filter")
    tables["events"][field][0] = value
    with pytest.raises(ValueError, match="default mismatch|stale"):
        lifecycle.report(tables, "internal-filter", 11, 0, 1, 3)


@pytest.fixture
def recovery_tables(physical_tables):
    positive = {name: {field: value.copy() for field, value in table.items()}
                for name, table in physical_tables.items()}
    for name, table in positive.items():
        original_event = table["event"].copy()
        table["run"][:] = 13
        table["subrun"][:] = 0
        table["event"][:] = np.where(original_event == BIG + 1, 2, 4)
        for field in ("track", "parent"):
            if field in table:
                table[field][:] = np.where(table[field] > 0, table[field] - BIG, 0)
    for field in capture.EVENT:
        positive["events"]["record" + field.capitalize()][:] = positive["events"][field]
    negative = fixture("stacking-rejection", run=13, size=2)
    for table in negative.values():
        table["event"][:] = 2 * table["event"] - 1
    negative["events"]["recordEvent"][:] = negative["events"]["event"]
    return {name: {field: np.concatenate((negative[name][field], value)) for field, value in table.items()}
            for name, table in positive.items()}


def test_recovery_checks_all_four_fixed_events(recovery_tables):
    result = lifecycle.recovery(recovery_tables, 13)
    assert result["recovery_fixture_passed"] and not result["bank_admissible"]
    assert result["accepted_events"] == [2, 4]
    assert [row["event"] for row in result["rejected_events"]["event_rows"]] == [1, 3]
    assert result["accepted_audit"]["events"] == 2
    assert "complete" in result["strict_audit_rejection"]


@pytest.mark.parametrize("name,field,row,value", [
    ("events", "event", 3, 2), ("events", "complete", 0, True),
    ("events", "complete", 3, False), ("events", "errors", 3, "reclassification_unsupported"),
    ("steps", "event", 0, 5), ("steps", "run", 0, 12),
    ("classifications", "killed", 2, True), ("tracks", "run", 0, 12),
])
def test_recovery_rejects_wrong_census_or_stale_state(recovery_tables, name, field, row, value):
    recovery_tables[name][field][row] = value
    with pytest.raises(ValueError):
        lifecycle.recovery(recovery_tables, 13)


@pytest.mark.parametrize("run,first,count", [(12, 1, 4), (13, 2, 4), (13, 1, 3), (13, 1, 0)])
def test_recovery_fixed_external_census(recovery_tables, run, first, count):
    with pytest.raises(ValueError):
        lifecycle.recovery(recovery_tables, run, first_event=first, expected_events=count)


def test_recovery_cli(tmp_path, recovery_tables, capsys):
    path, output = tmp_path / "mixed.root", tmp_path / "recovery.json"
    with uproot.recreate(path) as root:
        for name, table in recovery_tables.items():
            root.mktree("dump/" + name, table)
    args = [str(path), "--kind", "stacking-recovery", "--run", "13", "--expected-events", "4",
            "--output", str(output)]
    lifecycle.main(args)
    assert json.loads(output.read_text())["recovery_fixture_passed"]
    capsys.readouterr()
    output.unlink()
    recovery_tables["events"]["complete"][3] = False
    with uproot.update(path) as root:
        root.mktree("dump/events", recovery_tables["events"])
    with pytest.raises(ValueError):
        lifecycle.main(args)
    assert not output.exists()


@pytest.mark.parametrize("ids", [np.array([1, 1], dtype=np.int64), np.array([3, 1], dtype=np.int64),
    np.array([1., 3.]), np.array([1, 3], dtype=np.int32), np.array([1], dtype=np.int64),
    np.array([-1, 3], dtype=np.int64), np.array([2, 3], dtype=np.int64)])
def test_explicit_negative_census_cannot_coerce_or_hide_events(ids):
    tables = fixture("stacking-rejection", size=2)
    with pytest.raises(ValueError):
        lifecycle.report(tables, "stacking-rejection", 11, 0, 1, 2, event_ids=ids)


def test_recovery_full_int64_identity(recovery_tables):
    for name, table in recovery_tables.items():
        table["run"][:] = BIG + 1
        table["subrun"][:] = BIG + 3
    recovery_tables["events"]["recordRun"][:] = BIG + 1
    recovery_tables["events"]["recordSubrun"][:] = BIG + 3
    assert lifecycle.recovery(recovery_tables, BIG + 1, BIG + 3)["recovery_fixture_passed"]
    recovery_tables["steps"]["subrun"][0] = BIG + 2
    with pytest.raises(ValueError, match="unmatched"):
        lifecycle.recovery(recovery_tables, BIG + 1, BIG + 3)
