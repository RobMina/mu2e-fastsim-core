"""Check negative capture lifecycle fixtures; passing never admits a response-bank event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import capture
from .saved_g4 import frame, match, require, unique

KINDS = ("stacking-rejection", "internal-filter")
CLI_KINDS = (*KINDS, "stacking-recovery")
ERRORS = ("classification_rejection", "expected_one_incident_primary",
          "classification_tracking_census_mismatch")
BOOL = set("complete bankAdmissible recordComplete recordBankAdmissible capturePresent statusPresent eventIDsMatch "
           "statusClean overflowSimParticles entered ended preInside postInside firstExit killed".split())
STR = set("geometryTag physicsTag envelope errors validationErrors creator stopProcess name logical motherLogical "
          "material solid placement process".split())
FLOAT = set(capture.FLOATS + "deposit nonIonizingDeposit length density cpuTime realTime".split())
VECTORS = {"translation": 3, "rotation": 9, "halfSize": 3}


def _integer(value, label, minimum=0):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f"{label} must be an integer")
    require(minimum <= value <= np.iinfo(np.int64).max, f"{label} outside int64 range")
    return int(value)


def _schema(tables, kind):
    """Validate flat schema columns before using any identities, flags or diagnostic strings."""
    result = {}
    for name in capture.TABLES:
        require(name in tables, f"missing table {name}")
        fields = [*capture.EVENT, *capture.FIELDS[name].split(),
                  *(prefix + field for prefix in capture.STATES.get(name, ()) for field in capture.STATE)]
        require(set(fields) <= set(tables[name]), f"missing {name} fields: {set(fields) - set(tables[name])}")
        table = {key: np.asarray(tables[name][key]) for key in fields}
        require(table["event"].ndim == 1, f"{name}: malformed event")
        size = len(table["event"])
        for key, value in table.items():
            require(value.ndim > 0 and len(value) == size, f"{name}: malformed {key}")
            leaf = key.split("_")[-1]
            if key in VECTORS:
                width = 0 if name == "events" and kind == "internal-filter" else VECTORS[key]
                require(value.ndim == 2 and (size == 0 or value.shape[1] == width),
                        f"{name}: invalid {key} shape")
                require(value.dtype.kind == "f" and np.isfinite(value).all(), f"{name}: invalid {key} values")
                continue
            require(value.ndim == 1, f"{name}: malformed {key}")
            if key in capture.INT64 or leaf == "volume":
                require(value.dtype.kind == "i" and value.dtype.itemsize == 8, f"{name}: {key} must be int64")
            elif key in BOOL:
                require(value.dtype.kind == "b", f"{name}: {key} must be boolean")
            elif leaf in STR:
                require(size == 0 or (value.dtype.kind in "OU" and
                        pd.api.types.infer_dtype(value, skipna=False) == "string"),
                        f"{name}: {key} must be strings")
            elif leaf in FLOAT:
                require(value.dtype.kind == "f" and np.isfinite(value).all(), f"{name}: invalid {key} values")
            else:
                require(value.dtype.kind == "i", f"{name}: {key} must be integer")
        require((frame(table, capture.EVENT) >= 0).all().all(), f"{name}: negative event identity")
        result[name] = table
    return result


def report(tables, kind, run, subrun, first_event, expected_events, *, event_ids=None):
    """Require an independent exact census and a known negative lifecycle, never physics success."""
    require(kind in KINDS, "unknown negative fixture kind")
    run, subrun = _integer(run, "run"), _integer(subrun, "subrun")
    first_event = _integer(first_event, "first_event")
    expected_events = _integer(expected_events, "positive expected_events", 1)
    require(first_event + expected_events - 1 <= np.iinfo(np.int64).max, "event interval exceeds int64 range")
    tables = _schema(tables, kind)
    e, t, s, c, v = (tables[name] for name in capture.TABLES)
    events = unique(e, capture.EVENT, "event").assign(_event=np.arange(len(e["event"]), dtype=np.int64))
    require(len(events) == expected_events, "expected event census mismatch")
    require(np.all(e["run"] == run) and np.all(e["subrun"] == subrun), "expected run/subrun census mismatch")
    expected = np.arange(expected_events, dtype=np.int64) + first_event
    if event_ids is not None:
        expected = np.asarray(event_ids)
        require(expected.ndim == 1 and expected.dtype == np.dtype("int64") and len(expected) == expected_events and
                np.all(expected >= 0) and np.all(np.diff(expected) > 0), "invalid explicit event census")
        require(expected[0] == first_event, "explicit first event mismatch")
    require(np.array_equal(np.sort(e["event"]), expected), "expected event census mismatch")
    require(np.all(e["schema"] == 1), "unsupported capture schema")
    for field in "complete recordComplete bankAdmissible recordBankAdmissible".split():
        require(not e[field].any(), f"event {field} must be false")
    for name in capture.TABLES[1:]:
        table = tables[name]
        rows = unique(table, [*capture.EVENT, "index"], name)
        er = match(rows, events, capture.EVENT, name)["_event"].to_numpy(dtype=np.int64)
        counts = np.bincount(er, minlength=expected_events)
        require(np.array_equal(e["n" + name.capitalize()], counts), f"{name} census mismatch")
        require(np.all((table["index"] >= 0) & (table["index"] < counts[er])), f"{name} row index gap")
    errors = pd.Series(e["errors"])
    error_counts = np.where(errors.eq(""), 0, errors.str.count("\n").to_numpy() + 1)
    require(np.array_equal(e["nErrors"], error_counts), "recorder error census mismatch")
    validation = pd.Series(e["validationErrors"])
    if kind == "stacking-rejection":
        for field in "capturePresent statusPresent eventIDsMatch statusClean".split():
            require(e[field].all(), f"event {field} must be true")
        for field in "g4Status nTracks nSteps nG4Tracks overflowSimParticles nKilledStepLimit nKilledByFieldPropagator".split():
            require(np.all(e[field] == 0), f"invalid stacking event {field}")
        for field in capture.EVENT:
            require(np.array_equal(e[field], e["record" + field.capitalize()]), "record event identity mismatch")
        require(np.all(e["nClassifications"] == 1) and np.all(e["nVolumes"] == 3), "stacking child census mismatch")
        tokens = errors.str.split("\n").explode()
        require(np.all(error_counts == len(ERRORS)) and tokens.isin(ERRORS).all() and
                tokens.groupby(level=0).nunique().eq(len(ERRORS)).all(), "unexpected stacking recorder errors")
        require(validation.eq("Capture record is incomplete\nCapture record contains errors").all(),
                "unexpected stacking validation errors")
        unique(c, capture.TRACK, "classification")
        for field, value in (("index", 0), ("track", 1), ("parent", 0), ("killed", True)):
            require(np.all(c[field] == value), f"invalid classification {field}")
        unique(v, [*capture.EVENT, "id"], "volume")
        require(np.array_equal(v["id"], v["index"]), "volume id/index mismatch")
        for field in ("geometryTag", "physicsTag", "envelope"):
            require(np.all(e[field] != ""), f"empty {field}")
    else:
        for field in "capturePresent statusPresent eventIDsMatch statusClean".split():
            require(not e[field].any(), f"event {field} must be false")
        for name in capture.TABLES[1:]:
            require(len(tables[name]["event"]) == 0, f"internal-filter has stale {name} rows")
        for field in ("recordRun", "recordSubrun", "recordEvent"):
            require(np.all(e[field] == 0), f"internal-filter has stale {field}")
        require(errors.eq("").all(), "internal-filter recorder errors present")
        for field in ("geometryTag", "physicsTag", "envelope"):
            require(np.all(e[field] == ""), f"internal-filter has stale {field}")
        require(validation.str.fullmatch(
            r"Missing ComponentCapture: [^\n]+\nMissing StatusG4: [^\n]+\n"
            r"Capture record is incomplete\nStatusG4 is not clean").all(),
            "unexpected internal-filter validation errors")
        require(np.all(e["g4Status"] == np.iinfo(np.int32).max), "missing StatusG4 default mismatch")
        for field in "nG4Tracks overflowSimParticles nKilledStepLimit nKilledByFieldPropagator cpuTime realTime".split():
            require(np.all(e[field] == 0), f"internal-filter has stale {field}")
    try:
        capture.audit(tables, expected_events)
    except ValueError as exc:
        rejection = str(exc)
        require(bool(rejection), "strict audit rejection must explain the failure")
    else:
        raise ValueError("strict capture audit unexpectedly accepted a negative fixture")
    columns = [field for field in e if field not in VECTORS]
    return {"diagnostic_only": True, "bank_admissible": False, "negative_fixture_passed": True,
            "kind": kind, "run": run, "subrun": subrun, "first_event": first_event,
            "expected_events": expected_events, "events": len(events),
            "rows": {name: len(tables[name]["event"]) for name in capture.TABLES},
            "strict_audit_rejection": rejection,
            "event_rows": frame(e, columns).sort_values(capture.EVENT).to_dict(orient="records")}


def recovery(tables, run, subrun=0, first_event=1, expected_events=4):
    """Check fixed rejected events 1/3 and separately audit accepted events 2/4; never select bank donors."""
    run, subrun = _integer(run, "run"), _integer(subrun, "subrun")
    require(_integer(first_event, "first_event") == 1 and
            _integer(expected_events, "expected_events", 1) == 4, "recovery fixture requires events 1 through 4")
    tables = _schema(tables, "stacking-rejection")
    e = tables["events"]
    events = unique(e, capture.EVENT, "event")
    require(len(events) == 4 and np.all(e["run"] == run) and np.all(e["subrun"] == subrun) and
            np.array_equal(np.sort(e["event"]), np.arange(1, 5, dtype=np.int64)), "recovery event census mismatch")
    for name in capture.TABLES[1:]:
        match(frame(tables[name], capture.EVENT), events, capture.EVENT, name)
    rejected_tables = {name: {field: value[np.isin(table["event"], [1, 3])] for field, value in table.items()}
                       for name, table in tables.items()}
    rejected = report(rejected_tables, "stacking-rejection", run, subrun, 1, 2,
                      event_ids=np.array([1, 3], dtype=np.int64))
    accepted = {name: {field: value[np.isin(table["event"], [2, 4])] for field, value in table.items()}
                for name, table in tables.items()}
    audited = capture.audit(accepted, 2)
    c = accepted["classifications"]
    require(np.all(c["track"][c["index"] == 0] == 1), "accepted event does not restart track IDs")
    try:
        capture.audit(tables, 4)
    except ValueError as exc:
        rejection = str(exc)
    else:
        raise ValueError("strict capture audit accepted the mixed rejection fixture")
    return {"diagnostic_only": True, "bank_admissible": False, "recovery_fixture_passed": True,
            "kind": "stacking-recovery", "run": run, "subrun": subrun, "events": 4,
            "rejected_events": rejected, "accepted_events": [2, 4], "accepted_audit": audited,
            "strict_audit_rejection": rejection}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--kind", choices=CLI_KINDS, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--subrun", type=int, default=0)
    parser.add_argument("--first-event", type=int, default=1)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tables = capture.read(args.input)
    result = (recovery(tables, args.run, args.subrun, args.first_event, args.expected_events)
              if args.kind == "stacking-recovery" else
              report(tables, args.kind, args.run, args.subrun, args.first_event, args.expected_events))
    result["input"] = str(args.input)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
