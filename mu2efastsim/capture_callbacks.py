"""Strict constructed-object callback diagnostic, not Geant4 transport or bank validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from . import capture
from .capture_lifecycle import BOOL, FLOAT, STR, VECTORS, _integer
from .saved_g4 import frame, match, require, unique

CASES = np.array([
    "boundary-inward-inside", "boundary-outward-inside", "boundary-inward-outside",
    "boundary-outward-outside", "interior-zero-step", "interior-terminal-classification",
    "kill-track-and-secondaries", "suspend", "suspend-and-wait", "postpone-to-next-event",
    "duplicate-classification", "discard-resume-missing-capture", "recovery-after-discard",
], dtype=object)
BOUNDARY_ERROR = "boundary_born_secondary_ownership_unresolved"
OUTSIDE_ERRORS = "secondary_born_outside_envelope\n" + BOUNDARY_ERROR + "\ntrack_never_entered_component"
TERMINAL_ERROR = "nonterminal_or_kill_secondaries_status"
DUPLICATE_ERRORS = "reclassification_unsupported\nclassification_tracking_census_mismatch"
WORLD, ENVELOPE = "CallbackFixtureWorld", "CallbackFixtureEnvelope"


def _schema(tables):
    """Keep metadata jagged; reject coercion of identities, booleans and scalar states."""
    result = {}
    for name in capture.TABLES:
        require(name in tables, f"missing table {name}")
        fields = [*capture.EVENT, *capture.FIELDS[name].split(),
                  *(prefix + field for prefix in capture.STATES.get(name, ()) for field in capture.STATE)]
        require(set(fields) <= set(tables[name]), f"missing {name} fields: {set(fields) - set(tables[name])}")
        event = np.asarray(tables[name]["event"])
        require(event.ndim == 1, f"{name}: malformed event")
        size, table = len(event), {}
        for key in fields:
            if key in VECTORS:
                value = ak.Array(tables[name][key])
                require(value.ndim == 2 and len(value) == size, f"{name}: malformed {key}")
                require(not ak.any(ak.is_none(value, axis=0)) and not ak.any(ak.is_none(value, axis=1)),
                        f"{name}: missing {key} values")
                flat = ak.to_numpy(ak.flatten(value, axis=1), allow_missing=False)
                require(flat.dtype.kind == "f" and np.isfinite(flat).all(), f"{name}: invalid {key} values")
                table[key] = value
                continue
            value = np.asarray(tables[name][key])
            require(value.ndim == 1 and len(value) == size, f"{name}: malformed {key}")
            leaf = key.split("_")[-1]
            if key in capture.INT64 or leaf == "volume":
                require(value.dtype.kind == "i" and value.dtype.itemsize == 8, f"{name}: {key} must be int64")
            elif key in BOOL:
                require(value.dtype.kind == "b", f"{name}: {key} must be boolean")
            elif leaf in STR:
                require(size == 0 or (value.dtype.kind in "OU" and
                        pd.api.types.infer_dtype(value, skipna=False) == "string"), f"{name}: {key} must be strings")
            elif leaf in FLOAT:
                require(value.dtype.kind == "f" and np.isfinite(value).all(), f"{name}: invalid {key} values")
            else:
                require(value.dtype.kind == "i", f"{name}: {key} must be integer")
            table[key] = value
        require((frame(table, capture.EVENT) >= 0).all().all(), f"{name}: negative event identity")
        result[name] = table
    return result


def _check(table, field, expected, label):
    actual = table[field]
    good = (np.allclose(actual, expected, rtol=0, atol=1e-10) if actual.dtype.kind == "f"
            else np.all(actual == expected))
    require(good, f"{label}: {field} mismatch")


def _metadata(table, present, label):
    for field in VECTORS:
        if field not in table:
            continue
        width = VECTORS[field]
        counts = ak.to_numpy(ak.num(table[field], axis=1))
        require(np.array_equal(counts, np.where(present, width, 0)), f"{label}: {field} shape mismatch")
        flat = ak.to_numpy(ak.flatten(table[field], axis=1), allow_missing=False)
        row = np.eye(3).ravel() if field == "rotation" else np.full(width, 50. if field == "halfSize" else 0.)
        require(np.array_equal(flat, np.tile(row, np.count_nonzero(present))), f"{label}: {field} mismatch")


def _state(table, prefix, z, pz, energy, time, volume, status, label):
    """Check every stored state field, including absent entrance defaults and identity transforms."""
    expected = {field: 0. for field in capture.FLOATS}
    expected.update(z=z, lz=z, pz=pz, lpz=pz, energy=energy, kinetic=energy, time=time, weight=1.)
    expected.update(volume=volume, stepStatus=status, process="",
                    material=np.where(volume < 0, "", np.where(volume == 0, "G4_Galactic", "G4_POLYSTYRENE")),
                    placement=np.where(volume < 0, "", np.where(volume == 0, "/0:0", "/0:0/1:0")))
    for field, value in expected.items():
        _check(table, prefix + field, value, label)


def report(tables, run, subrun=0):
    """Require all thirteen pinned callback cases without fabricating StatusG4 or physical success."""
    run, subrun = _integer(run, "run"), _integer(subrun, "subrun")
    tables = _schema(tables)
    e, t, s, c, v = (tables[name] for name in capture.TABLES)
    events = unique(e, capture.EVENT, "event").assign(_event=np.arange(len(e["event"]), dtype=np.int64))
    require(len(events) == 13 and np.array_equal(np.sort(e["event"]), np.arange(1, 14, dtype=np.int64)),
            "expected exact 13-event census (IDs 1..13)")
    require(np.all(e["run"] == run) and np.all(e["subrun"] == subrun), "expected run/subrun census mismatch")
    present = e["event"] != 12
    complete = np.isin(e["event"], [5, 6, 13])
    _check(e, "schema", 1, "events")
    for field in "complete bankAdmissible recordBankAdmissible statusPresent statusClean overflowSimParticles".split():
        _check(e, field, False, "events")
    for field in ("capturePresent", "eventIDsMatch"):
        _check(e, field, present, "events")
    _check(e, "recordComplete", complete, "events")
    for field in capture.EVENT:
        _check(e, "record" + field.capitalize(), np.where(present, e[field], 0), "events")
    _check(e, "g4Status", np.iinfo(np.int32).max, "events")
    for field in "nG4Tracks nKilledStepLimit nKilledByFieldPropagator cpuTime realTime".split():
        _check(e, field, 0, "events")
    for field, value in (("geometryTag", "diagnostic-constructed-objects-v1"),
                         ("physicsTag", "NO-PHYSICS-TRANSPORT-v1"), ("envelope", ENVELOPE)):
        _check(e, field, np.where(present, value, ""), "events")
    _metadata(e, present, "events")
    for name, count in (("tracks", 2), ("steps", 3), ("classifications", 2), ("volumes", 2)):
        table = tables[name]
        rows = unique(table, [*capture.EVENT, "index"], name)
        er = match(rows, events, capture.EVENT, name)["_event"].to_numpy(dtype=np.int64)
        counts = np.bincount(er, minlength=13)
        expected = np.where(present, count + ((e["event"] == 11) & (name == "classifications")), 0)
        require(np.array_equal(counts, expected), f"{name}: callback case census mismatch")
        _check(e, "n" + name.capitalize(), counts, "events")
        require(np.all((table["index"] >= 0) & (table["index"] < counts[er])), f"{name}: row index gap")
    errors = np.select([e["event"] <= 2, np.isin(e["event"], [3, 4]),
                        (e["event"] >= 7) & (e["event"] <= 10), e["event"] == 11],
                       [BOUNDARY_ERROR, OUTSIDE_ERRORS, TERMINAL_ERROR, DUPLICATE_ERRORS], default="")
    _check(e, "errors", errors, "events recorder errors")
    _check(e, "nErrors", np.where(errors == "", 0, pd.Series(errors).str.count("\n").to_numpy() + 1), "events")
    validation = np.select([~present, complete], [
        "Missing ComponentCapture: fixture:componentCapture\nMissing StatusG4: fixture\n"
        "Capture record is incomplete\nStatusG4 is not clean",
        "Missing StatusG4: fixture\nStatusG4 is not clean\nStatusG4/captured track census mismatch"],
        default="Missing StatusG4: fixture\nCapture record is incomplete\nStatusG4 is not clean\n"
                "Capture record contains errors\nStatusG4/captured track census mismatch")
    _check(e, "validationErrors", validation, "events")

    unique(v, [*capture.EVENT, "id"], "volume")
    for field, value in (("id", v["index"]), ("copy", 0), ("solid", "G4Box"),
                         ("name", np.where(v["index"] == 0, WORLD, ENVELOPE)),
                         ("logical", np.where(v["index"] == 0, WORLD + "Logical", ENVELOPE + "Logical")),
                         ("material", np.where(v["index"] == 0, "G4_Galactic", "G4_POLYSTYRENE"))):
        _check(v, field, value, "volumes")
    require(np.all(v["density"] > 0), "volumes: nonpositive density")
    require(np.all(v["logical"] != ""), "volumes: empty logical volume")
    world = frame(v, [*capture.EVENT, "logical"]).loc[v["id"] == 0]
    mother = match(frame(v, capture.EVENT), world, capture.EVENT, "volume mother")["logical"].to_numpy()
    _check(v, "motherLogical", np.where(v["id"] == 0, "", mother), "volumes")
    _metadata(v, np.ones(len(v["event"]), dtype=bool), "volumes")

    unique(t, capture.TRACK, "track")
    parent = t["index"] == 0
    bad_status = ~parent & (t["event"] >= 7) & (t["event"] <= 10)
    outside = ~parent & np.isin(t["event"], [3, 4])
    for field, value in (("track", t["index"] + 1), ("parent", np.where(parent, 0, 1)), ("pdg", 22),
                         ("steps", np.where(parent, 3, 0)), ("secondaries", parent.astype(np.int64)),
                         ("status", np.where(parent, 0, np.where(bad_status, t["event"] - 4, 2))),
                         ("reason", np.where(parent, 2, np.where(bad_status, 3, 1))),
                         ("entered", ~outside), ("ended", True), ("creator", ""), ("stopProcess", "")):
        _check(t, field, value, "tracks")
    child_z = np.where(t["event"] <= 4, 50., 0.)
    child_time = np.where(t["event"] <= 4, 3., 2.)
    pz = np.where(parent, 10., np.where(np.isin(t["event"], [1, 3]), -1., 1.))
    energy = np.where(parent, 10., 1.)
    volume = np.where(parent | outside, 0, 1)
    _state(t, "begin_", np.where(parent, -50.001, child_z), pz, energy,
           np.where(parent, 0., child_time), volume, -1, "tracks begin")
    _state(t, "end_", np.where(parent, 50., child_z), pz, energy,
           np.where(parent, 3., child_time), volume, np.where(parent, 1, -1), "tracks end")
    _state(t, "entrance_", np.where(outside, 0., np.where(parent, -50., child_z)),
           np.where(outside, 0., pz), np.where(outside, 0., energy),
           np.where(outside, 0., np.where(parent, 1., child_time)), np.where(outside, -1, 1),
           np.where(parent, 1, -1), "tracks entrance")

    number = s["index"] + 1
    for field, value in (("track", 1), ("number", number),
                         ("secondaries", (number == np.where(s["event"] <= 4, 3, 2)).astype(np.int64)),
                         ("deposit", 0.), ("nonIonizingDeposit", 0.), ("trackStatus", 0),
                         ("preInside", number != 1), ("postInside", number != 3), ("firstExit", number == 3)):
        _check(s, field, value, "steps")
    pre_z = np.select([number == 1, number == 2], [-50.001, -50.], default=0.)
    post_z = np.select([number == 1, number == 2], [-50., 0.], default=50.)
    _check(s, "length", post_z - pre_z, "steps")
    _state(s, "pre_", pre_z, 10., 10., number - 1., np.where(number == 1, 0, 1), 7, "steps pre")
    _state(s, "post_", post_z, 10., 10., number.astype(float), np.where(number == 3, 0, 1),
           np.where(number == 2, 4, 1), "steps post")

    child = c["index"] != 0
    for field, value in (("track", np.where(child, 2, 1)), ("parent", child.astype(np.int64)), ("pdg", 22),
                         ("status", np.where(child & (c["event"] == 6), 2, 0)), ("killed", False)):
        _check(c, field, value, "classifications")
    tracks = frame(t, capture.TRACK).assign(_track=np.arange(len(t["event"]), dtype=np.int64))
    tr = match(frame(c, capture.TRACK), tracks, capture.TRACK, "classification")["_track"].to_numpy(dtype=np.int64)
    for field in capture.STATE:
        _check(c, "state_" + field, t["begin_" + field][tr], "classifications state")
    try:
        capture.audit(tables, 13)
    except ValueError as exc:
        rejection = str(exc)
        require(bool(rejection), "strict audit rejection must explain the failure")
    else:
        raise ValueError("strict capture audit unexpectedly accepted the callback fixture")
    columns = [field for field in e if field not in VECTORS]
    cases = frame(e, columns).assign(case=CASES[e["event"] - 1]).sort_values(capture.EVENT).to_dict(orient="records")
    return {"diagnostic_only": True, "bank_admissible": False, "physical_transport": False,
            "callback_fixture_passed": True, "run": run, "subrun": subrun, "events": 13,
            "rows": {name: len(tables[name]["event"]) for name in capture.TABLES},
            "cases": cases, "strict_audit_rejection": rejection}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--subrun", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(capture.read(args.input), args.run, args.subrun)
    result["input"] = str(args.input)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
