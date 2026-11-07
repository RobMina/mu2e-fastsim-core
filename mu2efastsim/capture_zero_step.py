"""Strict real-engine zero-step diagnostic; lifecycle success is not endpoint or physics certification.

The pinned fixture injects fStopAndKill on one primary in each of two events.
Finalized SimParticle endpoint momentum/energy are measurements, not acceptance
criteria. The required persisted-product readback must preserve every stored field.
All joins retain signed int64 identities; loops are over fixed schemas, never events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from . import capture
from .capture_callbacks import _schema as _capture_schema
from .capture_lifecycle import VECTORS, _integer
from .saved_g4 import frame, match, require, unique

CALLBACKS, EVENTS, SIMS = "zeroStepCallbacks", "zeroStepEvents", "zeroStepSims"
PATHS = {CALLBACKS: "g4run/zeroStepCallbacks", EVENTS: "finalization/zeroStepEvents",
         SIMS: "finalization/zeroStepSims"}
STATE = "x y z px py pz energy kinetic time properTime".split()
SIM_STATE = [field for field in STATE if field != "kinetic"]
SCHEMAS = {
    CALLBACKS: {
        "int64": [*capture.EVENT, *"index track parent steps".split()],
        "int": "phase pdg statusIn statusForward statusOut g4Version".split(),
        "bool": "diagnosticOnly bankAdmissible killedValid killed mutated forwarded stepPresent userInfoPresent mu2eInfoPresent forced pre_present pre_processPresent post_present post_processPresent".split(),
        "float": [*"trackLength infoKE infoDx infoDy infoDz infoTime infoProperTime".split(),
                  *(prefix + field for prefix in ("track_", "pre_", "post_") for field in STATE)],
        "str": "error infoCode pre_process post_process".split(),
    },
    EVENTS: {"int64": [*capture.EVENT, "nSims"], "int": [], "float": [], "str": [],
             "bool": "diagnosticOnly bankAdmissible capturePresent simsPresent statusPresent".split()},
    SIMS: {
        "int64": [*capture.EVENT, *"index key track parent nDaughters simStage".split()],
        "int": "pdg startStatus endStatus nSteps".split(),
        "bool": "endDefined isPrimary genPresent parentPresent".split(),
        "float": ["trackLength", "endKE", *(prefix + field for prefix in ("start_", "end_") for field in SIM_STATE)],
        "str": ["creationCode", "stoppingCode"],
    },
}
SCHEMAS[CALLBACKS]["int"] += ["pre_status", "post_status"]
NAMES = np.array(["World", "BoxInTheWorld", "CaptureEnvelope"], dtype=object)
MATERIALS = np.array(["G4_Galactic", "G4_Galactic", "G4_POLYSTYRENE"], dtype=object)
PHOTON = dict.fromkeys(STATE, 0.) | {"z": -50.001, "pz": 10., "energy": 10., "kinetic": 10.}
ATOL = 1e-10


def read(path, *, readback=False):
    """Read highest cycles independently for every tree; readback must have no G4 callback tree."""
    tables = capture.read(path)
    with uproot.open(path) as root:
        if readback:
            require(PATHS[CALLBACKS] not in root, "readback must not contain a callback tree")
        for name in ((EVENTS, SIMS) if readback else (CALLBACKS, EVENTS, SIMS)):
            require(PATHS[name] in root, f"missing tree {PATHS[name]}")
            tables[name] = {field: ak.to_numpy(value, allow_missing=False) for field, value in
                            root[PATHS[name]].arrays(library="ak", how=dict).items()}
    return tables


def _schema(tables, *, readback=False):
    names = (*capture.TABLES, EVENTS, SIMS) if readback else (*capture.TABLES, CALLBACKS, EVENTS, SIMS)
    require(set(tables) == set(names), "missing or unexpected diagnostic tables")
    result = _capture_schema(tables)
    # Validate and retain count leaves added by uproot's synthetic TTree writer.
    for name in capture.TABLES:
        extras = set(tables[name]) - set(result[name])
        allowed = {"n" + field for field in VECTORS if field in result[name]}
        require(extras <= allowed, f"{name}: unexpected fields {extras - allowed}")
        for field in extras:
            value = np.asarray(tables[name][field])
            require(value.ndim == 1 and value.dtype.kind == "i" and
                    np.array_equal(value, ak.to_numpy(ak.num(result[name][field[1:]], axis=1))),
                    f"{name}: invalid metadata count {field}")
            result[name][field] = value
    for name in ((EVENTS, SIMS) if readback else (CALLBACKS, EVENTS, SIMS)):
        fields = {field for group in SCHEMAS[name].values() for field in group}
        require(set(tables[name]) == fields, f"{name}: missing or unexpected fields")
        event = np.asarray(tables[name]["event"])
        require(event.ndim == 1, f"{name}: malformed event")
        size, table = len(event), {}
        for kind, group in SCHEMAS[name].items():
            for field in group:
                value = np.asarray(tables[name][field])
                require(value.ndim == 1 and len(value) == size, f"{name}: malformed {field}")
                good = {"int64": value.dtype.kind == "i" and value.dtype.itemsize == 8,
                        "int": value.dtype.kind == "i", "bool": value.dtype.kind == "b",
                        "float": value.dtype.kind == "f", "str": value.dtype.kind in "OU"}[kind]
                require(good, f"{name}: {field} must be {kind}")
                if kind == "float":
                    require(np.isfinite(value).all(), f"{name}: nonfinite {field}")
                elif kind == "str":
                    require(size == 0 or pd.api.types.infer_dtype(value, skipna=False) == "string",
                            f"{name}: {field} must be strings")
                table[field] = value
        result[name] = table
    return result


def _check(table, field, expected, label):
    actual = table[field]
    good = (np.allclose(actual, expected, rtol=0, atol=ATOL) if actual.dtype.kind == "f"
            else np.all(actual == expected))
    require(good, f"{label}: {field} mismatch")


def _checks(table, expected, label):
    for field, value in expected.items():
        _check(table, field, value, label)


def _metadata(table, label):
    for field, width in VECTORS.items():
        if field not in table:
            continue
        require(np.all(ak.to_numpy(ak.num(table[field], axis=1)) == width), f"{label}: {field} shape mismatch")
        value = ak.to_numpy(table[field], allow_missing=False)
        expected = np.eye(3).ravel() if field == "rotation" else np.full(width, 50. if field == "halfSize" else 0.)
        require(np.array_equal(value, np.broadcast_to(expected, value.shape)), f"{label}: {field} mismatch")


def _event_census(table, run, label):
    events = unique(table, capture.EVENT, label)
    require(len(events) == 2 and np.all(table["run"] == run) and np.all(table["subrun"] == 0) and
            np.array_equal(np.sort(table["event"]), np.array([1, 2], dtype=np.int64)),
            f"{label}: expected exact two-event census (run,0,1/2)")
    return events.assign(_event=np.arange(2, dtype=np.int64))


def _rows(table, events, count, label):
    rows = unique(table, [*capture.EVENT, "index"], label)
    er = match(rows, events, capture.EVENT, label)["_event"].to_numpy(dtype=np.int64)
    counts = np.bincount(er, minlength=2)
    require(np.all(counts == count), f"{label}: row census mismatch")
    require(np.all((table["index"] >= 0) & (table["index"] < count)), f"{label}: row index gap")
    return counts


def _capture_state(table, prefix, volume, *, absent=False):
    expected = dict.fromkeys(capture.FLOATS, 0.)
    if not absent:
        expected.update(PHOTON, lz=-50.001, lpz=10.)
    expected.update(weight=1., volume=volume, stepStatus=-1, process="",
                    material=np.where(volume < 0, "", "G4_Galactic"),
                    placement=np.where(volume < 0, "", "/0:0/1:1"))
    _checks(table, {prefix + field: value for field, value in expected.items()}, "capture state")
    if absent:
        for field in capture.FLOATS:
            require(np.all(table[prefix + field] == expected[field]), prefix + field + " absent state default mismatch")


def _capture(tables, run):
    e, t, s, c, v = (tables[name] for name in capture.TABLES)
    events = _event_census(e, run, "capture events")
    for name, count in (("tracks", 1), ("steps", 0), ("classifications", 1), ("volumes", 3)):
        counts = _rows(tables[name], events, count, name)
        _check(e, "n" + name.capitalize(), counts, "capture events")
    _checks(e, dict(schema=1, complete=False, recordComplete=False, bankAdmissible=False,
                    recordBankAdmissible=False, capturePresent=True, statusPresent=True, eventIDsMatch=True,
                    statusClean=True, overflowSimParticles=False, g4Status=0, nG4Tracks=1,
                    nKilledStepLimit=0, nKilledByFieldPropagator=0, nErrors=1,
                    errors="track_never_entered_component",
                    validationErrors="Capture record is incomplete\nCapture record contains errors",
                    geometryTag="diagnostic-engine-zero-step-box-v1",
                    physicsTag="DIAGNOSTIC-STATUS-INJECTION-NOT-RESPONSE-v1", envelope="CaptureEnvelope"), "capture events")
    for field in capture.EVENT:
        _check(e, "record" + field.capitalize(), e[field], "capture events")
    require(np.all(e["cpuTime"] >= 0) and np.all(e["realTime"] >= 0), "negative StatusG4 timing")
    _metadata(e, "capture events")
    _metadata(v, "volumes")
    unique(v, [*capture.EVENT, "id"], "volume")
    i = v["index"]
    _checks(v, dict(id=i, copy=i, name=NAMES[i], logical=NAMES[i],
                    motherLogical=np.array(["", "World", "BoxInTheWorld"], dtype=object)[i],
                    material=MATERIALS[i], solid="G4Box"), "volumes")
    require(np.allclose(v["density"], np.where(i == 2, 1.06, 1e-25), rtol=1e-12, atol=0),
            "volumes: density mismatch")
    unique(t, capture.TRACK, "capture track")
    _checks(t, dict(track=1, parent=0, pdg=22, steps=0, secondaries=0, status=2, reason=1,
                    entered=False, ended=True, creator="", stopProcess=""), "tracks")
    _capture_state(t, "begin_", np.ones(len(t["event"]), dtype=np.int64))
    _capture_state(t, "end_", np.ones(len(t["event"]), dtype=np.int64))
    _capture_state(t, "entrance_", np.full(len(t["event"]), -1, dtype=np.int64), absent=True)
    unique(c, capture.TRACK, "classification")
    _checks(c, dict(track=1, parent=0, pdg=22, status=np.where(c["event"] == 1, 2, 0), killed=False), "classifications")
    require(np.isin(c["state_volume"], [-1, 1]).all(), "classification volume must be absent or BoxInTheWorld")
    _capture_state(c, "state_", c["state_volume"])
    try:
        capture.audit(tables, 2)
    except ValueError as exc:
        rejection = str(exc)
        require(bool(rejection), "strict audit rejection must explain the failure")
    else:
        raise ValueError("strict capture audit unexpectedly accepted the zero-step diagnostic")
    return events, rejection


def _callbacks(table, events):
    _rows(table, events, 3, CALLBACKS)
    p, first = table["phase"], table["event"] == 1
    _checks(table, dict(phase=table["index"], track=1, parent=0, pdg=22, steps=0, trackLength=0.,
                        g4Version=1132, diagnosticOnly=True, bankAdmissible=False, forwarded=True,
                        killed=False, killedValid=p == 0, error="",
                        mutated=(first & (p == 0)) | (~first & (p == 1)),
                        statusIn=np.where((p == 2) | (first & (p == 1)), 2, 0),
                        statusForward=np.where(first | (p == 2), 2, 0),
                        statusOut=np.where(first | (p > 0), 2, 0)), CALLBACKS)
    _checks(table, {"track_" + field: value for field, value in PHOTON.items()}, CALLBACKS)
    require(np.all(table["stepPresent"][p == 2]), "posttracking must expose the engine step")
    for prefix in ("pre_", "post_"):
        present = table[prefix + "present"]
        require(np.all(present[p == 2]), "posttracking must expose " + prefix + "point")
        require(np.all(~present | table["stepPresent"]), prefix + "point without a step")
        _checks(table, {prefix + "status": np.where(present, 7, -1),
                        prefix + "processPresent": False, prefix + "process": ""}, CALLBACKS)
        _checks(table, {prefix + field: np.where(present, value, 0.) for field, value in PHOTON.items()}, CALLBACKS)
        for field in STATE:
            require(np.all(table[prefix + field][~present] == 0.), prefix + field + " absent point default mismatch")
    info = p > 0
    _checks(table, dict(userInfoPresent=info, mu2eInfoPresent=info, forced=False,
                        infoCode=np.where(info, "uninitialized", ""), infoKE=0., infoDx=0., infoDy=0., infoDz=0.,
                        infoTime=np.where(info, -1., 0.), infoProperTime=np.where(info, -1., 0.)), CALLBACKS)
    for field in ("trackLength", "infoKE", "infoDx", "infoDy", "infoDz", "infoTime", "infoProperTime"):
        expected = np.where(info, -1., 0.) if field in ("infoTime", "infoProperTime") else 0.
        require(np.all(table[field] == expected), CALLBACKS + ": " + field + " exact default mismatch")


def _finalization(tables, events, run):
    e, s = tables[EVENTS], tables[SIMS]
    _event_census(e, run, EVENTS)
    _checks(e, dict(nSims=1, diagnosticOnly=True, bankAdmissible=False,
                    capturePresent=True, simsPresent=True, statusPresent=True), EVENTS)
    _rows(s, events, 1, SIMS)
    unique(s, capture.TRACK, "finalized track")
    unique(s, [*capture.EVENT, "key"], "SimParticle key")
    _checks(s, dict(key=1, track=1, parent=0, nDaughters=0, simStage=0, pdg=22, nSteps=0,
                    trackLength=0., endDefined=True, isPrimary=True, genPresent=True, parentPresent=False,
                    startStatus=np.where(s["event"] == 1, 2, 0), endStatus=2,
                    creationCode="mu2ePrimary", stoppingCode="NotSpecified"), SIMS)
    _checks(s, {"start_" + field: PHOTON[field] for field in SIM_STATE}, SIMS)
    _checks(s, {"end_" + field: PHOTON[field] for field in ("x", "y", "z", "time", "properTime")}, SIMS)
    require(np.all(s["trackLength"] == 0.), "finalized trackLength must be exactly zero")
    require(np.all(s["endKE"] >= 0) and np.all(s["end_energy"] >= 0), "negative finalized endpoint energy")


def _equal(original, readback):
    """Exact value equality after full-int64 row joins, including all state and metadata fields."""
    for name in (*capture.TABLES, EVENTS, SIMS):
        left, right = original[name], readback[name]
        require(set(left) == set(right), f"readback {name}: field census mismatch")
        keys = capture.EVENT if name in ("events", EVENTS) else [*capture.EVENT, "index"]
        lr, rr = unique(left, keys, name), unique(right, keys, "readback " + name)
        require(len(lr) == len(rr), f"readback {name}: row census mismatch")
        rr = rr.assign(_row=np.arange(len(rr), dtype=np.int64))
        order = match(lr, rr, keys, "readback " + name)["_row"].to_numpy(dtype=np.int64)
        for field in left:
            a, b = left[field], right[field][order]
            if field in VECTORS:
                require(np.array_equal(ak.to_numpy(ak.num(a, axis=1)), ak.to_numpy(ak.num(b, axis=1))) and
                        np.array_equal(ak.to_numpy(ak.flatten(a)), ak.to_numpy(ak.flatten(b))),
                        f"readback {name}: {field} mismatch")
            else:
                require(np.array_equal(a, b), f"readback {name}: {field} mismatch")


def _endpoints(tables):
    s, t = tables[SIMS], tables["tracks"]
    tr = unique(t, capture.TRACK, "capture track").assign(_track=np.arange(len(t["event"]), dtype=np.int64))
    order = match(frame(s, capture.TRACK), tr, capture.TRACK, "finalized endpoint")["_track"].to_numpy(dtype=np.int64)
    rows = frame(s, [*capture.EVENT, "track"])
    endpoint_match = np.ones(len(s["event"]), dtype=bool)
    predicted_zero = np.ones(len(s["event"]), dtype=bool)
    for field in STATE:
        observed = s["endKE" if field == "kinetic" else "end_" + field]
        captured = t["end_" + field][order]
        rows["captured_end_" + field] = captured
        rows["sim_end_" + field] = observed
        rows["delta_" + field] = observed - captured
        endpoint_match &= np.isclose(observed, captured, rtol=0, atol=ATOL)
        if field in ("px", "py", "pz", "energy", "kinetic"):
            predicted_zero &= observed == 0.
    rows["endpoint_match"] = endpoint_match
    rows["predicted_zero_endpoint"] = predicted_zero
    return bool(endpoint_match.all()), rows.sort_values(capture.EVENT).to_dict(orient="records")


def report(tables, readback, run):
    """Validate the diagnostic lifecycle and persistence, without requiring a particular endpoint bug."""
    run = _integer(run, "run", 1)
    tables, readback = _schema(tables), _schema(readback, readback=True)
    events, rejection = _capture(tables, run)
    _callbacks(tables[CALLBACKS], events)
    _finalization(tables, events, run)
    readback_events, readback_rejection = _capture(readback, run)
    _finalization(readback, readback_events, run)
    _equal(tables, readback)
    endpoints_match, endpoint_rows = _endpoints(tables)
    return {"diagnostic_only": True, "bank_admissible": False, "real_engine": True, "physical_response": False,
            "engine_lifecycle_passed": True, "readback_identical": True,
            "endpoint_kinematics_match": endpoints_match, "endpoint_comparison_atol": ATOL,
            "run": run, "subrun": 0, "events": 2,
            "rows": {name: len(table["event"]) for name, table in tables.items()},
            "strict_audit_rejection": rejection, "readback_strict_audit_rejection": readback_rejection,
            "endpoint_rows": endpoint_rows,
            "callback_rows": frame(tables[CALLBACKS], list(tables[CALLBACKS])).sort_values(
                [*capture.EVENT, "index"]).to_dict(orient="records")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-endpoint-match", action="store_true",
                        help="require finalized endpoint kinematics to match the captured track")
    args = parser.parse_args(argv)
    require(args.output.resolve() not in (args.input.resolve(), args.readback.resolve()), "output must not overwrite input")
    result = report(read(args.input), read(args.readback, readback=True), args.run)
    if args.require_endpoint_match:
        require(result["endpoint_kinematics_match"], "finalized endpoint kinematics do not match captured track")
    result.update(input=str(args.input), readback=str(args.readback))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
