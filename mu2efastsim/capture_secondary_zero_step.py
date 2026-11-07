"""One-event engine-secondary status-injection diagnostic, never a response-bank certificate.

All joins preserve signed int64 identities. Only fixed schema/phase loops are used.
Synthetic inputs exercise the checker, not Geant4 or persisted-art runtime validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from . import capture, capture_zero_step as zero
from .capture_callbacks import _schema as capture_schema
from .capture_lifecycle import VECTORS
from .saved_g4 import frame, match, require, unique

CALLBACKS, EVENTS, SIMS = "secondaryZeroStepCallbacks", zero.EVENTS, zero.SIMS
PATHS = {CALLBACKS: "g4run/secondaryZeroStepCallbacks", EVENTS: zero.PATHS[EVENTS], SIMS: zero.PATHS[SIMS]}
STATE, SIM_STATE = zero.STATE, zero.SIM_STATE
SCHEMAS = {name: {kind: list(fields) for kind, fields in zero.SCHEMAS[old].items()}
           for name, old in ((CALLBACKS, zero.CALLBACKS), (EVENTS, EVENTS), (SIMS, SIMS))}
SCHEMAS[CALLBACKS]["str"] += ["mode", "creator"]
SCHEMAS[CALLBACKS]["bool"] += "creatorPresent cutEvaluated cutResult parentEnded".split()
SCHEMAS[CALLBACKS]["int64"] += ["stepSecondaries", "parentEndIndex"]
SCHEMAS[CALLBACKS]["float"] += ["weight", "dynamicMass"]
PHYSICS = "DIAGNOSTIC-SECONDARY-STATUS-INJECTION-NOT-RESPONSE-v1"
GEOMETRY = "diagnostic-secondary-zero-step-box-v1"


def read(path, *, readback=False):
    """Select the highest cycle independently for every required tree."""
    tables = capture.read(path)
    with uproot.open(path) as root:
        if readback:
            require(PATHS[CALLBACKS] not in root and zero.PATHS[zero.CALLBACKS] not in root,
                    "readback must not contain a callback tree")
        for name in ((EVENTS, SIMS) if readback else (CALLBACKS, EVENTS, SIMS)):
            require(PATHS[name] in root, "missing tree " + PATHS[name])
            tables[name] = {field: ak.to_numpy(value, allow_missing=False) for field, value in
                            root[PATHS[name]].arrays(library="ak", how=dict).items()}
    return tables


def _schema(tables, *, baseline=False, readback=False):
    names = capture.TABLES if baseline else (*capture.TABLES, EVENTS, SIMS)
    if not baseline and not readback:
        names = (*names, CALLBACKS)
    require(set(tables) == set(names), "missing or unexpected diagnostic tables")
    for name in names:
        for field, value in tables[name].items():
            require(not np.ma.isMaskedArray(value), name + ": masked " + field)
    result = capture_schema(tables)
    for name in capture.TABLES:
        extras = set(tables[name]) - set(result[name])
        require(extras <= {"n" + field for field in VECTORS if field in result[name]},
                name + ": unexpected fields")
        for field in extras:
            value = np.asarray(tables[name][field])
            require(value.ndim == 1 and value.dtype.kind == "i" and np.array_equal(
                value, ak.to_numpy(ak.num(result[name][field[1:]], axis=1))), "invalid metadata count " + field)
            result[name][field] = value
        for field in VECTORS:
            if field in result[name]:
                require(np.all(ak.to_numpy(ak.num(result[name][field], axis=1)) == VECTORS[field]),
                        name + ": invalid metadata shape " + field)
                result[name][field] = ak.to_numpy(result[name][field], allow_missing=False)
    for name in set(names) - set(capture.TABLES):
        schema, table = SCHEMAS[name], tables[name]
        require(set(table) == {field for fields in schema.values() for field in fields},
                name + ": missing or unexpected fields")
        event = np.asarray(table["event"])
        require(event.ndim == 1, name + ": malformed event")
        result[name] = {}
        for kind, fields in schema.items():
            for field in fields:
                value = np.asarray(table[field])
                require(value.ndim == 1 and len(value) == len(event), name + ": malformed " + field)
                good = {"int64": value.dtype.kind == "i" and value.dtype.itemsize == 8,
                        "int": value.dtype.kind == "i", "bool": value.dtype.kind == "b",
                        "float": value.dtype.kind == "f", "str": value.dtype.kind in "OU"}[kind]
                require(good, name + ": invalid type " + field)
                if kind == "float":
                    require(np.isfinite(value).all(), name + ": nonfinite " + field)
                if kind == "str":
                    require(not len(value) or pd.api.types.infer_dtype(value, skipna=False) == "string",
                            name + ": invalid string " + field)
                result[name][field] = value
    return result


def _check(table, expected, label):
    for field, value in expected.items():
        require(np.all(table[field] == value), label + ": " + field + " mismatch")


def _select(table, mask):
    return {field: value[mask] for field, value in table.items()}


def _order(left, right, keys, label, *, full=True):
    lr, rr = frame(left, keys), unique(right, keys, label)
    if full:
        unique(left, keys, label)
        require(len(lr) == len(rr), label + ": row census mismatch")
    rr = rr.assign(_row=np.arange(len(rr), dtype=np.int64))
    return match(lr, rr, keys, label)["_row"].to_numpy(dtype=np.int64)


def _equal(left, right, keys, label, fields=None):
    order = _order(left, right, keys, label)
    if fields is None:
        require(set(left) == set(right), label + ": field census mismatch")
        fields = left
    for field in fields:
        require(np.array_equal(left[field], right[field][order]), label + ": " + field + " mismatch")


def _census(table, label, *, event=False):
    _check(table, dict(run=10, subrun=0, event=1), label)
    keys = capture.EVENT if event else [*capture.EVENT, "index"]
    unique(table, keys, label)
    require(len(table["event"]) == 1 if event else np.array_equal(
        np.sort(table["index"]), np.arange(len(table["event"]), dtype=np.int64)), label + ": row census mismatch")


def _audit(tables):
    _census(tables["events"], "capture event", event=True)
    require(np.all(tables["events"]["cpuTime"] >= 0) and np.all(tables["events"]["realTime"] >= 0),
            "negative StatusG4 timing")
    children = np.count_nonzero(tables["tracks"]["parent"] != 0)
    for name in ("tracks", "steps"):
        counts = tables[name]["secondaries"]
        require(np.all((counts >= 0) & (counts <= children)), name + ": impossible secondary count")
    return capture.audit(tables, 1)


def _baseline(tables):
    tables = _schema(tables, baseline=True)
    audit = _audit(tables)
    t, s = tables["tracks"], tables["steps"]
    primary, children = _select(t, t["parent"] == 0), _select(t, t["parent"] == 1)
    _check(primary, dict(track=1, index=0, pdg=-13, steps=9, status=2, reason=1), "baseline primary")
    require(len(children["track"]) > 0, "baseline has no direct children")
    steps = _select(s, s["track"] == 1)
    require(np.array_equal(np.sort(steps["number"]), np.arange(1, 10)), "baseline primary step sequence")
    require(steps["secondaries"].sum(dtype=np.int64) == len(children["track"]), "baseline direct child census")
    return tables, audit, primary, steps, children


def _records(table, keys):
    return frame(table, list(table)).sort_values(keys).to_dict(orient="records")


def inspect_baseline(tables):
    """Export the full direct-child birth census, not a selected donor or response."""
    _, audit, primary, steps, children = _baseline(tables)
    fields = [*capture.EVENT, "track", "parent", "pdg", "creator",
              *(p + f for p in ("begin_", "entrance_") for f in capture.STATE)]
    return {"diagnostic_only": True, "bank_admissible": False, "physical_response": False,
            "baseline_audit": audit, "run": 10, "subrun": 0, "event": 1,
            "primary_rows": _records(primary, capture.TRACK),
            "primary_step_rows": _records(steps, [*capture.TRACK, "number"]),
            "direct_child_count": len(children["track"]), "direct_child_pdgs": children["pdg"].tolist(),
            "direct_child_birth_rows": _records({f: children[f] for f in fields}, capture.TRACK)}


def _capture(tables, baseline, primary, steps, children, mode):
    audit = _audit(tables)
    e, t, s, c = (tables[n] for n in capture.TABLES[:4])
    _check(e, dict(physicsTag=PHYSICS, geometryTag=GEOMETRY), "diagnostic tags")
    for field in ("envelope", *VECTORS):
        if field in e:
            require(np.array_equal(e[field], baseline["events"][field]), "baseline geometry " + field)
    _equal(tables["volumes"], baseline["volumes"], [*capture.EVENT, "index"], "baseline volumes")
    _equal(_select(t, t["parent"] == 0), primary, capture.TRACK, "baseline primary")
    _equal(s, steps, [*capture.TRACK, "number"], "baseline primary steps")
    child = t["parent"] != 0
    _check(_select(t, child), dict(parent=1, steps=0, secondaries=0, status=2, reason=1, stopProcess=""), "children")
    birth = [*capture.EVENT, "track", "parent", "pdg", "creator",
             *(p + f for p in ("begin_", "entrance_") for f in capture.STATE)]
    _equal(_select(t, child), children, capture.TRACK, "baseline child birth", birth)
    _check(c, dict(status=np.where((c["parent"] != 0) & (mode == "classification"), 2, 0), killed=False),
           "capture classification")
    return audit


def _callbacks(tables, mode):
    b, t, s, c = (tables[n] for n in (CALLBACKS, "tracks", "steps", "classifications"))
    _census(b, CALLBACKS)
    p, child = b["phase"], b["parent"] != 0
    require(np.isin(p, [0, 1, 2, 3]).all() and np.all(~child | (p != 3)), "callback phases")
    tr = _order(b, t, capture.TRACK, "callback track", full=False)
    _check(b, dict(parent=t["parent"][tr], pdg=t["pdg"][tr], mode=mode, creator=t["creator"][tr],
                   creatorPresent=child, diagnosticOnly=True, bankAdmissible=False, g4Version=1132,
                   forwarded=True, error="", forced=False, killed=False, killedValid=p == 0,
                   cutEvaluated=p == 3, cutResult=False, weight=1., dynamicMass=t["begin_mass"][tr]), CALLBACKS)
    require(np.all(b["creator"][child] != "") and np.all(b["track_kinetic"][child] > 0), "child creator/positive KE")
    phase_rows = {}
    for phase, prefix, source in ((0, "state_", c), (1, "begin_", t), (2, "end_", t)):
        part = _select(b, p == phase)
        order = _order(part, source, capture.TRACK, "callback phase " + str(phase))
        for field in STATE:
            _check(part, {"track_" + field: source[prefix + field][order]}, "callback state")
        if phase in (0, 1):
            require(np.array_equal(source["index"][order][np.argsort(part["index"])],
                                   np.arange(len(order), dtype=np.int64)), "capture callback order")
        phase_rows[phase] = _order(t, part, capture.TRACK, "callback phase order")
        phase_rows[phase] = part["index"][phase_rows[phase]]
    require(np.all(phase_rows[0] < phase_rows[1]) and np.all(phase_rows[1] < phase_rows[2]), "callback lifecycle order")
    parent_end = phase_rows[2][t["track"] == 1][0]
    _check(b, dict(parentEnded=child, parentEndIndex=np.where(child, parent_end, -1)), "parent completion")
    require(np.all(b["index"][child] > parent_end), "child before parent completion")
    require(b["index"][child & (p == 0)].max() < b["index"][child & (p == 1)].min(),
            "all child classifications must precede child begins")
    stepped = _select(b, p == 3)
    stepped["number"] = stepped["steps"]
    sr = _order(stepped, s, [*capture.TRACK, "number"], "primary step callbacks")
    require(np.array_equal(stepped["steps"][np.argsort(stepped["index"])], np.sort(s["number"])), "callback step sequence")
    require(np.all(stepped["index"] > phase_rows[1][t["track"] == 1][0]) and
            np.all(stepped["index"] < parent_end), "primary step lifecycle order")
    for field in STATE:
        _check(stepped, {"track_" + field: s["post_" + field][sr]}, "primary step track")
    for prefix in ("pre_", "post_"):
        expected = {prefix + field: s[prefix + field][sr] for field in STATE}
        expected.update({prefix + "status": s[prefix + "stepStatus"][sr],
                         prefix + "process": s[prefix + "process"][sr], prefix + "present": True})
        _check(stepped, expected, "primary step point")
    status = np.where(p == 2, 2, 0)
    status[child & (p == 1) & (mode == "classification")] = 2
    status[p == 3] = s["trackStatus"][sr]
    forward = status.copy()
    forward[child & (p == 0) & (mode == "classification")] = 2
    out = forward.copy()
    out[child & (p == 1)] = 2
    expected_secondaries = np.zeros(len(p), dtype=np.int64)
    expected_secondaries[p == 3] = s["secondaries"][sr]
    _check(b, dict(statusIn=status, statusForward=forward, statusOut=out, stepSecondaries=expected_secondaries,
                   mutated=child & (p == (0 if mode == "classification" else 1))), "callback status")
    expected_steps = np.where(p == 2, t["steps"][tr], 0)
    expected_steps[p == 3] = s["number"][sr]
    lengths = np.cumsum(s["length"][np.argsort(s["number"])])
    expected_length = np.zeros(len(p))
    expected_length[(p == 2) & ~child] = lengths[-1]
    expected_length[p == 3] = lengths[s["number"][sr] - 1]
    _check(b, dict(steps=expected_steps, trackLength=expected_length), "callback counters")
    info = p > 0
    defaults = dict(userInfoPresent=info, mu2eInfoPresent=info, infoCode=np.where(info, "uninitialized", ""),
                    infoKE=0., infoDx=0., infoDy=0., infoDz=0., infoTime=np.where(info, -1., 0.),
                    infoProperTime=np.where(info, -1., 0.))
    _check(_select(b, child), {f: v[child] if isinstance(v, np.ndarray) else v for f, v in defaults.items()}, "child info")
    require(np.all(~b["mu2eInfoPresent"] | b["userInfoPresent"]), "Mu2e info without user info")
    _check(_select(b, ~b["mu2eInfoPresent"]), dict(infoCode="", forced=False, infoKE=0., infoDx=0.,
               infoDy=0., infoDz=0., infoTime=0., infoProperTime=0.), "absent Mu2e info")
    require(np.all(b["stepPresent"][(p == 2) | (p == 3)]), "missing engine step")
    for prefix in ("pre_", "post_"):
        present = b[prefix + "present"]
        require(np.all(~present | b["stepPresent"]), "point without step")
        require(np.all(b[prefix + "processPresent"] == (b[prefix + "process"] != "")), "process presence mismatch")
        require(np.all(present[p == 2]), "missing endpoint point")
        primary_end = _select(b, ~child & (p == 2))
        last = np.argmax(s["number"])
        _check(primary_end, {prefix + f: s[prefix + f][last] for f in STATE}, "primary end step point")
        _check(primary_end, {prefix + "status": s[prefix + "stepStatus"][last],
                             prefix + "process": s[prefix + "process"][last]}, "primary end step point")
        _check(_select(b, ~present), {prefix + "status": -1, prefix + "processPresent": False,
                                    **{prefix + f: 0. for f in STATE}}, "absent point")
        end = _select(b, child & (p == 2))
        _check(end, {prefix + "status": 7, prefix + "process": "", prefix + "processPresent": False,
                     **{prefix + f: end["track_" + f] for f in STATE}}, "child end point")
    for field in STATE:
        _check(_select(b, child), {"track_" + field: t["begin_" + field][tr[child]]}, "constant child state")


def _stored(value, field):
    return value if field in ("x", "y", "z") else value.astype(np.float32).astype(np.float64)


def _finalization(tables, mode):
    e, s, t = (tables[n] for n in (EVENTS, SIMS, "tracks"))
    _census(e, EVENTS, event=True)
    _census(s, SIMS)
    _check(e, dict(nSims=len(t["track"]), diagnosticOnly=True, bankAdmissible=False,
                   capturePresent=True, simsPresent=True, statusPresent=True), EVENTS)
    tr = _order(s, t, capture.TRACK, "finalized track")
    unique(s, [*capture.EVENT, "key"], "SimParticle key")
    child = s["parent"] != 0
    _check(s, dict(key=s["track"], parent=t["parent"][tr], pdg=t["pdg"][tr], nDaughters=t["secondaries"][tr],
                   simStage=0, endDefined=True, isPrimary=~child, genPresent=~child, parentPresent=child,
                   nSteps=t["steps"][tr], endStatus=2, startStatus=np.where(child & (mode == "classification"), 2, 0)), SIMS)
    _check(_select(s, child), dict(creationCode=t["creator"][tr[child]], stoppingCode="NotSpecified", trackLength=0.), "child SimParticle")
    steps = tables["steps"]
    primary_length = np.cumsum(steps["length"][np.argsort(steps["number"])])[-1:]
    _check(_select(s, ~child), dict(creationCode="mu2ePrimary", stoppingCode="Decay",
               trackLength=_stored(primary_length, "trackLength")), "primary SimParticle")
    # Start states use current tracks for every particle; stepped endpoint kinematics remain exempt.
    for field in SIM_STATE:
        _check(s, {"start_" + field: _stored(t["begin_" + field][tr], field)}, "SimParticle start")
    rows = frame(_select(s, child), capture.TRACK)
    matches = np.ones(np.count_nonzero(child), dtype=bool)
    errors = {}
    for field in (*STATE, "trackLength"):
        observed = s["trackLength" if field == "trackLength" else "endKE" if field == "kinetic" else "end_" + field][child]
        raw = np.zeros(len(observed)) if field == "trackLength" else t["end_" + field][tr[child]]
        expected = _stored(raw, field)
        good = observed == expected
        matches &= good
        errors[field] = int(np.count_nonzero(~good))
        rows["captured_" + field], rows["expected_" + field], rows["sim_" + field] = raw, expected, observed
        rows["delta_" + field], rows["match_" + field] = observed - expected, good
    rows["endpoint_match"] = matches
    return bool(matches.all()), errors, rows.sort_values(capture.TRACK).to_dict(orient="records")


def report(tables, readback, baseline, mode):
    require(mode in ("classification", "begin"), "invalid mode")
    baseline, ba, primary, steps, children = _baseline(baseline)
    tables, readback = _schema(tables), _schema(readback, readback=True)
    ca = _capture(tables, baseline, primary, steps, children, mode)
    ra = _capture(readback, baseline, primary, steps, children, mode)
    _callbacks(tables, mode)
    matched, errors, endpoints = _finalization(tables, mode)
    _finalization(readback, mode)
    for name in (*capture.TABLES, EVENTS, SIMS):
        _equal(tables[name], readback[name], capture.EVENT if name in ("events", EVENTS) else [*capture.EVENT, "index"],
               "readback " + name)
    return {"diagnostic_only": True, "bank_admissible": False, "physical_response": False, "real_engine": True,
            "engine_lifecycle_passed": True, "readback_identical": True, "primary_baseline_identical": True,
            "direct_child_birth_baseline_identical": True, "volume_inventory_identical": True,
            "endpoint_roundtrip_match": matched, "endpoint_kinematics_match": matched,
            "endpoint_storage": "double positions; float32 round-trip momentum, energy, KE, times, length",
            "endpoint_field_errors": errors, "endpoint_rows": endpoints, "mode": mode,
            "run": 10, "subrun": 0, "event": 1, "events": 1, "direct_child_count": len(children["track"]),
            "direct_child_pdgs": children["pdg"].tolist(), "baseline_audit": ba, "capture_audit": ca, "readback_audit": ra,
            "stacking_decision": "fUrgent is source-derived, not directly measured",
            "rows": {name: len(table["event"]) for name, table in tables.items()},
            "callback_rows": _records(tables[CALLBACKS], [*capture.EVENT, "index"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--readback", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--mode", choices=("classification", "begin"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-baseline", action="store_true")
    parser.add_argument("--require-endpoint-match", action="store_true")
    args = parser.parse_args(argv)
    if args.inspect_baseline:
        require(args.input is None and args.readback is None and args.mode is None and not args.require_endpoint_match,
                "baseline inspection accepts only --baseline and --output")
    else:
        require(args.input is not None and args.readback is not None and args.mode is not None,
                "input, --readback and --mode are required")
    for path in (args.input, args.readback, args.baseline):
        if path is not None:
            require(args.output.resolve() != path.resolve() and not (args.output.exists() and path.exists() and
                    args.output.samefile(path)), "output must not overwrite input")
    require(not args.output.exists() and not args.output.is_symlink(), "output already exists")
    baseline = capture.read(args.baseline)
    result = inspect_baseline(baseline) if args.inspect_baseline else report(
        read(args.input), read(args.readback, readback=True), baseline, args.mode)
    if args.require_endpoint_match:
        require(result["endpoint_roundtrip_match"], "finalized endpoint round-trip mismatch")
    result.update(baseline=str(args.baseline), endpoint_match_required=args.require_endpoint_match)
    if args.input is not None:
        result.update(input=str(args.input), readback=str(args.readback))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
