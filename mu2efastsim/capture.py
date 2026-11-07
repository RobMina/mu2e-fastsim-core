"""Experimental schema-1 capture-smoke audit; passing never certifies a physics bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .saved_g4 import frame, match, require, unique

EVENT = ["run", "subrun", "event"]
TRACK = [*EVENT, "track"]
TABLES = ("events", "tracks", "steps", "classifications", "volumes")
STATES = {"tracks": ("begin_", "end_", "entrance_"), "steps": ("pre_", "post_"),
          "classifications": ("state_",)}
FLOATS = "x y z px py pz lx ly lz lpx lpy lpz energy kinetic mass charge time properTime weight".split()
STATE = [*FLOATS, "volume", "placement", "material", "process", "stepStatus"]
FIELDS = {
    "events": "schema geometryTag physicsTag envelope translation rotation halfSize complete bankAdmissible "
              "recordComplete recordBankAdmissible recordRun recordSubrun recordEvent capturePresent statusPresent "
              "eventIDsMatch errors validationErrors nTracks nSteps nClassifications nVolumes nErrors g4Status "
              "statusClean nG4Tracks overflowSimParticles nKilledStepLimit nKilledByFieldPropagator cpuTime realTime",
    "tracks": "index track parent steps secondaries pdg status reason entered ended creator stopProcess",
    "steps": "index track number secondaries deposit nonIonizingDeposit length trackStatus preInside postInside firstExit",
    "classifications": "index track parent pdg status killed",
    "volumes": "index id copy name logical motherLogical material solid density translation rotation",
}
INT64 = set(EVENT + "index track parent steps secondaries number id copy recordRun recordSubrun recordEvent "
            "nTracks nSteps nClassifications nVolumes nErrors".split())


def read(path, directory="dump"):
    """Read highest ROOT cycles; retain ragged metadata for mixed missing/present diagnostic products."""
    def column(key, value):
        if key in ("translation", "rotation", "halfSize"):
            counts = ak.to_numpy(ak.num(value, axis=1))
            if len(counts) and np.any(counts != counts[0]):
                return value
        return ak.to_numpy(value)
    with uproot.open(path) as root:
        return {name: {key: column(key, value) for key, value in
                       root[f"{directory}/{name}"].arrays(library="ak", how=dict).items()} for name in TABLES}


def vectors(table, prefix, fields):
    return np.column_stack([table[prefix + field] for field in fields])


def same(a, b, label, atol):
    require(np.allclose(a, b, rtol=0, atol=atol), label + " mismatch")


def state_equal(a, ap, ai, b, bp, bi, atol):
    same(vectors(a, ap, FLOATS)[ai], vectors(b, bp, FLOATS)[bi], "state continuity", atol)


def audit(tables, expected_events, *, atol=1e-7):
    """Fail closed on missing/partial events; atol is an absolute geometry/state tolerance (mm, MeV, ns)."""
    require(isinstance(expected_events, (int, np.integer)) and expected_events > 0, "positive expected events required")
    require(np.isfinite(atol) and 0 < atol <= 1e-5, "invalid audit tolerance")
    for name in TABLES:
        require(name in tables, f"missing table {name}")
        table = tables[name]
        fields = [*EVENT, *FIELDS[name].split(),
                  *(prefix + field for prefix in STATES.get(name, ()) for field in STATE)]
        require(set(fields) <= set(table), f"missing {name} fields: {set(fields) - set(table)}")
        for key in fields:
            try:
                value = np.asarray(table[key])
            except ValueError as exc:
                raise ValueError(f"{name}: malformed {key}") from exc
            require(value.ndim > 0 and len(value) == len(table["event"]), f"{name}: malformed {key}")
            if key in INT64 or key.endswith("_volume"):
                require(value.dtype.kind == "i" and value.dtype.itemsize == 8, f"{name}: {key} must be int64")
            if value.dtype.kind == "f":
                require(np.isfinite(value).all(), f"{name}: nonfinite {key}")
        require((frame(table, EVENT) >= 0).all().all(), f"{name}: negative event identity")
    e, t, s, c, v = (tables[name] for name in TABLES)
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    require(len(events) == expected_events, "expected event census mismatch")
    require(np.all(e["schema"] == 1), "unsupported capture schema")
    for field in "complete recordComplete capturePresent statusPresent eventIDsMatch statusClean".split():
        require(np.all(e[field] == True), f"event {field} is not true")
    for field in "bankAdmissible recordBankAdmissible overflowSimParticles nErrors g4Status nKilledStepLimit nKilledByFieldPropagator".split():
        require(np.all(e[field] == 0), f"invalid event {field}")
    for field in ("errors", "validationErrors"):
        require(np.all(e[field] == ""), f"event {field} present")
    for field in ("geometryTag", "physicsTag", "envelope"):
        require(np.all(e[field] != ""), f"empty {field}")
    for field in EVENT:
        require(np.array_equal(e[field], e["record" + field.capitalize()]), "record event identity mismatch")
    event_rows = {}
    for name in TABLES[1:]:
        table = tables[name]
        rows = unique(table, [*EVENT, "index"], name)
        event_rows[name] = match(rows, events, EVENT, name)["_event"].to_numpy()
        counts = np.bincount(event_rows[name], minlength=len(events))
        require(np.array_equal(e["n" + name.capitalize()], counts), f"{name} census mismatch")
        require(np.all((table["index"] >= 0) & (table["index"] < counts[event_rows[name]])), f"{name} row index gap")
    require(np.all(e["nTracks"] > 0) and np.all(e["nVolumes"] > 0), "empty capture event")
    require(np.array_equal(e["nG4Tracks"], e["nTracks"]), "G4 track census mismatch")
    tracks = unique(t, TRACK, "track").assign(_track=np.arange(len(t["track"])))
    require(np.all(t["track"] > 0) and np.all(t["parent"] >= 0), "invalid track/parent identity")
    require(np.all(t["entered"] == True) and np.all(t["ended"] == True), "unclosed/unentered track")
    require(np.all(np.isin(t["reason"], [1, 2])), "unsupported terminal reason")
    require(np.all(np.where(t["reason"] == 1, t["status"] == 2, np.isin(t["status"], [0, 1]))), "terminal status mismatch")
    primary = t["parent"] == 0
    require(np.all(np.bincount(event_rows["tracks"], weights=primary, minlength=len(events)) == 1), "expected one primary per event")
    parents = frame(t, [*EVENT, "parent"]).rename(columns={"parent": "track"})
    parent_rows = match(parents.loc[~primary], tracks, TRACK, "parent")["_track"].to_numpy()
    require(np.all(t["index"][parent_rows] < t["index"][~primary]), "ancestry cycle/order mismatch")
    require(np.array_equal(np.bincount(parent_rows, minlength=len(tracks)), t["secondaries"]), "secondary ancestry census mismatch")
    unique(c, TRACK, "classification")
    classified = match(frame(c, TRACK), tracks, TRACK, "classification")["_track"].to_numpy()
    require(len(classified) == len(tracks) and np.all(c["killed"] == False), "classification acceptance census mismatch")
    require(np.all(np.isin(c["status"], [0, 1, 2])), "unsupported classification status")
    require(np.all((c["status"] != 2) | ((t["steps"][classified] == 0) & ~primary[classified])),
            "terminal classification requires a zero-step child")
    for field in ("parent", "pdg"):
        require(np.array_equal(c[field], t[field][classified]), f"classification {field} mismatch")
    state_equal(c, "state_", slice(None), t, "begin_", classified, atol)
    rows = match(frame(s, [*TRACK, "index", "number"]), tracks, TRACK, "step")
    rows = rows.sort_values([*EVENT, "index"])
    require(np.array_equal(rows["number"], rows.groupby(TRACK, sort=False).cumcount() + 1), "step sequence gap/order mismatch")
    sr = match(frame(s, TRACK), tracks, TRACK, "step")["_track"].to_numpy()
    require(np.all(t["steps"] >= 0) and np.array_equal(np.bincount(sr, minlength=len(tracks)), t["steps"]), "track step census mismatch")
    require(np.all(s["secondaries"] >= 0), "negative secondary count")
    child_counts = np.zeros(len(tracks), dtype=np.int64)
    np.add.at(child_counts, sr, s["secondaries"])
    require(np.array_equal(child_counts, t["secondaries"]), "step secondary census mismatch")
    zero_step = t["steps"] == 0
    require(np.all((t["reason"][zero_step] == 1) & ~primary[zero_step]), "unsupported zero-step track")
    state_equal(t, "begin_", zero_step, t, "end_", zero_step, atol)
    for field in ("deposit", "nonIonizingDeposit", "length"):
        require(np.all(s[field] >= 0), f"negative raw {field}")
    exit_mask = s["preInside"] & ~s["postInside"]
    require(np.array_equal(exit_mask, s["firstExit"]), "first exit flag mismatch")
    require(np.array_equal(np.bincount(sr[exit_mask], minlength=len(tracks)), (t["reason"] == 2).astype(int)), "first exit census mismatch")
    require(np.all(s["number"][exit_mask] == t["steps"][sr[exit_mask]]), "steps after first exit")
    require(np.all(np.isin(s["trackStatus"][exit_mask], [0, 1])), "nontransport exit status")
    ordered = rows.sort_values([*TRACK, "number"])
    consecutive = ordered["_track"].to_numpy()[1:] == ordered["_track"].to_numpy()[:-1]
    previous, following = ordered.index.to_numpy()[:-1][consecutive], ordered.index.to_numpy()[1:][consecutive]
    require(np.all(np.isin(s["trackStatus"][previous], [0, 1])), "unsupported intermediate track status")
    state_equal(s, "post_", previous, s, "pre_", following, atol)
    require(np.array_equal(s["postInside"][previous], s["preInside"][following]), "step inside continuity mismatch")
    first, last = rows.groupby(TRACK, sort=False).head(1), rows.groupby(TRACK, sort=False).tail(1)
    state_equal(t, "begin_", first["_track"], s, "pre_", first.index, atol)
    state_equal(t, "end_", last["_track"], s, "post_", last.index, atol)
    require(np.array_equal(s["trackStatus"][last.index], t["status"][last["_track"]]), "last step status mismatch")
    entry = ~s["preInside"] & s["postInside"]
    require(np.array_equal(np.bincount(sr[entry], minlength=len(tracks)), primary.astype(int)), "entry census mismatch")
    state_equal(t, "entrance_", sr[entry], s, "post_", entry, atol)
    state_equal(t, "entrance_", ~primary, t, "begin_", ~primary, atol)
    outside = ~s["preInside"]
    require(np.all((t["parent"][sr[outside]] == 0) & (s["secondaries"][outside] == 0) &
                   (s["pre_material"][outside] == "G4_Galactic")), "nontrivial outside transport")
    for field, width in (("translation", 3), ("rotation", 9), ("halfSize", 3)):
        require(np.shape(e[field]) == (len(events), width), f"invalid event {field} shape")
    rotation, translation, half = e["rotation"].reshape(-1, 3, 3), e["translation"], e["halfSize"]
    same(rotation @ rotation.transpose(0, 2, 1), np.eye(3), "orthonormal rotation", 1e-10)
    same(np.linalg.det(rotation), 1, "proper rotation", 1e-10)
    require(np.all(half > 0), "invalid box halfSize")
    volumes = unique(v, [*EVENT, "id"], "volume")
    require(np.all(v["id"] >= 0), "negative volume ID")
    envelope = v["name"] == e["envelope"][event_rows["volumes"]]
    require(np.all(np.bincount(event_rows["volumes"][envelope], minlength=len(events)) == 1), "envelope census mismatch")
    require(np.all(v["solid"][envelope] == "G4Box"), "envelope is not a box")
    for field in ("translation", "rotation"):
        same(v[field][envelope], e[field][event_rows["volumes"][envelope]], "envelope " + field, atol)
    for name, prefixes in STATES.items():
        table, er = tables[name], event_rows[name]
        for prefix in prefixes:
            for local, world, shift in (("l", "", translation[er]), ("lp", "p", 0)):
                calculated = np.einsum("nij,nj->ni", rotation[er], vectors(table, prefix + local, "xyz")) + shift
                same(calculated, vectors(table, prefix + world, "xyz"), f"{name} {prefix}{world} covariance", atol)
            require(np.all(table[prefix + "weight"] == 1), "biased state weight")
            refs = frame(table, [*EVENT, prefix + "volume"]).rename(columns={prefix + "volume": "id"})
            require(np.all(refs["id"] >= -1), "invalid state volume")
            match(refs.loc[refs["id"] >= 0], volumes, [*EVENT, "id"], "state volume")
    for prefix, inside in (("pre_", s["preInside"]), ("post_", s["postInside"])):
        distance = np.abs(vectors(s, prefix + "l", "xyz")) - half[event_rows["steps"]]
        require(np.all(np.where(inside, np.all(distance <= atol, axis=1), np.any(distance >= -atol, axis=1))), "inside/box mismatch")
    boundary = entry | exit_mask
    distance = np.abs(vectors(s, "post_l", "xyz")) - half[event_rows["steps"]]
    require(np.all(np.all(distance[boundary] <= atol, axis=1) & np.any(np.abs(distance[boundary]) <= atol, axis=1)), "crossing not on exact box boundary")
    require(np.all(s["post_stepStatus"][boundary] == 1), "crossing lacks geometry boundary status")
    begin_distance = np.abs(vectors(t, "begin_l", "xyz")) - half[event_rows["tracks"]]
    require(np.all(np.any(begin_distance[primary] > atol, axis=1)), "primary not born strictly outside")
    require(np.all(begin_distance[~primary] < -atol), "secondary not born strictly inside")
    require(np.all(s["post_process"] != "mu2eSpecialCutsProcess"), "resource kill step")
    return {"events": len(events), "rows": {name: len(tables[name]["event"]) for name in TABLES},
            "bank_admissible": False, "experimental": True, "first_exits": int(exit_mask.sum()),
            "natural_stops": int(np.count_nonzero(t["reason"] == 1)), "secondaries": int((~primary).sum()),
            "zero_deposit_steps": int(np.count_nonzero(s["deposit"] == 0)),
            "raw_deposit": float(np.sum(s["deposit"])), "raw_nonionizing_deposit": float(np.sum(s["nonIonizingDeposit"])),
            "component_deposit": float(np.sum(s["deposit"][s["preInside"]])),
            "approach_deposit": float(np.sum(s["deposit"][outside]))}


def diagnose(tables):
    """Summarize invalid smoke records without accepting them or changing any stored value."""
    e, t, s = (tables[name] for name in ("events", "tracks", "steps"))
    outside = ~s["preInside"]
    columns = [*EVENT, "track", "number", "deposit", "secondaries", "pre_material", "post_material",
               "preInside", "postInside", "pre_lz", "post_lz", "pre_kinetic", "post_kinetic", "post_process"]
    print(frame(e, [*EVENT, "complete", "errors", "nTracks", "nSteps"]).to_string(index=False))
    print(frame(s, columns).loc[outside].head(20).to_string(index=False))
    return {"diagnostic_only": True, "bank_admissible": False, "events": len(e["event"]),
            "tracks": len(t["track"]), "steps": len(s["track"]), "outside_steps": int(outside.sum()),
            "outside_deposit": float(s["deposit"][outside].sum()),
            "outside_secondaries": int(s["secondaries"][outside].sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true", help="report invalid data without certifying it")
    args = parser.parse_args(argv)
    tables = read(args.input, args.directory)
    result = diagnose(tables) if args.diagnose else audit(tables, args.expected_events)
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
