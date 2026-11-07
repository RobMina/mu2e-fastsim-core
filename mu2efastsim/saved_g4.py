"""Audit saved-product exports; never promote incomplete histories to component-response trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

EVENT = ["run", "subrun", "event"]
COLLECTION = [*EVENT, "product_id"]
ROW = [*COLLECTION, "index"]
SIM = [*EVENT, "sim_product_id", "sim_key"]
KINDS = ("particles", "generators", "crv_steps", "step_points", "trajectories", "surface_steps", "status")
TABLES = ("events", "collections", *KINDS, "daughters", "trajectory_points")


def read(path, directory="dump"):
    """Read the highest ROOT tree cycles from one bounded export file."""
    with uproot.open(path) as root:
        return {name: root[f"{directory}/{name}"].arrays(library="np") for name in TABLES}


def frame(table, columns):
    return pd.DataFrame({k: table[k] for k in columns})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(table, columns, label):
    data = frame(table, columns)
    require(not data.duplicated().any(), f"duplicate {label} identity")
    return data


def match(left, right, on, label):
    merged = left.merge(right, on=on, how="left", validate="many_to_one", indicator=True)
    require(merged["_merge"].eq("both").all(), f"unmatched {label} identity")
    return merged.drop(columns="_merge")


def census(tables):
    """Validate integer identities, completed event rows and collection/trajectory accounting."""
    for name, table in tables.items():
        for key in EVENT:
            require(np.asarray(table[key]).dtype.kind in "iu", f"{name}: noninteger {key}")
        for key in ("product_id", "index", "sim_product_id", "sim_key", "parent_key", "parent_product_id",
                    "gen_key", "gen_product_id", "daughter_key", "daughter_product_id", "point_index", "daughter_index"):
            if key in table:
                require(np.asarray(table[key]).dtype.kind in "iu", f"{name}: noninteger {key}")
        for key in (*EVENT, "sim_key", "daughter_index"):
            if key in table:
                require(np.all(table[key] >= 0), f"{name}: negative {key}")
        if "sim_product_id" in table:
            require(np.all(table["sim_product_id"] > 0), f"{name}: invalid particle ProductID")
    events = unique(tables["events"], EVENT, "event")
    e = tables["events"]
    require(np.all(e["schema_version"] == 1), "unsupported saved-G4 schema")
    require(not np.any(e["complete_boundary_records"]), "raw export cannot certify complete boundary records")
    collections = unique(tables["collections"], [*EVENT, "kind", "tag"], "collection")
    collections = frame(tables["collections"], [*EVENT, "kind", "tag", "product_id", "present", "count"])
    match(collections, events, EVENT, "collection event")
    require(collections["kind"].isin(KINDS).all(), "unknown collection kind")
    require((collections["present"] | collections["count"].eq(-1)).all(), "missing product must have count -1")
    require((~collections["present"] | collections["count"].ge(0)).all(), "present product has negative count")
    require((collections["present"] == collections["product_id"].gt(0)).all(), "product ID/presence disagreement")
    require(not collections.loc[collections["present"]].duplicated([*COLLECTION, "kind"]).any(),
            "aliased collection tags select one product twice")
    observed = collections.groupby(EVENT).agg(n_collections=("kind", "size"), n_present=("present", "sum")).reset_index()
    observed["n_missing"] = observed["n_collections"] - observed["n_present"]
    expected = frame(e, [*EVENT, "n_collections", "n_missing"])
    compare = expected.merge(observed, on=EVENT, how="left", suffixes=("_expected", "_observed"), validate="one_to_one").fillna(0)
    for field in ("n_collections", "n_missing"):
        require(np.array_equal(compare[f"{field}_expected"], compare[f"{field}_observed"]), f"event {field} mismatch")
    for kind in KINDS:
        rows = unique(tables[kind], ROW, kind)
        col = collections.loc[collections["kind"].eq(kind) & collections["present"], [*COLLECTION, "count"]]
        joined = match(rows, col, COLLECTION, f"{kind} collection")
        require((joined["index"].ge(0) & joined["index"].lt(joined["count"])).all(), f"{kind} index out of range")
        counts = rows.groupby(COLLECTION).size().rename("observed").reset_index()
        joined = col.merge(counts, on=COLLECTION, how="left", validate="one_to_one").fillna({"observed": 0})
        require(np.array_equal(joined["count"], joined["observed"]), f"{kind} census mismatch")
    parts = unique(tables["particles"], SIM, "particle")
    require(np.array_equal(tables["particles"]["product_id"], tables["particles"]["sim_product_id"]),
            "particle collection/reference ProductID mismatch")
    for kind in ("crv_steps", "step_points", "trajectories", "surface_steps", "daughters"):
        match(frame(tables[kind], SIM), parts, SIM, f"{kind} particle")
    trajectory = frame(tables["trajectories"], [*ROW, "sim_product_id", "sim_key", "n_points"])
    point = unique(tables["trajectory_points"], [*ROW, "point_index"], "trajectory point")
    point = frame(tables["trajectory_points"], [*ROW, "sim_product_id", "sim_key", "point_index"])
    joined = match(point, trajectory, [*ROW, "sim_product_id", "sim_key"], "trajectory point header")
    require((joined["point_index"].ge(0) & joined["point_index"].lt(joined["n_points"])).all(), "point index out of range")
    counts = point.groupby(ROW).size().rename("observed").reset_index()
    joined = trajectory.merge(counts, on=ROW, how="left", validate="one_to_one").fillna({"observed": 0})
    require(np.array_equal(joined["n_points"], joined["observed"]), "trajectory point census mismatch")
    unique(tables["daughters"], [*ROW, "daughter_index"], "daughter")
    return collections


def link_counts(table, product, key, targets):
    """Count retained reference closure without converting integer IDs to floats."""
    links = frame(table, [*EVENT, product, key]).rename(columns={product: "sim_product_id", key: "sim_key"})
    active = links["sim_key"].ge(0)
    require((active == links["sim_product_id"].gt(0)).all(), "null pointer identity mismatch")
    merged = links.loc[active].merge(targets[SIM], on=SIM, how="left", validate="many_to_one", indicator=True)
    return {"links": int(active.sum()), "outside_export": int(merged["_merge"].ne("both").sum())}


def audit(tables):
    """Return measured retained-product coverage, not boundary completeness or physical efficiency."""
    collections = census(tables)
    parts = frame(tables["particles"], [*SIM, "pdg", "parent_key", "end_defined", "truncated"])
    flags = {"trajectories": "trajectory", "crv_steps": "crv", "step_points": "raw_step", "surface_steps": "surface"}
    for kind, name in flags.items():
        refs = frame(tables[kind], SIM).drop_duplicates()
        refs[name] = True
        parts = parts.merge(refs, on=SIM, how="left", validate="one_to_one")
        parts[name] = parts[name].eq(True)
    parts["crv_without_trajectory"] = parts["crv"] & ~parts["trajectory"]
    parts["count"] = 1
    species = parts.groupby("pdg", sort=True)[["count", *flags.values(), "crv_without_trajectory"]].sum().reset_index()
    links = {"parents": link_counts(tables["particles"], "parent_product_id", "parent_key", parts),
             "daughters": link_counts(tables["daughters"], "daughter_product_id", "daughter_key", parts)}
    gen = frame(tables["generators"], ROW).rename(columns={"product_id": "sim_product_id", "index": "sim_key"})
    links["generators"] = link_counts(tables["particles"], "gen_product_id", "gen_key", gen)
    surface = frame(tables["surface_steps"], ["surface_id", "surface_index"])
    raw = frame(tables["step_points"], ["tag", "volume"])
    status = tables["status"]
    result = {
        "events": len(tables["events"]["event"]),
        "rows": {kind: len(tables[kind]["event"]) for kind in TABLES},
        "missing_collections": int((~collections["present"]).sum()),
        "empty_collections": int((collections["present"] & collections["count"].eq(0)).sum()),
        "complete_component_bank": False,
        "nonfinite_fields": {name: {key: int(np.count_nonzero(~np.isfinite(value)))
                              for key, value in table.items() if np.asarray(value).dtype.kind == "f"
                              and np.any(~np.isfinite(value))}
                             for name, table in tables.items()},
        "retained_particle_coverage": species.to_dict("records"),
        "links": links,
        "truncated_particles": int(parts["truncated"].sum()),
        "undefined_particle_ends": int((~parts["end_defined"]).sum()),
        "raw_steps_by_tag_volume": raw.groupby(["tag", "volume"]).size().rename("count").reset_index().to_dict("records"),
        "surface_steps_by_id_index": surface.groupby(["surface_id", "surface_index"]).size().rename("count").reset_index().to_dict("records"),
        "g4_status": {"rows": len(status["event"]), "nonzero": int(np.count_nonzero(status["status"])),
                      "tracks": int(np.sum(status["n_tracks"])), "overflow_events": int(np.count_nonzero(status["overflow"])),
                      "killed_step_limit": int(np.sum(status["killed_step_limit"])), "killed_field": int(np.sum(status["killed_field"]))},
    }
    return result


def compare_crv(tables, legacy):
    """Compare common CrvStep fields at their saved precision on exactly the exported events."""
    steps = tables["crv_steps"]
    unique(steps, [*EVENT, "index"], "single CRV collection")
    events = frame(tables["events"], EVENT)
    old = frame(legacy, EVENT).merge(events.assign(selected=True), on=EVENT, how="left", validate="many_to_one")
    mask = old["selected"].eq(True).to_numpy()
    old = {key: value[mask] for key, value in legacy.items()}
    keys = frame(old, EVENT)
    keys["index"] = keys.groupby(EVENT).cumcount()
    newkeys = frame(steps, [*EVENT, "index"])
    require(len(keys) == len(newkeys), "legacy CRV row count mismatch")
    order = match(newkeys, keys.assign(row=np.arange(len(keys))), [*EVENT, "index"], "legacy CRV row")["row"].to_numpy()
    mapping = {"bar": ("bar", None), "sp": ("sim_key", None), "edep": ("visible", None),
               "t0": ("t0", None), "t1": ("t1", None), "p1": ("end_momentum_magnitude", None), "len": ("length", None)}
    for i, axis in enumerate("xyz"):
        mapping[f"{axis}0"] = ("x0", i)
        mapping[f"{axis}1"] = ("x1", i)
        mapping[f"p{axis}0"] = ("p0", i)
    if "sp_product_id" in old:
        mapping["sp_product_id"] = ("sim_product_id", None)
    mismatches = {}
    for name, (new, axis) in mapping.items():
        a = steps[new] if axis is None else steps[new][:, axis]
        b = old[name][order]
        if b.dtype.kind in "iu":
            bounds = np.iinfo(b.dtype)
            require(np.all((a >= bounds.min) & (a <= bounds.max)), f"legacy {name} integer overflow")
        a = a.astype(b.dtype, copy=False)
        bad = a != b
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            bad &= ~(np.isnan(a) & np.isnan(b))
        if np.any(bad):
            rows = np.flatnonzero(bad)[:3]
            mismatches[name] = {"count": int(np.count_nonzero(bad)), "new": a[rows].tolist(), "old": b[rows].tolist()}
    require(not mismatches, f"legacy CrvStep field mismatches: {mismatches}")
    return {"rows": len(order), "fields_equal_at_legacy_precision": list(mapping),
            "particle_ProductID_checked": "sp_product_id" in old}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = read(args.input)
    result = audit(tables)
    if args.legacy:
        with uproot.open(args.legacy) as root:
            result["legacy_crv_comparison"] = compare_crv(tables, root["dump/steps"].arrays(library="np"))
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
