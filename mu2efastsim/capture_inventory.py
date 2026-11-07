"""Strictly audited capture geometry and primary traversal diagnostics, never a physics bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import capture
from .saved_g4 import frame, require

GEOMETRY = "id copy name logical motherLogical material density solid".split()
TRANSFORMS = {"translation": 3, "rotation": 9}


def inventory(volumes, expected_events):
    """Collapse only identical full inventories; transforms are relative to each placement's mother."""
    data = frame(volumes, GEOMETRY)
    for field, width in TRANSFORMS.items():
        values = np.asarray(volumes[field])
        require(values.shape == (len(data), width), f"invalid volume {field} shape")
        for axis in range(width):
            data[f"{field}_{axis}"] = values[:, axis]
    # Every full placement row must occur once per audited event, including matching multiplicities.
    require(data.groupby(list(data.columns), dropna=False).size().eq(expected_events).all(),
            "multiple geometry inventories: cannot collapse events with nonidentical full inventories")
    data = data.drop_duplicates().sort_values("id")
    for field, width in TRANSFORMS.items():
        columns = [f"{field}_{axis}" for axis in range(width)]
        data[field] = data[columns].to_numpy().tolist()
        data = data.drop(columns=columns)
    return data.to_dict("records")


def volume_names(states, volumes, prefixes):
    """Resolve event-local volume IDs without float identity conversion; -1 has no volume."""
    for prefix in prefixes:
        lookup = frame(volumes, [*capture.EVENT, "id", "name", "material"]).rename(columns={
            "id": prefix + "volume", "name": prefix + "volume_name", "material": prefix + "volume_material"})
        states = states.merge(lookup, on=[*capture.EVENT, prefix + "volume"], how="left",
                              validate="many_to_one", indicator=True)
        require((states[prefix + "volume"].eq(-1) | states["_merge"].eq("both")).all(),
                "unmatched state volume identity")
        states = states.drop(columns="_merge")
        for field in ("volume_name", "volume_material"):
            column = prefix + field
            states[column] = states[column].astype(object).where(states[column].notna(), None)
    return states


def report(tables, expected_events):
    """Audit first, then retain every primary entrance and step, including approach/zero deposit."""
    audited = capture.audit(tables, expected_events)
    geometry = inventory(tables["volumes"], expected_events)
    tracks = frame(tables["tracks"], [*capture.EVENT, *capture.FIELDS["tracks"].split(),
                                    *("entrance_" + field for field in capture.STATE)])
    primaries = tracks.loc[tracks["parent"].eq(0)].sort_values(capture.TRACK)
    steps = frame(tables["steps"], [*capture.EVENT, *capture.FIELDS["steps"].split(),
                                  *(prefix + field for prefix in ("pre_", "post_") for field in capture.STATE)])
    steps = steps.merge(primaries[capture.TRACK], on=capture.TRACK, how="inner", validate="many_to_one")
    steps = steps.sort_values([*capture.TRACK, "number"])
    return {"diagnostic_only": True, "bank_admissible": False, "audit": audited,
            "geometry_transform_convention": "placement-local to mother: translation in mm, rotation row-major",
            "geometry_density_unit": "g/cm^3", "geometry_inventory": geometry,
            "primary_entrances": volume_names(primaries, tables["volumes"], ("entrance_",)).to_dict("records"),
            "primary_steps": volume_names(steps, tables["volumes"], ("pre_", "post_")).to_dict("records")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(capture.read(args.input, args.directory), args.expected_events)
    result["input"] = str(args.input)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
