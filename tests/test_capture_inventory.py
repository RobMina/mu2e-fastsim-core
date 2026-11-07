"""Synthetic inventory diagnostics; no transport or source-geometry certification."""
import json
import subprocess
import sys

import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_inventory
from mu2efastsim.saved_g4 import frame
from test_capture import BIG, tables  # noqa: F401 -- shared schema-1 fixture


@pytest.fixture
def uniform(tables):
    """Put both events in the same geometry while preserving their different outcomes."""
    tables["events"]["translation"][:] = 0
    tables["events"]["rotation"][:] = np.eye(3).ravel()
    tables["volumes"]["translation"][:] = 0
    tables["volumes"]["rotation"][:] = np.eye(3).ravel()
    for name, prefixes in capture.STATES.items():
        for prefix in prefixes:
            for field in ("x", "y", "z", "px", "py", "pz"):
                tables[name][prefix + field][:] = tables[name][prefix + "l" + field]
    return tables


def test_report_retains_primary_states_and_empty_outcome(uniform):
    uniform["steps"]["deposit"][0] = 2.7e-29
    uniform["steps"]["post_process"][:] = "Transportation"
    result = json.loads(json.dumps(capture_inventory.report(uniform, 2), allow_nan=False))
    assert result["diagnostic_only"] and not result["bank_admissible"]
    assert len(result["geometry_inventory"]) == 2
    assert result["geometry_inventory"][0]["id"] == BIG + 1
    assert result["geometry_inventory"][0]["rotation"] == np.eye(3).ravel().tolist()
    entrances, steps = result["primary_entrances"], result["primary_steps"]
    assert len(entrances) == 2 and len(steps) == 4
    assert entrances[1]["event"] == BIG + 2 and entrances[1]["track"] == BIG + 1
    assert entrances[0]["entrance_lz"] == -1 and entrances[0]["entrance_volume_name"] == "box"
    assert steps[0]["deposit"] == 2.7e-29 and not steps[0]["preInside"]
    assert steps[1]["deposit"] == 0 and steps[1]["length"] == 2
    assert steps[1]["pre_lz"] == -1 and steps[1]["post_lz"] == 1
    assert steps[1]["pre_material"] == "G4_Galactic" and steps[1]["post_material"] == "material"
    assert steps[1]["pre_volume_name"] == steps[1]["post_volume_name"] == "box"
    assert steps[1]["post_process"] == "Transportation" and steps[3]["deposit"] == 2
    for name in capture.TABLES:
        uniform[name] = {key: value[::-1].copy() for key, value in uniform[name].items()}
    assert capture_inventory.report(uniform, 2) == result


@pytest.mark.parametrize("identity", capture.EVENT)
def test_lookup_uses_every_event_key_without_rounding(tables, identity):
    # Deliberately different per-event names for the same >2**53 volume ID.
    volumes = {key: value.copy() for key, value in tables["volumes"].items()}
    states = frame(tables["steps"], [*capture.EVENT, "pre_volume", "post_volume"])
    for field in capture.EVENT:
        if field != identity:
            states[field] = BIG + 1
            volumes[field][:] = BIG + 1
    volumes["name"][:] = ["first", "world", "second", "world"]
    states.loc[1, "post_volume"] = -1
    resolved = capture_inventory.volume_names(states, volumes, ("pre_", "post_"))
    assert resolved["pre_volume_name"].tolist() == ["first", "first", "second", "second", "second"]
    assert resolved.loc[0, "pre_volume"] == BIG + 1
    assert resolved.loc[1, "post_volume_name"] is None
    states.loc[0, "pre_volume"] = BIG + 3
    with pytest.raises(ValueError, match="unmatched state volume"):
        capture_inventory.volume_names(states, volumes, ("pre_",))


def test_different_transformed_inventories_reject(tables):
    assert capture.audit(tables, 2)["events"] == 2
    with pytest.raises(ValueError, match="multiple geometry inventories"):
        capture_inventory.report(tables, 2)


@pytest.mark.parametrize("field", [*capture_inventory.GEOMETRY, "translation", "rotation"])
def test_every_inventory_field_is_compared_exactly(uniform, field):
    values = uniform["volumes"][field]
    if field in ("translation", "rotation"):
        values[3, 0] = np.nextafter(values[3, 0], np.inf)
    elif values.dtype.kind == "O":
        values[3] = "different"
    else:
        values[3] += 1
    assert capture.audit(uniform, 2)["events"] == 2
    with pytest.raises(ValueError, match="multiple geometry inventories"):
        capture_inventory.report(uniform, 2)


def test_missing_placement_is_not_silently_collapsed(uniform):
    uniform["volumes"] = {key: value[:3] for key, value in uniform["volumes"].items()}
    uniform["events"]["nVolumes"][1] = 1
    with pytest.raises(ValueError, match="multiple geometry inventories"):
        capture_inventory.report(uniform, 2)


def test_strict_audit_precedes_diagnostics(tables):
    tables["events"]["complete"][0] = False
    with pytest.raises(ValueError, match="complete"):
        capture_inventory.report(tables, 2)


def test_module_cli_and_no_report_on_failed_audit(tmp_path, uniform):
    source, output = tmp_path / "capture.root", tmp_path / "nested/report.json"
    with uproot.recreate(source) as root:
        for name, table in uniform.items():
            root.mktree("dump/" + name, table)
    command = [sys.executable, "-m", "mu2efastsim.capture_inventory", str(source),
               "--expected-events", "2", "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(output.read_text())
    assert result == json.loads(completed.stdout)
    assert result["primary_steps"][2]["run"] == BIG + 2
    output.unlink()
    uniform["events"]["complete"][0] = False
    with uproot.update(source) as root:
        root.mktree("dump/events", uniform["events"])
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "complete" in failed.stderr
    assert not output.exists()
