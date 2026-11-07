"""Synthetic zero-step schema tests, not evidence of an executed engine or persisted art products."""
import copy
import json

import awkward as ak
import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_zero_step as zero
from test_capture import BIG, empty


def fixture(run=15, endpoint=0.):
    tables = {}
    for name, count in (("events", 1), ("tracks", 1), ("steps", 0), ("classifications", 1), ("volumes", 3)):
        table = empty(name, 2 * count)
        table["run"][:] = run
        table["event"][:] = np.repeat(np.array([1, 2], dtype=np.int64), count)
        if name != "events":
            table["index"][:] = np.tile(np.arange(count, dtype=np.int64), 2)
        tables[name] = table
    e, t, c, v = (tables[name] for name in ("events", "tracks", "classifications", "volumes"))
    for field in "capturePresent statusPresent eventIDsMatch statusClean".split():
        e[field][:] = True
    for field in capture.EVENT:
        e["record" + field.capitalize()][:] = e[field]
    for field, value in dict(schema=1, nTracks=1, nClassifications=1, nVolumes=3, nG4Tracks=1, nErrors=1,
                             geometryTag="diagnostic-engine-zero-step-box-v1",
                             physicsTag="DIAGNOSTIC-STATUS-INJECTION-NOT-RESPONSE-v1", envelope="CaptureEnvelope",
                             errors="track_never_entered_component",
                             validationErrors="Capture record is incomplete\nCapture record contains errors").items():
        e[field][:] = value
    e["halfSize"][:] = 50.
    e["rotation"][:] = np.eye(3).ravel()
    e["cpuTime"][:] = [0.01, 0.02]
    e["realTime"][:] = [0.1, 0.2]
    v["id"][:] = v["copy"][:] = v["index"]
    v["name"][:] = v["logical"][:] = np.array(["World", "BoxInTheWorld", "CaptureEnvelope"], dtype=object)[v["index"]]
    v["motherLogical"][:] = np.array(["", "World", "BoxInTheWorld"], dtype=object)[v["index"]]
    v["material"][:] = np.where(v["index"] == 2, "G4_POLYSTYRENE", "G4_Galactic")
    v["density"][:] = np.where(v["index"] == 2, 1.06, 1e-25)
    v["solid"][:] = "G4Box"
    v["rotation"][:] = np.eye(3).ravel()
    t["track"][:] = t["reason"][:] = 1
    t["pdg"][:] = 22
    t["status"][:] = 2
    t["ended"][:] = True
    for prefix in ("begin_", "end_", "entrance_"):
        t[prefix + "weight"][:] = 1.
        t[prefix + "stepStatus"][:] = -1
        t[prefix + "volume"][:] = -1 if prefix == "entrance_" else 1
        if prefix != "entrance_":
            for field, value in dict(z=-50.001, lz=-50.001, pz=10., lpz=10., energy=10., kinetic=10.,
                                     material="G4_Galactic", placement="/0:0/1:1").items():
                t[prefix + field][:] = value
    c["track"][:] = 1
    c["pdg"][:] = 22
    c["status"][:] = [2, 0]
    for field in capture.STATE:
        c["state_" + field][:] = t["begin_" + field]
    c["state_volume"][:] = -1
    c["state_material"][:] = c["state_placement"][:] = ""

    for name, count in ((zero.CALLBACKS, 3), (zero.EVENTS, 1), (zero.SIMS, 1)):
        table = {}
        for kind, fields in zero.SCHEMAS[name].items():
            dtype = {"int64": np.int64, "int": np.int32, "bool": np.bool_, "float": np.float64, "str": object}[kind]
            for field in fields:
                table[field] = np.full(2 * count, "" if kind == "str" else 0, dtype=dtype)
        table["run"][:] = run
        table["event"][:] = np.repeat(np.array([1, 2], dtype=np.int64), count)
        if "index" in table:
            table["index"][:] = np.tile(np.arange(count, dtype=np.int64), 2)
        tables[name] = table
    b, f, s = (tables[name] for name in (zero.CALLBACKS, zero.EVENTS, zero.SIMS))
    b["phase"][:] = b["index"]
    b["track"][:] = 1
    b["pdg"][:] = 22
    b["g4Version"][:] = 1132
    b["diagnosticOnly"][:] = b["forwarded"][:] = True
    b["killedValid"][:] = b["phase"] == 0
    b["mutated"][:] = [True, False, False, False, True, False]
    b["statusIn"][:] = [0, 2, 2, 0, 0, 2]
    b["statusForward"][:] = [2, 2, 2, 0, 0, 2]
    b["statusOut"][:] = [2, 2, 2, 0, 2, 2]
    info = b["phase"] > 0
    b["userInfoPresent"][:] = b["mu2eInfoPresent"][:] = info
    b["infoCode"][:] = np.where(info, "uninitialized", "")
    b["infoTime"][:] = b["infoProperTime"][:] = np.where(info, -1., 0.)
    b["stepPresent"][:] = b["phase"] == 2
    for prefix in ("track_", "pre_", "post_"):
        present = True if prefix == "track_" else b["stepPresent"]
        for field, value in dict(z=-50.001, pz=10., energy=10., kinetic=10.).items():
            b[prefix + field][:] = np.where(present, value, 0.)
        if prefix != "track_":
            b[prefix + "present"][:] = present
            b[prefix + "status"][:] = np.where(present, 7, -1)
    for field in "diagnosticOnly capturePresent simsPresent statusPresent".split():
        f[field][:] = True
    f["nSims"][:] = 1
    s["key"][:] = s["track"][:] = 1
    s["pdg"][:] = 22
    s["startStatus"][:] = [2, 0]
    s["endStatus"][:] = 2
    for field in "endDefined isPrimary genPresent".split():
        s[field][:] = True
    s["creationCode"][:] = "mu2ePrimary"
    s["stoppingCode"][:] = "NotSpecified"
    s["start_z"][:] = s["end_z"][:] = -50.001
    s["start_pz"][:] = s["start_energy"][:] = 10.
    s["end_pz"][:] = s["end_energy"][:] = s["endKE"][:] = endpoint
    readback = copy.deepcopy({name: table for name, table in tables.items() if name != zero.CALLBACKS})
    return tables, readback


def check(tables=None, readback=None, run=15):
    if tables is None:
        tables, readback = fixture(run)
    return zero.report(tables, readback, run)


@pytest.mark.parametrize("endpoint,matched,predicted", [(0., False, True), (10., True, False), (3.25, False, False)])
def test_lifecycle_does_not_predetermine_endpoint(endpoint, matched, predicted):
    tables, readback = fixture(endpoint=endpoint)
    result = check(tables, readback)
    assert result["engine_lifecycle_passed"] and result["readback_identical"]
    assert result["real_engine"] and result["diagnostic_only"]
    assert not result["bank_admissible"] and not result["physical_response"]
    assert result["endpoint_kinematics_match"] is matched
    assert result["strict_audit_rejection"] == "event complete is not true"
    assert result["rows"] == dict(events=2, tracks=2, steps=0, classifications=2, volumes=6,
                                   zeroStepCallbacks=6, zeroStepEvents=2, zeroStepSims=2)
    assert result["endpoint_rows"][0]["endpoint_match"] is matched
    assert result["endpoint_rows"][1]["predicted_zero_endpoint"] is predicted
    assert result["endpoint_rows"][1]["delta_pz"] == endpoint - 10.
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="complete"):
        capture.audit(tables, 2)


@pytest.mark.parametrize("run", [BIG + 1, 2**63 - 1])
def test_full_int64_census_and_independent_row_order(run):
    tables, readback = fixture(run)
    tables = {name: {field: value[::-1] for field, value in table.items()} for name, table in tables.items()}
    result = json.loads(json.dumps(check(tables, readback, run)))
    assert result["endpoint_rows"][0]["run"] == run
    assert result["callback_rows"][0]["run"] == run
    assert result["callback_rows"][-1]["phase"] == 2
    with pytest.raises(ValueError, match="census"):
        check(tables, readback, run - 1)


@pytest.mark.parametrize("run", [-1, 0, True, 15., 2**63])
def test_invalid_run(run):
    tables, readback = fixture()
    with pytest.raises(ValueError, match="run"):
        check(tables, readback, run)


@pytest.mark.parametrize("name", [*capture.TABLES, zero.CALLBACKS, zero.EVENTS, zero.SIMS])
@pytest.mark.parametrize("field", capture.EVENT)
def test_full_identity_not_float_join(name, field):
    tables, readback = fixture(BIG + 1)
    if name == "steps":
        tables[name] = {key: np.repeat(value, 1, axis=0) for key, value in tables[name].items()}
        tables[name][field] = np.array([BIG + 2], dtype=np.int64)
    else:
        tables[name][field][0] += 1
    with pytest.raises(ValueError):
        check(tables, readback, BIG + 1)


@pytest.mark.parametrize("name", ["events", "tracks", "classifications", "volumes", zero.CALLBACKS, zero.EVENTS, zero.SIMS])
@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "index-gap"])
def test_exact_row_census(name, mode):
    tables, readback = fixture()
    table = tables[name]
    if mode in ("missing", "extra"):
        selection = np.arange(len(table["event"]) - 1) if mode == "missing" else np.r_[np.arange(len(table["event"])), 0]
        tables[name] = {field: value[selection] for field, value in table.items()}
    elif mode == "duplicate":
        table["event"][:] = 1
        if "index" in table:
            table["index"][:] = 0
    else:
        table["index" if "index" in table else "event"][0] = 99
    with pytest.raises(ValueError):
        check(tables, readback)


@pytest.mark.parametrize("name,field,dtype", [
    ("events", "run", float), ("events", "recordRun", float), ("events", "nTracks", np.int32),
    ("tracks", "track", np.int32), ("tracks", "ended", np.int64), ("tracks", "begin_volume", float),
    ("steps", "track", float), ("classifications", "killed", np.int32), ("volumes", "copy", float),
    (zero.CALLBACKS, "run", float), (zero.CALLBACKS, "index", np.int32),
    (zero.CALLBACKS, "stepPresent", np.int64), (zero.CALLBACKS, "phase", float),
    (zero.CALLBACKS, "track_energy", np.int32), (zero.EVENTS, "nSims", np.int32),
    (zero.EVENTS, "simsPresent", np.int32), (zero.SIMS, "simStage", np.int32),
    (zero.SIMS, "key", float), (zero.SIMS, "endDefined", np.int64), (zero.SIMS, "endKE", np.int64),
])
def test_strict_types(name, field, dtype):
    tables, readback = fixture()
    tables[name][field] = tables[name][field].astype(dtype)
    with pytest.raises(ValueError):
        check(tables, readback)


@pytest.mark.parametrize("name,field,value", [
    ("events", "complete", True), ("events", "recordComplete", True), ("events", "bankAdmissible", True),
    ("events", "recordBankAdmissible", True), ("events", "capturePresent", False),
    ("events", "statusClean", False), ("events", "g4Status", 1), ("events", "nG4Tracks", 0),
    ("events", "overflowSimParticles", True), ("events", "nKilledStepLimit", 1),
    ("events", "nKilledByFieldPropagator", 1), ("events", "cpuTime", -1.),
    ("events", "errors", "track_never_entered_component\nextra"), ("events", "validationErrors", ""),
    ("events", "geometryTag", "physical"), ("events", "physicsTag", "physical"),
    ("events", "recordRun", 16), ("events", "recordEvent", 2), ("events", "nSteps", 1),
    ("tracks", "begin_pz", 0.), ("tracks", "begin_lz", 0.), ("tracks", "end_kinetic", 0.),
    ("tracks", "end_properTime", -1.), ("tracks", "end_stepStatus", 7), ("tracks", "begin_volume", -1),
    ("tracks", "entrance_weight", 0.), ("tracks", "entrance_volume", 1), ("tracks", "entered", True),
    ("tracks", "ended", False), ("tracks", "steps", 1), ("tracks", "secondaries", 1),
    ("tracks", "status", 0), ("tracks", "reason", 2), ("tracks", "stopProcess", "mu2eSpecialCutsProcess"),
    ("classifications", "state_z", 0.), ("classifications", "state_volume", 0),
    ("classifications", "state_material", "G4_Galactic"), ("classifications", "status", 0),
    ("classifications", "killed", True), ("volumes", "density", 0.), ("volumes", "density", 2e-25),
    ("volumes", "name", "wrong"), ("volumes", "logical", "wrong"), ("volumes", "motherLogical", "wrong"),
    ("volumes", "copy", 1), ("volumes", "solid", "G4Tubs"), ("volumes", "material", "G4_AIR"),
    (zero.CALLBACKS, "phase", 1), (zero.CALLBACKS, "statusIn", 2), (zero.CALLBACKS, "statusForward", 0),
    (zero.CALLBACKS, "statusOut", 0), (zero.CALLBACKS, "mutated", False), (zero.CALLBACKS, "forwarded", False),
    (zero.CALLBACKS, "diagnosticOnly", False), (zero.CALLBACKS, "bankAdmissible", True),
    (zero.CALLBACKS, "g4Version", 1131), (zero.CALLBACKS, "killedValid", False), (zero.CALLBACKS, "killed", True),
    (zero.CALLBACKS, "error", "error"), (zero.CALLBACKS, "userInfoPresent", True),
    (zero.CALLBACKS, "mu2eInfoPresent", True), (zero.CALLBACKS, "forced", True),
    (zero.CALLBACKS, "infoCode", "uninitialized"), (zero.CALLBACKS, "infoKE", 10.),
    (zero.CALLBACKS, "infoTime", -1.), (zero.CALLBACKS, "infoDz", 1.),
    (zero.CALLBACKS, "track_z", 0.), (zero.CALLBACKS, "pre_pz", 10.),
    (zero.CALLBACKS, "pre_status", 7), (zero.CALLBACKS, "post_processPresent", True),
    (zero.CALLBACKS, "post_process", "transportation"),
    (zero.EVENTS, "nSims", 0), (zero.EVENTS, "simsPresent", False), (zero.EVENTS, "diagnosticOnly", False),
    (zero.EVENTS, "bankAdmissible", True), (zero.SIMS, "simStage", 4294967295),
    (zero.SIMS, "key", 2), (zero.SIMS, "nDaughters", 1), (zero.SIMS, "parentPresent", True),
    (zero.SIMS, "isPrimary", False), (zero.SIMS, "genPresent", False), (zero.SIMS, "endDefined", False),
    (zero.SIMS, "nSteps", 1), (zero.SIMS, "trackLength", 1.), (zero.SIMS, "pdg", 11),
    (zero.SIMS, "creationCode", "unknown"), (zero.SIMS, "stoppingCode", "unknown"),
    (zero.SIMS, "startStatus", 0), (zero.SIMS, "endStatus", 0), (zero.SIMS, "start_energy", 0.),
    (zero.SIMS, "end_z", 0.), (zero.SIMS, "end_time", -1.), (zero.SIMS, "end_properTime", -1.),
])
def test_reject_mutated_lifecycle(name, field, value):
    tables, readback = fixture()
    tables[name][field][0] = value
    with pytest.raises(ValueError):
        check(tables, readback)


@pytest.mark.parametrize("phase", [0, 1])
def test_early_step_presence_is_measured_not_assumed(phase):
    tables, readback = fixture()
    b = tables[zero.CALLBACKS]
    selection = b["phase"] == phase
    b["stepPresent"][selection] = True
    for prefix in ("pre_", "post_"):
        b[prefix + "present"][selection] = True
        b[prefix + "status"][selection] = 7
        for field in zero.STATE:
            b[prefix + field][selection] = b["track_" + field][selection]
    assert check(tables, readback)["engine_lifecycle_passed"]


@pytest.mark.parametrize("field", ["stepPresent", "pre_present", "post_present"])
def test_posttracking_requires_step_and_points(field):
    tables, readback = fixture()
    tables[zero.CALLBACKS][field][2] = False
    with pytest.raises(ValueError, match="posttracking"):
        check(tables, readback)


@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("field,value", [("userInfoPresent", False), ("mu2eInfoPresent", False),
    ("infoCode", ""), ("forced", True), ("infoKE", 10.), ("infoDx", 1.), ("infoDy", 1.),
    ("infoDz", 1.), ("infoTime", 0.), ("infoProperTime", 0.)])
def test_stored_user_information_is_uninitialized_without_steps(phase, field, value):
    tables, readback = fixture()
    b = tables[zero.CALLBACKS]
    b[field][b["phase"] == phase] = value
    with pytest.raises(ValueError, match=field):
        check(tables, readback)


@pytest.mark.parametrize("field,value", [("statusIn", 2), ("statusForward", 2),
                                         ("statusOut", 0), ("mutated", False)])
def test_second_event_injects_after_forwarding_begin(field, value):
    tables, readback = fixture()
    b = tables[zero.CALLBACKS]
    b[field][(b["event"] == 2) & (b["phase"] == 1)] = value
    with pytest.raises(ValueError, match=field):
        check(tables, readback)


def test_classification_can_have_initialized_navigation():
    tables, readback = fixture()
    for collection in (tables, readback):
        c = collection["classifications"]
        c["state_volume"][:] = 1
        c["state_material"][:] = "G4_Galactic"
        c["state_placement"][:] = "/0:0/1:1"
    assert check(tables, readback)["engine_lifecycle_passed"]


@pytest.mark.parametrize("field", ["endKE", "end_energy", "end_px", "end_py", "end_pz"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_endpoint_is_not_a_diagnostic_success(field, value):
    tables, readback = fixture()
    tables[zero.SIMS][field][0] = readback[zero.SIMS][field][0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        check(tables, readback)


@pytest.mark.parametrize("field", ["endKE", "end_energy"])
def test_negative_endpoint_energy_rejected(field):
    tables, readback = fixture()
    tables[zero.SIMS][field][0] = readback[zero.SIMS][field][0] = -0.001
    with pytest.raises(ValueError, match="negative"):
        check(tables, readback)


def test_arbitrary_finite_momentum_mismatch_and_signed_components_reported():
    tables, readback = fixture(endpoint=4.5)
    for collection in (tables, readback):
        collection[zero.SIMS]["end_px"][:] = [-2., 7.]
        collection[zero.SIMS]["end_pz"][:] = [-4., 3.]
    result = check(tables, readback)
    assert result["engine_lifecycle_passed"] and not result["endpoint_kinematics_match"]
    assert result["endpoint_rows"][0]["delta_pz"] == -14.
    assert result["endpoint_rows"][1]["delta_px"] == 7.


@pytest.mark.parametrize("name,field", [("events", "cpuTime"), ("events", "realTime"),
    ("volumes", "density"), ("tracks", "begin_z"), ("classifications", "state_energy"),
    (zero.SIMS, "endKE"), (zero.SIMS, "end_energy"), (zero.SIMS, "end_px")])
def test_readback_requires_exact_values_not_tolerant_physics_comparison(name, field):
    tables, readback = fixture()
    readback[name][field][0] = np.nextafter(readback[name][field][0], np.inf)
    with pytest.raises(ValueError, match="readback.*mismatch"):
        check(tables, readback)


@pytest.mark.parametrize("name", ["events", "tracks", "classifications", "volumes", zero.EVENTS, zero.SIMS])
def test_readback_joins_full_int64_ids(name):
    tables, readback = fixture(BIG + 1)
    readback[name]["run"][0] += 1
    with pytest.raises(ValueError):
        check(tables, readback, BIG + 1)


@pytest.mark.parametrize("name,field", [("events", "translation"), ("events", "rotation"), ("events", "halfSize"),
                                         ("volumes", "translation"), ("volumes", "rotation")])
@pytest.mark.parametrize("mode", ["mixed", "wrong", "nonfinite", "integer"])
def test_metadata_strict_independent_of_callback_fixture(name, field, mode):
    tables, readback = fixture()
    a = tables[name][field]
    if mode == "mixed":
        counts = np.full(len(a), a.shape[1], dtype=np.int64)
        counts[0] = 0
        tables[name][field] = ak.unflatten(a[1:].ravel(), counts)
    elif mode == "integer":
        tables[name][field] = a.astype(np.int64)
    else:
        a[0, 0] = np.nan if mode == "nonfinite" else 3.
    with pytest.raises(ValueError, match=field):
        check(tables, readback)


@pytest.mark.parametrize("mode", ["missing-table", "missing-field", "extra-field", "short", "matrix", "null-string", "readback-callbacks", "extra-step"])
def test_malformed_schema(mode):
    tables, readback = fixture()
    if mode == "missing-table":
        del tables[zero.CALLBACKS]
    elif mode == "missing-field":
        del tables[zero.SIMS]["simStage"]
    elif mode == "extra-field":
        tables[zero.SIMS]["unknown"] = np.zeros(2)
    elif mode == "short":
        tables[zero.CALLBACKS]["track"] = np.ones(5, dtype=np.int64)
    elif mode == "matrix":
        tables[zero.EVENTS]["event"] = np.ones((2, 1), dtype=np.int64)
    elif mode == "null-string":
        tables[zero.SIMS]["creationCode"][0] = None
    elif mode == "readback-callbacks":
        readback[zero.CALLBACKS] = tables[zero.CALLBACKS]
    else:
        tables["steps"] = empty("steps", 1)
        tables["steps"]["run"][:] = 15
        tables["steps"]["event"][:] = 1
        tables["events"]["nSteps"][0] = 1
    with pytest.raises(ValueError):
        check(tables, readback)


@pytest.mark.parametrize("name,field", [("tracks", "entrance_x"), ("tracks", "entrance_weight"),
    (zero.CALLBACKS, "pre_x"), (zero.CALLBACKS, "post_energy"), (zero.CALLBACKS, "infoKE"),
    (zero.CALLBACKS, "infoTime"), (zero.CALLBACKS, "trackLength"), (zero.SIMS, "trackLength")])
def test_defaults_and_zero_lengths_are_exact(name, field):
    tables, readback = fixture()
    tables[name][field][0] += 1e-12
    with pytest.raises(ValueError):
        check(tables, readback)


def test_ordinary_audit_must_reject(monkeypatch):
    monkeypatch.setattr(capture, "audit", lambda *args: {})
    with pytest.raises(ValueError, match="unexpectedly accepted"):
        check()


def test_report_does_not_modify_data():
    tables, readback = fixture()
    before = copy.deepcopy(tables)
    check(tables, readback)
    for name, table in tables.items():
        for field, value in table.items():
            np.testing.assert_array_equal(value, before[name][field])


def write_root(path, tables):
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree(zero.PATHS.get(name, "dump/" + name), table)


def test_root_highest_independent_cycles_and_cli_no_overwrite_on_failure(tmp_path, capsys):
    original, readback_path, output = tmp_path / "original.root", tmp_path / "readback.root", tmp_path / "report.json"
    tables, readback = fixture(BIG + 1)
    stale = copy.deepcopy(tables)
    stale[zero.CALLBACKS]["statusIn"][0] = 2
    write_root(original, stale)
    write_root(readback_path, readback)
    with uproot.update(original) as root:
        root.mktree(zero.PATHS[zero.CALLBACKS], tables[zero.CALLBACKS])
    with uproot.update(readback_path) as root:
        # Independent tree cycles and row order must not affect identity/value comparison.
        root.mktree(zero.PATHS[zero.SIMS], {field: value[::-1] for field, value in readback[zero.SIMS].items()})
    args = [str(original), "--readback", str(readback_path), "--run", str(BIG + 1), "--output", str(output)]
    zero.main(args)
    previous = output.read_bytes()
    result = json.loads(previous)
    assert json.loads(capsys.readouterr().out) == result
    assert result["engine_lifecycle_passed"] and not result["endpoint_kinematics_match"]
    assert result["endpoint_rows"][0]["run"] == BIG + 1
    readback[zero.SIMS]["endKE"][0] = 1.
    with uproot.update(readback_path) as root:
        root.mktree(zero.PATHS[zero.SIMS], readback[zero.SIMS])
    with pytest.raises(ValueError, match="readback.*endKE"):
        zero.main(args)
    assert output.read_bytes() == previous
    output.unlink()
    with pytest.raises(ValueError):
        zero.main(args)
    assert not output.exists() and capsys.readouterr().out == ""


@pytest.mark.parametrize("endpoint", [0., 3.25, 10.])
@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_optional_endpoint_gate_preserves_report(tmp_path, capsys, endpoint, existing_output):
    original, readback_path = tmp_path / "original.root", tmp_path / "readback.root"
    baseline = tmp_path / "baseline.json"
    tables, readback = fixture(endpoint=endpoint)
    write_root(original, tables)
    write_root(readback_path, readback)
    args = [str(original), "--readback", str(readback_path), "--run", "15", "--output"]
    # The default CLI must still record all measured endpoints, including discrepancies.
    zero.main([*args, str(baseline)])
    previous = baseline.read_bytes()
    expected = zero.report(zero.read(original), zero.read(readback_path, readback=True), 15)
    expected.update(input=str(original), readback=str(readback_path))
    assert json.loads(previous) == expected == json.loads(capsys.readouterr().out)
    assert expected["endpoint_kinematics_match"] is (endpoint == 10.)
    output = baseline if existing_output else tmp_path / "new" / "report.json"
    gated_args = [*args, str(output), "--require-endpoint-match"]
    if endpoint == 10.:
        zero.main(gated_args)
        assert output.read_bytes() == previous
        assert json.loads(capsys.readouterr().out) == expected
    else:
        with pytest.raises(ValueError, match="finalized endpoint kinematics do not match captured track"):
            zero.main(gated_args)
        if existing_output:
            assert output.read_bytes() == previous
        else:
            assert not output.exists() and not output.parent.exists()
        assert capsys.readouterr().out == ""
    assert baseline.read_bytes() == previous


@pytest.mark.parametrize("require_endpoint_match", [False, True])
@pytest.mark.parametrize("failure", ["readback", "lifecycle"])
def test_cli_validation_precedes_endpoint_gate(tmp_path, capsys, require_endpoint_match, failure):
    original, readback_path = tmp_path / "original.root", tmp_path / "readback.root"
    output = tmp_path / "new" / "report.json"
    tables, readback = fixture()  # The endpoint also fails the optional acceptance gate.
    if failure == "readback":
        readback[zero.SIMS]["endKE"][:] = 1.
        error = "readback.*endKE mismatch"
    else:
        tables[zero.CALLBACKS]["forwarded"][:] = False
        error = "zeroStepCallbacks: forwarded mismatch"
    write_root(original, tables)
    write_root(readback_path, readback)
    args = [str(original), "--readback", str(readback_path), "--run", "15", "--output", str(output)]
    if require_endpoint_match:
        args.append("--require-endpoint-match")
    with pytest.raises(ValueError, match=error):
        zero.main(args)
    assert not output.exists() and not output.parent.exists()
    assert capsys.readouterr().out == ""


def test_root_mixed_highest_metadata_is_rejected_without_report(tmp_path):
    original, readback_path, output = tmp_path / "original.root", tmp_path / "readback.root", tmp_path / "report.json"
    tables, readback = fixture()
    write_root(original, tables)
    write_root(readback_path, readback)
    tables["events"]["translation"] = ak.unflatten(np.zeros(3), np.array([3, 0]))
    with uproot.update(original) as root:
        root.mktree("dump/events", tables["events"])
    loaded = zero.read(original)
    assert isinstance(loaded["events"]["translation"], ak.Array)
    with pytest.raises(ValueError, match="translation"):
        zero.main([str(original), "--readback", str(readback_path), "--run", "15", "--output", str(output)])
    assert not output.exists()


def test_root_readback_cannot_claim_callbacks(tmp_path):
    path = tmp_path / "readback.root"
    tables, _ = fixture()
    write_root(path, tables)
    with pytest.raises(ValueError, match="must not contain"):
        zero.read(path, readback=True)


@pytest.mark.parametrize("omitted", ["--readback", "--run", "--output"])
def test_cli_required_arguments(omitted, tmp_path):
    options = {"--readback": "readback.root", "--run": "15", "--output": str(tmp_path / "report.json")}
    args = ["original.root"] + [value for key, val in options.items() if key != omitted for value in (key, val)]
    with pytest.raises(SystemExit):
        zero.main(args)
    assert not (tmp_path / "report.json").exists()
