"""Synthetic complete-capture regressions; these are not engine/runtime evidence."""
import copy
import json

import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_secondary_zero_step as diag
from test_capture import empty, copy_state

BIG = 2**53 + 1


def fixture(mode="classification", count=3):
    """Complete baseline with nine primary steps and independently transported children."""
    n = count + 1
    base = {name: empty(name, size) for name, size in
            (("events", 1), ("tracks", n), ("steps", 9 + count), ("classifications", n), ("volumes", 3))}
    for name, table in base.items():
        table["run"][:] = 10
        table["event"][:] = 1
        if "index" in table:
            table["index"][:] = np.arange(len(table["index"]))
    e, t, s, c, v = (base[name] for name in capture.TABLES)
    for field in "complete recordComplete capturePresent statusPresent eventIDsMatch statusClean".split():
        e[field][:] = True
    for field in capture.EVENT:
        e["record" + field.capitalize()][:] = e[field]
    for field, value in dict(schema=1, geometryTag="saved-stopping-box", physicsTag="saved-stopping-physics",
                             envelope="CaptureEnvelope", nTracks=n, nSteps=9+count, nClassifications=n,
                             nVolumes=3, nG4Tracks=n, cpuTime=.1, realTime=.3).items():
        e[field][:] = value
    e["halfSize"][:] = 50.
    e["rotation"][:] = np.eye(3).ravel()
    v["id"][:] = v["copy"][:] = v["index"]
    v["name"][:] = v["logical"][:] = ["World", "BoxInTheWorld", "CaptureEnvelope"]
    v["motherLogical"][:] = ["", "World", "BoxInTheWorld"]
    v["material"][:] = ["G4_Galactic", "G4_Galactic", "G4_POLYSTYRENE"]
    v["solid"][:] = "G4Box"
    v["density"][:] = [1e-25, 1e-25, 1.06]
    v["rotation"][:] = np.eye(3).ravel()
    t["track"][:] = np.r_[1, np.arange(BIG, BIG + count, dtype=np.int64)]
    t["parent"][1:] = 1
    t["pdg"][:] = np.r_[-13, np.resize([11, 12, -14], count)]
    t["steps"][:] = np.r_[9, np.ones(count, dtype=np.int64)]
    t["secondaries"][0] = count
    t["entered"][:] = t["ended"][:] = True
    t["status"][:] = 2
    t["reason"][:] = 1
    t["creator"][1:] = "Decay"
    t["stopProcess"][:] = "Decay"
    s["track"][:] = np.r_[np.ones(9, dtype=np.int64), t["track"][1:]]
    s["number"][:] = np.r_[np.arange(1, 10), np.ones(count, dtype=np.int64)]
    s["secondaries"][8] = count
    s["length"][:] = np.r_[.001, np.full(8, .5), np.full(count, .25)]
    s["trackStatus"][:] = np.r_[np.zeros(7, dtype=np.int32), 1, np.full(1+count, 2)]
    s["preInside"][:] = True
    s["preInside"][0] = False
    s["postInside"][:] = True
    s["pre_z"][:] = np.r_[-50.001, -50 + np.arange(8)*.5, np.zeros(count)]
    s["post_z"][:] = np.r_[-50 + np.arange(9)*.5, np.full(count, .25)]
    s["pre_time"][:] = np.r_[np.arange(9), np.full(count, 1000.123456789)]
    s["post_time"][:] = np.r_[np.arange(1, 10), np.full(count, 1000.623456789)]
    for prefix in ("pre_", "post_"):
        s[prefix + "lz"][:] = s[prefix + "z"]
        s[prefix + "pz"][:] = s[prefix + "lpz"][:] = 1.23456789
        s[prefix + "energy"][:] = 4.123456789
        s[prefix + "kinetic"][:] = 3.123456789
        s[prefix + "mass"][:] = 1.
        s[prefix + "properTime"][:] = s[prefix + "time"] * .33333
        s[prefix + "weight"][:] = 1.
        s[prefix + "volume"][:] = 2
        s[prefix + "material"][:] = "G4_POLYSTYRENE"
        s[prefix + "placement"][:] = "/0:0/1:1/2:2"
        s[prefix + "stepStatus"][:] = 4
    s["pre_volume"][0] = 1
    s["pre_material"][0] = "G4_Galactic"
    s["pre_stepStatus"][0] = 7
    s["pre_stepStatus"][1] = s["post_stepStatus"][0] = 1
    s["post_stepStatus"][8] = 2
    s["post_process"][:] = "Transportation"
    s["post_process"][8:] = "Decay"
    copy_state(t, "begin_", s, "pre_", np.r_[0, np.arange(9, 9+count)])
    copy_state(t, "end_", s, "post_", np.r_[8, np.arange(9, 9+count)])
    copy_state(t, "entrance_", t, "begin_", slice(None))
    for field in capture.STATE:
        t["entrance_" + field][0] = s["post_" + field][0]
    t["begin_stepStatus"][:] = t["end_stepStatus"][:] = -1
    t["entrance_stepStatus"][1:] = -1
    t["begin_process"][1:] = t["entrance_process"][1:] = "Decay"
    for field in ("track", "parent", "pdg"):
        c[field][:] = t[field]
    copy_state(c, "state_", t, "begin_", slice(None))
    capture.audit(base, 1)
    candidate = copy.deepcopy(base)
    e, t, s, c = (candidate[name] for name in capture.TABLES[:4])
    e["geometryTag"][:] = diag.GEOMETRY
    e["physicsTag"][:] = diag.PHYSICS
    e["nSteps"][:] = 9
    t["steps"][1:] = 0
    t["stopProcess"][1:] = ""
    for field in capture.STATE:
        t["end_" + field][1:] = t["begin_" + field][1:]
    t["end_process"][1:] = ""
    c["status"][1:] = 2 if mode == "classification" else 0
    candidate["steps"] = {f: value[:9].copy() for f, value in s.items()}
    s = candidate["steps"]
    capture.audit(candidate, 1)
    for name, size in ((diag.CALLBACKS, 12+3*count), (diag.EVENTS, 1), (diag.SIMS, n)):
        table = {}
        for kind, fields in diag.SCHEMAS[name].items():
            dtype = {"int64": np.int64, "int": np.int32, "bool": np.bool_, "float": np.float64, "str": object}[kind]
            for field in fields:
                table[field] = np.full(size, "" if kind == "str" else 0, dtype=dtype)
        table["run"][:] = 10
        table["event"][:] = 1
        if "index" in table:
            table["index"][:] = np.arange(size)
        candidate[name] = table
    b, f, sims = (candidate[name] for name in (diag.CALLBACKS, diag.EVENTS, diag.SIMS))
    tr = np.r_[np.zeros(12, dtype=np.int64), np.arange(1, n), np.repeat(np.arange(1, n)[::-1], 2)]
    b["phase"][:] = np.r_[0, 1, np.full(9, 3), 2, np.zeros(count), np.tile([1, 2], count)]
    p, child = b["phase"], tr != 0
    for field in ("track", "parent", "pdg", "creator"):
        b[field][:] = t[field][tr]
    b["mode"][:] = mode
    b["creatorPresent"][:] = child
    b["parentEnded"][:] = child
    b["parentEndIndex"][:] = np.where(child, 11, -1)
    b["g4Version"][:] = 1132
    b["diagnosticOnly"][:] = b["forwarded"][:] = True
    b["killedValid"][:] = p == 0
    b["cutEvaluated"][:] = p == 3
    b["weight"][:] = 1.
    b["dynamicMass"][:] = t["begin_mass"][tr]
    b["steps"][p == 3] = s["number"]
    b["steps"][11] = 9
    b["trackLength"][p == 3] = np.cumsum(s["length"])
    b["trackLength"][11] = b["trackLength"][10]
    b["stepSecondaries"][p == 3] = s["secondaries"]
    b["statusIn"][:] = np.where(p == 2, 2, 0)
    b["statusIn"][child & (p == 1)] = 2 if mode == "classification" else 0
    b["statusIn"][p == 3] = s["trackStatus"]
    b["statusForward"][:] = b["statusIn"]
    if mode == "classification":
        b["statusForward"][child & (p == 0)] = 2
    b["statusOut"][:] = b["statusForward"]
    b["statusOut"][child & (p == 1)] = 2
    b["mutated"][:] = child & (p == (0 if mode == "classification" else 1))
    info = p > 0
    b["userInfoPresent"][:] = b["mu2eInfoPresent"][:] = info
    b["infoCode"][:] = np.where(info, "uninitialized", "")
    b["infoTime"][:] = b["infoProperTime"][:] = np.where(info, -1., 0.)
    for field in diag.STATE:
        b["track_" + field][:] = t["begin_" + field][tr]
        b["track_" + field][p == 2] = t["end_" + field][tr[p == 2]]
        b["track_" + field][p == 3] = s["post_" + field]
    b["stepPresent"][:] = (p == 2) | (p == 3)
    for prefix in ("pre_", "post_"):
        b[prefix + "present"][:] = b["stepPresent"]
        b[prefix + "status"][:] = np.where(b["stepPresent"], 7, -1)
        b[prefix + "status"][p == 3] = s[prefix + "stepStatus"]
        b[prefix + "process"][p == 3] = s[prefix + "process"]
        b[prefix + "processPresent"][:] = b[prefix + "process"] != ""
        for field in diag.STATE:
            b[prefix + field][p == 2] = b["track_" + field][p == 2]
            b[prefix + field][p == 3] = s[prefix + field]
    for prefix in ("pre_", "post_"):
        b[prefix + "status"][11] = s[prefix + "stepStatus"][-1]
        b[prefix + "process"][11] = s[prefix + "process"][-1]
        b[prefix + "processPresent"][11] = s[prefix + "process"][-1] != ""
        for field in diag.STATE:
            b[prefix + field][11] = s[prefix + field][-1]
    for field in "diagnosticOnly capturePresent simsPresent statusPresent".split():
        f[field][:] = True
    f["nSims"][:] = n
    for field in ("track", "parent", "pdg"):
        sims[field][:] = t[field]
    sims["key"][:] = t["track"]
    sims["nDaughters"][:] = t["secondaries"]
    sims["nSteps"][:] = t["steps"]
    sims["startStatus"][1:] = 2 if mode == "classification" else 0
    sims["endStatus"][:] = 2
    sims["endDefined"][:] = True
    sims["isPrimary"][0] = sims["genPresent"][0] = True
    sims["parentPresent"][1:] = True
    sims["creationCode"][:] = np.where(t["parent"] == 0, "mu2ePrimary", t["creator"])
    sims["stoppingCode"][:] = np.where(t["parent"] == 0, "Decay", "NotSpecified")
    sims["trackLength"][0] = np.float32(np.sum(s["length"]))
    for prefix, source in (("start_", "begin_"), ("end_", "end_")):
        for field in diag.SIM_STATE:
            sims[prefix + field][:] = diag._stored(t[source + field], field)
    sims["endKE"][:] = diag._stored(t["end_kinetic"], "kinetic")
    # Deliberately unrelated legacy primary endpoint: must not impose a broad raw-end comparison.
    sims["end_pz"][0] = 999.
    t["index"][1:] = np.arange(1, n)[::-1]
    return candidate, copy.deepcopy({name: table for name, table in candidate.items() if name != diag.CALLBACKS}), base


def check(data, mode="classification"):
    return diag.report(*data, mode)


@pytest.mark.parametrize("mode", ["classification", "begin"])
@pytest.mark.parametrize("count", [1, 3, 5])
def test_complete_positive(mode, count):
    data = fixture(mode, count)
    result = check(data, mode)
    assert result["engine_lifecycle_passed"] and result["endpoint_roundtrip_match"]
    assert result["primary_baseline_identical"] and result["readback_identical"]
    assert result["diagnostic_only"] and result["real_engine"]
    assert not result["bank_admissible"] and not result["physical_response"]
    assert result["direct_child_count"] == count
    assert result["endpoint_rows"][0]["track"] == BIG
    assert result["endpoint_rows"][0]["captured_time"] != result["endpoint_rows"][0]["sim_time"]
    assert result["endpoint_rows"][0]["sim_time"] == float(np.float32(1000.123456789))
    assert not any(result["endpoint_field_errors"].values())
    assert diag.inspect_baseline(data[2])["direct_child_count"] == count
    json.dumps(result, allow_nan=False)


def test_independent_row_order_and_int64_identity():
    data = fixture()
    shuffled = tuple({n: {f: v[::-1].copy() for f, v in t.items()} for n, t in tables.items()} for tables in data)
    assert check(shuffled)["readback_identical"]
    data[0][diag.CALLBACKS]["track"][12] = BIG + 1
    with pytest.raises(ValueError):
        check(data)


@pytest.mark.parametrize("name", [*capture.TABLES, diag.CALLBACKS, diag.EVENTS, diag.SIMS])
@pytest.mark.parametrize("kind", ["missing", "extra", "float-id", "wrong-event", "duplicate"])
def test_strict_schema_and_census(name, kind):
    data = fixture()
    table = data[0][name]
    if kind == "missing":
        del table["event"]
    elif kind == "extra":
        table["unexpected"] = table["event"].copy()
    elif kind == "float-id":
        table["event"] = table["event"].astype(float)
    elif kind == "wrong-event":
        table["event"][0] = 2
    else:
        data[0][name] = {f: np.concatenate([v, v[:1]], axis=0) for f, v in table.items()}
    with pytest.raises((ValueError, KeyError)):
        check(data)


@pytest.mark.parametrize("field", [f for fields in diag.SCHEMAS[diag.CALLBACKS].values() for f in fields])
def test_callback_field_corruption(field):
    data = fixture()
    table = data[0][diag.CALLBACKS]
    # Child end covers all counters, statuses, defaults, points and state branches.
    row = len(table[field]) - 1
    value = table[field]
    if value.dtype.kind == "b":
        value[row] = not value[row]
    elif value.dtype.kind in "OU":
        value[row] = "wrong"
    else:
        value[row] += 1
    with pytest.raises(ValueError):
        check(data)


@pytest.mark.parametrize("field", [*diag.STATE, "trackLength"])
def test_finite_endpoint_disagreements_are_reported_not_rejected(field):
    data = fixture()
    leaf = "trackLength" if field == "trackLength" else "endKE" if field == "kinetic" else "end_" + field
    for tables in data[:2]:
        tables[diag.SIMS][leaf][1] += .5
    if field == "trackLength":
        with pytest.raises(ValueError, match="trackLength"):
            check(data)
    else:
        result = check(data)
        assert not result["endpoint_roundtrip_match"]
        assert result["endpoint_field_errors"][field] == 1


@pytest.mark.parametrize("field", diag.SIM_STATE)
def test_start_float_storage_is_exact(field):
    data = fixture()
    for tables in data[:2]:
        tables[diag.SIMS]["start_" + field][1] = np.nextafter(tables[diag.SIMS]["start_" + field][1], np.inf)
    with pytest.raises(ValueError, match="SimParticle start"):
        check(data)


@pytest.mark.parametrize("name", [*capture.TABLES, diag.EVENTS, diag.SIMS])
def test_every_persisted_field_compared(name):
    # Fixed-schema loop: independently corrupt every persisted scalar/metadata field.
    original = fixture()
    for field, value in original[1][name].items():
        data = copy.deepcopy(original)
        target = data[1][name][field]
        if value.dtype.kind == "b":
            target[0] = ~target[0]
        elif value.dtype.kind in "OU":
            target[0] = "wrong"
        else:
            target[0] += 1
        with pytest.raises(ValueError):
            check(data)


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("name", capture.TABLES)
def test_nonfinite_capture(target, name):
    data = fixture()
    field = next(f for f, v in data[target][name].items() if v.dtype.kind == "f")
    data[target][name][field][0] = np.nan
    with pytest.raises(ValueError):
        check(data)


@pytest.mark.parametrize("name", [diag.CALLBACKS, diag.SIMS])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_diagnostic_float_fields_nonfinite(name, bad):
    original = fixture()
    for field in diag.SCHEMAS[name]["float"]:
        data = copy.deepcopy(original)
        data[0][name][field][-1] = bad
        with pytest.raises(ValueError, match="nonfinite"):
            check(data)


@pytest.mark.parametrize("field", ["creator", "pdg", "begin_time", "entrance_properTime", "begin_x"])
def test_child_birth_changes_fail(field):
    data = fixture()
    target = data[0]["tracks"][field]
    target[1] = "wrong" if target.dtype.kind in "OU" else target[1] + 1
    with pytest.raises(ValueError):
        check(data)


def test_baseline_primary_all_fields_and_all_steps():
    original = fixture()
    for name, mask in (("tracks", original[2]["tracks"]["parent"] == 0),
                       ("steps", original[2]["steps"]["track"] == 1)):
        for field, value in original[2][name].items():
            data = copy.deepcopy(original)
            target = data[2][name][field]
            target[mask] = ~target[mask] if value.dtype.kind == "b" else "wrong" if value.dtype.kind in "OU" else target[mask] + 1
            with pytest.raises(ValueError):
                check(data)


@pytest.mark.parametrize("mode", ["", "both", None, 1])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode"):
        check(fixture(), mode)


def test_mode_mismatch_and_baseline_not_skipped():
    with pytest.raises(ValueError):
        check(fixture("begin"))
    data = fixture()
    data[2]["events"]["complete"][:] = False
    with pytest.raises(ValueError):
        diag.inspect_baseline(data[2])
    with pytest.raises(ValueError):
        check(data)


def write_root(path, tables, *, cycles=False):
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            key = "dump/" + name if name in capture.TABLES else diag.PATHS[name]
            if cycles:
                stale = {f: v[:0] for f, v in table.items()}
                root.mktree(key, stale)
            root.mktree(key, table)


def files(tmp_path, data):
    candidate, readback, baseline = (tmp_path / name for name in ("candidate.root", "readback.root", "baseline.root"))
    for path, tables in zip((candidate, readback, baseline), data):
        write_root(path, tables, cycles=True)
    return candidate, readback, baseline


def args(paths, output):
    candidate, readback, baseline = paths
    return [str(candidate), "--readback", str(readback), "--baseline", str(baseline), "--mode", "classification", "--output", str(output)]


def test_cli_highest_cycles_inspect_and_gated_reports(tmp_path, capsys):
    data = fixture()
    for tables in data[:2]:
        tables[diag.SIMS]["end_time"][1] = 0.
    paths = files(tmp_path, data)
    ordinary, gate, inspect = (tmp_path / name for name in ("ordinary.json", "gate.json", "inspect.json"))
    diag.main(args(paths, ordinary))
    assert not json.loads(ordinary.read_text())["endpoint_roundtrip_match"]
    with pytest.raises(ValueError, match="round-trip"):
        diag.main([*args(paths, gate), "--require-endpoint-match"])
    assert not gate.exists()
    diag.main(["--inspect-baseline", "--baseline", str(paths[2]), "--output", str(inspect)])
    result = json.loads(inspect.read_text())
    assert result["direct_child_count"] == 3 and len(result["primary_step_rows"]) == 9
    assert result["direct_child_birth_rows"][0]["track"] == BIG
    assert not result["bank_admissible"]


def test_cli_noclobber_aliases_and_success_gate(tmp_path, capsys):
    paths = files(tmp_path, fixture())
    success = tmp_path / "success.json"
    diag.main([*args(paths, success), "--require-endpoint-match"])
    saved = success.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        diag.main(args(paths, success))
    assert success.read_bytes() == saved
    for path in paths:
        with pytest.raises(ValueError, match="overwrite"):
            diag.main(args(paths, path))
    alias = tmp_path / "alias.root"
    alias.symlink_to(paths[2])
    with pytest.raises(ValueError, match="overwrite"):
        diag.main(args(paths, alias))
    hard = tmp_path / "hard.root"
    hard.hardlink_to(paths[0])
    with pytest.raises(ValueError, match="overwrite"):
        diag.main(args(paths, hard))


def test_readback_rejects_callback_tree(tmp_path):
    paths = files(tmp_path, fixture())
    with pytest.raises(ValueError, match="callback"):
        diag.read(paths[0], readback=True)


def test_primary_information_is_not_forced_to_child_defaults():
    data = fixture()
    b = data[0][diag.CALLBACKS]
    primary = (b["parent"] == 0) & b["mu2eInfoPresent"]
    b["infoKE"][primary] = 17.25
    b["infoCode"][primary] = "Decay"
    b["infoTime"][primary] = 1000.25
    assert check(data)["engine_lifecycle_passed"]


def test_child_earlier_step_presence_is_measured_not_assumed():
    data = fixture()
    b = data[0][diag.CALLBACKS]
    early = (b["parent"] == 1) & (b["phase"] < 2)
    b["stepPresent"][early] = True
    for prefix in ("pre_", "post_"):
        b[prefix + "present"][early] = True
        b[prefix + "status"][early] = 7
        for field in diag.STATE:
            b[prefix + field][early] = b["track_" + field][early]
    assert check(data)["engine_lifecycle_passed"]


@pytest.mark.parametrize("name", [*capture.TABLES, diag.CALLBACKS, diag.EVENTS, diag.SIMS])
def test_missing_rows_are_rejected(name):
    data = fixture()
    data[0][name] = {f: v[:-1] for f, v in data[0][name].items()}
    with pytest.raises(ValueError):
        check(data)


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("field", ["event", "complete", "cpuTime"])
def test_masked_capture_values_rejected(target, field):
    data = fixture()
    table = data[target]["events"]
    table[field] = np.ma.array(table[field], mask=True)
    with pytest.raises(ValueError, match="masked"):
        check(data)


@pytest.mark.parametrize("name,field", [(diag.CALLBACKS, "track"), (diag.CALLBACKS, "forwarded"),
                                        (diag.SIMS, "endKE"), ("volumes", "translation")])
def test_masked_diagnostic_or_metadata_rejected(name, field):
    data = fixture()
    data[0][name][field] = np.ma.array(data[0][name][field], mask=True)
    with pytest.raises(ValueError, match="masked"):
        check(data)


@pytest.mark.parametrize("field", [*diag.SIM_STATE, "stoppingCode", "trackLength"])
def test_primary_start_and_nonkinematic_finalization_corruption(field):
    data = fixture()
    leaf = field if field in ("stoppingCode", "trackLength") else "start_" + field
    for tables in data[:2]:
        tables[diag.SIMS][leaf][0] = "NotSpecified" if field == "stoppingCode" else -1.
    with pytest.raises(ValueError):
        check(data)


@pytest.mark.parametrize("field", ["infoKE", "infoDx", "infoDy", "infoDz", "infoTime", "infoProperTime", "infoCode", "forced"])
def test_absent_primary_info_payload_rejected(field):
    data = fixture()
    data[0][diag.CALLBACKS][field][0] = "Decay" if field == "infoCode" else 1
    with pytest.raises(ValueError):
        check(data)


def test_secondary_reduction_wraparound_rejected():
    data = fixture()
    for tables in data:
        tables["steps"]["secondaries"][:4] += np.int64(2**62)
    b = data[0][diag.CALLBACKS]
    rows = np.flatnonzero(b["phase"] == 3)[:4]
    b["stepSecondaries"][rows] += np.int64(2**62)
    with pytest.raises(ValueError, match="impossible secondary count"):
        check(data)
