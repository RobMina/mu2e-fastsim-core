"""Synthetic helper-fixture tests; no Geant4 build/run or physical-transport claim."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import uproot

from mu2efastsim import capture_end_kinematics as check


def fixture():
    """Independent synthetic matrix; sqrt here only creates self-consistent test data."""
    t = {
        "run": np.full(16, 16, dtype=np.int64), "subrun": np.zeros(16, dtype=np.int64),
        "event": np.ones(16, dtype=np.int64), "caseIndex": np.arange(1, 17, dtype=np.int32),
        "caseName": np.array([
            "zero_gamma_default_step", "zero_gamma_absent_no_step", "zero_gamma_stale_no_step",
            "zero_electron_absent_step", "zero_electron_default_no_step", "zero_electron_stale_step",
            "zero_electron_at_rest_step", "zero_electron_dynamic_mass", "one_gamma_raw_zero_recorded_7",
            "one_electron_raw_zero_recorded_7", "multi_gamma_raw_3_recorded_7",
            "multi_electron_raw_3_recorded_7", "one_gamma_default_info", "one_electron_default_info",
            "multi_electron_recorded_zero", "multi_electron_dynamic_mass"], dtype=object),
        "infoKind": np.array(["default", "absent", "recorded", "absent", "default", "recorded", "default",
                              "recorded", "recorded", "recorded", "recorded", "recorded", "default", "default",
                              "recorded", "recorded"], dtype=object),
        "pdg": np.array([22, 22, 22, 11, 11, 11, 11, 11, 22, 11, 22, 11, 22, 11, 11, 11], dtype=np.int32),
        "nSteps": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 4, 1, 1, 2, 2], dtype=np.int32),
        "stepPresent": np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
        "trackLength": np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 5., 0., 0., 2., 2.]),
        "rawKE": np.array([10., 10., 10., 10., 10., 10., 0., 10., 0., 0., 3., 3., 5., 5., 5., 3.]),
        "infoKE": np.array([0., np.nan, 9., np.nan, 0., 9., 0., 9., 7., 7., 7., 7., 0., 0., 0., 7.]),
        "error": np.full(16, "", dtype=object),
    }
    t["infoPresent"] = t["infoKind"] != "absent"
    for field in ("diagnosticOnly", "helperKEPresent", "helperMomentumPresent", "pass"):
        t[field] = np.ones(16, dtype=bool)
    for field in ("bankAdmissible", "realEngine"):
        t[field] = np.zeros(16, dtype=bool)
    for axis, raw, recorded in (("X", 0., 0.6), ("Y", 0., 0.8), ("Z", 1., 0.)):
        t["rawDir" + axis] = np.full(16, raw)
        t["infoDir" + axis] = np.where(t["infoPresent"], np.where(t["infoKind"] == "recorded", recorded, 0.), np.nan)
    t["definitionMass"] = np.where(t["pdg"] == 11, 0.51099895, 0.)
    t["dynamicMass"] = np.where(np.isin(t["caseIndex"], [8, 16]), 2.5, t["definitionMass"])
    t["rawE"] = t["rawKE"] + t["dynamicMass"]
    t["rawPx"] = np.zeros(16)
    t["rawPy"] = np.zeros(16)
    t["rawPz"] = np.sqrt(t["rawKE"] * (t["rawKE"] + 2.*t["dynamicMass"]))
    zero = t["nSteps"] == 0
    t["expectedKE"] = np.where(zero, t["rawKE"], t["infoKE"])
    t["expectedE"] = np.where(zero, t["rawE"], t["infoKE"] + t["definitionMass"])
    p = np.sqrt((t["infoKE"] + t["definitionMass"])**2 - t["definitionMass"]**2)
    for component, axis in (("Px", "X"), ("Py", "Y"), ("Pz", "Z")):
        t["expected" + component] = np.where(zero, t["raw" + component], p*t["infoDir" + axis])
    for field in ("KE", "Px", "Py", "Pz", "E"):
        t["actual" + field] = t["expected" + field].copy()
    return t


def write_tree(root, table):
    root.mktree(check.TREE, {field: "string" if value.dtype.kind in "OU" else value.dtype
                           for field, value in table.items()})
    root[check.TREE].extend(table)


def root_file(path, table=None):
    with uproot.recreate(path) as root:
        write_tree(root, fixture() if table is None else table)
    return path


def test_passing_matrix_and_root(tmp_path):
    expected = dict(diagnostic_only=True, bank_admissible=False, real_engine=False, cases=16,
                    zero_step=8, positive_step=8, exact_helper_match=True, run=16, subrun=0, event=1)
    assert check.report(fixture()) == expected
    assert check.report(check.read(root_file(tmp_path / "fixture.root"))) == expected


@pytest.mark.parametrize("field", [field for fields in check.SCHEMA.values() for field in fields])
def test_every_field_required(field):
    t = fixture()
    del t[field]
    with pytest.raises(ValueError, match="fields"):
        check.report(t)


def test_extra_field():
    t = fixture()
    t["unexpected"] = np.zeros(16)
    with pytest.raises(ValueError, match="fields"):
        check.report(t)


@pytest.mark.parametrize("field,dtype", [
    ("run", "int32"), ("subrun", "float64"), ("event", "uint64"),
    ("caseIndex", "int64"), ("pdg", "float64"), ("nSteps", "uint32"),
    ("pass", "int8"), ("actualKE", "float32"), ("expectedE", "int64"),
    ("caseName", "S64"),
])
def test_dtype(field, dtype):
    t = fixture()
    t[field] = t[field].astype(dtype)
    with pytest.raises(ValueError, match=field):
        check.report(t)


@pytest.mark.parametrize("bad", [None, 3, b"bad", np.nan])
def test_object_strings_are_really_strings(bad):
    t = fixture()
    t["error"][0] = bad
    with pytest.raises(ValueError, match="strings"):
        check.report(t)


@pytest.mark.parametrize("shape", [(), (16, 1), (15,), (17,)])
def test_malformed_field(shape):
    t = fixture()
    t["event"] = np.ones(shape, dtype=np.int64)
    with pytest.raises(ValueError, match="16 scalar rows"):
        check.report(t)


def test_masked_field():
    t = fixture()
    t["actualKE"] = np.ma.array(t["actualKE"], mask=False)
    with pytest.raises(ValueError, match="masked"):
        check.report(t)


@pytest.mark.parametrize("size", [0, 1, 8, 15, 17, 32])
def test_incomplete_extra_census(size):
    t = {field: np.resize(value, size) for field, value in fixture().items()}
    with pytest.raises(ValueError, match="16 scalar rows"):
        check.report(t)


@pytest.mark.parametrize("order", [np.arange(15, -1, -1), np.r_[np.arange(15), 14]])
def test_reordered_duplicate_cases(order):
    with pytest.raises(ValueError, match="caseIndex"):
        check.report({field: value[order] for field, value in fixture().items()})


@pytest.mark.parametrize("field,index,value", [
    ("run", 0, 17), ("run", 0, 2**53 + 16), ("subrun", 0, 1), ("event", 0, 2),
    ("caseIndex", 15, 15), ("caseName", 0, "wrong"), ("pdg", 0, 11), ("nSteps", 8, 0),
    ("nSteps", 11, 3), ("trackLength", 10, 0.), ("rawKE", 0, 9.),
    ("infoKind", 0, "recorded"), ("infoPresent", 1, True), ("stepPresent", 2, True),
    ("diagnosticOnly", 0, False), ("bankAdmissible", 0, True), ("realEngine", 0, True),
    ("helperKEPresent", 0, False), ("helperMomentumPresent", 0, False), ("pass", 0, False),
    ("error", 0, "failed"), ("infoKE", 2, 0.), ("rawDirX", 0, 0.1), ("rawDirZ", 0, 0.),
    ("infoDirX", 2, 0.), ("infoDirY", 0, 0.8), ("infoDirZ", 2, 1.),
    ("definitionMass", 0, 0.01), ("definitionMass", 3, 2.5), ("dynamicMass", 7, 0.51099895),
    ("dynamicMass", 15, 0.51099895), ("dynamicMass", 3, 2.5),
])
def test_fixed_matrix_mutations(field, index, value):
    t = fixture()
    t[field][index] = value
    with pytest.raises(ValueError):
        check.report(t)


@pytest.mark.parametrize("field", check.SCHEMA["float64"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_present_fields(field, value):
    t = fixture()
    t[field][0] = value
    with pytest.raises(ValueError):
        check.report(t)


@pytest.mark.parametrize("field", check.INFO)
@pytest.mark.parametrize("value", [0., 7., np.inf, -np.inf])
def test_absent_info_requires_nan(field, value):
    t = fixture()
    t[field][1] = value
    with pytest.raises(ValueError, match="NaN required"):
        check.report(t)


@pytest.mark.parametrize("field", check.STATE)
def test_single_ulp_helper_mismatch(field):
    t = fixture()
    t["actual" + field][8] = np.nextafter(t["expected" + field][8], np.inf)
    with pytest.raises(ValueError, match="byte mismatch"):
        check.report(t)


@pytest.mark.parametrize("field,index", [("KE", 6), ("Px", 0), ("Py", 0), ("Pz", 8), ("E", 12)])
def test_signed_zero_helper_mismatch(field, index):
    t = fixture()
    assert t["expected" + field][index] == 0.
    t["actual" + field][index] = -0.
    with pytest.raises(ValueError, match="byte mismatch"):
        check.report(t)


def test_zero_raw_byte_match_required_even_when_helper_agrees():
    t = fixture()
    t["expectedPx"][0] = t["actualPx"][0] = -0.
    with pytest.raises(ValueError, match="zero expected/raw Px"):
        check.report(t)


def test_stepped_ke_byte_match_required_even_when_helper_agrees():
    t = fixture()
    t["expectedKE"][14] = t["actualKE"][14] = -0.
    with pytest.raises(ValueError, match="stepped expected/recorded KE"):
        check.report(t)


def test_matching_negative_zero_and_non_native_endian():
    t = fixture()
    for prefix in ("raw", "actual", "expected"):
        t[prefix + "Px"][0] = -0.
    assert check.report(t)["exact_helper_match"]
    t = {field: value.astype(value.dtype.newbyteorder(">")) if value.dtype.kind in "if" else value
         for field, value in t.items()}
    assert check.report(t)["exact_helper_match"]


def test_no_python_libm_exact_oracle():
    t = fixture()
    # One-ULP C++-reference differences pass conservative physics checks; helper must match exactly.
    t["expectedPx"][9] = np.nextafter(t["expectedPx"][9], np.inf)
    t["actualPx"][9] = t["expectedPx"][9]
    assert check.report(t)["exact_helper_match"]


@pytest.mark.parametrize("field,index,value", [
    ("expectedKE", 9, 0.), ("expectedE", 15, 9.5), ("expectedPx", 9, -1.),
    ("expectedPy", 9, 1.), ("expectedPz", 9, 1.), ("expectedPx", 14, 1e-8),
    ("expectedPx", 9, 1e300),
])
def test_agreeing_helper_and_reference_still_need_physical_contract(field, index, value):
    t = fixture()
    t[field][index] = t[field.replace("expected", "actual")][index] = value
    with pytest.raises(ValueError):
        check.report(t)


@pytest.mark.parametrize("field,index,value", [("rawE", 7, 10.51099895), ("rawPz", 0, -10.),
                                               ("rawPx", 0, 1.), ("rawPz", 6, 1e-8),
                                               ("rawPz", 10, 1.)])
def test_raw_physical_contract_even_when_zero_expected_agrees(field, index, value):
    t = fixture()
    t[field][index] = value
    if index < 8:
        t[field.replace("raw", "expected")][index] = t[field.replace("raw", "actual")][index] = value
    with pytest.raises(ValueError):
        check.report(t)


def test_missing_tree_and_wrong_object(tmp_path):
    path = tmp_path / "missing.root"
    with uproot.recreate(path) as root:
        root["unrelated"] = "not the fixture"
    with pytest.raises(ValueError, match="missing tree"):
        check.read(path)
    with uproot.recreate(path) as root:
        root[check.TREE] = "not a TTree"
    with pytest.raises(ValueError, match="expected TTree"):
        check.read(path)


@pytest.mark.parametrize("latest_good", [False, True])
def test_highest_cycle(tmp_path, latest_good):
    good, bad = fixture(), fixture()
    bad["pass"][15] = False
    path = tmp_path / "cycles.root"
    with uproot.recreate(path) as root:
        write_tree(root, bad if latest_good else good)
        write_tree(root, good if latest_good else bad)
    with uproot.open(path) as root:
        assert root[check.TREE + ";1"].num_entries == 16
        assert root[check.TREE + ";2"].num_entries == 16
    if latest_good:
        assert check.report(check.read(path))["exact_helper_match"]
    else:
        with pytest.raises(ValueError, match="pass"):
            check.report(check.read(path))


def test_latest_partial_cycle_is_rejected(tmp_path):
    path = tmp_path / "partial.root"
    with uproot.recreate(path) as root:
        write_tree(root, fixture())
        write_tree(root, {field: value[:1] for field, value in fixture().items()})
    with pytest.raises(ValueError, match="16 scalar rows"):
        check.report(check.read(path))


def test_cli_success(tmp_path, capsys):
    path = root_file(tmp_path / "fixture.root")
    output = tmp_path / "subdir" / "report.json"
    check.main([str(path), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["input"] == str(path)
    assert result["exact_helper_match"]
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("existing", [False, True])
def test_cli_failure_no_success_or_clobber(tmp_path, existing):
    t = fixture()
    t["actualE"][0] += 1.
    path = root_file(tmp_path / "fixture.root", t)
    output = tmp_path / "report.json"
    if existing:
        output.write_text("preserve me")
    with pytest.raises(ValueError):
        check.main([str(path), "--output", str(output)])
    if existing:
        assert output.read_text() == "preserve me"
    else:
        assert not output.exists()


def test_cli_valid_input_still_refuses_existing_report(tmp_path):
    path = root_file(tmp_path / "fixture.root")
    output = tmp_path / "report.json"
    output.write_text("preserve me")
    with pytest.raises(ValueError, match="clobber"):
        check.main([str(path), "--output", str(output)])
    assert output.read_text() == "preserve me"


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_cli_input_aliases(tmp_path, alias, monkeypatch):
    path = root_file(tmp_path / "fixture.root")
    original = path.read_bytes()
    output = tmp_path / "alias.root"
    if alias == "same":
        output = path
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("fixture.root")
    elif alias == "symlink":
        output.symlink_to(path)
    else:
        os.link(path, output)
    with pytest.raises(ValueError, match="overwrite input"):
        check.main([str(path), "--output", str(output)])
    assert path.read_bytes() == original


def test_cli_dangling_output_symlink(tmp_path):
    path = root_file(tmp_path / "fixture.root")
    output, target = tmp_path / "alias.json", tmp_path / "target.json"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="clobber"):
        check.main([str(path), "--output", str(output)])
    assert output.is_symlink() and not target.exists()


def test_module_cli_failure_exit(tmp_path):
    path = root_file(tmp_path / "bad.root", {field: value[:1] for field, value in fixture().items()})
    output = tmp_path / "report.json"
    env = os.environ | {"PYTHONPATH": str(Path(check.__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-m", "mu2efastsim.capture_end_kinematics", str(path),
                             "--output", str(output)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "16 scalar rows" in result.stderr
    assert not output.exists()
