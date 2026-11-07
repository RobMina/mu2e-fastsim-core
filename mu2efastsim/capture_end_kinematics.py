"""Strict checker for the constructed 16-case end-kinematics helper fixture.

This is diagnostic-only evidence, not transport, persisted SimParticle readback,
real-engine finalization or response-bank admission. ROOT alone cannot certify
successful art endJob/process exit; retain the job logs and exit status separately.
C++ supplies the exact legacy momentum reference: Python deliberately does not
reconstruct sqrt/libm results. All row comparisons are vectorized; loops are over
fixed fields only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .saved_g4 import require

TREE = "fixture/endKinematics"
CASE_NAMES = np.array([
    "zero_gamma_default_step", "zero_gamma_absent_no_step", "zero_gamma_stale_no_step",
    "zero_electron_absent_step", "zero_electron_default_no_step", "zero_electron_stale_step",
    "zero_electron_at_rest_step", "zero_electron_dynamic_mass", "one_gamma_raw_zero_recorded_7",
    "one_electron_raw_zero_recorded_7", "multi_gamma_raw_3_recorded_7",
    "multi_electron_raw_3_recorded_7", "one_gamma_default_info", "one_electron_default_info",
    "multi_electron_recorded_zero", "multi_electron_dynamic_mass",
])
STATE = "KE Px Py Pz E".split()
INFO = "infoKE infoDirX infoDirY infoDirZ".split()
SCHEMA = {
    "int64": "run subrun event".split(),
    "int32": "caseIndex pdg nSteps".split(),
    "bool": "diagnosticOnly bankAdmissible realEngine stepPresent infoPresent helperKEPresent helperMomentumPresent pass".split(),
    "float64": [*"dynamicMass definitionMass trackLength rawDirX rawDirY rawDirZ".split(), *INFO,
                *(prefix + field for prefix in ("raw", "actual", "expected") for field in STATE)],
    "str": "caseName infoKind error".split(),
}
MATRIX = {
    "pdg": [22, 22, 22, 11, 11, 11, 11, 11, 22, 11, 22, 11, 22, 11, 11, 11],
    "nSteps": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 4, 1, 1, 2, 2],
    "trackLength": [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 5., 0., 0., 2., 2.],
    "rawKE": [10., 10., 10., 10., 10., 10., 0., 10., 0., 0., 3., 3., 5., 5., 5., 3.],
    "infoKind": ["default", "absent", "recorded", "absent", "default", "recorded", "default", "recorded",
                 "recorded", "recorded", "recorded", "recorded", "default", "default", "recorded", "recorded"],
    "stepPresent": [True, False, False, True, False, True, True, False,
                    True, True, True, True, True, True, True, True],
}
RECORDED_KE = np.array([0., np.nan, 9., np.nan, 0., 9., 0., 9., 7., 7., 7., 7., 0., 0., 0., 7.])


def read(path):
    """Read the highest tree cycle, including incomplete/failing checkpoints."""
    with uproot.open(path) as root:
        require(TREE in root, f"missing tree {TREE}")
        tree = root[TREE]  # uproot's unqualified lookup chooses the highest cycle.
        require(tree.classname == "TTree", f"{TREE}: expected TTree")
        return {field: ak.to_numpy(value, allow_missing=False)
                for field, value in tree.arrays(library="ak", how=dict).items()}


def _schema(table):
    require(set(table) == {field for fields in SCHEMA.values() for field in fields},
            "missing or unexpected fields")
    result = {}
    for kind, fields in SCHEMA.items():
        for field in fields:
            require(not np.ma.isMaskedArray(table[field]), f"{field}: masked data")
            value = np.asarray(table[field])
            require(value.ndim == 1 and len(value) == 16, f"{field}: expected 16 scalar rows")
            if kind == "str":
                require(value.dtype.kind in "OU" and pd.api.types.infer_dtype(value, skipna=False) == "string",
                        f"{field}: must be strings")
            else:
                dtype = np.dtype(kind)
                require(value.dtype.kind == dtype.kind and value.dtype.itemsize == dtype.itemsize,
                        f"{field}: must be {kind}")
                # Native-endian normalization preserves every float64 bit, including signed zero.
                value = value.astype(dtype, copy=False)
            result[field] = value
    return result


def _equal(actual, expected, label):
    require(np.all(actual == expected), f"{label}: fixed matrix mismatch")


def _exact(actual, expected, label):
    """Compare individual finite float64 representations, never padded structs."""
    require(np.array_equal(actual.view(np.uint64), expected.view(np.uint64)),
            f"{label}: byte mismatch (including signed zero)")


def _physical(table, prefix, mass):
    """Conservative independent constraints, not an alternative exact momentum oracle."""
    ke, px, py, pz, energy = (table[prefix + field] for field in STATE)
    require(np.all(ke >= 0.) and np.all(energy >= 0.), f"{prefix}: negative energy")
    _equal(energy, ke + mass, prefix + "E mass source")
    # Bounds reject wildly corrupt finite inputs before squaring (avoids overflow warnings).
    require(np.all(np.abs(np.column_stack((px, py, pz))) <= energy[:, None]),
            f"{prefix}: momentum exceeds energy")
    require(np.allclose(px*px + py*py + pz*pz, ke*(ke + 2.*mass), rtol=1e-12, atol=1e-12),
            f"{prefix}: mass-shell constraint")
    rest = ke == 0.
    require(np.all(np.column_stack((px, py, pz))[rest] == 0.), f"{prefix}: nonzero rest momentum")


def report(table):
    """Require the complete ordered passing matrix and bit-identical helper results."""
    t = _schema(table)
    for field, expected in dict(run=16, subrun=0, event=1, caseIndex=np.arange(1, 17), caseName=CASE_NAMES,
                                diagnosticOnly=True, bankAdmissible=False, realEngine=False,
                                helperKEPresent=True, helperMomentumPresent=True, pass_=True, error="").items():
        _equal(t["pass" if field == "pass_" else field], expected, field)
    for field, expected in MATRIX.items():
        _equal(t[field], np.asarray(expected), field)
    zero, stepped = t["nSteps"] == 0, t["nSteps"] > 0
    require(np.count_nonzero(zero) == 8 and np.count_nonzero(stepped) == 8, "expected 8 zero/8 positive steps")
    present, recorded = t["infoKind"] != "absent", t["infoKind"] == "recorded"
    _equal(t["infoPresent"], present, "infoPresent")
    # Only absent-information fields may be NaN; infinity is never a sentinel.
    for field in SCHEMA["float64"]:
        if field in INFO:
            require(np.isnan(t[field][~present]).all() and np.isfinite(t[field][present]).all(),
                    f"{field}: NaN required only for absent information")
        else:
            require(np.isfinite(t[field]).all(), f"{field}: nonfinite value")
    _equal(t["infoKE"][present], RECORDED_KE[present], "infoKE")
    for axis, raw, info in (("X", 0., 0.6), ("Y", 0., 0.8), ("Z", 1., 0.)):
        _equal(t["rawDir" + axis], raw, "rawDir" + axis)
        _equal(t["infoDir" + axis][present], np.where(recorded, info, 0.)[present], "infoDir" + axis)

    electron = t["pdg"] == 11
    mass = t["definitionMass"]
    _equal(mass[~electron], 0., "gamma definitionMass")
    require(np.unique(mass[electron]).size == 1 and
            np.allclose(mass[electron], 0.51099895, rtol=0., atol=1e-7),
            "electron definitionMass: inconsistent or non-electron mass")
    # Allow tiny Geant4/CLHEP constant-version differences, not altered definitions.
    override = np.isin(t["caseIndex"], [8, 16])
    _equal(t["dynamicMass"], np.where(override, 2.5, mass), "dynamicMass")
    require(np.all(t["dynamicMass"][override] != mass[override]), "mass override must differ from definition")

    for field in STATE:
        _exact(t["actual" + field], t["expected" + field], "actual/expected " + field)
        _exact(t["expected" + field][zero], t["raw" + field][zero], "zero expected/raw " + field)
    _exact(t["expectedKE"][stepped], t["infoKE"][stepped], "stepped expected/recorded KE")
    _physical(t, "raw", t["dynamicMass"])
    _physical(t, "expected", np.where(zero, t["dynamicMass"], mass))
    _equal(t["rawPx"], 0., "rawPx")
    _equal(t["rawPy"], 0., "rawPy")
    require(np.all(t["rawPz"] >= 0.), "rawPz: reversed direction")
    _equal(t["expectedPz"][stepped], 0., "stepped expectedPz")
    px, py = t["expectedPx"][stepped], t["expectedPy"][stepped]
    require(np.all(px >= 0.) and np.all(py >= 0.) and
            np.allclose(0.8*px, 0.6*py, rtol=1e-12, atol=1e-12), "stepped recorded direction mismatch")
    return {"diagnostic_only": True, "bank_admissible": False, "real_engine": False,
            "cases": 16, "zero_step": 8, "positive_step": 8, "exact_helper_match": True,
            "run": 16, "subrun": 0, "event": 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.output.resolve() != args.input.resolve() and
            not (args.output.exists() and args.input.exists() and args.output.samefile(args.input)),
            "output must not overwrite input or its aliases")
    # Exclusive creation also protects existing reports, hardlinks, and dangling symlinks.
    require(not args.output.exists() and not args.output.is_symlink(), "output already exists; refusing to clobber")
    result = report(read(args.input))
    result["input"] = str(args.input)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("x")
    try:
        with output:
            output.write(text)
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(text, end="")


if __name__ == "__main__":
    main()
