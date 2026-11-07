"""Small-sample capture distribution diagnostics, never a geometry-equivalence certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from . import capture
from .saved_g4 import frame, match, require, unique

INCIDENT = "lx ly lz lpx lpy lpz energy kinetic mass charge time properTime weight".split()


def observables(tables):
    """Aggregate one value per independent event, retaining zero-response trials."""
    e, t, s = (tables[name] for name in ("events", "tracks", "steps"))
    events = unique(e, capture.EVENT, "event").assign(_event=np.arange(len(e["event"])))
    tr = match(frame(t, capture.EVENT), events, capture.EVENT, "track")["_event"].to_numpy()
    sr = match(frame(s, capture.EVENT), events, capture.EVENT, "step")["_event"].to_numpy()
    count = len(events)
    return {
        "component_deposit_MeV": np.bincount(sr, weights=s["deposit"] * s["preInside"], minlength=count),
        "tracks": e["nTracks"], "steps": e["nSteps"],
        "first_exits": np.bincount(tr, weights=t["reason"] == 2, minlength=count),
        "neutral_tracks": np.bincount(tr, weights=t["begin_charge"] == 0, minlength=count),
        "zero_deposit_steps": np.bincount(sr, weights=s["deposit"] == 0, minlength=count),
        "at_rest_steps": np.bincount(sr, weights=s["post_stepStatus"] == 2, minlength=count),
    }


def compare(reference, candidate, expected_events):
    """Audit both fixed-state fixtures first; report differences without a pass/fail physics decision."""
    audits = [capture.audit(tables, expected_events) for tables in (reference, candidate)]
    require(expected_events > 1, "comparison needs at least two events")
    re, ce = reference["events"], candidate["events"]
    for key in ("physicsTag", "halfSize"):
        require(np.all(re[key] == re[key][0]) and np.all(ce[key] == re[key][0]), f"different fixture {key}")
    rt, ct = reference["tracks"], candidate["tracks"]
    rp, cp = rt["parent"] == 0, ct["parent"] == 0
    require(np.all(rt["pdg"][rp] == rt["pdg"][rp][0]) and
            np.all(ct["pdg"][cp] == rt["pdg"][rp][0]), "different primary species")
    rs = capture.vectors(rt, "entrance_", INCIDENT)[rp]
    cs = capture.vectors(ct, "entrance_", INCIDENT)[cp]
    capture.same(rs, rs[0], "reference fixed local entrance", 1e-7)
    capture.same(cs, rs[0], "candidate fixed local entrance", 1e-7)
    metrics = {}
    ro, co = observables(reference), observables(candidate)
    for key in ro:
        a, b = ro[key], co[key]
        se_a, se_b = float(np.std(a, ddof=1) / np.sqrt(len(a))), float(np.std(b, ddof=1) / np.sqrt(len(b)))
        delta, se = float(np.mean(b) - np.mean(a)), float(np.hypot(se_a, se_b))
        ks = ks_2samp(a, b, method="asymp")
        metrics[key] = {"reference_mean": float(np.mean(a)), "candidate_mean": float(np.mean(b)),
                        "reference_mean_se": se_a, "candidate_mean_se": se_b,
                        "difference": delta, "difference_se": se,
                        "difference_over_se": delta / se if se else None,
                        "ks_statistic": float(ks.statistic), "ks_pvalue_diagnostic": float(ks.pvalue)}
    return {"diagnostic_only": True, "bank_admissible": False, "geometry_equivalence_established": False,
            "seed_independence_verified_by_tables": False, "audits": audits,
            "note": "Small samples; no equivalence margins or multiplicity correction. KS p-values with ties are approximate. "
                    "Physics labels/box dimensions are checked, not complete material/physics manifest equality.",
            "local_incident_fields": INCIDENT,
            "max_candidate_local_entrance_difference": np.max(np.abs(cs - rs[0]), axis=0).tolist(),
            "metrics": metrics}


def compare_exact(reference, candidate, expected_events):
    """Audit and compare every stored capture field by identity, excluding timing values only."""
    audits = [capture.audit(tables, expected_events) for tables in (reference, candidate)]
    excluded = ["events.cpuTime", "events.realTime"]
    for name in capture.TABLES:
        a, b = reference[name], candidate[name]
        require(set(a) == set(b), f"{name}: fieldset mismatch")
        keys = capture.EVENT if name == "events" else [*capture.EVENT, "index"]
        left = unique(a, keys, name)
        right = unique(b, keys, name).assign(_candidate_row=np.arange(len(b["event"])))
        require(len(left) == len(right), f"{name}: row census mismatch")
        rows = match(left, right, keys, name)["_candidate_row"].to_numpy()
        for field in sorted(a):
            label = f"{name}.{field}"
            try:
                av, bv = np.asarray(a[field]), np.asarray(b[field])
            except ValueError as exc:
                raise ValueError(f"{label}: malformed array") from exc
            require(av.ndim > 0 and bv.ndim > 0 and
                    len(av) == len(left) and len(bv) == len(right), f"{label}: malformed rows")
            require(av.shape == bv.shape, f"{label}: shape mismatch")
            require(av.dtype == bv.dtype, f"{label}: dtype mismatch")
            if label not in excluded:
                require(np.array_equal(av, bv[rows]), f"{label}: exact field mismatch")
    return {"diagnostic_only": True, "bank_admissible": False, "capture_fields_identical": True,
            "excluded_fields": excluded, "audits": audits,
            "geometry_equivalence_established": False, "statistical_equivalence_established": False,
            "note": "Exact stored capture-field diagnostic only; no geometry or statistical equivalence established. "
                    "Timing values are excluded from equality, not from the audits."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exact", action="store_true", help="compare all capture fields exactly, except event timings")
    args = parser.parse_args(argv)
    for path in (args.reference, args.candidate):
        require(args.output.resolve() != path.resolve() and
                not (args.output.exists() and path.exists() and args.output.samefile(path)),
                "output must differ from both input paths")
    comparison = compare_exact if args.exact else compare
    result = comparison(capture.read(args.reference), capture.read(args.candidate), args.expected_events)
    result.update(reference=str(args.reference), candidate=str(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
