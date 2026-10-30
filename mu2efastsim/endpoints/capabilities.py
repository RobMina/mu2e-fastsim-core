"""Configuration-aware result contracts, without simulation or endpoint I/O imports.

``available`` means produced on a successful task, not commissioned physics.
Descriptors are additive to the kit's historical flat metric list. Conditional
requests are admitted, then checked against actual results; no metric is filtered.
Only explicitly configured, locally visible files are inspected (never PNFS,
symlink targets, remote endpoints, directory scans, or physics-data contents).
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat


_SCOPE = {
    "kind": "implementation_support",
    "physics_commissioned": False,
    "availability": "Returned by a successful task; not a physics-validity envelope.",
    "conditional": "Admitted provisionally; requested results must exist and be finite and nonsentinel.",
    "files": "Local presence observations only; no remote probing or lifetime guarantee.",
}
_EXTRACTED = tuple(f"clusters_per_generated_{s}" for s in ("EX", "T1", "T2", "M"))
_STOP_PROFILE = ("stops_foil_mean", "stop_r_mean", "stop_frac_upstream_half")
_DEADTIME = ("deadtime", "deadtime_err", "deadtime_full", "deadtime_full_lo", "deadtime_full_hi",
             "deadtime_selected")
_LOOKUP_IDS = ("6000_0", "2370_1", "3200_0")
_TYPE_PATTERN = r"clusters_per_generated_type[1-9][0-9]*"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COUNTS = {
    "events", "generated_events", "accepted_events", "events_with_steps", "jobs_ok", "library_reused_jobs",
    "stops", "muons", "foils", "n_terms", "pytest_failed", "vetoed", "selected", "passing_gate",
    "deadtime_selected", "primary_muons", "straw_steps", "digis", "hits", "seeds",
    "fit_attempts", "fit_successes", "events_with_fit", "written_events", "active_hits",
}
_UNITS = {
    "events_per_s": "events/s", "events_with_steps_per_s": "events/s",
    "ms_per_event_with_steps": "ms/event", "photon_rows_per_s": "rows/s", "photons_per_row": "photons/row",
    "sipms_per_s": "SiPMs/s", "charges_per_photon": "charges/photon", "stop_r_mean": "mm",
    "muons_per_s_warm": "muons/s", "gpu_hours": "GPU*h", "est_su": "SU",
    "accepted_per_gpu_hour": "events/(GPU*h)", "accepted_per_gpu_hour_chain": "events/(GPU*h)",
    "accepted_per_est_su": "events/SU",
}


def _field(obj, name, default=()):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _unit(name):
    if name in _UNITS:
        return _UNITS[name]
    if name in _COUNTS:
        return "count"
    if name.startswith("clusters_per_generated_"):
        return "clusters/event"
    if name.endswith("_s") or name == "input_wait_s_mean":
        return "s"
    return "1"  # fractions, relative deviations, chi2/ndf-like scores and boolean flags


def _metric(name, status="available", reason="Returned on successful task completion."):
    return {"availability": status, "unit": _unit(name), "reason": reason}


def _set(metrics, names, status, reason):
    for name in names:
        if name in metrics:
            metrics[name].update(availability=status, reason=reason)


def _crv_manifest():
    # kit_task deliberately has no numpy/awkward imports. Do not import the scorer.
    try:
        module = importlib.import_module("crvfastsim.kit_task")
    except ImportError:
        return None
    terms = getattr(module, "TERMS", None)
    return tuple(terms) if terms is not None else None


def _rule(when, *, unavailable=(), conditional=(), reason):
    return {"when": when, "unavailable": list(unavailable), "conditional": list(conditional), "reason": reason}


def _routes(task):
    if task == "chain_point":
        return [{"name": "streaming", "requires": ["stream_in"]}]
    if task in ("beam_point", "crv_response", "tracker_point"):
        rows = [{"name": "mounted", "requires": ["fnal_data"]},
                {"name": "streaming", "requires": ["stream_in", "no_fnal_data"]}]
        if task == "crv_response":
            rows[1]["when"] = {"tables": ["vr5"]}
            rows[0]["reason"] = "tables=all requires the mounted FNAL/EAF route."
        return rows
    if task in ("chain", "regression", "veto_c4"):
        return [{"name": "mounted", "requires": ["fnal_data"]}]
    return [{"name": "plain", "requires": []}]


def describe_tasks(tasks, chainpoint_module):
    """Return {task: additive descriptor}; never resolve files or contact endpoints.

    ``metric_rules`` use conjunctions of parameter membership tests. ``when={}``
    always applies. A pattern is recognized only when its ``when`` matches. Base
    schemas and all overrides explicitly distinguish available/conditional/unavailable.
    """
    definitions = dict(tasks)
    if chainpoint_module is not None:
        definitions["chain_point"] = {"metrics": chainpoint_module.METRICS, "params": chainpoint_module.PARAMS}
    result = {}
    for task, definition in definitions.items():
        metrics = {name: _metric(name) for name in _field(definition, "metrics")}
        rules, patterns, configs = [], [], {}
        if task == "beam_point":
            metrics.setdefault("input_wait_s", _metric("input_wait_s", "conditional", "Streaming route only."))
            _set(metrics, ("stop_frac_mu_minus",), "conditional", "Requires a nonempty negative-muon population.")
            _set(metrics, _STOP_PROFILE, "conditional", "Requires stops > 0; -1 is a missing-population sentinel.")
            for name in _STOP_PROFILE:
                if name in metrics:
                    metrics[name]["runtime_check"] = {"positive_metric": "stops", "reject_values": [-1]}
            if "stop_frac_mu_minus" in metrics:
                metrics["stop_frac_mu_minus"]["runtime_check"] = {"minimum": 0, "maximum": 1}
            if "stop_frac_err" in metrics:
                metrics["stop_frac_err"]["uncertainty"] = {
                    "of": "stop_frac", "kind": "binomial_standard_error",
                    "formula": "sqrt(p*(1-p)/muons)",
                    "scope": "Finite sampled-muon counting only; not model/systematic or independent-replica uncertainty.",
                    "gp_observation_noise": False,
                }
            rules.append(_rule({"repeat": [0]}, unavailable=["muons_per_s_warm"],
                               reason="repeat=0 performs no warm timing pass; producer writes -1."))
            configs = {"geometry": {"parameter": "geom", "file_kinds": ["geom"]},
                       "repeat": {"warm_timing_requires": "repeat > 0"}}
        elif task == "tracker_point":
            configs = {"scenario": "extracted_v04_primary_v1", "transport": "stage C sliced MSC",
                       "population": "all primary muons", "secondary_library": False,
                       "fit": "hybrid F2", "momentum_MeV_c": 2000, "fit_material": False,
                       "stock_event_conditioning": False, "F1_supported": False,
                       "denominator": "all generated events", "retention": "JSON metadata only"}
            for spec in metrics.values():
                spec["runtime_check"] = {"minimum": 0}
            metrics["fit_event_fraction"]["runtime_check"]["maximum"] = 1
            metrics["fit_event_fraction_se"]["uncertainty"] = {
                "of": "fit_event_fraction", "kind": "binomial_standard_error",
                "formula": "sqrt(p*(1-p)/generated_events)",
                "scope": "Generated-event counting only; not model/systematic or independent-replica uncertainty.",
                "gp_observation_noise": False}
        elif task == "chain_point":
            run_configs = getattr(chainpoint_module, "RUN_CONFIGS", {})
            nonextracted = [k for k in run_configs if k != 0]
            rules.append(_rule({"run_config": nonextracted}, unavailable=_EXTRACTED,
                               reason="EX/T1/T2/M labels apply only to run_config=0 (extracted_v04)."))
            patterns.append({"pattern": _TYPE_PATTERN, "when": {"run_config": nonextracted},
                             "status": "conditional", "unit": "clusters/event",
                             "reason": "Emitted only for sector types present in the merged cluster counts; no type census is assumed."})
            configs = {"run_config": {str(k): {"name": v[0], "gdml": v[1], "lookup_table_ids": list(v[2]),
                                               "extracted": bool(v[3])} for k, v in run_configs.items()},
                       "gen_set": {str(k): {"file_pattern": v[0], "files": len(v[1])}
                                   for k, v in getattr(chainpoint_module, "GEN_SETS", {}).items()},
                       "compatibility": "Extracted table sets and generated sets 4/5 require run_config=0; table sets 2/3 require nonzero run_config."}
            if hasattr(chainpoint_module, "NATIVE_GEN_SET"):
                configs["gen_set"][str(chainpoint_module.NATIVE_GEN_SET)] = {
                    "generator": "native-cry-1.7", "files": 0, "events_parameter": "gen_events",
                    "requires": "CRY-enabled image and matching core/crv code; extracted_v04 only",
                    "seed": "seed+1000*k per sub-job; caller selects independent seeds across points",
                    "restrictions": "files_per_job=1, shard_offset=max_events=0, no prefetch"}
            if hasattr(chainpoint_module, "DEVICE_GEN_SET"):
                configs["gen_set"][str(chainpoint_module.DEVICE_GEN_SET)] = {
                    "generator": "jax-cry-1.7", "files": 0, "events_parameter": "gen_events",
                    "requires": "JAX GPU and verified CRY distribution tables; extracted_v04 only",
                    "seed": "seed+1000*k per sub-job; caller selects independent seeds across points",
                    "restrictions": "files_per_job=1, shard_offset=max_events=0, no prefetch"}
        elif task == "crv_response":
            terms = _crv_manifest()
            if terms is None:
                _set(metrics, [n for n in metrics if n.startswith("term_")], "conditional",
                     "Frozen kit_task.TERMS unavailable; required-term completeness cannot be established.")
            else:
                for term in terms:
                    metrics.setdefault("term_" + term, _metric("term_" + term))
                _set(metrics, [n for n in metrics if n.startswith("term_") and n[5:] not in terms], "conditional",
                     "Not in the frozen kit_task.TERMS required-term manifest.")
                _set(metrics, ["term_" + term for term in terms], "available",
                     "Non-excluded frozen terms are required by kit_task.score_result on successful completion.")
            rules.append(_rule({"period": ["P3"]}, conditional=["term_dt_cluster_T2", "objective_all"],
                               reason="P3 excludes dt_cluster_T2; excluded terms and the all-term score need not be finite or present."))
            for name in ("input_wait_s", "inputs_reused"):
                metrics[name] = _metric(name, "conditional", "Streaming route only.")
            if "objective_sigma" in metrics:
                metrics["objective_sigma"]["uncertainty"] = {
                    "of": "objective", "kind": "candidate_statistical_estimate",
                    "scope": "Candidate Poisson-resampling/counting estimate; term sigmas combined in quadrature. "
                             "Not independent seed/input replicas, reference uncertainty, or model systematics.",
                    "gp_observation_noise": False,
                }
            configs = {"period": {"P4": {"excluded_terms": []}, "P3": {"excluded_terms": ["dt_cluster_T2"]},
                                  "raw": {"excluded_terms": []}},
                       "tables": {"vr5": {"routes": ["mounted", "streaming"]}, "all": {"routes": ["mounted"]}},
                       "required_terms": list(terms) if terms is not None else None,
                       "required_terms_source": "crvfastsim.kit_task.TERMS" if terms is not None else "unavailable",
                       "lookup_table_ids": list(_LOOKUP_IDS), "scenario": "extracted_v04"}
            if "response_config" in _field(definition, "params"):
                configs["response_config"] = {"schema": 1, "scenario": "extracted_v04",
                                              "lookup_table_ids": list(_LOOKUP_IDS), "lookup_tables_fixed": True,
                                              "role": "fixed_configuration", "asset_overrides": False}
        elif task == "chain":
            _set(metrics, _EXTRACTED, "conditional", "Emitted only for sector types with produced clusters.")
            _set(metrics, ("max_abs_cluster_rate_rel",), "conditional",
                 "Requires readable held-out G4 reference files and a nonzero reference cluster rate.")
        elif task == "veto_c4":
            _set(metrics, _DEADTIME, "conditional",
                 "Requires deadtime_run1a/ce_mix1bb_dev_sel.parquet and sufficient selected tracks/frames.")
            _set(metrics, ("inefficiency", "vetoed_frac", "vetoed_lo", "vetoed_hi"), "conditional",
                 "Requires a nonempty selected cosmic-track population.")
            if "deadtime_err" in metrics:
                metrics["deadtime_err"]["uncertainty"] = {
                    "of": "deadtime", "kind": "factorised_frame_standard_error",
                    "scope": "Frame-coverage sample standard deviation / sqrt(frames); needs at least two frames.",
                    "gp_observation_noise": False,
                }
        elif task == "v4_response":
            _set(metrics, [n for n in metrics if not n.endswith("_pass")], "conditional",
                 "Only returned when the corresponding V4 diagnostic JSON exists; pass=0 alone is still returned.")
        elif task == "reco_bench":
            _set(metrics, metrics, "conditional", "Metric must be present in the producer's benchmark JSON.")
        result[task] = {"metrics": metrics, "metric_rules": rules, "metric_patterns": patterns,
                        "configurations": configs, "route_requirements": _routes(task),
                        "file_kinds": ["geom"] if task == "beam_point" else [],
                        "validation_scope": deepcopy(_SCOPE)}
    return result


def _matches(when, params):
    return all(params.get(name) in values for name, values in when.items())


def _local_file(root, relative):
    """Present/absent/unknown without following symlinks or probing PNFS.

    An absent mount root is unknown, not evidence that an endpoint lacks the file.
    Once that root is locally visible, a missing exact child is negative evidence.
    """
    if not root:
        return "unknown", "No locally configured FNAL data root."
    try:
        root = Path(os.path.normpath(os.fspath(root)))
    except TypeError:
        return "unknown", "FNAL data root is not a local filesystem path."
    target = root / relative
    if not root.is_absolute() or "pnfs" in target.parts:
        return "unknown", "PNFS/relative-path inspection is refused; no local availability observation."
    current = Path(target.anchor)
    root_depth = len(root.parts)
    for depth, part in enumerate(target.parts[1:], 2):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if depth <= root_depth:
                return "unknown", "Configured FNAL root is not visible on this host."
            return "absent", f"Required file is absent under the locally visible FNAL root: {relative}."
        except OSError:
            return "unknown", "Local visibility could not be established."
        if stat.S_ISLNK(info.st_mode):
            return "unknown", "Symlink traversal is refused (may lead to PNFS or nonlocal storage)."
        if current != target and not stat.S_ISDIR(info.st_mode):
            return "unknown", "Configured path is not a locally visible directory."
    if not stat.S_ISREG(info.st_mode):
        return "absent", f"Required regular file is absent: {relative}."
    return "present", f"Observed local file {relative}; runtime readability and contents remain unchecked."


def _route(task, endpoint, params):
    mounted = bool(getattr(endpoint, "fnal_data", None))
    streaming = hasattr(endpoint, "stream_in")
    for route in _routes(task):
        requirements = route["requires"]
        if ("fnal_data" in requirements and not mounted or
                "no_fnal_data" in requirements and mounted or
                "stream_in" in requirements and not streaming or
                not _matches(route.get("when", {}), params)):
            continue
        return {"name": route["name"], "supported": True, "requirements": requirements,
                "endpoint": getattr(endpoint, "name", "unknown"),
                "reason": "Endpoint capabilities satisfy the registered route; not a live readiness check."}
    return {"name": "unavailable", "supported": False, "requirements": _routes(task),
            "endpoint": getattr(endpoint, "name", "unknown"),
            "reason": "No supported endpoint route for this configuration (tables=all requires mounted FNAL/EAF)."}


def resolve(task, params, endpoint, tasks, chainpoint_module):
    """Resolve an already normalized parameter point; no rendering or remote calls.

    Unsupported routes are reported (``route.supported=False``) and their metrics
    marked unavailable. Parameter, resource, and dispatch admission remain the
    caller's responsibility. ``configuration`` is the normalized point, with CRV
    inline JSON decoded for inspection when supplied.
    """
    descriptors = describe_tasks(tasks, chainpoint_module)
    if task not in descriptors:
        raise ValueError(f"fastsimkit: unknown capability task {task!r}")
    descriptor = descriptors[task]
    metrics = descriptor["metrics"]
    configuration = deepcopy({k: v for k, v in params.items() if k != "task"})
    if task == "crv_response" and configuration.get("response_config"):
        value = configuration["response_config"]
        if isinstance(value, str):
            configuration["response_config"] = json.loads(value)
    for rule in descriptor["metric_rules"]:
        if _matches(rule["when"], params):
            for status in ("unavailable", "conditional"):
                _set(metrics, rule[status], status, rule["reason"])
    patterns = [dict(p, availability=p["status"]) for p in descriptor["metric_patterns"] if _matches(p["when"], params)]
    route = _route(task, endpoint, params)
    stream_only = ("input_wait_s", "inputs_reused") if task == "crv_response" else ("input_wait_s",)
    if task in ("beam_point", "crv_response"):
        _set(metrics, stream_only, "available" if route["name"] == "streaming" else "unavailable",
             "Emitted by the streaming delegate only." if route["name"] == "streaming" else
             "Mounted/plain routes do not produce streaming wait or cache-reuse metrics.")
    if task == "veto_c4":
        observed, reason = _local_file(getattr(endpoint, "fnal_data", None),
                                       "deadtime_run1a/ce_mix1bb_dev_sel.parquet")
        _set(metrics, _DEADTIME, "unavailable" if observed == "absent" else "conditional",
             reason + " Requires sufficient selected tracks/frames and finite runtime output.")
    if task == "chain":
        observations = [_local_file(getattr(endpoint, "fnal_data", None),
                                    f"v2_fastC_v1/ref.001400_{i:08d}.root")
                        for i in range(11, 11 + params.get("shards", 1))]
        absent = all(s == "absent" for s, _ in observations)
        reason = ("No held-out G4 reference files exist under the locally visible FNAL root." if absent else
                  "G4 reference availability is conditional: " + observations[0][1] +
                  " A readable reference with nonzero cluster rates is required at runtime.")
        _set(metrics, ("max_abs_cluster_rate_rel",), "unavailable" if absent else "conditional", reason)
    if not route["supported"]:
        _set(metrics, metrics, "unavailable", route["reason"])
        for pattern in patterns:
            pattern.update(availability="unavailable", status="unavailable", reason=route["reason"])
    return {"task": task, "metrics": metrics, "metric_patterns": patterns, "configuration": configuration,
            "configurations": descriptor["configurations"], "route": route,
            "validation_scope": descriptor["validation_scope"], "file_kinds": descriptor["file_kinds"]}


def _lookup(name, resolved):
    metric = resolved["metrics"].get(name)
    if metric is not None:
        return metric
    for pattern in resolved.get("metric_patterns", []):
        if re.fullmatch(pattern["pattern"], name):
            return {"availability": pattern.get("availability", pattern.get("status", "conditional")),
                    "unit": pattern["unit"], "reason": pattern["reason"]}
    return None


def require_metrics(requested, resolved):
    """Validate optional metric names; return a sorted list, refusing duplicates.

    Conditional metrics remain explicitly conditional in ``resolved``. This does
    not materialize a dynamic sector census or promote conditional availability.
    """
    if requested is None:
        return []
    if not isinstance(requested, (list, tuple)):
        raise ValueError("fastsimkit: requested metrics must be a list of names")
    seen = set()
    for name in requested:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"fastsimkit: invalid requested metric name {name!r}")
        if name in seen:
            raise ValueError(f"fastsimkit: duplicate requested metric {name!r}")
        seen.add(name)
        metric = _lookup(name, resolved)
        if metric is None:
            raise ValueError(f"fastsimkit: unknown requested metric {name!r}")
        if metric["availability"] == "unavailable":
            raise ValueError(f"fastsimkit: requested metric {name!r} unavailable: {metric['reason']}")
    return sorted(seen)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def validate_results(metrics, requested, resolved):
    """Check requested outputs in place, without filtering or altering any metrics.

    Unrequested legacy NaNs/sentinels are retained: e.g. extracted sector labels on
    non-extracted chain points. Availability is not evidence that a finite output
    was actually observed, so even available requested metrics are rechecked.
    """
    names = require_metrics(requested, resolved)
    if not isinstance(metrics, dict):
        raise ValueError("fastsimkit: results metrics must be an object")
    for name in names:
        if name not in metrics:
            raise ValueError(f"fastsimkit: requested metric {name!r} missing from results")
        value = metrics[name]
        if not _finite(value):
            raise ValueError(f"fastsimkit: requested metric {name!r} must be a finite number")
        check = _lookup(name, resolved).get("runtime_check", {})
        if value in check.get("reject_values", ()):
            raise ValueError(f"fastsimkit: requested metric {name!r} is an unavailable-population sentinel")
        population = check.get("positive_metric")
        if population and (not _finite(metrics.get(population)) or metrics[population] <= 0):
            raise ValueError(f"fastsimkit: requested metric {name!r} requires {population} > 0")
        if ("minimum" in check and value < check["minimum"] or
                "maximum" in check and value > check["maximum"]):
            raise ValueError(f"fastsimkit: requested metric {name!r} outside its result domain")


def contract_results(metrics, requested, resolved):
    """Represent absent optional outputs explicitly, without emitting NaN/sentinels to BO."""
    validate_results(metrics, requested, resolved)
    output, unavailable = {}, {}
    for name, value in metrics.items():
        spec = _lookup(name, resolved)
        if spec is not None and spec["availability"] == "unavailable":
            unavailable[name] = spec["reason"]
            continue
        try:
            if spec is not None:
                validate_results(metrics, [name], resolved)
            elif not _finite(value):
                raise ValueError(f"unrecognized nonfinite result metric {name!r}")
        except ValueError as exc:
            if spec is None or spec["availability"] != "conditional":
                raise
            unavailable[name] = str(exc)
            continue
        output[name] = value
    for name, spec in resolved["metrics"].items():
        if name not in metrics and spec["availability"] != "available":
            unavailable.setdefault(name, spec["reason"])
    return output, unavailable
