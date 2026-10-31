"""Dependency-free parameter admission for the kit and its streaming delegates.

These are structural/physical input domains, not commissioned physics envelopes.
Numbers must be JSON numbers; whole-valued floats remain compatible with Mu2eBO.
Empty strings retain the CRV task's existing omitted-knob convention.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Parameter:
    type: str
    unit: str = "1"
    role: str = "execution"
    minimum: float | int | None = None
    maximum: float | int | None = None
    exclusive_minimum: bool = False
    choices: tuple = ()
    allow_empty: bool = False

    def parse(self, task: str, name: str, value: Any):
        prefix = f"fastsimkit: {task} {name}"
        if self.allow_empty and isinstance(value, str) and value == "":
            return ""
        if self.type in ("integer", "number"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{prefix} must be a finite {self.type}, got {value!r}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{prefix} must be finite, got {value!r}")
            if self.type == "integer" and isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{prefix} must be a whole number, got {value!r}")
            if self.minimum is not None and (value < self.minimum or
                                             (self.exclusive_minimum and value == self.minimum)):
                op = ">" if self.exclusive_minimum else ">="
                raise ValueError(f"{prefix} must be {op} {self.minimum}, got {value!r}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"{prefix} must be <= {self.maximum}, got {value!r}")
            if self.type == "integer":
                value = int(value)
            else:
                try:
                    value = float(value)
                except OverflowError:
                    raise ValueError(f"{prefix} must be representable as a finite number") from None
        elif self.type == "crv_response_config":
            from crvfastsim import response_config
            if not isinstance(value, (str, dict)):
                raise ValueError(f"{prefix} must be a JSON object or inline JSON string")
            if isinstance(value, str) and value and not value.lstrip().startswith('{'):
                raise ValueError(f"{prefix} must be inline JSON, not a kit-host file path")
            return response_config.canonical(response_config.parse(value, kit=True))
        elif not isinstance(value, str):
            raise ValueError(f"{prefix} must be a string, got {value!r}")
        if self.choices and value not in self.choices:
            raise ValueError(f"{prefix} must be one of {list(self.choices)}, got {value!r}")
        if self.type == "geometry":
            if not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or ".." in value.split("/"):
                raise ValueError(f"{prefix} must be a plain geometry name or path")
            if value.startswith("/"):
                from mu2efastsim import site
                root = site.geometry_prefix()
                if not root:
                    raise ValueError(f"{prefix} absolute path needs FASTSIMKIT_GEOMETRY_PREFIX (the FNAL storage "
                                     "prefix geometry files may live under); it is not set")
                if not value.startswith(root.rstrip("/") + "/"):
                    raise ValueError(f"{prefix} path must be on FNAL storage ({root.rstrip('/')}/...)")
            elif value != "nominal" and ("/" in value or not value.endswith(".txt") or value.startswith("-")):
                raise ValueError(f"{prefix} must be nominal or a .txt file name under beam/geom")
        return value

    def describe(self):
        result = asdict(self)
        result["choices"] = list(self.choices)
        return result


def integer(minimum=0, maximum=None, *, unit="count", role="fidelity"):
    return Parameter("integer", unit, role, minimum, maximum)


def number(minimum=None, maximum=None, *, unit="1", role="analysis", positive=False, optional=False):
    return Parameter("number", unit, role, minimum, maximum, positive, allow_empty=optional)


def enum(*choices, role="scenario"):
    return Parameter("string", role=role, choices=choices)


def schema(task: str, names) -> dict[str, Parameter]:
    seed = integer(0, 2 ** 31 - 1, role="stochastic_stream")
    if task in ("reco_bench", "v4_response"):
        specs = {"events": integer(1), "seed": seed}
    elif task == "response_bench":
        specs = {"rows": integer(1), "sipms": integer(1), "seed": seed}
    elif task == "chain":
        specs = {"shards": integer(1, 10), "max_events": integer(), "seed": seed}
    elif task == "regression":
        specs = {"events": integer(1), "seed": seed}
    elif task == "tracker_point":
        specs = {"sample": enum("extracted_v04_primary_v1"),
                 "shard": integer(1, 20, role="input_replica"), "seed": seed}
    elif task == "beam_point":
        specs = {"geom": Parameter("geometry", role="physics_design"), "muons": integer(1, 10 ** 7),
                 "seed": seed, "repeat": integer(0, 5, role="timing_repeat")}
    elif task == "veto_c4":
        specs = {"selection": enum("run1a_dev"), "dt_lo": number(unit="ns"), "dt_hi": number(unit="ns"),
                 "min_pe": number(0, unit="PE"), "min_layers": integer(0, 4, role="analysis"),
                 "gate_start_lo": number(unit="ns"), "gate_end_hi": number(unit="ns"),
                 "gate_mode": Parameter("integer", role="analysis", choices=(0, 1)),
                 "pmin": number(0, unit="MeV/c"), "pmax": number(0, unit="MeV/c")}
    elif task == "crv_response":
        specs = {"period": enum("P4", "P3", "raw"), "tables": enum("vr5", "all", role="fidelity"),
                 "events": integer(), "background": Parameter("integer", role="scenario", choices=(0, 1)),
                 "seed": seed}
        if "response_config" in names:
            specs["response_config"] = Parameter("crv_response_config", role="fixed_configuration")
        for key in names:
            if key in specs:
                continue
            if key.startswith(("speed_", "atten_")) or key in ("time_constant", "pe_stretch"):
                specs[key] = number(0, positive=True, optional=True, role="calibration",
                                    unit="ns" if key == "time_constant" else "1")
            elif key.startswith("yield_"):
                specs[key] = number(0, optional=True, role="calibration", unit="photons/MeV")
            elif key in ("cross_talk", "min_voltage", "wf_noise", "background_scale", "photon_yield_scale"):
                specs[key] = number(0, 1 if key == "cross_talk" else None, optional=True, role="calibration",
                                    unit="V" if key in ("min_voltage", "wf_noise") else "1")
    else:
        raise ValueError(f"fastsimkit: task {task!r} has no parameter schema")
    if set(specs) != set(names):
        raise ValueError(f"fastsimkit: parameter schema does not match task {task!r}")
    return specs


def validate(task: str, params: dict, names, defaults=None) -> dict:
    """Fill defaults and enforce the same task domain before either endpoint dispatch."""
    if not isinstance(params, dict) or any(not isinstance(k, str) for k in params):
        raise ValueError("fastsimkit: params must be an object with string keys")
    defaults = defaults or {}
    extra = sorted(set(params) - set(names) - {"task"})
    missing = sorted(set(names) - set(params) - set(defaults))
    if extra or missing:
        raise ValueError(f"fastsimkit: task {task!r} takes {list(names)}; extra {extra}, missing {missing}")
    if "task" in params and params["task"] != task:
        raise ValueError(f"fastsimkit: expected task {task!r}")
    specs = schema(task, names)
    out = {key: specs[key].parse(task, key, params.get(key, defaults.get(key))) for key in names}
    if task == "veto_c4":
        for lo, hi in (("dt_lo", "dt_hi"), ("pmin", "pmax")):
            if out[lo] >= out[hi]:
                raise ValueError(f"fastsimkit: {task} requires {lo} < {hi}")
        if out["gate_start_lo"] >= 0 and out["gate_end_hi"] >= 0 and out["gate_start_lo"] >= out["gate_end_hi"]:
            raise ValueError("fastsimkit: veto_c4 requires enabled gate_start_lo < gate_end_hi")
    if task == "crv_response":
        if out["tables"] == "all" and out["events"] == 0:
            raise ValueError("fastsimkit: crv_response events must be > 0 with tables=all")
        if "response_config" in out:
            from crvfastsim import response_config
            from crvfastsim.kit_task import KNOB_PARAMS
            response_config.apply_knobs(response_config.parse(out["response_config"], kit=True),
                                        {k: out[k] for k in KNOB_PARAMS})
    return out
