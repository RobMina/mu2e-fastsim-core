"""Warm-rate timer: run a workflow entry point for three passes in one process and report setup and warm events/s.

    python3 -m mu2efastsim.bench MODULE:FUNC --events N [--prepare MODULE:FUNC] [--kwargs JSON] [--dtype float32]
                                 [--workflow W2] [--boundary "stops -> reco CE track"] [--output-included]
                                 [--passes 3] [--out bench.json]

Every pass calls the entry point on the same input (``FUNC(n_events, **kwargs)``, or ``FUNC(prepared, **kwargs)``
when ``--prepare`` loads the input into host memory first; the prepare time is reported apart and is not setup).
The clock stops only after the device is synchronised: ``jax.block_until_ready`` on what the entry point returns,
then a CuPy device synchronise if CuPy is loaded. Return the device outputs from the entry point, or the timer
cannot wait for them.

Reported (and written as JSON):
  pass_s        wall seconds of each pass
  warm_rate     events / s over passes 2..P (steady state, after compile)
  setup_s       pass 1 minus the mean warm pass: compile + table/library load paid once per process
  n10           9 x setup_s x warm_rate, the job size (events) at which setup is 10 % of the wall time
  device        the job node's label: nvidia-smi -L (with the MIG profile) on a GPU, the CPU model otherwise
  dtype         as given, else the JAX default float (x64 flag), else float64 (NumPy)

The multiplier against stock is warm_rate / (stock events/s per single-core grid slot at the same boundary); the
stock rates and the scorecard live outside this package. Standard library only; JAX and CuPy are used if loaded.
"""
import argparse
import importlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def resolve(spec):
    """``package.module:function`` -> the callable."""
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"expected MODULE:FUNC, got {spec!r}")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj


def device_sync(result=None):
    """Wait until the device work behind ``result`` has finished (no-op for host results)."""
    jax = sys.modules.get("jax")
    if jax is not None:
        if hasattr(jax, "block_until_ready"):
            jax.block_until_ready(result)
        else:
            for leaf in jax.tree_util.tree_leaves(result):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()
    elif hasattr(result, "block_until_ready"):
        result.block_until_ready()
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        cupy.cuda.Device().synchronize()
    return result


def backend():
    """``gpu`` when a loaded JAX or CuPy runs on a GPU, else ``cpu``."""
    jax = sys.modules.get("jax")
    if jax is not None:
        try:
            return "gpu" if jax.default_backend() in ("gpu", "cuda", "rocm") else "cpu"
        except Exception:
            return "cpu"
    return "gpu" if "cupy" in sys.modules else "cpu"


def parse_nvidia_smi(text, visible=None):
    """Label of the device in use from ``nvidia-smi -L`` output.

    ``visible`` is CUDA_VISIBLE_DEVICES: a MIG UUID selects that slice, a GPU UUID or index selects that card;
    otherwise the first listed device (inside a Slurm GPU job only the allocated device is listed).
    """
    gpus = []                                   # [(index, name, uuid, [(profile, uuid), ...])]
    for line in text.splitlines():
        m = re.match(r"\s*GPU (\d+): (.+?) \(UUID: (\S+)\)", line)
        if m:
            gpus.append((m.group(1), m.group(2).strip(), m.group(3), []))
            continue
        m = re.match(r"\s*MIG (\S+)\s+Device\s+\d+: \(UUID: (\S+)\)", line)
        if m and gpus:
            gpus[-1][3].append((m.group(1), m.group(2)))
    if not gpus:
        return None
    first = (visible or "").split(",")[0].strip()
    for index, name, uuid, migs in gpus:
        for profile, mig_uuid in migs:
            if first and first == mig_uuid:
                return f"{name}, MIG {profile}"
    for index, name, uuid, migs in gpus:
        if first in (index, uuid):
            return f"{name}, MIG {migs[0][0]}" if migs else name
    index, name, uuid, migs = gpus[0]
    return f"{name}, MIG {migs[0][0]}" if migs else name


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def cpu_threads():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def device_label(kind=None):
    """The job node's device label: the GPU (and MIG profile) for ``gpu``, the CPU model for ``cpu``."""
    kind = kind or backend()
    if kind == "gpu":
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
            label = parse_nvidia_smi(out.stdout, os.environ.get("CUDA_VISIBLE_DEVICES"))
            if label:
                return label
        except (OSError, subprocess.SubprocessError):
            pass
        return "GPU (nvidia-smi unavailable)"
    return f"{cpu_model()} ({cpu_threads()} threads available)"


def default_dtype():
    jax = sys.modules.get("jax")
    if jax is not None:
        try:
            return "float64" if jax.config.jax_enable_x64 else "float32"
        except AttributeError:
            pass
    return "float64"


def time_passes(func, events, passes=3, args=(), kwargs=None, sync=device_sync, clock=time.perf_counter):
    """Wall seconds of ``passes`` calls of ``func(*args, **kwargs)``, each stopped after ``sync(result)``."""
    if passes < 2:
        raise ValueError("need at least two passes (one setup pass, one warm pass)")
    if events <= 0:
        raise ValueError("events must be positive")
    kwargs = kwargs or {}
    times = []
    for _ in range(passes):
        begin = clock()
        result = func(*args, **kwargs)
        sync(result)
        times.append(clock() - begin)
        del result
    return times


def summarise(times, events):
    """Setup seconds, warm events/s and N10 from per-pass wall times."""
    warm = times[1:]
    warm_mean = sum(warm) / len(warm)
    rate = events * len(warm) / sum(warm) if sum(warm) > 0 else float("inf")
    setup = max(times[0] - warm_mean, 0.0)
    return {"pass_s": times, "warm_pass_s": warm_mean, "warm_rate": rate, "setup_s": setup,
            "n10": 9.0 * setup * rate}


def run(func, events, passes=3, args=(), kwargs=None, dtype=None, workflow=None, boundary=None,
        output_included=False, prepare_s=None, sync=device_sync, clock=time.perf_counter, label=None):
    """Time ``func`` and return the report dict (see the module doc)."""
    times = time_passes(func, events, passes, args, kwargs, sync, clock)
    report = {"workflow": workflow, "boundary": boundary, "events": events, "passes": passes}
    report.update(summarise(times, events))
    kind = backend()
    report.update({
        "backend": kind,
        "device": label or device_label(kind),
        "dtype": dtype or default_dtype(),
        "dtype_source": "argument" if dtype else ("jax_x64_flag" if "jax" in sys.modules else "numpy_default"),
        "output_included": bool(output_included),
        "prepare_s": prepare_s,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "python": platform.python_version(),
        "jax": getattr(sys.modules.get("jax"), "__version__", None),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    return report


def format_report(r):
    return (f"{r['workflow'] or 'workflow'}: {r['events']} events x {r['passes']} passes on {r['device']} ({r['dtype']})\n"
            f"  pass s: {', '.join(f'{t:.3f}' for t in r['pass_s'])}\n"
            f"  warm {r['warm_rate']:.4g} events/s, setup {r['setup_s']:.3g} s, N10 {r['n10']:.3g} events"
            + (f", prepare {r['prepare_s']:.3g} s" if r.get("prepare_s") is not None else ""))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python3 -m mu2efastsim.bench",
                                description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("entry", help="MODULE:FUNC, called each pass as FUNC(N or prepared input, **kwargs)")
    p.add_argument("--events", type=int, required=True, help="events processed per pass")
    p.add_argument("--prepare", help="MODULE:FUNC called once as FUNC(N) to load the input into memory (reported as prepare_s, not setup)")
    p.add_argument("--kwargs", default="{}", help="JSON object of keyword arguments for the entry point")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--dtype", help="float32 or float64 (default: the JAX x64 flag, else float64)")
    p.add_argument("--workflow", help="workflow name, e.g. W5")
    p.add_argument("--boundary", help='input -> output in one line, e.g. "stock TS states -> stops"')
    p.add_argument("--output-included", action="store_true", help="the entry point writes its output")
    p.add_argument("--device-label", help="override the detected device label")
    p.add_argument("--out", type=Path, help="write the report JSON here")
    a = p.parse_args(argv)
    kwargs = json.loads(a.kwargs)
    if not isinstance(kwargs, dict):
        p.error("--kwargs must be a JSON object")
    func = resolve(a.entry)
    prepare_s = None
    arg = a.events
    if a.prepare:
        begin = time.perf_counter()
        arg = resolve(a.prepare)(a.events)
        prepare_s = time.perf_counter() - begin
    report = run(func, a.events, a.passes, (arg,), kwargs, a.dtype, a.workflow, a.boundary, a.output_included,
                 prepare_s, label=a.device_label)
    report["entry"] = a.entry
    report["prepare"] = a.prepare
    report["kwargs"] = kwargs
    report["command"] = " ".join(sys.argv) if argv is None else " ".join(["python3 -m mu2efastsim.bench", *argv])
    print(format_report(report))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
