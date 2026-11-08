"""Warm-rate timer: run a workflow entry point for three passes in one process and report setup and warm events/s.

    python3 -m mu2efastsim.bench MODULE:FUNC --events N [--prepare MODULE:FUNC] [--kwargs JSON] [--dtype float32]
                                 [--workflow W2] [--boundary "stops -> reco CE track"] [--output-included]
                                 [--fresh-inputs [--seed S]] [--passes 3] [--out bench.json]

By default every pass calls the entry point on the same input (``FUNC(n_events, **kwargs)``, or
``FUNC(prepared, **kwargs)`` when ``--prepare`` loads the input into host memory first; the prepare time is reported
apart and is not setup). With ``--fresh-inputs`` the prepare hook is called before every pass, outside the clock, as
``PREPARE(n_events, seed=S + pass_index)``, so each pass sees new events, as a production job does; a prepare hook
without a ``seed`` keyword is refused (never a silent fallback to fixed inputs).
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
  inputs        "fixed" (the default) or "fresh", with the seed of each pass
  compile_per_pass  JAX compiles per pass from jax.monitoring (count and seconds), when JAX is loaded

The multiplier against stock is warm_rate / (stock events/s per single-core grid slot at the same boundary); the
stock rates and the scorecard live outside this package. Standard library only; JAX and CuPy are used if loaded.
"""
import argparse
import importlib
import inspect
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


# JAX compile accounting. One process-wide listener is registered (jax.monitoring has no per-call scope); it adds to
# the counter of the pass that is running, if any. backend_compile_duration fires for every compile request that
# misses JAX's in-memory cache (jit or eager op), including the ones served by the persistent compilation cache,
# which are counted again under persistent_cache_hits.
COMPILE_EVENTS = {"/jax/core/compile/backend_compile_duration": "backend_compile",
                  "/jax/core/compile/jaxpr_trace_duration": "trace",
                  "/jax/core/compile/jaxpr_to_mlir_module_duration": "lowering"}
_compile_counter = [None]
_compile_listener = [False]


def _on_duration(event, duration, *args, **kwargs):
    counter, key = _compile_counter[0], COMPILE_EVENTS.get(event)
    if counter is not None and key:
        counter[key + "s"] += 1
        counter[key + "_s"] += float(duration)


def _on_event(event, *args, **kwargs):
    counter = _compile_counter[0]
    if counter is not None and event == "/jax/compilation_cache/cache_hits":
        counter["persistent_cache_hits"] += 1


def compile_listener():
    """Register the compile listener once, if JAX is loaded; True when compiles are being counted."""
    if _compile_listener[0]:
        return True
    monitoring = getattr(sys.modules.get("jax"), "monitoring", None)
    if monitoring is None:
        return False
    try:
        monitoring.register_event_duration_secs_listener(_on_duration)
        monitoring.register_event_listener(_on_event)
    except Exception:
        return False
    _compile_listener[0] = True
    return True


def new_compile_counter():
    counter = {"persistent_cache_hits": 0}
    for key in COMPILE_EVENTS.values():
        counter[key + "s"] = 0
        counter[key + "_s"] = 0.0
    return counter


def accepts_seed(func):
    """True if ``func`` can be called with a ``seed=`` keyword."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is p.VAR_KEYWORD or (p.name == "seed" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
               for p in params)


class FreshInputError(ValueError):
    """The workflow cannot draw new inputs for every pass."""


def fresh_inputs(prepare, events, seed=1, spec=None):
    """``inputs`` for :func:`time_passes`: pass ``i`` gets ``(prepare(events, seed=seed + i),)``.

    Raises :class:`FreshInputError` if ``prepare`` takes no ``seed`` keyword. The returned callable raises it too if
    two consecutive seeds give back the very same object (a hook that caches its input instead of drawing anew).
    """
    name = spec or getattr(prepare, "__qualname__", repr(prepare))
    if not accepts_seed(prepare):
        raise FreshInputError(
            f"--fresh-inputs: the prepare hook {name} takes no seed keyword, so it cannot draw new inputs for each "
            f"pass. This workflow cannot be timed with fresh inputs until its prepare hook accepts seed=...; "
            f"refusing to time fixed inputs as fresh ones.")
    last = [None]

    def inputs(index):
        value = prepare(events, seed=seed + index)
        if index and value is last[0]:
            raise FreshInputError(f"--fresh-inputs: the prepare hook {name} returned the same object for seeds "
                                  f"{seed + index - 1} and {seed + index}; it must draw new inputs for each seed.")
        last[0] = value
        return (value,)
    return inputs


def measure_passes(func, events, passes=3, args=(), kwargs=None, sync=device_sync, clock=time.perf_counter,
                   inputs=None):
    """Run the passes; return ``{"pass_s", "prepare_pass_s", "compile_per_pass"}``.

    ``inputs(pass_index) -> args`` (optional) is called before each pass, outside the clock, and replaces ``args``.
    ``compile_per_pass`` holds one counter per pass, or None for a pass that began before JAX was loaded.
    """
    if passes < 2:
        raise ValueError("need at least two passes (one setup pass, one warm pass)")
    if events <= 0:
        raise ValueError("events must be positive")
    kwargs = kwargs or {}
    times, prepare_times, compiles = [], [], []
    for index in range(passes):
        if inputs is not None:
            begin = time.perf_counter()
            args = inputs(index)
            prepare_times.append(time.perf_counter() - begin)
        counter = new_compile_counter() if compile_listener() else None
        _compile_counter[0] = counter
        try:
            begin = clock()
            result = func(*args, **kwargs)
            sync(result)
            times.append(clock() - begin)
        finally:
            _compile_counter[0] = None
        compiles.append(counter)
        del result
    return {"pass_s": times, "prepare_pass_s": prepare_times if inputs is not None else None,
            "compile_per_pass": compiles}


def time_passes(func, events, passes=3, args=(), kwargs=None, sync=device_sync, clock=time.perf_counter, inputs=None):
    """Wall seconds of ``passes`` calls of ``func(*args, **kwargs)``, each stopped after ``sync(result)``."""
    return measure_passes(func, events, passes, args, kwargs, sync, clock, inputs)["pass_s"]


def summarise(times, events):
    """Setup seconds, warm events/s and N10 from per-pass wall times."""
    warm = times[1:]
    warm_mean = sum(warm) / len(warm)
    rate = events * len(warm) / sum(warm) if sum(warm) > 0 else float("inf")
    setup = max(times[0] - warm_mean, 0.0)
    return {"pass_s": times, "warm_pass_s": warm_mean, "warm_rate": rate, "setup_s": setup,
            "n10": 9.0 * setup * rate}


def run(func, events, passes=3, args=(), kwargs=None, dtype=None, workflow=None, boundary=None,
        output_included=False, prepare_s=None, sync=device_sync, clock=time.perf_counter, label=None,
        inputs=None, seeds=None):
    """Time ``func`` and return the report dict (see the module doc).

    ``inputs`` (e.g. from :func:`fresh_inputs`) draws the arguments of every pass; ``seeds`` are recorded with it.
    """
    measured = measure_passes(func, events, passes, args, kwargs, sync, clock, inputs)
    report = {"workflow": workflow, "boundary": boundary, "events": events, "passes": passes,
              "inputs": "fresh" if inputs is not None else "fixed", "seeds": list(seeds) if seeds else None}
    report.update(summarise(measured["pass_s"], events))
    if measured["prepare_pass_s"] is not None:
        prepare_s = (prepare_s or 0.0) + sum(measured["prepare_pass_s"])
    counted = [c for c in measured["compile_per_pass"] if c is not None]
    report.update({
        "prepare_pass_s": measured["prepare_pass_s"],
        "compile_source": "jax.monitoring" if counted else None,
        "compile_per_pass": measured["compile_per_pass"] if counted else None,
    })
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
    seeds = r.get("seeds")
    text = (f"{r['workflow'] or 'workflow'}: {r['events']} events x {r['passes']} passes on {r['device']} ({r['dtype']}), "
            f"{r.get('inputs', 'fixed')} inputs" + (f" (seeds {', '.join(map(str, seeds))})" if seeds else "") + "\n"
            f"  pass s: {', '.join(f'{t:.3f}' for t in r['pass_s'])}\n"
            f"  warm {r['warm_rate']:.4g} events/s, setup {r['setup_s']:.3g} s, N10 {r['n10']:.3g} events"
            + (f", prepare {r['prepare_s']:.3g} s" if r.get("prepare_s") is not None else ""))
    if r.get("compile_per_pass"):
        text += "\n  JAX compiles per pass: " + ", ".join(
            "n/a" if c is None else f"{c['backend_compiles']} ({c['backend_compile_s']:.2f} s)"
            for c in r["compile_per_pass"])
    return text


def main(argv=None):
    p = argparse.ArgumentParser(prog="python3 -m mu2efastsim.bench",
                                description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("entry", help="MODULE:FUNC, called each pass as FUNC(N or prepared input, **kwargs)")
    p.add_argument("--events", type=int, required=True, help="events processed per pass")
    p.add_argument("--prepare", help="MODULE:FUNC called once as FUNC(N) to load the input into memory (reported as prepare_s, not setup)")
    p.add_argument("--fresh-inputs", action="store_true",
                   help="new inputs every pass: call the prepare hook before each pass, outside the clock, as "
                        "FUNC(N, seed=SEED + pass index); refused if the hook takes no seed")
    p.add_argument("--seed", type=int, default=1, help="first seed of --fresh-inputs (default 1)")
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
    if a.fresh_inputs and not a.prepare:
        p.error("--fresh-inputs needs --prepare: the prepare hook draws the inputs of every pass")
    func = resolve(a.entry)
    prepare_s = None
    arg = a.events
    inputs = seeds = None
    if a.fresh_inputs:
        try:
            inputs = fresh_inputs(resolve(a.prepare), a.events, a.seed, a.prepare)
        except FreshInputError as e:
            print(f"bench: error: {e}", file=sys.stderr)
            return 2
        seeds = [a.seed + i for i in range(a.passes)]
    elif a.prepare:
        begin = time.perf_counter()
        arg = resolve(a.prepare)(a.events)
        prepare_s = time.perf_counter() - begin
    try:
        report = run(func, a.events, a.passes, (arg,), kwargs, a.dtype, a.workflow, a.boundary, a.output_included,
                     prepare_s, label=a.device_label, inputs=inputs, seeds=seeds)
    except FreshInputError as e:
        print(f"bench: error: {e}", file=sys.stderr)
        return 2
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
