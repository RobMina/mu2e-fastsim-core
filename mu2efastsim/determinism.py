"""Run-to-run determinism of the GPU (XLA) kernels: MU2EFS_DEVICE_DETERMINISTIC=1 (default) or 0.

XLA autotunes each fused GPU kernel when it is compiled: it times candidate code generators and tile shapes and keeps
the fastest. On a shared or busy GPU the timings, and so the choices, change from process to process; different
choices sum in a different order, and float results then differ in the last bits (in float32 this can move a discrete
outcome, e.g. a threshold crossing). With the setting on, `--xla_gpu_autotune_level=0` is appended to XLA_FLAGS: XLA
uses its default kernel configuration for the device instead of timing candidates, so the same program gives the
same result in every process on one GPU model. It has no effect on CPU backends. Turning autotuning off is enough
because the fast-simulation kernels use no floating-point scatter-add (atomics); `--xla_gpu_deterministic_ops=true`
would also cover those, at a large cost in the integer scatters of the SiPM kernel (see the CRV device-precision doc).

XLA reads XLA_FLAGS when JAX creates its first backend, so `apply()` runs when the `mu2efastsim` package is imported;
every device module of the fast simulation imports it before using JAX. A value the user already put in XLA_FLAGS for
the same flag is left alone. `state()` (recorded by `precision.provenance()`) says whether the flag is set and whether
a JAX backend existed already when it was appended (then it took no effect in this process). Standard library only.
"""
import os
import sys

FLAG = "--xla_gpu_autotune_level"
VALUE = "0"
_STATE = {}


def requested():
    """The setting: MU2EFS_DEVICE_DETERMINISTIC, on unless "0" (or "false"/"off"/"no")."""
    value = os.environ.get("MU2EFS_DEVICE_DETERMINISTIC", "1").strip().lower()
    if value not in ("1", "0", "true", "false", "on", "off", "yes", "no", ""):
        raise ValueError(f"MU2EFS_DEVICE_DETERMINISTIC must be 1 or 0, not {value!r}")
    return value not in ("0", "false", "off", "no")


def _backend_started():
    xb = sys.modules.get("jax._src.xla_bridge")
    check = getattr(xb, "backends_are_initialized", None)
    try:
        return bool(check()) if check else False
    except Exception:
        return False


def apply():
    """Append the XLA determinism flag to XLA_FLAGS when the setting is on (once per process)."""
    if _STATE:
        return dict(_STATE)
    flags = os.environ.get("XLA_FLAGS", "")
    user = FLAG in flags
    on = requested()
    if on and not user:
        os.environ["XLA_FLAGS"] = (flags + " " if flags.strip() else "") + FLAG + "=" + VALUE
    _STATE.update(requested=on, flag_source="user" if user else ("setting" if on else "none"),
                  late=bool(on and not user and _backend_started()))
    return dict(_STATE)


def state():
    """Determinism record for provenance: the setting, where the flag came from, the flag value XLA sees now, and
    `late` (appended after a JAX backend already existed, so not in effect for this process)."""
    s = dict(_STATE) if _STATE else {"requested": requested(), "flag_source": "not applied", "late": False}
    value = None
    for tok in os.environ.get("XLA_FLAGS", "").split():
        if tok.startswith(FLAG):
            value = tok.partition("=")[2]
    s["xla_gpu_autotune_level"] = value
    return s
