"""Floating-point precision of the device (JAX) kernels: MU2EFS_DEVICE_FLOAT=float32 (default) or float64.

x64 stays enabled in both modes, so integers, keys, indices and sentinels are 64-bit; only floating arrays change.
Host NumPy code is float64 in both modes. Device modules use FLOAT for every floating dtype, TINY for the smallest
positive uniform draw and divide guards, dev() for host arrays they move to the device, host() for device results they
hand to host code, and jit in place of jax.jit: in float64 mode it is jax.jit itself, in float32 mode it also casts
float64 array arguments. In float32 mode, JAX array creators and samplers called without a dtype give float32 while
a dprec.jit kernel runs (traces) and when called from a module that declared itself precision-aware with
register(__name__); everywhere else (other stages, host code) they are unchanged and float64. Kernels shared with
such code compute in the dtype of their inputs (float_of, tiny_of).

Mechanism: in float32 mode the jnp creators (zeros, ones, empty, full, eye, identity, linspace, array, asarray,
arange) and jax.random samplers are replaced process-wide by wrappers, but a wrapper acts only inside a dprec.jit
call or for a registered caller; elsewhere it passes the call through unchanged (a thread-local read and a frame
lookup, no argument binding). Measured pass-through overhead per eager call: 0.2-0.5 us (jnp.zeros(3) 16.9 vs 16.7 us,
jnp.asarray 6.9 vs 6.4 us, jax.random.uniform 7.4 vs 6.9 us); calls inside compiled kernels cost nothing at run
time. Nothing is replaced in float64 mode.

float64 output is identical on every accelerator model for a given seed; float32 output repeats on one model but its
discrete results (pulse and cluster counts) can differ slightly between models, so provenance() records the precision
together with the device model. The test suites pin float64 (mu2efastsim.testing) because they check exact
device-versus-NumPy parity.
"""
import functools
import os
import sys
import threading

import numpy as np

DEFAULT = "float32"
NAME = os.environ.get("MU2EFS_DEVICE_FLOAT", DEFAULT) or DEFAULT
if NAME not in ("float64", "float32"):
    raise ValueError(f"MU2EFS_DEVICE_FLOAT must be float64 or float32, not {NAME!r}")
FLOAT = np.dtype(NAME)
TINY = 1e-300 if FLOAT == np.float64 else float(np.finfo(FLOAT).tiny)   # guard / smallest draw: as shipped in float64
SINGLE = FLOAT == np.float32


_SCOPE = threading.local()      # depth of precision-aware (dprec.jit) calls running in this thread
_AWARE = set()                  # modules whose own JAX creator/sampler calls follow the device float


def register(module_name):
    """Declare a module precision-aware: its jnp creators and jax.random samplers called without a dtype give the
    device float (eager calls included, e.g. tables built on the host and closed over by its kernels)."""
    _AWARE.add(module_name)


def scoped():
    """True while a float32-mode dprec.jit kernel runs or traces in this thread."""
    return getattr(_SCOPE, "depth", 0) > 0


def float_of(*xs):
    """Floating dtype a shared kernel computes in: the device float inside a dprec.jit kernel; elsewhere float32 when
    any input is float32, else float64 (callers outside the precision setting keep float64)."""
    if scoped():
        return FLOAT
    return np.dtype(np.float32) if any(getattr(x, "dtype", None) == np.float32 for x in xs) else np.dtype(np.float64)


def tiny_of(dtype):
    """Smallest positive draw / divide guard for dtype (1e-300 in float64, as shipped)."""
    return 1e-300 if np.dtype(dtype) == np.float64 else float(np.finfo(dtype).tiny)


def _cast(x):
    if getattr(x, "dtype", None) == np.float64:
        return x.astype(FLOAT)
    return x


def host(tree):
    """Device arrays (or a pytree of them) as NumPy, floats widened to float64 in float32 mode: host code is float64
    in both modes, and mixed float32/float64 NumPy (ufunc.at, fancy-index assignment) takes slow casting paths."""
    import jax
    def one(x):
        a = np.asarray(x)
        return a.astype(np.float64) if SINGLE and a.dtype == np.float32 else a
    return jax.tree_util.tree_map(one, tree)


def provenance():
    """Precision and accelerator of this process, for output metadata: device float, JAX backend, device models."""
    out = {"device_float": NAME}
    try:
        import jax
        devices = jax.devices()
        out.update(jax_backend=jax.default_backend(), device_kind=sorted({d.device_kind for d in devices}),
                   platform_version=getattr(devices[0].client, "platform_version", None) if devices else None)
    except Exception as e:      # JAX absent or no usable backend: precision alone
        out["jax_backend"] = f"unavailable: {type(e).__name__}"
    return out


def dev(tree):
    """Floating NumPy/JAX arrays (or a pytree of them) in the device float; other leaves unchanged."""
    if not SINGLE:
        return tree
    import jax
    return jax.tree_util.tree_map(_cast, tree)


AUDIT = os.environ.get("MU2EFS_F64_AUDIT", "") == "1"
AUDIT_DIR = os.environ.get("MU2EFS_F64_AUDIT_DIR", "")   # audit: write the lowered programs that hold float64
F64_OPS = {}   # audit: kernel name -> number of float64 tensors in its lowered program (first signature)


def _audit(name, jitted, args, kwargs):
    if name in F64_OPS:
        return
    try:
        text = jitted.lower(*args, **kwargs).as_text(debug_info=bool(AUDIT_DIR))
        F64_OPS[name] = text.count("xf64>")
        if AUDIT_DIR and F64_OPS[name]:
            os.makedirs(AUDIT_DIR, exist_ok=True)
            with open(os.path.join(AUDIT_DIR, name.replace("<", "").replace(">", "") + ".mlir"), "w") as f:
                f.write(text)
    except Exception as e:      # lowering can need static arguments the call already bound
        F64_OPS[name] = f"not lowered: {type(e).__name__}"


def report():
    """Audit census: kernels whose lowered program still holds float64 tensors."""
    return {k: v for k, v in sorted(F64_OPS.items()) if v}


def _jit32(fun=None, **kw):
    import jax
    if fun is None:
        return functools.partial(_jit32, **kw)
    jitted = jax.jit(fun, **kw)
    name = f"{getattr(fun, '__module__', '?')}.{getattr(fun, '__qualname__', getattr(fun, '__name__', '?'))}"

    def scope(f, *args, **kwargs):
        _SCOPE.depth = getattr(_SCOPE, "depth", 0) + 1
        try:
            return f(*dev(args), **dev(kwargs))
        finally:
            _SCOPE.depth -= 1

    @functools.wraps(fun)
    def call(*args, **kwargs):
        if AUDIT:
            scope(lambda *a, **k: _audit(name, jitted, a, k), *args, **kwargs)
        return scope(jitted, *args, **kwargs)

    call.lower = lambda *args, **kwargs: scope(jitted.lower, *args, **kwargs)
    call.jitted = jitted
    return call


_CREATORS = ("zeros", "ones", "empty", "full", "eye", "identity", "linspace", "array", "asarray", "arange")
_SAMPLERS = ("uniform", "normal", "exponential", "gamma", "truncated_normal", "gumbel", "binomial", "beta")


def _float_default(fn, sampler):
    import inspect
    params = list(inspect.signature(fn).parameters)
    pos = params.index("dtype") if "dtype" in params else len(params)   # positional slot of dtype

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # pass-through first (the common case: code outside precision-aware kernels and modules), no binding
        if not getattr(_SCOPE, "depth", 0) and (not _AWARE or sys._getframe(1).f_globals.get("__name__") not in _AWARE):
            return fn(*args, **kwargs)
        if kwargs.get("dtype") is not None or (len(args) > pos and args[pos] is not None):
            return fn(*args, **kwargs)
        if sampler:                          # draw in the device float, never draw float64 and round
            return fn(*args, dtype=FLOAT, **kwargs)
        out = fn(*args, **kwargs)            # creators: integer results stay integer
        return out.astype(FLOAT) if getattr(out, "dtype", None) == np.float64 else out
    return wrapped


def _install_float_defaults():
    """float32 mode: inside dprec.jit kernels and in registered modules, JAX array creators and samplers called without
    a dtype give float32 where they would give float64 under x64 (integer and explicit dtypes unchanged); elsewhere
    they are unchanged."""
    import jax
    import jax.numpy as jnp
    for mod, names, sampler in ((jnp, _CREATORS, False), (jax.random, _SAMPLERS, True)):
        for n in names:
            fn = getattr(mod, n, None)
            if fn is not None and not getattr(fn, "_dprec", False):
                w = _float_default(fn, sampler)
                w._dprec = True
                setattr(mod, n, w)


if SINGLE:
    _install_float_defaults()

if AUDIT:
    import atexit
    import json
    atexit.register(lambda: print("F64_AUDIT", json.dumps(report()), flush=True))


def jit(fun=None, **kw):
    """jax.jit in float64 mode; in float32 mode a jit that casts float64 array arguments to float32."""
    if SINGLE:
        return _jit32(fun, **kw)
    import jax
    return jax.jit(fun, **kw) if fun is not None else functools.partial(jax.jit, **kw)
