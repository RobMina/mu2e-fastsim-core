"""Vectorised ray-box crossings of straight tracks with axis-aligned boxes: JAX when available (CPU/GPU), else numpy.

Slab method. For N tracks and B boxes it computes the entry/exit parameters [N, B]; a crossing exists where
t_out > max(t_in, 0). Positions in mm. Both backends give identical results in float64 (tests/test_crossings.py).
Detector packages build the box bounds (CRV bars, tracker panel envelopes) from their own Offline-derived geometry.
"""
from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False


def _slab(xp, origin, direction, lo, hi):
    inv = 1.0 / xp.where(direction == 0.0, 1e-300, direction)
    t1 = (lo[None, :, :] - origin[:, None, :]) * inv[:, None, :]
    t2 = (hi[None, :, :] - origin[:, None, :]) * inv[:, None, :]
    return xp.max(xp.minimum(t1, t2), axis=-1), xp.min(xp.maximum(t1, t2), axis=-1)


if HAVE_JAX:
    _slab_jax = jax.jit(lambda o, d, lo, hi: _slab(jnp, o, d, lo, hi))


def slab_crossings(origin, direction, lo, hi, backend=None):
    """origin, direction: [N,3] (unit direction); lo, hi: [B,3]. Returns numpy t_in, t_out: [N,B] (mm)."""
    use_jax = HAVE_JAX if backend is None else backend == "jax"
    if use_jax:
        t_in, t_out = _slab_jax(jnp.asarray(origin), jnp.asarray(direction), jnp.asarray(lo), jnp.asarray(hi))
        return np.asarray(t_in), np.asarray(t_out)
    return _slab(np, np.asarray(origin), np.asarray(direction), np.asarray(lo), np.asarray(hi))


def crossings(origin, direction, lo, hi, chunk: int = 32768, backend=None):
    """Sparse crossings: rows (track index, box index, t_in, t_out, path length), chunked over tracks."""
    out = []
    for s in range(0, len(origin), chunk):
        t_in, t_out = slab_crossings(origin[s:s + chunk], direction[s:s + chunk], lo, hi, backend)
        t_in = np.maximum(t_in, 0.0)
        it, ib = np.nonzero(t_out > t_in)
        out.append(np.stack([it + s, ib, t_in[it, ib], t_out[it, ib], t_out[it, ib] - t_in[it, ib]], 1))
    return np.concatenate(out) if out else np.zeros((0, 5))
