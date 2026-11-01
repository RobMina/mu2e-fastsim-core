"""Device float precision (mu2efastsim.precision): float64 mode is jax.jit itself; float32 mode, run in a subprocess
because the mode is read at import, casts arguments and float defaults but keeps integers 64-bit."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
jax = pytest.importorskip("jax")
from mu2efastsim import precision as dprec                              # noqa: E402


def test_float64_mode_is_plain_jax_jit():
    if dprec.SINGLE:
        pytest.skip("suite runs in float32 mode")
    f = dprec.jit(lambda x: x * 2)
    assert type(f) is type(jax.jit(lambda x: x))
    assert dprec.TINY == 1e-300 and dprec.dev(np.ones(2)).dtype == np.float64
    assert dprec.host(np.ones(2, np.float32)).dtype == np.float32


FLOAT32_CHECK = r'''
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from mu2efastsim import precision as dprec, eloss_jax, materials
assert dprec.SINGLE and dprec.FLOAT == np.float32 and 0 < dprec.TINY < 1e-37
assert jnp.zeros(2).dtype == np.float64 and jax.random.uniform(jax.random.key(0), (2,)).dtype == np.float64  # outside kernels

@dprec.jit
def k(x, i):
    z, u = jnp.zeros(3), jax.random.uniform(jax.random.key(0), (3,))
    assert z.dtype == np.float32 and u.dtype == np.float32 and jnp.zeros(2, bool).dtype == bool
    assert jnp.arange(3).dtype == np.int64 and jnp.full(2, 2**62, jnp.int64).dtype == np.int64
    assert jnp.zeros(2, jnp.float64).dtype == np.float64 and jnp.asarray(np.arange(2)).dtype == np.int64
    assert jax.random.normal(jax.random.key(0), (4,), jnp.float64).dtype == np.float64
    return x * 2.0 + z, i + 1, eloss_jax.dedx_restricted(materials.POLYSTYRENE, x + 100.0, 1e9)

y, i, d = k(np.ones(3), np.array([2**40], np.int64))
assert y.dtype == np.float32 and i.dtype == np.int64 and int(i[0]) == 2**40 + 1 and d.dtype == np.float32
assert not dprec.scoped()
assert jnp.zeros(2).dtype == np.float64                    # scope ends with the kernel
d64 = eloss_jax.dedx_restricted(materials.POLYSTYRENE, np.array([101.0, 500.0]), 1e9)
s64 = eloss_jax.sample_loss(jax.random.key(1), materials.POLYSTYRENE, np.array([101.0, 500.0]), np.array([10.0, 10.0]), 1e9)
assert d64.dtype == np.float64 and s64.dtype == np.float64  # shared kernels: float64 callers stay float64
np.save(OUT, np.concatenate([np.asarray(d64), np.asarray(s64)]))
assert dprec.dev({"a": np.ones(2), "b": np.arange(2)})["a"].dtype == np.float32
h = dprec.host({"a": jnp.ones(2, jnp.float32), "b": jnp.arange(2)})
assert type(h["a"]) is np.ndarray and h["a"].dtype == np.float64 and h["b"].dtype == np.int64
print("ok")
'''


def test_float32_mode_scoped_to_kernels_shared_kernels_follow_inputs(tmp_path):
    out = {}
    for mode in ("float32", "float64"):
        env = dict(os.environ, MU2EFS_DEVICE_FLOAT=mode, PYTHONPATH=str(ROOT))
        code = f"OUT = {str(tmp_path / mode)!r}\n" + (FLOAT32_CHECK if mode == "float32" else
                                                   FLOAT32_CHECK.split("assert dprec.SINGLE")[0] + FLOAT64_PART)
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
        out[mode] = np.load(tmp_path / f"{mode}.npy")
    assert np.array_equal(out["float32"], out["float64"])     # float64 callers: bit-identical in both modes


FLOAT64_PART = r'''
d64 = eloss_jax.dedx_restricted(materials.POLYSTYRENE, np.array([101.0, 500.0]), 1e9)
s64 = eloss_jax.sample_loss(jax.random.key(1), materials.POLYSTYRENE, np.array([101.0, 500.0]), np.array([10.0, 10.0]), 1e9)
np.save(OUT, np.concatenate([np.asarray(d64), np.asarray(s64)]))
print("ok")
'''


def test_default_is_float32_and_provenance_records_it():
    env = {k: v for k, v in os.environ.items() if k != "MU2EFS_DEVICE_FLOAT"}
    env["PYTHONPATH"] = str(ROOT)
    code = ("from mu2efastsim import precision as p; d = p.provenance(); "
            "assert p.SINGLE and d['device_float'] == 'float32' and d['device_kind'], d; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def test_suite_runs_in_float64_and_host_crossings_stay_float64():
    from mu2efastsim import crossings
    assert dprec.NAME == os.environ.get("MU2EFS_DEVICE_FLOAT") == "float64" or dprec.SINGLE
    t_in, _ = crossings.slab_crossings(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), np.array([[-1.0, -1.0, 5.0]]),
                                       np.array([[1.0, 1.0, 6.0]]), backend="jax")
    assert t_in.dtype == np.float64 and t_in[0, 0] == 5.0


def test_unknown_precision_is_refused():
    env = dict(os.environ, MU2EFS_DEVICE_FLOAT="float16", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", "import mu2efastsim.precision"], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode != 0 and "MU2EFS_DEVICE_FLOAT" in out.stderr
