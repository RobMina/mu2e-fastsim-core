"""slab_jax: raw-key threefry bit for bit jax.random; the raw-key sampler and the device slab loss equal to
eloss_jax._sample and a fold_in + eloss_jax._sample slice loop to rounding (the same draws; separately compiled XLA
programs round differently in the last bit, as eloss_jax._sample(unroll=True) does against unroll=False); the Pallas
kernel (interpreter) equal to the XLA implementation to rounding; agreement with NumPy transport.slab_loss in
distribution."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
jax = pytest.importorskip("jax")
import jax.numpy as jnp                                                   # noqa: E402
from jax import lax                                                       # noqa: E402

from mu2efastsim import eloss_jax as EJ, materials as M, slab_jax as S, transport as tr   # noqa: E402

MATS = (M.CONCRETE_MARS, M.POLYSTYRENE, M.MILD_STEEL, M.ALUMINIUM, M.AIR, M.LEAD)


def _keys(n, seed=7):
    return jax.vmap(lambda i: jax.random.fold_in(jax.random.key(seed), i))(jnp.arange(n, dtype=jnp.uint32))


def _eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _close(a, b, rtol=1e-11, exact=0.9, within=1.0):
    """Equal to rounding: a fraction `within` of the values within rtol (a last-bit difference can move one slice
    boundary of a lane, and then that lane's later draws), a fraction `exact` bit for bit."""
    a, b = np.asarray(a), np.asarray(b)
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
    return np.mean(rel <= rtol) >= within and np.mean(a == b) >= exact


def test_raw_threefry_bits():
    keys = _keys(4096)
    k = S.raw(keys)
    for d in (0, 1, 5, 123456):
        assert _eq(jax.random.key_data(jax.vmap(lambda kk: jax.random.fold_in(kk, d))(keys)),
                   jnp.stack(S.fold_in(k, d), -1))
    sp = jax.vmap(lambda kk: jax.random.key_data(jax.random.split(kk, 9)))(keys)
    for j in range(9):
        assert _eq(sp[:, j], jnp.stack(S.fold_in(k, j), -1))
    for dt in (np.float64, np.float32):
        tiny = float(np.finfo(dt).tiny)
        assert _eq(jax.vmap(lambda kk: jax.random.uniform(kk, (), dt))(keys), S.uniform(k, dt))
        assert _eq(jax.vmap(lambda kk: jax.random.uniform(kk, (), dt, minval=tiny))(keys), S.uniform(k, dt, 0, tiny))
        u3 = jax.vmap(lambda kk: jax.random.uniform(kk, (3,), dt))(keys)
        for q in range(3):
            assert _eq(u3[:, q], S.uniform(k, dt, q))
        assert _eq(jax.vmap(lambda kk: jax.random.normal(kk, (), dt))(keys), S.normal(k, dt))


def test_cephes_erf_pair():
    x = jnp.linspace(-12.0, 12.0, 200_001)
    for f, g in ((S.erf_cephes, lax.erf), (S.erfc_cephes, lax.erfc)):
        sel = jnp.abs(x) < 1 if f is S.erf_cephes else x >= 0
        a, b = f(x)[sel], g(x)[sel]
        assert float(jnp.max(jnp.abs(a - b) / jnp.maximum(jnp.abs(b), 1e-300))) < 1e-14
    a = S.ndtr(x, S.erf_cephes, S.erfc_cephes)
    b = S.ndtr(x)
    assert float(jnp.max(jnp.abs(a - b) / jnp.maximum(b, 1e-300))) < 1e-14
    assert _eq(b, jax.scipy.special.ndtr(x))
    u = jnp.concatenate([jnp.linspace(0.0, 1.0, 100_001), jnp.geomspace(1e-300, 0.2, 2001)])
    a, b = S.ndtri(u), jax.scipy.special.ndtri(u)
    fin = jnp.isfinite(b)
    assert _eq(jnp.isfinite(a), fin) and float(jnp.max(jnp.abs(a - b)[fin] / jnp.abs(b)[fin])) < 1e-15


def _inputs(n, seed=3):
    rng = np.random.default_rng(seed)
    mi = rng.integers(0, len(MATS), n)
    cs = [EJ.consts(m) for m in MATS]
    c = EJ.MatConsts(*(jnp.asarray(np.array([getattr(cc, f) for cc in cs])[mi]) for f in EJ.MatConsts._fields))
    p = jnp.asarray(np.geomspace(60.0, 2e4, n)[rng.permutation(n)])
    L = jnp.asarray(rng.choice([0.0, 0.5, 5.0, 50.0, 300.0, 2000.0], n) * rng.uniform(0.5, 1.5, n))
    return c, p, L


def test_sample_matches_eloss_jax():
    n = 20_000
    keys = _keys(n, 11)
    c, p, L = _inputs(n)
    ref = jax.jit(jax.vmap(lambda kk, cc, pp, ll: EJ._sample(kk, cc, pp, ll, 1e9, EJ.MMU)))(keys, c, p, L)
    raw = jax.jit(jax.vmap(lambda k0, k1, cc, pp, ll: S.sample((k0, k1), cc, pp, ll, 1e9)))(*S.raw(keys), c, p, L)
    assert _close(ref, raw, 1e-13, 0.99)


def _reference_slab(keys, c, p, L):
    """The tracker's device slab sampler (fold_in(key, slice) + eloss_jax._sample), written out independently."""
    def one(key, cc, pp, ll):
        tk0 = jnp.sqrt(pp * pp + EJ.MMU ** 2) - EJ.MMU

        def body(s):
            i, left, lost = s
            go = (i < tr.N_SLICES_MAX) & (left > 0) & (tk0 - lost > tr.TK_STOP)
            tk = tk0 - lost
            pg = jnp.sqrt(jnp.clip((tk + EJ.MMU) ** 2 - EJ.MMU ** 2, 1e-6, None))
            dedx = EJ._dedx(cc, pg, 1e9, EJ.MMU)
            dedx_s = jnp.where(dedx > 0, dedx, 1.0)
            step = jnp.where(tk > 3.0 * dedx * left, left / tr.MAX_SLICES_PER_SLAB, tr.SLICE_FRAC * tk / dedx_s)
            step = jnp.minimum(left, jnp.maximum(step, tr.SLICE_FRAC * tk / dedx_s))
            step = jnp.where(left <= step * 1.0001, left, step)
            dl = EJ._sample(jax.random.fold_in(key, i), cc, pg, jnp.where(go, step, 0.0), 1e9, EJ.MMU)
            dl = jnp.where(dl >= tk, dedx * step, dl)
            return i + go, jnp.where(go, left - step, left), jnp.where(go, lost + dl, lost)

        cond = lambda s: (s[0] < tr.N_SLICES_MAX) & (s[1] > 0) & (tk0 - s[2] > tr.TK_STOP)
        i, left, lost = lax.while_loop(cond, body, (jnp.int32(0), ll, jnp.zeros_like(pp)))
        rest = (left > 0) & (tk0 - lost > tr.TK_STOP)
        tk = tk0 - lost
        pg = jnp.sqrt(jnp.clip((tk + EJ.MMU) ** 2 - EJ.MMU ** 2, 1e-6, None))
        lost = jnp.where(rest, lost + EJ._dedx(cc, pg, 1e9, EJ.MMU) * left, lost)
        return jnp.where(tk0 - lost <= tr.TK_STOP, tk0, jnp.minimum(lost, tk0)), i
    return jax.jit(jax.vmap(one))(keys, c, p, L)


def test_slab_xla_matches_reference():
    n = 4096
    keys = _keys(n, 5)
    c, p, L = _inputs(n, 5)
    ref, ri = _reference_slab(keys, c, p, L)
    loss, slices = S.slab_loss(S.raw(keys), c, p, L, impl="xla")
    same = np.asarray(ri) == np.asarray(slices)
    assert same.mean() > 0.995
    assert _close(np.asarray(ref)[same], np.asarray(loss)[same], 1e-9, 0.9)
    assert int(slices.max()) > 20 and int((slices == 1).sum()) > 0     # range-out and one-slice lanes both covered
    assert np.all(np.asarray(loss)[np.asarray(L) == 0] == 0.0)


def test_slab_kernel_interpreted_matches_xla():
    n = 512
    keys = _keys(n, 9)
    c, p, L = _inputs(n, 9)
    a, ia = S.slab_loss(S.raw(keys), c, p, L, impl="xla")
    b, ib = S.slab_loss(S.raw(keys), c, p, L, impl="interpret", block=64)
    same = np.asarray(ia) == np.asarray(ib)
    assert same.mean() > 0.99
    rel = np.abs(np.asarray(a) - np.asarray(b))[same] / np.maximum(np.asarray(a)[same], 1e-300)
    assert rel.max() < 1e-9


@pytest.mark.parametrize("mat, p, L", [(M.CONCRETE_MARS, 800.0, 300.0), (M.MILD_STEEL, 400.0, 100.0),
                                       (M.POLYSTYRENE, 3000.0, 20.0)])
def test_slab_matches_numpy_in_distribution(mat, p, L):
    n = 40_000
    keys = _keys(n, 13)
    cc = EJ.consts(mat)
    dev = np.asarray(S.slab_loss(S.raw(keys), cc, jnp.full(n, p), jnp.full(n, L), impl="xla")[0])
    ref = tr.slab_loss(np.random.default_rng(13), mat, np.full(n, p), np.full(n, L))
    err = np.hypot(dev.std(), ref.std()) / np.sqrt(n)
    assert abs(dev.mean() - ref.mean()) < 4 * err
    qd, qr = np.quantile(dev, [0.1, 0.5, 0.9]), np.quantile(ref, [0.1, 0.5, 0.9])
    assert np.all(np.abs(qd / qr - 1) < 0.02)


def _path_case(n, J, seed):
    rng = np.random.default_rng(seed)
    mats = [MATS[i % len(MATS)] for i in range(J)]
    cs = [EJ.consts(m) for m in mats]
    c = EJ.MatConsts(*(jnp.asarray(np.array([getattr(cc, f) for cc in cs])) for f in EJ.MatConsts._fields))
    L = jnp.asarray(rng.choice([0.0, 0.0, 3.0, 60.0, 400.0], (J, n)) * rng.uniform(0.5, 1.5, (J, n)))
    p = jnp.asarray(np.geomspace(150.0, 2e4, n)[rng.permutation(n)])
    f1 = jnp.asarray(np.arange(J) // 3, jnp.uint32)
    f2 = jnp.asarray(np.arange(J) % 3, jnp.uint32)
    mean = jnp.asarray(np.arange(J) % 3 == 2)
    return S.raw(_keys(n, seed)), p, L, c, f1, f2, mean


def test_path_xla_matches_segment_by_segment():
    k, p, L, c, f1, f2, mean = _path_case(600, 7, 21)
    out = np.asarray(S.path_loss(k, p, L, c, f1, f2, mean, impl="xla"))
    lost = jnp.zeros_like(p)
    for j in range(L.shape[0]):
        pp = S.p_after(p, lost)
        cj = EJ.MatConsts(*(f[j] for f in c))
        if bool(mean[j]):
            lost = lost + EJ._dedx(cj, jnp.maximum(pp, 1.0), 1e9, EJ.MMU) * L[j]
        else:
            dl, _ = S.slab_loss(S.fold_in(S.fold_in(k, f1[j]), f2[j]), cj, pp, L[j], impl="xla")
            lost = lost + jnp.where(L[j] > 0, dl, 0.0)
        assert _close(out[j], lost, 1e-9, 0.5, 0.99)


def test_path_kernel_interpreted_matches_xla():
    args = _path_case(64, 5, 22)
    a = np.asarray(S.path_loss(*args, impl="xla"))
    b = np.asarray(S.path_loss(*args, impl="interpret", block=32))
    assert a.shape == b.shape and _close(a, b, 1e-9, 0.5, 0.99)


def test_default_impl_off_cuda_is_xla(monkeypatch):
    monkeypatch.delenv("MU2EFS_SLAB_KERNEL", raising=False)
    if not S.cuda_backend():
        assert S.default_impl() == "xla"
    monkeypatch.setenv("MU2EFS_SLAB_KERNEL", "xla")
    assert S.default_impl() == "xla" and S.provenance()["impl"] == "xla"
    monkeypatch.setenv("MU2EFS_SLAB_KERNEL", "bogus")
    with pytest.raises(ValueError):
        S.default_impl()
