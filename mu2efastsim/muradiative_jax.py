"""JAX twin of muradiative.sample_loss: radiative muon loss (brems, pair, photonuclear) with fixed-shape lanes.

Same tables and arithmetic as muradiative.RadiativeTable.sample: the continuous part below V_CUT interpolated in ln E,
a Poisson count of discrete transfers, each at an energy node chosen with the interpolation weight and a fractional
transfer v from the node's inverse CDF. The count is drawn by jax.random.poisson; transfers beyond K_RAD per lane are
dropped (the discrete rate is below 2 per metre of steel at 1 TeV, so P(n > K_RAD) < 1e-9 for the slabs of the
transport). Draws differ from the NumPy sampler (jax.random keys), the distribution does not.
"""
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .eloss import MMU
from . import muradiative as R

jax.config.update("jax_enable_x64", True)

K_RAD = 8           # discrete transfers per lane


class RadTable(NamedTuple):
    lnE: object     # [nE] ln total energy of the nodes
    cont: object    # [nE] continuous mean loss below V_CUT [MeV/mm]
    rate: object    # [nE] discrete rate above V_CUT [1/mm]
    lvq: object     # [nE, N_U] ln v at the inverse-CDF quantiles
    tk_min: float   # no radiative loss below this kinetic energy [MeV]
    mass: float


def table(mat, mass=MMU):
    """Device RadTable of a material (built once from the cached NumPy table)."""
    t = R.table(mat, mass)
    return RadTable(jnp.asarray(t.lnE), jnp.asarray(t.cont), jnp.asarray(t.rate), jnp.asarray(np.log(t.vq)),
                    float(t.tk[0]), float(mass))


def stack(mats, mass=MMU):
    """RadTable of several materials (leading material axis on cont, rate, lvq; the energy grid is shared), indexed
    per lane by `mat` in sample_loss / after_ionisation."""
    ts = [R.table(m, mass) for m in mats]
    if any(not np.array_equal(t.lnE, ts[0].lnE) for t in ts):
        raise ValueError("radiative tables on different energy grids")
    return RadTable(jnp.asarray(ts[0].lnE), jnp.asarray(np.stack([t.cont for t in ts])),
                    jnp.asarray(np.stack([t.rate for t in ts])), jnp.asarray(np.log(np.stack([t.vq for t in ts]))),
                    float(ts[0].tk[0]), float(mass))


def _per_lane_keys(key, p):
    typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
    return key.ndim == p.ndim + (0 if typed else 1) and p.ndim > 0


def sample_loss(key, tab, p, length, mat=None, k_max=K_RAD):
    """Radiative loss [MeV] of muons of |p| [MeV/c] over `length` [mm] (lanes), at the entry energy. key: one key for
    all lanes or one per lane; mat: per-lane material index into a stack() table (None: a single-material table).
    Computes in the dtype of p."""
    if _per_lane_keys(key, p):
        one = lambda k, pp, ll, mm: sample_loss(k, tab, pp[None], ll[None], None if mm is None else mm[None], k_max)[0]
        return jax.vmap(one, in_axes=(0, 0, 0, None if mat is None else 0))(key, p, jnp.broadcast_to(length, p.shape),
                                                                             mat)
    dt = p.dtype
    m = tab.mass
    lnEg = tab.lnE.astype(dt)
    cont_t, rate_t, lvq_t = (x.astype(dt) for x in (tab.cont, tab.rate, tab.lvq))
    if mat is not None:
        cont_t, rate_t, lvq_t = cont_t[mat], rate_t[mat], lvq_t[mat]      # [lanes, nE] ...
    E = jnp.sqrt(p * p + m * m)
    tk = E - m
    on = tk >= tab.tk_min
    lnE = jnp.log(jnp.clip(E, jnp.exp(lnEg[0]), jnp.exp(lnEg[-1])))
    n_e = lnEg.shape[0]
    j = jnp.clip(jnp.searchsorted(lnEg, lnE) - 1, 0, n_e - 2)
    w = (lnE - lnEg[j]) / (lnEg[j + 1] - lnEg[j])
    pick = (lambda t, jj: t[jj]) if mat is None else (lambda t, jj: jnp.take_along_axis(t, jj[..., None], -1)[..., 0])
    cont = jnp.where(on, ((1 - w) * pick(cont_t, j) + w * pick(cont_t, j + 1)) * length, 0.0)
    lam = jnp.where(on, ((1 - w) * pick(rate_t, j) + w * pick(rate_t, j + 1)) * length, 0.0)
    kn, kj, kq = jax.random.split(key, 3)
    n = jax.random.poisson(kn, lam)
    shape = p.shape + (k_max,)
    node = j[..., None] + (jax.random.uniform(kj, shape, dtype=dt) < w[..., None])
    n_u = lvq_t.shape[-1]
    uq = jax.random.uniform(kq, shape, dtype=dt) * (n_u - 1)
    k = jnp.minimum(jnp.floor(uq).astype(jnp.int32), n_u - 2)
    f = uq - k
    if mat is None:
        lo, hi = lvq_t[node, k], lvq_t[node, k + 1]
    else:
        lane = jnp.arange(p.shape[0])[:, None] if p.ndim else 0
        lo, hi = lvq_t[lane, node, k], lvq_t[lane, node, k + 1]
    v = jnp.exp((1 - f) * lo + f * hi)
    take = jnp.arange(k_max) < n[..., None]
    loss = cont + jnp.sum(jnp.where(take, v * E[..., None], 0.0), axis=-1)
    return jnp.minimum(loss, tk)


def after_ionisation(key, tab, p, length, ion_loss, mat=None, k_max=K_RAD):
    """transport.slab_loss's radiative step on a device slab result: ion_loss [MeV] from the ionisation slices (capped,
    tk0 when stopped), the radiative draw at the slab-entry |p| added for muons still above TK_STOP, then the stop
    and the cap at the kinetic energy. Lanes with length 0 are unchanged."""
    from .transport import TK_STOP
    tk0 = jnp.sqrt(p * p + tab.mass ** 2) - tab.mass
    rad = sample_loss(key, tab, p, length, mat, k_max)
    lost = ion_loss + jnp.where((tk0 - ion_loss > TK_STOP) & (length > 0), rad, 0.0)
    return jnp.where(tk0 - lost <= TK_STOP, tk0, jnp.minimum(lost, tk0))
