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


def sample_loss(key, tab, p, length, k_max=K_RAD):
    """Radiative loss [MeV] of muons of |p| [MeV/c] over `length` [mm] (lanes), at the entry energy."""
    m = tab.mass
    E = jnp.sqrt(p * p + m * m)
    tk = E - m
    on = tk >= tab.tk_min
    lnE = jnp.log(jnp.clip(E, jnp.exp(tab.lnE[0]), jnp.exp(tab.lnE[-1])))
    n_e = tab.lnE.shape[0]
    j = jnp.clip(jnp.searchsorted(tab.lnE, lnE) - 1, 0, n_e - 2)
    w = (lnE - tab.lnE[j]) / (tab.lnE[j + 1] - tab.lnE[j])
    cont = jnp.where(on, ((1 - w) * tab.cont[j] + w * tab.cont[j + 1]) * length, 0.0)
    lam = jnp.where(on, ((1 - w) * tab.rate[j] + w * tab.rate[j + 1]) * length, 0.0)
    kn, kj, kq = jax.random.split(key, 3)
    n = jax.random.poisson(kn, lam)
    shape = p.shape + (k_max,)
    node = j[..., None] + (jax.random.uniform(kj, shape, dtype=p.dtype) < w[..., None])
    n_u = tab.lvq.shape[1]
    uq = jax.random.uniform(kq, shape, dtype=p.dtype) * (n_u - 1)
    k = jnp.minimum(jnp.floor(uq).astype(jnp.int32), n_u - 2)
    f = uq - k
    v = jnp.exp((1 - f) * tab.lvq[node, k] + f * tab.lvq[node, k + 1])
    take = jnp.arange(k_max) < n[..., None]
    loss = cont + jnp.sum(jnp.where(take, v * E[..., None], 0.0), axis=-1)
    return jnp.minimum(loss, tk)
