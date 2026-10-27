"""Opt-in restricted-soft plus explicit-hard spin-1/2 energy-loss requests (MeV, mm).

The caller MUST supply ``restricted_mean``: the mean SOFT energy over this step,
restricted at this same ``tcut`` (not a stopping power and not an inclusive mean).
It must be finite, nonnegative and zero for zero length. This module never infers
whether a table is inclusive/restricted, subtracts a hard mean, or supplies a
Bethe mean implicitly. Passing an inclusive mean double counts hard transfers.
The caller owns that physics contract, step subdivision and the energy ledger.

In admitted regimes, soft fluctuations are exactly the existing
``eloss_jax.sample_loss(..., mean=)`` model, including its fixed-round numerical
approximations. The split is partition-consistent only in these admitted regimes:
the non-Bohr Urban branch and the no-hard case (tcut >= Tmax). On positive paths,
the inherited Bohr branch with tcut < Tmax is UNSUPPORTED and returns NaN for
``soft_requested``. Its variance is A*length*(Tmax - beta**2*cut/2), not the
restricted analytical integral; adding explicit hard transfers would overlap.
Here cut = min(tcut, Tmax), and Bohr is selected by restricted_mean >=
_BOHR_MIN*cut and Tmax <= 2*cut. Callers must treat that NaN as numerically
unresolved, not zero, a physical stop, or permission to commit the hard request.
Extension into this unsupported regime awaits an audited fluctuation model;
this module does not modify the legacy sampler or infer replacement physics.
Independently, hard transfers form a compound Poisson process at the supplied,
frozen momentum:

    dN/(dx dT) = A [1/T**2 - beta**2/(Tmax*T) + 1/(2*E**2)],
    A = TWOPI_MC2_RCL2 * electron_density / beta**2,
    tcut <= T <= Tmax, E = kinetic_energy + mass.

This is the same spin-1/2 spectrum and rate as numpy ``eloss.sample_deltas``
(radiative corrections omitted). Poisson counts and inverse-1/T**2 rejection
sampling use device-side dynamic loops, with no hard-count cap, host event loop,
or variable-shaped secondary list. Cost grows with the largest count/rejection
history in a batch. The first two energy cumulants are integrals of T and T**2
against this intensity, NOT the variance of a single transfer.

``sample_loss_split`` takes separate soft/hard keys; the hard key is split again
for counts and transfer energies. Reusing keys intentionally repeats a channel;
callers must supply independent keys for independent draws. Results are requests:
neither individual channels nor their sum are clipped to kinetic energy, and no
secondary transport or deposited-energy assignment is performed here.

All physical inputs and MatConsts fields broadcast, including scalars. Close over
a Material under jit, or pass MatConsts dynamically. Valid inputs have finite
p >= 0, length >= 0, tcut > 0, mass > 0 and physical material constants. Momentum
uses the existing P_FLOOR convention. Value-domain validation is the caller's
responsibility; this module does not perform host checks on traced arrays.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from . import eloss_jax
from .eloss import MMU, TWOPI_MC2_RCL2, _BOHR_MIN


class SplitLoss(NamedTuple):
    """Requested soft/hard energies [MeV] and integer hard count; a JAX pytree."""
    soft_requested: object
    hard_requested: object
    hard_count: object


class HardMoments(NamedTuple):
    """Hard count rate [/mm], first cumulant [MeV/mm], second cumulant [MeV**2/mm]."""
    rate: object
    mean: object
    variance: object


def _parameters(p, tcut, mass):
    kinetic, beta2, _, _, tmax = eloss_jax.kinematics(p, mass)
    return beta2, tmax, kinetic + mass, jnp.minimum(tcut, tmax)


@jax.jit
def _hard_moments(ne, p, tcut, mass):
    ne, p, tcut, mass = jnp.broadcast_arrays(*(jnp.asarray(x, jnp.float64) for x in (ne, p, tcut, mass)))
    beta2, top, total, cut = _parameters(p, tcut, mass)
    width = top - cut
    z = width / cut
    log_ratio = jnp.log1p(z)
    small = z < 1e-3
    s = jnp.where(small, z, 0.0)
    # Series avoid subtractive cancellation when the hard interval nearly closes.
    z_minus_log = jnp.where(small, s * s * (0.5 + s * (-1 / 3 + s * (0.25 + s * (-0.2 + s / 6)))), z - log_ratio)
    log_minus_fraction = jnp.where(small, s * s * (0.5 + s * (-2 / 3 + s * (0.75 + s * (-0.8 + s * 5 / 6)))),
                                   log_ratio - width / top)
    scale = TWOPI_MC2_RCL2 * ne / beta2
    rate = scale * ((z_minus_log + (1 - beta2) * log_ratio) / top + width / (2 * total * total))
    mean = scale * (log_minus_fraction + (1 - beta2) * width / top + width * (top + cut) / (4 * total * total))
    variance = scale * width * ((1 - beta2) + beta2 * width / (2 * top)
                               + (top * top + top * cut + cut * cut) / (6 * total * total))
    active = tcut < top
    return HardMoments(*(jnp.where(active, value, 0.0) for value in (rate, mean, variance)))


def hard_moments_per_mm(mat, p, tcut, mass=MMU):
    """Analytic hard (count rate, energy mean, energy variance) per mm, zero if tcut >= Tmax.

    For fixed p over length L, E[N] = Var[N] = L*rate, E[hard] = L*mean,
    Var[hard] = L*variance. All outputs share the broadcast input shape.
    """
    return _hard_moments(eloss_jax.consts(mat).ne, p, tcut, mass)


def hard_rate_per_mm(mat, p, tcut, mass=MMU):
    """Mean hard count per mm, equivalent to numpy eloss.delta_rate_per_mm."""
    return hard_moments_per_mm(mat, p, tcut, mass).rate


def hard_mean_per_mm(mat, p, tcut, mass=MMU):
    """First hard-energy cumulant per mm [MeV/mm]."""
    return hard_moments_per_mm(mat, p, tcut, mass).mean


def hard_variance_per_mm(mat, p, tcut, mass=MMU):
    """Second hard-energy cumulant per mm [MeV**2/mm], not single-transfer variance."""
    return hard_moments_per_mm(mat, p, tcut, mass).variance


def _sample_hard(key, c, p, length, tcut, mass):
    count_key, energy_key = jax.random.split(key)
    rate = _hard_moments(c.ne, p, tcut, mass).rate
    count = jax.random.poisson(count_key, rate * length, shape=p.shape, dtype=jnp.int64)
    beta2, top, total, cut = _parameters(p, tcut, mass)
    envelope = 1 + 0.5 * (top / total) ** 2

    def pending(state):
        return jnp.any(state[1] > 0)

    def draw(state):
        key, remaining, energy = state
        key, proposal_key, accept_key = jax.random.split(key, 3)
        u = jax.random.uniform(proposal_key, p.shape, jnp.float64)
        transfer = cut / ((1 - u) + (cut / top) * u)
        weight = 1 - beta2 * transfer / top + 0.5 * (transfer / total) ** 2
        accepted = (remaining > 0) & (jax.random.uniform(accept_key, p.shape, jnp.float64) * envelope <= weight)
        return key, remaining - accepted.astype(remaining.dtype), energy + jnp.where(accepted, transfer, 0.0)

    _, _, energy = lax.while_loop(pending, draw, (energy_key, count, jnp.zeros_like(p)))
    return energy, count


@jax.jit
def _sample_split(soft_key, hard_key, c, p, length, tcut, mass, restricted_mean):
    values = jnp.broadcast_arrays(*(jnp.asarray(x, jnp.float64) for x in (p, length, tcut, mass, restricted_mean, *c)))
    p, length, tcut, mass, restricted_mean = values[:5]
    c = eloss_jax.MatConsts(*values[5:])
    soft = eloss_jax.sample_loss(soft_key, c, p, length, tcut, mass=mass, mean=restricted_mean)
    _, top, _, cut = _parameters(p, tcut, mass)
    # The inherited Bohr width is not restricted at cut when a hard interval
    # remains. Fail closed rather than silently double-counting fluctuations.
    unsupported = ((length > 0) & (tcut < top)
                   & (restricted_mean >= _BOHR_MIN * cut) & (top <= 2 * cut))
    soft = jnp.where(unsupported, jnp.nan, soft)
    hard, count = _sample_hard(hard_key, c, p, length, tcut, mass)
    return SplitLoss(soft, hard, count)


def sample_loss_split(soft_key, hard_key, mat, p, length, tcut, mass=MMU, *, restricted_mean):
    """Sample SplitLoss(soft_requested, hard_requested, hard_count), without kinetic clipping.

    restricted_mean is REQUIRED and keyword-only: caller-supplied restricted step
    energy [MeV], excluding transfers above this tcut. None is not a default mean.
    soft_key and hard_key must be independent explicit JAX keys. Inputs, including
    restricted_mean, mass and material fields, broadcast to the result shape.

    Positive-path Bohr requests with a nonempty hard interval return NaN soft
    energy (see module contract). The caller must classify these as numerically
    unresolved; finite hard outputs do not make the combined request admissible.
    Partition consistency is limited to admitted regimes; extending support
    requires an audited fluctuation model, not a change to the legacy sampler.
    """
    if restricted_mean is None:
        raise ValueError("restricted_mean must be supplied as restricted step energy [MeV], not None")
    return _sample_split(soft_key, hard_key, eloss_jax.consts(mat), p, length, tcut, mass, restricted_mean)
