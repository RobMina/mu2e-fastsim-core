# GPU port of the energy-loss sampler

Status (2026-10-16): `eloss.py` is numpy only. `crossings.py` is the only module with a JAX backend.

## What blocks a direct `jax.numpy` swap

| Construct in `eloss.py` | Why it does not jit | JAX replacement |
|---|---|---|
| `numpy.random.Generator` (`rng.random`, `normal`, `poisson`, `gamma`) | stateful RNG | explicit `jax.random` keys split per call |
| boolean-mask assignment (`out[gs] = ...`, `out[u] = ...`) | data-dependent shapes | compute every branch on the full array and select with `jnp.where` |
| `while todo.any()` rejection loops (truncated Gauss, delta-ray sampling) | data-dependent trip count | fixed number of rounds with masks, or inverse-CDF sampling |
| `np.repeat(..., nnb)` + `np.bincount` for the Poisson number of ionisations | variable output length | cap the count per step (`NMAX`) with a mask, or the Gaussian approximation above `_NMAXCONT`, as Geant4 does |

## Reference GPU implementation

Celeritas (Apache-2.0/MIT) has a GPU port of the same Geant4 Urban model:
`src/celeritas/em/distribution/EnergyLossUrbanDistribution.hh`, with `EnergyLossGaussianDistribution`,
`EnergyLossGammaDistribution`, `EnergyLossHelper` and `FluctuationData`. Use it as the reference for the branch
structure and the fixed-trip-count sampling, and validate the JAX port against `eloss.py` (same distributions,
statistically) before replacing it.
