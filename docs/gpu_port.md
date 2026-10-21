# GPU port of the energy-loss sampler

## Status (2026-10-21)

`mu2efastsim/eloss_jax.py` is a jit-able float64 JAX twin of `eloss.dedx_restricted` and `eloss.sample_loss`
(Geant4 11.3.2 MuBetheBloch + G4UniversalFluctuation/G4UrbanFluctuation). It is validated against the numpy module
(`tests/test_eloss_jax.py`: 21 tests pass) but **nothing calls it yet**: `eloss.py` stays the default and the reference.
Full core suite: 71 passed, 3 skipped.

## What blocked a direct `jax.numpy` swap, and how each was resolved

| Construct in `eloss.py` | Why it does not jit | Resolution in `eloss_jax.py` |
|---|---|---|
| `numpy.random.Generator` | stateful RNG | explicit `jax.random` key, split into one subkey per random quantity; same key gives bit-identical output |
| boolean-mask assignment (`out[gs] = ...`, `out[u] = ...`) | data-dependent shapes | every branch (Gauss, Gamma, Urban) is evaluated on the full array and selected with `jnp.where`; inputs are made safe outside each branch's mask so every lane stays finite |
| truncated-Gauss rejection loop (`_sample_gauss_clipped`) | data-dependent trip count | exact inverse-CDF sampling of the Gauss truncated to [0, 2 eav] (`ndtr`/`ndtri`) from one uniform per step. Same distribution, different random stream |
| `rng.gamma` (Bohr regime, mean/sigma < 2) | no data-independent algorithm | Marsaglia-Tsang with `GAMMA_ROUNDS = 10` fixed rounds (acceptance >= 0.95 per round, so P(no acceptance) < 1e-12), with the U^(1/alpha) boost for alpha < 1 |
| `rng.poisson` + `np.repeat`/`np.bincount` for the ionisation terms | variable output length | one `fori_loop` of `K_POISSON = 32` rounds does inverse-CDF Poisson for the two excitation means and the ionisation mean, and also adds the 1/E^2 ionisation term of each round taken. The cap is safe because every Poisson mean is <= 8 (see below): P(N > 32 \| mean 8) = 3.2e-11 |
| (delta-ray sampling, `sample_deltas`) | rejection loop, variable count | not ported yet |

### Why the Poisson means are <= 8, and the cancellation fix

* Excitations: means above `_NMAXCONT = 8` are handled by the Gaussian continuous part, as in Geant4, so the Poisson means are <= 8.
* Ionisation: Geant4 (and `eloss.py`) uses `alfa = w1 (8 + a3) / (8 w1 + a3)`, `namean = a3 w1 (alfa-1) / ((w1-1) alfa)` and
  `p3 = a3 - namean`. Algebraically this is `p3 = 8 a3 / (8 + a3) < 8` for any w1. But the G4 form cancels badly as
  w1 = cut/e0 approaches 1: in float64, p3 reaches 8.0008 at w1 - 1 = 1e-4 (the failing test), and about 32 at w1 - 1 = 1e-12.
  `eloss_jax._ion_split` therefore uses the closed forms `p3 = 8 a3/(8+a3)`, `namean = a3 - p3`,
  `alfa - 1 = a3 (w1-1)/(8 w1 + a3)` and `alfa1 = alfa log1p(alfa-1)/(alfa-1)`. These agree with the G4 expressions to 1e-9 wherever
  those are well conditioned (w1 - 1 >= 1e-3), and make p3 <= 8 hold exactly. In practice w1 = cut/10 eV is >= 99 for every
  cut we use (990 eV and up), so this changes no physics result. It only makes the K-cap argument hold everywhere.

## Validation (`python3 tests/test_eloss_jax.py`, N = 4e6 samples per side, numpy seed 0 vs JAX key 0)

Criteria per point:
* mean within 0.5 %;
* P(loss = 0) equal within 5 binomial sigma;
* quantiles q05..q99 of the non-zero losses within 2 %;
* `stats.ratio_band` (20 bins over the numpy q05-q95) max |ratio - 1| minus 2 sigma <= 1 %.

dE/dx is checked exactly (rtol 1e-12) for 8 materials x 3 cuts.

| point | mean ref [MeV] | branches G/Gm/U | mean rel | max abs q rel | band | band-2sig |
|---|---|---|---|---|---|---|
| gas 990eV 1mm 3GeV | 0.000196 | 0/0/1 | -0.027 % | 0.127 % | 1.61 % | 0.33 % |
| gas 990eV 5mm 3GeV | 0.0009792 | 0/0/1 | +0.097 % | 0.903 % | 1.33 % | 0.48 % |
| gas 990eV 10mm 300MeV | 0.001593 | 0/0/1 | +0.021 % | 0.505 % | 0.99 % | 0.00 % |
| gas 990eV 1-10mm 0.2-10GeV | 0.001007 | 0/0/1 | +0.059 % | 0.335 % | 0.70 % | 0.00 % |
| PS 87.2keV 20mm 3GeV | 3.305 | 0/0/1 | -0.003 % | 0.008 % | 1.19 % | 0.21 % |
| PS 87.2keV 20mm 300MeV | 3.441 | 0/0/1 | -0.003 % | 0.011 % | 0.73 % | 0.14 % |
| concrete 1e9 100mm 200MeV (gauss) | 43.61 | 1/0/0 | +0.006 % | 0.016 % | 0.97 % | 0.16 % |
| concrete 1e9 500mm 1GeV (urban, mean/cut 2.5) | 211.5 | 0/0/1 | -0.009 % | 0.036 % | 0.92 % | 0.00 % |
| concrete 1e9 50mm 3GeV (urban) | 23.32 | 0/0/1 | +0.015 % | 0.409 % | 2.37 % | 0.15 % |
| concrete 1e9 5%tk 100MeV (range-out) | 1.991 | 0/0/1 | -0.012 % | 0.026 % | 0.95 % | 0.00 % |
| concrete 1e9 5%tk 40MeV (range-out) | 0.3659 | 0/0/1 | -0.013 % | 0.033 % | 1.14 % | 0.17 % |
| PS 87.2keV 5%tk 60MeV (range-out) | 0.7375 | 0/0/1 | -0.007 % | 0.017 % | 1.12 % | 0.17 % |
| steel 1e9 2mm 50MeV (gauss) | 7.987 | 1/0/0 | +0.005 % | 0.014 % | 0.97 % | 0.16 % |
| steel 1e9 0.1um 1.85MeV (gamma) | 0.003348 | 0/1/0 | -0.014 % | 0.204 % | 1.19 % | 0.00 % |

Notes:
* **Zero losses**: in 1 mm of straw gas at 3 GeV/c, 11.8 % of the steps have exactly zero loss (no Poisson excitation or
  ionisation). The same fraction appears in numpy (0.11839) and JAX (0.11883), which is 0.95 sigma at 1e6. Because of this
  zero spike, plain q05/q10 ratios are 0/0, so the quantiles are taken on the non-zero part.
* **Band metric noise**: the tail bins of the band hold only about 3-15k entries at 1e6 (2-2.5 % statistical error). The
  max over 20 bins of (|z| - 2) is therefore often above 0. Numpy compared with numpy (two seeds, 1e6 each, 12 pairs)
  gave band-2sig values of 0-1.1 %, and JAX compared with numpy at 1e6 gave 0-2.3 % (8 pairs) for the concrete 50 mm point.
  At 6e6-1e7 per side, JAX vs numpy is consistent with numpy vs numpy: chi2 over 100 equal-probability bins p = 0.95
  (numpy-numpy 0.39), KS p = 0.32 (0.96), mean difference 0.4 sigma, and no bin above 1 % at 8e6. So the earlier 1.27 % was
  statistical noise, and the test now uses N = 4e6.
* **Gamma branch**: it only happens for p of about 0.9-3 MeV/c (the last micrometre of range: mean >= 10 Tmax, but
  mean/sigma < 2). The earlier "steel 2mm 50MeV (gamma)" point was actually in the Gauss branch, which is why
  `test_branch_coverage` failed. A 0.1 um step at 1.85 MeV/c in steel now covers the Gamma branch. This is a real branch,
  but it is irrelevant for slab_loss in practice.

## CPU benchmark (`python3 -m mu2efastsim.eloss_jax --bench`, Apple M5, 10 cores, jax 0.11.2 CPU, jit warm)

| case | n | numpy [steps/s] | JAX [steps/s] |
|---|---|---|---|
| straw gas 990 eV, 1-10 mm, 3 GeV/c | 1e5 | 5.2e6 | 2.8e6 |
| polystyrene 87.2 keV, 20 mm, 0.3-10 GeV/c | 1e5 | 4.3e6 | 2.8e6 |
| concrete 1e9, 5-50 mm, 0.2-3 GeV/c | 1e5 | 4.5e6 | 2.8e6 |
| straw gas | 1e6 | 5.1e6 | 3.1e6 |
| polystyrene | 1e6 | 4.0e6 | 3.0e6 |
| concrete | 1e6 | 4.2e6 | 3.0e6 |

The first call, including compilation, takes 0.3-0.7 s for each new shape. On CPU, JAX runs at about 0.6-0.75x the
speed of numpy. The fixed design costs a 32-round Poisson loop and a 10-round gamma loop on every lane, while numpy draws
only what each mask needs. The CPU port is therefore not a speed-up; the expected gain is on a GPU, or from fusing the
sampler into a larger jitted transport step. This has not been measured (no GPU available here).

## Plugging it in later

* **API**: `sample_loss(key, mat, p, length, tcut)` takes a `jax.random.key`, a `Material` (constants folded at trace time)
  or `MatConsts`, and arrays that broadcast to p's shape. For batches with mixed materials, use
  `stack_consts(mats, index)`, which builds per-step constant arrays and gives one compiled program for all materials.
* **slab_loss** (`transport.py`): the loop runs at most `N_SLICES_MAX = 200` slices. Each slice loses about
  `SLICE_FRAC = 5 %` of the kinetic energy; slices are coarser (`MAX_SLICES_PER_SLAB`) when the muon has energy to spare.
  A sampled loss of at least the remaining kinetic energy is replaced by the mean, and a muon stops below `TK_STOP`. The
  loop already has a fixed maximum trip count and works with masks, so it maps onto `lax.fori_loop`/`while_loop` as follows:
  * replace the boolean-index updates with `jnp.where`;
  * use `jax.random.fold_in(key, i)` per slice;
  * the trailing mean step for muons still inside after the cap becomes another `where`.

  Keep a backend switch (numpy default, like `crossings.py`) so the existing tests keep running on `eloss.py`.
* **In-gas sampling** (straws, tcut 990 eV): the in-gas path lengths come from the geometry, which is numpy except
  `crossings.py` (slab entry/exit, JAX backend). Pass fixed-shape arrays with length 0 for steps that miss: a mean below
  `_MINLOSS` gives loss = mean = 0. Delta rays above the cut (`sample_deltas`) still need a fixed-cap port, using
  inverse-CDF 1/T^2 with the spin-1/2 weight applied by a fixed number of rejection rounds.
* **Validation when switching**: rerun `tests/test_eloss_jax.py`. Statistical comparisons must use >= 4e6 samples, or the
  band metric must be calibrated against numpy compared with numpy (see Notes).

## Reference GPU implementation

Celeritas `src/celeritas/em/distribution/EnergyLossUrbanDistribution.hh` (with `EnergyLossGaussianDistribution`,
`EnergyLossGammaDistribution`, `EnergyLossHelper`, `FluctuationData`) was used as the guide for the branch structure.
