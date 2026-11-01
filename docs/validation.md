# Core: validation

Core's physics is validated where it is used: the energy loss, scattering, world and crossings through the CRV and
tracker comparisons with Geant4 (CRV V1, tracker T-V1, beam stop fraction; see each stage's `docs/validation.md`).
This page lists the checks of core components on their own. "Commit" is the core commit the check ran on, or core's
`main` on that date.

| Check | Reference | Sample | Metric | Result | Threshold | Verdict | Date | Commit |
|---|---|---|---|---|---|---|---|---|
| restricted dE/dx, JAX twin | `eloss.dedx_restricted` (NumPy) | 8 materials x 3 cuts | relative difference | <= 1e-12 | rtol 1e-12 | pass | 2026-10-21 | `f78fe53` |
| energy-loss sampler, JAX twin | `eloss.sample_loss` (NumPy) | 14 (material, p, length, cut) points, 4e6 samples per side | mean; P(loss = 0); q05..q99; binned ratio | all within | 0.5 %; 5 sigma; 2 %; 1 % beyond 2 sigma | pass | 2026-10-21 | `f78fe53` |
| muon energy loss and scattering vs Geant4 11.3.2 | Geant4 CrvSteps and StrawGasSteps | CRV V1 (1 M events), tracker T-V1 (20 shards) | crossings, path, visible energy, eIon MPV | see the CRV and tracker validation pages | V1, T-V1 thresholds | pass | 2026-10-16..20 | `65c3609`..`a6fd330` |
| crossings, JAX vs NumPy | NumPy slab crossings | CRV V1 shards | crossings | identical in float64 | identical | pass | 2026-10-16 | `65c3609` |
| device CRY generator | stock CRY tables (Offline `ExtractedCRY`) | 10 x 50 k accepted events per arm, A100 | accepted rate ratio; multiplicity; 2,263 spectrum bins | 0.99978 +- 0.0016; multiplicity 1.00598 vs 1.00561 (about 2 SE); no Bonferroni flag; positive-muon rate 2.8 SE low | descriptive (no agreed threshold) | consistent; kaons not observed | 2026-10-30 | `e7c56f6` |
| device CRY vs native CRY | the native C++ CRY reference | same | accepted rate ratio | 0.9989 +- 0.0018; multiplicity excursions < 2 SE | descriptive | consistent | 2026-10-30 | `e7c56f6` |
| device-generated CRV chain | table-driven chain | 50 k events, GPU response and reco | clusters per live second per sector type | EX 1.032 +- 0.032, T1 0.998 +- 0.072, T2 0.970 +- 0.061, M 1.008 +- 0.100 | descriptive | consistent | 2026-10-30 | `e7c56f6` |
| endpoint admission, Slurm recovery, artifact bindings | fake Slurm site and mocks | unit tests | refusal paths, at-most-once dispatch, retained code | pass | tests | pass (no live rollout implied) | 2026-10-25 | main |

## Electrons (not on `main`)

Electron and positron energy loss (Seltzer-Berger bremsstrahlung, Moller/Bhabha ionisation, straight slab step, NumPy)
is on an unmerged branch:

| Check | Reference | Result | Verdict |
|---|---|---|---|
| radiative stopping power | NIST ESTAR, 1-500 MeV, Al, W, Mylar, polystyrene | within 0.6 % | pass |
| collision stopping power | NIST ESTAR, 0.1-500 MeV | within 1 % | pass |
| thin layers (<= 0.1 X0), 105 MeV electrons | Geant4 electron gun | loss chi2/ndf 0.6-1.6; fractions losing > 1 / 5 MeV within 1 % | pass |
| 1 X0 of tungsten | Geant4 | loss chi2/ndf 16 (no multiple scattering in the model) | **fail** |
| positrons, 1 mm Al | Geant4 | loss chi2/ndf 8.4 (no annihilation in the model) | **fail** |

## Device precision (pending change)

A float32 device mode is validated on 100 CRV shards (within the float64 seed-to-seed spread) but not on `main`;
every result above is float64.
