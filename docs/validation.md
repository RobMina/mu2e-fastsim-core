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
| muon radiative losses (`muradiative`, default in `slab_loss`) | published muon b; Geant4 11.3 cross sections | Al, Cu, Pb, steel, 10 GeV-1 TeV | b = (1/E) dE/dx; sampled mean | Cu at 100 GeV 6.2e-6 cm2/g (brems 2.4, pair 3.4, photonuclear 0.4); sampled means within 5 % of b | published tables | pass | 2026-11-03 | `8410612` |
| radiative losses in the chains | Geant4 (tracker T-V1/T-V2/T-V3, extracted and Run 1A field-on; CRV V1 Run 1A; CRV deck) | 10 M generated tracker, 5 M CRV, 1 M Run 1A CRV, two seeds | every gate of those pages, paired against the ionisation-only transport | no gate changes beyond draw noise (tracker T-V3 yield -0.2 / -0.4 +- 0.4 %; CRV deck 0 histograms flagged against ionisation-only; CRV Run 1A crossings +0.19 % vs +0.24 % ionisation-only) | no gate worse | pass | 2026-11-04 | `b1cf5ab` |
| radiative losses in the device slab samplers (`slab_jax.slab_loss` / `path_loss` with `rad=`) | `transport.slab_loss` (NumPy default) and the NumPy world column | Cu 100 GeV, concrete 20 GeV, steel 300 GeV; a three-material path at 50 GeV; 4-6e4 lanes | mean; tail fractions above the NumPy q50-q99.9 | all within | 4 sigma; 5 sigma | pass | 2026-11-04 | branch |
| radiative draw, JAX twin (`muradiative_jax`) | `muradiative.sample_loss` (NumPy) | Cu 100 GeV, concrete 20 GeV, steel 1 TeV; 2e5 samples | mean; P(loss > 0); q90, q99 | all within | 4 sigma; 4 sigma; 10 % | pass | 2026-11-04 | branch |
| crossings, JAX vs NumPy | NumPy slab crossings | CRV V1 shards | crossings | identical in float64 | identical | pass | 2026-10-16 | `65c3609` |
| device CRY generator | stock CRY tables (Offline `ExtractedCRY`) | 10 x 50 k accepted events per arm, A100 | accepted rate ratio; multiplicity; 2,263 spectrum bins | 0.99978 +- 0.0016; multiplicity 1.00598 vs 1.00561 (about 2 SE); no Bonferroni flag; positive-muon rate 2.8 SE low | descriptive (no agreed threshold) | consistent; kaons not observed | 2026-10-30 | `e7c56f6` |
| device CRY vs native CRY | the native C++ CRY reference | same | accepted rate ratio | 0.9989 +- 0.0018; multiplicity excursions < 2 SE | descriptive | consistent | 2026-10-30 | `e7c56f6` |
| device-generated CRV chain | table-driven chain | 50 k events, GPU response and reco | clusters per live second per sector type | EX 1.032 +- 0.032, T1 0.998 +- 0.072, T2 0.970 +- 0.061, M 1.008 +- 0.100 | descriptive | consistent | 2026-10-30 | `e7c56f6` |
| endpoint admission, Slurm recovery, artifact bindings | fake Slurm site and mocks | unit tests | refusal paths, at-most-once dispatch, retained code | pass | tests | pass (no live rollout implied) | 2026-10-25 | main |

## Electrons

Electron and positron physics (Seltzer-Berger bremsstrahlung, Moller/Bhabha ionisation, mixed screened-Rutherford
multiple scattering, two-photon annihilation, slab stepping; NumPy, `eslab`). Geant4 reference: electron-gun
slabs with EM option 4, 1M events per case (200k for 1 X0 W); merged 2026-11-01:

| Check | Reference | Result | Verdict |
|---|---|---|---|
| radiative stopping power | NIST ESTAR, 1-500 MeV, Al, W, Mylar, polystyrene | within 0.6 % | pass |
| collision stopping power | NIST ESTAR, 0.1-500 MeV | within 1 % | pass |
| loss of transmitted primaries, nine slab cases (Al, Mylar, W; e- and e+; 20-105 MeV) | Geant4 | chi2/ndf 0.5-1.4 | pass |
| 1 X0 of tungsten, 105 MeV e- | Geant4 | loss chi2/ndf 0.8, mean 61.94 vs 62.00 +- 0.06 MeV; exit angle 1.4, displacement 1.1 | pass |
| photon and delta yields | Geant4 | within 0.3-0.5 % | pass |
| e+ annihilation in flight, 1 mm Al | Geant4 | 4.86 +- 0.22 vs 4.93 per 10^4 | pass |
| exit angle, 20 and 50 MeV in 1 mm Al | Geant4 | chi2/ndf 0.9 / 2.0 | pass |
| exit angle, 105 MeV in thin layers | Geant4 | chi2/ndf 7.4-91 (Geant4 WentzelVI condensed stepping vs exact sampling of the same cross section) | explained |
| exit angle, 95 MeV (Geant4 GS regime), 1 mm / 0.1 mm Al | Geant4 | chi2/ndf 2.1 / 9.9 (22.3 / 91 at 105 MeV); only Geant4 breaks the 1/(p beta) scaling across 100 MeV | pass / open (0.1 mm: fast core 2 % wider) |
| stopped / reflected fraction, 1 X0 W (tabulated e- Mott ratio, form factor above 100 MeV) | Geant4 | 18.28 / 18.28 vs 18.41 +- 0.09 % stopped; 0.336 / 0.336 vs 0.347 +- 0.013 % reflected (two seeds; form factor at all energies: 0.28 %, McKinley-Feshbach: 0.24 %) | pass |
| reflected fraction vs msc step settings, 1 X0 W | Geant4 | step cap 0.35 / 0.1 / 0.035 / 0.012 mm and 2 / 10 hard collisions per cap: 0.265-0.285 % (G4 0.347 +- 0.013); the deficit is in reflected electrons above 10 MeV (0.044 vs 0.081 %) | not a step artefact |
| nuclear form factor only above 100 MeV (`emsc.FF_TMIN = 100`, the default since 2026-11-03: Geant4 option 4 uses Goudsmit-Saunderson below 100 MeV, screened Rutherford times the point-nucleus Mott ratio, no form factor) | Geant4, the eleven slab cases, 1 X0 W with two seeds | 1 X0 W reflected 0.336 / 0.336 vs 0.347 +- 0.013 % (form factor everywhere: 0.28 %), loss chi2/ndf 1.0 / 1.3; 0.35 mm W reflected 0.042 vs 0.040 %; loss chi2/ndf 0.6-1.4 in the other cases (0.1 mm Al at 95 MeV 4.6, the Geant4 model change below); the conversion-electron chain (umbrella `cechain`, 108 k paired electrons) unchanged: one electron of 108 k changes fate, paired momentum and time differences identical | pass, default |
| transmitted loss, 0.1 mm Al, Geant4 95 vs 105 MeV (same slab) | Geant4 against itself; fast against itself | Geant4 chi2/ndf 4.9, pulls +5.5 / +7.2 / -5.3 at 13-31 keV (low side of the peak); fast 1.1; Geant4 vs fast 5.1 at 95 MeV, 0.9 at 105 MeV | explained: Geant4 model change at 100 MeV |
| helix step in a uniform field | analytic circle and pitch; composition of steps; zero field | exact to 1e-9 mm | pass |
| RK4 in a uniform field | the exact helix | fourth-order convergence | pass |
| path to a plane along the helix | analytic arcsin solution, re-entry, unreachable planes | 1e-9 relative | pass |
| slab in a transverse field (10 MeV e-, 20 mm air, 1 T) | path R asin(L/R), bend R(1 - cos), loss per mm as without field | within 0.3 % / 1 % / 2 % | pass |
| Mu2e DS field map, 105 MeV e- from the stopping target to z = 8410 mm | RK4 with 1 mm steps | helix with 10 mm steps: 18 um median, 46 um max after 3.7 m; 25 mm steps: 0.11 / 0.29 mm | pass |
| device slab stepper (`eslab_jax`) vs NumPy `eslab`, msc on, 200k e-/e+ per case: Al 0.1 / 1 mm 105 MeV, Al 1 mm 20 MeV, W 0.35 / 3.5 mm 105 MeV, e+ Al 1 mm, Mylar 2 mm; float64 and float32 | NumPy model | chi2/ndf of transmitted loss 0.58-1.30, exit angle 0.60-1.21, displacement 0.68-1.48; 1 X0 W stopped 18.21 vs 18.17 % (float64); photon and delta yields within 1 % except Mylar photons -0.8 % (float64) / -2.6 % (float32) | pass |
| `eslab_jax` with start depth, field and lane compaction vs NumPy `eslab`, 200k e-, float64: 0.1056 mm Al from mid-depth, isotropic, 1 T normal field (compaction 64); 1 mm Al, 1 T transverse; 1 X0 W, compaction 64, without and with 1 T transverse | NumPy model | loss chi2/ndf 1.05 / 0.57 / 1.05 / 0.76, exit angle 0.23 / 0.81 / 0.91 / 0.83, displacement 0.08 / 0.99 / 1.08 / 1.28; 1 X0 W stopped 18.21 vs 18.19 %, reflected 0.27 vs 0.28 % | pass |
| `eslab_jax` defaults with the new options present | outputs before the change | NumPy eslab and eslab_jax float64/float32 outputs byte-identical (sha256 over three cases) | pass |
| device helix face search over 16 revolutions, float64 and float32 | analytic first crossings and unreachable planes; 1 MeV e- in air, 0.481 T transverse field | first crossings found without a fixed scan cap; slab transmits at 5.166 mm path rather than reflecting after a later crossing | pass |
| slab starts on either face, inward and outward; NumPy e-/e+, device float64/float32 with and without compaction | no field, explicit zero field and 0.5 T normal field; msc off/on | outward starts exit with zero path and deposition, unchanged energy and position; inward starts transport | pass |
| device compaction chunk validation | zero, negative, fractional and boolean chunks | rejected with ValueError before compaction; positive integer chunks remain supported | pass |
| `eslab_jax` step factored out of the loop (`_slab_body`), per-lane layer types and opt-in unroll present | outputs before the change | `electron_slab` float64/float32 outputs byte-identical over 14 cases (plain, msc, e+, field, field without msc, start depth, compaction) | pass |
| `lane_step` with Al, polyethylene and W lanes interleaved in one loop, float64 and float32, msc with and without field, field without msc | `electron_slab` per material, same key | stopped, reflected, photon and delta counts identical per lane; energies, positions, directions, paths to rounding (1e-11 / 2e-5) | pass |
| `lane_step(unroll=True)` | `unroll=False` | same counts per lane, continuous outputs to 1e-11 | pass |
| muon `eloss_jax.sample_loss` with the electron options added | the sampler before the change | bit-identical on a 5 x 5 x 3 x 2 grid of materials, momenta, lengths and cuts, float64 and float32, arm64 and x86-64 | pass |

## Device precision

The CRV device kernels run in float32 by default (`MU2EFS_DEVICE_FLOAT`, 2026-11-01). Validated on 100 CRV shards on
V100 and MIG A100 slices: every aggregate within the float64 seed-to-seed spread; seq-0 pulse PE float32 - float64 =
-0.03 to -0.14 %, bounded at <= 0.3 % (2 sigma) per sector group (crv `docs/device-precision.md`); device tier-1b
sampler passes the deck rule at float32 (0 of 21 shapes flagged). Beam and tracker JAX outputs are bit-identical to
float64 (every jitted call hashed); float64 mode is bit-identical to the previous default.

Run-to-run reproducibility on the GPU (2026-11-01): XLA autotunes each fused kernel by timing candidates when it
compiles it, and the choice changed between processes (12-52 of about 1,400 kernels between two runs on an RTX PRO 6000
MIG slice, the full card and an A100), so CRV device output differed between runs of the same code and seed on every
card: float32 in transport, voltages, ADCs and cluster floats, float64 in digi voltages (1 ulp). With
`MU2EFS_DEVICE_DETERMINISTIC` (default on: `--xla_gpu_autotune_level=0`, `determinism.py`) separate processes are
byte-identical at every CRV stage: float32 on all three cards and a V100, float64 on the MIG slice. Cost on the MIG
slice (CRV chain warm pass): +0.8 % float32, +0.4 % float64, both within the run-to-run spread; the cold pass is 16-28 %
faster. `--xla_gpu_deterministic_ops=true` also repeats but slows the chain 1.7x. Across GPU kinds float values differ
and float32 discrete counts can differ slightly, so provenance records the precision, the determinism setting and the
device kind (crv `docs/device-precision.md`).
