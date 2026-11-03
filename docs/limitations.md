# Core: limitations

- **Electrons: slab physics, field stepping checked against itself and through the conversion-electron chain.**
  Electron and positron modules (`eslab` and its parts) are opt-in and validated in slabs against Geant4; field
  stepping (`ehelix`, `eslab(field=...)`, `eslab_jax(field=...)`) is checked against analytic helices, RK4 and the
  NumPy model, and end to end in the umbrella's conversion-electron chain against Geant4. Inside a slab the field is
  constant per track (taken at the entry or step start). The device stepper returns secondaries as per-track counts
  and energy sums. Open differences from Geant4: with the default form factor the reflected fraction in 1 X0 W is low
  (0.28 vs 0.35 %; `emsc.FF_TMIN = 100` closes it, opt-in); the transmitted-loss shape in 0.1 mm Al at 95 MeV
  (chi2/ndf 5.1, low side of the Landau peak) is Geant4's change of multiple-scattering model at 100 MeV, not the
  fast model (Geant4 95 vs 105 MeV differs there, fast 95 vs 105 does not).
- **No magnetic field** in the core muon transport: straight segments with multiple scattering, for the field-free
  extracted configuration. The beam stage has its own field-map stepping (umbrella `beam/`).
- **Muon radiative losses are off by default.** `transport.slab_loss` integrates ionisation only (the extracted-chain
  default); `slab_loss(..., radiative=True)` adds bremsstrahlung, pair production and photonuclear losses
  (`muradiative`, Geant4 11.3 differential cross sections, drawn at the slab-entry energy, none below 1 GeV kinetic).
  Its radiative b agrees with the published muon tables (Cu at 100 GeV 6.2e-6 cm2/g); in the tracker it moves the
  mean first-passage loss above 5 GeV/c by about 0.3 MeV/c. No device port.
- **Density effect of gases.** Materials without a NIST table use the solid branch of the Sternheimer
  parametrisation unless they are built with `state="gas"` (Geant4's gas branch). The tracker's straw gas is a gas in
  the Geant4 geometry; the solid branch puts the restricted ionisation of 100 MeV electrons in it about 2.4 % high and
  that of muons above about 10 GeV/c 1.5-2 % high (below 10 GeV/c the two branches agree). The default is unchanged.
- `eloss_jax` does not port delta-ray sampling (`sample_deltas`); the CRV transport has its own device sampler for it.
- The Geant4 match is to 11.3.2 with the production cuts of the extracted configuration; other cuts or Geant4
  versions are not validated.
- Device CRY: rare species (kaons) are not observed, so not validated; multiplicity and multimuon fractions are about
  two standard errors high in the one comparison made. Native CRY needs a separately built, licensed CRY 1.7 container
  extension ([native-cry.md](native-cry.md)).
- Device precision is float32 by default in the CRV kernels; a 0.1 % pulse-PE offset is bounded (<= 0.3 %), not removed.
  Device results repeat run to run on one GPU kind only with XLA autotuning off (the default
  `MU2EFS_DEVICE_DETERMINISTIC`), and are not bit-identical across GPU kinds (float32 discrete counts can differ
  slightly; `MU2EFS_DEVICE_FLOAT=float64` keeps the discrete results, with float values differing in the last bits). The
  setting is applied when `mu2efastsim` is imported; a process that creates a JAX backend before that runs with
  autotuning (`late` in the provenance).
- Endpoints: `chain_point` (streamed inputs) runs on Slurm sites only; on EAF the CRV task is `chain` (inputs read in
  place). The ESnet adapter is on branch `park/endpoint-esnet`. Admission and recovery are unit-tested; no endpoint
  rollout is implied by the tests. The kit host must reach the Slurm login host by ssh.
- Inputs: no collaborator input release is published; `inputs check-inputs` verifies a release, it does not provide
  one. The quickstart has not been run from a second user account.

## Studies kept off main

| Branch | Contents |
|---|---|
| `park/endpoint-esnet` | ESnet IRI v2 jobs and scripts, Globus helper, Polaris hello/env probes |
| `park/beam-loss-studies` | `eloss_split_jax` (slab-split loss study for the beam stage) |
