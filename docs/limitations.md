# Core: limitations

- **Electrons: slab physics, field stepping checked against itself only.** Electron and positron modules (`eslab` and
  its parts) are opt-in and validated in slabs against Geant4; field stepping (`ehelix`, `eslab(field=...)`) is checked
  against analytic helices and RK4, not yet against Geant4; the device stepper `eslab_jax` has no field, returns
  secondaries as per-track counts and energy sums, and keeps finished lanes in the loop (no compaction: the thick-W
  case runs ~3500 iterations); no conversion-electron or cosmic-fake chain yet. Open differences from Geant4: the
  reflected fraction in 1 X0 W, and in 0.1 mm Al at 95 MeV the transmitted-loss chi2/ndf of 5.1 (quantiles q10-q90
  within 0.3 %, yields within 1 sigma; not understood). Inside a slab the field is taken at the start of each step.
- **No magnetic field** in the core muon transport: straight segments with multiple scattering, for the field-free
  extracted configuration. The beam stage has its own field-map stepping (umbrella `beam/`).
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
