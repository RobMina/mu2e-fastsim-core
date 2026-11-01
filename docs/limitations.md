# Core: limitations

- **Electrons: slab physics only.** Electron and positron modules (`eslab` and its parts) are opt-in and validated in
  slabs; no field stepping, no JAX version, no conversion-electron or cosmic-fake chain yet; thin-layer exit angles at
  105 MeV and the stopped/reflected fractions in 1 X0 W differ from Geant4 (`docs/validation.md`).
- **No magnetic field** in the core transport: straight segments with multiple scattering, for the field-free
  extracted configuration. The beam stage has its own field-map stepping (umbrella `beam/`).
- `eloss_jax` does not port delta-ray sampling (`sample_deltas`); the CRV transport has its own device sampler for it.
- The Geant4 match is to 11.3.2 with the production cuts of the extracted configuration; other cuts or Geant4
  versions are not validated.
- Device CRY: rare species (kaons) are not observed, so not validated; multiplicity and multimuon fractions are about
  two standard errors high in the one comparison made. Native CRY needs a separately built, licensed CRY 1.7 container
  extension ([native-cry.md](native-cry.md)).
- Device precision is float64; float32 is validated on a branch only.
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
