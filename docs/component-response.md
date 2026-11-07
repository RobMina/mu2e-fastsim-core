# Component-local interaction replay (experimental, opt-in)

`mu2efastsim.component_response` is the shared NumPy starting point for geometry-portable interaction libraries.
It is not a production shower model, a complete geometry navigator or an adapter for a detector's existing bank.
No existing chain imports it. No trained bank or Geant4 physics validation is included.

## Boundary contract

A reusable interaction replaces transport **inside one finite component**, not the probability that a source
particle reaches the detector. The caller transports particles through the actual layout to the component boundary.
The bank returns all correlated internal deposits and all particles at their first outward boundary crossing.
Each exiting track is returned to external transport; a later re-entry is a new interaction. Do not also transport
or deposit the replayed interior explicitly. Detector light/charge/readout simulation remains downstream.

This first implementation supports field-free rectangular envelopes with identical component construction.
`ComponentType` carries a geometry/material version, physics/cuts version, local half sizes and local-volume count;
these labels must actually refer to the same construction and physics. It does not verify an external geometry file.
`Placement` adds a unique instance ID, a proper rotation, translation and local-to-global volume mapping.
The local envelope is centered at zero. Positions are mm, time ns, momentum MeV/c, total energy and raw deposits MeV.
Global points are `local @ rotation.T + translation`; vectors are rotated, never translated.
Boundary comparisons use a fixed 1e-7 mm position tolerance and 1e-12 direction-cosine tolerance; numerically
ambiguous grazing entries are unsupported. These are numerical tolerances, not tunable physics-matching windows.

A finite segmented component is not invariant under arbitrary internal shifts or rotations. Entrance state comprises
signed PDG, total energy, local position and full momentum. Reusing an exact local entrance state in a rigidly moved
copy is allowed; changing impact position, azimuth, energy, thickness, material or local field is not implicitly allowed.

## Array records

All tables are dictionaries of NumPy arrays with a shared first dimension. Three-vectors have shape `(N, 3)`;
`event` is the integer triplet `(run, subrun, event)`, not a bit-packed or floating-point label.

- Incoming particles: `event`, `track`, `parent`, `pdg`, `position`, `momentum`, `energy`, `time`. Track IDs are positive;
  `parent=-1` means no parent. Momentum is nonzero and total energy must be at least its magnitude.
- Bank entrances: `pdg`, `position`, `momentum`, `energy`, one row per complete independent trial. Identical entrance
  states may occur repeatedly, including zero-deposit trials. Positions lie on the envelope and point inward.
- Three offset arrays partition bank `particles`, `steps` and `exits` by entrance/trial. Every trial has at least
  the incoming particle as local particle0. Offsets for steps/exits may repeat, preserving empty outcomes.
- Bank particles: `pdg`, `parent`, `exited`. Local parents precede their children; particle0 has parent-1.
  `exited` is a boolean; false means the track ends inside the component. Every exited track has exactly one exit.
  This is first-exit capture, not a shower followed through the external detector and back into the module.
- Bank steps: `particle`, `volume`, `x0`, `x1`, `p0`, `p1`, `t0`, `t1`, `length`, `edep`. Particle and volume IDs are
  local, positions are inside the envelope, times are relative to entrance and energy is raw, not quenched/visible.
  The exporter must include relevant passive deposits if these rows are used for a complete energy ledger.
- Bank exits: `particle`, `position`, `momentum`, `energy`, `time`; positions lie on the boundary, momentum points
  outward and time is relative to entrance. Emission positions inside the component are not exits.

`Library` validates array shape, finite values, offsets, local ancestry, root species, envelope containment and
entry/exit directions. It checks geometry/physics compatibility before sampling. These checks do not establish
export completeness or Geant4 validity. The exporter must record stopped energy, material rest-mass exchange and
other terminal terms separately before an energy-closure claim is possible. Total incoming energy is **not** simply
sum(deposits)+sum(outgoing total energy) for all species/interactions without those terms.

## Replay and selection

`replay(library, placement, incoming, interaction, rng, next_track=...)`:

1. Transforms incident positions/momenta to local coordinates.
2. Matches the local entrance state to recorded trials, allowing only fixed floating-point roundoff tolerances.
   No nearest-bin, energy rescaling or internal step shifting is implemented. Unsupported inputs raise
   `UnsupportedEntrance` with their row indices **before consuming RNG**.
3. Uniformly selects one complete trial among identical-state trials. Each trial must have equal sampling weight
   at that state; importance-weighted or continuous-state interpolation models are future work.
4. Gathers its particle tree, steps and exits as one bundle; transforms all spatial/momentum fields and adds
   entrance time. It maps local volume IDs to the receiving instance's IDs.
5. Preserves the incoming track/parent identity for local particle0. Allocates unique child IDs starting at the
   caller's `next_track` cursor and returns the advanced cursor. The caller reserves that ID range against all
   other particles, including ones outside this batch. `interaction` is unique per event in this batch; reuse of
   a particle on a later visit is distinguished by its new interaction ID.

The result contains `particles`, `steps`, `exits`, selected trial indices and `next_track`. Event/interaction/instance
fields accompany all rows. A surviving primary keeps its track ID. New tracks keep their actual sampled parents,
including stopped/interacted ancestors. Null steps/exits are typed empty arrays, not omitted incoming trials.
Input arrays are copied on library/placement construction; replay does not modify the bank or incoming states.

`next_in_vacuum(particles, placements)` is a small **field-free vacuum-only handoff** for outside-to-box entries,
not a replacement for GDML material or field transport. It returns nearest forward entries and their source indices;
misses are absent, with no fabricated response. It adds relativistic flight time using supplied total energy/momentum,
leaves kinematics and identity intact, ignores outward/tangent starts on a face, permits immediate entry into an
adjacent touching box, and rejects starts strictly inside a component or ambiguous nearest entries. It does not
track the component interior or infer passive material between placements. Geometry overlap is the caller's concern.

## Tests and next slice

Synthetic tests establish software behavior: rigid covariance of the whole correlated bundle, nontrivial azimuth,
all particle/event keys, repeated trials and visits, empty outcomes, unsupported-state RNG behavior, proper rotations,
finite boundaries, volume mapping, and a two-component handoff with outgoing transport and changed placement.
They do **not** measure statistical agreement with Geant4. Run `python3 -m pytest tests/test_component_response.py`.

Next: reuse saved Geant4 products before considering new training. The opt-in
[`SavedG4TableDumper`](../art/README.md) exports retained particle states/ancestry, raw step collections,
trajectories, surface steps and CRV summaries without running G4. The initial inspected reference files contain
more than the legacy flat banks, but their trajectory thinning, particle compression and detector-event selection
prevent claiming complete exits or unbiased zero-response trials from those files alone. Export evidence first;
certify component records only where actual saved states and coverage support them. New simulation is not a
prerequisite for the extraction work. Purpose-built capture is being prepared separately in the
[component-training pilot](component-training.md); no training jobs have been submitted.

Then an opt-in detector adapter must remove double counting and validate one unchanged bank against nominal and
altered layouts. Finite-state support in this prototype is intentionally narrow; a useful continuous entrance-state
response requires a separately validated conditional sampler. Existing whole-detector step banks cannot acquire
missing boundary information through coordinate conversion alone.
