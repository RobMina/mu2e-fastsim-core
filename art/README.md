# Saved G4 input extraction (opt-in)

`FastSimInputs` is a small muse package in this `art/` work area, separate from detector-specific dumpers.
`SavedG4TableDumper` extends the saved-product extraction approach without changing their schemas or defaults.
It **reads existing art files only**: no Geant4, geometry construction, digitization or reconstruction is scheduled.
The output is an evidence-preserving intermediate, **not a complete component-response bank**.

## Build and use

Use a backing release compatible with the input products. The initial reference compatibility target is
SimJob MDC2025ay / Offline v13_39_00. `FastSimInputs` must be a real directory, not a symlink. Set the backing
link in this work area before building; build products remain ignored. The user runs the full build:

```bash
cd "$CORE_CODE/art"
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
muse setup
muse build -j 32
```

After a successful build, start with one event and then a bounded pilot:

```bash
mu2e -c FastSimInputs/fcl/dump_saved_g4.fcl -s "$INPUT_ART" -n 1 -T "$OUTPUT_ROOT"
```

`probe_saved_g4.fcl` uses the existing `DataProductDump` module and does not require this plugin to inventory
an input. Save `config_dumper -P` output separately with the input filename and software versions. A branch
inventory establishes what exists, not its event-level coverage. Do not infer historical settings from current FCL.

`dump_saved_g4.fcl` selects the observed reference-process tags explicitly, including both compressed and full
`cosmicTimeOffset` generator collections. Adapt the tag lists after inventorying another sample. Missing configured
products throw by default. `allowMissing:true` is for inventory work only: missing products have `present=false`,
`product_id=0`, `count=-1`, distinct from a present empty collection. Omitting a tag means it was not requested,
not that the corresponding product is absent in the source.

## Schema 1

All tables carry signed int64 `run, subrun, event`. Collection payloads carry `product_id, tag, index`;
`index` is the stored collection iteration index, **not** the SimParticle key. Particle references always carry
both ProductID and int64 key. Null links use product 0 / key -1. No packed or floating identity is used.

| Table | Content and caveat |
|---|---|
| `events` | One row after each event is successfully exported, including empty events; collection and missing counts. `complete_boundary_records=false` always. |
| `collections` | One row per configured kind/tag/event with presence, ProductID and size. |
| `particles` | Retained lifetime start/end states, parent and generator links with availability, stage/volume/status/process, truncation, daughter links, track length/step count and ion excitation fields. Undefined end fields are NaN/-1, with `end_defined=false`. |
| `daughters` | Ordered retained daughter links; compression may already have removed others. |
| `generators` | All particles in each configured collection, without species or energy selection. Compressed and full collections are separate, not interchangeable indices. |
| `crv_steps` | Aggregated bar passages, start momentum vector and scalar estimated end momentum; **visible** deposit, not raw deposited energy. |
| `step_points` | Retained raw pre/post positions and momentum vectors, pre time/proper time, raw/nonionizing/visible deposits, length, volume and ending process. No post time or boundary flag is invented. |
| `trajectories` | Retained trajectory header and point count, including zero-point trajectories. |
| `trajectory_points` | Stored ordered position/time/KE samples; no invented momentum vector. |
| `surface_steps` | Stored aggregate endpoints, start momentum/time, surface ID/index and deposit. **Detector coordinates**, unlike the other spatial tables. |
| `status` | G4 status, original track count, SimParticle overflow and step-limit/field kills. |

Positions are mm, momenta MeV/c, total/kinetic/deposited energies MeV and times ns. Floating fields are exported
as double without claiming to restore precision lost in the persisted products. SimParticle momentum/time and
trajectory coordinates were stored at float precision. `StepPointMC::volumeId` is collection-dependent: it is
not generally a SimParticle physical-volume index. The exporter applies no geometry transformation.

Only available response particle pointers are accepted; trajectory map/payload identities must agree. Ancestor
and generator pointer availability is recorded without replacing missing identities. Different configured products
may legitimately reference one another; consumers must check closure against the exported particle collections.
A successful process exit and an event census/count check are required before using a file: a fatal exception may
leave earlier rows or partial output. Nothing in a successful export certifies complete simulation coverage.

## What can and cannot be reused

Inspection of representative saved Run 1A and extracted reference files found compressed SimParticles,
trajectories, CrvSteps, raw virtual-detector StepPointMCs, SurfaceSteps, straw/calo summaries and StatusG4.
The first exporter covers the entries above; straw/calo summaries and optional physical-volume tables are not
exported yet. The inspected Run 1A file has no physical-volume metadata in its run/subrun products. A matching
geometry manifest is still needed for component frames, containment and local volume IDs.

Their embedded configuration selects detector-active events, removes neutrals from the CrvStep producer, keeps
referenced particles plus ancestry, and stores trajectories with 15 mm thinning, at least 5 retained points and
initial momentum at least 50 MeV/c. The saved raw step collection is virtual-detector-only, not all material steps.
These are findings about those inspected files, not a claim about every saved campaign.

Consequently:

- More information can be recovered **without rerunning G4**, and this exporter makes it accessible to NumPy.
- Trajectory interpolation is a diagnostic approximation, not an exact component exit or complete daughter census.
- SimParticle lifetime endpoints are not matched boundary states; the writer can pair a final position with
  pre-terminal-interaction momentum/time. Do not treat that tuple as an outgoing particle automatically.
- CrvStep visible deposits cannot be relabelled raw deposition; outgoing momentum direction is absent.
- The full-generator companion preserves generated-event states, but not the missing response of rejected events.
  Neither an absent particle nor absent steps certify a zero-response component trial.
- Existing raw boundary states may be used where geometry, first-exit ownership and collection coverage can be
  established. No automatic conversion to `component_response.Library` is provided or admitted by this schema.

## Validation and audit

The initial user build succeeded against the stated backing release. A one-event smoke test and three 100-event
pilots (Run 1A reference, extracted reference and extracted production donor) completed. All 17 common CrvStep
fields match fresh exports from the detector-specific dumper at their persisted precision, including particle
ProductID. Matching NaNs are preserved and reported, not repaired: the three pilots have 10, 63 and 78 nonfinite
CrvStep end times respectively. This is an input property, not introduced by this exporter.

Run the vectorized census/identity audit and optional legacy comparison on a bounded export:

```bash
python3 -m mu2efastsim.saved_g4 "$OUTPUT_ROOT" --legacy "$LEGACY_ROOT" --output "$AUDIT_JSON"
```

The audit checks completed event rows, configured collection sizes, integer identities, response-to-particle links,
trajectory headers/point indices/counts and retained parent/daughter/generator closure. It reports nonfinite fields,
missing versus empty products and retained per-species coverage; it never certifies a component-response bank.
Both Python tests and actual missing-product smoke tests exercise fail-closed behavior: strict extraction fails,
inventory mode records `count=-1`, and the auditor rejects the partial strict-failure output.

The pilots confirm the limitation: 831/844, 1984/1989 and 2228/2236 CRV-depositing electrons lack trajectories.
All retained links resolve, but compression still prevents claiming complete outgoing showers. Raw VD rows and
SurfaceSteps in these pilots do not cover CRV module boundaries. Continue inventorying richer saved source tiers
before admitting component records or proposing a new simulation campaign. No interpolation or missing-as-zero
fallback is enabled.
