# Component-training capture pilot

Status: bounded capture smoke and isolated CRV construction checks pass; no training campaign submitted. The
2026-11-06 user build includes the module tool and shared electronics-material helper. Nominal and transformed
finite-box fixtures each pass one-event and 100-event strict audits; the v10 R1 module now passes nominal and
transformed one-event strict audits and runtime inventory checks. One-event geometry covariance is not proof of
physics-distribution equivalence. All records remain bank-inadmissible. Existing physics banks are unchanged.

## Integration

The reference Mu2eG4 cut factory is a closed set of built-in predicates. `type: tool` delegates to an art class tool
implementing `IMu2eG4Cut`, with a nested `tool` parameter table. An independent optional `IMu2eG4Observer` interface
allows the **top-level common cut** to receive classification and pre/post-tracking callbacks. Built-in cuts do not
implement this interface. It does not alter the base cut interface/vtable or change the built-in cut decisions.

Use one recorder as the top-level `Mu2eG4CommonCut`, with empty stacking-only/stepping-only cuts. Do not bury it inside
union/intersection: their existing short-circuit semantics do not forward lifecycle observations. A recorder must
implement the complete event lifecycle, including cleanup of internally rejected events. Constructors must not book
ROOT trees: declaration-only temporary cut instances are also constructed. Prefer event products and an analyzer.

`classifiedTrack(track, killed)` runs after the stacking decision, including when the stacking-only predicate rejects
the track. Classification can occur more than once; IDs may not yet be final for rejected tracks. Do not treat this as
a unique birth census or manufacture art pointers for tracks never tracked. `beginTrack` follows SimParticle-start
creation; `endTrack` runs before SimParticle-end serialization and sees the actual G4Track state. Suspension and
postponement are not terminal deaths. Preserve real G4 information before a first-exit artificial kill changes it.

## Recorder target contract

The prepared schema-1 implementation records diagnostic tables but always sets `bankAdmissible=false`. It rejects
unsupported reclassification/resumption, boundary-born ownership and resource failures rather than silently losing
trials. Geometry/physics tags are labels, not the final hashed manifests; isotope/cutoff/energy accounting and
boundary-born ownership remain open. The isolated v10 R1 construction is runtime-checked, not a neighboring-module
partition. Compilation, tool loading, nominal/transformed transport and strict audits pass.
The companion `mu2efastsim.capture` auditor checks completed event counts, identities, boundaries and rigid transforms;
synthetic tests cannot establish physical completeness. The nominal 100-event box run retains 7,564 tracks/15,437 steps;
the independent-seed transformed run retains 7,111/14,544. Forced capacity overflow aborts; stacking-only rejection
reaches the common observer and yields an invalid checkpoint. Both failed outputs are rejected by the strict auditor.
These tests do not exhaust default-cut, suspension, zero-step and resource behavior.

Dedicated box photon transmission and mu+ stopping/decay pass strict audits, preserving physical zero outcomes and
at-rest children. Three-event stacking rejection and internal-discard fixtures pass the separate
`mu2efastsim.capture_lifecycle` negative checker while ordinary audit rejects every invalid file. Default art refuses
missing declared products; only the diagnostic internal-reset fixture disables that guard to inspect absent products.
The stricter positive auditor checks initial/nonfinal track statuses; all eight saved positive files still pass.
Combined local/GPVM04 tests now total 373. See `ComponentCapture/fcl/lifecycle.md` for fixture contracts and limits.
Consecutive discard success is not proof of reject-then-accept recovery or an allocation/leak measurement.
The auditor's conservative secondary-birth margin is its configured `atol`, not the G4 surface tolerance; a rejection
must stop the sample, never select surviving trials. Boundary-born ownership and actual zero-step/suspension fixtures
remain open. Box lifecycle coverage is not broader neutral/stopping coverage for the canonical CRV module.
The verified test-only plugin build now passes four-event reject/accept/reject/accept sequencing and all 13
constructed-object callback cases; see `ComponentCapture/fcl/callback-fixture.md`. Both accepted transport events
pass the positive audit; both rejected events have the exact rejection signature. Callback event 12 is explicitly
discarded after the expected resume exception, then event 13 reuses IDs cleanly. Dedicated vectorized checkers assert
the whole declared census and never admit donors; 632 combined local/GPVM04 Python tests pass. Ragged absent geometry
has an explicit ordinary-audit rejection, with no invented padding. Constructed callback records have no StatusG4
and are not physical transport evidence. The recorder and production hooks remain unchanged; engine scheduling,
Mu2eG4 zero-step finalization and internal-filter reject-then-accept recovery are not established by these fixtures.
The frozen `ComponentCapture/fcl/zero-step-engine.md` baseline retains two real-engine photon lifecycles with explicit
status injection. Lifecycle and exact persisted-product readback pass, but the baseline SimParticles lose the actual
10 MeV endpoint kinematics. The subsequent zero-step-only helper correction now passes its full user build and all
predeclared bounded checks; see `ComponentCapture/fcl/zero-step-fix.md`. All 16 constructed helper cases pass C++
byte equality/endJob census and the independent latest-cycle `capture_end_kinematics` checker. The unchanged
engine/readback inputs now retain KE/E 10 MeV and pz 10 MeV/c in both finalized SimParticles, with zero endpoint
deltas and exact persisted-field equality. Callback user information still has its original defaults. Ordinary
diagnostic output was saved before the separate opt-in `capture_zero_step --require-endpoint-match` acceptance.
Both upstream-never-entered histories remain invalid under ordinary audit, never physical zeros. Unchanged one-event
photon/stopping capture regressions pass `capture_compare --exact` against saved baselines, excluding only CPU/real
timing values. That comparison does not claim SimParticle endpoint coverage absent from those saved files.
The combined 1217 Python tests pass locally and on GPVM01. Genuine secondary scheduling, boundary ownership,
internal-filter reject-then-accept and broader bank-admission gates remain open; no campaign or certification follows.
The next `ComponentCapture/fcl/secondary-zero-step.md` diagnostic is source-prepared, not built/run: two separate
one-event modes retain the stopping seed/input and stop every engine-created child before stepping, at classification
or after recorder beginTrack. It reuses the saved primary/birth census and separate SimParticle readback. Artificial
terminal states can satisfy structural closure, but never constitute physical responses or change bank admission.

`python3 -m mu2efastsim.capture_compare REFERENCE CANDIDATE --expected-events 100 --output REPORT` audits both
inputs then reports per-event diagnostic distributions including zero-response outcomes. Mean component deposits
are 21.085 +/- 0.437 and 20.333 +/- 0.348 MeV; the 1.35-standard-error difference is not an equivalence certificate.
Local incident-state agreement is checked separately. Tags are not verified manifests and seed independence cannot
be inferred from the ROOT tables. Preserve finite-density vacuum approach deposits separately from component energy.

- Event-local buffered records with schema, component/physics manifest hashes, incident census and completion marker.
- Full integer source/event/track/parent identities; stable identities for any pre-tracking rejection need explicit handling.
- Every in-component G4 step: pre/post positions, momenta, total/kinetic energy, global/proper times, raw/nonionizing
  energy deposit, length, material/placement identity, defining process and step status. Include zero-deposit steps.
- First outward boundary crossing: exact post-point state, distinct exit reason, then stop only that branch. Do not
  kill legitimate already-produced children merely because their parent exits.
- Natural terminal states, no-step tracks and artificial/resource kills classified separately. At-rest physics remains
  enabled. Record secondaries created at the boundary with an explicit ownership convention.
- Any overflow, missing ancestry, unsupported classification or resource kill invalidates a training trial; never
  silently truncate or change it into a physical no-response outcome.
- No detector selection, positive-deposit threshold, neutrals filter or ancestor-only particle compression.
- No blanket deposit-equals-incoming-energy check: record material rest-mass/excitation and cutoff limitations explicitly.

The boundary must be a real finite G4 envelope; a coordinate-plane predicate alone can overshoot it. Membership
includes daughters and uses physical placement/touchable ancestry, not only logical-volume names. The record must
include passive components and interior air/gaps, not just scintillator. Outside transport is not part of the trial.

## Geometry and validation order

The bounded construction and remaining gates are detailed in [canonical CRV component](crv-component-construction.md).
Nominal v10 R1 module 0 retains 90 tracks/202 steps (36.961493634894936 MeV component deposit); its transformed
one-event trial retains 86/199 (33.59866941329343 MeV). Both pass strict audits, with 82 volume rows including two
vacuum ancestors, 80 component placements and 64 bar identities. All overlaps pass. Local entrance agreement is
8.6e-14 mm and 5.7e-14 MeV/c; response-distribution equivalence remains untested for this module. CMB omission
(0 placed/128 omitted) is source-faithful; this is not KPP or Run 1A geometry. The vectorized
`mu2efastsim.capture_inventory` audits before reporting inventory and primary traversal; 203 combined local tests
pass, including 18 new inventory tests. The nominal primary crosses four bars but misses FEBs; a separate FEB
fixture passes one-event strict audit (129 tracks/272 steps) and traverses FEBs 3/0. Exported GDML records their
expected density/element fractions. Standard-material initialization also confirms densities/composition but rejects
all primaries, so that initialization run is not physics capture or pre/post-extraction runtime A/B evidence.

1. **Recorder smoke geometry:** simple finite box in a vacuum world to test exact entry/exit, backscatter, stopping,
   neutral transmission and empty response. This is a software fixture, not a CRV module bank.
2. **Canonical CRV module:** reproduce a specific module from the pinned Offline construction, including all layers,
   absorbers, strongback and cover. Verify placement/material inventory and local sensitive-ID map against the source.
   Do not substitute a uniform slab or the union of scintillator bounds. `constructCRV` builds whole configured sectors
   and some passive structures outside their mother; copying only that mother is not sufficient.
3. **Geometry covariance:** translated/rotated versions, side orientations included, under the same local field-free
   conditions. Compare complete response distributions using independent trials; exact-state replay is a software test.
4. **Neighbor test:** explicit transport of recorded exits to another component, including returns as new interactions.
   Compare with full G4 and test for double counting.

The proposed small validation matrix is in [`training/component-pilot.json`](../training/component-pilot.json);
it is intentionally not launchable yet.
Repeated fixed states validate the current exact-state library API. A continuous empirical conditioning law and broad
physics coverage remain separate work, not something this pilot promises to establish.

## UVA execution gate

Use CPU Slurm jobs, one G4 process/thread per initial job; no GPU request. Start with a completed 100-event capture
smoke test, then the bounded matrix. Measure actual wall time, peak memory and bytes/event from finished jobs before
choosing production shard sizes; do not extrapolate total runtime from the smoke test.

Before dispatch: user build verified; capture/geometry checks passed; matching runtime image and custom libraries
staged with checksums; output path unique; seed/run/subrun mapping recorded; no existing job with the same request.
The current saved-product-only image/library is insufficient for new capture. Preparing or running a muse tarball
requires separate approval. Neither setup scripts nor worker scripts may build software. No batch worker or submit
command is supplied until the recorder and geometry gate pass, to avoid launching unrecorded or mislabeled G4 jobs.
