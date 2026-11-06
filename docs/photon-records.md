# Opt-in bremsstrahlung records

`mu2efastsim.eslab_jax.lane_step_photons` takes the same arguments as `lane_step`
and returns `(next_state, BremsPhotonRecord)`. It exposes the existing brems draw;
there are no extra random draws. The state-only entry point remains the default.

The fixed-shape record has one slot per charged-particle lane:

| Field | Shape | Meaning |
|---|---|---|
| `energy` | `(n,)` | Photon energy in MeV |
| `position` | `(n, 3)` | Emission position in the slab frame, mm |
| `direction` | `(n, 3)` | Unit photon direction before charged-particle recoil |
| `path` | `(n,)` | Accumulated parent path at emission, mm |
| `valid` | `(n,)` | A brems photon was emitted in this step |

All numeric fields are zero for invalid slots, including finished lanes. The
record can be carried through a JAX `scan` or buffered in a `while_loop` without
host-side compaction. Existing `nph` and `eph` counters include the same emissions.

The tests compare every next-state array byte for byte against `lane_step` over
multiple steps and seeds, for electrons and positrons, float32 and float64,
with and without magnetic fields and multiple scattering. They also check energy,
recoil direction, inactive lanes and a positron-annihilation-only control.

These are **birth records, not photon exit flux**. Annihilation photons, delta-ray
cascades, photon propagation and attenuation are not recorded or simulated by this
interface. A finite output buffer must report truncation separately; neither
missing slots nor unfinished charged-particle lanes represent zero emission.

## Optional complete-material boundary hooks

```python
next_state, photons = lane_step_photons(
    key, lt, mi, st, B=None, positron=False, msc=True,
    annihilation=True, mott=None, unroll=False,
    boundary=boundary, safety=safety,
)
```

`lane_step` accepts the same paired hooks and returns only `next_state`:

```python
next_state = lane_step(
    key, lt, mi, st, B=None, positron=False, msc=True,
    annihilation=True, mott=None, unroll=False,
    boundary=boundary, safety=safety,
)
```

`boundary` and `safety` are keyword-only, default to `None`, and must be supplied
**together** on either entry point. Supplying only one raises `ValueError` in the
Python entry point, before table/state conversion or tracing the transport body.
Both wrappers use the same underlying `_lane_step`; state-only transport selects
`return_photons=False` and does not allocate a photon record. Omitting both hooks
or explicitly supplying two `None` values retains the legacy slab path, random
stream and arrays. `electron_slab` is unchanged and has no boundary hooks.

Hooks are Python specialization choices, not dynamic per-lane switches: close
over them when tracing a JIT/scan/while-loop and reuse those closures. Geometry
arrays may be arguments to the outer jitted function. Separate state-only and
record-returning compilations may fuse floating-point operations differently;
finite-hook tests require integer/boolean state exactly and floating state at
tight numerical tolerance, rather than demanding cross-program byte identity.

Both callbacks must be deterministic, vectorized, JAX-traceable functions closed
over any per-lane geometry and frame information. They consume no random numbers:

* `boundary(x, d, p, q, B, smax) -> distance`: `x` and unit `d` are `(n, 3)`
  positions and directions in the slab frame, `p` is `(n,)` momentum in MeV/c at
  that leg, `q` is the scalar charge sign (`-1` electron, `+1` positron), and `B`
  is `(n, 3)` frozen field in tesla, or `None` for straight transport. `smax`
  is the nonnegative `(n,)` proposed path in mm. Return the earliest complete
  material exit in `[0, smax]`, or positive infinity if none occurs. This query
  **replaces**, rather than supplements, the legacy axial-face query: it must
  cover axial faces as well as every lateral boundary (including holes).
* `safety(x) -> distance`: return a nonnegative `(n,)` nearest-boundary distance
  in mm, considering axial and lateral boundaries. This replaces the planar
  distance used to classify range-trapped MSC tracks. Using planar safety near a
  rim could wrongly disable hard scattering and the normal MSC step cap; hence
  the paired-hook requirement, even when a particular call has MSC disabled.

MSC steps query the first leg at its actual start and direction, then the second
leg at the transported first-leg endpoint with its sampled MSC rotation. No loss
is applied between these legs, so their momentum is the same pre-loss momentum;
continuous loss remains pending until the existing loss point or a boundary exit.
For field transport the callbacks must intersect the same frozen-field helix as
`eslab_jax.helix`, including leave-and-return crossings, rather than test only the
endpoint. Boundary-start direction/curvature classification belongs to the
callback. Geometry tolerances must not be implemented by nudging the track.

Inactive lanes and second legs of first-leg-exited lanes receive zero proposed
paths. A callback may return zero there; its result is masked so it cannot cause
negative transport or change an inactive lane's state. Callbacks must handle
these full-batch rows without compaction. Proposed paths in the opt-in path are
clamped nonnegative, including a resumed state whose pending path already
exceeds its next loss interval.

An exit flushes continuous loss on the **previous pending path plus actual
traveled distance**, not on the untruncated proposal. Accumulated parent path and
position use the same truncated legs. Endpoint bremsstrahlung, delta production,
hard elastic scattering and in-flight annihilation are suppressed after an exit,
including a boundary coincident with the proposed endpoint. If the sampled
continuous loss stops the particle, stopping takes precedence over exit; existing
positron at-rest annihilation accounting is retained. Record shape, random key
splitting and random draw shapes do not change.

This is a geometry extension, not a replacement or validation of the general MSC
model: its preselected scattering point, soft-angle prescription and frozen-field
approximation remain unchanged. Core supplies the restricted detector-independent
[`geometry.annulus_jax`](geometry-kernels.md) query, not an integrated finite-foil
navigator, world-frame transformations, vacuum re-entry navigation or photon
transport. Its unsupported/unresolved results must be resolved or explicitly
failed before physics is committed, never silently converted to infinity or
passed as NaN callback distances. The caller must keep its actual per-foil
identity and local state authoritative.

The focused hook tests use synthetic cross-section tables, forced exits on each
MSC leg, and deterministic mean-loss controls to check actual-path accounting,
actual callback arguments, both charge signs, straight/helical transport, axial
and lateral exits, inactive rows, stopping precedence, interaction suppression and
near-rim safety. Null hooks are checked byte-for-byte; the existing photon tests
retain their multi-step state-only parity checks with physical material tables.
These unit tests do not establish finite-foil agreement with Geant4.
