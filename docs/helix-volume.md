# Development-only physical-volume-aware helix search

`mu2efastsim.geometry.helix_volume_jax` is an opt-in **DEVELOPMENT prototype**
for chronological boundary-candidate isolation in a finite annular volume. It
extends the left-first, fixed-shape search architecture of
[helix-contact.md](helix-contact.md), using the paired evaluator and enclosures
from [helix-bounds.md](helix-bounds.md). It is **not a full navigator**, production
crossing callback, or transport adapter. No result authorizes transport or
physics clearance. No `eslab_jax` defaults or existing APIs change.

The important difference from HC is **volume clipping**: a zero on an infinite
plane or cylinder is irrelevant when another enabled volume constraint is
strictly outside throughout that whole interval. Such an interval can be
excluded, allowing the search to reach a later physical boundary. Neither
endpoint-only clipping nor a root-point membership guess is used.

## Calls and supported inputs

```python
import jax
from mu2efastsim.geometry import helix_volume_jax as HV

volume = HV.enclose(x, u, p, q, B, slo, shi, z, half, inner, outer,
                    axis_x=0., axis_y=0.)
search = jax.jit(HV.isolate, static_argnames=('max_work', 'max_depth'))
result = search(x, u, p, q, B, slo, shi, z, half, inner, outer,
                axis_x=0., axis_y=0., atol=1e-6,
                max_work=512, max_depth=48)
```

Enable x64 before constructing inputs. Positions/distances are mm, momenta
MeV/c, fields tesla, and charges positron-charge units. `x,u` have identical
`(n,3)` shapes, `B` broadcasts to `(n,3)` (`None` means zero), and other numerical
inputs broadcast to `(n,)`. The field is constant along the whole trajectory
in each lane and may point in any direction. The annulus is aligned with global
z and centered at `(axis_x,axis_y,z)`.

HB's complete restricted trajectory and geometry domain, input dtype policy,
subnormal guards, analytic charge convention, and raw-derivative semantics apply
unchanged. Integer inputs are accepted; float32, boolean, complex, nonnumeric,
nonfinite, out-of-domain, and disabled-x64 inputs are unsupported rather than
silently promoted or approximated. Shape/broadcast errors raise `ValueError`.
Some invalid Python arguments may be rejected by JAX before the function traces.
The direction is not renormalized. HB's position/raw derivative is used directly;
it is not the legacy transport curve.

`atol` broadcasts to `(n,)` and must be positive, finite, normal float64 or an
integer. Unlike accepting a flushed-to-zero subnormal tolerance, this prototype
explicitly rejects it. It is a target candidate-interval width, not a distance
accuracy guarantee. `max_work` is a static nonnegative Python integer;
`max_depth` is a static Python integer in `[0,64]`. Booleans are not valid limits.
Invalid limits raise `ValueError`. The defaults are bounded resource limits,
not a promise of resolution. Zero work/depth is allowed.

## Inward-positive volume constraints and strict classification

Let `R2=(x-axis_x)**2+(y-axis_y)**2`. The surface-mask order is:

| Index | Inward-positive constraint | Enabled |
|---|---|---|
| 0 | `x_z-z+half` | Always |
| 1 | `z+half-x_z` | Always |
| 2 | `outer**2-R2` | Always |
| 3 | `R2-inner**2` | Only when `inner>0` |

`inner=0` disables the hole everywhere: classification, start checks, clipping,
search masks, and entry/exit conditions. Passing or starting on the axis is not
a hole boundary. HB still returns the fourth algebraic function, but HV ignores
it when disabled.

`enclose` returns `VolumeBounds(bounds, region, surfaces)`:

- `bounds` is the original `HB.HelixBounds`, including its support mask and NaNs
  on unsupported lanes.
- `region` is `(n,)`, describing the **whole supplied interval**, not its midpoint.
- `surfaces` is boolean `(n,4)`, retaining enabled zero possibilities not excluded
  by whole-interval clipping; unsupported lanes have false masks.

| Region constant | Value | Conditional meaning under HB's engineering boxes |
|---|---:|---|
| `REGION_UNCERTAIN` | 0 | Neither strict classification is justified. |
| `INTERIOR` | 1 | Every enabled constraint has `f_lo>0` throughout. |
| `EXTERIOR` | 2 | At least one enabled constraint has `f_hi<0` throughout. |
| `REGION_UNSUPPORTED` | 3 | The input/computed box is unsupported. |

The exterior test deliberately requires a **single constraint** negative over
the whole interval. A path can be exterior at every point but fail this test
because different constraints exclude different parts. Subdivision can help;
a sample-based union argument is not substituted. All inequalities are strict.
A zero-containing enclosure is uncertainty, not a proved boundary equality.

For each surface, the un-clipped possibility test is `f_lo<=0<=f_hi`. If a
constraint is strictly negative throughout, it cannot itself permit zero, so
it is necessarily an **other** constraint for every retained algebraic zero.
Thus clearing every surface mask in an `EXTERIOR` interval implements the desired
other-constraint clipping. Otherwise all possible surfaces are retained; ties
are never arbitrarily resolved.

## Search result and statuses

`VolumeSearch` contains `(n,)` arrays `status`, `slo`, `shi`, `work`,
`initial_region`, `domain_region`, and boolean `(n,4)` arrays `surfaces`,
`monotone`. There is no scalar `distance`, midpoint-root estimate, infinity
sentinel, or production callback contract.

`initial_region` classifies the input singleton `[slo,slo]`; `domain_region`
classifies the complete input interval. These are preflight diagnostics, not
post-contact material states. Invalid inputs, including invalid tolerances,
get `REGION_UNSUPPORTED` in both fields. `REGION_UNCERTAIN` does not imply that
a crossing exists. A whole-domain uncertain box may later be fully excluded
by subdivision. The final search `status` always governs; region diagnostics
cannot override an unsupported or exhausted search.

| Status | Value | Meaning |
|---|---:|---|
| `CANDIDATE` | 0 | Earliest unexcluded interval has width `<=atol`; contact existence/transition remains uncertain. |
| `ENTRY_BRACKET` | 1 | Candidate meeting the strict conditional outside-to-inside test below. |
| `EXIT_BRACKET` | 2 | Candidate meeting the strict conditional inside-to-outside test below. |
| `NO_CANDIDATE` | 3 | All search intervals excluded by strict interior/exterior engineering boxes; **not certified clearance**. |
| `UNSUPPORTED` | 4 | Invalid inputs or unsupported boxes/evaluations; never replaced by a clear result. |
| `PRECISION_UNRESOLVED` | 5 | Earliest unexcluded interval is wider than `atol` but has no strictly interior floating midpoint. |
| `WORK_LIMIT_UNRESOLVED` | 6 | Node or depth budget exhausted before resolution. |
| `START_UNRESOLVED` | 7 | The initial singleton permits a clipped physical boundary; no side is invented. |

Status integers are specific to HV, not interchangeable with HC constants.
For candidate/bracket statuses, `[slo,shi]` is a closed interval, **not an exact
root**. A candidate may be an overestimation or numerical-padding false positive,
even a small interval before the actual tangent point. Tangency is not positively
identified. Returning it instead of skipping to a later transverse crossing is
intentional. Near misses, persistent surface-following, endpoint contacts,
simultaneous surfaces, and uncertainty about other constraints may remain
`CANDIDATE` or unresolved.

For depth/precision exhaustion, the interval is the earliest unexcluded leaf.
For work exhaustion, it is the next chronological pending node, possibly not yet
tested. Its final mask can be empty; that does **not** turn exhaustion into
`NO_CANDIDATE`. At `max_work=0`, a non-start valid lane remains work-unresolved
even if preflight happened to classify the whole interval. For start uncertainty,
both endpoints equal input `slo` and `work=0`. Unsupported and no-candidate
endpoints are NaN, with false masks. Other endpoints remain finite.

`surfaces` unions the deciding and final reevaluation possibilities, so an
independent floating reevaluation cannot silently erase a tie. `monotone` marks
those retained possibilities whose final derivative interval excludes zero;
it is not an entry, exit, or contact flag. Completed lanes do not accrue further
work. `work` counts visited search nodes only, not constant preflight/final
checks, and never exceeds `max_work`.

### Conditional entry/exit, not general navigation

Only a `CANDIDATE` is promoted to an entry/exit bracket. All of these must hold:

1. Exactly one surface remains possible, including the deciding-mask union.
2. That surface remains possible in the final interval box.
3. Its derivative box is strictly positive (entry) or strictly negative (exit)
   throughout the returned interval.
4. The singleton endpoint boxes and the paired point evaluations both have
   strict opposite signs: negative-to-positive for entry, positive-to-negative
   for exit.
5. Every **other enabled** constraint has `f_lo>0` throughout the entire interval.
6. The interval/endpoint boxes and both point evaluations are supported.

Conditional on valid enclosures and the continuous frozen analytic helix, these
conditions give one unique simple surface zero and a corresponding physical
volume transition. Negative-to-positive follows the inward-positive convention.
They do not classify corners, tangencies, or one-sided starts. In particular,
starting on a boundary is left unresolved even if the velocity appears to point
inward or outward; no positional/time nudge or inferred side is used. An initial
infinite-surface zero is clipped instead if another enabled constraint is
strictly negative at the start.

## Chronology, implementation, and resource bounds

The implementation uses HB's scalar validation, `enclose`, and `evaluate`, and
reuses HC's compact stack/result carriers. It does not copy the helix equations
or introduce another trajectory evaluator. The search is the HC-style masked
`jax.lax.while_loop` with fixed `(n,max_depth+1)` stacks and a static work cap.
A visit excludes a node only if its volume enclosure is strict `INTERIOR` or
`EXTERIOR`. Otherwise it returns a narrow candidate, reports a limit, or replaces
the node by its right child and pushes its left child. Therefore all popped
intervals preceding the returned candidate were excluded before proceeding.

The search works from initially inside or outside without assuming either one.
Equal endpoints, equal endpoint signs, or a full-turn return are never evidence
that the intervening trajectory has no boundary. There are no per-track Python
loops, dynamic lane compaction, random draws, hidden state, unbounded refinement,
or fallback to straight/legacy propagation. Memory scales with lane count times
maximum depth. Runtime is bounded by node budget plus constant preflight/final
full-batch evaluations. Empty and heterogeneous batches are supported. This is
not a performance-validated or differentiable root API.

## Numerical limits and transport prohibition

All classifications, clipping exclusions, strict signs, and derivative tests
use HB's **uncertified float64 padding**. They are not directed-rounding interval
arithmetic or numerical certificates. The exact-arithmetic enclosure rationale
does not establish an outward floating-point bound for every backend/compiler.
Shared evaluator arithmetic avoids one source mismatch, not all numerical error.
Independent eager/JIT/fused evaluations can differ by ulps at the padding floor.
Preserving masks is conservative bookkeeping, not a numerical proof.

Small `atol` cannot overcome the padding floor. Large coordinates/phases, long
paths, thin geometry, near grazing, coincident constraints, and surface-following
can remain candidates, precision-unresolved, or work-unresolved. No unsupported
or unresolved status may become a no-hit distance, a clear result, or authorization
to transport. Field-map variation is not modeled.

**No transport integration is authorized.** Legacy `eslab_jax._helix/_sc` has
cancellation-prone arithmetic and separately normalized direction semantics;
these remain unreconciled and unchanged. Production use requires validated
outward numerical error bounds or equivalent certified arithmetic, explicit
trajectory-semantic reconciliation, full start/tangency/corner navigation
conventions, field-freezing validation, and end-to-end accuracy validation.
This change provides none of those certificates or integration changes.

## Validation scope

`tests/test_helix_volume_jax.py` covers analytic entry and exit at all four
surfaces; straight and curved trajectories; both charges; tilted-field face
contacts and field-parallel motion; neutral tilted fields; translated axes and
nonzero search starts; strict whole-interval interior/exterior classification;
disabled zero-radius holes; earlier extraneous planes, cylinders, hole zeros,
and hole tangencies before later physical entry; initially inside/outside and
one-sided boundary starts; outside endpoints enclosing an entry; full-turn
equal-endpoint leave/return; grazing and near-grazing before later crossings;
simultaneous surfaces and endpoint contacts; the singleton padding floor;
precision/work/depth budgets; mixed-lane independence; invalid inputs, dtype and
subnormal rejection; disabled x64; empty batches; and eager/JIT repeatability.
Ordinary cases replay dyadic left siblings against strict volume boxes to check
chronological exclusion. Padding-floor tests do not assert that a false-positive
candidate contains the mathematical contact.

These are regression checks, not numerical certificates, exhaustive root
completeness proofs, full-navigation validation, or physics validation. Run
locally from the core checkout, separately from beam pytest:

```sh
PYTHONPATH="$PWD" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MU2EFS_DEVICE_FLOAT=float64 \
  /Users/rmina/mu2e/.venv/bin/python3 -m pytest -q \
  tests/test_helix_volume_jax.py tests/test_helix_contact_jax.py \
  tests/test_helix_bounds_jax.py tests/test_annulus_jax.py
```
