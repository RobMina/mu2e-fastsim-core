# Development-only chronological helix surface isolation

`mu2efastsim.geometry.helix_contact_jax.isolate` is an **opt-in DEVELOPMENT
prototype**, not a navigator, production-compatible crossing callback, or transport
adapter. It uses the paired `helix_bounds_jax.enclose` / `evaluate` primitives.
Neither ordinary float64 padding nor a successful root bracket is certified.
**No result authorizes physics clearance or core transport.** No existing APIs,
transport defaults, or `eslab_jax` code are changed.

## Scope and call

```python
import jax
from mu2efastsim.geometry import helix_contact_jax as HC

search = jax.jit(HC.isolate, static_argnames=('max_work', 'max_depth'))
result = search(x, u, p, q, B, slo, shi, z, half, inner, outer,
                axis_x=0., axis_y=0., atol=1e-6,
                max_work=512, max_depth=48)
```

Enable JAX x64 before constructing inputs. Units, shapes, trajectory restrictions,
charge convention, global-z annulus, and finite frozen arbitrary-B assumption
are exactly those in [helix-bounds.md](helix-bounds.md). `x,u` have shape `(n,3)`;
`B` broadcasts there, with `None` meaning zero; other numerical inputs broadcast
to `(n,)`. Integer inputs are accepted; non-float64 floating, boolean, complex,
and nonnumeric inputs are unsupported, not silently upgraded. Shape/broadcast
errors raise `ValueError`. Some invalid Python values are rejected by JAX itself
before a jitted function can trace. The input direction is not renormalized.

`atol` is a positive finite float64 (or integer) interval-width target in mm.
`max_work` is a static nonnegative Python integer, and `max_depth` a static Python
integer in `[0,64]`; invalid limits raise `ValueError`. They must be marked static
when passed through `jax.jit`. Defaults are deliberately bounded, not a promise
that every supported helix can be resolved. A zero work/depth budget is allowed.

The prototype isolates **surface-function zero candidates**, not physical annular
boundary transitions. It searches the two entire planes and two entire cylinders:

| Mask index | Surface function |
|---|---|
| 0 | `x_z-z+half` |
| 1 | `z+half-x_z` |
| 2 | `outer**2 - ((x-axis_x)**2+(y-axis_y)**2)` |
| 3 | `((x-axis_x)**2+(y-axis_y)**2) - inner**2` |

With `inner=0`, mask 3 is disabled everywhere, including the starting-surface
check. Surface zeros are **not clipped to the annular volume**. A plane zero may
be beyond the outer radius; a cylinder zero may be above either face. There is
no inside/outside, entry/exit, tangent-transition, or material classification.
A tangent candidate is not skipped in favor of a later transverse root.

## Result contract

`ContactSearch` contains `(n,)` arrays `status`, `slo`, `shi`, `work` and boolean
`(n,4)` arrays `surfaces`, `monotone`. There is deliberately no scalar `distance`,
`crossing`, or `supported` callback contract and no infinity sentinel.

| Status | Meaning |
|---|---|
| `CANDIDATE = 0` | First unexcluded interval has width `<=atol`. Its boxes permit a surface zero; existence, uniqueness and tangency are unresolved. |
| `TRANSVERSE_BRACKET = 1` | A candidate with exactly one possible surface, strict opposite endpoint signs under singleton boxes and the paired point evaluator, and a derivative interval excluding zero. Under these engineering boxes it brackets a simple surface zero. |
| `NO_CANDIDATE = 2` | Every interval in the search domain has been excluded by the engineering boxes. This is **not certified clearance**. |
| `UNSUPPORTED = 3` | Input or computed boxes/evaluations are outside the primitive's support policy. |
| `PRECISION_UNRESOLVED = 4` | The earliest unexcluded interval is wider than `atol`, but its computed midpoint cannot strictly split it. |
| `WORK_LIMIT_UNRESOLVED = 5` | Node budget or depth budget exhausted before a decision. |
| `START_UNRESOLVED = 6` | A singleton enclosure at input `slo` includes zero for any enabled surface, including numerical padding uncertainty. |

For either candidate status, `slo,shi` are the returned closed interval; **never
interpret their midpoint as an exact root**. A `CANDIDATE` can be a padding or box
overestimation false positive, including an interval just before a tangency that
does not itself contain the true zero. The solver stops there rather than claiming
that an actually later zero is earliest. Multiple possible surfaces are retained
in the mask, with no arbitrary tie-breaking. Only `TRANSVERSE_BRACKET` asserts
zero existence/uniqueness conditional on the unvalidated engineering enclosures.

`surfaces` retains enabled functions permitting zero in either the deciding box
or the final reevaluation. `monotone` marks those possible surfaces whose final
derivative interval excludes zero; it
is **not** a contact or crossing flag. A monotone candidate can still lack a strict
endpoint sign change. Tangencies are not positively identified by this API.

For precision/depth exhaustion, the interval is the earliest unexcluded leaf.
For work exhaustion it is the next chronological pending node, which may not
have been tested yet and may have no possible surfaces on final inspection.
It must still be treated as unresolved. For start uncertainty, both endpoints
are the supplied `slo`. No derivative rule, side invention, or positional nudge
is applied to starting surfaces. Unsupported and no-candidate endpoints are
NaN, with false masks. Unresolved endpoints are finite, never infinity. `work`
counts visited nodes per lane, excludes preflight and final classification boxes,
and is at most `max_work`. Unsupported/preflight-start lanes visit zero nodes.

## Chronology and fixed-shape execution

A depth-first stack of capacity `max_depth+1` is allocated per lane. Each visit
computes `HB.enclose` on the top interval. A node is excluded only if **every
enabled surface interval excludes zero**. Otherwise it is returned when narrow,
reported unresolved at a limit, or bisected. The right child replaces the parent
and the left child is pushed on top. Thus every previously popped interval has
been excluded before the next chronological interval can be returned. Equal
endpoints, equal endpoint signs, and return to the initial position do not
exclude an interval. This includes full-turn leave-and-return trajectories.

The loop uses `jax.lax.while_loop` with a static work cap and fixed-shape stacks,
deterministic masked scatter updates, and lane-independent statuses. Completed
lanes do not advance while other lanes continue. There are no per-event Python
loops, dynamic lane compaction, random draws, or unbounded refinement. Memory
scales with batch size times maximum depth; work is bounded by the node cap.
Preflight and final full-shape evaluator calls are constant overhead. This is
not a performance-validated implementation or a differentiable root API.

## Numerical restrictions and transport prohibition

All exclusions and strict sign/derivative tests depend on HB's **uncertified
engineering padding**, not directed rounding. At the padding floor, an eager
box replay and a separately compiled/fused box evaluation can differ by ulps.
Deciding masks are preserved rather than erased by a later reevaluation. Tests
check ordinary eager/JIT agreement, but no universal bitwise, enclosure, or
cross-device guarantee is claimed. Arbitrarily small `atol` does not shrink the
numerical error floor: it can produce a false-positive candidate, precision
unresolved, or depth/work unresolved instead of a contact bracket. Long paths,
large phases/translations, thin features, near tangencies and persistent
surface-following motion can be expensive or unresolved. Initial surface-following
motion is start-unresolved. Field-map variation is not modeled.

**Transport integration is prohibited pending validated numerical error bounds
and legacy evaluator reconciliation.** In particular, shared use of HB's stable
position/raw-derivative arithmetic does not reconcile the cancellation-prone
legacy `eslab_jax._helix/_sc` evaluator or its separately normalized direction.
A future consumer must either use validated paired semantics or bound the
legacy discrepancy, establish outward numerical error on each backend, define
full volume/entry/exit/tangency/start conventions, and validate navigation and
field freezing. No fallback to legacy navigation or straight motion is supplied.
No unresolved status may be converted to a no-hit distance or physics clearance.

## Validation

`tests/test_helix_contact_jax.py` covers analytic straight faces/rims/holes, axial
full-turn leave-and-return for both charges, tilted-field analytic face contacts
and parallel motion, translated axes, nonzero search starts, disabled zero-radius
holes, zeros outside the volume, tangencies before later simple roots, starting
uncertainty, simultaneous surfaces, endpoint contacts, unsupported lanes,
precision/work/depth limits, curved tangencies and excluded full turns, neutral
tilted fields, mixed work-budget lanes, eager/JIT repeatability, disabled x64,
nonnumeric rejection and empty batches.
Ordinary cases replay dyadic left siblings against HB boxes to check earlier
exclusion; padding-floor tests deliberately do not assume bitwise agreement
between independent eager and fused JIT programs. These are regression checks,
not certified root completeness, numerical proof, or physics validation.

Run locally without installations or builds:

```sh
PYTHONPATH="$PWD" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MU2EFS_DEVICE_FLOAT=float64 \
  /Users/rmina/mu2e/.venv/bin/python3 -m pytest -q \
  tests/test_helix_contact_jax.py tests/test_helix_bounds_jax.py tests/test_annulus_jax.py
```
