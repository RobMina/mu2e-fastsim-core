# Frozen arbitrary-field helix evaluation and interval boxes

`mu2efastsim.geometry.helix_bounds_jax.evaluate` and `enclose` are opt-in
development primitives for future chronological earliest-root isolation. Neither
is a boundary-root solver, crossing callback, navigator, or transport integration.
No defaults or existing APIs change. No distances or clear-path verdicts are
returned. A supported box may be too wide to decide anything: **unresolved is
never clear**. In particular, equal endpoints do not imply an empty interior.

```python
from mu2efastsim.geometry import helix_bounds_jax as HB
point = HB.evaluate(x, u, p, q, B, s)
bounds = HB.enclose(x, u, p, q, B, slo, shi,
                    z, half, inner, outer, axis_x=0., axis_y=0.)
```

Enable JAX x64 before constructing inputs. Units are mm, MeV/c, tesla, and
positron-charge units. `x,u` have identical `(n,3)` shapes; `B` broadcasts to
`(n,3)` (`None` means zero), everything else broadcasts to `(n,)`. The field is
constant along each lane's complete interval and may point in any direction.
The annulus remains aligned with global z, centered at `(axis_x,axis_y,z)`.
There are no per-track Python loops, adaptive iterations, random draws, or
hidden state. Eager and `jax.jit` calls support heterogeneous lane batches.

`evaluate(x, u, p, q, B, s)` returns `HelixEvaluation(x, u, supported)`:
position `x` and **raw analytic derivative** `u=dx/ds`, each `(n,3)`, and a
boolean support mask `(n,)`. It does not normalize the returned derivative,
return surface values, or classify any crossing. `s` broadcasts to `(n,)`.
`enclose` uses the same private trajectory preparation and point arithmetic
at `m=slo+(shi-slo)*0.5`; the public evaluator is not a wrapper around the
existing transport evaluator. Shared expressions eliminate that known
arithmetic mismatch, but do not promise bitwise identity across separate
JIT programs, fusion choices, devices, or eager execution.

`HelixBounds` contains `x_lo,x_hi,u_lo,u_hi` of shape `(n,3)`,
`f_lo,f_hi,df_lo,df_hi` of shape `(n,4)`, and `supported` of shape `(n,)`.
For either API, every numeric output is NaN for an unsupported lane; valid lanes
are independent of invalid neighbors. Malformed shapes/broadcasts raise `ValueError`.
Nonnumeric, complex, boolean, and non-float64 floating inputs are unsupported,
not silently upgraded. Integer inputs are converted to float64. Nonnumeric
values can be rejected by an outer JAX wrapper before tracing this function;
the eager function returns unsupported for them. Disabled x64 returns
unsupported (its NaN arrays necessarily use the available float32 dtype).

## Exact-arithmetic mathematical bounds

Use the charge convention of `eslab_jax._helix`: with
`b=B/|B|`, `omega=0.299792458*q*|B|/p`, `v=(u·b)b`, `a=u-v`, and `w=u×b`,

- `u(s)=v+cos(omega*s)*a+sin(omega*s)*w`;
- `x(s)=x(0)+s*v+s*sinc(omega*s)*a+s*cosc(omega*s)*w`,
  where `sinc(t)=sin(t)/t`, `cosc(t)=(1-cos(t))/t`, with their zero limits;
- `kappa=0.299792458*|q|*|u×B|/p` is the constant acceleration norm.

For `m=(slo+shi)/2`, `h=(shi-slo)/2`, and `|t|<=h`, Taylor's theorem gives
`|x(m+t)-x(m)-u(m)*t| <= kappa*t²/2` and
`|u(m+t)-u(m)| <= kappa*|t|`. Consequently coordinate position half-widths
are `|u_i(m)|*h+kappa*h²/2`; an independent arc-length half-width is
`|u|*h`. Taking their componentwise minimum remains an enclosure. Direction
half-width is `min(kappa*h,2*|u|)`. The latter cap follows from constant speed.
These statements include straight, neutral, parallel, singleton, half-turn,
and multiple-turn intervals. There is no endpoint-only clearance shortcut.

Inputs whose norms differ from one by at most `1e-12` are accepted **without
renormalization**; the formulas and arc-length alternative use their actual
speed. Thus `s` is precisely arc length only for exactly unit input. This
avoids interpreting normalized directions as derivatives of an unnormalized
position curve. Specifically, `u_lo/u_hi` enclose the **raw analytic derivative**,
not the separately normalized direction returned by `eslab_jax._helix`.

These boxes target the analytic frozen helix evaluated by `evaluate`, not every
floating-point output of the existing transport evaluator. Its
`(1-cos(phase))/phase` calculation outside the small-angle branch can differ
from this stable midpoint formula by more
than the box padding near phase `1e-6`, even for a singleton interval. The new
`evaluate` avoids that cancellation by sharing the enclosure-center code,
including term grouping, sinc series, and half-angle cosc. The existing
`eslab_jax._helix/_sc` and all transport defaults are unchanged. A future
boundary consumer must use this paired evaluator, or explicitly bound the
numerical discrepancy to its actual trajectory. Shared source arithmetic is
not a certificate for either real-valued or floating-point enclosures.

For `R²=(x-axis_x)²+(y-axis_y)²`, surfaces are inward-positive, in order:

| index | function | derivative with respect to s |
|---|---|---|
| 0 | `x_z-z+half` | `u_z` |
| 1 | `z+half-x_z` | `-u_z` |
| 2 | `outer²-R²` | `-2*((x-axis_x)*u_x+(y-axis_y)*u_y)` |
| 3 | `R²-inner²` | `+2*((x-axis_x)*u_x+(y-axis_y)*u_y)` |

Squares of intervals containing zero have lower bound zero; otherwise the
smaller endpoint square is used. Upper bounds use the larger endpoint square.
Dot products sum the min/max of all four endpoint products. These box operations
ignore correlations, intentionally overestimating the ranges. With `inner=0`,
the fourth function is still returned algebraically, not a physical hole or
crossing classification. Axial functions have units mm, radial functions mm²;
derivatives have units 1 and mm respectively.

## Practical floating-point policy (not certification)

The exact-arithmetic inequalities above are mathematical guarantees. The
implementation evaluates them in ordinary float64, **not directed-rounding
interval arithmetic**, and does not provide a formal enclosure certificate on
all compilers/devices. It uses nested `hypot` norms, a fourth-order sinc series
for `|phase|<1e-4`, and `2*sin(phase/2)²/phase` for cosc. No field is deliberately
replaced by zero merely because it is small.

Bitwise checks, protected by an optimization barrier against JIT folding
into floating comparisons, reject an entirely subnormal field or charge
rather than accepting it as zero on the tested CPU backend. Both signs of
nonzero subnormal distance endpoints (`slo`, `shi`, or evaluator `s`) and
geometry scalars (`z`, `half`, `inner`, `outer`, `axis_x`, `axis_y`) are also
explicitly rejected, not silently zeroed or normalized. Those scalars must be
normal float64 values or exact zeros, in addition to the restrictions below.
Signed exact zeros are supported wherever those restrictions allow zero
(not for `half` or `outer`). The trajectory checks are shared by both APIs,
including neutral lanes and zero-field lanes; an out-of-domain input is not
excused by a zero factor elsewhere in the equations. Subnormal coordinate or
direction components, and very small field components mixed with ordinary
components, remain subject to the stated floating-point limitations.

Both APIs use the same restricted finite trajectory domain; `evaluate` checks
it as a singleton interval `[s,s]`. `enclose` additionally checks the interval
ordering and annular geometry. The complete enclosure domain is:

- each initial position and each geometry scalar bounded in absolute value by
  `1e8` mm; `half>0`, `0<=inner<outer`;
- `1e-6<=p<=1e9`, `q=0` or `1e-100<=|q|<=100`;
- `|B|=0` or `1e-100<=|B|<=1e4`;
- `0<=slo<=shi<=1e8`, `|omega|*shi<=1e4` radians (`0<=s<=1e8` and
  `|omega|*s<=1e4` for `evaluate`);
- unit-direction norm tolerance `1e-12` as described above;
- no nonzero subnormal distance endpoints or geometry scalars, of either sign.

`evaluate` has no geometry arguments, so it does not apply geometry restrictions.
No unsupported lane falls back to a straight approximation or existing transport
arithmetic. `supported=True` means the inputs and finite outputs meet this
engineering policy, **not** that the box is certified or a path is clear.

With `e=512*float64_epsilon` and `P=|omega|*shi`, speed is inflated by `1+e`
and evaluated acceleration is inflated to
`|omega|*(|w|+e*inflated_speed)*(1+e)`. Position padding is
`e*(1+|x_initial_i|+shi)*(1+P)` and direction padding is `e*(1+P)`.
Coordinate subtraction for radial intervals gets another
`e*(1+|axis_i|+max(|x_lo_i|,|x_hi_i|))`. Final axial-function padding is
`e*(1+|z|+half+max(|x_lo_z|,|x_hi_z|))`; radial-function padding is
`e*(1+R²_hi+inner²+outer²)`. Derivative padding is `e+direction_padding`
for axial functions and `e*(1+max(|radial_derivative_lo|,
|radial_derivative_hi|))` for radial functions. Any nonfinite output rejects
the lane. These are deliberately generous engineering error allowances, not
proved transcendental/roundoff bounds. Near domain limits or with translated
coordinates they can dominate a thin-geometry signal. Zero-width intervals
therefore still have nonzero numerical padding; do not discard it or replace
an unresolved interval by infinity.

A concrete numerical-floor regression uses `x=(0,0,0)`, `u=(1,0,0)`,
`p=q=1`, `B=(0,0,1e-4/K_FIELD)`, `slo=shi=1e8`, `z=0`, `half=1e-4`,
`inner=0`, and `outer=1e8`. The true helix stays at axial coordinate zero,
so both axial face functions are constantly `1e-4`. Yet the supported
singleton's axial function boxes cross zero: position padding alone is about
`0.114` mm. Subdividing the singleton reproduces the same boxes and cannot
resolve this numerical floor. This is an unresolved enclosure, not evidence
of a face contact, an unsupported input, or a clear-path certificate.

## Blockers before core transport integration

Sharing the evaluator is necessary but **not sufficient for certified bounds**.
The unresolved numerical blocker is a proved outward error bound for frame
construction, phase multiplication, polynomial/trigonometric evaluation, and
all interval/surface operations on each supported backend, or replacement with
validated directed-rounding interval arithmetic. Sampling cannot resolve it.
The raw analytic derivative is the mathematical target; the finite sinc series
and ordinary float operations are approximations, not a claim of exact automatic
differentiation of the implemented position expression.

Before boxes can authorize core transport, the consumer must also:

- choose the paired position/raw-derivative semantics and make any normalization
  at the transport interface explicit without silently changing the curve;
- implement chronological earliest-root isolation and refinement, with explicit
  unresolved/unsupported handling rather than clear-path fallbacks;
- define crossing, tangency, starting-surface, and inner-radius-zero behavior;
- validate integration, field-freezing assumptions, and navigation accuracy.

No root solver, transport adapter, or integration change is provided here.

## Validation scope

`tests/test_helix_bounds_jax.py` independently samples NumPy Rodrigues/direct-trig
helices, including position, direction, all four functions, and their first
derivatives. Tests include both charge signs, tilted and zero/tiny fields,
neutral/parallel motion, translated coordinates, singleton/narrow intervals,
half/full turns (including equal-endpoint leave-and-return), domain-edge
intervals, invalid lanes, float32 rejection, eager/JIT and batching. Sampling
is a regression check, not an exhaustive proof. No root distances, integrated
navigation accuracy, field-map transport, or detector physics are validated.

Tests also directly compare public `evaluate` positions/raw derivatives and
surface values derived from them against singleton and interval boxes. They
cover phases bracketing `1e-6` and the `1e-4` sinc threshold, opposite charges,
near-unit but unnormalized inputs within `1e-12`, signed zeros, supported tiny
fields/charges, both signs of subnormal distances and geometry scalars,
signed-zero acceptance and normal-value boundary controls, mixed valid/invalid
batches, the singleton axial padding floor, and eager/JIT calls. The small-phase position has an independent half-angle reference; the
nonunit straight derivative is checked for exact preservation. Neither these
tests nor shared arithmetic establishes a universal floating-point guarantee.
