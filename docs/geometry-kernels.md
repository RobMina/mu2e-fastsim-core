# Detector-independent geometry kernels

`mu2efastsim.geometry` owns reusable geometric queries, not detector layouts or
material navigation. Importing the namespace does not import JAX; opt in with
`from mu2efastsim.geometry import annulus_jax`. Detector packages own geometry
parameters, coordinate transformations, foil identities, vacuum navigation and
handoffs. No source-selection cuts or physics tables belong in this package.

## Restricted annular JAX kernel

```python
from mu2efastsim.geometry import annulus_jax

result = annulus_jax.crossing(
    x, u, p, q, B, smax, z, half, inner, outer, axis_x=0., entering=False,
)
distance = annulus_jax.safety(x, z, half, inner, outer, axis_x=0.)
```

The preserved public APIs are:

* `Crossing(distance, surface, supported)`, a named tuple of lane arrays.
* `crossing(x, u, p, q, B, smax, z, half, inner, outer, axis_x=0., *, entering=False)`.
* `safety(x, z, half, inner, outer, axis_x=0.)`.

Positions `x` and unit directions `u` have identical `(n, 3)` float64 shapes;
other parameters broadcast to `(n,)`, except `B`, which broadcasts to `(n, 3)`
or is `None`. Enable JAX float64 before use. Lengths are mm, momenta MeV/c and
fields tesla. The annulus has axis `(axis_x, 0)`, axial center `z`, positive
half-thickness `half`, and radii `0 <= inner < outer`. Momentum must be positive,
path limits finite and nonnegative. `q` is the signed charge in positron-charge
units. `safety` gives the nonnegative nearest face/rim/hole distance for inside
starts; it is not a separate supported-domain validator.

`crossing` returns the earliest complete material exit, or vacuum entry when
`entering=True`, in `[0, smax]`. Surface codes are 0 for no crossing/unsupported,
1 lower axial face, 2 upper axial face, 3 outer rim, and 4 inner hole. It handles
straight legs and frozen **axial fields only**, including radial leave-and-return
crossings, both helix phases and entry after axial admission. Boundary starts use
one-sided direction/curvature classification, not position nudges. Tangent touches
are not transitions; motion confined to a surface remains in the closed volume.

A clear, supported query returns `distance=+inf`, `surface=0`, `supported=True`.
Unsupported or numerically unresolved queries instead return `distance=NaN`,
`surface=0`, `supported=False`. Tilted fields, invalid inputs, incompatible start
regions, nearly repeated roots within the roundoff bound, nonzero curvature
outside `[1e-150, 1e100]/mm`, and proposed phase above `1e6` rad are not clear
paths. This is not an arbitrary-field or arbitrary-coordinate-range solver.

**Never turn unsupported/unresolved results into infinity.** Resolve/refine the
geometry decision, or stop with an explicit failure, before committing physics.
Passing an unresolved NaN directly to a transport callback also violates its
contract. Merely discarding a photon born outside material does not repair an
incorrect parent trajectory.

## Transport and JAX specialization

Both `eslab_jax.lane_step` and `lane_step_photons` accept paired, keyword-only
`boundary=None, safety=None` hooks. The boundary callback covers the complete
material volume on each actual straight/helical MSC leg; the safety callback
covers the same volume. See [photon-records.md](photon-records.md) for signatures,
inactive-row handling, actual-path loss and stopping precedence. No automatic
annulus adapter is installed: callers must establish the solver's supported
domain and handle failures before physics is committed. `electron_slab` remains
the legacy slab interface.

Callbacks are deterministic, vectorized, JAX-traceable closures with no random
draws. Close over geometry and transformations, or pass geometry arrays through
a jitted outer function. Hook presence, callback identity, `entering`, the
recording choice and other Python transport options specialize the traced
program; they are not per-lane dynamic switches. Reuse closures rather than
creating fresh jitted functions in a hot loop. The state-only wrapper selects
`return_photons=False` on the same underlying path and allocates no birth record.
Separate state-only/recorded JIT programs may fuse floating-point expressions
differently: compare integer/boolean state exactly and floats at tight numerical
tolerance. Disabled/default paths retain bytewise parity, including RNG-driven
state arrays.

## Validation scope

The 52 imported annulus tests retain independent straight-ray and circle
references, boundary starts, tangencies, no-hole geometry, both charges, full
turns, root-roundoff regressions and unsupported-domain checks. A frozen NumPy
straight-ray reference is test-only; core production code has no beam dependency.
Focused transport tests also compare finite state-only and record-returning steps
with synthetic rates and physical Al/W tables. These validate component contracts,
not integrated detector navigation, general finite-foil accuracy, shower transport,
photon exit flux or agreement with Geant4.
