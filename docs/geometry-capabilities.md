# Geometry FileRefs and task/configuration capabilities

This slice extends the existing Mu2eBO evaluator contract. It is technical
admission and artifact provenance, not a commissioned physics search envelope.
No new optimizer, scheduler, measurement hash, or statistical-noise mapping is
introduced. No live UVA/EAF deployment is implied by the tests.

## Bounded geometry handoff

`beam_point` accepts exactly one `name: geom`, `kind: geom` FileRef in `files`
**or** `inputs`, with no `params.geom`. A selector and an artifact together are
refused, including `geom: nominal`. Existing selector-only submissions continue
unchanged. Other task/file kinds, multiple artifacts, and unsupported fields
are refused in both `check` and `submit` before dispatch.

```json
{
  "name": "geom",
  "kind": "geom",
  "uri": "file:///absolute/path/on/kit/host/geom.txt",
  "sha256": "<64 lowercase hex digits>",
  "metadata": {
    "format": "offline_simpleconfig",
    "schema_version": 1,
    "renderer": "offline_simpleconfig",
    "renderer_sha256": "<64 lowercase hex digits>"
  }
}
```

The URI names a file readable by the **kit process**, not a presumed endpoint
mount or arbitrary remote host. Only normalized local `file://` URIs (empty
host or localhost) are supported; no root/HTTP fetching, `/pnfs` traversal or
symlinks. The final input must be a bounded regular ASCII file, at most 256 KiB;
the kit checks the content digest and retains those exact bytes. Missing legacy
digest/renderer fields are refused, not guessed. Renderer metadata is a caller
provenance claim, not proof that a trusted renderer executed.

The file must start with the one supported base include:

```text
#include "Offline/Mu2eG4/geom/geom_common.txt"
```

The remaining statements may only be finite numeric literals in these types:

- `double stoppingTarget.{deltaZ,z0InMu2e,holeRadius}`
- `double protonabsorber.distFromTargetEnd`
- `vector<double> stoppingTarget.{radii,halfThicknesses,zVars,holeRadii}`

Blank lines and `//` comments are preserved. Duplicate keys, mistyped fields,
expressions, unsupported syntax, additional/nested includes, arbitrary text
payloads, GDML, unrelated detector/field/material changes, and a scalar hole
shadowed by a per-foil hole vector are refused. Positive radius/thickness/pitch
and nonnegative-hole guards apply. This is deliberately narrower than Offline
SimpleConfig. Beam's existing effective-overlay and foil-array/spacing checks
still apply at execution; admission does not certify arbitrary combinations.

The bounded render contains candidate configuration, not an upload channel for
Mu2e event/conditions data. It is retained in the durable request and job script,
then written read-only as `out/geometry.txt` on both routes. Output paths are
canonicalized before use (including UVA's `/scratch` alias). UVA uses this file
for both public dependency staging and beam transport; EAF passes it to its
existing beam runtime. No mutable caller path is consumed after binding.

The base/include closure, field maps, and EAF runtime are still site-resolved.
A render digest **does not** certify those dependencies, full-runtime immutability,
stock equivalence, or candidate ranking. See [artifact bindings](artifact-bindings.md).

## Identity, retry, results and dependencies

New opt-in requests save normalized params, geometry bytes/evidence, requested
metrics, resolved capabilities, target identity and the plain-route rendered
body/resources before endpoint dispatch, under a per-handle durable lock. Beam's
streaming binding additionally includes geometry in its existing manifest.
Retries compare identity and reuse stored bytes/body; they do not reread a deleted
or replaced caller file or rerender a changed template. Changing geometry,
requested metrics, parameters or endpoint under the same handle is refused.
Do not attach a new contract to a legacy submitted handle.

`out/geometry-artifact.json` records the same descriptor, byte count, admitted
parameter keys and explicit render-only coverage. Results verify both evidence
and exact retained bytes. For artifact-bearing beam calls, returned `files`
contains **only** the geometry FileRef, with its digest and metadata intact, so
Mu2eBO's existing `files_from` forwarding sends exactly the reusable artifact.
Fetched diagnostic references move to `metadata.diagnostic_files`; they are not
deleted or newly allowed for transfer. Selector-only calls retain legacy files.
Downstream steps report the same render digest even though their local URI differs.

The candidate content hash and renderer-definition hash remain distinct. Mu2eBO
hashes canonical `{geom, derive}` definitions for `renderer_sha256`; different
point values may change content without changing renderer identity. Neither
local URI nor per-point content digest is added to board `measure_sha`.

## Configuration-aware metric contract

`describe()` preserves legacy fields and adds `capabilities_schema_version: 1`,
`requested_metrics_supported: true`, and per-task metric descriptors/rules,
configuration metadata, route requirements, units and uncertainty notes.
`available` means produced on successful task completion, **not** physics-valid.
`conditional` means a data population or runtime input is needed. `unavailable`
means the selected task/configuration/route cannot supply the quantity.

`check` and `submit` accept optional `requested_metrics: list[str]`. Unknown,
duplicate, invalid or known-unavailable names fail before dispatch. Conditional
requests remain explicit and must be finite/non-sentinel when results arrive.
Examples:

- Beam warm throughput is unavailable with `repeat=0`; stop-profile quantities
  require stops, and negative-muon fraction requires that population.
- Chain-point EX/T1/T2/M labels belong only to extracted configuration 0.
  Non-extracted `clusters_per_generated_typeN` is a conditional name pattern,
  not a fabricated sector census. A missing type still fails if requested.
- CRV P3's excluded `dt_cluster_T2` term and `objective_all` are conditional;
  non-excluded terms use the frozen task manifest. Wait/cache metrics exist only
  on streamed routes. `tables=all` requires mounted FNAL inputs.
- C4 deadtime and chain reference comparisons depend on their optional inputs.
  A missing file under a locally visible configured root is negative evidence;
  an inaccessible root is unknown, not proof of endpoint absence. No remote
  probing, PNFS namespace traversal, or physics data reads occur for this check.
- `objective_sigma` is not automatically GP noise; beam binomial error is not
  independent-replica, shared-reference or model uncertainty.

For opted-in results, known unavailable or failed conditional *unrequested*
metrics are omitted and explained in `metadata.unavailable_metrics`, rather than
sending NaN or a missing-population sentinel through BO's finite-metric contract.
Requested missing/nonfinite/sentinel metrics still fail. Unexpected nonfinite
outputs and malformed always-available outputs fail; nothing is changed to zero.
Legacy non-opt-in results keep their existing shape. Metric formulas do not change.

EAF now exposes its deterministic GPU-count and configured server-lifetime bounds
through the shared pure resource-admission hook; live capacity remains dynamic.

## Mu2eBO integration and compatibility

The companion `fastsimkit` fork changes preserve/validate optional FileRef digest
and metadata on result round-trips, emit them from the existing renderer, retain
schema-1 descriptors, and apply statically known capability rules at launch.
Dynamic values are not replaced by guessed defaults. Each fresh step then calls
the opt-in resolved check before its durable submit record; requested names come
from objectives, extra metrics and downstream `params_from` reads. `check_study`
uses the same hook for independent steps and explicitly skips unresolved upstream
dependencies. Concurrent independent steps are not a transactional all-study gate.
Existing handles/completed records are adopted without new admission. Legacy kits
and adapters receive neither new arguments nor extra checks.

The new rendered-beam fixture is a technical example, not an authorized production
optimization. Deployment requires both compatible client and kit code. Continue
to honor the Slurm lost-acknowledgement and coordinated-rollout gates. No source
snapshot garbage collection, retention-policy expansion, or live run is part of
this patch.

## Verification record (2026-10-25)

- Local combined endpoint suite: **434 passed, 4 platform skips** (setsid/flock);
  isolated GPVM publication stack with current CRV configuration tests: **450
  passed, no skips**. Seven multiprocessing fork warnings remain.
- BO dependency-light selection: **269 passed, 1 skipped**. A broader local
  run could not exercise MCP/LangGraph because those dependencies are absent in
  the laptop analysis venv; no packages were installed.
- The isolated GPVM `ana 2.8.0` BO suite then passed **380 tests, no skips**, including
  MCP toy-server and LangGraph paths and six cross-package synthetic tests. A prior
  attempt exposed the new fixture's expected-list update (fixed); one scheduler
  cancellation race test failed intermittently, then passed in the final run.
- Reviews caught/fixed canonical output aliases, retry dependence on current
  templates, EAF lifetime check/submit parity and NaN placeholders incompatible
  with BO's finite-metric result contract. No live scheduler/GPU jobs or physics
  simulations, builds, installs or upstream PRs were performed.

The endpoint tests are the artifact-binding list plus `core/tests/test_geometry_handoff.py`
and `core/tests/test_capabilities.py`; on the GPVM add `core/tests` to PYTHONPATH
and `core/tests/test_crv_response_config.py`. BO cross-package tests opt in with
`FASTSIM_CODE_ROOT=<umbrella>` and `python3 -m unittest tests.test_fastsim_handshake_integration`.
For separated publication worktrees also put their core/CRV/beam paths on PYTHONPATH.
