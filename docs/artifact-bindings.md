# Streamed-task artifact bindings and retained code

This bounded slice covers the Slurm streaming routes `chain_point`,
`crv_response`, and `beam_point`. It extends the existing evaluator and Slurm
submission fence, not Mu2eBO's scheduler or measurement hash. Plain tasks and
EAF are not upgraded by this slice. No live rollout or physics validation is
implied by the fixture tests.

## What is pinned

After parameter/resource admission, a new handle resolves:

- A fresh public-code clone under `<configured-code>.snapshots/<uuid>`, detached
  at the exact requested umbrella commit with the selected recursive submodule
  commits. The optional CRV branch override resolves exact core/CRV commits.
  Branches, tags and commit IDs are locators; the resolved commits are evidence.
- Each streamed file's source locator/argv, destination, size, MD5 and SHA-256.
  These are **pinned expectations**, not snapshots of the mutable FNAL source.
  Replacement after resolution must match the saved expectations or fail.
- The runtime SIF's SHA-256 and size at its configured path. This is a preflight
  observation, **not an image snapshot or a lifetime immutability guarantee**.

The version-1 manifest records coverage and `full_immutability: false` explicitly.
Its canonical digest identifies execution evidence only. It is not installed as
BO's board-level `measure_sha`, and the logical kit version is not changed. A beam
candidate's streamed geometry is included in its execution evidence without
forcing every design onto a new measurement board. Selecting a compatible
measurement subset and consuming these descriptors in BO remain future work.

## Prepare, bind, dispatch and resume

A per-handle POSIX lock and durable JSON write save the original normalized
parameters, target/source identity, manifest, complete rendered body and resources
**before** calling the endpoint. A binding digest detects damaged saved state.
A matching retry uses this original rendering and does not resolve today's code,
SIF, inputs or templates. Different parameters or target/source configuration
are refused. The existing Slurm intent then binds that body and retains its
no-blind-redispatch rule after an ambiguous outcome.

Source snapshots are atomically published after verification and made read-only.
No existing code checkout is reset or modified. Site filesystem aliases such as
UVA's `/scratch` are resolved at preparation; the resulting canonical snapshot
path is retained. Internal/escaping snapshot symlinks, dirty/staged/untracked
(including ignored) content, missing pins and changed executable modes fail
verification. The job verifies source bytes against Git objects before runtime.
Git, Python 3 and Bash are required on the endpoint outside the container.

There is **no automatic snapshot garbage collection**, including after failed
preparation, a lost preparation reply, cancellation or ambiguous submission.
A controller exit or missing scheduler record does not prove an artifact unused.
Plan storage/retention and inspect all bindings/intents before manual cleanup.
Read-only modes and launch-time guards protect cooperating workflows, not against
owner/admin modification after verification.

Legacy task records remain readable for status/cancel and completed-result
collection, labeled `legacy-unverified`; they are not silently upgraded.
Resubmission or resumed input streaming of an unbound legacy record is refused or
not attempted. Drain old campaigns with their original client before upgrading;
never erase an old record merely to get a new submission through.

## Staging and cache reuse

Input resolution is fresh for every new handle; CRV's path-only input-key/MD5
memoization and chain's library-MD5 memoization are removed. Node caches use full
SHA-256 content identities, with destination names and sizes in the CRV set key.
Candidate response knobs alone do not invalidate an unchanged input cache.

Transfers use the saved source argv and expected MD5. Controller-side SHA-256,
MD5 and size verification must succeed before publishing `ready`. Existing ready
caches are rehashed, not trusted by marker alone; a bad ready cache is refused
without overwriting it while another job may be reading it. The job itself also
checks the pinned staged bytes after waiting and before physics, including the
ready-on-entry path that can run before the controller next polls.

Controller cache writers and chain per-subjob staging serialize across
threads/processes sharing the same state root; a delayed recovery caller verifies
ready inputs without rewriting them. Independent state roots, old clients and external writers are not
coordinated. Job holder/cleanup logic remains the existing site mechanism; a
cleanup receipt is not evidence of secure deletion, and scheduler-query failure
handling in that holder policy is not redesigned here.

The job emits the exact bound `artifact-manifest.json`; results refuse a missing
or different manifest and expose it plus its digest in metadata. This verifies
the binding/guard evidence, not full dependency closure or an attestation against
concurrent external mutation. Existing metrics are unchanged.

## Explicit gaps and rollout gates

- EAF still uses mutable in-place code, a venv and FNAL inputs. Plain image-only
  tasks do not yet acquire these bindings.
- SIF bytes are checked at job start but not frozen throughout later container
  invocations. Replacing the SIF after verification remains unsafe.
- CVMFS-fetched lookup tables, Offline geometry include closure, field maps and
  other downloaded dependencies are not yet fully resolved/pinned here.
- [Geometry FileRefs](geometry-capabilities.md) now have a bounded beam target-overlay
  handoff; arbitrary geometry/include trees remain unsupported.
- Validation envelopes, configuration-aware metrics, replicas/uncertainties and
  approved data-out/retention policy remain separate commissioning requirements.
- Deploy all cooperating submitters together; validate real node tools, snapshot
  paths, SIF hashing cost and the Slurm comment-retention gates before live use.

## Verification

Tests use local Git fixture repositories, synthetic files, fake endpoints and
local process tests; they do not contact a live scheduler or transfer physics
data. They cover branch/SHA/submodule resolution, old snapshot survival after a
new ref, dirty/ignored/symlink rejection, changed bytes at stable input/image
paths, concurrent durable binding, original-rendering retries, cache corruption,
consumer-side guards and exact result-manifest matching.

From the umbrella:

```bash
PYTHONPATH=core:crv:beam PYTHONDONTWRITEBYTECODE=1 \
  /Users/rmina/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider --import-mode=importlib \
  core/tests/test_artifacts.py core/tests/test_snapshots.py \
  core/tests/test_kit_parameters.py core/tests/test_endpoints.py \
  core/tests/test_endpoint_eaf.py core/tests/test_chainpoint.py \
  core/tests/test_submission_recovery.py crv/tests/test_kit_uva.py \
  beam/tests/test_kit_uva.py crv/tests/test_kit_uva_node.py
```

On a free GPVM use the same file list with the documented `pyenv ana` setup.
The EAF `setsid` and node-holder `flock` tests skip where those commands are absent.

Verification on 2026-10-25: the local combined suite passed 360 tests with four
platform skips, followed by the added chain recovery-race regression passing in
the 92-test chain suite. The final isolated GPVM stack on fresh published heads
passed **377 tests, no skips**, including 12 additional current-main CRV
configuration tests (`core/tests/test_crv_response_config.py`; add `core/tests`
to `PYTHONPATH` for that module's sibling-test import). Seven multiprocessing
fork warnings remain. No live endpoint jobs, physics runs or builds were started.
