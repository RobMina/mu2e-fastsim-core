# Evaluator admission contract (2026-10-25)

This is the first boundary-hardening slice for the existing Mu2eBO native kit, not
another scheduler or a physics-validation certificate. No live endpoint deployment
or new physics run is implied by the local tests.

## Parameter and routing contract

`mu2efastsim.endpoints.parameters` defines dependency-free typed parameter domains.
`FastsimKit.render`, `check` and `submit` use them; the beam and CRV streaming
validators share the same definitions. Integers accept whole-valued JSON floats
because Mu2eBO emits them. Numeric strings, booleans as numbers, fractional counts,
NaN/Inf, unsupported enums/keys and invalid cross-field windows are refused.

CRV knobs may be omitted or set to the existing empty-string sentinel (nominal).
`null` and numeric strings are not nominal aliases. Positive propagation/stretch/
time-constant domains, nonnegative scales/noise/yields, and [0,1] crosstalk are input
sanity bounds, **not validated calibration ranges**. CRV `tables=all` requires an
explicit positive event target; its current shell task has no default target for
that set. It remains EAF-only. Beam geometry remains a restricted named file or
`/exp/mu2e/...` path; this is not content-addressed FileRef support.

`describe()` preserves `params`, `metrics`, and `accepts_lists`, and adds
`parameter_schema_version: 1` and `tasks`. Task descriptors contain parameter
units/roles/domains, defaults, required names, registered endpoint routes and the
current refusal of files/inputs. Mu2eBO's existing parser tolerates additive keys;
consuming this richer description for study preflight is still future work.
Endpoint routes describe registration, not measured hardware/physics support.
Configuration-dependent metric availability and validation envelopes are not yet
encoded. The logical kit version remains unchanged: this patch rejects invalid
requests and repairs shell construction, rather than redefining scored observables.

## Fixed CRV response configuration

`crv_response` accepts optional `response_config`: a schema-v1 JSON object or inline JSON string, with empty string
meaning the shipped nominal preset. Validation is delegated to the CRV package's standard-library-only
`response_config` module, then the full object is canonicalized as one shell-quoted argument. Kit-host paths and
changes to staged asset paths are refused. Effective config plus existing scalar knobs are validated together,
so overrides cannot bypass cross-field constraints. Plain EAF and streamed UVA requests use identical admission.

The parameter descriptor role is `fixed_configuration`; new SiPM/reco fields are not registered as scalar BO knobs.
The resolved R3 configuration/asset hashes are output provenance, not an automatic board measurement-hash update.
A study that changes its fixed configuration must establish a new measurement definition as appropriate. See
`crv/docs/response-configuration.md` in the matching umbrella. Older CRV task definitions without this optional
parameter remain compatible with the core schema; the new CRV task requires a core containing this extension.
Tests: `tests/test_crv_response_config.py` (fake endpoints only).

## Admission and command construction

`check` and `submit` share handle, parameter, artifact and delegate admission before
recording a task or dispatching it. Unsupported `files`/`inputs` fail on both paths.
Handles are restricted to alphanumeric/underscore starts followed by
alphanumeric/underscore/dot/hyphen characters.

Chain-point admission also runs directly in its handler's `check` and `submit`:
streaming capability, SU cap, stream-rate sanity and the existing input-wait plus
compute wall-budget heuristic are checked before code preparation or submission.
That heuristic is retained, not newly calibrated or a runtime prediction guarantee.
Successful checks may still read remote source MD5s while rendering; `check` is not
an offline artifact resolver. Slurm exposes a pure resource-budget hook used by
plain and streamed task admission before code preparation. Chain admission uses
the endpoint's configured SU rates, rather than assuming UVA's defaults. Scheduler
state and endpoint configuration can still change between check and submission.

CRV knob values are shell-quoted outside the double-quoted literal `name=` prefix.
This avoids nested quoting and preserves one `name=value` argv item; typed
validation independently rejects expansion-bearing strings before rendering.
Sentinel tests inspect adversarial text only, without executing it. Existing
benign knob-block tests exercise actual argv construction.

## Verification and outstanding work

Local mocked endpoint and parameter tests cover check/submit parity, rejected
requests without submissions/task records, the chain budget bypass, finite/count/
enum/cross-field rules, defaulted nominal CRV inputs, integral-float compatibility,
beam/CRV delegate parity, descriptor coverage and safe shell-token placement.

From the umbrella, with the laptop workspace venv:

```bash
PYTHONPATH=core:crv:beam PYTHONDONTWRITEBYTECODE=1 \
  /Users/rmina/mu2e/.venv/bin/python3 -m pytest -q -p no:cacheprovider --import-mode=importlib \
  core/tests/test_kit_parameters.py core/tests/test_endpoints.py \
  core/tests/test_endpoint_eaf.py core/tests/test_chainpoint.py \
  crv/tests/test_kit_uva.py beam/tests/test_kit_uva.py
```

`--import-mode=importlib` permits the CRV and beam suites' duplicate test basenames.
Result: **199 passed, 3 skipped**. The macOS host lacks `setsid`, so three existing
local EAF process tests skip. Five existing fork-from-multithreaded-process warnings
come from the channel-slot concurrency test. The isolated publication worktrees on
mu2egpvm02 (ana 2.8.0 / Python 3.12), based on current published core/CRV/umbrella
heads rather than the dirty shared checkout, pass **202 tests with no skips**.
These are mocked/local process tests, not live EAF or Slurm jobs.

Next slices, before general unattended BO commissioning:

1. [Slurm recovery](slurm-recovery.md) implements durable intent, same-handle locking
   and acknowledgement-loss reconciliation with mocked tests. Live CLI/comment
   retention verification and coordinated rollout remain required; an ambiguous
   intent never authorizes another dispatch.
2. Immutable runtime/code/image/input/conditions manifests, lifetime-safe job
   snapshots, resume verification and content-verified cache reuse.
3. Geometry FileRef handoff and task/configuration-aware metric capabilities in the
   existing Mu2eBO contract (not candidate geometry in board measurement identity).
4. Independent input/seed replica semantics, uncertainty/fidelity and analysis-ready
   output composition; do not promote truth-conditioned diagnostics to evaluators.
5. Agree retention, storage class and allowed outputs; cleanup receipts alone do not
   prove deletion. No data movement or retention-policy change is part of this patch.

## Opt-in native CRY chain points

`chain_point` additionally admits `gen_set=4` (CPU) or `gen_set=5` (JAX GPU) and positive `gen_events` per
sub-job for extracted_v04 only. Counts/key bounds, unsupported combinations,
CPU-generation wall allowance, preserved table identities and rollout limits
are documented in [native-cry.md](native-cry.md). This removes gen-table
resolution/streaming, not detector input bindings or tmpfs cleanup. Its native
receipts and exposure rates are result metadata; existing metric meanings and
board measurement identity are unchanged. Capability descriptors advertise
implementation support, not physics certification or a live rollout.
