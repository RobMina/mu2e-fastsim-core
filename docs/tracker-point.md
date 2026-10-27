# `tracker_point` adapter (registration only)

This adds endpoint registration and local/mock tests, **not rollout, live endpoint
readiness, performance certification or physics commissioning**. The matching
tracker package must be on the kit host's `PYTHONPATH`.

## Fixed pipeline and admission

| Parameter | Domain | Default |
|---|---|---|
| `sample` | `extracted_v04_primary_v1` only | `extracted_v04_primary_v1` |
| `shard` | integer 1–20 | 11 |
| `seed` | integer 0–2147483647 | 1 |

Whole-valued JSON floats are accepted; strings, booleans, NaN/Inf and unknown
parameters are refused. `files` and `inputs` are refused: the beam target geometry
FileRef mechanism is not a tracker GDML or conditions override.

The runner transports **all primary muons**, using stage C with sliced MSC,
**without a secondary library**, then digitises and runs **hybrid F2 at fixed
2000 MeV/c, with no fit material**. There is no F1 mode. The drift-source ROOT
file supplies conditions only, never stock event/seed/hit/fit inputs. Counts,
fit fraction and binomial counting SE use the complete generated-event
denominator, including zero-step and zero-fit events. This counting SE is not
model/systematic or independent-replica uncertainty.

Metrics (all required and finite): `generated_events`, `primary_muons`,
`events_with_steps`, `straw_steps`, `digis`, `hits`, `seeds`, `fit_attempts`,
`fit_successes`, `events_with_fit`, `written_events`, `fit_event_fraction`,
`fit_event_fraction_se`, `active_hits`, `steps_s`, `digitization_s`, `reco_s`,
`writer_s`, `wall_s`.

## Input roots and runtime

All mappings are relative to the **data root**, not its `crv_fastsim` child:

| Source | Job destination |
|---|---|
| `crv_fastsim/g4ref_v1_tables/gen.001400_{shard:08d}.root` | `gen.root` |
| `crv_fastsim/geometry/mu2e_extracted_v04.gdml` | `geometry.gdml` |
| `trkfs_digi_20261026/v2/digicond.npz` | `digi_cond.npz` |
| `trk_fastreco/cond_mc_au.json` | `reco_cond.json` |
| `trk_fastreco/trk_sa2/tracks_ref_mc_saveall.001400_00000000.root` | `drift_source.root` |

EAF uses the existing mounted route and runtime, deriving the data root from
`endpoint.fnal_data/..` (the baseline EAF setting names `crv_fastsim`). Inputs
are read in place. This route inherits the baseline mutable EAF checkout/venv;
it does **not** claim a retained source/runtime snapshot or pinned mounted input
binding. The runner checks input SHA256/size before and after consumption and refuses publication on change;
this is not an immutable snapshot or protection against transient change-and-restore. Runner provenance remains
output evidence, not a lifetime guarantee.

UVA uses `FASTSIMKIT_TRACKER_DATA` (default `/exp/mu2e/data/users/ram2aq`),
`FASTSIMKIT_FNAL_HOST` (default `mu2egpvm03`), `FASTSIMKIT_TRACKER_CODE` and
`FASTSIMKIT_CODE_REF`. Source snapshots prepare **core and tracker**, not beam
or CRV. The five inputs stream directly into a private per-job `/dev/shm`
tmpfs directory. Bound MD5/SHA256/size expectations are checked after streaming,
on ready-cache reuse, and again before consumption. Different params, sources,
targets or corrupted bindings are refused. A recovery caller rechecks transfer markers and job state inside the
stream lock; successful transfer is marked before releasing it. Stream errors request cancellation; input waiting
is bounded and an EXIT trap removes the scratch directory.

### Legacy SIF limitation — deliberately unchanged

This core baseline hashes `FASTSIMKIT_SIF` at admission and again before launch.
Its policy is **`verify-before-run-not-snapshot`**: the SIF is **not retained**,
there is no lifetime image snapshot, and an owner can change the source image
after the guard. The manifest explicitly records `full_immutability: false`.
Inputs are pinned expectations, not source snapshots; code snapshots are
read-only retained Git snapshots, not an owner-resistant security boundary.
The adapter does not port unrelated newer image-retention APIs into this core.

## Output and cleanup boundary

Both endpoint bodies invoke `trkfastsim.kitpoint` without `--retain-ntuple`.
The writer runs and checks `EventNtuple.root` in job scratch; only `metrics.json`,
`provenance.json`, and text/JSON execution diagnostics remain in `out/`.
There is **no new internal ROOT fetch permission**. Local writer tests may opt
into `--retain-ntuple`; endpoint task parameters cannot enable it.

Results require successful completion, a cleanup receipt, complete finite
metrics, preserved written/generated counts, full-denominator fraction/SE and
provenance. UVA additionally verifies the returned artifact manifest against the
original durable binding. A cleanup receipt is operational evidence, not proof
of deletion; SIGKILL/node loss can prevent traps and receipts. Existing site
cleanup policy still applies. No endpoint, image, installation or live data was
modified by this registration work.

## Local verification

With the laptop workspace venv and `PYTHONDONTWRITEBYTECODE=1`, using default
pytest import mode (tracker tests import sibling fixtures):

```bash
PYTHONPATH="$CORE:$TRACKER:$CRV:$BEAM" /Users/rmina/mu2e/.venv/bin/python3 \
  -m pytest -q -p no:cacheprovider \
  "$TRACKER/tests/test_kit_task.py" "$TRACKER/tests/test_kit_uva.py" \
  "$TRACKER/tests/test_kitpoint.py" "$CORE/tests/test_tracker_registration.py" \
  "$CORE/tests/test_kit_parameters.py" "$CORE/tests/test_endpoints.py" \
  "$CORE/tests/test_capabilities.py" "$CORE/tests/test_artifacts.py" \
  "$CORE/tests/test_endpoint_eaf.py" "$CORE/tests/test_chainpoint.py"
```

Result: **298 passed, 3 skipped** (existing macOS `setsid`-dependent EAF process
tests). `CORE` and `TRACKER` were the isolated `core_wt/kit` and `trkfs_wt/kit`
worktrees; CRV/beam were read-only optional import paths. Tests use synthetic
inputs, fake endpoints and local stub-shell execution, not live data or sites.
