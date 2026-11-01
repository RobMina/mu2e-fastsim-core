# mu2e-fastsim-core

Detector-independent core of the Mu2e Geant4-free fast simulation: muon energy loss and fluctuations matched to
Geant4 11.3.2, materials, the passive world from a GDML export, straight-line crossings, multiple scattering and slab
transport, the flat-table interchange format, statistics, the CRY cosmic generator backends, the compute-endpoint
adapters with the Mu2eBO kit (`fastsimkit`), and the site configuration and test tooling of all four repositories.

A submodule of the umbrella [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim), next to
[`mu2e-crv-fastsim`](https://github.com/RobMina/mu2e-crv-fastsim) and
[`mu2e-tracker-fastsim`](https://github.com/RobMina/mu2e-tracker-fastsim); work from a clone of the umbrella. This
repository holds code only: no Mu2e-internal conditions, tables, data or results.

## Status

| Part | State |
|---|---|
| muon energy loss, Urban fluctuations, Birks, deltas (`eloss`) | reference implementation; validated through CRV V1 and tracker T-V1 against Geant4 |
| JAX twin of the loss sampler (`eloss_jax`) | validated against NumPy; used by the beam stage |
| world, crossings, scattering, slab transport | validated through the CRV and tracker comparisons |
| CRY backends (`cry_native`, `cry_device`) | opt-in; device generator consistent with stock CRY (accepted rate 0.9998 +- 0.0016) |
| endpoints and `fastsimkit` | UVA (Slurm), Fermilab EAF, ALCF Polaris (IRI) and Aurora (PBS) job paths; admission and recovery unit-tested |
| electrons | on an unmerged branch (thin layers agree with Geant4; no multiple scattering yet) |
| device precision | float64 (float32 validated on a branch) |

Details: [`docs/validation.md`](docs/validation.md), [`docs/limitations.md`](docs/limitations.md).

## Quickstart

```bash
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
export PYTHONPATH=$PWD/core
python3 -m pytest -q -p no:cacheprovider core/tests       # no inputs needed; JAX tests skip without JAX
```

```python
import numpy as np
from mu2efastsim import eloss, materials
loss = eloss.sample_loss(np.random.default_rng(1), materials.POLYSTYRENE, p_MeV, length_mm, tcut_MeV)
```

More: [`docs/running.md`](docs/running.md) (install, input verification, the kit and its tasks, endpoint scripts).

## Map

| Path | Content |
|---|---|
| `mu2efastsim/eloss.py`, `eloss_jax.py`, `materials.py`, `transport.py` | energy loss, fluctuations, materials, multiple scattering and slab loss |
| `mu2efastsim/world.py`, `worldcensus.py`, `crossings.py` | GDML world, its census, ray-box crossings (NumPy or JAX) |
| `mu2efastsim/tables.py`, `match.py`, `stats.py`, `plots.py` | table I/O in the art dumper layout, key joins, statistics, plots |
| `mu2efastsim/cry_*.py`, `cosmic_cry.py`, `native/cry/` | CRY 1.7 native adapter and container recipe, device sampler, validation |
| `mu2efastsim/endpoints/` | endpoint adapters (Slurm over ssh, EAF) and the Mu2eBO kit |
| `mu2efastsim/site.py`, `inputs.py` | site configuration from the environment; input manifest check and fetch |
| `mu2efastsim/testing.py`, `testselect.py`, `ci/` | shared pytest hooks, the site-path checker, test selection, CI pins and helpers |
| `uva/`, `aurora/`, `iri/`, `eaf/` | endpoint job scripts (UVA Slurm, ALCF Aurora, ALCF Polaris IRI, Fermilab EAF) |
| `tests/` | pytest suite |

Docs: [`architecture.md`](docs/architecture.md), [`running.md`](docs/running.md), [`validation.md`](docs/validation.md),
[`limitations.md`](docs/limitations.md), [`gpu_port.md`](docs/gpu_port.md) (JAX loss sampler),
[`native-cry.md`](docs/native-cry.md), [`device-cry.md`](docs/device-cry.md), and the endpoint contracts
[`endpoint-admission.md`](docs/endpoint-admission.md), [`slurm-recovery.md`](docs/slurm-recovery.md),
[`artifact-bindings.md`](docs/artifact-bindings.md), [`geometry-capabilities.md`](docs/geometry-capabilities.md),
[`tracker-point.md`](docs/tracker-point.md).

## Site configuration

No site, user or storage path is built into the package. Every path, host and account comes from the environment
(`mu2efastsim/site.py`; the template is the umbrella's
[`site.env.example`](https://github.com/RobMina/mu2e-fastsim/blob/main/site.env.example)). Code that needs an unset
variable raises `mu2efastsim.site.SiteConfigError` naming it; shell scripts stop with `${VAR:?...}`.
`mu2efastsim.testing.check_site_paths`, run by every repository's `tests/test_no_site_paths.py`, keeps user names and
site paths out of the four repositories.

## Tests and CI

CI (`.github/workflows/tests.yml`) runs the suite on every pull request twice: with the pinned packages of
`ci/requirements.txt` (CPU JAX) and without JAX, where JAX tests must skip. Sibling repositories are checked out at
the same-named branch when it exists (`ci/checkout_siblings.sh`). Rules: the umbrella's `CONTRIBUTING.md`.

## Studies kept off main

| Branch | Contents |
|---|---|
| `park/endpoint-esnet` | ESnet IRI v2 jobs and scripts, Globus helper, Polaris hello/env probes |
| `park/beam-loss-studies` | `eloss_split_jax` (slab-split loss study for the beam stage) |

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
