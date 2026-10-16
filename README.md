# mu2e-fastsim-core

Detector-independent core of the Mu2e Geant4-free fast simulation for the extracted (KPP, no-field) cosmic
configuration: muon energy loss and fluctuations matched to Geant4 11.3.2, material constants, the passive world
from a GDML export, straight-line crossings, transport helpers, the flat-table interchange format and shared
statistics.

It is a submodule of the umbrella repository [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim), next to the
detector stages [`mu2e-crv-fastsim`](https://github.com/RobMina/mu2e-crv-fastsim) and
[`mu2e-tracker-fastsim`](https://github.com/RobMina/mu2e-tracker-fastsim).

This repository holds technical software only: no physics results, no Mu2e-internal conditions, tables or data.

## Contents

| Module | What it holds |
|---|---|
| `mu2efastsim/eloss.py` | restricted Bethe mean loss (`G4MuBetheBlochModel`), Urban fluctuation sampler (`G4UniversalFluctuation` + `G4UrbanFluctuation`), Birks saturation, delta-ray rate and sampling, radiation length, Highland angle |
| `mu2efastsim/materials.py` | `Material` with composition, density, I, Sternheimer parameters, Birks constant, fluctuation-model parameters; the materials of the extracted world |
| `mu2efastsim/world.py` | GDML world -> global convex pieces; per-material path column of straight segments |
| `mu2efastsim/crossings.py` | vectorised ray-box crossings (JAX on CPU/GPU, numpy fallback) |
| `mu2efastsim/transport.py` | correlated multiple scattering to a plane (uniform or banded material), sliced slab loss with range-out |
| `mu2efastsim/tables.py` | flat-table I/O in the art dumper layout (`dump/` directory, `(run, subrun, event)` keys) |
| `mu2efastsim/stats.py`, `plots.py` | robust statistics and plotting/stats-JSON helpers shared by every comparison |
| `iri/` | ALCF Polaris job drivers through the IRI API (inline job shipping, venv setup, GPU smoke test, git sync) |

## Backends

`crossings` runs on JAX (CPU or GPU) when JAX is installed and on numpy otherwise. `eloss`, `world` and
`transport` are numpy only today; the JAX port of the energy-loss sampler is planned (`docs/gpu_port.md`).

## Use

```bash
.venv/bin/python3 -m pip install -e path/to/mu2e-fastsim-core     # or put the umbrella repo root on PYTHONPATH
.venv/bin/python3 -m pytest path/to/mu2e-fastsim-core/tests
```

```python
from mu2efastsim import eloss, materials
loss = eloss.sample_loss(rng, materials.POLYSTYRENE, p_MeV, length_mm, tcut_MeV)
```

## History

The modules were first written in `mu2e-crv-fastsim` and copied here from its commit `f451d7b` (2026-10-16). Until the
CRV repository switches to this package, the copies may diverge; they are merged later by hand.

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
