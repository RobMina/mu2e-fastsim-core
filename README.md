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
| `mu2efastsim/eloss_jax.py` | jit-able float64 JAX twin of `eloss.dedx_restricted` and `eloss.sample_loss` (same G4 Urban branches, fixed-shape sampling, explicit keys; validated against the numpy module, `docs/gpu_port.md`); used by the beam stage (`beam/` in the umbrella) |
| `mu2efastsim/dedxtable.py` | PDG Bethe and radiative-loss tables for cross-checks |
| `mu2efastsim/cvmfsget.py` | fetch single files from a CVMFS repository over HTTP from a Stratum-1, without a CVMFS mount |
| `mu2efastsim/worldcensus.py` | census of the GDML world's solids and how `world.py` represents them |
| `mu2efastsim/materials.py` | `Material` with composition, density, I, Sternheimer parameters, Birks constant, fluctuation-model parameters; the materials of the extracted world and the tracker (support alloys, electronics, Mylar, W and Au of the sense wire) |
| `mu2efastsim/world.py` | GDML world -> global convex pieces; per-material path column of straight segments |
| `mu2efastsim/crossings.py` | vectorised ray-box crossings (JAX on CPU/GPU, numpy fallback) |
| `mu2efastsim/transport.py` | correlated multiple scattering to a plane (uniform or banded material, per-band \|p\|), sliced slab loss with range-out; scattering pieces for detector packages: Highland per layer with the log term on the cumulative x/X0 (`highland_var_increment`), the single-scattering tail (`chi_c2`, `chi_c2_electron`, `theta_nuclear`, `tail_threshold`, `single_scatter_tail`: screened Rutherford on nuclei with an exponential form factor, atomic electrons up to m_e/m_mu, optional in `msc_to_plane`/`world_loss`) |
| `mu2efastsim/tables.py` | flat-table I/O in the art dumper layout (`dump/` directory, `(run, subrun, event)` keys) |
| `mu2efastsim/stats.py`, `plots.py` | robust statistics (core sigma, binned-ratio band, MPV by Moyal fit `mpv_fit` and by histogram peak `mpv`) and plotting/stats-JSON helpers shared by every comparison |
| `mu2efastsim/match.py` | joins and lookups on multi-column integer keys, two-sample KS test (fast-vs-stock comparisons) |
| `iri/` | ALCF Polaris job drivers through the IRI API (inline job shipping, venv setup, GPU smoke test, git sync) |
| `mu2efastsim/endpoints/` | compute endpoints behind one interface (`submit`/`status`/`fetch`/`cancel`): `slurm_ssh` (any Slurm site over `ssh`; UVA today), `eaf` (Fermilab EAF: jobs in the user's JupyterHub server on an A100 MIG slice, driven with a hub token from the Fermilab network; one slice with a queue, auto-stop, keepalive), and `kit` = `fastsimkit`, the Mu2eBO evaluator-contract MCP server (one point = one job; tasks that read Fermilab-resident inputs in place run only on `eaf`). `chainpoint.py` runs a CRV chain point as N one-GPU jobs with streamed inputs. Tasks: `reco_bench`, `response_bench`, `v4_response`, `chain_point`, `chain`, `regression`, `veto_c4`, and `beam_point` when the umbrella's `beam/` is importable |
| `eaf/` | Fermilab EAF tools: `eafhub.py` (start, run, wait, stop a session from a GPVM), the P2 script on real steps, live tests of the kit and its queue, read-only probes of slice usage and server lifetime |
| `aurora/` | ALCF Aurora (Intel PVC) job scripts over SSH + PBS |
| `uva/` | UVA Rivanna/Afton Slurm jobs: step-0 probe, in-image validation with the real tables, KPP V-R2/V-R3 with a streamed reference; see `uva/README.md` |

## Backends

`crossings` runs on JAX (CPU or GPU) when JAX is installed and on numpy otherwise. `eloss`, `world` and
`transport` are numpy; `eloss_jax` is the JAX twin of the energy-loss sampler (delta-ray sampling not ported yet;
`docs/gpu_port.md`).

## Use

```bash
.venv/bin/python3 -m pip install -e path/to/mu2e-fastsim-core     # or put the umbrella repo root on PYTHONPATH
.venv/bin/python3 -m pytest path/to/mu2e-fastsim-core/tests
```

On a Mu2e GPVM no install is needed: `source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh && pyenv ana`, then
`source site.env` in the umbrella clone (it puts `core/` on `PYTHONPATH`) and
`python3 -m pytest -q -p no:cacheprovider core/tests`. The umbrella README and the tracker README describe the full
setup and the validation runs.

```python
from mu2efastsim import eloss, materials
loss = eloss.sample_loss(rng, materials.POLYSTYRENE, p_MeV, length_mm, tcut_MeV)
```

## History

The modules were first written in `mu2e-crv-fastsim` and moved here from its commit `f451d7b` (2026-10-16); the CRV
repository removed its copies and imports this package since 2026-10-16. `match.py` (multi-column key joins, lookups,
KS test) came from the CRV fast reco the same day; `tables.write(as_ttree=True)` from the CRV path-A work.

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
