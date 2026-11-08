# Core: running

Core is used from a clone of the umbrella [`mu2e-fastsim`](https://github.com/RobMina/mu2e-fastsim); its
[`docs/running.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/docs/running.md) has the environment,
`site.env` and the input release. Core itself needs no inputs for its tests.

## Install and test

```bash
pip install -e core            # numpy, pandas, uproot, matplotlib, scipy, hep-endpoints; JAX optional ("core[jax]")
python3 -m pytest -q -p no:cacheprovider core/tests
```

or put `core` on `PYTHONPATH` (the umbrella's `site.env` does). The endpoint adapters under `mu2efastsim.endpoints`
are configured from the experiment-neutral [hep-endpoints](https://github.com/RobMina/hep-endpoints) package
(`hependpoints`, standard library only), pinned to one commit in `pyproject.toml` and `ci/requirements.txt`. Without
it the endpoint tests are skipped and every other module works. The CI environment is pinned in
[`ci/requirements.txt`](../ci/requirements.txt); CI runs the suite with and without JAX.
`python3 -m mu2efastsim.testselect --why` lists the test files a change affects; `--run --isolate` runs them one
pytest process per file.

Test tiers (hooks in `mu2efastsim.testing`, shared by every fastsim suite):

| Tier | Selection | Runs |
|---|---|---|
| per PR | default | every pull request and push |
| nightly | `--runslow`: adds `@pytest.mark.slow` (large samples, duplicated parameter sets via `@pytest.mark.pr_params`) | scheduled nightly; manual dispatch; PRs that change a path in `ci/runslow_paths.txt` |
| weekly | `--runweekly -m weekly`: only `@pytest.mark.weekly` (validations of tens of minutes; `--runslow` does not select them) | scheduled weekly; manual dispatch with `weekly` |

## Library use

```python
import numpy as np
from mu2efastsim import eloss, materials
rng = np.random.default_rng(1)
loss = eloss.sample_loss(rng, materials.POLYSTYRENE, p_MeV, length_mm, tcut_MeV)
```

## Inputs

```bash
python3 -m mu2efastsim.inputs check-inputs --manifest RELEASE.json --root "$M2FS_INPUTS"   # verify sizes and SHA-256
python3 -m mu2efastsim.inputs fetch-inputs --manifest RELEASE.json --root "$M2FS_INPUTS" --source-dir SRC   # dry run
```

`fetch-inputs` copies only with `--execute` and only files whose manifest entry is marked approved for
redistribution; it never overwrites a mismatched file.

## The Mu2eBO kit

`python3 -m mu2efastsim.endpoints.kit` is `fastsimkit`, an MCP server with the Mu2eBO evaluator contract
(`submit`, `status`, `results`, `check`, `describe`, `cancel`). `FASTSIMKIT_ENDPOINT` selects the adapter: `uva`
(any Slurm site over ssh, `endpoints/slurm_ssh.py`) or `eaf` (Fermilab EAF JupyterHub). The settings are the
`FASTSIMKIT_*` block of the umbrella's `site.env.example`; [`uva/README.md`](../uva/README.md) has the `kits.toml`
entry. The kit host (where `fastsimkit` or any `mu2efastsim.endpoints` code runs) needs hep-endpoints installed at
the pinned commit:

```bash
pip install "hep-endpoints @ git+https://github.com/RobMina/hep-endpoints.git@<commit in core/pyproject.toml>"
```

(`pip install -e core` does this.) Jobs do not: the project image, the EAF venv and GPVM `pyenv ana` run task bodies
that never import `mu2efastsim.endpoints` or `hependpoints`, and `mu2efastsim.site` is standard library only
(`tests/test_job_side_imports.py`). The UVA GPU types, charge rates and the automatic GPU choice are
`endpoints/mu2e_config.py` (`UVA_GPU_TABLE`); `ci/endpoint_identity.py` renders a grid of submissions to compare
endpoint versions byte for byte. Tasks:

| Task | Returns | Endpoints | Validated |
|---|---|---|---|
| `reco_bench` | synthetic CRV reco throughput | any GPU | benchmark only, no physics output |
| `response_bench` | photon and SiPM throughput with the real lookup table and photon map | any GPU | benchmark of the V4-validated response |
| `v4_response` | the V4 photon and SiPM checks | any GPU | V4 pass on A100, PVC, B200 and T4: PE per SiPM within 0.20 %, times within 0.14 ns |
| `chain_point` | one point of the CRV fast chain as N one-GPU jobs, inputs streamed in | Slurm | the CRV default chain (crv `docs/validation.md`: cluster PE chi2/ndf 1.09 on 5 M held-out events, counts within 0.4 %) |
| `chain`, `regression` | the CRV chain on held-out shards vs the stock chain | EAF | as `chain_point` |
| `crv_response` | one response-tuning point (reco-r3) scored against KPP data | EAF, Slurm | V-R5: the objective has the same minimum as the stock path within the loop noise |
| `veto_c4` | CRV veto inefficiency of a veto card (CPU) | EAF | C4 rule evaluation on a cached Run 1A selection; the frame-level veto closure is not on main |
| `tracker_point` | stage-C steps, fast digis, hits, seeds and the F2 fit of one shard ([tracker-point.md](tracker-point.md)) | EAF, Slurm | T-V0, T-V1 pass; fast digis vs stock on the same steps 0 of 19 flagged; open: T-V3 track yield -1.42 +- 0.37 % |
| `beam_point` | stop fraction and stop profile of a target geometry (umbrella `beam/` on `PYTHONPATH`) | EAF, Slurm | 65 Geant4-evaluated design points: Geant4 / fast 1.0045 +- 0.0005 (+- 0.0077 from the predicted scale), rank correlation 0.998; paired stop fraction 0.5-0.7 % low |
| `beam_sob` | sob stand-in of a target geometry: stops, conversion electrons, calorimeter proxy, the studies' approximate sensitivity (umbrella `beam/`; CPU) | EAF, Slurm | 65 design points: Geant4 / fast 0.9920 +- 0.0006 (+- 0.0077 from the stops normalisation), rms 0.46 %, rank correlation 0.993; the calorimeter proxy keeps 1.7 % too many electrons |

## Endpoint job scripts

`uva/` (Slurm on UVA, [README](../uva/README.md)), `aurora/` (ALCF Aurora, SSH + PBS), `iri/` (ALCF Polaris through
the IRI API, [README](../iri/README.md)), `eaf/` (Fermilab EAF sessions from a GPVM). Each script names the
variables it needs and stops if one is unset.
