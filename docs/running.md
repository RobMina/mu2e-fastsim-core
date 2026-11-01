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
pytest process per file. Large-sample checks are `@pytest.mark.slow` and run with `--runslow`.

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

| Task | Returns | Endpoints |
|---|---|---|
| `reco_bench` | synthetic CRV reco throughput | any GPU |
| `response_bench` | photon and SiPM throughput with the real lookup table and photon map | any GPU |
| `v4_response` | the V4 photon and SiPM checks | any GPU |
| `chain_point` | one point of the CRV fast chain as N one-GPU jobs, inputs streamed in | Slurm |
| `chain`, `regression` | the CRV chain on held-out shards vs the stock chain | EAF |
| `veto_c4` | CRV veto inefficiency of a veto card (CPU) | EAF |
| `tracker_point` | stage-C steps, fast digis, hits, seeds and the F2 fit of one shard ([tracker-point.md](tracker-point.md)) | EAF, Slurm |
| `beam_point` | stop fraction and stop profile of a target geometry (umbrella `beam/` on `PYTHONPATH`) | EAF, Slurm |

## Endpoint job scripts

`uva/` (Slurm on UVA, [README](../uva/README.md)), `aurora/` (ALCF Aurora, SSH + PBS), `iri/` (ALCF Polaris through
the IRI API, [README](../iri/README.md)), `eaf/` (Fermilab EAF sessions from a GPVM). Each script names the
variables it needs and stops if one is unset.
