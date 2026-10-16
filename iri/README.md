# IRI job drivers (ALCF Polaris)

Jobs go to Polaris through the ALCF IRI API (`https://api.alcf.anl.gov`), with the genesis skill `amsc-skills/iri-api`
scripts ([AI-ModCon/genesis-skills](https://github.com/AI-ModCon/genesis-skills)) and a Python with globus-sdk. Read
`references/ALCF.md` in that skill first. Site settings (`IRI`, `IRI_PY`, `M2FS_EAGLE`, `ALCF_PROJECT`) come from
`site.env` (copy `site.env.example` of the umbrella repo).

| Item | Value |
|---|---|
| ALCF project | `$ALCF_PROJECT` (the job JSON `account`) |
| Polaris resource id | `55c1c993-1124-47f9-b823-514ba3849a9a` |
| Eagle filesystem resource id | `1c3ad9d4-2e91-42bc-becb-72b1fde1235c` |
| Work area | `$M2FS_EAGLE` (on Eagle) |
| Queues | `debug` (1-2 nodes, <= 1 h), or the project's production queue or reservation |

## What the ALCF IRI API actually accepts (tested 2026-10-16)

| Feature | State | Workaround |
|---|---|---|
| `upload` | **501 not implemented** | ship scripts inline: `tools/make_inline_job.py` base64-encodes the script into `bash -c` |
| `pre_launch` / `post_launch` | **501 not supported** | same |
| `arguments` for `/bin/bash` | at most 2 | script arguments go in as `set --` inside the shipped copy |
| newlines in a `bash -c` string | flattened to `; ` (heredocs break) | base64 |
| `directory` | ignored: jobs start in `$HOME` | the inline wrapper `cd`s first |
| job environment | bare non-login bash, empty `MODULEPATH` | `bash -l` restores it, but `module load conda` still silently fails (its CUDA/GCC checks) |
| ALCF conda | use the interpreter directly; the project venv is built on it (`scripts/setup_venv.sh`) | |
| outbound network | only through `http(s)_proxy=http://proxy.alcf.anl.gov:3128`; PyPI, GitHub and the CVMFS Stratum-1 answer 200; the FNAL conditions DB (QE :9443) does not | text-file conditions |
| `mkdir`, `ls`, `view` | work; return a `task_id` to poll | `tools/iri.sh ls\|cat` |
| `/dev/fuse`, user namespaces | present on compute nodes (`max_user_namespaces` 2,062,512) | cvmfsexec route possible |
| `/local/scratch` | 2.8 TB node-local NVMe | caches |
| GitHub | compute nodes: no direct route; HTTPS works through `proxy.alcf.anl.gov:3128` (the repo is public, so no key) | `scripts/git_sync.sh` sets `https_proxy` and clones over HTTPS |
| `getJob` | gives state and exit code only, no stdout path | read the files you named |

Literal paths only in the submitted JSON (the API does not expand `$HOME`/`$USER`): `tools/make_inline_job.py` expands
`${M2FS_EAGLE}` and `${ALCF_PROJECT}` in the templates before submission.

## Environment

Project venv `$M2FS_EAGLE/venv`: ALCF conda `2026-10-01` Python 3.13.15, `jax[cuda12]`
0.11.2, numpy 2.5.4, h5py 3.16, pyarrow 26. JAX's default fp32 matmul uses TF32 on A100 (measured 94-126 "TFLOP/s"
against a 19.5 FP32 peak): physics kernels must set `jax_default_matmul_precision="highest"` or use float64.

```bash
source site.env        # IRI, IRI_PY, M2FS_EAGLE, ALCF_PROJECT

# token (interactive first time)
$IRI_PY $IRI/scripts/token_manager.py ensure --facilities alcf --min-ttl 300

# any job: template JSON + local script -> inline JobSpec -> submit, wait, read output
python3 iri/tools/make_inline_job.py iri/jobs/hello_polaris.json iri/scripts/hello_gpu.sh --out /tmp/job.json
iri/tools/iri.sh submit /tmp/job.json          # prints the PBS job id
iri/tools/iri.sh wait <jobid> 280
iri/tools/iri.sh cat $M2FS_EAGLE/hello/hello.out
```

| Job | Purpose | Exit check |
|---|---|---|
| `jobs/setup_venv_polaris.json` + `scripts/setup_venv.sh` | build the project venv on Eagle | done 2026-10-16 (PBS 7752878, from mu2e-crv-fastsim) |
| `jobs/hello_polaris.json` + `scripts/hello_gpu.sh` | node/GPU inventory, JAX matmul per A100 | **passed** 2026-10-16 (PBS 7752911, from mu2e-crv-fastsim): 4x A100-SXM4-40GB, driver 580.65.06, JAX sees 4 devices |
| `jobs/git_sync_polaris.json` + `scripts/git_sync.sh` | clone or `--ff-only` pull the public `mu2e-fastsim` umbrella repo with its submodules into `$M2FS_EAGLE/code` over HTTPS through the ALCF proxy; `--args fresh` removes the clone first | prints HEAD of each submodule |

