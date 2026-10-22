# IRI job drivers (ALCF Polaris, ESnet East)

Jobs go to Polaris through the ALCF IRI API (`https://api.alcf.anl.gov`), with the genesis skill `amsc-skills/iri-api`
scripts ([AI-ModCon/genesis-skills](https://github.com/AI-ModCon/genesis-skills)) and a Python with globus-sdk. Read
`references/ALCF.md` in that skill first.

**API v2 (seen 2026-10-19).** The ALCF API moved to `/api/v2` (OpenAPI at `https://api.alcf.anl.gov/api/v2/openapi.json`;
the old `/openapi.json` and `/api/v1/...` answer 404). Operation ids, resource ids and the JobSpec fields used here are
unchanged; the filesystem `ls`/`view` calls now take a JSON body (`{"path": ...}`, `view` also `size`/`offset`) instead
of query parameters. `tools/iri.sh` passes `--openapi-url` (override with `IRI_OPENAPI_URL`); the genesis helper's
built-in ALCF OpenAPI URL is still the old one. Site settings (`IRI`, `IRI_PY`, `M2FS_EAGLE`, `ALCF_PROJECT`) come from
`site.env` (copy `site.env.example` of the umbrella repo).

| Item | Value |
|---|---|
| ALCF project | `$ALCF_PROJECT` (the job JSON `account`) |
| Polaris resource id | `55c1c993-1124-47f9-b823-514ba3849a9a` |
| Eagle filesystem resource id | `1c3ad9d4-2e91-42bc-becb-72b1fde1235c` |
| Work area | `$M2FS_EAGLE` (on Eagle) |
| Queues | `debug` (1-2 nodes, <= 1 h; can sit for hours), `capacity` (1-4 nodes, up to 168 h; the hackathon project often hits its per-project limit, "would exceed queue capacity's per-project limit"), `preemptable` (1-10 nodes; started within a minute on 2026-10-19), `prod` (>= 10 nodes); the hackathon reservation queue `R7728155` while it is active |

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

## When the API answers 403 or 401

- **HTTP 403 with a Cloudflare "Just a moment..." page** (even for `/openapi.json`): the client's network is challenged,
  not the account. Seen from a residential laptop connection on 2026-10-16 while a FNAL GPVM and the AI node got 200.
  Submit from a host that gets 200: copy the access token (`token_manager.py ensure --print-token`) and the JobSpec
  built by `make_inline_job.py` to that host's `/tmp` (mode 600) and POST with curl:
  `curl -A iri-api-client/1.0 -H "Authorization: Bearer $(cat tok)" -H "Content-Type: application/json"
  -X POST --data @job.json https://api.alcf.anl.gov/api/v2/compute/job/<polaris resource id>`; status:
  `GET /api/v2/compute/status/<resource id>/<job id>`.
- **HTTP 401 "Globus token not active"** while `token_manager.py status` says the token is valid: the stored ALCF token
  has no expiry, so `ensure --min-ttl` reuses it. Force a refresh with the stored refresh token
  (`token_manager.refresh_stored_tokens(tokens, ["alcf"])`, then `save_tokens`); interactive login only if that fails.

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
| `jobs/probe_polaris.json` + `scripts/probe_env.sh` | compute-node environment probe | diagnostic |
| `jobs/mkdir_eagle.json` | create the work area on Eagle | utility |
| `jobs/crv_reco_bench_polaris.json` + `scripts/crv_reco_bench.sh` | CRV fast pulse reco throughput, one process per A100 (`crvfastsim reco-bench` on synthetic waveforms) after the GPU tests; queue `capacity` (1-4 nodes, no debug wait); `--args N` events per GPU | `events_per_s` per GPU in the job output |
| `jobs/crv_photon_bench_polaris.json` + `scripts/crv_photon_bench.sh` | CRV photon substep kernel throughput (`crvfastsim.response.bench`, synthetic tables: no Mu2e-internal input), one process per A100; submitted to `preemptable` when `capacity` is full | **290-305 k rows/s per A100** (PBS 7761901, 2026-10-19) |
| `scripts/stage_r3_eagle.sh` (laptop, not a job) | stream the R3 inputs GPVM -> Eagle `r3_inputs/` (mode 700) over ssh through the laptop: the 20 MDC2025as step tables, CRVCalib/CRVPhoton CSVs, a `git archive` of `crvextrap` pinned by commit; md5-checked, resumable, `MANIFEST.txt`. Needs an ssh master to a Polaris login node; `--dry-run` lists the pairs | every file `streamed`/`present`, manifest written |
| `jobs/r3_polaris.json` + `scripts/r3_polaris.sh` | R3: `crvfastsim reco-r3 --jax --objective` on the staged step tables, one process per A100 (`--args TARGET`, default 1 M per node; extras default `--v05`), lookup tables from CVMFS into `/local/scratch`, then `--merge --wall` (job wall, queue excluded); outputs `$M2FS_EAGLE/r3/<time>_<jobid>/` | `r3_merge.txt`: `minutes_for_1M` (R3 <= 15) |
| `jobs/git_sync_polaris.json` + `scripts/git_sync.sh` | clone or `--ff-only` pull the public `mu2e-fastsim` umbrella repo with its submodules into `$M2FS_EAGLE/code` over HTTPS through the ALCF proxy; `--args fresh` removes the clone first | prints HEAD of each submodule |


## ESnet East (AWS EKS, IRI v2)

`IRI_FACILITY=esnet tools/iri.sh` drives `https://iri-dev.ppg.es.net/api/v2` with curl + jq and an AmSC personal access token (MyAmSC,
project selected when it is made; 30 days) from `$AMSC_TOKEN` or the mode-600 file `$AMSC_TOKEN_FILE` (default
`~/.amsc/pat`). No Globus login, no Cloudflare challenge: the laptop can submit. Every job is a Kubernetes pod running a
**public** OCI image (`container.image`); resources are per pod (`cpu_cores_per_process`, `memory` in bytes,
`gpu_cores_per_process`). The hackathon project maps every participant to one shared user (`hackathon2610`) with a
shared EFS home: unique run directories, nothing non-public on EFS, cancel only your own job ids.

| Item | Value |
|---|---|
| compute resource | `fb0aafe1-c780-55c0-b635-a7121f1b0ce5` (Low Priority, Kueue) |
| storage resource | `b6d59ba3-5c2d-57dd-b3b9-108832da0578` (EFS: `/data/home/<user>`, `/data/projects/<project>`) |
| pod (probe 2026-10-20) | 80 GB overlay root, `/dev/shm` 64 MB, no GPU/FUSE/userns in a CPU pod; open egress (PyPI, GitHub, ghcr, CVMFS S1) |
| files API | `mkdir`, `upload` (multipart), `ls`, `view`, `download` (base64, **max 2 MiB**), `rm` (recursive, `{path}` only) work |
| image | public `ghcr.io/robmina/mu2e-fastsim:<umbrella sha7>-cpu\|gpu` (and `latest-*`), built by `.github/workflows/image.yml` of the umbrella from `container/Dockerfile`; `/opt/mu2e-fastsim` is read-only for the pod uid, so every `--outdir` goes to `/tmp` |

```bash
RUN=$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import uuid;print(uuid.uuid4().hex[:8])')
export ESNET_WORK=/data/home/hackathon2610/mu2efastsim/$RUN
export IRI_FACILITY=esnet
tools/iri.sh mkdir $ESNET_WORK/probe
python3 tools/make_inline_job.py jobs/probe_esnet.json scripts/probe_esnet.sh --out /tmp/job.json
tools/iri.sh submit /tmp/job.json          # prints {"id": ...}
tools/iri.sh wait <id> 400
tools/iri.sh cat $ESNET_WORK/probe/probe.out
tools/iri.sh alloc                         # node-hours and bytes used / allocated (counters lag)
tools/iri.sh rm $ESNET_WORK                # recursive; clean up the shared home when done
# image jobs: ESNET_IMAGE_TAG=<sha7>-cpu with jobs/bench_esnet.json + scripts/bench_esnet.sh (real tables -> pod /tmp,
# deleted on exit) or jobs/repro_esnet.json + scripts/repro_esnet.sh (R5/P6 npz under the download cap)
```

### Globus (ESnet collection)

`tools/globus_esnet.py` (globus-sdk >= 4; on the laptop the conda env `/opt/miniconda3/envs/globus_env`) moves files
larger than the 2 MiB IRI download cap. Login once: `login-url` prints a URL, the user logs in and pastes the code to
`login-code <CODE>`; refresh tokens go to `~/.globus/mu2efastsim_transfer_tokens.json` (mode 600). Then `ls esnet /`,
`search <text>`, `transfer <src> <path> <dst> <path> [--recursive]`, `task <id> --wait 600`. The ESnet collection
(`a4d0333d-b9f0-442f-8d14-7d3a32596cb9`, a GCS v5 guest collection) has no `data_access` scope; access needs membership
in the Globus group "AmSC Genesis Hackathon 2610" (`54b528d9-c94d-11f1-aa50-0efff525aa9b`), otherwise every call answers
`PermissionDenied: No effective ACL rules`.
