# UVA Rivanna/Afton endpoint

Slurm jobs for the fast chain on UVA HPC (reachable only from the UVA network or the UVA Anywhere VPN, key-based ssh).
Site settings come from a private `site.env` (see `site.env.example`): `FASTSIMKIT_SLURM_HOST` (the ssh host),
`FASTSIMKIT_SLURM_ACCOUNT` (the Slurm allocation), `FASTSIMKIT_SCRATCH` (the work area on the cluster scratch, holding the
image, the job outputs and `kpp/`), `FASTSIMKIT_FNAL_HOST` and `FASTSIMKIT_FNAL_DATA` (the FNAL host and directory of the
Mu2e-internal inputs). The job scripts carry no account or log path: submit them with
`sbatch -A "$FASTSIMKIT_SLURM_ACCOUNT" -o "$FASTSIMKIT_SCRATCH/probe/slurm-%j.out" --export=ALL,FASTSIMKIT_SCRATCH="$FASTSIMKIT_SCRATCH"`
(the stream drivers do this themselves). Image: `mu2efastsim.sif` built on an `interactive` node with `container/build.sh`
(gpu variant keeps cuDNN and cuFFT). Measured cost ~60 SU per A100-hour (8-16 cores).

| Script | What | Inputs |
|---|---|---|
| `probe_uva.sh` | node/storage/network inventory, venv JAX on one A100, pytest, synthetic benches, R5 | synthetic |
| `validate_uva.sh` | inside the SIF: pytest, R5, V4 photons and SiPM with the real CRVConditions v6_0 tables (copy-compute-delete to tmpfs), real-table and chain benches; aborts if JAX is not on the GPU. Other GPUs: `sbatch --gres=gpu:b200:1 validate_uva.sh` | real tables, fetched per job from the CVMFS Stratum-1 |
| `kpp_reco_uva.sh` + `kpp_stream_uva.sh` | V-R2 and V-R3 on KPP data run 124155: the laptop streams the reference file GPVM -> job tmpfs with `srun --overlap`; outputs go back to the GPVM and are removed on UVA | Mu2e-internal reference file, never on UVA disk |

## Mu2eBO kit (`mu2efastsim.endpoints`)

`endpoints/slurm_ssh.py` is the endpoint adapter (submit / status / fetch / cancel over `ssh` + Slurm, local job records,
per-job SU cap); `endpoints/kit.py` is `fastsimkit`, an MCP server speaking the Mu2eBO evaluator contract on top of it
(one BO point = one job; tasks in `TASKS`, each writing `out/metrics.json`: `reco_bench`, `response_bench`,
`v4_response`, and `beam_point` from the umbrella's `beam/` (muon table streamed in, see `beam/beamfastsim/kit_uva.py`); `response_bench` and `v4_response` stage the real tables copy-compute-delete and must leave `out/cleanup.txt`).
`chain_point` (`endpoints/chainpoint.py`) is one BO point of the whole fast chain as N one-GPU sub-jobs (`<name>.j<k>`):
params `shards`, `files_per_job`, `shard_offset`, `max_events`, `seed`, `wall_minutes`; when a sub-job starts it writes
an `inbox` file and the kit streams the Mu2e-internal inputs from the FNAL host (`FASTSIMKIT_FNAL_HOST`,
`FASTSIMKIT_FNAL_DATA`) through the kit host into its tmpfs (`srun --overlap`, md5 checked, never on disk), 2 files at a
time per sub-job (`FASTSIMKIT_STREAMS`) and at most 6 streams overall (`FASTSIMKIT_MAX_CHANNELS`: one ssh connection per
host carries at most 10 channels). Code comes from a checkout on the cluster scratch (`FASTSIMKIT_CODE`, default `$FASTSIMKIT_SCRATCH/chain_code_kit`;
`FASTSIMKIT_CODE_REF`).
Metrics: accepted and generated events, accepted per GPU-hour (charged and chain-only), per estimated SU, input wait,
campaign time, clusters per generated event per sector type. The kit host must stay up and on the UVA network. A Mu2eBO `kits.toml` entry:

```toml
[fastsimkit]
command = ["${PYTHON}", "-m", "mu2efastsim.endpoints.kit"]
env_passthrough = ["SSH_AUTH_SOCK"]
set = { PYTHONPATH = "${M2FS_CODE}/core:${M2FS_CODE}/crv:${M2FS_CODE}/tracker:${M2FS_CODE}/beam", FASTSIMKIT_ENDPOINT = "uva", FASTSIMKIT_STATE_DIR = "${DATA_ROOT}/fastsimkit", FASTSIMKIT_MAX_SU_PER_JOB = "200" }
study_keys = {}
fixed_keys = { task = "string", events = "number", seed = "number", rows = "number", sipms = "number", shards = "number", files_per_job = "number", shard_offset = "number", max_events = "number", wall_minutes = "number", gen_set = "number", geom = "string", muons = "number", repeat = "number" }
accepts_lists = false
check = true
executors = ["grid"]
launch_stagger_s = 0
poll_s = [30.0, 300.0]
timeouts = { start = 60, submit = 120, status = 120, results = 900, check = 30, describe = 30, cancel = 120 }
```

The kit host must reach UVA by ssh (laptop on the UVA network/VPN today; FNAL hosts cannot). `tests/test_endpoints.py`
runs the adapter and the kit against a fake Slurm site.
