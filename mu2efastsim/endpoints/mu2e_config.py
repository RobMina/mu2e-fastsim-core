"""Mu2e configuration of the experiment-neutral hep-endpoints package.

Everything Mu2e-specific that used to live in the endpoint adapters: the job variable and token prefixes, the public
umbrella repository and its module list, the EAF fast-chain environment (Mu2e ana python, venv with JAX, PYTHONPATH of
core/crv/tracker), the UVA GPU types with their charge rates and the automatic GPU choice, and the defaults of the
local state directories. Site settings (hosts, accounts, storage paths) still come only from the environment, through
mu2efastsim.site.
"""
from __future__ import annotations

import shlex

from hependpoints.environment import JobEnvironment
from hependpoints.slurm_ssh import GpuTable, GpuType
from hependpoints.snapshots import SourceRepo

# Job scripts see $M2FS_JOB_DIR and $M2FS_OUT; Slurm job names m2fs_<handle>; intent tokens m2fs-v1-<hex>.
ENVIRONMENT = JobEnvironment(var_prefix="M2FS", job_prefix="m2fs")
OUT_VAR = ENVIRONMENT.out_var

# Public code snapshots: the umbrella and its selectable modules; a branch override moves core and crv only.
UMBRELLA_URL = "https://github.com/RobMina/mu2e-fastsim.git"
MODULES = ("core", "crv", "tracker")
BRANCH_MODULES = ("core", "crv")


def source_repo(url: str = UMBRELLA_URL) -> SourceRepo:
    return SourceRepo(url=url, modules=MODULES, default_modules=MODULES, branch_modules=BRANCH_MODULES,
                      require_modules=True)


# UVA GPU types: partition, SU per GPU-hour / core-hour / GB-hour (rc-website data/service.yaml and the Slurm billing
# weights / 10, read 2026-10-30), and the largest (cores, GB) a job of that type may ask for
UVA_GPU_TYPES = {
    "a100": ("gpu", 50.89, 0.403, 0.043, None),
    "a40": ("gpu", 18.669, 0.34, 0.02, None),
    "a6000": ("gpu", 14.273, 0.383, 0.062, None),
    "v100": ("gpu", 2.096, 0.058, 0.005, None),
    "b200": ("gpu", 127.42, 0.437, 0.044, None),
    "rtx_pro_6000": ("gpu", 33.43, 0.427, 0.049, None),
    "1g.24gb": ("gpu-mig", 0.0, 0.0, 0.0, (4, 32)),
}
UVA_CPU_RATES = (0.177, 0.0143)   # standard (Rivanna) per core-hour, per GB-hour
AUTO, UVA_SLICE = "auto", "1g.24gb"   # default gpu_type: the slice for one-GPU requests that fit it, else the task's card


def _gpu_type(name, partition, gpu, core, gb, cap):
    cores, mem = cap or (None, None)
    return GpuType(partition=partition, su_per_gpu_h=gpu, su_per_core_h=core, su_per_gb_h=gb,
                   max_gpus=1 if name == UVA_SLICE else None, max_cores=cores, max_mem_gb=mem)


UVA_GPU_TABLE = GpuTable(
    types={name: _gpu_type(name, *row) for name, row in UVA_GPU_TYPES.items()}, auto=(UVA_SLICE,), default=AUTO,
    cpu_su_per_core_h=UVA_CPU_RATES[0], cpu_su_per_gb_h=UVA_CPU_RATES[1], price_unknown_as="a100",
    fit_reason="{mem_gb} GB fits the {type} slice (<= {max_mem_gb} GB, cores capped at {max_cores})",
    nofit_reason="{gpus} GPU(s), {mem_gb} GB do not fit one {type} slice ({max_mem_gb} GB): the task's card")

# UVA per-job budget cap (SU) and the hint shown when the cluster is unreachable.
UVA_MAX_SU_PER_JOB = 500.0
UVA_UNREACHABLE_HINT = "UVA needs the UVA network or VPN"

# Local job records.
UVA_STATE_DIR = "~/.mu2efastsim/uva"
EAF_STATE_DIR = "~/.mu2efastsim/eaf"

# EAF fast-chain runtime: a venv on shared storage built on first use from the Mu2e ana env plus pip JAX.
ANA_PY = "/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python"
JAX_PKG = "jax[cuda12]==0.11.2"


def eaf_runtime(venv: str, code: str) -> dict:
    """Shell definitions of A (run a command in the fast-chain environment) and T (the same with the staged tables in
    CRVFS_LOOKUP_DIR) for the kit's task bodies on EAF (no Apptainer there)."""
    v, c = shlex.quote(venv), shlex.quote(code)
    a = (f'M2FS_VENV={v}; M2FS_CODE={c}\n'
         f'[ -x "$M2FS_VENV/bin/python" ] || {ANA_PY} -m venv --system-site-packages "$M2FS_VENV" || exit 2\n'
         f'"$M2FS_VENV/bin/python" -c "import jax" 2>/dev/null || "$M2FS_VENV/bin/python" -m pip install -q '
         f'"{JAX_PKG}" || exit 2\n'
         'git -C "$M2FS_CODE" log -1 --format="umbrella %h" 2>/dev/null; git -C "$M2FS_CODE/crv" log -1 --format="crv %h" 2>/dev/null\n'
         'A() { local c=$1; shift; [ "$c" = python ] && c="$M2FS_VENV/bin/python"; ( cd "$M2FS_CODE" && '
         'PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/crv:$M2FS_CODE/tracker" XLA_PYTHON_CLIENT_PREALLOCATE=false '
         '"$c" "$@" ); }\n')
    t = 'T() { ( export CRVFS_LOOKUP_DIR="$L"; A "$@" ); }'
    return {"a": a, "t": t}
