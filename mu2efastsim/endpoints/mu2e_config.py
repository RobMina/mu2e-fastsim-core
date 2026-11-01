"""Mu2e configuration of the experiment-neutral hep-endpoints package.

Everything Mu2e-specific that used to live in the endpoint adapters: the job variable and token prefixes, the public
umbrella repository and its module list, the EAF fast-chain environment (Mu2e ana python, venv with JAX, PYTHONPATH of
core/crv/tracker), the UVA charge rates and the defaults of the local state directories. Site settings (hosts,
accounts, storage paths) still come only from the environment, through mu2efastsim.site.
"""
from __future__ import annotations

import shlex

from hependpoints.environment import JobEnvironment
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


# UVA Rivanna/Afton charges (measured): ~53 SU per A100-hour plus ~0.5 SU per core-hour; per-job cap.
UVA_SU_PER_GPU_H = 53.0
UVA_SU_PER_CORE_H = 0.5
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
