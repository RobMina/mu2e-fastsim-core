"""Slurm over SSH for the fast chain (UVA Rivanna/Afton): hependpoints.slurm_ssh with the Mu2e settings.

Site settings come from arguments or the environment, never from the package: ``host`` ($FASTSIMKIT_SLURM_HOST),
``account`` ($FASTSIMKIT_SLURM_ACCOUNT) and ``workdir`` ($FASTSIMKIT_WORKDIR, else <FASTSIMKIT_SCRATCH>/kit); a
missing one raises mu2efastsim.site.SiteConfigError naming the variable when it is first needed. CPU jobs go to the
``standard`` partition; the budget guard uses the measured UVA rates (mu2e_config). Behaviour, job layout and the
durable-intent recovery are hependpoints' (its docs/slurm-recovery.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from hependpoints.runners import Runner  # noqa: F401
from hependpoints.slurm_ssh import FAILED, WORKING, map_state, ssh_runner  # noqa: F401
from hependpoints.slurm_ssh import SlurmSSHEndpoint as _SlurmSSHEndpoint
from hependpoints.slurm_ssh import su_estimate as _su_estimate

from mu2efastsim import site

from . import Resources
from . import mu2e_config as cfg


def su_estimate(r: Resources, su_per_gpu_h: float = cfg.UVA_SU_PER_GPU_H,
                su_per_core_h: float = cfg.UVA_SU_PER_CORE_H) -> float:
    return _su_estimate(r, su_per_gpu_h, su_per_core_h)


class SlurmSSHEndpoint(_SlurmSSHEndpoint):
    HOST_VARS = ("FASTSIMKIT_SLURM_HOST",)
    ACCOUNT_VARS = ("FASTSIMKIT_SLURM_ACCOUNT",)
    WORKDIR_VARS = ("FASTSIMKIT_WORKDIR",)
    CONFIG_HINT = site.HINT
    UNREACHABLE_HINT = cfg.UVA_UNREACHABLE_HINT

    def __init__(self, host: str | None = None, account: str | None = None, partition: str = "gpu",
                 workdir: str | None = None, state_dir: str | Path = cfg.UVA_STATE_DIR,
                 max_su_per_job: float = cfg.UVA_MAX_SU_PER_JOB, runner: Optional[Runner] = None, name: str = "uva",
                 su_per_gpu_h: float = cfg.UVA_SU_PER_GPU_H, su_per_core_h: float = cfg.UVA_SU_PER_CORE_H,
                 timeout_s: float = 120.0, stream_runner=None):
        workdir = workdir or site.setting("FASTSIMKIT_WORKDIR") or site.scratch() + "/kit"
        super().__init__(host=host, account=account, partition=partition, workdir=workdir, state_dir=state_dir,
                         max_su_per_job=max_su_per_job, runner=runner, name=name, su_per_gpu_h=su_per_gpu_h,
                         su_per_core_h=su_per_core_h, timeout_s=timeout_s, stream_runner=stream_runner,
                         cpu_partition="standard", environment=cfg.ENVIRONMENT)
