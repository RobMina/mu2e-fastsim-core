"""Slurm over SSH for the fast chain (UVA Rivanna/Afton): hependpoints.slurm_ssh with the Mu2e settings.

Site settings come from arguments or the environment, never from the package: ``host`` ($FASTSIMKIT_SLURM_HOST),
``account`` ($FASTSIMKIT_SLURM_ACCOUNT) and ``workdir`` ($FASTSIMKIT_WORKDIR, else <FASTSIMKIT_SCRATCH>/kit); a
missing one raises mu2efastsim.site.SiteConfigError naming the variable when it is first needed. CPU jobs go to the
``standard`` partition. Budget guard: a submit whose estimated charge exceeds ``max_su_per_job`` is refused (UVA's
published per-type SU rates, mu2e_config.UVA_GPU_TYPES). ``gpu_type`` picks the card of GPU jobs: "auto" (default)
puts a one-GPU job whose memory request fits the free RTX PRO 6000 MIG slice (``1g.24gb`` on ``gpu-mig``, at most
32 GB and 4 cores) on the slice and leaves any larger request on the card the task asked for; an explicit type
(``a40``, ``a100``, ``1g.24gb``, ...) applies to every GPU job; None keeps every task's own request. The choice and the
original request go into the job script header and the job record (``gpu_choice``). The mechanism (GPU-type table,
job layout, durable-intent recovery) is hependpoints' (its docs/gpu-types.md and docs/slurm-recovery.md); the UVA table
is mu2e_config.UVA_GPU_TABLE.
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

GPU_TYPES = cfg.UVA_GPU_TYPES
CPU_RATES = cfg.UVA_CPU_RATES
AUTO, SLICE = cfg.AUTO, cfg.UVA_SLICE


def su_estimate(r: Resources, su_per_gpu_h: Optional[float] = None, su_per_core_h: Optional[float] = None) -> float:
    """Charge estimate: explicit flat rates when given, else UVA's rates for r.gpu_type (A100 when unknown)."""
    return _su_estimate(r, su_per_gpu_h, su_per_core_h, gpu_table=cfg.UVA_GPU_TABLE)


class SlurmSSHEndpoint(_SlurmSSHEndpoint):
    HOST_VARS = ("FASTSIMKIT_SLURM_HOST",)
    ACCOUNT_VARS = ("FASTSIMKIT_SLURM_ACCOUNT",)
    WORKDIR_VARS = ("FASTSIMKIT_WORKDIR",)
    CONFIG_HINT = site.HINT
    UNREACHABLE_HINT = cfg.UVA_UNREACHABLE_HINT

    def __init__(self, host: str | None = None, account: str | None = None, partition: str = "gpu",
                 workdir: str | None = None, state_dir: str | Path = cfg.UVA_STATE_DIR,
                 max_su_per_job: float = cfg.UVA_MAX_SU_PER_JOB, runner: Optional[Runner] = None, name: str = "uva",
                 su_per_gpu_h: Optional[float] = None, su_per_core_h: Optional[float] = None,
                 timeout_s: float = 120.0, stream_runner=None, gpu_type: Optional[str] = AUTO):
        workdir = workdir or site.setting("FASTSIMKIT_WORKDIR") or site.scratch() + "/kit"
        # partition "gpu" is the default: each UVA type's own partition applies (gpu-mig for the slice)
        super().__init__(host=host, account=account, partition=None if partition == "gpu" else partition,
                         workdir=workdir, state_dir=state_dir, max_su_per_job=max_su_per_job, runner=runner,
                         name=name, su_per_gpu_h=su_per_gpu_h, su_per_core_h=su_per_core_h, timeout_s=timeout_s,
                         stream_runner=stream_runner, cpu_partition="standard", environment=cfg.ENVIRONMENT,
                         gpu_table=cfg.UVA_GPU_TABLE, gpu_type=gpu_type)
        self.su_rates = (su_per_gpu_h, su_per_core_h)   # flat rates only when given; chainpoint reads this pair
