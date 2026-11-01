"""Compute endpoints for the fast chain: one interface (submit / status / fetch / cancel) per site.

The generic parts (the Endpoint protocol, records, intents, the Slurm and EAF adapters, snapshots, artifact bindings)
live in the experiment-neutral hep-endpoints package (``hependpoints``); this package configures them for Mu2e
(``mu2e_config``, ``mu2efastsim.site``) and keeps the old import paths. The Mu2eBO kit (``mu2efastsim.endpoints.kit``)
sits on top and speaks the evaluator contract. States are the contract's (``working``, ``completed``, ``failed``,
``cancelled``). A handle is the caller's name for the job, and submitting the same name twice with the same script
returns the same job (idempotent), with a different script it is refused.

    slurm_ssh   Slurm over SSH (UVA Rivanna/Afton; any Slurm site reachable by ssh with a key)
    eaf         Fermilab EAF: the user's JupyterHub server on an A100 MIG slice, driven with a hub token from a GPVM
"""
from __future__ import annotations

from dataclasses import dataclass

import hependpoints as _hep
from hependpoints import STATES, Endpoint, EndpointError, JobRecord, JobStatus, StateDir, spec_sha  # noqa: F401


@dataclass(frozen=True)
class Resources(_hep.Resources):
    """What one fast-chain job asks for (the Mu2e defaults: one A100 with 8 cores). gpus counts devices of
    `gpu_type` ("" = any). Same fields as hependpoints.Resources, so spec_sha is unchanged."""
    cores: int = 8
    mem_gb: int = 48
    gpus: int = 1
    gpu_type: str = "a100"
    minutes: int = 60


def get(name: str, **kw) -> Endpoint:
    """An endpoint by site name, configured from keyword arguments (see each module)."""
    if name in ("uva", "slurm_ssh"):
        from .slurm_ssh import SlurmSSHEndpoint
        return SlurmSSHEndpoint(**kw)
    if name == "eaf":
        from .eaf import EAFEndpoint
        return EAFEndpoint(**kw)
    raise EndpointError(f"unknown endpoint {name!r}; known: uva (slurm_ssh), eaf")
