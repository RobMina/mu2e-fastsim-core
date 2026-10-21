"""Compute endpoints for the fast chain: one interface (submit / status / fetch / cancel) per site.

The Mu2eBO kit (``mu2efastsim.endpoints.kit``) sits on top and speaks the evaluator contract; an endpoint only moves a
job script to a site, runs it and brings its output directory back. States are the contract's
(``working``, ``completed``, ``failed``, ``cancelled``). A handle is the caller's name for the job, and submitting the
same name twice with the same script returns the same job (idempotent), with a different script it is refused.

    slurm_ssh   Slurm over SSH (UVA Rivanna/Afton; any Slurm site reachable by ssh with a key)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

STATES = ("working", "completed", "failed", "cancelled")


class EndpointError(RuntimeError):
    """A failure the caller should treat as a failed or refused call (transport, refusal, budget)."""


@dataclass(frozen=True)
class Resources:
    """What one job asks for. gpus counts devices of `gpu_type` ("" = any)."""
    cores: int = 8
    mem_gb: int = 48
    gpus: int = 1
    gpu_type: str = "a100"
    minutes: int = 60


@dataclass
class JobStatus:
    state: str
    message: str = ""
    site_state: str = ""
    job_id: str = ""


@dataclass
class JobRecord:
    """What an endpoint remembers about a submitted job (one JSON file per handle in its state dir)."""
    handle: str
    spec_sha: str
    job_id: str
    remote_dir: str
    submitted_at: float
    resources: Dict = field(default_factory=dict)
    final: Optional[Dict] = None


class Endpoint(Protocol):
    name: str

    def submit(self, handle: str, script: str, resources: Resources) -> JobRecord: ...

    def status(self, handle: str) -> JobStatus: ...

    def fetch(self, handle: str, dest: Path) -> List[Path]: ...

    def cancel(self, handle: str) -> str: ...


def spec_sha(script: str, resources: Resources) -> str:
    """Identity of a submission: the script text and the resources."""
    blob = json.dumps({"script": script, "resources": asdict(resources)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class StateDir:
    """Local job records, one JSON per handle, written atomically."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, handle: str) -> Path:
        if not handle or "/" in handle or handle.startswith("."):
            raise EndpointError(f"bad handle {handle!r}")
        return self.root / f"{handle}.json"

    def get(self, handle: str) -> Optional[JobRecord]:
        p = self._path(handle)
        return JobRecord(**json.loads(p.read_text())) if p.exists() else None

    def put(self, rec: JobRecord) -> None:
        p = self._path(rec.handle)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(rec), indent=1))
        os.replace(tmp, p)


def get(name: str, **kw) -> Endpoint:
    """An endpoint by site name, configured from keyword arguments (see each module)."""
    if name in ("uva", "slurm_ssh"):
        from .slurm_ssh import SlurmSSHEndpoint
        return SlurmSSHEndpoint(**kw)
    raise EndpointError(f"unknown endpoint {name!r}; known: uva (slurm_ssh)")
