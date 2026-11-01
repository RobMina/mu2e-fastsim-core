"""Retained source snapshots of the mu2e-fastsim umbrella: hependpoints.snapshots with the Mu2e repository.

The umbrella URL and module set are Mu2e configuration (mu2e_config); only core/crv receive an optional branch
override. The URL is read from this module at call time (tests point it at a local fixture).
"""
from __future__ import annotations

from hependpoints import snapshots as _snapshots
from hependpoints.snapshots import SourceRepo  # noqa: F401

from . import mu2e_config as cfg

UMBRELLA_URL = cfg.UMBRELLA_URL
_MODULES = set(cfg.MODULES)


def _repo():
    return cfg.source_repo(UMBRELLA_URL)


def prepare_code(endpoint, base, ref, modules=cfg.MODULES, branch="") -> dict:
    """Create a unique, retained snapshot, returning exact umbrella and recursive pins (hependpoints.snapshots)."""
    return _snapshots.prepare_code(endpoint, base, ref, _repo(), modules=modules, branch=branch)


def code_guard(snapshot) -> str:
    """Non-mutating Bash prefix verifying the snapshot before the runtime (hependpoints.snapshots.code_guard)."""
    return _snapshots.code_guard(snapshot, _repo())
