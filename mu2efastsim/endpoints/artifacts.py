"""Durable execution bindings for streamed tasks, separate from BO measurement identity: hependpoints.artifacts with
the Mu2e output variable ($M2FS_OUT), the umbrella snapshot guard and the beam geometry identity."""
from __future__ import annotations

from hependpoints import artifacts as _artifacts
from hependpoints.artifacts import (FINGERPRINT, _binding_digest, cache_command, checked_record,  # noqa: F401
                                    consumption_guard, digest, fingerprint_command, input_digest, make_manifest,
                                    parse_fingerprint, resolve_image, resolve_inputs, target_identity, validate_manifest,
                                    verify_cache, verify_result)

from . import mu2e_config as cfg


def runtime_prefix(manifest):
    """Guard the runtime and emit the exact admission manifest into $M2FS_OUT."""
    from . import snapshots
    return _artifacts.runtime_prefix(manifest, guard=lambda code: snapshots.code_guard(code), out_var=cfg.OUT_VAR)


def binding(directory, name, params, target, prepare, geometry=None):
    """Persist one original rendering before endpoint dispatch, under a cooperating-process lock."""
    from .geometry import identity
    return _artifacts.binding(directory, name, params, target, prepare, geometry, identity=identity,
                              label="fastsimkit")
