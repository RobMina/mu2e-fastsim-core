"""Durable local submission intent and locking for the single-cluster Slurm adapter.

An intent is a one-attempt fence, not proof that a scheduler accepted the job.
Once written, an unrecorded attempt may only be reconciled, never redispatched.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import EndpointError


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    for parent in path.resolve().parents:
        _sync_dir(parent)


def read_json(path):
    """Only a missing directory entry counts as absent state; damaged entries fail closed."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise EndpointError(f"invalid state entry {path}: expected a regular file")
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise EndpointError(f"unreadable or corrupt state {path}") from exc


def durable_json(path: Path, data):
    """Replace a JSON file only after syncing its unique same-directory temporary file."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def handle_key(handle):
    if not isinstance(handle, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", handle):
        raise EndpointError(f"bad handle {handle!r}")
    return hashlib.sha256(handle.encode()).hexdigest()


@contextmanager
def submission_lock(root: Path, handle: str):
    """Serialize cooperating processes sharing a state root; never unlink lock files."""
    key = handle_key(handle)
    _directory(root)
    directory = root / ".locks"
    _directory(directory)
    fd = os.open(directory / f"{key}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def request_digest(body, resources, target):
    payload = json.dumps({"body": body, "resources": resources, "target": target},
                         sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


class IntentStore:
    def __init__(self, root):
        self.root = Path(root) / ".intents"

    def _path(self, handle):
        return self.root / (handle_key(handle) + ".json")

    def get(self, handle):
        path = self._path(handle)
        intent = read_json(path)
        if intent is None:
            if path.exists():
                raise EndpointError(f"{handle!r}: corrupt submission intent")
            return None
        try:
            if not isinstance(intent, dict) or intent["version"] != 1 or intent["handle"] != handle:
                raise ValueError("invalid intent version or handle")
            if not (re.fullmatch(r"m2fs-v1-[0-9a-f]{32}", intent["token"])
                    and re.fullmatch(r"[0-9a-f]{64}", intent["request_sha"])
                    and re.fullmatch(r"[0-9a-f]{16}", intent["spec_sha"])
                    and isinstance(intent["target"], dict) and isinstance(intent["resources"], dict)
                    and isinstance(intent["remote_dir"], str)):
                raise ValueError("invalid intent identity")
            stamp = intent["created_at"]
            if isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp <= 0:
                raise ValueError("invalid intent timestamp")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise EndpointError(f"{handle!r}: corrupt submission intent; refusing dispatch or adoption") from exc
        return intent

    def create(self, handle, spec, request, target, resources, remote_dir, stamp):
        intent = {"version": 1, "handle": handle, "token": "m2fs-v1-" + uuid.uuid4().hex,
                  "spec_sha": spec, "request_sha": request, "target": target, "resources": resources,
                  "remote_dir": remote_dir, "created_at": stamp}
        _directory(self.root)
        durable_json(self._path(handle), intent)
        return intent
