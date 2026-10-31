"""Bounded rendered beam-target FileRefs; not a general SimpleConfig or GDML loader."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
from urllib.parse import unquote, urlsplit

BASE = "Offline/Mu2eG4/geom/geom_common.txt"
FORMAT = "offline_simpleconfig"
MAX_BYTES = 256 * 1024
SCALARS = {"stoppingTarget.deltaZ", "stoppingTarget.z0InMu2e", "stoppingTarget.holeRadius",
           "protonabsorber.distFromTargetEnd"}
VECTORS = {"stoppingTarget.radii", "stoppingTarget.halfThicknesses", "stoppingTarget.zVars",
           "stoppingTarget.holeRadii"}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
DECLARATION = re.compile(r"\s*(double|vector<\s*double\s*>)\s+([A-Za-z][A-Za-z0-9_.]*)\s*=\s*([^;]+);", re.S)
HEX = re.compile(r"[0-9a-f]{64}")


def reference(files, inputs, params):
    """Validate the descriptor without reading it; retries can reuse the bound bytes."""
    if not isinstance(files, list) or not isinstance(inputs, list):
        raise ValueError("fastsimkit: files and inputs must be lists")
    refs = files + inputs
    if not refs:
        return None
    if params.get("task") != "beam_point":
        raise ValueError("fastsimkit: files and inputs support only beam_point target geometry")
    if len(refs) != 1:
        raise ValueError("fastsimkit: beam_point accepts exactly one geometry FileRef across files and inputs")
    if "geom" in params:
        raise ValueError("fastsimkit: geom parameter conflicts with geometry FileRef")
    ref = refs[0]
    required = {"name", "kind", "uri", "sha256", "metadata"}
    if not isinstance(ref, dict) or set(ref) != required or ref.get("kind") != "geom" or ref.get("name") != "geom":
        raise ValueError("fastsimkit: geometry FileRef needs name=geom, kind=geom, uri, sha256 and metadata")
    if not isinstance(ref["sha256"], str) or not HEX.fullmatch(ref["sha256"]):
        raise ValueError("fastsimkit: geometry sha256 must be 64 lowercase hexadecimal characters")
    meta = ref["metadata"]
    if (not isinstance(meta, dict) or set(meta) != {"format", "schema_version", "renderer", "renderer_sha256"}
            or meta.get("format") != FORMAT or meta.get("renderer") != FORMAT
            or type(meta.get("schema_version")) is not int or meta["schema_version"] != 1
            or not isinstance(meta.get("renderer_sha256"), str) or not HEX.fullmatch(meta["renderer_sha256"])):
        raise ValueError("fastsimkit: geometry metadata requires offline_simpleconfig format/renderer, schema_version=1 and renderer_sha256")
    local_path(ref["uri"])
    return {**ref, "metadata": dict(meta)}


def _on_dcache(path):
    """True for a path below the top-level dCache (pnfs) mount."""
    parts = path.split("/", 2)
    return len(parts) == 3 and parts[0] == "" and parts[1] == "pnfs"


def local_path(uri):
    if not isinstance(uri, str) or any(ord(c) < 32 or ord(c) == 127 for c in uri):
        raise ValueError("fastsimkit: invalid geometry URI")
    try:
        parts = urlsplit(uri)
    except ValueError as exc:
        raise ValueError("fastsimkit: invalid geometry URI") from exc
    path = unquote(parts.path, errors="strict")
    if (parts.scheme != "file" or parts.netloc not in ("", "localhost") or parts.query or parts.fragment
            or not path.startswith("/") or path.startswith("//") or os.path.normpath(path) != path
            or any(ord(c) < 32 or ord(c) == 127 for c in path) or _on_dcache(path)):
        raise ValueError("fastsimkit: geometry must be a normalized local file URI readable by the kit host, "
                         "not on dCache (pnfs)")
    return Path(path)


def identity(artifact):
    if artifact is None:
        return None
    ref = artifact.get("reference", artifact)
    return {"sha256": ref["sha256"], "metadata": dict(ref["metadata"])}


def validate_text(data):
    """Only one approved base include followed by finite numeric target overrides is admitted."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("fastsimkit: geometry must be nonempty and at most 256 KiB")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("fastsimkit: geometry must be ASCII") from exc
    if any(ord(c) < 32 and c not in "\t\r\n" for c in text):
        raise ValueError("fastsimkit: geometry contains control characters")
    clean = re.sub(r"//[^\r\n]*", "", text).strip()
    include = re.match(r'\A#include\s+"' + re.escape(BASE) + r'"[ \t]*(?:\r?\n|$)', clean)
    if include is None:
        raise ValueError(f"fastsimkit: geometry must start with exactly one #include of {BASE}")
    pos, keys = include.end(), set()
    while clean[pos:].strip():
        match = DECLARATION.match(clean, pos)
        if match is None:
            raise ValueError("fastsimkit: unsupported geometry syntax or additional include")
        typ, key, raw = match.groups()
        is_vector = typ.startswith("vector")
        if key not in (VECTORS if is_vector else SCALARS) or key in keys:
            raise ValueError(f"fastsimkit: unsupported, mistyped or duplicate target geometry key {key!r}")
        raw = raw.strip()
        if is_vector:
            if not (raw.startswith("{") and raw.endswith("}")):
                raise ValueError(f"fastsimkit: {key} must be a numeric vector")
            values = raw[1:-1].split(",")
        else:
            values = [raw]
        if any(not re.fullmatch(NUMBER, v.strip()) or not math.isfinite(float(v)) for v in values):
            raise ValueError(f"fastsimkit: {key} requires finite numeric literals")
        numbers = [float(v) for v in values]
        if key in {"stoppingTarget.radii", "stoppingTarget.halfThicknesses", "stoppingTarget.deltaZ"} and min(numbers) <= 0:
            raise ValueError(f"fastsimkit: {key} must be positive")
        if key in {"stoppingTarget.holeRadius", "stoppingTarget.holeRadii"} and min(numbers) < 0:
            raise ValueError(f"fastsimkit: {key} must be nonnegative")
        keys.add(key)
        pos = match.end()
    if {"stoppingTarget.holeRadius", "stoppingTarget.holeRadii"} <= keys:
        raise ValueError("fastsimkit: scalar holeRadius is shadowed by holeRadii")
    return text, sorted(keys)


def resolve(ref, saved=None):
    if ref is None:
        if saved is not None:
            raise ValueError("fastsimkit: geometry differs from the bound request")
        return None
    if saved is not None:
        if identity(ref) != identity(saved):
            raise ValueError("fastsimkit: geometry differs from the bound request")
        validate_artifact(saved)
        return saved
    path = local_path(ref["uri"])
    for parent in (*reversed(path.parents), path):
        try:
            if stat.S_ISLNK(parent.lstat().st_mode):
                raise ValueError("fastsimkit: geometry URI may not traverse symlinks")
        except OSError as exc:
            raise ValueError(f"fastsimkit: cannot read geometry on kit host: {path}") from exc
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
                raise ValueError("fastsimkit: geometry must be a bounded regular file")
            data = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
            fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if fields(before) != fields(after):
                raise ValueError("fastsimkit: geometry changed while reading")
    except OSError as exc:
        raise ValueError(f"fastsimkit: cannot read geometry on kit host: {path}") from exc
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("fastsimkit: geometry digest does not match FileRef")
    text, keys = validate_text(data)
    return {"reference": ref, "text": text, "size": len(data), "parameter_keys": keys,
            "base_include": BASE, "coverage": "render-only; base/include dependencies remain site-resolved"}


def validate_artifact(artifact):
    ref = reference([artifact["reference"]], [], {"task": "beam_point"})
    text, keys = validate_text(artifact["text"].encode("ascii"))
    if (len(text.encode("ascii")) != artifact["size"] or keys != artifact["parameter_keys"]
            or hashlib.sha256(text.encode("ascii")).hexdigest() != ref["sha256"] or artifact["base_include"] != BASE):
        raise ValueError("fastsimkit: invalid bound geometry artifact")


def stage_shell(artifact):
    """Retain numeric candidate parameters with the job, never an arbitrary input-file payload."""
    validate_artifact(artifact)
    proof = {k: v for k, v in artifact.items() if k != "text"}
    return ("M2FS_OUT=$(readlink -f \"$M2FS_OUT\") || exit 3\n"
            + "M2FS_GEOMETRY=\"$M2FS_OUT/geometry.txt\"\n"
            + f"printf '%s' {shlex.quote(artifact['text'])} > \"$M2FS_GEOMETRY\" || exit 3\n"
            + f"printf '%s  %s\\n' {shlex.quote(artifact['reference']['sha256'])} \"$M2FS_GEOMETRY\" | sha256sum -c - >/dev/null || exit 3\n"
            + "chmod 400 \"$M2FS_GEOMETRY\" || exit 3\n"
            + f"printf '%s\\n' {shlex.quote(json.dumps(proof, sort_keys=True))} > \"$M2FS_OUT/geometry-artifact.json\" || exit 3\n")


def verify_result(out, artifact):
    validate_artifact(artifact)
    try:
        data = (Path(out) / "geometry.txt").read_bytes()
        proof = json.loads((Path(out) / "geometry-artifact.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("fastsimkit: missing or invalid consumed geometry evidence") from exc
    if data != artifact["text"].encode("ascii") or proof != {k: v for k, v in artifact.items() if k != "text"}:
        raise ValueError("fastsimkit: consumed geometry differs from the bound artifact")
    return {**artifact["reference"], "uri": (Path(out) / "geometry.txt").resolve().as_uri()}
