"""Explicit, checksum-pinned input inventory, verification and opt-in local/HTTPS fetch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import urlopen


BLOCK = 1024 * 1024


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid relative path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or any(s in ("", ".", "..") for s in value.split("/")):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def under(root, name):
    root = Path(root).resolve()
    dest = root / relative(name)
    if not dest.resolve().is_relative_to(root):
        raise ValueError(f"path escapes root: {name}")
    return dest


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def validate(obj):
    if not isinstance(obj, dict):
        raise ValueError("manifest must be a JSON object")
    if obj.get("schema_version") != 1 or not isinstance(obj.get("files"), list) or not obj["files"]:
        raise ValueError("expected nonempty schema_version=1 manifest with files list")
    seen = set()
    for row in obj["files"]:
        if not isinstance(row, dict):
            raise ValueError("each file must be an object")
        name = relative(row["path"])
        if name in seen:
            raise ValueError(f"duplicate manifest path: {name}")
        seen.add(name)
        if type(row.get("size_bytes")) is not int or row["size_bytes"] < 0:
            raise ValueError(f"invalid size: {name}")
        if not re.fullmatch("[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError(f"invalid sha256: {name}")
        for key in ("provenance", "redistribution"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"missing {key}: {name}")
    return obj


def manifest(path):
    return validate(json.loads(Path(path).read_text()))


def status(root, row):
    path = under(root, row["path"])
    try:
        if not path.is_file():
            return "missing"
        if path.stat().st_size != row["size_bytes"]:
            return "size-mismatch"
        return "ok" if digest(path) == row["sha256"] else "hash-mismatch"
    except OSError as exc:
        return f"unreadable: {exc}"


def check(obj, root):
    return [{"path": row["path"], "status": status(root, row)} for row in obj["files"]]


def inventory(spec):
    rows = []
    for item in spec["files"]:
        row = {k: v for k, v in item.items() if k != "source"}
        src = Path(item["source"])
        before = src.stat()
        row.update(size_bytes=before.st_size, sha256=digest(src))
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"input changed while hashing: {src}")
        rows.append(row)
    return validate({"schema_version": 1, "release": spec["release"], "files": rows})


def fetch(obj, root, source_dir=None, base_url=None, execute=False):
    if bool(source_dir) == bool(base_url):
        raise ValueError("choose exactly one approved --source-dir or --base-url")
    if base_url:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base URL must be HTTPS without credentials, query or fragment")
    results = []
    for row in obj["files"]:
        state = status(root, row)
        rec = {"path": row["path"], "status": state}
        results.append(rec)
        if state == "ok":
            continue
        if state != "missing":
            raise ValueError(f"refusing to overwrite {row['path']}: {state}")
        dest = under(root, row["path"])
        if dest.exists() or dest.is_symlink():
            raise ValueError(f"refusing existing destination: {dest}")
        rec["status"] = "would-fetch"
        if not execute:
            continue
        if row["redistribution"] != "approved":
            raise ValueError(f"redistribution not approved: {row['path']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = open(under(source_dir, row["path"]), "rb") if source_dir else urlopen(base_url.rstrip("/") + "/" + quote(row["path"], safe="/"), timeout=60)
        tmp = None
        try:
            with src, tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".input-", delete=False) as out:
                tmp = Path(out.name)
                if base_url and urlparse(src.geturl()).scheme != "https":
                    raise ValueError("refusing non-HTTPS redirect")
                h, n = hashlib.sha256(), 0
                for block in iter(lambda: src.read(BLOCK), b""):
                    n += len(block)
                    if n > row["size_bytes"]:
                        raise ValueError(f"oversize download: {row['path']}")
                    h.update(block)
                    out.write(block)
            if n != row["size_bytes"] or h.hexdigest() != row["sha256"]:
                raise ValueError(f"checksum/size mismatch: {row['path']}")
            os.link(tmp, dest)
            rec["status"] = "fetched"
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="hash explicit sources; writes no input copies")
    inv.add_argument("--spec", required=True)
    inv.add_argument("--out", required=True)
    for name in ("check-inputs", "fetch-inputs"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--manifest", required=True)
        cmd.add_argument("--root", required=True)
        if name == "fetch-inputs":
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--source-dir")
            group.add_argument("--base-url")
            cmd.add_argument("--execute", action="store_true", help="default is a read-only dry run")
    a = ap.parse_args(argv)
    try:
        if a.command == "inventory":
            obj = inventory(json.loads(Path(a.spec).read_text()))
            with open(a.out, "x") as out:
                json.dump(obj, out, indent=2)
                out.write("\n")
            manifest(a.out)
            return 0
        obj = manifest(a.manifest)
        rows = check(obj, a.root) if a.command == "check-inputs" else fetch(obj, a.root, a.source_dir, a.base_url, a.execute)
        print(json.dumps({"release": obj.get("release"), "files": rows}, indent=2))
        return 0 if all(r["status"] in ("ok", "fetched") for r in rows) else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
