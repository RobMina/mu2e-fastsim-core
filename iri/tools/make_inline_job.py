#!/usr/bin/env python3
"""Build an IRI JobSpec that ships a local bash script inline, base64-encoded.

ALCF IRI (2026-10) has no file upload and no pre_launch, allows at most 2 arguments for /bin/bash, and flattens
newlines in `bash -c` strings to "; " (which breaks heredocs). So the script travels as one base64 word that is
decoded to <directory>/<name> on the node and run from there; its arguments become `set --` in the decoded copy.
The template may use ${M2FS_EAGLE}, ${ALCF_PROJECT}, ${ESNET_WORK}, ${ESNET_PROJECT} and ${ESNET_IMAGE_TAG} (site.env):
they are expanded here, because the API does not expand variables, and M2FS_EAGLE is exported at the top of the shipped script.
"""
import argparse
import base64
import json
import os
import shlex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="JobSpec JSON (directory, stdout/stderr, resources, attributes)")
    ap.add_argument("script", help="local bash script to ship")
    ap.add_argument("--out", required=True)
    ap.add_argument("--args", nargs="*", default=[], help="positional arguments for the script")
    args = ap.parse_args()
    raw = os.path.expandvars(open(args.template).read())
    if "${" in raw:
        raise SystemExit(f"unexpanded variable in {args.template}: source site.env first")
    spec = json.loads(raw)
    d = spec["directory"].rstrip("/")
    dest = f"{d}/{os.path.basename(args.script)}"
    body = open(args.script).read()
    head = []
    if os.environ.get("M2FS_EAGLE"):
        head.append("export M2FS_EAGLE=" + shlex.quote(os.environ["M2FS_EAGLE"]))
    if args.args:
        head.append("set -- " + " ".join(shlex.quote(a) for a in args.args))
    if head:
        lines = body.split("\n", 1)
        body = "\n".join([lines[0], *head, lines[1] if len(lines) > 1 else ""])
    b64 = base64.b64encode(body.encode()).decode()
    spec["executable"] = "/bin/bash"
    spec["arguments"] = ["-c", f"cd {d} && echo {b64} | base64 -d > {dest} && exec bash {dest}"]
    json.dump(spec, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
