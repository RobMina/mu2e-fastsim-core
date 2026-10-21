#!/usr/bin/env python3
"""Globus transfers for the ESnet East collection (and any other collection) with globus-sdk >= 4, no CLI.

Login is two steps so it works from a non-interactive agent shell:
  globus_esnet.py login-url            # prints a URL; the user opens it, logs in, copies the code
  globus_esnet.py login-code <CODE>    # exchanges it; refresh tokens saved (mode 600) to $GLOBUS_TOKEN_FILE
Then:
  globus_esnet.py whoami | ls <coll> <path> | search <text> | info <coll>
  globus_esnet.py transfer <src coll> <src path> <dst coll> <dst path> [--recursive] [--label L]
  globus_esnet.py task <task id> [--wait SEC]
<coll> is a collection UUID or the alias "esnet" (AmSC Genesis Hackathon 2610 ESnet East; its "/" is the IRI path
/data/home/hackathon2610). Collections that need a data_access consent (GCS v5 mapped) go in GLOBUS_DATA_ACCESS
(comma-separated UUIDs or aliases; not the ESnet one, a guest collection with no data_access scope) before login-url; ConsentRequired errors print which one to add.
"""
import argparse
import json
import os
import sys
import time

import globus_sdk
from globus_sdk.scopes import TransferScopes

# Public native-app client of the Globus SDK tutorials; set GLOBUS_CLIENT_ID to your own registered native app.
CLIENT_ID = os.environ.get("GLOBUS_CLIENT_ID", "61338d24-54d5-408f-a10d-66c06b59f6d2")
TOKENS = os.path.expanduser(os.environ.get("GLOBUS_TOKEN_FILE", "~/.globus/mu2efastsim_transfer_tokens.json"))
FLOW = TOKENS + ".flow"
ALIASES = {"esnet": "a4d0333d-b9f0-442f-8d14-7d3a32596cb9"}
TRANSFER_RS = "transfer.api.globus.org"


def coll(c):
    return ALIASES.get(c, c)


def scopes():
    das = [d.strip() for d in os.environ.get("GLOBUS_DATA_ACCESS", "").split(",") if d.strip()]
    s = TransferScopes.all
    for d in das:
        s = s.with_dependency(globus_sdk.Scope(f"https://auth.globus.org/scopes/{coll(d)}/data_access", optional=True))
    return [globus_sdk.Scope("openid"), globus_sdk.Scope("email"), s]


def auth_client():
    return globus_sdk.NativeAppAuthClient(CLIENT_ID)


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)


def transfer_client():
    if not os.path.exists(TOKENS):
        sys.exit(f"no tokens in {TOKENS}: run login-url, then login-code")
    tok = json.load(open(TOKENS))[TRANSFER_RS]

    def on_refresh(resp):
        d = json.load(open(TOKENS))
        d[TRANSFER_RS].update(resp.by_resource_server[TRANSFER_RS])
        save(TOKENS, d)

    authz = globus_sdk.RefreshTokenAuthorizer(tok["refresh_token"], auth_client(), access_token=tok["access_token"],
                                              expires_at=tok["expires_at_seconds"], on_refresh=on_refresh)
    return globus_sdk.TransferClient(authorizer=authz)


def consent_hint(e):
    if isinstance(e, globus_sdk.TransferAPIError) and e.info.consent_required:
        print("ConsentRequired: required scopes", e.info.consent_required.required_scopes, file=sys.stderr)
        print("Add the collection to GLOBUS_DATA_ACCESS and redo login-url / login-code.", file=sys.stderr)
    raise e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login-url")
    sub.add_parser("login-code").add_argument("code")
    sub.add_parser("whoami")
    p = sub.add_parser("ls"); p.add_argument("coll"); p.add_argument("path", nargs="?", default="/")
    sub.add_parser("search").add_argument("text")
    sub.add_parser("info").add_argument("coll")
    p = sub.add_parser("transfer")
    for a in ("src", "src_path", "dst", "dst_path"):
        p.add_argument(a)
    p.add_argument("--recursive", action="store_true"); p.add_argument("--label", default="mu2efastsim")
    p = sub.add_parser("task"); p.add_argument("task_id"); p.add_argument("--wait", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "login-url":
        flow = auth_client().oauth2_start_flow(requested_scopes=scopes(), refresh_tokens=True)
        save(FLOW, {"verifier": flow.verifier, "scopes": [str(s) for s in scopes()]})
        print(flow.get_authorize_url(query_params={"prompt": "login"}))
        return
    if a.cmd == "login-code":
        f = json.load(open(FLOW))
        ac = auth_client()
        ac.oauth2_start_flow(requested_scopes=f["scopes"], refresh_tokens=True, verifier=f["verifier"])
        resp = ac.oauth2_exchange_code_for_tokens(a.code.strip())
        save(TOKENS, resp.by_resource_server)
        os.remove(FLOW)
        print("saved tokens for", sorted(resp.by_resource_server), "to", TOKENS)
        return
    if a.cmd == "whoami":
        d = json.load(open(TOKENS))
        for rs, t in d.items():
            print(rs, "| scope", t["scope"], "| expires", time.strftime("%F %T", time.localtime(t["expires_at_seconds"])),
                  "| refresh", bool(t.get("refresh_token")))
        return

    tc = transfer_client()
    try:
        if a.cmd == "ls":
            for e in tc.operation_ls(coll(a.coll), path=a.path):
                print(f"{e['type'][0]} {e['size']:>12} {e['last_modified']} {e['name']}")
        elif a.cmd == "search":
            for e in tc.endpoint_search(a.text, limit=25):
                print(e["id"], "|", e["display_name"], "|", e.get("entity_type"), "|", e.get("owner_string"))
        elif a.cmd == "info":
            e = tc.get_endpoint(coll(a.coll)).data
            keys = ("id", "display_name", "entity_type", "owner_string", "high_assurance", "gcs_version",
                    "host_endpoint_display_name", "non_functional", "description")
            print(json.dumps({k: e.get(k) for k in keys}, indent=1))
        elif a.cmd == "transfer":
            td = globus_sdk.TransferData(coll(a.src), coll(a.dst), label=a.label, sync_level="checksum",
                                         verify_checksum=True)
            td.add_item(a.src_path, a.dst_path, recursive=a.recursive)
            print("task", tc.submit_transfer(td)["task_id"])
        elif a.cmd == "task":
            if a.wait:
                tc.task_wait(a.task_id, timeout=a.wait, polling_interval=10)
            t = tc.get_task(a.task_id).data
            print(json.dumps({k: t.get(k) for k in ("status", "nice_status", "files", "files_transferred", "bytes_transferred",
                                                    "faults", "source_endpoint_display_name",
                                                    "destination_endpoint_display_name")}, indent=1))
    except globus_sdk.GlobusAPIError as e:
        consent_hint(e)


if __name__ == "__main__":
    main()
