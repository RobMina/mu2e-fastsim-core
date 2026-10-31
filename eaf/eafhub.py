#!/usr/bin/env python3
"""Drive a Fermilab EAF JupyterHub session from a GPVM with a hub API token (no browser after the token is made).

The token comes from $EAF_TOKEN_FILE (mode 600, outside any synced tree); make it once at <hub>/hub/token.
EAF mounts the FNAL shared storage ($FASTSIMKIT_EAF_SHARED_PREFIX, required by `run`), so `run` writes the job's log
there and the GPVM reads it directly.

    eafhub.py whoami | status | profiles [--html F] | start [key=value ...] | stop
    eafhub.py run --log <shared dir>/job.log [--wait] -- 'bash path/to/job_script.sh'
    eafhub.py wait --log <shared dir>/job.log
Needs: python3 with websocket-client (the Mu2e ana env has it). Hub: $EAF_HUB (default https://analytics-hub.fnal.gov).
"""
import argparse
import html
import json
import os
import re
import shlex
import ssl
import sys
import time
import urllib.error
import urllib.request

HUB = os.environ.get("EAF_HUB", "https://analytics-hub.fnal.gov").rstrip("/")


def token():
    f = os.environ.get("EAF_TOKEN_FILE")
    if not f or not os.path.exists(f):
        sys.exit("eafhub: set EAF_TOKEN_FILE to the file holding the hub API token (make one at %s/hub/token)" % HUB)
    if os.stat(f).st_mode & 0o077:
        sys.exit(f"eafhub: {f} is readable by others; chmod 600 it")
    return open(f).read().strip()


def call(method, path, body=None, raw=False):
    req = urllib.request.Request(HUB + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "token " + token(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
            data = r.read()
            return r.status, (data.decode(errors="replace") if raw else (json.loads(data) if data else None))
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        sys.exit(f"eafhub: {method} {path} -> HTTP {e.code}: {msg}")


def me():
    return call("GET", "/hub/api/user")[1]


def server(u):
    return (u.get("servers") or {}).get("")


def cmd_status(a):
    u = me()
    s = server(u)
    print(json.dumps({"name": u["name"], "server": s and {k: s.get(k) for k in ("ready", "pending", "url", "started",
                                                                                  "last_activity", "user_options")}},
                     indent=1))


def cmd_profiles(a):
    """Dump the spawn form's fields (profile slugs and option values) to choose `start` arguments."""
    u = me()
    page = open(a.html).read() if a.html else call("GET", f"/hub/spawn/{u['name']}", raw=True)[1]
    for m in re.finditer(r"<(input|option|select)\b([^>]*)>([^<]*)", page):
        at = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', m.group(2)))
        if m.group(1) == "select" or at.get("type") in ("radio", "checkbox", "hidden") or m.group(1) == "option":
            label = html.unescape(m.group(3)).strip()
            print(m.group(1), {k: at[k] for k in ("name", "value", "id", "checked", "selected") if k in at}, label[:80])


def cmd_start(a):
    u = me()
    if server(u) and server(u).get("ready"):
        print("already running:", server(u)["url"]); return
    opts = dict(kv.split("=", 1) for kv in a.options)
    st, _ = call("POST", f"/hub/api/users/{u['name']}/server", opts or None)
    print("start requested, HTTP", st, "options", opts)
    for _ in range(a.timeout // 10):
        s = server(me())
        if s and s.get("ready"):
            print("ready:", s["url"]); return
        time.sleep(10)
    sys.exit("eafhub: server not ready yet; check `status`")


def cmd_stop(a):
    u = me()
    st, _ = call("DELETE", f"/hub/api/users/{u['name']}/server")
    print("stop requested, HTTP", st)


def wait_log(log, timeout):
    done = log + ".done"
    t0 = time.time()
    while not os.path.exists(done):
        if time.time() - t0 > timeout:
            sys.exit(f"eafhub: no {done} after {timeout} s; the job may still be running")
        time.sleep(15)
    rc = open(done).read().strip()
    print(open(log).read()[-4000:])
    print(rc)
    return int(rc.split("=")[-1]) if rc.split("=")[-1].isdigit() else 1


def cmd_run(a):
    """Start the command detached in a server terminal; its output goes to --log (shared storage), exit code to --log.done."""
    import websocket
    s = server(me())
    if not (s and s.get("ready")):
        sys.exit("eafhub: no running server; `start` first")
    log = os.path.abspath(a.log)
    shared = os.environ.get("FASTSIMKIT_EAF_SHARED_PREFIX")
    if not shared:
        sys.exit("eafhub: set FASTSIMKIT_EAF_SHARED_PREFIX (storage mounted by both EAF and the GPVM)")
    if not log.startswith(shared):
        sys.exit(f"eafhub: --log must be under {shared} (shared by EAF and the GPVM)")
    for f in (log, log + ".done"):
        if os.path.exists(f):
            os.remove(f)
    inner = f"{a.command}; echo exit=$? > {shlex.quote(log + '.done')}"
    line = f"setsid nohup bash -lc {shlex.quote(inner)} > {shlex.quote(log)} 2>&1 < /dev/null &"
    st, term = call("POST", s["url"] + "api/terminals", {})
    ws = websocket.create_connection(HUB.replace("https://", "wss://") + s["url"] + f"terminals/websocket/{term['name']}",
                                     header=["Authorization: token " + token()], timeout=30)
    ws.send(json.dumps(["stdin", line + "\r"]))
    time.sleep(2)
    ws.close()
    print(f"started in terminal {term['name']}; log {log}")
    if a.wait:
        sys.exit(wait_log(log, a.timeout))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("whoami").set_defaults(f=lambda a: print(json.dumps(me(), indent=1)))
    sp.add_parser("status").set_defaults(f=cmd_status)
    p = sp.add_parser("profiles"); p.add_argument("--html", help="parse a saved spawn page instead"); p.set_defaults(f=cmd_profiles)
    p = sp.add_parser("start"); p.add_argument("options", nargs="*", help="user_options as key=value (see `profiles`)")
    p.add_argument("--timeout", type=int, default=900); p.set_defaults(f=cmd_start)
    sp.add_parser("stop").set_defaults(f=cmd_stop)
    p = sp.add_parser("run"); p.add_argument("--log", required=True); p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout", type=int, default=7200); p.add_argument("command"); p.set_defaults(f=cmd_run)
    p = sp.add_parser("wait"); p.add_argument("--log", required=True); p.add_argument("--timeout", type=int, default=7200)
    p.set_defaults(f=lambda a: sys.exit(wait_log(os.path.abspath(a.log), a.timeout)))
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
