#!/usr/bin/env python3
"""Inside an EAF server: what limits its lifetime (read-only). Pod spec (activeDeadlineSeconds, annotations),
the hub's idle-culler service if the hub lists it to this server, and the server's own JupyterHub env.
    python3 core/eaf/lifetime_probe.py
"""
import json
import os
import ssl
import urllib.request

SA = "/var/run/secrets/kubernetes.io/serviceaccount"


def k8s(path):
    tok = open(f"{SA}/token").read().strip()
    ctx = ssl.create_default_context(cafile=f"{SA}/ca.crt")
    api = "https://%s:%s" % (os.environ["KUBERNETES_SERVICE_HOST"], os.environ.get("KUBERNETES_SERVICE_PORT", "443"))
    req = urllib.request.Request(api + path, headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
            return r.status, json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return getattr(e, "code", None), str(e)[:200]


def hub(path):
    tok = os.environ.get("JUPYTERHUB_API_TOKEN", "")
    req = urllib.request.Request(os.environ["JUPYTERHUB_API_URL"] + path, headers={"Authorization": "token " + tok})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return getattr(e, "code", None), str(e)[:200]


ns = open(f"{SA}/namespace").read().strip()
st, pod = k8s(f"/api/v1/namespaces/{ns}/pods/{os.environ.get('HOSTNAME', '')}")
if isinstance(pod, dict):
    sp, md = pod["spec"], pod["metadata"]
    print("activeDeadlineSeconds:", sp.get("activeDeadlineSeconds"), " restartPolicy:", sp.get("restartPolicy"))
    print("startTime:", pod["status"].get("startTime"), " priorityClass:", sp.get("priorityClassName"))
    print("annotations:", {k: v for k, v in (md.get("annotations") or {}).items() if "token" not in k.lower()})
    print("labels:", md.get("labels"))
st, svc = hub("/services")
print("hub /services:", st, json.dumps(svc)[:800] if isinstance(svc, (dict, list)) else svc)
st, info = hub("/info")
print("hub /info:", st, json.dumps(info)[:300] if isinstance(info, dict) else info)
st, pods = k8s(f"/api/v1/namespaces/{ns}/pods")
if isinstance(pods, dict):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    ages = []
    for p in pods["items"]:
        if p["metadata"]["name"].startswith("jupyter-") and p["status"].get("startTime"):
            t = dt.datetime.fromisoformat(p["status"]["startTime"].replace("Z", "+00:00"))
            ages.append((now - t).total_seconds() / 3600)
        elif not p["metadata"]["name"].startswith("jupyter-"):
            cs = p["spec"].get("containers", [])
            print("other pod:", p["metadata"]["name"], [(c.get("name"), " ".join(c.get("args") or c.get("command") or [])[:300]) for c in cs])
    ages.sort()
    print(f"user servers: {len(ages)}; age hours min {ages[0]:.1f} median {ages[len(ages)//2]:.1f} max {ages[-1]:.1f}" if ages else "no user servers")
    print("ages (h):", [round(a, 1) for a in ages])
for k in sorted(os.environ):
    if k.startswith(("JUPYTERHUB_", "JPY_", "CULL", "SHUTDOWN")) and "TOKEN" not in k:
        print(k, "=", os.environ[k])
