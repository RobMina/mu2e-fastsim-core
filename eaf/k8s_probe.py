#!/usr/bin/env python3
"""Inside an EAF server: what the pod's Kubernetes service account may see about GPU nodes (read-only).
Prints the pod's own placement and, where RBAC allows, each node's allocatable MIG resources and what is in use.
    python3 core/eaf/k8s_probe.py
"""
import json
import os
import ssl
import urllib.error
import urllib.request

SA = "/var/run/secrets/kubernetes.io/serviceaccount"
API = "https://%s:%s" % (os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc"),
                         os.environ.get("KUBERNETES_SERVICE_PORT", "443"))


def get(path):
    tok = open(f"{SA}/token").read().strip() if os.path.exists(f"{SA}/token") else ""
    ctx = ssl.create_default_context(cafile=f"{SA}/ca.crt") if os.path.exists(f"{SA}/ca.crt") else ssl._create_unverified_context()
    req = urllib.request.Request(API + path, headers={"Authorization": "Bearer " + tok} if tok else {})
    try:
        with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200]
    except Exception as e:  # noqa: BLE001
        return None, repr(e)[:200]


def main():
    print("service account token:", os.path.exists(f"{SA}/token"), " namespace:",
          open(f"{SA}/namespace").read().strip() if os.path.exists(f"{SA}/namespace") else "?")
    ns = open(f"{SA}/namespace").read().strip() if os.path.exists(f"{SA}/namespace") else "default"
    st, me = get(f"/api/v1/namespaces/{ns}/pods/{os.environ.get('HOSTNAME', '')}")
    print("own pod:", st, (me.get("spec", {}).get("nodeName"), [c.get("resources") for c in me.get("spec", {}).get("containers", [])]) if isinstance(me, dict) else me)
    st, nodes = get("/api/v1/nodes")
    print("list nodes:", st)
    if isinstance(nodes, dict):
        for n in nodes.get("items", []):
            al = {k: v for k, v in n["status"].get("allocatable", {}).items() if "nvidia" in k}
            if al:
                print(" ", n["metadata"]["name"], al)
    st, pods = get("/api/v1/pods?fieldSelector=status.phase=Running")
    print("list pods (all namespaces):", st)
    if isinstance(pods, dict):
        used = {}
        for p in pods.get("items", []):
            for c in p["spec"].get("containers", []):
                for k, v in (c.get("resources", {}).get("limits") or {}).items():
                    if "nvidia" in k:
                        key = (p["spec"].get("nodeName"), k)
                        used[key] = used.get(key, 0) + int(v)
        for (node, k), v in sorted(used.items()):
            print("  in use", node, k, v)
    st, pl = get(f"/api/v1/namespaces/{ns}/pods")
    print(f"list pods in {ns}:", st)
    if isinstance(pl, dict):
        used = {}
        for p in pl.get("items", []):
            if p.get("status", {}).get("phase") not in ("Running", "Pending"):
                continue
            for c in p["spec"].get("containers", []):
                for k, v in (c.get("resources", {}).get("limits") or {}).items():
                    if "nvidia" in k:
                        key = (p["status"]["phase"], k)
                        used[key] = used.get(key, 0) + int(v)
        for (ph, k), v in sorted(used.items()):
            print(f"  {ph:8s} {k} {v}")


if __name__ == "__main__":
    main()
