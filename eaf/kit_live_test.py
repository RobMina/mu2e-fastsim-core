#!/usr/bin/env python3
"""Live test of fastsimkit on the EAF endpoint from a GPVM: check, submit, poll, results, then stop the server.
    EAF_TOKEN_FILE=... python3 core/eaf/kit_live_test.py [task] [key=value ...]   (default: v4_response events=2000 seed=1)
Slices are tried as EAF_SLICES (comma list, default 1g.10gb: the cheapest, usually free). State under
<crv data>/eafkit_state, job directories under <crv data>/eafkit, where <crv data> is mu2efastsim.site.crv_data()
($FASTSIMKIT_FNAL_DATA, $CRVFS_DATA or <M2FS_INPUTS>/crv_fastsim; see site.env.example).
"""
import json
import os
import sys
import time

from mu2efastsim import site
from mu2efastsim.endpoints.eaf import EAFEndpoint
from mu2efastsim.endpoints.kit import FastsimKit


def main():
    D = site.crv_data()
    task = sys.argv[1] if len(sys.argv) > 1 else "v4_response"
    params = {"task": task, **(dict(a.split("=", 1) for a in sys.argv[2:]) or {"events": "2000", "seed": "1"})}
    slices = os.environ.get("EAF_SLICES", "1g.10gb").split(",")
    ep = EAFEndpoint(workdir=f"{D}/eafkit", state_dir=f"{D}/eafkit_state/jobs", slices=slices)
    kit = FastsimKit(endpoint=ep, state_dir=f"{D}/eafkit_state")
    name = f"live_{task}_{time.strftime('%Y%m%d_%H%M%S')}"
    print("check", kit.check(name, params, [], []))
    t0 = time.time()
    try:
        print("submit", kit.submit(name, params, [], []), f"({time.time() - t0:.0f} s, slice {ep.state.get(name).resources.get('slice')})")
        while True:
            st = kit.status(name)
            if st["state"] != "working":
                break
            time.sleep(15)
        print("status", st, f"after {time.time() - t0:.0f} s")
        if st["state"] == "completed":
            r = kit.results(name)
            print("metrics", json.dumps(r["metrics"], indent=1))
            print("metadata", r["metadata"])
        print("job dir", ep._dir(name))
    finally:
        print("stop_idle:", ep.stop_idle())


if __name__ == "__main__":
    main()
