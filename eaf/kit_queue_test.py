#!/usr/bin/env python3
"""Live test of the EAF queue through fastsimkit: two points submitted back to back, run one after the other on one
slice, results of both, then the server stops by itself after the linger time (no stop_idle call here).
    EAF_TOKEN_FILE=... python3 core/eaf/kit_queue_test.py
"""
import json
import os
import time

from mu2efastsim.endpoints.eaf import EAFEndpoint
from mu2efastsim.endpoints.kit import FastsimKit

D = os.environ.get("CRVFS_DATA", "/exp/mu2e/data/users/ram2aq/crv_fastsim")


def main():
    ep = EAFEndpoint(workdir=f"{D}/eafkit", state_dir=f"{D}/eafkit_state/jobs",
                     slices=os.environ.get("EAF_SLICES", "1g.10gb").split(","), linger_s=60)
    kit = FastsimKit(endpoint=ep, state_dir=f"{D}/eafkit_state")
    tag = time.strftime("%Y%m%d_%H%M%S")
    pts = {f"q_chain_{tag}": {"task": "chain", "shards": 2, "max_events": 50000, "seed": 1},
           f"q_v4_{tag}": {"task": "v4_response", "events": 2000, "seed": 2}}
    t0 = time.time()
    for name, p in pts.items():
        print("submit", name, kit.submit(name, p, [], []), f"{time.time() - t0:.0f} s")
    print("status right after:", {n: kit.status(n)["message"] for n in pts})
    pending, order = set(pts), []
    while pending:
        for n in sorted(pending):
            st = kit.status(n)
            if st["state"] != "working":
                pending.discard(n)
                order.append(n)
                print(f"{time.time() - t0:5.0f} s {n}: {st['state']} {st['message']}")
                if st["state"] == "completed":
                    print(json.dumps(kit.results(n)["metrics"], indent=1))
        time.sleep(20)
    launched = {n: ep.state.get(n).resources.get("launched_at", 0) - t0 for n in pts}
    print("launched at (s):", {n: round(v) for n, v in launched.items()}, "finished in order", order)
    for i in range(8):                       # linger 60 s, then the next status call or after_results stops it
        s = ep.hub.server()
        print(f"{time.time() - t0:5.0f} s server:", "running" if s and s.get("ready") else "stopped")
        if not s:
            break
        time.sleep(20)
        kit.status(order[-1])
        ep.after_results(order[-1])


if __name__ == "__main__":
    main()
