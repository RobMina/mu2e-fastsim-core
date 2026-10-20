"""Aurora: which coinc_jax tiers compile and run on Intel PVC under which XLA_FLAGS.

Each case runs in its own process because the oneAPI compiler aborts the process (SPIR-V long-vector error).
usage: python finddense_probe.py [--events N] [--out stats.json]      (driver)
       python finddense_probe.py --child TIERS_SPEC --events N        (one case; TIERS_SPEC = "all" or a tier index)
Synthetic pulses only (reco.synth.coinc_events).
"""
import argparse
import json
import os
import subprocess
import sys
import time

FLAGS = {
    "base": "",
    "unroll1": "--xla_gpu_max_kernel_unroll_factor=1",
    "no_mlir_emitters": "--xla_gpu_enable_mlir_emitters=false",
    "mlir_emitter_level0": "--xla_gpu_mlir_emitter_level=0",
    "no_priority_fusion": "--xla_gpu_enable_priority_fusion=false",
}


def child(spec, n_events):
    import numpy as np
    import jax
    from crvfastsim.reco import cli as RC, coinc_jax as CJ, synth
    G = RC._coinc_geom()
    P = synth.coinc_events(n_events, G, seed=20261020)
    cols = [P[k] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")]
    status = np.zeros(4 * len(G.sector), np.int64)
    tiers = CJ.TIERS if spec == "all" else (CJ.TIERS[int(spec)],)
    t0 = time.perf_counter()
    CJ.find(*cols, G, status, tiers=tiers)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = CJ.find(*cols, G, status, tiers=tiers)
    dt = time.perf_counter() - t0
    ne = len(np.unique(P["ev"]))
    print("RESULT " + json.dumps({"device": str(jax.devices()[0]), "events": int(ne), "first_call_s": first, "pass_s": dt,
                                  "events_per_s": ne / dt, "clusters": int(len(out["cl_event"])),
                                  "fallback_events": int(len(out["fallback_events"]))}), flush=True)


def run_case(name, flags, spec, n_events, timeout):
    env = dict(os.environ, XLA_FLAGS=flags)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__, "--child", spec, "--events", str(n_events)], env=env,
                           capture_output=True, text=True, timeout=timeout)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    res = next((json.loads(l[7:]) for l in text.splitlines() if l.startswith("RESULT ")), None)
    err = next((l.strip()[:300] for l in text.splitlines()
                if any(k in l for k in ("Fatal Error", "Unknown flag", "Error:", "error:"))), None)
    rec = {"case": name, "tiers": spec, "xla_flags": flags, "rc": rc, "wall_s": round(time.perf_counter() - t0, 1),
           "result": res, "error": None if res else err}
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child")
    ap.add_argument("--events", type=int, default=20000)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child is not None:
        return child(a.child, a.events)
    from crvfastsim.reco import coinc_jax as CJ
    recs = [run_case(f"tier{i}_H{t[0]}_K{t[1]}", "", str(i), a.events, a.timeout) for i, t in enumerate(CJ.TIERS)]
    recs += [run_case(f"all_{k}", v, "all", a.events, a.timeout) for k, v in FLAGS.items()]
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(recs, fh, indent=1)


if __name__ == "__main__":
    main()
