"""Aurora: reproduce and isolate the V-R1 noise-on abort ("urEventWait must not be called for an internal event") on
synthetic SiPM charges, with the same batched JAX waveform synthesis as reco-vr1 --jax (cli._synth_jax_batched).

Each case runs in its own process (the runtime aborts the process). Cases vary noise, repetition and Level Zero
unified-runtime settings. No Mu2e-internal input.
usage: python vr1noise_probe.py [--events N] [--out stats.json]             (driver)
       python vr1noise_probe.py --child NOISE --events N --repeat R         (one case)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

CASES = [  # name, noise, repeat, extra env
    ("noise0", 0, 1, {}),
    ("noise1", 1, 1, {}),
    ("noise1_x3", 1, 3, {}),
    ("noise0_x3", 0, 3, {}),
    ("noise1_x2", 1, 2, {}),
    ("noise1_x3_sync_dispatch", 1, 3, {"M2FS_SYNC_EACH_BATCH": "1"}),
    ("noise1_x3_no_immediate_cl", 1, 3, {"UR_L0_USE_IMMEDIATE_COMMANDLISTS": "0"}),
    ("noise1_x3_no_event_cache", 1, 3, {"UR_L0_DISABLE_EVENTS_CACHING": "1"}),
    ("noise1_x3_no_reuse_events", 1, 3, {"UR_L0_REUSE_DISCARDED_EVENTS": "0"}),
    ("noise1_x3_no_counter_events", 1, 3, {"UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS": "0"}),
    ("noise1_x3_no_inorder_lists", 1, 3, {"UR_L0_USE_DRIVER_INORDER_LISTS": "0"}),
    ("noise1_x3_serialize", 1, 3, {"UR_L0_SERIALIZE": "2"}),
    ("noise1_x3_host_key", 1, 3, {"M2FS_HOST_KEY": "1"}),
    ("noise1_x3_single_key", 1, 3, {"M2FS_SINGLE_KEY": "1"}),
]


def child(noise, n_events, repeat):
    import numpy as np
    import jax
    from crvfastsim.reco import cli as RC, repro, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    row, t, q, tdc0 = repro._charges(n_events, 20261021)
    Q = {"row": row, "time": t, "charge": q}
    sync = os.environ.get("M2FS_SYNC_EACH_BATCH") == "1"
    if sync:  # block on every batch's outputs before the next dispatch
        orig = WJ.synth_dense

        def synced(*a, **k):
            return jax.block_until_ready(orig(*a, **k))
        WJ.synth_dense = synced
    if os.environ.get("M2FS_HOST_KEY") == "1" or os.environ.get("M2FS_SINGLE_KEY") == "1":
        # build each batch key on the host (numpy threefry-free path) instead of fold_in on the device
        orig2 = WJ.synth_dense
        single = os.environ.get("M2FS_SINGLE_KEY") == "1"

        def hostkey(T, W, n, ts, keep, tmpl, key, *rest):
            k = np.asarray(jax.device_get(key)) if not single else np.array([0, 7], np.uint32)
            return orig2(T, W, n, ts, keep, tmpl, jax.device_put(k), *rest)
        WJ.synth_dense = hostkey
    hs = []
    t0 = time.perf_counter()
    for r in range(repeat):
        out = RC._synth_jax_batched(WJ, Q, tdc0, cfg, 2000, noise=cfg.noise if noise else 0.0, seed=7 + r)
        h = hashlib.sha256(out["adc"].tobytes()).hexdigest()[:16]
        hs.append(h)
    dt = time.perf_counter() - t0
    print("RESULT " + json.dumps({"device": str(jax.devices()[0]), "rows": int(row[-1]) + 1, "digis": int(len(out["adc"])),
                                  "hashes": hs, "wall_s": dt}), flush=True)


def run_case(name, noise, repeat, env_extra, n_events, timeout):
    env = dict(os.environ, **env_extra)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__, "--child", str(noise), "--events", str(n_events), "--repeat",
                            str(repeat)], env=env, capture_output=True, text=True, timeout=timeout)
        rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    res = next((json.loads(l[7:]) for l in text.splitlines() if l.startswith("RESULT ")), None)
    err = [l.strip()[:240] for l in text.splitlines() if any(k in l for k in ("die:", "terminate", "Fatal", "Error"))][:3]
    rec = {"case": name, "env": env_extra, "rc": rc, "wall_s": round(time.perf_counter() - t0, 1), "result": res,
           "error": None if res else err}
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", type=int)
    ap.add_argument("--events", type=int, default=25000)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--cases", default=None, help="comma-separated case names (default all)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child is not None:
        return child(a.child, a.events, a.repeat)
    sel = set(a.cases.split(",")) if a.cases else None
    recs = [run_case(n, z, r, e, a.events, a.timeout) for n, z, r, e in CASES if not sel or n in sel]
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(recs, fh, indent=1)


if __name__ == "__main__":
    main()
