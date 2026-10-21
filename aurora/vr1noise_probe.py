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

STRESS = [  # name, noise, extra env; each run R times with repeat 5
    ("noise1_batch50k", 1, {"M2FS_BATCH": "50000"}),
    ("noise1_batch100k", 1, {"M2FS_BATCH": "100000"}),
    ("noise1_batch25k", 1, {"M2FS_BATCH": "25000"}),
    ("noise1_batch2k", 1, {"M2FS_BATCH": "2000"}),
]
CASES = [(f"{n}_r{r}", z, 5, e) for n, z, e in STRESS for r in range(int(os.environ.get("M2FS_REPS", 3)))]


def child(noise, n_events, repeat):
    import numpy as np
    import jax
    from crvfastsim.reco import cli as RC, repro, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    row, t, q, tdc0 = repro._charges(n_events, 20261021)
    Q = {"row": row, "time": t, "charge": q}
    if os.environ.get("M2FS_FORCE_FUSED") == "1" and hasattr(WJ, "SPLIT_NOISE"):
        WJ.SPLIT_NOISE = False
    print("split_noise", getattr(WJ, "_split_noise", lambda: None)(), flush=True)
    if os.environ.get("M2FS_HOST_EPS") == "1":   # noise drawn with numpy on the host (diagnostic only: other bits)
        WJ.SPLIT_NOISE = True
        hr = np.random.default_rng(1)
        WJ._normal = lambda key, shape: jax.device_put(hr.standard_normal(shape))
    if os.environ.get("M2FS_SYNC_ALL") == "1":   # every jax op blocks (no async dispatch at all)
        jax.config.update("jax_disable_jit", False)
        os.environ["M2FS_SYNC_EACH_BATCH"] = "1"
        _orig_pk = WJ.pack

        def pk_sync(*a, **k):
            r = _orig_pk(*a, **k)
            jax.effects_barrier()
            return r
        WJ.pack = pk_sync
    if os.environ.get("M2FS_GC_EACH_BATCH") == "1" or os.environ.get("M2FS_KEEP_REFS") == "1":
        import gc
        keep = []
        orig3 = WJ.synth_dense
        refs = os.environ.get("M2FS_KEEP_REFS") == "1"

        def wrap(*a, **k):
            r = orig3(*a, **k)
            if refs:          # device outputs stay alive (no buffer/event release while later batches run)
                keep.append(r)
            else:
                jax.block_until_ready(r)
                gc.collect()
            return r
        WJ.synth_dense = wrap
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
        out = RC._synth_jax_batched(WJ, Q, tdc0, cfg, int(os.environ.get("M2FS_BATCH", 2000)),
                                    noise=cfg.noise if noise else 0.0, seed=7 + r)
        h = hashlib.sha256(out["adc"].tobytes()).hexdigest()[:16]
        hs.append(h)
    dt = time.perf_counter() - t0
    print("RESULT " + json.dumps({"device": str(jax.devices()[0]), "rows": int(row[-1]) + 1, "digis": int(len(out["adc"])),
                                  "hashes": hs, "wall_s": dt}), flush=True)


def run_case(name, noise, repeat, env_extra, n_events, timeout):
    env = dict(os.environ, **env_extra)
    if env_extra.get("M2FS_CPU_THREADS"):   # XLA host threads and the PJRT client to one core
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
        env["OMP_NUM_THREADS"] = "1"
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
