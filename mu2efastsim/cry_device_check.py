"""GPU-only CRY generation timing and conditional/raw checks against the C++ reference."""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np

from . import cry_device as D, cry_tables as T
from .cosmic_cry import CosmicConfig
from .cry_native import CrySource


def _raw_summary(r):
    mask = np.asarray(r["mask"])
    return {"showers": len(r["clocks"]), "particles": int(r["particles"]),
            "mean_multiplicity": float(np.mean(r["multiplicity"])),
            "multiplicity_p99": float(np.quantile(r["multiplicity"], .99)),
            "max_multiplicity": int(np.max(r["multiplicity"])),
            "dt_s": float(r["clocks"][-1]),
            "rates": {str(p): float(np.sum(np.asarray(r["pdg"])[mask] == p)/float(r["clocks"][-1]))
                       for p in (*T.PDG, -13, -11, -211, -321)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gdml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--events", type=int, default=50000)
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU backend required")
    config = CosmicConfig.extracted(a.gdml)
    results = {"devices": [str(d) for d in jax.devices()], "raw": [], "accepted_passes": []}
    tables, meta = D.device_tables()
    results["tables"] = meta
    key = jax.random.key(7981)
    before = 0.0
    for i in range(a.repeat):
        start = time.perf_counter()
        r = D.raw(tables, jax.random.fold_in(key, i), 0., showers=100000, capacity=400000)
        jax.block_until_ready(r)
        elapsed = time.perf_counter()-start
        if r["overflow"] or r["invalid_pdf"]:
            raise RuntimeError("invalid/overflow raw device batch")
        summary = _raw_summary(jax.device_get(r))
        summary["wall_s"] = elapsed
        results["raw"].append(summary)
    start = time.perf_counter()
    with CrySource(seed=1445) as source:
        init = time.perf_counter()-start
        start = time.perf_counter()
        raw = source.next_showers(100000)
        elapsed = time.perf_counter()-start
    ids = raw.particles["pdg"]
    counts = np.bincount(raw.particles["shower"].astype(np.int64), minlength=100000)
    results["raw_cpp"] = {"init_s": init, "wall_s": elapsed, "particles": len(ids),
                           "mean_multiplicity": float(np.mean(counts)),
                           "max_multiplicity": int(np.max(counts)), "dt_s": raw.live_seconds,
                           "rates": {str(p): float(np.sum(ids == p)/raw.live_seconds)
                                     for p in (*T.PDG, -13, -11, -211, -321)}}
    with D.DeviceCosmicSource(config, seed=64201) as source:
        for i in range(a.repeat):
            start = time.perf_counter()
            before_kernel, before_copy = source.kernel_wall_s, source.copy_wall_s
            gen, subruns, receipt = source.generate(a.events, subrun=i+1)
            elapsed = time.perf_counter()-start
            results["accepted_passes"].append({"wall_s": elapsed, "kernel_s": source.kernel_wall_s-before_kernel,
                                                "copy_s": source.copy_wall_s-before_copy,
                                                "events": int(gen.event.nunique()), "particles": len(gen),
                                                "receipt": receipt})
    results["timing_scope"] = "GPU generation/projection/selection; host parsing setup, selected-buffer copies and DataFrame packing identified separately; cold pass includes compilation"
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
