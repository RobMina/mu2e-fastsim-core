"""Inspect native CRY software provenance and the stock gen-table schema on a compute node."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import uproot

from .cry_native import CrySource


def inspect(reference, gdml):
    with uproot.open(reference) as f:
        report = {"trees": f.classnames(recursive=True)}
        gen = f["dump/gen"].arrays(library="pd")
        report["gen_columns"] = list(gen.columns)
        report["gen_head"] = gen.head(8).to_dict(orient="list")
        report["gen_minmax"] = {k: [float(gen[k].min()), float(gen[k].max())]
                                 for k in ("x", "y", "z", "t", "idx", "genid") if k in gen}
        report["reference_subruns"] = f["dump/subruns"].arrays(library="pd").to_dict(orient="list")
        report["reference_events"] = gen[["run", "subrun", "event"]].drop_duplicates().shape[0]
    root = ET.parse(gdml).getroot()
    world = root.find("setup/world").get("ref")
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    report["gdml_world"] = ET.tostring(vols[world], encoding="unicode")[:2000]
    solid = vols[world].find("solidref").get("ref")
    solids = {s.get("name"): s for s in root.find("solids")}
    report["gdml_world_solid"] = [ET.tostring(solids[solid], encoding="unicode")]
    for pv in vols[world].findall("physvol"):
        if pv.get("name", "").startswith(("worldDirtBottom", "HallAir")):
            v = vols[pv.find("volumeref").get("ref")]
            report[pv.get("name")] = {"position": ET.tostring(pv.find("position"), encoding="unicode"),
                                       "solid": ET.tostring(solids[v.find("solidref").get("ref")], encoding="unicode")}
    report["gdml_sha256"] = hashlib.sha256(Path(gdml).read_bytes()).hexdigest()
    start = time.perf_counter()
    with CrySource(seed=1234567) as source:
        report["init_s"] = time.perf_counter() - start
        start = time.perf_counter()
        batch = source.next_showers(10000)
    report["raw_generation_s"] = time.perf_counter() - start
    report["raw_showers"] = len(batch.cumulative_seconds)
    report["raw_particles"] = len(batch.particles)
    report["raw_live_seconds"] = batch.live_seconds
    report["image_version"] = Path("/opt/cry/VERSION").read_text()
    report["image_build_sha256"] = Path("/opt/cry/BUILD.sha256").read_text()
    report["shim_bytes"] = Path("/opt/cry/lib/libmu2ecry.so").stat().st_size
    return report


def _summarise(gen, live, config):
    from .cosmic_cry import MASS, highest_intersection
    momentum = gen[["px", "py", "pz"]].to_numpy()
    target, hit = highest_intersection(gen[["x", "y", "z"]].to_numpy(), momentum,
                                       config.target_min_mm, config.target_max_mm)
    if not np.all(hit):
        raise ValueError("reference primary misses configured target box")
    gen = gen.copy()
    for j, a in enumerate("xyz"):
        gen["target_" + a] = target[:, j]
    key = ["run", "subrun", "event"]
    counts = gen.groupby(key, sort=False).size().to_numpy()
    n = len(counts)
    mu = gen[gen.pdg.abs() == 13]
    mus = mu.groupby(key, sort=False).size().to_numpy()
    result = {"accepted_per_s": n / live, "live_per_event_us": live / n * 1e6,
              "multiprimary_fraction": float(np.mean(counts > 1)),
              "multimuon_fraction": float(np.sum(mus > 1) / n),
              "mean_multiplicity": len(gen) / n, "shard_max_multiplicity": int(counts.max())}
    pdg_keys = (11, -11, 13, -13, 22, 2112, 2212, 211, -211, 321, -321)
    hist = {}
    for pdg in pdg_keys:
        rows = gen[gen.pdg == pdg]
        label = str(pdg)
        result[f"rate_pdg_{label}"] = len(rows) / live
        if not len(rows):
            continue
        kinetic = rows.e.to_numpy() - MASS[abs(pdg)]
        p = np.sqrt(np.maximum(rows.e.to_numpy()**2 - MASS[abs(pdg)]**2, 0))
        arrays = {"ke": (kinetic, np.geomspace(50, 200000, 45)),
                  "coszenith": (-rows.py.to_numpy() / np.maximum(p, 1e-30), np.linspace(-1, 1, 41)),
                  "azimuth": (np.arctan2(rows.pz.to_numpy(), rows.px.to_numpy()), np.linspace(-np.pi, np.pi, 41)),
                  "x": (rows.x.to_numpy(), np.linspace(-155000, 150000, 62)),
                  "y": (rows.y.to_numpy(), np.linspace(-5000, 20000, 51)),
                  "z": (rows.z.to_numpy(), np.linspace(-130000, 180000, 63)),
                  "t": (rows.t.to_numpy(), np.linspace(0, 100005, 51)),
                  **{"target_"+a: (rows["target_"+a].to_numpy(),
                                   np.linspace(config.target_min_mm[j], config.target_max_mm[j], 41))
                     for j, a in enumerate("xyz")}}
        for name, (values, edges) in arrays.items():
            h = np.histogram(values, np.r_[-np.inf, edges, np.inf])[0]
            hist[f"{label}:{name}"] = {"edges": edges.tolist(), "counts_per_event": (h / n).tolist()}
    hist["multiplicity"] = {"edges": list(range(1, 10)),
                             "counts_per_event": (np.bincount(np.minimum(counts, 9), minlength=10)[1:] / n).tolist()}
    if np.any(counts > 1):
        extent = gen.groupby(key, sort=False).agg(xlo=("target_x", "min"), xhi=("target_x", "max"),
                                                 zlo=("target_z", "min"), zhi=("target_z", "max"),
                                                 tlo=("t", "min"), thi=("t", "max"), n=("idx", "size"))
        extent = extent[extent.n > 1]
        spread = np.hypot(extent.xhi - extent.xlo, extent.zhi - extent.zlo) / 1000
        result["multi_lateral_median_m"] = float(np.median(spread))
        result["multi_lateral_p90_m"] = float(np.quantile(spread, .9))
        result["multi_dt_max_ns"] = float((extent.thi - extent.tlo).max())
        multi = gen[gen.duplicated(key, keep=False)]
        pairs = multi.merge(multi, on=key, suffixes=("_a", "_b"))
        pairs = pairs[pairs.idx_a < pairs.idx_b]
        for plane in ("world", "target"):
            prefix = "target_" if plane == "target" else ""
            distance = np.hypot(pairs[prefix+"x_a"]-pairs[prefix+"x_b"],
                                 pairs[prefix+"z_a"]-pairs[prefix+"z_b"])/1000
            result[f"multi_pair_{plane}_xz_median_m"] = float(np.median(distance))
            result[f"multi_pair_{plane}_xz_p90_m"] = float(np.quantile(distance, .9))
            edges = np.linspace(0, 100 if plane == "world" else 25, 51)
            hist[f"multi_pair_{plane}_xz"] = {"edges": edges.tolist(),
                "counts_per_event": (np.histogram(distance, np.r_[-np.inf, edges, np.inf])[0]/n).tolist()}
    return {"events": n, "particles": len(gen), "live_seconds": live, "metrics": result, "histograms": hist}


def compare(references, gdml, events, seed, native_out=None, native_inputs=None, backend="native"):
    if backend not in ("native", "device"):
        raise ValueError("unknown CRY validation backend")
    from . import tables
    from .cosmic_cry import CosmicConfig, CosmicSource
    config = CosmicConfig.extracted(gdml)
    if backend == "device":
        from .cry_device import DeviceCosmicSource
        Source = DeviceCosmicSource
    else:
        Source = CosmicSource
    if native_inputs and len(native_inputs) != len(references):
        raise ValueError("native-input and reference lists must have the same number of shards")
    native, stock, receipts, elapsed = [], [], [], []
    for i, path in enumerate(references):
        g = tables.read(path, "gen")
        sr = tables.read(path, "subruns")
        stock.append(_summarise(g, float(sr.livetime.sum()), config))
        start = time.perf_counter()
        if native_inputs:
            ng = tables.read(native_inputs[i], "gen")
            ns = tables.read(native_inputs[i], "subruns")
            receipt = {"reused_native_input": native_inputs[i]}
        else:
            with Source(config, seed=seed+i) as source:
                ng, ns, receipt = source.generate(events, subrun=1000+i)
        elapsed.append(time.perf_counter()-start)
        receipt["input_wall_s" if native_inputs else "generation_wall_s_including_init"] = elapsed[-1]
        receipts.append(receipt)
        native.append(_summarise(ng, float(ns.livetime.sum()), config))
        if native_out:
            tables.write(str(Path(native_out) / f"gen.native.{i:04d}.root"), {"gen": ng, "subruns": ns}, as_ttree=True)
    keys = sorted(set.intersection(*[set(row["metrics"]) for row in stock + native]))
    metrics = {}
    for key in keys:
        a = np.array([row["metrics"][key] for row in native])
        b = np.array([row["metrics"][key] for row in stock])
        ea = float(a.std(ddof=1)/np.sqrt(len(a))) if len(a) > 1 else None
        eb = float(b.std(ddof=1)/np.sqrt(len(b))) if len(b) > 1 else None
        ratio = float(a.mean()/b.mean()) if b.mean() else None
        er = (float(np.hypot(ea/b.mean(), a.mean()*eb/b.mean()**2))
              if ea is not None and eb is not None and b.mean() else None)
        metrics[key] = {"native": float(a.mean()), "stock": float(b.mean()),
                        "native_se": ea, "stock_se": eb, "ratio": ratio, "ratio_se": er}
    hist = {}
    hkeys = sorted(set.intersection(*[set(row["histograms"]) for row in stock+native]))
    for key in hkeys:
        a = np.array([row["histograms"][key]["counts_per_event"] for row in native])
        b = np.array([row["histograms"][key]["counts_per_event"] for row in stock])
        se = np.sqrt(a.var(axis=0, ddof=1)/len(a) + b.var(axis=0, ddof=1)/len(b)) if len(a)>1 else None
        hist[key] = {"edges": stock[0]["histograms"][key]["edges"],
                     "native": a.mean(axis=0).tolist(), "stock": b.mean(axis=0).tolist(),
                     "difference_se": se.tolist() if se is not None else None}
    spectra = {}
    if len(native) > 1:
        from scipy.stats import t as student_t
        tested = []
        for key in hkeys:
            a = np.array([row["histograms"][key]["counts_per_event"] for row in native])
            b = np.array([row["histograms"][key]["counts_per_event"] for row in stock])
            va, vb = a.var(axis=0, ddof=1)/len(a), b.var(axis=0, ddof=1)/len(b)
            populated = ((a.mean(axis=0)*sum(x["events"] for x in native) >= 20)
                         & (b.mean(axis=0)*sum(x["events"] for x in stock) >= 20) & ((va+vb) > 0))
            diff = a.mean(axis=0)-b.mean(axis=0)
            stat = np.abs(diff[populated])/np.sqrt((va+vb)[populated])
            dof = ((va+vb)[populated]**2
                   / (va[populated]**2/(len(a)-1) + vb[populated]**2/(len(b)-1)))
            pvalue = 2*student_t.sf(stat, dof)
            tested.extend(pvalue.tolist())
            hist[key]["tested_bins"] = np.flatnonzero(populated).tolist()
            hist[key]["welch_pvalues"] = pvalue.tolist()
            spectra[key] = {"tested_bins": len(stat), "max_abs_t": float(stat.max()) if len(stat) else None,
                            "min_unadjusted_p": float(pvalue.min()) if len(stat) else None,
                            "per_event_L1_difference": float(np.abs(diff).sum())}
        family = len(tested)
        for key in spectra:
            pv = hist[key]["welch_pvalues"]
            spectra[key]["bonferroni_min_p"] = min(1.0, min(pv)*family) if pv else None
        spectra["family_summary"] = {"tested_bins": family, "minimum_bonferroni_p": min(1.0,min(tested)*family) if tested else None,
                                      "method": "Welch t across independent shards; bins require >=20 particles per arm; Bonferroni across all tested bins; descriptive, no predeclared admission band"}
    return {"backend": backend, "spectra_summary": spectra,
            "error_method": "independent native/stock shard means; sample-SE over shards, no pairing; K=1 errors indeterminate",
            "native_events": sum(x["events"] for x in native), "stock_events": sum(x["events"] for x in stock),
            "native_max_multiplicity": max(x["metrics"]["shard_max_multiplicity"] for x in native),
            "stock_max_multiplicity": max(x["metrics"]["shard_max_multiplicity"] for x in stock),
            "native_input_read_wall_s" if native_inputs else "native_generation_wall_s": elapsed,
            "metrics": metrics, "histograms": hist, "receipts": receipts,
            "reference_files": references, "native_raw": native, "stock_raw": stock}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, nargs="+")
    p.add_argument("--gdml", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--events", type=int)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--backend", choices=("native", "device"), default="native")
    p.add_argument("--native-out")
    p.add_argument("--native-input", nargs="+")
    a = p.parse_args()
    report = (compare(a.reference, a.gdml, a.events, a.seed, a.native_out, a.native_input, a.backend) if a.events
              else inspect(a.reference[0], a.gdml))
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("histograms", "native_raw", "stock_raw")}, indent=2))


if __name__ == "__main__":
    main()
