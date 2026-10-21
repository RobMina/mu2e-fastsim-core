#!/usr/bin/env python3
"""Throughput of a set of independent UVA chain jobs (chain_array_uva.sh): accepted events per GPU-hour, per SU and
per hour of campaign wall time (first start to last end), from each job's runchain --emit shard and its Slurm and
MAM accounting. Stats only: counts and timings, no event data.

    throughput_uva.py jobs.tsv shard_<jobid>.json [...] --out throughput.json
jobs.tsv columns: jobid, start (epoch s), end (epoch s), elapsed_s, gpus, gpu_type, node, su, input_wait_s
(the job id is taken from a /<jobid>/ path component of each shard, or from its file name)
"""
import argparse
import csv
import json
import os
import re


def load_jobs(path):
    with open(path) as fh:
        return {r["jobid"]: r for r in csv.DictReader(fh, delimiter="\t")}


def summarise(jobs, shards):
    rows = []
    for path in shards:
        m = re.search(r"(?:^|/)(\d{6,})/", path) or re.search(r"(\d{6,})", os.path.basename(path))
        jid = m.group(1)
        s = json.load(open(path))
        j = jobs[jid]
        T = s["timers"]
        acc = int(T.get("events_with_steps", 0))
        el, gpus, su = float(j["elapsed_s"]), int(j["gpus"]), float(j["su"])
        work = sum(v for k, v in T.items() if k.endswith("_s") and k not in ("setup_s", "wall_s"))
        rows.append({"jobid": jid, "node": j["node"], "gpu_type": j["gpu_type"], "gpus": gpus,
                     "generated": int(s["generated_events"]), "accepted": acc, "elapsed_s": el,
                     "process_wall_s": float(s["wall_s"]), "setup_s": float(T.get("setup_s", 0.0)), "su": su,
                     "accepted_per_gpu_hour": 3600.0 * acc / (el * gpus), "accepted_per_su": acc / su if su else None,
                     "accepted_per_gpu_hour_steady": 3600.0 * acc / max(float(s["wall_s"]) - float(T.get("setup_s", 0.0)), 1e-9) / gpus,
                     "input_wait_s": float(j.get("input_wait_s") or 0.0),
                     "accepted_per_su_without_input_wait": acc / (su * (el - float(j.get("input_wait_s") or 0.0)) / el)
                     if su and el > float(j.get("input_wait_s") or 0.0) else None,
                     "start": float(j["start"]), "end": float(j["end"])})
    acc = sum(r["accepted"] for r in rows)
    gpu_h = sum(r["elapsed_s"] * r["gpus"] for r in rows) / 3600.0
    su = sum(r["su"] for r in rows)
    span = (max(r["end"] for r in rows) - min(r["start"] for r in rows)) / 3600.0 if rows else 0.0
    out = {"jobs": len(rows), "generated": sum(r["generated"] for r in rows), "accepted": acc,
           "gpu_hours": gpu_h, "su": su, "makespan_h": span,
           "accepted_per_gpu_hour": acc / gpu_h if gpu_h else None,
           "accepted_per_su": acc / su if su else None,
           "su_per_million_accepted": 1e6 * su / acc if acc else None,
           "accepted_per_campaign_hour": acc / span if span else None,
           "by_gpu_type": {}, "per_job": rows}
    for t in sorted({r["gpu_type"] for r in rows}):
        rr = [r for r in rows if r["gpu_type"] == t]
        a, g, s = sum(r["accepted"] for r in rr), sum(r["elapsed_s"] * r["gpus"] for r in rr) / 3600.0, sum(r["su"] for r in rr)
        out["by_gpu_type"][t] = {"jobs": len(rr), "accepted_per_gpu_hour": a / g if g else None,
                                 "accepted_per_su": a / s if s else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = summarise(load_jobs(a.jobs), a.shards)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_job"}, indent=1))
    for r in res["per_job"]:
        print(f"{r['jobid']} {r['gpu_type']:10s} acc {r['accepted']:6d} elapsed {r['elapsed_s']:6.0f} s su {r['su']:5.1f} "
              f"acc/GPU-h {r['accepted_per_gpu_hour']:9.0f} acc/SU {r['accepted_per_su'] or 0:8.0f}")


if __name__ == "__main__":
    main()
