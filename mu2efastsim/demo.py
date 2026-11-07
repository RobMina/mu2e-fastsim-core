"""Demo: three short fast chains on the input release, each headline number next to its reference value.

    python3 -m mu2efastsim.demo [--quick] [--out DIR] [--only beam,ce,crv]

Needs the site variables of a preset (site/*.env in the umbrella): M2FS_INPUTS (the input release), M2FS_DATA
(outputs; default DIR is $M2FS_DATA/demo), and core, crv and beam on PYTHONPATH. Each chain is the existing entry point,
run as a subprocess on the CPU (JAX_PLATFORMS=cpu unless set):

  beam  python3 -m beamfastsim point     20,000 DS-entrance muons (MDC2025ac) -> target stop fraction
  ce    python3 -m beamfastsim.cechain   (JAX backend) 2,000 conversion electrons from that table's target stops -> fraction that
                                         reaches the tracker entry plane
  crv   python3 -m crvfastsim runchain   1,000 generated cosmic events (default chain) -> coincidence clusters per
                                         generated event

--quick uses a quarter of each sample. A line passes when |value - reference| <= TOLERANCE_SIGMA * sqrt(stat^2 +
ref^2), stat being the value's counting error. The field maps and the beam geometry come from CVMFS (or
$BEAMFS_OFFLINE_ROOT / $BEAMFS_MAPS, as `python3 -m beamfastsim.stage` prints them). Writes <out>/demo_summary.json.
Exit 0 every line passes, 1 a line is outside its tolerance, 2 inputs or settings missing (nothing run), 3 a chain
failed. A smoke test of the setup, not a validation: the references are listed in the umbrella's QUICKSTART.md,
section "The demo", and measured in its docs/validation.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CVMFS = "/cvmfs/mu2e.opensciencegrid.org"
TOLERANCE_SIGMA = 3.0
BEAM_TABLE = "beam_fastsim/g4ref_full/g4ref.npz"
CRV_GEN = "crv_fastsim/g4ref_v1_tables/gen.001400_00000011.root"
CRV_FILES = ("crv_fastsim/tier1b_strict_v5/library.root", "crv_fastsim/geometry/mu2e_extracted_v04.gdml",
             "crv_fastsim/geometry/crvcalib_sim_best_v1_5_run1400.csv",
             "crv_fastsim/geometry/crvphoton_sim_best_v1_5_run1400.csv", "crv_fastsim/lookup_v6_0/photonMap.root")


@dataclass(frozen=True)
class Reference:
    value: float
    error: float
    source: str


# reference values (umbrella QUICKSTART.md, "The demo"): sample, how measured, and the error that enters the tolerance
REFERENCES = {
    "beam": Reference(0.36038, 0.00148, "Geant4 DS stage on all 105,350 muons of the table, clean stock rerun "
                      "(docs/validation.md; the first reference job, which stopped at event 105,100, gave 0.35919); "
                      "the fast chain gives 0.35834"),
    "ce": Reference(0.72598, 0.00223, "fast chain (JAX backend) on 40,000 electrons from the same stops, seed 2: "
                    "29,039 reach the plane; the CE chain agrees with Geant4 CeEndpoint (docs/validation.md)"),
    "crv": Reference(0.05822, 0.00108, "stock chain (Geant4 + digi + reco) on all 50,000 events of the same shard, "
                     "2,911 clusters (crv_fastsim/v2_fastC_v1/ref.001400_00000011.root); the fast chain gives 0.0576"),
}
SAMPLES = {"beam": 20000, "ce": 2000, "crv": 1000}


class Missing(Exception):
    """An input or setting the demo needs is absent; nothing was run."""


def _offline_and_maps():
    off = os.environ.get("BEAMFS_OFFLINE_ROOT", f"{CVMFS}/Musings/Offline/v10_47_00")
    maps = os.environ.get("BEAMFS_MAPS", f"{CVMFS}/DataFiles/BFieldMaps/Mau13")
    return off, maps


def commands(inputs: str, out: Path, n: dict) -> dict:
    """Command line per chain (lists for subprocess), all from the existing entry points."""
    off, maps = _offline_and_maps()
    g = f"{inputs}/crv_fastsim/geometry"
    py = sys.executable
    return {
        "beam": [py, "-m", "beamfastsim", "point", "--g4", f"{inputs}/{BEAM_TABLE}", "--n", str(n["beam"]),
                 "--seed", "1", "--out", str(out / "beam")],
        "ce": [py, "-m", "beamfastsim.cechain", "--offline-root", off, "--maps", f"{maps}/DSMap.header",
               f"{maps}/TSdMap.header", "--stops", f"{inputs}/{BEAM_TABLE}", "--n", str(n["ce"]), "--seed", "1",
               "--backend", "jax", "--out", str(out / "ce.npz")],
        "crv": [py, "-m", "crvfastsim", "runchain", f"{inputs}/{CRV_GEN}",
                "--tier1b-dir", f"{inputs}/crv_fastsim/tier1b_strict_v5", "--gdml", f"{g}/mu2e_extracted_v04.gdml",
                "--calib", f"{g}/crvcalib_sim_best_v1_5_run1400.csv",
                "--deviation", f"{g}/crvphoton_sim_best_v1_5_run1400.csv", "--max-events", str(n["crv"]),
                "--seed", "11", "--outdir", str(out / "crv")],
    }


def needed_files(inputs: str, chains=tuple(SAMPLES)) -> list:
    """Release and CVMFS files the selected chains read."""
    off, maps = _offline_and_maps()
    files = []
    if {"beam", "ce"} & set(chains):
        files += [f"{inputs}/{BEAM_TABLE}", f"{maps}/DSMap.header", f"{maps}/TSdMap.header",
                  f"{off}/Offline/Mu2eG4/geom/geom_common.txt"]
    if "crv" in chains:
        files += [f"{inputs}/{CRV_GEN}"] + [f"{inputs}/{f}" for f in CRV_FILES]
    return files


def last_json_object(text: str, key: str) -> dict:
    """The last JSON object in `text` that starts a line and holds `key` (the entry point may log other lines first)."""
    dec, found, pos = json.JSONDecoder(), None, 0
    for line in text.splitlines(keepends=True):
        body = line.lstrip()
        if body.startswith("{"):
            try:
                obj, _ = dec.raw_decode(text, pos + len(line) - len(body))
            except ValueError:
                obj = None
            if isinstance(obj, dict) and key in obj:
                found = obj
        pos += len(line)
    if found is None:
        raise ValueError(f"no JSON object with {key!r} in the output")
    return found


def headline(chain: str, out: Path, stdout: str) -> tuple:
    """(value, counting error, sample size) of a finished chain from the files and output it writes."""
    if chain == "beam":
        m = json.loads((out / "beam" / "metrics.json").read_text())
        return float(m["stop_frac"]), float(m["stop_frac_err"]), int(m["muons"])
    if chain == "ce":
        s = last_json_object(stdout, "fates")
        n, k = int(s["n"]), int(s["fates"]["plane"])
        f = k / n
        return f, math.sqrt(f * (1 - f) / n), n
    if chain == "crv":
        s = json.loads((out / "crv" / "runchain_stats.json").read_text())
        n = int(s["generated_events"])
        c = sum(s["clusters_per_generated"].values()) * n
        return c / n, math.sqrt(c) / n, n
    raise ValueError(chain)


def run_chain(cmd: list, log: Path) -> tuple:
    """Run one entry point on the CPU (JAX_PLATFORMS=cpu unless set); (exit code, stdout, seconds)."""
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.write_text(f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.returncode, p.stdout, time.time() - t0


def main(argv=None, runner: Callable = run_chain) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.demo", description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="a quarter of each sample")
    ap.add_argument("--out", help="output directory (default $M2FS_DATA/demo)")
    ap.add_argument("--only", default="beam,ce,crv", help="comma list of chains (beam, ce, crv)")
    a = ap.parse_args(argv)
    chains = [c for c in a.only.split(",") if c]
    if set(chains) - set(SAMPLES):
        ap.error(f"--only takes {', '.join(SAMPLES)}")
    n = {c: max(1, v // 4) if a.quick else v for c, v in SAMPLES.items()}
    try:
        inputs = os.environ.get("M2FS_INPUTS")
        if not inputs:
            raise Missing("M2FS_INPUTS is not set: source a site preset (site/*.env) after your access request")
        missing = [f for f in needed_files(inputs, chains) if not os.path.isfile(f)]
        if missing:
            raise Missing("missing or unreadable: " + ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
                          + " (input release, or CVMFS; see QUICKSTART.md)")
        out = Path(a.out or (os.environ.get("M2FS_DATA") or "") + "/demo")
        if not a.out and not os.environ.get("M2FS_DATA"):
            raise Missing("M2FS_DATA is not set and no --out given")
    except Missing as e:
        print(f"demo skipped: {e}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    cmds = commands(inputs, out, n)
    rows, worst = [], 0
    for c in chains:
        rc, stdout, sec = runner(cmds[c], out / f"{c}.log")
        ref = REFERENCES[c]
        if rc != 0:
            rows.append(dict(chain=c, exit_code=rc, seconds=round(sec, 1), log=str(out / f"{c}.log")))
            print(f"{c:5s} FAILED (exit {rc}) after {sec:.0f} s; log {out / f'{c}.log'}")
            worst = 3
            continue
        v, e, size = headline(c, out, stdout)
        tol = TOLERANCE_SIGMA * math.hypot(e, ref.error)
        ok = abs(v - ref.value) <= tol
        worst = max(worst, 0 if ok else 1)
        rows.append(dict(chain=c, value=v, stat_error=e, sample=size, reference=ref.value, reference_error=ref.error,
                         tolerance=tol, passed=ok, seconds=round(sec, 1), reference_source=ref.source))
        print(f"{c:5s} {LABELS[c]:44s} {v:.4f} +- {e:.4f}  reference {ref.value:.4f}  tolerance +- {tol:.4f}  "
              f"{'ok' if ok else 'OUTSIDE'}  ({size:,} {UNITS[c]}, {sec:.0f} s)")
    summary = {"quick": a.quick, "tolerance_sigma": TOLERANCE_SIGMA, "inputs": inputs,
               "release_manifest": os.environ.get("MANIFEST"), "chains": rows,
               "passed": worst == 0}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"summary {out / 'demo_summary.json'}")
    return worst


LABELS = {"beam": "target stop fraction", "ce": "CE reaching the tracker entry, per electron",
          "crv": "CRV coincidence clusters per generated event"}
UNITS = {"beam": "muons", "ce": "electrons", "crv": "generated events"}


if __name__ == "__main__":
    raise SystemExit(main())
