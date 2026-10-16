"""Flat-table I/O in the art dumper layout: trees in the TFileService directory ``dump/``, keyed by (run, subrun, event).

Detector packages define their own step trees (CrvStep, StrawGasStep) on top of the shared ``particles`` (SimParticle),
``gen`` (generated primary of every event) and ``subruns`` (CosmicLivetime) trees.
"""
import glob
import os

import pandas as pd
import uproot

KEY = ["run", "subrun", "event"]
DIR = "dump"
PRIMARY_MUON_PDG = 13


def expand(inputs):
    out = []
    for p in inputs:
        out += sorted(glob.glob(p)) or [p]
    return out


def read(path, tree, columns=None):
    with uproot.open(path) as f:
        return f[f"{DIR}/{tree}"].arrays(columns, library="pd")


def read_many(paths, tree, columns=None):
    return pd.concat([read(p, tree, columns) for p in expand(paths)], ignore_index=True)


def livetime(paths):
    """Total CosmicLivetime [s] over the subruns of the given files."""
    s = read_many(paths, "subruns")
    return float(s.drop_duplicates(["run", "subrun"]).livetime.sum())


def primary_muons(gen):
    """The generated primary muon of each event: one row per event (events with no muon are dropped)."""
    mu = gen[gen.pdg.abs() == PRIMARY_MUON_PDG]
    return mu.drop_duplicates(KEY).reset_index(drop=True)


def write(path, trees):
    """Write {tree name: DataFrame} into directory dump/ of a new ROOT file (empty or None frames are skipped)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with uproot.recreate(path) as f:
        for name, df in trees.items():
            if df is not None and len(df):
                f[f"{DIR}/{name}"] = {c: df[c].to_numpy() for c in df.columns}
    print(f"wrote {path}")
