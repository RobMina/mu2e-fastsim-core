"""Load pinned CRY 1.7 distributions; host setup only, no shower draws.

Model reference: CRYPrimary/CRYGenerator/CRYPdf, LLNL UCRL-CODE-227323.
See docs/third-party/cry.txt for the retained upstream and DOE notices.
"""
from dataclasses import dataclass
from datetime import date
import hashlib
import os
from pathlib import Path
import re

import numpy as np


from .cry_constants import SEA_LEVEL_SHA256
SPECIES = ("neutron", "gamma", "muon", "electron", "proton", "pion", "kaon")
PDG = (2112, 22, 13, 11, 2212, 211, 321)


@dataclass(frozen=True)
class Pdf:
    binning: str
    low: float
    high: float
    kind: str
    weights: np.ndarray

    def cdf(self):
        total = self.weights.sum(axis=-1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("unreachable zero-weight PDF row must be handled explicitly")
        cdf = np.cumsum(self.weights / total, axis=-1)
        cdf[:, -1] = 1.0
        return cdf

    def mean(self):
        n = self.weights.shape[1]
        if self.kind == "dis":
            centers = self.low + np.arange(n) * (self.high-self.low) / max(1, n-1)
        else:
            lo, hi = (np.log10([self.low, self.high]) if self.kind == "log" else (self.low, self.high))
            centers = lo + (np.arange(n)+.5) * (hi-lo) / n
            if self.kind == "log":
                centers = 10**centers
        return (self.weights * centers).sum(axis=-1) / np.maximum(self.weights.sum(axis=-1), 1e-300)


def parse(text):
    """Strict parser for the published semicolon/braced format, without executing file contents."""
    joined = " ".join(line.split("%", 1)[0] for line in text.splitlines())
    result = {}
    for statement in joined.split(";"):
        if not statement.strip():
            continue
        lhs, rhs = statement.strip().split("=", 1)
        kind, name = lhs.split(None, 1)
        values = [np.fromstring(row, sep=" ") for row in re.findall(r"\{([^{}]*)\}", rhs)]
        if not values or len({len(row) for row in values}) != 1 or not len(values[0]):
            raise ValueError("malformed CRY table array")
        array = np.stack(values)
        if not np.all(np.isfinite(array)):
            raise ValueError("nonfinite CRY table value")
        if kind == "pdf":
            match = re.fullmatch(r"(\w+)::(\w+)\[([^,]+),([^,]+),(dis|lin|log)\]", name.strip())
            if not match or np.any(array < 0):
                raise ValueError("malformed CRY PDF")
            key, binning, low, high, mode = match.groups()
            low, high = float(low), float(high)
            if not low < high or mode == "log" and low <= 0:
                raise ValueError("invalid CRY PDF range")
            value = Pdf(binning, low, high, mode, array)
        else:
            key = name.strip()
            if kind == "function":
                key = key.split("::")[0]
            if kind not in ("function", "parameter", "paramInt", "binning") or len(values) != 1:
                raise ValueError("unsupported CRY table declaration")
            value = array[0]
        if key in result:
            raise ValueError("duplicate CRY table declaration")
        result[key] = value
    return result


def _spectrum(energy, params):
    a, b, c, d = params
    return a * (energy+b)**c * energy**d


def load(path=None, *, latitude=41.8, when="2021-06-21"):
    """Verified sea-level 300 m/all-species setup matching the extracted CPU reference."""
    path = Path(path or Path(os.environ.get("CRYDATAPATH", "/opt/cry/data")) / "cosmics_0.data")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SEA_LEVEL_SHA256:
        raise ValueError("unverified CRY sea-level table signature")
    if isinstance(latitude, bool) or not np.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("invalid CRY latitude")
    day = date.fromisoformat(when)
    decimal_year = day.year + (day-date(day.year, 1, 1)).days / (date(day.year+1, 1, 1)-date(day.year, 1, 1)).days
    t = parse(data.decode("ascii"))
    primary, secondary = t["primaryBins"], t["secondaryBins"]
    cycle = abs(np.sin(np.pi*(decimal_year-t["solarMinDate"][0])/t["solarCycleLength"][0]))
    cutoff = t["bfieldCorr"][0] * np.cos(np.deg2rad(latitude))**t["bfieldCorr"][1]
    minimum, maximum = max(primary[0], cutoff), primary[-1]
    if not minimum < maximum:
        raise ValueError("empty CRY primary spectrum")
    flux = lambda e: (1-cycle)*_spectrum(e, t["primarySpectrumSolarMin"]) + cycle*_spectrum(e, t["primarySpectrumSolarMax"])
    n_pdf, fractions = t["nParticles300"], t["particleFractions"]
    mean = n_pdf.mean()
    weight = np.divide(fractions.weights.sum(axis=1), mean, out=np.zeros_like(mean), where=mean>0)
    points = primary[:-1, None] + np.arange(1000)[None, :] * .001 * np.diff(primary)[:, None]
    partial = (flux(points) * (points >= minimum)).sum(axis=1) * .001 * np.diff(primary)
    lifetime = 1 / (np.dot(partial, weight) * np.pi * 300**2)
    edges = np.logspace(np.log10(minimum), np.log10(maximum), 10001)
    centers = 10**((np.log10(edges[:-1])+np.log10(edges[1:]))/2)
    primary_weights = flux(centers) * weight[np.searchsorted(primary, centers, side="right")-1] * np.diff(edges)
    p_pdf = Pdf("", minimum, maximum, "log", primary_weights[None, :])
    for name in ("latDist",):
        pdf = t[name]
        if pdf.kind != "lin" or not np.all(pdf.weights == 1) or pdf.weights.shape != (len(primary)-1, 1):
            raise ValueError("device setup expects the verified uniform lateral distribution")
    if not np.all(np.diff(primary)>0) or not np.all(np.diff(secondary)>0):
        raise ValueError("unordered CRY energy bins")
    needed = ["KEDist", "TimeDist", "CosThetaDist", "ChargeDist"]
    zero_rows = {}
    for suffix in needed:
        for species in SPECIES:
            pdf = t[species+suffix]
            zero_rows[species+suffix] = np.flatnonzero(pdf.weights.sum(axis=1) == 0).tolist()
    return {"tables": t, "primary_pdf": p_pdf, "lifetime_s": float(lifetime), "primary_bin_weights": weight,
            "table_sha256": SEA_LEVEL_SHA256, "latitude": latitude, "date": when,
            "zero_rows": zero_rows, "subbox_m": 300.0}


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table")
    a = p.parse_args()
    result = load(a.table)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tables", "primary_pdf", "primary_bin_weights")}, indent=2))
    for name in ("nParticles300", "particleFractions", *[s+x for s in SPECIES for x in ("KEDist", "TimeDist", "CosThetaDist", "ChargeDist")]):
        pdf = result["tables"][name]
        print(name, pdf.weights.shape, pdf.kind, pdf.low, pdf.high)
    print("raw mean multiplicities", result["tables"]["nParticles300"].mean())


if __name__ == "__main__":
    main()
