"""Shared plotting helpers and the stats-JSON writer (same contract as crvextrap.plots and crvfastsim.plots)."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_FAST = "crimson"
C_REF = "navy"


def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save(fig, outdir, name, dpi=120):
    ensure_outdir(outdir)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write_json(outdir, name, obj):
    """Dump a stats dict beside the plots; NaN/inf become null so the JSON stays standard."""
    ensure_outdir(outdir)
    path = os.path.join(outdir, name)
    with open(path, "w") as fh:
        json.dump(_clean(obj), fh, indent=2)
    print(f"wrote {path}")
    return path


def ratio_panel(ax_top, ax_bot, edges, h_ref, h_fast, xlabel, labels=("G4 reference", "fast sim")):
    """Normalised overlay on top, fast/ref ratio with statistical errors below."""
    c = 0.5 * (edges[:-1] + edges[1:])
    nr, nf = max(h_ref.sum(), 1), max(h_fast.sum(), 1)
    ax_top.stairs(h_ref / nr, edges, color=C_REF, label=labels[0])
    ax_top.stairs(h_fast / nf, edges, color=C_FAST, label=labels[1])
    ax_top.legend(fontsize=8)
    ok = h_ref > 0
    r = np.full_like(c, np.nan, dtype=float)
    e = np.full_like(c, np.nan, dtype=float)
    r[ok] = (h_fast[ok] / nf) / (h_ref[ok] / nr)
    e[ok] = r[ok] * np.sqrt(1 / np.maximum(h_fast[ok], 1) + 1 / h_ref[ok])
    ax_bot.errorbar(c, r, yerr=e, fmt=".", color=C_FAST)
    ax_bot.axhline(1, color="k", lw=0.8)
    ax_bot.set_ylim(0.8, 1.2)
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("fast / G4")
