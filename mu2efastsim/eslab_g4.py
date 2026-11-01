"""Compare mu2efastsim.eslab with a Geant4 electron-gun slab reference.

Reference layout (one directory per case, one or more chunkNN/ subdirectories): `case.json` (material, thickness_mm,
T_MeV, pdg, events, seed, msc), `job.log` (the Geant4 couple table, from which the production cuts are read) and
`foil.root` with the Mu2eG4StudyReadBack `readback/tpart` tree: a slab of thickness t centred on z = 0 in a vacuum
world, a fixed-energy gun along +z, every secondary killed after its first step. The primary's end point and momentum
at the world boundary give its fate (transmitted, reflected, stopped inside, annihilated) and, straight back to the
slab face, its exit angle and lateral displacement; its brem photons (process code 16), delta rays (17) and
annihilation photons (2) are its direct children.

Each observable is compared with a two-sample chi2 on shared bins, sum (sqrt(m/n) a - sqrt(n/m) b)^2 / (a + b) over
bins with entries, which is chi2-distributed with one degree of freedom per bin when both samples come from the same
distribution. Usage:
    python3 -m mu2efastsim.eslab_g4 CASE_DIR [CASE_DIR ...] [--msc] [--no-annihilation] [--events N] [--out FILE]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import emsc, eslab, materials

ME = 0.51099895
CODE = {"annihil": 2, "eBrem": 16, "eIoni": 17, "primary": 56}
UNIT = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}
BRANCHES = ["evt", "trk", "pdg", "code", "parent_id", "p", "px", "py", "pz", "x", "y", "z", "pstop", "pxstop",
            "pystop", "pzstop", "xstop", "ystop", "zstop", "codestop", "endvol"]


def cuts_from_log(text, material):
    """(gamma cut, e- cut) [MeV] of the couple of `material` in a Geant4 /run/verbose 2 couple table."""
    for block in text.split("Index :")[1:]:
        if re.search(rf"Material\s*:\s*{re.escape(material)}\b", block):
            m = re.search(r"Energy thresholds\s*:\s*gamma\s+([\d.eE+-]+)\s*(\w+)\s+e-\s+([\d.eE+-]+)\s*(\w+)", block)
            if m:
                return float(m[1]) * UNIT[m[2]], float(m[3]) * UNIT[m[4]]
    raise ValueError(f"no couple for {material}")


def chi2_bins(a, b, edges):
    """Per-bin two-sample chi2 terms (0 for empty bins), bin mask with entries, and the ratio b/a scaled to equal
    sample sizes (nan where a is empty)."""
    ha, hb = np.histogram(a, edges)[0].astype(float), np.histogram(b, edges)[0].astype(float)
    na, nb = ha.sum(), hb.sum()
    m = (ha + hb) > 0
    if na == 0 or nb == 0:
        return np.full(ha.shape, np.nan), m, np.full(ha.shape, np.nan)
    num = (np.sqrt(nb / na) * ha - np.sqrt(na / nb) * hb) ** 2
    terms = np.where(m, num / np.where(m, ha + hb, 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (hb / nb) / (ha / na)
    return terms, m, ratio


def chi2(a, b, edges):
    """Two-sample chi2 and number of bins with entries for samples a, b of any sizes."""
    terms, m, _ = chi2_bins(a, b, edges)
    return [float(terms.sum()), int(m.sum())]


def quantile_edges(x, nbins, lo=None, hi=None):
    """Bin edges at equal-probability quantiles of x, widened to [lo, hi]."""
    e = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if lo is not None:
        e[0] = min(e[0], lo)
    if hi is not None:
        e[-1] = max(e[-1], hi)
    return e


def summary(x):
    if x.size == 0:
        return dict(n=0)
    q = np.quantile(x, [0.1, 0.5, 0.9, 0.99])
    return dict(n=int(x.size), mean=float(x.mean()), err=float(x.std() / np.sqrt(x.size)),
                rms=float(np.sqrt((x * x).mean())),
                q10=float(q[0]), q50=float(q[1]), q90=float(q[2]), q99=float(q[3]))


class Sample(dict):
    """Per-primary arrays (fate, T_exit, theta, r) and per-secondary arrays (track, energy, angle to +z) of one arm."""


def read_g4(chunks, thickness):
    """Concatenate the chunks of one Geant4 case into a Sample."""
    import uproot
    parts, off = [], 0
    for d in chunks:
        t = uproot.open(Path(d) / "foil.root")["readback/tpart"].arrays(BRANCHES, library="np")
        n = json.loads((Path(d) / "case.json").read_text())["events"]
        ev = t["evt"].astype(np.int64)
        ev = ev - ev.min()
        prim = (t["code"] == CODE["primary"]) & (t["trk"] == 1)
        if prim.sum() != n or np.unique(ev[prim]).size != n:
            raise ValueError(f"{d}: {prim.sum()} primaries for {n} events")
        order = np.argsort(ev[prim])
        g = {k: t[k][prim][order].astype(float) for k in BRANCHES}
        pst = np.maximum(g["pstop"], 1e-30)
        dirn = np.stack([g["pxstop"], g["pystop"], g["pzstop"]], axis=1) / pst[:, None]
        inside = g["endvol"] != 0
        annih = inside & (g["codestop"] == CODE["annihil"]) & (g["pstop"] > 0)   # in flight
        refl = ~inside & (dirn[:, 2] < 0)
        trans = ~inside & ~refl
        texit = np.where(inside, 0.0, np.sqrt(g["pstop"] ** 2 + ME * ME) - ME)
        face = np.where(refl, -thickness / 2, thickness / 2)
        lever = (g["zstop"] - face) / np.where(dirn[:, 2] != 0, dirn[:, 2], 1.0)
        x = g["xstop"] - lever * dirn[:, 0]
        y = g["ystop"] - lever * dirn[:, 1]
        child = t["parent_id"] == 1
        p = t["p"].astype(float)
        theta = np.arctan2(np.hypot(t["px"].astype(float), t["py"].astype(float)), t["pz"].astype(float))
        sec = {}
        for name, code, pdg in (("photons", CODE["eBrem"], 22), ("deltas", CODE["eIoni"], 11),
                                ("annihilation", CODE["annihil"], 22)):
            m = child & (t["code"] == code) & (t["pdg"] == pdg)
            e = p[m] if pdg == 22 else np.sqrt(p[m] ** 2 + ME * ME) - ME
            sec[name] = (ev[m] + off, e, theta[m])
        parts.append(dict(T=texit, trans=trans, refl=refl, stop=inside & ~annih, annih=annih,
                          theta=np.arctan2(np.hypot(g["pxstop"], g["pystop"]), g["pzstop"]), r=np.hypot(x, y), sec=sec,
                          n=n))
        off += n
    return _merge(parts)


def _merge(parts):
    out = Sample(n=sum(p["n"] for p in parts))
    for k in ("T", "trans", "refl", "stop", "annih", "theta", "r"):
        out[k] = np.concatenate([p[k] for p in parts])
    out["sec"] = {name: tuple(np.concatenate([p["sec"][name][i] for p in parts]) for i in range(3))
                  for name in parts[0]["sec"]}
    return out


def _theta(d):
    """Polar angle to +z, accurate at small angles."""
    return np.arctan2(np.hypot(d[:, 0], d[:, 1]), d[:, 2])


def run_fast(case, n, tcut, kcut, msc, annihilation, seed=None, **kw):
    """The fast slab with the case's material, thickness, energy and particle for n primaries."""
    mat = materials.BY_NAME[case["material"]]
    T0, L, pos = case["T_MeV"], case["thickness_mm"], case["pdg"] == -11
    rng = np.random.default_rng(case["seed"] if seed is None else seed)
    r = eslab.electron_slab(rng, mat, np.full(n, T0), np.tile([0.0, 0.0, 1.0], (n, 1)), L, tcut, kcut, pos,
                            msc=msc, annihilation=annihilation, **kw)
    stop = r.stopped & ~r.annihilated
    trans = ~r.stopped & ~r.reflected
    out = Sample(n=n, T=r.T, trans=trans, refl=r.reflected, stop=stop, annih=r.annihilated,
                 theta=_theta(r.direction), r=np.hypot(r.position[:, 0], r.position[:, 1]))
    out["sec"] = {name: (s.track, s.energy, _theta(s.direction)) for name, s in
                  (("photons", r.photons), ("deltas", r.deltas), ("annihilation", r.annihilation))}
    return out


def compare_samples(case, g, f, tcut, kcut):
    T0 = case["T_MeV"]
    out = dict(case=case.get("case"), events=[g["n"], f["n"]], material=case["material"],
               thickness_mm=case["thickness_mm"], T_MeV=T0, positron=case["pdg"] == -11, kcut_MeV=kcut,
               tcut_MeV=tcut)
    for k in ("trans", "refl", "stop", "annih"):
        a, b = g[k].sum(), f[k].sum()
        out[f"frac_{k}"] = [float(a / g["n"]), float(b / f["n"])]
        out[f"frac_{k}_err"] = [float(np.sqrt(max(a, 1)) / g["n"]), float(np.sqrt(max(b, 1)) / f["n"])]
    gl, fl = T0 - g["T"][g["trans"]], T0 - f["T"][f["trans"]]
    out["loss_g4"], out["loss_fast"] = summary(gl), summary(fl)
    le = np.concatenate([[0.0], np.geomspace(1e-3, T0, 41)])
    out["loss_chi2_ndf"] = chi2(gl, fl, le)
    out["loss_chi2_ndf_quantile"] = chi2(gl, fl, quantile_edges(gl, 50, 0.0, T0))
    for lab, thr in (("gt1", 1.0), ("gt5", 5.0), ("gt20", 20.0)):
        out[f"frac_loss_{lab}"] = [float((gl > thr).mean()), float((fl > thr).mean())]
    # transmitted primaries leaving with little energy, per generated primary
    low = lambda s, t: float((s["trans"] & (s["T"] < t)).mean())
    out["frac_trans_T_below"] = {str(t): [low(g, t), low(f, t)] for t in (0.1, 0.3, 1.0, 3.0, 10.0)}
    for k in ("theta", "r"):
        a, b = g[k][g["trans"]], f[k][f["trans"]]
        out[f"{k}_g4"], out[f"{k}_fast"] = summary(a), summary(b)
        out[f"{k}_chi2_ndf"] = chi2(a, b, quantile_edges(a, 50, 0.0, max(a.max(), b.max()) * 1.0001))
        if k == "theta":
            # 50 equal-probability G4 bins: core = the first 45 (below the G4 90 % quantile), tail = the last 5
            terms, m, ratio = chi2_bins(a, b, quantile_edges(a, 50, 0.0, max(a.max(), b.max()) * 1.0001))
            out["theta_chi2_core_tail"] = [[float(terms[:45].sum()), int(m[:45].sum())],
                                           [float(terms[45:].sum()), int(m[45:].sum())]]
            out["theta_ratio_fast_g4"] = [round(float(v), 4) for v in ratio]
            q = np.quantile(a, [0.5, 0.9, 0.99])
            out["theta_tail_fracs"] = {f"gt_g4q{int(100 * p)}": [float((a > v).mean()), float((b > v).mean())]
                                       for p, v in zip((0.5, 0.9, 0.99), q)}
    for name, lo, hi in (("photons", kcut, T0), ("deltas", tcut, T0 if case["pdg"] == -11 else T0 / 2),
                         ("annihilation", 1e-3, T0 + 2 * ME)):
        ga, fa = g["sec"][name], f["sec"][name]
        e = np.geomspace(lo, hi, 31)
        th = np.concatenate([[0.0], np.geomspace(1e-4, np.pi, 30)])
        out[name] = dict(per_event=[ga[0].size / g["n"], fa[0].size / f["n"]],
                         per_event_err=[np.sqrt(ga[0].size) / g["n"], np.sqrt(fa[0].size) / f["n"]],
                         energy_chi2_ndf=chi2(ga[1], fa[1], e),
                         theta_chi2_ndf=chi2(ga[2], fa[2], th))
    return out


def compare(case_dir, msc=True, annihilation=True, events=None, seed=None, **kw):
    chunks = sorted(str(p) for p in Path(case_dir).glob("chunk*") if (p / "foil.root").exists())
    case = json.loads((Path(chunks[0]) / "case.json").read_text())
    kcut, tcut = cuts_from_log((Path(chunks[0]) / "job.log").read_text(), case["material"])
    g = read_g4(chunks, case["thickness_mm"])
    f = run_fast(case, events or g["n"], tcut, kcut, msc, annihilation, seed, **kw)
    out = compare_samples(case, g, f, tcut, kcut)
    out["fast_msc"], out["fast_annihilation"], out["g4_msc"] = msc, annihilation, case.get("msc", True)
    out["fast_options"] = {k: v for k, v in kw.items() if v is not None}
    return out


def table(reports):
    """Markdown rows: fates, loss of transmitted primaries, exit angle and lateral displacement, secondaries."""
    f = lambda c: f"{c[0] / max(c[1], 1):.1f}"
    rows = ["| case | trans G4/fast | refl G4/fast | stop G4/fast | annih G4/fast | loss mean G4/fast | loss chi2/ndf "
            "| theta rms G4/fast [mrad] | theta chi2/ndf | r mean G4/fast [um] | r chi2/ndf | photons/ev G4/fast | "
            "k chi2/ndf | photon theta chi2/ndf | deltas/ev G4/fast | w chi2/ndf | delta theta chi2/ndf |",
            "|" + "---|" * 17]
    for r in reports:
        fr = lambda k: f"{r[k][0]:.4g} / {r[k][1]:.4g}"
        ph, de = r["photons"], r["deltas"]
        rows.append(
            f"| {r['case']} | {fr('frac_trans')} | {fr('frac_refl')} | {fr('frac_stop')} | {fr('frac_annih')} | "
            f"{r['loss_g4']['mean']:.4f} / {r['loss_fast']['mean']:.4f} (+-{r['loss_g4']['err']:.4f}) | "
            f"{f(r['loss_chi2_ndf'])} | {1e3 * r['theta_g4']['rms']:.3f} / {1e3 * r['theta_fast']['rms']:.3f} | "
            f"{f(r['theta_chi2_ndf'])} | {1e3 * r['r_g4']['mean']:.2f} / {1e3 * r['r_fast']['mean']:.2f} | "
            f"{f(r['r_chi2_ndf'])} | {ph['per_event'][0]:.4f} / {ph['per_event'][1]:.4f} | "
            f"{f(ph['energy_chi2_ndf'])} | "
            f"{f(ph['theta_chi2_ndf'])} | {de['per_event'][0]:.4f} / {de['per_event'][1]:.4f} | "
            f"{f(de['energy_chi2_ndf'])} | {f(de['theta_chi2_ndf'])} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.eslab_g4")
    ap.add_argument("cases", nargs="+", help="Geant4 case directories, or report JSON files with --table")
    ap.add_argument("--table", action="store_true", help="print a markdown table of report JSON files")
    ap.add_argument("--msc", action="store_true", help="fast arm with multiple scattering")
    ap.add_argument("--no-annihilation", action="store_true")
    ap.add_argument("--events", type=int, help="fast-arm primaries (default: as many as Geant4)")
    ap.add_argument("--seed", type=int, help="fast-arm seed (default: the case's)")
    ap.add_argument("--max-step", type=float, help="fast-arm msc step cap [mm] (default: thickness / 10)")
    ap.add_argument("--hard-per-step", type=float, help="fast-arm least hard collisions per msc step cap")
    ap.add_argument("--no-mott", action="store_true", help="diagnostic: hard collisions without the Mott factor")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.no_mott:
        emsc.MOTT = False
    if a.table:
        print(table([r for c in a.cases for r in (lambda x: x if isinstance(x, list) else [x])(
            json.loads(Path(c).read_text()))]))
        return
    kw = {k: v for k, v in (("max_step", a.max_step), ("hard_per_step", a.hard_per_step)) if v is not None}
    rep = [compare(c, a.msc, not a.no_annihilation, a.events, a.seed, **kw) for c in a.cases]
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
