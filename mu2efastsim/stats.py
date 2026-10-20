"""Robust statistics shared by every comparison, so the numbers are consistent across analyses."""
import numpy as np


def resid_stats(arr):
    """n, mean, std, median and robust core sigma (p84-p16)/2 of an array; NaNs dropped."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "coresig": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0)), "median": float(np.median(a)),
            "coresig": float((np.percentile(a, 84) - np.percentile(a, 16)) / 2.0)}


def mpv(x, lo=0.0, hi=15.0, nbins=150, half_window=3):
    """Most probable value: count-weighted mean of the bin centres within +-half_window bins of the histogram peak."""
    h, e = np.histogram(np.asarray(x, dtype=float), bins=nbins, range=(lo, hi))
    k = int(np.argmax(h))
    sl = slice(max(k - half_window, 0), min(k + half_window + 1, nbins))
    c = 0.5 * (e[:-1] + e[1:])
    w = h[sl].astype(float)
    return float((c[sl] * w).sum() / w.sum()) if w.sum() else float("nan")


def mpv_fit(x, lo=0.0, hi=15.0, nbins=200, lo_q=0.10, hi_q=0.60):
    """Most probable value and its statistical error from a Moyal (Landau approximation) fit to the histogram core.

    Fit range: the lo_q to hi_q quantiles of the sample inside [lo, hi] (rising edge, peak, start of the falling edge;
    not the long tail). Weighted least squares on counts with Poisson errors. Returns (mpv, err); nan if the fit fails.
    Unlike mpv (argmax), it uses the whole core, so a flat-topped peak does not make it jump between bins.
    """
    from scipy.optimize import curve_fit
    h, e = np.histogram(np.asarray(x, dtype=float), bins=nbins, range=(lo, hi))
    h = h.astype(float)
    c = 0.5 * (e[:-1] + e[1:])
    if h.sum() < 100:
        return float("nan"), float("nan")
    cdf = np.cumsum(h) / h.sum()
    sel = (cdf >= lo_q) & (cdf <= hi_q) & (h > 0)
    if sel.sum() < 4:
        return float("nan"), float("nan")

    def moyal(t, a, mu, sig):
        z = (t - mu) / sig
        return a * np.exp(-0.5 * (z + np.exp(-z)))

    k = int(np.argmax(np.where(sel, h, 0)))
    p0 = (h[k] / np.exp(-0.5), c[k], max(0.3 * c[k], (hi - lo) / nbins))
    try:
        par, cov = curve_fit(moyal, c[sel], h[sel], p0=p0, sigma=np.sqrt(h[sel]), absolute_sigma=True, maxfev=10000)
        return float(par[1]), float(np.sqrt(cov[1, 1]))
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")


def ratio_band(fast, ref, q=(0.05, 0.95), nbins=20, min_ref=400):
    """Binned fast/ref of unit-normalised histograms over the reference's q-range.

    Returns the max |ratio - 1| over bins with >= min_ref reference entries, also with a 2-sigma statistical allowance
    subtracted (the quantity compared with the V1/V2 binned-ratio thresholds), and the edges and counts used.
    """
    fast = np.asarray(fast, dtype=float)
    ref = np.asarray(ref, dtype=float)
    lo, hi = np.quantile(ref, q)
    hf, edges = np.histogram(fast, bins=nbins, range=(lo, hi))
    hr, _ = np.histogram(ref, bins=nbins, range=(lo, hi))
    ok = hr >= min_ref
    if not ok.any():
        return {"max_abs_dev": None, "max_abs_dev_minus_2sigma": None, "nbins_used": 0,
                "edges": edges, "h_fast": hf, "h_ref": hr}
    r = (hf[ok] / len(fast)) / (hr[ok] / len(ref))
    err = r * np.sqrt(1 / np.maximum(hf[ok], 1) + 1 / hr[ok])
    return {"max_abs_dev": float(np.abs(r - 1).max()),
            "max_abs_dev_minus_2sigma": float(np.maximum(np.abs(r - 1) - 2 * err, 0).max()),
            "nbins_used": int(ok.sum()), "edges": edges, "h_fast": hf, "h_ref": hr}
