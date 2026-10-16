"""Array joins on multi-column integer keys and a two-sample KS test, shared by the fast-vs-stock comparisons.

Keys are int64 arrays [N] or [N, ncol] (e.g. (event, collection index, sequence)); every operation is a sort/merge on a
packed byte view of the rows, so there are no per-row Python loops and no bit-packing limits.
"""
import numpy as np


def as_void(k):
    """Rows of an int64 key array as one opaque scalar each (sortable, comparable)."""
    k = np.ascontiguousarray(np.asarray(k, np.int64).reshape(len(k), -1))
    return k.view(np.dtype((np.void, k.dtype.itemsize * k.shape[1]))).ravel()


def join(ka, kb):
    """Indices (ia, ib) of the rows with equal keys in both arrays (keys unique within each side)."""
    _, ia, ib = np.intersect1d(as_void(ka), as_void(kb), assume_unique=False, return_indices=True)
    return ia, ib


def lookup(keys, table_keys, table_vals, missing=np.nan):
    """Many-to-one lookup: the table value of the row of table_keys equal to each row of keys (table keys unique)."""
    vk, vt = as_void(keys), as_void(table_keys)
    out = np.full(len(vk), missing, dtype=float)
    if not len(vt):
        return out
    order = np.argsort(vt)
    pos = np.clip(np.searchsorted(vt[order], vk), 0, len(vt) - 1)
    hit = vt[order][pos] == vk
    out[hit] = np.asarray(table_vals, float)[order][pos[hit]]
    return out


def ks_2samp(a, b):
    """Two-sample KS statistic and asymptotic p-value (no scipy dependency)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return float("nan"), float("nan")
    x = np.concatenate([a, b])
    d = np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0, 1))
    return float(d), p
