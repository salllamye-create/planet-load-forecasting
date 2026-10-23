"""Statistical tests for the v2 tables.

* dm_test           -- Diebold-Mariano (HAC / HLN), re-used verbatim from v1 ``analysis.py``.
* holm              -- Holm step-down adjusted p-values.
* mcs               -- Model Confidence Set (Hansen, Lunde & Nason, Econometrica 2011) with the
                       stationary bootstrap of Politis & Romano (1994) and the T_max statistic.
* pairwise_dm_table -- every model against a reference, raw and Holm-adjusted p-values.

Loss arrays are one value per test window (``per_window_ae`` or ``per_window`` in the
``results*/errors/<tag>.npz`` files); consecutive windows overlap, so every procedure here
uses an autocorrelation-robust variance (HAC for DM, block bootstrap for the MCS).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import dm_test  # noqa: E402  (v1 implementation; sign: stat < 0 -> e1 has lower loss)

__all__ = ["dm_test", "holm", "mcs", "pairwise_dm_table", "stationary_bootstrap_indices",
           "dm_bootstrap", "effective_n"]


# ------------------------------------------------------------------ Holm step-down
def holm(pvals):
    """Holm (1979) step-down adjusted p-values, aligned with the input.

    Non-finite entries are left as NaN and do not count towards the family size.
    Adjusted values are monotone in the raw ordering and never below the raw value.
    """
    p = np.asarray(pvals, float).ravel()
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * p[j])
        adj[rank] = min(1.0, running)
    out[order] = adj
    return out


# ------------------------------------------------------------------ stationary bootstrap
def stationary_bootstrap_indices(n, B, block_len, rng):
    """(B, n) int32 index matrix of Politis-Romano stationary-bootstrap resamples.

    Block lengths are geometric with mean ``block_len`` (``block_len == 1`` is the iid
    bootstrap); blocks wrap around the end of the series.
    """
    p = 1.0 / float(block_len)
    t = np.arange(n, dtype=np.int64)
    out = np.empty((B, n), dtype=np.int32)
    step = max(1, min(B, int(2e7 // max(n, 1))))       # ~20M entries per chunk
    for b0 in range(0, B, step):
        b = min(step, B - b0)
        new = rng.random((b, n)) < p
        new[:, 0] = True
        starts = rng.integers(0, n, size=(b, n), dtype=np.int64)
        s = np.maximum.accumulate(np.where(new, t, 0), axis=1)     # start position of the current block
        first = np.take_along_axis(starts, s, axis=1)              # series index where that block began
        out[b0:b0 + b] = (first + (t - s)) % n
    return out


def _bootstrap_means(L, B, block_len, rng):
    """(B, m) bootstrap means of the columns of L (n, m) under the stationary bootstrap.

    Only means are needed, so each resample is reduced to a count vector and the means come
    from one matrix product; memory stays O(chunk x n)."""
    n, m = L.shape
    out = np.empty((B, m))
    step = max(1, min(B, int(1e7 // max(n, 1))))
    for b0 in range(0, B, step):
        b = min(step, B - b0)
        idx = stationary_bootstrap_indices(n, b, block_len, rng).astype(np.int64)
        flat = idx + (np.arange(b, dtype=np.int64) * n)[:, None]
        counts = np.bincount(flat.ravel(), minlength=b * n).reshape(b, n).astype(float)
        out[b0:b0 + b] = counts @ L / n
    return out


# ------------------------------------------------------------------ Model Confidence Set
def mcs(losses, alpha=0.10, B=2000, block_len=None, stat="Tmax", seed=0):
    """Model Confidence Set of Hansen, Lunde & Nason (2011).

    Parameters
    ----------
    losses : (n_windows, n_models) array of per-window losses (lower is better).
    alpha : MCS level; the set contains every model whose MCS p-value is >= alpha.
    B : number of stationary-bootstrap resamples (shared by every elimination step, as in HLN).
    block_len : expected block length; default max(1, round(n ** (1/3))).
    stat : 'Tmax' (deviation from the set average, HLN's T_max built on t_{i.}) or
           'TR' (largest pairwise |t_ij|, HLN's T_R).
    seed : seed of the bootstrap RNG.

    Returns dict with ``in_set`` (bool per model), ``p_values`` (MCS p-values, monotone along
    the elimination order), ``elimination_order`` (model indices, worst first; the last entry
    is the final survivor), ``tstats`` (statistic of each model at the step it was eliminated),
    plus ``block_len``, ``n``, ``alpha``, ``stat``, ``B``.
    """
    L = np.asarray(losses, float)
    if L.ndim != 2:
        raise ValueError("losses must be (n_windows, n_models)")
    ok = np.isfinite(L).all(axis=1)
    L = L[ok]
    n, m = L.shape
    if n < 10:
        raise ValueError("too few finite windows for the MCS")
    if m == 1:
        return dict(in_set=np.ones(1, bool), p_values=np.ones(1), elimination_order=[0],
                    tstats=np.zeros(1), block_len=1, n=n, alpha=alpha, stat=stat, B=B)
    if block_len is None:
        block_len = max(1, int(round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    Lbar = L.mean(axis=0)                                   # (m,)
    Dboot = _bootstrap_means(L, B, block_len, rng) - Lbar   # (B, m): centred bootstrap means
    tiny = 1e-300

    alive = list(range(m))
    elim, p_seq, t_seq = [], [], []
    while len(alive) > 1:
        A = np.asarray(alive)
        if stat == "Tmax":
            dbar = Lbar[A] - Lbar[A].mean()                                # d_bar_{i.}
            dstar = Dboot[:, A] - Dboot[:, A].mean(axis=1, keepdims=True)  # d*_{i.} - d_bar_{i.}
            sd = np.sqrt(np.maximum((dstar ** 2).mean(axis=0), tiny))
            t = dbar / sd
            T = float(t.max())
            Tstar = (dstar / sd).max(axis=1)
            worst = int(np.argmax(t))
            tw = float(t[worst])
        elif stat == "TR":
            dij = Lbar[A][:, None] - Lbar[A][None, :]                      # (k, k)
            Dk = Dboot[:, A]
            dstar = Dk[:, :, None] - Dk[:, None, :]                        # (B, k, k)
            sd = np.sqrt(np.maximum((dstar ** 2).mean(axis=0), tiny))
            np.fill_diagonal(sd, 1.0)
            tij = dij / sd
            T = float(np.abs(tij).max())
            Tstar = np.abs(dstar / sd).max(axis=(1, 2))
            worst = int(np.argmax(tij.max(axis=1)))                        # largest t_{ij}: worst model
            tw = float(tij.max(axis=1)[worst])
        else:
            raise ValueError("stat must be 'Tmax' or 'TR'")
        p_seq.append(float(np.mean(Tstar >= T)))
        t_seq.append(tw)
        elim.append(alive.pop(worst))
    elim.append(alive[0])
    p_seq.append(1.0)
    t_seq.append(0.0)

    p_mcs = np.maximum.accumulate(np.asarray(p_seq))      # HLN: p_hat(e_i) = max_{j <= i} P_{M_j}
    p_values = np.empty(m)
    p_values[elim] = p_mcs
    tstats = np.empty(m)
    tstats[elim] = t_seq
    return dict(in_set=p_values >= alpha, p_values=p_values, elimination_order=elim,
                tstats=tstats, block_len=block_len, n=n, alpha=alpha, stat=stat, B=B)


# ------------------------------------------------------------------ bootstrap DM, effective n
def dm_bootstrap(e1, e2, h=1, B=1000, block_len=None, seed=0, empirical=False):
    """Diebold-Mariano test with a stationary-bootstrap standard error for the mean loss
    differential (Politis & Romano 1994), the resampling scheme of the MCS: the expected block
    length defaults to max(n^(1/3), h), which covers the h-1 target overlap of consecutive
    windows, where the Bartlett kernel of ``dm_test`` can under-state the long-run variance.
    Returns (stat, p): stat = mean differential / bootstrap sd of the mean (negative: e1 has
    the lower loss); p = two-sided p-value of stat under the normal reference distribution.
    empirical=True returns instead the bootstrap p-value P(|dbar* - dbar| >= |dbar|) floored at
    1/B, which cannot resolve below 1/B and is therefore useless under a Holm correction over
    tens of tests.
    """
    d = np.asarray(e1, float) - np.asarray(e2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    if block_len is None:
        block_len = max(1, int(round(n ** (1.0 / 3.0))), int(h))
    rng = np.random.default_rng(seed)
    means = _bootstrap_means(d[:, None], B, block_len, rng)[:, 0]
    dbar = float(d.mean())
    sd = float(means.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan"), float("nan")
    stat = dbar / sd
    if empirical:
        return stat, max(float(np.mean(np.abs(means - dbar) >= abs(dbar))), 1.0 / B)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / sqrt(2.0))))
    return stat, float(p)


def effective_n(d, lag=None):
    """Effective sample size of a loss-differential series under its HAC long-run variance,
    n / (1 + 2 sum_k w_k rho_k) with Bartlett weights w_k = 1 - k/(lag+1) (default lag n^(1/3)).
    With consecutive overlapping windows this is far below n and is the number the Diebold-
    Mariano statistic effectively rests on."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan")
    dc = d - d.mean()
    g0 = float((dc * dc).mean())
    if g0 <= 0:
        return float("nan")
    lag = lag or max(1, int(round(n ** (1.0 / 3.0))))
    lag = min(int(lag), max(1, n // 4))
    s = 1.0
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * float((dc[k:] * dc[:-k]).mean()) / g0
    return float(n / max(s, 1e-9))


# ------------------------------------------------------------------ DM table
def pairwise_dm_table(err_dict, ref, h):
    """Diebold-Mariano of every model in ``err_dict`` against ``ref``.

    ``err_dict`` maps model name -> per-window loss array (e.g. ``per_window_ae``), all on
    the same test windows. Returns {model: {stat, p, p_holm, n}}; ``stat < 0`` means the
    reference has the lower loss. Holm is applied across the models in the dict.
    """
    if ref not in err_dict:
        raise KeyError(f"reference model {ref!r} not in err_dict")
    e_ref = np.asarray(err_dict[ref], float)
    names = [m for m in err_dict if m != ref]
    rows = {}
    for m in names:
        e = np.asarray(err_dict[m], float)
        if len(e) != len(e_ref):
            rows[m] = dict(stat=float("nan"), p=float("nan"), n=int(min(len(e), len(e_ref))))
            continue
        s, p = dm_test(e_ref, e, h=h)
        rows[m] = dict(stat=s, p=p, n=int(len(e)))
    adj = holm([rows[m]["p"] for m in names])
    for m, q in zip(names, adj):
        rows[m]["p_holm"] = float(q)
    return rows
