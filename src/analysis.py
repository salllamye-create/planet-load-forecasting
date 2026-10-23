"""Aggregate every result file into the paper's tables, statistics and figures."""
import glob, json, os, math, itertools
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
ERR = os.path.join(RES, "errors")
PAPER = os.path.join(HERE, "paper")
FIG = os.path.join(HERE, "figures")
os.makedirs(PAPER, exist_ok=True); os.makedirs(FIG, exist_ok=True)

ORDER = ["SeasonalNaive", "LSTM", "TCN", "Informer", "Autoformer", "NBEATS", "FITS",
         "NLinear", "DLinear", "RLinear", "TimeMixer", "iTransformer", "PatchTST",
         "CalTST", "PLANET"]
PRETTY = {"SeasonalNaive": "Seasonal-Na\\\"ive", "LSTM": "LSTM", "TCN": "TCN",
          "Informer": "Informer", "Autoformer": "Autoformer", "NBEATS": "N-BEATS",
          "FITS": "FITS", "NLinear": "NLinear", "DLinear": "DLinear", "RLinear": "RLinear",
          "TimeMixer": "TimeMixer", "iTransformer": "iTransformer", "PatchTST": "PatchTST",
          "CalTST": "PatchTST+Cal",
          "PLANET": "\\textbf{PLANET}", "HistGBR": "GBRT", "TSO": "TSO day-ahead"}
DS_PRETTY = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
             "ECL": "Electricity", "OPSD-DE": "DE", "OPSD-FR": "FR", "OPSD-GB": "GB"}


def load_all(tag_filter=None):
    out = []
    for p in glob.glob(os.path.join(RES, "*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "model" not in r or "mse" not in r:
            continue
        t = r.get("tag", "")
        if tag_filter is not None and not tag_filter(t):
            continue
        out.append(r)
    return out


def key(r):
    return (r["dataset"], r["model"], r["L"], r["H"], r["seed"], r.get("tag", ""))


def main_records():
    """Point-forecast records for the headline tables: no tag, plus the L=96
    Informer/Autoformer runs folded in as the better of the two lookbacks.

    The choice between lookbacks is made on VALIDATION loss. Selecting it on test MSE
    would hand those two baselines a test-set model selection that no other method gets.
    """
    recs = {}
    for r in load_all(lambda t: t in ("", "_L96")):
        k = (r["dataset"], r["model"], r["H"], r["seed"])
        cur = recs.get(k)
        if cur is None:
            recs[k] = r
            continue
        a, b = r.get("val_mse", float("inf")), cur.get("val_mse", float("inf"))
        a = a if np.isfinite(a) else float("inf")
        b = b if np.isfinite(b) else float("inf")
        if a < b:
            recs[k] = r
    return recs


# ------------------------------------------------------------------ Diebold-Mariano
def dm_test(e1, e2, h=1):
    """Two-sided Diebold-Mariano on squared-error loss differentials with a
    Newey-West HAC variance (Harvey-Leybourne-Newbold small-sample correction)."""
    d = np.asarray(e1, float) - np.asarray(e2, float)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dbar = d.mean()
    dc = d - dbar
    g0 = float((dc * dc).mean())
    # Consecutive test windows overlap by L-1 steps and their targets overlap by h-1, so the
    # loss differential is strongly autocorrelated. The truncation lag must cover that:
    # h-1 is the textbook choice for h-step forecasts, floored by the usual n^(1/3) rule.
    lag = max(1, int(round(n ** (1 / 3))), h - 1)
    lag = min(lag, max(1, n // 4))
    s = g0
    for k in range(1, lag + 1):
        gk = float((dc[k:] * dc[:-k]).mean())
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if s <= 0:
        return float("nan"), float("nan")
    stat = dbar / math.sqrt(s / n)
    corr = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-9))
    stat *= corr
    from math import erf
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / math.sqrt(2.0))))
    return float(stat), float(p)


def per_window(dataset, model, L, H, seed=2021, tag="", key="per_window"):
    """key='per_window' -> z-scored squared error (matches the MSE tables)
       key='per_window_ae' -> target-channel absolute error (matches the WAPE tables)"""
    p = os.path.join(ERR, f"{dataset}_{model}_L{L}_H{H}_s{seed}{tag}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z[key] if key in z.files else None


# ------------------------------------------------------------------ LaTeX helpers
def fmt(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def bold_best(vals, nd=3, lower_better=True):
    """Return formatted strings with the best in \\bf and the runner-up underlined."""
    arr = np.array([v if v is not None and np.isfinite(v) else np.inf for v in vals], float)
    if not lower_better:
        arr = -arr
    order = np.argsort(arr)
    out = [fmt(v, nd) for v in vals]
    if len(order) and np.isfinite(arr[order[0]]):
        out[order[0]] = "\\bfseries " + out[order[0]]
    if len(order) > 1 and np.isfinite(arr[order[1]]):
        out[order[1]] = "\\underline{" + out[order[1]] + "}"
    return out


def table_main(datasets, horizons, models, recs, caption, label, metric=("mse", "mae"),
               seed=2021, nd=3):
    """Datasets as row groups, models as columns -- the LTSF convention."""
    ncol = len(models)
    lines = [
        "\\begin{table*}[!t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{2.4pt}", "\\renewcommand{\\arraystretch}{1.05}",
        "\\footnotesize",
        "\\begin{tabular}{ll" + "cc" * ncol + "}", "\\toprule",
        "\\multicolumn{2}{c}{} & " + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{PRETTY.get(m, m)}}}" for m in models) + " \\\\",
        "\\cmidrule(lr){3-" + str(2 + 2 * ncol) + "}",
        "Data & $H$ & " + " & ".join("MSE & MAE" for _ in models) + " \\\\", "\\midrule",
    ]
    for d in datasets:
        for i, H in enumerate(horizons):
            row_mse, row_mae = [], []
            for m in models:
                r = recs.get((d, m, H, seed))
                row_mse.append(r["mse"] if r else None)
                row_mae.append(r["mae"] if r else None)
            bm, ba = bold_best(row_mse, nd), bold_best(row_mae, nd)
            head = f"\\multirow{{{len(horizons)}}}{{*}}{{{DS_PRETTY.get(d,d)}}}" if i == 0 else ""
            lines.append(f"{head} & {H} & " + " & ".join(
                f"{a} & {b}" for a, b in zip(bm, ba)) + " \\\\")
        lines.append("\\midrule")
    # average rank over every (dataset, horizon) cell
    ranks = {m: [] for m in models}
    for d in datasets:
        for H in horizons:
            v = [recs[(d, m, H, seed)]["mse"] if (d, m, H, seed) in recs else np.inf for m in models]
            o = np.argsort(np.argsort(v)) + 1
            for m, rk in zip(models, o):
                ranks[m].append(rk)
    lines.append("\\multicolumn{2}{l}{Avg.\\ MSE rank} & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{np.mean(ranks[m]):.2f}}}" for m in models) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def table_grid_ops(datasets, horizons, models, recs, caption, label, seed=2021):
    """Operational metrics on the national grids, in physical units."""
    lines = ["\\begin{table*}[!t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", "\\setlength{\\tabcolsep}{4pt}", "\\footnotesize",
             "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Grid & Method & WAPE (\\%) & MAE (MW) & RMSE (MW) & Peak MAPE (\\%) & Peak-time err.\\ (h) \\\\",
             "\\midrule"]
    for d in datasets:
        for H in horizons:
            vals = {m: recs.get((d, m, H, seed)) for m in models}
            wl = [vals[m]["wape"] if vals[m] else None for m in models]
            bb = bold_best(wl, 2)
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{DS_PRETTY.get(d,d)}, $H={H}$~h}}}} \\\\")
            for m, w in zip(models, bb):
                r = vals[m]
                if not r:
                    continue
                lines.append(f" & {PRETTY.get(m,m)} & {w} & {fmt(r.get('mae_orig'),0)} & "
                             f"{fmt(r.get('rmse_orig'),0)} & {fmt(r.get('peak_mape'),2)} & "
                             f"{fmt(r.get('peak_time_err'),2)} \\\\")
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


if __name__ == "__main__":
    recs = main_records()
    print(f"{len(recs)} point-forecast records")
    got = {}
    for (d, m, H, s) in recs:
        got.setdefault(d, set()).add(m)
    for d in sorted(got):
        print(f"  {d:10s} {len(got[d]):2d} models: {sorted(got[d])}")
