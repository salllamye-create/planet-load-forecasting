"""Generate every LaTeX table in the paper from the saved result files.

Tables are assembled by plain line concatenation rather than f-strings: LaTeX is
backslash- and brace-dense, and every formatting engine fights with one or the other.
"""
import glob, json, os
import numpy as np

import analysis as an

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, ERR = os.path.join(HERE, "results"), os.path.join(HERE, "results", "errors")
TAB = os.path.join(HERE, "paper", "tables")
os.makedirs(TAB, exist_ok=True)

ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
GRID = ["OPSD-DE", "OPSD-FR", "OPSD-GB"]
BENCH_H = [96, 192, 336, 720]
GRID_H = [24, 96, 336]
MODELS = an.ORDER + ["HistGBR"]

BS = chr(92)          # backslash
NL = BS + BS          # LaTeX row terminator


def write(name, lines):
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  [tab] " + name + ".tex")


def cell(v, nd=3):
    return "--" if v is None or not np.isfinite(v) else format(float(v), "." + str(nd) + "f")


def mark(vals, nd=3):
    """Bold the best and underline the runner-up among the finite entries."""
    a = np.array([v if (v is not None and np.isfinite(v)) else np.inf for v in vals], float)
    out = [cell(v, nd) for v in vals]
    if not np.isfinite(a).any():
        return out
    order = np.argsort(a)
    out[order[0]] = BS + "bfseries " + out[order[0]]
    if len(order) > 1 and np.isfinite(a[order[1]]):
        out[order[1]] = BS + "underline{" + out[order[1]] + "}"
    return out


def head(env, caption, label, colspec, extra="", tabcolsep="3.2pt"):
    return [BS + "begin{" + env + "}[!t]", BS + "centering",
            BS + "caption{" + caption + "}", BS + "label{" + label + "}",
            BS + "setlength{" + BS + "tabcolsep}{" + tabcolsep + "}" + BS + "footnotesize" + extra,
            BS + "begin{tabular}{" + colspec + "}", BS + "toprule"]


def foot(env):
    return [BS + "bottomrule", BS + "end{tabular}", BS + "end{" + env + "}"]


def mc(n, align, text):
    return BS + "multicolumn{" + str(n) + "}{" + align + "}{" + text + "}"


def cmid(pairs):
    return " ".join(BS + "cmidrule(lr){" + str(a) + "-" + str(b) + "}" for a, b in pairs)


# ------------------------------------------------------------------ 0. datasets
def tab_datasets(recs=None):
    rows = []
    for d in ETT + ["ECL"] + GRID:
        p = os.path.join(HERE, "data", "proc", d + ".json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        tr, va, te = (m["splits"][k] for k in ("train", "val", "test"))
        rows.append(" & ".join([
            an.DS_PRETTY.get(d, d), m["freq"], str(m["C"]),
            format(m["T"], ",").replace(",", BS + ","),
            m["start"][:7] + "--" + m["end"][:7],
            "%d/%d/%d" % (tr[1] - tr[0], va[1] - va[0], te[1] - te[0]),
            "MS" if d.startswith("OPSD") else "M"]) + " " + NL)
    if not rows:
        return
    cap = ("Benchmark corpus. " + BS + "textsc{m} scores every channel; " + BS +
           "textsc{ms} takes all channels as input and scores the load channel. The national "
           "grids additionally carry the TSO's own published day-ahead load forecast.")
    write("tab_datasets",
          head("table", cap, "tab:datasets", "lccrccc") +
          ["Dataset & Freq. & $C$ & $T$ & Span & Train/Val/Test & Setting " + NL, BS + "midrule"] +
          rows + foot("table"))


# ------------------------------------------------------------------ 1. benchmark summary
def tab_benchmark(recs):
    datasets = [d for d in ETT + ["ECL"] if any((d, "PLANET", H, 2021) in recs for H in BENCH_H)]
    models = [m for m in an.ORDER
              if any((d, m, H, 2021) in recs for d in datasets for H in BENCH_H)]
    if not datasets or not models:
        return
    hs = {d: [H for H in BENCH_H if (d, "PLANET", H, 2021) in recs] for d in datasets}

    def avg(d, m, k):
        v = [recs[(d, m, H, 2021)][k] for H in hs[d] if (d, m, H, 2021) in recs]
        return float(np.mean(v)) if len(v) == len(hs[d]) and v else None

    cols = {d: (mark([avg(d, m, "mse") for m in models]),
                mark([avg(d, m, "mae") for m in models])) for d in datasets}
    ranks = {m: [] for m in models}
    for d in datasets:
        for H in hs[d]:
            avail = [m for m in models if (d, m, H, 2021) in recs]
            if len(avail) < 3:
                continue
            vals = [recs[(d, m, H, 2021)]["mse"] for m in avail]
            for m, r in zip(avail, np.argsort(np.argsort(vals)) + 1):
                ranks[m].append(int(r))
    rk = mark([float(np.mean(ranks[m])) if ranks[m] else None for m in models], 2)
    body = []
    for i, m in enumerate(models):
        cs = [cols[d][0][i] + " & " + cols[d][1][i] for d in datasets]
        body.append(an.PRETTY.get(m, m) + " & " + " & ".join(cs) + " & " + rk[i] + " " + NL)
    cap = ("Long-horizon benchmark, averaged over the evaluated horizons (multivariate "
           "setting, $L=336$, z-scored data). Best in " + BS + "textbf{bold}, second " +
           BS + "underline{underlined}. Informer and Autoformer are reported at the better "
           "of $L=336$ and their native $L=96$. Per-horizon results are in Table~" +
           BS + "ref{tab:benchfull}.")
    write("tab_benchmark",
          head("table*", cap, "tab:benchmark", "l" + "cc" * len(datasets) + "c", tabcolsep="3.4pt") +
          [" & " + " & ".join(mc(2, "c", an.DS_PRETTY.get(d, d)) for d in datasets) +
           " & Avg. " + NL,
           cmid([(2 + 2 * i, 3 + 2 * i) for i in range(len(datasets))]),
           "Method & " + " & ".join(["MSE & MAE"] * len(datasets)) + " & rank " + NL,
           BS + "midrule"] + body + foot("table*"))


# ------------------------------------------------------------------ 2. full per-horizon
def tab_benchfull(recs, models=None):
    models = models or ["Autoformer", "FITS", "DLinear", "TimeMixer", "iTransformer",
                        "PatchTST", "CalTST", "PLANET"]
    datasets = [d for d in ETT + ["ECL"] if any((d, "PLANET", H, 2021) in recs for H in BENCH_H)]
    models = [m for m in models
              if any((d, m, H, 2021) in recs for d in datasets for H in BENCH_H)]
    if not datasets or not models:
        return
    lines = []
    for d in datasets:
        hs = [H for H in BENCH_H if (d, "PLANET", H, 2021) in recs]
        for i, H in enumerate(hs):
            mse = mark([recs[(d, m, H, 2021)]["mse"] if (d, m, H, 2021) in recs else None
                        for m in models])
            mae = mark([recs[(d, m, H, 2021)]["mae"] if (d, m, H, 2021) in recs else None
                        for m in models])
            pre = (BS + "multirow{" + str(len(hs)) + "}{*}{" + an.DS_PRETTY.get(d, d) + "}"
                   if i == 0 else "")
            lines.append(pre + " & " + str(H) + " & " +
                         " & ".join(a + " & " + b for a, b in zip(mse, mae)) + " " + NL)
        lines.append(BS + "midrule")
    lines = lines[:-1]
    write("tab_benchfull",
          head("table*", "Per-horizon benchmark results (multivariate setting, $L=336$).",
               "tab:benchfull", "ll" + "cc" * len(models), tabcolsep="2.6pt") +
          ["Data & $H$ & " + " & ".join(mc(2, "c", an.PRETTY.get(m, m)) for m in models) +
           " " + NL,
           cmid([(3, 2 + 2 * len(models))]),
           " & & " + " & ".join(["MSE & MAE"] * len(models)) + " " + NL,
           BS + "midrule"] + lines + foot("table*"))


# ------------------------------------------------------------------ 3. national grids
def tab_grids(recs):
    models = [m for m in MODELS if any((d, m, H, 2021) in recs for d in GRID for H in GRID_H)]
    hs = [H for H in GRID_H if any((d, m, H, 2021) in recs for d in GRID for m in models)]
    if not models or not hs:
        return
    lines = []
    for d in GRID:
        if not any((d, m, H, 2021) in recs for m in models for H in hs):
            continue
        lines.append(mc(1 + 3 * len(hs), "l", BS + "textit{" + an.DS_PRETTY.get(d, d) + "}") +
                     " " + NL)
        cols = {}
        for H in hs:
            def g(k):
                return [recs[(d, m, H, 2021)].get(k) if (d, m, H, 2021) in recs else None
                        for m in models]
            cols[H] = (mark(g("wape"), 2), mark(g("peak_mape"), 2), mark(g("peak_time_err"), 2))
        for i, m in enumerate(models):
            if not any((d, m, H, 2021) in recs for H in hs):
                continue
            lines.append(BS + "quad " + an.PRETTY.get(m, m) + " & " +
                         " & ".join(cols[H][0][i] + " & " + cols[H][1][i] + " & " + cols[H][2][i]
                                    for H in hs) + " " + NL)
        lines.append(BS + "midrule")
    if not lines:
        return
    lines = lines[:-1]
    cap = ("National system load forecasting (" + BS + "textsc{ms} setting, $L=336$). "
           "WAPE and peak height error in per cent; peak timing error in hours. Best in " +
           BS + "textbf{bold}.")
    write("tab_grids",
          head("table*", cap, "tab:grids", "l" + "ccc" * len(hs), tabcolsep="3.0pt") +
          ["Method & " + " & ".join(mc(3, "c", "$H=" + str(H) + "$" + BS + ",h") for H in hs) +
           " " + NL,
           cmid([(2 + 3 * i, 4 + 3 * i) for i in range(len(hs))]),
           " & " + " & ".join(["WAPE & Peak & $" + BS + "Delta t$"] * len(hs)) + " " + NL,
           BS + "midrule"] + lines + foot("table*"))


# ------------------------------------------------------------------ 4. day-ahead vs TSO
def tab_dayahead(recs=None):
    p = os.path.join(RES, "dayahead.json")
    if not os.path.exists(p):
        return
    da = json.load(open(p))
    grids = [g for g in GRID if g in da]
    if not grids:
        return
    order = ["TSO"] + [m for m in ["SeasonalNaive", "HistGBR", "DLinear", "TimeMixer",
                                   "iTransformer", "PatchTST", "CalTST", "PLANET", "PLANETX"]
                       if any(m in da[g] for g in grids)]
    lines = []
    for g in grids:
        c = da[g]
        pr = c.get("_protocol", {})
        w = mark([c[m]["wape"] if m in c else None for m in order], 2)
        pk = mark([c[m]["peak_mape"] if m in c else None for m in order], 2)
        dt = mark([c[m]["peak_time_err"] if m in c else None for m in order], 2)
        mae = mark([c[m]["mae"] if m in c else None for m in order], 0)
        lines.append(mc(6, "l", BS + "textit{" + an.DS_PRETTY.get(g, g) + "} (" +
                        str(pr.get("n_days", "?")) + " days, leads " +
                        str(pr.get("lead_hours", ["?", "?"])[0]) + "--" +
                        str(pr.get("lead_hours", ["?", "?"])[1]) + BS + ",h)") + " " + NL)
        for i, m in enumerate(order):
            if m not in c:
                continue
            lines.append(BS + "quad " + an.PRETTY.get(m, m) + " & " + w[i] + " & " + mae[i] +
                         " & " + pk[i] + " & " + dt[i] + " & " +
                         format(c[m]["bias"], "+.0f") + " " + NL)
        lines.append(BS + "midrule")
    lines = lines[:-1]
    cap = ("Day-ahead forecasts against the TSO's own published forecast. Every method is held "
           "to the same information cut-off: the last observed hour is 11:00" + BS + ",UTC on "
           "day $D-1$ and the scored block is 00:00--23:00" + BS + ",UTC on day $D$ "
           "(lead times 13--36" + BS + ",h).")
    write("tab_dayahead",
          head("table", cap, "tab:dayahead", "lrrrrr", tabcolsep="3.4pt") +
          ["Method & WAPE (" + BS + "%) & MAE (MW) & Peak (" + BS + "%) & $" + BS +
           "Delta t$ (h) & Bias (MW) " + NL, BS + "midrule"] + lines + foot("table"))


# ------------------------------------------------------------------ 5. ablations
def _abl(d, H, k, L=336, seed=2021):
    p = os.path.join(RES, "%s_PLANET_L%d_H%d_s%d_abl-%s.json" % (d, L, H, seed, k))
    return json.load(open(p)) if os.path.exists(p) else None


ABL_LABEL = {
    "no_pl": "w/o phase-locked tier (= backbone alone)",
    "no_resid": "w/o residual network (closed form alone)",
    "no_slow": "w/o climatological tier",
    "relative": "relative-time basis (phase not locked)",
    "uniform_w": "uniform least-squares weights",
    "no_beta_token": "w/o seasonal-state conditioning",
    "no_daytype": "w/o day-type interactions",
    "no_trend": "w/o local trend term",
}


def tab_ablation(recs):
    from run_all import ABLATIONS
    cells = [(d, H) for d in ["OPSD-DE", "ECL", "ETTh1"] for H in [96, 336]
             if (d, "PLANET", H, 2021) in recs and any(_abl(d, H, k)
                 for k in ABLATIONS if k != "full")]
    if not cells:
        return
    rows = []
    for k in [x for x in ABLATIONS if x != "full"]:
        vals = [(_abl(d, H, k) or {}).get("mse") for d, H in cells]
        if not any(v is not None for v in vals):
            continue
        dl = [100.0 * (v - recs[(d, "PLANET", H, 2021)]["mse"]) /
              recs[(d, "PLANET", H, 2021)]["mse"]
              for v, (d, H) in zip(vals, cells) if v is not None]
        rows.append(BS + "quad " + ABL_LABEL.get(k, k) + " & " +
                    " & ".join(cell(v) for v in vals) + " & " +
                    format(float(np.mean(dl)), "+.1f") + BS + "% " + NL)
    if not rows:
        return
    full = [recs[(d, "PLANET", H, 2021)]["mse"] for d, H in cells]
    dsets = []
    for d, H in cells:
        if not dsets or dsets[-1][0] != d:
            dsets.append([d, 1])
        else:
            dsets[-1][1] += 1
    write("tab_ablation",
          head("table*",
               "Component ablation (test MSE). The final column is the mean relative change "
               "over the cells shown; positive means removing the component makes the model "
               "worse.", "tab:ablation", "l" + "c" * len(cells) + "c", tabcolsep="4pt") +
          ["Variant & " + " & ".join(mc(n, "c", an.DS_PRETTY.get(d, d)) for d, n in dsets) +
           " & Mean " + NL,
           cmid([(2 + sum(n for _, n in dsets[:i]), 1 + sum(n for _, n in dsets[:i + 1]))
                 for i in range(len(dsets))]),
           " & " + " & ".join("$H{=}" + str(H) + "$" for _, H in cells) + " & $" + BS +
           "Delta$MSE " + NL, BS + "midrule",
           BS + "textbf{PLANET} (full) & " + " & ".join(cell(v) for v in full) + " & -- " + NL,
           BS + "midrule"] + rows + foot("table*"))


# ------------------------------------------------------------------ 6. significance
def tab_dm(recs, ref="PLANET"):
    datasets = ETT + ["ECL"] + GRID
    lines = []
    for m in [x for x in MODELS if x != ref]:
        w = t = l = 0
        stats = []
        for d in datasets:
            for H in (BENCH_H if d not in GRID else GRID_H):
                e1, e2 = an.per_window(d, ref, 336, H), an.per_window(d, m, 336, H)
                if e1 is None or e2 is None or len(e1) != len(e2):
                    continue
                s, p = an.dm_test(e1, e2, h=H)
                if not np.isfinite(p):
                    continue
                stats.append(s)
                if p < 0.05 and s < 0:
                    w += 1
                elif p < 0.05 and s > 0:
                    l += 1
                else:
                    t += 1
        if stats:
            lines.append(an.PRETTY.get(m, m) + " & " + str(w) + " & " + str(t) + " & " +
                         str(l) + " & " + format(float(np.median(stats)), "+.2f") + " " + NL)
    if not lines:
        return
    cap = ("Diebold--Mariano tests of PLANET against each baseline over all "
           "(dataset, horizon) cells, on per-window squared-error differentials with a "
           "Newey--West HAC variance and the Harvey--Leybourne--Newbold correction. "
           "A win is a cell where PLANET is significantly better at the 5" + BS + "% level.")
    write("tab_dm",
          head("table", cap, "tab:dm", "lcccr", tabcolsep="5pt") +
          ["Baseline & PLANET wins & ties & losses & median DM " + NL, BS + "midrule"] +
          lines + foot("table"))


# ------------------------------------------------------------------ 7. probabilistic
def tab_prob(recs=None):
    PROBD = os.path.join(RES, "prob")
    if not os.path.isdir(PROBD):
        return
    lines, cur = [], None
    for g in GRID:
        for m in ["DLinear", "PatchTST", "iTransformer", "PLANET"]:
            p = os.path.join(PROBD, "%s_%s_L336_H24_s2021.json" % (g, m))
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            if g != cur:
                lines.append(mc(8, "l", BS + "textit{" + an.DS_PRETTY.get(g, g) + "}") + " " + NL)
                cur = g
            lines.append(BS + "quad " + an.PRETTY.get(m, m) + " & " +
                         " & ".join([format(r["crps_cqr"], ".0f"), format(r["pinball_cqr"], ".0f"),
                                     format(r["picp_raw"], ".1f"), format(r["picp_cqr"], ".1f"),
                                     format(r["picp_aci"], ".1f"), format(r["mpiw_aci"], ".0f"),
                                     format(r["winkler_aci"], ".0f")]) + " " + NL)
    if not lines:
        return
    cap = ("Probabilistic day-ahead forecasting ($H=24$" + BS + ",h, nominal 90" + BS + "% "
           "intervals, MW). PICP is shown for the raw quantile head, after conformalised "
           "quantile regression, and after adaptive conformal inference; width and Winkler "
           "score are for the ACI intervals.")
    write("tab_prob",
          head("table", cap, "tab:prob", "lrrrrrrr", tabcolsep="3.2pt") +
          [" & & & " + mc(3, "c", "PICP (" + BS + "%)") + " & & " + NL,
           cmid([(4, 6)]),
           "Method & CRPS & Pinball & raw & CQR & ACI & MPIW & Winkler " + NL,
           BS + "midrule"] + lines + foot("table"))


# ------------------------------------------------------------------ 8. cost
def tab_cost(recs, d="OPSD-DE", H=96):
    models = [m for m in an.ORDER if (d, m, H, 2021) in recs]
    if not models:
        return
    lines = []
    for m in models:
        r = recs[(d, m, H, 2021)]
        lines.append(an.PRETTY.get(m, m) + " & " +
                     format(r["params"] / 1e3, ",.0f").replace(",", BS + ",") + " & " +
                     format(r["train_time_s"], ".0f") + " & " +
                     format(r["infer_time_s"], ".1f") + " & " +
                     format(r["mse"], ".4f") + " " + NL)
    cap = ("Computational cost on " + an.DS_PRETTY.get(d, d) + " ($H=" + str(H) + "$), one "
           "Quadro RTX 4000. Training time is to the early-stopping point.")
    write("tab_cost",
          head("table", cap, "tab:cost", "lrrrr", tabcolsep="5pt") +
          ["Method & Params (k) & Train (s) & Infer (s) & MSE " + NL, BS + "midrule"] +
          lines + foot("table"))


# ------------------------------------------------------------------ 9. weather pathway
def tab_exo(recs):
    lines = []
    for g in GRID:
        for H in [24, 96]:
            base = recs.get((g, "PLANET", H, 2021))
            if not base:
                continue
            ent = [("PLANET (no weather)", base)]
            for mode in ["persist", "noisy", "perfect"]:
                p = os.path.join(RES, "%s_PLANETX_L336_H%d_s2021_exo-%s.json" % (g, H, mode))
                if os.path.exists(p):
                    ent.append(("PLANET-X (" + BS + "textsc{" + mode + "})", json.load(open(p))))
            if len(ent) < 2:
                continue
            lines.append(mc(4, "l", BS + "textit{" + an.DS_PRETTY.get(g, g) + ", $H=" +
                            str(H) + "$" + BS + ",h}") + " " + NL)
            w = mark([e[1]["wape"] for e in ent], 2)
            for i, (nm, r) in enumerate(ent):
                lines.append(BS + "quad " + nm + " & " + w[i] + " & " +
                             format(r["mse"], ".4f") + " & " +
                             cell(r.get("peak_mape"), 2) + " " + NL)
            lines.append(BS + "midrule")
    if not lines:
        return
    lines = lines[:-1]
    cap = ("Effect of the weather pathway under three information regimes. " + BS +
           "textsc{perfect} uses the reanalysis and is an ex-post upper bound; " + BS +
           "textsc{noisy} degrades it to operational NWP skill; " + BS + "textsc{persist} "
           "uses yesterday's weather.")
    write("tab_exo",
          head("table", cap, "tab:exo", "lrrr", tabcolsep="5pt") +
          ["Method & WAPE (" + BS + "%) & MSE & Peak (" + BS + "%) " + NL, BS + "midrule"] +
          lines + foot("table"))


ALL = [tab_datasets, tab_benchmark, tab_benchfull, tab_grids, tab_dayahead,
       tab_ablation, tab_dm, tab_prob, tab_cost, tab_exo]

if __name__ == "__main__":
    recs = an.main_records()
    print("%d records" % len(recs))
    for fn in ALL:
        try:
            fn(recs)
        except Exception as e:
            import traceback
            print("  [skip] %s: %s" % (fn.__name__, e))
            traceback.print_exc(limit=1)
