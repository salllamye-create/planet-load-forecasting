"""Build every LaTeX table of the v2 paper from the record selection of paper_numbers.py.

    python src/make_tables_v2.py [--results results_v2] [--fm DIR] [--out paper/tables_v2]
        [--regime R2] [--ref PLANET] [--models PLANET,CalTST,PatchTST,NBEATS,DLinear,GBRT,TaoVanilla]
        [--mcs-B 2000] [--mcs-alpha 0.10] [--mcs-block horizon|cuberoot] [--no-mcs]
        [--holm-family dataset|cell|all] [--data data/proc_v2] [--cost-cell OPSD-DE:96]
        [--only tab_grids_v2,tab_cost_v2]

Tables written to paper/tables_v2/<name>.tex:
  tab_grids_v2      systems x horizons x models, WAPE mean (seed sd), MCS dagger, Holm-DM letters
  tab_regimes_v2    WAPE under R0 / R1 / R2 and the R2-vs-R0 change, headline models only
  tab_dayahead_v2   operator day-ahead forecast vs every learned model (results/dayahead.json)
  tab_ablation_v2   PLANET ablations per backbone: relative MSE change (+ tab_ablation_wape_v2)
  tab_bench_v2      M-setting benchmark (ECL, ETT): MSE / MAE averaged over horizons, mean rank
  tab_prob_v2       CRPS / pinball / PICP raw-CQR-ACI / MPIW / Winkler (results/prob/*.json)
  tab_cost_v2       parameters, training and inference time on one cell
  tab_datasets_v2   corpus (data/proc_v2/*.json + AUDIT.md)
  tab_fm_v2         zero-shot foundation models beside the best learned model, L=1024 block apart

Record selection (regime / size / L / lr on validation, seed averaging) is imported from
paper_numbers.py so the tables and the prose macros can never disagree.  v1 style: booktabs,
\\footnotesize, best bold, runner-up underlined, seed sd in small grey after the mean where two
or more seeds exist.  Every cell without a record prints --.  Tables are assembled by string
concatenation (LaTeX is backslash- and brace-dense).
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as an  # noqa: E402  (v1 PRETTY / DS_PRETTY / ORDER)
import paper_numbers as pn  # noqa: E402
import stats  # noqa: E402

HERE = pn.HERE
BS = chr(92)
NL = BS + BS
NAN = float("nan")

GRID8 = ["OPSD-DE", "OPSD-FR", "OPSD-GB", "OPSD-ES", "OPSD-IT", "OPSD-NL", "OPSD-PL", "OPSD-BE"]
GRID_H = [24, 96, 336]
BENCH_H = [96, 192, 336, 720]
DS_PRETTY = dict(an.DS_PRETTY, **{"OPSD-ES": "ES", "OPSD-IT": "IT", "OPSD-NL": "NL",
                                  "OPSD-PL": "PL", "OPSD-BE": "BE"})
PRETTY = dict(an.PRETTY, **{
    "GBRT": "GBRT", "TaoVanilla": "Hong (once)", "TaoRefit": "Hong (refit)", "TaoRefitAR": "Hong (refit)+AR",
    "TFT": "TFT", "TiDE": "TiDE", "TimeXer": "TimeXer",
    "NHiTS": "N-HiTS", "NHITS": "N-HiTS", "NBEATSx": "N-BEATSx", "xPatch": "xPatch", "TimesNet": "TimesNet",
    "Chronos2": "Chronos-2", "TimesFM2": "TimesFM-2.0", "Moirai2": "Moirai-2",
    "TSO": "Operator (published)", "TSO_adj": "Operator (bias-adjusted)",
    "Comb2": "Comb. (PLANET, TiDE)", "CombAll": "Comb. (all ten)",
    "PLANETB": BS + "textbf{PLANET-B}", "PLANETX": "PLANET-X"})
ORDER = an.ORDER + ["HistGBR", "GBRT", "TaoVanilla", "TaoRefit", "TaoRefitAR", "TFT", "TiDE", "TimeXer", "NHiTS", "NHITS", "NBEATSx",
                    "xPatch", "TimesNet", "PLANETB", "PLANETX", "Chronos2", "TimesFM2", "Moirai2"]
ABL_ORDER = ["no_pl", "no_resid", "relative", "utc", "utc_basis", "no_local", "no_daytype",
             "daytype_wk", "no_holiday", "no_beta_token", "basis_film", "uniform_w", "no_slow",
             "no_trend", "lean", "no_exo", "cal_patch"]
ABL_LABEL = {
    "no_pl": "w/o phase-locked tier (= backbone alone)",
    "no_resid": "w/o residual network (closed form alone)",
    "no_slow": "w/o climatological tier",
    "relative": "relative-time basis (phase not locked)",
    "uniform_w": "uniform least-squares weights",
    "no_beta_token": "w/o seasonal-state conditioning",
    "no_daytype": "w/o day-type interactions",
    "no_trend": "w/o local trend term",
    "no_holiday": "w/o holiday day-type",
    "daytype_wk": "weekend day-types only (w/o holiday day-type)",
    "basis_film": "FiLM instead of projected conditioning (basis backbone)",
    "lean": "lean composition (w/o climatological tier, trend, learned weights)",
    "utc": "basis on UTC instead of local civil time",
    "utc_basis": "basis on UTC instead of local civil time",
    "no_local": "basis on UTC instead of local civil time",
    "no_exo": "w/o shared exogenous head",
    "cal_patch": "calendar-aligned patching",
}
FIELDS = ["wape", "mse", "mae", "peak_mape", "peak_time_err", "params", "train_time_s", "infer_time_s"]

OUT = os.path.join(HERE, "paper", "tables_v2")
PRODUCED = []


# ------------------------------------------------------------------ LaTeX helpers
def tex(s):
    return str(s).replace("_", BS + "_").replace("%", BS + "%").replace("&", BS + "&")


def pretty(m):
    for suf, lab in (("-ens", " (seed ens.)"), ("-refit", " (monthly refit)")):
        if m.endswith(suf):
            return pretty(m[:-len(suf)]) + lab
    return PRETTY.get(m, tex(m))


def dsname(d):
    return DS_PRETTY.get(d, tex(d))


def order_models(models):
    def key(m):
        base = m
        for suf in ("-ens", "-refit"):
            if m.endswith(suf):
                base = m[:-len(suf)]
        return (ORDER.index(base) if base in ORDER else len(ORDER), base, m)
    return sorted(set(models), key=key)


def cell(v, nd=3):
    v = pn.fin(v)
    return "--" if not np.isfinite(v) else format(float(v), "." + str(nd) + "f")


def small(txt):
    return "{" + BS + "scriptsize" + BS + "color{gray}" + txt + "}"


def sd_txt(sd, n, nd):
    sd = pn.fin(sd)
    return small("(" + format(sd, "." + str(nd) + "f") + ")") if n >= 2 and np.isfinite(sd) else ""


def signed(v, nd=1):
    v = pn.fin(v)
    if not np.isfinite(v):
        return "--"
    if round(v, nd) == 0:
        v = 0.0
    return BS + "ensuremath{" + format(v, "+." + str(nd) + "f") + "}"


def mark(vals, nd=3, lower=True):
    """Best in bold, runner-up underlined among the finite entries (number only)."""
    a = np.array([pn.fin(v) for v in vals], float)
    out = [cell(v, nd) for v in vals]
    idx = np.flatnonzero(np.isfinite(a))
    if len(idx) == 0:
        return out
    key = a[idx] if lower else -a[idx]
    order = idx[np.argsort(key, kind="stable")]
    out[order[0]] = BS + "textbf{" + out[order[0]] + "}"
    if len(order) > 1:
        out[order[1]] = BS + "underline{" + out[order[1]] + "}"
    return out


def head(env, caption, label, colspec, tabcolsep="3pt", stretch=None):
    lines = [BS + "begin{" + env + "}[!t]", BS + "centering",
             BS + "caption{" + caption + "}", BS + "label{" + label + "}",
             BS + "setlength{" + BS + "tabcolsep}{" + tabcolsep + "}" + BS + "footnotesize"]
    if stretch:
        lines.append(BS + "renewcommand{" + BS + "arraystretch}{" + str(stretch) + "}")
    # the tabular sits in an adjustbox limited to the line width: a table a few points too wide
    # is scaled slightly instead of running into the margin (or the next column)
    return lines + [BS + "begin{adjustbox}{max width=" + BS + "linewidth}",
                    BS + "begin{tabular}{" + colspec + "}", BS + "toprule"]


def foot(env):
    return [BS + "bottomrule", BS + "end{tabular}", BS + "end{adjustbox}", BS + "end{" + env + "}"]


def mc(n, align, text):
    return BS + "multicolumn{" + str(n) + "}{" + align + "}{" + text + "}"


def cmid(pairs):
    return " ".join(BS + "cmidrule(lr){" + str(a) + "-" + str(b) + "}" for a, b in pairs)


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    PRODUCED.append(name)
    print("  [tab] " + name + ".tex")
    return True


# ------------------------------------------------------------------ context
def build_ctx(args):
    fm = args.fm if args.fm is not None else os.path.join(args.results, "fm")
    recs = pn.load_records(args.results, fm)
    for r in recs:
        if r["_regime"] is None and r["_src"] == "main":
            r["_regime"] = "v1"
    head_reg = pn.choose_regime(recs, args.regime)
    regimes = sorted({r["_regime"] for r in recs if r["_kind"] == "headline" and r["_regime"] is not None})
    ctx = dict(recs=recs, head=head_reg, regimes=regimes, cells={}, agg={}, args=args)
    for reg in regimes:
        cells = pn.cells_from(pn.select_headline(recs, reg))
        ctx["cells"][reg] = cells
        ctx["agg"][reg] = {k: {f: pn.seed_stats(rs, f) for f in FIELDS} for k, rs in cells.items()}
    print(f"{len(recs)} records; regimes {regimes}; headline {head_reg!r}; "
          + ", ".join(f"{r}: {len(c)} cells" for r, c in ctx["cells"].items()))
    return ctx


def stat(agg, key, field):
    return agg.get(key, {}).get(field, (NAN, NAN, 0))


def cell_errors(cells, d, H):
    e = {m: pn.seed_mean_err(rs) for (dd, m, HH), rs in cells.items() if dd == d and HH == H}
    e = {m: a for m, a in e.items() if a is not None}
    if len(e) < 2:
        return {}
    n = Counter(len(a) for a in e.values()).most_common(1)[0][0]
    return {m: a for m, a in e.items() if len(a) == n}


def grid_tests(ctx, systems, horizons):
    """MCS membership and Holm-adjusted DM letters, keyed (dataset, H, model).

    Letters: 'a' = the reference is significantly better than the model, 'b' = the model is
    significantly better than the reference (Holm-adjusted p < 0.05 within --holm-family)."""
    args, cells = ctx["args"], ctx["cells"][ctx["head"]]
    errs = {(d, H): cell_errors(cells, d, H) for d in systems for H in horizons}
    raw = []
    for (d, H), e in errs.items():
        if args.ref in e:
            for m, a in e.items():
                if m != args.ref:
                    s, p = stats.dm_test(e[args.ref], a, h=int(H))
                    raw.append((d, H, m, s, p))
    fam = defaultdict(list)
    for item in raw:
        d, H = item[0], item[1]
        fam[{"cell": (d, H), "dataset": d, "all": "all"}[args.holm_family]].append(item)
    letters = {}
    for items in fam.values():
        adj = stats.holm([p for *_, p in items])
        for (d, H, m, s, p), q in zip(items, adj):
            letters[(d, H, m)] = "a" if (q < 0.05 and s < 0) else "b" if (q < 0.05 and s > 0) else ""
    mcs_in = {}
    if not args.no_mcs:
        for (d, H), e in errs.items():
            if not e:
                continue
            names = sorted(e)
            L = np.stack([e[m] for m in names], axis=1)
            bl = max(1, int(round(L.shape[0] ** (1 / 3))))
            if args.mcs_block == "horizon":
                bl = max(bl, int(H))
            res = stats.mcs(L, alpha=args.mcs_alpha, B=args.mcs_B, block_len=bl, seed=0)
            for i, m in enumerate(names):
                mcs_in[(d, H, m)] = bool(res["in_set"][i])
    return mcs_in, letters


def systems_in(agg):
    # the multi-point weather copies (-mp) are headline-kind records of a variant dataset and
    # belong to the sensitivity analysis, not to the grid tables
    present = {k[0] for k in agg if k[0].startswith("OPSD") and not pn.is_variant(k[0])}
    return [d for d in GRID8 if d in present] + sorted(present - set(GRID8))


# ------------------------------------------------------------------ 1. grids
def tab_grids_v2(ctx):
    args, reg = ctx["args"], ctx["head"]
    agg = ctx["agg"][reg]
    systems = systems_in(agg)
    horizons = [H for H in GRID_H if any(k[0] in systems and k[2] == H for k in agg)]
    models = order_models(m for (d, m, H) in agg if d in systems)
    if not systems or not models:
        return False
    mcs_in, letters = grid_tests(ctx, systems, horizons)

    def block(hs):
        lines = []
        for H in hs:
            lines.append(mc(1 + len(systems), "l", BS + "textit{$H=" + str(H) + "$" + BS + ",h}") + " " + NL)
            cols = {d: mark([stat(agg, (d, m, H), "wape")[0] for m in models], 2) for d in systems}
            for i, m in enumerate(models):
                if not any((d, m, H) in agg for d in systems):
                    continue
                row = []
                for d in systems:
                    if (d, m, H) not in agg:
                        row.append("--")
                        continue
                    mean, sd, n = stat(agg, (d, m, H), "wape")
                    sup = (BS + "dagger{}" if mcs_in.get((d, H, m)) else "") + letters.get((d, H, m), "")
                    row.append(cols[d][i] + sd_txt(sd, n, 2) + ("$^{" + sup + "}$" if sup else ""))
                lines.append(pretty(m) + " & " + " & ".join(row) + " " + NL)
            lines.append(BS + "midrule")
        return lines[:-1]

    legend = ("WAPE (" + BS + "%) as the mean over seeds, seed standard deviation in grey "
              "(the seed count follows when it is not three). Best in " + BS + "textbf{bold}, runner-up "
              + BS + "underline{underlined}. $" + BS + "dagger$: member of the model confidence set ($"
              + BS + "alpha=" + format(args.mcs_alpha, ".2f") + "$, $T_{" + BS + "max}$, stationary bootstrap). "
              "Superscript a: " + pretty(args.ref) + " significantly better than the method; b: the "
              "method significantly better than " + pretty(args.ref) + " (Diebold--Mariano on "
              "per-window absolute errors, Holm-adjusted within each system, $p<0.05$).")
    cap_a = ("National system load forecasting, regime " + tex(reg) + ", one-day and four-day horizons "
             "(every model receives the same past load, calendar, holiday and NWP-skill weather "
             "covariates; lookback selected on validation). " + legend)
    cap_b = ("National system load forecasting, regime " + tex(reg) + ", two-week horizon (continuation of "
             "Table~" + BS + "ref{tab:grids-v2}; same protocol and notation).")
    hdr = ["Method & " + " & ".join(dsname(d) for d in systems) + " " + NL, BS + "midrule"]
    ok = write("tab_grids_v2", head("table*", cap_a, "tab:grids-v2", "l" + "c" * len(systems), tabcolsep="3pt")
               + hdr + block(horizons[:2]) + foot("table*"))
    if len(horizons) > 2:
        write("tab_grids_v2_b", head("table*", cap_b, "tab:grids-v2-b", "l" + "c" * len(systems), tabcolsep="3pt")
              + hdr + block(horizons[2:]) + foot("table*"))
    return ok


# ------------------------------------------------------------------ 2. information regimes
def tab_regimes_v2(ctx):
    args = ctx["args"]
    regs = [r for r in ("R0", "R1", "R2") if r in ctx["agg"]]
    if len(regs) < 2:
        return False
    allkeys = set().union(*(ctx["agg"][r].keys() for r in regs))
    systems = [d for d in GRID8 if any(k[0] == d for k in allkeys)]
    horizons = [H for H in GRID_H if any(k[2] == H and k[0] in systems for k in allkeys)]
    present = {m for (d, m, H) in allkeys if d in systems}
    models = [m for m in args.models if m in present] or order_models(present)
    if not systems or not models:
        return False
    lines = []
    for d in systems:
        if not any((d, m, H) in allkeys for m in models for H in horizons):
            continue
        cols = {}
        for H in horizons:
            for r in regs:
                cols[(H, r)] = mark([stat(ctx["agg"][r], (d, m, H), "wape")[0] for m in models], 2)
        first = True
        for i, m in enumerate(models):
            if not any((d, m, H) in allkeys for H in horizons):
                continue
            row = []
            for H in horizons:
                for r in regs:
                    row.append(cols[(H, r)][i])
                a, b = stat(ctx["agg"].get("R2", {}), (d, m, H), "wape")[0], \
                    stat(ctx["agg"].get("R0", {}), (d, m, H), "wape")[0]
                row.append(signed(pn.pct_change(a, b)))
            lab = (BS + "multirow{" + str(sum(1 for mm in models if any((d, mm, H) in allkeys for H in horizons)))
                   + "}{*}{" + dsname(d) + "}") if first else ""
            first = False
            lines.append(lab + " & " + pretty(m) + " & " + " & ".join(row) + " " + NL)
        lines.append(BS + "midrule")
    lines = lines[:-1]
    ncol = len(regs) + 1
    cap_full = ("Value of each information source. WAPE (" + BS + "%, seed mean) of the headline "
           "methods under regime R0 (past load only), R1 (+ calendar and holidays) and R2 "
           "(+ NWP-skill weather); $" + BS + "Delta$ is the relative change of WAPE from R0 to R2 "
           "(negative = weather and calendar help). Best per column in " + BS + "textbf{bold}. "
           "PLANET and PatchTST+Cal carry the calendar in their design matrix, so their R0 cells are "
           "not past-only: for them the R0--R1 step measures what the shared head adds beyond the "
           "built-in basis.")
    hdr1 = " & & " + " & ".join(mc(ncol, "c", "$H=" + str(H) + "$" + BS + ",h") for H in horizons) + " " + NL
    hdr2 = "System & Method & " + " & ".join(" & ".join(regs) + " & $" + BS + "Delta$" for _ in horizons) + " " + NL
    hdr = [hdr1, cmid([(3 + ncol * i, 2 + ncol * (i + 1)) for i in range(len(horizons))]), hdr2, BS + "midrule"]
    # split the system blocks (separated by midrules) into two tables of equal size
    blocks, cur = [], []
    for ln in lines:
        if ln == BS + "midrule":
            blocks.append(cur); cur = []
        else:
            cur.append(ln)
    blocks.append(cur)
    half = (len(blocks) + 1) // 2

    def joined(bl):
        out = []
        for b in bl:
            out += b + [BS + "midrule"]
        return out[:-1]

    cap_a = cap_full + " First four systems."
    cap_b = ("Value of each information source, continued (Table~" + BS + "ref{tab:regimes-v2}; same "
             "protocol and notation): remaining systems.")
    spec = "ll" + ("c" * ncol) * len(horizons)
    ok = write("tab_regimes_v2", head("table*", cap_a, "tab:regimes-v2", spec, tabcolsep="2.6pt")
               + hdr + joined(blocks[:half]) + foot("table*"))
    if len(blocks) > half:
        write("tab_regimes_v2_b", head("table*", cap_b, "tab:regimes-v2-b", spec, tabcolsep="2.6pt")
              + hdr + joined(blocks[half:]) + foot("table*"))
    return ok


# ------------------------------------------------------------------ 3. day-ahead vs TSO
def _avg_dicts(vs):
    keys = {k for v in vs for k, x in v.items() if isinstance(x, (int, float))}
    out = {}
    for k in keys:
        a = np.array([pn.fin(v.get(k)) for v in vs], float)
        a = a[np.isfinite(a)]
        out[k] = (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else NAN, int(len(a))) if len(a) else (NAN, NAN, 0)
    return out


def load_dayahead(path):
    da = json.load(open(path, encoding="utf-8"))
    out = {}
    for d, c in da.items():
        if not isinstance(c, dict):
            continue
        groups = defaultdict(list)
        for k, v in c.items():
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                groups[k] += [x for x in v if isinstance(x, dict)]
            elif isinstance(v, dict):
                if "model" in v:                       # v2: "<model>|L<L>|s<seed><suffix>"
                    mm = pn.V2_TAG_RE.search(str(v.get("suffix") or "").strip("_"))
                    if mm and mm.group(1) != REGIME_DA[0]:
                        continue                       # the table shows the headline regime
                    groups[v["model"]].append(v)
                else:
                    mm = re.match(r"(.+)_s(\d+)$", k)
                    groups[mm.group(1) if mm else k].append(v)
        out[d] = dict(protocol=c.get("_protocol", {}), models={m: _avg_dicts(vs) for m, vs in groups.items()})
    return out


REGIME_DA = ["R2"]


def tab_dayahead_v2(ctx):
    p = os.path.join(ctx["args"].results, "dayahead.json")
    if not os.path.exists(p):
        return False
    REGIME_DA[0] = ctx["head"]
    da = load_dayahead(p)
    p8 = os.path.join(ctx["args"].results, "dayahead_08.json")
    da8 = load_dayahead(p8) if os.path.exists(p8) else {}
    systems = [d for d in GRID8 if d in da] + sorted(d for d in da if d not in GRID8)
    if not systems:
        return False

    def dm_mark(entry):
        p_, s_ = entry.get("dm_p", (NAN,))[0], entry.get("dm_stat", (NAN,))[0]
        if not np.isfinite(pn.fin(p_)) or p_ >= 0.05:
            return ""
        return "$^{" + BS + ("dagger" if s_ < 0 else "ddagger") + "}$"

    def system_lines(d):
        c = da[d]["models"]
        c8 = da8.get(d, {}).get("models", {})
        if not c or not any(m != "TSO" for m in c):     # operator-only entries (e.g. the -mp corpora)
            return []
        pr = da[d]["protocol"]
        bases = order_models(m for m in c if not m.startswith("TSO") and not m.endswith("-ens"))
        order = ["TSO"] + bases
        g = lambda cc, m, k: cc.get(m, {}).get(k, (NAN, NAN, 0))  # noqa: E731
        w = mark([g(c, m, "wape")[0] for m in order], 2)
        ens_key = lambda m: m if (m == "TSO" or m.endswith("-refit")) else m + "-ens"  # noqa: E731
        wens = mark([g(c, ens_key(m), "wape")[0] for m in order], 2)
        wens8 = mark([g(c8, ens_key(m), "wape")[0] for m in order], 2)
        pk = mark([g(c, m, "peak_mape")[0] for m in order], 2)
        lh = pr.get("lead_hours", ["?", "?"])
        out = [mc(6, "l", BS + "textit{" + dsname(d) + "} (" + str(pr.get("n_days", "?")) +
                  " days, leads " + str(lh[0]) + "--" + str(lh[1]) + BS + ",h)") + " " + NL]
        for i, m in enumerate(order):
            ws = g(c, m, "wape")
            bias = g(c, m, "bias_pct")[0]
            refit = m.endswith("-refit")
            e11 = c.get(ens_key(m), {}); e8 = c8.get(ens_key(m), {})
            col_w = w[i] + ("" if refit else sd_txt(ws[1], ws[2], 2)) + (dm_mark(c.get(m, {})) if refit else "")
            if m == "TSO":                               # the operator's series under both cut-offs
                col_e, col_e8 = wens[i], (wens8[i] if e8 else "--")
            elif refit:
                col_e, col_e8 = "--", "--"
            else:
                col_e = wens[i] + dm_mark(e11)
                col_e8 = (wens8[i] + dm_mark(e8)) if e8 else "--"
            out.append(BS + "quad " + pretty(m) + " & " + col_w + " & " + col_e + " & " + col_e8 + " & " + pk[i] +
                       " & " + ("--" if not np.isfinite(pn.fin(bias)) else format(0.0 if abs(bias) < 0.05 else bias, "+.1f"))
                       + " " + NL)
        return out

    blocks = [system_lines(d) for d in systems]
    blocks = [b for b in blocks if b]
    if not blocks:
        return False
    PER = 3                       # system blocks per table: a block is about fifteen rows

    def joined(bl):
        out = []
        for b in bl:
            out += b + [BS + "midrule"]
        return out[:-1]

    cap = ("Day-ahead forecasts against each transmission operator's own published forecast "
           "(TSO row first). Every method is held to the same information cut-off: last observed "
           "hour 11:00" + BS + ",UTC on day $D-1$, scored block 00:00--23:00" + BS + ",UTC on "
           "day $D$; the 08:00 column repeats the protocol with the cut-off at 08:00" + BS + ",UTC "
           "(leads 16--39" + BS + ",h). WAPE: seed mean with the seed sd in grey; ens.: the mean of the "
           "three seed forecasts; monthly-refit rows retrain the model from scratch at the start of every "
           "test month (one seed); combination rows average the member models' forecasts, per seed and "
           "of the seed ensembles. $" + BS + "dagger$ / $" + BS + "ddagger$: the forecast is significantly "
           "better / worse than the operator (Diebold--Mariano on daily absolute errors, $p<0.05$). "
           "Peak: mean absolute error of the daily peak in " + BS + "%; bias in " + BS + "% of mean load. "
           "Best per system in " + BS + "textbf{bold}.")
    cap_b = ("Day-ahead forecasts against the operators, continued (Table~" + BS + "ref{tab:dayahead-v2}; "
             "same protocol and notation).")
    hdr = [" & & " + mc(2, "c", "WAPE ens. (" + BS + "%)") + " & & " + NL, cmid([(3, 4)]),
           "Method & WAPE (" + BS + "%) & 11:00 & 08:00 & Peak (" + BS + "%) & Bias (" + BS + "%) " + NL, BS + "midrule"]
    parts = [blocks[i:i + PER] for i in range(0, len(blocks), PER)]
    ok = write("tab_dayahead_v2", head("table", cap, "tab:dayahead-v2", "lrrrrr", tabcolsep="2.5pt")
               + hdr + joined(parts[0]) + foot("table"))
    for sfx, part in zip("bcdef", parts[1:]):
        write("tab_dayahead_v2_" + sfx, head("table", cap_b, "tab:dayahead-v2-" + sfx, "lrrrrr", tabcolsep="2.5pt")
              + hdr + joined(part) + foot("table"))
    return ok


# ------------------------------------------------------------------ 4. ablations
def _rel_changes(abl_recs, full_recs, field):
    """Per-seed relative change (%) of `field`, seed-paired with the full model where possible."""
    full_by_seed = {r["_seed"]: pn.fin(r.get(field)) for r in full_recs}
    full_mean = np.nanmean([v for v in full_by_seed.values()]) if full_by_seed else NAN
    out = []
    for r in abl_recs:
        base = full_by_seed.get(r["_seed"], full_mean)
        base = base if np.isfinite(pn.fin(base)) else full_mean
        out.append(pn.pct_change(r.get(field), base))
    out = np.array([v for v in out if np.isfinite(v)], float)
    if len(out) == 0:
        return NAN, NAN, 0
    hw = float(student_t.ppf(0.975, len(out) - 1) * out.std(ddof=1) / np.sqrt(len(out))) if len(out) > 1 else NAN
    return float(out.mean()), hw, int(len(out))


def _ablation_table(ctx, field, name, label, what):
    args, reg = ctx["args"], ctx["head"]
    cells = ctx["cells"][reg]
    abl = [r for r in ctx["recs"] if r["_kind"] == "abl" and r["_regime"] == reg and r["model"] == args.ref]
    if not abl:
        return False
    bb = lambda r: r.get("_backbone") or r.get("backbone") or "patch"  # noqa: E731
    groups = defaultdict(list)
    for r in abl:
        groups[(bb(r), r["_variant"], r["dataset"], r["H"])].append(r)
    backbones = sorted({k[0] for k in groups})
    cellkeys = sorted({(k[2], k[3]) for k in groups}, key=lambda x: (x[0] not in GRID8, x[0], x[1]))
    variants = [v for v in ABL_ORDER if any(k[1] == v for k in groups)] + \
        sorted({k[1] for k in groups if k[1] not in ABL_ORDER and k[1] != "full"})

    def full_recs(backbone, d, H):
        f = groups.get((backbone, "full", d, H))
        if f:
            return f
        rs = cells.get((d, args.ref, H), [])
        match = [r for r in rs if (r.get("_backbone") or r.get("backbone")) == backbone]
        return match or rs

    nd = 4 if field == "mse" else 2
    lines = []
    for backbone in backbones:
        if len(backbones) > 1:
            lines.append(mc(len(cellkeys) + 3, "l", BS + "textit{backbone: " + tex(backbone) + "}") + " " + NL)
        fulls = []
        for d, H in cellkeys:
            fr = full_recs(backbone, d, H)
            mean, sd, n = pn.seed_stats(fr, field) if fr else (NAN, NAN, 0)
            fulls.append(cell(mean, nd) + sd_txt(sd, n, nd))
        lines.append(pretty(args.ref) + " (full, " + what + ") & " + " & ".join(fulls) + " & -- & -- " + NL)
        for v in variants:
            row, means_mse, means_w = [], [], []
            for d, H in cellkeys:
                rs = groups.get((backbone, v, d, H))
                if not rs:
                    row.append("--")
                    continue
                fr = full_recs(backbone, d, H)
                m_, hw, n = _rel_changes(rs, fr, field)
                row.append(signed(m_) + (small("$" + BS + "pm$" + format(hw, ".1f")) if np.isfinite(hw) else ""))
                mm, _, _ = _rel_changes(rs, fr, "mse")
                mw, _, _ = _rel_changes(rs, fr, "wape")
                means_mse.append(mm)
                means_w.append(mw)
            if all(c == "--" for c in row):
                continue
            lines.append(BS + "quad " + ABL_LABEL.get(v, tex(v)) + " & " + " & ".join(row) + " & " +
                         signed(np.nanmean(means_mse) if means_mse else NAN) + " & " +
                         signed(np.nanmean(means_w) if means_w else NAN) + " " + NL)
        lines.append(BS + "midrule")
    lines = lines[:-1]
    dsets = []
    for d, H in cellkeys:
        if not dsets or dsets[-1][0] != d:
            dsets.append([d, 1])
        else:
            dsets[-1][1] += 1
    cap = ("Component ablation of " + pretty(args.ref) + " (regime " + tex(reg) + "). Cells are the "
           "relative change of test " + what + " when the component is removed, mean over seeds with "
           "a 95" + BS + "% $t$-interval half-width from the seeds in grey (positive = removal hurts); "
           "the full model row gives the absolute value. The last two columns average the relative "
           "MSE and WAPE change over the cells shown.")
    return write(name,
                 head("table*", cap, label, "l" + "c" * len(cellkeys) + "cc", tabcolsep="2pt") +
                 ["Variant & " + " & ".join(mc(n, "c", dsname(d)) for d, n in dsets) +
                  " & " + mc(2, "c", "Mean") + " " + NL,
                  cmid([(2 + sum(n for _, n in dsets[:i]), 1 + sum(n for _, n in dsets[:i + 1]))
                        for i in range(len(dsets))] + [(len(cellkeys) + 2, len(cellkeys) + 3)]),
                  " & " + " & ".join("$H{=}" + str(H) + "$" for _, H in cellkeys) +
                  " & $" + BS + "Delta$MSE & $" + BS + "Delta$WAPE " + NL, BS + "midrule"] +
                 lines + foot("table*"))


def tab_ablation_v2(ctx):
    ok = _ablation_table(ctx, "mse", "tab_ablation_v2", "tab:ablation-v2", "MSE")
    if ok:
        _ablation_table(ctx, "wape", "tab_ablation_wape_v2", "tab:ablation-wape-v2", "WAPE")
    return ok


# ------------------------------------------------------------------ 5. M-setting benchmark
def tab_bench_v2(ctx):
    args, reg = ctx["args"], ctx["head"]
    # the two benchmark panels carry no weather and are run under R1 only: take the regime
    # that actually holds their records (R1 in the v2 pipeline), not the headline regime
    is_bench = lambda d: d == "ECL" or d.startswith("ETT")
    reg = next((r for r in ("R1", reg, "R0", "R2") if r in ctx["agg"]
                and any(is_bench(k[0]) for k in ctx["agg"][r])), reg)
    agg, cells = ctx["agg"][reg], ctx["cells"][reg]
    datasets = [d for d in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL"] if any(k[0] == d for k in agg)]
    models = order_models(m for (d, m, H) in agg if d in datasets)
    if not datasets or not models:
        return False
    hs = {d: sorted({k[2] for k in agg if k[0] == d}) for d in datasets}

    def avg(d, m, field):
        """Horizon average of the seed means; sd of the per-seed horizon average over common seeds."""
        if not all((d, m, H) in agg for H in hs[d]):
            return NAN, NAN, 0
        mean = float(np.mean([stat(agg, (d, m, H), field)[0] for H in hs[d]]))
        seeds = set.intersection(*({r["_seed"] for r in cells[(d, m, H)]} for H in hs[d]))
        if len(seeds) >= 2:
            per = [np.mean([pn.fin(next(r.get(field) for r in cells[(d, m, H)] if r["_seed"] == s))
                            for H in hs[d]]) for s in sorted(seeds)]
            return mean, float(np.std(per, ddof=1)), len(seeds)
        return mean, NAN, len(seeds)

    stats_ = {(d, m): (avg(d, m, "mse"), avg(d, m, "mae")) for d in datasets for m in models}
    cols = {d: (mark([stats_[(d, m)][0][0] for m in models], 3),
                mark([stats_[(d, m)][1][0] for m in models], 3)) for d in datasets}
    ranks = defaultdict(list)
    for d in datasets:
        for H in hs[d]:
            avail = [m for m in models if (d, m, H) in agg]
            if len(avail) < 3:
                continue
            vals = np.array([stat(agg, (d, m, H), "mse")[0] for m in avail])
            for m, r in zip(avail, rankdata(vals, method="average")):
                ranks[m].append(float(r))
    rk = mark([np.mean(ranks[m]) if ranks[m] else NAN for m in models], 2)
    body = []
    for i, m in enumerate(models):
        cs = []
        for d in datasets:
            (_, sdm, n), (_, sda, _) = stats_[(d, m)]
            cs.append(cols[d][0][i] + sd_txt(sdm, n, 3) + " & " + cols[d][1][i] + sd_txt(sda, n, 3))
        body.append(pretty(m) + " & " + " & ".join(cs) + " & " + rk[i] + " " + NL)
    cap = ("Long-horizon benchmark in the multivariate (" + BS + "textsc{m}) setting, averaged over "
           "the evaluated horizons (z-scored data, protocol lookback 336, seed mean with "
           "seed sd in grey). Best in " + BS + "textbf{bold}, second " + BS + "underline{underlined}; "
           "the last column is the mean MSE rank over every (dataset, horizon) cell.")
    return write("tab_bench_v2",
                 head("table*", cap, "tab:bench-v2", "l" + "cc" * len(datasets) + "c", tabcolsep="3pt") +
                 [" & " + " & ".join(mc(2, "c", dsname(d)) for d in datasets) + " & Avg. " + NL,
                  cmid([(2 + 2 * i, 3 + 2 * i) for i in range(len(datasets))]),
                  "Method & " + " & ".join(["MSE & MAE"] * len(datasets)) + " & rank " + NL,
                  BS + "midrule"] + body + foot("table*"))


# ------------------------------------------------------------------ 6. probabilistic
def tab_prob_v2(ctx):
    files = sorted(glob.glob(os.path.join(ctx["args"].results, "prob", "*.json")))
    groups = defaultdict(list)
    for p in files:
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if {"dataset", "model", "H"} <= set(r):
            groups[(r["dataset"], r["model"], r["H"])].append(r)
    if not groups:
        return False
    blocks = sorted({(d, H) for d, _, H in groups}, key=lambda x: (x[0] not in GRID8, GRID8.index(x[0]) if x[0] in GRID8 else x[0], x[1]))
    COLS = [("crps_cqr", 0), ("pinball_cqr", 0), ("picp_raw", 1), ("picp_cqr", 1), ("picp_aci", 1),
            ("mpiw_aci", 0), ("winkler_aci", 0)]
    lines = []
    for d, H in blocks:
        models = order_models(m for (dd, m, HH) in groups if dd == d and HH == H)
        st = {m: _avg_dicts(groups[(d, m, H)]) for m in models}
        marked = {}
        for k, nd in COLS:
            vals = [st[m].get(k, (NAN,))[0] for m in models]
            marked[k] = mark(vals, nd) if not k.startswith("picp") else [cell(v, nd) for v in vals]
        lines.append(mc(8, "l", BS + "textit{" + dsname(d) + ", $H=" + str(H) + "$" + BS + ",h}") + " " + NL)
        for i, m in enumerate(models):
            n = st[m].get("crps_cqr", (NAN, NAN, 0))[2]
            lines.append(BS + "quad " + pretty(m) + " & " + " & ".join(marked[k][i] for k, _ in COLS) +
                         (" " if n < 2 else "") + " " + NL)
        lines.append(BS + "midrule")
    lines = lines[:-1]
    cap = ("Day-ahead probabilistic forecasting (nominal 90" + BS + "% intervals, MW, seed mean). "
           "qCRPS (the quantile-weighted CRPS approximation over the seven levels) and pinball loss are "
           "those of the quantiles after conformalised quantile regression of every level pair; PICP is "
           "shown for the raw head, after CQR and after delayed-feedback adaptive conformal inference "
           "(the level of each step is updated only with outcomes observed by the forecast origin); "
           "width and Winkler score are for the ACI intervals. Best per block in " +
           BS + "textbf{bold} (PICP is not ranked).")
    cap_b = ("Day-ahead probabilistic forecasting, continued (Table~" + BS + "ref{tab:prob-v2}; same "
             "protocol and notation).")
    hdr = [" & & & " + mc(3, "c", "PICP (" + BS + "%)") + " & & " + NL, cmid([(4, 6)]),
           "Method & qCRPS & Pinball & raw & CQR & ACI & MPIW & Winkler " + NL, BS + "midrule"]
    blocks_, cur = [], []
    for ln in lines:
        if ln == BS + "midrule":
            blocks_.append(cur); cur = []
        else:
            cur.append(ln)
    blocks_.append(cur)
    half = (len(blocks_) + 1) // 2

    def joined(bl):
        out = []
        for b in bl:
            out += b + [BS + "midrule"]
        return out[:-1]

    ok = write("tab_prob_v2", head("table", cap, "tab:prob-v2", "lrrrrrrr", tabcolsep="3pt")
               + hdr + joined(blocks_[:half]) + foot("table"))
    if len(blocks_) > half:
        write("tab_prob_v2_b", head("table", cap_b, "tab:prob-v2-b", "lrrrrrrr", tabcolsep="3pt")
              + hdr + joined(blocks_[half:]) + foot("table"))
    return ok


# ------------------------------------------------------------------ 7. cost
def tab_cost_v2(ctx):
    args, reg = ctx["args"], ctx["head"]
    agg = ctx["agg"][reg]
    d, H = args.cost_cell.split(":")
    H = int(H)
    models = order_models(m for (dd, m, HH) in agg if dd == d and HH == H)
    if not models:
        return False
    g = lambda k: [stat(agg, (d, m, H), k) for m in models]  # noqa: E731
    params, tr, inf, w, mse = g("params"), g("train_time_s"), g("infer_time_s"), g("wape"), g("mse")
    wm, mm = mark([x[0] for x in w], 2), mark([x[0] for x in mse], 4)
    lines = []
    for i, m in enumerate(models):
        pk = params[i][0]
        lines.append(pretty(m) + " & " +
                     ("--" if not np.isfinite(pn.fin(pk)) else format(pk / 1e3, ",.0f").replace(",", "{,}")) + " & " +
                     cell(tr[i][0], 0) + sd_txt(tr[i][1], tr[i][2], 0) + " & " +
                     cell(inf[i][0], 1) + sd_txt(inf[i][1], inf[i][2], 1) + " & " +
                     wm[i] + sd_txt(w[i][1], w[i][2], 2) + " & " + mm[i] + " " + NL)
    cap = ("Computational cost on " + dsname(d) + " ($H=" + str(H) + "$), one Quadro RTX 4000, seed "
           "mean with seed sd in grey. Training time is to the early-stopping point; zero-shot "
           "models report inference only.")
    return write("tab_cost_v2",
                 head("table", cap, "tab:cost-v2", "lrrrrr", tabcolsep="4pt") +
                 ["Method & Params (k) & Train (s) & Infer (s) & WAPE (" + BS + "%) & MSE " + NL,
                  BS + "midrule"] + lines + foot("table"))


# ------------------------------------------------------------------ 8. datasets
def parse_audit(path):
    rows, header, keep = {}, None, False
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        if not line.startswith("|"):
            header = None
            continue
        cs = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header, keep = cs, any("holidays" in h for h in cs)
            continue
        if all(set(c) <= set("-: ") for c in cs) or not keep or len(cs) != len(header):
            continue
        rows[cs[0]] = dict(zip(header, cs))
    return rows


def tab_datasets_v2(ctx):
    ddir = ctx["args"].data
    audit = parse_audit(os.path.join(ddir, "AUDIT.md"))
    hol_key = next((k for k in next(iter(audit.values()), {}) if "holidays/yr" in k), None)
    tso_key = next((k for k in next(iter(audit.values()), {}) if "TSO" in k), None)
    rows = []
    for d in GRID8 + ["ECL", "ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        p = os.path.join(ddir, d + ".json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p, encoding="utf-8"))
        tr, va, te = (m["splits"][k] for k in ("train", "val", "test"))
        a = audit.get(d, {})
        hol = re.match(r"[\d.]+", a.get(hol_key, "")) if hol_key else None
        hol = hol.group(0) if hol else None
        tso = a.get(tso_key, "") if tso_key else ""
        tso = tso if re.match(r"[\d.]+%", tso) else None
        if hol is None or (tso is None and d.startswith("OPSD")):
            try:                                          # fall back to the arrays themselves
                z = np.load(os.path.join(ddir, d + ".npz"))
                if hol is None and "cal_flags" in z.files:
                    days = np.unique(z["local_stamps"][z["cal_flags"][:, 0] > 0] // 86400)
                    years = (z["stamps"][-1] - z["stamps"][0]) / (365.25 * 86400)
                    hol = format(len(days) / years, ".1f") if years > 0 else None
                if tso is None and "aux" in z.files and d.startswith("OPSD"):
                    tso = format(100.0 * np.isfinite(z["aux"][:, 0]).mean(), ".1f") + "%"
            except Exception:
                pass
        rows.append(" & ".join([
            dsname(d), tex(m.get("freq", "--")), str(m.get("C", "--")),
            format(int(m["T"]), ",").replace(",", "{,}"),
            str(m.get("start", ""))[:7] + "--" + str(m.get("end", ""))[:7],
            tex(m.get("tz", "--")), hol or "--", tex(tso) if tso else "--",
            "%d/%d/%d" % (tr[1] - tr[0], va[1] - va[0], te[1] - te[0]),
            "MS" if d.startswith("OPSD") else "M"]) + " " + NL)
    if not rows:
        return False
    cap = ("Benchmark corpus v2. $T$ rows at the stated frequency, $C$ channels (target last); "
           "holidays per year are distinct national public-holiday dates in the span; TSO cov.\\ is "
           "the coverage of the operator's published day-ahead forecast used in Table~" + BS +
           "ref{tab:dayahead-v2}. " + BS + "textsc{ms} scores the load channel from all inputs; " +
           BS + "textsc{m} scores every channel. Splits are chronological.")
    return write("tab_datasets_v2",
                 head("table*", cap, "tab:datasets-v2", "lccrclccccc".replace("lccrclccccc", "lccrclcccc"), tabcolsep="3.5pt") +
                 ["Dataset & Freq. & $C$ & $T$ & Span & Zone & Hol./yr & TSO cov. & Train/Val/Test & Setting " + NL,
                  BS + "midrule"] + rows + foot("table*"))


# ------------------------------------------------------------------ 9. foundation models
def tab_fm_v2(ctx):
    reg = ctx["head"]
    agg = ctx["agg"][reg]
    fm = [r for r in ctx["recs"] if r["_src"] == "fm" or r.get("zero_shot")]
    if not fm:
        return False
    systems = [d for d in GRID8 if any(r["dataset"] == d for r in fm)]
    horizons = [H for H in GRID_H if any(r["H"] == H for r in fm)]
    groups = defaultdict(list)
    for r in fm:
        groups[(r["model"], r["_regime"], r["L"], r["dataset"], r["H"])].append(r)
    keys = sorted({(m, rg, L) for (m, rg, L, d, H) in groups},
                  key=lambda k: (k[2] == 1024, ORDER.index(k[0]) if k[0] in ORDER else 99, k[0], str(k[1])))
    multi_reg = {m: len({rg for (mm, rg, L) in keys if mm == m}) > 1 for m in {k[0] for k in keys}}

    def label(m, rg, L):
        s = pretty(m) + (" (" + tex(rg) + ")" if multi_reg[m] and rg else "")
        return s + (" ($L{=}1024$)" if L == 1024 else "")

    fm_models = {k[0] for k in keys}
    lines = []
    for H in horizons:
        lines.append(mc(1 + len(systems), "l", BS + "textit{$H=" + str(H) + "$" + BS + ",h}") + " " + NL)
        std = [k for k in keys if k[2] != 1024]
        long = [k for k in keys if k[2] == 1024]
        best = {}
        for d in systems:
            cand = {m: stat(agg, (d, m, H), "wape")[0] for (dd, m, HH) in agg
                    if dd == d and HH == H and m not in fm_models}
            cand = {m: v for m, v in cand.items() if np.isfinite(v)}
            best[d] = min(cand, key=cand.get) if cand else None
        for block in (std, long):
            if not block:
                continue
            vals = {}
            for k in block:
                vals[k] = [pn.seed_stats(groups[(k[0], k[1], k[2], d, H)], "wape")[0]
                           if (k[0], k[1], k[2], d, H) in groups else NAN for d in systems]
            vals["best"] = [stat(agg, (d, best[d], H), "wape")[0] if best[d] else NAN for d in systems]
            rowsk = list(block) + ["best"]
            marked = [mark([vals[k][j] for k in rowsk], 2) for j in range(len(systems))]
            for i, k in enumerate(rowsk):
                if k == "best":
                    if block is long:
                        continue
                    row = [marked[j][i] + (small(pretty(best[d]).replace(BS + "textbf", "")) if best[d] else "")
                           for j, d in enumerate(systems)]
                    lines.append("Best learned model & " + " & ".join(row) + " " + NL)
                else:
                    lines.append(label(*k) + " & " + " & ".join(marked[j][i] for j in range(len(systems))) + " " + NL)
            if block is std and long:
                lines.append(BS + "cmidrule(lr){1-" + str(1 + len(systems)) + "}")
        lines.append(BS + "midrule")
    lines = lines[:-1]
    cap = ("Zero-shot foundation models (inference only, no fine-tuning) beside the best learned "
           "model of each cell (regime " + tex(reg) + "; the learned model's name in grey). WAPE (" +
           BS + "%). Rows at $L{=}1024$ use a longer context than the protocol lookback and form a "
           "separate block. Training-corpus contamination is discussed in the text.")
    return write("tab_fm_v2",
                 head("table*", cap, "tab:fm-v2", "l" + "c" * len(systems), tabcolsep="3.5pt") +
                 ["Method & " + " & ".join(dsname(d) for d in systems) + " " + NL, BS + "midrule"] +
                 lines + foot("table*"))


# ------------------------------------------------------------------ main
def tab_config_v2(ctx):
    """Appendix: the configuration selected on validation for every neural model -- width multiplier,
    residual backbone, learning rate per system, and how often the selected learning rate and lookback
    sit at an edge of their grids (a signal that the grid may be too narrow)."""
    reg = ctx["head"]
    cells = ctx["cells"].get(reg, {})
    systems = [d for d in GRID8 if any(k[0] == d for k in cells)]
    if not systems:
        return False
    LR_EDGE, L_EDGE = {1e-4, 1e-2}, {168, 720}
    rows = {}
    for (d, m, H), rs in cells.items():
        if d not in systems or not rs:
            continue
        r0 = rs[0]
        if m == "SeasonalNaive" or not r0.get("lr") or r0.get("size") is None or not r0.get("params"):
            continue
        row = rows.setdefault(m, dict(size=r0.get("size"), bb=r0.get("backbone"), lr={}, L={}))
        row["lr"][d] = float(r0["lr"])
        row["L"][(d, H)] = int(r0.get("L") or 336)
    if not rows:
        return False
    models = order_models(rows)

    def lr_txt(v):
        if v is None or not np.isfinite(v):
            return "--"
        e = int(np.floor(np.log10(v) + 1e-9)); mant = v / 10 ** e
        return ("$10^{%d}$" % e) if abs(mant - 1) < 1e-6 else ("$%g" % mant) + BS + "times10^{%d}$" % e

    lines = []
    for m in models:
        row = rows[m]
        n_lr = sum(1 for v in row["lr"].values() if any(abs(v - e) < 1e-12 for e in LR_EDGE))
        n_L = sum(1 for v in row["L"].values() if v in L_EDGE)
        size = row["size"]
        size_txt = "--" if m in pn_nosize() else ("$%g" % size + BS + "times$")
        bb = (row["bb"] or "") if m in ("PLANET", "PLANETX") else "--"
        lines.append(pretty(m) + " & " + size_txt + " & " + tex(bb) + " & "
                     + " & ".join(lr_txt(row["lr"].get(d)) for d in systems)
                     + " & %d/%d & %d/%d " % (n_lr, len(row["lr"]), n_L, len(row["L"])) + NL)
    cap = ("Configurations selected on validation under the protocol, regime " + tex(reg) + ": width "
           "multiplier relative to the published size (chosen on DE and FR), residual backbone (PLANET "
           "variants), learning rate per system, and the number of systems whose selected learning rate "
           "sits at an edge of the grid $" + BS + "{10^{-4},10^{-2}" + BS + "}$ and of (system, horizon) "
           "cells whose selected lookback sits at an edge of $" + BS + "{168,720" + BS + "}$. The lookbacks "
           "themselves are in the released result files.")
    hdr = ["Method & Width & Backbone & " + " & ".join(dsname(d) for d in systems) + " & lr edge & $L$ edge " + NL,
           BS + "midrule"]
    return write("tab_config_v2", head("table*", cap, "tab:config-v2", "lcc" + "c" * len(systems) + "cc", tabcolsep="3pt")
                 + hdr + lines + foot("table*"))


def tab_weather_v2(ctx):
    """Per-system weather dependence (training split) beside the standing of the proposed model
    against the leading model and against the learned-calendar control, mean over the three grid
    horizons, and the tier's measured value where the ablation ran."""
    reg, args = ctx["head"], ctx["args"]
    agg = ctx["agg"][reg]
    systems = [d for d in GRID8 if any(k[0] == d for k in agg)]
    if not systems:
        return False
    rows = []
    for d in systems:
        wx = pn.weather_dependence(args.data, d)
        if not wx:
            continue
        def gap(m_ref, m_other):
            v = [pn.pct_change(stat(agg, (d, m_ref, H), "wape")[0], stat(agg, (d, m_other, H), "wape")[0]) for H in GRID_H]
            v = [x for x in v if np.isfinite(x)]
            return float(np.mean(v)) if v else NAN
        rows.append((d, wx["r2"], wx["winter_summer"], gap("PLANET", "TiDE"), gap("PLANET", "CalTST")))
    rows.sort(key=lambda r: -r[1])
    lines = []
    for d, r2, ws, g_t, g_c in rows:
        lines.append(dsname(d) + " & " + cell(r2, 2) + " & " + cell(ws, 2) + " & " + signed(g_t) + " & " + signed(g_c) + " " + NL)
    cap = ("Weather dependence of each system and the standing of the proposed model. $R^2$: share of the "
           "variance of weekday daily mean load explained by a quadratic in daily mean temperature on the "
           "training split; W/S: ratio of winter (DJF) to summer (JJA) monthly mean load; $" + BS +
           "Delta$: relative difference of PLANET's WAPE to TiDE's and to PatchTST+Cal's under regime " + tex(reg) +
           ", mean over the three horizons (negative: PLANET lower). Systems are ordered by weather dependence.")
    return write("tab_weather_v2", head("table", cap, "tab:weather-v2", "lcccc", tabcolsep="4pt")
                 + ["System & $R^2$ & W/S & $" + BS + "Delta$ vs TiDE (" + BS + "%) & $" + BS +
                    "Delta$ vs PatchTST+Cal (" + BS + "%) " + NL, BS + "midrule"] + lines + foot("table"))


def pn_nosize():
    try:
        import exp_v2
        return set(exp_v2.NO_SIZE)
    except Exception:
        return set()


ALL = [tab_grids_v2, tab_regimes_v2, tab_dayahead_v2, tab_ablation_v2, tab_bench_v2, tab_prob_v2,
       tab_cost_v2, tab_datasets_v2, tab_fm_v2, tab_config_v2, tab_weather_v2]


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "results_v2"))
    ap.add_argument("--fm", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--regime", default="R2")
    ap.add_argument("--ref", default="PLANET")
    ap.add_argument("--models", default="PLANET,PLANETB,CalTST,PatchTST,NBEATS,DLinear,TFT,TiDE,NHITS,TimeXer,GBRT,HistGBR,TaoVanilla",
                    help="headline models of tab_regimes_v2 (comma-separated)")
    ap.add_argument("--holm-family", default="dataset", choices=["cell", "dataset", "all"])
    ap.add_argument("--mcs-B", type=int, default=2000)
    ap.add_argument("--mcs-alpha", type=float, default=0.10)
    ap.add_argument("--mcs-block", default="horizon", choices=["cuberoot", "horizon"])
    ap.add_argument("--no-mcs", action="store_true")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "proc_v2"))
    ap.add_argument("--cost-cell", default="OPSD-DE:96")
    ap.add_argument("--only", default=None, help="comma-separated table names")
    args = ap.parse_args(argv)
    args.results = os.path.abspath(args.results)
    args.models = [m for m in args.models.split(",") if m]
    OUT = os.path.abspath(args.out)
    ctx = build_ctx(args)
    only = set(args.only.split(",")) if args.only else None
    for fn in ALL:
        if only and fn.__name__ not in only:
            continue
        try:
            if not fn(ctx):
                print("  [none] " + fn.__name__ + ": no records")
        except Exception as e:
            import traceback
            print("  [skip] %s: %s" % (fn.__name__, e))
            traceback.print_exc(limit=2)
    print("produced: " + (", ".join(PRODUCED) if PRODUCED else "(nothing)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
