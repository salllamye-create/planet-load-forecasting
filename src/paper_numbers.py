"""Single source of truth for every number quoted in the paper.

    python src/paper_numbers.py [--results results_v2] [--fm results_v2/fm] [--out paper/numbers.tex]
                          [--regime R2] [--ref PLANET] [--mcs-B 2000] [--no-mcs]
                          [--holm-family dataset|cell|all] [--extra a.json b.json ...]
    python src/paper_numbers.py --check [--paper paper]

Generate mode scans every result record (``<results>/*.json`` and ``<fm>/*.json``), selects the
headline records per V2_SPEC.md (regime and size from the suffix or record fields; every
remaining choice -- lookback L, size, learning rate -- made on ``val_mse`` only, exactly as v1
``analysis.main_records`` did for the Informer/Autoformer L=96 runs), averages over seeds, and
writes ``paper/numbers.tex`` with one ``\\csname n@KEY\\endcsname`` macro per number, used in
the prose as ``\\nv{KEY}``.  The KEY scheme is documented at the top of the generated file.

Check mode scans ``paper/sec/*.tex`` and ``paper/main.tex`` for result-looking numeric literals
outside ``\\nv{...}`` (decimal numbers and percentages), lists file:line for each, and exits
with status 1 if any are found, so "no hand-typed result in the prose" can be enforced.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats  # noqa: E402

# prose names for macros whose value is a model name (the tables have their own map)
PRETTY_TXT = {"NBEATS": "N-BEATS", "NHITS": "N-HiTS", "NBEATSx": "N-BEATSx", "CalTST": "PatchTST+Cal",
              "PLANETX": "PLANET-X", "HistGBR": "GBRT", "TaoVanilla": "Hong (once)",
              "TaoRefit": "Hong (refit)", "TaoRefitAR": "Hong (refit)+AR", "Chronos2": "Chronos-2",
              "TimesFM2": "TimesFM-2.0", "Moirai2": "Moirai-2", "SeasonalNaive": "Seasonal-Na\\\"ive"}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# metric key in the macro name -> (record field or callable, decimals)
METRICS = OrderedDict([
    ("wape", ("wape", 2)), ("mse", ("mse", 4)), ("mae", ("mae", 3)),
    ("peak", ("peak_mape", 2)), ("dt", ("peak_time_err", 2)), ("ramp", ("ramp_mae", 0)),
    ("params_k", (lambda r: r["params"] / 1e3 if r.get("params") is not None else None, 1)),
    ("train_s", ("train_time_s", 0)), ("infer_s", ("infer_time_s", 1)),
    ("mae_mw", ("mae_orig", 0)), ("rmse_mw", ("rmse_orig", 0)), ("mape", ("mape", 2)),
    ("val_mse", ("val_mse", 4)), ("epochs", ("epochs_run", 0)),
])
# day-ahead / TSO json fields -> (macro key, decimals)
DA_FIELDS = OrderedDict([("wape", ("wape", 2)), ("mae", ("mae", 0)), ("rmse", ("rmse", 0)),
                         ("mape", ("mape", 2)), ("bias", ("bias", 0)), ("ramp_mae", ("ramp", 0)),
                         ("peak_mae", ("peak_mae", 0)), ("peak_mape", ("peak", 2)),
                         ("peak_time_err", ("dt", 2)), ("n_days", ("n_days", "int")),
                         ("bias_pct", ("bias_pct", "pct")),
                         ("dm_stat", ("dm_stat", 2)), ("dm_p", ("dm_p", "p")),
                         ("wape_lockdown", ("wape_lock", 2)), ("wape_excl_lockdown", ("wape_xlock", 2)),
                         ("n_days_lockdown", ("n_days_lock", "int")),
                         ("wape_holiday", ("wape_hol", 2)), ("wape_nonholiday", ("wape_nonhol", 2)),
                         ("n_days_holiday", ("n_days_hol", "int")), ("n_members", ("n_members", "int"))])


def _daf(v, nd):
    """Format a day-ahead field: signed percentage, integer, or fixed decimals."""
    if nd == "pct":
        return fpct(v)
    if nd == "int":
        return fint(v)
    if nd == "p":
        return fp(v)
    return fnum(v, nd)

V2_TAG_RE = re.compile(r"v2-(R\d)(?:-(.*))?$")
SIZES = {"small", "published", "pub", "large", "base", "xs", "s", "m", "l", "xl"}


# ------------------------------------------------------------------ formatting
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def fnum(v, nd):
    v = fin(v)
    if not np.isfinite(v):
        return "--"
    if nd == "int":
        return fint(v)
    s = f"{v:.{nd}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    if s.startswith("-"):                      # a true minus sign in text as in math
        s = "\\ensuremath{" + s + "}"
    return s


def fint(v):
    v = fin(v)
    if not np.isfinite(v):
        return "--"
    s = format(int(round(v)), ",")
    return s.replace(",", "{,}")


def fpct(v):
    """Relative change in per cent, one decimal, explicit sign (math-safe in text and math)."""
    v = fin(v)
    if not np.isfinite(v):
        return "--"
    if round(v, 1) == 0:
        v = 0.0
    return "\\ensuremath{%+.1f\\%%}" % v


def fp(p):
    p = fin(p)
    if not np.isfinite(p):
        return "--"
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def frank(v):
    return fnum(v, 2)


def pct_change(new, base):
    """100 * (new - base) / base ; positive = new is larger."""
    new, base = fin(new), fin(base)
    return 100.0 * (new - base) / base if np.isfinite(new) and np.isfinite(base) and base else float("nan")


def gain(a, b):
    """Relative improvement of a over b in per cent (positive = a lower/better)."""
    return -pct_change(a, b)


# ------------------------------------------------------------------ records
def parse_suffix(suffix, rec):
    """Classify a record from its tag suffix.

    v2: ``_v2-<R>[-<size>][-fm][-abl-<variant>-<backbone> | -exo-<mode>]``.
    v1: ``''`` and ``_L96`` are headline, ``_abl-<variant>`` / ``_exo-<mode>`` ablations and
    weather-pathway runs.  Anything else (lr sweeps, lookback sweeps, pilots) is 'other'.
    Zero-shot foundation-model runs at L=1024 are kind 'fm-long' (their own table block, never a
    headline cell).  Returns (regime, size, kind, variant, backbone).
    """
    regime, size, backbone = rec.get("regime"), rec.get("size"), rec.get("backbone")
    rest = (suffix or "").strip("_")
    m = V2_TAG_RE.match(rest)
    if m:
        regime = regime or m.group(1)
        toks = [t for t in (m.group(2) or "").split("-") if t]
    else:
        toks = rest.split("-") if rest else []
    kind, variant = "headline", None
    if "abl" in toks:
        i = toks.index("abl")
        after = toks[i + 1:]
        kind = "abl"
        if m and len(after) >= 2:
            variant, backbone = "-".join(after[:-1]), after[-1]
        else:
            variant = "-".join(after) or None
        toks = toks[:i]
    elif "exo" in toks:
        i = toks.index("exo")
        kind, variant = "exo", "-".join(toks[i + 1:]) or None
        toks = toks[:i]
    if "fm" in toks:
        toks.remove("fm")
        size = size or "fm"
        if kind == "headline" and rec.get("L") == 1024:
            kind = "fm-long"
    if toks:
        if m and all(t in SIZES or t == str(size) for t in toks):
            size = size or "-".join(toks)
        elif not m and toks == ["L96"] and kind == "headline":
            pass
        else:
            kind, variant = "other", "-".join(toks) + (f"+{variant}" if variant else "")
    return regime, size, kind, variant, backbone


# ---- corrupt target hours (data_v2.BAD_HOURS): the windows whose target range covers such an
# hour are dropped from the per-window arrays (DM, MCS) and removed from the cell's WAPE and MSE
# (the record keeps the unmasked values under *_unmasked). One hour of the French test series.
_META, _BADW, _SUMY = {}, {}, {}


def _bad_windows(evald, H):
    key = (evald, int(H))
    if key not in _BADW:
        try:
            import data_v2
            if evald in data_v2.BAD_HOURS:
                if evald not in _META:
                    _META[evald] = json.load(open(os.path.join(data_v2.PROC2, evald + ".json"), encoding="utf-8"))
                _BADW[key] = list(data_v2.bad_window_idx(_META[evald], evald, int(H)))
            else:
                _BADW[key] = []
        except Exception:                        # noqa: BLE001  (no processed corpus at hand)
            _BADW[key] = []
    return _BADW[key]


def _sum_abs_y(evald, H):
    """(sum |y| over every test window, sum |y| over the bad windows, n windows), physical units."""
    key = (evald, int(H))
    if key not in _SUMY:
        import data_v2
        meta = _META[evald]
        z = np.load(os.path.join(data_v2.PROC2, evald + ".npz"))
        y = np.abs(z["data"][:, meta["target_idx"]].astype(float))
        te0, te1 = meta["splits"]["test"]
        n = te1 - te0 - int(H) + 1
        cs = np.concatenate([[0.0], np.cumsum(y[te0:te1])])
        per = cs[int(H):int(H) + n] - cs[:n]
        bad = _bad_windows(evald, H)
        _SUMY[key] = (float(per.sum()), float(per[bad].sum()), int(n))
    return _SUMY[key]


def _apply_bad_hours(r):
    evald = r.get("eval_data") or r["dataset"]
    bad = _bad_windows(evald, r["H"])
    if not bad or r["_src"] != "main":
        return
    S_all, S_bad, n = _sum_abs_y(evald, r["H"])
    pw, pa = load_err_raw(r, "per_window"), load_err_raw(r, "per_window_ae")
    nb = len(bad)
    if pw is not None and len(pw) == n and np.isfinite(fin(r.get("mse"))):
        r["mse_unmasked"] = r["mse"]
        r["mse"] = float((fin(r["mse"]) * n - pw[bad].sum()) / (n - nb))
    if pa is not None and len(pa) == n and np.isfinite(fin(r.get("wape"))):
        r["wape_unmasked"] = r["wape"]
        r["wape"] = float(100.0 * (fin(r["wape"]) / 100.0 * S_all - int(r["H"]) * pa[bad].sum()) / (S_all - S_bad))
    r["_masked_windows"] = nb


def load_records(results_dir, fm_dir):
    recs = []
    for src, d in (("main", results_dir), ("fm", fm_dir)):
        if not d or not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(r, dict) or not {"dataset", "model", "H", "mse"} <= set(r):
                continue
            r = dict(r)
            r.pop("history", None)
            stem = os.path.splitext(os.path.basename(p))[0]
            base = f"{r['dataset']}_{r['model']}_L{r.get('L')}_H{r['H']}_s{r.get('seed')}"
            suffix = stem[len(base):] if stem.startswith(base) else str(r.get("tag") or "")
            regime, size, kind, variant, backbone = parse_suffix(suffix, r)
            r.update(_src=src, _file=p, _dir=d, _stem=stem, _suffix=suffix, _kind=kind,
                     _variant=variant, _size=size, _regime=regime, _backbone=backbone,
                     _seed=r.get("seed", 0))
            _apply_bad_hours(r)
            recs.append(r)
    return recs


def choose_regime(recs, wanted):
    """Headline regime: the requested one if present, else the single/most common one."""
    regs = Counter(r["_regime"] for r in recs if r["_kind"] == "headline")
    if not regs:
        return None
    if wanted in regs:
        return wanted
    return regs.most_common(1)[0][0]


def select_headline(recs, regime):
    """(dataset, model, H, seed) -> record; among candidates pick the lowest val_mse.

    Candidates are headline-kind records of the regime (v1: the untagged run plus the
    ``_L96`` run of Informer/Autoformer; v2: every (L, size, lr) run carrying the regime tag).
    Ties keep the first in suffix order, so the untagged run wins over ``_L96`` as in v1.
    Zero-shot foundation-model runs without a regime are admitted to every regime.
    """
    groups = defaultdict(list)
    for r in recs:
        if r["_kind"] != "headline" or r["_src"] == "fm" or r.get("zero_shot"):
            continue                    # zero-shot models are references (fm.* macros, tab_fm)
        if r["_regime"] != regime:
            continue
        groups[(r["dataset"], r["model"], r["H"], r["_seed"])].append(r)
    sel = {}
    for k, rs in groups.items():
        rs = sorted(rs, key=lambda r: (r["_suffix"], r["_file"]))
        best = rs[0]
        for r in rs[1:]:
            a, b = fin(r.get("val_mse")), fin(best.get("val_mse"))
            a = a if np.isfinite(a) else float("inf")
            b = b if np.isfinite(b) else float("inf")
            if a < b:
                best = r
        sel[k] = best
    return sel


def cells_from(sel):
    """(dataset, model, H) -> list of selected records (one per seed)."""
    cells = defaultdict(list)
    for (d, m, H, s), r in sel.items():
        cells[(d, m, H)].append(r)
    return cells


def metric_value(r, field):
    v = field(r) if callable(field) else r.get(field)
    return fin(v)


DIVERGED = set()


def keep_seeds(recs):
    """The uniform exclusion rule for a seed whose training diverged: validation MSE non-finite,
    or above ten times the median validation MSE of the other seeds of the same cell. Applied
    wherever seeds are averaged; the excluded runs are listed in proto.diverged_runs."""
    recs = list(recs)
    if len(recs) < 2:
        return recs
    v = np.array([fin(r.get("val_mse")) if r.get("val_mse") is not None else np.nan for r in recs], float)
    keep = []
    for i, r in enumerate(recs):
        if r.get("val_mse") is None:                     # no training loss (statistical models)
            keep.append(r); continue
        others = np.delete(v, i)
        others = others[np.isfinite(others)]
        ok = np.isfinite(v[i]) and (len(others) == 0 or v[i] <= 10.0 * float(np.median(others)))
        if ok:
            keep.append(r)
        else:
            DIVERGED.add(r.get("_stem", f"{r.get('dataset')}_{r.get('model')}_H{r.get('H')}_s{r.get('seed')}"))
    return keep


def seed_stats(recs, field):
    recs = keep_seeds(recs)
    vals = np.array([metric_value(r, field) for r in recs], float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return float(vals.mean()), sd, int(len(vals))


def load_err_raw(r, key="per_window_ae"):
    for d in (r["_dir"], os.path.dirname(r["_dir"])):
        p = os.path.join(d, "errors", r["_stem"] + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            if key in z.files:
                return np.asarray(z[key], float)
    return None


def load_err(r, key="per_window_ae"):
    """Per-window loss array with the windows covering a corrupt hour removed."""
    a = load_err_raw(r, key)
    if a is None:
        return None
    bad = _bad_windows(r.get("eval_data") or r["dataset"], r["H"])
    if bad and len(a) == _sum_abs_y(r.get("eval_data") or r["dataset"], r["H"])[2]:
        a = np.delete(a, bad)
    return a


def seed_mean_err(recs, key="per_window_ae"):
    # seeds excluded by the diverged-seed rule (keep_seeds) are left out here as well, exactly as
    # seed_stats leaves them out of the means (the seed count printed in the tables shows it)
    arrs = [a for a in (load_err(r, key) for r in keep_seeds(recs)) if a is not None and np.isfinite(a).all()]
    if not arrs:
        return None
    n = Counter(len(a) for a in arrs).most_common(1)[0][0]
    arrs = [a for a in arrs if len(a) == n]
    return np.mean(np.stack(arrs, 0), axis=0)


GRID_H = (24, 96, 336)
DA_BIAS_TOL = 3.0        # |published operator bias| above this (%) marks a biased series


def scopes_of(d):
    """Aggregation scopes a dataset belongs to: itself, all, grid|bench, and ett for ETT."""
    if is_variant(d):
        return (d,)
    out = (d, "all", "grid" if is_grid(d) else "bench")
    return out + (("ett",) if str(d).startswith("ETT") else ())


BUILTIN_CAL = {"PLANET", "PLANETX", "CalTST", "TaoVanilla", "TaoRefit", "TaoRefitAR"}   # calendar inside the architecture


def emit_mech(M, recs, regime, agg, summary):
    """Learning-curve (-lc{f}), drift (-drift{f}) and transfer (-xfer-{target}) macros:
    mech.<kind>.<ds>.<m>.H<H>.f<f>.wape | .dwape_pct (vs the headline cell), and means over
    the systems mech.<kind>.grid.<m>[.H<H>].f<f>.mean_dwape_pct; transfer: mech.xfer.<tgt>.<m>
    .H<H>.wape | .dwape_pct (mean over sources), mech.xfer.grid.<m>[.H<H>].mean_wape |
    .mean_dwape_pct | .n_pairs."""
    def mrecs(pfx):
        return [r for r in recs if r["_kind"] == "other" and r["_regime"] == regime and r["_src"] == "main"
                and str(r["_variant"] or "").startswith(pfx)]
    n = 0
    for kind in ("lc", "drift"):
        g = defaultdict(list)
        for r in mrecs(kind):
            try:
                f = float(r.get("train_frac") or r["_variant"][len(kind):])
            except ValueError:
                continue
            g[(r["dataset"], r["model"], r["H"], f)].append(r)
        per = defaultdict(list)
        for (d, m, H, f), rs in g.items():
            w = seed_stats(rs, "wape")[0]
            M.add(f"mech.{kind}.{d}.{m}.H{H}.f{f:g}.wape", fnum(w, 2))
            full = agg.get((d, m, H))
            if full:
                dw = pct_change(w, full["wape"])
                M.add(f"mech.{kind}.{d}.{m}.H{H}.f{f:g}.dwape_pct", fpct(dw))
                per[(m, H, f)].append(dw)
                per[(m, None, f)].append(dw)
            n += len(rs)
        for (m, H, f), v in per.items():
            key = f"mech.{kind}.grid.{m}." + (f"H{H}." if H is not None else "") + f"f{f:g}"
            M.add(key + ".mean_dwape_pct", fpct(np.mean(v)))
            M.add(key + ".n_cells", str(len(v)))
    g = defaultdict(list)
    for r in mrecs("xfer"):
        tgt = r.get("eval_data") or str(r["_variant"])[5:]
        g[(r["dataset"], tgt, r["model"], r["H"])].append(r)
    per_t = defaultdict(list)
    for (src, tgt, m, H), rs in g.items():
        w = seed_stats(rs, "wape")[0]
        M.add(f"mech.xfer.{tgt}.from.{src}.{m}.H{H}.wape", fnum(w, 2))
        full = agg.get((tgt, m, H))
        dw = pct_change(w, full["wape"]) if full else NAN
        if full:
            M.add(f"mech.xfer.{tgt}.from.{src}.{m}.H{H}.dwape_pct", fpct(dw))
        per_t[(tgt, m, H)].append((w, dw))
        n += len(rs)
    per_m = defaultdict(list)
    for (tgt, m, H), v in per_t.items():
        M.add(f"mech.xfer.{tgt}.{m}.H{H}.wape", fnum(np.mean([a for a, _ in v]), 2))
        M.add(f"mech.xfer.{tgt}.{m}.H{H}.dwape_pct", fpct(np.nanmean([b for _, b in v])))
        per_m[(m, H)] += v
        per_m[(m, None)] += v
    for (m, H), v in per_m.items():
        key = f"mech.xfer.grid.{m}" + (f".H{H}" if H is not None else "")
        M.add(key + ".mean_wape", fnum(np.mean([a for a, _ in v]), 2))
        M.add(key + ".mean_dwape_pct", fpct(np.nanmean([b for _, b in v])))
        M.add(key + ".n_pairs", str(len(v)))
    if n:
        summary.append(f"  mechanism records: {n}")


def emit_revision(M, recs, regime, agg, summary, aggs=None):
    """Revision experiments (2026-10-19 referee plan):
    path.<ds>.<m>.H<H>.wape | .dwape_vs_<base>_pct | .dwape_vs_PLANET_pct | .dm_stat | .dm_p
        the covariate-pathway control (-viahead: a library model with its native covariate
        pathway switched off and the block routed through the shared head, e.g. TiDEh vs TiDE;
        DM: native vs via-head on per-window AE, stat < 0 -> native better);
        path.grid.<m>.mean_dwape_vs_<base>_pct | .n_cells_vs_<base> | .n_worse_vs_<base> |
        .n_sig_worse | .n_sig_better
    exo.ar.<ds>.<m>.H<H>.wape | .dwape_pct   AR(1) correlated weather noise (-arnoise) vs the
        white-noise headline cell; exo.ar.grid.<m>.mean_dwape_pct | .max_dwape_pct | .n_cells;
        exo.ar.<ds>.H<H>.gap_PLANET_vs_TiDE_pct | .gap_white_pct and their grid means
    lrsel.<R>.<ds>.<m>.lr_R2 | .lr_own | .changed | .val_dmse_pct   learning rate selected on the
        validation loss under regime R vs the R2 selection, at the R2-selected width/backbone;
        lrsel.<R>.n_models | .n_changed | .mean_val_dmse_pct | .min_val_dmse_pct
    lrsel.<R>.<ds>.<m>.H<H>.wape_own | .dwape_own_pct   the R0/R1 cell re-run with the rate selected
        under its own regime (-ownlr) against the headline cell of that regime (R2 rate);
        lrsel.<R>.grid.mean_dwape_own_pct | .max_gain_own_pct | .n_cells_own | .n_better_own;
        lrsel.own.<ds>.<Ra>_vs_<Rb>.mean_dwape_pct   regime steps over the re-run models with own
        rates (pooled over the models without a built-in calendar, as info.*; .mean_dwape_all_pct
        over every re-run model), and lrsel.r2.<ds>.<Ra>_vs_<Rb>.* the same cells at the R2 rate
    """
    def variant_recs(v):
        return [r for r in recs if r["_kind"] == "other" and r["_src"] == "main"
                and r["_regime"] == regime and r["_variant"] == v]
    cells = cells_from(select_headline(recs, regime))
    # (a) pathway control
    g = defaultdict(list)
    for r in variant_recs("viahead"):
        g[(r["dataset"], r["model"], r["H"])].append(r)
    per = defaultdict(list)
    for (d, m, H), rs in sorted(g.items(), key=lambda x: (x[0][0], x[0][1], hsort(x[0][2]))):
        w = seed_stats(rs, "wape")[0]
        base = m[:-1] if m.endswith("h") else m
        M.add(f"path.{d}.{m}.H{H}.wape", fnum(w, 2))
        M.add(f"path.{d}.{m}.H{H}.n_seeds", str(len(keep_seeds(rs))))
        nat, ref_ = agg.get((d, base, H)), agg.get((d, "PLANET", H))
        if nat:
            dw = pct_change(w, nat["wape"])
            M.add(f"path.{d}.{m}.H{H}.dwape_vs_{base}_pct", fpct(dw))
            per[(m, f"vs_{base}")].append(dw)
        if ref_:
            dr = pct_change(w, ref_["wape"])
            M.add(f"path.{d}.{m}.H{H}.dwape_vs_PLANET_pct", fpct(dr))
            per[(m, "vs_PLANET")].append(dr)
        a, b = seed_mean_err(rs), seed_mean_err(cells.get((d, base, H), []))
        if a is not None and b is not None and len(a) == len(b):
            s, p = stats.dm_test(b, a, h=int(H))
            M.add(f"path.{d}.{m}.H{H}.dm_stat", fnum(s, 2))
            M.add(f"path.{d}.{m}.H{H}.dm_p", fp(p))
            per[(m, "sig_worse")].append(int(np.isfinite(p) and p < 0.05 and s < 0))
            per[(m, "sig_better")].append(int(np.isfinite(p) and p < 0.05 and s > 0))
    for (m, k), v in per.items():
        if k.startswith("vs_"):
            M.add(f"path.grid.{m}.mean_dwape_{k}_pct", fpct(np.nanmean(v)))
            M.add(f"path.grid.{m}.n_cells_{k}", str(len(v)))
            M.add(f"path.grid.{m}.n_worse_{k}", str(sum(1 for x in v if x > 0)))
        else:
            M.add(f"path.grid.{m}.n_{k}", str(int(sum(v))))
    if g:
        summary.append(f"  pathway-control cells: {len(g)}")
    # (b) AR(1) noise sensitivity
    g = defaultdict(list)
    for r in variant_recs("arnoise"):
        g[(r["dataset"], r["model"], r["H"])].append(r)
    per, wa = defaultdict(list), {}
    for (d, m, H), rs in sorted(g.items(), key=lambda x: (x[0][0], x[0][1], hsort(x[0][2]))):
        w = seed_stats(rs, "wape")[0]
        wa[(d, m, H)] = w
        M.add(f"exo.ar.{d}.{m}.H{H}.wape", fnum(w, 2))
        base = agg.get((d, m, H))
        if base:
            dw = pct_change(w, base["wape"])
            M.add(f"exo.ar.{d}.{m}.H{H}.dwape_pct", fpct(dw))
            per[m].append(dw)
    for m, v in per.items():
        M.add(f"exo.ar.grid.{m}.mean_dwape_pct", fpct(np.mean(v)))
        M.add(f"exo.ar.grid.{m}.max_dwape_pct", fpct(np.max(v)))
        M.add(f"exo.ar.grid.{m}.n_cells", str(len(v)))
    gaps = defaultdict(list)
    for (d, H) in sorted({(d, H) for d, _, H in wa}, key=lambda x: (x[0], hsort(x[1]))):
        if (d, "PLANET", H) in wa and (d, "TiDE", H) in wa and (d, "PLANET", H) in agg and (d, "TiDE", H) in agg:
            ga = pct_change(wa[(d, "PLANET", H)], wa[(d, "TiDE", H)])
            gw = pct_change(agg[(d, "PLANET", H)]["wape"], agg[(d, "TiDE", H)]["wape"])
            M.add(f"exo.ar.{d}.H{H}.gap_PLANET_vs_TiDE_pct", fpct(ga))
            M.add(f"exo.ar.{d}.H{H}.gap_white_pct", fpct(gw))
            gaps["ar"].append(ga); gaps["white"].append(gw)
    if gaps:
        M.add("exo.ar.grid.gap_PLANET_vs_TiDE_mean_pct", fpct(np.mean(gaps["ar"])))
        M.add("exo.ar.grid.gap_white_mean_pct", fpct(np.mean(gaps["white"])))
        M.add("exo.ar.grid.n_gap_cells", str(len(gaps["ar"])))
    if g:
        summary.append(f"  AR-noise cells: {len(g)}")
    # (c) learning-rate selection per regime
    tune = defaultdict(dict)                      # (regime, d, m) -> {(size, lr, backbone): val}
    for r in recs:
        if (r["_kind"] == "other" and r["_src"] == "main" and str(r["_variant"] or "").endswith("-tune")
                and int(r.get("H") or 0) == 96 and r.get("lr") is not None):
            tune[(r["_regime"], r["dataset"], r["model"])][
                (float(r.get("size") or 1.0), float(r["lr"]), r.get("backbone") or "patch")] = fin(r.get("val_mse"))
    acc = defaultdict(list)
    for (rg, d, m), runs in sorted(tune.items(), key=str):
        if rg == regime or rg is None:
            continue
        base = tune.get((regime, d, m))
        if not base:
            continue
        cfgs = {(k, bb) for (k, _, bb) in runs}
        if len(cfgs) == 1:        # the regime's sweep ran at the protocol's (width, backbone), which
            cfg = next(iter(cfgs))  # is selected on the mean over the reference systems, not on d alone
        else:
            kb = min(base, key=lambda k: base[k] if np.isfinite(base[k]) else np.inf)
            cfg = (kb[0], kb[2])
        own = {lr: v for (k, lr, bb), v in runs.items() if (k, bb) == cfg and np.isfinite(v)}
        r2 = {lr: v for (k, lr, bb), v in base.items() if (k, bb) == cfg and np.isfinite(v)}
        if not own or not r2:
            continue
        lr_r2, lr_own = min(r2, key=r2.get), min(own, key=own.get)
        M.add(f"lrsel.{rg}.{d}.{m}.lr_R2", f"{lr_r2:g}")
        M.add(f"lrsel.{rg}.{d}.{m}.lr_own", f"{lr_own:g}")
        M.add(f"lrsel.{rg}.{d}.{m}.changed", "1" if lr_own != lr_r2 else "0")
        dv = pct_change(own[lr_own], own.get(lr_r2, NAN))     # own-best vs the R2 rate under this regime
        M.add(f"lrsel.{rg}.{d}.{m}.val_dmse_pct", fpct(dv))
        acc[rg].append((lr_own != lr_r2, dv))
    for rg, v in acc.items():
        M.add(f"lrsel.{rg}.n_models", str(len(v)))
        M.add(f"lrsel.{rg}.n_changed", str(sum(1 for c, _ in v if c)))
        dvs = [x for _, x in v if np.isfinite(x)]
        if dvs:
            M.add(f"lrsel.{rg}.mean_val_dmse_pct", fpct(np.mean(dvs)))
            M.add(f"lrsel.{rg}.min_val_dmse_pct", fpct(np.min(dvs)))
    if acc:
        summary.append(f"  lr-per-regime checks: {sum(len(v) for v in acc.values())}")
    # (c2) R0/R1 cells re-run with the learning rate selected under their own regime (-ownlr;
    #      a cell whose own rate equals the R2 rate keeps its headline record, which then stands
    #      in for it in the regime steps below)
    aggs = aggs or {}
    own = defaultdict(list)
    for r in recs:
        if (r["_kind"] == "other" and r["_src"] == "main" and r["_variant"] == "ownlr"
                and r["_regime"] in aggs):
            own[(r["_regime"], r["dataset"], r["model"], r["H"])].append(r)
    wo, per = {}, defaultdict(list)
    for (rg, d, m, H), rs in sorted(own.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], hsort(x[0][3]))):
        w = seed_stats(rs, "wape")[0]
        wo[(rg, d, m, H)] = w
        M.add(f"lrsel.{rg}.{d}.{m}.H{H}.wape_own", fnum(w, 2))
        M.add(f"lrsel.{rg}.{d}.{m}.H{H}.n_seeds_own", str(len(keep_seeds(rs))))
        base = aggs[rg].get((d, m, H))
        if base:
            dw = pct_change(w, base["wape"])
            M.add(f"lrsel.{rg}.{d}.{m}.H{H}.dwape_own_pct", fpct(dw))
            if np.isfinite(dw):
                per[rg].append(dw)
    for rg, v in per.items():
        M.add(f"lrsel.{rg}.grid.mean_dwape_own_pct", fpct(np.mean(v)))
        M.add(f"lrsel.{rg}.grid.max_gain_own_pct", fpct(np.min(v)))
        M.add(f"lrsel.{rg}.grid.n_cells_own", str(len(v)))
        M.add(f"lrsel.{rg}.grid.n_better_own", str(sum(1 for x in v if x < 0)))
    keys = sorted({(d, m, H) for _, d, m, H in wo}, key=lambda x: (x[0], x[1], hsort(x[2])))

    def own_cell(rg, d, m, H, use_own):
        if use_own and (rg, d, m, H) in wo:
            return wo[(rg, d, m, H)]
        a = aggs.get(rg, {}).get((d, m, H))
        return a["wape"] if a else NAN
    regs = [rg for rg in ("R0", "R1", "R2") if rg in aggs]
    steps = defaultdict(list)
    for i in range(1, len(regs)):
        for j in range(i):
            ra, rb = regs[i], regs[j]
            for (d, m, H) in keys:
                for lab, use in (("own", True), ("r2", False)):
                    dw = pct_change(own_cell(ra, d, m, H, use), own_cell(rb, d, m, H, use))
                    if np.isfinite(dw):
                        M.add(f"lrsel.{lab}.{d}.{m}.H{H}.{ra}_vs_{rb}.dwape_pct", fpct(dw))
                        steps[(lab, d, ra, rb, "all")].append(dw)
                        if m not in BUILTIN_CAL:
                            steps[(lab, d, ra, rb, "pooled")].append(dw)
    for (lab, d, ra, rb, sc), v in steps.items():
        sfx = "_all" if sc == "all" else ""
        M.add(f"lrsel.{lab}.{d}.{ra}_vs_{rb}.mean_dwape{sfx}_pct", fpct(np.mean(v)))
        M.add(f"lrsel.{lab}.{d}.{ra}_vs_{rb}.n_cells{sfx}", str(len(v)))
    if own:
        summary.append(f"  own-rate cells: {len(own)}")
    # (d) multi-point weather: the -mp copies against the single-point headline cells
    per, wm = defaultdict(list), {}
    for (d, m, H), rs in sorted(cells.items(), key=lambda x: (x[0][0], x[0][1], hsort(x[0][2]))):
        if not is_variant(d):
            continue
        base_d = d[:-3]
        w = seed_stats(rs, "wape")[0]
        wm[(base_d, m, H)] = w
        M.add(f"mp.{base_d}.{m}.H{H}.wape", fnum(w, 2))
        M.add(f"mp.{base_d}.{m}.H{H}.n_seeds", str(len(keep_seeds(rs))))
        base = agg.get((base_d, m, H))
        if base:
            dw = pct_change(w, base["wape"])
            M.add(f"mp.{base_d}.{m}.H{H}.dwape_pct", fpct(dw))
            per[m].append(dw)
    for m, v in per.items():
        M.add(f"mp.grid.{m}.mean_dwape_pct", fpct(np.mean(v)))
        M.add(f"mp.grid.{m}.min_dwape_pct", fpct(np.min(v)))
        M.add(f"mp.grid.{m}.n_cells", str(len(v)))
    gaps = defaultdict(list)
    for (d, H) in sorted({(d, H) for d, _, H in wm}, key=lambda x: (x[0], hsort(x[1]))):
        if all((d, m, H) in wm and (d, m, H) in agg for m in ("PLANET", "TiDE")):
            gm = pct_change(wm[(d, "PLANET", H)], wm[(d, "TiDE", H)])
            gs = pct_change(agg[(d, "PLANET", H)]["wape"], agg[(d, "TiDE", H)]["wape"])
            M.add(f"mp.{d}.H{H}.gap_PLANET_vs_TiDE_pct", fpct(gm))
            M.add(f"mp.{d}.H{H}.gap_single_pct", fpct(gs))
            gaps["mp"].append(gm); gaps["single"].append(gs)
    if gaps:
        M.add("mp.grid.gap_PLANET_vs_TiDE_mean_pct", fpct(np.mean(gaps["mp"])))
        M.add("mp.grid.gap_single_mean_pct", fpct(np.mean(gaps["single"])))
        M.add("mp.grid.n_gap_cells", str(len(gaps["mp"])))
    if wm:
        summary.append(f"  multi-point weather cells: {len(wm)}")


SPIKE_REL, SPIKE_NEIGH = 0.15, 0.05
_WX = {}


def weather_dependence(proc, d):
    """Per-system weather dependence on the training split, weekdays, daily means: R^2 of a
    quadratic in daily mean temperature for daily mean load, the winter (DJF) to summer (JJA)
    ratio of monthly mean load, and the linear correlation. Cached per dataset."""
    import pandas as pd
    key = (proc, d)
    if key in _WX:
        return _WX[key]
    z = np.load(os.path.join(proc, d + ".npz"))
    meta = json.load(open(os.path.join(proc, d + ".json"), encoding="utf-8"))
    a, b = meta["splits"]["train"]
    if "temp" not in meta["channels"]:
        _WX[key] = None
        return None
    ti = meta["channels"].index("temp")
    st = pd.to_datetime(z["stamps"][a:b].astype(np.int64), unit="s")
    df = pd.DataFrame({"load": z["data"][a:b, meta["target_idx"]].astype(float),
                       "temp": z["data"][a:b, ti].astype(float)}, index=st)
    day = df.resample("D").mean()
    day = day[day.index.dayofweek < 5].dropna()
    X = np.c_[np.ones(len(day)), day["temp"], day["temp"] ** 2]
    beta, *_ = np.linalg.lstsq(X, day["load"], rcond=None)
    r2 = 1.0 - np.var(day["load"] - X @ beta) / np.var(day["load"])
    mon = df.resample("MS").mean()
    w = mon[mon.index.month.isin([12, 1, 2])]["load"].mean()
    s = mon[mon.index.month.isin([6, 7, 8])]["load"].mean()
    _WX[key] = dict(r2=float(r2), winter_summer=float(w / s), corr=float(day["load"].corr(day["temp"])),
                    n_days=int(len(day)))
    return _WX[key]


def emit_combo(M, args, summary):
    """Cross-model combinations on the headline grid at the stored seed (combo_v2.py):
    comb.<ds>.H<H>.<c>.wape | .dwape_vs_TiDE_pct | .dwape_vs_PLANET_pct | .dm_vs_TiDE.stat | .p
    comb.grid.<c>.mean_wape | .mean_dwape_vs_TiDE_pct | .mean_dwape_vs_PLANET_pct
    comb.grid.<c>.n_better_TiDE | .n_sig_better_TiDE | .n_sig_worse_TiDE (and _PLANET)
    comb.grid.<m>.mean_wape_single   the single models at the stored seed (TiDE, PLANET, best_val)
    comb.grid.n_cells | comb.seed | comb.B ; comb.grid.CombTop3.n_cells_with_PLANET | _with_TiDE"""
    path = os.path.join(args.results, "combo.json")
    if not os.path.exists(path):
        return
    J = json.load(open(path, encoding="utf-8"))
    proto = J.pop("_protocol", {})
    cells = {k: v for k, v in J.items() if isinstance(v, dict) and "singles" in v}
    if not cells:
        return
    M.add("comb.seed", str(proto.get("seed", "")))
    M.add("comb.B", str(proto.get("B", "")))
    M.add("comb.grid.n_cells", str(len(cells)))
    combos = sorted({c for v in cells.values() for c in v if c.startswith("Comb")})
    for ref in ("TiDE", "PLANET"):
        M.add(f"comb.grid.{ref}.mean_wape_single", fnum(np.mean([v["singles"][ref]["wape"] for v in cells.values()]), 2))
    M.add("comb.grid.best_val.mean_wape_single",
          fnum(np.mean([v["singles"][v["best_val"]]["wape"] for v in cells.values()]), 2))
    M.add("comb.grid.best_val.n_cells_PLANET", str(sum(v["best_val"] == "PLANET" for v in cells.values())))
    M.add("comb.grid.best_val.n_cells_TiDE", str(sum(v["best_val"] == "TiDE" for v in cells.values())))
    for c in combos:
        w, dw, cnt = [], {"TiDE": [], "PLANET": []}, Counter()
        for key, v in cells.items():
            if c not in v:
                continue
            d, H = key.split("|H")
            e = v[c]
            w.append(e["wape"])
            M.add(f"comb.{d}.H{H}.{c}.wape", fnum(e["wape"], 2))
            for ref in ("TiDE", "PLANET"):
                g = pct_change(e["wape"], v["singles"][ref]["wape"])
                dw[ref].append(g)
                M.add(f"comb.{d}.H{H}.{c}.dwape_vs_{ref}_pct", fpct(g))
                dm = e.get(f"dm_vs_{ref}", {})
                M.add(f"comb.{d}.H{H}.{c}.dm_vs_{ref}.stat", fnum(dm.get("stat", NAN), 2))
                M.add(f"comb.{d}.H{H}.{c}.dm_vs_{ref}.p", fnum(dm.get("p", NAN), 3))
                cnt[f"better_{ref}"] += int(e["wape"] < v["singles"][ref]["wape"])
                if np.isfinite(dm.get("p", NAN)) and dm["p"] < 0.05:
                    cnt[f"sig_{'better' if dm['stat'] < 0 else 'worse'}_{ref}"] += 1
            if c == "CombTop3":
                cnt["with_PLANET"] += int("PLANET" in e["members"])
                cnt["with_TiDE"] += int("TiDE" in e["members"])
        if not w:
            continue
        M.add(f"comb.grid.{c}.n_cells", str(len(w)))
        M.add(f"comb.grid.{c}.mean_wape", fnum(np.mean(w), 2))
        for ref in ("TiDE", "PLANET"):
            M.add(f"comb.grid.{c}.mean_dwape_vs_{ref}_pct", fpct(np.mean(dw[ref])))
            for k in ("better", "sig_better", "sig_worse"):
                M.add(f"comb.grid.{c}.n_{k}_{ref}", str(cnt[f"{k}_{ref}"]))
        if c == "CombTop3":
            M.add("comb.grid.CombTop3.n_cells_with_PLANET", str(cnt["with_PLANET"]))
            M.add("comb.grid.CombTop3.n_cells_with_TiDE", str(cnt["with_TiDE"]))
    summary.append(f"  combination cells: {len(cells)} ({', '.join(combos)})")


def emit_data_notes(M, args, summary):
    """Data-quality facts quoted in the corpus description, from the processed corpus:
    data.OPSD-NL.step_pct            mean load of calendar 2016 over calendar 2015, in per cent
                                     (the reporting-perimeter step of January 2016);
    data.<ds>.bad_hour.value_gw | .lo_gw | .hi_gw | .stamp   the corrupt hour and its neighbours;
    data.<ds>.n_spikes               isolated single-hour spikes: |x_t - (x_{t-1}+x_{t+1})/2| above
                                     15 % of mean load while the two neighbours agree within 5 %;
    data.spike.rel_pct | .neigh_pct  the two thresholds."""
    import pandas as pd
    try:
        import data_v2
    except Exception:                                 # noqa: BLE001
        return
    proc = args.data or data_v2.PROC2
    M.add("data.spike.rel_pct", r"%g\%%" % (100 * SPIKE_REL))
    M.add("data.spike.neigh_pct", r"%g\%%" % (100 * SPIKE_NEIGH))
    for d in sorted(f[:-5] for f in os.listdir(proc) if f.startswith("OPSD-") and f.endswith(".json")):
        if is_variant(d):
            continue
        try:
            z = np.load(os.path.join(proc, d + ".npz"))
            meta = json.load(open(os.path.join(proc, d + ".json"), encoding="utf-8"))
        except Exception:                             # noqa: BLE001
            continue
        y = z["data"][:, meta["target_idx"]].astype(float)
        st = pd.to_datetime(z["stamps"].astype(np.int64), unit="s")
        m = float(np.nanmean(y))
        wx = weather_dependence(proc, d)
        if wx:
            M.add(f"data.{d}.wx_r2", fnum(wx["r2"], 2))
            M.add(f"data.{d}.wx_winter_summer", fnum(wx["winter_summer"], 2))
            M.add(f"data.{d}.wx_corr", fnum(wx["corr"], 2))
        mid = 0.5 * (y[:-2] + y[2:])
        spike = (np.abs(y[1:-1] - mid) > SPIKE_REL * m) & (np.abs(y[:-2] - y[2:]) < SPIKE_NEIGH * m)
        M.add(f"data.{d}.n_spikes", fint(int(spike.sum())))
        if d == "OPSD-NL":
            s = pd.Series(y, index=st)
            a, b = s["2015"].mean(), s["2016"].mean()
            M.add("data.OPSD-NL.step_pct", fpct(100.0 * (b / a - 1.0)))
            M.add("data.OPSD-NL.mean_2015_gw", fnum(a / 1000, 1))
            M.add("data.OPSD-NL.mean_2016_gw", fnum(b / 1000, 1))
        for j in data_v2.bad_hours_idx(d, z["stamps"]):
            M.add(f"data.{d}.bad_hour.value_gw", fnum(y[j] / 1000, 0))
            nb = [y[k] for k in (j - 1, j + 1) if 0 <= k < len(y)]
            M.add(f"data.{d}.bad_hour.lo_gw", fnum(min(nb) / 1000, 0))
            M.add(f"data.{d}.bad_hour.hi_gw", fnum(max(nb) / 1000, 0))
            M.add(f"data.{d}.bad_hour.stamp", st[j].strftime("%Y-%m-%d %H:00") + " UTC")
            split = next((k for k, (a_, b_) in meta["splits"].items() if a_ <= j < b_), "?")
            M.add(f"data.{d}.bad_hour.split", split)
    summary.append("  data notes emitted")


def emit_epochs(M, recs, regime, summary, cap=30):
    """proto.epochs.mean | .median | .frac_capped | .cap | .n_runs over the headline runs of the
    regime (every neural model, every system, every seed)."""
    ep = [fin(r.get("epochs_run")) for r in recs
          if r["_kind"] == "headline" and r["_regime"] == regime and r["_src"] == "main"
          and r.get("epochs_run") is not None and r.get("train_time_s")]
    ep = [e for e in ep if np.isfinite(e)]
    if not ep:
        return
    cap = max(cap, int(max(ep)))
    M.add("proto.epochs.cap", str(cap))
    M.add("proto.epochs.mean", fnum(np.mean(ep), 1))
    M.add("proto.epochs.median", fnum(np.median(ep), 0))
    M.add("proto.epochs.frac_capped", fpct(100.0 * np.mean([e >= cap for e in ep])).replace("+", ""))
    M.add("proto.epochs.n_runs", str(len(ep)))
    summary.append(f"  epochs run: mean {np.mean(ep):.1f}, {100*np.mean([e >= cap for e in ep]):.0f}% at the cap")


def emit_budget(M, recs, regime, agg, summary, variant="ep90"):
    """Budget-sensitivity check: headline models re-trained at a larger epoch budget (tag
    -ep30). budget.<ds>.<m>.H<H>.wape | .dwape_pct | .epochs ; budget.<ds>.mean|min|max_dwape_pct,
    .n_cells ; budget.<ds>.H<H>.spearman | .best_small | .best_large ; budget.<ds>.mean_spearman,
    .n_best_changed."""
    from scipy.stats import spearmanr
    g = defaultdict(list)
    for r in recs:
        if r["_kind"] == "other" and r["_regime"] == regime and r["_src"] == "main" and r["_variant"] == variant:
            g[(r["dataset"], r["model"], r["H"])].append(r)
    if not g:
        return
    per_d = defaultdict(list)
    cells = defaultdict(dict)          # (d, H) -> m -> (w_small, w_large)
    for (d, m, H), rs in g.items():
        w = seed_stats(rs, "wape")[0]
        M.add(f"budget.{d}.{m}.H{H}.wape", fnum(w, 2))
        M.add(f"budget.{d}.{m}.H{H}.epochs", fnum(seed_stats(rs, "epochs_run")[0], 0))
        base = agg.get((d, m, H))
        if base:
            dw = pct_change(w, base["wape"])
            M.add(f"budget.{d}.{m}.H{H}.dwape_pct", fpct(dw))
            if np.isfinite(dw):
                per_d[d].append(dw)
                cells[(d, H)][m] = (base["wape"], w)
    for d, v in per_d.items():
        M.add(f"budget.{d}.mean_dwape_pct", fpct(np.mean(v)))
        M.add(f"budget.{d}.min_dwape_pct", fpct(np.min(v)))
        M.add(f"budget.{d}.max_dwape_pct", fpct(np.max(v)))
        M.add(f"budget.{d}.n_cells", str(len(v)))
    rho_d, changed = defaultdict(list), Counter()
    for (d, H), mm in cells.items():
        if len(mm) >= 3:
            a = [mm[m][0] for m in mm]; b = [mm[m][1] for m in mm]
            rho = float(spearmanr(a, b).correlation)
            M.add(f"budget.{d}.H{H}.spearman", fnum(rho, 2))
            rho_d[d].append(rho)
            bs, bl = min(mm, key=lambda m: mm[m][0]), min(mm, key=lambda m: mm[m][1])
            M.add(f"budget.{d}.H{H}.best_small", PRETTY_TXT.get(bs, bs))
            M.add(f"budget.{d}.H{H}.best_large", PRETTY_TXT.get(bl, bl))
            changed[d] += int(bs != bl)
    for d, v in rho_d.items():
        M.add(f"budget.{d}.mean_spearman", fnum(np.mean(v), 2))
        M.add(f"budget.{d}.n_best_changed", str(changed[d]))
        M.add(f"budget.{d}.n_horizons", str(len(v)))
    summary.append(f"  budget-check cells: {sum(len(v) for v in per_d.values())}")


NAN = float("nan")


def is_grid(dataset):
    return dataset.startswith("OPSD") and not dataset.endswith("-mp")


def is_variant(dataset):
    """Sensitivity copies of a system (multi-point weather, -mp): per-cell macros only."""
    return str(dataset).endswith("-mp")


def hsort(H):
    return int(H)


# ------------------------------------------------------------------ macro store
class Macros:
    def __init__(self):
        self.d = OrderedDict()

    def add(self, key, value):
        key = re.sub(r"\s+", "-", str(key))
        if any(c in key for c in "{}\\#%"):
            raise ValueError(f"bad macro key {key!r}")
        if key in self.d and self.d[key] != value:
            raise ValueError(f"duplicate macro key {key!r}: {self.d[key]!r} vs {value!r}")
        self.d[key] = str(value)

    def write(self, path, header_lines):
        lines = ["% " + h for h in header_lines]
        # robust (safe in captions) and visible when a key is missing: prints ?? in the draft
        lines.append("\\makeatletter\\protected\\def\\nv#1{\\ifcsname n@#1\\endcsname"
                     "\\csname n@#1\\endcsname\\else\\textbf{??}\\fi}\\makeatother")
        lines.append("\\makeatletter")
        for k in sorted(self.d):
            lines.append("\\expandafter\\gdef\\csname n@%s\\endcsname{%s}" % (k, self.d[k]))
        lines.append("\\makeatother")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")


KEY_DOC = """KEY scheme (dot-separated; <ds> = dataset, <m> = model, <H> = horizon, <R> = regime):
  <ds>.<m>.H<H>.<metric>          headline cell, mean over seeds; metrics: wape mse mae peak dt
                                  ramp params_k train_s infer_s mae_mw rmse_mw mape val_mse epochs
  <ds>.<m>.H<H>.<metric>_sd       seed standard deviation (ddof=1; -- with one seed)
  <ds>.<m>.H<H>.n_seeds | .L | .size   seeds averaged, selected lookback, size tag
  <ds>.<m>.H<H>.rank[_wape]       rank in the cell by mean MSE [WAPE]; <ds>.H<H>.n_models
  <ds>.<m>.meanrank[_wape]        mean rank over the horizons of <ds>
  all|grid|bench.<m>.meanrank[_wape] | .n_cells | .n_best   over all / OPSD / non-OPSD cells
  ref.<ds>.H<H>.best_baseline | .gain_wape_pct | .gain_mse_pct   reference vs best baseline
  ref.<ds>.H<H>.vs.<m>.gain_wape_pct | .gain_mse_pct               reference vs each model
  ref.<ds>|all|grid|bench.mean_gain_wape_pct | mean_gain_mse_pct
  abl.<ds>.<variant>.H<H>.dmse_pct | .dwape_pct | .mse | .wape | .n_seeds  (reference model;
                                  abl.<m>.<ds>.<variant>... for any other model)
  abl.<variant>.mean_dmse_pct | .max_dmse_pct | .min_dmse_pct | .n_cells
  exo.<ds>.<mode>.H<H>.wape | .mse | .peak | .gain_wape_pct ; exo.<mode>.mean_gain_wape_pct
  dm.<ds>.<m>.H<H>.stat | .p | .p_holm    Diebold-Mariano, reference vs <m> on per-window AE
                                  (stat < 0: reference better); Holm within --holm-family
  dm.<m>.wins | .ties | .losses   cells where the reference is significantly better / neither /
                                  worse at Holm-adjusted p < 0.05; *_raw with raw p; dm.total.*
  dm.<m>.median_stat | .n_cells ; dm.n_tests | .holm_family
  mcs.<ds>.H<H>.<m>.p | .in       MCS p-value and membership (1/0) at alpha; mcs.<ds>.H<H>.n_in
  mcs.<ds>.H<H>.members ; mcs.<m>.n_in | .n_cells ; mcs.grid|bench.<m>.n_in ; mcs.alpha | .B
  info.<ds>.<m>.H<H>.<Ra>_vs_<Rb>.dwape_pct   WAPE change from regime Rb to Ra (v2 only)
  da.<ds>.<m>.<field> | da.<ds>.<m>.gain_wape_pct (vs TSO) | da.<ds>.n_days | da.<ds>.best
  tso.<ds>.<field> ; prob.<ds>.<m>.H<H>.<field> ; data.<ds>.<field>
  comb.<ds>.H<H>.<c>.wape | .dwape_vs_TiDE_pct ; comb.grid.<c>.mean_wape | .mean_dwape_vs_<ref>_pct |
                                  .n_better_<ref> | .n_sig_better_<ref>   grid combinations at the stored seed
  n.models.learned | n.models.fm | n.systems   distinct learned / zero-shot models (headline regime;
                                  TaoVanilla+TaoRefit count once, TaoRefitAR separately) and OPSD systems with records
  proto.nwp.temp_rmse | proto.head.params_k    NWP temperature RMSE at 24 h (K); ExoHead(E=14,
                                  L=336, H=24) parameters in thousands
  grid|bench|ett.<m>.mean_wape | .mean_wape_sd | .mean_peak | .mean_dt   per-model means over cells
  ref.grid.H<H>.mean_gain_wape_pct ; grid.median_gap_wape_pct (reference vs median learned method)
  abl.<bb>.<ds>.<variant>.H<H>.* | abl.<bb>.<variant>.mean_dmse_pct   per residual backbone
  info.grid.<m>.<Ra>_vs_<Rb>.mean_dwape_pct | info.grid.<Ra>_vs_<Rb>.mean_dwape_pct (pooled over
                                  the methods without a built-in calendar)
  mech.lc|drift.<ds>.<m>.H<H>.f<f>.wape | .dwape_pct ; mech.lc|drift.grid.<m>[.H<H>].f<f>.mean_dwape_pct
  mech.xfer.<tgt>.<m>.H<H>.wape | .dwape_pct ; mech.xfer.grid.<m>[.H<H>].mean_wape | .mean_dwape_pct
  da.grid.<m>.n_beats_tso | .n_systems | .mean_gain_wape_pct ; da.grid.n_systems | .any.n_beats_tso
  da.grid.<m>.n_beats_tso_(hol|lock|xlock)_unbiased | .n_systems_*_unbiased | .mean_wape_*_unbiased
  mcs.<scope>.<m>.H<H>.n_in ; mcs.<scope>.H<H>.size_min | .size_max | .size_mean | .n_cells
  prob.grid[.H<H>].<m>.mean_<field> | .n_best_crps ; prob.grid.n_systems | .min_picp_aci | .max_picp_aci
  fm[.<R>][.L1024].<ds>.<m>.H<H>.wape | .dwape_vs_best_pct ; fm.grid[...].<m>[.H<H>].n_beats_best
  cost.<ref>_vs_backbone.train_pct | .infer_pct | .params_pct ; cost.backbone
  budget.<ds>.<m>.H<H>.wape | .dwape_pct ; budget.<ds>.mean_dwape_pct | .mean_spearman | .n_best_changed
  dm.<ds>.<m>.H<H>.p_boot | .p_boot_holm | .stat_boot   stationary-bootstrap DM (block max(n^1/3, H));
  dm.<m>.wins_boot | .wins_allseeds | .wins_tested ; dm.<ds>.H<H>.n | .n_eff_median ; dm.n_eff_median
  path.* exo.ar.* lrsel.*   revision experiments (see emit_revision); proto.n_diverged | .diverged_runs
  dayahead_08.*             the 08:00 UTC origin, same keys as da.*; <m>-ens = seed ensemble;
  da.<ds>.<m>.dm_stat | .dm_p | .sig_better | .n_seeds_sig_better | .wape_lock | .wape_xlock | .wape_hol
  da.<ds>.<m>-refit.dwape_vs_once_pct (monthly refit vs the once-trained run of the same seed) ;
  da.grid.<m>-refit.mean|min|max_dwape_vs_once_unbiased_pct | .n_systems_vs_once_unbiased | .n_better_than_once_unbiased
  Non-headline regimes carry the prefix <R>. on every key above (e.g. R1.OPSD-DE.PLANET.H96.wape).
Formats: WAPE 2 dp, MSE 4, MAE 3, percentages 1 dp with explicit sign (\\ensuremath),
  p-values \\ensuremath{<}0.001 or 3 dp, ranks 2 dp, integers with {,} grouping, missing = --."""


# ------------------------------------------------------------------ generation
def emit_regime(M, recs, regime, prefix, args, summary):
    sel = select_headline(recs, regime)
    cells = cells_from(sel)
    if not cells:
        return
    datasets = sorted({d for d, _, _ in cells})
    summary.append(f"regime {regime!r}: {len(sel)} records -> {len(cells)} cells on {len(datasets)} datasets")

    # ---- headline metrics per cell
    agg = {}
    for (d, m, H), rs in cells.items():
        row = {}
        for mk, (field, nd) in METRICS.items():
            mean, sd, n = seed_stats(rs, field)
            row[mk] = mean
            row[mk + "_sd"] = sd
            M.add(f"{prefix}{d}.{m}.H{H}.{mk}", fnum(mean, nd))
            M.add(f"{prefix}{d}.{m}.H{H}.{mk}_sd", fnum(sd, nd))
        M.add(f"{prefix}{d}.{m}.H{H}.n_seeds", str(len(keep_seeds(rs))))
        Ls = Counter(r.get("L") for r in rs).most_common(1)[0][0]
        M.add(f"{prefix}{d}.{m}.H{H}.L", str(Ls))
        row["backbone"] = Counter(r.get("backbone") for r in rs).most_common(1)[0][0] or "patch"
        sz = Counter(r.get("_size") for r in rs).most_common(1)[0][0]
        M.add(f"{prefix}{d}.{m}.H{H}.size", str(sz) if sz else "--")
        agg[(d, m, H)] = row

    # ---- ranks
    from scipy.stats import rankdata
    ranks = defaultdict(lambda: defaultdict(list))   # scope -> model -> [rank]
    ranks_w = defaultdict(lambda: defaultdict(list))
    nbest = defaultdict(Counter)
    # aggregates, ranks, DM and MCS use the protocol horizons of the grids; the H=48 day-ahead
    # cells keep their per-cell macros but do not enter any count or mean
    cell_keys = sorted({(d, H) for d, _, H in cells if not is_variant(d) and (not is_grid(d) or hsort(H) in GRID_H)},
                       key=lambda x: (x[0], hsort(x[1])))
    headline_cells = set(cell_keys)
    for d, H in cell_keys:
        models = sorted(m for (dd, m, HH) in cells if dd == d and HH == H)
        M.add(f"{prefix}{d}.H{H}.n_models", str(len(models)))
        for metric, store in (("mse", ranks), ("wape", ranks_w)):
            vals = np.array([agg[(d, m, H)][metric] for m in models])
            ok = np.isfinite(vals)
            if ok.sum() < 2:
                continue
            rk = np.full(len(models), np.nan)
            rk[ok] = rankdata(vals[ok], method="average")
            for m, r in zip(models, rk):
                if np.isfinite(r):
                    M.add(f"{prefix}{d}.{m}.H{H}.rank" + ("" if metric == "mse" else "_wape"), frank(r))
                    for scope in scopes_of(d):
                        store[scope][m].append(float(r))
                        if metric == "mse" and r == 1:
                            nbest[scope][m] += 1
    for scope in set(ranks) | set(ranks_w):
        for m in set(ranks[scope]) | set(ranks_w[scope]):
            if ranks[scope].get(m):
                M.add(f"{prefix}{scope}.{m}.meanrank", frank(np.mean(ranks[scope][m])))
                if scope in ("all", "grid", "bench", "ett"):
                    M.add(f"{prefix}{scope}.{m}.n_cells", str(len(ranks[scope][m])))
                    M.add(f"{prefix}{scope}.{m}.n_best", str(nbest[scope][m]))
            if ranks_w[scope].get(m):
                M.add(f"{prefix}{scope}.{m}.meanrank_wape", frank(np.mean(ranks_w[scope][m])))
    for scope in ("all", "grid", "bench", "ett") + tuple(datasets):
        ncell = len({(d, H) for d, H in cell_keys if scope in scopes_of(d)})
        if ncell:
            M.add(f"{prefix}{scope}.n_cells", str(ncell))
            M.add(f"{prefix}{scope}.n_models", str(len({m for dd, m, _ in cells if scope in scopes_of(dd)})))

    # ---- per-model aggregates over the cells of a scope (prose: mean WAPE, seed sd, peaks)
    for scope in ("grid", "bench", "ett"):
        keys = [k for k in agg if scope in scopes_of(k[0]) and (k[0], k[2]) in headline_cells]
        for m in sorted({k[1] for k in keys}):
            rows = [agg[k] for k in keys if k[1] == m]
            M.add(f"{prefix}{scope}.{m}.n_cells_agg", str(len(rows)))
            for mk, nd in (("wape", 2), ("peak", 2), ("dt", 2), ("mse", 4), ("mae", 3)):
                M.add(f"{prefix}{scope}.{m}.mean_{mk}", fnum(np.nanmean([r[mk] for r in rows]), nd))
            sds = [r["wape_sd"] for r in rows if np.isfinite(fin(r["wape_sd"]))]
            M.add(f"{prefix}{scope}.{m}.mean_wape_sd", fnum(np.mean(sds) if sds else NAN, 2))

    # ---- reference vs best baseline
    ref = args.ref
    gains = defaultdict(lambda: defaultdict(list))
    gaps = {}
    for d, H in cell_keys:
        if (d, ref, H) not in agg:
            continue
        others = {m: agg[(d, m, H)] for (dd, m, HH) in cells if dd == d and HH == H and m != ref}
        if not others:
            continue
        for metric in ("wape", "mse"):
            cand = {m: v[metric] for m, v in others.items() if np.isfinite(v[metric])}
            if not cand:
                continue
            bb = min(cand, key=cand.get)
            g = gain(agg[(d, ref, H)][metric], cand[bb])
            if metric == "wape":
                M.add(f"{prefix}ref.{d}.H{H}.best_baseline", bb)
            else:
                M.add(f"{prefix}ref.{d}.H{H}.best_baseline_mse", bb)
            M.add(f"{prefix}ref.{d}.H{H}.gain_{metric}_pct", fpct(g))
            for scope in scopes_of(d) + ((f"grid.H{H}",) if is_grid(d) else ()):
                gains[scope][metric].append(g)
            for m, v in cand.items():
                M.add(f"{prefix}ref.{d}.H{H}.vs.{m}.gain_{metric}_pct", fpct(gain(agg[(d, ref, H)][metric], v)))
            if metric == "wape" and len(cand) >= 3:
                med = float(np.median(list(cand.values())))
                gaps.setdefault("grid" if is_grid(d) else "bench", []).append(pct_change(agg[(d, ref, H)][metric], med))
                M.add(f"{prefix}ref.{d}.H{H}.median_gap_wape_pct", fpct(pct_change(agg[(d, ref, H)][metric], med)))
    for scope, byk in gains.items():
        for metric, v in byk.items():
            M.add(f"{prefix}ref.{scope}.mean_gain_{metric}_pct", fpct(np.mean(v)))
            M.add(f"{prefix}ref.{scope}.min_gain_{metric}_pct", fpct(np.min(v)))
            M.add(f"{prefix}ref.{scope}.max_gain_{metric}_pct", fpct(np.max(v)))
            M.add(f"{prefix}ref.{scope}.n_cells_gain", str(len(v)))
    for scope, v in gaps.items():
        M.add(f"{prefix}{scope}.median_gap_wape_pct", fpct(np.mean(v)))

    # ---- ablations: per backbone, seed-paired against the full model on the same backbone
    # (falling back to the headline cell); un-prefixed keys refer to the headline backbone
    abl = defaultdict(list)
    for r in recs:
        if r["_kind"] == "abl" and (r["_regime"] == regime or r["_regime"] is None and regime == "v1"):
            bb = r.get("_backbone") or r.get("backbone") or "patch"
            abl[(r["dataset"], r["model"], r["H"], r["_variant"], bb)].append(r)
    by_variant = defaultdict(list)        # (bb, var) -> [(dmse, dwape)]
    by_variant_head = defaultdict(list)   # var -> [(dmse, dwape)] on the headline backbone
    n_abl = n_runs = 0
    for (d, m, H, var, bb), rs in sorted(abl.items(), key=lambda x: (x[0][0], x[0][1], hsort(x[0][2]), x[0][3], x[0][4])):
        n_runs += len(rs)
        full = agg.get((d, m, H))
        fullrs = abl.get((d, m, H, "full", bb))
        head_bb = full["backbone"] if full else "patch"
        heads = [f"{prefix}abl.{bb}.{d}.{var}.H{H}" if m == ref else f"{prefix}abl.{bb}.{m}.{d}.{var}.H{H}"]
        if bb == head_bb:
            heads.append(f"{prefix}abl.{d}.{var}.H{H}" if m == ref else f"{prefix}abl.{m}.{d}.{var}.H{H}")
        mse, _, n = seed_stats(rs, "mse")
        wape, _, _ = seed_stats(rs, "wape")
        for h in heads:
            M.add(h + ".mse", fnum(mse, 4))
            M.add(h + ".wape", fnum(wape, 2))
            M.add(h + ".n_seeds", str(n))
        if var == "full":
            continue
        if fullrs:                                        # seed-paired relative change
            fb = {r["_seed"]: (fin(r.get("mse")), fin(r.get("wape"))) for r in fullrs}
            fm_, fw_ = np.nanmean([v[0] for v in fb.values()]), np.nanmean([v[1] for v in fb.values()])
            dms = [pct_change(fin(r.get("mse")), fb.get(r["_seed"], (fm_, fw_))[0]) for r in rs]
            dws = [pct_change(fin(r.get("wape")), fb.get(r["_seed"], (fm_, fw_))[1]) for r in rs]
            dm_, dw_ = float(np.nanmean(dms)), float(np.nanmean(dws))
        elif full is not None:
            dm_, dw_ = pct_change(mse, full["mse"]), pct_change(wape, full["wape"])
        else:
            continue
        for h in heads:
            M.add(h + ".dmse_pct", fpct(dm_))
            M.add(h + ".dwape_pct", fpct(dw_))
        if m == ref and np.isfinite(dm_):
            by_variant[(bb, var)].append((dm_, dw_))
            if bb == head_bb:
                by_variant_head[var].append((dm_, dw_))
        n_abl += 1
    for (bb, var), v in by_variant.items():
        M.add(f"{prefix}abl.{bb}.{var}.mean_dmse_pct", fpct(np.mean([a for a, _ in v])))
        M.add(f"{prefix}abl.{bb}.{var}.max_dmse_pct", fpct(np.max([a for a, _ in v])))
        M.add(f"{prefix}abl.{bb}.{var}.min_dmse_pct", fpct(np.min([a for a, _ in v])))
        M.add(f"{prefix}abl.{bb}.{var}.mean_dwape_pct", fpct(np.mean([b for _, b in v])))
        M.add(f"{prefix}abl.{bb}.{var}.n_cells", str(len(v)))
    for var, v in by_variant_head.items():
        M.add(f"{prefix}abl.{var}.mean_dmse_pct", fpct(np.mean([a for a, _ in v])))
        M.add(f"{prefix}abl.{var}.max_dmse_pct", fpct(np.max([a for a, _ in v])))
        M.add(f"{prefix}abl.{var}.min_dmse_pct", fpct(np.min([a for a, _ in v])))
        M.add(f"{prefix}abl.{var}.mean_dwape_pct", fpct(np.mean([b for _, b in v])))
        M.add(f"{prefix}abl.{var}.n_cells", str(len(v)))
    if n_runs:
        M.add(f"{prefix}abl.n_runs", str(n_runs))
        M.add(f"{prefix}abl.n_cells", str(n_abl))
        M.add(f"{prefix}abl.n_backbones", str(len({k[4] for k in abl})))
    summary.append(f"  ablation cells: {n_abl} ({len(by_variant)} variant x backbone), {n_runs} runs")

    # ---- weather pathway (v1 PLANETX _exo-<mode>) vs the reference headline cell
    exo = defaultdict(list)
    for r in recs:
        if r["_kind"] == "exo" and (r["_regime"] == regime or r["_regime"] is None and regime == "v1"):
            exo[(r["dataset"], r["H"], r["_variant"])].append(r)
    exo_gain = defaultdict(list)
    for (d, H, mode), rs in sorted(exo.items(), key=lambda x: (x[0][0], hsort(x[0][1]), x[0][2])):
        base = agg.get((d, ref, H))
        w, _, _ = seed_stats(rs, "wape")
        mse, _, _ = seed_stats(rs, "mse")
        pk, _, _ = seed_stats(rs, "peak_mape")
        head = f"{prefix}exo.{d}.{mode}.H{H}"
        M.add(head + ".wape", fnum(w, 2))
        M.add(head + ".mse", fnum(mse, 4))
        M.add(head + ".peak", fnum(pk, 2))
        if base is not None:
            g = gain(w, base["wape"])
            M.add(head + ".gain_wape_pct", fpct(g))
            if np.isfinite(g):
                exo_gain[mode].append(g)
    for mode, v in exo_gain.items():
        M.add(f"{prefix}exo.{mode}.mean_gain_wape_pct", fpct(np.mean(v)))

    # ---- per-window losses for DM and MCS
    errs = {}
    for d, H in cell_keys:
        e = {}
        for (dd, m, HH), rs in cells.items():
            if dd == d and HH == H:
                a = seed_mean_err(rs)
                if a is not None:
                    e[m] = a
        if len(e) >= 2:
            n = Counter(len(a) for a in e.values()).most_common(1)[0][0]
            errs[(d, H)] = {m: a for m, a in e.items() if len(a) == n}

    # ---- Diebold-Mariano with Holm within the chosen family
    # HAC (Bartlett, lag max(n^(1/3), H-1)) and stationary-bootstrap (block max(n^(1/3), H),
    # the MCS scheme) variants of the test; the effective sample size of the differential under
    # the HAC long-run variance is reported per cell
    fam_p, fam_pb = defaultdict(list), defaultdict(list)       # family key -> [(d, m, H, p)]
    dm_stat, dm_boot, neff = {}, {}, defaultdict(list)
    for (d, H), e in errs.items():
        if ref not in e:
            continue
        n = len(e[ref])
        bl = max(1, int(round(n ** (1 / 3))), int(H))
        for m, a in e.items():
            if m == ref:
                continue
            s, p = stats.dm_test(e[ref], a, h=int(H))
            dm_stat[(d, m, H)] = (s, p)
            sb, pb = stats.dm_bootstrap(e[ref], a, h=int(H), B=args.dm_B, block_len=bl, seed=0)
            dm_boot[(d, m, H)] = (sb, pb)
            neff[(d, H)].append(stats.effective_n(e[ref] - a, lag=max(1, int(round(n ** (1 / 3))), int(H) - 1)))
            fam = {"cell": (d, H), "dataset": d, "all": "all"}[args.holm_family]
            fam_p[fam].append((d, m, H, p))
            fam_pb[fam].append((d, m, H, pb))
    p_holm, pb_holm = {}, {}
    for store, fam_d in ((p_holm, fam_p), (pb_holm, fam_pb)):
        for fam, items in fam_d.items():
            adj = stats.holm([p for _, _, _, p in items])
            for (d, m, H, _), q in zip(items, adj):
                store[(d, m, H)] = float(q)
    for (d, H), v in neff.items():
        M.add(f"{prefix}dm.{d}.H{H}.n", fint(len(errs[(d, H)][ref])))
        M.add(f"{prefix}dm.{d}.H{H}.n_eff_median", fint(np.nanmedian(v)))
    if neff:
        allv = [x for v in neff.values() for x in v if np.isfinite(x)]
        M.add(f"{prefix}dm.n_eff_median", fint(np.median(allv)))
        M.add(f"{prefix}dm.n_eff_min", fint(np.min(allv)))
    COUNT_KEYS = ("wins", "ties", "losses", "wins_raw", "ties_raw", "losses_raw",
                  "wins_boot", "ties_boot", "losses_boot")
    counts = defaultdict(lambda: Counter())
    stat_list = defaultdict(list)
    for (d, m, H), (s, p) in sorted(dm_stat.items(), key=lambda x: (x[0][0], x[0][1], hsort(x[0][2]))):
        q = p_holm[(d, m, H)]
        sb, pb = dm_boot[(d, m, H)]
        qb = pb_holm[(d, m, H)]
        M.add(f"{prefix}dm.{d}.{m}.H{H}.stat", fnum(s, 2))
        M.add(f"{prefix}dm.{d}.{m}.H{H}.p", fp(p))
        M.add(f"{prefix}dm.{d}.{m}.H{H}.p_holm", fp(q))
        M.add(f"{prefix}dm.{d}.{m}.H{H}.stat_boot", fnum(sb, 2))
        M.add(f"{prefix}dm.{d}.{m}.H{H}.p_boot", fp(pb))
        M.add(f"{prefix}dm.{d}.{m}.H{H}.p_boot_holm", fp(qb))
        if not np.isfinite(p):
            continue
        stat_list[m].append(s)
        for suffix, pv in (("", q), ("_raw", p), ("_boot", qb)):
            if pv < 0.05 and s < 0:
                counts[m]["wins" + suffix] += 1
            elif pv < 0.05 and s > 0:
                counts[m]["losses" + suffix] += 1
            else:
                counts[m]["ties" + suffix] += 1
    # per-seed agreement: a Holm verdict of the seed-mean test counts as confirmed when every
    # seed's own test (raw p < 0.05) has the same sign
    agree = defaultdict(Counter)
    for (d, m, H), (s, p) in dm_stat.items():
        q = p_holm[(d, m, H)]
        if not (np.isfinite(q) and q < 0.05):
            continue
        er = {r["_seed"]: load_err(r) for r in keep_seeds(cells.get((d, ref, H), []))}
        em = {r["_seed"]: load_err(r) for r in keep_seeds(cells.get((d, m, H), []))}
        seeds = [sd for sd in er if sd in em and er[sd] is not None and em[sd] is not None
                 and len(er[sd]) == len(em[sd])]
        if len(seeds) < 2:
            continue
        verdicts = [stats.dm_test(er[sd], em[sd], h=int(H)) for sd in seeds]
        ok = all(np.isfinite(pp) and pp < 0.05 and np.sign(ss) == np.sign(s) for ss, pp in verdicts)
        side = "wins" if s < 0 else "losses"
        agree[m][side + "_allseeds"] += int(ok)
        agree[m][side + "_tested"] += 1
    total = Counter()
    for m, c in counts.items():
        for k in COUNT_KEYS:
            M.add(f"{prefix}dm.{m}.{k}", str(c[k]))
            total[k] += c[k]
        for k in ("wins_allseeds", "wins_tested", "losses_allseeds", "losses_tested"):
            M.add(f"{prefix}dm.{m}.{k}", str(agree[m][k]))
            total[k] += agree[m][k]
        M.add(f"{prefix}dm.{m}.n_cells", str(len(stat_list[m])))
        M.add(f"{prefix}dm.{m}.median_stat", fnum(np.median(stat_list[m]), 2))
    if counts:
        for k in COUNT_KEYS + ("wins_allseeds", "wins_tested", "losses_allseeds", "losses_tested"):
            M.add(f"{prefix}dm.total.{k}", str(total[k]))
        M.add(f"{prefix}dm.n_tests", str(len(dm_stat)))
        M.add(f"{prefix}dm.holm_family", args.holm_family)
        M.add(f"{prefix}dm.B", str(args.dm_B))
    summary.append(f"  DM tests: {len(dm_stat)} (Holm family = {args.holm_family}); "
                   f"totals {dict(total)}")

    # ---- Model Confidence Set per (dataset, H)
    if not args.no_mcs:
        n_in = defaultdict(Counter)
        n_cells = Counter()
        mcs_cells = {}
        n_in_h = defaultdict(Counter)       # scope -> (model, H) -> cells retained
        set_sizes = defaultdict(list)       # (scope, H) -> set sizes
        for (d, H), e in errs.items():
            names = sorted(e)
            L = np.stack([e[m] for m in names], axis=1)
            n = L.shape[0]
            bl = max(1, int(round(n ** (1 / 3))))
            if args.mcs_block == "horizon":
                bl = max(bl, int(H))
            res = stats.mcs(L, alpha=args.mcs_alpha, B=args.mcs_B, block_len=bl, seed=0)
            members = [names[i] for i in np.argsort(L.mean(0)) if res["in_set"][i]]
            for scope in scopes_of(d):
                set_sizes[(scope, H)].append(len(members))
            mcs_cells[d] = mcs_cells.get(d, 0) + 1
            for i, m in enumerate(names):
                M.add(f"{prefix}mcs.{d}.H{H}.{m}.p", fp(res["p_values"][i]))
                M.add(f"{prefix}mcs.{d}.H{H}.{m}.in", "1" if res["in_set"][i] else "0")
                n_cells[m] += 1
                if res["in_set"][i]:
                    for scope in scopes_of(d):
                        n_in[scope][m] += 1
                        n_in_h[scope][(m, H)] += 1
            M.add(f"{prefix}mcs.{d}.H{H}.n_in", str(len(members)))
            M.add(f"{prefix}mcs.{d}.H{H}.members", ", ".join(members))
            M.add(f"{prefix}mcs.{d}.H{H}.block_len", str(res["block_len"]))
        for m in n_cells:
            M.add(f"{prefix}mcs.{m}.n_in", str(n_in["all"][m]))
            M.add(f"{prefix}mcs.{m}.n_cells", str(n_cells[m]))
            for scope in ("grid", "bench", "ett") + tuple(mcs_cells):
                M.add(f"{prefix}mcs.{scope}.{m}.n_in", str(n_in[scope][m]))
        for (scope, H), v in set_sizes.items():        # set sizes by horizon (Sec. 3.3)
            M.add(f"{prefix}mcs.{scope}.H{H}.size_min", str(min(v)))
            M.add(f"{prefix}mcs.{scope}.H{H}.size_max", str(max(v)))
            M.add(f"{prefix}mcs.{scope}.H{H}.size_mean", fnum(np.mean(v), 1))
            M.add(f"{prefix}mcs.{scope}.H{H}.n_cells", str(len(v)))
        for scope, c in n_in_h.items():
            for (m, H), k in c.items():
                M.add(f"{prefix}mcs.{scope}.{m}.H{H}.n_in", str(k))
        for scope in ("grid", "bench", "ett"):
            M.add(f"{prefix}mcs.{scope}.n_cells", str(sum(v for d, v in mcs_cells.items() if scope in scopes_of(d))))
        for d, v in mcs_cells.items():
            M.add(f"{prefix}mcs.{d}.n_cells", str(v))
        if errs:
            M.add(f"{prefix}mcs.alpha", fnum(args.mcs_alpha, 2))
            M.add(f"{prefix}mcs.B", str(args.mcs_B))
            M.add(f"{prefix}mcs.n_cells", str(len(errs)))
        summary.append(f"  MCS cells: {len(errs)} (alpha={args.mcs_alpha}, B={args.mcs_B}, block={args.mcs_block})")
    return agg


def emit_extras(M, args, summary, head="R2", agg=None, recs=None):
    rdir = args.results
    agg = agg or {}
    # day-ahead vs TSO: the 11:00 UTC origin (dayahead.json -> da.*) and the 08:00 UTC origin
    # (dayahead_08.json -> dayahead_08.*), each with its own aggregates over the systems
    da_files = [p for p in [os.path.join(rdir, "dayahead.json"), os.path.join(rdir, "dayahead_08.json")]
                + list(args.extra or []) if os.path.exists(p)]
    for p in da_files:
        try:
            da = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        pre = "da" if stem == "dayahead" else stem.replace(".", "_")
        da_agg = defaultdict(list)
        da_sig = defaultdict(list)                  # model -> [(system, ensemble significantly better)]
        da_split = defaultdict(list)                # (model, hol|lock|xlock) -> [(system, model WAPE, TSO WAPE)]
        da_refit = defaultdict(list)                # "<m>-refit" -> [(system, refit WAPE, once-trained WAPE)]
        tso_bias = {}                               # system -> published operator bias (%)
        if not isinstance(da, dict):
            continue
        for d, c in da.items():
            if not isinstance(c, dict):
                continue
            if all(isinstance(v, (int, float)) for v in c.values()):        # tso_baseline-style
                for f, (k, nd) in DA_FIELDS.items():
                    if f in c:
                        M.add(f"{pre}.{d}.{k}", _daf(c[f], nd))
                continue
            proto = c.get("_protocol", {})
            if proto:
                M.add(f"{pre}.{d}.n_days", fint(proto.get("n_days")))
                lh = proto.get("lead_hours")
                if lh:
                    M.add(f"{pre}.{d}.lead_lo", str(lh[0]))
                    M.add(f"{pre}.{d}.lead_hi", str(lh[1]))
            tso = c.get("TSO", {})
            if tso and isinstance(tso.get("bias_pct"), (int, float)):
                tso_bias[d] = float(tso["bias_pct"])
            if tso:
                for f, (k, nd) in DA_FIELDS.items():
                    if f in tso:
                        M.add(f"{pre}.{d}.TSO.{k}", _daf(tso[f], nd))
                        if pre == "da":
                            M.add(f"tso.{d}.{k}", _daf(tso[f], nd))
            groups = defaultdict(list)                      # (model, regime) -> [seed entries]
            for m, v in c.items():
                if m.startswith("_") or not isinstance(v, dict) or m == "TSO" or m == "TSO_adj":
                    continue
                if "model" in v:
                    mm = V2_TAG_RE.search(str(v.get("suffix") or "").strip("_"))
                    groups[(v["model"], mm.group(1) if mm else None)].append(v)
                else:
                    groups[(m, None)].append(v)
            best, bw = None, float("inf")
            for (m, rg), vs in groups.items():
                pfx = f"{pre}." + ("" if rg in (None, head) else f"{rg}.") + f"{d}.{m}"
                v = {f: float(np.mean([x[f] for x in vs if isinstance(x.get(f), (int, float))]))
                     for f in DA_FIELDS if any(isinstance(x.get(f), (int, float)) for x in vs)}
                ens = len(vs) == 1 and bool(vs[0].get("n_members"))
                if not ens:                         # a DM p-value is not averaged over seeds
                    v.pop("dm_p", None); v.pop("dm_stat", None)
                    nsig = sum(1 for x in vs if isinstance(x.get("dm_p"), (int, float))
                               and x["dm_p"] < 0.05 and float(x.get("dm_stat") or 0.0) < 0)
                    if any(isinstance(x.get("dm_p"), (int, float)) for x in vs):
                        M.add(f"{pfx}.n_seeds_sig_better", str(nsig))
                for f, (k, nd) in DA_FIELDS.items():
                    if f in v:
                        M.add(f"{pfx}.{k}", _daf(v[f], nd))
                M.add(f"{pfx}.n_seeds", str(len(vs)))
                if ens and "dm_p" in v:
                    sig = bool(np.isfinite(v["dm_p"]) and v["dm_p"] < 0.05 and v["dm_stat"] < 0)
                    M.add(f"{pfx}.sig_better", "1" if sig else "0")
                    if rg in (None, head) and is_grid(d):
                        da_sig[m].append((d, sig))
                if tso and "wape" in v:
                    g = gain(v["wape"], tso.get("wape"))
                    M.add(f"{pfx}.gain_wape_pct", fpct(g))
                    if "peak_mape" in v and "peak_mape" in tso:
                        M.add(f"{pfx}.gain_peak_pct", fpct(gain(v["peak_mape"], tso["peak_mape"])))
                    tsoa = c.get("TSO_adj", {})
                    if tsoa and "wape" in tsoa and m != "TSO_adj":
                        M.add(f"{pfx}.gain_wape_adj_pct", fpct(gain(v["wape"], tsoa["wape"])))
                    if rg in (None, head) and is_grid(d) and m != "TSO_adj":
                        da_agg[m].append((d, v["wape"], tso["wape"], g, tsoa.get("wape", tso["wape"])))
                        for fld, tag in (("wape_holiday", "hol"), ("wape_lockdown", "lock"),
                                         ("wape_excl_lockdown", "xlock")):
                            if isinstance(v.get(fld), (int, float)) and isinstance(tso.get(fld), (int, float)):
                                da_split[(m, tag)].append((d, float(v[fld]), float(tso[fld])))
                if rg in (None, head) and fin(v.get("wape")) < bw:
                    best, bw = m, fin(v.get("wape"))
            # the monthly-refit complement against the once-trained model: the run of the same
            # seed where it exists (the refit uses one seed), else the seed mean
            for (m, rg), vs in groups.items():
                if not m.endswith("-refit") or rg not in (None, head):
                    continue
                base = groups.get((m[:-len("-refit")], rg))
                wr = [x["wape"] for x in vs if isinstance(x.get("wape"), (int, float))]
                if not base or not wr:
                    continue
                seeds = {x.get("seed") for x in vs}
                same = [x["wape"] for x in base if x.get("seed") in seeds and isinstance(x.get("wape"), (int, float))]
                wb = [x["wape"] for x in base if isinstance(x.get("wape"), (int, float))]
                if not wb:
                    continue
                w_once = float(np.mean(same if same else wb))
                dw = pct_change(float(np.mean(wr)), w_once)
                M.add(f"{pre}.{d}.{m}.dwape_vs_once_pct", fpct(dw))
                if is_grid(d) and np.isfinite(dw):
                    da_refit[m].append((d, float(np.mean(wr)), w_once))
            if best:
                M.add(f"{pre}.{d}.best", best)
        if da_agg:
            systems = sorted({d for v in da_agg.values() for d, _, _, _, _ in v})
            M.add(f"{pre}.grid.n_systems", str(len(systems)))
            won, won_adj = set(), set()
            for m, v in da_agg.items():
                M.add(f"{pre}.grid.{m}.n_systems", str(len(v)))
                M.add(f"{pre}.grid.{m}.n_beats_tso", str(sum(1 for _, w, t, _, _ in v if w < t)))
                M.add(f"{pre}.grid.{m}.n_beats_tso_adj", str(sum(1 for _, w, _, _, ta in v if w < ta)))
                M.add(f"{pre}.grid.{m}.mean_gain_wape_pct", fpct(np.mean([g for _, _, _, g, _ in v])))
                M.add(f"{pre}.grid.{m}.mean_gain_wape_adj_pct", fpct(np.mean([gain(w, ta) for _, w, _, _, ta in v])))
                M.add(f"{pre}.grid.{m}.mean_wape", fnum(np.mean([w for _, w, _, _, _ in v]), 2))
                won |= {d for d, w, t, _, _ in v if w < t}
                won_adj |= {d for d, w, _, _, ta in v if w < ta}
            M.add(f"{pre}.grid.any.n_beats_tso", str(len(won)))
            M.add(f"{pre}.grid.any.n_beats_tso_adj", str(len(won_adj)))
            # the systems whose published operator series is unbiased on the scored days
            # (|bias| <= DA_BIAS_TOL %); the comparison with the operators is read on these
            unb = {d for d in systems if abs(tso_bias.get(d, 0.0)) <= DA_BIAS_TOL}
            M.add(f"{pre}.grid.bias_tol_pct", r"%g\%%" % DA_BIAS_TOL)
            M.add(f"{pre}.grid.n_unbiased", str(len(unb)))
            M.add(f"{pre}.grid.biased_systems", ", ".join(d.replace("OPSD-", "") for d in systems if d not in unb))
            for m, v in da_agg.items():
                vu = [t for t in v if t[0] in unb]
                M.add(f"{pre}.grid.{m}.n_beats_tso_unbiased", str(sum(1 for _, w, t, _, _ in vu if w < t)))
                if vu:
                    M.add(f"{pre}.grid.{m}.mean_gain_wape_unbiased_pct", fpct(np.mean([g for _, _, _, g, _ in vu])))
            M.add(f"{pre}.grid.any.n_beats_tso_unbiased", str(len({d for d in won if d in unb})))
            for m, v in da_refit.items():               # refit against once-trained, unbiased systems
                vu = [(d, wr, wo) for d, wr, wo in v if d in unb]
                if not vu:
                    continue
                dws = [pct_change(wr, wo) for _, wr, wo in vu]
                M.add(f"{pre}.grid.{m}.n_systems_vs_once_unbiased", str(len(vu)))
                M.add(f"{pre}.grid.{m}.mean_dwape_vs_once_unbiased_pct", fpct(np.mean(dws)))
                M.add(f"{pre}.grid.{m}.min_dwape_vs_once_unbiased_pct", fpct(np.min(dws)))
                M.add(f"{pre}.grid.{m}.max_dwape_vs_once_unbiased_pct", fpct(np.max(dws)))
                M.add(f"{pre}.grid.{m}.n_better_than_once_unbiased", str(sum(1 for x in dws if x < 0)))
            for m, v in da_sig.items():
                M.add(f"{pre}.grid.{m}.n_sig_beats_tso", str(sum(1 for _, s in v if s)))
                M.add(f"{pre}.grid.{m}.n_sig_beats_tso_unbiased", str(sum(1 for d, s in v if s and d in unb)))
            for (m, tag), v in da_split.items():        # holiday / lockdown splits on the unbiased systems
                vu = [x for x in v if x[0] in unb]
                if not vu:
                    continue
                M.add(f"{pre}.grid.{m}.n_beats_tso_{tag}_unbiased", str(sum(1 for _, w, tw in vu if w < tw)))
                M.add(f"{pre}.grid.{m}.n_systems_{tag}_unbiased", str(len(vu)))
                M.add(f"{pre}.grid.{m}.mean_wape_{tag}_unbiased", fnum(np.mean([w for _, w, _ in vu]), 2))
                if m == "PLANET":
                    M.add(f"{pre}.grid.tso.mean_wape_{tag}_unbiased", fnum(np.mean([tw for _, _, tw in vu]), 2))
            M.add(f"{pre}.grid.tso.mean_wape", fnum(np.mean([np.mean([t for _, _, t, _, _ in v]) for v in da_agg.values()]), 2))
            M.add(f"{pre}.grid.tso_adj.mean_wape", fnum(np.mean([np.mean([ta for _, _, _, _, ta in v]) for v in da_agg.values()]), 2))
        summary.append(f"  extra: {os.path.relpath(p, HERE)}")
    # TSO baseline file
    p = os.path.join(rdir, "tso_baseline.json")
    if os.path.exists(p):
        try:
            t = json.load(open(p, encoding="utf-8"))
            for d, c in t.items():
                for f, (k, nd) in DA_FIELDS.items():
                    if f in c:
                        M.add(f"tso.{d}.{k}", _daf(c[f], nd))
            summary.append(f"  extra: {os.path.relpath(p, HERE)}")
        except Exception:
            pass
    # probabilistic: per (dataset, model, H) seed-averaged, then aggregates over the systems
    PROB_ND = {"picp": 1, "ace": 2, "crps": 0, "pinball": 0, "mpiw": 0, "winkler": 0, "mae": 0, "wape": 2}
    pgroups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(rdir, "prob", "*.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if {"dataset", "model", "H"} <= set(r):
            pgroups[(r["dataset"], r["model"], r["H"])].append(r)
    SKIP = ("dataset", "model", "H", "L", "seed", "lr", "quantiles", "tag")
    pstat = {}
    for (d, m, H), rs in pgroups.items():
        v = {}
        for k in {k for r in rs for k in r if k not in SKIP and isinstance(r[k], (int, float))}:
            v[k] = float(np.mean([r[k] for r in rs if isinstance(r.get(k), (int, float))]))
            nd = next((n for pfx, n in PROB_ND.items() if k.startswith(pfx)), 3)
            M.add(f"prob.{d}.{m}.H{H}.{k}", fnum(v[k], nd))
        M.add(f"prob.{d}.{m}.H{H}.n_seeds", str(len(rs)))
        pstat[(d, m, H)] = v
    if pstat:
        summary.append(f"  probabilistic cells: {len(pstat)}")
        # look-ahead diagnostic: the immediate-feedback update (_aci0, the first-pass defect) against
        # the operational delayed-feedback update (_aci), over the grid cells that carry both
        both = [v for k, v in pstat.items() if is_grid(k[0]) and "picp_aci0" in v and "picp_aci" in v]
        if both:
            M.add("prob.grid.lookahead_gap_pp", fnum(np.mean([v["picp_aci0"] - v["picp_aci"] for v in both]), 1))
            M.add("prob.grid.lookahead_width_pct",
                  fpct(np.mean([pct_change(v["mpiw_aci0"], v["mpiw_aci"]) for v in both if "mpiw_aci0" in v and "mpiw_aci" in v])))
            M.add("prob.grid.lookahead_winkler_pct",
                  fpct(np.mean([pct_change(v["winkler_aci0"], v["winkler_aci"]) for v in both if "winkler_aci0" in v and "winkler_aci" in v])))
            M.add("prob.grid.n_lookahead_cells", str(len(both)))
        for lev in (80, 50):
            for nm in ("raw", "cqr", "aci"):
                key = f"picp{lev}_{nm}"
                vals = [v[key] for k, v in pstat.items() if is_grid(k[0]) and key in v]
                if vals:
                    M.add(f"prob.grid.mean_{key}", fnum(np.mean(vals), 1))
                    M.add(f"prob.grid.min_{key}", fnum(np.min(vals), 1))
                    M.add(f"prob.grid.max_{key}", fnum(np.max(vals), 1))
        AGG = ["crps_cqr", "pinball_cqr", "picp_raw", "picp_cqr", "picp_aci", "mpiw_aci", "winkler_aci", "crps_aci"]
        for Hsel in sorted({H for _, _, H in pstat}) + [None]:
            keys = [k for k in pstat if is_grid(k[0]) and (Hsel is None or k[2] == Hsel)]
            if not keys:
                continue
            tag = "prob.grid." + (f"H{Hsel}." if Hsel is not None else "")
            systems = sorted({k[0] for k in keys})
            M.add(tag + "n_systems", str(len(systems)))
            nbest = Counter()
            for d in systems:
                for H in sorted({k[2] for k in keys if k[0] == d}):
                    cand = {k[1]: pstat[k].get("crps_cqr", NAN) for k in keys if k[0] == d and k[2] == H}
                    cand = {m: c for m, c in cand.items() if np.isfinite(c)}
                    if cand:
                        nbest[min(cand, key=cand.get)] += 1
            for m in sorted({k[1] for k in keys}):
                rows = [pstat[k] for k in keys if k[1] == m]
                M.add(tag + f"{m}.n_cells", str(len(rows)))
                M.add(tag + f"{m}.n_best_crps", str(nbest[m]))
                for f in AGG:
                    vals = [r[f] for r in rows if f in r]
                    if vals:
                        nd = next((n for pfx, n in PROB_ND.items() if f.startswith(pfx)), 3)
                        M.add(tag + f"{m}.mean_{f}", fnum(np.mean(vals), nd))
            for f in ("picp_aci", "picp_cqr", "picp_raw"):
                vals = [pstat[k][f] for k in keys if f in pstat[k]]
                if vals:
                    M.add(tag + f"min_{f}", fnum(np.min(vals), 1))
                    M.add(tag + f"max_{f}", fnum(np.max(vals), 1))
    # zero-shot foundation models against the best learned model of the same cell
    fmdir = args.fm if args.fm is not None else os.path.join(rdir, "fm")
    fmrecs = [r for r in load_records(rdir, fmdir) if r["_src"] == "fm" or r.get("zero_shot")]
    if fmrecs and agg:
        best = {}
        for (d, m, H), row in agg.items():
            w = fin(row.get("wape"))
            if np.isfinite(w) and (d, H) not in best or (np.isfinite(w) and w < best[(d, H)][1]):
                best[(d, H)] = (m, w)
        fg = defaultdict(list)
        for r in fmrecs:
            fg[(r["model"], r["_regime"], r.get("L"), r["dataset"], r["H"])].append(r)
        fagg = defaultdict(list)
        for (m, rg, L, d, H), rs in fg.items():
            w = seed_stats(rs, "wape")[0]
            pfx = "fm." + ("" if rg in (None, head) else f"{rg}.") + ("L1024." if L == 1024 else "") + f"{d}.{m}.H{H}"
            M.add(pfx + ".wape", fnum(w, 2))
            M.add(pfx + ".peak", fnum(seed_stats(rs, "peak_mape")[0], 2))
            if (d, H) in best:
                M.add(pfx + ".best_learned", best[(d, H)][0])
                M.add(pfx + ".dwape_vs_best_pct", fpct(pct_change(w, best[(d, H)][1])))
                if is_grid(d):
                    fagg[(m, rg, L, H)].append(pct_change(w, best[(d, H)][1]))
                    fagg[(m, rg, L, None)].append(pct_change(w, best[(d, H)][1]))
        for (m, rg, L, H), v in fagg.items():
            pfx = "fm.grid." + ("" if rg in (None, head) else f"{rg}.") + ("L1024." if L == 1024 else "") + m + (f".H{H}" if H is not None else "")
            M.add(pfx + ".n_cells", str(len(v)))
            M.add(pfx + ".n_beats_best", str(sum(1 for x in v if x < 0)))
            M.add(pfx + ".mean_dwape_vs_best_pct", fpct(np.mean(v)))
        summary.append(f"  foundation-model records: {len(fmrecs)}")
    # cost overhead of the reference over its own backbone on the cost cell
    if agg:
        d, H = (getattr(args, "cost_cell", None) or "OPSD-DE:96").split(":")
        H = int(H)
        row = agg.get((d, args.ref, H))
        if row:
            bbm = "NBEATS" if row.get("backbone") == "basis" else "PatchTST"
            base = agg.get((d, bbm, H))
            M.add("cost.backbone", bbm)
            if base:
                for f in ("train_s", "infer_s", "params_k"):
                    M.add(f"cost.{args.ref}_vs_backbone.{f.replace('_s', '').replace('_k', '')}_pct",
                          fpct(pct_change(row.get(f), base.get(f))))
            # parameters of the tier itself: full model minus the backbone-alone ablation on the
            # headline backbone (same width, same head); fallback: the headline baseline
            full = [r for r in (recs or []) if r["_kind"] == "abl" and r["_regime"] == head and r["model"] == args.ref
                    and r["dataset"] == d and r["H"] == H and r["_variant"] == "full"
                    and (r.get("_backbone") or r.get("backbone")) == row.get("backbone")]
            nopl = [r for r in (recs or []) if r["_kind"] == "abl" and r["_regime"] == head and r["model"] == args.ref
                    and r["dataset"] == d and r["H"] == H and r["_variant"] == "no_pl"
                    and (r.get("_backbone") or r.get("backbone")) == row.get("backbone")]
            if full and nopl and full[0].get("params") and nopl[0].get("params"):
                M.add("cost.tier.params_k", fnum((full[0]["params"] - nopl[0]["params"]) / 1e3, 1))
                M.add("cost.tier.params_source", "ablation")
            elif base and np.isfinite(fin(row.get("params_k"))) and np.isfinite(fin(base.get("params_k"))):
                M.add("cost.tier.params_k", fnum(row["params_k"] - base["params_k"], 1))
                M.add("cost.tier.params_source", "headline")
            # training-time ratio of the reduced-budget models to PatchTST on the cost cell
            try:
                import run_all_v2 as _r
                pt = agg.get((d, "PatchTST", H))
                ratios = [agg[(d, m, H)]["train_s"] / pt["train_s"] for m in _r.EXPENSIVE
                          if (d, m, H) in agg and pt and np.isfinite(fin(agg[(d, m, H)].get("train_s")))
                          and np.isfinite(fin(pt.get("train_s"))) and pt["train_s"] > 0]
                if ratios:
                    M.add("cost.expensive.ratio_min", fnum(min(ratios), 0))
                    M.add("cost.expensive.ratio_max", fnum(max(ratios), 0))
                    M.add("cost.expensive.n", str(len(ratios)))
            except Exception:                        # noqa: BLE001
                pass
    # corpus metadata
    ddir = args.data or (os.path.join(HERE, "data", "proc_v2") if os.path.isdir(os.path.join(HERE, "data", "proc_v2"))
                         else os.path.join(HERE, "data", "proc"))
    n_data = 0
    for p in sorted(glob.glob(os.path.join(ddir, "*.json"))):
        try:
            m = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "T" not in m or "splits" not in m:
            continue
        d = m.get("name", os.path.splitext(os.path.basename(p))[0])
        M.add(f"data.{d}.T", fint(m["T"]))
        M.add(f"data.{d}.C", str(m.get("C", "--")))
        M.add(f"data.{d}.freq", str(m.get("freq", "--")))
        M.add(f"data.{d}.start", str(m.get("start", "--"))[:10])
        M.add(f"data.{d}.end", str(m.get("end", "--"))[:10])
        for k in ("train", "val", "test"):
            a, b = m["splits"].get(k, (0, 0))
            M.add(f"data.{d}.n_{k}", fint(b - a))
        # weather sensitivity of the load on the training split (grids with a temperature
        # channel): correlation of daily-mean load with daily-mean temperature, and the ratio
        # of mean winter (DJF) to mean summer (JJA) load
        npz = os.path.splitext(p)[0] + ".npz"
        chans = m.get("channels") or []
        if os.path.exists(npz) and "temp" in chans and m.get("target") in chans and str(d).startswith("OPSD"):
            try:
                import datetime as _dt
                z = np.load(npz)
                X = z["data"]
                tr0, tr1 = m["splits"]["train"]
                n = (tr1 - tr0) // 24 * 24
                dl = X[tr0:tr0 + n, chans.index(m["target"])].reshape(-1, 24).mean(1)
                dtp = X[tr0:tr0 + n, chans.index("temp")].reshape(-1, 24).mean(1)
                M.add(f"data.{d}.corr_temp", fnum(float(np.corrcoef(dl, dtp)[0, 1]), 2))
                t0 = _dt.datetime.fromisoformat(str(m["start"])[:19]) + _dt.timedelta(hours=int(tr0))
                mon = np.array([(t0 + _dt.timedelta(days=i)).month for i in range(len(dl))])
                w, s_ = dl[np.isin(mon, [12, 1, 2])], dl[np.isin(mon, [6, 7, 8])]
                if len(w) and len(s_):
                    M.add(f"data.{d}.winter_summer", fnum(float(w.mean() / s_.mean()), 2))
            except Exception:                    # noqa: BLE001
                pass
        n_data += 1
    if n_data:
        summary.append(f"  datasets with metadata: {n_data} ({os.path.relpath(ddir, HERE)})")


def emit_protocol(M, recs, head, summary):
    """Protocol constants quoted in the text: model and system counts, NWP noise level at 24 h,
    size of the shared exogenous head.  Each degrades to 0 or -- when its source is absent."""
    learned, fm_models, systems = set(), set(), set()
    for r in recs:
        if is_grid(r["dataset"]):
            systems.add(r["dataset"])
        if r["_kind"] not in ("headline", "fm-long") or r["_regime"] != head:
            continue
        if r["_src"] == "fm" or r.get("zero_shot"):
            fm_models.add(r["model"])
        else:                                   # once / refit are one statistical model; +AR errors another
            learned.add("Tao" if r["model"] in ("TaoVanilla", "TaoRefit") else r["model"])
    M.add("n.models.learned", str(len(learned)))
    M.add("n.models.fm", str(len(fm_models)))
    M.add("n.systems", str(len(systems)))
    try:                                        # structural counts from the driver
        import run_all_v2 as _r
        M.add("proto.n_v1_models", str(len(_r.V1_MODELS)))
        M.add("proto.n_nf_models", str(len(_r.NF_MODELS)))
        M.add("proto.n_neural", str(len(_r.ALL_MODELS)))
        M.add("proto.n_expensive", str(len(_r.EXPENSIVE)))
        M.add("proto.n_headline", str(len(_r.HEADLINE)))
        M.add("proto.n_nosize", str(len(set(_r.exp.NO_SIZE) & set(_r.ALL_MODELS))))
    except Exception as e:                      # noqa: BLE001
        summary.append(f"  driver counts unavailable: {e}")
    try:
        import data_v2
        M.add("proto.nwp.temp_rmse", fnum(data_v2.NWP_ABS_24H["temp"], 1))
        M.add("proto.nwp.wspd_ms", fnum(data_v2.WSPD_MS_24H, 1))
        M.add("proto.nwp.rhum", fnum(data_v2.NWP_ABS_24H["rhum"], 0))
        M.add("proto.nwp.rad", fnum(data_v2.NWP_ABS_24H["rad"], 0))
        M.add("proto.nwp.wind_cap", fnum(100 * data_v2.NWP_REL_CAP_24H["wind"], 0) + r"\%")
        M.add("proto.nwp.solar_cap", fnum(100 * data_v2.NWP_REL_CAP_24H["solar"], 0) + r"\%")
        M.add("proto.nwp.cap_pctl", fnum(getattr(data_v2, "NWP_CAP_PCTL", 99.9), 1))
    except Exception as e:                      # noqa: BLE001
        M.add("proto.nwp.temp_rmse", "--")
        summary.append(f"  proto.nwp.temp_rmse unavailable: {e}")
    try:
        import exo
        n = sum(p.numel() for p in exo.ExoHead(E=14, L=336, H=24).parameters())
        M.add("proto.head.params_k", fnum(n / 1e3, 1))
    except Exception as e:                      # noqa: BLE001
        M.add("proto.head.params_k", "--")
        summary.append(f"  proto.head.params_k unavailable: {e}")
    summary.append(f"  protocol: {len(learned)} learned models, {len(fm_models)} zero-shot, "
                   f"{len(systems)} systems")


def generate(args):
    fm = args.fm if args.fm is not None else os.path.join(args.results, "fm")
    recs = load_records(args.results, fm)
    summary = [f"{len(recs)} result records from {os.path.relpath(args.results, HERE)}"
               + (f" + {os.path.relpath(fm, HERE)}" if os.path.isdir(fm) else "")]
    kinds = Counter((r["_kind"], r["_regime"]) for r in recs)
    summary.append("  by (kind, regime): " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=str)))
    for r in recs:
        if r["_regime"] is None and r["_src"] == "main":
            r["_regime"] = "v1"
    M = Macros()
    head = choose_regime(recs, args.regime)
    if head is None:
        print("no headline records found", file=sys.stderr)
        return 1
    summary.append(f"headline regime: {head}")
    regimes = sorted({r["_regime"] for r in recs if r["_kind"] == "headline" and r["_regime"] is not None})
    aggs = {}
    for reg in regimes:
        prefix = "" if reg == head else f"{reg}."
        aggs[reg] = emit_regime(M, recs, reg, prefix, args, summary) or {}
    # value of information between regimes (v2): WAPE change from Rb to Ra
    regs = [r for r in regimes if r != "v1"]
    for i in range(1, len(regs)):
        for j in range(i):
            ra, rb = regs[i], regs[j]
            per_m, pooled = defaultdict(list), defaultdict(list)
            for key in aggs.get(ra, {}):
                if key in aggs.get(rb, {}):
                    d, m, H = key
                    dw = pct_change(aggs[ra][key]["wape"], aggs[rb][key]["wape"])
                    M.add(f"info.{d}.{m}.H{H}.{ra}_vs_{rb}.dwape_pct", fpct(dw))
                    if np.isfinite(dw):
                        for sc in scopes_of(d)[1:]:
                            per_m[(sc, m)].append(dw)
                            if m not in BUILTIN_CAL:
                                pooled[sc].append(dw)
            for (sc, m), v in per_m.items():
                M.add(f"info.{sc}.{m}.{ra}_vs_{rb}.mean_dwape_pct", fpct(np.mean(v)))
                M.add(f"info.{sc}.{m}.{ra}_vs_{rb}.n_cells", str(len(v)))
            for sc, v in pooled.items():
                M.add(f"info.{sc}.{ra}_vs_{rb}.mean_dwape_pct", fpct(np.mean(v)))
                M.add(f"info.{sc}.{ra}_vs_{rb}.n_cells", str(len(v)))
    emit_mech(M, recs, head, aggs.get(head, {}), summary)
    emit_budget(M, recs, head, aggs.get(head, {}), summary)
    emit_revision(M, recs, head, aggs.get(head, {}), summary, aggs=aggs)
    emit_data_notes(M, args, summary)
    emit_combo(M, args, summary)
    emit_epochs(M, recs, head, summary)
    emit_extras(M, args, summary, head=head, agg=aggs.get(head, {}), recs=recs)
    emit_protocol(M, recs, head, summary)
    M.add("proto.n_diverged", str(len(DIVERGED)))
    M.add("proto.diverged_runs", ", ".join(s.replace("_", r"\_") for s in sorted(DIVERGED)) or "none")
    n_masked = sum(1 for r in recs if r.get("_masked_windows"))
    M.add("proto.n_masked_records", str(n_masked))
    summary.append(f"  diverged seeds excluded: {len(DIVERGED)} {sorted(DIVERGED)}; records with masked windows: {n_masked}")
    header = [f"GENERATED by src/paper_numbers.py from {os.path.relpath(args.results, HERE)} -- do not edit.",
              "Usage in the prose: \\input{numbers} then \\nv{KEY}.",
              f"Reference model: {args.ref}; headline regime: {head}; seeds averaged per cell."] + KEY_DOC.split("\n")
    M.write(args.out, header)
    summary.append(f"wrote {os.path.relpath(args.out, HERE)}: {len(M.d)} macros")
    print("\n".join(summary))
    return 0


# ------------------------------------------------------------------ check mode
SKIP_ENV = {"equation", "equation*", "align", "align*", "eqnarray", "algorithmic", "algorithm",
            "table", "table*", "tabular", "thebibliography"}   # figure captions are prose: scanned
STRIP_RE = re.compile(r"\\(?:nv|cite|citep|citet|ref|eqref|label|url|input|include|pageref|autoref|"
                      r"bibliography|bibliographystyle|includegraphics|graphicspath|usepackage|"
                      r"documentclass|setlength|hypersetup)\*?(?:\[[^\]]*\])?\{[^}]*\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
PCT_RE = re.compile(r"(?<![\w.])[+\-]?\d+(?:\.\d+)?\s*(?:\\,)?\s*(?:\\%|(?<!\\)%|(?:\$\\,|\\,|\s)pp\b)")
DEC_RE = re.compile(r"(?<![\w.\-])[+\-]?\d+\.\d+(?![\d.])")   # 926.8k and 3.4M are results too
CTX_BEFORE = re.compile(r"(?:section|sec\.|table|tab\.|fig\.|figure|eq\.|algorithm|version|python|cuda|"
                        r"pytorch|numpy|ieeetran|learning rate|\blr\b|dropout|weight decay|patience|"
                        r"driver|cudnn|neuralforecast|scikit-learn|pandas|scipy|"
                        r"momentum|tolerance|nominal|\\gamma\s*=|\\alpha\s*=|\\eta\s*=|"
                        r"\\epsilon\s*=|\\lambda\s*=|\\beta\s*=|\bv)\s*[~\\,{]*\$?\s*$", re.I)
CTX_AFTER = re.compile(r"^\s*(?:\\,)?\s*(?:K\b|GB|MB|TB|GHz|kV|V\b|-bit|mm\b|px\b)")
CONST = {"365.25", "8766", "0.05", "0.01", "0.1", "0.10", "0.9", "0.90", "0.95", "0.99", "0.5", "1.0",
         "0.0", "2.0", "0.001", "0.0001", "0.0005"}
PCT_CONST = {"5", "90", "95", "99", "100"}


def check_file(path, defined):
    findings, used = [], []
    env_depth = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = COMMENT_RE.sub("", raw.rstrip("\n"))
            for env in re.findall(r"\\begin\{([A-Za-z*]+)\}", line):
                if env in SKIP_ENV:
                    env_depth += 1
            for env in re.findall(r"\\end\{([A-Za-z*]+)\}", line):
                if env in SKIP_ENV:
                    env_depth = max(0, env_depth - 1)
            used += re.findall(r"\\nv\{([^}]*)\}", line)
            if env_depth > 0:
                continue
            text = STRIP_RE.sub(" ", line)
            hits, taken = [], []
            for rx in (PCT_RE, DEC_RE):
                for mt in rx.finditer(text):
                    if any(a < mt.end() and mt.start() < b for a, b in taken):
                        continue
                    tok = mt.group(0)
                    num = re.match(r"[+\-]?\d+(?:\.\d+)?", tok).group(0).lstrip("+-")
                    is_pct = rx is PCT_RE
                    if is_pct and "." not in num and num in PCT_CONST:
                        continue
                    if not is_pct and num in CONST:
                        continue
                    before, after = text[max(0, mt.start() - 40):mt.start()], text[mt.end():mt.end() + 12]
                    if CTX_BEFORE.search(before) or (not is_pct and CTX_AFTER.match(after)):
                        continue
                    taken.append((mt.start(), mt.end()))
                    hits.append(tok.strip())
            if hits:
                findings.append((ln, hits, text.strip()))
    return findings, used


def check(args):
    paper = args.paper
    files = sorted(glob.glob(os.path.join(paper, args.sec, "*.tex"))) + [os.path.join(paper, args.main)]
    files = [f for f in files if os.path.exists(f)]
    defined = set()
    npath = os.path.join(paper, "numbers.tex")
    if os.path.exists(npath):
        defined = set(re.findall(r"\\csname n@([^\\]+)\\endcsname\{", open(npath, encoding="utf-8").read()))
    n_lit = n_lines = 0
    undefined = []
    for f in files:
        findings, used = check_file(f, defined)
        rel = os.path.relpath(f, HERE)
        for ln, hits, text in findings:
            n_lit += len(hits)
            n_lines += 1
            print(f"{rel}:{ln}: {', '.join(hits)}  |  {text[:90]}")
        for k in used:
            if defined and k not in defined:
                undefined.append((rel, k))
    print(f"\n{n_lit} result-like literal(s) on {n_lines} line(s) in {len(files)} file(s) outside \\nv{{}}")
    if defined:
        print(f"{len(undefined)} \\nv{{}} key(s) not defined in {os.path.relpath(npath, HERE)}")
        for rel, k in undefined:
            print(f"  {rel}: \\nv{{{k}}}")
    return 1 if (n_lit or undefined) else 0


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(HERE, "results_v2"))
    ap.add_argument("--fm", default=None, help="zero-shot foundation-model runs (default <results>/fm)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra day-ahead/TSO summary JSON files")
    ap.add_argument("--data", default=None, help="processed-corpus dir for data.* macros")
    ap.add_argument("--out", default=os.path.join(HERE, "paper", "numbers.tex"))
    ap.add_argument("--regime", default="R2", help="headline information regime (v2)")
    ap.add_argument("--ref", default="PLANET", help="reference model for DM / gains / ablations")
    ap.add_argument("--holm-family", default="dataset", choices=["cell", "dataset", "all"])
    ap.add_argument("--mcs-B", type=int, default=2000)
    ap.add_argument("--dm-B", type=int, default=1000, help="stationary-bootstrap resamples of the DM variant")
    ap.add_argument("--mcs-alpha", type=float, default=0.10)
    ap.add_argument("--mcs-block", default="horizon", choices=["cuberoot", "horizon"],
                    help="block length: n^(1/3), or max(n^(1/3), H) to cover the H-1 target overlap")
    ap.add_argument("--no-mcs", action="store_true")
    ap.add_argument("--check", action="store_true", help="list hand-typed result literals in the prose")
    ap.add_argument("--paper", default=os.path.join(HERE, "paper"))
    ap.add_argument("--sec", default="sec", help="section directory under --paper to check")
    ap.add_argument("--main", default="main.tex", help="main file under --paper to check")
    args = ap.parse_args(argv)
    for k in ("results", "out", "paper"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    if args.check:
        return check(args)
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
