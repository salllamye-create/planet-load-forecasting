"""Every results figure of the v2 paper, built from the saved v2 result files.

    python src/fig_results_v2.py [--which horizon,regimes,...|F1,F2,...] [--results results_v2]
                                 [--fm <results>/fm] [--proc data/proc_v2] [--out figures_v2]
                                 [--regime R2] [--ref PLANET] [--models A,B,...]
                                 [--metric wape|mse] [--H 96] [--drift-frac 0.5]
                                 [--mcs-B 2000] [--mcs-alpha 0.10] [--numbers paper/numbers.tex]

Record selection (regime, size, lookback, seeds) is delegated to ``paper_numbers`` so that every
figure shows exactly the cells the tables and the prose quote. Each figure function renders what
exists and prints an explicit ``[skip]`` / ``[note]`` line for what does not; nothing raises on a
missing cell. Output: ``<out>/<name>.pdf`` and ``.png`` (Type-42 fonts, IEEE column geometry).

Figures
  F1 horizon    error versus lead time from ``errors/<tag>.npz`` ``per_step_mae`` (converted to
                MW with the training std of the target when the processed corpus is available)
  F2 regimes    WAPE under R0 / R1 / R2 per model and horizon (systems averaged, per-system dots)
  F3 mcs        Model-Confidence-Set membership matrix with the WAPE rank printed per cell
  F4 mech       learning curves, temporal drift, cross-grid transfer (PLAN_v2 E1-E3)
  F5 ablation   relative WAPE (or MSE) change of every ablation variant versus the full model,
                per backbone, cell dots + mean with a 95 % CI
  F6 dayahead   operator (TSO) day-ahead WAPE as a reference and the learned models as points
  F7 prob       CRPS per model and system; PICP raw / CQR / ACI against the nominal 90 % line
  F8 arch       the v2 architecture schematic (vector, no chart data)
  F9 fm         zero-shot foundation models against the best learned model per system, horizon

Colour follows the model identity in a fixed order (never cycled); every series also carries its
own dash pattern and marker so the figures survive the greyscale print edition. The six hues are
the v1 ``figstyle`` palette (validated on the adjacent-pair gate with the dataviz checker); model
families that share a hue (PatchTST / CalTST, N-BEATS / N-HiTS, TiDE / DLinear) differ in dash and
marker, and panels with dot marks keep few series per axis or facet by system.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs           # noqa: E402
import paper_numbers as pn      # noqa: E402
import stats                    # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ identities and styles
GRID = ["OPSD-DE", "OPSD-FR", "OPSD-GB", "OPSD-ES", "OPSD-IT", "OPSD-NL", "OPSD-PL", "OPSD-BE"]
HEADLINE = ["PLANET", "PLANETX", "PatchTST", "CalTST", "NBEATS", "NHITS", "TFT", "TiDE", "TimeXer",
            "DLinear", "GBRT", "TaoRefit", "TaoVanilla"]          # learned models shown in F1/F2
FM_MODELS = ["Chronos2", "TimesFM2", "Moirai2"]
BLUE, ORANGE, GREEN, PURPLE, BROWN, LBLUE = fs.PALETTE            # fixed slots, v1 order
INK2 = "#2B2B2B"
STYLE = {   # identity -> (colour, dash, width, marker, zorder); fixed, never cycled
    "PLANET":        (ORANGE, "-", 1.5, "o", 10),
    "PLANETX":       (ORANGE, (0, (4, 1.5)), 1.3, "s", 9),
    "PatchTST":      (BLUE, (0, (5, 1.6)), 1.1, "s", 6),
    "CalTST":        (BLUE, (0, (3, 1, 1, 1)), 1.1, "*", 6),
    "NBEATS":        (BROWN, (0, (6, 1.6, 1.4, 1.6)), 1.1, "v", 5),
    "NHITS":         (BROWN, (0, (1.6, 1.4)), 1.1, "^", 5),
    "TFT":           (GREEN, (0, (5, 1.6)), 1.1, "D", 5),
    "TiDE":          (PURPLE, (0, (1.6, 1.4)), 1.1, "P", 5),
    "TimeXer":       (LBLUE, (0, (6, 1.6, 1.4, 1.6)), 1.1, "X", 4),
    "DLinear":       (PURPLE, (0, (5, 1.6)), 1.1, "d", 4),
    "GBRT":          (fs.MUTED, (0, (1.6, 1.4)), 1.0, "h", 3),
    "TaoRefit":      (INK2, (0, (6, 2)), 1.0, "p", 3),
    "TaoVanilla":    (INK2, (0, (1, 2)), 1.0, "8", 3),
    "SeasonalNaive": (fs.MUTED, (0, (1, 2)), 0.9, "o", 1),
    "Chronos2":      (GREEN, (0, (6, 1.6, 1.4, 1.6)), 1.1, "<", 5),
    "TimesFM2":      (PURPLE, (0, (6, 1.6, 1.4, 1.6)), 1.1, ">", 5),
    "Moirai2":       (BROWN, (0, (5, 1.6)), 1.1, "H", 5),
    "TSO":           (INK2, (0, (6, 2)), 1.3, None, 6),
    "best":          (fs.INK, "-", 1.3, "_", 8),
}
_FALLBACK_MARKERS = ["s", "^", "v", "D", "P", "X", "*"]     # grey + shape for unlisted models (no grey fixed marker)
_fallback = {}
PRETTY = {"NBEATS": "N-BEATS", "NHITS": "N-HiTS", "PLANETX": "PLANET-X", "CalTST": "PatchTST+Cal", "TaoRefit": "Hong (refit)",
          "TaoRefitAR": "Hong (refit)+AR",
          "TaoVanilla": "Hong (once)", "SeasonalNaive": "Seasonal naive", "Chronos2": "Chronos-2",
          "TimesFM2": "TimesFM-2", "Moirai2": "Moirai-2", "TSO": "operator (TSO)"}
DS_PRETTY = {"ECL": "Electricity"}
SHORT = {"TaoRefit": "Hong", "TaoVanilla": "Hong-1", "TaoRefitAR": "Hong+AR", "PatchTST": "PTST", "CalTST": "PTST+C", "NBEATS": "N-B",
         "NHITS": "N-H", "PLANETX": "PL-X", "TimeXer": "TXer", "DLinear": "DLin", "SeasonalNaive": "naive"}   # tick-row labels
REGIME_LABEL = {"R0": "R0\nload only", "R1": "R1\n+calendar", "R2": "R2\n+weather", "v1": "v1"}
ABL_LABEL = {"no_pl": "no phase-locked tier", "no_resid": "no residual backbone",
             "no_slow": "no climatological tier", "no_trend": "no local trend",
             "daytype_wk": "no holiday day-type", "no_daytype": "no day-type terms",
             "uniform_w": "unweighted LS", "no_beta_token": "no state conditioning",
             "relative": "relative-time basis", "basis_film": "FiLM state conditioning",
             "lean": "lean basis", "utc": "UTC basis (not local time)", "full": "full model"}
BACKBONE_LABEL = {"patch": "patch Transformer backbone", "basis": "doubly-residual stack backbone",
                  None: "patch Transformer backbone (v1)"}
# one-hue sequential ramp (dataviz reference palette, blue 100 -> 700) for magnitude cells
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                    "#256abf", "#184f95", "#0d366b"])
IN_SET, OUT_SET = "#B7D3F6", "#FFFFFF"


def pretty(m):
    return PRETTY.get(m, m)


def dsname(d):
    return DS_PRETTY.get(d, d[5:] if d.startswith("OPSD-") else d)


def style(m):
    if m not in STYLE:
        if m not in _fallback:
            _fallback[m] = _FALLBACK_MARKERS[len(_fallback) % len(_FALLBACK_MARKERS)]
            print(f"  [note] no fixed style for model {m!r}; drawn in grey with marker {_fallback[m]!r}")
        return dict(color=fs.MUTED, linestyle="-", linewidth=0.9, marker=_fallback[m], zorder=2)
    c, ls, lw, mk, z = STYLE[m]
    return dict(color=c, linestyle=ls, linewidth=lw, marker=mk, zorder=z)


def line_kw(m):
    s = style(m)
    return dict(color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"], zorder=s["zorder"])


def handle(m, marker=True, **kw):
    s = style(m)
    d = dict(color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"],
             marker=s["marker"] if marker else None, markersize=3.6)
    d.update(kw)
    return Line2D([0], [0], **d)


def order_models(models):
    rank = {m: i for i, m in enumerate(HEADLINE + FM_MODELS + ["SeasonalNaive"])}
    return sorted(models, key=lambda m: (rank.get(m, 99), m))


def order_systems(ds):
    rank = {d: i for i, d in enumerate(GRID)}
    return sorted(ds, key=lambda d: (rank.get(d, 99), d))


def hsort(H):
    return int(H)


# ------------------------------------------------------------------ records
def load(results, fm=None):
    """All records of a results directory, classified as ``paper_numbers.generate`` does."""
    fm = fm or os.path.join(results, "fm")
    recs = pn.load_records(results, fm)
    for r in recs:
        if r["_regime"] is None and r["_src"] == "main":
            r["_regime"] = "v1"
    return recs


def regime_of(recs, wanted):
    reg = pn.choose_regime(recs, wanted)
    if reg is not None and reg != wanted:
        print(f"  [note] regime {wanted} has no headline records; using {reg}")
    return reg


def cells(recs, regime, learned_only=False, models=None):
    """(dataset, model, H) -> [selected record per seed] for the headline cells of a regime."""
    out = pn.cells_from(pn.select_headline(recs, regime))
    out = {k: v for k, v in out.items() if not pn.is_variant(k[0])}   # the -mp copies are a sensitivity, not a system
    if learned_only:
        out = {k: v for k, v in out.items() if not any(r["_src"] == "fm" or r.get("zero_shot") for r in v)}
    if models:
        out = {k: v for k, v in out.items() if k[1] in models}
    return out


def mean_of(rs, field="wape"):
    return pn.seed_stats(rs, field)[0]


def proc_dir_for(results, proc):
    if proc:
        return proc
    return os.path.join(HERE, "data", "proc_v2" if os.path.basename(results).endswith("_v2") else "proc")


_CORPUS = {}


def corpus(d, proc):
    """(stamps, |target| in physical units, meta) of a processed dataset, or None."""
    key = (d, proc)
    if key not in _CORPUS:
        npz, js = os.path.join(proc, d + ".npz"), os.path.join(proc, d + ".json")
        if os.path.exists(npz) and os.path.exists(js):
            z = np.load(npz)
            meta = json.load(open(js))
            y = z["data"][:, meta["target_idx"]].astype(np.float64)
            _CORPUS[key] = (z["stamps"].astype(np.int64), y, meta)
        else:
            _CORPUS[key] = None
    return _CORPUS[key]


def target_sd(d, proc):
    c = corpus(d, proc)
    if c is None:
        return None
    _, y, meta = c
    a, b = meta["splits"]["train"]
    return float(y[a:b].std() or 1.0)


def spread(ys, gap):
    """Push label positions apart by at least ``gap`` (keeps the sorted order)."""
    ys = np.asarray(ys, float)
    out = ys.copy()
    order = np.argsort(ys)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    from scipy.stats import t
    return float(t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def save(fig, name, out):
    fs.save(fig, name, out)


def legend_below(fig, handles, labels, ncol, pad=0.012, **kw):
    """A figure legend anchored just below the lowest axis decoration (measured, not guessed)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lo = min(ax.get_tightbbox(r).transformed(inv).y0 for ax in fig.axes if ax.get_visible())
    kw.setdefault("fontsize", 6.5)
    return fig.legend(handles, labels, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, lo - pad), **kw)


# ================================================================== F1 error vs lead time
def fig_horizon(recs, args):
    reg = regime_of(recs, args.regime)
    H = args.horizon_H
    cs = {k: v for k, v in cells(recs, reg, learned_only=True, models=args.models or HEADLINE).items()
          if hsort(k[2]) == H and k[0].startswith("OPSD")}
    curves = {}
    for (d, m, _), rs in cs.items():
        v = pn.seed_mean_err(rs, "per_step_mae")
        if v is not None and len(v) == H:
            curves[(d, m)] = v
    if not curves:
        print(f"  [skip] fig_horizon_v2: no per_step_mae arrays for H={H}, regime {reg}")
        return
    systems = order_systems({d for d, _ in curves})
    models = order_models({m for _, m in curves})
    physical = all(target_sd(d, args.proc) is not None for d in systems)
    ncol = 4 if len(systems) > 4 else len(systems)
    nrow = int(np.ceil(len(systems) / ncol))
    fs.setup()
    fig, axes = plt.subplots(nrow, ncol, figsize=(fs.DBL, 1.15 * nrow + 0.55), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, d in enumerate(systems):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        sd = target_sd(d, args.proc) if physical else 1.0
        for m in models:
            if (d, m) not in curves:
                continue
            v = curves[(d, m)] * sd
            ax.plot(np.arange(1, H + 1), v, label=pretty(m), **line_kw(m))
        ax.set_title(dsname(d), loc="left", fontsize=7.4, pad=2)
        ax.set_xlim(1, H)
        ax.set_xticks([1] + [t for t in (84, 168, 252, 336, 24, 48, 72, 96) if t <= H and t > 1][:4]
                      if H > 96 else [1, 24, 48, 72, 96][: (H // 24) + 1])
        if i // ncol == nrow - 1:
            ax.set_xlabel("lead time (h)")
        if i % ncol == 0:
            ax.set_ylabel("MAE (MW)" if physical else "MAE (z-scored)")
        ax.tick_params(labelsize=6.3)
    fig.subplots_adjust(wspace=0.32, hspace=0.42)
    legend_below(fig, [handle(m, marker=False) for m in models], [pretty(m) for m in models],
                 ncol=min(len(models), 6))
    save(fig, "fig_horizon_v2", args.out)


# ================================================================== F2 information regimes
def fig_regimes(recs, args):
    regimes = sorted({r["_regime"] for r in recs if r["_kind"] == "headline"
                      and str(r["_regime"]).startswith("R")})
    if len(regimes) < 2:
        print(f"  [skip] fig_regimes_v2: needs >= 2 regimes, found {regimes}")
        return
    want = args.models or HEADLINE
    W = {}   # (regime, d, m, H) -> seed-mean WAPE
    for reg in regimes:
        for (d, m, H), rs in cells(recs, reg, learned_only=True, models=want).items():
            v = mean_of(rs, "wape")
            if np.isfinite(v):
                W[(reg, d, m, hsort(H))] = v
    horizons = [h for h in sorted({k[3] for k in W}) if h in pn.GRID_H]
    # only models run under every regime (Hong's regression cannot run R0)
    models = order_models({k[2] for k in W if all((r, k[1], k[2], k[3]) in W for r in regimes)})
    if not W or not horizons:
        print("  [skip] fig_regimes_v2: no WAPE cells")
        return
    fs.setup()
    fig, axes = plt.subplots(1, len(horizons), figsize=(fs.DBL * (0.34 * len(horizons)), 2.05),
                             squeeze=False)
    xs = np.arange(len(regimes))
    used = set()
    for ax, H in zip(axes[0], horizons):
        for k, m in enumerate(models):
            off = (k - (len(models) - 1) / 2) * min(0.08, 0.5 / max(len(models), 1))
            per_reg = {}
            for i, reg in enumerate(regimes):
                pts = {d: W[(reg, d, m, H)] for d in GRID + sorted({kk[1] for kk in W if not kk[1].startswith("OPSD")})
                       if (reg, d, m, H) in W}
                if pts:
                    per_reg[reg] = pts
            if not per_reg:
                continue
            common = set.intersection(*[set(p) for p in per_reg.values()])
            st = style(m)
            mx, my = [], []
            for i, reg in enumerate(regimes):
                if reg not in per_reg:
                    continue
                vals = [per_reg[reg][d] for d in sorted(common)] if common else list(per_reg[reg].values())
                ax.plot([xs[i] + off] * len(vals), vals, linestyle="none", marker=st["marker"] or "o",
                        markersize=2.2, color=st["color"], alpha=0.35, markeredgewidth=0, zorder=2)
                mx.append(xs[i] + off); my.append(float(np.mean(vals)))
            ax.plot(mx, my, marker=st["marker"], markersize=3.8, markeredgecolor="white",
                    markeredgewidth=0.5, label=pretty(m), **line_kw(m))
            used.add(m)
        ax.set_xticks(xs)
        ax.set_xticklabels([REGIME_LABEL.get(r, r) for r in regimes], fontsize=6.5)
        ax.set_xlim(-0.5, len(regimes) - 0.5)
        ax.set_title(f"$H={H}$ h", loc="left", fontsize=7.4)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("WAPE (%), systems mean")
    models = [m for m in models if m in used]
    fig.text(0.995, 0.995, "small dots: individual systems", ha="right", va="top", fontsize=6.0,
             color=fs.MUTED)
    fig.subplots_adjust(wspace=0.30)
    legend_below(fig, [handle(m) for m in models], [pretty(m) for m in models], ncol=min(len(models), 6))
    save(fig, "fig_regimes_v2", args.out)


# ================================================================== F3 MCS membership
def read_macros(path):
    out = {}
    if not path or not os.path.exists(path):
        return out, []
    head = []
    rx = re.compile(r"\\csname n@(.+?)\\endcsname\{(.*)\}$")
    for line in open(path, encoding="utf-8"):
        if line.startswith("%") and len(head) < 3:
            head.append(line.strip("% \n"))
        m = rx.search(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out, head


def mcs_membership(cs, reg, args):
    """{(d, H): {model: in_set}} from paper/numbers.tex when it was generated from this results
    directory, otherwise recomputed with stats.mcs exactly as paper_numbers does."""
    macros, head = read_macros(args.numbers)
    src_ok = bool(head) and (f"from {os.path.basename(args.results)} " in head[0])
    headline = re.search(r"headline regime: (\S+?);", head[2]).group(1) if len(head) > 2 and \
        re.search(r"headline regime: (\S+?);", head[2]) else None
    prefix = "" if reg == headline else f"{reg}."
    groups = {}
    for (d, m, H), rs in cs.items():
        groups.setdefault((d, hsort(H)), {})[m] = rs
    # the macros are used only when they cover every cell with exactly these models; a stale
    # numbers.tex (fewer records than the directory now holds) is ignored as a whole
    covered = src_ok and all(f"{prefix}mcs.{d}.H{H}.{m}.in" in macros
                             for (d, H), by_m in groups.items() for m in by_m) and \
        all(macros.get(f"{prefix}mcs.{d}.H{H}.n_in") is not None for (d, H) in groups)
    if src_ok and not covered:
        print(f"  [note] {os.path.relpath(args.numbers, HERE)} does not cover the current cells; recomputing the MCS")
    out, recomputed = {}, 0
    for (d, H), by_m in groups.items():
        names = sorted(by_m)
        keys = [f"{prefix}mcs.{d}.H{H}.{m}.in" for m in names]
        if covered:
            out[(d, H)] = {m: macros[k] == "1" for m, k in zip(names, keys)}
            continue
        e = {}
        for m in names:
            a = pn.seed_mean_err(by_m[m])
            if a is not None:
                e[m] = a
        if len(e) < 2:
            continue
        n = pd.Series([len(a) for a in e.values()]).mode()[0]
        e = {m: a for m, a in e.items() if len(a) == n}
        names = sorted(e)
        L = np.stack([e[m] for m in names], axis=1)
        bl = max(1, int(round(L.shape[0] ** (1 / 3))))
        if args.mcs_block == "horizon":
            bl = max(bl, int(H))
        res = stats.mcs(L, alpha=args.mcs_alpha, B=args.mcs_B, block_len=bl, seed=0)
        out[(d, H)] = {m: bool(res["in_set"][i]) for i, m in enumerate(names)}
        recomputed += 1
    if recomputed:
        print(f"  [note] MCS recomputed for {recomputed} cell(s) (B={args.mcs_B}, alpha={args.mcs_alpha})")
    elif out:
        print(f"  [note] MCS membership read from {os.path.relpath(args.numbers, HERE)}")
    return out


def fig_mcs(recs, args):
    reg = regime_of(recs, args.regime)
    cs = cells(recs, reg)
    if not cs:
        print("  [skip] fig_mcs_v2: no headline cells")
        return
    member = mcs_membership(cs, reg, args)
    if not member:
        print("  [skip] fig_mcs_v2: no MCS cells (fewer than two models per cell?)")
        return
    from scipy.stats import rankdata
    colkeys = sorted((k for k in member if hsort(k[1]) in pn.GRID_H),
                     key=lambda k: (GRID.index(k[0]) if k[0] in GRID else 99, k[0], hsort(k[1])))
    models = order_models({m for v in member.values() for m in v})
    ranks = {}
    for (d, H) in colkeys:
        ms = [m for m in models if (d, m, H) in cs or (d, m, str(H)) in cs]
        vals = np.array([mean_of(cs.get((d, m, H), cs.get((d, m, str(H)), [])), "wape") for m in ms])
        ok = np.isfinite(vals)
        rk = np.full(len(ms), np.nan)
        if ok.sum() >= 2:
            rk[ok] = rankdata(vals[ok], method="average")
        for m, r in zip(ms, rk):
            ranks[(d, H, m)] = r
    fs.setup()
    nrow, ncol = len(models), len(colkeys)
    fig = plt.figure(figsize=(min(fs.DBL, 0.30 * ncol + 1.55), 0.21 * nrow + 0.72))
    ax = fig.add_axes([0.20, 0.06, 0.70, 0.74])
    ax.set_xlim(0, ncol); ax.set_ylim(nrow, 0)
    ax.set_xticks([]); ax.set_yticks(np.arange(nrow) + 0.5)
    ax.set_yticklabels([pretty(m) for m in models], fontsize=6.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_in = {m: 0 for m in models}
    for j, (d, H) in enumerate(colkeys):
        for i, m in enumerate(models):
            if m not in member[(d, H)]:
                ax.add_patch(Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, facecolor="#F2F2F2",
                                       edgecolor="none"))
                continue
            inside = member[(d, H)][m]
            n_in[m] += int(inside)
            ax.add_patch(Rectangle((j + 0.03, i + 0.06), 0.94, 0.88, facecolor=IN_SET if inside else OUT_SET,
                                   edgecolor=fs.GRID, linewidth=0.4))
            r = ranks.get((d, H, m))
            if r is not None and np.isfinite(r):
                ax.text(j + 0.5, i + 0.52, f"{r:g}", ha="center", va="center", fontsize=6.0,
                        color=fs.INK if inside else fs.MUTED, fontweight="bold" if inside else "normal")
        ax.text(j + 0.5, -0.12, f"{H}", ha="center", va="bottom", fontsize=5.8, color=fs.MUTED)
    # system group labels
    j = 0
    while j < ncol:
        d = colkeys[j][0]
        k = j
        while k < ncol and colkeys[k][0] == d:
            k += 1
        ax.text((j + k) / 2, -0.75, dsname(d), ha="center", va="bottom", fontsize=6.6, color=fs.INK)
        if k < ncol:
            ax.plot([k, k], [-0.05, nrow], color=fs.MUTED, lw=0.5, clip_on=False)
        j = k
    for i, m in enumerate(models):
        ax.text(ncol + 0.25, i + 0.52, f"{n_in[m]}/{sum(m in member[k] for k in colkeys)}",
                ha="left", va="center", fontsize=5.9, color=fs.MUTED)
    ax.text(ncol + 0.25, -0.12, "in set", ha="left", va="bottom", fontsize=5.8, color=fs.MUTED)
    ax.text(0, -1.45, "$H$ (h) per system; cell = WAPE rank; shaded = in the model confidence set "
            f"($\\alpha={args.mcs_alpha:g}$)", ha="left", va="bottom", fontsize=6.1, color=fs.MUTED)
    legend_below(fig, [Rectangle((0, 0), 1, 1, facecolor=IN_SET, edgecolor=fs.GRID, lw=0.4),
                       Rectangle((0, 0), 1, 1, facecolor=OUT_SET, edgecolor=fs.GRID, lw=0.4)],
                 ["in the MCS", "excluded"], ncol=2, fontsize=6.3, handlelength=1.3)
    save(fig, "fig_mcs_v2", args.out)


# ================================================================== F4 mechanism experiments
def _other(recs, reg, pfx):
    return [r for r in recs if r["_kind"] == "other" and r["_regime"] == reg and r["_src"] == "main"
            and str(r["_variant"] or "").startswith(pfx)]


def _seed_group(rs, key):
    g = {}
    for r in rs:
        g.setdefault(key(r), []).append(r)
    return g


def _learning_curves(recs, reg, H, models, base):
    lc = [r for r in _other(recs, reg, "lc") if hsort(r["H"]) == H]
    g = _seed_group(lc, lambda r: (r["dataset"], r["model"], float(r.get("train_frac") or
                                                                    r["_variant"][2:])))
    pts = {}   # (m, d) -> {frac: wape}
    for (d, m, f), rs in g.items():
        if models and m not in models:
            continue
        pts.setdefault((m, d), {})[f] = mean_of(rs, "wape")
    for (m, d) in list(pts):
        full = base.get((d, m, H)) or base.get((d, m, str(H)))
        if full:
            pts[(m, d)][1.0] = mean_of(full, "wape")
    return pts


def _drift_curve(r, proc, month_days=30.4375):
    c = corpus(r["dataset"], proc)
    ae = pn.load_err(r, "per_window_ae")
    if c is None or ae is None:
        return None
    stamps, y, meta = c
    L, H = int(r["L"]), int(r["H"])
    tr0, tr1 = meta["splits"]["train"]
    te0, te1 = meta["splits"]["test"]
    stride = int(r.get("train_stride") or 5)
    tr_starts = np.arange(max(0, tr0 - L), tr1 - L - H + 1, stride)
    n = max(1, int(round(len(tr_starts) * float(r.get("train_frac") or 1.0))))
    train_end = stamps[min(tr_starts[n - 1] + L + H - 1, len(stamps) - 1)]
    te_starts = np.arange(max(0, te0 - L), te1 - L - H + 1, 1)
    if len(te_starts) != len(ae):
        return None
    cs = np.concatenate([[0.0], np.cumsum(np.abs(y))])
    den = cs[te_starts + L + H] - cs[te_starts + L]           # sum |y| over each target block
    num = ae.astype(float) * H                                 # per_window_ae is the mean over H
    months = np.floor((stamps[te_starts + L] - train_end) / (month_days * 86400.0)).astype(int)
    out = {}
    for mo in np.unique(months):
        sel = months == mo
        if den[sel].sum() > 0:
            out[int(mo)] = 100.0 * num[sel].sum() / den[sel].sum()
    return out


def _drift(recs, reg, H, models, proc, frac):
    dr = [r for r in _other(recs, reg, "drift") if hsort(r["H"]) == H and r.get("train_first", True)]
    if models:
        dr = [r for r in dr if r["model"] in models]
    if not dr:
        return {}, None
    fr = lambda r: float(r.get("train_frac") or r["_variant"][5:])
    if frac is None:                       # largest fraction every present model has
        by_m = {}
        for r in dr:
            by_m.setdefault(r["model"], set()).add(fr(r))
        common = set.intersection(*by_m.values())
        frac = max(common) if common else max(max(v) for v in by_m.values())
    dr = [r for r in dr if abs(fr(r) - frac) < 1e-9]
    g = _seed_group(dr, lambda r: (r["model"], r["dataset"]))
    curves = {}   # (m, d) -> {month: wape}
    for (m, d), rs in g.items():
        per = [c for c in (_drift_curve(r, proc) for r in rs) if c]
        if per:
            keys = sorted(set.union(*[set(c) for c in per]))
            curves[(m, d)] = {k: float(np.mean([c[k] for c in per if k in c])) for k in keys}
    return curves, frac


def _transfer(recs, reg, H, models, base):
    xf = [r for r in _other(recs, reg, "xfer") if hsort(r["H"]) == H]
    g = _seed_group(xf, lambda r: (r["model"], r["dataset"], r.get("eval_data") or r["_variant"][5:]))
    mats = {}   # m -> {(src, tgt): wape}
    for (m, s, t), rs in g.items():
        if models and m not in models:
            continue
        mats.setdefault(m, {})[(s, t)] = mean_of(rs, "wape")
    for m in mats:
        for d in {s for s, _ in mats[m]} | {t for _, t in mats[m]}:
            full = base.get((d, m, H)) or base.get((d, m, str(H)))
            if full:
                mats[m][(d, d)] = mean_of(full, "wape")
    return mats


def fig_mech(recs, args):
    reg = regime_of(recs, args.regime)
    H = args.H
    base = cells(recs, reg, learned_only=True)
    lc = _learning_curves(recs, reg, H, args.models, base)
    drift, frac = _drift(recs, reg, H, args.models, args.proc, args.drift_frac)
    xfer = _transfer(recs, reg, H, args.models, base)
    panels = [p for p, ok in (("lc", bool(lc)), ("drift", bool(drift)), ("xfer", bool(xfer))) if ok]
    for p, name in (("lc", "learning-curve (_lc) runs"), ("drift", "temporal-drift (_drift) runs"),
                    ("xfer", "cross-grid (_xfer) runs")):
        if p not in panels:
            print(f"  [note] fig_mech_v2: no {name} at H={H}, regime {reg}; panel omitted")
    if not panels:
        print("  [skip] fig_mech_v2: no mechanism records")
        return
    fs.setup()
    widths = {"lc": 1.0, "drift": 1.15, "xfer": 1.0 + 0.55 * max(0, len(xfer) - 1)}
    fig = plt.figure(figsize=(fs.DBL if len(panels) > 1 else fs.COL, 2.05))
    gs = fig.add_gridspec(1, len(panels), width_ratios=[widths[p] for p in panels], wspace=0.34)
    letters = iter("abc")
    legend_models = []
    if "lc" in panels:
        ax = fig.add_subplot(gs[0, panels.index("lc")])
        models = order_models({m for m, _ in lc})
        for m in models:
            fracs = sorted(set.union(*[set(v) for (mm, _), v in lc.items() if mm == m]))
            st = style(m)
            for (mm, d), v in lc.items():
                if mm == m:
                    ax.plot([f for f in fracs if f in v], [v[f] for f in fracs if f in v], linestyle="none",
                            marker=st["marker"] or "o", markersize=2.0, color=st["color"], alpha=0.35,
                            markeredgewidth=0, zorder=2)
            ys = [np.mean([v[f] for (mm, _), v in lc.items() if mm == m and f in v]) for f in fracs]
            ax.plot(fracs, ys, marker=st["marker"], markersize=3.6, markeredgecolor="white",
                    markeredgewidth=0.5, **line_kw(m))
            legend_models.append(m)
        ax.set_xscale("log")
        allf = sorted({f for v in lc.values() for f in v})
        ax.set_xticks(allf); ax.set_xticklabels([f"{100 * f:g}%" for f in allf], fontsize=6.4)
        ax.minorticks_off()
        ax.set_xlabel("training fraction (most recent)")
        ax.set_ylabel("WAPE (%), systems mean")
        ax.set_title(f"({next(letters)}) Learning curve, $H={H}$ h", loc="left", fontsize=7.4)
    if "drift" in panels:
        ax = fig.add_subplot(gs[0, panels.index("drift")])
        models = order_models({m for m, _ in drift})
        for m in models:
            st = style(m)
            per = [v for (mm, _), v in drift.items() if mm == m]
            for v in per:
                ks = sorted(v)
                ax.plot(ks, [v[k] for k in ks], linestyle=st["linestyle"], color=st["color"],
                        linewidth=0.5, alpha=0.30, zorder=2)
            ks = sorted(set.union(*[set(v) for v in per]))
            ax.plot(ks, [np.mean([v[k] for v in per if k in v]) for k in ks], **line_kw(m))
            if m not in legend_models:
                legend_models.append(m)
        ax.set_xlabel("months after training end")
        ax.set_ylabel("monthly WAPE (%)")
        ax.set_title(f"({next(letters)}) Drift, oldest {100 * frac:g}% of the split",
                     loc="left", fontsize=7.4)
    if "xfer" in panels:
        models = order_models(xfer)
        sub = gs[0, panels.index("xfer")].subgridspec(1, len(models), wspace=0.08)
        vmax = max(v for mm in xfer.values() for v in mm.values() if np.isfinite(v))
        vmin = min(v for mm in xfer.values() for v in mm.values() if np.isfinite(v))
        for k, m in enumerate(models):
            ax = fig.add_subplot(sub[0, k])
            systems = order_systems({s for s, _ in xfer[m]} | {t for _, t in xfer[m]})
            n = len(systems)
            M = np.full((n, n), np.nan)
            for (s, t), v in xfer[m].items():
                M[systems.index(s), systems.index(t)] = v
            ax.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
            cell_in = (fs.DBL * widths["xfer"] / sum(widths[p] for p in panels)) / max(len(models), 1) / n
            if cell_in >= 0.17:
                for i in range(n):
                    for j in range(n):
                        if np.isfinite(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=4.8,
                                    color="white" if (M[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55 else fs.INK)
            ax.set_xticks(range(n)); ax.set_xticklabels([dsname(s) for s in systems], fontsize=5.6, rotation=90)
            ax.set_yticks(range(n))
            ax.set_yticklabels([dsname(s) for s in systems] if k == 0 else [], fontsize=5.6)
            ax.tick_params(length=0)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k == len(models) // 2:
                ax.set_xlabel("evaluated on", fontsize=6.4)
            if k == 0:
                ax.set_ylabel("trained on", fontsize=6.4)
            ax.set_title(f"({next(letters)}) {pretty(m)}" if k == 0 else pretty(m), loc="center", fontsize=5.8)
        fig.text(0.995, 0.995, "transfer WAPE (%), diagonal = in-domain", ha="right", va="top",
                 fontsize=6.0, color=fs.MUTED)
    if legend_models:
        legend_below(fig, [handle(m) for m in legend_models], [pretty(m) for m in legend_models],
                     ncol=min(len(legend_models), 6))
    save(fig, "fig_mech_v2", args.out)


# ================================================================== F5 ablations
def fig_ablation(recs, args):
    reg = regime_of(recs, args.regime)
    ref, metric = args.ref, args.metric
    abl = [r for r in recs if r["_kind"] == "abl" and r["model"] == ref and r["_regime"] == reg]
    if not abl:
        print(f"  [skip] fig_ablation_v2: no ablation records for {ref} in regime {reg}")
        return
    head = pn.select_headline(recs, reg)          # (d, m, H, seed) -> record; v1 reference
    full = {}                                     # (backbone, d, H, seed) -> reference record
    for r in abl:
        if r["_variant"] == "full":
            full[(r["_backbone"], r["dataset"], hsort(r["H"]), r["_seed"])] = r
    rel = {}                                      # (backbone, variant, d, H) -> [per-seed % change]
    for r in abl:
        if r["_variant"] == "full":
            continue
        bb, d, H, s = r["_backbone"], r["dataset"], hsort(r["H"]), r["_seed"]
        base = full.get((bb, d, H, s)) or (head.get((d, ref, r["H"], s)) if bb is None else None)
        if base is None:
            continue
        v = pn.pct_change(r.get(metric), base.get(metric))
        if np.isfinite(v):
            rel.setdefault((bb, r["_variant"], d, H), []).append(v)
    if not rel:
        print("  [skip] fig_ablation_v2: no ablation cell has a matching full-model record")
        return
    backbones = sorted({k[0] for k in rel}, key=lambda b: ("patch", "basis", None).index(b) if b in ("patch", "basis", None) else 9)
    variants = sorted({k[1] for k in rel})
    per_cell = {k: float(np.mean(v)) for k, v in rel.items()}          # seed mean per cell
    summary = {}                                                       # (bb, var) -> (mean, ci, n)
    for bb in backbones:
        for var in variants:
            vals = [per_cell[k] for k in per_cell if k[0] == bb and k[1] == var]
            if vals:
                ci = ci95(vals) if len(vals) > 1 else ci95(rel[next(k for k in rel if k[0] == bb and k[1] == var)])
                summary[(bb, var)] = (float(np.mean(vals)), ci, len(vals))
    order = sorted(variants, key=lambda v: -np.nanmean([summary[(bb, v)][0] for bb in backbones if (bb, v) in summary]))
    hlist = sorted({k[3] for k in rel})
    hmarks = {H: ["o", "^", "s", "D"][i % 4] for i, H in enumerate(hlist)}
    hoff = {H: (i - (len(hlist) - 1) / 2) * 0.16 for i, H in enumerate(hlist)}
    fs.setup()
    fig, axes = plt.subplots(1, len(backbones), figsize=(fs.COL if len(backbones) == 1 else fs.DBL * 0.78,
                                                          0.24 * len(order) + 0.95), sharey=True, squeeze=False)
    ys = np.arange(len(order))
    for ax, bb in zip(axes[0], backbones):
        for i, var in enumerate(order):
            for k, v in per_cell.items():
                if k[0] == bb and k[1] == var:
                    ax.plot(v, i + hoff[k[3]], marker=hmarks[k[3]], markersize=2.6, color=fs.MUTED,
                            alpha=0.6, markeredgewidth=0, linestyle="none", zorder=3)
            if (bb, var) in summary:
                mean, ci, n = summary[(bb, var)]
                ax.errorbar(mean, i, xerr=ci if np.isfinite(ci) else 0, fmt="o", markersize=4.2,
                            color=ORANGE if mean > 0 else BLUE, ecolor=fs.INK, elinewidth=0.8, capsize=2.2,
                            capthick=0.8, zorder=5, markeredgecolor="white", markeredgewidth=0.5)
        ax.axvline(0, color=fs.MUTED, lw=0.7, zorder=2)
        ax.set_yticks(ys); ax.set_yticklabels([ABL_LABEL.get(v, v) for v in order], fontsize=6.6)
        ax.set_title(BACKBONE_LABEL.get(bb, str(bb)), loc="left", fontsize=7.4)
        ax.set_xlabel(f"change in {metric.upper()} vs full model (%)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.15)
    axes[0][0].invert_yaxis()
    hs = [Line2D([0], [0], marker="o", color=ORANGE, linestyle="none", markersize=4.2),
          Line2D([0], [0], marker="o", color=BLUE, linestyle="none", markersize=4.2)]
    ls = ["mean over cells, worse (95% CI)", "mean over cells, better (95% CI)"]
    for H, mk in hmarks.items():
        hs.append(Line2D([0], [0], marker=mk, color=fs.MUTED, linestyle="none", markersize=2.8))
        ls.append(f"cell (system, $H={H}$ h), seed mean")
    fig.subplots_adjust(wspace=0.10)
    legend_below(fig, hs, ls, ncol=2, fontsize=6.2)
    save(fig, "fig_ablation_v2", args.out)


# ================================================================== F6 day-ahead vs operator
def fig_dayahead(recs, args):
    p = os.path.join(args.results, "dayahead.json")
    if not os.path.exists(p):
        print(f"  [skip] fig_dayahead_v2: {os.path.relpath(p, HERE)} missing")
        return
    da = json.load(open(p, encoding="utf-8"))
    reg = args.regime
    systems = [d for d in order_systems(da) if isinstance(da[d], dict) and "TSO" in da[d]]
    if not systems:
        print("  [skip] fig_dayahead_v2: no TSO entries")
        return
    learned = {}      # (d, model) -> {L: [wape per seed]}
    for d in systems:
        for k, v in da[d].items():
            if k.startswith("_") or k == "TSO" or not isinstance(v, dict) or "wape" not in v:
                continue
            model = v.get("model") or k.split("|")[0]
            if model.endswith(("-ens", "-refit")) or model.startswith("Comb"):   # ensembles, refits, combinations: table only
                continue
            suffix = v.get("suffix", "")
            r_reg = pn.parse_suffix(suffix, v)[0] if suffix else None
            if r_reg is not None and r_reg != reg:
                continue
            learned.setdefault((d, model), {}).setdefault(v.get("L", 336), []).append(v["wape"])
    pts = {}
    for (d, m), byL in learned.items():
        L = 336 if 336 in byL else max(byL, key=lambda l: len(byL[l]))
        pts[(d, m)] = float(np.mean(byL[L]))
    if not pts:
        print("  [note] fig_dayahead_v2: no learned day-ahead entries yet; operator reference only")
    models = order_models({m for _, m in pts})
    if args.models:
        models = [m for m in models if m in args.models]
    # operator-only entries (the multi-point weather corpora carry no day-ahead model) are not shown
    systems = [d for d in systems if any((d, m) in pts for m in models)] or systems
    systems = sorted(systems, key=lambda d: da[d]["TSO"]["wape"])
    fs.setup()
    fig, ax = plt.subplots(figsize=(fs.COL, 2.1))
    xs = np.arange(len(systems))
    for i, d in enumerate(systems):
        ax.plot([i - 0.36, i + 0.36], [da[d]["TSO"]["wape"]] * 2, color=INK2, lw=1.3, zorder=6,
                solid_capstyle="butt")
    for k, m in enumerate(models):
        off = (k - (len(models) - 1) / 2) * min(0.09, 0.6 / max(len(models), 1))
        st = style(m)
        xv = [i + off for i, d in enumerate(systems) if (d, m) in pts]
        yv = [pts[(d, m)] for d in systems if (d, m) in pts]
        ax.plot(xv, yv, linestyle="none", marker=st["marker"] or "o", markersize=3.8, color=st["color"],
                markeredgecolor="white", markeredgewidth=0.5, zorder=st["zorder"], label=pretty(m))
    ax.set_xticks(xs); ax.set_xticklabels([dsname(d) for d in systems])
    ax.set_xlim(-0.6, len(systems) - 0.4)
    lead = da[systems[0]].get("_protocol", {}).get("lead_hours", [13, 36])
    ax.set_ylabel(f"day-ahead WAPE (%), leads {lead[0]}-{lead[1]} h")
    ax.set_xlabel("system, sorted by operator error")
    ax.grid(axis="x", visible=False)
    hs = [Line2D([0], [0], color=INK2, lw=1.3)] + [handle(m, marker=True, linestyle="none") for m in models]
    ls = ["operator (TSO) forecast"] + [pretty(m) for m in models]
    legend_below(fig, hs, ls, ncol=3 if len(ls) > 4 else len(ls), fontsize=6.2)
    save(fig, "fig_dayahead_v2", args.out)


# ================================================================== F7 probabilistic
def fig_prob(recs, args):
    pdir = os.path.join(args.results, "prob")
    files = sorted(glob.glob(os.path.join(pdir, "*.json")))
    if not files:
        print(f"  [skip] fig_prob_v2: no records in {os.path.relpath(pdir, HERE)}")
        return
    rows = {}
    for p in files:
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not {"dataset", "model", "H"} <= set(r):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        base = f"{r['dataset']}_{r['model']}_L{r.get('L')}_H{r['H']}_s{r.get('seed')}"
        suffix = stem[len(base):] if stem.startswith(base) else str(r.get("tag") or "")
        reg = pn.parse_suffix(suffix, r)[0]
        if reg is not None and reg != args.regime:
            continue
        rows.setdefault((r["dataset"], r["model"], hsort(r["H"])), []).append(r)
    if not rows:
        print(f"  [skip] fig_prob_v2: no probabilistic records for regime {args.regime}")
        return
    H = args.prob_H if args.prob_H in {k[2] for k in rows} else sorted({k[2] for k in rows})[0]
    rows = {k: v for k, v in rows.items() if k[2] == H}
    systems = order_systems({k[0] for k in rows})
    models = order_models({k[1] for k in rows})
    if args.models:
        models = [m for m in models if m in args.models]
    crps_key = next((k for k in ("crps_cqr", "crps_aci", "crps_raw") if any(k in r for v in rows.values() for r in v)), None)
    rel = all(("mae_median" in r and "wape_median" in r) for v in rows.values() for r in v)

    def crps(rs):
        vals = []
        for r in rs:
            c = pn.fin(r.get(crps_key))
            if rel and pn.fin(r.get("wape_median")) > 0:
                c = 100.0 * c / (r["mae_median"] / (r["wape_median"] / 100.0))
            vals.append(c)
        return float(np.nanmean(vals)) if vals else float("nan")

    fs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(fs.DBL * 0.72, 2.0), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.32))
    ax = axes[0]
    xs = np.arange(len(systems))
    for k, m in enumerate(models):
        off = (k - (len(models) - 1) / 2) * min(0.1, 0.6 / max(len(models), 1))
        st = style(m)
        xv = [i + off for i, d in enumerate(systems) if (d, m, H) in rows]
        yv = [crps(rows[(d, m, H)]) for d in systems if (d, m, H) in rows]
        ax.plot(xv, yv, linestyle="none", marker=st["marker"] or "o", markersize=3.8, color=st["color"],
                markeredgecolor="white", markeredgewidth=0.5, zorder=st["zorder"])
    ax.set_xticks(xs); ax.set_xticklabels([dsname(d) for d in systems])
    ax.set_xlim(-0.6, len(systems) - 0.4)
    ax.set_ylabel("CRPS (% of mean load)" if rel else "CRPS (MW)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"(a) Sharpness and calibration, $H={H}$ h", loc="left", fontsize=7.4)
    ax = axes[1]
    stages = [s for s in ("raw", "cqr", "aci") if any(f"picp_{s}" in r for v in rows.values() for r in v)]
    sx = np.arange(len(stages))
    ax.axhline(90, color=fs.INK, lw=0.8, ls=(0, (4, 2)), zorder=4)
    ax.text(-0.36, 90.3, "nominal 90%", fontsize=6.0, ha="left", va="bottom", color=fs.INK)
    for k, m in enumerate(models):
        off = (k - (len(models) - 1) / 2) * 0.05
        st = style(m)
        sys_vals = {}
        for d in systems:
            if (d, m, H) in rows:
                sys_vals[d] = [float(np.nanmean([pn.fin(r.get(f"picp_{s}")) for r in rows[(d, m, H)]])) for s in stages]
        if not sys_vals:
            continue
        for d, v in sys_vals.items():
            ax.plot(sx + off, v, linestyle="none", marker=st["marker"] or "o", markersize=2.0, color=st["color"],
                    alpha=0.35, markeredgewidth=0, zorder=2)
        ax.plot(sx + off, np.nanmean(list(sys_vals.values()), axis=0), marker=st["marker"], markersize=3.6,
                markeredgecolor="white", markeredgewidth=0.5, **line_kw(m))
    ax.set_xticks(sx); ax.set_xticklabels(["quantile\nhead", "+CQR", "+ACI"][: len(stages)], fontsize=6.4)
    ax.set_xlim(-0.4, len(stages) - 0.6)
    ax.set_ylabel("PICP (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Coverage by calibration stage", loc="left", fontsize=7.4)
    legend_below(fig, [handle(m) for m in models], [pretty(m) for m in models], ncol=min(len(models), 5))
    save(fig, "fig_prob_v2", args.out)


# ================================================================== F8 architecture
def _box(ax, x, y, w, h, text, fc="#FFFFFF", ec=fs.MUTED, fontsize=6.1, lw=0.7, tc=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                                linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4,
                color=tc or fs.INK, linespacing=1.35)


def _arrow(ax, p, q, color=None, lw=0.75, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, linewidth=lw,
                                 color=color or fs.MUTED, zorder=2, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0.8, shrinkB=0.8))


def _tag(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=5.6, color="white", ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=color, ec="none"))


def fig_arch(recs, args):
    fs.setup()
    fig = plt.figure(figsize=(fs.DBL, 2.75))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    GR, BL, OR, GN, PU = fs.GRID, BLUE, ORANGE, GREEN, PURPLE
    # ---- inputs (regime-labelled)
    _box(ax, 0.010, 0.775, 0.150, 0.130, "weather block\nNWP-degraded future\n(temp, rhum, rad, wspd, ...)",
         fc="#F3F5F8", fontsize=5.8)
    _tag(ax, 0.012, 0.915, "R2 = R1 + weather", PU)
    _box(ax, 0.010, 0.535, 0.150, 0.150, "calendar block\nlocal hour, weekday, weekend,\nholiday, bridge day, Xmas",
         fc="#F3F5F8", fontsize=5.8)
    _tag(ax, 0.012, 0.695, "R1 = R0 + calendar", PU)
    _box(ax, 0.010, 0.290, 0.150, 0.150, "lookback $\\mathbf{X}_{1:L}$\nload and channels\nlocal time $\\tau_{1:L+H}$",
         fc="#F3F5F8", fontsize=6.0)
    _tag(ax, 0.012, 0.450, "R0 = load only", PU)
    ax.text(0.085, 0.235, "every model in a regime\nreceives the identical block", ha="center", va="top",
            fontsize=5.5, color=fs.MUTED, style="italic")
    # ---- tier 1: climatology, basis, normalisation
    _box(ax, 0.215, 0.775, 0.215, 0.130, "climatological tier (optional)\n$g_c=\\psi(\\tau)^{\\top}\\mathbf{B}_c$\nannual harmonics, global",
         fc="#EAF1FA", ec=BL, tc=BL, fontsize=5.9)
    _box(ax, 0.215, 0.535, 0.215, 0.150,
         "phase-locked basis $\\Phi(\\tau)$\nlocal civil time (DST-aware)\ndaily $k{\\leq}6$, weekly $k{\\leq}3$, trend,\n"
         "day-types: Sat, Sun, holiday", fc="#EAF1FA", ec=BL, tc=BL, fontsize=5.7)
    _box(ax, 0.215, 0.290, 0.215, 0.150, "instance norm $\\hat{x}=(\\tilde{\\mathbf{X}}-\\mu)/\\sigma$\n"
         "observation weights $w$\n(learned, $\\sum_t w_t=L$)", fc="#FDF0E8", ec=OR, tc=OR, fontsize=5.8)
    # ---- tier 2: solver and shared head
    _box(ax, 0.485, 0.775, 0.245, 0.130, "shared exogenous head\npoint-wise response + thermal context\n"
         "+ depth-wise temporal conv; same module\nfor every model without native covariates",
         fc="#E9F6F1", ec=GN, tc=GN, fontsize=5.4)
    _box(ax, 0.485, 0.480, 0.245, 0.205,
         "closed-form solver (per window)\n$\\beta_c=(\\Phi_L^{\\top}W_c\\Phi_L+\\Lambda_c)^{-1}\\Phi_L^{\\top}W_c\\hat{x}_c$\n"
         "seasonal state $\\beta$ re-estimated\nat test time, no gradient steps", fc="#FDF0E8", ec=OR, lw=1.2, tc=OR,
         fontsize=5.8)
    # ---- residual backbones (one of two)
    _box(ax, 0.215, 0.040, 0.515, 0.200, "", fc="#FFFFFF", ec=fs.MUTED, ls=(0, (3, 2)), lw=0.6)
    ax.text(0.222, 0.228, "residual backbone, one of two, chosen on validation", fontsize=5.7,
            color=fs.MUTED, va="center", ha="left", zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.6))
    _box(ax, 0.230, 0.060, 0.235, 0.135, "patch Transformer\nchannel-independent,\n$\\beta$ prepended as a state token",
         fc="#E9F6F1", ec=GN, tc=GN, fontsize=5.7)
    _box(ax, 0.485, 0.060, 0.235, 0.135, "doubly-residual MLP stack\nN-BEATS-style blocks,\n$\\beta$ projected onto the input",
         fc="#E9F6F1", ec=GN, tc=GN, fontsize=5.7)
    ax.text(0.4745, 0.128, "or", ha="center", va="center", fontsize=6.2, color=fs.MUTED, zorder=5)
    # ---- sum and output
    ax.add_patch(plt.Circle((0.835, 0.560), 0.030, facecolor="white", edgecolor=fs.INK, lw=0.8, zorder=5))
    ax.text(0.835, 0.560, "$+$", ha="center", va="center", fontsize=8.5, color=fs.INK, zorder=6)
    _box(ax, 0.905, 0.485, 0.085, 0.150, "$\\hat{\\mathbf{Y}}_{L+1:L+H}$", fc="#F3F5F8", fontsize=7.4)
    # ---- arrows
    _arrow(ax, (0.160, 0.840), (0.485, 0.860), color=PU)                    # weather -> head
    _arrow(ax, (0.160, 0.640), (0.485, 0.815), color=PU, rad=-0.18)         # calendar -> head
    _arrow(ax, (0.160, 0.590), (0.215, 0.600), color=PU)                    # calendar -> basis (day-types)
    ax.text(0.187, 0.612, "day-types", fontsize=5.0, color=PU, ha="center", va="bottom")
    _arrow(ax, (0.160, 0.400), (0.215, 0.400))                              # X -> norm
    _arrow(ax, (0.160, 0.420), (0.215, 0.800), rad=0.0)                     # X (tau) -> climatology
    _arrow(ax, (0.322, 0.775), (0.322, 0.685), color=BL)                    # g -> subtract
    ax.text(0.330, 0.730, "$-\\,g$", fontsize=5.8, color=BL, va="center")
    _arrow(ax, (0.322, 0.535), (0.322, 0.440), color=BL)                    # basis -> norm/solver path
    _arrow(ax, (0.430, 0.610), (0.485, 0.610), color=BL)                    # Phi_L, Phi_H -> solver
    ax.text(0.457, 0.622, "$\\Phi_L,\\Phi_H$", fontsize=5.4, color=BL, ha="center", va="bottom")
    _arrow(ax, (0.430, 0.365), (0.485, 0.520), color=OR, rad=-0.15)         # x_hat, w -> solver
    ax.text(0.442, 0.462, "$\\hat{x},\\,w$", fontsize=5.6, color=OR, ha="right", va="bottom")
    _arrow(ax, (0.607, 0.480), (0.607, 0.240), color=OR)                    # residual + beta -> backbones
    ax.text(0.615, 0.360, "$\\mathbf{R}=\\hat{x}-\\Phi_L\\beta,\\;\\beta$", fontsize=5.5, color=OR, va="center")
    _arrow(ax, (0.730, 0.580), (0.805, 0.565), color=OR)                    # Phi_H beta -> sum
    ax.text(0.768, 0.594, "$\\Phi_H\\beta$", fontsize=5.6, color=OR, ha="center", va="bottom")
    _arrow(ax, (0.730, 0.130), (0.835, 0.530), color=GN, rad=-0.30)         # R_fut -> sum
    ax.text(0.800, 0.235, "$\\mathbf{R}^{\\mathrm{fut}}$", fontsize=5.6, color=GN, ha="left", va="center")
    _arrow(ax, (0.730, 0.830), (0.835, 0.590), color=GN, rad=0.30)          # exo correction -> sum
    ax.text(0.742, 0.882, "exo correction", fontsize=5.3, color=GN, ha="left", va="center")
    # g_fut -> sum, routed along the top so it crosses no box
    ax.plot([0.322, 0.322, 0.835], [0.905, 0.955, 0.955], color=BL, lw=0.75, zorder=2,
            solid_joinstyle="miter")
    _arrow(ax, (0.835, 0.955), (0.835, 0.592), color=BL)
    ax.text(0.580, 0.962, "$g(\\tau_{L+1:L+H})$", fontsize=5.6, color=BL, ha="center", va="bottom")
    _arrow(ax, (0.865, 0.560), (0.905, 0.560), color=fs.INK)                # sum -> Y
    save(fig, "fig_arch_v2", args.out)


# ================================================================== F9 foundation models
def fig_fm(recs, args):
    fm = [r for r in recs if r["_src"] == "fm" or r.get("zero_shot")]
    if not fm:
        print("  [skip] fig_fm_v2: no zero-shot foundation-model records")
        return
    reg = regime_of(recs, args.regime)
    learned = cells(recs, reg, learned_only=True)
    best = {}     # (d, H) -> (model, wape)
    for (d, m, H), rs in learned.items():
        v = mean_of(rs, "wape")
        if np.isfinite(v) and ((d, hsort(H)) not in best or v < best[(d, hsort(H))][1]):
            best[(d, hsort(H))] = (m, v)
    pts = {}      # (model, d, H, long) -> (wape, regime)
    for r in fm:
        key = (r["model"], r["dataset"], hsort(r["H"]), r["_kind"] == "fm-long" or int(r.get("L") or 0) > 512)
        cand = pts.get(key)
        w = pn.fin(r.get("wape"))
        if not np.isfinite(w):
            continue
        if cand is None or (cand[1] != reg and r["_regime"] == reg):
            pts[key] = (w, r["_regime"])
    if not pts:
        print("  [skip] fig_fm_v2: FM records carry no WAPE")
        return
    horizons = sorted({k[2] for k in pts})
    systems = order_systems({k[1] for k in pts} | {d for d, _ in best})
    models = order_models({k[0] for k in pts})
    fs.setup()
    fig, axes = plt.subplots(1, len(horizons), figsize=(fs.DBL * min(1.0, 0.36 * len(horizons)), 2.05),
                             squeeze=False, sharey=False)
    used_reg, long_used = set(), False
    for ax, H in zip(axes[0], horizons):
        syst = [d for d in systems if any(k[1] == d and k[2] == H for k in pts) or (d, H) in best]
        xs = np.arange(len(syst))
        names = []
        for i, d in enumerate(syst):
            if (d, H) in best:
                m, v = best[(d, H)]
                ax.plot([i - 0.34, i + 0.34], [v, v], color=fs.INK, lw=1.2, zorder=6, solid_capstyle="butt")
                names.append(SHORT.get(m, m))
            else:
                names.append(None)
        # the best model's name goes in a second tick row, once per run of identical winners
        tr = ax.get_xaxis_transform()
        i = 0
        while i < len(names):
            j = i
            while j + 1 < len(names) and names[j + 1] == names[i]:
                j += 1
            if names[i]:
                ax.text((i + j) / 2, -0.20, names[i], fontsize=5.4, color=fs.MUTED, ha="center", va="top",
                        transform=tr, clip_on=False)
                if j > i:
                    ax.plot([i - 0.3, j + 0.3], [-0.185, -0.185], color=fs.MUTED, lw=0.5, transform=tr,
                            clip_on=False)
            i = j + 1
        for k, m in enumerate(models):
            off = (k - (len(models) - 1) / 2) * 0.16
            st = style(m)
            for long in (False, True):
                xv, yv = [], []
                for i, d in enumerate(syst):
                    if (m, d, H, long) in pts:
                        xv.append(i + off); yv.append(pts[(m, d, H, long)][0])
                        used_reg.add(pts[(m, d, H, long)][1])
                if xv:
                    long_used |= long
                    ax.plot(xv, yv, linestyle="none", marker=st["marker"] or "o", markersize=4.2,
                            color=st["color"], markerfacecolor="white" if long else st["color"],
                            markeredgecolor=st["color"] if long else "white", markeredgewidth=0.8 if long else 0.5,
                            zorder=st["zorder"] + (0 if long else 1))
        ax.set_xticks(xs); ax.set_xticklabels([dsname(d) for d in syst])
        ax.set_xlim(-0.6, len(syst) - 0.4)
        ax.set_title(f"$H={H}$ h", loc="left", fontsize=7.4)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("WAPE (%)")
    hs = [Line2D([0], [0], color=fs.INK, lw=1.2)]
    ls = ["best learned model (named under the system)"]
    for m in models:
        st = style(m)
        hs.append(Line2D([0], [0], marker=st["marker"], color=st["color"], linestyle="none", markersize=4.2))
        ls.append(f"{pretty(m)} (zero-shot, $L=336$)")
    if long_used:
        hs.append(Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=fs.INK,
                         linestyle="none", markersize=4.2))
        ls.append("open: long context $L=1024$")
    if not used_reg <= {reg}:
        fig.text(0.995, 0.995, f"FM points: regime {reg} where run, else " +
                 ", ".join(sorted(r for r in used_reg if r and r != reg)),
                 ha="right", va="top", fontsize=6.0, color=fs.MUTED)
    fig.subplots_adjust(wspace=0.28)
    legend_below(fig, hs, ls, ncol=min(len(ls), 4), fontsize=6.3)
    save(fig, "fig_fm_v2", args.out)


# ================================================================== driver
FIGS = [("F1", "horizon", fig_horizon), ("F2", "regimes", fig_regimes), ("F3", "mcs", fig_mcs),
        ("F4", "mech", fig_mech), ("F5", "ablation", fig_ablation), ("F6", "dayahead", fig_dayahead),
        ("F7", "prob", fig_prob), ("F8", "arch", fig_arch), ("F9", "fm", fig_fm)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="all", help="comma list of names or F-numbers (default all)")
    ap.add_argument("--results", default=os.path.join(HERE, "results_v2"))
    ap.add_argument("--fm", default=None, help="zero-shot runs (default <results>/fm)")
    ap.add_argument("--proc", default=None, help="processed corpus (default data/proc_v2 or data/proc)")
    ap.add_argument("--out", default=os.path.join(HERE, "figures_v2"))
    ap.add_argument("--regime", default="R2")
    ap.add_argument("--ref", default="PLANET")
    ap.add_argument("--models", default=None, help="comma list restricting the models drawn")
    ap.add_argument("--metric", default="wape", choices=["wape", "mse"], help="ablation metric")
    ap.add_argument("--H", type=int, default=96, help="horizon of the mechanism figure")
    ap.add_argument("--horizon-H", type=int, default=336, help="horizon of the lead-time figure")
    ap.add_argument("--prob-H", type=int, default=24)
    ap.add_argument("--drift-frac", type=float, default=None, help="training fraction of the drift panel")
    ap.add_argument("--numbers", default=os.path.join(HERE, "paper", "numbers.tex"))
    ap.add_argument("--mcs-B", type=int, default=2000)
    ap.add_argument("--mcs-alpha", type=float, default=0.10)
    ap.add_argument("--mcs-block", default="horizon", choices=["cuberoot", "horizon"])
    args = ap.parse_args(argv)
    args.results = os.path.abspath(args.results)
    args.out = os.path.abspath(args.out)
    args.proc = proc_dir_for(args.results, args.proc)
    args.models = [m.strip() for m in args.models.split(",")] if args.models else None
    os.makedirs(args.out, exist_ok=True)
    want = [w.strip().lower() for w in args.which.split(",")] if args.which != "all" else None
    todo = [(k, n, f) for k, n, f in FIGS if want is None or k.lower() in want or n in want]
    if want and len(todo) != len(want):
        print(f"[warn] unknown figure names in --which: {set(want) - {k.lower() for k, _, _ in todo} - {n for _, n, _ in todo}}")
    recs = load(args.results, args.fm)
    print(f"{len(recs)} records from {os.path.relpath(args.results, HERE)} -> {os.path.relpath(args.out, HERE)}")
    failed = 0
    for k, name, fn in todo:
        print(f"-- {k} {name}")
        try:
            fn(recs, args)
        except Exception as e:                      # noqa: BLE001  (one bad figure must not stop the rest)
            import traceback
            failed += 1
            print(f"  [fail] {name}: {e}")
            traceback.print_exc(limit=3)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
