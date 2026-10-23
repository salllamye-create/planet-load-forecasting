"""Every results figure in the paper, built from the saved result files."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import analysis as an

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, ERR, PROC = (os.path.join(HERE, "results"), os.path.join(HERE, "results", "errors"),
                  os.path.join(HERE, "data", "proc"))
PROBD = os.path.join(RES, "prob")


def _res(d, m, H, L=336, seed=2021, tag=""):
    p = os.path.join(RES, f"{d}_{m}_L{L}_H{H}_s{seed}{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


_SCALER = {}


def _scaler(d):
    """Train-split mean/std of the target channel, used to put saved forecasts back
    into physical units (exp.py stores them z-scored)."""
    if d not in _SCALER:
        z = np.load(os.path.join(PROC, d + ".npz"))
        meta = json.load(open(os.path.join(PROC, d + ".json")))
        a, b = meta["splits"]["train"]
        col = z["data"][a:b, meta["target_idx"]].astype(np.float64)
        sd = col.std() or 1.0
        _SCALER[d] = (col.mean(), sd)
    return _SCALER[d]


def _pred(d, m, H, L=336, seed=2021, tag=""):
    p = os.path.join(ERR, f"{d}_{m}_L{L}_H{H}_s{seed}{tag}_pred.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    mu, sd = _scaler(d)
    return {"idx": z["idx"], "pred": z["pred"] * sd + mu, "true": z["true"] * sd + mu}


def _steps(d, m, H, L=336, seed=2021, tag=""):
    p = os.path.join(ERR, f"{d}_{m}_L{L}_H{H}_s{seed}{tag}.npz")
    return np.load(p) if os.path.exists(p) else None


def _stamps(d):
    z = np.load(os.path.join(PROC, d + ".npz"))
    return pd.to_datetime(z["stamps"], unit="s")


# ------------------------------------------------------------------ 1. error vs horizon step
def fig_horizon(panels=(("OPSD-DE", 336), ("ECL", 336), ("ETTm1", 336)),
                models=("PLANET", "PatchTST", "iTransformer", "DLinear", "TimeMixer")):
    fs.setup()
    panels = [p for p in panels if _steps(p[0], "PLANET", p[1]) is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(fs.DBL, 1.95), squeeze=False)
    for ax, (d, H) in zip(axes[0], panels):
        for m in models:
            z = _steps(d, m, H)
            if z is None:
                continue
            v = z["per_step_mae"]
            k = max(1, len(v) // 120)
            ax.plot(np.arange(len(v))[::k] + 1, pd.Series(v).rolling(k, min_periods=1).mean()[::k],
                    label=m, **fs.style(m))
        ax.set_title(f"{an.DS_PRETTY.get(d,d)}, $H={H}$", loc="left", fontsize=7.6)
        ax.set_xlabel("forecast step $h$")
    axes[0][0].set_ylabel("MAE (normalised)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fs.save(fig, "fig_horizon")


# ------------------------------------------------------------------ 2. ablations
def fig_ablation(datasets=("OPSD-DE", "ECL", "ETTh1", "ETTm1"), horizons=(96, 336)):
    fs.setup()
    from run_all import ABLATIONS
    labels = {"no_pl": "no phase-locked tier", "no_resid": "no residual net",
              "no_slow": "no climatological tier", "relative": "relative-time basis",
              "uniform_w": "unweighted LS", "no_beta_token": "no state conditioning",
              "no_daytype": "no day-type terms", "no_trend": "no local trend"}
    keys = [k for k in ABLATIONS if k != "full"]
    rows, ok = [], []
    for k in keys:
        vals = []
        for d in datasets:
            for H in horizons:
                f = _res(d, "PLANET", H)
                a = _res(d, "PLANET", H, tag=f"_abl-{k}")
                if f and a:
                    vals.append(100.0 * (a["mse"] - f["mse"]) / f["mse"])
        if vals:
            rows.append(vals); ok.append(k)
    if not rows:
        return
    means = [np.mean(v) for v in rows]
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(fs.COL, 0.32 * len(ok) + 0.75))
    yp = np.arange(len(ok))
    for i, o in enumerate(order):
        col = "#C44601" if means[o] > 0 else "#0B5FA5"
        ax.barh(i, means[o], color=col, height=0.62, zorder=3)
        ax.plot(rows[o], [i] * len(rows[o]), "o", ms=2.4, color=fs.INK,
                alpha=0.55, zorder=4, mew=0)
        ax.text(means[o] + (0.35 if means[o] > 0 else -0.35), i, f"{means[o]:+.1f}%",
                va="center", ha="left" if means[o] > 0 else "right", fontsize=6.3,
                color=fs.INK)
    ax.set_yticks(yp); ax.set_yticklabels([labels.get(ok[o], ok[o]) for o in order])
    ax.axvline(0, color=fs.MUTED, lw=0.7)
    ax.set_xlabel("change in MSE when the component is removed (%)")
    ax.margins(x=0.22)
    ax.grid(axis="y", visible=False)
    fs.save(fig, "fig_ablation")


# ------------------------------------------------------------------ 3. lookback sensitivity
def fig_lookback(datasets=("OPSD-DE", "ETTh1"), models=("PLANET", "PatchTST", "DLinear"),
                 looks=(96, 192, 336, 512, 720), H=96):
    fs.setup()
    fig, axes = plt.subplots(1, len(datasets), figsize=(fs.DBL * 0.66, 1.85), squeeze=False)
    any_pt = False
    for ax, d in zip(axes[0], datasets):
        for m in models:
            xs, ys = [], []
            for L in looks:
                r = _res(d, m, H, L=L, tag="_look") or _res(d, m, H, L=L)
                if r:
                    xs.append(L); ys.append(r["mse"])
            if xs:
                any_pt = True
                st = fs.style(m)
                ax.plot(xs, ys, marker=fs.MODEL_STYLE[m]["m"], ms=3.2, label=m, **st)
        ax.set_title(an.DS_PRETTY.get(d, d), loc="left", fontsize=7.6)
        ax.set_xlabel("lookback $L$"); ax.set_xticks(list(looks))
    if not any_pt:
        plt.close(fig); return
    axes[0][0].set_ylabel(f"test MSE ($H={H}$)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fs.save(fig, "fig_lookback")


# ------------------------------------------------------------------ 4. accuracy vs cost
def fig_efficiency(d="OPSD-DE", H=96):
    fs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(fs.DBL * 0.72, 2.0))
    pts = []
    for m in an.ORDER:
        r = _res(d, m, H)
        if r and r["params"] > 0:
            pts.append((m, r["params"], r["train_time_s"], r["mse"]))
    if not pts:
        plt.close(fig); return
    for ax, xi, xlab in [(axes[0], 1, "trainable parameters"),
                         (axes[1], 2, "training time (s)")]:
        for m, p, t, mse in pts:
            v = (p, t)[xi - 1]
            st = fs.MODEL_STYLE.get(m, {})
            ax.scatter(v, mse, s=26 if m == "PLANET" else 15,
                       color=st.get("c", fs.MUTED), marker=st.get("m", "o"),
                       zorder=6 if m == "PLANET" else 4, linewidths=0.5,
                       edgecolors="white")
            if m in ("PLANET", "PatchTST", "DLinear", "iTransformer", "Autoformer", "FITS"):
                ax.annotate(m, (v, mse), textcoords="offset points", xytext=(3.5, 2.5),
                            fontsize=6.0, color=fs.INK)
        ax.set_xscale("log"); ax.set_xlabel(xlab)
    axes[0].set_ylabel(f"test MSE")
    fig.suptitle(f"{an.DS_PRETTY.get(d,d)}, $H={H}$", x=0.06, y=1.02, ha="left", fontsize=7.6)
    fs.save(fig, "fig_efficiency")


# ------------------------------------------------------------------ 5. regime-shift robustness
SHIFTS = [("2019-12-20", "2020-01-06", "year-end"),
          ("2020-03-16", "2020-05-10", "lockdown"),
          ("2020-05-11", "2020-06-30", "recovery")]
CALM = ("2019-09-01", "2019-12-19")


def fig_shift(d="OPSD-DE", H=24,
              models=("PLANET", "CalTST", "PatchTST", "NBEATS", "DLinear"), anchor=None):
    """Three independent regime shifts fall inside the OPSD test window. The recurring one
    (year-end holidays) turns out to cost more than the one-off pandemic shock."""
    fs.setup()
    stamps = _stamps(d)
    have = [m for m in models if _pred(d, m, H) is not None]
    if not have:
        print("  [skip] fig_shift: no saved forecasts")
        return
    anchor = anchor or have[0]

    fig = plt.figure(figsize=(fs.DBL, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.30)

    # (a) rolling accuracy across the whole test period, all three shifts shaded
    ax = fig.add_subplot(gs[0, 0])
    series = {}
    for m in have:
        z = _pred(d, m, H)
        t = stamps[z["idx"] + 336]
        ae = pd.Series(np.abs(z["pred"] - z["true"]).sum(1), index=t).sort_index()
        ab = pd.Series(np.abs(z["true"]).sum(1), index=t).sort_index()
        series[m] = (ae, ab)
        w = (ae.rolling(24 * 14, min_periods=48).sum() /
             ab.rolling(24 * 14, min_periods=48).sum()) * 100
        ax.plot(w.index, w.values, label=m, **fs.style(m))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.10 * (hi - lo), hi)
    lo2, hi2 = ax.get_ylim()
    for i, (s0, s1, lab) in enumerate(SHIFTS):
        a, b = pd.Timestamp(s0), pd.Timestamp(s1)
        ax.axvspan(a, b, color="#C44601", alpha=0.11, zorder=0)
        ax.text(a + (b - a) / 2, lo2 + (0.005 if i % 2 == 0 else 0.055) * (hi2 - lo2),
                lab, fontsize=5.9, color="#C44601", va="bottom", ha="center")
    ax.set_ylabel("14-day rolling WAPE (%)")
    ax.set_title("(a) Accuracy across the test period", loc="left", fontsize=7.5)
    ax.legend(ncol=2, loc="upper left", fontsize=6.2)

    # (b) excess error over the calm baseline, by shift
    ax = fig.add_subplot(gs[0, 1])
    wape = lambda ae, ab, a, b: 100.0 * ae.loc[a:b].sum() / ab.loc[a:b].sum()
    w = 0.8 / max(len(have), 1)
    xs = np.arange(len(SHIFTS))
    for k, m in enumerate(have):
        ae, ab = series[m]
        calm = wape(ae, ab, *CALM)
        vals = [wape(ae, ab, s0, s1) - calm for s0, s1, _ in SHIFTS]
        ax.bar(xs + (k - len(have) / 2 + 0.5) * w, vals, w * 0.88, label=m,
               color=fs.MODEL_STYLE.get(m, {}).get("c", fs.MUTED), zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.axhline(0, color=fs.MUTED, lw=0.7)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, _, l in SHIFTS], fontsize=6.3)
    ax.set_ylabel("excess WAPE over calm (pp)")
    ax.set_title("(b) Cost of each shift", loc="left", fontsize=7.5)
    ax.grid(axis="x", visible=False)

    # (c) a single day inside the most damaging shift
    ax = fig.add_subplot(gs[0, 2])
    base = _pred(d, anchor, H)
    tt = stamps[base["idx"] + 336]
    j = int(np.argmin(np.abs((tt - pd.Timestamp("2019-12-25 00:00")).total_seconds())))
    hrs = pd.date_range(tt[j] + pd.Timedelta(hours=1), periods=H, freq="h")
    ax.plot(hrs, base["true"][j] / 1000, label="Actual", **fs.style("Actual"))
    for m in have:
        ax.plot(hrs, _pred(d, m, H)["pred"][j] / 1000, label=m, **fs.style(m))
    ax.set_ylabel("load (GW)")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(byhour=(0, 6, 12, 18)))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_title("(c) Christmas Day, 25 Dec 2019", loc="left", fontsize=7.5)
    ax.legend(ncol=2, fontsize=6.0, loc="upper left")
    fs.save(fig, "fig_shift")


# ------------------------------------------------------------------ 6. decomposition
def fig_decomp(d="OPSD-DE", H=96, L=336):
    """Runs a fresh PLANET forward pass purely to visualise its internal split."""
    import torch
    from data import WindowSet
    import exp as E
    fs.setup()
    ck = os.path.join(HERE, "checkpoints", f"{d}_PLANET_L{L}_H{H}.pt")
    if not os.path.exists(ck):
        print("  [skip] fig_decomp: no checkpoint"); return
    meta = json.load(open(os.path.join(PROC, d + ".json")))
    m, bc = E.build_model("PLANET", L, H, meta["C"])
    m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    ds = WindowSet(d, "test", L, H, None, bc)
    tr = WindowSet(d, "train", L, H, None, bc)
    ds.data = (np.load(os.path.join(PROC, d + ".npz"))["data"] - tr.mu) / tr.sd
    i = min(3500, len(ds) - 1)
    x, y, Phi, Psi, tf, s = ds[i]
    dd = m.decompose(x[None], Phi[None], Psi[None])
    sd_, mu_ = tr.sd[meta["target_idx"]], tr.mu[meta["target_idx"]]
    g = lambda v: (v[0, :, meta["target_idx"]].numpy() * sd_ + mu_) / 1000.0
    t = np.arange(H)
    fig, ax = plt.subplots(figsize=(fs.COL, 1.85))
    ax.plot(t, y[:, meta["target_idx"]].numpy() * sd_ / 1000 + mu_ / 1000,
            label="Actual", **fs.style("Actual"))
    ax.plot(t, g(dd["total"]), label="PLANET", **fs.style("PLANET"))
    ax.plot(t, g(dd["fast"]), label="phase-locked tier", color="#0B5FA5", lw=1.0, ls="--")
    ax.plot(t, dd["resid"][0, :, meta["target_idx"]].numpy() * sd_ / 1000,
            label="residual net", color="#00916B", lw=0.9, ls=":")
    ax.set_xlabel("forecast step $h$ (hours)"); ax.set_ylabel("load (GW)")
    ax.legend(ncol=2, fontsize=6.3)
    fs.save(fig, "fig_decomp")


# ------------------------------------------------------------------ 7. probabilistic
def fig_prob(d="OPSD-DE", H=24, models=("PLANET", "PatchTST", "DLinear")):
    fs.setup()
    got = [(m, json.load(open(os.path.join(PROBD, f"{d}_{m}_L336_H{H}_s2021.json"))))
           for m in models
           if os.path.exists(os.path.join(PROBD, f"{d}_{m}_L336_H{H}_s2021.json"))]
    if not got:
        return
    fig, axes = plt.subplots(1, 3, figsize=(fs.DBL, 1.95))
    # (a) coverage before/after calibration
    ax = axes[0]
    w = 0.26
    for i, (m, r) in enumerate(got):
        for j, (k, hatch) in enumerate([("picp_raw", ""), ("picp_cqr", "///"), ("picp_aci", "...")]):
            ax.bar(i + (j - 1) * w, r[k], w * 0.92, color=fs.MODEL_STYLE[m]["c"],
                   alpha=[0.45, 0.72, 1.0][j], zorder=3,
                   label=["quantile head", "+CQR", "+ACI"][j] if i == 0 else None)
    ax.axhline(90, color=fs.INK, lw=0.8, ls="--", zorder=4)
    ax.text(len(got) - 0.5, 90.4, "nominal 90%", fontsize=6.1, ha="right", color=fs.INK)
    ax.set_xticks(range(len(got))); ax.set_xticklabels([m for m, _ in got], fontsize=6.4)
    ax.set_ylabel("PICP (%)"); ax.set_ylim(60, 100)
    ax.set_title("(a) Interval coverage", loc="left", fontsize=7.6)
    ax.legend(fontsize=6.0, ncol=1, loc="lower left")
    # (b) CRPS / Winkler
    ax = axes[1]
    for i, (m, r) in enumerate(got):
        ax.bar(i - 0.17, r["crps_cqr"], 0.32, color=fs.MODEL_STYLE[m]["c"], zorder=3)
        ax.bar(i + 0.17, r["winkler_aci"] / 10, 0.32, color=fs.MODEL_STYLE[m]["c"],
               alpha=0.5, zorder=3)
    ax.set_xticks(range(len(got))); ax.set_xticklabels([m for m, _ in got], fontsize=6.4)
    ax.set_ylabel("CRPS (MW) $\\;/\\;$ Winkler$/10$")
    ax.set_title("(b) Sharpness \\& proper scores", loc="left", fontsize=7.6)
    # (c) an interval band
    ax = axes[2]
    p = os.path.join(PROBD, f"{d}_PLANET_L336_H{H}_s2021_q.npz")
    if os.path.exists(p):
        z = np.load(p)
        j = min(4000, len(z["Y"]) - 1)
        t = np.arange(H)
        ax.fill_between(t, z["lo"][j] / 1000, z["hi"][j] / 1000, color="#C44601",
                        alpha=0.18, lw=0, zorder=2, label="90% PI (ACI)")
        ax.plot(t, z["P"][j, :, len(z["P"][0, 0]) // 2] / 1000, label="median",
                **fs.style("PLANET"))
        ax.plot(t, z["Y"][j] / 1000, label="Actual", **fs.style("Actual"))
        ax.set_xlabel("forecast step $h$ (hours)"); ax.set_ylabel("load (GW)")
        ax.legend(fontsize=6.2, loc="best")
    ax.set_title("(c) Calibrated day-ahead band", loc="left", fontsize=7.6)
    fs.save(fig, "fig_prob")


# ------------------------------------------------------------------ 8. vs the TSO's own forecast
def fig_tso(grids=("OPSD-DE", "OPSD-FR", "OPSD-GB"), H=24,
            models=("SeasonalNaive", "HistGBR", "DLinear", "PatchTST", "iTransformer", "PLANET")):
    fs.setup()
    tso = json.load(open(os.path.join(RES, "tso_baseline.json"))) if os.path.exists(
        os.path.join(RES, "tso_baseline.json")) else {}
    fig, ax = plt.subplots(figsize=(fs.COL, 1.9))
    w = 0.8 / (len(models) + 1)
    xs = np.arange(len(grids))
    for k, m in enumerate(list(models) + ["TSO"]):
        vals = []
        for d in grids:
            if m == "TSO":
                vals.append(tso.get(d, {}).get("wape", np.nan))
            else:
                r = _res(d, m, H)
                vals.append(r["wape"] if r else np.nan)
        st = fs.MODEL_STYLE.get(m, {})
        ax.bar(xs + (k - len(models) / 2) * w, vals, w * 0.9, label=an.PRETTY.get(m, m)
               .replace("\\textbf{", "").replace("}", ""),
               color=st.get("c", fs.MUTED), zorder=3,
               alpha=1.0 if m in ("PLANET", "TSO") else 0.8,
               hatch="//" if m == "TSO" else None, edgecolor="white", linewidth=0.4)
    ax.set_xticks(xs); ax.set_xticklabels([an.DS_PRETTY.get(g, g) for g in grids])
    ax.set_ylabel(f"WAPE (%), $H={H}$ h")
    ax.legend(ncol=2, fontsize=6.1)
    ax.grid(axis="x", visible=False)
    fs.save(fig, "fig_tso")


ALL = dict(horizon=fig_horizon, ablation=fig_ablation, lookback=fig_lookback,
           efficiency=fig_efficiency, shift=fig_shift, decomp=fig_decomp,
           prob=fig_prob, tso=fig_tso)



# ------------------------------------------------------------------ 9. what the solver learned
def fig_weights(d="OPSD-DE", H=96, L=336):
    """Opens a trained PLANET and shows the two learned quantities that govern the solver:
    the observation weights and the per-channel ridge penalties."""
    import torch
    from data import WindowSet
    import exp as E
    fs.setup()
    ck = os.path.join(HERE, "checkpoints", f"{d}_PLANET_L{L}_H{H}.pt")
    if not os.path.exists(ck):
        print("  [skip] fig_weights: no checkpoint"); return
    meta = json.load(open(os.path.join(PROC, d + ".json")))
    m, bc = E.build_model("PLANET", L, H, meta["C"],
                          {"n_year_slow": E.slow_order(meta)})
    m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    ds = WindowSet(d, "test", L, H, None, bc)
    stamps = _stamps(d)

    fig = plt.figure(figsize=(fs.DBL, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.28)

    # (a) average weight profile over the lookback
    idxs = np.linspace(0, len(ds) - 1, 220).astype(int)
    W, tgt = [], meta["target_idx"]
    with torch.no_grad():
        for i in idxs:
            x, y, Phi, Psi, tf, s = ds[i]
            W.append(m.decompose(x[None], Phi[None], Psi[None])["weights"][0, :, tgt].numpy())
    W = np.stack(W)
    ax = fig.add_subplot(gs[0, 0])
    age = np.arange(-L, 0)
    ax.plot(age, W.mean(0), color="#C44601", lw=1.2, zorder=4)
    ax.fill_between(age, np.percentile(W, 10, 0), np.percentile(W, 90, 0),
                    color="#C44601", alpha=0.16, lw=0, zorder=3)
    ax.axhline(1.0, color=fs.MUTED, lw=0.7, ls="--", zorder=2)
    ax.text(-L + 6, 1.03, "uniform", fontsize=6.1, color=fs.MUTED, va="bottom")
    ax.set_xlabel("lookback position (hours before origin)")
    ax.set_ylabel("learned weight $w$")
    ax.set_title("(a) Observation weights", loc="left", fontsize=7.5)

    # (b) learned penalty per basis group, for the supervised channel.
    # NB: under the MS setting only the target channel receives gradient, so the other
    # channels' penalties stay at initialisation and say nothing -- we show the groups
    # within the target channel instead, which is what the model actually learned.
    ax = fig.add_subplot(gs[0, 1])
    lam = m.lam().detach().numpy()
    lam_t = lam[tgt] if lam.ndim == 2 else lam
    nd, nw = bc["n_day"], bc["n_week"]
    groups, i = [], 0
    groups.append(("intercept", [i])); i += 1
    if bc.get("trend", True):
        groups.append(("trend", [i])); i += 1
    for k in range(nd):
        groups.append((f"daily $k{{=}}{k+1}$", [i, i + 1])); i += 2
    for k in range(nw):
        groups.append((f"weekly $k{{=}}{k+1}$", [i, i + 1])); i += 2
    groups.append(("day-type", list(range(i, len(lam_t)))))
    vals = [float(np.mean(lam_t[ix])) for _, ix in groups]
    names = [n for n, _ in groups]
    init = 336 * (0.01 + 1e-3)
    cols = ["#C44601" if v < init else "#0B5FA5" for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.7, zorder=3)
    ax.axvline(init, color=fs.INK, lw=0.8, ls="--", zorder=4)
    ax.text(init * 1.05, len(vals) - 0.4, "init", fontsize=6.0, color=fs.INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel(r"learned penalty $\lambda$ (log)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Which basis terms survive", loc="left", fontsize=7.5)

    # (c) a window containing a public holiday
    ax = fig.add_subplot(gs[0, 2])
    origin = stamps[ds.starts + L - 1]
    want = pd.Timestamp("2019-12-29 00:00")
    j = int(np.argmin(np.abs((origin - want).total_seconds())))
    x, y, Phi, Psi, tf, s = ds[j]
    with torch.no_grad():
        dd = m.decompose(x[None], Phi[None], Psi[None])
    w = dd["weights"][0, :, tgt].numpy()
    sd_, mu_ = ds.sd[tgt], ds.mu[tgt]
    series = (x[:, tgt].numpy() * sd_ + mu_) / 1000.0
    t = stamps[s: s + L]
    ax.plot(t, series, color=fs.INK, lw=0.85, zorder=4, label="load")
    ax2 = ax.twiny(); ax2.axis("off")            # weights share the x-axis, plotted as a band
    lo, hi = series.min(), series.max()
    ax.fill_between(t, lo - 0.02 * (hi - lo), lo + 0.20 * (hi - lo) * (w / max(w.max(), 1e-9)),
                    color="#C44601", alpha=0.45, lw=0, zorder=2, label="weight $w$")
    for hol, lab in [("2019-12-25", "25 Dec"), ("2019-12-26", "26 Dec"), ("2020-01-01", "1 Jan")]:
        ts = pd.Timestamp(hol)
        if t[0] <= ts <= t[-1]:
            ax.axvline(ts, color="#7D3C98", lw=0.7, ls=":", zorder=5)
            ax.text(ts, hi, lab, fontsize=5.9, color="#7D3C98", rotation=90,
                    va="top", ha="right")
    ax.set_ylabel("load (GW)")
    ax.legend(fontsize=6.1, loc="lower left", ncol=2, framealpha=0.85, frameon=True)
    ax.set_title("(c) Holidays are down-weighted", loc="left", fontsize=7.5)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(6.0)
    fs.save(fig, "fig_weights")


ALL["weights"] = fig_weights


if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for k in which:
        try:
            print(f"-- {k}")
            ALL[k]()
        except Exception as e:
            import traceback
            print(f"  [fail] {k}: {e}")
            traceback.print_exc(limit=2)
