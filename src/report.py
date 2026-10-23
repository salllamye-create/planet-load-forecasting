"""Compute every number quoted in the paper's prose, so the text is written from
measured values rather than from memory. Dumps results/report.json and prints a summary."""
import glob, json, os
import numpy as np
import pandas as pd

import analysis as an

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, ERR, PROC = (os.path.join(HERE, "results"), os.path.join(HERE, "results", "errors"),
                  os.path.join(HERE, "data", "proc"))

ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
GRID = ["OPSD-DE", "OPSD-FR", "OPSD-GB"]
BENCH_H, GRID_H = [96, 192, 336, 720], [24, 96, 336]
REF = "PLANET"


def pct(a, b):
    """Relative improvement of a over b, in per cent (positive = a better)."""
    return 100.0 * (b - a) / b if (b and np.isfinite(a) and np.isfinite(b)) else float("nan")


def build():
    recs = an.main_records()
    R = {"n_records": len(recs)}
    models = [m for m in an.ORDER + ["HistGBR"]
              if any((d, m, H, 2021) in recs for d in ETT + ["ECL"] + GRID
                     for H in BENCH_H + GRID_H)]

    # ---------------------------------------------------------- headline benchmark
    bench = {}
    for d in ETT + ["ECL"]:
        hs = [H for H in BENCH_H if (d, REF, H, 2021) in recs]
        if not hs:
            continue
        row = {}
        for m in models:
            v = [recs[(d, m, H, 2021)]["mse"] for H in hs if (d, m, H, 2021) in recs]
            a = [recs[(d, m, H, 2021)]["mae"] for H in hs if (d, m, H, 2021) in recs]
            if len(v) == len(hs):
                row[m] = dict(mse=float(np.mean(v)), mae=float(np.mean(a)))
        if REF in row:
            others = {k: v for k, v in row.items() if k != REF}
            best = min(others, key=lambda k: others[k]["mse"]) if others else None
            bench[d] = dict(per_model=row, best_baseline=best,
                            gain_mse=pct(row[REF]["mse"], others[best]["mse"]) if best else None,
                            gain_mae=pct(row[REF]["mae"], others[best]["mae"]) if best else None,
                            n_h=len(hs))
    R["benchmark"] = bench

    # cell-level wins and average rank
    cells, wins, ranks = 0, 0, {m: [] for m in models}
    for d in ETT + ["ECL"] + GRID:
        hs = BENCH_H if d not in GRID else GRID_H
        for H in hs:
            present = [m for m in models if (d, m, H, 2021) in recs]
            if REF not in present or len(present) < 3:
                continue
            vals = [recs[(d, m, H, 2021)]["mse"] for m in present]
            order = np.argsort(np.argsort(vals)) + 1
            for m, r in zip(present, order):
                ranks[m].append(int(r))
            cells += 1
            if present[int(np.argmin(vals))] == REF:
                wins += 1
    R["cells"] = dict(total=cells, planet_best=wins,
                      avg_rank={m: round(float(np.mean(v)), 2) for m, v in ranks.items() if v},
                      n_models=len({m for m in models if ranks[m]}))

    # ---------------------------------------------------------- national grids
    grids = {}
    for d in GRID:
        for H in GRID_H:
            present = {m: recs[(d, m, H, 2021)] for m in models if (d, m, H, 2021) in recs}
            if REF not in present:
                continue
            others = {k: v for k, v in present.items() if k != REF}
            if not others:
                continue
            bb = min(others, key=lambda k: others[k].get("wape", np.inf))
            grids[f"{d}_H{H}"] = dict(
                planet_wape=present[REF].get("wape"), best_baseline=bb,
                best_wape=others[bb].get("wape"),
                gain=pct(present[REF].get("wape"), others[bb].get("wape")),
                planet_peak=present[REF].get("peak_mape"),
                best_peak=others[bb].get("peak_mape"),
                planet_dt=present[REF].get("peak_time_err"),
                best_dt=others[bb].get("peak_time_err"),
                planet_mae_mw=present[REF].get("mae_orig"))
    R["grids"] = grids
    gains = [v["gain"] for v in grids.values() if np.isfinite(v.get("gain", np.nan))]
    R["grid_gain_mean"] = round(float(np.mean(gains)), 2) if gains else None
    R["grid_gain_range"] = [round(float(np.min(gains)), 2), round(float(np.max(gains)), 2)] if gains else None

    # ---------------------------------------------------------- ablations
    abl = {}
    from run_all import ABLATIONS
    for k in [x for x in ABLATIONS if x != "full"]:
        ds = []
        for d in ["OPSD-DE", "ECL", "ETTh1"]:
            for H in [96, 336]:
                f = recs.get((d, REF, H, 2021))
                p = os.path.join(RES, f"{d}_PLANET_L336_H{H}_s2021_abl-{k}.json")
                if f and os.path.exists(p):
                    a = json.load(open(p))
                    ds.append(dict(cell=f"{d}/H{H}", full=f["mse"], abl=a["mse"],
                                   delta=100.0 * (a["mse"] - f["mse"]) / f["mse"]))
        if ds:
            abl[k] = dict(cells=ds, mean_delta=round(float(np.mean([x["delta"] for x in ds])), 2),
                          max_delta=round(float(np.max([x["delta"] for x in ds])), 2))
    R["ablation"] = abl

    # relative-vs-absolute basis, broken out by horizon (the central novelty claim)
    rel = {}
    for d in ["OPSD-DE", "ECL", "ETTh1"]:
        for H in [96, 336]:
            f = recs.get((d, REF, H, 2021))
            p = os.path.join(RES, f"{d}_PLANET_L336_H{H}_s2021_abl-relative.json")
            if f and os.path.exists(p):
                rel[f"{d}_H{H}"] = round(100.0 * (json.load(open(p))["mse"] - f["mse"]) / f["mse"], 2)
    R["relative_basis_cost"] = rel

    # ---------------------------------------------------------- significance
    dm = {}
    for m in [x for x in models if x != REF]:
        w = t = l = 0
        for d in ETT + ["ECL"] + GRID:
            for H in (BENCH_H if d not in GRID else GRID_H):
                e1, e2 = an.per_window(d, REF, 336, H), an.per_window(d, m, 336, H)
                if e1 is None or e2 is None or len(e1) != len(e2):
                    continue
                s, p = an.dm_test(e1, e2, h=H)
                if not np.isfinite(p):
                    continue
                if p < 0.05 and s < 0: w += 1
                elif p < 0.05 and s > 0: l += 1
                else: t += 1
        if w + t + l:
            dm[m] = dict(win=w, tie=t, loss=l)
    R["dm"] = dm
    R["dm_total"] = dict(win=sum(v["win"] for v in dm.values()),
                         tie=sum(v["tie"] for v in dm.values()),
                         loss=sum(v["loss"] for v in dm.values()))

    # ---------------------------------------------------------- cost
    cost = {}
    for m in ["PLANET", "PatchTST", "DLinear", "iTransformer", "Autoformer", "FITS"]:
        r = recs.get(("OPSD-DE", m, 96, 2021))
        if r:
            cost[m] = dict(params=r["params"], train_s=r["train_time_s"],
                           infer_s=r["infer_time_s"], epochs=r["epochs_run"])
    if "PLANET" in cost and "PatchTST" in cost:
        cost["overhead_vs_patchtst_pct"] = round(
            100.0 * (cost["PLANET"]["train_s"] / max(cost["PLANET"]["epochs"], 1)) /
            (cost["PatchTST"]["train_s"] / max(cost["PatchTST"]["epochs"], 1)) - 100.0, 1)
    R["cost"] = cost

    # ---------------------------------------------------------- day-ahead vs TSO
    p = os.path.join(RES, "dayahead.json")
    if os.path.exists(p):
        da = json.load(open(p))
        R["dayahead"] = {g: {m: round(v["wape"], 3) for m, v in c.items() if m != "_protocol"}
                         for g, c in da.items()}
        R["dayahead_vs_tso"] = {
            g: round(pct(c.get(REF, {}).get("wape", np.nan), c["TSO"]["wape"]), 2)
            for g, c in da.items() if "TSO" in c and REF in c}

    # ---------------------------------------------------------- probabilistic
    PROBD = os.path.join(RES, "prob")
    if os.path.isdir(PROBD):
        pr = {}
        for f in sorted(glob.glob(os.path.join(PROBD, "*_L336_H24_s2021.json"))):
            r = json.load(open(f))
            pr[f"{r['dataset']}_{r['model']}"] = {
                k: round(r[k], 3) for k in
                ["crps_cqr", "picp_raw", "picp_cqr", "picp_aci", "mpiw_aci", "winkler_aci"]
                if k in r}
        R["prob"] = pr

    # ---------------------------------------------------------- weather pathway
    exo = {}
    for g in GRID:
        for H in [24, 96]:
            base = recs.get((g, REF, H, 2021))
            if not base:
                continue
            for mode in ["persist", "noisy", "perfect"]:
                f = os.path.join(RES, f"{g}_PLANETX_L336_H{H}_s2021_exo-{mode}.json")
                if os.path.exists(f):
                    r = json.load(open(f))
                    exo[f"{g}_H{H}_{mode}"] = dict(
                        wape=round(r["wape"], 3), base_wape=round(base["wape"], 3),
                        gain=round(pct(r["wape"], base["wape"]), 2))
    R["exo"] = exo
    for mode in ["persist", "noisy", "perfect"]:
        v = [x["gain"] for k, x in exo.items() if k.endswith(mode)]
        if v:
            R[f"exo_gain_{mode}"] = round(float(np.mean(v)), 2)

    # ---------------------------------------------------------- regime-shift robustness
    # Three independent shifts fall inside the OPSD test window. Year-end holidays recur
    # every year and are the *recurring* case; the 2020 lockdown is the one-off case. We
    # also recompute the headline ranking with the lockdown excised, so no claim in the
    # paper depends on that single event.
    SHIFTS = {
        "yearend_2019": ("2019-12-20", "2020-01-06"),
        "lockdown_2020": ("2020-03-16", "2020-05-10"),
        "recovery_2020": ("2020-05-11", "2020-06-30"),
    }
    CALM = ("2019-09-01", "2019-12-19")
    cov = {}
    for m in ["PLANET", "CalTST", "PatchTST", "iTransformer", "NBEATS", "DLinear",
              "TimeMixer", "HistGBR", "SeasonalNaive"]:
        f = os.path.join(ERR, f"OPSD-DE_{m}_L336_H24_s2021_pred.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        pz = np.load(os.path.join(PROC, "OPSD-DE.npz"))
        meta = json.load(open(os.path.join(PROC, "OPSD-DE.json")))
        a, b = meta["splits"]["train"]
        col = pz["data"][a:b, meta["target_idx"]].astype(np.float64)
        mu_, sd_ = col.mean(), (col.std() or 1.0)
        pred, true = z["pred"] * sd_ + mu_, z["true"] * sd_ + mu_
        t = pd.to_datetime(pz["stamps"], unit="s")[z["idx"] + 336]
        ae = pd.Series(np.abs(pred - true).sum(1), index=t).sort_index()
        ab = pd.Series(np.abs(true).sum(1), index=t).sort_index()
        wape = lambda s0, s1: 100.0 * ae.loc[s0:s1].sum() / ab.loc[s0:s1].sum()
        row = {"calm": round(float(wape(*CALM)), 3)}
        for nm, (s0, s1) in SHIFTS.items():
            row[nm] = round(float(wape(s0, s1)), 3)
            row[nm + "_excess"] = round(float(row[nm] - row["calm"]), 3)
        # headline WAPE with the lockdown window removed entirely
        lk0 = pd.Timestamp(SHIFTS["lockdown_2020"][0])
        lk1 = pd.Timestamp(SHIFTS["lockdown_2020"][1]) + pd.Timedelta(days=1)
        keep = ~((ae.index >= lk0) & (ae.index < lk1))   # whole final day, not 00:00 of it
        row["all"] = round(float(100.0 * ae.sum() / ab.sum()), 3)
        row["all_ex_lockdown"] = round(float(100.0 * ae[keep].sum() / ab[keep].sum()), 3)
        cov[m] = row
    R["regime_shift"] = cov

    return R


def summarise(R):
    out = []
    A = out.append
    A(f"records: {R['n_records']}   cells: {R['cells']['total']}   "
      f"PLANET best in {R['cells']['planet_best']}")
    ar = R["cells"]["avg_rank"]
    A("avg MSE rank: " + ", ".join(f"{k}={v}" for k, v in sorted(ar.items(), key=lambda x: x[1])))
    A("")
    A("-- benchmark (mean over horizons) --")
    for d, v in R["benchmark"].items():
        A(f"  {d:7s} PLANET MSE={v['per_model']['PLANET']['mse']:.4f}  "
          f"best baseline {v['best_baseline']}={v['per_model'][v['best_baseline']]['mse']:.4f}  "
          f"gain {v['gain_mse']:+.2f}% MSE / {v['gain_mae']:+.2f}% MAE")
    A("")
    A("-- national grids (WAPE %) --")
    for k, v in R["grids"].items():
        A(f"  {k:14s} PLANET={v['planet_wape']:.3f}  best {v['best_baseline']}={v['best_wape']:.3f}"
          f"  gain {v['gain']:+.2f}%   peak {v['planet_peak']:.2f} vs {v['best_peak']:.2f}")
    if R.get("grid_gain_mean") is not None:
        A(f"  mean WAPE gain over best baseline: {R['grid_gain_mean']:+.2f}% "
          f"(range {R['grid_gain_range']})")
    A("")
    A("-- ablations (mean dMSE% when removed) --")
    for k, v in sorted(R["ablation"].items(), key=lambda x: -x[1]["mean_delta"]):
        A(f"  {k:16s} {v['mean_delta']:+7.2f}%  (max {v['max_delta']:+.2f}%)")
    if R.get("relative_basis_cost"):
        A("  relative-time basis cost by cell: " +
          ", ".join(f"{k}={v:+.1f}%" for k, v in R["relative_basis_cost"].items()))
    A("")
    if R.get("dm_total"):
        t = R["dm_total"]
        A(f"-- Diebold-Mariano: {t['win']} wins / {t['tie']} ties / {t['loss']} losses --")
    if R.get("dayahead_vs_tso"):
        A("-- vs TSO day-ahead (WAPE gain %) --")
        for g, v in R["dayahead_vs_tso"].items():
            A(f"  {g}: {v:+.2f}%   " +
              "  ".join(f"{m}={w}" for m, w in sorted(R['dayahead'][g].items(), key=lambda x: x[1])))
    if R.get("regime_shift"):
        A("-- regime-shift robustness, OPSD-DE H=24 (WAPE %, excess over calm baseline) --")
        A(f"  {'model':14s} {'calm':>6s} {'yearend':>16s} {'lockdown':>16s} "
          f"{'recovery':>16s} {'all':>7s} {'ex-lock':>8s}")
        for m, v in sorted(R["regime_shift"].items(), key=lambda x: x[1]["all"]):
            A(f"  {m:14s} {v['calm']:6.2f} "
              f"{v['yearend_2019']:7.2f} (+{v['yearend_2019_excess']:.2f}) "
              f"{v['lockdown_2020']:7.2f} (+{v['lockdown_2020_excess']:.2f}) "
              f"{v['recovery_2020']:7.2f} (+{v['recovery_2020_excess']:.2f}) "
              f"{v['all']:7.2f} {v['all_ex_lockdown']:8.2f}")
    if R.get("prob"):
        A("-- probabilistic (H=24) --")
        for k, v in R["prob"].items():
            A(f"  {k:26s} CRPS={v.get('crps_cqr')}  PICP raw/CQR/ACI="
              f"{v.get('picp_raw')}/{v.get('picp_cqr')}/{v.get('picp_aci')}  "
              f"MPIW={v.get('mpiw_aci')}  Winkler={v.get('winkler_aci')}")
    if R.get("exo"):
        A("-- weather pathway (mean WAPE gain over PLANET) --")
        for mode in ["persist", "noisy", "perfect"]:
            if f"exo_gain_{mode}" in R:
                A(f"  {mode:9s} {R[f'exo_gain_{mode}']:+.2f}%")
    if R.get("cost"):
        A("-- cost (OPSD-DE, H=96) --")
        for m, v in R["cost"].items():
            if isinstance(v, dict):
                A(f"  {m:14s} {v['params']/1e3:8.1f}k params  {v['train_s']:6.0f}s / "
                  f"{v['epochs']} ep  infer {v['infer_s']:.1f}s")
        if "overhead_vs_patchtst_pct" in R["cost"]:
            A(f"  per-epoch overhead of PLANET vs PatchTST: "
              f"{R['cost']['overhead_vs_patchtst_pct']:+.1f}%")
    return "\n".join(out)


if __name__ == "__main__":
    R = build()
    json.dump(R, open(os.path.join(RES, "report.json"), "w"), indent=2, default=float)
    txt = summarise(R)
    open(os.path.join(RES, "report.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
