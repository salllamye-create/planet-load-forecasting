"""Day-ahead operator protocol, v2 (eight national systems, every regime and seed).

Every learned model is held to the transmission operator's own information cut-off: the
last observed hour is 11:00 UTC on day D-1 and the scored block is 00:00-23:00 UTC of day D
(lead times 13-36 h). The TSO's published day-ahead forecast is scored on exactly the same
target timestamps. Learned forecasts come from the saved H=48 test predictions written by
exp_v2.py with --collect -1 (results_v2/errors/<tag>_pred.npz); the models are therefore
trained once, at the 48 h horizon, and are not refitted during the test period.

Revision additions (2026-10-19):
  * a second origin at 08:00 UTC, the publication deadline of the day-ahead market in most of
    the systems (leads 16-39 h), written to results_v2/dayahead_08.json;
  * the seed ensemble (mean of the seed forecasts of a model) scored as its own entry
    ("<model>|L<L>|ens<suffix>", model name "<model>-ens"); cross-model combinations, the
    equal-weight mean of the member forecasts per seed ("<combo>|L0|s<seed><suffix>", model name
    "<combo>") and of the member seed ensembles ("<combo>|L0|ens<suffix>", "<combo>-ens"), for
    Comb2 = PLANET + TiDE and CombAll = the ten headline models (COMBOS);
  * a Diebold-Mariano test of every learned entry against the operator on the daily mean
    absolute errors (HAC variance, Bartlett lag n^(1/3)); stat < 0 means the learned forecast
    has the lower loss;
  * monthly WAPE, WAPE inside / outside the first-wave lockdown window of spring 2020 and on
    holiday days (a scored day is a holiday day when at least 12 of its 24 target hours carry
    the national-holiday flag), for every entry including the operator;
  * days whose target block contains a corrupt hour (data_v2.BAD_HOURS) are dropped.

Output: results_v2/dayahead.json (11:00 UTC origin) and results_v2/dayahead_08.json (08:00)
  {dataset: {"TSO": metrics, "<model>|L<L>|s<seed><suffix>": metrics, ..., "_protocol": {...}}}
where <suffix> is the record suffix (e.g. _v2-R2); the analysis averages the seed entries of
each (model, suffix) and selects the regime by the suffix.
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_v2 import bad_hours_idx
from stats import dm_test

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "proc_v2")
RES = os.path.join(HERE, "results_v2")
ERR = os.path.join(RES, "errors")
# name -> (origin hour UTC, first lead index, last lead index + 1) into the H=48 forecast:
# target step j of a window starting at s is at row s + L + j, the origin row is s + L - 1
ORIGINS = {"dayahead": (11, 12, 36), "dayahead_08": (8, 15, 39)}
ORIGIN_HOUR, LEAD0, LEAD1 = ORIGINS["dayahead"]
LOCKDOWN = (pd.Timestamp("2020-03-10"), pd.Timestamp("2020-05-31"))   # first-wave window, UTC days
HOLIDAY_MIN_HOURS = 12
TRAIN_H = 48
HEADLINE = ["NBEATS", "PatchTST", "CalTST", "PLANET", "PLANETX", "TFT", "TiDE", "NHITS", "TimeXer", "DLinear"]
COMBOS = {"Comb2": ["PLANET", "TiDE"], "CombAll": HEADLINE}


def wape(pred, true):
    return 100.0 * float(np.abs(pred - true).sum() / max(np.abs(true).sum(), 1e-9))


def metrics(pred, true):
    e = pred - true
    out = dict(bias_pct=float(100.0 * e.mean() / true.mean()),
               mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
               wape=wape(pred, true),
               mape=100.0 * float((np.abs(e) / np.maximum(np.abs(true), 1e-6)).mean()),
               bias=float(e.mean()),
               ramp_mae=float(np.abs(np.diff(pred, 1) - np.diff(true, 1)).mean()),
               n_days=int(len(true)))
    yp, ypi = true.max(1), true.argmax(1); pp, ppi = pred.max(1), pred.argmax(1)
    out.update(peak_mae=float(np.abs(pp - yp).mean()),
               peak_mape=100.0 * float((np.abs(pp - yp) / np.maximum(yp, 1e-6)).mean()),
               peak_time_err=float(np.abs(ppi - ypi).mean()))
    return out


def tallies(pred, true, days, hol):
    """Monthly / lockdown / holiday breakdown of the WAPE (days: DatetimeIndex of the scored
    days, hol: bool per day)."""
    out = {}
    mon = days.strftime("%Y-%m")
    out["monthly_wape"] = {m: round(wape(pred[mon == m], true[mon == m]), 3) for m in sorted(set(mon))}
    lock = (days >= LOCKDOWN[0]) & (days <= LOCKDOWN[1])
    out["wape_lockdown"] = wape(pred[lock], true[lock]) if lock.any() else float("nan")
    out["wape_excl_lockdown"] = wape(pred[~lock], true[~lock]) if (~lock).any() else float("nan")
    out["n_days_lockdown"] = int(lock.sum())
    out["wape_holiday"] = wape(pred[hol], true[hol]) if hol.any() else float("nan")
    out["wape_nonholiday"] = wape(pred[~hol], true[~hol]) if (~hol).any() else float("nan")
    out["n_days_holiday"] = int(hol.sum())
    return out


def dm_vs(pred, true, tso):
    """Diebold-Mariano on daily mean absolute errors, learned (e1) vs operator (e2)."""
    e1 = np.abs(pred - true).mean(1); e2 = np.abs(tso - true).mean(1)
    s, p = dm_test(e1, e2, h=1)
    return dict(dm_stat=s, dm_p=p)


def run_origin(name, L=336, H=TRAIN_H, datasets=None):
    origin_hour, lead0, lead1 = ORIGINS[name]
    out = {}
    names = datasets or sorted(f[:-5] for f in os.listdir(PROC) if f.startswith("OPSD-") and f.endswith(".json"))
    for d in names:
        z = np.load(os.path.join(PROC, d + ".npz"))
        meta = json.load(open(os.path.join(PROC, d + ".json")))
        if "aux" not in z.files:
            continue
        stamps = pd.to_datetime(z["stamps"], unit="s")
        load = z["data"][:, meta["target_idx"]].astype(float)
        tso = z["aux"][:, 0].astype(float)
        holiday = z["cal_flags"][:, 0].astype(float) if "cal_flags" in z.files else np.zeros(len(load))
        te0, te1 = meta["splits"]["test"]
        a, b = meta["splits"]["train"]
        col = z["data"][a:b, meta["target_idx"]].astype(np.float64)
        mu_, sd_ = col.mean(), (col.std() or 1.0)

        starts = np.arange(max(0, te0 - L), te1 - L - H + 1)
        oh = stamps[starts + L - 1].hour.to_numpy()
        sel = starts[oh == origin_hour]
        tgt = sel[:, None] + L + np.arange(lead0, lead1)[None, :]
        keep = (tgt[:, -1] < te1) & np.isfinite(tso[tgt]).all(1)
        bad = np.zeros(len(load), bool)
        bad[bad_hours_idx(d, z["stamps"])] = True
        n_bad_days = int((keep & bad[tgt].any(1)).sum())
        keep &= ~bad[tgt].any(1)
        sel, tgt = sel[keep], tgt[keep]
        truth = load[tgt]
        days = pd.DatetimeIndex(stamps[tgt[:, 0]]).normalize()
        hol = holiday[tgt].sum(1) >= HOLIDAY_MIN_HOURS
        # The published operator series of some systems (NL, GB) sits several percent below the
        # published actual load, a perimeter difference between the two ENTSO-E series rather
        # than forecast error, and the offset is not constant over time (a validation-estimated
        # correction over-corrects DE and under-corrects NL), so no correction is applied: the
        # operator is reported as published, its bias on the scored days is printed, and the
        # validation-span ratio of actual to forecast is kept here as a diagnostic.
        va0, va1 = meta["splits"]["val"]
        ok = np.isfinite(tso[va0:va1]) & np.isfinite(load[va0:va1])
        scale = float(load[va0:va1][ok].mean() / tso[va0:va1][ok].mean()) if ok.any() else 1.0
        tso_block = tso[tgt]
        res = {"TSO": dict(metrics(tso_block, truth), **tallies(tso_block, truth, days, hol)),
               "_protocol": dict(origin_hour_utc=origin_hour, lead_hours=[lead0 + 1, lead1],
                                 n_days=int(len(sel)), first=str(stamps[tgt[0, 0]]),
                                 last=str(stamps[tgt[-1, -1]]), tso_val_ratio=round(scale, 4),
                                 train_horizon=H, lookback_default=L, n_days_dropped_corrupt=n_bad_days,
                                 lockdown_window=[str(LOCKDOWN[0].date()), str(LOCKDOWN[1].date())],
                                 holiday_min_hours=HOLIDAY_MIN_HOURS, n_days_holiday=int(hol.sum()),
                                 dm="daily MAE, learned vs operator, HAC lag n^(1/3); stat<0 learned better")}
        groups, per_seed = {}, {}
        for p in sorted(glob.glob(os.path.join(ERR, f"{d}_*_L*_H{H}_s*_pred.npz"))):
            stem = os.path.basename(p)[:-len("_pred.npz")]
            m = re.match(rf"{re.escape(d)}_(.+?)_L(\d+)_H{H}_s(\d+)(.*)$", stem)
            if not m:
                continue
            model, Lr, seed, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            if "-refit" in suffix:                   # monthly-refit protocol (refit_dayahead.py)
                model, suffix = model + "-refit", suffix.replace("-refit", "")
            zz = np.load(p)
            pos = {int(v): i for i, v in enumerate(zz["idx"])}
            rows = [pos[int(s)] for s in sel if int(s) in pos]
            if len(rows) < 0.9 * len(sel):
                print(f"   [skip] {d}/{model}{suffix}: only {len(rows)}/{len(sel)} windows saved")
                continue
            pr = zz["pred"][rows][:, lead0:lead1] * sd_ + mu_
            tr = zz["true"][rows][:, lead0:lead1] * sd_ + mu_
            assert np.allclose(tr, truth[:len(rows)], rtol=1e-3, atol=1e-1), f"{d}/{model} target mismatch"
            n = len(rows)
            key = f"{model}|L{Lr}|s{seed}{suffix}"
            res[key] = metrics(pr, tr)
            res[key].update(tallies(pr, tr, days[:n], hol[:n]))
            res[key].update(dm_vs(pr, tr, tso_block[:n]))
            res[key].update(model=model, L=Lr, seed=seed, suffix=suffix)
            groups.setdefault((model, Lr, suffix), []).append(pr)
            per_seed[(model, suffix, seed)] = pr
        ens = {}
        for (model, Lr, suffix), preds in groups.items():
            if len(preds) < 2:
                continue
            n = min(len(p_) for p_ in preds)
            pr = np.mean([p_[:n] for p_ in preds], axis=0)
            ens[(model, suffix)] = pr
            key = f"{model}|L{Lr}|ens{suffix}"
            res[key] = metrics(pr, truth[:n])
            res[key].update(tallies(pr, truth[:n], days[:n], hol[:n]))
            res[key].update(dm_vs(pr, truth[:n], tso_block[:n]))
            res[key].update(model=model + "-ens", L=Lr, seed=None, suffix=suffix, n_members=len(preds))

        # cross-model combinations: the equal-weight mean of the member forecasts at the same
        # seed (seed-level entries) and of the member seed ensembles ("<combo>-ens")
        def _entry(key, pr, **fields):
            n = len(pr)
            res[key] = metrics(pr, truth[:n])
            res[key].update(tallies(pr, truth[:n], days[:n], hol[:n]))
            res[key].update(dm_vs(pr, truth[:n], tso_block[:n]))
            res[key].update(**fields)
        for suffix in sorted({suf for (_, suf, _) in per_seed}):
            for cname, members in COMBOS.items():
                for seed in sorted({s for (_, suf, s) in per_seed if suf == suffix}):
                    ps = [per_seed.get((m, suffix, seed)) for m in members]
                    if all(p_ is not None for p_ in ps):
                        n = min(len(p_) for p_ in ps)
                        _entry(f"{cname}|L0|s{seed}{suffix}", np.mean([p_[:n] for p_ in ps], axis=0),
                               model=cname, L=0, seed=seed, suffix=suffix, members=list(members))
                ps = [ens.get((m, suffix)) for m in members]
                if all(p_ is not None for p_ in ps):
                    n = min(len(p_) for p_ in ps)
                    _entry(f"{cname}|L0|ens{suffix}", np.mean([p_[:n] for p_ in ps], axis=0),
                           model=cname + "-ens", L=0, seed=None, suffix=suffix, n_members=len(ps),
                           members=list(members))
        out[d] = res
        learned = {k: v for k, v in res.items() if k not in ("TSO", "_protocol")}
        print(f"  [{name}] {d}: {res['_protocol']['n_days']} days ({n_bad_days} dropped), "
              f"TSO WAPE={res['TSO']['wape']:.2f}%, {len(learned)} learned entries")
    json.dump(out, open(os.path.join(RES, name + ".json"), "w"), indent=2)
    return out


def run(L=336, H=TRAIN_H, datasets=None):
    out = {}
    for name in ORIGINS:
        out[name] = run_origin(name, L, H, datasets)
    return out


if __name__ == "__main__":
    run()
