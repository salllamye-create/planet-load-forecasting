"""Day-ahead protocol: compare every method against the TSO's own published forecast.

ENTSO-E day-ahead load forecasts are issued before the ~12:00 D-1 market gate. To make
the comparison honest, the learned models are held to the same information cut-off:
forecasts are taken from windows whose last observed hour is 11:00 UTC on day D-1, and
scored on steps 13-36 of the horizon, i.e. the 24 hours 00:00-23:00 UTC of day D
(lead times 13-36 h). The TSO series is scored on exactly the same target timestamps.
"""
import glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC, RES, ERR = (os.path.join(HERE, "data", "proc"), os.path.join(HERE, "results"),
                  os.path.join(HERE, "results", "errors"))

ORIGIN_HOUR = 11        # last observed hour, UTC, on day D-1
LEAD0, LEAD1 = 12, 36   # 0-indexed slice of the horizon -> steps 13..36


def metrics(pred, true):
    e = pred - true
    out = dict(
        mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
        wape=100.0 * float(np.abs(e).sum() / np.abs(true).sum()),
        mape=100.0 * float((np.abs(e) / np.maximum(np.abs(true), 1e-6)).mean()),
        bias=float(e.mean()),
        ramp_mae=float(np.abs(np.diff(pred, 1) - np.diff(true, 1)).mean()),
        n_days=int(len(true)),
    )
    yp, ypi = true.max(1), true.argmax(1)
    pp, ppi = pred.max(1), pred.argmax(1)
    out.update(peak_mae=float(np.abs(pp - yp).mean()),
               peak_mape=100.0 * float((np.abs(pp - yp) / np.maximum(yp, 1e-6)).mean()),
               peak_time_err=float(np.abs(ppi - ypi).mean()))
    return out


def run(L=336, H=48):
    out = {}
    for d in ["OPSD-DE", "OPSD-FR", "OPSD-GB"]:
        z = np.load(os.path.join(PROC, d + ".npz"))
        meta = json.load(open(os.path.join(PROC, d + ".json")))
        if "aux" not in z.files:
            continue
        stamps = pd.to_datetime(z["stamps"], unit="s")
        load = z["data"][:, meta["target_idx"]].astype(float)
        tso = z["aux"][:, 0].astype(float)
        te0, te1 = meta["splits"]["test"]

        # candidate window starts inside the test split, matching the neural protocol
        starts = np.arange(max(0, te0 - L), te1 - L - H + 1)
        oh = stamps[starts + L - 1].hour.to_numpy()
        sel = starts[oh == ORIGIN_HOUR]
        tgt = sel[:, None] + L + np.arange(LEAD0, LEAD1)[None, :]
        keep = (tgt[:, -1] < te1) & np.isfinite(tso[tgt]).all(1)
        sel, tgt = sel[keep], tgt[keep]
        truth = load[tgt]

        res = {"TSO": metrics(tso[tgt], truth)}
        res["_protocol"] = dict(origin_hour_utc=ORIGIN_HOUR, lead_hours=[LEAD0 + 1, LEAD1],
                                n_days=int(len(sel)),
                                first=str(stamps[tgt[0, 0]]), last=str(stamps[tgt[-1, -1]]))

        # same-day comparison for every model that has saved H=48 forecasts
        for p in sorted(glob.glob(os.path.join(ERR, f"{d}_*_L{L}_H{H}_s2021_pred.npz"))):
            m = os.path.basename(p)[len(d) + 1:].split(f"_L{L}_")[0]
            zz = np.load(p)
            # exp.py stores forecasts z-scored; restore physical units
            a, b = meta["splits"]["train"]
            col = z["data"][a:b, meta["target_idx"]].astype(np.float64)
            mu_, sd_ = col.mean(), (col.std() or 1.0)
            pos = {int(v): i for i, v in enumerate(zz["idx"])}
            rows = [pos[int(s)] for s in sel if int(s) in pos]
            if len(rows) < 0.9 * len(sel):
                print(f"   [skip] {d}/{m}: only {len(rows)}/{len(sel)} windows saved")
                continue
            pr = zz["pred"][rows][:, LEAD0:LEAD1] * sd_ + mu_
            tr = zz["true"][rows][:, LEAD0:LEAD1] * sd_ + mu_
            assert np.allclose(tr, truth[:len(rows)], rtol=1e-3, atol=1e-1), f"{d}/{m} target mismatch"
            res[m] = metrics(pr, tr)
        out[d] = res
        print(f"  {d}: {res['_protocol']['n_days']} day-ahead days, "
              f"TSO WAPE={res['TSO']['wape']:.2f}%  " +
              "  ".join(f"{k}={v['wape']:.2f}%" for k, v in res.items()
                        if k not in ("TSO", "_protocol")))
    json.dump(out, open(os.path.join(RES, "dayahead.json"), "w"), indent=2)

    # a flat view keyed by dataset, for the figure helper
    flat = {d: v.get("TSO", {}) for d, v in out.items()}
    json.dump(flat, open(os.path.join(RES, "tso_baseline.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    run()
