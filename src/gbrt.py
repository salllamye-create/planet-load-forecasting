"""Gradient-boosted regression trees on engineered load-forecasting features.

This is the workhorse of operational load forecasting and of the GEFCom-style
competition literature, so it belongs in any smart-grid benchmark. One model is
fitted per horizon step (the "direct" multi-horizon strategy), on lagged load,
calendar variables and lagged weather -- strictly past information only, exactly
the information set the neural models receive.
"""
import argparse, json, os, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "proc")
RES = os.path.join(HERE, "results")
ERR = os.path.join(RES, "errors")
os.makedirs(ERR, exist_ok=True)

LAGS = [1, 2, 3, 4, 5, 6, 12, 18, 24, 25, 26, 47, 48, 72, 96, 120, 144, 168, 169, 192, 336]
ROLL = [24, 168]


def build(name, L, H):
    z = np.load(os.path.join(PROC, name + ".npz"))
    meta = json.load(open(os.path.join(PROC, name + ".json")))
    data, stamps = z["data"].astype(np.float64), z["stamps"].astype(np.float64) / 3600.0
    tgt = meta["target_idx"]
    y = data[:, tgt]
    T = len(y)
    hod = np.mod(stamps, 24.0)
    dow = np.mod(np.floor(stamps / 24.0) + 4.0, 7.0)
    doy = np.mod(np.floor(stamps / 24.0), 365.25)

    feats, names = [], []
    for l in LAGS:
        if l > L:
            continue
        v = np.full(T, np.nan); v[l:] = y[:-l]
        feats.append(v); names.append(f"lag{l}")
    for w in ROLL:
        c = np.convolve(y, np.ones(w) / w, mode="full")[:T]
        v = np.full(T, np.nan); v[w:] = c[w - 1:-1]
        feats.append(v); names.append(f"roll{w}")
    # exogenous channels (weather, wind, solar): lagged by one step only -- past information
    for j, ch in enumerate(meta["channels"]):
        if j == tgt:
            continue
        v = np.full(T, np.nan); v[1:] = data[1:, j] * 0 + data[:-1, j]
        feats.append(v); names.append(f"{ch}_lag1")
    feats += [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
              np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
              np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
              hod, dow, (dow >= 5).astype(float)]
    names += ["sin_h", "cos_h", "sin_d", "cos_d", "sin_y", "cos_y", "hod", "dow", "weekend"]
    X = np.column_stack(feats)
    return X, y, meta, names


def run(a):
    X, y, meta, names = build(a.data, a.L, a.H)
    T = len(y)
    tr0, tr1 = meta["splits"]["train"]
    va0, va1 = meta["splits"]["val"]
    te0, te1 = meta["splits"]["test"]
    mu = y[tr0:tr1].mean(); sd = y[tr0:tr1].std()

    # window starts matching the neural protocol exactly
    # same stride as the neural methods, so "every method shares the same windows" holds
    starts_tr = np.arange(max(0, tr0), tr1 - a.L - a.H + 1, a.stride)
    starts_te = np.arange(max(0, te0 - a.L), te1 - a.L - a.H + 1)
    anchor_tr, anchor_te = starts_tr + a.L - 1, starts_te + a.L - 1   # last observed step

    ok = ~np.isnan(X).any(1)
    anchor_tr = anchor_tr[ok[anchor_tr]]
    preds = np.zeros((len(anchor_te), a.H))
    truth = np.stack([y[anchor_te + 1 + h] for h in range(a.H)], 1)
    t0 = time.time()
    for h in range(a.H):
        m = HistGradientBoostingRegressor(max_iter=a.iters, learning_rate=0.06,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=a.seed)
        m.fit(X[anchor_tr], y[anchor_tr + 1 + h])
        preds[:, h] = m.predict(X[anchor_te])
        if h % 12 == 0:
            print(f"   step {h+1}/{a.H} ({time.time()-t0:.0f}s)", flush=True)

    e = preds - truth
    zn = e / sd                                       # comparable to the z-scored neural metrics
    out = dict(dataset=a.data, model="HistGBR", L=a.L, H=a.H, seed=a.seed,
               mse=float((zn ** 2).mean()), mae=float(np.abs(zn).mean()),
               rmse_orig=float(np.sqrt((e ** 2).mean())), mae_orig=float(np.abs(e).mean()),
               wape=100.0 * float(np.abs(e).sum() / np.abs(truth).sum()),
               mape=100.0 * float((np.abs(e) / np.maximum(np.abs(truth), 1e-6)).mean()),
               ramp_mae=float(np.abs(np.diff(preds, axis=1) - np.diff(truth, axis=1)).mean()),
               n_windows=len(anchor_te), params=0, train_stride=a.stride,
               train_time_s=round(time.time() - t0, 1),
               epochs_run=0, val_mse=float("nan"), tag=a.tag, lr=0.06, batch=0,
               infer_time_s=0.0, amp=False, history=[])
    if a.H >= 24:
        nd = a.H // 24
        yd = truth[:, :nd * 24].reshape(-1, nd, 24); pdd = preds[:, :nd * 24].reshape(-1, nd, 24)
        yp, ypi = yd.max(-1), yd.argmax(-1); pp, ppi = pdd.max(-1), pdd.argmax(-1)
        out["peak_mae"] = float(np.abs(pp - yp).mean())
        out["peak_mape"] = 100.0 * float((np.abs(pp - yp) / np.maximum(np.abs(yp), 1e-6)).mean())
        out["peak_time_err"] = float(np.abs(ppi - ypi).mean())
    tag = f"{a.data}_HistGBR_L{a.L}_H{a.H}_s{a.seed}{a.tag}"
    json.dump(out, open(os.path.join(RES, tag + ".json"), "w"), indent=2)
    np.savez_compressed(os.path.join(ERR, tag + ".npz"),
                        per_window=(zn ** 2).mean(1),
                        per_window_ae=np.abs(e).mean(1),      # WAPE-consistent DM loss
                        per_step_mse=(zn ** 2).mean(0), per_step_mae=np.abs(zn).mean(0))
    print(f"[{tag}] MSE={out['mse']:.4f} MAE={out['mae']:.4f} WAPE={out['wape']:.2f}% "
          f"({out['train_time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--tag", type=str, default="")
    run(ap.parse_args())
