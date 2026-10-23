"""Day-ahead protocol with monthly refits (revision, 2026-10-19).

The headline day-ahead comparison scores models trained once, at the end of the training split,
against operators who refit continuously. This script retrains the two reference learned models
(PLANET and TiDE) from scratch at the start of every test month on every row before that month,
with early stopping on the two months preceding it, and forecasts the month's 48-hour windows.
The monthly forecasts are merged into one saved-prediction file in the protocol's z-units,
results_v2/errors/<ds>_<model>_L336_H48_s2021_v2-R2-refit_pred.npz, which dayahead_v2.py scores
under the operator's information cut-off like every other entry (model name "<model>-refit").

Per month the training set is [0, m0 - VAL_H), the validation set [m0 - VAL_H, m0) and the
forecast origins the hours of the month; every window's targets lie after its training data, so
nothing is leaked. One seed (2021); the tuned learning rate, width, backbone and the protocol
lookback of 336 of the point model. Resumable: finished monthly runs are skipped.
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_v2 as exp
import run_all_v2 as r
from data_v2 import PROC2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_v2")
ERR = os.path.join(RES, "errors")
MODELS = ["PLANET", "TiDE"]
SEED, L, H = 2021, 336, 48
VAL_H = 24 * 61                                   # two months of validation before each refit


def test_months(meta, stamps):
    """(month code, first row, last row + 1) of every UTC month of the test split."""
    te0, te1 = meta["splits"]["test"]
    st = pd.to_datetime(stamps.astype(np.int64), unit="s")
    mcode = (st.year * 12 + st.month - 1).to_numpy()
    out = []
    for m in np.unique(mcode[te0:te1]):
        m0, m1 = int(np.searchsorted(mcode, m)), int(np.searchsorted(mcode, m + 1))
        out.append((int(m), max(m0, te0), min(m1, te1)))
    return out


def run_system(d, model):
    z = np.load(os.path.join(PROC2, d + ".npz"))
    meta = json.load(open(os.path.join(PROC2, d + ".json")))
    stamps = z["stamps"]
    T = len(stamps)
    data32 = z["data"].astype(np.float32)
    tgt = meta["target_idx"]
    a, b = meta["splits"]["train"]
    mu_p, sd_p = float(data32[a:b, tgt].mean()), float(data32[a:b, tgt].std())   # protocol scaler
    lr, k = r.best_tune(d, model)
    bb = r.best_bb(model)
    parts, prov = [], []
    for m, m0, m1 in test_months(meta, stamps):
        ym = f"{m // 12:04d}{m % 12 + 1:02d}"
        tag = f"_v2-R2-refit-m{ym}"
        tr1 = m0 - VAL_H
        j = r.A(data=d, model=model, L=L, H=H, seed=SEED, lr=lr, size=k, backbone=bb, collect=-1,
                tag=tag, splits=f"0,{tr1},{m0},{min(m1 + H, T)}", **r.DS[d])
        t0 = time.time()
        if not r.done(j):
            exp.run(j)
        p = np.load(os.path.join(ERR, j.key() + "_pred.npz"))
        mu_r, sd_r = float(data32[:tr1, tgt].mean()), float(data32[:tr1, tgt].std())  # refit scaler
        sd_r = sd_r if sd_r >= 1e-6 else 1.0
        idx = p["idx"].astype(np.int64)
        origin = idx + L - 1
        keep = (origin >= m0) & (origin < m1)
        phys_p = p["pred"][keep] * sd_r + mu_r
        phys_t = p["true"][keep] * sd_r + mu_r
        parts.append((idx[keep], (phys_p - mu_p) / sd_p, (phys_t - mu_p) / sd_p))
        prov.append(dict(month=ym, train_rows=tr1, val_rows=[tr1, m0], origins=[m0, m1], n_windows=int(keep.sum()),
                         seconds=round(time.time() - t0, 1)))
        print(f"  [refit] {d} {model} {ym}: {int(keep.sum())} windows, train rows {tr1}", flush=True)
    idx = np.concatenate([q[0] for q in parts])
    order = np.argsort(idx)
    stem = f"{d}_{model}_L{L}_H{H}_s{SEED}_v2-R2-refit"
    np.savez_compressed(os.path.join(ERR, stem + "_pred.npz"), idx=idx[order],
                        pred=np.concatenate([q[1] for q in parts])[order].astype(np.float32),
                        true=np.concatenate([q[2] for q in parts])[order].astype(np.float32))
    # provenance only (no "mse" key, so the analysis does not read it as a result record)
    json.dump(dict(dataset=d, model=model + "-refit", L=L, H=H, seed=SEED, lr=lr, size=k, backbone=bb,
                   val_hours=VAL_H, months=prov, protocol="monthly refit from scratch; see refit_dayahead.py"),
              open(os.path.join(RES, stem + ".json"), "w"), indent=2)
    print(f"[refit] {d} {model}: {len(idx)} windows over {len(parts)} months -> {stem}_pred.npz", flush=True)


def main():
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else r.GRID
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS
    for d in datasets:
        for m in models:
            stem = f"{d}_{m}_L{L}_H{H}_s{SEED}_v2-R2-refit"
            if os.path.exists(os.path.join(ERR, stem + "_pred.npz")):
                print(f"[refit] {stem} exists, skipped", flush=True)
                continue
            run_system(d, m)


if __name__ == "__main__":
    main()
