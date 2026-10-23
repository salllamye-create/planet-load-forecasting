"""GBRT (HistGradientBoostingRegressor) under the v2 information regimes (V2_SPEC.md).

The v1 tree baseline (gbrt.py) is kept in everything that matters -- the boosting
hyper-parameters, the direct multi-horizon strategy (one model per horizon step), the window
anchoring and stride, the metrics and the result files -- with the feature set decided by the
regime instead of being fixed:

  R0  lagged load and rolling means of the load only (past information, no calendar);
  R1  R0 + the 8-column calendar block of the TARGET step of each horizon model (sin/cos
      local hour, sin/cos local day-of-week, weekend, holiday, bridge, xmas): known future;
  R2  R1 + weather: the lag-1 past weather of v1, plus the weather at the target step
      degraded to NWP skill exactly as exp_v2.degrade_cov: Gaussian noise with
      sd = data_v2.nwp_noise_phys(channel, training split) * sqrt(lead / 24) in physical
      units (lead = h + 1; temp 1.2 K, rhum 6, rad 40, wspd 1.1 absolute; wind / solar
      generation 8% / 5% of the training 99.9th percentile), seeded. The rule is recorded as
      "nwp_rule": "v2-capacity-relative" with the per-channel sd in the JSON.
      The noise is applied to the training rows as well, as exp_v2 degrades the block in
      every training epoch of the neural models.  --exo_mode perfect|persist as exp_v2.

Reads data/proc_v2/, writes results_v2/<tag>.json + results_v2/errors/<tag>.npz with
tag = {data}_GBRT_L{L}_H{H}_s{seed}_v2-{regime}; resumable (an existing JSON is skipped).
Seeds: every listed seed for R2, the first one for R0/R1 (run_all_v2 rule).

Feature alignment: v1 used information up to anchor-1 only; here the features at the anchor
include the anchor hour itself (lag_fix=1, recorded as "lag_fix" in the JSON).

Note on the seed: without early stopping and with < 200k rows HistGradientBoostingRegressor
is deterministic, so across R2 seeds only the NWP noise changes.

Usage
  python src/gbrt_v2.py --datasets OPSD-DE --horizons 24 --regimes R2 --seeds 2021
  for i in 0 1 2 3; do OMP_NUM_THREADS=1 python src/gbrt_v2.py --shard $i/4 & done   # sweep
"""
import argparse, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classic_v2 import (GRIDS, CAL_COLS, RES_DEFAULT, load_v2, starts, metrics,
                        record, write_result, done, parse_list, seeds_for)

LAGS = [1, 2, 3, 4, 5, 6, 12, 18, 24, 25, 26, 47, 48, 72, 96, 120, 144, 168, 169, 192, 336]
ROLL = [24, 168]
HP = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
          early_stopping=False)                                         # v1 gbrt.py


def static_features(y, data, widx, channels, L, lag_fix=False):
    """v1 feature construction: lagged load, rolling means, lag-1 exogenous channels.
    In v1 the features at index t used information up to t-1 only (lag1[t] = y[t-1]; the
    anchor hour itself was unused). With lag_fix (the default) they are shifted by one so
    that the features at the anchor use everything up to and including the anchor hour
    (lag1[anchor] = y[anchor], roll24 = mean of the last 24 observed hours, weather at the
    anchor): the information set of the neural models. --lag_fix 0 (tag marker -v1lag)
    reproduces the v1 alignment."""
    T = len(y)
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
    for j in widx:
        v = np.full(T, np.nan); v[1:] = data[:-1, j]
        feats.append(v); names.append(f"{channels[j]}_lag1")
    X = np.column_stack(feats)
    if lag_fix:
        X = np.vstack([X[1:], np.full((1, X.shape[1]), np.nan)])
    return X, names


def run_one(ds, H, regime, seed, a, out_dir):
    meta, data, cal = ds["meta"], ds["data"], ds["cal"]
    name, L = ds["name"], a.L
    tgt, ch = meta["target_idx"], meta["channels"]
    y = data[:, tgt]
    sd_t = float(ds["sd"][tgt])
    widx = list(meta.get("weather_idx", [])) if regime == "R2" else []
    suffix = f"_v2-{regime}" + ("" if a.exo_mode == "noisy" else f"-{a.exo_mode}") + \
             ("" if a.lag_fix else "-v1lag")
    tag = f"{name}_GBRT_L{L}_H{H}_s{seed}{suffix}"
    if done(out_dir, tag) and not a.force:
        print(f"[{tag}] exists, skipped", flush=True); return

    Xs, names_s = static_features(y, data, widx, ch, L, a.lag_fix)
    anc_tr = starts(meta, "train", L, H, a.stride) + L - 1                # last observed step
    anc_va = starts(meta, "val", L, H) + L - 1
    anc_te = starts(meta, "test", L, H) + L - 1
    ok = ~np.isnan(Xs).any(1)
    anc_tr = anc_tr[ok[anc_tr]]
    assert ok[anc_va].all() and ok[anc_te].all()
    Xs_tr, Xs_va, Xs_te = Xs[anc_tr], Xs[anc_va], Xs[anc_te]
    names = list(names_s)
    if regime in ("R1", "R2"):
        names += [c + "_tgt" for c in CAL_COLS]
    W, noise_sd, nwp_sd = None, None, None
    if widx:
        W = data[:, widx]
        # physical-unit NWP error sd at 24 h lead: the shared v2 rule (absolute for
        # temp/rhum/rad/wspd, capacity-relative for wind/solar generation), computed on the
        # raw training-split exogenous block exactly as data_v2.WindowSetV2 does
        from data_v2 import nwp_noise_phys
        tr0, tr1 = meta["splits"]["train"]
        wnames = [ch[j] for j in widx]
        noise_sd = np.array(nwp_noise_phys(wnames, data[tr0:tr1][:, widx]), dtype=np.float64)
        nwp_sd = {n: round(float(s), 4) for n, s in zip(wnames, noise_sd)}
        names += [f"{ch[j]}_tgt_{a.exo_mode}" for j in widx]
    E = (8 if regime != "R0" else 0) + len(widx)
    gen = np.random.default_rng([seed + 7, H])

    truth = np.stack([y[anc_te + 1 + h] for h in range(H)], 1)
    truth_va = np.stack([y[anc_va + 1 + h] for h in range(H)], 1)
    preds, preds_va = np.zeros_like(truth), np.zeros_like(truth_va)
    print(f"[{tag}] regime={regime} feats={len(names)} train={len(anc_tr)} val={len(anc_va)} "
          f"test={len(anc_te)} iters={a.iters}", flush=True)
    t_fit, t_inf, t0 = 0.0, 0.0, time.time()
    for h in range(H):
        blocks = []
        for anc, base in ((anc_tr, Xs_tr), (anc_va, Xs_va), (anc_te, Xs_te)):
            tau = anc + 1 + h
            parts = [base]
            if regime in ("R1", "R2"):
                parts.append(cal[tau])
            if W is not None:
                if a.exo_mode == "persist":                           # yesterday, same hour
                    w = W[anc - 23 + (h % 24)]
                else:
                    w = W[tau]
                    if a.exo_mode == "noisy":
                        s_h = noise_sd * np.sqrt((h + 1) / 24.0)
                        w = w + gen.normal(size=w.shape) * s_h
                parts.append(w)
            blocks.append(np.hstack(parts))
        X_tr, X_va, X_te = blocks
        m = HistGradientBoostingRegressor(max_iter=a.iters, random_state=seed, **HP)
        t1 = time.time(); m.fit(X_tr, y[anc_tr + 1 + h]); t_fit += time.time() - t1
        t1 = time.time(); preds[:, h] = m.predict(X_te); t_inf += time.time() - t1
        preds_va[:, h] = m.predict(X_va)
        if (h + 1) % 24 == 0 or h + 1 == H:
            print(f"   step {h+1}/{H} ({time.time()-t0:.0f}s)", flush=True)

    met, arrs = metrics(preds, truth, sd_t)
    val_mse = float((((preds_va - truth_va) / sd_t) ** 2).mean())
    rec = record(name, "GBRT", L, H, seed, regime, a.exo_mode, met, t_fit, t_inf, suffix,
                 E=E, params=0, train_stride=a.stride, val_mse=val_mse, history=[],
                 lr=HP["learning_rate"], iters=a.iters, feature_names=names,
                 lag_fix=bool(a.lag_fix), n_train=int(len(anc_tr)),
                 nwp_rule="v2-capacity-relative" if widx else None, nwp_sd_24h=nwp_sd)
    write_result(out_dir, tag, rec, arrs)
    print(f"[{tag}] MSE={met['mse']:.4f} MAE={met['mae']:.4f} WAPE={met['wape']:.2f}% "
          f"val={val_mse:.4f} ({t_fit:.0f}s fit)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(GRIDS))
    ap.add_argument("--horizons", default="24,96,336")
    ap.add_argument("--regimes", default="R0,R1,R2")
    ap.add_argument("--seeds", default="2021,2022,2023")
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--exo_mode", default="noisy", choices=["noisy", "perfect", "persist"])
    ap.add_argument("--lag_fix", type=int, default=1,
                    help="1 (default): features at the anchor use everything up to and "
                         "including the anchor hour; 0: the v1 alignment (up to anchor-1)")
    ap.add_argument("--force", type=int, default=0)
    ap.add_argument("--out", default=RES_DEFAULT)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    a = ap.parse_args()
    i, n = (int(v) for v in a.shard.split("/"))
    seeds = parse_list(a.seeds, int)
    jobs = [(H, d, rg, s) for H in parse_list(a.horizons, int) for d in parse_list(a.datasets)
            for rg in parse_list(a.regimes) for s in seeds_for(rg, seeds)]
    jobs = jobs[i::n]
    os.makedirs(os.path.join(a.out, "errors"), exist_ok=True)
    print(f"{len(jobs)} jobs in shard {a.shard}", flush=True)
    cache, T0 = {}, time.time()
    for k, (H, d, rg, s) in enumerate(jobs):
        if d not in cache:
            cache = {d: load_v2(d)}
        run_one(cache[d], H, rg, s, a, a.out)
        print(f"-- job {k+1}/{len(jobs)} done, elapsed {(time.time()-T0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
