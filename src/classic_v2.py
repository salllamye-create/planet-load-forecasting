"""Shared helpers for the classical v2 baselines (gbrt_v2.py, tao_vanilla.py).

Corpus loading (data/proc_v2/), window starts, metrics and the result record are written to
match data_v2.WindowSetV2 and exp_v2.evaluate()/run() exactly, so every number is comparable
with the neural v2 runs: z-scoring by the float32 training statistics of data_v2, test starts
= arange(max(0, te0-L), te1-L-H+1) at stride 1, train starts at stride 5, WAPE / MAE / RMSE in
physical units, peak metrics per 24-step forecast day, ramp MAE, and the per-window arrays of
results_v2/errors/<tag>.npz.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC2 = os.path.join(HERE, "data", "proc_v2")
RES_DEFAULT = os.path.join(HERE, "results_v2")

GRIDS = ["OPSD-DE", "OPSD-FR", "OPSD-GB", "OPSD-ES", "OPSD-IT", "OPSD-NL", "OPSD-PL", "OPSD-BE"]
CAL_COLS = ["sin_hod", "cos_hod", "sin_dow", "cos_dow", "weekend", "holiday", "bridge", "xmas"]
REGIMES = ("R0", "R1", "R2")

try:                                    # single source of truth (data_v2 imports torch)
    from data_v2 import NWP_ABS_24H as NWP_RMSE_24H
except Exception:                       # CPU worker without torch: the identical table
    NWP_RMSE_24H = {"temp": 1.2, "rhum": 6.0, "rad": 40.0, "wspd": 1.1 * 3.6}   # wspd column km/h


def parse_list(s, cast=str):
    return [cast(v.strip()) for v in str(s).split(",") if v.strip()]


def seeds_for(regime, seeds):
    """R2 runs every seed (NWP noise, model randomness); R0/R1 the first one (run_all_v2 rule)."""
    return list(seeds) if regime == "R2" else list(seeds[:1])


def load_v2(name):
    z = np.load(os.path.join(PROC2, name + ".npz"))
    meta = json.load(open(os.path.join(PROC2, name + ".json")))
    data32 = z["data"].astype(np.float32)
    tr0, tr1 = meta["splits"]["train"]
    mu = data32[tr0:tr1].mean(0); sd = data32[tr0:tr1].std(0); sd[sd < 1e-6] = 1.0   # data_v2 rule
    stamps = z["stamps"].astype(np.int64)
    local = z["local_stamps"].astype(np.int64)
    local_h = local.astype(np.float64) / 3600.0
    hod = np.mod(local_h, 24.0)
    dow = np.mod(np.floor(local_h / 24.0) + 4.0, 7.0)                # 0 = Monday (data.py)
    flags = z["cal_flags"].astype(np.float64)                        # holiday, bridge, xmas
    cal = np.column_stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                           np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                           (dow >= 5).astype(np.float64), flags])   # the 8-column R1 block
    return dict(name=name, meta=meta, data=data32.astype(np.float64),
                mu=mu.astype(np.float64), sd=sd.astype(np.float64),
                stamps=stamps, local=local, local_h=local_h, hod=hod, dow=dow, flags=flags, cal=cal)


def starts(meta, flag, L, H, stride=1):
    """Window start indices of data_v2.WindowSetV2 (first lookback position)."""
    s0, s1 = meta["splits"][flag]
    st = np.arange(max(0, s0 - L), s1 - L - H + 1, stride, dtype=np.int64)
    if len(st) == 0:
        raise ValueError(f"{meta['name']}/{flag}: no windows for L={L} H={H}")
    return st


def metrics(preds, truth, sd_t, steps_per_day=24):
    """exp_v2.evaluate() on physical-unit (n_windows, H) arrays; sd_t = training sd of the target."""
    e = preds - truth
    zn = e / sd_t
    out = dict(mse=float((zn ** 2).mean()), mae=float(np.abs(zn).mean()), n_windows=int(len(truth)),
               per_channel_mse=[round(float((zn ** 2).mean()), 6)],
               rmse_orig=float(np.sqrt((e ** 2).mean())), mae_orig=float(np.abs(e).mean()),
               wape=100.0 * float(np.abs(e).sum() / max(np.abs(truth).sum(), 1e-9)),
               mape=100.0 * float((np.abs(e) / np.maximum(np.abs(truth), 1e-6)).mean()),
               ramp_mae=float(np.abs(np.diff(preds, axis=1) - np.diff(truth, axis=1)).mean()))
    H = truth.shape[1]
    if H >= steps_per_day:
        nd = H // steps_per_day
        yd = truth[:, :nd * steps_per_day].reshape(-1, nd, steps_per_day)
        pdd = preds[:, :nd * steps_per_day].reshape(-1, nd, steps_per_day)
        yp, ypi = yd.max(-1), yd.argmax(-1); pp, ppi = pdd.max(-1), pdd.argmax(-1)
        out["peak_mae"] = float(np.abs(pp - yp).mean())
        out["peak_mape"] = 100.0 * float((np.abs(pp - yp) / np.maximum(np.abs(yp), 1e-6)).mean())
        out["peak_time_err"] = float(np.abs(ppi - ypi).mean())
    arrs = dict(per_window=(zn ** 2).mean(1), per_window_ae=np.abs(e).mean(1),
                per_step_mse=(zn ** 2).mean(0), per_step_mae=np.abs(zn).mean(0))
    return out, arrs


def record(data, model, L, H, seed, regime, exo_mode, met, train_time, infer_time, tag_suffix,
           E, params, train_stride, val_mse, history, lr=0.0, **extra):
    """The exp_v2.run() output dict (v1 exp.py fields + the v2 fields of V2_SPEC.md)."""
    rec = dict(dataset=data, model=model, L=L, H=H, seed=seed, lr=lr, batch=0, params=int(params),
               epochs_run=0, n_year_slow=None, train_stride=train_stride, val_mse=val_mse,
               setting="MS", regime=regime, size=1.0, basis_time="local",
               exo_mode=exo_mode if regime == "R2" else None, backbone=None, l_selected=False,
               E=int(E), K=0, train_frac=1.0, train_first=False, eval_data=None, bad_steps=0,
               train_time_s=round(float(train_time), 1), infer_time_s=round(float(infer_time), 2),
               amp=False, tag=tag_suffix, extra={}, history=history)
    rec.update(extra)
    rec.update(met)
    return rec


def write_result(out_dir, tag, rec, arrs):
    err = os.path.join(out_dir, "errors")
    os.makedirs(err, exist_ok=True)
    np.savez_compressed(os.path.join(err, tag + ".npz"), **arrs)
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(rec, f, indent=2)


def done(out_dir, tag):
    return os.path.exists(os.path.join(out_dir, tag + ".json"))
