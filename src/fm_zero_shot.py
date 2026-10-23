"""Zero-shot time-series foundation-model baselines for the v2 harness (inference only).

Models
------
* ``Chronos2``  -- ``amazon/chronos-2`` through ``chronos.Chronos2Pipeline`` (chronos-forecasting 2.x).
  Point forecast = the 0.5 quantile of the predictive quantiles. Regime R2 passes the spec's
  calendar block and the weather block as *known-future* covariates through the pipeline's
  list-of-dicts interface (``target`` / ``past_covariates`` / ``future_covariates``): past rows of
  every covariate are the true (z-scored) values, future rows of the weather channels are degraded
  (``exo_mode``, default ``noisy``, seeded) with the v2 capacity-relative NWP rule
  (``data_v2.nwp_noise_phys``; a fallback copy lives here): physical sd at 24 h of temp 1.2 K,
  rhum 6, rad 40, wspd 1.1 m/s (= 3.96 in the km/h column), wind 0.08 x and solar 0.05 x the 99.9th percentile of the training
  split, grown with sqrt(lead/24). The JSON records ``nwp_rule = "v2-capacity-relative"``.
* ``TimesFM2``  -- ``google/timesfm-2.0-500m-pytorch`` through ``timesfm`` 1.3.0 (PyTorch backend),
  ``forecast`` for R0 (median point forecast). For R2 the 1.3.0 package exposes
  ``forecast_with_covariates`` but its ``xreg_lib`` needs JAX, which is not installed in this shared
  environment (and must not be installed here), so the same in-context linear XReg
  (``xreg_mode='xreg + timesfm'``, ridge 0, per-input target normalisation, one OLS pooled over the
  inference batch, TimesFM run on the residual) is ported line-for-line to numpy in
  ``TimesFMRunner.timesfm_xreg``; the numerics match the JAX implementation up to float precision.

Protocol
--------
Same test windows as ``data.WindowSet`` (starts = arange(max(0, te0-L), te1-L-H+1), stride 1,
ascending), target = the ``load`` channel (last), MS setting. The models see the raw physical
series; the metrics are computed exactly as ``exp.evaluate`` does, both in physical units (WAPE,
MAE, RMSE, MAPE, peak/ramp metrics) and in the v1 z-scored units using the training-split mean/std
of the target channel. L = 336 is the shared-lookback protocol row; L = 1024 is reported as a
secondary row *outside* the shared-lookback protocol (foundation models benefit from long context).

Outputs (spec V2_SPEC.md): ``results_v2/fm/<tag>.json`` and ``results_v2/fm/errors/<tag>.npz`` with
``tag = {data}_{model}_L{L}_H{H}_s0_v2-{regime}-fm``. ``--dry`` writes 20-window checks to
``results_v2/fm/dry/`` instead.

Training-corpus contamination caveat (reported in the paper)
------------------------------------------------------------
Both models are pre-trained on large public time-series corpora that include public electricity
datasets (ECL/Electricity, ETT, GluonTS/LOTSA-style collections) and possibly national-grid load
series. The 2015-2020 ENTSO-E period used here may therefore overlap the models' pre-training
data, so these zero-shot numbers are an optimistic reference, not a clean out-of-sample baseline.

Usage
-----
    python src/fm_zero_shot.py --datasets OPSD-DE --horizons 24 --models Chronos2 --regimes R0 --L 336 --dry
    python src/fm_zero_shot.py --datasets all
"""
import argparse, json, math, os, sys, time, warnings
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from data import calendar_features, SEC_PER_HOUR          # noqa: E402  (v1 helpers, read-only)

OUT = os.path.join(HERE, "results_v2", "fm")
NWP_RULE = "v2-capacity-relative"
# v2 NWP-error rule (shared: data_v2.nwp_noise_phys). Physical-unit sd at 24 h lead per exogenous
# channel; generation channels are relative to the system's size (the v1 absolute-MW constants
# were calibrated on DE and injected 2-8 z-units of noise on BE/NL wind and solar).
NWP_PHYS_24H = {"temp": 1.2, "rhum": 6.0, "rad": 40.0, "wspd": 1.1 * 3.6}   # wspd column is km/h
NWP_CAP_FRAC = {"wind": 0.08, "solar": 0.05}             # x the 99.9th percentile of the train split


def nwp_noise_phys(names, raw_train):
    """Fallback copy of data_v2.nwp_noise_phys: (E,) physical sd at 24 h for channels `names`,
    raw_train (T_train, E) in the same column order. Unknown channels: 10% of the training sd."""
    out = []
    for j, n in enumerate(names):
        col = np.asarray(raw_train[:, j], dtype=np.float64)
        if n in NWP_PHYS_24H:
            out.append(NWP_PHYS_24H[n])
        elif n in NWP_CAP_FRAC:
            out.append(NWP_CAP_FRAC[n] * float(np.nanpercentile(col, 99.9)))
        else:
            out.append(0.10 * float(np.nanstd(col)))
    return np.asarray(out, dtype=np.float64)


def nwp_rule(names, raw_train):
    """Use the shared rule from src/data_v2.py when present, else the local copy."""
    try:
        from data_v2 import nwp_noise_phys as shared
        return np.asarray(shared(names, raw_train), dtype=np.float64), "data_v2.nwp_noise_phys"
    except Exception:
        return nwp_noise_phys(names, raw_train), "fm_zero_shot.nwp_noise_phys (fallback copy)"


def degrade_weather(fut_z, past_z, s_z, mode, gen):
    """Future weather rows -> what a forecaster would have (V2_SPEC R2), z-scored blocks.
    fut_z (B,H,E) true future, past_z (B,L,E) true past, s_z (E,) noise sd in z-units at 24 h."""
    if mode == "perfect":
        return fut_z
    B, H, E = fut_z.shape
    if mode == "persist":                                            # yesterday, repeated
        src = np.arange(H) % 24 + (past_z.shape[1] - 24)
        return past_z[:, src]
    lead = torch.arange(1, H + 1, dtype=torch.float32)
    scale = torch.sqrt(lead / 24.0).view(1, H, 1)
    noise = torch.randn((B, H, E), dtype=torch.float32, generator=gen)
    s = torch.as_tensor(s_z, dtype=torch.float32).view(1, 1, E)
    return (torch.from_numpy(np.ascontiguousarray(fut_z)) + noise * s * scale).numpy()
CHECKPOINT = {"Chronos2": "amazon/chronos-2", "TimesFM2": "google/timesfm-2.0-500m-pytorch"}
CAL_NAMES = ["sin_hod", "cos_hod", "sin_dow", "cos_dow", "is_weekend", "holiday", "bridge", "xmas"]
CONTAMINATION = ("Zero-shot: pre-training corpora of Chronos-2 and TimesFM-2.0 include public "
                 "electricity datasets and may overlap the 2015-2020 ENTSO-E period; treat as an "
                 "optimistic reference, not a clean out-of-sample baseline.")
H_MAX = 336


# --------------------------------------------------------------------------- data
def proc_dir(prefer_v2=True):
    v2 = os.path.join(HERE, "data", "proc_v2")
    if prefer_v2 and os.path.isdir(v2) and list_grids(v2):
        return v2
    return os.path.join(HERE, "data", "proc")


def list_grids(proc):
    return sorted(f[:-5] for f in os.listdir(proc)
                  if f.endswith(".json") and f.startswith("OPSD-")
                  and os.path.exists(os.path.join(proc, f[:-5] + ".npz")))


class Series:
    """One processed dataset: raw channels, train-statistics z-scoring, calendar block."""

    def __init__(self, name, proc):
        z = np.load(os.path.join(proc, name + ".npz"))
        self.meta = meta = json.load(open(os.path.join(proc, name + ".json")))
        self.name, self.proc = name, proc
        data = z["data"].astype(np.float32)
        self.raw = data
        self.T, self.C = data.shape
        tr0, tr1 = meta["splits"]["train"]
        mu = data[tr0:tr1].mean(0)
        sd = data[tr0:tr1].std(0)
        sd[sd < 1e-6] = 1.0
        self.mu, self.sd = mu, sd
        self.z = (data - mu) / sd                                   # v1 WindowSet.data
        self.ti = int(meta["target_idx"]) % self.C
        self.te0, self.te1 = meta["splits"]["test"]
        self.spd = {"1h": 24, "15min": 96}.get(meta.get("freq", "1h"), 24)
        widx = meta.get("weather_idx")
        self.weather_idx = list(widx) if widx else [i for i in range(self.C) if i != self.ti]
        self.weather_names = [meta["channels"][i] for i in self.weather_idx]
        # NWP-error sd per weather channel: physical units at 24 h lead (v2 rule) and z-units
        self.nwp_sd_phys, self.nwp_source = nwp_rule(self.weather_names,
                                                     data[tr0:tr1][:, self.weather_idx])
        self.nwp_sd_z = self.nwp_sd_phys / np.maximum(sd[self.weather_idx], 1e-6)
        # calendar block from local civil time when the v2 build provides it, else UTC (v1 data)
        if "local_stamps" in z.files:
            st = z["local_stamps"].astype(np.float64) / SEC_PER_HOUR
            self.basis_time = "local"
        else:
            st = z["stamps"].astype(np.float64) / SEC_PER_HOUR
            self.basis_time = "utc"
        hod, dow = calendar_features(st)
        if "cal_flags" in z.files:
            flags = z["cal_flags"].astype(np.float32)
            self.cal_flags_available = True
        else:
            flags = np.zeros((self.T, 3), np.float32)
            self.cal_flags_available = False
        cal = np.stack([np.sin(2 * np.pi * hod / 24.0), np.cos(2 * np.pi * hod / 24.0),
                        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
                        (dow >= 5.0).astype(np.float64)], -1).astype(np.float32)
        self.cal = np.concatenate([cal, flags], -1)                  # (T, 8)

    def test_starts(self, L, H):
        lo = max(0, self.te0 - L)
        starts = np.arange(lo, self.te1 - L - H + 1, dtype=np.int64)
        if len(starts) == 0:
            raise ValueError(f"{self.name}: no test windows for L={L} H={H}")
        return starts

    def target_windows(self, starts, L, H):
        idx = starts[:, None] + np.arange(L + H)[None, :]
        w = self.raw[idx, self.ti]                                  # (B, L+H) physical
        return w[:, :L], w[:, L:]

    def covariate_block(self, starts, L, H, exo_mode, gen):
        """R2 block: calendar (8) + weather (E). Past rows true, future weather degraded with
        the v2 capacity-relative NWP rule (noise drawn in z-units, seeded `gen`)."""
        idx = starts[:, None] + np.arange(L + H)[None, :]
        cal = self.cal[idx]                                          # (B, L+H, 8)
        wz = self.z[idx][:, :, self.weather_idx]                     # (B, L+H, E) z-scored
        past_w = wz[:, :L]
        fut = degrade_weather(wz[:, L:], past_w, self.nwp_sd_z, exo_mode, gen)
        names = CAL_NAMES + self.weather_names
        past = np.concatenate([cal[:, :L], past_w], -1).astype(np.float32)
        futr = np.concatenate([cal[:, L:], fut], -1).astype(np.float32)
        return names, past, futr


# --------------------------------------------------------------------------- models
class ChronosRunner:
    name = "Chronos2"

    def __init__(self, dtype=torch.float16):
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(CHECKPOINT[self.name], device_map="cuda",
                                                     dtype=dtype)
        self.dtype = str(dtype).replace("torch.", "")
        self.n_params = int(sum(p.numel() for p in self.pipe.model.parameters()))
        assert H_MAX <= self.pipe.model_prediction_length, "would need long-horizon unrolling"

    def predict(self, hist, H, L, cov=None):
        """hist (B, L) physical; cov = (names, past (B,L,K), fut (B,H,K)) or None -> (B, H)."""
        B = hist.shape[0]
        if cov is None:
            inputs = hist[:, None, :].astype(np.float32)             # (B, 1, L)
            bs = B
        else:
            names, past, fut = cov
            inputs = [dict(target=hist[i].astype(np.float32),
                           past_covariates={n: past[i, :, k] for k, n in enumerate(names)},
                           future_covariates={n: fut[i, :, k] for k, n in enumerate(names)})
                      for i in range(B)]
            bs = B * (1 + len(names))
        _, med = self.pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=[0.5],
                                             batch_size=bs, context_length=L)
        return torch.stack(med)[:, 0, :].float().numpy()


class TimesFMRunner:
    name = "TimesFM2"

    def __init__(self, amp=True, full_ctx=False, xreg_mode="xreg + timesfm", ridge=0.0):
        import timesfm
        assert xreg_mode in ("xreg + timesfm", "timesfm + xreg")
        self.xreg_mode, self.ridge = xreg_mode, float(ridge)
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=64, horizon_len=128,
                                           num_layers=50, use_positional_embedding=False,
                                           context_len=2048, point_forecast_mode="median"),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=CHECKPOINT[self.name]))
        self.amp = amp
        self.full_ctx = full_ctx
        self.context_len = 2048
        self.dtype = "float16-autocast" if amp else "float32"
        self.n_params = int(sum(p.numel() for p in self.tfm._model.parameters()))

    def _set_context(self, L, H):
        """TimesFM front-pads every input to `context_len` (2048 for the 2.0 checkpoint) and the
        padded patches are masked, so the smallest multiple of the 32-step patch that holds the
        lookback plus the auto-regressively decoded history gives the same forecast (checked:
        WAPE identical to 3 decimals, max |diff| ~1e-3 of the series scale under fp16) at a
        fraction of the cost. --tfm_full_ctx keeps the checkpoint's 2048."""
        nd = math.ceil(H / self.tfm.output_patch_len)
        ctx = 2048 if self.full_ctx else min(2048, math.ceil((L + (nd - 1) * self.tfm.output_patch_len) / 32) * 32)
        self.tfm.context_len = ctx
        self.tfm._horizon_start = ctx - self.tfm.input_patch_len
        self.context_len = ctx

    def _forecast(self, inputs, L, H, on_context=False):
        self._set_context(L, H)
        self.tfm.horizon_len = H
        self.tfm.global_batch_size = self.tfm.per_core_batch_size = len(inputs)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            mean, _ = self.tfm.forecast(inputs, freq=[0] * len(inputs), forecast_context_len=L,
                                        return_forecast_on_context=on_context)
        return np.asarray(mean, dtype=np.float64)

    def predict(self, hist, H, L, cov=None):
        inputs = [hist[i].astype(np.float32) for i in range(hist.shape[0])]
        if cov is None:
            return self._forecast(inputs, L, H)[:, :H]
        return self.timesfm_xreg(inputs, cov, L, H)

    @staticmethod
    def _ols(x_train, x_test, y, ridge):
        """xreg_lib.BatchedInContextXRegLinear.fit for numerical covariates: standardise the
        pooled context rows, prepend an intercept, ridge/OLS via pinv; returns (fit on context,
        fit on horizon, beta)."""
        x_mean = x_train.mean(0, keepdims=True)
        w = x_train.std(0, keepdims=True)
        x_std = np.where(w > 1e-6, w, 1.0)
        x_train = np.pad((x_train - x_mean) / x_std, ((0, 0), (1, 0)), constant_values=1.0)
        x_test = np.pad((x_test - x_mean) / x_std, ((0, 0), (1, 0)), constant_values=1.0)
        beta = np.linalg.pinv(x_train.T @ x_train + ridge * np.eye(x_train.shape[1]),
                              hermitian=True) @ x_train.T @ y
        return x_train @ beta, x_test @ beta, beta

    def timesfm_xreg(self, inputs, cov, L, H):
        """numpy port of timesfm_base.forecast_with_covariates(normalize_xreg_target_per_input=True)
        with the library's two modes. The spec block is all-numerical (the 0/1 flags equal a
        one-hot with drop='first'), so only the numerical branch of xreg_lib is needed.
        'xreg + timesfm' (library default): one OLS pooled over the batch on the (globally
        standardised) covariates with intercept, TimesFM forecasts the in-context residual, the
        linear part is added back on the horizon. 'timesfm + xreg': TimesFM forecasts first, the
        OLS is fitted on its in-context residual (context minus the first patch) and extrapolated."""
        from timesfm.timesfm_base import _normalize, _renormalize
        names, past, fut = cov
        B, K = len(inputs), len(names)
        hs = self.tfm._horizon_start
        x_test = fut.reshape(B * H, K).astype(np.float64)
        if self.xreg_mode == "xreg + timesfm":
            targets, stats = _normalize([np.asarray(x, dtype=np.float64) for x in inputs])
            fit_ctx, xreg, _ = self._ols(past.reshape(B * L, K).astype(np.float64), x_test,
                                         np.concatenate(targets), self.ridge)
            resid = [t - c for t, c in zip(targets, fit_ctx.reshape(B, L))]
            mean = self._forecast(resid, L, H, on_context=True)
            hs = self.tfm._horizon_start
            out = mean[:, hs:hs + H] + xreg.reshape(B, H)
            return np.stack(_renormalize(list(out), stats))
        tl = L - self.tfm.input_patch_len                    # no forecast on the first patch
        mean = self._forecast(inputs, L, H, on_context=True)
        hs = self.tfm._horizon_start
        resid = [np.asarray(x, dtype=np.float64)[-tl:] - m[hs - tl:hs] for x, m in zip(inputs, mean)]
        targets, stats = _normalize(resid)
        _, xreg, _ = self._ols(past[:, -tl:, :].reshape(B * tl, K).astype(np.float64), x_test,
                               np.concatenate(targets), self.ridge)
        xreg = np.stack(_renormalize(list(xreg.reshape(B, H)), stats))
        return mean[:, hs:hs + H] + xreg


# --------------------------------------------------------------------------- metrics
def score(pred, true, mu_t, sd_t, spd=24):
    """Reproduces exp.evaluate() for the MS setting from physical-unit point forecasts."""
    pt = pred.astype(np.float64); yt = true.astype(np.float64)
    p = (pt - mu_t) / sd_t; y = (yt - mu_t) / sd_t                 # z-scored (v1 units)
    e = p - y
    n, H = e.shape
    per_window = (e ** 2).mean(1)
    out = dict(mse=float((e ** 2).mean()), mae=float(np.abs(e).mean()), n_windows=int(n),
               per_channel_mse=[round(float((e ** 2).mean()), 6)])
    arrs = dict(per_window=per_window.astype(np.float32),
                per_step_mse=(e ** 2).mean(0).astype(np.float32),
                per_step_mae=np.abs(e).mean(0).astype(np.float32))
    d = pt - yt
    out.update(rmse_orig=float(np.sqrt((d ** 2).mean())), mae_orig=float(np.abs(d).mean()),
               wape=100.0 * float(np.abs(d).sum()) / max(float(np.abs(yt).sum()), 1e-9),
               mape=100.0 * float((np.abs(d) / np.maximum(np.abs(yt), 1e-6)).mean()))
    dr = (pt[:, 1:] - pt[:, :-1]) - (yt[:, 1:] - yt[:, :-1])
    out["ramp_mae"] = float(np.abs(dr).mean()) if dr.size else 0.0
    if H >= spd:
        nd = H // spd
        yd = yt[:, :nd * spd].reshape(-1, nd, spd); pd_ = pt[:, :nd * spd].reshape(-1, nd, spd)
        yp, ypi = yd.max(-1), yd.argmax(-1); pp, ppi = pd_.max(-1), pd_.argmax(-1)
        out.update(peak_mae=float(np.abs(pp - yp).mean()),
                   peak_mape=100.0 * float((np.abs(pp - yp) / np.maximum(np.abs(yp), 1e-6)).mean()),
                   peak_time_err=float(np.abs(ppi - ypi).mean()))
    arrs["per_window_ae"] = np.abs(d).mean(1).astype(np.float32)
    return out, arrs


def fill_nonfinite(pred):
    """Linear interpolation over non-finite steps inside a window (guard; reported when used)."""
    pred = np.where(np.isfinite(pred), pred, np.nan)
    for r in np.where(np.isnan(pred).any(1))[0]:
        v = pred[r]; m = np.isnan(v)
        v[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), v[~m]) if (~m).any() else 0.0
    return pred


# --------------------------------------------------------------------------- driver
def run_one(runner, ds, L, H, regime, args, out_dir):
    tag = f"{ds.name}_{runner.name}_L{L}_H{H}_s{args.seed}_v2-{regime}-fm{args.suffix}"
    jpath = os.path.join(out_dir, tag + ".json")
    if os.path.exists(jpath) and not args.force:
        print(f"[{tag}] exists, skip", flush=True); return
    starts = ds.test_starts(L, H)
    if args.dry:
        starts = starts[:args.dry_windows]
    gen = torch.Generator().manual_seed(args.seed * 1_000_003 + L * 1009 + H * 13)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    # Chronos-2 with covariates puts (1+K) series per window into one group-attention batch;
    # its throughput per window is best at ~8 windows (120 series) per forward and collapses
    # above ~64 (and beyond the 8 GB card WDDM pages to host memory instead of raising OOM).
    wb = min(args.batch, args.batch_cov) if (regime == "R2" and runner.name == "Chronos2") else args.batch
    preds = []
    t0 = time.time()
    for c0 in range(0, len(starts), args.cov_chunk):                 # noise draw independent of wb
        cs = starts[c0:c0 + args.cov_chunk]
        hist, _ = ds.target_windows(cs, L, H)
        cov = ds.covariate_block(cs, L, H, args.exo_mode, gen) if regime == "R2" else None
        i = 0
        while i < len(cs):
            sl = slice(i, i + wb)
            sub = None if cov is None else (cov[0], cov[1][sl], cov[2][sl])
            try:
                preds.append(runner.predict(hist[sl], H, L, sub))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if wb == 1:
                    raise
                wb = max(1, wb // 2)
                print(f"   OOM -> batch {wb}", flush=True)
                continue
            i += wb
    torch.cuda.synchronize()
    infer_time = time.time() - t0
    pred = np.concatenate(preds, 0)
    _, true = ds.target_windows(starts, L, H)
    assert pred.shape == true.shape, (pred.shape, true.shape)
    n_bad = int((~np.isfinite(pred)).any(1).sum())
    if n_bad:
        print(f"   !! {n_bad} windows with non-finite forecasts -> interpolated", flush=True)
        pred = fill_nonfinite(pred)
    r, arrs = score(pred, true, float(ds.mu[ds.ti]), float(ds.sd[ds.ti]), ds.spd)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    cov_names = (CAL_NAMES + ds.weather_names) if regime == "R2" else []
    out = dict(dataset=ds.name, model=runner.name, L=L, H=H, seed=args.seed, lr=None, batch=wb,
               params=runner.n_params, epochs_run=0, n_year_slow=None, train_stride=None,
               val_mse=None, setting="MS", exo_mode=args.exo_mode if regime == "R2" else None,
               bad_steps=n_bad, train_time_s=0.0, infer_time_s=round(infer_time, 2),
               amp=runner.dtype != "float32", tag=f"_v2-{regime}-fm{args.suffix}", history=[],
               **r,
               regime=regime, size="fm", basis_time=ds.basis_time, backbone=CHECKPOINT[runner.name],
               l_selected=False, protocol_lookback=(L == 336), zero_shot=True, dtype=runner.dtype,
               gpu_peak_mem_gb=round(peak_gb, 3), batch_windows=wb, covariates=cov_names,
               context_len=getattr(runner, "context_len", L),
               nwp_rule=NWP_RULE if regime == "R2" else None,
               nwp_rule_source=ds.nwp_source if regime == "R2" else None,
               nwp_sd_phys=({n: round(float(v), 4) for n, v in zip(ds.weather_names, ds.nwp_sd_phys)}
                            if regime == "R2" else None),
               xreg_mode=(runner.xreg_mode if (regime == "R2" and runner.name == "TimesFM2") else None),
               xreg_ridge=(runner.ridge if (regime == "R2" and runner.name == "TimesFM2") else None),
               cal_flags_available=ds.cal_flags_available, proc_dir=os.path.basename(ds.proc),
               dry=bool(args.dry), contamination_caveat=CONTAMINATION)
    os.makedirs(os.path.join(out_dir, "errors"), exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "errors", tag + ".npz"), **arrs)
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] n={r['n_windows']} WAPE={r['wape']:.2f}% MAE={r['mae_orig']:.0f} "
          f"mse_z={r['mse']:.4f} peak_mape={r.get('peak_mape', float('nan')):.2f}% "
          f"t={infer_time:.0f}s mem={peak_gb:.2f}GB batch={wb}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="all", help="comma list or 'all' (every OPSD-* in the proc dir)")
    ap.add_argument("--horizons", default="24,96,336")
    ap.add_argument("--models", default="Chronos2,TimesFM2")
    ap.add_argument("--regimes", default="R0,R2")
    ap.add_argument("--L", default="336,1024")
    ap.add_argument("--exo_mode", default="noisy", choices=["perfect", "noisy", "persist"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64, help="windows per forward pass (halved on OOM)")
    ap.add_argument("--batch_cov", type=int, default=8, help="windows per forward for Chronos-2 with covariates")
    ap.add_argument("--cov_chunk", type=int, default=512, help="windows per covariate/noise chunk")
    ap.add_argument("--dry", action="store_true", help="20 test windows only, written to results_v2/fm/dry/")
    ap.add_argument("--dry_windows", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="recompute even if the JSON exists")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 (Chronos weights / TimesFM autocast)")
    ap.add_argument("--tfm_full_ctx", action="store_true", help="TimesFM: keep the 2048-step padded context")
    ap.add_argument("--xreg_mode", default="xreg + timesfm", choices=["xreg + timesfm", "timesfm + xreg"],
                    help="TimesFM R2: forecast_with_covariates xreg_mode (library default 'xreg + timesfm')")
    ap.add_argument("--xreg_ridge", type=float, default=0.0, help="TimesFM R2: ridge penalty (library default 0)")
    ap.add_argument("--suffix", default="", help="extra tag suffix for variant runs, e.g. _tx")
    ap.add_argument("--proc", default="", help="override the processed-data directory")
    args = ap.parse_args()
    global H_MAX
    warnings.filterwarnings("ignore")
    proc = args.proc or proc_dir()
    names = list_grids(proc) if args.datasets == "all" else args.datasets.split(",")
    Ls = [int(v) for v in args.L.split(",")]
    Hs = [int(v) for v in args.horizons.split(",")]
    H_MAX = max(Hs)
    regimes = args.regimes.split(",")
    out_dir = os.path.join(OUT, "dry") if args.dry else OUT
    os.makedirs(out_dir, exist_ok=True)
    print(f"proc={proc} datasets={names} L={Ls} H={Hs} regimes={regimes} models={args.models} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)
    series = {n: Series(n, proc) for n in names}
    t_all = time.time()
    for mname in args.models.split(","):
        runner = (ChronosRunner(torch.float32 if args.fp32 else torch.float16) if mname == "Chronos2"
                  else TimesFMRunner(amp=not args.fp32, full_ctx=args.tfm_full_ctx,
                                     xreg_mode=args.xreg_mode, ridge=args.xreg_ridge))
        print(f"== {mname} ({runner.n_params/1e6:.1f}M params, {runner.dtype})", flush=True)
        for n in names:
            for L in Ls:
                for H in Hs:
                    for reg in regimes:
                        run_one(runner, series[n], L, H, reg, args, out_dir)
        del runner; torch.cuda.empty_cache()
    print(f"total wall {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
