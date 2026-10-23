"""v2 windowing: the v1 protocol (data.py) plus the covariate block of V2_SPEC.md, a
local-civil-time option for the phase-locked basis, and holiday day-types.

Reads data/proc_v2/<NAME>.npz|json. The window protocol, z-scoring and split rules are the
v1 ones (WindowSet in data.py) so every v2 number is comparable with v1.

Per item the loader yields
    x    (L, C)       lookback, dataset-z-scored
    y    (H, C)       target block, dataset-z-scored
    Phi  (L+H, K)     absolute-time fast basis (PLANET / CalTST)
    Psi  (L+H, Ks)    absolute-time slow basis (kept for the v1 climatological tier)
    tf   (L+H, 5)     Informer/Autoformer calendar embedding inputs (UTC, as v1)
    cov  (L+H, E)     covariate block for the regime (E = 0 -> a zero-size tensor)
    idx  ()           window start index
"""
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data import (calendar_features, slow_basis, slow_dim, time_features, PERIODS,
                  SEC_PER_HOUR)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC2 = os.path.join(HERE, "data", "proc_v2")
TREND_COL = 1
CAL_COLS = ["sin_hod", "cos_hod", "sin_dow", "cos_dow", "weekend", "holiday", "bridge", "xmas"]
REGIMES = ("R0", "R1", "R2")

# --------------------------------------------------------------------------- NWP skill
# Day-ahead forecast error at 24 h lead used to degrade the reanalysis into the information
# a forecaster would really hold (regime R2). Weather variables carry an absolute error that
# transfers across systems; renewable *generation* forecasts do not -- their error scales with
# the installed fleet -- so they are expressed relative to a capacity proxy (the 99.9th
# percentile of the training split), at the normalised RMSE typical of national day-ahead
# wind (~8% of capacity) and PV (~5% of capacity) forecasts. (v1 expressed the generation
# error in megawatts calibrated on Germany, which was 7.5 standard deviations of noise on the
# Belgian wind series; every v2 covariate result uses this rule.)
# Units are those of the corpus columns (Open-Meteo ERA5): temp degC, rhum %, rad W/m2, and
# wind speed in km/h -- the intended 1.1 m/s error is therefore 3.96 km/h. (Before 2026-09-28
# the value 1.1 was applied in km/h, 3.6x too small; every R2 result was regenerated.)
WSPD_MS_24H = 1.1
NWP_ABS_24H = {"temp": 1.2, "rhum": 6.0, "rad": 40.0, "wspd": WSPD_MS_24H * 3.6}
NWP_REL_CAP_24H = {"wind": 0.08, "solar": 0.05}


def nwp_noise_phys(names, raw_train):
    """Physical-unit noise sd at 24 h lead per exogenous channel.

    names     : channel names (e.g. 'temp', 'wind'); raw_train: (T_train, n) physical values
    Unknown names fall back to 10% of the channel's training standard deviation."""
    out = []
    for j, n in enumerate(names):
        col = np.asarray(raw_train[:, j], dtype=np.float64)
        if n in NWP_ABS_24H:
            out.append(NWP_ABS_24H[n])
        elif n in NWP_REL_CAP_24H:
            out.append(NWP_REL_CAP_24H[n] * float(np.nanpercentile(col, 99.9)))
        else:
            out.append(0.10 * float(np.nanstd(col)))
    return out


# --------------------------------------------------------------------------- basis v2
def build_basis_v2(stamps_h, n_day=6, n_week=3, n_year=0, trend=True, masks=(), ref=None,
                   relative=False):
    """Absolute-time harmonic design matrix with generalised day-type interactions.

    stamps_h : (N,) hours since epoch -- UTC or local civil time, the caller decides
    masks    : sequence of (N,) {0,1} day-type indicators (Saturday, Sunday, holiday, ...);
               each adds an intercept column and interactions with the first two daily
               harmonic pairs, exactly the v1 construction but with an open mask list
    """
    s = np.asarray(stamps_h, dtype=np.float64)
    ref = s[..., :1] if ref is None else ref
    s_used = (s - ref) if relative else s
    cols = [np.ones_like(s)]
    if trend:
        cols.append((s - ref) / 168.0)

    def harmonics(period, n):
        out = []
        for k in range(1, n + 1):
            w = 2.0 * np.pi * k / period
            out.append(np.sin(w * s_used)); out.append(np.cos(w * s_used))
        return out

    daily = harmonics(PERIODS["day"], n_day)
    cols += daily
    cols += harmonics(PERIODS["week"], n_week)
    cols += harmonics(PERIODS["year"], n_year)
    for m in masks:
        mm = np.asarray(m, dtype=np.float64)
        cols.append(mm)
        for h in daily[:4]:
            cols.append(mm * h)
    return np.stack(cols, axis=-1).astype(np.float32)


def basis_dim_v2(n_day=6, n_week=3, n_year=0, trend=True, n_masks=2):
    return 1 + (1 if trend else 0) + 2 * (n_day + n_week + n_year) + n_masks * 5


def n_masks_for(daytype):
    """daytype: False | 'weekend' (Sat, Sun) | 'weekend+holiday' (Sat, Sun, holiday)."""
    if not daytype:
        return 0
    return 3 if daytype == "weekend+holiday" else 2


# --------------------------------------------------------------------------- dataset
class WindowSetV2(Dataset):
    def __init__(self, name, flag, L, H, scaler=None, basis_cfg=None, stride=1,
                 regime="R1", basis_time="local", splits=None):
        assert regime in REGIMES, regime
        z = np.load(os.path.join(PROC2, name + ".npz"))
        meta = json.load(open(os.path.join(PROC2, name + ".json")))
        if splits:                                   # row-index override (monthly-refit protocol)
            meta = dict(meta); meta["splits"] = dict(meta["splits"], **{k: list(v) for k, v in splits.items()})
        self.meta, self.name, self.L, self.H = meta, name, L, H
        self.regime, self.basis_time = regime, basis_time
        data = z["data"].astype(np.float32)
        stamps = z["stamps"].astype(np.float64) / SEC_PER_HOUR            # UTC hours
        local = z["local_stamps"].astype(np.float64) / SEC_PER_HOUR      # local civil hours
        flags = z["cal_flags"].astype(np.float32)                         # (T, 3)
        self.aux = z["aux"].astype(np.float32) if "aux" in z.files else None

        tr0, tr1 = meta["splits"]["train"]
        if scaler is None:
            mu = data[tr0:tr1].mean(0); sd = data[tr0:tr1].std(0); sd[sd < 1e-6] = 1.0
            scaler = (mu, sd)
        self.mu, self.sd = scaler
        self.scaler = scaler
        self.raw = data
        self.data = (data - self.mu) / self.sd
        self.stamps, self.local, self.flags = stamps, local, flags

        s0, s1 = meta["splits"][flag]
        lo = max(0, s0 - L)
        self.starts = np.arange(lo, s1 - L - H + 1, stride, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError(f"{name}/{flag}: no windows for L={L} H={H}")

        self.tf_all = time_features(stamps)

        # ---- covariate block ------------------------------------------------------
        hod, dow = calendar_features(local)
        cal = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                        (dow >= 5).astype(np.float64)], -1).astype(np.float32)
        cal = np.concatenate([cal, flags], -1)                            # (T, 8)
        self.weather_idx = list(meta.get("weather_idx", []))
        self.cov_names, self.cov_weather_cols = [], []
        if regime == "R0":
            self.cov_all = np.zeros((len(stamps), 0), np.float32)
        elif regime == "R1":
            self.cov_all = cal; self.cov_names = list(CAL_COLS)
        else:
            w = self.data[:, self.weather_idx] if self.weather_idx else np.zeros((len(stamps), 0), np.float32)
            self.cov_all = np.concatenate([cal, w], -1).astype(np.float32)
            self.cov_names = list(CAL_COLS) + [meta["channels"][i] for i in self.weather_idx]
            self.cov_weather_cols = list(range(len(CAL_COLS), len(CAL_COLS) + len(self.weather_idx)))
        self.E = self.cov_all.shape[1]
        # NWP-skill noise for each weather covariate, in z-units of the dataset scaling
        # (exp_v2.degrade_cov multiplies by sqrt(lead / 24 h))
        self.cov_weather_names = [meta["channels"][i] for i in self.weather_idx] if regime == "R2" else []
        self.cov_weather_sd = [float(self.sd[i]) for i in self.weather_idx] if regime == "R2" else []
        if regime == "R2" and self.weather_idx:
            phys = nwp_noise_phys(self.cov_weather_names, data[tr0:tr1][:, self.weather_idx])
            self.cov_weather_noise_z = [p / max(s, 1e-6) for p, s in zip(phys, self.cov_weather_sd)]
        else:
            self.cov_weather_noise_z = []

        # ---- phase-locked bases ---------------------------------------------------
        self.basis_cfg = basis_cfg
        self.Phi_all, self.Psi_all = None, None
        if basis_cfg is not None:
            cfg = dict(basis_cfg)
            n_year_slow = cfg.pop("n_year_slow", 3)
            self.Psi_all = slow_basis(stamps, n_year_slow, ref=float(stamps[0]))
            t = local if basis_time == "local" else stamps
            _, dow_b = calendar_features(t)
            masks = []
            dt = cfg.pop("daytype", "weekend+holiday")
            if dt:
                masks = [(dow_b == 5.0), (dow_b == 6.0)]
                if dt == "weekend+holiday":
                    masks.append(flags[:, 0] > 0.5)
            self._masks, self._t = masks, t
            self._cfg = {k: v for k, v in cfg.items() if k in ("n_day", "n_week", "n_year", "trend")}
            self._relative = bool(cfg.get("relative", False))
            self.K = basis_dim_v2(self._cfg.get("n_day", 6), self._cfg.get("n_week", 3),
                                  self._cfg.get("n_year", 0), self._cfg.get("trend", True),
                                  len(masks))
            if not self._relative:
                self.Phi_all = build_basis_v2(t, masks=masks, ref=t[:1], **self._cfg)
            self._trend_col = TREND_COL if self._cfg.get("trend", True) else None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i]); L, H = self.L, self.H
        x = self.data[s: s + L]; y = self.data[s + L: s + L + H]
        out = [torch.from_numpy(x), torch.from_numpy(y)]
        if self.basis_cfg is not None:
            if self.Phi_all is not None:
                Phi = np.array(self.Phi_all[s: s + L + H], copy=True)
                if self._trend_col is not None:
                    Phi[:, self._trend_col] -= Phi[0, self._trend_col]
            else:
                sl = slice(s, s + L + H)
                Phi = build_basis_v2(self._t[sl], masks=[m[sl] for m in self._masks],
                                     relative=True, **self._cfg)
            out.append(torch.from_numpy(np.ascontiguousarray(Phi)))
            out.append(torch.from_numpy(np.ascontiguousarray(self.Psi_all[s: s + L + H])))
        else:
            out += [torch.zeros(1), torch.zeros(1)]
        out.append(torch.from_numpy(np.ascontiguousarray(self.tf_all[s: s + L + H])))
        out.append(torch.from_numpy(np.ascontiguousarray(self.cov_all[s: s + L + H])))
        out.append(torch.tensor(s, dtype=torch.long))
        return tuple(out)


def make_loaders_v2(name, L, H, batch, basis_cfg=None, regime="R1", basis_time="local",
                    workers=0, test_batch=None, train_stride=1, train_frac=1.0,
                    train_first=False, eval_data=None, splits=None):
    """train_frac < 1 keeps that fraction of the training windows, the most recent ones by
    default (learning curves, E1) or the earliest ones with train_first (temporal drift, E3).
    eval_data evaluates on another dataset's validation/test splits, z-scored with that
    dataset's own training statistics (cross-grid transfer, E2)."""
    tr = WindowSetV2(name, "train", L, H, None, basis_cfg, train_stride, regime, basis_time, splits)
    if train_frac < 1.0:
        n = max(1, int(round(len(tr.starts) * train_frac)))
        tr.starts = tr.starts[:n] if train_first else tr.starts[-n:]
    ev = eval_data or name
    ev_tr = tr if ev == name else WindowSetV2(ev, "train", L, H, None, basis_cfg, train_stride, regime, basis_time, splits)
    va = WindowSetV2(ev, "val", L, H, ev_tr.scaler, basis_cfg, 1, regime, basis_time, splits)
    te = WindowSetV2(ev, "test", L, H, ev_tr.scaler, basis_cfg, 1, regime, basis_time, splits)
    if ev != name:
        tr.scaler = ev_tr.scaler       # metrics in the evaluated system's physical units
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    tb = test_batch or batch
    return (DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True, **kw),
            DataLoader(va, batch_size=tb, shuffle=False, **kw),
            DataLoader(te, batch_size=tb, shuffle=False, **kw), tr)


# --------------------------------------------------------------------------- corrupt hours
# Hours of the published actual-load series that are physically impossible and are excluded
# from every test score (a window whose target range covers such an hour is dropped from the
# point scores; the hour itself is masked in the probabilistic scores and in the day-ahead
# protocol). Found by the data audit of the revision (2026-10-19): one hour in the French
# series reads 158 GW between neighbours of 47-50 GW.
BAD_HOURS = {"OPSD-FR": ["2020-07-07T17:00:00Z"],
             "OPSD-FR-mp": ["2020-07-07T17:00:00Z"]}       # the multi-point variant shares the load series


def bad_hours_idx(name, stamps=None):
    """Row indices of the corrupt target hours of a dataset (empty for every other one)."""
    spec = BAD_HOURS.get(name, [])
    if not spec:
        return []
    if stamps is None:
        stamps = np.load(os.path.join(PROC2, name + ".npz"))["stamps"]
    import pandas as pd
    stamps = np.asarray(stamps).astype(np.int64)
    out = []
    for s in spec:
        t = int(pd.Timestamp(s).timestamp())
        j = int(np.searchsorted(stamps, t))
        if j < len(stamps) and stamps[j] == t:
            out.append(j)
    return out


def bad_target_mask(meta, name, L, H, n_windows=None, split="test", stamps=None):
    """(N, H) bool, True where a target hour is usable; False on the corrupt hours."""
    s0, s1 = meta["splits"][split]
    st = np.arange(max(0, s0 - L), s1 - L - H + 1)
    if n_windows is not None:
        st = st[:n_windows]
    mask = np.ones((len(st), H), bool)
    for j in bad_hours_idx(name, stamps):
        k = j - st - L
        ok = (k >= 0) & (k < H)
        mask[np.nonzero(ok)[0], k[ok]] = False
    return mask


def bad_window_idx(meta, name, H, split="test", stamps=None):
    """Indices (in window order, which is target-aligned and independent of the lookback once
    the split start exceeds it) of the windows whose target range covers a corrupt hour."""
    s0, s1 = meta["splits"][split]
    out = set()
    for j in bad_hours_idx(name, stamps):
        for w in range(max(0, j - s0 - H + 1), j - s0 + 1):
            if w < s1 - s0 - H + 1:
                out.add(w)
    return sorted(out)


def list_datasets_v2():
    return sorted(f[:-5] for f in os.listdir(PROC2) if f.endswith(".json"))
