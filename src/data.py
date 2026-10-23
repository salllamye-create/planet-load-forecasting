"""Datasets, windowing and the absolute-time calendar basis used by PLANET."""
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "proc")

SEC_PER_HOUR = 3600.0
# Periods in hours. 8766 h = 365.25 d keeps the annual harmonic phase-locked across leap years.
PERIODS = {"day": 24.0, "week": 168.0, "year": 8766.0}


# --------------------------------------------------------------------------- basis
def calendar_features(stamps_h: np.ndarray):
    """stamps_h: float array of UTC hours since epoch. Returns (hour_of_day, day_of_week)."""
    # 1970-01-01 was a Thursday -> shift by 4 days so 0 == Monday
    hod = np.mod(stamps_h, 24.0)
    dow = np.mod(np.floor(stamps_h / 24.0) + 4.0, 7.0)
    return hod, dow


def build_basis(stamps_h, n_day=6, n_week=3, n_year=3, trend=True, daytype=True,
                ref=None, relative=False):
    """Absolute-time harmonic design matrix Phi.

    stamps_h : (..., N) float64 UTC hours since epoch
    relative : if True, replace absolute time by position within the window (ablation)
    ref      : reference time used to centre the trend term (broadcastable to stamps_h[...,:1])

    Returns Phi with shape (..., N, K), float32.
    """
    s = np.asarray(stamps_h, dtype=np.float64)
    if ref is None:
        ref = s[..., :1]
    if relative:
        # ablation: the basis only knows the offset from the window start, not the wall clock
        s_used = s - ref
    else:
        s_used = s

    cols = [np.ones_like(s)]
    if trend:
        # Weeks since the reference epoch. NOTE: when the basis is precomputed over the
        # whole record (the fast path in WindowSet), `ref` is the record start, so this
        # column must be re-referenced to each window at slice time -- see TREND_COL.
        cols.append((s - ref) / 168.0)

    def harmonics(period, n):
        out = []
        for k in range(1, n + 1):
            w = 2.0 * np.pi * k / period
            out.append(np.sin(w * s_used))
            out.append(np.cos(w * s_used))
        return out

    daily = harmonics(PERIODS["day"], n_day)
    cols += daily
    cols += harmonics(PERIODS["week"], n_week)
    cols += harmonics(PERIODS["year"], n_year)

    if daytype:
        # day-type x daily-shape interaction: weekday / Saturday / Sunday.
        # Uses the true calendar even in the "relative" ablation, so the ablation isolates
        # *phase* locking rather than removing calendar information altogether.
        _, dow = calendar_features(s)
        masks = [(dow <= 4.0), (dow == 5.0), (dow == 6.0)]
        # interact the first 2 daily harmonic pairs with Sat / Sun indicators (weekday is the base level)
        for m in masks[1:]:
            mm = m.astype(np.float64)
            cols.append(mm)
            for h in daily[:4]:
                cols.append(mm * h)

    Phi = np.stack(cols, axis=-1)
    return Phi.astype(np.float32)


TREND_COL = 1          # index of the local-trend column in Phi when trend=True


def basis_dim(n_day=6, n_week=3, n_year=3, trend=True, daytype=True):
    k = 1 + (1 if trend else 0) + 2 * (n_day + n_week + n_year)
    if daytype:
        k += 2 * (1 + 4)
    return k


def slow_basis(stamps_h, n_year=3, ref=None):
    """Slow (climatological) tier: annual harmonics + multi-year trend, on absolute time.

    The trend column is expressed in years since the start of the record so that every
    column is O(1); an un-centred epoch-based column is numerically poor and lets the
    climatological coefficient diverge during training.
    """
    s = np.asarray(stamps_h, dtype=np.float64)
    r = float(s.min()) if ref is None else float(ref)
    cols = [np.ones_like(s), (s - r) / 8766.0]          # years since record start
    for k in range(1, n_year + 1):
        w = 2.0 * np.pi * k / PERIODS["year"]
        cols.append(np.sin(w * s))
        cols.append(np.cos(w * s))
    return np.stack(cols, axis=-1).astype(np.float32)


def slow_dim(n_year=3):
    return 2 + 2 * n_year


def time_features(stamps_h):
    """Standard Informer/Autoformer calendar embedding inputs, scaled to [-0.5, 0.5]."""
    s = np.asarray(stamps_h, dtype=np.float64)
    hod, dow = calendar_features(s)
    days = np.floor(s / 24.0).astype(np.int64)
    # civil date from days since 1970-01-01 (proleptic Gregorian)
    z = days + 719468
    era = np.floor_divide(np.where(z >= 0, z, z - 146096), 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = np.where(mp < 10, mp + 3, mp - 9)
    return np.stack([hod / 23.0 - 0.5, dow / 6.0 - 0.5,
                     (d - 1) / 30.0 - 0.5, (m - 1) / 11.0 - 0.5,
                     (doy) / 365.0 - 0.5], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- dataset
class WindowSet(Dataset):
    """Sliding-window dataset with the standard LTSF train-statistics z-scoring."""

    def __init__(self, name, flag, L, H, scaler=None, basis_cfg=None, stride=1):
        z = np.load(os.path.join(PROC, name + ".npz"))
        meta = json.load(open(os.path.join(PROC, name + ".json")))
        self.meta, self.name, self.L, self.H = meta, name, L, H
        data = z["data"].astype(np.float32)
        stamps = z["stamps"].astype(np.float64) / SEC_PER_HOUR  # UTC hours since epoch
        self.aux = z["aux"].astype(np.float32) if "aux" in z.files else None

        tr0, tr1 = meta["splits"]["train"]
        if scaler is None:
            mu = data[tr0:tr1].mean(0)
            sd = data[tr0:tr1].std(0)
            sd[sd < 1e-6] = 1.0
            scaler = (mu, sd)
        self.mu, self.sd = scaler
        self.scaler = scaler
        self.raw = data
        self.data = (data - self.mu) / self.sd
        self.stamps = stamps

        s0, s1 = meta["splits"][flag]
        # windows are indexed by the first lookback position; the lookback may reach back
        # into the previous split (standard LTSF protocol) but targets stay inside the split
        lo = max(0, s0 - L)
        self.starts = np.arange(lo, s1 - L - H + 1, stride, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError(f"{name}/{flag}: no windows for L={L} H={H}")
        self.tf_all = time_features(stamps)                      # (T, 5) calendar embedding inputs
        self.basis_cfg = basis_cfg
        self.Phi_all, self.Psi_all = None, None
        if basis_cfg is not None:
            cfg = {k: v for k, v in basis_cfg.items() if k != "n_year_slow"}
            # one reference epoch for the whole series, so train/val/test share a scale
            self.Psi_all = slow_basis(stamps, basis_cfg.get("n_year_slow", 3),
                                      ref=float(stamps[0]))
            if not cfg.get("relative", False):
                self.Phi_all = build_basis(stamps, **cfg)         # absolute-time -> precompute once
            self._cfg = cfg
            # Precomputing over the whole record leaves the trend column measured from the
            # record start (0..300 weeks on OPSD), which is O(100) rather than O(1): it
            # wrecks the conditioning of the normal equations and the ridge then shrinks the
            # local-slope direction to almost nothing. Re-reference it per window.
            self._trend_col = TREND_COL if cfg.get("trend", True) else None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        L, H = self.L, self.H
        x = self.data[s: s + L]
        y = self.data[s + L: s + L + H]
        out = [torch.from_numpy(x), torch.from_numpy(y)]
        if self.basis_cfg is not None:
            if self.Phi_all is not None:
                Phi = np.array(self.Phi_all[s: s + L + H], copy=True)
                if self._trend_col is not None:
                    Phi[:, self._trend_col] -= Phi[0, self._trend_col]
            else:
                Phi = build_basis(self.stamps[s: s + L + H], **self._cfg)
            out.append(torch.from_numpy(np.ascontiguousarray(Phi)))
            out.append(torch.from_numpy(np.ascontiguousarray(self.Psi_all[s: s + L + H])))
        else:
            out.append(torch.zeros(1))
            out.append(torch.zeros(1))
        out.append(torch.from_numpy(np.ascontiguousarray(self.tf_all[s: s + L + H])))
        out.append(torch.tensor(s, dtype=torch.long))
        return tuple(out)


def make_loaders(name, L, H, batch, basis_cfg=None, workers=0, test_stride=1,
                 test_batch=None, train_stride=1):
    # train_stride>1 subsamples the (heavily overlapping) training windows; it is applied
    # identically to every method and never to the validation or test splits.
    tr = WindowSet(name, "train", L, H, None, basis_cfg, stride=train_stride)
    va = WindowSet(name, "val", L, H, tr.scaler, basis_cfg)
    te = WindowSet(name, "test", L, H, tr.scaler, basis_cfg, stride=test_stride)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    tb = test_batch or batch
    return (DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True, **kw),
            DataLoader(va, batch_size=tb, shuffle=False, **kw),
            DataLoader(te, batch_size=tb, shuffle=False, **kw),
            tr)


def list_datasets():
    return sorted(f[:-5] for f in os.listdir(PROC) if f.endswith(".json"))
