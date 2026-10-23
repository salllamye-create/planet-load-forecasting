"""Tao Hong's "vanilla benchmark" with the recency effect, under the v2 information regimes.

The statistical reference model of the GEFCom load-forecasting literature (Hong 2010; Hong,
Pinson & Fan 2014, IJF): an OLS regression of hourly load on

    Trend + Month + Weekday + Hour + Weekday x Hour
    + T + T^2 + T^3 + (T, T^2, T^3) x Month + (T, T^2, T^3) x Hour

with the recency effect of Wang, Liu & Hong (2016, IJF): the same cubic block for each of the
h preceding hourly temperatures T_{t-1..t-h} and for the daily moving-average temperatures of
the d preceding 24-hour blocks. (d, h) is chosen on the validation split among
{0,1,2,3} x {0,6,12,24}. National holidays are treated as Sunday (cal_flags[:, 0], local
day-of-week); month / weekday / hour come from the local civil time; the trend is hours since
the start of the training split; categoricals are treatment-coded one-hots; the fit is
numpy least squares on the column-standardised normal equations with a ridge of 1e-6.

Three model names share the specification:
  TaoVanilla  fitted once on the training split (the textbook benchmark); the linear trend is
              kept or dropped by validation loss, like the recency configuration;
  TaoRefit    re-fitted monthly on every row before the start of the local month that contains
              the forecast origin (operator practice; strictly past data, so no leakage). The
              recency configuration is chosen on validation with the train-only fit, exactly
              as for TaoVanilla; the fitting window (expanding, or the last 36 or 24 months)
              and the trend term are then chosen by the validation loss of monthly-refit
              forecasts over the validation months. The rolling windows exist because a
              linear trend fitted on an expanding window extrapolates any level change in the
              published series (the Dutch series steps up by 17 % in January 2016).
  TaoRefitAR  TaoRefit plus an autoregressive model of the regression residual (regression with
              AR errors, the classical dynamic-regression benchmark): the AR(p) coefficients are
              estimated by ridge least squares on the training residuals of the train-only fit
              at the selected recency and trend, the order p is chosen among {0, 24, 48, 168, 240} by
              the validation loss of the monthly-refit forecasts with the correction (p = 0 is
              TaoRefit itself), and each window's forecast adds the recursive AR forecast of the
              residual started from the p residuals ending at the forecast origin (actual load
              minus the regression evaluated with the coefficients of the origin's month and the
              true past temperatures). The AR history uses only the p hours before the origin,
              within the common lookback of L = 336.

Regimes: R1 = calendar-only model (every T term dropped); R2 = full model with the
temperature at the target step and in the recency terms degraded to NWP skill exactly as
exp_v2.degrade_cov (sd = 1.2 K * sqrt(lead/24), lead = hours after the last observed step;
past hours keep the true temperature). R0 cannot exist for a pure covariate regression
(without the calendar there is nothing left), so only R1 and R2 are written. The forecast for
a window is the model evaluated at its H target steps (a regression on known-future
covariates); the fit is in physical units.

Writes results_v2/<tag>.json + results_v2/errors/<tag>.npz,
tag = {data}_{TaoVanilla|TaoRefit}_L{L}_H{H}_s{seed}_v2-{regime}; resumable.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classic_v2 import (GRIDS, NWP_RMSE_24H, RES_DEFAULT, load_v2, starts, metrics, record,
                        write_result, done, parse_list, seeds_for)

N_CAL = 1 + 1 + 11 + 6 + 23 + 6 * 23          # 180 calendar columns
N_TB = 3 * (1 + 11 + 23)                       # 105 columns per temperature variable
MAX_H, MAX_D = 24, 3
K_MAX = 1 + MAX_H + MAX_D                      # current T, 24 hourly lags, 3 daily means
LOOKBACK = 24 * MAX_D                          # rows needed before the first training row
CONFIGS = [(d, h) for d in (0, 1, 2, 3) for h in (0, 6, 12, 24)]
MODELS = ("TaoVanilla", "TaoRefit", "TaoRefitAR")
AR_ORDERS = (0, 24, 48, 168, 240)              # TaoRefitAR residual AR order (0 = no correction);
                                               # 240 + 72 h of recency lags stays inside L = 336
AR_RIDGE = 1e-6
TREND_COL = 1                                  # column of the linear trend in the calendar block
WINDOWS = (0, 36, 24)                          # TaoRefit fitting window in months (0 = expanding)
TRENDS = (True, False)                         # with / without the linear trend


def onehot(v, n):
    """Treatment coding: level 0 is the reference, columns for levels 1..n-1."""
    M = np.zeros((len(v), n - 1))
    m = v > 0
    M[np.nonzero(m)[0], v[m] - 1] = 1.0
    return M


def cal_block(hour, wday, month, trend):
    Mo, Wd, Hr = onehot(month, 12), onehot(wday, 7), onehot(hour, 24)
    WH = (Wd[:, :, None] * Hr[:, None, :]).reshape(len(hour), -1)
    return np.hstack([np.ones((len(hour), 1)), trend[:, None], Mo, Wd, Hr, WH]), Mo, Hr


def temp_block(x, Mo, Hr):
    cols = []
    for p in (1, 2, 3):
        xp = (x ** p)[:, None]
        cols += [xp, xp * Mo, xp * Hr]
    return np.hstack(cols)


def temp_vars(ts, rows):
    """(n, K_MAX) recency variables at the rows: T_t, T_{t-1..t-24}, daily means of the
    24-hour blocks 1..3 before t (Wang, Liu & Hong 2016)."""
    n = len(rows)
    V = np.zeros((n, K_MAX))
    V[:, 0] = ts[rows]
    for j in range(1, MAX_H + 1):
        V[:, j] = ts[rows - j]
    cs = np.concatenate([[0.0], np.cumsum(ts)])
    for d in range(1, MAX_D + 1):
        V[:, MAX_H + d] = (cs[rows - 24 * (d - 1)] - cs[rows - 24 * d]) / 24.0
    return V


def var_ids(d, h):
    return [0] + list(range(1, h + 1)) + [MAX_H + i for i in range(1, d + 1)]


def cols_for(ks):
    cols = list(range(N_CAL))
    for k in ks:
        cols += [N_CAL + k * N_TB + j for j in range(N_TB)]
    return np.array(cols)


def solve(G, g, N, cols, lam=1e-6, trend=True):
    """Ridge least squares on the selected columns of the normal equations; with trend=False
    the trend column is left out of the fit and its coefficient set to zero, so the compact
    coefficient layout (calendar block, then one block per recency variable) is unchanged."""
    cols = np.asarray(cols)
    keep = np.ones(len(cols), bool) if trend else (cols != TREND_COL)
    c = cols[keep]
    Gs, gs = G[np.ix_(c, c)], g[c]
    D = np.sqrt(np.diag(Gs) / N); D[D <= 0] = 1.0
    A = Gs / (N * np.outer(D, D)); b = gs / (N * D)
    beta = np.zeros(len(cols))
    beta[keep] = np.linalg.solve(A + lam * np.eye(len(c)), b) / D
    return beta


def coef_tables(beta, ks, month, hour):
    """c0[t] (calendar part) and B[t, i, p]: the cubic coefficients of variable ks[i] at every
    row, so that yhat(t) = c0[t] + sum_i sum_p B[t, i, p] * x_i^(p+1)."""
    b_cal = beta[:N_CAL]
    B = np.zeros((len(month), len(ks), 3))
    for i in range(len(ks)):
        for p in range(3):
            off = N_CAL + i * N_TB + p * 35
            bm = np.concatenate([[0.0], beta[off + 1: off + 12]])
            bh = np.concatenate([[0.0], beta[off + 12: off + 35]])
            B[:, i, p] = beta[off] + bm[month] + bh[hour]
    return b_cal, B


class Tao:
    def __init__(self, ds, regime):
        meta = ds["meta"]
        self.ds, self.regime, self.meta = ds, regime, meta
        self.tr0, self.tr1 = meta["splits"]["train"]
        self.y = ds["data"][:, meta["target_idx"]]
        self.sd_t = float(ds["sd"][meta["target_idx"]])
        self.hour = (np.round(ds["hod"]).astype(int)) % 24
        wday = ds["dow"].astype(int)
        self.wday = np.where(ds["flags"][:, 0] > 0.5, 6, wday)            # holiday -> Sunday
        lt = pd.to_datetime(ds["local"], unit="s")
        self.month = lt.month.values - 1
        self.mcode = (lt.year * 12 + lt.month - 1).values                  # local month index
        self.trend = (ds["stamps"] - ds["stamps"][self.tr0]) / 3600.0
        self.Xcal, self.Mo, self.Hr = cal_block(self.hour, self.wday, self.month, self.trend)
        if regime == "R2":
            ti = meta["channels"].index("temp")
            temp = ds["data"][:, ti]
            self.mu_T, self.sd_T = temp[self.tr0:self.tr1].mean(), temp[self.tr0:self.tr1].std()
            self.ts = (temp - self.mu_T) / self.sd_T                        # conditioning only
            self.noise_z = NWP_RMSE_24H["temp"] / self.sd_T                  # exp_v2.degrade_cov
        else:
            self.ts = None
        self.betas = {}

    # ---- design / fit ------------------------------------------------------------------
    def design(self, r):
        X = [self.Xcal[r]]
        if self.ts is not None:
            V = temp_vars(self.ts, r)
            X += [temp_block(V[:, k], self.Mo[r], self.Hr[r]) for k in range(K_MAX)]
        return np.hstack(X)

    def accumulate(self, G, g, rows, chunk=4096):
        for c in range(0, len(rows), chunk):
            r = rows[c:c + chunk]
            X = self.design(r)
            G += X.T @ X; g += X.T @ self.y[r]
        return G, g

    def fit(self):
        rows = np.arange(self.tr0 + LOOKBACK, self.tr1)
        P = N_CAL + (K_MAX * N_TB if self.ts is not None else 0)
        G, g = self.accumulate(np.zeros((P, P)), np.zeros(P), rows)
        self.G, self.g, self.N = G, g, len(rows)
        cfgs = CONFIGS if self.ts is not None else [(0, 0)]
        for d, h in cfgs:
            ks = var_ids(d, h) if self.ts is not None else []
            for tr in TRENDS:
                self.betas[(d, h, tr)] = (ks, solve(G, g, self.N, cols_for(ks), trend=tr))

    def monthly_betas(self, cfg, a_first, a_last, window=0, trend=True):
        """TaoRefit: for every local month that contains a forecast origin in [a_first, a_last]
        the coefficients fitted on the rows before the first row of that month: every row since
        the start of training (window=0, the expanding window: training Gram plus every later
        row) or the rows of the last `window` months (rolling window; the Gram matrix is slid by
        adding the entering month and subtracting the leaving one). Strictly past data."""
        ks = var_ids(*cfg) if self.ts is not None else []
        cols = cols_for(ks)
        first_row = self.tr0 + LOOKBACK
        P = len(self.G)
        G = g = None
        N, lo, hi = 0, first_row, first_row
        if window == 0:
            G, g, N, lo, hi = self.G.copy(), self.g.copy(), self.N, first_row, self.tr1
        out = {}
        for m in np.unique(self.mcode[a_first:a_last + 1]):
            m_start = int(np.searchsorted(self.mcode, m))
            w_lo = first_row if window == 0 else max(first_row, int(np.searchsorted(self.mcode, m - window)))
            if G is None:                                   # first rolling fit, from scratch
                rows = np.arange(w_lo, m_start)
                G, g = self.accumulate(np.zeros((P, P)), np.zeros(P), rows)
                N, lo, hi = len(rows), w_lo, m_start
            else:
                add = np.arange(hi, max(hi, m_start))
                if len(add):
                    G, g = self.accumulate(G, g, add); N += len(add); hi = m_start
                if w_lo > lo:                               # rows leaving the window
                    drop = np.arange(lo, w_lo)
                    Gd, gd = self.accumulate(np.zeros((P, P)), np.zeros(P), drop)
                    G -= Gd; g -= gd; N -= len(drop); lo = w_lo
            out[int(m)] = (ks, solve(G, g, N, cols, trend=trend), int(hi), int(lo))
        return out

    # ---- forecast --------------------------------------------------------------------
    def predict(self, cfg, st, L, H, exo_mode, gen, chunk=256):
        ks, beta = self.betas[cfg]
        return self.predict_beta(ks, beta, st, L, H, exo_mode, gen, chunk)

    def predict_beta(self, ks, beta, st, L, H, exo_mode, gen, chunk=256):
        b_cal, B = coef_tables(beta, ks, self.month, self.hour)
        c0 = self.Xcal @ b_cal
        out = np.zeros((len(st), H))
        q = L + np.arange(H)
        nsd = self.noise_z * np.sqrt(np.arange(1, H + 1) / 24.0) if self.ts is not None else None
        for c in range(0, len(st), chunk):
            s = st[c:c + chunk]; n = len(s)
            tau = s[:, None] + q[None, :]
            yhat = c0[tau].copy()
            if ks:
                Pth = self.ts[s[:, None] + np.arange(L + H)[None, :]]      # true temperature path
                if exo_mode == "noisy":
                    Pth[:, L:] += gen.normal(size=(n, H)) * nsd
                elif exo_mode == "persist":
                    Pth[:, L:] = Pth[:, L - 24 + (np.arange(H) % 24)]
                cs = np.concatenate([np.zeros((n, 1)), np.cumsum(Pth, 1)], 1)
                for i, k in enumerate(ks):
                    if k <= MAX_H:
                        x = Pth[:, q - k]
                    else:
                        d = k - MAX_H
                        x = (cs[:, q - 24 * (d - 1)] - cs[:, q - 24 * d]) / 24.0
                    Bk = B[:, i, :][tau]
                    yhat += Bk[..., 0] * x + Bk[..., 1] * x ** 2 + Bk[..., 2] * x ** 3
            out[c:c + n] = yhat
        return out

    def predict_refit(self, mbetas, st, L, H, exo_mode, gen):
        """Each window is forecast with the coefficients of the month of its origin (st+L-1);
        windows are ascending, so the noise stream equals the once-fitted model's."""
        out = np.zeros((len(st), H))
        mon = self.mcode[st + L - 1]
        for m in np.unique(mon):
            idx = np.nonzero(mon == m)[0]
            ks, beta = mbetas[int(m)][:2]
            out[idx] = self.predict_beta(ks, beta, st[idx], L, H, exo_mode, gen)
        return out

    def truth(self, st, L, H):
        return np.stack([self.y[st + L + h] for h in range(H)], 1)

    # ---- residual AR model (TaoRefitAR) -------------------------------------------------
    def fitted_rows(self, ks, beta, rows):
        """The regression evaluated at arbitrary rows with the true temperatures (in-sample
        values; the rows must be at least LOOKBACK after the start of the series)."""
        b_cal, B = coef_tables(beta, ks, self.month, self.hour)
        yhat = self.Xcal[rows] @ b_cal
        if ks:
            V = temp_vars(self.ts, rows)
            for i, k in enumerate(ks):
                x = V[:, k]
                Bk = B[rows, i, :]
                yhat = yhat + Bk[:, 0] * x + Bk[:, 1] * x ** 2 + Bk[:, 2] * x ** 3
        return yhat

    def train_resid(self, cfg, chunk=8192):
        """Residuals of the train-only fit `cfg` on the training rows."""
        ks, beta = self.betas[cfg]
        rows = np.arange(self.tr0 + LOOKBACK, self.tr1)
        out = np.empty(len(rows))
        for c in range(0, len(rows), chunk):
            r = rows[c:c + chunk]
            out[c:c + len(r)] = self.y[r] - self.fitted_rows(ks, beta, r)
        return out

    @staticmethod
    def fit_ar(e, p, lam=AR_RIDGE):
        """AR(p) of a residual series by ridge least squares: e_t = sum_j phi_j e_{t-j}."""
        n = len(e) - p
        X = np.stack([e[p - j: p - j + n] for j in range(1, p + 1)], 1)   # column j-1: e_{t-j}
        y = e[p:]
        G = X.T @ X / n
        g = X.T @ y / n
        return np.linalg.solve(G + lam * np.eye(p), g)

    def ar_history(self, mbetas, st, L, p):
        """(n, p) residuals at the p rows ending at each window's origin st+L-1, oldest first,
        under the coefficients of the origin's month (mbetas[month] -> (ks, beta, ...))."""
        o = st + L - 1
        R = o[:, None] - p + 1 + np.arange(p)[None, :]
        E = np.empty(R.shape)
        mon = self.mcode[o]
        for m in np.unique(mon):
            idx = np.nonzero(mon == m)[0]
            ks, beta = mbetas[int(m)][:2]
            lo, hi = int(R[idx].min()), int(R[idx].max()) + 1
            rows = np.arange(lo, hi)
            e = self.y[rows] - self.fitted_rows(ks, beta, rows)
            E[idx] = e[R[idx] - lo]
        return E

    @staticmethod
    def ar_forecast(phi, E, H):
        """Recursive AR forecast of the residual over H steps from the histories E (n, p)."""
        rev = phi[::-1]
        buf = E.copy()
        out = np.zeros((len(E), H))
        for h in range(H):
            out[:, h] = buf @ rev
            buf = np.concatenate([buf[:, 1:], out[:, h:h + 1]], 1)
        return out


def run_grid(name, model_name, horizons, regimes, seeds, a, out_dir):
    ds = load_v2(name)
    meta, L = ds["meta"], a.L
    refit = model_name in ("TaoRefit", "TaoRefitAR")
    use_ar = model_name == "TaoRefitAR"
    for regime in [r for r in regimes if r in ("R1", "R2")]:
        suffix = f"_v2-{regime}" + ("" if a.exo_mode == "noisy" or regime == "R1" else f"-{a.exo_mode}")
        todo = [(H, s) for H in horizons for s in seeds_for(regime, seeds)
                if a.force or not done(out_dir, f"{name}_{model_name}_L{L}_H{H}_s{s}{suffix}")]
        if not todo:
            print(f"[{name} {model_name} {regime}] all done, skipped", flush=True); continue
        t0 = time.time()
        model = Tao(ds, regime)
        model.fit()
        t_fit = time.time() - t0
        print(f"[{name} {model_name} {regime}] fitted {len(model.betas)} configs on {model.N} rows "
              f"({t_fit:.0f}s)", flush=True)
        mcache, vcache, rcache = {}, {}, {}
        for seed in seeds_for(regime, seeds):
            # recency and trend selection on the validation windows at day-ahead (H=24), own NWP
            # noise, train-only fit (the same rule for TaoVanilla and TaoRefit)
            t1 = time.time()
            hist, best, cfg = [], float("inf"), (0, 0, True)
            st_va = starts(meta, "val", L, 24)
            y_va = model.truth(st_va, L, 24)
            for (d, h) in (CONFIGS if regime == "R2" else [(0, 0)]):
                for tr in TRENDS:
                    gen = np.random.default_rng([seed + 7, 0])
                    p = model.predict((d, h, tr), st_va, L, 24, a.exo_mode, gen)
                    v = float((((p - y_va) / model.sd_t) ** 2).mean())
                    hist.append(dict(epoch=0, train=float("nan"), val=v, recency_days=d, recency_hours=h, trend=tr))
                    if v < best - 1e-12:
                        best, cfg = v, (d, h, tr)
            print(f"[{name} {model_name} {regime} s{seed}] recency (days, hours) = {cfg[:2]}  "
                  f"trend={cfg[2]}  val_mse={best:.4f}", flush=True)
            mbetas, refit_cfg = None, None
            if refit:
                # fitting window and trend: validation loss of monthly-refit forecasts over the
                # validation months at the selected recency (coefficients are deterministic and
                # cached across seeds; the NWP noise stream is the seed's own)
                key = cfg[:2]
                o_first, o_last = int(st_va[0] + L - 1), int(st_va[-1] + L - 1)
                bestv, sel = float("inf"), (0, True)
                for W in WINDOWS:
                    for tr in TRENDS:
                        if (key, W, tr) not in vcache:
                            vcache[(key, W, tr)] = model.monthly_betas(key, o_first, o_last, window=W, trend=tr)
                        gen = np.random.default_rng([seed + 7, 0])
                        p = model.predict_refit(vcache[(key, W, tr)], st_va, L, 24, a.exo_mode, gen)
                        v = float((((p - y_va) / model.sd_t) ** 2).mean())
                        hist.append(dict(epoch=1, train=float("nan"), val=v, window_months=W, trend=tr))
                        if v < bestv - 1e-12:
                            bestv, sel = v, (W, tr)
                refit_cfg, best = sel, bestv
                print(f"[{name} {model_name} {regime} s{seed}] refit window = "
                      f"{'expanding' if sel[0] == 0 else str(sel[0]) + ' months'}  trend={sel[1]}  "
                      f"val_mse={bestv:.4f}", flush=True)
                if (key, sel) not in mcache:                 # deterministic: shared by seeds and H
                    st24 = starts(meta, "test", L, 24)      # the widest set of origins
                    mcache[(key, sel)] = model.monthly_betas(key, int(st24[0] + L - 1), int(st24[-1] + L - 1),
                                                             window=sel[0], trend=sel[1])
                    print(f"[{name} {model_name} {regime}] {len(mcache[(key, sel)])} monthly refits "
                          f"({time.time()-t1:.0f}s)", flush=True)
                mbetas = mcache[(key, sel)]
            phi, p_sel = None, 0
            if use_ar:
                # residual AR model: coefficients from the training residuals of the train-only
                # fit at the selected recency / trend (cached across seeds); the order is chosen
                # by the validation loss of the monthly-refit forecasts with the correction
                if cfg not in rcache:
                    rcache[cfg] = model.train_resid(cfg)
                e_tr = rcache[cfg]
                vb = vcache[(key, sel[0], sel[1])]
                besta = float("inf")
                for p in AR_ORDERS:
                    ph = model.fit_ar(e_tr, p) if p > 0 else None
                    gen = np.random.default_rng([seed + 7, 0])
                    pv = model.predict_refit(vb, st_va, L, 24, a.exo_mode, gen)
                    if p > 0:
                        pv = pv + model.ar_forecast(ph, model.ar_history(vb, st_va, L, p), 24)
                    v = float((((pv - y_va) / model.sd_t) ** 2).mean())
                    hist.append(dict(epoch=2, train=float("nan"), val=v, ar_order=p))
                    if v < besta - 1e-12:
                        besta, p_sel, phi = v, p, ph
                best = besta
                print(f"[{name} {model_name} {regime} s{seed}] AR order = {p_sel}  val_mse={besta:.4f}",
                      flush=True)
            t_sel = time.time() - t1
            for H in horizons:
                tag = f"{name}_{model_name}_L{L}_H{H}_s{seed}{suffix}"
                if (H, seed) not in todo:
                    print(f"[{tag}] exists, skipped", flush=True); continue
                st_te = starts(meta, "test", L, H)
                gen = np.random.default_rng([seed + 7, H])
                t2 = time.time()
                if refit:
                    preds = model.predict_refit(mbetas, st_te, L, H, a.exo_mode, gen)
                    if use_ar and p_sel > 0:
                        preds = preds + model.ar_forecast(phi, model.ar_history(mbetas, st_te, L, p_sel), H)
                else:
                    preds = model.predict(cfg, st_te, L, H, a.exo_mode, gen)
                t_inf = time.time() - t2
                met, arrs = metrics(preds, model.truth(st_te, L, H), model.sd_t)
                ks, beta = model.betas[cfg]
                extra = dict(recency=dict(days=cfg[0], hours=cfg[1]), n_train=int(model.N), ridge=1e-6,
                             trend=bool(refit_cfg[1] if refit else cfg[2]),
                             refit=(dict(kind="monthly", window_months=int(refit_cfg[0]),
                                         window="expanding" if refit_cfg[0] == 0 else f"{refit_cfg[0]} months")
                                    if refit else None),
                             n_refits=len(mbetas) if refit else 0,
                             ar=(dict(order=int(p_sel), orders=list(AR_ORDERS), ridge=AR_RIDGE,
                                      fitted_on="training residuals of the train-only fit")
                                 if use_ar else None))
                rec = record(name, model_name, L, H, seed, regime, a.exo_mode, met,
                             t_fit + t_sel, t_inf, suffix, E=8 if regime == "R1" else 9,
                             params=len(beta) + int(p_sel), train_stride=1, val_mse=best, history=hist,
                             **extra)
                write_result(out_dir, tag, rec, arrs)
                print(f"[{tag}] MSE={met['mse']:.4f} MAE={met['mae']:.4f} WAPE={met['wape']:.2f}% "
                      f"recency={cfg[:2]} trend={extra['trend']}"
                      + (f" window={extra['refit']['window']}" if refit else "")
                      + (f" ar={p_sel}" if use_ar else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TaoVanilla", choices=list(MODELS))
    ap.add_argument("--datasets", default=",".join(GRIDS))
    ap.add_argument("--horizons", default="24,96,336")
    ap.add_argument("--regimes", default="R1,R2")
    ap.add_argument("--seeds", default="2021,2022,2023")
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--exo_mode", default="noisy", choices=["noisy", "perfect", "persist"])
    ap.add_argument("--force", type=int, default=0)
    ap.add_argument("--out", default=RES_DEFAULT)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "errors"), exist_ok=True)
    T0 = time.time()
    for d in parse_list(a.datasets):
        run_grid(d, a.model, parse_list(a.horizons, int), parse_list(a.regimes),
                 parse_list(a.seeds, int), a, a.out)
        print(f"-- {d} done, elapsed {(time.time()-T0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
