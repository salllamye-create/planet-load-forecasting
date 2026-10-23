"""Probabilistic load forecasting: quantile training, conformalised intervals and
the operational scores a control-room actually cares about.

All methods share one recipe:
  * pinball (quantile) loss over a fixed level set,
  * conformalised quantile regression (CQR) calibrated on the validation split,
  * adaptive conformal inference (ACI) updating the level online across the test split,
so that any difference between methods comes from the forecaster, not the calibrator.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

from data import make_loaders, basis_dim, slow_dim
from baselines import REGISTRY
from planet import PLANET
import exp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results", "prob")
os.makedirs(RES, exist_ok=True)

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
LO, HI = 0, len(QUANTILES) - 1          # indices of the 5 % / 95 % levels -> 90 % interval


class QuantileWrapper(nn.Module):
    """Turns any point forecaster into a quantile forecaster by widening its output
    head to H*Q and reshaping. Monotonicity across levels is imposed by construction:
    the model emits a median plus non-negative increments."""

    def __init__(self, name, L, H, C, n_q, extra=None):
        super().__init__()
        self.H, self.n_q = H, n_q
        cfg = dict(exp.MODEL_CFG.get(name, {}))
        cfg.update(extra or {})
        self.native = name == "PLANET"
        if self.native:
            bc = dict(exp.BASIS_CFG)
            bc.update({k: v for k, v in cfg.items() if k in
                       ("n_day", "n_week", "n_year", "trend", "daytype", "relative", "n_year_slow")})
            kf = basis_dim(bc["n_day"], bc["n_week"], bc["n_year"], bc["trend"], bc["daytype"])
            ks = slow_dim(bc["n_year_slow"])
            mc = {k: v for k, v in cfg.items() if k not in
                  ("n_day", "n_week", "n_year", "trend", "daytype", "relative", "n_year_slow")}
            self.basis_cfg = bc
            self.m = PLANET(L, H, C, kf, ks, n_quant=n_q, **mc)
        else:
            self.basis_cfg = dict(exp.BASIS_CFG)
            self.m = REGISTRY[name](L=L, H=H * n_q, C=C, **cfg)

    def forward(self, x, Phi, Psi, tf):
        o = self.m(x, Phi, Psi, tf)
        if not self.native:
            B, _, C = o.shape
            o = o.reshape(B, self.n_q, self.H, C).permute(0, 2, 3, 1)     # (B,H,C,Q)
        med = o[..., self.n_q // 2: self.n_q // 2 + 1]
        dn = -torch.flip(torch.cumsum(torch.flip(
            torch.nn.functional.softplus(o[..., :self.n_q // 2]), (-1,)), -1), (-1,))
        up = torch.cumsum(torch.nn.functional.softplus(o[..., self.n_q // 2 + 1:]), -1)
        return torch.cat([med + dn, med, med + up], -1)                   # sorted across levels


def pinball(pred, true, qs):
    """pred (B,H,C,Q), true (B,H,C) -> mean pinball loss."""
    e = true.unsqueeze(-1) - pred
    q = torch.as_tensor(qs, device=pred.device, dtype=pred.dtype)
    return torch.maximum(q * e, (q - 1.0) * e).mean()


def crps_from_quantiles(pin_per_q, qs):
    """Trapezoidal integral of 2*pinball(q) over q in (0,1) -- the standard
    quantile-decomposition approximation to CRPS."""
    q = np.asarray(qs, dtype=float)
    v = 2.0 * np.asarray(pin_per_q, dtype=float)
    return float(np.trapz(v, q) / (q[-1] - q[0]))


def winkler(lo, hi, y, alpha):
    w = hi - lo
    below = (y < lo).astype(float) * (2.0 / alpha) * (lo - y)
    above = (y > hi).astype(float) * (2.0 / alpha) * (y - hi)
    return float((w + below + above).mean())


def collect(model, loader, dev, target_idx, scaler):
    """Returns per-window arrays of quantile forecasts and truths, in original units."""
    model.eval()
    P, Y = [], []
    mu, sd = scaler[0][target_idx], scaler[1][target_idx]
    with torch.no_grad():
        for x, y, Phi, Psi, tf, _ in loader:
            x, y = x.to(dev), y.to(dev)
            p = model(x, Phi.to(dev), Psi.to(dev), tf.to(dev)).float()
            P.append((p[:, :, target_idx, :] * sd + mu).cpu().numpy())
            Y.append((y[:, :, target_idx] * sd + mu).cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)       # (N,H,Q), (N,H)


def conformalise(Pc, Yc, Pt, alpha=0.10, gamma=0.01):
    """CQR offsets from the calibration split, then ACI online over the test split.

    Scores and offsets are kept per horizon step, which is the usual multi-horizon
    treatment: step h=1 is far easier to cover than step h=H.
    """
    N, H, Q = Pt.shape
    E = np.maximum(Pc[:, :, LO] - Yc, Yc - Pc[:, :, HI])      # (Nc, H) conformity scores
    nC = E.shape[0]
    # the split-conformal offset is fixed per horizon step, so it is computed once
    k = min(max(int(np.ceil((1 - alpha) * (nC + 1))) - 1, 0), nC - 1)
    off = np.partition(E, k, axis=0)[k]                       # (H,)
    return Pt[:, :, LO] - off, Pt[:, :, HI] + off, np.full(H, alpha)


def conformalise_aci(Pc, Yc, Pt, Yt, alpha=0.10, gamma=0.005):
    """Adaptive conformal inference: the working level is nudged after every window so
    that empirical coverage tracks the nominal level even under drift."""
    N, H, Q = Pt.shape
    E = np.maximum(Pc[:, :, LO] - Yc, Yc - Pc[:, :, HI])
    Es = np.sort(E, axis=0)
    nC = Es.shape[0]
    lo_t, hi_t = Pt[:, :, LO], Pt[:, :, HI]
    out_lo, out_hi = np.empty((N, H)), np.empty((N, H))
    alph = np.full(H, alpha)
    for i in range(N):
        a = np.clip(alph, 1e-3, 0.5)
        k = np.clip(np.ceil((1 - a) * (nC + 1)).astype(int) - 1, 0, nC - 1)
        off = Es[k, np.arange(H)]
        out_lo[i] = lo_t[i] - off
        out_hi[i] = hi_t[i] + off
        err = ((Yt[i] < out_lo[i]) | (Yt[i] > out_hi[i])).astype(float)
        alph = alph + gamma * (alpha - err)
    return out_lo, out_hi


def score(P, Y, lo, hi, alpha=0.10, tag=""):
    pin_q = [float(np.maximum(q * (Y - P[:, :, j]), (q - 1) * (Y - P[:, :, j])).mean())
             for j, q in enumerate(QUANTILES)]
    cov = float(((Y >= lo) & (Y <= hi)).mean())
    return {
        f"pinball{tag}": float(np.mean(pin_q)),
        f"crps{tag}": crps_from_quantiles(pin_q, QUANTILES),
        f"picp{tag}": 100.0 * cov,
        f"ace{tag}": 100.0 * abs(cov - (1 - alpha)),
        f"mpiw{tag}": float((hi - lo).mean()),
        f"winkler{tag}": winkler(lo, hi, Y, alpha),
    }


# --------------------------------------------------------------------------- v2 calibration
# Symmetric level pairs of QUANTILES and their nominal miscoverage: (5, 95) -> 0.10,
# (10, 90) -> 0.20, (25, 75) -> 0.50. Every pair is conformalised; the median is left as emitted.
PAIRS = [(0, 6, 0.10), (1, 5, 0.20), (2, 4, 0.50)]


def qcrps_from_quantiles(pin_per_q, qs):
    """Quantile-weighted approximation to the CRPS: the trapezoidal integral of 2 x pinball(q)
    over the emitted levels, divided by the level range. It is not the CRPS of a continuous
    predictive distribution (the tails beyond the 5/95 levels are not scored); the paper calls
    it the quantile-weighted CRPS approximation."""
    return crps_from_quantiles(pin_per_q, qs)


def _offsets(Pc, Yc, lo, hi, alpha):
    """Split-conformal offset per horizon step for the (lo, hi) pair, and the sorted scores."""
    E = np.maximum(Pc[:, :, lo] - Yc, Yc - Pc[:, :, hi])        # (Nc, H) conformity scores
    nC = E.shape[0]
    k = min(max(int(np.ceil((1 - alpha) * (nC + 1))) - 1, 0), nC - 1)
    Es = np.sort(E, axis=0)
    return Es[k], Es


def conformalise_q(Pc, Yc, Pt, pairs=PAIRS):
    """CQR (Romano et al. 2019) on every symmetric level pair: fixed per-step offsets from the
    calibration split applied to the test quantiles. Returns the calibrated (N, H, Q) array,
    rearranged to be monotone across levels (Chernozhukov, Fernandez-Val & Galichon 2010)."""
    Pq = Pt.copy()
    for lo, hi, alpha in pairs:
        off, _ = _offsets(Pc, Yc, lo, hi, alpha)
        Pq[:, :, lo] = Pt[:, :, lo] - off
        Pq[:, :, hi] = Pt[:, :, hi] + off
    return np.sort(Pq, axis=-1)


def conformalise_aci_q(Pc, Yc, Pt, Yt, gamma=0.005, pairs=PAIRS, delayed=True, mask=None):
    """Adaptive conformal inference (Gibbs & Candes 2021) on every level pair, per horizon step.

    delayed=True is the operational protocol: the working level of step k is updated at window
    i with the outcome of window i-k-1 at step k, the most recent window whose step k has been
    observed at the origin of window i (consecutive windows are one step apart). delayed=False
    updates with window i's own outcome, i.e. the whole horizon one step later; this look-ahead
    is what the first v2 pass used and is kept only to quantify it. mask (N, H) False marks
    outcomes that never enter an update (corrupt target hours). The calibration scores stay
    fixed; only the level moves, so the issued interval is always a quantile of the
    calibration scores at the current working level.
    Returns the calibrated (N, H, Q) array (monotone across levels) and the level trajectory
    (N, n_pairs, H) actually used for each window.
    """
    N, H, Q = Pt.shape
    Pq = Pt.copy()
    alpha_tr = np.empty((N, len(pairs), H))
    ks = np.arange(H)
    for j, (lo, hi, alpha) in enumerate(pairs):
        _, Es = _offsets(Pc, Yc, lo, hi, alpha)
        nC = Es.shape[0]
        alph = np.full(H, alpha)
        out_lo, out_hi = np.empty((N, H)), np.empty((N, H))
        err = np.zeros((N, H))
        for i in range(N):
            if delayed:
                j0 = i - ks - 1                               # window whose step k is now observed
                ok = j0 >= 0
                j0c = np.maximum(j0, 0)
                if mask is not None:
                    ok &= mask[j0c, ks]
                alph = alph + gamma * (alpha - err[j0c, ks]) * ok
            a = np.clip(alph, 1e-3, 1 - 1e-3)
            kk = np.clip(np.ceil((1 - a) * (nC + 1)).astype(int) - 1, 0, nC - 1)
            off = Es[kk, ks]
            out_lo[i] = Pt[i, :, lo] - off
            out_hi[i] = Pt[i, :, hi] + off
            err[i] = ((Yt[i] < out_lo[i]) | (Yt[i] > out_hi[i])).astype(float)
            alpha_tr[i, j] = a
            if not delayed:
                upd = err[i] if mask is None else np.where(mask[i], err[i], alpha)
                alph = alph + gamma * (alpha - upd)
        Pq[:, :, lo], Pq[:, :, hi] = out_lo, out_hi
    return np.sort(Pq, axis=-1), alpha_tr


def score_q(Pq, Y, alpha=0.10, tag="", mask=None, lo=LO, hi=HI):
    """Scores of a (N, H, Q) quantile array against Y (N, H), over the unmasked entries:
    mean pinball over the levels, the quantile-weighted CRPS approximation, and the coverage /
    width / Winkler score of the (lo, hi) interval."""
    w = np.ones(Y.shape, bool) if mask is None else mask.astype(bool)
    pin_q = [float(np.maximum(q * (Y - Pq[:, :, j]), (q - 1) * (Y - Pq[:, :, j]))[w].mean())
             for j, q in enumerate(QUANTILES)]
    L_, U_, y = Pq[:, :, lo][w], Pq[:, :, hi][w], Y[w]
    cov = float(((y >= L_) & (y <= U_)).mean())
    return {
        f"pinball{tag}": float(np.mean(pin_q)),
        f"crps{tag}": qcrps_from_quantiles(pin_q, QUANTILES),
        f"picp{tag}": 100.0 * cov,
        f"ace{tag}": 100.0 * abs(cov - (1 - alpha)),
        f"mpiw{tag}": float((U_ - L_).mean()),
        f"winkler{tag}": winkler(L_, U_, y, alpha),
    }


def pair_coverage(Pq, Y, mask=None, pairs=PAIRS):
    """Empirical coverage (%) of every level pair's interval."""
    w = np.ones(Y.shape, bool) if mask is None else mask.astype(bool)
    out = {}
    for lo, hi, alpha in pairs:
        y = Y[w]
        out[int(round(100 * (1 - alpha)))] = 100.0 * float(((y >= Pq[:, :, lo][w]) & (y <= Pq[:, :, hi][w])).mean())
    return out


def run(a):
    dev = torch.device("cuda")
    exp.set_seed(a.seed)
    meta = json.load(open(os.path.join(HERE, "data", "proc", a.data + ".json")))
    C, tgt = meta["C"], meta["target_idx"]
    model = QuantileWrapper(a.model, a.L, a.H, C, len(QUANTILES),
                            json.loads(a.extra) if a.extra else None).to(dev)
    tr, va, te, trset = make_loaders(a.data, a.L, a.H, a.batch, model.basis_cfg,
                                     test_batch=a.batch)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.05)
    tag = f"{a.data}_{a.model}_L{a.L}_H{a.H}_s{a.seed}{a.tag}"
    print(f"[prob:{tag}] params={sum(p.numel() for p in model.parameters())/1e3:.0f}k", flush=True)

    best, bad, bsd, t0 = float("inf"), 0, None, time.time()
    for ep in range(a.epochs):
        model.train(); tl = nb = 0
        for x, y, Phi, Psi, tf, _ in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = pinball(model(x, Phi.to(dev), Psi.to(dev), tf.to(dev)), y, QUANTILES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += float(loss); nb += 1
        sch.step()
        model.eval(); vl = vn = 0
        with torch.no_grad():
            for x, y, Phi, Psi, tf, _ in va:
                x, y = x.to(dev), y.to(dev)
                vl += float(pinball(model(x, Phi.to(dev), Psi.to(dev), tf.to(dev)), y, QUANTILES)); vn += 1
        v = vl / max(vn, 1)
        print(f"   ep{ep:02d} train={tl/max(nb,1):.5f} val={v:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best - 1e-7:
            best, bad = v, 0
            bsd = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("   early stop", flush=True); break
    if bsd:
        model.load_state_dict(bsd)

    Pc, Yc = collect(model, va, dev, tgt, trset.scaler)      # calibration split
    Pt, Yt = collect(model, te, dev, tgt, trset.scaler)      # test split
    res = dict(dataset=a.data, model=a.model, H=a.H, L=a.L, seed=a.seed, lr=a.lr,
               quantiles=QUANTILES, val_pinball=best, train_time_s=round(time.time() - t0, 1))
    res.update(score(Pt, Yt, Pt[:, :, LO], Pt[:, :, HI], 0.10, "_raw"))
    lo, hi, _ = conformalise(Pc, Yc, Pt, 0.10)
    res.update(score(Pt, Yt, lo, hi, 0.10, "_cqr"))
    lo2, hi2 = conformalise_aci(Pc, Yc, Pt, Yt, 0.10, a.gamma)
    res.update(score(Pt, Yt, lo2, hi2, 0.10, "_aci"))
    # deterministic accuracy of the median head, for reference
    res["mae_median"] = float(np.abs(Pt[:, :, len(QUANTILES) // 2] - Yt).mean())
    res["wape_median"] = 100.0 * float(np.abs(Pt[:, :, len(QUANTILES) // 2] - Yt).sum() / np.abs(Yt).sum())
    with open(os.path.join(RES, tag + ".json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(RES, tag + "_q.npz"), P=Pt.astype(np.float32),
                        Y=Yt.astype(np.float32), lo=lo2.astype(np.float32), hi=hi2.astype(np.float32))
    print(f"[prob:{tag}] CRPS={res['crps_cqr']:.1f} PICP_aci={res['picp_aci']:.1f}% "
          f"MPIW={res['mpiw_aci']:.0f} Winkler={res['winkler_aci']:.0f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--extra", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    run(ap.parse_args())
