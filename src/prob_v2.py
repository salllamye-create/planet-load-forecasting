"""Probabilistic load forecasting, v2: the v1 recipe (pinball training on a fixed quantile
set, conformalised quantile regression on validation, adaptive conformal inference online
over the test split) on the v2 harness -- every regime, every headline model, the shared
covariate block and head. Outputs results_v2/prob/<tag>.json and <tag>_q.npz.

Quantile heads: v1-signature models are widened to H*Q outputs (v1 QuantileWrapper); PLANET
emits Q outputs natively; neuralforecast models use their own MQLoss head. Monotonicity across
levels is imposed by construction (median plus non-negative increments).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from data import slow_dim
from data_v2 import make_loaders_v2
from baselines import REGISTRY
from planet import PLANET
from exo import WithExo
from nf_models import NF_CLASSES, FUTURE_VIA_HEAD, NFWrap, scaled_cfg
import exp_v2 as exp
from prob import (QUANTILES, LO, HI, PAIRS, pinball, qcrps_from_quantiles, winkler,
                  conformalise_q, conformalise_aci_q, score_q, pair_coverage)
from data_v2 import bad_target_mask

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_v2", "prob")
os.makedirs(RES, exist_ok=True)


class QHead(nn.Module):
    """Sorted quantiles from a raw (B,H,C,Q) block: median + cumulative softplus increments."""
    def forward(self, o):
        q = o.shape[-1]
        med = o[..., q // 2: q // 2 + 1]
        dn = -torch.flip(torch.cumsum(torch.flip(nn.functional.softplus(o[..., :q // 2]), (-1,)), -1), (-1,))
        up = torch.cumsum(nn.functional.softplus(o[..., q // 2 + 1:]), -1)
        return torch.cat([med + dn, med, med + up], -1)


class QuantileModel(nn.Module):
    """Any point forecaster turned into a sorted-quantile forecaster on the v2 harness.

    Under the MS setting channel-independent models are built for the target channel only
    (identical forecast, a fraction of the compute); the output is then (B, H, 1, Q) and
    `out_t` = 0. The shared exogenous head is applied after the quantile reshape, to every
    quantile alike (a covariate shifts the whole predictive distribution).
    """
    needs_cov = True

    def __init__(self, name, L, H, C, E, K_fast, K_slow, ms, target_idx, size_k, backbone, extra, seed):
        super().__init__()
        self.H, self.Q, self.L = H, len(QUANTILES), L
        self.ci = bool(ms) and name in exp.CHANNEL_INDEPENDENT and C > 1
        self.t = target_idx % C
        Cm = 1 if self.ci else C
        self.C, self.out_t = Cm, (0 if self.ci else self.t)
        self.sort = QHead()
        self.nf = name in NF_CLASSES
        self.widened = False
        if name in ("PLANET", "PLANETX"):
            cfg = exp.sized(name, exp.MODEL_CFG[name], size_k)
            cfg.update({k: v for k, v in extra.items() if k not in exp.BASIS_CFG})
            self.m = PLANET(L, H, Cm, K_fast, K_slow, resid=backbone, n_quant=self.Q,
                            cov_native=(name == "PLANETX"), E=E, **cfg)
        elif self.nf:
            from neuralforecast.losses.pytorch import MQLoss
            assert name in ("TFT", "TiDE", "NHITS", "NBEATSx", "TimesNet"), \
                "probabilistic v2 covers the univariate library models (native multi-quantile head)"
            cfg = scaled_cfg(name, size_k); cfg.update(extra)
            self.m = NFWrap(name, L, H, Cm, E, seed=seed, keep_q=True,
                            loss=MQLoss(quantiles=list(QUANTILES)), **cfg)
        else:
            cfg = exp.sized(name, exp.MODEL_CFG.get(name, {}), size_k); cfg.update(extra)
            if name == "CalTST":
                # the calendar tokens need the true H; widen the output head instead
                cfg["K"] = K_fast
                self.m = REGISTRY[name](L=L, H=H, C=Cm, out_mult=self.Q, **cfg)
            else:
                self.m = REGISTRY[name](L=L, H=H * self.Q, C=Cm, **cfg)
            self.widened = True
        from exo import ExoHead
        # the shared head only where exp_v2 attaches it: models without a native known-future
        # interface (library models read the block natively, except TimesNet/TimeXer)
        native_future = (self.nf and name not in FUTURE_VIA_HEAD) or getattr(self.m, "native_cov", False)
        self.head = ExoHead(E, L, H, n_out=1 if ms else Cm) \
            if (E > 0 and getattr(self.m, "trainable", True) and not native_future) else None
        self.ms = ms

    def forward(self, x, Phi=None, Psi=None, tf=None, cov=None):
        if self.ci:
            x = x[:, :, self.t:self.t + 1]
        o = exp.call(self.m, x, Phi, Psi, tf, cov)
        B = o.shape[0]
        if self.widened:                                   # (B, H*Q, C) -> (B, H, C, Q)
            o = o.reshape(B, self.Q, self.H, self.C).permute(0, 2, 3, 1)
        if self.head is not None and cov is not None:
            corr = self.head(cov[:, :self.L], cov[:, self.L:])         # (B, H, 1|C)
            if self.ms:
                o = o.clone(); o[:, :, self.out_t, :] = o[:, :, self.out_t, :] + corr[:, :, :1]
            else:
                o = o + corr.unsqueeze(-1)
        return self.sort(o)


def collect(model, loader, dev, deg, target_idx, scaler, ms):
    model.eval(); P, Y = [], []
    mu, sd = scaler[0][target_idx], scaler[1][target_idx]
    ot = model.out_t
    with torch.no_grad():
        for x, y, Phi, Psi, tf, cov, _ in loader:
            x, y = x.to(dev), y.to(dev)
            cov = deg(cov.to(dev))
            p = model(x, Phi.to(dev), Psi.to(dev), tf.to(dev), cov).float()
            P.append((p[:, :, ot, :] * sd + mu).cpu().numpy())
            Y.append((y[:, :, target_idx] * sd + mu).cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def run(a):
    dev = torch.device("cuda")
    exp.set_seed(a.seed)
    extra = json.loads(a.extra) if a.extra else {}
    meta = json.load(open(os.path.join(HERE, "data", "proc_v2", a.data + ".json")))
    C, tgt = meta["C"], meta["target_idx"]
    ms = bool(a.ms)
    basis_cfg = dict(exp.BASIS_CFG); basis_cfg.update({k: v for k, v in extra.items() if k in exp.BASIS_CFG})
    basis_cfg["n_year_slow"] = extra.get("n_year_slow", exp.slow_order(meta, exp.BASIS_CFG["n_year_slow"]))
    needs_basis = a.model in ("PLANET", "PLANETX", "CalTST")
    tr, va, te, trset = make_loaders_v2(a.data, a.L, a.H, a.batch, basis_cfg if needs_basis else None,
                                        a.regime, a.basis_time, test_batch=a.batch, train_stride=a.train_stride)
    E, K_fast, K_slow = trset.E, (trset.K if needs_basis else 0), slow_dim(basis_cfg["n_year_slow"])
    model = QuantileModel(a.model, a.L, a.H, C, E, K_fast, K_slow, ms, tgt, a.size, a.backbone, extra, a.seed).to(dev)
    exp.set_seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed + 7)
    deg = lambda cov: exp.degrade_cov(cov, a.L, trset.cov_weather_cols, trset.cov_weather_noise_z,
                                      a.exo_mode, gen)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.05)
    tag = f"{a.data}_{a.model}_L{a.L}_H{a.H}_s{a.seed}{a.tag}"
    if os.path.exists(os.path.join(RES, tag + ".json")) and not getattr(a, "force", 0):
        print(f"[prob:{tag}] exists, skipped", flush=True)
        return None
    print(f"[prob:{tag}] E={E} params={sum(p.numel() for p in model.parameters())/1e3:.0f}k regime={a.regime}", flush=True)

    def loss_on(x, y, Phi, Psi, tf, cov):
        p = model(x, Phi, Psi, tf, cov)
        if ms:
            p, y = p[:, :, model.out_t:model.out_t + 1], y[:, :, tgt:tgt + 1]
        return pinball(p, y, QUANTILES)

    best, bad, bsd, t0 = float("inf"), 0, None, time.time()
    for ep in range(a.epochs):
        model.train(); tl = nb = 0
        for x, y, Phi, Psi, tf, cov, _ in tr:
            exp.thermal_guard()
            x, y = x.to(dev), y.to(dev)
            cov = deg(cov.to(dev))
            opt.zero_grad(set_to_none=True)
            loss = loss_on(x, y, Phi.to(dev), Psi.to(dev), tf.to(dev), cov)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += float(loss); nb += 1
        sch.step()
        model.eval(); vl = vn = 0
        with torch.no_grad():
            for x, y, Phi, Psi, tf, cov, _ in va:
                x, y = x.to(dev), y.to(dev)
                cov = deg(cov.to(dev))
                vl += float(loss_on(x, y, Phi.to(dev), Psi.to(dev), tf.to(dev), cov)); vn += 1
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

    Pc, Yc = collect(model, va, dev, deg, tgt, trset.scaler, ms)
    Pt, Yt = collect(model, te, dev, deg, tgt, trset.scaler, ms)
    res = dict(dataset=a.data, model=a.model, H=a.H, L=a.L, seed=a.seed, lr=a.lr, regime=a.regime,
               size=a.size, basis_time=a.basis_time, exo_mode=a.exo_mode, quantiles=QUANTILES,
               val_pinball=best, train_time_s=round(time.time() - t0, 1), tag=a.tag)
    # corrupt target hours (data_v2.BAD_HOURS) are excluded from every score and never update
    # the adaptive level; the mask is saved with the arrays
    mask = bad_target_mask(meta, a.data, a.L, a.H, n_windows=Pt.shape[0])
    res.update(score_q(Pt, Yt, 0.10, "_raw", mask))
    Pq_c = conformalise_q(Pc, Yc, Pt)                                  # CQR, every level pair
    res.update(score_q(Pq_c, Yt, 0.10, "_cqr", mask))
    Pq_a, atr = conformalise_aci_q(Pc, Yc, Pt, Yt, a.gamma, delayed=True, mask=mask)   # operational ACI
    res.update(score_q(Pq_a, Yt, 0.10, "_aci", mask))
    Pq_0, _ = conformalise_aci_q(Pc, Yc, Pt, Yt, a.gamma, delayed=False, mask=mask)    # look-ahead, for the record
    res.update(score_q(Pq_0, Yt, 0.10, "_aci0", mask))
    for nm, Pq in (("raw", Pt), ("cqr", Pq_c), ("aci", Pq_a)):
        for lev, c in pair_coverage(Pq, Yt, mask).items():
            res[f"picp{lev}_{nm}"] = c
    res["n_masked"] = int((~mask).sum())
    res["aci"] = dict(gamma=a.gamma, delayed=True, pairs=[list(p) for p in PAIRS],
                      n_calibration=int(Pc.shape[0]))
    med = Pt[:, :, len(QUANTILES) // 2]
    res["mae_median"] = float(np.abs(med - Yt)[mask].mean())
    res["wape_median"] = 100.0 * float(np.abs(med - Yt)[mask].sum() / np.abs(Yt[mask]).sum())
    json.dump(res, open(os.path.join(RES, tag + ".json"), "w"), indent=2)
    f32 = np.float32
    np.savez_compressed(os.path.join(RES, tag + "_q.npz"), P=Pt.astype(f32), Y=Yt.astype(f32),
                        Pc=Pc.astype(f32), Yc=Yc.astype(f32), Pq_cqr=Pq_c.astype(f32),
                        Pq_aci=Pq_a.astype(f32), alpha_aci=atr.astype(f32), mask=mask,
                        lo=Pq_a[:, :, LO].astype(f32), hi=Pq_a[:, :, HI].astype(f32))
    print(f"[prob:{tag}] CRPS={res['crps_cqr']:.1f} PICP_aci={res['picp_aci']:.1f}% "
          f"(look-ahead {res['picp_aci0']:.1f}%) MPIW={res['mpiw_aci']:.0f} "
          f"Winkler={res['winkler_aci']:.0f} masked={res['n_masked']}", flush=True)
    return res


if __name__ == "__main__":
    ap = exp.parser()
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--auto", type=int, default=0,
                    help="take lr, width, backbone and lookback from the tuning/lookback sweeps (run_all_v2)")
    ap.add_argument("--force", type=int, default=0)
    a = ap.parse_args()
    if a.auto:
        import run_all_v2 as r
        a.lr, a.size = r.best_tune(a.data, a.model)
        a.backbone = r.best_bb(a.model)
        a.L = r.best_look(a.data, a.model, a.H)
        print(f"[prob] tuned settings: lr={a.lr:g} size={a.size:g} backbone={a.backbone} L={a.L}", flush=True)
    run(a)
