"""v2 training / evaluation harness (V2_SPEC.md).

Same loop as v1 exp.py -- identical windows, normalisation, optimiser family, budget and
early stopping for every method -- plus:
  * information regimes R0 / R1 / R2 with the shared covariate block (data_v2.py);
  * the shared exogenous head for models without native covariate inputs (exo.py);
  * the official neuralforecast baselines (nf_models.py);
  * a size multiplier for the equal-budget tuning sweep, a local/UTC basis switch and a
    backbone switch for PLANET;
  * results under results_v2/.
"""
import argparse, json, os, time, math, random
import numpy as np
import torch
import torch.nn as nn

from data import slow_dim
from data_v2 import make_loaders_v2, REGIMES
from baselines import REGISTRY
from planet import PLANET
from exo import WithExo
from nf_models import NF_CLASSES, FUTURE_VIA_HEAD, build_nf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_v2")
ERR = os.path.join(RES, "errors")
os.makedirs(ERR, exist_ok=True)

# PLANET v2 basis: v1 harmonics, day-types extended with the national holiday, phase
# locked to local civil time. Which of trend / climatological tier / weighting / day-types
# survive is decided by the v2 backward-elimination ablation, not assumed (PLAN_v2 M1).
BASIS_CFG = dict(n_day=6, n_week=3, n_year=0, trend=True, daytype="weekend+holiday",
                 relative=False, n_year_slow=3)

MODEL_CFG = {
    "SeasonalNaive": dict(),
    "DLinear":       dict(kernel=25),
    "NLinear":       dict(),
    "RLinear":       dict(dropout=0.1),
    "FITS":          dict(base_period=24, n_harm=6),
    "PatchTST":      dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, patch=16, stride=8, dropout=0.2),
    "iTransformer":  dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, dropout=0.1),
    "TimeMixer":     dict(d_model=64, n_layers=2, n_scales=3, dropout=0.1),
    "NBEATS":        dict(n_blocks=9, width=256, depth=4),
    "LSTM":          dict(hidden=128, layers=2, dropout=0.1),
    "TCN":           dict(ch=64, levels=5, kernel=3, dropout=0.1),
    "Informer":      dict(d_model=256, n_heads=8, e_layers=2, d_layers=1, d_ff=512, dropout=0.1, factor=5),
    "Autoformer":    dict(d_model=256, n_heads=8, e_layers=2, d_layers=1, d_ff=512, dropout=0.1, factor=3),
    "CalTST":        dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, patch=16, stride=8, dropout=0.2),
    "PLANET":        dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, patch=16, stride=8, dropout=0.2,
                          n_blocks=9, width=256, depth=4),
    # PLANET-X: identical, plus the native covariate pathway (per-step width 4, 128-d embedding)
    "PLANETX":       dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, patch=16, stride=8, dropout=0.2,
                          n_blocks=9, width=256, depth=4, cov_width=4, d_cov=128),
}
# width knobs scaled by the size multiplier (0.5 / 1 / 2)
SIZE_KEYS = {
    "PatchTST": ["d_model", "d_ff"], "CalTST": ["d_model", "d_ff"],
    "PLANET": ["d_model", "d_ff", "width"], "PLANETX": ["d_model", "d_ff", "width"],
    "iTransformer": ["d_model", "d_ff"],
    "TimeMixer": ["d_model"], "NBEATS": ["width"], "LSTM": ["hidden"], "TCN": ["ch"],
    "Informer": ["d_model", "d_ff"], "Autoformer": ["d_model", "d_ff"],
}
NO_SIZE = {"SeasonalNaive", "DLinear", "NLinear", "RLinear", "FITS"}
# Channel-independent forecasters: every channel is encoded on its own, so under the MS
# setting (only the load channel is supervised and scored) their non-target outputs are
# dead compute. Training them on the target channel alone gives the identical target
# forecast at a fraction of the cost. Multivariate models (variates interact) keep all channels.
CHANNEL_INDEPENDENT = {"SeasonalNaive", "DLinear", "NLinear", "RLinear", "FITS", "PatchTST",
                       "NBEATS", "LSTM", "TCN", "TimeMixer", "CalTST", "PLANET", "PLANETX",
                       "TFT", "TiDE", "NHITS", "NBEATSx", "TimesNet", "TiDEh"}


class TargetOnly(nn.Module):
    """Run a channel-independent model on the target channel only and place its forecast
    back in the target slot of a (B, H, C) output (other slots are zero and never scored)."""
    needs_cov = True

    def __init__(self, base, C, target_idx):
        super().__init__()
        self.base, self.C, self.t = base, C, target_idx % C

    def forward(self, x, Phi=None, Psi=None, tf=None, cov=None):
        xt = x[:, :, self.t:self.t + 1]
        y = call(self.base, xt, Phi, Psi, tf, cov)                     # (B, H, 1[, Q])
        out = y.new_zeros(y.shape[:2] + (self.C,) + y.shape[3:])
        out[:, :, self.t] = y[:, :, 0]
        return out

    @property
    def trainable(self):
        return getattr(self.base, "trainable", True)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)
FP32_ONLY = {"FITS", "Autoformer"}
HEADS_ROUND = 8


_TG = {"t": 0.0, "paused": 0.0}


def thermal_guard(hi=86, lo=81, every=10.0, max_wait=7200):
    """Pause while the GPU sits at its thermal limit (Quadro RTX 4000: target 83 C, maximum
    operating 92 C). Polled at most every `every` seconds from the batch loops; the time spent
    paused is accumulated in _TG["paused"] and excluded from the reported training time."""
    import subprocess
    now = time.time()
    if now - _TG["t"] < every:
        return
    _TG["t"] = now

    def temp():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return -1
    t = temp()
    if t >= hi:
        t0 = time.time()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        while t > lo and time.time() - t0 < max_wait:
            time.sleep(3)
            t = temp()
        _TG["paused"] += time.time() - t0
        _TG["t"] = time.time()
        msg = f"[thermal] paused {time.time() - t0:.0f}s, now {t} C"
        print("   " + msg, flush=True)
        try:        # the pipeline filters stage output, so every pause is also logged here
            with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "logs", "thermal.log"), "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
        except Exception:
            pass


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def slow_order(meta, requested=3):
    tr0, tr1 = meta["splits"]["train"]
    sph = {"1h": 1.0, "15min": 4.0}.get(meta.get("freq", "1h"), 1.0)
    years = (tr1 - tr0) / sph / 8766.0
    return 0 if years < 2.0 else min(requested, int(years))


def sized(name, cfg, k):
    cfg = dict(cfg)
    for key in SIZE_KEYS.get(name, []):
        if key in cfg:
            v = max(8, int(round(cfg[key] * k)))
            if key == "d_model":
                v = max(HEADS_ROUND, (v // HEADS_ROUND) * HEADS_ROUND)
            cfg[key] = v
    return cfg


def build_model(name, L, H, C, E, K_fast, K_slow, ms, target_idx, size_k=1.0,
                backbone="patch", extra=None, seed=1):
    extra = dict(extra or {})
    if ms and name in CHANNEL_INDEPENDENT and C > 1:
        inner = build_model(name, L, H, 1, E, K_fast, K_slow, ms, 0, size_k, backbone, extra, seed)
        return TargetOnly(inner, C, target_idx)
    head_kw = {k: extra.pop(k) for k in ("exo_d", "exo_kernel") if k in extra}
    head_kw = {"d": head_kw.get("exo_d", 48), "kernel": head_kw.get("exo_kernel", 9)}
    if name in NF_CLASSES:
        m = build_nf(name, L, H, C, E, size_k=size_k, seed=seed, **extra)
        if E > 0 and name in FUTURE_VIA_HEAD:
            m = WithExo(m, E, L, H, C, target_idx, ms, **head_kw)
        return m
    cfg = sized(name, MODEL_CFG.get(name, {}), size_k)
    if name in ("PLANET", "PLANETX"):
        mcfg = {k: v for k, v in extra.items() if k not in BASIS_CFG}
        cfg.update(mcfg)
        m = PLANET(L, H, C, K_fast, K_slow, resid=backbone, cov_native=(name == "PLANETX"), E=E, **cfg)
    else:
        cfg.update(extra)
        if name == "SeasonalNaive":
            cfg.setdefault("steps_per_day", 24)
        if name == "CalTST":
            cfg["K"] = K_fast
        m = REGISTRY[name](L=L, H=H, C=C, **cfg)
    if E > 0 and getattr(m, "trainable", True) and not getattr(m, "native_cov", False):
        m = WithExo(m, E, L, H, C, target_idx, ms, **head_kw)
    return m


# --------------------------------------------------------------------- weather regimes
# AR(1) coefficient of the correlated-noise sensitivity ("noisy_ar"): e-folding time of 12 h
# for the hour-to-hour correlation of the weather-forecast error along the horizon, so that the
# noise is a smooth error path rather than white noise (whose daily mean nearly averages out).
AR_PHI = math.exp(-1.0 / 12.0)


def degrade_cov(cov, L, wcols, noise_z, mode, gen=None):
    """Turn the true future weather rows of the covariate block into what a forecaster
    would really have (V2_SPEC.md, regime R2): noise_z is the 24 h NWP-skill error of each
    weather column in z-units (data_v2.nwp_noise_phys / dataset sd), grown with the square
    root of lead time. Calendar columns are untouched. mode "noisy" draws the error
    independently at every hour (the headline proxy); "noisy_ar" draws an AR(1) error path
    along the horizon with the same marginal sd at every lead (sensitivity analysis)."""
    if mode == "perfect" or not wcols:
        return cov
    cov = cov.clone()
    H = cov.shape[1] - L
    idx = torch.as_tensor(wcols, device=cov.device)
    if mode == "persist":
        src = torch.arange(H, device=cov.device) % 24 + (L - 24)
        cov[:, L:, idx] = cov[:, src][:, :, idx]
        return cov
    lead = torch.arange(1, H + 1, device=cov.device, dtype=cov.dtype)
    scale = torch.sqrt(lead / 24.0).view(1, H, 1)
    s = torch.tensor(list(noise_z), device=cov.device, dtype=cov.dtype).view(1, 1, -1)
    noise = torch.randn((cov.shape[0], H, len(wcols)), device=cov.device, dtype=cov.dtype,
                        generator=gen)
    if mode == "noisy_ar":
        phi, sq = AR_PHI, math.sqrt(1.0 - AR_PHI ** 2)
        e = torch.empty_like(noise)
        e[:, 0] = noise[:, 0]
        for k in range(1, H):                      # unit marginal variance at every lead
            e[:, k] = phi * e[:, k - 1] + sq * noise[:, k]
        noise = e
    cov[:, L:, idx] = cov[:, L:, idx] + noise * s * scale
    return cov


def call(model, x, Phi, Psi, tf, cov):
    if getattr(model, "needs_cov", False):
        return model(x, Phi, Psi, tf, cov)
    return model(x, Phi, Psi, tf)


def evaluate(model, loader, dev, amp, deg, target_idx, ms, want_arrays=False, scaler=None,
             collect_pred=0, steps_per_day=24):
    model.eval()
    n, s_mse, s_mae = 0, 0.0, 0.0
    per_win, per_win_ae, per_step_se, per_step_ae, per_chan_se, keep = [], [], None, None, None, []
    den = dict(sse=0.0, sae=0.0, sape=0.0, say=0.0, n=0, peak_ae=[], peak_dt=[], peak_rel=[],
               ramp_ae=0.0, ramp_n=0)
    ti = int(target_idx)
    with torch.no_grad():
        for x, y, Phi, Psi, tf, cov, idx in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            Phi, Psi, tf, cov = Phi.to(dev), Psi.to(dev), tf.to(dev), cov.to(dev)
            cov = deg(cov)
            with torch.autocast("cuda", enabled=bool(amp)):
                p = call(model, x, Phi, Psi, tf, cov)
            p = p.float()
            if ms:
                tj = ti % p.shape[-1]
                p, y = p[:, :, tj:tj + 1], y[:, :, tj:tj + 1]
            e = p - y
            b = x.shape[0]
            s_mse += float((e ** 2).mean()) * b; s_mae += float(e.abs().mean()) * b; n += b
            per_win.append((e ** 2).mean(dim=(1, 2)).cpu().numpy())
            ss, aa, cc = (e ** 2).mean(dim=(0, 2)) * b, e.abs().mean(dim=(0, 2)) * b, (e ** 2).mean(dim=(0, 1)) * b
            per_step_se = ss if per_step_se is None else per_step_se + ss
            per_step_ae = aa if per_step_ae is None else per_step_ae + aa
            per_chan_se = cc if per_chan_se is None else per_chan_se + cc
            if scaler is not None:
                mu, sd = scaler
                mu_t = torch.as_tensor(mu[ti], device=dev); sd_t = torch.as_tensor(sd[ti], device=dev)
                tj = 0 if ms else ti
                pt, yt = p[:, :, tj] * sd_t + mu_t, y[:, :, tj] * sd_t + mu_t
                d = pt - yt
                den["sse"] += float((d ** 2).sum()); den["sae"] += float(d.abs().sum())
                den["say"] += float(yt.abs().sum())
                den["sape"] += float((d.abs() / yt.abs().clamp_min(1e-6)).sum()); den["n"] += d.numel()
                if yt.shape[1] >= steps_per_day:
                    nd = yt.shape[1] // steps_per_day
                    yd = yt[:, :nd * steps_per_day].reshape(-1, nd, steps_per_day)
                    pd_ = pt[:, :nd * steps_per_day].reshape(-1, nd, steps_per_day)
                    yp, ypi = yd.max(-1); pp, ppi = pd_.max(-1)
                    den["peak_ae"].append((pp - yp).abs().flatten().cpu().numpy())
                    den["peak_rel"].append(((pp - yp).abs() / yp.abs().clamp_min(1e-6)).flatten().cpu().numpy())
                    den["peak_dt"].append((ppi - ypi).abs().float().flatten().cpu().numpy())
                per_win_ae.append(d.abs().mean(1).cpu().numpy())
                dr = (pt[:, 1:] - pt[:, :-1]) - (yt[:, 1:] - yt[:, :-1])
                den["ramp_ae"] += float(dr.abs().sum()); den["ramp_n"] += dr.numel()
            if collect_pred and (collect_pred < 0 or len(keep) < collect_pred):
                keep.append((idx.numpy(), p[:, :, 0 if ms else ti].cpu().numpy(),
                             y[:, :, 0 if ms else ti].cpu().numpy()))
    out = dict(mse=s_mse / n, mae=s_mae / n, n_windows=n)
    if want_arrays:
        out["_per_window"] = np.concatenate(per_win)
        out["_per_step_mse"] = (per_step_se / n).cpu().numpy()
        out["_per_step_mae"] = (per_step_ae / n).cpu().numpy()
        out["per_channel_mse"] = [round(float(v), 6) for v in (per_chan_se / n).cpu().numpy()]
        if per_win_ae:
            out["_per_window_ae"] = np.concatenate(per_win_ae)
    if scaler is not None and den["n"]:
        out.update(rmse_orig=math.sqrt(den["sse"] / den["n"]), mae_orig=den["sae"] / den["n"],
                   wape=100.0 * den["sae"] / max(den["say"], 1e-9), mape=100.0 * den["sape"] / den["n"],
                   ramp_mae=den["ramp_ae"] / max(den["ramp_n"], 1))
        if den["peak_ae"]:
            out.update(peak_mae=float(np.concatenate(den["peak_ae"]).mean()),
                       peak_mape=100.0 * float(np.concatenate(den["peak_rel"]).mean()),
                       peak_time_err=float(np.concatenate(den["peak_dt"]).mean()))
    if collect_pred:
        out["_keep"] = keep
    return out


def run(args):
    _TG["paused"] = 0.0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    extra = json.loads(args.extra) if args.extra else {}
    meta = json.load(open(os.path.join(HERE, "data", "proc_v2", args.data + ".json")))
    C, ti_ = meta["C"], meta["target_idx"]
    ms = bool(args.ms)
    spd_ = {"1h": 24, "15min": 96}.get(meta.get("freq", "1h"), 24)

    basis_cfg = dict(BASIS_CFG)
    basis_cfg.update({k: v for k, v in extra.items() if k in BASIS_CFG})
    if "n_year_slow" not in extra:
        basis_cfg["n_year_slow"] = slow_order(meta, BASIS_CFG["n_year_slow"])
    needs_basis = args.model in ("PLANET", "PLANETX", "CalTST")
    splits = None
    if getattr(args, "splits", ""):                  # "tr0,tr1,va1,te1" row indices (monthly refit)
        s_ = [int(x) for x in str(args.splits).split(",")]
        splits = {"train": [s_[0], s_[1]], "val": [s_[1], s_[2]], "test": [s_[2], s_[3]]}
    tr, va, te, trset = make_loaders_v2(args.data, args.L, args.H, args.batch,
                                        basis_cfg if needs_basis else None, args.regime,
                                        args.basis_time, workers=args.workers,
                                        test_batch=args.test_batch or args.batch,
                                        train_stride=args.train_stride,
                                        train_frac=args.train_frac, train_first=bool(args.train_first),
                                        eval_data=args.eval_data or None, splits=splits)
    E = trset.E
    K_fast = trset.K if needs_basis else 0
    K_slow = slow_dim(basis_cfg["n_year_slow"])
    model = build_model(args.model, args.L, args.H, C, E, K_fast, K_slow, ms, ti_,
                        args.size, args.backbone, extra, seed=args.seed).to(dev)
    set_seed(args.seed)      # library constructors may reseed the global RNG (neuralforecast does)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = getattr(model, "trainable", True)
    amp = bool(args.amp) and args.model not in FP32_ONLY and torch.cuda.is_available()

    gen = torch.Generator(device=dev); gen.manual_seed(args.seed + 7)
    wcols, wnz = trset.cov_weather_cols, trset.cov_weather_noise_z
    deg = lambda cov: degrade_cov(cov, args.L, wcols, wnz, args.exo_mode, gen)

    tag = f"{args.data}_{args.model}_L{args.L}_H{args.H}_s{args.seed}{args.tag}"
    print(f"[{tag}] C={C} E={E} K={K_fast} params={nparam/1e3:.1f}k train={len(trset)} "
          f"regime={args.regime} size={args.size:g} basis={args.basis_time} "
          f"backbone={args.backbone if args.model=='PLANET' else '-'} amp={amp} dev={dev}", flush=True)

    hist, best, bad, t0, n_bad_steps = [], float("inf"), 0, time.time(), 0
    if trainable:
        named = dict(model.named_parameters())
        slow_p = [q for k, q in named.items() if k.endswith("beta_g")]
        lam_p = [q for k, q in named.items() if k.endswith("rho_lam") or k.endswith("weighter.rho")]
        ids = {id(q) for q in slow_p} | {id(q) for q in lam_p}
        rest = [q for q in named.values() if id(q) not in ids]
        groups = [{"params": rest, "weight_decay": args.wd, "lr": args.lr}]
        if slow_p:
            groups.append({"params": slow_p, "weight_decay": args.wd_slow, "lr": args.lr})
        if lam_p:
            groups.append({"params": lam_p, "weight_decay": 0.0, "lr": min(args.lr * args.lr_mult_lam, 0.05)})
        opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
        scaler_amp = torch.amp.GradScaler("cuda", enabled=amp)
        crit = nn.MSELoss()
        bestsd = None
        for ep in range(args.epochs):
            model.train(); tl, nb = 0.0, 0
            for x, y, Phi, Psi, tf, cov, _ in tr:
                thermal_guard()
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                Phi, Psi, tf, cov = Phi.to(dev), Psi.to(dev), tf.to(dev), cov.to(dev)
                cov = deg(cov)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=bool(amp)):
                    out_ = call(model, x, Phi, Psi, tf, cov)
                    if ms:
                        loss = crit(out_[:, :, ti_:ti_ + 1], y[:, :, ti_:ti_ + 1])
                    else:
                        loss = crit(out_, y)
                if not torch.isfinite(loss):
                    n_bad_steps += 1; continue
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(opt); scaler_amp.update()
                tl += float(loss); nb += 1
            sch.step()
            v = evaluate(model, va, dev, amp, deg, ti_, ms)
            hist.append(dict(epoch=ep, train=tl / max(nb, 1), val=v["mse"]))
            print(f"   ep{ep:02d} train={tl/max(nb,1):.5f} val={v['mse']:.5f} ({time.time()-t0:.0f}s)", flush=True)
            if np.isfinite(v["mse"]) and v["mse"] < best - 1e-6:
                best, bad = v["mse"], 0
                bestsd = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print("   early stop", flush=True); break
        if bestsd is not None:
            model.load_state_dict(bestsd)
    elif hasattr(model, "candidates"):
        best, bestp = float("inf"), model.period
        for p_ in model.candidates():
            model.period = p_
            v = evaluate(model, va, dev, amp, deg, ti_, ms)["mse"]
            if v < best:
                best, bestp = v, p_
        model.period = bestp
        hist.append(dict(epoch=0, train=float("nan"), val=best, period=bestp))
    else:
        best = evaluate(model, va, dev, amp, deg, ti_, ms)["mse"]
    train_time = time.time() - t0 - _TG["paused"]

    if args.save_ckpt:
        os.makedirs(os.path.join(HERE, "checkpoints_v2"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(HERE, "checkpoints_v2", tag + ".pt"))

    ti = time.time()
    r = evaluate(model, te, dev, amp, deg, ti_, ms, want_arrays=True, scaler=trset.scaler,
                 collect_pred=args.collect, steps_per_day=spd_)
    infer_time = time.time() - ti
    arrs = dict(per_window=r.pop("_per_window"), per_step_mse=r.pop("_per_step_mse"),
                per_step_mae=r.pop("_per_step_mae"))
    if "_per_window_ae" in r:
        arrs["per_window_ae"] = r.pop("_per_window_ae")
    np.savez_compressed(os.path.join(ERR, tag + ".npz"), **arrs)
    keep = r.pop("_keep", None)
    if keep:
        np.savez_compressed(os.path.join(ERR, tag + "_pred.npz"),
                            idx=np.concatenate([k[0] for k in keep]),
                            pred=np.concatenate([k[1] for k in keep]),
                            true=np.concatenate([k[2] for k in keep]))
    out = dict(dataset=args.data, model=args.model, L=args.L, H=args.H, seed=args.seed, lr=args.lr,
               batch=args.batch, params=nparam, epochs_run=len(hist), n_year_slow=basis_cfg.get("n_year_slow"),
               train_stride=args.train_stride, val_mse=best, setting="MS" if ms else "M",
               regime=args.regime, size=args.size, basis_time=args.basis_time,
               exo_mode=args.exo_mode if E > 0 and wcols else None,
               backbone=args.backbone if args.model in ("PLANET", "PLANETX") else None, E=E, K=K_fast,
               train_frac=args.train_frac, train_first=bool(args.train_first),
               eval_data=args.eval_data or None, splits=(getattr(args, "splits", "") or None),
               bad_steps=int(n_bad_steps), train_time_s=round(train_time, 1),
               infer_time_s=round(infer_time, 2), amp=amp, tag=args.tag, extra=extra, history=hist,
               **{k: v for k, v in r.items()})
    with open(os.path.join(RES, tag + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] MSE={r['mse']:.4f} MAE={r['mae']:.4f} WAPE={r.get('wape', float('nan')):.2f}% "
          f"time={train_time:.0f}s", flush=True)
    return out


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--L", type=int, default=336); ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2021); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--wd_slow", type=float, default=1e-2)
    ap.add_argument("--lr_mult_lam", type=float, default=50.0)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--test_batch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--collect", type=int, default=0); ap.add_argument("--save_ckpt", type=int, default=0)
    ap.add_argument("--ms", type=int, default=1); ap.add_argument("--train_stride", type=int, default=5)
    ap.add_argument("--regime", default="R2", choices=list(REGIMES))
    ap.add_argument("--exo_mode", default="noisy", choices=["perfect", "noisy", "persist", "noisy_ar"])
    ap.add_argument("--basis_time", default="local", choices=["local", "utc"])
    ap.add_argument("--size", type=float, default=1.0)
    ap.add_argument("--backbone", default="patch", choices=["patch", "basis"])
    ap.add_argument("--extra", type=str, default=""); ap.add_argument("--tag", type=str, default="")
    # mechanism experiments (PLAN_v2 E1-E3): fraction of training windows kept (from the end of
    # the split, or from its start with --train_first 1), and evaluation on another dataset
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--train_first", type=int, default=0)
    ap.add_argument("--eval_data", type=str, default="")
    ap.add_argument("--splits", type=str, default="", help="row-index split override tr0,tr1,va1,te1 (monthly refit)")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
