"""Unified training / evaluation harness. Every method sees exactly the same
windows, the same normalisation, the same optimiser family and the same budget."""
import argparse, json, os, time, math, random
import numpy as np
import torch
import torch.nn as nn

from data import make_loaders, basis_dim, slow_dim, list_datasets
from baselines import REGISTRY
from planet import PLANET
from planetx import PLANETX, degrade

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
ERR = os.path.join(RES, "errors")
os.makedirs(ERR, exist_ok=True)

BASIS_CFG = dict(n_day=6, n_week=3, n_year=0, trend=True, daytype=True,
                 relative=False, n_year_slow=3)

# per-model hyper-parameters (kept at the values published by each method's authors)
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
    "PLANET":        dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, patch=16, stride=8, dropout=0.2),
    "PLANETX":       dict(d_model=128, n_layers=3, n_heads=8, d_ff=256, patch=16, stride=8,
                          dropout=0.2, exo_d=48, exo_kernel=9),
}
# models whose forward path contains FFT / linear-solve kernels that must stay in fp32
FP32_ONLY = {"FITS", "Autoformer"}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def slow_order(meta, requested=3):
    """The climatological tier is only identifiable when the training split actually
    spans several annual cycles. With a single year, annual harmonics are confounded
    with the trend and extrapolate badly, so the tier degrades to level+trend."""
    tr0, tr1 = meta["splits"]["train"]
    steps_per_hour = {"1h": 1.0, "15min": 4.0}.get(meta.get("freq", "1h"), 1.0)
    years = (tr1 - tr0) / steps_per_hour / 8766.0
    if years < 2.0:
        return 0
    return min(requested, int(years))


def build_model(name, L, H, C, extra=None):
    cfg = dict(MODEL_CFG.get(name, {}))
    cfg.update(extra or {})
    if name in ("PLANET", "PLANETX"):
        bc = dict(BASIS_CFG); bc.update({k: v for k, v in cfg.items() if k in
                                         ("n_day", "n_week", "n_year", "trend", "daytype", "relative", "n_year_slow")})
        kf = basis_dim(bc["n_day"], bc["n_week"], bc["n_year"], bc["trend"], bc["daytype"])
        ks = slow_dim(bc["n_year_slow"])
        mcfg = {k: v for k, v in cfg.items() if k not in
                ("n_day", "n_week", "n_year", "trend", "daytype", "relative", "n_year_slow")}
        if name == "PLANETX":
            return PLANETX(L, H, C, kf, ks, target_idx=mcfg.pop("target_idx", -1), **mcfg), bc
        return PLANET(L, H, C, kf, ks, **mcfg), bc
    if name == "SeasonalNaive":
        cfg.setdefault("steps_per_day", 24)
    if name == "CalTST":
        # give it exactly the design matrix PLANET's solver sees
        cfg.setdefault("K", basis_dim(BASIS_CFG["n_day"], BASIS_CFG["n_week"],
                                      BASIS_CFG["n_year"], BASIS_CFG["trend"],
                                      BASIS_CFG["daytype"]))
    return REGISTRY[name](L=L, H=H, C=C, **cfg), None


def forward(model, x, y, Phi, Psi, tf, exo_cfg=None):
    """Calls the model, supplying the future exogenous block when it asks for one."""
    if not getattr(model, "needs_exo", False):
        return model(x, Phi, Psi, tf)
    exo = degrade(y, exo_cfg["channels"], exo_cfg["sd"], exo_cfg["mode"], x_past=x)
    return model(x, Phi, Psi, tf, exo)


def metrics_norm(pred, true):
    e = pred - true
    return float((e ** 2).mean()), float(e.abs().mean())


def evaluate(model, loader, dev, amp=False, want_arrays=False, target_idx=-1,
             scaler=None, collect_pred=0, ms=False, exo_cfg=None, steps_per_day=24):
    model.eval()
    n, s_mse, s_mae = 0, 0.0, 0.0
    target_idx = int(target_idx)            # negative indices break the ms slice below
    per_win, per_step_se, per_step_ae, per_chan_se = [], None, None, None
    den = dict(sse=0.0, sae=0.0, sape=0.0, say=0.0, n=0, peak_ae=[], peak_dt=[],
               peak_rel=[], ramp_ae=0.0, ramp_n=0)
    per_win_ae = []
    keep = []
    with torch.no_grad():
        for bi, (x, y, Phi, Psi, tf, idx) in enumerate(loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            Phi, Psi, tf = Phi.to(dev), Psi.to(dev), tf.to(dev)
            with torch.autocast("cuda", enabled=bool(amp)):
                p = forward(model, x, y, Phi, Psi, tf, exo_cfg)
            p = p.float()
            if ms:                      # supervise/score the load channel only
                ti = target_idx % p.shape[-1]
                p = p[:, :, ti:ti + 1]
                y = y[:, :, ti:ti + 1]
            e = p - y
            b = x.shape[0]
            s_mse += float((e ** 2).mean()) * b
            s_mae += float(e.abs().mean()) * b
            n += b
            per_win.append((e ** 2).mean(dim=(1, 2)).cpu().numpy())
            ss = (e ** 2).mean(dim=(0, 2)) * b
            aa = e.abs().mean(dim=(0, 2)) * b
            cc = (e ** 2).mean(dim=(0, 1)) * b
            per_step_se = ss if per_step_se is None else per_step_se + ss
            per_step_ae = aa if per_step_ae is None else per_step_ae + aa
            per_chan_se = cc if per_chan_se is None else per_chan_se + cc
            if scaler is not None:
                mu, sd = scaler
                mu_t = torch.as_tensor(mu[target_idx], device=dev)
                sd_t = torch.as_tensor(sd[target_idx], device=dev)
                ti = 0 if ms else target_idx
                pt = p[:, :, ti] * sd_t + mu_t
                yt = y[:, :, ti] * sd_t + mu_t
                d = pt - yt
                den["sse"] += float((d ** 2).sum()); den["sae"] += float(d.abs().sum())
                den["say"] += float(yt.abs().sum())
                den["sape"] += float((d.abs() / yt.abs().clamp_min(1e-6)).sum())
                den["n"] += d.numel()
                spd = steps_per_day
                if yt.shape[1] >= spd:
                    nd = yt.shape[1] // spd
                    yd = yt[:, :nd * spd].reshape(-1, nd, spd)
                    pd_ = pt[:, :nd * spd].reshape(-1, nd, spd)
                    yp, ypi = yd.max(-1); pp, ppi = pd_.max(-1)
                    den["peak_ae"].append((pp - yp).abs().flatten().cpu().numpy())
                    den["peak_rel"].append(((pp - yp).abs() / yp.abs().clamp_min(1e-6)).flatten().cpu().numpy())
                    den["peak_dt"].append((ppi - ypi).abs().float().flatten().cpu().numpy())
                per_win_ae.append(d.abs().mean(1).cpu().numpy())
                dr = (pt[:, 1:] - pt[:, :-1]) - (yt[:, 1:] - yt[:, :-1])
                den["ramp_ae"] += float(dr.abs().sum()); den["ramp_n"] += dr.numel()
            if collect_pred and (collect_pred < 0 or len(keep) < collect_pred):
                keep.append((idx.numpy(), p[:, :, 0 if ms else target_idx].cpu().numpy(),
                             y[:, :, 0 if ms else target_idx].cpu().numpy()))
    out = dict(mse=s_mse / n, mae=s_mae / n, n_windows=n)
    if want_arrays:
        out["_per_window"] = np.concatenate(per_win)
        out["_per_step_mse"] = (per_step_se / n).cpu().numpy()
        out["_per_step_mae"] = (per_step_ae / n).cpu().numpy()
        out["per_channel_mse"] = [round(float(v), 6) for v in (per_chan_se / n).cpu().numpy()]
        if per_win_ae:
            out["_per_window_ae"] = np.concatenate(per_win_ae)
    if scaler is not None and den["n"]:
        # WAPE (= sum|e| / sum|y|) is the scale-free metric of record for load forecasting;
        # plain MAPE is only reported where the target is strictly positive.
        out.update(rmse_orig=math.sqrt(den["sse"] / den["n"]),
                   mae_orig=den["sae"] / den["n"],
                   wape=100.0 * den["sae"] / max(den["say"], 1e-9),
                   mape=100.0 * den["sape"] / den["n"],
                   ramp_mae=den["ramp_ae"] / max(den["ramp_n"], 1))
        if den["peak_ae"]:
            out.update(peak_mae=float(np.concatenate(den["peak_ae"]).mean()),
                       peak_mape=100.0 * float(np.concatenate(den["peak_rel"]).mean()),
                       peak_time_err=float(np.concatenate(den["peak_dt"]).mean()))
    if collect_pred:
        out["_keep"] = keep
    return out


def run(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    extra = json.loads(args.extra) if args.extra else {}

    meta = json.load(open(os.path.join(HERE, "data", "proc", args.data + ".json")))
    C = meta["C"]
    if args.model in ("PLANET", "PLANETX") and "n_year_slow" not in extra:
        extra["n_year_slow"] = slow_order(meta, BASIS_CFG["n_year_slow"])
    probe, bc = build_model(args.model, args.L, args.H, C, extra)
    basis_cfg = bc if bc is not None else dict(BASIS_CFG)

    tr, va, te, trset = make_loaders(args.data, args.L, args.H, args.batch, basis_cfg,
                                     workers=args.workers,
                                     test_batch=args.test_batch or args.batch,
                                     train_stride=args.train_stride)
    model = probe.to(dev)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = getattr(model, "trainable", True)
    amp = bool(args.amp) and args.model not in FP32_ONLY and torch.cuda.is_available()
    ms, ti_ = bool(args.ms), meta["target_idx"]
    spd_ = {"1h": 24, "15min": 96}.get(meta.get("freq", "1h"), 24)
    exo_cfg = None
    if getattr(probe, "needs_exo", False):
        # degrade() works on the full channel block, so the noise scale must cover every
        # channel; the model selects its exogenous subset afterwards.
        exo_cfg = dict(mode=args.exo_mode, sd=None,
                       channels=list(enumerate(meta["channels"])))

    if exo_cfg is not None:
        exo_cfg["sd"] = trset.sd
    tag = f"{args.data}_{args.model}_L{args.L}_H{args.H}_s{args.seed}{args.tag}"
    print(f"[{tag}] C={C} params={nparam/1e3:.1f}k train={len(trset)} amp={amp} "
          f"setting={'MS' if ms else 'M'} dev={dev}", flush=True)

    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    if trainable:
        # the log-penalties act multiplicatively over orders of magnitude, so they get a
        # larger step size than the network weights; the climatological tier gets a ridge.
        named = dict(model.named_parameters())
        slow_p = [q for k, q in named.items() if k.endswith("beta_g")]
        lam_p = [q for k, q in named.items() if k.endswith("rho_lam") or k.endswith("weighter.rho")]
        ids = {id(q) for q in slow_p} | {id(q) for q in lam_p}
        rest = [q for q in named.values() if id(q) not in ids]
        groups = [{"params": rest, "weight_decay": args.wd, "lr": args.lr}]
        if slow_p:
            groups.append({"params": slow_p, "weight_decay": args.wd_slow, "lr": args.lr})
        if lam_p:
            groups.append({"params": lam_p, "weight_decay": 0.0,
                           "lr": min(args.lr * args.lr_mult_lam, 0.05)})
        opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
        scaler_amp = torch.amp.GradScaler("cuda", enabled=amp)
        crit = nn.MSELoss()
        bestsd, n_bad_steps = None, 0
        for ep in range(args.epochs):
            model.train(); tl, nb = 0.0, 0
            for x, y, Phi, Psi, tf, _ in tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                Phi, Psi, tf = Phi.to(dev), Psi.to(dev), tf.to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=bool(amp)):
                    out_ = forward(model, x, y, Phi, Psi, tf, exo_cfg)
                    if ms:
                        out_ = out_[:, :, ti_:ti_ + 1]
                        loss = crit(out_, y[:, :, ti_:ti_ + 1])
                    else:
                        loss = crit(out_, y)
                if not torch.isfinite(loss):
                    n_bad_steps += 1           # skip a diverged step rather than poison the weights
                    continue
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(opt); scaler_amp.update()
                tl += float(loss); nb += 1
            sch.step()
            v = evaluate(model, va, dev, amp, target_idx=ti_, ms=ms, exo_cfg=exo_cfg)
            hist.append(dict(epoch=ep, train=tl / max(nb, 1), val=v["mse"]))
            print(f"   ep{ep:02d} train={tl/max(nb,1):.5f} val={v['mse']:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if not np.isfinite(v["mse"]):
                print("   !! non-finite validation loss", flush=True)
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
        # non-parametric: "training" is choosing the seasonal period on the validation split
        best, bestp = float("inf"), model.period
        for p_ in model.candidates():
            model.period = p_
            v = evaluate(model, va, dev, amp, target_idx=ti_, ms=ms)["mse"]
            print(f"   period={p_:4d} val={v:.5f}", flush=True)
            if v < best:
                best, bestp = v, p_
        model.period = bestp
        hist.append(dict(epoch=0, train=float("nan"), val=best, period=bestp))
    else:
        best = evaluate(model, va, dev, amp, target_idx=ti_, ms=ms, exo_cfg=exo_cfg)["mse"]
    train_time = time.time() - t0

    if args.save_ckpt:
        os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(HERE, "checkpoints", f"{args.data}_{args.model}_L{args.L}_H{args.H}.pt"))

    ti = time.time()
    r = evaluate(model, te, dev, amp, want_arrays=True, target_idx=meta["target_idx"],
                 scaler=trset.scaler, collect_pred=args.collect, ms=ms, exo_cfg=exo_cfg,
                 steps_per_day=spd_)
    infer_time = time.time() - ti

    _arrs = dict(per_window=r.pop("_per_window"),
                 per_step_mse=r.pop("_per_step_mse"),
                 per_step_mae=r.pop("_per_step_mae"))
    if "_per_window_ae" in r:
        _arrs["per_window_ae"] = r.pop("_per_window_ae")
    np.savez_compressed(os.path.join(ERR, tag + ".npz"), **_arrs)
    keep = r.pop("_keep", None)
    if keep:
        np.savez_compressed(os.path.join(ERR, tag + "_pred.npz"),
                            idx=np.concatenate([k[0] for k in keep]),
                            pred=np.concatenate([k[1] for k in keep]),
                            true=np.concatenate([k[2] for k in keep]))

    out = dict(dataset=args.data, model=args.model, L=args.L, H=args.H, seed=args.seed,
               lr=args.lr, batch=args.batch, params=nparam, epochs_run=len(hist),
               n_year_slow=extra.get("n_year_slow"), train_stride=args.train_stride,
               val_mse=best, setting="MS" if ms else "M", exo_mode=getattr(args, "exo_mode", None) if exo_cfg else None, bad_steps=int(locals().get("n_bad_steps", 0)),
               train_time_s=round(train_time, 1),
               infer_time_s=round(infer_time, 2), amp=amp, tag=args.tag,
               history=hist, **{k: v for k, v in r.items()})
    with open(os.path.join(RES, tag + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] MSE={r['mse']:.4f} MAE={r['mae']:.4f} "
          f"WAPE={r.get('wape', float('nan')):.2f}% time={train_time:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--L", type=int, default=336)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--wd_slow", type=float, default=1e-2)
    ap.add_argument("--lr_mult_lam", type=float, default=50.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--test_batch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--collect", type=int, default=0)
    ap.add_argument("--save_ckpt", type=int, default=0)
    ap.add_argument("--ms", type=int, default=0)
    ap.add_argument("--train_stride", type=int, default=1)
    ap.add_argument("--exo_mode", default="perfect", choices=["perfect", "noisy", "persist"])
    ap.add_argument("--extra", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    run(ap.parse_args())
