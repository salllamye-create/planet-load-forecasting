"""Experiment driver: builds the job matrix, runs it sequentially, resumes on restart."""
import argparse, json, os, sys, time, traceback, gc
import torch

import exp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True)

ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
GRID = ["OPSD-DE", "OPSD-FR", "OPSD-GB"]
ALL_MODELS = ["SeasonalNaive", "NLinear", "DLinear", "RLinear", "FITS", "NBEATS",
              "LSTM", "TCN", "Informer", "Autoformer", "TimeMixer", "iTransformer",
              "PatchTST", "CalTST", "PLANET"]

# batch size / precision per dataset (ECL has 321 channels and dominates memory)
# ETT and ECL follow the published multivariate (M) protocol so the numbers stay
# comparable with the literature. The national grids are a load-forecasting task:
# every channel is available as input, the load channel is what gets predicted and
# scored (the "MS" setting of the Informer/Autoformer codebase).
DS = {
    # Window strides subsample the *training* windows only (never validation or test), and
    # are applied identically to every method, to fit a single-GPU budget.
    #
    # The stride must be COPRIME with the number of steps per day. A stride of 2 on hourly
    # data lets training windows start only at even hours, while validation and test use all
    # 24 -- so the alignment between a model's patch grid and the wall clock is systematically
    # different at train and test time. Strides of 5 (hourly, 15-min) and 7 (Electricity)
    # cycle through every start hour.
    "ETTh1": dict(batch=64, amp=0, ms=0),
    "ETTh2": dict(batch=64, amp=0, ms=0),
    "ETTm1": dict(batch=64, amp=0, ms=0, train_stride=5),
    "ETTm2": dict(batch=64, amp=0, ms=0, train_stride=5),
    "OPSD-DE": dict(batch=64, amp=0, ms=1, train_stride=5),
    "OPSD-FR": dict(batch=64, amp=0, ms=1, train_stride=5),
    "OPSD-GB": dict(batch=64, amp=0, ms=1, train_stride=5),
    "ECL": dict(batch=16, amp=1, test_batch=16, ms=0, train_stride=7),
}
LR_GRID = [1e-4, 5e-4, 1e-3]


class A:  # lightweight argparse.Namespace stand-in
    def __init__(self, **kw):
        d = dict(data=None, model=None, L=336, H=96, seed=2021, lr=1e-4, wd=0.0,
                 batch=64, test_batch=0, epochs=15, patience=3, workers=0, amp=0,
                 collect=0, save_ckpt=0, ms=0, wd_slow=1e-2, lr_mult_lam=50.0,
                 exo_mode="perfect", train_stride=1,
                 extra="", tag="")
        d.update(kw)
        self.__dict__.update(d)

    def key(self):
        return f"{self.data}_{self.model}_L{self.L}_H{self.H}_s{self.seed}{self.tag}"


def done(a):
    return os.path.exists(os.path.join(RES, a.key() + ".json"))


def best_lr(data, model, H=96, fallback=1e-4):
    """Best validation lr for this (dataset, model, horizon).

    The search is run at the same horizon and the same epoch/patience budget as the final
    run: a cosine schedule tuned over 5 epochs is a different schedule from one annealed
    over 10, so an lr chosen on the short proxy is not the lr that is best on the real run.
    Falls back to the same model's choice at another horizon, then to a reference dataset.
    """
    best, blr = float("inf"), None
    for lr in LR_GRID:
        p = os.path.join(RES, f"{data}_{model}_L336_H{H}_s2021_lr{lr:g}.json")
        if os.path.exists(p):
            v = json.load(open(p))["val_mse"]
            if v < best:
                best, blr = v, lr
    if blr is not None:
        return blr
    for altH in (96, 24, 336, 192, 720):      # same dataset, nearest available horizon
        for lr in LR_GRID:
            p = os.path.join(RES, f"{data}_{model}_L336_H{altH}_s2021_lr{lr:g}.json")
            if os.path.exists(p):
                v = json.load(open(p))["val_mse"]
                if v < best:
                    best, blr = v, lr
        if blr is not None:
            return blr
    # otherwise reuse the lr chosen on the reference dataset of the same family
    ref = "ETTh1" if data.startswith("ETT") else ("ECL" if data == "ECL" else "OPSD-DE")
    for lr in LR_GRID:
        p = os.path.join(RES, f"{ref}_{model}_L336_H{H}_s2021_lr{lr:g}.json")
        if os.path.exists(p):
            v = json.load(open(p))["val_mse"]
            if v < best:
                best, blr = v, lr
    return blr if blr is not None else fallback


# ------------------------------------------------------------------ job builders
def jobs_lr(datasets, models, horizons=(96,)):
    out = []
    for d in datasets:
        for m in models:
            if m == "SeasonalNaive":
                continue
            for H in horizons:
                for lr in LR_GRID:
                    out.append(A(data=d, model=m, H=H, lr=lr, tag=f"_lr{lr:g}", **DS[d]))
    return out


SPD = {"ETTm1": 96, "ETTm2": 96}          # steps per day (15-minute data)


def jobs_main(datasets, models, horizons, seeds=(2021,)):
    out = []
    for d in datasets:
        for m in models:
            lr = best_lr(d, m)
            ex = json.dumps({"steps_per_day": SPD[d]}) if (m == "SeasonalNaive" and d in SPD) else ""
            for H in horizons:
                for s in seeds:
                    # keep full target-channel forecasts on the national grids: the
                    # operational tables and the COVID case study are built from them
                    col = -1 if (d.startswith("OPSD") and s == seeds[0]) else 0
                    ck = 1 if (m == "PLANET" and d == "OPSD-DE" and s == seeds[0]) else 0
                    out.append(A(data=d, model=m, H=H, lr=lr, seed=s, collect=col,
                                 save_ckpt=ck, extra=ex, **DS[d]))
    return out


def jobs_shortlook(datasets, models=("Informer", "Autoformer"), horizons=(96, 192, 336, 720)):
    """Informer/Autoformer were designed for L=96; give them their native lookback too."""
    out = []
    for d in datasets:
        for m in models:
            lr = best_lr(d, m)
            for H in horizons:
                out.append(A(data=d, model=m, L=96, H=H, lr=lr, tag="_L96", **DS[d]))
    return out


ABLATIONS = {
    "full":          {},
    "no_pl":         {"use_pl": False},                       # residual backbone only
    "no_resid":      {"use_resid": False},                    # closed-form tiers only
    "no_slow":       {"use_slow": False},                     # drop the climatological tier
    "relative":      {"relative": True},                      # relative-time basis (phase not locked)
    "uniform_w":     {"learn_w": False},                      # unweighted least squares
    "no_beta_token": {"beta_token": False},                   # no seasonal-state conditioning
    "no_daytype":    {"daytype": False},                      # drop day-type interactions
    "no_trend":      {"trend": False},                        # drop the local trend column
}


def jobs_ablation(datasets, horizons, seeds=(2021,)):
    out = []
    for d in datasets:
        lr = best_lr(d, "PLANET")
        for nm, ex in ABLATIONS.items():
            if nm == "full":
                continue
            for H in horizons:
                for s in seeds:
                    out.append(A(data=d, model="PLANET", H=H, lr=lr, seed=s,
                                 extra=json.dumps(ex), tag=f"_abl-{nm}", **DS[d]))
    return out


def jobs_lookback(datasets, models=("PLANET", "PatchTST", "DLinear"),
                  looks=(96, 192, 336, 512, 720), H=96):
    out = []
    for d in datasets:
        for m in models:
            lr = best_lr(d, m)
            for L in looks:
                out.append(A(data=d, model=m, L=L, H=H, lr=lr, tag="_look", **DS[d]))
    return out


# ------------------------------------------------------------------ runner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lr", "main", "shortlook", "ablation", "lookback"])
    ap.add_argument("--datasets", default="ETT")
    ap.add_argument("--models", default="all")
    ap.add_argument("--horizons", default="96,192,336,720")
    ap.add_argument("--seeds", default="2021")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lrs", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    dmap = {"ETT": ETT, "GRID": GRID, "ECL": ["ECL"], "ALL": ETT + GRID + ["ECL"]}
    datasets = dmap.get(a.datasets, a.datasets.split(","))
    models = ALL_MODELS if a.models == "all" else a.models.split(",")
    horizons = [int(x) for x in a.horizons.split(",")]
    seeds = tuple(int(x) for x in a.seeds.split(","))

    global LR_GRID
    if a.lrs:
        LR_GRID = [float(x) for x in a.lrs.split(",")]
    if a.stage == "lr":         js = jobs_lr(datasets, models, horizons)
    elif a.stage == "main":     js = jobs_main(datasets, models, horizons, seeds)
    elif a.stage == "shortlook":js = jobs_shortlook(datasets, horizons=horizons)
    elif a.stage == "ablation": js = jobs_ablation(datasets, horizons, seeds)
    else:                       js = jobs_lookback(datasets, H=horizons[0])

    for j in js:
        j.epochs = a.epochs
        j.patience = a.patience
    todo = [j for j in js if not done(j)]
    print(f"stage={a.stage} jobs={len(js)} todo={len(todo)}", flush=True)
    if a.dry:
        for j in todo:
            print("  ", j.key(), f"lr={j.lr:g}")
        return

    logf = open(os.path.join(LOGS, f"{a.stage}_{int(time.time())}.log"), "w")
    t0 = time.time()
    for i, j in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {j.key()} lr={j.lr:g} "
              f"({(time.time()-t0)/60:.1f} min elapsed) ===", flush=True)
        try:
            r = exp.run(j)
            logf.write(json.dumps({k: v for k, v in r.items() if k != "history"}) + "\n")
            logf.flush()
        except torch.cuda.OutOfMemoryError:
            print("   !! OOM -- retrying at half batch", flush=True)
            torch.cuda.empty_cache(); gc.collect()
            try:
                j.batch = max(2, j.batch // 2); j.test_batch = max(2, (j.test_batch or j.batch) // 2)
                r = exp.run(j)
                logf.write(json.dumps({k: v for k, v in r.items() if k != "history"}) + "\n"); logf.flush()
            except Exception as e:
                print("   !! FAILED", e, flush=True); traceback.print_exc()
        except Exception as e:
            print("   !! FAILED", e, flush=True); traceback.print_exc()
        finally:
            torch.cuda.empty_cache(); gc.collect()
    print(f"\nstage {a.stage} complete in {(time.time()-t0)/60:.1f} min", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
