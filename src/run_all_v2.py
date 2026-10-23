"""v2 experiment driver (PLAN_v2.md sections 6-9): builds the job matrix, runs it serially,
resumes on restart. Every selection (learning rate, size, lookback) is made on validation
loss, and the choice is recorded by the analysis, never by the driver.

Stages
  tune      lr x size grid, regime R2, H=96, one seed            -> _v2-R2-k{size}-lr{lr}-tune
  lookback  L in {168,336,512,720} at the tuned (lr,size), every model, per system and horizon -> _v2-R2-look
  main      tuned (lr,size,L); every regime x 3 seeds              -> _v2-{regime}
  ablation  PLANET component ablation, 3 seeds                     -> _v2-R2-abl-{variant}
  mech      learning curves, temporal drift, cross-grid transfer, 3 seeds -> _v2-R2-lc{frac} / -drift{frac} / -xfer-{src}
Records made under a superseded selection (lr, width, backbone or lookback differing from the
current one) are moved to results_v2/stale/ before a stage runs, so the analysis only ever sees
the protocol's selections.
"""
import argparse, json, os, sys, time, traceback, gc, glob
import torch

import exp_v2 as exp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_v2")
LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True); os.makedirs(RES, exist_ok=True)

GRID = ["OPSD-DE", "OPSD-FR", "OPSD-GB", "OPSD-ES", "OPSD-IT", "OPSD-NL", "OPSD-PL", "OPSD-BE"]
ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
V1_MODELS = ["SeasonalNaive", "NLinear", "DLinear", "RLinear", "FITS", "NBEATS", "LSTM", "TCN",
             "Informer", "Autoformer", "TimeMixer", "iTransformer", "PatchTST", "CalTST", "PLANET", "PLANETX"]
NF_MODELS = ["TFT", "TiDE", "NHITS", "NBEATSx", "TimeXer", "TSMixerx", "XLinear"]
ALL_MODELS = V1_MODELS + NF_MODELS
HEADLINE = ["NBEATS", "PatchTST", "CalTST", "PLANET", "PLANETX", "TFT", "TiDE", "NHITS", "TimeXer", "DLinear"]

DS = {d: dict(batch=64, amp=0, ms=1, train_stride=5) for d in GRID}
DS.update({"ETTh1": dict(batch=64, amp=0, ms=0, train_stride=1),
           "ETTh2": dict(batch=64, amp=0, ms=0, train_stride=1),
           "ETTm1": dict(batch=64, amp=0, ms=0, train_stride=5),
           "ETTm2": dict(batch=64, amp=0, ms=0, train_stride=5),
           "ECL":   dict(batch=16, amp=1, test_batch=16, ms=0, train_stride=7)})
LR_GRID = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
SIZES = [0.5, 1.0, 2.0]
REF = ["OPSD-DE", "OPSD-FR"]              # size sweep runs here; the choice transfers
# The five most expensive models (5-30x a PatchTST run). They receive the full tuning budget
# like every other model; the set only drives the memory caps below and the cost macros.
EXPENSIVE = {"TFT", "TimeXer", "Informer", "Autoformer"}
LOOKS = [168, 336, 512, 720]
SEEDS = (2021, 2022, 2023)
# Variants that reuse another model's tuning (identical architecture, one pathway switched):
# TiDEh is TiDE with the covariate block routed through the shared head, run at TiDE's selected
# learning rate, width and lookback so that only the pathway differs.
TUNE_ALIAS = {"TiDEh": "TiDE"}
# Data variants that reuse the base system's selections (lr, width, backbone, lookback): the
# multi-point weather corpora differ from the base system in the weather channels only.
DATA_ALIAS = {"OPSD-FR-mp": "OPSD-FR", "OPSD-ES-mp": "OPSD-ES", "OPSD-IT-mp": "OPSD-IT"}
for _k, _v in DATA_ALIAS.items():
    DS[_k] = dict(DS[_v])


class A:
    def __init__(self, **kw):
        d = dict(data=None, model=None, L=336, H=96, seed=2021, lr=1e-3, wd=0.0, wd_slow=1e-2,
                 lr_mult_lam=50.0, batch=64, test_batch=0, epochs=30, patience=5, workers=0,
                 amp=0, collect=0, save_ckpt=0, ms=1, train_stride=5, regime="R2",
                 exo_mode="noisy", basis_time="local", size=1.0, backbone="patch", extra="",
                 tag="", train_frac=1.0, train_first=0, eval_data="", splits="")
        d.update(kw); self.__dict__.update(d)

    def key(self):
        return f"{self.data}_{self.model}_L{self.L}_H{self.H}_s{self.seed}{self.tag}"

    def stage_tag_has_config(self):
        return "-tune" in self.tag


FAILED = os.path.join(RES, "failed")
os.makedirs(FAILED, exist_ok=True)


def _nfail(a):
    p = os.path.join(FAILED, a.key() + ".n")
    if not os.path.exists(p):
        return 0
    try:
        return int(open(p).readline().strip() or 0)
    except ValueError:
        return 0


def _batch_hint(a):
    """Batch size a previous process found necessary for this job (written on OOM)."""
    p = os.path.join(FAILED, a.key() + ".batch")
    try:
        return int(open(p).read().strip()) if os.path.exists(p) else None
    except ValueError:
        return None


def _write_hint(a, batch):
    with open(os.path.join(FAILED, a.key() + ".batch"), "w") as f:
        f.write(f"{batch}\n")


def _cuda_ok():
    """False when the CUDA context is wedged (any call raises after an asynchronous error)."""
    try:
        torch.cuda.synchronize()
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _record_fail(a, err):
    n = _nfail(a) + 1
    with open(os.path.join(FAILED, a.key() + ".n"), "w") as f:
        f.write(f"{n}\n{err}\n")
    return n


def done(a):
    """Finished, or failed in two separate process lives (a deterministic failure such as a
    model that does not fit the GPU at any batch size); the paper reports such cells as --."""
    return os.path.exists(os.path.join(RES, a.key() + ".json")) or _nfail(a) >= 2


STALE = os.path.join(RES, "stale")


def _prune_stale(jobs):
    """Move records made under a superseded selection out of the analysis' sight: a record at
    the job's key whose lr / width / backbone differ from the job's, or a record for the same
    (data, model, H, seed, tag) at a different lookback. Returns the number of files moved."""
    import shutil
    moved = 0
    for j in jobs:
        if j.stage_tag_has_config():
            continue                                     # tune tags encode lr and width
        pat = os.path.join(RES, f"{j.data}_{j.model}_L*_H{j.H}_s{j.seed}{j.tag}.json")
        for pth in glob.glob(pat):
            try:
                rec = json.load(open(pth))
            except Exception:
                continue
            same_cfg = (abs(float(rec.get("lr", j.lr)) - j.lr) < 1e-12 and
                        abs(float(rec.get("size", j.size)) - j.size) < 1e-9 and
                        (j.model not in ("PLANET", "PLANETX") or (rec.get("backbone") or "patch") == j.backbone))
            # the lookback sweep legitimately holds every L; elsewhere a different L is stale
            same_L = j.tag.endswith("-look") or int(rec.get("L", j.L)) == j.L
            if same_cfg and same_L:
                continue
            os.makedirs(STALE, exist_ok=True)
            stem = os.path.basename(pth)[:-5]
            for f in [pth] + glob.glob(os.path.join(RES, "errors", stem + "*.npz")):
                try:
                    shutil.move(f, os.path.join(STALE, os.path.basename(f)))
                    moved += 1
                except Exception:
                    pass
    return moved


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _cleanup():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _val(path):
    try:
        v = json.load(open(path))["val_mse"]
        return v if v == v else float("inf")
    except Exception:
        return float("inf")


BACKBONES = ("patch", "basis")            # PLANET's residual network: chosen on validation


def _tune_runs(data, model, H=96):
    model = TUNE_ALIAS.get(model, model)
    out = {}
    for p in glob.glob(os.path.join(RES, f"{data}_{model}_L336_H{H}_s2021_v2-R2-k*-lr*-tune.json")):
        r = json.load(open(p))
        out[(float(r["size"]), float(r["lr"]), r.get("backbone") or "patch")] = _val(p)
    return out


def best_size_bb(model, H=96):
    """(size, backbone) with the lowest mean over the reference systems of the best-lr
    validation loss. The backbone dimension is only searched for PLANET."""
    if model in exp.NO_SIZE:
        return (1.0, "patch")
    score = {}
    for d in REF:
        for (k, lr, bb), v in _tune_runs(d, model, H).items():
            score.setdefault((k, bb), {}).setdefault(d, []).append(v)
    if not score:
        return (1.0, "patch")
    means = {key: sum(min(vs) for vs in per.values()) / len(per) for key, per in score.items()}
    return min(means, key=means.get)


def best_size(model, H=96):
    return best_size_bb(model, H)[0]


def best_bb(model, H=96):
    return best_size_bb(model, H)[1] if model in ("PLANET", "PLANETX") else "patch"


def best_tune(data, model, H=96):
    """(lr, size): size (and, for PLANET, backbone) chosen on the reference systems, learning
    rate on this system's own lr sweep at that configuration (fallback: DE, then lr 1e-3)."""
    k, bb = best_size_bb(model, H)
    for d in (DATA_ALIAS.get(data, data), "OPSD-DE"):
        runs = {lr: v for (kk, lr, b), v in _tune_runs(d, model, H).items() if kk == k and b == bb}
        if runs:
            return (min(runs, key=runs.get), k)
    return (1e-3, k)


def best_tune_regime(data, model, regime, H=96):
    """Learning rate selected on this system's lr sweep under `regime` (the per-regime selection
    check of the revision) at the size and backbone chosen under R2. None if that sweep does not
    exist or selects the R2 rate: the headline record of the cell is then the record."""
    k, bb = best_size_bb(model, H)
    lr_r2, _ = best_tune(data, model, H)
    runs = {}
    pat = f"{DATA_ALIAS.get(data, data)}_{TUNE_ALIAS.get(model, model)}_L336_H{H}_s2021_v2-{regime}-k*-lr*-tune.json"
    for p in glob.glob(os.path.join(RES, pat)):
        r = json.load(open(p))
        if float(r["size"]) == k and (r.get("backbone") or "patch") == bb:
            runs[float(r["lr"])] = _val(p)
    if not runs:
        return None
    lr = min(runs, key=runs.get)
    return None if lr == lr_r2 else lr


def best_look(data, model, H, lr=None, size=None):
    """Lookback with the lowest validation loss in the lookback sweep of this system, else of
    the reference system (the sweep is run on DE for the headline models and transferred);
    models without a sweep keep the protocol lookback 336."""
    model = TUNE_ALIAS.get(model, model)
    data = DATA_ALIAS.get(data, data)
    for d in ((data, "OPSD-DE") if str(data).startswith("OPSD") else (data,)):
        best, arg = float("inf"), None
        for L in LOOKS:
            p = os.path.join(RES, f"{d}_{model}_L{L}_H{H}_s2021_v2-R2-look.json")
            if os.path.exists(p):
                v = _val(p)
                if v < best:
                    best, arg = v, L
        if arg is not None:
            return arg
    return 336


def jobs_tune(datasets, models, H=96, regime="R2"):
    """Reference systems: full lr x size grid. Other systems: lr grid at the size selected on
    the reference systems. Every model, the expensive ones included. regime != R2 runs the
    learning-rate grid under another information regime (the per-regime selection check;
    size and backbone stay those selected under R2)."""
    out = []
    for d in datasets:
        for m in models:
            if m == "SeasonalNaive":
                continue
            if m in exp.NO_SIZE:
                sizes = [1.0]
            elif d in REF and regime == "R2":
                sizes = SIZES
            else:
                sizes = [best_size(m, H)]
            if m in ("PLANET", "PLANETX"):
                bbs = BACKBONES if (d in REF and regime == "R2") else (best_bb(m, H),)
            else:
                bbs = (None,)
            for bb in bbs:
                for k in sizes:
                    for lr in LR_GRID:
                        tag = f"_v2-{regime}-k{k:g}-lr{lr:g}" + (f"-bb{bb}" if bb else "") + "-tune"
                        out.append(A(data=d, model=m, H=H, lr=lr, size=k, backbone=bb or "patch",
                                     regime=regime, tag=tag, **DS[d]))
    return out


def jobs_lookback(datasets, models, horizons):
    out = []
    for d in datasets:
        for m in models:
            lr, k = best_tune(d, m)
            for H in horizons:
                for L in LOOKS:
                    out.append(A(data=d, model=m, L=L, H=H, lr=lr, size=k, backbone=best_bb(m),
                                 tag="_v2-R2-look", **DS[d]))
    return out


def jobs_main(datasets, models, horizons, regimes=("R2", "R0", "R1"), seeds=SEEDS, lr_regime="R2"):
    out = []
    for d in datasets:
        for m in models:
            lr, k = best_tune(d, m)
            for H in horizons:
                L = best_look(d, m, H, lr, k)
                for rg in regimes:
                    lr_rg = lr
                    if lr_regime == "own" and rg != "R2":      # per-regime learning rate (revision check)
                        lr_rg = best_tune_regime(d, m, rg)
                        if lr_rg is None:                       # same rate as under R2: the headline record stands
                            continue
                    for s in seeds:
                        col = -1 if (d.startswith("OPSD") and rg == "R2" and (s == seeds[0] or H == 48)) else 0
                        out.append(A(data=d, model=m, L=L, H=H, lr=lr_rg, size=k, seed=s, regime=rg,
                                     backbone=best_bb(m), collect=col, tag=f"_v2-{rg}", **DS[d]))
    return out


ABLATIONS = {
    "no_pl":         {"use_pl": False},
    "no_resid":      {"use_resid": False},
    "no_slow":       {"use_slow": False},
    "no_trend":      {"trend": False},
    "daytype_wk":    {"daytype": "weekend"},           # holiday day-type removed
    "no_daytype":    {"daytype": False},
    "uniform_w":     {"learn_w": False},
    "no_beta_token": {"beta_token": False},
    "relative":      {"relative": True},
    "basis_film":    {"beta_cond": "film"},            # only meaningful with --backbone basis
    "lean":          {"use_slow": False, "trend": False, "daytype": "weekend+holiday", "learn_w": False},
}


def jobs_ablation(datasets, horizons, seeds=SEEDS, backbones=("patch", "basis"), variants=None):
    """variants: restrict to these names (full / utc / the ABLATIONS keys); None = all."""
    out = []
    want = (lambda nm: True) if not variants else (lambda nm: nm in variants)
    for d in datasets:
        lr, k = best_tune(d, "PLANET")
        for bb in backbones:
            for H in horizons:
                L = best_look(d, "PLANET", H, lr, k)
                for s in seeds:
                    # the full model on each backbone is the ablation reference
                    if want("full"):
                        out.append(A(data=d, model="PLANET", L=L, H=H, lr=lr, size=k, seed=s,
                                     backbone=bb, tag=f"_v2-R2-abl-full-{bb}", **DS[d]))
                    if want("utc"):
                        out.append(A(data=d, model="PLANET", L=L, H=H, lr=lr, size=k, seed=s,
                                     backbone=bb, basis_time="utc", tag=f"_v2-R2-abl-utc-{bb}", **DS[d]))
                    for nm, ex in ABLATIONS.items():
                        if nm == "basis_film" and bb != "basis":
                            continue
                        if not want(nm):
                            continue
                        out.append(A(data=d, model="PLANET", L=L, H=H, lr=lr, size=k, seed=s,
                                     backbone=bb, extra=json.dumps(ex),
                                     tag=f"_v2-R2-abl-{nm}-{bb}", **DS[d]))
    return out


def jobs_mech(datasets, models, H=96, fracs=(0.1, 0.25, 0.5), seeds=SEEDS):
    out = []
    for d in datasets:
        for m in models:
            lr, k = best_tune(d, m)
            L = best_look(d, m, H, lr, k)
            for s in seeds:
                bb = best_bb(m)
                for f in fracs:                              # E1 learning curves (recent windows)
                    out.append(A(data=d, model=m, L=L, H=H, lr=lr, size=k, seed=s, train_frac=f,
                                 backbone=bb, tag=f"_v2-R2-lc{f:g}", **DS[d]))
                    out.append(A(data=d, model=m, L=L, H=H, lr=lr, size=k, seed=s, train_frac=f,
                                 train_first=1, backbone=bb, tag=f"_v2-R2-drift{f:g}", **DS[d]))   # E3 drift
                for src in datasets:                         # E2 zero-shot transfer
                    if src == d:
                        continue
                    out.append(A(data=src, model=m, L=L, H=H, lr=lr, size=k, seed=s, eval_data=d,
                                 backbone=bb, tag=f"_v2-R2-xfer-{d}", **DS[src]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "lookback", "main", "ablation", "mech"])
    ap.add_argument("--datasets", default="GRID")
    ap.add_argument("--models", default="all")
    ap.add_argument("--horizons", default="24,48,96,336")   # 48 h feeds the day-ahead protocol
    ap.add_argument("--seeds", default="2021,2022,2023")
    ap.add_argument("--regimes", default="R2,R0,R1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to every job tag, e.g. -ep30 for the budget check")
    ap.add_argument("--exclude", default="", help="comma list of models to leave out of this invocation (scheduling only)")
    ap.add_argument("--exo_mode", default="noisy", choices=["perfect", "noisy", "persist", "noisy_ar"],
                    help="weather degradation of the R2 block (noisy_ar: AR(1) sensitivity; use with --suffix)")
    ap.add_argument("--lr_regime", default="R2", choices=["R2", "own"],
                    help="main stage: learning rate selected under R2 (headline) or under each regime's own "
                         "sweep, for the R0/R1 cells whose selection differs (use with --suffix=-ownlr)")
    ap.add_argument("--variants", default="", help="ablation stage: comma list of variants to run (default all)")
    ap.add_argument("--backbones", default="patch,basis", help="ablation stage: residual backbones")
    a = ap.parse_args()
    dmap = {"GRID": GRID, "GRID3": GRID[:3], "ETT": ETT, "ECL": ["ECL"]}
    datasets = dmap.get(a.datasets, a.datasets.split(","))
    mmap = {"all": ALL_MODELS, "v1": V1_MODELS, "nf": NF_MODELS, "headline": HEADLINE}
    models = mmap.get(a.models, a.models.split(","))
    models = [m for m in models if m not in set(x for x in a.exclude.split(",") if x)]
    horizons = [int(x) for x in a.horizons.split(",")]
    seeds = tuple(int(x) for x in a.seeds.split(","))
    regs = tuple(a.regimes.split(","))
    if a.stage == "tune":       js = jobs_tune(datasets, models, horizons[0] if len(horizons) == 1 else 96,
                                               regime=regs[0] if len(regs) == 1 else "R2")
    elif a.stage == "lookback": js = jobs_lookback(datasets, models, horizons)
    elif a.stage == "main":     js = jobs_main(datasets, models, horizons, regimes=regs, seeds=seeds,
                                              lr_regime=a.lr_regime)
    elif a.stage == "ablation": js = jobs_ablation(datasets, horizons, seeds=seeds,
                                                   backbones=tuple(a.backbones.split(",")),
                                                   variants=[v for v in a.variants.split(",") if v] or None)
    else:                       js = jobs_mech(datasets, models, horizons[0], seeds=seeds)
    for j in js:
        j.epochs, j.patience = a.epochs, a.patience
        j.tag += a.suffix
        j.exo_mode = a.exo_mode
        # the attention-heavy library models exceed 8 GB at batch 64 once L >= 512 or H = 336
        # (the failure surfaces as a cuBLAS allocation error that wedges the CUDA context);
        # start them at batch 16 there, as the OOM handler would end up doing anyway
        if j.model in EXPENSIVE and (j.L >= 512 or j.H >= 336):
            j.batch = min(j.batch, 16); j.test_batch = min(j.test_batch or 64, 16)
        if j.model == "TimesNet":                      # 2-D FFT convolutions: 8 GB overflows at 64
            j.batch = min(j.batch, 32); j.test_batch = min(j.test_batch or 64, 32)
        hint = _batch_hint(j)
        if hint:
            j.batch = min(j.batch, hint); j.test_batch = min(j.test_batch or 64, hint)
    n_stale = 0 if a.dry else _prune_stale(js)          # a dry run must have no side effect
    todo = [j for j in js if not done(j)]
    print(f"stage={a.stage} jobs={len(js)} todo={len(todo)} stale_moved={n_stale}", flush=True)
    if a.dry:
        for j in todo[:40]:
            print("  ", j.key(), f"lr={j.lr:g} size={j.size:g}")
        if len(todo) > 40:
            print(f"   ... {len(todo)-40} more")
        return
    logf = open(os.path.join(LOGS, f"v2_{a.stage}_{int(time.time())}.log"), "w")
    t0, consecutive = time.time(), 0
    for i, j in enumerate(todo):
        print(f"\n=== [{i+1}/{len(todo)}] {j.key()} lr={j.lr:g} size={j.size:g} "
              f"({(time.time()-t0)/60:.1f} min elapsed) ===", flush=True)
        attempts, ok, err = 0, False, ""
        while not ok:
            try:
                r = exp.run(j)
                logf.write(json.dumps({k: v for k, v in r.items() if k != "history"}) + "\n"); logf.flush()
                ok = True
            except Exception as e:
                _cleanup()
                err = repr(e)[:300]
                if _is_oom(e):
                    if j.batch <= 2:
                        print("   !! FAILED at batch 2:", err, flush=True)
                        _record_fail(j, err); consecutive += 1
                        break
                    nb = max(2, j.batch // 2)
                    _write_hint(j, nb)
                    j.batch, j.test_batch = nb, max(2, (j.test_batch or 64) // 2)
                    if isinstance(e, torch.cuda.OutOfMemoryError) and attempts < 4 and _cuda_ok():
                        attempts += 1
                        print(f"   !! OOM -- retrying at batch {j.batch}", flush=True)
                        continue
                    # an asynchronous CUDA error leaves the context unusable: hand the job back to
                    # a fresh process at the reduced batch (no failure is counted)
                    print(f"   !! CUDA context unusable after OOM -- restarting the stage (batch hint {nb})", flush=True)
                    logf.close(); sys.exit(3)
                print("   !! FAILED", err, flush=True); traceback.print_exc()
                _record_fail(j, err)
                consecutive += 1
                if not _cuda_ok():
                    print("   !! CUDA context unusable -- restarting the stage", flush=True)
                    logf.close(); sys.exit(3)
                break
        _cleanup()
        if ok:
            consecutive = 0
        elif consecutive >= 2:
            print("   !! two consecutive failures -- exiting for a clean restart", flush=True)
            logf.close(); sys.exit(3)
    print(f"\nstage {a.stage} complete in {(time.time()-t0)/60:.1f} min", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
