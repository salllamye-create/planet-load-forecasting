"""Cross-model forecast combinations on the headline grid at seed 2021, the seed whose test
predictions are stored (run_all_v2 collects them for the first seed of every headline cell):
equal-weight means of the member forecasts, scored on the test windows of the headline regime
and compared with the single members at the same seed.

  Comb2     PLANET + TiDE
  CombTop3  the three headline models with the lowest validation MSE in the cell (chosen on
            validation, never on test)
  CombAll   every headline model with stored predictions in the cell

Writes results_v2/combo.json: per system x horizon the WAPE / MSE of each combination, of
every single model at seed 2021 and of the best-by-validation single model, the members, and
the Diebold-Mariano statistic of each combination against TiDE and PLANET on the per-window
mean absolute error (stationary-bootstrap standard error, expected block length max(n^1/3, H);
stat < 0: the combination has the lower loss). The windows that cover a corrupt hour are
dropped exactly as paper_numbers drops them (data_v2.bad_window_idx). CPU only.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_v2 import bad_window_idx          # noqa: E402
from stats import dm_bootstrap              # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(HERE, "data", "proc_v2")
RES = os.path.join(HERE, "results_v2")
ERR = os.path.join(RES, "errors")
GRID = ["OPSD-DE", "OPSD-FR", "OPSD-GB", "OPSD-ES", "OPSD-IT", "OPSD-NL", "OPSD-PL", "OPSD-BE"]
HEADLINE = ["NBEATS", "PatchTST", "CalTST", "PLANET", "PLANETX", "TFT", "TiDE", "NHITS", "TimeXer", "DLinear"]
COMBOS = {"Comb2": ["PLANET", "TiDE"], "CombAll": HEADLINE}
SEED, REGIME, HS, B = 2021, "R2", (24, 96, 336), 1000


def wape(p, y):
    return 100.0 * float(np.abs(p - y).sum() / max(np.abs(y).sum(), 1e-9))


def main():
    out = {}
    for d in GRID:
        z = np.load(os.path.join(PROC, d + ".npz"))
        meta = json.load(open(os.path.join(PROC, d + ".json"), encoding="utf-8"))
        a, b = meta["splits"]["train"]
        col = z["data"][a:b, meta["target_idx"]].astype(np.float64)
        mu, sd = float(col.mean()), float(col.std() or 1.0)
        for H in HS:
            preds, vals, idx0, truth = {}, {}, None, None
            for m in HEADLINE:
                fs = glob.glob(os.path.join(ERR, f"{d}_{m}_L*_H{H}_s{SEED}_v2-{REGIME}_pred.npz"))
                if len(fs) != 1:
                    continue
                zz = np.load(fs[0])
                stem = os.path.basename(fs[0])[:-len("_pred.npz")]
                rec = json.load(open(os.path.join(RES, stem + ".json"), encoding="utf-8"))
                # the stored idx is the first lookback row, which depends on the model's L; the
                # windows are aligned on their first target row idx + L
                tgt0 = zz["idx"].astype(np.int64) + int(rec["L"])
                if idx0 is None:
                    idx0, truth = tgt0, zz["true"].astype(np.float64) * sd + mu
                elif not np.array_equal(tgt0, idx0):
                    print(f"  [skip] {stem}: window set differs")
                    continue
                else:
                    assert np.allclose(zz["true"].astype(np.float64) * sd + mu, truth, rtol=1e-3, atol=1.0), stem
                preds[m] = zz["pred"].astype(np.float64) * sd + mu
                vals[m] = float(rec["val_mse"])
            if "PLANET" not in preds or "TiDE" not in preds:
                print(f"  [skip] {d} H{H}: PLANET or TiDE predictions missing")
                continue
            bad = bad_window_idx(meta, d, H, split="test")
            keep = np.ones(len(idx0), bool)
            if bad:
                assert max(bad) < len(idx0), (d, H, max(bad), len(idx0))
                keep[np.asarray(bad)] = False
            y = truth[keep]
            blk = max(int(round(len(y) ** (1.0 / 3.0))), H)

            def score(p):
                return dict(wape=wape(p[keep], y), mse=float((((p[keep] - y) / sd) ** 2).mean()))

            members = {k: list(v) for k, v in COMBOS.items()}
            members["CombTop3"] = sorted(preds, key=vals.get)[:3]
            cell = dict(n_windows=int(keep.sum()), n_dropped=int((~keep).sum()), n_models=len(preds),
                        seed=SEED, regime=REGIME, best_val=min(vals, key=vals.get),
                        singles={m: dict(score(p), val_mse=vals[m]) for m, p in preds.items()})
            for c, ms in members.items():
                ms = [m for m in ms if m in preds]
                if len(ms) < 2:
                    continue
                p = np.mean([preds[m] for m in ms], axis=0)
                e = dict(score(p), members=ms, n_members=len(ms))
                ae_c = np.abs(p[keep] - y).mean(1)
                for ref in ("TiDE", "PLANET"):
                    ae_r = np.abs(preds[ref][keep] - y).mean(1)
                    st, pv = dm_bootstrap(ae_c, ae_r, h=H, B=B, block_len=blk)
                    e[f"dm_vs_{ref}"] = dict(stat=float(st), p=float(pv))
                cell[c] = e
            out[f"{d}|H{H}"] = cell
            s = cell["singles"]
            print(f"  {d} H{H}: TiDE {s['TiDE']['wape']:.2f}  PLANET {s['PLANET']['wape']:.2f}  "
                  f"Comb2 {cell['Comb2']['wape']:.2f}  Top3 {cell['CombTop3']['wape']:.2f} "
                  f"{cell['CombTop3']['members']}  All {cell['CombAll']['wape']:.2f} "
                  f"({cell['CombAll']['n_members']})  [{cell['n_windows']} windows, {cell['n_dropped']} dropped]",
                  flush=True)
    path = os.path.join(RES, "combo.json")
    json.dump(dict(_protocol=dict(seed=SEED, regime=REGIME, combos=COMBOS, top_k=3,
                                  dm="per-window MAE, combination vs reference; stationary-bootstrap SE, "
                                     "block max(n^1/3, H); stat<0 combination better", B=B),
                   **out), open(path, "w"), indent=2)
    print(f"wrote {path}: {len(out)} cells")


if __name__ == "__main__":
    main()
