import json, torch, numpy as np, sys
import exp
from argparse import Namespace

def go(data, lr, extra, tag, H=96, ep=10, ms=0, batch=64):
    a = Namespace(data=data, model="PLANET", L=336, H=H, seed=2021, lr=lr, wd=0.0,
                  wd_slow=1e-2, lr_mult_lam=50.0, batch=batch, test_batch=0, epochs=ep,
                  patience=3, workers=0, amp=0, collect=0, save_ckpt=0, ms=ms,
                  exo_mode="perfect", extra=json.dumps(extra), tag=tag)
    return exp.run(a)

if __name__ == "__main__":
    data = sys.argv[1]; ms = int(sys.argv[2])
    for lr in [1e-4, 5e-4, 1e-3]:
        r = go(data, lr, {}, f"_lam{lr:g}", ms=ms)
        print(f"  >>> {data} lr={lr:g} MSE={r['mse']:.4f} MAE={r['mae']:.4f}", flush=True)
