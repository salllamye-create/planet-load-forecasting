"""The official neuralforecast implementations behind the shared forward signature.

    forward(x, Phi, Psi, tf, cov) -> (B, H, C)

Wrapping the library's own modules -- rather than re-implementing them -- keeps each
baseline faithful to its authors' code while every method still shares the v2 windows,
normalisation, optimiser, budget and information set. Only the model *module* is used;
neuralforecast's trainer, scaler and data pipeline are bypassed.

Native covariate interface (V2_SPEC.md, principle P1):
  * univariate models (TFT, TiDE, N-HiTS, N-BEATSx, TimesNet) are channel-independent: every
    channel is a separate series and the covariate block is supplied as `futr_exog` over
    the lookback and the horizon;
  * multivariate models (TimeXer, TSMixerx, XLinear) take all channels as `insample_y` and the
    block as `futr_exog`/`hist_exog` in their [B, F, L+h, N] convention.
  TimesNet and TimeXer natively read only the past part of the block (their authors' design);
  the training harness therefore also gives them the shared future head (`WithExo`), which
  is stated in the paper.
"""
import inspect
import torch
import torch.nn as nn

from neuralforecast.models import TFT, TiDE, NHITS, NBEATSx, TimesNet, TimeXer, TSMixerx, XLinear
from neuralforecast.losses.pytorch import MAE

NF_CLASSES = {"TFT": TFT, "TiDE": TiDE, "NHITS": NHITS, "NBEATSx": NBEATSx,
              "TimesNet": TimesNet, "TimeXer": TimeXer, "TSMixerx": TSMixerx, "XLinear": XLinear,
              "TiDEh": TiDE}                       # TiDE fed through the shared head (pathway control)
MULTIVARIATE = {"TimeXer", "TSMixerx", "XLinear"}
PAST_ONLY = {"TimeXer"}                            # library rejects futr_exog for this model
# Models whose covariate block reaches them through the shared head (exo.WithExo): TimesNet and
# TimeXer because their native interface reads only the past block; TiDEh is TiDE with its native
# covariate pathway switched off (the library sees no exogenous columns) so that the block enters
# through the same head as for PLANET -- the control for the covariate-pathway confound.
FUTURE_VIA_HEAD = {"TimesNet", "TimeXer", "TiDEh"}
HEAD_ONLY = {"TiDEh"}

# published default sizes (neuralforecast defaults) and the knobs the size sweep scales
NF_CFG = {
    "TFT":      dict(hidden_size=128, n_head=4, dropout=0.1),
    "TiDE":     dict(hidden_size=512, decoder_output_dim=8, temporal_decoder_dim=128,
                     num_encoder_layers=1, num_decoder_layers=1, temporal_width=4, dropout=0.3,
                     layernorm=False),   # the library applies LayerNorm to the 1-dim decoder output, collapsing point forecasts to a constant
    "NHITS":    dict(stack_types=["identity"] * 3, n_blocks=[1, 1, 1], mlp_units=[[512, 512]] * 3,
                     n_pool_kernel_size=[2, 2, 1], n_freq_downsample=[4, 2, 1], dropout_prob_theta=0.0),
    "NBEATSx":  dict(stack_types=["identity", "trend", "seasonality"], n_blocks=[1, 1, 1],
                     mlp_units=[[512, 512]] * 3, dropout_prob_theta=0.0),
    "TimesNet": dict(hidden_size=64, dropout=0.1, conv_hidden_size=64, top_k=5, num_kernels=6,
                     encoder_layers=2),
    "TimeXer":  dict(patch_len=16, hidden_size=512, n_heads=8, e_layers=2, d_ff=2048, dropout=0.1),
    "TSMixerx": dict(n_block=2, ff_dim=64, dropout=0.1),
    "XLinear":  dict(hidden_size=256),
}
NF_CFG["TiDEh"] = dict(NF_CFG["TiDE"])
SIZE_KEYS = {
    "TFT": ["hidden_size"], "TiDE": ["hidden_size", "temporal_decoder_dim"],
    "NHITS": ["mlp_units"], "NBEATSx": ["mlp_units"],
    "TimesNet": ["hidden_size", "conv_hidden_size"], "TimeXer": ["hidden_size", "d_ff"],
    "TSMixerx": ["ff_dim"], "XLinear": ["hidden_size"],
}
SIZE_KEYS["TiDEh"] = list(SIZE_KEYS["TiDE"])


def scaled_cfg(name, k):
    """Scale the width knobs of a model by k (0.5 / 1 / 2 -> the small / published / large sizes)."""
    cfg = dict(NF_CFG[name])
    for key in SIZE_KEYS[name]:
        v = cfg[key]
        if isinstance(v, list):
            cfg[key] = [[max(8, int(u * k)) for u in layer] for layer in v]
        else:
            cfg[key] = max(8, int(v * k))
    return cfg


class NFWrap(nn.Module):
    needs_cov = True

    def __init__(self, name, L, H, C, E, seed=1, keep_q=False, **cfg):
        super().__init__()
        cls = NF_CLASSES[name]
        self.keep_q = keep_q                    # return (B,H,C,Q) from a multi-quantile head
        self.name, self.L, self.H, self.C, self.E = name, L, H, C, E
        self.multivariate = name in MULTIVARIATE
        self.past_only = name in PAST_ONLY
        names = [f"c{i}" for i in range(E)] if E > 0 else None
        kw = dict(h=H, input_size=L, loss=MAE(), valid_loss=None,
                  futr_exog_list=None if self.past_only else names,
                  hist_exog_list=names if self.past_only else None, stat_exog_list=None,
                  random_seed=seed)
        sig = inspect.signature(cls.__init__).parameters
        if "scaler_type" in sig:
            kw["scaler_type"] = "identity"        # the harness z-scores once, uniformly
        if "n_series" in sig:
            kw["n_series"] = C
        for k, v in cfg.items():
            if k in sig:
                kw[k] = v
        self.m = cls(**kw)

    def forward(self, x, Phi=None, Psi=None, tf=None, cov=None):
        # Per-window instance normalisation of every target channel (mean and sd over the
        # lookback), inverted on the forecast. The re-implemented baselines carry RevIN and
        # PLANET its own instance normalisation; the library models are normally run with the
        # library's temporal scaler, which the harness bypasses by calling forward() directly.
        # Without this step the library models would be the only ones to see raw levels.
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-5
        y = self._forward((x - mu) / sd, cov)
        if y.dim() == 4:                                   # (B, H, C, Q) quantile head
            return y * sd.unsqueeze(-1) + mu.unsqueeze(-1)
        return y * sd + mu

    def _forward(self, x, cov):
        B, L, C = x.shape
        H = self.H
        if self.E > 0:
            assert cov is not None and cov.shape[-1] == self.E, "covariate block missing"
        if not self.multivariate:
            ys = x.permute(0, 2, 1).reshape(B * C, L, 1)
            batch = dict(insample_y=ys, insample_mask=torch.ones_like(ys),
                         futr_exog=cov.repeat_interleave(C, 0) if self.E > 0 else None,
                         hist_exog=None, stat_exog=None)
            out = self.m(batch)
            if self.keep_q:
                return out.reshape(B * C, H, -1).reshape(B, C, H, -1).permute(0, 2, 1, 3)
            out = out.reshape(B * C, H, -1)[..., 0].reshape(B, C, H).permute(0, 2, 1)
            return out
        futr, hist = None, None
        if self.E > 0:
            blk = cov.permute(0, 2, 1).unsqueeze(-1).expand(B, self.E, L + H, C).contiguous()
            if self.past_only:
                hist = blk[:, :, :L]                                # [B, X, L, N]
            else:
                futr = blk                                          # [B, F, L+h, N]
        batch = dict(insample_y=x, insample_mask=torch.ones_like(x), futr_exog=futr,
                     hist_exog=hist, stat_exog=None)
        out = self.m(batch)
        return out.reshape(B, H, -1)[:, :, :C]


def build_nf(name, L, H, C, E, size_k=1.0, seed=1, **override):
    cfg = scaled_cfg(name, size_k)
    cfg.update(override)
    if name in HEAD_ONLY:
        E = 0                                      # the block reaches the model through WithExo
    return NFWrap(name, L, H, C, E, seed=seed, **cfg)


if __name__ == "__main__":                       # shape smoke test on random data
    import sys
    L, H, C, E = 336, 96, 7, 14
    x = torch.randn(2, L, C); cov = torch.randn(2, L + H, E)
    for name in NF_CLASSES:
        try:
            m = build_nf(name, L, H, C, E)
            with torch.no_grad():
                y = m(x, cov=cov)
            n = sum(p.numel() for p in m.parameters())
            print(f"{name:9s} out={tuple(y.shape)} params={n/1e3:.0f}k")
        except Exception as e:
            print(f"{name:9s} FAILED: {type(e).__name__}: {e}")
