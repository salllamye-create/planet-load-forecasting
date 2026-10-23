"""PLANET-X: PLANET plus an explicit weather pathway.

A channel-independent backbone cannot, by construction, use the exogenous channels of a
multivariate input: each series is encoded on its own. That is fine for the standard
benchmarks but wasteful in a control room, where day-ahead weather forecasts are on the
desk before the load forecast is made.

PLANET-X adds one small branch that maps the *known future* weather onto an additive load
correction. It is deliberately shaped like the classical load-temperature response: a
point-wise non-linear response applied at every horizon step, a short temporal convolution
for thermal inertia, and a calendar-conditioned gain (the same degree of warming moves
industrial and residential load differently on a Tuesday and on a Sunday).

Weather is never perfectly known ahead of time, so the branch is evaluated under three
information regimes -- `perfect` (reanalysis, an ex-post upper bound), `noisy` (reanalysis
degraded by a lead-time-dependent error matched to operational NWP skill) and `persist`
(yesterday's weather) -- and the paper reports all of them.
"""
import math
import torch
import torch.nn as nn

from planet import PLANET


class WeatherResponse(nn.Module):
    def __init__(self, n_exo, H, K_fast=0, d=48, kernel=9, dropout=0.1):
        super().__init__()
        self.H = H
        self.point = nn.Sequential(nn.Linear(n_exo, d), nn.GELU(),
                                   nn.Linear(d, d), nn.GELU())
        self.past = nn.Sequential(nn.Linear(4 * n_exo, d), nn.GELU())
        self.cal = nn.Linear(K_fast, d) if K_fast else None
        self.temporal = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 1)
        self.out.weight.data.mul_(0.01); self.out.bias.data.zero_()

    def forward(self, exo_fut, exo_past, Phi_H=None):
        """exo_fut (B,H,E), exo_past (B,L,E) -> (B,H) additive correction."""
        h = self.point(exo_fut)                                   # non-linear response
        ctx = torch.cat([exo_past.mean(1), exo_past[:, -1],
                         exo_past.amax(1), exo_past.amin(1)], -1)
        h = h + self.past(ctx).unsqueeze(1)                       # thermal history
        if self.cal is not None and Phi_H is not None:
            h = h * torch.sigmoid(self.cal(Phi_H))                # calendar-conditioned gain
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        return self.out(self.drop(self.norm(h))).squeeze(-1)


class PLANETX(PLANET):
    needs_exo = True

    def __init__(self, L, H, C, K_fast, K_slow, exo_idx=None, target_idx=-1,
                 exo_d=48, exo_kernel=9, **kw):
        super().__init__(L, H, C, K_fast, K_slow, **kw)
        self.target_idx = target_idx % C
        self.exo_idx = [i for i in range(C) if i != self.target_idx] if exo_idx is None else list(exo_idx)
        self.register_buffer("exo_sel", torch.tensor(self.exo_idx, dtype=torch.long))
        self.weather = WeatherResponse(len(self.exo_idx), H, K_fast, exo_d, exo_kernel,
                                       kw.get("dropout", 0.1))

    def forward(self, x, Phi, Psi, tf=None, exo=None):
        y = super().forward(x, Phi, Psi, tf)
        if exo is None:
            return y
        corr = self.weather(exo.index_select(-1, self.exo_sel),
                            x.index_select(-1, self.exo_sel),
                            Phi[:, self.L:].float())
        y = y.clone()
        y[:, :, self.target_idx] = y[:, :, self.target_idx] + corr
        return y


# --------------------------------------------------------------------- info regimes
# RMSE of an operational day-ahead 2 m-temperature forecast is ~1.0-1.5 K at 24 h and
# grows roughly with the square root of lead time; the same relative degradation is
# applied to the other weather channels.
NWP_RMSE_24H = {"temp": 1.2, "rhum": 6.0, "rad": 40.0, "wspd": 1.1,
                "solar": 900.0, "wind": 1400.0}


def degrade(exo_true, channels, sd, mode="perfect", x_past=None, gen=None):
    """Turn the reanalysis future into the information a forecaster would really have."""
    if mode == "perfect":
        return exo_true
    if mode == "persist":                       # yesterday, repeated
        H = exo_true.shape[1]
        idx = torch.arange(H, device=exo_true.device) % 24 + (x_past.shape[1] - 24)
        return x_past[:, idx]
    B, H, E = exo_true.shape
    lead = torch.arange(1, H + 1, device=exo_true.device, dtype=exo_true.dtype)
    scale = torch.sqrt(lead / 24.0).view(1, H, 1)
    s = torch.tensor([NWP_RMSE_24H.get(c, 1.0) / max(float(sd[i]), 1e-6)
                      for i, c in channels], device=exo_true.device,
                     dtype=exo_true.dtype).view(1, 1, E)
    noise = torch.randn(exo_true.shape, device=exo_true.device,
                        dtype=exo_true.dtype, generator=gen)
    return exo_true + noise * s * scale
