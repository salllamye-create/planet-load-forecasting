"""Shared exogenous head (v2 protocol, principle P1).

Every model in an information regime receives the identical covariate block
cov (B, L+H, E): the calendar block (regime R1) plus the NWP-degraded weather block
(regime R2), see V2_SPEC.md. A model with native known-future inputs consumes it
directly. A model without them is wrapped in `WithExo`, which adds the output of one
small head -- the same module for every such model, and the same one PLANET uses --
so that the differences left in the tables are architectural rather than
informational.

The head is the v1 PLANET-X weather response generalised to an arbitrary covariate
block: a point-wise non-linear response at every horizon step, a summary of the
covariate history (thermal inertia), a short depth-wise temporal convolution, and a
small-initialised linear output so that a wrapped model starts exactly at its own
forecast.
"""
import torch
import torch.nn as nn


class ExoHead(nn.Module):
    def __init__(self, E, L, H, n_out=1, d=48, kernel=9, dropout=0.1):
        super().__init__()
        self.E, self.L, self.H, self.n_out = E, L, H, n_out
        self.point = nn.Sequential(nn.Linear(E, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.past = nn.Sequential(nn.Linear(4 * E, d), nn.GELU())
        self.temporal = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, n_out)
        self.out.weight.data.mul_(0.01)
        self.out.bias.data.zero_()

    def forward(self, cov_past, cov_fut):
        """cov_past (B,L,E), cov_fut (B,H,E) -> additive correction (B,H,n_out)."""
        h = self.point(cov_fut)
        ctx = torch.cat([cov_past.mean(1), cov_past[:, -1],
                         cov_past.amax(1), cov_past.amin(1)], -1)
        h = h + self.past(ctx).unsqueeze(1)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)
        return self.out(self.drop(self.norm(h)))


class WithExo(nn.Module):
    """Wrap any v1-signature forecaster so it consumes the covariate block through the
    shared head. With E == 0 (regime R0) the wrapper is the identity around the base."""

    needs_cov = True

    def __init__(self, base, E, L, H, C, target_idx=-1, ms=True, d=48, kernel=9, dropout=0.1):
        super().__init__()
        self.base = base
        self.L, self.H, self.C, self.ms = L, H, C, ms
        self.target_idx = target_idx % C
        self.head = ExoHead(E, L, H, n_out=1 if ms else C, d=d, kernel=kernel,
                            dropout=dropout) if E > 0 else None

    def forward(self, x, Phi=None, Psi=None, tf=None, cov=None):
        if getattr(self.base, "needs_cov", False):
            y = self.base(x, Phi, Psi, tf, cov)
        else:
            y = self.base(x, Phi, Psi, tf)
        if self.head is None or cov is None:
            return y
        corr = self.head(cov[:, :self.L], cov[:, self.L:])            # (B,H,1) or (B,H,C)
        if y.dim() == 4:                                               # quantile outputs (B,H,C,Q)
            corr = corr.unsqueeze(-1)
        if self.ms:
            y = y.clone()
            y[:, :, self.target_idx] = y[:, :, self.target_idx] + corr[:, :, 0]
            return y
        return y + corr

    # pass-through helpers used by the training loop
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)
