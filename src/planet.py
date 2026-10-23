"""PLANET -- Phase-Locked Adaptive Network for Energy Time-series forecasting.

Two-tier phase-locked decomposition:

  slow tier   a per-channel coefficient vector over an absolute-time annual/trend basis,
              learned globally by SGD (climatology; identifiable only across the full record);
  fast tier   a per-window *weighted ridge* fit over an absolute-time daily/weekly/day-type
              basis, solved in closed form inside the forward pass, so the seasonal state is
              re-estimated for every instance at test time with no gradient steps;
  residual    a channel-independent patch Transformer over what the two tiers leave behind,
              conditioned on the local seasonal coefficients.

Because both bases are evaluated at true wall-clock time, the seasonal phase is locked to the
calendar: it neither leaks across window boundaries nor drifts as the horizon grows.
"""
import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------- weighting net
class ObservationWeighter(nn.Module):
    """Input-conditioned diagonal weights for the least-squares fit (channel-independent).

    Produces w in R^{B x L x C}, w >= 0, sum_t w = L. A learnable exponential-recency prior
    is added in logit space so the layer starts near uniform weighting and can learn to
    discount stale or anomalous lookback points (holidays, outages, meter faults).
    """

    def __init__(self, L, hidden=16):
        super().__init__()
        self.L = L
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
        )
        self.net[-1].weight.data.mul_(0.01)
        self.net[-1].bias.data.zero_()
        # softplus(rho) = recency half-life in steps; init very long -> almost uniform
        self.rho = nn.Parameter(torch.tensor(math.log(math.e ** 6 - 1)))
        age = torch.arange(L - 1, -1, -1, dtype=torch.float32)
        self.register_buffer("age", age)

    def forward(self, xn):                       # xn: (B, L, C)
        B, L, C = xn.shape
        z = xn.permute(0, 2, 1).reshape(B * C, 1, L)
        logit = self.net(z).reshape(B, C, L).permute(0, 2, 1)      # (B, L, C)
        logit = 3.0 * torch.tanh(logit)
        tau = torch.nn.functional.softplus(self.rho) * self.L      # decay length in steps
        logit = logit - (self.age / tau).view(1, L, 1)
        return torch.softmax(logit, dim=1) * L                     # sums to L over time


# --------------------------------------------------------------------- PLLS solver
def phase_locked_ls(Phi_L, Phi_H, xn, w, lam):
    """Batched weighted ridge on an absolute-time basis, solved in closed form.

    Phi_L (B,L,K) Phi_H (B,H,K) xn (B,L,C) w (B,L,C) lam (C,K) or (K,)
    returns s_hist (B,L,C), s_fut (B,H,C), beta (B,C,K)

    The penalty is indexed by (channel, basis function): a channel with no
    calendar structure -- wind speed, say -- simply learns a large penalty and
    shrinks its whole seasonal component away, leaving the work to the residual net.
    """
    B, L, K = Phi_L.shape
    C = xn.shape[-1]
    # Two algebraically identical routes to Phi^T W Phi; pick the one with the
    # smaller intermediate. Few channels (C < K) favours scaling the basis by
    # sqrt(w); many channels (ECL, C = 321) favours contracting the outer products.
    if C < K:
        Psi = Phi_L.unsqueeze(1) * w.permute(0, 2, 1).unsqueeze(-1).sqrt()   # (B,C,L,K)
        A = Psi.transpose(-1, -2) @ Psi
    else:
        O = (Phi_L.unsqueeze(-1) * Phi_L.unsqueeze(-2)).reshape(B, L, K * K)
        A = torch.bmm(w.transpose(1, 2), O).reshape(B, C, K, K)
    rhs = torch.bmm((w * xn).transpose(1, 2), Phi_L)               # Phi^T W x   -> (B, C, K)
    if lam.dim() == 1:
        lam = lam.unsqueeze(0).expand(C, K)
    A = A + torch.diag_embed(lam).unsqueeze(0)
    # A is symmetric positive definite (a weighted Gram matrix plus a strictly positive ridge), so
    # the batch of K x K systems is solved by Cholesky factorisation in float32; a diagonal jitter
    # retry covers the float32 rounding case, and the result is differentiable in A and rhs.
    with torch.autocast(device_type="cuda", enabled=False):
        A32, r32 = A.float(), rhs.float().unsqueeze(-1)
        Lc, info = torch.linalg.cholesky_ex(A32)
        if bool((info != 0).any()):
            eye = torch.eye(K, device=A32.device, dtype=A32.dtype)
            jit = 1e-4 * A32.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
            Lc, info = torch.linalg.cholesky_ex(A32 + jit * eye)
        beta = torch.cholesky_solve(r32, Lc).squeeze(-1).to(rhs.dtype)   # (B, C, K)
    bt = beta.transpose(1, 2)                                      # (B, K, C)
    return torch.bmm(Phi_L, bt), torch.bmm(Phi_H, bt), beta


# --------------------------------------------------------------------- native covariates (PLANET-X)
class CovEncoder(nn.Module):
    """Native covariate pathway of PLANET-X: the covariate block over lookback and horizon
    (B, L+H, E) -> per-step linear E -> w with GELU -> flatten -> linear to a d_cov vector.
    The same shape of pathway TiDE and N-BEATSx use for their exogenous inputs."""

    def __init__(self, E, T, w=4, d_cov=128):
        super().__init__()
        self.step = nn.Sequential(nn.Linear(E, w), nn.GELU())
        self.out = nn.Sequential(nn.Linear(T * w, d_cov), nn.GELU())
        self.d_cov = d_cov

    def forward(self, cov):                        # (B, T, E) -> (B, d_cov)
        B = cov.shape[0]
        return self.out(self.step(cov).reshape(B, -1))


# --------------------------------------------------------------------- residual encoder
class PatchEncoder(nn.Module):
    """Channel-independent patch Transformer (the PatchTST backbone), optionally
    conditioned on the local seasonal coefficients via a prepended state token."""

    def __init__(self, L, H, d_model=128, n_layers=3, n_heads=8, d_ff=256,
                 patch=16, stride=8, dropout=0.2, beta_dim=0, n_out=1, cov_dim=0):
        super().__init__()
        self.patch, self.stride, self.L, self.H, self.n_out = patch, stride, L, H, n_out
        self.n_patch = (L - patch) // stride + 1
        self.embed = nn.Linear(patch, d_model)
        self.beta_dim, self.cov_dim = beta_dim, cov_dim
        n_tok = self.n_patch + (1 if beta_dim else 0) + (1 if cov_dim else 0)
        if beta_dim:
            self.beta_proj = nn.Linear(beta_dim, d_model)
        if cov_dim:
            self.cov_proj = nn.Linear(cov_dim, d_model)      # one covariate token (PLANET-X)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=False)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Dropout(dropout),
                                  nn.Linear(n_tok * d_model, H * n_out))

    def forward(self, r, beta=None, c=None):     # r: (B, L, C); c: (B, cov_dim)
        B, L, C = r.shape
        z = r.permute(0, 2, 1).reshape(B * C, L)
        z = z.unfold(-1, self.patch, self.stride)            # (B*C, n_patch, patch)
        z = self.embed(z)
        if self.beta_dim:
            bt = self.beta_proj(beta.reshape(B * C, -1)).unsqueeze(1)
            z = torch.cat([bt, z], dim=1)
        if self.cov_dim and c is not None:
            ct = self.cov_proj(c).repeat_interleave(C, 0).unsqueeze(1)
            z = torch.cat([ct, z], dim=1)
        z = self.drop(z + self.pos)
        z = self.enc(z)
        o = self.head(z)                                     # (B*C, H*n_out)
        o = o.reshape(B, C, self.H, self.n_out)
        return o.permute(0, 2, 1, 3)                         # (B, H, C, n_out)


# --------------------------------------------------------------------- alternative residual net
class BasisResidual(nn.Module):
    """An N-BEATS-style doubly-residual stack over the residual series.

    The phase-locked tiers make no assumption about what models the remainder, so the
    residual network is a swappable component. Offering a second, architecturally very
    different backbone lets us test whether the tier's contribution is a property of the
    decomposition or an artefact of one particular encoder.
    """

    def __init__(self, L, H, n_blocks=9, width=256, depth=4, dropout=0.0,
                 beta_dim=0, n_out=1, beta_cond="proj", cov_dim=0):
        super().__init__()
        self.L, self.H, self.n_out = L, H, n_out
        self.beta_dim, self.beta_cond, self.cov_dim = beta_dim, beta_cond, cov_dim
        if beta_dim:
            self.beta_proj = nn.Linear(beta_dim, L)
            self.beta_proj.weight.data.mul_(0.01); self.beta_proj.bias.data.zero_()
        # v2 (M4b): FiLM conditioning of every block's hidden state on the seasonal
        # coefficients -- zero-initialised so the stack starts as the unconditioned model.
        self.film = None
        if beta_dim and beta_cond == "film":
            self.film = nn.ModuleList([nn.Linear(beta_dim, 2 * width) for _ in range(n_blocks)])
            for f in self.film:
                f.weight.data.zero_(); f.bias.data.zero_()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            layers, d = [], L + cov_dim                  # PLANET-X: covariate embedding joins every block input
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d = width
            self.blocks.append(nn.ModuleDict(dict(
                fc=nn.Sequential(*layers),
                back=nn.Linear(width, L),
                fore=nn.Linear(width, H * n_out))))

    def forward(self, r, beta=None, c=None):         # r: (B, L, C); c: (B, cov_dim)
        B, L, C = r.shape
        z = r.permute(0, 2, 1).reshape(B * C, L)
        if self.beta_dim and beta is not None:
            z = z + self.beta_proj(beta.reshape(B * C, -1))
        out = z.new_zeros(B * C, self.H * self.n_out)
        bflat = beta.reshape(B * C, -1) if (self.film is not None and beta is not None) else None
        cflat = c.repeat_interleave(C, 0) if (self.cov_dim and c is not None) else None
        for i, blk in enumerate(self.blocks):
            h = blk["fc"](torch.cat([z, cflat], -1) if cflat is not None else z)
            if bflat is not None:
                g, b = self.film[i](bflat).chunk(2, -1)
                h = h * (1.0 + g) + b
            z = z - blk["back"](h)
            out = out + blk["fore"](h)
        out = out.reshape(B, C, self.H, self.n_out)
        return out.permute(0, 2, 1, 3)               # (B, H, C, n_out)


# --------------------------------------------------------------------- PLANET
class PLANET(nn.Module):
    def __init__(self, L, H, C, K_fast, K_slow, d_model=128, n_layers=3, n_heads=8,
                 d_ff=256, patch=16, stride=8, dropout=0.2,
                 use_pl=True, use_slow=True, learn_w=True, beta_token=True,
                 use_resid=True, resid="patch", n_blocks=9, width=256, depth=4,
                 lam_init=0.01, lam_per_channel=True, n_quant=1, beta_cond="proj",
                 cov_native=False, E=0, cov_width=4, d_cov=128, **kw):
        super().__init__()
        self.L, self.H, self.C = L, H, C
        # PLANET-X: the covariate block enters the residual backbone natively instead of
        # through the shared exogenous head (needs_cov makes the harness pass it in)
        self.cov_native = bool(cov_native and E > 0)
        self.needs_cov = self.cov_native
        self.native_cov = self.cov_native
        self.cov_enc = CovEncoder(E, L + H, cov_width, d_cov) if self.cov_native else None
        cov_dim = d_cov if self.cov_native else 0
        self.use_pl, self.use_slow, self.learn_w = use_pl, use_slow, learn_w
        self.use_resid = use_resid
        self.n_quant = n_quant
        assert use_pl or use_resid, "at least one of the two branches must be active"
        if use_slow:
            self.beta_g = nn.Parameter(torch.zeros(C, K_slow))
        self.lam_per_channel = lam_per_channel
        if use_pl:
            shape = (C, K_fast) if lam_per_channel else (K_fast,)
            self.rho_lam = nn.Parameter(torch.full(shape, math.log(lam_init)))
            self.weighter = ObservationWeighter(L) if learn_w else None
        bd = K_fast if (use_pl and beta_token) else 0
        if not use_resid:
            self.enc = None
        elif resid == "basis":
            self.enc = BasisResidual(L, H, n_blocks, width, depth, dropout, bd, n_quant,
                                     beta_cond=beta_cond, cov_dim=cov_dim)
        else:
            self.enc = PatchEncoder(L, H, d_model, n_layers, n_heads, d_ff, patch, stride,
                                    dropout, beta_dim=bd, n_out=n_quant, cov_dim=cov_dim)
        if not use_resid:   # keeps the optimiser non-empty for the closed-form-only ablation
            self.null = nn.Parameter(torch.zeros(1))

    def lam(self):
        """Ridge penalty, parameterised multiplicatively.

        The useful range of the penalty spans several orders of magnitude: a channel with
        clean daily/weekly structure wants almost none, an aperiodic one wants enough to
        erase its seasonal component entirely. An exponential parameterisation (paired with
        a larger step size on these coefficients) lets the model traverse that range in a
        few epochs, where an additive one cannot.
        """
        return self.L * (torch.exp(self.rho_lam.clamp(-12.0, 6.0)) + 1e-3)

    def forward(self, x, Phi, Psi, tf=None, cov=None):
        """x (B,L,C); Phi (B,L+H,K_fast); Psi (B,L+H,K_slow); cov (B,L+H,E) for PLANET-X
        -> (B,H,C) or (B,H,C,Q)"""
        B, L, C = x.shape
        H = self.H
        c = self.cov_enc(cov.float()) if (self.cov_native and cov is not None) else None
        # The decomposition tiers are solved in fp32 regardless of any surrounding autocast:
        # the normal equations are mildly ill-conditioned and half precision is not safe here.
        with torch.autocast("cuda", enabled=False):
            x, Phi, Psi = x.float(), Phi.float(), Psi.float()
            if self.use_slow:
                g = torch.einsum("btk,ck->btc", Psi, self.beta_g)
                x1, g_fut = x - g[:, :L], g[:, L:]
            else:
                x1, g_fut = x, None

            mu = x1.mean(1, keepdim=True)
            sd = x1.std(1, keepdim=True) + 1e-5
            xn = (x1 - mu) / sd

            beta = None
            if self.use_pl:
                Phi_L, Phi_H = Phi[:, :L], Phi[:, L:]
                w = self.weighter(xn) if self.learn_w else xn.new_ones(B, L, C)
                s_hist, s_fut, beta = phase_locked_ls(Phi_L, Phi_H, xn, w, self.lam())
                r = xn - s_hist
            else:
                s_fut, r = torch.zeros(B, H, C, device=x.device, dtype=x.dtype), xn

        if self.use_resid:
            o = self.enc(r, beta, c).float()                 # (B, H, C, Q)
        else:
            o = s_fut.new_zeros(B, H, C, self.n_quant) + 0.0 * self.null
        yn = s_fut.unsqueeze(-1) + o
        y = yn * sd.unsqueeze(-1) + mu.unsqueeze(-1)
        if g_fut is not None:
            y = y + g_fut.unsqueeze(-1)
        return y.squeeze(-1) if self.n_quant == 1 else y

    # -- diagnostics used by the paper's decomposition figure -----------------
    @torch.no_grad()
    def decompose(self, x, Phi, Psi):
        B, L, C = x.shape
        H = self.H
        g = torch.einsum("btk,ck->btc", Psi, self.beta_g) if self.use_slow else torch.zeros_like(Phi[..., :1]).expand(B, L + H, C) * 0
        x1 = x - g[:, :L]
        mu = x1.mean(1, keepdim=True); sd = x1.std(1, keepdim=True) + 1e-5
        xn = (x1 - mu) / sd
        w = self.weighter(xn) if self.learn_w else xn.new_ones(B, L, C)
        s_hist, s_fut, beta = phase_locked_ls(Phi[:, :L], Phi[:, L:], xn, w, self.lam())
        o = self.enc(xn - s_hist, beta)[..., 0]
        return dict(slow=g[:, L:], fast=s_fut * sd + mu, resid=o * sd,
                    weights=w, beta=beta,
                    total=(s_fut.unsqueeze(-1) + o.unsqueeze(-1)).squeeze(-1) * sd + mu + g[:, L:])
