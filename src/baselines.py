"""Faithful compact re-implementations of the competing forecasters.

Every model exposes the same signature  forward(x, Phi, Psi, tf) -> (B, H, C)
so that all methods are trained and evaluated by exactly the same loop.
  x   (B, L, C)     lookback, dataset-z-scored
  Phi (B, L+H, Kf)  absolute-time fast basis   (PLANET only)
  Psi (B, L+H, Ks)  absolute-time slow basis   (PLANET only)
  tf  (B, L+H, 5)   calendar features          (Informer / Autoformer temporal embedding)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ common blocks
class _NoRev(nn.Module):
    """Pass-through in place of RevIN. Informer and Autoformer are reported without instance
    normalisation (SETUP): their records were made before the wrapper gained it, and the re-run
    under the full protocol was cancelled for the compute budget (PLAN_v2.md, 2026-10-20 18:30)."""
    def forward(self, x, mode):
        return x


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., ICLR 2022)."""

    def __init__(self, C, affine=True, eps=1e-5):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.w = nn.Parameter(torch.ones(C))
            self.b = nn.Parameter(torch.zeros(C))

    def forward(self, x, mode):
        if mode == "norm":
            self.mu = x.mean(1, keepdim=True).detach()
            self.sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mu) / self.sd
            return x * self.w + self.b if self.affine else x
        x = (x - self.b) / (self.w + self.eps ** 2) if self.affine else x
        return x * self.sd + self.mu


class MovingAvg(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.avg = nn.AvgPool1d(k, stride=1, padding=0)

    def forward(self, x):                      # (B, L, C)
        f = x[:, :1].repeat(1, (self.k - 1) // 2, 1)
        b = x[:, -1:].repeat(1, self.k - 1 - (self.k - 1) // 2, 1)
        return self.avg(torch.cat([f, x, b], 1).permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, k=25):
        super().__init__()
        self.ma = MovingAvg(k)

    def forward(self, x):
        t = self.ma(x)
        return x - t, t


# ============================================================ trivial / classical
class SeasonalNaive(nn.Module):
    """Repeat the last observed cycle. The period is chosen on the validation split from
    the natural candidates for the sampling rate (one day, one week), clipped to what the
    lookback can actually supply."""
    trainable = False

    def __init__(self, L, H, C, period=None, steps_per_day=24, **kw):
        super().__init__()
        self.L, self.H, self.steps_per_day = L, H, steps_per_day
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.period = int(period) if period else self.candidates()[0]

    def candidates(self):
        c = [p for p in (7 * self.steps_per_day, self.steps_per_day) if p <= self.L]
        return c or [self.L]

    def forward(self, x, Phi=None, Psi=None, tf=None):
        p = min(self.period, self.L)
        idx = (torch.arange(self.H, device=x.device) % p) + (self.L - p)
        return x[:, idx] + 0.0 * self.dummy


# ============================================================ linear family
class DLinear(nn.Module):
    def __init__(self, L, H, C, kernel=25, individual=False, **kw):
        super().__init__()
        self.decomp = SeriesDecomp(kernel)
        self.individual, self.C = individual, C
        if individual:
            self.ls = nn.ModuleList([nn.Linear(L, H) for _ in range(C)])
            self.lt = nn.ModuleList([nn.Linear(L, H) for _ in range(C)])
        else:
            self.ls, self.lt = nn.Linear(L, H), nn.Linear(L, H)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        s, t = self.decomp(x)
        s, t = s.permute(0, 2, 1), t.permute(0, 2, 1)
        if self.individual:
            s = torch.stack([self.ls[i](s[:, i]) for i in range(self.C)], 1)
            t = torch.stack([self.lt[i](t[:, i]) for i in range(self.C)], 1)
        else:
            s, t = self.ls(s), self.lt(t)
        return (s + t).permute(0, 2, 1)


class NLinear(nn.Module):
    def __init__(self, L, H, C, **kw):
        super().__init__()
        self.lin = nn.Linear(L, H)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        last = x[:, -1:].detach()
        y = self.lin((x - last).permute(0, 2, 1)).permute(0, 2, 1)
        return y + last


class RLinear(nn.Module):
    """RevIN + a single shared linear map (Li et al., 2023)."""

    def __init__(self, L, H, C, dropout=0.1, **kw):
        super().__init__()
        self.rev = RevIN(C)
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(L, H)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        z = self.rev(x, "norm")
        z = self.lin(self.drop(z.permute(0, 2, 1))).permute(0, 2, 1)
        return self.rev(z, "denorm")


class FITS(nn.Module):
    """Frequency Interpolation Time Series (Xu et al., ICLR 2024): a complex-valued
    linear interpolator on the low-frequency rFFT coefficients of the lookback."""

    def __init__(self, L, H, C, base_period=24, n_harm=6, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.cut = min(L // 2 + 1, int(round(L / base_period)) * n_harm + 1)
        self.out = int(self.cut * (L + H) / L) + 1
        self.re = nn.Linear(self.cut, self.out, bias=False)
        self.im = nn.Linear(self.cut, self.out, bias=False)
        nn.init.eye_(self.re.weight[:self.cut, :self.cut]) if self.out >= self.cut else None
        self.im.weight.data.zero_()

    def forward(self, x, Phi=None, Psi=None, tf=None):
        mu = x.mean(1, keepdim=True)
        sd = x.std(1, keepdim=True) + 1e-5
        z = (x - mu) / sd
        f = torch.fft.rfft(z, dim=1)[:, :self.cut]                 # (B, cut, C)
        fr, fi = f.real.permute(0, 2, 1), f.imag.permute(0, 2, 1)
        orr = self.re(fr) - self.im(fi)
        oii = self.re(fi) + self.im(fr)
        o = torch.complex(orr, oii).permute(0, 2, 1)               # (B, out, C)
        full = z.new_zeros(z.shape[0], (self.L + self.H) // 2 + 1, z.shape[2], dtype=torch.cfloat)
        n = min(self.out, full.shape[1])
        full[:, :n] = o[:, :n]
        y = torch.fft.irfft(full, n=self.L + self.H, dim=1) * ((self.L + self.H) / self.L)
        return y[:, self.L:] * sd + mu


# ============================================================ PatchTST
class PatchTST(nn.Module):
    def __init__(self, L, H, C, d_model=128, n_layers=3, n_heads=8, d_ff=256,
                 patch=16, stride=8, dropout=0.2, **kw):
        super().__init__()
        self.rev = RevIN(C)
        self.patch, self.stride, self.L, self.H = patch, stride, L, H
        n = (L - patch) // stride + 1
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, n, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(n * d_model, H))

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        z = self.rev(x, "norm").permute(0, 2, 1).reshape(B * C, L)
        z = self.embed(z.unfold(-1, self.patch, self.stride))
        z = self.enc(self.drop(z + self.pos))
        y = self.head(z).reshape(B, C, self.H).permute(0, 2, 1)
        return self.rev(y, "denorm")


# ============================================================ iTransformer
class ITransformer(nn.Module):
    def __init__(self, L, H, C, d_model=128, n_layers=3, n_heads=8, d_ff=256, dropout=0.1, **kw):
        super().__init__()
        self.embed = nn.Linear(L, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, H)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        mu = x.mean(1, keepdim=True)
        sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        z = ((x - mu) / sd).permute(0, 2, 1)                       # (B, C, L) -- variates as tokens
        z = self.enc(self.drop(self.embed(z)))
        y = self.proj(z).permute(0, 2, 1)
        return y * sd + mu


# ============================================================ TimeMixer (past-decomposable mixing)
class TimeMixer(nn.Module):
    def __init__(self, L, H, C, d_model=64, n_layers=2, n_scales=3, down=2,
                 dropout=0.1, kernel=25, **kw):
        super().__init__()
        self.L, self.H, self.n_scales, self.down = L, H, n_scales, down
        self.rev = RevIN(C)
        self.lens = [L // (down ** i) for i in range(n_scales)]
        self.decomp = SeriesDecomp(kernel)
        self.embed = nn.ModuleList([nn.Sequential(nn.Linear(l, d_model), nn.GELU()) for l in self.lens])
        self.n_layers = n_layers
        # bottom-up seasonal mixing (fine -> coarse) and top-down trend mixing (coarse -> fine)
        self.s_mix = nn.ModuleList([nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_scales - 1)])
                                    for _ in range(n_layers)])
        self.t_mix = nn.ModuleList([nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_scales - 1)])
                                    for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.pred = nn.ModuleList([nn.Linear(d_model, H) for _ in range(n_scales)])
        self.wts = nn.Parameter(torch.ones(n_scales) / n_scales)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        z = self.rev(x, "norm").permute(0, 2, 1)                   # (B, C, L)
        xs = [z]
        for _ in range(self.n_scales - 1):
            xs.append(F.avg_pool1d(xs[-1], self.down, self.down))
        xs = [e(v[..., -l:]) for e, v, l in zip(self.embed, xs, self.lens)]   # (B, C, d) per scale

        for li in range(self.n_layers):
            seas, tren = [], []
            for v in xs:
                s, t = self.decomp(v)
                seas.append(s); tren.append(t)
            for i in range(self.n_scales - 1):                     # bottom-up: fine informs coarse
                seas[i + 1] = seas[i + 1] + self.s_mix[li][i](seas[i])
            for i in range(self.n_scales - 2, -1, -1):             # top-down: coarse informs fine
                tren[i] = tren[i] + self.t_mix[li][i](tren[i + 1])
            xs = [self.norm[li](self.drop(s + t) + v) for s, t, v in zip(seas, tren, xs)]

        w = torch.softmax(self.wts, 0)
        y = sum(w[i] * self.pred[i](xs[i]) for i in range(self.n_scales))     # (B, C, H)
        return self.rev(y.permute(0, 2, 1), "denorm")


# ============================================================ N-BEATS
class NBeatsBlock(nn.Module):
    def __init__(self, L, H, width=256, depth=4):
        super().__init__()
        layers = []
        d = L
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.fc = nn.Sequential(*layers)
        self.tb = nn.Linear(width, L)
        self.tf = nn.Linear(width, H)

    def forward(self, z):
        h = self.fc(z)
        return self.tb(h), self.tf(h)


class NBEATS(nn.Module):
    """Generic doubly-residual stack (Oreshkin et al., ICLR 2020), channel-independent."""

    def __init__(self, L, H, C, n_blocks=9, width=256, depth=4, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.blocks = nn.ModuleList([NBeatsBlock(L, H, width, depth) for _ in range(n_blocks)])
        self.rev = RevIN(C)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        z = self.rev(x, "norm").permute(0, 2, 1).reshape(B * C, L)
        fc = z.new_zeros(B * C, self.H)
        for blk in self.blocks:
            b, f = blk(z)
            z = z - b
            fc = fc + f
        y = fc.reshape(B, C, self.H).permute(0, 2, 1)
        return self.rev(y, "denorm")


# ============================================================ LSTM / TCN
class LSTMForecaster(nn.Module):
    def __init__(self, L, H, C, hidden=128, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.rev = RevIN(C)
        self.H = H
        self.rnn = nn.LSTM(1, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, H)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        z = self.rev(x, "norm").permute(0, 2, 1).reshape(B * C, L, 1)
        o, _ = self.rnn(z)
        y = self.head(o[:, -1]).reshape(B, C, self.H).permute(0, 2, 1)
        return self.rev(y, "denorm")


class Chomp(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        return x[..., :-self.s] if self.s > 0 else x


class TCN(nn.Module):
    def __init__(self, L, H, C, ch=64, levels=5, kernel=3, dropout=0.1, **kw):
        super().__init__()
        self.rev = RevIN(C); self.H = H
        layers, cin = [], 1
        for i in range(levels):
            d = 2 ** i
            pad = (kernel - 1) * d
            layers += [nn.Conv1d(cin, ch, kernel, padding=pad, dilation=d), Chomp(pad),
                       nn.GELU(), nn.Dropout(dropout)]
            cin = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(ch, H)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        z = self.rev(x, "norm").permute(0, 2, 1).reshape(B * C, 1, L)
        h = self.net(z)[..., -1]
        y = self.head(h).reshape(B, C, self.H).permute(0, 2, 1)
        return self.rev(y, "denorm")


# ============================================================ embeddings for enc-dec transformers
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, n_tf=5):
        super().__init__()
        self.tok = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tok.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.tim = nn.Linear(n_tf, d_model, bias=False)
        pe = torch.zeros(5000, d_model)
        pos = torch.arange(5000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, tf):
        h = self.tok(x.permute(0, 2, 1)).transpose(1, 2) + self.pe[:, :x.shape[1]]
        if tf is not None:
            h = h + self.tim(tf)
        return self.drop(h)


# ============================================================ Informer
class ProbAttention(nn.Module):
    """ProbSparse self-attention (Zhou et al., AAAI 2021)."""

    def __init__(self, mask_flag=False, factor=5, dropout=0.1):
        super().__init__()
        self.mask_flag, self.factor = mask_flag, factor
        self.drop = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, n_top, n_sample):
        B, Hh, Lk, E = K.shape
        Lq = Q.shape[2]
        idx = torch.randint(Lk, (Lq, n_sample), device=K.device)
        K_s = K.unsqueeze(-3).expand(B, Hh, Lq, Lk, E)[:, :, torch.arange(Lq).unsqueeze(1), idx, :]
        Q_K_s = torch.matmul(Q.unsqueeze(-2), K_s.transpose(-2, -1)).squeeze(-2)
        M = Q_K_s.max(-1)[0] - Q_K_s.sum(-1) / Lk
        M_top = M.topk(n_top, sorted=False)[1]
        Q_red = Q[torch.arange(B)[:, None, None], torch.arange(Hh)[None, :, None], M_top, :]
        return torch.matmul(Q_red, K.transpose(-2, -1)), M_top

    def forward(self, Q, K, V):
        B, Lq, Hh, E = Q.shape
        Lk = K.shape[1]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        U_part = min(Lk, int(self.factor * math.ceil(math.log(max(Lk, 2)))))
        u = min(Lq, int(self.factor * math.ceil(math.log(max(Lq, 2)))))
        scores_top, index = self._prob_QK(Q, K, u, U_part)
        scores_top = scores_top / math.sqrt(E)
        if self.mask_flag:
            ctx = V.cumsum(-2)
            mask = torch.ones(Lq, Lk, device=Q.device, dtype=torch.bool).triu(1)
            m = mask[index.reshape(-1)].reshape(B, Hh, u, Lk)
            scores_top = scores_top.masked_fill(m, -torch.inf)
        else:
            ctx = V.mean(-2, keepdim=True).expand(B, Hh, Lq, V.shape[-1]).clone()
        attn = self.drop(torch.softmax(scores_top, -1))
        upd = torch.matmul(attn, V)
        ctx[torch.arange(B)[:, None, None], torch.arange(Hh)[None, :, None], index, :] = upd.type_as(ctx)
        return ctx.transpose(1, 2).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, inner, d_model, n_heads):
        super().__init__()
        self.inner, self.h = inner, n_heads
        dk = d_model // n_heads
        self.q = nn.Linear(d_model, dk * n_heads); self.k = nn.Linear(d_model, dk * n_heads)
        self.v = nn.Linear(d_model, dk * n_heads); self.o = nn.Linear(dk * n_heads, d_model)

    def forward(self, q, k, v):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        q = self.q(q).view(B, Lq, self.h, -1)
        k = self.k(k).view(B, Lk, self.h, -1)
        v = self.v(v).view(B, Lk, self.h, -1)
        out = self.inner(q, k, v).view(B, Lq, -1)
        return self.o(out)


class EncLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff, dropout, act="gelu"):
        super().__init__()
        self.attn = attn
        self.c1 = nn.Conv1d(d_model, d_ff, 1); self.c2 = nn.Conv1d(d_ff, d_model, 1)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = F.gelu if act == "gelu" else F.relu

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x, x, x)))
        y = self.drop(self.act(self.c1(x.transpose(-1, 1))))
        y = self.drop(self.c2(y).transpose(-1, 1))
        return self.n2(x + y)


class ConvDistill(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c = nn.Conv1d(d, d, 3, padding=1, padding_mode="circular")
        self.n = nn.BatchNorm1d(d)
        self.p = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return self.p(F.elu(self.n(self.c(x.permute(0, 2, 1))))).transpose(1, 2)


class DecLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff, dropout):
        super().__init__()
        self.sa, self.ca = self_attn, cross_attn
        self.c1 = nn.Conv1d(d_model, d_ff, 1); self.c2 = nn.Conv1d(d_ff, d_model, 1)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem):
        x = self.n1(x + self.drop(self.sa(x, x, x)))
        x = self.n2(x + self.drop(self.ca(x, mem, mem)))
        y = self.drop(F.gelu(self.c1(x.transpose(-1, 1))))
        y = self.drop(self.c2(y).transpose(-1, 1))
        return self.n3(x + y)


class Informer(nn.Module):
    def __init__(self, L, H, C, label_len=None, d_model=256, n_heads=8, e_layers=2,
                 d_layers=1, d_ff=512, dropout=0.1, factor=5, distil=True, revin=False, **kw):
        super().__init__()
        self.L, self.H = L, H
        self.rev = RevIN(C) if revin else _NoRev()   # revin=False reproduces the reported records
        self.label_len = label_len or min(L // 2, 168)
        self.enc_emb = DataEmbedding(C, d_model, dropout)
        self.dec_emb = DataEmbedding(C, d_model, dropout)
        self.enc_layers = nn.ModuleList([
            EncLayer(AttentionLayer(ProbAttention(False, factor, dropout), d_model, n_heads),
                     d_model, d_ff, dropout) for _ in range(e_layers)])
        self.distil = nn.ModuleList([ConvDistill(d_model) for _ in range(e_layers - 1)]) if distil else None
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_layers = nn.ModuleList([
            DecLayer(AttentionLayer(ProbAttention(True, factor, dropout), d_model, n_heads),
                     AttentionLayer(ProbAttention(False, factor, dropout), d_model, n_heads),
                     d_model, d_ff, dropout) for _ in range(d_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, C)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        x = self.rev(x, "norm")
        ll = self.label_len
        tfx = tf[:, :L] if tf is not None else None
        tfy = tf[:, L - ll:] if tf is not None else None
        dec_in = torch.cat([x[:, -ll:], x.new_zeros(B, self.H, C)], 1)
        e = self.enc_emb(x, tfx)
        for i, lyr in enumerate(self.enc_layers):
            e = lyr(e)
            if self.distil is not None and i < len(self.distil):
                e = self.distil[i](e)
        e = self.enc_norm(e)
        d = self.dec_emb(dec_in, tfy)
        for lyr in self.dec_layers:
            d = lyr(d, e)
        return self.rev(self.proj(self.dec_norm(d))[:, -self.H:], "denorm")


# ============================================================ Autoformer
class AutoCorrelation(nn.Module):
    def __init__(self, factor=3, dropout=0.1):
        super().__init__()
        self.factor = factor
        self.drop = nn.Dropout(dropout)

    def time_delay_agg(self, values, corr):
        B, Hh, E, L = values.shape
        top_k = max(1, int(self.factor * math.log(max(L, 2))))
        mean_corr = corr.mean(1).mean(1)                            # (B, L)
        idx = torch.topk(mean_corr, top_k, dim=-1)[1]               # (B, k)
        w = torch.stack([mean_corr[torch.arange(B), idx[:, i]] for i in range(top_k)], -1)
        w = torch.softmax(w, -1)
        init = values.repeat(1, 1, 1, 2)
        out = torch.zeros_like(values)
        ar = torch.arange(L, device=values.device)
        for i in range(top_k):
            shift = idx[:, i].view(B, 1, 1, 1) + ar.view(1, 1, 1, L)
            g = torch.gather(init, -1, shift.expand(B, Hh, E, L))
            out = out + g * w[:, i].view(B, 1, 1, 1)
        return out

    def forward(self, q, k, v):
        B, Lq, Hh, E = q.shape
        Lk = k.shape[1]
        if Lq > Lk:
            pad = torch.zeros_like(q[:, :Lq - Lk])
            k = torch.cat([k, pad], 1); v = torch.cat([v, pad], 1)
        else:
            k, v = k[:, :Lq], v[:, :Lq]
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=Lq, dim=-1)
        out = self.time_delay_agg(v.permute(0, 2, 3, 1).contiguous(), corr)
        return out.permute(0, 3, 1, 2).contiguous()


class AutoEncLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff, kernel, dropout):
        super().__init__()
        self.attn = attn
        self.c1 = nn.Conv1d(d_model, d_ff, 1, bias=False); self.c2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        self.d1, self.d2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(x, x, x))
        x, _ = self.d1(x)
        y = self.drop(F.gelu(self.c1(x.transpose(-1, 1))))
        y = self.drop(self.c2(y).transpose(-1, 1))
        res, _ = self.d2(x + y)
        return res


class AutoDecLayer(nn.Module):
    def __init__(self, sa, ca, d_model, C, d_ff, kernel, dropout):
        super().__init__()
        self.sa, self.ca = sa, ca
        self.c1 = nn.Conv1d(d_model, d_ff, 1, bias=False); self.c2 = nn.Conv1d(d_ff, d_model, 1, bias=False)
        self.d1, self.d2, self.d3 = (SeriesDecomp(kernel) for _ in range(3))
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(d_model, C, 3, padding=1, padding_mode="circular", bias=False)

    def forward(self, x, mem):
        x = x + self.drop(self.sa(x, x, x)); x, t1 = self.d1(x)
        x = x + self.drop(self.ca(x, mem, mem)); x, t2 = self.d2(x)
        y = self.drop(F.gelu(self.c1(x.transpose(-1, 1))))
        y = self.drop(self.c2(y).transpose(-1, 1))
        x, t3 = self.d3(x + y)
        t = self.proj((t1 + t2 + t3).permute(0, 2, 1)).transpose(1, 2)
        return x, t


class Autoformer(nn.Module):
    def __init__(self, L, H, C, label_len=None, d_model=256, n_heads=8, e_layers=2,
                 d_layers=1, d_ff=512, dropout=0.1, factor=3, kernel=25, revin=False, **kw):
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.rev = RevIN(C) if revin else _NoRev()   # revin=False reproduces the reported records
        self.label_len = label_len or min(L // 2, 168)
        self.decomp = SeriesDecomp(kernel)
        self.enc_emb = DataEmbedding(C, d_model, dropout)
        self.dec_emb = DataEmbedding(C, d_model, dropout)
        self.enc = nn.ModuleList([AutoEncLayer(AttentionLayer(AutoCorrelation(factor, dropout), d_model, n_heads),
                                               d_model, d_ff, kernel, dropout) for _ in range(e_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec = nn.ModuleList([AutoDecLayer(AttentionLayer(AutoCorrelation(factor, dropout), d_model, n_heads),
                                               AttentionLayer(AutoCorrelation(factor, dropout), d_model, n_heads),
                                               d_model, C, d_ff, kernel, dropout) for _ in range(d_layers)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, C, bias=True)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        x = self.rev(x, "norm")
        ll, H = self.label_len, self.H
        mean = x.mean(1, keepdim=True).repeat(1, H, 1)
        seas_init, trend_init = self.decomp(x[:, -ll:])
        dec_seas = torch.cat([seas_init, x.new_zeros(B, H, C)], 1)
        dec_tr = torch.cat([trend_init, mean], 1)
        tfx = tf[:, :L] if tf is not None else None
        tfy = tf[:, L - ll:] if tf is not None else None
        e = self.enc_emb(x, tfx)
        for lyr in self.enc:
            e = lyr(e)
        e = self.enc_norm(e)
        d = self.dec_emb(dec_seas, tfy)
        t = dec_tr
        for lyr in self.dec:
            d, dt = lyr(d, e)
            t = t + dt
        return self.rev((self.proj(self.dec_norm(d)) + t)[:, -H:], "denorm")


# ============================================================ calendar-conditioned PatchTST
class CalTST(nn.Module):
    """PatchTST given exactly the calendar information PLANET's solver uses, but learned
    globally instead of solved per window.

    This is the control that separates "the model needed the calendar" from "the model
    needed the closed-form, instance-adaptive fit of the calendar". The same absolute-time
    design matrix Phi is patched and embedded: past-basis patches are added to the series
    tokens, future-basis patches are appended as extra tokens the head can read.
    """

    def __init__(self, L, H, C, d_model=128, n_layers=3, n_heads=8, d_ff=256,
                 patch=16, stride=8, dropout=0.2, K=30, out_mult=1, **kw):
        super().__init__()
        self.rev = RevIN(C)
        self.patch, self.stride, self.L, self.H, self.K = patch, stride, L, H, K
        # out_mult > 1 widens only the output head (H*out_mult values per channel, e.g. one
        # block of H per quantile level); the token layout and the calendar design matrix
        # keep the true L and H, which widening H itself would break.
        self.out_mult = out_mult
        self.n_l = (L - patch) // stride + 1
        self.n_h = max(1, (H - patch) // stride + 1) if H >= patch else 1
        self.h_pad = max(0, patch - H)
        self.embed = nn.Linear(patch, d_model)
        self.cal_past = nn.Linear(patch * K, d_model)
        self.cal_fut = nn.Linear(patch * K, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_l + self.n_h, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout),
                                  nn.Linear((self.n_l + self.n_h) * d_model, H * out_mult))

    def _cal_patches(self, P):                       # P: (B, T, K) -> (B, n, patch*K)
        B, T, K = P.shape
        if T < self.patch:
            P = torch.cat([P, P[:, -1:].expand(B, self.patch - T, K)], 1)
        z = P.permute(0, 2, 1).unfold(-1, self.patch, self.stride)     # (B, K, n, patch)
        return z.permute(0, 2, 1, 3).reshape(B, z.shape[2], K * self.patch)

    def forward(self, x, Phi=None, Psi=None, tf=None):
        B, L, C = x.shape
        z = self.rev(x, "norm").permute(0, 2, 1).reshape(B * C, L)
        z = self.embed(z.unfold(-1, self.patch, self.stride))          # (B*C, n_l, d)
        if Phi is not None and Phi.dim() == 3 and Phi.shape[-1] == self.K:
            cp = self.cal_past(self._cal_patches(Phi[:, :L]))[:, :self.n_l]
            cf = self.cal_fut(self._cal_patches(Phi[:, L:]))[:, :self.n_h]
            z = z + cp.repeat_interleave(C, 0)
            z = torch.cat([z, cf.repeat_interleave(C, 0)], 1)
        else:                                                          # no basis supplied
            z = torch.cat([z, z.new_zeros(B * C, self.n_h, z.shape[-1])], 1)
        z = self.enc(self.drop(z + self.pos))
        y = self.head(z).reshape(B, C, self.H * self.out_mult).permute(0, 2, 1)
        return self.rev(y, "denorm")


# ============================================================ registry
REGISTRY = {
    "SeasonalNaive": SeasonalNaive, "DLinear": DLinear, "NLinear": NLinear,
    "RLinear": RLinear, "FITS": FITS, "PatchTST": PatchTST, "iTransformer": ITransformer,
    "TimeMixer": TimeMixer, "NBEATS": NBEATS, "LSTM": LSTMForecaster, "TCN": TCN,
    "Informer": Informer, "Autoformer": Autoformer, "CalTST": CalTST,
}
