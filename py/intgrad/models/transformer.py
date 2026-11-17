"""Byte transformer for plan/001 M3 (fp first; quantization arrives in M4).

Decoder-only, pre-RMSNorm, RoPE, QK-norm (RMSNorm + gain on q and k per
head), causal SDPA attention, ReLU² MLP (4d), tied 256 x d byte
embedding / output head, no biases. Magnitude control (001-D9): QK-norm,
z-loss on the output logits, optional logit soft-cap c*tanh(x/c).

Range telemetry: forward(x, tap=Tap()) records every tensor class against
the byte grids (would-clamp / would-underflow rates, |x| percentiles in
octaves); attention scores are computed explicitly for one sequence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from intgrad import quant as Q

PRESETS = {
    "XS": dict(d=128, L=2, H=4, T=256),
    "S": dict(d=256, L=4, H=4, T=512),
    "M": dict(d=384, L=6, H=6, T=1024),
    "L": dict(d=512, L=8, H=8, T=1024),
}

BYTE_MAX = float(Q.DEC8[Q.MAX_CODE] / Q.SCALE)      # 25.99: byte-grid clamp
BYTE_MIN = float(Q.THR8[1] / Q.SCALE)               # 1.38e-4: below -> 0
FX_MAX = float(Q.FXSTEP * 127)                      # 12.7: fixed-point clamp
FX_CLASSES = ("scores", "logits")                   # live on the fx grid


@dataclass
class Tap:
    """Collects per-tensor-class range statistics during one forward."""
    max_samples: int = 1 << 20
    stats: dict = field(default_factory=dict)

    @torch.no_grad()
    def __call__(self, name: str, x: torch.Tensor) -> None:
        a = x.detach().float().abs().reshape(-1)
        if a.numel() > self.max_samples:
            idx = torch.randint(0, a.numel(), (self.max_samples,), device=a.device)
            a = a[idx]
        nz = a[a > 0]
        q = torch.tensor([0.01, 0.5, 0.99, 0.999], device=a.device)
        pct = torch.log2(torch.quantile(nz, q)).tolist() if nz.numel() else [float("nan")] * 4
        fx = any(name.startswith(c) for c in FX_CLASSES)
        self.stats[name] = {
            "absmax": a.max().item(),
            "log2_p01": round(pct[0], 2), "log2_p50": round(pct[1], 2),
            "log2_p99": round(pct[2], 2), "log2_p999": round(pct[3], 2),
            "clamp": (a > (FX_MAX if fx else BYTE_MAX)).float().mean().item(),
            "underflow": 0.0 if fx else ((a > 0) & (a < BYTE_MIN)).float().mean().item(),
        }


def rope_cache(T: int, dh: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.arange(T, device=device).float()[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)                 # [T, dh/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T], sin[:T]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class CausalConv(nn.Module):
    """1-D convolution over time on [B, T, C] that only looks backward:
    output t depends on inputs t - (k-1)*dilation … t (left zero padding)."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int = 1,
                 depthwise: bool = False, identity_init: bool = False):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation, bias=False,
                              groups=cin if depthwise else 1)
        if identity_init:                       # starts as a no-op (depthwise)
            with torch.no_grad():
                self.conv.weight.zero_()
                self.conv.weight[:, :, -1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0)))
        return y.transpose(1, 2)


class Block(nn.Module):
    def __init__(self, d: int, H: int, L: int, qk_norm: bool = True,
                 qkv_conv: int = 0):
        super().__init__()
        self.H, self.dh = H, d // H
        self.qk_norm = qk_norm
        # Primer-style causal depthwise conv on q, k, v (kernel qkv_conv)
        self.qkv_conv = CausalConv(3 * d, 3 * d, qkv_conv, depthwise=True,
                                   identity_init=True) if qkv_conv else None
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = nn.RMSNorm(self.dh)
        self.k_norm = nn.RMSNorm(self.dh)
        self.o = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)
        for lin in (self.qkv, self.up):
            nn.init.normal_(lin.weight, std=0.02)
        for lin in (self.o, self.down):                   # residual branches
            nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * L))

    def forward(self, x, rope, tap: Tap | None = None, i: int = 0):
        B, T, d = x.shape
        h = self.n1(x)
        qkv = self.qkv(h)
        if self.qkv_conv is not None:
            qkv = self.qkv_conv(qkv)
        q, k, v = qkv.view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, d)
        if tap is not None:
            tap(f"norm{i}.attn_in", h)
            tap(f"q{i}", q)
            tap(f"k{i}", k)
            tap(f"v{i}", v)
            s = (q[:1] @ k[:1].transpose(-1, -2)) / math.sqrt(self.dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            tap(f"scores{i}", s[..., mask])
            tap(f"attn_out{i}", a)
        x = x + self.o(a)
        h = self.n2(x)
        u = self.up(h)
        m = torch.relu(u).square()
        if tap is not None:
            tap(f"norm{i}.mlp_in", h)
            tap(f"mlp_pre{i}", u)
            tap(f"mlp_post{i}", m)
        x = x + self.down(m)
        if tap is not None:
            tap(f"resid{i}", x)
        return x


class ByteTransformer(nn.Module):
    def __init__(self, d: int, L: int, H: int, T: int, zloss: float = 1e-4,
                 softcap: float = 0.0, qk_norm: bool = True,
                 stem: str = "embed", stem_dim: int = 32, qkv_conv: int = 0):
        """stem "embed": tied 256 x d embedding / output head (baseline).
        stem "conv": 256 x stem_dim embedding -> causal conv stem_dim -> d
        (kernel 4) -> + causal depthwise conv (kernel 4, dilation 2) on
        GELU(h); receptive field 7 bytes; untied d -> 256 output head.
        qkv_conv k > 0: Primer-style causal depthwise conv on q, k, v."""
        super().__init__()
        self.d, self.L, self.H, self.T = d, L, H, T
        self.zloss, self.softcap, self.stem = zloss, softcap, stem
        if stem == "embed":
            self.emb = nn.Embedding(256, d)
            nn.init.normal_(self.emb.weight, std=0.02)
        elif stem == "conv":
            self.emb = nn.Embedding(256, stem_dim)
            nn.init.normal_(self.emb.weight, std=1.0)
            self.stem_up = CausalConv(stem_dim, d, 4)
            # output std ≈ 0.02, like the baseline embedding
            nn.init.normal_(self.stem_up.conv.weight, std=0.02 / math.sqrt(stem_dim * 4))
            self.stem_dw = CausalConv(d, d, 4, dilation=2, depthwise=True)
            nn.init.normal_(self.stem_dw.conv.weight, std=0.02)
            self.head = nn.Linear(d, 256, bias=False)
            nn.init.normal_(self.head.weight, std=0.02)
        else:
            raise ValueError(f"unknown stem {stem!r}")
        self.blocks = nn.ModuleList(Block(d, H, L, qk_norm, qkv_conv) for _ in range(L))
        self.nf = nn.RMSNorm(d)
        self._rope = None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x.long())
        if self.stem == "embed":
            return e
        h = self.stem_up(e)
        return h + self.stem_dw(F.gelu(h))

    def head_weight(self) -> torch.Tensor:
        return self.emb.weight if self.stem == "embed" else self.head.weight

    def rope(self, device):
        if self._rope is None or self._rope[0].device != torch.device(device):
            self._rope = rope_cache(self.T, self.d // self.H, device)
        return self._rope

    @property
    def mid(self) -> int:
        """Index of the first upper block: the latent interface sits
        between blocks[mid-1] and blocks[mid] (project direction: recurrence
        and memory attach to the latent space in the middle of the stack)."""
        return self.L // 2

    def forward(self, x: torch.Tensor, tap: Tap | None = None,
                mid_hook=None) -> torch.Tensor:
        """x: [B, T] bytes -> logits [B, T, 256] (float32).
        mid_hook(h_mid) -> h_mid' may read or replace the mid-stack latent
        [B, T, d] (plan 002: recurrence, memory)."""
        h = self.embed(x)
        if tap is not None:
            tap("emb", h)
        rope = self.rope(x.device)
        for i, blk in enumerate(self.blocks):
            if i == self.mid:
                if tap is not None:
                    tap("h_mid", h)
                if mid_hook is not None:
                    h = mid_hook(h)
            h = blk(h, rope, tap, i)
        logits = F.linear(self.nf(h), self.head_weight()).float()
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if tap is not None:
            tap("logits", logits)
            for name, p in self.named_parameters():
                tap(f"w.{name}", p)
        return logits

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(total, ce): ce in nats per byte; total adds zloss * lse²."""
        lg = logits.reshape(-1, 256)
        ce = F.cross_entropy(lg, y.reshape(-1).long())
        if self.zloss:
            return ce + self.zloss * torch.logsumexp(lg, -1).square().mean(), ce
        return ce, ce

    @torch.no_grad()
    def chunk_bits(self, chunks: torch.Tensor, batch: int = 64) -> torch.Tensor:
        """Bits for chunks[:, 1:] (eval-v1 protocol). Sliding windows with
        stride T/2: each target is scored once, in the earliest window that
        contains it, so it sees between T/2 and T bytes of context (all of
        its in-chunk context if it is within the first window)."""
        N, Lc = chunks.shape
        T, stride = self.T, self.T // 2
        n_in = Lc - 1                                       # inputs 0..L-2
        if n_in <= T:
            starts = [0]
        else:
            starts = list(range(0, n_in - T + 1, stride))
            if starts[-1] + T < n_in:
                starts.append(n_in - T)
        out = torch.empty(N, Lc - 1, device=chunks.device)
        covered = 0
        for s in starts:
            w = min(T, n_in - s)
            lo = max(covered, s)                            # first new target - 1
            for b in range(0, N, batch):
                xb = chunks[b:b + batch, s:s + w]
                yb = chunks[b:b + batch, s + 1:s + w + 1].long()
                lg = self(xb)
                ce = F.cross_entropy(lg.reshape(-1, 256), yb.reshape(-1),
                                     reduction="none").view(yb.shape)
                out[b:b + batch, lo:s + w] = ce[:, lo - s:] / math.log(2)
            covered = s + w
        return out

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, n: int, temp: float = 0.8,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """prompt [B, P] bytes -> [B, n] sampled continuation bytes."""
        x = prompt.long()
        out = []
        for _ in range(n):
            lg = self(x[:, -self.T:])[:, -1] / temp
            nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator)
            out.append(nxt)
            x = torch.cat([x, nxt], 1)
        return torch.cat(out, 1).to(torch.uint8)


def build(preset: str | None = None, **kw) -> ByteTransformer:
    cfg = dict(PRESETS[preset]) if preset else {}
    cfg.update(kw)
    return ByteTransformer(**cfg)
