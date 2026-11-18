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

    @torch.no_grad()
    def rel(self, name: str, x: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        """Max-relative range (softmax inputs): fraction of valid entries more
        than 25.5 (an unsigned byte at step 0.1) below their row max."""
        x = x.detach().float()
        if valid is not None:
            x = x.masked_fill(~valid, float("-inf"))
        d = x.amax(-1, keepdim=True) - x
        ok = torch.isfinite(d)
        self.stats.setdefault(name, {})["relclamp"] = (
            (d[ok] > float(Q.FXSTEP * 255)).float().mean().item())


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
        if self.conv.groups == self.conv.in_channels == self.conv.out_channels:
            return self._depthwise(x)
        y = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0)))
        return y.transpose(1, 2)

    def _depthwise(self, x: torch.Tensor) -> torch.Tensor:
        """Same result as the Conv1d path, as k shifted multiply-adds on
        [B, T, C] (no transposes; Conv1d is slow for depthwise kernels)."""
        w = self.conv.weight[:, 0, :]                    # [C, k]
        k, dil, T = w.shape[1], self.conv.dilation[0], x.shape[1]
        xp = F.pad(x, (0, 0, self.pad, 0))               # pad time on the left
        y = xp[:, self.pad:self.pad + T] * w[:, -1]
        for j in range(k - 1):                           # tap j looks back (k-1-j)*dil
            off = self.pad - (k - 1 - j) * dil
            y = y + xp[:, off:off + T] * w[:, j]
        return y


class QSite(nn.Module):
    """One activation site on the byte grid with a delayed power-of-two
    scale: the scale comes from the amax history of the last `hist`
    training forwards times `headroom` (no pass over the current tensor,
    causal, and fixed at eval). Values above the window clamp (STE gradient
    0 there); the window reaches 17.5 octaves down. Before any history
    exists, the current amax is used."""

    def __init__(self, hist: int = 16, headroom: float = 2.0):
        super().__init__()
        self.register_buffer("hist", torch.zeros(hist))
        self.register_buffer("clamp", torch.zeros(()))   # last training clamp rate
        self.headroom = headroom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = x.detach().abs().amax().float()
        hm = self.hist.max()
        s = Q.pow2_scale(torch.where(hm > 0, hm * self.headroom, cur))
        if self.training:
            with torch.no_grad():
                self.hist.copy_(torch.cat([self.hist[1:], cur.view(1)]))
                self.clamp.copy_((x.detach().abs() * s > Q.TOP8).float().mean())
        return Q._QActT.apply(x, s)


def _sites(names: str, on: bool) -> nn.ModuleDict:
    return nn.ModuleDict({n: QSite() for n in names.split()} if on else {})


def _q(sites: nn.ModuleDict, name: str, x: torch.Tensor) -> torch.Tensor:
    return sites[name](x) if name in sites else x


@dataclass(frozen=True)
class QAT:
    """Quantization-aware forward (plan/001 M4). Weights are quantized by
    their storage (Lion master codes, quant.enabled); this covers the rest.
    acts:    activation sites on the byte grid at per-tensor pow2 absmax
    resid:   "fp" (residual stream wide) or "q8" (byte grid after each add)
    softmax: "none" or "maxrel" (attention scores and logits, 001-D12)
    probs:   attention probabilities on the byte grid at scale 4096"""
    acts: bool = False
    resid: str = "fp"
    softmax: str = "none"
    probs: bool = False

    def __post_init__(self):
        if self.resid not in ("fp", "q8") or self.softmax not in ("none", "maxrel"):
            raise ValueError(f"bad QAT config {self}")

    @property
    def explicit_attn(self) -> bool:
        return self.softmax != "none" or self.probs


class Block(nn.Module):
    def __init__(self, d: int, H: int, L: int, qk_norm: bool = True,
                 qkv_conv: int = 0, qat: QAT = QAT(), window: int = 0, patch: int = 0):
        """window > 0: sliding-window causal attention (each position sees
        the last `window` positions, itself included); 0 = full causal.
        patch > 0: causal attention only inside aligned blocks of `patch`
        positions (MEGABYTE-style local model; O(T * patch) scores)."""
        super().__init__()
        self.qat = qat
        self.window, self.patch = window, patch
        self.qs = _sites("n1 q k v attn n2 up mlp", qat.acts)
        self.rs = _sites("r1 r2", qat.resid == "q8")
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

    def attend(self, q, k, v, mask=None):
        """Causal attention; explicit when scores or probabilities are
        quantized (SDPA never materializes them). mask: optional bool
        [B, 1, T, T] (True = may attend), replacing the causal mask; or a
        tuple (seg [B, T], W): attend only to earlier positions of the same
        segment, all within the last W positions (local, O(T·W))."""
        if isinstance(mask, tuple):
            return self._attend_local(q, k, v, *mask)
        if mask is not None:
            return self._attend(q, k, v, mask)
        if self.patch:
            B, H, T, dh = q.shape
            p = self.patch
            blk = lambda t: t.reshape(B, H, T // p, p, dh)
            return self._attend(blk(q), blk(k), blk(v)).reshape(B, H, T, dh)
        return self._attend(q, k, v)

    def _attend(self, q, k, v, causal=None):
        qat = self.qat
        T = q.shape[-2]
        if self.window:
            i = torch.arange(T, device=q.device)
            dist = i[:, None] - i[None, :]
            causal = (dist >= 0) & (dist < self.window)
        if not qat.explicit_attn:
            if causal is None:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        if causal is None:
            causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.dh))
        s = s.masked_fill(~causal, float("-inf"))
        if qat.softmax == "maxrel":
            s = Q.logit_maxrel(s).masked_fill(~causal, float("-inf"))
        p = torch.softmax(s, -1)
        if qat.probs:
            p = Q.act_fixed(p, Q.PROB_SCALE)
        return p @ v

    def _attend_local(self, q, k, v, seg, W):
        """Segment-causal attention over a band of the last W positions:
        same result as the dense segment mask when no segment is longer
        than W, at O(T·W) instead of O(T²)."""
        qat = self.qat
        win = lambda t: F.pad(t, (0, 0, W - 1, 0)).unfold(2, W, 1)       # [B, H, T, dh, W]
        kw, vw = win(k), win(v)
        sp = F.pad(seg, (W - 1, 0), value=-1).unfold(1, W, 1)            # [B, T, W]
        ok = (sp == seg[..., None])[:, None]                             # [B, 1, T, W]
        s = torch.einsum("bhtd,bhtdw->bhtw", q, kw) * (1.0 / math.sqrt(self.dh))
        s = s.masked_fill(~ok, float("-inf"))
        if qat.softmax == "maxrel":
            s = Q.logit_maxrel(s).masked_fill(~ok, float("-inf"))
        p = torch.softmax(s, -1)
        if qat.probs:
            p = Q.act_fixed(p, Q.PROB_SCALE)
        return torch.einsum("bhtw,bhtdw->bhtd", p, vw)

    def forward(self, x, rope, tap: Tap | None = None, i: int = 0, mask=None):
        B, T, d = x.shape
        qs, rs = self.qs, self.rs
        h = _q(qs, "n1", self.n1(x))
        qkv = self.qkv(h)
        if self.qkv_conv is not None:
            qkv = self.qkv_conv(qkv)
        q, k, v = qkv.view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        q, k, v = _q(qs, "q", q), _q(qs, "k", k), _q(qs, "v", v)
        a = _q(qs, "attn", self.attend(q, k, v, mask).transpose(1, 2).reshape(B, T, d))
        if tap is not None:
            tap(f"norm{i}.attn_in", h)
            tap(f"q{i}", q)
            tap(f"k{i}", k)
            tap(f"v{i}", v)
            s = (q[:1] @ k[:1].transpose(-1, -2)) / math.sqrt(self.dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            tap(f"scores{i}", s[..., mask])
            tap.rel(f"scores{i}", s, mask)
            tap(f"attn_out{i}", a)
        x = _q(rs, "r1", x + self.o(a))
        h = _q(qs, "n2", self.n2(x))
        u = _q(qs, "up", self.up(h))
        m = _q(qs, "mlp", torch.relu(u).square())
        if tap is not None:
            tap(f"norm{i}.mlp_in", h)
            tap(f"mlp_pre{i}", u)
            tap(f"mlp_post{i}", m)
        x = _q(rs, "r2", x + self.down(m))
        if tap is not None:
            tap(f"resid{i}", x)
        return x


class ByteLM(nn.Module):
    """Shared by every byte LM here: [B, T] bytes -> logits [B, T, 256].
    Subclasses set T, H, qat, zloss and implement forward(x, tap, mid_hook)."""

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(total, ce): ce in nats per byte; total adds zloss * lse²."""
        lg = logits.reshape(-1, 256)
        ce = F.cross_entropy(lg, y.reshape(-1).long())
        if self.zloss:
            return ce + self.zloss * torch.logsumexp(lg, -1).square().mean(), ce
        return ce, ce

    def max_batch(self, cap: int = 64) -> int:
        """Inference batch: explicit attention keeps <= 2^27 scores per tensor."""
        b = cap
        if self.qat.explicit_attn:
            while b > 1 and b * self.attn_elems() > 1 << 27:
                b //= 2
        return b

    def attn_elems(self) -> int:
        """Attention scores per sequence and layer (H x Tq x Tk)."""
        return self.H * self.T * self.T

    @torch.no_grad()
    def chunk_bits(self, chunks: torch.Tensor, batch: int = 0) -> torch.Tensor:
        """Bits for chunks[:, 1:] (eval-v1 protocol). Sliding windows with
        stride T/2: each target is scored once, in the earliest window that
        contains it, so it sees between T/2 and T bytes of context (all of
        its in-chunk context if it is within the first window)."""
        N, Lc = chunks.shape
        T, stride = self.T, self.T // 2
        batch = batch or self.max_batch()
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
        """prompt [B, P] bytes -> [B, n] sampled continuation bytes
        (in batches of max_batch(), one shared generator)."""
        mb = self.max_batch()
        if len(prompt) > mb:
            return torch.cat([self.sample(prompt[i:i + mb], n, temp, generator)
                              for i in range(0, len(prompt), mb)])
        was = self.training
        self.eval()                       # frozen quantization scales
        x = prompt.long()
        out = []
        for _ in range(n):
            lg = self(x[:, -self.T:])[:, -1] / temp
            nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator)
            out.append(nxt)
            x = torch.cat([x, nxt], 1)
        self.train(was)
        return torch.cat(out, 1).to(torch.uint8)


class ByteTransformer(ByteLM):
    def __init__(self, d: int, L: int, H: int, T: int, zloss: float = 1e-4,
                 softcap: float = 0.0, qk_norm: bool = True,
                 stem: str = "embed", stem_dim: int = 32, qkv_conv: int = 0,
                 logit_quant: str = "none", qat: QAT | dict | None = None):
        """stem "embed": tied 256 x d embedding / output head (baseline).
        stem "conv": 256 x stem_dim embedding -> causal conv stem_dim -> d
        (kernel 4) -> + causal depthwise conv (kernel 4, dilation 2) on
        GELU(h); receptive field 7 bytes; untied d -> 256 output head.
        qkv_conv k > 0: Primer-style causal depthwise conv on q, k, v."""
        super().__init__()
        self.d, self.L, self.H, self.T = d, L, H, T
        self.zloss, self.softcap, self.stem = zloss, softcap, stem
        if logit_quant not in ("none", "fx", "maxrel"):
            raise ValueError(f"logit_quant {logit_quant!r}")
        self.logit_quant = logit_quant      # logits-only QAT (X12)
        self.qat = qat = QAT(**qat) if isinstance(qat, dict) else (qat or QAT())
        if qat.softmax == "maxrel":
            self.logit_quant = "maxrel"
        self.qs = _sites("emb nf", qat.acts)
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
        self.blocks = nn.ModuleList(Block(d, H, L, qk_norm, qkv_conv, qat) for _ in range(L))
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
        h = _q(self.qs, "emb", self.embed(x))
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
        logits = F.linear(_q(self.qs, "nf", self.nf(h)), self.head_weight()).float()
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if self.logit_quant == "fx":
            logits = Q.logit_fx(logits)
        elif self.logit_quant == "maxrel":
            logits = Q.logit_maxrel(logits)
        if tap is not None:
            tap("logits", logits)
            tap.rel("logits", logits)
            for name, p in self.named_parameters():
                tap(f"w.{name}", p)
        return logits


def build(preset: str | None = None, **kw) -> ByteTransformer:
    cfg = dict(PRESETS[preset]) if preset else {}
    cfg.update(kw)
    return ByteTransformer(**cfg)
