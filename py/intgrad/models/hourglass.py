"""Hourglass byte LM (plan/001 M6): byte encoder -> downsample by k ->
global transformer on T/k patches -> shifted upsample -> byte decoder.

    bytes [B, T] -> emb (256 x d_b) -> enc ByteBlocks (causal conv + MLP)
         -> patches: k consecutive byte vectors -> Linear(k d_b -> d)
         -> global Blocks (RoPE, QK-norm, q/k/v conv; mid-stack latent here)
         -> Linear(d -> k d_b), shifted right by k-1 bytes
         -> + encoder output (byte skip) -> dec ByteBlocks -> norm -> tied head

Causality: patch j covers bytes jk .. jk+k-1, so its global vector is known
once byte jk+k-1 is seen. The k upsampled vectors of patch j go to byte
positions jk+k-1 .. jk+2k-2: the position that completes patch j and the
k-1 positions inside patch j+1 (each predicts the next byte). Positions
0 .. k-2 get no global vector, only the byte path. Inside a patch, the
bytes seen so far reach the decoder through the skip and the causal convs
(kernel >= k).

Related work: Hourglass (Nawrot et al. 2021), MEGABYTE (Yu et al. 2023),
SpaceByte (Slagle 2024), BLT (Pagnoni et al. 2024), H-Net (Hwang et al. 2025).
Fixed stride first; word-boundary patches next if this pays off.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from intgrad import quant as Q
from intgrad.models.transformer import (QAT, Block, ByteLM, CausalConv, Tap, _q, _sites,
                                        rope_cache)

PRESETS = {
    # global part = the flat preset of the same name; byte width d/2
    "XS": dict(d=128, L=2, H=4, d_b=64),
    "S": dict(d=256, L=4, H=4, d_b=128),
    "M": dict(d=384, L=6, H=6, d_b=192),
    "L": dict(d=512, L=8, H=8, d_b=256),
}


class ByteBlock(nn.Module):
    """Byte-level residual block: causal depthwise conv (time mixing) and a
    ReLU² MLP (channel mixing), both pre-RMSNorm."""

    def __init__(self, d: int, kernel: int, n_res: int, qat: QAT = QAT(), mlp: int = 4):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.conv = CausalConv(d, d, kernel, depthwise=True)
        nn.init.normal_(self.conv.conv.weight, std=1.0 / kernel)
        self.up = nn.Linear(d, mlp * d, bias=False)
        self.down = nn.Linear(mlp * d, d, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02 / math.sqrt(2 * n_res))
        self.qs = _sites("n1 conv n2 up mlp", qat.acts)
        self.rs = _sites("r1 r2", qat.resid == "q8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qs, rs = self.qs, self.rs
        x = _q(rs, "r1", x + _q(qs, "conv", self.conv(_q(qs, "n1", self.n1(x)))))
        u = _q(qs, "up", self.up(_q(qs, "n2", self.n2(x))))
        return _q(rs, "r2", x + self.down(_q(qs, "mlp", torch.relu(u).square())))


class Hourglass(ByteLM):
    def __init__(self, d: int, L: int, H: int, T: int, k: int = 4, d_b: int = 128,
                 enc: int = 2, dec: int = 2, kernel: int = 0, zloss: float = 1e-4,
                 qk_norm: bool = True, qkv_conv: int = 0, qat: QAT | dict | None = None,
                 byte_mix: str = "conv", byte_window: int = 64, byte_dh: int = 64):
        """T: byte context (a multiple of k); the global stack sees T/k patches.
        byte_mix "conv": ByteBlocks (causal depthwise conv + MLP);
        "attn": transformer Blocks with sliding-window attention over the
        last byte_window bytes (heads of width byte_dh), as in SpaceByte."""
        super().__init__()
        if T % k:
            raise ValueError(f"T={T} must be a multiple of k={k}")
        self.d, self.L, self.H, self.T, self.k, self.d_b = d, L, H, T, k, d_b
        self.zloss = zloss
        self.qat = qat = QAT(**qat) if isinstance(qat, dict) else (qat or QAT())
        kernel = kernel or max(4, k)
        n_res = enc + dec
        if byte_mix not in ("conv", "attn"):
            raise ValueError(f"byte_mix {byte_mix!r}")
        self.byte_mix, self.byte_H = byte_mix, max(1, d_b // byte_dh)

        def byte_block():
            if byte_mix == "conv":
                return ByteBlock(d_b, kernel, n_res, qat)
            return Block(d_b, self.byte_H, n_res, qk_norm, qkv_conv, qat, window=byte_window)

        self.emb = nn.Embedding(256, d_b)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.enc = nn.ModuleList(byte_block() for _ in range(enc))
        self.n_down = nn.RMSNorm(d_b)
        self.down = nn.Linear(k * d_b, d, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(k * d_b))
        self.blocks = nn.ModuleList(Block(d, H, L, qk_norm, qkv_conv, qat) for _ in range(L))
        self.n_up = nn.RMSNorm(d)
        self.up = nn.Linear(d, k * d_b, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        self.dec = nn.ModuleList(byte_block() for _ in range(dec))
        self.nf = nn.RMSNorm(d_b)
        self.qs = _sites("emb down up nf", qat.acts)
        self._rope = None
        self._brope = None

    def _bblk(self, blk, h):
        if self.byte_mix == "conv":
            return blk(h)
        if self._brope is None or self._brope[0].device != h.device:
            self._brope = rope_cache(self.T, self.d_b // self.byte_H, h.device)
        return blk(h, self._brope)

    @property
    def mid(self) -> int:
        """First upper global block: the mid-stack latent (patch level)."""
        return self.L // 2

    def attn_elems(self) -> int:
        Tg = self.T // self.k
        return self.H * Tg * Tg

    def rope(self, device):
        if self._rope is None or self._rope[0].device != torch.device(device):
            self._rope = rope_cache(self.T // self.k, self.d // self.H, device)
        return self._rope

    def forward(self, x: torch.Tensor, tap: Tap | None = None,
                mid_hook=None) -> torch.Tensor:
        """x: [B, T'] bytes, any T' <= T (right-padded to a multiple of k;
        padding only follows real positions, so it cannot leak back)."""
        B, T0 = x.shape
        k, qs = self.k, self.qs
        pad = -T0 % k
        if pad:
            x = F.pad(x, (0, pad))
        T = T0 + pad
        h = _q(qs, "emb", self.emb(x.long()))
        for blk in self.enc:
            h = self._bblk(blk, h)
        if tap is not None:
            tap("emb", h)
        g = self.down(_q(qs, "down", self.n_down(h)).reshape(B, T // k, k * self.d_b))
        rope = self.rope(x.device)
        for i, blk in enumerate(self.blocks):
            if i == self.mid:
                if tap is not None:
                    tap("h_mid", g)
                if mid_hook is not None:
                    g = mid_hook(g)
            g = blk(g, rope, tap, i)
        u = self.up(_q(qs, "up", self.n_up(g))).reshape(B, T, self.d_b)
        u = F.pad(u, (0, 0, k - 1, 0))[:, :T]          # patch j -> bytes jk+k-1 ..
        h = h + u
        for blk in self.dec:
            h = self._bblk(blk, h)
        logits = F.linear(_q(qs, "nf", self.nf(h)), self.emb.weight).float()[:, :T0]
        if self.qat.softmax == "maxrel":
            logits = Q.logit_maxrel(logits)
        if tap is not None:
            tap("logits", logits)
            tap.rel("logits", logits)
            for name, p in self.named_parameters():
                tap(f"w.{name}", p)
        return logits


def build(preset: str | None = None, **kw) -> Hourglass:
    cfg = dict(PRESETS[preset]) if preset else {}
    cfg.update(kw)
    return Hourglass(**cfg)
