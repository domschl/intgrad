"""Byte MLP — the parity vehicle (port of src/net.c ig_mlp_forward).

Predicts the next byte from the previous `ctx` bytes (one-hot, so layer 1
is a row lookup). The accumulation order is the C oracle's (bias first,
then terms in index order, separate multiply and add), so quantized
logits are bit-identical to C for identical parameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from intgrad.quant import QuantConfig


class ByteMLP(nn.Module):
    def __init__(self, ctx: int, nh: int, qc: QuantConfig | None = None,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.ctx, self.nh = ctx, nh
        self.qc = qc or QuantConfig()
        g = generator
        self.W1 = nn.Parameter((torch.rand(ctx * 256, nh, generator=g) * 2 - 1) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(nh))
        self.W2 = nn.Parameter((torch.rand(nh, 256, generator=g) * 2 - 1) * (0.3 / math.sqrt(nh)))
        self.b2 = nn.Parameter(torch.zeros(256))
        for p in self.parameters():
            self.qc.project_(p)

    def flat_params(self) -> list[nn.Parameter]:
        """Canonical (C) order: W1, b1, W2, b2."""
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, ctx] bytes -> quantized logits [B, 256]."""
        x = x.long()
        acc = self.b1.expand(x.shape[0], -1)
        for p in range(self.ctx):
            acc = acc + self.W1[p * 256 + x[:, p]]
        h = torch.relu(self.qc.act(acc))
        out = self.b2.expand(x.shape[0], -1)
        for j in range(self.nh):
            out = out + h[:, j:j + 1] * self.W2[j]
        return self.qc.logit(out)


def windows(chunks: torch.Tensor, ctx: int, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All (context, target) pairs for positions 1..L-1 of each chunk;
    context bytes before the chunk start are `pad`."""
    n, L = chunks.shape
    padded = torch.cat([torch.full((n, ctx), pad, dtype=chunks.dtype, device=chunks.device),
                        chunks], dim=1)
    idx = torch.arange(1, L, device=chunks.device)[:, None] + torch.arange(ctx, device=chunks.device)[None, :]
    ctxs = padded[:, idx]                     # [n, L-1, ctx]
    return ctxs.reshape(-1, ctx), chunks[:, 1:].reshape(-1)


def load_c_checkpoint(path: str, qc: QuantConfig | None = None) -> ByteMLP:
    """Parameters (forward values) of a C oracle checkpoint (format v8)."""
    import struct

    import numpy as np
    buf = open(path, "rb").read()
    if buf[:4] != b"IGCK":
        raise ValueError(f"{path}: not an intgrad checkpoint")
    ver, ctx, nh = struct.unpack_from("<iii", buf, 4)
    if ver != 8:
        raise ValueError(f"{path}: checkpoint v{ver}, need v8")
    m = ByteMLP(ctx, nh, qc)
    rec = np.frombuffer(buf, dtype=np.dtype([("d", "<f4"), ("wq", "<i2")]),
                        count=sum(p.numel() for p in m.flat_params()), offset=20)
    off = 0
    with torch.no_grad():
        for p in m.flat_params():
            p.copy_(torch.from_numpy(rec["d"][off:off + p.numel()].copy()).view_as(p))
            off += p.numel()
    return m
