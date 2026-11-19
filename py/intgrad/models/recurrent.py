"""Plan 002: gated diagonal linear recurrence (RG-LRU / Mamba style), a
temporal-mixing branch added to every transformer block.

Per channel c and byte t (x_t: the normed block input):

    u_t = conv4(W_u x_t)                          short causal conv (as in Mamba/Griffin)
    r_t = σ(W_r x_t),  i_t = σ(W_i x_t)           recurrence and input gates
    log a_t = c · r_t · log σ(Λ)                  input-dependent decay in (0, 1)
    h_t = a_t ⊙ h_{t−1} + sqrt(1 − a_t²) ⊙ (i_t ⊙ u_t)
    y_t = W_o (h_t ⊙ gelu(W_g x_t))

The decay lives in log space, so values near 1 stay trainable (Λ is
initialised so that σ(Λ)^c is uniform in [0.9, 0.999]). At a document
start the carried state is dropped (a_t = 0), anywhere in the window. The
state h is handed from one window to the next of the same stream.

The scan runs in fp32 as two levels of short loops: inside chunks of C
bytes (all chunks at once), then across chunk ends; C + T/C sequential
steps instead of T.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from intgrad.models.transformer import CausalConv

C_DECAY = 8.0


def linear_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None,
                chunk: int = 32) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t over dim 1. a, b: [B, T, D]; h0: [B, D] or
    None (zeros). Returns h [B, T, D]."""
    B, T, D = b.shape
    C = min(chunk, T)
    pad = -T % C
    if pad:                                         # a = 1, b = 0: carries h, adds nothing
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, pad))
    n = (T + pad) // C
    a = a.reshape(B, n, C, D)
    b = b.reshape(B, n, C, D)
    hs, ps = [], []                                 # local state (zero init) and decay product
    h = torch.zeros(B, n, D, device=b.device, dtype=b.dtype)
    p = torch.ones(B, n, D, device=b.device, dtype=b.dtype)
    for t in range(C):
        h = a[:, :, t] * h + b[:, :, t]
        p = p * a[:, :, t]
        hs.append(h)
        ps.append(p)
    hl = torch.stack(hs, 2)                         # [B, n, C, D]
    pl = torch.stack(ps, 2)
    carry = h0 if h0 is not None else torch.zeros(B, D, device=b.device, dtype=b.dtype)
    ins = []                                        # state entering each chunk
    for j in range(n):
        ins.append(carry)
        carry = pl[:, j, -1] * carry + hl[:, j, -1]
    hin = torch.stack(ins, 1)                       # [B, n, D]
    out = hl + pl * hin[:, :, None]
    return out.reshape(B, n * C, D)[:, :T]


class RecMix(nn.Module):
    def __init__(self, d: int, L: int, d_rec: int = 0, conv: int = 4, chunk: int = 32):
        super().__init__()
        self.d_rec = dr = d_rec or d
        self.chunk = chunk
        self.norm = nn.RMSNorm(d)
        self.inp = nn.Linear(d, 4 * dr, bias=False)    # u, r, i, g
        nn.init.normal_(self.inp.weight, std=0.02)
        self.conv = CausalConv(dr, dr, conv, depthwise=True, identity_init=True) if conv else None
        a0 = torch.empty(dr).uniform_(0.9, 0.999)       # σ(Λ)^c at r = 1
        s = torch.exp(torch.log(a0) / C_DECAY)          # σ(Λ)
        self.lam = nn.Parameter(torch.log(s) - torch.log1p(-s))
        self.out = nn.Linear(dr, d, bias=False)
        nn.init.normal_(self.out.weight, std=0.02 / math.sqrt(2 * L))

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None,
                reset: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, T, d] -> (x + mix, final state [B, d_rec]). reset [B, T]
        bool: the carried state is dropped before position t."""
        u, r, i, g = self.inp(self.norm(x)).chunk(4, -1)
        if self.conv is not None:
            u = self.conv(u)
        f32 = torch.float32
        log_a = C_DECAY * torch.sigmoid(r.to(f32)) * F.logsigmoid(self.lam.to(f32))
        a = torch.exp(log_a)
        bx = torch.sqrt(torch.clamp(1 - a * a, min=0.0)) * (torch.sigmoid(i.to(f32)) * u.to(f32))
        if reset is not None:
            a = a.masked_fill(reset[..., None], 0.0)
        h = linear_scan(a, bx, None if h0 is None else h0.to(f32), self.chunk)
        y = h.to(x.dtype) * F.gelu(g)
        return x + self.out(y), h[:, -1]
