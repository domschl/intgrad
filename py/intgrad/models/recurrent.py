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

state N > 1 (plan 002 N3a, Mamba-2 / SSD style): channels are grouped in
heads of P; each head carries a P x N state with one decay per head:

    S_t = a_t S_{t-1} + (sqrt(1 - a_t²) i_t u_t) ⊗ k_t      k, q ∈ R^N, L2-normed
    o_t = S_t q_t                                          (y as above with o for h)

computed chunk-wise: inside a chunk as a decay-masked (q·k) matmul, across
chunks by handing the P x N chunk states on (ssd_scan).
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


def ssd_scan(log_a: torch.Tensor, b: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
             S0: torch.Tensor | None = None, reset: torch.Tensor | None = None,
             chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """S_t = a_t S_{t-1} + b_t ⊗ k_t,  o_t = S_t q_t  per head, a_t = 0 where
    reset. log_a [B, T, H] (finite, <= 0); b [B, T, H, P]; k, q [B, T, H, N];
    S0 [B, H, P, N] or None; reset [B, T] bool or None.
    Returns (o [B, T, H, P], S_T [B, H, P, N])."""
    Bt, T, H, P = b.shape
    N = k.shape[-1]
    C = min(chunk, T)
    pad = -T % C
    if reset is None:
        reset = torch.zeros(Bt, T, dtype=torch.bool, device=b.device)
    if pad:                                         # a = 1, b = k = q = 0: carries S, adds nothing
        log_a = F.pad(log_a, (0, 0, 0, pad))
        b, k, q = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (b, k, q))
        reset = F.pad(reset, (0, pad))
    n = (T + pad) // C
    # [B, H, n, C, ...]
    la = log_a.masked_fill(reset[..., None], 0.0).reshape(Bt, n, C, H).permute(0, 3, 1, 2)
    b, k, q = (z.reshape(Bt, n, C, H, -1).permute(0, 3, 1, 2, 4) for z in (b, k, q))
    seg = torch.cumsum(reset.reshape(Bt, n, C).long(), -1)[:, None]      # [B, 1, n, C]
    cs = torch.cumsum(la, -1)                                            # [B, H, n, C]
    tri = torch.ones(C, C, dtype=torch.bool, device=b.device).tril()
    same = (seg[..., :, None] == seg[..., None, :]) & tri                # [B, 1, n, C(t), C(s)]
    dec = torch.exp((cs[..., :, None] - cs[..., None, :]).masked_fill(~same, -math.inf))
    o = (dec * (q @ k.transpose(-1, -2))) @ b                            # inside the chunk
    w_end = torch.exp(cs[..., -1:] - cs) * (seg == seg[..., -1:])        # s -> chunk end
    Sc = (b * w_end[..., None]).transpose(-1, -2) @ k                    # [B, H, n, P, N]
    surv = torch.exp(cs) * (seg == 0)                                    # S_in -> t
    S = S0 if S0 is not None else b.new_zeros(Bt, H, P, N)
    ins = []
    for j in range(n):
        ins.append(S)
        S = surv[:, :, j, -1, None, None] * S + Sc[:, :, j]
    Sin = torch.stack(ins, 2)                                            # [B, H, n, P, N]
    o = o + surv[..., None] * (q @ Sin.transpose(-1, -2))
    o = o.permute(0, 2, 3, 1, 4).reshape(Bt, n * C, H, P)[:, :T]
    return o, S


class RecMix(nn.Module):
    def __init__(self, d: int, L: int, d_rec: int = 0, conv: int = 4, chunk: int = 32,
                 state: int = 1, head: int = 16):
        super().__init__()
        self.d_rec = dr = d_rec or d
        self.chunk = chunk
        self.N = state
        nd = dr                                         # decays: one per channel (N = 1) or head
        if state > 1:
            if dr % head:
                raise ValueError("d_rec must be a multiple of rec_head")
            self.P, self.H = head, dr // head
            nd = self.H
        self.norm = nn.RMSNorm(d)
        # u, r, i, g (+ k, q per head)
        self.inp = nn.Linear(d, 3 * dr + nd + (2 * nd * state if state > 1 else 0), bias=False)
        nn.init.normal_(self.inp.weight, std=0.02)
        self.conv = CausalConv(dr, dr, conv, depthwise=True, identity_init=True) if conv else None
        a0 = torch.empty(nd).uniform_(0.9, 0.999)       # σ(Λ)^c at r = 1
        s = torch.exp(torch.log(a0) / C_DECAY)          # σ(Λ)
        self.lam = nn.Parameter(torch.log(s) - torch.log1p(-s))
        self.out = nn.Linear(dr, d, bias=False)
        nn.init.normal_(self.out.weight, std=0.02 / math.sqrt(2 * L))

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None,
                reset: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, T, d] -> (x + mix, final state [B, d_rec]). reset [B, T]
        bool: the carried state is dropped before position t."""
        dr = self.d_rec
        z = self.inp(self.norm(x))
        nd = self.lam.numel()
        u, r = z[..., :dr], z[..., dr:dr + nd]          # order u, r, i, g as with N = 1
        i, g = z[..., dr + nd:2 * dr + nd], z[..., 2 * dr + nd:3 * dr + nd]
        if self.conv is not None:
            u = self.conv(u)
        f32 = torch.float32
        log_a = C_DECAY * torch.sigmoid(r.to(f32)) * F.logsigmoid(self.lam.to(f32))
        a = torch.exp(log_a)
        if self.N > 1:
            Bt, T, _ = x.shape
            H, P, N = self.H, self.P, self.N
            kq = z[..., 3 * dr + nd:].to(f32).view(Bt, T, H, 2 * N)
            k, q = F.normalize(kq[..., :N], dim=-1), F.normalize(kq[..., N:], dim=-1)
            bx = (torch.sqrt(torch.clamp(1 - a * a, min=0.0))[..., None]
                  * (torch.sigmoid(i.to(f32)) * u.to(f32)).view(Bt, T, H, P))
            o, S = ssd_scan(log_a, bx, k, q, None if h0 is None else h0.to(f32), reset, self.chunk)
            y = o.reshape(Bt, T, dr).to(x.dtype) * F.gelu(g)
            return x + self.out(y), S
        bx = torch.sqrt(torch.clamp(1 - a * a, min=0.0)) * (torch.sigmoid(i.to(f32)) * u.to(f32))
        if reset is not None:
            a = a.masked_fill(reset[..., None], 0.0)
        h = linear_scan(a, bx, None if h0 is None else h0.to(f32), self.chunk)
        y = h.to(x.dtype) * F.gelu(g)
        return x + self.out(y), h[:, -1]
