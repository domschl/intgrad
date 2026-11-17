"""Capability probes (plan/001 §4.3). Each returns a dict with `pass`.

P1 memorize: train on tests/fixtures/overfit.txt until CE reaches the
   achievable floor H(next byte | ctx bytes) of the file (+ 0.01 nats).
   lr 3e-3: at 1e-2 quantized runs saturate (clamped logits/activations
   get no STE gradient and never recover) while fp still converges (X5).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from intgrad import data
from intgrad.lion import Lion
from intgrad.models.mlp import ByteMLP
from intgrad.quant import QuantConfig

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def cond_entropy_nats(ctxs: torch.Tensor, tgt: torch.Tensor) -> float:
    """H(target | context) of the empirical windows, in nats per window."""
    by = defaultdict(Counter)
    for c, t in zip(map(bytes, ctxs.tolist()), tgt.tolist()):
        by[c][t] += 1
    n, h = len(tgt), 0.0
    for cnt in by.values():
        tot = sum(cnt.values())
        h -= sum(k * math.log(k / tot) for k in cnt.values())
    return h / n


def p1_memorize(qc: QuantConfig, device="cpu", ctx: int = 8, nh: int = 128,
                steps: int = 1500, lr: float = 3e-3, seed: int = 0) -> dict:
    raw = torch.tensor(list((FIXTURES / "overfit.txt").read_bytes()), dtype=torch.uint8)
    seq = torch.cat([torch.full((ctx,), data.EOD, dtype=torch.uint8), raw])
    idx = torch.arange(len(raw))[:, None] + torch.arange(ctx)[None, :]
    ctxs, tgt = seq[idx], raw.long()
    floor = cond_entropy_nats(ctxs, tgt)
    torch.manual_seed(seed)
    m = ByteMLP(ctx, nh, qc, generator=torch.Generator().manual_seed(seed)).to(device)
    opt = Lion(m.parameters(), lr=lr, wd=0.0, qc=qc,
               generator=torch.Generator(device=device).manual_seed(seed))
    x, y = ctxs.to(device), tgt.to(device)
    warm = steps // 10
    for s in range(steps):
        g = lr * (s + 1) / warm if s < warm else \
            lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm)))
        for grp in opt.param_groups:
            grp["lr"] = g
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = F.cross_entropy(m(x), y).item()
    return {"ce": final, "floor": floor, "pass": final <= floor + 0.01}
