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


# ---------------------------------------------------------------- transformer probes

SEP = 0x02


def _copy_batch(B: int, n: int, T: int, g: torch.Generator):
    """payload(n) SEP payload(n) EOD, padded; loss on the copy + EOD."""
    pay = torch.randint(97, 123, (B, n), generator=g, dtype=torch.uint8)
    seq = torch.zeros(B, T + 1, dtype=torch.uint8)
    seq[:, :n] = pay
    seq[:, n] = SEP
    seq[:, n + 1:2 * n + 1] = pay
    seq[:, 2 * n + 1] = data.EOD
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, n:2 * n + 1] = True                      # targets after SEP
    return seq[:, :-1], seq[:, 1:], mask


def _recall_batch(B: int, pairs: int, T: int, g: torch.Generator):
    """k1 v1 k2 v2 ... '?' kq vq: distinct letter keys, digit values;
    loss on the answer only."""
    keys = torch.stack([torch.randperm(26, generator=g)[:pairs] for _ in range(B)]) + 97
    vals = torch.randint(48, 58, (B, pairs), generator=g)
    q = torch.randint(0, pairs, (B,), generator=g)
    seq = torch.zeros(B, T + 1, dtype=torch.uint8)
    kv = torch.stack([keys, vals], -1).reshape(B, -1)
    seq[:, :2 * pairs] = kv.to(torch.uint8)
    seq[:, 2 * pairs] = ord("?")
    seq[:, 2 * pairs + 1] = keys[torch.arange(B), q].to(torch.uint8)
    seq[:, 2 * pairs + 2] = vals[torch.arange(B), q].to(torch.uint8)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 2 * pairs + 1] = True
    return seq[:, :-1], seq[:, 1:], mask


def _tf(preset, T, qc: QuantConfig | None, device, arch: str = "flat", **model_kw):
    """Probe model; qc (optional) quantizes it like ig-train does: weights
    via Lion master codes (qc.enabled), forward sites via qc.acts…
    arch "hourglass" (M6): model_kw may set k, d_b, enc, dec; T is rounded
    up to a multiple of k (shorter inputs are padded by the model)."""
    from intgrad.models import hourglass, transformer
    qc = qc or QuantConfig(enabled=False)
    qat = transformer.QAT(acts=qc.acts, resid=qc.resid, softmax=qc.softmax, probs=qc.probs)
    if arch == "hourglass":
        k = model_kw.get("k", 4)
        m = hourglass.build(preset, T=-(-T // k) * k, qat=qat, **model_kw)
    else:
        m = transformer.build(preset, T=T, qat=qat, **model_kw)
    for p in m.parameters():
        qc.project_(p)
    return m.to(device)


def _train_masked(model, make, steps: int, lr: float, wd: float, B: int, device, seed: int,
                  betas=(0.9, 0.99), qc: QuantConfig | None = None):
    from intgrad.train import param_groups
    g = torch.Generator().manual_seed(seed)
    qc = qc or QuantConfig(enabled=False)
    opt = Lion(param_groups(model, wd), lr=lr, wd=wd, qc=qc, betas=betas)
    warm = max(1, steps // 10)
    for s in range(steps):
        lr_s = lr * (s + 1) / warm if s < warm else \
            lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm)))
        for grp in opt.param_groups:
            grp["lr"] = lr_s
        x, y, mask = (t.to(device) for t in make(B, g))
        lg = model(x)
        loss = F.cross_entropy(lg[mask], y[mask].long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    g_test = torch.Generator().manual_seed(seed + 999)
    model.eval()                                     # frozen quantization scales
    with torch.no_grad():
        x, y, mask = (t.to(device) for t in make(1024, g_test))
        acc = (model(x).argmax(-1)[mask] == y[mask].long()).float().mean().item()
    return acc


def p2_copy(device="cpu", steps=1500, lr=1e-3, wd=0.1, n=24, seed=0, preset="XS",
            qc: QuantConfig | None = None, **model_kw) -> dict:
    torch.manual_seed(seed)
    T = 2 * n + 4
    m = _tf(preset, T, qc, device, **model_kw)
    acc = _train_masked(m, lambda B, g: _copy_batch(B, n, T, g), steps, lr, wd, 128, device, seed,
                        qc=qc)
    return {"acc": acc, "pass": acc >= 0.99}


def _mqar_batch(B: int, pairs: int, T: int, g: torch.Generator):
    """Multi-query associative recall (Zoology, Arora et al. 2023):
    k1 v1 … kp vp, then every key again in shuffled order, each followed by
    its value. Loss on every answer (p supervised targets per sequence)."""
    keys = torch.stack([torch.randperm(26, generator=g)[:pairs] for _ in range(B)]) + 97
    vals = torch.randint(48, 58, (B, pairs), generator=g)
    order = torch.stack([torch.randperm(pairs, generator=g) for _ in range(B)])
    qk = torch.gather(keys, 1, order)
    qv = torch.gather(vals, 1, order)
    seq = torch.zeros(B, T + 1, dtype=torch.uint8)
    seq[:, :2 * pairs] = torch.stack([keys, vals], -1).reshape(B, -1).to(torch.uint8)
    seq[:, 2 * pairs:4 * pairs] = torch.stack([qk, qv], -1).reshape(B, -1).to(torch.uint8)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 2 * pairs:4 * pairs:2] = True            # x = query key, y = its value
    return seq[:, :-1], seq[:, 1:], mask


def p3_recall(device="cpu", steps=5000, lr=1e-4, wd=0.1, pairs=8, seed=0, preset="XS",
              batch=512, betas=(0.95, 0.98), qc: QuantConfig | None = None, **model_kw) -> dict:
    """P3 = MQAR. The single-query variant (_recall_batch) is kept for the
    X7 record: it was not learned in 10k steps by Lion or AdamW."""
    torch.manual_seed(seed)
    T = 4 * pairs + 2
    m = _tf(preset, T, qc, device, **model_kw)
    acc = _train_masked(m, lambda B, g: _mqar_batch(B, pairs, T, g), steps, lr, wd, batch,
                        device, seed, betas, qc=qc)
    return {"acc": acc, "pass": acc >= 0.95}


def p1_memorize_tf(device="cpu", steps=600, lr=1e-3, wd=0.1, seed=0, preset="XS",
                   qc: QuantConfig | None = None) -> dict:
    """Transformer P1: overfit.txt as one sequence (EOD first); with the
    full prefix as context every position is predictable -> floor ~0."""
    raw = torch.tensor(list((FIXTURES / "overfit.txt").read_bytes()), dtype=torch.uint8)
    seq = torch.cat([torch.tensor([data.EOD], dtype=torch.uint8), raw])
    T = len(seq) - 1
    torch.manual_seed(seed)
    m = _tf(preset, T, qc, device)
    x, y = seq[None, :-1], seq[None, 1:]
    mask = torch.ones(1, T, dtype=torch.bool)
    _train_masked(m, lambda B, g: (x, y, mask), steps, lr, wd, 1, device, seed, qc=qc)
    m.eval()
    with torch.no_grad():
        ce = F.cross_entropy(m(x.to(device))[0], y[0].long().to(device)).item()
    return {"ce": ce, "pass": ce < 0.01}
