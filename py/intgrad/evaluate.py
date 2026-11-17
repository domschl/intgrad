"""Score models on the frozen eval set (plan/001 §4.4): bits/byte over
positions 1..L-1 of every chunk, from in-chunk context only."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from intgrad import data
from intgrad.eval_set import EvalSet, report
from intgrad.models.mlp import ByteMLP, windows


@torch.no_grad()
def chunk_bits(model, chunks: torch.Tensor, batch: int = 1 << 16) -> torch.Tensor:
    """[N, L-1] bits for predicting chunks[:, 1:]."""
    model.eval()
    if isinstance(model, ByteMLP):
        ctxs, tgt = windows(chunks, model.ctx, data.EOD)
        out = torch.empty(len(tgt), device=chunks.device)
        for i in range(0, len(tgt), batch):
            lg = model(ctxs[i:i + batch])
            out[i:i + batch] = F.cross_entropy(lg, tgt[i:i + batch].long(), reduction="none")
        bits = out.view(chunks.shape[0], -1) / math.log(2)
    else:
        bits = model.chunk_bits(chunks)
    model.train()
    return bits


def evaluate(model, ev: EvalSet, device) -> dict:
    chunks = torch.from_numpy(ev.chunks.copy()).to(device)
    bits = chunk_bits(model, chunks).double().cpu().numpy()
    return report(bits, ev.src, ev.sources)
