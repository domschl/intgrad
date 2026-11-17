"""Lion with byte-numerics parameter storage, bit-identical to the C oracle
(ig_mlp_lion_update in src/net.c) given identical gradients.

Per parameter tensor, in flat (row-major) order:
    g  = grad                                  (pure_int: byte grid at pow2 absmax scale)
    c  = b1*m + (1-b1)*g ;  u = sign(c)
    step = lr * (u + wd * w)                   (w = forward value)
    storage:  float:  w -= step
              w2:     wq = enc16(dec16(wq) - step [, SR]) ; w = high byte of wq
              w1:     w = rt8_sr(w - step) (or rt8 without SR)
    m  = b2*m + (1-b2)*g ;  quant: m = dec16(enc16(m, s), s), s = pow2_scale(max|m|)

All constants are float32 values computed the way C computes them
((1.0f - 0.9f) is not 0.1f). SR uniforms come from `sr_uniform(n, device)`
(default: torch.rand); parity tests inject the C xorshift stream.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from intgrad import quant as Q

B1, B2 = np.float32(0.9), np.float32(0.99)
OMB1, OMB2 = np.float32(1.0) - B1, np.float32(1.0) - B2


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 2e-3, wd: float = 0.1,
                 qc: Q.QuantConfig | None = None,
                 sr_uniform: Callable[[int, torch.device], torch.Tensor] | None = None,
                 generator: torch.Generator | None = None):
        super().__init__(params, dict(lr=lr, wd=wd))
        self.qc = qc or Q.QuantConfig()
        self.gen = generator
        self.sr_uniform = sr_uniform or self._rand
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["m"] = torch.zeros_like(p, dtype=torch.float32)
                if self.qc.enabled and self.qc.w2:
                    st["wq"] = Q.enc16(p.detach(), self.qc.scale)

    def _rand(self, n: int, device) -> torch.Tensor:
        return torch.rand(n, generator=self.gen, device=device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        qc = self.qc
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dev = p.device
                lr = Q._f32(group["lr"], dev)
                wd = Q._f32(group["wd"], dev)
                st = self.state[p]
                g = p.grad.to(torch.float32)
                if qc.enabled and qc.pure_int:
                    g = Q.rt8(g, Q.pow2_scale(g.abs().max()))
                mo = st["m"]
                c = Q._f32(B1, dev) * mo + Q._f32(OMB1, dev) * g
                u = torch.sign(c)
                step = lr * (u + wd * p)
                if not qc.enabled:
                    p.sub_(step)
                elif qc.w2:
                    sr = self.sr_uniform(p.numel(), dev).view_as(p) if qc.w2_sr else None
                    st["wq"] = Q.enc16(Q.dec16(st["wq"], qc.scale) - step, qc.scale, sr)
                    p.copy_(Q.w16_fwd(st["wq"], qc.scale))
                elif qc.sr:
                    sr = self.sr_uniform(p.numel(), dev).view_as(p)
                    p.copy_(Q.rt8_sr(p - step, sr, qc.scale))
                else:
                    p.copy_(Q.rt8(p - step, qc.scale))
                mn = Q._f32(B2, dev) * mo + Q._f32(OMB2, dev) * g
                if qc.enabled:
                    s = Q.pow2_scale(mn.abs().max())
                    mn = Q.dec16(Q.enc16(mn, s), s)
                st["m"] = mn
        return None
