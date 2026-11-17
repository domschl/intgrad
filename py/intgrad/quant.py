"""Byte numerics for PyTorch, bit-identical to the C oracle (src/logquant.c).

Canonical grids (see include/ig/logquant.h): float32 tables computed in
double,  V(t) = float32(exp2(t * 0.1 / 256) - 1)  with t in 1/256 steps:
    DEC8[c]  = V(256 c)        byte grid values, c = 0..127
    THR8[k]  = V(256 k - 128)  byte rounding thresholds, k = 1..127
    DEC16[M] = V(M)            16-bit grid values, M = 0..32512
    THR16[k] = V(k - 0.5)      16-bit thresholds
A value x at power-of-two scale s has a = |x| * s (exact); its code is the
number of thresholds <= a. Codes here are SIGNED integers (sign * magnitude);
the C byte layout (128 + q for negatives) is a storage detail.

Grids in use (plan/001 §2):
    act / weights   byte symlog grid at scale 256        rt8 / QAct
    logits          fixed-point int8 * 0.1, half away    fx  / QFx
    master weights  16-bit symlog grid (forward = high byte)
    Lion momentum, pure-int gradients: byte / 16-bit grid at a per-tensor
                    power-of-two absmax scale (pow2_scale)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MAX_CODE = 127
SUB = 256
MAX_M = MAX_CODE * SUB
SCALE = 256.0          # global value scale (activations, weights)
FXSTEP = np.float32(0.1)


def _V(t: np.ndarray) -> np.ndarray:
    return (np.exp2(t * 0.1 / 256.0) - 1.0).astype(np.float32)


DEC8 = _V(256.0 * np.arange(MAX_CODE + 1, dtype=np.float64))
THR8 = np.concatenate([[0.0], _V(256.0 * np.arange(1, MAX_CODE + 1) - 128.0)]).astype(np.float32)
DEC16 = _V(np.arange(MAX_M + 1, dtype=np.float64))
THR16 = np.concatenate([[0.0], _V(np.arange(1, MAX_M + 1) - 0.5)]).astype(np.float32)

_cache: dict[torch.device, dict[str, torch.Tensor]] = {}


def tables(device) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    t = _cache.get(device)
    if t is None:
        t = _cache[device] = {
            "dec8": torch.from_numpy(DEC8).to(device),
            "thr8": torch.from_numpy(THR8[1:].copy()).to(device),
            "dec8_1": torch.from_numpy(DEC8[1:].copy()).to(device),
            "dec16": torch.from_numpy(DEC16).to(device),
            "thr16": torch.from_numpy(THR16[1:].copy()).to(device),
            "dec16_1": torch.from_numpy(DEC16[1:].copy()).to(device),
            # thr8x[k] = THR8[k] for k = 1..127, thr8x[0] = -inf, thr8x[128] = +inf
            "thr8x": torch.from_numpy(np.concatenate(
                [[-np.inf], THR8[1:], [np.inf]]).astype(np.float32)).to(device),
        }
    return t


def _f32(v, device) -> torch.Tensor:
    return torch.as_tensor(v, dtype=torch.float32, device=device)


def _count_le(sorted_t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(sorted_t, a.contiguous(), right=True)


def max_abs(scale) -> float:
    return float(DEC8[MAX_CODE] / np.float32(scale))


# ---------------------------------------------------------------- byte grid

def encode8(x: torch.Tensor, scale) -> torch.Tensor:
    """Signed byte codes (int16, |code| <= 127); 0 and NaN -> 0."""
    T = tables(x.device)
    a = x.abs() * _f32(scale, x.device)
    q = _count_le(T["thr8"], torch.nan_to_num(a, nan=0.0)).to(torch.int16)
    q = torch.where(torch.isnan(x), torch.zeros_like(q), q)
    return torch.where(x < 0, -q, q)


def decode8(q: torch.Tensor, scale) -> torch.Tensor:
    T = tables(q.device)
    m = T["dec8"][q.abs().long()] / _f32(scale, q.device)
    return torch.where(q < 0, -m, m)


def rt8(x: torch.Tensor, scale=SCALE) -> torch.Tensor:
    return decode8(encode8(x, scale), scale)


def rt8_fast(x: torch.Tensor, scale) -> torch.Tensor:
    """rt8 without searchsorted (elementwise + two gathers, compiles well),
    bit-identical to rt8: estimate the code from log2, then correct it by
    one against the canonical thresholds (code k <=> THR8[k] <= a < THR8[k+1])."""
    T = tables(x.device)
    s = _f32(scale, x.device)
    a = torch.nan_to_num(x.abs() * s, nan=0.0)
    c = torch.floor(torch.log2(a + 1.0) * 10.0 + 0.5).clamp(0, MAX_CODE).long()
    thr = T["thr8x"]
    c = c + (a >= thr[c + 1]).long() - (a < thr[c]).long()
    m = T["dec8"][c] / s
    return torch.where(x < 0, -m, m)


def _sr_pick(dec_1: torch.Tensor, dec: torch.Tensor, maxc: int,
             a: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """SR between neighbouring grid points; returns (index, value)."""
    lo = _count_le(dec_1, a)                      # largest c with dec[c] <= a
    top = lo >= maxc
    lo_c = torch.clamp(lo, max=maxc - 1)
    vlo, vhi = dec[lo_c], dec[lo_c + 1]
    p = (a - vlo) / (vhi - vlo)
    up = u < p
    idx = torch.where(top, torch.full_like(lo, maxc), torch.where(up, lo_c + 1, lo_c))
    val = torch.where(top, dec[maxc].expand_as(a), torch.where(up, vhi, vlo))
    return idx, val


def rt8_sr(x: torch.Tensor, u: torch.Tensor, scale=SCALE) -> torch.Tensor:
    T = tables(x.device)
    s = _f32(scale, x.device)
    _, v = _sr_pick(T["dec8_1"], T["dec8"], MAX_CODE, x.abs() * s, u)
    r = v / s
    r = torch.where(x < 0, -r, r)
    return torch.where((x > 0) | (x < 0), r, torch.zeros_like(r))


def pow2_scale(mx: torch.Tensor) -> torch.Tensor:
    """Largest power of two s with mx * s <= DEC8[127] (exact, via frexp)."""
    mx = mx.to(torch.float32)
    lim = _f32(DEC8[MAX_CODE], mx.device)
    r = lim / mx
    _, e = torch.frexp(torch.where(torch.isinf(r) | ~(mx > 0), torch.ones_like(r), r))
    e = e - 1
    e = torch.where(torch.isinf(r), torch.full_like(e, 126), e).clamp(-126, 126)
    s = torch.ldexp(torch.ones_like(r), e)
    return torch.where(mx > 0, s, torch.ones_like(s))


# ---------------------------------------------------------------- fixed point

def fx(x: torch.Tensor, step=FXSTEP) -> torch.Tensor:
    """clamp(round_half_away(x / step), -128, 127) * step (C: lroundf)."""
    st = _f32(step, x.device)
    q = (x / st).clamp(-128.0, 127.0)
    t = torch.trunc(q)
    f = q - t                                      # exact
    r = t + (f >= 0.5).to(q.dtype) - (f <= -0.5).to(q.dtype)
    return r * st


# ---------------------------------------------------------------- 16 bit

def enc16(x: torch.Tensor, scale, u: torch.Tensor | None = None) -> torch.Tensor:
    """Signed 16-bit codes (int16); u=None -> round half up, else SR."""
    T = tables(x.device)
    a = x.abs() * _f32(scale, x.device)
    if u is None:
        M = _count_le(T["thr16"], a)
    else:
        M, _ = _sr_pick(T["dec16_1"], T["dec16"], MAX_M, a, u)
    M = M.to(torch.int16)
    M = torch.where((x > 0) | (x < 0), M, torch.zeros_like(M))
    return torch.where(x < 0, -M, M)


def dec16(c: torch.Tensor, scale) -> torch.Tensor:
    T = tables(c.device)
    m = T["dec16"][c.abs().long()] / _f32(scale, c.device)
    return torch.where(c < 0, -m, m)


def w16_fwd(c: torch.Tensor, scale=SCALE) -> torch.Tensor:
    """Forward view of a master code: its high byte, (|M| + 128) >> 8."""
    T = tables(c.device)
    hi = (c.abs().long() + SUB // 2) // SUB
    m = T["dec8"][hi] / _f32(scale, c.device)
    return torch.where(c < 0, -m, m)


# ---------------------------------------------------------------- STE

class _QAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x.abs() <= max_abs(scale))
        return rt8(x, scale)

    @staticmethod
    def backward(ctx, g):
        (keep,) = ctx.saved_tensors
        return g * keep, None


class _QFx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        lim = np.float32(127.0) * np.float32(step)
        ctx.save_for_backward(x.abs() <= float(lim))
        return fx(x, step)

    @staticmethod
    def backward(ctx, g):
        (keep,) = ctx.saved_tensors
        return g * keep, None


class _QFxMaxRel(torch.autograd.Function):
    """Max-relative fixed-point logits: d = x - max(x) <= 0 is stored as an
    unsigned byte on the grid step * {0..255}, i.e. [-25.5, 0] at step 0.1.
    STE: gradient 1 where d is in range, 0 where it clamped (more than 25.5
    below the top logit, p < 1e-11). CE is shift-invariant, so the max
    carries no gradient."""

    @staticmethod
    def forward(ctx, x, step):
        st = np.float32(step)
        m = x.amax(-1, keepdim=True)
        d = x - m
        lo = float(np.float32(-255.0) * st)
        ctx.save_for_backward(d >= lo)
        stt = _f32(st, x.device)
        q = (d / stt).clamp(-255.0, 0.0)               # unsigned byte: 0 … 255 steps
        t = torch.trunc(q)
        r = t - ((q - t) <= -0.5).to(q.dtype)          # round half away (q <= 0)
        return m + r * stt

    @staticmethod
    def backward(ctx, g):
        (keep,) = ctx.saved_tensors
        return g * keep, None


class _QActT(torch.autograd.Function):
    """Byte-grid STE at a tensor (power-of-two) scale; gradient 0 where
    the value clamped at the top of the grid."""

    @staticmethod
    def forward(ctx, x, s):
        lim = _f32(DEC8[MAX_CODE], x.device) / s
        ctx.save_for_backward(x.abs() <= lim)
        return rt8_fast(x, s)

    @staticmethod
    def backward(ctx, g):
        (keep,) = ctx.saved_tensors
        return g * keep, None


PROB_SCALE = 4096.0     # pow2_scale(1.0): softmax probabilities in [0, 1]


def act_fixed(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _QActT.apply(x, _f32(scale, x.device))


def logit_maxrel(x: torch.Tensor, step=FXSTEP) -> torch.Tensor:
    return _QFxMaxRel.apply(x, step)


def logit_fx(x: torch.Tensor, step=FXSTEP) -> torch.Tensor:
    return _QFx.apply(x, np.float32(step))


@dataclass
class QuantConfig:
    """Numerics switches (plan/001 §2). enabled=False is the float oracle."""
    enabled: bool = True
    w2: bool = True          # two-byte master weights (else 1 byte + SR)
    w2_sr: bool = False      # SR on the low byte
    sr: bool = True          # SR for one-byte storage
    pure_int: bool = False   # gradients on the byte grid before the optimizer
    scale: float = SCALE
    fxstep: float = float(FXSTEP)
    # transformer forward (models/transformer.QAT); enabled = weight storage
    acts: bool = False
    resid: str = "fp"
    softmax: str = "none"
    probs: bool = False

    def act(self, x: torch.Tensor) -> torch.Tensor:
        return _QAct.apply(x, self.scale) if self.enabled else x

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        return _QFx.apply(x, np.float32(self.fxstep)) if self.enabled else x

    def project_(self, p: torch.Tensor) -> None:
        """Put a freshly initialized parameter on its storage grid."""
        if self.enabled:
            with torch.no_grad():
                p.copy_(rt8(p, self.scale))
