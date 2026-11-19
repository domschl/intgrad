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


def utf8_complete(x: torch.Tensor) -> torch.Tensor:
    """[B, T] bytes -> bool [B, T]: byte t completes a character, decided
    from bytes <= t only (a lead byte announces its length, so no
    look-ahead). ASCII completes itself; a continuation byte completes the
    sequence its lead announced; a stray continuation byte (window start
    mid-character, broken text) counts as a 1-byte character. A lead whose
    sequence breaks never completes; the next character does."""
    x = x.long()
    cont = (x & 0xC0) == 0x80
    ln = torch.where(x < 0x80, 1, torch.where(x >= 0xF0, 4, torch.where(
        x >= 0xE0, 3, torch.where(x >= 0xC0, 2, 0))))

    def back(t, m, fill):                         # value at position t - m
        return F.pad(t, (m, 0), value=fill)[:, :t.shape[1]]

    c1, c2 = back(cont, 1, False), back(cont, 2, False)
    l1, l2, l3 = back(ln, 1, 0), back(ln, 2, 0), back(ln, 3, 0)
    done = (cont & (l1 == 2)) | (cont & c1 & (l2 == 3)) | (cont & c1 & c2 & (l3 == 4))
    inside = cont & ((l1 == 3) | (l1 == 4) | (c1 & (l2 == 4)))
    return (ln == 1) | done | (cont & ~done & ~inside)


def patch_ends(x: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """[B, T] bytes -> int [B, T]: the global slot whose patch ends at byte
    t, or -1. Slots are positions of the global sequence (T // k of them).
    fixed: every k bytes (slot j ends at jk+k-1).
    utf8:  slot j ends at the first character end at or after jk+k-1 (the
           k-byte grid snapped forward so no character is split; a slot
           whose end coincides with the previous one stays empty).
    chars: slot j = characters jk+1 … jk+k (k complete characters)."""
    B, T = x.shape
    t = torch.arange(T, device=x.device)
    if mode == "fixed":
        return torch.where(t % k == k - 1, t // k, -1).expand(B, T)
    comp = utf8_complete(x)
    if mode == "chars":
        c = torch.cumsum(comp.long(), 1)
        return torch.where(comp & (c % k == 0), c // k - 1, -1)
    if mode != "utf8":
        raise ValueError(f"patching {mode!r}")
    big = torch.full_like(x, T, dtype=torch.long)
    pos = torch.where(comp, t.expand(B, T), big)
    nxt = torch.flip(torch.cummin(torch.flip(pos, [1]), 1).values, [1])   # next char end >= t
    e = nxt[:, k - 1::k]                                                   # [B, T // k]
    prev = F.pad(e, (1, 0), value=-1)[:, :-1]
    j = torch.arange(e.shape[1], device=x.device).expand_as(e)
    valid = (e < T) & (e != prev)
    out = torch.full((B, T + 1), -1, dtype=torch.long, device=x.device)
    out = out.scatter(1, torch.where(valid, e, T), torch.where(valid, j, -1))
    return out[:, :T]


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
                 byte_mix: str = "conv", byte_window: int = 64, byte_dh: int = 64,
                 patching: str = "fixed", enc_window: int = 0,
                 enc_wide: int = 0, skip: str = ""):
        """T: byte context (a multiple of k); the global stack sees T/k patches.
        byte_mix "conv": ByteBlocks (causal depthwise conv + MLP);
        "attn": transformer Blocks with sliding-window attention over the
        last byte_window bytes (heads of width byte_dh), as in SpaceByte;
        "patch": transformer Blocks whose attention stays inside the
        current patch (MEGABYTE-style): everything beyond the patch must go
        through the global core (X21: the wider byte paths short-cut it).
        patching (see patch_ends): "fixed" k-byte patches; "utf8" k-byte
        grid snapped to character ends; "chars" k characters per patch.
        utf8/chars need byte_mix "patch" (segment-masked byte attention).
        Both paths have the same width: the patch vector is built from the
        last k encoder states up to the patch end, and the upsample gives
        each later byte its own vector by offset from that end (clipped at
        k-1), so on pure ASCII utf8 patching is exactly fixed patching.
        enc_window > 0 (utf8/chars only): a wide encoder whose attention
        spans the last enc_window bytes across patch boundaries, so a
        patch vector can use the text before its patch. The decoder stays
        patch-local, and its skip then comes from the byte embedding, not
        the encoder: anything beyond the current patch still reaches the
        decoder only through the core (no X21 shortcut).
        enc_wide > 0 (with enc_window): keep the enc patch-local layers
        (the decoder skip comes from them) and stack enc_wide wide layers
        on top that feed only the down projection.
        skip: "enc" (patch-local encoder output) or "emb" (byte embedding);
        default "emb" for an all-wide encoder, else "enc"."""
        super().__init__()
        if T % k:
            raise ValueError(f"T={T} must be a multiple of k={k}")
        self.d, self.L, self.H, self.T, self.k, self.d_b = d, L, H, T, k, d_b
        self.zloss = zloss
        self.qat = qat = QAT(**qat) if isinstance(qat, dict) else (qat or QAT())
        kernel = kernel or max(4, k)
        n_res = enc + enc_wide + dec
        if byte_mix not in ("conv", "attn", "patch"):
            raise ValueError(f"byte_mix {byte_mix!r}")
        self.byte_mix, self.byte_H = byte_mix, max(1, d_b // byte_dh)
        if patching not in ("fixed", "utf8", "chars"):
            raise ValueError(f"patching {patching!r}")
        if patching != "fixed" and byte_mix != "patch":
            raise ValueError("utf8/chars patching needs byte_mix 'patch'")
        self.patching = patching
        # longest patch of valid UTF-8: utf8 snaps a k-grid end forward by
        # <= 3 bytes; chars holds k characters of <= 4 bytes. (Broken runs
        # of lead bytes can be longer; the local band then drops their
        # oldest keys, still causal.)
        self.seg_window = k + 3 if patching == "utf8" else 4 * k
        self.dense_seg = False
        var = patching != "fixed"
        if enc_window and not var:
            raise ValueError("enc_window needs utf8/chars patching")
        if enc_wide and not enc_window:
            raise ValueError("enc_wide needs enc_window")
        self.enc_window, self.enc_wide = enc_window, enc_wide
        all_wide = bool(enc_window) and not enc_wide
        skip = skip or ("emb" if all_wide else "enc")
        if skip not in ("enc", "emb") or (skip == "enc" and all_wide):
            raise ValueError(f"skip {skip!r}: an all-wide encoder must skip from 'emb' (X21 shortcut)")
        self.skip, self.all_wide = skip, all_wide

        def byte_block():
            if byte_mix == "conv":
                return ByteBlock(d_b, kernel, n_res, qat)
            if byte_mix == "patch":                 # no q/k/v conv: it would cross patches
                return Block(d_b, self.byte_H, n_res, qk_norm, 0, qat, patch=0 if var else k)
            return Block(d_b, self.byte_H, n_res, qk_norm, qkv_conv, qat, window=byte_window)

        self.emb = nn.Embedding(256, d_b)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.enc = nn.ModuleList(
            Block(d_b, self.byte_H, n_res, qk_norm, 0, qat, window=enc_window) if all_wide
            else byte_block()
            for _ in range(enc))
        self.enc_w = nn.ModuleList(          # wide layers that feed only the down path
            Block(d_b, self.byte_H, n_res, qk_norm, 0, qat, window=enc_window)
            for _ in range(enc_wide))
        self.n_down = nn.RMSNorm(d_b)
        kd = k * d_b                            # k byte states per patch, both ways
        self.down = nn.Linear(kd, d, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(kd))
        self.blocks = nn.ModuleList(Block(d, H, L, qk_norm, qkv_conv, qat) for _ in range(L))
        self.n_up = nn.RMSNorm(d)
        self.up = nn.Linear(d, kd, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        self.dec = nn.ModuleList(byte_block() for _ in range(dec))
        self.nf = nn.RMSNorm(d_b)
        self.qs = _sites("emb down up nf", qat.acts)
        self._rope = None
        self._brope = None

    def _bblk(self, blk, h, mask=None):
        if self.byte_mix == "conv":
            return blk(h)
        if self._brope is None or self._brope[0].device != h.device:
            self._brope = rope_cache(self.T, self.d_b // self.byte_H, h.device)
        return blk(h, self._brope, mask=mask)

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
        Tg = T // k
        mask = slot = None
        if self.patching != "fixed":
            slot = patch_ends(x, k, self.patching)            # [B, T], -1 = no end here
            end = slot >= 0
            seg = torch.cumsum(end.long(), 1) - end.long()    # patch index of each byte
            if self.dense_seg:                                # reference path (tests)
                tt = torch.arange(T, device=x.device)
                mask = ((seg[:, :, None] == seg[:, None, :]) & (tt[None, :, None] >= tt[None, None, :]))[:, None]
            else:                                             # patches are <= seg_window bytes
                mask = (seg, self.seg_window)
        h = _q(qs, "emb", self.emb(x.long()))
        e = h
        emask = None if self.all_wide else mask           # wide: SDPA sliding-window band
        for blk in self.enc:
            h = self._bblk(blk, h, emask)
        skip = e if self.skip == "emb" else h
        for blk in self.enc_w:                            # cross-patch, down path only
            h = self._bblk(blk, h)
        if tap is not None:
            tap("emb", h)
        if slot is None:
            g = self.down(_q(qs, "down", self.n_down(h)).reshape(B, Tg, k * self.d_b))
        else:                                   # last k bytes up to each patch end -> its slot
            hn = F.pad(_q(qs, "down", self.n_down(h)), (0, 0, k - 1, 0))
            v = self.down(hn.unfold(1, k, 1).transpose(2, 3).reshape(B, T, k * self.d_b))
            idx = torch.where(slot >= 0, slot, Tg)[..., None].expand(-1, -1, self.d)
            g = torch.zeros(B, Tg + 1, self.d, device=x.device, dtype=v.dtype).scatter(1, idx, v)[:, :Tg]
        rope = self.rope(x.device)
        for i, blk in enumerate(self.blocks):
            if i == self.mid:
                if tap is not None:
                    tap("h_mid", g)
                if mid_hook is not None:
                    g = mid_hook(g)
            g = blk(g, rope, tap, i)
        if slot is None:
            u = self.up(_q(qs, "up", self.n_up(g))).reshape(B, T, self.d_b)
            u = F.pad(u, (0, 0, k - 1, 0))[:, :T]      # patch j -> bytes jk+k-1 ..
        else:                                   # latest patch completed at or before t
            U = F.pad(self.up(_q(qs, "up", self.n_up(g))), (0, 0, 1, 0))   # slot -1 -> 0
            U = U.reshape(B, (Tg + 1) * k, self.d_b)
            tt = torch.arange(T, device=x.device).expand(B, T)
            last = torch.cummax(slot, 1).values + 1
            endp = torch.cummax(torch.where(slot >= 0, tt, -1), 1).values
            off = (tt - endp).clamp(max=k - 1)
            u = torch.gather(U, 1, (last * k + off)[..., None].expand(-1, -1, self.d_b))
        h = skip + u
        for blk in self.dec:
            h = self._bblk(blk, h, mask)
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
