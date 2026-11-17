"""ctypes binding to the C oracle (build/libintgrad_ref.so) for parity tests.

Build it first:  cmake -B build -G Ninja && cmake --build build
Override the path with IG_REF_LIB.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("IG_REF_LIB", ROOT / "build" / "libintgrad_ref.so"))


def load():
    if not LIB.exists():
        return None
    L = C.CDLL(str(LIB))
    f, i16, u8, vp = C.c_float, C.c_int16, C.c_uint8, C.c_void_p
    sig = {
        "ig_encode_s": (u8, [f, f]), "ig_decode_s": (f, [u8, f]),
        "ig_roundtrip_s": (f, [f, f]), "ig_roundtrip_sr_s": (f, [f, f, f]),
        "ig_pow2_scale": (f, [f]), "ig_fx_roundtrip": (f, [f]),
        "ig_enc16": (i16, [f, f, f]), "ig_dec16": (f, [i16, f]),
        "ig_w16_fwd": (f, [i16]), "ig_lq_table": (C.POINTER(f), [C.c_int, C.POINTER(C.c_int)]),
        "ig_mlp_new": (vp, [C.c_int, C.c_int, C.c_uint]), "ig_mlp_free": (None, [vp]),
        "ig_mlp_nparams": (C.c_long, [vp]),
        "ig_mlp_export": (None, [vp, vp, vp, vp]), "ig_mlp_import": (None, [vp, vp, vp, vp]),
        "ig_lion_get_state": (C.c_long, [vp]), "ig_lion_set_state": (None, [vp, C.c_long]),
        "ig_sr_get_state": (C.c_uint64, []), "ig_sr_set_state": (None, [C.c_uint64]),
        "ig_mlp_window_backward": (C.c_double, [vp, vp]), "ig_mlp_logits": (None, [vp, vp, vp]),
        "ig_mlp_lion_update": (None, [vp, f]), "ig_mlp_zero_grad": (None, [vp]),
        "ig_mlp_scale_grads": (None, [vp, f]), "ig_mlp_opt_reset": (None, []),
        "ig_set_lion_betas": (None, [f, f]), "ig_fx_maxrel": (None, [vp, vp, C.c_int]),
        "ig_set_modes": (None, [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, f]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(L, name)
        fn.restype, fn.argtypes = res, args
    return L


def ptr(a: np.ndarray):
    return a.ctypes.data_as(C.c_void_p)


def table(L, which: int) -> np.ndarray:
    n = C.c_int()
    p = L.ig_lq_table(which, C.byref(n))
    return np.ctypeslib.as_array(p, shape=(n.value,)).copy()


def u8_to_signed(c: int) -> int:
    return -(c - 128) if c >= 128 else c


class XorShift64:
    """The C SR stream: s ^= s<<13; s ^= s>>7; s ^= s<<17; u = (s>>40)/2^24."""

    def __init__(self, state: int):
        self.s = state & 0xFFFFFFFFFFFFFFFF

    def uniforms(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        s, M = self.s, 0xFFFFFFFFFFFFFFFF
        for i in range(n):
            s ^= (s << 13) & M
            s ^= s >> 7
            s ^= (s << 17) & M
            out[i] = np.float32(s >> 40) / np.float32(16777216.0)
        self.s = s
        return out
