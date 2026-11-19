"""M7 (plan/001 §5): position curve and cross-window loss of a byte LM,
the baseline any recurrent state (plan 002) must beat.

Data: the sequential batcher over the val bucket (B document-continuous
streams, fixed seed), so window t+1 of a stream continues window t.

position curve  bits per byte at each offset 0..T-1 of a window, averaged
                over all windows (reported in log-spaced bins).
cross-window    for continuation windows (the first W bytes lie in a doc
                that began before the window): bits of those W bytes
                cold:  from the window alone (what a non-recurrent model gets)
                warm:  with the preceding T-W bytes as context (a window
                       shifted back by T-W; what a state carrying the
                       previous window could at best recover)
                and, for reference, the first W bytes of windows whose
                first byte starts a document (a true cold start).
carried         (recurrent models, plan 002) the same windows scored in
                stream order with the recurrent state handed from each
                window to the next; "closed" = (cold - carried) / gap.

  uv run python -m intgrad.position runs/<run> [--streams 128] [--windows 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from intgrad import data

W_DEFAULT = 64


def stream_windows(index: data.Index, bucket: int, streams: int, T: int, n: int, seed: int):
    """n+1 consecutive windows per stream: x [S, n+1, T+1] (inputs and the
    next-byte target), doc_start [S, n+1, T+1]. Window j+1 continues j."""
    bt = data.SequentialBatcher(index, bucket, batch=streams, seq=T, seed=seed)
    xs, st = [], []
    for _ in range(n + 1):
        b = bt.next()
        xs.append(np.concatenate([b.x, b.y[:, -1:]], 1))
        st.append(np.concatenate([b.doc_start, np.zeros((streams, 1), bool)], 1))
    return np.stack(xs, 1), np.stack(st, 1)


@torch.no_grad()
def carried_bits(model, x: torch.Tensor, st: torch.Tensor, batch: int) -> torch.Tensor:
    """Windows of each stream in order, the recurrent state handed from one
    to the next (reset at document starts). x [S, n+1, T+1] -> bits
    [S, n, T] for windows 1..n (window 0 only builds the state)."""
    S, n1, T1 = x.shape
    out = torch.empty(S, n1 - 1, T1 - 1, device=x.device)
    for i in range(0, S, batch):
        state = None
        for j in range(n1):
            xb = x[i:i + batch, j]
            lg, state = model.forward_state(xb[:, :-1], state, st[i:i + batch, j, :-1])
            if j:
                out[i:i + batch, j - 1] = F.cross_entropy(
                    lg.float().reshape(-1, 256), xb[:, 1:].reshape(-1).long(),
                    reduction="none").view(len(xb), -1) / math.log(2)
    return out


@torch.no_grad()
def window_bits(model, x: torch.Tensor, batch: int) -> torch.Tensor:
    """x [N, T+1] -> bits [N, T] for x[:, 1:] from x[:, :-1]."""
    out = torch.empty(x.shape[0], x.shape[1] - 1, device=x.device)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        lg = model(xb[:, :-1]).float()
        out[i:i + batch] = F.cross_entropy(lg.reshape(-1, 256), xb[:, 1:].reshape(-1).long(),
                                           reduction="none").view(len(xb), -1) / math.log(2)
    return out


def _bins(T: int) -> list[tuple[int, int]]:
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, T]
    edges = sorted({e for e in edges if e <= T})
    return list(zip(edges[:-1], edges[1:]))


def _mean_se(v: np.ndarray) -> dict:
    return {"bpb": round(float(v.mean()), 4),
            "se": round(float(v.std(ddof=1) / math.sqrt(len(v))), 4) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def measure(model, x: np.ndarray, st: np.ndarray, W: int = W_DEFAULT, batch: int = 32,
            device="cuda") -> dict:
    """x, st from stream_windows. Windows 1..n of each stream are scored
    (window 0 only supplies the warm context of window 1)."""
    model.eval()
    S, n1, T1 = x.shape
    T = T1 - 1
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    cold = window_bits(model, xt[:, 1:].reshape(-1, T1), batch).view(S, n1 - 1, T)
    # warm: last T-W inputs of window j-1, then the first W(+1) bytes of window j
    warm_x = torch.cat([xt[:, :-1, W:T], xt[:, 1:, :W + 1]], 2).reshape(-1, T1)
    warm = window_bits(model, warm_x, batch).view(S, n1 - 1, T)[..., T - W:]
    rec = getattr(model, "rec", "none") != "none"
    if rec:
        carried = carried_bits(model, xt, torch.from_numpy(np.ascontiguousarray(st)).to(device),
                               batch).double().cpu().numpy()
    cold, warm = cold.double().cpu().numpy(), warm.double().cpu().numpy()
    stw, xw = st[:, 1:], x[:, 1:]
    # continuation: no doc starts and no EOD among the first W inputs/targets
    cont = ~stw[..., :W + 1].any(-1) & ~(xw[..., :W + 1] == data.EOD).any(-1)
    start = stw[..., 0]
    curve = cold.reshape(-1, T).mean(0)
    per_win = cold.reshape(-1, T)
    res = {
        "T": T, "W": W, "windows": int(S * (n1 - 1)), "streams": S,
        "bpb": _mean_se(per_win.mean(1)),
        "position": [{"lo": lo, "hi": hi, "bpb": round(float(curve[lo:hi].mean()), 4)}
                     for lo, hi in _bins(T)],
        "cross_window": {
            "cold": _mean_se(cold[..., :W][cont].mean(-1)),
            "warm": _mean_se(warm[cont].mean(-1)),
            "gap": _mean_se((cold[..., :W] - warm)[cont].mean(-1)),
            "doc_start": _mean_se(cold[..., :W][start].mean(-1)) if start.any() else None,
        },
    }
    if rec:                                  # plan 002: the state carried from the previous window
        cw = res["cross_window"]
        cw["carried"] = _mean_se(carried[..., :W][cont].mean(-1))
        cw["closed"] = round((cw["cold"]["bpb"] - cw["carried"]["bpb"]) / max(cw["gap"]["bpb"], 1e-9), 4)
        res["bpb_carried"] = _mean_se(carried.reshape(-1, T).mean(1))
        res["position_carried"] = [{"lo": lo, "hi": hi, "bpb": round(float(carried.reshape(-1, T)[:, lo:hi].mean()), 4)}
                                   for lo, hi in _bins(T)]
    model.train()
    return res, curve


def load_run(run: Path, device):
    from intgrad.quant import QuantConfig
    from intgrad.train import DEFAULTS, build_model, merge
    cfg = merge(DEFAULTS, tomllib.loads((run / "config.toml").read_text()))
    m = build_model(cfg, QuantConfig(**cfg["quant"]), torch.Generator().manual_seed(0)).to(device)
    m.load_state_dict(torch.load(run / "ckpt.pt", map_location=device, weights_only=False)["model"])
    return cfg, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--streams", type=int, default=128)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg, model = load_run(a.run, a.device)
    index = data.Index(cfg["data"]["index"])
    x, st = stream_windows(index, data.VAL, a.streams, model.T, a.windows, a.seed)
    res, curve = measure(model, x, st, a.W, a.batch, a.device)
    res["data"] = {"bucket": "val", "seed": a.seed,
                   "blake2b": hashlib.blake2b(x.tobytes(), digest_size=16).hexdigest()}
    (a.run / "position.json").write_text(json.dumps(res, indent=1))
    np.save(a.run / "position_curve.npy", curve)
    cw = res["cross_window"]
    print(f"{a.run.name}: bpb {res['bpb']['bpb']:.4f} | first {a.W} B of continuation windows: "
          f"cold {cw['cold']['bpb']:.4f}, warm {cw['warm']['bpb']:.4f}, gap {cw['gap']['bpb']:.4f}"
          f" ± {cw['gap']['se']:.4f} (n {cw['gap']['n']})")
    print("  position: " + "  ".join(f"[{p['lo']},{p['hi']}) {p['bpb']:.3f}" for p in res["position"]))
    if "carried" in cw:
        print(f"  carried state: window bpb {res['bpb_carried']['bpb']:.4f}, first {a.W} B "
              f"{cw['carried']['bpb']:.4f}, gap closed {100 * cw['closed']:.1f}%")
        print("  position (carried): " + "  ".join(f"[{p['lo']},{p['hi']}) {p['bpb']:.3f}"
                                                 for p in res["position_carried"]))


if __name__ == "__main__":
    main()
