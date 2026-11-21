"""Single training entry point (plan/001 §3): every run is a TOML config.

    uv run ig-train configs/x4-mlp-q.toml [--set train.seed=7] [--resume RUN_DIR]

Run directory runs/<name>-<seed>/:
    config.toml   resolved config      meta.json     git hash, index/eval hashes
    metrics.jsonl one record per log/eval step   ckpt.pt  model+optimizer+batcher

Schedule: linear warmup to lr, then
    sched "cosine"  cosine to 0 at train.steps (the total horizon; the C trainer's)
    sched "wsd"     warmup-stable-decay: lr held, then the last train.decay share
                    of steps decays to 0, shape "sqrt" (1 - sqrt(p)) or "linear".
                    A paused stable-phase run (train.stop) can be resumed with a
                    longer train.steps: the decay moves with the horizon.
Loss = mean CE over the batch.

Live logging (ig-dash reads it): val_every > 0 logs a quick val estimate on
every val_frac-th eval-v1 chunk; eval_samples > 0 writes that many samples
of eval_sample_len bytes to samples/<step>.bin at each eval_every point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from intgrad import data
from intgrad.eval_set import EvalSet
from intgrad.evaluate import evaluate
from intgrad.lion import Lion
from intgrad.quant import QuantConfig

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "name": "run",
    "model": {"kind": "mlp", "ctx": 4, "hidden": 96},
    "quant": {"enabled": True, "w2": True, "w2_sr": False, "sr": True, "pure_int": False,
              # transformer only (M4): enabled = weight storage; forward sites:
              "acts": False, "resid": "fp", "softmax": "none", "probs": False},
    "train": {"steps": 4000, "batch": 32, "lr": 2e-3, "wd": 0.1, "warmup": 200,
              "sched": "cosine", "seed": 42, "log_every": 100, "eval_every": 0,
              "decay": 0.2, "decay_shape": "sqrt",   # sched "wsd"
              "val_every": 0, "val_frac": 8,         # quick val: every val_frac-th eval chunk
              "eval_samples": 0, "eval_sample_len": 256,
              "stop": 0,  # >0: pause at this step (horizon stays train.steps)
              "telemetry_every": 0, "samples": 0, "sample_len": 1024,
              "compile": False, "final_eval": True, "betas": [0.9, 0.99],
              "micro_batch": 0,  # >0: gradient accumulation over micro-batches
              "norm_wd": 0.0, "conv_decay": "zero",
              "lr_emb": 1.0, "lr_vec": 1.0,  # lr multipliers: embedding/head, gains
              "rec_carry": True,  # plan 002: hand the recurrent state to the next window
              "rec_drop": 0.0,   # carry: share of rows per step whose carried state is zeroed (plan 002 N2b)
              "device": "cuda" if torch.cuda.is_available() else "cpu"},
    "data": {"index": "data/split-v1/idx.bin", "eval": "data/eval-v1",
             "batcher": "random",   # "sequential": document-continuous streams (plan 002)
             "streams": 0,          # sequential: pool of S streams (0 = batch), B drawn per step
             "weights": {}},        # sequential: {source: weight}, byte shares per source (plan 003)
}



class _Peaks:
    """Peak CUDA memory of training steps vs evaluation, tracked separately:
    eval batches are sized by the model's attention (max_batch), so a
    combined peak says little about what training needs."""

    def __init__(self, dev):
        self.cuda, self.dev, self.train, self.ev = dev.type == "cuda", dev, 0, 0

    def eval(self, fn):
        if not self.cuda:
            return fn()
        self.train = max(self.train, torch.cuda.max_memory_allocated(self.dev))
        torch.cuda.reset_peak_memory_stats(self.dev)
        r = fn()
        self.ev = max(self.ev, torch.cuda.max_memory_allocated(self.dev))
        torch.cuda.reset_peak_memory_stats(self.dev)
        return r

    @staticmethod
    def gib(b):
        return round(b / 2**30, 2)

def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def set_override(cfg: dict, spec: str) -> None:
    key, val = spec.split("=", 1)
    *path, last = key.split(".")
    d = cfg
    for p in path:
        d = d.setdefault(p, {})
    d[last] = tomllib.loads(f"v = {val}")["v"]


def build_model(cfg: dict, qc: QuantConfig, gen: torch.Generator):
    m = cfg["model"]
    if m["kind"] == "mlp":
        from intgrad.models.mlp import ByteMLP
        return ByteMLP(m["ctx"], m["hidden"], qc, generator=gen)
    if m["kind"] == "transformer":
        from intgrad.models import hourglass, transformer
        qat = transformer.QAT(acts=qc.acts, resid=qc.resid, softmax=qc.softmax, probs=qc.probs)
        if m.get("arch", "flat") == "hourglass":     # M6
            kw = {k: m[k] for k in ("d", "L", "H", "T", "k", "d_b", "enc", "dec", "kernel",
                                    "zloss", "qk_norm", "qkv_conv", "byte_mix", "byte_window",
                                    "byte_dh", "patching", "enc_window",
                                    "enc_wide", "skip") if k in m}
            model = hourglass.build(m.get("preset"), qat=qat, **kw)
        else:
            kw = {k: m[k] for k in ("d", "L", "H", "T", "zloss", "softcap", "qk_norm",
                                    "stem", "stem_dim", "qkv_conv", "logit_quant",
                                    "rec", "rec_conv", "rec_chunk", "rec_state", "rec_head",
                                    "rec_nonlin") if k in m}
            model = transformer.build(m.get("preset"), qat=qat, **kw)
        for p in model.parameters():          # weights start on their grid
            qc.project_(p)
        return model
    raise ValueError(f"unknown model kind {m['kind']!r}")


def context_len(cfg: dict, model) -> int:
    return model.ctx if cfg["model"]["kind"] == "mlp" else model.T


def param_groups(model, wd: float, norm_wd: float = 0.0, conv_decay: str = "zero",
                 lr_emb: float = 1.0, lr_vec: float = 1.0) -> list[dict]:
    """Weight decay groups.
    matrices (2-D, incl. embeddings): wd toward 0 (Lion bound |w| <= 1/wd);
    gains/biases (1-D): no decay by default, or norm_wd toward their init
        value (gains -> 1; Lion bound |g - 1| <= 1/norm_wd);
    conv kernels (3-D): conv_decay "zero" (decay toward 0 with the matrices,
        the M3 default), "init" (toward their identity init) or "none".
    lr_emb / lr_vec: lr multipliers ("lr_mult") for the byte embedding (tied
        output head) and the 1-D gains (M5)."""
    if cfg_is_mlp(model):                      # C-oracle parity: wd everywhere
        return [{"params": list(model.parameters()), "wd": wd}]
    emb = model.emb.weight if lr_emb != 1.0 else None
    mats = [p for p in model.parameters() if p.ndim == 2 and p is not emb]
    vecs = [p for p in model.parameters() if p.ndim < 2]
    convs = [p for p in model.parameters() if p.ndim >= 3]
    groups = [{"params": mats, "wd": wd}]
    groups.append({"params": vecs, "wd": norm_wd, "anchor": "init", "lr_mult": lr_vec} if norm_wd
                  else {"params": vecs, "wd": 0.0, "lr_mult": lr_vec})
    if emb is not None:
        groups.append({"params": [emb], "wd": wd, "lr_mult": lr_emb})
    if convs:
        if conv_decay == "zero":
            groups[0]["params"] = mats + convs
        elif conv_decay == "init":
            groups.append({"params": convs, "wd": wd, "anchor": "init"})
        elif conv_decay == "none":
            groups.append({"params": convs, "wd": 0.0})
        else:
            raise ValueError(f"conv_decay {conv_decay!r}")
    return groups


def cfg_is_mlp(model) -> bool:
    from intgrad.models.mlp import ByteMLP
    return isinstance(model, ByteMLP)


def utf8_valid_fraction(samples: list[bytes]) -> float:
    """Share of samples that decode as UTF-8 (a truncated final character
    at the end of a sample is allowed)."""
    ok = 0
    for b in samples:
        try:
            b.decode("utf-8")
            ok += 1
        except UnicodeDecodeError as e:
            ok += e.start >= len(b) - 3 and e.reason == "unexpected end of data"
    return ok / max(len(samples), 1)


def lr_at(step: int, t: dict) -> float:
    lr, warm, steps = t["lr"], t["warmup"], t["steps"]
    if warm > 0 and step < warm:
        return lr * (step + 1) / warm
    if t["sched"] == "cosine" and steps > warm:
        return lr * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / (steps - warm)))
    if t["sched"] == "wsd":
        s0 = max(warm, steps - round(t["decay"] * steps))
        if step < s0 or steps <= s0:
            return lr
        p = (step - s0) / (steps - s0)
        if t["decay_shape"] == "sqrt":
            return lr * (1.0 - math.sqrt(p))
        if t["decay_shape"] == "linear":
            return lr * (1.0 - p)
        raise ValueError(f"train.decay_shape {t['decay_shape']!r}")
    if t["sched"] != "const":
        raise ValueError(f"train.sched {t['sched']!r}")
    return lr


def write_samples(model, path: Path, n: int, length: int, seed: int, dev) -> list[bytes]:
    """n samples of length bytes from an EOD prompt (own generator: the
    training RNG streams are untouched), each terminated by 0x03 0x0a."""
    g = torch.Generator(device=dev).manual_seed(seed)
    prompt = torch.full((n, 1), data.EOD, dtype=torch.uint8, device=dev)
    with torch.no_grad():
        blobs = [bytes(r) for r in model.sample(prompt, length, generator=g).cpu().numpy()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(r + b"\x03\n" for r in blobs))
    return blobs


def blake(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def toml_dump(d: dict, prefix: str = "") -> str:
    lines, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            lines.append(f"{k} = {json.dumps(v)}")
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v in tables:
        out += f"\n[{prefix}{k}]\n" + toml_dump(v, f"{prefix}{k}.")
    return out


def run(cfg: dict, resume: Path | None = None, out_root: Path = ROOT / "runs",
        overrides: list[str] = ()) -> dict:
    t, q = cfg["train"], cfg["quant"]
    dev = torch.device(t["device"])
    run_dir = resume or out_root / f"{cfg['name']}-s{t['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    idx_path, ev_path = ROOT / cfg["data"]["index"], ROOT / cfg["data"]["eval"]
    if not resume:
        (run_dir / "config.toml").write_text(toml_dump(cfg))
        (run_dir / "meta.json").write_text(json.dumps({
            "git": git_hash(), "index_blake2b": blake(idx_path),
            "eval": json.loads((ev_path / "meta.json").read_text())["val"]["blake2b"],
            "created": int(time.time())}, indent=2) + "\n")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(t["seed"])
    gen = torch.Generator().manual_seed(t["seed"])
    qc = QuantConfig(**q)
    model = build_model(cfg, qc, gen).to(dev)
    sr_gen = torch.Generator(device=dev).manual_seed(t["seed"] + 1)
    opt = Lion(param_groups(model, t["wd"], t["norm_wd"], t["conv_decay"], t["lr_emb"], t["lr_vec"]),
               lr=t["lr"], wd=t["wd"], qc=qc, generator=sr_gen,
               betas=tuple(t["betas"]))
    ix = data.Index(idx_path)
    ctx = context_len(cfg, model)
    is_mlp = cfg_is_mlp(model)
    fwd = torch.compile(model) if t["compile"] else model
    rec = getattr(model, "rec", "none") != "none"      # plan 002 recurrent branch
    fwd_state = (torch.compile(model.forward_state) if t["compile"] else model.forward_state) if rec else None
    rstate = None                                      # per-layer [S, d_rec] (or [S, H, P, N]), one row per stream
    kind = cfg["data"]["batcher"]
    if kind not in ("random", "sequential"):
        raise ValueError(f"data.batcher {kind!r}")
    Batcher = data.SequentialBatcher if kind == "sequential" else data.RandomBatcher
    bkw = {"streams": cfg["data"]["streams"], "weights": cfg["data"]["weights"] or None} \
        if kind == "sequential" else {}
    if kind != "sequential" and cfg["data"]["weights"]:
        raise ValueError("data.weights needs data.batcher = \"sequential\"")
    batcher = Batcher(ix, data.TRAIN, batch=t["batch"], seq=ctx, seed=t["seed"], **bkw)
    ev = EvalSet(ev_path, "val")
    ev_quick = ev.subset(t["val_frac"]) if t["val_every"] else None
    start = 0
    if resume:
        ck = torch.load(run_dir / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        batcher.load_state_dict(ck["batcher"])
        sr_gen.set_state(ck["sr_gen"])
        start = ck["step"]
        rstate = ck.get("rec_state")

    log = open(run_dir / "metrics.jsonl", "a")
    t0, last = time.time(), {}
    end = min(t["steps"], t["stop"]) if t["stop"] else t["steps"]
    peaks = _Peaks(dev)
    for step in range(start, end):
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        b = batcher.next()
        x = torch.from_numpy(b.x).to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if is_mlp:
            y = torch.from_numpy(b.y[:, -1].astype("int64")).to(dev)
            loss = ce = F.cross_entropy(fwd(x), y)
            loss.backward()
        else:
            y = torch.from_numpy(b.y).to(dev, non_blocking=True)
            mb = t["micro_batch"] or len(x)            # gradient accumulation
            if len(x) % mb:
                raise ValueError("train.batch must be a multiple of train.micro_batch")
            loss = ce = 0.0
            carry = rec and kind == "sequential" and t["rec_carry"]
            reset = torch.from_numpy(b.doc_start).to(dev) if rec and b.doc_start is not None else None
            new_state = []
            rows = torch.from_numpy(b.rows).to(dev) if carry else None
            keep = None                                # N2b: random cold starts inside documents
            if carry and t["rec_drop"] > 0:
                u = np.random.Generator(np.random.PCG64([t["seed"], 1 << 21, step])).random(len(x))
                keep = torch.from_numpy(u >= t["rec_drop"]).to(dev)
            for i in range(0, len(x), mb):
                if rec:
                    s_in = [s[rows[i:i + mb]] if keep is None else s[rows[i:i + mb]] * keep[i:i + mb].view(-1, *[1] * (s.dim() - 1))
                            for s in rstate] if carry and rstate is not None else None
                    lg, s_out = fwd_state(x[i:i + mb], s_in, None if reset is None else reset[i:i + mb])
                    new_state.append([s.detach() for s in s_out])
                else:
                    lg = fwd(x[i:i + mb])
                li, ci = model.loss(lg, y[i:i + mb])
                (li * (mb / len(x))).backward()        # mean over the full batch
                loss = loss + li.detach() * (mb / len(x))
                ce = ce + ci.detach() * (mb / len(x))
            if rec and carry:                          # gradient cut at the window edge (k = 1)
                if rstate is None:                     # never-visited streams: zero state
                    rstate = [new_state[0][j].new_zeros(batcher.S, *new_state[0][j].shape[1:])
                              for j in range(len(new_state[0]))]
                for j, ss in enumerate(zip(*new_state)):
                    rstate[j][rows] = torch.cat(ss)
        opt.step()
        if step % t["log_every"] == 0 or step == end - 1:
            el = time.time() - t0
            seen = (step + 1 - start) * t["batch"] * (1 if is_mlp else ctx)
            last = {"step": step, "lr": lr, "loss": loss.item(), "ce_bits": ce.item() / math.log(2),
                    "time": round(el, 1), "kB_per_s": round(seen / max(el, 1e-9) / 1e3, 1)}
            log.write(json.dumps(last) + "\n")
            log.flush()
        if t["telemetry_every"] and not is_mlp and (step % t["telemetry_every"] == 0 or step == end - 1):
            from intgrad.models.transformer import Tap
            tap = Tap()
            with torch.no_grad():
                model(x[: max(1, len(x) // 4)], tap=tap)
            log.write(json.dumps({"step": step, "telemetry": tap.stats}) + "\n")
            log.flush()
        if t["val_every"] and (step + 1) % t["val_every"] == 0 and step + 1 < t["steps"]:
            r = peaks.eval(lambda: evaluate(model, ev_quick, dev))
            log.write(json.dumps({"step": step + 1, "val_quick": {"bpb": r["bpb"], "se": r["se"]}}) + "\n")
            log.flush()
        if t["eval_every"] and (step + 1) % t["eval_every"] == 0 and step + 1 < t["steps"]:
            r = peaks.eval(lambda: evaluate(model, ev, dev))
            log.write(json.dumps({"step": step + 1, "eval": r}) + "\n")
            log.flush()
            print(f"step {step + 1}: val {r['bpb']:.4f} ± {r['se']:.4f}", flush=True)
            if t["eval_samples"] and not is_mlp:
                write_samples(model, run_dir / "samples" / f"{step + 1:07d}.bin", t["eval_samples"],
                              t["eval_sample_len"], t["seed"], dev)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "batcher": batcher.state_dict(), "sr_gen": sr_gen.get_state(), "rec_state": rstate,
                "step": end}, run_dir / "ckpt.pt")
    final = peaks.eval(lambda: evaluate(model, ev, dev)) if t.get("final_eval", True) \
        else {"bpb": float("nan"), "se": 0.0}
    if dev.type == "cuda":                     # training and eval peaks kept apart (eval batch differs)
        final["peak_gib"] = peaks.gib(peaks.train)
        final["eval_peak_gib"] = peaks.gib(peaks.ev)
    log.write(json.dumps({"step": end, "eval": final, "final": end == t["steps"]}) + "\n")
    if t["samples"] and not is_mlp:
        blobs = write_samples(model, run_dir / "samples.bin", t["samples"], t["sample_len"],
                              t["seed"], dev)
        final["utf8_valid"] = utf8_valid_fraction(blobs)
        log.write(json.dumps({"step": end, "samples": t["samples"],
                              "utf8_valid": final["utf8_valid"]}) + "\n")
    log.close()
    print(f"{run_dir.name}: final val {final['bpb']:.4f} ± {final['se']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--set", action="append", default=[], help="key.path=toml_value")
    ap.add_argument("--resume")
    a = ap.parse_args()
    if a.resume:
        cfg = merge(DEFAULTS, tomllib.loads((Path(a.resume) / "config.toml").read_text()))
    else:
        cfg = merge(DEFAULTS, tomllib.loads(Path(a.config).read_text()) if a.config else {})
    for s in a.set:
        set_override(cfg, s)
    run(cfg, Path(a.resume) if a.resume else None)


if __name__ == "__main__":
    main()
