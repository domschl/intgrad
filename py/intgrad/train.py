"""Single training entry point (plan/001 §3): every run is a TOML config.

    uv run ig-train configs/x4-mlp-q.toml [--set train.seed=7] [--resume RUN_DIR]

Run directory runs/<name>-<seed>/:
    config.toml   resolved config      meta.json     git hash, index/eval hashes
    metrics.jsonl one record per log/eval step   ckpt.pt  model+optimizer+batcher

Schedule = the C trainer's: linear warmup to lr, then cosine to 0 at
train.steps (the total horizon). Loss = mean CE over the batch.
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
    "quant": {"enabled": True, "w2": True, "w2_sr": False, "sr": True, "pure_int": False},
    "train": {"steps": 4000, "batch": 32, "lr": 2e-3, "wd": 0.1, "warmup": 200,
              "sched": "cosine", "seed": 42, "log_every": 100, "eval_every": 0,
              "stop": 0,  # >0: pause at this step (horizon stays train.steps)
              "device": "cuda" if torch.cuda.is_available() else "cpu"},
    "data": {"index": "data/split-v1/idx.bin", "eval": "data/eval-v1"},
}


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
    raise ValueError(f"unknown model kind {m['kind']!r}")


def lr_at(step: int, t: dict) -> float:
    lr, warm, steps = t["lr"], t["warmup"], t["steps"]
    if warm > 0 and step < warm:
        return lr * (step + 1) / warm
    if t["sched"] == "cosine" and steps > warm:
        return lr * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / (steps - warm)))
    return lr


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

    torch.manual_seed(t["seed"])
    gen = torch.Generator().manual_seed(t["seed"])
    qc = QuantConfig(**q)
    model = build_model(cfg, qc, gen).to(dev)
    sr_gen = torch.Generator(device=dev).manual_seed(t["seed"] + 1)
    opt = Lion(model.parameters(), lr=t["lr"], wd=t["wd"], qc=qc, generator=sr_gen)
    ix = data.Index(idx_path)
    ctx = cfg["model"]["ctx"]
    batcher = data.RandomBatcher(ix, data.TRAIN, batch=t["batch"], seq=ctx, seed=t["seed"])
    ev = EvalSet(ev_path, "val")
    start = 0
    if resume:
        ck = torch.load(run_dir / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        batcher.load_state_dict(ck["batcher"])
        sr_gen.set_state(ck["sr_gen"])
        start = ck["step"]

    log = open(run_dir / "metrics.jsonl", "a")
    t0, last = time.time(), {}
    end = min(t["steps"], t["stop"]) if t["stop"] else t["steps"]
    for step in range(start, end):
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr
        b = batcher.next()
        x = torch.from_numpy(b.x).to(dev, non_blocking=True)
        y = torch.from_numpy(b.y[:, -1].astype("int64")).to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % t["log_every"] == 0 or step == t["steps"] - 1:
            last = {"step": step, "lr": lr, "loss": loss.item(), "time": round(time.time() - t0, 1)}
            log.write(json.dumps(last) + "\n")
            log.flush()
        if t["eval_every"] and (step + 1) % t["eval_every"] == 0 and step + 1 < t["steps"]:
            r = evaluate(model, ev, dev)
            log.write(json.dumps({"step": step + 1, "eval": r}) + "\n")
            log.flush()
            print(f"step {step + 1}: val {r['bpb']:.4f} ± {r['se']:.4f}", flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "batcher": batcher.state_dict(), "sr_gen": sr_gen.get_state(),
                "step": end}, run_dir / "ckpt.pt")
    final = evaluate(model, ev, dev) if t.get("final_eval", True) else {"bpb": float("nan"), "se": 0.0}
    log.write(json.dumps({"step": end, "eval": final, "final": end == t["steps"]}) + "\n")
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
