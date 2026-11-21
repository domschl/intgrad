"""ig-score — score finished runs on a frozen eval set (plan 003: eval-v2).

    uv run --project py ig-score runs/<run> [runs/<run> ...] [--eval data/eval-v2] [--bucket val]

Loads each run's checkpoint (as ig-position does), scores the eval set with
the eval-v1 protocol (in-chunk context only, zero recurrent state) and
writes runs/<run>/eval-<name>-<bucket>.json. Prints one line per run and a
table per source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from intgrad.eval_set import EvalSet
from intgrad.evaluate import evaluate
from intgrad.position import load_run

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval-v2")
    ap.add_argument("--bucket", default="val")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = EvalSet(a.eval, a.bucket)
    res = {}
    for run in a.runs:
        _, model = load_run(run, dev)
        r = evaluate(model, ev, dev)
        (run / f"eval-{ev.meta['name']}-{a.bucket}.json").write_text(json.dumps(r, indent=2) + "\n")
        res[run.name] = r
        print(f"{run.name}: {ev.meta['name']} {a.bucket} {r['bpb']:.4f} ± {r['se']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    srcs = [s for s in ev.sources if any(s in r for r in res.values())]
    print(f"\n{'source':12s} " + " ".join(f"{n[:22]:>22s}" for n in res))
    for s in srcs:
        print(f"{s:12s} " + " ".join(f"{r[s]['bpb']:22.4f}" if s in r else " " * 22 for r in res.values()))


if __name__ == "__main__":
    main()
