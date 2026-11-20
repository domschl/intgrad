"""ig-dash: a local, live dashboard over runs/ (plan/002 tooling).

    uv run --project py ig-dash [--port 8765] [--runs runs] [--host 127.0.0.1]

Serves one page plus a small JSON API, read straight from each run
directory (config.toml, meta.json, metrics.jsonl, position.json,
samples.bin, samples/<step>.bin). Nothing is copied or uploaded: runs/ and
generated samples stay on this machine (they can echo the private corpus).
Binds to localhost by default.

    /api/runs                       one summary per run (status, progress, metrics, config)
    /api/run?name=R                 curves: train, quick val, eval-v1 (+ per source), lr
    /api/samples?name=R&step=S      samples at eval step S, or step=final
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("dash.html")
STALE_S = 600            # no metrics line for this long: the run is no longer alive


def _flat(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


class Runs:
    """Parsed run directories, re-read only when a file changed."""

    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[str, tuple[tuple, dict]] = {}

    def dirs(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir() if (p / "config.toml").exists())

    def get(self, name: str) -> Path | None:
        if "/" in name or name.startswith("."):
            return None
        p = self.root / name
        return p if (p / "config.toml").exists() else None

    def load(self, d: Path) -> dict:
        files = [d / f for f in ("config.toml", "meta.json", "metrics.jsonl", "position.json")]
        key = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
        hit = self.cache.get(d.name)
        if hit and hit[0] == key:
            return hit[1]
        cfg = tomllib.loads(files[0].read_text())
        meta = json.loads(files[1].read_text()) if files[1].exists() else {}
        recs = []
        if files[2].exists():
            for line in files[2].read_text().splitlines():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:          # a line being written right now
                    pass
        pos = json.loads(files[3].read_text()) if files[3].exists() else None
        r = {"cfg": cfg, "meta": meta, "recs": recs, "pos": pos,
             "mtime": files[2].stat().st_mtime if files[2].exists() else meta.get("created", 0)}
        self.cache[d.name] = (key, r)
        return r

    def summary(self, d: Path) -> dict:
        r = self.load(d)
        t = r["cfg"].get("train", {})
        steps = t.get("steps", 0)
        train = [x for x in r["recs"] if "ce_bits" in x]
        fin = next((x for x in reversed(r["recs"]) if "final" in x), None)
        evals = [x for x in r["recs"] if "eval" in x and x["eval"].get("bpb") == x["eval"].get("bpb")]
        quick = [x for x in r["recs"] if "val_quick" in x]
        step = max([x["step"] for x in r["recs"]] or [0])
        age = time.time() - r["mtime"]
        if fin and fin["final"]:
            status = "done"
        elif fin:
            status = "paused"
        elif age < STALE_S and r["recs"]:
            status = "running"
        else:
            status = "stopped"
        rate = train[-1]["step"] / train[-1]["time"] if train and train[-1].get("time", 0) > 0 else 0
        last_eval = evals[-1]["eval"] if evals else None
        pos = r["pos"]
        return {
            "name": d.name, "status": status, "step": step, "steps": steps,
            "eta_s": round((steps - step) / rate) if status == "running" and rate else None,
            "age_s": round(age), "created": r["meta"].get("created", 0), "git": r["meta"].get("git"),
            "train": round(train[-1]["ce_bits"], 4) if train else None,
            "kB_per_s": train[-1].get("kB_per_s") if train else None,
            "quick": quick[-1]["val_quick"]["bpb"] if quick else None,
            "eval": last_eval["bpb"] if last_eval else None,
            "eval_step": evals[-1]["step"] if evals else None,
            "final": bool(fin and fin["final"]),
            "sources": {k: v["bpb"] for k, v in (last_eval or {}).items() if isinstance(v, dict)},
            "peak_gib": (fin or {}).get("eval", {}).get("peak_gib"),
            "carried": pos.get("bpb_carried", {}).get("bpb") if pos else None,
            "closed": pos.get("cross_window", {}).get("closed") if pos else None,
            "config": _flat(r["cfg"]),
        }

    def curves(self, d: Path) -> dict:
        r = self.load(d)
        rec = r["recs"]
        smp = sorted(int(p.stem) for p in (d / "samples").glob("*.bin")) if (d / "samples").is_dir() else []
        return {
            "train": [[x["step"], x["ce_bits"], x.get("lr")] for x in rec if "ce_bits" in x],
            "quick": [[x["step"], x["val_quick"]["bpb"]] for x in rec if "val_quick" in x],
            "eval": [[x["step"], x["eval"]["bpb"],
                      {k: v["bpb"] for k, v in x["eval"].items() if isinstance(v, dict)}]
                     for x in rec if "eval" in x and x["eval"].get("bpb") == x["eval"].get("bpb")],
            "samples": smp + (["final"] if (d / "samples.bin").exists() else []),
            "position": r["pos"],
        }

    @staticmethod
    def samples(d: Path, step: str) -> list[str]:
        p = d / "samples.bin" if step == "final" else d / "samples" / f"{int(step):07d}.bin"
        if not p.exists():
            return []
        return [b.decode("utf-8", errors="replace") for b in p.read_bytes().split(b"\x03\n") if b]


def handler(runs: Runs):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):              # quiet
            pass

        def send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj):
            self.send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/runs":
                    out = []
                    for d in runs.dirs():
                        try:
                            out.append(runs.summary(d))
                        except (OSError, ValueError, KeyError, TypeError) as e:   # an odd old run
                            out.append({"name": d.name, "status": "unreadable", "error": str(e),
                                        "step": 0, "steps": 0, "created": 0, "sources": {}, "config": {}})
                    return self.js(out)
                d = runs.get(q.get("name", ""))
                if d is None:
                    return self.send(404, b"no such run", "text/plain")
                if u.path == "/api/run":
                    return self.js(runs.curves(d))
                if u.path == "/api/samples":
                    return self.js(runs.samples(d, q.get("step", "final")))
                self.send(404, b"not found", "text/plain")
            except (OSError, ValueError, KeyError) as e:
                self.send(500, str(e).encode(), "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), handler(Runs(a.runs.resolve())))
    print(f"ig-dash: http://{a.host}:{a.port}/  ({a.runs})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
