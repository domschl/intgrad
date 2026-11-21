"""ig-fetch — download the public plan-003 sources into data/raw/hf (§2.3).

    uv run --project py ig-fetch [--only NAME ...] [--list] [--root data/raw/hf]
                                 [--archive ~/Storage/Library/Corpora/HuggingFace]

Files come from the Hugging Face hub over plain HTTPS (no token, no hub
library): https://huggingface.co/datasets/<repo>/resolve/<rev>/<path>.
Downloads resume (.part files) and are verified against the size the hub
reports. Every finished file gets a line in <root>/SOURCES.jsonl:
repo, path, revision, license, size, sha256, time. With --archive, each
verified file (and SOURCES.jsonl) is also copied to the same relative path
under the archive directory. Gated datasets are not
listed here (they need a user token and accepted terms).

Nothing is uploaded and no user data is sent: the User-Agent is generic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HUB = "https://huggingface.co"
UA = {"User-Agent": "intgrad-fetch/0.1"}

# name: (repo, license, [directory or file paths], max files per directory)
SOURCES = {
    "edu-en": ("HuggingFaceFW/fineweb-edu", "odc-by", ["sample/10BT"], 2),
    "cosmo": ("HuggingFaceTB/cosmopedia", "apache-2.0",
              ["data/khanacademy", "data/openstax", "data/stanford", "data/wikihow"], 0),
    "cosmo-math": ("HuggingFaceTB/cosmopedia", "apache-2.0", ["data/auto_math_text"], 2),
    "cosmo-stories": ("HuggingFaceTB/cosmopedia", "apache-2.0", ["data/stories"], 2),
    "web-de": ("HuggingFaceFW/fineweb-2", "odc-by", ["data/deu_Latn/train"], 1),
    "math": ("HuggingFaceTB/finemath", "odc-by", ["finemath-4plus"], 8),
    "code": ("codeparrot/github-code-clean", "per-file (filtered to permissive)", ["data"], 8),
    "tool-glaive": ("glaiveai/glaive-function-calling-v2", "apache-2.0",
                    ["glaive-function-calling-v2.json"], 0),
    "tool-hermes": ("NousResearch/hermes-function-calling-v1", "apache-2.0", [""], 0),
}


def _get(url: str, headers: dict | None = None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}),
                                  timeout=60)


def revision(repo: str) -> str:
    with _get(f"{HUB}/api/datasets/{repo}") as r:
        return json.load(r)["sha"]


def listing(repo: str, rev: str, path: str, limit: int) -> list[tuple[str, int]]:
    """(path, size) of the data files under path (or path itself), sorted, first `limit`."""
    def tree(p: str) -> list[dict]:
        with _get(f"{HUB}/api/datasets/{repo}/tree/{rev}" + (f"/{p}" if p else "")) as r:
            return json.load(r)
    try:
        items = tree(path)
    except urllib.error.HTTPError:                         # path is a file: list its directory
        items = [x for x in tree(path.rpartition("/")[0]) if x["path"] == path]
    files = sorted((x["path"], x.get("size", 0)) for x in items if x.get("type") == "file"
                   and x["path"].rsplit(".", 1)[-1] in ("parquet", "json", "jsonl"))
    return files[:limit] if limit else files


def download(repo: str, rev: str, path: str, size: int, dest: Path) -> str:
    """Resumable download; returns the sha256 of the finished file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have < size:
        hdr = {"Range": f"bytes={have}-"} if have else {}
        with _get(f"{HUB}/datasets/{repo}/resolve/{rev}/{path}", hdr) as r, open(part, "ab") as f:
            if have and r.status != 206:                   # server ignored the range
                f.truncate(0)
            t0, n = time.time(), 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                n += len(chunk)
                if n % (256 << 20) < (1 << 20):
                    print(f"  {path}: {(have + n) / 1e6:.0f}/{size / 1e6:.0f} MB "
                          f"({n / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)", flush=True)
    if part.stat().st_size != size:
        raise IOError(f"{path}: size {part.stat().st_size} != {size}")
    h = hashlib.sha256()
    with open(part, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    part.rename(dest)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "hf")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES))
    ap.add_argument("--list", action="store_true", help="show the files and sizes, download nothing")
    ap.add_argument("--archive", type=Path, help="also copy every verified file here")
    a = ap.parse_args()
    log = a.root / "SOURCES.jsonl"
    done = set()
    if log.exists():
        done = {(r["repo"], r["path"]) for r in map(json.loads, log.read_text().splitlines())}
    total = 0
    for name in a.only or SOURCES:
        repo, lic, paths, limit = SOURCES[name]
        rev = revision(repo)
        for p in paths:
            for path, size in listing(repo, rev, p, limit):
                total += size
                dest = a.root / repo / path
                state = "have" if (repo, path) in done and dest.exists() else "get"
                print(f"{name:14s} {state} {repo}/{path} {size / 1e6:.0f} MB", flush=True)
                if a.list:
                    continue
                if state == "get":
                    sha = download(repo, rev, path, size, dest)
                    with open(log, "a") as f:
                        f.write(json.dumps({"name": name, "repo": repo, "path": path, "revision": rev,
                                            "license": lic, "bytes": size, "sha256": sha,
                                            "time": int(time.time())}) + "\n")
                if a.archive:
                    arc = a.archive / repo / path
                    if not (arc.exists() and arc.stat().st_size == size):
                        arc.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(dest, arc.with_suffix(arc.suffix + ".part"))
                        arc.with_suffix(arc.suffix + ".part").rename(arc)
                    shutil.copy2(log, a.archive / "SOURCES.jsonl")
    print(f"total {total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
