"""make_split — hash-pinned train/val/test buckets + binary index (v2).

Bucket rule (deterministic, content-based, no seed to lose):
    bucket = blake2b-64(key) % 1000, key = the doc bytes, EXCEPT for the
    English wiki sources (wiki-defs = 2017 lead paragraphs, wiki-en = 2015
    full articles), whose key is the normalized ARTICLE TITLE: a definition
    and the article it was taken from must share a bucket (B10: 30% of val
    articles had their lead in a train definition under per-doc hashing;
    keying on the lead text instead still leaked ~14% because of edits).
    Titles of articles come from the manifest src; titles of definitions
    from the raw dump (--wiki-defs-raw, line number in src). Without a
    title the normalized lead (LEAD_LETTERS letters) is the fallback key.
    < val_permille                  -> val (1)
    < val_permille + test_permille  -> test (2)
    else                            -> train (0)

idx.bin v2 (little-endian, no padding), consumed by src/dataset.c and
intgrad.data:
    'IGIX' u32 version=2
    u32 nfiles   { u32 len, path bytes }*    paths RELATIVE to idx.bin's dir
    u32 nsources { u8 len, name bytes }*     source table (id = position)
    per doc, in corpus/manifest order (consecutive docs of a shard stay
    adjacent, which the sequential batcher relies on):
        u32 file_id, u32 off, u32 len, u8 bucket, u8 source     (14 bytes)

split.json records counts and bytes per bucket and per source.

Usage:
  uv run ig-make-split --corpus ../data/corpus-v2 --corpus ../data/corpus-v1-wiki \\
      --out ../data/split-v1
  uv run ig-make-split ... --out ../data/split-v1 --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

MAGIC = b"IGIX"
VERSION = 2
DOC = struct.Struct("<IIIBB")
SOURCES = ["other", "books", "wiki-defs", "wiki-en", "wiki-de", "text"]


LEAD_LETTERS = 48
_NONALPHA = re.compile(rb"[^a-z]+")
GROUP_BY_LEAD = {"wiki-defs", "wiki-en"}


_NONALNUM = re.compile(r"[^0-9a-z]+")


def norm_title(t: str) -> bytes:
    return _NONALNUM.sub("", t.lower()).encode()


def bucket_key(b: bytes, source: str, title: str | None = None) -> bytes:
    if source in GROUP_BY_LEAD:
        if title and norm_title(title):
            return b"title:" + norm_title(title)
        lead = _NONALPHA.sub(b"", b[:600].lower())[:LEAD_LETTERS]
        if len(lead) >= LEAD_LETTERS // 2:
            return b"lead:" + lead
    return b


def load_def_titles(path: str | None) -> list[str] | None:
    """Titles of the "Title ||| definition" dump, indexed by line number."""
    if not path:
        return None
    titles = []
    with open(path, "rb") as f:
        for line in f:
            t, sep, _ = line.partition(b" ||| ")
            titles.append(t.strip().decode("utf-8", "replace") if sep else "")
    return titles


def doc_title(d: dict, def_titles: list[str] | None) -> str | None:
    src = d.get("src", "")
    if "|||" in src:                       # wikixml: "LANG|||file:Title"
        return src.split("|||", 1)[1].split(":", 1)[1]
    if def_titles is not None and src.startswith("wiki2017") and "#" in src:
        return def_titles[int(src.rsplit("#", 1)[1])]
    return None


def doc_bucket(b: bytes, val_per: int, test_per: int) -> int:
    h = int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")
    m = h % 1000
    if m < val_per:
        return 1
    if m < val_per + test_per:
        return 2
    return 0


def doc_source(d: dict) -> int:
    """Source id from a manifest record (older manifests have no 'kind')."""
    kind = d.get("kind")
    src = d.get("src", "")
    if kind is None:
        if "|||" in src:
            kind = "wikixml"
        elif "wiki2017" in src:
            kind = "wiki2017"
        elif "gutenberg" in src:
            kind = "gutenberg"
        elif "textcorpus" in src:
            kind = "textcorpus"
    if kind == "gutenberg":
        return SOURCES.index("books")
    if kind == "wiki2017":
        return SOURCES.index("wiki-defs")
    if kind == "wikixml":
        name = "wiki-" + d.get("lang", "")
        return SOURCES.index(name) if name in SOURCES else 0
    if kind == "textcorpus":
        return SOURCES.index("text")
    return 0


def scan(corpora: list[str], out: Path, val_per: int, test_per: int,
         def_titles: list[str] | None = None):
    files: list[str] = []
    fid_of: dict[str, int] = {}
    docs = bytearray()
    n = 0
    counts = [[0, 0] for _ in range(3)]
    per_src = {s: [[0, 0] for _ in range(3)] for s in SOURCES}
    for cdir in map(Path, corpora):
        mpath = cdir / "manifest.jsonl"
        if not mpath.exists():
            sys.exit(f"no manifest in {cdir}")
        with open(mpath, encoding="utf-8") as mf:
            for line in mf:
                rec = json.loads(line)
                spath = (cdir / rec["shard"]).resolve()
                rel = os.path.relpath(spath, out.resolve())
                if rel not in fid_of:
                    fid_of[rel] = len(files)
                    files.append(rel)
                fid = fid_of[rel]
                data = spath.read_bytes()
                for d in rec["docs"]:
                    b = data[d["off"]:d["off"] + d["len"]]
                    sid = doc_source(d)
                    key = bucket_key(b, SOURCES[sid], doc_title(d, def_titles))
                    bk = doc_bucket(key, val_per, test_per)
                    docs += DOC.pack(fid, d["off"], d["len"], bk, sid)
                    n += 1
                    counts[bk][0] += 1
                    counts[bk][1] += d["len"]
                    per_src[SOURCES[sid]][bk][0] += 1
                    per_src[SOURCES[sid]][bk][1] += d["len"]
        print(f"{cdir}: {n} docs so far", file=sys.stderr)
    return files, bytes(docs), n, counts, per_src


def write_index(path: Path, files: list[str], docs: bytes) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(files)))
        for p in files:
            pb = p.encode()
            f.write(struct.pack("<I", len(pb)) + pb)
        f.write(struct.pack("<I", len(SOURCES)))
        for s in SOURCES:
            f.write(struct.pack("<B", len(s)) + s.encode())
        f.write(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-permille", type=int, default=1)
    ap.add_argument("--test-permille", type=int, default=1)
    ap.add_argument("--wiki-defs-raw",
                    help="raw 'Title ||| definition' dump: titles for split grouping")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare with the existing idx.bin")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    def_titles = load_def_titles(args.wiki_defs_raw)
    files, docs, n, counts, per_src = scan(args.corpus, out, args.val_permille,
                                           args.test_permille, def_titles)
    tmp = out / ("idx.check" if args.check else "idx.bin.tmp")
    write_index(tmp, files, docs)
    if args.check:
        same = tmp.read_bytes() == (out / "idx.bin").read_bytes()
        tmp.unlink()
        print("CHECK", "OK" if same else "FAILED", f"({n} docs)")
        sys.exit(0 if same else 1)
    tmp.replace(out / "idx.bin")
    (out / "split.json").write_text(json.dumps({
        "index_version": VERSION,
        "method": f"blake2b-64(key) % 1000 < {args.val_permille} -> val, "
                  f"< {args.val_permille + args.test_permille} -> test; key = doc, "
                  f"or the normalized title (fallback: {LEAD_LETTERS}-letter lead) "
                  f"for {sorted(GROUP_BY_LEAD)}",
        "wiki_defs_titles": bool(def_titles),
        "created": int(time.time()),
        "corpora": [os.path.relpath(Path(c).resolve(), out.resolve()) for c in args.corpus],
        "sources": SOURCES,
        "n_docs_total": n,
        "buckets": {name: {"docs": c[0], "bytes": c[1]}
                    for name, c in zip(("train", "val", "test"), counts)},
        "per_source": {s: {name: {"docs": c[0], "bytes": c[1]}
                           for name, c in zip(("train", "val", "test"), v)}
                       for s, v in per_src.items() if any(x[0] for x in v)},
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print(json.dumps({"docs": n, "train": counts[0], "val": counts[1],
                      "test": counts[2]}))


if __name__ == "__main__":
    main()
