# /// script
# requires-python = ">=3.11"
# ///
"""make_split.py — hash-pinned val/test splits + binary index for ig-train.

Bucket rule (deterministic, content-based, no seed to lose):
    bucket = blake2b-64(doc) % 1000
    bucket <  val_permille                    -> val
    bucket <  val_permille + test_permille    -> test
    else                                      -> train
A doc's assignment depends only on its bytes, so the split is stable across
corpus rebuilds and re-runs. --check re-verifies idx.bin against this rule.

Outputs in --out:
  split.json   pinned metadata: counts, bytes, val unigram-entropy baseline
  val.bin / test.bin   held-out docs concatenated (SEP), for external tools
  idx.bin      binary index consumed by ig-train (src/dataset.c):
      'IGIX' u32=ver u32=nfiles  {u32 pathlen, path bytes}*
      per doc: u32 file_id, u32 off, u32 len, u8 bucket  (13 B, no padding)

Usage:
  uv run tools/make_split.py --corpus data/corpus-v0 --corpus data/corpus-v1-wiki \
      --out data/split-v0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import struct
import sys
import time
from pathlib import Path

MAGIC = b"IGIX"
VERSION = 1
SEP = b"\n \n"


def doc_bucket(b: bytes, val_per: int, test_per: int) -> int:
    h = int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")
    m = h % 1000
    if m < val_per:
        return 1  # val
    if m < val_per + test_per:
        return 2  # test
    return 0      # train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-permille", type=int, default=1)
    ap.add_argument("--test-permille", type=int, default=1)
    ap.add_argument("--check", action="store_true",
                    help="re-verify existing idx.bin and exit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    file_id: dict[str, int] = {}
    docs: list[tuple[int, int, int, int]] = []
    parts = {1: [], 2: []}
    counts = [0, 0, 0]
    nbytes = [0, 0, 0]
    unigram = [0] * 256
    t0 = time.time()

    for cdir in args.corpus:
        cdir = Path(cdir)
        mpath = cdir / "manifest.jsonl"
        if not mpath.exists():
            sys.exit(f"no manifest in {cdir}")
        n_before = len(docs)
        with open(mpath, encoding="utf-8") as mf:
            for line in mf:
                rec = json.loads(line)
                spath = (cdir / rec["shard"]).resolve()
                key = str(spath)
                if key not in file_id:
                    file_id[key] = len(files)
                    files.append(key)
                fid = file_id[key]
                with open(spath, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    for d in rec["docs"]:
                        b = mm[d["off"]:d["off"] + d["len"]]
                        bk = doc_bucket(b, args.val_permille, args.test_permille)
                        docs.append((fid, d["off"], d["len"], bk))
                        counts[bk] += 1
                        nbytes[bk] += d["len"]
                        if bk:
                            parts[bk].append(b + SEP)
                            if bk == 1:
                                for ch in b:
                                    unigram[ch] += 1
        print(f"{cdir}: +{len(docs)-n_before} docs", file=sys.stderr)

    # unigram-entropy baseline on val (bits per byte floor any model must beat)
    tot = sum(unigram) or 1
    ent = -sum((c / tot) * math.log2(c / tot) for c in unigram if c)

    for bk, name in ((1, "val.bin"), (2, "test.bin")):
        with open(out / name, "wb") as f:
            f.write(b"".join(parts[bk]))

    if args.check:
        data = (out / "idx.bin").read_bytes()
        ok = True
        pos = 12
        nfiles = struct.unpack_from("<I", data, 8)[0]
        for _ in range(nfiles):
            pl = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + pl
        nd = (len(data) - pos) // 13
        for i, (fid, off, ln, bk) in enumerate(docs[:nd]):
            f2, o2, l2, b2 = struct.unpack_from("<IIIB", data, pos + 13 * i)
            if (fid, off, ln, bk) != (f2, o2, l2, b2):
                print(f"MISMATCH at doc {i}", file=sys.stderr)
                ok = False
                break
        print("CHECK", "OK" if ok else "FAILED", f"({nd} docs compared)")
        sys.exit(0 if ok else 1)

    with open(out / "idx.bin", "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(files)))
        for p in files:
            pb = p.encode()
            f.write(struct.pack("<I", len(pb)))
            f.write(pb)
        buf = bytearray()
        for fid, off, ln, bk in docs:
            buf += struct.pack("<IIIB", fid, off, ln, bk)
        f.write(buf)

    (out / "split.json").write_text(json.dumps({
        "method": f"blake2b-64(doc) % 1000 < {args.val_permille} -> val, "
                  f"< {args.val_permille + args.test_permille} -> test",
        "created": int(time.time()),
        "corpora": [str(Path(c).resolve()) for c in args.corpus],
        "n_docs_total": len(docs),
        "n_train": counts[0], "n_val": counts[1], "n_test": counts[2],
        "bytes_train": nbytes[0], "bytes_val": nbytes[1], "bytes_test": nbytes[2],
        "val_unigram_entropy_bpb": round(ent, 4),
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print(json.dumps({"docs": len(docs), "train": counts[0], "val": counts[1],
                      "test": counts[2], "val_unigram_bpb": round(ent, 4)}))


if __name__ == "__main__":
    main()
