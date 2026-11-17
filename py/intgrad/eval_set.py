"""Frozen evaluation set (plan/001 §4.4, eval-v1) and reference baselines.

freeze     N chunks x L bytes from a held-out bucket at uniformly random
           positions of its virtual stream (byte-uniform, may cross docs
           with EOD), fixed seed. Written to <out>/<bucket>.bin (N*L bytes),
           <bucket>.src (N bytes: source id of the chunk's first doc) and
           meta.json (hashes, sources, parameters).
baselines  bits/byte on the frozen set, per source, with standard errors
           over chunks: unigram and interpolated Kneser-Ney 5-gram trained
           on a train sample, xz/zstd on the whole set, zstd per chunk with
           a dictionary trained on train data.

Every model scores the same targets: positions 1..L-1 of each chunk,
predicted from bytes inside the chunk only.

  uv run ig-eval-freeze freeze --index ../data/split-v1/idx.bin --out ../data/eval-v1
  uv run ig-eval-freeze baselines --index ../data/split-v1/idx.bin --eval ../data/eval-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import time
from pathlib import Path

import numpy as np

from intgrad import data

LN2 = math.log(2.0)


# ---------------------------------------------------------------- freeze

def freeze(index: data.Index, bucket: int, n: int, length: int, seed: int):
    bt = data.RandomBatcher(index, bucket, batch=1, seq=length - 1, seed=seed)
    chunks = np.empty((n, length), dtype=np.uint8)
    src = np.empty(n, dtype=np.uint8)
    ids = bt.stream.ids
    for i, u in enumerate(bt.rng.integers(0, bt.total, size=n)):
        k, off = bt.locate(int(u))
        bt.stream.read(k, off, chunks[i], None)
        src[i] = index.docs["source"][ids[k]]
    return chunks, src


class EvalSet:
    def __init__(self, path: str | Path, bucket: str = "val"):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text())
        L = self.meta["length"]
        raw = (path / f"{bucket}.bin").read_bytes()
        if hashlib.blake2b(raw, digest_size=16).hexdigest() != self.meta[bucket]["blake2b"]:
            raise ValueError(f"{path}/{bucket}.bin does not match its pinned hash")
        self.chunks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, L)
        self.src = np.frombuffer((path / f"{bucket}.src").read_bytes(), dtype=np.uint8)
        self.sources: list[str] = self.meta["sources"]


def report(bits: np.ndarray, src: np.ndarray, sources: list[str]) -> dict:
    """bits: [N, L-1] bits per predicted byte -> mean, SE, per source."""
    per_chunk = bits.mean(axis=1)
    out = {"bpb": round(float(per_chunk.mean()), 4),
           "se": round(float(per_chunk.std(ddof=1) / math.sqrt(len(per_chunk))), 4)}
    for sid in np.unique(src):
        m = per_chunk[src == sid]
        out[sources[sid]] = {"bpb": round(float(m.mean()), 4), "n": int(len(m))}
    return out


# ---------------------------------------------------------------- n-grams

def _keys(d: np.ndarray, k: int) -> np.ndarray:
    """uint64 keys of all k-grams (big-endian packed bytes), k <= 8."""
    n = len(d) - k + 1
    key = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        key = (key << np.uint64(8)) | d[j:j + n].astype(np.uint64)
    return key


class KneserNey:
    """Interpolated Kneser-Ney byte n-gram (one discount per order),
    vectorized: counts are sorted uint64 key arrays + searchsorted."""

    def __init__(self, train: np.ndarray, order: int = 5):
        self.N = order
        self.keys: dict[int, np.ndarray] = {}
        self.cnt: dict[int, np.ndarray] = {}
        self.ctx: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.D: dict[int, float] = {}
        uk, c = np.unique(_keys(train, order), return_counts=True)
        self.keys[order], self.cnt[order] = uk, c
        for k in range(order - 1, 0, -1):        # continuation counts
            suffix = self.keys[k + 1] & np.uint64((1 << (8 * k)) - 1)
            self.keys[k], self.cnt[k] = np.unique(suffix, return_counts=True)
        for k in range(1, order + 1):
            c = self.cnt[k]
            n1, n2 = int((c == 1).sum()), int((c == 2).sum())
            d = n1 / (n1 + 2 * n2) if n1 + n2 else 0.5
            self.D[k] = min(max(d, 0.1), 0.95)   # >0: unseen bytes keep mass
            if k > 1:
                hk, start, n1ctx = np.unique(self.keys[k] >> np.uint64(8),
                                             return_index=True, return_counts=True)
                self.ctx[k] = (hk, np.add.reduceat(c, start), n1ctx)
        self.total1 = int(self.cnt[1].sum())

    @staticmethod
    def _lookup(keys, vals, q):
        i = np.searchsorted(keys, q)
        i = np.minimum(i, len(keys) - 1)
        hit = keys[i] == q
        return np.where(hit, vals[i], 0)

    def bits(self, chunks: np.ndarray) -> np.ndarray:
        """Bits for predicting chunks[:, 1:] from in-chunk context."""
        Nc, L = chunks.shape
        p = None
        for k in range(1, self.N + 1):
            # key of the k-gram ENDING at each target position t (t >= k-1)
            kk = np.zeros((Nc, L), dtype=np.uint64)
            for j in range(k):
                kk[:, k - 1:] = (kk[:, k - 1:] << np.uint64(8)) | \
                    chunks[:, j:L - k + 1 + j].astype(np.uint64)
            avail = np.zeros((Nc, L), dtype=bool)
            avail[:, k - 1:] = True
            c = self._lookup(self.keys[k], self.cnt[k], kk).astype(np.float64)
            D = self.D[k]
            if k == 1:
                pk = (np.maximum(c - D, 0) + D * len(self.keys[1]) / 256.0) / self.total1
            else:
                hk, cctx, n1ctx = self.ctx[k]
                h = kk >> np.uint64(8)
                cc = self._lookup(hk, cctx, h).astype(np.float64)
                nn = self._lookup(hk, n1ctx, h).astype(np.float64)
                ok = avail & (cc > 0)
                safe = np.where(cc > 0, cc, 1.0)
                pk = np.where(ok, (np.maximum(c - D, 0) + D * nn * p) / safe, p)
            p = pk
        return -np.log2(p[:, 1:])


def unigram_bits(train: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    c = np.bincount(train, minlength=256).astype(np.float64) + 1.0
    lp = np.log2(c / c.sum())
    return -lp[chunks[:, 1:]]


def compressor_bpb(chunks: np.ndarray, dict_train: list[bytes]) -> dict:
    import zstandard as zstd
    raw = chunks.tobytes()
    out = {"xz -9e (whole set)": 8 * len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) / len(raw),
           "zstd -19 (whole set)": 8 * len(zstd.ZstdCompressor(level=19).compress(raw)) / len(raw)}
    zd = zstd.train_dictionary(110 << 10, dict_train)
    cz = zstd.ZstdCompressor(level=19, dict_data=zd)
    tot = sum(len(cz.compress(c.tobytes())) for c in chunks)
    out["zstd -19 + 110KB train dict (per chunk)"] = 8 * tot / len(raw)
    return {k: round(v, 4) for k, v in out.items()}


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--index", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--chunks", type=int, default=512)
    f.add_argument("--length", type=int, default=2048)
    f.add_argument("--seed", type=int, default=20261117)
    b = sub.add_parser("baselines")
    b.add_argument("--index", required=True)
    b.add_argument("--eval", required=True)
    b.add_argument("--train-mb", type=int, default=100)
    b.add_argument("--order", type=int, default=5)
    b.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    ix = data.Index(a.index)
    if a.cmd == "freeze":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        idx_hash = hashlib.blake2b(Path(a.index).read_bytes(), digest_size=16).hexdigest()
        meta = {"name": out.name, "length": a.length, "chunks": a.chunks, "seed": a.seed,
                "index_blake2b": idx_hash, "sources": ix.sources, "created": int(time.time())}
        for name, bucket, seed in (("val", data.VAL, a.seed), ("test", data.TEST, a.seed + 1)):
            chunks, src = freeze(ix, bucket, a.chunks, a.length, seed)
            raw = chunks.tobytes()
            (out / f"{name}.bin").write_bytes(raw)
            (out / f"{name}.src").write_bytes(src.tobytes())
            meta[name] = {"blake2b": hashlib.blake2b(raw, digest_size=16).hexdigest(),
                          "per_source": {ix.sources[s]: int((src == s).sum())
                                         for s in np.unique(src)}}
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))
        return

    ev = EvalSet(a.eval, "val")
    T = 1 << 16
    nwin = (a.train_mb << 20) // T
    bt = data.RandomBatcher(ix, data.TRAIN, batch=64, seq=T - 1, seed=a.seed)
    parts = []
    while len(parts) < nwin:
        parts.extend(bt.next().x)
    train = np.concatenate(parts[:nwin]).astype(np.uint8)
    assert len(train) >= (a.train_mb << 20) * 0.99, len(train)
    res = {"eval": ev.meta["name"], "train_sample_mb": a.train_mb}
    t0 = time.time()
    res["unigram"] = report(unigram_bits(train, ev.chunks), ev.src, ev.sources)
    kn = KneserNey(train, a.order)
    res[f"kn{a.order}"] = report(kn.bits(ev.chunks), ev.src, ev.sources)
    dict_samples = [bytes(p[i:i + 2048]) for p in parts[:nwin] for i in range(0, T, 8192)][:20000]
    res["compressors"] = compressor_bpb(ev.chunks, dict_samples)
    res["runtime_s"] = round(time.time() - t0, 1)
    (Path(a.eval) / "baselines.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
