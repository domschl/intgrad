"""Byte data for training: index v2 reader and two batchers.

A bucket (train/val/test) is read as a VIRTUAL STREAM: its documents, each
followed by the end-of-document byte EOD (0x03, plan/001 D7). The corpus
contains no C0 control bytes besides \\t and \\n, so EOD is unambiguous.

RandomBatcher      rows are windows at uniformly random stream positions
                   (= length-weighted documents, every byte equally likely);
                   windows may cross document boundaries (with EOD).
SequentialBatcher  B independent streams; row b of batch t continues row b
                   of batch t-1 byte-for-byte. Each stream walks its own
                   share of a seeded per-epoch document permutation. A
                   `doc_start` mask marks inputs that begin a document, so a
                   recurrent model can reset (or keep) state there.

Both are deterministic given the seed, and resumable via state_dict().
Batches are numpy uint8 arrays x, y of shape [B, T] with y = x shifted by
one byte (next-byte targets).
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EOD = 0x03
PAD = 0x00
TRAIN, VAL, TEST = 0, 1, 2

DOC_DTYPE = np.dtype([("file", "<u4"), ("off", "<u4"), ("len", "<u4"),
                      ("bucket", "u1"), ("source", "u1")])
assert DOC_DTYPE.itemsize == 14


class Index:
    """idx.bin v2 (see intgrad.corpus.make_split)."""

    def __init__(self, path: str | Path):
        path = Path(path)
        with open(path, "rb") as f:
            buf = f.read()
        if buf[:4] != b"IGIX":
            raise ValueError(f"{path}: bad magic")
        ver, nfiles = struct.unpack_from("<II", buf, 4)
        if ver != 2:
            raise ValueError(f"{path}: index v{ver}, need v2 (rebuild with ig-make-split)")
        pos = 12
        self.paths: list[Path] = []
        for _ in range(nfiles):
            (pl,) = struct.unpack_from("<I", buf, pos)
            rel = buf[pos + 4:pos + 4 + pl].decode()
            self.paths.append((path.parent / rel).resolve())
            pos += 4 + pl
        (nsrc,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        self.sources: list[str] = []
        for _ in range(nsrc):
            sl = buf[pos]
            self.sources.append(buf[pos + 1:pos + 1 + sl].decode())
            pos += 1 + sl
        if (len(buf) - pos) % DOC_DTYPE.itemsize:
            raise ValueError(f"{path}: truncated doc table")
        self.docs = np.frombuffer(buf, dtype=DOC_DTYPE, offset=pos)
        self._maps: list[np.ndarray | None] = [None] * nfiles
        self._bucket_ids: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.docs)

    def _map(self, fid: int) -> np.ndarray:
        m = self._maps[fid]
        if m is None:
            with open(self.paths[fid], "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            m = self._maps[fid] = np.frombuffer(mm, dtype=np.uint8)
        return m

    def doc(self, i: int) -> np.ndarray:
        """Bytes of document i (zero-copy view)."""
        d = self.docs[i]
        off = int(d["off"])
        return self._map(int(d["file"]))[off:off + int(d["len"])]

    def bucket_ids(self, bucket: int) -> np.ndarray:
        """Doc indices of a bucket, in corpus order."""
        ids = self._bucket_ids.get(bucket)
        if ids is None:
            ids = self._bucket_ids[bucket] = np.flatnonzero(self.docs["bucket"] == bucket)
        return ids


class _Stream:
    """Reader over the virtual stream of a doc-id list (doc + EOD each)."""

    def __init__(self, index: Index, ids: np.ndarray):
        self.ix = index
        self.ids = ids
        self.lens = index.docs["len"][ids].astype(np.int64)
        self.cum = np.cumsum(self.lens + 1)      # stream end of doc k (+ EOD)
        self.total = int(self.cum[-1]) if len(ids) else 0

    def pos(self, k: int, off: int) -> int:
        """Absolute stream position of (doc k, offset off)."""
        return (int(self.cum[k - 1]) if k > 0 else 0) + off

    def read(self, k: int, off: int, out: np.ndarray, starts: np.ndarray | None,
             wrap: bool = True) -> tuple[int, int]:
        """Fill `out` starting at doc position k, byte offset off (off == len
        means the EOD byte). Marks doc-start positions in `starts`. Returns
        the position after the last byte read."""
        n, filled = len(out), 0
        while filled < n:
            if k >= len(self.ids):
                if not wrap:
                    raise IndexError("stream exhausted")
                k = 0
            L = int(self.lens[k])
            if off < L:
                take = min(L - off, n - filled)
                out[filled:filled + take] = self.ix.doc(int(self.ids[k]))[off:off + take]
                if off == 0 and starts is not None:
                    starts[filled] = True
                filled += take
                off += take
            else:
                out[filled] = EOD
                filled += 1
                k, off = k + 1, 0
        return k, off


@dataclass
class Batch:
    x: np.ndarray                    # [B, T] uint8 inputs
    y: np.ndarray                    # [B, T] uint8 next-byte targets
    doc_start: np.ndarray | None     # [B, T] bool: x[b, t] begins a document
    rows: np.ndarray | None = None   # [B] stream id of each row (SequentialBatcher)


class RandomBatcher:
    """Windows at uniform positions of a bucket's virtual stream."""

    def __init__(self, index: Index, bucket: int, batch: int, seq: int, seed: int):
        self.stream = _Stream(index, index.bucket_ids(bucket))
        self.cum = self.stream.cum
        self.total = self.stream.total
        self.B, self.T = batch, seq
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def locate(self, u: int) -> tuple[int, int]:
        k = int(np.searchsorted(self.cum, u, side="right"))
        start = int(self.cum[k] - (self.stream.lens[k] + 1))
        return k, u - start

    def next(self) -> Batch:
        w = np.empty((self.B, self.T + 1), dtype=np.uint8)
        for b, u in enumerate(self.rng.integers(0, self.total, size=self.B)):
            k, off = self.locate(int(u))
            self.stream.read(k, off, w[b], None)
        return Batch(w[:, :-1], w[:, 1:], None)

    def state_dict(self) -> dict:
        return {"kind": "random", "rng": self.rng.bit_generator.state}

    def load_state_dict(self, st: dict) -> None:
        assert st["kind"] == "random"
        self.rng.bit_generator.state = st["rng"]


class SequentialBatcher:
    """B rows drawn from S >= B document-continuous streams over a bucket.

    Epoch e uses the permutation P_e = permutation(seed, e) of the bucket's
    docs; stream s reads P_e[s::S] in order and starts epoch e+1 when done.
    Every stream therefore reads whole documents, so byte coverage within
    an epoch is uniform by construction.

    With S = B every batch holds the next window of all streams (row b =
    stream b). With a pool S > B (a multiple of B) the streams are dealt out
    in rounds of S/B batches: each round permutes the streams and every
    stream advances by exactly one window per round, so consecutive batches
    come from different documents. Batch.rows names each row's stream."""

    def __init__(self, index: Index, bucket: int, batch: int, seq: int, seed: int,
                 streams: int = 0):
        self.ix = index
        self.ids = index.bucket_ids(bucket)
        S = streams or batch
        if S % batch:
            raise ValueError("streams must be a multiple of batch")
        if len(self.ids) < S:
            raise ValueError("fewer documents than streams")
        self.B, self.S, self.T, self.seed = batch, S, seq, seed
        self.step = 0
        self._perm_cache: dict[int, np.ndarray] = {}
        self._streams: dict[tuple[int, int], _Stream] = {}
        # per stream: epoch, doc position k, offset, carried byte + start flag
        self.epoch = [0] * S
        self.k = [0] * S
        self.off = [0] * S
        self.carry: list[int | None] = [None] * S
        self.carry_start = [False] * S

    def _rows(self) -> np.ndarray:
        if self.S == self.B:
            return np.arange(self.B)
        per = self.S // self.B
        r, j = divmod(self.step, per)
        perm = np.random.Generator(np.random.PCG64([self.seed, 1 << 20, r])).permutation(self.S)
        return perm[j * self.B:(j + 1) * self.B]

    def _perm(self, epoch: int) -> np.ndarray:
        p = self._perm_cache.get(epoch)
        if p is None:
            rng = np.random.Generator(np.random.PCG64([self.seed, epoch]))
            p = self.ids[rng.permutation(len(self.ids))]
            self._perm_cache = {epoch: p, **{e: v for e, v in self._perm_cache.items()
                                             if e >= epoch - 1}}
        return p

    def _stream(self, b: int) -> _Stream:
        key = (self.epoch[b], b)
        s = self._streams.get(key)
        if s is None:
            s = _Stream(self.ix, self._perm(self.epoch[b])[b::self.S])
            self._streams = {k: v for k, v in self._streams.items() if k[1] != b}
            self._streams[key] = s
        return s

    def _read_row(self, b: int, out: np.ndarray, starts: np.ndarray) -> None:
        filled = 0
        while filled < len(out):
            s = self._stream(b)
            if self.k[b] >= len(s.ids):          # epoch done for this stream
                self.epoch[b] += 1
                self.k[b], self.off[b] = 0, 0
                continue
            # read at most to the end of this stream's epoch
            remaining = s.total - s.pos(self.k[b], self.off[b])
            n = min(len(out) - filled, remaining)
            self.k[b], self.off[b] = s.read(self.k[b], self.off[b], out[filled:filled + n],
                                            starts[filled:filled + n], wrap=False)
            filled += n

    def next(self) -> Batch:
        rows = self._rows()
        w = np.empty((self.B, self.T + 1), dtype=np.uint8)
        st = np.zeros((self.B, self.T + 1), dtype=bool)
        for i, b in enumerate(rows):
            if self.carry[b] is None:            # first window of the stream: prime one byte
                self._read_row(b, w[i], st[i])
            else:
                w[i, 0] = self.carry[b]
                st[i, 0] = self.carry_start[b]
                self._read_row(b, w[i, 1:], st[i, 1:])
            self.carry[b] = int(w[i, -1])
            self.carry_start[b] = bool(st[i, -1])
        self.step += 1
        return Batch(w[:, :-1], w[:, 1:], st[:, :-1], rows)

    def state_dict(self) -> dict:
        return {"kind": "sequential", "seed": self.seed, "B": self.B, "S": self.S, "T": self.T,
                "step": self.step,
                "epoch": list(self.epoch), "k": list(self.k), "off": list(self.off),
                "carry": list(self.carry), "carry_start": list(self.carry_start)}

    def load_state_dict(self, st: dict) -> None:
        assert st["kind"] == "sequential" and st["seed"] == self.seed and st["B"] == self.B
        assert st.get("S", st["B"]) == self.S
        self.step = st.get("step", 0)
        self.epoch, self.k, self.off = list(st["epoch"]), list(st["k"]), list(st["off"])
        self.carry, self.carry_start = list(st["carry"]), list(st["carry_start"])


def _bench():
    """python -m intgrad.data --index IDX [--batch B] [--seq T] [--steps N]"""
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    ix = Index(a.index)
    for cls in (RandomBatcher, SequentialBatcher):
        bt = cls(ix, TRAIN, a.batch, a.seq, seed=0)
        bt.next()                                   # warm: permutations, maps
        t0 = time.perf_counter()
        for _ in range(a.steps):
            bt.next()
        dt = time.perf_counter() - t0
        mb = a.steps * a.batch * a.seq / 1e6
        print(f"{cls.__name__:18s} B={a.batch} T={a.seq}: {mb / dt:7.1f} MB/s "
              f"({a.steps / dt:.0f} batches/s)")


if __name__ == "__main__":
    _bench()
