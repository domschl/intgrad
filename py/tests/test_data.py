"""Batchers + index v2 on a synthetic corpus (built through make_split)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from intgrad import data
from intgrad.corpus import make_split

SEP = b"\n \n"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Two shards of random lowercase docs; docs whose split bucket is
    val/test are rewritten with a marker byte 'Z' so leakage is visible."""
    root = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    cdir = root / "c"
    cdir.mkdir()
    manifest = []
    for shard in range(2):
        buf = bytearray()
        docs = []
        for _ in range(300):
            n = int(rng.integers(1, 3000))
            doc = bytes(rng.integers(97, 123, size=n, dtype=np.uint8))
            if make_split.doc_bucket(doc, 100, 100) != 0:     # val/test (20%)
                doc = b"Z" + doc[1:]
                while make_split.doc_bucket(doc, 100, 100) == 0:
                    doc = doc + b"z"                          # keep it held out
            if buf:
                buf += SEP
            docs.append({"off": len(buf), "len": len(doc), "src": f"s{shard}",
                         "kind": "textcorpus", "lang": "en"})
            buf += doc
        name = f"shard-{shard:04d}.bin"
        (cdir / name).write_bytes(bytes(buf))
        manifest.append({"shard": name, "bytes": len(buf), "docs": docs})
    (cdir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    out = root / "split"
    subprocess.run([sys.executable, "-m", "intgrad.corpus.make_split", "--corpus", str(cdir),
                    "--out", str(out), "--val-permille", "100", "--test-permille", "100"],
                   check=True, capture_output=True)
    return data.Index(out / "idx.bin")


def stream_bytes(ix, ids):
    return b"".join(bytes(ix.doc(int(i))) + bytes([data.EOD]) for i in ids)


def test_index_roundtrip(corpus):
    ix = corpus
    assert len(ix) == 600
    assert ix.sources[ix.docs["source"][0]] == "text"
    tr, va = ix.bucket_ids(data.TRAIN), ix.bucket_ids(data.VAL)
    assert len(tr) > 400 and len(va) > 20
    assert all(bytes(ix.doc(int(i)))[:1] == b"Z" for i in va)
    assert not any(b"Z" in bytes(ix.doc(int(i))) for i in tr)
    assert np.all(np.diff(tr) > 0)                       # corpus order kept


@pytest.mark.parametrize("kind", ["random", "sequential"])
def test_no_heldout_bytes_in_train_batches(corpus, kind):
    cls = data.RandomBatcher if kind == "random" else data.SequentialBatcher
    bt = cls(corpus, data.TRAIN, batch=8, seq=256, seed=1)
    for _ in range(200):
        b = bt.next()
        assert not (b.x == ord("Z")).any() and not (b.y == ord("Z")).any()


def test_sequential_rows_are_continuous_documents(corpus):
    B, T = 4, 97
    bt = data.SequentialBatcher(corpus, data.TRAIN, batch=B, seq=T, seed=3)
    steps = 120
    xs = [bt.next() for _ in range(steps)]
    perm = bt._perm(0)
    for b in range(B):
        got = np.concatenate([xs[0].x[b, :1]] + [bt_.y[b] for bt_ in xs]).tobytes()
        want = stream_bytes(corpus, perm[b::B])
        n = min(len(got), len(want))
        assert n > 1000 and got[:n] == want[:n]
        # x shifted by one is y, within and across batches
        for t in range(steps - 1):
            assert xs[t + 1].x[b, 0] == xs[t].y[b, -1]


def test_sequential_epoch_rollover_and_coverage(corpus):
    B, T = 3, 512
    bt = data.SequentialBatcher(corpus, data.TRAIN, batch=B, seq=T, seed=5)
    total = sum(int(corpus.docs["len"][i]) + 1 for i in corpus.bucket_ids(data.TRAIN))
    seen = 0
    while min(bt.epoch) == 0:
        bt.next()
        seen += B * T
    assert seen >= total // B          # each stream finished its share
    assert max(bt.epoch) <= 1


def test_doc_start_mask(corpus):
    bt = data.SequentialBatcher(corpus, data.TRAIN, batch=2, seq=200, seed=7)
    prev_last = [None, None]
    for step in range(50):
        b = bt.next()
        for r in range(2):
            x = b.x[r]
            before = np.concatenate([[prev_last[r] if prev_last[r] is not None else data.EOD],
                                     x[:-1]])
            assert np.array_equal(b.doc_start[r], before == data.EOD)
            prev_last[r] = x[-1]


@pytest.mark.parametrize("kind", ["random", "sequential"])
def test_resume_is_exact(corpus, kind):
    cls = data.RandomBatcher if kind == "random" else data.SequentialBatcher
    a = cls(corpus, data.TRAIN, batch=4, seq=300, seed=11)
    for _ in range(37):
        a.next()
    st = json.loads(json.dumps(a.state_dict()))          # must be JSON-able
    ref = [a.next().x.copy() for _ in range(20)]
    c = cls(corpus, data.TRAIN, batch=4, seq=300, seed=11)
    c.load_state_dict(st)
    for r in ref:
        assert np.array_equal(c.next().x, r)


def test_random_positions_are_byte_uniform(corpus):
    bt = data.RandomBatcher(corpus, data.TRAIN, batch=1, seq=8, seed=13)
    n = 400_000
    u = bt.rng.integers(0, bt.total, size=n)
    k = np.searchsorted(bt.cum, u, side="right")
    counts = np.bincount(k, minlength=len(bt.cum))
    expect = (bt.stream.lens + 1) / bt.total * n
    chi2 = float(((counts - expect) ** 2 / expect).sum())
    dof = len(expect) - 1
    assert chi2 < dof + 5 * np.sqrt(2 * dof)
    # windows read from a located position match the virtual stream
    ids = corpus.bucket_ids(data.TRAIN)
    full = stream_bytes(corpus, ids)
    for pos in (0, 5, int(bt.cum[3]) - 1, bt.total - 3):
        k0, off = bt.locate(pos)
        out = np.empty(6, dtype=np.uint8)
        bt.stream.read(k0, off, out, None)
        want = (full + full)[pos:pos + 6]
        assert out.tobytes() == want


def test_wiki_leads_share_a_bucket():
    from intgrad.corpus.make_split import bucket_key, doc_bucket
    art = b"Anarchism is a political philosophy that advocates self-governed societies. " * 5
    defn = b"Anarchism is a political philosophy that advocates self-governed societies."
    ka, kd = bucket_key(art, "wiki-en"), bucket_key(defn, "wiki-defs")
    assert ka == kd
    assert doc_bucket(ka, 300, 300) == doc_bucket(kd, 300, 300)
    assert bucket_key(art, "books") == art           # other sources: content hash
    # titles win over (edited) leads
    edited = b"Anarchism (/x/) is a philosophy and movement that rejects authority."
    assert bucket_key(edited, "wiki-defs", "Anarchism") == bucket_key(art, "wiki-en", "anarchism")
