"""Kneser-Ney baseline sanity: proper distributions, beats unigram."""
import numpy as np

from intgrad.eval_set import KneserNey, unigram_bits


_vr = np.random.default_rng(0)
WORDS = [bytes(_vr.integers(97, 123, size=_vr.integers(2, 8), dtype=np.uint8))
         for _ in range(300)]


def corpus(seed, n):
    """Zipfian word sequence over a shared vocabulary."""
    rng = np.random.default_rng(seed)
    words = WORDS
    return np.frombuffer(b" ".join(words[i] for i in rng.zipf(1.3, n) % 300), dtype=np.uint8)


def test_kn_is_normalized_over_all_next_bytes():
    kn = KneserNey(corpus(0, 50_000), order=5)
    rng = np.random.default_rng(1)
    for ctx in (b"the q", b"zzzzz", corpus(2, 50)[:4].tobytes() + b"a"):
        ctx = ctx[:4]
        chunks = np.array([list(ctx) + [w] for w in range(256)], dtype=np.uint8)
        p = 2.0 ** -kn.bits(chunks)[:, -1]
        assert abs(p.sum() - 1.0) < 1e-6, (ctx, p.sum())
    # also at chunk start (lower orders only)
    chunks = np.array([[w0, w] for w0 in (97, 98) for w in range(256)], dtype=np.uint8)
    p = 2.0 ** -kn.bits(chunks)[:, -1]
    assert abs(p[:256].sum() - 1) < 1e-6 and abs(p[256:].sum() - 1) < 1e-6


def test_kn_beats_unigram_on_heldout():
    train, test = corpus(0, 200_000), corpus(9, 20_000)
    chunks = test[: (len(test) // 256) * 256].reshape(-1, 256)
    kn = KneserNey(train, order=5).bits(chunks).mean()
    uni = unigram_bits(train, chunks).mean()
    assert kn < uni - 1.0


def test_stratified_freeze_takes_equal_chunks_per_source(tmp_path):
    import json
    import subprocess
    import sys

    from intgrad import data
    from intgrad.corpus import make_split
    from intgrad.eval_set import freeze_stratified
    rng = np.random.default_rng(2)
    buf, docs = bytearray(), []
    for i in range(600):                     # code docs uppercase, text lowercase, code rarer
        kind, lo = ("code", 65) if i % 10 == 0 else ("textcorpus", 97)
        doc = bytes(rng.integers(lo, lo + 26, size=int(rng.integers(300, 3000)), dtype=np.uint8))
        if buf:
            buf += b"\n \n"
        docs.append({"off": len(buf), "len": len(doc), "src": f"d{i}", "kind": kind, "lang": "en"})
        buf += doc
    c = tmp_path / "c"
    c.mkdir()
    (c / "shard-0000.bin").write_bytes(bytes(buf))
    (c / "manifest.jsonl").write_text(json.dumps({"shard": "shard-0000.bin", "bytes": len(buf),
                                                  "docs": docs}) + "\n")
    subprocess.run([sys.executable, "-m", "intgrad.corpus.make_split", "--corpus", str(c),
                    "--out", str(tmp_path / "s"), "--val-permille", "300", "--test-permille", "0"],
                   check=True, capture_output=True)
    ix = data.Index(tmp_path / "s" / "idx.bin")
    chunks, src = freeze_stratified(ix, data.VAL, 16, 256, seed=7)
    names = [ix.sources[s] for s in src]
    assert names.count("code") == 16 and names.count("text") == 16
    for ch, s in zip(chunks, src):           # each chunk is from its own source only
        letters = ch[(ch >= 65) & (ch < 123)]
        assert ((letters < 91) == (ix.sources[s] == "code")).all()
    again, _ = freeze_stratified(ix, data.VAL, 16, 256, seed=7)
    assert np.array_equal(chunks, again)
    only, s2 = freeze_stratified(ix, data.VAL, 4, 256, seed=7, skip={"text"})
    assert {ix.sources[s] for s in s2} == {"code"}
