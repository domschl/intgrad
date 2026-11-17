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
