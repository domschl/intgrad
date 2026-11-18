"""Hourglass byte LM (plan/001 M6): causality across and inside patches,
padding, eval protocol, quantized variant, training entry point."""
import pytest
import torch

from intgrad.models.hourglass import Hourglass, build

QAT_ALL = dict(acts=True, resid="q8", softmax="maxrel", probs=True)


def tiny(k=4, **kw):
    torch.manual_seed(0)
    return Hourglass(d=32, L=2, H=2, T=32, k=k, d_b=16, qkv_conv=3, **kw)


@pytest.mark.parametrize("mix", ["conv", "attn"])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("t", [0, 3, 4, 7, 13, 30])
def test_causal_at_every_offset(k, t, mix):
    """Changing bytes after position t never changes logits at <= t, for
    t at patch starts, inside patches and at patch ends."""
    m = tiny(k, byte_mix=mix, byte_window=8, byte_dh=8)
    x = torch.randint(0, 256, (2, 32))
    x2 = x.clone()
    x2[:, t + 1:] = (x2[:, t + 1:] + 1 + torch.randint(0, 200, x2[:, t + 1:].shape)) % 256
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.equal(a[:, :t + 1], b[:, :t + 1])
    if t + 1 < 32:
        assert not torch.equal(a[:, t + 1:], b[:, t + 1:])


def test_global_path_reaches_the_next_patch():
    """Byte 0 influences position k-1 onward through the global path (with
    the byte convs too short to carry it there on their own)."""
    torch.manual_seed(0)
    m = Hourglass(d=32, L=2, H=2, T=32, k=4, d_b=16, kernel=2, enc=1, dec=1)
    x = torch.randint(0, 256, (1, 32))
    x2 = x.clone()
    x2[0, 0] = (x2[0, 0] + 1) % 256
    with torch.no_grad():
        d = (m(x) - m(x2)).abs().amax(-1)[0]
    assert d[3] > 0 and d[20] > 0           # completes patch 0; far later


def test_padding_matches_the_unpadded_prefix():
    m = tiny(4)
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        full = m(x)
        part = m(x[:, :13])                 # 13 is not a multiple of k
    assert part.shape == (2, 13, 256)
    assert torch.allclose(part, full[:, :13], atol=1e-5)


def test_chunk_bits_and_sample_run():
    m = tiny(4)
    ch = torch.randint(0, 256, (3, 70))
    bits = m.chunk_bits(ch)
    assert bits.shape == (3, 69) and torch.isfinite(bits).all()
    g = torch.Generator().manual_seed(0)
    out = m.sample(torch.full((2, 1), 3, dtype=torch.uint8), 9, generator=g)
    assert out.shape == (2, 9)


def test_quantized_is_causal_after_warmup_and_trains():
    m = tiny(4, qat=QAT_ALL)
    x = torch.randint(0, 256, (2, 32))
    m(x)                                    # fill the scale histories
    m.eval()
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 256
    assert torch.equal(m(x)[:, :10], m(x2)[:, :10])
    m.train()
    loss, _ = m.loss(m(x), x)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_mid_hook_is_patch_level():
    m = tiny(4)
    seen = {}
    m(torch.randint(0, 256, (2, 32)), mid_hook=lambda h: seen.setdefault("h", h))
    assert seen["h"].shape == (2, 8, 32)


def test_presets_build():
    for p in ("S", "M", "L"):
        m = build(p, T=1024, k=4)
        assert m.attn_elems() == m.H * 256 * 256


def test_train_entry_point(tmp_path):
    from intgrad import train
    cfg = train.merge(train.DEFAULTS, {
        "name": "hg", "model": {"kind": "transformer", "arch": "hourglass", "d": 32, "L": 2,
                                "H": 2, "T": 32, "k": 4, "d_b": 16},
        "quant": {"enabled": False},
        "train": {"steps": 3, "batch": 2, "log_every": 1, "final_eval": False, "device": "cpu"}})
    import tomllib
    from pathlib import Path
    if not (train.ROOT / cfg["data"]["index"]).exists():
        pytest.skip("private corpus not present")
    train.run(cfg, out_root=tmp_path)
    assert (tmp_path / "hg-s42" / "ckpt.pt").exists()


def test_sliding_window_limits_the_byte_path():
    """With byte attention of window 4 and no global path reaching it, a
    byte influences only the next 3 + 3 positions (enc + dec windows)."""
    from intgrad.models.transformer import Block
    torch.manual_seed(0)
    blk = Block(16, 2, 1, window=4)
    from intgrad.models.transformer import rope_cache
    x = torch.randn(1, 32, 16)
    x2 = x.clone(); x2[0, 5] += 1.0
    with torch.no_grad():
        d = (blk(x, rope_cache(32, 8, "cpu")) - blk(x2, rope_cache(32, 8, "cpu"))).abs().amax(-1)[0]
    assert d[:5].max() == 0 and d[5:9].min() > 0 and d[9:].max() == 0
