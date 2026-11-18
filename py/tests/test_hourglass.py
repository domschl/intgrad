"""Hourglass byte LM (plan/001 M6): causality across and inside patches,
padding, eval protocol, quantized variant, training entry point."""
import pytest
import torch

from intgrad.models.hourglass import Hourglass, build

QAT_ALL = dict(acts=True, resid="q8", softmax="maxrel", probs=True)


def tiny(k=4, **kw):
    torch.manual_seed(0)
    return Hourglass(d=32, L=2, H=2, T=32, k=k, d_b=16, qkv_conv=3, **kw)


@pytest.mark.parametrize("mix", ["conv", "attn", "patch"])
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


def test_patch_local_byte_path_forces_the_core():
    """byte_mix "patch": with the global path zeroed, a byte influences
    nothing outside its own patch (all cross-patch information goes
    through the core)."""
    m = tiny(4, byte_mix="patch", byte_dh=8)
    m.up.weight.data.zero_()
    x = torch.randint(0, 256, (1, 32))
    x2 = x.clone(); x2[0, 9] = (x2[0, 9] + 1) % 256       # patch 2 = bytes 8..11
    with torch.no_grad():
        d = (m(x) - m(x2)).abs().amax(-1)[0]
    assert d[:9].max() == 0 and d[9:12].min() > 0 and d[12:].max() == 0


# ------------------------------------------------------------- UTF-8 patching

from intgrad.models.hourglass import patch_ends, utf8_complete

CHARS = ["a", "b", " ", "é", "ß", "€", "中", "ה", "😀"]       # 1, 2, 3, 4 bytes


def utf8_text(n_bytes: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    out = b""
    while len(out) < n_bytes:
        out += CHARS[torch.randint(0, len(CHARS), (1,), generator=g).item()].encode()
    return torch.tensor(list(out[:n_bytes]))


def test_utf8_complete_is_causal_and_exact():
    s = "aé€😀".encode()                                   # ends at 0, 2, 5, 9
    assert utf8_complete(torch.tensor([list(s)]))[0].nonzero().flatten().tolist() == [0, 2, 5, 9]
    stray = torch.tensor([[0xA9, 0x41, 0xC3, 0x41, 0xC3, 0xA9]])   # stray cont, broken lead
    assert utf8_complete(stray)[0].tolist() == [True, True, False, True, False, True]
    x = utf8_text(64, 0)[None]
    full = utf8_complete(x)
    for t in range(64):                                       # no look-ahead
        assert torch.equal(utf8_complete(x[:, :t + 1])[0], full[0, :t + 1])


@pytest.mark.parametrize("mode", ["utf8", "chars"])
def test_patches_never_split_a_character(mode):
    x = utf8_text(256, 1)[None]
    slot = patch_ends(x, 4, mode)[0]
    comp = utf8_complete(x)[0]
    ends = (slot >= 0).nonzero().flatten()
    assert comp[ends].all()                                   # every patch ends a character
    s = slot[slot >= 0]
    assert torch.equal(s, s.sort().values) and len(s.unique()) == len(s)
    assert s.max() < 256 // 4


def test_utf8_patching_on_ascii_is_fixed_patching():
    x = torch.tensor([list(b"plain ascii text, exactly fixed!")])
    assert torch.equal(patch_ends(x, 4, "utf8"), patch_ends(x, 4, "fixed"))


@pytest.mark.parametrize("mode", ["utf8", "chars"])
@pytest.mark.parametrize("seed", [2, 7])
def test_utf8_patching_is_causal(mode, seed):
    """Every split point t: a different future never changes logits <= t."""
    torch.manual_seed(0)
    m = Hourglass(d=32, L=2, H=2, T=32, k=4, d_b=16, byte_mix="patch", byte_dh=8, patching=mode)
    x = utf8_text(32, seed)[None]
    with torch.no_grad():
        a = m(x)
        for t in range(31):
            x2 = x.clone()
            x2[:, t + 1:] = (x2[:, t + 1:] + 1 + torch.randint(0, 250, (1, 31 - t))) % 256
            b = m(x2)
            assert torch.equal(a[:, :t + 1], b[:, :t + 1]), t


@pytest.mark.parametrize("mode", ["utf8", "chars"])
def test_utf8_patching_trains_and_scores(mode):
    torch.manual_seed(0)
    m = Hourglass(d=32, L=2, H=2, T=32, k=4, d_b=16, byte_mix="patch", byte_dh=8, patching=mode)
    x = utf8_text(32, 4)[None]
    loss, _ = m.loss(m(x), x)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert m.chunk_bits(utf8_text(70, 5)[None]).shape == (1, 69)
