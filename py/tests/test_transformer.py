"""Byte transformer: causality, eval protocol, telemetry, presets."""
import math

import torch
import torch.nn.functional as F

from intgrad.models.transformer import PRESETS, Tap, build


def tiny(**kw):
    torch.manual_seed(0)
    return build(d=32, L=2, H=2, T=16, **kw)


def test_causal_and_shapes():
    m = tiny()
    x = torch.randint(0, 256, (3, 16))
    a = m(x)
    assert a.shape == (3, 16, 256)
    x2 = x.clone()
    x2[:, 9:] = torch.randint(0, 256, (3, 7))
    b = m(x2)
    assert torch.allclose(a[:, :9], b[:, :9], atol=1e-5)    # no peeking ahead
    assert not torch.allclose(a[:, 9:], b[:, 9:])


def test_chunk_bits_scores_every_target_once_with_enough_context():
    m = tiny()
    chunks = torch.randint(0, 256, (4, 61), dtype=torch.uint8)
    bits = m.chunk_bits(chunks)
    assert bits.shape == (4, 60) and torch.isfinite(bits).all()
    # first window: exact full-context reference
    ref = F.cross_entropy(m(chunks[:, :16]).reshape(-1, 256),
                          chunks[:, 1:17].reshape(-1).long(), reduction="none").view(4, 16)
    assert torch.allclose(bits[:, :16], ref / math.log(2), atol=1e-5)
    # a later target is scored from the window starting at T/2 = 8
    t = 20                                       # target index (byte position 20)
    lg = m(chunks[:, 8:24])
    want = F.cross_entropy(lg[:, t - 1 - 8], chunks[:, t].long(), reduction="none") / math.log(2)
    assert torch.allclose(bits[:, t - 1], want, atol=1e-5)


def test_loss_has_zloss_and_softcap_bounds_logits():
    m = tiny(zloss=1e-2, softcap=5.0)
    x = torch.randint(0, 256, (2, 16))
    lg = m(x)
    assert lg.abs().max() <= 5.0
    total, ce = m.loss(lg, x)
    assert total > ce


def test_tap_records_all_classes():
    m = tiny()
    tap = Tap()
    m(torch.randint(0, 256, (2, 16)), tap=tap)
    for k in ("emb", "q0", "k1", "scores1", "mlp_pre0", "resid1", "logits", "w.emb.weight"):
        assert k in tap.stats, k
    s = tap.stats["scores0"]
    assert 0.0 <= s["clamp"] <= 1.0 and s["underflow"] == 0.0


def test_preset_sizes():
    want = {"XS": 0.43e6, "S": 3.2e6, "M": 10.7e6, "L": 25.3e6}
    for name in PRESETS:
        n = sum(p.numel() for p in build(name).parameters())
        assert abs(n - want[name]) / want[name] < 0.05, (name, n)


import pytest


@pytest.mark.parametrize("kw", [dict(stem="conv"), dict(qkv_conv=3),
                                dict(stem="conv", qkv_conv=3)])
def test_variants_are_causal(kw):
    torch.manual_seed(0)
    m = build(d=32, L=2, H=2, T=24, **kw)
    with torch.no_grad():                     # make the identity-init convs non-trivial
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randint(0, 256, (2, 24))
    for cut in (1, 7, 15):
        x2 = x.clone()
        x2[:, cut:] = torch.randint(0, 256, (2, 24 - cut))
        a, b = m(x), m(x2)
        assert torch.allclose(a[:, :cut], b[:, :cut], atol=1e-5), (kw, cut)
        assert not torch.allclose(a[:, cut:], b[:, cut:])


def test_variant_sizes_are_close_to_baseline():
    base = sum(p.numel() for p in build("S").parameters())
    for kw in (dict(stem="conv"), dict(qkv_conv=3), dict(d=192, L=7, H=3)):
        n = sum(p.numel() for p in build("S", **kw).parameters())
        assert abs(n - base) / base < 0.03, (kw, n, base)


def test_mid_hook_sees_and_can_replace_the_mid_latent():
    m = tiny()
    x = torch.randint(0, 256, (2, 16))
    seen = {}
    base = m(x, mid_hook=lambda h: seen.setdefault("h", h) )
    assert seen["h"].shape == (2, 16, 32) and m.mid == 1
    assert torch.allclose(base, m(x))
    changed = m(x, mid_hook=lambda h: h + 1.0)
    assert not torch.allclose(base, changed)


@pytest.mark.parametrize("k,dil", [(3, 1), (4, 2)])
def test_depthwise_shift_path_matches_conv1d(k, dil):
    from intgrad.models.transformer import CausalConv
    torch.manual_seed(0)
    c = CausalConv(12, 12, k, dilation=dil, depthwise=True)
    torch.nn.init.normal_(c.conv.weight)
    x = torch.randn(2, 9, 12)
    ref = c.conv(F.pad(x.transpose(1, 2), (c.pad, 0))).transpose(1, 2)
    assert torch.allclose(c(x), ref, atol=1e-6)


QAT_ALL = dict(acts=True, resid="q8", softmax="maxrel", probs=True)


def test_explicit_attention_matches_sdpa():
    from types import SimpleNamespace
    m = tiny()
    blk = m.blocks[0]
    q, k, v = (torch.randn(2, 2, 16, 16) for _ in range(3))
    ref = blk.attend(q, k, v)
    blk.qat = SimpleNamespace(explicit_attn=True, softmax="none", probs=False)
    assert torch.allclose(blk.attend(q, k, v), ref, atol=1e-5)


def test_qat_sites_put_activations_on_the_grid():
    from intgrad import quant as Q
    m = tiny(qat=QAT_ALL)
    seen = {}
    for name, mod in m.named_modules():
        if type(mod).__name__ == "QSite":
            mod.register_forward_hook(lambda md, i, o, n=name: seen.__setitem__(n, (o, md)))
    m(torch.randint(0, 256, (2, 16)))
    assert len(seen) == 2 + 2 * (8 + 2)
    for n, (o, md) in seen.items():                         # on the grid at some pow2 scale
        o = o.detach()
        assert any(torch.equal(Q.rt8(o, 2.0 ** e), o) for e in range(-10, 30)), n


def test_qat_model_is_causal_after_warmup_and_has_gradients():
    m = tiny(qat=QAT_ALL)
    x = torch.randint(0, 256, (2, 16))
    m(x)                                                   # fill the scale histories
    m.eval()
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 256
    a, b = m(x), m(x2)
    assert torch.equal(a[:, :10], b[:, :10])
    m.train()
    loss, _ = m.loss(m(x), x)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_qat_scales_are_frozen_in_eval():
    m = tiny(qat=QAT_ALL)
    x = torch.randint(0, 256, (2, 16))
    m(x)
    h = [s.hist.clone() for s in m.modules() if type(s).__name__ == "QSite"]
    m.eval()
    m(x * 0 + 7)
    assert all(torch.equal(a, s.hist) for a, s in
               zip(h, (s for s in m.modules() if type(s).__name__ == "QSite")))
