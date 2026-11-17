"""PyTorch byte numerics: properties that hold independently of C."""
import numpy as np
import torch

from intgrad import quant as Q


def test_byte_grid_roundtrip_and_monotone():
    x = torch.logspace(-6, 1.3, 20000)
    q = Q.encode8(x, Q.SCALE)
    assert torch.all(q[1:] >= q[:-1]) and q.max() <= 127
    rt = Q.rt8(x)
    rel = ((rt - x).abs() / x)[x > 0.01]
    assert rel.max() < 0.06    # symlog: half-step grows toward 0 as (1+a)/a
    assert torch.equal(Q.rt8(-x), -rt)
    assert Q.rt8(torch.tensor([0.0, float("nan")])).tolist() == [0.0, 0.0]
    assert Q.rt8(torch.tensor([1e9])).item() == Q.max_abs(Q.SCALE)


def test_sr_is_unbiased():
    x = torch.full((400_000,), 0.3)
    u = torch.rand(400_000, generator=torch.Generator().manual_seed(0))
    assert abs(Q.rt8_sr(x, u).mean().item() - 0.3) < 1e-3
    c = Q.enc16(x, Q.SCALE, u)
    assert abs(Q.dec16(c, Q.SCALE).double().mean().item() - 0.3) < 2e-6


def test_w16_forward_view_is_byte_grid():
    x = torch.linspace(-3, 3, 5001)
    wq = Q.enc16(x, Q.SCALE)
    assert torch.equal(Q.w16_fwd(wq), Q.rt8(Q.dec16(wq, Q.SCALE)))
    big = x.abs() > 0.05
    assert ((Q.dec16(wq, Q.SCALE) - x).abs() / x.abs())[big].max() < 3e-4


def test_fx_rounds_half_away_and_clamps():
    st = np.float32(0.1)
    q = torch.tensor([0.5, 1.5, 2.5, -0.5, -2.5, 126.5, 300.0, -300.0]) * float(st)
    got = (Q.fx(q) / torch.tensor(st)).round().tolist()
    assert got == [1, 2, 3, -1, -3, 127, 127, -128]


def test_ste_masks_clamped_values():
    qc = Q.QuantConfig()
    x = torch.tensor([0.5, 30.0, -30.0], requires_grad=True)
    qc.act(x).sum().backward()
    assert x.grad.tolist() == [1.0, 0.0, 0.0]
    y = torch.tensor([1.0, 13.0], requires_grad=True)
    qc.logit(y).sum().backward()
    assert y.grad.tolist() == [1.0, 0.0]


def test_pow2_scale_fits_absmax():
    for mx in (1e-12, 3e-4, 0.7, 1.0, 25.6, 4e5):
        s = Q.pow2_scale(torch.tensor(mx, dtype=torch.float32)).item()
        assert np.log2(s) == int(np.log2(s))
        assert mx * s <= Q.DEC8[127] < 2 * mx * s


def test_maxrel_logits_grid_and_gradient():
    x = torch.tensor([[2.0, 1.94, -10.0, -30.0]], requires_grad=True)
    y = Q.logit_maxrel(x)
    d = (y - 2.0).detach()
    assert torch.allclose(d, torch.tensor([[0.0, -0.1, -12.0, -25.5]]), atol=1e-5)
    y.sum().backward()
    assert x.grad.tolist() == [[1.0, 1.0, 1.0, 0.0]]       # only the clamped one is cut
