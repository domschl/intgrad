"""Plan 002 recurrent branch (models/recurrent.py) and its use in ByteTransformer."""
import pytest
import torch

from intgrad.models.recurrent import RecMix, linear_scan
from intgrad.models.transformer import ByteTransformer


def _seq_scan(a, b, h0):
    h, out = h0.clone(), []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, 1)


@pytest.mark.parametrize("T,chunk", [(64, 16), (50, 16), (7, 32), (33, 1)])
def test_linear_scan_matches_sequential(T, chunk):
    g = torch.Generator().manual_seed(0)
    a = torch.rand(3, T, 5, generator=g)
    a[0, T // 2] = 0.0                                     # a reset in the middle
    b = torch.randn(3, T, 5, generator=g)
    h0 = torch.randn(3, 5, generator=g)
    torch.testing.assert_close(linear_scan(a, b, h0, chunk), _seq_scan(a, b, h0))
    torch.testing.assert_close(linear_scan(a, b, None, chunk), _seq_scan(a, b, torch.zeros(3, 5)))


def test_recmix_carry_equals_one_long_pass():
    """Without the conv (which does not carry), two halves with the state
    handed over give exactly the single pass over both."""
    torch.manual_seed(0)
    m = RecMix(16, 2, conv=0, chunk=8)
    x = torch.randn(2, 40, 16)
    full, _ = m(x)
    y1, s = m(x[:, :24])
    y2, _ = m(x[:, 24:], s)
    torch.testing.assert_close(torch.cat([y1, y2], 1), full)


def test_recmix_decay_init_range():
    m = RecMix(64, 2)
    a = torch.sigmoid(m.lam) ** 8.0
    assert a.min() >= 0.9 - 1e-4 and a.max() <= 0.999 + 1e-4


def _model():
    torch.manual_seed(0)
    return ByteTransformer(d=32, L=2, H=2, T=32, rec="lru", rec_chunk=8).eval()


def _windows():
    g = torch.Generator().manual_seed(1)
    return (torch.randint(97, 123, (2, 32), generator=g),
            torch.randint(97, 123, (2, 32), generator=g))


def test_forward_is_zero_state_forward_state():
    m = _model()
    x, _ = _windows()
    torch.testing.assert_close(m(x), m.forward_state(x)[0])


def test_recurrent_model_is_causal():
    m = _model()
    x, _ = _windows()
    base = m(x)
    for t in range(0, 31, 5):
        y = x.clone()
        y[:, t + 1:] = (y[:, t + 1:] + 7) % 256
        torch.testing.assert_close(m(y)[:, : t + 1], base[:, : t + 1])


def test_carried_state_reaches_the_next_window():
    m = _model()
    w1, w2 = _windows()
    _, s = m.forward_state(w1)
    _, s_other = m.forward_state((w1 + 3) % 256)
    a, _ = m.forward_state(w2, s)
    b, _ = m.forward_state(w2, s_other)
    assert not torch.allclose(a, b)                         # earlier window matters
    assert all(t.shape == (2, 32) for t in s)


def test_reset_erases_the_previous_window():
    """With a document start at position p of window 2, the recurrent
    state from p on must not depend on window 1. Checked with one layer:
    with more, a later layer's attention still reads positions < p, whose
    states did see window 1 (attention is not document-masked, as in the
    base model)."""
    torch.manual_seed(0)
    m = ByteTransformer(d=32, L=1, H=2, T=32, rec="lru", rec_chunk=8).eval()
    w1, w2 = _windows()
    p = 11
    reset = torch.zeros(2, 32, dtype=torch.bool)
    reset[:, p] = True
    _, s = m.forward_state(w1)
    _, s_other = m.forward_state((w1 + 3) % 256)
    a, _ = m.forward_state(w2, s, reset)
    b, _ = m.forward_state(w2, s_other, reset)
    torch.testing.assert_close(a[:, p:], b[:, p:])
    assert not torch.allclose(a[:, :p], b[:, :p])           # before the reset it still does
