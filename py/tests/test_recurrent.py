"""Plan 002 recurrent branch (models/recurrent.py) and its use in ByteTransformer."""
import pytest
import torch

from intgrad.models.recurrent import RecMix, linear_scan, ssd_scan
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


def _seq_ssd(log_a, b, k, q, S0, reset):
    a = torch.exp(log_a).masked_fill(reset[..., None], 0.0)
    S, out = S0.clone(), []
    for t in range(b.shape[1]):
        S = a[:, t, :, None, None] * S + b[:, t, :, :, None] * k[:, t, :, None, :]
        out.append((S * q[:, t, :, None, :]).sum(-1))
    return torch.stack(out, 1), S


@pytest.mark.parametrize("T,chunk", [(64, 16), (50, 16), (7, 32), (33, 1)])
def test_ssd_scan_matches_sequential(T, chunk):
    g = torch.Generator().manual_seed(0)
    B, H, P, N = 3, 2, 4, 5
    log_a = -torch.rand(B, T, H, generator=g)
    b = torch.randn(B, T, H, P, generator=g)
    k, q = torch.randn(B, T, H, N, generator=g), torch.randn(B, T, H, N, generator=g)
    S0 = torch.randn(B, H, P, N, generator=g)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[0, T // 2] = True                                 # a reset in the middle
    reset[1, 0] = True
    if T > 20:
        reset[2, [3, 4, T - 2]] = True                      # several, adjacent, near the end
    for s0 in (S0, None):
        ref = _seq_ssd(log_a, b, k, q, torch.zeros_like(S0) if s0 is None else s0, reset)
        o, S = ssd_scan(log_a, b, k, q, s0, reset, chunk)
        torch.testing.assert_close(o, ref[0])
        torch.testing.assert_close(S, ref[1])


@pytest.mark.parametrize("state", [1, 4])
def test_recmix_carry_equals_one_long_pass(state):
    """Without the conv (which does not carry), two halves with the state
    handed over give exactly the single pass over both."""
    torch.manual_seed(0)
    m = RecMix(16, 2, conv=0, chunk=8, state=state, head=4)
    x = torch.randn(2, 40, 16)
    full, _ = m(x)
    y1, s = m(x[:, :24])
    y2, _ = m(x[:, 24:], s)
    torch.testing.assert_close(torch.cat([y1, y2], 1), full)


def test_recmix_decay_init_range():
    m = RecMix(64, 2)
    a = torch.sigmoid(m.lam) ** 8.0
    assert a.min() >= 0.9 - 1e-4 and a.max() <= 0.999 + 1e-4


@pytest.fixture(params=[1, 4], ids=["N1", "N4"])
def state(request):
    return request.param


def _model(state=1):
    torch.manual_seed(0)
    return ByteTransformer(d=32, L=2, H=2, T=32, rec="lru", rec_chunk=8,
                           rec_state=state, rec_head=8).eval()


def test_state_one_keeps_the_n1_parameters():
    """rec_state 1 is the N1/N2 model: same parameter shapes and init."""
    torch.manual_seed(0)
    a = ByteTransformer(d=32, L=2, H=2, T=32, rec="lru", rec_chunk=8).state_dict()
    b = _model(1).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)


def _windows():
    g = torch.Generator().manual_seed(1)
    return (torch.randint(97, 123, (2, 32), generator=g),
            torch.randint(97, 123, (2, 32), generator=g))


def test_forward_is_zero_state_forward_state(state):
    m = _model(state)
    x, _ = _windows()
    torch.testing.assert_close(m(x), m.forward_state(x)[0])


def test_recurrent_model_is_causal(state):
    m = _model(state)
    x, _ = _windows()
    base = m(x)
    for t in range(0, 31, 5):
        y = x.clone()
        y[:, t + 1:] = (y[:, t + 1:] + 7) % 256
        torch.testing.assert_close(m(y)[:, : t + 1], base[:, : t + 1])


def test_carried_state_reaches_the_next_window(state):
    m = _model(state)
    w1, w2 = _windows()
    _, s = m.forward_state(w1)
    _, s_other = m.forward_state((w1 + 3) % 256)
    a, _ = m.forward_state(w2, s)
    b, _ = m.forward_state(w2, s_other)
    assert not torch.allclose(a, b)                         # earlier window matters
    assert all(t.shape == ((2, 32) if state == 1 else (2, 4, 8, 4)) for t in s)


def test_reset_erases_the_previous_window(state):
    """With a document start at position p of window 2, the recurrent
    state from p on must not depend on window 1. Checked with one layer:
    with more, a later layer's attention still reads positions < p, whose
    states did see window 1 (attention is not document-masked, as in the
    base model)."""
    torch.manual_seed(0)
    m = ByteTransformer(d=32, L=1, H=2, T=32, rec="lru", rec_chunk=8,
                        rec_state=state, rec_head=8).eval()
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
