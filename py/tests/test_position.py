"""M7 position curve / cross-window loss (intgrad.position)."""
import numpy as np
import torch

from intgrad import data
from intgrad.models.transformer import ByteTransformer
from intgrad.position import measure, stream_windows, window_bits
from test_data import corpus  # noqa: F401  (pytest fixture)

T, W = 32, 8


def _model():
    torch.manual_seed(0)
    return ByteTransformer(d=32, L=1, H=2, T=T).eval()


def test_stream_windows_are_consecutive(corpus):  # noqa: F811
    x, st = stream_windows(corpus, data.VAL, streams=4, T=T, n=6, seed=1)
    assert x.shape == (4, 7, T + 1) and st.shape == x.shape
    # the target byte closing window j is the first input of window j+1
    np.testing.assert_array_equal(x[:, :-1, T], x[:, 1:, 0])
    # a stream is its docs back to back, EOD-separated: EOD follows a doc end
    for s in range(4):
        flat = np.concatenate([x[s, 0]] + [x[s, j, 1:] for j in range(1, 7)])
        sflat = np.concatenate([st[s, 0]] + [st[s, j, 1:] for j in range(1, 7)])
        after_eod = np.flatnonzero(flat[:-1] == data.EOD) + 1
        assert sflat[after_eod].all()


def test_warm_and_cold_score_the_right_bytes(corpus):  # noqa: F811
    S, n = 3, 4
    x, st = stream_windows(corpus, data.VAL, streams=S, T=T, n=n, seed=2)
    m = _model()
    res, curve = measure(m, x, st, W=W, batch=4, device="cpu")
    assert curve.shape == (T,) and res["windows"] == S * n
    # recompute from the flat stream: window j starts at stream offset j*T;
    # warm scores its first W targets from the T-W bytes before them
    cold, warm, cont = [], [], []
    for s in range(S):
        flat = np.concatenate([x[s, 0]] + [x[s, j, 1:] for j in range(1, n + 1)])
        fst = np.concatenate([st[s, 0]] + [st[s, j, 1:] for j in range(1, n + 1)])
        for j in range(1, n + 1):
            p = j * T
            win = torch.from_numpy(flat[p:p + T + 1].copy())[None]
            ctx = torch.from_numpy(flat[p + W - T:p + W + 1].copy())[None]
            cold.append(window_bits(m, win, 1)[0, :W].mean().item())
            warm.append(window_bits(m, ctx, 1)[0, T - W:].mean().item())
            cont.append(not fst[p:p + W + 1].any() and not (flat[p:p + W + 1] == data.EOD).any())
    cont = np.array(cont)
    assert cont.sum() == res["cross_window"]["cold"]["n"] > 0
    cw = res["cross_window"]
    assert abs(np.array(cold)[cont].mean() - cw["cold"]["bpb"]) < 6e-5  # report rounds to 4 decimals
    assert abs(np.array(warm)[cont].mean() - cw["warm"]["bpb"]) < 6e-5
    assert abs(cw["cold"]["bpb"] - cw["warm"]["bpb"]) > 1e-3   # context matters


def test_continuation_excludes_doc_starts():
    m = _model()
    rng = np.random.default_rng(0)
    x = rng.integers(97, 123, size=(2, 3, T + 1)).astype(np.uint8)
    st = np.zeros(x.shape, bool)
    st[0, 1, 3] = True                           # doc starts inside the first W bytes
    st[1, 2, 0] = True                           # doc starts at the window start
    res, _ = measure(m, x, st, W=W, batch=4, device="cpu")
    cw = res["cross_window"]
    assert cw["cold"]["n"] == 2 and cw["doc_start"]["n"] == 1


def test_carried_scores_windows_in_stream_order(corpus):  # noqa: F811
    """Recurrent model: 'carried' equals scoring each stream's windows one
    after another with the state handed over (and differs from cold)."""
    S, n = 2, 3
    x, st = stream_windows(corpus, data.VAL, streams=S, T=T, n=n, seed=3)
    torch.manual_seed(0)
    m = ByteTransformer(d=32, L=1, H=2, T=T, rec="lru", rec_chunk=8).eval()
    res, _ = measure(m, x, st, W=W, batch=1, device="cpu")
    tot = []
    for s in range(S):
        state = None
        for j in range(n + 1):
            xb = torch.from_numpy(x[s, j][None].copy())
            lg, state = m.forward_state(xb[:, :-1], state, torch.from_numpy(st[s, j, :-1][None].copy()))
            if j:
                tot.append(torch.nn.functional.cross_entropy(lg[0], xb[0, 1:].long()).item() / np.log(2))
    assert abs(np.mean(tot) - res["bpb_carried"]["bpb"]) < 6e-5
    assert res["bpb_carried"]["bpb"] != res["bpb"]["bpb"]
