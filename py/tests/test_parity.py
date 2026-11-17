"""Parity: PyTorch byte numerics vs the C oracle (plan/001 §4.2).

Primitives and one Lion step must match BIT-EXACTLY; the MLP forward is
bit-exact by construction (same accumulation order); loss and gradients
match to float tolerance (C expf/logf vs torch's implementations).
"""
import numpy as np
import pytest
import torch

from intgrad import quant as Q
from intgrad.lion import Lion
from intgrad.models.mlp import ByteMLP

from cref import XorShift64, load, ptr, table, u8_to_signed

L = load()
pytestmark = pytest.mark.skipif(L is None, reason="build/libintgrad_ref.so not built")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def cvec(fn, *cols, dtype=np.float32):
    return np.array([fn(*vals) for vals in zip(*cols)], dtype=dtype)


def sweep(scale: float) -> np.ndarray:
    """Values straddling every byte threshold (±3 ulp) plus random magnitudes."""
    thr = Q.THR8[1:] / np.float32(scale)
    xs = [thr]
    for k in range(1, 4):
        xs.append(np.nextafter(thr, np.float32(np.inf)))
        xs.append(np.nextafter(thr, np.float32(0)))
        thr = xs[-2]
    rng = np.random.default_rng(int(scale) % 1000 + 7)
    xs.append((10.0 ** rng.uniform(-9, 3, 3000)).astype(np.float32))
    x = np.concatenate(xs).astype(np.float32)
    return np.concatenate([x, -x, np.array([0.0, 1e30, -1e30], np.float32)])


def test_tables_identical():
    for which, ours in enumerate((Q.DEC8, Q.THR8, Q.DEC16, Q.THR16)):
        theirs = table(L, which)
        assert theirs.shape == ours.shape
        assert np.array_equal(theirs.view(np.uint32), ours.view(np.uint32)), which


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [256.0, 1.0, 2.0 ** -10, 2.0 ** 20])
def test_byte_grid_codes_exact(device, scale):
    x = sweep(scale)
    c_codes = np.array([u8_to_signed(L.ig_encode_s(v, scale)) for v in x], np.int16)
    t_codes = Q.encode8(torch.from_numpy(x).to(device), scale).cpu().numpy()
    assert np.array_equal(c_codes, t_codes)
    c_rt = cvec(L.ig_roundtrip_s, x, [scale] * len(x))
    t_rt = Q.rt8(torch.from_numpy(x).to(device), scale).cpu().numpy()
    assert np.array_equal(c_rt.view(np.uint32), t_rt.view(np.uint32))


@pytest.mark.parametrize("device", DEVICES)
def test_sr_16bit_fx_pow2_exact(device):
    rng = np.random.default_rng(3)
    x = np.concatenate([sweep(256.0)[:2000], rng.normal(0, 3, 2000).astype(np.float32)])
    u = rng.random(len(x)).astype(np.float32)
    tx, tu = torch.from_numpy(x).to(device), torch.from_numpy(u).to(device)
    S = 256.0
    eq = lambda a, b: np.array_equal(np.asarray(a).view(np.uint32) if np.asarray(a).dtype == np.float32 else a,
                                     np.asarray(b).view(np.uint32) if np.asarray(b).dtype == np.float32 else b)
    assert eq(cvec(L.ig_roundtrip_sr_s, x, [S] * len(x), u), Q.rt8_sr(tx, tu, S).cpu().numpy())
    c16 = cvec(L.ig_enc16, x, [-1.0] * len(x), [S] * len(x), dtype=np.int16)
    assert eq(c16, Q.enc16(tx, S).cpu().numpy())
    assert eq(cvec(L.ig_enc16, x, u, [S] * len(x), dtype=np.int16), Q.enc16(tx, S, tu).cpu().numpy())
    assert eq(cvec(L.ig_dec16, c16, [S] * len(x)), Q.dec16(torch.from_numpy(c16).to(device), S).cpu().numpy())
    assert eq(cvec(L.ig_w16_fwd, c16), Q.w16_fwd(torch.from_numpy(c16).to(device)).cpu().numpy())
    xf = (x * 3).astype(np.float32)
    xf = np.concatenate([xf, (np.arange(-1300, 1300) * np.float32(0.05)).astype(np.float32)])
    assert eq(cvec(L.ig_fx_roundtrip, xf), Q.fx(torch.from_numpy(xf).to(device)).cpu().numpy())
    mx = np.abs(np.concatenate([x, [1e-38, 3e-45, 1e38]])).astype(np.float32)
    t_s = np.array([Q.pow2_scale(torch.tensor(v, device=device)).item() for v in mx], np.float32)
    assert eq(cvec(L.ig_pow2_scale, mx), t_s)


# ---------------------------------------------------------------- Lion + MLP

CTX, NH = 3, 8


def c_model(quant, pure_int, w2, w2_sr, sr, wd=0.1):
    L.ig_set_modes(quant, pure_int, w2, w2_sr, sr, wd)
    L.ig_mlp_opt_reset()
    return L.ig_mlp_new(CTX, NH, 5)


def random_state(n, rng, quant):
    w = (rng.normal(0, 0.05, n)).astype(np.float32)
    wt = torch.from_numpy(w)
    wq = Q.enc16(wt, Q.SCALE).numpy() if quant else np.zeros(n, np.int16)
    data = Q.w16_fwd(torch.from_numpy(wq)).numpy() if quant else w
    grad = (rng.normal(0, 1e-3, n) * (rng.random(n) < 0.7)).astype(np.float32)
    mom = (rng.normal(0, 1e-3, n)).astype(np.float32)
    return data, wq, grad, mom


def torch_model(device, qc, data):
    m = ByteMLP(CTX, NH, qc).to(device)
    with torch.no_grad():
        off = 0
        for p in m.flat_params():
            p.copy_(torch.from_numpy(data[off:off + p.numel()]).view_as(p))
            off += p.numel()
    return m


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["float", "w2", "w2_sr", "w1_sr", "w1_rtn", "w2_pure_int"])
def test_lion_step_bit_exact(device, mode):
    quant = mode != "float"
    w2 = mode.startswith("w2")
    w2_sr, sr, pure = mode == "w2_sr", mode != "w1_rtn", mode.endswith("pure_int")
    cm = c_model(quant, pure, w2, w2_sr, sr)
    n = L.ig_mlp_nparams(cm)
    rng = np.random.default_rng(11)
    data, wq, grad, mom = random_state(n, rng, quant)
    if quant and not w2:
        data = Q.rt8(torch.from_numpy(data)).numpy()
    L.ig_mlp_import(cm, ptr(data), ptr(wq), ptr(grad))
    L.ig_lion_set_state(ptr(mom), n)
    seed = 0x1234_5678_9ABC_DEF1
    L.ig_sr_set_state(seed)
    lr = np.float32(3e-3)
    L.ig_mlp_lion_update(cm, lr)
    c_data, c_wq = np.empty(n, np.float32), np.empty(n, np.int16)
    L.ig_mlp_export(cm, ptr(c_data), ptr(c_wq), None)
    c_mom = np.empty(n, np.float32)
    L.ig_lion_get_state(ptr(c_mom))
    L.ig_mlp_free(cm)

    qc = Q.QuantConfig(enabled=quant, w2=w2, w2_sr=w2_sr, sr=sr, pure_int=pure)
    tm = torch_model(device, qc, data)
    xs = XorShift64(seed)
    opt = Lion(tm.flat_params(), lr=float(lr), wd=0.1, qc=qc,
               sr_uniform=lambda k, dev: torch.from_numpy(xs.uniforms(k)).to(dev))
    off = 0
    for p in tm.flat_params():
        k = p.numel()
        p.grad = torch.from_numpy(grad[off:off + k]).view_as(p).to(device)
        opt.state[p]["m"] = torch.from_numpy(mom[off:off + k]).view_as(p).to(device)
        if quant and w2:
            opt.state[p]["wq"] = torch.from_numpy(wq[off:off + k]).view_as(p).to(device)
        off += k
    opt.step()
    t_data = torch.cat([p.detach().reshape(-1) for p in tm.flat_params()]).cpu().numpy()
    t_mom = torch.cat([opt.state[p]["m"].reshape(-1) for p in tm.flat_params()]).cpu().numpy()
    assert np.array_equal(c_data.view(np.uint32), t_data.view(np.uint32))
    assert np.array_equal(c_mom.view(np.uint32), t_mom.view(np.uint32))
    if quant and w2:
        t_wq = torch.cat([opt.state[p]["wq"].reshape(-1) for p in tm.flat_params()]).cpu().numpy()
        assert np.array_equal(c_wq, t_wq)
    assert not np.array_equal(c_data, data)                 # something moved


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quant", [True, False])
def test_mlp_forward_exact_backward_close(device, quant):
    cm = c_model(quant, 0, 1, 0, 1)
    n = L.ig_mlp_nparams(cm)
    rng = np.random.default_rng(5)
    data, wq, _, _ = random_state(n, rng, quant)
    data = (data * 4).astype(np.float32)                     # livelier logits
    if quant:
        data = Q.rt8(torch.from_numpy(data)).numpy()
    L.ig_mlp_import(cm, ptr(data), None, None)
    B = 16
    win = rng.integers(0, 256, size=(B, CTX + 1), dtype=np.uint8)
    c_logits = np.empty((B, 256), np.float32)
    L.ig_mlp_zero_grad(cm)
    c_loss = 0.0
    for b in range(B):
        L.ig_mlp_logits(cm, ptr(np.ascontiguousarray(win[b, :CTX])), ptr(c_logits[b]))
        c_loss += L.ig_mlp_window_backward(cm, ptr(np.ascontiguousarray(win[b])))
    L.ig_mlp_scale_grads(cm, np.float32(1.0 / B))
    c_grad = np.empty(n, np.float32)
    L.ig_mlp_export(cm, None, None, ptr(c_grad))
    L.ig_mlp_free(cm)

    qc = Q.QuantConfig(enabled=quant)
    tm = torch_model(device, qc, data)
    x = torch.from_numpy(win[:, :CTX]).to(device)
    y = torch.from_numpy(win[:, CTX]).long().to(device)
    logits = tm(x)
    assert np.array_equal(c_logits.view(np.uint32), logits.detach().cpu().numpy().view(np.uint32))
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert abs(loss.item() - c_loss / B) < 1e-5 * max(1.0, abs(c_loss / B))
    t_grad = torch.cat([p.grad.reshape(-1) for p in tm.flat_params()]).cpu().numpy()
    np.testing.assert_allclose(t_grad, c_grad, rtol=1e-4, atol=1e-7)
