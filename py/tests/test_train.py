"""Trainer: pause + resume continues bit-exactly (CPU, real data if present)."""
import copy

import pytest
import torch

from intgrad import train

need = pytest.mark.skipif(not (train.ROOT / "data/split-v1/idx.bin").exists(),
                          reason="private data not available")


@need
@pytest.mark.parametrize("quant,batcher,rec,streams", [
    (True, "random", False, 0), (False, "random", False, 0), (False, "sequential", False, 0),
    (False, "sequential", True, 0), (False, "sequential", True, 32), (False, "sequential", True, -32)])
def test_resume_is_bit_exact(tmp_path, quant, batcher, rec, streams):
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"]["batcher"] = batcher
    cfg["data"]["streams"] = abs(streams)      # 32 streams, 8 per batch: state per stream
    cfg["train"]["rec_drop"] = 0.25 if streams < 0 else 0.0   # negative: with state clearing
    if rec:                                    # plan 002: carried recurrent state survives resume
        cfg["model"] = {"kind": "transformer", "d": 32, "L": 2, "H": 2, "T": 32, "rec": "lru",
                        "rec_chunk": 8}
    else:
        cfg["model"].update(ctx=3, hidden=16)
    cfg["quant"]["enabled"] = quant
    cfg["quant"]["w2_sr"] = quant              # exercise the SR generator too
    cfg["train"].update(steps=40, batch=8, warmup=5, device="cpu", log_every=10,
                        final_eval=False)
    straight = copy.deepcopy(cfg)
    straight["name"] = "a"
    train.run(straight, out_root=tmp_path)
    paused = copy.deepcopy(cfg)
    paused["name"] = "b"
    paused["train"]["stop"] = 20
    train.run(paused, out_root=tmp_path)
    resumed = copy.deepcopy(cfg)
    resumed["name"] = "b"
    train.run(resumed, resume=tmp_path / "b-s42")
    a = torch.load(tmp_path / "a-s42/ckpt.pt", weights_only=False)
    b = torch.load(tmp_path / "b-s42/ckpt.pt", weights_only=False)
    assert a["step"] == b["step"] == 40
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k


def test_utf8_valid_fraction_allows_truncated_tail_only():
    ok = "Grüße, 世界".encode()
    samples = [ok, ok[:-1], ok[:3] + b"\xff" + ok[3:], b"\xc3"]
    # whole; truncated in the last char (allowed); invalid byte inside; lone lead at end (allowed)
    assert train.utf8_valid_fraction(samples) == 0.75


def test_micro_batches_give_the_same_gradient():
    from intgrad.models.transformer import build
    torch.manual_seed(0)
    m = build(d=32, L=2, H=2, T=16)
    x = torch.randint(0, 256, (8, 16))
    y = torch.randint(0, 256, (8, 16))
    m.loss(m(x), y)[0].backward()
    full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    for i in range(0, 8, 2):
        (m.loss(m(x[i:i + 2]), y[i:i + 2])[0] * 0.25).backward()
    for a, p in zip(full, m.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)


def test_lion_anchor_decays_toward_init():
    from intgrad.lion import Lion
    from intgrad.quant import QuantConfig
    g = torch.nn.Parameter(torch.ones(4) * 3.0)
    opt = Lion([{"params": [g], "wd": 1.0, "anchor": "init"}], lr=0.01, wd=1.0,
               qc=QuantConfig(enabled=False))
    with torch.no_grad():
        g.copy_(torch.tensor([3.0, 5.0, -1.0, 3.0]))        # moved away from init 3
    for _ in range(2000):
        g.grad = torch.zeros(4)
        opt.step()
    assert torch.allclose(g.detach(), torch.full((4,), 3.0), atol=1e-3)


def test_lr_multipliers_split_embedding_and_gains():
    from intgrad.models.transformer import build
    m = build("XS")
    gs = train.param_groups(m, 3.0, lr_emb=2.0, lr_vec=0.5)
    by = {id(p): g for g in gs for p in g["params"]}
    assert by[id(m.emb.weight)]["lr_mult"] == 2.0 and by[id(m.emb.weight)]["wd"] == 3.0
    assert by[id(m.nf.weight)]["lr_mult"] == 0.5
    assert by[id(m.blocks[0].qkv.weight)].get("lr_mult", 1.0) == 1.0
    assert sum(len(g["params"]) for g in gs) == len(list(m.parameters()))


@need
def test_stream_pool_state_follows_its_stream(tmp_path):
    """lr = 0 keeps the model fixed; the state the trainer stores for each
    stream of a pool must equal a sequential pass over that stream's
    windows (with its doc resets)."""
    from intgrad import data
    from intgrad.quant import QuantConfig
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"].update(batcher="sequential", streams=32)
    cfg["model"] = {"kind": "transformer", "d": 32, "L": 2, "H": 2, "T": 32, "rec": "lru",
                    "rec_chunk": 8}
    cfg["train"].update(steps=14, batch=8, micro_batch=4, warmup=1, lr=0.0, device="cpu",
                        log_every=10, final_eval=False)
    cfg["name"] = "p"
    train.run(cfg, out_root=tmp_path)
    ck = torch.load(tmp_path / "p-s42/ckpt.pt", weights_only=False)
    m = train.build_model(cfg, QuantConfig(**cfg["quant"]), torch.Generator().manual_seed(0))
    m.load_state_dict(ck["model"])
    m.eval()
    bt = data.SequentialBatcher(data.Index(train.ROOT / cfg["data"]["index"]), data.TRAIN, batch=8, seq=32,
                                seed=cfg["train"]["seed"], streams=32)
    state = {}
    with torch.no_grad():
        for _ in range(14):
            b = bt.next()
            for i, s in enumerate(b.rows.tolist()):
                _, state[s] = m.forward_state(torch.from_numpy(b.x[i:i + 1]), state.get(s),
                                              torch.from_numpy(b.doc_start[i:i + 1]))
    assert len(state) == 32
    for s, st in state.items():
        for j in range(len(st)):
            torch.testing.assert_close(ck["rec_state"][j][s], st[j][0], rtol=1e-4, atol=1e-5)


@need
def test_rec_drop_one_equals_no_carry(tmp_path):
    """train.rec_drop = 1 zeroes every carried state: the same run as
    rec_carry = false. rec_drop = 0.5 differs from both."""
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"].update(batcher="sequential", streams=16)
    cfg["model"] = {"kind": "transformer", "d": 32, "L": 2, "H": 2, "T": 32, "rec": "lru",
                    "rec_chunk": 8}
    cfg["train"].update(steps=12, batch=8, warmup=2, device="cpu", log_every=10, final_eval=False)
    out = {}
    for name, kw in {"drop1": {"rec_drop": 1.0}, "nocarry": {"rec_carry": False},
                     "drop5": {"rec_drop": 0.5}, "carry": {}}.items():
        c = copy.deepcopy(cfg)
        c["name"] = name
        c["train"].update(kw)
        train.run(c, out_root=tmp_path)
        out[name] = torch.load(tmp_path / f"{name}-s42/ckpt.pt", weights_only=False)["model"]
    same = lambda a, b: all(torch.equal(out[a][k], out[b][k]) for k in out[a])
    assert same("drop1", "nocarry")
    assert not same("drop5", "carry") and not same("drop5", "nocarry")
