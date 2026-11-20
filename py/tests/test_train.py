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
    (False, "sequential", 1, 0), (False, "sequential", 1, 32), (False, "sequential", 1, -32),
    (False, "sequential", 4, -32), (False, "sequential", "nl", -32)])
def test_resume_is_bit_exact(tmp_path, quant, batcher, rec, streams):
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"]["batcher"] = batcher
    cfg["data"]["streams"] = abs(streams)      # 32 streams, 8 per batch: state per stream
    cfg["train"]["rec_drop"] = 0.25 if streams < 0 else 0.0   # negative: with state clearing
    if rec:                                    # plan 002: carried state (N = rec) survives resume
        cfg["model"] = {"kind": "transformer", "d": 32, "L": 2, "H": 2, "T": 32, "rec": "lru",
                        "rec_chunk": 8, "rec_head": 8}
        cfg["model"].update({"rec_nonlin": "mid"} if rec == "nl" else {"rec_state": rec})
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
@pytest.mark.parametrize("N", [1, 4])
def test_stream_pool_state_follows_its_stream(tmp_path, N):
    """lr = 0 keeps the model fixed; the state the trainer stores for each
    stream of a pool must equal a sequential pass over that stream's
    windows (with its doc resets)."""
    from intgrad import data
    from intgrad.quant import QuantConfig
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"].update(batcher="sequential", streams=32)
    cfg["model"] = {"kind": "transformer", "d": 32, "L": 2, "H": 2, "T": 32, "rec": "lru",
                    "rec_chunk": 8, "rec_state": N, "rec_head": 8}
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
@pytest.mark.parametrize("N", [1, 4])
def test_rec_drop_one_equals_no_carry(tmp_path, N):
    """train.rec_drop = 1 zeroes every carried state: the same run as
    rec_carry = false. rec_drop = 0.5 differs from both."""
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"].update(batcher="sequential", streams=16)
    cfg["model"] = {"kind": "transformer", "d": 32, "L": 2, "H": 2, "T": 32, "rec": "lru",
                    "rec_chunk": 8, "rec_state": N, "rec_head": 8}
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


def _sched(**kw):
    t = dict(train.DEFAULTS["train"], lr=1.0, warmup=10, steps=110, sched="wsd", **kw)
    return [train.lr_at(s, t) for s in range(t["steps"])]


@pytest.mark.parametrize("shape", ["sqrt", "linear"])
def test_wsd_warms_up_holds_then_decays_the_last_share(shape):
    lr = _sched(decay=0.2, decay_shape=shape)          # decay over the last 22 steps
    assert lr[9] == 1.0 and lr[0] == pytest.approx(0.1)
    assert all(v == 1.0 for v in lr[10:88])
    tail = lr[88:]
    assert all(a > b for a, b in zip(tail, tail[1:])) and 0 < tail[-1] < 0.25
    p = (100 - 88) / 22
    assert lr[100] == pytest.approx(1 - (p ** 0.5 if shape == "sqrt" else p))


def test_wsd_longer_horizon_keeps_the_stable_prefix():
    """A stable-phase pause can be resumed with a longer horizon: the
    schedule up to the old decay start is unchanged."""
    short, long = _sched(), [train.lr_at(s, dict(train.DEFAULTS["train"], lr=1.0, warmup=10,
                                                  steps=220, sched="wsd")) for s in range(220)]
    assert short[:88] == long[:88]


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        _sched(decay_shape="cubic")
    with pytest.raises(ValueError):
        train.lr_at(500, dict(train.DEFAULTS["train"], sched="cosin"))


@need
def test_quick_val_and_periodic_samples_are_logged(tmp_path):
    import json
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["model"] = {"kind": "transformer", "d": 32, "L": 1, "H": 2, "T": 32}
    cfg["train"].update(steps=8, batch=4, warmup=1, device="cpu", log_every=4, val_every=4,
                        val_frac=64, eval_every=4, eval_samples=2, eval_sample_len=8,
                        final_eval=False)
    cfg["name"] = "v"
    train.run(cfg, out_root=tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "v-s42/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs if "val_quick" in r] == [4]
    blob = (tmp_path / "v-s42/samples/0000004.bin").read_bytes()
    assert blob.count(b"\x03\n") == 2 and len(blob) == 2 * (8 + 2)
