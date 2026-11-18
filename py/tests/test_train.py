"""Trainer: pause + resume continues bit-exactly (CPU, real data if present)."""
import copy

import pytest
import torch

from intgrad import train

need = pytest.mark.skipif(not (train.ROOT / "data/split-v1/idx.bin").exists(),
                          reason="private data not available")


@need
@pytest.mark.parametrize("quant", [True, False])
def test_resume_is_bit_exact(tmp_path, quant):
    cfg = copy.deepcopy(train.DEFAULTS)
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
