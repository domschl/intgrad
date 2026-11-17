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
