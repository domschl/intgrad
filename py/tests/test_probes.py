"""Capability probes as tests (plan/001 §4.3)."""
import pytest
import torch

from intgrad.probes import p1_memorize
from intgrad.quant import QuantConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODES = {"fp": QuantConfig(enabled=False), "w2": QuantConfig(),
         "w1_sr": QuantConfig(w2=False), "w2_pure_int": QuantConfig(pure_int=True)}


@pytest.mark.parametrize("mode", list(MODES))
def test_p1_memorize(mode):
    r = p1_memorize(MODES[mode], DEV)
    assert r["pass"], r
