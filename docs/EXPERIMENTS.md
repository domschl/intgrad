# intgrad — Experiment log

Byte-level LM, `ctx=4, hidden=96, batch=4, lr=5e-3, warmup=200, cosine,
4000 steps, seed 42`, evaluated as val bpb on the frozen split-v0
(800 windows; test in parens). Unigram floor: **4.74**.

## E1 — Value-scale sweep, fake-quant mode (`--scale S`)

| config | val bpb | notes |
|---|---|---|
| `--no-quant` (fp oracle) | **3.41** | reference, not deployable per C1 |
| S=64   | 4.33 | coarse grid near zero |
| S=256 (default) | 4.30 | |
| S=1024 | 4.27 | mild improvement; range shrinks to ±6.4 |

**Finding:** global scale is *not* the bottleneck — the log8↔fp gap (~0.9 bpb)
barely moves with S. Dominant suspect: forward quantization noise itself
(Δ=0.1 grid ⇒ ~3.6% multiplicative noise on *every* value: weights,
activations, and — worst — the logsumexp/logit path that computes CE).

## E2 — Gradient-scale sweep, pure-int mode (`--pure-int --gscale G`)

| G | val bpb | verdict |
|---|---|---|
| 2^12 (4096, default) | 4.32 | fine but conservative |
| **2^16 (65536)** | **4.30** | best; ≈ fake-quant level |
| 2^20 | 4.83 | overshoots → saturation |

**Findings:**
1. Gradient quantization is NOT the binding constraint yet: pure-int at a
   reasonable gscale matches fake-quant within noise (~0.02 bpb). Earlier
   grad-stats showed collapse≈0%, sat≈0% at moderate gscale — consistent.
2. The pure-int penalty at this scale is small; the 0.9 bpb gap vs fp is a
   **forward-noise problem**, which no gscale can fix.

## Conclusions / next levers (feeds M3.5)

1. **Do not chase scales globally.** S and gscale each buy ≤0.05 bpb;
   adopt gscale≈2^16 as the working point for this arch, re-tune per arch.
2. **Decompose the forward gap** (cheap code change): flags to quantize
   weights-only / activations-only / logits-LUT-only, to attribute the
   0.9 bpb. Prime suspects: (a) CE computed from quantized logsumexp,
   (b) relu/hidden activations.
3. **Per-tensor scales** (plan M4) — expected win is mostly range safety
   for larger nets, not this gap.
4. **The gap may shrink with capacity**: at ctx=4 an MLP can't use much
   signal anyway (fp 3.41 is still far from a transformer's ~2.0); rerun
   this matrix at M3 transformer scale before drawing C1 conclusions.
5. If forward noise stays dominant, options: finer DELTA at equal 8 bits
   via block-shared exponents (per-32-byte-block scale = free-ish in
   hardware terms), or two-byte "double-log" values for the logits path
   only (needs a constraint ruling — flagged for decision).

## Repro

```sh
F="--index data/split-v0/idx.bin --ctx 4 --hidden 96 --batch 4 --lr 5e-3 \
   --warmup 200 --eval-every 2000 --eval-windows 800 --steps 4000 --seed 42"
./build/ig-train $F --no-quant
./build/ig-train $F --scale 1024
./build/ig-train $F --pure-int --gscale 65536
```
