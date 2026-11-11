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

## E3 — Memorization probe (`data/overfit.txt`, 240 B)

Can the net drive loss → 0 on data small enough to memorize? Separates
optimization/quantization floors from generalization.

**Before the clamp-STE fix:** fp 0.008 ✓, but *all* quantized modes stuck
at ≈ 1.4 nats regardless of gscale/S — suspicious.

## E4 — Attribution ladder + the clamp bug (fixed)

Isolation flags added: `--fp-logits` (float from logits onward),
`--fp-params` (float parameter storage), `--loss-fp`, plus the D6 hybrid
grid (fixed-point bytes for logits/CE, exact int32-style GEMM accumulation,
requantized once per logit).

Finding: float logits changed nothing (1.36 ≈ 1.40) → not the logit grid.
The real culprit was **STE ignoring the quantizer's clamp region**:
d(clamp)/dx = 0 outside range, but gradients flowed through saturated
values — Adam accumulated momentum against immovable stored numbers,
freezing convergence. With clamp-aware STE (`ig_Value.clamped` masks
backprop; also fixes softmax weights computed from a clamped LSE):

| config | final loss (3000 steps) |
|---|---|
| fp oracle | 0.0000 |
| full-quant hybrid (log8 acts + fx logits) | **0.0000** |
| hybrid + pure-int (gscale 2^16) | 0.0000 |
| fp-logits / fp-params variants | 0.0000 |

Deterministic across repeats and seeds; ASan clean.
**Conclusion: the byte-quantization scheme does NOT block optimization.
All previously attributed “quant floors” were the clamp-STE bug.**
(Transparency note: one intermediate measurement run showed the old floor
with the clamp fix supposedly in — suspect a stale build in that sequence;
final numbers are from clean rebuilds and reproducible.)

## E5c — CE-metric artifact found & fixed (invalidates E4/E5 numbers)

While running E6 the depth-2 model "reached 0.09 bpb val" — impossible.
Cause: with lse and target-logit each requantized on the fx grid, both
saturate at the same edge (12.7) and CE = 12.7 − 12.7 = **0 for
confidently WRONG predictions**. Fixed: CE arithmetic is exact (mode 2)
over quantized logits; the CE scalar itself is never quantized (transient,
not stored; clamping it would also kill gradient on very-wrong calls).
*Rule extracted: never clamp two operands of a subtraction where 0 is the
desired outcome — quantify error only at the end of the identity.*
Corrected headline numbers (4000 steps, split-v0):

| config | val bpb (was, artifact) |
|---|---|
| fp oracle (1-layer) | 3.42 |
| hybrid 1-layer | 4.87 (was 4.20) |
| hybrid depth-2 | 4.69 |
| fp oracle depth-2 | 3.65 |

The overfit-probe memorization result (all modes → true 0.0000 CE) stands.

## E5 — Hybrid grid on the real corpus [SUPERSEDED by E5c]

Same protocol as E1 (ctx4/nh96/4000 steps, frozen split):

| config | val bpb | Δ vs oracle |
|---|---|---|
| fp oracle | 3.42 | — |
| pure log8 (E1) | 4.30 | 0.89 |
| **hybrid (D6)** | **4.20** | **0.79** |

Remaining gap interpretation: memorization proves optimization is healthy,
so the residual gap on dense data is representational (grid noise biasing
a capacity-limited model), not an optimizer artifact. Both numbers are
also far above this arch's ceiling (4000 steps, ctx=4 MLP); re-evaluate
the gap at M3 scale before any constraint-level conclusions.

## E6 — Depth vs bits (user hypothesis: gap = capacity, not noise)

Depth-2 (96→64) at the same 8-bit budget: val 4.69 vs 1-layer 4.87 —
depth helps the quantized model (+0.19) and *hurts* the fp oracle
(3.65 vs 3.42), but the quant gap persists (1.0 bpb). Verdict: depth is
worth having, but does not by itself close the gap at this vehicle scale.

The planned "w16" weight-resolution probe was **invalid**: in
(symlog) grids the absolute step near x is ≈ (1+|x|s)·ln2·Δ/s — scale and
delta interact — so Δ/4 at s/256 was actually ~5× coarser near |w|≈0.05
(val 7.9 — consistent, not a mystery). A true capacity probe needs real
16-bit storage (ig_encode16), deferred; the role-grid mechanism
(ig_set_role_grid) is kept — it is the substrate for M4 per-tensor scales.

## Conclusions / next levers (feeds M3.5)

0. **D6 adopted (hybrid grids, clamp-aware STE mandatory).** Log-byte for
   weights/activations; fixed-point byte (step 0.1, ±12.7) for logits/CE
   with one requantization per dot product (int32-style accumulation).
   Every quantizer clamps AND masks backprop at the clamp (E4).
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
