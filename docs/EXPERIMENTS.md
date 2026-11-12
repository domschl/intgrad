# intgrad — Experiment log

Byte-level LM, `ctx=4, hidden=96, batch=4, lr=5e-3, warmup=200, cosine,
4000 steps, seed 42`, evaluated as val bpb on the frozen split-v0
(800 windows; test in parens). Unigram floor: **4.74**.

> **Current defaults (D7, D8):** two-byte weights (16-bit master code,
> the forward pass reads the high byte), low byte round-to-nearest;
> pure-int gradients use a per-tensor pow2 absmax scale (E7 → E8).
> Optimizer **Lion** (lr 2e-3, wd 0.1, batch 32, 16-bit momentum, E9);
> **E1–E8 used Adam** — repro commands below pass `--opt adam`.
>
> **Read E7 first.** Every quantized number in E1–E6 was measured with
> round-to-nearest parameter storage, which discarded ~98% of all weight
> updates. The *interpretations* in E1, E2, E5, E5c and E6 ("forward
> noise", "representational gap", "gscale 2^16 best", "depth helps quant")
> are invalid; the bug-fix findings of E4 (clamp STE) and E5c (CE
> saturation artifact) stand.

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

## E7 — The real bottleneck: lost weight updates (stochastic rounding)

**Isolation that was missing from E4's ladder, run on the real corpus:**

| config (old code, RTN storage) | val bpb |
|---|---|
| fp oracle | 3.470 |
| full quant | 4.875 (worse than unigram 4.74) |
| full quant forward, **float parameter storage** (`--fp-params`) | **3.474** |

Quantizing every forward value costs nothing; the entire gap was
parameter *storage*. Mechanism: the log grid's relative step is 7.2%
(2^0.1; ≈ a 3.8-bit mantissa, ~10× coarser than bf16), so any update
smaller than ~3.6% of |w| rounds back to the same byte. Adam steps are
~lr·m/√v ≈ 0.1% of |w|. Instrumented: **66% of nonzero updates lost at
step 100, 95% at step 300, 98–100% from step 1000 on** — weights only
moved in the brief high-lr window, then froze. Also explains: E1 (scale
moves the window, not the relative step → no effect), E3/E4 (the overfit
probe only memorizes at high lr, e.g. `--lr 3e-2` → 0.10 nats, where steps
clear the grid; at default lr it stalls at 2.64), E2 (gscale "didn't
matter" because the gradient barely reached the weights).

**Fix: stochastic rounding of parameter storage** (`ig_rt2_sr`; the
default for 1-byte storage, `--w1 --no-sr` for ablation; superseded as the
overall default by two-byte weights, E8/D7). Unbiased (E[rt(x)] = x), still one byte per
weight, no float master copy. Seeded xorshift; state is in checkpoint v5
(bit-exact resume preserved). Guarded by ctest `sr_train`.

Companion fixes in the same change:
- `ig_encode_s` no longer floors nonzero values to code 1. In pure-int
  mode that lifted ~1e-10 gradients to ~1e-6 at gscale 2^16, which Adam
  (eps 1e-8) turned into full-size steps (pure-int+SR diverged to 7.7 bpb);
  it also made the zero-collapse statistic structurally always 0.
- Pure-int gradients use a **per-tensor power-of-two absmax scale** chosen
  each step (default; `--gscale F` = old fixed global scale). Saturation
  is 0% by construction. A fixed 2^16 clamps (max |g| ≈ 0.1) and hurts.
- Every layer now accumulates exactly and quantizes once (`ig_requant`,
  clamp-aware STE); previously hidden layers re-rounded each partial sum on
  the log grid (terms < ~3.5% of the sum swamped, order-dependent), and the
  logit fx requant did not set the clamp mask.
- Params are initialized on the "w" grid; checkpoint byte codes use it too.

**Results** (same protocol, 4000 steps; seed 42 / seed 7):

| config | val bpb s42 | val bpb s7 | Δ vs fp |
|---|---|---|---|
| fp oracle | 3.483 | 3.438 | — |
| quant, 1-byte SR (E7 default; superseded by D7) | **3.505** | **3.471** | +0.02 / +0.03 |
| pure-int, SR, dyn gscale | **3.513** | **3.476** | +0.03 / +0.04 |
| quant, `--no-sr` (RTN ablation) | 4.881 | | +1.40 |
| quant, `--scale 64` / `1024` | 3.510 / 3.529 | | |
| pure-int, fixed gscale 4096 / 65536 | 3.510 / 3.684 | | |
| depth-2 (96→64): fp / quant / pure-int | 3.679 / 3.667 / 3.613 | | |

The quant gap is now at seed-noise level (fp alone spans 0.045 across two
seeds) — **already inside the M3.5 gate (≤ 0.15) at this vehicle scale.**
(fp numbers differ by ~0.01 from E5c's binary: float rounding order
changed with the refactor; within noise.) Depth-2 is worse for every mode
at 4000 steps — under-trained, not a quantization effect.

Update statistics (quant, seed 42): subgrid (= would be lost without SR)
70% @100 → 97% @1000 → 99.7% @3000. SR is carrying essentially all
learning after warmup.

Pure-int gradient health (step 3500), count fraction vs **L1-mass
fraction** (`--grad-stats`):

| tensor | dyn absmax: collapse | fixed 2^16: saturation |
|---|---|---|
| W1 | 0% (mass 0%) | 0.04% (mass **3.4%**) |
| b1 | 0% (mass 0%) | 14.6% (mass **10.0%**) |
| W2 | 39.8% (mass **0.012%**) | 0.26% (mass **20.6%**) |
| b2 | 68.8% (mass **0.004%**) | 1.17% (mass **37.4%**) |

Counts mislead in both directions: dyn scaling collapses many
softmax-tail gradients of rare bytes (~1e-7 vs max 0.46) that carry
almost no mass (bpb unaffected: 3.513 vs 3.505), while fixed 2^16
saturates few entries that carry a fifth to a third of the mass (and
would have *passed* a count-based "< 1%" gate). **M5.5 gate redefined
(user decision) to magnitude-weighted: < 2% of L1 mass lost to collapse,
< 1% to saturation, per tensor.** Dyn absmax passes; fixed 2^16 fails.

## E8 — Two-byte weights vs SR variance (long / low-lr runs)

Risk from E7: 1-byte SR is unbiased but injects weight noise
~√(Δ·Σ|update|); it should show up when the model gets good enough to
need weights finer than the 7.2% grid. **Two-byte weights** (`--w2`):
each parameter keeps a 16-bit master log code (sign + M in units of
Δ/256, `ig_Value.wq`); the forward and backward passes read only the high
byte round(M/256) — the ordinary 1-byte code, so forward numerics are
unchanged — and the optimizer update lands in the full 16 bits, i.e. the
low byte is a residual that accumulates sub-grid updates (Kahan-style
master weight in bytes; resolution 0.027% relative; low→high carry is
integer arithmetic on M). Low byte is SR-rounded by default (noise 256×
below 1-byte SR); with `--no-sr` the path is fully deterministic.
Checkpoint v6 stores the codes; bit-exact resume is ctest-guarded in both
modes.

Protocol: as E7 but **20 000 steps** (5× horizon), seed 42, val bpb at
5k / 10k / 15k / 20k. Low-lr arm: lr 1e-3.

| lr 5e-3 | 5k | 10k | 15k | 20k | Δ@20k vs fp |
|---|---|---|---|---|---|
| fp oracle | 3.427 | 3.247 | 3.146 | **3.108** | — |
| 1-byte SR | 3.442 | 3.333 | 3.286 | 3.254 | **+0.146** |
| 1-byte SR, pure-int | 3.513 | 3.325 | 3.238 | 3.210 | +0.102 |
| **w2, SR** | 3.418 | 3.243 | 3.172 | **3.129** | **+0.021** |
| **w2, RTN (deterministic)** | 3.406 | 3.231 | 3.154 | **3.120** | **+0.012** |
| **w2, SR, pure-int** | 3.448 | 3.257 | 3.172 | **3.136** | **+0.028** |
| 1-byte RTN (`--no-sr`) | 4.839 | 4.969 | 5.111 | 5.145 | +2.04 |

| lr 1e-3 | 20k | Δ vs fp |
|---|---|---|
| fp oracle | 3.363 | — |
| 1-byte SR | 3.370 | +0.007 |
| w2, SR | 3.363 | +0.000 |
| w2, RTN | 3.369 | +0.006 |
| w2, SR, pure-int | 3.361 | −0.002 |

**Findings**
1. **The SR floor is real and grows with training:** 1-byte SR's gap goes
   +0.015 → +0.086 → +0.141 → +0.146 over 5k → 20k steps, i.e. it already
   leaves the M3.5 budget (≤ 0.15) on a 120 k-param MLP. E7's "gap
   closed" holds only for short runs. It shows at high lr / long
   horizons (model quality rising), not at low lr where the fp model
   itself is weak (lr 1e-3: all within 0.007).
2. **Two-byte weights remove it:** +0.01–0.03 at 20k for every variant,
   pure-int included — seed-noise level (E7: 0.045 across seeds).
3. **The low byte does not need SR:** deterministic w2 is as good as w2+SR
   (3.120 vs 3.129) — the residual's 0.027% resolution is fine enough,
   which also removes the RNG from the update path if desired.
4. Update stats agree: at step 15k, 1-byte SR "moves" 0.4% of params per
   step, w2 moves 15.4% of master codes (the low byte keeps learning).
5. 1-byte RTN gets *worse* with longer training (4.84 → 5.15), above the
   unigram floor: frozen weights + drifting high-lr updates.

**Decision D7 (2026-11-12):** two-byte weights become the default and
C1 is amended accordingly (PLAN §9); the low byte uses round-to-nearest
by default (finding 3), SR remains available as `--w2-sr`.

Caveat: single seed for the 20k arm (each run ~10 min); differences
≥ 0.1 are well outside the measured seed noise, differences ≤ 0.03 are not.

## E9 — Lion vs Adam (int-friendly optimizer candidate)

Motivation (user): Adam loses its elegance under int scaling (two
moments, one with ~2× the octave range of g, sqrt, division, eps/zero
handling); Lion (Chen et al. 2023) keeps ONE state and updates by
lr·sign(·). Accepted costs: larger batch, re-tuned hyperparameters.

**Implementation** (`--opt lion`): c = β1·m + (1−β1)·g; w −= lr·(sign(c) +
wd·w); m = β2·m + (1−β2)·g, β1 = 0.9, β2 = 0.99; decoupled `--wd` (also
applies to Adam = AdamW). Momentum on a log grid with a per-tensor pow2
absmax scale re-chosen each step: `--lion-m b2` (16-bit codes, RTN;
default when quantized), `b1` (1-byte RTN), `b1sr` (1-byte SR), `f32`.
Int view: sign(c) is a sign test plus one log-code comparison of |m| vs
|g| (offset log(β1/(1−β1))); no sqrt, no division, no eps. Checkpoint v7
stores m; bit-exact resume and quantized-vs-fp are ctest-guarded.

**Protocol:** matched data = matched compute (scalar engine): batch 32 ×
2500 steps (80k windows, = E8's 20k × 4) and batch 128 × 2500 steps
(320k windows); warmup 200, cosine, seed 42, val bpb at 4 points.
Quantized = default stack (two-byte weights, D7).

| batch 32 (80k windows) | lr | val bpb |
|---|---|---|
| Adam fp | 5e-3 / 1e-2 / 2e-2 | 3.226 / 3.109 / **3.099** |
| Adam quant | 5e-3 / 1e-2 | 3.230 / **3.103** |
| Lion fp | 5e-4 / 1e-3 / 2e-3 / 4e-3 | 3.440 / 3.280 / **3.205** / 3.474 (unstable) |
| Lion quant | 1e-3 / 2e-3 / 4e-3 | 3.299 / 3.242 / 3.440 |
| Lion quant, wd 0.1 | 2e-3 / 3e-3 | **3.226** / 3.285 |
| Lion pure-int | 2e-3 | 3.233 |
| Lion m-state b1sr / b1 (RTN) | 2e-3 | 3.252 / 3.563 (freezes, as predicted) |
| Lion fp, batch 4 × 20k steps | 5e-4 | 3.629 (small-batch failure) |

| batch 128 (320k windows) | lr | val bpb |
|---|---|---|
| Adam fp | 1e-2 / 2e-2 | 2.900 / **2.846** |
| Adam quant | 1e-2 / 2e-2 | 2.910 / **2.839** |
| Lion fp | 1e-3 / 2e-3 / 3e-3 | 3.120 / **2.944** / 2.955 |
| Lion quant, wd 0.1 | 2e-3 / 3e-3 | 2.960 / **2.958** |

**Findings**
1. **Lion trails Adam by ~0.10 bpb (fp) / ~0.12 (quantized) at matched
   data, at batch 32 AND 128** — the gap does not shrink with batch in
   this range. Adam reaches Lion's final b128 quality (2.944) after about
   half the data (2.996 at 160k windows, 2.877 at 240k).
2. Lion is quantization-robust: quant gap +0.015–0.04 with a 16-bit
   momentum, pure-int ≈ quant. 1-byte SR momentum costs +0.01–0.03; 1-byte
   RTN momentum freezes (β2 = 0.99 moves m 1%/step < 3.6% half-step).
3. Lion's lr optimum is bracketed (2–3e-3, ≈ Adam's /5–/10); 4e-3 is
   unstable. Adam's optimum is NOT bracketed (2e-2 > 1e-2 at both batch
   sizes) — Adam is if anything under-tuned here, so the gap is not a
   tuning artifact in Adam's favor.
4. Batch 4 is a non-starter for Lion (3.63 vs Adam 3.11).
5. wd 0.1 helps Lion slightly (3.242 → 3.226 at b32).

**Documented Lion hyperparameters (this vehicle):** lr ≈ Adam-lr/5–10
(2–3e-3), β1 0.9 / β2 0.99, wd 0.1, batch ≥ 32, warmup 200, cosine,
m-state 16-bit (b2).

**Decision D8 (user, 2026-11-12): Lion becomes the default optimizer,
Adam remains available (`--opt adam`).** The ≈ 1.5–2× data cost at this
scale is accepted for int-elegance (one state, sign update) and the
ability to scale batch; Lion defaults: lr 2e-3, wd 0.1, batch 32 (each
only if not given explicitly), batch < 32 warns. Caveats: 120 k-param MLP, single seed, Lion's published advantage is on
transformers at much larger batches (≥ 4k sequences), β2 not swept.
Re-run the matched-data comparison at M3 (transformer, realistic batch)
and revisit D8 if the gap stays ≳ 0.1 bpb.

## Conclusions / next levers (supersedes the pre-E7 list)

0. **SR on parameter storage is mandatory** (as PLAN §1 anticipated);
   D6 hybrid grids + clamp-aware STE + single requant per layer stay.
1. **The quant gap is closed at M2 scale for short runs only** (E7);
   over 20k steps 1-byte SR drifts to +0.15 bpb (E8).
2. **Two-byte weights fix that** (+0.01–0.03 at 20k, also in pure-int,
   also without SR). **Adopted as default (D7, C1 amended 2026-11-12)**
   with a round-to-nearest low byte (`--w2-sr` for SR, `--w1` for 1-byte
   ablations). Parameters occupy 2 bytes; everything the network computes
   with stays 1 byte.
3. Pure-int: dynamic per-tensor pow2 absmax scale is the working point;
   no global gscale tuning needed. Adam moments are still float
   (reference mode) — log-byte moments are the next M5.5 item.
4. ~~Re-define the collapse metric~~ done: M5.5 gate is mass-weighted.
5. Watch `upd subgrid/moved` in the step log; any change to weight
   storage must keep ctest `sr_train` green.
6. **Optimizer: Lion is the default (D8)** — int-native single 16-bit
   momentum; accepted cost ~0.1 bpb (≈ 1.5–2× data) vs Adam on this
   vehicle at batch 32–128. Re-measure at M3. Adam (`--opt adam`) keeps
   float moments (reference mode).

## Repro

```sh
# E1–E8 were run with Adam (the default before D8):
F="--opt adam --index data/split-v0/idx.bin --ctx 4 --hidden 96 --batch 4 \
   --lr 5e-3 --warmup 200 --eval-every 2000 --eval-windows 800 --steps 4000 --seed 42"
./build/ig-train $F --no-quant
./build/ig-train $F --scale 1024
./build/ig-train $F --pure-int --gscale 65536
# E7
./build/ig-train $F --w1            # quant, 1-byte weights + SR
./build/ig-train $F --w1 --no-sr    # RTN ablation
./build/ig-train $F --w1 --pure-int # dyn per-tensor gscale (default)
# E8 (B = F with --steps 20000 --eval-every 5000)
./build/ig-train $B                 # two-byte weights, RTN low byte (D7 default)
./build/ig-train $B --w2-sr         # two-byte weights, SR low byte
./build/ig-train $B --w1            # 1-byte SR (drifts +0.15)
./build/ig-train $B --pure-int --w2-sr
# E9 (C = --index ... --ctx 4 --hidden 96 --warmup 200 --eval-windows 800 --seed 42)
./build/ig-train $C --batch 128 --steps 2500 --eval-every 625 --opt adam --lr 2e-2
./build/ig-train $C --batch 128 --steps 2500 --eval-every 625 --opt lion --lr 3e-3 --wd 0.1
./build/ig-train $C --batch 32 --steps 2500 --eval-every 625 --opt lion --lr 2e-3 --lion-m b1sr
```
