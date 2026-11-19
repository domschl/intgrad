# Plan 001 — Reset: byte-level toy LLM with 1-byte compute, 2-byte weights, Lion

*Created 2026-11-17. This is the first plan of a fresh start: earlier
plans, experiment logs and corpora were deleted (they remain only in git
history, up to commit `4ad56b1`). Later plans are numbered `002-…`, `003-…` and
build on this one. A plan is never rewritten after it is closed. Results
are appended to its log (§8), and changes of direction go into the next
plan.*

## 0. What this project is

A **toy LLM** that learns to continue text, with four fixed properties:

1. **Byte-centric.** Input is raw UTF-8 and output is raw UTF-8. Vocabulary
   = 256 byte values. No tokenizer.
2. **Byte numerics.** Everything the network *computes with* (weights as
   read by the forward pass, activations, logits, optionally gradients)
   is one byte. **Parameters are stored as 1 or 2 bytes:** the default is a
   16-bit master code whose high byte is the compute weight. 1-byte
   storage with stochastic rounding stays selectable.
3. **Lion optimizer.** It is the only optimizer. It gets replaced only if
   Lion turns out to be a dead end that cannot be fixed (criteria in M5).
4. **Learnability first.** A scalable decoder-only transformer must
   demonstrably learn (probes plus a scaling ladder) before any recurrence
   work starts. Recurrent state is the subject of plan 002. This plan
   prepares the data path for it.

Out of scope: tool use, teacher models, distillation, optimizers other
than Lion.

**Project direction (user, 2026-11-17).** We cannot train a competitive
model on this hardware, so the goal is a *small* model that trains within
these constraints: a **cognitive core**. The focus is not on storing lots
of knowledge but on learning the essence of reasoning. Two facets will be
explored: **recurrence** and **memory**. The **interface for those
experiments is the latent space in the middle of the transformer stack**
(between blocks L/2−1 and L/2). The M3 model exposes it as `mid_hook`
(read or replace `h_mid` [B, T, d]), and range telemetry includes `h_mid`.
Plan 002 builds on this.

---

## 1. Starting point

### 1.1 Design carried over (hypotheses, not results)

The numerics below come from earlier small-MLP work. Those measurements
were taken with a buggy harness (§1.2) on a 4-byte-context MLP, so they
count only as **design hypotheses. M4 must confirm each one on the
transformer.**

| Design choice | Why it is believed to matter |
|---|---|
| Log-byte grid (symlog, Δ = 0.1) + **clamp-aware STE** | without the clamp mask, optimizers push against saturated values and training stalls |
| **CE exact** over quantized logits, never quantized itself | clamping lse and target on one grid edge scores a confidently wrong prediction as loss 0 |
| Exact accumulation + **one requant per output** | re-rounding partial sums swamps small terms and is order-dependent |
| **Two-byte master weights** (forward reads the high byte) | the byte grid's 7.2% step discards most optimizer updates; 1-byte SR adds noise that grows with training |
| **Lion** with a **16-bit momentum** | one state, sign update, no sqrt/division; a 1-byte RTN momentum freezes (β2 = 0.99 moves it < half a step) |
| Pure-int gradients with a **per-tensor pow2 absmax scale** | saturation impossible by construction |

### 1.2 Defects found and fixed while setting up (M0/M1)

| # | Defect | Impact | Status |
|---|---|---|---|
| B1 | Linear-int8 grid mode made every weight a fresh graph leaf: only biases trained | old "linear int8 is worse" conclusion invalid | code removed; log vs linear re-opened in M4 |
| B2 | Train sampling uniform per document (books: 61% of bytes, 0.67% of picks) | trained mostly on short wiki leads | fixed: length-weighted / byte-uniform |
| B3 | Eval used one window from each of the first val docs in index order (all wiki definitions) | val bpb measured one source | fixed: eval-v1 (§4.4) |
| B4 | `idx.bin` stored absolute shard paths | index breaks if the repo moves | fixed: index v2 |
| B5 | The model never saw a document boundary | needed for sequential streaming / recurrence | fixed: EOD `0x03` in the batchers |
| B6 | Scalar autograd: ~440 windows/s on a 120k-param MLP | cannot train a transformer | D2: PyTorch reference engine |
| B7 | Harness: single seeds, no frozen eval set, unpaired fp runs, ~30 never-retired flags | conclusions flipped repeatedly | rules in §4.5 |
| B8 | Files under `data/` tracked in git | violates the privacy rule | fixed: `data/` fully ignored |
| B9 | Checkpoint format with v1–v7 compat branches | complexity | fixed: single format v8 |
| B10 | Split hashed per document: 30% of val wiki articles had their lead in a train definition | optimistic wiki val bpb | fixed (X2): English wiki grouped by title |
| B11 | 27 k of 50 k Gutenberg books stored 2–4× (encoding variants); exact dedup missed them | books over-weighted, val leakage | fixed (X1): variant selection + near-dup fingerprints |
| B12 | Gutenberg boilerplate in 95% of books | training on license text | fixed (X1): `intgrad.corpus.gutenberg` |

### 1.3 Lessons (why the earlier work ran in circles)

1. The **vehicle could not show the effects** being asked about (no
   attention, no depth, 4 bytes of context).
2. **Harness bugs surfaced only after conclusions were drawn.** No test
   layer checked that a configuration trains at all before comparing it.
3. **No frozen protocol**: eval set, seeds and budgets changed between
   experiment families.
4. **Flags were never retired**, so the configuration space only grew.

The rules in §4.5 address each of these.

### 1.4 State of the code after M0

The C library (`libintgrad`, `ig-train`) is the **numerical oracle**:
scalar autograd with the exact quantization semantics. It is small, exact
and auditable, but it is not the trainer for scalable models. Lion only,
one checkpoint format, `data/` and `runs/` never committed, fixtures in
`tests/fixtures/`.

---

## 2. Numerics specification (the contract both engines implement)

**Canonical form (M2):** every grid is defined by float32 tables computed
once in double, `V(t) = float32(exp2(t·0.1/256) − 1)` with t in 1/256 grid
steps (byte values `V(256c)`, byte thresholds `V(256k − 128)`, 16-bit
values `V(M)`, thresholds `V(k − ½)`). A code is the number of thresholds ≤
|x|·scale, so no transcendental function runs at encode time. Scales are
powers of two (exact), and per-tensor scales are chosen with `frexp`
(exact). C is compiled with `-ffp-contract=off`, so no hidden FMAs. This
makes C and PyTorch bit-identical on CPU and GPU, and it is also the form
an integer kernel would use (threshold table).

| Tensor | Grid | Bytes | Notes |
|---|---|---|---|
| weights (compute view) | symlog Δ=0.1, scale 256 (max ≈ 25.6) | 1 | high byte of the master code |
| weights (storage) | 16-bit symlog, 256 sub-steps per byte step | **2 (default)** or 1 + SR | `w2` / `w1` switch |
| activations (layer outputs) | symlog Δ=0.1 | 1 | one requant per output after exact accumulation |
| logits (output + attention scores) | fixed-point int8 · 0.1 (±12.7) | 1 | absolute error is what CE/softmax feel |
| CE loss | exact | — | never quantized |
| gradients | float (default) or symlog with per-tensor pow2 absmax scale (`pure-int`) | (1) | pure-int = research flag |
| Lion momentum | 16-bit symlog, per-tensor pow2 absmax scale | 2 | |
| STE | 1 in range, **0 where the quantizer clamped** | — | mandatory |

New for the transformer (decided in M3/M4, recorded in the §8 log):
softmax probabilities (candidate: fixed-point u8 in [0,1], step 1/255),
RMSNorm gain/statistics (candidate: exact statistic, requant output),
embeddings (same as weights; tied with the output head).

## 3. Architecture of the code base after plan 001

```
plan/                 numbered plans (this is 001)
docs/DATA.md          corpus pipeline (no data in git)
include/ src/ tests/  C oracle: libintgrad, ig-train, ctest
py/                   uv project `intgrad` (M1: corpus pipeline + data; M2: engine)
  intgrad/corpus/       build_shards, make_split (index v2), gutenberg (cleaner)
  intgrad/data.py       index v2 reader, random + sequential batchers
  intgrad/eval_set.py   eval-v1 freeze, baselines (unigram, KN-5, xz, zstd)
  intgrad/quant.py      grids, STE, w16 master weights   (parity with C)
  intgrad/lion.py       Lion with 16-bit momentum         (parity with C)
  intgrad/models/       mlp.py (parity vehicle), transformer.py (M3)
  intgrad/train.py      single entry point `ig-train`, TOML config + --set overrides
  intgrad/evaluate.py   eval-v1 bpb per source
  intgrad/probes.py     capability probes (P1 …)
  tests/                pytest: unit, parity (ctypes vs build/libintgrad_ref.so), probes
configs/              one TOML per experiment
runs/                 gitignored: <name>-s<seed>/{config.toml, meta.json, metrics.jsonl, ckpt.pt}
data/                 gitignored, PRIVATE
```

Rules: one trainer entry point; every experiment is a config file; a flag
that exists only for one experiment is deleted when the experiment closes.

---

## 4. Test methodology

Five layers, cheapest first. A configuration may only be *compared*
(layer 4) after it has passed layers 1–3.

### 4.1 Layer 1 — unit tests (`ctest`, `pytest`, < 1 min)
Quantizer round-trip and monotonicity, clamp masks, w16 forward-view ==
1-byte grid, SR unbiasedness, Lion single-step against a hand-computed
value, sampler length-weighting, batcher continuity (§5 M1), checkpoint
bit-exact resume.

### 4.2 Layer 2 — parity, C oracle ⇄ PyTorch (`pytest`, < 1 min)
`py/tests/test_parity.py` calls the C oracle directly through ctypes
(`build/libintgrad_ref.so` + `ig/flat.h`), on CPU and CUDA. Bit-exact:
the tables, codes over a sweep of every threshold ±3 ulp at four scales
plus random magnitudes, SR, 16-bit encode/decode, the high-byte view, the
fixed-point grid, `pow2_scale`, and one full Lion step in six modes
(float, w2, w2-SR, w1-SR, w1-RTN, pure-int) including master codes and
momentum. MLP forward logits are also bit-exact; loss and gradients agree
to float tolerance (C `expf`/`logf` vs torch's). Mutation-checked: the
"natural" float constant 0.1 for (1 − 0.9f), half-to-even rounding in
the fixed-point grid, and a log2-based encoder each make these tests fail.

### 4.3 Layer 3 — capability probes (deterministic, minutes on GPU)
Each probe has a pass threshold and runs in fp *and* quantized mode:

| Probe | Data | Pass | Proves |
|---|---|---|---|
| P1 memorize | `tests/fixtures/overfit.txt` | CE < 0.01 nats | optimization path works |
| P2 copy | synthetic: random bytes, separator, repeat | ≥ 99% byte accuracy on the copy half | attention routes information |
| P3 associative recall | synthetic `k1=v1;k2=v2;…;k3?` | ≥ 95% | content-based lookup in context |
| P4 UTF-8 well-formedness | 32 × 1 KB samples at T=0.8 | ≥ 99% valid UTF-8 | byte model learned encoding structure |
| P5 learns from data | 5 min real-data run | val bpb < unigram − 1.0 | the full stack, end to end |

P2/P3 are the critical additions: they fail for any model without working
attention, so they catch the "trains but doesn't really learn" bug class
(B1-style) in minutes.

### 4.4 Layer 4 — benchmark protocol (eval-v1)
- **Frozen eval set**: 512 chunks × 2 KB from val (length-weighted,
  fixed seed), stored as `data/eval-v1/val.bin` with its blake2b hash
  recorded in §7. Same for test (touched only at plan close).
- Report **bpb ± standard error** over chunks, plus a **per-source**
  breakdown: wiki-en, wiki-de, wiki-defs, books.
- **Baselines** on the same set: unigram, byte 5-gram (Kneser-Ney, M1),
  `xz -9` and `zstd -19` compressed size. A model that doesn't beat xz
  isn't a language model yet.
- **Scaling ladder** (fixed bytes seen): XS/S/M/L (§5 M3). Monotone bpb in
  size is a correctness check, not a result.
- **Position curve**: bpb as a function of byte offset within the context
  window (0…ctx). It is the yardstick for recurrence in plan 002.

### 4.5 Layer 5 — experiment rules
1. Every experiment has a written **question, hypothesis and the decision
   it informs** (in §8) *before* it runs.
2. Every quantized run has an **fp twin** (same seed, data order, steps).
3. A difference < 0.05 bpb needs **≥ 2 seeds** before it is claimed.
4. A run directory holds config, git hash, eval-set hash and logs, and is
   reproducible with one command.
5. A result that later turns out to be produced by a bug is **marked
   invalid in place** (strike-through + pointer), never silently edited.

---

## 5. Milestones

### M0 — Cleanup ✅ (this commit)
See §1.4. Acceptance: `ctest` green (5 tests), ASan build green, no tracked
file under `data/`.

### M1 — Data path: index v2, batchers, frozen eval ✅
Done 2026-11-17. All acceptance checks pass (pytest `test_data.py`,
throughput random 134 / sequential 245 MB/s). Added along the way:
Gutenberg cleanup + dedup (B11/B12, X1), corpus-v2, title-grouped split
(B10, X2). Deviation from item 2: random windows are drawn from the
bucket's *virtual stream* (docs joined by EOD) instead of within one doc.
That is still byte-uniform, and documents shorter than the context
(wiki definitions average ~700 B) are no longer excluded.

1. **Index v2** (`make_split.py`): paths relative to the index file, a
   `u8 source` field per doc (wiki-en / wiki-de / wiki-defs / books /
   other), and docs kept in corpus order (needed for sequential mode).
   Reader in C (`dataset.c`) and Python (`data.py`, numpy memmap).
2. **Random batcher** (existing semantics, now length-weighted):
   `B × (T+1)` windows, each from a length-weighted doc at a uniform
   offset. Used for all of plan 001's training.
3. **Sequential batcher** (prepares recurrence): `B` independent
   *streams*. Stream `b` owns a deterministic shuffled sequence of train
   docs and reads them **consecutively**: batch `t` row `b` is the
   continuation of batch `t−1` row `b`. At a document end it emits the
   **EOD byte** (D7) and continues with the stream's next doc. Each step
   also returns a per-row `doc_start` mask (offsets where a new doc
   began), so a recurrent model can choose to reset its state there.
   State = per-stream (permutation seed, doc cursor, byte offset), stored
   in checkpoints for bit-exact resume. Because every stream reads whole
   documents, byte coverage is uniform by construction (no length
   weighting needed).
4. **Frozen eval-v1** + baselines (unigram, 5-gram KN, xz, zstd).
5. Near-dup check wiki-defs ⇄ wiki articles (B10). Low priority. If leakage
   is > 1% of val, rebuild the split with title-based grouping.

Acceptance: pytest checks that (a) concatenating stream `b`'s rows over
`t` reproduces its docs byte-for-byte with EOD separators, (b) no
val/test byte ever appears in train batches, (c) resume after a checkpoint
continues the exact same byte stream, (d) random-mode byte histogram
matches the length weighting (χ² test) and (e) throughput ≥ 100 MB/s from
page cache.

### M2 — PyTorch reference engine + parity ✅
Done 2026-11-17: 59 pytest (parity, unit, probes P1 in 4 modes, resume)
and 5 ctest green; X4 shows the gap reproduced across engines; P5 3.44 bpb
after 33 s. Deviations from the items below: parity runs through ctypes
against the C library instead of golden files (stronger: arbitrary
inputs), and master codes live in Lion's state (`state["wq"]`) instead of
a `W16Param` class (fewer moving parts). The C quantizer was made
table-canonical (§2) so that bit-exact parity is possible. Finding X5:
saturation is absorbing under the clamp-masked STE.

1. `py/` uv project (`torch` CUDA wheel for the RTX 4070 Ti, `numpy`,
   `pytest`). Run through `uv run`.
2. `quant.py`: symlog encode/decode, fake-quant with clamp-aware STE
   (custom `autograd.Function`), fx grid, `W16Param` (int16 master codes;
   forward view = high byte; optional 1-byte+SR storage).
3. `lion.py`: Lion over `W16Param`, 16-bit momentum with per-tensor pow2
   scale, decoupled wd, optional pure-int gradient quantization.
4. `ig-golden` (C) + parity tests (§4.2).
5. Port the MLP and reproduce the oracle: same 20k-step protocol, fp vs
   w2 gap within ±0.03 of the C engine's.

Acceptance: parity tests green; MLP gap reproduced; probes P1 + P5 pass.

### M3 — Scalable test transformer (fp first, no recurrence) ✅
Done 2026-11-17. Acceptance: P2 and P3 pass at XS and S; P4 32/32 valid
UTF-8 at all sizes; the ladder is monotone (X10); S beats xz and KN-5;
throughput recorded (S 477 kB/s compiled, above the 300 kB/s estimate).
Outcomes: Lion β (0.95, 0.98), lr 3e-3, wd 3 (X7, X8, X11); q/k/v causal
depthwise conv kept, conv stem and deeper/narrower dropped (X9);
`torch.compile` and micro-batching in the trainer; mid-stack latent
hook (`mid_hook`). Inputs for M4: logit and attention-score ranges
(X10), Lion's weight bound 1/wd, C oracle betas (X7).
Decoder-only, pre-norm (RMSNorm), causal self-attention, RoPE, MLP with
ReLU² (cheap as a LUT; SwiGLU is an ablation), tied 256 × d byte
embedding / output head, no biases. One config scales it:

| preset | d | L | heads | ctx | params (≈) |
|---|---|---|---|---|---|
| XS | 128 | 2 | 4 | 256 | 0.4 M |
| S | 256 | 4 | 4 | 512 | 3.2 M |
| M | 384 | 6 | 6 | 1024 | 10.7 M |
| L | 512 | 8 | 8 | 1024 | 25 M |

Training: random batcher, Lion (start: lr 2e-3, wd 0.1, β 0.9/0.99),
warmup + cosine, fp32 master + TF32 matmuls in fp mode. Eval on eval-v1
every N steps. Samples are written to `runs/` only (D5).

**Magnitude control (001-D9, from X5/X6).** Lion grows weights and
activations without bound when gradients keep their sign, and under byte
numerics clamped values get no gradient. M3 therefore builds in, from the
first run:
- pre-RMSNorm everywhere, **QK-norm** (RMSNorm on q and k: bounded
  attention scores);
- **z-loss** on the output logits (1e-4 · lse², keeps the normalizer near 0);
- an optional **logit soft-cap** `c·tanh(x/c)` with c below the fixed-point
  clamp (gradient stays alive). It is off by default and gets switched on
  if telemetry shows logit clamping;
- **range telemetry** per tensor class (embeddings, residual stream per
  layer, normalized activations, q/k/v, attention scores, MLP pre/post
  activation, logits, weights): absmax and |x| percentiles in octaves,
  plus the rates that *would* clamp (> 26.0) or underflow (< 1.4e-4) on
  the global byte grid. This is logged in fp runs too, so M4 knows which
  tensors need their own scale before quantizing anything;
- an early **Lion lr × wd sweep** on XS (3 × 3, P1/P2 and 2k-step bpb),
  pulled forward from M5.

**Architecture variants (001-D11, data decides).** Run at ≈ equal
parameter count on S (2 seeds each), scored on eval-v1 overall and per
source (German wiki has the most multi-byte characters), with probes
P2/P3 and the range telemetry:
- (a) plain tied embedding (baseline);
- (b) **conv stem**: 256 × 32 byte embedding → causal conv 32 → d
  (kernel 4) → residual causal depthwise conv (kernel 4, dilation 2),
  receptive field 7 bytes; untied output head d → 256;
- (c) **Primer-style**: causal depthwise conv (kernel 3) on q, k and v in
  every block;
- (d) **width vs depth**: d 256 × 4 layers vs d 192 × 7 layers (H 4 / 3,
  head width 64 in both; equal 12·L·d² budget).

All convolutions are causal: position t sees bytes ≤ t only. Strided
(patching) and hourglass variants stay plan-002 candidates.

**P3 is multi-query associative recall (MQAR).** The single-query version
was not learned within 10k steps by either Lion or AdamW (29% on 8 pairs,
X7); with several queries per sequence the supervision is dense enough.

Acceptance (fp mode): P2, P3, P4 pass at XS/S; the ladder is monotone at a
fixed byte budget; S beats xz and the 5-gram on eval-v1; throughput
measured and recorded (estimate: S ≥ 300 kB/s on the 4070 Ti).

### M4 — Quantized transformer (the core result of plan 001) ✅
Outcome: full byte numerics (stage d) pass the gate at S (+0.013 bpb)
and M (−0.004), and the probes pass quantized (X13, X14). Weight storage
stays w2 (X15). Softmax inputs are max-relative (D12). Activation scales
use delayed absmax; learned scales (D10) and the linear-int8 re-test were
dropped by user decision after X13 showed no range pressure on the log
grid.

**Status (started 2026-11-17).** Built:
- `quant.acts` / `resid` / `softmax` / `probs` switches. `quant.enabled` is
  the weight storage (Lion 16-bit master codes, forward = high byte).
- `QSite` activation sites with a delayed pow2 scale: amax history of 16
  training forwards × 2 headroom. This keeps the forward causal and fixes
  the scale at eval.
- Explicit attention whenever scores or probabilities are quantized.
- `rt8_fast`: exact byte rounding without searchsorted.
- C oracle: configurable Lion betas and `ig_fx_maxrel`, both bit-exact.

Throughput at S: full quantization runs 2.1× slower than fp.

Schedule (GPU time):

| phase | runs | time |
|---|---|---|
| A: stages a–e at S | 2 seeds | ≈ 2 h |
| B: learned scales vs delayed absmax (D10) | — | ≈ 1.5 h |
| C: w1-SR vs w2, log vs linear int8 | — | ≈ 1 h |
| D: gate at S and M, ladder horizon | quantized M ≈ 80 min per run | ≈ 4 h |
| probes P1–P4, quantized | — | ≈ 0.5 h |
| sensitivity sweep, only if the gate fails | — | + 2–3 h |

**Per-tensor scaling (from the M3 telemetry).** So far only gradients and
Lion momentum have per-tensor scales; all weights and activations share
one global window (clamp ±26.0, underflow < 1.4e-4). Before (a) below,
introduce per-tensor power-of-two scales where the telemetry demands it:
weights per tensor (per output channel if their rows differ by octaves),
activations by delayed absmax (amax history over recent steps, ~1 octave
of headroom, so a kernel needs no extra pass), and max-relative
unsigned bytes for attention scores and logits (001-D12, X12). Decide explicitly how
the residual stream is stored (byte grid with its own scale, or wider).
Master weights re-encode when their scale changes (rare). Power-of-two
scales are within the parity-verified contract (§4.2 tests 2⁻¹⁰ … 2²⁰).

**Learned scales (001-D10).** Compare, for activations, attention
scores and logits:
- (a) delayed-absmax pow2 scales (above);
- (b) **learned log₂ scales** rounded to a power of two in the forward pass
  (TQT, Jain et al. 2020), initialized from (a). The scale gets gradient
  LSQ-style (Esser et al. 2020): clamped elements push it up, underflowing
  and linear-region elements push it down. This gives clamped values a
  path back, which the clamp-masked STE alone does not (X5/X6). It is
  trained by Lion in its own param group (own lr, no wd).

Oscillation at power-of-two boundaries is handled three ways:
- (b1) Lion's own momentum only;
- (b2) an EMA of the latent log₂ s, rounded for the forward pass
  (momentum on the value);
- (b3) hysteresis.

Expectation to test: on the symlog grid an octave is exactly 10 codes, so
a factor-2 flip leaves in-range grid points in place and only moves the
clamp/underflow edges (and the near-zero linear region). Flips should
then be cheap, unlike on linear int8 grids, and (b1) may suffice.
Metrics: bpb vs the fp twin, clamp/underflow rates over time, and **scale
flips per 1k steps** per tensor. Weight scales stay statistics-based
unless (b) clearly wins, because their 2-byte master codes would need
re-encoding on every scale change (alternative to test then: master
codes at a fixed canonical scale, learned scale on the forward view only).

Turn on §2 numerics layer by layer, each step with its fp twin:
(a) weights w2, (b) + activations, (c) + logits/attention scores max-relative
(001-D12), (d) + softmax probabilities, (e) + pure-int gradients. Then the
ablations that were open or invalid: w1-SR vs w2 storage; log vs
*properly implemented* linear-int8 weight grid (re-run of B1).

Gate: full-quant (a–d) within **0.10 bpb** of fp at S and M, and P1–P4
pass quantized. If the gate fails, run a **sensitivity sweep** (quantize
one tensor class at a time) before touching anything else.

### M5 — Lion at transformer scale ✅
Outcome: Lion is not a dead end. Neither criterion holds: fp and
quantized Lion are stable up to L, and the quantization cost is ≤ 0.013
bpb at S/M/L (X14, X19). The tuned defaults are unchanged (X16–X18):
lr 3e-3, wd 3, betas (0.95, 0.98), batch 64, one lr for all groups.
The controlling quantity is lr·wd ≈ 6–9e-3.

Sweep lr × wd × β2 × batch on S (2 seeds at the optimum), per-group lr
for embeddings/norms, and check stability at M/L. Lion counts as a
**dead end** only if, after this sweep, (i) fp-Lion diverges or stalls at
M/L, or (ii) quantized-Lion cannot reach the M4 gate while fp-Lion can.
Only then does a plan 00x consider another optimizer.

**Status (started 2026-11-18).** Neither dead-end criterion holds
going in: fp-Lion is stable up to L (X10), and quantized Lion passes the
gate (X14). M5 therefore tunes Lion **for the quantized model** (stage
d, the target numerics) and checks stability at L.
- Round 1, screening: S, 4000 steps, seed 42. Grid lr {1.5, 3, 6}e-3 ×
  wd {1, 3} × β2 {0.98, 0.99}, with β1 0.95 (X7).
- Round 2, at the round-1 best:
  - per-group lr multipliers for the embedding/head (`train.lr_emb`) and
    the gains (`train.lr_vec`), {0.3, 3} each;
  - batch {32, 128} at a constant byte budget.
- Round 3: best vs the current default at the ladder horizon, 2 seeds.
- Round 4: quantized L at the ladder horizon (stability), plus M if the
  defaults changed.

Estimated GPU time ≈ 6–7 h.

### M6 — Hourglass: sequence compression before the transformer (user proposal, 2026-11-18)
Idea: a byte-level encoder compresses the stream by k before the main
transformer; a byte-level decoder expands it again before the logits.
The core then reasons over syllable-sized units, and its compute per byte
drops: attention by ~k², the rest by ~k, or the same compute buys a
larger core. The mid-stack latent, the interface for plan 002, becomes
patch-level.

Prior work: Hourglass (Nawrot et al. 2021), CANINE (Clark et al.
2021), MEGABYTE (Yu et al. 2023), SpaceByte (Slagle 2024), BLT (Pagnoni
et al. 2024: entropy-based patches), H-Net (Hwang, Wang, Gu 2025:
chunking learned end to end).

Design (`py/intgrad/models/hourglass.py`):
- emb (256 × d_b) → 2 byte blocks (causal depthwise conv k ≥ max(4, k) +
  ReLU² MLP) → concatenate k bytes → Linear(k·d_b → d) → global Blocks
  (unchanged: RoPE, QK-norm, q/k/v conv, QAT sites).
- Then Linear(d → k·d_b), **shifted right by k−1 bytes**, so patch j
  only reaches positions ≥ jk+k−1. This is mutation-checked: without the
  shift, the causality tests fail.
- Then + the encoder output (byte skip) → 2 byte blocks → tied head.
- The decoder is autoregressive per byte: bytes inside the current patch
  come through the skip and the causal convs, and only completed patches
  through the global path.

Comparison: the same 32 KB per step and 300 MB as the m3-ladder (fp), so
bpb on eval-v1 sits directly against the flat models. The axis that
matters is **bpb vs training time**. The context is T = 2048 bytes for
k = 4 (eval-v1 chunks are 2048 bytes, so longer T cannot pay off there).

Rounds:
1. S-k2, S-k4, M-k4, L-k4, fp.
2. If it pays off: probes, especially P3 MQAR (recall must survive the
   compression), and a quantization re-check of the new parts.
3. Then patch boundaries at word / UTF-8 boundaries (SpaceByte-style).

Learned chunking (H-Net) stays a later research item. The choice made
here is the architecture M7 measures and plan 002 builds on.

**Patch boundaries are language-agnostic (user decision, 2026-11-18).**
No word or space boundaries: they do not carry over to many scripts, and
they reintroduce tokenizer-like decisions. The only structural rule is
**never split a UTF-8 character** (`model.patching`):
- `utf8`: the k-byte grid, with each patch end snapped forward to the next
  character end. ASCII gives exactly the fixed patches; there are at most
  T/k patches.
- `chars`: k complete characters per patch, which evens out information
  per patch across scripts.

"Character complete" is decided from past bytes only: the lead byte
announces the length. Stray continuation bytes (a window starting
mid-character, broken text) count as 1-byte characters. The patch vector
is built from the last k encoder states up to the patch end, and each later
byte gets its own upsample vector by offset from that end (clipped at
k−1). So the path has the same width as fixed patching and is identical to
it on ASCII (tested). Byte attention is masked to the patch segment.
**Default after X24: `patching = "utf8"`.** Causality is tested at every split point and
mutation-checked: a decoder that sees a patch one byte early fails.
Learned or entropy boundaries (BLT/H-Net) remain later options, still
constrained to character ends.

### M7 — Recurrence readiness (hand-off to plan 002)
Using the sequential batcher, measure the non-recurrent model's
**position curve** and **cross-window loss**: the bpb of the first 64 bytes
of each window when the preceding window is in the same doc, vs a cold
start. This is the baseline any recurrent state in plan 002 must beat.
Deliverables: metric implemented, baseline numbers recorded,
`plan/002-recurrent-state.md` drafted.

**Status (2026-11-19).** Base: tuned flat L, `configs/m7-base.toml` (X31,
X32); the hourglass is shelved (X30). Metric: `ig-position` (tested,
mutation-checked). Baseline: X33. Plan 002: drafted.

---

## 6. Decisions (plan 001)

- **001-D1 Scope**: toy byte-level LM. Tool use, teacher and distillation
  are out of scope.
- **001-D2 Engines**: PyTorch reference for anything scalable. The C scalar
  engine is the numerical oracle, bound to PyTorch by parity tests. C/CUDA
  int kernels are a later plan.
- **001-D3 Numerics**: §2: hybrid grids, clamp STE, exact CE, 2-byte master weights
  with 1-byte compute (§1.1, to be confirmed in M4). 1-byte storage
  + SR remains selectable.
- **001-D4 Optimizer**: Lion only. Replaced only on
  the M5 dead-end criteria.
- **001-D5 Data privacy**: `data/` and `runs/` are never committed. No
  corpus excerpts or model samples in docs, plans or commit messages.
  Fixtures written for this repo live in `tests/fixtures/`. Metadata
  (counts, sizes, hashes, aggregate bpb) is fine.
- **001-D6 Sampling**: length-weighted (every byte equally likely) for
  random batches, train and eval (B2/B3).
- **001-D7 Special bytes**: the corpus contains no C0 controls except
  `\t`/`\n` (enforced by `build_shards.py`), so **`0x03` (ETX) = end of
  document** and **`0x00` = padding (loss-masked)**. These are the only two
  reserved bytes for now.
- **001-D8 Eval protocol**: eval-v1 (§4.4) is the only number that goes
  into result tables. Test is evaluated only at plan close.
- **001-D11 Architecture variants are decided by data (user decision).**
  Conv stem, Primer-style q/k/v convolutions and width vs depth run as M3
  comparisons (§5 M3). We keep what measurably helps. Outcome (X9): keep
  the q/k/v conv (kernel 3); drop the conv stem (redundant with it) and
  the deeper/narrower shape.
- **001-D10 Learned quantization scales are an M4 experiment (user
  decision).** TQT/LSQ-style learned pow2 scales vs delayed absmax, with
  oscillation handled by Lion momentum, latent EMA or hysteresis
  (§5 M4). Measured, not assumed. *Closed without the experiment (user
  decision, 2026-11-18).* X13 found no clamps and negligible underflow
  at delayed-absmax scales on the log grid, so a learned scale has
  nothing to trade there. The linear-int8 re-test, where it would
  matter, is skipped too.
- **001-D9 Lion stays; magnitudes are controlled by design (user
  decision after X6).** The right lr works, and M3 builds in QK-norm,
  z-loss, an optional soft-cap, range telemetry and an early lr × wd
  sweep. There is no AdamW diagnostic twin in M3; that idea is revisited
  only if we get stuck or data points at a shortcoming of Lion.
- **001-D12 Softmax inputs are stored max-relative (X12).** Attention
  scores and logits are stored as row max − value in an unsigned byte
  ([−25.5, 0] at step 0.1); only entries with p < 1e-11 clamp. Gains
  and q/k/v conv kernels stay without decay: bounding them costs bpb.
  The decay anchor (`train.norm_wd`, `train.conv_decay init`) remains as
  an option.

## 7. Pinned artifacts

| Artifact | Value |
|---|---|
| corpora | `data/corpus-v2` (books cleaned + deduped, wiki-2017 defs, text; 21.5 GB) + `data/corpus-v1-wiki` (18.6 GB) |
| split | `data/split-v1`, index v2 blake2b `885d7ce042173809b349ba1f5bdd36fa`; 1‰ val / 1‰ test, English wiki grouped by title; 10.81 M train docs, 40.0 GB |
| train byte shares | books 46.5%, wiki-en 32.5%, wiki-de 14.0%, wiki-defs 6.9%, text 0.1% |
| eval-v1 | 512 × 2048 B; val blake2b `035bcfa09a4e8bc6930299822f1590a6` (books 261, wiki-en 127, wiki-de 82, wiki-defs 16, text 26 chunks); test `7572ca3a43221abc55301df07c8c8018` |

Baselines on eval-v1 val (bits/byte over positions 1..2047 of each chunk, ± SE over chunks):

| baseline | all | books | wiki-en | wiki-de | wiki-defs |
|---|---|---|---|---|---|
| unigram (100 MB train sample) | 4.738 ± 0.014 | 4.585 | 4.806 | 5.078 | 4.894 |
| **Kneser-Ney 5-gram** (100 MB train sample) | **2.415 ± 0.014** | 2.366 | 2.376 | 2.626 | 2.417 |
| xz -9e (whole 1 MB set) | 2.897 | | | | |
| zstd -19 (whole set) | 2.969 | | | | |
| zstd -19 + 110 KB train dictionary, per chunk | 3.229 | | | | |

The bar for M3 is the KN-5 line: a transformer that doesn't clearly beat
2.42 bpb has not learned more than local byte statistics.

## 8. Experiment log (plan 001)

Format: `X<n>` id, question, hypothesis, decision informed, config,
result (± SE, seeds), verdict.

**X1 — Gutenberg cleanup precision/recall (2026-11-17).** Q: can PG
boilerplate be stripped with high precision without touching the book?
H: markers + zone-restricted paragraph scoring + a 30% guard suffice.
Decision: what goes into corpus-v2. Method: 3 × 400 random books after
variant selection; manual review of every class of removed paragraph, plus
residual PG-specific mentions. Iterations fixed: weak-signal double
counting, line-anchored "Produced by" matching prose, a tail note rule
that ate end-notes, and title paragraphs taken by the sandwich rule.
Result: no reviewed removal was book text, except one book about digital
libraries that loses tail paragraphs (accepted). Residual PG text in
0.25–0.5% of books (was 95%). ≈ 4.5% of book bytes removed; guard fired
30× in the full build. Near-dup fingerprint: 58/60 variant pairs found,
0/300 false positives. Verdict: adopted (corpus-v2).

**X2 — Wiki definitions ⇄ articles leakage (B10, 2026-11-17).** Q: does
the per-document hash split leak wiki leads across buckets? Method: the
share of held-out docs whose normalized lead (letters 0–48 / 0–24 /
100–148) matches a train doc of the other wiki source. Result: per-doc
hashing 30.2% / 22.8% (articles / definitions). Lead-text grouping gave 0%
by construction, but 14% on letters 100–148 because of first-sentence
edits. **Title grouping: 0.5% / 0.4% (0–48), 1.2% / 1.7% (0–24).** The
remaining 100–148 matches are definitions too short for the key plus
templated stubs; ≤ 0.7% are candidate true leaks. Verdict: title grouping
adopted (split-v1).

**X3 — eval-v1 baselines (2026-11-17).** See §7. A bug found on the way
(the n-gram train sample was 4 MB, not 100 MB, because the loop compared
bytes with a window count) was fixed before recording. An assert guards
it now.

**X4 — MLP vehicle, C oracle vs PyTorch (M2 acceptance, 2026-11-17).**
Q: does the PyTorch engine reproduce the oracle's training behaviour, not
just single steps? Decision: whether PyTorch can replace C for training.
Protocol: ctx 4, hidden 96, batch 32, Lion lr 2e-3 wd 0.1, warmup 200,
cosine, 4000 steps (128 k windows), seeds 42 and 7, fp and quantized (w2).
Each engine uses its own sampler stream from split-v1. All models are
scored by the same evaluator on eval-v1 val (C checkpoints loaded into the
bit-exact PyTorch forward). `configs/x4-mlp.toml`; raw results in
`runs/x4/summary.json`.

| engine | fp (s42 / s7) | quant w2 (s42 / s7) | gap q − fp (mean) | time/run |
|---|---|---|---|---|
| C oracle | 3.454 / 3.462 | 3.441 / 3.428 | −0.023 | 300 s (CPU) |
| PyTorch | 3.457 / 3.487 | 3.437 / 3.448 | −0.029 | 33 s (GPU) |

SE per number ≈ 0.02 (chunk level). Verdict: gaps agree within 0.006
(bound ±0.03), and absolute values agree within seed spread. **PyTorch
replaces C for all training from here on.** Both engines show quantized
≤ fp at this scale; with 2 seeds this is not a claim, only a note for M4.

**X5 — Saturation is absorbing (P1 probe, 2026-11-17).** Memorizing
`overfit.txt` (ctx 8, hidden 128, 1500 full-batch steps): fp reaches the
floor H(next | ctx) = 0.0324 nats at lr 1e-2, but every quantized mode
stalls (w2 0.18, w1-SR 0.14, pure-int 0.21; 0.43 after 6000 steps). At
lr 1e-2, 99.99% of logit accumulators clamp at ±12.7 (max 14.8 k) and 29%
of hidden pre-activations clamp at 25.6. Windows whose logits all pinned
to the same edge score ln 256. The clamp-masked STE gives clamped values
zero gradient, so nothing pulls them back. At lr ≤ 3e-3 all quantized
modes reach the floor exactly. Consequence for M4: watch clamp rates per
tensor class; levers are lr, a weak restoring gradient through the clamp
("leaky" STE), and range control (logit scale, normalization). The probe
runs at lr 3e-3.

**X6 — Is the saturation a Lion property? (2026-11-17).** Same probe
as X5 at lr 1e-2, quantized forward, *float* storage, varying the
optimizer. "Max |logit acc|" is the largest logit before the clamp.

| optimizer | lr | wd | CE (floor 0.0324) | hidden clamped | max \|logit acc\| |
|---|---|---|---|---|---|
| Lion | 1e-2 | 0 / 0.1 / 1.0 | 0.108 / 0.046 / 0.035 | 33% / 18% / 0% | 14.9 k / 9.4 k / 507 |
| Lion | 3e-3 | 0 | 0.0324 | 0% | 1.7 k |
| AdamW | 1e-2 / 3e-2 / 1e-1 | 0 | 0.0324 | 0% | 16 / 18 / 31 |
| SGD + momentum | 0.3 / 1.0 | 0 | 0.035 / 0.036 | 0% | 16 / 17 |
| Lion, no quantization | 1e-2 | 0 | 0.0324 | (42% beyond 26) | 33.8 k |

Verdict: the magnitude growth is a Lion property. The sign update keeps
full size while gradients vanish, so only weight decay bounds it, and
magnitudes grow 30–500× beyond AdamW/SGD. The damage comes from
quantization: hidden values beyond the fixed ±26 window die under the
clamp-masked STE. Clamped logits are mostly harmless, since the target
pins at the top and the rest at the bottom. Countermeasures: 001-D9, M3
magnitude control, M4 per-tensor scaling.

**X7 — Associative recall: Lion hyperparameters, not Lion (2026-11-17).**
Q: why does the recall probe P3 fail? XS transformer, fp, trained from
scratch on the probe task, greedy accuracy on fresh sequences.
- Single-query recall (8 pairs, one supervised answer per sequence):
  Lion 22–30% and AdamW 26–30%, even after 10k steps at batch 256–4096.
  With 2 pairs AdamW reaches 100% and Lion 55–76%. Verdict: the probe was
  too sparse to be a useful test → replaced by MQAR (all keys queried).
- MQAR, 8 pairs: Lion with the inherited β (0.9, 0.99) reaches 26–46% at
  lr 1e-4 … 3e-3, wd 0 or 0.1, batch 128 or 512, up to 20k steps.
  **AdamW lr 1e-3: 100% in 5k steps.** That is the trigger for revisiting
  Lion (001-D9), so Lion got its paper's recommended operating point:

| Lion (5k steps) | β | batch | MQAR acc |
|---|---|---|---|
| lr 1e-4 / 3e-4 | 0.9, 0.99 | 128 | 0.45 / 0.46 |
| lr 1e-4 / 3e-4 | **0.95, 0.98** | 128 | **0.76** / 0.46 |
| lr 3e-4 | 0.9, 0.99 | 512 | 0.46 |
| **lr 1e-4** | **0.95, 0.98** | **512** | **1.00** |

Verdict: the inherited β2 = 0.99 is too slow for abrupt, sparse learning
signals, and Lion needs a larger effective batch than AdamW. Lion is not
a dead end. P3 now runs at lr 1e-4, β (0.95, 0.98), batch 512, 5k steps.
The transformer defaults follow from the corpus lr × wd × β sweep (M3).
Note for M4: the C oracle hard-codes β (0.9, 0.99); it needs
configurable betas before quantized parity runs at other values.

**X8 — Lion sweep on the corpus, XS (2026-11-17).** XS (0.43 M), fp,
2,000 steps × 128 × 256 B (65 MB), warmup 200, cosine; eval-v1 val bpb
(chunk SE ≈ 0.013–0.015; seed-to-seed spread up to 0.05 at this horizon).
`configs/m3-xs-sweep.toml`.

| β | lr 3e-4 wd 0.1 / 1 | lr 1e-3 wd 0.1 / 1 / 3 / 10 | lr 3e-3 wd 0.1 / 1 / 3 / 10 | lr 1e-2 wd 1 / 3 |
|---|---|---|---|---|
| 0.9, 0.99 | 2.910 / 2.865 | 2.935 / 2.805 / – / – | 2.910 / 2.557 / – / – | – |
| 0.95, 0.98 | 2.489 / 2.460 | 2.654 / 2.518 (s7 2.479) / 2.313 (s7 2.302) / 2.217 | 2.666 / 2.398 (s7 2.344) / **2.222 (s7 2.219)** / 2.239 | 2.254 / 2.228 |

Verdict: β (0.95, 0.98) wins in every cell (X7 confirmed on the real
task); weight decay must be far stronger than the inherited 0.1. The
optimum is a plateau at lr·wd ≈ 0.01–0.03 (with RMSNorm the network is
largely scale-invariant, so wd also raises the effective step size).
**Transformer defaults: lr 3e-3, wd 3.0, β (0.95, 0.98)**, stable over 2
seeds. XS beats the KN-5 baseline (2.415) by 0.19 bpb after 43 s. The
horizon is short; the ladder runs re-check the optimum at larger budgets.

Telemetry at the default (step 2,000) gives M4 input: **30% of logits
exceed the fixed-point clamp ±12.7** (median |logit| ≈ 10, p99 ≈ 27, max
35; z-loss pins lse near 0, so unlikely bytes sit at large negative
logits). Attention scores clamp 0.9% (layer 1). Nothing else clamps; the
mid-stack latent stays within the byte window (max 22, p99 ≈ 4). The logit
grid's step/range (or the soft-cap) is the first M4 decision.

**X9 — Architecture variants on S (D11, 2026-11-17).** S (≈ 3.2 M),
fp, 4,000 steps × 64 × 512 B (131 MB), Lion lr 3e-3 wd 3 β (0.95, 0.98),
warmup 300, cosine; eval-v1 val bpb, seeds 42 / 7.
`configs/m3-s-variants.toml`.

| variant | params | bpb s42 / s7 | mean | Δ | books / wiki-en / wiki-de / defs | speed (eager → compiled) |
|---|---|---|---|---|---|---|
| (a) baseline | 3.21 M | 1.7931 / 1.7932 | 1.7931 | – | 1.790 / 1.749 / 1.849 / 1.743 | 368 → 562 kB/s |
| (b) conv stem | 3.26 M | 1.7572 / 1.7526 | 1.7549 | −0.038 | 1.752 / 1.711 / 1.807 / 1.711 | 345 |
| (c) q/k/v conv | 3.22 M | 1.7413 / 1.7385 | **1.7399** | **−0.053** | 1.738 / 1.697 / 1.785 / 1.697 | 242 → 479 kB/s |
| (d) d 192 × 7 | 3.15 M | 1.7654 / 1.7809 | 1.7732 | −0.020 | 1.770 / 1.731 / 1.826 / 1.729 | 298 |
| (b) + (c) | 3.27 M | 1.7273 / 1.7398 | 1.7336 | −0.060 | (s42) 1.727 / 1.682 / 1.775 / 1.682 | – |

Verdict: **keep (c)**, the causal depthwise conv on q, k, v (Primer): it is
the largest single gain, consistent over seeds and sources. (b) helps
alone but is redundant next to (c): the combination is only 0.006 better
than (c), well inside the seed spread. Both inject local n-gram
information, and once every layer has its own short conv the stem adds
nothing measurable. (d) is marginal (0.02 at 0.016 spread) and 19%
slower: dropped, so the current widths stay. The gains are similar across
sources; wiki-de (most multi-byte characters) gains only slightly more.
`torch.compile` is now the default for transformer training (+53% on S;
q/k/v-conv overhead 35% → 15%). The conv stem stays in the code as a
candidate front end for patching/hourglass designs (M6).

**X10 — Scaling ladder (M3 acceptance, 2026-11-17).** Chosen architecture
(q/k/v conv, tied embedding), fp, 300 MB per size (9,155 steps × 32 kB),
Lion lr 3e-3 wd 3 β (0.95, 0.98), warmup 300, cosine, compiled; L with
micro-batches of 8 (fp32 attention does not fit otherwise). Seed 42.
`configs/m3-ladder.toml`.

| preset | params | eval-v1 bpb | books / wiki-en / wiki-de / defs | train speed | P4 UTF-8 valid |
|---|---|---|---|---|---|
| XS | 0.43 M | 1.946 ± 0.010 | 1.918 / 1.914 / 2.061 / 1.898 | 1,673 kB/s | 32/32 |
| S | 3.2 M | 1.668 ± 0.009 | 1.674 / 1.621 / 1.700 / 1.611 | 477 kB/s | 32/32 |
| M | 10.7 M | 1.580 ± 0.008 | 1.594 / 1.523 / 1.595 / 1.530 | 166 kB/s | 32/32 |
| L | 25.3 M | **1.536 ± 0.008** | 1.556 / 1.477 / 1.541 / 1.480 | 85 kB/s | 32/32 |

Monotone in size; S beats KN-5 (2.415) by 0.75 and xz (2.90) by 1.2 bpb.
300 MB is far below compute-optimal for M/L (≈ 30 and 12 bytes per
parameter), so the gaps between the larger sizes understate their
potential. P2 copy and P3 MQAR pass at XS and S with this architecture
(100% each).

Telemetry inputs for M4 (step 9,155):
- **logits** exceed the fixed-point clamp ±12.7 in 51% (XS) → 73% (L)
  of entries, max |logit| 47–60;
- **attention scores** clamp heavily at L only: 46% (layer 4), 43%
  (layer 6), 31% (layer 1), vs ≤ 1% at M. QK-norm gains are vectors,
  so they get no weight decay, and Lion grows them freely: sharper
  attention, harmless in fp, but it would clamp on the score grid;
- everything else stays inside the byte window. Weights are bounded by
  Lion's equilibrium |w| ≤ 1/wd = 0.33, i.e. only the lower ~6.4
  octaves of the global weight grid are used.

**X11 — Is wd 3 a short-horizon artifact? (user question, 2026-11-17).**
In fp nothing is clamped, so the strong decay cannot be a quantization
artifact. Mechanism (X8 telemetry): max |w| = 1/wd exactly (Lion's
sign-update equilibrium); with wd 0.1 the weights grow ×55 and the
residual stream reaches p99 ≈ 8,000. The decay is magnitude control, not
classical regularization. Check at the ladder horizon (XS, 300 MB, seed 42):

| lr, wd | 3e-3, 1 | **3e-3, 3** | 3e-3, 10 | 3e-4, 0.1 |
|---|---|---|---|---|
| bpb | 1.948 | **1.946** | 1.985 | 2.091 |

Verdict: at the longer horizon the optimum widens toward weaker decay (wd
1 ≈ 3; 10 is now too strong). Bounding magnitudes via a low lr instead
is clearly worse. We keep wd 3: as good as 1, with a tighter weight bound
(0.33 vs 1.0) for M4.

**X12 — Logit overflow and growing gains (user request, 2026-11-17).**
Question: at L, logits clamp on the fixed-point grid (all of it on the
negative side), and attention scores clamp because the QK-norm gains
grow without decay (up to ~10). Candidate fixes: decay gains toward 1
(Lion decay anchored at the initial value, `train.norm_wd`), decay the
q/k/v conv kernels toward identity instead of 0 (`train.conv_decay
init`), and logit QAT on the current grid (`fx`) vs max-relative
(`maxrel`: logit − row max stored as an unsigned byte, [−25.5, 0] at
step 0.1). S, 4000 steps, q/k/v conv 3, seeds 42 and 7 (seed spread
≈ 0.004 bpb). Clamp rates are fx-grid rates, worst layer, from the last
telemetry:

| variant | bpb (mean) | Δ | score clamp | logit clamp | max gain | conv id-tap |
|---|---|---|---|---|---|---|
| a base | 1.7370 | — | 18.0% | 58.9% | 5.24 | 0.056 |
| b gains → 1, wd 3 | 1.7618 | +0.025 | 0.0% | 33.1% | 1.33 | 0.068 |
| c gains → 1, wd 1 | 1.7483 | +0.011 | 0.2% | 40.3% | 1.99 | 0.061 |
| d = b + conv → identity | 1.7539 | +0.017 | 0.0% | 34.0% | 1.33 | 0.881 |
| e = c + conv → identity | 1.7532 | +0.016 | 0.0% | 41.7% | 1.99 | 0.881 |
| f logit QAT, fx grid | 1.7391 | +0.002 | 21.0% | 54.9% | 5.40 | 0.056 |
| g logit QAT, max-relative | 1.7397 | +0.003 | 17.9% | 58.2% | 5.29 | 0.056 |

Findings:
- **Gain decay works mechanically but costs 0.011–0.025 bpb.** Sharp
  attention is used, not an accident: gains grow because the model
  needs scores spanning more than ±12.7. Conv decay toward identity
  keeps the identity tap (0.88 vs 0.06) but does not pay for itself.
- **Logit QAT is neutral at S** for both grids (within seed noise).
- **Max-relative storage also fits attention scores**, since softmax is
  shift-invariant like CE. Post-hoc on the base checkpoints
  (`Tap.rel`, 64 val chunks), the fraction of entries more than 25.5
  below their row max is:

  | | scores (worst layer) | logits |
  |---|---|---|
  | seed 42 | 18.0% | 17.0% |
  | seed 7 | 6.3% | 16.7% |

  Those entries carry probability < e^−25.5 ≈ 8e-12, so dropping them
  loses no mass. The fx-grid "clamp" counted mostly them (fx 22% vs
  rel 18% in the worst layer). The remaining fx-only clamps are real
  range loss at the top of the row, which max-relative removes.

Verdict (001-D12): no gain decay, no conv anchor (both stay available,
default off). Attention scores and logits use the max-relative unsigned
byte in M4.

**X13 — M4 phase A: cumulative quantization stages at S (2026-11-17).**
S preset, 4000 steps, q/k/v conv 3, Lion (0.95, 0.98), lr 3e-3, wd 3,
seeds 42 and 7. The fp twin is X12 a-base (1.7349 / 1.7390).
Activation sites use the delayed pow2 scale (16-forward amax history,
×2 headroom).

| stage (cumulative) | s42 | s7 | Δ vs fp |
|---|---|---|---|
| a weights (16-bit master, byte forward) | 1.7558 | 1.7429 | +0.012 |
| b1 + activations (residual stream fp) | 1.7503 | 1.7634 | +0.020 |
| b2 + residual stream on the byte grid | 1.7508 | 1.7533 | +0.015 |
| c + max-relative scores and logits | 1.7505 | 1.7558 | +0.016 |
| d + attention probabilities (byte grid, scale 4096) | 1.7554 | 1.7489 | +0.015 |
| e + pure-int gradients | 1.7573 | 1.7566 | +0.020 |

Findings:
- **Full byte numerics cost ≈ 0.015–0.020 bpb at S**, a fifth of the
  0.10 gate. Almost all of it comes with the weights (stage a); later
  stages are within seed noise, which is larger quantized (0.013 between
  seeds at stage a).
- **Range is not a problem on the log grid.** At the frozen delayed
  scales (32 val chunks, `runs/m4a_range.py`), no site clamps.
  Underflow is 0.09% overall, worst 0.4–0.7% (ReLU² output, final
  norm), and those values sit ~17 octaves below each site's max.
- **Throughput.** The phase-A runs were 3× slower than fp because the
  activation sites broke torch.compile graphs: `float()` of numpy
  scalars traced as tensors, plus data-dependent limits. With that
  fixed, there are 0 graph breaks and full quantization costs 1.55× fp
  per step at S. At M, micro-batch 16 runs at 310 ms/step and peaks at
  8.7 GiB. The numerics are unchanged, so the results stand.

Consequence for 001-D10 (learned scales): with no clamps and negligible
underflow, a learned scale has nothing to trade on the log grid, as
§5 M4 expected. Where a learned scale does trade range against
resolution is the linear int8 grid, which phase C re-tests.

**X14 — M4 gate: full byte numerics at S and M (2026-11-18).** Stage d
of X13: 16-bit master weights with byte forward, activations and the
residual stream on the byte grid (delayed pow2 scales), max-relative
scores and logits, and byte-grid attention probabilities. Ladder
horizon (300 MB), seed 42, against the m3-ladder fp twins (X10).

| step | S fp | S quant | M fp | M quant |
|---|---|---|---|---|
| 2000 | 2.0756 | 2.0984 | 1.9982 | 2.0173 |
| 4000 | 1.9863 | 1.9966 | 1.9259 | 1.9245 |
| 6000 | 1.8887 | 1.8948 | 1.8145 | 1.8108 |
| 8000 | 1.7289 | 1.7410 | 1.6466 | 1.6415 |
| **9155** | **1.6684** | **1.6814 (+0.013)** | **1.5797** | **1.5760 (−0.004)** |

**The gate passes:** S +0.013 and M −0.004, against the allowed 0.10.
Samples are 32/32 valid UTF-8 at both sizes. The quantized model starts
behind (+0.02 at step 2000) and catches up; at M the gap is within
noise.

Throughput (quant vs fp):

| | quant | fp |
|---|---|---|
| S | 235 kB/s | 477 kB/s |
| M (micro-batch 16) | 87 kB/s | 166 kB/s |

The first M attempt hit OOM in the interim eval: 64 windows of explicit
attention at T = 1024. The eval batch is now sized to ≤ 2²⁷ scores per
tensor; that is safe because the scales are frozen at eval.

Probes, fp vs stage-d quantization (probes take `qc=`; scored with frozen
scales; `runs/m4_probes.py`). Pass thresholds: P2 ≥ 0.99, P3 ≥ 0.95.

| | P1 memorize (CE) | P2 copy | P3 MQAR |
|---|---|---|---|
| fp, XS / S | 0.0000 / 0.0000 | 1.000 / 1.000 | 1.000 / 1.000 |
| quant, XS / S | 0.0001 / 0.0000 | 0.992 / 1.000 | 1.000 / 1.000 |

All pass quantized. P2 at XS is closest to its threshold. P4 (UTF-8
validity) is 32/32 at S and M (above).

**X15 — M4 phase C: weight storage, 1 byte + SR vs 2-byte master
(2026-11-18).** Stage d (X13), S, 4000 steps. The w2 numbers are X13
stage d.

| storage | s42 | s7 | mean | Δ vs fp |
|---|---|---|---|---|
| w2: 16-bit master, RTN, byte forward | 1.7554 | 1.7489 | 1.7522 | +0.015 |
| w1-SR: byte only, stochastic rounding | 1.7647 | 1.7681 | 1.7664 | +0.029 |

w1-SR costs about +0.014 bpb more than w2 at S; both of its seeds are
worse than both w2 seeds. In exchange it halves the weight state. It
works, so it is an option if memory binds (plan 002), but w2 stays the
default.

**X16 — M5 round 1: Lion lr × wd × β2, quantized S (2026-11-18).**
Stage d, S, 4000 steps, seed 42, β1 0.95.

| lr \ wd, β2 | wd 1, 0.98 | wd 1, 0.99 | wd 3, 0.98 | wd 3, 0.99 |
|---|---|---|---|---|
| 1.5e-3 | 1.8420 | 2.0248 | 1.7619 | 1.7924 |
| 3e-3 | 1.7756 | 1.8354 | **1.7514** | 1.7612 |
| 6e-3 | 1.7527 | 1.7626 | 1.7768 | 1.7696 |

Findings:
- **The current default is the optimum** (lr 3e-3, wd 3, β2 0.98).
- **The controlling variable is lr·wd**, Lion's decoupled decay rate
  per step: 1.5e-3 → 1.84; 3e-3 → 1.78; 4.5e-3 → 1.76; 6e-3 → 1.75;
  9e-3 → 1.75; 18e-3 → 1.78 (β2 0.98). The optimum is lr·wd ≈ 6–9e-3,
  i.e. weights forget over ~110–170 steps. Weak decay with a low lr is
  badly worse, because nothing bounds the magnitudes (X6).
- **β2 0.98 beats 0.99 in 5 of 6 cells** (the exception is lr 6e-3,
  wd 3), so round 2 tries 0.95.
- **Same-seed reproducibility:** this default cell reruns m4a-d-probs
  s42 (1.7554) and gives 1.7514. Compiled kernels are nondeterministic,
  so differences under ~0.005 between single runs are noise.

**X17 — M5 round 2: β2, per-group lr, batch (2026-11-18).** Stage d,
S, 4000 steps, seed 42, at the X16 optimum (default 1.7514; noise
≈ ±0.005).

| change | bpb | verdict |
|---|---|---|
| β2 0.95 | 1.7608 | worse; keep 0.98 |
| lr_emb 0.3 (embedding / tied head) | **1.7469** | −0.0045, borderline: round 3 |
| lr_emb 3 | 1.7527 | noise |
| lr_vec 0.3 (gains) | 1.7713 | worse |
| lr_vec 3 | 1.7547 | noise |
| batch 32, 8000 steps (same bytes) | 1.7809 | worse |
| batch 128, 2000 steps (micro 64) | 1.8231 | worse |

The gains need their full lr (cf. X12: they must grow). Batch 64 stays;
at a fixed byte budget, other batch sizes would need their own lr·wd
retuning.

**X18 — M5 round 3: lr_emb 0.3 at the ladder horizon (2026-11-18).**
Quantized S (stage d), 9155 steps.

| | s42 | s7 | mean |
|---|---|---|---|
| default | 1.6814 (X14) | 1.6783 | 1.6799 |
| lr_emb 0.3 | 1.6794 | 1.6880 | 1.6837 |

No gain; the X17 hint was noise. **The Lion defaults stand** for the
quantized model: lr 3e-3, wd 3, betas (0.95, 0.98), batch 64, one lr
for all groups.

**X19 — M5 round 4: quantized L at the ladder horizon (2026-11-18).**
Stage d, L (d512, 8 layers, T1024), batch 32 (micro-batch 8, 7.9 GiB),
seed 42, against m3-ladder-L (fp).

| step | 2000 | 4000 | 6000 | 8000 | 9155 |
|---|---|---|---|---|---|
| fp | 1.9627 | 1.8827 | 1.7736 | 1.6035 | 1.5364 |
| quant | 1.9827 | 1.8977 | 1.7844 | 1.6132 | **1.5478 (+0.011)** |

Training was stable throughout, at 46 kB/s (1.8 h). The quantization
cost does not grow with scale:

| | S | M | L |
|---|---|---|---|
| Δ vs fp | +0.013 | −0.004 | +0.011 |

Post-training sampling hit OOM: 32 unbatched sequences with explicit
attention at T1024. `sample()` now batches like `chunk_bits`
(`max_batch`). P4 was rerun from the checkpoint with
`runs/m5_sample.py`.

**P4 at L quantized: 30/32 valid (93.75%), below the 99% bar.** The two
failures:
- a degenerate repeating loop in Hebrew text, with continuation bytes
  where a lead byte belongs;
- one stray continuation byte (0x82) inside ASCII.

Teacher-forced on 64 val chunks (`runs/utf8_mass.py`), the probability
mass on bytes that would break UTF-8 at T = 0.8 is the same for fp and
quant:

| run | invalid mass / byte | expected events per 1 KB |
|---|---|---|
| S fp | 1.45e-5 | 0.015 |
| S quant | 7.9e-6 | 0.008 |
| L fp | 7.8e-6 | 0.008 |
| L quant | 8.4e-6 | 0.009 |

So on real text quantization does not degrade UTF-8 structure. The
failures come from free generation drifting into rare territory, and
2/32 has a 95% CI of ≈ 1–21%. 96 more samples each from fp-L and
quant-L decide it (pending).

**X20 — M6 round 1: hourglass vs flat, fp (2026-11-18).** Same
32 KB/step and 300 MB as the m3-ladder, seed 42. Byte path: 2 + 2 conv
ByteBlocks (kernel 4), byte width d/2. The core is the flat preset of the
same name.

| model | bpb | train time | UTF-8 |
|---|---|---|---|
| flat S / M / L | 1.668 / 1.580 / 1.536 | 10.5 / 30 / 59 min | 32/32 |
| hourglass S-k2 (T 1024) | 1.766 | 9.3 min | 32/32 |
| hourglass S-k4 (T 2048) | 1.896 | 5.2 min | 32/32 |
| hourglass M-k4 | 1.839 | 10.8 min | 31/32 |
| hourglass L-k4 | 1.824 | 18.8 min | 32/32 |

The hourglass is worse at equal time, as the user expected with a small
core. But scaling the core S→L (7× parameters) gains only 0.07 bpb,
against 0.13 for flat. Diagnostic (`runs/m6_offset.py`, 256 val chunks):
bpb by input position inside the patch, t mod k.

| model | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| flat S | 1.672 | 1.669 | 1.672 | 1.675 |
| S-k4 | 1.887 | 1.905 | 1.931 | 1.872 |
| L-k4 | 1.814 | 1.831 | 1.850 | 1.814 |

At t mod 4 = 3 the patch vector is freshest (it covers byte t). Staleness
inside a patch costs ~0.06, but even the freshest position is 0.14–0.2
behind flat. The loss is in the patch representation, not in the
decoder's timing. The prime suspect is the conv-only byte path
(receptive field ≈ 7 bytes, no attention) acting as a weak "tokenizer";
published hourglass models use attention at byte level.

Round 2 tests both explanations at S: byte blocks with sliding-window
attention (window 64, `model.byte_mix = "attn"`) vs a deeper core (L 8),
and both together.

**X21 — M6 round 2: byte attention vs deeper core; the byte-path
shortcut (2026-11-18).** S, fp, seed 42, as X20.

| variant | bpb |
|---|---|
| S-k4, conv byte path (X20) | 1.896 |
| S-k4, sliding-window byte attention (window 64) | 1.910 |
| S-k4, conv, core L 8 | 1.929 |
| S-k4, attention + L 8 | 1.911 |
| S-k2, conv (X20) | 1.766 |
| S-k2, attention | 1.927 |

Nothing helped, and a deeper core was worse. Ablation
(`runs/m6_ablate.py`, 128 val chunks): zero the upsampled global vector.

| model | bpb | global off | rms(up) / rms(byte stream) |
|---|---|---|---|
| S-k2 conv | 1.784 | 2.761 (+0.98) | 0.18 / 0.34 |
| S-k4 conv | 1.913 | 2.536 (+0.62) | 0.10 / 0.32 |
| L-k4 conv | 1.838 | 2.415 (+0.58) | 0.13 / 0.41 |
| S-k4 attn | 1.929 | 2.066 (+0.14) | 0.02 / 0.13 |

Diagnosis: **the byte path short-cuts the core.** Byte layers that see
across patch boundaries (convs ≈ 7 bytes, window attention 64 bytes)
learn local statistics fast, and the model leans on them. The stronger
the byte path, the less the core is used (attention: +0.14, with the
loss flat over the patch offset). That runs against the cognitive-core
goal. **Fix (round 3): a patch-local byte path**, `byte_mix = "patch"`,
MEGABYTE-style:
- encoder attention stays inside each patch (a patch vector is a learned
  token of its own k bytes);
- the decoder sees the previous patch's core vector plus the bytes so far
  in its own patch;
- all cross-patch information must pass through the core (tested);
- attention costs O(T·k), which also realizes the VRAM saving.

**VRAM (user, 2026-11-18)** becomes a third M6 axis next to bpb and
time. 12 GB is the hard limit. Core activations shrink by k and
attention scores by k², which is headroom for a larger core and for plan
002 state. ig-train now logs `peak_gib` with the final eval.

**X22 — M6 round 3: patch-local byte path (2026-11-18).**
`byte_mix = "patch"`, fp, seed 42, 300 MB. Peak VRAM is training only.

| model | bpb | time | VRAM | global off |
|---|---|---|---|---|
| S-k4 patch | 1.994 | 7.1 min | 2.1 GB | +1.52 |
| S-k4 patch, core L 8 | 1.969 | 9.8 min | 2.9 GB | |
| S-k2 patch | 1.785 | 10.5 min | 2.9 GB | |
| M-k4 patch | 1.916 | 13.1 min | 3.7 GB | +1.53 |

Reference: flat S 1.668 (10.5 min, ~3 GB), flat M 1.580 (30 min).

**The core now carries the model**: switching the global path off costs
1.5 bpb, against 0.14–0.6 with the wider byte paths. **Core capacity now
pays**: L 8 beats L 4, and M beats S by 0.08; in round 2 a deeper core was
*worse*. The gap to flat is still large (M-k4 is +0.34 vs flat M, at half
the time and ~half the VRAM), so the hourglass is now core-limited.
Patch offset at S-k4: 1.995 / 2.004 / 2.029 / 1.958.

Round 4 spends the VRAM headroom on the core: L-k4 patch, and an XL core
(d 768, 12 layers).

**X23 — M6 round 4: core size with the patch-local byte path (2026-11-18).**
k 4, fixed patches, fp, seed 42, 300 MB.

| core | bpb | time | peak VRAM | params |
|---|---|---|---|---|
| S (d256 L4) | 1.994 | 7.1 min | 2.1 GB | 4.0 M |
| M (d384 L6) | 1.916 | 13.1 min | 3.7 GB | ~12 M |
| L (d512 L8) | **1.847** | 21.8 min | 5.7 GB | ~28 M |
| XL (d768 L12, d_b 256) | 1.879 | 43.7 min | 9.8 GB | 89.8 M |

The core scales up to L (−0.07 to −0.08 per step); XL is worse. 300 MB
is only 75 M patches for 90 M parameters (data-starved), and lr 3e-3 was
tuned at S. Compared with flat M (1.580, 30 min) the gap is still 0.27
bpb. The open levers are patch boundaries (round 5), training length,
and lr at large width.

**X24 — M6 round 5: language-agnostic patch boundaries (2026-11-18).**
M core, k 4, patch-local byte path, fp, seed 42, 300 MB.

| patching | bpb | wiki-en | wiki-de | books | time | peak VRAM |
|---|---|---|---|---|---|---|
| fixed 4-byte grid | 1.916 | 1.890 | 2.017 | 1.884 | 13.1 min | 3.7 GB |
| utf8 (grid snapped to char ends) | 1.894 | 1.863 | 2.000 | 1.864 | 25.2 min | 4.0 GB |
| chars (4 characters) | **1.890** | 1.862 | 1.994 | 1.858 | 25.1 min | 4.0 GB |
| utf8, narrow (bug, see below) | 1.973 | 1.954 | 2.115 | 1.927 | 24.9 min | 4.0 GB |

- **The first run was confounded.** The variable-patch path passed only
  the last byte state to the core (d_b instead of k·d_b) and gave one
  vector to every byte of the next patch. It was +0.06 on every source,
  English included. Fixed in `59720b3`; the narrow runs are kept as
  `*-narrow`.
- **After the fix, not splitting characters helps by 0.022–0.026 bpb.**
  The gain is about uniform across sources: German −0.018/−0.024,
  English −0.027, books −0.020/−0.026. English wiki is not pure ASCII
  (typographic quotes, dashes, names), so whole characters per patch help
  there too. This is a single seed: about 2 eval-se and paired on the
  same eval set, but seed noise is unmeasured.
- **utf8 and chars are equal within noise.** utf8 is the default: its
  patch length in bytes is bounded (≤ k+3), so compute per byte is
  predictable. chars puts 12 bytes into a CJK patch and leaves global
  slots empty on multi-byte text.
- **Cost:** 2× wall time from the dense [B,1,T,T] segment mask (explicit
  attention instead of fused SDPA). Peak VRAM +0.3 GB. Quality is not
  affected; chunked segment attention is the fix before M7 scales up.
- M6 standing: hourglass L-k4 is 1.847 (fixed); the utf8 gain applies on
  top. The gap to flat (L 1.536) is about 0.3. The training curves show
  the hourglass still closing the gap at the end of the schedule, with a
  larger late cosine drop than flat (0.08–0.09 vs 0.067). The next
  levers are lr for the hourglass core and a longer training run.

**X25 — M6 round 6: L core with utf8 patching, Lion lr × wd (2026-11-18).**
L (d512 L8, d_b 256), k 4, utf8, patch-local byte path, local O(T·W)
segment attention (`6bb37ca`), fp, seed 42, 300 MB.

| run | lr | wd | lr·wd | bpb | wiki-en | wiki-de | books | late drop |
|---|---|---|---|---|---|---|---|---|
| L-k4 fixed (X23) | 3e-3 | 3 | 9e-3 | 1.847 | 1.819 | 1.950 | 1.817 | 0.086 |
| L-utf8 | 3e-3 | 3 | 9e-3 | 1.829 | 1.791 | 1.910 | 1.810 | 0.080 |
| L-utf8 | 1.5e-3 | 6 | 9e-3 | 1.868 | 1.846 | 1.971 | 1.833 | 0.083 |
| L-utf8 | 1e-3 | 9 | 9e-3 | 1.826 | 1.786 | 1.911 | 1.807 | 0.078 |
| L-utf8 | 1.5e-3 | 3 | 4.5e-3 | **1.809** | 1.775 | 1.897 | 1.787 | 0.055 |

- **utf8 helps at L as well:** −0.018 against fixed, about the same as at
  M (X24).
- **Halving lr·wd is best** (−0.020 against the default). Its late cosine
  drop shrinks from 0.080 to 0.055: the noise floor that the curves
  showed. The hourglass core prefers a lower lr·wd than the flat S
  optimum (6–9e-3, X16–X18).
- **At constant lr·wd = 9e-3 the result is not monotone in lr** (1.5e-3
  is worse than both 3e-3 and 1e-3, by 0.04). That is larger than the
  expected seed noise and is unexplained; the product is not the whole
  story at L.
- Wall time is 23.5 min (local band; the dense mask would take ~45).
  Peak VRAM is 7.3 GB against 5.7 for fixed: the band unfolds K/V to
  [B,H,T,dh,W]. That is acceptable under 12 GB; a fused band kernel
  would remove it.
- The best is at the edge of the sweep. Round 7 brackets it: lr·wd
  2.25e-3 (lr 1.5e-3, wd 1.5), and lr·wd 4.5e-3 at lr 3e-3 / wd 1.5 and
  at lr 7.5e-4 / wd 6, to separate lr from wd.

**X26 — M6 round 7: separating lr from wd at L-utf8 (2026-11-18).**
Same setup as X25. All seven L-utf8 runs:

| lr \ wd | 1.5 | 3 | 6 | 9 |
|---|---|---|---|---|
| 3e-3 | 1.830 | 1.829 | | |
| 1.5e-3 | 1.808 | 1.809 | 1.868 | |
| 1e-3 | | | | 1.826 |
| 7.5e-4 | | | **1.793** | |

- **The lr is the lever:** at a constant lr·wd = 4.5e-3, going from lr
  3e-3 → 1.5e-3 → 7.5e-4 gives 1.830 → 1.809 → 1.793.
- **wd is nearly neutral from 1.5 to 3** (Δ ≤ 0.001 at both lrs), but
  **wd ≥ 6 hurts** (lr 1.5e-3: +0.059 from wd 3 to wd 6).
- That explains X25's non-monotone result: the lr·wd product rule from
  flat S (X16–X18) does not carry over to the hourglass core. Its optimum
  lies at a lower lr. lr 1e-3 · wd 9 and lr 7.5e-4 · wd 6 both carry a wd
  penalty, so lr 7.5e-4 at wd 3 should be better still.
- The lower lr also learns faster early: 2.415 at step 2000 against
  2.602 at the default.
- The best is still at the edge of the sweep. Round 8: lr 7.5e-4,
  5e-4 and 3e-4, each at wd 3.

**X27 — M6 round 8: lr optimum for the L-utf8 hourglass (2026-11-18).**
wd 3, otherwise as X25.

| lr | bpb | wiki-en | wiki-de | books | late drop |
|---|---|---|---|---|---|
| 3e-3 (default) | 1.829 | 1.791 | 1.910 | 1.810 | 0.080 |
| 1.5e-3 | 1.809 | 1.775 | 1.897 | 1.787 | 0.055 |
| **7.5e-4** | **1.765** | 1.720 | 1.833 | 1.754 | 0.035 |
| 5e-4 | 1.795 | 1.762 | 1.877 | 1.774 | 0.027 |
| 3e-4 | 1.847 | 1.831 | 1.958 | 1.807 | 0.020 |

- **Optimum bracketed at lr 7.5e-4 / wd 3:** −0.064 against the
  default, and −0.082 against L-k4 fixed at the default lr (X23).
- lr 5e-4 leads at step 4000 (2.157 vs 2.177) but falls behind in the
  tail. That is the usual low-lr pattern; the late cosine drop shrinks
  monotonically with lr.
- **Hourglass default from here: lr 7.5e-4, wd 3.** That is 4× below the
  flat optimum found at S.
- Gap to flat L (1.536) is 0.23. The flat L lr was never tuned at L, so
  round 9 also runs flat L at lr 1.5e-3, to keep the comparison fair.
- **Round 9:** the best setting trained 4× longer (36,620 steps, 1.2 GB,
  no data repetition: the corpus is about 20 GB), then flat L at lr 1.5e-3.

**X28 — M6 round 9: 4× training and a fair flat L (2026-11-18).**

| run | data | bpb | wiki-en | wiki-de | books | wall time |
|---|---|---|---|---|---|---|
| hourglass L-utf8, lr 7.5e-4 | 300 MB | 1.765 | 1.720 | 1.833 | 1.754 | 23.5 min |
| hourglass L-utf8, lr 7.5e-4 | 1.2 GB | 1.644 | 1.588 | 1.693 | 1.646 | 93.4 min |
| flat L, lr 3e-3 (M3) | 300 MB | 1.536 | 1.477 | 1.541 | 1.556 | 58.8 min |
| flat L, lr 1.5e-3 | 300 MB | **1.525** | 1.466 | 1.531 | 1.546 | 58.6 min |

- **4× data buys the hourglass 0.121 bpb** (about 0.06 per doubling). The
  4× curve flattens between steps 12k and 20k, and most of the gain comes
  in the cosine tail. At this length the lr is probably again too high
  in the middle of the schedule.
- **Flat L gains a little from a lower lr** (−0.011). The fair gap at equal
  data is 0.24 bpb.
- **At equal wall time the gap is still about 0.16** (hourglass at about
  2.5× data, interpolated: about 1.685, against flat L at 1.525). At 0.06 per
  doubling, the hourglass would need about 16× the data (about 6 h) to reach
  flat L at 300 MB. At this scale flat is ahead per byte and per GPU-hour.
- **VRAM, measured correctly:** in training alone the hourglass L needs
  3.8 GiB at 76 ms/step, against 5.9 GiB at 174 ms/step for flat L
  (compiled, bf16, the same 32 KiB per step). The logged 7.3 GB was the eval
  peak: eval batches are sized by core attention, so the hourglass
  evaluates 4× the bytes per batch. Training and eval peaks are now
  logged separately.
- **M6 conclusion for this scale:** the hourglass delivers what it promised
  on cost (2.3× faster, −35% VRAM) but not on quality per compute. It
  becomes the better choice only where VRAM binds, i.e. cores well above L.
  That is the open question for the M7 base.

**X29 — M6 round 10: where the byte path loses information (2026-11-19).**
L core, utf8, lr 7.5e-4, wd 3, 300 MB; one change per run against 1.765.

| run | change | bpb | wiki-en | wiki-de | books | wall time |
|---|---|---|---|---|---|---|
| baseline | enc 2, dec 2, d_b 256, patch-local | 1.765 | 1.720 | 1.833 | 1.754 | 23.5 min |
| dec4 | 4 decoder layers | 1.754 | 1.717 | 1.824 | 1.737 | 28.5 min |
| enc4 | 4 encoder layers (patch-local) | 1.766 | 1.723 | 1.835 | 1.755 | 28.6 min |
| encw64 | all-wide encoder, window 64, embedding skip | 1.933 | 1.934 | 2.075 | 1.874 | 31.0 min |
| encw256 | same, window 256 | 1.902 | 1.896 | 2.036 | 1.849 | 30.7 min |
| **k2** | patches of 2 bytes | **1.605** | 1.551 | 1.632 | 1.612 | 39.1 min |
| dec8, d_b 512 | | OOM | | | | |

- **The compression ratio is the lever.** k2 gains 0.16 and beats the
  4× k4 run (1.644 in 93 min) in 39 min. Per GPU-hour k2 is the best
  hourglass so far, 0.08 behind flat L (1.525, 58.6 min).
- **Byte-path depth hardly matters.** Decoder depth gives −0.011; encoder
  depth gives nothing. The k-byte patch vector, not the byte layers, is
  the bottleneck.
- **The all-wide encoder is much worse, but confounded.** It also moves the
  decoder skip from the patch-local encoder output to the raw embedding.
  Round 11 separates the two: an embedding-skip control, and `enc_wide`
  (local encoder for the skip, wide layers feeding only the down path).
- **OOM, and a VRAM correction to X28.** dec8 and d_b 512 ran out of memory
  in one 16×2048 pass. Training peaks at 16×2048 per pass are 7.3 GiB
  (baseline), 9.2 (dec4/enc4), 6.6 (all-wide encoder) and 9.7 (k2). X28's
  "3.8 GiB in training" came from a probe, not this setting, and is wrong.
  Flat L runs with micro-batches of 8×1024 (8 KiB per pass) and peaks at
  3.3 GiB. At the same 8 KiB per pass the hourglass peaks at 2.1 GiB, so
  X28's direction (about −35%) holds. dec8 and d_b 512 fit at 3.6 and
  3.2 GiB with `micro_batch` 4.
- **Round 11:** embedding-skip control; `enc_wide` 2 layers at windows 64
  and 256; dec8 and d_b 512 with micro-batches; k2 + dec4.
- **Decision (2026-11-19):** round 11 is the last M6 round. Unless it brings
  the hourglass within about 0.03 bpb of flat L at equal wall time, M6 is
  shelved: code and tests stay, and flat L, after a short hyper-parameter
  sweep, becomes the M7 base. Rationale: round 10 shows the gap is core
  compute per position (k is the lever, byte-path depth is not), and at
  this scale VRAM does not bind (flat L trains in 3.3 GiB). The hourglass
  may return where compute per position binds, e.g. a recurrent core.

**X30 — M6 round 11, the last M6 round (2026-11-19).** Same setting as X29.

| run | change | bpb | wiki-en | wiki-de | books | wall time |
|---|---|---|---|---|---|---|
| baseline | enc 2, dec 2, d_b 256 | 1.765 | 1.720 | 1.833 | 1.754 | 23.5 min |
| embskip | patch-local encoder, decoder skip from the embedding | 2.050 | 2.074 | 2.229 | 1.969 | 24.0 min |
| encwide64 | local encoder (skip) + 2 wide layers into down, window 64 | 1.789 | 1.750 | 1.872 | 1.770 | 35.5 min |
| encwide256 | same, window 256 | 1.803 | 1.769 | 1.892 | 1.779 | 35.2 min |
| dec8 | 8 decoder layers (micro-batch 4) | 1.724 | 1.684 | 1.775 | 1.715 | 34.6 min |
| db512 | d_b 512 (micro-batch 4) | 1.768 | 1.735 | 1.844 | 1.748 | 37.4 min |
| k2 (X29) | patches of 2 bytes | 1.605 | 1.551 | 1.632 | 1.612 | 39.1 min |
| k2-dec4 | k2 + 4 decoder layers (micro-batch 8) | 1.607 | 1.550 | 1.636 | 1.615 | 43.4 min |

- **The decoder skip carries most of the byte path.** Moving it from the
  encoder output to the embedding costs 0.285. Against that control, the
  all-wide encoder of X29 recovers 0.12–0.15: wide context helps the patch
  vector only when the decoder has nothing better.
- **Wide context on top of a local encoder does not help** (+0.024 at 64,
  +0.038 at 256). The core already supplies the earlier text.
- **Decoder depth helps a little at k4** (dec4 −0.011, dec8 −0.041) and not
  at all at k2 (k2-dec4 ±0). Width (d_b 512) does nothing.
- **Stop rule (X29) not met.** The best hourglass, k2 at 1.605 in 39 min,
  would reach about 1.57 at flat L's 59 min (0.06 per doubling of data),
  about 0.045 behind flat L (1.525). **M6 is shelved; flat L is the M7
  base**, after the sweep in `runs/m7a.sh` (lr 1e-3 and 7.5e-4; wd 1 and 6;
  betas 0.9/0.99; warmup 1000).

**X31 — M7 prep: flat L sweep, round 1 (2026-11-19).** Flat L, 300 MB, Lion,
one change per run against lr 1.5e-3, wd 3, betas 0.95/0.98, warmup 300
(1.525). Micro-batch 16: 6.25 GiB, 57.7 min per run.

| run | bpb | Δ | wiki-en | wiki-de | books |
|---|---|---|---|---|---|
| **lr 7.5e-4** | **1.494** | −0.031 | 1.429 | 1.489 | 1.520 |
| lr 1e-3 | 1.501 | −0.024 | 1.436 | 1.499 | 1.525 |
| wd 1 | 1.505 | −0.020 | 1.442 | 1.501 | 1.529 |
| warmup 1000 | 1.513 | −0.012 | 1.450 | 1.512 | 1.537 |
| betas 0.9/0.99 | 1.521 | −0.004 | 1.462 | 1.527 | 1.541 |
| wd 6 | 1.542 | +0.017 | 1.482 | 1.545 | 1.562 |

- **Lower lr is the lever again**, with the same optimum as the hourglass
  core (7.5e-4). The curve is flattening (3e-3 → 1.5e-3 → 1e-3 → 7.5e-4:
  1.536, 1.525, 1.501, 1.494).
- **Less decay and a longer warmup help a little.** With Lion the decay
  per step is lr·wd, so lowering lr already lowers it. Betas are neutral.
- **Tuning moves flat L 0.031 further from the hourglass**: about 0.08
  at equal wall time.
- **Round 2 (running):** lr 5e-4; lr 7.5e-4 with wd 1.5; lr 7.5e-4 with
  warmup 1000.

**X32 — M7 prep: flat L sweep, round 2, and the M7 base (2026-11-19).**

| run | bpb | wiki-en | wiki-de | books |
|---|---|---|---|---|
| lr 7.5e-4 (X31) | 1.494 | 1.429 | 1.489 | 1.520 |
| lr 5e-4 | 1.491 | 1.427 | 1.484 | 1.516 |
| lr 7.5e-4, wd 1.5 | 1.502 | 1.442 | 1.499 | 1.524 |
| **lr 7.5e-4, warmup 1000** | **1.487** | 1.423 | 1.479 | 1.513 |

- The lr optimum is flat between 5e-4 and 7.5e-4.
- At the lower lr, less decay now hurts (+0.008). Lion's decay per step
  is lr·wd, and 7.5e-4 · 3 is already 2.25e-3.
- **M7 base: `configs/m7-base.toml`** (flat L, lr 7.5e-4, wd 3,
  warmup 1000, betas 0.95/0.98, micro-batch 16, 300 MB): 1.487.
- Tuning gain over M3's L is 0.049 (1.536 → 1.487). Single steps are
  below 0.05, so they are configuration choices, not claims (§4.5 rule 3).
- **Seed 43, trained from the config itself: 1.4855** (s42: 1.4870; mean
  1.486). The config reproduces the sweep run, and the seed spread is
  0.002.

**X33 — M7 baseline: position curve and cross-window loss (2026-11-19).**
`ig-position` (`intgrad/position.py`): the val bucket as 128 sequential
streams (seed 1234), 33 consecutive 1024-byte windows each, windows 1–32
scored (4,096 windows, 4.2 MB). Continuation windows are those whose
first 65 bytes contain no document start and no EOD (3,968 of 4,096).
- **cold**: the first 64 bytes of a continuation window scored from the
  window alone.
- **warm**: the same bytes scored with the 960 bytes before them.

| run | window bpb | cold | warm | gap |
|---|---|---|---|---|
| **M7 base (lr 7.5e-4, warmup 1000)** | 1.483 | 1.863 | 1.399 | **0.463 ± 0.005** |
| M7 base, seed 43 | 1.482 | 1.861 | 1.397 | 0.464 ± 0.005 |
| lr 7.5e-4 | 1.491 | 1.872 | 1.408 | 0.464 |
| lr 5e-4 | 1.489 | 1.869 | 1.405 | 0.465 |

Position curve of the M7 base (bpb per offset bin):

| 0 | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64–127 | 128–255 | 256–511 | 512–767 | 768–1023 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.95 | 3.48 | 2.84 | 2.23 | 1.94 | 1.77 | 1.67 | 1.60 | 1.52 | 1.47 | 1.43 | 1.42 |

- **Cold starts cost almost half a bit per byte over the first 64 bytes**,
  and nearly every window is a continuation. This is the number a
  recurrent state in plan 002 has to win back. A state that carried one
  full window perfectly would reach the warm number.
- **The curve still falls between 512 and 1024 bytes** (1.43 → 1.42),
  so context beyond one window still has value. A state may beat
  "warm", which only looks back one window.
- The gap is the same at every tuned setting (0.463–0.465). It is a
  property of the windowing, not of the tuning.

## 9. Risks

1. **Quantized attention/softmax** is new territory (the MLP had none).
   Mitigation: M4 is staged per tensor class with fp twins.
2. **Parity drift** between C and PyTorch (float op order, `exp2`/`log2`
   implementations). Mitigation: parity is defined on *codes*, with a
   golden sweep dense enough to hit every code boundary.
3. **Byte-level compute cost**: ~4× more steps than BPE for the same
   text. Acceptable for a toy model. Budget plan 001 at ≤ 5 GB of bytes
   seen per run at M size.
4. **Lion at small batch**: Lion needs a large batch. With ctx 512–1024 the
   batch in bytes is large even at B = 32–64, but M5 has to confirm this.
