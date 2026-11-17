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

### M3 — Scalable test transformer (fp first, no recurrence)
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

### M4 — Quantized transformer (the core result of plan 001)
**Per-tensor scaling (from the M3 telemetry).** So far only gradients and
Lion momentum have per-tensor scales; all weights and activations share
one global window (clamp ±26.0, underflow < 1.4e-4). Before (a) below,
introduce per-tensor power-of-two scales where the telemetry demands it:
weights per tensor (per output channel if their rows differ by octaves),
activations by delayed absmax (amax history over recent steps, ~1 octave
of headroom, so a kernel needs no extra pass), and a per-class
fixed-point step for attention scores and logits. Decide explicitly how
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
(a) weights w2, (b) + activations, (c) + logits/attention scores on the
fx grid, (d) + softmax probabilities, (e) + pure-int gradients. Then the
ablations that were open or invalid: w1-SR vs w2 storage; log vs
*properly implemented* linear-int8 weight grid (re-run of B1).

Gate: full-quant (a–d) within **0.10 bpb** of fp at S and M, and P1–P4
pass quantized. If the gate fails, run a **sensitivity sweep** (quantize
one tensor class at a time) before touching anything else.

### M5 — Lion at transformer scale
Sweep lr × wd × β2 × batch on S (2 seeds at the optimum), per-group lr
for embeddings/norms, and check stability at M/L. Lion counts as a
**dead end** only if, after this sweep, (i) fp-Lion diverges or stalls at
M/L, or (ii) quantized-Lion cannot reach the M4 gate while fp-Lion can.
Only then does a plan 00x consider another optimizer.

### M6 — Recurrence readiness (hand-off to plan 002)
Using the sequential batcher, measure the non-recurrent model's
**position curve** and **cross-window loss**: the bpb of the first 64 bytes
of each window when the preceding window is in the same doc, vs a cold
start. This is the baseline any recurrent state in plan 002 must beat.
Deliverables: metric implemented, baseline numbers recorded,
`plan/002-recurrent-state.md` drafted.

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
  (§5 M4). Measured, not assumed.
- **001-D9 Lion stays; magnitudes are controlled by design (user
  decision after X6).** The right lr works, and M3 builds in QK-norm,
  z-loss, an optional soft-cap, range telemetry and an early lr × wd
  sweep. There is no AdamW diagnostic twin in M3; that idea is revisited
  only if we get stuck or data points at a shortcoming of Lion.

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
candidate front end for patching/hourglass designs (plan 002).

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
