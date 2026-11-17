# Plan 001 — Reset: byte-level toy LLM with 1-byte compute, 2-byte weights, Lion

*Created 2026-11-17. Supersedes `docs/history/PLAN-v0.md`. Later plans are
numbered `002-…`, `003-…` and build on this one. A plan is never rewritten
after it is closed. Results are appended to its log (§8), and changes of
direction go into the next plan.*

## 0. What this project is

A **toy LLM** that learns to continue text, with four fixed properties:

1. **Byte-centric.** Input is raw UTF-8 and output is raw UTF-8. Vocabulary
   = 256 byte values. No tokenizer.
2. **Byte numerics.** Everything the network *computes with* (weights as
   read by the forward pass, activations, logits, optionally gradients)
   is one byte. **Parameters are stored as 1 or 2 bytes:** the default is a
   16-bit master code whose high byte is the compute weight (old D7).
   1-byte storage with stochastic rounding stays selectable.
3. **Lion optimizer.** It is the only optimizer. It gets replaced only if
   Lion turns out to be a dead end that cannot be fixed (criteria in M5).
4. **Learnability first.** A scalable decoder-only transformer must
   demonstrably learn (probes plus a scaling ladder) before any recurrence
   work starts. Recurrent state is the subject of plan 002. This plan
   prepares the data path for it.

Out of scope (removed, recoverable from git `154876c`): the tool-use
protocol, `ig-pretty`, the teacher/trace generator, the distillation plans,
Adam/SGD, linear-int8 grid emulation and the GEMM hardware probes.

---

## 1. Review of the code base at `154876c`

### 1.1 What holds (keep)

| Finding | Evidence | Caveat |
|---|---|---|
| Log-byte grid (symlog, Δ=0.1) + **clamp-aware STE** lets quantized training memorize to 0 loss | E4 | — |
| **CE exact** over quantized logits, never quantized itself | E5c (saturation artifact) | — |
| Exact accumulation + **one requant per output** (int32-GEMM semantics) | E7 | — |
| **Two-byte master weights** stay within +0.01–0.03 bpb of fp over 20k steps; 1-byte SR drifts to +0.15 | E8 | 120k-param MLP, ctx 4, 1 seed |
| **Lion** with a **16-bit momentum** is quantization-robust (+0.015–0.04); 1-byte RTN momentum freezes | E9 | same vehicle |
| Pure-int gradients with **per-tensor pow2 absmax scale** are ≈ quant (no saturation by construction) | E7/E8 | emulated, float accumulation |

All of these were measured on a 4-byte-context MLP. **Plan 001 has to
re-confirm each one on the transformer (M4).** None of them is a law yet.

### 1.2 Bugs and defects found in this review

| # | Defect | Impact | Status |
|---|---|---|---|
| B1 | **E-E1/E10 linear-grid mode cut the gradient graph**: `ig_value(wview(...))` made each weight a fresh leaf, so W1/W2 never learned and only the biases trained | All `lin*` results (≈4.70 bpb ≈ unigram) are invalid, so the "linear int8 costs 1.7 bpb" conclusion is wrong | code removed; question re-opened in M4 |
| B2 | **Train sampling was uniform per document**: books are 61% of train bytes but 0.67% of documents, so a byte in a book was ~700× less likely to be seen than a byte in a wiki definition | Model trained mostly on short wiki leads | **fixed**: length-weighted sampling (`ig_ds_sample_doc`, ctest `dataset`) |
| B3 | **Eval took one window from each of the first N val docs in index order**, which is in practice only corpus-v0 wiki definitions | "val bpb" measured wiki-defs only, at ≤800 windows (noisy) | **fixed** for now (length-weighted, seeded); proper frozen eval set in M1 |
| B4 | `idx.bin` stores **absolute shard paths** | Index breaks if the repo moves | M1 (index v2: relative paths + source id) |
| B5 | Documents are joined by `"\n \n"` in shards, but the model **never sees a document boundary** (windows stay inside one doc) | Needed for sequential streaming / recurrence | M1 (EOD byte, D7 below) |
| B6 | Scalar autograd: ~440 windows/s on a 120k-param MLP | A transformer cannot be trained on it | D2: PyTorch reference engine |
| B7 | Experiment harness: single seeds, no frozen eval set, fp-oracle runs not paired, flags added per experiment and never removed (≈30 flags) | Conclusions flipped three times (E1–E6 invalidated by E7, E10 by B1, all numbers by B2/B3) | rules in §4.5 |
| B8 | Untracked-but-committed files under `data/` (synthetic traces), fixtures in `data/` | Violates the "data/ is private" rule | **fixed**: `data/` fully ignored, fixtures in `tests/fixtures/` |
| B9 | Checkpoint format accreted v1–v7 compat branches + Adam moments | Complexity | **fixed**: single format v8 |
| B10 | Possible near-duplicate leakage between corpus-v0 wiki-2017 definitions and corpus-v1 wiki-2015 articles (split is by exact content hash) | Optimistic val bpb on wiki | spot check: 0/40 val-snippet hits in v0 (no exact leakage). Near-dup check in M1 (low priority) |

### 1.3 Why the experiments ran in circles

1. The **vehicle could not show the effects being asked about.** A ctx-4
   MLP has no attention, no depth and no long-range structure, so "does
   depth help quant", "grid choice" and so on were answered on a model whose
   ceiling is ~2.8 bpb.
2. **Harness bugs were found only after the conclusions** (B1–B3, E5c, E7).
   No test layer checked that a configuration *trains at all* before it
   was compared.
3. **No frozen protocol**: the eval set, seeds, step budget and batch
   changed between experiment families, so tables are not comparable
   across E-numbers.
4. **Flags were never retired**, so the combinatorial space kept growing
   and each new result had to be checked against old ablations.

The rules in §4.5 address each of these.

### 1.4 Cleanup done in this commit (M0)

- Removed: tool-use stack (`protocol.h`, `ig-pretty`, `gen_traces.py`),
  `lingrid.[ch]` (B1), GEMM probes, Adam/SGD, `nh2`, role grids
  (`--w-delta/--a-delta/--w-scale/--a-scale`), isolation flags
  (`--fp-logits/--loss-fp/--fp-params`), fixed `--gscale`, `--scale`,
  Lion momentum ablations (`b1/b1sr/f32` flags), float-stub
  `ig_logmul/ig_logadd`, checkpoint v1–v7 branches.
- `net.c` 742 → ~370 lines, with one `ig_mlp_groups()` helper replacing four
  copies of the group tables.
- Fixed B2/B3 (length-weighted sampling for train *and* eval) + new ctest
  `dataset`. New tests in `logquant` (w16 view, SR unbiasedness, fx clamp).
- `data/` and `runs/` fully git-ignored. Fixtures moved to `tests/fixtures/`.
  Old run logs and checkpoints moved to the private `runs/legacy/`.
- `docs/PLAN.md`, `docs/EXPERIMENTS.md` moved to `docs/history/` with a
  superseded header and the B1 note.

The C code is now the **numerical oracle** (`libintgrad`, `ig-train`):
small, exact, auditable. It is not the trainer for scalable models.

---

## 2. Numerics specification (the contract both engines implement)

| Tensor | Grid | Bytes | Notes |
|---|---|---|---|
| weights (compute view) | symlog Δ=0.1, scale 256 (max ≈ 25.6) | 1 | high byte of the master code |
| weights (storage) | 16-bit symlog, 256 sub-steps per byte step | **2 (default)** or 1 + SR | `w2` / `w1` switch |
| activations (layer outputs) | symlog Δ=0.1 | 1 | one requant per output after exact accumulation |
| logits (output + attention scores) | fixed-point int8 · 0.1 (±12.7) | 1 | absolute error is what CE/softmax feel |
| CE loss | exact | — | never quantized (E5c) |
| gradients | float (default) or symlog with per-tensor pow2 absmax scale (`pure-int`) | (1) | pure-int = research flag |
| Lion momentum | 16-bit symlog, per-tensor pow2 absmax scale | 2 | |
| STE | 1 in range, **0 where the quantizer clamped** | — | mandatory |

New for the transformer (decided in M3/M4, recorded in the §7 log):
softmax probabilities (candidate: fixed-point u8 in [0,1], step 1/255),
RMSNorm gain/statistics (candidate: exact statistic, requant output),
embeddings (same as weights; tied with the output head).

## 3. Architecture of the code base after plan 001

```
plan/                 numbered plans (this is 001)
docs/DATA.md          corpus pipeline (no data in git)
docs/history/         frozen v0 plan + experiment log
include/ src/ tests/  C oracle: libintgrad, ig-train, ctest
tools/                build_shards.py, make_split.py (+ M1: index v2, eval freeze)
py/                   (M2) uv project `intgrad`
  intgrad/quant.py      grids, STE, w16 master weights   (parity with C)
  intgrad/lion.py       Lion with 16-bit momentum         (parity with C)
  intgrad/data.py       idx reader, random + sequential batchers
  intgrad/models/       mlp.py (parity vehicle), transformer.py
  intgrad/train.py      single entry point, config-file driven
  intgrad/eval.py       frozen-eval bpb, per-source, probes
  tests/                pytest (unit + parity against C golden files)
configs/              (M3) one TOML per experiment
runs/                 gitignored: <date>-<name>/{config, git hash, log, ckpt, samples}
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
A small C tool (`ig-golden`, M2) writes golden vectors: encode/decode over
a dense float sweep, w16 updates, Lion steps on fixed inputs, one MLP
forward/backward on fixed weights. PyTorch must reproduce **codes
exactly** and floats within 1 ulp-class tolerance. This turns the C engine
into the reference for every quantizer the transformer uses.

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

### M1 — Data path: index v2, batchers, frozen eval
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

### M2 — PyTorch reference engine + parity
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

Training: random batcher, Lion (lr/wd per E9 as the starting point),
warmup + cosine, fp32 master + TF32 matmuls in fp mode. Eval on eval-v1
every N steps. Samples are written to `runs/` only (D5).

Acceptance (fp mode): P2, P3, P4 pass at XS/S; the ladder is monotone at a
fixed byte budget; S beats xz and the 5-gram on eval-v1; throughput
measured and recorded (estimate: S ≥ 300 kB/s on the 4070 Ti).

### M4 — Quantized transformer (the core result of plan 001)
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
  are removed from the tree (git history keeps them).
- **001-D2 Engines**: PyTorch reference for anything scalable. The C scalar
  engine is the numerical oracle, bound to PyTorch by parity tests. C/CUDA
  int kernels are a later plan.
- **001-D3 Numerics**: §2. Old D6 (hybrid grids, clamp STE, exact CE) and
  D7 (2-byte master weights, 1-byte compute) carry over. 1-byte storage
  + SR remains selectable.
- **001-D4 Optimizer**: Lion only (old D8). Adam removed. Replaced only on
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

## 7. Pinned artifacts

| Artifact | Value |
|---|---|
| split | `data/split-v0` (blake2b content buckets, 1‰ val / 1‰ test), 10.85 M train docs, 56.5 GB |
| train byte shares (by source size class) | books 60.7%, wiki articles 32.9%, wiki defs 6.4% |
| eval-v1 hash | *(M1)* |
| unigram / 5-gram / xz / zstd on eval-v1 | *(M1)* |

## 8. Experiment log (plan 001)

Format: `X<n>` id, question, hypothesis, decision informed, config,
result (± SE, seeds), verdict. Empty at creation.

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
