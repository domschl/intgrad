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
| B4 | `idx.bin` stores **absolute shard paths** | Index breaks if the repo moves | **fixed (M1)**: index v2, relative paths + source id |
| B5 | Documents are joined by `"\n \n"` in shards, but the model **never sees a document boundary** (windows stay inside one doc) | Needed for sequential streaming / recurrence | **fixed (M1)**: batchers insert EOD `0x03` |
| B6 | Scalar autograd: ~440 windows/s on a 120k-param MLP | A transformer cannot be trained on it | D2: PyTorch reference engine |
| B7 | Experiment harness: single seeds, no frozen eval set, fp-oracle runs not paired, flags added per experiment and never removed (≈30 flags) | Conclusions flipped three times (E1–E6 invalidated by E7, E10 by B1, all numbers by B2/B3) | rules in §4.5 |
| B8 | Untracked-but-committed files under `data/` (synthetic traces), fixtures in `data/` | Violates the "data/ is private" rule | **fixed**: `data/` fully ignored, fixtures in `tests/fixtures/` |
| B9 | Checkpoint format accreted v1–v7 compat branches + Adam moments | Complexity | **fixed**: single format v8 |
| B11 | **Gutenberg books duplicated**: 27 k of 50 k book ids exist as 2–4 encoding variants (`N`, `N-8`, `N-0`), and exact-hash dedup removed only 362 docs; variants also straddled the split | Books double/triple weighted, val leakage | **fixed (M1, X1)**: variant selection + near-dup fingerprints |
| B12 | **Gutenberg boilerplate** in 95% of book docs (license headers in old formats, "End of Project Gutenberg's …" lines unmatched, credits, transcriber notes) | Training on license text | **fixed (M1, X1)**: `intgrad.corpus.gutenberg` |
| B10 | Possible near-duplicate leakage between corpus-v0 wiki-2017 definitions and corpus-v1 wiki-2015 articles (split is by exact content hash) | Optimistic val bpb on wiki | **found and fixed (M1, X2)**: 30% of val articles had their lead in a train definition; the split now groups English wiki by title |

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
py/                   uv project `intgrad` (M1: corpus pipeline + data; M2: engine)
  intgrad/corpus/       build_shards, make_split (index v2), gutenberg (cleaner)
  intgrad/data.py       index v2 reader, random + sequential batchers
  intgrad/eval_set.py   eval-v1 freeze, baselines (unigram, KN-5, xz, zstd)
  intgrad/quant.py      grids, STE, w16 master weights   (parity with C)
  intgrad/lion.py       Lion with 16-bit momentum         (parity with C)
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
