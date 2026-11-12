# intgrad — Project Plan

A deep-learning framework in plain C, in the spirit of Karpathy's
*micrograd* (scalar autograd, everything explicit), with three hard
constraints and one goal:

| # | Constraint | Interpretation (see also §9 Clarifications) |
|---|-----------|----------------------------------------------|
| C1 | Only **integer bytes** as datatype | Every scalar the **forward/backward pass** touches (weight as read by the forward pass, activation, gradient, logit) is one byte (`uint8` log domain, or fixed-point byte for logits, D6). **Parameters are stored as two bytes** (D7, amended 2026-11-12): a 16-bit master log code whose high byte is the 1-byte weight the network computes with; the low byte only accumulates optimizer updates. Float exists only transiently inside the reference implementation; the target is pure-int kernels. |
| C2 | **No tokenizer** | The model consumes and emits raw UTF-8 bytes (alphabet = 256). |
| C3 | Hardware: **CPU, 128 GB RAM, RTX 4070 Ti (12 GB VRAM)** | Everything must fit/train on this box. |
| G1 | **Cognitive core** | A small LLM: coherent text + learned tool usage, minimal baked-in knowledge. |

---

## 1. Are the constraints consistent? (Honest assessment)

Mostly, with two real tensions we design around:

1. **Log-byte quantization vs. *training* stability.** INT8 *inference* is
   routine; end-to-end INT8 *training* is hard (optimizer state, small
   gradients, softmax denominators). Mitigations we adopt:
   - **Quant-aware training (QAT):** forward pass is passed through the byte
     round-trip at every op (implemented already), so the network is robust
     to the arithmetic it will actually run in.
   - **Log domain** rather than linear INT8: relative precision is uniform
     (~3.5% per value with our 7-bit magnitude, Δ=0.1), which handles the
     many-orders-of-magnitude dynamic range of activations/gradients — the
     exact failure mode of linear INT8.
   - **Pure-int training is the MAINLINE goal (Decision D1, 2026-07-27;
     parameter storage amended to 2 bytes by D7, 2026-11-12).**
     No float master weights by default: gradients live in the same
     log-byte domain (`--pure-int` flag, implemented), optimizer state is
     log-quantized, weight updates happen as additions in the log domain
     with **stochastic rounding** (required at our precision, else small
     gradients never move weights). Float fake-quant remains available
     (`--no-quant` / quant-off) purely as the diagnostic oracle.
     Strategy: instrument gradient flow (magnitudes, zero-collapse and
     saturation fractions per tensor, implemented as `ig_grad_stats`),
     adapt int scaling (per-tensor gradient scales, gradient clipping in
     the log domain, loss scaling as scale-shift which is free in this
     domain) — and only if a gate fails after these levers do we fall back
     to fp32 optimizer state (last resort, explicit sign-off needed).
   - Known research risks we carry: pure-int Adam moments are finicky
     (mitigation: log-domain moment estimates = EMA of log|g|, exact
     integer ops), stochastic-rounding variance (mitigation: higher-entropy
     rounding dither, batch-size increase). These get their own milestone
     gate (M5.5).
2. **Byte-level models need ~3–4× more compute/tokens than BPE models** for
   the same text quality (one word ≈ 3–5 bytes ≈ 3–5 prediction steps).
   For a *cognitive core* this is acceptable and even desirable: bytes give
   robustness to typos/UTF-8/code-mixing, a fixed 256 vocab (nice bonus:
   **logit distillation from any teacher becomes trivially alignable at the
   byte level**, see §5), and tool-protocol syntax (JSON-ish) is pure ASCII
   where byte models do fine.

Verdict: feasible for a **5–30 M parameter** model producing short coherent
text + reliable scripted tool-call behavior. Not feasible: competing with
7B+ models on open-world knowledge — which is precisely the "minimal
built-in knowledge, rely on tools" thesis.

---

## 2. Numerics design (C1)

Implemented in `include/ig/logquant.h` (reference form):

```
code 0        -> 0.0                     (zero is exactly representable)
code 1..127   -> +(2^(code·0.1) − 1)/S
code 129..255 -> −(2^((code−128)·0.1) − 1)/S
```

- **Multiplication** → add magnitudes, XOR signs (one LUT or one `iadd`).
- **Addition** → `log(eᵖ+eᑫ) = max + log1p(e^(−|p−q|))` → 128×128 LUT,
  precomputed at init, cache-resident (16 KB).
- **Scales**: global `S` for now; per-tensor scales (calibrated from
  histograms of a calibration run) in M4; per-channel for GEMM weights in
  M5. In pure-int mode, scales are restricted to powers of two (a shift),
  so nothing but ints exists at runtime; scale *choice* is offline
  calibration work, not a runtime datatype.
- **Nonlinearities** (tanh/softmax/RMSNorm): LUTs over the 256-code domain —
  every activation function becomes a 256-byte table. This is the secret
  weapon of byte-log arithmetic: activation cost = one table lookup.
- **Parameter storage uses stochastic rounding** (E7, implemented, default):
  the 7.2% grid step is ~70× an Adam step, so round-to-nearest discards
  ~98% of updates. Pure-int gradients use a per-tensor power-of-two
  absmax scale chosen every step (no saturation by construction).
  1-byte SR variance DOES floor long runs (E8: +0.15 bpb at 20k steps).
  **Two-byte weights are the default (D7)**: 16-bit master log code per
  parameter, forward/backward read only its high byte, the low byte is a
  residual that accumulates sub-grid updates; round-to-nearest on the
  16-bit grid (deterministic, no RNG in the update path). E8: +0.012 bpb
  vs fp at 20k steps (pure-int: +0.028). `--w2-sr` adds SR on the low
  byte, `--w1` restores 1-byte storage (with SR) for ablations.
- **Optimizer: Lion (default, D8)** — one momentum on a 16-bit log grid
  (per-tensor pow2 absmax scale), update lr·sign(·): int-native, no
  sqrt/division/eps. Defaults: lr 2e-3, β 0.9/0.99, wd 0.1, batch 32.
  Adam stays available (`--opt adam`; float moments, reference mode).
  Known cost (E9): ~0.1 bpb vs Adam at matched data on the M2 vehicle.
- **Dot products**: accumulate in a wider integer accumulator (int32) over a
  32-byte block, then requantize block sums. ("Integer bytes as datatype"
  is honored at tensor granularity; a carry accumulator is unavoidable in
  any integer dot product — clarification Q3.)

## 3. Model & scale targets (C2, C3)

| Phase | Model | Params | VRAM (train) | Notes |
|-------|-------|--------|--------------|-------|
| M2 | byte-MLP (in repo now) | ~0.2 M | <1 GB | autograd+quant smoke vehicle |
| M3–M4 | byte-transformer: d=256, L=6, heads=4, ctx=512, weight-tied emb | ~7 M | ~4–6 GB (Adam-float) / ~1.5 GB (int-state) | "coherent babble + protocol mimicry" |
| M5–M6 | byte-transformer: d=384, L=8, ctx=1024 | ~20–30 M | ~8–11 GB | the real cognitive core |
| M7 | optional MoE (8×small FFN, top-2) | ~60 M total / ~25 M active | fits | cheap capacity for tool-protocol variety |

Throughput estimates on 4070 Ti: float reference ~3–6 k tok/s (seq-par);
int8 tensor-core path (M5) 15–30 k Bbytes/s → 1 B bytes ≈ 1–3 days. Budget
plan assumes ≤ 2 B bytes total across pretrain + distill.

## 4. Training-data strategy

Principle: **narrow, high-quality, protocol-rich, legally clean.** A 10–20 M
model cannot memorize the web; we spend the budget where coherence and
tool-behavior live.

0. **Language curriculum (Decision D4)**: corpus v1 = Latin-script
   European languages (en, de, fr, es, it, pt, pl, nl, … — sized by
   available clean data, English largest). v2 adds Cyrillic, then CJK —
   these are *free* at the datatype level (bytes don't care) and cost only
   data + compute, no architecture change. The input conv stem (§4.6) is
   designed now so v2 needs no re-train from scratch (extend vocab of
   layers, keep stem).
1. **Coherence substrate (~300 M–1 G bytes)** — public-domain & permissive
   prose: Gutenberg books (public domain), Wikipedia dump (CC-BY-SA,
   abstracts subset), Python/JSON/Markdown from permissively-licensed GitHub
   slice (for structure), OSCAR/ Dolma subsets filtered to CC-BY. Dedup
   (MinHash), quality-filter (fastText classifier), length-normalize.
2. **Dialogue & instruction (~100–300 M bytes)** — OpenAssistant-style
   conversations, UltraChat/Infinity-Instruct (check licenses per set),
   translated/filtered to English + your target languages.
3. **Tool-use corpus (~50–200 M bytes, the crown jewels)** — synthesized:
   - Define a compact **wire protocol** in ASCII (M4): e.g.
     `call:get_weather{"city":"Paris"}` … `result:{...}` with reserved
     control markers (§9 Q4 — bytes 0x00–0x08 as in-stream control codes).
   - Use a big teacher model (local: Qwen/Llama class quantized via
     llama.cpp on the 4070 Ti or CPU+128 GB; or API) to generate thousands
     of multi-turn trajectories: task → call → result → answer, including
     **error paths** (bad args, tool unavailable, retry), parallel calls,
     and *refusals to call* when no tool is needed.
   - Augment with an existing agent corpus (ToolBench/ToolLLM-style traces,
     xLAM, Glaive function-calling sets — verify licenses).
   - Self-verification filter: replay every synthetic trajectory against a
     sandboxed mock-tool executor; discard trajectories whose results don't
     reproduce. This turns data generation into a **RL-free rejection
     sampling** loop and keeps labels honest.
4. **"Minimal knowledge" rule**: cap any single factual domain; explicitly
   *pin* only a small core (language, arithmetic tables, the tool schema,
   self-identity). Facts that a tool can serve (weather, current events)
   are deliberately **not** in the corpus, so the model learns to reach for
   tools instead of hallucinating — this is the actual cognitive-core
   mechanism and should be treated as a *data-instruction*: "if unsure and
   a tool can tell you, call the tool."
5. **Mixture & curriculum**: pretrain (1) → midtrain (1+2, rising dialogue
   share) → tool-SFT (3, heavy replay of 2 to avoid catastrophic forgetting)
   → optional preference-style fine-tune on rejection-sampled best traces.
   All at byte level: dataset pipeline is just `u8` streams + document
   boundaries + loss masks (loss on model turns only).
6. **Teacher infrastructure (Decision D3)**: local llama.cpp server
   `http://127.0.0.1:8180` (OpenAI-compatible). Current pool: Qwen3.8-Next
   (default generation teacher), Gemma4 12/26/31 B for cross-checking
   (judge / rejection sampling), EmbeddingGemma for dedup, clustering and
   near-duplicate exclusion. `tools/gen_traces.py` drives it; more models
   can be dropped into the server presets as needed. Invocation follows the
   repo's uv convention: `uv run tools/gen_traces.py` (PEP 723, stdlib-only).

### 4.6 Input conv stem for multibyte UTF-8 (D4 design answer)

Question: *small convolution at the start to handle multibyte sequences?*
**Yes — as a cheap ablatable causal-conv stem; it's the right call for our
size class.** Reasoning:

- A transformer has no privileged notion of a UTF-8 character: a 2–4 byte
  character is just a run of adjacent positions. Attention *can* learn it,
  but spends scarce depth/heads of a ≤30 M model on local composition.
  Bytes Under Attack (Chirkova & Vulić, EACL 2022) showed a char-level conv
  before byte-level T5 fixes most multibyte splicing damage; ByteLatent
  Transformer (2024) reaches the same conclusion via 4-byte patching.
- Our stem: width-3 causal depthwise conv → pointwise to d_model, log-int
  arithmetic, ~5–10 k params. It captures the UTF-8 lead/continuation
  pattern (0b110xxxxx followed by 0b10xxxxxx) and n-gram spelling in one
  layer, freeing attention for long-range structure. Dilated variants
  stay open for CJK-heavy v2.
- It is *also* what makes the language curriculum cheap: the stem is
  script-agnostic (operates on bytes), so v2 languages need no new stem.
- M3 acceptance includes a stem on/off ablation on Latin text and (added
  when v2 data arrives) on CJK bpb. If the ablation shows ≤0.01 bpb gain
  we drop it — the stem is engineering, not dogma.

Wire protocol (Decision D2 — control bytes):

- `0x00 ESC` + role byte inside the stream: `0x01` call-start, `0x02`
  args-separator, `0x03` result-ok, `0x04` result-err, `0x05` end-of-turn,
  `0x06/0x07` sys/user turn markers. Defined in `include/ig/protocol.h`
  and never appear in prose, so they cannot be spliced from training text.
- `ig-pretty` (tools/) translates streams ⇄ human-readable
  `⟪CALL⟫ … ⟪ARGS⟫ … ⟪RES⟫ …` markup for logs and debugging (D2: raw
  byte stream stays canonical, human-readability is a view).

## 5. Distillation — yes, and it's the plan's backbone

Can distillation speed up training? **Yes, dramatically, in three forms:**

1. **Response (black-box) distillation** — already in the data strategy:
   the teacher writes the SFT corpus. This is the single biggest win: the
   student doesn't pay for the teacher's web-scale pretraining to learn
   *behavior*, only to learn *capacity*.
2. **White-box logit KD — unusually clean here.** Because the student's
   vocabulary is exactly 256 bytes, any teacher's byte-level distribution
   can be recovered by marginalizing its subword distribution over
   continuations (sum probabilities of all teacher tokens starting with
   byte b — computable from the teacher's logits in one pass per step).
   Loss: `KL(teacher_bytes ‖ student_bytes)` + CE with the hard target.
   Expect 2–4× sample-efficiency on the SFT/tool phase; logit supervision
   per byte is ~4–5× denser than one-hot per byte.
3. **Feature/self distillation** for the quant constraint: train the fp32
   teacher (small, e.g. 30 M), then distill into the byte-quant student of
   the same or smaller size ("capacity gap" stays tiny — the regime where
   KD works best). This beats pure QAT by recovering accuracy lost to the
   log-byte grid, and it's cheaper than longer QAT runs.

Caveats (documented, not fatal): teacher bias/hallucination propagates —
mitigated by the trajectory-replay filter (§4.3); KD on prose saturates
fast, so use KD mostly for midtrain/SFT, plain CE for raw pretraining.

## 6. Compute plan on this box (C3)

- **M0–M4: pure C, CPU.** Scalar autograd is a *reference*, not a trainer.
  Real training uses the block kernel path: tensors as `u8` arrays,
  tiled int8 GEMM (accumulate int32) with OpenMP; 128 GB RAM means we hold
  fp32 master weights + Adam state + full dataset mmap'd without thought.
- **M5: CUDA backend** behind a thin `ig_backend` vtable (cpu | cuda).
  4070 Ti has AVX-unrelated but abundant int8 tensor cores (Ada, no
  fp8-training requirement). Training with our fake-quant scheme runs as
  fp16 GEMM w/ quantize/dequantize hooks first (simplest correct path),
  true IMMA kernels after numerical parity is proven.
- Parity harness (M5 gate): every kernel vs. the scalar-C reference,
  tolerance = byte-grid resolution.
- RAM-guarded data: mmap + 256 MB shard files (GPT-2 style), shuffle at
  shard granularity.

## 7. Milestones

| ID | Deliverable | Acceptance test | Est. |
|----|-------------|-----------------|------|
| **M0** ✅ | Scaffold: CMake+Ninja, layout, CI-able `ctest` | build+tests green | done |
| **M1** ✅ | Log-byte codec + micrograd-style autograd, quant-aware forward; gradcheck tests | `ctest` green; `--no-quant` vs log8 parity diff measurable | done |
| **M2** | Byte-MLP next-byte trainer (**in repo, hardening**): proper batched SGD→Adam, LR schedule, checkpoints | loss↓ on sample corpus; byte-level val bpb reported; generated text resembles corpus style | ~1 wk |
| **M3** | **Byte-transformer** (pre-LN/RMSNorm, RoPE, SwiGLU via LUTs, tied 256-way head) + **ablatable causal byte-conv stem** (§4.6), ctx 512, ~7 M params, on tensor core path (CPU int blocks + LUT activations) | val bpb beats M2-MLP by ≥15%; conv-stem ablation reported; 512-byte samples with paragraph-level coherence | 2–3 wk |
| M3.5 | **Numerics gate**: fp32-mode vs log8-mode bpb gap | gap ≤ 0.15 bpb after QAT fine-tune | +1 wk |
| **M4** | **Tool protocol + agent data pipeline**: wire format, mock-tool sandbox, teacher-trajectory generator + replay filter, loss-masking data format | 10 k verified trajectories; held-out trace-set (JSON validity 100%) | 2 wk |
| **M5** | **CUDA backend** + int8 block GEMM; parity harness; throughput ≥ 10 k Bbytes/s train | parity tests green; 24 h soak run stable | 3–4 wk |
| **M5.5** | **Pure-int training gate (mainline, D1)**: log-domain Adam + stochastic rounding + gradient-flow instrumentation; per-tensor gradient scales calibrated | pure-int run reaches ≥ 95% of fp-oracle val bpb at M3 scale; no NaN/divergence over 48 h soak; on every tensor, quantization destroys < 2% of the gradient L1 mass by zero-collapse and < 1% by saturation (magnitude-weighted, `--grad-stats` "mass"; count fractions are diagnostics only — E7) | 2–3 wk |
| **M6** | **Distillation runs**: fp32 teacher (≈30 M) → log8 student; byte-marginalized KD from a local 7–14 B llama.cpp teacher for SFT/tool phase | student ≥ 80% of teacher tool-call success rate; prose coherence eval (§8) ≥ 0.8 of teacher | 2–3 wk |
| **M7** | **Cognitive-core eval + tuning**: held-out tool benchmarks, refusal-to-hallucinate probes ("what's today's weather?" → must call tool), long-context needle test; optional MoE; optional pure-int-training experiment (research flag) | benchmark suite scripted; core behaviors ≥ target table agreed in M4 | 2–3 wk |
| **M8** | Release: `ig-run` inference REPL (byte stream in/out, tool bridge), quantized model card, repro `make demo` | demo runs on this box in <4 GB VRAM | 1 wk |

## 8. Evaluation (how we know it "works")

- **Coherence**: byte-per-char ratio, n-gram repetition rate, and LLM-judge
  (teacher as judge, 1–5 rubric) on 100 sampled continuations; tracked per run.
- **Tool use**: exact-match + schema-valid rate on held-out trajectories;
  multi-step tasks (2–4 calls, incl. one error-recovery path).
- **Anti-knowledge probes**: questions answerable only via tool → % that
  emit a call instead of a factual hallucination. This is the cognitive-core
  KPI, we track it from M4 onward.
- **Numerics**: bpb gap log8-vs-fp32 on a fixed val shard.

## 9. Decisions log (settled 2026-07-27) & remaining questions

- **D1 — Pure-int training is mainline.** No float master weights / fp32
  optimizer by default. We instrument gradient flow, tune int scaling
  (per-tensor power-of-two scales, log-domain clipping, stochastic
  rounding); fp32 optimizer state is the *last resort only*, requiring an
  explicit failed-gate sign-off. Gate: **M5.5**. Float mode stays in the
  codebase as the diagnostic oracle (`--no-quant`).
- **D2 — Control bytes for the tool protocol** (see §4.6 +
  `include/ig/protocol.h`). Human-readability via the `ig-pretty`
  translator; raw byte stream stays canonical.
- **D3 — Teacher is local only**: llama.cpp server at `127.0.0.1:8180`
  (Qwen3.8-Next active; Gemma4 pool; EmbeddingGemma for dedup).
- **D4 — Language curriculum**: Latin-family corpus first, then Cyrillic /
  CJK by adding data only. Input causal byte-conv stem included in M3
  (§4.6), with ablation evidence required to keep it.
- **D5 — Licensing**: code is **MIT** (`LICENSE`); training data is
  **research-only**, so we do *not* gate corpora on redistributable
  licenses. We still keep per-shard provenance manifests (debugging data
  bugs is 10× easier with provenance) and still prefer public-domain /
  permissive sources where quality is equal (they're cleaner and stable).

- **D6 — Hybrid quantization grids (user proposal, adopted).** Every value
  is still one byte (C1 holds); the *grid family* varies by tensor role:
  log-byte (Δ=0.1) for weights/activations where dynamic range matters,
  fixed-point byte (step 0.1, range ±12.7) for logits/CE where CE error
  scales with ABSOLUTE error — plus int32-style exact accumulation inside
  dot products with one requantization per result (matches real int8 GEMM
  semantics). Mandatory companion: **clamp-aware STE** — quantizers pass
  gradient 1:1 in range and 0 in the clamp region (E4: without this,
  optimizers freeze against saturated values; with it, full-quant
  memorizes to 0.0000 loss).
  AMENDMENT (E5c): CE is computed EXACT from quantized logits and is
  never itself quantized — both lse and target clamp on any grid means a
  confidently wrong prediction can score loss = 0 (the 0.09-bpb artifact).
  Error budgets attach to STORED tensors only, never to intermediates of
  a subtractive identity.

- **D7 — Two-byte parameter storage, C1 amended (user decision,
  2026-11-12).** Evidence E7/E8: on the 7.2%-step byte grid, round-to-
  nearest storage loses ~98% of updates; 1-byte stochastic rounding fixes
  short runs but drifts to +0.15 bpb vs fp over 20k steps (SR weight
  noise); a 16-bit master code whose high byte drives the forward pass
  stays within +0.01–0.03 (also pure-int). Ruling: C1 holds for
  everything the forward/backward pass computes with (1-byte weights,
  activations, gradients, logits); **parameter storage is 2 bytes**,
  the low byte being an update accumulator only. Default low-byte
  rounding is **round-to-nearest** (deterministic) — SR on the low byte
  was not needed in E8 (3.120 RTN vs 3.129 SR) and stays an option
  (`--w2-sr`). Rationale: keeps the byte-compute spirit of C1 in the
  network while letting training scale to M3 horizons.

- **D8 — Lion is the default optimizer; Adam is an option (user
  decision, 2026-11-12).** Rationale: Adam loses its elegance under int
  scaling (two moments, v spanning ~2× the octave range of g, sqrt,
  division, eps/zero-point handling); Lion keeps one state and a sign
  update whose int form is a sign test plus one log-code comparison. We
  can afford larger batches and re-tuned hyperparameters. **Accepted,
  documented cost (E9):** on the M2 MLP vehicle Lion trails Adam by
  ~0.10 bpb fp / ~0.12 quantized at matched data, at batch 32 and 128
  (≈ 1.5–2× data). Lion is quantization-robust (+0.015–0.04 quant gap with
  a 16-bit momentum; pure-int ≈ quant). Defaults: lr 2e-3, wd 0.1,
  batch 32, β 0.9/0.99, momentum `b2`; batch < 32 warns. Re-measure the
  Lion/Adam gap at M3 (transformer, realistic batch) and revisit if it
  stays ≳ 0.1 bpb.

Remaining minor questions (non-blocking, answer anytime):

- **R1** Max tool-result size before truncation/summarization policy
  (affects ctx=1024 budget)? Default plan: 512 bytes inline, else the tool
  returns a handle the model can `read(handle, offset)`.
- **R2** Is M5.5 (pure-int gate) a hard blocker for M6 distillation, or
  may distillation start against fp-oracle checkpoints in parallel
  (faster overall, slightly more rework if the gate fails)?
- **R3** Stochastic-rounding seed policy: fixed per-run for
  reproducibility (default) vs entropy?
- **A8** (unchanged) micrograd parity = scalar autograd kept forever as
  the numerical oracle; production training uses the batched path.

## 10. Top risks

1. **Pure-int training diverges or plateaus (D1 raises this from risk #1
   to the project's central bet).** Status (E7/E8): the M2-scale
   plateau was lost updates (fixed by SR) and SR variance (fixed by
   two-byte weights); pure-int + two-byte weights (D7 default) is within 0.03 bpb of fp over
   20k steps. Optimizer state: Lion (D8) keeps one 16-bit momentum, so the
   former "log-byte Adam moments" item is moot for the default path.
   Open: M3 scale. Levers in order: per-tensor power-of-2
   gradient scales → log-domain gradient clipping → stochastic-rounding
   dither tuning → larger effective batch → log-domain Adam (M5.5).
   Fallback of last resort (sign-off required): fp32 optimizer state,
   keep int weights/activations/gradients. The gradient-flow
   instrumentation (`ig_grad_stats`) exists specifically to tell us which
   lever to pull.
2. **Byte-level pretrain too slow on CPU-only phase** → start CUDA earlier
   (swap M5 before M4 if throughput < 2 k Bbytes/s on CPU).
3. **Teacher bias in tool traces** → replay filter catches most; keep a
   human-audited 500-trace golden set; cross-check with a second teacher
   family (Qwen vs Gemma4) on the golden set.
4. **Coherence ceiling** (~20 M model) → manage scope: short-form
   assistant behavior, not novels; MoE option in M7.
5. **Data-quality debt without license gates** → research-only (D5) lets
   us use more data but removes the "curated-source" free quality filter;
   counter with dedup (MinHash + EmbeddingGemma near-dup), classifier
   quality-filter, per-shard provenance manifests.
