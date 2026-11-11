# intgrad — Project Plan

A deep-learning framework in plain C, in the spirit of Karpathy's
*micrograd* (scalar autograd, everything explicit), with three hard
constraints and one goal:

| # | Constraint | Interpretation (see also §9 Clarifications) |
|---|-----------|----------------------------------------------|
| C1 | Only **integer bytes** as datatype | Every scalar (weight, activation, gradient) is one `uint8` in a **logarithmic domain**. Float exists only transiently inside the reference implementation; the target is pure-int kernels. |
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
   - **Float master weights** on the outside (128 GB RAM makes this cheap):
     the SGD/Adam update accumulates gradients in float and rounds the
     *stored* weight through the byte grid. This is standard "fake-quant"
     QAT. If you insist on *no float anywhere*, we switch to
     stochastic-rounding + log-domain optimizer state in a later milestone
     (flagged as research-risky, milestone M5).
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
  histograms of a calibration run) in M4; per-channel for GEMM weights in M5.
- **Nonlinearities** (tanh/softmax/RMSNorm): LUTs over the 256-code domain —
  every activation function becomes a 256-byte table. This is the secret
  weapon of byte-log arithmetic: activation cost = one table lookup.
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
| **M3** | **Byte-transformer** (pre-LN/RMSNorm, RoPE, SwiGLU via LUTs, tied 256-way head), ctx 512, ~7 M params, on tensor core path (CPU int blocks + LUT activations) | val bpb beats M2-MLP by ≥15%; 512-byte samples with paragraph-level coherence | 2–3 wk |
| M3.5 | **Numerics gate**: fp32-mode vs log8-mode bpb gap | gap ≤ 0.15 bpb after QAT fine-tune | +1 wk |
| **M4** | **Tool protocol + agent data pipeline**: wire format, mock-tool sandbox, teacher-trajectory generator + replay filter, loss-masking data format | 10 k verified trajectories; held-out trace-set (JSON validity 100%) | 2 wk |
| **M5** | **CUDA backend** + int8 block GEMM; parity harness; throughput ≥ 10 k Bbytes/s train | parity tests green; 24 h soak run stable | 3–4 wk |
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

## 9. Clarifications & open questions (⚠ = need your answer)

Resolved ambiguities (assumptions made, reversible):

- **A1** "Integer bytes" = every *tensor element* is 1 byte; dot products
  necessarily use a wider int accumulator per block, and (assumption
  **⚠Q1**) optimizer state + master weights may live in fp32 ("fake-quant
  QAT"). Pure-int optimizer (no float anywhere, incl. updates) is scheduled
  as opt-in research milestone M7-R because INT8-state Adam is known to be
  finicky at small scale.
- **A2** Log scheme is symmetric log(1+x) (keeps exact zero + sign). A pure
  log-domain (zero = special code) variant with better mul accuracy is an
  M3 alternative; we benchmark both.
- **A3** "No tokenizer" applies to the model, not to humans: an inference
  CLI may pretty-print UTF-8. Byte-level also removes the *detokenizer*
  failure class (splices, mojibake).
- **A4** Tool interface = ASCII wire protocol embedded in the byte stream
  with reserved control bytes (0x00 = start-call, 0x01 = result, 0x02 =
  end-of-turn). They're valid UTF-8 but never occur in prose — **⚠Q2**: do
  you prefer pure-printable marker sequences (e.g. `⟨⟨call …⟩⟩`) for
  human-readable traces? Trade-off: markers can be spliced/half-emitted;
  control bytes can't but are invisible in logs.
- **A5** English-first core; CJK etc. naturally supported by the byte
  alphabet, but coherence quality follows data volume per script. **⚠Q3**:
  target languages for the core corpus?
- **A6** Teacher models: any open model we can run locally
  (llama.cpp / Ollama on this box; 128 GB RAM even allows big CPU teachers
  for data-gen), plus **⚠Q4**: do you also have API access you want used
  for the higher-quality trajectory generation?
- **A7** Licenses: plan assumes training on public-domain/permissive sets;
  if this is research-only, more sources open up. **⚠Q5**: is this a
  research or redistributable-product project?
- **A8** Karpathy-micrograd parity means: scalar autograd reference kept
  forever as the *numerical oracle*; production training obviously uses the
  batched path. (No expectation that scalar autograd itself trains the LLM.)

## 10. Top risks

1. **Log-byte quant during training diverges/plateaus** → M3.5 gate;
   fallbacks: per-channel scales → STE variants → fp32-train + quant-infer
   only (violates C1's spirit; last resort, flagged).
2. **Byte-level pretrain too slow on CPU-only phase** → start CUDA earlier
   (swap M5 before M4 if throughput < 2 k Bbytes/s on CPU).
3. **Teacher bias in tool traces** → replay filter catches most; keep a
   human-audited 500-trace golden set.
4. **Coherence ceiling** (~20 M model) → manage scope: short-form
   assistant behavior, not novels; MoE option in M7.
5. **Dataset license creep** → keep per-shard provenance manifest from day 1.
