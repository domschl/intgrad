# Plan 002 — Recurrent state at the mid-stack latent (draft)

*Drafted 2026-11-19 as the M7 deliverable of plan 001. Status: **draft for
review**. Nothing here runs until the open questions in §6 are settled.
Plan 001 stays the reference for numerics, data, eval and experiment rules
(§4.5 there). Results go into §8 of this plan.*

## 0. Goal

Give the plan 001 model a **state that carries across context windows**,
attached at the **mid-stack latent** (between blocks L/2−1 and L/2,
`mid_hook`). This is the "recurrence and memory" direction of the
cognitive core (plan 001 §0). The state must be paid for with constant
memory per stream, whatever the document length.

Success is measured against the M7 baseline (plan 001 X33):

| measure (M7 base, flat L, 1024-byte windows) | bpb |
|---|---|
| eval-v1 | 1.487 |
| first 64 bytes of continuation windows, cold | 1.863 |
| the same bytes with the previous 960 bytes as context (warm) | 1.399 |
| **cross-window gap** | **0.463 ± 0.005** |

A recurrent state **works** if it closes a clear share of the gap (target:
≥ 50%, i.e. cold ≤ 1.63) without making eval-v1 worse. eval-v1 scores
chunks from in-chunk context only, so it measures the same cold model,
and it must not regress.

## 1. Starting point (from plan 001)

- **Model:** flat L, `configs/m7-base.toml` (d 512, 8 layers, T 1024,
  q/k/v conv 3, lr 7.5e-4, wd 3, warmup 1000, 300 MB). fp. The quantized
  twin cost +0.011 at L (X19), so byte numerics are re-checked at the
  milestones, not on every run.
- **Hourglass:** shelved (plan 001 X30). It may return if compute per
  position binds, e.g. for depth recurrence (R3).
- **Data:** `SequentialBatcher`, i.e. B document-continuous streams with a
  per-row `doc_start` mask; resume is exact (plan 001 M1).
- **Metric:** `ig-position` (position curve and cross-window loss).

## 2. Design space

All variants read from and write to the latent at L/2 and leave the byte
embedding, the head and the lower/upper stacks as in the base.

- **R1 — mid-layer window cache (Transformer-XL at one layer).** Block L/2
  also attends to the cached keys/values of the previous window's
  mid-stack latents, with gradient stopped. The cache costs T·d per
  stream. It is not a constant-size state, but it shows how much of the
  gap the mid-stack latent can carry at all. It is the ceiling for R2.
- **R2 — fixed-size recurrent state (memory slots).** M slots × d (e.g.
  M = 16–64) per stream:
  - **read:** block L/2 cross-attends to the slots;
  - **write:** at the window end, a slot update (cross-attention from the
    slots to the window's mid latents, gated residual) produces the next
    state;
  - **training:** BPTT through k consecutive windows (k = 2–4), detached
    beyond that.

  This is the real target: constant memory, learned compression.
- **R3 — recurrence in depth (later, or plan 003).** The middle blocks are
  applied n times with shared weights. This is a different axis (thinking
  longer rather than remembering longer). Its yardstick would be eval-v1
  and the probes (P2 copy, P3 MQAR), not the cross-window gap.

**State at document boundaries:** reset to the learned initial state at
`doc_start`, or carry it on? The base default is reset, since carried
state across unrelated documents is noise. A carried-state run tests
whether the model learns to ignore it.

## 3. Training protocol

- **Fair control first:** the base trained with the sequential batcher and
  no state. Plan 001 trained with random windows. Sequential order changes
  the data stream (whole documents, a different book/wiki mix per step),
  so all R-runs compare against this control, not against 1.487.
- **Same budget:** 300 MB per run, the base hyper-parameters; lr re-checked
  only if a variant is unstable.
- **TBPTT memory:** k windows of backprop multiply activations by k. At
  micro-batch 16 the base needs 6.25 GiB; k = 2 fits, and k = 4 needs
  micro-batch 8 (the math is unchanged).
- **Seeds:** two seeds for any claimed difference below 0.05 (plan 001
  §4.5 rule 3).

## 4. Milestones

### N1 — Sequential control
Train the base with `SequentialBatcher`, no state. Measure eval-v1 and
the cross-window gap. Deliverable: the reference numbers for N2–N4.

### N2 — R1: how much can the mid-stack latent carry?
Window cache at L/2 (stop-grad). Measures: share of the gap closed, cost
(VRAM, ms/step).
- If R1 closes little: the mid latent is the wrong place, or one layer is
  not enough. Then test a cache at every layer as an upper bound before
  going on.

### N3 — R2: fixed-size state
Slots M ∈ {16, 64}, BPTT k ∈ {2, 4}, reset at doc start vs carried.
Target: ≥ 50% of the gap at constant memory. Then the quantized twin of
the best R2 (byte numerics on the state path: range telemetry of slots
and gates).

### N4 — Longer than one window
Does the state help beyond the warm number? Position curve over several
windows of a long document (books): cold, warm (one window back), R2.
Hand-off: the base for memory/probe work (P3 MQAR across windows).

## 5. Decisions (plan 002)

- **002-D1 Interface:** the state attaches at the mid-stack latent (L/2).
  Variants at other layers are diagnostics, not candidates.
- **002-D2 Baseline:** cross-window numbers are compared with the
  sequential control (N1), with the same windows (`ig-position`, seed
  1234).
- **002-D3 Constant memory:** a candidate must have constant state per
  stream (R1 is a diagnostic ceiling only).

## 6. Open questions (for review)

1. Is the ≥ 50% gap target the right bar, or should success also need a
   probe (P3 MQAR with the keys in the previous window)?
2. R3 (depth recurrence) in this plan after N4, or a separate plan 003?
3. Reset at document boundaries as default (proposed), or carried state?
4. Should the quantized twin run at every milestone, or only at N3's
   winner (proposed)?

## 7. Risks

- **The gap may be mostly lexical** (names, exact phrases) and need far
  more than a few slots. R1 vs R2 shows that directly.
- **The sequential batcher alone may shift eval-v1** (data order); N1
  isolates that.
- **BPTT instability with Lion:** sign updates on a recurrent path can be
  large. Watch the gate and slot norms in telemetry; clamp or norm the
  state if needed.

## 8. Experiment log

*(empty)*
