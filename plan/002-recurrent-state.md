# Plan 002 — Recurrent state in every block

*Drafted 2026-11-19 as the M7 deliverable of plan 001; revised the same day
after review. Status: **active** (D1–D7 settled 2026-11-19). Plan 001 stays the reference for
numerics, data, eval and experiment rules (§4.5 there). Results go into §8
of this plan.*

## 0. Goal and scope

Give the plan 001 model a **recurrent state that is preserved across
context windows**: the state at the end of window t is the starting state
of window t+1 of the same stream.

**Scope of this phase (user decisions, 2026-11-19):**
- **Recurrence and state preservation only.** Memory access (caches, slots,
  retrieval, recall probes) is out of scope.
- **The recurrence sits in every transformer block**, as a branch next to
  attention, the way the q/k/v convolution was added (plan 001 X9). Special
  treatment of the mid-stack latent is deferred.
- **Mamba-style selective diagonal recurrence.** Nonlinear RNNs
  (LSTM/GRU) are not feasible on this hardware (strictly sequential per
  byte).
- **State reset at document boundaries, set by hand** (`doc_start`).
  Automatic resets (e.g. from perplexity) are a later topic.

Success is measured against the M7 baseline (plan 001 X33):

| measure (M7 base, flat L, 1024-byte windows) | bpb |
|---|---|
| eval-v1 (2 seeds) | 1.486 (1.487, 1.486) |
| first 64 bytes of continuation windows, cold | 1.863 |
| the same bytes with the previous 960 bytes as context (warm) | 1.399 |
| **cross-window gap** | **0.463 ± 0.005** (seed 43: 0.464) |

The recurrence **works** if the carried state closes a clear share of the
gap (≥ 50%, D6) without making eval-v1 worse.

## 1. Starting point (from plan 001)

- **Model:** flat L, `configs/m7-base.toml` (d 512, 8 layers, T 1024,
  q/k/v conv 3, lr 7.5e-4, wd 3, warmup 1000, 300 MB), fp. The hourglass
  is shelved (plan 001 X30).
- **Data:** `SequentialBatcher`, i.e. B document-continuous streams with a
  per-row `doc_start` mask. Resume is exact (plan 001 M1).
- **Metric:** `ig-position` (position curve and cross-window loss).

## 2. The recurrent branch

Each block gets a recurrent mixer next to attention, with its own pre-norm
and a residual add. Per channel c and byte t:

    a_t = exp(−softplus(Λ_c) · σ(W_a x_t))         input-dependent decay in (0, 1)
    h_t = a_t ⊙ h_{t−1} + sqrt(1 − a_t²) ⊙ (B_t x_t)
    y_t = (C_t h_t) ⊙ gate(x_t)

- **Mamba style:** the decay, input and output maps are input-dependent
  (selective).
- **Stable to train:** the decay is parametrized in log space, so values
  near 1 (long memory) stay trainable.
- **State dimension N per channel is an experiment variable:**
  - N = 1 (RG-LRU-like): state d per layer, 4 K numbers for L;
  - N = 16 (Mamba-1 style): 16× the state.
- **Training:** a parallel scan within the window in plain PyTorch (chunked
  for N > 1). A CUDA/Triton kernel only if speed or memory demands it.
- **Document reset:** `a_t = 0` and `h_{t−1}` dropped at `doc_start`. This
  works anywhere in a window, so no special case at window edges.
- **Window boundary:** the final h of every layer is handed to the next
  window of the same stream.

## 3. Training protocol

- **Unrolling, default:** full backprop through the scan within a window.
  The state is carried into the next window with the gradient cut at the
  boundary (k = 1). No extra memory.
- **Unrolling, experiment:** k = 2 (also backprop into the previous window).
  It shows whether gradient beyond the window pays for about 2× the
  activations.
- **Controls:**
  - (a) the base trained with the sequential batcher and no recurrence
    (sequential order changes the data stream against plan 001's random
    windows);
  - (b) the recurrent model with the state reset at every window (no
    carry). It separates the effect of the carried state from that of the
    extra branch.
- **Same budget:** 300 MB per run, base hyper-parameters. lr is re-checked
  only if a variant is unstable.
- **Seeds:** two seeds for any claimed difference below 0.05 (plan 001
  §4.5 rule 3).

## 4. Milestones

### N1 — Sequential control
Base with `SequentialBatcher`, no recurrence: eval-v1 and the cross-window
gap. These are the reference numbers for everything below.

### N2 — Recurrent branch, N = 1, every block
- Implementation: scan, doc reset, state hand-off across windows, and
  state in checkpoints (exact resume).
- `ig-position` with the carried state: each window is scored with the
  state from the previous one.
- Tests:
  - causality;
  - reset (a doc start erases all influence of earlier bytes);
  - carry (two windows with a carried state = one double-length pass);
  - each mutation-checked.
- Runs: carry vs no carry (control b).

### N3 — Unrolling and state size
Order agreed 2026-11-20: a second N2b seed first (D6), then N3a, then N3b.
- **N3a, state size:** N = 16 vs N = 1 (`model.rec_state`). A per-channel
  N-dim state through the linear scan would store [B, T, d, N] per layer
  (~16 GB at micro-batch 8), so N > 1 uses the chunked SSD form (Mamba-2):
  heads of P = 16 channels (`model.rec_head`), a P × N state per head,
  one decay per head (32 decay rates per layer instead of 512), L2-normed
  write/read vectors k, q. Same N2b protocol (rec_drop 0.1).
- **N3b, one nonlinear layer — shelved 2026-11-20** (see §8: no gain,
  62% throughput; code kept behind `model.rec_nonlin`): the middle block's recurrence replaced by
  a gated nonlinear one, h = a·h + b·tanh(W h + x), the other blocks stay
  linear (precedent: xLSTM's mostly-mLSTM stacks with a few sLSTM blocks).
  W dense first (pure PyTorch, sequential in that one layer); if it helps,
  restrict W to block-diagonal (blocks of 64) for a fused kernel. Control:
  the same layer linear with matched state size and parameters. Expected
  to show on structured text (nesting, quoting) more than on the gap.
- k = 2 vs k = 1.
- Long documents (books): does the state help beyond one window (the
  warm number)? Position curve over several windows.

### N3c — Capacity within 12 GB (before N4)
Agreed 2026-11-20: find the best fp model the card can train before
quantizing, so N4 has a best-case reference. 300 MB is about 8 bytes per
parameter at N2b's 36 M (plan 001 X23: a 90 M core got *worse* at
300 MB), so capacity is tested on more data.
1. N2b at 3× the data (900 MB, 27,465 steps, cosine over the full run):
   does the 36 M model gain from data alone?
2. Deeper N2b (d 512, 12 blocks, 54 M) on the same 900 MB. Depth before
   width: the per-layer lr (7.5e-4, tuned at d 512) should still hold.
   Width (d 768) only with an lr check.

VRAM and speed of N2b sizes (60-step probes, sequential batcher, rec_drop
0.1, fp32 master, compiled; params ≈ 17·L·d²):

| d × blocks | params | micro-batch | peak train | speed | 900 MB |
|---|---|---|---|---|---|
| 512 × 8 (N2b) | 36 M | 8 | 5.4 GiB | 58.5 kB/s | 4.3 h |
| 512 × 12 | 54 M | 8 | 8.0 GiB | 39.2 kB/s | 6.4 h |
| 512 × 16 | 71 M | 4 | 6.2 GiB | 31.2 kB/s | 8.0 h |
| 768 × 8 | 80 M | 8 | 8.4 GiB | 33.4 kB/s | 7.5 h |
| 768 × 12 | 120 M | 4 | 7.5 GiB | 22.5 kB/s | 11.1 h |
| 1024 × 8 | 143 M | 4 | 7.2 GiB | 21.6 kB/s | 11.6 h |

512 × 16 and 768 × 12 do not fit at micro-batch 8. Memory is not the
limit up to ~140 M; time is (speed scales with parameters). The eval
batch now shrinks with T·d (it ran out of memory at d 1024).

### N4 — Byte numerics
Quantized twin of the best N2/N3 variant (stage d, plan 001 M4). The decay
needs its own grid. Range telemetry of h, a_t and the gates.

## 5. Decisions (plan 002)

- **002-D1 Placement:** the recurrent branch is in every block. Mid-stack
  special treatment is deferred.
- **002-D2 Type:** Mamba-style selective diagonal linear recurrence, in log
  space. No nonlinear per-byte RNN.
- **002-D3 Reset:** at `doc_start`, by hand. Automatic reset is a later
  topic.
- **002-D4 Scope:** recurrence and state preservation only, no memory access.
- **002-D5 Baseline:** cross-window numbers are compared with the
  sequential control (N1) on the same windows (`ig-position`, seed 1234).

- **002-D6 Bar:** success = the carried state closes ≥ 50% of the
  cross-window gap (cold ≤ 1.63 at the M7 numbers; relative to N1's gap
  once measured), with no eval-v1 regression (user, 2026-11-19).
- **002-D7 Numerics:** fp until N4. The quantized twin runs only at N4,
  once the recurrence is shown to work (user, 2026-11-19).

## 7. Risks

- **A small state may not carry enough.** At N = 1 the state is 4 K
  numbers per stream. The N = 16 run shows whether size is the limit.
- **The decay under byte numerics:** 0.999 and 1.0 are the same on the
  byte grid. The decay must stay in log space or at higher precision.
  To be decided before N4.
- **The sequential batcher alone may shift eval-v1** (data order); N1
  isolates that.
- **Lion on a recurrent path:** sign updates on Λ can be large. Watch the
  decay distribution in telemetry, and consider a lower lr multiplier for
  the decay parameters (`lr_vec`-style).
- **Pure-PyTorch scan speed:** a chunked scan at T = 1024 may be slow; a
  kernel is then an engineering step, not a design change.

- **Reset is a state reset, not a document mask.** At `doc_start` the
  recurrent state is dropped, but attention inside the window still sees
  the bytes before the document start (as in the base, which trained on
  windows that cross documents). So in a multi-layer model, information
  from an earlier window can still reach a later layer through attention to
  pre-reset positions. If that matters, document-masked attention is the
  fix; the cross-window metric uses continuation windows only and is not
  affected.

## 8. Experiment log

### N1 — sequential control, 32 streams (2026-11-19)
M7 base, `data.batcher = "sequential"`, B = 32 streams, seed 42, 300 MB.

| measure | random windows (X33) | N1, 32 streams |
|---|---|---|
| eval-v1 | 1.487 | **1.577** |
| books / wiki-en / wiki-de / text | 1.513 / 1.423 / 1.479 / 1.611 | 1.602 / 1.485 / 1.552 / 1.910 |
| cold, first 64 B of continuation windows | 1.863 | 1.923 |
| warm | 1.399 | 1.461 |
| gap | 0.463 | 0.461 ± 0.005 |

- The gap is unchanged, but eval-v1 is 0.09 worse, at every eval step
  (0.13 at step 2000) and in every domain, most in "text" (+0.30).
- Cause (likely): 32 streams read whole documents, so a long document
  (a book of 1 MB = 1000 windows) holds its row for about 1000 steps.
  Consecutive batches are strongly correlated and share few documents.
- So N1 is not a usable control: any recurrent model trained this way
  inherits a 0.09 loss that has nothing to do with the state.

**Fix: a stream pool** (`data.streams = S`). S ≥ B streams; each round of
S/B steps permutes the streams and deals them out B at a time, so every
stream advances one window per round and consecutive batches come from
different documents. The trainer keeps one recurrent state per stream
([S, d] per layer, 16 MB at S = 1024) and gathers/scatters it by
`Batch.rows`. A stream's state is from weights at most 2·S/B steps old.
Tests: pool streams continuous, one advance per round, exact resume, and
the stored state of every stream equals a sequential pass over its
windows (lr 0; mutation-checked).

### N1b — pool control; N2 — state carried (2026-11-19)

| measure | N1b (pool, S 1024) | N2 carry: cold | N2 carry: carried |
|---|---|---|---|
| eval-v1 (zero state) | 1.505 | 1.521 | — |
| books / wiki-en / wiki-de / text | 1.566 / 1.405 / 1.449 / 1.633 | 1.606 / 1.394 / 1.439 / 1.628 | — |
| first 64 B of continuation windows | 1.839 (warm 1.378, gap 0.461) | 1.951 | **1.597** |
| whole windows (ig-position) | 1.459 | 1.459 | 1.408 |

- Carried, N2 closes (1.839 − 1.597) / 0.461 = **52%** of N1b's gap
  (bar 50%). ig-position prints 60%, against N2's own larger gap.
- **Cold, N2 is worse, and most on books** (+0.040 vs +0.016 overall).

**Why books (64 streams × 160 windows of val, by document age = windows
since the last document start):**

| age | N1b cold | N2 cold | N2 carried | N2 first 64 B: cold − N1b |
|---|---|---|---|---|
| 0 | 1.418 | 1.431 | 1.365 | +0.09 |
| 1–3 | 1.462 | 1.451 | 1.411 | +0.11 |
| 4–15 | 1.494 | 1.491 | 1.446 | +0.11 |
| 16–47 | 1.502 | 1.525 | 1.455 | +0.20 |
| 48–95 | 1.510 | 1.565 | 1.464 | +0.27 |
| ≥ 96 | 1.520 | 1.588 | 1.477 | +0.32 |

- **No drift:** the carried state helps more the deeper the window
  (carried − cold −0.05 at age 1–3, −0.11 at ≥ 96), its rms stays at
  0.077–0.079 over 160 windows, and carried N2 beats N1b at every age
  (−0.04 to −0.05).
- **Cold-start dependence:** in training the state is zero only at a
  document start. A zero state deep in a document never occurs, and the
  model leans on the state more the longer a document runs. eval-v1
  chunks of books are mostly deep inside a book, so books lose most.
- Not a defect of the recurrence, but a train/eval mismatch. Candidate
  fix (N2b): state dropout, i.e. zero the carried state of a share of rows
  per step (e.g. 10%), so mid-document cold starts are trained.

**N2 cost** (GPU probe, fp32, 16 × 1024 per pass): rec = lru is 1.46× the
time of the base (310 vs 212 ms) with a 9.6 GiB peak (base 6.2); params
36 M vs 25 M (d_rec = d adds 5d² per layer). Runs use micro_batch 8.

### N2 — no-carry control (2026-11-19)
Same recurrent model, state reset at every window (`train.rec_carry = false`).

| measure | N1b | N2 no carry | N2 carry |
|---|---|---|---|
| eval-v1 (zero state) | 1.505 | **1.491** | 1.521 |
| books | 1.566 | 1.549 | 1.606 |
| whole windows, cold / carried | 1.459 / — | 1.447 / 1.475 | 1.459 / **1.408** |
| first 64 B, cold / carried | 1.839 / — | 1.816 / 1.714 | 1.951 / **1.597** |
| gap (cold − warm) | 0.461 | 0.445 | 0.587 |

- **The branch alone helps:** −0.014 on eval-v1 and −0.012 on windows
  (11 M more parameters, 1.56× the time).
- **The carried state on top:** whole windows 1.408 vs 1.447 (−0.039);
  first 64 B 1.597 vs 1.816, i.e. 49% of the no-carry gap (52% of N1b's).
  At the bar, one seed.
- **Cost of carry training at zero state:** eval-v1 1.521 vs 1.491
  (+0.030), the cold-start dependence above.
- Given a carried state it was never trained with, the no-carry model
  still gains on the first 64 B (1.714) but loses on whole windows
  (1.475): the state is usable but not trained to be.

### N2b — carried state, 10% of states cleared per step (2026-11-20)
`train.rec_drop = 0.1`: each step, 10% of the rows start from a zero
state (random cold starts inside documents). Otherwise as N2 carry.

| measure | N1b | N2 no carry | N2 carry | **N2b** |
|---|---|---|---|---|
| eval-v1 (zero state) | 1.505 | 1.491 | 1.521 | **1.490** |
| books | 1.566 | 1.549 | 1.606 | 1.549 |
| whole windows, carried | — | — | 1.408 | 1.410 |
| first 64 B, cold / carried | 1.839 / — | 1.816 / — | 1.951 / 1.597 | 1.854 / **1.601** |
| gap closed, vs N1b | — | — | 52% | **52%** |

- **The cold-start loss is gone, the carried gain is kept:** eval-v1 1.490
  (= no carry, −0.015 vs N1b), books back to 1.549; carried windows and
  first 64 B within 0.004 of N2 carry.
- By document age (64 × 160 windows), N2b at zero state vs N1b: whole
  windows −0.017 to −0.021 from age 1 on (+0.008 at age 0); first 64 B
  +0.01 to +0.04 (N2 carry: +0.09 to +0.32). Carried vs zero state: −0.03
  per window at every age, −0.24 to −0.29 on the first 64 B. State rms
  0.084–0.090 over 160 windows, no drift.
- **D6:** 52% of the gap closed (bar 50%) and no eval-v1 regression: N2b
  passes, on one seed. The 50% margin is small (2 points), so plan §3 asks
  for a second seed before the claim is final.


### N2b — second seed (2026-11-20)
Seed 43, same config (cosine, rec_drop 0.1), position eval on the same val
windows (blake2b 4bef4490…). Gap closed is measured against N1b (seed 42).

| | N1b s42 | N2b s42 | N2b s43 |
|---|---|---|---|
| eval-v1 (zero state) | 1.505 | 1.490 | **1.489** |
| books | 1.566 | 1.549 | 1.551 |
| whole windows, zero state / carried | 1.459 / – | 1.448 / 1.410 | 1.444 / 1.406 |
| first 64 B, cold / carried | 1.839 / – | 1.854 / 1.601 | 1.848 / 1.586 |
| gap closed, vs N1b | – | 52% | **55%** |

- **D6 confirmed on two seeds:** 52% and 55% of N1b's gap closed (bar
  50%), eval-v1 1.490 / 1.489 vs N1b's 1.505. Seed spread is small:
  0.0005 on eval-v1, 3 points on gap closed.
- N1b exists on seed 42 only; both N2b seeds are measured against it.

### N3a — state N = 16 (SSD form) (2026-11-20)
N2b protocol (cosine, rec_drop 0.1, seed 42) with `model.rec_state=16`:
heads of 16 channels, a 16 × 16 state per head, one decay per head.

| | N2b s42 | N2b s43 | N3a s42 |
|---|---|---|---|
| eval-v1 (zero state) | 1.490 | 1.489 | 1.501 |
| books / text | 1.549 / 1.622 | 1.551 / 1.621 | 1.560 / 1.636 |
| whole windows, zero state / carried | 1.448 / 1.410 | 1.444 / 1.406 | 1.458 / 1.422 |
| first 64 B, carried | 1.601 | 1.586 | 1.605 |
| gap closed, vs N1b | 52% | 55% | 51% |
| throughput, peak memory | 58 kB/s, 5.4 GiB | 55 kB/s, 5.4 GiB | 50 kB/s, 6.6 GiB |

- **N = 16 does not help:** +0.011 on eval-v1 (seed spread 0.0005),
  worse on every source; the carried state closes no more of the gap.
  Quick val trails N2b s43 from step 1000 on (+0.03 to +0.05 early, about
  +0.012 at the end).
- 16× the state did not buy cross-window information, so the remaining
  gap is not (only) a state-capacity limit. Confound: the SSD form also
  shares one decay over 16 channels (32 decay rates per layer instead of
  512) and normalises k and q. Not separated by this run.

### N3b — nonlinear recurrence in the middle block (2026-11-20) — shelved
N2b protocol (cosine, rec_drop 0.1, seed 42) with `model.rec_nonlin="mid"`:
block 4 of 8 runs h = a·h + √(1−a²)·σ(i)·tanh(u + W h) with a dense,
zero-initialised 512 × 512 W; the other seven blocks are N2b.

| | N2b s42 | N2b s43 | N3b s42 |
|---|---|---|---|
| eval-v1 (zero state) | 1.490 | 1.489 | 1.495 |
| books / wiki-defs / text | 1.549 / 1.382 / 1.622 | 1.551 / 1.382 / 1.621 | 1.554 / 1.403 / 1.619 |
| whole windows, zero state / carried | 1.448 / 1.410 | 1.444 / 1.406 | 1.456 / 1.415 |
| first 64 B, carried | 1.601 | 1.586 | 1.601 |
| gap closed, vs N1b | 52% | 55% | 53% |
| throughput, peak memory | 58 kB/s, 5.4 GiB | 55 kB/s, 5.4 GiB | 36 kB/s, 5.2 GiB |

- **No gain:** +0.0055 on eval-v1 against the paired seed (seed spread
  0.0005); gap closed 53%, inside the N2b range. Only text is marginally
  better (−0.002); wiki-defs is +0.02 worse.
- W is used, not ignored: Frobenius norm 49, top singular value 40, 164
  singular values above 1. The block learnt a strongly nonlinear state
  map, which did not turn into lower loss or more cross-window carry at
  this scale and step budget.
- Quick val: ahead of N2b s43 early (−0.04 at step 1200), behind from
  step 4000 on (+0.005 to +0.01).
- Cost: the sequential loop runs at 62% of N2b's throughput; at equal
  wall clock the gap would be larger.
- Not run: the dense-W-without-tanh control, a second seed, the
  block-diagonal / kernel optimisations. With no gain at equal steps there
  is nothing for them to attribute or speed up.
- **Shelved.** Code stays (`rec_nonlin` defaults to "none", tested).
  Caveat: the expected benefit of nonlinear recurrence is state tracking
  (counting, nesting), which average bpb at 36 M parameters may not show;
  revisit only if a state-tracking probe is added. Scheduled: plan 003
  C6 reruns it on the new mixture (code, structured data).

### N2b — WSD schedule instead of cosine (2026-11-20)
N2b s42 with `train.sched="wsd"`, decay share 0.2, 1−√p shape (lr 7.5e-4
held from step 1000 to 7200, then decayed to 0 by 9000).

| | N2b s42 (cosine) | N2b s43 (cosine) | N2b s42 (WSD 0.2) |
|---|---|---|---|
| eval-v1 (zero state) | 1.490 | 1.489 | 1.533 |
| books / text | 1.549 / 1.622 | 1.551 / 1.621 | 1.582 / 1.684 |
| whole windows, zero state / carried | 1.448 / 1.410 | 1.444 / 1.406 | 1.503 / 1.469 |
| first 64 B, carried | 1.601 | 1.586 | 1.676 |
| gap closed, vs N1b | 52% | 55% | 47% |

- **WSD 0.2 is clearly worse:** +0.043 on eval-v1, worse on every source.
- The stable phase stalls: quick val 1.743 at step 6000, 1.738 at 7200
  (cosine fell from 1.644 to 1.564 over the same steps); +0.17 behind at
  the start of the decay.
- The decay recovers 0.21 in 1800 steps and is still falling steeply at
  the end (−0.02 per 200 steps): 20% decay is too short here. Likely
  reason: Lion's sign update has a fixed step size, so at constant lr the
  run sits on a noise floor and nearly all progress comes from annealing;
  decoupled weight decay (lr·wd) also stays at its peak.
- Cosine stays the default. WSD is only worth revisiting for runs that
  must be extendable, with a much longer decay share (≥ 0.5).

### N3c — capacity on 900 MB (2026-11-21)
N2b protocol (cosine, rec_drop 0.1, seed 42) on 900 MB (27,465 steps),
36 M (d 512 × 8) and 54 M (d 512 × 12).

| | N2b 300 MB | N2b 900 MB | N2b-L12 900 MB |
|---|---|---|---|
| eval-v1 (zero state) | 1.490 | 1.407 | **1.389** |
| books / wiki-en / wiki-de / text | 1.549 / 1.391 / 1.432 / 1.622 | 1.450 / 1.328 / 1.362 / 1.556 | 1.435 / 1.307 / 1.340 / 1.540 |
| whole windows, zero state / carried | 1.448 / 1.410 | 1.379 / 1.341 | 1.358 / 1.323 |
| first 64 B, cold / carried | 1.854 / 1.601 | 1.802 / 1.523 | 1.788 / 1.500 |
| gap closed (own cold-warm gap) | 52% | 55% | 56% |
| bytes per parameter | 8 | 25 | 17 |
| train time, peak memory | 1.4 h, 5.4 GiB | 4.3 h, 5.4 GiB | 6.4 h, 8.0 GiB |

- **Data is the main lever:** 3× the data gives −0.083 on eval-v1, on
  every source. The 36 M model is far from saturated at 300 MB.
- **Depth helps, less:** 12 blocks give a further −0.018, consistent on
  every source and at every eval point from step 12k on (−0.011 to
  −0.017), at 1.5× the time.
- At equal time the data lever looks stronger: by a log-linear guess
  (−0.083 per 3×), 1.5× more data at 36 M would be worth ≈ −0.03, more
  than the −0.018 from depth. Not measured yet.
- The recurrent state keeps working at scale: the carried state closes
  55–56% of each model's own gap (the gap itself grows, 0.489 → 0.516,
  because the warm context improves faster than the cold one).
