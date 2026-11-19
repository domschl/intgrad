# Plan 002 — Recurrent state in every block (draft)

*Drafted 2026-11-19 as the M7 deliverable of plan 001; revised the same day
after review. Status: **draft for review**. Plan 001 stays the reference for
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
- k = 2 vs k = 1.
- N = 16 vs N = 1.
- Long documents (books): does the state help beyond one window (the
  warm number)? Position curve over several windows.

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

## 8. Experiment log

*(empty)*
