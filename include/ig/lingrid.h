/* E-E1: hardware-linear quantization grids.
 *
 * The log byte grid (C1/D6) cannot be evaluated by any matrix engine we
 * have access to: tensor cores (cuBLASLt int8), Apple AMX (BNNS int8),
 * x86 AVX-VNNI (vpdpbusd) and ARM dotprod (usdot) all compute LINEAR
 * int8 dot products. Adopting a matrix engine therefore means adopting a
 * linear quantizer for the GEMM operands -- this module is the fake-quant
 * emulation of exactly that grid, so the bpb price of the switch can be
 * measured before any kernel exists.
 *
 * Semantics emulated (the hardware's, not the log engine's):
 *   - master weights live at full precision (the engine does NOT store
 *     weights quantized between steps -- D7's two-byte accumulator is a
 *     log-grid artifact and does not apply here);
 *   - the operand actually consumed is q(w) = clamp(round(w/s), -127, 127) * s,
 *     with s per output channel for weights (per-tensor = the weaker
 *     fallback some hardware offers) and per tensor for activations;
 *   - products/partials accumulate exactly (int32), requantized once at
 *     the output (unchanged from D6);
 *   - straight-through gradient: d q(w)/d w = 1 inside the range.
 *
 * Grids are attached per ROLE (weights / activations) so the log and
 * linear variants can be crossed against each other with everything else
 * in the stack (fx-grid logits, Lion, two-byte storage) held fixed.
 */
#ifndef IG_LINGRID_H
#define IG_LINGRID_H

enum {
    IG_WGRID_LOG = 0,  /* default: the log byte grid via ig_rt_w / wq high byte */
    IG_WGRID_LIN_PT,   /* linear int8, one scale per tensor              */
    IG_WGRID_LIN_PC    /* linear int8, one scale per output channel      */
};

enum {
    IG_AGRID_LOG = 0,  /* default: log byte activation grid              */
    IG_AGRID_LIN       /* linear int8, per-tensor scale from EMA absmax  */
};

extern int ig_wgrid_mode;
extern int ig_agrid_mode;

/* Per-step refresh of the weight scales from the current master weights.
 * ngroups lists (base,count) parameter groups in the trainer's group order
 * (W1,b1,W2,b2,W2b,b2b). chan_stride describes the row layout:
 *   chan_stride[g] > 0 -> rows of length chan_stride[g] share a scale
 *   chan_stride[g] == 0 -> one scale for the whole group (per-tensor)
 * Must be called after backward and before the optimizer touches the
 * params, so the scales stay constant across a whole step. */
void ig_lin_refresh_w(void *const *base, const long *count,
                      const long *chan_stride, int ngroups);

/* Forward-view quantization of the master weight at (group, index).
 * Group order is the trainer's: W1=0, b1=1, W2=2, b2=3, W2b=4, b2b=5. */
float ig_qw_gi(int gi, long idx);
/* The parameter's linear grid step (0 in log mode) for the subgrid stat. */
float ig_wq_step(int gi, long idx);

/* Activation quantization on the linear grid (per-tensor scale tracked as
 * an EMA of observed absmax per activation slot: 0 = hidden, 1 = hidden2). */
float ig_qa(int slot, float x);
/* Observe a tensor max to drive the per-tensor scale (calibration). */
void ig_aobserve(int slot, float tensor_absmax);
float ig_a_scale(int slot);

#endif /* IG_LINGRID_H */
