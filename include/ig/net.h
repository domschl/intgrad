/* Byte-level MLP proof-of-concept: predicts next UTF-8 byte from the last
 * CTX bytes (concatenated one-hot inputs). This is the M2 milestone vehicle:
 * it exercises the autograd + log-quant path end-to-end before the
 * transformer arrives in M3.
 */
#ifndef IG_NET_H
#define IG_NET_H

#include "ig/value.h"

#define IG_VOCAB 256

typedef struct {
    int ctx, nh;
    int nh2;                /* optional second hidden layer width (E6), 0=off */
    ig_Value **W1; /* [ctx * IG_VOCAB * nh] */
    ig_Value **b1; /* [nh] */
    ig_Value **W2; /* [src * IG_VOCAB], src = nh2 ? nh2 : nh */
    ig_Value **b2; /* [IG_VOCAB] */
    ig_Value **W2b;/* [nh * nh2] (only when nh2 > 0) */
    ig_Value **b2b;/* [nh2] */
} ig_MLP;

ig_MLP *ig_mlp_new(int ctx, int nh, unsigned seed);
ig_MLP *ig_mlp_new2(int ctx, int nh, int nh2, unsigned seed);
void    ig_mlp_free(ig_MLP *m);

/* Forward on ctx bytes, returns malloc'd logits[IG_VOCAB] (graph nodes). */
ig_Value **ig_mlp_forward(ig_MLP *m, const uint8_t *in);

/* loss = CE(logits, target) built into the graph. */
ig_Value *ig_mlp_loss(ig_MLP *m, ig_Value **logits, uint8_t target);
extern int ig_loss_fp;    /* diagnostic: CE from stored values, no requant */
extern int ig_fp_logits;  /* diagnostic: float from the logits onward */
extern int ig_fp_params;  /* diagnostic: params stored float (no grid) */

/* Weight-update numerics (E7). ig_sr: stochastic rounding of 1-byte
 * parameter storage (ig_w2 == 0; default 1; 0 = round-to-nearest, which
 * discards ~98% of Adam updates on the byte grid). ig_gscale_dynamic: in pure-int mode, each
 * tensor's gradients use a per-step power-of-two absmax scale (default 1;
 * 0 = the fixed global ig_get_gscale()). */
extern int ig_sr;
extern int ig_gscale_dynamic;
/* Two-byte weights (E8; DEFAULT 1 per decision D7): each parameter keeps a
 * 16-bit master log code (ig_Value.wq); forward/backward read only its
 * high byte. ig_w2_sr (default 0) stochastically rounds the low byte;
 * by default the update path is deterministic (round-to-nearest on the
 * 16-bit grid — E8: as good as SR). ig_sr applies to 1-byte mode only. */
extern int ig_w2;
extern int ig_w2_sr;

/* Counters since the last reset, over nonzero parameter steps (quant mode
 * only): n steps, how many were below half the local grid spacing
 * ("subgrid": lost under round-to-nearest, SR-dependent otherwise), and
 * how many actually changed the stored value. */
void ig_mlp_update_stats(long *n, long *subgrid, long *moved, int reset);

void ig_mlp_zero_grad(ig_MLP *m);
void ig_mlp_sgd_update(ig_MLP *m, float lr); /* update + byte round-trip */
void ig_mlp_scale_grads(ig_MLP *m, float s); /* for batch-mean loss */

/* Adam (M2). Biases-corrected; single-MLP-per-process state (documented
 * limitation until the tensor path). Honors ig_pure_int (gradient byte
 * round-trip) and ig_quant_enabled (param grid round-trip). */
void ig_mlp_adam_update(ig_MLP *m, float lr);
void ig_mlp_adam_reset(void); /* resets ALL optimizer state (Adam + Lion) */

/* Lion (E9): one momentum tensor, update lr*sign(.) — see net.c. Same
 * conventions as Adam (ig_pure_int, ig_wd, two-byte weights). */
void ig_mlp_lion_update(ig_MLP *m, float lr);
#define IG_OPTQ_F32  0  /* float state (oracle) */
#define IG_OPTQ_B2   1  /* 16-bit log codes, RTN, per-tensor pow2 absmax scale */
#define IG_OPTQ_B1   2  /* 1-byte log codes, RTN (ablation: freezes) */
#define IG_OPTQ_B1SR 3  /* 1-byte log codes, stochastic rounding */
extern int ig_lion_m;   /* -1 = auto: B2 when quantized, F32 otherwise */
int ig_lion_m_mode(void);
/* Optimizer state currently held (e.g. after ig_mlp_load2): bit 0 = Adam
 * moments, bit 1 = Lion momentum. */
int ig_mlp_opt_state(void);
extern float ig_wd;     /* decoupled weight decay, default 0 */

/* Checkpoints: 'IGCK' + dims + scales + (fp32 value, byte code) per param
 * + Adam moments + run-progress state (steps done, RNG position) so a
 * resumed run continues the LR schedule and data order bit-exactly (v3).
 * steps_out/rng_out receive the stored progress (0 if absent). */
int     ig_mlp_save2(const ig_MLP *m, const char *path, long steps_done,
                    uint64_t rng); /* atomic: writes <path>.tmp then rename */
ig_MLP *ig_mlp_load2(const char *path, long *steps_out, uint64_t *rng_out);
#define ig_mlp_save(m, p) ig_mlp_save2((m), (p), 0, 0)
#define ig_mlp_load(p)    ig_mlp_load2((p), NULL, NULL)

/* Gradient-flow analysis (D1 instrumentation). Prints one line per param
 * group: n, mean|g|, max|g|, zero-collapse fraction (gradients that
 * quantize to 0 at the current gscale) and saturation fraction. */
typedef struct {
    long n;
    double mean_abs;
    float max_abs;
    float zero_frac;    /* nonzero g collapsed to 0 by quantization */
    float true_zero_frac; /* structurally-zero gradients (never touched) */
    float sat_frac;     /* g clamps at the grid edge */
    /* M5.5 gate metrics (magnitude-weighted): share of the tensor's
     * gradient L1 mass destroyed by quantization. Count fractions above
     * stay as diagnostics: collapsing many negligible softmax-tail
     * gradients is harmless, losing a little mass is not. */
    float collapse_mass; /* sum|g| of collapsed entries / sum|g| */
    float sat_mass;      /* sum(|g| - max_repr) over clamped / sum|g| */
} ig_GradStat;

ig_GradStat ig_mlp_grad_stats(const ig_MLP *m, int group); /* group 0..3 = W1,b1,W2,b2 */

#endif /* IG_NET_H */
