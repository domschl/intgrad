/* Byte-level MLP reference vehicle: predicts the next UTF-8 byte from the
 * last CTX bytes (concatenated one-hot inputs), built on the scalar
 * autograd in value.h.
 *
 * Role since plan/001: the NUMERICAL ORACLE for the log-byte scheme
 * (bit-exact, slow, easy to audit). Scalable models live in the PyTorch
 * reference (plan/001 M2+); every quantizer there must match this code.
 *
 * Numerics (plan/001 §2, decisions D6/D7/D8 carried over):
 *   - forward/backward compute with 1-byte log-grid weights/activations,
 *     fixed-point byte logits, exact accumulation + one requant per output;
 *   - parameters are stored as a 16-bit master log code (two-byte, default)
 *     or as 1 byte with stochastic rounding (ig_w2 = 0);
 *   - optimizer: Lion, one momentum on a 16-bit log grid.
 */
#ifndef IG_NET_H
#define IG_NET_H

#include "ig/value.h"

#define IG_VOCAB 256
#define IG_NGROUPS 4 /* W1, b1, W2, b2 */

typedef struct {
    int ctx, nh;
    ig_Value **W1; /* [ctx * IG_VOCAB * nh] */
    ig_Value **b1; /* [nh] */
    ig_Value **W2; /* [nh * IG_VOCAB] */
    ig_Value **b2; /* [IG_VOCAB] */
} ig_MLP;

ig_MLP *ig_mlp_new(int ctx, int nh, unsigned seed);
void    ig_mlp_free(ig_MLP *m);
long    ig_mlp_nparams(const ig_MLP *m);

/* Parameter groups in canonical order (W1, b1, W2, b2): fills lists/ns,
 * returns IG_NGROUPS. Used by the optimizer, stats and checkpoints. */
int ig_mlp_groups(const ig_MLP *m, ig_Value **lists[IG_NGROUPS],
                  long ns[IG_NGROUPS]);

/* Forward on ctx bytes, returns malloc'd logits[IG_VOCAB] (graph nodes). */
ig_Value **ig_mlp_forward(ig_MLP *m, const uint8_t *in);
/* loss = CE(logits, target), computed exactly over quantized logits. */
ig_Value *ig_mlp_loss(ig_MLP *m, ig_Value **logits, uint8_t target);

void ig_mlp_zero_grad(ig_MLP *m);
void ig_mlp_scale_grads(ig_MLP *m, float s); /* for batch-mean loss */

/* Parameter storage (quantized mode only):
 *   ig_w2 = 1 (default): two-byte weights — 16-bit master log code
 *           (ig_Value.wq), forward reads its high byte; ig_w2_sr adds
 *           stochastic rounding of the low byte (default off: RTN).
 *   ig_w2 = 0: one-byte weights; ig_sr = 1 (default) stochastic rounding,
 *           ig_sr = 0 round-to-nearest (known-broken; kept only so the
 *           regression test keeps its teeth). */
extern int ig_w2;
extern int ig_w2_sr;
extern int ig_sr;

/* Lion (Chen et al. 2023): c = b1*m + (1-b1)*g; w -= lr*(sign(c) + wd*w);
 * m = b2*m + (1-b2)*g; b1 0.9, b2 0.99. Momentum lives on a 16-bit log
 * grid with a per-tensor pow2 absmax scale (float when quant is off). */
extern float ig_wd; /* decoupled weight decay, default 0.1 */
void ig_mlp_lion_update(ig_MLP *m, float lr);
/* Lion betas (default 0.9, 0.99; the transformer uses 0.95, 0.98, X7). */
void ig_set_lion_betas(float b1, float b2);
void ig_mlp_opt_reset(void);

/* Counters since the last reset over nonzero parameter steps (quant mode):
 * n steps, "subgrid" (below half the local grid step: lost under RTN),
 * and how many actually changed the stored code. */
void ig_mlp_update_stats(long *n, long *subgrid, long *moved, int reset);

/* Gradient-flow statistics for pure-int mode (M5.5-style gate metrics). */
typedef struct {
    long n;
    double mean_abs;
    float max_abs;
    float zero_frac;      /* nonzero g collapsed to 0 by quantization */
    float true_zero_frac; /* structurally-zero gradients */
    float sat_frac;       /* g clamped at the grid edge */
    float collapse_mass;  /* L1 mass share lost to collapse */
    float sat_mass;       /* L1 mass share lost to saturation */
} ig_GradStat;
ig_GradStat ig_mlp_grad_stats(const ig_MLP *m, int group);

/* Checkpoint (format v8): dims, scale, per param (fp32 value, 16-bit
 * master code), Lion momentum, steps done, data RNG, SR RNG. Atomic save
 * (tmp + rename); resume is bit-exact (ctest checkpoint). */
int     ig_mlp_save(const ig_MLP *m, const char *path, long steps_done,
                    uint64_t rng);
ig_MLP *ig_mlp_load(const char *path, long *steps_out, uint64_t *rng_out);

#endif /* IG_NET_H */
