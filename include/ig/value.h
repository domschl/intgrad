/* Micrograd-style scalar autograd, with quantization applied at every op
 * output (quant-aware training): values are floats in the host code, but
 * each forward result is passed through the byte round-trip so the network
 * learns to be robust to its own integer arithmetic.
 *
 * Gradients are accumulated in float; in pure-int mode they are quantized
 * before the optimizer sees them.
 */
#ifndef IG_VALUE_H
#define IG_VALUE_H

#include <stdint.h>

typedef struct ig_Value {
    float data;
    float grad;
    struct ig_Value **children;
    int nchildren;
    void (*backprop)(struct ig_Value *self); /* NULL for leaves */
    int _seen;                               /* internal: topo-pass marker */
    /* Quantization was CLAMPED at the grid edge: d(clamp)/dx = 0 there,
     * so backprop through this node must be blocked. Without this the
     * optimizer pushes gradient into an immovable stored value (Momentum
     * accumulates against a wall — the E4 stuck-loss floor). */
    unsigned clamped : 1;
    /* Parameters only, two-byte-weight mode (E8): 16-bit master log code;
     * `data` is always its high-byte (forward) value. 0 with data != 0
     * means "not yet initialized from data". */
    int16_t wq;
} ig_Value;

/* Toggle quantization on/off (off = exact float mode, used for gradient checks). */
extern int ig_quant_enabled;

/* D1: pure-int mode. Gradients are quantized to the log-byte grid (with a
 * per-tensor power-of-two absmax scale) before they touch the weights, so the whole
 * value->gradient->update path is byte-representable. Reference mode: the
 * float accumulation inside this file emulates an int accumulator; the
 * true int-domain accumulator arrives with the tensor path (M5.5). */
extern int ig_pure_int;

/* D6 hybrid grid around bounded-range paths (logits/CE):
 *   0 = log grid (default, everywhere else)
 *   1 = fixed-point byte grid per op (ig_fx_roundtrip)
 *   2 = wide accumulator pass-through: op outputs stay exact, emulating an
 *       int32 GEMM accumulator; the caller requantizes once at the end. */
extern int ig_fx_mode;

/* Graph arena. ig_reset() frees ALL intermediate values; parameters live
 * outside the arena (ig_value_persist). */
void        ig_reset(void);                 /* free arena intermediates */
ig_Value   *ig_value(float x);              /* leaf, arena-allocated */
ig_Value   *ig_value_persist(float x);      /* leaf, malloc (for parameters) */
void        ig_value_free_persist(ig_Value *v);

void        ig_backward(ig_Value *loss);    /* topological reverse-mode AD */

/* Ops (all quantize their output when ig_quant_enabled): */
ig_Value   *ig_add(ig_Value *a, ig_Value *b);
ig_Value   *ig_sub(ig_Value *a, ig_Value *b);
ig_Value   *ig_mul(ig_Value *a, ig_Value *b);
ig_Value   *ig_tanh(ig_Value *a);
ig_Value   *ig_relu(ig_Value *a);
ig_Value   *ig_logsumexp(const ig_Value **xs, int n);

/* Quantize a (typically exact, ig_fx_mode==2) accumulator exactly once
 * onto the given grid, with clamp-aware STE. Identity when quant is off. */
#define IG_GRID_ACT 0   /* log-byte activation grid */
#define IG_GRID_FX  1   /* fixed-point byte grid (logits) */
ig_Value   *ig_requant(ig_Value *a, int grid);

#endif /* IG_VALUE_H */
