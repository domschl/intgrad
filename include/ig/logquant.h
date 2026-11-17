/* Log-domain byte quantization: every number the network computes with is
 * ONE byte.
 *
 * Symlog code (uint8):
 *   code 0         -> 0.0
 *   code 1..127    -> +DEC8[code] / scale
 *   code 129..255  -> -DEC8[code-128] / scale
 *   code 128       -> reserved (never produced)
 *
 * CANONICAL DEFINITION (shared bit-for-bit with py/intgrad/quant.py):
 * every grid is defined by float32 tables computed once in double,
 *     V(t) = (float)(exp2(t * 0.1 / 256.0) - 1.0),   t in 1/256 grid steps
 *     DEC8[c]  = V(256 c)            c = 0..127      byte grid values
 *     THR8[k]  = V(256 k - 128)      k = 1..127      byte rounding thresholds
 *     DEC16[M] = V(M)                M = 0..32512    16-bit grid values
 *     THR16[k] = V(k - 0.5)          k = 1..32512    16-bit thresholds
 * Encoding counts thresholds <= |x|*scale (round half up); no libm call at
 * encode time, so the result does not depend on log2/exp2 implementations.
 * Scales are powers of two, so |x|*scale and v/scale are exact.
 *
 * DELTA = 0.1: relative step 2^0.1 ≈ 7.2% in the multiplicative regime;
 * near zero the spacing is absolute ≈ 0.07/scale. Default value scale 256
 * (max |x| ≈ 25.6).
 */
#ifndef IG_LOGQUANT_H
#define IG_LOGQUANT_H

#include <stdint.h>

#define IG_DELTA 0.1f
#define IG_MAX_CODE 127
#define IG_W16_SUB 256
#define IG_W16_MAXM (IG_MAX_CODE * IG_W16_SUB)

void  ig_set_scale(float scale); /* global value scale (default 256, pow2) */
float ig_get_scale(void);

/* Byte grid at an explicit (power-of-two) scale. */
uint8_t ig_encode_s(float x, float scale);
float   ig_decode_s(uint8_t code, float scale);
float   ig_roundtrip_s(float x, float scale);
/* Stochastic rounding, unbiased in value space; u ~ U[0,1). */
float   ig_roundtrip_sr_s(float x, float scale, float u);
float   ig_max_abs(float scale);  /* largest representable |x| */

/* Byte grid at the global value scale (weights, activations). */
uint8_t ig_encode(float x);
float   ig_decode(uint8_t code);
float   ig_roundtrip(float x);
float   ig_roundtrip_sr(float x, float u);

/* Largest power of two s with mx * s <= max representable magnitude at
 * scale 1 (exact, via frexp); 1 if mx is not > 0. Per-tensor absmax scale
 * for gradients and optimizer state. */
float ig_pow2_scale(float mx);

/* Fixed-point byte grid: value = clamp(round(x/step), -128, 127) * step,
 * round half away from zero. Constant absolute error (logits). */
void  ig_set_fxstep(float step); /* default 0.1 -> range ±12.7 */
float ig_get_fxstep(void);
float ig_fx_roundtrip(float x);
/* max-relative fixed point of one softmax row (n values), 001-D12 */
void  ig_fx_maxrel(const float *x, float *out, int n);

/* 16-bit symlog grid: code = sign * M, M in units of DELTA/256.
 * u < 0 -> round to nearest (half up), u in [0,1) -> SR. */
int16_t ig_enc16(float x, float u, float scale);
float   ig_dec16(int16_t c, float scale);

/* Two-byte weights at the global scale. The high byte (M + 128) >> 8 is
 * the ordinary 1-byte code and the ONLY thing forward/backward read; the
 * low byte accumulates sub-grid updates. */
int16_t ig_w16_encode(float x, float u);
float   ig_w16_full(int16_t wq); /* optimizer view (16 bit) */
float   ig_w16_fwd(int16_t wq);  /* forward view (high byte) */

/* Canonical tables (for parity tests): 0 DEC8, 1 THR8, 2 DEC16, 3 THR16.
 * THR tables are indexed by k (entry 0 unused, 0.0). */
const float *ig_lq_table(int which, int *n);

#endif /* IG_LOGQUANT_H */
