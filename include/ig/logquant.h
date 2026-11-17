/* Log-domain byte quantization: every number the network computes with is
 * ONE byte.
 *
 * Symlog code (uint8):
 *   code 0         -> 0.0
 *   code 1..127    -> +(2^(code*DELTA) - 1) / SCALE
 *   code 129..255  -> -(2^((code-128)*DELTA) - 1) / SCALE
 *   code 128       -> reserved (never produced)
 *
 * DELTA = 0.1: relative step 2^0.1 ≈ 7.2% (≤ 3.6% rounding error) in the
 * multiplicative regime; near zero the spacing is absolute ≈ 0.07/SCALE.
 * SCALE (default 256) positions the window: max |x| ≈ 25.6.
 *
 * Plus two companion grids:
 *   - fixed-point byte grid (int8 · fxstep) for logits (D6), and
 *   - a 16-bit refinement of the symlog grid (256 sub-steps per byte step)
 *     for two-byte master weights (D7) and the Lion momentum.
 */
#ifndef IG_LOGQUANT_H
#define IG_LOGQUANT_H

#include <stdint.h>

#define IG_DELTA 0.1f
#define IG_MAX_CODE 127

void  ig_set_scale(float scale); /* global value scale (default 256) */
float ig_get_scale(void);

/* Generic symlog grid with explicit delta/scale: faithful nearest code,
 * values below half the first step map to exact zero. */
uint8_t ig_encode2(float x, float delta, float scale);
float   ig_rt2(float x, float delta, float scale);
/* Stochastic rounding (unbiased in value space), u ~ U[0,1). */
float   ig_rt2_sr(float x, float delta, float scale, float u);
float   ig_max_abs(float delta, float scale); /* largest representable |x| */

/* DELTA grid at an explicit scale (gradients use per-tensor pow2 scales). */
uint8_t ig_encode_s(float x, float scale);
float   ig_decode_s(uint8_t code, float scale);

/* DELTA grid at the global value scale (weights, activations). */
uint8_t ig_encode(float x);
float   ig_decode(uint8_t code);
float   ig_roundtrip(float x);
float   ig_roundtrip_sr(float x, float u);

/* Fixed-point byte grid (D6): value = clamp(round(x/step), -128, 127)·step.
 * Constant absolute error — right for logits (CE error is absolute). */
void  ig_set_fxstep(float step); /* default 0.1 -> range ±12.7 */
float ig_get_fxstep(void);
float ig_fx_roundtrip(float x);

/* 16-bit symlog grid: code = sign · M, M in units of delta/256
 * (M ≤ 127·256). u < 0 -> round to nearest, u in [0,1) -> SR. */
int16_t ig_enc16(float x, float u, float delta, float scale);
float   ig_dec16(int16_t c, float delta, float scale);

/* Two-byte weights (D7) on the global grid. The high byte
 * round(M/256) is the ordinary 1-byte code and the ONLY thing the
 * forward/backward pass reads; the low byte accumulates sub-grid updates. */
int16_t ig_w16_encode(float x, float u);
float   ig_w16_full(int16_t wq); /* optimizer view (16 bit) */
float   ig_w16_fwd(int16_t wq);  /* forward view (high byte) */

#endif /* IG_LOGQUANT_H */
