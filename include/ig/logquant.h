/* Log-domain byte quantization: every number is stored as ONE byte.
 *
 * Encoding (single uint8 "IGCode"):
 *   code 0            -> 0.0
 *   code 1..127       -> +(2^(code*DELTA) - 1) / SCALE
 *   code 129..255     -> -(2^((code-128)*DELTA) - 1) / SCALE
 *   code 128          -> reserved (never produced by ig_encode)
 *
 * DELTA=0.1 gives ~3.5% max relative error per value.
 * Multiplication is cheap in this domain: code(a*b) ~= code(a) + code(b)
 * (exact for pure log quantization, approximated here by the log(1+x)
 * "symlog" variant, which keeps zero representable and small values alive).
 *
 * SCALE shifts the representable window: values live in [~2^-? .. (2^12.7-1)/SCALE].
 * With SCALE=256: max |x| ~= 25.6, resolution near 0 ~= 1/(10*SCALE) relative.
 * Later milestones replace the global scale with per-tensor scales calibrated
 * from activation/gradient histograms.
 */
#ifndef IG_LOGQUANT_H
#define IG_LOGQUANT_H

#include <stdint.h>

#define IG_DELTA 0.1f
#define IG_BITS  8
#define IG_MAX_CODE 127

void     ig_set_scale(float scale);   /* value scale (default 256) */
float    ig_get_scale(void);
void     ig_set_gscale(float scale);  /* gradient scale (default 4096; D1 knob) */
float    ig_get_gscale(void);

uint8_t  ig_encode2(float x, float delta, float scale); /* generic grid; faithful nearest (tiny -> 0) */
uint8_t  ig_encode(float x);          /* float -> byte (value scale) */
float    ig_decode(uint8_t code);     /* byte  -> float (value scale) */
float    ig_roundtrip(float x);       /* decode(encode(x)); the simulated quantization error */

/* Explicit-scale variants: gradients use ig_get_gscale(), weights/acts the
 * value scale; per-tensor scales (M4+) pass their own scale here. */
uint8_t  ig_encode_s(float x, float scale);
float    ig_decode_s(uint8_t code, float scale);
float    ig_roundtrip_g(float g);     /* roundtrip at the gradient scale */

/* ---- Fixed-point byte grid (hybrid proposal, D6) --------------------
 * Two's-complement int8 on an absolute step: value = code * fxstep.
 * Constant ABSOLUTE error fxstep/2 — the right grid for bounded-range
 * tensors where relative error hurts: logits and the softmax/CE path
 * (log-grid absolute error there is ~DELTA*|logit| and floors CE loss).
 * With fxstep=0.1: range ±12.7, CE noise floor ~0.05 nats.
 */
void     ig_set_fxstep(float step);   /* default 0.1f */
float    ig_get_fxstep(void);
float    ig_fx_roundtrip(float x);    /* clamp+round onto the int8·step grid */

/* ---- E6 capacity probe: per-role grid resolution -------------------
 * delta is the log-grid's relative step (IG_DELTA default 0.1 ≈ 7.2%).
 * Halving delta doubles precision but halves the log2-reachable range at
 * 8 bits — a 16-bit value would get both. ig_rt2(x, delta, scale) is a
 * generic grid round-trip; roles carry their own knobs (defaults = the
 * standard byte grid). Setting w-delta≈IG_DELTA/4 emulates ~16-bit weight
 * storage while activations stay 8-bit (and vice versa).
 */
float ig_rt2(float x, float delta, float scale);
void  ig_set_role_grid(const char *role, float delta, float scale); /* "act"|"w" */
void  ig_get_role_grid(const char *role, float *delta, float *scale);
float ig_rt_act(float x);   /* forward-value grid round-trip */
float ig_rt_w(float x);     /* parameter-storage grid round-trip */
uint8_t ig_encode_w(float x); /* parameter byte code on the "w" role grid */
float ig_max_abs(float delta, float scale); /* largest representable |x| */

/* Stochastic rounding (E7): rounds to one of the two neighbouring grid
 * values with probability proportional to proximity, so E[rt(x)] = x.
 * u ~ U[0,1). Mandatory for parameter storage: Adam steps are ~0.1% of
 * |w| while the grid step is ~7%, so round-to-nearest discards nearly
 * every update (E7: 95-100% lost). */
float ig_rt2_sr(float x, float delta, float scale, float u);
float ig_rt_w_sr(float x, float u);

/* ---- Two-byte weights (E8, opt-in `--w2`) ---------------------------
 * A parameter's master value is a 16-bit log code wq = sign * M, with M
 * in units of w_delta/256 on the same symlog "w" grid (M <= 127*256).
 * High byte = round(M/256) = the ordinary 1-byte code: it is the ONLY
 * thing the forward/backward pass reads. Low byte = residual that absorbs
 * optimizer updates too small to move the high byte (Kahan-style master
 * weight, expressed in bytes; relative resolution 2^(0.1/256)-1 ≈ 0.027%).
 * Carry low->high is plain integer arithmetic on M.
 *   ig_w16_encode: u in [0,1) -> stochastic rounding, u < 0 -> nearest
 *   ig_w16_full:   full 16-bit value (optimizer view)
 *   ig_w16_fwd:    high-byte value (forward view) */
int16_t ig_w16_encode(float x, float u);
/* Same 16-bit symlog grid with explicit delta/scale (optimizer state). */
int16_t ig_enc16(float x, float u, float delta, float scale);
float   ig_dec16(int16_t c, float delta, float scale);
float   ig_w16_full(int16_t wq);
float   ig_w16_fwd(int16_t wq);

/* Log-domain integer helpers (the "fast path" of the future int kernel): */
uint8_t  ig_logmul(uint8_t a, uint8_t b);   /* magnitude-domain add with clamp */
uint8_t  ig_logadd(uint8_t a, uint8_t b);   /* via LUT-style log1p(exp()) identity */

#endif /* IG_LOGQUANT_H */
