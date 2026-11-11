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

uint8_t  ig_encode(float x);          /* float -> byte (value scale) */
float    ig_decode(uint8_t code);     /* byte  -> float (value scale) */
float    ig_roundtrip(float x);       /* decode(encode(x)); the simulated quantization error */

/* Explicit-scale variants: gradients use ig_get_gscale(), weights/acts the
 * value scale; per-tensor scales (M4+) pass their own scale here. */
uint8_t  ig_encode_s(float x, float scale);
float    ig_decode_s(uint8_t code, float scale);
float    ig_roundtrip_g(float g);     /* roundtrip at the gradient scale */

/* Log-domain integer helpers (the "fast path" of the future int kernel): */
uint8_t  ig_logmul(uint8_t a, uint8_t b);   /* magnitude-domain add with clamp */
uint8_t  ig_logadd(uint8_t a, uint8_t b);   /* via LUT-style log1p(exp()) identity */

#endif /* IG_LOGQUANT_H */
