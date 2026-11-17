#include "ig/logquant.h"

#include <math.h>

static float g_scale = 256.0f;
static float g_fxstep = 0.1f;

void ig_set_scale(float scale) { g_scale = scale; }
float ig_get_scale(void) { return g_scale; }

/* ---- generic symlog byte grid ---- */

uint8_t ig_encode2(float x, float delta, float scale)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0; /* 0 or NaN -> zero */
    float a = fabsf(x) * scale;
    int q = (int)lroundf(log2f(1.0f + a) / delta);
    if (q < 1) return 0; /* faithful: tiny values are 0, never inflated to code 1 */
    if (q > IG_MAX_CODE) q = IG_MAX_CODE;
    return (uint8_t)(x < 0.0f ? 128 + q : q);
}

static float dec2(uint8_t c, float delta, float scale)
{
    if ((c & 0x7Fu) == 0) return 0.0f; /* zero + reserved code 128 */
    float m = (exp2f((float)(c & 0x7Fu) * delta) - 1.0f) / scale;
    return (c & 0x80u) ? -m : m;
}

float ig_rt2(float x, float delta, float scale)
{
    return dec2(ig_encode2(x, delta, scale), delta, scale);
}

float ig_rt2_sr(float x, float delta, float scale, float u)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0.0f;
    float a = fabsf(x) * scale, r;
    int lo = (int)floorf(log2f(1.0f + a) / delta);
    if (lo >= IG_MAX_CODE) {
        r = exp2f((float)IG_MAX_CODE * delta) - 1.0f; /* clamp */
    } else {
        if (lo < 0) lo = 0;
        float vlo = exp2f((float)lo * delta) - 1.0f;
        float vhi = exp2f((float)(lo + 1) * delta) - 1.0f;
        r = u < (a - vlo) / (vhi - vlo) ? vhi : vlo; /* linear in value: unbiased */
    }
    r /= scale;
    return x < 0.0f ? -r : r;
}

float ig_max_abs(float delta, float scale)
{
    return (exp2f((float)IG_MAX_CODE * delta) - 1.0f) / scale;
}

uint8_t ig_encode_s(float x, float scale) { return ig_encode2(x, IG_DELTA, scale); }
float ig_decode_s(uint8_t code, float scale) { return dec2(code, IG_DELTA, scale); }

uint8_t ig_encode(float x) { return ig_encode_s(x, g_scale); }
float ig_decode(uint8_t code) { return ig_decode_s(code, g_scale); }
float ig_roundtrip(float x) { return ig_rt2(x, IG_DELTA, g_scale); }
float ig_roundtrip_sr(float x, float u) { return ig_rt2_sr(x, IG_DELTA, g_scale, u); }

/* ---- fixed-point byte grid ---- */

void ig_set_fxstep(float step) { g_fxstep = step; }
float ig_get_fxstep(void) { return g_fxstep; }

float ig_fx_roundtrip(float x)
{
    float q = x / g_fxstep;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (float)lroundf(q) * g_fxstep;
}

/* ---- 16-bit symlog grid ---- */

#define W16_SUB 256
#define W16_MAXM (IG_MAX_CODE * W16_SUB)

int16_t ig_enc16(float x, float u, float delta, float scale)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0;
    float a = fabsf(x) * scale;
    float d16 = delta / (float)W16_SUB;
    float qf = log2f(1.0f + a) / d16;
    int M;
    if (u < 0.0f) {
        M = (int)lroundf(qf);
    } else {
        int lo = (int)floorf(qf);
        if (lo < 0) lo = 0;
        float vlo = exp2f((float)lo * d16) - 1.0f;
        float vhi = exp2f((float)(lo + 1) * d16) - 1.0f;
        M = u < (a - vlo) / (vhi - vlo) ? lo + 1 : lo;
    }
    if (M > W16_MAXM) M = W16_MAXM;
    return (int16_t)(x < 0.0f ? -M : M);
}

float ig_dec16(int16_t wq, float delta, float scale)
{
    int M = wq < 0 ? -wq : wq;
    float m = (exp2f((float)M * delta / (float)W16_SUB) - 1.0f) / scale;
    return wq < 0 ? -m : m;
}

int16_t ig_w16_encode(float x, float u) { return ig_enc16(x, u, IG_DELTA, g_scale); }
float ig_w16_full(int16_t wq) { return ig_dec16(wq, IG_DELTA, g_scale); }

float ig_w16_fwd(int16_t wq)
{
    int M = wq < 0 ? -wq : wq;
    int hi = (M + W16_SUB / 2) / W16_SUB; /* integer: (M + 128) >> 8 */
    float m = (exp2f((float)hi * IG_DELTA) - 1.0f) / g_scale;
    return wq < 0 ? -m : m;
}
