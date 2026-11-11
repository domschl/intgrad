#include "ig/logquant.h"

#include <math.h>

static float g_scale = 256.0f;

void ig_set_scale(float scale) { g_scale = scale; }
float ig_get_scale(void) { return g_scale; }

uint8_t ig_encode(float x)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0; /* 0 or NaN -> zero */
    int neg = x < 0.0f;
    float a = fabsf(x) * g_scale;
    int q = (int)lroundf(log2f(1.0f + a) / IG_DELTA);
    if (q < 1) q = 1;
    if (q > IG_MAX_CODE) q = IG_MAX_CODE;
    return (uint8_t)(neg ? 128 + q : q);
}

float ig_decode(uint8_t code)
{
    if (code == 0 || code == 128) return 0.0f; /* zero + reserved code */
    if (code < 128) return (exp2f((float)code * IG_DELTA) - 1.0f) / g_scale;
    return -(exp2f((float)(code - 128) * IG_DELTA) - 1.0f) / g_scale;
}

float ig_roundtrip(float x) { return ig_decode(ig_encode(x)); }

uint8_t ig_logmul(uint8_t a, uint8_t b)
{
    /* Reference implementation. A real int kernel keeps a 128x128 LUT keyed
     * by magnitude codes and handles the sign with one XOR of sign bits. */
    return ig_encode(ig_decode(a) * ig_decode(b));
}

uint8_t ig_logadd(uint8_t a, uint8_t b)
{
    /* log-domain addition identity:
     *   log(e^p + e^q) = max(p,q) + log1p(exp(-|p-q|))
     * which becomes a table lookup once p,q are integer codes. */
    return ig_encode(ig_decode(a) + ig_decode(b));
}
