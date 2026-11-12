#include "ig/logquant.h"

#include <math.h>
#include <string.h>

static float g_scale = 256.0f;
/* role grids: "act" (forward values) and "w" (parameter storage) */
static float g_act_delta = IG_DELTA, g_act_scale = 256.0f;
static float g_w_delta = IG_DELTA, g_w_scale = 256.0f;
static float g_gscale = 4096.0f;

void ig_set_scale(float scale)
{
    g_scale = scale;
    /* --scale is the global value-scale knob (E1/E2 semantics): it moves
     * both role grids unless --w-scale/--a-scale override afterwards. */
    g_act_scale = scale;
    g_w_scale = scale;
}
float ig_get_scale(void) { return g_scale; }
void ig_set_gscale(float scale) { g_gscale = scale; }
float ig_get_gscale(void) { return g_gscale; }

/* Faithful nearest-code encoder on an arbitrary log grid. Values below
 * half the first step round to code 0 (exact zero). The old q>=1 floor
 * inflated tiny gradients by orders of magnitude (1e-10 -> 1e-6 at gscale
 * 2^16), which Adam (eps=1e-8) turned into full-size steps; it also made
 * the zero-collapse statistic structurally always 0. */
uint8_t ig_encode2(float x, float delta, float scale)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0; /* 0 or NaN -> zero */
    int neg = x < 0.0f;
    float a = fabsf(x) * scale;
    int q = (int)lroundf(log2f(1.0f + a) / delta);
    if (q < 1) return 0;
    if (q > IG_MAX_CODE) q = IG_MAX_CODE;
    return (uint8_t)(neg ? 128 + q : q);
}

uint8_t ig_encode_s(float x, float scale) { return ig_encode2(x, IG_DELTA, scale); }

float ig_decode_s(uint8_t code, float scale)
{
    if (code == 0 || code == 128) return 0.0f; /* zero + reserved code */
    if (code < 128) return (exp2f((float)code * IG_DELTA) - 1.0f) / scale;
    return -(exp2f((float)(code - 128) * IG_DELTA) - 1.0f) / scale;
}

uint8_t ig_encode(float x) { return ig_encode_s(x, g_scale); }
float ig_decode(uint8_t code) { return ig_decode_s(code, g_scale); }
float ig_roundtrip(float x) { return ig_decode_s(ig_encode_s(x, g_scale), g_scale); }
float ig_roundtrip_g(float g) { return ig_decode_s(ig_encode_s(g, g_gscale), g_gscale); }

float ig_rt2(float x, float delta, float scale)
{
    uint8_t c = ig_encode2(x, delta, scale);
    if (c == 0) return 0.0f;
    float m = (exp2f((float)(c & 0x7Fu) * delta) - 1.0f) / scale;
    return (c & 0x80u) ? -m : m;
}

float ig_rt2_sr(float x, float delta, float scale, float u)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0.0f;
    float a = fabsf(x) * scale;
    int lo = (int)floorf(log2f(1.0f + a) / delta);
    float r;
    if (lo >= IG_MAX_CODE) {
        r = exp2f((float)IG_MAX_CODE * delta) - 1.0f; /* clamp */
    } else {
        if (lo < 0) lo = 0;
        float vlo = exp2f((float)lo * delta) - 1.0f;
        float vhi = exp2f((float)(lo + 1) * delta) - 1.0f;
        float p = (a - vlo) / (vhi - vlo); /* linear in value space: unbiased */
        r = u < p ? vhi : vlo;
    }
    r /= scale;
    return x < 0.0f ? -r : r;
}

void ig_get_role_grid(const char *role, float *d, float *s)
{
    if (!strcmp(role, "w")) { *d = g_w_delta; *s = g_w_scale; }
    else { *d = g_act_delta; *s = g_act_scale; }
}

void ig_set_role_grid(const char *role, float d, float s)
{
    if (!strcmp(role, "w")) { g_w_delta = d; g_w_scale = s; }
    else if (!strcmp(role, "act")) { g_act_delta = d; g_act_scale = s; }
}

float ig_rt_act(float x) { return ig_rt2(x, g_act_delta, g_act_scale); }
float ig_rt_w(float x) { return ig_rt2(x, g_w_delta, g_w_scale); }

/* ---- Two-byte weights (E8) ---- */
#define W16_SUB 256 /* low byte: 256 sub-steps per forward grid step */
#define W16_MAXM (IG_MAX_CODE * W16_SUB)

int16_t ig_w16_encode(float x, float u)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0;
    float a = fabsf(x) * g_w_scale;
    float d16 = g_w_delta / (float)W16_SUB;
    float qf = log2f(1.0f + a) / d16;
    int M;
    if (u < 0.0f) {
        M = (int)lroundf(qf);
    } else { /* SR on the 16-bit grid, unbiased in value space */
        int lo = (int)floorf(qf);
        if (lo < 0) lo = 0;
        float vlo = exp2f((float)lo * d16) - 1.0f;
        float vhi = exp2f((float)(lo + 1) * d16) - 1.0f;
        M = u < (a - vlo) / (vhi - vlo) ? lo + 1 : lo;
    }
    if (M > W16_MAXM) M = W16_MAXM;
    return (int16_t)(x < 0.0f ? -M : M);
}

float ig_w16_full(int16_t wq)
{
    int M = wq < 0 ? -wq : wq;
    float m = (exp2f((float)M * g_w_delta / (float)W16_SUB) - 1.0f) / g_w_scale;
    return wq < 0 ? -m : m;
}

float ig_w16_fwd(int16_t wq)
{
    int M = wq < 0 ? -wq : wq;
    int hi = (M + W16_SUB / 2) / W16_SUB; /* integer: (M + 128) >> 8 */
    float m = (exp2f((float)hi * g_w_delta) - 1.0f) / g_w_scale;
    return wq < 0 ? -m : m;
}
float ig_rt_w_sr(float x, float u) { return ig_rt2_sr(x, g_w_delta, g_w_scale, u); }
uint8_t ig_encode_w(float x) { return ig_encode2(x, g_w_delta, g_w_scale); }
float ig_max_abs(float delta, float scale)
{
    return (exp2f((float)IG_MAX_CODE * delta) - 1.0f) / scale;
}

static float g_fxstep = 0.1f;
void ig_set_fxstep(float step) { g_fxstep = step; }
float ig_get_fxstep(void) { return g_fxstep; }

float ig_fx_roundtrip(float x)
{
    float q = x / g_fxstep;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return lroundf(q) * g_fxstep;
}

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
