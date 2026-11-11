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

uint8_t ig_encode_s(float x, float scale)
{
    if (!(x > 0.0f) && !(x < 0.0f)) return 0; /* 0 or NaN -> zero */
    int neg = x < 0.0f;
    float a = fabsf(x) * scale;
    int q = (int)lroundf(log2f(1.0f + a) / IG_DELTA);
    if (q < 1) q = 1;
    if (q > IG_MAX_CODE) q = IG_MAX_CODE;
    return (uint8_t)(neg ? 128 + q : q);
}

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
    if (!(x > 0.0f) && !(x < 0.0f)) return 0.0f;
    int neg = x < 0.0f;
    float a = fabsf(x) * scale;
    int q = (int)lroundf(log2f(1.0f + a) / delta);
    if (q < 0) q = 0; /* faithful nearest: small values may snap to 0
                       * (the legacy q>=1 floor is harmless at delta=0.1/S=256
                       * but biases coarser probe grids) */
    if (q > IG_MAX_CODE) q = IG_MAX_CODE;
    float m = (exp2f((float)q * delta) - 1.0f) / scale;
    return neg ? -m : m;
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
