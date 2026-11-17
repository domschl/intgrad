#include "ig/logquant.h"

#include <math.h>

static float g_scale = 256.0f;
static float g_fxstep = 0.1f;

void ig_set_scale(float scale) { g_scale = scale; }
float ig_get_scale(void) { return g_scale; }

/* ---- canonical tables ---------------------------------------------- */

static float DEC8[IG_MAX_CODE + 1], THR8[IG_MAX_CODE + 1];
static float DEC16[IG_W16_MAXM + 1], THR16[IG_W16_MAXM + 1];
static int g_init;

static float V(double t) { return (float)(exp2(t * 0.1 / 256.0) - 1.0); }

static void init(void)
{
    if (g_init) return;
    for (int c = 0; c <= IG_MAX_CODE; c++) DEC8[c] = V(256.0 * c);
    THR8[0] = 0.0f;
    for (int k = 1; k <= IG_MAX_CODE; k++) THR8[k] = V(256.0 * k - 128.0);
    for (int m = 0; m <= IG_W16_MAXM; m++) DEC16[m] = V((double)m);
    THR16[0] = 0.0f;
    for (int k = 1; k <= IG_W16_MAXM; k++) THR16[k] = V((double)k - 0.5);
    g_init = 1;
}

const float *ig_lq_table(int which, int *n)
{
    init();
    switch (which) {
    case 0: *n = IG_MAX_CODE + 1; return DEC8;
    case 1: *n = IG_MAX_CODE + 1; return THR8;
    case 2: *n = IG_W16_MAXM + 1; return DEC16;
    case 3: *n = IG_W16_MAXM + 1; return THR16;
    default: *n = 0; return 0;
    }
}

/* number of k in 1..n with t[k] <= a  (t[1..n] ascending) */
static int count_le(const float *t, int n, float a)
{
    int lo = 0, hi = n; /* answer in [lo, hi] */
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (t[mid] <= a) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/* ---- byte grid ------------------------------------------------------- */

uint8_t ig_encode_s(float x, float scale)
{
    init();
    if (!(x > 0.0f) && !(x < 0.0f)) return 0; /* 0 or NaN -> zero */
    int q = count_le(THR8, IG_MAX_CODE, fabsf(x) * scale);
    if (q == 0) return 0;
    return (uint8_t)(x < 0.0f ? 128 + q : q);
}

float ig_decode_s(uint8_t c, float scale)
{
    init();
    float m = DEC8[c & 0x7Fu] / scale;
    return (c & 0x80u) ? -m : m;
}

float ig_roundtrip_s(float x, float scale) { return ig_decode_s(ig_encode_s(x, scale), scale); }

/* SR between the neighbouring grid values DEC[lo] <= a < DEC[lo+1]. */
static float sr_pick(const float *dec, int maxc, float a, float u)
{
    int lo = count_le(dec, maxc, a); /* largest c with dec[c] <= a (dec[0]=0) */
    if (lo >= maxc) return dec[maxc];
    float vlo = dec[lo], vhi = dec[lo + 1];
    float p = (a - vlo) / (vhi - vlo);
    return u < p ? vhi : vlo;
}

float ig_roundtrip_sr_s(float x, float scale, float u)
{
    init();
    if (!(x > 0.0f) && !(x < 0.0f)) return 0.0f;
    float r = sr_pick(DEC8, IG_MAX_CODE, fabsf(x) * scale, u) / scale;
    return x < 0.0f ? -r : r;
}

float ig_max_abs(float scale)
{
    init();
    return DEC8[IG_MAX_CODE] / scale;
}

uint8_t ig_encode(float x) { return ig_encode_s(x, g_scale); }
float ig_decode(uint8_t code) { return ig_decode_s(code, g_scale); }
float ig_roundtrip(float x) { return ig_roundtrip_s(x, g_scale); }
float ig_roundtrip_sr(float x, float u) { return ig_roundtrip_sr_s(x, g_scale, u); }

float ig_pow2_scale(float mx)
{
    init();
    if (!(mx > 0.0f)) return 1.0f;
    float r = DEC8[IG_MAX_CODE] / mx;
    int e;
    if (isinf(r)) e = 126;
    else {
        frexpf(r, &e); /* r = f * 2^e, f in [0.5, 1) -> floor(log2 r) = e - 1 */
        e -= 1;
    }
    if (e > 126) e = 126;
    if (e < -126) e = -126;
    return ldexpf(1.0f, e);
}

/* ---- fixed-point byte grid ------------------------------------------ */

void ig_set_fxstep(float step) { g_fxstep = step; }
float ig_get_fxstep(void) { return g_fxstep; }

float ig_fx_roundtrip(float x)
{
    float q = x / g_fxstep;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (float)lroundf(q) * g_fxstep;
}

/* ---- 16-bit grid ------------------------------------------------------ */

int16_t ig_enc16(float x, float u, float scale)
{
    init();
    if (!(x > 0.0f) && !(x < 0.0f)) return 0;
    float a = fabsf(x) * scale;
    int M;
    if (u < 0.0f) {
        M = count_le(THR16, IG_W16_MAXM, a);
    } else {
        int lo = count_le(DEC16, IG_W16_MAXM, a);
        if (lo >= IG_W16_MAXM) M = IG_W16_MAXM;
        else {
            float vlo = DEC16[lo], vhi = DEC16[lo + 1];
            M = u < (a - vlo) / (vhi - vlo) ? lo + 1 : lo;
        }
    }
    return (int16_t)(x < 0.0f ? -M : M);
}

float ig_dec16(int16_t wq, float scale)
{
    init();
    int M = wq < 0 ? -wq : wq;
    float m = DEC16[M] / scale;
    return wq < 0 ? -m : m;
}

int16_t ig_w16_encode(float x, float u) { return ig_enc16(x, u, g_scale); }
float ig_w16_full(int16_t wq) { return ig_dec16(wq, g_scale); }

float ig_w16_fwd(int16_t wq)
{
    init();
    int M = wq < 0 ? -wq : wq;
    int hi = (M + IG_W16_SUB / 2) / IG_W16_SUB; /* integer: (M + 128) >> 8 */
    float m = DEC8[hi] / g_scale;
    return wq < 0 ? -m : m;
}
