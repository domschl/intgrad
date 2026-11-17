#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "ig/logquant.h"

int main(void)
{
    /* zero maps to code 0 and back */
    assert(ig_encode(0.0f) == 0);
    assert(ig_decode(0) == 0.0f);

    /* sign handling */
    assert(ig_encode(-1.0f) > 128);
    assert(ig_encode(1.0f) < 128);

    /* round-trip relative error across the useful range */
    /* Away from zero the grid is multiplicative (~7.2% step -> <6% error).
     * Near zero symlog spacing becomes absolute: (2^DELTA - 1)/SCALE. */
    int nt;
    float abs_grid = ig_lq_table(0, &nt)[1] / ig_get_scale();
    float vals[] = { 0.001f, 0.01f, 0.1f, 0.5f, 1.0f, 3.0f, 10.0f, 20.0f,
                     -0.001f, -0.3f, -2.0f, -15.0f };
    for (size_t i = 0; i < sizeof vals / sizeof *vals; i++) {
        float x = vals[i];
        float y = ig_roundtrip(x);
        float err = fabsf(y - x);
        float tol = (fabsf(x) >= 0.01f) ? 0.06f * fabsf(x) : abs_grid;
        if (err > tol) {
            fprintf(stderr, "FAIL: %g -> %g (err %g > tol %g)\n", x, y, err, tol);
            return 1;
        }
    }

    /* positive codes are non-decreasing everywhere ... */
    uint8_t prev = 0;
    for (float x = 0.001f; x < 20.0f; x *= 1.1f) {
        uint8_t c = ig_encode(x);
        if (c < prev || c == 0 || c >= 128) {
            fprintf(stderr, "FAIL: bad code ordering at %g\n", x);
            return 1;
        }
        prev = c;
    }
    /* ... and strictly increasing in the multiplicative regime, */
    /* with the 10% probe step exceeding the ~7.2% grid step.     */
    prev = 0;
    for (float x = 0.05f; x < 20.0f; x *= 1.1f) {
        uint8_t c = ig_encode(x);
        if (c <= prev) {
            fprintf(stderr, "FAIL: code not strictly monotonic at %g\n", x);
            return 1;
        }
        prev = c;
    }

    /* saturation: values beyond range clamp instead of wrapping */
    assert(ig_encode(1e9f) < 128);
    assert(ig_decode(ig_encode(1e9f)) > 20.0f);

    /* two-byte weights: the forward view is exactly the 1-byte grid value,
     * and the 16-bit view resolves far finer than one byte step */
    for (float x = -3.0f; x < 3.0f; x += 0.137f) {
        int16_t wq = ig_w16_encode(x, -1.0f);
        assert(ig_w16_fwd(wq) == ig_roundtrip(ig_w16_full(wq)));
        if (fabsf(x) > 0.05f) assert(fabsf(ig_w16_full(wq) - x) < 3e-4f * fabsf(x));
    }

    /* stochastic rounding is unbiased: E[rt_sr(x)] ~= x */
    {
        double acc = 0.0;
        uint32_t r = 1;
        const int N = 20000;
        for (int i = 0; i < N; i++) {
            r = r * 1664525u + 1013904223u;
            acc += ig_roundtrip_sr(0.3f, (float)(r >> 8) / 16777216.0f);
        }
        assert(fabs(acc / N - 0.3) < 0.002);
    }

    /* fixed-point logit grid clamps symmetrically-ish at ±12.7 / -12.8 */
    assert(fabsf(ig_fx_roundtrip(100.0f) - 12.7f) < 1e-4f);
    assert(fabsf(ig_fx_roundtrip(0.04f)) < 1e-6f);

    puts("logquant: all tests passed");
    return 0;
}
