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
    float abs_grid = (exp2f(IG_DELTA) - 1.0f) / ig_get_scale();
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

    /* log-domain helpers agree with float arithmetic */
    uint8_t a = ig_encode(2.0f), b = ig_encode(3.0f);
    assert(fabsf(ig_decode(ig_logmul(a, b)) - 6.0f) < 0.4f);
    assert(fabsf(ig_decode(ig_logadd(a, b)) - 5.0f) < 0.4f);

    puts("logquant: all tests passed");
    return 0;
}
