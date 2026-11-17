/* Training-numerics regression: on the byte grid the relative
 * step is ~7%, so round-to-nearest one-byte parameter storage silently
 * discards most small updates and quantized training stalls. Stochastic rounding must keep quantized training close to the
 * float oracle — and the round-to-nearest ablation must still be clearly
 * worse (otherwise this test no longer guards the bug class).
 *
 * Small, deterministic: train the byte-MLP (Lion) on tests/fixtures/sample.txt, then report
 * mean CE over every window of the file (quantized forward for quant runs).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ig/data.h"
#include "ig/net.h"
#include "ig/value.h"

#define CTX 3
#define NH 32
#define STEPS 800
#define BATCH 2
#define LR 3e-3f /* Lion */
#define WARMUP 100

static uint64_t rs;
static uint64_t nx(void)
{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return rs;
}


static double run(const uint8_t *d, size_t n, int quant, int sr, int w2)
{
    ig_quant_enabled = quant;
    ig_sr = sr;       /* 1-byte mode */
    ig_w2 = w2;
    ig_w2_sr = sr;    /* two-byte mode: low byte */
    rs = 0xDEADBEEFull;
    ig_mlp_opt_reset();
    ig_MLP *m = ig_mlp_new(CTX, NH, 42);
    for (int s = 0; s < STEPS; s++) {
        float lr = s < WARMUP ? LR * (float)(s + 1) / WARMUP
                 : LR * 0.5f * (1.0f + cosf(3.14159265f * (float)(s - WARMUP)
                                            / (float)(STEPS - WARMUP)));
        ig_mlp_zero_grad(m);
        for (int b = 0; b < BATCH; b++) {
            size_t pos = (size_t)(nx() % (n - CTX - 1));
            ig_Value **lg = ig_mlp_forward(m, d + pos);
            ig_backward(ig_mlp_loss(m, lg, d[pos + CTX]));
            free(lg);
        }
        ig_mlp_scale_grads(m, 1.0f / BATCH);
        ig_mlp_lion_update(m, lr);
        ig_reset();
    }
    double tot = 0.0;
    size_t cnt = 0;
    for (size_t pos = 0; pos + CTX < n; pos++, cnt++) {
        ig_Value **lg = ig_mlp_forward(m, d + pos);
        tot += ig_mlp_loss(m, lg, d[pos + CTX])->data;
        free(lg);
        if (cnt % 128 == 0) ig_reset();
    }
    ig_reset();
    ig_mlp_free(m);
    return tot / (double)cnt;
}

int main(void)
{
    size_t n = 0;
    uint8_t *d = ig_load_file(IG_TEST_DATA, &n);
    if (!d || n < 64) { fprintf(stderr, "cannot load %s\n", IG_TEST_DATA); return 1; }

    double fp = run(d, n, 0, 1, 0);
    double sr = run(d, n, 1, 1, 0);
    double rtn = run(d, n, 1, 0, 0);
    double w2sr = run(d, n, 1, 1, 1);
    double w2rtn = run(d, n, 1, 0, 1);
    printf("sr_train: CE nats  fp %.3f  quant+SR %.3f  quant+RTN %.3f"
           "  w2+SR %.3f  w2+RTN %.3f\n", fp, sr, rtn, w2sr, w2rtn);

    if (sr > fp + 0.35) {
        fprintf(stderr, "FAIL: quantized (SR) %.3f not within 0.35 of fp %.3f\n", sr, fp);
        return 1;
    }
    if (rtn < sr + 0.5) {
        fprintf(stderr, "FAIL: RTN ablation %.3f no longer clearly worse than SR %.3f"
                        " (test lost its teeth — retune)\n", rtn, sr);
        return 1;
    }
    /* Two-byte weights must work with AND without SR: the low byte
     * resolves updates the high byte cannot. */
    if (w2sr > fp + 0.35 || w2rtn > fp + 0.35) {
        fprintf(stderr, "FAIL: two-byte weights (SR %.3f / RTN %.3f) not within"
                        " 0.35 of fp %.3f\n", w2sr, w2rtn, fp);
        return 1;
    }
    free(d);
    puts("sr_train: passed");
    return 0;
}
