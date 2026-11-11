/* Byte-level next-byte predictor trainer (M2 proof of concept).
 *
 *   ig-train [--data F] [--steps N] [--ctx N] [--hidden N] [--lr F]
 *            [--sample N] [--seed N] [--no-quant]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ig/data.h"
#include "ig/logquant.h"
#include "ig/net.h"
#include "ig/value.h"

static uint32_t rng;
static float frand01(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (float)(rng >> 8) / 16777216.0f;
}

static int sample_arg(const ig_MLP *m, const uint8_t *ctxp, float temp)
{
    ig_Value **logits = ig_mlp_forward((ig_MLP *)m, ctxp);
    float *p = malloc(sizeof(float) * IG_VOCAB);
    float maxl = logits[0]->data, sum = 0;
    for (int i = 0; i < IG_VOCAB; i++) {
        p[i] = expf((logits[i]->data - maxl) / temp);
        sum += p[i];
    }
    float r = frand01() * sum, acc = 0;
    int pick = IG_VOCAB - 1;
    for (int i = 0; i < IG_VOCAB; i++) {
        acc += p[i];
        if (r <= acc) { pick = i; break; }
    }
    free(p);
    free(logits);
    return pick;
}

int main(int argc, char **argv)
{
    const char *path = "data/sample.txt";
    int steps = 2000, ctx = 4, nh = 64, nsample = 0, seed = 42;
    int grad_stats = 0;
    float lr = 0.05f;

    for (int i = 1; i < argc; i++) {
        long v;
        if (!strcmp(argv[i], "--data") && i + 1 < argc) path = argv[++i];
#define OPT(n, tv) else if (!strcmp(argv[i], n) && i + 1 < argc) { v = atol(argv[++i]); tv; }
        OPT("--steps", steps = (int)v)
        OPT("--ctx", ctx = (int)v)
        OPT("--hidden", nh = (int)v)
        OPT("--sample", nsample = (int)v)
        OPT("--seed", seed = (int)v)
        OPT("--lr", lr = (float)atof(argv[i]))
#undef OPT
        else if (!strcmp(argv[i], "--no-quant")) ig_quant_enabled = 0;
        else if (!strcmp(argv[i], "--pure-int")) ig_pure_int = 1;
        else if (!strcmp(argv[i], "--grad-stats")) grad_stats = 1;
        else if (!strcmp(argv[i], "--gscale") && i + 1 < argc) ig_set_gscale((float)atof(argv[++i]));
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    size_t len;
    uint8_t *data = ig_load_file(path, &len);
    if ((int)len < ctx + 2) {
        fprintf(stderr, "ig: training data too short (%zu bytes)\n", len);
        return 1;
    }
    rng = (uint32_t)seed;

    ig_MLP *m = ig_mlp_new(ctx, nh, (unsigned)seed);
    printf("ig-train: %zu bytes, ctx=%d hidden=%d quant=%s pure-int=%d gscale=%g params=%d\n",
           len, ctx, nh, ig_quant_enabled ? "log8" : "off",
           ig_pure_int, ig_get_gscale(),
           ctx * IG_VOCAB * nh + nh + nh * IG_VOCAB + IG_VOCAB);

    clock_t t0 = clock();
    for (int step = 0; step < steps; step++) {
        int pos = ctx + (int)(frand01() * (float)(len - ctx - 1));
        ig_mlp_zero_grad(m);
        ig_Value **logits = ig_mlp_forward(m, data + pos - ctx);
        ig_Value *loss = ig_mlp_loss(m, logits, data[pos]);
        ig_backward(loss);
        if (grad_stats && step % 200 == 0) {
            static const char *names[4] = { "W1", "b1", "W2", "b2" };
            for (int gi = 0; gi < 4; gi++) {
                ig_GradStat gs = ig_mlp_grad_stats(m, gi);
                printf("  grads[%s] n=%ld mean|g|=%.3e max|g|=%.3e collapse=%.2f%% true0=%.1f%% sat=%.2f%%\n",
                       names[gi], gs.n, gs.mean_abs, gs.max_abs,
                       100.0 * gs.zero_frac, 100.0 * gs.true_zero_frac,
                       100.0 * gs.sat_frac);
            }
        }
        ig_mlp_sgd_update(m, lr);
        free(logits); /* pointer array only; graph nodes live in the arena */
        if (step % 100 == 0 || step == steps - 1)
            printf("step %5d/%d  loss %.4f  ppl %7.1f\n",
                   step, steps, loss->data, expf(loss->data));
        ig_reset();
    }
    printf("done in %.2fs\n", (double)(clock() - t0) / CLOCKS_PER_SEC);

    if (nsample > 0) {
        uint8_t cxbuf[64];
        for (int i = 0; i < ctx; i++) cxbuf[i] = data[(size_t)(frand01() * (float)(len - 1))];
        fputs("--- sample (temp 0.8) ---\n", stdout);
        for (int i = 0; i < nsample; i++) {
            int b = sample_arg(m, cxbuf, 0.8f);
            fputc(b, stdout);
            memmove(cxbuf, cxbuf + 1, (size_t)ctx - 1);
            cxbuf[ctx - 1] = (uint8_t)b;
        }
        fputc('\n', stdout);
    }
    ig_mlp_free(m);
    free(data);
    return 0;
}
