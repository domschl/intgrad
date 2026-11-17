/* Checkpoint state-fidelity regression: a run of 20 updates must produce
 * bit-identical parameters whether executed straight through or as
 * 10 steps + save + resume + 10 steps. Guards params, master codes, Lion
 * momentum, step counters and RNG continuity. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ig/logquant.h"
#include "ig/net.h"
#include "ig/value.h"

#define NUP 20

static uint64_t rs = 12345;
static uint64_t nx(void)
{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return rs;
}

/* One deterministic update: forward/backward on a synthetic window. */
static void one_update(ig_MLP *m, uint8_t *win, int ctx)
{
    for (int i = 0; i < ctx; i++) win[i] = (uint8_t)(nx() & 0xFF);
    win[ctx] = (uint8_t)(nx() & 0xFF);
    ig_mlp_zero_grad(m);
    ig_Value **logits = ig_mlp_forward(m, win);
    ig_Value *loss = ig_mlp_loss(m, logits, win[ctx]);
    ig_backward(loss);
    ig_mlp_lion_update(m, 1e-3f);
    free(logits);
    ig_reset();
}

/* Checkpoint's params after `to_step` updates (exclusive end). */
static void train_range(ig_MLP *m, int ctx, long from, long to, uint8_t *win,
                        const char *save_at)
{
    for (long s = from; s < to; s++) one_update(m, win, ctx);
    if (save_at) {
        if (ig_mlp_save(m, save_at, to, 0xC0FFEE) != 0) exit(1);
    }
}

static int run_case(int w2)
{
    const int ctx = 2, nh = 8;
    uint8_t win[8];
    const char *ck = "/tmp/ig_test_ckpt.bin";
    long mid = NUP / 2;

    /* straight: NUP updates in one process */
    ig_mlp_opt_reset();
    ig_MLP *a = ig_mlp_new(ctx, nh, 7);
    train_range(a, ctx, 0, NUP, win, "/tmp/ig_test_ckpt_straight.bin");

    /* split: mid updates, save, resume, remaining */
    rs = 12345; /* reset our test rng so both paths see identical inputs */
    ig_mlp_opt_reset();
    ig_MLP *b = ig_mlp_new(ctx, nh, 7);
    train_range(b, ctx, 0, mid, win, ck);
    ig_mlp_free(b);

    long steps_done = -1;
    uint64_t rng_back = 0;
    ig_MLP *c = ig_mlp_load(ck, &steps_done, &rng_back);
    if (!c) { fprintf(stderr, "load failed\n"); return 1; }
    if (steps_done != mid || rng_back != 0xC0FFEE) {
        fprintf(stderr, "FAIL: steps_done=%ld expected %ld\n", steps_done, mid);
        return 1;
    }
    train_range(c, ctx, mid, NUP, win, NULL);

    /* compare params of straight vs resumed */
    ig_Value **la[4] = { a->W1, a->b1, a->W2, a->b2 };
    ig_Value **lb[4] = { c->W1, c->b1, c->W2, c->b2 };
    long ns[4] = { (long)ctx * IG_VOCAB * nh, nh,
                   (long)nh * IG_VOCAB, IG_VOCAB };
    long diffs = 0, checked = 0, lowbyte = 0;
    for (int g = 0; g < 4; g++) {
        for (long i = 0; i < ns[g]; i++, checked++) {
            if (la[g][i]->data != lb[g][i]->data
                || la[g][i]->wq != lb[g][i]->wq) diffs++;
            if (la[g][i]->wq % 256 != 0) lowbyte++; /* residual byte in use */
        }
    }
    if (w2 && lowbyte == 0) {
        fprintf(stderr, "FAIL: w2 run never used the low byte\n");
        return 1;
    }
    if (checked == 0 || diffs > 0) {
        fprintf(stderr, "FAIL: %ld/%ld params differ after resume\n",
                diffs, checked);
        return 1;
    }
    printf("checkpoint: resume is bit-exact over %ld params (w2=%d)\n",
           checked, w2);
    unlink(ck);
    unlink("/tmp/ig_test_ckpt_straight.bin");
    ig_mlp_free(a);
    ig_mlp_free(c);
    return 0;
}

int main(void)
{
    ig_w2 = 0; /* one-byte storage + SR: the SR RNG must resume too */
    if (run_case(0)) return 1;
    rs = 12345;
    ig_w2 = 1;
    if (run_case(1)) return 1;
    return 0;
}
