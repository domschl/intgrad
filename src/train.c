/* Byte-level next-byte predictor trainer (M2 hardened).
 *
 * Two data modes:
 *   --data F               raw file, random windows (toy mode)
 *   --index IDX/idx.bin    mmap'd dataset from tools/make_split.py; train
 *                          sampling touches train-bucket docs only, val
 *                          bpb is measured on hash-pinned held-out docs.
 *
 * Features: gradient-accumulation batches, Adam (default) or SGD,
 * linear warmup + cosine decay, periodic val-bpb eval, checkpoints.
 *
 *   ig-train [--index P | --data F] [--steps N] [--ctx N] [--hidden N]
 *            [--lr F] [--batch N] [--opt adam|sgd] [--warmup N]
 *            [--eval-every N] [--eval-windows N] [--save P] [--load P]
 *            [--sample N] [--seed N] [--no-quant] [--pure-int]
 *            [--grad-stats] [--gscale F]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ig/data.h"
#include "ig/dataset.h"
#include "ig/logquant.h"
#include "ig/net.h"
#include "ig/value.h"

static uint64_t rng_state;
static uint64_t nx(void) { return ig_splitmix64(&rng_state); }

/* One averaged CE evaluation over the held-out bucket, in bits/byte. */
static double eval_bpb(ig_MLP *m, const ig_Dataset *ds, int bucket,
                       int windows, int ctx)
{
    double tot = 0.0;
    int n = 0, got = 0;
    uint64_t er = 0x5EEDull + (uint64_t)bucket * 0x9E3779B97F4A7C15ull;
    /* per-call seeded: eval is deterministic and comparable across runs */
    for (size_t i = 0; i < ds->n_val + ds->n_test && n < windows; i++) {
        const uint8_t *doc;
        uint32_t len;
        if (ig_ds_eval_doc(ds, bucket, i, &doc, &len) != 0) continue;
        if ((int)len < ctx + 2) continue;
        uint32_t pos = (uint32_t)(ig_splitmix64(&er) % (len - (uint32_t)ctx - 1));
        ig_Value **logits = ig_mlp_forward(m, doc + pos);
        uint8_t target = doc[pos + ctx];
        ig_Value *loss = ig_mlp_loss(m, logits, target);
        tot += loss->data;
        n++;
        free(logits);
        got++;
        if (got % 256 == 0) ig_reset(); /* bound arena during eval */
    }
    ig_reset();
    return n ? tot / ((double)n * log(2.0)) : 0.0;
}

static int sample_byte(const ig_MLP *m, const uint8_t *ctxp, float temp)
{
    ig_Value **logits = ig_mlp_forward((ig_MLP *)m, ctxp);
    float *p = malloc(sizeof(float) * IG_VOCAB);
    float maxl = logits[0]->data, sum = 0;
    for (int i = 0; i < IG_VOCAB; i++) {
        p[i] = expf((logits[i]->data - maxl) / temp);
        sum += p[i];
    }
    float r = (float)(nx() >> 40) / 16777216.0f * sum, acc = 0;
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
    const char *path = NULL, *index = NULL;
    const char *save = NULL, *load = NULL;
    int steps = 2000, ctx = 4, nh = 64, nsample = 0, seed = 42;
    int batch = 1, warmup = 100, eval_every = 0, eval_windows = 512;
    int grad_stats = 0, use_adam = 1, sched_cosine = 1, save_every = 0;
    float lr = 3e-3f;

    for (int i = 1; i < argc; i++) {
        long v;
        if (!strcmp(argv[i], "--data") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--index") && i + 1 < argc) index = argv[++i];
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) load = argv[++i];
#define OPT(n, tv) else if (!strcmp(argv[i], n) && i + 1 < argc) { v = atol(argv[++i]); tv; }
        OPT("--steps", steps = (int)v)
        OPT("--ctx", ctx = (int)v)
        OPT("--hidden", nh = (int)v)
        OPT("--sample", nsample = (int)v)
        OPT("--seed", seed = (int)v)
        OPT("--batch", batch = (int)v)
        OPT("--warmup", warmup = (int)v)
        OPT("--eval-every", eval_every = (int)v)
        OPT("--eval-windows", eval_windows = (int)v)
        OPT("--save-every", save_every = (int)v)
        OPT("--lr", lr = (float)atof(argv[i]))
#undef OPT
        else if (!strcmp(argv[i], "--opt") && i + 1 < argc) {
            const char *o = argv[++i];
            if (!strcmp(o, "adam")) use_adam = 1;
            else if (!strcmp(o, "sgd")) use_adam = 0;
            else { fprintf(stderr, "bad --opt %s\n", o); return 2; }
        }
        else if (!strcmp(argv[i], "--sched") && i + 1 < argc) {
            const char *s2 = argv[++i];
            if (!strcmp(s2, "const")) sched_cosine = 0;
            else if (!strcmp(s2, "cosine")) sched_cosine = 1;
            else { fprintf(stderr, "bad --sched %s\n", s2); return 2; }
        }
        else if (!strcmp(argv[i], "--loss-fp")) ig_loss_fp = 1;
        else if (!strcmp(argv[i], "--fp-logits")) ig_fp_logits = 1;
        else if (!strcmp(argv[i], "--fp-params")) ig_fp_params = 1;
        else if (!strcmp(argv[i], "--no-quant")) ig_quant_enabled = 0;
        else if (!strcmp(argv[i], "--pure-int")) ig_pure_int = 1;
        else if (!strcmp(argv[i], "--grad-stats")) grad_stats = 1;
        else if (!strcmp(argv[i], "--gscale") && i + 1 < argc) ig_set_gscale((float)atof(argv[++i]));
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) ig_set_scale((float)atof(argv[++i]));
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!path && !index && !load) path = "data/sample.txt";
    rng_state = (uint64_t)seed * 0x9E3779B97F4A7C15ull | 1u;

    uint8_t *raw = NULL;
    size_t raw_len = 0;
    if (path) raw = ig_load_file(path, &raw_len);

    ig_Dataset *ds = NULL;
    if (index) {
        ds = ig_ds_load(index);
        if (!ds) return 1;
        printf("ig-train: dataset %zu train docs (%.2f GB), %zu val, %zu test\n",
               ds->n_train, (double)ds->train_bytes / 1e9, ds->n_val, ds->n_test);
    }

    ig_MLP *m = NULL;
    long start_step = 0;
    uint64_t ck_rng = 0;
    if (load) {
        m = ig_mlp_load2(load, &start_step, &ck_rng);
        if (!m) return 1;
        ctx = m->ctx; nh = m->nh;
        if (ck_rng) rng_state = ck_rng; /* continue data order bit-exactly */
        printf("ig-train: resumed %s (ctx=%d nh=%d) at step %ld\n",
               load, ctx, nh, start_step);
    } else {
        m = ig_mlp_new(ctx, nh, (unsigned)seed);
    }
    printf("ig-train: ctx=%d hidden=%d quant=%s pure-int=%d opt=%s params=%d\n",
           ctx, nh, ig_quant_enabled ? "log8" : "off", ig_pure_int,
           use_adam ? "adam" : "sgd",
           ctx * IG_VOCAB * nh + nh + nh * IG_VOCAB + IG_VOCAB);

    clock_t t0 = clock();
    for (long step = start_step; step < steps; step++) {
        float lr_t = lr;
        if (warmup > 0 && step < warmup) lr_t = lr * (float)(step + 1) / (float)warmup;
        else if (sched_cosine && steps > warmup) {
            float p = (float)(step - warmup) / (float)(steps - warmup);
            lr_t = lr * 0.5f * (1.0f + cosf(3.14159265f * p));
        }

        ig_mlp_zero_grad(m);
        double loss_acc = 0.0;
        for (int b = 0; b < batch; b++) {
            const uint8_t *win;
            if (ds) {
                const uint8_t *doc;
                uint32_t len;
                if (ig_ds_train_doc(ds, &rng_state, ctx + 2, &doc, &len) != 0)
                    continue;
                uint32_t pos = (uint32_t)(nx() % (len - (uint32_t)ctx - 1));
                win = doc + pos;
            } else {
                size_t pos = (size_t)(nx() % (raw_len - (size_t)ctx - 1));
                win = raw + pos;
            }
            ig_Value **logits = ig_mlp_forward(m, win);
            uint8_t target = win[ctx];
            ig_Value *loss = ig_mlp_loss(m, logits, target);
            ig_backward(loss);
            loss_acc += loss->data;
            free(logits);
        }
        if (batch > 1) ig_mlp_scale_grads(m, 1.0f / (float)batch);
        if (grad_stats && step % 500 == 0) {
            static const char *names[4] = { "W1", "b1", "W2", "b2" };
            for (int gi = 0; gi < 4; gi++) {
                ig_GradStat gs = ig_mlp_grad_stats(m, gi);
                printf("  grads[%s] mean|g|=%.3e max|g|=%.3e collapse=%.2f%% sat=%.2f%%\n",
                       names[gi], gs.mean_abs, gs.max_abs,
                       100.0 * gs.zero_frac, 100.0 * gs.sat_frac);
            }
        }
        if (use_adam) ig_mlp_adam_update(m, lr_t);
        else ig_mlp_sgd_update(m, lr_t);
        if (step % 100 == 0 || step == (long)steps - 1)
            printf("step %5ld/%d  lr %.4f  loss %.4f  ppl %8.1f\n",
                   step, steps, lr_t, loss_acc / batch, expf((float)(loss_acc / batch)));
        if (ds && eval_every > 0 && (step + 1) % eval_every == 0)
            printf("** step %5ld  val bpb %.4f\n", step + 1,
                   eval_bpb(m, ds, 1, eval_windows, ctx));
        if (save && save_every > 0 && (step + 1) % save_every == 0)
            ig_mlp_save2(m, save, step + 1, rng_state);
        ig_reset();
    }
    printf("done in %.2fs\n", (double)(clock() - t0) / CLOCKS_PER_SEC);

    if (ds && eval_every > 0)
        printf("final val bpb %.4f  (test bpb %.4f)\n",
               eval_bpb(m, ds, 1, eval_windows, ctx),
               eval_bpb(m, ds, 2, eval_windows, ctx));
    if (save) {
        long done = steps > start_step ? steps : start_step;
        if (ig_mlp_save2(m, save, done, rng_state) == 0)
            printf("saved %s (steps_done=%ld)\n", save, done);
    }

    if (nsample > 0) {
        uint8_t cxbuf[64];
        if (ds) {
            const uint8_t *doc;
            uint32_t len;
            if (ig_ds_train_doc(ds, &rng_state, ctx + 1, &doc, &len) == 0)
                memcpy(cxbuf, doc, (size_t)ctx);
        } else {
            for (int i = 0; i < ctx; i++) cxbuf[i] = raw[(size_t)(nx() % raw_len)];
        }
        fputs("--- sample (temp 0.8) ---\n", stdout);
        for (int i = 0; i < nsample; i++) {
            int b = sample_byte(m, cxbuf, 0.8f);
            fputc(b, stdout);
            memmove(cxbuf, cxbuf + 1, (size_t)ctx - 1);
            cxbuf[ctx - 1] = (uint8_t)b;
        }
        fputc('\n', stdout);
    }
    ig_mlp_free(m);
    ig_ds_free(ds);
    free(raw);
    return 0;
}
