/* ig-train: byte-level MLP trainer on the scalar reference engine.
 *
 * This is the numerics ORACLE (plan/001): small, exact, slow. Scalable
 * training happens in the PyTorch reference (plan/001 M2+).
 *
 *   ig-train [--index IDX | --data FILE] [--steps N] [--ctx N] [--hidden N]
 *            [--lr F] [--batch N] [--warmup N] [--sched cosine|const]
 *            [--wd F] [--betas B1 B2] [--eval-every N] [--eval-windows N]
 *            [--save P] [--save-every N] [--load P] [--sample N] [--seed N]
 *            [--no-quant] [--pure-int] [--grad-stats] [--w1] [--w2-sr]
 *
 *   --index      idx.bin v2 from ig-make-split (py/intgrad/corpus)
 *   --data       raw file, random windows (smoke mode)
 *   --no-quant   exact float oracle
 *   --pure-int   gradients quantized to the log-byte grid (per-tensor pow2)
 *   --w1         one-byte weight storage with stochastic rounding
 *   --w2-sr      two-byte weights with SR on the low byte (default RTN)
 *
 * Optimizer: Lion (lr 2e-3, wd 0.1, batch 32 defaults).
 * --steps is the TOTAL horizon; resume with --load and the same flags.
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

/* Mean CE over `windows` held-out windows of a bucket, in bits/byte.
 * Docs are drawn length-weighted (every held-out byte equally likely, the
 * same distribution training samples from) with a fixed seed per bucket,
 * so the eval set is identical across runs. */
static double eval_bpb(ig_MLP *m, const ig_Dataset *ds, int bucket,
                       int windows, int ctx)
{
    double tot = 0.0;
    int n = 0;
    uint64_t er = 0x5EEDull + (uint64_t)bucket * 0x9E3779B97F4A7C15ull;
    while (n < windows) {
        const uint8_t *doc;
        uint32_t len;
        if (ig_ds_sample_doc(ds, bucket, &er, ctx + 2, &doc, &len) != 0) break;
        uint32_t pos = (uint32_t)(ig_splitmix64(&er) % (len - (uint32_t)ctx - 1));
        ig_Value **logits = ig_mlp_forward(m, doc + pos);
        tot += ig_mlp_loss(m, logits, doc[pos + ctx])->data;
        free(logits);
        if (++n % 256 == 0) ig_reset();
    }
    ig_reset();
    return n ? tot / ((double)n * log(2.0)) : 0.0;
}

static int sample_byte(ig_MLP *m, const uint8_t *ctxp, float temp)
{
    ig_Value **logits = ig_mlp_forward(m, ctxp);
    float p[IG_VOCAB], maxl = logits[0]->data, sum = 0, acc = 0;
    for (int i = 1; i < IG_VOCAB; i++)
        if (logits[i]->data > maxl) maxl = logits[i]->data;
    for (int i = 0; i < IG_VOCAB; i++) {
        p[i] = expf((logits[i]->data - maxl) / temp);
        sum += p[i];
    }
    free(logits);
    float r = (float)(nx() >> 40) / 16777216.0f * sum;
    for (int i = 0; i < IG_VOCAB; i++) {
        acc += p[i];
        if (r <= acc) return i;
    }
    return IG_VOCAB - 1;
}

int main(int argc, char **argv)
{
    const char *path = NULL, *index = NULL, *save = NULL, *load = NULL;
    int steps = 2000, ctx = 4, nh = 64, nsample = 0, seed = 42;
    int batch = 32, warmup = 100, eval_every = 0, eval_windows = 512;
    int grad_stats = 0, sched_cosine = 1, save_every = 0;
    float lr = 2e-3f;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if (!strcmp(a, "--data") && has) path = argv[++i];
        else if (!strcmp(a, "--index") && has) index = argv[++i];
        else if (!strcmp(a, "--save") && has) save = argv[++i];
        else if (!strcmp(a, "--load") && has) load = argv[++i];
#define IOPT(n, var) else if (!strcmp(a, n) && has) var = atoi(argv[++i]);
        IOPT("--steps", steps)
        IOPT("--ctx", ctx)
        IOPT("--hidden", nh)
        IOPT("--sample", nsample)
        IOPT("--seed", seed)
        IOPT("--batch", batch)
        IOPT("--warmup", warmup)
        IOPT("--eval-every", eval_every)
        IOPT("--eval-windows", eval_windows)
        IOPT("--save-every", save_every)
#undef IOPT
        else if (!strcmp(a, "--lr") && has) lr = (float)atof(argv[++i]);
        else if (!strcmp(a, "--wd") && has) ig_wd = (float)atof(argv[++i]);
        else if (!strcmp(a, "--betas") && i + 2 < argc) {
            float b1 = (float)atof(argv[++i]);
            ig_set_lion_betas(b1, (float)atof(argv[++i]));
        }
        else if (!strcmp(a, "--sched") && has) {
            const char *s = argv[++i];
            if (!strcmp(s, "const")) sched_cosine = 0;
            else if (!strcmp(s, "cosine")) sched_cosine = 1;
            else { fprintf(stderr, "bad --sched %s\n", s); return 2; }
        }
        else if (!strcmp(a, "--no-quant")) ig_quant_enabled = 0;
        else if (!strcmp(a, "--pure-int")) ig_pure_int = 1;
        else if (!strcmp(a, "--grad-stats")) grad_stats = 1;
        else if (!strcmp(a, "--w1")) ig_w2 = 0;
        else if (!strcmp(a, "--w2-sr")) ig_w2_sr = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (batch < 32)
        fprintf(stderr, "ig-train: warning: Lion with batch %d < 32 "
                        "(sign updates are noisy at small batch)\n", batch);
    if (!path && !index && !load) {
        fprintf(stderr, "ig-train: need --index, --data or --load\n");
        return 2;
    }
    rng_state = (uint64_t)seed * 0x9E3779B97F4A7C15ull | 1u;

    uint8_t *raw = NULL;
    size_t raw_len = 0;
    if (path) raw = ig_load_file(path, &raw_len);
    if (raw && raw_len < (size_t)ctx + 2) {
        fprintf(stderr, "ig-train: %s is too short\n", path);
        return 1;
    }
    ig_Dataset *ds = NULL;
    if (index) {
        if (!(ds = ig_ds_load(index))) return 1;
        printf("ig-train: dataset %zu train docs (%.2f GB), %zu val, %zu test\n",
               ds->n_train, (double)ds->train_bytes / 1e9, ds->n_val, ds->n_test);
    }
    if (!raw && !ds) {
        fprintf(stderr, "ig-train: --load needs --index or --data to train on\n");
        return 2;
    }

    ig_MLP *m;
    long start_step = 0;
    uint64_t ck_rng = 0;
    if (load) {
        if (!(m = ig_mlp_load(load, &start_step, &ck_rng))) return 1;
        ctx = m->ctx;
        nh = m->nh;
        if (ck_rng) rng_state = ck_rng; /* continue data order bit-exactly */
        printf("ig-train: resumed %s (ctx=%d nh=%d) at step %ld\n",
               load, ctx, nh, start_step);
    } else {
        m = ig_mlp_new(ctx, nh, (unsigned)seed);
        if (!m) { fprintf(stderr, "ig-train: bad dims\n"); return 2; }
    }
    printf("ig-train: ctx=%d hidden=%d params=%ld quant=%s pure-int=%d "
           "weights=%s lion lr=%g batch=%d wd=%g\n",
           ctx, nh, ig_mlp_nparams(m), ig_quant_enabled ? "log8" : "off",
           ig_pure_int, ig_w2 ? (ig_w2_sr ? "2B-sr" : "2B") : "1B-sr",
           lr, batch, ig_wd);

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
        int nb = 0;
        for (int b = 0; b < batch; b++) {
            const uint8_t *win;
            if (ds) {
                const uint8_t *doc;
                uint32_t len;
                if (ig_ds_train_doc(ds, &rng_state, ctx + 2, &doc, &len) != 0)
                    continue;
                win = doc + nx() % (len - (uint32_t)ctx - 1);
            } else {
                win = raw + nx() % (raw_len - (size_t)ctx - 1);
            }
            ig_Value **logits = ig_mlp_forward(m, win);
            ig_Value *loss = ig_mlp_loss(m, logits, win[ctx]);
            ig_backward(loss);
            loss_acc += loss->data;
            nb++;
            free(logits);
        }
        if (nb > 1) ig_mlp_scale_grads(m, 1.0f / (float)nb);
        if (grad_stats && step % 500 == 0) {
            static const char *names[IG_NGROUPS] = { "W1", "b1", "W2", "b2" };
            for (int g = 0; g < IG_NGROUPS; g++) {
                ig_GradStat gs = ig_mlp_grad_stats(m, g);
                printf("  grads[%s] mean|g|=%.3e max|g|=%.3e collapse=%.2f%% "
                       "(mass %.4f%%) sat=%.2f%% (mass %.4f%%)\n",
                       names[g], gs.mean_abs, gs.max_abs,
                       100.0 * gs.zero_frac, 100.0 * gs.collapse_mass,
                       100.0 * gs.sat_frac, 100.0 * gs.sat_mass);
            }
        }
        ig_mlp_lion_update(m, lr_t);
        if (step % 100 == 0 || step == (long)steps - 1) {
            long un, usub, umov;
            ig_mlp_update_stats(&un, &usub, &umov, 1);
            double l = nb ? loss_acc / nb : 0.0;
            printf("step %5ld/%d  lr %.4f  loss %.4f  ppl %8.1f",
                   step, steps, lr_t, l, exp(l));
            if (un > 0)
                printf("  upd subgrid %.1f%% moved %.1f%%",
                       100.0 * usub / un, 100.0 * umov / un);
            putchar('\n');
        }
        if (ds && eval_every > 0 && (step + 1) % eval_every == 0)
            printf("** step %5ld  val bpb %.4f\n", step + 1,
                   eval_bpb(m, ds, 1, eval_windows, ctx));
        if (save && save_every > 0 && (step + 1) % save_every == 0)
            ig_mlp_save(m, save, step + 1, rng_state);
        ig_reset();
    }
    printf("done in %.2fs\n", (double)(clock() - t0) / CLOCKS_PER_SEC);

    if (ds && eval_every > 0)
        printf("final val bpb %.4f  (test bpb %.4f)\n",
               eval_bpb(m, ds, 1, eval_windows, ctx),
               eval_bpb(m, ds, 2, eval_windows, ctx));
    if (save) {
        long done = steps > start_step ? steps : start_step;
        if (ig_mlp_save(m, save, done, rng_state) == 0)
            printf("saved %s (steps_done=%ld)\n", save, done);
    }

    if (nsample > 0) {
        uint8_t *cx = malloc((size_t)ctx);
        const uint8_t *src = raw;
        uint32_t len = (uint32_t)raw_len;
        if (ds && ig_ds_train_doc(ds, &rng_state, ctx + 1, &src, &len) != 0) src = NULL;
        if (src) {
            memcpy(cx, src, (size_t)ctx);
            fputs("--- sample (temp 0.8) ---\n", stdout);
            for (int i = 0; i < nsample; i++) {
                int b = sample_byte(m, cx, 0.8f);
                fputc(b, stdout);
                memmove(cx, cx + 1, (size_t)ctx - 1);
                cx[ctx - 1] = (uint8_t)b;
            }
            fputc('\n', stdout);
        }
        free(cx);
    }
    ig_mlp_free(m);
    ig_ds_free(ds);
    free(raw);
    return 0;
}
