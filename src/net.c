#include "ig/net.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ig/logquant.h"

int ig_fp_params = 0; /* diagnostic: params stored float (no grid round-trip
                       * on update): isolates parameter-grid cost from the
                       * activation/logit grids. */

static uint32_t rng_state;
static float frand01(void)
{ /* xorshift32 */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state >> 8) / 16777216.0f;
}

ig_MLP *ig_mlp_new(int ctx, int nh, unsigned seed)
{
    ig_MLP *m = calloc(1, sizeof *m);
    if (!m) { fprintf(stderr, "ig: oom\n"); exit(1); }
    rng_state = seed ? seed : 0x9E3779B9u;
    int nin = ctx * IG_VOCAB;

    m->ctx = ctx;
    m->nh = nh;
    m->W1 = malloc(sizeof(ig_Value *) * (size_t)nin * (size_t)nh);
    m->b1 = calloc((size_t)nh, sizeof(ig_Value *));
    m->W2 = malloc(sizeof(ig_Value *) * (size_t)nh * IG_VOCAB);
    m->b2 = calloc(IG_VOCAB, sizeof(ig_Value *));
    if (!m->W1 || !m->b1 || !m->W2 || !m->b2) {
        fprintf(stderr, "ig: oom\n");
        exit(1);
    }
    for (long i = 0; i < (long)nin * nh; i++)
        m->W1[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * 0.05f);
    for (int i = 0; i < nh; i++) m->b1[i] = ig_value_persist(0.0f);
    for (long i = 0; i < (long)nh * IG_VOCAB; i++)
        m->W2[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * 0.3f);
    for (int i = 0; i < IG_VOCAB; i++) m->b2[i] = ig_value_persist(0.0f);
    return m;
}

void ig_mlp_free(ig_MLP *m)
{
    if (!m) return;
    int nin = m->ctx * IG_VOCAB;
    for (long i = 0; i < (long)nin * m->nh; i++) ig_value_free_persist(m->W1[i]);
    for (int i = 0; i < m->nh; i++) ig_value_free_persist(m->b1[i]);
    for (long i = 0; i < (long)m->nh * IG_VOCAB; i++) ig_value_free_persist(m->W2[i]);
    for (int i = 0; i < IG_VOCAB; i++) ig_value_free_persist(m->b2[i]);
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m);
}

ig_Value **ig_mlp_forward(ig_MLP *m, const uint8_t *in)
{
    int nh = m->nh;
    ig_Value **hv = malloc(sizeof(ig_Value *) * (size_t)nh);
    ig_Value **logits = malloc(sizeof(ig_Value *) * IG_VOCAB);

    /* One-hot inputs: only the active byte per position contributes. */
    for (int j = 0; j < nh; j++) {
        ig_Value *acc = m->b1[j];
        for (int p = 0; p < m->ctx; p++) {
            ig_Value *w = m->W1[(size_t)(p * IG_VOCAB + in[p]) * nh + j];
            acc = ig_add(acc, w); /* x_i == 1, so plain add; grad d/dW = 1 */
        }
        hv[j] = ig_relu(acc);
    }
    /* D6 hybrid grid: the output/CE path uses the fixed-point byte grid
     * (constant absolute error), since CE noise scales with the ABSOLUTE
     * logit error; the log grid's 7% relative step there floors loss (E3). */
    int qsav = ig_quant_enabled;
    if (ig_fp_logits) ig_quant_enabled = 0; /* isolation: float logits */
    ig_fx_mode = ig_fp_logits ? 0 : 2; /* exact accumulation (int32 GEMM) */
    for (int k = 0; k < IG_VOCAB; k++) {
        ig_Value *acc = m->b2[k];
        for (int j = 0; j < nh; j++)
            acc = ig_add(acc, ig_mul(hv[j], m->W2[(size_t)j * IG_VOCAB + k]));
        logits[k] = acc;
    }
    if (!ig_fp_logits) {
        ig_fx_mode = 1;
        for (int k = 0; k < IG_VOCAB; k++) /* single requantization per logit */
            logits[k]->data = ig_fx_roundtrip(logits[k]->data);
    }
    ig_fx_mode = 0;
    ig_quant_enabled = qsav;
    free(hv);
    return logits;
}

int ig_loss_fp = 0; /* diagnostic: CE arithmetic from stored values, no requant */
int ig_fp_logits = 0; /* diagnostic (D6 isolation ladder): logits accumulate
                       * and stay FLOAT from the output layer onward; only
                       * the path up to hidden activations is byte-quantized.
                       * W2 storage still rounds to its grid on update. */

ig_Value *ig_mlp_loss(ig_MLP *m, ig_Value **logits, uint8_t target)
{
    (void)m;
    int qsav = ig_quant_enabled;
    if (ig_loss_fp || ig_fp_logits) ig_quant_enabled = 0;
    ig_fx_mode = (ig_loss_fp || ig_fp_logits) ? 0 : 1; /* D6 CE on fx grid */
    ig_Value *lse = ig_logsumexp((const ig_Value **)logits, IG_VOCAB);
    ig_Value *loss = ig_sub(lse, logits[target]);
    ig_fx_mode = 0;
    ig_quant_enabled = qsav;
    return loss;
}

static void zero_grad_list(ig_Value **ps, long n)
{
    for (long i = 0; i < n; i++) ps[i]->grad = 0.0f;
}

void ig_mlp_zero_grad(ig_MLP *m)
{
    zero_grad_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh);
    zero_grad_list(m->b1, m->nh);
    zero_grad_list(m->W2, (long)m->nh * IG_VOCAB);
    zero_grad_list(m->b2, IG_VOCAB);
}

static void sgd_list(ig_Value **ps, long n, float lr)
{
    for (long i = 0; i < n; i++) {
        ig_Value *p = ps[i];
        float g = p->grad;
        if (ig_pure_int) g = ig_roundtrip_g(g); /* D1: gradient on the byte grid */
        float v = p->data - lr * g;
        p->data = (ig_quant_enabled && !ig_fp_params) ? ig_roundtrip(v) : v;
    }
}

static ig_GradStat stats_list(ig_Value **ps, long n)
{
    ig_GradStat s = { 0 };
    s.n = n;
    long ztrue = 0, zq = 0, sat = 0;
    for (long i = 0; i < n; i++) {
        float g = ps[i]->grad;
        float a = fabsf(g);
        s.mean_abs += a;
        if (a > s.max_abs) s.max_abs = a;
        uint8_t c = ig_encode_s(g, ig_get_gscale());
        if (g == 0.0f) ztrue++;             /* structural zero (e.g. one-hot) */
        else if (c == 0) zq++;              /* nonzero gradient collapsed by quantization */
        else if ((c & 0x7Fu) == IG_MAX_CODE) sat++;
    }
    if (n > 0) {
        s.mean_abs /= (double)n;
        s.zero_frac = (float)zq / (float)n; /* collapse only; true zeros excluded */
        s.true_zero_frac = (float)ztrue / (float)n;
        s.sat_frac = (float)sat / (float)n;
    }
    return s;
}

ig_GradStat ig_mlp_grad_stats(const ig_MLP *m, int group)
{
    switch (group) {
    case 0: return stats_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh);
    case 1: return stats_list(m->b1, m->nh);
    case 2: return stats_list(m->W2, (long)m->nh * IG_VOCAB);
    default: return stats_list(m->b2, IG_VOCAB);
    }
}

void ig_mlp_sgd_update(ig_MLP *m, float lr)
{
    sgd_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh, lr);
    sgd_list(m->b1, m->nh, lr);
    sgd_list(m->W2, (long)m->nh * IG_VOCAB, lr);
    sgd_list(m->b2, IG_VOCAB, lr);
}

static void scale_list(ig_Value **ps, long n, float s)
{
    for (long i = 0; i < n; i++) ps[i]->grad *= s;
}

void ig_mlp_scale_grads(ig_MLP *m, float s)
{
    scale_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh, s);
    scale_list(m->b1, m->nh, s);
    scale_list(m->W2, (long)m->nh * IG_VOCAB, s);
    scale_list(m->b2, IG_VOCAB, s);
}

/* ---------------- Adam ---------------- */

typedef struct { float m, v; } AdamCell;
static AdamCell *adam_state;
static long adam_n;
static int adam_t;

static void adam_ensure(long total)
{
    if (adam_n == total) return;
    free(adam_state);
    adam_state = calloc((size_t)total, sizeof(AdamCell));
    if (!adam_state) { fprintf(stderr, "ig: oom adam\n"); exit(1); }
    adam_n = total;
    adam_t = 0;
}

static long mlp_nparams(const ig_MLP *m)
{
    return (long)m->ctx * IG_VOCAB * m->nh + m->nh
         + (long)m->nh * IG_VOCAB + IG_VOCAB;
}

void ig_mlp_adam_reset(void)
{
    free(adam_state);
    adam_state = NULL;
    adam_n = 0;
    adam_t = 0;
}

void ig_mlp_adam_update(ig_MLP *m, float lr)
{
    adam_ensure(mlp_nparams(m));
    adam_t++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)adam_t);
    float bc2 = 1.0f - powf(b2, (float)adam_t);
    long k = 0;
    ig_Value **lists[4] = { m->W1, m->b1, m->W2, m->b2 };
    long ns[4] = { (long)m->ctx * IG_VOCAB * m->nh, m->nh,
                   (long)m->nh * IG_VOCAB, IG_VOCAB };
    for (int gi = 0; gi < 4; gi++) {
        for (long i = 0; i < ns[gi]; i++) {
            ig_Value *p = lists[gi][i];
            float g = p->grad;
            if (ig_pure_int) g = ig_roundtrip_g(g); /* D1 */
            AdamCell *c = &adam_state[k++];
            c->m = b1 * c->m + (1.0f - b1) * g;
            c->v = b2 * c->v + (1.0f - b2) * g * g;
            float step = lr * (c->m / bc1) / (sqrtf(c->v / bc2) + eps);
            float nv = p->data - step;
            p->data = (ig_quant_enabled && !ig_fp_params) ? ig_roundtrip(nv) : nv;
        }
    }
}

/* ---------------- checkpoints ---------------- */

int ig_mlp_save2(const ig_MLP *m, const char *path, long steps_done,
                 uint64_t rng)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "ig: cannot write %s\n", tmp); return -1; }
    char magic[4] = { 'I', 'G', 'C', 'K' };
    int vers = 3; /* v2: Adam state, v3: + steps_done & RNG (schedule state) */
    fwrite(magic, 1, 4, f);
    fwrite(&vers, sizeof vers, 1, f);
    fwrite(&m->ctx, sizeof m->ctx, 1, f);
    fwrite(&m->nh, sizeof m->nh, 1, f);
    float s1 = ig_get_scale(), s2 = ig_get_gscale();
    fwrite(&s1, sizeof s1, 1, f);
    fwrite(&s2, sizeof s2, 1, f);
    ig_Value **lists[4] = { m->W1, m->b1, m->W2, m->b2 };
    long ns[4] = { (long)m->ctx * IG_VOCAB * m->nh, m->nh,
                   (long)m->nh * IG_VOCAB, IG_VOCAB };
    for (int gi = 0; gi < 4; gi++) {
        for (long i = 0; i < ns[gi]; i++) {
            float d = lists[gi][i]->data;
            uint8_t code = ig_encode(d); /* the byte-only view of the model */
            fwrite(&d, sizeof d, 1, f);
            fwrite(&code, 1, 1, f);
        }
    }
    /* Adam state (v2): float moments are the reference-mode storage; the
     * int-domain state format is milestone M5.5 territory. */
    fwrite(&adam_t, sizeof adam_t, 1, f);
    long total = mlp_nparams(m);
    for (long i = 0; i < total; i++) {
        AdamCell c = adam_state ? adam_state[i] : (AdamCell){ 0, 0 };
        fwrite(&c, sizeof c, 1, f);
    }
    int64_t sd = steps_done;
    fwrite(&sd, sizeof sd, 1, f);
    fwrite(&rng, sizeof rng, 1, f);
    if (ferror(f) || fclose(f) != 0) {
        fprintf(stderr, "ig: save failed\n");
        remove(tmp);
        return -1;
    }
    if (rename(tmp, path) != 0) { perror("ig: rename ckpt"); return -1; }
    return 0;
}

ig_MLP *ig_mlp_load2(const char *path, long *steps_out, uint64_t *rng_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ig: cannot open %s\n", path); return NULL; }
    char magic[4];
    int vers, ctx, nh;
    float s1, s2;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "IGCK", 4) != 0
        || fread(&vers, sizeof vers, 1, f) != 1
        || (vers != 1 && vers != 2 && vers != 3)
        || fread(&ctx, sizeof ctx, 1, f) != 1
        || fread(&nh, sizeof nh, 1, f) != 1
        || fread(&s1, sizeof s1, 1, f) != 1
        || fread(&s2, sizeof s2, 1, f) != 1) {
        fprintf(stderr, "ig: bad checkpoint %s\n", path);
        fclose(f);
        return NULL;
    }
    ig_set_scale(s1);
    ig_set_gscale(s2);
    ig_MLP *m = ig_mlp_new(ctx, nh, 1);
    ig_Value **lists[4] = { m->W1, m->b1, m->W2, m->b2 };
    long ns[4] = { (long)ctx * IG_VOCAB * nh, nh,
                   (long)nh * IG_VOCAB, IG_VOCAB };
    for (int gi = 0; gi < 4; gi++) {
        for (long i = 0; i < ns[gi]; i++) {
            float d;
            uint8_t code;
            if (fread(&d, sizeof d, 1, f) != 1 || fread(&code, 1, 1, f) != 1) {
                fprintf(stderr, "ig: truncated checkpoint %s\n", path);
                ig_mlp_free(m);
                fclose(f);
                return NULL;
            }
            lists[gi][i]->data = d;
        }
    }
    if (vers >= 2) {
        adam_ensure((long)ctx * IG_VOCAB * nh + nh
                    + (long)nh * IG_VOCAB + IG_VOCAB);
        if (fread(&adam_t, sizeof adam_t, 1, f) != 1) {
            fprintf(stderr, "ig: ckpt missing adam t\n");
            ig_mlp_free(m);
            fclose(f);
            return NULL;
        }
        for (long i = 0; i < adam_n; i++) {
            if (fread(&adam_state[i], sizeof(AdamCell), 1, f) != 1) {
                fprintf(stderr, "ig: ckpt missing adam state\n");
                ig_mlp_free(m);
                fclose(f);
                return NULL;
            }
        }
    } else {
        ig_mlp_adam_reset(); /* v1: no optimizer state stored */
    }
    if (vers >= 3) {
        int64_t sd = 0;
        uint64_t rg = 0;
        if (fread(&sd, sizeof sd, 1, f) != 1 || fread(&rg, sizeof rg, 1, f) != 1) {
            fprintf(stderr, "ig: ckpt missing progress state\n");
            ig_mlp_free(m);
            fclose(f);
            return NULL;
        }
        if (steps_out) *steps_out = (long)sd;
        if (rng_out) *rng_out = rg;
    }
    fclose(f);
    return m;
}
