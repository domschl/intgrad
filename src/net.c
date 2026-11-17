#include "ig/net.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ig/logquant.h"

int ig_w2 = 1;      /* two-byte weights (D7) */
int ig_w2_sr = 0;   /* SR on the low byte (RTN suffices) */
int ig_sr = 1;      /* SR for one-byte storage */
float ig_wd = 0.1f; /* Lion decoupled weight decay */

static uint32_t init_rng;
static uint64_t sr_state = 0x9E3779B97F4A7C15ull;
static long st_n, st_subgrid, st_moved;

static void *xmalloc(size_t n)
{
    void *p = calloc(1, n ? n : 1);
    if (!p) { fprintf(stderr, "ig: out of memory\n"); exit(1); }
    return p;
}

static float sr_u(void)
{ /* xorshift64 -> U[0,1), 24 bits */
    sr_state ^= sr_state << 13;
    sr_state ^= sr_state >> 7;
    sr_state ^= sr_state << 17;
    return (float)(sr_state >> 40) / 16777216.0f;
}

static float frand01(void)
{ /* xorshift32, init only */
    init_rng ^= init_rng << 13;
    init_rng ^= init_rng >> 17;
    init_rng ^= init_rng << 5;
    return (float)(init_rng >> 8) / 16777216.0f;
}

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

int ig_mlp_groups(const ig_MLP *m, ig_Value **lists[IG_NGROUPS],
                  long ns[IG_NGROUPS])
{
    lists[0] = m->W1; ns[0] = (long)m->ctx * IG_VOCAB * m->nh;
    lists[1] = m->b1; ns[1] = m->nh;
    lists[2] = m->W2; ns[2] = (long)m->nh * IG_VOCAB;
    lists[3] = m->b2; ns[3] = IG_VOCAB;
    return IG_NGROUPS;
}

long ig_mlp_nparams(const ig_MLP *m)
{
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS], n = 0;
    for (int g = 0; g < ig_mlp_groups(m, l, ns); g++) n += ns[g];
    return n;
}

ig_MLP *ig_mlp_new(int ctx, int nh, unsigned seed)
{
    if (ctx < 1 || nh < 1) return NULL;
    ig_MLP *m = xmalloc(sizeof *m);
    init_rng = seed ? seed : 0x9E3779B9u;
    sr_state = 0x9E3779B97F4A7C15ull ^ ((uint64_t)init_rng << 17) ^ init_rng;
    m->ctx = ctx;
    m->nh = nh;
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    /* sizes first (pointers are NULL until allocated) */
    ig_mlp_groups(m, l, ns);
    m->W1 = xmalloc(sizeof(ig_Value *) * (size_t)ns[0]);
    m->b1 = xmalloc(sizeof(ig_Value *) * (size_t)ns[1]);
    m->W2 = xmalloc(sizeof(ig_Value *) * (size_t)ns[2]);
    m->b2 = xmalloc(sizeof(ig_Value *) * (size_t)ns[3]);
    for (long i = 0; i < ns[0]; i++)
        m->W1[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * 0.05f);
    for (long i = 0; i < ns[1]; i++) m->b1[i] = ig_value_persist(0.0f);
    float w2b = 0.3f / sqrtf((float)nh);
    for (long i = 0; i < ns[2]; i++)
        m->W2[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * w2b);
    for (long i = 0; i < ns[3]; i++) m->b2[i] = ig_value_persist(0.0f);
    return m;
}

void ig_mlp_free(ig_MLP *m)
{
    if (!m) return;
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    for (int g = 0; g < ig_mlp_groups(m, l, ns); g++) {
        for (long i = 0; i < ns[g]; i++) ig_value_free_persist(l[g][i]);
        free(l[g]);
    }
    free(m);
}

ig_Value **ig_mlp_forward(ig_MLP *m, const uint8_t *in)
{
    int nh = m->nh;
    ig_Value **hv = xmalloc(sizeof(ig_Value *) * (size_t)nh);
    ig_Value **logits = xmalloc(sizeof(ig_Value *) * IG_VOCAB);

    /* int8-GEMM semantics (D6): products and partial sums are exact
     * (ig_fx_mode 2 emulates the int32 accumulator); each output is
     * quantized exactly ONCE via ig_requant (clamp-aware STE). */
    ig_fx_mode = 2;
    for (int j = 0; j < nh; j++) {
        ig_Value *acc = m->b1[j];
        for (int p = 0; p < m->ctx; p++) /* one-hot input: plain add */
            acc = ig_add(acc, m->W1[(long)(p * IG_VOCAB + in[p]) * nh + j]);
        hv[j] = ig_relu(ig_requant(acc, IG_GRID_ACT));
    }
    /* Logits use the fixed-point byte grid: CE error scales with the
     * ABSOLUTE logit error. */
    for (int k = 0; k < IG_VOCAB; k++) {
        ig_Value *acc = m->b2[k];
        for (int j = 0; j < nh; j++)
            acc = ig_add(acc, ig_mul(hv[j], m->W2[(long)j * IG_VOCAB + k]));
        logits[k] = ig_requant(acc, IG_GRID_FX);
    }
    ig_fx_mode = 0;
    free(hv);
    return logits;
}

ig_Value *ig_mlp_loss(ig_MLP *m, ig_Value **logits, uint8_t target)
{
    (void)m;
    /* CE is EXACT over the quantized logits and never quantized itself:
     * clamping lse and target on the same grid edge makes a confidently
     * WRONG prediction score 0. */
    ig_fx_mode = 2;
    ig_Value *lse = ig_logsumexp((const ig_Value **)logits, IG_VOCAB);
    ig_Value *loss = ig_sub(lse, logits[target]);
    ig_fx_mode = 0;
    return loss;
}

void ig_mlp_zero_grad(ig_MLP *m)
{
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    for (int g = 0; g < ig_mlp_groups(m, l, ns); g++)
        for (long i = 0; i < ns[g]; i++) l[g][i]->grad = 0.0f;
}

void ig_mlp_scale_grads(ig_MLP *m, float s)
{
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    for (int g = 0; g < ig_mlp_groups(m, l, ns); g++)
        for (long i = 0; i < ns[g]; i++) l[g][i]->grad *= s;
}

/* ------------------------------------------------------------------ */
/* Parameter storage + gradient quantization                          */
/* ------------------------------------------------------------------ */

void ig_mlp_update_stats(long *n, long *subgrid, long *moved, int reset)
{
    if (n) *n = st_n;
    if (subgrid) *subgrid = st_subgrid;
    if (moved) *moved = st_moved;
    if (reset) st_n = st_subgrid = st_moved = 0;
}

/* Apply p -= step on the storage grid + update statistics. */
static void store_param(ig_Value *p, float step)
{
    float old = p->data;
    if (!ig_quant_enabled) { p->data = old - step; return; }
    int moved;
    if (ig_w2) {
        if (p->wq == 0 && old != 0.0f) p->wq = ig_w16_encode(old, -1.0f);
        int16_t wq0 = p->wq;
        p->wq = ig_w16_encode(ig_w16_full(wq0) - step, ig_w2_sr ? sr_u() : -1.0f);
        p->data = ig_w16_fwd(p->wq);
        moved = p->wq != wq0;
    } else {
        float nv = old - step;
        p->data = ig_sr ? ig_roundtrip_sr(nv, sr_u()) : ig_roundtrip(nv);
        moved = p->data != old;
    }
    if (step != 0.0f) {
        float half = 0.5f * 0.69314718f * IG_DELTA
                   * (1.0f / ig_get_scale() + fabsf(old));
        st_n++;
        if (fabsf(step) < half) st_subgrid++;
        if (moved) st_moved++;
    }
}

/* Per-tensor power-of-two absmax scale: the largest pow2 that keeps
 * max|x| inside the log grid. Saturation is impossible by construction. */
static float absmax_scale(float mx)
{
    if (!(mx > 0.0f)) return 1.0f;
    return exp2f(floorf(log2f(ig_max_abs(IG_DELTA, 1.0f) / mx)));
}

static float group_gscale(ig_Value *const *ps, long n)
{
    float mx = 0.0f;
    for (long i = 0; i < n; i++) {
        float a = fabsf(ps[i]->grad);
        if (a > mx) mx = a;
    }
    return absmax_scale(mx);
}

static float grad_q(float g, float gs)
{
    return ig_pure_int ? ig_decode_s(ig_encode_s(g, gs), gs) : g;
}

ig_GradStat ig_mlp_grad_stats(const ig_MLP *m, int group)
{
    ig_GradStat s = { 0 };
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    if (group < 0 || group >= ig_mlp_groups(m, l, ns)) return s;
    ig_Value **ps = l[group];
    long n = ns[group];
    float gs = group_gscale(ps, n);
    float gmax = ig_max_abs(IG_DELTA, gs);
    long ztrue = 0, zq = 0, sat = 0;
    double zmass = 0.0, smass = 0.0;
    s.n = n;
    for (long i = 0; i < n; i++) {
        float g = ps[i]->grad, a = fabsf(g);
        s.mean_abs += a;
        if (a > s.max_abs) s.max_abs = a;
        if (g == 0.0f) ztrue++;
        else if (ig_encode_s(g, gs) == 0) { zq++; zmass += a; }
        else if (a > gmax) { sat++; smass += a - gmax; }
    }
    if (n > 0) {
        double tot = s.mean_abs;
        s.mean_abs /= (double)n;
        s.zero_frac = (float)zq / (float)n;
        s.true_zero_frac = (float)ztrue / (float)n;
        s.sat_frac = (float)sat / (float)n;
        if (tot > 0.0) {
            s.collapse_mass = (float)(zmass / tot);
            s.sat_mass = (float)(smass / tot);
        }
    }
    return s;
}

/* ------------------------------------------------------------------ */
/* Lion                                                                */
/* ------------------------------------------------------------------ */
/* Int view: sign(c) needs no arithmetic when m and g agree in sign,
 * otherwise one log-code comparison |m| vs |g| (offset log(b1/(1-b1))).
 * The 16-bit momentum grid matters: b2 = 0.99 moves m ~1% per step,
 * below the 1-byte half-step (3.6%), so a 1-byte RTN momentum freezes. */

static float *lion_m, *lion_tmp;
static long lion_n, lion_tmp_n;

void ig_mlp_opt_reset(void)
{
    free(lion_m);
    lion_m = NULL;
    lion_n = 0;
}

static void lion_ensure(long total, long maxgroup)
{
    if (lion_n != total) {
        free(lion_m);
        lion_m = xmalloc(sizeof(float) * (size_t)total);
        lion_n = total;
    }
    if (lion_tmp_n < maxgroup) {
        free(lion_tmp);
        lion_tmp = xmalloc(sizeof(float) * (size_t)maxgroup);
        lion_tmp_n = maxgroup;
    }
}

void ig_mlp_lion_update(ig_MLP *m, float lr)
{
    const float b1 = 0.9f, b2 = 0.99f;
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS], maxg = 0;
    int ng = ig_mlp_groups(m, l, ns);
    for (int g = 0; g < ng; g++) if (ns[g] > maxg) maxg = ns[g];
    lion_ensure(ig_mlp_nparams(m), maxg);
    long k = 0;
    for (int g = 0; g < ng; g++) {
        float gs = ig_pure_int ? group_gscale(l[g], ns[g]) : 0.0f;
        float mx = 0.0f;
        for (long i = 0; i < ns[g]; i++) {
            ig_Value *p = l[g][i];
            float gr = grad_q(p->grad, gs);
            float mo = lion_m[k + i];
            float c = b1 * mo + (1.0f - b1) * gr;
            float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            store_param(p, lr * (u + ig_wd * p->data));
            float mn = b2 * mo + (1.0f - b2) * gr;
            lion_tmp[i] = mn;
            if (fabsf(mn) > mx) mx = fabsf(mn);
        }
        float s = absmax_scale(mx);
        for (long i = 0; i < ns[g]; i++)
            lion_m[k + i] = ig_quant_enabled
                ? ig_dec16(ig_enc16(lion_tmp[i], -1.0f, IG_DELTA, s), IG_DELTA, s)
                : lion_tmp[i];
        k += ns[g];
    }
}

/* ------------------------------------------------------------------ */
/* Checkpoints (v8)                                                    */
/* ------------------------------------------------------------------ */

#define CK_VERSION 8

int ig_mlp_save(const ig_MLP *m, const char *path, long steps_done,
                uint64_t rng)
{
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "ig: cannot write %s\n", tmp); return -1; }
    int32_t hdr[3] = { CK_VERSION, m->ctx, m->nh };
    float scale = ig_get_scale();
    fwrite("IGCK", 1, 4, f);
    fwrite(hdr, sizeof hdr, 1, f);
    fwrite(&scale, sizeof scale, 1, f);
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    for (int g = 0; g < ig_mlp_groups(m, l, ns); g++)
        for (long i = 0; i < ns[g]; i++) {
            fwrite(&l[g][i]->data, sizeof(float), 1, f);
            fwrite(&l[g][i]->wq, sizeof(int16_t), 1, f);
        }
    int64_t ln = lion_m ? lion_n : 0, sd = steps_done;
    fwrite(&ln, sizeof ln, 1, f);
    if (ln) fwrite(lion_m, sizeof(float), (size_t)ln, f);
    fwrite(&sd, sizeof sd, 1, f);
    fwrite(&rng, sizeof rng, 1, f);
    fwrite(&sr_state, sizeof sr_state, 1, f);
    if (ferror(f) || fclose(f) != 0) {
        fprintf(stderr, "ig: save failed\n");
        remove(tmp);
        return -1;
    }
    if (rename(tmp, path) != 0) { perror("ig: rename ckpt"); return -1; }
    return 0;
}

ig_MLP *ig_mlp_load(const char *path, long *steps_out, uint64_t *rng_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ig: cannot open %s\n", path); return NULL; }
    char magic[4];
    int32_t hdr[3];
    float scale;
    ig_MLP *m = NULL;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "IGCK", 4) != 0
        || fread(hdr, sizeof hdr, 1, f) != 1 || fread(&scale, sizeof scale, 1, f) != 1) {
        fprintf(stderr, "ig: bad checkpoint %s\n", path);
        goto fail;
    }
    if (hdr[0] != CK_VERSION) {
        fprintf(stderr, "ig: checkpoint %s has format v%d, need v%d\n",
                path, hdr[0], CK_VERSION);
        goto fail;
    }
    ig_set_scale(scale);
    m = ig_mlp_new(hdr[1], hdr[2], 1);
    if (!m) goto bad;
    ig_Value **l[IG_NGROUPS];
    long ns[IG_NGROUPS];
    for (int g = 0; g < ig_mlp_groups(m, l, ns); g++)
        for (long i = 0; i < ns[g]; i++)
            if (fread(&l[g][i]->data, sizeof(float), 1, f) != 1
                || fread(&l[g][i]->wq, sizeof(int16_t), 1, f) != 1) goto bad;
    int64_t ln, sd;
    uint64_t rg;
    if (fread(&ln, sizeof ln, 1, f) != 1 || (ln && ln != ig_mlp_nparams(m))) goto bad;
    ig_mlp_opt_reset();
    if (ln) {
        lion_ensure(ln, 0);
        if (fread(lion_m, sizeof(float), (size_t)ln, f) != (size_t)ln) goto bad;
    }
    if (fread(&sd, sizeof sd, 1, f) != 1 || fread(&rg, sizeof rg, 1, f) != 1
        || fread(&sr_state, sizeof sr_state, 1, f) != 1) goto bad;
    fclose(f);
    if (steps_out) *steps_out = (long)sd;
    if (rng_out) *rng_out = rg;
    return m;
bad:
    fprintf(stderr, "ig: truncated/corrupt checkpoint %s\n", path);
fail:
    ig_mlp_free(m);
    fclose(f);
    return NULL;
}
