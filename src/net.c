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

/* ---- weight-update numerics (E7) ---- */
int ig_sr = 1;             /* stochastic rounding of parameter storage */
int ig_gscale_dynamic = 1; /* pure-int: per-tensor pow2 absmax gradient scale */
int ig_w2 = 1;             /* two-byte weights: 16-bit master code (E8, D7 default) */
int ig_w2_sr = 0;          /* SR on the low byte (off by default: not needed, E8) */
float ig_wd = 0.0f;        /* decoupled weight decay (AdamW / Lion) */
int ig_lion_m = -1;        /* Lion momentum grid, IG_OPTQ_*; -1 = auto */
static uint64_t sr_state = 0x9E3779B97F4A7C15ull;
static long st_n, st_subgrid, st_moved;

static float sr_u(void)
{ /* xorshift64 -> U[0,1) with 24 bits */
    sr_state ^= sr_state << 13;
    sr_state ^= sr_state >> 7;
    sr_state ^= sr_state << 17;
    return (float)(sr_state >> 40) / 16777216.0f;
}

void ig_mlp_update_stats(long *n, long *subgrid, long *moved, int reset)
{
    if (n) *n = st_n;
    if (subgrid) *subgrid = st_subgrid;
    if (moved) *moved = st_moved;
    if (reset) st_n = st_subgrid = st_moved = 0;
}

/* Store an updated parameter: grid round-trip (SR by default) + stats.
 * "subgrid" = |step| below half the local grid spacing, i.e. an update
 * that round-to-nearest would silently discard. */
static void store_param(ig_Value *p, float step)
{
    float old = p->data;
    if (!ig_quant_enabled || ig_fp_params) { p->data = old - step; return; }
    int moved;
    if (ig_w2) {
        /* Two-byte weight: the update lands in the 16-bit master code;
         * the forward value is its high byte. */
        if (p->wq == 0 && old != 0.0f) p->wq = ig_w16_encode(old, -1.0f);
        int16_t wq0 = p->wq;
        float tot = ig_w16_full(wq0) - step;
        p->wq = ig_w16_encode(tot, ig_w2_sr ? sr_u() : -1.0f);
        p->data = ig_w16_fwd(p->wq);
        moved = p->wq != wq0;
    } else {
        float nv = old - step;
        p->data = ig_sr ? ig_rt_w_sr(nv, sr_u()) : ig_rt_w(nv);
        moved = p->data != old;
    }
    if (step != 0.0f) {
        float d, sc;
        ig_get_role_grid("w", &d, &sc);
        float half = 0.5f * 0.69314718f * d * (1.0f / sc + fabsf(old));
        st_n++;
        if (fabsf(step) < half) st_subgrid++;
        if (moved) st_moved++;
    }
}

/* Gradient scale actually used for a parameter group. Dynamic mode picks
 * the largest power of two that keeps max|g| inside the grid (absmax
 * scaling, one integer max-reduction per tensor per step): saturation is
 * impossible by construction, and range is spent where the gradients are.
 * A fixed global gscale clamps (E7: 2^16 diverges once updates are real). */
static float group_gscale(ig_Value *const *ps, long n)
{
    if (!ig_gscale_dynamic) return ig_get_gscale();
    float mx = 0.0f;
    for (long i = 0; i < n; i++) {
        float a = fabsf(ps[i]->grad);
        if (a > mx) mx = a;
    }
    if (mx == 0.0f) return ig_get_gscale();
    float lim = ig_max_abs(IG_DELTA, 1.0f); /* |g|*scale must stay <= lim */
    return exp2f(floorf(log2f(lim / mx)));
}

static float grad_q(float g, float gs)
{
    return ig_pure_int ? ig_decode_s(ig_encode_s(g, gs), gs) : g;
}
static float frand01(void)
{ /* xorshift32 */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state >> 8) / 16777216.0f;
}

/* Registration order for Adam/checkpoint flat indexing: W1, b1, W2, b2,
 * W2b, b2b — keep allocation order in sync with mlp_nparams and save2. */
ig_MLP *ig_mlp_new2(int ctx, int nh, int nh2, unsigned seed)
{
    if (ctx < 1 || nh < 1 || nh2 < 0) return NULL;
    ig_MLP *m = calloc(1, sizeof *m);
    if (!m) { fprintf(stderr, "ig: oom\n"); exit(1); }
    rng_state = seed ? seed : 0x9E3779B9u;
    sr_state = 0x9E3779B97F4A7C15ull ^ ((uint64_t)rng_state << 17) ^ rng_state;
    int nin = ctx * IG_VOCAB;
    int src = nh2 ? nh2 : nh;

    m->ctx = ctx;
    m->nh = nh;
    m->nh2 = nh2;
    m->W1 = malloc(sizeof(ig_Value *) * (size_t)nin * (size_t)nh);
    m->b1 = calloc((size_t)nh, sizeof(ig_Value *));
    m->W2 = malloc(sizeof(ig_Value *) * (size_t)src * IG_VOCAB);
    m->b2 = calloc(IG_VOCAB, sizeof(ig_Value *));
    m->W2b = nh2 ? malloc(sizeof(ig_Value *) * (size_t)nh * nh2) : NULL;
    m->b2b = nh2 ? calloc((size_t)nh2, sizeof(ig_Value *)) : NULL;
    if (!m->W1 || !m->b1 || !m->W2 || !m->b2 || (nh2 && (!m->W2b || !m->b2b))) {
        fprintf(stderr, "ig: oom\n");
        exit(1);
    }
    for (long i = 0; i < (long)nin * nh; i++)
        m->W1[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * 0.05f);
    for (int i = 0; i < nh; i++) m->b1[i] = ig_value_persist(0.0f);
    float w2b = 0.3f / sqrtf((float)src);
    for (long i = 0; i < (long)src * IG_VOCAB; i++)
        m->W2[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * w2b);
    for (int i = 0; i < IG_VOCAB; i++) m->b2[i] = ig_value_persist(0.0f);
    if (nh2) {
        float wb = 0.3f / sqrtf((float)nh);
        for (long i = 0; i < (long)nh * nh2; i++)
            m->W2b[i] = ig_value_persist((frand01() * 2.0f - 1.0f) * wb);
        for (int i = 0; i < nh2; i++) m->b2b[i] = ig_value_persist(0.0f);
    }
    return m;
}

ig_MLP *ig_mlp_new(int ctx, int nh, unsigned seed)
{
    return ig_mlp_new2(ctx, nh, 0, seed);
}

void ig_mlp_free(ig_MLP *m)
{
    if (!m) return;
    int nin = m->ctx * IG_VOCAB;
    for (long i = 0; i < (long)nin * m->nh; i++) ig_value_free_persist(m->W1[i]);
    for (int i = 0; i < m->nh; i++) ig_value_free_persist(m->b1[i]);
    int src = m->nh2 ? m->nh2 : m->nh;
    for (long i = 0; i < (long)src * IG_VOCAB; i++) ig_value_free_persist(m->W2[i]);
    for (int i = 0; i < IG_VOCAB; i++) ig_value_free_persist(m->b2[i]);
    if (m->nh2) {
        for (long i = 0; i < (long)m->nh * m->nh2; i++) ig_value_free_persist(m->W2b[i]);
        for (int i = 0; i < m->nh2; i++) ig_value_free_persist(m->b2b[i]);
    }
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->W2b); free(m->b2b);
    free(m);
}

ig_Value **ig_mlp_forward(ig_MLP *m, const uint8_t *in)
{
    int nh = m->nh;
    ig_Value **hv = malloc(sizeof(ig_Value *) * (size_t)nh);
    ig_Value **logits = malloc(sizeof(ig_Value *) * IG_VOCAB);

    /* Every layer = int8 GEMM semantics (D6): products and partial sums are
     * exact (ig_fx_mode 2 emulates the int32 accumulator), and each output
     * is quantized exactly ONCE via ig_requant (clamp-aware STE). Earlier
     * versions re-rounded every partial sum on the log grid, which swamps
     * terms below ~3.5% of the running sum and is order-dependent. */
    ig_fx_mode = 2;
    /* One-hot inputs: only the active byte per position contributes. */
    for (int j = 0; j < nh; j++) {
        ig_Value *acc = m->b1[j];
        for (int p = 0; p < m->ctx; p++) {
            ig_Value *w = m->W1[(size_t)(p * IG_VOCAB + in[p]) * nh + j];
            acc = ig_add(acc, w); /* x_i == 1, so plain add; grad d/dW = 1 */
        }
        hv[j] = ig_relu(ig_requant(acc, IG_GRID_ACT)); /* relu of a grid value stays on grid */
    }
    ig_Value **srcv = hv;
    int nsrc = nh;
    ig_Value **hv2 = NULL;
    if (m->nh2) {
        hv2 = malloc(sizeof(ig_Value *) * (size_t)m->nh2);
        for (int j = 0; j < m->nh2; j++) {
            ig_Value *acc = m->b2b[j];
            for (int i = 0; i < nh; i++)
                acc = ig_add(acc, ig_mul(hv[i], m->W2b[(size_t)i * m->nh2 + j]));
            hv2[j] = ig_relu(ig_requant(acc, IG_GRID_ACT));
        }
        srcv = hv2;
        nsrc = m->nh2;
    }
    /* D6 hybrid grid: the output/CE path uses the fixed-point byte grid
     * (constant absolute error), since CE noise scales with the ABSOLUTE
     * logit error. */
    int qsav = ig_quant_enabled;
    if (ig_fp_logits) ig_quant_enabled = 0; /* isolation: float logits */
    for (int k = 0; k < IG_VOCAB; k++) {
        ig_Value *acc = m->b2[k];
        for (int j = 0; j < nsrc; j++)
            acc = ig_add(acc, ig_mul(srcv[j], m->W2[(size_t)j * IG_VOCAB + k]));
        logits[k] = ig_requant(acc, IG_GRID_FX); /* single requantization per logit */
    }
    ig_fx_mode = 0;
    ig_quant_enabled = qsav;
    free(hv2);
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
    /* CE arithmetic is EXACT (mode 2) over quantized inputs; only the
     * final CE value is requantized (fx step). Quantizing lse and target
     * separately is a trap: both saturate the fx grid at the same edge,
     * so loss = 12.7 - 12.7 = 0 — a confidently WRONG prediction reports
     * zero loss (the E6 depth-2 "0.09 bpb" artifact). CE itself carries
     * the error budget and needs no intermediate quantization. */
    ig_fx_mode = ig_loss_fp || ig_fp_logits ? 0 : 2;
    ig_Value *lse = ig_logsumexp((const ig_Value **)logits, IG_VOCAB);
    ig_Value *loss = ig_sub(lse, logits[target]);
    /* CE is NOT requantized: it is a transient scalar (never stored), and
     * clamping it would kill gradient on "very wrong" predictions. */
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
    int src = m->nh2 ? m->nh2 : m->nh;
    zero_grad_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh);
    zero_grad_list(m->b1, m->nh);
    zero_grad_list(m->W2, (long)src * IG_VOCAB);
    zero_grad_list(m->b2, IG_VOCAB);
    if (m->nh2) {
        zero_grad_list(m->W2b, (long)m->nh * m->nh2);
        zero_grad_list(m->b2b, m->nh2);
    }
}

static void sgd_list(ig_Value **ps, long n, float lr)
{
    float gs = ig_pure_int ? group_gscale(ps, n) : 0.0f;
    for (long i = 0; i < n; i++) {
        ig_Value *p = ps[i];
        float g = grad_q(p->grad, gs); /* D1: gradient on the byte grid */
        store_param(p, lr * g);
    }
}

static ig_GradStat stats_list(ig_Value **ps, long n)
{
    float gs = group_gscale(ps, n);
    float gmax = ig_max_abs(IG_DELTA, gs);
    ig_GradStat s = { 0 };
    s.n = n;
    long ztrue = 0, zq = 0, sat = 0;
    double zmass = 0.0, smass = 0.0;
    for (long i = 0; i < n; i++) {
        float g = ps[i]->grad;
        float a = fabsf(g);
        s.mean_abs += a;
        if (a > s.max_abs) s.max_abs = a;
        uint8_t c = ig_encode_s(g, gs);
        if (g == 0.0f) ztrue++;             /* structural zero (e.g. one-hot) */
        else if (c == 0) { zq++; zmass += a; } /* collapsed by quantization */
        else if (a > gmax) { sat++; smass += a - gmax; }
    }
    if (n > 0) {
        double tot = s.mean_abs;
        s.mean_abs /= (double)n;
        s.zero_frac = (float)zq / (float)n; /* collapse only; true zeros excluded */
        s.true_zero_frac = (float)ztrue / (float)n;
        s.sat_frac = (float)sat / (float)n;
        if (tot > 0.0) {
            s.collapse_mass = (float)(zmass / tot);
            s.sat_mass = (float)(smass / tot);
        }
    }
    return s;
}

ig_GradStat ig_mlp_grad_stats(const ig_MLP *m, int group)
{
    int src = m->nh2 ? m->nh2 : m->nh;
    switch (group) {
    case 0: return stats_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh);
    case 1: return stats_list(m->b1, m->nh);
    case 2: return stats_list(m->W2, (long)src * IG_VOCAB);
    case 3: return stats_list(m->b2, IG_VOCAB);
    case 4: return stats_list(m->W2b, (long)m->nh * m->nh2);
    case 5: return stats_list(m->b2b, m->nh2);
    default: { ig_GradStat z = { 0 }; return z; }
    }
}

void ig_mlp_sgd_update(ig_MLP *m, float lr)
{
    int src = m->nh2 ? m->nh2 : m->nh;
    sgd_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh, lr);
    sgd_list(m->b1, m->nh, lr);
    sgd_list(m->W2, (long)src * IG_VOCAB, lr);
    sgd_list(m->b2, IG_VOCAB, lr);
    if (m->nh2) {
        sgd_list(m->W2b, (long)m->nh * m->nh2, lr);
        sgd_list(m->b2b, m->nh2, lr);
    }
}

static void scale_list(ig_Value **ps, long n, float s)
{
    for (long i = 0; i < n; i++) ps[i]->grad *= s;
}

void ig_mlp_scale_grads(ig_MLP *m, float s)
{
    int src = m->nh2 ? m->nh2 : m->nh;
    scale_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh, s);
    scale_list(m->b1, m->nh, s);
    scale_list(m->W2, (long)src * IG_VOCAB, s);
    scale_list(m->b2, IG_VOCAB, s);
    if (m->nh2) {
        scale_list(m->W2b, (long)m->nh * m->nh2, s);
        scale_list(m->b2b, m->nh2, s);
    }
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
    int src = m->nh2 ? m->nh2 : m->nh;
    long n = (long)m->ctx * IG_VOCAB * m->nh + m->nh
           + (long)src * IG_VOCAB + IG_VOCAB;
    if (m->nh2) n += (long)m->nh * m->nh2 + m->nh2;
    return n;
}

static float *lion_m;
static long lion_n;

void ig_mlp_adam_reset(void)
{
    free(adam_state);
    adam_state = NULL;
    adam_n = 0;
    adam_t = 0;
    free(lion_m); /* resets ALL optimizer state (Adam + Lion) */
    lion_m = NULL;
    lion_n = 0;
}

void ig_mlp_adam_update(ig_MLP *m, float lr)
{
    adam_ensure(mlp_nparams(m));
    adam_t++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)adam_t);
    float bc2 = 1.0f - powf(b2, (float)adam_t);
    long k = 0;
    int src = m->nh2 ? m->nh2 : m->nh;
    ig_Value **lists[6] = { m->W1, m->b1, m->W2, m->b2, m->W2b, m->b2b };
    long ns[6] = { (long)m->ctx * IG_VOCAB * m->nh, m->nh,
                   (long)src * IG_VOCAB, IG_VOCAB,
                   (long)m->nh * m->nh2, m->nh2 };
    int ng = m->nh2 ? 6 : 4;
    for (int gi = 0; gi < ng; gi++) {
        float gs = ig_pure_int ? group_gscale(lists[gi], ns[gi]) : 0.0f;
        for (long i = 0; i < ns[gi]; i++) {
            ig_Value *p = lists[gi][i];
            float g = grad_q(p->grad, gs); /* D1 */
            AdamCell *c = &adam_state[k++];
            c->m = b1 * c->m + (1.0f - b1) * g;
            c->v = b2 * c->v + (1.0f - b2) * g * g;
            float step = lr * ((c->m / bc1) / (sqrtf(c->v / bc2) + eps)
                               + ig_wd * p->data);
            store_param(p, step);
        }
    }
}

/* ---------------- Lion (E9) ----------------
 * Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms":
 *   c = b1*m + (1-b1)*g;  w -= lr*(sign(c) + wd*w);  m = b2*m + (1-b2)*g
 * One state tensor, a constant-magnitude update, no sqrt, no division.
 * Int view: sign(c) needs no arithmetic at all — equal signs of m and g
 * give that sign, otherwise compare log codes |m| vs |g| with the
 * constant offset log(b1/(1-b1)). The EMA of m is a log-domain add (LUT).
 *
 * m lives on a log grid with a per-tensor power-of-two absmax scale
 * re-chosen every step (as the gradients). lion_m holds the grid VALUES
 * (exactly representable, like p->data): b2 = 16-bit symlog codes RTN
 * (default when quantized), b1 / b1sr = 1-byte RTN / SR (ablations),
 * f32 = float (oracle). b2 (EMA weight) = 0.99 moves m by ~1% per step,
 * below the 1-byte half-step (3.6%): 1-byte RTN freezes m (E7 mechanism). */
static float *lion_tmp;
static long lion_tmp_n;

static void lion_ensure(long total, long maxgroup)
{
    if (lion_n != total) {
        free(lion_m);
        lion_m = calloc((size_t)total, sizeof(float));
        if (!lion_m) { fprintf(stderr, "ig: oom lion\n"); exit(1); }
        lion_n = total;
    }
    if (lion_tmp_n < maxgroup) {
        free(lion_tmp);
        lion_tmp = malloc(sizeof(float) * (size_t)maxgroup);
        if (!lion_tmp) { fprintf(stderr, "ig: oom lion\n"); exit(1); }
        lion_tmp_n = maxgroup;
    }
}

int ig_lion_m_mode(void)
{
    if (ig_lion_m >= 0) return ig_lion_m;
    return ig_quant_enabled ? IG_OPTQ_B2 : IG_OPTQ_F32;
}

static float optq(float x, int mode, float s)
{
    switch (mode) {
    case IG_OPTQ_B2:   return ig_dec16(ig_enc16(x, -1.0f, IG_DELTA, s), IG_DELTA, s);
    case IG_OPTQ_B1:   return ig_rt2(x, IG_DELTA, s);
    case IG_OPTQ_B1SR: return ig_rt2_sr(x, IG_DELTA, s, sr_u());
    default:           return x;
    }
}

int ig_mlp_opt_state(void)
{
    return (adam_t > 0 ? 1 : 0) | (lion_m ? 2 : 0);
}

void ig_mlp_lion_update(ig_MLP *m, float lr)
{
    const float b1 = 0.9f, b2 = 0.99f;
    int src = m->nh2 ? m->nh2 : m->nh;
    ig_Value **lists[6] = { m->W1, m->b1, m->W2, m->b2, m->W2b, m->b2b };
    long ns[6] = { (long)m->ctx * IG_VOCAB * m->nh, m->nh,
                   (long)src * IG_VOCAB, IG_VOCAB,
                   (long)m->nh * m->nh2, m->nh2 };
    int ng = m->nh2 ? 6 : 4;
    long maxg = 0;
    for (int gi = 0; gi < ng; gi++) if (ns[gi] > maxg) maxg = ns[gi];
    lion_ensure(mlp_nparams(m), maxg);
    int mode = ig_lion_m_mode();
    float lim = ig_max_abs(IG_DELTA, 1.0f);
    long k = 0;
    for (int gi = 0; gi < ng; gi++) {
        float gs = ig_pure_int ? group_gscale(lists[gi], ns[gi]) : 0.0f;
        float mx = 0.0f;
        for (long i = 0; i < ns[gi]; i++) {
            ig_Value *p = lists[gi][i];
            float g = grad_q(p->grad, gs);
            float mo = lion_m[k + i];
            float c = b1 * mo + (1.0f - b1) * g;
            float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            store_param(p, lr * (u + ig_wd * p->data));
            float mn = b2 * mo + (1.0f - b2) * g;
            lion_tmp[i] = mn;
            if (fabsf(mn) > mx) mx = fabsf(mn);
        }
        float s = mx > 0.0f ? exp2f(floorf(log2f(lim / mx))) : 1.0f;
        for (long i = 0; i < ns[gi]; i++)
            lion_m[k + i] = optq(lion_tmp[i], mode, s);
        k += ns[gi];
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
    int vers = 7; /* v2: Adam, v3: +steps/RNG, v4: +nh2 (E6), v5: +SR rng,
                   * v6: +16-bit master codes (two-byte weights, E8),
                   * v7: +Lion momentum (E9) */
    fwrite(magic, 1, 4, f);
    fwrite(&vers, sizeof vers, 1, f);
    fwrite(&m->ctx, sizeof m->ctx, 1, f);
    fwrite(&m->nh, sizeof m->nh, 1, f);
    if (vers >= 4) fwrite(&m->nh2, sizeof m->nh2, 1, f);
    float s1 = ig_get_scale(), s2 = ig_get_gscale();
    fwrite(&s1, sizeof s1, 1, f);
    fwrite(&s2, sizeof s2, 1, f);
    int src = m->nh2 ? m->nh2 : m->nh;
    ig_Value **lists[6] = { m->W1, m->b1, m->W2, m->b2, m->W2b, m->b2b };
    long ns[6] = { (long)m->ctx * IG_VOCAB * m->nh, m->nh,
                   (long)src * IG_VOCAB, IG_VOCAB,
                   (long)m->nh * m->nh2, m->nh2 };
    int ng = m->nh2 ? 6 : 4;
    for (int gi = 0; gi < ng; gi++) {
        for (long i = 0; i < ns[gi]; i++) {
            float d = lists[gi][i]->data;
            uint8_t code = ig_encode_w(d); /* the byte-only view of the model */
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
    fwrite(&sr_state, sizeof sr_state, 1, f); /* v5 */
    for (int gi = 0; gi < ng; gi++) /* v6 */
        for (long i = 0; i < ns[gi]; i++)
            fwrite(&lists[gi][i]->wq, sizeof(int16_t), 1, f);
    int64_t ln = lion_m ? lion_n : 0; /* v7 */
    fwrite(&ln, sizeof ln, 1, f);
    if (ln) fwrite(lion_m, sizeof(float), (size_t)ln, f);
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
    int vers, ctx, nh, nh2 = 0;
    float s1, s2;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "IGCK", 4) != 0
        || fread(&vers, sizeof vers, 1, f) != 1
        || vers < 1 || vers > 7
        || fread(&ctx, sizeof ctx, 1, f) != 1
        || fread(&nh, sizeof nh, 1, f) != 1
        || (vers >= 4 && fread(&nh2, sizeof nh2, 1, f) != 1)
        || fread(&s1, sizeof s1, 1, f) != 1
        || fread(&s2, sizeof s2, 1, f) != 1) {
        fprintf(stderr, "ig: bad checkpoint %s\n", path);
        fclose(f);
        return NULL;
    }
    ig_set_scale(s1);
    ig_set_gscale(s2);
    ig_MLP *m = ig_mlp_new2(ctx, nh, nh2, 1);
    int src = nh2 ? nh2 : nh;
    ig_Value **lists[6] = { m->W1, m->b1, m->W2, m->b2, m->W2b, m->b2b };
    long ns[6] = { (long)ctx * IG_VOCAB * nh, nh,
                   (long)src * IG_VOCAB, IG_VOCAB,
                   (long)nh * nh2, nh2 };
    int ng = nh2 ? 6 : 4;
    for (int gi = 0; gi < ng; gi++) {
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
        adam_ensure(mlp_nparams(m));
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
        if (vers >= 5 && fread(&sr_state, sizeof sr_state, 1, f) != 1) {
            fprintf(stderr, "ig: ckpt missing SR state\n");
            ig_mlp_free(m);
            fclose(f);
            return NULL;
        }
        for (int gi = 0; vers >= 6 && gi < ng; gi++) {
            for (long i = 0; i < ns[gi]; i++) {
                if (fread(&lists[gi][i]->wq, sizeof(int16_t), 1, f) != 1) {
                    fprintf(stderr, "ig: ckpt missing master codes\n");
                    ig_mlp_free(m);
                    fclose(f);
                    return NULL;
                }
            }
        }
        if (vers >= 7) {
            int64_t ln = 0;
            free(lion_m);
            lion_m = NULL;
            lion_n = 0;
            if (fread(&ln, sizeof ln, 1, f) != 1 || (ln && ln != mlp_nparams(m))) {
                fprintf(stderr, "ig: ckpt bad Lion state\n");
                ig_mlp_free(m);
                fclose(f);
                return NULL;
            }
            if (ln) {
                lion_m = malloc(sizeof(float) * (size_t)ln);
                if (!lion_m || fread(lion_m, sizeof(float), (size_t)ln, f) != (size_t)ln) {
                    fprintf(stderr, "ig: ckpt truncated Lion state\n");
                    ig_mlp_free(m);
                    fclose(f);
                    return NULL;
                }
                lion_n = ln;
            }
        }
        if (steps_out) *steps_out = (long)sd;
        if (rng_out) *rng_out = rg;
    }
    fclose(f);
    return m;
}
