#include "ig/net.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ig/logquant.h"

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
    for (int k = 0; k < IG_VOCAB; k++) {
        ig_Value *acc = m->b2[k];
        for (int j = 0; j < nh; j++)
            acc = ig_add(acc, ig_mul(hv[j], m->W2[(size_t)j * IG_VOCAB + k]));
        logits[k] = acc;
    }
    free(hv);
    return logits;
}

ig_Value *ig_mlp_loss(ig_MLP *m, ig_Value **logits, uint8_t target)
{
    (void)m;
    ig_Value *lse = ig_logsumexp((const ig_Value **)logits, IG_VOCAB);
    return ig_sub(lse, logits[target]);
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
        float v = p->data - lr * p->grad;
        p->data = ig_quant_enabled ? ig_roundtrip(v) : v;
    }
}

void ig_mlp_sgd_update(ig_MLP *m, float lr)
{
    sgd_list(m->W1, (long)m->ctx * IG_VOCAB * m->nh, lr);
    sgd_list(m->b1, m->nh, lr);
    sgd_list(m->W2, (long)m->nh * IG_VOCAB, lr);
    sgd_list(m->b2, IG_VOCAB, lr);
}
