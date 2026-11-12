#include "ig/value.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ig/logquant.h"

int ig_quant_enabled = 1;
int ig_pure_int = 0;
int ig_fx_mode = 0;

/* ------------------------------------------------------------------ */
/* Bump arena for graph intermediates (micrograd: python GC does this) */
/* ------------------------------------------------------------------ */

typedef struct Chunk {
    struct Chunk *next;
    size_t cap, off;
} Chunk;

static Chunk *g_arena;

static void *arena_alloc(size_t sz)
{
    sz = (sz + 15u) & ~(size_t)15u;
    Chunk *c = g_arena;
    if (!c || c->cap - c->off < sz) {
        size_t cap = sz > 65536 ? sz : 65536;
        Chunk *n = malloc(sizeof(Chunk) + cap);
        if (!n) { fprintf(stderr, "ig: out of memory\n"); exit(1); }
        n->next = g_arena;
        n->cap = cap;
        n->off = 0;
        g_arena = n;
        c = n;
    }
    void *p = (char *)(c + 1) + c->off;
    c->off += sz;
    return p;
}

void ig_reset(void)
{
    Chunk *c = g_arena;
    while (c) { Chunk *n = c->next; free(c); c = n; }
    g_arena = NULL;
}

static float qz(float x)
{
    if (!ig_quant_enabled) return x;
    if (ig_fx_mode == 2) return x; /* int32-accumulator emulation: no requant */
    return ig_fx_mode ? ig_fx_roundtrip(x) : ig_rt_act(x);
}

/* True iff quantizing x would clamp at the grid edge (vs. round normally).
 * In-range rounding has STE derivative 1; clamping has derivative 0. */
static int grid_clamped(float x, int fx)
{
    if (fx) return fabsf(x) > 127.0f * ig_get_fxstep();
    float d, s;
    ig_get_role_grid("act", &d, &s);
    return fabsf(x) > ig_max_abs(d, s);
}

static int qz_clamped(float x)
{
    if (!ig_quant_enabled || ig_fx_mode == 2) return 0;
    return grid_clamped(x, ig_fx_mode == 1);
}

static ig_Value *mk(void)
{
    ig_Value *v = arena_alloc(sizeof *v);
    memset(v, 0, sizeof *v);
    return v;
}

ig_Value *ig_value(float x)
{
    ig_Value *v = mk();
    v->data = qz(x);
    return v;
}

ig_Value *ig_value_persist(float x)
{
    ig_Value *v = calloc(1, sizeof *v);
    if (!v) { fprintf(stderr, "ig: out of memory\n"); exit(1); }
    v->data = ig_quant_enabled ? ig_rt_w(x) : x; /* params live on the w grid */
    return v;
}

void ig_value_free_persist(ig_Value *v) { free(v); }

static ig_Value *op(ig_Value **children, int n, float raw,
                    void (*bp)(ig_Value *))
{
    ig_Value *v = mk();
    v->data = qz(raw);
    v->clamped = qz_clamped(raw);
    v->backprop = bp;
    v->nchildren = n;
    v->children = arena_alloc(sizeof(ig_Value *) * (size_t)n);
    memcpy(v->children, children, sizeof(ig_Value *) * (size_t)n);
    return v;
}

/* ------------------------------------------------------------------ */
/* Backward: Kahn-free topological order via DFS, pass-counter marks   */
/* ------------------------------------------------------------------ */

static ig_Value **g_order;
static size_t g_n, g_cap;
static int g_pass;

static void push_order(ig_Value *v)
{
    if (g_n == g_cap) {
        g_cap = g_cap ? g_cap * 2 : 1024;
        g_order = realloc(g_order, g_cap * sizeof *g_order);
        if (!g_order) { fprintf(stderr, "ig: oom\n"); exit(1); }
    }
    g_order[g_n++] = v;
}

static void dfs(ig_Value *v)
{
    if (v->_seen == g_pass) return;
    v->_seen = g_pass;
    for (int i = 0; i < v->nchildren; i++) dfs(v->children[i]);
    push_order(v);
}

void ig_backward(ig_Value *loss)
{
    g_pass++;
    g_n = 0;
    dfs(loss);
    loss->grad += 1.0f;
    for (size_t i = g_n; i > 0; i--) {
        ig_Value *v = g_order[i - 1];
        if (v->backprop) v->backprop(v);
    }
}

/* ------------------------------------------------------------------ */
/* Ops                                                                  */
/* ------------------------------------------------------------------ */

/* Every backprop honors the clamp mask: d(quant)/dx = 0 outside range,
 * 1 inside (STE). See value.h note on ig_Value::clamped. */

static void bp_add(ig_Value *s)
{
    if (s->clamped) return;
    for (int i = 0; i < s->nchildren; i++)
        s->children[i]->grad += s->grad;
}

static void bp_sub(ig_Value *s)
{
    if (s->clamped) return;
    s->children[0]->grad += s->grad;
    s->children[1]->grad -= s->grad;
}

static void bp_mul(ig_Value *s)
{
    if (s->clamped) return;
    ig_Value *a = s->children[0], *b = s->children[1];
    a->grad += s->grad * b->data;
    b->grad += s->grad * a->data;
}

static void bp_tanh(ig_Value *s)
{
    if (s->clamped) return;
    float local = 1.0f - s->data * s->data;
    s->children[0]->grad += s->grad * local;
}

static void bp_relu(ig_Value *s)
{
    if (s->clamped) return;
    ig_Value *a = s->children[0];
    if (a->data > 0.0f) a->grad += s->grad;
}

static void bp_logsumexp(ig_Value *s)
{
    if (s->clamped) return;
    for (int i = 0; i < s->nchildren; i++)
        s->children[i]->grad +=
            s->grad * expf(s->children[i]->data - s->data);
}

/* Requantize once (end of an exact int32-style accumulation). STE: the
 * gradient passes 1:1 in range, 0 where the grid clamped. */
static void bp_requant(ig_Value *s)
{
    if (s->clamped) return;
    s->children[0]->grad += s->grad;
}

ig_Value *ig_requant(ig_Value *a, int grid)
{
    ig_Value *cs[1] = { a };
    int sav = ig_fx_mode;
    ig_fx_mode = 0;
    ig_Value *v = op(cs, 1, a->data, bp_requant); /* exact copy ... */
    ig_fx_mode = sav;
    if (ig_quant_enabled) { /* ... then exactly one quantization */
        int fx = grid == IG_GRID_FX;
        v->data = fx ? ig_fx_roundtrip(a->data) : ig_rt_act(a->data);
        v->clamped = grid_clamped(a->data, fx);
    } else {
        v->data = a->data;
        v->clamped = 0;
    }
    return v;
}

ig_Value *ig_add(ig_Value *a, ig_Value *b)
{
    ig_Value *cs[2] = { a, b };
    return op(cs, 2, a->data + b->data, bp_add);
}

ig_Value *ig_sub(ig_Value *a, ig_Value *b)
{
    ig_Value *cs[2] = { a, b };
    return op(cs, 2, a->data - b->data, bp_sub);
}

ig_Value *ig_mul(ig_Value *a, ig_Value *b)
{
    ig_Value *cs[2] = { a, b };
    return op(cs, 2, a->data * b->data, bp_mul);
}

ig_Value *ig_tanh(ig_Value *a)
{
    ig_Value *cs[1] = { a };
    return op(cs, 1, tanhf(a->data), bp_tanh);
}

ig_Value *ig_relu(ig_Value *a)
{
    ig_Value *cs[1] = { a };
    return op(cs, 1, a->data > 0.0f ? a->data : 0.0f, bp_relu);
}

ig_Value *ig_logsumexp(const ig_Value **xs, int n)
{
    float m = xs[0]->data;
    for (int i = 1; i < n; i++)
        if (xs[i]->data > m) m = xs[i]->data;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(xs[i]->data - m);
    return op((ig_Value **)xs, n, m + logf(sum), bp_logsumexp);
}
