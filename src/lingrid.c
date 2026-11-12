/* E-E1 implementation: linear int8 fake-quant grids (see include/ig/lingrid.h). */
#include "ig/lingrid.h"
#include "ig/value.h"

#include <math.h>
#include <stdlib.h>

int ig_wgrid_mode = IG_WGRID_LOG;
int ig_agrid_mode = IG_AGRID_LOG;

#define IG_LIN_MAX 127 /* symmetric, no -128 asymmetry: keeps STE honest */

/* ---- weight scales ---------------------------------------------------- */
#define IG_MAX_GROUPS 8

typedef struct {
    ig_Value **base; /* the group's parameter nodes          */
    long n;          /* number of parameters in the group   */
    int mode;        /* 0 per-tensor, 1 rows, 2 columns     */
    int rowlen;      /* row length for mode 1               */
    long nchan;      /* number of scales used               */
} Group;

static Group gs[IG_MAX_GROUPS];
static int gs_ngroups;
static float *gs_scale;   /* one scale per channel, laid out group by group */
static long *gs_soff;     /* scale-array offset per group */
static long gs_scale_cap;

static long chan_index(const Group *g, long i)
{
    if (g->mode == 1) return i / g->rowlen; /* rows share a scale */
    if (g->mode == 2) return i % g->rowlen; /* columns share a scale */
    return 0;
}

void ig_lin_refresh_w(void *const *base, const long *count,
                      const long *chan_stride, int ngroups)
{
    if (ngroups > IG_MAX_GROUPS) ngroups = IG_MAX_GROUPS;
    long need = 0;
    for (int gi = 0; gi < ngroups; gi++) {
        gs[gi].base = (ig_Value **)base[gi];
        gs[gi].n = count[gi];
        if (chan_stride[gi] == 0) { gs[gi].mode = 0; }
        else if (chan_stride[gi] > 0) { gs[gi].mode = 1; gs[gi].rowlen = (int)chan_stride[gi]; }
        else { gs[gi].mode = 2; gs[gi].rowlen = (int)-chan_stride[gi]; }
        gs[gi].nchan = gs[gi].mode == 0 ? 1
                     : gs[gi].mode == 1 ? gs[gi].n / gs[gi].rowlen
                                        : gs[gi].rowlen;
        need += gs[gi].nchan + 1;
    }
    gs_ngroups = ngroups;
    if (need > gs_scale_cap) {
        free(gs_scale);
        gs_scale = calloc((size_t)need, sizeof(float));
        free(gs_soff);
        gs_soff = calloc((size_t)ngroups, sizeof(long));
        gs_scale_cap = need;
    }
    long so = 0;
    for (int gi = 0; gi < ngroups; gi++) {
        gs_soff[gi] = so;
        long nchan = gs[gi].nchan;
        float *mx = calloc((size_t)nchan, sizeof(float));
        const Group *g = &gs[gi];
        if (g->mode == 0) {
            for (long i = 0; i < g->n; i++) {
                float a = fabsf(g->base[i]->data);
                if (a > mx[0]) mx[0] = a;
            }
        } else {
            for (long i = 0; i < g->n; i++) {
                long c = chan_index(g, i);
                float a = fabsf(g->base[i]->data);
                if (a > mx[c]) mx[c] = a;
            }
        }
        for (long c = 0; c < nchan; c++) {
            /* power-of-two scale: a shift on the integer path */
            float s = mx[c] > 0.0f ? exp2f(ceilf(log2f(mx[c] / IG_LIN_MAX))) : 0.0f;
            gs_scale[so + c] = s;
        }
        so += nchan + 1;
        free(mx);
    }
}

/* Per-group lookup for the forward pass: ig_qw is called with the group
 * index and the parameter index inside that group. */
float ig_qw_gi(int gi, long idx)
{
    if (gi < 0 || gi >= gs_ngroups) return 0.0f;
    const Group *g = &gs[gi];
    float s = gs_scale[gs_soff[gi] + chan_index(g, idx)];
    if (s <= 0.0f) return 0.0f;
    float q = nearbyintf(g->base[idx]->data / s);
    if (q > IG_LIN_MAX) q = IG_LIN_MAX;
    if (q < -IG_LIN_MAX) q = -IG_LIN_MAX;
    return q * s;
}

/* Linear-grid step for the parameter at (group, index): the subgrid
 * statistic uses this instead of the log-grid spacing. 0 in log mode. */
float ig_wq_step(int gi, long idx)
{
    if (ig_wgrid_mode == IG_WGRID_LOG || gi >= gs_ngroups) return 0.0f;
    return gs_scale[gs_soff[gi] + chan_index(&gs[gi], idx)];
}

/* ---- activation scales (per tensor, EMA of observed absmax) ---------- */
#define IG_A_SLOTS 4
static float a_ema[IG_A_SLOTS];

void ig_aobserve(int slot, float amax)
{
    if (slot < 0 || slot >= IG_A_SLOTS) return;
    if (ig_agrid_mode != IG_AGRID_LIN) return;
    /* EMA of absmax, fast up / slow down: the scale must not clip a
     * tensor that grows, but must not stay loose after a spike either. */
    float cur = a_ema[slot];
    a_ema[slot] = amax > cur ? amax : cur * 0.99f + amax * 0.01f;
}

float ig_a_scale(int slot)
{
    float m = (slot >= 0 && slot < IG_A_SLOTS) ? a_ema[slot] : 0.0f;
    /* power-of-two step: the int path divides by a shift */
    if (m <= 0.0f) return 0.0f;
    return exp2f(ceilf(log2f(m / IG_LIN_MAX)));
}

float ig_qa(int slot, float x)
{
    if (ig_agrid_mode != IG_AGRID_LIN) return x;
    float s = ig_a_scale(slot);
    if (s <= 0.0f) return x;
    float q = nearbyintf(x / s);
    if (q > IG_LIN_MAX) q = IG_LIN_MAX;
    if (q < -IG_LIN_MAX) q = -IG_LIN_MAX;
    return q * s;
}
