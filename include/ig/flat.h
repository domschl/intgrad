/* Flat-array access to the C oracle, for parity tests from Python (ctypes,
 * py/tests/test_parity.py). Parameters are laid out in canonical group
 * order (W1, b1, W2, b2), each tensor row-major, ig_mlp_nparams() values. */
#ifndef IG_FLAT_H
#define IG_FLAT_H

#include <stdint.h>

#include "ig/net.h"

/* data/wq/grad may be NULL to skip that field. */
void ig_mlp_export(const ig_MLP *m, float *data, int16_t *wq, float *grad);
void ig_mlp_import(ig_MLP *m, const float *data, const int16_t *wq,
                   const float *grad);

/* Lion momentum: returns its length (0 = no state); copies into out if
 * non-NULL. set_state allocates/overwrites it. */
long ig_lion_get_state(float *out);
void ig_lion_set_state(const float *mom, long n);

/* Stochastic-rounding RNG (xorshift64, u = (s >> 40) / 2^24 per draw). */
uint64_t ig_sr_get_state(void);
void     ig_sr_set_state(uint64_t s);

/* forward + CE + backward on one window (grads ACCUMULATE); returns CE in
 * nats; frees the graph. */
double ig_mlp_window_backward(ig_MLP *m, const uint8_t *window);
/* forward only: quantized logits of the byte after `ctx` input bytes. */
void   ig_mlp_logits(ig_MLP *m, const uint8_t *in, float *out);

/* global switches (also settable from Python) */
void ig_set_modes(int quant, int pure_int, int w2, int w2_sr, int sr, float wd);

#endif /* IG_FLAT_H */
