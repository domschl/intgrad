/* Byte-level MLP proof-of-concept: predicts next UTF-8 byte from the last
 * CTX bytes (concatenated one-hot inputs). This is the M2 milestone vehicle:
 * it exercises the autograd + log-quant path end-to-end before the
 * transformer arrives in M3.
 */
#ifndef IG_NET_H
#define IG_NET_H

#include "ig/value.h"

#define IG_VOCAB 256

typedef struct {
    int ctx, nh;
    ig_Value **W1; /* [ctx * IG_VOCAB * nh] */
    ig_Value **b1; /* [nh] */
    ig_Value **W2; /* [nh * IG_VOCAB] */
    ig_Value **b2; /* [IG_VOCAB] */
} ig_MLP;

ig_MLP *ig_mlp_new(int ctx, int nh, unsigned seed);
void    ig_mlp_free(ig_MLP *m);

/* Forward on ctx bytes, returns malloc'd logits[IG_VOCAB] (graph nodes). */
ig_Value **ig_mlp_forward(ig_MLP *m, const uint8_t *in);

/* loss = CE(logits, target) built into the graph. */
ig_Value *ig_mlp_loss(ig_MLP *m, ig_Value **logits, uint8_t target);

void ig_mlp_zero_grad(ig_MLP *m);
void ig_mlp_sgd_update(ig_MLP *m, float lr); /* update + byte round-trip */

#endif /* IG_NET_H */
