/* Gradient checks with quantization disabled (exact float mode). */
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "ig/value.h"

int main(void)
{
    ig_quant_enabled = 0;

    /* f(a,b) = a*b + tanh(a) - b*b at a=0.7, b=-1.3 */
    float a = 0.7f, b = -1.3f;

    ig_Value *va = ig_value(a), *vb = ig_value(b);
    ig_Value *r = ig_sub(ig_add(ig_mul(va, vb), ig_tanh(va)), ig_mul(vb, vb));
    ig_backward(r);
    float ga = va->grad, gb = vb->grad;
    assert(fabsf(ga - (b + (1.0f - tanhf(a) * tanhf(a)))) < 1e-3f);
    assert(fabsf(gb - a) > 0); /* df/da? no: df/db = a - 2b */
    assert(fabsf(gb - (a - 2.0f * b)) < 1e-3f);
    ig_reset();

    /* finite differences */
    float h = 1e-3f;
#define FRED(x_, y_) ( (x_)*(y_) + tanhf(x_) - (y_)*(y_) )
    float dfa = (FRED(a + h, b) - FRED(a - h, b)) / (2 * h);
    float dfb = (FRED(a, b + h) - FRED(a, b - h)) / (2 * h);
    assert(fabsf(dfa - ga) < 1e-2f);
    assert(fabsf(dfb - gb) < 1e-2f);

    /* shared node accumulates gradients: (a used three times) */
    va = ig_value(2.0f);
    ig_Value *q = ig_add(ig_add(va, va), va); /* dq/da = 3 */
    ig_backward(q);
    assert(fabsf(va->grad - 3.0f) < 1e-5f);
    ig_reset();

    /* logsumexp gradients form a softmax distribution */
    const ig_Value *xs[4];
    ig_Value *xv[4];
    float xv_init[4] = { 1.0f, 2.0f, 0.5f, -1.0f };
    float tot = 0;
    for (int i = 0; i < 4; i++) xv[i] = ig_value(xv_init[i]), xs[i] = xv[i];
    ig_Value *lse = ig_logsumexp(xs, 4);
    ig_backward(lse);
    float sumexp = 0;
    for (int i = 0; i < 4; i++) sumexp += expf(xv_init[i]);
    for (int i = 0; i < 4; i++) {
        float expect = expf(xv_init[i]) / sumexp;
        assert(fabsf(xv[i]->grad - expect) < 1e-4f);
        tot += xv[i]->grad;
    }
    assert(fabsf(tot - 1.0f) < 1e-4f);
    ig_reset();

    puts("autograd: all tests passed");
    return 0;
}
