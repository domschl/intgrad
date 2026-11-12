/* S1b: what a matrix engine can do for M3-scale GEMMs, on the machines at
 * hand -- measured, not guessed.
 *
 * Units that matter for intgrad: throughput, plus the Bbytes/s it implies
 * for the M3 byte-transformer (~50 MFLOP per training byte: 6*N with
 * N=7M params, plus attention).
 *
 * Paths measured
 *   A  cblas_sgemm fp32        -- Apple Accelerate / AMX (m4mini)
 *   B  NEON usdot int8          -- ARM dotprod, hand-written kernel (m4mini)
 *   B' AVX-VNNI vpdpbusd int8  -- x86 VNNI, hand-written kernel (x86 boxes)
 *   C  AVX2 maddubs int8       -- pre-VNNI x86 fallback (compile-gated)
 *
 * NOT available where it might be expected, and worth recording:
 *   - int8 in public C BLAS on macOS (neither cblas_gemm_s8 nor hgemm);
 *   - int8 AMX via the legacy BNNS layer API (float-only); the int8 NDArray
 *     API needs MPSGraph/Metal, which is a different backend decision.
 * So for int8 matrix math the choices are CUDA tensor cores (this box),
 * Metal MPSGraph (m4mini), or these vector-unit kernels (everywhere).
 *
 * Build
 *   macOS:  cc -O3 -mcpu=apple-m4 gemm_probe.c -framework Accelerate -o gemm_probe
 *   Linux:  cc -O3 -march=native gemm_probe.c -lm -o gemm_probe
 *           (BLAS path optional: -lblas -DIG_HAVE_BLAS)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define HAVE_BLAS 1
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* The M3 linear shapes: M = tokens per step (batch*ctx), K = d_model. */
static struct { const char *name; int M, K, N; } shapes[] = {
    { "qkv",      512, 256,  768 },
    { "qkv",     4096, 256,  768 },
    { "qkv",    16384, 256,  768 },
    { "ffn_up",  4096, 256, 1024 },
    { "ffn_up", 16384, 256, 1024 },
    { "attn_QK", 4096, 256,   64 },
    { "out_proj",4096, 256,  256 },
    { "wide",    4096,1024, 1024 },
    { "ceiling", 8192,2048, 2048 },
};
#define NSH ((int)(sizeof shapes / sizeof shapes[0]))

static volatile int64_t sink; /* keep the measured kernels alive */

static void fill_u8(uint8_t *p, size_t n, unsigned *s)
{
    for (size_t i = 0; i < n; i++) { *s = *s * 1103515245u + 12345u; p[i] = (uint8_t)(*s >> 23); }
}
static void fill_f32(float *p, size_t n, unsigned *s)
{
    for (size_t i = 0; i < n; i++) {
        *s = *s * 1103515245u + 12345u;
        p[i] = (float)((int)((*s >> 16) & 0xffff) - 32768) / 65536.0f;
    }
}

static void row(const char *path, const char *nm, int M, int K, int N,
                double secs, int ok)
{
    if (!ok) { printf("%-9s %-9s %6d %5d %6d  %10s\n", path, nm, M, K, N, "n/a"); return; }
    double tops = 2.0 * M * K * N / secs / 1e12;
    printf("%-9s %-9s %6d %5d %6d  %8.2f T  %8.3f ms  %8.0f kB/s@M3\n",
           path, nm, M, K, N, tops, secs * 1e3,
           tops * 1e12 / 50e6 * 1e-3); /* Bbytes/s for a 50 MFLOP/byte model */
}

/* ---- path A: fp32 BLAS ------------------------------------------------ */
static void bench_blas(void)
{
    printf("== A: fp32 GEMM ==\n");
#ifdef HAVE_BLAS
    unsigned s = 1u;
    for (int i = 0; i < NSH; i++) {
        int M = shapes[i].M, K = shapes[i].K, N = shapes[i].N;
        float *A = aligned_alloc(64, (size_t)M * K * 4);
        float *B = aligned_alloc(64, (size_t)K * N * 4);
        float *C = aligned_alloc(64, (size_t)M * N * 4);
        fill_f32(A, (size_t)M * K, &s); fill_f32(B, (size_t)K * N, &s);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, A, K, B, N, 0.0f, C, N);
        double best = 1e30;
        int reps = (int)(2e9 / (2.0 * M * K * N)); if (reps < 2) reps = 2;
        for (int r = 0; r < 3; r++) {
            double t0 = now();
            for (int k = 0; k < reps; k++)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                            1.0f, A, K, B, N, 0.0f, C, N);
            double dt = (now() - t0) / reps; if (dt < best) best = dt;
        }
        row("A/fp32", shapes[i].name, M, K, N, best, 1);
        free(A); free(B); free(C);
    }
#else
    for (int i = 0; i < NSH; i++) row("A/fp32", shapes[i].name, shapes[i].M, shapes[i].K, shapes[i].N, 0, 0);
#endif
}

/* ---- path B/B': asymmetric int8 GEMM (u8 activations x s8 weights -> s32)
 * This is exactly the primitive the matrix engines provide -- NEON usdot,
 * AVX-VNNI vpdpbusd, and (with zero-points folded into the bias) the int8
 * tensor-core path. Activations are stored u8 = q + zp, weights s8; a real
 * engine folds the zp correction into a precomputed per-channel bias, which
 * is why the asymmetric form is the one worth timing. B is stored [N][K]
 * (weights are stationary in training). */
static void bench_neon(void)
{
    printf("\n== B: int8 u8*s8->s32 (asym), NEON usdot, 1 thread ==\n");
#if defined(__ARM_NEON)
    unsigned s = 3u;
    for (int i = 0; i < NSH; i++) {
        int M = shapes[i].M, K = shapes[i].K, N = shapes[i].N;
        uint8_t *A = aligned_alloc(64, (size_t)M * K);
        int8_t *Bt = aligned_alloc(64, (size_t)K * N); /* [N][K] */
        int32_t *C = aligned_alloc(64, (size_t)M * N * 4);
        fill_u8(A, (size_t)M * K, &s);
        fill_u8((uint8_t *)Bt, (size_t)K * N, &s);
        double best = 1e30;
        int reps = (int)(1e9 / (2.0 * M * K * N)); if (reps < 1) reps = 1;
        for (int r = 0; r < 3; r++) {
            double t0 = now();
            for (int m = 0; m < M; m++) {
                const uint8_t *ar = A + (size_t)m * K;
                int32_t *cr = C + (size_t)m * N;
                for (int n = 0; n < N; n++) {
                    const int8_t *b = Bt + (size_t)n * K;
#if defined(__ARM_FEATURE_DOTPROD)
                    int32x4_t acc = vdupq_n_s32(0);
                    for (int k = 0; k + 16 <= K; k += 16)
                        acc = vusdotq_s32(acc, vld1q_s8(b + k), vld1q_u8(ar + k));
                    cr[n] = vaddvq_s32(acc);
#else
                    int32_t acc = 0;
                    for (int k = 0; k < K; k++) acc += (int)ar[k] * (int)b[k];
#endif
                }
            }
            /* sink: without a read of C the whole kernel is dead code and
             * the compiler deletes it (measured: 0.000 ms, a lie). */
            sink += C[(size_t)(r * N) + (unsigned)r];
            double dt = (now() - t0) / reps; if (dt < best) best = dt;
        }
        row("B/neon", shapes[i].name, M, K, N, best, 1);
        free(A); free(Bt); free(C);
    }
#else
    for (int i = 0; i < NSH; i++) row("B/neon", shapes[i].name, shapes[i].M, shapes[i].K, shapes[i].N, 0, 0);
#endif
}

static void bench_vnni(void)
{
    printf("\n== B': int8 u8*s8->s32 (asym), AVX-VNNI vpdpbusd, 1 thread ==\n");
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
    unsigned s = 5u;
    for (int i = 0; i < NSH; i++) {
        int M = shapes[i].M, K = shapes[i].K, N = shapes[i].N;
        uint8_t *A = aligned_alloc(64, (size_t)M * K);
        int8_t *Bt = aligned_alloc(64, (size_t)K * N);
        int32_t *C = aligned_alloc(64, (size_t)M * N * 4);
        fill_u8(A, (size_t)M * K, &s);
        fill_u8((uint8_t *)Bt, (size_t)K * N, &s);
        double best = 1e30;
        int reps = (int)(1e9 / (2.0 * M * K * N)); if (reps < 1) reps = 1;
        for (int r = 0; r < 3; r++) {
            double t0 = now();
            for (int m = 0; m < M; m++) {
                const uint8_t *ar = A + (size_t)m * K;
                int32_t *cr = C + (size_t)m * N;
                for (int n = 0; n < N; n++) {
                    const int8_t *b = Bt + (size_t)n * K;
                    __m256i acc = _mm256_setzero_si256();
                    for (int k = 0; k + 32 <= K; k += 32)
                        acc = _mm256_dpbusd_epi32(
                            acc, _mm256_loadu_si256((const __m256i *)(ar + k)),
                                 _mm256_loadu_si256((const __m256i *)(b + k)));
                    /* horizontal sum of 8 lanes */
                    __m128i lo = _mm256_castsi256_si128(acc),
                            hi = _mm256_extracti128_si256(acc, 1);
                    __m128i s2 = _mm_add_epi32(lo, hi);
                    s2 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0x4E));
                    s2 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
                    cr[n] = _mm_cvtsi128_si32(s2);
                }
            }
            sink += C[(size_t)(r * N) + (unsigned)r];
            double dt = (now() - t0) / reps; if (dt < best) best = dt;
        }
        row("B'/vnni", shapes[i].name, M, K, N, best, 1);
        free(A); free(Bt); free(C);
    }
#else
    for (int i = 0; i < NSH; i++) row("B'/vnni", shapes[i].name, shapes[i].M, shapes[i].K, shapes[i].N, 0, 0);
#endif
}

int main(void)
{
#if defined(__APPLE__)
    printf("intgrad gemm_probe: Apple silicon (Accelerate fp32 + NEON int8)\n");
#else
    printf("intgrad gemm_probe: x86 (AVX-VNNI int8)\n");
#endif
    printf("%-9s %-9s %6s %5s %6s  %11s  %10s  %s\n",
           "path", "layer", "M", "K", "N", "throughput", "ms/GEMM", "implied");
    bench_blas();
    bench_neon();
    bench_vnni();
    return 0;
}
