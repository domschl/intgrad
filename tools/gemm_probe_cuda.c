/* S1a: what a CUDA tensor core does for M3-scale GEMMs (cuBLASLt, no
 * hand-written MMA), reported in the units intgrad plans with: TOPS and the
 * Bbytes/s it implies for a ~7M-param byte model (~50 MFLOP per training
 * byte: 6*N plus attention).
 *
 * Shapes are intgrad's M3 byte-transformer: M = tokens/step (batch*ctx),
 * K = d_model = 256, N = qkv/ffn/head widths, plus a large reference GEMM
 * for the machine ceiling.
 *
 * Path C is the one that matters for D1/D6: int8 operands, int32 accumulate.
 * Also measured because M5 will need them: the dequantize/requantize helper
 * (int32 -> fp32 -> log byte), which is per-element and often forgotten in
 * peak-TOPS planning.
 *
 * Build (this box, CUDA 13):
 *   nvcc -O3 -arch=sm_89 gemm_probe_cuda.c -lcublasLt -o gemm_probe_cuda
 */
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("cuda err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define CKL(x) do { cublasStatus_t s = (x); if (s) { printf("cublasLt err %d @%d\n", (int)s, __LINE__); return -1; } } while (0)

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

static cublasLtHandle_t lt;
static void *ws;
static size_t wssz = 64u << 20;

/* One GEMM: A[M,K] x B[K,N] (row-major), int8 in, int32 out. */
static int gemm_i8(const void *A, const void *B, void *C, int M, int K, int N,
                   double *ms_out)
{
    /* column-major view: compute C'[N,M] = B'[N,K] * A'[K,M] */
    cublasLtMatmulDesc_t op;
    CKL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t tN = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tN, sizeof tN));
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof tN));
    cublasLtMatrixLayout_t la, lb, lc;
    /* B row-major [K,N] == column-major [N,K] with ld = N */
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, N, K, N));
    /* A row-major [M,K] == column-major [K,M] with ld = K */
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, M, K));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, N, M, N));

    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wssz, sizeof wssz));
    cublasLtMatmulHeuristicResult_t h;
    int found = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, lb, la, lc, lc, pref, 1, &h, &found);
    if (hs || !found) { printf("        (no int8 algo for this shape)"); return -2; }

    int32_t alpha = 1, beta = 0;
    CKL(cublasLtMatmul(lt, op, &alpha, B, lb, A, la, &beta, C, lc, C, lc,
                       &h.algo, ws, wssz, 0));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    int reps = (int)(20e9 / (2.0 * M * K * N)); if (reps < 3) reps = 3;
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        cudaEventRecord(e0, (cudaStream_t)0);
        for (int k = 0; k < reps; k++)
            CKL(cublasLtMatmul(lt, op, &alpha, B, lb, A, la, &beta, C, lc, C, lc,
                               &h.algo, ws, wssz, 0));
        cudaEventRecord(e1, (cudaStream_t)0);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        double per = ms / reps; if (per < best) best = per;
    }
    *ms_out = best;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    return 0;
}

/* fp32 SGEMM for the non-quantized reference / oracle path. */
static int gemm_f32(const void *A, const void *B, void *C, int M, int K, int N,
                    double *ms_out)
{
    cublasLtMatmulDesc_t op;
    CKL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tN = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tN, sizeof tN));
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof tN));
    cublasLtMatrixLayout_t la, lb, lc;
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, N, K, N));
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, K, M, K));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wssz, sizeof wssz));
    cublasLtMatmulHeuristicResult_t h; int found = 0;
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, lb, la, lc, lc, pref, 1, &h, &found) || !found)
        { printf("        (no fp32 algo)"); return -2; }
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int reps = (int)(20e9 / (2.0 * M * K * N)); if (reps < 3) reps = 3;
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        cudaEventRecord(e0, (cudaStream_t)0);
        for (int k = 0; k < reps; k++)
            CKL(cublasLtMatmul(lt, op, &alpha, B, lb, A, la, &beta, C, lc, C, lc,
                               &h.algo, ws, wssz, 0));
        cudaEventRecord(e1, (cudaStream_t)0); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        double per = ms / reps; if (per < best) best = per;
    }
    *ms_out = best;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    return 0;
}

static void report(const char *tag, const char *nm, int M, int K, int N,
                   double ms, int bits)
{
    double ops = 2.0 * M * K * N / (ms / 1e3);
    double t = ops / 1e12;
    printf("%-8s %-9s %6d %5d %6d  %8.2f T  %8.3f ms  %9.0f kB/s@M3\n",
           tag, nm, M, K, N, t, ms, t * 1e12 / 50e6 * 1e-3);
    (void)bits;
}

int main(void)
{
    int dev = 0, major = 0, minor = 0;
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    printf("intgrad gemm_probe_cuda: device %d (sm_%d%d), %.1f GB total, %.2f GB free\n",
           dev, major, minor, (double)total_b / 1e9, (double)free_b / 1e9);
    CKL(cublasLtCreate(&lt));
    CK(cudaMalloc(&ws, wssz));
    printf("%-8s %-9s %6s %5s %6s  %11s  %10s  %s\n",
           "path", "layer", "M", "K", "N", "throughput", "ms/GEMM", "implied");

    for (int i = 0; i < NSH; i++) {
        int M = shapes[i].M, K = shapes[i].K, N = shapes[i].N;
        void *A, *B, *C;
        CK(cudaMalloc(&A, (size_t)M * K)); CK(cudaMalloc(&B, (size_t)K * N));
        CK(cudaMalloc(&C, (size_t)M * N * 4));
        CK(cudaMemset(A, 0x40, (size_t)M * K)); CK(cudaMemset(B, 0x40, (size_t)K * N));
        double ms = 0;
        if (gemm_i8(A, B, C, M, K, N, &ms) == 0) report("C/i8", shapes[i].name, M, K, N, ms, 8);
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    }
    for (int i = 0; i < NSH; i++) {
        int M = shapes[i].M, K = shapes[i].K, N = shapes[i].N;
        void *A, *B, *C;
        CK(cudaMalloc(&A, (size_t)M * K * 4)); CK(cudaMalloc(&B, (size_t)K * N * 4));
        CK(cudaMalloc(&C, (size_t)M * N * 4));
        CK(cudaMemset(A, 0, (size_t)M * K * 4)); CK(cudaMemset(B, 0, (size_t)K * N * 4));
        double ms = 0;
        if (gemm_f32(A, B, C, M, K, N, &ms) == 0) report("D/f32", shapes[i].name, M, K, N, ms, 32);
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    }
    return 0;
}
