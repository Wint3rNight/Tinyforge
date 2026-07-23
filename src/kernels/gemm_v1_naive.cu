// Naive GEMM baseline — one thread per output element of C = A·B
// (A: M×K, B: K×N, C: M×N, all row-major FP32). Phase 2. Deliberately simple;
// its job is to set up the correctness→benchmark→profile pipeline and give the
// next phases a number to beat. cuBLAS is wired in as baseline + oracle only.
//
// Design rationale, roofline, and the Nsight (LSU-bound) finding live in
// STUDY_NOTES §2.1 and docs/phase2-baseline.md — not repeated here.

#include <tinyforge/benchmark.hpp>
#include <tinyforge/cuda_check.hpp>
#include <tinyforge/reference.hpp>

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// cuBLAS status check, local to this file (keeps cuda_check.hpp cublas-free).
#define TF_CUBLAS_CHECK(expr)                                                   \
    do {                                                                        \
        cublasStatus_t _tf_s = (expr);                                          \
        if (_tf_s != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error at %s:%d in `%s`: %s\n",         \
                         __FILE__, __LINE__, #expr,                             \
                         cublasGetStatusString(_tf_s));                         \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

namespace {

// threadIdx.x → columns (n) is the one decision that matters: with a 32-wide
// block a warp spans 32 consecutive columns, so B/C loads coalesce and A
// broadcasts. Map it to rows and it scatters ~10× slower. (STUDY_NOTES §2.1.)
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // n — fastest-varying
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // m
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// cuBLAS is column-major; our buffers are row-major. A row-major r×c buffer is
// the column-major storage of its transpose, so we ask for Cᵀ = Bᵀ·Aᵀ (feed B
// first, A second, swap m/n) and it lands in our row-major C. (STUDY_NOTES §5.7.)
void cublas_sgemm_rowmajor(cublasHandle_t h,
                           const float* dA, const float* dB, float* dC,
                           int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    TF_CUBLAS_CHECK(cublasSgemm(h,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,      // cuBLAS m,n,k  (= N,M,K here)
                                &alpha,
                                dB, N,        // "A_cublas" = our B, ld = N
                                dA, K,        // "B_cublas" = our A, ld = K
                                &beta,
                                dC, N));      // "C_cublas" = our C, ld = N
}

constexpr double kPeakFP32_GFLOPS = 9100.0;  // RTX 3050 FP32 compute roof
constexpr int    kCpuOracleMaxN   = 1024;    // gemm_cpu above this is too slow

bool run_one_size(cublasHandle_t h, int n) {
    const int M = n, N = n, K = n;           // square for Phase 2
    const std::size_t elemsA = static_cast<std::size_t>(M) * K;
    const std::size_t elemsB = static_cast<std::size_t>(K) * N;
    const std::size_t elemsC = static_cast<std::size_t>(M) * N;

    // Host buffers
    std::vector<float> hA(elemsA), hB(elemsB);
    std::vector<float> hC_naive(elemsC), hC_cublas(elemsC);
    tinyforge::fill_random(hA.data(), static_cast<int>(elemsA), 0xC0FFEEu);
    tinyforge::fill_random(hB.data(), static_cast<int>(elemsB),
                           0xC0FFEEu ^ 0x5A5A5A5Au);

    // Device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    TF_CUDA_CHECK(cudaMalloc(&dA, elemsA * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&dB, elemsB * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&dC, elemsC * sizeof(float)));
    TF_CUDA_CHECK(cudaMemcpy(dA, hA.data(), elemsA * sizeof(float),
                             cudaMemcpyHostToDevice));
    TF_CUDA_CHECK(cudaMemcpy(dB, hB.data(), elemsB * sizeof(float),
                             cudaMemcpyHostToDevice));

    const dim3 block(32, 8);                 // 32 wide → warp = 32 cols of one row
    const dim3 grid((N + block.x - 1) / block.x,
                    (M + block.y - 1) / block.y);

    // ── Correctness BEFORE timing (the golden rule) ────────────────────────
    // 1) naive vs cuBLAS at every size — cheap, GPU-side, our large-N oracle.
    gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    TF_CUDA_CHECK_LAST();
    TF_CUDA_CHECK(cudaMemcpy(hC_naive.data(), dC, elemsC * sizeof(float),
                             cudaMemcpyDeviceToHost));

    cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K);
    TF_CUDA_CHECK(cudaMemcpy(hC_cublas.data(), dC, elemsC * sizeof(float),
                             cudaMemcpyDeviceToHost));

    auto cmp_cublas = tinyforge::compare_matrices(
        hC_cublas.data(), hC_naive.data(), static_cast<int>(elemsC));
    if (!cmp_cublas.passed) {
        std::fprintf(stderr,
                     "N=%d FAILED (naive vs cuBLAS): mism=%d first_bad=%d "
                     "max_abs=%.3e max_rel=%.3e\n",
                     n, cmp_cublas.n_mismatches, cmp_cublas.first_bad_idx,
                     cmp_cublas.max_abs_diff, cmp_cublas.max_rel_diff);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return false;
    }

    // 2) at small sizes, also check against the independent CPU triple-loop.
    if (n <= kCpuOracleMaxN) {
        std::vector<float> hC_ref(elemsC);
        tinyforge::gemm_cpu(hA.data(), hB.data(), hC_ref.data(), M, N, K);
        auto cmp_cpu = tinyforge::compare_matrices(
            hC_ref.data(), hC_naive.data(), static_cast<int>(elemsC));
        if (!cmp_cpu.passed) {
            std::fprintf(stderr,
                         "N=%d FAILED (naive vs CPU): mism=%d first_bad=%d "
                         "max_abs=%.3e max_rel=%.3e\n",
                         n, cmp_cpu.n_mismatches, cmp_cpu.first_bad_idx,
                         cmp_cpu.max_abs_diff, cmp_cpu.max_rel_diff);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            return false;
        }
    }

    // Benchmark. GFLOPS is the metric for GEMM (bytes_per_run left 0): a naive
    // kernel's traffic is mostly cache-served, so "achieved GB/s" is meaningless.
    const double flops = 2.0 * static_cast<double>(M) * N * K;

    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 3;
    cfg.timed_runs    = 15;
    cfg.flops_per_run = flops;

    auto r_naive = tinyforge::benchmark_kernel([&] {
        gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }, cfg);

    auto r_cublas = tinyforge::benchmark_kernel([&] {
        cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K);
    }, cfg);

    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "gemm_naive  N=%d", n);
    tinyforge::print_result(lbl, r_naive);
    std::snprintf(lbl, sizeof(lbl), "cuBLAS      N=%d", n);
    tinyforge::print_result(lbl, r_cublas);
    std::printf("    → naive %.1f%% of FP32 peak,  %.1f%% of cuBLAS%s\n\n",
                100.0 * r_naive.gflops / kPeakFP32_GFLOPS,
                r_cublas.gflops > 0.0 ? 100.0 * r_naive.gflops / r_cublas.gflops
                                      : 0.0,
                n <= kCpuOracleMaxN ? "   [CPU-checked]" : "   [cuBLAS-checked]");

    TF_CUDA_CHECK(cudaFree(dA));
    TF_CUDA_CHECK(cudaFree(dB));
    TF_CUDA_CHECK(cudaFree(dC));
    return true;
}

}  // namespace

int main() {
    int dev = 0;
    TF_CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    cublasHandle_t handle{};
    TF_CUBLAS_CHECK(cublasCreate(&handle));
    // Force true FP32 (no TF32 tensor-core substitution) so cuBLAS is an honest
    // correctness oracle for our FP32 kernel — a 1e-3 tolerance won't survive
    // TF32's ~10-bit mantissa.
    TF_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::printf("== gemm_v1_naive ==  device=%s  SMs=%d  arch=sm_%d%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    std::printf("FP32 compute roof: %.0f GFLOPS   (naive AI=0.25 → ~56 GFLOPS "
                "memory roof before caching)\n\n", kPeakFP32_GFLOPS);

    const int sizes[] = {256, 512, 1024, 2048, 4096};

    int rc = 0;
    for (int n : sizes) {
        if (!run_one_size(handle, n)) { rc = 1; break; }
    }

    TF_CUBLAS_CHECK(cublasDestroy(handle));
    return rc;
}
