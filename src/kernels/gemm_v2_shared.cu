// Shared-memory tiled GEMM — Phase 3. Each block cooperatively loads BS×BS tiles
// of A and B into shared memory once, then every thread reads its operands from
// there instead of re-fetching them from global memory. Attacks the LSU
// saturation measured in Phase 2 by cutting global-load instruction count ~BS×.
//
// Runs the naive baseline in the same process so the ladder is measured under
// identical clock/thermal conditions. Rationale, tile-size trade-offs, and the
// Nsight before/after live in STUDY_NOTES §3.1 — not repeated here.

#include <tinyforge/benchmark.hpp>
#include <tinyforge/cuda_check.hpp>
#include <tinyforge/gpu_wake.cuh>
#include <tinyforge/reference.hpp>

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

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

// SM count, filled in by main(); used to size the P-state wake-up burn.
int g_num_sms = 16;


// Phase 2 baseline, duplicated here for a same-run comparison. (Both drivers
// share a lot; extracting a common gemm harness is a Phase 4 cleanup.)
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// Square tiles: BM = BN = BK = BS, one output element per thread, so the block
// is exactly BS×BS threads. Each element loaded into shared memory is then read
// BS times by the block — that reuse factor is the whole point.
//
// The two __syncthreads() are both required: the first so nobody computes on a
// half-filled tile, the second so nobody overwrites a tile others still read.
template <int BS>
__global__ void gemm_shared(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    const int tx = threadIdx.x;                  // column within the tile
    const int ty = threadIdx.y;                  // row within the tile
    const int col = blockIdx.x * BS + tx;        // global column of C
    const int row = blockIdx.y * BS + ty;        // global row of C

    float acc = 0.0f;

    const int n_tiles = (K + BS - 1) / BS;
    for (int t = 0; t < n_tiles; ++t) {
        const int a_col = t * BS + tx;           // this thread's slot in the A tile
        const int b_row = t * BS + ty;           // ...and in the B tile

        // Cooperative load: one A element and one B element per thread. Both
        // reads are coalesced (consecutive tx → consecutive addresses).
        // Out-of-range slots load 0 so the dot product is unaffected.
        As[ty][tx] = (row < M   && a_col < K) ? A[row * K + a_col]   : 0.0f;
        Bs[ty][tx] = (b_row < K && col   < N) ? B[b_row * N + col]   : 0.0f;

        __syncthreads();

        // Compute on the tile. As[ty][k] is warp-uniform (broadcast) and
        // Bs[k][tx] walks consecutive banks — no bank conflicts either way.
        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// See STUDY_NOTES §5.7 for the transpose-swap trick (row-major via col-major).
void cublas_sgemm_rowmajor(cublasHandle_t h,
                           const float* dA, const float* dB, float* dC,
                           int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    TF_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K, &alpha,
                                dB, N, dA, K, &beta, dC, N));
}

// Achievable FP32 ceiling MEASURED with src/kernels/fp32_peak.cu (pure FMA,
// no memory traffic). The 9100 figure from the datasheet overstates this
// card by ~30%; run-to-run it lands at 6.7-6.9 TFLOPS.
constexpr double kPeakFP32_GFLOPS = 6884.0;
constexpr int    kCpuOracleMaxN   = 1024;

// Run `launch` once, copy the result back, and compare against `ref`.
template <typename Launch>
bool verify(const char* who, int n,
            const std::vector<float>& ref, std::vector<float>& got,
            float* dC, Launch&& launch) {
    TF_CUDA_CHECK(cudaMemset(dC, 0, got.size() * sizeof(float)));
    launch();
    TF_CUDA_CHECK_LAST();
    TF_CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

    auto cmp = tinyforge::compare_matrices(ref.data(), got.data(),
                                           static_cast<int>(got.size()));
    if (!cmp.passed) {
        std::fprintf(stderr,
                     "N=%d FAILED (%s): mism=%d first_bad=%d max_abs=%.3e "
                     "max_rel=%.3e\n",
                     n, who, cmp.n_mismatches, cmp.first_bad_idx,
                     cmp.max_abs_diff, cmp.max_rel_diff);
    }
    return cmp.passed;
}

bool run_one_size(cublasHandle_t h, int n) {
    const int M = n, N = n, K = n;
    const std::size_t elemsA = static_cast<std::size_t>(M) * K;
    const std::size_t elemsB = static_cast<std::size_t>(K) * N;
    const std::size_t elemsC = static_cast<std::size_t>(M) * N;
    const std::size_t bytesC = elemsC * sizeof(float);

    std::vector<float> hA(elemsA), hB(elemsB);
    std::vector<float> hRef(elemsC), hGot(elemsC);
    tinyforge::fill_random(hA.data(), static_cast<int>(elemsA), 0xC0FFEEu);
    tinyforge::fill_random(hB.data(), static_cast<int>(elemsB),
                           0xC0FFEEu ^ 0x5A5A5A5Au);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    TF_CUDA_CHECK(cudaMalloc(&dA, elemsA * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&dB, elemsB * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&dC, bytesC));
    TF_CUDA_CHECK(cudaMemcpy(dA, hA.data(), elemsA * sizeof(float),
                             cudaMemcpyHostToDevice));
    TF_CUDA_CHECK(cudaMemcpy(dB, hB.data(), elemsB * sizeof(float),
                             cudaMemcpyHostToDevice));

    // Reference = cuBLAS, itself spot-checked against the CPU oracle at small N.
    cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K);
    TF_CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    if (n <= kCpuOracleMaxN) {
        std::vector<float> hCpu(elemsC);
        tinyforge::gemm_cpu(hA.data(), hB.data(), hCpu.data(), M, N, K);
        auto cmp = tinyforge::compare_matrices(hCpu.data(), hRef.data(),
                                               static_cast<int>(elemsC));
        if (!cmp.passed) {
            std::fprintf(stderr, "N=%d FAILED (cuBLAS vs CPU): max_abs=%.3e\n",
                         n, cmp.max_abs_diff);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            return false;
        }
    }

    const dim3 nb(32, 8);
    const dim3 ng((N + nb.x - 1) / nb.x, (M + nb.y - 1) / nb.y);
    const dim3 b16(16, 16), g16((N + 15) / 16, (M + 15) / 16);
    const dim3 b32(32, 32), g32((N + 31) / 32, (M + 31) / 32);

    auto launch_naive = [&] { gemm_naive<<<ng, nb>>>(dA, dB, dC, M, N, K); };
    auto launch_s16   = [&] { gemm_shared<16><<<g16, b16>>>(dA, dB, dC, M, N, K); };
    auto launch_s32   = [&] { gemm_shared<32><<<g32, b32>>>(dA, dB, dC, M, N, K); };
    auto launch_cublas= [&] { cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K); };

    // Correctness before timing, for every kernel.
    bool ok = true;
    ok &= verify("naive",      n, hRef, hGot, dC, launch_naive);
    ok &= verify("shared<16>", n, hRef, hGot, dC, launch_s16);
    ok &= verify("shared<32>", n, hRef, hGot, dC, launch_s32);
    if (!ok) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return false;
    }

    // The CPU oracle above is a multi-second single-threaded loop; the GPU
    // idles down during it. Re-wake after all host work, before timing.
    tinyforge::wake_gpu(g_num_sms);

    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 3;
    cfg.timed_runs    = 15;
    cfg.flops_per_run = 2.0 * static_cast<double>(M) * N * K;

    auto r_naive  = tinyforge::benchmark_kernel(launch_naive,  cfg);
    auto r_s16    = tinyforge::benchmark_kernel(launch_s16,    cfg);
    auto r_s32    = tinyforge::benchmark_kernel(launch_s32,    cfg);
    auto r_cublas = tinyforge::benchmark_kernel(launch_cublas, cfg);

    std::printf("N=%d\n", n);
    tinyforge::print_result("  naive        ", r_naive);
    tinyforge::print_result("  shared<16>   ", r_s16);
    tinyforge::print_result("  shared<32>   ", r_s32);
    tinyforge::print_result("  cuBLAS       ", r_cublas);
    std::printf("  → best tiled %.2f× naive,  %.1f%% of peak,  %.1f%% of cuBLAS\n\n",
                (r_s32.gflops > r_s16.gflops ? r_s32.gflops : r_s16.gflops)
                    / r_naive.gflops,
                100.0 * (r_s32.gflops > r_s16.gflops ? r_s32.gflops : r_s16.gflops)
                    / kPeakFP32_GFLOPS,
                100.0 * (r_s32.gflops > r_s16.gflops ? r_s32.gflops : r_s16.gflops)
                    / r_cublas.gflops);

    TF_CUDA_CHECK(cudaFree(dA));
    TF_CUDA_CHECK(cudaFree(dB));
    TF_CUDA_CHECK(cudaFree(dC));
    return true;
}

}  // namespace

// Optional argv[1] = single matrix size, so `ncu ... ./gemm_v2_shared 1024`
// profiles a deterministic launch sequence instead of the whole sweep.
int main(int argc, char** argv) {
    TF_CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{}; 
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_num_sms = prop.multiProcessorCount;

    cublasHandle_t handle{};
    TF_CUBLAS_CHECK(cublasCreate(&handle));
    TF_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::printf("== gemm_v2_shared ==  device=%s  SMs=%d  arch=sm_%d%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    std::printf("FP32 compute roof: %.0f GFLOPS\n\n", kPeakFP32_GFLOPS);

    const int all_sizes[] = {256, 512, 1024, 2048, 4096};
    const int only = (argc > 1) ? std::atoi(argv[1]) : 0;

    int rc = 0;
    if (only > 0) {
        rc = run_one_size(handle, only) ? 0 : 1;
    } else {
        for (int n : all_sizes) {
            if (!run_one_size(handle, n)) { rc = 1; break; }
        }
    }

    TF_CUBLAS_CHECK(cublasDestroy(handle));
    return rc;
}
