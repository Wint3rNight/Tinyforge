// Phase 5 — bank-conflict elimination + tile-shape sweep.
//
// Same algorithm as the Phase 4 vectorized kernel; the only new idea is PAD:
// `As[BK][BM + PAD]`. With PAD=0 the transposing store has all writing threads
// landing on one bank (BM=128 is a multiple of the 32 banks); PAD=4 shifts each
// row by 4 banks and spreads them. PAD must stay a multiple of 4 so the float4
// reads out of As remain 16-byte aligned.
//
// Rationale + measurements: STUDY_NOTES §5.1.

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


template <int BM, int BN, int BK, int TM, int TN, int PAD>
__global__ void gemm_tuned(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
    constexpr int THREADS = (BM * BN) / (TM * TN);
    // Each thread must move exactly one float4 per pass on both tiles.
    static_assert((BM * BK / 4) % THREADS == 0, "A tile / THREADS mismatch");
    static_assert((BK * BN / 4) % THREADS == 0, "B tile / THREADS mismatch");
    static_assert((BM + PAD) % 4 == 0, "PAD must keep As rows 16B aligned");

    __shared__ float As[BK][BM + PAD];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int innerRowA = tid / (BK / 4), innerColA = (tid % (BK / 4)) * 4;
    const int innerRowB = tid / (BN / 4), innerColB = (tid % (BN / 4)) * 4;
    constexpr int strideA = THREADS / (BK / 4);
    constexpr int strideB = THREADS / (BN / 4);

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        #pragma unroll
        for (int o = 0; o < BM; o += strideA) {
            const float4 v = *reinterpret_cast<const float4*>(
                &A[(innerRowA + o) * K + innerColA]);
            As[innerColA + 0][innerRowA + o] = v.x;
            As[innerColA + 1][innerRowA + o] = v.y;
            As[innerColA + 2][innerRowA + o] = v.z;
            As[innerColA + 3][innerRowA + o] = v.w;
        }
        #pragma unroll
        for (int o = 0; o < BK; o += strideB)
            *reinterpret_cast<float4*>(&Bs[innerRowB + o][innerColB]) =
                *reinterpret_cast<const float4*>(
                    &B[(innerRowB + o) * N + innerColB]);

        __syncthreads();
        A += BK;
        B += BK * N;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<const float4*>(&As[k][threadRow * TM + i]);
            #pragma unroll
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + j]);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
            *reinterpret_cast<float4*>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j]) =
                *reinterpret_cast<const float4*>(&acc[i][j]);
}

void cublas_sgemm_rowmajor(cublasHandle_t h, const float* dA, const float* dB,
                           float* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    TF_CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                dB, N, dA, K, &beta, dC, N));
}

// Achievable FP32 ceiling MEASURED with src/kernels/fp32_peak.cu (pure FMA,
// no memory traffic). The 9100 figure from the datasheet overstates this
// card by ~30%; run-to-run it lands at 6.7-6.9 TFLOPS.
constexpr double kPeakFP32_GFLOPS = 6884.0;
constexpr int    kCpuOracleMaxN   = 1024;

template <typename Launch>
bool verify(const char* who, int n, const std::vector<float>& ref,
            std::vector<float>& got, float* dC, Launch&& launch) {
    TF_CUDA_CHECK(cudaMemset(dC, 0, got.size() * sizeof(float)));
    launch();
    TF_CUDA_CHECK_LAST();
    TF_CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    auto cmp = tinyforge::compare_matrices(ref.data(), got.data(),
                                           static_cast<int>(got.size()));
    if (!cmp.passed)
        std::fprintf(stderr, "N=%d FAILED (%s): mism=%d max_abs=%.3e\n",
                     n, who, cmp.n_mismatches, cmp.max_abs_diff);
    return cmp.passed;
}

bool run_one_size(cublasHandle_t h, int n) {
    const int M = n, N = n, K = n;
    const std::size_t eA = static_cast<std::size_t>(M) * K;
    const std::size_t eB = static_cast<std::size_t>(K) * N;
    const std::size_t eC = static_cast<std::size_t>(M) * N;

    std::vector<float> hA(eA), hB(eB), hRef(eC), hGot(eC);
    tinyforge::fill_random(hA.data(), static_cast<int>(eA), 0xC0FFEEu);
    tinyforge::fill_random(hB.data(), static_cast<int>(eB), 0xC0FFEEu ^ 0x5A5A5A5Au);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    TF_CUDA_CHECK(cudaMalloc(&dA, eA * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&dB, eB * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&dC, eC * sizeof(float)));
    TF_CUDA_CHECK(cudaMemcpy(dA, hA.data(), eA * sizeof(float), cudaMemcpyHostToDevice));
    TF_CUDA_CHECK(cudaMemcpy(dB, hB.data(), eB * sizeof(float), cudaMemcpyHostToDevice));

    cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K);
    TF_CUDA_CHECK(cudaMemcpy(hRef.data(), dC, eC * sizeof(float), cudaMemcpyDeviceToHost));
    if (n <= kCpuOracleMaxN) {
        std::vector<float> hCpu(eC);
        tinyforge::gemm_cpu(hA.data(), hB.data(), hCpu.data(), M, N, K);
        auto c = tinyforge::compare_matrices(hCpu.data(), hRef.data(), static_cast<int>(eC));
        if (!c.passed) {
            std::fprintf(stderr, "N=%d FAILED (cuBLAS vs CPU): max_abs=%.3e\n",
                         n, c.max_abs_diff);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            return false;
        }
    }

    // The CPU oracle above is a single-threaded N³ loop taking seconds, during
    // which the GPU falls back to its idle P-state. Re-wake AFTER all host work
    // and immediately before timing, or the first size measured reports idle
    // clocks (N=1024 read 8× low before this was added).
    tinyforge::wake_gpu(g_num_sms);

    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 3;
    cfg.timed_runs    = 15;
    cfg.flops_per_run = 2.0 * static_cast<double>(M) * N * K;

    std::printf("N=%d\n", n);
    bool ok = true;
    double best = 0.0; const char* best_name = "-";

    auto try_cfg = [&](const char* name, auto&& launch) {
        if (!verify(name, n, hRef, hGot, dC, launch)) { ok = false; return; }
        auto r = tinyforge::benchmark_kernel(launch, cfg);
        char lbl[48]; std::snprintf(lbl, sizeof(lbl), "  %-22s", name);
        tinyforge::print_result(lbl, r);
        if (r.gflops > best) { best = r.gflops; best_name = name; }
    };

    const dim3 b256(256);
    const dim3 g128(N / 128, M / 128);
    const dim3 g64(N / 64, M / 64);

    if (M % 128 == 0 && N % 128 == 0 && K % 16 == 0) {
        try_cfg("128x128x8  pad0",  [&]{ gemm_tuned<128,128,8,8,8,0><<<g128,b256>>>(dA,dB,dC,M,N,K); });
        try_cfg("128x128x8  pad4",  [&]{ gemm_tuned<128,128,8,8,8,4><<<g128,b256>>>(dA,dB,dC,M,N,K); });
        try_cfg("128x128x16 pad4",  [&]{ gemm_tuned<128,128,16,8,8,4><<<g128,b256>>>(dA,dB,dC,M,N,K); });
    }
    if (M % 64 == 0 && N % 64 == 0 && K % 16 == 0) {
        try_cfg("64x64x16   pad4",  [&]{ gemm_tuned<64,64,16,4,4,4><<<g64,b256>>>(dA,dB,dC,M,N,K); });
    }

    auto r_cublas = tinyforge::benchmark_kernel(
        [&]{ cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K); }, cfg);
    tinyforge::print_result("  cuBLAS (strict FP32)  ", r_cublas);

    // Same cuBLAS call with its tensor-core path unlocked. NOT used as the
    // correctness oracle — TF32's ~10-bit mantissa won't hold our tolerance —
    // but reported so the comparison above isn't read as "beats cuBLAS".
    TF_CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    auto r_tf32 = tinyforge::benchmark_kernel(
        [&]{ cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K); }, cfg);
    TF_CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
    tinyforge::print_result("  cuBLAS (TF32 tensor)  ", r_tf32);
    std::printf("  → best: %s  %.0f GFLOPS   %.1f%% of peak   "
                "%.1f%% of cuBLAS-FP32   %.1f%% of cuBLAS-TF32\n\n",
                best_name, best, 100.0 * best / kPeakFP32_GFLOPS,
                100.0 * best / r_cublas.gflops, 100.0 * best / r_tf32.gflops);

    TF_CUDA_CHECK(cudaFree(dA)); TF_CUDA_CHECK(cudaFree(dB)); TF_CUDA_CHECK(cudaFree(dC));
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    TF_CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_num_sms = prop.multiProcessorCount;
    cublasHandle_t h{};
    TF_CUBLAS_CHECK(cublasCreate(&h));
    TF_CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));

    std::printf("== gemm_v4_tuned ==  %s  SMs=%d\n", prop.name, prop.multiProcessorCount);
    std::printf("measured FP32 peak: %.0f GFLOPS (src/kernels/fp32_peak.cu)\n",
                kPeakFP32_GFLOPS);
    std::printf("waking GPU out of idle P-state...\n");
    tinyforge::wake_gpu(prop.multiProcessorCount);
    std::printf("done.\n\n");

    const int all[] = {1024, 2048, 4096};
    const int only = (argc > 1) ? std::atoi(argv[1]) : 0;
    int rc = 0;
    if (only > 0) rc = run_one_size(h, only) ? 0 : 1;
    else for (int n : all) if (!run_one_size(h, n)) { rc = 1; break; }

    TF_CUBLAS_CHECK(cublasDestroy(h));
    return rc;
}
