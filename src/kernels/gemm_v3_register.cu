

// Register-tiled GEMM — Phase 4. Each thread computes a TM×TN block of C held in
// registers, so one value fetched from shared memory feeds TM (or TN) FMAs
// instead of one. This is the step that actually attacks the LSU saturation
// measured in Phase 2 and left untouched by Phase 3.
//
//   Phase 3 inner loop:  2 shared loads : 1 FMA
//   Phase 4 inner loop: (TM+TN) loads   : TM*TN FMAs   = 16 : 64 at 8×8
//
// Also carries a float4-vectorized variant so the incremental effect of wide
// loads is measurable separately. Rationale + profiles: STUDY_NOTES §4.1.

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


// ── Phase 2 / Phase 3 kernels, duplicated for a same-run ladder ──────────────
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

template <int BS>
__global__ void gemm_shared(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * BS + tx, row = blockIdx.y * BS + ty;
    float acc = 0.0f;
    for (int t = 0; t < (K + BS - 1) / BS; ++t) {
        const int a_col = t * BS + tx, b_row = t * BS + ty;
        As[ty][tx] = (row < M   && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col   < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BS; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ── Phase 4: 2D register tiling ──────────────────────────────────────────────
// Block computes a BM×BN tile of C using BK-deep slabs. Thread count is
// (BM*BN)/(TM*TN) — at 128×128 tiles with 8×8 per thread that's 256 threads,
// each owning 64 accumulators in registers.
//
// As is stored TRANSPOSED ([BK][BM]) so the inner loop reads consecutive floats
// when gathering a thread's TM rows. Note the transposing store hits one bank
// 8 ways — a known, deliberate Phase 5 target (padding/swizzle); it is amortised
// over BK compute iterations so it costs a few percent, not a factor.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    constexpr int THREADS = (BM * BN) / (TM * TN);

    __shared__ float As[BK][BM];   // transposed
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN);   // which TM-row group
    const int threadCol = tid % (BN / TN);   // which TN-col group

    // Advance to this block's tiles.
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // Cooperative-load indices. Each thread loads several elements per slab.
    const int innerRowA = tid / BK, innerColA = tid % BK;
    const int innerRowB = tid / BN, innerColB = tid % BN;
    constexpr int strideA = THREADS / BK;
    constexpr int strideB = THREADS / BN;

    float acc[TM][TN] = {};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        #pragma unroll
        for (int o = 0; o < BM; o += strideA)
            As[innerColA][innerRowA + o] = A[(innerRowA + o) * K + innerColA];
        #pragma unroll
        for (int o = 0; o < BK; o += strideB)
            Bs[innerRowB + o][innerColB] = B[(innerRowB + o) * N + innerColB];

        __syncthreads();
        A += BK;
        B += BK * N;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // One shared read per output ROW and per output COLUMN...
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[k][threadRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k][threadCol * TN + j];
            // ...then TM*TN FMAs against them. This ratio is the whole phase.
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
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i][j];
}

// ── Phase 4b: same, with float4 (128-bit) loads/stores ───────────────────────
// One float4 instruction moves 4 floats, cutting load-instruction count 4× on
// the global→shared path and on the C write-back. Requires 16B alignment, hence
// the divisibility guard at the call site.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register_vec(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    constexpr int THREADS = (BM * BN) / (TM * TN);

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN);
    const int threadCol = tid % (BN / TN);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // float4 load indices: each thread moves 4 contiguous floats per step.
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
            // Transposing store: the float4 spans 4 rows of As.
            As[innerColA + 0][innerRowA + o] = v.x;
            As[innerColA + 1][innerRowA + o] = v.y;
            As[innerColA + 2][innerRowA + o] = v.z;
            As[innerColA + 3][innerRowA + o] = v.w;
        }
        #pragma unroll
        for (int o = 0; o < BK; o += strideB) {
            *reinterpret_cast<float4*>(&Bs[innerRowB + o][innerColB]) =
                *reinterpret_cast<const float4*>(
                    &B[(innerRowB + o) * N + innerColB]);
        }

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

// Tile config: 128×128 block tile, 8-deep slab, 8×8 per thread → 256 threads.
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int kRegThreads = (BM * BN) / (TM * TN);

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
        std::fprintf(stderr,
                     "N=%d FAILED (%s): mism=%d first_bad=%d max_abs=%.3e "
                     "max_rel=%.3e\n", n, who, cmp.n_mismatches,
                     cmp.first_bad_idx, cmp.max_abs_diff, cmp.max_rel_diff);
    return cmp.passed;
}

bool run_one_size(cublasHandle_t h, int n) {
    const int M = n, N = n, K = n;
    const std::size_t elemsA = static_cast<std::size_t>(M) * K;
    const std::size_t elemsB = static_cast<std::size_t>(K) * N;
    const std::size_t elemsC = static_cast<std::size_t>(M) * N;
    const std::size_t bytesC = elemsC * sizeof(float);

    std::vector<float> hA(elemsA), hB(elemsB), hRef(elemsC), hGot(elemsC);
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

    cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K);
    TF_CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    if (n <= kCpuOracleMaxN) {
        std::vector<float> hCpu(elemsC);
        tinyforge::gemm_cpu(hA.data(), hB.data(), hCpu.data(), M, N, K);
        auto cmp = tinyforge::compare_matrices(hCpu.data(), hRef.data(),
                                               static_cast<int>(elemsC));
        if (!cmp.passed) {
            std::fprintf(stderr, "N=%d FAILED (cuBLAS vs CPU)\n", n);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            return false;
        }
    }

    const dim3 nb(32, 8), ng((N + 31) / 32, (M + 7) / 8);
    const dim3 b16(16, 16), g16((N + 15) / 16, (M + 15) / 16);
    const dim3 rb(kRegThreads), rg(N / BN, M / BM);

    // The register kernels assume exact tiling (no bounds checks in the hot
    // loop) — skip rather than silently produce garbage.
    const bool reg_ok = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);

    auto launch_naive  = [&] { gemm_naive<<<ng, nb>>>(dA, dB, dC, M, N, K); };
    auto launch_s16    = [&] { gemm_shared<16><<<g16, b16>>>(dA, dB, dC, M, N, K); };
    auto launch_reg    = [&] { gemm_register<BM,BN,BK,TM,TN><<<rg, rb>>>(dA, dB, dC, M, N, K); };
    auto launch_regvec = [&] { gemm_register_vec<BM,BN,BK,TM,TN><<<rg, rb>>>(dA, dB, dC, M, N, K); };
    auto launch_cublas = [&] { cublas_sgemm_rowmajor(h, dA, dB, dC, M, N, K); };

    bool ok = true;
    ok &= verify("naive",      n, hRef, hGot, dC, launch_naive);
    ok &= verify("shared<16>", n, hRef, hGot, dC, launch_s16);
    if (reg_ok) {
        ok &= verify("register",     n, hRef, hGot, dC, launch_reg);
        ok &= verify("register+vec", n, hRef, hGot, dC, launch_regvec);
    }
    if (!ok) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return false; }

    tinyforge::wake_gpu(g_num_sms);   // after all host-side work, immediately before timing

    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 3;
    cfg.timed_runs    = 15;
    cfg.flops_per_run = 2.0 * static_cast<double>(M) * N * K;

    auto r_naive  = tinyforge::benchmark_kernel(launch_naive,  cfg);
    auto r_s16    = tinyforge::benchmark_kernel(launch_s16,    cfg);
    auto r_cublas = tinyforge::benchmark_kernel(launch_cublas, cfg);
    tinyforge::BenchResult r_reg{}, r_vec{};
    if (reg_ok) {
        r_reg = tinyforge::benchmark_kernel(launch_reg,    cfg);
        r_vec = tinyforge::benchmark_kernel(launch_regvec, cfg);
    }

    std::printf("N=%d\n", n);
    tinyforge::print_result("  naive        ", r_naive);
    tinyforge::print_result("  shared<16>   ", r_s16);
    if (reg_ok) {
        tinyforge::print_result("  register 8x8 ", r_reg);
        tinyforge::print_result("  register+vec ", r_vec);
    } else {
        std::printf("  register     : skipped (needs M,N %% %d == 0 and K %% %d == 0)\n",
                    BM, BK);
    }
    tinyforge::print_result("  cuBLAS       ", r_cublas);
    if (reg_ok) {
        const double best = r_vec.gflops > r_reg.gflops ? r_vec.gflops : r_reg.gflops;
        std::printf("  → best %.2f× naive,  %.1f%% of peak,  %.1f%% of cuBLAS\n\n",
                    best / r_naive.gflops, 100.0 * best / kPeakFP32_GFLOPS,
                    100.0 * best / r_cublas.gflops);
    } else {
        std::printf("\n");
    }

    TF_CUDA_CHECK(cudaFree(dA));
    TF_CUDA_CHECK(cudaFree(dB));
    TF_CUDA_CHECK(cudaFree(dC));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    TF_CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_num_sms = prop.multiProcessorCount;

    cublasHandle_t handle{};
    TF_CUBLAS_CHECK(cublasCreate(&handle));
    TF_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::printf("== gemm_v3_register ==  device=%s  SMs=%d  arch=sm_%d%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    std::printf("tile: BM=%d BN=%d BK=%d  thread tile: %dx%d  threads/block=%d\n",
                BM, BN, BK, TM, TN, kRegThreads);
    std::printf("FP32 compute roof: %.0f GFLOPS\n\n", kPeakFP32_GFLOPS);

    const int all_sizes[] = {256, 512, 1024, 2048, 4096};
    const int only = (argc > 1) ? std::atoi(argv[1]) : 0;

    int rc = 0;
    if (only > 0) {
        rc = run_one_size(handle, only) ? 0 : 1;
    } else {
        for (int n : all_sizes)
            if (!run_one_size(handle, n)) { rc = 1; break; }
    }

    TF_CUBLAS_CHECK(cublasDestroy(handle));
    return rc;
}
