// Pure-FMA microbenchmark — measures the card's ACHIEVABLE sustained FP32 rate.
//
// Resolves STUDY_NOTES O7: the 9100 GFLOPS figure we had been quoting looks
// wrong for this laptop SKU (cuBLAS only reaches ~4.5 TFLOPS, and a vendor SGEMM
// normally lands at 80–90% of true peak). Everything here is register-resident
// with no memory traffic at all, so the only limit is FMA issue rate.
//
// Independent accumulator chains defeat the ~4-cycle FMA dependency latency —
// with one chain you measure latency, not throughput.

#include <tinyforge/benchmark.hpp>
#include <tinyforge/cuda_check.hpp>

#include <cstdio>

namespace {

constexpr int kChains = 8;   // independent dependency chains per thread

__global__ void fma_peak(float* __restrict__ out, int iters) {
    float a = 1.0000001f + threadIdx.x * 1e-7f;
    float b = 0.9999999f;
    float acc[kChains];
    #pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c] = static_cast<float>(c);

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int c = 0; c < kChains; ++c) acc[c] = fmaf(a, b, acc[c]);
    }

    float s = 0.0f;
    #pragma unroll
    for (int c = 0; c < kChains; ++c) s += acc[c];
    if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // never true
}

}  // namespace

int main() {
    TF_CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int block = 256;
    const int grid  = prop.multiProcessorCount * 8;
    // Each launch must run long enough — and the warmup long enough in total —
    // to pull the GPU out of its idle P-state. At iters=4096 (2.8 ms/launch) the
    // card stayed at 210 MHz / 10 W and reported ~776 GFLOPS, which measures the
    // idle clock, not the hardware. Same trap as STUDY_NOTES O1. ~65 ms/launch
    // with 20 warmups gives >1 s of sustained load before timing starts.
    const int iters = 100000;

    float* d_out = nullptr;
    TF_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * grid * block));

    // 2 FLOPs per fmaf, kChains per iteration, per thread.
    const double flops = 2.0 * static_cast<double>(iters) * kChains
                       * static_cast<double>(grid) * block;

    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 20;
    cfg.timed_runs    = 20;
    cfg.flops_per_run = flops;

    auto r = tinyforge::benchmark_kernel(
        [&] { fma_peak<<<grid, block>>>(d_out, iters); }, cfg);

    std::printf("== fp32_peak ==  %s  SMs=%d  grid=%d block=%d chains=%d iters=%d\n",
                prop.name, prop.multiProcessorCount, grid, block, kChains, iters);
    tinyforge::print_result("fma_peak", r);
    // cudaDeviceProp::clockRate was removed in CUDA 13; use the attribute API.
    int clock_khz = 0;
    TF_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const double ghz   = clock_khz / 1e6;
    const int    cores = prop.multiProcessorCount * 128;

    std::printf("\nAchievable sustained FP32: %.0f GFLOPS (%.2f TFLOPS)\n",
                r.gflops, r.gflops / 1000.0);
    std::printf("Theoretical at boost clock: %d cores x 2 flop x %.2f GHz = %.0f GFLOPS\n",
                cores, ghz, cores * 2.0 * ghz);

    TF_CUDA_CHECK(cudaFree(d_out));
    return 0;
}
