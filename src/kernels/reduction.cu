// Parallel reduction: sum N floats into one float.
//
// Bandwidth-bound: reads N*4 bytes, does ~N adds. 4 B/FLOP → ceiling at
// 224 GB/s ÷ 4 = 56 GFLOPS on this card.
//
// Four versions on the Mark Harris progression:
//   v1  interleaved      naive — divergent within every warp
//   v2  sequential addr  same algorithm, contiguous active threads
//   v3  warp-shuffle     last 32 elements done in registers via __shfl_down_sync
//   v4  atomicAdd        v3 + grid-level combine on GPU (no CPU final sum)

#include <tinyforge/benchmark.hpp>
#include <tinyforge/cuda_check.hpp>
#include <tinyforge/reference.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

constexpr int kBlock = 256;

// ── v1: interleaved (naive). Active threads scattered through the warp.
__global__ void reduce_v1_interleaved(const float* __restrict__ in,
                                      float* __restrict__ partials,
                                      int n) {
    __shared__ float sdata[kBlock];
    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    // Tree: stride doubles. Active thread iff (tid % (2*s) == 0).
    // Within a warp, every other lane is idle → divergence at every step.
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// ── v2: sequential addressing. Same tree, active threads contiguous.
__global__ void reduce_v2_seq_addr(const float* __restrict__ in,
                                   float* __restrict__ partials,
                                   int n) {
    __shared__ float sdata[kBlock];
    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    // Stride halves. Active threads are 0..s-1 — contiguous, whole warps
    // either fully active or fully idle.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

// ── v3: warp-shuffle tail. Last 32 elements reduced in registers.
__global__ void reduce_v3_warpshuffle(const float* __restrict__ in,
                                      float* __restrict__ partials,
                                      int n) {
    __shared__ float sdata[kBlock];
    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    // Shared-memory phase: 256 → 64 elements. Stops when only one warp
    // would be active.
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-shuffle phase: 64 → 1, no shared memory, no __syncthreads.
    // Warps execute in lockstep so the implicit ordering is enough.
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, off);
        }
        if (tid == 0) partials[blockIdx.x] = val;
    }
}

// ── v4: atomicAdd into a single global scalar. No CPU final sum needed.
__global__ void reduce_v4_atomic(const float* __restrict__ in,
                                 float* __restrict__ out_scalar,
                                 int n) {
    __shared__ float sdata[kBlock];
    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFu, val, off);
        }
        // One atomic per BLOCK, not per thread. 512 atomics on Ampere FP32
        // atomicAdd is hardware-native, basically free at this scale.
        if (tid == 0) atomicAdd(out_scalar, val);
    }
}

int pick_grid(int n, int block, int num_sms) {
    const int blocks_to_cover_n = (n + block - 1) / block;
    const int saturate          = num_sms * 32;
    return std::min(blocks_to_cover_n, saturate);
}

double sum_cpu_double(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x;
    return s;
}

bool run_one_size(int n, int num_sms) {
    std::printf("\n── N = %d ─────────────────────────────────────────────────\n", n);

    std::vector<float> h_in(static_cast<std::size_t>(n));
    tinyforge::fill_random(h_in.data(), n);
    const double cpu_ref = sum_cpu_double(h_in);

    float* d_in = nullptr;
    const std::size_t bytes_in = static_cast<std::size_t>(n) * sizeof(float);
    TF_CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
    TF_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes_in, cudaMemcpyHostToDevice));

    const int block = kBlock;
    const int grid  = pick_grid(n, block, num_sms);

    float* d_partials = nullptr;
    float* d_scalar   = nullptr;
    TF_CUDA_CHECK(cudaMalloc(&d_partials, grid * sizeof(float)));
    TF_CUDA_CHECK(cudaMalloc(&d_scalar,   sizeof(float)));
    std::vector<float> h_partials(grid);

    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 5;
    cfg.timed_runs    = 30;
    cfg.flops_per_run = static_cast<double>(n);
    cfg.bytes_per_run = static_cast<double>(n) * sizeof(float);

    // Accumulation error in FP32 grows roughly sqrt(N)·ε per "magnitude". For
    // N=128M, ε~1.2e-7, expected error ~1.4e-3 of the typical sum magnitude.
    // Use generous absolute and relative tolerance.
    auto validate = [&](const char* name, double gpu_val) {
        const double err = std::abs(gpu_val - cpu_ref);
        const double tol = std::max(1e-2, 5e-3 * std::abs(cpu_ref));
        if (err > tol) {
            std::fprintf(stderr,
                         "  %-22s FAILED  gpu=%.4e  cpu=%.4e  abs_err=%.3e  tol=%.3e\n",
                         name, gpu_val, cpu_ref, err, tol);
            return false;
        }
        return true;
    };

    auto run_partial_variant = [&](const char* name, void (*kern)(const float*, float*, int)) {
        kern<<<grid, block>>>(d_in, d_partials, n);
        TF_CUDA_CHECK_LAST();
        TF_CUDA_CHECK(cudaMemcpy(h_partials.data(), d_partials,
                                 grid * sizeof(float), cudaMemcpyDeviceToHost));
        double gpu_sum = 0.0;
        for (float p : h_partials) gpu_sum += p;
        if (!validate(name, gpu_sum)) return;

        auto r = tinyforge::benchmark_kernel([&] {
            kern<<<grid, block>>>(d_in, d_partials, n);
        }, cfg);
        tinyforge::print_result(name, r);
    };

    run_partial_variant("v1 interleaved",  reduce_v1_interleaved);
    run_partial_variant("v2 seq_addr",     reduce_v2_seq_addr);
    run_partial_variant("v3 warpshuffle",  reduce_v3_warpshuffle);

    // v4: single-scalar output via atomicAdd. Must zero d_scalar before each
    // launch in the timed loop (otherwise atomicAdd accumulates across runs).
    {
        TF_CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
        reduce_v4_atomic<<<grid, block>>>(d_in, d_scalar, n);
        TF_CUDA_CHECK_LAST();
        float gpu_scalar = 0.0f;
        TF_CUDA_CHECK(cudaMemcpy(&gpu_scalar, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
        if (validate("v4 atomic", static_cast<double>(gpu_scalar))) {
            auto r = tinyforge::benchmark_kernel([&] {
                cudaMemsetAsync(d_scalar, 0, sizeof(float), 0);
                reduce_v4_atomic<<<grid, block>>>(d_in, d_scalar, n);
            }, cfg);
            tinyforge::print_result("v4 atomic", r);
        }
    }

    TF_CUDA_CHECK(cudaFree(d_in));
    TF_CUDA_CHECK(cudaFree(d_partials));
    TF_CUDA_CHECK(cudaFree(d_scalar));
    return true;
}

}  // namespace

int main() {
    int dev = 0;
    TF_CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    constexpr double kPeakBW_GBps = 224.0;
    std::printf("== reduction ==  device=%s  SMs=%d  arch=sm_%d%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    std::printf("theoretical ceiling: %.0f GB/s memory  →  %.1f GFLOPS at 4 B/FLOP\n",
                kPeakBW_GBps, kPeakBW_GBps / 4.0);

    const int sizes[] = {
        1 << 20,    //   1 M
        1 << 24,    //  16 M
        1 << 26,    //  64 M
        1 << 27,    // 128 M
    };

    int rc = 0;
    for (int n : sizes) {
        if (!run_one_size(n, prop.multiProcessorCount)) { rc = 1; break; }
    }
    return rc;
}
