//vector add with grid-stride loop.
//
//   c[i] = a[i] + b[i]                for i in [0, N)
//
// Bandwidth-bound: 3 floats touched per FLOP (read a, read b, write c).
// Theoretical FP32 add peak ~9.1 TFLOPS, but we are nowhere near it — the
// 224 GB/s memory ceiling caps us at 224e9 / 12 = 18.7 GFLOPS for this kernel.
// That is the actual ceiling worth measuring against.
//
// Compared to the SAXPY in test_setup.cu, this kernel uses a grid-stride loop
// so a single launch covers any N: we pick the grid size to saturate the SMs
// and each thread sweeps over many elements at a stride of (block * grid).

#include <tinyforge/benchmark.hpp>
#include <tinyforge/cuda_check.hpp>
#include <tinyforge/reference.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Grid sizing for a bandwidth-bound kernel: enough blocks to keep every SM
// busy with several waves of work, then let the grid-stride loop handle the
// rest. The exact multiplier doesn't matter much past ~16; we use 32.
int pick_grid(int n, int block, int num_sms) {
    const int blocks_to_cover_n = (n + block - 1) / block;
    const int saturate          = num_sms * 32;
    return std::min(blocks_to_cover_n, saturate);
}

bool run_one_size(int n, int num_sms) {
    // Host buffers
    std::vector<float> ha(static_cast<std::size_t>(n));
    std::vector<float> hb(static_cast<std::size_t>(n));
    std::vector<float> hc_ref(static_cast<std::size_t>(n));
    std::vector<float> hc_gpu(static_cast<std::size_t>(n));

    tinyforge::fill_random(ha.data(), n, 0xC0FFEEu);
    tinyforge::fill_random(hb.data(), n, 0xC0FFEEu ^ 0x5A5A5A5Au);

    for (int i = 0; i < n; ++i) hc_ref[i] = ha[i] + hb[i];

    // Device buffers
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    TF_CUDA_CHECK(cudaMalloc(&da, bytes));
    TF_CUDA_CHECK(cudaMalloc(&db, bytes));
    TF_CUDA_CHECK(cudaMalloc(&dc, bytes));
    TF_CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    TF_CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = pick_grid(n, block, num_sms);

    // Correctness
    // One launch + readback + compare BEFORE any timing.
    vector_add<<<grid, block>>>(da, db, dc, n);
    TF_CUDA_CHECK_LAST();
    TF_CUDA_CHECK(cudaMemcpy(hc_gpu.data(), dc, bytes, cudaMemcpyDeviceToHost));

    auto cmp = tinyforge::compare_matrices(hc_ref.data(), hc_gpu.data(), n);
    if (!cmp.passed) {
        std::fprintf(stderr,
                     "vector_add N=%d FAILED: n_mismatches=%d  first_bad=%d  "
                     "max_abs=%.3e  max_rel=%.3e\n",
                     n, cmp.n_mismatches, cmp.first_bad_idx,
                     cmp.max_abs_diff, cmp.max_rel_diff);
        cudaFree(da); cudaFree(db); cudaFree(dc);
        return false;
    }

    // Benchmark
    tinyforge::BenchConfig cfg;
    cfg.warmup_runs   = 5;
    cfg.timed_runs    = 30;
    cfg.flops_per_run = static_cast<double>(n);             // 1 add/elem
    cfg.bytes_per_run = 3.0 * static_cast<double>(n) * sizeof(float);

    auto r = tinyforge::benchmark_kernel([&] {
        vector_add<<<grid, block>>>(da, db, dc, n);
    }, cfg);

    char label[64];
    std::snprintf(label, sizeof(label), "vector_add N=%d (grid=%d)", n, grid);
    tinyforge::print_result(label, r);

    TF_CUDA_CHECK(cudaFree(da));
    TF_CUDA_CHECK(cudaFree(db));
    TF_CUDA_CHECK(cudaFree(dc));
    return true;
}

}  // namespace

int main() {
    int dev = 0;
    TF_CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    TF_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    // RTX 3050 spec: 224 GB/s memory bandwidth. Vector add is 12 bytes/FLOP,
    // so the bandwidth roof translates to ~18.7 GFLOPS — that is the ceiling
    // any number we print here should be compared against.
    constexpr double kPeakBW_GBps = 224.0;
    std::printf("== vector_add ==  device=%s  SMs=%d  arch=sm_%d%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    std::printf("theoretical ceiling: %.0f GB/s memory  →  %.1f GFLOPS at 12 B/FLOP\n\n",
                kPeakBW_GBps, kPeakBW_GBps / 12.0);

    const int sizes[] = {
        1 << 20,    //   1 M
        1 << 24,    //  16 M
        1 << 26,    //  64 M
        1 << 27,    // 128 M
    };

    int rc = 0;
    for (int n : sizes) {
        if (!run_one_size(n, prop.multiProcessorCount)) {
            rc = 1;
            break;
        }
    }
    return rc;
}
