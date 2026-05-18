#include <tinyforge/benchmark.hpp>
#include <tinyforge/cuda_check.hpp>
#include <tinyforge/reference.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

__global__ void noop_kernel() {}

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    std::printf("== Tinyforge Phase 0 setup smoke test ==\n\n");

    {
        const float A[4] = {1, 2, 3, 4};
        const float B[4] = {5, 6, 7, 8};
        float C[4]       = {0};
        tinyforge::gemm_cpu(A, B, C, 2, 2, 2);
        std::printf("[1] gemm_cpu 2x2: got [%g %g %g %g]  expect [19 22 43 50]\n",
                    C[0], C[1], C[2], C[3]);
    }

    {
        std::vector<float> a(16), b(16);
        tinyforge::fill_random(a.data(), 16);
        std::copy(a.begin(), a.end(), b.begin());

        auto r = tinyforge::compare_matrices(a.data(), b.data(), 16);
        std::printf("[2a] compare(identical): passed=%d  max_abs=%.2e\n",
                    r.passed, r.max_abs_diff);

        b[7] += 0.5f;
        r = tinyforge::compare_matrices(a.data(), b.data(), 16);
        std::printf("[2b] compare(1 diff):    passed=%d  max_abs=%.2e  first_bad=%d\n",
                    r.passed, r.max_abs_diff, r.first_bad_idx);
    }

    {
        const int N = 1 << 20;
        std::vector<float> hx(N, 1.0f), hy(N, 2.0f);

        float *dx = nullptr, *dy = nullptr;
        TF_CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
        TF_CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
        TF_CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        TF_CUDA_CHECK(cudaMemcpy(dy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        const int block = 256;
        const int grid  = (N + block - 1) / block;

        auto r_noop = tinyforge::benchmark_kernel([&] {
            noop_kernel<<<1, 32>>>();
        }, {.warmup_runs = 5, .timed_runs = 100});
        tinyforge::print_result("noop_kernel", r_noop);

        tinyforge::BenchConfig cfg;
        cfg.warmup_runs   = 5;
        cfg.timed_runs    = 50;
        cfg.flops_per_run = 2.0 * static_cast<double>(N);
        cfg.bytes_per_run = 3.0 * static_cast<double>(N) * sizeof(float);

        auto r_saxpy = tinyforge::benchmark_kernel([&] {
            saxpy<<<grid, block>>>(2.5f, dx, dy, N);
        }, cfg);
        tinyforge::print_result("saxpy (N=1M)", r_saxpy);

        TF_CUDA_CHECK(cudaFree(dx));
        TF_CUDA_CHECK(cudaFree(dy));
    }

    std::printf("\nall checks executed.\n");
    return 0;
}
