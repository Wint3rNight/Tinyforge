#pragma once

#include <tinyforge/cuda_check.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace tinyforge {

struct BenchConfig {
    int    warmup_runs    = 3;
    int    timed_runs     = 20;
    double flops_per_run  = 0.0;
    double bytes_per_run  = 0.0;
};

struct BenchResult {
    double mean_ms = 0.0;
    double std_ms  = 0.0;
    double min_ms  = 0.0;
    double max_ms  = 0.0;
    double gflops  = 0.0;
    double gbps    = 0.0;
    int    n_runs  = 0;
};

template <typename Launch>
inline BenchResult benchmark_kernel(Launch&& launch, BenchConfig cfg = {}) {
    cudaEvent_t start{}, stop{};
    TF_CUDA_CHECK(cudaEventCreate(&start));
    TF_CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < cfg.warmup_runs; ++i) {
        launch();
    }
    TF_CUDA_CHECK(cudaDeviceSynchronize());
    TF_CUDA_CHECK_LAST();

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(cfg.timed_runs));

    for (int i = 0; i < cfg.timed_runs; ++i) {
        TF_CUDA_CHECK(cudaEventRecord(start));
        launch();
        TF_CUDA_CHECK(cudaEventRecord(stop));
        TF_CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        TF_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(static_cast<double>(ms));
    }

    TF_CUDA_CHECK(cudaEventDestroy(start));
    TF_CUDA_CHECK(cudaEventDestroy(stop));

    BenchResult r{};
    r.n_runs = cfg.timed_runs;

    double sum = 0.0;
    r.min_ms = samples.front();
    r.max_ms = samples.front();
    for (double s : samples) {
        sum     += s;
        r.min_ms = std::min(r.min_ms, s);
        r.max_ms = std::max(r.max_ms, s);
    }
    r.mean_ms = sum / static_cast<double>(samples.size());

    double sq = 0.0;
    for (double s : samples) {
        const double d = s - r.mean_ms;
        sq += d * d;
    }
    r.std_ms = std::sqrt(sq / static_cast<double>(samples.size()));

    const double seconds = r.mean_ms * 1e-3;
    if (cfg.flops_per_run > 0.0) {
        r.gflops = (cfg.flops_per_run / seconds) / 1e9;
    }
    if (cfg.bytes_per_run > 0.0) {
        r.gbps = (cfg.bytes_per_run / seconds) / 1e9;
    }
    return r;
}

inline void print_result(const std::string& name, const BenchResult& r) {
    std::printf("%-28s  mean=%8.3f ms  std=%7.3f  min=%7.3f  max=%7.3f  n=%-4d",
                name.c_str(),
                r.mean_ms, r.std_ms, r.min_ms, r.max_ms, r.n_runs);
    if (r.gflops > 0.0) std::printf("  %8.2f GFLOPS", r.gflops);
    if (r.gbps   > 0.0) std::printf("  %7.2f GB/s",   r.gbps);
    std::printf("\n");
}

}  // namespace tinyforge
