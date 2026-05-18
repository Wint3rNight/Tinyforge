#include <tinyforge/reference.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace tinyforge {

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void fill_random(float* data, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

CompareResult compare_matrices(const float* expected,
                               const float* actual,
                               int          n,
                               double       atol,
                               double       rtol) {
    CompareResult r{};
    for (int i = 0; i < n; ++i) {
        const double e   = static_cast<double>(expected[i]);
        const double a   = static_cast<double>(actual[i]);
        const double abs_diff = std::abs(e - a);
        const double rel_diff = abs_diff / (std::abs(e) + 1e-12);

        r.max_abs_diff = std::max(r.max_abs_diff, abs_diff);
        r.max_rel_diff = std::max(r.max_rel_diff, rel_diff);

        const double tol = atol + rtol * std::abs(e);
        if (abs_diff > tol) {
            if (r.first_bad_idx < 0) r.first_bad_idx = i;
            r.n_mismatches += 1;
        }
    }
    r.passed = (r.n_mismatches == 0);
    return r;
}

}  // namespace tinyforge
