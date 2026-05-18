#pragma once

namespace tinyforge {

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

void fill_random(float* data, int n, unsigned seed = 0xC0FFEEu);

struct CompareResult {
    bool   passed        = true;
    double max_abs_diff  = 0.0;
    double max_rel_diff  = 0.0;
    int    n_mismatches  = 0;
    int    first_bad_idx = -1;
};

CompareResult compare_matrices(const float* expected,
                               const float* actual,
                               int          n,
                               double       atol = 1e-3,
                               double       rtol = 1e-3);

}  // namespace tinyforge
