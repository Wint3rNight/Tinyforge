#pragma once

#include <tinyforge/cuda_check.hpp>

// Pull the GPU out of its idle power state before timing anything.
//
// This card idles at ~210 MHz and only reaches ~1950 MHz under sustained load,
// so a short kernel launched on a cold GPU measures the idle clock rather than
// the hardware. Worse, long host-side work — a CPU reference loop, for instance
// — lets it fall back down again, so waking once at startup is not enough.
//
// Call this AFTER all host-side work and IMMEDIATELY before timing. Skipping it
// understated small-matrix results in this project by up to 8x.

namespace tinyforge {

static __global__ void tf_wake_kernel(float* __restrict__ sink, int iters) {
    float a = 1.0000001f, b = 0.9999999f, acc = threadIdx.x;
    for (int i = 0; i < iters; ++i) acc = fmaf(a, b, acc);
    if (acc == 12345.678f) *sink = acc;   // never true; keeps the loop alive
}

inline void wake_gpu(int num_sms, int rounds = 15, int iters = 200000) {
    float* d = nullptr;
    TF_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
    for (int i = 0; i < rounds; ++i)
        tf_wake_kernel<<<num_sms * 8, 256>>>(d, iters);
    TF_CUDA_CHECK(cudaDeviceSynchronize());
    TF_CUDA_CHECK(cudaFree(d));
}

}  // namespace tinyforge
