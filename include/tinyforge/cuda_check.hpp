#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TF_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        cudaError_t _tf_err = (expr);                                          \
        if (_tf_err != cudaSuccess) {                                          \
            std::fprintf(stderr,                                               \
                         "CUDA error at %s:%d in `%s`: %s\n",                  \
                         __FILE__, __LINE__, #expr,                            \
                         cudaGetErrorString(_tf_err));                         \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define TF_CUDA_CHECK_LAST() TF_CUDA_CHECK(cudaGetLastError())
