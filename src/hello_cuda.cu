#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("hello from block %d thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    int device = 0;
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);
    std::printf("device: %s (sm_%d%d), %zu MiB\n",
                props.name, props.major, props.minor,
                props.totalGlobalMem / (1024 * 1024));

    hello_kernel<<<2, 4>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
