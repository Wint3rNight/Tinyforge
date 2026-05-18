# Tinyforge

CUDA performance lab and (eventually) a minimal transformer inference engine, built from scratch on an RTX 3050. No cuBLAS, no cuDNN.

## Status

Phase 0 — project setup.

## Build

Requires CUDA toolkit 12.0+ and CMake 3.20+.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/hello_cuda
```

## Roadmap

- **v0.5** — eight progressively optimized GEMM kernels with benchmark harness, Nsight profiling, and roofline analysis.

Target hardware: NVIDIA RTX 3050 (sm_86).
