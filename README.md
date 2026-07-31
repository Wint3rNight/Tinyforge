# Tinyforge

CUDA performance lab and (eventually) a minimal transformer inference engine, built from scratch on an RTX 3050. No cuBLAS, no cuDNN.

## Status

Phase 3 complete (v0.2) — shared-memory tiling, profiled against the naive baseline. Per-phase progress is tracked by the annotated git tags (`v0.1`, `v0.2`, …) and the results table below.

## GEMM results so far

Square FP32 GEMM on RTX 3050 (sm_86, 9.1 TFLOPS FP32 roof). cuBLAS SGEMM (pedantic FP32) as the reference. Every kernel is validated against cuBLAS and a CPU oracle before timing.

| Kernel | 4096³ GFLOPS | % of FP32 peak | % of cuBLAS |
|---|---|---|---|
| v1 naive | **458** | 5.0% | 13.5% |
| v2 shared-memory tiled (32×32) | 360 | 4.0% | 10.6% |
| v2 shared-memory tiled (16×16) | 388 | 4.3% | 11.4% |
| _cuBLAS (reference)_ | _3395_ | _37%_ | _100%_ |

**Phase 3 is a deliberate negative result.** Nsight showed the naive kernel is bound by *load-instruction throughput* — LSU pegged at 99% while DRAM sat idle at 58%. Shared-memory tiling reduces DRAM **traffic**, which cut memory traffic 4.7× (112 → 24 GB/s) and made the kernel ~12% **slower**: the inner loop is still 2 loads per FMA, and a shared-memory load costs a load-store slot exactly like a global one. The bottleneck never moved.

The fix is to change that ratio, which requires each thread to compute multiple outputs — register tiling, Phase 4. The table grows a row per phase.

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
