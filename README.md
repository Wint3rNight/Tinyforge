# Tinyforge

CUDA performance lab and (eventually) a minimal transformer inference engine, built from scratch on an RTX 3050. No cuBLAS, no cuDNN.

## Status

Phase 4 complete (v0.3-alpha) — **~8× faster than the naive baseline, reaching ~90% of cuBLAS**. Per-phase progress is tracked by the annotated git tags (`v0.1`, `v0.2`, …) and the results table below.

## GEMM results

Square FP32 GEMM at 4096³ on an RTX 3050 Laptop (sm_86). cuBLAS SGEMM (pedantic FP32, same run, same clocks) as the reference. Every kernel is validated against cuBLAS and a CPU oracle before timing.

| Kernel | GFLOPS | vs naive | % of cuBLAS |
|---|---|---|---|
| v1 naive | 465 | 1.00× | 11.3% |
| v2 shared-memory tiled | 430 | 0.92× | 10.4% |
| v3 register tiled (8×8/thread) | 2286 | 4.92× | 55.5% |
| **v3 + `float4` vectorized loads** | **3773** | **8.11×** | **91.6%** |
| _cuBLAS (reference)_ | _4119_ | _8.86×_ | _100%_ |

Representative run. This is a thermally-limited laptop part, so absolute GFLOPS drift a few percent between runs — across runs the vectorized kernel lands at **~7.6–8.1× naive and ~89–92% of cuBLAS** (both it and the cuBLAS reference move together). At 1024³ it reaches **96% of cuBLAS**.

### The optimization story

Nsight showed the naive kernel was bound by **load-instruction throughput**, not bandwidth — the load/store unit sat at 99% while DRAM idled at 58%.

- **Shared-memory tiling made it slower (0.92×).** It cut DRAM traffic 4.7×, but the inner loop still issued 2 loads per FMA, and a shared-memory load costs a load/store slot exactly like a global one. Optimizing a resource that already had slack.
- **Register tiling fixed it (4.92×).** Giving each thread an 8×8 block of outputs changes the ratio to 16 loads per 64 FMAs — 8× more arithmetic per load. LSU utilization fell 99% → 62%.
- **Vectorized `float4` loads added 1.65×** by cutting load instructions another 4×. LSU fell to 33%, IPC rose 1.16 → 2.39.

Notably, **occupancy dropped from 99% to 33% while performance rose 8×** — register tiling trades thread-level parallelism for instruction-level parallelism, and on this workload that trade wins decisively.

Remaining known gaps: an 8-way shared-memory bank conflict on the A-tile store, and no tile-shape sweep. Both are Phase 5.

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
