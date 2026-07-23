# Tinyforge

CUDA performance lab and (eventually) a minimal transformer inference engine, built from scratch on an RTX 3050. No cuBLAS, no cuDNN.

## Status

Phase 2 complete (v0.2-alpha) — naive GEMM baseline with a cuBLAS reference and the first roofline + Nsight analysis. See [CHANGELOG.md](CHANGELOG.md) for per-phase progress and [docs/phase2-baseline.md](docs/phase2-baseline.md) for the Phase 2 writeup.

## GEMM results so far

Square FP32 GEMM on RTX 3050 (sm_86, 9.1 TFLOPS FP32 roof). cuBLAS SGEMM (pedantic FP32) as the reference. Every kernel is validated for correctness before timing.

| Kernel | 4096³ GFLOPS | % of FP32 peak | % of cuBLAS |
|---|---|---|---|
| v1 naive | 463 | 5.1% | 11.2% |
| _cuBLAS (reference)_ | _4121_ | _45%_ | _100%_ |

The naive kernel profiles as **load-store-unit bound** (LSU 99%, DRAM only 58%, FMA units idle at 28%) — it's slow because of the *number* of redundant loads, not memory bandwidth. Removing that redundancy with shared-memory tiling is Phase 3. The table grows a row per phase.

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
