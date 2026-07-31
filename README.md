# Tinyforge

A CUDA performance lab: FP32 matrix multiplication optimized from a naive baseline to cuBLAS-class throughput on an RTX 3050, with every step measured and profiled. The kernels are written from scratch — cuBLAS appears only as a benchmark reference and correctness oracle.

## Status

Phase 5 complete (v0.4) — **9.4× faster than the naive baseline, matching cuBLAS SGEMM in strict FP32.** Per-phase progress is tracked by annotated git tags (`v0.1`, `v0.2`, …) and the table below.

## Results

Square FP32 GEMM at 4096³ on an RTX 3050 Laptop (sm_86, 16 SMs). Every kernel is validated against cuBLAS at all sizes and against a CPU triple-loop oracle at N ≤ 1024 **before** it is timed.

The card's FP32 ceiling here is **measured, not quoted from a spec sheet** — a pure-FMA microbenchmark ([`fp32_peak.cu`](src/kernels/fp32_peak.cu)) gives **6668 GFLOPS** sustained.

| Kernel | GFLOPS | vs naive | % of cuBLAS | % of measured peak |
|---|---|---|---|---|
| v1 naive | 465 | 1.00× | 11% | 7% |
| v2 shared-memory tiled | 430 | 0.92× | 10% | 6% |
| v3 register tiled (8×8/thread) | 2286 | 4.92× | 55% | 34% |
| v3 + `float4` vectorized loads | 3939 | 8.47× | 95% | 59% |
| **v4 tuned (BK=16, padded)** | **4359** | **9.37×** | **105%** | **65%** |
| _cuBLAS (reference)_ | _4151_ | _8.93×_ | _100%_ | _62%_ |

At 2048³ the tuned kernel reaches 100.4% of cuBLAS; at 1024³, 97%.

> **Caveat on "105% of cuBLAS."** cuBLAS is pinned to `CUBLAS_PEDANTIC_MATH` so it stays a valid FP32 correctness oracle. With TF32 enabled it would use tensor cores and be substantially faster. The honest claim is narrow: *matches or slightly exceeds cuBLAS SGEMM in strict FP32 mode, on square matrices, on this card.*

## The optimization story

Profiling drove every decision, including one step that made things worse.

**The naive kernel was bound by load-instruction throughput, not bandwidth.** Nsight showed the load/store unit pegged at 99% while DRAM sat idle at 58% and the FMA units ran at 28%. The problem was the sheer *number* of load instructions, not the bytes moved.

- **Shared-memory tiling made it 8% slower.** It cut DRAM traffic 4.7× — but DRAM was never the constraint. The inner loop still issued 2 loads per FMA, and a shared-memory load consumes a load/store slot exactly like a global one. A textbook optimization applied to a resource that already had 42% slack.
- **Register tiling fixed it (4.9×).** Giving each thread an 8×8 block of outputs turns the inner loop into an outer product: 16 loads feed 64 FMAs instead of 2 feeding 1. LSU utilization fell 99% → 62%.
- **`float4` loads added another 1.7×** by cutting load instructions 4× more. LSU fell to 33%; IPC rose 1.16 → 2.39.
- **Deepening the K-slab (BK 8→16) added 11%**, by halving the barrier and tile-load overhead per unit of compute.

Two results worth highlighting:

**Occupancy fell from 99% to 33% while performance rose 8×.** Register tiling trades thread-level parallelism for instruction-level parallelism — 64 independent FMAs per thread keep the pipelines fed without a deep warp pool. Occupancy is a means, not an end.

**Eliminating shared-memory bank conflicts was worth ~1%.** Padding removed them completely (16.8M → 0, verified by counter), but they were never the bottleneck — a deeper K-slab *with* conflicts beat a shallow one without them.

## Known gaps

- **Load-side bank conflicts (268M) remain** — 16× larger than the store conflicts that padding fixed. Requires XOR swizzling; padding provably cannot fix it under `float4` alignment constraints.
- **Warp tiling and `cp.async` not implemented** — the target was already exceeded; both are learning exercises from here.
- **Tensor cores deferred to v0.6.** FP32 only for now.

## Build and run

Requires CUDA toolkit 12.0+ and CMake 3.20+.

```sh
cmake --preset default
cmake --build build -j

./build/gemm_v4_tuned        # the tuned kernel + tile sweep vs cuBLAS
./build/gemm_v3_register     # register tiling, full ladder from naive
./build/fp32_peak            # measure the card's real FP32 ceiling
```

Each GEMM binary takes an optional size argument (`./build/gemm_v4_tuned 4096`) for a single-size run with a deterministic launch order, which makes Nsight profiling reproducible.

## Notes on measurement

Benchmarking a laptop GPU honestly turned out to be a recurring problem. Two fixes matter:

- **The GPU must be woken out of its idle P-state before timing.** A short kernel on an idle card runs at 210 MHz instead of 1950 MHz — the first version of the peak benchmark reported 776 GFLOPS instead of 6668 for exactly this reason.
- **Long host-side work invalidates the next measurement.** Running the CPU correctness oracle (a single-threaded N³ loop) let the GPU idle back down, making N=1024 read 5.3× low. The harness now re-wakes the GPU after all host work, immediately before timing.

Target hardware: NVIDIA RTX 3050 Laptop (sm_86). Not portable across architectures by design.
