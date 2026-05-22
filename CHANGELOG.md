# Changelog

All notable changes to Tinyforge.

## [v0.1] — 2026-05-22

End of Phase 1 — CUDA fundamentals via vector add and reduction. Internalized the execution model (threads, warps, shared memory, atomics, warp shuffles) and the bandwidth-vs-compute roofline before touching GEMM.

### Added
- `src/kernels/vector_add.cu` — vector addition with grid-stride loop, N sweep across {1M, 16M, 64M, 128M}, correctness-then-benchmark pattern.
- `src/kernels/reduction.cu` — four progressive parallel-reduction kernels:
  - v1 interleaved (naive, warp divergence at every tree step)
  - v2 sequential addressing (whole warps active/idle, no in-warp divergence)
  - v3 warp-shuffle tail (final 32→1 done in registers via `__shfl_down_sync`)
  - v4 atomicAdd combine (skips the CPU final sum, full GPU pipeline)
- `results/ncu/` — first Nsight Compute profiles (full metric set) for vector_add at N=64M and each of the four reduction variants.
- `CHANGELOG.md` (this file).

### Performance

| Kernel | N | Mean | Throughput | % of 224 GB/s peak |
|---|---|---|---|---|
| vector_add | 64 M | 4.44 ms | 181 GB/s | 81% (95% under ncu clock-lock) |
| vector_add | 128 M | 8.98 ms | 179 GB/s | 80% |
| reduce v1 interleaved | 64 M | 1.43 ms | 187 GB/s | 84% |
| reduce v2 seq_addr | 64 M | 1.43 ms | 188 GB/s | 84% |
| reduce v3 warpshuffle | 64 M | 1.43 ms | 187 GB/s | 84% |
| reduce v4 atomic | 64 M | 1.43 ms | 187 GB/s | 84% |

Speedup ladder visible at N=16M: v1 = 6.65 ms vs v3 = 0.36 ms (~18× faster) — divergence cost is observable when the reduction phase isn't drowned in memory traffic. At N=64M+ all four variants converge to the same ~187 GB/s ceiling because reduction is bandwidth-bound at that scale.

### Notes

- **Open question O1 from Phase 0 resolved.** The Phase 0 SAXPY's 8.5 GB/s reading was a P8 idle-state artifact: at N=1M (~60 µs of work) the GPU never leaves its idle clock state. At N≥64M the workload is long enough that the GPU ramps to P0 and reports its real bandwidth. **Implication for the benchmark protocol:** any new kernel needs a runtime ≥ ~1 ms before its GB/s number is trustworthy.

- **O2 (cold-launch throwaway) deferred.** Current 5-warmup-run protocol is sufficient; no observed case where it isn't.

- **O3 opened.** v1 reduction reports 6.65 ms at N=16M with tight std (3%), but only 1.43 ms at N=64M for 4× more data. Anomalous — likely a thermal/clock interaction at intermediate kernel durations. Flagged for investigation when GPU clocks are locked (probably Phase 7 polish).

- **Nsight Compute findings.**
  - vector_add @ N=64M hits **95% memory throughput, 10% compute throughput** — textbook memory-bound. Every instruction stalls ~321 cycles waiting on DRAM.
  - Reduction v1 burns **2.5× more SM compute throughput** than v3/v4 for the same answer (49% vs 20%). The divergent interleaved tree wastes lane cycles on masked-off threads and shared-memory traffic; the warp-shuffle tail eliminates both for the last six tree steps.
  - All four reduction variants show ~32 active threads/warp on average, because the (non-divergent) grid-stride preload dominates instruction count — divergence cost is only visible in *Compute SM Throughput*, not in *Active Threads/Warp*. **Lesson: read multiple Nsight metrics together; any single one can mislead.**
