# Phase 2 — Naive GEMM Baseline

**Kernel:** [`src/kernels/gemm_v1_naive.cu`](../src/kernels/gemm_v1_naive.cu) · **Card:** RTX 3050 Laptop (sm_86, 16 SMs, 9.1 TFLOPS FP32 roof, 224 GB/s DRAM) · **Date:** 2026-07-24

One thread per output element of `C = A·B` (row-major, FP32, square). Each thread walks the K dimension reading one row of A and one column of B from global memory. The point of this kernel is not speed — it is to establish the correctness→benchmark→profile pipeline and to *measure* why naive matmul is slow, giving Phases 3–6 a number to beat.

## Results

Square GEMM, `N` = M = N = K. cuBLAS SGEMM (pedantic FP32, no TF32) is the baseline. All kernels validated correct first: naive vs cuBLAS at every size, plus naive vs the CPU triple-loop oracle at N ≤ 1024.

| N | naive mean | naive GFLOPS | % of FP32 peak | cuBLAS GFLOPS | naive / cuBLAS |
|---|---|---|---|---|---|
| 256  | 0.65 ms | 51 | 0.6% | 285 | 18% |
| 512  | 5.05 ms | 53 | 0.6% | 365 | 15% |
| 1024 | 4.86 ms* | 442* | 4.9% | 4273 | 3.2% |
| 2048 | 36.0 ms | 477 | 5.2% | 4548 | 10.5% |
| 4096 | 296.6 ms | 463 | 5.1% | 4121 | 11.2% |

\* N=1024 harness *mean* was 15.9 ms with std=19 ms — the GPU was flapping between idle (P8) and active clocks mid-measurement, exactly the Phase 1 P-state artifact. The **min** was 4.86 ms, which matches the Nsight clock-locked duration to the millisecond; that is the honest warm number. Treat N ≥ 2048 as the trustworthy rows (Phase 1's "launch must run ≥ ~1 ms *and* stay warm" rule).

**Headline:** warm naive GEMM sustains **~465 GFLOPS ≈ 5% of the FP32 peak, ~11% of cuBLAS.** That is the baseline the next five phases climb from.

## Roofline placement

Naive GEMM's arithmetic intensity is fixed by the algorithm: per output element it does `2K` FLOPs against `2K` floats (`8K` bytes) read from memory → **AI = 0.25 FLOP/byte**. The RTX 3050 ridge point (where the memory roof meets the 9.1 TFLOPS compute roof) is at 9100 ÷ 224 ≈ **41 FLOP/byte**. Naive GEMM sits ~160× to the *left* of the ridge — deep in memory-bound territory, nowhere near the compute roof.

The bare memory roof at AI=0.25 is 224 × 0.25 ≈ **56 GFLOPS** if every byte came from DRAM. We measure ~465, i.e. **~8× above the bare DRAM roof** — because the caches serve most of the traffic for free (see below). That cache reuse is accidental, a free-rider effect of the L1/L2, not something the kernel earns. Phases 3–5 turn it into *earned* on-chip reuse and drag the AI to the right of the ridge.

## Nsight Compute deep-dive (warm N=1024, `results/ncu/gemm_v1_naive_N1024.ncu-rep`)

This is where the baseline earns its keep. The naive profile is subtler than the textbook "memory-bandwidth-bound" story, and the subtlety is the whole lesson.

| Metric | Value | Reading |
|---|---|---|
| Duration | 4.86 ms | matches harness `min` — clock-locked, honest |
| Compute (SM) Throughput | 99.4% | **both SoL bars pegged — see note** |
| Memory Throughput | 99.4% | " |
| **Mem Pipes Busy (LSU)** | **99.4%** | **the actual bottleneck** |
| FMA / ALU pipe | 28.4% | the compute units we *want* busy — starved |
| DRAM Throughput | 58.5% (112 GB/s) | **not** the wall |
| L1/TEX Hit Rate | 87.7% | most loads never reach DRAM |
| L2 Hit Rate | 18.7% | irrelevant — L1 already caught 88% |
| Achieved Occupancy | 99.0% (47.5/48 warps) | **not** an occupancy problem |
| Issued Warp / Scheduler | 0.29 | schedulers idle 71% of cycles — stalled |
| Registers / Thread | 40 | modest |

**What it says.** Both Speed-of-Light headline bars read an identical 99.4% because they are both surfacing the *same* saturated pipe: the **Load-Store Unit (LSU)**. Nsight names it outright — *"the highest utilized pipeline (99.4%) is LSU."* The kernel is not DRAM-bandwidth-bound (DRAM is at 58%); it is **load-instruction-throughput bound**. Each element of A is re-fetched N times and each element of B M times, so the kernel issues an enormous number of `LD` instructions. L1 absorbs 88% of them — so they're cheap in *bytes* — but there are so many that the LSU pipe itself saturates. Meanwhile the FMA units, the ones doing the actual multiply-adds, idle at 28%, and the warp schedulers issue an instruction only 29% of cycles because every warp is queued behind the LSU.

**Why this matters for the story.** The §1.4 pre-work predicted "Compute throughput low, Memory throughput high." Reality refined it: on this kernel *both* SoL bars are high because they collapse onto one pipe (LSU), and the real diagnosis needs the pipe-level breakdown, not the SoL headline. The correct one-sentence verdict is: **naive GEMM is bound by the number of load instructions, not by DRAM bandwidth — occupancy and DRAM are both fine; the LSU is the wall.**

## Why it's slow, and what Phase 3 changes

Naive GEMM burns its cycles *issuing loads*, not computing. Every thread independently re-reads the same A rows and B columns its neighbours are also reading; the redundancy is served by L1 (cheap in bytes) but still costs one LSU slot per load (expensive in instructions).

Phase 3 (shared-memory tiling) attacks exactly this: load each BK-deep tile of A and B from global memory *once* into `__shared__`, then have every thread in the block read its operands from shared memory during the K-loop. That collapses the global-load instruction count by roughly the tile width, unclogs the LSU, and lets the FMA pipe finally climb — moving the kernel toward the compute roof. The predicted profile shift: LSU drops, FMA rises, `Issued Warp/Scheduler` climbs, GFLOPS multiplies. We'll measure it and find out.

## Reproduce

```sh
cmake --preset default && cmake --build build --target gemm_v1_naive -j
./build/gemm_v1_naive
# profile (needs GPU perf-counter permission; see STUDY_NOTES §5.6):
ncu --kernel-name gemm_naive --launch-skip 40 --launch-count 1 --set full \
    -f -o results/ncu/gemm_v1_naive_N1024 ./build/gemm_v1_naive
ncu-ui results/ncu/gemm_v1_naive_N1024.ncu-rep
```
