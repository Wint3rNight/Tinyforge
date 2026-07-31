#!/usr/bin/env bash
# Full benchmark sweep -> results/final_sweep_raw.txt, then regenerate CSV + charts.
# Takes several minutes: the naive kernel alone is ~300 ms per launch at 4096.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="/opt/cuda/bin:$PATH"

cmake --build build -j

OUT=results/final_sweep_raw.txt
mkdir -p results assets

{
  echo "### fp32_peak";        ./build/fp32_peak
  echo "### gemm_v3_register"; ./build/gemm_v3_register
  echo "### gemm_v4_tuned";    ./build/gemm_v4_tuned
} > "$OUT" 2>&1

echo "raw output -> $OUT"

# Numbers in make_charts.py are transcribed from a sweep rather than parsed, so
# that the committed charts stay stable. Update them there if this run differs.
python3 scripts/make_charts.py
