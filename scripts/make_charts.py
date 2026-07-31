#!/usr/bin/env python3
"""Generate results/final_sweep.csv and the three README charts as SVG.

No third-party dependencies — SVG is emitted directly so the charts render on
GitHub in both light and dark themes via an embedded prefers-color-scheme block.

Data comes from results/final_sweep_raw.txt (produced by scripts/run_benchmarks.sh)
and from the Nsight profiles recorded in docs/phase*.md.
"""

import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")

SIZES = [256, 512, 1024, 2048, 4096]

# GFLOPS from the final sweep (results/final_sweep_raw.txt). None = not run.
SERIES = [
    ("naive",              [437.6,  489.5,  441.6,  476.7,  466.1]),
    ("shared tiled",       [387.1,  431.4,  389.5,  435.1,  429.7]),
    ("register 8x8",       [417.9, 1653.3, 2183.1, 2269.6, 2285.2]),
    ("+ float4",           [660.9, 2571.9, 4124.0, 3950.6, 3726.6]),
    ("tuned (BK=16)",      [None,   None,  4133.9, 4458.1, 4288.8]),
    ("cuBLAS",             [1649.1, 3391.4, 4295.5, 4618.1, 4119.0]),
]

# Headline bar chart uses one same-run measurement at 4096 so every bar is
# directly comparable, including cuBLAS with its tensor-core path unlocked.
BARS_4096 = [
    ("naive",                    466.1,  "ours"),
    ("shared tiled",             429.7,  "ours"),
    ("register 8x8",            2285.2,  "ours"),
    ("+ float4",                3726.6,  "ours"),
    ("tuned (BK=16)",           4424.1,  "ours"),
    ("cuBLAS — strict FP32",    4216.4,  "ref"),
    ("cuBLAS — TF32 tensor",    7728.4,  "tc"),
]

# Roofline: measured at N=1024. AI = FLOPs / measured DRAM bytes.
#   dram_bytes = dram_pct * 224e9 * seconds
PROFILED = [  # name, duration_s, dram_pct, gflops
    ("naive",         4.86e-3, 0.585, 441.6),
    ("shared tiled",  5.51e-3, 0.263, 389.5),
    ("register 8x8",  1.11e-3, 0.107, 2183.1),
    ("+ float4",      0.565e-3, 0.1576, 4124.0),
    ("tuned (BK=16)", 0.515e-3, 0.2186, 4133.9),
]

PEAK_GFLOPS = 6884.0     # measured, src/kernels/fp32_peak.cu
DRAM_GBPS = 224.0

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

STYLE = """
  <style>
    .surface { fill: #fcfcfb; }
    .t-pri   { fill: #0b0b0b; }
    .t-sec   { fill: #52514e; }
    .t-mut   { fill: #78766f; }
    .grid    { stroke: #e3e2dd; stroke-width: 1; }
    .axis    { stroke: #c9c7c0; stroke-width: 1; }
    .roof    { stroke: #9a9890; stroke-width: 2; stroke-dasharray: 6 4; fill: none; }
    .ref     { fill: #9a9890; }
    .accent  { fill: #2a78d6; }
    .accent-s{ stroke: #2a78d6; }
%SERIES_LIGHT%
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    @media (prefers-color-scheme: dark) {
      .surface { fill: #1a1a19; }
      .t-pri   { fill: #ffffff; }
      .t-sec   { fill: #c3c2b7; }
      .t-mut   { fill: #98968c; }
      .grid    { stroke: #2f2f2d; }
      .axis    { stroke: #4a4a46; }
      .roof    { stroke: #7c7a73; }
      .ref     { fill: #7c7a73; }
      .accent  { fill: #3987e5; }
      .accent-s{ stroke: #3987e5; }
%SERIES_DARK%
    }
  </style>
"""


def style_block():
    l = "\n".join(f"    .s{i} {{ fill: {c}; }}  .s{i}-s {{ stroke: {c}; }}"
                  for i, c in enumerate(LIGHT))
    d = "\n".join(f"      .s{i} {{ fill: {c}; }}  .s{i}-s {{ stroke: {c}; }}"
                  for i, c in enumerate(DARK))
    return STYLE.replace("%SERIES_LIGHT%", l).replace("%SERIES_DARK%", d)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_open(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{esc(title)}">\n'
            f'  <title>{esc(title)}</title>\n'
            f'{style_block()}'
            f'  <rect class="surface" x="0" y="0" width="{w}" height="{h}" rx="6"/>\n')


# ── Chart 1: horizontal bars, GFLOPS per kernel at 4096³ ────────────────────
# Magnitude comparison → emphasis form: our kernels one hue, cuBLAS gray.
def chart_bars(path):
    rows = sorted(BARS_4096, key=lambda r: r[1])
    W, H = 760, 372
    L, R, T = 178, 96, 56
    plot_w = W - L - R
    bar_h, gap = 30, 12
    vmax = max(r[1] for r in rows) * 1.06

    o = [svg_open(W, H, "GEMM throughput by kernel at 4096 cubed")]
    o.append(f'  <text class="t-pri" x="20" y="26" font-size="15" font-weight="600">'
             f'GEMM throughput at 4096³ · single run, all bars comparable</text>\n')
    o.append(f'  <text class="t-sec" x="20" y="44" font-size="12">'
             f'RTX 3050 Laptop · measured FP32 ceiling {PEAK_GFLOPS:.0f} GFLOPS · '
             f'higher is better</text>\n')

    for i, (name, val, kind) in enumerate(rows):
        y = T + i * (bar_h + gap)
        w = max(2.0, plot_w * val / vmax)
        cls = {"ours": "accent", "ref": "ref", "tc": "ref"}[kind]
        # cuBLAS-with-tensor-cores is hatched: same library, different hardware.
        extra = ' fill-opacity="0.45"' if kind == "tc" else ""
        weight = "600" if kind == "ours" else "500"
        o.append(f'  <text class="t-sec" x="{L - 10}" y="{y + bar_h/2 + 4}" '
                 f'font-size="12" text-anchor="end">{esc(name)}</text>\n')
        o.append(f'  <rect class="{cls}" x="{L}" y="{y}" width="{w:.1f}" '
                 f'height="{bar_h}" rx="4"{extra}/>\n')
        if kind == "tc":
            o.append(f'  <rect class="ref" x="{L}" y="{y}" width="{w:.1f}" '
                     f'height="{bar_h}" rx="4" fill="none" stroke="#9a9890" '
                     f'stroke-width="2"/>\n')
        o.append(f'  <text class="t-pri" x="{L + w + 8:.1f}" y="{y + bar_h/2 + 4}" '
                 f'font-size="12" font-weight="{weight}">{val:,.0f}</text>\n')

    base_y = T + len(rows) * (bar_h + gap)
    o.append(f'  <line class="axis" x1="{L}" y1="{T - 8}" x2="{L}" y2="{base_y - gap + 4}"/>\n')
    o.append(f'  <text class="t-mut" x="{L}" y="{base_y + 15}" font-size="11">GFLOPS</text>\n')
    o.append(f'  <text class="t-mut" x="{W - 20}" y="{base_y + 15}" font-size="11" '
             f'text-anchor="end">blue = this project · grey = cuBLAS</text>\n')
    o.append("</svg>\n")
    open(path, "w").write("".join(o))


# ── Chart 2: GFLOPS vs matrix size, one line per kernel ─────────────────────
def chart_lines(path):
    W, H = 720, 400
    L, R, T, B = 62, 132, 56, 46
    pw, ph = W - L - R, H - T - B
    vmax = 5000.0
    xs = [L + pw * i / (len(SIZES) - 1) for i in range(len(SIZES))]

    def yv(v):
        return T + ph - ph * (v / vmax)

    o = [svg_open(W, H, "GEMM throughput versus matrix size")]
    o.append(f'  <text class="t-pri" x="20" y="26" font-size="15" font-weight="600">'
             f'Throughput vs matrix size</text>\n')
    o.append(f'  <text class="t-sec" x="20" y="44" font-size="12">'
             f'square FP32 GEMM · GFLOPS · each kernel validated before timing</text>\n')

    for g in range(0, 5001, 1000):
        y = yv(g)
        o.append(f'  <line class="grid" x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}"/>\n')
        o.append(f'  <text class="t-mut" x="{L - 8}" y="{y + 4:.1f}" font-size="11" '
                 f'text-anchor="end">{g:,}</text>\n')
    for i, n in enumerate(SIZES):
        o.append(f'  <text class="t-mut" x="{xs[i]:.1f}" y="{T + ph + 18}" font-size="11" '
                 f'text-anchor="middle">{n}</text>\n')
    o.append(f'  <line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>\n')
    o.append(f'  <text class="t-mut" x="{L + pw / 2:.0f}" y="{H - 10}" font-size="11" '
             f'text-anchor="middle">matrix size N (N×N×N)</text>\n')

    # Direct labels at the right end satisfy the light-mode contrast relief rule.
    ends = []
    for si, (name, vals) in enumerate(SERIES):
        pts = [(xs[i], yv(v)) for i, v in enumerate(vals) if v is not None]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        o.append(f'  <path class="s{si}-s" d="{d}" fill="none" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>\n')
        for x, y in pts:
            o.append(f'  <circle class="s{si}" cx="{x:.1f}" cy="{y:.1f}" r="4"/>\n')
        ends.append((pts[-1][1], name, si))

    ends.sort()
    prev = -1e9
    for y, name, si in ends:
        y = max(y, prev + 13)
        prev = y
        o.append(f'  <circle class="s{si}" cx="{L + pw + 12}" cy="{y:.1f}" r="4"/>\n')
        o.append(f'  <text class="t-sec" x="{L + pw + 22}" y="{y + 4:.1f}" '
                 f'font-size="11">{esc(name)}</text>\n')
    o.append("</svg>\n")
    open(path, "w").write("".join(o))


# ── Chart 3: roofline (log-log), measured AI from profiled DRAM traffic ─────
def chart_roofline(path):
    W, H = 720, 400
    L, R, T, B = 62, 150, 56, 46
    pw, ph = W - L - R, H - T - B
    ai_lo, ai_hi = 1.0, 400.0
    g_lo, g_hi = 100.0, 10000.0

    def x_of(ai):
        return L + pw * (math.log10(ai) - math.log10(ai_lo)) / (math.log10(ai_hi) - math.log10(ai_lo))

    def y_of(g):
        return T + ph - ph * (math.log10(g) - math.log10(g_lo)) / (math.log10(g_hi) - math.log10(g_lo))

    ridge = PEAK_GFLOPS / DRAM_GBPS
    o = [svg_open(W, H, "Roofline placement of each kernel")]
    o.append(f'  <text class="t-pri" x="20" y="26" font-size="15" font-weight="600">'
             f'Roofline — measured arithmetic intensity</text>\n')
    o.append(f'  <text class="t-sec" x="20" y="44" font-size="12">'
             f'GFLOPS vs AI from profiled DRAM traffic at N=1024 · ridge point '
             f'{ridge:.0f} FLOP/byte</text>\n')

    for g in [100, 1000, 10000]:
        y = y_of(g)
        o.append(f'  <line class="grid" x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}"/>\n')
        o.append(f'  <text class="t-mut" x="{L - 8}" y="{y + 4:.1f}" font-size="11" '
                 f'text-anchor="end">{g:,}</text>\n')
    for ai in [1, 10, 100]:
        x = x_of(ai)
        o.append(f'  <line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T + ph}"/>\n')
        o.append(f'  <text class="t-mut" x="{x:.1f}" y="{T + ph + 18}" font-size="11" '
                 f'text-anchor="middle">{ai}</text>\n')

    # Roofs: memory-bound slope then the flat compute ceiling.
    d = (f'M {x_of(ai_lo):.1f} {y_of(ai_lo * DRAM_GBPS):.1f} '
         f'L {x_of(ridge):.1f} {y_of(PEAK_GFLOPS):.1f} '
         f'L {x_of(ai_hi):.1f} {y_of(PEAK_GFLOPS):.1f}')
    o.append(f'  <path class="roof" d="{d}"/>\n')
    o.append(f'  <text class="t-mut" x="{x_of(1.5):.1f}" y="{y_of(1.5 * DRAM_GBPS) - 8:.1f}" '
             f'font-size="10" transform="rotate(-31 {x_of(1.5):.1f} '
             f'{y_of(1.5 * DRAM_GBPS) - 8:.1f})">DRAM roof · 224 GB/s</text>\n')
    o.append(f'  <text class="t-mut" x="{x_of(ai_hi) - 4:.1f}" y="{y_of(PEAK_GFLOPS) - 8:.1f}" '
             f'font-size="10" text-anchor="end">compute roof · {PEAK_GFLOPS:.0f} GFLOPS</text>\n')

    labels = []
    for name, dur, dram_pct, gf in PROFILED:
        ai = (2 * 1024 ** 3) / (dram_pct * DRAM_GBPS * 1e9 * dur)
        x, y = x_of(ai), y_of(gf)
        o.append(f'  <circle class="accent" cx="{x:.1f}" cy="{y:.1f}" r="6"/>\n')
        labels.append((y, x, name, ai))

    labels.sort()
    prev = -1e9
    for y, x, name, ai in labels:
        ly = max(y, prev + 26)   # each label is two lines (name + AI)
        prev = ly
        o.append(f'  <line class="axis" x1="{x + 8:.1f}" y1="{y:.1f}" '
                 f'x2="{L + pw + 8:.1f}" y2="{ly:.1f}"/>\n')
        o.append(f'  <text class="t-sec" x="{L + pw + 13}" y="{ly + 4:.1f}" font-size="11">'
                 f'{esc(name)}</text>\n')
        o.append(f'  <text class="t-mut" x="{L + pw + 13}" y="{ly + 15:.1f}" font-size="9">'
                 f'AI {ai:.0f}</text>\n')

    o.append(f'  <line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>\n')
    o.append(f'  <text class="t-mut" x="{L + pw / 2:.0f}" y="{H - 10}" font-size="11" '
             f'text-anchor="middle">arithmetic intensity (FLOP / DRAM byte)</text>\n')
    o.append("</svg>\n")
    open(path, "w").write("".join(o))


def write_csv(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "N", "gflops", "pct_of_cublas", "pct_of_measured_peak"])
        cub = dict(zip(SIZES, SERIES[-1][1]))
        for name, vals in SERIES:
            for n, v in zip(SIZES, vals):
                if v is None:
                    continue
                w.writerow([name, n, f"{v:.1f}", f"{100 * v / cub[n]:.1f}",
                            f"{100 * v / PEAK_GFLOPS:.1f}"])


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    write_csv(os.path.join(RESULTS, "final_sweep.csv"))
    chart_bars(os.path.join(ASSETS, "gflops-by-kernel.svg"))
    chart_lines(os.path.join(ASSETS, "gflops-vs-size.svg"))
    chart_roofline(os.path.join(ASSETS, "roofline.svg"))
    print("wrote results/final_sweep.csv and assets/*.svg")
