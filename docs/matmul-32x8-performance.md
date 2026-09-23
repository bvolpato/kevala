# Shared FP32 matrix kernels

Measured on 2026-09-23 against main at `f8147e1`. The shared kernel improves Laya,
Kev, SemIf, and Gemma on the measured RTX 5070 Ti. These results cover one GPU and
browser configuration, not a universal performance ranking.
[The measurements](benchmarks/matmul-32x8-linux-2026-09-23.json) include samples,
profiles, source hashes, and all nine model checks.

## What changed

The FP32 matrix kernel now uses a 32×8 workgroup. Each thread reuses two weight
vectors across multiple input rows and keeps its output accumulators in named
`vec2` values. The short variants use two, four, or six accumulators; the largest
uses eight. Tile dimensions and host dispatch counts stay the same.

This changes thread placement and compiler scheduling. Q8 dequantization, the
order of dot products, FP32 accumulation, split-K reduction, bias, residual, and
ReLU operations remain the same. FP16 and two-group variants retain their prior
layout. The automatic selector and manual API overrides continue to work.

The change is shared by projection and feed-forward matrices across architectures.
It also helps the small matrices in Gemma's selected-row tail. We tested paired
projection fusion separately, but its remaining whole-suite benefit was only
about 0.4% after this improvement, so it is not included.

## Full-model results

Ubuntu, Ryzen 9 9950X3D, RTX 5070 Ti 16 GB, NVIDIA driver 595.71.05, Chrome
149.0.7827.200. Chrome ran headless in an isolated session with the NVIDIA Vulkan
ICD and precise developer timestamps. It exposed subgroups but no `shader-f16`.
GPU jobs ran serially and rejected software adapters.

Each model uses three request sizes, two warmups, and five measured requests per
size with unique inputs. A model's result is the geometric mean of its three
median GPU times. The overall result is the geometric mean across four models.

| Model | Tokens | Main | Updated kernel | Reduction |
| --- | --- | ---: | ---: | ---: |
| Laya | 47 / 140 / 512 | 12.789 ms | 11.776 ms | 7.9% |
| Kev 0.8B | 31 / 124 / 532 | 17.741 ms | 16.438 ms | 7.3% |
| SemIf 2B | 105 / 198 / 606 | 61.823 ms | 55.510 ms | 10.2% |
| Gemma E2B | 99 / 193 / 601 | 40.731 ms | 36.603 ms | 10.1% |
| Overall | | 27.493 ms | 25.043 ms | 8.9% |

GPU time excludes loading, CPU preparation, queue waiting, and readback. These
models use different workloads, so the table is not an equal-task comparison
between model families. It also does not mean that every request finishes below
the reported value. Round-trip samples in the JSON include profiling overhead.

All 18 isolated matrix shapes improved in the paired run, by 7.1–16.9%. GPU
clocks and load-time recurrence selection can vary across runs. The JSON retains
the individual samples rather than only the summary.

Firefox 152.0.3 completed the four-family regression suite at 31.493 ms overall
using its unchanged FP16 path. No Firefox speedup is attributed to this change.

## Correctness

- All 18 paired matrix output arrays have a maximum absolute difference of zero
  from main. All 44 independent CPU reference checks pass, including small full
  matrices, sampled large matrices, partial tiles, split reduction, bias,
  residual, and ReLU.
- All nine supported weights retain **132/132 reference decisions** in Chrome.
  The 225 recorded Kev, SemIf, and Gemma probabilities are unchanged with the same
  recurrence tuning choice. No probability tolerance was widened.
- Chrome Laya retains its existing strict failure: maximum probability difference
  `0.024006444195624588` exceeds the `0.024` threshold. Main produces the same
  value. This is not reported as a passing probability check.
- The Kev cache hit and extension check has zero probability error.
- Local checks passed 64 Rust workspace tests, 102 JavaScript tests, four WGSL
  rendering tests, 75 JavaScript syntax checks, and all three WASM builds and
  validation checks.

Weights, quantization, and CPU kernels are unchanged. The generated WASM binaries
remain build and release artifacts; they are not checked into Git.

## Reproduce

Build with `pnpm build`, place the model packs in `tmp/`, then serve the repository
with `node scripts/serve.mjs . --port=18106 --isolate`. Run the browser suite on an
isolated hardware Chrome session:

```sh
KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
node scripts/bench-gpu-suite.mjs http://127.0.0.1:18106/ tmp/matmul-results
```

The paired matrix harness is `dev/gpu-bench.html`. Use
`kernels=runtime,runtime_baseline` and `baselineWasm=<url>` to compare two built
kernel sources on the same inputs and adapter. The JSON records the shapes,
seed, warmup count, sample count, and per-shape repetitions.
