# Avoid padded FP32 matrix rows

The shared matrix dispatcher now uses its existing 48-row tile when it covers the input
with the same number of workgroups as the 64-row tile. This applies to 65–96 and 129–144
tokens on the FP32, single-column-group path. It removes 25% of padded row arithmetic
without changing the grid, K splits, ordered dot products, or reduction.

The selection depends on the operation shape and precision. It applies to Laya, Kev,
SemIf, and Gemma. FP16 and two-column-group dispatch are unchanged. There are no new
shaders, weights, load-time benchmarks, or user controls.

## Measurements

Measured serially on an NVIDIA RTX 5070 Ti, driver 595.71.05, Ubuntu, Chrome
149.0.7827.200. Chrome used hardware WebGPU, FP32 shaders, and precise timestamps in a
dedicated headless browser. The baseline is main at `4c948879`, after the shared 32×8
kernel change. The candidate contains this dispatch change without experimental
paired-projection fusion.

Each model value is the geometric mean of median GPU time across three request sizes,
with two warmups and five measured requests per size. It is not unprofiled round-trip
latency. The sample set and raw profiles are in the [benchmark artifact](benchmarks/fp32-row-padding-linux-2026-09-23.json).

| Model | Before | After |
| --- | ---: | ---: |
| Laya | 12.063 ms | 11.322 ms |
| Kev 0.8B | 16.264 ms | 16.174 ms |
| SemIf 2B | 56.509 ms | 56.315 ms |
| Gemma E2B | 36.609 ms | 36.428 ms |
| Four-family geometric mean | 25.241 ms | 24.757 ms |

Laya improves 6.1% across the three sizes. Its affected 140-token request improves
from 12.092 to 10.259 ms on the GPU, a 15.2% reduction. The four-family summary improves
1.9%. The small changes in the other model summaries are not claimed as speedups from
this policy; their request sizes differ, and timing varies between runs.

A separate sweep covers ten eligible matrix shapes, including Laya, Kev, SemIf, and
Gemma dimensions. The smaller tile runs 1.14–1.31× as fast, and every paired output
array has zero maximum absolute difference. All 24 CPU-reference checks pass. A second
hardware check verifies that the benchmark's `runtime` and explicit `:auto` selectors
both exercise the same new policy.

Firefox 152.0.3 completes the four-model regression suite at 31.387 ms. It uses the
unchanged FP16 path; this is not a Firefox speedup claim. Measurements cover one GPU.

## Correctness and validation

- All nine supported weights preserve 132/132 reference decisions.
- All 225 recorded Kev, SemIf, and Gemma probabilities exactly match the baseline
  with matching recurrence selection. This also corrects the previous report's count
  of 255; the recorded arrays and their comparisons are unchanged.
- Chrome Laya retains its existing maximum probability difference of
  `0.024006444195624588`, above the unchanged `0.024` threshold. Its strict probability
  check is not green. Its runner does not record individual probabilities.
- The Kev cache hit and extension check has zero probability error.
- 103 JavaScript tests, 75 JavaScript syntax checks, three WASM validations, and site
  staging pass. The new test covers row-policy boundaries, split counts, and FP16/G2
  exclusions. Rust and generated WASM are unchanged.
- Independent review found no correctness issues in grid sizing, split reduction,
  tails, epilogues, or the existing fused-kernel fallback.

The artifact includes timestamp samples, model profiles, golden outputs, browser
metadata, and source/WASM hashes. WASM files remain generated build artifacts.
