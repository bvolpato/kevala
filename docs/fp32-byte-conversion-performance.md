# Exact byte conversion in shared FP32 matrix kernels

The FP32 matrix loader now sign-extends each packed int8 byte and converts the
result to FP32 directly. This replaces the mantissa bitcast and subtraction
sequence on the single-column-group path. Both methods represent every integer
from -128 through 127 exactly, including positive zero.

The change applies to Laya, Kev, SemIf, and Gemma. It adds no weight format,
precision mode, tuning step, or model-specific rule. The FP16 and two-column-group
paths retain their existing conversion. Matrix tiling, K splits, scaling,
accumulation, bias, residual addition, and ReLU are unchanged.

## Measurements

Measured serially on an NVIDIA RTX 5070 Ti, driver 595.71.05, Ubuntu, Chrome
149.0.7827.200, using hardware WebGPU and precise timestamps in a dedicated
headless browser. The baseline is main at `4218d994`, including the Gemma RMS
fusion. Both production versions exclude experimental paired-projection fusion.

Each model value is the geometric mean of median GPU times across three request
sizes, with two warmups and five measured requests per size. The candidate column
averages both complete production runs geometrically. The fresh baseline ran
between them. These are profiled GPU times, not unprofiled round-trip latency.

| Model | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Laya | 11.338 ms | 10.924 ms | 3.65% |
| Kev 0.8B | 16.119 ms | 15.402 ms | 4.45% |
| SemIf 2B | 55.490 ms | 54.309 ms | 2.13% |
| Gemma E2B | 35.495 ms | 34.827 ms | 1.88% |
| Four-family geometric mean | 24.494 ms | 23.751 ms | 3.03% |

The two candidate aggregates were 23.612 and 23.891 ms. The preceding production
baseline was 24.310 ms; using it instead of the fresh baseline gives a 2.30%
reduction. Timing variation remains visible, so these measurements establish a
modest improvement on this GPU, not a universal speedup.

Paired matrix tests in both measurement orders generally improve by 2–4%.
One shape varies from 3.10% faster to 0.11% slower, and a long Laya shape improves
by less than 1%. Two earlier full runs on the research branch average a 2.67%
improvement. They are retained in the artifact but excluded from the table because
that branch includes unshipped projection fusion.

Full samples, outputs, selection diagnostics, and final source/WASM hashes are in
the [benchmark artifact](benchmarks/fp32-byte-conversion-linux-2026-09-23.json).
Larger weights have correctness coverage, not throughput claims in this report.

## Correctness and validation

- All 256 possible bytes in each of four packed positions decode identically,
  including negative values and zero.
- Sixteen paired research matrix arrays and eleven production arrays have zero
  maximum absolute difference. All 82 CPU-reference checks pass at unchanged
  limits, including row boundaries, ragged columns, odd K splits, bias, residual
  addition, and ReLU.
- All nine model packs preserve 132/132 reference decisions. All 225 recorded
  Kev, SemIf, and Gemma probabilities match the previous production revision
  exactly when recurrence selection matches.
- Kev 9B initially selected four recurrence lanes instead of the baseline's 16.
  Its maximum probability change was `0.0000007152557373046875`, within its
  unchanged guard. A repeat selected 16 lanes and matched exactly. Both attempts
  remain in the artifact.
- Chrome Laya retains its existing maximum probability difference of
  `0.024006444195624588`, above its unchanged `0.024` threshold. That strict guard
  remains failing; its decisions and maximum error match the baseline.
- Firefox passes seven forced-FP32 complete-array comparisons with zero difference
  and 30 CPU-reference checks. A separate run checks FP32 alongside unchanged
  native FP16, passing 60 CPU-reference checks and the existing cross-precision
  bounds. No whole-model Firefox speedup is claimed.
- 64 Rust tests, 103 JavaScript tests, 76 JavaScript syntax checks, all three WASM
  validations, and site staging pass. Independent review found no correctness issue.

Generated WASM remains an untracked build artifact. This optimization follows the
[32×8 workgroup mapping](matmul-32x8-performance.md),
[row-padding selection](fp32-row-padding-performance.md), and
[explicit output stores](fp32-explicit-stores-performance.md).
