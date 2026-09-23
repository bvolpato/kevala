# Shared GPU kernels and Gemma execution

Measured on 2026-09-23 against main at `8ee6ac6`. These results cover one RTX 5070 Ti
workstation. They establish gains on the measured workloads, not a universal performance ranking.
[Measurements and correctness results](benchmarks/gpu-latency-linux-2026-09-23.json)
include the individual samples and all nine model guards.

## What changed

- **Shared matrix multiplication:** explicit inner-loop unrolling and named FP32 accumulators
  improve compiler output. The generic FP32 kernel uses this for four-row, single-group tiles;
  the wide kernel uses it for four-row FP16 staging. Shorter tiles retain their original loops.
  The Q8 weights, arithmetic order, split reduction, and epilogues are unchanged. This applies
  to Laya, Kev, SemIf, and Gemma. The wide staging layout from PR #21 is already in the baseline.
- **Qwen DeltaNet recurrence:** named scalar state removes dynamic private-array indexing.
  The runtime compares four, eight, and sixteen cooperating lanes on the loaded model's head
  dimensions. A candidate needs a stable 5% win at both 128 and 512 tokens. Four lanes remains
  the fallback when calibration is unavailable or fails. Candidate outputs and saved states
  must also pass a numerical comparison against the compiled four-lane kernel on the actual
  adapter before timings can select them. This applies to Kev and SemIf.
- **Gemma shared-KV layers:** the final layers reuse K/V produced by earlier layers. Their
  attention queries, projections, and feed-forward operations can therefore run only for the
  requested output rows. K/V retain the full sequence, and attention uses the original token
  positions. PLE uses the original embeddings and token IDs. Row order, duplicates, empty
  selections, and models without a shared-KV tail remain supported.

The selected-row change removes work from Gemma E2B's 20 shared layers and E4B's 18 shared
layers. The other kernel changes operate across supported architectures or Qwen head shapes.

## Measurement

Ubuntu, Ryzen 9 9950X3D, RTX 5070 Ti 16 GB, NVIDIA driver 595.71.05. GPU jobs ran serially
with the NVIDIA Vulkan ICD and a hardware-adapter check. Chrome 149.0.7827.200 ran headless
in an isolated CDP session with precise developer timestamps. Firefox 152.0.3 used an isolated
headless profile. Chrome exposed subgroups but no `shader-f16` on this driver; Firefox exposed
`shader-f16` but did not use subgroups. Feature support can differ on other installations.

Each family has three input sizes, two warmups, and five measured requests per size with
unique inputs. The metric is the geometric mean of per-case medians of summed GPU kernel
milliseconds. It excludes loading, CPU preparation, queue waiting, and readback. The families
use different workloads, so this is not a model quality or equal-task speed comparison.

| Family | Input sizes, tokens |
| --- | --- |
| Laya | 47 / 140 / 512 |
| Kev 0.8B | 31 / 124 / 532 |
| SemIf 2B | 105 / 198 / 606 |
| Gemma E2B | 99 / 193 / 601 |

| Browser | Model | Main | Retained changes | Time reduction |
| --- | --- | ---: | ---: | ---: |
| Chrome | Laya | 13.459 ms | 12.947 ms | 3.8% |
| Chrome | Kev 0.8B | 20.170 ms | 17.979 ms | 10.9% |
| Chrome | SemIf 2B | 67.965 ms | 63.086 ms | 7.2% |
| Chrome | Gemma E2B | 89.188 ms | 40.457 ms | 54.6% |
| Firefox | Laya | 15.077 ms | 14.487 ms | 3.9% |
| Firefox | Kev 0.8B | 24.367 ms | 19.885 ms | 18.4% |
| Firefox | SemIf 2B | 80.322 ms | 68.806 ms | 14.3% |
| Firefox | Gemma E2B | 104.250 ms | 48.204 ms | 53.8% |

The four-family geometric mean falls from 35.816 to 27.763 ms in Chrome and from 41.880 to
31.265 ms in Firefox. These suites measured local experiment revision `1b1ba64`, including
runtime numerical guards. Later validation at `d967bd8` removes redundant calibration queue
waits; inference kernels are unchanged. The preceding run is retained in the JSON to show run
variation. Load-time tuning and GPU clocks cause variation, especially for short requests. The isolated matrix and recurrence comparisons
provide stronger evidence for their individual kernel changes than small full-suite differences.

Round-trip samples in the JSON come from these **profiled requests**, including timestamp
instrumentation overhead. They are not measurements of unprofiled serving latency. For the
longest Gemma case, Chrome's profiled p50 falls from 224.4 to 87.6 ms. Laya's corresponding
wall p50 moves from 38.4 to 38.3 ms. Firefox wall times show coarse
100 ms steps. A GPU-time gain does not imply the same round-trip gain.

## Correctness

Weights and existing probability tolerances are unchanged. All nine supported model sizes
were checked in Chrome and Firefox, with **132/132 reference decisions matching in each**.
Firefox passes all nine probability thresholds. Chrome passes eight:

- Chrome Laya's maximum probability difference is `0.024006444195624588`, just above the
  existing `0.024` threshold. Main produces the exact same value. This remains a strict
  baseline failure and is not reported as a passing threshold check.
- Firefox Laya is within its threshold at `0.023506444195624587`.
- All eight non-Laya sizes are within their existing model-specific limits in both browsers.

The matrix unrolls produce bit-for-bit identical full output arrays in paired tests. Independent
CPU checks cover full small matrices and sampled large coordinates, including bias, residual,
ReLU, split reduction, and partial tiles. Recurrence checks compare all output and cache-state
values against an independent CPU implementation, including grouped heads, cached prefixes,
empty segments, and long sequences. Eight/sixteen lanes can change floating-point reduction
association; their errors remain within the existing guards.

The synthetic Gemma check compares every requested hidden-state element with the full-row
execution graph. Both browsers produce bit-for-bit identical results for lengths 1, 31, and 65,
including reordered and duplicate rows, empty selection, growth, and zero-sharing fallback.
Real E2B/E4B golden checks supplement this graph comparison. Kev cache hit and extension
checks have zero probability differences from their matching fresh computations.

Local validation passed 102 JavaScript tests, 64 release Rust workspace tests with all pinned
tokenizers present, four WGSL rendering tests, JavaScript syntax checks, all three WASM builds
and ABI checks, and site staging. Hardware GPU tests validate shader compilation and execution;
WGSL rendering tests alone do not.

## Rejected experiments

- A 128-row matrix tile regressed short shapes by 45–66%.
- FP32 wide-kernel selection and packed-half staging did not establish enough gain for their
  added dispatch complexity.
- Thirty-two recurrence lanes were slower than sixteen, especially on Firefox. Sixteen lanes
  also regress some larger Firefox head shapes, which is why recurrence selection is measured.
- A 32-query Laya attention tile improved long shapes but slowed shorter shapes by 14–26%.
- Gemma subgroup reductions produced identical outputs in twelve paired cases but slowed the
  601-token, 512-dimensional global-attention case by 6.6%.
- A two-plane packed-integer matrix path improved larger shapes by 24–38%, including
  activation conversion, but failed the unchanged FP32 error limits on outliers, cancellation,
  and some large outputs. It was rejected for correctness.
- Adding a third activation plane passed the unchanged numerical limits, but total matrix
  time increased by 25.6% across six shapes. It was rejected for performance.

Rejected kernels are excluded from the production change.
