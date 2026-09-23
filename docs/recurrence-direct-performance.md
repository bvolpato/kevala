# Subgroup direct-load recurrence performance

The `kev_recur_lanes16` kernel now loads each lane's eight Q and K values directly from `C` as named scalar values when compiled with `SUBGROUPS=1`. The existing subgroup reductions and state layout remain in place. With `SUBGROUPS=0`, the kernel keeps the shared-memory `qs`/`ks` staging path; the rendered fallback WGSL is byte-identical to the baseline.

The runtime dispatcher and recurrence tuner are unchanged. They still benchmark lane widths 4, 8, and 16 at 128 and 512 tokens, apply the existing CORE/STATE numerical guard, and select a wider lane only after a stable 5% win at both sizes. Lane 4 remains the fallback. This change adds a specialization to the existing 16-lane kernel; it does not force the runtime to select it.

## Production-shaped Chrome suite

The baseline is checkout `90abb5297ddd3a9e4d444f1a435c0d7605409374`. The candidate uses that same base plus the recurrence shader and rendered-check harness changes. The suite ran local production-WASM builds on an RTX 5070 Ti (16 GB, driver 595.71.05), Ryzen 9 9950X3D, Ubuntu, and Chrome 149.0.7827.200 through the NVIDIA Vulkan ICD. Each model metric is the geometric mean of its three input-size median summed GPU kernel times, with five measured requests after two warmups. Both complete runs per version are combined geometrically.

| Model family | Baseline GPU ms | Candidate GPU ms | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Kev 0.8B | 14.571 | 14.376 | 1.34% lower | Eligible affected family |
| SemIf Qwen3.5 2B | 49.971 | 49.585 | 0.77% lower | Eligible affected family |
| Laya | 10.054 | 10.180 | 1.25% higher | Control variation; not attributed |
| Gemma 4 E2B | 33.435 | 32.845 | 1.76% lower | Control variation; not attributed |
| Four-family geometric mean | 22.243 | 22.095 | 0.66% lower | Descriptive only; not attributed |

The Kev and SemIf results are the eligible whole-model measurements for this recurrence change. These small differences vary between runs; the isolated kernel measurements below provide clearer performance evidence. The Laya and Gemma rows are controls. The four-family aggregate combines affected and control models, so it is not presented as a product-wide gain. These are local served-build measurements, not a deployed-site measurement.

## Rendered recurrence measurements

The production-WASM Chrome harness compared the candidate to a baseline WASM built from the same `90abb529` source. It measured ten cases with ten samples and three warmups in reversed order. The six model-derived 16- and 32-head cases and the 48-head long-stage case measured between **1.076x and 1.114x** faster in the direct-load specialization. CORE and STATE outputs were exact between baseline and candidate across all ten cases. The CPU-reference checks passed with a maximum absolute difference of `3.73e-9`.

A separate four-case run compiled the shared fallback with `SUBGROUPS=0`. It verified exact CORE/STATE output and byte-identical rendered WGSL. Its timings were flat and are only a fallback regression check.

## Model guards

Nine Chrome production fixtures compared the candidate with the production baseline. All **132/132 decisions** and all **351 probability values** matched exactly. The complete per-question details and `maxDp` values also matched. The baseline records were captured for the preceding BM56 candidate, whose source tree is the same `90abb529` baseline; their raw filenames retain that earlier experiment's prefix.

| Model | Decisions | Baseline = candidate maxDp | Existing limit | Result |
| --- | ---: | ---: | ---: | --- |
| Gemma 4 E2B | 12/12 | 0.01911902 | 0.030 | Pass |
| Gemma 4 E4B | 12/12 | 0.00992773 | 0.030 | Pass |
| Kev 0.8B | 13/13 | 0.00968474 | 0.011 | Pass |
| Kev 4B | 13/13 | 0.00996225 | 0.011 | Pass |
| Kev 9B | 5/5 | 0.00463277 | 0.005 | Pass |
| Laya | 41/41 | 0.02400644 | 0.024 | Existing strict-limit miss |
| SemIf Qwen3.5 0.8B | 12/12 | 0.03081912 | 0.032 | Pass |
| SemIf Qwen3.5 2B | 12/12 | 0.02975827 | 0.032 | Pass |
| SemIf Qwen3.5 4B | 12/12 | 0.02048463 | 0.022 | Pass |

Chrome Laya's `maxDp` is `0.024006444195624588` in both versions, above its strict `0.024` limit. This unchanged miss is not reported as a pass, and the limit was not widened. A Firefox SemIf Qwen3.5 2B spot guard passed 12/12 at `0.02859807` against `0.032`.

The paired Firefox full-suite check measured a four-family aggregate of 31.583 ms for the baseline and 30.502 ms for the candidate. Treat this single-pair difference as browser/control variation; it is not a speed claim. The Kev cache regression check verified its counters and exact probabilities for cache-hit versus miss and extension versus fresh inputs, with zero error.

## Reproduction

Serve the production candidate build and baseline build at the local URLs below. The full-suite script records three input sizes per fixture; run each version twice and combine both complete runs geometrically as in the report.

```sh
node scripts/serve.mjs . --port=18119 --cors

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18117/ /tmp/recur-direct-production-base
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18119/ /tmp/recur-direct-production-candidate
```

The [machine-readable artifact](benchmarks/recurrence-direct-linux-2026-09-23.json) embeds the exact Chrome and Firefox suite files, both sets of nine model guards, the production-rendered direct and shared-path checks, and the cache and Firefox spot checks with SHA-256 hashes.
