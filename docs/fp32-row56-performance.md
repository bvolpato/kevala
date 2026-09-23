# FP32 BM56 row tile performance

The FP32 generic matrix path now has an optional physical 56-row tile for R4, single-column-group matmuls. It keeps the virtual 64-row workgroup grid, split rule, reducer, shared-memory allocation, and arithmetic order. The runtime selects it only when `ceil(T / 56) == ceil(T / 64)`. R3 remains preferred where it already applies. FP16, grouped matmuls, and fused matmul kernels keep their existing paths.

The final production-WASM microbenchmark covered 13 rendered shapes. Its geometric mean of per-shape median GPU time fell from **0.067688 ms** with generic BM64 to **0.061934 ms** with BM56, a **1.093x** speedup. Every shape was faster. This is a kernel measurement, not an end-to-end application speedup.

## Production-shaped model suite

The suite used local served builds on an NVIDIA RTX 5070 Ti (16 GB, driver 595.71.05), Ryzen 9 9950X3D, Ubuntu, and Chrome 149.0.7827.200 through the NVIDIA Vulkan ICD. The baseline checkout `b80ea44` has the same tree as merged revision `94303e47`; the candidate adds six BM56 source changes to that tree. No paired-projection fusion is included. Each model ran five measured requests per input size after two warmups. The reported family metric is the geometric mean of the three per-size median summed GPU kernel times. Both complete runs per version are combined geometrically.

| Model family | Baseline GPU ms | Candidate GPU ms | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| SemIf Qwen3.5 2B | 52.893 | 51.259 | 3.09% lower | Eligible affected matmuls |
| Gemma 4 E2B | 34.373 | 34.039 | 0.97% lower | Overlaps baseline run variation; no reliable whole-model gain claim |
| Laya | 10.258 | 10.242 | 0.16% lower | Control variation; not attributed |
| Kev 0.8B | 14.893 | 15.051 | 1.06% higher | Control variation; not attributed |
| Four-family geometric mean | 22.957 | 22.773 | 0.80% lower | Descriptive only; not attributed |

The SemIf Qwen3.5 2B result is an eligible affected-matmul gain. Gemma 4 E2B is 0.97% lower, which overlaps the observed baseline repeat variation, so it does not support a reliable whole-model gain claim. The small Laya and Kev changes are control variation. The four-family aggregate is descriptive and is not presented as a product-level gain. These were local served builds, not a deployed-site measurement.

## Isolated matmul measurements

All measurements used Chrome WebGPU timestamp queries, 15 timed samples, and three warmups per case. The two prototype runs preserved their distinct measurement order in the raw record. The final row is the direct production-WASM renderer harness from the candidate at `94303e47` plus the six BM56 source changes.

| Run | Shapes | Generic BM64 geometric mean | BM56 geometric mean | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Initial prototype order | 8 | 0.262024 ms | 0.244172 ms | 1.073x |
| Prototype with reversed order | 13 | 0.066111 ms | 0.061283 ms | 1.079x |
| Final production-WASM rendered harness | 13 | 0.067688 ms | 0.061934 ms | 1.093x |

The final harness rendered both WGSL variants from the production candidate WASM. It measured the eight model-derived matmul shapes plus five small row-boundary shapes. All 13 BM56 medians beat their paired BM64 medians. Across its 38 whole-output comparisons, BM56 was bit-identical to BM64; all 76 CPU-reference checks passed.

## Model guards

The 18 Chrome production guard runs compared baseline and candidate on nine model fixtures. All **132/132 reference decisions** matched in both versions. Every baseline/candidate decision vector matched, and all **351 recorded probability values** were exactly equal between versions. Each model's `maxDp` was also unchanged. The limits below are the existing per-model limits; none were widened.

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

Laya's Chrome `maxDp` is above its strict limit in both versions by `0.0000064442`; this pre-existing miss is not reported as a pass. A separate Firefox Gemma 4 E2B native-FP16 spot guard passed 12/12 at `0.01969242` against its `0.030` limit. It does not exercise BM56. Its first attempted run used an invalid `golden` query key and was rejected; the corrected `reference` run is the reported result.

A separate Chrome Kev 0.8B cache check verified counters and had zero error for hit versus miss and extension versus fresh probabilities. It is a regression check, not a BM56 performance result.

One paired Firefox full-suite regression check exercised the unchanged native-FP16 path. Its four-family GPU aggregate was 30.867 ms for the baseline and 31.125 ms for the candidate; it is not a BM56 speed claim. Laya and Kev were 2.62% and 2.49% slower, while SemIf 2B and Gemma E2B were 0.78% and 0.94% faster. The initial Firefox Gemma 4 E2B guard attempt used an invalid `golden` query key and was rejected; the corrected `reference` run is reported above and the rejected raw result is preserved in the artifact.

## Reproduction

These commands assume the corresponding local build is served at the stated port, the model packs are available under `tmp/`, and Chrome is already running with DevTools at `127.0.0.1:9333`. They reuse that Chrome configuration and explicitly select the NVIDIA Vulkan ICD.

```sh
# Final matrix harness: candidate server only; the baseline does not contain BM56.
CASES='99x6144x1536,99x1536x6144,193x6144x1536,193x1536x6144,105x12288x2048,105x2048x6144,198x12288x2048,198x2048x6144,53x70x96,55x70x160,56x70x96,111x70x1024,112x70x160'
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py --result gpu --cdp http://127.0.0.1:9333 --timeout 900 \
  --url "http://127.0.0.1:18117/dev/gpu-bench.html?cases=$CASES&kernels=matmul_row56@256:4/1,matmul@256:4/1&samples=15&warmups=3&sourceCommit=94303e47-plus-row56" \
  --output /tmp/row56-rendered-guards.json

# Production-shaped suite: repeat each command once with an output directory ending in -repeat.
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18114/ /tmp/row56-production-base
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18117/ /tmp/row56-production-candidate

# Existing Laya strict probability guard: candidate server. This reproduces the known
# unchanged maxDp of 0.02400644, above the strict 0.024 threshold.
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py --result parity --cdp http://127.0.0.1:9333 --timeout 900 --max-dp 0.024 \
  --url 'http://127.0.0.1:18117/parity.html#auto&backend=webgpu&pack=local' \
  --output /tmp/row56-candidate-laya-parity.json
```

The [machine-readable artifact](benchmarks/fp32-row56-linux-2026-09-23.json) includes the raw Chrome full-suite runs, two prototype and two rendered microbenchmark records, all 18 Chrome model-guard files, the Firefox full-suite pair, and supplemental Firefox and cache checks.
