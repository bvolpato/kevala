# Packed vector loads for recurrent state

The subgroup-enabled `kev_recur_lanes16` specialization now reads its Q and K projection data as `vec4<f32>` values. Each lane loads two packed vectors for Q and two for K, then uses the existing scalar arithmetic and subgroup reductions. The `SUBGROUPS=0` specialization retains the previous scalar storage binding and shared fallback. Runtime dispatch and recurrence autotuning are unchanged; lane widths 4, 8, and 16 are still measured with numerical guards, and lane 4 remains the fallback.

The candidate uses merged main `eb67049b1cc221b9f48ef10248ed2c26082acc15` plus this shader change. The benchmark harness was already on main. No paired-projection fusion is included.

## Production-shaped Chrome suite

The table combines both complete runs per version geometrically. Each family value is the geometric mean of the median summed GPU kernel times at three input sizes, with two warmups and five measured requests per size. The four-family row is the geometric mean of the four family metrics. These are GPU profile measurements, not per-request or wall-time guarantees.

| Model family | Baseline GPU ms | Candidate GPU ms | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Laya | 9.866 | 9.789 | 0.78% lower | Control variation; not attributed |
| Kev 0.8B | 14.073 | 13.875 | 1.41% lower | Affected recurrence path |
| SemIf Qwen3.5 2B | 49.075 | 48.816 | 0.53% lower | Affected recurrence path |
| Gemma 4 E2B | 32.197 | 32.530 | 1.04% higher | Control variation; not attributed |
| Four-family geometric mean | 21.642 | 21.550 | 0.42% lower | Descriptive only; not attributed |

The small whole-model changes do not establish a broad end-to-end gain. The Laya and Gemma families are controls for this recurrence specialization, and the four-family mean combines affected and control paths.

The production suites ran on one Ubuntu workstation with an NVIDIA RTX 5070 Ti (16 GB), NVIDIA driver 595.71.05, AMD Ryzen 9 9950X3D, and Chrome 149.0.7827.200 using WebGPU through the NVIDIA Vulkan ICD. Results describe this machine, browser, and workload; they are not a universal ranking or a deployed-site measurement.

## Recurrence time in model profiles

The production request profiles expose the `gpu.recur` component for each of five measured requests at each input size. The table takes the median per size and then geometrically combines both complete runs. The final row for each model is the geometric mean across its three input sizes and both runs.

| Model | Tokens | Baseline recurrence ms | Packed recurrence ms | Change |
| --- | ---: | ---: | ---: | ---: |
| Kev 0.8B | 31 | 0.426 | 0.347 | 18.52% lower |
| Kev 0.8B | 124 | 1.609 | 1.316 | 18.21% lower |
| Kev 0.8B | 532 | 6.373 | 5.211 | 18.24% lower |
| **Kev 0.8B, three-size geometric mean** | n/a | **1.635** | **1.336** | **18.32% lower** |
| SemIf Qwen3.5 2B | 105 | 1.404 | 1.156 | 17.69% lower |
| SemIf Qwen3.5 2B | 198 | 2.494 | 2.074 | 16.83% lower |
| SemIf Qwen3.5 2B | 606 | 7.341 | 5.869 | 20.06% lower |
| **SemIf Qwen3.5 2B, three-size geometric mean** | n/a | **2.952** | **2.414** | **18.20% lower** |

These profile components show the direct production path in its model workloads. The smaller whole-model movement reflects work outside recurrence and variation in the other measured components; profile time should not be read as request latency.

## Rendered kernel checks

The final production-WASM Chrome harness measured ten cases with ten samples and three warmups in reversed order. CORE and STATE outputs matched exactly between baseline and candidate in all ten cases; the CPU-reference checks had maximum absolute error `3.7253e-9`.

| Value heads (`LIN_HEADS`) | Cases | Paired speedup range | Geometric mean speedup |
| --- | ---: | ---: | ---: |
| 16 | 3 | 1.186–1.245x | 1.221x |
| 32 | 3 | 1.437–1.471x | 1.452x |
| 48 long-stage case | 1 | 1.535x | 1.535x |

The head dimension is 128; the recurrence lane width remains 16.

A separate four-case run compiled `SUBGROUPS=0`. It checked exact CORE/STATE output on the unchanged fallback; its timings were flat and are not a performance claim. The cache regression check verified counters and exact probabilities, with zero error for hit versus miss and extension versus fresh inputs.

## Chrome model guards

All nine production fixtures matched the baseline: **132/132 decisions**, **351/351 probability values**, full per-question details, and `maxDp` values were exact. Existing probability limits were retained. Chrome Laya remains at `maxDp=0.024006444195624588`, above its strict `0.024` threshold in both versions; this known miss is not reported as a pass.

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

## Firefox regression checks

Firefox 152.0.3 used the unchanged scalar fallback. Its single-pair four-family aggregate was 29.972 ms at baseline and 30.141 ms for the candidate. This is a cross-browser regression check, not a speed claim for the packed subgroup path. The Firefox SemIf Qwen3.5 2B spot guard passed 12/12 at `maxDp=0.02859807` against the existing `0.032` limit.

## Reproduction

With the merged-main baseline and candidate builds served at ports 18121 and 18125, run both full suites twice. The raw artifact records all suite inputs, hashes, model guards, and direct/shared rendered checks.

```sh
node scripts/serve.mjs . --port=18125 --cors

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18121/ /tmp/recur-packed-production-base
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18125/ /tmp/recur-packed-production-candidate
```

The [machine-readable artifact](benchmarks/recurrence-packed-linux-2026-09-23.json) embeds the raw production evidence with SHA-256 hashes.
