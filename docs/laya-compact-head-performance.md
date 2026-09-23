# Laya compact final-head performance

Laya's final head now gathers the requested rows before its last row-local normalization and feed-forward network. The two projections process only those rows, and the residual is gathered with them. Earlier attention and head layers still process the full sequence.

This host-only change applies to FP32, single-group generic matmuls with 1–16 requested rows, fewer than the sequence length. It uses the existing R1 shaders with a virtual row count that preserves the original K partitions and keeps matrix and reducer partial-buffer strides consistent. Native FP16, other kernel selections, requests for more than 16 rows, full-row requests, and models without head layers keep the original path. No shader, weight, or tolerance changes are included. The runtime diff is 36 added and 11 removed lines in `js/src/gpu.js`.

## Production-build timings

The baseline is merged revision `90abb529`; the candidate adds the compact-head host change and its guard page. Both use identical WASM files. These measurements used local served builds on Ubuntu, an NVIDIA RTX 5070 Ti with driver 595.71.05, and a Ryzen 9 9950X3D. Chrome 149.0.7827.200 used the NVIDIA Vulkan ICD and the FP32 generic path because `shader-f16` was unavailable.

Each run measured five requests after two warmups at 47, 140, and 512 tokens. The metric is the geometric mean of the three per-size median summed GPU kernel times. It excludes loading and initialization and is not wall-clock decision latency. GPU jobs ran serially in baseline/candidate order, followed by a second baseline/candidate pair.

| Run | Baseline GPU ms | Candidate GPU ms |
| --- | ---: | ---: |
| Pair 1 | 10.531708 | 10.315750 |
| Pair 2 | 10.309252 | 9.979429 |
| Geometric mean of both runs | **10.419886** | **10.146196** |

The combined result is **2.63% lower GPU time**. Its aggregate remains above 10 ms. These are local production-build measurements, not deployed-site timings or evidence of a sub-10 ms production aggregate.

| Tokens | Baseline GPU ms, combined | Candidate GPU ms, combined | Change |
| --- | ---: | ---: | ---: |
| 47 | 4.448279 | 4.315228 | 2.99% lower |
| 140 | 9.632455 | 9.339001 | 3.05% lower |
| 512 | 26.403407 | 25.918244 | 1.84% lower |

A Firefox 152.0.3 timing pair exercised the unchanged native-FP16 path: 14.493473 ms baseline and 14.232609 ms candidate. Its 1.80% difference is a control observation and is not attributed to compact-head execution.

## Correctness and validation

The synthetic guard explicitly requests FP32 on both Chrome and Firefox. All **12 cases per browser** passed, with **zero hidden-output difference** for every recorded comparison. The guard uses hidden width 1024 and intermediate width 4096 and checks original row IDs, reordered and duplicate rows, multiple segments, 16-row selection, the 321-token split boundary, and full-row, 17-row, and zero-head-layer fallbacks. Recorded dispatches retained the original split counts. The existing hidden-output limit remains `2e-5`.

Chrome matched all **41 reference decisions**. Its candidate and baseline had exactly equal **126 probability values**, using the already-recorded PR31 result as the baseline. Both have `maxDp = 0.024006444195624588`, above the existing strict `0.024` limit. This pre-existing miss remains a failure; it was not relaxed. The Firefox native-FP16 model guard passed **41/41 decisions**, with `maxDp = 0.023506444195624587` under the same limit.

The final default `pnpm test` run passed **105/105** tests. An earlier run passed 104/105 and failed the offline reporter's committed-summary check; its first targeted retry also failed. All 23 reporter, fixture, and result files and the pinned Node binary were identical to PR31. Later default retries passed. This is recorded as intermittent execution sensitivity, not a proven V8 defect. The diagnostic `--jitless` run passed the reporter checks but disabled WebAssembly; `NODE_OPTIONS=--no-opt` was rejected before tests. Syntax checks, all three WASM checks, and site staging passed. No unrelated reporter changes were made.

## Reproduction and evidence

Serve the baseline at port 18122 and candidate at 18121, both with CORS, with identical WASM files and the local Laya pack. The commands below assume Chrome is already available through DevTools at port 9333. Run the timing command once per build and repeat the pair.

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py --result bench --cdp http://127.0.0.1:9333 --timeout 900 \
  --url 'http://127.0.0.1:18121/bench.html#auto&backend=webgpu&pack=local&model=laya&profile=1&runs=5&warmups=2&unique=1&shapes=0,1,2' \
  --output /tmp/laya-compact-timing.json

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py --result gpu --cdp http://127.0.0.1:9333 --timeout 900 \
  --url 'http://127.0.0.1:18121/dev/laya-compact-head-check.html?baseline=http://127.0.0.1:18122/js/src/gpu.js' \
  --output /tmp/laya-compact-guard.json

# Reproduces the unchanged Chrome strict-limit miss described above.
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py --result parity --cdp http://127.0.0.1:9333 --timeout 900 --max-dp 0.024 \
  --url 'http://127.0.0.1:18121/parity.html#auto&backend=webgpu&pack=local' \
  --output /tmp/laya-compact-parity.json
```

For Firefox, replace `--cdp http://127.0.0.1:9333` with `--browser firefox`. The [machine-readable artifact](benchmarks/laya-compact-head-linux-2026-09-23.json) contains all six timing runs, both synthetic guards, three model-parity records, source and artifact hashes, and the preserved local test diagnostics.
