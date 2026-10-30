# Projection runtime validation, October 30, 2026

The release check exercised all 11 catalog models with local Q8 packs on Firefox WebGPU, plus the available smaller BF16 variants. It checked choice, noul and score responses, normalized probabilities, token usage, repeated calls and batched calls. These are runtime and regression checks, not model accuracy results.

[Recorded identities, runtime hashes, timings, numerical guards and limitations](benchmarks/projection-runtime-2026-10-30.json) identify the tested build. Rust 1.95.0 and Node 24.21.0 built and validated all three WASM flavors. JavaScript and Rust tests, Clippy, CPU numerical guards, the package check and static-site staging passed. Unsupported subgroup checks are explicit skips.

## Model coverage

| Model | Precision | Longest tested request, input tokens |
| --- | --- | ---: |
| `laya` | Q8 | 706 |
| `kev-0.8b` | Q8 | 268 |
| `kev-0.8b` | BF16 | 268 |
| `kev-4b` | Q8 | 268 |
| `kev-4b` | BF16 | 268 |
| `kev-9b` | Q8 | 268 |
| `h2o-lightning-4b` | Q8 | 369 |
| `h2o-lightning-4b` | BF16 | 369 |
| `clef-flash` | Q8 | 406 |
| `semif-qwen3.5-0.8b` | Q8 | 430 |
| `semif-qwen3.5-2b` | Q8 | 430 |
| `semif-qwen3.5-4b` | Q8 | 430 |
| `gemma-4-e2b` | Q8 | 891 |
| `gemma-4-e4b` | Q8 | 891 |

The machine also had an unrelated GPU workload, which remained running. Kev 9B BF16 needs 13,853,820,928 bytes for its persistent GPU trunk before scratch; it was not executed with about 11 GiB of free NVIDIA VRAM. Its Q8 variant passed. Clef passed with the bounded paragraph profile `1,2,3`; the larger profile exhausted available NVIDIA memory. An attempted AMD full-model Clef run exited its browser and is not counted as a pass.

## Performance scope

The loaded Kev 0.8B BF16 calibration selected faster single-row matrix-vector kernels after full-output numerical checks and alternating GPU timestamp measurements. These kernel results do not establish an ordinary-request latency gain.

Clef's paired check used one loaded model and alternated the old handoff with resident embeddings and cached lexical means. Hidden values, both WASM head inputs and public answers matched exactly. The JSON records the per-request medians and aggregate reduction. Warm resident requests performed no embedding or lexical readback.

The Gemma E2B image check compared complete hidden vectors and raw probabilities with the grid fusion off/on/off at suffix lengths 96, 98, 112 and 113. The numerical guards passed; the timing gates retained the baseline in this run. A repeatable camera FPS gain is not established.

## Batch compatibility and reproduction

In a matched baseline comparison, SemIf 0.8B Q8 produced the same single-request and batch raw vectors before and after these changes. The unchanged baseline also produced a 0.0011942386627197266 single-versus-batch difference because batching changes split-K reduction grouping. The smoke harness keeps repeated-call checks at 0.0005. Its default batch API compatibility bound is the larger of 0.0005 and half a rounded answer unit; an explicit override is recorded with each run. This is not a raw numerical-equivalence guarantee.

Serve the checkout, then run one model at a time:

```sh
pnpm serve --port 18137
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --browser firefox --timeout 660 \
  --url 'http://127.0.0.1:18137/dev/model-smoke-check.html?auto=1&model=kev-0.8b&pack=/tmp/kev-0.8b-bf16.kevala&precision=bf16&samples=3&warmups=1' \
  --output /tmp/kev-0.8b-bf16-smoke.json
```

The GPU runner and memory requirements are Linux-specific here. Other GPUs and browsers need their own measurements. Larger unpublished models still require a converted pack URL; this deployment does not publish model weights.
