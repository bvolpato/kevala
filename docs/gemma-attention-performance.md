# Gemma 4 attention performance

The tiled attention kernel reduces Gemma E2B GPU time by **31–33%** and E4B GPU time
by **23.2%** across three tested input sizes. These are Firefox WebGPU measurements
on an RTX 5070 Ti, not a claim of state-of-the-art performance or equivalent gains
on every device. Laya, Kev, and SemIf use other attention kernels.

The [measurement record](benchmarks/gemma-attention-linux-2026-09-23.json) includes
per-request GPU timings, numerical guards, all 1,728 decision comparisons, and the
rejected candidates. Measurements were taken on September 23, 2026 UTC.

## What changed

The old Gemma kernel reduced one key at a time with all 256 workgroup lanes. It
repeated workgroup barriers and online-softmax rescaling for every key.

The new kernel assigns eight lanes to each key and processes 32 keys concurrently.
It combines four such groups into a 128-key tile, updates the online softmax once
per tile, and keeps output accumulators in registers. Four-component dot products
serve dimensions divisible by four; a scalar fallback handles other supported
dimensions. It retains FP32 arithmetic, causal and sliding-window masks, grouped
query heads, and the existing 256/512-dimensional Gemma heads. It needs no new
WebGPU feature and uses 3,600 bytes of workgroup storage.

The implementation draws on tiled attention and reduction patterns described by
[FlashInfer](https://docs.flashinfer.ai/tutorials/kv_layout.html) and
[vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/).
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) informed the
matrix experiments. This is original WGSL code. Native CUDA tensor-core and
memory-transfer primitives do not transfer directly to this browser implementation.

## Full-model GPU timings

Environment: Ubuntu, Firefox 152.0.3, RTX 5070 Ti 16 GB, NVIDIA driver 595.71.05,
Ryzen 9 9950X3D. GPU jobs ran serially with the NVIDIA Vulkan ICD selected explicitly.
The runner rejects software adapters. Both sizes use the existing Q8 packs and
automatic matrix tuning, with unique inputs and two warmups per case.

The metric is the geometric mean of three case medians. Each case median comes
from the sum of GPU kernel timestamps for each request. E2B uses seven measured
requests per case and E4B uses five. Loading, CPU preparation, browser waits, and
readback are excluded. Actual token counts are 99, 193, and 601; the benchmark's
approximate display labels do not include all prompt tokens.

| Model | Original GPU aggregate | Retained kernel | Reduction |
|---|---:|---:|---:|
| Gemma E2B | 158.31 ms | 106.17 ms | 32.94% |
| Gemma E4B | 265.36 ms | 203.92 ms | 23.15% |

The E2B baseline repeated with five measured requests per case at 153.90 ms,
2.79% below the initial run. The retained
kernel is 31.01% faster against that repeat. The baseline source is commit
`7c49c6fac3a685527b584c58d98b22af4d7159c5`; the evidence records the retained shader hash.

| Model | Tokens | Total GPU before → after | Attention before → after |
|---|---:|---:|---:|
| E2B | 99 | 61.93 → 51.14 ms | 10.20 → 2.06 ms |
| E2B | 193 | 125.91 → 92.96 ms | 34.75 → 5.82 ms |
| E2B | 601 | 508.81 → 251.74 ms | 296.72 → 44.97 ms |
| E4B | 99 | 109.57 → 97.46 ms | 12.18 → 2.51 ms |
| E4B | 193 | 221.94 → 184.47 ms | 42.31 → 7.24 ms |
| E4B | 601 | 768.36 → 471.70 ms | 351.94 → 53.08 ms |

On the integrated AMD GPU with RADV Mesa 25.2.8, the three-case attention
microbenchmark fell from 22.87 to 11.84 ms, a **48.24% reduction**. This is a kernel
microbenchmark, not a full-model latency result. Full-model AMD correctness was
checked separately below.

## Round-trip latency

With profiling disabled, E2B round-trip medians changed as follows:

| Tokens | Before | After |
|---:|---:|---:|
| 99 | 100.08 ms | 100.06 ms |
| 193 | 200.12 ms | 100.08 ms |
| 601 | 600.42 ms | 600.40 ms |

The geometric mean falls from 229.10 to 181.84 ms, or 20.63%. The longest request
remains near 600 ms despite roughly halving its GPU work. Browser completions
arrived in approximately 100 ms steps. Without profiling, the existing runtime
splits inputs above 256 tokens into six chunks and awaits GPU completion between
chunks. These observations suggest synchronization limits; they do not isolate
the browser or driver cause. Profiling disables those chunks. This change leaves
submission policy intact, and no E4B wall-time improvement is claimed.

## Correctness

The independent harness computes float64 CPU dot products and softmax probabilities.
It checks every output for finite values. Small cases compare every output;
long cases compare all dimensions at sampled query/head pairs. The threshold is
`1e-5 + 1e-5 * max(1, abs(reference))` per checked value.

Checks cover both masks, grouped heads, partial tiles, odd dimensions, peaked
softmax inputs, window boundaries, and a 4,096-token global-attention case. The
final long-case sweep has maximum absolute error **5.65e-6**. Earlier peaked-input
guards and their exact errors are also recorded in the evidence.

Both sizes preserve every selection from the previously published Q8 decision
benchmark. The 864 rows per model include three option orders per source case, so
they are not 864 independent examples. Floating-point reduction order changes
probabilities; the values are not bit-identical.

| Model | Same Q8 selections | Correct before / after | Maximum probability change |
|---|---:|---:|---:|
| E2B | 864/864 | 671 / 671 | 0.04610 |
| E4B | 864/864 | 785 / 785 | 0.00909 |

Original-weight fixtures are a separate comparison:

| Runtime / model | Matching selections | Maximum probability difference |
|---|---:|---:|
| NVIDIA E2B | 12/12 | 0.01999 |
| NVIDIA E4B | 12/12 | 0.00921 |
| AMD E2B | 12/12 | 0.01879 |

The unchanged-family NVIDIA regression checks pass Laya 41/41, Kev 0.8B 13/13,
and SemIf 2B 12/12 within their existing tolerances. Local validation also passes
91 JavaScript tests, four WGSL tests, syntax checks, Rust formatting, all three
WebAssembly builds and validation, and site staging. Generated WebAssembly and
model binaries remain untracked. These checks do not establish general model
quality; see [BENCHMARK.md](../BENCHMARK.md) for the broader accuracy limitations.

## Rejected experiments

The bounded loop logged two keeps, five discards, and one strategy refinement.
The following candidates were compared with the retained E2B aggregate of 106.17 ms.
Their code is excluded from the final change.

| Candidate | E2B GPU aggregate | Decision |
|---|---:|---|
| Wider 128-column matrix tile with named accumulators | 107.25 ms | Mixed matrix results; total time regressed |
| Skip empty key groups in attention tails | 107.62 ms | Extra control overhead erased savings |
| Four lanes per key, 64 concurrent keys | 108.46 ms | Slower than eight lanes per key |
| FP16 products widened to FP32 sums | 132.51 ms | 24.8% total regression; sampled large matrices 37–49% slower |
| Share KV tiles across eight queries | 105.57 ms | Only 0.56% better for 203 added lines and another pipeline |

Matrix tests included representative Laya, Qwen, and Gemma shapes. They did not
establish a shared matrix improvement. The current matrix selector, CPU kernels,
and weight format remain unchanged. The remaining large costs are feed-forward
matrix multiplies and browser completion waits.

## Reproduce

Build and serve the checkout with `pnpm build` and `pnpm serve --port=18097`.
The attention guard needs no model download:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result gpu --timeout 600 \
  --url 'http://127.0.0.1:18097/dev/gemma-attn-bench.html?cases=4096x2x1x512x0x1,1025x8x2x256x512x1,129x4x1x257x128x0&samples=3&magnitude=4'
```

For full-model profiles, place the existing packs at `tmp/gemma-4-e2b-q8.kevala`
and `tmp/gemma-4-e4b-q8.kevala`:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result bench --timeout 600 \
  --url 'http://127.0.0.1:18097/bench.html#auto&backend=webgpu&pack=local&model=gemma-4-e2b&profile=1&runs=7&warmups=2&unique=1&shapes=0,1,2'

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result decision --timeout 1200 \
  --url 'http://127.0.0.1:18097/dev/decision-bench.html?model=gemma-4-e2b&backend=webgpu&pack=local&dataset=all&permutations=3'
```

Use `gemma-4-e4b` and five measured profile runs for E4B. For wall timing, change
`--result bench` to `--result latency` and `profile=1` to `profile=0`. Select
`/usr/share/vulkan/icd.d/radeon_icd.json` for the AMD adapter on this machine.
