# Documentation

Start with [How kevala works](architecture.md). The guides describe the current code. The reports record one measurement each, with its date, hardware, and method.

## Guides

| Guide | Subject |
|---|---|
| [How kevala works](architecture.md) | The Rust engine, the WebAssembly ABI, the browser runtime, the pack format, and the backends. |
| [Choosing and running models](models.md) | The model families, their sizes, and their memory needs. |
| [Packs: loading, converting and publishing](packs.md) | How the runtime loads a pack, and how to convert, check, and publish one. |
| [Pack model card](pack-model-card.md) | The card that the hosted pack repository shows. |
| [Gemma 4 decisions](gemma4.md) | The Gemma 4 integration, its limits, and its verification. |
| [Quantization choice](quantization.md) | Why the packs use int8 weights, and what that costs. |
| [Automatic CPU tuning](cpu-tuning.md) | How the runtime selects a CPU kernel and a worker count. |
| [Adding a model family](adding-a-model.md) | The steps to add a backbone, a head, or a request template. |
| [Tetris demo policy and measurements](tetris-policy.md) | How the Tetris demo uses a decision model. |

## Benchmark methods and results

| Report | Subject |
|---|---|
| [Decision benchmark](../BENCHMARK.md) | The decision accuracy and latency protocol, with reproduction commands. |
| [Optional model validation](model-benchmarks.md) | Parity and latency of the larger Kev and SemIf packs. |
| [Firefox Linux GPU kernel sweep](gpu-benchmarks.md) | GPU kernel timings and numerical guards on Firefox and Linux. |
| [M4 Max kernel sweep](m4-max-kernel-performance.md) | GPU kernel timings on Apple M4 Max. |
| [M4 Max kernel safety audit](kernel-safety.md) | Numerical repairs for Metal. |
| [CPU kernel profiling and benchmarks](cpu-benchmarks.md) | CPU profiles, worker scaling, and SIMD validation. |
| [Matrix kernels for Laya, Kev, and SemIf](semif-matmul.md) | The matrix kernel selection on Firefox and Linux. |

## Kernel optimization reports

Each report covers one retained change.

| Report | Kernel |
|---|---|
| [Shared WebGPU matrix layout](shared-matmul-performance.md) | Matrix staging shared across families |
| [Shared FP32 matrix kernels](matmul-32x8-performance.md) | FP32 matrix tiles |
| [Avoid padded FP32 matrix rows](fp32-row-padding-performance.md) | FP32 matrix row padding |
| [Explicit FP32 matrix output stores](fp32-explicit-stores-performance.md) | FP32 matrix output stores |
| [Exact byte conversion in shared FP32 matrix kernels](fp32-byte-conversion-performance.md) | FP32 matrix weight loads |
| [FP32 BM56 row tile performance](fp32-row56-performance.md) | FP32 matrix 56-row tiles |
| [Laya FP32 attention performance](laya-fp32-attention-performance.md) | Laya attention without FP16 |
| [Laya compact final-head performance](laya-compact-head-performance.md) | Laya final decision head |
| [FP32 query-tiled Qwen attention](qwen-fp32-attention-performance.md) | Qwen attention without FP16 |
| [Subgroup direct-load recurrence performance](recurrence-direct-performance.md) | Kev recurrence with subgroups |
| [Packed vector loads for recurrent state](recurrence-packed-performance.md) | Kev recurrent state loads |
| [Gemma 4 attention performance](gemma-attention-performance.md) | Gemma attention |
| [Gemma normalization and residual fusion](gemma-rms-residual-performance.md) | Gemma normalization |
| [Shared GPU kernels and Gemma execution](gpu-latency-performance.md) | End-to-end latency after the shared kernel work |

## Raw data

- [`docs/benchmarks/`](benchmarks/) holds the raw samples that the kernel reports cite.
- [`benchmarks/results/`](../benchmarks/results/) holds the raw results of each decision campaign.
