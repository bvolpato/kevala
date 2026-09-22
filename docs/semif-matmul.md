# Matrix kernels for Laya, Kev, and SemIf on Firefox/Linux

Measured September 22, 2026 on Ubuntu with Firefox 152.0.3, an RTX 5070 Ti
(NVIDIA 595.71.05), and the Ryzen 9950X3D integrated GPU (RADV, Mesa 25.2.8).
These browser runs use WebGPU through Vulkan.

The curated machine-readable evidence is in
[semif-matmul-linux-2026-09-22.json](benchmarks/semif-matmul-linux-2026-09-22.json).
It keeps the raw run names, per-case medians, selector decisions, matrix medians,
and correctness summaries without copying the full browser logs.

## Model results

These are GPU kernel times with automatic selection enabled in the final runtime.

| GPU | Model | Baseline | Automatic | Reduction | Calibration | Total load |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | SemIf 0.8B | 49.58 ms | 41.33 ms | 16.6% | 483 ms | 2.65 s |
| RTX 5070 Ti | SemIf 2B | 103.09 ms | 80.49 ms | 21.9% | 582 ms | 4.95 s |
| RTX 5070 Ti | SemIf 4B | 246.94 ms | 189.94 ms | 23.1% | 751 ms | 9.54 s |
| RTX 5070 Ti | Kev 0.8B | 28.47 ms | 26.05 ms | 8.5% | 481 ms | 2.61 s |
| RTX 5070 Ti | Kev 4B | 139.89 ms | 112.30 ms | 19.7% | 679 ms | 9.34 s |
| RTX 5070 Ti | Kev 9B | 247.72 ms | 196.25 ms | 20.8% | 1,182 ms | 18.52 s |
| RTX 5070 Ti | Laya | 19.62 ms | 15.54 ms | 20.8% | 595 ms | 2.23 s |
| AMD integrated | SemIf 0.8B | 1,230.07 ms | 1,231.51 ms | -0.1% | 2,174 ms | 4.64 s |
| AMD integrated | Laya | 477.45 ms | 475.03 ms | 0.5% | 1,695 ms | 3.19 s |

The SemIf 0.8B and 2B loads measured all five distinct projection shapes. Forced-wide runs
measured 41.44 ms and 80.40 ms, respectively. Automatic selection retains those
gains while allowing other adapters and shapes to keep the generic kernel.
Baseline load times were 2.38 s and 4.94 s. These are separate fresh-browser runs,
so total load differences include filesystem and compilation variability.

The previously reported 178 ms result was a SemIf 2B round trip. The separate
automatic SemIf 2B benchmark here measured 80.49 ms of GPU kernels. Those numbers cover
different scopes: the round trip includes browser scheduling, queue completion,
and readback, while the GPU aggregate sums timestamped kernels across three
request sizes.

The additional model rows use the same paired baseline and automatic protocol
with five samples per case. Laya uses the same selected matrix path through its
existing non-Qwen layouts.

On AMD, SemIf 0.8B kept every measured shape on the generic kernel. Its 0.12%
timing difference is within run-to-run variation. Calibration stopped after the
two largest projection shapes and exceeded its soft budget while the second shape
completed. Laya likewise kept the generic kernel and measured a 0.5% difference;
its calibration stopped after three shapes. AMD baseline loading took 2.53 s for
SemIf and 1.51 s for Laya; calibration accounts for the extra startup cost.

## Retained kernel

The wide projection kernel stages two 32-value quantization blocks at a time.
Each thread keeps its row accumulators in named f32 vectors. This removes dynamic
private-array indexing and halves the number of shared-memory barriers compared
with the existing 32-value tile. Weights remain int8, shared tiles remain f16,
and dot products and accumulation remain f32 in the same order.

On NVIDIA, the large 0.8B matrix cases measured 24–28% less GPU time. The exact
compiler or driver optimization responsible for the register change has not been
isolated. On AMD, the same large matrices improved only about 2%, below the
selector's 5% threshold.

The design investigation used [DeepGEMM's shape and hardware heuristics](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/jit_kernels/heuristics/sm90.hpp)
as a reference for choosing implementations from measured hardware behavior.
The retained implementation is WGSL and uses Kevala's existing Q8 format.

## Automatic selection and API overrides

```js
const model = await Kevala.load({
  model: "semif-qwen3.5-0.8b",
  backend: "webgpu",
  gpuKernel: "auto", // default; also "generic" or "wide"
});
console.log(model.info.gpuTuning);
```

Kev, SemIf, and Laya calibrate against the actual loaded Q8 projection weights.
Repeated layer shapes share a decision. Calibration measures representative inputs of
16, 32, 48, 128, and 512 tokens, covering all four row variants and two ranges
for the largest tile. It prioritizes the MLP and QKV projections.

Each point alternates the kernels across three paired GPU timestamp samples,
with four dispatches per sample. The wide kernel must win every pair and improve
the median by at least 5%, with at least two pairs also clearing that threshold.
Ties, incomplete measurements, and noisy results select the generic kernel.

Calibration uses at most 128 MiB of scratch and stops starting new shapes after
a 1.5-second budget. An already submitted shape finishes, so this is a soft
budget. Unmeasured shapes use the generic kernel. No tuning profile persists
between loads, because Firefox may hide the identity of the physical adapter.
Both `shader-f16` and `timestamp-query` are required for automatic measurement.
Without either, the generic kernel remains available. An optional calibration
failure clears measured choices back to generic.

Overrides skip measurement. `"wide"` requires f16 and a successfully compiled
wide pipeline; an unsupported explicit request rejects GPU initialization.
With `backend: "webgpu"`, this rejects loading. `backend: "auto"` retains the
existing adapter and CPU fallback behavior. The option applies to all model families.
`info.gpuTuning` reports the method, measurements, selected kernels, elapsed time,
and whether calibration stopped at its budget.

## Matrix spot checks

The final wide kernel was compared with the runtime-selected generic path on
the NVIDIA adapter. Values are median GPU milliseconds from three samples after
one warmup. The small cases exercise partial rows; the larger cases cover the
SemIf and Kev projection shapes.

| Shape (T x N x K) | Runtime | Wide | Change |
| --- | ---: | ---: | ---: |
| 7 x 132 x 96 | 0.00658 | 0.00500 | -24.1% |
| 31 x 132 x 160 | 0.01191 | 0.00979 | -17.9% |
| 45 x 132 x 288 | 0.01674 | 0.01228 | -26.7% |
| 441 x 12,288 x 2,048 | 2.22109 | 1.69805 | -23.5% |
| 441 x 2,048 x 6,144 | 1.36165 | 0.99344 | -27.0% |
| 441 x 19,456 x 2,560 | 4.44330 | 3.15670 | -29.0% |
| 441 x 2,560 x 9,728 | 2.27632 | 1.60093 | -29.7% |

Both kernels agreed exactly across all seven shapes in the cross-kernel comparison
and passed the CPU reference checks. On AMD, the same wide path was faster on
the larger final spot checks but slower on the smallest case. The automatic
selector therefore retained the generic path for the measured AMD shapes.

## Measurement method

The baseline is `main` at `c6ab879`, with the local benchmark model resolver fixed
so every published name loads its matching pack. Previously, local SemIf names
fell through a two-model lookup and could silently benchmark Laya.

The SemIf benchmark uses distinct states to bypass inference-cache hits, two
warmups, then seven samples for 0.8B or five for the larger models. Inputs contain
105, 198, and 606 tokens. Kev uses five samples at 31, 124, and 532 tokens; Laya
uses five samples at 47, 140, and 512 tokens. AMD runs use five samples. Each
case takes the median of summed GPU kernel timestamps; the reported aggregate is
the geometric mean across the three cases. Downloads, loading, and calibration
are excluded from inference time and reported separately.

GPU timestamps exclude CPU work, browser scheduling, and readback. Firefox's
roughly 100 ms completion intervals can hide kernel gains in round-trip latency.
These measurements do not establish competitive performance against other
inference engines or state-of-the-art decision accuracy.

## Correctness and feature fallbacks

The final automatic selector passed the pinned reference decisions on the NVIDIA
adapter for every published model checked in this sweep. The limits below were
fixed before the run and no tolerance was changed.

| GPU | Model | Decisions | Maximum probability difference | Limit |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | SemIf 0.8B | 12/12 | 0.030830 | 0.032 |
| RTX 5070 Ti | SemIf 2B | 12/12 | 0.028985 | 0.031 |
| RTX 5070 Ti | SemIf 4B | 12/12 | 0.020469 | 0.022 |
| RTX 5070 Ti | Kev 4B | 13/13 | 0.009944 | 0.011 |
| RTX 5070 Ti | Kev 9B | 5/5 | 0.004692 | 0.005 |
| RTX 5070 Ti | Laya | 41/41 | 0.023506 | 0.024 |
| AMD integrated | SemIf 0.8B | 12/12 | 0.031412 | 0.032 |
| AMD integrated | Laya | 41/41 | 0.023506 | 0.024 |

The broader NVIDIA guard also passed 17 attention shapes with three verification
repeats, all seven final matrix shapes, the basic GPU kernels with maximum error
`9.48e-6`, Laya at 41/41 with maximum difference `0.023506`, and Kev 0.8B at
13/13 with maximum difference `0.010214`. Cache miss, hit, and extension paths
matched exactly in the cache guard. The final SemIf 0.8B probability arrays and
decisions also matched the baseline parity details byte-for-byte across 12
questions.

The selector remains usable when Firefox does not expose optional features. A
no-`shader-f16` smoke selected the generic kernel with reason `no-shader-f16`, and
a no-`timestamp-query` smoke selected it with reason `no-timestamp-query`. Both
loads completed on WebGPU and produced finite output. These were one-case smoke
runs, not performance comparisons.

The short game smoke also ran SemIf 2B and Kev 4B through the Tetris placement
policy on NVIDIA. Each five-piece game survived all five moves, made legal
choices, and produced finite probabilities. This validates integration and model
loading; it is not a gameplay quality benchmark.

## Rejected experiments

The 12-iteration research log contains two retained changes, eight rejected
trials, and two strategy adjustments. Rejected code is excluded from the PR.

| Candidate | SemIf 0.8B GPU aggregate | Outcome |
| --- | ---: | --- |
| Existing kernel | 49.58 ms | Baseline |
| Shape constants | 49.77 ms | No improvement; more compilation |
| 64-value staging | 47.30 ms | Retained, then combined with named registers |
| Outer products | 341.32 ms | Much slower; first version also failed compilation |
| Packed int8 weight layout | 49.51 ms | Insufficient gain |
| 128-value staging | 50.93 ms | Slower |
| Weights converted to f16 at load time | 48.25 ms | Slower than retained kernel; twice the weight storage |
| Named accumulators with 64-value staging | 41.44 ms | Retained |
| Padded input tile | 43.54 ms | Slower; first version failed the numerical guard |

The retained unpadded kernel passed the full GPU guards. Rejected candidates
were stopped after their relevant matrix or compilation checks and full-model
performance measurement; they did not all run the cross-model guard suite.

## Reproduce

Build all WASM flavors with `pnpm build`, place the published packs in `tmp/`,
and run `pnpm serve --port=18086`. Select one Vulkan ICD per run:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
uv run scripts/bench-gpu.py --result bench --timeout 600 \
  --url 'http://127.0.0.1:18086/bench.html#auto&backend=webgpu&pack=local&model=semif-qwen3.5-0.8b&runs=7&warmups=2&profile=1&unique=1&shapes=0,1,2' \
  --output tmp/semif-gpu.json
```

Use `gpuKernel=generic` or `gpuKernel=wide` in the fragment to compare overrides.
Use `model=semif-qwen3.5-2b` and `runs=5` for the larger model. AMD runs use
`/usr/share/vulkan/icd.d/radeon_icd.json` on this workstation.
The matrix harness accepts `matmul_wide_h@256:auto` alongside `runtime`.
The wide variant supports one 64-column group per workgroup.
