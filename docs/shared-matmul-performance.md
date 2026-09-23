# Shared WebGPU matrix layout

This change applies to the Q8 matrix kernel used by Laya, Kev, SemIf, and Gemma 4.
It follows the Gemma-only attention optimization in [PR #20](https://github.com/bvolpato/kevala/pull/20).

## What changed

The wide kernel stages 64 input values and computes 64 output columns. Its FP16
weight tile previously used an eight-slot XOR permutation, inherited from the
FP32 layout. A `vec4<f16>` occupies two 32-bit shared-memory banks. On GPUs with
32 such banks, using all 16 slots can avoid paired column conflicts. This follows
the bank organization described in the [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
The bank explanation is an inference from the layout; the performance evidence
below uses WebGPU timestamps, not hardware bank-conflict counters.

The new permutation covers the whole 64-value tile, including the bit that
selects its two quantization blocks. Applying a four-bit XOR inside each
32-value block would cross row boundaries and produce incorrect results.

Only FP16 staging with four rows per thread uses the new permutation. Paired
validation found a 7.5% regression in a short NVIDIA matrix when applying it to
all row sizes, so smaller row tiles keep the original permutation. FP32 staging,
Q8 weights, accumulation order, split-K, memory allocation, and public API stay
the same. Existing automatic selection still compares the generic and wide
kernels on the loaded model's shapes.

## Measurement

Ubuntu, Firefox 152.0.3, NVIDIA RTX 5070 Ti (16 GB, driver 595.71.05), Ryzen
9950X3D. GPU jobs ran serially. A separate Vulkan ICD selected the integrated
AMD GPU for additional kernel checks. Firefox hides the adapter name; each paired
result records its ICD, hardware-adapter check, device features, and limits.

Full-model measurements use three input sizes per model, two warmups, five
measured requests per size, unique input states, and automatic kernel selection.
The metric is the geometric mean of per-case median sums of GPU kernel times.
It excludes loading and does not claim a corresponding browser round-trip gain.
Request caching cannot satisfy the unique inputs.

The raw report includes the initial baseline, a fresh baseline after the search,
retained candidates, final validation, and rejected experiment measurements.
Short-input timings vary with load-time kernel selection and GPU clock state.
The interleaved matrix test is the stronger evidence for the isolated kernel
change: old and new shaders come from their exact WASM builds, share inputs, and
alternate timing order in each round.

### Full-model GPU time

| Model | Initial baseline | Fresh baseline | Final | Final repeat |
| --- | ---: | ---: | ---: | ---: |
| laya | 16.21 ms | 15.11 ms | 15.04 ms | 15.55 ms |
| kev-0.8b | 24.91 ms | 25.59 ms | 24.40 ms | 24.74 ms |
| semif-qwen3.5-2b | 80.89 ms | 81.48 ms | 78.90 ms | 79.57 ms |
| gemma-4-e2b | 107.26 ms | 106.00 ms | 104.47 ms | 104.68 ms |
| Four-family geometric mean | 43.26 ms | 42.75 ms | 41.70 ms | 42.31 ms |

Gemma E2B is 1.2–1.5% faster than the fresh baseline in these two final runs.
Kev and SemIf also improve in both runs. Laya varies around the fresh baseline,
so this does not establish a full-model Laya gain. Short-case movement exceeds
the isolated matrix gain in some cases; it cannot all be attributed to this
shader change. The aggregate improvement is 1.0–2.5% against the fresh baseline.

### Interleaved matrix timing

Dimensions are `tokens × output columns × reduction width`. These are kernel
measurements with fixed dispatch settings, independent of load-time selection.

| Adapter / case | Baseline | Final | Time reduction |
| --- | ---: | ---: | ---: |
| NVIDIA: 99x6144x1536 | 0.188544 ms | 0.181897 ms | 3.53% |
| NVIDIA: 193x1536x6144 | 0.402016 ms | 0.390016 ms | 2.98% |
| NVIDIA: 601x6144x1536 | 0.862336 ms | 0.837968 ms | 2.83% |
| NVIDIA: 105x12288x2048 | 0.482096 ms | 0.467400 ms | 3.05% |
| NVIDIA: 198x2048x6144 | 0.497352 ms | 0.483488 ms | 2.79% |
| NVIDIA: 47x1536x768 | 0.017518 ms | 0.017501 ms | 0.10% |
| NVIDIA: 31x1024x3584 | 0.035616 ms | 0.035589 ms | 0.08% |
| NVIDIA / Laya: 47x5248x1024 | 0.048410 ms | 0.048251 ms | 0.33% |
| NVIDIA / Laya: 140x5248x1024 | 0.178110 ms | 0.171417 ms | 3.76% |
| NVIDIA / Laya: 512x1024x2624 | 0.226665 ms | 0.219973 ms | 2.95% |
| AMD: 3x70x96 | 0.008820 ms | 0.009217 ms | -4.51% |
| AMD: 31x256x1024 | 0.093281 ms | 0.093269 ms | 0.01% |
| AMD: 47x1536x768 | 0.484368 ms | 0.484929 ms | -0.12% |
| AMD: 193x1536x6144 | 15.308306 ms | 15.082730 ms | 1.47% |

The seven full-row NVIDIA matrices improve by 2.8–3.8%, including actual Laya
projection dimensions. Short tiles are essentially unchanged on NVIDIA. AMD
shows a 1.47% gain on the full-row case, near-zero changes on two short cases,
and a 4.51% slowdown (0.40 μs) on the smallest case. No universal speedup is claimed.

[Raw samples, tuning decisions, shader/WASM hashes, and validation results](benchmarks/shared-matmul-linux-2026-09-23.json).

## Correctness

The paired tests compare every output element against the old implementation
and also check an independent CPU calculation (all elements for small cases,
49 coordinates for large cases). Small cases exercise bias,
residual addition, ReLU, split-K, and partial tiles. Large cases check the matrix
output and compare the full old/new arrays. They do not repeat every epilogue at
large dimensions.

Model checks use the existing golden fixtures and their existing probability
tolerances. No weights or tolerances were changed. The larger Kev, SemIf, and
Gemma variants run in final validation rather than every search iteration.
All nine weights passed: 132/132 expected decisions. Every paired matrix output
was exactly equal to the baseline. The Kev state-cache check also passed with
zero probability differences across cache misses, hits, and extensions.

Local validation passed 91 JavaScript tests, four WGSL parsing tests, JavaScript
syntax checks, Rust formatting, all three WASM builds and ABI checks, and site
staging. These GPU results complement the CPU and packaging gates in CI.

## Rejected work

The foreground search used ten iterations: one retained candidate, seven
rejected candidates, and two strategy adjustments. It tested higher split-K
occupancy, a cooperative Kev/SemIf attention port, deduplicated packed-weight
loads, 128-thread workgroups, explicit FP32 FMA accumulation, smaller row tiles,
and FP32 input staging. Some isolated timings improved, but the complete suite
regressed, the gain was within run variation, or the memory/precision tradeoff
was too large for the measured gain. Rejected code is excluded from this change.

The retained candidate was subsequently restricted to full row tiles after the
interleaved test exposed the short-shape regression. That final restriction is
included in the published measurements.

## Reproduce

Build each revision with `pnpm build`. Keep generated WASM files outside Git.
Serve the baseline and changed checkouts with `pnpm serve` on separate ports,
with the published model packs available in each checkout's `tmp/` directory.

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18100/ tmp/baseline
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  node scripts/bench-gpu-suite.mjs http://127.0.0.1:18099/ tmp/current
```

For an interleaved comparison, copy the baseline's `kevala-base.wasm` into
`tmp/baseline-base.wasm` in the changed checkout, then run:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  uv run scripts/bench-gpu.py --result gpu --timeout 600 \
  --url 'http://127.0.0.1:18099/dev/gpu-bench.html?baselineWasm=../tmp/baseline-base.wasm&cases=99x6144x1536,193x1536x6144,601x6144x1536&kernels=matmul_wide_h_baseline@256:auto,matmul_wide_h@256:auto&samples=7' \
  --output tmp/matmul-paired.json
```

Use `radeon_icd.json` instead of `nvidia_icd.json` for the secondary adapter.
The `_baseline` variants require `baselineWasm`; both their matrix and reduction
shaders come from that build. URLs are recorded in the result, and the published
report additionally records the actual WASM digests.
