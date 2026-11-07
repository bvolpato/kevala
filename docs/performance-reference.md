# Performance reference for projection kernels

The original 2026-11-06 capture compares bounded synthetic projection screens and a separate native reference. The dated extension below compares one real Q8 projection with its exact FP32 weight expansion, using synthetic activations. Neither establishes that every Kevala kernel is globally optimal or that a kernel change improves full-model latency. No candidate from these screens was enabled in production. The original [summary and raw capture manifest](benchmarks/performance-reference-2026-11-06/summary.json) identifies the November 6 source commit, measurements, failures, and compressed raw evidence. It omits model weights and compiled binaries.

## Native cuBLAS reference

The native run used an NVIDIA GeForce RTX 5070 Ti and CUDA 13.2 with cuBLAS 13.4.0. It measured resident FP32 GEMM with `CUBLAS_COMPUTE_32F_PEDANTIC`; TF32 was disabled. Q8 weights were dequantized on the host to FP32 before timing, and the input and expanded weights were already resident on the GPU. The timed scope excludes Q8 decoding, uploads, bias, residual, activation, model setup, and browser dispatch. Each shape had three warmups and nine samples; outputs passed the CPU FP32 and CPU double tolerance checks.

| T × N × K | Median GEMM time |
| --- | ---: |
| 47 × 18,432 × 2,560 | 0.414 ms |
| 128 × 18,432 × 2,560 | 0.613 ms |
| 512 × 18,432 × 2,560 | 2.393 ms |
| 512 × 24,576 × 4,096 | 3.620 ms |

These are a reference, not a hardware ceiling. FP32-expanded weights take 4 bytes per weight, compared with about 1.125 bytes per weight for Q8 block-32 values and scales. The largest case kept about 440 MiB of device buffers live, excluding the cuBLAS workspace. The much smaller Q8 representation, dequantization, transfer, allocation, and browser work are outside this GEMM timing. See the [cuBLAS compute-type documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex) and the [native capture record](benchmarks/performance-reference-2026-11-06/summary.json) for the precise scope and all 19 shapes.

For isolated CPU Q8 row decoding timings, see the [native AVX2 row report](native-q8-performance.md). It uses a different host-side scope and does not alter the GPU comparison above.

## Chrome feature boundary

The browser capability capture used Chrome for Testing 156.0.8078.4 with Dawn revision [`0a2c7df818e285d6db0f085135139cda04cee8bb`](https://chromium.googlesource.com/chromium/src/+/refs/tags/156.0.8078.4/DEPS). On this workstation, NVIDIA driver 595.71.05 is below Dawn's Vulkan NVIDIA shader-f16 gate of 615.71. The adapter therefore did not expose `shader-f16`. Dawn's [pinned Vulkan feature gate](https://dawn.googlesource.com/dawn/+/0a2c7df818e285d6db0f085135139cda04cee8bb/src/dawn/native/vulkan/PhysicalDeviceVk.cpp) is the reason this browser capture cannot support an f16 performance claim.

The adapter did expose Chromium's experimental subgroup-matrix feature, subgroup operations, and packed 4×8-bit integer dot products. Matrix configurations are reported through `adapter.info.subgroupMatrixConfigs`; these are browser-specific experimental capabilities, not a portable fast path. The experimental INT8 and DP4A probes quantized FP32 activations, so they did not preserve the existing Q8-weight-by-FP32-activation arithmetic. Synthetic outputs passed the declared tolerance checks, but no full-model probability-parity result was produced.

## Candidate screens

The timings below are isolated GPU timestamp measurements. A screen result alone does not prove an end-to-end model improvement.

| Candidate | Representative result | Decision |
| --- | --- | --- |
| Q8 × F32 software prefetch | At 512 × 24,576 × 4,096, current was 5.326 ms; dynamic prefetch was 6.327 ms; shape-specific prefetch was 7.116 ms. | Reject: both prefetch variants were slower. |
| DP4A with activation preparation | At 512 × 24,576 × 4,096, the frozen baseline was 5.275 ms and the DP4A composite was 5.781 ms. | Reject: slower in this case, changes activation precision, and lacks full-model parity. |
| INT8 subgroup-matrix path | At the same shape, baseline was 5.511 ms; the two INT8 composites were 23.939 ms and 21.692 ms. | Reject: slower, quantizes activations, and lacks full-model parity. |
| Q8 × F32 workgroup tiles | Some individual shapes improved, but the gain did not repeat consistently across reruns and tail shapes. | No production change: no stable cross-shape winner. |
| Gemma Q8 × F32 MLP screens | Corrected reruns passed exact output checks. Some T=16/32 cases were locally 5–11% faster, while other cases were slower; no loaded-model win was measured. | No production change: local shape wins did not establish model latency improvement. |

The [GPU Goldmines paper](https://arxiv.org/abs/2610.17321) is relevant WebGPU kernel-tuning research, but its reported ranks are not used here. This capture has no matched competitor run with the same hardware, browser, precision, shapes, inputs, and timing scope.

## Excluded failures and limits

These failures did not produce valid timing or correctness evidence: the first Chromium capability probe returned `no WebGPU adapter`; the initial Gemma harness stopped with `ReferenceError: gpuJsSha256 is not defined` before a screen ran; and a portable-tile request was rejected because its case exceeded the harness bounds (`T<=512`, `N<=32768`, `K<=16384`, with K a multiple of 32). Later capability and Gemma probes are recorded separately. The archive also records two older failed capability-harness source versions that could not be recovered byte-for-byte; their hashes are listed in its manifest. None of these are kernel correctness failures.

The screens cover selected projection shapes, not all model paths, all supported devices, image inference, sustained application workloads, or competing libraries under a matched protocol. Treat browser and native references separately. Promote a candidate only after numerical gates, repeatable paired timings, and unprofiled full-model latency checks pass, as described in the [kernel profiling workflow](kernel-profiling.md).

## November 7: matched real-weight FP32 cache screen

**Decision: reject promotion.** Replacing on-the-fly Q8 decoding with a resident FP32 cache did not produce a convincing projection win. This screen used the real Kev-0.8B `L.0.gate_up` matrix, shaped 7,168 × 1,024, with deterministic finite activations. The Q8 and FP32 paths used the same weights, inputs, FP32 workgroup tiles, row geometry, split-K rule, bias and output modes. Each cached weight was `Math.fround(signed_int8 * original_f32_scale_per_K32_block)`. No model or runtime default changed.

The host was the Ryzen 9 9950X3D and RTX 5070 Ti with NVIDIA driver 595.71.05. Chrome 156.0.8078.4 used Vulkan and a hardware adapter. The device requested timestamp queries and omitted `shader-f16`. Each arm had three warmups and nine alternating ABBA/BAAB quartets. Common repetition counts came from one Q8 calibration targeting 20 ms, capped at 128 repetitions. The T=1 and T=47 batches reached the cap before reaching that target. All slower samples remain included.

The primary ratio is the median of nine Q8/FP32 quartet ratios. Independent medians and observed ranges are separate statistics; the ranges are not confidence intervals. The identical Q8 A/A control used the same pipeline, buffers and inputs. No A/A correction was applied.

| T | Independent Q8 / FP32 medians | Paired Q8 / FP32 ratio | Observed ratio range | FP32 faster | Identical Q8 A/A ratio | A/A range | A/A second arm faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.026744 / 0.032933 ms | 0.81010 | 0.67673–0.87313 | 0/9 | 0.99430 | 0.94772–1.05080 | 3/9 |
| 47 | 0.048487 / 0.049320 ms | 0.97661 | 0.94538–1.04182 | 2/9 | 0.99686 | 0.94952–1.01825 | 4/9 |
| 158 | 0.146694 / 0.148352 ms | 0.99052 | 0.94741–1.02847 | 4/9 | 0.98947 | 0.97383–1.02095 | 3/9 |
| 512 | 0.413942 / 0.408906 ms | 1.00629 | 0.98430–1.02847 | 6/9 | 0.99528 | 0.98254–1.04205 | 3/9 |

FP32 was slower at T=1 and showed mixed paired results at T=47 and T=158. Its 0.63% paired latency reduction at T=512 fell within the observed A/A spread. These results do not support a cache promotion. T=1 is a padded matrix-kernel memory control; it does not measure the runtime's matrix-vector selector.

All active output bits matched between Q8 and FP32 before and after timing. Finite-output checks, output and partial-buffer canaries, and the bias, residual and ReLU controls passed. Independent audit recomputed the paired summaries and verified positive, chronological GPU timestamps. Timings cover a resident projection's matrix dispatch and split-K reducer when used. They exclude fetching, uploads, cache construction, pipeline compilation, readback and model execution.

One cache construction took 93.2 ms: 38.8 ms for CPU decoding and 54.4 ms for allocation, upload and completion. This is one observed construction, not a repeated benchmark. The cache added 28 MiB beside the 7.875 MiB Q8 representation, for 35.875 MiB of coexisting weight buffers. Releasing Q8 would instead make the representation increase 20.125 MiB. The exact peak of explicitly created GPU buffers was 69,103,968 bytes. It excludes driver, pipeline, query, browser and WebAssembly allocations and is not a total VRAM peak.

The [November 7 evidence archive](benchmarks/f32-weight-cache-2026-11-07/README.md) preserves the source protocol, raw samples, receipts, identity records and the unexecuted earlier source draft. The whole-pack hash is historical; this run freshly bound the header and selected weight/scale ranges to local and HTTP hashes and checked before/after file metadata. It does not claim a fresh whole-pack or whole-HTTP hash. Exact CI runtime bytes were verified for this local experiment, while the referenced CI71 Pages deployment remained unverified.
