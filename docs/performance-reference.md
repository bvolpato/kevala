# Performance reference for projection kernels

This 2026-11-06 capture compares bounded synthetic projection screens and a separate native reference. It does not establish that every Kevala kernel is globally optimal, or that a kernel change improves full-model latency. No candidate from these screens was enabled in production. The archived [summary and raw capture manifest](benchmarks/performance-reference-2026-11-06/summary.json) identifies the source commit, measurements, failures, and compressed raw evidence. It omits model weights and compiled binaries.

## Native cuBLAS reference

The native run used an NVIDIA GeForce RTX 5070 Ti and CUDA 13.2 with cuBLAS 13.4.0. It measured resident FP32 GEMM with `CUBLAS_COMPUTE_32F_PEDANTIC`; TF32 was disabled. Q8 weights were dequantized on the host to FP32 before timing, and the input and expanded weights were already resident on the GPU. The timed scope excludes Q8 decoding, uploads, bias, residual, activation, model setup, and browser dispatch. Each shape had three warmups and nine samples; outputs passed the CPU FP32 and CPU double tolerance checks.

| T × N × K | Median GEMM time |
| --- | ---: |
| 47 × 18,432 × 2,560 | 0.414 ms |
| 128 × 18,432 × 2,560 | 0.613 ms |
| 512 × 18,432 × 2,560 | 2.393 ms |
| 512 × 24,576 × 4,096 | 3.620 ms |

These are a reference, not a hardware ceiling. FP32-expanded weights take 4 bytes per weight, compared with about 1.125 bytes per weight for Q8 block-32 values and scales. The largest case kept about 440 MiB of device buffers live, excluding the cuBLAS workspace. The much smaller Q8 representation, dequantization, transfer, allocation, and browser work are outside this GEMM timing. See the [cuBLAS compute-type documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex) and the [native capture record](benchmarks/performance-reference-2026-11-06/summary.json) for the precise scope and all 19 shapes.

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
