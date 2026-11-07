# Native AVX2 2×4 linear path

On one Ryzen 9 9950X3D host, a bounded fresh-process CLI comparison recorded median paired baseline-to-candidate ratios of **2.3275× for Laya Q8** and **2.0486× for Kev 0.8B Q8**. Each candidate won all nine measured quartets for its model. The independent arm medians were 1,401.6 to 596.65 ms for Laya and 1,217.95 to 616.15 ms for Kev. These measurements cover one fixed request on one host. They do not establish a general model ranking or a universal speedup.

Final native production validation passed after three numerical guards were rerun with explicit artifact paths. The receipts and hashes appear below.

## Scope and method

The comparison used the frozen v2 candidate against the previously accepted AVX2 baseline kernel. The frozen source records base commit `97778af0ed881c876313fcefa5a17399ceb10f2d`. The baseline `kernels.rs` SHA-256 is `4ef34fe7ac7ac6dc1508bdfcc4085f6fb99f2d7157879d183543b65e6038adb9`; the v2 candidate SHA-256 is `d7bfece0d5fdbaa4e22c6681ff32720e4bbe074378d83fd61f814b0dc280623f`.

The specialization applies in the default x86_64 build when runtime AVX2 is available and `T >= 2`, `N >= 4`, and `K >= 64`. Globally AVX2-enabled, ARM, and WebAssembly paths remain unchanged. Each output preserves the four `k % 4` accumulation lanes and the existing reduction order. If any output is nonfinite, the generic implementation recomputes the full result.

Each model used three discarded observations per arm, followed by nine alternating ABBA and BAAB quartets. Each quartet ran two baseline and two candidate calls. The run recorded 12 discarded and 72 measured observations across 84 unique CLI process IDs. Every process loaded the model and used the same frozen state and question request. Model loading was measured separately.

The inference timer surrounds `Model::decide` after model loading and parsing the state and questions, and ends before output serialization. It includes preparation, the forward pass, and response construction inside `Model::decide`. The reported p50s are arm medians across 18 observations. A paired ratio is the baseline time divided by candidate time for a quartet; the table reports the median and range of nine quartet ratios. This is fresh-process inference time, not process wall time or warm resident latency. Timer resolution is 0.1 ms.

| Model | Baseline p50 | Candidate p50 | Median paired ratio, baseline / candidate | Range across nine quartets | Candidate faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya Q8 | 1,401.6 ms | 596.65 ms | 2.3275 | 2.1538–2.7019 | 9/9 |
| Kev 0.8B Q8 | 1,217.95 ms | 616.15 ms | 2.0486 | 1.9049–2.2143 | 9/9 |

All 84 full response byte strings and parsed numeric bits matched their frozen controls exactly. The archive preserves the complete original responses. The model files had the same hashes before and after the paired run.

The run checked child affinity on CPU 7 before `execve`. It did not record continuous scheduler residency. Host and SMT sibling activity appeared in telemetry snapshots, but those snapshots do not explain any individual timing or control result. This is not a quiet-host claim.

## Bounded kernel screen

A separate v2 micro screen ran three warmups and nine interleaved quartets with common baseline-calibrated repetition counts. Its baseline calibration target was 20 ms. It used real model weight fixtures with deterministic finite activations. It checked actual public output bits and canaries before and after measurement. The screen is one bounded paired sample, not a promotion decision.

The ratio is the median of nine paired baseline-to-candidate ratios. The range is the minimum and maximum of those nine ratios. Conventional arm medians use the average of the two middle samples. The paired ratio and ratio of independent medians can differ, especially for short, noisy controls.

| Case | T × N × K | Baseline median | Candidate median | Median paired ratio, baseline / candidate | Range across nine quartets | Candidate faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 memory | 158 × 1024 × 4096 | 58.692 ms | 25.951 ms | 2.2615 | 1.6910–2.5832 | 9/9 |
| F32 memory KV | 158 × 2048 × 1024 | 25.932 ms | 12.559 ms | 2.2017 | 1.8225–2.6341 | 9/9 |
| Q8 trunk output | 64 × 4096 × 4096 | 83.969 ms | 37.643 ms | 2.2123 | 2.0133–2.4478 | 9/9 |
| F32 single-row control | 1 × 1024 × 4096 | 414.702 µs | 401.986 µs | 0.9460 | 0.7262–1.2458 | 3/9 |
| Q8 single-row control | 1 × 4096 × 4096 | 8.109 ms | 8.383 ms | 0.9616 | 0.6558–1.0249 | 2/9 |
| F32 K32 control | 2 × 4 × 32 | 36.334 ns | 37.516 ns | 0.9865 | 0.9447–1.0892 | 4/9 |
| F32 K64 boundary | 2 × 4 × 64 | 53.329 ns | 33.569 ns | 1.5678 | 1.2621–2.0739 | 9/9 |
| F32 N3 control | 2 × 3 × 64 | 37.360 ns | 46.994 ns | 0.8203 | 0.7387–1.2442 | 2/9 |
| Q8 nonfinite K64 control | 2 × 4 × 64 | 150.814 ns | 134.026 ns | 1.2018 | 1.0760–1.2527 | 9/9 |

The single-row, K32, and N3 controls had mixed paired results. Their ranges cross 1. This screen does not establish no regression for every shape. No slowdown is attributed to a specific host process, shared SMT activity, or frequency effect.

## Numerical and code-generation gates

The frozen v1 callback attempt failed strict NaN payload and sign parity. It recorded 6 differing values in the default build and 949 in the globally AVX2-enabled build, all in nonfinite cases. Its code-generation isolation check also failed. The v1 failure and source are retained separately; those results are not pooled with v2.

The accepted v2 gate summary records zero differences across 118 configurations and 6,368 output values in each of the default and globally AVX2-enabled builds. It also records all canaries passing. V2 restores the original generic definition and uses a separate AVX2 loop. Selected normalized generic and globally AVX2 instruction sequences match after address and relocation normalization. That normalization preserves opcodes, registers, immediates, and other operands. It does not prove whole-binary equivalence.

In the selected CLI AVX2 loop, the recorded instruction count fell from 51 to 17. Both loops contain four multiply and four add instructions and eight YMM math instructions, with no stack references, calls, `vzeroupper`, or FMA in the inspected loop. This is a scoped code-generation result, not a whole-program instruction-count claim.

## Validation status

The original production-validation receipt records passing rustfmt, native and WASM Clippy, fixture-required release workspace tests, actual Kev cache integration, three shipping and three CPU benchmark WASM builds, ABI and calibration checks, and three numerical guards. The before and after Rust, runtime, fixture, and measured CLI hashes are unchanged. Three corrective numerical guards then passed with explicit artifact paths, the override unset, and before/after artifact hashes. Their receipt is `numerical-artifact-binding-v2-r1/receipt.json` (SHA-256 `3ad9b9a350cda8c88109e3404d94aa2f2ab6fbfdd2e0b69918245af5b863e53a`); amendment `numerical-binding-amendment-v2-r1.json` has SHA-256 `ce296ce68b0b50bff616142c1130a8d17fedbac01204e2c9e494dbd497611423`, and the binding inventory has SHA-256 `3732451842f13e3e64498f23cf6e268fe0cfd20e4e8561ebcb45239fe8d9b73d`. The independent artifact-binding audit passed. These checks do not expand the benchmark sample or establish general performance.

The [evidence archive](benchmarks/native-2x4-2026-11-07/README.md) contains the unchanged v1 failure, accepted v2 gate and micro results, paired samples and responses, exact source identities, original and corrective validation records, and derived statistics. It excludes model weights, measured activation and hidden-state buffers, compiled binaries, build trees, and credentials. The full micro disassemblies are represented by exact hash descriptors; relevant function extracts are included.
