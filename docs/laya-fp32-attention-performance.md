# Laya FP32 attention performance

The FP32 query tile improves isolated Laya attention timings and reduces Laya's full-model GPU aggregate in the measured Chrome run. The full-model comparison uses two complete baseline runs and two candidate runs. The benchmark used local served builds, so these measurements do not describe a deployed site.

Measured on an NVIDIA RTX 5070 Ti (16 GB, driver 595.71.05), Ryzen 9 9950X3D, Ubuntu, and Chrome 149.0.7827.200 through the NVIDIA Vulkan ICD. The production baseline checkout is `d7a07d7`, whose tree is identical to merged revision `3ff550c0`. Its parity page received only the probability-recording diagnostic used by the candidate. Unshipped paired-projection fusion is excluded.

## Full-model comparison

Each case uses the median of five samples after two warmups. The Laya metric is the geometric mean of summed GPU kernel times for 47, 140, and 512 input tokens. Baseline and candidate values below are geometric means across both complete runs for that version.

| Model | Baseline GPU ms | Candidate GPU ms | Change |
| --- | ---: | ---: | ---: |
| Laya | 10.735 | 10.126 | 5.67% lower |
| Kev 0.8B control | 14.846 | 14.684 | 1.10% lower |
| SemIf Qwen3.5 2B control | 52.510 | 52.962 | 0.86% higher |
| Gemma 4 E2B control | 34.387 | 34.472 | 0.25% higher |
| Four-family geometric mean | 23.161 | 22.826 | 1.45% lower |

The four-family result is descriptive. The three other model paths are controls, and their timing variation is not attributed to this Laya change.

| Laya tokens | Summed GPU kernels, ms | Attention, ms | Browser p50, ms |
| ---: | ---: | ---: | ---: |
| 47 | 4.232 → 4.185 | 0.393 → 0.349 | 7.594 → 7.597 |
| 140 | 9.917 → 9.450 | 1.092 → 0.782 | 13.342 → 12.345 |
| 512 | 29.475 → 26.255 | 5.547 → 2.259 | 34.450 → 31.099 |

Each pair in this table is the geometric mean across the two run medians. The browser p50 includes browser and request overhead; the 47-token case did not improve in that measurement even though its attention kernel time fell.

## Attention microbenchmark

The final Chrome microbenchmark used WebGPU timestamp queries, three warmups, and 15 timed samples per case. It ran eight global or windowed shapes against the existing `attention_subgroup` kernel. The tile was faster in all eight cases, with a geometric mean speedup of **1.67×**. Earlier runs with the two kernel orders reversed also improved all eight cases, with geometric means of 1.67× and 1.68×.

| Shape | Existing kernel, ms | FP32 tile, ms | Speedup |
| --- | ---: | ---: | ---: |
| 47 tokens, global | 0.010599 | 0.009615 | 1.10× |
| 47 tokens, window 64 | 0.010674 | 0.009294 | 1.15× |
| 140 tokens, global | 0.040481 | 0.023549 | 1.72× |
| 140 tokens, window 64 | 0.029308 | 0.023497 | 1.25× |
| 512 tokens, global | 0.301928 | 0.113336 | 2.66× |
| 512 tokens, window 64 | 0.094716 | 0.044049 | 2.15× |
| 47/140/512 tokens, global | 0.340832 | 0.151029 | 2.26× |
| 47/140/512 tokens, window 64 | 0.116905 | 0.068262 | 1.71× |

The slash-separated cases are three segments in one batch. The six forced shared-reduction guard cases passed the same CPU-reference limit; they validate that WGSL specialization but are not a runtime fallback for devices that lack `subgroup_id`.

## Selection and correctness

The FP32 path uses a 64-query by 16-key tile and 30,224 bytes of workgroup storage. The runtime selects it only when the existing FP16 attention tile was not selected, the adapter supports 32-lane subgroups, workgroup storage is at least 30,224 bytes, and WGSL advertises `subgroup_id`. Devices that do not meet these conditions keep the existing subgroup or generic attention selection. The existing native-FP16 selection remains ahead of this branch and is unchanged.

The final microbenchmark passed its CPU-reference guard at 0.002. Its maximum FP32-tile error against that reference was `2.3047693e-7`; the largest paired tile-versus-existing-kernel output difference was `3.8743019e-7`.

Both Chrome and Firefox Laya fixture guards retained 41/41 argmax decisions in baseline and candidate. Within each browser, all 126 candidate probabilities were exactly equal to baseline probabilities. This does not mean either version exactly matches the reference probabilities. Chrome's reference `maxDp` is `0.0240064442`, above the strict `0.024` guard, in both baseline and candidate. Firefox is `0.0235064442` in both versions and passes that guard. The Chrome strict-guard miss is unchanged and is not reported as a pass.

The candidate is based on `3ff550c0f7cdafaca2575be438c2409030d092d0` with five scoped source changes. The machine-readable [measurement artifact](benchmarks/laya-fp32-attention-linux-2026-09-23.json) contains the raw microbenchmarks, both full-suite repeats per version, browser parity outputs, and local validation results.

## Local validation

- All three WASM flavors built and validated.
- `cargo test --release --workspace`: 64 tests passed.
- `pnpm test`: 103 tests passed.
- `pnpm check`: 77 JavaScript files checked.
- `pnpm stage:site` and `git diff --check` passed.
