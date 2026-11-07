# Native AVX2 automatic 4×4 linear path

On the Ryzen 9 9950X3D host, the native x86_64 default now selects a four-row AVX2 tile only for large projections. In a paired fresh-process comparison, this reduced measured Laya Q8 inference latency by **9.8%** and Kev 0.8B Q8 latency by **7.2%** versus the accepted native 2×4 default. Every full response matched its frozen control byte-for-byte and all parsed numeric values matched exactly.

This is a scoped result for two Q8 requests on one shared host. It is not an all-model or quiet-host claim. The candidate passed full production validation and an independent audit of its 55-file result inventory. The [evidence archive](benchmarks/native-4x4-2026-11-07/README.md) records the frozen sources, raw measurements, parity gates, and validation receipts.

## Selection and compatibility

The paired model capture used frozen campaign base `9d7ee8623aad5a1fdd65dde3484fa416fa3f14f4` and candidate kernel SHA-256 `a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da`. For release, those exact kernel bytes were transplanted unchanged onto merged commit `dd6c94a25cb707112ff7f146ceff756d1157cfd0`, which contains the accepted native AVX2 2×4 kernel. This separates the measured source identity from the release base.

In the default x86_64 build without global AVX2, the tile selector uses automatic mode. Runtime AVX2 calls use the four-row tile when `T >= 8`, `N >= 1024`, and `K >= 1024`; all other shapes retain the accepted two-row tile. Other targets keep their existing defaults: ARM uses four rows, and WebAssembly or globally AVX2-enabled builds use two. Explicit tile zero and tile one remain available. The generic, decoder, and arithmetic bodies remain unchanged. Nonfinite outputs continue to use the existing generic fallback.

The public tile getter now reports `2` for the native automatic default. On default x86_64 builds without global AVX2, `set_tile(2)` selects this automatic mode; `set_tile(0)` and `set_tile(1)` keep the explicit tile selections. Other targets retain their previous setter behavior. Callers that inspect or set the tile should account for this contract change.

## Model comparison

The test compared the candidate's real default against the accepted two-row default. It did not set a tile override. Each model used three discarded observations per arm, then nine alternating ABBA and BAAB quartets. Each quartet contains two baseline and two candidate requests. All slower quartets remain included.

The primary metric surrounds `Model::decide` in a fresh CLI process, after model loading and request parsing and before output serialization. Model load time was recorded separately. This is request latency after load, not warm resident latency or process wall time. Each child was checked for CPU 7 affinity before execution. Host telemetry does not establish continuous CPU residency or explain individual samples.

| Model | Baseline independent median | Candidate independent median | Median paired ratio, baseline / candidate | Paired range | Candidate wins | Paired latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya Q8 | 548.4 ms | 476.7 ms | 1.1085× | 0.9535–1.2847× | 8/9 | 9.79% |
| Kev 0.8B Q8 | 582.6 ms | 535.85 ms | 1.0772× | 0.9489–1.2123× | 8/9 | 7.17% |

The paired reduction is calculated from the median of the nine baseline-to-candidate quartet ratios. It is not calculated by dividing the two independent medians. The ranges include results on both sides of parity, so these bounded samples do not show a win in every quartet.

The capture recorded 84 complete CLI requests across both models: 12 discarded and 72 measured. All 84 response byte strings and 420 parsed numeric values matched their frozen controls. Model files, source files, helpers, binaries, and controls had matching before-and-after hashes. No retries or host changes were used.

## Kernel evidence

The candidate passed strict parity checks in both the default and globally AVX2-enabled builds: each covered 179 configurations and 297,954 output values with no differences, including nonfinite cases and canaries. The MSRV build check passed. The comparison retained exact output bits and checked the active public dispatch.

In the selected native CLI loop, the packed four-row core uses 27 instructions. Two accepted two-row cores use 17 instructions each. The inspected loops have no FMA or stack spills. This is a scoped assembly result, not a whole-program instruction count or proof that every request benefits.

An earlier tile-one prototype micro-screen showed larger gains on selected bulk shapes, with mixed results on small and nonfinite controls. Those results motivated further investigation; they are not the production-default model measurements above and were not used to calculate the reported latency reductions. The archive keeps the prototype and its controls distinct from the automatic-mode result.

## Validation status

Full production validation passed: all 16 frozen stages succeeded, including formatting, clippy, workspace release tests with the Kev cache fixture, six WebAssembly builds, both ABI checks, and three numerical guards bound to their generated artifacts. Each guard passed 1,408 cases. The receipt is SHA-256 `057a37da8fe56854ba569cb876ce7253a6dfc5dc4d842d1dc0e9a685ff2d8df6`; its exact 55-file result inventory is `d30cb21e6d815c0fdd3f8afabeb633f0e606a8a56410c6299ecb45ddf299a1c0`. Independent review verified all 55 inventory entries and the final execution audit, SHA-256 `7c15a11bcf4a9af5a0bbb59962aac38642f086285740ded130d5ec58e3d2baf4`.

The benchmark and validation packet do not establish the exact bytes delivered by GitHub Pages. Review the candidate's PR and deployment evidence separately before making a live-delivery claim.
