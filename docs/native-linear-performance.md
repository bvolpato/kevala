# Native AVX2 linear dispatch

The default x86-64 build now uses an AVX2 version of the existing bulk linear kernel on eligible CPUs and matrix shapes. On the measured Ryzen 9 9950X3D host, the final median paired inference-time reductions were 17.51% for Laya Q8 and 14.94% for Kev 0.8B Q8. Every complete public response matched its baseline control exactly. These results cover one short request per model on one host.

The [evidence summary](benchmarks/native-linear-2026-11-06/summary.json), [archive manifest](benchmarks/native-linear-2026-11-06/manifest.json), and [archive format](benchmarks/native-linear-2026-11-06/README.md) preserve the final measurements, source, validation, assembly extracts, and rejected experiments.

## Dispatch and numerical behavior

The change moves the original arithmetic body into an always-inlined helper and calls it once through an `#[target_feature(enable = "avx2")]` wrapper. The body remains byte-identical. The compiler can pack pairs of four-lane operations into eight-lane instructions without adding FMA or changing the reduction order. F32, BF16, and Q8 matrices retain their existing decoding and tile behavior.

The runtime AVX2 path requires all of the following:

- An x86-64 build without globally enabled AVX2.
- At least two input rows, four output columns, and a reduction width of 64.
- Runtime AVX2 support.

The geometry checks run before feature detection. Other shapes use the original body. A build already compiled with `-C target-feature=+avx2` also uses the original body directly, without a redundant dispatch wrapper. The change does not set global compiler flags or require FMA.

After an accelerated call, the wrapper scans the output for nonfinite values. If any output is nonfinite, it recomputes the whole call through the original body outside the AVX2 target context. This preserves the observed original NaN payload and sign behavior. The successful finite path includes the output scan in the reported timings.

## Native inference calls

Both arms used the same common base, Rust 1.95.0, existing model packs, and CPU 7 affinity. Every observation started a fresh CLI process and loaded the model. The CLI timer surrounds `Model::decide`, after model load and state/questions JSON parsing, and before output JSON serialization and stdout. It includes preparation, the complete native forward pass, and response construction within `Model::decide`. Model loading, process wall time, and peak RSS are recorded separately. This is inference time in a fresh process, rather than end-to-end CLI/process time or warm resident latency.

Each model had three discarded observations per arm followed by nine alternating ABBA/BAAB quartets. A quartet ratio is the sum of its two baseline inference times divided by the sum of its two candidate times. The reported ratio is the median of the nine quartet ratios; latency reduction is `100 * (1 - 1 / ratio)`. Independent arm medians are shown separately.

| Model | Reported input/output tokens | Baseline median | Candidate median | Paired ratio | Paired reduction | Winning quartets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya Q8 | 40 / 0 | 1,748.05 ms | 1,443.15 ms | 1.21225 | 17.51% | 9/9 |
| Kev 0.8B Q8 | 23 / 22 | 1,499.10 ms | 1,274.90 ms | 1.17561 | 14.94% | 9/9 |

The paired ratios ranged from 1.09484 to 1.26018 for Laya and from 1.14648 to 1.29024 for Kev. All 72 measured and 12 discarded complete responses matched the controls, including every returned numeric value. There were no retries or process-cap failures. Full model-file and executable hashes were unchanged before and after the protocol. The CLI inference timer has 0.1 ms resolution. The usage field `output_tokens` counts tokens in serialized answers, rather than generated continuation tokens. Kev's 23 input tokens include the state and question branches; its 22 output tokens describe answer accounting.

The exact median paired reductions are 17.50889261251727% and 14.938061695409289%, respectively. The earlier four-call pilot is retained as a correctness and protocol control. Its single observations do not support a separate speed claim. No longer requests, additional models, or Clef 9B run are included, and these results do not rank models or establish performance on other CPUs.

## Linear kernel screen and controls

The final common-base screen used three warmups, nine alternating ABBA rounds, identical repetition counts in both arms, and baseline calibration targeting 20 ms. The first three cases used real matrix weights with deterministic finite synthetic activations. All output bits and canaries passed. The weights and activation bytes are excluded from the published archive. The table uses [corrected derived statistics](benchmarks/native-linear-2026-11-06/derived-statistics.json): each absolute median is the average of the two middle values among 18 arm observations. The original raw summary fields named `baselineMedianNsPerCall` and `candidateMedianNsPerCall` contain the upper-middle value, at sorted index 9. Those raw bytes remain unchanged. The nine-round paired medians are unaffected.

| Case | T × N × K | Baseline median | Candidate median | Paired ratio | Winning rounds |
| --- | --- | ---: | ---: | ---: | ---: |
| F32 memory projection | 158 × 1024 × 4096 | 92.164 ms | 68.654 ms | 1.3423 | 9/9 |
| F32 memory KV | 158 × 2048 × 1024 | 42.813 ms | 28.576 ms | 1.5107 | 9/9 |
| Q8 trunk output | 64 × 4096 × 4096 | 127.687 ms | 94.488 ms | 1.3324 | 9/9 |
| F32 single row | 1 × 1024 × 4096 | 441.530 µs | 467.081 µs | 1.0621 | 5/9 |
| Q8 single row | 1 × 4096 × 4096 | 7.013 ms | 7.370 ms | 0.9876 | 4/9 |
| F32 K32 control | 2 × 4 × 32 | 43.80 ns | 43.86 ns | 0.9006 | 4/9 |
| F32 K64 boundary | 2 × 4 × 64 | 68.52 ns | 58.73 ns | 1.2251 | 9/9 |
| F32 N3 control | 2 × 3 × 64 | 45.35 ns | 48.79 ns | 1.0120 | 5/9 |

The controls remain on the original arithmetic path. They are noisy and do not establish a stable no-regression result. In particular, the K32 control's paired ratio ranged from 0.7767 to 1.1235. Ratios of independent medians differ from medians of paired ratios, which explains apparently conflicting control columns. The K64 boundary won all nine rounds; the threshold avoids dispatching the smaller K32 case. Frequency counters and cross-hardware behavior were not measured.

## Validation and rejected evidence

The final source is based on `4c659292a9aafcb4b7abac91c94b11455b37792e`. Its kernel file SHA-256 is `4ef34fe7ac7ac6dc1508bdfcc4085f6fb99f2d7157879d183543b65e6038adb9`. Final native formatting, native and WASM Clippy, fixture-required release workspace tests, the real Kev cache integration, Rust 1.85.1 checks, default/global-AVX2 kernel tests, both CLI builds, and all three shipping and numerical-harness WASM flavors passed. The archive retains the commands, outputs, and binary hash descriptors.

Separate opaque callers compared the **actual baseline public function** with the **actual candidate public function** in default and globally AVX2-enabled builds. Each compared 5,484 output bit patterns across 96 configurations, including signed zero, subnormals, infinities, NaN payloads, unaligned inputs, bias, both tiles, and row/column tails. Each included 30 eligible cases with nonfinite outputs. Both passed every bit and canary check.

Earlier v1/v2/v3 experiments and failures remain separate from the final claims. The first attempt failed strict nonfinite parity. A later globally AVX2-enabled test compared against a private inlined oracle and reported `0xffc05678` versus `0x7fc01234`. That oracle had a different compiler context and was not a valid substitute for the original public function's NaN behavior. The failure was retained and diagnosed with the independent public-function comparison; the bit tolerance was not relaxed. A failed source-archive attempt and its repair receipts are also retained.

The final actual-CLI assembly check found a 71-instruction XMM hot loop with eight multiply and eight add instructions in the baseline, versus a 51-instruction AVX2 loop with four multiply and four add instructions in the candidate. The checked hot loop had no FMA, spills, calls, or `vzeroupper`. The generic path contained no VEX/YMM instructions, and globally AVX2-enabled builds had no redundant dispatch wrapper. Complete relevant function listings, checked hot-loop windows, extraction source, commands, and binary hashes are published. Whole-CLI disassembly remains local with size/SHA descriptors.
