# Performance follow-up

Q8 matrix projections remain the main GPU work across the catalog. Clef also has a substantial CPU finish, and Gemma retains attention and MLP work. This follow-up records the remaining bottlenecks, rejected experiments and a separate cached-state optimization. It does not claim that every kernel is the fastest possible.

## Exploratory catalog profile

The Chromium Q8 sweep covered all 11 catalog models, three request sizes and two wall/profile blocks per model. The archive includes all 44 raw blocks. Each block has three warmups and nine samples. The frozen baseline is `4f1bd8998441ebc5579d0bd23b027ec7b88e1e92`, with runtime fingerprint `69840234eb7434ff891c6c164bf5efb7088ac2d99e0131974c83ba7230c821e2`. The host was an AMD Ryzen 9 9950X3D with an NVIDIA GeForce RTX 5070 Ti on Linux. Chromium used WebGPU without shader-f16.

This table shows the longest text fixture for each model. GPU time is the median sum of instrumented kernel timestamps from a separate run. Kernel shares divide each label median by the sum of label medians. They are not shares of request wall time.

| Model | Tokens | Wall ms | GPU timestamp sum ms | Largest GPU labels and shares |
|---|---:|---:|---:|---|
| laya | 512 | 45.75 | 25.67 | `mm.wi` 33.9%, `mm.wqkv` 21.0%, `mm.wo2` 19.4% |
| kev-0.8b | 533 | 81.15 | 46.76 | `mm.gate_up` 24.6%, `mm.qkvz` 21.3%, `mm.down` 14.6% |
| kev-4b | 533 | 324.15 | 264.81 | `mm.gate_up` 36.9%, `mm.down` 21.0%, `mm.qkvz` 18.6% |
| kev-9b | 533 | 565.50 | 485.33 | `mm.gate_up` 42.6%, `mm.down` 23.4%, `mm.qkvz` 16.4% |
| h2o-lightning-4b | 594 | 313.95 | 268.98 | `mm.gate_up` 38.2%, `mm.down` 19.9%, `mm.qkvz` 19.0% |
| clef-flash | 660 | 1347.95 | 571.37 | `mm.gate_up` 43.2%, `mm.down` 23.4%, `mm.qkvz` 16.3% |
| semif-qwen3.5-0.8b | 607 | 80.10 | 50.63 | `mm.gate_up` 24.0%, `mm.qkvz` 21.2%, `mm.down` 14.0% |
| semif-qwen3.5-2b | 607 | 165.25 | 112.95 | `mm.gate_up` 35.6%, `mm.down` 21.2%, `mm.qkvz` 18.4% |
| semif-qwen3.5-4b | 607 | 336.25 | 281.67 | `mm.gate_up` 36.8%, `mm.down` 20.8%, `mm.qkvz` 18.4% |
| gemma-4-e2b | 602 | 82.75 | 61.17 | `mm.down` 19.3%, `attn` 17.7%, `mm.up` 14.7% |
| gemma-4-e4b | 602 | 217.55 | 190.04 | `mm.down` 24.0%, `mm.up` 21.3%, `mm.gate` 21.1% |

Laya spends most GPU time in its MLP and QKV projections. Qwen-derived models and the Clef backbone are dominated by gate/up, down and QKV projections. Small Qwen models also show recurrence as a material cost. Gemma has large gate/up/down projections and attention. Some Gemma subgroup labels ran in only nine of 18 samples because runtime calibration selected different optional paths.

The wall and GPU measurements cannot be subtracted to estimate CPU time. Profiling changes pass boundaries and can disable or alter optional paths. The host was contended, and earlier agent-owned builds overlapped parts of this campaign. The catalog is a guide for choosing experiments, not a causal before/after result or a stable model ranking. Output agreement is not a task-accuracy evaluation. Laya and Clef expose rounded public outputs, while nine models expose raw distributions.

## Next targets

1. Prioritize large Qwen gate/up and down projections, which dominate their recorded GPU work.
2. Target Laya's MLP input and QKV projections.
3. Measure Gemma MLP and attention changes by their actual token, window and head-dimension families.
4. Continue Clef CPU-finish work with real captured inputs, including both short and long requests.

Retain a candidate only after repeated paired complete-request measurements show a benefit and its numerical gates pass at unchanged tolerances. Fix the request shapes, source identities and kernel choices across arms. Isolated kernel wins are useful screens, but they do not replace complete-request validation.

## Experiments that were not retained

| Experiment | Evidence and decision |
|---|---|
| Q8 K64 staging | The current decoder's K64 variant was slower in every screened shape. It required 32 KiB workgroup storage. The screen and repeat remain archived. |
| Q8 signed-byte decoders | `unpack4xI8` improved the 512×24576×4096 screen by 5.8%, then only 2.6% in the repeat. Marlin results also varied. Neither met a repeatable 5% keeper threshold. No shader decoder change was retained. |
| Clef single-row projection | Four exact native row cases regressed. Full captured and synthetic finishes were mixed across shapes and WASM flavors. The candidate Rust source, patch and raw pairs remain archived. |
| Overlapping transfer and computation | Initial paired Kev/Laya results looked favorable, but reversing Laya's runtime roles did not reproduce a consistent gain. The experiment was not retained. |

The rejected matrix screen used seeded synthetic weights, full-output bit comparisons and sampled independent CPU checks. It was not a complete-model speed test. Clef finish captures exclude the GPU backbone and preserve hashes rather than hidden arrays. None of these negative trials is counted as an optimization win.

The separate [Q8 row dequantization report](q8-row-dequantization-performance.md) contains both row-helper protocols, including the two initial regressions and the repeated-call follow-up. Its small, mixed complete-request controls remain distinct from the isolated row measurements.

## Avoiding cached-state embedding work

The retained cached-state path defers native state embedding until the GPU planner determines that it needs state rows. Exact cache hits need only question-stage embedding. Both comparison arms include the Q8 row change. This is a separate comparison from the catalog baseline. The paired harness checks cold requests, exact hits, actual prefix extensions and cleared-cache fresh passes. Both runtime arms use CPU kernel `2x4`, relaxed WASM, one thread, sequential load/calibration and matched GPU selections. Twenty ABBA/BAAB pairs measure each two-question cached-state fixture.

The measured hit calls prove one fewer native stage-1 embedding call in the candidate. Both arms perform no state GPU rows and keep the same cache hit counters. Request wall-time medians and median paired ratios use different aggregation, so one cannot be derived from the other.

| Run | Cached fixture | Original ms | Candidate ms | Paired ratio | Paired change |
|---|---|---:|---:|---:|---:|
| Q8 first | 2 questions, short cached state | 13.15 | 13.10 | 1.0110 | +1.1% |
| Q8 first | 2 questions, medium cached state | 12.00 | 11.55 | 0.9916 | -0.8% |
| Q8 first | 2 questions, long cached state | 14.05 | 12.80 | 0.9187 | -8.1% |
| Q8 repeat | 2 questions, short cached state | 7.95 | 7.80 | 0.9812 | -1.9% |
| Q8 repeat | 2 questions, medium cached state | 8.65 | 8.45 | 0.9828 | -1.7% |
| Q8 repeat | 2 questions, long cached state | 10.50 | 9.40 | 0.9212 | -7.9% |

The first Q8 short fixture regressed by 1.1%, and all recorded rows remain visible. Candidate-versus-original maximum deltas were 0 for raw probabilities, public outputs and hidden values. The recorded hidden/output hashes match across arms for each cold, hit, extension and fresh phase. All measured public and raw outputs also match. The raw probability tolerance remains 0.0005 and the hidden-row tolerance remains 0.00001.

Within each runtime, the largest extension-versus-fresh differences were 3.8743e-7 for probabilities and 7.6294e-6 for hidden values. These differences were the same in both arms, with public output delta 0. The aggregate correctness maxima include these within-runtime comparisons and must not be read as candidate-versus-original divergence.

The BF16 campaign stopped at `hidden row differs: 0.0000152587890625` in the original arm's long extension-versus-fresh guard. The unchanged tolerance was 0.00001. The failure and partial trace are archived unchanged. They do not establish candidate A/B divergence, a complete BF16 timing campaign or a generalized BF16 speedup.

Cold and extension calls leave the candidate's native scratch at stage 1, while the original ends at stage 2. Cache hits end at stage 2. Deferred misses can retain stage-1 scratch until the next embedding. The BF16 trace recorded 196,608 more committed WASM bytes in the candidate. Linear-memory byte lengths are high-water values, not live-allocation or transient-peak measurements. This change does not establish lower peak memory, camera throughput or foreground rendering responsiveness.

## Evidence

The [profiling guide](kernel-profiling.md) provides reusable commands, serialization rules and numerical gates. The [summary](benchmarks/performance-follow-up-2026-11-06/summary.json) contains all request shapes, focused timing ratios, source fingerprints and artifact receipts. Gzip JSON archives preserve exact source/result/capture bytes as base64 with SHA-256 and byte counts. Each receipt includes decoded and compressed hashes and lengths. No model weights, hidden arrays or executable binaries are archived.

| Artifact | Compressed bytes |
|---|---:|
| [Exploratory Q8 catalog summary and 44 raw blocks](benchmarks/performance-follow-up-2026-11-06/raw/catalog.json.gz) | 790,494 |
| [q8-decode-screen: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/q8-decode-screen.json.gz) | 45,215 |
| [q8-decode-repeat: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/q8-decode-repeat.json.gz) | 29,921 |
| [clef-head-captured-relaxed: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/clef-head-captured-relaxed.json.gz) | 14,129 |
| [clef-wasm-synthetic-relaxed: exact synthetic finish result](benchmarks/performance-follow-up-2026-11-06/raw/clef-wasm-synthetic-relaxed.json.gz) | 3,839 |
| [clef-wasm-synthetic-simd: exact synthetic finish result](benchmarks/performance-follow-up-2026-11-06/raw/clef-wasm-synthetic-simd.json.gz) | 4,140 |
| [Rejected Clef native row and finish logs](benchmarks/performance-follow-up-2026-11-06/raw/clef-native.json.gz) | 7,831 |
| [overlap-kev08: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/overlap-kev08.json.gz) | 50,826 |
| [overlap-laya: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/overlap-laya.json.gz) | 34,077 |
| [overlap-laya-reverse: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/overlap-laya-reverse.json.gz) | 54,346 |
| [lazy-state-q8-valid: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/lazy-state-q8-valid.json.gz) | 132,036 |
| [lazy-state-q8-repeat: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/lazy-state-q8-repeat.json.gz) | 132,213 |
| [lazy-state-bf16: exact result, capture and source](benchmarks/performance-follow-up-2026-11-06/raw/lazy-state-bf16.json.gz) | 62,303 |
| [Exact cached-state JS and test change sources](benchmarks/performance-follow-up-2026-11-06/raw/lazy-change.json.gz) | 51,361 |
| [Exact deduplicated source bytes and original path aliases](benchmarks/performance-follow-up-2026-11-06/raw/source.json.gz) | 492,361 |
