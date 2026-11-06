# Clef head projection performance

The staged GPU head produced **median paired latency reductions of 27–37%** across three Clef request lengths in three fresh browser processes on this host. All 81 paired quartets favored the staged path. This is a comparison with the same candidate binary's ordinary CPU head finish, on one shared GPU backbone. It does not establish a universal fastest implementation or a model-accuracy improvement.

The first two trials use frozen `staged-runtime-v2`, with explicit `clefHead: "auto"` selection; that runtime kept CPU as the default. The separate final trial uses signed, rebased source and `staged-runtime-v3`, which defaults to auto and validates during normal model-load warmup. Explicit `clefHead: "cpu"` remains CPU. The two runtime identities and their tables remain separate.

## What moved to the GPU

Clef-flash has hidden width 4096 and head width 1024. The change stages seven F32 projections in two GPU waves while keeping the existing head attention and readout on the CPU.

| Wave | Projection | Matrices | Input width K | Output width N |
|---|---|---:|---:|---:|
| 1 | Hidden sequence to memory M | 1 | 4096 | 1024 |
| 2 | Routing normalized-memory K/V | 2 | 1024 | 2048 |
| 2 | Decoder memory K/V | 4 | 1024 | 2048 |

The CPU makes a private copy of backbone hidden rows, applies the original final RMSNorm and head LayerNorm, and exposes normalized S. After GPU computation, M is copied back so the CPU can compute the two routing memory norms. The second wave uses those two inputs and resident GPU M for the four decoder projections. Its K/V results return to the CPU. Query projections, field self attention, softmax, weighted values, output projections, feedforward layers and scoring use the shared original head implementation.

The seven weight slices and six biases occupy about **64.05 MiB** on the GPU. Total head GPU buffers are bounded at **256 MiB**, including those static buffers and request scratch. The Rust staging area has a separate **256 MiB host scratch bound**. These bounds are additional to the backbone and the CPU-resident model store. The full head weights are not uploaded to the GPU.

## Exactness and fallback

The eligible path uses relaxed SIMD WASM, CPU tile 0, and explicit WGSL `fma`. Bulk rows preserve the CPU's four-lane accumulation and horizontal sum. Record-local odd rows preserve its two alternating accumulators. Bias is added in the original position, including the `+0` addition for an unbiased projection.

Static weights and biases are checked once. Each request checks S, M and the normalized M inputs. Nonzero X/W magnitudes must lie in `[2^-37, 2^21)` and nonzero bias magnitudes in `[2^-97, 2^21)`, with K at most 4096. Zero of either sign is allowed. These sufficient bounds exclude subnormal and overflowing intermediates. They do not prove that every WGSL implementation uses the same arithmetic, so a new staged head first compares every GPU output bit with the loaded-weight CPU reference. There is no tolerance gate.

Other WASM flavors, tiles, unsupported geometry, finite values outside the range, and allocation/budget limits use the ordinary CPU finish. A nonfinite GPU output, reference mismatch or device failure disables and releases the head GPU buffers. A finite range rejection falls back for that request. Caller hidden and lexical buffers remain intact. Stage IDs, phase checks and synchronous status checks before post-await writes protect cancellation and prepare/model replacement.

Both the runtime-v2 precision run and the final runtime-v3 repeat used the pinned local Q8 pack and T=158, 251 and 659. For each length they checked all seven projection inputs, CPU references and GPU outputs against the original CPU graph, then checked unrounded raw logits and the public response. Each run passed with no fallback and unchanged sources: zero differences in 10,936,320 input F32 words and 14,217,216 output F32 words per reference/output comparison across the three cases, plus six raw logits. These are exact finite-bit checks on the recorded fixtures, not a model-accuracy evaluation.

The final public-default probe omitted the `clefHead` option and mode overrides. Normal load selected the GPU head and made one staged warmup with `validate=1`. All three user requests then used staged execution with `validate=0`, reused that validation and matched the control responses. This establishes the final default and warmup behavior independently of the comparative timing harness.

Raw logits came from a separate QA-only WASM snapshot that adds RAM capture immediately before response rounding. Its original Rust fingerprint matches the production validation exactly; the saved diff adds only capture/reset code to two WASM files. The production API does not expose raw logits. Archive files retain hashes, ranges and parity metadata, not captured activation, weight or logit arrays.

## Complete-request method

The runner used Linux, headless Chromium **156.0.8078.4**, WebGPU and an NVIDIA GeForce RTX 5070 Ti (`nvidia blackwell`). Each trial used a fresh Selenium browser process and loaded one GPU model. Both arms used the same candidate WASM and the same backbone. The CPU arm selected the existing `kevala_clef_finish`; the auto arm used the new staged finish. This is not a comparison of different backbone binaries.

Each fixture has one record, one `noul` field and two options. The encoded lengths were 158, 251 and 659 tokens. There were three warmup calls per arm, then nine serial ABBA/BAAB quartets per fixture. Each quartet contains two CPU and two auto calls. Profiling, response caching and state caching were disabled. Resident weights, reusable scratch and the fixed-option lexical means were retained for both arms.

The measure is page `performance.now()` around the complete awaited `model.decide` call. It includes the backbone, head guards, projection transfers/readback, CPU continuation and response handling. Loading, setup, warmups, first CPU-reference validation, output hashing and comparisons are excluded. No kernel duration is subtracted from a request time.

The displayed time is the median of 18 individual requests per arm. Each paired ratio is the auto mean divided by the CPU mean within one quartet; the table reports the median of nine ratios. The ratio need not equal the ratio of the two displayed time medians. The three processes remain separate trials.

The first two trials used runtime-v2:

| Trial | Tokens | CPU finish arm, ms | Auto arm, ms | Median paired auto/CPU | Median paired reduction | Winning quartets |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 158 | 374.80 | 263.35 | 0.716151 | 28.4% | 9/9 |
| 1 | 251 | 460.85 | 314.20 | 0.670449 | 33.0% | 9/9 |
| 1 | 659 | 1220.10 | 774.05 | 0.640554 | 35.9% | 9/9 |
| 2 | 158 | 340.65 | 242.70 | 0.721602 | 27.8% | 9/9 |
| 2 | 251 | 463.90 | 334.10 | 0.711543 | 28.8% | 9/9 |
| 2 | 659 | 1123.50 | 725.60 | 0.632950 | 36.7% | 9/9 |

The exact final runtime-v3 trial used signed commit `be350ec53b774bbfff3733d690cc9689625a02e7`:

| Final trial | Tokens | CPU finish arm, ms | Auto arm, ms | Median paired auto/CPU | Median paired reduction | Winning quartets |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 158 | 348.25 | 258.40 | 0.730300 | 26.97% | 9/9 |
| 3 | 251 | 494.30 | 343.35 | 0.698756 | 30.12% | 9/9 |
| 3 | 659 | 1230.30 | 838.60 | 0.672034 | 32.80% | 9/9 |

All 324 timed calls had matching public-response hashes within their fixture. Counters confirmed that every CPU call used the ordinary finish and every auto call used the staged path without fallback. The ratios, medians and wins were recomputed from the retained quartet samples. The six runtime-v2 paired reductions span 27.84–36.71%; the three final reductions span 26.97–32.80%. The headline rounds their combined range without pooling trials.

Cold work remains material. Comparative model load took 44.907 s, 38.217 s and 33.472 s in the three processes. Their first auto warmup calls, which included first-reference validation and head setup, took **548.6 ms**, **513.0 ms** and **612.4 ms**. The comparative harness deliberately controls head modes and performs that first validation in its excluded auto warmup; the separate public-default probe proves validation during ordinary model-load warmup. All warmup observations are retained. They are excluded from the warm tables and are not isolated setup costs.

## Rejected attempts and coverage limits

The earlier non-relaxed shader spelled `acc + x * w`. On this Chrome/Blackwell combination, its output matched the fused reference, with zero differences, but differed from the unfused reference at **17/32** tail outputs and **315/512** bulk outputs in each of three repetitions. Explicit `fma` produced the same fused results. The unfused SIMD path was rejected; SIMD/base head execution remains on the CPU. The failed run, emitted shaders, exact diagnostic source and retained rejected source versions are part of the evidence.

The five central guard attempts are retained with their original classifications:

| Attempt | Result | Registered | Compiled | Numerically validated | Composite timed | Failed | Unsupported | Unmeasured |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | Rejected unfused arithmetic | 84 | 81 | 0 | 0 | 2 | 3 | 81 |
| v2 | Device startup failure | 84 | 0 | 0 | 0 | 84 | 0 | 84 |
| v3 | Metric transport failure | 84 | 81 | 0 | 0 | 2 | 3 | 81 |
| v4 | Device startup failure | 84 | 81 | 0 | 0 | 2 | 3 | 81 |
| v5 | Completed, partial catalog scope | 84 | 81 | 2 | 2 | 0 | 3 | 79 |

The v2 failure count reflects a failed startup probe and unknown capabilities, not 84 demonstrated arithmetic defects. The v3 runner rejected `invalid metricMs: None`; it is not a successful timing capture. The v5 guard numerically validated and timed the two new Clef kernels through composite fixtures. It recorded no isolated or profiled kernel timings. The three unsupported and 79 otherwise unmeasured catalog entries remain outside its numerical/timing claims. Compilation is not numerical equivalence or a catalog-wide performance result.

Startup attempts also reported `Failed to execute 'requestDevice' on 'GPUAdapter': A valid external Instance reference no longer exists.` A bounded diagnostic made 18 sequential requests in six fresh profiles: Vulkan had 9/9 successes and GL had 8/9, with the failed GL profile succeeding on its next two requests. That does not establish a compositor workaround or a root cause. The absence of a recorded GPU-process crash line does not prove the process stayed alive. No garbage-collection cause is asserted.

These results cover one browser, GPU, local pack, flavor, tile and fixture set. They do not establish other GPUs, browsers, concurrent workloads, cold-start improvement, catalog-wide optimality or a state-of-the-art ranking. Earlier synthetic seven-projection timings are retained as development evidence and are not substituted for complete-request results.

## Identities and reproduction

The first two trials' runtime fingerprint is `4cd7f4932ec08a27852512d4ec854bd18f967bdfe782b2bda520c21e76a4be70`; its Rust fingerprint is `91d83e5d3e5841f240c0ccc02adee746bf1fd3c65e122ba300b3da115633f589`, from a dirty worktree based on `91c464557d78d6aa111df84d39545da27e99c343`.

The final runtime fingerprint is `eb46c3119c1befb08f1e63bcd3499583ed6684af00d9d6094caa845ca7754bc6`. Its Rust fingerprint is `8b7f43f4247542966cf4ef1c92730a90e208ed7327456024da98940941b407ad`, at signed commit `be350ec53b774bbfff3733d690cc9689625a02e7`. Rust **1.95.0** built all three WASM flavors. Final formatting, workspace/all-target Clippy with warnings denied, the required-fixture release workspace suite, all three builds and the 54-export check passed. The real optional Kev cache integration was supplied through `KEVALA_KEV_PACK` and passed. That test is validation evidence, not a latency measurement.

| Measured artifact | SHA-256 |
|---|---|
| Runtime-v2 production relaxed WASM | `396c937ae3ad37c2dcc5552f841975d365141e8ef533dc8f884b298b13ef7686` |
| Runtime-v2 QA relaxed WASM | `c0c1421fdfa106ee80235bcf9234f1935a6fd81d88c85ed7f075454ddead25fe` |
| Final production relaxed WASM | `939f4c424900c251577b548bc61a9b7ccc4a7ef576468cf37967dd4dae2ef700` |
| Final QA relaxed WASM | `228df30e89a9b511c852de44c29d14e945c3dbfc6872c7ed309a2ea852804d9f` |
| Original CPU capture WASM | `d7898fa1fb7dcf0c83e44ae3eb5890f17d76065f2e231c67b3cb3628e94fa8bb` |
| Pinned pack header | `c09bba078c04d4053f4791cc557bebda7038250faf030d4153c5a96ffd7fcb19` |

The [evidence directory](benchmarks/clef-head-2026-11-06/README.md), [summary](benchmarks/clef-head-2026-11-06/summary.json) and [archive manifest](benchmarks/clef-head-2026-11-06/manifest.json) retain compressed, decoded and per-file hashes; exact harness/controller and executed-variant source bytes; runtime and binary SHA descriptors; QA base source and exact patch; validation commands; raw timing samples; and sanitized precision metadata. Compiled binaries, checkpoint weights, target/cache trees and private raw arrays are excluded.

Closure is required for the executed GPU experiments and final signed source/validation. A few overwritten, intermediate native/Node validation-only source revisions were not snapshotted. Their original hash descriptors and raw receipts remain archived, and the manifest lists those gaps explicitly. Missing bytes were not reconstructed. This does not weaken the numerical gates or final source closure.
