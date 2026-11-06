# Measured model and kernel optimization

Clef Flash Q8 used **6.1–9.1% less whole-request wall time** in the recorded Chromium and RTX 5070 Ti comparisons. A separate paired CPU-finish replay used **24.4–26.3% less CPU time** on identical captured inputs. The detailed timings appear below.

Raw distributions matched for 9 models, and recorded public outputs matched for Laya/Clef. The declared validation tolerance remained 0.0005. Native logits are not exported for Laya/Clef.

The complete catalog comparison is mixed. Laya and Kev 0.8B have slower sequential wall medians. These results remain visible. They are not conclusively attributed to noise. Qwen final-row pruning and Gemma long attention have scoped kernel evidence, without a whole-Qwen, whole-Gemma or camera/FPS claim.

[Public summary](benchmarks/model-optimization-2026-11-06/summary.json) contains source identities, all artifact hashes and links to exact deterministic gzip archives. The separate [profiling method](kernel-profiling.md) explains the measures.

## Method and source identity

The initial before/after sweeps measured eleven available Q8 models with three deterministic text shapes, nine samples after three warmups, and two unprofiled/profiled blocks per runtime. The 88 raw blocks are archived in full. Model loading, tuning and fixture hashing are excluded from request timings. GPU profiles use separate instrumented requests and must be read independently from wall latency. A separate resident paired control is archived without replacing the initial measurements.

The baseline commit is `44239322be01f283c00d41d1af2a6debdda08f24`. Its fingerprint is `013ceb532401b44fc7492b09ad49a16c44a4d1015fa2ce20ffa582ce86aae37a`. The candidate base commit is `44239322be01f283c00d41d1af2a6debdda08f24`. Its fingerprint is `69840234eb7434ff891c6c164bf5efb7088ac2d99e0131974c83ba7230c821e2`. The candidate was captured from an uncommitted worktree. Its 52 runtime asset hashes match the verified final source snapshot.

## Complete whole-request comparison

Negative changes mean lower measured latency. Every row has recorded output difference 0, with its probability source shown below. Raw distributions are available for 9 models. Laya/Clef expose rounded public noul. The unchanged declared tolerance is 0.0005. GPU totals are diagnostic, not a share of wall latency.

| Model | Output comparison | Request | Before wall, ms | After wall, ms | Wall change | Before GPU, ms | After GPU, ms |
|---|---|---|---:|---:|---:|---:|---:|
| laya | rounded public noul | 1 question, ~40 tokens | 6.20 | 7.10 | +14.5% | 4.10 | 4.09 |
| laya | rounded public noul | 1 question, ~128 tokens | 11.90 | 12.80 | +7.6% | 9.34 | 9.54 |
| laya | rounded public noul | 1 question, ~512 tokens | 31.50 | 33.50 | +6.3% | 25.32 | 25.23 |
| kev-0.8b | raw distribution | 1 question, ~40 tokens | 6.05 | 6.45 | +6.6% | 4.34 | 4.29 |
| kev-0.8b | raw distribution | 1 question, ~128 tokens | 16.80 | 18.40 | +9.5% | 14.42 | 14.42 |
| kev-0.8b | raw distribution | 1 question, ~512 tokens | 56.70 | 58.00 | +2.3% | 46.52 | 46.48 |
| kev-4b | raw distribution | 1 question, ~40 tokens | 23.65 | 23.35 | -1.3% | 20.67 | 20.63 |
| kev-4b | raw distribution | 1 question, ~128 tokens | 73.45 | 75.05 | +2.2% | 69.75 | 69.97 |
| kev-4b | raw distribution | 1 question, ~512 tokens | 269.05 | 273.35 | +1.6% | 255.06 | 255.46 |
| kev-9b | raw distribution | 1 question, ~40 tokens | 40.20 | 41.10 | +2.2% | 36.70 | 36.86 |
| kev-9b | raw distribution | 1 question, ~128 tokens | 129.75 | 131.30 | +1.2% | 124.52 | 126.27 |
| kev-9b | raw distribution | 1 question, ~512 tokens | 492.25 | 496.30 | +0.8% | 475.63 | 475.72 |
| h2o-lightning-4b | raw distribution | 1 question, ~40 tokens | 50.50 | 50.40 | -0.2% | 47.79 | 47.03 |
| h2o-lightning-4b | raw distribution | 1 question, ~128 tokens | 90.15 | 89.90 | -0.3% | 86.61 | 85.61 |
| h2o-lightning-4b | raw distribution | 1 question, ~512 tokens | 281.05 | 278.20 | -1.0% | 271.15 | 264.77 |
| clef-flash | rounded public noul | 1 question, ~40 tokens | 274.35 | 257.60 | -6.1% | 144.60 | 144.57 |
| clef-flash | rounded public noul | 1 question, ~128 tokens | 398.60 | 363.20 | -8.9% | 205.21 | 204.89 |
| clef-flash | rounded public noul | 1 question, ~512 tokens | 1024.80 | 931.25 | -9.1% | 560.02 | 561.24 |
| semif-qwen3.5-0.8b | raw distribution | 1 question, ~40 tokens | 14.80 | 14.80 | 0.0% | 11.89 | 11.87 |
| semif-qwen3.5-0.8b | raw distribution | 1 question, ~128 tokens | 22.85 | 23.10 | +1.1% | 20.05 | 20.03 |
| semif-qwen3.5-0.8b | raw distribution | 1 question, ~512 tokens | 58.75 | 60.10 | +2.3% | 50.09 | 49.62 |
| semif-qwen3.5-2b | raw distribution | 1 question, ~40 tokens | 27.60 | 28.10 | +1.8% | 24.96 | 24.45 |
| semif-qwen3.5-2b | raw distribution | 1 question, ~128 tokens | 47.80 | 48.55 | +1.6% | 44.21 | 44.07 |
| semif-qwen3.5-2b | raw distribution | 1 question, ~512 tokens | 122.55 | 123.20 | +0.5% | 110.54 | 110.28 |
| semif-qwen3.5-4b | raw distribution | 1 question, ~40 tokens | 62.65 | 63.30 | +1.0% | 58.95 | 59.04 |
| semif-qwen3.5-4b | raw distribution | 1 question, ~128 tokens | 112.20 | 112.15 | 0.0% | 107.27 | 107.49 |
| semif-qwen3.5-4b | raw distribution | 1 question, ~512 tokens | 287.25 | 287.20 | 0.0% | 273.21 | 272.65 |
| gemma-4-e2b | raw distribution | 1 question, ~40 tokens | 17.50 | 17.85 | +2.0% | 14.64 | 14.62 |
| gemma-4-e2b | raw distribution | 1 question, ~128 tokens | 26.10 | 26.50 | +1.5% | 22.95 | 22.97 |
| gemma-4-e2b | raw distribution | 1 question, ~512 tokens | 68.65 | 68.25 | -0.6% | 61.00 | 59.61 |
| gemma-4-e4b | raw distribution | 1 question, ~40 tokens | 44.40 | 43.75 | -1.5% | 40.24 | 40.64 |
| gemma-4-e4b | raw distribution | 1 question, ~128 tokens | 75.05 | 74.50 | -0.7% | 70.86 | 70.69 |
| gemma-4-e4b | raw distribution | 1 question, ~512 tokens | 193.65 | 192.50 | -0.6% | 185.68 | 184.08 |

## Resident paired control

The corrected control initializes and calibrates the two runtimes sequentially, then keeps both resident while requests alternate in ABBA/BAAB order. Selected recurrence, matrix and matvec policies must match for causal attribution. This separate experiment does not replace the initial sequential results. An earlier overlapping-load attempt selected different recurrence kernels and is archived as excluded evidence. The Before and After columns show medians of individual completed requests. Paired change uses the median candidate/baseline ratio after averaging each arm's two calls within a pair. It need not equal the change calculated from the two per-request medians. The ratio is dimensionless.

| Model | Request | Before wall, ms | After wall, ms | Paired change | Attribution gate |
|---|---|---:|---:|---:|---|
| laya | 1 question, ~40 tokens | 5.90 | 5.90 | +0.9% | matched tuning |
| laya | 1 question, ~128 tokens | 11.70 | 11.70 | +0.4% | matched tuning |
| laya | 1 question, ~512 tokens | 31.55 | 31.30 | 0.0% | matched tuning |
| kev-0.8b | 1 question, ~40 tokens | 6.25 | 6.25 | -0.8% | matched tuning |
| kev-0.8b | 1 question, ~128 tokens | 17.05 | 17.20 | +1.5% | matched tuning |
| kev-0.8b | 1 question, ~512 tokens | 55.50 | 55.65 | -0.4% | matched tuning |

## Scoped retained changes

### Clef CPU finish

The routing head projects only Q for query rows and K/V for memory rows. Self-attention computes shared Q/K/V projections once. A paired replay uses identical captured GPU-backbone outputs with fixed CPU tile and matching WASM identities. It times synchronous `kevala_clef_finish` only. Public serialized responses match exactly. Raw logits are not exported by this ABI.

| Request | Before CPU finish, ms | After CPU finish, ms | Change |
|---|---:|---:|---:|
| 1 question, ~40 tokens | 120.20 | 90.90 | -24.4% |
| 1 question, ~128 tokens | 181.10 | 135.60 | -25.1% |
| 1 question, ~512 tokens | 439.50 | 323.80 | -26.3% |

### Qwen final rows

The final block computes eligible MLP work for selected scorer rows while retaining the source-length split-K partition and exact gathered outputs. All four final Qwen guards cover 13 cases each: Q8 and BF16 in Chromium and Firefox. Their raw selected-row intermediates and native responses pass exact checks. Kernel/composite timing excludes uploads, copies and readback. Whole-model timings above remain mixed. Earlier v1 guards are archived separately and are not used as final-runtime timing. Cache-extension coverage remains unproven for fixtures whose native tokenizer did not produce a reusable extension.

A scoped example uses 161 source rows and 3 selected scorer rows. Values below cover the loaded-weight final-block operation, not whole-model requests.

| Browser | Precision | Before composite, ms | After composite, ms | Change |
|---|---|---:|---:|---:|
| chromium | q8 | 0.24 | 0.11 | -56.8% |
| firefox | q8 | 0.53 | 0.25 | -52.2% |
| chromium | bf16 | 0.23 | 0.10 | -55.6% |
| firefox | bf16 | 0.43 | 0.36 | -16.2% |

### Gemma long attention

The existing fixed 32-lane subgroup shader can serve causal, unbatched requests without a joined prefix at 257..640 tokens. Selection requires complete exact outputs before and after timing and stable gains of at least 5% at 257, 512 and 602 tokens. Selection is keyed by exact head dimension and window. Existing legacy calibration runs first. Incomplete long calibration retains the original kernel. The total 750 ms deadline remains, with 64 MiB bounded long scratch. Attention calibration across the recorded catalog loads increased from 82.7–90.9 ms to 499.7–612.3 ms. All 16 captures completed without truncation. This adds cold-start work that is excluded from the whole-request table. Two 32-case guards cover both head dimensions, KV-head counts, windows and token boundaries. No WGSL arithmetic or image budget changed.

At 602 tokens, H256, 8 query heads and window 512, the registered kernel examples are:

| KV heads | Before kernel, ms | After kernel, ms | Change |
|---:|---:|---:|---:|
| 1 | 0.87 | 0.78 | -9.9% |
| 2 | 0.85 | 0.76 | -10.4% |

The final supported Chromium GPU guard produced 56 archived receipts, with 3 capability skips and no reported failures. These supplementary numerical receipts do not promote additional timing claims.

## Rejected experiment

A synthetic chunk-completion fence passed its independent integer-output check but did not establish a robust timing gain. Its runtime implementation was reverted. Both browser receipts and a byte-reconstructible rejected-source snapshot are archived. It contributes no retained speedup.

## Limits

- Results apply to the recorded hardware, browser, pack hashes, runtime bytes and fixtures. They do not establish universally fastest kernels or model accuracy.
- The initial full catalog baseline and candidate were separate sequential campaigns. Host load changed. Mixed wall results, including Laya/Kev 0.8B slowdowns, remain slower observations without a conclusive noise attribution. The resident paired control is archived separately.
- GPU profiles are separate instrumented requests and change execution. GPU-time changes cannot be subtracted from wall latency or promoted to an application speedup.
- The whole-request table contains only eleven available Q8 models in Chromium, three deterministic text-request shapes, and two blocks per runtime. It does not establish BF16 catalog, worker/UI, batch, mobile, camera sensor-to-screen performance or FPS.
- Raw distributions matched for 9 models, and recorded public outputs matched for Laya/Clef. Each reported comparison difference was 0 at the unchanged declared tolerance 0.0005. Laya/Clef do not export native logits/raw distributions. Rounded-output equality does not establish logit equality. Numerical agreement is not an accuracy benchmark.
- Qwen final-block and Gemma attention improvements are scoped kernel/composite measurements. Their full-model request medians remain mixed. No whole-Qwen or whole-Gemma speedup is claimed.
- Earlier v1 Qwen Q8 guards are retained as historical numerical evidence. Final claims use the new candidate-v2 Q8 and BF16 guards. Failed poison/plumbing and preliminary captures do not contribute passing cases.
- Some Qwen cache-extension fixtures did not produce a reusable native-tokenizer extension. Exact responses passed, but extension-path coverage is not established where the raw guard names this limitation.
- The Clef CPU replay times only synchronous kevala_clef_finish on identical captured inputs. GPU backbone, prepare, input copies, hashes, loading and compilation are excluded. Raw logits are not exposed. Exactness covers serialized public responses.
- No model weights, coordinator sub-pack, hidden input arrays or lexical input arrays are included in the archive. Only their hashes/byte counts, source identities, fixture responses and raw timing/output checks are retained.
- The rejected completion-fence implementation is preserved as a source snapshot and raw experiment only. Its runtime change is absent from the final candidate.
- The initial resident-control attempt overlapped both engines' GPU calibration. Kev selected different recurrence kernels between arms, so those timings do not isolate the code change. Original captures and exact harness sources are archived as excluded control evidence. Corrected controls must serialize calibration and record load order.
