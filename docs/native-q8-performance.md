# Native Q8 AVX2 row dequantization

The timed candidate speeds up full Q8 row reads in an isolated native helper on one AMD Ryzen 9 9950X3D. This is row-decoding evidence only. It does not show a full model or application latency improvement.

## Timed results

The v2 candidate dispatches rows with at least 64 output values to an AVX2 helper. The table reports the median paired ratio as baseline time divided by candidate time. Values above 1 favor the candidate. Absolute times are conventional medians in nanoseconds per row, summarized as the median of 27 case-level medians across K, block size, and batch. These two medians use different aggregation, so they do not need to agree algebraically.

| Output width | Paired ratio | Baseline ns/row | Candidate ns/row | Slower paired cases |
|---:|---:|---:|---:|---:|
| 5 | 0.9554 | 8.041 | 8.137 | 23/27 |
| 17 | 0.9521 | 8.488 | 8.699 | 23/27 |
| 63 | 0.9896 | 17.324 | 17.736 | 14/27 |
| 64 | 1.1456 | 13.312 | 11.380 | 0/27 |
| 128 | 1.3198 | 20.971 | 15.865 | 0/27 |
| Full K | 1.5846 | 338.624 | 194.445 | 0/27 |

Full-width cases range from 1.2085× to 2.1688× across K=1024, 2560, and 4096; block sizes 16, 32, and 64; and batch sizes 1, 158, and 659. Widths 64 and 128 were faster in every paired case. Widths below 64 retained small mixed or slower results in this capture. The machine was busy, so these short-row differences need a less contended follow-up before they are treated as stable.

The initial v1 experiment dispatched every nonempty row. It was rejected: width 5 had a median paired ratio of 0.8185 and was slower in all 27 cases; width 17 had a median of 0.9064 and was slower in 25 of 27. V2 adds the 64-value threshold. V2 still did not establish a win below the threshold.

Absolute time fields in the original v1 and v2 summaries used the upper middle of 18 observations. [Corrected derived statistics](benchmarks/native-q8-avx2-2026-11-06/derived-statistics.json) recompute the conventional even-sample median from the original rounds. The original summaries and their raw samples remain unchanged in the evidence archive.

## Method and source identity

The helper reads synthetic Q8 rows for K=1024, 2560, and 4096, block sizes 16, 32, and 64, and batches of 1, 158, and 659. It warms each arm three times and measures nine alternating ABBA/BAAB pairs. Each round times four serial batches. Fixture preparation and complete output-bit comparisons happen outside the timer. The driver checks exact float bits before and after timing.

The process was pinned to CPU 7, where the observed governor was `performance`. The compiler used the default x86_64 target features, including SSE2, and the row helper used runtime AVX2 detection. The recorded load averages were 29.85, 42.35, and 39.9. CPU affinity does not reserve an idle core.

The v1 and v2 sources both derive from base commit `91c464557d78d6aa111df84d39545da27e99c343`. V1 is identified by patch SHA `797cb5874bc607b2db7fd7ad914cd1b9087ca5f1605cc8573341f506c1c0a128`; timed v2 uses patch SHA `5150155f7531b5e441fb1ecdb3946db06ab275032b9b22d656321d40e139b00a`. V2 calls the row helper once after the width threshold. Its recorded disassembly shows AVX2 sign extension, integer-to-float conversion, multiplication, and vector stores. Scalar handling remains for incomplete chunks.

V3 extends only the test scale set with positive and negative infinity plus quiet and signaling NaN payloads. The frozen v2-to-v3 parity receipt confirms that the production helper and dispatch code match the timed v2 source; `store.rs` also contains the added test code. V3 was not retimed. Focused and workspace release tests, formatting, and Clippy passed. The base, SIMD, and relaxed WASM builds passed on v2; those binaries were reused after the documented test-only v3 parity check, and the final ABI passed `check-wasm`. The optional Kev pack integration fixture was unavailable. This campaign did not measure complete native model requests. Non-AVX2 hardware was not tested.

## Guard and preflight records

The intentional lock-guard check ran with `KEVALA_PROFILE_LOCK_HELD` unset. The harness rejected it by panicking because the canonical lock path was absent. Release `panic=abort` produced `SIGABRT`, not the initially expected exit code 101. The corrected guard validation accepts and records that signal; the failed guard did not produce timing data.

An earlier toolchain discovery command also failed before compilation because the shell selected Homebrew Rust 1.98.1 and its Cargo rejected `+1.95.0`. Recorded builds then used absolute Rust 1.95.0 toolchain paths. The raw evidence preserves both records and the exact commands.

## Evidence and reproduction

The [machine-readable summary](benchmarks/native-q8-avx2-2026-11-06/summary.json) links the timing source, corrected derived statistics, and final validation source. The archives contain the exact v1 and v2 patches, benchmark harness and lockfiles, raw paired timings, build and test commands, host metadata, assembly inspection, source manifests, and final validation receipts. The v3 archive keeps the test-only patch separate from the timed v2 source. The [archive manifest](benchmarks/native-q8-avx2-2026-11-06/archive-manifest.json) records compressed and decoded tar hashes plus a SHA-256 and size for every archived file. No compiled binaries, model data, target directories, or WASM binaries are included.

- [Raw timing, command, assembly, and validation evidence](benchmarks/native-q8-avx2-2026-11-06/raw-evidence-v1-v2-v3.tar.gz)
- [Rejected v1 source and harness](benchmarks/native-q8-avx2-2026-11-06/source-v1-rejected.tar.gz)
- [Timed v2 source and harness](benchmarks/native-q8-avx2-2026-11-06/source-v2-timed.tar.gz)
- [Final v3 test-only source delta](benchmarks/native-q8-avx2-2026-11-06/source-v3-tests.tar.gz)

The source archives identify base commit `91c464557d78d6aa111df84d39545da27e99c343`, carry the exact patch and harness, and include full source hash manifests. Their build manifests record the pinned Rust toolchain and offline, locked commands. Timing uses the canonical profile lock.
