# Q8 row dequantization performance

The block-based Q8 row path reduced median paired time in all 24 repeated-call cases on this host, by 1.6–91.1%. The largest SIMD and base cases improved by only about 1.6%. The earlier single-call protocol recorded two slower WASM cases, +10.8% and +18.6%. Both protocols remain below. The separate Kev and Laya request controls showed small, mixed changes and do not establish a whole-model or catalog-wide speedup.

Measured on November 6, 2026, on an AMD Ryzen 9 9950X3D and Linux x86_64. The helper used native code and Node 24.21.0 with V8 13.6.233.17-node.53. Rust 1.95.0 built the helpers. Complete-request controls used Chromium 156.0.8078.4 with WebGPU on an NVIDIA GeForce RTX 5070 Ti.

## Change

Previously, `Store::row` calculated the Q8 block index and loaded its scale for each output value. The retained path loads one scale per block and uses the existing `dequant16` for complete 16-value chunks. A scalar loop handles each remaining tail. It preserves the ABI and storage layout and adds no temporary allocation. F32 and BF16 rows use their existing paths.

The baseline is commit `4f1bd8998441ebc5579d0bd23b027ec7b88e1e92`. The candidate is identified by its Rust source hash and frozen runtime fingerprint in the [summary](benchmarks/q8-row-dequant-2026-11-06/summary.json). The archive preserves the exact Rust sources and patch. Only the three WASM files differ in the two 52-asset runtime manifests. The report does not assign a commit hash to the uncommitted candidate.

## Shared helper method

Both protocols use the same synthetic 512-row Q8 table, block size 32 and deterministic row IDs. Each arm reads the same selected rows in one instance. The reference executes the old scalar formula. The candidate calls `Store::row`. Each shape has three warmup rounds and nine alternating AB/BA pairs. Allocation and correctness checks are outside timing.

Both check bitwise equality at output widths 5, 17, 63 and full width before timing. The WASM driver also compares every output bit after each timed pair. The native driver checks parity before timing. The Rust regression covers more block sizes, partial rows, extreme scales and untouched output tails. Store and dequantization math are identical between the protocols. Only the timing drivers and native timing loop changed.

The relaxed, SIMD and base WASM flavors use `+simd128,+relaxed-simd`, `+simd128,-relaxed-simd` and `-simd128,-relaxed-simd` respectively. The helper release profile uses optimization level 3, fat LTO, one code generation unit and abort-on-panic.

Each table reports median milliseconds per complete selected-row batch. The dimensionless ratio is the median of the nine optimized/scalar ratios. A negative change means less time. The ratio need not equal the ratio of the two displayed time medians. The protocols are not pooled or averaged.

## Protocol 1: unpinned single calls

This capture ran from 18:22:52 to 18:22:56 UTC. Each timed side ran one row batch. CPU affinity was unrestricted, and the host load average was about 32 on 32 logical CPUs. All six native cases and 16 of 18 WASM cases favored the optimized path. The relaxed and base cases at width 4096 with 659 rows were slower by 10.8% and 18.6%. These observations remain part of the evidence.

| Flavor | Row width | Selected rows | Scalar ms | Optimized ms | Paired ratio | Paired change |
|---|---:|---:|---:|---:|---:|---:|
| relaxed | 1024 | 158 | 0.2061 | 0.1604 | 0.6695 | -33.0% |
| relaxed | 1024 | 659 | 0.9062 | 0.2107 | 0.2398 | -76.0% |
| relaxed | 2560 | 158 | 0.5339 | 0.3073 | 0.4570 | -54.3% |
| relaxed | 2560 | 659 | 2.4701 | 2.1496 | 0.7223 | -27.8% |
| relaxed | 4096 | 158 | 0.8906 | 0.2787 | 0.2066 | -79.3% |
| relaxed | 4096 | 659 | 5.1268 | 5.2550 | 1.1079 | +10.8% |
| simd | 1024 | 158 | 0.2034 | 0.0540 | 0.2647 | -73.5% |
| simd | 1024 | 659 | 0.8682 | 0.3608 | 0.4253 | -57.5% |
| simd | 2560 | 158 | 0.5655 | 0.6119 | 0.8483 | -15.2% |
| simd | 2560 | 659 | 3.1181 | 2.1340 | 0.7561 | -24.4% |
| simd | 4096 | 158 | 0.8220 | 0.1419 | 0.1780 | -82.2% |
| simd | 4096 | 659 | 4.2523 | 3.3860 | 0.7738 | -22.6% |
| base | 1024 | 158 | 0.2064 | 0.1822 | 0.7377 | -26.2% |
| base | 1024 | 659 | 0.8782 | 0.3250 | 0.3455 | -65.5% |
| base | 2560 | 158 | 0.5431 | 0.4138 | 0.6884 | -31.2% |
| base | 2560 | 659 | 2.4030 | 2.2569 | 0.9690 | -3.1% |
| base | 4096 | 158 | 0.8111 | 0.2842 | 0.2928 | -70.7% |
| base | 4096 | 659 | 4.5316 | 4.5206 | 1.1862 | +18.6% |
| native | 1024 | 158 | 0.1888 | 0.0223 | 0.1124 | -88.8% |
| native | 1024 | 659 | 0.8695 | 0.0954 | 0.1113 | -88.9% |
| native | 2560 | 158 | 0.4613 | 0.0432 | 0.0927 | -90.7% |
| native | 2560 | 659 | 2.0468 | 0.7033 | 0.3123 | -68.8% |
| native | 4096 | 158 | 0.7773 | 0.0995 | 0.1285 | -87.1% |
| native | 4096 | 659 | 4.2038 | 3.5594 | 0.7973 | -20.3% |

## Protocol 2: CPU 8 and 32 repetitions

The follow-up ran from 18:30:03 to 18:30:32 UTC and applied the same method to all 24 cases. Each warmup round made 32 calls per arm. Each timed side also made 32 calls and reported elapsed time divided by 32. `taskset -c 8` fixed the process to CPU 8. A preceding one-second `/proc/stat` sample found no CPU below 25% busy. CPU 8 had the lowest sampled use at 36.7% and a 32 MiB L3 cache. The full selection data remains in the archive.

Repeated calls amortize scheduler slices, and fixed affinity limits migration. They also change cache reuse and measurement duration. CPU affinity does not reserve an idle CPU. The host load average ranged from about 24 to 28 during this capture. This protocol does not prove that scheduler noise caused the first protocol's regressions.

| Flavor | Row width | Selected rows | Scalar ms | Optimized ms | Paired ratio | Paired change |
|---|---:|---:|---:|---:|---:|---:|
| relaxed | 1024 | 158 | 0.2026 | 0.0363 | 0.1802 | -82.0% |
| relaxed | 1024 | 659 | 0.8405 | 0.1173 | 0.1422 | -85.8% |
| relaxed | 2560 | 158 | 0.4864 | 0.0599 | 0.1211 | -87.9% |
| relaxed | 2560 | 659 | 1.9378 | 0.2208 | 0.1006 | -89.9% |
| relaxed | 4096 | 158 | 0.7601 | 0.0743 | 0.0978 | -90.2% |
| relaxed | 4096 | 659 | 5.1660 | 4.2776 | 0.8999 | -10.0% |
| simd | 1024 | 158 | 0.2080 | 0.0370 | 0.1779 | -82.2% |
| simd | 1024 | 659 | 0.8569 | 0.1399 | 0.1652 | -83.5% |
| simd | 2560 | 158 | 0.5184 | 0.0780 | 0.1413 | -85.9% |
| simd | 2560 | 659 | 2.7425 | 0.9192 | 0.3090 | -69.1% |
| simd | 4096 | 158 | 0.7966 | 0.1118 | 0.1377 | -86.2% |
| simd | 4096 | 659 | 5.9655 | 5.7564 | 0.9839 | -1.6% |
| base | 1024 | 158 | 0.2047 | 0.0619 | 0.2973 | -70.3% |
| base | 1024 | 659 | 0.8640 | 0.3156 | 0.3470 | -65.3% |
| base | 2560 | 158 | 0.4942 | 0.1728 | 0.3510 | -64.9% |
| base | 2560 | 659 | 2.9968 | 1.6841 | 0.6725 | -32.7% |
| base | 4096 | 158 | 1.2674 | 0.3521 | 0.4044 | -59.6% |
| base | 4096 | 659 | 5.7026 | 5.6750 | 0.9841 | -1.6% |
| native | 1024 | 158 | 0.1920 | 0.0171 | 0.0885 | -91.1% |
| native | 1024 | 659 | 0.8115 | 0.0942 | 0.1042 | -89.6% |
| native | 2560 | 158 | 0.4775 | 0.0542 | 0.1073 | -89.3% |
| native | 2560 | 659 | 2.5339 | 0.4173 | 0.1647 | -83.5% |
| native | 4096 | 158 | 0.7727 | 0.0798 | 0.1014 | -89.9% |
| native | 4096 | 659 | 5.9672 | 5.3086 | 0.8888 | -11.1% |

All repeated-call median paired ratios favored the optimized arm. At width 4096 with 659 rows, the relaxed ratio was about 0.900 and the SIMD and base ratios were about 0.984. Those last two margins are small. This is scoped row-helper evidence, not a universal speed guarantee.

## Complete-request controls

These controls are unchanged by the repeated-call helper protocol. Both complete runtimes remained resident. The harness loaded and calibrated A, drained its GPU queue, then loaded and calibrated B. Each shape used three warmups and 20 measured pairs. Each pair made four serial calls in ABBA or BAAB order with identical request text across those calls. Text changed between pairs. All recorded GPU selections matched across arms and remained stable. Both arms used CPU kernel `2x4`, relaxed WASM and one thread. Caches and profiling were disabled.

The first Kev run used original A and optimized B. The repeat reversed those roles. The table always reports optimized/original. For the repeat, the report takes the reciprocal of each raw pair ratio, recomputes the median and swaps the physical time medians. It does not invert the reported median.

Times below are per-request medians. Each paired ratio uses the mean of two calls per side within each pair. Its change need not follow from the two displayed time medians. Actual token counts are shown.

| Model/run | Tokens | Original ms | Optimized ms | Paired ratio | Paired change |
|---|---:|---:|---:|---:|---:|
| kev-0.8b | 31–32 | 7.15 | 7.25 | 0.9861 | -1.4% |
| kev-0.8b | 124–125 | 19.10 | 18.55 | 0.9725 | -2.7% |
| kev-0.8b | 532–533 | 59.20 | 59.10 | 1.0042 | +0.4% |
| laya | 47 | 6.60 | 6.60 | 0.9813 | -1.9% |
| laya | 140 | 11.85 | 11.70 | 0.9873 | -1.3% |
| laya | 512 | 33.20 | 32.45 | 0.9831 | -1.7% |
| kev-0.8b (reversed arms) | 31–32 | 6.20 | 6.20 | 1.0000 | 0.0% |
| kev-0.8b (reversed arms) | 124–125 | 17.05 | 16.80 | 0.9881 | -1.2% |
| kev-0.8b (reversed arms) | 532–533 | 55.10 | 54.90 | 0.9968 | -0.3% |

Kev raw distributions matched exactly. Laya's recorded rounded public `noul` values matched exactly. The unchanged probability tolerance was 0.0005. Laya does not export raw logits or distributions in these captures.

## Limits and evidence

The host remained contended in both helper protocols. The first protocol includes two regressions, and the second uses repeated calls on one selected CPU. Their difference does not isolate one cause. The report retains all samples and makes no universal row-path or causal whole-model claim. Headless request times do not establish camera throughput or foreground rendering performance.

The first Kev harness repeated identical `cpuKernel` and `flavor` option values. The later version removes those duplicate keys and adds fixed-CPU metadata to partial diagnostics. Both use the same CPU settings. Both exact versions remain archived.

The [machine-readable summary](benchmarks/q8-row-dequant-2026-11-06/summary.json) keeps the two helper protocols separate, with raw times, per-pair ratios and source identities. All nine first-report archives remain byte-for-byte unchanged. An additional archive preserves the first report and summary. The new protocol includes its exact source, timestamps, CPU selection, commands and binary hashes.

Each artifact is deterministic gzip JSON. Exact input bytes are base64 with their own SHA-256 and byte count. Receipts give the decoded JSON SHA-256, gzip SHA-256 and both lengths. Archives contain no model weights or executable binaries. Earlier helper logs remain preceding evidence.

| Artifact | Compressed bytes |
|---|---:|
| [Original and optimized 52-asset runtime manifests](benchmarks/q8-row-dequant-2026-11-06/raw/runtime-manifests.json.gz) | 3,427 |
| [Protocol 1: helper metadata and exact NDJSON](benchmarks/q8-row-dequant-2026-11-06/raw/helper-final.json.gz) | 13,808 |
| [Preceding helper timing and build log bytes](benchmarks/q8-row-dequant-2026-11-06/raw/helper-preceding.json.gz) | 5,662 |
| [q8-row-kev08: exact result, capture and source bytes](benchmarks/q8-row-dequant-2026-11-06/raw/q8-row-kev08.json.gz) | 69,488 |
| [q8-row-laya: exact result, capture and source bytes](benchmarks/q8-row-dequant-2026-11-06/raw/q8-row-laya.json.gz) | 50,858 |
| [q8-row-kev08-reverse: exact result, capture and source bytes](benchmarks/q8-row-dequant-2026-11-06/raw/q8-row-kev08-reverse.json.gz) | 68,827 |
| [Exact original and optimized Rust source and patch](benchmarks/q8-row-dequant-2026-11-06/raw/rust-change.json.gz) | 8,185 |
| [Protocol 1: helper source and binary hashes](benchmarks/q8-row-dequant-2026-11-06/raw/helper-source.json.gz) | 9,532 |
| [Exact paired-control harness versions and source dependencies](benchmarks/q8-row-dequant-2026-11-06/raw/harness-source.json.gz) | 85,136 |
| [Protocol 2 metadata, exact NDJSON and affinity selection](benchmarks/q8-row-dequant-2026-11-06/raw/helper-amortized.json.gz) | 16,151 |
| [Protocol 2 helper, wrapper and report-builder source](benchmarks/q8-row-dequant-2026-11-06/raw/helper-amortized-source.json.gz) | 32,459 |
| [Exact first-protocol report and summary snapshots](benchmarks/q8-row-dequant-2026-11-06/raw/protocol1-report.json.gz) | 39,827 |
