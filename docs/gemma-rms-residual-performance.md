# Gemma normalization and residual fusion

Gemma applies weighted RMS normalization to each attention, FFN, and PLE branch,
then adds the result to the residual stream. The RMS shader now supports that
addition directly. This removes three dispatches per layer and the intermediate
branch writes and reads, including in the compact final layers.

The full and compact normalized-branch buffers are also removed. For an E2B
request with 601 tokens, the allocated capacities of 1024 and 64 rows save
6.375 MiB. The standalone layer scalar remains separate. Weight formats,
quantization, matrix kernels, and model selection are unchanged.

## Measurements

Measured serially on an NVIDIA RTX 5070 Ti, driver 595.71.05, Ubuntu, Chrome
149.0.7827.200, using hardware WebGPU and precise timestamps in a dedicated
headless browser. The baseline is main at `3d6de079`. Both versions exclude the
experimental paired-projection fusion.

Each value is the geometric mean of median GPU time across E2B requests of 99,
193, and 601 tokens, with two warmups and five measured requests per size.
These are profiled GPU times, not unprofiled round-trip latency.

| Run | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Fresh adjacent Chrome comparison | 35.935 ms | 35.556 ms | 1.05% |
| Initial production Chrome suite | 36.175 ms | 35.430 ms | 2.06% |
| Firefox 152.0.3 regression suite | 49.537 ms | 48.607 ms | 1.88% |

The initial Chrome profile shows a 19–21% reduction in the combined RMS and
residual operation time at each request size. Whole-model gains are modest and
subject to timing variation. The fresh adjacent pair gives the more conservative
estimate. This PR changes only Gemma; timing differences for Laya, Kev, and SemIf
are regression measurements, not speedups attributable to this patch.

Full samples, profiles, numerical comparisons, and source/WASM hashes are in the
[benchmark artifact](benchmarks/gemma-rms-residual-linux-2026-09-23.json). E4B has
correctness coverage and kernel timings at its actual hidden width of 2560, but
no whole-model throughput measurement in this experiment. Measurements cover
one GPU.

## Correctness and validation

- Chrome and Firefox each pass 45 full-array comparisons of separate and fused
  operations, with zero differing bits. These cover widths 32, 257, 1536, 2048,
  and 2560; 1, 8, and 65 rows; and signed, zero, and large inputs.
- An independent f64 reference has maximum absolute error
  `0.000015925447712561436`, below the existing `0.00002` bound. The paired kernel
  paths come from the candidate module; the CPU reference checks their math
  independently.
- Full and selected Gemma hidden states match exactly at 1, 31, and 65 tokens in
  both browsers. Duplicate selections, empty selections, and the configuration
  without shared KV layers also pass.
- All nine model packs retain 132/132 reference decisions. All 225 recorded Kev,
  SemIf, and Gemma probabilities exactly match the previous production revision,
  with matching Qwen recurrence selection. This includes all 60 Gemma probabilities.
- Chrome Laya retains its existing maximum probability difference of
  `0.024006444195624588`, above its unchanged `0.024` threshold. That strict guard
  remains failing. Its runner does not record individual probabilities.
- Both Gemma sizes pass the existing Firefox model guards, totaling 24 decisions.
- 64 Rust tests, 103 JavaScript tests, 76 JavaScript syntax checks, three WASM
  validations, and site staging pass. Independent review found no correctness issue.

Bit equality is measured evidence for these browser/GPU combinations. WGSL
permits floating-point reassociation and fusion, so expression order alone does
not guarantee it on every compiler. The implementation keeps the existing
normalization reduction and arithmetic, with one additional residual read and
addition per output element. Generated WASM remains an untracked build artifact.

The focused browser check is available at `dev/gemma-rms-check.html`. It requires
a hardware WebGPU adapter and publishes its results as `window.gpuBench` for
the existing `scripts/bench-gpu.py --result gpu` runner.
