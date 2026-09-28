# M4 Max kernel sweep

This sweep starts from `b2f9d956aa41d6b7992e623f70a2ba9f7d1c2723`, after the
[Metal numerical repairs](kernel-safety.md). It does not compare against the
earlier, numerically broken Gemma or Qwen kernels.

Hardware: Apple M4 Max, 40 GPU cores, 64 GiB unified memory, macOS 26.4.1.
Browser: Chrome for Testing 151.0.7922.34, WebGPU/Metal. The browser exposes the
adapter as `apple metal-3`. Measurements use hidden targets in the shared agent
Chrome, not newly launched browsers.

[Raw samples, shader fingerprints, experiment patches and validation outputs](benchmarks/m4-max-kernels-2026-09-25.json)
cover all 11 models and all 35 registered kernels. The
[model recheck](benchmarks/m4-max-model-recheck-2026-09-25.json) adds ten complete
model reruns and a separate fresh-target Kev-9B comparison. The measured build
contains uncommitted changes; its served JS/WASM fingerprint is
`63ea8225c433f17b0277ec8fa30da1892412758ed8cbb645335aed13aadc1695`.

## End-to-end results

The repeatable model-level win is Laya's 140-token request: mean awaited latency
falls **16.3%, from 24.78 to 20.74 ms** in the recheck. Median GPU time falls from
23.23 to 19.60 ms. This independently reproduces the initial capture's 14.9%
reduction. Its short and 512-token requests have no resolved improvement.

The table reports the current build at the middle input size, with 18 measured
requests per model and metric. Ten rows use follow-up measurements. Gemma 4 E4B
uses its complete initial same-build capture, explicitly labeled below.
Wall latency is an unprofiled arithmetic mean. GPU time is a separately
instrumented median, so the two columns must not be subtracted to estimate CPU
overhead. All three input sizes, per-request token counts, raw samples and
per-kernel profiles are in the linked artifacts. These are latency observations,
not before/after speedup claims for every model.

| Model | Tokens | Mean wall, ms | Profiled median GPU, ms | Capture |
| --- | ---: | ---: | ---: | --- |
| Laya | 140 | 20.74 | 19.60 | Recheck |
| Bruv1-0.8B | 198–199 | 44.21 | 42.60 | Recheck |
| Bruv1-4B | 198–199 | 411.18 | 327.42 | Recheck |
| Kev-0.8B | 124–125 | 28.72 | 27.49 | Recheck |
| Kev-4B | 124–125 | 218.84 | 188.91 | Recheck |
| Kev-9B | 124–125 | 471.72 | 500.53 | Recheck |
| SemIf-0.8B | 198–199 | 59.27 | 75.20 | Recheck |
| SemIf-2B | 198–199 | 108.16 | 107.87 | Recheck |
| SemIf-4B | 198–199 | 336.01 | 363.23 | Recheck |
| Gemma 4 E2B | 193–194 | 68.00 | 91.91 | Recheck |
| Gemma 4 E4B | 193–194 | 211.20 | 268.73 | Initial same-build sweep |

Speedup claims require both builds' block medians to agree within 10%, at least
5% pooled-median change, and at least 5% separation between both block ranges.
These are predefined repeatability checks, not statistical confidence intervals.
No cross-model geometric-mean speedup is claimed.

## Remaining bottlenecks

In the initial capture, matrix projections account for approximately **81–96%**
of profiled GPU time in most model/input cases. The exception is long Gemma 4 E2B: matrices
account for 63%, leaving attention as another material target. The small
normalization gains cannot produce a large universal model speedup.

The existing tiled Laya/Qwen attention and shape-dependent recurrent selectors
remain important. Forcing one variant globally loses on other shapes. Further
matrix work needs shape-specific evidence or exposure of native matrix
operations through WebGPU. Query/key reuse in long Gemma attention is worth a
separate measured experiment; it is not an improvement claimed in this build.

## Retained changes

### Avoid padded FP16 matrix rows

The generic matrix kernel now uses a 48-row tile instead of a 64-row tile when
both tiles produce the same number of workgroups. This applies to 65–96 and
129–144 tokens. The column tile, split-K partition and arithmetic order remain
unchanged. The already-optimized FP32 behavior is preserved. The wide FP16
kernel keeps its existing row selection, so this does not discard its
NVIDIA-oriented unrolled path.

Paired measurements use 11 samples, four warmups, rotating variant order, and
approximately 20 GFLOP per batch. Times include split-K reduction when present.

| Tokens × output width × input width | R4, ms | R3, ms | Reduction |
| --- | ---: | ---: | ---: |
| 65 × 5248 × 1024 | 0.20339 | 0.16949 | 16.7% |
| 80 × 5248 × 1024 | 0.21085 | 0.17381 | 17.6% |
| 96 × 5248 × 1024 | 0.21730 | 0.17591 | 19.0% |
| 129 × 3072 × 1024 | 0.18874 | 0.15466 | 18.1% |
| 140 × 3072 × 1024 | 0.19091 | 0.15957 | 16.4% |
| 144 × 3072 × 1024 | 0.18767 | 0.15788 | 15.9% |

Every output bit matches the R4 control in these six shapes and three additional
column-tail fixtures. The small fixture also covers bias, residual addition and
ReLU epilogues. Independent CPU comparisons remain enabled. This is a win for
these padded shapes, not a claim that all matrices improve by 19%.

### Finish row reductions within subgroups

Laya layer norm, Qwen RMSNorm and attention preparation, and Gemma RMSNorm and
PLE retain the shared-memory reduction down to 32 entries. Each subgroup then
reads that same immutable prefix through `subgroup_invocation_id`, performs an
explicit ordered shuffle tree, and broadcasts its result privately.

This removes five workgroup barriers per reduction without using an unspecified
`subgroupAdd` summation order or assuming that local invocation indices match
physical subgroup lanes. Subgroups never write the prefix back. Laya retains a
final workgroup barrier before its second reduction can overwrite the scratch.
The host enables this path only for verified 32-lane subgroups. Devices without
them retain the workgroup-only path.

Final paired timings below use the shipped shader templates, seed `20260925`,
11 samples, four warmups, and 128 dispatches per timestamp batch. Each cell is
baseline → current GPU milliseconds per dispatch. These reductions are a small
part of most models' total runtime; they do not imply a 17% model speedup.

| Kernel | 47 tokens | 128 tokens | 512 tokens |
| --- | ---: | ---: | ---: |
| Laya layer norm | 0.007680 → 0.006656 | 0.005632 → 0.005120 | 0.012800 → 0.011776 |
| Qwen RMSNorm | 0.003584 → 0.003072 | 0.004096 → 0.003584 | 0.009728 → 0.009216 |
| Qwen attention preparation | 0.006144 → 0.005120 | 0.011776 → 0.010752 | 0.040448 → 0.036864 |
| Gemma RMSNorm | 0.004608 → 0.004096 | 0.005120 → 0.005120 | 0.012288 → 0.011776 |
| Gemma PLE | 0.004096 → 0.004096 | 0.005632 → 0.005120 | 0.013312 → 0.013312 |

All 20 seeded fixtures (including one token) match every baseline output bit.
The largest independent CPU-reference absolute error is below `8.4e-7`.
Median gains span 0–17%. Several differences are only one amortized timestamp
quantum, and some shapes tie; the report retains that resolution rather than
claiming every shape benefits.

An additional 12 strict paired fixtures cover Qwen hidden widths 2048, 2560 and
4096 at 1/47/128/512 tokens. All are bit-identical and have independent CPU
absolute error below `5.1e-7`. Together with the default 1024-width fixtures,
this covers every current Qwen model width, not only the smallest model.

## All-kernel inventory

All 35 registered compute kernels ran at three lengths, giving 105 paired
fixtures. Every checked output bit matches the prior build. Six kernels have
independent CPU references in this inventory; the rest additionally require the
separate numerical guards. These are synthetic, isolated dispatch timings,
not the time of a model layer. Forced alternatives such as `matmul_wide` do not
imply the runtime selects them. Matrix fixtures here are 1024 × 1024 with R4;
the retained R3 comparisons above use the actual larger projection shapes.

<details>
<summary>Current GPU milliseconds per dispatch, medians of 11 batched samples</summary>

| Kernel | 47 tokens | 128 tokens | 512 tokens |
| --- | ---: | ---: | ---: |
| `matmul` | 0.122880 | 0.061440 | 0.147456 |
| `matmul_wide` | 0.159744 | 0.327680 | 1.175552 |
| `reduce` | 0.004608 | 0.008192 | 0.027136 |
| `norm` | 0.004608 | 0.005120 | 0.011776 |
| `rope` | 0.006144 | 0.008704 | 0.030720 |
| `attention` | 0.045056 | 0.110592 | 1.282048 |
| `attention_subgroup` | 0.024576 | 0.086016 | 0.815104 |
| `attention_tile` | 0.012288 | 0.020480 | 0.184320 |
| `attention_tile_f32` | 0.012288 | 0.028672 | 0.311296 |
| `geglu` | 0.007168 | 0.017408 | 0.072192 |
| `gather` | 0.001024 | 0.001024 | 0.001536 |
| `kev_rms` | 0.003072 | 0.003584 | 0.009216 |
| `kev_gather` | 0.001536 | 0.005120 | 0.004608 |
| `kev_gates` | 0.006656 | 0.006656 | 0.026112 |
| `kev_conv` | 0.011264 | 0.025600 | 0.090624 |
| `kev_save_tail` | 0.001536 | 0.001536 | 0.001536 |
| `kev_qknorm` | 0.006144 | 0.014336 | 0.049664 |
| `kev_recur_lanes` | 0.040960 | 0.110592 | 0.475136 |
| `kev_recur_lanes8` | 0.032768 | 0.081920 | 0.339968 |
| `kev_recur_lanes16` | 0.032768 | 0.090112 | 0.376832 |
| `kev_gnorm` | 0.004608 | 0.008704 | 0.028160 |
| `kev_aprep` | 0.005120 | 0.010752 | 0.036864 |
| `kev_save_kv` | 0.002048 | 0.003072 | 0.008704 |
| `kev_attention_keys` | 0.004096 | 0.008192 | 0.024576 |
| `kev_attention` | 0.032768 | 0.131072 | 1.613824 |
| `kev_attention_tile` | 0.024576 | 0.057344 | 0.397312 |
| `kev_silumul` | 0.009216 | 0.023552 | 0.102400 |
| `gemma4_embed` | 0.002560 | 0.003584 | 0.010240 |
| `gemma4_rms` | 0.004096 | 0.004608 | 0.011264 |
| `gemma4_qkv` | 0.010240 | 0.023040 | 0.084480 |
| `gemma4_attention` | 0.036864 | 0.143360 | 1.970176 |
| `gemma4_gelu` | 0.014848 | 0.040960 | 0.188416 |
| `gemma4_ple` | 0.004096 | 0.005120 | 0.012800 |
| `gemma4_residual` | 0.002560 | 0.003584 | 0.009728 |
| `gemma4_gather` | 0.003072 | 0.002048 | 0.002560 |

</details>

## Rejected experiments

The search includes real projection shapes, small column tails, split targets
32/64/128/256/512, row counts 1–4, column groups 1–2, FP16/FP32 tiles, named
accumulators, K64 loops, byte conversion, attention variants and recurrent lane
counts. Unchanged kernels and existing selectors remain in place unless a
candidate has a repeatable win and passes correctness checks.

| Experiment | Observation | Decision |
| --- | --- | --- |
| Apply the FP32 32×8 named-accumulator kernel to FP16 | Most real shapes slow down; the 31/47-token shapes regress substantially | Reject |
| Sign-extend Q8 bytes instead of the FP16 mantissa conversion | Mostly ties or small mixed changes | Reject |
| Disable wide R4 unrolling | Removes a 7–10× slowdown relative to that forced wide variant; usually only a small change versus the automatically selected generic kernel | Do not globally change the NVIDIA-oriented variant |
| FP16 BM56 with masked rows | Bit-exact, but real shapes regress approximately 8–16% | Reject |
| Use two output-column groups | No stable advantage on the tested real shapes | Reject |
| Change split-K target globally | Shape-dependent improvements and regressions; changes the accumulation partition | Keep 256 |
| Replace Qwen gated-norm reduction | Tiny output-bit differences despite the ordered tree | Reject |
| Replace Qwen convolution reduction | Approximately 2–5% changes, with extra complexity for two logical heads | Reject |
| Replace Gemma QKV reduction | Small mixed changes | Keep existing kernel |
| Force one attention or recurrent implementation for every input | Existing tiled attention and guarded 4/8/16-lane selection already cover distinct workloads | Keep selectors |

Changing CUDA kernels is not a substitute for measuring WGSL. The applicable
lessons are work partitioning, register pressure, padded work and synchronization.
Apple's [GPU performance guidance](https://developer.apple.com/videos/play/tech-talks/111373/)
and [FlashAttention-2](https://arxiv.org/abs/2307.08691) motivate those experiments;
they do not establish a measured speedup on this machine. DeepGEMM and FlashInfer
are not drop-in WebGPU implementations.

I also checked the adapter's optional features. This Chrome session does not
expose `chromium-experimental-subgroup-matrix` or its WGSL extension. That is a
browser API limitation, not a claim that M4 lacks Metal SIMD-group matrices.
The [subgroup-matrix proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md)
describes the native Metal operations and the draft WebGPU mapping. No browser
flags or shared-browser restarts were used to enable experimental APIs.

## Measurement protocol

`scripts/compare-gpu.mjs` pins separate baseline/current runtimes and verifies
that served JavaScript and all three WASM artifacts match their local SHA-256
fingerprints. It verifies them again after each model. Benchmark inputs and
loaded model revisions must agree across the compared blocks.
Token counts are checked per request, not assumed constant within a block:
Qwen tokenizers split the two-digit run numbers differently from single digits.
The same ordered token-count vectors must match across builds and timing modes.

The runner downloads each pack once to a private temporary directory, checks its
size and actual SHA-256 against the catalog, serves those identical bytes locally
to both builds with persistent download caching disabled, and checks the file
hash again after the comparison. It deletes its own temporary files afterward.
This avoids repeated multi-GB downloads during navigation without assuming that
an incognito browser cache successfully stores a large pack. Loading remains
outside the timed region.

The default downloader uses Node's `fetch`. With the Hugging Face CLI installed,
`--download=hf` uses its downloader with chunk caching disabled. Both paths
require the pinned catalog size and actual file SHA-256 to match.
`--local-packs=<directory>` instead verifies existing `<model>-q8.kevala` files
and creates private hard links on the same filesystem. Cleanup removes those
links, not the source files. Corrupt packs are rejected before inference.

The model runner uses its own hidden targets in the default browser context and
closes only those targets. The current runner creates a fresh target for every
wall/profile block and decision replay. The original all-model sweep reused one
target per model; archived controller sources identify the protocol used by each
collection. Target isolation is a lifecycle improvement, not a measured kernel
speedup. A preliminary isolated-context run crashed Chrome
with `NOTREACHED hit. rph_with_bc_reference` pointing to `CreateHiddenTarget`.
Its incomplete model is excluded and rerun in full. The default context avoids
destroying a browser context while hidden-target renderer references remain;
pack and inference caching stay disabled. The shared browser is never closed.

Each model runs baseline/current/current/baseline blocks. Every block measures
unprofiled wall latency separately from instrumented GPU timestamps, with three
warmups and nine unique measured requests at each of three input sizes. State
caching is disabled. Loading, conversion and calibration are excluded. Each
build therefore contributes 18 samples per input size and metric. Raw samples,
block medians, arithmetic means, calibration decisions and model revisions are
retained. Pooled medians are useful for the controlled comparison; arithmetic
means are reported separately and must not be described as medians.

Load-time matrix calibration runs again after each reload. Its diagnostics are
retained. The model comparison therefore includes automatic selector behavior;
the isolated paired-kernel experiments provide the stronger attribution to a
particular shader change.

The decision replay pairs 36 authored Kevala cases and three option-order
permutations per build and model. These 108 rows are a regression test, not 108
independent held-out examples and not a new general-accuracy estimate. The
dataset is `benchmarks/decisions/kevala-authored36.jsonl`; the report records its
hash and every before/after probability vector. The README's larger accuracy
table remains a separate, explicitly dated evaluation.

`dev/kernel-bench.html` inventories all 35 registered kernels at 47/128/512
tokens. It resets mutable inputs before each timed batch, rotates paired variant
order, initializes out-of-place outputs to NaN, and checks complete finite
overwrite. Norm, RMSNorm, attention preparation, PLE and legacy Q/K normalization
also have CPU references. Other inventory cases are finite-output smoke checks,
not independent numerical proofs. Matrix and reducer timings here are isolated,
not a complete projection. Repeated in-place dispatches within a batch can change
their inputs. Use the independent GPU guard suite for numerical validation.

WebGPU timestamps on this browser have a 65.536 µs quantum. Batching amortizes
that resolution; individual short per-model dispatches can round to zero. Small
changes near that resolution require repeatable paired evidence before a win
is claimed.

An attempted native Metal System Trace reported
`Fatal logging system error: The log archive is corrupt or incomplete and cannot be read`.
Its exported shader-sample table contained zero samples. No occupancy, register
or memory-bandwidth counters are claimed. The evidence here is GPU timestamps,
controlled wall measurements and numerical checks.

## Validation

- 130 JavaScript tests and 64 Rust tests pass.
- All 26 supported GPU guards pass, including full-width Qwen normalization,
  Gemma modes and tail cases, recurrence/cache checks and activation extremes.
- All 35 kernels compile in 127 feature-compatible specializations. The
  featureless device compiles 87 and explicitly skips 40 optional variants.
- All 105 inventory pairs and 32 seeded reduction pairs match output bits.
  Matrix padding checks compare complete outputs, including epilogues.
- All 11 models complete 108 paired regression decisions each: **1188/1188
  predictions unchanged**, no errors. This is agreement, not 100% accuracy.
  Maximum probability change is `0.0013319849967956543`; probabilities are not
  universally bit-identical. No existing fidelity tolerance is loosened.
- Ten complete follow-up model reruns add 1080 paired predictions with zero
  changes. A separate fresh-target Kev-9B comparison adds another 108, also
  unchanged. Maximum follow-up probability change is `0.001037001609802246`.
  The recheck artifact retains all raw blocks, paired probabilities, dataset
  hashes and controller sources. Its Gemma E4B row reuses the initial capture;
  no interrupted download or partial inference counts as a completed rerun.
- Laya's independent golden check remains 41/41, max probability error
  `0.022706444195624564`; Kev's remains 13/13, max `0.010131031274795532`.
- All three WASM flavors validate. The staged site builds. Four live browser
  cases reject cross-origin runtime/pack URLs. The CDP cleanup regression is
  reproduced against the prior runner with a mocked transport; both cleanup
  modes and preservation of the original failure pass after the fix.
- The model runner's mocked transport reproduces the prior one-target lifecycle
  and verifies ten fresh owned targets after the change. Cleanup preserves the
  original inference error. Verified local packs survive cleanup; corrupt packs
  are rejected before a target is created. Real Kev-9B execution validates the
  fresh-target and local-pack paths end to end.
- The packed package validates: 28 files and three WASM modules. Packaging used
  `npm pack --force --ignore-scripts` with the already validated WASM artifacts
  because the installed Node 26 does not match the repository's Node 24 pin.
  This does not change the pin or establish validation on the pinned runtime.

## Reproduce without opening browser windows

Use the existing development server and running shared Chrome CDP endpoint;
none of these commands launches a browser. These measurements use a non-isolated
origin (`crossOriginIsolated: false`), one WASM thread and the relaxed WASM flavor.
To match it, serve the checkout without `--isolate` or COOP/COEP headers.
Build the baseline with the same pinned Rust toolchain and all three WASM
flavors before rebuilding the current checkout.

```sh
mkdir -p tmp/m4-opt/baseline
git archive b2f9d956aa41d6b7992e623f70a2ba9f7d1c2723 | tar -x -C tmp/m4-opt/baseline
(cd tmp/m4-opt/baseline && RUSTUP_TOOLCHAIN=1.95.0 scripts/build-wasm.sh)
RUSTUP_TOOLCHAIN=1.95.0 scripts/build-wasm.sh

node scripts/compare-gpu.mjs \
  --base=http://127.0.0.1:8123/ \
  --cdp=http://127.0.0.1:9333 \
  --baseline=tmp/m4-opt/baseline/js/src/index.js \
  --current=js/src/index.js \
  --output=tmp/m4-opt/comparison

uv run scripts/bench-gpu.py \
  --cdp http://127.0.0.1:9333 --cdp-default-context --result gpu \
  --url 'http://127.0.0.1:8123/dev/kernel-bench.html?baselineWasm=/tmp/m4-opt/baseline/js/src/kevala-base.wasm&samples=11&warmups=4' \
  --output tmp/m4-opt/all-kernels.json

KEVALA_BENCH_CDP=http://127.0.0.1:9333 \
KEVALA_BENCH_URL=http://127.0.0.1:8123 \
  node scripts/gpu-guard.mjs tmp/m4-opt/guard

node --test tests/*.test.mjs
cargo +1.95.0 test --workspace --locked
node scripts/check-wasm.mjs
```

Run GPU jobs serially. Do not edit either runtime during a comparison. A failed
or interrupted model does not count as completed evidence; rerun its entire
paired comparison. Native Metal traces should not overlap final timing runs.
