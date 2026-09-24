# Decision benchmarks

This benchmark measures Kevala's typed `choice` API across every published weight pack.
Accuracy, option-order stability, conversion fidelity, and latency are separate measurements.
The fixtures are small synthetic tasks. They do not establish general reasoning ability,
calibrated confidence, or a ranking against other inference engines.

## Results: September 22–24, 2026

All ten packs completed all 864 requested decisions with valid probabilities and no runtime
failures: **8,640 WebGPU decisions in total**. Each cell below gives accuracy and the number
of correct option orders. Pack sizes are binary GiB, not VRAM requirements.
The first nine runs were recorded September 22; Bruv 0.8B was recorded September 24 UTC
on the same GPU and driver after its training job had exited.

| Pack | Download GiB | Kevala authored | SemIf authored | SemIf perturbations |
|---|---:|---:|---:|---:|
| Laya | 0.45 | 75.0% (81/108) | 62.3% (269/432) | 69.4% (225/324) |
| Bruv 0.8B | 0.80 | 90.7% (98/108) | 79.6% (344/432) | 64.8% (210/324) |
| Kev 0.8B | 0.80 | 89.8% (97/108) | 71.1% (307/432) | 70.7% (229/324) |
| Kev 4B | 4.43 | 91.7% (99/108) | 89.4% (386/432) | 90.1% (292/324) |
| Kev 9B | 8.35 | 96.3% (104/108) | 91.9% (397/432) | 96.3% (312/324) |
| SemIf 0.8B | 0.80 | 60.2% (65/108) | 50.9% (220/432) | 41.7% (135/324) |
| SemIf 2B | 1.98 | 85.2% (92/108) | 63.4% (274/432) | 55.6% (180/324) |
| SemIf 4B | 4.42 | 100.0% (108/108) | 84.0% (363/432) | 74.7% (242/324) |
| Gemma 4 E2B IT | 4.86 | 94.4% (102/108) | 78.2% (338/432) | 71.3% (231/324) |
| Gemma 4 E4B IT | 7.83 | 100.0% (108/108) | 89.1% (385/432) | 90.1% (292/324) |

### Option-order stability and latency

An order flip means at least one of the three rotations changed the selected semantic option.
Lower is better. The three flip columns refer to 36, 144, and 108 source cases respectively.
They are descriptive counts; related SemIf cases are not independent observations.

| Pack | Kevala flips | SemIf authored flips | SemIf perturbation flips | Wall p50 ms | Wall p95 ms |
|---|---:|---:|---:|---:|---:|
| Laya | 10/36 | 22/144 | 17/108 | 99.9 | 100.3 |
| Bruv 0.8B | 2/36 | 16/144 | 14/108 | 99.9 | 100.2 |
| Kev 0.8B | 2/36 | 19/144 | 11/108 | 99.9 | 100.3 |
| Kev 4B | 2/36 | 6/144 | 7/108 | 100.0 | 199.9 |
| Kev 9B | 1/36 | 3/144 | 0/108 | 200.0 | 299.7 |
| SemIf 0.8B | 9/36 | 70/144 | 41/108 | 99.9 | 100.2 |
| SemIf 2B | 13/36 | 93/144 | 84/108 | 99.9 | 100.2 |
| SemIf 4B | 0/36 | 45/144 | 40/108 | 200.0 | 202.3 |
| Gemma 4 E2B IT | 5/36 | 38/144 | 45/108 | 100.0 | 200.0 |
| Gemma 4 E4B IT | 0/36 | 17/144 | 14/108 | 200.0 | 202.9 |

Bruv 0.8B answered 652/864 decisions correctly across the three suites. Kev 9B had the
highest accuracy on the two SemIf suites in this comparison. SemIf 4B and Gemma
E4B answered all new Kevala cases correctly, but their lower scores on the SemIf suites show
why one small suite is insufficient. Several direct-option models were sensitive to option
order. Use these measurements to select candidates for your own evaluation, not as a universal
model ranking.

The [ten-pack summary JSON](benchmarks/results/decisions-linux-2026-09-22/summary-bruv.json) includes per-family
scores, grouped 95% bootstrap intervals, position bias, and validation results. The
[original nine-pack summary](benchmarks/results/decisions-linux-2026-09-22/summary.json) remains
available. The [raw outputs](benchmarks/results/decisions-linux-2026-09-22/)
include every answer, probability, and timing. The six Kev/SemIf runs predate runtime fixture
hash capture; their artifacts explicitly record post-run verification of the unchanged fixture
files. Laya, Bruv, and both Gemma runs captured and checked those hashes in the running page. All ten
local pack files were independently hashed in full and matched the pinned catalog digests.

## Tasks and scoring

The frozen [dataset manifest](benchmarks/decisions/manifest.json) records the source revisions,
licenses, byte hashes, and counts:

| Suite | Source cases | Independent source groups | Scored option orders |
|---|---:|---:|---:|
| Kevala authored | 36 | 36 | 108 |
| SemIf authored | 144 | 36 | 432 |
| SemIf perturbations | 108 | The same 36 SemIf groups | 324 |

Each suite contains evidence interpretation, rule application, and candidate selection tasks.
Kevala's 36 cases were authored for this evaluation and checked for ambiguous labels before
the recorded comparison. They have not been human adjudicated. Their correct positions are
balanced within each task family.

SemIf's fixtures are copied unchanged from revision
[`1f2dea3e25379f9dfc98cb83c324f00ab5deda37`](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/benchmarks/data).
SemIf supplies both the direct-option scoring method and these evaluation tasks. Its MIT
attribution is retained in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). These measurements use
Kevala's request adapters and quantized packs; they are not reproductions of SemIf's published
native results.

Each source case runs with its original option order and two cyclic rotations. The same
evidence, criterion, and semantic option IDs are supplied to every model through the typed API.
Each family uses its own supported prompt and readout. The answer is compared by semantic ID,
not by its displayed position. No text generation or reasoning trace is requested.

- **Accuracy:** correct decisions divided by all requested decisions, including failures.
- **Coverage:** valid decisions divided by all requested decisions. Invalid scores, exceptions,
  and missing answers remain failures.
- **Order flips:** source cases whose semantic prediction changes across the three option orders.
- **95% intervals:** 1,000 deterministic bootstrap draws of independent source groups. Option
  rotations and SemIf perturbations stay with their original group. Small suites have wide
  uncertainty; a perfect sample score is not a population guarantee.
- **Position bias:** the distribution of selected positions compared with the balanced gold
  positions, recorded with the raw results.

The three suites are reported separately because the SemIf perturbations are related to its
authored examples. The [metrics implementation](benchmarks/decisions/metrics.js) preserves that
grouping and checks row IDs and gold labels.

## Runtime and timing

The recorded browser runs use Ubuntu, Firefox 152.0.3, an NVIDIA RTX 5070 Ti with 16 GB VRAM
and driver 595.71.05, and an AMD Ryzen 9 9950X3D host. GPU jobs run sequentially. The NVIDIA
Vulkan ICD is selected explicitly, and the runner rejects a software adapter or CPU fallback.
These are WebGPU runs. CUDA is used separately for the original-weight reference comparison.

Every browser run uses a fresh isolated headless Firefox profile, batch size one, automatic
GPU matrix selection, and the same local copies of the published Q8 packs. Downloading,
initialization, tuning, and warmup are excluded from decision latency and recorded separately.
Pack caching and cross-request inference caching are disabled. Duplicate sharing within one
batch remains available, although these runs have only one request per batch.

Wall time measures the complete awaited `decide()` call, including tokenization, GPU work,
readback, and result formatting. Profiling is disabled during those measurements. Firefox's
timer privacy settings are relaxed only in the temporary benchmark profile, and the observed
clock resolution is recorded. Firefox queue completion can produce roughly 100 ms steps in
round-trip latency even with a fine clock. These values therefore describe this browser and
driver, not just the kernel execution time.

### Latest four-family GPU profile

This Chrome profile compares merged main `eb67049` with that build plus packed loads in the
subgroup recurrence kernel. Each family value is the geometric mean of median summed GPU kernel
times across three input sizes; two complete runs per version are combined geometrically. The
four-family value is the geometric mean of family metrics. It measures GPU profile time, not
per-request or wall latency. Runs used an RTX 5070 Ti (16 GB), driver 595.71.05, Ryzen 9
9950X3D, Ubuntu, Chrome 149.0.7827.200, and the NVIDIA Vulkan ICD.

| Family | Baseline GPU ms | Packed GPU ms | Change |
|---|---:|---:|---:|
| Laya | 9.866 | 9.789 | 0.78% lower |
| Kev 0.8B | 14.073 | 13.875 | 1.41% lower |
| SemIf Qwen3.5 2B | 49.075 | 48.816 | 0.53% lower |
| Gemma 4 E2B | 32.197 | 32.530 | 1.04% higher |
| Four-family geometric mean | 21.642 | 21.550 | 0.42% lower |

Kev and SemIf exercise the affected recurrence path. Laya and Gemma are controls, and the
four-family change is descriptive rather than an attributed end-to-end gain. See the
[packed recurrence report](docs/recurrence-packed-performance.md) for the actual recurrence
profile component, rendered microbenchmarks, and correctness guards.

For GPU timestamp measurements and prior matrix optimizations, see
[GPU matrix tuning](docs/semif-matmul.md). Gemma's initial profiles and NVIDIA/AMD/native
reference checks are in [Gemma 4 verification](docs/gemma4.md#verification).
The latest shared FP32 results cover Laya, Kev, SemIf, and Gemma in
[exact byte conversion](docs/fp32-byte-conversion-performance.md). Gemma's separate
[normalization fusion](docs/gemma-rms-residual-performance.md) includes both sizes'
correctness checks.
The Qwen-specific [FP32 attention tile report](docs/qwen-fp32-attention-performance.md)
includes paired kernel measurements, full-model timings, and model-guard results.
The [Laya FP32 attention report](docs/laya-fp32-attention-performance.md) includes paired
kernel measurements, full-model repeats, and browser parity guards.
The [FP32 BM56 row tile report](docs/fp32-row56-performance.md) includes final-WASM
matmul timings, production-shaped model repeats, and model guards.
The [subgroup direct-load recurrence report](docs/recurrence-direct-performance.md) includes
production-rendered recurrence timings, repeated model suites, and parity guards.
The [packed recurrence-load report](docs/recurrence-packed-performance.md) compares vector
loads in the subgroup kernel with full-suite and model-guard results.
The [Laya compact final-head report](docs/laya-compact-head-performance.md) records
a 2.63% reduction in summed GPU time across two paired production-build runs,
with exact hidden-output comparisons and the unchanged Chrome probability-limit miss.

## Base weights, instruction tuning, and precision

The website initially uses Google's Gemma 4 E2B and E4B **instruction-tuned (`-it`)** weights.
The base comparison uses the same Kevala cases and option permutations, with original BF16
weights for both variants. Base models receive an explicit raw completion prompt; instruction
models use their native chat template with thinking disabled. This compares a checkpoint and
its appropriate prompt protocol together. It does not isolate training from prompt format.

The original BF16 checkpoints produced these results on the 108 Kevala option orders:

| Gemma 4 checkpoint | Correct | Accuracy | Cases with an order flip |
|---|---:|---:|---:|
| E2B base | 45/108 | 41.7% | 30/36 |
| E2B IT | 101/108 | 93.5% | 6/36 |
| E4B base | 62/108 | 57.4% | 18/36 |
| E4B IT | 108/108 | 100.0% | 0/36 |

These results support the IT default for Kevala's zero-shot direct option scoring. They do
not rule out base weights with a trained readout, examples, or a different prompt. Those
alternatives were not evaluated. The [four-variant reference record](benchmarks/results/decisions-linux-2026-09-22/gemma-base-vs-it.json)
contains the exact prompts, token IDs, probabilities, and pinned source revisions.

The reference helper verifies that each answer label is exactly one token and does not change
the prompt boundary. It loads the learned persistent buffers and uses the upstream PLE path.
Its CUDA timings are not directly comparable with browser wall time: it uses a different
runtime and keeps the large per-layer embedding table on the CPU.

Q8-versus-BF16 decision agreement compares the same instruction checkpoint, task, option order,
and typed-option prompt. Differences include both weight quantization and runtime arithmetic.
They are not a pure measurement of quantization loss. Kevala stores INT8 weights with an FP32
scale per 32 values, then uses floating-point matrix arithmetic and FP32 accumulation.

| IT model | Q8/BF16 matching decisions | Q8 correct | BF16 correct | Mean absolute probability difference | Maximum difference |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B | 107/108 | 102/108 | 101/108 | 0.004156 | 0.530106 |
| Gemma 4 E4B | 108/108 | 108/108 | 108/108 | 0.0000145 | 0.0007614 |

**E2B has a material probability outlier.** For `kev-evidence-10::perm:rotate2`, the probability
of `contradicted` is 0.7773 with the original BF16 reference, 0.5259 when those original weights
use FP32 arithmetic, and 0.2472 with WebGPU Q8. The Q8 answer changes to `insufficient`.
This happens to match the fixture label, but does not show that quantization improves quality.

A [numerical diagnostic](benchmarks/results/decisions-linux-2026-09-22/gemma-e2b-numerical-diagnostic.json)
checked the three largest E2B differences against native Q8. Token IDs matched exactly, all
three decisions agreed with WebGPU, and the maximum native/WebGPU probability difference was
0.0132, within the existing 0.03 Gemma validation limit. Against the additional FP32 reference,
WebGPU Q8 still matched 107/108 decisions, with a maximum probability difference of 0.2787.
These checks support consistent Q8 execution across the two backends. They do not establish
probability parity with the original weights. Scores should not be treated as calibrated confidence.

## Reproduce

Build the generated WASM files and place the pinned packs in `tmp/`, or use `pack=hosted` to
download them. Large packs are opt-in and are not downloaded by ordinary CI.

```sh
pnpm build
pnpm serve --port=18093

VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result decision --timeout 1200 \
  --url 'http://127.0.0.1:18093/dev/decision-bench.html#model=gemma-4-e2b&backend=webgpu&pack=local&dataset=all&permutations=3' \
  --output tmp/decision-gemma-4-e2b.json

uv run tools/benchmark_gemma_variants.py \
  --device cuda --dtype bfloat16 --output tmp/gemma-base-vs-it.json
```

Use a model ID from the [catalog](js/src/source.js), such as `kev-4b`, `semif-qwen3.5-2b`,
or `gemma-4-e4b`. `dataset=kevala` selects only the 36 new cases;
`dataset=semif` selects both upstream suites. `profile=1` records a separate instrumented
request before the uninstrumented decision sweep.

Recompute the published scores from the saved outputs without downloading weights:

```sh
pnpm exec node scripts/report-decisions.mjs \
  --results benchmarks/results/decisions-linux-2026-09-22 \
  --reference benchmarks/results/decisions-linux-2026-09-22/gemma-base-vs-it.json \
  --output tmp/decision-summary.json
```

The [browser harness](dev/decision-bench.html), [reference helper](tools/benchmark_gemma_variants.py),
and raw results make failures and individual probabilities inspectable. Tetris smoke checks
are documented separately; a few valid placements do not measure gameplay quality.
