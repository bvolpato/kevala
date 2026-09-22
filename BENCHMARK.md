# Decision benchmarks

This benchmark measures Kevala's typed `choice` API across every published weight pack.
Accuracy, option-order stability, conversion fidelity, and latency are separate measurements.
The fixtures are small synthetic tasks. They do not establish general reasoning ability,
calibrated confidence, or a ranking against other inference engines.

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

For GPU timestamp measurements and prior matrix optimizations, see
[GPU matrix tuning](docs/semif-matmul.md). Gemma's initial profiles and NVIDIA/AMD/native
reference checks are in [Gemma 4 verification](docs/gemma4.md#verification).

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

Change `model` to any name in the model table. `dataset=kevala` selects only the 36 new cases;
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
