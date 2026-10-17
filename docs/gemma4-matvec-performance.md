# Gemma 4 decoder matrix-vector kernels

Gemma's single-row Q8 decoder projections now have direct matrix-vector kernels.
The Chrome camera check reduced Balanced decision latency from 51.74 ms to
45.83 ms and increased displayed throughput from 19.1 to 21.5 completed decisions
per second. These are Linux RTX 5070 Ti measurements. M4 Max performance remains
unmeasured.

## Selection and precision

The F32 path assigns 64 lanes to each output and reduces their partial sums in
workgroup memory. Four outputs share a workgroup. It avoids padded token rows,
global split-K scratch writes, and a separate reduction dispatch.

The Q8/F16 path handles 16 outputs with 64 threads and 2,176 bytes of shared memory.
It retains the existing operand rounding, split boundaries, sequential dot-product
order, split summation order, and epilogue. Accumulation stays F32. Matching operand
precision alone was insufficient: an earlier parallel reduction exceeded the
unchanged full-model probability guard and was rejected.

At load time, auto tuning compares each candidate against the selected existing
matmul on actual loaded projection weights and deterministic nonzero inputs.
Selection requires a numerical pass and three alternating timing pairs with a
stable improvement of at least 5%. F16 requires exact equality for every compared
output; F32 retains a bounded rounding allowance. Unstable, slower, failed, or
unmeasured shapes keep the existing kernel. Only Q8 operations with one active row
use the new path. Explicit `gpuKernel: "generic"` and `"wide"` retain their existing
behavior.

Calibration uses GPU timestamps when available and repeated submit/readback wall
timing otherwise, provided the browser's clock resolution is usable. Sampling has
a 1.5-second deadline, pending readback maps are cancelled at that deadline, and
scratch is capped at 64 MiB. Pipeline compilation may outlast the sampling deadline;
an already expired deadline selects no candidate. Results are local to that model
load and appear in the cloneable `info.gpuTuning.matvec` diagnostics.

BF16/F32 kernel variants retain F32 operands, including BF16's wider exponent
range. Model weights, quantization, image normalization, image budgets, questions,
and scheduling are unchanged.

## Completed camera decisions

Chrome 153 used the real application scheduler and a changing 320 × 240 synthetic
camera at 60 FPS. Each stage completed 32 requests; latency statistics discard four
warmups. Each stage had 32 distinct input hashes, one inference in flight, and the
same loaded model across detail changes. Displayed FPS is the recorder's rolling
completed-decision count.

| Stage | Existing p50 / p95 | Auto p50 / p95 | Displayed FPS, existing → auto |
|---|---:|---:|---:|
| Detail | 63.43 / 64.73 ms | 57.35 / 58.62 ms | 15.4 → 17.0 |
| Balanced | 51.74 / 52.86 ms | 45.83 / 46.58 ms | 19.1 → 21.5 |
| Fast | 47.89 / 49.35 ms | 41.33 / 44.07 ms | 20.6 → 23.6 |
| Balanced repeat | 51.80 / 55.82 ms | 45.82 / 47.04 ms | 18.9 → 21.5 |

The existing control uses `gpuKernel: "generic"`. This Chrome adapter exposes no
shader F16, so the prior auto path also selected generic. See the
[control](benchmarks/gemma4-matvec-2026-10-16/camera-control.json) and
[auto](benchmarks/gemma4-matvec-2026-10-16/camera-auto.json) receipts.
The camera runner's decision timer begins after capture and includes its input
hashing; it does not measure sensor-to-painted-result latency. GPU sharing, short
stages, and run order limit these results.

## Correctness checks

The browser numerical harness checks Q8, BF16, and F32 against an independent CPU
dequantization reference and the existing matmul in 86 shape/dtype cases. It covers all six bias/mode
combinations, output tails, cancellation, Q8 -128, non-power-of-two scales, BF16/F32
exponent extremes, actual E2B MLP dimensions, odd split boundaries, and an empty
final split. The [Firefox F16 pairs](benchmarks/gemma4-matvec-2026-10-16/matvec-firefox.json)
matched exactly; the [Chrome F32 kernel](benchmarks/gemma4-matvec-2026-10-16/matvec-chrome.json)
differed from the CPU reference by at most `2.642e-7`. The tiny production decoder guard
also compares selected rows, duplicate rows, shared prefixes, changing images,
GPU-resident projections, and nonfinite projection rejection.

The full-model F16 check loads one decoder and vision companion and switches only
the validated matvec selection off/on/off. It compares four public images at
70/40/16-token budgets, retaining the unchanged `5e-4` probability limit and checking
0.5 threshold flips, finite outputs, image token counts, and shared-prefix geometry.
The [two-question check](benchmarks/gemma4-matvec-2026-10-16/f16-full-model-paired.json)
and [five-question check](benchmarks/gemma4-matvec-2026-10-16/f16-five-question-paired.json)
each matched every raw probability exactly over 36 measured decisions and 18
warmups. The isolated diagnostic proxy changed only the model's validated selection
set; it did not enable profiling or change inference inputs. Separate serial Chrome
loads with generic versus auto kernels had a maximum probability difference of
[`7.153e-7`](benchmarks/gemma4-matvec-2026-10-16/f32-full-model-comparison.json)
with no 0.5 threshold flips. These are numerical regression checks, not a
representative expression-accuracy evaluation.

Loaded-weight F16 calibration selected the 12288×1536 and 6144×1536 projections,
reducing their median kernel times by 40% and 15%, respectively. Other shapes kept
the existing kernel. Firefox's completed decision timing remained near 100 ms in
this headless environment; faster kernels did not establish a Firefox camera FPS
improvement. Without timestamps, [Chrome selected stable candidates](benchmarks/gemma4-matvec-2026-10-16/sdk-wall-chrome.json)
in 222 ms. [Firefox's F16 wall-clock run](benchmarks/gemma4-matvec-2026-10-16/sdk-wall-f16-firefox.json)
kept the baseline after noisy timing pairs and stopped calibration at its deadline.

Validation included JavaScript checks, unit tests, all three WASM builds and ABI
checks, package/site staging, Rust formatting, clippy, and the native test suite.
The full supported GPU guard passed before the F16 accumulation adjustment; final
targeted Chrome/Firefox numerical, decoder, pipeline-compilation, full-model, and
no-timestamp checks passed after it. Unsupported subgroup paths were skipped by
the device capability guard.

## Reproduce

Serve the checkout with the local decoder and vision packs, then open
`/dev/gemma4-matvec-check.html` for the numerical checks. The same page runs through
`pnpm guard:gpu`. Compare full-model results from
`/dev/gemma4-camera-check.html?autorun=1&images=../tmp/vision-quality/images.json`
with `kernel=generic` and `kernel=auto` in the same browser.

For application throughput, run these sequentially with the same browser binary:

```sh
uv run scripts/bench-reaction.py --binary /path/to/chrome \
  --kernel generic --camera-fps 60 --frames 32 --output tmp/matvec-control.json
uv run scripts/bench-reaction.py --binary /path/to/chrome \
  --kernel auto --camera-fps 60 --frames 32 --output tmp/matvec-auto.json
```

Use `features=none` on the SDK harness to exercise wall-clock calibration without
optional GPU features, or `features=f16` for F16 without timestamps. Keep heavy
model/browser jobs serial and resource bounded.
