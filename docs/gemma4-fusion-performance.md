# Gemma 4 fusion performance

This note records the retained exactness-checked fusions and their measurement limits.
They change execution and intermediate storage only. Model weights, quantization,
image budgets, questions, and probability thresholds are unchanged.

## Wrap-up

The latest controlled Fast comparison improves completed decisions from **42.61
to 44.17 per second**, or **3.67%**, with one loaded model, unchanged inputs and
settings, 480 measured decisions per mode, and balanced off/on/on/off ordering.
Decision p50 changes from 22.938 to 22.100 ms and p95 from 24.730 to 23.845 ms.
These are Linux RTX 5070 Ti / Chromium measurements with changing synthetic
camera frames, not M4 Max or sensor-to-painted-result measurements. The 60 FPS
target was not reached. Optimization stopped at the user's request.

The earlier and later sustained all-budget checkpoints show the accumulated
progress at fixed image settings:

| Image setting | Earlier sustained checkpoint | Later all-budget checkpoint |
| --- | ---: | ---: |
| Detail, budget 70 | 19.86 FPS | 24.07 FPS |
| Balanced, budget 40 | 25.82 FPS | 32.08 FPS |
| Fast, budget 16 | 30.13 FPS | 41.02 FPS |

These checkpoints used separate runs and code revisions, so their differences
are campaign trends rather than isolated causal speedups. The latest 44.17 FPS
result above tests Fast only; Balanced and Detail were not remeasured afterward.
Fast remains optional because its smaller image grid can miss detail. None of
the latest fusion, cache or image-copy changes lowers an existing image budget,
changes stored precision or replaces model weights.

Retained work covers direct and two-question shared-weight matrix-vector kernels;
batched independent questions; static text K/V and PLE caches; dead-row pruning;
device-calibrated prefix tiles, attention and MLP variants; clip, RMS, residual,
RoPE, pooling, split-reduction and GELU fusions; exact two-pass GPU bicubic image
preparation; and latest-frame scheduling with one decision in flight. The newest
grid-stacked gate/up fusion removes 42 decoder dispatches per frame, with an
approximately 9.1% loaded-weight composite GPU-time reduction. Worker-owned RGBA
borrowing removes two 320×240 pixel copies, or 600 KiB of allocations per frame.
The guarded joint vision Q/K/V kernel improves composite GPU time but had a
neutral final camera comparison, so it is not credited with an FPS gain.

The final local package contains 56 verified files and three rebuilt WASM
flavors. Numerical, model, fallback, camera lifecycle, JavaScript, Rust and
package validation are recorded below. The newest fusions pass exact full-hidden
and raw-probability comparisons; older fused attention retains its documented
bounded numerical tolerance. These fixtures establish numerical regression
coverage, not representative expression-classification accuracy. Measurements
were recorded before publication; deployment status is tracked separately.

## Retained changes

- **Vision input clipping:** On the actual E2B vision pack, compatible Q/K/V, gate/up,
  down, and attention inputs absorb 112 clip passes. The enabled/disabled full-output
  guard reports zero delta. See the [vision guard](../dev/gemma4-vision-check.html).
- **Matvec plus GELU:** The paired kernel passes complete-output comparisons for
  T=1, 2, and 5 in both Chrome and Firefox. See the
  [matvec-GELU guard](../dev/gemma4-matvec-gelu-check.html).
- **Adjacent RMS work:** Residual-plus-next-RMS and RMS-plus-scalar together pass
  225 complete-output checks per browser, bit-for-bit, in Chrome and Firefox.
  The fused shaders retain the source mode selection and conditional add
  so compiler contraction follows the baseline. The exactness guard was not loosened.
  Each model load also compares every layer's actual RMS weights before enabling
  fusion. Any mismatch, readback deadline, or validation failure retains the
  original kernels. The E2B checks compared 322,560 values across all 35 layers
  exactly in both browsers, using 2.53 MiB of bounded scratch.
  See the [RMS fusion guard](../dev/gemma4-rms-fusion-check.html).
- **Independent question branches:** Batches of 2 to 8 preserve each branch's logical
  attention boundaries and floating-point operation order. Split-part fallback bounds
  scratch to 64 MiB; larger requests keep the existing path. The
  [decoder guard](../dev/gemma-tail-check.html) covers the batching cases.
- **F16 R3 prefill:** Auto calibration uses actual loaded weights for T=65 to 96 and
  pairs gate/up GELU. It requires exact outputs and at least 5% improvement in three
  alternating pairs, with a 750 ms calibration limit and 32 MiB scratch cap. It selected
  the 6144×1536 and 12288×1536 shapes, whose median kernel times improved 27% and 25%
  against the current effective baseline.
- **F16 K64 prefill:** Auto eligibility is limited to R3 at T=65 to 96, with a 1 second
  calibration limit and 64 MiB scratch cap. F32 is rejected. These limits keep the
  candidate local to model load and bounded in memory.
- **CPU image preprocessing:** RGB accumulation now shares each coefficient loop across
  channels, and vertical resize writes directly into the padded patch layout. The
  horizontal U8 rounding boundary, Pillow-compatible bicubic coefficients, row-major
  patch order, positions, and zero padding remain unchanged. Full patch and position
  outputs match the pre-change implementation bit-for-bit at budgets 70, 40, and 16;
  the existing pinned Pillow fixture checks also pass. Internal compact preparation
  additionally writes normalized active pixels directly, preserving both F32
  rounding boundaries and eliminating a padded allocation and normalization copy.
  Its separate measurements appear below.

For 320×240 input, one bounded Node 22.22.2 process used 20 alternating warmups and 50
alternating baseline/current samples per budget. Systemd limited the process to one CPU
and 512 MiB of memory. Every measured pair had identical complete outputs.

| Image-token budget | Baseline median | Current median | Speedup |
|---:|---:|---:|---:|
| 70 | 3.219 ms | 2.309 ms | 1.39× |
| 40 | 2.038 ms | 1.453 ms | 1.40× |
| 16 | 1.250 ms | 0.843 ms | 1.48× |

At budget 40, the separate warmed `visionImagePlan` stage measured 0.427 ms median and
0.575 ms p95. These are CPU preprocessing measurements, not camera FPS.

The calibration deadlines bound sampling; asynchronous pipeline compilation can
outlast them. Expired, unsupported, slower, noisy, or numerically different
candidates keep the existing kernels. Explicit generic/wide overrides retain the
unfused decoder and sequential question branches.

## Full-model and device limits

The five-question Chrome same-model check and Firefox F16 check covered image budgets
70, 40, and 16 across 36 measured decisions after 18 warmups. Both reported zero raw
probability delta and zero threshold flips. Loaded-weight F16 measurements selected only
the 6144×1536 and 12288×1536 prefill shapes. F32 direct-FMA showed no stable gain, the
F32 prefill pair was 1–8% slower, and F32 K64 was about 50% slower, so those candidates
remain rejected.

Chromium measurements identify the local Linux RTX 5070 Ti. Earlier Firefox receipts
report only a generic adapter name, so they establish browser/F16 behavior without
proving which physical GPU was used. A later [constrained adapter probe](benchmarks/gemma4-fusions-2026-10-17/firefox-nvidia-adapter.json)
uses the NVIDIA-only Vulkan manifest and confirms an owned Firefox GPU context on
the RTX 5070 Ti with `shader-f16`. None of these establish M4, CUDA, native-GPU,
or MacBook performance. No M4 measurement of these changes is available.
The earlier [vision results](gemma4-vision-performance.md) and
[matvec results](gemma4-matvec-performance.md) retain their own platform and stage
details.

## Sustained camera decisions

The [camera receipt](benchmarks/gemma4-fusions-2026-10-16/camera-30fps-camera-sustained-fusion.json)
uses the real recorder, one loaded model, two unchanged questions, and Chromium's
changing 320×240 synthetic camera at 60 Hz. Each stage completed 240 decisions;
latencies and completion rates discard four warmups. Completion FPS uses actual
completion timestamps. These stages also outlast the UI's five-second rolling
window. This is not a sensor-to-painted-result latency measurement.

| Stage | Decision p50 / p95 | Completed decisions/s |
|---|---:|---:|
| Detail | 49.35 / 50.64 ms | 19.86 |
| Balanced, default | 38.19 / 39.25 ms | 25.82 |
| Fast | 32.57 / 35.14 ms | 30.13 |
| Balanced repeat | 38.57 / 39.93 ms | 25.49 |

The [control run](benchmarks/gemma4-fusions-2026-10-16/camera-30fps-camera-sustained-control.json)
keeps the existing calibrated matvec, the new CPU processor, and the same scheduler,
but disables the new GPU fusions and question batching. Its Balanced p50 was
44.80 ms and Fast p50 was 40.52 ms: the candidate reduces these by about 15% and
20%, respectively. Runs used separate browser loads of the same local packs;
run order and shared GPU load limit attribution.

Balanced remains below 30 FPS on this device. Fast is an existing optional setting
with a detail-loss hint; it retains its previous image budget. No default image
quality was lowered to reach 30 FPS. Camera capture now requests up to 60 Hz and
the scheduler admits one inference at a time. Zero-delay tasks use a cancellable
MessageChannel; missing frames back off instead of spinning.

The final [five-question F16 comparison](benchmarks/gemma4-fusions-2026-10-16/camera-30fps-rms-loaded-five-firefox.json)
and [F32 comparison](benchmarks/gemma4-fusions-2026-10-16/camera-30fps-rms-loaded-five-chrome.json),
[vision clipping comparison](benchmarks/gemma4-fusions-2026-10-16/camera-30fps-vision-clips-final.json),
and [CPU preprocessing comparison](benchmarks/gemma4-fusions-2026-10-16/gemma4-image-preprocess-2026-10-16.json)
retain the raw evidence. The image set is a numerical regression smoke check,
not a representative expression-accuracy evaluation.

## Further fusion and pruning, October 17

Automatic decoder mode now also trims the last KV-source layer to selected question
rows after capturing its full K/V. The moved projections retain their original row
tile, kernel choice, and split-K partition. The common prefix stops after that
layer's K/V because its remaining hidden rows have no consumer.

Combined PLE inputs for unchanged suffix and selected text tokens are cached as
exact F32 GPU copies. The cache excludes image-derived rows, commits entries only
after successful inference, and invalidates on token, selection, graph, execution
policy, device, or weight changes. Its cap is 16 MiB. The two-question profiled
request reused 35 layer buffers totaling 1,433,600 bytes; the five-question check used
3,562,496 bytes. Each warm hit skips the original projection, embedding lookup,
and PLE combination.

The vision path adds two portable fusions:

- Q/K RMSNorm and axial RoPE share one 64-thread workgroup. It preserves the
  original 256-lane reduction tree by zeroing inactive lanes, and materializes
  the normalized F32 values before rotation. The same kernel normalizes V with
  rotation disabled.
- Pooling and its unweighted RMSNorm share a workgroup for hidden widths up to
  1024. Larger widths retain the separate passes. Pool accumulation order,
  optional standardization, and the intermediate F32 rounding remain intact.

Together these remove 33 vision dispatches per E2B image. The
[Chrome guard](benchmarks/gemma4-fusions-2026-10-17/vision-head-final-chrome.json)
compared complete output bits. At 567 rows, 12 heads, and head dimension 64:

| Operation | Separate passes | Current kernel |
|---|---:|---:|
| Q/K normalization and RoPE | 30.50 µs | 11.56 µs |
| V normalization | 26.50 µs | 10.13 µs |

The [Firefox guard](benchmarks/gemma4-fusions-2026-10-17/vision-head-final-firefox.json)
also matched every output bit. Chromium identifies the RTX 5070 Ti; that earlier
Firefox receipt has a generic adapter identity. These are kernel timings, not
full-frame speedups or M4 results. Pool/normalization guards also matched
complete bits in both browsers, including standardization, cancellation, extreme
finite values, and output tails.

The [actual vision tower comparison](benchmarks/gemma4-fusions-2026-10-17/vision-actual-fusions-chrome.json)
has zero enabled/disabled output delta and retains the pinned F32 reference error
of `2.065300941467285e-5` under the unchanged `1e-4` tolerance. Combined
[Chrome](benchmarks/gemma4-fusions-2026-10-17/all-fusions-loaded-chrome.json) and
[Firefox F16](benchmarks/gemma4-fusions-2026-10-17/all-fusions-loaded-firefox.json)
same-load off/on/off checks also have zero probability delta and zero threshold
flips at budgets 70, 40, and 16. Their metadata verifies source pruning, 35 warm
PLE cache hits, and all 33 removed vision passes.

Fewer synchronization barriers alone did not help: the experimental subgroup
attention variant retained exact outputs but ran slower, so production keeps the
portable fused attention kernel. Exact Q8 mantissa decoding also lacked a stable
performance win and remains outside production.

The [new profile summary](benchmarks/gemma4-fusions-2026-10-17/summary.json) records
about 8.55 ms of vision GPU work and 23.79 ms of decoder GPU work at the default
budget with two questions. Decoder command encoding took about 0.63 ms. The
largest remaining decoder costs are gate/up/down projections; reducing CPU pixel
work further cannot remove that GPU cost. The complete timestamp trace is
[compressed JSON](benchmarks/gemma4-fusions-2026-10-17/camera-profile.json.gz).

Other render processes were active during these measurements. The changing-camera
smoke run completed 60 decisions per stage, exercised stop/restart, and reported
20.71 decisions/s at Balanced and 23.21 at Fast. This short contended run neither
meets 60 FPS nor replaces the earlier idle sustained measurement. Image budgets,
model weights, and tolerances remain unchanged; M4 performance is unmeasured.

## Further decoder candidates and idle recheck

A fixed-32 subgroup decoder attention candidate keeps the original reduction tree,
online-softmax boundaries, ascending value accumulation, and branch offsets. It
reduces workgroup barriers from 40 to four per 128-key tile. The
[complete-output guard](benchmarks/gemma4-fusions-2026-10-17/decoder-attention-subgroup-chrome.json)
passes bit-for-bit for 256/512-wide heads, causal/sliding/bidirectional attention,
ragged tiles, independent branches, zero/peaked inputs, and guarded output tails.
Representative prefix and suffix kernels improve about 13–22%; the 512-wide
two-row tail improves less than 5% and should retain the baseline. A separate
four-key prefetch experiment changed low-order output bits and was removed.

The [F16 batched MLP guard](benchmarks/gemma4-fusions-2026-10-17/batch-mlp-kernel-nvidia-firefox.json)
preserves logical split-T=1 while sharing gate/up input tiles. All gate/up split
partials and final GELU outputs match exactly at physical T97 and T241 for the
6144/12288-by-1536 shapes. Kernel medians improve 9–17%. The receipt includes an
owned NVIDIA Firefox GPU context. This candidate still requires exact-length,
loaded-weight calibration and a successful bounded scratch allocation before use.

The [two-question loaded F16 comparison](benchmarks/gemma4-fusions-2026-10-17/batch-mlp-loaded-two-nvidia-firefox.json)
exercises the paired batch kernel at T97 with scratch ready and reports zero raw
probability delta across all three budgets. A five-question calibration selected
only a shared-tail shape that has no eligible batch op; that run is not runtime
fusion evidence. Shape filtering now rejects candidates absent from the active
graph. Firefox's roughly 100 ms wall-clock steps are not a kernel-speed result.

Adjacent vision post-normalization/residual and next normalization now share one
workgroup while writing both original outputs. This removes another 31 dispatches
on E2B, for 64 total from the norm/rotary/pool/adjacent-norm changes. Both
[Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-residual-norm-chrome.json) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-residual-norm-firefox.json)
compare both complete outputs exactly, including independent clamp modes,
disabled-input poison, signed zero, and output tails. At 567 rows/width 768,
the two-pass kernel median falls from 8.75 to 7.31 microseconds. These small
stage savings do not imply a comparable frame-rate gain. Actual-pack comparisons
in [Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-residual-actual-chrome.json)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-residual-actual-firefox.json)
retain zero enabled/disabled delta and the pinned F32 error under 1e-4.

The [generic](benchmarks/gemma4-fusions-2026-10-17/weight-expand-final-generic-chrome.json)
and [wide](benchmarks/gemma4-fusions-2026-10-17/weight-expand-final-wide-chrome.json)
weight-expansion probes preserve every output/partial bit, but expansion raises
weight storage to 3.56 times Q8 plus scales. Generic kernels are 1–11% slower.
One wide-only sample wins within that slower family, but still loses against the
existing generic kernel. No expanded-weight cache is enabled.

Profiler timestamps and output readback now begin mapping together rather than
serially. Firefox's [upstream polling investigation](https://bugzilla.mozilla.org/show_bug.cgi?id=1870699#c55)
explains why sequential mappings can add timer intervals despite short GPU work.
The change removes an avoidable profiling wait; it does not fix browser polling
or establish a new camera throughput result.

The [idle sustained camera recheck](benchmarks/gemma4-fusions-2026-10-17/camera-sustained-idle.json)
uses the existing enabled source/PLE/vision fusions and 240 changing frames per
stage. It predates the additional decoder candidates above. Other GPU render
work had cleared before this run.

| Stage | Decision p50 / p95 | Completed decisions/s |
|---|---:|---:|
| Detail | 47.26 / 49.01 ms | 20.86 |
| Balanced | 38.31 / 55.14 ms | 24.02 |
| Fast | 31.20 / 48.64 ms | 29.90 |
| Balanced repeat | 39.39 / 43.58 ms | 24.56 |

The measured camera path remains below 60 completed decisions/s. Stage-level wins
do not establish an end-to-end improvement, and no image budget or precision was
reduced for these measurements.

## Enabled decoder/cache and final camera check

The fixed-32 decoder attention kernel is now selected per loaded device after
exact output-bit checks and stable paired GPU timing wins. Compilation, feature,
deadline, numerical, and performance failures retain the original attention.
Prefill selection is calibrated at 89/97 rows; the separate two-row tail can
remain portable when its gain is below 5%. The calibration wins are representative
samples, not a guarantee for every context length.

The causal text prefix before the first image now retains K/V across frames. The
cache derives its boundary from the image span and checks exact preceding tokens,
logical geometry, execution policy, device, and weight lifetime. It never reuses
image-derived K/V as static text. At Balanced it retains 3,280,896 bytes and skips
38 text rows, leaving 51 dynamic prefix rows. Full logical split-K policy remains
unchanged, including the calibrated F16 R3 MLP. Failed or partial captures cannot
commit. Late GPU validation errors also invalidate PLE copies after a successful
readback; a focused regression fails without that invalidation.

The vision split reducer now shares a WG64 with 64-wide head normalization and RoPE.
It sums each PART plane in the original order, materializes the projected F32
value, then applies the existing clipping, RMS tree, learned multiply, and rotation.
Single-split projections and other head dimensions keep their original path.
[Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-norm-chrome.json) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-norm-firefox.json)
guards compare every output bit. At 315 rows/12 heads/five splits, Q/K median
time falls from about 11.0 to 8.4 microseconds. This removes 48 more E2B dispatches,
for 112 total from norm/rotary/pool/reducer/adjacent-norm fusions, separately from
the 112 eliminated input clip passes. Actual-pack comparisons in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-actual-chrome.json) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-actual-firefox.json)
retain zero enabled/disabled output delta and the pinned reference error.

Final five-question same-load off/on/off runs in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/latest-all-five-chrome.json) and
[Firefox F16](benchmarks/gemma4-fusions-2026-10-17/latest-all-five-firefox.json)
exercise the cache, 112 removed vision passes, and eligible decoder kernels across
all three budgets. Both have zero raw probability delta and no threshold flips.
Firefox's receipt confirms ready/selected batched MLP scratch; Chromium confirms
actual subgroup attention dispatches. Two-question checks also pass.

The final [sustained camera run](benchmarks/gemma4-fusions-2026-10-17/camera-latest-sustained.json)
uses 240 changing frames per stage with the same image and question settings:

| Stage | Decision p50 / p95 | Completed decisions/s |
|---|---:|---:|
| Detail | 46.13 / 47.53 ms | 21.38 |
| Balanced | 35.47 / 36.97 ms | 27.77 |
| Fast | 28.09 / 30.28 ms | 34.73 |
| Balanced repeat | 35.73 / 37.23 ms | 27.58 |

The new [profile summary](benchmarks/gemma4-fusions-2026-10-17/latest-summary.json)
records about 8.09 ms vision, 7.79 ms prefix, 9.44 ms suffix, and 5.13 ms tail GPU work
at Balanced. Decoder command encoding is about 0.61 ms. Gate/up/down matrix work
remains dominant; reducing CPU image normalization further cannot close the gap.
The [full trace](benchmarks/gemma4-fusions-2026-10-17/latest-camera-profile.json.gz)
keeps coarse and fine instrumentation separate. These local results still do not
meet 60 completed decisions/s, and the new kernels have not been measured on M4.

## Compact prefix tile selection

Caching the 38 static text rows leaves 51 active prefix rows at Balanced and 28
at Fast. Retiling those rows can reduce padding while retaining the original
matrix family, precision, and split-K partition. Matmul-only globals preserve
the original virtual tile count; attention positions and full prefix K/V keep
their original logical length. Detail retains the current tile.

Selection is per shape and device, using the loaded Q8 weights. It requires
identical complete output and partial bits, finite values, intact tail guards,
and at least 5% improvement in three alternating timestamp pairs. Calibration
has a 750 ms sampling deadline, a 32 MiB scratch cap, and two cached contexts.
It preserves already selected K64 and paired prefill MLP kernels. Unsupported,
slower, noisy, or numerically different candidates retain the existing tile.

The isolated loaded-model comparisons in
[Chrome F32](benchmarks/gemma4-fusions-2026-10-17/prefix-retile-loaded-chrome.json)
and [Firefox F16](benchmarks/gemma4-fusions-2026-10-17/prefix-retile-loaded-firefox.json)
report zero raw probability delta and no threshold flips at all three budgets.
Chrome exercises 76 retiled matrices per Balanced frame and 128 per Fast frame.
Its selected matrix stages improve roughly 18–37% at Balanced and 12–23% at
Fast. These are matrix-stage measurements, not camera FPS. Firefox selects fewer
shapes before its readback-limited deadline and retains only completed guards.

The combined five-question comparison also reports zero probability delta in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/retile-latest-five-chrome.json) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/retile-latest-five-firefox.json).
The [sustained camera recheck](benchmarks/gemma4-fusions-2026-10-17/camera-retile-sustained.json)
uses 240 changing frames per stage, the same two questions and unchanged budgets:

| Stage | Decision p50 / p95 | Completed decisions/s |
|---|---:|---:|
| Detail | 46.39 / 48.02 ms | 21.27 |
| Balanced | 34.29 / 36.00 ms | 28.69 |
| Fast | 27.48 / 30.42 ms | 35.38 |
| Balanced repeat | 34.45 / 36.13 ms | 28.58 |

The recorder keeps one request in flight, uses the same loaded model, and stops
all camera tracks on exit. The receipt records a 6,000 MiB memory limit, zero
swap, about 4,193 MiB peak memory, and no CPU quota throttling. This remains below
60 completed decisions/s and does not measure M4, CUDA, or native Metal/MLX.

The [retile profile](benchmarks/gemma4-fusions-2026-10-17/retile-camera-profile.json.gz)
records 8.13 ms vision, 6.58 ms prefix, 9.41 ms suffix and 4.88 ms tail GPU work.
The [summary](benchmarks/gemma4-fusions-2026-10-17/retile-summary.json) includes the
source hashes, numerical checks, camera lifecycle checks and resource limits.

## Weight sharing across two reaction questions

The two-row kernels share each Q8 weight load and expansion across the two
independent reaction questions. F32 keeps the original lane64 dot and reduction
order for each component. F16 keeps the original half-rounded operands, scalar
dot sequence, ascending split reduction and signed-zero epilogue. It stages two
input rows while reusing one weight tile. Precision and stored weights are
unchanged.

The paired gate/up/GELU guards pass all output bits and tails in
[Chrome F32](benchmarks/gemma4-fusions-2026-10-17/matvec-two-rows-chrome.json) and
[Firefox F16](benchmarks/gemma4-fusions-2026-10-17/matvec-two-gelu-f16-firefox.json).
Paired kernel times improve roughly 36–39% and 34–40%, respectively. The plain
projection guards also pass in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/matvec-two-plain-chrome.json) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/matvec-two-plain-firefox.json), but
some small projections become slower. These are kernel timings, not frame rates.

The runtime therefore selects each shape separately from actual loaded weights,
against its current effective baseline. It requires exact complete outputs,
signed zero and tail guards before and after repeated dispatches, plus a stable
5% win in three alternating timestamp pairs. Sampling has a 1,000 ms deadline
and 32 MiB scratch cap. It tries the largest shared-tail MLP pair first. A deadline
keeps only completed guards; numerical, GPU or timing failures clear both new
selection sets. Unsupported devices retain their existing kernels. Only two
independent rows are eligible; moved source rows keep their original arithmetic.

The loaded-model comparisons in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/matvec-two-loaded-chrome.json) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/matvec-two-loaded-firefox.json)
have zero raw probability delta and no threshold flips at all three budgets.
Chrome exercises 100 new matrices per frame and rejects the slower 256×1536
projection. Firefox retains its validated shared-tail pair, exercising 20 new
matrices per frame before its deadline. The five-question path keeps the original
kernels; its [Chrome comparison](benchmarks/gemma4-fusions-2026-10-17/matvec-two-five-chrome.json)
and [Firefox comparison](benchmarks/gemma4-fusions-2026-10-17/matvec-two-five-firefox.json)
also have zero delta and confirm that the new kernels are not dispatched.

The [sustained camera run](benchmarks/gemma4-fusions-2026-10-17/camera-two-rows-sustained.json)
again completes 240 changing frames per stage with the same settings:

| Stage | Decision p50 / p95 | Completed decisions/s |
|---|---:|---:|
| Detail | 45.01 / 46.30 ms | 21.90 |
| Balanced | 32.62 / 34.19 ms | 29.95 |
| Fast | 25.88 / 27.46 ms | 37.80 |
| Balanced repeat | 32.91 / 34.39 ms | 29.75 |

The [fresh profile](benchmarks/gemma4-fusions-2026-10-17/two-rows-camera-profile.json.gz)
records 8.00 ms vision, 6.55 ms prefix, 9.67 ms suffix and 4.34 ms tail GPU work.
Its result metadata confirms 100 shared-weight matrices and 76 retiled prefix
matrices at Balanced. The counter was added after the sustained run and changes
only diagnostic metadata. These Linux results remain below 60 completed frames/s;
the new path is guarded on each device but has not been measured on M4.
The [two-row summary](benchmarks/gemma4-fusions-2026-10-17/two-rows-summary.json)
records source hashes, actual selected shapes, dispatch counts, resource limits
and validation. The tiny [decoder guard](benchmarks/gemma4-fusions-2026-10-17/two-rows-decoder-guard-chrome.json)
also passes with the current defaults.

## Fast vision split reducers and GELU

The [Fast profile](benchmarks/gemma4-fusions-2026-10-17/two-rows-fast-camera-profile.json.gz)
places vision at about 3.50 ms, dynamic prefix at 4.73 ms, independent suffixes at
9.11 ms, and selected decision rows at 4.04 ms. These are timestamped stage
measurements with profiling overhead, not camera FPS. Suffix gate/up/down remains
the largest target.

Fast uses 108 real vision patches and three split planes for each 3072×768 gate
and up projection. Both GEMMs now retain their original kernels and write to
disjoint aligned views of the existing 12 MiB split arena. One kernel then sums
each plane in ascending order, stores both F32 results in workgroup memory, and
applies the original clipped vision GELU. It preserves the vision expression,
including its behavior beyond ±10, and does not use the decoder's saturation
shortcut. If the two planes do not fit, the existing reducers and GELU remain.

This removes 32 dispatches and about 81 MiB of intermediate gate/up write/read
traffic per Fast frame. The [Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-gelu-registered-chrome.json)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-gelu-registered-firefox.json)
guards compare all outputs exactly for clipping modes, signed zeros, cancellation,
ragged widths, split counts, and poisoned tails. The representative three-pass
chain falls from about 10 µs to 4.3 µs. This saves roughly 0.09 ms across the
16 layers in isolation; it does not imply a large camera FPS gain.

Complete checkpoint vision outputs match with the fusion disabled and enabled in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-gelu-actual-chrome.json)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-gelu-actual-firefox.json).
Fast removes 32 passes; Balanced and Detail retain their unsplit projections and
remove none. Both browsers also pass the same-loaded-model off/on/off comparison
with zero raw probability delta and no threshold flips at all three budgets:
[Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-gelu-loaded-chrome.json),
[Firefox F16](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-gelu-loaded-firefox.json).
The fusion is enabled by default. Weights, image budgets, and precision stay unchanged.

The [sustained recorder run](benchmarks/gemma4-fusions-2026-10-17/camera-reduce-gelu-sustained.json)
uses 240 changing frames per stage at a requested 60 Hz, discards four warmups,
and measures completion timestamps. The same model handles every stage, only one
decision runs at a time, and camera shutdown and layout checks pass.

| Stage | Decision p50 / p95 | Completed decisions/s |
|---|---:|---:|
| Detail | 45.32 / 47.80 ms | 21.74 |
| Balanced | 32.81 / 34.16 ms | 29.96 |
| Fast | 25.04 / 27.26 ms | 39.08 |
| Balanced repeat | 32.84 / 34.25 ms | 29.90 |

Fast reports 144 removed vision passes, compared with 112 at the other budgets,
separately from the existing input-clip fusions. Its previous campaign reached
37.80 decisions/s. The new campaign's larger wall-time difference cannot be
attributed entirely to a roughly 0.09 ms isolated kernel saving. Peak workload
memory is 4,404,428,800 bytes under a 6000 MiB cap, swap is disabled, and CPU quota
throttling is zero. These measurements remain below 60 FPS and do not measure M4.

The new F32 paired suffix probes preserve the baseline R4 split partitions while
trying physical R1, R2, and R3 tiles at T97. All outputs and split partials match,
but all are slower: the best R2 result regresses about 10–11% on the two tested
matrix shapes. They remain dev-only:
[R1](benchmarks/gemma4-fusions-2026-10-17/paired-r1-f32-chrome.json),
[R2](benchmarks/gemma4-fusions-2026-10-17/paired-r2-f32-chrome.json),
[R3](benchmarks/gemma4-fusions-2026-10-17/paired-r3-f32-chrome.json).

A [GPU-decoded half-weight cache probe](../dev/gemma4-weight-half-check.html)
changes only the weight loader of an existing F16 path. It retains exact output
and partial bits, but the [primary shapes](benchmarks/gemma4-fusions-2026-10-17/weight-half-firefox.json)
improve only about 1.6–2.8% while decoded
weights use about 78% more storage than Q8 plus scales. No decoded-weight cache
is enabled. This avoids extra model residency for a gain below the existing
5% selection threshold.

The next unimplemented fusion candidates are O/DOWN reduction into vision
residual normalization, and decoder RMS-plus-scalar into the next attention or
final RMS. Static graph analysis finds up to 32 and 49 removable dispatches,
respectively. Both need original F32 storage boundaries and reduction trees;
their benefits remain unmeasured. PLE projection into GELU and tail Q normalization
into attention are smaller candidates. None of these estimates establishes 60 FPS
or M4 performance.

The [fusion summary](benchmarks/gemma4-fusions-2026-10-17/reduce-gelu-summary.json)
records final source and WASM hashes, guards, stage medians, resource limits, and
validation. JavaScript checks, Rust formatting/Clippy and WGSL rendering tests,
all three WASM builds and their ABI checks, and the package check pass. The full
GPU suite was not rerun; its previously documented Laya parity failure remains.

## Remaining fusion probes

The [gate/up reducer probe](../dev/gemma4-mlp-reduce-check.html) keeps both original
matrix kernels, stores separate partial planes, and combines the two reductions
with GELU. It reduces five dispatches to three without changing matrix arithmetic.
The [Chrome F32 receipt](benchmarks/gemma4-fusions-2026-10-17/mlp-reduce-f32-chrome.json)
has identical complete outputs and partial bits, including signed zero and tail
guards. At T97, median GPU time is 0.2584 to 0.2493 ms; at T241 it is 0.5364 to
0.5292 ms. These gains of 3.5% and 1.3% are below the unchanged 5% selection
threshold, so this probe is not enabled in the runtime. Unsplit projections keep
their original path.

The [DOWN reducer/RMS probe](../dev/gemma4-reduce-rms-check.html) replaces a split
reduction and weighted RMS residual addition with one dispatch, using a shared
F32 store before the RMS products. The first mode2-only shader changed low bits
and was rejected. Preserving the baseline factor selection and mode branches
fixes the difference: complete outputs match bit-for-bit in 43 cases for both
[shared](benchmarks/gemma4-fusions-2026-10-17/reduce-rms-corrected-chrome.json) and
[subgroup](benchmarks/gemma4-fusions-2026-10-17/reduce-rms-subgroup-chrome.json)
Chrome variants. The shared variant saves roughly 3–4 microseconds per call at
T97/T241, or about 0.05 ms across 14 suffix layers. This remains a dev prototype;
the runtime keeps its validated reducer and RMS kernels.
The [Firefox feature-requested check](benchmarks/gemma4-fusions-2026-10-17/reduce-rms-firefox.json)
also passes all cases exactly. These synthetic PART checks exercise F32 reduction
and RMS arithmetic; requesting `shader-f16` does not execute an F16 matmul.

The [narrow Q8 load probe](../dev/gemma4-q8-load-check.html) replaces a 16-byte
packed-weight load and selection with the corresponding direct 8-byte load.
Its [F32 comparison](benchmarks/gemma4-fusions-2026-10-17/q8-load-f32-chrome.json)
is exact across all outputs and split partials in 20 cases, but gains are below
1% with small regressions. It remains a prototype. WGSL's `unpack4xI8` is also
exact for raw signed bytes, but [Dawn's pinned implementation](https://dawn.googlesource.com/dawn/+/28ffc61fc935c599677cff4dddfa4a12209ce500/src/tint/lang/core/ir/transform/builtin_polyfill.cc)
polyfills it with the same shifts on Metal/Vulkan, so it is not a demonstrated
decoder speedup.

The [BN128 probe](../dev/gemma4-matmul-128-check.html) doubles the physical column
tile while retaining the original virtual BN64 split rule and per-output dot
order. It shares the input tile across more columns and uses 24 KiB workgroup
storage. Its [15-case F32 guard](benchmarks/gemma4-fusions-2026-10-17/matmul-128-f32-chrome.json)
matches every output and split partial exactly, including ragged columns. It is
slower across all measured shapes, by about 2–40%, so the runtime keeps BN64.

## Narrow prefix tiles and further exact fusions

The next October 17 pass retains four more changes. These measurements use the
local RTX 5070 Ti through Chrome/Vulkan and Firefox/Vulkan. M4 performance remains
unmeasured.

**Vision projection reduction plus residual and next norm.** O and DOWN split
reductions feed the existing residual/normalization fusion directly. The final
DOWN retains its separate reducer because no next input norm follows it. This
removes 31 dispatches per E2B image at every existing camera budget, using the
existing partial buffer and 9 KiB of workgroup storage. Fast kernels measured
about 6.5 to 5.1 microseconds; Balanced and Detail saved about 0.07 and 0.1 ms per
tower. These small kernel savings do not by themselves establish a camera FPS gain.

The [Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-residual-registered-chrome.json.gz)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-residual-registered-firefox.json.gz)
registered-kernel guards pass 86 exact cases, including all residual/norm flags,
split counts, signed zero and output tails. Actual checkpoint projector outputs
match with the feature on and off at all three budgets in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-residual-actual-chrome.json.gz)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/vision-reduce-residual-actual-firefox.json.gz).

**F32 prefix column tiles.** For calibrated R2 prefix projections with at most 32
active rows, BN32 halves the physical column tile and workgroup size. The virtual
BN64 split rule, Q8 decoding, F32 dot order and partial layout stay unchanged.
Workgroup storage falls from 16 to 8 KiB. Every loaded shape must pass exact output,
partial and tail comparisons before and after three alternating timing pairs,
with a stable gain above 5%. Calibration considers at most eight shapes, uses at
most 32 MiB of controlled scratch, and has a one-second sampling deadline.
F16, suffixes, unsupported plans, slower shapes and numerical failures retain
their existing kernels.

The [loaded-weight Chrome check](benchmarks/gemma4-fusions-2026-10-17/bn32-actual-chrome.json.gz)
selected gate/up, DOWN, O and a small projection shape. Gate/up improved about 14%,
DOWN 11%, and O 8%; Q shapes were slower and stayed unchanged. Its Fast decisions
executed 108 selected projections with zero probability delta. The
[Firefox full-model check](benchmarks/gemma4-fusions-2026-10-17/bn32-actual-firefox.json.gz)
verified the F16 fallback with zero selected BN32 projections and zero probability
delta. Registered F32 output and partial guards also pass in both browsers.

**PLE RMS, scalar and next weighted RMS.** The new shader stores the same scaled
F32 residual before applying the following norm. It preserves both reduction trees,
the scalar operation and epsilon. The planner pairs only matching globals and buffer
ranges within each physical graph; it crosses only the independent PLE prologue.
It leaves the source-layer-to-shared-layer boundary unchanged. This removes 48
dispatches per two-question camera decision. The optional loaded-weight gate checks
both residual and norm outputs across every layer and the final norm. Failure keeps
the previously validated RMS fusions enabled and disables only the new fusion.
The shared validation deadline is 750 ms; E2B adds 1.73 MiB of temporary GPU storage.

The [registered Chrome guard](benchmarks/gemma4-fusions-2026-10-17/rms-scale-next-registered-subgroup-chrome.json.gz)
passes 60 exact cases. Fair timings use one compute pass for each path, including
both baseline dispatches. At T=2/28/97, the saving is about 1.6–1.9 microseconds per
occurrence. Portable shared-memory guards pass in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/rms-scale-next-fair-shared-chrome.json.gz)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/rms-scale-next-fair-shared-firefox.json.gz).
Full-model on/off/repeat checks in
[Chrome](benchmarks/gemma4-fusions-2026-10-17/rms-scale-next-actual-chrome.json.gz)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/rms-scale-next-actual-firefox.json.gz)
report zero probability delta, and the loaded-weight gates compare both outputs
exactly across all 35 layers.

**Paired BN32 prefix gate/up and GELU.** The already selected F32 prefix projections
can share an input tile and write the final activation directly. This reduces three
dispatches to one for each unsplit Fast MLP. The new kernel keeps independent gate/up
accumulators, the existing GELU expression and virtual split rule, and uses 12 KiB of
workgroup storage. Split cases retain both partial planes and the existing paired
reducer. The runtime checks that both planes fit its existing scratch buffer.
Its additional calibration accepts only matching loaded Q8 gate/up tensors that
already selected BN32, with exact checks before and after timing and a stable gain
above 5%. It considers at most two shapes, with a 750 ms sampling deadline and a
32 MiB controlled temporary-memory limit.

The [registered Chrome guard](benchmarks/gemma4-fusions-2026-10-17/prefix-gelu-bn32-registered-chrome.json.gz)
passes 38 exact cases, including every active row count from 1 to 32, ragged columns,
cancellation, saturation, inactive rows, both partial planes and tails. The
[Firefox prototype guard](benchmarks/gemma4-fusions-2026-10-17/prefix-gelu-bn32-f32-firefox.json.gz)
passes the same arithmetic across eight cases, and the
[registered Firefox check](benchmarks/gemma4-fusions-2026-10-17/prefix-gelu-bn32-registered-firefox.json.gz)
repeats the actual 6144×1536, T=28 case. The loaded-weight
[full-model Chrome comparison](benchmarks/gemma4-fusions-2026-10-17/prefix-gelu-bn32-actual-chrome.json.gz)
reports zero probability changes and 14 selected prefix fusions in Fast mode.
Its calibrated gate/up/GELU latency improves about 16% versus the already narrowed
separate kernels, using 13.24 MiB of controlled calibration scratch.

With every new path toggled together, the
[Chrome](benchmarks/gemma4-fusions-2026-10-17/next-fusions-all-chrome.json.gz)
and [Firefox](benchmarks/gemma4-fusions-2026-10-17/next-fusions-all-firefox.json.gz)
full-model checks report zero probability delta across four images, both default
questions and all three budgets, with an unfused repeat. Fast removes 107 additional
dispatches per decision: 31 in vision, 48 adjacent normalization dispatches, and 28
from the 14 prefix MLP pairs. Unsupported F16 prefix paths retain their previous
projections; the normalization and vision fusions still run.

Review also found an existing long-request boundary bug: slicing immediately after
a fused RMS could execute its scalar again in the next command chunk. RMS and MLP
fusions now require their actual partner operations to remain in the same slice.
A regression reproduces the double scalar before the fix. The
[257-token GPU guard](benchmarks/gemma4-fusions-2026-10-17/fusion-chunks-tail-chrome.json.gz)
checks all final hidden values bit-for-bit with RMS-next toggled off and on.

Further measured candidates remain disabled:

- [F32 K64 suffix tiling](benchmarks/gemma4-fusions-2026-10-17/batch-k64-f32-chrome.json.gz)
  is exact but about 47–50% slower on the main shapes.
- [WG128/BM28 suffix tiling](benchmarks/gemma4-fusions-2026-10-17/bm28-f32-chrome.json.gz)
  is exact but about 40% slower on the main shapes.
- [Compile-time N/K constants](benchmarks/gemma4-fusions-2026-10-17/shaped-f32-chrome.json.gz)
  preserve outputs and partials but regress the actual projection shapes.
- [Layer interleaving](benchmarks/gemma4-fusions-2026-10-17/interleave-all-chrome.json.gz)
  preserves raw hidden words and prefix/final K/V hashes through cold and warm
  caches. Timing includes state restoration copies and verifies equal synthetic
  copy work against the segmented control. It is about 1–4% slower than the original
  decoder across Fast, Balanced and Detail, so it stays out of the runtime.

The updated [Fast profile](benchmarks/gemma4-fusions-2026-10-17/camera-next-fusions-fast-profile.json.gz)
keeps the normal caches and submission boundaries. Four coarse timestamp samples
give median GPU times of 3.35 ms for vision, 4.05 ms for the shared prefix, 9.02 ms
for question suffixes and 4.03 ms for the selected-row tail. The decoder remains
the main bottleneck. These instrumented phase times are diagnostic measurements,
not sustained camera FPS.
The separate kernel-level run identifies suffix gate/up/DOWN as the largest
projection costs, at about 1.85/1.86/1.83 ms per decision. Those per-kernel
boundaries add instrumentation overhead and are not used to calculate camera FPS.

The [sustained camera repeat](benchmarks/gemma4-fusions-2026-10-17/camera-next-fusions-sustained-repeat.json.gz)
uses the real Recorder, two default questions, one loaded model and a changing
60 Hz input track. Each budget completes 240 decisions and discards four warmups.

| Budget | Median decision | p95 decision | Completed decisions/s |
|---|---:|---:|---:|
| Detail 70 | 44.58 ms | 46.14 ms | 22.08 |
| Balanced 40 | 33.03 ms | 34.05 ms | 29.93 |
| Fast 16 | 24.67 ms | 27.06 ms | 39.44 |
| Balanced 40 repeat | 33.06 ms | 34.14 ms | 29.91 |

The first [sustained run](benchmarks/gemma4-fusions-2026-10-17/camera-next-fusions-sustained.json.gz)
measured 39.39 Fast decisions/s but recorded 57 soft memory-limit events. The
repeat raises only the soft limit from 4200 to 5000 MiB, retaining the 6000 MiB
hard limit, zero swap and four-CPU quota. It records no soft/hard memory-limit,
OOM or CPU-throttling events; peak charged memory is 4,525,920,256 bytes.
Both runs retain one decision in flight, stop the camera track and clear the
video on exit, preserve Reaction Recorder as the last tab, and report no page errors.
The earlier retained build measured 39.08 Fast decisions/s in a separate run.
The end-to-end gain is modest; these measurements do not establish 60 FPS or
M4 Max performance.
The [current validation summary](benchmarks/gemma4-fusions-2026-10-17/next-fusions-summary.json)
records runtime/WASM hashes, packaged artifact, full-model parity, camera limits
and the unmet 60 FPS target. These changes are local and staged; they are not deployed.

## Further fusion measurements

The [BN32 suffix guard](benchmarks/gemma4-fusions-2026-10-17/suffix-gelu-bn32-f32-chrome.json.gz)
checks the final activation, both split partial planes, inactive padding and tails
before and after timing. Its narrower paired F32 gate/up tile is exact but about
7–12% slower on the two actual MLP shapes. The
[F16 comparison](benchmarks/gemma4-fusions-2026-10-17/suffix-gelu-bn32-first-f16-firefox.json.gz)
compares against the existing paired F16 kernel, rather than two separate
projections. At T97, 6144×1536 it measures 0.303536 ms for the current kernel and
0.303952 ms for BN32. Neither suffix candidate is enabled.

Replacing the two-row tail's shared-memory reduction with separate scalar
subgroup trees is also exact, but its
[first guard](benchmarks/gemma4-fusions-2026-10-17/matvec-two-subgroup-quick-chrome.json.gz)
makes the paired GELU projections about 5% slower. A branch intended to skip
unused subgroups is rejected by the compiler with
`'subgroupShuffleDown' must only be called from subgroup uniform control flow`;
the [receipt](benchmarks/gemma4-fusions-2026-10-17/matvec-two-subgroup-active-v2-quick-chrome.json.gz)
retains the diagnostic. No uniformity diagnostic is suppressed.
The [uniform vector variant](benchmarks/gemma4-fusions-2026-10-17/matvec-two-subgroup-vector-full-chrome.json.gz)
instead executes every subgroup, preserves each component's original reduction
tree, and omits the unused final broadcast. Its synthetic actual-shape paired
GELU kernels improve about 7–8%, with exact output bits, signed zeros, cancellation,
saturation and tail guards. These isolated kernel timings are not camera FPS.

The refinement is enabled only for fixed-32 F32 devices after calibration against
the already selected two-row kernel. It uses the existing one-second calibration
deadline and 32 MiB scratch limit, requires exact loaded-weight outputs and tails
before and after timing, and accepts only a stable improvement of at least 5%.
Optional compilation, timing or numerical failures retain the previous kernels.
The default and F16 rendered shaders remain byte-identical.
The [registered WASM guard](benchmarks/gemma4-fusions-2026-10-17/matvec-two-subgroup-vector-registered-chrome.json.gz)
passes every case. In the
[loaded-model Chrome comparison](benchmarks/gemma4-fusions-2026-10-17/two-row-vector-model-chrome.json.gz),
the two paired GELU shapes improve about 8%, and the unchanged two-question image
workload uses 76 subgroup dispatches per candidate decision. All four image
fixtures and three budgets preserve all 3072 hidden words and raw probabilities
exactly through disabled/enabled/disabled comparisons.
The [Firefox F16 comparison](benchmarks/gemma4-fusions-2026-10-17/two-row-vector-model-firefox.json.gz)
preserves the same outputs with zero subgroup dispatches and reports
`f16-baseline` as the fallback reason. These NVIDIA browser measurements do not
establish an M4 Max speed improvement.

The [sustained Recorder run](benchmarks/gemma4-fusions-2026-10-17/camera-tail-vector-sustained.json.gz)
retains one loaded model, the two default questions, the same image budgets and
a changing 60 Hz input track. Each phase completes 240 decisions with four
warmups discarded.

| Budget | Median decision | p95 decision | Completed decisions/s |
|---|---:|---:|---:|
| Detail 70 | 44.59 ms | 45.86 ms | 22.09 |
| Balanced 40 | 33.09 ms | 33.86 ms | 29.96 |
| Fast 16 | 24.73 ms | 26.51 ms | 39.53 |
| Balanced 40 repeat | 32.66 ms | 34.29 ms | 29.97 |

This is close to the previous separate run's 39.44 Fast decisions/s; it does not
establish a material end-to-end speed gain. The kernel improvements are local to
the selected tail projections. The run records one decision in flight, no page
errors, stopped tracks and cleared video, no overflow, and Reaction Recorder as
the last tab. Peak charged memory is 4,575,547,392 bytes under the unchanged
6000 MiB hard limit and zero swap. Memory-limit, OOM and CPU-throttling events are
all zero. The [runtime and artifact summary](benchmarks/gemma4-fusions-2026-10-17/tail-vector-summary.json)
records the exact source/WASM hashes and packaged artifact. The 60 FPS target
remains unmet, and these changes are local rather than deployed.

Directly binding cached PLE vectors removes 35 copies and their forced compute
pass boundaries, but the
[same-model Fast comparison](benchmarks/gemma4-fusions-2026-10-17/ple-alias-fast-chrome.json.gz)
does not show a speed benefit. Both modes preserve all 3072 decoder hidden words
and raw probabilities exactly, with 35 cold captures and 35 warm cache hits.
The copy path's whole-decoder GPU median is 17.669680 ms versus 17.709248 ms for
the alias path; wall medians are 25.315 ms and 25.910 ms. The timestamp interval
includes encoded buffer copies and excludes vision work. The alias remains a
development experiment.

An address-only split K/V attention variant also preserves output bits in its
[first copy-inclusive comparison](benchmarks/gemma4-fusions-2026-10-17/attention-split-copy-first-chrome.json.gz).
The initial implementation branches at each scalar load. Portable local/global
kernels take 0.126/0.216 ms versus 0.075/0.108 ms for joined attention including
the four copies. The subgroup version is also slower. Removing copies alone
does not justify enabling this implementation.

Hoisting the K buffer choice outside the dot loop and traversing prefix then
suffix V keys in their original order removes that regression. The
[hoisted Chrome guard](benchmarks/gemma4-fusions-2026-10-17/attention-split-hoisted-full-chrome.json.gz)
passes all 94 portable/subgroup cases exactly, including branch masks, ragged
tails, zero/cancellation inputs and poisoned output tails. The
[Firefox guard](benchmarks/gemma4-fusions-2026-10-17/attention-split-hoisted-full-firefox.json.gz)
passes its portable cases. Copy-inclusive local subgroup attention measures
0.057808 ms versus 0.066016 ms for the joined baseline, about 14% higher kernel
throughput. Global subgroup attention changes from 0.093856 to 0.092784 ms, below
the runtime selection threshold. The
[probe summary](benchmarks/gemma4-fusions-2026-10-17/split-attention-probe-summary.json)
records shader hashes and the 32 MiB resource bound. This remains a development
prototype and is not included in the sustained camera result above.
The subsequent [full-model comparison](benchmarks/gemma4-fusions-2026-10-17/split-attention-model-all-chrome.json.gz)
uses one loaded model, four image fixtures and all three budgets. Cold fallback
and warm paired decisions preserve every hidden word and raw probability exactly;
each enabled warm decision removes 11 joins and 44 copy commands. It does not
establish a consistent speed benefit. Balanced decoder GPU medians change from
20.397 to 21.101 ms and wall medians from 32.838 to 33.318 ms. Fast GPU medians
change from 17.485 to 17.633 ms while wall medians change from 25.303 to 24.815 ms.
The inconsistent results keep this variant out of the runtime.

Serializing the original split-K partitions within the paired MLP workgroup
also retains exact final activations and tails. It eliminates the PART stores
and separate GELU reducer, while preserving each partition and its ascending
reduction order. The [F16 primary](benchmarks/gemma4-fusions-2026-10-17/mlp-serial-split-first-f16-firefox.json.gz)
measures 0.311472 ms against 0.304704 ms for the current paired path. The
[F32 primary](benchmarks/gemma4-fusions-2026-10-17/mlp-serial-split-first-f32-chrome.json.gz)
measures 0.259552 ms against 0.257440 ms for the effective separate-projection
baseline. Neither is enabled.

Register prefetch of the next Q8 K32 block also preserves every activation,
partial sum, mode branch and output tail before and after timing. The
[three-shape Chrome comparison](benchmarks/gemma4-fusions-2026-10-17/q8-prefetch-main-shapes-chrome.json.gz)
is about 5% slower for gate, wide gate and DOWN. It remains a development probe.

The [matrix capability probe](benchmarks/gemma4-fusions-2026-10-17/matrix-capability-chrome.json.gz)
queries Chrome 153 on the local NVIDIA adapter without creating a device or
dispatching work. It exposes only integer subgroup matrix configurations
(i8 to i32 and u8 to u32), with no F32 or F16 configuration. These cannot directly
replace the existing floating-point arithmetic. The recorded `metricMs: 1` is
the benchmark wrapper's contract sentinel, not a performance result. M4 browser
capabilities and performance remain unmeasured.

## Smaller paired F16 suffix tile

The paired gate/up/GELU kernel now has an optional 104-row physical tile for
independent suffixes with 97–104 active rows. Thirteen accumulators cover these
rows in one workgroup tile, while the existing 64-row logical split policy,
ordered FP32 dot products, F16 tile casts, partial planes and GELU stay unchanged.
The tile uses 14,848 bytes of shared memory. Other row counts retain their current
kernel.

The [complete-output guard](benchmarks/gemma4-fusions-2026-10-17/suffix-tall104-guards-firefox.json.gz)
checks active outputs, both partial planes, NaN input padding and output tails
before and after timing. The actual 6144×1536 matrix at T97 changes from 0.302472
to 0.269688 ms, about 11% less GPU time. T104 also improves; T105 becomes slower,
which is why selection stops at 104. Large T224/T241 cases exceeding the existing
64 MiB temporary-memory cap are explicitly skipped. The
[registered WASM guard](benchmarks/gemma4-fusions-2026-10-17/suffix-tall104-registered-primary-firefox.json.gz)
repeats the exact-output check with the built shader.

Calibration compares the new tile against the already selected paired kernel,
requires a stable gain above 5%, and retains that kernel on any optional failure.
The earlier 112-row candidate had its strongest win on the wide matrix used by
the two-row shared tail, outside this batched hot path. Its
[full-model check](benchmarks/gemma4-fusions-2026-10-17/tall-model-firefox.json.gz)
preserved outputs but failed the positive-selection requirement after calibration
spent its deadline on buffer readbacks. It was replaced rather than promoted.

The new calibration checks every output and partial word on the GPU, including
finite values, signed zero and padding sentinels. It reads a 16-byte status before
and after timing instead of mapping the full buffers. Real GPU negative checks
in [Chrome](benchmarks/gemma4-fusions-2026-10-17/batch-validator-chrome.json.gz) and
[Firefox](benchmarks/gemma4-fusions-2026-10-17/batch-validator-firefox.json.gz)
detect deliberately corrupted final partials, signed zero, NaN/Inf and tails.
The total 750 ms calibration deadline and 64 MiB scratch cap are unchanged.

The [loaded-model comparison](benchmarks/gemma4-fusions-2026-10-17/tall104-model-firefox.json.gz)
now selects and executes the candidate at all three image budgets. Four image
fixtures, two questions and off/on/off comparisons preserve all hidden words and
raw probabilities exactly. Each enabled decision uses 14 new tile dispatches.
Calibration finishes in 550.44 ms and measures the loaded matrix at 0.324664 ms
versus 0.287528 ms, an 11.4% GPU-time reduction. This establishes a kernel gain,
not a new camera FPS result or an M4 measurement.

The analogous single-projection F32 tile is not enabled. Its
[gate comparison](benchmarks/gemma4-fusions-2026-10-17/matmul-tall104-first-chrome.json.gz)
is exact but slower, 0.136184 ms versus 0.126936 ms for the current BM56 kernel.
The [DOWN comparison](benchmarks/gemma4-fusions-2026-10-17/matmul-tall104-down-chrome.json.gz)
is effectively unchanged at 0.126216 versus 0.125808 ms. F16 and F32 therefore
retain separate selection policies.

## PLE reduction and GELU

Each eligible non-shared layer's PLE gate projection has a single immediate
consumer: GELU followed by multiplication with the PLE vector. The final split-K
reducer now performs those operations after materializing the original ordered
sum in shared F32 storage. It keeps the existing matmul family, precision,
logical split count, bias/mode branches and GELU saturation rules. Unsplit
projections, two-row matvec tails, missing pipelines and separated operation
slices retain the original GELU operation.

The [registered Chrome guard](benchmarks/gemma4-fusions-2026-10-17/ple-reduce-registered-chrome.json.gz)
and [Firefox guard](benchmarks/gemma4-fusions-2026-10-17/ple-reduce-registered-firefox.json.gz)
check complete output bits, finite values, poisoned disabled inputs, signed zero
and tails before and after timing. They cover all reducer row policies, cached
logical lengths, ragged dimensions and bias/mode branches. At the Fast prefix
and suffix shapes, Chrome GPU times change from 0.004116/0.004192 ms to
0.003044/0.003068 ms. This saves roughly one microsecond per eligible pair.

The [isolated Chrome model comparison](benchmarks/gemma4-fusions-2026-10-17/ple-reduce-isolated-model-chrome.json.gz)
and [Firefox comparison](benchmarks/gemma4-fusions-2026-10-17/ple-reduce-isolated-model-firefox.json.gz)
preserve every hidden word and probability across four images and all three
budgets. Each enabled warm decision removes 28 GELU dispatches. The other
selected dispatch counts and batch-MLP selections remain unchanged.

This required separating static K/V invalidation from matmul calibration keys:
PLE toggles must recapture static K/V, while retaining the same retile, BN32 and
prefix-GELU selections. The first comparison changed those selections and is
not evidence of the fusion's performance. Focused checks cover separate K/V
capture, retained tuning, graph identity and the 257-token chunk fallback.

The [sustained production Recorder run](benchmarks/gemma4-fusions-2026-10-17/camera-paired-fusions-sustained.json.gz)
measures 240 completed decisions per phase with a changing 60 Hz synthetic
camera and four discarded warmups:

| Mode | p50 / p95 decision time | Completed decisions/s |
| --- | --- | --- |
| Detail | 44.50 / 45.95 ms | 22.14 |
| Balanced | 32.99 / 33.97 ms | 29.95 |
| Fast | 24.69 / 27.24 ms | 39.41 |
| Balanced repeat | 32.99 / 34.17 ms | 29.93 |

It keeps one decision in flight, reuses the same model, stops every track and
clears the video. Peak cgroup memory is 4,555,771,904 bytes under the 6,000 MiB
hard cap, with no memory pressure, OOM, swap or CPU throttling events. This does
not establish an overall improvement over the earlier 39.53 Fast result.
The 60 FPS target remains unmet, and M4 performance remains unmeasured.

The [same-model webcam off/on/off comparison](benchmarks/gemma4-fusions-2026-10-17/camera-ple-reduce-paired-v2.json.gz)
confirms zero/28/zero PLE dispatches and constant unrelated kernel counts, but
throughput drifts from 35.84 to 37.93 to 38.50 decisions/s. Its instrumented
browser also records `memory.high` pressure events. These timings are unsuitable
for attributing a camera FPS gain to the fusion.

An optional F16 shared-memory row-map change also passes exact output and
partial checks, but the [paired BM104 comparison](benchmarks/gemma4-fusions-2026-10-17/suffix-half-bank-first-firefox.json.gz)
is effectively unchanged: 0.270656 versus 0.271208 ms. It remains development-only.

The [paired-fusions artifact summary](benchmarks/gemma4-fusions-2026-10-17/paired-fusions-summary.json)
records that revision's source and receipt hashes, selection bounds, package validation
and the remaining FPS and hardware measurement limits.

## Additional fusion probes

The contiguous K/V arena prototype lets eleven layers' prefix and suffix
producers write directly into their retained per-layer K/V buffers. Original
attention shaders then read the same contiguous layout. It removes 88 copies
and 22 copy-related pass boundaries per warm decision, using about 2.1 MiB of
extra cache capacity. Cold capture, shared source layers, and profiling retain
the original path.

The [full-model comparison](benchmarks/gemma4-fusions-2026-10-17/kv-arena-all-chrome.json.gz)
covers four images and three budgets, with 288 measured decisions, cold-path
comparisons, and unchanged loaded-kernel selections. Every paired comparison
matches all 3,072 hidden words and raw probabilities exactly. Static cache rows
and poisoned allocation tails remain intact, and warm graphs are reused.
The measured GPU intervals do not show a consistent gain, so this prototype
remains disabled and is not part of the production runtime.

The [T97 dead-row probe](benchmarks/gemma4-fusions-2026-10-17/tail-mask-gate-fixed-chrome.json.gz)
skips only unused accumulator rows in the existing F32 BM56 kernel. Complete
outputs, split partials, modes, bias cases and allocation tails match by bits
before and after timing. The gate matrix regresses from 0.126904 to 0.129800 ms,
so the production kernel remains unchanged.

Combining nine vision submissions into one preserves full hidden outputs and
raw probabilities in both [Chrome](benchmarks/gemma4-fusions-2026-10-17/vision-submit-chrome.json.gz)
and [Firefox F16](benchmarks/gemma4-fusions-2026-10-17/vision-submit-firefox.json.gz).
The [changing-camera comparison](benchmarks/gemma4-fusions-2026-10-17/camera-vision-submit-v2-summary.json.gz)
measures 120 Fast decisions per split/single/split phase after four warmups:
38.93, 39.47, and 39.28 completed decisions/s. Median vision submission wall
time is 1.385, 1.108, and 1.342 ms, respectively. This small throughput
difference is insufficient to establish a reliable FPS gain. The default stays
split. Every decision reports 9/1/9 vision submits; the original receipt's
summary field accidentally contained unrelated dispatch counts, corrected in
the linked derived summary from its unchanged per-decision samples.

Preview callbacks observe 56.80/60.05/59.73 frames/s, separately from completed
decisions. One decision remains in flight, all tracks stop, and the video clears.
A mid-run resource snapshot records a 4,535,844,864-byte peak and zero memory
pressure, OOM, swap or CPU throttling events; it is not a final resource audit.

## Compact image preparation

The internal image path now fuses patch packing with vision normalization.
It writes only active pixels, eliminating the padded patch allocation and the
subsequent full scan, allocation and normalization copy. A 256-entry lookup
table preserves both original F32 rounding steps:
`f32(2 * (f32(byte / 255) - 0.5))`. Bicubic coefficients, horizontal U8
rounding, dimensions, patch order and image budgets remain unchanged.

Private handles expose dimensions and token counts only. Pixel ownership
transfers once to the vision encoder; failures, aborts and successful uploads
clear owned pixels. Public `prepareGemma4Image` and arbitrary patch-input
validation keep their existing contracts. Noncompact execution retains the
public processor.

The repeatable CPU comparison is:

```sh
node scripts/bench-gemma4-image-prep.mjs > /tmp/gemma4-image-prep.json
```

One bounded Node 22.22.2 process checks all pixels and geometry for four rotating
320×240 frames, then measures 60 alternating pairs after 20 warmup pairs per
budget. The [receipt](benchmarks/gemma4-fusions-2026-10-17/gemma4-compact-prep-current-public-cpu.json.gz)
includes preprocessing, planning and cleanup:

| Budget | Public median / p95 | Compact median / p95 | Saved array allocation/frame |
| ---: | ---: | ---: | ---: |
| 16 | 0.963 / 1.292 ms | 0.802 / 0.879 ms | 442,368 bytes |
| 40 | 1.940 / 2.328 ms | 1.419 / 1.580 ms | 1,105,920 bytes |
| 70 | 2.987 / 3.727 ms | 2.232 / 2.707 ms | 1,935,360 bytes |

Both [Chrome F32](benchmarks/gemma4-fusions-2026-10-17/compact-pixels-model-chrome.json.gz)
and [Firefox F16](benchmarks/gemma4-fusions-2026-10-17/compact-pixels-model-firefox.json.gz)
compare four images and three budgets in off/on/off runs. All 3,072 hidden words
and raw probabilities match exactly. Every candidate decision positively
exercises compact preparation, with unchanged kernel selections, dispatch
counts and split submission policy. These CPU measurements are separate from
camera FPS.

The [camera off/on/off run](benchmarks/gemma4-fusions-2026-10-17/camera-compact-pixels.json.gz)
keeps the same model and split submission policy, with 120 measured Fast
decisions per phase. Completed throughput is 39.33/39.52/38.98 decisions/s;
median decision time is 24.83/24.73/25.08 ms. This does not establish a material
overall FPS improvement. The candidate's preview callbacks remain near 60 Hz,
separately from decision throughput. Final resource counters record a
4,467,159,040-byte peak, no pressure/OOM/CPU-throttling events and zero allowed
swap. Every phase stops its camera tracks and clears the preview.

Compact preparation is enabled by default for the internal compact-patch path
after these exactness and lifecycle checks. The 60-decision/s target remains
unmet, and M4 performance remains unmeasured.

The [final production-path camera run](benchmarks/gemma4-fusions-2026-10-17/camera-fusions-final.json.gz)
uses the default-enabled path without the proxy or experimental counters, with
240 completed decisions per phase and four discarded warmups:

| Mode | p50 / p95 decision time | Completed decisions/s |
| --- | --- | ---: |
| Detail | 43.68 / 44.94 ms | 22.70 |
| Balanced | 32.63 / 33.43 ms | 30.24 |
| Fast | 24.70 / 26.64 ms | 39.48 |
| Balanced repeat | 32.97 / 33.63 ms | 30.01 |

Peak cgroup memory is 4,517,167,104 bytes under the 6,000 MiB hard cap, with no
memory-pressure, OOM, swap or CPU-throttling events. The same model serves all
phases; one decision stays in flight, tracks stop, and the video clears. This
is essentially unchanged from the preceding independent 39.41 Fast run. An
[intermediate compact-path run](benchmarks/gemma4-fusions-2026-10-17/camera-compact-production.json.gz)
reached 40.17; that variation is not evidence of a material sustained FPS gain.

## F16 single-projection tile

The registered `matmul_tall` kernel extends the smaller physical row tile to
ordinary Q and DOWN projections. Its 32×8 workgroup covers 104 rows with thirteen
named FP32 accumulator pairs. Q8 decoding, F16 tile conversions, per-output dot
order, the original virtual BM64 split rule and the existing R4 reducer remain
unchanged. Runtime eligibility is limited to independent suffixes at T97–104,
Q2048×1536 and DOWN1536×6144, Q8 mode zero without bias, and the auto-selected
F16 baseline. Other cases retain their current kernels.

The [registered-kernel guard](benchmarks/gemma4-fusions-2026-10-17/f16-single-tall-registered-firefox.json.gz)
compares complete output and split-partial buffers before and after timing,
including mode/bias variants, signed zero, NaN input padding and allocation tails.
All checks are exact. Its synthetic-input medians improve from 0.254776 to
0.161392 ms for DOWN and from 0.092000 to 0.061040 ms for Q. Registered source
matches the independently derived prototype byte-for-byte.

The [loaded-model comparison](benchmarks/gemma4-fusions-2026-10-17/batch-single-model-firefox.json.gz)
uses the actual selected K64 `wide` baseline, which is already faster than the
synthetic guard's generic baseline. Loaded-weight calibration measures:

| Shape | Effective baseline median | Candidate median | Less GPU time |
| --- | ---: | ---: | ---: |
| DOWN1536×6144 | 0.173128 ms | 0.161072 ms | 6.96% |
| Q2048×1536 | 0.064328 ms | 0.060776 ms | 5.52% |

Both shapes are selected, with 26 new-kernel dispatches per candidate decision
and zero in both controls. Four images and three budgets produce exact raw
probabilities and all 3,072 hidden words across 36 measured decisions and 18
warmups. Other selected kernels and dispatch counts stay constant. The
[Chrome fallback comparison](benchmarks/gemma4-fusions-2026-10-17/batch-single-fallback-chrome.json.gz)
also matches exactly and records zero candidate dispatches when shader-f16 is
unavailable.

Calibration requires a stable gain above 5% in three alternating pairs, validates
all padded output and partial words before and after timing with 16-byte status
readbacks, and has a 750 ms deadline and 64 MiB controlled-scratch limit. Two
token/policy contexts are cached. Compilation, deadline, feature, numerical or
timing failures retain the baseline; later optional failures preserve earlier
successful selections. Dispatch reuses existing buffers and groups, adding no
steady-state scratch. These kernel gains do not establish an overall F16 camera
FPS gain, and M4 remains unmeasured.

The [preceding artifact summary](benchmarks/gemma4-fusions-2026-10-17/compact-and-single-summary.json)
records source and receipt hashes, runtime selection limits, final camera
measurements and local package validation. These changes have not been deployed
or published.

## Remaining work profile and exact fusion probes

A fresh [profile before GPU image preparation](benchmarks/gemma4-fusions-2026-10-17/remaining-work-profile-summary.json)
uses unchanged model weights, precision, questions and Fast image geometry.
Unprofiled controls have a 24.72 ms median decision time. Coarse GPU timestamps
attribute 9.08 ms to independent suffixes, 4.18 ms to the selected tail, 3.99 ms
to the shared prefix and 3.56 ms to vision. Fine instrumentation ranks suffix
gate, up and down projections at a combined 5.64 ms. The profiling passes and
extra maps add overhead, so these stage measurements cannot be converted into
an expected camera FPS. Matching-frame probabilities remain bit-exact.

Two [additional exact kernel probes](benchmarks/gemma4-fusions-2026-10-17/remaining-fusion-probes-summary.json)
remain disabled:

| Candidate | Current median | Candidate median | Result |
| --- | ---: | ---: | --- |
| F32 paired gate/up/GELU with K16 staging | 0.257440 ms | 0.349232 ms | 35.7% longer |
| Q8 byte permutation, gate | 0.127048 ms | 0.124176 ms | 2.26% less GPU time |
| Q8 byte permutation, down | 0.125664 ms | 0.123248 ms | 1.92% less GPU time |

The K16 fusion halves shared storage from 24 KiB to 12 KiB while preserving
K32 quantization, split partitions, ordered dots and the GELU reducer. Full
outputs and paired partials match raw bits before and after timing, including
ragged dimensions, signed-zero inputs, cancellation, saturation and allocation
tails. K16 doubles the barrier count per K32 block; the smaller storage does not
produce a speedup on this GPU.

The weight-layout probe changes only the global Q8 vector address and permutes
bytes without changing scales or precision. Both primary shapes pass full
output and partial-buffer checks across all mode/bias combinations before and
after timing. Gains are below the 5% selection threshold and require an extra
packed-weight allocation. Neither experiment changes the production path or
establishes a camera throughput gain. These measurements use the RTX 5070 Ti;
M4 Max remains unmeasured.

The exact Marlin Q8 decode was also tested inside both ordinary F32 row56 and
paired K32 matrix kernels. The current F32 kernels use integer sign extension;
this probe replaces only that decoder function. All byte lanes and signed-scale
products agree, and complete outputs and partials match before and after timing.
The primary full-MLP medians are 0.256152 ms for the current ordinary path,
0.261392 ms for ordinary Marlin and 0.262024 ms for paired Marlin. Both variants
remain disabled. These are GEMM measurements, separate from earlier matvec
decode probes.

## Exact GPU image preparation

Two registered integer kernels now resize RGBA frames directly into the existing
vision pixel buffer. The first performs horizontal Q22 bicubic filtering with
the original byte clamp. The second fuses vertical filtering, the existing
two-round F32 normalization LUT, and compact patch packing. No image budget,
model weight, precision or question changes. The normal path adds no pixel
readback.

The [registered pixel guard](benchmarks/gemma4-fusions-2026-10-17/gpu-image-kernel-summary.json)
checks 5,253,120 normalized values per browser across 21 cases, including odd
dimensions, portrait inputs, alpha, single-pixel input and edge impulses.
Chromium and Firefox match every CPU-oracle bit, geometry and allocation tail.
Both registered sources match the independently measured DEV prototype. At
320×240, the two GPU passes take approximately 0.010–0.014 ms across the three
budgets. These timestamps exclude upload, CPU encoding, clears, copies and maps.
Single cold CPU timings in those receipts are not a warmed throughput baseline.

The private prepared-image handle snapshots at most 4 MiB of raw RGBA before
the coordinator erases the original pixels. It retains consume-once provenance
and canonical geometry checks. Extra GPU preprocessing buffers are capped at
8 MiB, including minimum buffer sizes; the vision tower's existing buffers and
CPU metadata are separate. One active GPU geometry is reused, and complete raw,
intermediate and vision pixel buffers are cleared in queue order. Reusing valid
bindings avoids another GPU setup-error-scope wait every frame. New allocations
remain scoped; cached upload errors propagate to the enclosing vision error
scope. Unsupported geometry, profiler use or optional setup failures retain CPU
preprocessing. Device loss retains the existing error and cleanup behavior.

Full-model off/on/off comparisons in Chromium and Firefox match all 3,072 hidden
words and raw probabilities exactly across four images and three budgets. Each
browser exercises 36 measured decisions and 18 warmups, with GPU preprocessing
counters zero/one/zero and unchanged decoder selections, vision fusions and
submission policy. The development proxy's `compactpixels` control name is
reused for this experiment; it toggles GPU preprocessing while keeping the
current private CPU path enabled in both controls.

A sustained same-model Fast comparison completes 240 decisions per phase:

| Mode | Decision p50 / p95 | Completed FPS | CPU preprocessing p50 |
| --- | ---: | ---: | ---: |
| CPU control | 24.685 / 26.225 ms | 39.56 | 0.99 ms |
| GPU image preparation | 23.650 / 25.140 ms | 41.23 | 0.07 ms |
| CPU control repeat | 24.638 / 26.625 ms | 39.65 | 0.99 ms |

The preview remains approximately 60 Hz, with one decision in flight, stopped
tracks and cleared video after each phase. Peak cgroup memory is 4,564,680,704
bytes under the 6,000 MiB hard cap, with no memory-pressure, OOM, swap or CPU
throttling events. This is a roughly 4% completed-decision improvement on the
RTX 5070 Ti, not a 60 FPS result or an M4 Max measurement. GPU preprocessing is
enabled by default after the exact model and sustained-camera checks.

The [final runtime and artifact summary](benchmarks/gemma4-fusions-2026-10-17/gpu-image-runtime-summary.json)
records the source hashes, exact model comparisons, paired camera receipt and
local package. A clean final run after cached setup-wait removal completes 240
decisions per phase, discarding the first four:

| Image budget | Decision p50 / p95 | Completed FPS |
| --- | ---: | ---: |
| Detail 70 | 40.955 / 42.125 ms | 24.07 |
| Balanced 40 | 30.740 / 31.705 ms | 32.08 |
| Fast 16 | 23.635 / 25.945 ms | 41.02 |
| Balanced repeat | 30.678 / 31.655 ms | 32.10 |

Peak memory is 4,546,326,528 bytes, with no pressure, OOM, swap or CPU-throttling
events; camera lifecycle checks pass. An intermediate run reached 41.64 Fast
FPS. That variation does not establish an additional throughput gain from the
cached setup change. The 60 FPS target remains unmet, and M4 Max is unmeasured.
The package and site are staged locally, with no deployment or publication.

The decoder projections remain the leading measured work. Further unmeasured
fusion candidates are the attention-output split reducer with residual/FFN
normalization, and the Q/K/V split reducers with their per-head normalization
and RoPE pass. They must retain each projection's ascending split sum and the
existing normalization trees. Their likely saving is the epilogue and buffer
traffic, rather than the underlying projection time; no speedup is claimed.

## Follow-up decoder tile and barrier probes

The [decoder follow-up receipts](benchmarks/gemma4-fusions-2026-10-17/decoder-fusion-followup-summary.json)
record four further exact F32 experiments against the current registered kernels.
Each uses three alternating GPU-timestamp pairs with four repetitions per sample.
These are isolated kernel measurements on the RTX 5070 Ti, not camera FPS or M4
Max results. The 51-source production baseline remains unchanged.

| Candidate | Exactness | Measured result |
| --- | --- | --- |
| BM56 plus shifted BM48 suffix tiles | All outputs and split partials, six mode/bias combinations, before and after timing | Gate 7.6% slower; down 1.4% less time; Q 4.9% slower |
| WG128 two-row matvec with two logical lane accumulators | All output bits, finiteness and tails, including zero and GELU edge fixtures | Paired gate neutral; down 21.9–26.7% slower |
| Named 4×4 register tile, K32 and unchanged BM56 | All outputs and split partials, six mode/bias combinations, before and after timing | Gate/down/Q 14–15% slower |
| Named 8×8 register tile, K32 and physical BM128/BN128 | Same complete output and partial guards; original virtual split policy | Gate/down 45–47% slower; Q 15.6% slower |

The smaller matvec workgroup removes a shared-memory barrier while preserving
the original logical reduction tree. The larger matrix tiles increase reuse
without changing Q8 decoding or dot-product order. Neither change improves the
main measured shapes, so all four candidates remain DEV-only. No camera speedup
is claimed from these experiments.

## Query normalization fusion and RMS retention guards

A further DEV candidate moves query RMSNorm and RoPE into the attention query
load. It retains the original full 256-lane norm tree, shared normalized-query
materialization and complete attention body. K/V processing is unchanged. It
uses seven storage bindings and 5,648 bytes of workgroup storage. The initial
scope is the compact two-question tail, where it could remove 21 query-only
QKV dispatches. This first candidate still reads raw Q twice, so it avoids two
Q-sized global transfers per fused operation, rather than three.

The Chromium guard passes all output bits, finiteness, tails and input
immutability across 28 portable/subgroup cases, before and after timing. It
includes both head dimensions, branch offsets, window boundaries, zero,
cancellation, large finite inputs and no rotation. Each baseline repetition
restores raw Q before the timestamped QKV-plus-attention pass. That artificial
reset is excluded from both timings.

| Compact T2 operation | Baseline | Fused | Less GPU time |
| --- | ---: | ---: | ---: |
| Subgroup, head 256 | 0.018960 ms | 0.016128 ms | 14.9% |
| Subgroup, head 512 | 0.025680 ms | 0.023104 ms | 10.0% |
| Portable, head 256 | 0.018416 ms | 0.017344 ms | 5.8% |
| Portable, head 512 | 0.024592 ms | 0.023424 ms | 4.7% |

These are composite kernel measurements, with no camera FPS inference. T97
benefits are below the selection threshold, and the subgroup 512-wide T97 case
regresses slightly. The earlier profile's total QKV time is about 0.35 ms, so this
fusion cannot close the remaining 60 FPS gap by itself.

Firefox also passes all 14 portable cases exactly. Its T2 local/global composite
GPU times decrease by 6.7% and 13.1%, respectively. Those timestamps establish
neither Firefox camera throughput nor M4 Max performance.

The Chromium and Firefox real-model off/on/off checks pass all 3,072 hidden words
and raw probabilities exactly across four labeled images and budgets 16/40/70.
Each browser runs 36 measured decisions plus 18 warmups with actual fusion counts
0/21/0, including the compact source layer. The existing attention selection, matmul selections,
other fusions and runtime flags stay fixed. A separate source review found no
extra Q consumer or Q/context alias. A future production integration must keep
the candidate's attention variant synchronized with any runtime selection
change; the isolated prototype deliberately keeps that selection fixed.

The sustained camera check completes 240 measured decisions per phase after four
warmups, with one model instance and one decision in flight:

| Mode | Decision p50 / p95 | Completed FPS | Fusions per decision |
| --- | ---: | ---: | ---: |
| Off | 23.603 / 25.085 ms | 41.33 | 0 |
| On | 23.445 / 25.175 ms | 41.63 | 21 |
| Off repeat | 23.405 / 24.585 ms | 41.81 | 0 |

The on result lies between the two controls, so it does not establish an
end-to-end improvement. The candidate remains DEV-only. Peak cgroup memory is
4,490,092,544 bytes with no pressure, OOM, swap or CPU-throttling events. All
frames have changing synthetic hashes, tracks stop, video clears, and unrelated
kernel/runtime state remains constant. Preview callbacks initially average
45.2 Hz, then approximately 60 Hz in the latter phases; preview cadence is
separate from completed decisions. The initial camera harness omitted the
submission-mode metadata it checked; the table uses the corrected passing run.
The follow-up summary archives the final model and camera receipts and the exact
DEV instrumentation. Firefox exercises `shader-f16` without subgroups; its wall
timings retain the known completion/privacy effect and are not an FPS result.

Retaining the six D1536 intermediate RMS updates in registers removes a storage
barrier and two residual-buffer reads. That candidate fails the exactness guard
on both portable and subgroup paths at T12, residual mode 2. Stored residual Y
matches, but normalized H has 1,373 bit differences in the portable run; the first
is `0x3bcc4ca6` versus `0x3bcc4ca5`. The mismatch occurs after the stored update;
the probe does not isolate the second moment from the final register use. No
tolerance is relaxed and this candidate is disabled. The first harness attempt
also lacked an end-timestamp index; only corrected timings are archived, and no
performance claim is based on the failed guard.

A separate isolation removes only the RMS storage barrier, preserving every
other rendered-source byte. All 72 cases pass for each portable and subgroup
variant. Each lane reads only the residual elements that it wrote, with the
existing workgroup reductions retained. Kernel timing changes are below 1% in
either direction, so this exact but neutral candidate also remains DEV-only.

## Vision weight sharing and paired projections

The [vision follow-up receipts](benchmarks/gemma4-fusions-2026-10-17/vision-and-occupancy-summary.json)
separate padded image metadata from the executed tower shape. At 320×240 and
budget 16, preprocessing reports 144 padded patches and 12 soft tokens. Compact
vision executes 108 valid patch rows, using R4, physical BM56, virtual BM64 and
three gate/up splits. A T144/R3 measurement is a valid full square image shape,
but does not measure the current 4:3 camera kernel.

| Candidate and shape | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Decoder BN32 paired serial-split gate/up, T97 | 0.255488 ms | 0.275064 ms | 7.7% slower |
| BF16 packed shared gate weights, T144 | 0.051096 ms | 0.054048 ms | 5.8% slower |
| BF16 packed shared down weights, T144 | 0.050600 ms | 0.053144 ms | 5.0% slower |
| Paired BF16 gate/up plus unchanged GELU, T144 | 0.096896 ms | 0.084776 ms | 12.5% less GPU time |
| Paired BF16 gate/up plus unchanged GELU, actual T108 | 0.074280 ms | 0.069872 ms | 5.9% less GPU time |

These Chromium measurements use alternating GPU timestamp pairs. Every candidate
retains BF16 word expansion, F32 arithmetic and the original dot and split
reduction order. The paired kernel shares one input tile and retains independent
gate/up partial planes, followed by the existing reducer/GELU. It uses eight
storage bindings and 16 KiB of workgroup storage, with no F16 conversion. Full
output, partial, finiteness and tail checks pass before and after timing. The
R3 and R4 expanded guards also cover mode/bias combinations, ragged dimensions,
signed zero, cancellation and large/tiny finite BF16 fixtures.

Firefox passes the actual T108 primary guard with both bias settings and full
partials and outputs before and after timing. Its composite GPU time changes
from 0.112520 to 0.088240 ms. The expanded R3 Firefox run reaches the 240-second
wrapper timeout, so no expanded Firefox pass is claimed. Neither browser's
microkernel timestamps establish a camera FPS improvement. The actual T108
Chromium saving across 16 layers is only about 0.07 ms.

The actual-model comparison passes all 3,072 hidden words and raw probabilities
exactly across four images and budgets 16/40/70. It uses one model for 36 measured
decisions and 18 warmups, with 42 comparisons. Fast uses 0/16/0 paired dispatches;
the other budgets fall back at their actual 315 and 567 compact rows. Existing
kernel choices, runtime flags, submission mode and other fusion counts match.
An independent source review found no actionable integration defect.

The sustained camera check completes 240 measured decisions plus four warmups
per phase with one model and one decision in flight:

| Paired vision mode | Decision p50 / p95 | Completed FPS |
| --- | ---: | ---: |
| Off | 22.817 / 24.350 ms | 42.84 |
| On | 22.595 / 25.340 ms | 42.78 |
| Off repeat | 22.807 / 24.475 ms | 42.80 |

This does not establish an end-to-end gain. The candidate remains DEV-only and
runtime defaults remain unchanged. Preview callbacks are approximately 60 Hz,
separate from completed decisions. All 244 frames per phase have changing hashes,
all three streams close, video clears, and kernel selections and unrelated state
stay fixed. These unchanged sources measured differently from earlier controls;
the higher control FPS is not attributed to a code change. The 60 completed FPS
target remains unmet.

The first standalone packed-weight guard checked uninitialized PART tail words.
The corrected probe explicitly poisons active partials with NaN and initializes
both tails before dispatch. The passing receipts use that corrected guard.

## Joint vision Q/K/V projection

A further BF16 candidate computes Q, K and V from one shared input tile. Each
projection retains its original F32 dot order and six split-K planes. The three
existing reduction/normalization/RoPE kernels remain unchanged. At the actual
T108 camera shape, this replaces three matrix dispatches with one, reducing the
composite from six dispatches to four per layer, or 32 fewer across the tower.
Its matrix workgroup count is 144 versus 432 for three separate projections;
the mathematical work and normalized outputs are unchanged.

| Composite GPU time | Separate | Joint Q/K/V | Less time |
| --- | ---: | ---: | ---: |
| Chromium | 0.049912 ms | 0.040808 ms | 18.2% |
| Firefox | 0.085312 ms | 0.063008 ms | 26.1% |

Chromium passes complete PART and normalized/RoPE output bits, finite values,
input-padding and allocation tails before and after timing across normal,
signed-zero, cancellation, tiny-subnormal BF16 and large-finite BF16 fixtures.
Firefox passes the primary complete guard. The candidate declares 20 KiB of
shared storage and checks device limits. These are kernel GPU timestamps, with
no camera FPS inference. The guarded production implementation registers this
exact shader as `gemma4_vision_qkv` in all three WASM builds.

Selection uses actual first-layer Q/K/V weights on the current GPU. It checks
every partial and normalized/RoPE output bit, finite values and allocation tails
before and after three alternating timing pairs. All three pairs must save at
least 5% of composite GPU time. Calibration uses approximately 15.1 MB of
temporary storage, copies no weights, and destroys its temporary buffers and
query set afterward. The one-second cutoff is best effort: an in-flight pipeline
compilation cannot be cancelled. Missing timestamps, unsupported geometry or
limits, numerical failure, inconclusive timing and ordinary setup errors retain
the original path. Device loss propagates as an initialization error. Unselected
candidate pipeline references are released.

The steady camera path reuses existing scratch, with no new GPU buffers or CPU
readback. Profiling, explicit generic kernels, other input shapes and disabled
prerequisite fusions use the original graph. Changing those states rebuilds the
graph. Diagnostics contain only cloneable data, preserving worker messaging.

The final production-model check compares all 3,072 hidden words and raw
probabilities exactly across four images and budgets 16/40/70, including cold and
warm outputs. All 42 comparisons pass with unchanged unrelated flags and kernel
selections. Budget 16 records 0/16/0 joint dispatches in off/on/off phases; larger
budgets remain on the original path. On this RTX GPU, initialization selects the
fusion after approximately 43 ms of calibration, measuring 0.049832 versus
0.040680 ms for Q/K/V plus their unchanged reducers.

Firefox also selects the registered production kernel and passes all 14 exact
comparisons across the four Fast fixtures. Its calibration measures 0.065864
versus 0.046448 ms and completes in approximately 286 ms. This verifies the
non-subgroup browser path; its completion timings are not treated as camera FPS.
Firefox redacts adapter identity, and this final run captures no owned NVIDIA
context, so it adds browser correctness evidence without independently naming
the physical adapter. An explicit Chromium generic override also passes exact
comparison with zero joint dispatches and no candidate pipeline.

The balanced prototype camera run measured 42.85 off versus 43.53 on FPS,
with all four ABBA groups favoring the candidate. The final registered production
path repeats the same 16-block design with one model, 480 measured decisions per
mode and 184 discarded warmups:

| Production Q/K/V mode | Decision p50 / p95 | Completed FPS |
| --- | ---: | ---: |
| Off | 22.817 / 24.375 ms | 42.84 |
| On | 22.707 / 25.155 ms | 42.82 |

The final result is neutral, with two balanced groups favoring each mode. The
prototype's 1.58% FPS gain did not reproduce, so no robust overall camera gain is
claimed. Selection is justified by the guarded composite GPU saving, without
claiming that it predicts end-to-end throughput. Every measured frame uses the
actual T108 tower shape. All 16 streams close, video clears, one decision stays
in flight, and unrelated selections, dispatch counts and vision fusions remain
constant. Peak cgroup memory is 4,665,282,560 bytes, with no pressure, OOM, swap
or CPU-throttling events and no page errors. The 60 completed FPS goal remains
unmet. All model packs, precision, questions and image budgets remain unchanged.

## Compiled Vulkan shader statistics

A process-local diagnostic Vulkan layer captures executable properties for the
actual browser pipelines. It enables the advertised extension and feature only
for that process, requests compile statistics and leaves shader source unchanged.
The archived source and manifest cap capture at 256 pipelines and 16 MiB. Both
bounded output-array counts are passed to the driver with their allocated
capacities. The instrumented numerical guards pass; their timings are excluded
from performance claims.

| Shader | Registers | Shared memory |
| --- | ---: | ---: |
| Registered BF16 R3 matrix kernel | 80 | 16 KiB |
| Packed BF16 R3 matrix kernel | 64 | 12 KiB |
| Registered BF16 R4/BM56 matrix kernel | 80 | 16 KiB |
| Paired BF16 R4/BM56 matrix kernel | 80 | 16 KiB |
| Joint BF16 R4/BM56 Q/K/V kernel | 128 | 24 KiB allocated |
| Registered decoder BM56 matrix kernel | 80 | 16 KiB |
| Paired decoder BN32 serial-split kernel | 80 | 16 KiB |
| Decoder BM112/BN32 kernel | 128 | 24 KiB allocated |

Lower register and shared-memory use does not make the standalone packed kernel
faster. These compile statistics do not measure dynamic occupancy or bandwidth.
The driver's raw local-memory statistic is near 64 GiB even for small reducers;
it is archived without interpretation and is not used to claim spills. No M4
Max, native Metal/MLX or CUDA measurement is established here.

A further decoder tile covers the same 112 padded suffix rows with one BM112
tile and twice as many BN32 column tiles. It preserves the original virtual
split rule, Q8 decoding and all per-output operations, and uses 14 scalar
accumulators instead of seven pairs. Complete outputs and partials pass all
seven fixtures and all six mode/bias combinations before and after timing.
However, gate/down/Q take 75–82% longer. The declared 18 KiB shared array becomes
a 24 KiB driver allocation, and register use increases from 80 to 128 despite
the equal accumulator count. Those compile statistics identify a resource
increase; they do not establish its dynamic stall contribution. The candidate
remains DEV-only.

## Decoder projection follow-up

The [next decoder receipts](benchmarks/gemma4-fusions-2026-10-17/decoder-next-summary.json)
start from the registered Q/K/V production path. A fresh coarse trace records
3.20 ms vision, 3.99 ms prefix, 8.89 ms suffix and 3.85 ms tail work. CPU command
encoding is approximately 0.5 ms total. Fine instrumentation ranks suffix gate,
up and down at about 1.81 ms each. The final six unprofiled controls have a
21.9 ms median. Instrumentation changes timings; these are stage diagnostics,
not an inferred sustained camera rate.

A grid-stacked gate/up candidate keeps the original BM56/BN64 matrix kernel per
workgroup, including its seven vector accumulators and 16 KiB shared tile. It
assigns projection and split to the dispatch grid, writes independent partial
planes, then runs the existing fused reducer/GELU. This changes five dispatches
to two without the register increase of a paired projection within each
workgroup. At T97/N6144/K1536/S3, GPU time changes from 0.255168 to 0.231664 ms,
a 9.21% reduction. A separate two-matmul plus fused-reducer control measures
0.245880 ms, isolating a further 5.78% saving from grid consolidation. Complete
outputs and both partial planes match before and after timing in five normal,
ragged, zero, cancellation and saturation fixtures. This is a kernel result;
the following prototype full-model and camera checks establish its current limit.

The one-model off/on/off comparison passes all 42 full-hidden-vector and raw
probability comparisons across four fixtures and budgets 16/40/70. Each enabled
decision uses 14 fusions because the question suffix remains T97 at all budgets.
Other kernel choices, runtime flags, prefix counters and vision fusions match.
A loaded-weight calibration with the original five-dispatch baseline reusing one
PART buffer measures 0.257376 versus 0.233304 ms. Every timing pair exceeds the
5% selection threshold; complete partial/output guards pass before and after.
It finishes in 23.17 ms with 57.85 MB of controlled temporary and retained
context storage. The first calibration harness changed diagnostics between
comparison phases and correctly failed its isolation assertion despite exact
outputs. The passing receipt moves calibration before all recorded phases.

A balanced prototype camera check uses 16 ABBA blocks, 480 measured decisions
per mode and 184 discarded warmups:

| Grid MLP mode | Decision p50 / p95 | Completed FPS |
| --- | ---: | ---: |
| Off | 22.447 / 23.620 ms | 43.67 |
| On | 21.757 / 24.160 ms | 44.66 |

All four balanced groups favor fusion in mean completion time; the pooled rate
improves by 2.27%. The p95 becomes worse, so this does not establish less jitter.
These are point estimates without a statistical confidence claim. One decision
stays in flight, all 16 streams close, video clears, geometry is T108/soft12, and
unrelated state remains constant. Peak memory is 4,634,705,920 bytes, with no
pressure, OOM, swap, CPU throttling or page errors. The 60 FPS goal remains unmet.
The production implementation registers the exact shader as `gemma4_mlp_grid`
and selects it per loaded device and actual weights. It requires exact complete
outputs and both partial planes before and after three timing pairs, with every
pair at least 5% faster than the original single-PART baseline. Its calibration
uses 43,551,552 bytes of controlled temporary and retained storage, retaining
14,303,280 bytes for steady dispatch. It copies no weights or inputs and creates
no GPU buffers per steady frame. Missing features, explicit kernel overrides,
other shapes, F16, existing paired selections and ordinary optional errors keep
the prior path. Device loss propagates. Weight replacement and disposal release
the retained context. The compilation cutoff is best effort because in-flight
compilation cannot be cancelled.

The registered production camera run repeats the balanced 16-block workload:

| Production grid MLP mode | Decision p50 / p95 | Completed FPS |
| --- | ---: | ---: |
| Off | 22.938 / 24.730 ms | 42.61 |
| On | 22.100 / 23.845 ms | 44.17 |

All four balanced groups favor fusion, saving 0.665/0.814/0.921/0.923 ms per
completed decision. The pooled rate improves by 3.67%, with better p95 in this
run. The prototype had a worse p95, so these runs do not establish a general
jitter guarantee. One model handles 480 measured decisions per mode, with 184
discarded warmups, changing input frames and one decision in flight. All streams
close, video clears, unrelated state remains constant and no page error occurs.
Peak cgroup memory is 4,627,460,096 bytes, with zero pressure, OOM, swap and CPU
throttling events.

That camera capture precedes a cleanup-only correction to pending device-loss
detection during failed calibration. The inference path and core graph source
remain byte-identical; the exact measured source is archived separately. The
regression fails before that correction, then passes after it, and independent
review finds no remaining actionable issue. A final model proof checks the
frozen corrected source. The F16 Firefox model proof passes exact comparison
with zero grid dispatches and `ineligible-baseline` diagnostics. M4 Max remains
unmeasured; the 60 completed FPS target is still unmet.

Other exact probes remain disabled:

| Candidate | Main shape result | Disposition |
| --- | --- | --- |
| Subgroup sharing of packed Q8 loads and scales | 14–16% slower | DEV-only |
| Subgroup sharing of scales only | 22–24% slower | DEV-only |
| Alternating two K32 shared tiles, half the barriers | 36–40% slower | DEV-only |
| WG128 with unchanged physical BM56/BN64 | 27–36% slower | DEV-only |
| O split reduction plus unchanged double RMS | T97 about 20–21% less time; T28 about 25% slower | DEV-only |

All matrix probes compare full output and partial bits, finite values and tails
before and after timing. The O epilogue passes 27 fixtures for both portable and
fixed-32 subgroup trees, retaining ordered split sums and the residual storage
boundary. Its T97 saving is only about 0.035 ms across 14 suffix layers. The
short prefix regression prevents selecting it globally.

A final bounded Vulkan diagnostic requests internal shader representations for
eight captured pipelines. The numerical guard passes, but the driver returns no
representations for any of them despite advertising capture support. The grid
and original BM56 matrix pipelines both report 80 registers and 16 KiB shared
memory. No instruction disassembly or dynamic occupancy, bandwidth or spill
conclusion follows from this probe, and all instrumented timings are excluded.
Its source, raw log and guard are archived in the decoder follow-up summary.

## Worker-owned image pixels

The worker image path now borrows its already structured-cloned RGBA request
bytes through decoding and raw GPU preparation. It removes two pixel-data copies,
or 614,400 allocated bytes per 320×240 frame. Public helpers still copy by default.
The request queue owns the final wipe, after inference completes; decoded Blob
and URL pixels and CPU preparation buffers keep their existing cleanup. GPU
upload copies the bytes before asynchronous execution, and offset typed-array
views retain their original extent. No transfer-list or Safari clone behavior
changes. Independent lifecycle review found no concrete defect.

Copy/borrow/off comparisons in Chromium and Firefox each pass all 42 full hidden
output and raw probability comparisons across four images and all three budgets.
Every call checks caller bytes unchanged and worker alias false/true/false, with
unchanged model/kernel state. This proves the allocation removal preserves those
outputs; it does not establish an isolated camera FPS improvement.

## Validation

The 0.2.0 release preparation corrects the Laya guard described in the historical
results below. The old GPU gate compared Q8 weights with the unquantized PyTorch
fixture. The new [Laya guard](../dev/laya-parity-check.html) separately compares
WebGPU with single-threaded base WASM on the same Q8 pack at `5e-4`, and checks
both against the established native/CPU Q8 golden limit of `0.03`. The recorded
golden delta remains `0.024006444195624588`; all 41 public probability vectors
match exactly between WASM and Chromium WebGPU. Model weights and arithmetic
are unchanged. The supported Chromium GPU suite completes with the new guard,
including the registered QKV and grid MLP numerical checks. M4 camera timing
remains unmeasured.

The final grid MLP and worker-pixel changes passed JavaScript checks and tests,
Rust formatting/clippy and focused WGSL tests, all three rebuilt WASM ABI checks,
and package validation. All 56 unpacked package files match the workspace, and
all 60 recorded final source hashes match. The final registered grid kernel
passes exact actual-model comparisons at all three image budgets. F16 and
explicit generic overrides retain the previous execution path and pass exact
comparisons with zero grid dispatches. The generic proof harness was corrected
to accept the existing unbatched override's absent grid context, while still
requiring zero eligible/fused operations and exact full outputs. This correction
changes no runtime behavior. Independent review found no remaining actionable
correctness or ownership defect. Full package and evidence details are in the
[decoder follow-up summary](benchmarks/gemma4-fusions-2026-10-17/decoder-next-summary.json).
Final site staging passes its three WASM ABI checks. All package files and report
evidence match their staged copies. Playwright MCP verifies the Reaction route,
last tab, image-only model choices and HTTP 200 for both new helpers, all WASM
flavors, this report and the follow-up summary. Pointer automation again times
out waiting for stability; a DOM click opens the picker. A first smoke assertion
incorrectly counted the separate advanced cache-clear button as a model choice;
restricting it to model choices passes without changing application code. The
optional staged local model catalog returns 404, while the workspace catalog
returns 200. The static smoke does not load weights or start the camera; the
separate full-model and camera proofs above cover those operations.

The preceding guarded Q/K/V checkpoint passed exact actual-model comparisons
in Chromium and Firefox, explicit generic fallback, and the balanced camera
lifecycle/state check. JavaScript checks and tests, Rust formatting/clippy and
focused WGSL rendering tests, all three rebuilt WASM exports, and the fresh
package check passed. All 55 files of that earlier package matched the workspace
at that checkpoint.
Independent review found no remaining actionable correctness or lifecycle defect.
Site staging and its WASM ABI checks pass. The runtime, report and archived
receipts match their staged copies. A Playwright MCP smoke verifies the Reaction
Recorder route, image-only picker, final tab position and HTTP 200 for the new
helper, WASM builds and linked receipts. Pointer clicks timed out waiting for
stability in that browser session; direct navigation and a DOM click verified
the picker. The static staging server omits the optional local model catalog, so
its catalog request returns 404; model loading was verified separately above.
These are local results; the package and site have not been published or deployed.

The October 17 additions passed focused kernel compilation and numerical guards
in both browsers, same-model comparisons, the camera lifecycle smoke, and current
JavaScript, Rust, WASM, package, and site-staging checks. The full GPU suite was
not rerun after these additions.

JavaScript checks and tests, native WGSL rendering, Rust formatting and clippy,
all three WASM builds and ABI checks, package validation, and site staging passed.
The earlier Chrome numerical GPU suite passed its supported kernel checks, Gemma guards,
Kev parity, and cache checks. Its Laya F32 parity check remains a separate failure
at `0.024006444195624588` versus the unchanged `0.024` limit; no tolerance was
increased. The legacy Laya shader output is byte-identical across 147 HEAD/current
variants. The [runtime baseline comparison](benchmarks/gemma4-fusions-2026-10-16/laya-baseline-comparison.json)
also produces identical Laya probabilities using HEAD's shared GPU runtime, with
the same parity overage. The
[validation summary](benchmarks/gemma4-fusions-2026-10-16/gpu-validation-summary.json)
separates this existing failure from the passing checks.

To measure another machine using the local server and model packs:

```sh
uv run scripts/bench-reaction.py --binary /path/to/chrome --frames 240 --camera-fps 60 --output tmp/reaction-camera.json
```
