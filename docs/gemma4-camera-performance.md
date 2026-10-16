# Gemma 4 camera resolution and fused attention

Reaction Recorder defaults to Balanced analysis at 336 × 240 instead of enlarging a
320 × 240 camera capture to 432 × 336. It also uses fused F32 vision attention.
In the October 16 Linux camera check, completed decisions increased from 13.6 FPS
to 19.8 FPS, with a repeat Balanced stage at 18.7 FPS. These are shared RTX 5070 Ti
measurements, not M4 Max measurements or guaranteed frame rates.

## Resolution policy

| Setting | Maximum image tokens | Analysis size | Active patches | Actual image tokens |
|---|---:|---:|---:|---:|
| Fast | 16 | 192 × 144 | 108 | 12 |
| Balanced, default | 40 | 336 × 240 | 315 | 35 |
| Detail, previous behavior | 70 | 432 × 336 | 567 | 63 |

These dimensions apply to the recorder's 320 × 240 capture. Balanced processes 44%
fewer patches; Fast processes 81% fewer. Shrinking only the capture canvas would
not have reduced inference work: the old processor enlarged any 4:3 input to the
same 432 × 336 grid. Normalization was already RGB/255 followed by the vision
encoder's conversion to [-1, 1]. That normalization remains unchanged.

`decide(state, questions, { parts, imageTokens: 40 })` selects a budget without
reloading. Each `decideMany` item can choose its own budget. Values must be integers
from 1 through the loaded capacity; omission retains the loaded budget. Load-time
capacity presets remain 70, 140, 280, 560, and 1120. Budgets below 70 are experimental
custom settings below [Google's documented presets](https://ai.google.dev/gemma/docs/capabilities/vision).

The live selector invalidates an old setting's pending result and resets its FPS
window. It keeps the model and camera active. There is one decision in flight and
no queue of stale frames. FPS counts completed decisions, with fresh image encoding
and decoding for every frame.

## Fused F32 attention

The vision kernel processes 16 queries and 16 keys per tile, combines scores,
stable online softmax and weighted values, and avoids a quadratic score buffer.
It uses 14,080 bytes of workgroup storage, within WebGPU's minimum 16 KiB limit,
without optional F16 or subgroup features. Head dimension 64 and at most 630
processed patches use fusion; other geometries retain separate score and value
kernels. `Kevala.load({ ..., visionAttention: "split" })` selects the control.
`timing.image.attention` reports `fused-f32`, `split-f32`, or `mixed` for a batch.

Vision weights remain BF16 and arithmetic F32. The decoder uses the existing Q8
pack; this change introduces no additional quantization. GPU-resident projections,
finite-value checks, independent image buffers and shared question prefixes remain
active.

The Chrome 153 vision-only check used the actual pinned E2B companion, minimum
WebGPU features, five warmed unprofiled samples and separate GPU timestamp samples:

| Measurement | Split | Fused |
|---|---:|---:|
| Balanced encoder wall, median | 15.42 ms | 13.11 ms |
| Detail encoder wall, median | 29.42 ms | 20.55 ms |
| Balanced attention GPU time, all layers | 3.224 ms | 1.056 ms |
| Detail attention GPU time, all layers | 9.920 ms | 2.979 ms |

Standalone attention improved about 3.45× in Chrome and 2× in Firefox. These kernel
ratios do not describe complete camera inference. See the
[Chrome attention guard](benchmarks/gemma4-camera-resolution-2026-10-16/vision-fused-attention-chrome.json),
[Firefox guard](benchmarks/gemma4-camera-resolution-2026-10-16/vision-fused-attention-firefox.json),
[split tower](benchmarks/gemma4-camera-resolution-2026-10-16/vision-fused-tower-control-chrome.json),
and [fused tower](benchmarks/gemma4-camera-resolution-2026-10-16/vision-fused-tower-candidate-chrome.json).

## Completed camera decisions

The application check used Chrome 153's changing synthetic camera, the real SDK,
the application's scheduler and two unchanged person/smile questions. It loaded one
model and changed detail live. Each stage completed 24 requests; latency statistics
discard four warmups, while displayed FPS is the actual recorder's rolling value.

| Camera stage | Mean decision | p50 | p95 | Displayed FPS |
|---|---:|---:|---:|---:|
| Previous Detail, split attention | 71.48 ms | 71.26 ms | 73.63 ms | 13.6 |
| Balanced, split attention | 55.00 ms | 55.04 ms | 55.70 ms | 17.9 |
| Detail, fused attention | 60.86 ms | 60.50 ms | 63.04 ms | 15.9 |
| Balanced, fused attention | 50.05 ms | 49.54 ms | 53.06 ms | 19.8 |
| Fast, fused attention | 48.27 ms | 47.86 ms | 52.65 ms | 20.5 |
| Balanced, fused, repeat stage | 52.95 ms | 52.95 ms | 54.27 ms | 18.7 |

The final browser had a 3,200 MiB memory ceiling, zero swap, a four-core quota,
2.15 GiB cgroup memory peak and no CPU throttling. Other workstation workloads
remained active. Short stages, GPU sharing and run order limit the comparison.
Both runs verified one inference in flight and the same model across settings.
The final run also explicitly checked stopped camera tracks and cleared video.
See the [split camera receipt](benchmarks/gemma4-camera-resolution-2026-10-16/camera-app-resolution.json)
and [final camera receipt](benchmarks/gemma4-camera-resolution-2026-10-16/camera-app-fused-final.json).

## Numerical and image checks

The actual BF16 tower stays within the unchanged `1e-4` pinned F32 reference guard;
the largest observed error was approximately `2.066e-5`. Checks cover camera grids,
changed pixels after resizing, the original reference, padded layouts, GPU handoff,
NaN/infinity flags, head-32 and larger-grid fallbacks. Direct attention checks cover
tails, grouped heads, masked/all-masked keys and large logits. Fused versus split
attention differs by at most `5.96e-7` in those checks.

Four public images, each repeated three times, compared person and smile scores
with the same weights, prompts and capture crop at all resolutions. Balanced and
Detail had no decisions crossing 0.5; their largest raw probability difference was
0.07585. Fast missed a visible smile in the outdoor portrait. Detail and Balanced
both missed the person in the historical daguerreotype. This is a small smoke set,
not a representative facial-expression accuracy evaluation, and the scores are
uncalibrated. Fast therefore remains optional with a detail-loss hint.

At the same resolutions, fused versus split full-model raw probabilities differed
by at most `1.67e-6`. A mixed 16/70/40-budget batch matched independent decisions
exactly in the final run. The [split image receipt](benchmarks/gemma4-camera-resolution-2026-10-16/camera-budget-real-final.json),
[fused image receipt](benchmarks/gemma4-camera-resolution-2026-10-16/camera-fused-real-final.json)
and [image sources, hashes and rights](benchmarks/gemma4-camera-resolution-2026-10-16/sources.json)
retain the evidence. Image pixels are not included in those receipts.

## Reproduce

On a device with the published packs, open the
[opt-in browser benchmark](https://bvolpato.github.io/kevala/dev/gemma4-camera-check.html?hosted=1).
It loads only after **Run resolution benchmark**, reuses cached weights when
available, and exports a JSON result with browser, adapter, timing and score data.
Its generated frames test throughput and runtime behavior, not facial accuracy.
Add `attention=split` for the separate-kernel control. Stop other model tabs first.

For a local real-camera-path check with synthetic input, serve the checkout with
its decoder and companion in `tmp/`, then run on the Mac without Linux Vulkan flags:

```sh
node scripts/serve.mjs . --port 18086
uv run scripts/bench-reaction.py \
  --binary '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --output tmp/m4-reaction.json
```

Generate the numerical camera reference from the extracted pinned vision weights:

```sh
uv run tools/golden_gemma4_vision.py --local tmp/gemma4-vision/E2B --camera \
  --output tmp/golden-gemma4-camera-e2b.json
uv run scripts/bench-gpu.py --result gpu --backend webgpu \
  --url 'http://127.0.0.1:18086/dev/gemma4-vision-check.html?fixture=../tmp/golden-gemma4-camera-e2b.json&pack=../tmp/gemma-4-e2b-vision-bf16.kevala&profile=1' \
  --output tmp/camera-vision.json
```

The full-model harness is `dev/gemma4-camera-check.html`; `images=URL` accepts a JSON
array of `{id,url,expectedPerson,expectedSmile}` entries for labeled local images.
The retained smoke images use the source URLs in `sources.json`, downloaded locally
and referenced by a local manifest. The harness center-crops each to 320 × 240.

On M4 Max these changes run through browser WebGPU over Metal. Kevala has no native
MLX or CUDA inference backend. Physical M4 Max vision/camera timing remains unverified;
the existing September M4 reports measured text workloads. All large GPU checks here
ran serially under memory limits.
