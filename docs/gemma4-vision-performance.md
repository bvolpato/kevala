# Gemma 4 vision performance

See [camera resolution and fused attention](gemma4-camera-performance.md) for the
October 16 defaults, current camera measurements, quality limits and M4 reproduction.
The dated measurements below retain their original settings.

The retained vision changes reduced warmed encoder latency from 901 ms to 99 ms in the
2026-10-13 Firefox run. This is a vision-only measurement, excluding the text decoder,
camera capture, image decoding, and Pillow-compatible resizing.

The host has an NVIDIA RTX 5070 Ti with 16 GB VRAM. Firefox 152.0.3 used an isolated,
headless WebGPU session with timestamp queries and no optional F16 or subgroup kernels.
Its adapter reported hardware acceleration but withheld vendor and device strings.
Other workstation workloads remained active. Each browser run had a 2 GB memory limit,
no swap, and a two-core CPU quota. Firefox's timer privacy settings were unchanged.

## Results

The actual E2B BF16 companion processed a pinned reference image with 630 padded input
patches, 576 real patches, and 64 projected image tokens. Each variant had five warmup
runs, five unprofiled samples, one profiled warmup, and five profiled samples. Profiling
puts each dispatch in its own timestamped pass and adds overhead.

| Median metric | Padded, serial V, awaited chunks | Compact, parallel V, split chunks |
|---|---:|---:|
| Unprofiled wall time | 901 ms | 99 ms |
| Profiled wall time | 1001 ms | 200 ms |
| Sum of GPU timestamps per sample | 63.402 ms | 32.736 ms |
| Attention GPU time | 32.258 ms | 3.480 ms |
| Score-dot GPU time | 12.894 ms | 11.896 ms |
| Processed patch rows | 630 | 576 |

The combined wall improvement is 9.1 times; the GPU timestamp sum improves 1.94 times.
The wall improvement includes removing repeated queue-completion waits. It is not an
isolated attention-kernel speedup or a measurement of webcam classification FPS.
See the [control](benchmarks/gemma4-vision-2026-10-13/control.json) and
[optimized samples](benchmarks/gemma4-vision-2026-10-13/optimized.json).

## Retained changes

- The weighted V reduction shares key work across all available lanes. The old kernel
  used only 16 of 256 lanes for a 64-dimensional head, each looping over every key.
- The runtime removes masked patch rows before projection, attention, and pooling.
  It preserves real patch order, pixel values, positions, and spatial pooling cells.
  A 320 by 240 camera frame uses 567 real patches instead of 630 padded rows.
- Vision submits separate command buffers after every two layers and waits for the
  final readback. It avoids a queue-completion round trip after every pair of layers.
  An explicit `submit: "await"` override retains the previous synchronization policy.
- Image sessions stream the decoder's original Q8 per-layer embedding tables to a
  temporary OPFS file, with a system RAM fallback when OPFS is unavailable, and upload
  only requested rows. This moves 2,642,411,520 weight bytes
  off the GPU; combined decoder and vision GPU weights occupy 2,902,743,948 bytes.
  OPFS sessions also retain the Q8 token embedding table in that file, removing a further
  452,984,832 GPU weight bytes and bringing the combined weights to 2,449,759,116 bytes.

Dedicated workers use OPFS synchronous access handles and reusable read buffers. The
bounded [row-store benchmark](benchmarks/gemma4-vision-2026-10-13/ple-store.json) compared
seven-table Q8 gather batches in Chrome 153: repeated median time was 67.5 ms for the
Window writable-stream/Blob path and 1.3 ms for the worker access-handle path, with
bit-identical output. Each batch selects 256 rows, including duplicates and PAD, from
9 MiB of stored tables. These are lookup and dequantization timings, excluding model
inference, GPU uploads, and the GPU row cache. The first gather follows file creation;
the OS cache was not evicted. Other workloads remained active and API call counters
were enabled during measurement. Reproduce at `dev/gemma4-ple-store-check.html?autorun=1`.

The vision weights remain BF16 and arithmetic remains F32. No new quantization was used.
Repeated decoder prompts reuse at most 16 MiB of GPU PLE row uploads; image projections and
decoder output are recomputed for every frame. The real OPFS-backed GPU guard is bit-exact
with host and GPU lookup, including changed images with a warmed prompt cache. The recorder
schedules the next frame immediately when inference exceeds its 33 ms minimum start interval,
rather than adding the former 200 ms delay after each inference.
These October 13 measurements precede the tiled scores, shared decoder prefix, and GPU
projection handoff described below.

## Numerical validation

The optimized actual companion matches the pinned Transformers F32 reference with
maximum absolute error 1.9908e-5 and relative L2 error 1.9567e-6, within the 1e-4 guard.
The reference source commit is `2c4914fb939fe9de0d8e7a798af4684d552f18b4`; the model
revision is `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The companion SHA-256 is
`6556225c040a5b1d0bc8874a2833ce17cb7579d390f3225995c09ae589214d04`.

The [tiny tower checks](benchmarks/gemma4-vision-2026-10-13/tiny.json) cover clipping,
standardization, and reordered padding. The [decoder checks](benchmarks/gemma4-vision-2026-10-13/decoder.json)
verify finite, nonzero final rows, image-sensitive output, duplicate row selection,
chunked sequences, and bit-exact host versus GPU per-layer embedding lookup.
The [OPFS decoder guard](benchmarks/gemma4-vision-2026-10-13/decoder-opfs.json) also
checks stored word embeddings, changed images with a warmed prompt cache, and literal
PAD token identities without reusing image-placeholder word rows.
These component checks do not measure person or expression classification accuracy.

## Reproduction

Build the WASM modules and [convert the pinned companion](gemma4.md#vision-companion).
Generate the actual reference fixture with `tools/golden_gemma4_vision.py`, then run
`scripts/bench-gpu.py` against the local `dev/gemma4-vision-check.html` page:

```sh
uv run scripts/bench-gpu.py --result gpu --backend webgpu \
  --url 'http://127.0.0.1:18086/dev/gemma4-vision-check.html?fixture=../tmp/golden-gemma4-vision-e2b.json&pack=../tmp/gemma-4-e2b-vision-bf16.kevala&profile=1' \
  --output tmp/vision-optimized.json
```

For the control, add `padded=1`, `submit=await`, and
`attention=../docs/benchmarks/gemma4-vision-2026-10-13/attention-before.wgsl` to the URL.
The retained control receipt used the pre-optimization WASM attention kernel; the
provided shader is that kernel's exact rendered source. The default tiny fixture does
not download model weights. Full-model checks are explicit opt-in through
`dev/gemma4-image-check.html`; do not run several large model checks concurrently.

`kevala.profile(true)` includes `vision.*` and decoder labels in `timing.gpu`.
Image responses also report `timing.image.decode`, `preprocess`, and `forward`, plus
tokenization and readout wall times. `timing.ple` reports embedding lookup/upload wall
time and cache hits; `timing.decoder` reports final-row RMS values for image requests.
Turn profiling off when measuring normal latency.

## Camera path improvements, October 15

The recorder previously encoded the image once, then ran the decoder over both complete
question prompts. It now computes their identical image and text prefix once, retains each
non-shared layer's K/V, and evaluates independent causal suffixes. The 20 shared-K/V layers
still run on each question's selected final row. Questions and readouts are unchanged.
Every frame recomputes vision and decoder states from fresh pixels.
Short requests submit the prefix, suffixes, and readback as one command buffer, using
packed branch parameters to preserve each sequence's positions and attention boundaries.
Sharing falls back for more than eight questions, 256 tokens per sequence, or 512 unique
decoded tokens, keeping longer requests on the existing chunked path.

The 64-dimensional vision score kernel stages 16 queries and 16 keys in 8,704 bytes of
workgroup storage. It preserves F32 dot-product order, masks, and grouped KV heads. Other
head dimensions retain the original kernel. Resizing coefficients and spatial geometry are
cached with bounded retention, exact geometry validation, and independent output arrays.

Unprofiled inference keeps projected F32 rows on the GPU until the decoder consumes them.
Each image owns its output buffer, so later images and shape changes cannot overwrite it.
A finite-value kernel writes a status flag checked with the decoder's final readback,
preserving NaN and infinity rejection without another synchronization. GPU projection
residency is capped at 64 MiB per batch, with the existing CPU path for overflow. Prefix K/V
is capped at 64 MiB. PLE uploads retain the 16 MiB, eight-entry cache, with in-flight entries
pinned until readback. Vision weights remain BF16 and arithmetic F32.

`timing.prefixReuse` reports sharing, common prefix length, original token count, and
repeated tokens avoided. The default questions avoid 117 of 331 decoder tokens.
`decide(..., { shareImagePrefix: false })` provides a control. Per-kernel profiling uses the
original separate decoder passes and CPU projection readback. Use unprofiled full-frame
timing for throughput. `timing.image.forwardMode` distinguishes completed `encoder-wall`
timing from GPU `submission` timing; `mixed` denotes both within a batch. In submission
mode, `timing.forward` includes queued vision work. `timing.total` and the recorder's
capture-to-result latency still measure completed requests.

GPU guards cover score tails, masked rows, grouped heads, independent image buffers across
shape changes, mixed CPU/GPU images, and nonfinite projector output through both decoder
paths. The actual BF16 tower remains within its pinned `1e-4` reference guard. These checks
establish numerical behavior, not person or facial-expression accuracy.

The following measurements used the RTX 5070 Ti on Linux with other workstation workloads
active. The camera check used Chrome 153, its synthetic camera, the actual application
scheduler and model, a four-core quota, a 3,200 MiB memory ceiling, and no swap. Neither
paired camera run was CPU-throttled. Both processed 33 frames with one inference in flight;
the camera stopped cleanly. Wall statistics discard the first five requests.

| Measurement | Main at `556bdae` | Optimized |
|---|---:|---:|
| Vision score GPU time, Firefox, 576 patches | 12.625 ms | 5.560 ms |
| Warm vision encoder wall, Chrome, 576 patches | 33.3 ms | 26.7 ms |
| Camera `decide` mean wall, Chrome, 567 patches | 118.6 ms | 92.3 ms |
| Camera `decide` median wall | 117.7 ms | 76.8 ms |
| Last displayed processed FPS | 8.3 | 10.5 |

See the [paired camera summary](benchmarks/gemma4-throughput-2026-10-15/summary.json),
[control](benchmarks/gemma4-throughput-2026-10-15/camera-control.json),
[optimized camera](benchmarks/gemma4-throughput-2026-10-15/camera-final.json), and
[full-model probability check](benchmarks/gemma4-throughput-2026-10-15/full-model.json).
The last check correctly distinguishes generated red/blue frames and compares shared
versus independent question probabilities within `2.8e-15` on those inputs. This is one
paired camera run on a shared GPU. Earlier short runs hovered around 10 FPS, and individual
frames remain variable; these results do not establish a guaranteed FPS or MacBook speedup.

Reproduce with the local decoder and companion advertised by `scripts/serve.mjs`:

```sh
uv run scripts/bench-gpu.py --browser chromium --result gpu \
  --chromium-arg=--enable-unsafe-webgpu \
  --chromium-arg=--use-angle=vulkan --chromium-arg=--enable-features=Vulkan \
  --url 'http://127.0.0.1:18086/dev/gemma4-image-check.html?autorun=1&profile=1&parity=1' \
  --output tmp/image-throughput.json
```

The Chromium flags apply to the isolated Linux test process. They are not settings required
for a user's browser. Component guards are at `dev/gemma4-vision-scores-check.html`,
`dev/gemma4-vision-check.html?gpu=1`, and `dev/gemma-tail-check.html`. MacBook performance
requires measurement on that device.
