# Gemma 4 vision performance

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
- Image sessions retain the decoder's original Q8 per-layer embedding tables in
  system RAM and upload only requested rows. This moves 2,642,411,520 weight bytes
  off the GPU; combined decoder and vision GPU weights occupy 2,902,743,948 bytes.

The vision weights remain BF16 and arithmetic remains F32. No new quantization was used.
Score-dot attention is the largest remaining measured vision kernel. CPU-to-GPU image
upload and projected-row readback remain separate from the text decoder. Further fusion
or shared decoder prefixes need fresh measurements and numerical validation.

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
tokenization and readout wall times. Turn profiling off when measuring normal latency.
