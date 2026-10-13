# Gemma 4 decisions

Kevala uses Google's instruction-tuned Gemma 4 E2B and E4B weights for text decisions.
The model names are `gemma-4-e2b` and `gemma-4-e4b`. The source revisions are recorded in
[`tools/model-sources.json`](../tools/model-sources.json) and each converted pack.

## Architecture

E2B and E4B are **dense models with per-layer embeddings**, not mixtures of experts.
The Gemma 4 MoE model is 26B A4B, which is a separate architecture and is not covered by this
integration. Google's [model card](https://huggingface.co/google/gemma-4-E2B-it) distinguishes
effective parameter count from the total including embedding tables.

| | E2B | E4B |
|---|---:|---:|
| Effective parameters | 2.3B | 4.5B |
| Total parameters, including embeddings | 5.1B | 8B |
| Decoder layers | 35 | 42 |
| Hidden width | 1536 | 2560 |
| Query heads / KV heads | 8 / 1 | 8 / 2 |
| Layers sharing earlier keys and values | 20 | 18 |
| Local attention window | 512 | 512 |

Both sizes alternate sliding and full causal attention. Local heads have 256 dimensions;
global heads have 512. They use Q/K/V normalization, different rotary embeddings for local
and global attention, GELU-gated feed-forward layers, and a per-layer embedding residual.
E2B doubles the feed-forward width in the layers that share keys and values. These details
require a Gemma backend; selecting a different checkpoint in the Qwen backend is insufficient.

The implementation follows the pinned [Transformers Gemma 4 reference](https://github.com/huggingface/transformers/blob/2c4914fb939fe9de0d8e7a798af4684d552f18b4/src/transformers/models/gemma4/modeling_gemma4.py).
The decoder converter omits the vision and audio towers and declares only the text modality.
E2B can load a separate vision companion for browser image decisions on WebGPU.

## Vision companion

The E2B image path combines the existing 5,217,421,952-byte Q8 decoder with a
337,302,144-byte BF16 vision companion. The companion preserves the source matrix and positional
table values; normalization weights and clipping bounds are stored in FP32. Its GPU activation
arithmetic is FP32. It must match the decoder's source repository, pinned revision, and hidden
width. The public decoder pack is published; the vision companion currently needs local conversion
and serving.

Image sessions keep the decoder's original Q8 per-layer embedding tables in system RAM and
upload only the rows needed for each frame. This moves 2,642,411,520 bytes off the GPU without
changing the weights or embedding math. The decoder and vision tower use about 2.9 GB of GPU
weights; scratch buffers, transfer staging, and browser overhead need additional memory.
The tables use about 2.6 GB of system RAM. `info.memory` reports weight placement; it does not
measure available memory or include scratch buffers.

Fetch only the pinned vision tower and projector, then convert them:

```sh
uv run tools/fetch_gemma_vision.py --output-dir tmp/gemma4-vision/E2B
cargo build --release -p kevala-cli
target/release/kevala convert-vision tmp/gemma4-vision/E2B \
  -o tmp/gemma-4-e2b-vision-bf16.kevala --name gemma-4-e2b-vision \
  --source google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --precision bf16 --author "Google DeepMind" --license apache-2.0
pnpm build
pnpm serve
```

The range extractor downloads 337,089,408 original tensor bytes without fetching the decoder.
It validates HTTP ranges and source identity, and writes a manifest with the pin, offsets, byte
counts, and hashes. Generated weights stay in `tmp/`. `convert-vision` currently accepts BF16 only.

```js
const kevala = await Kevala.load({
  model: "gemma-4-e2b",
  vision: "http://127.0.0.1:8080/tmp/gemma-4-e2b-vision-bf16.kevala",
  imageTokens: 70,
  backend: "webgpu",
});
```

Pass images through `decide(state, questions, { parts: [{ type: "image", data: image }] })`.
See the [image API](../README.md#images-with-gemma-4-e2b) for accepted data forms and limits.
Use the loaded `info.modalities` to gate image controls. Loading the decoder alone leaves it
text-only, and other model families need their own implemented image hook and encoder.

The runtime defaults to a 70-token image budget; 140, 280, 560, and 1120 are also accepted.
The larger budgets require larger GPU buffers; loading checks the requested attention buffer
against the device's limits before uploading the weights.

The vision encoder skips padded patches and submits separate command buffers after every two
layers without waiting for every chunk. An explicit `submit: "await"` override retains awaited
chunks. Optional GPU profiles include `vision.*` kernel labels alongside the text decoder;
image responses also separate decoding, preprocessing, vision, tokenization, and readout wall
times. See the [vision measurements and numerical guards](gemma4-vision-performance.md).
The pinned image processor's default is 280. Actual token counts depend on resized dimensions;
a square image at budget 70 yields 64 tokens. Preprocessing follows the pinned RGB bicubic resize,
16-pixel patches, and 3-by-3 pooling. E2B retains its causal decoder attention. Image placeholders
use PAD identity for per-layer embeddings and the projected image rows for contextual embeddings.
Image scores use the same uncalibrated option readout as text; the text benchmark below does not
measure image quality or latency. Image placeholders count toward the 4096-token input limit.

## Decision method

The prompt supplies evidence, a criterion, and options labeled `A` through `P`, using Gemma's
own instruction chat template with thinking disabled. One forward pass produces the final
hidden state. Kevala projects it onto the selected label rows, applies Gemma's final logit
soft cap, and takes a softmax over the supplied options. It does not generate an answer string.

This adapts [SemIf's direct option scoring method](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37)
to Gemma. The weights are Google's frozen instruction weights, with no SemIf adapter or
additional decision training. The SemIf MIT notice is retained in
[`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES). The output scores are conditional on the
listed options, not calibrated confidence that a decision is correct.

## Memory and precision

The embedding tables account for much of each model's download. Conversion splits the
per-layer table into one tensor per layer, keeping individual buffers within browser GPU
limits. WebGPU owns the embeddings and transformer weights. The WebAssembly coordinator
holds the tokenizer and the small set of answer-label weights.

Decoder packs use symmetric INT8 weights with an FP32 scale per 32 values. Norms and answer-label
rows remain FP32. Matrix kernels decode weights into floating-point tiles and accumulate in
FP32; INT8 storage does not imply INT8 activation arithmetic. Converted weights and generated
WebAssembly binaries are build or distribution artifacts, not source files tracked in Git.

The initial Kevala limit is 4096 input tokens per question. Overlong requests fail instead of
silently dropping evidence. This is smaller than the upstream model's context limit. Browser
CPU and Node WebAssembly cannot load these packs because each exceeds the allocation limit
of under 2 GiB; use WebGPU or the native 64-bit CLI with sufficient memory.

## Verification

The [Gemma attention performance report](gemma-attention-performance.md) records the
GPU kernel timings, numerical checks, and rejected optimization experiments for
the tiled attention implementation.

[`tools/golden_gemma.py`](../tools/golden_gemma.py) runs the upstream text model and records
the exact chat prompt, token IDs, selected logits, probabilities, and source revision.
Tokenization must match exactly. Native and browser comparisons must report probability
differences and decision agreement separately from timing. Fixture agreement alone does not
establish general model quality or state-of-the-art performance.

The broader [decision benchmark](../BENCHMARK.md) compares 108 option orders against the
original weights. E2B matches 107/108 BF16 decisions and E4B matches 108/108, but E2B has a
maximum probability difference of 0.53 on one case. Native Q8 and original-weight FP32
diagnostics are included there. The smaller parity fixture limits below do not bound all inputs.

On September 22, 2026, Firefox 152.0.3 on Ubuntu passed these checks against
Transformers 5.17.0 running the original BF16 checkpoints on an RTX 5070 Ti:

| Runtime | Model | Matching decisions | Maximum absolute probability difference |
|---|---|---:|---:|
| NVIDIA WebGPU, automatic matrix tuning | E2B | 12/12 | 0.01907 |
| NVIDIA WebGPU, automatic matrix tuning | E4B | 12/12 | 0.01209 |
| AMD integrated WebGPU, generic matrix kernel | E2B | 12/12 | 0.01826 |
| Native CPU | E2B | 12/12 | 0.01912 |
| Native CPU, three selected cases | E4B | 3/3 | 0.00995 |

All native input token sequences matched exactly. Both sizes also completed a five-piece,
one-seed Tetris smoke run using batches of four candidate states, with valid placements and
finite scores. The shared-runtime GPU regression checks remained Laya 41/41 and Kev 13/13.
The [validation record](benchmarks/gemma4-linux-2026-09-22.json) contains the probabilities,
pack hashes, source revisions, and game traces. Its Firefox wall clocks were quantized, so
these checks are not a latency comparison.

Reproduce the browser check after building with `pnpm build` and serving with
`pnpm serve --port=18092`:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json uv run scripts/bench-gpu.py \
  --result parity --max-dp 0.03 --timeout 600 \
  --url 'http://127.0.0.1:18092/parity-kev.html#backend=webgpu&pack=local&model=gemma-4-e2b&reference=./tests/fixtures/golden-gemma4-e2b.json&cache=0'

target/release/kevala parity-gemma tmp/gemma-4-e2b-q8.kevala \
  tests/fixtures/golden-gemma4-e2b.json
```

Change the model and reference together for E4B. Routine CI downloads only the pinned
tokenizer metadata and runs the small fixtures; full weights and GPU checks remain opt-in.
