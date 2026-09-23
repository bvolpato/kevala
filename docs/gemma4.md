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
The separate converter omits the vision and audio towers and declares only the text modality.

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

Packs use symmetric INT8 weights with an FP32 scale per 32 values. Norms and answer-label
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
