---
license: apache-2.0
library_name: kevala
tags:
  - kevala
  - webgpu
  - webassembly
  - int8
base_model:
  - convaiinnovations/laya
  - jaredpalmer/kev-0.8b
  - jaredpalmer/kev-4b
  - jaredpalmer/kev-9b
  - Qwen/Qwen3.5-0.8B
  - Qwen/Qwen3.5-2B
  - Qwen/Qwen3.5-4B
  - google/gemma-4-E2B-it
  - google/gemma-4-E4B-it
---

# kevala packs

Pre-converted int8 packs for [kevala](https://github.com/bvolpato/kevala), which runs decision models
in the browser on WebGPU or WebAssembly. Loading a pack from here skips the in-browser conversion and
downloads about half the bytes of the original checkpoints.

| file | model | from | size |
|---|---|---|---:|
| `laya-q8.kevala` | Laya | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) at `1c5edc17a7acd8701df6fc341c0d179f1c62c982` | 479 MB |
| `kev-0.8b-q8.kevala` | Kev-0.8B | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) at `54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8`, with its base [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 857 MB |
| `kev-4b-q8.kevala` | Kev-4B | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) at `485ace8703592fcf405488b262449990824cfed1` | 4.76 GB |
| `kev-9b-q8.kevala` | Kev-9B | [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) at `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c` | 8.96 GB |
| `semif-qwen3.5-0.8b-q8.kevala` | SemIf Qwen3.5-0.8B | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) at `2fc06364715b967f1860aea9cf38778875588b17` | 855 MB |
| `semif-qwen3.5-2b-q8.kevala` | SemIf Qwen3.5-2B | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) at `15852e8c16360a2fea060d615a32b45270f8a8fc` | 2.13 GB |
| `semif-qwen3.5-4b-q8.kevala` | SemIf Qwen3.5-4B | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 4.75 GB |
| `gemma-4-e2b-q8.kevala` | Gemma 4 E2B | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) at `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | 5.217 GB |
| `gemma-4-e4b-q8.kevala` | Gemma 4 E4B | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) at `ee0ef6023621cff504d758262d4e04895a5af4a2` | 8.407 GB |

The Laya, Kev, and SemIf packs were made with the kevala command line: weights are
int8 with one f32 scale per 32 values (symmetric absmax); norms, biases, gates and decision heads stay
f32; Kev's LoRA adapter is merged in f32 before quantization. The preferred command is
`kevala convert <checkpoint-dir> -o out.kevala`; `convert-kev`, `convert-semif`, and `convert-gemma`
remain compatibility aliases for the same validated path. The header of every pack records its source,
revision, author, license and conversion. Against the upstream PyTorch code on the fixtures,
Laya keeps 41/41 argmax agreement (largest probability difference 0.024) and Kev-0.8B 13/13 (0.010).

Sizes above use decimal units. Each new pack has an adjacent JSON manifest with its exact byte
count, SHA-256, base checkpoint revision, and configuration. Kev-4B uses Qwen3.5-4B-Base at
`1001bb4d826a52d1f399e183466143f4da7b741b`; Kev-9B uses Qwen3.5-9B-Base at
`68c46c4b3498877f3ef123c856ecfde50c39f404`. These are text-only packs.

The Gemma 4 E2B and E4B packs are also text-only. They use the common `kevala convert` path (with
`convert-gemma` retained as a compatibility alias) to
extract the dense language model from Google's multimodal checkpoints and do not include the
vision or audio towers. The sizes above are the converted text pack sizes.

## SemIf method

[SemIf](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37)
provides the `direct-options-v1` prompt and scoring method used by the SemIf Qwen packs and adapted
for the Gemma packs. These packs use frozen instruction weights, with no SemIf adapter or
fine-tuning. One forward pass scores the next-token labels A-P; Kevala keeps only those output-head
rows in f32 and applies a softmax over the declared options. It does not generate an answer or chain
of thought. SemIf Qwen reuses evidence prefixes between questions; the initial Gemma backend runs each question independently.

The Qwen 4B model follows SemIf's reference choice. The 0.8B and 2B variants apply the same method
to smaller models. All support up to 16 options per question. Their scores are conditional on the
listed options and **uncalibrated as decision confidence**. These are Kevala conversions, not
official SemIf checkpoints or a drop-in replacement for the SemIf Python API.

Gemma uses Google's instruction chat template with thinking disabled, then applies the same direct
option readout. It scores the final hidden state against the declared label rows and applies Gemma's
final logit soft cap before the option softmax.

## Validation and requirements

The existing packs were checked on an RTX 5070 Ti (16 GB), Firefox 152.0.3, Ubuntu, using hardware
WebGPU. These small fixture checks measure conversion fidelity, not task accuracy or a model ranking:

| Pack | Reference precision | Matching decisions | Largest absolute probability difference |
|---|---|---:|---:|
| Kev-4B | BF16, LoRA merged in F32 | 13/13 | 0.00986 |
| Kev-9B | F32 on CPU | 5/5 | 0.00481 |
| SemIf 0.8B | F32 on CUDA | 12/12 | 0.03108 |
| SemIf 2B | F32 on CUDA | 12/12 | 0.02901 |
| SemIf 4B | BF16 on CUDA | 12/12 | 0.02006 |
| Gemma 4 E2B IT | BF16 on CUDA | 12/12 | 0.01907 |
| Gemma 4 E4B IT | BF16 on CUDA | 12/12 | 0.01209 |

The Gemma packs target WebGPU in supported browsers or the native 64-bit CPU CLI. Browser
CPU and Node WebAssembly loading require packs under 2 GiB; use the native 64-bit CLI for these
larger packs. WebGPU needs sufficient GPU memory and per-buffer limits. Large packs remain opt-in
and are not downloaded by routine CI. Caching avoids later downloads, but loading still transfers
weights to the GPU and needs working memory.

The Gemma backend accepts at most 4096 input tokens per question. Overlong requests fail instead of silently
dropping evidence. This runtime limit is smaller than the upstream model context limits.

```js
import { Kevala } from "kevala";
const kevala = await Kevala.load({ model: "laya" }); // fetches laya-q8.kevala from here
```

For Laya and Kev-0.8B, `from: "checkpoint"` converts the original weights in the browser. Other
models require these packs. The native converter reads the checkpoint's Qwen2Tokenizer configuration
and applies its normalization and added-token overlay, so ordinary conversion needs no Python or
Transformers tokenizer materialization. Pass `--tokenizer` only for an explicit, verified override.
The pinned conversion helper is `tools/convert_models.py`; it invokes the common native path. The
commands that made and checked these packs, and the steps to publish new ones, are in
[docs/packs.md](https://github.com/bvolpato/kevala/blob/main/docs/packs.md).

## Credits and licenses

- Laya is by Nandakishor M, Convai Innovations, under Apache-2.0.
- Kev models are by Jared Palmer, under Apache-2.0. Their Qwen3.5 base models are by the Qwen team,
  under Apache-2.0. Kev's trained LoRA adapters and pointer heads are preserved in these conversions.
- Qwen3.5 instruction weights are by the Qwen team, under Apache-2.0.
- Gemma 4 E2B and E4B instruction weights are by Google DeepMind and are published under
  [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license); their pinned source revisions are
  listed above. The packs retain the source model metadata.
- SemIf's prompt and direct option-logit readout are MIT-licensed, copyright 2026 TheoLeeCJ.
  The complete notice is in `THIRD_PARTY_NOTICES`.

The Kevala tooling and pack metadata are distributed under Apache-2.0. These packs are derivative
works of the listed source models and carry each source model's license and behavior limits: read the
original model cards before relying on them.
