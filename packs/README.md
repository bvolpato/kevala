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
---

# kevala packs

Pre-converted int8 packs for [kevala](https://github.com/bvolpato/kevala), which runs decision models
in the browser on WebGPU or WebAssembly. Loading a pack from here skips the in-browser conversion and
downloads about half the bytes of the original checkpoints.

| file | model | from | size |
|---|---|---|---:|
| `laya-q8.kevala` | Laya | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) at `1c5edc17a7acd8701df6fc341c0d179f1c62c982` | 479 MB |
| `kev-0.8b-q8.kevala` | Kev-0.8B | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) at `54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8`, with its base [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 857 MB |

Each pack was made with the kevala command line (`kevala convert`, `kevala convert-kev`): weights are
int8 with one f32 scale per 32 values (symmetric absmax); norms, biases, gates and decision heads stay
f32; Kev's LoRA adapter is merged in f32 before quantization. The header of every pack records its
source, revision, author, license and conversion. Against the upstream PyTorch code on the fixtures,
Laya keeps 41/41 argmax agreement (largest probability difference 0.024) and Kev 13/13 (0.010).

```js
import { Kevala } from "kevala";
const kevala = await Kevala.load({ model: "laya" }); // fetches laya-q8.kevala from here
```

To convert the original weights instead, pass `from: "checkpoint"`. The commands that made and
checked these packs, and the steps to publish new ones, are in
[docs/packs.md](https://github.com/bvolpato/kevala/blob/main/docs/packs.md).

## Credits and licenses

- Laya is by Nandakishor M, Convai Innovations, under Apache-2.0.
- Kev-0.8B is by Jared Palmer, under Apache-2.0. Its base, Qwen3.5-0.8B-Base, is by the Qwen team,
  under Apache-2.0.

These packs are derivative works of those models, redistributed under the same license. They carry
the models' own behavior and limits: read the original model cards before relying on them.
