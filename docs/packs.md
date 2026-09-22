# Packs: loading, converting and publishing

A pack (`.kevala`) is a model's weights, tokenizer and config in one file: int8 weights with one f32
scale per 32 values, and a header that records the source, revision, author, license and conversion.
The engine only ever runs packs. This page covers where they come from and how to make new ones.

## Three ways to load a model

```js
import { Kevala } from "kevala";

// 1. By name: the pinned int8 pack from Hugging Face (479 MB for Laya, 857 MB for Kev).
//    When the pack is unreachable, it falls back to converting the original weights.
const kevala = await Kevala.load({ model: "laya" });

// 2. From the original weights: downloads the authors' checkpoint at its pinned revision
//    (843 MB for Laya, 1.6 GB for Kev) and converts it in the browser.
const fromCheckpoint = await Kevala.load({ model: "laya", from: "checkpoint" });

// 3. A .kevala file you host or already have: a URL, an ArrayBuffer, or a Blob.
const own = await Kevala.load({ model: "https://example.com/laya-q8.kevala" });
```

Both ways of loading by name store the pack under one key per revision and quantization, so later
visits read it from disk whichever way it arrived. To convert again when a pack is already stored,
clear it first (`clearCache()`, or "Clear stored packs" on the site) or pass `cache: false`. The
site takes `?from=checkpoint` to do the same, for example
`https://bvolpato.github.io/kevala/?from=checkpoint`.

A pack URL is fetched in parallel byte ranges when the server supports them (Hugging Face, GitHub
Pages, S3, any static host with `Accept-Ranges`), and as one stream otherwise. It has to send
`Content-Length`, and CORS headers when it is on another origin.

## Downloading a pack directly

The packs live in [bvolpato/kevala-packs](https://huggingface.co/bvolpato/kevala-packs). The runtime
pins them to one commit (`PACKS` in [`js/src/source.js`](../js/src/source.js)); use the same commit
to get the exact bytes it loads:

```sh
rev=<commit from js/src/source.js>
hf download bvolpato/kevala-packs laya-q8.kevala kev-0.8b-q8.kevala --revision "$rev" --local-dir packs
# or, without the Hugging Face CLI
curl -L -o laya-q8.kevala "https://huggingface.co/bvolpato/kevala-packs/resolve/$rev/laya-q8.kevala"
```

Then serve the file yourself (pass its URL as `model`), or load it server side:

```js
import { loadFile } from "kevala/node";
const kevala = await loadFile("packs/laya-q8.kevala");
```

## Converting the original weights

Build the command line, then download each checkpoint at the revision pinned in `MODELS`
([`js/src/source.js`](../js/src/source.js)). Only the files the converter reads are needed: the Laya
repo also holds multilingual and typed-decision variants that would triple the download.

```sh
cargo build --release -p kevala-cli
kevala=target/release/kevala

# Laya (843 MB)
hf download convaiinnovations/laya model.safetensors encoder/config.json rl_agent_config.json \
  tokenizer/tokenizer.json --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir ckpt/laya
$kevala convert ckpt/laya -o packs/laya-q8.kevala --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982

# Kev-0.8B: its LoRA adapter and pointer head, and the Qwen3.5 base they apply to
hf download jaredpalmer/kev-0.8b adapter_config.json adapter_model.safetensors head.pt tokenizer.json \
  --revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --local-dir ckpt/kev-0.8b
hf download Qwen/Qwen3.5-0.8B-Base config.json model.safetensors-00001-of-00001.safetensors \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --local-dir ckpt/qwen3.5-0.8b-base
$kevala convert-kev --base ckpt/qwen3.5-0.8b-base --kev ckpt/kev-0.8b -o packs/kev-0.8b-q8.kevala \
  --kev-revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
```

The revision flags only label the pack header; the weights are whatever the directories hold, so
download them at the same revisions. The browser conversion (`from: "checkpoint"`) runs the same Rust
converter, compiled to WebAssembly.

## Checking a pack

Every pack should match the upstream PyTorch models on the fixtures before anyone loads it:

```sh
$kevala inspect packs/laya-q8.kevala                                # header, sizes, provenance
$kevala parity packs/laya-q8.kevala tests/fixtures/golden.json      # want argmax 41/41, max|dp| < 0.03
$kevala parity-kev packs/kev-0.8b-q8.kevala tests/fixtures/golden-kev.json   # want argmax 13/13
```

The fixtures come from the authors' own PyTorch code ([`tools/golden.py`](../tools/golden.py) and
[`tools/golden_kev.py`](../tools/golden_kev.py)); regenerate them when a model's revision changes.

## Publishing packs

[`packs/README.md`](../packs/README.md) is the model card of the Hugging Face repo. Uploading needs a
Hugging Face token with write access to the repo (`hf auth login`; an `HF_TOKEN` in the environment
takes precedence over the saved login).

```sh
hf repo create bvolpato/kevala-packs --exist-ok
hf upload bvolpato/kevala-packs packs . --include "*.kevala" --include README.md \
  --commit-message "Laya and Kev-0.8B int8 packs"
curl -s https://huggingface.co/api/models/bvolpato/kevala-packs | jq -r .sha   # the commit to pin
```

Then point the runtime at that commit:

1. Set `PACKS` in `js/src/source.js` to `https://huggingface.co/bvolpato/kevala-packs/resolve/<sha>`,
   and each model's `pack` to the byte size of its file.
2. Load each model on the site with its stored pack cleared, and check that the progress shows the
   pack downloading, not a conversion.
3. Release a new version, so the CDN and npm serve the new pin.

A pack is pinned by commit, never by `main`, so a later upload never changes what an existing
release loads. Browsers store a pack under its model's revision and block size (`upstreamKey` in
`js/src/source.js`). A pack republished for the same revision keeps that key, and browsers that
stored the previous one keep using it: change the key whenever the bytes change for another reason.
