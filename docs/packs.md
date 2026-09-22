# Packs: loading, converting and publishing

A pack (`.kevala`) is a model's weights, tokenizer and config in one file: int8 matrix weights with
one f32 scale per 32 values, selected tensors in f32, and a header that records the source, revision,
author, license and conversion.
The engine only ever runs packs. This page covers where they come from and how to make new ones.

## Three ways to load a model

```js
import { Kevala } from "kevala";

// 1. By name: the pinned int8 pack from Hugging Face (479 MB for Laya, 857 MB for Kev-0.8B).
//    Laya and Kev-0.8B can fall back to browser conversion if the hosted pack is unavailable.
const kevala = await Kevala.load({ model: "laya" });

// 2. From the original weights: downloads the authors' checkpoint at its pinned revision
//    (843 MB for Laya, 1.6 GB for Kev-0.8B) and converts it in the browser.
//    This option is supported only for Laya and Kev-0.8B.
const fromCheckpoint = await Kevala.load({ model: "laya", from: "checkpoint" });

// 3. A .kevala file you host or already have: a URL, an ArrayBuffer, or a Blob.
const own = await Kevala.load({ model: "https://example.com/laya-q8.kevala" });
```

Kev-4B, Kev-9B, and the SemIf Qwen models require a converted `.kevala` pack. They do not
download and convert a large checkpoint automatically when a hosted pack is unavailable. See
[the model guide](models.md) for names, readout differences, and memory limits.

Named loading keys packs by source/base revisions and block size. The additional model entries
also include `packSha256`, so changing their converted bytes selects a new cache entry. Later visits
read the cached file whichever way it arrived. To convert again when a pack is already stored,
clear it first (`clearCache()`, or "Clear stored packs" on the site) or pass `cache: false`. The
site takes `?from=checkpoint` to do the same, for example
`https://bvolpato.github.io/kevala/?from=checkpoint`.

A pack URL is fetched in parallel byte ranges when the server supports them (Hugging Face, GitHub
Pages, S3, or another static host), and as one stream otherwise. Send `Content-Length` for progress
and range loading, and CORS headers when the host is on another origin.

## Downloading a pack directly

The packs live in [bvolpato/kevala-packs](https://huggingface.co/bvolpato/kevala-packs). The runtime
pins them to one commit (`PACKS` in [`js/src/source.js`](../js/src/source.js)); use the same commit
to get the exact bytes it loads:

```sh
rev=<commit from js/src/source.js>
uvx --from huggingface-hub hf download bvolpato/kevala-packs laya-q8.kevala kev-0.8b-q8.kevala \
  --revision "$rev" --local-dir packs
# or, without the Hugging Face CLI
curl -L -o laya-q8.kevala "https://huggingface.co/bvolpato/kevala-packs/resolve/$rev/laya-q8.kevala"
```

Then serve the file yourself (pass its URL as `model`), or load it server side:

```js
import { loadFile } from "kevala/node";
const kevala = await loadFile("packs/laya-q8.kevala");
```

`kevala/node` uses the CPU WebAssembly backend and rejects packs of 2 GiB or more before reading
their weights. Use browser WebGPU or the native 64-bit CLI for larger packs. A downloaded file
being cached does not reduce the memory required to load it.

## Converting the original weights

Build the command line, then download each checkpoint at the revision pinned in `MODELS`
([`js/src/source.js`](../js/src/source.js)). Only the files the converter reads are needed: the Laya
repo also holds multilingual and typed-decision variants that would triple the download.

```sh
cargo build --release -p kevala-cli
kevala=target/release/kevala

# Laya (843 MB)
uvx --from huggingface-hub hf download convaiinnovations/laya model.safetensors encoder/config.json rl_agent_config.json \
  tokenizer/tokenizer.json --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir ckpt/laya
$kevala convert ckpt/laya -o packs/laya-q8.kevala --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982

# Kev-0.8B: its LoRA adapter and pointer head, and the Qwen3.5 base they apply to
uvx --from huggingface-hub hf download jaredpalmer/kev-0.8b adapter_config.json adapter_model.safetensors head.pt tokenizer.json \
  --revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --local-dir ckpt/kev-0.8b
uvx --from huggingface-hub hf download Qwen/Qwen3.5-0.8B-Base config.json model.safetensors-00001-of-00001.safetensors \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --local-dir ckpt/qwen3.5-0.8b-base
$kevala convert-kev --base ckpt/qwen3.5-0.8b-base --kev ckpt/kev-0.8b -o packs/kev-0.8b-q8.kevala \
  --kev-revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
```

The revision flags only label the pack header; the weights are whatever the directories hold, so
download them at the same revisions. The browser conversion (`from: "checkpoint"`) runs the same Rust
converter, compiled to WebAssembly.

<a id="optional-kev-and-semif-style-models"></a>

### Optional Kev and SemIf models

Use [`tools/convert_models.py`](../tools/convert_models.py) for Kev-4B, Kev-9B, and the SemIf
Qwen3.5-0.8B, 2B, and 4B packs. Its source manifest,
[`tools/model-sources.json`](../tools/model-sources.json), pins every model revision, the exact base
checkpoint for each Kev adapter, and the SemIf method revision.

```sh
cargo build --release -p kevala-cli
uv run tools/convert_models.py --models kev-4b semif-qwen3.5-0.8b

# Optional larger conversions. Each selected model is converted serially.
uv run tools/convert_models.py --models kev-9b semif-qwen3.5-2b semif-qwen3.5-4b
```

The default output directory is `tmp/`; `--output-dir` overrides it. Each conversion writes a
`<model>-q8.kevala` file and a neighboring JSON manifest with its SHA-256, byte size, source
revisions, and pack metadata. The native converter reads the safetensors index and tensor ranges
across shards. It still holds the output pack and conversion scratch in RAM, so reserve enough
disk and memory for the selected checkpoints and run large conversions one at a time.

Kev conversion merges the adapter into the matching base in f32 before quantizing and retains
the trained pointer head. SemIf conversion uses a frozen Qwen instruction model with no
adapter, stores native output rows for labels `A` through `P` in f32, and records the method's
provenance. The text backbone is packed; the vision tower and multi-token-prediction weights are
not part of the inference pack.

For SemIf, the helper loads `AutoTokenizer` and calls `save_pretrained()` before passing its
materialized `tokenizer.json` to `convert-semif --tokenizer`. This step matters: Transformers can
replace the raw Qwen checkpoint's pre-tokenizer and added-token configuration. Passing the raw
file can produce different prompt IDs. Prefer the helper over assembling these inputs manually.

Conversion does not run model parity or game benchmarks, and it does not upload unless
`--publish` is supplied. Keep the `.kevala` files and downloaded checkpoints out of Git.

## Checking a pack

Check a converted pack against reference fixtures from the same upstream model and revision
before publishing it. The existing Laya and Kev-0.8B commands are:

```sh
$kevala inspect packs/laya-q8.kevala                                # header, sizes, provenance
$kevala parity packs/laya-q8.kevala tests/fixtures/golden.json      # want argmax 41/41, max|dp| < 0.03
$kevala parity-kev packs/kev-0.8b-q8.kevala tests/fixtures/golden-kev.json   # want argmax 13/13
$kevala parity-semif tmp/semif-qwen3.5-0.8b-q8.kevala tests/fixtures/golden-semif.json --max-dp 0.05
```

The fixtures come from the authors' own PyTorch code ([`tools/golden.py`](../tools/golden.py) and
[`tools/golden_kev.py`](../tools/golden_kev.py)); regenerate them when a model's revision changes.
[`tools/golden_semif.py`](../tools/golden_semif.py) records the complete prompt IDs, native label
logits, and conditional probabilities for a pinned Qwen instruction model using the SemIf contract.
Do not use the Kev-0.8B results as evidence for Kev-4B, Kev-9B, or a SemIf model. Large-model
validation is opt-in and belongs with conversion or runtime changes that affect those models;
routine PR checks need not download and run every checkpoint. For seeded game comparisons, see
[the Tetris evaluation instructions](models.md#conversion-and-validation).

`parity-semif` requires exact prompt IDs, matching choices, and a maximum absolute score difference
of 0.05 by default. This is a separate int8 fidelity threshold: the checked 0.8B pack differs from
its float32 reference by up to 0.0311 on WebGPU, above Laya's 0.03 gate. The threshold does not
establish decision accuracy or calibration. Use `--max-dp` to impose a stricter application limit.

## Publishing packs

[`packs/README.md`](../packs/README.md) is the model card of the Hugging Face repo. Uploading needs a
Hugging Face token with write access to the repo. Authenticate with
`uvx --from huggingface-hub hf auth login`; an `HF_TOKEN` in the environment takes precedence over
the saved login.

After validating optional packs, publish the existing files and their manifests without converting
again:

```sh
uv run tools/convert_models.py --models kev-4b kev-9b \
  semif-qwen3.5-0.8b semif-qwen3.5-2b semif-qwen3.5-4b \
  --upload-only --publish bvolpato/kevala-packs --model-card packs/README.md
```

Use the same `--output-dir` as conversion if it was overridden. The helper uploads the selected
packs and their manifests. `--model-card` includes that README, `LICENSE`, and
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES) in the same commit. Keep the card's listed models,
provenance, measured validation, and limitations consistent with the uploaded files.
For manual uploads, the Hugging Face CLI is also available through `uvx`:

```sh
uvx --from huggingface-hub hf repo create bvolpato/kevala-packs --exist-ok
uvx --from huggingface-hub hf upload bvolpato/kevala-packs packs . --include "*.kevala" --include README.md \
  --commit-message "Laya and Kev-0.8B int8 packs"
curl -s https://huggingface.co/api/models/bvolpato/kevala-packs | jq -r .sha   # the commit to pin
```

Then point the runtime at that commit:

1. Set `PACKS` in `js/src/source.js` to `https://huggingface.co/bvolpato/kevala-packs/resolve/<sha>`,
   each model's `pack` to its byte size, and its `packSha256` to the digest from the conversion manifest.
2. Load each model on the site with its stored pack cleared, and check that the progress shows the
   pack downloading, not a conversion.
3. Release a new version, so npm and the CDN serve the new pin. jsDelivr caches `kevala@latest`, so
   purge each file after publishing: `https://purge.jsdelivr.net/npm/kevala@latest/js/src/<file>`.

A pack is pinned by commit, never by `main`, so a later upload never changes what an existing
release loads. New entries include the pack hash in their cache key (`upstreamKey` in
`js/src/source.js`). Laya and Kev-0.8B retain their legacy revision/block keys while their bytes are
unchanged. If republishing either with different bytes, add or update `packSha256` as well as the
hosted pin; changing the hosted URL alone does not invalidate a named model's legacy cache entry.

Commit converter source, pinned input manifests, documentation, and small reference fixtures to
Git. Publish weight binaries and their conversion manifests to Hugging Face. `.kevala` packs are
not source files, and should not be committed to this repository or bundled into the npm package.
