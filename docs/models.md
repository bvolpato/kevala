# Choosing and running models

The runtime catalog registers Laya, Kev v1.0, H2O-Lightning-4B, Clef-flash, SemIf, and Gemma 4.
The website preview picker is currently limited to 4B and smaller: it offers Laya, Kev-0.8B/4B,
H2O-Lightning-4B, and SemIf 0.8B/2B/4B, with Gemma 4 E2B as an image-demo exception.
Kev-9B, Clef-flash, and Gemma 4 E4B remain available through
the JS runtime API but are not shown in the picker while that cap is active.
Kev v1.0 starts at 0.8B; SemIf starts at 0.8B. Changing a size waits for **Load** before fetching
weights. Gemma 4 has E2B and E4B dense text models in the runtime catalog.
For refreshed Kev and the new models, BF16 is the default precision and Q8 is an explicit option.
Saved choices and explicit model URLs retain their selected size. The model picker shows actual
local pack availability when running the development server. It discovers validated `tmp/` packs
automatically; `?pack=local` makes the preview local-only and disables variants without an exact
matching local pack.

The published pack catalog pins [Laya and SemIf](https://huggingface.co/bvolpato/kevala-packs/tree/45da41504c6c117eca940103e49dd5eb1c3eab4f) and [Gemma 4](https://huggingface.co/bvolpato/kevala-packs/tree/c70e9136938026e00cbbdec9db027d862963f180). Kev v1.0, H2O-Lightning, and Clef-flash use newer pinned checkpoints and do not point to the older Kev pack bytes. Their converted packs are not published yet.
The JS runtime registers every model below. Published packs load by name, supported checkpoints can
convert in the browser, and native-only models without published packs need a converted `.kevala`
URL. With the development server, the picker automatically discovers validated local files from
`tmp/`. Add `?pack=local` to make the preview local-only; variants without an exact pack stay
disabled.

| Model name | Weights | Decision readout | Output pack size |
|---|---|---|---:|
| `laya` | ModernBERT-large with Laya's trained decision layers | Marker scorer and act head | 479 MB |
| `kev-0.8b` | Qwen3.5-0.8B-Base with Kev v1.0's merged LoRA adapter | Trained pointer head | 1.51 GB BF16; 857 MB Q8 |
| `kev-4b` | Qwen3.5-4B-Base with Kev v1.0's merged LoRA adapter | Trained pointer head | 8.43 GB BF16; 4.76 GB Q8 |
| `kev-9b` | Qwen3.5-9B-Base with Kev v1.0's merged LoRA adapter | Trained pointer head | 15.90 GB BF16; 8.96 GB Q8 |
| `h2o-lightning-4b` | H2O-Lightning-4B, based on Qwen3.5-4B | H2O prompt, label readout, and calibration | 8.43 GB BF16; 4.75 GB Q8 |
| `clef-flash` | Cloudflare Clef-flash, based on Qwen3.5-9B | Joint decision head | About 18.41 GB BF16; 10.59 GB Q8 |
| `semif-qwen3.5-0.8b` | Frozen Qwen3.5-0.8B instruction model | Native option-label logits | 855 MB |
| `semif-qwen3.5-2b` | Frozen Qwen3.5-2B instruction model | Native option-label logits | 2.13 GB |
| `semif-qwen3.5-4b` | Frozen Qwen3.5-4B instruction model | Native option-label logits | 4.75 GB |
| `gemma-4-e2b` | Frozen Gemma 4 E2B instruction decoder; optional vision companion | Native option-label logits | 5.22 GB decoder |
| `gemma-4-e4b` | Frozen Gemma 4 E4B instruction model, text-only | Native option-label logits | 8.41 GB |

Clef sizes are estimates; the other BF16/Q8 sizes come from exact local conversions or published pack manifests. Runtime memory is higher.
Model names describe parameter sizes, not download sizes or memory requirements. The runtime's
[`MODELS`](../js/src/source.js) catalog pins source revisions; conversion writes a manifest with
the actual byte count, SHA-256, precision, source revisions, and configuration. The Gemma packs are
exactly 5,217,421,952 bytes (E2B) and 8,407,043,136 bytes (E4B). Only Laya and Kev-0.8B support
browser conversion. Kev-4B/9B, H2O-Lightning, and Clef-flash need native conversion until smaller
browser conversion paths exist. Their first-load path fails fast while no matching pack is
published; load a converted `.kevala` URL when you have one.
Gemma 4 E2B image decisions add a separately loaded 337,302,144-byte BF16 vision companion on
WebGPU. That companion is not published yet; see [conversion and loading](gemma4.md#vision-companion).
Gate image controls on the loaded `info.modalities`, not the model name or upstream capabilities.

```js
import { Kevala } from "kevala";

const model = await Kevala.load({ model: "kev-0.8b", backend: "webgpu" }); // BF16 by default
// Use precision: "q8" only when you want a smaller quantized pack; "f32" is available in the API and native CLI.
const result = await model.decide("The customer was charged twice.", {
  refund: { type: "noul", instructions: "Should billing investigate a duplicate charge?" },
});
console.log(result.answers, result.raw_probabilities);
model.dispose();
```

Kev-4B/9B, H2O-Lightning, and Clef-flash require native conversion while no matching pack is
published. The picker exposes a required URL field for native-only models within its 4B cap. For
models outside the cap, load a converted pack through the runtime API, for example
`{ name: "clef-flash", url: "https://example.com/model.kevala" }`. If you pass an explicit
`precision` with a pack URL, Kevala checks it against the pack header before uploading weights.

<a id="how-semif-style-scoring-works"></a>
<a id="how-semif-scoring-works"></a>

## How SemIf scoring works

[SemIf](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37)
provides a decision method for frozen language models. Its `direct-options-v1` prompt supplies
evidence, a criterion, and options labeled `A` through `P`. Kevala applies Qwen3.5's non-thinking
chat template and reads the next-token logits for those labels at the final prompt position.
A softmax over the declared labels produces the option scores. There is no generated answer or
autoregressive decoding loop, although the model still processes the prompt through its full backbone.

These packs contain Qwen instruction weights with this prompt and readout. There is no SemIf
adapter or additional training step. SemIf's upstream Qwen3.5 reference uses the 4B model; the
0.8B and 2B choices extend the method to smaller models in Kevala. They need their own evaluation.
Kev uses a different checkpoint: its LoRA adapter and pointer head are trained for decisions.

Kev v1.0's source revisions come from its [official release](https://github.com/jaredpalmer/kev/releases/tag/kev-1.0).
Kev v1.0 responses use four decimal places and a revised normalized confidence calculation. The
converter records this response format in the pack config; older Kev packs keep their previous
response behavior. H2O-Lightning uses its own decision prompt, next-token label readout, and calibration temperatures.
The prompt contract and calibration values are recorded in the pinned H2O source configuration.
Clef-flash includes a separate joint head on the Qwen3.5-9B backbone. These integrations are
text-only in Kevala. Upstream image support is not wired into this runtime.

Gemma 4 uses Google's frozen E2B or E4B instruction weights with the same direct option readout,
but renders Gemma's own instruction chat template with thinking disabled. E2B and E4B are dense
text models with per-layer embeddings, not MoE models. Their packs omit the source checkpoint's
vision and audio towers. There is no SemIf adapter or additional decision training.

Q8 conversion quantizes the backbone to int8. BF16 conversion keeps backbone matrices unquantized;
F32 stores them in f32 and uses substantially more memory. Both preserve the readout in f32. The
engine computes
only those output rows rather than materializing logits for the full vocabulary. For SemIf Qwen
packs, the shared evidence prefix runs once for a request's questions, and recent prefixes can be
reused across requests. Prefix reuse compares exact token IDs from complete prompts, so splitting
the prompt does not change its tokenization. Model caches are separate from the browser's stored
pack files.

### Interpreting the scores

SemIf and Gemma 4 packs support `noul`, `choice`, and `score`, with **at most 16 options per question**.
They return the Kev response shape, including full precision `raw_probabilities`, plus:

```json
{"probability_status": "conditional option score; uncalibrated as decision confidence"}
```

The probabilities are conditional on the listed labels. A high score means the model favors that
option over the alternatives in this prompt; it is not a measured probability that the decision
is correct. Changing the criterion, options, order, or model can change the distribution. Evaluate
thresholds on representative data before using them to route or reject real requests. A larger
model or a higher Tetris score alone does not establish better general decision quality.

This integration uses Kevala's typed-question API, rather than providing the SemIf Python API.
Kevala also accepts a single-option choice and arbitrary JSON states; upstream SemIf's validator
requires 2-16 options and a nonempty string, object, or list state. The checked compatibility is
the prompt and numerical scoring method for inputs accepted by both.

The prompt and direct option-logit readout are adapted from SemIf's MIT code. Its evidence-prefix
approach applies to the SemIf Qwen packs. The full notice is retained in
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES). Qwen and Kev weights retain their own model licenses
and pinned provenance.

## Memory and backends

| Backend | What must fit |
|---|---|
| Browser WebGPU | Transformer weights, embedding tables, and working buffers on the GPU; tokenizer and readout in a WASM coordinator |
| Browser WASM / `kevala/node` | The whole pack in one WASM allocation, plus working memory; Gemma 4 packs exceed this limit |
| Native 64-bit CLI | Pack weights and working memory in system RAM; the loader reads directly into aligned storage |

Rust's aligned allocations in wasm32 must be under 2 GiB. Browser CPU and Node loading reject
oversized packs early. WebGPU bypasses this restriction for transformer and embedding weights:
source and destination offsets remain precise beyond 4 GiB, and the pack streams into individual GPU
tensors. Kev, H2O-Lightning, and Clef-flash keep embedding tables in GPU row buffers and gather only
the requested rows there, so those tables do not use coordinator memory. The coordinator subset
must still fit under the WASM allocation limit. The native CLI does not have that wasm32 limit, but
sufficient system RAM is required.

GPU limits apply to individual buffers as well as total available memory. Larger batches also
need more activation and cache memory. The runtime splits batches of requests when needed; a
single request must still fit. `backend: "webgpu"` reports a GPU load failure. With
`backend: "auto"`, CPU fallback can also fail if the pack exceeds the WASM limit.

Browser caching avoids downloading the same weights again. It does not make them GPU-resident
between page loads or reduce the memory required for inference. Large packs need sufficient
browser storage quota, and cached load time still depends on file size, storage, shader setup,
and warmup. Run one large model at a time when comparing performance.

## Conversion and validation

The pinned-source conversion entry point is [`tools/convert_models.py`](../tools/convert_models.py).
It uses pinned revisions in [`tools/model-sources.json`](../tools/model-sources.json), downloads
the required checkpoints, and invokes the native converter serially. SemIf conversion
first saves a tokenizer materialized by Transformers `AutoTokenizer`; using the raw checkpoint's
`tokenizer.json` can change tokenization. See [Packs](packs.md#optional-kev-and-semif-models)
for conversion and publication commands. Gemma conversion extracts the dense text trunk from the
multimodal checkpoint and requires WebGPU in the browser or the native 64-bit CLI; its Kevala input
limit is 4096 tokens.

The current reference checks cover small conversion-fidelity suites, not general model accuracy or
confidence calibration. Kev results compare browser conversion and WebGPU inference with pinned
Kev v1.0 CPU F32 outputs from a merged-F32 LoRA. H2O results compare WebGPU inference with pinned
H2O v1.2.3 CUDA BF16 outputs, widening the selected LM label rows to F32. Clef results compare with
the pinned BF16 Qwen backbone and original joint head and lexical readout, using CPU decoder offload.
Each delta is against pinned upstream probabilities.

| Model and format | Runtime path | Reference precision and path | Decisions | Max probability difference | Notes |
|---|---|---|---:|---:|---|
| Kev v1.0 0.8B BF16 | Browser conversion + WebGPU | Pinned Kev v1.0 CPU F32, LoRA merged in F32 | 13/13 | 0.002812028 | |
| Kev v1.0 0.8B Q8 | Browser conversion + WebGPU | Pinned Kev v1.0 CPU F32, LoRA merged in F32 | 13/13 | 0.0066531 | |
| H2O-Lightning-4B BF16 | WebGPU | Pinned H2O v1.2.3 CUDA BF16; label rows widened to F32 | 3/3 | 0.004327924 | |
| H2O-Lightning-4B Q8 | WebGPU | Pinned H2O v1.2.3 CUDA BF16; label rows widened to F32 | 3/3 | 0.007321618 | |
| Clef-flash BF16 | Native streamed inference | Pinned BF16 Qwen with original joint head/readout, CPU decoder offload | 5/5 | 0.009297 | Peak RSS 4.75 GiB |
| Clef-flash Q8 | Native inference | Pinned BF16 Qwen with original joint head/readout, CPU decoder offload | 5/5 | 0.028130 | Two records; exact tokens/spans; peak RSS 9.95 GiB; zero swaps; within the predeclared 0.03 tolerance |
| Clef-flash Q8 | WebGPU | Pinned BF16 Qwen with original joint head/readout, CPU decoder offload | 5/5 | 0.028166843 | Browser probabilities are rounded to four decimals |

These checks show implementation agreement on the listed fixtures only. They do not establish a
general quality ranking between these models.

Weight binaries belong in the Hugging Face repository, not Git. Routine development checks do
not require downloading or running every large model. Validate a new pack explicitly against its
own upstream reference and record the source revision, precision, tokenizer, hardware, and backend.
The README's Laya and previous Kev benchmark numbers do not apply automatically to these additional models,
including Gemma 4. See [Gemma 4 notes](gemma4.md) and [the optional-model measurements](model-benchmarks.md)
for conversion fidelity and seeded games.

For a local Tetris comparison, put the selected packs in `tmp/`, build the WASM modules, and run
`pnpm serve`. The opt-in evaluation page accepts, for example:

```text
http://127.0.0.1:8080/dev/tetris-eval.html#model=kev-4b&backend=webgpu&pieces=20&seeds=1,2,3&batch=4
http://127.0.0.1:8080/dev/tetris-eval.html#model=semif-qwen3.5-0.8b&backend=webgpu&pieces=20&seeds=1,2,3&batch=4
http://127.0.0.1:8080/dev/tetris-eval.html#model=gemma-4-e2b&backend=webgpu&pieces=20&seeds=1,2,3&batch=4
```

Use the same seeds, piece limit, candidate descriptions, and batching when comparing models.
Record both game outcomes and decision latency. These runs measure a specific game policy;
they are not a general model ranking.
