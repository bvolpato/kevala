# Choosing and running models

Laya is the demo default. Kev-0.8B is the small decoder option. Larger Kev models and the
SemIf-style Qwen models are additional choices, loaded only when selected. Their weights are
distributed as `.kevala` packs on [Hugging Face](https://huggingface.co/bvolpato/kevala-packs).

The catalog pins the [verified pack revision](https://huggingface.co/bvolpato/kevala-packs/tree/45da41504c6c117eca940103e49dd5eb1c3eab4f).
Load any model below by name. With the development server, `?pack=local#/tetris` loads local files
from `tmp/` instead.

| Model name | Weights | Decision readout | Pack download |
|---|---|---|---:|
| `laya` | ModernBERT-large with Laya's trained decision layers | Marker scorer and act head | 479 MB |
| `kev-0.8b` | Qwen3.5-0.8B-Base with Kev's merged LoRA adapter | Trained pointer head | 857 MB |
| `kev-4b` | Qwen3.5-4B-Base with Kev's merged LoRA adapter | Trained pointer head | 4.76 GB |
| `kev-9b` | Qwen3.5-9B-Base with Kev's merged LoRA adapter | Trained pointer head | 8.96 GB |
| `semif-qwen3.5-0.8b` | Frozen Qwen3.5-0.8B instruction model | Native option-label logits | 855 MB |
| `semif-qwen3.5-2b` | Frozen Qwen3.5-2B instruction model | Native option-label logits | 2.13 GB |
| `semif-qwen3.5-4b` | Frozen Qwen3.5-4B instruction model | Native option-label logits | 4.75 GB |

Pack sizes above use decimal MB/GB; runtime memory is higher. Model names describe parameter sizes,
not download sizes or memory requirements. The runtime's [`MODELS`](../js/src/source.js) catalog
records pack bytes; conversion also writes a manifest with
the byte count, SHA-256, source revisions, and configuration. Only Laya and Kev-0.8B support
`from: "checkpoint"` in the browser. All other choices load a previously converted pack.

```js
import { Kevala } from "kevala";

const model = await Kevala.load({
  model: "kev-4b", // or "semif-qwen3.5-2b"
  backend: "webgpu",
  onProgress: console.log,
});
const result = await model.decide("The customer was charged twice.", {
  refund: { type: "noul", instructions: "Should billing investigate a duplicate charge?" },
});
console.log(result.answers, result.raw_probabilities);
model.dispose();
```

## How SemIf-style scoring works

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

Kevala quantizes the backbone to int8 and preserves the selected label rows in f32. It computes
only those output rows rather than materializing logits for the full vocabulary. The shared
evidence prefix runs once for a request's questions, and recent prefixes can be reused across
requests. Prefix reuse compares exact token IDs from complete prompts, so splitting the prompt
does not change its tokenization. Model caches are separate from the browser's stored pack files.

### Interpreting the scores

SemIf-style packs support `noul`, `choice`, and `score`, with **at most 16 options per question**.
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

The prompt, direct option-logit readout, and evidence-prefix approach are adapted from SemIf's MIT
code. The full notice is retained in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES). Qwen and Kev
weights retain their own model licenses and pinned provenance.

## Memory and backends

| Backend | What must fit |
|---|---|
| Browser WebGPU | Transformer weights and working buffers on the GPU; tokenizer, embeddings, and readout in a WASM coordinator |
| Browser WASM / `kevala/node` | The whole Kev or SemIf pack in one WASM allocation, plus working memory |
| Native 64-bit CLI | Pack weights and working memory in system RAM; the loader reads directly into aligned storage |

Rust's aligned allocations in wasm32 must be under 2 GiB. Browser CPU and Node loading reject
oversized packs early. WebGPU bypasses this restriction for the transformer weights: source and
destination offsets remain precise beyond 4 GiB, and the pack streams into individual GPU tensors.
Its coordinator subset must still fit under the WASM allocation limit. The native CLI does not
have that wasm32 limit, but sufficient system RAM is required.

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
the required checkpoints, and invokes the native converter serially. SemIf-style conversion
first saves a tokenizer materialized by Transformers `AutoTokenizer`; using the raw checkpoint's
`tokenizer.json` can change tokenization. See [Packs](packs.md#optional-kev-and-semif-style-models)
for conversion and publication commands.

Weight binaries belong in the Hugging Face repository, not Git. Routine development checks do
not require downloading or running every large model. Validate a new pack explicitly against its
own upstream reference and record the source revision, precision, tokenizer, hardware, and backend.
The README's Laya and Kev-0.8B parity numbers do not apply automatically to these additional models.
See [the optional-model measurements](model-benchmarks.md) for conversion fidelity and seeded games.

For a local Tetris comparison, put the selected packs in `tmp/`, build the WASM modules, and run
`pnpm serve`. The opt-in evaluation page accepts, for example:

```text
http://127.0.0.1:8080/dev/tetris-eval.html#model=kev-4b&backend=webgpu&pieces=20&seeds=1,2,3&batch=4
http://127.0.0.1:8080/dev/tetris-eval.html#model=semif-qwen3.5-0.8b&backend=webgpu&pieces=20&seeds=1,2,3&batch=4
```

Use the same seeds, piece limit, candidate descriptions, and batching when comparing models.
Record both game outcomes and decision latency. These runs measure a specific game policy;
they are not a general model ranking.
