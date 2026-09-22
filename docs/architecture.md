# How kevala works

kevala runs System 1 decision models (Laya, Kev, and whatever family comes next) inside a web page.
A request is a state plus typed questions (`noul`, `choice`, `score`), and the answer is a probability
for every option, in one forward pass. Nothing leaves the tab.

```
 page ──► index.js ──postMessage──► engine worker ──► coordinator (WebAssembly, Rust)
                                        │                tokenizer · request template · embeddings · head · response
                                        │
                                        ├──► WebGPU trunk (WGSL kernels, the whole pass on the GPU)
                                        └──► WebAssembly shard workers (tensor parallel, CPU)
```

## Layers

**Rust core (`crates/kevala`, zero dependencies).** Everything that has to be exact lives here, and it
compiles for native targets and `wasm32-unknown-unknown` alike:

- `json.rs`: JSON parsing plus a byte-exact Python `json.dumps`, because Laya serializes states with it.
- `unicode.rs`, `tokenizer.rs`: Hugging Face compatible byte-level BPE (NFC normalizer, added tokens,
  GPT-2 and Qwen2 pre-tokenizer regexes as hand-written scanners). Checked against the `tokenizers`
  library on fixture corpora and millions of fuzzed strings.
- `content.rs`: the request model. `state` is text or JSON; `parts` carries typed content by modality.
- `runtime.rs`: the family registry (`Model` trait, `FAMILIES`), keyed by the pack's `config.arch`.
- Families: `engine.rs` + `sequence.rs` + `model.rs` (Laya), `kev.rs` (Kev). Each owns its template,
  backbone and head.
- `kernels.rs`, `simd.rs`: CPU kernels over a four-lane vector type that maps to WebAssembly SIMD128
  (with relaxed-SIMD fused multiply-add when available), NEON natively, or plain arrays.
- `pack.rs`, `convert.rs`, `convert_kev.rs`, `torchpt.rs`: the `.kevala` format and converters from
  upstream checkpoints (safetensors, LoRA adapters, `torch.save` files).

**WebAssembly ABI (`crates/kevala-wasm`).** A small C ABI: load a pack, decide, plus the entry points the
GPU and shard backends drive (prepare, embed, finish). Three builds ship: `relaxed` (SIMD128 +
relaxed-simd), `simd`, and `base`; the runtime picks the best one the browser validates.

**Browser runtime (`js/src`, plain ES modules, no dependencies).**

- `index.js`: the public API (`Kevala.load`, `decide`, `decideMany`, `dispose`).
- `engine-worker.js`: family-agnostic pipeline. Resolves the pack source, streams it, chooses the
  backend, and packs concurrent requests into shared forward passes.
- `archs/*.js`: architecture plugins. Each says how its family runs on the GPU or across shards, and
  how to convert its upstream checkpoint in the browser.
- `gpu.js`, `gpu-kev.js`: WebGPU kernels and trunks.
- `source.js`: model registry, downloads with progress, OPFS/Cache API storage, and the streaming
  layout applier.

## The `.kevala` pack

One file per model: a 16-byte prefix (`KVLA`, format version, header length), a JSON header with the
model provenance, the family config and a tensor table, the tokenizer blob, then tensors, each 64-byte
aligned so a loader can hand them to WebAssembly or WebGPU without reshuffling.

Tensors are `f32` or `q8`: int8 row-major weights with one f32 scale per 32 weights of a row
(symmetric absmax). Matrices are q8; norms, biases, small gate projections and decision heads stay f32.

Because every tensor's offset is in the header, a pack can be split while it streams: the runtime
computes *layouts* (lists of byte-range copies) for the coordinator, for a GPU trunk, or for each
tensor-parallel shard, and applies them to the stream so no worker ever holds bytes it does not keep.

## Where the weights come from

`Kevala.load({ model: "laya" })` downloads the upstream checkpoint at a pinned Hugging Face revision,
converts it in the browser (the same Rust converter the CLI uses, compiled to WebAssembly), stores the
pack in the Origin Private File System and loads it. Later visits read it from disk. Nothing is
re-hosted. For production you can convert once with the CLI and serve the `.kevala` file yourself:
`Kevala.load({ model: "https://example.com/laya-q8.kevala" })`.

Kev's conversion downloads only the byte range of the Qwen3.5 base that holds the language model (the
vision tower and the multi-token-prediction head are skipped), merges Kev's LoRA adapter in f32 and
quantizes each tensor as it arrives.

Checkpoints download as six 16 MB byte ranges at a time, handed to the converter in order: a
single stream from Hugging Face's CDN ran at about 24 MB/s where parallel ranges reached 38 MB/s
on the same connection. Only the converted pack is stored, so switching backend or reloading the
page reads it back from disk in under a second.

## Backends

**WebGPU.** The coordinator embeds tokens and runs the family's head; the GPU runs every transformer
layer and returns only the rows the head reads. Kernels:

- int8-weight matmul: weights are widened in workgroup memory, XOR-swizzled to avoid bank conflicts,
  and stored there as f16 when the GPU has `shader-f16` (products and sums stay f32). Workgroup
  memory traffic is what limits this kernel, and halving it made every shape 1.6-1.75x faster, with
  rounding (about 3e-4 relative) far below the int8 weights' own error. Tiles are 64 columns by 16,
  32, 48 or 64 rows, picked by the input length, so a 20-token request does not pay for 64 rows.
  Narrow outputs at short lengths split K across workgroups (a 1024-wide projection at 64 tokens is
  16 tiles, too few to fill a GPU).
- LayerNorm / RMSNorm, rotary embeddings, sliding-window and global attention (Laya), causal GQA
  attention with online softmax and an output gate (Kev), GeGLU / SwiGLU. On GPUs whose subgroups
  are exactly 32 lanes (Apple, NVIDIA), Laya's attention takes 32 keys per step with one key per
  lane, so each query's running max and sum come from `subgroupMax` / `subgroupAdd`.
- Kev's Gated DeltaNet: causal depthwise conv, q/k L2 norm, the gated delta rule, gated RMSNorm.
  The recurrence is sequential in time, so its cost is the length of each token's dependency
  chain: with subgroups, four lanes share a value column (32 of its 128 keys each) and combine
  their partial dot products with `subgroupShuffleXor`, which made it 5x faster than one thread
  per column. The next token's q and k are staged while the current one computes.

Where the time goes is visible per kernel: `kevala.profile(true)` adds `timing.gpu` (milliseconds
per kernel) to every response, and the Playground's Profile tab shows it for any request.
`dev/matmul-bench.html` times matmul variants (tile rows, column groups, split-K target, f16
tiles) on the models' shapes, best of several trials, and checks they agree.

A pass over more than 256 tokens goes out as a few command buffers, waiting for the queue between
them, so a big batch does not freeze the page's rendering while the GPU works.

Kev runs a request's state once, keeps each DeltaNet layer's recurrent state and conv tail and each
attention layer's keys and values, and starts every question branch from them (see Caches below).

**WebAssembly.** One instance runs the whole model, or (Laya) the layers split across N shard workers:
each shard holds a contiguous range of attention heads and MLP columns, turns the replicated residual
stream into a partial update, and the partial updates sum to the full layer update (two exchanges per
layer). This works without `SharedArrayBuffer`, so it needs no cross-origin isolation. On the CPU the
matmul register tile (2x4 or 4x4) is picked per device by timing both at startup.

## Caches

A causal decoder like Kev has a KV cache; a bidirectional encoder like Laya cannot, because every
token attends to every other token, question tokens included, so nothing computed for one question's
sequence is valid for another's. What kevala reuses:

- **Within a request (Kev).** Questions share the state: it runs once, and every question branch
  starts from its carry (the KV cache of the 6 attention layers plus the recurrent state and conv
  tail of the 18 Gated DeltaNet layers). Kev's own server does the same.
- **Across requests (Kev).** The carries of the 4 most recent states of 32 to 1024 tokens stay
  resident (in GPU buffers on WebGPU, in memory on the CPU). A repeated state skips its pass. A
  state that extends a cached one (a growing conversation, a log with new lines, a document with a
  new paragraph) runs only its new tokens, continuing the cached carry. Both are exact: the tests
  check them against a cold engine (`crates/kevala/tests/kev_cache.rs`, `dev/cache-test.html`).
  On WebGPU a repeated 512-token state answers in 28 ms instead of 334 ms.
- **Short single-question rows** (states under 32 tokens) are run as one causal row instead, which
  halves the GPU dispatches; they are cheaper to recompute than to cache.
- **Within a batch (both).** Concurrent requests are packed into one forward pass, and
  `decideMany` packs many states explicitly.

## Fidelity

Parity is checked against the upstream PyTorch code, not against another port:

- `tools/golden.py` runs Laya through the `laya` SDK 0.3.5; `tools/golden_kev.py` runs Kev through
  Kev's own `kev.checkpoint` / `kev.api` code. Both write fixtures under `tests/fixtures/`.
- `kevala parity` / `kevala parity-kev` (native) and `parity.html` / `parity-kev.html` (browser)
  compare token ids exactly and probabilities within a tolerance.
- With f32 weights the Laya port matches PyTorch to within 5e-5 on every logit. int8 packs keep every
  argmax, with a max probability difference of 0.024 (Laya, 41 questions) and 0.0097 (Kev, 13).
