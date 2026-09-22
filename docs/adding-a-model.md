# Adding a model family

kevala is organised so a new family (a new backbone, head, request template, or modality) plugs in
without touching the others. A family is identified by the `arch` string in its pack's config.

## 1. Rust: the family

Add a module under `crates/kevala/src/` that implements `runtime::Model`:

```rust
pub trait Model: Any {
    fn arch(&self) -> &'static str;
    fn info(&self) -> &Value;                 // provenance from the pack header
    fn modalities(&self) -> &[Modality];      // what `parts` it can read
    fn tokenizer(&self) -> &Tokenizer;
    fn decide(&mut self, requests: &[Request]) -> Result<Vec<Value>, String>;
    fn as_any(&mut self) -> &mut dyn Any;     // for backends that drive the pieces themselves
}
```

and register it in `runtime::FAMILIES`:

```rust
Family { arch: "myfamily", about: "one line", load: load_myfamily },
```

A family owns three pieces; reuse what exists:

- **Template**: request to token sequences. Laya renders one sequence per question with a `[MASK]`
  per option (`sequence.rs`); Kev renders causal rows sharing the state (`kev.rs`, `render` /
  `encode`). Match your reference implementation byte for byte and add a golden fixture.
- **Backbone**: the layers. `kernels.rs` has int8-weight matmul, layer/RMS norm, attention, rotary
  embeddings and activations over `simd::F4`; `kev.rs` has causal GQA attention and Gated DeltaNet.
- **Head**: hidden states to answer distributions (Laya's marker scorer, Kev's pointer head).

`runtime::text_requests` checks modalities and folds text parts into the state for text-only
families.

## 2. Rust: the converter

Write the upstream checkpoint into a pack with `pack::Writer` (declare tensors, `layout()`, write the
bytes). Set `config.arch`, `config.modalities` and whatever config your family reads. If the family
should be convertible in the browser, make the converter streaming: `convert_kev::KevConvert` shows
the shape (header first, then source tensors in file order, each output tensor written as soon as its
inputs are complete). Add a CLI command in `crates/kevala-cli`.

Tensors whose names start with `enc.`, `head.` or `L.` are trunk tensors (`model::is_trunk`); the
rest belong to the coordinator. Keep that split and the GPU and shard layouts work unchanged.

## 3. JavaScript: the architecture plugin (optional)

Without a plugin, a family the WebAssembly build knows already runs on the CPU in one instance through
`kevala_decide`. A plugin adds a GPU trunk, shard support, and in-browser conversion. It is a module whose
default export follows `js/src/archs/index.js`:

```js
export default {
  arch: "myfamily",
  about: "one line",
  maxShards: (header) => 1,
  createGpu: (gpu, layout, header) => new MyGpuTrunk(gpu, layout, header),
  initGpu: async (engine) => engine.gpu.init(),
  run: async (engine, requests) => ({ responses, timing }),
  convert: async (module, spec, opts) => packBytesOrStream,
};
```

Built-in plugins are registered in `js/src/archs/index.js`. Third-party plugins load by URL, no fork
needed:

```js
const kevala = await Kevala.load({ model: "https://example.com/my.kevala", plugins: ["https://example.com/myfamily.js"] });
```

`gpu.js` exports the building blocks: `GpuWeights` (streams trunk tensors into GPU buffers),
`pipeline()` (compiles WGSL with readable errors), `matmulPipelines` / `encodeMatmul` (the int8
matmul with length-sized tiles and automatic split-K), and the `gpu.wgsl(kernel, spec)` function
the worker passes to `createGpu`, which returns any kernel from the binary. New kernels go in
`crates/kevala/src/wgsl/` as `.wgsl` files, listed in `crates/kevala/src/gpu.rs`; the
`tests/wgsl.rs` test renders each one for every specialization.

## 4. The model registry

Add an entry to `MODELS` in `js/src/source.js` so pages can load it by name:

```js
"mymodel-1b": { arch: "myfamily", repo: "org/mymodel", revision: "<commit sha>", block: 32, license: "..." },
```

Pin a commit, never a branch: the pack cache is keyed by revision.

## 5. Modalities

Requests carry typed `parts` (`{ type: "text" | "image" | "audio", ... }`). A family lists what it reads
in `config.modalities`; anything else is rejected with an error naming the part. To add images to a
family, add an encoder that turns image parts into embeddings (Qwen3.5 checkpoints ship a vision tower
the Kev backbone could use), splice them at the template's placeholder tokens, and add `"image"` to the
pack's modalities.

## 6. Prove it

Write a reference dump from the upstream code (see `tools/golden.py`, `tools/golden_kev.py`), commit
the fixture, and add a parity command and a browser parity page. Token ids must match exactly; report
argmax agreement and the maximum probability difference at the precision you ship.
