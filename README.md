<h1 align="center">kevala</h1>

<p align="center"><b>Ask questions about text and get answers from small models running on the user's own GPU.</b></p>

<p align="center">
  <a href="https://bvolpato.github.io/kevala/#/tetris"><img src="docs/tetris.gif" alt="Tetris played live by the Laya decision model on WebGPU: the model scores every landing spot and presses the keys to get there" width="760"></a>
</p>

<p align="center">
  <a href="https://bvolpato.github.io/kevala/">Live site</a> ·
  <a href="https://bvolpato.github.io/kevala/#/playground">Playground</a> ·
  <a href="https://bvolpato.github.io/kevala/#/tetris">Tetris</a> ·
  <a href="https://www.npmjs.com/package/kevala">npm</a> ·
  <a href="examples/">Examples</a> ·
  <a href="docs/architecture.md">How it works</a>
</p>

kevala runs System 1 decision models, [Laya](https://huggingface.co/convaiinnovations/laya) and
[Kev](https://github.com/jaredpalmer/kev) today, inside the browser. You give it a piece of text or
JSON and typed questions (`noul` for yes/no, `choice`, `score`), and it returns a probability for
every option of every question in one forward pass. There is no model server, no API key, and no
data leaving the tab.

The engine is Rust with zero dependencies, compiled to WebAssembly, plus WebGPU kernels written for
these models. The browser runtime is a few plain ES modules. The first load downloads a pinned int8
pack of the model from [Hugging Face](https://huggingface.co/bvolpato/kevala-packs) and keeps it in
the browser, so there is nothing to host. If the pack is unreachable, kevala converts the original
checkpoint in the browser instead.

## Quick start

From a CDN, in any page:

```html
<script type="module">
  import { Kevala } from "https://cdn.jsdelivr.net/npm/kevala@latest/js/src/index.js";

  const kevala = await Kevala.load({ model: "laya", onProgress: console.log });
  const r = await kevala.decide("We were billed twice. Refund the duplicate today or we cancel.", {
    team: { type: "choice", instructions: "Which team should handle this?",
            criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages", other: "anything else" } },
    churn: { type: "noul", instructions: "Does the customer threaten to leave?" },
  });
  console.log(r.answers.team.choice, r.answers.churn.noul); // "billing" 0.87
</script>
```

Or from npm, with any bundler (Vite, webpack, esbuild) or none:

```sh
npm install kevala
```

```js
import { Kevala } from "kevala";
```

Building with a coding agent? The site has [a prompt to paste into it](https://bvolpato.github.io/kevala/#/home/agent),
and [`skills/kevala/SKILL.md`](skills/kevala/SKILL.md) is the same guide as an agent skill.

The first visit downloads the model's int8 pack from Hugging Face (Laya: 479 MB). It is stored in the
site's Origin Private File System, so later visits load in under a second. It works from any origin
and needs no special headers. To load the original weights and convert them in the browser instead,
pass `from: "checkpoint"`; to serve the weights yourself, pass the URL of a `.kevala` file. See
[Packs](docs/packs.md) for both, and for how the packs are made and published.

## Models

| name | family | backbone | head | first download | stored pack |
|---|---|---|---|---|---|
| `laya` | `laya` | ModernBERT-large encoder (28 layers) | 2-layer transformer + marker scorer + act head | about 850 MB | 479 MB |
| `kev-0.8b` | `kev` | Qwen3.5-0.8B decoder (18 Gated DeltaNet + 6 gated attention layers), Kev's LoRA merged | pointer head | about 1.6 GB | 857 MB |

Pick with `Kevala.load({ model: "kev-0.8b" })`, or pass the URL of a `.kevala` pack you host. Families are
pluggable: see [Adding a model family](docs/adding-a-model.md).

## Speed

On an Apple M4 Max, with WebGPU in Chrome:

| | Laya | Kev-0.8B |
|---|---:|---:|
| short request (one question, 30 to 45 tokens) | **11 ms** | **11 ms** |
| the same state asked again (cache hit) | | **10 ms** |

Laya scores 32 states in one pass in 186 ms, and Kev answers a 533-token state in 116 ms. Without
WebGPU, a short request takes about 0.4 s on one CPU core. The Playground's Profile tab (or
`kevala.profile(true)`) shows the time of every GPU kernel for any request. [`bench.html`](https://bvolpato.github.io/kevala/bench.html) measures your own machine.

Kev reuses its KV cache: all questions about a state share one pass over it, and the carries of recent
states stay resident, so a repeated state only runs its question tokens and a state that extends a
cached one (a growing conversation) only runs its new tokens. Laya is a bidirectional encoder, where a KV
cache is impossible; it packs every question of a request into one pass instead.

## Fidelity

Every number is checked against the upstream PyTorch code (`tools/golden.py` runs the `laya` SDK,
`tools/golden_kev.py` runs Kev's own code), natively and in the browser.

| | token ids | argmax agreement | max probability difference |
|---|---|---|---|
| Laya, f32 weights | 41 / 41 exact | 41 / 41 | < 0.0001 |
| Laya, int8 pack (CPU, WebAssembly, WebGPU) | 41 / 41 exact | 41 / 41 | 0.024 |
| Kev-0.8B, int8 pack (CPU, WebAssembly, WebGPU) | 8 / 8 requests exact | 13 / 13 | 0.0097 |

The tokenizers also match the Hugging Face `tokenizers` library on the fixture corpora and on about
ten million fuzzed strings. Open [`parity.html`](https://bvolpato.github.io/kevala/parity.html) to rerun the Laya check in
your own browser.

## Demos and examples

- [Playground](https://bvolpato.github.io/kevala/#/playground): any state, any questions, any model
  and backend, with the response as highlighted JSON and the code to reproduce it.
- [Tetris](https://bvolpato.github.io/kevala/#/tetris): the model plays. When a piece appears, code
  lists every spot it can land in and describes each outcome in words; the model scores them all in
  one batched pass, and the piece presses the keys (turn, left, right, drop) toward the best one.
- [Guardrail](https://bvolpato.github.io/kevala/#/guardrail): a prompt-injection and jailbreak gate in
  front of an LLM that acts on confident answers and escalates the unsure ones.
- [Inbox](https://bvolpato.github.io/kevala/#/inbox): triage a mailbox, with rows filling in as each
  batch of answers returns.
- [`examples/`](examples/): complete single-file pages you can copy into a site: a minimal page, a
  comment form that blocks toxic posts, an LLM cascade that escalates only the unsure cases, a game
  loop.
- [`skills/kevala/SKILL.md`](skills/kevala/SKILL.md): teaches a coding agent to add kevala to a site.

## API

```js
import { Kevala, MODELS, cacheInfo, clearCache } from "kevala";

const kevala = await Kevala.load({
  model: "laya",         // a MODELS name, a .kevala URL, or an ArrayBuffer/Blob
  backend: "auto",       // "webgpu", "wasm", or "auto" (WebGPU when available)
  threads: 8,            // WebAssembly workers for the CPU backend
  from: "pack",          // "pack": the pinned int8 pack; "checkpoint": convert the original weights
  cache: true,           // keep the pack in origin storage
  onProgress: (p) => {}, // { phase: download | convert | cache | init | warmup, loaded, total }
  signal,                // AbortSignal
  plugins: [],           // URLs of extra architecture plugins
});

await kevala.decide(state, questions, { parts });  // one request, one forward pass
await kevala.decideMany([{ state, questions }]);   // many states, still one pass
kevala.info;      // { arch, backend, gpu, threads, modalities, model, config, pack, loadMs }
await kevala.profile(true); // later responses carry timing.gpu: milliseconds per GPU kernel
kevala.dispose();
```

Questions use the System One request shape (`type`, `instructions`, `criteria`), and each family
answers in its own reference format: Laya like `laya` 0.3.5 (`action.act_probability`, 4 decimals),
Kev like `kev.serve` (2 decimals, plus `raw_probabilities` at full precision). For starting points,
`import { presets } from "kevala"` has the `laya` SDK's question sets for triage, email,
guardrails, moderation and routing.

Requests may carry typed `parts` (`{ type: "text", text }`; `image` and `audio` for packs whose
`info.modalities` list them). A part the model cannot read is an error, never silently dropped.

Server side, the same engine runs in Node.js, Deno or Bun without workers or a GPU:

```js
import { loadFile } from "kevala/node";
const kevala = await loadFile("laya-q8.kevala");
kevala.decide("I want my money back.", { refund: { type: "noul", instructions: "Does the customer ask for money back?" } });
```

## How it works

```
page ─► index.js ─► engine worker ─► coordinator (Rust → WebAssembly): tokenizer, template, embeddings, head
                         ├─► WebGPU trunk: int8 matmul (split-K), attention, Gated DeltaNet, norms
                         └─► WebAssembly shard workers: tensor-parallel layers, no SharedArrayBuffer needed
```

- **One Rust core, zero crates.** JSON, Unicode tables, byte-level BPE tokenizers, the request
  templates, both model families, the `.kevala` pack format and the checkpoint converters (safetensors,
  LoRA adapters, `torch.save` files) are all in `crates/kevala`, and the same code runs natively for the
  CLI and tests.
- **GPU kernels in the crate too.** WebGPU only runs WGSL, so the kernels are `.wgsl` sources in
  `crates/kevala/src/wgsl`, specialized by Rust (f16 tiles, tile rows for the input length, subgroup
  variants) and served by the WebAssembly binary. The JavaScript only builds pipelines and dispatches.
- **`.kevala` packs** hold int8 weights with a scale per 32 weights, 64-byte aligned, and can be split
  while they stream: the coordinator, the GPU and each CPU shard receive only the bytes they keep.
- **Pluggable families.** A pack's `config.arch` picks a family from a registry in Rust and an
  architecture plugin in JavaScript. Each family owns its template, backbone and head, and declares its
  modalities.

Details: [docs/architecture.md](docs/architecture.md). Packs, and how to convert and publish them:
[docs/packs.md](docs/packs.md).

## Command line and building from source

```sh
scripts/build-wasm.sh                        # js/src/kevala-{relaxed,simd,base}.wasm
cargo build --release -p kevala-cli            # target/release/kevala

kevala convert <laya-checkpoint-dir> -o laya-q8.kevala
kevala convert-kev --base <qwen3.5-dir> --kev <kev-dir> -o kev-0.8b-q8.kevala   # see docs/packs.md
kevala decide laya-q8.kevala --state "..." --questions '{"q": {"type": "noul", "instructions": "..."}}'
kevala parity laya-q8.kevala tests/fixtures/golden.json
kevala bench kev-0.8b-q8.kevala --tokens 128
kevala wgsl matmul --f16 --rows 3                  # a GPU kernel, specialized

node scripts/serve.mjs . --port=8080         # static server for the site and examples
cargo test --release                         # Rust tests (tokenizer, sequence and cache tests skip without their files)
```

Rust 1.85 or newer, and nothing else. Python is only used to generate the reference fixtures.

## Limits

- First visits are heavy: a 479 MB pack for Laya, 857 MB for Kev. Later visits read it from disk.
  You can also convert once with the CLI and serve the pack from your own host.
- Without WebGPU, a request takes about a second on a fast laptop core, more on phones. Laya splits
  across CPU workers; Kev runs in one instance on the CPU for now.
- Browser backgrounding throttles CPU work: benchmarks from a hidden tab are several times slower.
- WebGPU was verified on Apple Silicon in Chromium. Other GPUs and browsers (Safari 26, Firefox) are
  expected to work but were not tested for this release.
- int8 weights move probabilities by up to 0.024 (Laya) and 0.0097 (Kev) on the fixtures, with no
  argmax changes. The models themselves have their own limits: see the
  [Laya](https://huggingface.co/convaiinnovations/laya) and [Kev](https://github.com/jaredpalmer/kev)
  model cards before trusting a threshold.
- Requests can carry image and audio parts, but no shipped pack reads them yet.

## Credits and license

- Laya is by Nandakishor M, Convai Innovations (Apache-2.0).
- Kev is by Jared Palmer (Apache-2.0). Its base, Qwen3.5-0.8B-Base, is by the Qwen team (Apache-2.0).
- The Laya parity fixtures reuse cases from [laya-web](https://github.com/nvkudva/laya-web), and the
  question-writing advice follows [brain function collapse](https://brainfunctioncollapse.com/laya).

kevala is licensed under [Apache-2.0](LICENSE). The weights keep their own licenses; kevala downloads them
from their upstream repositories and does not redistribute them.
