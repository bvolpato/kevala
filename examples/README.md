# kevala examples

Small, complete pages. Each one imports the current website runtime from
`https://bvolpato.github.io/kevala/js/src/index.js`, loads a model once and calls `decide` or
`decideMany`. Everything it needs is in the file: copy one into your site and it works. No build
step and no helper dependencies.

Each page accepts `?model=<model-id-or-kevala-url>&backend=<auto|webgpu|wasm>`. The model defaults to
`laya` and the backend defaults to `auto` (WebGPU when available, then WebAssembly). Known model
IDs include the catalog entries below; a custom `.kevala` URL can be passed as the `model` value.
The page status or log shows the selected model. The examples use the website runtime because the
current `kevala@latest` jsDelivr package does not yet include the newer model families.

## Run them

Open a file from any static server (a `file://` page cannot start the workers):

```sh
pnpm install --frozen-lockfile
pnpm serve
open http://127.0.0.1:8080/examples/basic.html
```

With no query parameters, the examples use `model=laya` and `backend=auto`. First visits download
the selected model's pinned pack (or convert a supported checkpoint) and cache it in the browser.

The model menu and the API also expose `kev-0.8b`, `kev-4b`, `kev-9b`,
`semif-qwen3.5-0.8b`, `semif-qwen3.5-2b`, `semif-qwen3.5-4b`, `gemma-4-e2b`, and `gemma-4-e4b`.
SemIf uses frozen Qwen3.5 instruction weights and direct option scoring, with no trained adapter.

## The examples

| File | What it shows |
| --- | --- |
| [`basic.html`](basic.html) | The minimum: `Kevala.load()` with progress, then one `decide()` with a `noul`, a `choice` and a `score` question, and the raw response. |
| [`moderation-form.html`](moderation-form.html) | A comment form that checks the text on submit with three moderation questions (toxic, harassment, threat), written out in the file. `P >= 0.75` blocks, `0.45 to 0.75` holds for review, lower posts. Thresholds are constants at the top. |
| [`llm-cascade.html`](llm-cascade.html) | A support widget. kevala classifies the intent; if it is confident (`P >= 0.9` and an optional `act_probability >= 0.5`) and a canned answer exists, it answers at once; otherwise it calls a slow fake `askLLM()` you would replace with your API. Counts how many calls were saved. |
| [`game-loop.html`](game-loop.html) | A ship dodging rocks. Each tick the code describes every move in words ("A rock is close, two rows above the ship") and asks one `noul` question for all moves in one `decideMany`; the ship takes the highest P(safe). At most one request is in flight, so the game never waits. |

Writing good questions, in short: ask what the text *says* (perception), not what to do. Compute
numbers and comparisons in code and state them in words. Describe every option in `criteria`.

## Response shapes

```js
const r = await kevala.decide(state, questions);
r.answers.urgent.noul;               // noul: P(true)
r.answers.team.choice;               // choice: the argmax key, plus r.answers.team.probabilities
r.answers.tone.score;                // score: expected level, plus .legend and .probabilities
r.answers.team.action.act_probability; // Laya only: its estimate that acting on the answer is safe
r.timing;                            // { forward, total, tokens, batched } in ms
```

Kev, SemIf, and Gemma 4 follow the decoder response format (answers rounded to 2 places and
`raw_probabilities` at full precision, with no Laya `action`). SemIf also reports that its scores
are conditional on the listed options and are not calibrated decision confidence. Code that must
work with all model families should treat `action` and `probability_status` as optional, as
`llm-cascade.html` does.

## Self-host a .kevala pack

Converting in the browser is convenient, but for production you may prefer to convert once and
serve the pack from your own CDN. Build the CLI and use the common architecture-driven command:

```sh
cargo build --release -p kevala-cli

# Laya: a local snapshot of convaiinnovations/laya (model.safetensors, encoder/config.json,
# rl_agent_config.json, tokenizer/tokenizer.json)
huggingface-cli download convaiinnovations/laya --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --local-dir laya
target/release/kevala convert laya -o laya-q8.kevala

# Kev-0.8B: the Qwen3.5-0.8B base plus Kev's adapter and pointer head
huggingface-cli download Qwen/Qwen3.5-0.8B-Base --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --local-dir qwen35
huggingface-cli download jaredpalmer/kev-0.8b --revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --local-dir kev
target/release/kevala convert qwen35 --adapter kev -o kev-0.8b-q8.kevala \
  --revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68

# The adapter-first spelling is equivalent:
target/release/kevala convert kev --base qwen35 -o kev-0.8b-q8.kevala \
  --revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68

# A plain Qwen3.5 or Gemma 4 text checkpoint uses direct option scoring by default:
target/release/kevala convert qwen35-instruction -o semif-qwen3.5-q8.kevala --readout direct-options

# check a pack
target/release/kevala inspect laya-q8.kevala
target/release/kevala decide laya-q8.kevala --state "Refund me today or I cancel" \
  --questions '{"churn":{"type":"noul","instructions":"Does the customer threaten to leave?"}}'
```

`convert-kev`, `convert-semif`, and `convert-gemma` remain compatibility aliases for the same
validated path. The native converter applies the Qwen2Tokenizer normalization and added-token
overlay from the checkpoint configuration, so ordinary conversion needs no Python tokenizer
materialization; `--tokenizer` is available for an explicit verified override.

Then load it by URL:

```js
const kevala = await Kevala.load({ model: "https://cdn.example.com/models/laya-q8.kevala" });
```

A relative URL resolves against the page, like any other link.

The pack is streamed and stored in the browser (Origin Private File System, Cache API as a
fallback) under its URL, so give each pack version its own URL (for example `laya-q8-1c5edc1.kevala`).

### Headers

| Header | Value | Why |
| --- | --- | --- |
| `Content-Type` | `application/octet-stream` | Any binary type works; this avoids transforms. |
| `Content-Length` | the file size | kevala reports byte progress from it. |
| `Content-Encoding` | none (serve it uncompressed) | int8 weights barely compress, and on-the-fly gzip usually drops `Content-Length`. |
| `Cache-Control` | `public, max-age=31536000, immutable` | Safe when the URL changes with the pack. |
| `Access-Control-Allow-Origin` | your site's origin (or `*`) | Needed when the pack is on another origin. |
| `Accept-Ranges` | `bytes` | Optional; lets tools probe the header without a full download. |

Also serve the `.wasm` files next to `index.js` as `application/wasm` so browsers can compile them
while they download. No `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers are
needed: kevala does not use `SharedArrayBuffer`.

nginx:

```nginx
location ~ \.kevala$ {
  types { } default_type application/octet-stream;
  gzip off;
  add_header Cache-Control "public, max-age=31536000, immutable";
  add_header Access-Control-Allow-Origin "https://your-site.example";
}
location ~ \.wasm$ { types { application/wasm wasm; } }
```

Caddy:

```caddy
@packs path *.kevala
header @packs Cache-Control "public, max-age=31536000, immutable"
header @packs Access-Control-Allow-Origin "https://your-site.example"
```

For S3, R2 or GCS buckets, set the object's content type to `application/octet-stream`, do not
enable compression for `.kevala`, and add a CORS rule that allows `GET` from your origin.

## Node.js, Deno and Bun

The same WebAssembly engine runs on a server or in a script, in one instance, without workers or a
GPU. It reads a pack converted by the CLI (or one exported from a browser's storage), including
Kev and SemIf packs.

```sh
pnpm add kevala
```

```js
import { loadFile } from "kevala/node";

const kevala = await loadFile("laya-q8.kevala");
const r = kevala.decide("I was charged twice, refund one of them.", {
  refund: { type: "noul", instructions: "Does the customer ask for money back?" },
});
console.log(r.answers.refund.noul);

// many states in one pass
const rs = kevala.decideMany(tickets.map((state) => ({ state, questions })));
```

`decide` is synchronous here and returns the same response shape as in the browser. One core
answers a short Laya request in about 0.4 s; `pnpm exec node scripts/bench-node.mjs <pack>` times your machine.

## Licenses

Laya (Nandakishor M, Convai Innovations), Kev-0.8B, Kev-4B, and Kev-9B (Jared Palmer), and their
Qwen3.5 base models are Apache-2.0. SemIf's direct option scoring method is MIT; its Qwen3.5
instruction weights remain under Qwen's Apache-2.0 license. If you self-host packs, keep the
license and attribution with them.
