---
name: kevala
description: Add fast, private, typed decisions to a web page with kevala, which runs System 1 decision models (Laya, Kev-0.8B) in the browser on WebGPU or WebAssembly, with no server and no dependencies. Use when a site needs to classify, route, score, moderate, or gate text on the client (comment moderation, prompt-injection guards, ticket triage, form spam, game AI that reads a described state), or when the user mentions kevala, Laya, Kev, typed decisions, or a System One API in the browser.
---

# Integrating kevala

kevala answers typed questions about a piece of text inside the page. You pass a `state` (text or JSON)
and a dict of questions; it returns a probability for every option of every question in one forward
pass. It never generates text. Weights download once from Hugging Face as an int8 pack and stay in
the site's origin storage; nothing is sent anywhere.

Reach for it when the page needs a decision (which queue, is this toxic, how urgent, should this
prompt be blocked). Do not use it for free-form answers, extraction, arithmetic or multi-step
reasoning; keep an LLM for those and put kevala in front as a cheap first pass.

## Load once, reuse

Import from the CDN in a plain page, or `npm install kevala` and `import { Kevala } from "kevala"`
with a bundler.

```js
import { Kevala } from "https://cdn.jsdelivr.net/npm/kevala@latest/js/src/index.js";

const kevala = await Kevala.load({
  model: "laya",               // or "kev-0.8b", or the URL of a .kevala pack you host
  onProgress: (p) => show(p),  // {phase, file, loaded, total}: download, convert, cache, init, warmup
});
```

- First visit: Laya downloads a 479 MB int8 pack from Hugging Face, Kev-0.8B an 857 MB one.
  `from: "checkpoint"` downloads the original weights instead (843 MB / 1.6 GB) and converts them in
  the browser. Later visits load from disk in under a second. Always show progress and
  load on a user action (a button), never on page load for every visitor.
- `kevala.info.backend` is `webgpu` or `wasm-*`. WebGPU is 10-50x faster; the WebAssembly fallback works
  everywhere but a request can take seconds.
- Call `kevala.dispose()` when the feature goes away.

## Ask

```js
const r = await kevala.decide("Refund the duplicate charge today or we cancel.", {
  team:  { type: "choice", instructions: "Which team should handle this?",
           criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages", other: "anything else" } },
  churn: { type: "noul", instructions: "Does the customer threaten to leave?" },
  urgency: { type: "score", instructions: "How urgent is this?", criteria: ["can wait", "soon", "today"] },
});
r.answers.team.choice;               // "billing"
r.answers.team.probabilities.billing; // 0.97
r.answers.churn.noul;                // P(true)
```

Ask everything about one state in one `decide` call. For many states (candidate moves, a list of
emails), use `decideMany([{ state, questions }, ...])`: one forward pass for all of them. Concurrent
`decide` calls are also packed together automatically.

Responses follow each model's reference format: Laya rounds to 4 places and adds `confidence` and
`action.act_probability`; Kev rounds to 2 places and adds `raw_probabilities` at full precision.

## Questions that work

- Ask what the text says, not what to do about it. "Where is the piece relative to the gap?" works;
  "Which way should it move?" often inverts.
- Put the state into words. Compute comparisons, thresholds and counts in code and state the
  conclusion ("the stack has two holes on the left").
- Describe every option: `{ billing: "invoices, payments, refunds" }` beats `["billing"]`.
- Keep option lists short and the state front-loaded (Laya reads 512 tokens in total).
- Try two or three phrasings on real examples and keep the best.

## Wire it into the page

- Keep one request in flight and let the latest input win: when the user types, debounce
  (about 200 ms on WebGPU, a second on the CPU) and drop answers for text that has changed.
- For a list (emails, comments, candidate moves), call `decideMany` in growing batches (1, 2, 4,
  8, ...) and render each batch as it returns, so the first rows appear after one short pass.
- `Kevala.load` rejects when the model cannot load (no storage, a failed download); show the error
  and keep the page usable. Pass `signal` from an `AbortController` to cancel a download.

## Act on probabilities

Pick thresholds from labelled examples, not 0.5 by default, and send the unsure middle somewhere
slower (a person, an LLM). Probabilities are the product; accuracy out of the box is modest on nuanced
or graded questions. Measure on 50-200 of the site's own examples and report the numbers.

## Server side

The same engine runs in Node.js, Deno or Bun, in one instance, without workers or a GPU. It reads a
pack file: download one from [bvolpato/kevala-packs](https://huggingface.co/bvolpato/kevala-packs)
(`hf download bvolpato/kevala-packs laya-q8.kevala`) or convert one with the CLI.

```js
import { loadFile } from "kevala/node"; // npm install kevala
const kevala = await loadFile("laya-q8.kevala");
const r = kevala.decide("I want my money back.", { refund: { type: "noul", instructions: "Does the customer ask for money back?" } });
```

## Hosting

- Pages must be served over HTTPS or localhost (storage and WebGPU need a secure context).
- No special headers are needed. Cross-origin isolation is not required.
- To serve the weights from your own host, put the `.kevala` file from bvolpato/kevala-packs (or one
  made with `kevala convert` / `kevala convert-kev`) behind `Content-Length` and byte ranges, and pass
  its URL as `model`.

## Examples to copy

Complete single-file pages: https://github.com/bvolpato/kevala/tree/main/examples (a minimal page,
a moderation gate on a comment form, an LLM cascade that answers confident cases locally, a game
loop). Live demos: https://bvolpato.github.io/kevala/
