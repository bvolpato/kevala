// How it works: the architecture, latency, the fidelity numbers and the limits. Static; no model
// needed.

import { css, REPO } from "../ui.js";

const DOCS = `${REPO}/blob/main/docs`;

const ARCH_ALT =
  "Diagram: pinned Hugging Face checkpoints stream into an in-browser int8 converter and a cached .kevala pack; " +
  "the Rust core compiled to WebAssembly runs Laya, Kev, SemIf, or Gemma 4 on WebGPU; Laya can also use tensor-parallel WebAssembly workers; " +
  "your page gets typed answers.";

const TEMPLATE = `<div class="wrap">
  <div class="page-head">
    <div class="eyebrow">How it works</div>
    <h1>From a Hugging Face checkpoint to an answer, in the browser</h1>
    <p>kevala streams the authors' weights at a pinned revision, converts supported checkpoints to int8 in the tab and runs Laya, Kev, SemIf, or Gemma 4 with a dependency-free Rust core. WebGPU runs the browser path; native 64-bit CPU support covers larger packs, while the large Gemma packs require WebGPU in the browser or native CPU. Laya can split its CPU layers across workers; Kev and SemIf use one CPU instance when that path is available. This page covers each step, the latency, and how close the results stay to the PyTorch reference.</p>
    <nav class="how-toc" aria-label="On this page">
      <a href="#/how/architecture">Architecture</a>
      <a href="#/how/bench">Speed</a>
      <a href="#/how/fidelity">Fidelity</a>
      <a href="#/how/limits">Limits</a>
      <a href="#/how/more">More</a>
    </nav>
  </div>

  <section class="how-sec first" id="architecture">
    <figure class="arch card">
      <img src="assets/architecture.svg" alt="${ARCH_ALT}" width="1120" height="660">
    </figure>
    <div class="grid-4 how-points">
      <div>
        <span class="n">1</span><h3>Rust core</h3>
        <p class="muted small">Tokenizers, kernels and the model families in dependency-free Rust, compiled to three WebAssembly builds (relaxed-SIMD, SIMD128, baseline). The same binary runs in Node.</p>
      </div>
      <div>
        <span class="n">2</span><h3>Streamed packs</h3>
        <p class="muted small">The int8 pack streams from Hugging Face at a pinned commit, six byte ranges at a time, and is stored in the browser (Origin Private File System, with the Cache API as a fallback). Laya and Kev-0.8B can also quantize their original checkpoints in the browser when the pack is unreachable; larger Kev, all SemIf choices, and Gemma 4 use converted packs.</p>
      </div>
      <div>
        <span class="n">3</span><h3>WebGPU or workers</h3>
        <p class="muted small">On WebGPU, WGSL kernels run every transformer layer of Laya, Kev, SemIf, and Gemma 4. Without it, Laya's layers split across tensor-parallel WebAssembly workers, with two exchanges per layer; Kev and SemIf run in one CPU instance. Gemma 4 needs WebGPU in the browser or the native 64-bit CPU CLI.</p>
      </div>
      <div>
        <span class="n">4</span><h3>Batching</h3>
        <p class="muted small"><code>decideMany</code> packs many states into a single forward pass, and concurrent calls are batched automatically.</p>
      </div>
    </div>
    <p class="muted small plug">Architectures are pluggable: a families registry in the Rust core and an architecture plugin per model family in <code>js/src/archs</code>, loadable with <code>Kevala.load({ plugins: [url] })</code>. Read <a href="${DOCS}/architecture.md">the architecture notes</a>, <a href="${DOCS}/gemma4.md">the Gemma 4 notes</a>, and <a href="${DOCS}/adding-a-model.md">how to add a model</a>.</p>
  </section>

  <section class="how-sec" id="bench">
    <div class="eyebrow">Speed</div>
    <h2>Latency</h2>
    <p class="muted">Apple M4 Max, WebGPU in Chrome, cold requests unless noted.</p>
    <div class="grid-4 tiles">
      <div class="tile"><span class="tl">Laya, short request</span><b>11 ms</b><span class="ts">one question, about 45 tokens</span></div>
      <div class="tile"><span class="tl">Kev-0.8B, short request</span><b>13 ms</b><span class="ts">one question, about 30 tokens</span></div>
      <div class="tile"><span class="tl">Kev-0.8B, cached state</span><b>10 ms</b><span class="ts">the same state asked again</span></div>
      <div class="tile"><span class="tl">32 states, one pass</span><b>186 ms</b><span class="ts">Laya, about 1,500 tokens</span></div>
    </div>
    <p class="muted small">Without WebGPU the same short request takes about 0.4 s on one CPU core. <a href="bench.html">Measure your own machine →</a></p>
  </section>

  <section class="how-sec" id="fidelity">
    <div class="eyebrow">Fidelity</div>
    <h2>Same answers as the PyTorch reference</h2>
    <p class="lede">Golden fixtures are generated with each model's own reference code and replayed through kevala. At int8 the argmax matches on every checked Laya, Kev, SemIf, and Gemma 4 fixture; the score differences below are fixture fidelity measurements, not general decision quality or calibration.</p>
    <div class="grid-4 tiles">
      <div class="tile"><span class="tl">Token ids</span><b>tracked</b><span class="ts">Laya, Qwen, and Gemma reference tokenizers used by the fixtures</span></div>
      <div class="tile"><span class="tl">Laya argmax</span><b>41 / 41</b><span class="ts">questions agree with PyTorch</span></div>
      <div class="tile"><span class="tl">Kev argmax</span><b>31 / 31</b><span class="ts">0.8B, 4B, and 9B fixtures</span></div>
      <div class="tile"><span class="tl">SemIf argmax</span><b>36 / 36</b><span class="ts">three sizes, 12 questions each</span></div>
      <div class="tile"><span class="tl">Gemma E2B GPU</span><b>12 / 12</b><span class="ts">max |Δp| 0.01906186</span></div>
      <div class="tile"><span class="tl">Gemma E4B GPU</span><b>12 / 12</b><span class="ts">max |Δp| 0.0120864</span></div>
      <div class="tile"><span class="tl">Gemma E2B native</span><b>12 / 12</b><span class="ts">max |Δp| 0.019120</span></div>
      <div class="tile"><span class="tl">Max |Δp| at int8 (7 models)</span><b>≈0.031</b><span class="ts">Laya, Kev, and SemIf · <a href="${DOCS}/semif-matmul.md">GPU report</a> · <a href="${DOCS}/gemma4.md#verification">Gemma checks</a></span></div>
    </div>
    <p class="small parity-links">
      <a href="parity.html">Replay the Laya fixtures in your browser →</a>
      <a href="parity-kev.html">Replay the Kev and SemIf fixtures →</a>
    </p>
  </section>

  <section class="how-sec" id="limits">
    <div class="grid-2 limits">
      <div>
        <div class="eyebrow">Limits</div>
        <h2>Known limits</h2>
        <p class="muted">kevala is new. These are its limits today.</p>
      </div>
      <ul class="limit-list">
        <li><b>Large first download.</b> Laya downloads a 479 MB int8 pack and keeps it in browser storage. The weights also occupy GPU or WebAssembly memory while loaded.</li>
        <li><b>Larger sizes need more memory.</b> Kev packs range from 857 MB to 8.96 GB; SemIf Qwen3.5 packs range from 855 MB to 4.75 GB; Gemma 4 packs are 5.22 GB (E2B) or 8.41 GB (E4B). Runtime memory is higher. Kev and SemIf sliders start at 0.8B, Gemma 4 starts at E2B, and loading begins when you press Load.</li>
        <li><b>WebGPU is not available in every browser.</b> For compatible packs, kevala falls back to WebAssembly, which gives the same answers several times slower: a short Laya request takes about 160 ms on 8 CPU workers, against tens of ms on WebGPU. Large Gemma packs require WebGPU in the browser or native 64-bit CPU.</li>
        <li><b>No multi-step reasoning.</b> These are System 1 models: they score or read what a text says in one pass. SemIf scores are conditional on the listed options and are not calibrated decision confidence. Compute facts in code and state them in words, and do not ask the model to do arithmetic or plan.</li>
        <li><b>Context limits.</b> Laya reads up to 512 tokens per state and is trained on English. Gemma 4 accepts up to 4096 input tokens in Kevala.</li>
        <li><b>int8 shifts probabilities slightly.</b> They can differ from fp32 by up to 0.024 (Laya), so leave a margin around thresholds near a decision boundary.</li>
      </ul>
    </div>
  </section>

  <section class="how-sec" id="more">
    <div class="eyebrow">More</div>
    <h2>Read further</h2>
    <div class="grid-3 more">
      <a class="card pad link-card" href="${DOCS}/packs.md">
        <h3>Packs</h3>
        <p class="muted small">Load an int8 pack or, for Laya and Kev-0.8B, original weights; then convert, check and publish packs. Gemma 4 uses text-only converted packs.</p>
        <span class="go">docs/packs.md →</span>
      </a>
      <a class="card pad link-card" href="bench.html">
        <h3>Benchmark your machine</h3>
        <p class="muted small">Time five request shapes, from one short question to 32 states in one pass.</p>
        <span class="go">bench.html →</span>
      </a>
      <a class="card pad link-card" href="${DOCS}/architecture.md">
        <h3>Architecture notes</h3>
        <p class="muted small">The engine, the pack format, the kernels and the worker layout.</p>
        <span class="go">docs/architecture.md →</span>
      </a>
      <a class="card pad link-card" href="${DOCS}/adding-a-model.md">
        <h3>Adding a model</h3>
        <p class="muted small">Register a family in the Rust core and write an architecture plugin.</p>
        <span class="go">docs/adding-a-model.md →</span>
      </a>
      <a class="card pad link-card" href="${REPO}">
        <h3>Source</h3>
        <p class="muted small">The engine, the site and the examples, under Apache-2.0.</p>
        <span class="go">github.com/bvolpato/kevala →</span>
      </a>
    </div>
  </section>
</div>`;

export function mount(el) {
  css(new URL("./how.css", import.meta.url).href);
  el.innerHTML = TEMPLATE;
  const scrollToSection = (id) => el.querySelector(`#${CSS.escape(id)}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  // the router only scrolls on a hash change; a second click on the same link must too
  el.querySelector(".how-toc").addEventListener("click", (e) => {
    const link = e.target.closest("a");
    if (!link || location.hash !== link.getAttribute("href")) return;
    e.preventDefault();
    scrollToSection(link.getAttribute("href").split("/").pop());
  });
  return {};
}
