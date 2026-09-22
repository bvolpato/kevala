// How it works: the architecture, headline speed, the fidelity numbers and the honest limits.
// Static; no model needed.

import { css, REPO } from "../ui.js";

const DOCS = `${REPO}/blob/main/docs`;

const TEMPLATE = `<div class="wrap">
  <div class="page-head">
    <div class="eyebrow">How it works</div>
    <h1>From a Hugging Face checkpoint to a decision, all client-side</h1>
    <p>kevala streams the authors' weights at a pinned revision, converts them to int8 in the tab and runs them with a dependency-free Rust core, on WebGPU or on WebAssembly workers. This page shows the path, what it costs and how close it stays to the PyTorch reference.</p>
    <nav class="how-toc" aria-label="On this page">
      <a href="#/how/architecture">Architecture</a><a href="#/how/bench">Speed</a><a href="#/how/fidelity">Fidelity</a><a href="#/how/limits">Honest limits</a><a href="#/how/more">Go deeper</a>
    </nav>
  </div>

  <section class="how-sec first" id="architecture">
    <figure class="arch card">
      <img src="assets/architecture.svg" alt="Diagram: pinned Hugging Face checkpoints stream into an in-browser int8 converter and a cached .kevala pack; the Rust core compiled to WebAssembly runs Laya or Kev on WebGPU or on tensor-parallel WebAssembly workers; your page gets typed answers." width="1120" height="660">
    </figure>
    <div class="grid-4 how-points">
      <div><span class="n">1</span><h3>Rust core</h3><p class="muted small">Tokenizers, kernels and both architectures in dependency-free Rust, compiled to three WebAssembly builds (relaxed-SIMD, SIMD128, baseline). The same binary runs in Node.</p></div>
      <div><span class="n">2</span><h3>Streamed packs</h3><p class="muted small">Weights stream from Hugging Face at a pinned revision and are quantized to int8 on the fly. The pack is stored in the browser (Origin Private File System, Cache API as fallback); nothing is re-hosted.</p></div>
      <div><span class="n">3</span><h3>WebGPU or workers</h3><p class="muted small">On WebGPU, WGSL kernels run every layer of both models. Without it, Laya's layers split across tensor-parallel WebAssembly workers, two exchanges per layer.</p></div>
      <div><span class="n">4</span><h3>Many states, one pass</h3><p class="muted small"><code>decideMany</code> packs many states into a single forward pass, and concurrent calls are batched automatically.</p></div>
    </div>
    <p class="muted small plug">Architectures are pluggable: a families registry in the Rust core and an architecture plugin per model family in <code>js/src/archs</code>, loadable with <code>Kevala.load({ plugins: [url] })</code>. Read <a href="${DOCS}/architecture.md">the architecture notes</a> and <a href="${DOCS}/adding-a-model.md">how to add a model</a>.</p>
  </section>

  <section class="how-sec" id="bench">
    <div class="eyebrow">Speed</div>
    <h2>Milliseconds, in the browser</h2>
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
    <p class="lede">Golden fixtures are generated with each model's own reference code and replayed through kevala. At int8 the argmax never moves; probabilities stay within a few hundredths.</p>
    <div class="grid-4 tiles">
      <div class="tile"><span class="tl">Token ids</span><b>exact</b><span class="ts">both tokenizers, every fixture</span></div>
      <div class="tile"><span class="tl">Laya argmax</span><b>41 / 41</b><span class="ts">questions agree with PyTorch</span></div>
      <div class="tile"><span class="tl">Kev argmax</span><b>13 / 13</b><span class="ts">questions agree with PyTorch</span></div>
      <div class="tile"><span class="tl">Max |Δp| at int8</span><b>0.024 <small>/ 0.0097</small></b><span class="ts">Laya / Kev, worst option</span></div>
    </div>
    <p class="small parity-links"><a href="parity.html">Run the Laya parity check in your browser →</a><a href="parity-kev.html">Kev-0.8B parity check →</a></p>
  </section>

  <section class="how-sec" id="limits">
    <div class="grid-2 limits">
      <div>
        <div class="eyebrow">Honest limits</div>
        <h2>What to expect</h2>
        <p class="muted">kevala is young. These are the edges today.</p>
      </div>
      <ul class="limit-list">
        <li><b>Big first download.</b> Laya fetches about 850 MB once and keeps a 479 MB int8 pack in browser storage; the weights also live in GPU or WebAssembly memory while loaded. Load on demand, not on page view.</li>
        <li><b>Kev is the heavy one.</b> About 1.6 GB and a minute to convert on the first visit, and an 857 MB pack to keep. On WebGPU it is fast; on the CPU fallback it takes seconds per decision.</li>
        <li><b>WebGPU is not everywhere.</b> Without it kevala falls back to WebAssembly: same answers, several times slower (a short Laya request takes about 160 ms on 8 CPU workers, versus tens of ms on WebGPU).</li>
        <li><b>Perception, not reasoning.</b> These are System 1 models: fast, calibrated reads of what a text says. Put the facts in words; don't ask them to do arithmetic or plan.</li>
        <li><b>English, 512 tokens.</b> Laya reads up to 512 tokens per state and is trained on English.</li>
        <li><b>int8 is close, not identical.</b> Probabilities can differ from fp32 by up to 0.024 (Laya). Thresholds near a boundary deserve a margin.</li>
      </ul>
    </div>
  </section>

  <section class="how-sec" id="more">
    <div class="eyebrow">Go deeper</div>
    <h2>Check it yourself</h2>
    <div class="grid-3 more">
      <a class="card pad link-card" href="parity.html"><h3>Laya parity</h3><p class="muted small">Replay the golden fixtures against the PyTorch reference, in this browser.</p><span class="go">parity.html →</span></a>
      <a class="card pad link-card" href="parity-kev.html"><h3>Kev-0.8B parity</h3><p class="muted small">The same check for Kev, against Kev's own PyTorch code.</p><span class="go">parity-kev.html →</span></a>
      <a class="card pad link-card" href="bench.html"><h3>Benchmark your machine</h3><p class="muted small">Time five request shapes, from one short question to 32 states in one pass.</p><span class="go">bench.html →</span></a>
      <a class="card pad link-card" href="${DOCS}/architecture.md"><h3>Architecture notes</h3><p class="muted small">The engine, the pack format, the kernels and the worker layout.</p><span class="go">docs/architecture.md →</span></a>
      <a class="card pad link-card" href="${DOCS}/adding-a-model.md"><h3>Adding a model</h3><p class="muted small">Register a family in the Rust core and write an architecture plugin.</p><span class="go">docs/adding-a-model.md →</span></a>
      <a class="card pad link-card" href="${REPO}"><h3>Source</h3><p class="muted small">The engine, the site and the examples, Apache-2.0.</p><span class="go">github.com/bvolpato/kevala →</span></a>
    </div>
  </section>
</div>`;

export function mount(el) {
  css(new URL("./how.css", import.meta.url).href);
  el.innerHTML = TEMPLATE;
  const scrollTo = (id) => el.querySelector(`#${CSS.escape(id)}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  // the router only scrolls on a hash change; a second click on the same link must too
  el.querySelector(".how-toc").addEventListener("click", (e) => {
    const a = e.target.closest("a");
    if (!a || location.hash !== a.getAttribute("href")) return;
    e.preventDefault();
    scrollTo(a.getAttribute("href").split("/").pop());
  });
  return {};
}
