// How it works: the architecture, latency, the fidelity numbers and the limits. Static; no model
// needed.

import { css, DOCS, REPO } from "../ui.js";


const ARCH_ALT =
  "Diagram: pinned Hugging Face checkpoints convert to a BF16 or Q8 .kevala pack; " +
  "the Rust core compiled to WebAssembly runs Laya, Kev, H2O-Lightning, Clef-flash, SemIf, or Gemma 4 on WebGPU; Laya can also use tensor-parallel WebAssembly workers; " +
  "your page gets typed answers.";

const TEMPLATE = `<div class="wrap">
  <div class="page-head">
    <div class="eyebrow">How it works</div>
    <h1>From a Hugging Face checkpoint to an answer, in the browser</h1>
    <p>kevala loads pinned Q8 packs or converts supported checkpoints to BF16/Q8, then runs Laya, Kev, H2O-Lightning, Clef-flash, SemIf, or Gemma 4 with a dependency-free Rust core. Large BF16 packs need WebGPU in the browser or the native 64-bit CPU CLI. The current browser converter handles Laya and Kev-0.8B; larger checkpoints need native conversion. This page covers each step, the latency, and how closely the verified packs match their references.</p>
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
        <p class="muted small">A published pack streams from Hugging Face at a pinned commit, six byte ranges at a time, and is stored in browser storage. Refreshed Kev defaults to BF16; Q8 is optional. Laya and Kev-0.8B can convert their original checkpoints in the browser. Kev-4B/9B, H2O-Lightning, and Clef-flash need native conversion until their full output fits the browser converter.</p>
      </div>
      <div>
        <span class="n">3</span><h3>WebGPU or workers</h3>
        <p class="muted small">On WebGPU, WGSL kernels run every supported transformer layer. Without it, Laya's layers split across tensor-parallel WebAssembly workers; smaller Kev packs can use one CPU instance. Packs over the WebAssembly allocation limit need WebGPU in the browser or the native 64-bit CPU CLI.</p>
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
    <p class="muted">For decision quality and NVIDIA WebGPU measurements across the previously validated packs, read the <a href="${REPO}/blob/main/BENCHMARK.md">decision benchmark report</a>. It does not measure the refreshed Kev, H2O-Lightning, or Clef-flash checkpoints yet.</p>
    <p class="muted">Earlier Q8 packs on Apple M4 Max, WebGPU in Chrome, cold requests unless noted.</p>
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
    <h2>Conversion fidelity</h2>
    <p class="lede">The refreshed Kev, H2O-Lightning and Clef-flash implementations match all decisions in their checked upstream fixtures. The results below measure agreement with pinned reference outputs. General decision accuracy and confidence calibration need evaluation on representative data.</p>
    <div class="reference-table" role="region" aria-label="Current model reference checks" tabindex="0">
      <table class="data">
        <caption>Reference checks · October 12, 2026</caption>
        <thead><tr><th scope="col">Model</th><th scope="col">Precision</th><th scope="col">Backend</th><th scope="col" class="num">Decisions matched</th><th scope="col" class="num">Max probability difference</th></tr></thead>
        <tbody>
          <tr><th scope="row">Kev v1.0 0.8B</th><td>BF16 · default</td><td>WebGPU</td><td class="num">13 / 13</td><td class="num">0.002812</td></tr>
          <tr><th scope="row">Kev v1.0 0.8B</th><td>Q8 · optional</td><td>WebGPU</td><td class="num">13 / 13</td><td class="num">0.006653</td></tr>
          <tr><th scope="row">H2O-Lightning-4B</th><td>BF16 · default</td><td>WebGPU</td><td class="num">3 / 3</td><td class="num">0.004328</td></tr>
          <tr><th scope="row">H2O-Lightning-4B</th><td>Q8 · optional</td><td>WebGPU</td><td class="num">3 / 3</td><td class="num">0.007322</td></tr>
          <tr><th scope="row">Cloudflare Clef-flash</th><td>BF16 · default</td><td>Native CPU, streamed layers</td><td class="num">5 / 5</td><td class="num">0.009297</td></tr>
          <tr><th scope="row">Cloudflare Clef-flash</th><td>Q8 · optional</td><td>Native CPU</td><td class="num">5 / 5</td><td class="num">0.028130</td></tr>
          <tr><th scope="row">Cloudflare Clef-flash</th><td>Q8 · optional</td><td>WebGPU</td><td class="num">5 / 5</td><td class="num">0.028167</td></tr>
        </tbody>
      </table>
    </div>
    <p class="muted small">Kev compares against upstream F32 outputs; H2O and Clef compare against upstream BF16 backbones with their original readouts. Kev includes browser checkpoint conversion. Clef's browser probabilities are rounded to four decimals. <a href="${DOCS}/models.md#conversion-and-validation">Reference details and memory requirements →</a></p>
    <h3>Earlier Q8 results</h3>
    <p class="muted small">These measurements cover previously validated packs. The broader <a href="${REPO}/blob/main/BENCHMARK.md">decision benchmark</a> found one changed E2B answer across 108 option orders, with a material probability shift.</p>
    <div class="grid-4 tiles">
      <div class="tile"><span class="tl">Token ids</span><b>tracked</b><span class="ts">Laya, Qwen, and Gemma reference tokenizers used by the fixtures</span></div>
      <div class="tile"><span class="tl">Laya argmax</span><b>41 / 41</b><span class="ts">questions agree with PyTorch</span></div>
      <div class="tile"><span class="tl">Kev argmax</span><b>31 / 31</b><span class="ts">0.8B, 4B, and 9B fixtures</span></div>
      <div class="tile"><span class="tl">SemIf argmax</span><b>36 / 36</b><span class="ts">three sizes, 12 questions each</span></div>
      <div class="tile"><span class="tl">Gemma E2B GPU</span><b>12 / 12</b><span class="ts">max |Δp| 0.01906186</span></div>
      <div class="tile"><span class="tl">Gemma E4B GPU</span><b>12 / 12</b><span class="ts">max |Δp| 0.0120864</span></div>
      <div class="tile"><span class="tl">Gemma E2B native</span><b>12 / 12</b><span class="ts">max |Δp| 0.019120</span></div>
      <div class="tile"><span class="tl">Max |Δp| at Q8 (7 older packs)</span><b>≈0.031</b><span class="ts">Previously validated Laya, Kev, and SemIf packs · <a href="${DOCS}/semif-matmul.md">GPU report</a> · <a href="${DOCS}/gemma4.md#verification">Gemma checks</a></span></div>
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
        <li><b>Large first download.</b> Laya downloads a 479 MB Q8 pack. Refreshed Kev, H2O-Lightning, and Clef-flash default to much larger BF16 packs; Q8 is available as an explicit smaller option. Runtime memory is higher.</li>
        <li><b>Large models need more memory.</b> Kev v1.0 BF16 outputs range from 1.51 GB to 15.90 GB. H2O-Lightning-4B is 8.43 GB; Clef-flash is 18.41 GB. Existing SemIf and Gemma 4 packs retain their published Q8 sizes. Kev and SemIf sliders start at their smallest size, and loading begins when you press Load.</li>
        <li><b>WebGPU is not available in every browser.</b> For compatible packs, kevala falls back to WebAssembly, which gives the same answers several times slower. Packs above the under-2-GiB WebAssembly allocation limit require WebGPU in the browser or native 64-bit CPU.</li>
        <li><b>No multi-step reasoning.</b> These are System 1 models: they score or read what a text says in one pass. SemIf scores are conditional on the listed options and are not calibrated decision confidence. Compute facts in code and state them in words, and do not ask the model to do arithmetic or plan.</li>
        <li><b>Context limits.</b> Laya reads up to 512 tokens per state and is trained on English. Gemma 4 accepts up to 4096 input tokens in Kevala.</li>
        <li><b>Quantization changes probabilities.</b> Differences depend on the model and input. The decision benchmark includes an E2B difference of 0.53 against BF16 on one case. Validate decisions and thresholds on your own data; see the <a href="${DOCS}/quantization.md">precision comparison</a>.</li>
      </ul>
    </div>
  </section>

  <section class="how-sec" id="more">
    <div class="eyebrow">More</div>
    <h2>Read further</h2>
    <div class="grid-3 more">
      <a class="card pad link-card" href="${DOCS}/packs.md">
        <h3>Packs</h3>
        <p class="muted small">Load a published Q8 pack or convert pinned checkpoints to BF16 by default. Choose Q8 explicitly for smaller packs. Large models use native conversion.</p>
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
