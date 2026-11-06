const SAMPLES = 9;
const WARMUPS = 3;
const MAX_COORDINATOR_BYTES = 640 * 1024 ** 2;
const MAX_INPUT_BYTES = 64 * 1024 ** 2;
const MAX_CODE_BYTES = 64 * 1024 ** 2;
const PACK_BYTES = 10586832640;
const HEADER_BYTES = 53994;
const HEADER_SHA256 = "c09bba078c04d4053f4791cc557bebda7038250faf030d4153c5a96ffd7fcb19";
const SCOPE = "CPU finish only, not end-to-end speedup";
const enc = new TextEncoder();
const check = (condition, message) => { if (!condition) throw new Error(message); };
const hex = (bytes) => [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
const sha256 = async (bytes) => hex(await crypto.subtle.digest("SHA-256", bytes));
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

export function optionsFor(href) {
  const page = new URL(href), q = page.searchParams;
  const sameOrigin = (value) => {
    const url = new URL(value, page);
    check(url.origin === page.origin && ["http:", "https:"].includes(url.protocol) && !url.username && !url.password && !url.hash, "benchmark resources must be same-origin HTTP(S) URLs without credentials or fragments");
    return url;
  };
  const runtime = sameOrigin(q.get("runtime") || "../js/src/index.js");
  check(runtime.pathname.endsWith("/index.js") && !runtime.search, "runtime must be an index.js URL without a query");
  const packSha256 = q.get("packSha256");
  check(/^[a-f0-9]{64}$/.test(packSha256 || ""), "packSha256 must be the verified offline full-pack SHA-256");
  const tile = q.has("tile") ? Number(q.get("tile")) : 0;
  check(tile === 0 || tile === 1, "tile must be 0 (2x4) or 1 (4x4)");
  const flavor = q.get("flavor") || undefined;
  check(flavor === undefined || ["base", "simd", "relaxed"].includes(flavor), "flavor must be base, simd or relaxed");
  const baseline = sameOrigin(q.get("baseline") || "../tmp/baseline-runtime/");
  if (!baseline.pathname.endsWith("/")) baseline.pathname += "/";
  check(!baseline.search, "baseline directory must not have a query");
  return { runtime, currentBase: new URL("./", runtime), baseline, pack: sameOrigin(q.get("pack") || "../tmp/clef-flash-q8.kevala"), packSha256, tile, flavor };
}

export function timingSummary(samples) {
  check(samples.length > 0 && samples.every((ms) => typeof ms === "number" && Number.isFinite(ms) && ms > 0), "CPU finish samples must be positive finite milliseconds");
  const sorted = [...samples].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return { medianMs: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1], samples };
}

async function resource(url) {
  const started = performance.now(), response = await fetch(url, { cache: "no-store" });
  check(response.ok, `${url}: HTTP ${response.status}`);
  const bytes = await codeBytes(response);
  return { bytes, identity: { url: url.href, bytes: bytes.length, sha256: await sha256(bytes), fetchMs: performance.now() - started } };
}

async function codeBytes(response) {
  const length = Number(response.headers.get("content-length"));
  check(Number.isSafeInteger(length) && length > 0 && length <= MAX_CODE_BYTES, "code resource requires a bounded Content-Length of at most 64 MiB");
  const bytes = new Uint8Array(await response.arrayBuffer());
  check(bytes.length === length, "code resource length changed while reading");
  return bytes;
}

async function javascriptIdentity(entry) {
  // pageEngine uses a computed import URL, so include its source explicitly.
  const pending = [entry, new URL("./engine-worker.js", entry)], visited = new Set(), assets = [];
  let totalBytes = 0;
  while (pending.length) {
    const url = pending.pop();
    if (visited.has(url.href)) continue;
    visited.add(url.href);
    const result = await resource(url);
    totalBytes += result.bytes.length;
    check(visited.size <= 256 && totalBytes <= 32 * 1024 ** 2, "JavaScript identity graph exceeds its bounded size");
    assets.push({ url: url.href, bytes: result.bytes.length, sha256: result.identity.sha256 });
    const source = new TextDecoder().decode(result.bytes);
    for (const match of source.matchAll(/\b(?:from\s+|import\s*\(?\s*)["'](\.[^"']+)["']/g)) {
      const imported = new URL(match[1], url);
      if (imported.origin === entry.origin && imported.pathname.endsWith(".js") && !imported.search) pending.push(imported);
    }
  }
  return assets.sort((a, b) => a.url.localeCompare(b.url));
}

async function packIdentity(options) {
  const response = await fetch(options.pack, { headers: { Range: `bytes=0-${HEADER_BYTES - 1}` }, cache: "no-store" });
  check(response.status === 206 && response.headers.get("content-range") === `bytes 0-${HEADER_BYTES - 1}/${PACK_BYTES}`, "pack server must provide the verified full-file length and bounded header range");
  const bytes = new Uint8Array(await response.arrayBuffer());
  check(bytes.length === HEADER_BYTES && await sha256(bytes) === HEADER_SHA256, "served Clef Q8 pack header differs from the verified offline pack");
  return { url: options.pack.href, bytes: PACK_BYTES, sha256: options.packSha256, headerBytes: HEADER_BYTES, headerSha256: HEADER_SHA256,
    fullHashVerification: "controller-supplied offline full-file SHA-256; browser verifies header and length, not the entire multi-GB file" };
}

function captureExports(wasm, snapshot, options) {
  const original = wasm.x;
  wasm.x = { ...original,
    kevala_engine_load(ptr, bytes) {
      try {
        check(!snapshot.coordinator, "the backbone must load exactly one coordinator");
        check(bytes > 0 && bytes <= MAX_COORDINATOR_BYTES, "coordinator exceeds the 640 MiB snapshot bound");
        // The caller owns ptr here. engine_load takes ownership, so this is the only snapshot boundary.
        const started = performance.now();
        snapshot.coordinator = wasm.bytes(ptr, bytes).slice();
        snapshot.copyMs = performance.now() - started;
      } catch (error) {
        // The original loader has not been called, so ownership still belongs to this caller.
        original.kevala_free(ptr, bytes);
        throw error;
      }
      return original.kevala_engine_load(ptr, bytes);
    },
    kevala_clef_prepare(ptr, bytes) {
      if (snapshot.active) {
        check(!snapshot.active.request && bytes <= MAX_INPUT_BYTES, "unexpected or oversized prepare input");
        snapshot.active.request = wasm.bytes(ptr, bytes).slice();
      }
      const rc = original.kevala_clef_prepare(ptr, bytes);
      if (snapshot.active && rc === 0) snapshot.active.plan = wasm.outText();
      return rc;
    },
    kevala_clef_finish(hiddenPtr, hiddenCount, lexicalPtr, lexicalCount) {
      if (snapshot.active) {
        check(!snapshot.active.hidden && (hiddenCount + lexicalCount) * 4 <= MAX_INPUT_BYTES, "unexpected or oversized head input");
        // These are valid const inputs owned by the caller until finish returns.
        snapshot.active.hidden = wasm.f32(hiddenPtr, hiddenCount).slice();
        snapshot.active.lexical = wasm.f32(lexicalPtr, lexicalCount).slice();
      }
      const rc = original.kevala_clef_finish(hiddenPtr, hiddenCount, lexicalPtr, lexicalCount);
      if (snapshot.active && rc === 0) snapshot.active.response = wasm.outText();
      return rc;
    },
  };
  wasm.tile = options.tile;
  original.kevala_set_tile(options.tile);
  return () => { wasm.x = original; };
}

async function loadCoordinator(create, Wasm, module, bytes, tile) {
  const started = performance.now(), wasm = await create.call(Wasm, module, tile);
  const instantiatedMs = performance.now() - started;
  const [ptr, length] = wasm.put(bytes);
  const loadStarted = performance.now();
  // Ownership transfers even when the loader returns an error. Never free this allocation here.
  wasm.call(() => wasm.check(wasm.x.kevala_engine_load(ptr, length)));
  return { wasm, instantiateMs: instantiatedMs, engineLoadMs: performance.now() - loadStarted };
}

function replay(wasm, captured) {
  wasm.withInput(captured.request, (ptr, bytes) => wasm.check(wasm.x.kevala_clef_prepare(ptr, bytes)));
  check(wasm.outText() === captured.plan, "CPU coordinator prepare plan differs from the captured plan");
  const [hiddenPtr, hiddenBytes] = wasm.put(new Uint8Array(captured.hidden.buffer));
  let lexicalPtr, lexicalBytes;
  try {
    [lexicalPtr, lexicalBytes] = wasm.put(new Uint8Array(captured.lexical.buffer));
    const started = performance.now();
    const rc = wasm.call(() => wasm.x.kevala_clef_finish(hiddenPtr, captured.hidden.length, lexicalPtr, captured.lexical.length));
    const ms = performance.now() - started;
    wasm.check(rc);
    return { ms, response: wasm.outText() };
  } finally {
    wasm.x.kevala_free(hiddenPtr, hiddenBytes);
    if (lexicalPtr !== undefined) wasm.x.kevala_free(lexicalPtr, lexicalBytes);
  }
}

export function completedEnvelope(details) {
  check(details.cases.length === 3 && details.cases.every((item) => item.pairs.length === SAMPLES && item.warmupPairs.length === WARMUPS && item.correctness.serializedResponsesEqual === true
    && item.baseline.samples.length === SAMPLES && item.candidate.samples.length === SAMPLES
    && timingSummary(item.baseline.samples).medianMs === item.baseline.medianMs && timingSummary(item.candidate.samples).medianMs === item.candidate.medianMs
    && item.pairs.every((pair, index) => pair.serializedResponsesEqual === true && pair.baselineMs === item.baseline.samples[index] && pair.candidateMs === item.candidate.samples[index])), "all three request shapes require nine exact paired results");
  return { ...details, status: "done", done: true, backend: "webgpu", method: SCOPE,
    // Required by the generic gpu runner. The labeled first-shape CPU median is not an aggregate.
    metricMs: details.cases[0].candidate.medianMs, metricScope: "candidate CPU finish median, short request only",
    correctness: { ok: true, exactSerializedResponses: true, maxPublicOutputDelta: 0 } };
}

async function main() {
  const log = (line) => { document.getElementById("log").textContent += `${line}\n`; };
  const options = optionsFor(location.href), snapshot = { coordinator: null, active: null };
  const started = performance.now();
  const { Kevala } = await import(options.runtime.href);
  const { Wasm, compile, wasmFlavor } = await import(new URL("wasm.js", options.currentBase));
  const { parsePackHeader } = await import(new URL("pack-layout.js", options.currentBase));
  const { archPlugin } = await import(new URL("archs/index.js", options.currentBase));
  const flavor = wasmFlavor(options.flavor), currentWasmUrl = new URL(`kevala-${flavor}.wasm`, options.currentBase);
  const originalFetch = globalThis.fetch, create = Wasm.create, plugin = archPlugin("clef"), originalCreateGpu = plugin.createGpu;
  let currentIdentity, model, primary, restoreExports, gpu, baselineCpu, candidateCpu;
  try {
    const pack = await packIdentity(options), sources = await javascriptIdentity(options.runtime);
    const harness = await resource(new URL(import.meta.url));
    globalThis.fetch = async (...args) => {
      const response = await originalFetch(...args);
      const requested = args[0] instanceof Request ? args[0].url : new URL(String(args[0]), document.baseURI).href;
      if (requested === currentWasmUrl.href) {
        check(response.ok, `current WASM: HTTP ${response.status}`);
        const bytes = await codeBytes(response.clone());
        currentIdentity = { url: currentWasmUrl.href, bytes: bytes.length, sha256: await sha256(bytes) };
      }
      return response;
    };
    const compileStarted = performance.now(), currentModule = await compile(options.currentBase.href, flavor);
    const currentCompileMs = performance.now() - compileStarted;
    globalThis.fetch = originalFetch;
    check(currentIdentity, "current WASM must be fetched and identified in this fresh benchmark page");
    const baselineResource = await resource(new URL(`kevala-${flavor}.wasm`, options.baseline));
    const baselineCompileStarted = performance.now(), baselineModule = await WebAssembly.compile(baselineResource.bytes);
    const baselineCompileMs = performance.now() - baselineCompileStarted;
    Wasm.create = async function(module, tile) {
      check(module === currentModule && !primary, "unexpected coordinator module or duplicate GPU model instance");
      const instance = await create.call(Wasm, module, tile);
      primary = instance;
      restoreExports = captureExports(instance, snapshot, options);
      return instance;
    };
    plugin.createGpu = function(runtime, ...args) { gpu = runtime; return originalCreateGpu.call(this, runtime, ...args); };
    log("Loading one frozen current Clef GPU model and copying its coordinator before ownership transfer.");
    const loadStarted = performance.now();
    model = await Kevala.load({ model: { name: "clef-flash", url: options.pack.href }, precision: "q8", backend: "webgpu", onPage: true,
      wasmBase: options.currentBase.href, flavor, threads: 1, cache: false, stateCache: false, gpuKernel: "auto", submit: "await" });
    const coldModelLoadMs = performance.now() - loadStarted;
    Wasm.create = create;
    plugin.createGpu = originalCreateGpu;
    check(model.info.arch === "clef" && model.info.onPage && model.info.pack.bytes === PACK_BYTES, "expected one on-page Clef Q8 model");
    check(gpu && !gpu.adapter.isFallbackAdapter && !gpu.adapter.info?.isFallbackAdapter && !/swiftshader|llvmpipe|software/i.test(gpu.name), "a hardware WebGPU backbone is required for capture");
    const header = parsePackHeader(snapshot.coordinator);
    check(!header.tensors.some((tensor) => tensor.name === "emb" || tensor.name === "clef.lexical" || tensor.name.startsWith("L.")), "CPU coordinator snapshot must exclude the GPU backbone and embedding tables");
    const subpack = { bytes: snapshot.coordinator.length, sha256: await sha256(snapshot.coordinator), snapshotBoundary: "copied immediately before kevala_engine_load ownership transfer" };
    baselineCpu = await loadCoordinator(create, Wasm, baselineModule, snapshot.coordinator, options.tile);
    candidateCpu = await loadCoordinator(create, Wasm, currentModule, snapshot.coordinator, options.tile);
    check(primary.tile === options.tile && baselineCpu.wasm.tile === options.tile && candidateCpu.wasm.tile === options.tile, "all coordinators must use the same fixed CPU tile");
    snapshot.coordinator = null;
    const words = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
    const question = { q: { type: "noul", instructions: "Is the customer asking for money back?" } };
    const fixtures = [{ name: "1 question, ~40 tokens", state: "Refund the duplicate charge today or we cancel." },
      { name: "1 question, ~128 tokens", state: words.repeat(6) }, { name: "1 question, ~512 tokens", state: words.repeat(30) }];
    const cases = [];
    for (const fixture of fixtures) {
      window.gpuBench.stage = `capturing ${fixture.name}`;
      snapshot.active = {};
      await model.decide(`Run 3. ${fixture.state}`, question);
      const captured = snapshot.active;
      snapshot.active = null;
      check(captured.request && captured.hidden && captured.lexical && captured.plan && captured.response, "one complete prepare/finish capture is required per request shape");
      const identity = { requestSha256: await sha256(captured.request), planSha256: await sha256(enc.encode(captured.plan)), hiddenSha256: await sha256(captured.hidden), lexicalSha256: await sha256(captured.lexical),
        requestBytes: captured.request.byteLength, planBytes: enc.encode(captured.plan).byteLength, hiddenBytes: captured.hidden.byteLength, lexicalBytes: captured.lexical.byteLength,
        hiddenValues: captured.hidden.length, lexicalValues: captured.lexical.length, tokens: JSON.parse(captured.plan).ids.length };
      const pairs = [], warmupPairs = [], samples = { baseline: [], candidate: [] };
      for (let pair = 0; pair < WARMUPS + SAMPLES; pair++) {
        window.gpuBench.stage = `CPU replay ${fixture.name}, pair ${pair + 1}/${WARMUPS + SAMPLES}`;
        const order = pair % 2 ? ["candidate", "baseline"] : ["baseline", "candidate"], results = {};
        for (const arm of order) results[arm] = replay(arm === "baseline" ? baselineCpu.wasm : candidateCpu.wasm, captured);
        check(results.baseline.response === results.candidate.response && results.candidate.response === captured.response, "baseline, candidate and captured serialized public responses must match exactly");
        const record = { order, baselineMs: results.baseline.ms, candidateMs: results.candidate.ms, serializedResponsesEqual: true };
        if (pair >= WARMUPS) {
          samples.baseline.push(results.baseline.ms); samples.candidate.push(results.candidate.ms);
          pairs.push({ sample: pair - WARMUPS, ...record });
        } else warmupPairs.push({ warmup: pair, ...record });
        await tick();
      }
      cases.push({ name: fixture.name, ...identity, baseline: timingSummary(samples.baseline), candidate: timingSummary(samples.candidate), pairs, warmupPairs,
        response: JSON.parse(captured.response), correctness: { serializedResponsesEqual: true, maxPublicOutputDelta: 0 } });
      log(`${fixture.name}: ${cases.at(-1).baseline.medianMs.toFixed(2)} -> ${cases.at(-1).candidate.medianMs.toFixed(2)} ms CPU finish; exact public output`);
    }
    const info = gpu.adapter.info || {};
    return completedEnvelope({ schemaVersion: 1, scope: SCOPE, flavor, tile: options.tile, cpuKernel: options.tile ? "4x4" : "2x4", pack, subpack, harness: harness.identity,
      environment: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency || null, crossOriginIsolated: globalThis.crossOriginIsolated === true },
      runtime: { entry: options.runtime.href, javascriptScope: "literal relative JavaScript imports from runtime index and computed engine-worker entry", javascript: sources,
        javascriptSha256: await sha256(enc.encode(sources.map((source) => `${new URL(source.url).pathname}:${source.sha256}\n`).join(""))),
        baselineWasm: baselineResource.identity, candidateWasm: currentIdentity },
      cold: { currentFetchHashCompileMs: currentCompileMs, baselineCompileMs, modelLoadMs: coldModelLoadMs, coordinatorCopyMs: snapshot.copyMs, baselineInstantiateMs: baselineCpu.instantiateMs, candidateInstantiateMs: candidateCpu.instantiateMs,
        baselineEngineLoadMs: baselineCpu.engineLoadMs, candidateEngineLoadMs: candidateCpu.engineLoadMs },
      adapter: { name: gpu.name, vendor: info.vendor || null, architecture: info.architecture || null, device: info.device || null, description: info.description || null,
        isFallbackAdapter: Boolean(gpu.adapter.isFallbackAdapter || info.isFallbackAdapter), features: [...gpu.device.features].sort() },
      protocol: { samples: SAMPLES, warmups: WARMUPS, backboneRequestsPerFixture: 1, engineLoadWarmup: true, pairedOrder: "alternating baseline/candidate then candidate/baseline",
        timed: "synchronous kevala_clef_finish call only, including its internal normalization, head and response serialization",
        excluded: "GPU backbone, prepare, JS input allocation/copy, output decoding, hashes, cold loading and compilation", maxCoordinatorBytes: MAX_COORDINATOR_BYTES, maxInputBytes: MAX_INPUT_BYTES },
      limitations: ["Raw logits are not exported by this ABI; equality covers exact serialized public responses.", "The GPU model performs its normal load warmup, then one untimed backbone capture per fixture. GPU weights are loaded once.", "CPU coordinator models have no unload ABI; owned input allocations are freed, instance references are released, and the browser process exit releases their memory."],
      cases, elapsedMs: performance.now() - started });
  } finally {
    globalThis.fetch = originalFetch;
    Wasm.create = create;
    plugin.createGpu = originalCreateGpu;
    snapshot.active = null;
    snapshot.coordinator = null;
    restoreExports?.();
    model?.dispose();
    // There is no coordinator-unload ABI. Drop all instance references; transferred pack buffers belong to those instances.
    for (const cpu of [baselineCpu, candidateCpu]) if (cpu) { cpu.wasm.x = null; cpu.wasm.instance = null; }
    primary = null;
  }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", method: SCOPE };
  main().then((result) => { window.gpuBench = result; }).catch((error) => {
    window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error), correctness: { ok: false } };
    document.getElementById("log").textContent += `${error.stack || error}\n`;
  });
}
