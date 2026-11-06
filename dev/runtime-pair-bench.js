const encoder = new TextEncoder(), decoder = new TextDecoder();
const check = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const hex = (bytes) => [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
const sha256 = async (bytes) => hex(await crypto.subtle.digest("SHA-256", bytes));
const MAX_PACK_BYTES = 2 ** 31 - 1, MAX_CODE_BYTES = 64 * 1024 ** 2;
const WORDS = "the customer wrote in about a recurring charge they did not recognise and wants a refund today ";
const QUESTION = { type: "noul", instructions: "Is the customer asking for money back?" };
const SHAPES = [
  { name: "1 question, ~40 tokens", state: "Refund the duplicate charge today or we cancel." },
  { name: "1 question, ~128 tokens", state: WORDS.repeat(6) },
  { name: "1 question, ~512 tokens", state: WORDS.repeat(30) },
];

export function optionsFor(href) {
  const page = new URL(href), q = new URLSearchParams(page.search);
  for (const [key, value] of new URLSearchParams(page.hash.slice(1))) q.set(key, value);
  const sameOrigin = (value) => {
    const url = new URL(value, page);
    check(url.origin === page.origin && ["http:", "https:"].includes(url.protocol) && !url.username && !url.password && !url.hash && !url.search, "resources must be same-origin HTTP(S) URLs without credentials, query or fragment");
    return url;
  };
  const integer = (key, fallback, min, max) => {
    const value = Number(q.get(key) ?? fallback);
    check(Number.isSafeInteger(value) && value >= min && value <= max, `${key} must be an integer from ${min} through ${max}`);
    return value;
  };
  const model = q.get("model") || "laya", precision = q.get("precision") || "q8";
  check(["laya", "kev-0.8b"].includes(model), "resident pairs allow only laya or kev-0.8b");
  check(["q8", "bf16"].includes(precision) && (model !== "laya" || precision === "q8"), "unsupported small-model precision");
  const baseline = sameOrigin(q.get("baseline") || "../tmp/baseline-runtime/index.js");
  const candidate = sameOrigin(q.get("candidate") || "../js/src/index.js");
  check(baseline.pathname.endsWith("/index.js") && candidate.pathname.endsWith("/index.js") && new URL("./", baseline).href !== new URL("./", candidate).href, "runtime entries must be index.js in different directories");
  const packSha256 = q.get("packSha256");
  check(!packSha256 || /^[a-f0-9]{64}$/.test(packSha256), "packSha256 must be a lowercase SHA-256");
  return { model, precision, baseline, candidate, pack: sameOrigin(q.get("pack") || `../tmp/${model}-${precision}.kevala`), packSha256,
    samples: integer("samples", 9, 3, 20), warmups: integer("warmups", 3, 1, 5), deadlineMs: integer("deadlineMs", 240000, 30000, 600000),
    maxGpuBytes: integer("maxGpuBytes", 6 * 1024 ** 3, 1024 ** 3, 6 * 1024 ** 3), probabilityTolerance: 0.0005, submit: "await" };
}

export function timingSummary(samples) {
  check(samples.length > 0 && samples.every((value) => finite(value) && value > 0), "request timings must be positive finite milliseconds");
  const sorted = [...samples].sort((a, b) => a - b), middle = sorted.length >> 1;
  return { medianMs: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1], samples };
}

export function ratioSummary(samples) {
  const timing = timingSummary(samples);
  return { median: timing.medianMs, p95: timing.p95Ms, samples, unit: "candidate/baseline" };
}

export function selectedTuning(trunk, model) {
  check(trunk?.tuning?.selection instanceof Map, "loaded model has no inspectable matrix selection");
  const matrices = trunk.precisionTuning instanceof Map ? [...trunk.precisionTuning] : [[trunk.tuning.diagnostics?.dtype || "q8", trunk.tuning]];
  const matvecs = trunk.precisionMatvec instanceof Map ? [...trunk.precisionMatvec] : [];
  const matrix = matrices.map(([precision, tuning]) => ({ precision, defaultKernel: "generic",
    selection: [...tuning.selection].map(([key, kernel]) => { check(typeof key === "string" && ["generic", "wide"].includes(kernel), "invalid selected matrix kernel"); return { key, kernel }; }).sort((a, b) => a.key.localeCompare(b.key)) })).sort((a, b) => a.precision.localeCompare(b.precision));
  const matvec = matvecs.map(([precision, tuning]) => ({ precision, enabled: Boolean(tuning.pipelines),
    columns: tuning.pipelines?.columns || null, f16: tuning.pipelines?.f16 ?? null, selection: [...tuning.selection].sort() })).sort((a, b) => a.precision.localeCompare(b.precision));
  const recurrence = model === "kev-0.8b" ? { lanes: trunk.tuning.diagnostics?.recurrence?.lanes, zTiles: trunk.recurTiles } : null;
  if (recurrence) check([4, 8, 16].includes(recurrence.lanes) && recurrence.zTiles === recurrence.lanes / 2, "invalid selected recurrence geometry");
  return { gpuKernel: trunk.gpuKernel, recurrence, matrix, matvec };
}

export function tuningAttribution(a, b) {
  const reasons = [];
  if (a.gpuKernel !== b.gpuKernel) reasons.push(`kernel policy differs: A ${a.gpuKernel}, B ${b.gpuKernel}`);
  if (JSON.stringify(a.recurrence) !== JSON.stringify(b.recurrence)) reasons.push(`recurrence selection differs: A ${JSON.stringify(a.recurrence)}, B ${JSON.stringify(b.recurrence)}`);
  const matrices = (value) => new Map(value.matrix.map((entry) => [entry.precision, new Map(entry.selection.map(({ key, kernel }) => [key, kernel]))]));
  const left = matrices(a), right = matrices(b);
  for (const precision of new Set([...left.keys(), ...right.keys()])) {
    const l = left.get(precision) || new Map(), r = right.get(precision) || new Map();
    for (const key of new Set([...l.keys(), ...r.keys()])) if ((l.get(key) || "generic") !== (r.get(key) || "generic")) reasons.push(`matrix ${precision}/${key} differs: A ${l.get(key) || "generic"}, B ${r.get(key) || "generic"}`);
  }
  const vectors = (value) => new Map(value.matvec.map((entry) => [entry.precision, entry]));
  const lv = vectors(a), rv = vectors(b);
  for (const precision of new Set([...lv.keys(), ...rv.keys()])) {
    const l = lv.get(precision), r = rv.get(precision), ls = new Set(l?.selection), rs = new Set(r?.selection);
    for (const key of new Set([...ls, ...rs])) if (ls.has(key) !== rs.has(key)) reasons.push(`matvec ${precision}/${key} differs: A ${ls.has(key)}, B ${rs.has(key)}`);
    if ((ls.size || rs.size) && (l?.enabled !== r?.enabled || l?.columns !== r?.columns || l?.f16 !== r?.f16)) reasons.push(`matvec ${precision} pipeline geometry differs`);
  }
  return { attributionValid: reasons.length === 0, attributionReasons: reasons,
    attributionScope: "all loaded recurrence, matrix and matvec selections; conservative across token buckets, including unexercised selections" };
}

export async function loadResidentEngines(modules, options, snapshots, engines, loadTrace, signal) {
  for (const [index, module] of modules.entries()) {
    signal?.throwIfAborted();
    const record = { side: index === 0 ? "A" : "B", startedMs: performance.now(), queueDrained: false };
    loadTrace.push(record);
    const engine = await module.Kevala.load({ ...options, signal });
    engines[index] = engine;
    record.loadedMs = performance.now();
    const gpu = snapshots[index].gpu;
    check(gpu?.device?.queue && !gpu.lost, "loaded runtime has no live GPU queue to drain");
    record.drainStartedMs = performance.now();
    await gpu.device.queue.onSubmittedWorkDone();
    check(!gpu.lost, "runtime device was lost while draining its load");
    record.completedMs = performance.now(); record.queueDrained = true;
    record.queueDrainMs = record.completedMs - record.drainStartedMs;
    snapshots[index].loadMs = record.loadedMs - record.startedMs;
  }
  check(loadTrace.length === modules.length && loadTrace.every((record, index) => record.queueDrained && (index === 0 || record.startedMs >= loadTrace[index - 1].completedMs)), "runtime uploads or calibration overlapped");
}

export function decisionValues(response, model, responseModel) {
  check(response?.model === responseModel && response.answers && JSON.stringify(Object.keys(response.answers)) === '["q"]', "response model or complete answer keys differ");
  const answer = response.answers.q;
  check(answer?.type === "noul" && finite(answer.noul) && answer.noul >= 0 && answer.noul <= 1, "invalid noul answer");
  const raw = Object.hasOwn(response, "raw_probabilities");
  check(model !== "kev-0.8b" || raw, "Kev must provide raw probabilities");
  let probabilities = [1 - answer.noul, answer.noul];
  if (raw) {
    const map = response.raw_probabilities;
    check(map && typeof map === "object" && !Array.isArray(map) && JSON.stringify(Object.keys(map)) === '["q"]', "invalid raw probability map");
    const values = map.q;
    check(values && typeof values === "object", "invalid raw probability distribution");
    check(Array.isArray(values) || Object.keys(values).length === 2 && Object.hasOwn(values, "false") && Object.hasOwn(values, "true"), "invalid raw option keys");
    probabilities = Array.isArray(values) ? Array.from(values) : [values.false, values.true];
  }
  check(probabilities.length === 2 && probabilities.every((value) => finite(value) && value >= 0 && value <= 1) && Math.abs(probabilities[0] + probabilities[1] - 1) <= 0.0005, "invalid normalized probability distribution");
  return { raw, probabilities, answer };
}

export function compareDecisions(reference, candidate, model, responseModel, tolerance) {
  const a = decisionValues(reference, model, responseModel), b = decisionValues(candidate, model, responseModel);
  check(a.raw === b.raw && JSON.stringify(Object.keys(a.answer).sort()) === JSON.stringify(Object.keys(b.answer).sort()), "decision output contract changed");
  for (const key of Object.keys(a.answer)) if (key !== "noul") check(JSON.stringify(a.answer[key]) === JSON.stringify(b.answer[key]), `answer field ${key} changed`);
  const probabilityDelta = Math.max(...a.probabilities.map((value, index) => Math.abs(value - b.probabilities[index])));
  const publicNoulDelta = Math.abs(a.answer.noul - b.answer.noul);
  check(probabilityDelta <= tolerance && publicNoulDelta <= 2 * tolerance + 0.0001, `paired decision differs: raw ${probabilityDelta}, public ${publicNoulDelta}`);
  check((a.answer.noul >= 0.5) === (b.answer.noul >= 0.5), "paired binary choice changed");
  return { probabilityDelta, publicNoulDelta };
}

async function resource(url, signal, fetcher = fetch) {
  const response = await fetcher(url, { cache: "no-store", signal });
  const length = Number(response.headers.get("content-length"));
  check(response.ok && Number.isSafeInteger(length) && length > 0 && length <= MAX_CODE_BYTES, `unbounded or failed code resource: ${url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  check(bytes.length === length, `code resource length changed: ${url}`);
  return bytes;
}

export async function runtimeFingerprint(entry, signal, fetcher = fetch) {
  const base = new URL("./", entry), visited = new Set(), assets = [];
  const pending = ["index.js", "node.js", "engine-worker.js", "shard-worker.js", "kevala-base.wasm", "kevala-simd.wasm", "kevala-relaxed.wasm"].map((path) => new URL(path, base));
  let totalBytes = 0;
  while (pending.length) {
    const url = pending.pop();
    if (visited.has(url.href)) continue;
    visited.add(url.href);
    check(url.origin === base.origin && url.pathname.startsWith(base.pathname) && !url.search && !url.hash, "runtime import leaves its isolated tree");
    const bytes = await resource(url, signal, fetcher);
    totalBytes += bytes.length;
    check(visited.size <= 256 && totalBytes <= 512 * 1024 ** 2, "runtime tree exceeds its bounded inventory");
    assets.push({ path: url.pathname.slice(base.pathname.length), bytes: bytes.length, sha256: await sha256(bytes) });
    if (!url.pathname.endsWith(".js")) continue;
    for (const match of decoder.decode(bytes).matchAll(/\b(?:from\s+|import\s*\(?\s*|new\s+URL\(\s*)["'](\.[^"']+\.js)["']/g)) pending.push(new URL(match[1], url));
  }
  assets.sort((a, b) => a.path.localeCompare(b.path));
  return { entry: entry.href, sha256: await sha256(encoder.encode(assets.map((asset) => `${asset.path}:${asset.sha256}\n`).join(""))),
    scope: "reachable JavaScript plus explicit Node/worker entries and all three WASM flavors", bytes: totalBytes, assets };
}

async function packIdentity(options, modules, signal) {
  const layout = await import(new URL("./pack-layout.js", options.baseline));
  const head = await fetch(options.pack, { method: "HEAD", cache: "no-store", signal });
  const bytes = Number(head.headers.get("content-length"));
  check(head.ok && Number.isSafeInteger(bytes) && bytes > 0 && bytes <= MAX_PACK_BYTES, "resident paired pack must be smaller than 2 GiB with a bounded Content-Length");
  const range = async (length) => {
    const response = await fetch(options.pack, { cache: "no-store", signal, headers: { Range: `bytes=0-${length - 1}` } });
    check(response.status === 206 && response.headers.get("content-range") === `bytes 0-${length - 1}/${bytes}`, "pack server must provide bounded ranges and the verified full-file length");
    const value = new Uint8Array(await response.arrayBuffer()); check(value.length === length, "pack header range is incomplete"); return value;
  };
  const prefix = await range(16), headerBytes = layout.packHeaderLength(prefix);
  check(headerBytes >= 16 && headerBytes <= 16 * 1024 ** 2, "pack header exceeds its 16 MiB bound");
  const raw = await range(headerBytes), header = layout.parsePackHeader(raw);
  for (const module of modules) {
    const spec = module.MODELS[options.model];
    check(header.model?.name === (spec.name || options.model) && (header.config?.arch || "laya") === spec.arch && layout.packPrecision(header) === options.precision, "pack model, architecture, or precision differs from the requested catalog model");
    check(header.model.source === `https://huggingface.co/${spec.repo}` && header.model.revision === spec.revision, "pack source or revision differs from catalog");
    if (spec.base) check(header.model.base === `https://huggingface.co/${spec.base.repo}` && header.model.base_revision === spec.base.revision, "pack base revision differs from catalog");
    const expectedBytes = spec.precisionPackBytes?.[options.precision] ?? spec.pack;
    if (Number.isSafeInteger(expectedBytes) && spec.precisionPackApprox?.[options.precision] !== true) check(bytes === expectedBytes, "pack size differs from catalog");
    const minimum = layout.packSize(header, headerBytes);
    check(minimum === bytes || Math.ceil(minimum / 64) * 64 === bytes, "pack file length disagrees with its tensor layout");
  }
  const pinned = options.precision === "q8" ? modules[0].MODELS[options.model].packSha256 : modules[0].MODELS[options.model].precisionPackSha256?.[options.precision];
  const digest = options.packSha256 || pinned;
  check(/^[a-f0-9]{64}$/.test(digest || "") && (!pinned || digest === pinned), "provide the offline verified packSha256; a catalog-pinned hash must match");
  return { url: options.pack.href, bytes, sha256: digest, headerBytes, headerSha256: await sha256(raw), headerJsonSha256: await sha256(encoder.encode(JSON.stringify(header))),
    fullHashVerification: "controller-supplied offline digest; browser verifies header identity, layout, source, revision and file length", header,
    responseModel: options.model === "laya" ? "laya-rl-agent" : "kev-latest" };
}

function trackDevice(device, side, budget) {
  const original = device.createBuffer, owned = new Set();
  const state = { side, liveBytes: 0, peakBytes: 0, allocations: 0 };
  const release = (allocation) => {
    if (!owned.delete(allocation)) return;
    state.liveBytes -= allocation.bytes; budget.liveBytes -= allocation.bytes;
  };
  device.createBuffer = function(descriptor) {
    const bytes = descriptor.size;
    check(Number.isSafeInteger(bytes) && bytes >= 0 && budget.liveBytes + bytes <= budget.limitBytes, "combined requested GPU-buffer allocation bound exceeded");
    const buffer = original.call(this, descriptor), destroy = buffer.destroy;
    const allocation = { bytes }; owned.add(allocation);
    state.liveBytes += bytes; budget.liveBytes += bytes; state.allocations++;
    state.peakBytes = Math.max(state.peakBytes, state.liveBytes); budget.peakBytes = Math.max(budget.peakBytes, budget.liveBytes);
    buffer.destroy = function() { release(allocation); return destroy.call(this); };
    return buffer;
  };
  device.lost.then(() => { for (const allocation of owned) release(allocation); });
  return { state, restore: () => { device.createBuffer = original; } };
}

function frameTracker() {
  let last = performance.now(), gap = 0, frames = 0, animation;
  const tick = (time) => { gap = Math.max(gap, time - last); last = time; frames++; document.querySelector("#frame").style.transform = `translateX(${frames % 60}rem)`; animation = requestAnimationFrame(tick); };
  animation = requestAnimationFrame(tick);
  return () => { cancelAnimationFrame(animation); return { frameGapMs: Math.max(gap, performance.now() - last), frameCount: frames }; };
}

async function runBrowser() {
  const options = optionsFor(location.href), abort = new AbortController(), signal = abort.signal;
  const snapshots = [], engines = [], restore = [], trace = [], loadTrace = [];
  const budget = { limitBytes: options.maxGpuBytes, liveBytes: 0, peakBytes: 0 };
  const log = (text) => { document.querySelector("#log").textContent += `\n${text}`; };
  const dispose = () => { for (const engine of engines) engine?.dispose(); for (const snapshot of snapshots) snapshot.gpu?.device.destroy(); };
  const timer = setTimeout(() => { abort.abort(new Error("resident runtime pair deadline exceeded")); dispose(); }, options.deadlineMs);
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (input, init = {}) => originalFetch(input, { ...init, cache: "no-store" });
  const partial = { trace, loadTrace, calibrationSerialized: true, loadOrder: ["A", "B"] };
  window.gpuBench = { status: "running", backend: "webgpu", diagnostics: partial };
  try {
    const harness = [];
    for (const path of ["./runtime-pair-bench.html", "./runtime-pair-bench.js"]) {
      const url = new URL(path, import.meta.url), bytes = await resource(url, signal);
      harness.push({ url: url.href, bytes: bytes.length, sha256: await sha256(bytes) });
    }
    const entries = [options.baseline, options.candidate];
    const before = [];
    for (const entry of entries) before.push(await runtimeFingerprint(entry, signal));
    const modules = await Promise.all(entries.map((entry, index) => { const url = new URL(entry); url.searchParams.set("resident-pair", `${index}-${Date.now()}`); return import(url.href); }));
    check(JSON.stringify(modules[0].MODELS[options.model]) === JSON.stringify(modules[1].MODELS[options.model]), "runtime model catalogs differ");
    const pack = await packIdentity(options, modules, signal);
    partial.pack = { ...pack, header: undefined };
    for (const [index, entry] of entries.entries()) {
      const { archPlugin } = await import(new URL("./archs/index.js", entry));
      const plugin = archPlugin(modules[index].MODELS[options.model].arch), original = plugin.createGpu;
      const snapshot = { side: index === 0 ? "A" : "B", devices: [] }; snapshots.push(snapshot);
      plugin.createGpu = function(gpu, layout, header, ...args) {
        snapshot.gpu = gpu; snapshot.header = header;
        const tracked = trackDevice(gpu.device, snapshot.side, budget); snapshot.devices.push(tracked.state); restore.push(tracked.restore);
        snapshot.trunk = original.call(this, gpu, layout, header, ...args);
        return snapshot.trunk;
      };
      restore.push(() => { plugin.createGpu = original; });
    }
    // Finish A's uploads, calibration and final queued writes before starting B's load.
    await loadResidentEngines(modules, { model: { name: options.model, url: options.pack.href }, precision: options.precision, backend: "webgpu", onPage: true, cache: false, stateCache: false, profile: false, threads: 1, submit: options.submit }, snapshots, engines, loadTrace, signal);
    const tuningBefore = snapshots.map((snapshot) => selectedTuning(snapshot.trunk, options.model));
    partial.selectedTuning = tuningBefore;
    const adapterKey = (snapshot) => {
      const gpu = snapshot.gpu, info = gpu?.adapter.info || {};
      check(gpu && !gpu.lost && !info.isFallbackAdapter && !gpu.adapter.isFallbackAdapter && !/swiftshader|llvmpipe|software/i.test(gpu.name), "hardware WebGPU is required");
      return { name: gpu.name, vendor: info.vendor || null, architecture: info.architecture || null, device: info.device || null, description: info.description || null,
        isFallbackAdapter: false, features: [...gpu.device.features].sort(), limits: { maxBufferSize: gpu.device.limits.maxBufferSize, maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: gpu.device.limits.maxComputeWorkgroupStorageSize } };
    };
    const adapters = snapshots.map(adapterKey);
    check(JSON.stringify(adapters[0]) === JSON.stringify(adapters[1]) && snapshots[0].gpu.device !== snapshots[1].gpu.device, "runtimes must use separate devices on the same hardware/feature set");
    for (const [index, engine] of engines.entries()) {
      check(engine.info.backend === "webgpu" && engine.info.onPage === true && engine.info.pack?.cached === false && engine.info.pack.bytes === pack.bytes, "engine backend, context or pack cache differs");
      check(engine.info.model?.name === pack.header.model.name && engine.info.model.revision === pack.header.model.revision && engine.info.model.source === pack.header.model.source, "loaded model identity differs from the verified pack");
      check(await sha256(encoder.encode(JSON.stringify(snapshots[index].header))) === pack.headerJsonSha256, "loaded GPU header differs from the verified pack");
      check(engine.info.arch === modules[index].MODELS[options.model].arch && (options.model !== "kev-0.8b" || engine.info.stateCache?.enabled === false), "loaded architecture or state-cache policy differs");
    }
    const cases = [], correctness = { ok: true, probabilityTolerance: options.probabilityTolerance, maxProbabilityDelta: 0, maxPublicNoulDelta: 0 };
    for (const [shape, fixture] of SHAPES.entries()) {
      const pairs = [];
      for (let pair = 0; pair < options.warmups + options.samples; pair++) {
        signal.throwIfAborted();
        const request = { state: `Run ${pair}. ${fixture.state}`, questions: { q: QUESTION } };
        const requestHash = await sha256(encoder.encode(JSON.stringify([request])));
        const order = pair % 2 ? [1, 0, 0, 1] : [0, 1, 1, 0], calls = [];
        for (const [position, side] of order.entries()) {
          const stopFrames = frameTracker(), started = performance.now();
          let response, wallMs, frames;
          try { response = await engines[side].decide(request.state, request.questions); wallMs = performance.now() - started; }
          finally { frames = stopFrames(); }
          const { timing, ...output } = response;
          const sample = { shape, pair, measured: pair >= options.warmups, position, side: side === 0 ? "A" : "B", requestHash, wallMs, ...frames, timing, output };
          trace.push(sample); calls.push(sample);
          check(finite(wallMs) && wallMs > 0 && Number.isSafeInteger(timing?.tokens) && timing.tokens > 0 && timing.gpu == null && timing.batched === 1, "invalid unprofiled request timing or token count");
          if (options.model === "kev-0.8b") check(timing.cache?.tokensSaved === 0 && timing.cache?.hits === 0 && timing.cache?.extensions === 0, "uncached control reused a state carry");
          decisionValues(output, options.model, pack.responseModel);
          if (calls.length > 1) {
            check(timing.tokens === calls[0].timing.tokens, "paired actual token counts differ");
            const delta = compareDecisions(calls[0].output, output, options.model, pack.responseModel, options.probabilityTolerance);
            correctness.maxProbabilityDelta = Math.max(correctness.maxProbabilityDelta, delta.probabilityDelta);
            correctness.maxPublicNoulDelta = Math.max(correctness.maxPublicNoulDelta, delta.publicNoulDelta);
          }
        }
        if (pair >= options.warmups) {
          const mean = (side) => calls.filter((call) => call.side === side).reduce((sum, call) => sum + call.wallMs, 0) / 2;
          const baselineMs = mean("A"), candidateMs = mean("B");
          pairs.push({ pair, requestHash, order: calls.map((call) => call.side), tokens: calls[0].timing.tokens, baselineMs, candidateMs, ratio: candidateMs / baselineMs, deltaPercent: (candidateMs / baselineMs - 1) * 100, calls });
        }
      }
      check(pairs.length === options.samples && new Set(pairs.map((pair) => pair.requestHash)).size === options.samples, "incomplete or repeated measured request pairs");
      const summary = { name: fixture.name, shape, tokenCounts: pairs.map((pair) => pair.tokens), pairs,
        baseline: timingSummary(pairs.flatMap((pair) => pair.calls.filter((call) => call.side === "A").map((call) => call.wallMs))),
        candidate: timingSummary(pairs.flatMap((pair) => pair.calls.filter((call) => call.side === "B").map((call) => call.wallMs))),
        pairedBaseline: timingSummary(pairs.map((pair) => pair.baselineMs)), pairedCandidate: timingSummary(pairs.map((pair) => pair.candidateMs)),
        pairedRatio: ratioSummary(pairs.map((pair) => pair.ratio)) };
      cases.push(summary); log(`${fixture.name}: A ${summary.baseline.medianMs.toFixed(2)} ms, B ${summary.candidate.medianMs.toFixed(2)} ms; paired ratio ${summary.pairedRatio.median.toFixed(4)} candidate/baseline`);
    }
    const after = [];
    for (const entry of entries) after.push(await runtimeFingerprint(entry, signal));
    before.forEach((fingerprint, index) => check(fingerprint.sha256 === after[index].sha256, "runtime tree changed during paired requests"));
    const packAfter = await packIdentity(options, modules, signal);
    check(packAfter.headerSha256 === pack.headerSha256 && packAfter.bytes === pack.bytes, "pack header or length changed during paired requests");
    check(snapshots.every((snapshot) => !snapshot.gpu.lost), "a paired runtime device was lost");
    for (const identity of harness) check(await sha256(await resource(new URL(identity.url), signal)) === identity.sha256, "paired harness changed during requests");
    const tuningAfter = snapshots.map((snapshot) => selectedTuning(snapshot.trunk, options.model));
    const attribution = tuningAttribution(tuningBefore[0], tuningBefore[1]);
    for (const [index, selection] of tuningBefore.entries()) if (JSON.stringify(selection) !== JSON.stringify(tuningAfter[index])) {
      attribution.attributionValid = false; attribution.attributionReasons.push(`side ${index === 0 ? "A" : "B"} tuning changed during paired requests`);
    }
    const geometric = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
    return { schemaVersion: 1, status: "done", done: true, backend: "webgpu", benchmark: "resident-runtime-pairs", model: options.model, precision: options.precision,
      method: "geometric mean of three candidate unprofiled completed-request median milliseconds", metricMs: geometric(cases.map((item) => item.candidate.medianMs)),
      baselineMetricMs: geometric(cases.map((item) => item.baseline.medianMs)), adapter: adapters[0], correctness, pack: { ...pack, header: undefined },
      ...attribution,
      harness,
      runtime: before.map((identity, index) => ({ side: index === 0 ? "A" : "B", ...identity, verifiedAfterSha256: after[index].sha256, loadMs: snapshots[index].loadMs, modelInfo: engines[index].info.model, gpuTuning: engines[index].info.gpuTuning, selectedTuning: tuningBefore[index], selectedTuningAfter: tuningAfter[index] })),
      environment: { userAgent: navigator.userAgent, brands: navigator.userAgentData?.brands || null, hardwareConcurrency: navigator.hardwareConcurrency, visibility: document.visibilityState, onPage: true },
      resources: { packLimitBytes: MAX_PACK_BYTES, simultaneouslyResidentPacks: 2, combinedPackBytes: 2 * pack.bytes, gpuBuffers: { ...budget, scope: "requested GPUBuffer sizes before disposal; not a measurement of physical VRAM", devices: snapshots.flatMap((snapshot) => snapshot.devices.map((state) => ({ ...state }))) } },
      protocol: { samples: options.samples, warmups: options.warmups, sequence: "four serial calls per pair; ABBA then BAAB, alternating start side", cache: false, stateCache: false, profile: false, submit: options.submit,
        loadOrder: ["A", "B"], calibrationSerialized: true, queueDrainedBeforeNextLoad: true,
        requestIdentity: "same deterministic Run i state and question for all four calls; distinct measured pairs", pairStatistic: "arithmetic mean of each side's two calls; ratio B/A per pair",
        timing: "unprofiled completed-request wall milliseconds; loading, source hashing, request hashing and output validation excluded", sourceCaches: "fetch no-store; isolated runtime directories and page engines", responsiveness: "animation-frame gaps; headless callbacks do not establish foreground compositor responsiveness" },
      diagnostics: { cases, loadTrace, warmupCalls: trace.filter((sample) => !sample.measured) } };
  } catch (error) { return { status: "error", done: true, backend: "webgpu", error: String(error?.message || error), diagnostics: partial }; }
  finally { clearTimeout(timer); abort.abort(); dispose(); restore.reverse().forEach((callback) => callback()); globalThis.fetch = originalFetch; }
}

if (typeof document !== "undefined") {
  window.gpuBench = { status: "running", backend: "webgpu" };
  runBrowser().then((result) => { window.gpuBench = window.bench = result; document.querySelector("#log").textContent += `\n${JSON.stringify(result, null, 2)}`; }).catch((error) => {
    window.gpuBench = window.bench = { status: "error", done: true, backend: "webgpu", error: String(error?.message || error) }; document.querySelector("#log").textContent += `\n${window.gpuBench.error}`;
  });
}
