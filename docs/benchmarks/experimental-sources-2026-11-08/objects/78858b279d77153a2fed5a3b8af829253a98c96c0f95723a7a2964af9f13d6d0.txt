import { Kevala } from "../js/src/index.js";
import { archPlugin } from "../js/src/archs/index.js";
import { adapterFallbackMetadata } from "./adapter-metadata.js";
import { LAYA_Q8_GOLDEN_MAX_DP } from "./laya-parity-metrics.js";
import { runLayaGeGluProtocol } from "../scripts/private/laya-geglu-protocol.mjs";

const check = (ok, message) => { if (!ok) throw new Error(message); };
const sha256 = async (bytes) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((b) => b.toString(16).padStart(2, "0")).join("");
const text = new TextEncoder();
const median = (values) => { const sorted = [...values].sort((a, b) => a - b), i = sorted.length >> 1; return sorted.length % 2 ? sorted[i] : (sorted[i - 1] + sorted[i]) / 2; };
const outputJson = (responses) => JSON.stringify(responses.map(({ timing, ...output }) => output));
const tuningJson = (trunk) => JSON.stringify({ gpuKernel: trunk.gpuKernel, selection: [...trunk.tuning.selection].sort(),
  kernel: trunk.mm.kernel, f16: trunk.mm.f16, groups: trunk.mm.groups, splitTarget: trunk.mm.splitTarget });

function observePrivateLoad(partial) {
  const NativeWorker = globalThis.Worker, NativeChannel = globalThis.MessageChannel, listeners = [];
  const diagnostic = partial.webGpuInitialization = { pageGpuPresent: !!navigator.gpu, workers: 0, pageChannels: 0, firstError: null, retrySuppressed: false };
  const observe = (target, source) => {
    const listener = ({ data }) => {
      if (!diagnostic.firstError && data?.type === "error" && data.id === undefined && data.code === "WEBGPU_INIT") {
        diagnostic.firstError = { source, message: data.message, code: data.code };
      }
    };
    target.addEventListener("message", listener);
    listeners.push([target, listener]);
  };
  globalThis.Worker = class extends NativeWorker {
    constructor(...args) { super(...args); diagnostic.workers++; observe(this, "worker-engine"); }
  };
  globalThis.MessageChannel = class extends NativeChannel {
    constructor(...args) { super(...args); diagnostic.pageChannels++; observe(this.port1, "page-engine"); }
  };
  return () => {
    globalThis.Worker = NativeWorker; globalThis.MessageChannel = NativeChannel;
    for (const [target, listener] of listeners) target.removeEventListener("message", listener);
    diagnostic.observersRestored = true;
  };
}

async function readJson(url, signal) {
  const response = await fetch(url, { cache: "no-store", signal });
  check(response.ok, `Could not read ${url}: HTTP ${response.status}`);
  return response.json();
}

async function verifyResources(manifest, signal) {
  const observed = [];
  for (const entry of manifest.resources) {
    const url = new URL(`../${entry.path}`, import.meta.url);
    check(url.origin === location.origin, "Manifest resource left the serving origin");
    const response = await fetch(url, { cache: "no-store", signal });
    check(response.ok && Number(response.headers.get("content-length")) === entry.bytes, `Resource size differs: ${entry.path}`);
    const bytes = await response.arrayBuffer();
    check(bytes.byteLength === entry.bytes && await sha256(bytes) === entry.sha256, `Resource hash differs: ${entry.path}`);
    observed.push({ ...entry });
  }
  return observed;
}

async function verifyPackHeader(manifest, signal) {
  const { pack } = manifest;
  const response = await fetch(pack.url, { cache: "no-store", signal, headers: { Range: `bytes=0-${pack.headerBytes - 1}` } });
  check(response.status === 206 && response.headers.get("content-range") === `bytes 0-${pack.headerBytes - 1}/${pack.bytes}`, "Pack header range or file size differs");
  const bytes = await response.arrayBuffer();
  check(bytes.byteLength === pack.headerBytes && await sha256(bytes) === pack.headerSha256, "Pack header hash differs");
  return { headerBytes: bytes.byteLength, headerSha256: pack.headerSha256, fullSha256: pack.sha256,
    fullHashVerification: "controller verified the local pack before launch; page verifies header and loaded model at bookends" };
}

function probabilityVector(answer) {
  check(answer && ["noul", "choice", "score"].includes(answer.type), "Golden guard received an invalid answer type");
  const values = answer.type === "noul" ? { false: 1 - answer.noul, true: answer.noul } : answer.probabilities;
  check(values && typeof values === "object" && !Array.isArray(values), "Golden guard received no probability map");
  const keys = Object.keys(values).sort(), vector = keys.map((key) => values[key]);
  check(keys.length > 0 && vector.every((v) => Number.isFinite(v) && v >= 0 && v <= 1), "Golden guard received a non-finite or invalid probability");
  check(Math.abs(vector.reduce((a, b) => a + b, 0) - 1) <= keys.length * 0.0000501 + 1e-7, "Golden guard probabilities are not normalized within the existing API rounding bound");
  const best = keys[vector.indexOf(Math.max(...vector))];
  if (answer.type === "choice") check(answer.choice === best || values[answer.choice] === Math.max(...vector), "Choice is inconsistent with its probability vector");
  return { keys, vector, best };
}

async function defaultOffGoldenGuard(engine, trunk, golden, signal) {
  check(!trunk.layaGeGlu && !trunk.profiler, "Default-off guard must precede prototype configuration and profiling");
  const cases = [];
  let currentFixture = null;
  try {
    for (const fixture of golden.cases) {
      signal.throwIfAborted();
      currentFixture = { id: fixture.id };
      const response = await engine.decide(fixture.state, fixture.questions);
      currentFixture.response = JSON.parse(outputJson([response]))[0];
      check(response.model === fixture.response.model && JSON.stringify(Object.keys(response.answers).sort()) === JSON.stringify(Object.keys(fixture.questions).sort()), "Default-off response model or complete answer keys differ");
      const questions = Object.keys(fixture.questions).map((id) => {
        const expected = probabilityVector(fixture.response.answers[id]), actual = probabilityVector(response.answers[id]);
        check(response.answers[id].type === fixture.questions[id].type && JSON.stringify(expected.keys) === JSON.stringify(actual.keys), `Default-off option contract differs: ${fixture.id}/${id}`);
        const maxDp = Math.max(...actual.vector.map((value, i) => Math.abs(value - expected.vector[i])));
        check(maxDp <= LAYA_Q8_GOLDEN_MAX_DP && actual.best === expected.best, `Default-off Q8 golden guard failed: ${fixture.id}/${id}, maxDp=${maxDp}`);
        return { id, maxDp, argmaxMatches: true, expected: expected.vector, actual: actual.vector };
      });
      cases.push({ id: fixture.id, questions });
    }
    check(cases.length === golden.cases.length && cases.length > 0, "Default-off golden guard has incomplete coverage");
    return { ok: true, scope: "existing Q8 golden response probability vectors and argmax; no WASM/WebGPU parity claim",
      maxDp: Math.max(...cases.flatMap((entry) => entry.questions.map((q) => q.maxDp))), tolerance: LAYA_Q8_GOLDEN_MAX_DP, cases };
  } catch (error) { throw Object.assign(error, { defaultOffPartial: { cases, currentFixture, tolerance: LAYA_Q8_GOLDEN_MAX_DP } }); }
}

function quartetGains(quartets, field) {
  return quartets.map(({ order, samples }) => {
    const mean = (lane) => samples.filter((sample) => sample.lane === lane).reduce((sum, sample) => sum + sample[field], 0) / 2;
    const baselineMs = mean("A"), candidateMs = mean("B");
    check(Number.isFinite(baselineMs) && baselineMs > 0 && Number.isFinite(candidateMs) && candidateMs > 0, "Invalid paired timing");
    return { order, baselineMs, candidateMs, gain: (baselineMs - candidateMs) / baselineMs };
  });
}

function acceptance(protocol) {
  const request = quartetGains(protocol.quartets, "wallMs"), kernel = quartetGains(protocol.pairedKernel.quartets, "pairedMs");
  const aaSpread = Math.max(...protocol.aa.map(({ samples }) => Math.abs((samples[0].wallMs - samples[1].wallMs) / samples[0].wallMs)));
  const requestMedianGain = median(request.map((q) => q.gain)), kernelMedianGain = median(kernel.map((q) => q.gain));
  const orderGains = Object.fromEntries(["ABBA", "BAAB"].map((order) => [order, median(request.filter((q) => q.order === order).map((q) => q.gain))]));
  const kernelPassed = kernelMedianGain >= 0.05 && kernel.every((q) => q.gain > 0) && kernel.filter((q) => q.gain >= 0.05).length >= 6;
  const requestPassed = Object.values(orderGains).every((gain) => gain > 0) && request.filter((q) => q.gain > 0).length >= 8 && requestMedianGain > aaSpread;
  return { candidateAccepted: kernelPassed && requestPassed, promotion: "not authorized; remains private and default off", kernelPassed, requestPassed,
    kernelMedianGain, requestMedianGain, requestOrderMedianGains: orderGains, maximumAbsoluteAaSpread: aaSpread, kernel, request,
    policy: "predeclared 5 percent stable kernel gain and repeatable request gain beyond all A/A spread; no exclusions, corrections, or retries" };
}

async function profilePaired(engine, { trunk, configure, run, warmups, quartets, aaPairs }) {
  const result = { warmups: [], aa: [], quartets: [], scope: "sum of all 28 mm.wi+geglu baseline pairs versus all 28 private paired kernels", timestampRoundingMs: 0.001 };
  let reference = null;
  check(await engine.profile(true), "timestamp-query is unavailable");
  try {
    const sample = async (lane) => {
      await configure(lane === "A" ? "off" : "on");
      const profiler = trunk.profiler, originalPass = profiler.pass, counts = {};
      profiler.pass = function(encoder, label) { counts[label] = (counts[label] || 0) + 1; return originalPass.call(this, encoder, label); };
      let responses;
      try { responses = await run(); } finally { profiler.pass = originalPass; }
      const output = outputJson(responses);
      reference ??= output;
      check(output === reference && responses[0].timing?.tokens === 512, "Profiled request output or active T differs");
      const gpu = responses[0].timing.gpu;
      check(gpu && Object.values(gpu).every((v) => Number.isFinite(v) && v >= 0), "Invalid GPU timestamp profile");
      const pairedMs = lane === "A" ? gpu["mm.wi"] + gpu.geglu : gpu["mm.wi.geglu.private"];
      check(Number.isFinite(pairedMs) && pairedMs > 0, "Missing paired MLP timestamp coverage");
      check(lane === "A" ? counts["mm.wi"] === 28 && counts.geglu === 28 && !counts["mm.wi.geglu.private"] : counts["mm.wi.geglu.private"] === 28 && !counts["mm.wi"] && !counts.geglu,
        "Profiled MLP dispatch coverage differs from all 28 layers");
      if (lane === "B") check(trunk.layaGeGlu.lastEligibility?.enabled === true, "Profiled candidate did not consume the exact request permit");
      return { lane, pairedMs, gpu, dispatchCounts: counts };
    };
    for (let i = 0; i < warmups; i++) for (const lane of "AB") result.warmups.push(await sample(lane));
    check(aaPairs === quartets.length, "Kernel A/A and quartet plan differs");
    for (const [round, order] of quartets.entries()) {
      const aa = [], samples = [];
      result.aa.push({ round, samples: aa });
      for (const lane of "AA") aa.push(await sample(lane));
      result.quartets.push({ round, order, samples });
      for (const lane of order) samples.push(await sample(lane));
    }
    return result;
  } catch (error) { throw Object.assign(error, { pairedKernelPartial: result }); }
  finally { await engine.profile(false); }
}

async function runPage() {
  let engine, snapshot;
  const abort = new AbortController(), signal = abort.signal;
  const partial = { stage: "identity", protocol: "private-laya-geglu-v2", defaultMode: "off", loads: 0 };
  const publish = (value) => { window.gpuBench = value; document.getElementById("log").textContent = JSON.stringify(value, null, 2); };
  publish({ status: "running", backend: "webgpu", diagnostics: partial });
  const deadline = setTimeout(() => { abort.abort(new Error("Private Laya page 600 second deadline exceeded")); engine?.dispose(); }, 600000);
  const plugin = archPlugin("laya"), original = plugin.createGpu;
  try {
    const manifestUrl = new URL(new URLSearchParams(location.search).get("manifest"), location.href);
    check(manifestUrl.origin === location.origin && manifestUrl.pathname.startsWith("/tmp/laya-geglu-manifest-"), "Manifest must be the runner-owned local manifest");
    const manifest = await readJson(manifestUrl, signal);
    partial.manifest = manifest;
    partial.resourcesBefore = await verifyResources(manifest, signal);
    partial.packBefore = await verifyPackHeader(manifest, signal);
    plugin.createGpu = function(gpu, layout, header, actualEngine) {
      check(++partial.loads === 1, "Only one Laya GPU model instance is permitted");
      const trunk = original.call(this, gpu, layout, header, actualEngine);
      snapshot = { gpu, header, trunk, actualEngine };
      return trunk;
    };
    partial.stage = "load-default-off";
    const loadOptions = { backend: "webgpu", onPage: true, model: { name: "laya", url: manifest.pack.url }, precision: "q8",
      threads: 1, cache: false, stateCache: false, gpuKernel: "auto", profile: false, submit: "await", signal,
      onProgress(event) {
        if (event.message !== "WebGPU failed; trying the low-power adapter") return;
        partial.webGpuInitialization.retrySuppressed = true;
        const first = partial.webGpuInitialization.firstError;
        check(first, "Private campaign suppressed retry without observing the first WebGPU initialization error");
        throw Object.assign(new Error(first.message), { code: first.code });
      } };
    const restoreLoadObservers = observePrivateLoad(partial);
    try { engine = await Kevala.load(loadOptions); } finally { restoreLoadObservers(); }
    plugin.createGpu = original;
    check(snapshot?.actualEngine.gpu === snapshot.trunk && engine.info.onPage === true && engine.info.backend === "webgpu" && engine.info.arch === "laya", "The loaded page engine or captured Laya trunk differs");
    check(engine.info.pack.cached === false && engine.info.pack.bytes === manifest.pack.bytes && engine.info.stateCache.enabled === false, "Pack or state cache policy differs");
    check(await sha256(text.encode(JSON.stringify(snapshot.header))) === manifest.pack.headerJsonSha256, "Loaded GPU model header differs from the verified pack");
    await snapshot.gpu.device.queue.onSubmittedWorkDone();
    check(!snapshot.gpu.lost, "GPU device was lost during load");
    const fallback = adapterFallbackMetadata(snapshot.gpu.adapter), info = snapshot.gpu.adapter.info || {};
    check(fallback.isFallbackAdapter !== true && fallback.fallbackAdapterSource !== "conflict" && !/swiftshader|llvmpipe|software/i.test(snapshot.gpu.name), "Software or conflicting WebGPU adapter");
    partial.adapter = { name: snapshot.gpu.name, vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
      ...fallback, features: [...snapshot.gpu.device.features].sort(), limits: { maxComputeWorkgroupStorageSize: snapshot.gpu.device.limits.maxComputeWorkgroupStorageSize } };
    partial.model = engine.info;
    const coordinatorState = () => ({ flavor: snapshot.actualEngine.flavor, apiFlavor: engine.info.flavor,
      tile: snapshot.actualEngine.coord.tile, apiCpuTiles: engine.info.cpuTiles });
    partial.coordinatorBefore = coordinatorState();
    partial.tuningBefore = tuningJson(snapshot.trunk);
    partial.stage = "default-off-golden-guard";
    partial.defaultOffGuard = await defaultOffGoldenGuard(engine, snapshot.trunk, await readJson("../tests/fixtures/golden.json", signal), signal);
    partial.stage = "fixed-private-protocol";
    const run = async (requests, { beforeDecide }) => {
      signal.throwIfAborted();
      check(requests.length === 1, "Private protocol requires exactly one request");
      beforeDecide(requests);
      return [await engine.decide(requests[0].state, requests[0].questions)];
    };
    const protocol = await runLayaGeGluProtocol({ trunk: snapshot.trunk, run, profilePaired: (options) => profilePaired(engine, options), loadOptions });
    partial.stage = "bookends";
    partial.tuningAfter = tuningJson(snapshot.trunk);
    partial.coordinatorAfter = coordinatorState();
    check(JSON.stringify(partial.coordinatorBefore) === JSON.stringify(partial.coordinatorAfter), "Default coordinator flavor or tile changed between runtime arms");
    check(partial.tuningBefore === partial.tuningAfter && !snapshot.gpu.lost, "Tuning changed or GPU device was lost");
    partial.resourcesAfter = await verifyResources(manifest, signal);
    partial.packAfter = await verifyPackHeader(manifest, signal);
    const result = { status: "done", done: true, backend: "webgpu", model: "laya", adapter: partial.adapter, threads: engine.info.threads, coordinator: partial.coordinatorBefore,
      correctness: { ok: true, defaultOff: partial.defaultOffGuard, prototype: protocol.validation }, protocol, acceptance: acceptance(protocol), diagnostics: partial,
      metricMs: median(protocol.quartets.flatMap((q) => q.samples.filter((s) => s.lane === "A").map((s) => s.wallMs))),
      metric: "baseline unprofiled complete-request median milliseconds; no promotion" };
    publish(result);
  } catch (error) {
    publish({ status: "error", done: true, backend: "webgpu", error: String(error?.message || error), code: error.code || null,
      correctness: { ok: false }, diagnostics: partial, defaultOffPartial: error.defaultOffPartial || null,
      protocolPartial: error.partialResult || null, pairedKernelPartial: error.pairedKernelPartial || null });
  } finally { clearTimeout(deadline); plugin.createGpu = original; engine?.dispose(); }
}

runPage();
