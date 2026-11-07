import { FIXTURES, MEASURED_RUNS, ORDERS, QUARTETS, WARMUP_RUNS, canonicalJson, jsonSnapshot, requestFor } from "./fixtures.mjs";

const MODEL = "laya";
const REFERENCE_SHA256 = "2f6316f374500475038c3dab3280223c25b568cb4b061cd676e79eccb37b7375";
const RUNTIME = {
  name: "staged-runtime-v3",
  commit: "be350ec53b774bbfff3733d690cc9689625a02e7",
  identitySha256: "c5d3c308a5a0c5fd23f3d60f65093c21a20029adbb2a3672a6bbb4eb397eab4a",
  fingerprint: "eb46c3119c1befb08f1e63bcd3499583ed6684af00d9d6094caa845ca7754bc6",
  wasmSha256: "939f4c424900c251577b548bc61a9b7ccc4a7ef576468cf37967dd4dae2ef700",
};
const PACK = {
  bytes: 478886720,
  sha256: "b9cab75c38f026236b8d736377aa40f889d842c219d37da45cc06c782395555c",
  headerBytes: 26313,
  headerSha256: "f82cc6ce21f0b3f55000f505db8d7ee08f5f956ce9d1d1206607bf2c6476ecc8",
};
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const encoder = new TextEncoder();
const hex = (bytes) => [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
const sha256 = async (value) => hex(await crypto.subtle.digest("SHA-256", typeof value === "string" ? encoder.encode(value) : value));
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

function replaceInstanceMethod(target, name, makeWrapper, restores) {
  const descriptor = Object.getOwnPropertyDescriptor(target, name);
  if (descriptor && descriptor.configurable === false && descriptor.writable === false) {
    throw new Error(`cannot instrument ${name}: instance method is not replaceable`);
  }
  const original = target[name];
  if (typeof original !== "function") throw new Error(`cannot instrument ${name}: instance method is missing`);
  const wrapper = makeWrapper(original);
  Object.defineProperty(target, name, { configurable: true, enumerable: descriptor?.enumerable || false, writable: true, value: wrapper });
  if (target[name] !== wrapper) throw new Error(`cannot instrument ${name}: instance method replacement did not stick`);
  restores.push(() => {
    if (descriptor) Object.defineProperty(target, name, descriptor);
    else delete target[name];
  });
}

function selectedPlan(gpu, info, packPrecision) {
  const selected = gpu.tuning?.selection instanceof Map ? [...gpu.tuning.selection].sort(([a], [b]) => a.localeCompare(b)) : [];
  return {
    backend: info.backend,
    arch: info.arch,
    precision: packPrecision,
    gpuKernel: gpu.gpuKernel,
    subgroup32: gpu.subgroup32,
    attention: { kernel: gpu.attnKernel || null, queries: gpu.attnQueries },
    matmul: { f16: gpu.mm?.f16 ?? null, groups: gpu.mm?.groups ?? null, splitTarget: gpu.mm?.splitTarget ?? null, selection: selected },
    pipelines: Object.fromEntries(["pNorm", "pRope", "pAttn", "pGeglu", "pGather"].map((key) => [key, gpu[key]?.label || null])),
  };
}

async function browserFixturePreflight() {
  const checks = [];
  for (const fixture of FIXTURES) {
    for (let i = 0; i < MEASURED_RUNS.length; i++) {
      const run = MEASURED_RUNS[i];
      const request = requestFor(fixture, run);
      const rawRequestJson = JSON.stringify([request]);
      const actualHash = await sha256(rawRequestJson);
      const expectedHash = fixture.measuredHashes[i];
      if (actualHash !== expectedHash) throw new Error(`${fixture.id} request Run ${run} hash mismatch: expected ${expectedHash}, got ${actualHash}`);
      checks.push({ fixture: fixture.id, run, expectedHash, actualHash });
    }
  }
  if (checks.length !== 18) throw new Error(`fixture preflight expected 18 measured hashes, got ${checks.length}`);
  return { referenceSha256: REFERENCE_SHA256, checks };
}

async function verifyRuntime(entry) {
  const identityUrl = new URL("../staged-runtime-v3-identity.json", entry);
  const response = await fetch(identityUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`frozen runtime identity: HTTP ${response.status}`);
  const identityBytes = await response.arrayBuffer();
  const identity = JSON.parse(new TextDecoder().decode(identityBytes));
  if (await sha256(identityBytes) !== RUNTIME.identitySha256 || identity.sourceCommit !== RUNTIME.commit || identity.fingerprint !== RUNTIME.fingerprint) {
    throw new Error("staged-runtime-v3 identity differs from the frozen source receipt");
  }
  const wasmResponse = await fetch(new URL("kevala-relaxed.wasm", entry), { cache: "no-store" });
  if (!wasmResponse.ok) throw new Error(`frozen runtime WASM: HTTP ${wasmResponse.status}`);
  if (await sha256(await wasmResponse.arrayBuffer()) !== RUNTIME.wasmSha256) throw new Error("staged-runtime-v3 relaxed WASM hash changed");
  return { identityUrl: identityUrl.href, identitySha256: RUNTIME.identitySha256, fingerprint: identity.fingerprint, wasmSha256: RUNTIME.wasmSha256 };
}

function instrumentGpu(instance, waitEvents, getActive, restores, hiddenReturns) {
  const queue = instance.device?.queue;
  if (!queue) throw new Error("Laya GPU context has no queue");
  replaceInstanceMethod(queue, "onSubmittedWorkDone", (original) => function (...args) {
    const active = getActive();
    const started = performance.now();
    const record = (error = null) => {
      if (active) waitEvents.push({ requestId: active.id, kind: "queue.onSubmittedWorkDone", durationMs: performance.now() - started, error });
    };
    let result;
    try { result = original.apply(this, args); }
    catch (error) { record(String(error?.message || error)); throw error; }
    return Promise.resolve(result).then((value) => { record(); return value; }, (error) => { record(String(error?.message || error)); throw error; });
  }, restores);

  const hookedBuffers = new WeakSet();
  const instrumentReadback = (buffer) => {
    if (!buffer || hookedBuffers.has(buffer)) return;
    hookedBuffers.add(buffer);
    replaceInstanceMethod(buffer, "mapAsync", (original) => function (...args) {
      const active = getActive();
      const started = performance.now();
      const record = (error = null) => {
        if (active) waitEvents.push({ requestId: active.id, kind: "mapRead.mapAsync", durationMs: performance.now() - started, error });
      };
      let result;
      try { result = original.apply(this, args); }
      catch (error) { record(String(error?.message || error)); throw error; }
      return Promise.resolve(result).then((value) => { record(); return value; }, (error) => { record(String(error?.message || error)); throw error; });
    }, restores);
  };

  const ensure = instance.ensure;
  if (typeof ensure !== "function") throw new Error("Laya GPU context has no ensure method for readback instrumentation");
  replaceInstanceMethod(instance, "ensure", (original) => function (...args) {
    const result = original.apply(this, args);
    instrumentReadback(this.readback);
    return result;
  }, restores);
  instrumentReadback(instance.readback);

  const forward = instance.forward;
  if (typeof forward !== "function") throw new Error("Laya GPU context has no forward method for hidden-buffer capture");
  replaceInstanceMethod(instance, "forward", (original) => async function (...args) {
    const hidden = await original.apply(this, args);
    hiddenReturns.push(hidden);
    return hidden;
  }, restores);
}

let activeRequest = null;
let gpu = null;
let webgpu = null;
let kevala = null;
let loadedPack = null;
let SUBMIT = null;
let submitBefore = null;
const waitEvents = [];
const hiddenReturns = [];
const restores = [];
const rafGaps = [];
let rafActive = false;
let lastFrame = null;
let frameCount = 0;

function requestAnimationFrameLoop(timestamp) {
  if (lastFrame !== null) rafGaps.push({ from: lastFrame, to: timestamp, gapMs: timestamp - lastFrame });
  lastFrame = timestamp;
  frameCount++;
  if (rafActive) requestAnimationFrame(requestAnimationFrameLoop);
}

async function runCall(fixture, run, mode, tag, planJson) {
  const request = requestFor(fixture, run);
  const rawRequestJson = JSON.stringify([request]);
  const requestHash = await sha256(rawRequestJson);
  const measuredIndex = run - MEASURED_RUNS[0];
  if (measuredIndex >= 0 && requestHash !== fixture.measuredHashes[measuredIndex]) {
    throw new Error(`${fixture.id} request Run ${run} hash changed after preflight`);
  }
  const requestCapture = jsonSnapshot(request);
  const requestCanonicalHash = await sha256(requestCapture.json);
  const currentPlanJson = canonicalJson(selectedPlan(gpu, kevala.info, loadedPack.precision));
  if (currentPlanJson !== planJson) throw new Error("a non-scheduling execution plan changed during the replay");
  SUBMIT.mode = mode;
  if (SUBMIT.mode !== mode) throw new Error(`submission controller did not accept ${mode}`);

  const id = `${tag}:${fixture.id}:Run${run}:${mode}`;
  const beforeHidden = hiddenReturns.length;
  const frameStart = frameCount;
  const started = performance.now();
  activeRequest = { id, mode, fixture: fixture.id, run };
  let response;
  let finished;
  let frameEndCount;
  try {
    response = await kevala.decide(request.state, request.questions);
  } finally {
    finished = performance.now();
    frameEndCount = frameCount;
    activeRequest = null;
  }
  const wallMs = finished - started;
  const captured = hiddenReturns.length - beforeHidden;
  if (captured !== 1) throw new Error(`${fixture.id} Run ${run} returned ${captured} GPU hidden buffers, expected one`);
  if (response.timing?.tokens !== fixture.expectedTokens) throw new Error(`${fixture.id} Run ${run} token count ${response.timing?.tokens}, expected ${fixture.expectedTokens}`);
  const hidden = hiddenReturns[hiddenReturns.length - 1];
  const hiddenBytes = new Uint8Array(hidden.buffer, hidden.byteOffset, hidden.byteLength);
  const hiddenSha256 = await sha256(hiddenBytes);
  const { timing, ...publicValue } = response;
  const publicCapture = jsonSnapshot(publicValue);
  const publicSha256 = await sha256(publicCapture.json);
  if (canonicalJson(selectedPlan(gpu, kevala.info, loadedPack.precision)) !== planJson) throw new Error("a non-scheduling execution plan changed during the request");
  const callWaits = waitEvents.filter((event) => event.requestId === id);
  const queueWaits = callWaits.filter((event) => event.kind === "queue.onSubmittedWorkDone");
  const mapWaits = callWaits.filter((event) => event.kind === "mapRead.mapAsync");
  const expectedQueueWaits = fixture.expectedTokens > 256 && mode === "await" ? 5 : 0;
  if (queueWaits.length !== expectedQueueWaits) throw new Error(`${fixture.id} Run ${run} ${mode} recorded ${queueWaits.length} queue drains, expected ${expectedQueueWaits}`);
  if (mapWaits.length !== 1) throw new Error(`${fixture.id} Run ${run} recorded ${mapWaits.length} map-read waits, expected one`);
  const frameWindowClosed = await new Promise((resolve) => {
    let settled = false;
    const timeout = setTimeout(() => {
      settled = true;
      resolve(false);
    }, 250);
    requestAnimationFrame(() => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      resolve(true);
    });
  });
  const overlappingFrames = rafGaps.filter((gap) => gap.to >= started && gap.from <= finished);
  return {
    id,
    mode,
    run,
    requestHash,
    requestCanonicalJson: requestCapture.json,
    requestCanonicalHash,
    requestSnapshot: requestCapture.snapshot,
    wallMs,
    timing: { prepare: timing?.prepare ?? null, forward: timing?.forward ?? null, total: timing?.total ?? null, tokens: timing?.tokens ?? null },
    publicJson: publicCapture.json,
    publicSha256,
    publicSnapshot: publicCapture.snapshot,
    hiddenBuffer: { bytes: hidden.byteLength, floats: hidden.length, sha256: hiddenSha256 },
    waits: {
      queueOnSubmittedWorkDone: { count: queueWaits.length, totalMs: queueWaits.reduce((sum, item) => sum + item.durationMs, 0), events: queueWaits },
      mapRead: { count: mapWaits.length, totalMs: mapWaits.reduce((sum, item) => sum + item.durationMs, 0), events: mapWaits },
    },
    requestAnimationFrame: {
      frameCountDelta: frameEndCount - frameStart,
      closeStatus: frameWindowClosed ? "closed" : "timeout",
      maxGapMs: frameWindowClosed && overlappingFrames.length ? Math.max(...overlappingFrames.map((gap) => gap.gapMs)) : null,
      gaps: overlappingFrames,
    },
    planSha256: await sha256(currentPlanJson),
  };
}

function pairedSummary(calls) {
  const awaitCalls = calls.filter((call) => call.mode === "await");
  const splitCalls = calls.filter((call) => call.mode === "split");
  const awaitWallMs = awaitCalls.reduce((sum, call) => sum + call.wallMs, 0) / awaitCalls.length;
  const splitWallMs = splitCalls.reduce((sum, call) => sum + call.wallMs, 0) / splitCalls.length;
  const maxGap = (items) => {
    const gaps = items.map((item) => item.requestAnimationFrame.maxGapMs).filter(Number.isFinite);
    return gaps.length ? Math.max(...gaps) : null;
  };
  return {
    awaitWallMs,
    splitWallMs,
    splitToAwaitWallRatio: splitWallMs / awaitWallMs,
    awaitMaxFrameGapMs: maxGap(awaitCalls),
    splitMaxFrameGapMs: maxGap(splitCalls),
    awaitFrameCloseTimeouts: awaitCalls.filter((item) => item.requestAnimationFrame.closeStatus === "timeout").length,
    splitFrameCloseTimeouts: splitCalls.filter((item) => item.requestAnimationFrame.closeStatus === "timeout").length,
    awaitQueueWaitMs: awaitCalls.reduce((sum, call) => sum + call.waits.queueOnSubmittedWorkDone.totalMs, 0) / awaitCalls.length,
    splitQueueWaitMs: splitCalls.reduce((sum, call) => sum + call.waits.queueOnSubmittedWorkDone.totalMs, 0) / splitCalls.length,
  };
}

async function run() {
  window.bench = { done: false, status: "running" };
  let runtimeIdentity;
  let loadMs = null;
  let planCapture = null;
  let loadWarmupHidden = null;
  let partialCases = [];
  try {
    const fixturePreflight = await browserFixturePreflight();
    const runtimeEntry = new URL("../staged-runtime-v3/index.js", import.meta.url);
    runtimeIdentity = await verifyRuntime(runtimeEntry);
    const packHead = await fetch(new URL("../../laya-q8.kevala", import.meta.url), { method: "HEAD", cache: "no-store" });
    if (!packHead.ok || Number(packHead.headers.get("content-length")) !== PACK.bytes) throw new Error("Laya Q8 pack length differs from the pinned capture");
    const { Kevala } = await import(runtimeEntry.href);
    const { archPlugin } = await import(new URL("archs/index.js", runtimeEntry));
    const { packPrecision } = await import(new URL("pack-layout.js", runtimeEntry));
    ({ SUBMIT } = await import(new URL("gpu.js", runtimeEntry)));
    submitBefore = SUBMIT.mode;
    const plugin = archPlugin("laya");
    const originalCreateGpu = plugin.createGpu;
    plugin.createGpu = function captureGpu(context, layout, header, ...args) {
      webgpu = context;
      loadedPack = {
        model: header?.model?.name || null,
        precision: packPrecision(header),
        arch: header?.config?.arch || "laya",
      };
      gpu = originalCreateGpu.call(this, context, layout, header, ...args);
      instrumentGpu(gpu, waitEvents, () => activeRequest, restores, hiddenReturns);
      return gpu;
    };
    const loadStarted = performance.now();
    kevala = await Kevala.load({
      backend: "webgpu",
      onPage: true,
      model: { name: MODEL, url: new URL("../../laya-q8.kevala", import.meta.url).href },
      precision: "q8",
      flavor: "relaxed",
      threads: 1,
      cache: false,
      stateCache: false,
      gpuKernel: "auto",
      profile: false,
      submit: "await",
    });
    plugin.createGpu = originalCreateGpu;
    loadMs = performance.now() - loadStarted;
    if (!gpu || kevala.info.backend !== "webgpu" || !kevala.info.onPage || !navigator.gpu) throw new Error("frozen Laya engine did not load on-page with WebGPU");
    if (SUBMIT.mode !== "await") throw new Error("on-page engine and harness do not share the frozen SUBMIT module instance");
    if (loadedPack?.model !== MODEL || loadedPack.precision !== "q8" || loadedPack.arch !== "laya" || kevala.info.pack?.bytes !== PACK.bytes) {
      throw new Error(`loaded Laya pack header differs from the pin: ${JSON.stringify(loadedPack)}`);
    }
    if (hiddenReturns.length !== 1) throw new Error(`runtime load warmup returned ${hiddenReturns.length} GPU hidden buffers, expected one`);
    const warmHidden = hiddenReturns[0];
    loadWarmupHidden = { bytes: warmHidden.byteLength, floats: warmHidden.length, sha256: await sha256(new Uint8Array(warmHidden.buffer, warmHidden.byteOffset, warmHidden.byteLength)) };
    hiddenReturns.length = 0;
    waitEvents.length = 0;
    const hardware = {
      browser: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency || null,
      crossOriginIsolated: globalThis.crossOriginIsolated === true,
      adapter: webgpu?.adapter?.info || null,
      features: [...gpu.device.features].sort(),
      limits: { maxBufferSize: gpu.device.limits.maxBufferSize, maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: gpu.device.limits.maxComputeWorkgroupStorageSize },
    };
    planCapture = jsonSnapshot(selectedPlan(gpu, kevala.info, loadedPack.precision));
    const planSha256 = await sha256(planCapture.json);
    rafActive = true;
    requestAnimationFrame(requestAnimationFrameLoop);

    const warmupModes = ["await", "split", "await"];
    for (const fixture of FIXTURES) {
      const warmups = [];
      for (let i = 0; i < WARMUP_RUNS.length; i++) {
        warmups.push(await runCall(fixture, WARMUP_RUNS[i], warmupModes[i], `warmup-${i}`, planCapture.json));
      }
      const quartets = [];
      for (let quartet = 0; quartet < QUARTETS; quartet++) {
        const order = ORDERS[quartet % ORDERS.length];
        const runIndex = MEASURED_RUNS[quartet];
        const calls = [];
        for (let position = 0; position < order.length; position++) {
          calls.push(await runCall(fixture, runIndex, order[position], `q${quartet}-p${position}`, planCapture.json));
        }
        quartets.push({ quartet, run: runIndex, order, paired: pairedSummary(calls), calls });
        log(`${fixture.id}: quartet ${quartet + 1}/${QUARTETS} complete`);
      }
      const samples = quartets.flatMap((quartet) => quartet.calls.map(({ wallMs, mode, requestHash }) => ({ wallMs, mode, requestHash })));
      const wallSamples = samples.map((sample) => sample.wallMs);
      partialCases.push({
        id: fixture.id,
        name: fixture.name,
        expectedTokens: fixture.expectedTokens,
        p50: median(wallSamples),
        samples,
        warmups,
        quartets,
      });
    }
    const caseRows = partialCases.map(({ id, name, expectedTokens, p50, samples, warmups, quartets }) => ({ id, name, expectedTokens, p50, samples, warmups, quartets }));
    window.bench = {
      done: true,
      status: "done",
      backend: kevala.info.backend,
      threads: kevala.info.threads,
      model: MODEL,
      profile: false,
      runtime: { ...RUNTIME, entry: runtimeEntry.href, identity: runtimeIdentity, loadMs },
      pack: { ...PACK, url: new URL("../../laya-q8.kevala", import.meta.url).href, loadedHeader: loadedPack, header: kevala.info.pack },
      engine: { singleResident: true, onPage: true, submitModuleShared: true, hardware },
      protocol: { comparison: "SUBMIT.mode await vs split on one resident engine", warmupRuns: [...WARMUP_RUNS], measuredRuns: [...MEASURED_RUNS], quartets: QUARTETS, orders: ORDERS, profile: false, queueFence: "No extra queue fence; only the runtime's onSubmittedWorkDone and MAP_READ.mapAsync calls are instrumented." },
      fixturePreflight: { ...fixturePreflight, referenceSha256: REFERENCE_SHA256 },
      loadWarmupHidden,
      fixedPlan: { json: planCapture.json, sha256: planSha256, snapshot: planCapture.snapshot, rule: "selected plan canonical JSON and hash must stay unchanged before every request" },
      instrumentation: { scope: "GpuTrunk, GPUQueue, and MAP_READ GPUBuffer instance methods only; no prototype patches", queueMethod: "GPUQueue.onSubmittedWorkDone", mapMethod: "GpuTrunk.readback.mapAsync", hiddenOutput: "complete Float32Array returned by GpuTrunk.forward, hashed after decide wall timing" },
      correctness: { ok: true, strictWithinQuartet: true, allRequestTokensExact: true, fixedPlan: true, waitCountsExpected: true },
      cases: caseRows,
    };
  } catch (error) {
    window.bench = { done: true, status: "error", backend: kevala?.info?.backend || null, threads: kevala?.info?.threads || null, error: String(error?.message || error), partialCases };
    log(`ERROR ${window.bench.error}`);
  } finally {
    rafActive = false;
    activeRequest = null;
    if (SUBMIT && submitBefore) SUBMIT.mode = submitBefore;
    for (const restore of restores.splice(0).reverse()) {
      try { restore(); } catch {}
    }
    try { await kevala?.profile(false); } catch {}
    kevala?.dispose();
  }
}

run();
