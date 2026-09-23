import { kernelSource } from "./wgsl.js";
import { pipeline, requestDevice } from "../js/src/gpu.js";

const logNode = document.getElementById("log");
const log = (line) => (logNode.textContent += `${line}\n`);
const query = new URLSearchParams(location.search);
const NO_STATE = 0xffffffff;
const STATE_STRIDE = 16384;
const HEAD_DIM = 128;

const plain = Object.freeze({ hidden: 256, heads: 2, kv_heads: 1, lin_key_heads: 2, lin_heads: 2, rotary: 64 });
const grouped = Object.freeze({ hidden: 256, heads: 4, kv_heads: 2, lin_key_heads: 2, lin_heads: 4, rotary: 64 });

const CASES = [
  {
    name: "stage0-empty-short",
    cfg: plain,
    stage: 0,
    cpu: true,
    segments: [
      { len: 0, parent: NO_STATE, dst: 0 },
      { len: 1, parent: NO_STATE, dst: 0 },
      { len: 7, parent: NO_STATE, dst: 0 },
      { len: 31, parent: NO_STATE, dst: 0 },
      { len: 129, parent: NO_STATE, dst: 0 },
    ],
  },
  {
    name: "stage1-cached-grouped",
    cfg: grouped,
    stage: 1,
    cpu: true,
    segments: [
      { len: 7, parent: 1, dst: 2 },
      { len: 31, parent: 1, dst: 3 },
    ],
  },
  {
    name: "stage0-cached-129",
    cfg: grouped,
    stage: 0,
    cpu: true,
    segments: [{ len: 129, parent: 1, dst: 0 }],
  },
  {
    name: "stage1-long-528",
    cfg: plain,
    stage: 1,
    cpu: false,
    segments: [{ len: 528, parent: 1, dst: 2 }],
  },
];

function numberParam(name, fallback, minimum, maximum) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  if (!Number.isInteger(value)) return fallback;
  return Math.max(minimum, Math.min(maximum, value));
}

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
}

const f32 = Math.fround;
const add = (a, b) => f32(a + b);
const sub = (a, b) => f32(a - b);
const mul = (a, b) => f32(a * b);

function dimensions(cfg) {
  const qk = cfg.lin_key_heads * HEAD_DIM;
  const out = cfg.lin_heads * HEAD_DIM;
  return { qk, out, dim: 2 * qk + out };
}

function makeCaseData(test, seed) {
  const { qk, out, dim } = dimensions(test.cfg);
  const T = test.segments.reduce((sum, segment) => sum + segment.len, 0);
  const C = new Float32Array(Math.max(1, T * dim));
  const AB = new Float32Array(Math.max(1, T * 2 * test.cfg.lin_heads));
  const random = rng(seed);
  // Keep q/k products small enough that f32 and CPU reference differences stay visible.
  for (let i = 0; i < C.length; i++) C[i] = f32((random() - 0.5) * 0.08);
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < test.cfg.lin_heads; h++) {
      const base = t * 2 * test.cfg.lin_heads + h;
      AB[base] = f32(0.9 + random() * 0.1);
      AB[base + test.cfg.lin_heads] = f32(random());
    }
  }
  const stateSlots = 4;
  const STATE = new Float32Array(stateSlots * test.cfg.lin_heads * STATE_STRIDE);
  for (let i = 0; i < STATE.length; i++) STATE[i] = f32((random() - 0.5) * 0.04);
  const segs = new Uint32Array(Math.max(8, test.segments.length * 8));
  let start = 0;
  test.segments.forEach((segment, index) => {
    segs.set([start, segment.len, segment.parent, 0, 0, segment.dst, 0, 0], index * 8);
    start += segment.len;
  });
  return { T, qk, out, dim, C, AB, STATE, segs, stateSlots };
}

function cpuReference(test, data) {
  const { cfg, stage } = test;
  const { T, qk, out, dim, C, AB, STATE, segs } = data;
  const expectedCore = new Float32Array(Math.max(1, T * out));
  const expectedState = STATE.slice();
  const ratio = cfg.lin_heads / cfg.lin_key_heads;
  for (let si = 0; si < test.segments.length; si++) {
    const start = segs[si * 8];
    const len = segs[si * 8 + 1];
    const parent = segs[si * 8 + 2];
    const dst = segs[si * 8 + 5];
    if (!len) continue;
    for (let h = 0; h < cfg.lin_heads; h++) {
      const kh = Math.floor(h / ratio);
      for (let half = 0; half < 2; half++) {
        for (let column = half * 64; column < (half + 1) * 64; column++) {
          const lanes = [];
          for (let lane = 0; lane < 4; lane++) {
            const k0 = lane * 32;
            const col = new Float32Array(32);
            const base = ((parent === NO_STATE ? 0 : (parent * cfg.lin_heads + h) * STATE_STRIDE) + column);
            for (let i = 0; i < 32; i++) col[i] = parent === NO_STATE ? 0 : STATE[base + (k0 + i) * HEAD_DIM];
            lanes.push({ k0, col });
          }
          for (let r = 0; r < len; r++) {
            const t = start + r;
            const qBase = t * dim + kh * HEAD_DIM;
            const kBase = t * dim + qk + kh * HEAD_DIM;
            const v = C[t * dim + 2 * qk + h * HEAD_DIM + column];
            const ab = t * 2 * cfg.lin_heads + h;
            const decay = AB[ab];
            const beta = AB[ab + cfg.lin_heads];
            const partial = [];
            for (const lane of lanes) {
              let dot = 0;
              for (let i = 0; i < 32; i++) {
                dot = add(dot, mul(lane.col[i], C[kBase + lane.k0 + i]));
              }
              partial.push(dot);
            }
            const kv = add(add(partial[0], partial[1]), add(partial[2], partial[3]));
            const delta = mul(sub(v, mul(decay, kv)), beta);
            const outputs = [];
            for (const lane of lanes) {
              let o = 0;
              for (let i = 0; i < 32; i++) {
                const x = add(mul(decay, lane.col[i]), mul(C[kBase + lane.k0 + i], delta));
                lane.col[i] = x;
                o = add(o, mul(x, C[qBase + lane.k0 + i]));
              }
              outputs.push(o);
            }
            expectedCore[t * out + h * HEAD_DIM + column] = add(add(outputs[0], outputs[1]), add(outputs[2], outputs[3]));
          }
          if (stage === 1) {
            const stateBase = (dst * cfg.lin_heads + h) * STATE_STRIDE + column;
            for (const lane of lanes) for (let i = 0; i < 32; i++) expectedState[stateBase + (lane.k0 + i) * HEAD_DIM] = lane.col[i];
          }
        }
      }
    }
  }
  return { core: expectedCore, state: expectedState };
}

function maxDifference(actual, expected) {
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  let exact = actual.length === expected.length;
  for (let i = 0; i < Math.min(actual.length, expected.length); i++) {
    const abs = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) {
      exact = false;
      maxAbs = Infinity;
      worst = { index: i, actual: actual[i], expected: expected[i] };
      continue;
    }
    const relative = abs / Math.max(1, Math.abs(expected[i]));
    if (abs !== 0) exact = false;
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index: i, actual: actual[i], expected: expected[i] };
    }
    maxRelative = Math.max(maxRelative, relative);
  }
  if (actual.length !== expected.length) exact = false;
  return { exact, maxAbs, maxRelative, worst };
}

function cpuCheck(actual, expected) {
  const diff = maxDifference(actual, expected);
  let expectedScale = 1;
  for (const value of expected) {
    if (!Number.isFinite(value)) return { ok: false, ...diff };
    expectedScale = Math.max(expectedScale, Math.abs(value));
  }
  const ok = diff.maxAbs <= 2e-5 + 2e-5 * expectedScale;
  return { ok, ...diff };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function adapterInfo(gpu) {
  const info = gpu.adapter.info || {};
  return {
    name: gpu.name,
    vendor: info.vendor || null,
    architecture: info.architecture || null,
    device: info.device || null,
    description: info.description || null,
    isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter),
    features: [...gpu.device.features].sort(),
    limits: {
      maxBufferSize: gpu.device.limits.maxBufferSize,
      maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: gpu.device.limits.maxComputeWorkgroupStorageSize,
    },
    powerPreference: gpu.powerPreference,
  };
}

async function withGpuErrors(device, lost, phase, work, uncaptured) {
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let value;
  let failure;
  try {
    value = await work();
  } catch (error) {
    failure = error;
  }
  const scopes = await Promise.allSettled([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  const errors = scopes.filter((result) => result.status === "fulfilled" && result.value).map((result) => result.value);
  if (errors.length) throw new Error(`WebGPU ${phase}: ${errors.map((error) => error.message).join("; ")}`);
  if (failure) throw failure;
  if (uncaptured.length) throw new Error(`WebGPU ${phase}: uncaptured error: ${uncaptured.splice(0).join("; ")}`);
  if (lost.value) throw new Error(`WebGPU ${phase}: device lost: ${lost.value.message || lost.value.reason || "unknown reason"}`);
  return value;
}

function makeBuffer(device, bytes, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, bytes), usage });
}

async function readBuffer(device, source, bytes, lost, uncaptured, label) {
  const readback = makeBuffer(device, bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, `${label}.readback`);
  try {
    return await withGpuErrors(device, lost, `${label} readback`, async () => {
      const encoder = device.createCommandEncoder({ label: `${label}.readback` });
      encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ, 0, bytes);
      const values = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
      readback.unmap();
      return values;
    }, uncaptured);
  } finally {
    readback.destroy();
  }
}

function makeTimer(device) {
  if (!device.features.has("timestamp-query")) throw new Error("recurrence guard requires timestamp-query");
  return {
    querySet: device.createQuerySet({ type: "timestamp", count: 2, label: "recur-check.timestamps" }),
    resolved: makeBuffer(device, 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, "recur-check.timestamp-resolved"),
    read: makeBuffer(device, 256, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "recur-check.timestamp-read"),
  };
}

async function dispatch(device, pipelineObject, bindGroup, segments, heads, timer, lost, uncaptured, label, { reps = 1, zTiles = 2 } = {}) {
  return withGpuErrors(device, lost, label, async () => {
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass(timer ? { timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    pass.setPipeline(pipelineObject);
    pass.setBindGroup(0, bindGroup);
    for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(segments, heads, zTiles);
    pass.end();
    if (timer) {
      encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolved, 0);
      encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (!timer) return null;
    await timer.read.mapAsync(GPUMapMode.READ, 0, 16);
    const stamps = new BigInt64Array(timer.read.getMappedRange(0, 16).slice(0));
    timer.read.unmap();
    return Number(stamps[1] - stamps[0]) / 1e6 / reps;
  }, uncaptured);
}

async function main() {
  window.gpuBench = { status: "running", backend: "webgpu" };
  let gpu;
  let timer;
  const owned = [];
  const uncaptured = [];
  const lost = { value: null };
  try {
    gpu = await requestDevice();
    gpu.device.lost.then((info) => (lost.value = info));
    gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
    const device = gpu.device;
    const subgroupParam = query.get("subgroups");
    if (subgroupParam !== null && subgroupParam !== "0" && subgroupParam !== "1") {
      throw new Error(`subgroups must be 0 or 1, got ${subgroupParam}`);
    }
    const useSubgroups = subgroupParam === "1";
    if (useSubgroups && (!device.features.has("subgroups") || !gpu.subgroup4)) {
      throw new Error("subgroups=1 requested, but this device has no usable subgroup4 path");
    }
    const candidate = query.get("candidate") || "current";
    const baselineCandidate = query.get("baselineCandidate") || "current";
    const widths = { current: 4, recur8: 8, recur16: 16 };
    for (const name of [candidate, baselineCandidate]) {
      if (!Object.hasOwn(widths, name)) throw new Error(`unknown recurrence candidate ${name}`);
      if (useSubgroups && (gpu.adapter.info?.subgroupMinSize || 0) < widths[name]) {
        throw new Error(`${name} needs subgroups of at least ${widths[name]} lanes; use subgroups=0 for the shared path`);
      }
    }
    const timingHeadsParam = query.get("timingHeads");
    let timingHeads = null;
    let cases = CASES;
    if (timingHeadsParam !== null) {
      const parsed = Number(timingHeadsParam);
      if (!Number.isInteger(parsed) || ![16, 32, 48].includes(parsed)) {
        throw new Error(`timingHeads must be 16, 32, or 48, got ${timingHeadsParam}`);
      }
      timingHeads = parsed;
      const timingCfg = Object.freeze({ ...plain, lin_key_heads: Math.min(16, parsed), lin_heads: parsed });
      cases = CASES.map((test) => test.name === "stage1-long-528" ? { ...test, cfg: timingCfg } : test);
    }
    timer = makeTimer(device);
    owned.push(timer.resolved, timer.read);
    const current = await kernelSource();
    const baselineParam = query.get("baselineWasm");
    const baselineUrl = baselineParam ? new URL(baselineParam, document.baseURI).href : null;
    const baseline = baselineUrl ? await kernelSource(baselineUrl) : null;
    const samples = numberParam("samples", 3, 1, 10);
    const warmups = numberParam("warmups", 1, 0, 5);
    const pipelineCache = new Map();
    const getPipeline = async (source, cfg) => {
      const sourceKey = source === current ? "current" : "baseline";
      const lanes = widths[source === current ? candidate : baselineCandidate];
      const kernel = `kev_recur_lanes${lanes === 4 ? "" : lanes}`;
      const zTiles = lanes / 2;
      const cfgKey = `${sourceKey}|${kernel}|${JSON.stringify(cfg)}`;
      if (!pipelineCache.has(cfgKey)) {
        const code = source(kernel, { subgroups: useSubgroups, kev: cfg });
        pipelineCache.set(cfgKey, pipeline(device, code, `${kernel}.${sourceKey}.${cfg.lin_heads}`));
      }
      return { pipeline: await pipelineCache.get(cfgKey), kernel, zTiles };
    };
    const timing = [];
    const correctness = { ok: true, cpu: [], variants: [], errors: [] };
    let cpuMaxAbs = 0;
    let oldNewExact = baseline ? true : null;
    let oldNewWithinTolerance = baseline ? true : null;
    for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
      const test = cases[caseIndex];
      const data = makeCaseData(test, 0x4b455641 + caseIndex * 0x9e3779b9);
      const expected = test.cpu ? cpuReference(test, data) : null;
      const T = data.T;
      const heads = test.cfg.lin_heads;
      const segCount = test.segments.length;
      const input = makeBuffer(device, data.C.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, `${test.name}.C`);
      const ab = makeBuffer(device, data.AB.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, `${test.name}.AB`);
      const segs = makeBuffer(device, data.segs.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, `${test.name}.segs`);
      const state = makeBuffer(device, data.STATE.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, `${test.name}.STATE`);
      const core = makeBuffer(device, Math.max(16, T * data.out * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, `${test.name}.CORE`);
      const globals = makeBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, `${test.name}.globals`);
      owned.push(input, ab, segs, state, core, globals);
      const params = new Uint32Array([T, 0, segCount, test.stage]);
      const upload = async () => withGpuErrors(device, lost, `${test.name} upload`, async () => {
        device.queue.writeBuffer(input, 0, data.C);
        device.queue.writeBuffer(ab, 0, data.AB);
        device.queue.writeBuffer(segs, 0, data.segs);
        device.queue.writeBuffer(state, 0, data.STATE);
        device.queue.writeBuffer(core, 0, new Float32Array(Math.max(1, T * data.out)));
        device.queue.writeBuffer(globals, 0, params);
        await device.queue.onSubmittedWorkDone();
      }, uncaptured);
      const run = async (source, label, collectOutput) => {
        const variant = await getPipeline(source, test.cfg);
        const p = variant.pipeline;
        await upload();
        const bind = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [globals, input, ab, segs, state, core].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        await dispatch(device, p, bind, segCount, heads, null, lost, uncaptured, `${test.name}.${label}.correctness`, { zTiles: variant.zTiles });
        if (!collectOutput) return { p, bind, kernel: variant.kernel, zTiles: variant.zTiles };
        const coreOut = await readBuffer(device, core, Math.max(1, T * data.out) * 4, lost, uncaptured, `${test.name}.${label}.CORE`);
        const stateOut = await readBuffer(device, state, data.STATE.byteLength, lost, uncaptured, `${test.name}.${label}.STATE`);
        return { p, bind, kernel: variant.kernel, zTiles: variant.zTiles, core: coreOut, state: stateOut };
      };
      const currentResult = await run(current, "current", true);
      const currentCpu = expected ? { core: cpuCheck(currentResult.core, expected.core), state: cpuCheck(currentResult.state, expected.state) } : null;
      if (currentCpu) {
        const max = Math.max(currentCpu.core.maxAbs, currentCpu.state.maxAbs);
        cpuMaxAbs = Math.max(cpuMaxAbs, max);
        correctness.cpu.push({ case: test.name, variant: "current", core: currentCpu.core, state: currentCpu.state, ok: currentCpu.core.ok && currentCpu.state.ok });
        if (!currentCpu.core.ok || !currentCpu.state.ok) correctness.ok = false;
      }
      let baselineResult = null;
      if (baseline) {
        baselineResult = await run(baseline, "baseline", true);
        const baselineCpu = expected ? { core: cpuCheck(baselineResult.core, expected.core), state: cpuCheck(baselineResult.state, expected.state) } : null;
        if (baselineCpu) {
          const max = Math.max(baselineCpu.core.maxAbs, baselineCpu.state.maxAbs);
          cpuMaxAbs = Math.max(cpuMaxAbs, max);
          correctness.cpu.push({ case: test.name, variant: "baseline", core: baselineCpu.core, state: baselineCpu.state, ok: baselineCpu.core.ok && baselineCpu.state.ok });
          if (!baselineCpu.core.ok || !baselineCpu.state.ok) correctness.ok = false;
        }
        const coreDiff = maxDifference(currentResult.core, baselineResult.core);
        const stateDiff = maxDifference(currentResult.state, baselineResult.state);
        const coreGuard = cpuCheck(currentResult.core, baselineResult.core);
        const stateGuard = cpuCheck(currentResult.state, baselineResult.state);
        const exact = coreDiff.exact && stateDiff.exact;
        const withinTolerance = coreGuard.ok && stateGuard.ok;
        correctness.variants.push({ case: test.name, exact, withinTolerance, core: coreDiff, state: stateDiff, coreGuard, stateGuard });
        oldNewExact &&= exact;
        oldNewWithinTolerance &&= withinTolerance;
        if (!(exact || ((candidate !== "current" || baselineCandidate !== "current") && withinTolerance))) correctness.ok = false;
      }
      const reps = T <= 7 ? 16 : T <= 31 ? 8 : T <= 129 ? 2 : 1;
      const currentSamples = [];
      const baselineSamples = [];
      for (let warmup = 0; warmup < warmups; warmup++) {
        await upload();
        await dispatch(device, currentResult.p, currentResult.bind, segCount, heads, null, lost, uncaptured, `${test.name}.current.warmup`, { reps, zTiles: currentResult.zTiles });
      }
      for (let sample = 0; sample < samples; sample++) {
        await upload();
        currentSamples.push(await dispatch(device, currentResult.p, currentResult.bind, segCount, heads, timer, lost, uncaptured, `${test.name}.current.sample`, { reps, zTiles: currentResult.zTiles }));
        if (baselineResult) {
          await upload();
          baselineSamples.push(await dispatch(device, baselineResult.p, baselineResult.bind, segCount, heads, timer, lost, uncaptured, `${test.name}.baseline.sample`, { reps, zTiles: baselineResult.zTiles }));
        }
      }
      const currentMedian = median(currentSamples);
      timing.push({ case: test.name, T, samples: currentSamples, medianMs: currentMedian, baselineSamples: baselineResult ? baselineSamples : undefined, baselineMedianMs: baselineResult ? median(baselineSamples) : undefined });
      log(`${test.name}: current ${currentMedian.toFixed(4)} ms${baselineResult ? `, baseline ${median(baselineSamples).toFixed(4)} ms` : ""}`);
    }
    const metricMs = Math.exp(timing.reduce((sum, item) => sum + Math.log(item.medianMs), 0) / timing.length);
    const result = {
      status: "done",
      backend: "webgpu",
      metricMs,
      metric: "geometric mean of recurrence case median GPU milliseconds",
      method: "WebGPU timestamp-query",
      samples,
      warmups,
      baselineWasm: baselineUrl,
      candidate,
      baselineCandidate,
      timingHeads,
      variantProvenance: {
        current: { source: "current", wasmUrl: current.wasmUrl, kernel: `kev_recur_lanes${widths[candidate] === 4 ? "" : widths[candidate]}`, zTiles: widths[candidate] / 2, subgroups: useSubgroups },
        ...(baseline ? { baseline: { source: "baseline", wasmUrl: baseline.wasmUrl, kernel: `kev_recur_lanes${widths[baselineCandidate] === 4 ? "" : widths[baselineCandidate]}`, zTiles: widths[baselineCandidate] / 2, subgroups: useSubgroups } } : {}),
      },
      cases: timing,
      cpuMaxAbs,
      oldNewExact,
      oldNewWithinTolerance,
      correctness,
      adapter: adapterInfo(gpu),
    };
    window.gpuBench = result;
  } catch (error) {
    const message = `${error?.message || error}\n${error?.stack || ""}`;
    log(`ERROR ${message}`);
    window.gpuBench = { status: "error", backend: "webgpu", error: message, correctness: { ok: false, errors: [message] } };
  } finally {
    for (const buffer of owned) buffer.destroy();
    timer?.querySet.destroy();
    gpu?.device.destroy();
  }
}

main();
