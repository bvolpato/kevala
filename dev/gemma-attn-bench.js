// Paired guard for production Gemma attention versus a compile-time head dimension.
// The case format is T x query-heads x KV-heads x head-dim x window x causal. The
// shader has a 256-lane workgroup and supports the 256 and 512 dimensional heads
// used by Gemma 4. Q is already prepared, so this benchmark deliberately does not
// apply an attention scale. Small even dimensions are included in the default
// guard to exercise partial qtile loads as well as the production 256/512 cases.

import { kernelSource } from "./wgsl.js";
import { pipeline, requestDevice } from "../js/src/gpu.js";

const query = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => {
  logNode.textContent += `${line}\n`;
};
const state = { status: "running", backend: "webgpu", kernel: "gemma4_attention" };
window.gpuBench = state;

let U;
let activeDevice = null;

const DEFAULT_CASES = [
  { T: 1, heads: 1, kvHeads: 1, dim: 2, window: 8, causal: 1 },
  { T: 5, heads: 4, kvHeads: 2, dim: 254, window: 2, causal: 1 },
  { T: 9, heads: 4, kvHeads: 1, dim: 258, window: 32, causal: 0 },
  { T: 3, heads: 8, kvHeads: 1, dim: 256, window: 0, causal: 1 },
  { T: 7, heads: 8, kvHeads: 2, dim: 256, window: 4, causal: 1 },
  { T: 11, heads: 8, kvHeads: 4, dim: 512, window: 0, causal: 0 },
  { T: 17, heads: 16, kvHeads: 2, dim: 512, window: 5, causal: 1 },
];

const PRODUCTION_CASES = [99, 193, 601].flatMap((T) => [
  { T, heads: 8, kvHeads: 1, dim: 256, window: 512, causal: 1 },
  { T, heads: 8, kvHeads: 1, dim: 512, window: 0, causal: 1 },
]);

function specializeHeadDim(code, dim) {
  if (!Number.isInteger(dim) || dim < 1 || dim > 512) throw new Error("head dimension must be an integer from 1 to 512");
  const declaration = "let H = p.head_dim;";
  if (code.split(declaration).length !== 2) throw new Error("Gemma shader must contain exactly one dynamic head-dimension declaration");
  return code.replace(declaration, `const H = ${dim}u;`);
}

function intParam(name, fallback, min, max) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  return Number.isInteger(value) && value >= min && value <= max ? value : fallback;
}

function floatParam(name, fallback, min, max) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
}

function parseCausal(value) {
  if (value === "1" || value === "true" || value === "causal") return 1;
  if (value === "0" || value === "false" || value === "bidirectional" || value === "bi") return 0;
  throw new Error(`causal must be 0/1, causal, or bidirectional; got ${value}`);
}

function labelCase(c) {
  return `${c.T}x${c.heads}x${c.kvHeads}x${c.dim}x${c.window}x${c.causal}`;
}

function parseCases() {
  const raw = query.get("cases");
  if (raw === "production") return PRODUCTION_CASES.map((c) => ({ ...c, label: labelCase(c) }));
  if (raw === "all") return [...DEFAULT_CASES, ...PRODUCTION_CASES].map((c) => ({ ...c, label: labelCase(c) }));
  if (!raw || raw === "default") return DEFAULT_CASES.map((c) => ({ ...c, label: labelCase(c) }));
  if (raw === "fast") return [DEFAULT_CASES[3], DEFAULT_CASES[5]].map((c) => ({ ...c, label: labelCase(c) }));
  const cases = raw.split(",").map((item) => {
    const fields = item.trim().split("x");
    if (fields.length !== 6) throw new Error("cases must be comma-separated T x heads x kvHeads x dim x window x causal shapes");
    const [T, heads, kvHeads, dim, window] = fields.slice(0, 5).map(Number);
    const causal = parseCausal(fields[5]);
    if (![T, heads, kvHeads, dim, window].every(Number.isInteger) || T < 1 || T > 65535 || heads < 1 || heads > 1024 || kvHeads < 1 || kvHeads > heads || heads % kvHeads !== 0 || window < 0 || window > 0xffffffff || dim < 1 || dim > 512) {
      throw new Error(`invalid attention case ${item}; expected positive T, divisible heads/kvHeads, dim 1..512, and a u32 window`);
    }
    const c = { T, heads, kvHeads, dim, window, causal };
    return { ...c, label: labelCase(c) };
  });
  if (!cases.length) throw new Error("cases must contain at least one case");
  return cases;
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s ^= s >>> 16;
    return (s >>> 0) / 0x100000000;
  };
}

function makeInputs(c, seed, magnitude) {
  const random = rng(seed);
  const q = new Float32Array(c.T * c.heads * c.dim);
  const k = new Float32Array(c.T * c.kvHeads * c.dim);
  const v = new Float32Array(c.T * c.kvHeads * c.dim);
  // Keep scores in a numerically quiet range by default. A magnitude query
  // parameter can deliberately create peaked softmax distributions.
  for (const values of [q, k, v]) for (let i = 0; i < values.length; i++) values[i] = Math.fround((random() - 0.5) * magnitude);
  return { q, k, v };
}

function attentionRange(c, t) {
  const start = c.window > 0 ? t + 1 - Math.min(t + 1, c.window) : 0;
  const end = c.causal ? t + 1 : c.T;
  return [start, end];
}

// Keep the reference mathematically independent from the production reduction:
// JavaScript's Number gives a straightforward float64 dot over every dimension.
function dotReference(q, k, qbase, kbase, dim) {
  let sum = 0;
  for (let d = 0; d < dim; d++) sum += q[qbase + d] * k[kbase + d];
  return sum;
}

function samplePositions(size) {
  return [...new Set([0, 1, Math.floor(size / 2), Math.max(0, size - 2), size - 1])].filter((x) => x >= 0 && x < size);
}

function cpuTargets(c) {
  // Full checks cover all outputs for normal guard cases. Long cases retain all
  // dimensions for representative queries and heads, keeping a 4096-token guard
  // useful without turning page load into a second attention implementation run.
  const full = c.T <= 128 && c.T * c.heads * c.dim <= 524288;
  return {
    full,
    tokens: full ? Array.from({ length: c.T }, (_, i) => i) : samplePositions(c.T),
    heads: full ? Array.from({ length: c.heads }, (_, i) => i) : samplePositions(c.heads),
  };
}

function cpuReference(c, input) {
  const targets = cpuTargets(c);
  const checks = [];
  for (const t of targets.tokens) {
    const [start, end] = attentionRange(c, t);
    const scores = new Float64Array(end - start);
    let max = -Infinity;
    for (const h of targets.heads) {
      const kvHead = Math.floor(h / (c.heads / c.kvHeads));
      const qbase = (t * c.heads + h) * c.dim;
      max = -Infinity;
      for (let j = start; j < end; j++) {
        const score = dotReference(input.q, input.k, qbase, (j * c.kvHeads + kvHead) * c.dim, c.dim);
        scores[j - start] = score;
        max = Math.max(max, score);
      }
      const weights = new Float64Array(scores.length);
      let denom = 0;
      for (let j = 0; j < scores.length; j++) {
        weights[j] = Math.exp(scores[j] - max);
        denom += weights[j];
      }
      for (let d = 0; d < c.dim; d++) {
        let value = 0;
        for (let j = start; j < end; j++) {
          value += (weights[j - start] / denom) * input.v[(j * c.kvHeads + kvHead) * c.dim + d];
        }
        checks.push({ index: (t * c.heads + h) * c.dim + d, expected: value });
      }
    }
  }
  return { ...targets, checks };
}

function compareCpu(values, reference) {
  const tolerance = { absolute: 1e-5, relative: 1e-5 };
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) return { ok: false, samples: reference.checks.length, message: `non-finite output at index ${index}` };
  }
  for (const { index, expected } of reference.checks) {
    const actual = values[index];
    if (!Number.isFinite(expected)) return { ok: false, samples: reference.checks.length, message: `non-finite reference at index ${index}` };
    const abs = Math.abs(actual - expected);
    const relative = abs / Math.max(1, Math.abs(expected));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index, expected, actual };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > tolerance.absolute + tolerance.relative * Math.max(1, Math.abs(expected))) {
      return { ok: false, samples: reference.checks.length, maxAbs, maxRelative, worst, message: `CPU mismatch at output index ${index}` };
    }
  }
  return { ok: true, samples: reference.checks.length, maxAbs, maxRelative, worst, tolerance };
}

function compareOutputs(dynamic, compiled) {
  if (dynamic.length !== compiled.length) return { ok: false, samples: Math.max(dynamic.length, compiled.length), message: "variant output lengths differ" };
  const tolerance = { absolute: 1e-5, relative: 1e-5 };
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  for (let index = 0; index < dynamic.length; index++) {
    if (!Number.isFinite(dynamic[index]) || !Number.isFinite(compiled[index])) return { ok: false, samples: dynamic.length, message: `non-finite variant output at index ${index}` };
    const abs = Math.abs(dynamic[index] - compiled[index]);
    const relative = abs / Math.max(1, Math.abs(dynamic[index]), Math.abs(compiled[index]));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index, dynamic: dynamic[index], compiled: compiled[index] };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > tolerance.absolute + tolerance.relative * Math.max(1, Math.abs(dynamic[index]), Math.abs(compiled[index]))) {
      return { ok: false, samples: dynamic.length, maxAbs, maxRelative, worst, tolerance, message: `variant mismatch at output index ${index}` };
    }
  }
  return { ok: true, samples: dynamic.length, maxAbs, maxRelative, worst, tolerance };
}

function makeBuffer(device, size, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage });
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

async function checked(device, operation, phase, uncaptured) {
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let value;
  let failure;
  try {
    value = await operation();
  } catch (error) {
    failure = error;
  }
  const scopes = await Promise.allSettled([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  const errors = scopes.filter((item) => item.status === "fulfilled" && item.value).map((item) => item.value);
  if (errors.length) throw new Error(`WebGPU ${phase}: ${errors.map((error) => error.message).join("; ")}`);
  if (failure) throw failure;
  if (uncaptured.length) throw new Error(`WebGPU ${phase}: ${uncaptured.splice(0).map((item) => item.message).join("; ")}`);
  if (activeDevice?.lost) throw new Error(`WebGPU ${phase}: device lost: ${activeDevice.lost.message || activeDevice.lost.reason || "unknown reason"}`);
  return value;
}

function makeTimer(device) {
  if (!device.features.has("timestamp-query")) return { timestamp: false, method: "performance.now() wall clock" };
  const slots = 2;
  return {
    timestamp: true,
    method: "WebGPU timestamp-query",
    slots,
    querySet: device.createQuerySet({ type: "timestamp", count: slots * 2, label: "gemma-attn.timestamps" }),
    resolved: makeBuffer(device, slots * 16, U.QUERY_RESOLVE | U.COPY_SRC, "gemma-attn.timestamp-resolved"),
    read: makeBuffer(device, 256, U.MAP_READ | U.COPY_DST, "gemma-attn.timestamp-read"),
  };
}

async function timedDispatch(device, timer, run, reps, uncaptured, phase) {
  return checked(device, async () => {
    const encoder = device.createCommandEncoder({ label: phase });
    const pass = timer.timestamp
      ? encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })
      : encoder.beginComputePass();
    pass.setPipeline(run.pipeline);
    pass.setBindGroup(0, run.group);
    for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(run.T, run.heads, 1);
    pass.end();
    if (timer.timestamp) {
      encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolved, 0);
      encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16);
    }
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (!timer.timestamp) return (performance.now() - started) / reps;
    await timer.read.mapAsync(GPUMapMode.READ, 0, 16);
    const timestamps = new BigInt64Array(timer.read.getMappedRange(0, 16).slice(0));
    timer.read.unmap();
    return Number(timestamps[1] - timestamps[0]) / 1e6 / reps;
  }, phase, uncaptured);
}

// Runs both variants in one command buffer, alternating their order across samples. Each
// variant receives its own timestamp pair, so the comparison is interleaved on the same queue
// submission instead of relying on separate runs with drifting GPU state.
async function timedPair(device, timer, runs, reps, uncaptured, phase) {
  if (!timer.timestamp) {
    const timings = [];
    for (let i = 0; i < runs.length; i++) timings.push(await timedDispatch(device, timer, runs[i], reps, uncaptured, `${phase}.${i}`));
    return timings;
  }
  return checked(device, async () => {
    const encoder = device.createCommandEncoder({ label: phase });
    for (let i = 0; i < runs.length; i++) {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(runs[i].pipeline);
      pass.setBindGroup(0, runs[i].group);
      for (let rep = 0; rep < reps; rep++) pass.dispatchWorkgroups(runs[i].T, runs[i].heads, 1);
      pass.end();
    }
    encoder.resolveQuerySet(timer.querySet, 0, runs.length * 2, timer.resolved, 0);
    encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, runs.length * 16);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await timer.read.mapAsync(GPUMapMode.READ, 0, runs.length * 16);
    const timestamps = new BigInt64Array(timer.read.getMappedRange(0, runs.length * 16).slice(0));
    timer.read.unmap();
    return runs.map((_, i) => Number(timestamps[2 * i + 1] - timestamps[2 * i]) / 1e6 / reps);
  }, phase, uncaptured);
}

async function readOutput(device, output, bytes, uncaptured, phase) {
  const readback = makeBuffer(device, bytes, U.MAP_READ | U.COPY_DST, `${phase}.readback`);
  try {
    return await checked(device, async () => {
      const encoder = device.createCommandEncoder({ label: `${phase}.readback` });
      encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ, 0, bytes);
      const copy = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
      readback.unmap();
      return copy;
    }, `${phase} readback`, uncaptured);
  } finally {
    readback.destroy();
  }
}

function estimateWork(c) {
  let keys = 0;
  for (let t = 0; t < c.T; t++) {
    const [start, end] = attentionRange(c, t);
    keys += end - start;
  }
  return keys * c.heads * c.dim;
}

async function main() {
  const cases = parseCases();
  const seed = intParam("seed", 0x4b455641, 1, 0xffffffff);
  const warmups = intParam("warmups", 1, 0, 10);
  const samples = intParam("samples", 5, 1, 20);
  const reverse = intParam("reverse", 0, 0, 1);
  const magnitude = floatParam("magnitude", 0.5, 1e-6, 16);
  const gpu = await requestDevice();
  const device = gpu.device;
  U = GPUBufferUsage;
  activeDevice = gpu;
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push({ message: event.error?.message || String(event.error) }));
  const source = await kernelSource();
  const code = source("gemma4_attention");
  const timer = makeTimer(device);
  const layout = device.createBindGroupLayout({
    label: "gemma-attn",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const attentionDynamic = await pipeline(device, code, "gemma4_attention_dynamic", pipelineLayout);
  const compiledPipelines = new Map();
  const correctness = { ok: true, cases: [], tolerance: { absolute: 1e-5, relative: 1e-5 }, fullOutputComparison: true };
  const results = [];
  const dynamicMedians = [];
  const compiledMedians = [];
  log(`device: ${gpu.name}; timer: ${timer.method}; dynamic/compiled head dimension`);
  for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
    const c = cases[caseIndex];
    if (!compiledPipelines.has(c.dim)) {
      compiledPipelines.set(c.dim, await pipeline(device, specializeHeadDim(code, c.dim), `gemma4_attention_h${c.dim}`, pipelineLayout));
    }
    const input = makeInputs(c, (seed + Math.imul(caseIndex, 0x9e3779b9)) >>> 0, magnitude);
    const reference = cpuReference(c, input);
    const bytesO = c.T * c.heads * c.dim * 4;
    const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${c.label}.globals`);
    const params = makeBuffer(device, 32, U.UNIFORM | U.COPY_DST, `${c.label}.params`);
    const Q = makeBuffer(device, input.q.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.Q`);
    const K = makeBuffer(device, input.k.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.K`);
    const V = makeBuffer(device, input.v.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.V`);
    const ODynamic = makeBuffer(device, bytesO, U.STORAGE | U.COPY_SRC, `${c.label}.O.dynamic`);
    const OCompiled = makeBuffer(device, bytesO, U.STORAGE | U.COPY_SRC, `${c.label}.O.compiled`);
    const POS = makeBuffer(device, c.T * 4, U.STORAGE | U.COPY_DST, `${c.label}.positions`);
    const group = (output) => device.createBindGroup({
      layout,
      entries: [globals, params, Q, K, V, output, POS].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const work = estimateWork(c);
    const reps = Math.max(1, Math.min(64, Math.round(8e7 / Math.max(1, work))));
    const dynamicRun = { pipeline: attentionDynamic, group: group(ODynamic), T: c.T, heads: c.heads };
    const compiledRun = { pipeline: compiledPipelines.get(c.dim), group: group(OCompiled), T: c.T, heads: c.heads };
    await checked(device, async () => {
      device.queue.writeBuffer(globals, 0, new Uint32Array([c.T, 0, c.T, 0]));
      device.queue.writeBuffer(params, 0, new Uint32Array([c.heads, c.kvHeads, c.dim, c.window, c.causal, 0, 0, 0]));
      device.queue.writeBuffer(Q, 0, input.q);
      device.queue.writeBuffer(K, 0, input.k);
      device.queue.writeBuffer(V, 0, input.v);
      device.queue.writeBuffer(POS, 0, Uint32Array.from({ length: c.T }, (_, i) => i));
      await device.queue.onSubmittedWorkDone();
    }, `${c.label} upload`, uncaptured);
    const orderFor = (round) => (round + reverse) % 2 ? [compiledRun, dynamicRun] : [dynamicRun, compiledRun];
    for (let i = 0; i < warmups; i++) await timedPair(device, timer, orderFor(i), reps, uncaptured, `${c.label} warmup ${i}`);
    const dynamicTimings = [];
    const compiledTimings = [];
    for (let i = 0; i < samples; i++) {
      const order = orderFor(i);
      const paired = await timedPair(device, timer, order, reps, uncaptured, `${c.label} sample ${i}`);
      const dynamicIndex = order[0] === dynamicRun ? 0 : 1;
      dynamicTimings.push(paired[dynamicIndex]);
      compiledTimings.push(paired[1 - dynamicIndex]);
    }
    const median = (timings) => [...timings].sort((a, b) => a - b)[Math.floor(timings.length / 2)];
    const dynamicMedianMs = median(dynamicTimings);
    const compiledMedianMs = median(compiledTimings);
    const dynamicValues = await readOutput(device, ODynamic, bytesO, uncaptured, `${c.label}.dynamic`);
    const compiledValues = await readOutput(device, OCompiled, bytesO, uncaptured, `${c.label}.compiled`);
    const dynamicCpu = compareCpu(dynamicValues, reference);
    const compiledCpu = compareCpu(compiledValues, reference);
    const equality = compareOutputs(dynamicValues, compiledValues);
    if (!dynamicCpu.ok || !compiledCpu.ok || !equality.ok) correctness.ok = false;
    correctness.cases.push({
      case: c.label,
      dynamic: dynamicCpu,
      compiled: compiledCpu,
      equality,
      fullCheck: reference.full,
      fullOutputElements: bytesO / 4,
      checkedQueryHeads: reference.tokens.length * reference.heads.length,
    });
    dynamicMedians.push(dynamicMedianMs);
    compiledMedians.push(compiledMedianMs);
    results.push({
      case: c.label,
      T: c.T,
      heads: c.heads,
      kvHeads: c.kvHeads,
      headDim: c.dim,
      window: c.window,
      causal: Boolean(c.causal),
      work,
      reps,
      samples: compiledTimings,
      medianMs: compiledMedianMs,
      dynamicSamples: dynamicTimings,
      dynamicMedianMs,
      compiledSamples: compiledTimings,
      compiledMedianMs,
      speedup: dynamicMedianMs / compiledMedianMs,
      interleaved: true,
      correctness: { dynamic: dynamicCpu, compiled: compiledCpu, equality },
    });
    log(`${c.label}: dynamic=${dynamicMedianMs.toFixed(4)} ms compiled=${compiledMedianMs.toFixed(4)} ms (${reps} dispatches/sample), ${dynamicCpu.ok && compiledCpu.ok && equality.ok ? "correct" : "FAILED"} maxAbs=${Math.max(dynamicCpu.maxAbs || 0, compiledCpu.maxAbs || 0, equality.maxAbs || 0).toExponential(2)}`);
    for (const buffer of [globals, params, Q, K, V, ODynamic, OCompiled, POS]) buffer.destroy();
  }
  const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
  const metricMs = geometricMean(compiledMedians);
  const dynamicMetricMs = geometricMean(dynamicMedians);
  const result = {
    status: correctness.ok ? "done" : "error",
    done: true,
    backend: "webgpu",
    kernel: "gemma4_attention",
    metricMs,
    dynamicMetricMs,
    speedup: dynamicMetricMs / metricMs,
    metric: "geometric mean of per-case median Gemma attention with compiled head dimension",
    method: `${timer.method}, interleaved dynamic/compiled head dimension`,
    provenance: { wasmUrl: source.wasmUrl, specialization: "only let H = p.head_dim replaced by const H = case.dim" },
    seed,
    magnitude,
    warmups,
    samples,
    reverse: Boolean(reverse),
    cases: results,
    adapter: adapterInfo(gpu),
    correctness,
  };
  if (!correctness.ok) result.error = "Gemma attention CPU or full-output parity failed";
  window.gpuBench = result;
  log(`metricMs=${metricMs.toFixed(4)}; dynamicMetricMs=${dynamicMetricMs.toFixed(4)}; correctness=${correctness.ok ? "ok" : "FAILED"}`);
  timer.querySet?.destroy();
  timer.resolved?.destroy();
  timer.read?.destroy();
}

main().catch((error) => {
  const message = `${error?.message || error}\n${error?.stack || ""}`;
  window.gpuBench = { ...state, status: "error", done: true, error: message };
  log(`ERROR ${message}`);
}).finally(() => {
  activeDevice?.device.destroy();
  activeDevice = null;
});
