// Focused numerical and timing guard for production Gemma attention and the isolated
// grouped-query tile candidate. The case format is T x query-heads x KV-heads x
// head-dim x window x causal. Q is already prepared, so this benchmark deliberately
// does not apply an attention scale. The candidate handles GQA groups of four or eight
// with 256 or 512 dimensional heads; the baseline remains available for other cases.

import { kernelSource } from "./wgsl.js";
import { pipeline, requestDevice } from "../js/src/gpu.js";

const query = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => {
  logNode.textContent += `${line}\n`;
};
const state = { status: "running", backend: "webgpu", kernel: "gemma4_attention_gqa", baselineKernel: "gemma4_attention" };
window.gpuBench = state;

let U;
let activeDevice = null;

const DEFAULT_CASES = [
  { T: 1, S: 1, heads: 8, kvHeads: 1, dim: 256, window: 0, causal: 1, positions: "permuted" },
  { T: 3, S: 3, heads: 8, kvHeads: 1, dim: 256, window: 2, causal: 1, positions: "permuted" },
  { T: 4, S: 4, heads: 8, kvHeads: 1, dim: 256, window: 0, causal: 0, positions: "permuted" },
  { T: 7, S: 7, heads: 8, kvHeads: 2, dim: 256, window: 4, causal: 1, positions: "permuted" },
  { T: 8, S: 8, heads: 8, kvHeads: 2, dim: 512, window: 0, causal: 1, positions: "permuted" },
  { T: 9, S: 9, heads: 8, kvHeads: 2, dim: 512, window: 5, causal: 0, positions: "permuted" },
  { T: 31, S: 31, heads: 8, kvHeads: 1, dim: 256, window: 0, causal: 1, positions: "permuted" },
  { T: 127, S: 127, heads: 8, kvHeads: 1, dim: 256, window: 0, causal: 1, positions: "permuted" },
  { T: 128, S: 128, heads: 8, kvHeads: 2, dim: 512, window: 32, causal: 1, positions: "permuted" },
  { T: 129, S: 129, heads: 8, kvHeads: 2, dim: 512, window: 0, causal: 1, positions: "permuted" },
  { T: 255, S: 255, heads: 8, kvHeads: 1, dim: 256, window: 16, causal: 1, positions: "permuted" },
  { T: 256, S: 256, heads: 8, kvHeads: 2, dim: 512, window: 0, causal: 0, positions: "permuted" },
  { T: 257, S: 257, heads: 16, kvHeads: 4, dim: 256, window: 7, causal: 1, positions: "permuted" },
  { T: 99, S: 99, heads: 8, kvHeads: 1, dim: 256, window: 512, causal: 1, positions: "identity" },
  { T: 99, S: 99, heads: 8, kvHeads: 1, dim: 512, window: 0, causal: 1, positions: "identity" },
  { T: 193, S: 193, heads: 8, kvHeads: 1, dim: 256, window: 512, causal: 1, positions: "identity" },
  { T: 193, S: 193, heads: 8, kvHeads: 1, dim: 512, window: 0, causal: 1, positions: "identity" },
  { T: 601, S: 601, heads: 8, kvHeads: 1, dim: 256, window: 512, causal: 1, positions: "identity" },
  { T: 601, S: 601, heads: 8, kvHeads: 1, dim: 512, window: 0, causal: 1, positions: "identity" },
  { T: 1, S: 601, heads: 8, kvHeads: 1, dim: 256, window: 512, causal: 1, positions: "tail" },
  { T: 8, S: 601, heads: 8, kvHeads: 1, dim: 256, window: 512, causal: 1, positions: "tail" },
  { T: 1, S: 601, heads: 8, kvHeads: 1, dim: 512, window: 0, causal: 1, positions: "tail" },
  { T: 8, S: 601, heads: 8, kvHeads: 1, dim: 512, window: 0, causal: 1, positions: "tail" },
];

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
  const sequence = c.S && c.S !== c.T ? `xS${c.S}` : "";
  return `${c.T}x${c.heads}x${c.kvHeads}x${c.dim}x${c.window}x${c.causal}${sequence}`;
}

function parseCases() {
  const raw = query.get("cases");
  if (!raw || raw === "default") return DEFAULT_CASES.map((c) => ({ ...c, label: labelCase(c) }));
  if (raw === "fast") return DEFAULT_CASES.slice(13).map((c) => ({ ...c, label: labelCase(c) }));
  const cases = raw.split(",").map((item) => {
    const fields = item.trim().split("x");
    if (fields.length !== 6 && fields.length !== 7) throw new Error("cases must be comma-separated T x heads x kvHeads x dim x window x causal [x S] shapes");
    const [T, heads, kvHeads, dim, window] = fields.slice(0, 5).map(Number);
    const S = fields.length === 7 ? Number(fields[6]) : T;
    const causal = parseCausal(fields[5]);
    if (![T, S, heads, kvHeads, dim, window].every(Number.isInteger) || T < 1 || T > 65535 || S < 1 || S > 65535 || heads < 1 || heads > 1024 || kvHeads < 1 || kvHeads > heads || heads % kvHeads !== 0 || window < 0 || window > 0xffffffff || dim < 1 || dim > 512) {
      throw new Error(`invalid attention case ${item}; expected positive T/S, divisible heads/kvHeads, dim 1..512, and a u32 window`);
    }
    const positionMode = query.get("positions");
    const positions = positionMode === "permuted" || positionMode === "tail" ? positionMode : "identity";
    const c = { T, S, heads, kvHeads, dim, window, causal, positions };
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
  const k = new Float32Array(c.S * c.kvHeads * c.dim);
  const v = new Float32Array(c.S * c.kvHeads * c.dim);
  const positions = new Uint32Array(c.T);
  // Keep scores in a numerically quiet range by default. A magnitude query
  // parameter can deliberately create peaked softmax distributions.
  for (const values of [q, k, v]) for (let i = 0; i < values.length; i++) values[i] = Math.fround((random() - 0.5) * magnitude);
  // Identity positions model production timing. Tail positions model selected-row
  // compact execution. Permuted positions are a separate arbitrary-POS guard.
  for (let t = 0; t < c.T; t++) {
    positions[t] = c.positions === "permuted" ? (Math.imul(t, 5) + 3) % c.S : c.positions === "tail" ? c.S - c.T + t : Math.min(t, c.S - 1);
  }
  return { q, k, v, positions };
}

function attentionRange(c, t, positions) {
  const position = positions[t];
  const start = c.window > 0 ? position + 1 - Math.min(position + 1, c.window) : 0;
  const end = c.causal ? position + 1 : c.S;
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
    const [start, end] = attentionRange(c, t, input.positions);
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
  return {
    timestamp: true,
    method: "WebGPU timestamp-query",
    querySet: device.createQuerySet({ type: "timestamp", count: 2, label: "gemma-attn.timestamps" }),
    resolved: makeBuffer(device, 16, U.QUERY_RESOLVE | U.COPY_SRC, "gemma-attn.timestamp-resolved"),
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
    for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(run.dispatchX, run.dispatchY, 1);
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

async function dispatchOnce(device, run, uncaptured, phase) {
  return checked(device, async () => {
    const encoder = device.createCommandEncoder({ label: phase });
    const pass = encoder.beginComputePass();
    pass.setPipeline(run.pipeline);
    pass.setBindGroup(0, run.group);
    pass.dispatchWorkgroups(run.dispatchX, run.dispatchY, 1);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }, phase, uncaptured);
}

function estimateWork(c, positions) {
  let keys = 0;
  for (let t = 0; t < c.T; t++) {
    const [start, end] = attentionRange(c, t, positions);
    keys += end - start;
  }
  return keys * c.heads * c.dim;
}

function tileConfig(c) {
  const group = c.heads / c.kvHeads;
  if (![4, 8].includes(group) || ![256, 512].includes(c.dim)) return null;
  return { group, tokenRows: 32 / group, keyCount: c.dim === 256 ? 8 : 4 };
}

function compareArrays(actual, expected, label) {
  const tolerance = { absolute: 1e-5, relative: 1e-5 };
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  if (actual.length !== expected.length) return { ok: false, message: `${label} length mismatch` };
  for (let index = 0; index < actual.length; index++) {
    if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index])) return { ok: false, message: `${label} non-finite output at index ${index}` };
    const abs = Math.abs(actual[index] - expected[index]);
    const relative = abs / Math.max(1, Math.abs(expected[index]));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index, expected: expected[index], actual: actual[index] };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > tolerance.absolute + tolerance.relative * Math.max(1, Math.abs(expected[index]))) {
      return { ok: false, maxAbs, maxRelative, worst, message: `${label} mismatch at output index ${index}`, tolerance };
    }
  }
  return { ok: true, maxAbs, maxRelative, worst, tolerance };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function main() {
  const cases = parseCases();
  const seed = intParam("seed", 0x4b455641, 1, 0xffffffff);
  const warmups = intParam("warmups", 1, 0, 10);
  const samples = intParam("samples", 5, 1, 20);
  const magnitude = floatParam("magnitude", 0.5, 1e-6, 16);
  const gpu = await requestDevice();
  const device = gpu.device;
  U = GPUBufferUsage;
  activeDevice = gpu;
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push({ message: event.error?.message || String(event.error) }));
  const source = await kernelSource();
  const timer = makeTimer(device);
  const layout = device.createBindGroupLayout({
    label: "gemma-attn-paired",
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
  const attention = await pipeline(device, source("gemma4_attention"), "gemma4_attention", pipelineLayout);
  const maxStorage = device.limits.maxComputeWorkgroupStorageSize;
  if (!gpu.subgroup4) throw new Error("gemma4_attention_gqa requires a hardware device with subgroup builtins");
  if (!navigator.gpu.wgslLanguageFeatures?.has("subgroup_id")) throw new Error("subgroup_id language feature required");
  const forceShared = query.get("forceShared") === "1";
  const useSubgroupRows = Boolean(gpu.subgroup32 && !forceShared);
  const candidateStorage = 17672;
  if (maxStorage < candidateStorage) throw new Error(`gemma4_attention_gqa requires at least ${candidateStorage} bytes of workgroup storage; device provides ${maxStorage}`);
  const attentionGqa = await pipeline(device, source("gemma4_attention_gqa"), "gemma4_attention_gqa", pipelineLayout);
  const correctness = {
    ok: true,
    cases: [],
    tolerance: { absolute: 1e-5, relative: 1e-5 },
    candidateStorage,
  };
  const results = [];
  const baselineMedians = [];
  const candidateMedians = [];
  log(`device: ${gpu.name}; timer: ${timer.method}; candidate storage=${candidateStorage}/${maxStorage}; subgroup32=${useSubgroupRows}; forceShared=${forceShared}`);

  const measurePair = async (baselineRun, candidateRun, reps, label) => {
    const baselineTimings = [];
    const candidateTimings = [];
    const run = async (which, phase) => timedDispatch(device, timer, which, reps, uncaptured, phase);
    for (let i = 0; i < warmups; i++) {
      if (candidateRun && i % 2) {
        await run(candidateRun, `${label} candidate warmup ${i}`);
        await run(baselineRun, `${label} baseline warmup ${i}`);
      } else {
        await run(baselineRun, `${label} baseline warmup ${i}`);
        if (candidateRun) await run(candidateRun, `${label} candidate warmup ${i}`);
      }
    }
    for (let i = 0; i < samples; i++) {
      if (candidateRun && i % 2) {
        candidateTimings.push(await run(candidateRun, `${label} candidate sample ${i}`));
        baselineTimings.push(await run(baselineRun, `${label} baseline sample ${i}`));
      } else {
        baselineTimings.push(await run(baselineRun, `${label} baseline sample ${i}`));
        if (candidateRun) candidateTimings.push(await run(candidateRun, `${label} candidate sample ${i}`));
      }
    }
    return {
      baseline: { timings: baselineTimings, medianMs: median(baselineTimings) },
      candidate: candidateRun ? { timings: candidateTimings, medianMs: median(candidateTimings) } : null,
      order: "alternating",
    };
  };

  for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
    const c = cases[caseIndex];
    const input = makeInputs(c, (seed + Math.imul(caseIndex, 0x9e3779b9)) >>> 0, magnitude);
    const reference = cpuReference(c, input);
    const tile = tileConfig(c);
    const bytesQ = input.q.byteLength;
    const bytesK = input.k.byteLength;
    const bytesV = input.v.byteLength;
    const bytesO = c.T * c.heads * c.dim * 4;
    const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${c.label}.globals`);
    const params = makeBuffer(device, 32, U.UNIFORM | U.COPY_DST, `${c.label}.params`);
    const Q = makeBuffer(device, bytesQ, U.STORAGE | U.COPY_DST, `${c.label}.Q`);
    const K = makeBuffer(device, bytesK, U.STORAGE | U.COPY_DST, `${c.label}.K`);
    const V = makeBuffer(device, bytesV, U.STORAGE | U.COPY_DST, `${c.label}.V`);
    const O = makeBuffer(device, bytesO, U.STORAGE | U.COPY_SRC | U.COPY_DST, `${c.label}.O`);
    const unwritten = new Float32Array(bytesO / 4).fill(NaN);
    const POS = makeBuffer(device, c.T * 4, U.STORAGE | U.COPY_DST, `${c.label}.positions`);
    const group = device.createBindGroup({
      layout,
      entries: [globals, params, Q, K, V, O, POS].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const work = estimateWork(c, input.positions);
    const reps = Math.max(1, Math.min(64, Math.round(8e7 / Math.max(1, work))));
    const baselineRun = { pipeline: attention, group, dispatchX: c.T, dispatchY: c.heads };
    const candidateRun = tile && { pipeline: attentionGqa, group, dispatchX: Math.ceil(c.T / tile.tokenRows), dispatchY: c.kvHeads };
    await checked(device, async () => {
      device.queue.writeBuffer(globals, 0, new Uint32Array([c.T, 0, c.S, 0]));
      device.queue.writeBuffer(params, 0, new Uint32Array([c.heads, c.kvHeads, c.dim, c.window, c.causal, tile?.tokenRows || 0, tile?.keyCount || 0, useSubgroupRows ? 1 : 0]));
      device.queue.writeBuffer(Q, 0, input.q);
      device.queue.writeBuffer(K, 0, input.k);
      device.queue.writeBuffer(V, 0, input.v);
      device.queue.writeBuffer(POS, 0, input.positions);
      await device.queue.onSubmittedWorkDone();
    }, `${c.label} upload`, uncaptured);

    const timing = await measurePair(baselineRun, candidateRun, reps, c.label);
    device.queue.writeBuffer(O, 0, unwritten);
    await dispatchOnce(device, baselineRun, uncaptured, `${c.label} baseline correctness`);
    const baselineValues = await readOutput(device, O, bytesO, uncaptured, `${c.label} baseline`);
    const baselineCpu = compareCpu(baselineValues, reference);
    let candidateTiming = null;
    let candidateValues = null;
    let candidateCpu = null;
    let paired = null;
    if (candidateRun) {
      candidateTiming = timing.candidate;
      device.queue.writeBuffer(O, 0, unwritten);
      await dispatchOnce(device, candidateRun, uncaptured, `${c.label} candidate correctness`);
      candidateValues = await readOutput(device, O, bytesO, uncaptured, `${c.label} candidate`);
      candidateCpu = compareCpu(candidateValues, reference);
      paired = compareArrays(candidateValues, baselineValues, "candidate/baseline");
    }
    const caseOk = baselineCpu.ok && (!candidateCpu || (candidateCpu.ok && paired.ok));
    if (!caseOk) correctness.ok = false;
    const checks = {
      case: c.label,
      sequenceLength: c.S,
      positions: c.positions,
      fullCheck: reference.full,
      checkedQueryHeads: reference.tokens.length * reference.heads.length,
      baseline: baselineCpu,
      candidate: candidateCpu,
      paired,
    };
    correctness.cases.push(checks);
    baselineMedians.push(timing.baseline.medianMs);
    if (candidateTiming) candidateMedians.push(candidateTiming.medianMs);
    results.push({
      case: c.label,
      T: c.T,
      S: c.S,
      heads: c.heads,
      kvHeads: c.kvHeads,
      headDim: c.dim,
      window: c.window,
      causal: Boolean(c.causal),
      positions: c.positions,
      tile,
      work,
      reps,
      baseline: { ...timing.baseline, correctness: baselineCpu },
      candidate: candidateTiming ? { ...candidateTiming, correctness: candidateCpu, paired } : null,
      timingOrder: timing.order,
    });
    log(`${c.label}: baseline=${timing.baseline.medianMs.toFixed(4)} ms, candidate=${candidateTiming?.medianMs.toFixed(4) || "skipped"} ms, ${caseOk ? "correct" : "FAILED"}`);
    for (const buffer of [globals, params, Q, K, V, O, POS]) buffer.destroy();
  }
  const baselineMetricMs = Math.exp(baselineMedians.reduce((sum, value) => sum + Math.log(value), 0) / baselineMedians.length);
  const candidateMetricMs = candidateMedians.length ? Math.exp(candidateMedians.reduce((sum, value) => sum + Math.log(value), 0) / candidateMedians.length) : null;
  const result = {
    status: correctness.ok ? "done" : "error",
    done: true,
    backend: "webgpu",
    kernel: "gemma4_attention_gqa",
    baselineKernel: "gemma4_attention",
    metricMs: candidateMetricMs,
    baselineMetricMs,
    candidateMetricMs,
    metric: "geometric mean of per-case median Gemma 4 attention milliseconds",
    method: timer.method,
    seed,
    magnitude,
    warmups,
    samples,
    subgroup32: useSubgroupRows,
    forceShared,
    cases: results,
    adapter: adapterInfo(gpu),
    correctness,
  };
  if (!correctness.ok) result.error = "Gemma 4 attention paired CPU or output parity failed";
  window.gpuBench = result;
  log(`baselineMetricMs=${baselineMetricMs.toFixed(4)}; candidateMetricMs=${candidateMetricMs?.toFixed(4) || "n/a"}; correctness=${correctness.ok ? "ok" : "FAILED"}`);
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
