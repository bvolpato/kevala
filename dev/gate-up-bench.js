import { kernelSource } from "./wgsl.js";
import { dispatchMatmul, matmulConfig, matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, rowsPerThread } from "../js/src/gpu.js";

// This page is an isolated experiment. It compares one fused gate/up+GELU
// dispatch with the generic gate matmul, generic up matmul, and gemma4_gelu
// sequence used by the runtime. The fused kernel intentionally rejects split-K.

const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const DEFAULT_CASES = [
  [7, 67, 64],
  [65, 130, 96],
  [99, 6144, 1536],
  [193, 6144, 1536],
  [601, 6144, 1536],
];
const DEFAULT_PATTERNS = ["finite", "cancellation", "normal", "outlier", "finite"];

let U;
let activeDevice = null;
const lostDevices = new WeakMap();

const numberParam = (query, name, fallback, min, max) => {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  return Number.isFinite(value) ? Math.max(min, Math.min(max, Math.floor(value))) : fallback;
};

function parseCases(query) {
  const raw = query.get("cases");
  if (!raw) return DEFAULT_CASES;
  const parsed = raw.split(",").map((item) => item.split("x").map(Number));
  if (!parsed.length || parsed.some((shape) => shape.length !== 3 || shape.some((n) => !Number.isInteger(n) || n <= 0))) {
    throw new Error("cases must be comma-separated T x N x K shapes");
  }
  return parsed;
}

function parsePatterns(query, count) {
  const raw = query.get("patterns");
  const patterns = raw ? raw.split(",") : DEFAULT_PATTERNS;
  const allowed = new Set(["finite", "normal", "cancellation", "outlier"]);
  if (patterns.length !== count || patterns.some((pattern) => !allowed.has(pattern))) {
    throw new Error(`patterns must contain ${count} values from finite,normal,cancellation,outlier`);
  }
  return patterns;
}

function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
}

function signedByte(value) {
  return value < 128 ? value : value - 256;
}

function inputValue(pattern, random, index, width) {
  if (pattern === "cancellation") {
    const sign = (index & 1) ? -1 : 1;
    return Math.fround(sign * (0.125 + random() * 0.375));
  }
  if (pattern === "outlier") {
    if (index % 257 === 0) return Math.fround((index & 1) ? -32 : 32);
    return Math.fround((random() - 0.5) * 0.02);
  }
  if (pattern === "finite") return Math.fround((random() - 0.5) * 0.75);
  return Math.fround((random() - 0.5) * Math.min(0.8, 2 / Math.sqrt(Math.max(1, width))));
}

function weightValue(pattern, random, index) {
  if (pattern === "cancellation") return 127;
  if (pattern === "outlier") {
    if (index % 193 === 0) return (index & 1) ? 0 : 255;
    return 127 + ((random() * 7) | 0) - 3;
  }
  if (pattern === "finite") return (random() * 256) | 0;
  return (random() * 256) | 0;
}

function makeInput(T, N, K, pattern, seed) {
  if (K % 32) throw new Error(`K must be a multiple of 32, got ${K}`);
  const random = makeRng(seed);
  const x = new Float32Array(T * K);
  const gateW = new Uint8Array(N * K);
  const upW = new Uint8Array(N * K);
  const gateS = new Float32Array(N * (K / 32));
  const upS = new Float32Array(N * (K / 32));
  for (let i = 0; i < x.length; i++) x[i] = inputValue(pattern, random, i, K);
  if (pattern === "cancellation") {
    for (let i = 0; i < x.length; i += 2) x[i + 1] = -x[i];
  }
  for (let i = 0; i < gateW.length; i++) gateW[i] = weightValue(pattern, random, i);
  for (let i = 0; i < upW.length; i++) upW[i] = weightValue(pattern, random, i + 17);
  for (let i = 0; i < gateS.length; i++) gateS[i] = Math.fround(pattern === "outlier" ? 0.02 + random() * 0.03 : 0.005 + random() * 0.012);
  for (let i = 0; i < upS.length; i++) upS[i] = Math.fround(pattern === "outlier" ? 0.02 + random() * 0.03 : 0.005 + random() * 0.012);
  return { T, N, K, pattern, x, gateW, upW, gateS, upS };
}

function geluCpu(x) {
  // Keep the same expression and constants as gemma4_gelu.wgsl, with f32
  // rounding at each operation in the independent reference.
  const c = Math.fround(0.7978845608028654);
  const scaled = Math.fround(Math.fround(0.044715) * x);
  const cubic = Math.fround(Math.fround(scaled * x) * x);
  const inner = Math.fround(x + cubic);
  const tanhArg = Math.fround(c * inner);
  const onePlus = Math.fround(1 + Math.fround(Math.tanh(tanhArg)));
  return Math.fround(Math.fround(0.5 * x) * onePlus);
}

function projectionCpu(input, row, col, weights, scales) {
  const { N, K, x } = input;
  const blocks = K / 32;
  let sum = 0;
  for (let k = 0; k < K; k++) {
    const weight = Math.fround(signedByte(weights[col * K + k]) * scales[col * blocks + Math.floor(k / 32)]);
    sum = Math.fround(sum + Math.fround(x[row * K + k] * weight));
  }
  return sum;
}

function cpuValue(input, row, col) {
  const gate = projectionCpu(input, row, col, input.gateW, input.gateS);
  const up = projectionCpu(input, row, col, input.upW, input.upS);
  let activated = geluCpu(gate);
  if (input.activation === 1) {
    const x = gate * 0.70710678118;
    const a = Math.abs(x);
    let erf;
    if (a < 0.5) {
      const z = x * x;
      erf = x * (1.1283791671 + z * (-0.3761263890 + z * (0.1128379167 + z * (-0.0268661706 + z * 0.0052239776))));
    } else {
      const t = 1 / (1 + 0.3275911 * a);
      const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
      erf = x >= 0 ? y : -y;
    }
    activated = Math.fround(0.5 * gate * (1 + erf));
  } else if (input.activation === 2) activated = Math.fround(gate / (1 + Math.exp(-gate)));
  return Math.fround(activated * up);
}

function sampleIndices(T, N) {
  if (T * N <= 8192) return Array.from({ length: T * N }, (_, i) => i);
  const rows = [...new Set([0, 1, 2, Math.floor(T / 2), Math.max(0, T - 3), Math.max(0, T - 2), T - 1])].filter((r) => r < T);
  const cols = [...new Set([0, 1, 2, Math.floor(N / 2), Math.max(0, N - 3), Math.max(0, N - 2), N - 1])].filter((c) => c < N);
  const out = new Set();
  for (const row of rows) for (const col of cols) out.add(row * N + col);
  return [...out];
}

function compareCpu(input, values, label) {
  const indices = sampleIndices(input.T, input.N);
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  let ok = true;
  for (const index of indices) {
    const row = Math.floor(index / input.N);
    const col = index % input.N;
    const expected = cpuValue(input, row, col);
    const actual = values[index];
    if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
      return { ok: false, samples: indices.length, message: `${label} CPU check found a non-finite value at ${row},${col}` };
    }
    const abs = Math.abs(actual - expected);
    const relative = abs / Math.max(1, Math.abs(expected));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { row, col, expected, actual };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > 0.0005 + 0.0001 * Math.max(1, Math.abs(expected))) ok = false;
  }
  return { ok, samples: indices.length, maxAbs, maxRelative, worst, message: ok ? undefined : `${label} CPU check exceeded f32 tolerance` };
}

function finiteSummary(values, label) {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) return { ok: false, message: `${label} output is non-finite at ${i}` };
  }
  return { ok: true };
}

function compareArrays(a, b) {
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  let ok = true;
  for (let i = 0; i < a.length; i++) {
    const abs = Math.abs(a[i] - b[i]);
    const relative = abs / Math.max(1, Math.abs(a[i]));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index: i, expected: a[i], actual: b[i] };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > 0.0005 + 0.0001 * Math.max(1, Math.abs(a[i]))) ok = false;
  }
  return { ok, samples: a.length, maxAbs, maxRelative, worst, message: ok ? undefined : "fused output differs from generic baseline" };
}

function makeBuffer(device, bytes, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, bytes), usage });
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
  };
}

function makeLayout(device, entries) {
  return device.createBindGroupLayout({ entries: entries.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
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
  if (uncaptured.length) throw new Error(`WebGPU ${phase}: uncaptured error: ${uncaptured.splice(0).map((item) => item.message).join("; ")}`);
  const lost = lostDevices.get(device);
  if (lost) throw new Error(`WebGPU ${phase}: device lost: ${lost.message || lost.reason || "unknown reason"}`);
  return value;
}

function makeTimer(device, reps) {
  if (!device.features.has("timestamp-query")) return { timestamp: false, method: "performance.now() wall clock" };
  return {
    timestamp: true,
    method: "WebGPU timestamp-query",
    querySet: device.createQuerySet({ type: "timestamp", count: 2 * reps, label: "gate-up.timestamps" }),
    resolved: makeBuffer(device, 16 * reps, U.QUERY_RESOLVE | U.COPY_SRC, "gate-up.timestamp-resolved"),
    read: makeBuffer(device, 16 * reps, U.MAP_READ | U.COPY_DST, "gate-up.timestamp-read"),
  };
}

async function runSequence(device, timer, sequence, reps, uncaptured, label) {
  return checked(device, async () => {
    const encoder = device.createCommandEncoder({ label });
    for (let i = 0; i < reps; i++) {
      const pass = timer?.timestamp
        ? encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } })
        : encoder.beginComputePass();
      sequence(pass);
      pass.end();
    }
    if (timer?.timestamp) {
      encoder.resolveQuerySet(timer.querySet, 0, 2 * reps, timer.resolved, 0);
      encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16 * reps);
    }
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (timer?.timestamp) {
      await timer.read.mapAsync(GPUMapMode.READ, 0, 16 * reps);
      const values = new BigInt64Array(timer.read.getMappedRange(0, 16 * reps));
      let total = 0;
      for (let i = 0; i < reps; i++) total += Number(values[2 * i + 1] - values[2 * i]) / 1e6;
      timer.read.unmap();
      return total / reps;
    }
    return (performance.now() - started) / reps;
  }, label, uncaptured);
}

async function readBuffer(device, buffer, bytes, uncaptured, label) {
  const readback = makeBuffer(device, bytes, U.MAP_READ | U.COPY_DST, `${label}.readback`);
  try {
    return await checked(device, async () => {
      const encoder = device.createCommandEncoder({ label: `${label}.readback` });
      encoder.copyBufferToBuffer(buffer, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ, 0, bytes);
      const values = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
      readback.unmap();
      return values;
    }, `${label} readback`, uncaptured);
  } finally {
    readback.destroy();
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function dispatchGelu(pass, pipelineObject, group, T, N) {
  pass.setPipeline(pipelineObject);
  pass.setBindGroup(0, group);
  const work = Math.ceil((T * N) / 256);
  pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
}

function dispatchFused(pass, pipelineObject, group, T, N) {
  pass.setPipeline(pipelineObject);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 64));
}

async function main() {
  const query = new URLSearchParams(location.search);
  const seed = numberParam(query, "seed", 0x4b455641, 1, 0xffffffff);
  const warmups = numberParam(query, "warmups", 2, 0, 10);
  const samples = numberParam(query, "samples", 5, 3, 20);
  const activation = numberParam(query, "activation", 0, 0, 2);
  const packed = activation !== 0;
  const shapes = parseCases(query);
  const patterns = parsePatterns(query, shapes.length);
  const gpu = await requestDevice({ features: ["timestamp-query"] });
  U = GPUBufferUsage;
  const device = gpu.device;
  const target = matmulConfig(device).splitTarget;
  activeDevice = device;
  lostDevices.set(device, null);
  device.lost.then((info) => lostDevices.set(device, info));
  if (gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter) throw new Error("software WebGPU adapter is not allowed");
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push({ name: event.error?.name || "GPUError", message: event.error?.message || String(event.error) }));
  const provenance = adapterInfo(gpu);
  const source = await kernelSource();
  const mmLayout = matmulLayout(device);
  const reduceLayoutObject = reduceLayout(device);
  const pairLayout = makeLayout(device, ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"]);
  const geluLayout = makeLayout(device, packed ? ["uniform", "uniform", "read-only-storage", "storage"] : ["uniform", "uniform", "read-only-storage", "read-only-storage", "storage"]);
  const mmPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [mmLayout] });
  const reducePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [reduceLayoutObject] });
  const pairPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [pairLayout] });
  const geluPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [geluLayout] });
  const pairCode = source("matmul_gate_up", { f16: false, rows: 4, groups: 1, splitTarget: target });
  const geluCode = source(["gemma4_gelu", "geglu", "kev_silumul"][activation]);
  const pairPipeline = await pipeline(device, pairCode, "matmul_gate_up", pairPipelineLayout);
  const geluPipeline = await pipeline(device, geluCode, "gemma4_gelu", geluPipelineLayout);
  const genericPipelines = new Map();
  for (const rows of [1, 2, 3, 4]) {
    const mmCode = source("matmul", { f16: false, rows, groups: 1, splitTarget: target });
    const reduceCode = source("reduce", { rows, groups: 1, splitTarget: target });
    genericPipelines.set(rows, {
      mm: await pipeline(device, mmCode, `matmul.generic.r${rows}`, mmPipelineLayout),
      reduce: await pipeline(device, reduceCode, `reduce.generic.r${rows}`, reducePipelineLayout),
    });
  }

  const cases = [];
  const correctness = { ok: true, cpuReference: "f32 signed-int8 projections with the gemma4_gelu tanh expression", cpu: [], variants: [], errors: [] };
  const fusedMedians = [];
  const baselineMedians = [];

  for (let caseIndex = 0; caseIndex < shapes.length; caseIndex++) {
    const [T, N, K] = shapes[caseIndex];
    if (K % 32) throw new Error(`matmul_gate_up requires K divisible by 32: ${T}x${N}x${K}`);
    const pairSplits = mmSplits(T, N, K, target, 64, 64);
    if (pairSplits !== 1) throw new Error(`matmul_gate_up does not support split-K: ${T}x${N}x${K} has ${pairSplits} splits at target ${target}`);
    const pattern = patterns[caseIndex];
    const input = makeInput(T, N, K, pattern, (seed + caseIndex * 0x9e3779b9) >>> 0);
    input.activation = activation;
    const baselineN = packed ? 2 * N : N;
    const label = `${T}x${N}x${K}:${pattern}`;
    const bytes = T * N * 4;
    const inputBytes = T * K * 4;
    const weightBytes = N * K;
    const scaleBytes = N * (K / 32) * 4;
    const X = makeBuffer(device, inputBytes, U.STORAGE | U.COPY_DST, `${label}.X`);
    const gateW = makeBuffer(device, weightBytes * (packed ? 2 : 1), U.STORAGE | U.COPY_DST, `${label}.gateW`);
    const upW = makeBuffer(device, weightBytes, U.STORAGE | U.COPY_DST, `${label}.upW`);
    const gateS = makeBuffer(device, scaleBytes * (packed ? 2 : 1), U.STORAGE | U.COPY_DST, `${label}.gateS`);
    const upS = makeBuffer(device, scaleBytes, U.STORAGE | U.COPY_DST, `${label}.upS`);
    const gate = makeBuffer(device, bytes * (packed ? 2 : 1), U.STORAGE | U.COPY_SRC, `${label}.gate`);
    const up = makeBuffer(device, bytes, U.STORAGE | U.COPY_SRC, `${label}.up`);
    const fused = makeBuffer(device, bytes, U.STORAGE | U.COPY_SRC, `${label}.fused`);
    const zeroBias = makeBuffer(device, baselineN * 4, U.STORAGE | U.COPY_DST, `${label}.bias`);
    const partialFloats = Math.max(1, mmSplits(T, baselineN, K, target, 16 * rowsPerThread(T), 64) * T * baselineN);
    const part = makeBuffer(device, partialFloats * 4, U.STORAGE, `${label}.partials`);
    const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${label}.globals`);
    const params = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${label}.params`);
    const pairParams = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${label}.pair-params`);
    const geluParams = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${label}.gelu-params`);
    const zeros = new Float32Array(N);
    await checked(device, async () => {
      device.queue.writeBuffer(X, 0, input.x);
      device.queue.writeBuffer(gateW, 0, input.gateW);
      device.queue.writeBuffer(upW, 0, input.upW);
      device.queue.writeBuffer(gateS, 0, input.gateS);
      if (packed) {
        device.queue.writeBuffer(gateW, weightBytes, input.upW);
        device.queue.writeBuffer(gateS, scaleBytes, input.upS);
      }
      device.queue.writeBuffer(upS, 0, input.upS);
      device.queue.writeBuffer(zeroBias, 0, zeros);
      device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
      device.queue.writeBuffer(params, 0, new Uint32Array([baselineN, K, 0, 0]));
      device.queue.writeBuffer(pairParams, 0, new Uint32Array([N, K, packed ? N : 0, activation]));
      device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
      await device.queue.onSubmittedWorkDone();
    }, `${label} input upload`, uncaptured);

    const rows = rowsPerThread(T);
    const generic = genericPipelines.get(rows);
    const gateGroup = device.createBindGroup({ layout: mmLayout, entries: [globals, params, X, gateW, gateS, zeroBias, gate, part].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const upGroup = device.createBindGroup({ layout: mmLayout, entries: [globals, params, X, upW, upS, zeroBias, up, part].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const gateReduce = device.createBindGroup({ layout: reduceLayoutObject, entries: [globals, params, part, zeroBias, gate].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const upReduce = device.createBindGroup({ layout: reduceLayoutObject, entries: [globals, params, part, zeroBias, up].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const geluBuffers = packed ? [globals, geluParams, gate, fused] : [globals, geluParams, gate, up, fused];
    const geluGroup = device.createBindGroup({ layout: geluLayout, entries: geluBuffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const fusedGroup = device.createBindGroup({ layout: pairLayout, entries: [globals, pairParams, X, gateW, gateS, packed ? gateW : upW, packed ? gateS : upS, fused].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const gateOp = { N: baselineN, K, group: gateGroup, reduce: gateReduce, pMatmul: generic.mm, pReduce: generic.reduce };
    const upOp = { N, K, group: upGroup, reduce: upReduce, pMatmul: generic.mm, pReduce: generic.reduce };
    const baselineSequence = (pass) => {
      dispatchMatmul(pass, generic.mm, generic.reduce, gateOp, T, target, rows, 1);
      if (!packed) dispatchMatmul(pass, generic.mm, generic.reduce, upOp, T, target, rows, 1);
      dispatchGelu(pass, geluPipeline, geluGroup, T, N);
    };
    const fusedSequence = (pass) => dispatchFused(pass, pairPipeline, fusedGroup, T, N);
    const reps = Math.max(1, Math.min(32, Math.round(2e9 / Math.max(1, 2 * T * N * K))));
    for (let i = 0; i < warmups; i++) {
      await runSequence(device, null, i & 1 ? fusedSequence : baselineSequence, 1, uncaptured, `${label} warmup`);
    }
    const timer = makeTimer(device, reps);
    const baselineSamples = [];
    const fusedSamples = [];
    for (let i = 0; i < samples; i++) {
      if (i & 1) {
        fusedSamples.push(await runSequence(device, timer, fusedSequence, reps, uncaptured, `${label} fused sample`));
        baselineSamples.push(await runSequence(device, timer, baselineSequence, reps, uncaptured, `${label} baseline sample`));
      } else {
        baselineSamples.push(await runSequence(device, timer, baselineSequence, reps, uncaptured, `${label} baseline sample`));
        fusedSamples.push(await runSequence(device, timer, fusedSequence, reps, uncaptured, `${label} fused sample`));
      }
    }
    const baselineMedian = median(baselineSamples);
    const fusedMedian = median(fusedSamples);
    baselineMedians.push(baselineMedian);
    fusedMedians.push(fusedMedian);

    await runSequence(device, null, baselineSequence, 1, uncaptured, `${label} baseline correctness`);
    const baselineOutput = await readBuffer(device, fused, bytes, uncaptured, `${label} baseline output`);
    await runSequence(device, null, fusedSequence, 1, uncaptured, `${label} fused correctness`);
    const fusedOutput = await readBuffer(device, fused, bytes, uncaptured, `${label} fused output`);
    const finiteBaseline = finiteSummary(baselineOutput, `${label} baseline`);
    const finiteFused = finiteSummary(fusedOutput, `${label} fused`);
    const cpuBaseline = compareCpu(input, baselineOutput, `${label} baseline`);
    const cpuFused = compareCpu(input, fusedOutput, `${label} fused`);
    const variant = compareArrays(baselineOutput, fusedOutput);
    correctness.cpu.push({ shape: label, baseline: cpuBaseline, fused: cpuFused });
    correctness.variants.push({ shape: label, ...variant });
    if (!finiteBaseline.ok || !finiteFused.ok || !cpuBaseline.ok || !cpuFused.ok || !variant.ok) correctness.ok = false;
    const entry = {
      shape: label,
      T,
      N,
      K,
      pattern,
      pairSplits,
      baselineSplits: mmSplits(T, baselineN, K, target, 16 * rows, 64),
      baselineRows: rows,
      fusedRows: 4,
      baselineDispatches: (packed ? 1 : 2) * (mmSplits(T, baselineN, K, target, 16 * rows, 64) > 1 ? 2 : 1) + 1,
      fusedDispatches: 1,
      reps,
      baselineSamples,
      fusedSamples,
      baselineMedianMs: baselineMedian,
      fusedMedianMs: fusedMedian,
      speedup: baselineMedian / fusedMedian,
      cpuSamples: cpuFused.samples,
      variantMaxAbs: variant.maxAbs,
      variantMaxRelative: variant.maxRelative,
    };
    cases.push(entry);
    log(`${label}: baseline=${baselineMedian.toFixed(3)}ms fused=${fusedMedian.toFixed(3)}ms speedup=${entry.speedup.toFixed(3)}x baselineSplits=${entry.baselineSplits}`);
    timer.querySet?.destroy();
    timer.resolved?.destroy();
    timer.read?.destroy();
    for (const buffer of [X, gateW, upW, gateS, upS, gate, up, fused, zeroBias, part, globals, params, pairParams, geluParams]) buffer.destroy();
  }

  const geometric = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
  const metricMs = geometric(fusedMedians);
  const baselineMetricMs = geometric(baselineMedians);
  const result = {
    status: correctness.ok ? "done" : "error",
    backend: "webgpu",
    metricMs,
    metric: "geometric mean of fused gate/up+GELU median GPU milliseconds",
    baselineMetricMs,
    activation,
    packed,
    fusedDispatches: 1,
    method: device.features.has("timestamp-query") ? "WebGPU timestamp-query" : "performance.now() wall clock",
    seed,
    warmups,
    samples,
    target,
    cases,
    adapter: provenance,
    correctness,
  };
  if (!correctness.ok) result.error = "fused gate/up correctness checks failed";
  window.gpuBench = result;
  log(`metricMs=${metricMs.toFixed(4)} baselineMetricMs=${baselineMetricMs.toFixed(4)} method=${result.method}`);
  device.destroy();
  activeDevice = null;
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().catch((error) => {
  const message = `${error?.message || error}\n${error?.stack || ""}`;
  const previous = window.gpuBench && typeof window.gpuBench === "object" ? window.gpuBench : {};
  window.gpuBench = { ...previous, status: "error", backend: "webgpu", error: message };
  log(`ERROR ${message}`);
}).finally(() => {
  if (activeDevice) activeDevice.destroy();
});
