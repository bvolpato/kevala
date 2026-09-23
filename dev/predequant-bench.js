import { kernelSource } from "./wgsl.js";
import { dispatchMatmul, matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice } from "../js/src/gpu.js";

// Experimental weight pre-dequantization. Total includes rebuilding f32 scratch every operation;
// cached timing measures matmul plus reduction after that scratch is already populated.
const TOLERANCE = { abs: 0.0005, relative: 0.0001, relativeFloor: 1 };
const DEFAULT_CASES = [[3, 70, 96], [47, 5248, 1024], [99, 6144, 1536], [198, 2048, 6144], [512, 5248, 1024], [606, 12288, 2048]];
const GUARDS = [
  { shape: [1, 1, 32], kind: "zero" },
  { shape: [17, 65, 96], kind: "outlier" },
  { shape: [19, 67, 256], kind: "cancellation" },
  { shape: [65, 67, 128], kind: "random" },
  { shape: [3, 70, 32], kind: "block_reset" },
  { shape: [3, 70, 64], kind: "block_reset" },
  { shape: [3, 70, 96], kind: "block_reset" },
];
const log = (line) => { document.getElementById("log").textContent += `${line}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const label = ([T, N, K]) => `${T}x${N}x${K}`;
let activeDevice;

function numberParam(query, name, fallback, min, max) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${name} must be an integer in [${min}, ${max}]`);
  return value;
}

function validateShape(shape) {
  if (shape.length !== 3 || shape.some((n) => !Number.isSafeInteger(n) || n <= 0)) throw new Error("cases must be comma-separated positive integer T x N x K shapes");
  const [T, N, K] = shape;
  if (K % 32) throw new Error(`K must be a multiple of 32, got ${K}`);
  if (Math.ceil(T / 64) > 65535 || Math.ceil(N / 64) > 65535 || T * K > 0xffffffff || N * K > 0xffffffff || 8 * T * N > 0xffffffff) {
    throw new Error(`shape exceeds shader indexing or dispatch limits: ${label(shape)}`);
  }
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

function makeInput(T, N, K, seed, kind = "random") {
  validateShape([T, N, K]);
  const random = makeRng(seed);
  const x = new Float32Array(T * K);
  const w = new Int8Array(N * K);
  const scales = new Float32Array(N * K / 32);
  const bias = new Float32Array(N);
  const residual = new Float32Array(T * N);
  for (let i = 0; i < x.length; i++) x[i] = (random() - 0.5) * 0.8;
  for (let i = 0; i < w.length; i++) w[i] = Math.floor(random() * 256) - 128;
  for (let i = 0; i < scales.length; i++) scales[i] = 0.005 + random() * 0.012;
  for (let i = 0; i < bias.length; i++) bias[i] = (random() - 0.5) * 0.2;
  for (let i = 0; i < residual.length; i++) residual[i] = (random() - 0.5) * 0.4;
  if (kind === "zero") x.fill(0);
  if (kind === "outlier") {
    for (let i = 0; i < x.length; i++) x[i] = i % 32 === 0 ? (i % 64 === 0 ? 16 : -16) : (random() - 0.5) * 0.04;
  }
  if (kind === "cancellation") {
    for (let i = 0; i < x.length; i += 2) {
      x[i] = 3 + random() * 0.01;
      x[i + 1] = -x[i] + (random() - 0.5) * 0.0001;
    }
    for (let i = 0; i < w.length; i += 2) w[i + 1] = w[i];
  }
  if (kind === "block_reset") {
    // Exact values expose block addressing and accumulation errors at K32/K64/K96.
    for (let i = 0; i < x.length; i++) x[i] = [127, 254, -127][Math.floor((i % K) / 32) % 3];
    for (let i = 0; i < w.length; i++) w[i] = Math.floor(i / K) % 2 === 0 ? 1 : -1;
    scales.fill(1);
  }
  return { T, N, K, x, w, scales, bias, residual, kind };
}

function validateInput(input) {
  validateShape([input.T, input.N, input.K]);
  for (const key of ["x", "scales", "bias", "residual"]) {
    for (let i = 0; i < input[key].length; i++) {
      if (!Number.isFinite(input[key][i])) throw new Error(`${key}[${i}] must be finite`);
    }
  }
}

function rejectionChecks() {
  const failures = [];
  for (const value of [NaN, Infinity, -Infinity]) {
    const input = makeInput(1, 1, 32, 1);
    input.x[0] = value;
    let rejected = false;
    try { validateInput(input); } catch (error) { rejected = error.message === "x[0] must be finite"; }
    if (!rejected) failures.push(`accepted ${String(value)}`);
  }
  let rejectedK = false;
  try { validateShape([1, 1, 33]); } catch (error) { rejectedK = error.message === "K must be a multiple of 32, got 33"; }
  if (!rejectedK) failures.push("accepted K=33");
  return { ok: failures.length === 0, rejected: ["NaN", "+Infinity", "-Infinity", "K=33"], failures };
}

function cpuValue(input, index, mode, withBias) {
  const { N, K, x, w, scales, bias, residual } = input;
  const row = Math.floor(index / N);
  const col = index % N;
  let sum = 0;
  for (let k = 0; k < K; k++) {
    const weight = Math.fround(w[col * K + k] * scales[col * K / 32 + Math.floor(k / 32)]);
    sum = Math.fround(sum + Math.fround(x[row * K + k] * weight));
  }
  if (withBias) sum = Math.fround(sum + bias[col]);
  if (mode === 1) sum = Math.fround(sum + residual[index]);
  if (mode === 2) sum = Math.max(0, sum);
  return sum;
}

function cpuIndices(T, N) {
  if (T * N <= 8192) return Array.from({ length: T * N }, (_, i) => i);
  const rows = [...new Set([0, 1, 15, 16, 31, 32, 63, 64, Math.floor(T / 2), T - 2, T - 1])].filter((r) => r >= 0 && r < T);
  const cols = [...new Set([0, 1, 15, 16, 31, 32, 63, 64, Math.floor(N / 2), N - 2, N - 1])].filter((c) => c >= 0 && c < N);
  const indices = new Set(rows.flatMap((row) => cols.map((col) => row * N + col)));
  const random = makeRng(T ^ N);
  for (let i = 0; i < 128; i++) indices.add(Math.floor(random() * T * N));
  return [...indices];
}

function compare(actual, expectedAt, indices) {
  let mismatches = 0;
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  let samples = 0;
  const examples = [];
  for (const index of indices) {
    samples++;
    const expected = expectedAt(index);
    const value = actual[index];
    const finite = Number.isFinite(value) && Number.isFinite(expected);
    const abs = finite ? Math.abs(value - expected) : Infinity;
    const relative = finite ? abs / Math.max(1, Math.abs(expected)) : Infinity;
    if (abs > maxAbs) { maxAbs = abs; worst = { index, actual: value, expected }; }
    maxRelative = Math.max(maxRelative, relative);
    if (!finite || abs > TOLERANCE.abs + TOLERANCE.relative * Math.max(1, Math.abs(expected))) {
      mismatches++;
      if (examples.length < 4) examples.push({ index, actual: value, expected, abs });
    }
  }
  return { ok: mismatches === 0, samples, mismatches, maxAbs, maxRelative, worst, examples };
}

async function checked(gpu, phase, fn, uncaptured) {
  const { device } = gpu;
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let result;
  let failure;
  try { result = await fn(); } catch (error) { failure = error; }
  const scopes = await Promise.allSettled([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  const errors = scopes.flatMap((scope) => scope.status === "fulfilled" ? scope.value ? [scope.value.message] : [] : [String(scope.reason)]);
  if (errors.length) throw new Error(`${phase}: ${errors.join("; ")}`);
  if (failure) throw failure;
  if (uncaptured.length) throw new Error(`${phase}: ${uncaptured.splice(0).join("; ")}`);
  if (gpu.lost) throw new Error(`${phase}: device lost: ${gpu.lost.message || gpu.lost.reason}`);
  return result;
}

function makeBuffer(device, size, usage, label) {
  if (size > device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && size > device.limits.maxStorageBufferBindingSize)) {
    throw new Error(`${label}: requested ${size} bytes exceeds device buffer limits`);
  }
  return device.createBuffer({ size: Math.max(16, size), usage, label });
}

function bind(device, layout, buffers) {
  return device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
}

function dequantizeDispatch(pass, run) {
  const n = Math.ceil(run.N * run.K / 4 / 256);
  pass.setPipeline(run.pDequantize);
  pass.setBindGroup(0, run.dequantize);
  pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
}

function encode(pass, run, phase) {
  if (run.name === "predequant" && (phase === "total" || phase === "dequantize")) dequantizeDispatch(pass, run);
  if (phase === "dequantize") return;
  if (phase === "total" || phase === "cached") {
    dispatchMatmul(pass, run.pMatmul, run.pReduce, run, run.T, run.target, 4, 1);
  } else if (phase === "matmul") {
    pass.setPipeline(run.pMatmul);
    pass.setBindGroup(0, run.group);
    pass.dispatchWorkgroups(Math.ceil(run.N / 64), Math.ceil(run.T / 64), run.splits);
  } else if (phase === "reduce" && run.splits > 1) {
    const n = Math.ceil(run.T * run.N / 256);
    pass.setPipeline(run.pReduce);
    pass.setBindGroup(0, run.reduce);
    pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
  }
}

async function timedBatch(ctx, run, phase, reps) {
  const { gpu, timer, uncaptured } = ctx;
  const { device } = gpu;
  return checked(gpu, `${run.name}.${phase}`, async () => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    for (let i = 0; i < reps; i++) encode(pass, run, phase);
    pass.end();
    encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolved, 0);
    encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await timer.read.mapAsync(GPUMapMode.READ, 0, 16);
    const timestamps = new BigUint64Array(timer.read.getMappedRange(0, 16));
    const ms = Number(timestamps[1] - timestamps[0]) / 1e6 / reps;
    timer.read.unmap();
    if (!Number.isFinite(ms) || ms < 0) throw new Error(`invalid timestamp duration ${ms}`);
    return ms;
  }, uncaptured);
}

async function readOutput(ctx, output, count) {
  const { gpu, uncaptured } = ctx;
  const { device } = gpu;
  const read = makeBuffer(device, count * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "output.read");
  try {
    return await checked(gpu, "read output", async () => {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(output, 0, read, 0, count * 4);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ, 0, count * 4);
      const values = new Float32Array(read.getMappedRange(0, count * 4).slice(0));
      read.unmap();
      return values;
    }, uncaptured);
  } finally { read.destroy(); }
}

async function runCase(ctx, input, measured, options, correctness) {
  validateInput(input);
  const { gpu, layouts, pipes, uncaptured } = ctx;
  const { device } = gpu;
  const { T, N, K, kind } = input;
  const U = GPUBufferUsage;
  const target = options.target;
  const splits = mmSplits(T, N, K, target, 64, 64);
  const resources = [];
  const alloc = (bytes, usage, name) => {
    const buffer = makeBuffer(device, bytes, usage, name);
    resources.push(buffer);
    return buffer;
  };
  try {
    const buffers = await checked(gpu, "allocate inputs", async () => {
      const globals = alloc(16, U.UNIFORM | U.COPY_DST, "globals");
      const params = alloc(16, U.UNIFORM | U.COPY_DST, "params");
      const X = alloc(input.x.byteLength, U.STORAGE | U.COPY_DST, "X");
      const D = alloc(N * K * 4, U.STORAGE | U.COPY_SRC, "W.dequantized_f32");
      const W = alloc(input.w.byteLength, U.STORAGE | U.COPY_DST, "W");
      const S = alloc(input.scales.byteLength, U.STORAGE | U.COPY_DST, "weight.scales");
      const B = alloc(input.bias.byteLength, U.STORAGE | U.COPY_DST, "bias");
      device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
      for (const [buffer, values] of [[X, input.x], [W, input.w], [S, input.scales], [B, input.bias]]) device.queue.writeBuffer(buffer, 0, values);
      const dequantize = bind(device, layouts.dequantize, [globals, params, W, S, D]);
      const runs = ["generic", "predequant"].map((name) => {
        const Y = alloc(T * N * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, `${name}.Y`);
        const PART = alloc(splits > 1 ? splits * T * N * 4 : 16, U.STORAGE, `${name}.PART`);
        const group = bind(device, layouts.matmul, [globals, params, X, name === "predequant" ? D : W, S, B, Y, PART]);
        const reduce = bind(device, layouts.reduce, [globals, params, PART, B, Y]);
        return { name, T, N, K, target, splits, group, reduce, dequantize, Y, pMatmul: pipes[name], pReduce: pipes.reduce, pDequantize: pipes.dequantize, phases: {} };
      });
      await device.queue.onSubmittedWorkDone();
      return { params, runs, D };
    }, uncaptured);
    const { params, runs, D } = buffers;
    const order = (round) => runs.map((_, i) => runs[(round + i) % runs.length]);
    const reps = options.reps || Math.max(1, Math.min(512, Math.round(2e10 / (2 * T * N * K))));
    if (measured) {
      for (let round = 0; round < options.warmups; round++) {
        for (const run of order(round)) await timedBatch(ctx, run, "total", reps);
      }
      for (const phase of ["total", "dequantize", "matmul", "reduce", "cached"]) {
        const phaseReps = phase === "reduce" ? Math.max(reps, 256) : reps;
        for (const run of runs) run.phases[phase] = { reps: phaseReps, samples: [] };
        for (let round = 0; round < options.samples; round++) {
          for (const run of order(round)) {
            const absent = (phase === "dequantize" && run.name === "generic") || (phase === "reduce" && splits === 1);
            run.phases[phase].samples.push(absent ? 0 : await timedBatch(ctx, run, phase, phaseReps));
          }
        }
        for (const run of runs) run.phases[phase].medianMs = median(run.phases[phase].samples);
      }
      if (runs.some((run) => run.phases.total.medianMs <= 0)) throw new Error("total timestamp duration was zero; increase reps");
    }
    const epilogues = T * N <= 8192 ? [[0, false], [0, true], [1, true], [2, true]] : [[0, false]];
    for (const [mode, withBias] of epilogues) {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, Number(withBias)]));
      const indices = cpuIndices(T, N);
      const reference = new Map(indices.map((index) => [index, cpuValue(input, index, mode, withBias)]));
      const outputs = {};
      for (const run of runs) {
        device.queue.writeBuffer(run.Y, 0, input.residual);
        await timedBatch(ctx, run, "total", 1);
        outputs[run.name] = await readOutput(ctx, run.Y, T * N);
        const cpu = compare(outputs[run.name], (index) => reference.get(index), indices);
        correctness.cpu.push({ shape: label([T, N, K]), kind, kernel: run.name, mode, bias: withBias, coverage: indices.length === T * N ? "full" : "sampled", ...cpu });
        if (!cpu.ok) correctness.ok = false;
        if (run.name === "predequant" && T * N <= 8192) {
          device.queue.writeBuffer(run.Y, 0, input.residual);
          await timedBatch(ctx, run, "cached", 1);
          outputs.cached = await readOutput(ctx, run.Y, T * N);
          const cached = compare(outputs.cached, (index) => reference.get(index), indices);
          correctness.cpu.push({ shape: label([T, N, K]), kind, kernel: "cached", mode, bias: withBias, coverage: "full", ...cached });
          if (!cached.ok) correctness.ok = false;
        }
      }
      for (const kernel of outputs.cached ? ["predequant", "cached"] : ["predequant"]) {
        const full = compare(outputs[kernel], (index) => outputs.generic[index], outputs[kernel].keys());
        correctness.variants.push({ shape: label([T, N, K]), kind, kernel, mode, bias: withBias, coverage: "full", ...full });
        if (!full.ok) correctness.ok = false;
      }
    }
    if (N * K <= 65536) {
      const weights = await readOutput(ctx, D, N * K);
      const expected = Float32Array.from(input.w, (weight, i) => weight * input.scales[Math.floor(i / 32)]);
      const actualBits = new Uint32Array(weights.buffer);
      const expectedBits = new Uint32Array(expected.buffer);
      let mismatches = 0;
      for (let i = 0; i < weights.length; i++) if (actualBits[i] !== expectedBits[i]) mismatches++;
      correctness.dequantizedWeights.push({ shape: label([T, N, K]), kind, coverage: "full", comparison: "bitwise f32", samples: weights.length, mismatches, ok: mismatches === 0 });
      if (mismatches) correctness.ok = false;
    }
    const timings = runs.map((run) => ({ kernel: run.name, rows: 4, groups: 1, splitTarget: target, splits, reps, medianMs: run.phases.total?.medianMs ?? null, phases: run.phases }));
    const speedup = measured ? timings[0].medianMs / timings[1].medianMs : null;
    const cachedSpeedup = measured ? timings[0].medianMs / timings[1].phases.cached.medianMs : null;
    log(`${label([T, N, K])} ${kind}${measured ? `: generic=${timings[0].medianMs.toFixed(4)}ms predequant total=${timings[1].medianMs.toFixed(4)}ms speedup=${speedup.toFixed(3)}x cached=${timings[1].phases.cached.medianMs.toFixed(4)}ms` : " guard"}; splits=${splits}`);
    return { shape: label([T, N, K]), T, N, K, kind, measured, timings, speedup, cachedSpeedup, dequantizedWeightBytes: N * K * 4, packedWeightBytes: N * K, scaleBytes: input.scales.byteLength };
  } finally {
    for (const buffer of resources) buffer.destroy();
  }
}

async function main() {
  const query = new URLSearchParams(location.search);
  const options = {
    seed: numberParam(query, "seed", 0x4b455641, 1, 0xffffffff),
    target: numberParam(query, "target", 256, 1, 1000000),
    warmups: numberParam(query, "warmups", 2, 0, 10),
    samples: numberParam(query, "samples", 7, 3, 20),
    reps: numberParam(query, "reps", 0, 1, 4096),
  };
  const shapes = query.has("cases") ? query.get("cases").split(",").map((shape) => shape.split("x").map(Number)) : DEFAULT_CASES;
  for (const shape of shapes) validateShape(shape);
  const gpu = await requestDevice({ features: ["timestamp-query"] });
  activeDevice = gpu.device;
  const { device } = gpu;
  if (!device.features.has("timestamp-query")) throw new Error("pre-dequantization benchmark requires timestamp-query for comparable GPU measurements");
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  const source = await kernelSource(query.has("wasm") ? new URL(query.get("wasm"), location.href).href : undefined);
  const spec = { f16: false, rows: 4, groups: 1, splitTarget: options.target };
  const codes = { generic: source("matmul", spec), predequant: source("matmul_f32", spec), dequantize: source("dequant_q8", spec), reduce: source("reduce", spec) };
  const hash = async (text) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  const shaderHashes = Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)])));
  const info = gpu.adapter.info || {};
  const provenance = {
    experiment: "FP32 pre-dequantized Q8 weights with unchanged generic tiled matrix arithmetic",
    sourceBaseCommit: "c727398",
    wasmUrl: source.wasmUrl,
    shaderHashes,
    spec,
    page: location.href,
    userAgent: navigator.userAgent,
    timestamp: new Date().toISOString(),
    wgslLanguageFeatures: [...(navigator.gpu.wgslLanguageFeatures || [])].sort(),
    note: "The total metric includes dequantization on every matrix operation. Cached timing excludes dequantization and assumes the N*K*4-byte scratch survives between operations; persistent caching is not integrated into the runtime.",
  };
  const adapter = {
    name: gpu.name, vendor: info.vendor || null, architecture: info.architecture || null,
    device: info.device || null, description: info.description || null,
    isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter),
    features: [...device.features].sort(), adapterFeatures: [...gpu.adapter.features].sort(),
    powerPreference: gpu.powerPreference,
    limits: { maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize },
  };
  window.gpuBench = { status: "running", backend: "webgpu", adapter, provenance };
  const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  const layouts = {
    matmul: matmulLayout(device), reduce: reduceLayout(device),
    dequantize: device.createBindGroupLayout({ entries: [entry(0, "uniform"), entry(1, "uniform"), entry(2, "read-only-storage"), entry(3, "read-only-storage"), entry(4, "storage")] }),
  };
  const pipes = await checked(gpu, "compile pipelines", async () => {
    const out = {};
    for (const [name, code] of Object.entries(codes)) {
      const layout = layouts[name === "generic" || name === "predequant" ? "matmul" : name];
      out[name] = await pipeline(device, code, `predequant-bench.${name}`, device.createPipelineLayout({ bindGroupLayouts: [layout] }));
    }
    return out;
  }, uncaptured);
  const U = GPUBufferUsage;
  const timer = {
    querySet: device.createQuerySet({ type: "timestamp", count: 2 }),
    resolved: makeBuffer(device, 16, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp.resolved"),
    read: makeBuffer(device, 256, U.MAP_READ | U.COPY_DST, "timestamp.read"),
  };
  try {
    const ctx = { gpu, timer, layouts, pipes, uncaptured };
    const rejectedInputs = rejectionChecks();
    const correctness = { ok: rejectedInputs.ok, tolerance: TOLERANCE, rejectedInputs, cpu: [], variants: [], dequantizedWeights: [] };
    const cases = [];
    for (let i = 0; i < shapes.length; i++) {
      cases.push(await runCase(ctx, makeInput(...shapes[i], (options.seed + i * 0x9e3779b9) >>> 0), true, options, correctness));
    }
    for (let i = 0; i < GUARDS.length; i++) {
      cases.push(await runCase(ctx, makeInput(...GUARDS[i].shape, (options.seed + i + 1) >>> 0, GUARDS[i].kind), false, options, correctness));
    }
    const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
    const measured = cases.filter((item) => item.measured);
    const metricMs = geometricMean(measured.map((item) => item.timings[1].medianMs));
    const baselineMetricMs = geometricMean(measured.map((item) => item.timings[0].medianMs));
    const cachedMetricMs = geometricMean(measured.map((item) => item.timings[1].phases.cached.medianMs));
    window.gpuBench = {
      status: "done", backend: "webgpu", method: "WebGPU timestamp-query", metricMs, baselineMetricMs,
      speedup: baselineMetricMs / metricMs, cachedMetricMs, cachedSpeedup: baselineMetricMs / cachedMetricMs,
      metric: "geometric mean of per-shape median total GPU milliseconds, including weight dequantization and split reduction",
      phaseTiming: "Each phase is measured separately; total measures the actual dequantize, matmul, reduce dispatch sequence. Phase medians are diagnostic and are not added to form total.",
      timingOrder: "Alternated generic/predequant order each sample round, with identical split targets and BM64/BN64 tiles",
      cachedTiming: "Measured matmul plus split reduction over previously dequantized scratch; matmul-only and reduction-only phases are also reported separately.",
      kernels: ["matmul:f32:R4", "dequant_q8+matmul_f32:R4"], options, cases, adapter, provenance, correctness,
    };
    log(`correctness.ok=${correctness.ok}; total metric=${metricMs.toFixed(4)}ms; speedup=${(baselineMetricMs / metricMs).toFixed(3)}x`);
  } finally {
    timer.querySet.destroy();
    timer.resolved.destroy();
    timer.read.destroy();
  }
}

// These pure helpers can be checked under Node without requesting an adapter or running GPU work.
export { compare, cpuIndices, cpuValue, encode, makeInput, rejectionChecks, validateInput, validateShape };
if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", backend: "webgpu" };
  main().catch((error) => {
    window.gpuBench = { ...window.gpuBench, status: "error", backend: "webgpu", error: `${error.message || error}\n${error.stack || ""}` };
    log(`ERROR ${error.message || error}`);
  }).finally(() => activeDevice?.destroy());
}
