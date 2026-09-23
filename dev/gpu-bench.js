import { kernelSource } from "./wgsl.js";
import { dispatchMatmul, matmulConfig, matmulLayout, mmSplits, reduceLayout, requestDevice, rowsPerThread, pipeline } from "../js/src/gpu.js";

const logNode = document.getElementById("log");
const log = (line) => {
  logNode.textContent += `${line}\n`;
};

let U;
const lostDevices = new WeakMap();
let activeDevice = null;
const DEFAULT_CASES = [
  [44, 3072, 1024],
  [44, 5248, 1024],
  [44, 1024, 2624],
  [44, 1024, 1024],
  [44, 4096, 1024],
  [44, 1024, 4096],
  [140, 3072, 1024],
  [512, 3072, 1024],
  [512, 5248, 1024],
  [32, 8192, 1024],
  [128, 1024, 3584],
  [128, 1024, 2048],
  [128, 5120, 1024],
  [512, 7168, 1024],
  [45, 132, 96],
];

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

function parseKernel(raw, T, config) {
  const f16Available = config.f16;
  if (raw === "runtime") {
    return { label: raw, name: "matmul", f16: config.f16, target: config.splitTarget, rows: rowsPerThread(T), groups: config.groups };
  }
  const match = raw.match(/^([a-z_]+)(?:@(\d+))?(?::(\w+))?(?:\/(\d+))?$/);
  if (!match || !["matmul", "matmul_h", "matmul_wide", "matmul_wide_h"].includes(match[1])) {
    throw new Error(`invalid kernel variant ${raw}`);
  }
  const rows = match[3] === "auto" ? rowsPerThread(T) : match[3] ? Number(match[3]) : 4;
  const groups = match[4] ? Number(match[4]) : 1;
  const target = match[2] ? Number(match[2]) : 128;
  const f16 = match[1].endsWith("_h");
  if (![1, 2, 3, 4].includes(rows) || ![1, 2].includes(groups)) throw new Error(`invalid rows/groups in ${raw}`);
  if (match[1].startsWith("matmul_wide") && groups !== 1) throw new Error(`${raw} requires groups=1`);
  if (!Number.isInteger(target) || target < 1 || target > (0xffffffff - 3) / 3) throw new Error(`invalid split target in ${raw}`);
  if (f16 && !f16Available) throw new Error(`${raw} requires the shader-f16 feature`);
  return { label: raw, name: f16 ? match[1].slice(0, -2) : match[1], f16, target, rows, groups };
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

function makeInput(T, N, K, seed) {
  if (K % 32) throw new Error(`K must be a multiple of 32, got ${K}`);
  const random = makeRng(seed);
  const x = new Float32Array(T * K);
  const w = new Uint8Array(N * K);
  const scales = new Float32Array(N * (K / 32));
  const bias = new Float32Array(N);
  const residual = new Float32Array(T * N);
  for (let i = 0; i < x.length; i++) x[i] = Math.fround((random() - 0.5) * 0.8);
  for (let i = 0; i < w.length; i++) w[i] = Math.floor(random() * 256);
  for (let i = 0; i < scales.length; i++) scales[i] = Math.fround(0.005 + random() * 0.012);
  for (let i = 0; i < bias.length; i++) bias[i] = Math.fround((random() - 0.5) * 0.2);
  for (let i = 0; i < residual.length; i++) residual[i] = Math.fround((random() - 0.5) * 0.4);
  return { T, N, K, x, w, scales, bias, residual };
}

function signedByte(value) {
  return value < 128 ? value : value - 256;
}

function cpuValue(input, row, col, mode, withBias) {
  const { T, N, K, x, w, scales, bias, residual } = input;
  let sum = 0;
  const blocks = K / 32;
  for (let k = 0; k < K; k++) {
    const weight = signedByte(w[col * K + k]) * scales[col * blocks + Math.floor(k / 32)];
    sum = Math.fround(sum + Math.fround(x[row * K + k] * weight));
  }
  if (withBias) sum = Math.fround(sum + bias[col]);
  if (mode === 1) sum = Math.fround(sum + residual[row * N + col]);
  if (mode === 2) sum = Math.max(sum, 0);
  return sum;
}

function sampleIndices(T, N, all = false) {
  if (all) return Array.from({ length: T * N }, (_, i) => i);
  const rows = [...new Set([0, 1, 2, Math.floor(T / 2), Math.max(0, T - 3), Math.max(0, T - 2), T - 1])].filter((r) => r < T);
  const cols = [...new Set([0, 1, 2, Math.floor(N / 2), Math.max(0, N - 3), Math.max(0, N - 2), N - 1])].filter((c) => c < N);
  const out = new Set();
  for (const row of rows) for (const col of cols) out.add(row * N + col);
  return [...out];
}

function finiteSummary(values) {
  for (const value of values) if (!Number.isFinite(value)) return { ok: false, message: "GPU output contains a non-finite value" };
  return { ok: true };
}

function compareCpu(input, values, kernel, mode, withBias) {
  const all = input.T * input.N <= 8192;
  const indices = sampleIndices(input.T, input.N, all);
  const absTolerance = kernel.f16 ? 0.03 : 0.0005;
  const relativeTolerance = kernel.f16 ? 0.002 : 0.0001;
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  for (const index of indices) {
    const row = Math.floor(index / input.N);
    const col = index % input.N;
    const expected = cpuValue(input, row, col, mode, withBias);
    const actual = values[index];
    if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
      return { ok: false, message: `non-finite result at row ${row}, col ${col}`, mode, withBias };
    }
    const abs = Math.abs(actual - expected);
    const relative = abs / Math.max(1, Math.abs(expected));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { row, col, expected, actual };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > absTolerance + relativeTolerance * Math.max(1, Math.abs(expected))) {
      return { ok: false, samples: indices.length, maxAbs, maxRelative, worst, message: `${kernel.label} CPU mismatch at row ${row}, col ${col}` };
    }
  }
  return { ok: true, samples: indices.length, maxAbs, maxRelative, worst };
}

function compareVariants(outputs, labels, tolerance) {
  if (labels.length < 2) return { ok: true, samples: 0, maxAbs: 0, maxRelative: 0 };
  const baseline = outputs[labels[0]];
  let maxAbs = 0;
  let maxRelative = 0;
  for (const label of labels.slice(1)) {
    const values = outputs[label];
    for (let i = 0; i < baseline.length; i++) {
      const abs = Math.abs(values[i] - baseline[i]);
      maxAbs = Math.max(maxAbs, abs);
      maxRelative = Math.max(maxRelative, abs / Math.max(1, Math.abs(baseline[i])));
      if (abs > tolerance.abs + tolerance.relative * Math.max(1, Math.abs(baseline[i]))) {
        return { ok: false, compared: [labels[0], label], maxAbs, maxRelative, message: `kernel mismatch at output ${i}: ${labels[0]} vs ${label}` };
      }
    }
  }
  return { ok: true, compared: labels, maxAbs, maxRelative };
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
  if (uncaptured.length) {
    const error = uncaptured.splice(0).map((item) => item.message).join("; ");
    throw new Error(`WebGPU ${phase}: uncaptured error: ${error}`);
  }
  const lost = lostDevices.get(device);
  if (lost) throw new Error(`WebGPU ${phase}: device lost: ${lost.message || lost.reason || "unknown reason"}`);
  return value;
}

function makeBuffer(device, size, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage });
}

async function readOutput(device, output, bytes, uncaptured, label) {
  const readback = makeBuffer(device, bytes, U.MAP_READ | U.COPY_DST, `${label}.readback`);
  try {
    const values = await checked(device, async () => {
      const encoder = device.createCommandEncoder({ label: `${label}.readback` });
      encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ, 0, bytes);
      const copy = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
      readback.unmap();
      return copy;
    }, `${label} readback`, uncaptured);
    return values;
  } finally {
    readback.destroy();
  }
}

function makeTimer(device, useTimestamps) {
  if (!useTimestamps) return { method: "performance.now() wall clock", timestamp: false };
  return {
    method: "WebGPU timestamp-query",
    timestamp: true,
    querySet: device.createQuerySet({ type: "timestamp", count: 2, label: "gpu-bench.timestamps" }),
    resolved: makeBuffer(device, 16, U.QUERY_RESOLVE | U.COPY_SRC, "gpu-bench.timestamp-resolved"),
    read: makeBuffer(device, 256, U.MAP_READ | U.COPY_DST, "gpu-bench.timestamp-read"),
  };
}

async function timedBatch(device, timer, op, T, reps, target, rows, groups, uncaptured, phase) {
  return checked(device, async () => {
    const encoder = device.createCommandEncoder({ label: phase });
    const pass = timer.timestamp
      ? encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })
      : encoder.beginComputePass();
    for (let i = 0; i < reps; i++) dispatchMatmul(pass, op.pMatmul, op.pReduce, op, T, target, rows, groups);
    pass.end();
    const started = performance.now();
    if (timer.timestamp) {
      encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolved, 0);
      encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16);
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (timer.timestamp) {
      await timer.read.mapAsync(GPUMapMode.READ, 0, 16);
      const timestamps = new BigInt64Array(timer.read.getMappedRange(0, 16));
      const elapsed = Number(timestamps[1] - timestamps[0]) / 1e6;
      timer.read.unmap();
      return elapsed / reps;
    }
    return (performance.now() - started) / reps;
  }, phase, uncaptured);
}

function caseLabel([T, N, K]) {
  return `${T}x${N}x${K}`;
}

async function main() {
  const query = new URLSearchParams(location.search);
  const seed = numberParam(query, "seed", 0x4b455641, 1, 0xffffffff);
  const warmups = numberParam(query, "warmups", 2, 0, 10);
  const samples = numberParam(query, "samples", 7, 3, 20);
  const requestedKernels = (query.get("kernels") || "runtime").split(",").filter(Boolean);
  const shapes = parseCases(query);
  const gpu = await requestDevice();
  U = GPUBufferUsage;
  const device = gpu.device;
  activeDevice = device;
  lostDevices.set(device, null);
  device.lost.then((info) => lostDevices.set(device, info));
  if (gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter) throw new Error("software WebGPU adapter is not allowed");
  const uncaptured = [];
  const onError = (event) => uncaptured.push({ name: event.error?.name || "GPUError", message: event.error?.message || String(event.error) });
  device.addEventListener("uncapturederror", onError);
  const provenance = adapterInfo(gpu);
  const source = await kernelSource();
  const timestampAvailable = device.features.has("timestamp-query");
  const timer = makeTimer(device, timestampAvailable);
  const layouts = { matmul: matmulLayout(device), reduce: reduceLayout(device) };
  const pipelineLayouts = {
    matmul: device.createPipelineLayout({ bindGroupLayouts: [layouts.matmul] }),
    reduce: device.createPipelineLayout({ bindGroupLayouts: [layouts.reduce] }),
  };
  const pipelineCache = new Map();
  const getPipelines = async (variant, T) => {
    const key = `${variant.label}|${T}`;
    if (!pipelineCache.has(key)) {
      const code = source(variant.name, { f16: variant.f16, rows: variant.rows, groups: variant.groups, splitTarget: variant.target });
      const reduceCode = source("reduce", { rows: variant.rows, groups: variant.groups, splitTarget: variant.target });
      pipelineCache.set(key, Promise.all([
        pipeline(device, code, `${variant.label}.matmul`, pipelineLayouts.matmul),
        pipeline(device, reduceCode, `${variant.label}.reduce`, pipelineLayouts.reduce),
      ]));
    }
    const [pMatmul, pReduce] = await pipelineCache.get(key);
    return { pMatmul, pReduce };
  };
  const variantsFor = (T) => requestedKernels.map((name) => parseKernel(name, T, matmulConfig(device)));
  const allVariants = variantsFor(shapes[0][0]);
  const labels = allVariants.map((variant) => variant.label);
  if (new Set(labels).size !== labels.length) throw new Error("kernels contains a duplicate variant");
  const cases = [];
  const correctness = { ok: true, cpu: [], variants: [], errors: [] };
  const medians = [];
  const variantTolerance = { abs: device.features.has("shader-f16") ? 0.03 : 0.0005, relative: device.features.has("shader-f16") ? 0.002 : 0.0001 };

  for (let shapeIndex = 0; shapeIndex < shapes.length; shapeIndex++) {
    const shape = shapes[shapeIndex];
    const [T, N, K] = shape;
    const variantObjects = variantsFor(T);
    const partialFloats = Math.max(0, ...variantObjects.map((variant) => {
      const splits = mmSplits(T, N, K, variant.target, 16 * variant.rows, 64 * variant.groups);
      return splits > 1 ? splits * T * N : 0;
    }));
    const input = makeInput(T, N, K, (seed + shapeIndex * 0x9e3779b9) >>> 0);
    const bytes = T * N * 4;
    const X = makeBuffer(device, input.x.byteLength, U.STORAGE | U.COPY_DST, `${caseLabel(shape)}.X`);
    const W = makeBuffer(device, input.w.byteLength, U.STORAGE | U.COPY_DST, `${caseLabel(shape)}.W`);
    const S = makeBuffer(device, input.scales.byteLength, U.STORAGE | U.COPY_DST, `${caseLabel(shape)}.scales`);
    const B = makeBuffer(device, input.bias.byteLength, U.STORAGE | U.COPY_DST, `${caseLabel(shape)}.bias`);
    const Y = makeBuffer(device, bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, `${caseLabel(shape)}.Y`);
    const part = makeBuffer(device, partialFloats * 4, U.STORAGE, `${caseLabel(shape)}.split-partials`);
    const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${caseLabel(shape)}.globals`);
    const params = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${caseLabel(shape)}.params`);
    const zero = new Float32Array(T * N);
    await checked(device, async () => {
      device.queue.writeBuffer(X, 0, input.x);
      device.queue.writeBuffer(W, 0, input.w);
      device.queue.writeBuffer(S, 0, input.scales);
      device.queue.writeBuffer(B, 0, input.bias);
      device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
      device.queue.writeBuffer(Y, 0, zero);
      await device.queue.onSubmittedWorkDone();
    }, `${caseLabel(shape)} input upload`, uncaptured);

    const outputs = {};
    const timings = [];
    const shouldCheckEpilogues = T * N <= 8192;
    const runs = [];
    for (const variant of variantObjects) {
      const pipes = await getPipelines(variant, T);
      const group = device.createBindGroup({ layout: layouts.matmul, entries: [globals, params, X, W, S, B, Y, part].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const reduce = device.createBindGroup({ layout: layouts.reduce, entries: [globals, params, part, B, Y].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const op = { N, K, group, reduce, pMatmul: pipes.pMatmul, pReduce: pipes.pReduce };
      const splits = mmSplits(T, N, K, variant.target, 16 * variant.rows, 64 * variant.groups);
      // about 20 GFLOP per timed batch, so the timestamp resolution (tens of microseconds on some
      // browsers) stays small against it
      const reps = Math.max(1, Math.min(512, Math.round(2e10 / Math.max(1, 2 * T * N * K))));
      runs.push({ variant, op, splits, reps, samples: [] });
    }
    // variants take turns in every round, starting from a different one each time, so GPU clock
    // ramps and thermal drift fall on all of them alike
    const turn = (round) => runs.map((_, i) => runs[(i + round) % runs.length]);
    for (let i = 0; i < warmups; i++) {
      for (const run of turn(i)) await timedBatch(device, timer, run.op, T, run.reps, run.variant.target, run.variant.rows, run.variant.groups, uncaptured, `${caseLabel(shape)} ${run.variant.label} warmup`);
    }
    for (let i = 0; i < samples; i++) {
      for (const run of turn(i)) {
        run.samples.push(await timedBatch(device, timer, run.op, T, run.reps, run.variant.target, run.variant.rows, run.variant.groups, uncaptured, `${caseLabel(shape)} ${run.variant.label} sample`));
      }
    }
    for (const { variant, op, splits, reps, samples: samplesForKernel } of runs) {
      const sorted = [...samplesForKernel].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      timings.push({ kernel: variant.label, rows: variant.rows, groups: variant.groups, splitTarget: variant.target, splits, reps, samples: samplesForKernel, medianMs: median });
      await checked(device, async () => {
        device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
        device.queue.writeBuffer(Y, 0, zero);
        await device.queue.onSubmittedWorkDone();
      }, `${caseLabel(shape)} ${variant.label} reset`, uncaptured);
      await timedBatch(device, timer, op, T, 1, variant.target, variant.rows, variant.groups, uncaptured, `${caseLabel(shape)} ${variant.label} correctness`);
      outputs[variant.label] = await readOutput(device, Y, bytes, uncaptured, `${caseLabel(shape)} ${variant.label}`);
      const finite = finiteSummary(outputs[variant.label]);
      if (!finite.ok) throw new Error(`${caseLabel(shape)} ${variant.label}: ${finite.message}`);
      const cpu = compareCpu(input, outputs[variant.label], variant, 0, false);
      correctness.cpu.push({ shape: caseLabel(shape), kernel: variant.label, mode: 0, bias: false, ...cpu });
      if (!cpu.ok) correctness.ok = false;
      if (shouldCheckEpilogues) {
        for (const mode of [1, 2]) {
          const withBias = true;
          await checked(device, async () => {
            device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, 1]));
            device.queue.writeBuffer(Y, 0, input.residual);
            await device.queue.onSubmittedWorkDone();
          }, `${caseLabel(shape)} mode ${mode} upload`, uncaptured);
          await timedBatch(device, timer, op, T, 1, variant.target, variant.rows, variant.groups, uncaptured, `${caseLabel(shape)} mode ${mode}`);
          const epilogue = await readOutput(device, Y, bytes, uncaptured, `${caseLabel(shape)} mode ${mode}`);
          const cpu = compareCpu(input, epilogue, variant, mode, withBias);
          correctness.cpu.push({ shape: caseLabel(shape), kernel: variant.label, mode, bias: withBias, ...cpu });
          if (!cpu.ok) correctness.ok = false;
        }
        await checked(device, async () => {
          device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
          device.queue.writeBuffer(Y, 0, zero);
          await device.queue.onSubmittedWorkDone();
        }, `${caseLabel(shape)} epilogue reset`, uncaptured);
      }
    }
    const variantCheck = compareVariants(outputs, labels, variantTolerance);
    correctness.variants.push({ shape: caseLabel(shape), ...variantCheck });
    if (!variantCheck.ok) correctness.ok = false;
    const shapeMedians = timings.map((item) => item.medianMs);
    const shapeMedian = shapeMedians[0];
    medians.push(shapeMedian);
    cases.push({ shape: caseLabel(shape), T, N, K, timings, medianMs: shapeMedian });
    log(`${caseLabel(shape)}: ${timings.map((item) => `${item.kernel}=${item.medianMs.toFixed(3)}ms`).join(", ")} splits=${timings[0].splits}`);
    for (const buffer of [X, W, S, B, Y, part, globals, params]) buffer.destroy();
  }
  const metricMs = Math.exp(medians.reduce((sum, value) => sum + Math.log(value), 0) / medians.length);
  const result = {
    status: "done",
    backend: "webgpu",
    metricMs,
    metric: "geometric mean of per-shape median GPU milliseconds",
    method: timer.method,
    seed,
    warmups,
    samples,
    kernels: labels,
    cases,
    adapter: provenance,
    correctness,
  };
  if (!correctness.ok) {
    result.status = "error";
    result.error = "GPU benchmark correctness checks failed";
    window.gpuBench = result;
    return;
  }
  window.gpuBench = result;
  log(`metricMs=${metricMs.toFixed(4)} method=${timer.method}`);
  timer.querySet?.destroy();
  timer.resolved?.destroy();
  timer.read?.destroy();
  device.destroy();
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().finally(() => activeDevice?.destroy()).catch((error) => {
  const message = `${error?.message || error}\n${error?.stack || ""}`;
  const previous = window.gpuBench && typeof window.gpuBench === "object" ? window.gpuBench : {};
  window.gpuBench = { ...previous, status: "error", backend: "webgpu", error: message };
  log(`ERROR ${message}`);
});
