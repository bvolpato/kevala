import { matmulLayout, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const SHAPES = [[1536, 12288], [1536, 6144], [2048, 1536], [1536, 2048], [256, 1536], [1536, 256]];
const SLOTS = [0, 1, 2].flatMap((mode) => [0, 1].map((bias) => ({ mode, bias })));
const SCRATCH_LIMIT = 64 * 1024 * 1024;
const SENTINEL = 12345.25;
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

async function pipelines(device, source, f16) {
  const layout = matmulLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const rendered = source("matvec", { weights: "q8", f16, rows: 1, groups: 1, splitTarget: 256 });
  const entry = rendered.indexOf("@compute");
  if (entry < 0) throw new Error("production matvec entry point is missing");
  // Keep the production bindings, sx, split rule, and F16 weight tile verbatim.
  const prelude = rendered.slice(0, entry)
    .replace(/var<workgroup> partials: array<f32, 256>;\s*/, "")
    .replace(/var<workgroup> inputs: array<vec4<f16>, 16>;\s*/, "");
  const url = new URL(f16 ? "./gemma4-matvec-two-f16.wgsl" : "./gemma4-matvec-two.wgsl", import.meta.url);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`candidate shader HTTP ${response.status}: ${url}`);
  const fragment = await response.text();
  return { layout, columns: f16 ? 16 : 4, f16,
    baseline: await pipeline(device, rendered, `matvec-two.baseline.${f16}`, pl),
    candidate: await pipeline(device, prelude + fragment, `matvec-two.candidate.${f16}`, pl) };
}

function dispatch(pass, pipes, group, N, T, candidate) {
  if (candidate && T !== 2) throw new RangeError("two-row matvec requires T=2");
  pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(N / pipes.columns), candidate ? 1 : T);
}

function inputs(N, K, pattern) {
  const weights = new Int8Array(N * K), scales = new Float32Array(N * K / 32);
  const x = new Float32Array(2 * K), bias = new Float32Array(N), old = new Float32Array(2 * N);
  for (let row = 0; row < 2; row++) for (let k = 0; k < K; k++) {
    x[row * K + k] = pattern === "zeros" ? ((row + k) % 2 ? -0 : 0)
      : Math.sin((k & ~1) * 0.037 + row * 0.73 + 0.17) * (row ? 0.09 : 0.07);
  }
  for (let n = 0; n < N; n++) {
    bias[n] = pattern === "zeros" ? (n % 2 ? -0 : 0) : (n % 9 - 4) * 0.0137;
    for (let row = 0; row < 2; row++) old[row * N + n] = pattern === "zeros" ? (row ? -0 : 0) : Math.sin(n * 0.3 + row) * 0.03;
    for (let block = 0; block < K / 32; block++) {
      scales[n * K / 32 + block] = (n + block) % 2 ? -0.00037 : 0.00029 + (block % 7) * 0.000031;
      for (let lane = 0; lane < 32; lane++) {
        const k = block * 32 + lane;
        weights[n * K + k] = pattern === "cancellation" || n === 0
          ? (lane % 2 ? -1 : 1) * (1 + (n * 13 + (lane & ~1) * 7 + block * 11) % 127)
          : (n * 37 + k * 53 + (k >>> 5) * 19) % 256 - 128;
      }
    }
  }
  if (N > 1) { weights[K] = -128; weights[K + 1] = 127; }
  return { weights, scales, x, bias, old };
}

function compare(actual, initial, N, slot, label) {
  const length = initial.length, bits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const tail = new Uint32Array(initial.buffer);
  let zeros = 0, negativeZeros = 0;
  for (let i = 0; i < 2 * N; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(actual[length + i])) throw new Error(`${label} mode=${slot.mode} bias=${slot.bias}: nonfinite output at ${i}`);
    if (bits[i] !== bits[length + i]) throw new Error(`${label} mode=${slot.mode} bias=${slot.bias}: bit mismatch at ${i}: 0x${bits[i].toString(16)} vs 0x${bits[length + i].toString(16)} (${actual[i]} vs ${actual[length + i]})`);
    if (actual[i] === 0) zeros++;
    if (bits[i] === 0x80000000) negativeZeros++;
  }
  for (let i = 2 * N; i < length; i++) if (bits[i] !== tail[i] || bits[length + i] !== tail[i]) throw new Error(`${label}: tail overwrite at ${i}`);
  return { mode: slot.mode, bias: slot.bias, bitMismatches: 0, zeros, negativeZeros };
}

async function timing(device, pipes, groups, N, make) {
  if (!device.features.has("timestamp-query")) return { reason: "no-timestamp-query" };
  const querySet = device.createQuerySet({ type: "timestamp", count: 12 });
  const resolved = make(256, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC, "timestamps.resolve");
  const read = make(256, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, "timestamps.read");
  const repetitions = 4, samples = { baseline: [], candidate: [] };
  try {
    const encoder = device.createCommandEncoder();
    const warmup = encoder.beginComputePass();
    dispatch(warmup, pipes, groups[0], N, 2, false);
    dispatch(warmup, pipes, groups[1], N, 2, true);
    warmup.end();
    let query = 0;
    const indices = [];
    for (let sample = 0; sample < 3; sample++) for (const candidate of sample % 2 ? [true, false] : [false, true]) {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: query, endOfPassWriteIndex: query + 1 } });
      for (let repeat = 0; repeat < repetitions; repeat++) dispatch(pass, pipes, groups[Number(candidate)], N, 2, candidate);
      pass.end(); indices.push({ query, label: candidate ? "candidate" : "baseline" }); query += 2;
    }
    encoder.resolveQuerySet(querySet, 0, 12, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, 96);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ, 0, 96);
    try {
      const stamps = new BigUint64Array(read.getMappedRange(0, 96));
      for (const { query, label } of indices) samples[label].push(Number(stamps[query + 1] - stamps[query]) / 1e6 / repetitions);
    } finally { read.unmap(); }
    const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
    return { method: "GPU timestamp", samples: 3, repetitions, ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs };
  } finally { querySet.destroy(); }
}

async function check(runtime, pipes, N, K, pattern, measure = false) {
  const { device } = runtime, U = GPUBufferUsage, owned = [];
  let scratchBytes = 0, weightBytes = 0;
  const make = (bytes, usage, label, data = null, weight = false) => {
    const size = Math.max(16, bytes);
    if (size > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error(`buffer limit: ${label}`);
    if (weight) weightBytes += size; else scratchBytes += size;
    if (scratchBytes > SCRATCH_LIMIT) throw new Error("two-row guard exceeds 64 MiB scratch budget");
    const buffer = device.createBuffer({ size, usage, label });
    owned.push(buffer); if (data) device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  try {
    const data = inputs(N, K, pattern);
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals", new Uint32Array([2, 0, 0, 0]));
    const x = make(data.x.byteLength, U.STORAGE | U.COPY_DST, "X", data.x);
    const weights = make(data.weights.byteLength, U.STORAGE | U.COPY_DST, "W", data.weights, true);
    const scales = make(data.scales.byteLength, U.STORAGE | U.COPY_DST, "S", data.scales, true);
    const bias = make(data.bias.byteLength, U.STORAGE | U.COPY_DST, "B", data.bias);
    const disabledBias = make(N * 4, U.STORAGE | U.COPY_DST, "disabled-B-NaN", new Float32Array(N).fill(NaN));
    const part = make(16, U.STORAGE, "unused-PART");
    const bytes = (2 * N + 4) * 4;
    const read = make(SLOTS.length * 2 * bytes, U.MAP_READ | U.COPY_DST, "output.read");
    const prepared = SLOTS.map((slot) => {
      const params = make(16, U.UNIFORM | U.COPY_DST, "params", new Uint32Array([N, K, slot.mode, slot.bias]));
      const initial = new Float32Array(2 * N + 4).fill(NaN);
      if (slot.mode === 1) initial.set(data.old);
      initial.fill(SENTINEL, 2 * N);
      const outputs = [0, 1].map(() => make(bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y", initial));
      const groups = outputs.map((output) => device.createBindGroup({ layout: pipes.layout, entries: [globals, params, x, weights, scales, slot.bias ? bias : disabledBias, output, part].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
      return { slot, initial, outputs, groups };
    });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (const item of prepared) for (const candidate of [false, true]) dispatch(pass, pipes, item.groups[Number(candidate)], N, 2, candidate);
    pass.end();
    prepared.forEach((item, index) => item.outputs.forEach((output, method) => encoder.copyBufferToBuffer(output, 0, read, (index * 2 + method) * bytes, bytes)));
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    let results;
    try {
      const actual = new Float32Array(read.getMappedRange());
      results = prepared.map((item, index) => compare(actual.subarray(index * 2 * (2 * N + 4), (index + 1) * 2 * (2 * N + 4)), item.initial, N, item.slot, `${pipes.f16 ? "f16" : "f32"} ${N}x${K} ${pattern}`));
    } finally { read.unmap(); }
    const rejectedRows = [];
    if (N === 17 && K === 32 && pattern === "zeros") for (const T of [1, 5]) {
      let rejected = false;
      try { dispatch(null, pipes, null, N, T, true); } catch (error) { if (!(error instanceof RangeError)) throw error; rejected = true; }
      if (!rejected) throw new Error(`candidate accepted unsupported T=${T}`);
      const item = prepared[0], unchanged = new Float32Array(2 * N + 4).fill(SENTINEL);
      device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
      device.queue.writeBuffer(item.outputs[1], 0, unchanged);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipes.candidate); pass.setBindGroup(0, item.groups[1]); pass.dispatchWorkgroups(Math.ceil(N / pipes.columns), 1); pass.end();
      encoder.copyBufferToBuffer(item.outputs[1], 0, read, 0, bytes); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ, 0, bytes);
      try { if (new Float32Array(read.getMappedRange(0, bytes)).some((value) => value !== SENTINEL)) throw new Error(`shader wrote for unsupported T=${T}`); }
      finally { read.unmap(); }
      rejectedRows.push(T);
    }
    device.queue.writeBuffer(globals, 0, new Uint32Array([2, 0, 0, 0]));
    const measured = measure ? await timing(device, pipes, prepared[0].groups, N, make) : null;
    return { N, K, T: 2, f16: pipes.f16, pattern, bitMismatches: 0, finite: true, signedZeroBitsChecked: true, tailGuard: true, rejectedRows, results, scratchBytes, weightBytes, timing: measured };
  } finally { for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  const runtime = await requestDevice({ baseline: params.has("baseline") });
  try {
    const source = await kernelSource(params.get("wasm") || undefined), cases = [];
    const variants = [false, ...(runtime.device.features.has("shader-f16") && params.get("f16") !== "0" ? [true] : [])];
    for (const f16 of variants) {
      const pipes = await withGpuErrors(runtime, () => pipelines(runtime.device, source, f16), `building two-row ${f16 ? "f16" : "f32"} matvec`);
      const shapes = [...SHAPES.map(([N, K]) => ({ N, K, pattern: "normal", measure: true })),
        ...[32, 1056].flatMap((K) => ["normal", "zeros", "cancellation"].map((pattern) => ({ N: 17, K, pattern })))];
      for (const shape of shapes) {
        const result = await withGpuErrors(runtime, () => check(runtime, pipes, shape.N, shape.K, shape.pattern, shape.measure), `checking two-row matvec ${f16 ? "f16" : "f32"} ${shape.N}x${shape.K}`);
        cases.push(result); window.gpuBench.completed = cases.length; log(result);
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matvec-two", metricMs: performance.now() - started,
      adapter: { name: runtime.name, features: [...runtime.device.features] }, wasm: source.wasmUrl, scratchLimitBytes: SCRATCH_LIMIT,
      correctness: { ok: true, tolerance: 0, fullBits: true, cases } };
    log({ done: true, elapsedMs: window.gpuBench.metricMs });
  } finally { runtime.device.destroy(); }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matvec-two", completed: 0 };
main().catch((error) => { window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) }; log(error.stack || error); });
