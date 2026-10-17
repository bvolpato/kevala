// Bounded exact-output probe; does not run a model or change the production vision graph.
import { mmSplits, pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const search = new URLSearchParams(location.search);
const state = { status: "running", backend: "webgpu", kernel: "gemma4_vision_reduce_gelu_probe" };
window.gpuBench = state;
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const U = GPUBufferUsage;
const BUDGET = 64 * 1024 * 1024;
const SENTINELS = new Float32Array([12345.25, -1731.5, 913.75, -0]);
const CLIPPED = { gate: [-0.125, 0.375], up: [-0.5, 0.0625], down: [-0.015625, 0.025], clampDown: true };
const OPEN = { gate: [-Infinity, Infinity], up: [-Infinity, Infinity], down: [-Infinity, Infinity], clampDown: false };

function cases() {
  const out = [];
  for (const dtype of ["bf16", "f32"]) out.push({ T: 108, N: 3072, K: 768, splits: 3, dtype, pattern: "signed", bounds: CLIPPED, timing: dtype === "bf16" });
  for (const dtype of ["bf16", "f32"]) for (const splits of [1, 2, 3, 5, 8]) out.push({ T: 17, N: 3072, K: Math.max(768, splits * 128), splits, dtype, pattern: "signed", bounds: CLIPPED });
  for (const N of [1, 17, 255, 257, 3071, 3073]) for (const splits of [1, 3]) out.push({ T: 4, N, K: 768, splits, dtype: splits === 1 ? "f32" : "bf16", pattern: "signed", bounds: CLIPPED });
  for (const dtype of ["bf16", "f32"]) for (const pattern of ["zero", "cancellation", "endpoints"]) out.push({ T: 4, N: 257, K: 1024, splits: pattern === "endpoints" ? 1 : 8, dtype, pattern, bounds: OPEN });
  for (const bounds of [OPEN,
    { ...OPEN, gate: CLIPPED.gate }, { ...OPEN, up: CLIPPED.up },
    { ...OPEN, gate: CLIPPED.gate, up: CLIPPED.up },
    { ...CLIPPED, gate: [0, 0] }, { ...CLIPPED, gate: [-0, -0], up: [-0.125, 0.5] },
    { ...CLIPPED, down: [0.125, 0.5] }, { ...CLIPPED, down: [-0.5, -0.125] },
  ]) out.push({ T: 4, N: 257, K: 768, splits: 3, dtype: "bf16", pattern: "signed", bounds });
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function sizes(c) {
  const count = c.T * c.N, outputBytes = (count + SENTINELS.length) * 4, partBytes = (count * c.splits + SENTINELS.length) * 4;
  // GPU buffers, CPU fixtures and both readback copies, plus queue upload reserves.
  const estimatedBytes = 6 * partBytes + 25 * outputBytes + 4096;
  if (count % 4 || estimatedBytes > BUDGET) throw new Error(`invalid or oversized probe case: ${estimatedBytes} bytes`);
  return { count, outputBytes, partBytes, readBytes: outputBytes * 4, estimatedBytes };
}

function fixture(c, size) {
  const gate = new Float32Array(size.partBytes / 4).fill(NaN), up = new Float32Array(size.partBytes / 4).fill(NaN);
  const cancellation = [1048576, 0.015625, -1048576, 0.00737, -0.25, 0.1731, -0.0317, 0.413];
  const endpoints = [-12, -10, -9.999, -0, 0, 9.999, 10, 12];
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.count; i++) {
    const at = split * size.count + i;
    if (c.pattern === "zero") { gate[at] = (i + split) % 2 ? -0 : 0; up[at] = (i + split) % 3 ? 0 : -0; }
    else if (c.pattern === "endpoints") { gate[at] = endpoints[i % endpoints.length]; up[at] = endpoints[(i * 3 + 1) % endpoints.length] * 0.0625; }
    else if (c.pattern === "cancellation") { gate[at] = ((i % 13) - 6) / 8 * cancellation[split]; up[at] = ((i % 17) - 8) / 16 * cancellation[split]; }
    else {
      gate[at] = (1.337 + split * 0.193) * (split % 2 ? -1 : 1) * (Math.sin(i * 0.017 + split * 0.137) * 0.731 + Math.cos(i * 0.031 - split * 0.193) * 0.417);
      up[at] = (0.713 + split * 0.117) * (split % 3 ? 1 : -1) * (Math.cos(i * 0.023 - split * 0.173) * 0.317 + Math.sin(i * 0.041 + split * 0.193) * 0.531);
    }
  }
  const reducedGate = new Float32Array(size.count), reducedUp = new Float32Array(size.count);
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.count; i++) {
    reducedGate[i] = Math.fround(reducedGate[i] + gate[split * size.count + i]);
    reducedUp[i] = Math.fround(reducedUp[i] + up[split * size.count + i]);
  }
  return { gate, up, reducedGate, reducedUp };
}

function compare(label, expected, actual) {
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length), b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let differences = 0, maxAbs = 0, firstDifference = null;
  if (expected.length !== actual.length) throw new Error(`${label} length mismatch`);
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`${label} nonfinite output at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) { differences++; firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] }; }
  }
  const check = { label, bitExact: differences === 0, differences, maxAbs, firstDifference, checkedOutputs: expected.length };
  if (differences) throw new Error(`${label}: ${JSON.stringify(check)}`);
  return check;
}

async function checkCase(gpu, kernels, reducerFor, c) {
  const size = sizes(c), input = fixture(c, size), device = gpu.device;
  const target = c.timing ? 256 : Math.ceil(c.N / 64) * Math.ceil(c.T / 64) * c.splits;
  if (mmSplits(c.T, c.N, c.K, target, 64, 64) !== c.splits) throw new Error("baseline split count disagrees with the explicit candidate count");
  const reduce = await reducerFor(c.dtype, target), owned = [];
  let profiler, mapped = false, allocatedBytes = 0;
  const make = (bytes, usage, value, label) => {
    const count = Math.max(16, bytes); allocatedBytes += count;
    if (allocatedBytes > BUDGET) throw new Error("GPU probe buffer budget exceeded");
    const buffer = device.createBuffer({ size: count, usage, label }); owned.push(buffer);
    if (value) device.queue.writeBuffer(buffer, 0, value);
    return buffer;
  };
  const params = (values) => {
    const data = new ArrayBuffer(Math.max(16, values.length * 4)), ints = new Uint32Array(data), floats = new Float32Array(data);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(data.byteLength, U.UNIFORM | U.COPY_DST, data, "vision reducer/GELU parameters");
  };
  const bind = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const float = (f) => ({ f });
  const b = c.bounds;
  const geluValues = [size.count, float(b.down[0]), float(b.down[1]), Number(b.clampDown), float(b.gate[0]), float(b.gate[1]), float(b.up[0]), float(b.up[1])];
  let read;
  try {
    const globals = params([c.T, 0, 0, 0]), mm = params([c.N, c.K, 0, 0]);
    const geluP = params(geluValues), fusedP = params([...geluValues, c.splits, 0, 0, 0]);
    const gatePart = make(size.partBytes, U.STORAGE | U.COPY_DST, input.gate, "gate partials with poisoned tail");
    const upPart = make(size.partBytes, U.STORAGE | U.COPY_DST, input.up, "up partials with poisoned tail");
    const bias = make(16, U.STORAGE | U.COPY_DST, new Float32Array(4).fill(NaN), "disabled poisoned bias");
    const poison = new Float32Array(size.outputBytes / 4).fill(NaN); poison.set(SENTINELS, size.count);
    const outputs = Array.from({ length: 4 }, (_, i) => make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, `vision reducer/GELU output${i}`));
    const [gate, up, baseline, candidate] = outputs;
    read = make(size.readBytes, U.MAP_READ | U.COPY_DST, null, "vision reducer/GELU readback");
    const gateBind = bind(reduce, [globals, mm, gatePart, bias, gate]), upBind = bind(reduce, [globals, mm, upPart, bias, up]);
    const geluBind = bind(kernels.gelu, [geluP, gate, up, baseline]), fusedBind = bind(kernels.fused, [fusedP, gatePart, upPart, candidate]);
    const dispatch = (pass, next) => {
      if (next) { pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(Math.ceil(size.count / 1024)); }
      else {
        pass.setPipeline(reduce); pass.setBindGroup(0, gateBind); pass.dispatchWorkgroups(Math.ceil(size.count / 256));
        pass.setPipeline(reduce); pass.setBindGroup(0, upBind); pass.dispatchWorkgroups(Math.ceil(size.count / 256));
        pass.setPipeline(kernels.gelu); pass.setBindGroup(0, geluBind); pass.dispatchWorkgroups(Math.ceil(size.count / 1024));
      }
    };
    const copy = (encoder) => outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, read, i * size.outputBytes, size.outputBytes));
    const check = async () => {
      await read.mapAsync(GPUMapMode.READ); mapped = true;
      const actual = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); mapped = false;
      const stride = size.outputBytes / 4;
      for (let i = 0; i < outputs.length; i++) compare(`output${i} tail`, SENTINELS, actual.subarray(i * stride + size.count, (i + 1) * stride));
      return {
        gateReduction: compare("original gate reduction vs serial scalar F32", input.reducedGate, actual.subarray(0, size.count)),
        upReduction: compare("original up reduction vs serial scalar F32", input.reducedUp, actual.subarray(stride, stride + size.count)),
        parity: compare("original reducers+vision GELU vs candidate", actual.subarray(stride * 2, stride * 2 + size.count), actual.subarray(stride * 3, stride * 3 + size.count)),
        outputTailGuard: true,
      };
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end();
    copy(encoder); device.queue.submit([encoder.finish()]);
    const correctness = await check();
    const timings = { baseline: [], candidate: [] };
    let repeatedCorrectness = null;
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 2);
      const repetitions = 4;
      for (let sample = 0; sample < 6; sample++) {
        const encoder = device.createCommandEncoder();
        for (const next of sample % 2 ? [true, false] : [false, true]) {
          const pass = profiler.pass(encoder, next ? "candidate" : "baseline");
          for (let repetition = 0; repetition < repetitions; repetition++) dispatch(pass, next);
          pass.end();
        }
        profiler.finish(encoder); device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const name of ["baseline", "candidate"]) timings[name].push(measured[name] / repetitions);
      }
      const encoder = device.createCommandEncoder(); copy(encoder); device.queue.submit([encoder.finish()]); repeatedCorrectness = await check();
    }
    const endpoint = (value) => Object.is(value, -0) ? "-0" : Number.isFinite(value) ? value : String(value);
    const bounds = { ...c.bounds, gate: c.bounds.gate.map(endpoint), up: c.bounds.up.map(endpoint), down: c.bounds.down.map(endpoint) };
    return { ...c, bounds, splitTarget: target, correctness, repeatedCorrectness, timings, ownedGpuBufferBytes: allocatedBytes, estimatedCaseBufferBytes: size.estimatedBytes, ok: true };
  } finally { if (mapped) read?.unmap(); profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), f16Feature = search.get("f16") === "1";
  const gpu = await requestDevice({ features: ["timestamp-query", ...(f16Feature ? ["shader-f16"] : [])] });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    if (f16Feature && !gpu.device.features.has("shader-f16")) throw new Error("requested shader-f16 feature is unavailable");
    const source = await kernelSource();
    const code = source("gemma4_vision_reduce_gelu");
    const kernels = await withGpuErrors(gpu, async () => ({ gelu: await pipeline(gpu.device, source("gemma4_vision_gelu"), "vision_gelu_baseline"), fused: await pipeline(gpu.device, code, "vision_reduce_gelu_probe") }), "compiling vision reducer/GELU probe");
    const reducers = new Map();
    const reducerFor = (dtype, splitTarget) => {
      const key = `${dtype}:${splitTarget}`;
      if (!reducers.has(key)) reducers.set(key, pipeline(gpu.device, source("reduce", { weights: dtype, f16: false, rows: 4, groups: 1, splitTarget }), `vision_reduce_${key}`));
      return reducers.get(key);
    };
    const results = [];
    for (const c of cases()) { const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, reducerFor, c), "checking vision reducer/GELU"); results.push(result); log(result); }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    const info = gpu.adapter.info || {};
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started, f16Feature,
      arithmetic: "F32 split PART tensors; no matmul, F16 operand conversion, or default quantization",
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features] },
      compile: { ok: true, kernel: "gemma4_vision_reduce_gelu", wasmUrl: source.wasmUrl, workgroupSize: 256, sharedBytes: 8192 },
      resourceBounds: { maxCaseBufferEstimateBytes: Math.max(...results.map((c) => c.estimatedCaseBufferBytes)), budgetBytes: BUDGET, sequentialCases: true },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code }; log(error.stack || String(error)); });
