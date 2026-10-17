// Exact and bounded probe for ordered PART reduction into both accepted vision norm outputs.
import { mmSplits, pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const search = new URLSearchParams(location.search);
const state = { status: "running", backend: "webgpu", kernel: "gemma4_vision_reduce_residual_probe" };
window.gpuBench = state;
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const U = GPUBufferUsage;
const BUDGET = 96 * 1024 * 1024;
const SENTINELS = new Float32Array([12345.25, -1731.5, 913.75, -0]);
const DEFAULT = { rows: 17, width: 768, splits: 3, weighted: true, residual: true, weightedNext: true,
  clampAfter: false, clampAfterNext: true, bounds: [-0.75, 1.25], boundsNext: [-0.1875, 0.3125], pattern: "signed" };

function cases() {
  const out = [];
  for (const [projection, splits, K] of [["O", 6, 768], ["DOWN", 8, 3072]]) out.push({ ...DEFAULT, rows: 108, splits, K, projection, timing: true });
  for (const [rows, splits] of [[315, 5], [567, 3]]) for (const [projection, K] of [["O", 768], ["DOWN", 3072]]) out.push({ ...DEFAULT, rows, splits, K, projection, timing: true });
  for (const splits of [1, 2, 3, 5, 6, 8]) out.push({ ...DEFAULT, splits });
  for (let flags = 0; flags < 32; flags++) out.push({ ...DEFAULT, rows: flags % 2 ? 17 : 1, width: [32, 260, 768, 1024][flags % 4], splits: [1, 3, 8][flags % 3],
    weighted: Boolean(flags & 1), residual: Boolean(flags & 2), weightedNext: Boolean(flags & 4), clampAfter: Boolean(flags & 8), clampAfterNext: Boolean(flags & 16) });
  for (const width of [4, 28, 252, 260, 764, 772, 1020, 1024]) for (const rows of [1, 17]) out.push({ ...DEFAULT, rows, width, splits: rows === 1 ? 6 : 8 });
  for (const [bounds, boundsNext, clampAfter, clampAfterNext] of [
    [[-Infinity, Infinity], [-Infinity, Infinity], false, false],
    [[-Infinity, Infinity], [-Infinity, Infinity], true, true],
    [[-0.75, Infinity], [-Infinity, 0.3125], false, true],
    [[-Infinity, 1.25], [-0.1875, Infinity], true, false],
    [[-0, 0], [-0, 0], false, true],
    [[0.125, 0.5], [-0.5, -0.125], true, false],
  ]) out.push({ ...DEFAULT, bounds, boundsNext, clampAfter, clampAfterNext });
  for (const width of [32, 768]) for (const pattern of ["zero", "cancellation"]) for (const splits of [6, 8]) out.push({ ...DEFAULT, width, splits, pattern, bounds: [-Infinity, Infinity], boundsNext: [-Infinity, Infinity], clampAfterNext: false });
  for (const weightPattern of ["zero-post", "zero-next", "negative"]) out.push({ ...DEFAULT, weightPattern });
  for (const invalid of [{ width: 0 }, { width: 1028 }, { width: 766 }, { splits: 0 }, { splits: 9 }, { mode: 1 }, { mode: 2 }, { bias: 1 }, { rows: 0 }]) out.push({ ...DEFAULT, rows: 1, invalid });
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function sizes(c) {
  const count = c.rows * c.width, outputBytes = (count + SENTINELS.length) * 4, partBytes = (count * c.splits + SENTINELS.length) * 4;
  // Include GPU buffers, host fixtures and readbacks, and queue upload reserves.
  const estimateBytes = 4 * partBytes + 36 * outputBytes + 8 * (c.width + SENTINELS.length) * 4 + 8192;
  if (estimateBytes > BUDGET) throw new Error(`case estimate ${estimateBytes} exceeds ${BUDGET}`);
  return { count, outputBytes, partBytes, readBytes: outputBytes * 5, estimateBytes };
}

function fixture(c, size) {
  const part = new Float32Array(size.partBytes / 4).fill(NaN), raw = new Float32Array(size.count);
  const cancellation = [1048576, 0.015625, -1048576, 0.00737, -0.25, 0.1731, -0.0317, 0.413];
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.count; i++) {
    const scale = c.pattern === "cancellation" ? cancellation[split] : (split % 2 ? -1 : 1) * (1.337 + split * 0.193);
    const value = c.pattern === "zero" ? (i + split) % 2 ? -0 : 0 : c.pattern === "cancellation" ? ((i % 13) - 6) / 8 : Math.sin(i * 0.017 + split * 0.137) * 0.731 + Math.cos(i * 0.031 - split * 0.193) * 0.417;
    part[split * size.count + i] = value * scale;
  }
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.count; i++) raw[i] = Math.fround(raw[i] + part[split * size.count + i]);
  const residual = new Float32Array(size.count + SENTINELS.length).fill(NaN);
  if (c.residual) for (let i = 0; i < size.count; i++) residual[i] = c.pattern === "zero" ? i % 3 ? 0 : -0 : 0.473 * Math.sin(i * 0.137 - 0.417);
  const post = new Float32Array(c.width + SENTINELS.length).fill(NaN), next = post.slice();
  for (let i = 0; i < c.width; i++) {
    if (c.weighted) post[i] = c.weightPattern === "zero-post" ? 0 : c.weightPattern === "negative" ? -0.731 - 0.317 * Math.abs(Math.sin(i * 0.217)) : i % 17 === 0 ? 0 : 0.731 + 1.317 * Math.sin(i * 0.217 - 0.3);
    if (c.weightedNext) next[i] = c.weightPattern === "zero-next" ? 0 : c.weightPattern === "negative" ? -0.913 - 0.417 * Math.abs(Math.cos(i * 0.193)) : i % 19 === 0 ? 0 : 0.913 - 1.173 * Math.cos(i * 0.193 + 0.2);
  }
  const rejected = new Float32Array(size.count + SENTINELS.length);
  for (let i = 0; i < size.count; i++) rejected[i] = i % 13 === 0 ? -0 : 0.193 * Math.sin(i * 0.071 + 0.313);
  rejected.set(SENTINELS, size.count);
  return { part, raw, residual, post, next, rejected, eps: 3.71e-5, epsNext: 1.337e-6 };
}

function compare(label, expected, actual) {
  if (expected.length !== actual.length) throw new Error(`${label} length mismatch`);
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length), b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let differences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`${label} nonfinite value at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) { differences++; firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] }; }
  }
  const check = { label, bitExact: differences === 0, differences, maxAbs, firstDifference, checkedOutputs: expected.length };
  if (differences) throw new Error(`${label}: ${JSON.stringify(check)}`);
  return check;
}

async function checkCase(gpu, kernels, reduceFor, c) {
  const { device } = gpu, size = sizes(c), input = fixture(c, size);
  const K = c.K || Math.max(768, c.splits * 128), splitTarget = c.timing ? 256 : Math.ceil(c.width / 64) * Math.ceil(c.rows / 64) * c.splits;
  if (mmSplits(c.rows, c.width, K, splitTarget, 64, 64) !== c.splits) throw new Error("baseline split count disagrees with the explicit candidate count");
  const reduce = await reduceFor(splitTarget), owned = [];
  let allocatedBytes = 0, read, mapped = false, profiler;
  const make = (bytes, usage, value, label) => {
    const count = Math.max(16, bytes); allocatedBytes += count;
    if (allocatedBytes > BUDGET) throw new Error("GPU probe buffer budget exceeded");
    const buffer = device.createBuffer({ size: count, usage, label }); owned.push(buffer);
    if (value) device.queue.writeBuffer(buffer, 0, value);
    return buffer;
  };
  const uniform = (values) => {
    const data = new ArrayBuffer(Math.max(16, values.length * 4)), ints = new Uint32Array(data), floats = new Float32Array(data);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(data.byteLength, U.UNIFORM | U.COPY_DST, data, "vision reducer/residual parameters");
  };
  const storage = (value, label) => make(value.byteLength, U.STORAGE | U.COPY_DST, value, label);
  const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const float = (f) => ({ f });
  try {
    const globals = uniform([c.rows, 0, 0, 0]), mm = uniform([c.width, K, 0, 0]);
    const params = [c.rows, c.width, Number(c.weighted), Number(c.residual), float(input.eps), ...c.bounds.map(float), Number(c.clampAfter), Number(c.weightedNext), Number(c.clampAfterNext), c.splits, 0, float(input.epsNext), ...c.boundsNext.map(float), 0];
    const pNorm = uniform(params), candidateParams = params.slice();
    for (const [key, index] of [["rows", 0], ["width", 1], ["splits", 10], ["mode", 11], ["bias", 15]]) if (c.invalid?.[key] !== undefined) candidateParams[index] = c.invalid[key];
    const pFused = uniform(candidateParams);
    const PART = storage(input.part, "partial planes with poisoned tail"), W = storage(input.post, "post norm weights"), R = storage(input.residual, "residual or disabled poisoned input"), Wnext = storage(input.next, "next norm weights");
    const bias = storage(new Float32Array(4).fill(NaN), "disabled poisoned reducer bias");
    const poison = new Float32Array(size.outputBytes / 4).fill(NaN); poison.set(SENTINELS, size.count);
    const outputs = ["old projection", "old residual", "old next norm", "fused residual", "fused next norm"].map((label, i) => make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, i >= 3 && c.invalid ? input.rejected : poison, label));
    const [raw, oldY, oldH, newY, newH] = outputs;
    read = make(size.readBytes, U.COPY_DST | U.MAP_READ, null, "vision reducer/residual readback");
    const reduceBind = group(reduce, [globals, mm, PART, bias, raw]), normBind = group(kernels.norm, [pNorm, raw, W, R, Wnext, oldY, oldH]);
    const fusedBind = group(kernels.fused, [pFused, PART, W, R, Wnext, newY, newH]);
    const dispatch = (pass, next, extraRow = false) => {
      if (next) { pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(c.rows + Number(extraRow)); }
      else {
        pass.setPipeline(reduce); pass.setBindGroup(0, reduceBind); pass.dispatchWorkgroups(Math.ceil(size.count / 256));
        pass.setPipeline(kernels.norm); pass.setBindGroup(0, normBind); pass.dispatchWorkgroups(c.rows);
      }
    };
    const copy = (encoder) => outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, read, i * size.outputBytes, size.outputBytes));
    const check = async () => {
      await read.mapAsync(GPUMapMode.READ); mapped = true;
      const actual = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); mapped = false;
      const stride = size.outputBytes / 4, value = (index) => actual.subarray(index * stride, index * stride + size.count);
      for (let i = 0; i < outputs.length; i++) compare(`output${i} tail`, SENTINELS, actual.subarray(i * stride + size.count, (i + 1) * stride));
      return {
        reduction: compare("original reducer vs serial scalar F32", input.raw, value(0)),
        residual: compare(c.invalid ? "rejected candidate preserves Y" : "residual Y", c.invalid ? input.rejected.subarray(0, size.count) : value(1), value(3)),
        normalized: compare(c.invalid ? "rejected candidate preserves H" : "next norm H", c.invalid ? input.rejected.subarray(0, size.count) : value(2), value(4)),
        outputTailGuard: true,
      };
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true, true); pass.end();
    copy(encoder); device.queue.submit([encoder.finish()]);
    const correctness = await check(), timings = { baseline: [], candidate: [] };
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
    return { ...c, bounds: c.bounds.map(endpoint), boundsNext: c.boundsNext.map(endpoint), K, splitTarget, correctness, repeatedCorrectness, timings,
      ownedGpuBufferBytes: allocatedBytes, estimatedCaseBufferBytes: size.estimateBytes, disabledInputsPoisoned: true, extraRowDispatched: true, ok: true };
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
    const code = source("gemma4_vision_reduce_residual");
    const kernels = await withGpuErrors(gpu, async () => ({ norm: await pipeline(gpu.device, source("gemma4_vision_residual_norm"), "vision_residual_norm_baseline"), fused: await pipeline(gpu.device, code, "vision_reduce_residual_probe") }), "compiling vision reducer/residual probe");
    const reducers = new Map();
    const reduceFor = (splitTarget) => {
      if (!reducers.has(splitTarget)) reducers.set(splitTarget, pipeline(gpu.device, source("reduce", { weights: "bf16", f16: false, rows: 4, groups: 1, splitTarget }), `vision_reduce_target${splitTarget}`));
      return reducers.get(splitTarget);
    };
    const results = [];
    for (const c of cases()) { const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, reduceFor, c), "checking vision reducer/residual"); results.push(result); log(result); }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    const info = gpu.adapter.info || {};
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started, f16Feature,
      arithmetic: "F32 synthetic PART tensors, ordered scalar reduction, and both original vec4 RMSNorm trees",
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features] },
      compile: { ok: true, kernel: "gemma4_vision_reduce_residual", wasmUrl: source.wasmUrl, workgroupSize: 256, sharedBytes: 9216 },
      resourceBounds: { maxCaseBufferEstimateBytes: Math.max(...results.map((c) => c.estimatedCaseBufferBytes)), budgetBytes: BUDGET, sequentialCases: true },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code }; log(error.stack || String(error)); });
