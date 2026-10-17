// Exact-output guard for split-K reduction + head64 RMSNorm + optional RoPE.
import { mmSplits, pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const state = { status: "running", backend: "webgpu", kernel: "gemma4_vision_reduce_norm_rope" };
window.gpuBench = state;
const search = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const U = GPUBufferUsage;
const SENTINEL = 12345.25;
const BUDGET_BYTES = 64 * 1024 * 1024;

function cases() {
  const out = [];
  const modes = [
    { weighted: true, skipRope: false, clipped: true },
    { weighted: false, skipRope: true, clipped: true },
    { weighted: true, skipRope: false, clipped: false },
    { weighted: false, skipRope: false, clipped: false },
  ];
  for (const splits of [1, 2, 3, 5, 8]) for (const heads of [1, 12]) for (const [index, rows] of [1, 17, 315, 567].entries()) {
    out.push({ rows, heads, splits, ...modes[(index + splits + heads) % modes.length], pattern: "signed" });
  }
  for (const splits of [1, 2, 3, 5, 8]) for (const skipRope of [false, true]) out.push({ rows: 17, heads: 12, splits, weighted: !skipRope, skipRope, clipped: false, pattern: "zero" });
  for (const clipped of [false, true]) out.push({ rows: 17, heads: 1, splits: 3, weighted: true, skipRope: true, clipped, pattern: "signed" });
  for (const splits of [5, 8]) out.push({ rows: 17, heads: 1, splits, weighted: true, skipRope: false, clipped: false, pattern: "cancellation" });
  for (const [rows, splits] of [[315, 5], [567, 3]]) for (const skipRope of [false, true]) out.push({ rows, heads: 12, splits, weighted: !skipRope, skipRope, clipped: true, pattern: "signed", timing: true });
  for (const invalid of [{ dim: 32 }, { splits: 0 }, { splits: 9 }, { mode: 1 }, { bias: 1 }]) out.push({ rows: 1, heads: 1, splits: 3, weighted: true, skipRope: false, clipped: true, pattern: "signed", invalid });
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function sizes(c) {
  const matrixFloats = c.rows * c.heads * 64;
  const matrixBytes = matrixFloats * 4;
  const partBytes = (matrixFloats * c.splits + 4) * 4;
  const outputBytes = matrixBytes + 16;
  const ropeBytes = c.rows * 64 * 4;
  // GPU buffers, live CPU arrays/readback, and one full PART upload reserve stay below64 MiB.
  const estimateBytes = 3 * partBytes + 2 * ropeBytes + matrixBytes + 10 * outputBytes + 2048;
  if (estimateBytes > BUDGET_BYTES) throw new Error(`case buffer estimate ${estimateBytes} exceeds ${BUDGET_BYTES}`);
  return { matrixFloats, matrixBytes, partBytes, outputBytes, estimateBytes };
}

function fixture(c, size) {
  const part = new Float32Array(size.partBytes / 4).fill(NaN);
  const cancellation = [1048576, 0.015625, -1048576, 0.00737, -0.25, 0.1731, -0.0317, 0.413];
  const partialScales = Array.from({ length: c.splits }, (_, split) => c.pattern === "cancellation" ? cancellation[split] : (split % 2 ? -1 : 1) * (1.337 + split * 0.193));
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.matrixFloats; i++) {
    const value = c.pattern === "zero" ? i % 2 ? -0 : 0 : c.pattern === "cancellation" ? ((i % 13) - 6) / 8 : Math.sin(i * 0.017 + split * 0.137) * 0.731 + Math.cos(i * 0.031 - split * 0.193) * 0.417;
    part[split * size.matrixFloats + i] = value * partialScales[split];
  }
  const expectedRaw = new Float32Array(size.matrixFloats);
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.matrixFloats; i++) expectedRaw[i] = Math.fround(expectedRaw[i] + part[split * size.matrixFloats + i]);
  const weights = Float32Array.from({ length: 64 }, (_, i) => !c.weighted ? NaN : i % 17 === 0 ? 0 : 0.731 + 1.317 * Math.sin(i * 0.217 - 0.3));
  const rope = new Float32Array(c.rows * 64);
  for (let row = 0; row < c.rows; row++) for (let axis = 0; axis < 2; axis++) for (let pair = 0; pair < 16; pair++) {
    const padded = c.rows > 3 && row >= c.rows - 3;
    const position = padded ? -1 : axis ? Math.floor(row / 7) : row % 7;
    const inverse = Math.fround(1 / Math.fround(Math.pow(100, pair / 16)));
    const angle = Math.fround(position * inverse);
    const at = (row * 32 + axis * 16 + pair) * 2;
    rope[at] = Math.cos(angle); rope[at + 1] = Math.sin(angle);
  }
  if (c.skipRope) rope.fill(NaN);
  return { part, expectedRaw, weights, rope, partialScales, eps: 1e-6, lo: c.clipped ? -0.75 : -Infinity, hi: c.clipped ? 1.25 : Infinity };
}

function compare(label, expected, actual) {
  const oldBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  const newBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let bitDifferences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`${label} nonfinite value at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (oldBits[i] !== newBits[i]) {
      bitDifferences++;
      firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: oldBits[i], actualBits: newBits[i] };
    }
  }
  return { label, ok: bitDifferences === 0, bitExact: bitDifferences === 0, bitDifferences, maxAbs, firstDifference, checkedOutputs: expected.length };
}

async function checkCase(gpu, kernels, getReduce, c) {
  const { device } = gpu;
  const size = sizes(c), input = fixture(c, size);
  const N = c.heads * 64, K = c.timing ? 768 : c.splits * 128;
  const splitTarget = c.timing ? 256 : Math.ceil(N / 64) * Math.ceil(c.rows / 64) * c.splits;
  const actualSplits = mmSplits(c.rows, N, K, splitTarget, 64, 64);
  if (actualSplits !== c.splits) throw new Error(`baseline split policy selected ${actualSplits}, expected ${c.splits}`);
  const reduce = await getReduce(splitTarget);
  const owned = [];
  let allocatedBytes = 0, profiler;
  const make = (bytes, usage, data, label) => {
    const actual = Math.max(16, bytes);
    if (allocatedBytes + actual > BUDGET_BYTES) throw new Error("case GPU buffers exceed64 MiB");
    const buffer = device.createBuffer({ label, size: actual, usage });
    allocatedBytes += actual;
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (values) => {
    const data = new ArrayBuffer(Math.max(16, values.length * 4));
    const ints = new Uint32Array(data), floats = new Float32Array(data);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(data.byteLength, U.UNIFORM | U.COPY_DST, data);
  };
  const storage = (data, label) => make(data.byteLength, U.STORAGE | U.COPY_DST, data, label);
  const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const f32 = (f) => ({ f });
  try {
    const globals = uniform([c.rows, 0, 0, 0]);
    const pReduce = uniform([N, K, 0, 0]);
    const normParams = [c.rows, c.heads, 64, Number(c.weighted), f32(input.eps), f32(input.lo), f32(input.hi), Number(c.skipRope)];
    const pNorm = uniform(normParams);
    const pFused = uniform([c.rows, c.heads, c.invalid?.dim ?? 64, ...normParams.slice(3), c.invalid?.splits ?? c.splits, c.invalid?.mode ?? 0, c.invalid?.bias ?? 0, 0]);
    const PART = storage(input.part, "split partials with poisoned tail"), W = storage(input.weights, "norm weights"), C = storage(input.rope, "axial rotary positions");
    const bias = storage(new Float32Array(4).fill(NaN), "disabled bias");
    const poison = new Float32Array(size.outputBytes / 4).fill(NaN);
    poison.fill(SENTINEL, size.matrixFloats);
    const outputs = ["old raw projection", "old normalized", "fused normalized"].map((label) => make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, label));
    const reduceBind = group(reduce, [globals, pReduce, PART, bias, outputs[0]]);
    const normBind = group(kernels.norm, [pNorm, outputs[0], W, C, outputs[1]]);
    const fusedBind = group(kernels.fused, [pFused, PART, W, C, outputs[2]]);
    const readback = make(size.outputBytes * outputs.length, U.COPY_DST | U.MAP_READ);
    const dispatch = (pass, name) => {
      if (name === "split") {
        pass.setPipeline(reduce); pass.setBindGroup(0, reduceBind); pass.dispatchWorkgroups(Math.ceil(size.matrixFloats / 256));
        pass.setPipeline(kernels.norm); pass.setBindGroup(0, normBind); pass.dispatchWorkgroups(c.rows * c.heads);
      } else {
        pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(c.rows * c.heads);
      }
    };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch(pass, "split"); dispatch(pass, "fused"); pass.end();
    outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * size.outputBytes, size.outputBytes));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const values = outputs.map((_, i) => {
      const base = i * poison.length;
      for (let j = size.matrixFloats; j < poison.length; j++) if (actual[base + j] !== SENTINEL) throw new Error(`output ${i} overwrote sentinel ${j - size.matrixFloats}`);
      return actual.subarray(base, base + size.matrixFloats);
    });
    const raw = compare("original reducer vs CPU serial sum", input.expectedRaw, values[0]);
    if (!raw.ok) throw new Error(`original reducer mismatch: ${JSON.stringify(raw)}`);
    let parity;
    if (c.invalid) {
      if (!values[2].every(Number.isNaN)) throw new Error(`unsupported candidate parameters wrote output: ${JSON.stringify(c.invalid)}`);
      parity = { ok: true, rejected: true, unchangedPoison: true, checkedOutputs: size.matrixFloats };
    } else {
      parity = compare("reduce + norm/RoPE vs fused", values[1], values[2]);
      if (!parity.ok) throw new Error(`reduce/norm/RoPE ${JSON.stringify(c)} differs: ${JSON.stringify(parity)}`);
    }
    const timings = { split: [], fused: [] };
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 2);
      const repeats = 16;
      for (let sample = 0; sample < 8; sample++) {
        const encoder = device.createCommandEncoder();
        for (const name of sample % 2 ? ["fused", "split"] : ["split", "fused"]) {
          const pass = profiler.pass(encoder, name);
          for (let n = 0; n < repeats; n++) dispatch(pass, name);
          pass.end();
        }
        profiler.finish(encoder);
        device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const name of ["split", "fused"]) timings[name].push(measured[name] / repeats);
      }
    }
    return { ...c, dim: 64, N, K, splitTarget, partialScales: input.partialScales, raw, parity, outputTailGuard: true, disabledInputsPoisoned: true,
      ownedGpuBufferBytes: allocatedBytes, estimatedCaseBufferBytes: size.estimateBytes, budgetBytes: BUDGET_BYTES, timings, ok: true };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now();
  const gpu = await requestDevice({ baseline: true });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const source = await kernelSource();
    const kernels = await withGpuErrors(gpu, async () => ({
      norm: await pipeline(gpu.device, source("gemma4_vision_norm_rope"), "gemma4_vision_norm_rope"),
      fused: await pipeline(gpu.device, source("gemma4_vision_reduce_norm_rope"), "gemma4_vision_reduce_norm_rope"),
    }), "compiling vision reducer/norm/RoPE guard");
    const reducers = new Map();
    const getReduce = (splitTarget) => {
      if (!reducers.has(splitTarget)) reducers.set(splitTarget, pipeline(gpu.device, source("reduce", { weights: "bf16", f16: false, rows: 4, groups: 1, splitTarget }), `reduce_target${splitTarget}`));
      return reducers.get(splitTarget);
    };
    const results = [];
    for (const c of cases()) {
      const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, getReduce, c), "checking vision reducer/norm/RoPE");
      results.push(result); log(JSON.stringify(result));
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started,
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features], workgroupStorageBytes: gpu.device.limits.maxComputeWorkgroupStorageSize },
      compile: { ok: true, source: "current kevala-base.wasm rendered production WGSL", workgroupSize: 64, sharedBytes: 1536 },
      resourceBounds: { maxCaseBufferEstimateBytes: Math.max(...results.map((c) => c.estimatedCaseBufferBytes)), budgetBytes: BUDGET_BYTES, sequentialCases: true },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code };
  log(error.stack || error);
});
