// Compare fused vision Q/K RMSNorm + RoPE against the existing two-dispatch F32 pipeline.
import { pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const state = { status: "running", backend: "webgpu", kernel: "gemma4_vision_norm_rope" };
window.gpuBench = state;
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const U = GPUBufferUsage;
const SENTINEL = 12345.25;

function fixture(c) {
  const { rows, heads, dim, pattern } = c;
  const paddedRows = rows > 3 ? 3 : 0;
  const positions = Int32Array.from({ length: rows * 2 }, (_, i) => {
    const row = Math.floor(i / 2);
    return row >= rows - paddedRows ? -1 : i % 2 ? Math.floor(row / 7) : row % 7;
  });
  const x = Float32Array.from({ length: rows * heads * dim }, (_, i) => {
    if (pattern === "zero") return i % 2 ? -0 : 0;
    const value = Math.sin(i * 0.173 + 0.127) * 1.337 + Math.cos(i * 0.031) * 0.719;
    return pattern === "clamped" ? value * 32 : value;
  });
  const weighted = c.weighted ?? pattern !== "unweighted";
  // A disabled weight must not be read. Enabled weights exercise sign, zero, and non-power values.
  const weights = Float32Array.from({ length: dim }, (_, i) => weighted ? i % 17 === 0 ? 0 : 0.731 + 1.317 * Math.sin(i * 0.217 - 0.3) : NaN);
  const rope = new Float32Array(rows * dim);
  const quarter = dim / 4;
  for (let row = 0; row < rows; row++) for (let axis = 0; axis < 2; axis++) for (let pair = 0; pair < quarter; pair++) {
    const inverse = Math.fround(1 / Math.fround(Math.pow(10000, pair / quarter)));
    const angle = Math.fround(positions[row * 2 + axis] * inverse);
    const at = (row * dim / 2 + axis * quarter + pair) * 2;
    rope[at] = Math.cos(angle);
    rope[at + 1] = Math.sin(angle);
  }
  if (c.skipRope) rope.fill(NaN);
  return { x, weights, rope, paddedRows, weighted, eps: 3.71e-5, lo: pattern === "clamped" ? -0.75 : -Infinity, hi: pattern === "clamped" ? 1.25 : Infinity };
}

function compare(oldValues, fusedValues) {
  const oldBits = new Uint32Array(oldValues.buffer, oldValues.byteOffset, oldValues.length);
  const fusedBits = new Uint32Array(fusedValues.buffer, fusedValues.byteOffset, fusedValues.length);
  let bitDifferences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < oldValues.length; i++) {
    if (!Number.isFinite(oldValues[i]) || !Number.isFinite(fusedValues[i])) throw new Error(`nonfinite output at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(oldValues[i] - fusedValues[i]));
    if (oldBits[i] !== fusedBits[i]) {
      bitDifferences++;
      firstDifference ||= { index: i, old: oldValues[i], fused: fusedValues[i], oldBits: oldBits[i], fusedBits: fusedBits[i] };
    }
  }
  return { ok: bitDifferences === 0, bitExact: bitDifferences === 0, bitDifferences, maxAbs, firstDifference, checkedOutputs: oldValues.length };
}

async function checkCase(gpu, kernels, c) {
  const { device } = gpu;
  const { x, weights, rope, paddedRows, weighted, eps, lo, hi } = fixture(c);
  const owned = [];
  let profiler;
  const make = (size, usage, data, label) => {
    const buffer = device.createBuffer({ label, size: Math.max(16, size), usage });
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (values) => {
    const raw = new ArrayBuffer(Math.max(16, values.length * 4));
    const ints = new Uint32Array(raw), floats = new Float32Array(raw);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(raw.byteLength, U.UNIFORM | U.COPY_DST, raw);
  };
  const storage = (data, label) => make(data.byteLength, U.STORAGE | U.COPY_DST, data, label);
  const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const normParams = uniform([c.rows * c.heads, c.dim, Number(weighted), 0, { f: eps }, { f: lo }, { f: hi }, 0]);
    const ropeParams = uniform([c.rows, c.heads, c.dim, 0]);
    const fusedParams = uniform([c.rows, c.heads, c.dim, Number(weighted), { f: eps }, { f: lo }, { f: hi }, Number(Boolean(c.skipRope))]);
    const X = storage(x, "input"), W = storage(weights, "norm weights"), C = storage(rope, "rotary positions");
    const unused = storage(new Float32Array(4), "unused residual");
    const outputFloats = x.length + 4, outputBytes = outputFloats * 4;
    const poison = new Float32Array(outputFloats).fill(NaN);
    poison.fill(SENTINEL, x.length);
    const labels = c.skipRope ? ["split", "fused"] : ["normalized", "split", "fused"];
    const splitIndex = c.skipRope ? 0 : 1, fusedIndex = splitIndex + 1;
    const outputs = labels.map((label) => make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, label));
    const readback = make(outputBytes * outputs.length, U.COPY_DST | U.MAP_READ);
    const normBind = group(kernels.norm, [normParams, X, W, unused, outputs[0]]);
    const ropeBind = c.skipRope ? null : group(kernels.rope, [ropeParams, C, outputs[0], outputs[splitIndex]]);
    const fusedBind = group(kernels.fused, [fusedParams, X, W, C, outputs[fusedIndex]]);
    const dispatch = (pass, name) => {
      if (name === "split") {
        pass.setPipeline(kernels.norm); pass.setBindGroup(0, normBind); pass.dispatchWorkgroups(c.rows * c.heads);
        if (!c.skipRope) { pass.setPipeline(kernels.rope); pass.setBindGroup(0, ropeBind); pass.dispatchWorkgroups(Math.ceil(x.length / 256)); }
      } else {
        pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(c.rows * c.heads);
      }
    };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch(pass, "split"); dispatch(pass, "fused"); pass.end();
    outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * outputBytes, outputBytes));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const values = outputs.map((_, i) => {
      const base = i * outputFloats;
      for (let j = x.length; j < outputFloats; j++) if (actual[base + j] !== SENTINEL) throw new Error(`output ${i} overwrote sentinel ${j - x.length}`);
      const value = actual.subarray(base, base + x.length);
      for (let j = 0; j < value.length; j++) if (!Number.isFinite(value[j])) throw new Error(`output ${i} contains nonfinite value at ${j}`);
      return value;
    });
    const parity = compare(values[splitIndex], values[fusedIndex]);
    if (!parity.ok) throw new Error(`norm/RoPE ${JSON.stringify(c)} differs: ${JSON.stringify(parity)}`);
    const timings = { split: [], fused: [] };
    if (c.rows >= 315 && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 2);
      const repeats = 16;
      for (let i = 0; i < 8; i++) {
        const encoder = device.createCommandEncoder();
        for (const name of i % 2 ? ["fused", "split"] : ["split", "fused"]) {
          const pass = profiler.pass(encoder, name);
          for (let j = 0; j < repeats; j++) dispatch(pass, name);
          pass.end();
        }
        profiler.finish(encoder);
        device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (i >= 2) for (const name of ["split", "fused"]) timings[name].push(measured[name] / repeats);
      }
    }
    return { ...c, paddedRows, weighted, disabledRopePoisoned: Boolean(c.skipRope), eps, bounds: [Number.isFinite(lo) ? lo : "-Infinity", Number.isFinite(hi) ? hi : "Infinity"], outputTailGuard: true, parity, timings, ok: true };
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
      norm: await pipeline(gpu.device, source("gemma4_vision_norm"), "gemma4_vision_norm"),
      rope: await pipeline(gpu.device, source("gemma4_vision_rope"), "gemma4_vision_rope"),
      fused: await pipeline(gpu.device, source("gemma4_vision_norm_rope"), "gemma4_vision_norm_rope"),
    }), "compiling vision norm/RoPE guard");
    const cases = [];
    for (const dim of [32, 64, 128, 256]) for (const pattern of ["signed", "clamped", "zero", "unweighted"]) for (const skipRope of [false, true]) {
      cases.push({ rows: 1, heads: 1, dim, pattern, skipRope }, { rows: 17, heads: 4, dim, pattern, skipRope });
    }
    for (const dim of [32, 64, 128, 256]) cases.push({ rows: 17, heads: 4, dim, pattern: "clamped", weighted: false, skipRope: true });
    for (const rows of [315, 567]) cases.push({ rows, heads: 12, dim: 64, pattern: "clamped" }, { rows, heads: 12, dim: 64, pattern: "clamped", weighted: false, skipRope: true });
    const results = [];
    for (const c of cases) {
      const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, c), `checking vision norm/RoPE ${JSON.stringify(c)}`);
      results.push(result); log(JSON.stringify(result));
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started,
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features], workgroupStorageBytes: gpu.device.limits.maxComputeWorkgroupStorageSize },
      compile: { ok: true, source: "current kevala-base.wasm rendered production WGSL", workgroupSize: 64, sharedBytes: 2048 },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code };
  log(error.stack || error);
});
