// Exact-output comparison against decoder attention, including logical/physical batched KV rows.
import { pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";
import { calibrateGemma4Attention } from "../js/src/gpu-gemma4-attention.js";

const state = { status: "running", backend: "webgpu", kernel: "gemma4_attention_subgroup" };
window.gpuBench = state;
const search = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const U = GPUBufferUsage;
const SENTINEL = 12345.25;

function plain(T, dim, window, causal, label = "plain") {
  return { label, T, S: T, heads: 4, kvHeads: 2, dim, window, causal, boundary: 0, batch: false,
    positions: Uint32Array.from({ length: T }, (_, i) => i), offsets: new Uint32Array(T) };
}

function branched(lengths, prefix, dim, window, { tail = false, selected = lengths.map((_, i) => i), label = "batch" } = {}) {
  const positions = [], offsets = [];
  let offset = 0;
  for (const [branch, length] of lengths.entries()) {
    if (!tail || selected.includes(branch)) {
      for (const row of tail ? [length - 1] : Array.from({ length }, (_, i) => i)) {
        positions.push(prefix + row);
        offsets.push(offset);
      }
    }
    offset += length;
  }
  return { label, T: positions.length, S: prefix + offset, heads: 4, kvHeads: 2, dim, window, causal: 1, boundary: prefix, batch: true,
    positions: Uint32Array.from(positions), offsets: Uint32Array.from(offsets) };
}

function cases() {
  const out = [];
  for (const dim of [256, 512]) {
    for (const T of [1, 17, 127, 128, 129, 257]) {
      out.push(plain(T, dim, 0, 1), plain(T, dim, 65, 1), plain(T, dim, 0, 0));
    }
    for (const pattern of ["peaked", "zero"]) out.push({ ...plain(129, dim, 0, 1), pattern });
    out.push(
      branched([3, 5, 9], 127, dim, 65),
      branched([64, 65], 89, dim, 0),
      branched([127, 130], 129, dim, 129),
      branched([42, 55], 89, dim, 0, { tail: true, selected: [1], label: "single-batch-tail" }),
    );
    const window = dim === 256 ? 512 : 0;
    out.push(
      { ...plain(89, dim, window, 1, "prefix"), heads: 8, timing: true },
      { ...branched([42, 55], 89, dim, window, { label: "suffix" }), heads: 8, timing: true },
      { ...branched([42, 55], 89, dim, window, { tail: true, label: "tail" }), heads: 8, timing: true },
    );
  }
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function fixture(c, seed) {
  let state = seed >>> 0;
  const random = () => {
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
  const q = Float32Array.from({ length: c.T * c.heads * c.dim }, (_, i) => c.pattern === "zero" ? 0 : c.pattern === "peaked" ? 4 + i % 4 / 8 : (random() - 0.5) * 0.5);
  const k = Float32Array.from({ length: c.S * c.kvHeads * c.dim }, (_, i) => c.pattern === "zero" ? 0 : c.pattern === "peaked" ? 4 + (Math.floor(i / (c.kvHeads * c.dim)) % 7 - 3) / 128 : (random() - 0.5) * 0.5);
  const v = Float32Array.from({ length: k.length }, () => (random() - 0.5) * 0.5);
  return { q, k, v };
}

function range(c, t) {
  const position = c.positions[t];
  const start = c.window > 0 ? position + 1 - Math.min(position + 1, c.window) : 0;
  return [start, c.causal ? position + 1 : c.S];
}

function cpuReference(c, input) {
  const tokens = [...new Set([0, Math.floor(c.T / 2), c.T - 1])];
  const heads = [...new Set([0, c.heads - 1])];
  const checks = [];
  for (const t of tokens) for (const head of heads) {
    const [start, end] = range(c, t);
    const qbase = (t * c.heads + head) * c.dim;
    const kvHead = Math.floor(head / (c.heads / c.kvHeads));
    const scores = new Float64Array(end - start);
    let top = -Infinity;
    for (let j = start; j < end; j++) {
      const physical = j + (j >= c.boundary && c.batch ? c.offsets[t] : 0);
      const kbase = (physical * c.kvHeads + kvHead) * c.dim;
      let score = 0;
      for (let d = 0; d < c.dim; d++) score += input.q[qbase + d] * input.k[kbase + d];
      scores[j - start] = score;
      top = Math.max(top, score);
    }
    let denominator = 0;
    for (let j = 0; j < scores.length; j++) denominator += scores[j] = Math.exp(scores[j] - top);
    const expected = new Float64Array(c.dim);
    for (let j = start; j < end; j++) {
      const physical = j + (j >= c.boundary && c.batch ? c.offsets[t] : 0);
      const vbase = (physical * c.kvHeads + kvHead) * c.dim;
      const weight = scores[j - start] / denominator;
      for (let d = 0; d < c.dim; d++) expected[d] += weight * input.v[vbase + d];
    }
    for (let d = 0; d < c.dim; d++) checks.push({ index: qbase + d, expected: expected[d] });
  }
  return checks;
}

function compare(oldValues, candidate) {
  const oldBits = new Uint32Array(oldValues.buffer, oldValues.byteOffset, oldValues.length);
  const newBits = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let bitDifferences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < oldValues.length; i++) {
    if (!Number.isFinite(oldValues[i]) || !Number.isFinite(candidate[i])) throw new Error(`nonfinite output at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(oldValues[i] - candidate[i]));
    if (oldBits[i] !== newBits[i]) {
      bitDifferences++;
      firstDifference ||= { index: i, old: oldValues[i], candidate: candidate[i], oldBits: oldBits[i], candidateBits: newBits[i] };
    }
  }
  return { ok: bitDifferences === 0, bitExact: bitDifferences === 0, bitDifferences, maxAbs, firstDifference, checkedOutputs: oldValues.length };
}

function compareCpu(values, checks) {
  const tolerance = { absolute: 1e-5, relative: 1e-5 };
  let maxAbs = 0;
  for (const { index, expected } of checks) {
    if (!Number.isFinite(expected)) throw new Error(`nonfinite CPU reference at ${index}`);
    const error = Math.abs(values[index] - expected);
    maxAbs = Math.max(maxAbs, error);
    if (error > tolerance.absolute + tolerance.relative * Math.max(1, Math.abs(expected))) throw new Error(`CPU mismatch at ${index}: ${values[index]} != ${expected}`);
  }
  return { ok: true, maxAbs, checkedOutputs: checks.length, tolerance };
}

async function checkCase(gpu, kernels, layout, c, seed) {
  const { device } = gpu;
  const input = fixture(c, seed);
  const checks = cpuReference(c, input);
  const owned = [];
  let profiler;
  const make = (size, usage, data, label) => {
    const buffer = device.createBuffer({ label, size: Math.max(16, size), usage });
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (data) => make(data.byteLength, U.UNIFORM | U.COPY_DST, data);
  const storage = (data) => make(data.byteLength, U.STORAGE | U.COPY_DST, data);
  try {
    const globals = uniform(new Uint32Array([c.T, 0, c.S, c.boundary]));
    const params = uniform(new Uint32Array([c.heads, c.kvHeads, c.dim, c.window, c.causal, Number(c.batch), 0, 0]));
    const Q = storage(input.q), K = storage(input.k), V = storage(input.v);
    const positions = new Uint32Array(c.T * 2);
    positions.set(c.positions); positions.set(c.offsets, c.T);
    const POS = storage(positions);
    const outputFloats = input.q.length + 4, outputBytes = outputFloats * 4;
    const poison = new Float32Array(outputFloats).fill(NaN);
    poison.fill(SENTINEL, input.q.length);
    const names = ["old", "subgroup"];
    const outputs = names.map((name) => make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, name));
    const groups = outputs.map((output) => device.createBindGroup({ layout, entries: [globals, params, Q, K, V, output, POS].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    const readback = make(outputBytes * outputs.length, U.COPY_DST | U.MAP_READ);
    const dispatch = (pass, i) => {
      pass.setPipeline(kernels[i]); pass.setBindGroup(0, groups[i]); pass.dispatchWorkgroups(c.T, c.heads);
    };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch(pass, 0); dispatch(pass, 1); pass.end();
    outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * outputBytes, outputBytes));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const values = outputs.map((_, i) => {
      const base = i * outputFloats;
      for (let j = input.q.length; j < outputFloats; j++) if (actual[base + j] !== SENTINEL) throw new Error(`${names[i]} output overwrote sentinel ${j - input.q.length}`);
      return actual.subarray(base, base + input.q.length);
    });
    const parity = compare(values[0], values[1]);
    if (!parity.ok) throw new Error(`decoder attention ${c.label} T${c.T} D${c.dim} differs: ${JSON.stringify(parity)}`);
    const cpu = compareCpu(values[1], checks);
    const timings = { old: [], subgroup: [] };
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 2);
      const repeats = 16;
      for (let sample = 0; sample < 8; sample++) {
        const encoder = device.createCommandEncoder();
        for (const i of sample % 2 ? [1, 0] : [0, 1]) {
          const pass = profiler.pass(encoder, names[i]);
          for (let n = 0; n < repeats; n++) dispatch(pass, i);
          pass.end();
        }
        profiler.finish(encoder);
        device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const name of names) timings[name].push(measured[name] / repeats);
      }
    }
    const { positions: ignoredPositions, offsets: ignoredOffsets, ...summary } = c;
    return { ...summary, positionRange: [Math.min(...c.positions), Math.max(...c.positions)], branchOffsets: [...new Set(c.offsets)], parity, cpu, outputTailGuard: true, timings, ok: true };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now();
  const gpu = await requestDevice({ features: ["subgroups", "timestamp-query"] });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const info = gpu.adapter.info || {};
    if (!gpu.device.features.has("subgroups") || !navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") || info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32) {
      throw new Error(`decoder subgroup attention requires subgroups, WGSL subgroup_id, and fixed32 lanes; got ${JSON.stringify({ features: [...gpu.device.features], subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, subgroupId: Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id")) })}`);
    }
    const source = await kernelSource();
    const layout = gpu.device.createBindGroupLayout({ entries: ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
    const pipelineLayout = gpu.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const kernels = await withGpuErrors(gpu, async () => [
      await pipeline(gpu.device, source("gemma4_attention"), "gemma4_attention", pipelineLayout),
      await pipeline(gpu.device, source("gemma4_attention_subgroup"), "gemma4_attention_subgroup", pipelineLayout),
    ], "compiling decoder subgroup attention guard");
    let calibration;
    if (search.get("tune") === "1") {
      const tuned = await calibrateGemma4Attention(gpu, source, { heads: 8, kvHeads: 1, localHeadDim: 256, globalHeadDim: 512, window: 512 }, kernels[0]);
      calibration = { ...tuned.diagnostics, selected: [...tuned.selection] };
      log(JSON.stringify({ calibration }));
      if (calibration.points.some((point) => point.numericalCheck?.ok === false)) throw new Error("attention calibration numerical guard failed");
    }
    const results = [];
    for (const [index, c] of cases().entries()) {
      const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, layout, c, (0x4b455641 + Math.imul(index, 0x9e3779b9)) >>> 0), `checking decoder subgroup attention ${c.label}`);
      results.push(result); log(JSON.stringify(result));
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started,
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features], subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, workgroupStorageBytes: gpu.device.limits.maxComputeWorkgroupStorageSize },
      compile: { ok: true, source: "current kevala-base.wasm rendered production WGSL", workgroupSize: 256, sharedBytes: 2576, requiredSubgroupSize: 32, subgroupId: true },
      ...(calibration ? { calibration } : {}),
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code };
  log(error.stack || error);
});
