// Direct numerical guard for the production Gemma 4 vision attention WGSL. The CPU reference
// checks stable softmax probabilities and weighted V output, including masked and large logits.
import { pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const state = { status: "running", backend: "webgpu", kernel: "gemma4_vision_attention" };
window.gpuBench = state;
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const DIMS = [32, 48, 64, 256];
const ROWS = [1, 17, 257];
const RTOL = 2e-5;
const ATOL = 2e-5;

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state ^ (state >>> 16), 0x45d9f3b) ^ 0x45d9f3b) >>> 0;
    state = (Math.imul(state ^ (state >>> 16), 0x45d9f3b) ^ 0x45d9f3b) >>> 0;
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
}

function cases() {
  const out = [];
  for (const dim of DIMS) for (const rows of ROWS) {
    out.push({ rows, heads: 1, kvHeads: 1, dim });
    out.push({ rows, heads: 4, kvHeads: 2, dim });
  }
  return out.map((c) => ({ ...c, label: `r${c.rows}-h${c.heads}-kv${c.kvHeads}-d${c.dim}` }));
}

function inputs(c, seed) {
  const random = rng(seed);
  const scores = new Float32Array(c.rows * c.heads * c.rows);
  const head4 = c.dim / 4;
  const values = new Float32Array(c.rows * c.kvHeads * head4 * 4);
  for (let key = 0; key < values.length; key++) values[key] = Math.fround(random() * 2 - 1);
  for (let head = 0; head < c.heads; head++) for (let row = 0; row < c.rows; row++) {
    const base = (head * c.rows + row) * c.rows;
    const peak = (row * 5 + head * 7) % c.rows;
    for (let key = 0; key < c.rows; key++) {
      const masked = c.rows > 1 && key !== peak && (key + row + head) % 13 === 0;
      let score;
      if (masked) score = -1e30;
      else if (key === peak) score = 1008;
      else if (key === (peak + 1) % c.rows) score = -1000;
      else score = Math.fround(1000 + random() * 4);
      scores[base + key] = score;
    }
  }
  return { scores, values };
}

function reference(c, scores, values) {
  const head4 = c.dim / 4;
  const outputs = new Float32Array(c.rows * c.heads * head4 * 4);
  const probabilities = new Float32Array(scores.length);
  let maxProbabilitySumError = 0;
  for (let head = 0; head < c.heads; head++) {
    const kvHead = Math.floor(head / (c.heads / c.kvHeads));
    for (let row = 0; row < c.rows; row++) {
      const scoreBase = (head * c.rows + row) * c.rows;
      let maximum = -Infinity;
      for (let key = 0; key < c.rows; key++) maximum = Math.max(maximum, scores[scoreBase + key]);
      const weights = new Float32Array(c.rows);
      let total = 0;
      for (let key = 0; key < c.rows; key++) {
        weights[key] = Math.fround(Math.exp(Math.fround(scores[scoreBase + key] - maximum)));
        total = Math.fround(total + weights[key]);
      }
      if (!Number.isFinite(total) || total <= 0) throw new Error(`${c.label} CPU softmax sum is invalid`);
      const inverse = Math.fround(1 / total);
      let probabilitySum = 0;
      for (let key = 0; key < c.rows; key++) {
        const probability = Math.fround(weights[key] * inverse);
        probabilities[scoreBase + key] = probability;
        probabilitySum += probability;
        const vBase = (key * c.kvHeads + kvHead) * head4 * 4;
        const outBase = (row * c.heads + head) * head4 * 4;
        for (let col = 0; col < head4 * 4; col++) {
          const product = Math.fround(values[vBase + col] * probability);
          outputs[outBase + col] = Math.fround(outputs[outBase + col] + product);
        }
      }
      maxProbabilitySumError = Math.max(maxProbabilitySumError, Math.abs(probabilitySum - 1));
    }
  }
  return { outputs, probabilities, maxProbabilitySumError };
}

function compareOutput(actual, expected) {
  if (actual.length !== expected.length) throw new Error(`output length ${actual.length} != ${expected.length}`);
  let maxAbs = 0, maxScaledError = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) throw new Error(`output contains a nonfinite value at ${i}`);
    const error = Math.abs(actual[i] - expected[i]);
    const scaled = error / (ATOL + RTOL * Math.abs(expected[i]));
    maxAbs = Math.max(maxAbs, error);
    maxScaledError = Math.max(maxScaledError, scaled);
  }
  return { maxAbs, maxScaledError, ok: maxScaledError <= 1 };
}

function checkProbabilities(c, actualWeights, expected) {
  let maxAbs = 0, maxSumError = 0, minimum = Infinity, maximum = -Infinity;
  for (let head = 0; head < c.heads; head++) for (let row = 0; row < c.rows; row++) {
    const base = (head * c.rows + row) * c.rows;
    let total = 0;
    for (let key = 0; key < c.rows; key++) {
      const weight = actualWeights[base + key];
      if (!Number.isFinite(weight) || weight < 0) throw new Error(`${c.label} produced invalid unnormalized softmax weight`);
      total += weight;
    }
    if (!Number.isFinite(total) || total <= 0) throw new Error(`${c.label} produced an empty softmax row`);
    let sum = 0;
    for (let key = 0; key < c.rows; key++) {
      const probability = actualWeights[base + key] / total;
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error(`${c.label} produced an invalid normalized probability`);
      const delta = Math.abs(probability - expected.probabilities[base + key]);
      maxAbs = Math.max(maxAbs, delta);
      minimum = Math.min(minimum, probability);
      maximum = Math.max(maximum, probability);
      sum += probability;
    }
    maxSumError = Math.max(maxSumError, Math.abs(sum - 1));
  }
  const ok = maxAbs <= 2e-6 && maxSumError <= 2e-6;
  return { ok, maxAbs, maxSumError, minimum, maximum, storageSemantics: "unnormalized weights; normalized here for validation" };
}

function allocate(device, size, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage });
}

async function runCase(runtime, kernel, c, seed) {
  const { device } = runtime;
  const U = GPUBufferUsage;
  const head4 = c.dim / 4;
  const { scores, values } = inputs(c, seed);
  const expected = reference(c, scores, values);
  const scoreBytes = scores.byteLength;
  const valueBytes = values.byteLength;
  const outputBytes = c.rows * c.heads * head4 * 16;
  const readBytes = scoreBytes + outputBytes;
  const buffers = [];
  const make = (size, usage, name) => {
    const buffer = allocate(device, size, usage, `${c.label} ${name}`);
    buffers.push(buffer);
    return buffer;
  };
  try {
    return await withGpuErrors(runtime, async () => {
      const params = make(16, U.UNIFORM | U.COPY_DST, "parameters");
      const S = make(scoreBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "scores");
      const V = make(valueBytes, U.STORAGE | U.COPY_DST, "values");
      const Y = make(outputBytes, U.STORAGE | U.COPY_SRC, "output");
      const readback = make(readBytes, U.MAP_READ | U.COPY_DST, "readback");
      device.queue.writeBuffer(params, 0, new Uint32Array([c.rows, c.heads, c.kvHeads, c.dim]));
      device.queue.writeBuffer(S, 0, scores);
      device.queue.writeBuffer(V, 0, values);
      const bind = device.createBindGroup({
        layout: kernel.getBindGroupLayout(0),
        entries: [params, S, V, Y].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const encoder = device.createCommandEncoder({ label: `${c.label} attention guard` });
      const pass = encoder.beginComputePass({ label: `${c.label} vision attention` });
      pass.setPipeline(kernel);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(c.rows, c.heads);
      pass.end();
      encoder.copyBufferToBuffer(S, 0, readback, 0, scoreBytes);
      encoder.copyBufferToBuffer(Y, 0, readback, scoreBytes, outputBytes);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ, 0, readBytes);
      const copy = new Float32Array(readback.getMappedRange(0, readBytes).slice(0));
      readback.unmap();
      const actualWeights = copy.subarray(0, scores.length);
      const actualOutputs = copy.subarray(scores.length, scores.length + outputBytes / 4);
      const probability = checkProbabilities(c, actualWeights, expected);
      const output = compareOutput(actualOutputs, expected.outputs);
      const ok = probability.ok && output.ok;
      return {
        ...c,
        head4,
        keyLanes: Math.floor(256 / head4),
        activeLanes: Math.floor(256 / head4) * head4,
        checkedOutputs: actualOutputs.length,
        probability,
        output,
        cpuProbabilitySumError: expected.maxProbabilitySumError,
        ok,
      };
    }, `running ${c.label} vision attention`);
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

async function main() {
  const started = performance.now();
  const runtime = await requestDevice({ baseline: true, features: [] });
  const uncaptured = [];
  runtime.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const source = await kernelSource();
    const code = source("gemma4_vision_attention");
    const layout = runtime.device.createBindGroupLayout({
      label: "gemma4 vision attention guard",
      entries: ["uniform", "storage", "read-only-storage", "storage"].map((type, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: type === "storage" && (binding === 1 || binding === 3) ? "storage" : type },
      })),
    });
    const pipelineLayout = runtime.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const kernel = await withGpuErrors(runtime, () => pipeline(runtime.device, code, "gemma4_vision_attention", pipelineLayout), "compiling Gemma 4 vision attention");
    const results = [];
    let ok = true;
    for (const [index, c] of cases().entries()) {
      const result = await runCase(runtime, kernel, c, 0x4b455641 + index * 0x9e3779b9);
      results.push(result);
      ok &&= result.ok;
      log(`${c.label}: ${result.ok ? "ok" : "FAILED"}; keyLanes=${result.keyLanes}, activeLanes=${result.activeLanes}, maxAbs=${result.output.maxAbs.toExponential(2)}, maxP=${result.probability.maxAbs.toExponential(2)}, sumErr=${result.probability.maxSumError.toExponential(2)}`);
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = {
      ...state,
      status: ok ? "done" : "error",
      done: true,
      adapter: {
        name: runtime.name,
        vendor: runtime.adapter.info?.vendor || null,
        architecture: runtime.adapter.info?.architecture || null,
        device: runtime.adapter.info?.device || null,
        isFallbackAdapter: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter),
        deviceFeatures: [...runtime.device.features],
        limits: { maxComputeWorkgroupSizeX: runtime.device.limits.maxComputeWorkgroupSizeX, maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize },
      },
      compile: { ok: true, source: "current kevala-base.wasm rendered production WGSL", workgroupSize: 256, metadataErrorScopes: true },
      fixture: { rows: ROWS, headDims: DIMS, headConfigs: [{ heads: 1, kvHeads: 1 }, { heads: 4, kvHeads: 2 }], scoreOffset: 1000, maskedLogit: -1e30 },
      metricMs: performance.now() - started,
      metric: "wall clock for direct numerical WGSL guard; not a performance benchmark",
      correctness: {
        ok,
        tolerance: { outputAbsolute: ATOL, outputRelative: RTOL, probabilityAbsolute: 2e-6, probabilitySum: 2e-6 },
        cases: results,
      },
      ...(ok ? {} : { error: "Gemma 4 vision attention CPU parity failed" }),
    };
  } finally {
    runtime.device.destroy();
  }
}

async function fusedCase(runtime, source, c) {
  const { device } = runtime, U = GPUBufferUsage;
  const owned = [];
  const make = (bytes, usage, data) => {
    const buffer = device.createBuffer({ size: Math.max(16, bytes), usage });
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  let profiler;
  try {
    // Binary-exact large Q/K isolate softmax stability from scalar CPU versus GPU FMA rounding.
    const queries = Float32Array.from({ length: c.rows * c.heads * 64 }, (_, i) => c.large ? 4 + i % 4 / 8 : 0.5 * Math.sin(i * 0.027 + 0.1));
    const keys = Float32Array.from({ length: c.rows * c.kvHeads * 64 }, (_, i) => c.large ? 4 + (Math.floor(i / (c.kvHeads * 64)) % 7 - 3) / 128 : 0.5 * Math.cos(i * 0.031 - 0.3));
    const values = Float32Array.from({ length: keys.length }, (_, i) => Math.sin(i * 0.013));
    const positions = Int32Array.from({ length: c.rows * 2 }, (_, i) => c.allMasked || c.mask && Math.floor(i / 2) % 5 === 0 ? -1 : Math.floor(i / 2));
    const params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([c.rows, c.heads, c.kvHeads, 64]));
    const inputs = [queries, keys, values, positions].map((data) => make(data.byteLength, U.STORAGE | U.COPY_DST, data));
    const scores = make(c.rows * c.rows * c.heads * 4, U.STORAGE);
    const outputs = [0, 1].map(() => make(queries.byteLength, U.STORAGE | U.COPY_SRC | U.COPY_DST, new Float32Array(queries.length).fill(NaN)));
    const readback = make(queries.byteLength * 2, U.COPY_DST | U.MAP_READ);
    const kernels = [];
    for (const [name, types, buffers, groups] of [
      ["gemma4_vision_scores_tiled", ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"], [params, inputs[0], inputs[1], inputs[3], scores], [Math.ceil(c.rows / 16) ** 2 * c.heads]],
      ["gemma4_vision_attention", ["uniform", "storage", "read-only-storage", "storage"], [params, scores, inputs[2], outputs[0]], [c.rows, c.heads]],
      ["gemma4_vision_attention_fused", ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"], [params, ...inputs, outputs[1]], [Math.ceil(c.rows / 16), c.heads]],
    ]) {
      const layout = device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
      const compute = await pipeline(device, source(name), name, device.createPipelineLayout({ bindGroupLayouts: [layout] }));
      const group = device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      kernels.push({ compute, group, groups });
    }
    const dispatch = (pass, fused) => {
      for (const kernel of fused ? kernels.slice(2) : kernels.slice(0, 2)) {
        pass.setPipeline(kernel.compute);
        pass.setBindGroup(0, kernel.group);
        pass.dispatchWorkgroups(...kernel.groups);
      }
    };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch(pass, false); dispatch(pass, true); pass.end();
    outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * queries.byteLength, queries.byteLength));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const baseline = actual.subarray(0, queries.length), fused = actual.subarray(queries.length);
    const parity = compareOutput(fused, baseline);
    if (!parity.ok) throw new Error(`fused ${JSON.stringify(c)} differs from split: ${parity.maxAbs}`);
    let cpu = null;
    if (c.rows <= 33) {
      const scores = new Float32Array(c.rows * c.rows * c.heads);
      for (let head = 0; head < c.heads; head++) for (let row = 0; row < c.rows; row++) for (let key = 0; key < c.rows; key++) {
        const at = (head * c.rows + row) * c.rows + key;
        if (positions[key * 2] < 0) { scores[at] = -1e30; continue; }
        const qi = (row * c.heads + head) * 64, ki = (key * c.kvHeads + Math.floor(head / (c.heads / c.kvHeads))) * 64;
        let sum = 0;
        for (let col = 0; col < 64; col += 4) {
          let dot = 0;
          for (let part = 0; part < 4; part++) dot = Math.fround(dot + Math.fround(queries[qi + col + part] * keys[ki + col + part]));
          sum = Math.fround(sum + dot);
        }
        scores[at] = sum;
      }
      const expected = reference(c, scores, values).outputs;
      cpu = { fused: compareOutput(fused, expected), split: compareOutput(baseline, expected) };
      if (!cpu.fused.ok || !cpu.split.ok) throw new Error(`fused ${JSON.stringify(c)} differs from CPU: ${cpu.fused.maxAbs}, split ${cpu.split.maxAbs}`);
    }
    const timings = { split: [], fused: [] };
    if (c.rows >= 315 && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 4);
      const repeats = 16;
      for (let i = 0; i < 7; i++) {
        const encoder = device.createCommandEncoder();
        for (const fused of i % 2 ? [true, false] : [false, true]) {
          const pass = profiler.pass(encoder, fused ? "fused" : "split");
          for (let n = 0; n < repeats; n++) dispatch(pass, fused);
          pass.end();
        }
        profiler.finish(encoder);
        device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (i >= 2) for (const name of ["split", "fused"]) timings[name].push(measured[name] / repeats);
      }
    }
    return { ...c, parity, cpu, timings, ok: true };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function fusedMain() {
  const started = performance.now();
  const runtime = await requestDevice({ baseline: true });
  try {
    const source = await kernelSource();
    const results = [];
    for (const c of [
      { rows: 1, heads: 4, kvHeads: 2, dim: 64 },
      { rows: 17, heads: 4, kvHeads: 2, dim: 64, allMasked: true },
      { rows: 17, heads: 4, kvHeads: 2, dim: 64, mask: true },
      { rows: 33, heads: 4, kvHeads: 2, dim: 64, large: true },
      ...[315, 567, 630].map((rows) => ({ rows, heads: 12, kvHeads: 12, dim: 64, mask: true })),
    ]) {
      const result = await withGpuErrors(runtime, () => fusedCase(runtime, source, c), "checking fused vision attention");
      results.push(result); log(JSON.stringify(result));
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", metricMs: performance.now() - started,
      adapter: { name: runtime.name, ...runtime.adapter.info, isFallbackAdapter: Boolean(runtime.adapter.isFallbackAdapter || runtime.adapter.info?.isFallbackAdapter), features: [...runtime.device.features], workgroupStorageBytes: runtime.device.limits.maxComputeWorkgroupStorageSize },
      correctness: { ok: true, cases: results }, kernel: "gemma4_vision_attention_fused" };
  } finally { runtime.device.destroy(); }
}

(new URLSearchParams(location.search).has("fused") ? fusedMain() : main()).catch((error) => {
  window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code };
  log(error.stack || error);
});
