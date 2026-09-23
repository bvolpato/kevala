// Bounded WebGPU check for the compact final Qwen stage-2 tail. It compares the production graph
// with identity-row output, then exercises reordered duplicates, empty rows, a cache hit, and both
// possible final layer kinds. The synthetic pack is intentionally tiny but uses the production
// Qwen dimensions required by the specialized kernels.

import { Profiler, requestDevice } from "../js/src/gpu.js";
import { GpuKev, kevConfig } from "../js/src/gpu-kev.js";
import { kernelSource } from "./wgsl.js";

const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };

const HIDDEN = 256;
const INTERMEDIATE = 256;
const HEADS = 1;
const KV_HEADS = 1;
const LIN_KEY_HEADS = 2;
const LIN_HEADS = 2;
const ATTENTION_Q = HEADS * 256;
const ATTENTION_KV = 2 * KV_HEADS * 256;
const ATTENTION_WIDTH = 2 * ATTENTION_Q + ATTENTION_KV;
const LINEAR_DIM = 2 * LIN_KEY_HEADS * 128 + LIN_HEADS * 128;
const LINEAR_PROJ = LINEAR_DIM + LIN_HEADS * 128;
const LINEAR_OUT = LIN_HEADS * 128;

const BASE_CONFIG = {
  hidden_size: HIDDEN,
  intermediate_size: INTERMEDIATE,
  num_attention_heads: HEADS,
  num_key_value_heads: KV_HEADS,
  head_dim: 256,
  rotary_dim: 64,
  rope_theta: 10000,
  linear_num_key_heads: LIN_KEY_HEADS,
  linear_num_value_heads: LIN_HEADS,
  linear_key_head_dim: 128,
  linear_value_head_dim: 128,
  linear_conv_kernel_dim: 4,
  rms_norm_eps: 1e-6,
  max_state: 96,
  max_branch: 96,
  vocab_size: 64,
};

function product(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function tensor(name, dtype, shape) {
  const elements = product(shape);
  const size = dtype === "q8" ? elements : elements * 4;
  const scalesSize = dtype === "q8" ? shape[0] * (shape[1] / 32) * 4 : 0;
  return { name, dtype, shape, size, scalesSize };
}

function syntheticTensors(layerTypes) {
  const out = [];
  for (let i = 0; i < layerTypes.length; i++) {
    const n = (part) => `L.${i}.${part}`;
    out.push(tensor(n("in_norm"), "f32", [HIDDEN]), tensor(n("post_norm"), "f32", [HIDDEN]));
    if (layerTypes[i] === "full_attention") {
      out.push(
        tensor(n("qkv"), "q8", [ATTENTION_WIDTH, HIDDEN]),
        tensor(n("q_norm"), "f32", [256]),
        tensor(n("k_norm"), "f32", [256]),
        tensor(n("o"), "q8", [HIDDEN, ATTENTION_Q]),
      );
    } else {
      out.push(
        tensor(n("qkvz"), "q8", [LINEAR_PROJ, HIDDEN]),
        tensor(n("a"), "f32", [LIN_HEADS, HIDDEN]),
        tensor(n("b"), "f32", [LIN_HEADS, HIDDEN]),
        tensor(n("conv"), "f32", [LINEAR_DIM, 4]),
        tensor(n("dt_bias"), "f32", [LIN_HEADS]),
        tensor(n("neg_a"), "f32", [LIN_HEADS]),
        tensor(n("gnorm"), "f32", [128]),
        tensor(n("out"), "q8", [HIDDEN, LINEAR_OUT]),
      );
    }
    out.push(tensor(n("gate_up"), "q8", [2 * INTERMEDIATE, HIDDEN]), tensor(n("down"), "q8", [HIDDEN, INTERMEDIATE]));
  }
  return out;
}

function packSynthetic(tensors) {
  let offset = 0;
  const metadata = tensors.map((entry) => {
    const item = { name: entry.name, dtype: entry.dtype, shape: entry.shape, size: entry.size, offset };
    if (entry.dtype === "q8") item.block = 32;
    offset += entry.size;
    if (entry.dtype === "q8") {
      item.scales_size = entry.scalesSize;
      item.scales_offset = offset;
      offset += entry.scalesSize;
    }
    return item;
  });
  const bytes = new Uint8Array(offset);
  const view = new DataView(bytes.buffer);
  for (let ti = 0; ti < metadata.length; ti++) {
    const item = metadata[ti];
    if (item.dtype === "q8") {
      for (let i = 0; i < item.size; i++) {
        const value = ((i * 13 + ti * 7) % 15) - 7;
        bytes[item.offset + i] = value < 0 ? value + 256 : value;
      }
      for (let i = 0; i < item.scales_size / 4; i++) view.setFloat32(item.scales_offset + i * 4, 0.01 + (i % 3) * 0.002, true);
    } else {
      const values = new Float32Array(bytes.buffer, item.offset, item.size / 4);
      for (let i = 0; i < values.length; i++) {
        if (item.name.endsWith(".neg_a")) values[i] = -0.1;
        else if (item.name.endsWith(".dt_bias")) values[i] = 0.05;
        else if (item.name.endsWith(".conv")) values[i] = 0.02 + (i % 4) * 0.01;
        else if (item.name.endsWith(".a") || item.name.endsWith(".b")) values[i] = 0.01 + (i % 5) * 0.002;
        else values[i] = 1.0 + (i % 3) * 0.01;
      }
    }
  }
  const header = new TextEncoder().encode(JSON.stringify({ tensors: metadata }));
  const prefix = new Uint8Array(16 + header.byteLength);
  new DataView(prefix.buffer).setUint32(8, header.byteLength, true);
  prefix.set(header, 16);
  return { prefix, bytes };
}

function inputs(stateLength, branchLength) {
  const x1 = Float32Array.from({ length: stateLength * HIDDEN }, (_, i) => Math.fround(Math.sin(i * 0.013) * 0.2));
  const x2 = Float32Array.from({ length: branchLength * HIDDEN }, (_, i) => Math.fround(Math.cos((i + 17) * 0.017) * 0.2));
  const ids = Uint32Array.from({ length: stateLength }, (_, i) => (i * 7 + 3) % BASE_CONFIG.vocab_size);
  return { x1, x2, ids };
}

function batch(stateLength, branchLength, rows) {
  return {
    T1: stateLength,
    T2: branchLength,
    states: [[0, stateLength]],
    branches: [[0, branchLength, 0]],
    rows,
  };
}

function adapterInfo(runtime) {
  const info = runtime.adapter.info || {};
  return {
    name: runtime.name,
    vendor: info.vendor || null,
    architecture: info.architecture || null,
    device: info.device || null,
    description: info.description || null,
    isFallbackAdapter: Boolean(info.isFallbackAdapter || runtime.adapter.isFallbackAdapter),
    features: [...runtime.device.features].sort(),
    limits: {
      maxBufferSize: runtime.device.limits.maxBufferSize,
      maxStorageBufferBindingSize: runtime.device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize,
    },
  };
}

function compareRows(all, selected, rows) {
  let maxAbs = 0;
  for (let r = 0; r < rows.length; r++) {
    for (let d = 0; d < HIDDEN; d++) {
      const abs = Math.abs(selected[r * HIDDEN + d] - all[rows[r] * HIDDEN + d]);
      if (!Number.isFinite(abs)) return { ok: false, maxAbs, message: "non-finite selected output" };
      maxAbs = Math.max(maxAbs, abs);
    }
  }
  return { ok: maxAbs <= 2e-4, maxAbs, tolerance: 2e-4 };
}

async function makeModel(runtime, layerTypes, packed, Model = GpuKev) {
  const header = { config: { ...BASE_CONFIG, layer_types: layerTypes } };
  const model = new Model(runtime, { prefix: packed.prefix }, kevConfig(header));
  model.write(0, packed.bytes);
  await model.init();
  return model;
}

async function checkConfig(runtime, layerTypes, Baseline) {
  const packed = packSynthetic(syntheticTensors(layerTypes));
  const stateLength = 65;
  const branchLength = 7;
  const { x1, x2, ids } = inputs(stateLength, branchLength);
  const allRows = Uint32Array.from({ length: branchLength }, (_, i) => i);
  const selectedRows = Uint32Array.from([branchLength - 1, 1, branchLength - 1, 0, 4, 2, 3, 5, 4]);

  const reference = await makeModel(runtime, layerTypes, packed, Baseline);
  const all = await reference.forward(x1, x2, batch(stateLength, branchLength, allRows), ids);
  const selectedModel = await makeModel(runtime, layerTypes, packed);
  selectedModel.profiler = runtime.device.features.has("timestamp-query") ? new Profiler(runtime.device) : null;
  const start = performance.now();
  const selected = await selectedModel.forward(x1, x2, batch(stateLength, branchLength, selectedRows), ids);
  const wallMs = performance.now() - start;
  const profileMs = selectedModel.lastProfile ? Object.values(selectedModel.lastProfile).reduce((sum, value) => sum + value, 0) : 0;
  const comparison = compareRows(all, selected, selectedRows);
  if (!comparison.ok) throw new Error(`${layerTypes.at(-1)} selected mismatch maxAbs=${comparison.maxAbs}`);
  const grownRows = Uint32Array.from({ length: 131 }, (_, i) => selectedRows[i % selectedRows.length]);
  const grown = await selectedModel.forward(x1, x2, batch(stateLength, branchLength, grownRows), ids);
  const growthComparison = compareRows(all, grown, grownRows);
  if (!growthComparison.ok) throw new Error(`${layerTypes.at(-1)} growth mismatch maxAbs=${growthComparison.maxAbs}`);
  const empty = await selectedModel.forward(x1, x2, batch(stateLength, branchLength, []), ids);
  if (empty.length !== 0) throw new Error(`${layerTypes.at(-1)} empty selection returned ${empty.length} values`);

  // The first call populated the cache. A cache hit must produce the same selected rows while
  // skipping stage 1; use a fresh model for the comparison so this also checks carry reuse.
  const cacheModel = await makeModel(runtime, layerTypes, packed);
  await cacheModel.forward(x1, x2, batch(stateLength, branchLength, allRows), ids);
  const cached = await cacheModel.forward(new Float32Array(0), x2, batch(stateLength, branchLength, selectedRows), ids);
  const cacheComparison = compareRows(all, cached, selectedRows);
  if (!cacheComparison.ok) throw new Error(`${layerTypes.at(-1)} cache mismatch maxAbs=${cacheComparison.maxAbs}`);
  if (cacheModel.stats.hits !== 1) throw new Error(`${layerTypes.at(-1)} did not exercise exactly one cache hit`);
  return {
    finalLayer: layerTypes.at(-1),
    stateLength,
    branchLength,
    selected: [...selectedRows],
    rowsExceedSequence: selectedRows.length > branchLength,
    selectedMaxAbs: comparison.maxAbs,
    growthMaxAbs: growthComparison.maxAbs,
    grownRows: grownRows.length,
    cacheMaxAbs: cacheComparison.maxAbs,
    emptyRows: true,
    cacheHit: cacheModel.stats.hits === 1,
    wallMs,
    gpuMs: profileMs,
  };
}

async function main() {
  const baselineUrl = new URLSearchParams(location.search).get("baseline");
  const Baseline = baselineUrl ? (await import(new URL(baselineUrl, location.href).href)).GpuKev : GpuKev;
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  runtime.wgsl = await kernelSource();
  runtime.kernel = "generic";
  const cases = [
    await checkConfig(runtime, ["linear_attention", "full_attention"], Baseline),
    await checkConfig(runtime, ["full_attention", "linear_attention"], Baseline),
  ];
  const metricSamples = cases.map((item) => item.gpuMs > 0 ? item.gpuMs : Math.max(0.001, item.wallMs));
  const metricMs = Math.exp(metricSamples.reduce((sum, value) => sum + Math.log(value), 0) / metricSamples.length);
  window.gpuBench = {
    status: "done",
    done: true,
    backend: "webgpu",
    metricMs,
    metric: "geometric mean of selected-forward milliseconds (GPU kernel timestamps when available)",
    method: runtime.device.features.has("timestamp-query") ? "timestamp-query" : "performance.now",
    adapter: adapterInfo(runtime),
    correctness: {
      ok: true,
      baseline: baselineUrl || "identity rows",
      cases,
      tolerance: 2e-4,
      duplicateOrder: true,
      emptyRows: true,
      rowsExceedSequence: true,
      cacheHit: true,
      finalFullAndLinear: true,
    },
  };
  log(`done: full and linear final layers match identity rows; metricMs=${metricMs.toFixed(4)}`);
}

window.gpuBench = { status: "running", backend: "webgpu", check: "qwen-final-tail" };
main().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
  log(`error: ${error?.stack || error}`);
});
