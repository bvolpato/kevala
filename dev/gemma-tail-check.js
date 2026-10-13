// Real WebGPU smoke check for the compact shared-KV Gemma tail. The synthetic pack is
// intentionally tiny: the comparison exercises the production GpuGemma4 graph, not a
// CPU model, while keeping this page suitable for a quick browser guard.

import { Profiler, requestDevice } from "../js/src/gpu.js";
import { GpuGemma4, gemma4Config } from "../js/src/gpu-gemma4.js";
import { kernelSource } from "./wgsl.js";

const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };

const HEADER_CONFIG = {
  hidden_size: 32,
  intermediate_size: 32,
  num_hidden_layers: 4,
  num_attention_heads: 1,
  num_key_value_heads: 1,
  num_kv_shared_layers: 2,
  layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
  head_dim: 32,
  global_head_dim: 32,
  sliding_window: 4,
  max_position_embeddings: 512,
  vocab_size: 262144,
  vocab_size_per_layer_input: 262144,
  rms_norm_eps: 1e-6,
  embedding_scale: 2,
  hidden_size_per_layer_input: 32,
  ple_embedding_scale: 1.125,
  ple_input_scale: Math.SQRT1_2,
  ple_projection_scale: 0.1,
  rope_parameters: {
    sliding_attention: { rope_theta: 10000 },
    full_attention: { rope_theta: 1000000, partial_rotary_factor: 0.25 },
  },
};

function product(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

function rowStats(values, width) {
  let minRms = Infinity;
  let maxRms = 0;
  let maxAbs = 0;
  let nonfinite = 0;
  for (let row = 0; row < values.length / width; row++) {
    let sum = 0;
    for (let d = 0; d < width; d++) {
      const value = values[row * width + d];
      if (!Number.isFinite(value)) nonfinite++;
      sum += value * value;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
    const rms = Math.sqrt(sum / width);
    minRms = Math.min(minRms, rms);
    maxRms = Math.max(maxRms, rms);
  }
  return { rows: values.length / width, minRms, maxRms, maxAbs, nonfinite };
}

function tensor(name, dtype, shape) {
  const elements = product(shape);
  // Q8 packs four signed bytes into each u32, but the pack format still reserves
  // one byte per weight. The shader reads four weights from every four-byte word.
  const size = dtype === "q8" ? elements : elements * 4;
  const scalesSize = dtype === "q8" ? shape[0] * (shape[1] / 32) * 4 : 0;
  return { name, dtype, shape, size, scalesSize };
}

function syntheticTensors() {
  const out = [tensor("embed", "q8", [HEADER_CONFIG.vocab_size, 32]), tensor("ple.norm", "f32", [32]), tensor("norm", "f32", [32])];
  for (let i = 0; i < 4; i++) {
    const n = (part) => `l.${i}.${part}`;
    out.push(tensor(`ple.${i}.proj`, "q8", [32, 32]), tensor(`ple.${i}.embed`, "q8", [HEADER_CONFIG.vocab_size, 32]));
    for (const part of ["attn_norm", "attn_post_norm", "ffn_norm", "ffn_post_norm", "ple_norm"]) out.push(tensor(n(part), "f32", [32]));
    out.push(tensor(n("q"), "q8", [32, 32]), tensor(n("qn"), "f32", [32]), tensor(n("o"), "q8", [32, 32]));
    // Keep K/V for every layer so the same synthetic pack can exercise the
    // no-shared-KV fallback after the shared-tail run.
    out.push(tensor(n("k"), "q8", [32, 32]), tensor(n("kn"), "f32", [32]), tensor(n("v"), "q8", [32, 32]));
    out.push(tensor(n("gate"), "q8", [32, 32]), tensor(n("up"), "q8", [32, 32]), tensor(n("down"), "q8", [32, 32]));
    out.push(tensor(n("ple_gate"), "q8", [32, 32]), tensor(n("ple_out"), "q8", [32, 32]), tensor(n("scalar"), "f32", [1]));
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
        const value = ((i * 17 + ti * 11) % 15) - 7;
        bytes[item.offset + i] = value < 0 ? value + 256 : value;
      }
      for (let i = 0; i < item.scales_size / 4; i++) view.setFloat32(item.scales_offset + i * 4, 0.02 + (i % 3) * 0.003, true);
    } else {
      const values = new Float32Array(bytes.buffer, item.offset, item.size / 4);
      for (let i = 0; i < values.length; i++) values[i] = item.name.endsWith(".scalar") ? 1 : 1 + ((i + ti) % 5) * 0.01;
    }
  }
  const header = new TextEncoder().encode(JSON.stringify({ tensors: metadata }));
  const prefix = new Uint8Array(16 + header.byteLength);
  prefix.set([0x4b, 0x56, 0x4c, 0x41]); // KVLA
  new DataView(prefix.buffer).setUint32(4, 1, true); // format version
  new DataView(prefix.buffer).setUint32(8, header.byteLength, true);
  prefix.set(header, 16);
  return { prefix, bytes };
}

async function main() {
  const source = await kernelSource();
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  runtime.wgsl = source;
  runtime.kernel = "generic";
  const packed = packSynthetic(syntheticTensors());
  const makeModel = async (config, { hostPle = false } = {}) => {
    const cfg = gemma4Config({ config });
    cfg.hostPle = hostPle;
    const model = new GpuGemma4(runtime, { prefix: packed.prefix }, cfg);
    model.write(0, packed.bytes);
    await model.init();
    model.profiler = runtime.device.features.has("timestamp-query") ? new Profiler(runtime.device) : null;
    return { cfg, model };
  };
  const { cfg, model } = await makeModel(HEADER_CONFIG);
  const cases = [];
  const finalRows = [];
  window.gpuBench.correctness = { ok: false, cases, finalRows };
  const assertFinalRows = (values, rows, label) => {
    const stats = { label, ...rowStats(values, cfg.hidden) };
    finalRows.push(stats);
    // These synthetic final-norm weights are in [1, 1.04]. A zero readback can
    // make two graph comparisons agree, but cannot satisfy this independent bound.
    if (values.length !== rows * cfg.hidden || stats.nonfinite || stats.minRms < 0.9 || stats.maxRms > 1.1) {
      throw new Error(`invalid normalized final rows: ${JSON.stringify(stats)}`);
    }
  };
  const read = async (buffer, floats) => {
    const copy = runtime.device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = runtime.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, copy, 0, floats * 4);
      runtime.device.queue.submit([encoder.finish()]);
      await copy.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(copy.getMappedRange().slice(0));
      copy.unmap();
      return values;
    } finally { copy.destroy(); }
  };
  const metricSamples = [];
  let maxAbs = 0;
  const lengths = [1, 31, 65];
  const selectedRows = (length) => length === 1 ? [0, 0] : [length - 1, 1, length - 1];
  const runShared = async () => {
    model.build();
    for (const length of lengths) {
      const ids = Uint32Array.from({ length }, (_, i) => (3 + i * 7) % HEADER_CONFIG.vocab_size);
      const allRows = Uint32Array.from({ length }, (_, i) => i);
      const selected = Uint32Array.from(selectedRows(length));
      const all = await model.forward(ids, allRows);
      if (length === 1) {
        window.gpuBench.correctness.inputStates = {
          embedding: rowStats(await read(model.input, cfg.hidden), cfg.hidden),
          pleToken: rowStats(await read(model.pleToken, cfg.pleDim), cfg.pleDim),
          residual: rowStats(await read(model.x, cfg.hidden), cfg.hidden),
        };
      }
      assertFinalRows(all, length, `shared all T=${length}`);
      const selectedStarted = performance.now();
      const picked = await model.forward(ids, selected);
      const selectedElapsed = performance.now() - selectedStarted;
      const profileMs = model.lastProfile ? Object.values(model.lastProfile).reduce((sum, value) => sum + value, 0) : 0;
      metricSamples.push(profileMs > 0 ? profileMs : Math.max(0.001, selectedElapsed));
      assertFinalRows(picked, selected.length, `shared selected T=${length}`);
      let caseMaxAbs = 0;
      for (let row = 0; row < selected.length; row++) for (let d = 0; d < cfg.hidden; d++) {
        caseMaxAbs = Math.max(caseMaxAbs, Math.abs(picked[row * cfg.hidden + d] - all[selected[row] * cfg.hidden + d]));
      }
      maxAbs = Math.max(maxAbs, caseMaxAbs);
      cases.push({ tokens: length, selected: [...selected], causal: true, sharedKv: true, maxAbs: caseMaxAbs });
      if (!Number.isFinite(caseMaxAbs) || caseMaxAbs > 2e-5) throw new Error(`selected tail mismatch T=${length} maxAbs=${caseMaxAbs}`);
      const empty = await model.forward(ids, []);
      if (empty.length !== 0) throw new Error(`empty selection returned ${empty.length} values`);
    }
  };
  await runShared();

  // Check both the independent embedding contract and the selected-row graph with images.
  const imageIds = Uint32Array.from([3, 258880, 258880, 4, 5]);
  const imageRows = Float32Array.from({ length: 2 * cfg.hidden }, (_, i) => Math.sin(i * 0.17));
  const imageSpans = [{ start: 1, len: 2, rows: imageRows }];
  const imageAll = await model.forward(imageIds, [0, 1, 2, 3, 4], imageSpans);
  assertFinalRows(imageAll, imageIds.length, "image all");
  const injected = await read(model.input, imageIds.length * cfg.hidden);
  for (let i = 0; i < imageRows.length; i++) if (injected[cfg.hidden + i] !== imageRows[i]) throw new Error("image rows were scaled or injected after the PLE input copy");
  const pleTokens = await read(model.pleToken, imageIds.length * cfg.pleDim);
  const ple = model.weight("ple.3.embed").info;
  const view = new DataView(packed.bytes.buffer);
  for (let row = 1; row <= 2; row++) for (let d = 0; d < cfg.pleDim; d++) {
    const byte = packed.bytes[ple.offset + d];
    const expected = Math.fround(Math.fround((byte < 128 ? byte : byte - 256) * view.getFloat32(ple.scales_offset, true)) * cfg.pleEmbeddingScale);
    if (Math.abs(pleTokens[row * cfg.pleDim + d] - expected) > 1e-7) throw new Error("image PLE lookup did not use PAD identity");
  }
  const imageSelected = [1, 4, 1];
  const imagePicked = await model.forward(imageIds, imageSelected, imageSpans);
  assertFinalRows(imagePicked, imageSelected.length, "image selected");
  let imageMaxAbs = 0;
  for (let row = 0; row < imageSelected.length; row++) for (let d = 0; d < cfg.hidden; d++) imageMaxAbs = Math.max(imageMaxAbs, Math.abs(imagePicked[row * cfg.hidden + d] - imageAll[imageSelected[row] * cfg.hidden + d]));
  if (!Number.isFinite(imageMaxAbs) || imageMaxAbs > 2e-5) throw new Error(`image selected-tail mismatch maxAbs=${imageMaxAbs}`);
  cases.push({ tokens: imageIds.length, selected: imageSelected, imageRows: 2, causal: true, sharedKv: true, maxAbs: imageMaxAbs });
  const changedImageRows = Float32Array.from(imageRows, (value, i) => 2.5 * Math.cos(i * 0.13) - value);
  const changedImage = await model.forward(imageIds, [4], [{ start: 1, len: 2, rows: changedImageRows }]);
  assertFinalRows(changedImage, 1, "changed image selected");
  let imageHiddenDelta = 0;
  for (let d = 0; d < cfg.hidden; d++) imageHiddenDelta = Math.max(imageHiddenDelta, Math.abs(changedImage[d] - imageAll[4 * cfg.hidden + d]));
  window.gpuBench.correctness.imageHiddenDelta = imageHiddenDelta;
  if (!Number.isFinite(imageHiddenDelta) || imageHiddenDelta <= 1e-6) throw new Error(`changing projected image rows did not change the final text row: maxAbs=${imageHiddenDelta}`);

  // The host mode retains the pack's exact Q8 bytes, then copies independent
  // per-layer row slices before PLE combine. Compare full, compact and chunked graphs.
  const { model: hostModel } = await makeModel(HEADER_CONFIG, { hostPle: true });
  let hostPleMaxAbs = 0;
  let hostPleExact = true;
  const hostCases = [];
  const compareHost = async (ids, selected, spans = []) => {
    const dense = await model.forward(ids, selected, spans);
    const host = await hostModel.forward(ids, selected, spans);
    assertFinalRows(dense, selected.length, `dense PLE T=${ids.length} R=${selected.length}`);
    assertFinalRows(host, selected.length, `host PLE T=${ids.length} R=${selected.length}`);
    let maxAbs = 0;
    for (let i = 0; i < dense.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(dense[i] - host[i]));
      hostPleExact &&= dense[i] === host[i];
    }
    if (!Number.isFinite(maxAbs) || maxAbs > 2e-5) throw new Error(`host PLE mismatch T=${ids.length} selected=${selected.length} maxAbs=${maxAbs}`);
    if (hostModel.hostPleUpload !== null || hostModel.hostPleOffsets !== null) throw new Error("host PLE temporary upload survived a forward");
    hostPleMaxAbs = Math.max(hostPleMaxAbs, maxAbs);
    hostCases.push({ tokens: ids.length, selected: selected.length, images: spans.length, maxAbs });
  };
  for (const length of [1, 31, 65, 257]) {
    const ids = Uint32Array.from({ length }, (_, i) => (3 + i * 7) % HEADER_CONFIG.vocab_size);
    await compareHost(ids, Uint32Array.from({ length }, (_, i) => i));
    await compareHost(ids, selectedRows(length));
  }
  await compareHost(imageIds, [0, 1, 2, 3, 4], imageSpans);
  const hostPleTokens = await read(hostModel.pleToken, imageIds.length * cfg.pleDim);
  const densePleTokens = await read(model.pleToken, imageIds.length * cfg.pleDim);
  for (let i = 0; i < densePleTokens.length; i++) if (densePleTokens[i] !== hostPleTokens[i]) throw new Error(`host PLE row differs from the GPU lookup at ${i}`);
  await compareHost(imageIds, imageSelected, imageSpans);
  const expectedHostBytes = 4 * HEADER_CONFIG.vocab_size * cfg.pleDim * 1.125;
  if (hostModel.memory.hostWeights !== expectedHostBytes || hostModel.memory.gpuWeights !== model.memory.gpuWeights - expectedHostBytes) throw new Error("host PLE weight accounting differs from original pack bytes");
  if (hostModel.weights.missing().length || [...hostModel.tensors.values()].some((entry) => entry.hostTable && (entry.buf || entry.sbuf || !entry.host))) throw new Error("host PLE tables are incomplete or still allocated on the GPU");

  // Native Gemma scoring rejects bidirectional text attention. Keep that
  // contract explicit while still exercising the parser's failure path.
  let bidirectionalRejected = false;
  try {
    gemma4Config({ config: { ...HEADER_CONFIG, use_bidirectional_attention: "all" } });
  } catch {
    bidirectionalRejected = true;
  }
  if (!bidirectionalRejected) throw new Error("bidirectional Gemma config was accepted");

  // A separately parsed no-sharing config checks that arbitrary row selection
  // falls back to the ordinary full graph without compact-tail allocations.
  const { cfg: fallbackCfg, model: fallbackModel } = await makeModel({ ...HEADER_CONFIG, num_kv_shared_layers: 0 });
  const fallbackLength = 31;
  const fallbackIds = Uint32Array.from({ length: fallbackLength }, (_, i) => (5 + i * 11) % HEADER_CONFIG.vocab_size);
  const fallbackSelected = Uint32Array.from([fallbackLength - 1, 1, fallbackLength - 1]);
  const fallbackAll = await fallbackModel.forward(fallbackIds, Uint32Array.from({ length: fallbackLength }, (_, i) => i));
  const fallbackPicked = await fallbackModel.forward(fallbackIds, fallbackSelected);
  assertFinalRows(fallbackAll, fallbackLength, "no-shared all");
  assertFinalRows(fallbackPicked, fallbackSelected.length, "no-shared selected");
  let fallbackMaxAbs = 0;
  for (let row = 0; row < fallbackSelected.length; row++) for (let d = 0; d < fallbackCfg.hidden; d++) {
    fallbackMaxAbs = Math.max(fallbackMaxAbs, Math.abs(fallbackPicked[row * fallbackCfg.hidden + d] - fallbackAll[fallbackSelected[row] * fallbackCfg.hidden + d]));
  }
  if (!Number.isFinite(fallbackMaxAbs) || fallbackMaxAbs > 2e-5) throw new Error(`no-shared fallback mismatch maxAbs=${fallbackMaxAbs}`);
  cases.push({ tokens: fallbackLength, selected: [...fallbackSelected], causal: true, sharedKv: false, maxAbs: fallbackMaxAbs });

  const metricMs = Math.exp(metricSamples.reduce((sum, value) => sum + Math.log(value), 0) / metricSamples.length);
  const info = runtime.adapter.info || {};
  const correctness = { ...window.gpuBench.correctness, ok: true, cases, finalRows, imageHiddenDelta, maxAbs: Math.max(maxAbs, fallbackMaxAbs, imageMaxAbs, hostPleMaxAbs), tolerance: 2e-5, emptyRows: true, duplicateOrder: true, lengths, bidirectionalRejected, noSharedFallback: true, imageInjection: true, imagePlePad: true, hostPle: { cases: hostCases, maxAbs: hostPleMaxAbs, bitExact: hostPleExact, savedGpuBytes: hostModel.memory.hostWeights } };
  const adapter = {
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
  window.gpuBench = {
    status: "done",
    done: true,
    backend: "webgpu",
    metricMs,
    metric: "geometric mean of selected-tail forward milliseconds (GPU kernel timestamps when available)",
    method: model.profiler ? "timestamp-query" : "performance.now",
    adapter,
    correctness,
  };
  log(`done: selected rows match all rows; maxAbs=${maxAbs}; metricMs=${metricMs.toFixed(4)}`);
}

window.gpuBench = { status: "running", backend: "webgpu", check: "gemma4-selected-tail" };
main().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
  log(`error: ${error?.stack || error}`);
});
