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
  max_position_embeddings: 128,
  vocab_size: 64,
  vocab_size_per_layer_input: 64,
  rms_norm_eps: 1e-6,
  embedding_scale: 1,
  ple_dim: 32,
  ple_embedding_scale: 1,
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

function tensor(name, dtype, shape) {
  const elements = product(shape);
  // Q8 packs four signed bytes into each u32, but the pack format still reserves
  // one byte per weight. The shader reads four weights from every four-byte word.
  const size = dtype === "q8" ? elements : elements * 4;
  const scalesSize = dtype === "q8" ? shape[0] * (shape[1] / 32) * 4 : 0;
  return { name, dtype, shape, size, scalesSize };
}

function syntheticTensors() {
  const out = [tensor("embed", "q8", [64, 32]), tensor("ple.norm", "f32", [32]), tensor("norm", "f32", [32])];
  for (let i = 0; i < 4; i++) {
    const n = (part) => `l.${i}.${part}`;
    out.push(tensor(`ple.${i}.proj`, "q8", [32, 32]), tensor(`ple.${i}.embed`, "q8", [64, 32]));
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
  const makeModel = async (config) => {
    const cfg = gemma4Config({ config });
    const model = new GpuGemma4(runtime, { prefix: packed.prefix }, cfg);
    model.write(0, packed.bytes);
    await model.init();
    model.profiler = runtime.device.features.has("timestamp-query") ? new Profiler(runtime.device) : null;
    return { cfg, model };
  };
  const { cfg, model } = await makeModel(HEADER_CONFIG);
  const cases = [];
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
      const selectedStarted = performance.now();
      const picked = await model.forward(ids, selected);
      const selectedElapsed = performance.now() - selectedStarted;
      const profileMs = model.lastProfile ? Object.values(model.lastProfile).reduce((sum, value) => sum + value, 0) : 0;
      metricSamples.push(profileMs > 0 ? profileMs : Math.max(0.001, selectedElapsed));
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
  let fallbackMaxAbs = 0;
  for (let row = 0; row < fallbackSelected.length; row++) for (let d = 0; d < fallbackCfg.hidden; d++) {
    fallbackMaxAbs = Math.max(fallbackMaxAbs, Math.abs(fallbackPicked[row * fallbackCfg.hidden + d] - fallbackAll[fallbackSelected[row] * fallbackCfg.hidden + d]));
  }
  if (!Number.isFinite(fallbackMaxAbs) || fallbackMaxAbs > 2e-5) throw new Error(`no-shared fallback mismatch maxAbs=${fallbackMaxAbs}`);
  cases.push({ tokens: fallbackLength, selected: [...fallbackSelected], causal: true, sharedKv: false, maxAbs: fallbackMaxAbs });

  const metricMs = Math.exp(metricSamples.reduce((sum, value) => sum + Math.log(value), 0) / metricSamples.length);
  const info = runtime.adapter.info || {};
  const correctness = { ok: true, cases, maxAbs: Math.max(maxAbs, fallbackMaxAbs), tolerance: 2e-5, emptyRows: true, duplicateOrder: true, lengths, bidirectionalRejected, noSharedFallback: true };
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
