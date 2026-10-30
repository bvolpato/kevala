// Gemma 4 E2B/E4B text trunk on WebGPU.
//
// E2B and E4B are dense models. Their distinctive GPU requirements are the packed
// per-layer embeddings (PLE), 256-wide local heads, 512-wide proportional-RoPE
// global heads, and shared KV states in the final layers. This implementation keeps
// projection tensors and word embeddings on the device. Image sessions can keep
// original Q8 PLE tables on the host and upload only the rows a request uses.

import { GpuWeights, pipeline, matmulPipelines, dispatchMatmul, encodeMatmul, encodeMatvec, inErrorScopes, mmSplits, rowsPerThread, SPLIT_SCRATCH, YIELD_TOKENS, YIELD_CHUNKS, chunks, endChunk, withGpuErrors } from "./gpu.js";
import { calibrateMatmul, selectedMatmulKernel } from "./gpu-tuning.js";
import { calibrateMatvec } from "./gpu-matvec.js";
import { calibrateGemma4K64, gemma4K64Eligible } from "./gpu-gemma4-k64.js";
import { calibrateGemma4Mlp, encodeGemma4Mlp, gemma4MlpBindGroup, gemma4MlpEligible } from "./gpu-gemma4-mlp.js";
import { calibrateGemma4BatchMlp, encodeGemma4BatchMlp, gemma4BatchMlpBindGroups, gemma4BatchMlpEligible, gemma4BatchMlpPartBytes } from "./gpu-gemma4-batch-mlp.js";
import { calibrateGemma4BatchSingle, encodeGemma4BatchSingle, gemma4BatchSingleEligible } from "./gpu-gemma4-batch-single.js";
import { calibrateGemma4GridMlp, encodeGemma4GridMlp, Gemma4GridMlpCache, gemma4GridMlpEligible, gemma4GridMlpTokens } from "./gpu-gemma4-grid-mlp.js";
import { validateGemma4Rms } from "./gpu-gemma4-rms.js";
import { gemma4RmsScaleNextPlan } from "./gpu-gemma4-rms-plan.js";
import { calibrateGemma4Attention } from "./gpu-gemma4-attention.js";
import { calibrateMatvecGelu, encodeMatvecGelu, matvecGeluBindGroup } from "./gpu-matvec-gelu.js";
import { calibrateMatvecTwo, encodeMatvecTwo } from "./gpu-matvec-two.js";
import { gpuVisionProjection } from "./gpu-gemma4-vision.js";
import { Gemma4TextPrefixCache, prefixRowView, textPrefixCachePlan } from "./gpu-gemma4-prefix-cache.js";
import { calibrateGemma4PrefixRetile } from "./gpu-gemma4-prefix-retile.js";
import { calibrateGemma4Bn32 } from "./gpu-gemma4-bn32.js";
import { calibrateGemma4PrefixGelu } from "./gpu-gemma4-prefix-gelu.js";
import { createGemma4PleReduce, gemma4PleReduceAdjacent, gemma4PleReduceChoice, gemma4PleReduceGroup } from "./gpu-gemma4-ple.js";

let U;

const GEMMA4_KERNELS = [
  ["EMBED", "gemma4_embed"],
  ["RMS", "gemma4_rms"],
  ["RMS_RESIDUAL_NORM", "gemma4_rms_residual_norm"],
  ["RMS_SCALE", "gemma4_rms_scale"],
  ["QKV", "gemma4_qkv"],
  ["ATTN", "gemma4_attention"],
  ["GELU", "gemma4_gelu"],
  ["PLE", "gemma4_ple"],
  ["RESIDUAL", "gemma4_residual"],
  ["GATHER", "gemma4_gather"],
];

const f32 = (v) => ({ f: v });
const pow2 = (n) => 2 ** Math.ceil(Math.log2(Math.max(1, n)));
const PLE_VECTOR_CACHE_LIMIT = 16 * 1024 * 1024;

function bf16Round(value) {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  f[0] = Math.fround(value);
  u[0] = (u[0] + 0x7fff + ((u[0] >>> 16) & 1)) & 0xffff0000;
  return f[0];
}

/** The text config: the flat pack config, or the `text_config` nested in an upstream config. */
function textConfig(header) {
  const c = header?.config || {};
  return c.text_config || c;
}

const invalid = (what) => new Error(`Gemma 4 GPU config ${what}`);
const positiveInteger = (n) => Number.isSafeInteger(n) && n > 0;

function validImageRows(rows, length, hidden, device) {
  if (rows instanceof Float32Array) return rows.length === length * hidden && !rows.some((value) => !Number.isFinite(value));
  return !!gpuVisionProjection(rows, device, length * hidden * 4);
}

function mappedImageRows(readback, bytes, flags) {
  try {
    const mapped = readback.getMappedRange(0, bytes + flags * 4);
    if (new Uint32Array(mapped, bytes, flags).some((value) => value !== 0)) throw new Error("Gemma 4 image projection contains non-finite values");
    return new Float32Array(mapped.slice(0, bytes));
  } finally { readback.unmap(); }
}

function sharedImagePrefix(sequences) {
  if (sequences.length < 2) return 0;
  const first = sequences[0];
  if (!first?.imageSpans?.length) return 0;
  const ids = sequences.map((sequence) => sequence.ids);
  if (ids.some((sequence) => !(sequence instanceof Uint32Array) || sequence.length < 2)) return 0;
  let common = Math.min(...ids.map((sequence) => sequence.length));
  for (let i = 1; i < ids.length && common; i++) {
    let j = 0;
    while (j < common && ids[0][j] === ids[i][j]) j++;
    common = j;
  }
  if (!common || ids.some((sequence) => sequence.length === common)) return 0;
  for (let i = 1; i < sequences.length; i++) {
    const spans = sequences[i].imageSpans || [];
    if (spans.length !== first.imageSpans.length || spans.some((span, j) => {
      const original = first.imageSpans[j];
      return span.start !== original.start || span.len !== original.len || span.rows !== original.rows;
    })) return 0;
  }
  if (first.imageSpans.some((span) => span.start < 0 || span.len < 1 || span.start + span.len > common)) return 0;
  return common;
}

/**
 * Reads the pack's text config (Hugging Face field names, as the converter copies them) into the
 * fields the GPU trunk uses. It accepts what the Rust `Gemma4Config` accepts, so a pack the
 * coordinator cannot load fails here with the same reason.
 */
export function gemma4Config(header) {
  const c = textConfig(header);
  const flag = (key) => {
    if (!Object.hasOwn(c, key)) return false;
    if (typeof c[key] !== "boolean") throw invalid(`${key} must be boolean`);
    return c[key];
  };
  if (Object.hasOwn(c, "hidden_activation") && c.hidden_activation !== "gelu_pytorch_tanh") throw invalid("supports only hidden_activation=gelu_pytorch_tanh");
  for (const unsupported of ["attention_bias", "attention_k_eq_v", "enable_moe_block"]) {
    if (flag(unsupported)) throw invalid(`does not support ${unsupported}=true`);
  }

  const layers = Number(c.num_hidden_layers);
  const hidden = Number(c.hidden_size);
  const intermediate = Number(c.intermediate_size);
  if (![layers, hidden, intermediate].every(positiveInteger)) throw invalid("has invalid model dimensions");
  const heads = Number(c.num_attention_heads);
  // The K=V attention path, the only one with a separate global head count, is rejected above.
  const kvHeads = Number(c.num_key_value_heads);
  if (!positiveInteger(heads) || !positiveInteger(kvHeads)) throw invalid("has invalid attention head count");
  if (heads % kvHeads) throw invalid("requires attention heads divisible by KV heads");
  const localHeadDim = Number(c.head_dim ?? 256);
  const globalHeadDim = Number(c.global_head_dim ?? 512);
  const headDim = (n, limit) => positiveInteger(n) && n <= limit && n % 2 === 0;
  if (!headDim(localHeadDim, 256) || !headDim(globalHeadDim, 512)) throw invalid("supports even local head dimensions up to 256 and global dimensions up to 512");
  const kvSharedLayers = Number(c.num_kv_shared_layers ?? 0);
  if (!Number.isSafeInteger(kvSharedLayers) || kvSharedLayers < 0 || kvSharedLayers >= layers) throw invalid("has invalid KV sharing count");
  const maxPosition = Number(c.max_position_embeddings ?? 131072);
  if (!positiveInteger(maxPosition)) throw invalid("has invalid position limit");
  const pleDim = Number(c.hidden_size_per_layer_input ?? 256);
  const vocab = Number(c.vocab_size ?? 262144);
  const pleVocab = Number(c.vocab_size_per_layer_input ?? vocab);
  if (![pleDim, vocab, pleVocab].every(positiveInteger)) throw invalid("has invalid PLE or vocabulary dimensions");
  if (hidden % 4 || intermediate % 4 || pleDim % 4) throw invalid("requires hidden, intermediate, and PLE dimensions divisible by four");
  if (c.use_bidirectional_attention != null) throw invalid("does not support use_bidirectional_attention for direct scoring");

  const layerTypes = Array.isArray(c.layer_types) ? c.layer_types.map(String) : null;
  if (!layerTypes || layerTypes.length !== layers) throw invalid("requires layer_types for every decoder layer");
  if (layerTypes.some((t) => t !== "sliding_attention" && t !== "full_attention")) throw invalid("has invalid layer_types");
  if (layerTypes.at(-1) !== "full_attention") throw invalid("requires the final decoder layer to be full_attention");
  // The last layers reuse the keys and values of the last earlier layer of their attention type.
  const sharedStart = layers - kvSharedLayers;
  const sources = new Set(layerTypes.slice(0, sharedStart));
  for (const type of new Set(layerTypes.slice(sharedStart))) {
    if (!sources.has(type)) throw invalid(`has no non-shared KV source for ${type}`);
  }
  const window = Number(c.sliding_window ?? 512);
  if (!positiveInteger(window)) throw invalid("has invalid sliding window");

  const rope = c.rope_parameters || {};
  const localRope = rope.sliding_attention || {};
  const globalRope = rope.full_attention || {};
  const localRopeType = typeof localRope.rope_type === "string" ? localRope.rope_type : "default";
  const globalRopeType = typeof globalRope.rope_type === "string" ? globalRope.rope_type : "proportional";
  if (localRopeType !== "default" || globalRopeType !== "proportional") throw invalid(`supports only sliding rope_type=default and full rope_type=proportional (got ${localRopeType}/${globalRopeType})`);
  const ropeGlobalFraction = Number(globalRope.partial_rotary_factor ?? 0.25);
  if (!Number.isFinite(ropeGlobalFraction) || ropeGlobalFraction < 0 || ropeGlobalFraction > 1 || Math.floor((globalHeadDim * ropeGlobalFraction) / 2) * 2 > 256) throw invalid("has unsupported global proportional RoPE");

  return {
    hidden,
    layers,
    intermediate,
    heads,
    kvHeads,
    localHeadDim,
    globalHeadDim,
    layerTypes,
    full: layerTypes.map((t) => t === "full_attention"),
    sharedStart,
    kvSharedLayers,
    window,
    maxPosition,
    vocab,
    pleDim,
    pleVocab,
    eps: Number(c.rms_norm_eps ?? 1e-6),
    // the converter records these scales; an older pack without them gets the same values
    embeddingScale: Number(c.embedding_scale ?? bf16Round(Math.sqrt(hidden))),
    pleEmbeddingScale: Number(c.ple_embedding_scale ?? bf16Round(Math.sqrt(pleDim))),
    pleInputScale: Number(c.ple_input_scale ?? Math.SQRT1_2),
    pleProjectionScale: Number(c.ple_projection_scale ?? 1 / Math.sqrt(hidden)),
    useDoubleWideMlp: !!c.use_double_wide_mlp,
    ropeLocalTheta: Number(localRope.rope_theta ?? c.rope_theta ?? 10000),
    ropeGlobalTheta: Number(globalRope.rope_theta ?? c.rope_theta ?? 1000000),
    ropeGlobalFraction,
    causal: 1,
  };
}

/** Dense Gemma 4 text trunk. The caller owns CPU tokenization and readout labels. */
export class GpuGemma4 {
  constructor(gpu, layout, cfg) {
    U = GPUBufferUsage;
    this.device = gpu.device;
    this.runtime = gpu;
    this.subgroup32 = !!gpu.subgroup32;
    this.wgsl = gpu.wgsl;
    this.name = gpu.name;
    this.gpuKernel = gpu.kernel || "auto";
    this.cfg = cfg;
    const hostTables = cfg.hostPle === true ? Array.from({ length: cfg.layers }, (_, i) => `ple.${i}.embed`) : [];
    if (cfg.hostWord) hostTables.push("embed");
    this.weights = new GpuWeights(gpu.device, layout, { hostTables, hostStore: cfg.hostStore });
    this.pleCache = new Map();
    this.pleCacheBytes = 0;
    this.tensors = this.weights.tensors;
    this.memory = {
      hostWeights: 0,
      opfsWeights: 0,
      gpuWeights: 0,
      embeddingStorage: cfg.hostStore ? "opfs-q8-rows" : cfg.hostPle === true ? "host-q8-rows" : "gpu-q8",
    };
    for (const t of this.weights.header.tensors) {
      if (t.dtype === "q8" && Number(t.block) !== 32) throw new Error(`Gemma 4 WebGPU requires block-32 q8 tensors; ${t.name} declares block ${t.block}`);
      if (t.dtype === "q8" && (!Array.isArray(t.shape) || t.shape.length !== 2 || Number(t.shape[1]) % 32)) throw new Error(`Gemma 4 WebGPU requires q8 matrix columns divisible by 32; ${t.name} has shape ${JSON.stringify(t.shape)}`);
      const bytes = t.size + (t.scales_size || 0);
      if (this.tensors.get(t.name).hostTable) this.memory[cfg.hostStore ? "opfsWeights" : "hostWeights"] += bytes;
      else {
        this.memory.gpuWeights += bytes;
        this.checkBuffer(t.size, t.name);
        if (t.scales_size) this.checkBuffer(t.scales_size, `${t.name}.scales`);
      }
    }
    this.capacity = 0;
    this.uniforms = [];
    this.ropeMax = 0;
    this.shared = new Map();
    this.prefixKV = null;
    this.prefixCaptureOps = null;
    this.prefixSuffixOps = null;
    this.batchSuffixes = this.gpuKernel === "auto";
    this.fuseRms = this.gpuKernel === "auto";
    this.fuseAttentionSubgroups = this.gpuKernel === "auto";
    this.lastAttentionDispatches = 0;
    this.shareMatvecRows = this.gpuKernel === "auto";
    this.lastMatvecTwoDispatches = 0;
    this.batchMatmul = null;
    this.batchPart = null;
    this.fuseBatchPrefillGelu = this.gpuKernel === "auto";
    this.batchMlpCache = new Map();
    this.batchMlp = null;
    this.batchMlpPipelines = null;
    this.batchMlpScratchReady = false;
    this.batchSingleEnabled = this.gpuKernel === "auto";
    this.batchSingleCache = new Map();
    this.batchSingle = null;
    this.batchSinglePipelines = null;
    this.lastBatchSingleDispatches = 0;
    this.fuseGridMlp = true;
    this.gridMlp = null;
    this.gridMlpKey = null;
    this.gridMlpCache = new Gemma4GridMlpCache();
    this.lastGridMlpDispatches = 0;
    this.compactSourceRows = this.gpuKernel === "auto";
    this.sourceMatmul = null;
    this.lastSourceCompaction = { enabled: false };
    this.cachePleVectors = this.gpuKernel === "auto";
    this.pleVectorCache = null;
    this.activePleVectors = null;
    this.pleVectorSlots = null;
    this.lastPleVectorCache = { enabled: false };
    this.cacheTextPrefix = this.gpuKernel === "auto";
    this.retileTextPrefix = this.gpuKernel === "auto";
    this.narrowPrefixColumns = this.gpuKernel === "auto";
    this.fuseRmsNext = this.gpuKernel === "auto";
    this.fusePrefixGelu = this.gpuKernel === "auto";
    this.prefixRetileCache = new Map();
    this.prefixRetile = null;
    this.lastPrefixRetileDispatches = 0;
    this.lastPrefixBn32Dispatches = 0;
    this.lastPrefixGeluDispatches = 0;
    this.lastRmsNextDispatches = 0;
    this.pleReduceEnabled = true;
    this.lastPleReduceDispatches = 0;
    this.textPrefixCache = new Gemma4TextPrefixCache(this.device);
    this.textPrefixGraph = null;
    this.activeTextPrefixLease = null;
    this.lastTextPrefixCache = { enabled: false };
    this.lastPrefixReuse = null;
    this.tailCapacity = 0;
    this.tail = null;
  }

  write(dst, bytes) {
    if (this.gridMlpCache) this.gridMlpCache.clear();
    else this.gridMlp?.destroy();
    this.gridMlp = null;
    this.gridMlpKey = null;
    this.batchSingleCache.clear();
    this.batchSingle = null;
    this.clearPleVectorCache();
    this.clearTextPrefixCache();
    this.prefixRetileCache.clear();
    this.prefixRetile = null;
    this.weights.write(dst, bytes);
  }

  async flushWrites() { await this.cfg.hostStore?.flush(); }

  async finishWeights() { await this.cfg.hostStore?.finish(); }

  dispose() {
    if (this.gridMlpCache) this.gridMlpCache.clear();
    else this.gridMlp?.destroy();
    this.gridMlp = null;
    this.gridMlpKey = null;
    this.clearPleVectorCache();
    this.clearTextPrefixCache();
    this.textPrefixCache.dispose();
    this.prefixRetileCache.clear();
    this.prefixRetile = null;
    this.batchMlpCache.clear();
    this.batchMlp = null;
    this.batchMlpPipelines = null;
    this.batchSingleCache.clear();
    this.batchSingle = null;
    this.batchSinglePipelines = null;
    for (const entry of this.pleCache.values()) entry.buffer.destroy();
    this.pleCache.clear();
    this.pleCacheBytes = 0;
  }

  missing() {
    return this.weights.missing();
  }

  checkBuffer(size, label) {
    const limit = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    if (!Number.isSafeInteger(size) || size > limit) {
      throw Object.assign(new Error(`Gemma 4 WebGPU buffer ${label} needs ${Math.ceil(size / 1048576)} MiB; this device allows ${Math.floor(limit / 1048576)} MiB per buffer`), { code: "WEBGPU_INIT" });
    }
  }

  buffer(floats, extra = 0, label = "Gemma 4 scratch") {
    this.checkBuffer(floats * 4, label);
    return this.device.createBuffer({ label, size: Math.max(16, floats * 4), usage: U.STORAGE | extra });
  }

  staticBuffer(arr, label) {
    this.checkBuffer(arr.byteLength, label);
    const b = this.device.createBuffer({ label, size: Math.max(16, arr.byteLength), usage: U.STORAGE | U.COPY_DST });
    this.device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  uniform(values) {
    const slots = Math.max(1, Math.ceil(values.length / 4));
    const raw = new ArrayBuffer(slots * 16);
    const u = new Uint32Array(raw);
    const f = new Float32Array(raw);
    values.forEach((v, i) => {
      if (typeof v === "object") f[i] = v.f;
      else u[i] = v;
    });
    const b = this.device.createBuffer({ label: "Gemma 4 uniform", size: raw.byteLength, usage: U.UNIFORM | U.COPY_DST });
    this.device.queue.writeBuffer(b, 0, raw);
    this.uniforms.push(b);
    return b;
  }

  async init() {
    const miss = this.missing();
    if (miss.length) throw new Error(`Gemma 4 GPU trunk is missing ${miss.length} tensors (${miss[0]}...)`);
    const d = this.device;
    const generic = await matmulPipelines(d, this.wgsl);
    const built = await Promise.all(GEMMA4_KERNELS.map(async ([key, name]) => [key, await pipeline(d, this.wgsl(name, { subgroups: this.subgroup32 && (key.startsWith("RMS") || key === "PLE") }), name)]));
    this.p = Object.fromEntries(built);
    try {
      this.p.RMS_SCALE_NEXT = await withGpuErrors(this.runtime, () => pipeline(d,
        this.wgsl("gemma4_rms_scale_next", { subgroups: this.subgroup32 }), "gemma4_rms_scale_next"), "preparing Gemma RMS scale/next fusion");
    } catch (error) { this.rmsNextPipelineError = String(error?.message || error); }
    this.tuning = await calibrateMatmul(d, this.wgsl, this.weights, { kernel: this.gpuKernel, pipelines: { generic } });
    this.mm = this.tuning.pipelines.generic;
    this.pleReduce = await createGemma4PleReduce(d, this.wgsl);
    this.tuning.diagnostics.pleReduce = this.pleReduce.diagnostics;
    this.matvec = await calibrateMatvec(d, this.wgsl, this.weights, this.tuning, { kernel: this.gpuKernel });
    this.tuning.diagnostics.matvec = this.matvec.diagnostics;
    this.matvecGelu = await calibrateMatvecGelu(d, this.wgsl, this.weights, this.matvec, { gelu: this.p.GELU, enabled: this.gpuKernel === "auto" });
    this.tuning.diagnostics.matvecGelu = this.matvecGelu.diagnostics;
    this.matvecTwo = await calibrateMatvecTwo(d, this.wgsl, this.weights, {
      matvec: this.matvec, matvecGelu: this.matvecGelu, gelu: this.p.GELU,
      sharedStart: this.cfg.sharedStart, enabled: this.gpuKernel === "auto", subgroup32: this.subgroup32,
    });
    this.tuning.diagnostics.matvecTwo = this.matvecTwo.diagnostics;
    this.k64 = await calibrateGemma4K64(d, this.wgsl, this.weights, this.tuning, { kernel: this.gpuKernel });
    this.tuning.diagnostics.k64 = this.k64.diagnostics;
    this.mlp = await calibrateGemma4Mlp(d, this.wgsl, this.weights, this.tuning, { k64: this.k64, gelu: this.p.GELU, kernel: this.gpuKernel });
    this.tuning.diagnostics.mlp = this.mlp.diagnostics;
    this.tuning.diagnostics.batchMlp = { enabled: false, reason: "disabled", contexts: [] };
    this.tuning.diagnostics.batchSingle = { enabled: false, reason: "not-calibrated", contexts: [] };
    this.rmsValidation = await validateGemma4Rms(d, this.p, this.weights, this.cfg, { enabled: this.gpuKernel === "auto" });
    this.tuning.diagnostics.rmsFusion = this.rmsValidation.diagnostics;
    this.tuning.diagnostics.rmsScaleNext = this.rmsValidation.nextDiagnostics || { status: "disabled", reason: this.rmsNextPipelineError || "pipeline-unavailable" };
    this.attention = await calibrateGemma4Attention(this.runtime, this.wgsl, this.cfg, this.p.ATTN, { enabled: this.gpuKernel === "auto" });
    this.tuning.diagnostics.attention = this.attention.diagnostics;
    this.globals = d.createBuffer({ label: "Gemma 4 globals", size: 16, usage: U.UNIFORM | U.COPY_DST });
    // A zero bias and fallback scale buffer. Normal Gemma projection tensors are Q8;
    // the fallback keeps bind groups valid if a future converter emits an unscaled bias.
    this.zeros = this.staticBuffer(new Float32Array(16384), "Gemma 4 zero constants");
    await this.ensureRope(64);
    this.ensure(64);
    this.initialized = true;
  }

  async ensureRope(needed) {
    if (needed > this.cfg.maxPosition) throw new Error(`Gemma 4 sequence length ${needed} exceeds max_position_embeddings ${this.cfg.maxPosition}`);
    const max = Math.min(this.cfg.maxPosition, Math.max(64, pow2(needed)));
    if (max <= this.ropeMax) return;
    const pairs = 128;
    const data = new Float32Array(2 * max * pairs * 2);
    const writeTable = (table, headDim, rotary, theta) => {
      const n = rotary / 2;
      for (let pair = 0; pair < n; pair++) {
        const inv = Math.fround(1 / Math.fround(Math.pow(theta, (2 * pair) / headDim)));
        for (let pos = 0; pos < max; pos++) {
          const angle = Math.fround(pos * inv);
          const at = ((table * max + pos) * pairs + pair) * 2;
          data[at] = Math.cos(angle);
          data[at + 1] = Math.sin(angle);
        }
      }
      // Padded proportional-RoPE frequencies are zero, hence cos=1 and sin=0.
      for (let pos = 0; pos < max; pos++) for (let pair = n; pair < pairs; pair++) {
        const at = ((table * max + pos) * pairs + pair) * 2;
        data[at] = 1;
      }
    };
    writeTable(0, this.cfg.localHeadDim, this.cfg.localHeadDim, this.cfg.ropeLocalTheta);
    writeTable(1, this.cfg.globalHeadDim, Math.floor((this.cfg.globalHeadDim * this.cfg.ropeGlobalFraction) / 2) * 2, this.cfg.ropeGlobalTheta);
    const old = this.rope;
    this.rope = this.staticBuffer(data, "Gemma 4 RoPE table");
    this.ropeMax = max;
    if (old) old.destroy();
    if (this.capacity && this.p) this.build();
  }

  destroyDynamic() {
    this.clearPleVectorCache();
    this.clearTextPrefixCache();
    for (const b of this.dynamic || []) b.destroy();
    for (const b of this.uniforms || []) b.destroy();
    this.dynamic = [];
    this.uniforms = [];
    this.shared = new Map();
    this.prefixKV = null;
    this.tailBuffers = [];
    this.tailCapacity = 0;
    this.tail = null;
    this.ops = null;
    this.prefixOps = null;
    this.prefixCaptureOps = null;
    this.prefixSuffixOps = null;
    this.batchPrefixSuffixOps = null;
    this.batchTailOps = null;
    this.batchPart = null;
    this.batchMlpScratchReady = false;
    this.tailOps = null;
    this.sourcePrefixOps = null;
    this.sourcePrefixCaptureOps = null;
    this.sourcePrefixSuffixOps = null;
    this.sourceBatchPrefixSuffixOps = null;
    this.sourceTailOps = null;
    this.sourceBatchTailOps = null;
    this.createTextPrefixOps = null;
  }

  ensure(tokens) {
    if (tokens <= this.capacity) return;
    const cfg = this.cfg;
    const cap = Math.max(64, pow2(tokens));
    this.destroyDynamic();
    const d = this.device;
    this.dynamic = [];
    const make = (n, extra = 0, label = "Gemma 4 scratch") => {
      const b = this.buffer(n, extra, label);
      this.dynamic.push(b);
      return b;
    };
    const D = cfg.hidden;
    const Q = cfg.heads * cfg.globalHeadDim;
    const KV = cfg.kvHeads * cfg.globalHeadDim;
    const maxI = cfg.intermediate * (cfg.useDoubleWideMlp ? 2 : 1);
    const P = cfg.pleDim;
    this.x = make(cap * D, U.COPY_DST | U.COPY_SRC, "Gemma 4 hidden states");
    this.input = make(cap * D, U.COPY_DST | U.COPY_SRC, "Gemma 4 input embedding");
    this.h = make(cap * D, 0, "Gemma 4 normalized states");
    this.branch = make(cap * D, 0, "Gemma 4 residual branch");
    this.q = make(cap * Q, 0, "Gemma 4 queries");
    this.k = make(cap * KV, U.COPY_SRC | U.COPY_DST, "Gemma 4 keys");
    this.v = make(cap * KV, U.COPY_SRC | U.COPY_DST, "Gemma 4 values");
    this.suffixK = cfg.hostPle ? make(cap * KV, U.COPY_SRC | U.COPY_DST, "Gemma 4 suffix keys") : null;
    this.suffixV = cfg.hostPle ? make(cap * KV, U.COPY_SRC | U.COPY_DST, "Gemma 4 suffix values") : null;
    this.ctx = make(cap * Q, 0, "Gemma 4 attention context");
    this.gate = make(cap * maxI, 0, "Gemma 4 MLP gate");
    this.up = make(cap * maxI, 0, "Gemma 4 MLP up");
    this.act = make(cap * maxI, 0, "Gemma 4 MLP activation");
    this.pleToken = make(cap * P, U.COPY_SRC | (cfg.hostPle === true ? U.COPY_DST : 0), "Gemma 4 PLE token embedding");
    this.pleProj = make(cap * P, 0, "Gemma 4 PLE projection");
    this.ple = make(cap * P, U.COPY_SRC | U.COPY_DST, "Gemma 4 PLE input");
    this.pleGate = make(cap * P, 0, "Gemma 4 PLE gate");
    this.pleAct = make(cap * P, 0, "Gemma 4 PLE activation");
    this.ids = make(cap, U.COPY_DST, "Gemma 4 token ids");
    this.pleIds = make(cap, U.COPY_DST, "Gemma 4 PLE token ids");
    this.pos = make(2 * cap, U.COPY_DST, "Gemma 4 token positions and branch offsets");
    this.rows = make(cap, U.COPY_DST, "Gemma 4 output rows");
    this.gathered = make(cap * D, U.COPY_SRC, "Gemma 4 gathered rows");
    this.checkBuffer(cap * (D + 1) * 4, "Gemma 4 readback");
    this.readback = d.createBuffer({ label: "Gemma 4 readback", size: Math.max(16, cap * (D + 1) * 4), usage: U.MAP_READ | U.COPY_DST });
    this.dynamic.push(this.readback);
    this.part = make(SPLIT_SCRATCH, 0, "Gemma 4 matmul split scratch");
    // Keep one KV pair per layer type. Shared layers point their attention bind group
    // at the pair written by the last non-shared layer of that type.
    for (const type of ["sliding_attention", "full_attention"]) {
      this.shared.set(type, {
        k: make(cap * KV, U.COPY_DST, `Gemma 4 shared ${type} keys`),
        v: make(cap * KV, U.COPY_DST, `Gemma 4 shared ${type} values`),
      });
    }
    this.capacity = cap;
    if (this.p) this.build();
  }

  ensureTail(rows) {
    if (!this.cfg.kvSharedLayers || this.cfg.sharedStart <= 0 || rows <= this.tailCapacity) return;
    for (const b of this.tailBuffers || []) {
      const i = this.dynamic.indexOf(b);
      if (i >= 0) this.dynamic.splice(i, 1);
      b.destroy();
    }
    const cap = Math.max(64, pow2(rows));
    const d = this.device;
    const D = this.cfg.hidden;
    const Q = this.cfg.heads * this.cfg.globalHeadDim;
    const maxI = this.cfg.intermediate * (this.cfg.useDoubleWideMlp ? 2 : 1);
    const P = this.cfg.pleDim;
    const buffers = [];
    const make = (n, extra = 0, label = "Gemma 4 compact tail scratch") => {
      const b = this.buffer(n, extra, label);
      this.dynamic.push(b);
      buffers.push(b);
      return b;
    };
    const globals = d.createBuffer({ label: "Gemma 4 compact tail globals", size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.dynamic.push(globals);
    buffers.push(globals);
    const sourceGlobals = d.createBuffer({ label: "Gemma 4 compact source globals", size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.dynamic.push(sourceGlobals);
    buffers.push(sourceGlobals);
    this.tailBuffers = buffers;
    this.tailCapacity = cap;
    this.tail = {
      globals,
      sourceGlobals,
      x: make(cap * D, U.COPY_DST | U.COPY_SRC, "Gemma 4 compact hidden states"),
      input: make(cap * D, 0, "Gemma 4 compact input embedding"),
      h: make(cap * D, U.COPY_SRC, "Gemma 4 compact normalized states"),
      branch: make(cap * D, 0, "Gemma 4 compact residual branch"),
      q: make(cap * Q, 0, "Gemma 4 compact queries"),
      ctx: make(cap * Q, 0, "Gemma 4 compact attention context"),
      gate: make(cap * maxI, 0, "Gemma 4 compact MLP gate"),
      up: make(cap * maxI, 0, "Gemma 4 compact MLP up"),
      act: make(cap * maxI, 0, "Gemma 4 compact MLP activation"),
      pleToken: make(cap * P, this.cfg.hostPle === true ? U.COPY_DST : 0, "Gemma 4 compact PLE token embedding"),
      pleProj: make(cap * P, 0, "Gemma 4 compact PLE projection"),
      ple: make(cap * P, U.COPY_SRC | U.COPY_DST, "Gemma 4 compact PLE input"),
      pleGate: make(cap * P, 0, "Gemma 4 compact PLE gate"),
      pleAct: make(cap * P, 0, "Gemma 4 compact PLE activation"),
      ids: make(cap, U.COPY_DST, "Gemma 4 compact token ids"),
      pos: make(2 * cap, U.COPY_DST, "Gemma 4 compact token positions and branch offsets"),
      part: make(SPLIT_SCRATCH, 0, "Gemma 4 compact matmul split scratch"),
    };
    if (this.p) this.build();
  }

  weight(name) {
    const e = this.tensors.get(name);
    if (!e) throw new Error(`Gemma 4 GPU layout has no tensor ${name}`);
    return e;
  }

  clearPleVectorCache() {
    for (const entry of this.pleVectorCache?.entries.values() || []) entry.buffer.destroy();
    this.pleVectorCache = null;
    this.activePleVectors = null;
    this.pleVectorSlots = null;
  }

  clearTextPrefixGraph() {
    for (const buffer of this.textPrefixGraph?.uniforms || []) {
      const index = this.uniforms.indexOf(buffer);
      if (index >= 0) this.uniforms.splice(index, 1);
      buffer.destroy();
    }
    this.textPrefixGraph = null;
  }

  clearTextPrefixCache() {
    this.textPrefixCache?.invalidate();
    this.clearTextPrefixGraph();
  }

  textPrefixPolicy(compact, includePleReduce = true) {
    return JSON.stringify({
      compact,
      kernel: this.gpuKernel,
      matmul: [...this.tuning.selection],
      pipelines: Object.entries(this.tuning.pipelines).filter(([, pipes]) => pipes).map(([name, pipes]) => [name, pipes.f16, pipes.groups, pipes.splitTarget]),
      matvec: [...this.matvec.selection].sort(),
      matvecTwo: this.shareMatvecRows === true ? [[...this.matvecTwo.selection.plain].sort(), [...this.matvecTwo.selection.gelu].sort()] : [],
      k64: [...this.k64.selection].sort(),
      k64Enabled: this.useK64 !== false,
      rms: this.fuseRms === true && this.rmsValidation.enabled,
      rmsNext: this.fuseRmsNext === true && this.rmsValidation.nextEnabled,
      mlp: [...this.mlp.selection].sort(),
      prefillGelu: this.fusePrefillGelu !== false,
      subgroups: this.subgroup32,
      attention: this.fuseAttentionSubgroups === true ? [...this.attention.selection].sort() : [],
      prefixRetile: this.retileTextPrefix === true,
      prefixBn32: this.narrowPrefixColumns === true,
      prefixGelu: this.fusePrefixGelu === true,
      pleReduce: includePleReduce ? this.pleReduceEnabled === true && this.pleReduce?.diagnostics.enabled === true : undefined,
      ropeMax: this.ropeMax,
    });
  }

  preparePleVectorCache(signature, scopes) {
    this.activePleVectors = null;
    this.pleVectorSlots = null;
    this.lastPleVectorCache = { enabled: false, hits: 0, captures: 0, bytes: 0 };
    if (this.cachePleVectors !== true || this.profiler || !scopes.length) {
      this.clearPleVectorCache();
      this.lastPleVectorCache.reason = this.cachePleVectors !== true ? "disabled" : this.profiler ? "profiling" : "no-text-scopes";
      return;
    }
    const cfg = this.cfg;
    const bytes = scopes.reduce((sum, scope) => sum + scope.rows * (scope.end - scope.start) * cfg.pleDim * 4, 0);
    if (!Number.isSafeInteger(bytes) || bytes > PLE_VECTOR_CACHE_LIMIT || scopes.some((scope) => scope.rows * cfg.pleDim * 4 > this.device.limits.maxBufferSize)) {
      this.clearPleVectorCache();
      this.lastPleVectorCache.reason = "cache-limit";
      return;
    }
    const policy = {
      kernel: this.gpuKernel,
      matmul: [...this.tuning.selection],
      matvec: [...this.matvec.selection].sort(),
      k64: [...this.k64.selection].sort(),
      k64Enabled: this.useK64 !== false,
      rms: this.fuseRms,
      rmsNext: this.fuseRmsNext === true && this.rmsValidation.nextEnabled,
      matvecTwo: this.shareMatvecRows === true ? [[...this.matvecTwo.selection.plain].sort(), [...this.matvecTwo.selection.gelu].sort()] : [],
      matvecGelu: this.fuseMatvecGelu !== false,
      prefillGelu: this.fusePrefillGelu !== false,
      subgroups: this.subgroup32,
      config: [cfg.hidden, cfg.pleDim, cfg.eps, cfg.embeddingScale, cfg.pleEmbeddingScale, cfg.pleInputScale, cfg.pleProjectionScale, cfg.hostWord, cfg.hostPle],
    };
    const key = JSON.stringify([signature, scopes, policy]);
    if (this.pleVectorCache?.key !== key || this.pleVectorCache?.device !== this.device) {
      this.clearPleVectorCache();
      this.pleVectorCache = { key, device: this.device, entries: new Map(), scopes: new Map(scopes.map((scope) => [scope.slot, scope])), bytes: 0 };
    }
    this.activePleVectors = { cache: this.pleVectorCache, pending: new Set() };
    this.lastPleVectorCache = { enabled: true, hits: 0, captures: 0, bytes: this.pleVectorCache.bytes, capacityBytes: bytes };
  }

  pleVectorEntry(op, tokens) {
    const active = this.activePleVectors;
    const slot = this.pleVectorSlots?.[op.pleScope];
    const scope = active?.cache.scopes.get(slot);
    if (!scope || scope.rows !== tokens || op.layer < scope.start || op.layer >= scope.end) return null;
    const key = `${slot}:${op.layer}`;
    let entry = active.cache.entries.get(key);
    if (!entry) {
      const bytes = tokens * this.cfg.pleDim * 4;
      const buffer = this.device.createBuffer({ label: `Gemma 4 cached PLE ${key}`, size: Math.max(16, bytes), usage: U.COPY_SRC | U.COPY_DST });
      entry = { buffer, bytes, ready: false };
      active.cache.entries.set(key, entry);
      active.cache.bytes += bytes;
      this.lastPleVectorCache.bytes = active.cache.bytes;
    }
    return entry;
  }

  finishPleVectorCache(success) {
    if (success) for (const entry of this.activePleVectors?.pending || []) entry.ready = true;
    else if (this.activePleVectors) this.clearPleVectorCache();
    this.activePleVectors = null;
    this.pleVectorSlots = null;
  }

  async ensureSourcePipelines() {
    if (this.sourceMatmul) return;
    const names = Object.keys(this.tuning.pipelines).filter((name) => this.tuning.pipelines[name]);
    if (this.k64.pipelines) names.push("k64");
    this.sourceMatmul = Object.fromEntries(await Promise.all(names.map(async (name) => {
      const kernel = name === "wide" ? "matmul_wide" : name === "k64" ? "matmul_k64" : "matmul";
      return [name, await matmulPipelines(this.device, this.wgsl, `${kernel}_gemma_source`)];
    })));
  }

  async preparePrefixRetile(plan) {
    let result = null;
    if (this.retileTextPrefix === true && plan && plan.activeRows <= 64) {
      const cacheKey = `${plan.calibrationKey || plan.key}|bn32:${this.narrowPrefixColumns === true}|gelu:${this.fusePrefixGelu === true}`;
      result = this.prefixRetileCache.get(cacheKey);
      if (!result) {
        const calibrated = await calibrateGemma4PrefixRetile(this.device, this.weights, this.tuning, {
          plan, sourcePipelines: this.sourceMatmul,
          k64: this.useK64 !== false ? this.k64 : null,
          mlp: this.fusePrefillGelu !== false ? this.mlp : null,
        });
        const bn32 = await calibrateGemma4Bn32(this.device, this.wgsl, this.weights, {
          plan, retile: calibrated, sourcePipelines: this.sourceMatmul, enabled: this.narrowPrefixColumns === true,
        });
        const prefixGelu = await calibrateGemma4PrefixGelu(this.device, this.wgsl, this.weights, {
          plan, retile: calibrated, bn32, gelu: this.p.GELU, enabled: this.fusePrefixGelu === true,
        });
        result = { ...calibrated, key: plan.key, bn32, prefixGelu,
          diagnostics: { ...calibrated.diagnostics, bn32: bn32.diagnostics, prefixGelu: prefixGelu.diagnostics } };
        if (this.prefixRetileCache.size >= 2) this.prefixRetileCache.delete(this.prefixRetileCache.keys().next().value);
        this.prefixRetileCache.set(cacheKey, result);
      }
      if (result.key !== plan.key) {
        result = { ...result, key: plan.key };
        this.prefixRetileCache.set(cacheKey, result);
      }
    }
    if (this.prefixRetile !== result) {
      this.clearTextPrefixGraph();
      this.prefixRetile = result;
    }
    this.tuning.diagnostics.prefixRetile = { enabled: !!result?.selection.size,
      reason: result?.diagnostics.reason || (this.retileTextPrefix === true ? "ineligible" : "disabled"),
      contexts: [...this.prefixRetileCache.values()].map((entry) => entry.diagnostics) };
  }

  encodeSourceMatmul(pass, op, tokens, plePartner = null) {
    const original = this.sourceTokens;
    if (this.retileTextPrefix === true && op.logicalRows && op.prefixRetile) {
      const config = op.prefixRetile;
      const pipes = this.sourceMatmul[config.kernel];
      if (this.narrowPrefixColumns === true && config.bn32) {
        const splits = mmSplits(config.splitTokens, config.N, config.K, pipes.splitTarget, 32, 64);
        const ple = plePartner && gemma4PleReduceChoice(this.pleReduce, op, plePartner, tokens, pipes, 2, config.splitTokens, "source", true);
        pass.setPipeline(config.bn32);
        pass.setBindGroup(0, config.group);
        pass.dispatchWorkgroups(Math.ceil(config.N / 32), Math.ceil(tokens / 32), splits);
        if (splits > 1) {
          const work = Math.ceil(tokens * config.N / 256);
          pass.setPipeline(ple?.pipeline || pipes.reduce[2]);
          pass.setBindGroup(0, ple?.group || config.reduce);
          pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
        }
        this.lastPrefixBn32Dispatches++;
        this.lastPrefixRetileDispatches++;
        return !!ple;
      }
      const ple = plePartner && gemma4PleReduceChoice(this.pleReduce, op, plePartner, tokens, pipes, config.rows, config.splitTokens, "source", true);
      dispatchMatmul(pass, config.row56 ? pipes.mm56 : pipes.mm[config.rows], pipes.reduce[config.rows], config,
        tokens, pipes.splitTarget, config.rows, pipes.groups, tokens, config.splitTokens, ple);
      this.lastPrefixRetileDispatches++;
      return !!ple;
    }
    if ((original === 1 || op.batched && original <= 8) && op.dtype === "q8" && this.matvec.selection.has(`${op.N}:${op.K}`)) {
      encodeMatvec(pass, this.matvec.pipelines, op, tokens);
      return false;
    }
    const selected = this.gpuKernel === "auto" ? selectedMatmulKernel(this.tuning.selection, op.N, op.K, op.batched ? 1 : original) : this.gpuKernel;
    const k64 = this.useK64 !== false && !op.batched && op.dtype === "q8" && gemma4K64Eligible(original) && this.k64.selection.has(`${op.N}:${op.K}`);
    const kind = k64 ? "k64" : this.sourceMatmul[selected] ? selected : "generic";
    const baseline = k64 ? this.k64.pipelines : this.tuning.pipelines[kind];
    const pipes = this.sourceMatmul[kind];
    const rows = rowsPerThread(original, baseline);
    const ple = plePartner && gemma4PleReduceChoice(this.pleReduce, op, plePartner, tokens, pipes, rows, op.batched ? 1 : original, "source");
    const row56 = rows === 4 && baseline.f16 === false && baseline.groups === 1 && Math.ceil(original / 56) === Math.ceil(original / 64);
    if (row56 && pipes.mm56) {
      const splits = mmSplits(op.batched ? 1 : original, op.N, op.K, pipes.splitTarget, 64, pipes.groups * 64);
      pass.setPipeline(pipes.mm56);
      pass.setBindGroup(0, op.group);
      pass.dispatchWorkgroups(Math.ceil(op.N / (pipes.groups * 64)), Math.ceil(tokens / 56), splits);
      if (splits > 1) {
        const work = Math.ceil(tokens * op.N / 256);
        pass.setPipeline(ple?.pipeline || pipes.reduce[rows]);
        pass.setBindGroup(0, ple?.group || op.reduce);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
      return !!ple;
    }
    dispatchMatmul(pass, (row56 && pipes.mm56) || pipes.mm[rows], pipes.reduce[rows], op, tokens, pipes.splitTarget, rows, pipes.groups, tokens, op.batched ? 1 : original, ple);
    return !!ple;
  }

  ensureBatchPart(tokens) {
    let floats = 0;
    for (const op of this.batchPrefixSuffixOps) {
      if (op.kind !== "mm") continue;
      const splits = mmSplits(1, op.N, op.K, 256, 16, 64);
      if (splits > 1) floats = Math.max(floats, splits * tokens * op.N);
    }
    const bytes = floats * 4;
    const limit = Math.min(64 * 1024 * 1024, this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize);
    if (!Number.isSafeInteger(bytes) || bytes > limit) return false;
    if (bytes <= (this.batchPart || this.part).size) return true;
    const previous = this.batchPart;
    this.batchPart = this.buffer(floats, 0, "Gemma 4 batched matmul split scratch");
    this.dynamic.push(this.batchPart);
    this.build();
    if (previous) {
      this.dynamic.splice(this.dynamic.indexOf(previous), 1);
      previous.destroy();
    }
    return true;
  }

  async prepareBatchMlp(tokens, compact = false) {
    let result = null;
    let reason = this.fuseBatchPrefillGelu === true ? "unsupported-physical-tokens" : "disabled";
    if (this.fuseBatchPrefillGelu === true && gemma4BatchMlpEligible(tokens)) {
      const ops = compact ? this.sourceBatchPrefixSuffixOps : this.batchPrefixSuffixOps;
      const eligibleShapes = new Set((ops || []).filter((op) => op.kind === "mm" && op.label === "mm.gate" && !op.sourcePolicy).map((op) => `${op.N}:${op.K}`));
      const key = `${tokens}|${[...eligibleShapes].sort().join(",")}`;
      result = this.batchMlpCache.get(key);
      if (!result) {
        result = await calibrateGemma4BatchMlp(this.device, this.wgsl, this.weights, this.tuning, {
          tokens, batchPipelines: this.batchMatmul, gelu: this.p.GELU, kernel: this.gpuKernel, pipelines: this.batchMlpPipelines, eligibleShapes,
        });
        this.batchMlpPipelines ||= result.pipelines;
        if (this.batchMlpCache.size >= 2) this.batchMlpCache.delete(this.batchMlpCache.keys().next().value);
        this.batchMlpCache.set(key, result);
      }
      reason = result.diagnostics.reason;
    }
    const selected = result?.selection.size ? result : null;
    if (this.batchMlp !== selected) {
      this.batchMlp = selected;
      this.batchMlpScratchReady = false;
      if (this.capacity && this.p) this.build();
    }
    this.tuning.diagnostics.batchMlp = { enabled: !!selected, tokens, reason, scratchReady: this.batchMlpScratchReady,
      contexts: [...this.batchMlpCache.values()].map((entry) => entry.diagnostics) };
    return selected;
  }

  async prepareBatchSingle(tokens, compact = false) {
    let result = null;
    if (gemma4BatchSingleEligible(tokens)) {
      const ops = compact ? this.sourceBatchPrefixSuffixOps : this.batchPrefixSuffixOps;
      const eligibleShapes = new Set((ops || []).filter((op) => gemma4BatchSingleEligible(tokens, op)).map((op) => `${op.N}:${op.K}`));
      const policy = [...eligibleShapes].sort().map((key) => {
        const [N, K] = key.split(":").map(Number), selected = selectedMatmulKernel(this.tuning.selection, N, K, 1);
        const pipes = this.batchMatmul[selected] || this.batchMatmul.generic;
        return [key, selected, pipes?.f16, pipes?.groups, pipes?.splitTarget];
      });
      const key = JSON.stringify([tokens, this.gpuKernel, policy]);
      result = this.batchSingleCache.get(key);
      if (!result) {
        result = await calibrateGemma4BatchSingle(this.device, this.wgsl, this.weights, this.tuning, {
          tokens, batchPipelines: this.batchMatmul, eligibleShapes, kernel: this.gpuKernel, pipelines: this.batchSinglePipelines,
        });
        this.batchSinglePipelines ||= result.pipelines;
        if (this.batchSingleCache.size >= 2) this.batchSingleCache.delete(this.batchSingleCache.keys().next().value);
        this.batchSingleCache.set(key, result);
      }
    }
    this.batchSingle = result;
    this.tuning.diagnostics.batchSingle = { enabled: this.batchSingleEnabled === true && !!result?.selection.size, tokens,
      reason: result?.diagnostics.reason || "unsupported-physical-tokens", contexts: [...this.batchSingleCache.values()].map((entry) => entry.diagnostics) };
    return result;
  }

  async prepareGridMlp(tokens, compact = false) {
    if (gemma4GridMlpTokens(tokens)) {
      const selected = this.gpuKernel === "auto" ? selectedMatmulKernel(this.tuning.selection, 6144, 1536, 1) : this.gpuKernel;
      const baseline = this.batchMatmul[selected] || this.batchMatmul.generic;
      const pairedSelected = this.batchMlp?.selection.has("6144:1536") === true;
      const ops = compact ? this.sourceBatchPrefixSuffixOps : this.batchPrefixSuffixOps;
      const pair = (ops || []).find((op, i) => gemma4GridMlpEligible(tokens, op, ops[i + 1], ops[i + 2], baseline));
      const key = JSON.stringify([tokens, this.gpuKernel, selected, baseline?.kernel, baseline?.f16, baseline?.groups,
        baseline?.splitTarget, !!baseline?.mm56, pairedSelected, pair?.gridMlpSource.gate.info.name, pair?.gridMlpSource.up.info.name]);
      if (this.gridMlpKey !== key) {
        this.gridMlpCache ||= new Gemma4GridMlpCache();
        const result = await this.gridMlpCache.get(key, tokens, () => calibrateGemma4GridMlp(this.device, this.wgsl, pair?.gridMlpSource, {
          tokens, baseline, gelu: this.p.GELU, automatic: this.gpuKernel === "auto", pairedSelected, runtime: this.runtime,
        }));
        this.gridMlp = result;
        this.gridMlpKey = key;
      }
    }
    this.tuning.diagnostics.gridMlp = { enabled: this.fuseGridMlp === true && this.gridMlp?.tokens === tokens && !!this.gridMlp?.selection.size,
      tokens, context: this.gridMlp?.diagnostics || null };
    return this.gridMlp;
  }

  async ensureBatchMlpPart(tokens, compact = false) {
    this.batchMlpScratchReady = false;
    if (this.fuseBatchPrefillGelu !== true || !gemma4BatchMlpEligible(tokens) || this.batchMlp?.tokens !== tokens) return false;
    let bytes = 0;
    for (const op of (compact ? this.sourceBatchPrefixSuffixOps : this.batchPrefixSuffixOps) || []) {
      if (!op.batchMlpFused || op.sourcePolicy || !this.batchMlp.selection.has(`${op.N}:${op.K}`)) continue;
      bytes = Math.max(bytes, gemma4BatchMlpPartBytes(tokens, op.N, op.K));
    }
    const diagnostics = this.tuning.diagnostics.batchMlp;
    const limit = Math.min(64 * 1024 * 1024, this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize);
    if (!bytes || !Number.isSafeInteger(bytes) || bytes > limit) {
      diagnostics.scratchReady = false;
      diagnostics.scratchReason = bytes ? "scratch-limit" : "no-selected-batch-ops";
      return false;
    }
    if (bytes > (this.batchPart || this.part).size) {
      let replacement;
      try {
        await inErrorScopes(this.device, () => { replacement = this.buffer(bytes / 4, 0, "Gemma 4 paired batch MLP split scratch"); });
      } catch (error) {
        replacement?.destroy();
        diagnostics.scratchReady = false;
        diagnostics.scratchReason = "allocation-failed";
        diagnostics.scratchError = String(error?.message || error);
        return false;
      }
      const previous = this.batchPart;
      this.batchPart = replacement;
      this.dynamic.push(replacement);
      this.build();
      if (previous) {
        this.dynamic.splice(this.dynamic.indexOf(previous), 1);
        previous.destroy();
      }
    }
    this.batchMlpScratchReady = true;
    diagnostics.scratchReady = true;
    diagnostics.scratchBytes = bytes;
    delete diagnostics.scratchReason;
    delete diagnostics.scratchError;
    return true;
  }

  build() {
    this.clearPleVectorCache();
    this.clearTextPrefixCache();
    for (const b of this.uniforms) b.destroy();
    this.uniforms = [];
    const d = this.device;
    const cfg = this.cfg;
    const resource = (buffer) => buffer.buffer ? buffer : { buffer };
    const entries = (resources) => resources.map((buffer, binding) => ({ binding, resource: resource(buffer) }));
    const bg = (p, resources) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries(resources) });
    const view = (plan, buffer, width) => {
      const result = prefixRowView(plan, buffer, width);
      if (!result) throw new Error(`Gemma 4 static prefix has an invalid ${width}-wide row view`);
      return result;
    };
    const custom = (target, name, values, resources, label = name, dispatch) => {
      const bindings = [target.globals, this.uniform(values), ...resources];
      const op = { kind: name, label, pipeline: this.p[name], group: bg(this.p[name], bindings), dispatch };
      if (name === "ATTN" && this.attention?.pipeline && this.attention.selection.size) {
        op.attentionDim = values[2];
        op.subgroup = d.createBindGroup({ layout: this.attention.layout, entries: entries(bindings) });
      }
      return op;
    };
    const mm = (target, name, input, output, mode = 0, label = `mm.${name.split(".").at(-1)}`) => {
      const e = this.weight(name);
      const [N, K] = e.info.shape.map(Number);
      const params = this.uniform([N, K, mode, 0]);
      const scales = e.sbuf || this.zeros;
      const op = {
        kind: "mm",
        label,
        N,
        K,
        dtype: e.info.dtype,
        mode,
        bias: 0,
        batched: target.batched === true,
        sourcePolicy: target.sourcePolicy === true,
        logicalRows: target.logicalRows,
        params,
        group: d.createBindGroup({ layout: this.mm.layout, entries: entries([target.globals, params, input, e.buf, scales, this.zeros, output, target.part]) }),
        reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: entries([target.globals, params, target.part, this.zeros, output]) }),
      };
      const config = mode === 0 && e.info.dtype === "q8" && target.prefixPlan && target.prefixPlan.key === this.prefixRetile?.key
        ? this.prefixRetile.selection.get(`${N}:${K}`) : null;
      if (config) {
        const globals = this.uniform([target.prefixPlan.activeRows, config.splitTokens, target.logicalRows, 0]);
        op.prefixRetile = { ...config, N, K, globals,
          bn32: this.prefixRetile.bn32?.selection.has(`${N}:${K}`) ? this.prefixRetile.bn32.pipeline : null,
          group: d.createBindGroup({ layout: this.mm.layout, entries: entries([globals, params, input, e.buf, scales, this.zeros, output, target.part]) }),
          reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: entries([globals, params, target.part, this.zeros, output]) }) };
      }
      return op;
    };
    const rms = (target, name, input, output, width = cfg.hidden, mode = 1, label = `rms.${name.split(".").at(-1)}`) => {
      const op = custom(target, "RMS", [width, f32(cfg.eps), mode, 0], [input, this.weight(name).buf, output], label, (pass) => pass.dispatchWorkgroups(this.T));
      op.rmsInput = { name, input, output, width, mode, eps: cfg.eps, globals: target.globals };
      return op;
    };
    const scale = (target, name, label = "layer.scalar") => custom(target, "RESIDUAL", [cfg.hidden, 1, 0, 0], [target.x, this.zeros, this.weight(name).buf], label, (pass) => pass.dispatchWorkgroups(this.T));
    const appendLayer = (ops, target, i, compact, { capturePrefix = false, joinPrefix = false, kvOnly = false } = {}) => {
      const type = cfg.full[i] ? "full_attention" : "sliding_attention";
      const headDim = cfg.full[i] ? cfg.globalHeadDim : cfg.localHeadDim;
      const kvHeads = cfg.kvHeads;
      const shared = compact || i >= cfg.sharedStart;
      const kvWidth = kvHeads * headDim;
      const rowPlan = target.prefixPlan;
      if (rowPlan) {
        const intermediate = Number(this.weight(`l.${i}.gate`).info.shape[0]);
        const widths = { x: cfg.hidden, input: cfg.hidden, h: cfg.hidden, branch: cfg.hidden,
          q: cfg.heads * headDim, ctx: cfg.heads * headDim, gate: intermediate, up: intermediate, act: intermediate,
          pleToken: cfg.pleDim, pleProj: cfg.pleDim, ple: cfg.pleDim, pleGate: cfg.pleDim, pleAct: cfg.pleDim };
        target = { ...target, ...Object.fromEntries(Object.entries(widths).map(([name, width]) => [name, view(rowPlan, target[name], width)])),
          qkvK: view(rowPlan, target.k, kvWidth), qkvV: view(rowPlan, target.v, kvWidth) };
      }
      const rawK = target.qkvK || target.k || this.k;
      const rawV = target.qkvV || target.v || this.v;
      const rotary = cfg.full[i] ? Math.floor((cfg.globalHeadDim * cfg.ropeGlobalFraction) / 2) * 2 : cfg.localHeadDim;
      const n = (s) => `l.${i}.${s}`;
      const store = cfg.kvSharedLayers > 0 && !compact && !shared && cfg.layerTypes.slice(0, cfg.sharedStart).findIndex((t) => t === cfg.layerTypes[i]) >= 0 && cfg.layerTypes.slice(0, cfg.sharedStart).map(String).lastIndexOf(cfg.layerTypes[i]) === i;
      // PLE projection uses the image-injected input embedding, then is
      // combined immediately before this layer consumes it.
      if (!kvOnly) {
        const projection = mm(target, `ple.${i}.proj`, target.input, target.pleProj, 0, "mm.ple.proj");
        if (target.pleScope) Object.assign(projection, { pleScope: target.pleScope, layer: i, pleDst: target.ple });
        ops.push(projection);
        const pleEmbed = this.weight(`ple.${i}.embed`);
        if (pleEmbed.hostTable) ops.push({ kind: "copy-ple", label: "ple.embed.host", layer: i, dst: target.pleToken });
        else ops.push(custom(target, "EMBED", [cfg.pleDim, Math.ceil(cfg.pleDim / 32), f32(cfg.pleEmbeddingScale), 0], [target.pleIds || target.ids, pleEmbed.buf, pleEmbed.sbuf || this.zeros, target.pleToken], "ple.embed", (pass) => pass.dispatchWorkgroups(this.T)));
        ops.push(custom(target, "PLE", [cfg.pleDim, cfg.hidden, f32(cfg.eps), f32(cfg.pleInputScale), f32(cfg.pleProjectionScale)], [target.pleToken, target.pleProj, this.weight("ple.norm").buf, target.ple], "ple.combine", (pass) => pass.dispatchWorkgroups(this.T)));
        if (target.pleScope) ops.push({ kind: "copy-ple-vector", pleScope: target.pleScope, layer: i, src: target.ple });
      }
      ops.push(rms(target, n("attn_norm"), target.x, target.h, cfg.hidden, 1, "rms.attn"));
      if (!kvOnly) ops.push(mm(target, n("q"), target.h, target.q, 0, "mm.q"));
      if (!shared) {
        ops.push(mm(target, n("k"), target.h, rawK, 0, "mm.k"));
        ops.push(mm(target, n("v"), target.h, rawV, 0, "mm.v"));
      }
      const qn = this.weight(n("qn")).buf;
      const kn = shared ? this.zeros : this.weight(n("kn")).buf;
      // A compact tail computes Q for selected rows only. K/V are already in the
      // full-length shared pair, so the unused K/V groups must remain disabled.
      const queryHeads = kvOnly ? 0 : cfg.heads;
      ops.push(custom(target, "QKV", [queryHeads, kvHeads, headDim, rotary, cfg.full[i] ? 1 : 0, this.ropeMax * 128, shared ? 0 : 1, 0, f32(cfg.eps)], [target.q, rawK, rawV, qn, kn, this.rope, target.pos], "qkv", (pass) => pass.dispatchWorkgroups(this.T, queryHeads + (shared ? 0 : 2 * kvHeads))));
      if (joinPrefix && !shared) ops.push({ kind: "copy-kv-join", label: `kv.join.${i}`, layer: i, width: kvWidth });
      if (capturePrefix && !shared) ops.push({ kind: "copy-kv-prefix", label: `kv.prefix.${i}`, layer: i, width: kvWidth, staticRows: rowPlan?.staticRows, totalRows: rowPlan?.logicalRows });
      if (store) {
        const pair = this.shared.get(type);
        ops.push({ kind: "copy-kv", label: `kv.share.${type}`, srcK: target.k, srcV: target.v, dstK: pair.k, dstV: pair.v, width: kvWidth, total: joinPrefix, totalRows: rowPlan?.logicalRows });
      }
      if (kvOnly) return;
      const pair = shared ? this.shared.get(type) : { k: target.k, v: target.v };
      ops.push(custom(target, "ATTN", [cfg.heads, kvHeads, headDim, cfg.full[i] ? 0 : cfg.window, cfg.causal, Number(target.batched === true), 0, 0], [target.q, pair.k, pair.v, target.ctx, target.pos], "attn", (pass) => pass.dispatchWorkgroups(this.T, cfg.heads)));
      ops.push(mm(target, n("o"), target.ctx, target.branch, 0, "mm.o"));
      const postAttention = rms(target, n("attn_post_norm"), target.branch, target.x, cfg.hidden, 2, "rms.attn.post");
      postAttention.rmsFusion = custom(target, "RMS_RESIDUAL_NORM", [cfg.hidden, f32(cfg.eps), 2, 0], [target.branch, this.weight(n("attn_post_norm")).buf, target.x, this.weight(n("ffn_norm")).buf, target.h], "rms.attn.ffn.fused", (pass) => pass.dispatchWorkgroups(this.T));
      postAttention.rmsPartner = rms(target, n("ffn_norm"), target.x, target.h, cfg.hidden, 1, "rms.ffn");
      ops.push(postAttention);
      ops.push(postAttention.rmsPartner);
      const gateOp = mm(target, n("gate"), target.h, target.gate, 0, "mm.gate");
      ops.push(gateOp);
      const upOp = mm(target, n("up"), target.h, target.up, 0, "mm.up");
      ops.push(upOp);
      const I = Number(this.weight(n("gate")).info.shape[0]);
      ops.push(custom(target, "GELU", [I, 0, 0, 0], [target.gate, target.up, target.act], "gelu", (pass) => {
        const work = Math.ceil((this.T * I) / 256);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }));
      gateOp.mlpPartners = [upOp, ops.at(-1)];
      const gateWeight = this.weight(n("gate"));
      const upWeight = this.weight(n("up"));
      if (target.batched === true && target.sourcePolicy !== true) {
        gateOp.gridMlpSource = { globals: target.globals, input: target.h, gate: gateWeight, up: upWeight, output: target.act };
      }
      const prefixConfig = gateOp.prefixRetile;
      if (rowPlan && prefixConfig && this.prefixRetile.prefixGelu?.selection.has(`${gateOp.N}:${gateOp.K}`) &&
          gateWeight.info.dtype === "q8" && upWeight.info.dtype === "q8" && upWeight.info.shape[0] === gateOp.N && upWeight.info.shape[1] === gateOp.K) {
        const splits = mmSplits(prefixConfig.splitTokens, gateOp.N, gateOp.K, 256, 32, 64);
        if (splits === 1 || 2 * splits * rowPlan.activeRows * gateOp.N * 4 <= target.part.size) {
          const pipes = this.prefixRetile.prefixGelu.pipelines;
          const globals = this.uniform([rowPlan.activeRows, prefixConfig.splitTokens, rowPlan.logicalRows, 0]);
          const params = this.uniform([gateOp.N, gateOp.K, prefixConfig.splitTokens, 0]);
          gateOp.prefixGelu = { N: gateOp.N, K: gateOp.K, splitTokens: prefixConfig.splitTokens, pipelines: pipes,
            group: d.createBindGroup({ layout: pipes.layout, entries: entries([globals, params, target.h, gateWeight.buf, gateWeight.sbuf, upWeight.buf, upWeight.sbuf, target.act, target.part]) }),
            reduce: d.createBindGroup({ layout: pipes.reduceLayout, entries: entries([globals, params, target.part, target.act]) }) };
        }
      }
      if (!rowPlan && this.matvecTwo?.selection.gelu.has(`${gateOp.N}:${gateOp.K}`) && gateWeight.info.dtype === "q8" && upWeight.info.dtype === "q8" && upWeight.info.shape[0] === gateOp.N && upWeight.info.shape[1] === gateOp.K) {
        gateOp.twoFused = { N: gateOp.N, K: gateOp.K, group: matvecGeluBindGroup(d, this.matvecTwo.pipelines.gelu, {
          globals: target.globals,
          params: this.uniform([gateOp.N, gateOp.K, 0, 0]),
          input: target.h,
          gate: gateWeight.buf,
          gateScales: gateWeight.sbuf,
          up: upWeight.buf,
          upScales: upWeight.sbuf,
          output: target.act,
        }) };
      }
      if (target.batched === true && target.sourcePolicy !== true && this.batchMlp?.selection.has(`${gateOp.N}:${gateOp.K}`) && gateWeight.info.dtype === "q8" && upWeight.info.dtype === "q8" && upWeight.info.shape[0] === gateOp.N && upWeight.info.shape[1] === gateOp.K) {
        gateOp.batchMlpFused = { N: gateOp.N, K: gateOp.K, batched: true, ...gemma4BatchMlpBindGroups(d, this.batchMlp.pipelines, {
          globals: target.globals,
          params: this.uniform([gateOp.N, gateOp.K, 1, 0]),
          input: target.h,
          gate: gateWeight.buf,
          gateScales: gateWeight.sbuf,
          up: upWeight.buf,
          upScales: upWeight.sbuf,
          output: target.act,
          partials: target.part,
        }) };
      }
      if (this.mlp.selection.has(`${gateOp.N}:${gateOp.K}`) && gateWeight.info.dtype === "q8" && upWeight.info.dtype === "q8" && upWeight.info.shape[0] === gateOp.N && upWeight.info.shape[1] === gateOp.K) {
        const params = this.uniform([gateOp.N, gateOp.K, target.logicalRows || 0, 0]);
        gateOp.prefillFused = { N: gateOp.N, group: rowPlan ? bg(this.mlp.pipelines.pipeline, [target.globals, params, target.h, gateWeight.buf, gateWeight.sbuf, upWeight.buf, upWeight.sbuf, target.act, this.part]) : gemma4MlpBindGroup(d, this.mlp.pipelines, {
          globals: target.globals,
          params,
          input: target.h,
          gate: gateWeight.buf,
          gateScales: gateWeight.sbuf,
          up: upWeight.buf,
          upScales: upWeight.sbuf,
          output: target.act,
          partials: this.part,
        }) };
      }
      if (!rowPlan && this.matvecGelu.selection.has(`${gateOp.N}:${gateOp.K}`) && gateWeight.info.dtype === "q8" && upWeight.info.dtype === "q8" && upWeight.info.shape[0] === gateOp.N && upWeight.info.shape[1] === gateOp.K) {
        gateOp.fused = { N: gateOp.N, group: matvecGeluBindGroup(d, this.matvecGelu.pipelines, {
          globals: target.globals,
          params: this.uniform([gateOp.N, gateOp.K, 0, 0]),
          input: target.h,
          gate: gateWeight.buf,
          gateScales: gateWeight.sbuf,
          up: upWeight.buf,
          upScales: upWeight.sbuf,
          output: target.act,
        }) };
      }
      ops.push(mm(target, n("down"), target.act, target.branch, 0, "mm.down"));
      ops.push(rms(target, n("ffn_post_norm"), target.branch, target.x, cfg.hidden, 2, "rms.ffn.post"));
      const pleGate = mm(target, n("ple_gate"), target.x, target.pleGate, 0, "mm.ple.gate");
      const pleGelu = custom(target, "GELU", [cfg.pleDim, 1, 0, 0], [target.pleGate, target.ple, target.pleAct], "gelu.ple", (pass) => {
        const work = Math.ceil((this.T * cfg.pleDim) / 256);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      });
      if (!compact && i < cfg.sharedStart && this.pleReduce?.layout) {
        const resources = { params: pleGate.params, partials: target.part, bias: this.zeros, raw: target.pleGate, ple: target.ple, output: target.pleAct };
        pleGate.pleReduce = { N: pleGate.N, K: pleGate.K, partner: pleGelu,
          group: gemma4PleReduceGroup(d, this.pleReduce, { ...resources, globals: target.globals }),
          retiledGroup: pleGate.prefixRetile ? gemma4PleReduceGroup(d, this.pleReduce, { ...resources, globals: pleGate.prefixRetile.globals }) : null };
      }
      ops.push(pleGate, pleGelu);
      ops.push(mm(target, n("ple_out"), target.pleAct, target.branch, 0, "mm.ple.out"));
      const postPle = rms(target, n("ple_norm"), target.branch, target.x, cfg.hidden, 2, "rms.ple");
      postPle.rmsFusion = custom(target, "RMS_SCALE", [cfg.hidden, f32(cfg.eps), 2, 0], [target.branch, this.weight(n("ple_norm")).buf, target.x, this.weight(n("scalar")).buf], "rms.ple.scale.fused", (pass) => pass.dispatchWorkgroups(this.T));
      if (this.p.RMS_SCALE_NEXT) {
        const nextName = i + 1 < cfg.layers ? `l.${i + 1}.attn_norm` : "norm";
        postPle.rmsScaleNext = { nextName, width: cfg.hidden, eps: cfg.eps, globals: target.globals, residual: target.x, output: target.h,
          fusion: custom(target, "RMS_SCALE_NEXT", [cfg.hidden, f32(cfg.eps), 2, 0],
            [target.branch, this.weight(n("ple_norm")).buf, target.x, this.weight(n("scalar")).buf, this.weight(nextName).buf, target.h],
            "rms.ple.scale.next.fused", (pass) => pass.dispatchWorkgroups(this.T)) };
      }
      ops.push(postPle);
      postPle.rmsPartner = scale(target, n("scalar"), "scale");
      ops.push(postPle.rmsPartner);
    };
    const full = {
      globals: this.globals,
      part: this.part,
      x: this.x,
      input: this.input,
      h: this.h,
      branch: this.branch,
      q: this.q,
      k: this.k,
      v: this.v,
      ctx: this.ctx,
      gate: this.gate,
      up: this.up,
      act: this.act,
      pleToken: this.pleToken,
      pleProj: this.pleProj,
      ple: this.ple,
      pleGate: this.pleGate,
      pleAct: this.pleAct,
      ids: this.ids,
      pleIds: this.pleIds,
      pos: this.pos,
    };
    const ops = [];
    const embed = this.weight("embed");
    const embedWidth = Number(embed.info.shape[1]);
    if (cfg.hostWord) ops.push({ kind: "copy-embed", label: "embed.stored" });
    else ops.push(custom(full, "EMBED", [embedWidth, Math.ceil(embedWidth / 32), f32(cfg.embeddingScale), 0], [full.ids, embed.buf, embed.sbuf || this.zeros, full.x], "embed", (pass) => pass.dispatchWorkgroups(this.T)));
    ops.push({ kind: "copy-images", label: "embed.images" });
    ops.push({ kind: "copy-input", label: "embed.copy", src: this.x, dst: this.input });
    let prefixOps = null;
    const canSharePrefix = cfg.hostPle === true && cfg.kvSharedLayers > 0 && cfg.sharedStart > 0;
    const prefixCaptureOps = canSharePrefix ? ops.slice() : null;
    const suffixTarget = canSharePrefix ? { ...full, qkvK: this.suffixK, qkvV: this.suffixV, pleScope: "suffix" } : null;
    const prefixSuffixOps = canSharePrefix && this.tail ? ops.slice() : null;
    const batchPrefixSuffixOps = prefixSuffixOps ? ops.slice() : null;
    const sourceLayer = cfg.kvSharedLayers > 0 ? cfg.sharedStart - 1 : -1;
    let sourcePrefixOps = null;
    let sourcePrefixCaptureOps = null;
    let sourcePrefixSuffixOps = null;
    let sourceBatchPrefixSuffixOps = null;
    for (let i = 0; i < cfg.layers; i++) {
      if (i === cfg.sharedStart) prefixOps = ops.slice();
      if (i === sourceLayer) {
        sourcePrefixOps = ops.slice();
        appendLayer(sourcePrefixOps, full, i, false, { kvOnly: true });
        if (prefixCaptureOps) {
          sourcePrefixCaptureOps = prefixCaptureOps.slice();
          appendLayer(sourcePrefixCaptureOps, full, i, false, { kvOnly: true, capturePrefix: true });
        }
        if (prefixSuffixOps) {
          sourcePrefixSuffixOps = prefixSuffixOps.slice();
          appendLayer(sourcePrefixSuffixOps, suffixTarget, i, false, { kvOnly: true, joinPrefix: true });
          sourceBatchPrefixSuffixOps = batchPrefixSuffixOps.slice();
          appendLayer(sourceBatchPrefixSuffixOps, { ...suffixTarget, part: this.batchPart || this.part, batched: true }, i, false, { kvOnly: true, joinPrefix: true });
        }
      }
      appendLayer(ops, full, i, false);
      if (prefixCaptureOps && i < cfg.sharedStart) appendLayer(prefixCaptureOps, full, i, false, { capturePrefix: true });
      if (prefixSuffixOps && i < cfg.sharedStart) appendLayer(prefixSuffixOps, suffixTarget, i, false, { joinPrefix: true });
      if (batchPrefixSuffixOps && i < cfg.sharedStart) appendLayer(batchPrefixSuffixOps, { ...suffixTarget, part: this.batchPart || this.part, batched: true }, i, false, { joinPrefix: true });
    }
    if (!prefixOps) prefixOps = ops.slice();
    ops.push(rms(full, "norm", full.x, full.h, cfg.hidden, 1, "rms.final"));
    ops.push(custom(full, "GATHER", [cfg.hidden, 0, 0, 0], [full.h, this.rows, this.gathered], "gather", (pass) => pass.dispatchWorkgroups(this.R)));
    this.ops = ops;
    this.prefixOps = prefixOps;
    this.prefixCaptureOps = prefixCaptureOps;
    this.prefixSuffixOps = prefixSuffixOps;
    this.batchPrefixSuffixOps = batchPrefixSuffixOps;
    this.sourcePrefixOps = sourcePrefixOps;
    this.sourcePrefixCaptureOps = sourcePrefixCaptureOps;
    this.sourcePrefixSuffixOps = sourcePrefixSuffixOps;
    this.sourceBatchPrefixSuffixOps = sourceBatchPrefixSuffixOps;
    this.createTextPrefixOps = canSharePrefix ? (plan, compactSource) => {
      const uniformStart = this.uniforms.length;
      try {
        const target = { ...full, prefixPlan: plan, sourcePolicy: true, logicalRows: plan.logicalRows };
        const x = view(plan, this.x, cfg.hidden);
        const input = view(plan, this.input, cfg.hidden);
        const prefix = cfg.hostWord
          ? [{ kind: "copy-embed", label: "embed.stored", dst: x }]
          : [custom(target, "EMBED", [embedWidth, Math.ceil(embedWidth / 32), f32(cfg.embeddingScale), 0], [target.ids, embed.buf, embed.sbuf || this.zeros, x], "embed", (pass) => pass.dispatchWorkgroups(this.T))];
        prefix.push({ kind: "copy-images", label: "embed.images" });
        prefix.push({ kind: "copy-input", label: "embed.copy", src: x, dst: input });
        for (let i = 0; i < cfg.sharedStart; i++) appendLayer(prefix, target, i, false, { capturePrefix: true, kvOnly: compactSource && i === sourceLayer });
        prefix.rmsNextLinks = gemma4RmsScaleNextPlan(prefix);
        return { key: plan.key, ops: prefix, uniforms: this.uniforms.slice(uniformStart) };
      } catch (error) {
        for (const buffer of this.uniforms.splice(uniformStart)) buffer.destroy();
        throw error;
      }
    } : null;
    this.tailOps = null;
    this.batchTailOps = null;
    this.sourceTailOps = null;
    this.sourceBatchTailOps = null;
    if (this.tail && cfg.kvSharedLayers && cfg.sharedStart > 0) {
      const tail = this.tail;
      const compact = {
        globals: tail.globals,
        part: tail.part,
        x: tail.x,
        input: tail.input,
        h: tail.h,
        branch: tail.branch,
        q: tail.q,
        ctx: tail.ctx,
        gate: tail.gate,
        up: tail.up,
        act: tail.act,
        pleToken: tail.pleToken,
        pleProj: tail.pleProj,
        ple: tail.ple,
        pleGate: tail.pleGate,
        pleAct: tail.pleAct,
        ids: tail.ids,
        pos: tail.pos,
        pleScope: "tail",
      };
      const tailOps = [];
      // The selected rows are gathered in caller order. This keeps duplicates and
      // arbitrary ordering exact while leaving the shared K/V pair full length.
      tailOps.push(custom(compact, "GATHER", [cfg.hidden, 0, 0, 0], [this.x, this.rows, compact.x], "tail.gather.hidden", (pass) => pass.dispatchWorkgroups(this.T)));
      tailOps.push(custom(compact, "GATHER", [cfg.hidden, 0, 0, 0], [this.input, this.rows, compact.input], "tail.gather.input", (pass) => pass.dispatchWorkgroups(this.T)));
      for (let i = cfg.sharedStart; i < cfg.layers; i++) appendLayer(tailOps, compact, i, true);
      tailOps.push(rms(compact, "norm", compact.x, compact.h, cfg.hidden, 1, "rms.final"));
      this.tailOps = tailOps;
      const batchTailOps = tailOps.slice(0, 2);
      const batchCompact = { ...compact, batched: true };
      for (let i = cfg.sharedStart; i < cfg.layers; i++) appendLayer(batchTailOps, batchCompact, i, true);
      batchTailOps.push(rms(batchCompact, "norm", compact.x, compact.h, cfg.hidden, 1, "rms.final"));
      this.batchTailOps = batchTailOps;
      const sourceCompact = { ...compact, globals: tail.sourceGlobals, sourcePolicy: true };
      const sourceTailOps = tailOps.slice(0, 2);
      appendLayer(sourceTailOps, sourceCompact, sourceLayer, true);
      sourceTailOps.push(...tailOps.slice(2));
      this.sourceTailOps = sourceTailOps;
      const sourceBatchTailOps = batchTailOps.slice(0, 2);
      appendLayer(sourceBatchTailOps, { ...sourceCompact, batched: true }, sourceLayer, true);
      sourceBatchTailOps.push(...batchTailOps.slice(2));
      this.sourceBatchTailOps = sourceBatchTailOps;
    }
    for (const graph of [this.ops, this.prefixOps, this.prefixCaptureOps, this.prefixSuffixOps, this.batchPrefixSuffixOps,
      this.sourcePrefixOps, this.sourcePrefixCaptureOps, this.sourcePrefixSuffixOps, this.sourceBatchPrefixSuffixOps,
      this.tailOps, this.batchTailOps, this.sourceTailOps, this.sourceBatchTailOps]) {
      if (graph) graph.rmsNextLinks = gemma4RmsScaleNextPlan(graph);
    }
  }

  encode(enc, ops, T, prof = this.profiler) {
    let pass = prof ? null : enc.beginComputePass();
    const skippedNorms = this.fuseRmsNext === true && this.rmsValidation.nextEnabled && ops.rmsNextLinks?.size ? new Set() : null;
    for (let index = 0; index < ops.length; index++) {
      const op = ops[index];
      if (skippedNorms?.has(op)) continue;
      if (op.kind === "copy-ple-vector") {
        const entry = this.pleVectorEntry(op, T);
        if (!entry || entry.ready) continue;
        pass?.end();
        enc.copyBufferToBuffer(op.src, 0, entry.buffer, 0, entry.bytes);
        this.activePleVectors.pending.add(entry);
        this.lastPleVectorCache.captures++;
        pass = prof ? null : enc.beginComputePass();
        continue;
      }
      if (op.kind === "mm" && op.pleScope) {
        const entry = this.pleVectorEntry(op, T);
        if (entry?.ready) {
          pass?.end();
          enc.copyBufferToBuffer(entry.buffer, 0, op.pleDst, 0, entry.bytes);
          this.lastPleVectorCache.hits++;
          index += 3;
          pass = prof ? null : enc.beginComputePass();
          continue;
        }
      }
      if (op.kind === "copy-images" && !this.imageSpans?.length) continue;
      if (["copy-input", "copy-kv", "copy-kv-prefix", "copy-kv-join", "copy-images", "copy-ple", "copy-embed"].includes(op.kind)) {
        pass?.end();
        if (op.kind === "copy-embed") enc.copyBufferToBuffer(this.hostPleUpload, 0, op.dst?.buffer || this.x, op.dst?.offset || 0, T * this.cfg.hidden * 4);
        else if (op.kind === "copy-images") {
          for (const span of this.imageSpans) enc.copyBufferToBuffer(span.projection?.buffer || this.imageUpload, span.projection ? 0 : span.offset, this.x, span.start * this.cfg.hidden * 4, span.len * this.cfg.hidden * 4);
        } else if (op.kind === "copy-ple") enc.copyBufferToBuffer(this.hostPleUpload, this.hostPleOffsets[op.layer], op.dst.buffer || op.dst, op.dst.offset || 0, T * this.cfg.pleDim * 4);
        else if (op.kind === "copy-input") enc.copyBufferToBuffer(op.src.buffer || op.src, op.src.offset || 0, op.dst.buffer || op.dst, op.dst.offset || 0, T * this.cfg.hidden * 4);
        else if (op.kind === "copy-kv-prefix") {
          const pair = this.prefixKV?.get(op.layer);
          if (!pair) throw new Error(`Gemma 4 missing captured prefix K/V for layer ${op.layer}`);
          if (op.staticRows) {
            const staticBytes = op.staticRows * op.width * 4;
            enc.copyBufferToBuffer(pair.k, 0, this.k, 0, staticBytes);
            enc.copyBufferToBuffer(pair.v, 0, this.v, 0, staticBytes);
          }
          const bytes = (op.totalRows || T) * op.width * 4;
          enc.copyBufferToBuffer(this.k, 0, pair.k, 0, bytes);
          enc.copyBufferToBuffer(this.v, 0, pair.v, 0, bytes);
          if (this.activeTextPrefixLease) this.textPrefixCache.recordCapture(this.activeTextPrefixLease, op.layer);
        } else if (op.kind === "copy-kv-join") {
          const pair = this.prefixKV?.get(op.layer);
          if (!pair) throw new Error(`Gemma 4 missing captured prefix K/V for layer ${op.layer}`);
          const prefixBytes = this.prefixLength * op.width * 4;
          const suffixBytes = T * op.width * 4;
          enc.copyBufferToBuffer(pair.k, 0, this.k, 0, prefixBytes);
          enc.copyBufferToBuffer(pair.v, 0, this.v, 0, prefixBytes);
          enc.copyBufferToBuffer(this.suffixK, 0, this.k, prefixBytes, suffixBytes);
          enc.copyBufferToBuffer(this.suffixV, 0, this.v, prefixBytes, suffixBytes);
        } else {
          const bytes = (op.totalRows || (op.total ? this.prefixLength + T : T)) * op.width * 4;
          enc.copyBufferToBuffer(op.srcK, 0, op.dstK, 0, bytes);
          enc.copyBufferToBuffer(op.srcV, 0, op.dstV, 0, bytes);
        }
        pass = prof ? null : enc.beginComputePass();
        continue;
      }
      const mlpTogether = op.mlpPartners && ops[index + 1] === op.mlpPartners[0] && ops[index + 2] === op.mlpPartners[1];
      const gridKernel = this.gpuKernel === "auto" && mlpTogether && this.fuseGridMlp === true && this.gridMlp?.selection.has("6144:1536")
        ? selectedMatmulKernel(this.tuning.selection, op.N, op.K, 1) : null;
      const gridBaseline = gridKernel && (this.batchMatmul[gridKernel] || this.batchMatmul.generic);
      const gridFused = gridBaseline && this.gridMlp.tokens === T && !this.batchMlp?.selection.has("6144:1536") &&
        gemma4GridMlpEligible(T, op, ops[index + 1], ops[index + 2], gridBaseline);
      const prefixGelu = mlpTogether && this.fusePrefixGelu === true && op.logicalRows && op.prefixGelu;
      const fused = mlpTogether && !op.sourcePolicy && (T === 1 || op.batched && T <= 8) && this.fuseMatvecGelu !== false && op.fused;
      const twoFused = mlpTogether && this.shareMatvecRows === true && !op.sourcePolicy && op.batched && T === 2 && this.fuseMatvecGelu !== false && op.twoFused;
      const prefillFused = mlpTogether && (!op.sourcePolicy || op.logicalRows) && !op.batched && gemma4MlpEligible(op.logicalRows || T) && this.fusePrefillGelu !== false && op.prefillFused;
      const batchPrefillFused = mlpTogether && !op.sourcePolicy && op.batched && this.fuseBatchPrefillGelu === true && this.batchMlpScratchReady && this.batchMlp?.tokens === T && gemma4BatchMlpEligible(T) && op.batchMlpFused;
      const nextFusion = skippedNorms && ops.rmsNextLinks.get(op);
      const rmsFusion = this.fuseRms === true && this.rmsValidation.enabled && ops[index + 1] === op.rmsPartner && (nextFusion?.fusion || op.rmsFusion);
      const attentionScope = T === 2 ? "tail" : T >= 16 && T <= 256 ? "prefill" : null;
      const subgroupAttention = op.subgroup && this.fuseAttentionSubgroups === true && attentionScope && this.attention.selection.has(`${op.attentionDim}:${attentionScope}`);
      if (prof) pass = prof.pass(enc, gridFused ? "mm.grid.gate.up.gelu" : prefixGelu ? "mm.prefix.bn32.gate.up.gelu" : twoFused ? "mm.two.gate.up.gelu" : fused ? "mm.gate.up.gelu" : prefillFused ? "mm.prefill.gate.up.gelu" : batchPrefillFused ? "mm.batch.gate.up.gelu" : rmsFusion ? rmsFusion.label : subgroupAttention ? "attn.subgroup" : op.label);
      if (gridFused) {
        encodeGemma4GridMlp(pass, this.gridMlp, op, T, this.device, gridBaseline);
        this.lastGridMlpDispatches++;
        index += 2;
      } else if (prefixGelu) {
        const splits = mmSplits(prefixGelu.splitTokens, prefixGelu.N, prefixGelu.K, 256, 32, 64);
        pass.setPipeline(prefixGelu.pipelines.matmul);
        pass.setBindGroup(0, prefixGelu.group);
        pass.dispatchWorkgroups(Math.ceil(prefixGelu.N / 32), Math.ceil(T / 32), splits);
        if (splits > 1) {
          const work = Math.ceil(T * prefixGelu.N / 256);
          pass.setPipeline(prefixGelu.pipelines.reduce);
          pass.setBindGroup(0, prefixGelu.reduce);
          pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
        }
        this.lastPrefixGeluDispatches++;
        index += 2;
      } else if (twoFused) {
        encodeMatvecTwo(pass, this.matvecTwo.pipelines.gelu, op.twoFused, T);
        this.lastMatvecTwoDispatches++;
        index += 2;
      } else if (fused) {
        encodeMatvecGelu(pass, this.matvecGelu.pipelines, op.fused, T);
        index += 2;
      } else if (prefillFused) {
        if (op.logicalRows) {
          pass.setPipeline(this.mlp.pipelines.pipeline);
          pass.setBindGroup(0, op.prefillFused.group);
          pass.dispatchWorkgroups(Math.ceil(op.N / 64), Math.ceil(T / 48), 1);
        } else encodeGemma4Mlp(pass, this.mlp.pipelines, op.prefillFused, T);
        index += 2;
      } else if (batchPrefillFused) {
        encodeGemma4BatchMlp(pass, this.batchMlp.pipelines, op.batchMlpFused, T);
        index += 2;
      } else if (rmsFusion) {
        pass.setPipeline(rmsFusion.pipeline);
        pass.setBindGroup(0, rmsFusion.group);
        rmsFusion.dispatch(pass);
        if (nextFusion && rmsFusion === nextFusion.fusion) {
          skippedNorms.add(nextFusion.next);
          this.lastRmsNextDispatches++;
        }
        index++;
      } else if (op.kind === "mm") {
        const plePartner = op.pleReduce && this.pleReduceEnabled === true && gemma4PleReduceAdjacent(op, ops[index + 1], T, true) ? ops[index + 1] : null;
        if (op.sourcePolicy) {
          if (this.encodeSourceMatmul(pass, op, T, plePartner) === true && plePartner) { index++; this.lastPleReduceDispatches++; }
        } else if (this.shareMatvecRows === true && op.batched && T === 2 && op.dtype === "q8" && this.matvecTwo.selection.plain.has(`${op.N}:${op.K}`)) {
          encodeMatvecTwo(pass, this.matvecTwo.pipelines.plain, op, T);
          this.lastMatvecTwoDispatches++;
        } else if ((T === 1 || op.batched && T <= 8) && op.dtype === "q8" && this.matvec.selection.has(`${op.N}:${op.K}`)) {
          encodeMatvec(pass, this.matvec.pipelines, op, T);
        } else if (this.useK64 !== false && !op.batched && op.dtype === "q8" && gemma4K64Eligible(T) && this.k64.selection.has(`${op.N}:${op.K}`)) {
          const selected = this.k64.pipelines;
          const ple = plePartner && gemma4PleReduceChoice(this.pleReduce, op, plePartner, T, selected, rowsPerThread(T, selected), T, "plain");
          encodeMatmul(pass, selected, op, T, T, ple);
          if (ple) { index++; this.lastPleReduceDispatches++; }
        } else {
          const kernel = this.gpuKernel === "auto" ? selectedMatmulKernel(this.tuning.selection, op.N, op.K, op.batched ? 1 : T) : this.gpuKernel;
          const pipes = op.batched ? this.batchMatmul : this.tuning.pipelines;
          const selected = pipes[kernel] || pipes.generic;
          const ple = plePartner && gemma4PleReduceChoice(this.pleReduce, op, plePartner, T, selected, rowsPerThread(T, selected), op.batched ? 1 : T, op.batched ? "batch" : "plain");
          if (!ple && this.gpuKernel === "auto" && this.batchSingleEnabled === true && this.batchSingle?.tokens === T && this.batchSingle.selection.has(`${op.N}:${op.K}`) && gemma4BatchSingleEligible(T, op) && selected.f16 === true && selected.groups === 1 && selected.splitTarget === 256 && rowsPerThread(T, selected) === 4) {
            encodeGemma4BatchSingle(pass, this.batchSingle.pipelines, selected, op, T);
            this.lastBatchSingleDispatches++;
          } else encodeMatmul(pass, selected, op, T, op.batched ? 1 : T, ple);
          if (ple) { index++; this.lastPleReduceDispatches++; }
        }
      } else if (subgroupAttention) {
        pass.setPipeline(this.attention.pipeline);
        pass.setBindGroup(0, op.subgroup);
        op.dispatch(pass);
        this.lastAttentionDispatches++;
      } else {
        pass.setPipeline(op.pipeline);
        pass.setBindGroup(0, op.group);
        op.dispatch(pass);
      }
      if (prof) {
        pass.end();
        pass = null;
      }
    }
    pass?.end();
  }

  async hostPleRows(ids, pleIds, owned, pins, prefixLayers, tailIds = null) {
    if (this.cfg.hostPle !== true) return null;
    const cfg = this.cfg;
    const embedFloats = cfg.hostWord ? ids.length * cfg.hidden : 0;
    const floats = embedFloats + (prefixLayers * ids.length + (cfg.layers - prefixLayers) * (tailIds?.length || 0)) * cfg.pleDim;
    this.checkBuffer(floats * 4, "Gemma 4 shared-prefix host PLE rows");
    const key = JSON.stringify([prefixLayers, Array.from(pleIds), cfg.hostWord ? Array.from(ids) : null, tailIds && Array.from(tailIds)]);
    let cached = this.pleCache.get(key);
    const hit = !!cached;
    if (cached) {
      this.pleCache.delete(key);
      this.pleCache.set(key, cached);
    } else {
      const hostRows = new Float32Array(floats);
      const offsets = [];
      if (cfg.hostWord) await this.weights.gatherStoredRows("embed", ids, cfg.embeddingScale, hostRows, 0);
      let offset = embedFloats;
      for (let layer = 0; layer < cfg.layers; layer++) {
        const lookupIds = layer < prefixLayers ? pleIds : tailIds || [];
        offsets[layer] = offset * 4;
        if (lookupIds?.length) await this.weights.gatherStoredRows(`ple.${layer}.embed`, lookupIds, cfg.pleEmbeddingScale, hostRows, offset);
        offset += lookupIds.length * cfg.pleDim;
      }
      const buffer = this.device.createBuffer({ label: "Gemma 4 shared-prefix PLE rows", size: Math.max(16, hostRows.byteLength), usage: U.COPY_SRC | U.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, hostRows);
      cached = { buffer, offsets, bytes: hostRows.byteLength, active: 0 };
      if (cached.bytes <= 16 * 1024 * 1024) {
        let canCache = true;
        while (this.pleCache.size >= 8 || this.pleCacheBytes + cached.bytes > 16 * 1024 * 1024) {
          const oldest = [...this.pleCache].find(([, entry]) => !entry.active)?.[0];
          if (oldest === undefined) { canCache = false; break; }
          const entry = this.pleCache.get(oldest);
          this.pleCacheBytes -= entry.bytes;
          entry.buffer.destroy();
          this.pleCache.delete(oldest);
        }
        if (canCache) {
          this.pleCache.set(key, cached);
          this.pleCacheBytes += cached.bytes;
        } else cached.temporary = true;
      } else cached.temporary = true;
      if (cached.temporary) owned.push(buffer);
    }
    cached.active = (cached.active || 0) + 1;
    pins.push(cached);
    this.hostPleUpload = cached.buffer;
    this.hostPleOffsets = cached.offsets;
    return { bytes: cached.bytes, cacheHit: hit };
  }

  /** Run a request's common image prefix once, then each independent causal suffix. */
  async forwardSharedPrefix(sequences) {
    this.lastBatchSingleDispatches = 0;
    this.lastPleReduceDispatches = 0;
    this.lastAttentionDispatches = 0;
    this.lastMatvecTwoDispatches = 0;
    this.lastPrefixRetileDispatches = 0;
    this.lastPrefixBn32Dispatches = 0;
    this.lastPrefixGeluDispatches = 0;
    this.lastRmsNextDispatches = 0;
    const sharedPrefixTokens = sharedImagePrefix(sequences);
    this.lastSourceCompaction = { enabled: false };
    this.lastTextPrefixCache = { enabled: false, hit: false, staticRows: 0, activeRows: sharedPrefixTokens, bytes: 0, reason: this.cacheTextPrefix === true ? "ineligible" : "disabled" };
    this.lastPrefixReuse = { enabled: false, reusedTokens: 0, totalTokens: sequences.reduce((sum, sequence) => sum + (sequence.ids?.length || 0), 0), sharedPrefixTokens };
    if (!sharedPrefixTokens || !this.prefixCaptureOps || this.profiler) {
      this.clearTextPrefixCache();
      return null;
    }
    this.lastProfile = null;
    let success = false;
    try {
      const result = await withGpuErrors(this.runtime, () => this.forwardSharedRows(sequences, sharedPrefixTokens), "running Gemma 4 shared-prefix inference");
      success = result !== null;
      return result;
    } finally {
      if (this.activeTextPrefixLease) {
        this.lastTextPrefixCache.committed = this.textPrefixCache.finish(this.activeTextPrefixLease, success);
        this.activeTextPrefixLease = null;
      }
      if (!success) {
        this.clearTextPrefixCache();
        this.clearPleVectorCache();
      }
      this.imageSpans = [];
      this.hostPleUpload = null;
      this.hostPleOffsets = null;
    }
  }

  async forwardSharedRows(sequences, sharedPrefixTokens) {
    this.lastGridMlpDispatches = 0;
    this.lastBatchSingleDispatches = 0;
    this.lastPleReduceDispatches = 0;
    const cfg = this.cfg;
    const q = this.device.queue;
    const owned = [];
    const plePins = [];
    let prefixKV = new Map();
    let output = null;
    let outputRead = null;
    let submitted = false;
    let completed = false;
    let imageBytes = 0;
    let pleMs = 0;
    let pleCacheHits = 0;
    try {
      let maxTokens = 0;
      for (const sequence of sequences) {
        const { ids, imageSpans = [] } = sequence;
        if (!ids.length || ids.some((id) => id >= cfg.vocab || id >= cfg.pleVocab)) throw new Error("Gemma 4 token id is outside an embedding vocabulary");
        let end = 0;
        for (const span of imageSpans) {
          if (!Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.len) || span.len < 1 || span.start < end || span.start + span.len > sharedPrefixTokens || !validImageRows(span.rows, span.len, cfg.hidden, this.device)) throw new Error("Gemma 4 shared image spans must be complete, ordered projected rows within the common prefix");
          end = span.start + span.len;
          if (ids.subarray(span.start, end).some((id) => id !== 258880)) throw new Error("Gemma 4 image span does not match image placeholder tokens");
          if (sequence === sequences[0] && span.rows instanceof Float32Array) imageBytes += span.rows.byteLength;
        }
        maxTokens = Math.max(maxTokens, ids.length);
      }
      const uniqueTokens = sharedPrefixTokens + sequences.reduce((sum, sequence) => sum + sequence.ids.length - sharedPrefixTokens, 0);
      if (sequences.length > 8 || maxTokens > 256 || uniqueTokens > 512) return null;
      const suffixTokens = uniqueTokens - sharedPrefixTokens;
      const batchUploadFloats = (cfg.hostWord ? suffixTokens * cfg.hidden : 0) + (cfg.sharedStart * suffixTokens + cfg.kvSharedLayers * sequences.length) * cfg.pleDim;
      let batch = this.batchSuffixes && sequences.length >= 2 && sequences.every((sequence) => {
        const length = sequence.ids.length - sharedPrefixTokens;
        return length >= 1 && length <= 64;
      }) && batchUploadFloats * 4 <= 16 * 1024 * 1024;
      const prefixKvBytes = Array.from({ length: cfg.sharedStart }, (_, layer) => {
        const headDim = cfg.full[layer] ? cfg.globalHeadDim : cfg.localHeadDim;
        return sharedPrefixTokens * cfg.kvHeads * headDim * 8;
      }).reduce((sum, bytes) => sum + bytes, 0);
      if (prefixKvBytes > 64 * 1024 * 1024) return null;
      if (cfg.hostPle === true) {
        const prefixBytes = ((cfg.hostWord ? sharedPrefixTokens * cfg.hidden : 0) + cfg.sharedStart * sharedPrefixTokens * cfg.pleDim) * 4;
        let pleBytes = prefixBytes;
        const uploadLimit = 16 * 1024 * 1024;
        if (prefixBytes > uploadLimit) return null;
        for (const sequence of sequences) {
          const suffixTokens = sequence.ids.length - sharedPrefixTokens;
          const suffixFloats = (cfg.hostWord ? suffixTokens * cfg.hidden : 0) + cfg.sharedStart * suffixTokens * cfg.pleDim + (cfg.layers - cfg.sharedStart) * cfg.pleDim;
          const suffixBytes = suffixFloats * 4;
          if (suffixBytes > uploadLimit) return null;
          pleBytes += suffixBytes;
        }
        if (pleBytes > 64 * 1024 * 1024) return null;
      }
      if (batch && !this.batchMatmul) {
        const names = Object.keys(this.tuning.pipelines).filter((name) => this.tuning.pipelines[name]);
        this.batchMatmul = Object.fromEntries(await Promise.all(names.map(async (name) => [name, await matmulPipelines(this.device, this.wgsl, name === "wide" ? "matmul_wide_gemma_batch" : "matmul_gemma_batch")])));
      }
      await this.ensureRope(maxTokens);
      this.ensure(batch ? uniqueTokens : maxTokens);
      this.ensureTail(batch ? sequences.length : 1);
      if (!this.prefixCaptureOps || !this.prefixSuffixOps || !this.tailOps) return null;
      if (batch && !this.ensureBatchPart(suffixTokens)) batch = false;
      const compact = this.compactSourceRows === true && !!this.sourceTailOps;
      if (batch && (this.fuseBatchPrefillGelu === true || this.batchMlp)) {
        await this.prepareBatchMlp(suffixTokens, compact);
        if (this.batchMlp) await this.ensureBatchMlpPart(suffixTokens, compact);
      }
      if (batch) await this.prepareBatchSingle(suffixTokens, compact);
      if (batch) await this.prepareGridMlp(suffixTokens, compact);
      if (compact) await this.ensureSourcePipelines();
      const textPlan = this.cacheTextPrefix === true ? textPrefixCachePlan({ device: this.device, cfg, policy: this.textPrefixPolicy(compact), calibrationPolicy: this.textPrefixPolicy(compact, false) }, sequences, sharedPrefixTokens) : null;
      if (textPlan) await this.ensureSourcePipelines();
      await this.preparePrefixRetile(textPlan);
      if (this.textPrefixGraph?.key !== textPlan?.key) this.clearTextPrefixGraph();
      const textLease = this.textPrefixCache.begin(textPlan);
      this.activeTextPrefixLease = textLease;
      const warmPrefix = textLease?.hit === true;
      const prefixRowStart = warmPrefix ? textPlan.staticRows : 0;
      const prefixActiveRows = sharedPrefixTokens - prefixRowStart;
      if (warmPrefix && !this.textPrefixGraph) this.textPrefixGraph = this.createTextPrefixOps(textPlan, compact);
      if (textLease) this.lastTextPrefixCache = { enabled: true, hit: warmPrefix, staticRows: textPlan.staticRows, activeRows: prefixActiveRows, logicalRows: sharedPrefixTokens, bytes: textPlan.bytes, capturedLayers: 0, committed: false };
      const prefixLayers = cfg.sharedStart - Number(compact);
      const pleScopes = batch ? [
        { slot: "suffix.batch", rows: suffixTokens, start: 0, end: prefixLayers },
        { slot: "tail.batch", rows: sequences.length, start: prefixLayers, end: cfg.layers },
      ] : sequences.flatMap((sequence, index) => [
        { slot: `suffix.${index}`, rows: sequence.ids.length - sharedPrefixTokens, start: 0, end: prefixLayers },
        { slot: `tail.${index}`, rows: 1, start: prefixLayers, end: cfg.layers },
      ]);
      this.preparePleVectorCache(["shared", batch, compact, sharedPrefixTokens, sequences.map((sequence) => Array.from(sequence.ids))], pleScopes);
      if (textLease) prefixKV = textLease.kv;
      else for (let layer = 0; layer < cfg.sharedStart; layer++) {
        const headDim = cfg.full[layer] ? cfg.globalHeadDim : cfg.localHeadDim;
        const layerWidth = cfg.kvHeads * headDim;
        const bytes = sharedPrefixTokens * layerWidth * 4;
        this.checkBuffer(bytes, `shared prefix K/V layer ${layer}`);
        const k = this.device.createBuffer({ label: `Gemma 4 prefix K ${layer}`, size: Math.max(16, bytes), usage: U.COPY_SRC | U.COPY_DST });
        const v = this.device.createBuffer({ label: `Gemma 4 prefix V ${layer}`, size: Math.max(16, bytes), usage: U.COPY_SRC | U.COPY_DST });
        owned.push(k, v);
        prefixKV.set(layer, { k, v, width: layerWidth });
      }
      this.prefixKV = prefixKV;
      this.prefixLength = sharedPrefixTokens;
      const imageUpload = this.device.createBuffer({ label: "Gemma 4 shared-prefix image rows", size: Math.max(16, imageBytes), usage: U.COPY_SRC | U.COPY_DST });
      this.imageUpload = imageUpload;
      owned.push(imageUpload);
      let imageOffset = 0;
      const imageSpans = sequences[0].imageSpans.map((span) => {
        const projection = gpuVisionProjection(span.rows, this.device, span.len * cfg.hidden * 4);
        const item = { ...span, offset: imageOffset, projection };
        if (!projection) {
          q.writeBuffer(imageUpload, imageOffset, span.rows);
          imageOffset += span.rows.byteLength;
        }
        return item;
      });
      const prefixIds = sequences[0].ids.subarray(prefixRowStart, sharedPrefixTokens);
      const prefixPleIds = prefixIds.slice();
      for (const span of imageSpans) prefixPleIds.fill(0, span.start - prefixRowStart, span.start + span.len - prefixRowStart);
      const parameterWords = batch
        ? 12 + 3 * prefixActiveRows + 4 * suffixTokens + 4 * sequences.length + (compact ? 4 : 0)
        : 4 + 3 * prefixActiveRows + sequences.reduce((sum, sequence) => sum + 11 + 3 * (sequence.ids.length - sharedPrefixTokens) + (compact ? 4 : 0), 0);
      const parameters = new Uint32Array(parameterWords);
      const parameterUpload = this.device.createBuffer({ label: "Gemma 4 branch parameters", size: Math.max(16, parameters.byteLength), usage: U.COPY_SRC | U.COPY_DST });
      owned.push(parameterUpload);
      const encoder = this.device.createCommandEncoder({ label: "Gemma 4 shared prefix and suffixes" });
      let parameterOffset = 0;
      const stage = (destination, values) => {
        parameters.set(values, parameterOffset);
        encoder.copyBufferToBuffer(parameterUpload, parameterOffset * 4, destination, 0, values.byteLength);
        parameterOffset += values.length;
      };
      this.imageSpans = imageSpans;
      this.T = prefixActiveRows;
      this.R = sharedPrefixTokens;
      this.sourceTokens = sharedPrefixTokens;
      const prefixPositions = Uint32Array.from({ length: prefixActiveRows }, (_, i) => prefixRowStart + i);
      stage(this.globals, new Uint32Array([prefixActiveRows, sharedPrefixTokens, sharedPrefixTokens, 0]));
      stage(this.ids, prefixIds);
      stage(this.pleIds, prefixPleIds);
      stage(this.pos, prefixPositions);
      const prefixPleStarted = performance.now();
      const prefixPle = await this.hostPleRows(prefixIds, prefixPleIds, owned, plePins, prefixLayers);
      pleCacheHits += Number(prefixPle?.cacheHit || false);
      pleMs += performance.now() - prefixPleStarted;
      this.encode(encoder, warmPrefix ? this.textPrefixGraph.ops : compact ? this.sourcePrefixCaptureOps : this.prefixCaptureOps, prefixActiveRows, null);
      if (textLease) this.lastTextPrefixCache.capturedLayers = textLease.captured.size;

      const hiddenBytes = cfg.hidden * 4;
      const outputBytes = sequences.length * hiddenBytes;
      const flags = imageSpans.filter((span) => span.projection).map((span) => span.projection.status);
      const readBytes = outputBytes + flags.length * 4;
      this.checkBuffer(readBytes, "shared-prefix output rows and projection status");
      output = this.device.createBuffer({ label: "Gemma 4 shared-prefix output rows", size: Math.max(16, outputBytes), usage: U.COPY_SRC | U.COPY_DST });
      outputRead = this.device.createBuffer({ label: "Gemma 4 shared-prefix readback", size: Math.max(16, readBytes), usage: U.MAP_READ | U.COPY_DST });
      owned.push(output, outputRead);
      this.imageSpans = [];
      if (batch) {
        const ids = new Uint32Array(suffixTokens);
        const positions = new Uint32Array(2 * suffixTokens);
        const rows = new Uint32Array(sequences.length);
        const tailPositions = new Uint32Array(2 * sequences.length);
        const tailIds = new Uint32Array(sequences.length);
        let offset = 0;
        for (const [index, sequence] of sequences.entries()) {
          const suffix = sequence.ids.subarray(sharedPrefixTokens);
          ids.set(suffix, offset);
          for (let row = 0; row < suffix.length; row++) {
            positions[offset + row] = sharedPrefixTokens + row;
            positions[suffixTokens + offset + row] = offset;
          }
          rows[index] = offset + suffix.length - 1;
          tailPositions[index] = sequence.ids.length - 1;
          tailPositions[sequences.length + index] = offset;
          tailIds[index] = suffix.at(-1);
          offset += suffix.length;
        }
        this.T = suffixTokens;
        this.R = sequences.length;
        stage(this.globals, new Uint32Array([suffixTokens, sequences.length, uniqueTokens, sharedPrefixTokens]));
        stage(this.ids, ids);
        stage(this.pleIds, ids);
        stage(this.pos, positions);
        stage(this.rows, rows);
        stage(this.tail.globals, new Uint32Array([sequences.length, sequences.length, uniqueTokens, sharedPrefixTokens]));
        if (compact) stage(this.tail.sourceGlobals, new Uint32Array([sequences.length, 1, uniqueTokens, sharedPrefixTokens]));
        stage(this.tail.pos, tailPositions);
        stage(this.tail.ids, tailIds);
        const pleStarted = performance.now();
        const suffixPle = await this.hostPleRows(ids, ids, owned, plePins, prefixLayers, tailIds);
        pleCacheHits += Number(suffixPle?.cacheHit || false);
        pleMs += performance.now() - pleStarted;
        this.pleVectorSlots = { suffix: "suffix.batch", tail: "tail.batch" };
        this.encode(encoder, compact ? this.sourceBatchPrefixSuffixOps : this.batchPrefixSuffixOps, suffixTokens, null);
        this.T = sequences.length;
        this.sourceTokens = suffixTokens;
        this.encode(encoder, compact ? this.sourceBatchTailOps : this.batchTailOps, sequences.length, null);
        encoder.copyBufferToBuffer(this.tail.h, 0, output, 0, outputBytes);
      } else for (let index = 0; index < sequences.length; index++) {
        const ids = sequences[index].ids;
        const suffix = ids.subarray(sharedPrefixTokens);
        const pleIds = suffix.slice();
        const length = suffix.length;
        const positions = Uint32Array.from({ length }, (_, i) => sharedPrefixTokens + i);
        this.T = length;
        this.R = 1;
        stage(this.globals, new Uint32Array([length, 1, ids.length, 0]));
        stage(this.ids, suffix);
        stage(this.pleIds, pleIds);
        stage(this.pos, positions);
        stage(this.rows, new Uint32Array([length - 1]));
        const selectedPosition = ids.length - 1;
        stage(this.tail.globals, new Uint32Array([1, 1, ids.length, 0]));
        if (compact) stage(this.tail.sourceGlobals, new Uint32Array([1, length, ids.length, 0]));
        stage(this.tail.pos, new Uint32Array([selectedPosition]));
        stage(this.tail.ids, new Uint32Array([pleIds[length - 1]]));
        const pleStarted = performance.now();
        const tailIds = new Uint32Array([pleIds[length - 1]]);
        const suffixPle = await this.hostPleRows(suffix, pleIds, owned, plePins, prefixLayers, tailIds);
        pleCacheHits += Number(suffixPle?.cacheHit || false);
        pleMs += performance.now() - pleStarted;
        this.pleVectorSlots = { suffix: `suffix.${index}`, tail: `tail.${index}` };
        this.encode(encoder, compact ? this.sourcePrefixSuffixOps : this.prefixSuffixOps, length, null);
        this.T = 1;
        this.sourceTokens = length;
        this.encode(encoder, compact ? this.sourceTailOps : this.tailOps, 1, null);
        encoder.copyBufferToBuffer(this.tail.h, 0, output, index * hiddenBytes, hiddenBytes);
      }
      encoder.copyBufferToBuffer(output, 0, outputRead, 0, outputBytes);
      flags.forEach((status, i) => encoder.copyBufferToBuffer(status, 0, outputRead, outputBytes + i * 4, 4));
      q.writeBuffer(parameterUpload, 0, parameters);
      q.submit([encoder.finish()]);
      submitted = true;
      await outputRead.mapAsync(GPUMapMode.READ, 0, readBytes);
      const result = mappedImageRows(outputRead, outputBytes, flags.length);
      completed = true;
      const totalTokens = sequences.reduce((sum, sequence) => sum + sequence.ids.length, 0);
      this.lastPrefixReuse = { enabled: true, batchedSuffixes: Boolean(batch), compactSourceRows: compact, cachedTextPrefix: warmPrefix, staticPrefixRows: warmPrefix ? prefixRowStart : 0, retiledPrefixMatrices: this.lastPrefixRetileDispatches, narrowPrefixMatrices: this.lastPrefixBn32Dispatches, rmsNextFusions: this.lastRmsNextDispatches, pleReduceFusions: this.lastPleReduceDispatches, prefixGeluFusions: this.lastPrefixGeluDispatches, sharedMatvecMatrices: this.lastMatvecTwoDispatches, splitScratchBytes: batch ? (this.batchPart || this.part).size : this.part.size, reusedTokens: (sequences.length - 1) * sharedPrefixTokens + prefixRowStart, totalTokens, sharedPrefixTokens };
      this.lastSourceCompaction = { enabled: compact, sourceLayer: cfg.sharedStart - 1, prefixRows: sharedPrefixTokens, originalRows: suffixTokens, selectedRows: sequences.length };
      this.lastPle = { ms: pleMs, cacheHit: pleCacheHits > 0, cacheHits: pleCacheHits };
      return result;
    } finally {
      if (submitted && !completed) await q.onSubmittedWorkDone().catch(() => {});
      this.finishPleVectorCache(completed);
      this.imageUpload = null;
      this.prefixKV = null;
      this.imageSpans = [];
      this.hostPleUpload = null;
      this.hostPleOffsets = null;
      for (const cached of plePins) cached.active--;
      for (const buffer of owned) buffer.destroy();
    }
  }

  async forward(inputIds, rows = null, imageSpans = []) {
    try {
      return await withGpuErrors(this.runtime, () => this.forwardRows(inputIds, rows, imageSpans), "running Gemma 4 inference");
    } catch (error) {
      this.clearPleVectorCache();
      this.clearTextPrefixCache();
      throw error;
    } finally {
      this.imageSpans = [];
    }
  }

  async forwardRows(inputIds, rows, imageSpans) {
    this.lastGridMlpDispatches = 0;
    this.lastBatchSingleDispatches = 0;
    this.lastPleReduceDispatches = 0;
    this.lastAttentionDispatches = 0;
    this.lastMatvecTwoDispatches = 0;
    this.lastPrefixRetileDispatches = 0;
    this.lastPrefixBn32Dispatches = 0;
    this.lastPrefixGeluDispatches = 0;
    this.lastRmsNextDispatches = 0;
    this.clearTextPrefixCache();
    this.lastTextPrefixCache = { enabled: false, reason: "unshared-request" };
    this.lastSourceCompaction = { enabled: false };
    if (!this.initialized) throw new Error("Gemma 4 GPU trunk is not initialized");
    const ids = inputIds instanceof Uint32Array ? inputIds : Uint32Array.from(inputIds);
    const T = ids.length;
    if (!T) throw new Error("Gemma 4 GPU forward needs at least one token");
    if (ids.some((id) => id >= this.cfg.vocab || id >= this.cfg.pleVocab)) throw new Error("Gemma 4 token id is outside an embedding vocabulary");
    const pleIds = ids.slice();
    let imageBytes = 0;
    let end = 0;
    this.imageSpans = imageSpans.map((span) => {
      if (!Number.isSafeInteger(span.start) || !Number.isSafeInteger(span.len) || span.len < 1 || span.start < end || span.start + span.len > T || !validImageRows(span.rows, span.len, this.cfg.hidden, this.device)) throw new Error("Gemma 4 image spans must contain ordered, nonoverlapping projected rows within the sequence");
      end = span.start + span.len;
      if (ids.subarray(span.start, end).some((id) => id !== 258880)) throw new Error("Gemma 4 image span does not match image placeholder tokens");
      pleIds.fill(0, span.start, end);
      const offset = imageBytes;
      const projection = gpuVisionProjection(span.rows, this.device, span.len * this.cfg.hidden * 4);
      if (!projection) imageBytes += span.rows.byteLength;
      return { ...span, offset, projection };
    });
    const selected = rows == null ? [T - 1] : Array.from(rows, Number);
    const profiler = this.profiler;
    if (selected.some((r) => !Number.isSafeInteger(r) || r < 0 || r >= T)) throw new Error("Gemma 4 output row is outside the input sequence");
    if (!selected.length) {
      this.lastProfile = null;
      this.preparePleVectorCache(["empty"], []);
      return new Float32Array(0);
    }
    await this.ensureRope(T);
    // A caller-provided identity selection can run the ordinary full-row graph. Any
    // other selection uses the compact shared-KV tail when the model has one.
    const allRows = selected.length === T && selected.every((row, i) => row === i);
    this.ensure(Math.max(T, selected.length));
    if (!allRows) this.ensureTail(selected.length);
    this.R = selected.length;
    const q = this.device.queue;
    const positions = new Uint32Array(T);
    for (let i = 0; i < T; i++) positions[i] = i;
    q.writeBuffer(this.globals, 0, new Uint32Array([T, selected.length, T, 0]));
    q.writeBuffer(this.ids, 0, ids);
    q.writeBuffer(this.pleIds, 0, pleIds);
    q.writeBuffer(this.pos, 0, positions);
    q.writeBuffer(this.rows, 0, new Uint32Array(selected));
    const useTail = !allRows && !!this.tailOps && !!this.prefixOps;
    const compact = useTail && this.compactSourceRows === true && selected.length <= 64 && (T !== 1 || selected.length <= 16) && !!this.sourceTailOps;
    if (compact) await this.ensureSourcePipelines();
    if (useTail) {
      q.writeBuffer(this.tail.globals, 0, new Uint32Array([selected.length, selected.length, T, 0]));
      if (compact) q.writeBuffer(this.tail.sourceGlobals, 0, new Uint32Array([selected.length, T, T, 0]));
      q.writeBuffer(this.tail.pos, 0, new Uint32Array(selected));
      q.writeBuffer(this.tail.ids, 0, Uint32Array.from(selected, (row) => pleIds[row]));
    }
    const textRows = useTail && selected.every((row) => !this.imageSpans.some((span) => row >= span.start && row < span.start + span.len));
    this.preparePleVectorCache(["selected", compact, Array.from(ids), selected], textRows ? [{ slot: "tail", rows: selected.length, start: this.cfg.sharedStart - Number(compact), end: this.cfg.layers }] : []);
    this.pleVectorSlots = { tail: "tail" };
    const pleOwned = [];
    const plePins = [];
    let completed = false;
    try {
      if (this.cfg.hostPle === true) {
        const pleStarted = performance.now();
        const prefixLayers = useTail ? this.cfg.sharedStart - Number(compact) : this.cfg.layers;
        const tailIds = useTail ? Uint32Array.from(selected, (row) => pleIds[row]) : null;
        const cached = await this.hostPleRows(ids, pleIds, pleOwned, plePins, prefixLayers, tailIds);
        this.lastPle = { ms: performance.now() - pleStarted, cacheHit: cached.cacheHit, bytes: cached.bytes };
      }
      if (imageBytes) {
        this.checkBuffer(imageBytes, "Gemma 4 image embedding upload");
        this.imageUpload = this.device.createBuffer({ label: "Gemma 4 projected images", size: imageBytes, usage: U.COPY_SRC | U.COPY_DST });
        for (const span of this.imageSpans) if (!span.projection) q.writeBuffer(this.imageUpload, span.offset, span.rows);
      }
      const longChunks = !profiler && T > YIELD_TOKENS;
      const runOps = useTail ? compact ? this.sourcePrefixOps : this.prefixOps : this.ops;
      const plans = (longChunks ? chunks(runOps, YIELD_CHUNKS) : [runOps]).map((ops, i) => ({ ops, rows: T, flushBefore: i > 0 }));
      if (useTail) plans.push({ ops: compact ? this.sourceTailOps : this.tailOps, rows: selected.length, flushBefore: false });
      this.sourceTokens = T;
      let enc = this.device.createCommandEncoder({ label: "Gemma 4 forward" });
      for (let i = 0; i < plans.length; i++) {
        if (plans[i].flushBefore) enc = await endChunk(this.device, enc);
        this.T = plans[i].rows;
        this.encode(enc, plans[i].ops, plans[i].rows, profiler);
      }
      profiler?.finish(enc);
      const bytes = selected.length * this.cfg.hidden * 4;
      const flags = this.imageSpans.filter((span) => span.projection).map((span) => span.projection.status);
      const readBytes = bytes + flags.length * 4;
      enc.copyBufferToBuffer(useTail ? this.tail.h : this.gathered, 0, this.readback, 0, bytes);
      flags.forEach((status, i) => enc.copyBufferToBuffer(status, 0, this.readback, bytes + i * 4, 4));
      q.submit([enc.finish()]);
      const pending = [this.readback.mapAsync(GPUMapMode.READ, 0, readBytes), profiler ? profiler.collect() : null];
      try {
        const [, profile] = await Promise.all(pending);
        if (profiler) this.lastProfile = profile;
      } catch (error) {
        await Promise.allSettled(pending);
        if (this.readback.mapState === "mapped") this.readback.unmap();
        throw error;
      }
      const out = mappedImageRows(this.readback, bytes, flags.length);
      completed = true;
      this.lastSourceCompaction = { enabled: compact, sourceLayer: this.cfg.sharedStart - 1, originalRows: T, selectedRows: selected.length };
      return out;
    } finally {
      this.finishPleVectorCache(completed);
      this.imageUpload?.destroy();
      this.imageUpload = null;
      this.imageSpans = [];
      this.hostPleUpload = null;
      this.hostPleOffsets = null;
      for (const cached of plePins) cached.active--;
      for (const buffer of pleOwned) buffer.destroy();
    }
  }
}
