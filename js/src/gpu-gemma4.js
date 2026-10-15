// Gemma 4 E2B/E4B text trunk on WebGPU.
//
// E2B and E4B are dense models. Their distinctive GPU requirements are the packed
// per-layer embeddings (PLE), 256-wide local heads, 512-wide proportional-RoPE
// global heads, and shared KV states in the final layers. This implementation keeps
// projection tensors and word embeddings on the device. Image sessions can keep
// original Q8 PLE tables on the host and upload only the rows a request uses.

import { GpuWeights, pipeline, matmulPipelines, encodeMatmul, SPLIT_SCRATCH, YIELD_TOKENS, YIELD_CHUNKS, chunks, endChunk, withGpuErrors } from "./gpu.js";
import { calibrateMatmul, selectedMatmulKernel } from "./gpu-tuning.js";
import { gpuVisionProjection } from "./gpu-gemma4-vision.js";

let U;

const GEMMA4_KERNELS = [
  ["EMBED", "gemma4_embed"],
  ["RMS", "gemma4_rms"],
  ["QKV", "gemma4_qkv"],
  ["ATTN", "gemma4_attention"],
  ["GELU", "gemma4_gelu"],
  ["PLE", "gemma4_ple"],
  ["RESIDUAL", "gemma4_residual"],
  ["GATHER", "gemma4_gather"],
];

const f32 = (v) => ({ f: v });
const pow2 = (n) => 2 ** Math.ceil(Math.log2(Math.max(1, n)));

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
    this.lastPrefixReuse = null;
    this.tailCapacity = 0;
    this.tail = null;
  }

  write(dst, bytes) {
    this.weights.write(dst, bytes);
  }

  async flushWrites() { await this.cfg.hostStore?.flush(); }

  async finishWeights() { await this.cfg.hostStore?.finish(); }

  dispose() {
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
    const built = await Promise.all(GEMMA4_KERNELS.map(async ([key, name]) => [key, await pipeline(d, this.wgsl(name, { subgroups: this.subgroup32 && (key === "RMS" || key === "PLE") }), name)]));
    this.p = Object.fromEntries(built);
    this.tuning = await calibrateMatmul(d, this.wgsl, this.weights, { kernel: this.gpuKernel, pipelines: { generic } });
    this.mm = this.tuning.pipelines.generic;
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
    this.tailOps = null;
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
    this.ple = make(cap * P, 0, "Gemma 4 PLE input");
    this.pleGate = make(cap * P, 0, "Gemma 4 PLE gate");
    this.pleAct = make(cap * P, 0, "Gemma 4 PLE activation");
    this.ids = make(cap, U.COPY_DST, "Gemma 4 token ids");
    this.pleIds = make(cap, U.COPY_DST, "Gemma 4 PLE token ids");
    this.pos = make(cap, U.COPY_DST, "Gemma 4 token positions");
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
    this.tailBuffers = buffers;
    this.tailCapacity = cap;
    this.tail = {
      globals,
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
      ple: make(cap * P, 0, "Gemma 4 compact PLE input"),
      pleGate: make(cap * P, 0, "Gemma 4 compact PLE gate"),
      pleAct: make(cap * P, 0, "Gemma 4 compact PLE activation"),
      ids: make(cap, U.COPY_DST, "Gemma 4 compact token ids"),
      pos: make(cap, U.COPY_DST, "Gemma 4 compact token positions"),
      part: make(SPLIT_SCRATCH, 0, "Gemma 4 compact matmul split scratch"),
    };
    if (this.p) this.build();
  }

  weight(name) {
    const e = this.tensors.get(name);
    if (!e) throw new Error(`Gemma 4 GPU layout has no tensor ${name}`);
    return e;
  }

  build() {
    for (const b of this.uniforms) b.destroy();
    this.uniforms = [];
    const d = this.device;
    const cfg = this.cfg;
    const bg = (p, resources) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: resources.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const custom = (target, name, values, resources, label = name, dispatch) => ({
      kind: name,
      label,
      pipeline: this.p[name],
      group: bg(this.p[name], [target.globals, this.uniform(values), ...resources]),
      dispatch,
    });
    const mm = (target, name, input, output, mode = 0, label = `mm.${name.split(".").at(-1)}`) => {
      const e = this.weight(name);
      const [N, K] = e.info.shape.map(Number);
      const params = this.uniform([N, K, mode, 0]);
      const scales = e.sbuf || this.zeros;
      return {
        kind: "mm",
        label,
        N,
        K,
        group: d.createBindGroup({ layout: this.mm.layout, entries: [target.globals, params, input, e.buf, scales, this.zeros, output, target.part].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
        reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: [target.globals, params, target.part, this.zeros, output].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
      };
    };
    const rms = (target, name, input, output, width = cfg.hidden, mode = 1, label = `rms.${name.split(".").at(-1)}`) => custom(target, "RMS", [width, f32(cfg.eps), mode, 0], [input, this.weight(name).buf, output], label, (pass) => pass.dispatchWorkgroups(this.T));
    const scale = (target, name, label = "layer.scalar") => custom(target, "RESIDUAL", [cfg.hidden, 1, 0, 0], [target.x, this.zeros, this.weight(name).buf], label, (pass) => pass.dispatchWorkgroups(this.T));
    const appendLayer = (ops, target, i, compact, { capturePrefix = false, joinPrefix = false } = {}) => {
      const type = cfg.full[i] ? "full_attention" : "sliding_attention";
      const headDim = cfg.full[i] ? cfg.globalHeadDim : cfg.localHeadDim;
      const kvHeads = cfg.kvHeads;
      const shared = compact || i >= cfg.sharedStart;
      const kvWidth = kvHeads * headDim;
      const rawK = target.qkvK || target.k || this.k;
      const rawV = target.qkvV || target.v || this.v;
      const rotary = cfg.full[i] ? Math.floor((cfg.globalHeadDim * cfg.ropeGlobalFraction) / 2) * 2 : cfg.localHeadDim;
      const n = (s) => `l.${i}.${s}`;
      const store = cfg.kvSharedLayers > 0 && !compact && !shared && cfg.layerTypes.slice(0, cfg.sharedStart).findIndex((t) => t === cfg.layerTypes[i]) >= 0 && cfg.layerTypes.slice(0, cfg.sharedStart).map(String).lastIndexOf(cfg.layerTypes[i]) === i;
      // PLE projection uses the image-injected input embedding, then is
      // combined immediately before this layer consumes it.
      ops.push(mm(target, `ple.${i}.proj`, target.input, target.pleProj, 0, "mm.ple.proj"));
      const pleEmbed = this.weight(`ple.${i}.embed`);
      if (pleEmbed.hostTable) ops.push({ kind: "copy-ple", label: "ple.embed.host", layer: i, dst: target.pleToken });
      else ops.push(custom(target, "EMBED", [cfg.pleDim, Math.ceil(cfg.pleDim / 32), f32(cfg.pleEmbeddingScale), 0], [target.pleIds || target.ids, pleEmbed.buf, pleEmbed.sbuf || this.zeros, target.pleToken], "ple.embed", (pass) => pass.dispatchWorkgroups(this.T)));
      ops.push(custom(target, "PLE", [cfg.pleDim, cfg.hidden, f32(cfg.eps), f32(cfg.pleInputScale), f32(cfg.pleProjectionScale)], [target.pleToken, target.pleProj, this.weight("ple.norm").buf, target.ple], "ple.combine", (pass) => pass.dispatchWorkgroups(this.T)));
      ops.push(rms(target, n("attn_norm"), target.x, target.h, cfg.hidden, 1, "rms.attn"));
      ops.push(mm(target, n("q"), target.h, target.q, 0, "mm.q"));
      if (!shared) {
        ops.push(mm(target, n("k"), target.h, rawK, 0, "mm.k"));
        ops.push(mm(target, n("v"), target.h, rawV, 0, "mm.v"));
      }
      const qn = this.weight(n("qn")).buf;
      const kn = shared ? this.zeros : this.weight(n("kn")).buf;
      // A compact tail computes Q for selected rows only. K/V are already in the
      // full-length shared pair, so the unused K/V groups must remain disabled.
      ops.push(custom(target, "QKV", [cfg.heads, kvHeads, headDim, rotary, cfg.full[i] ? 1 : 0, this.ropeMax * 128, shared ? 0 : 1, 0, f32(cfg.eps)], [target.q, rawK, rawV, qn, kn, this.rope, target.pos], "qkv", (pass) => pass.dispatchWorkgroups(this.T, cfg.heads + (shared ? 0 : 2 * kvHeads))));
      if (joinPrefix && !shared) ops.push({ kind: "copy-kv-join", label: `kv.join.${i}`, layer: i, width: kvWidth });
      if (capturePrefix && !shared) ops.push({ kind: "copy-kv-prefix", label: `kv.prefix.${i}`, layer: i, width: kvWidth });
      if (store) {
        const pair = this.shared.get(type);
        ops.push({ kind: "copy-kv", label: `kv.share.${type}`, srcK: target.k, srcV: target.v, dstK: pair.k, dstV: pair.v, width: kvWidth, total: joinPrefix });
      }
      const pair = shared ? this.shared.get(type) : { k: target.k, v: target.v };
      ops.push(custom(target, "ATTN", [cfg.heads, kvHeads, headDim, cfg.full[i] ? 0 : cfg.window, cfg.causal, 0, 0, 0], [target.q, pair.k, pair.v, target.ctx, target.pos], "attn", (pass) => pass.dispatchWorkgroups(this.T, cfg.heads)));
      ops.push(mm(target, n("o"), target.ctx, target.branch, 0, "mm.o"));
      ops.push(rms(target, n("attn_post_norm"), target.branch, target.x, cfg.hidden, 2, "rms.attn.post"));
      ops.push(rms(target, n("ffn_norm"), target.x, target.h, cfg.hidden, 1, "rms.ffn"));
      ops.push(mm(target, n("gate"), target.h, target.gate, 0, "mm.gate"));
      ops.push(mm(target, n("up"), target.h, target.up, 0, "mm.up"));
      const I = Number(this.weight(n("gate")).info.shape[0]);
      ops.push(custom(target, "GELU", [I, 0, 0, 0], [target.gate, target.up, target.act], "gelu", (pass) => {
        const work = Math.ceil((this.T * I) / 256);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }));
      ops.push(mm(target, n("down"), target.act, target.branch, 0, "mm.down"));
      ops.push(rms(target, n("ffn_post_norm"), target.branch, target.x, cfg.hidden, 2, "rms.ffn.post"));
      ops.push(mm(target, n("ple_gate"), target.x, target.pleGate, 0, "mm.ple.gate"));
      ops.push(custom(target, "GELU", [cfg.pleDim, 1, 0, 0], [target.pleGate, target.ple, target.pleAct], "gelu.ple", (pass) => {
        const work = Math.ceil((this.T * cfg.pleDim) / 256);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }));
      ops.push(mm(target, n("ple_out"), target.pleAct, target.branch, 0, "mm.ple.out"));
      ops.push(rms(target, n("ple_norm"), target.branch, target.x, cfg.hidden, 2, "rms.ple"));
      ops.push(scale(target, n("scalar"), "scale"));
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
    const suffixTarget = canSharePrefix ? { ...full, qkvK: this.suffixK, qkvV: this.suffixV } : null;
    const prefixSuffixOps = canSharePrefix && this.tail ? ops.slice() : null;
    for (let i = 0; i < cfg.layers; i++) {
      if (i === cfg.sharedStart) prefixOps = ops.slice();
      appendLayer(ops, full, i, false);
      if (prefixCaptureOps && i < cfg.sharedStart) appendLayer(prefixCaptureOps, full, i, false, { capturePrefix: true });
      if (prefixSuffixOps && i < cfg.sharedStart) appendLayer(prefixSuffixOps, suffixTarget, i, false, { joinPrefix: true });
    }
    if (!prefixOps) prefixOps = ops.slice();
    ops.push(rms(full, "norm", full.x, full.h, cfg.hidden, 1, "rms.final"));
    ops.push(custom(full, "GATHER", [cfg.hidden, 0, 0, 0], [full.h, this.rows, this.gathered], "gather", (pass) => pass.dispatchWorkgroups(this.R)));
    this.ops = ops;
    this.prefixOps = prefixOps;
    this.prefixCaptureOps = prefixCaptureOps;
    this.prefixSuffixOps = prefixSuffixOps;
    this.tailOps = null;
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
      };
      const tailOps = [];
      // The selected rows are gathered in caller order. This keeps duplicates and
      // arbitrary ordering exact while leaving the shared K/V pair full length.
      tailOps.push(custom(compact, "GATHER", [cfg.hidden, 0, 0, 0], [this.x, this.rows, compact.x], "tail.gather.hidden", (pass) => pass.dispatchWorkgroups(this.T)));
      tailOps.push(custom(compact, "GATHER", [cfg.hidden, 0, 0, 0], [this.input, this.rows, compact.input], "tail.gather.input", (pass) => pass.dispatchWorkgroups(this.T)));
      for (let i = cfg.sharedStart; i < cfg.layers; i++) appendLayer(tailOps, compact, i, true);
      tailOps.push(rms(compact, "norm", compact.x, compact.h, cfg.hidden, 1, "rms.final"));
      this.tailOps = tailOps;
    }
  }

  encode(enc, ops, T, prof = this.profiler) {
    let pass = prof ? null : enc.beginComputePass();
    for (const op of ops) {
      if (op.kind === "copy-images" && !this.imageSpans?.length) continue;
      if (["copy-input", "copy-kv", "copy-kv-prefix", "copy-kv-join", "copy-images", "copy-ple", "copy-embed"].includes(op.kind)) {
        pass?.end();
        if (op.kind === "copy-embed") enc.copyBufferToBuffer(this.hostPleUpload, 0, this.x, 0, T * this.cfg.hidden * 4);
        else if (op.kind === "copy-images") {
          for (const span of this.imageSpans) enc.copyBufferToBuffer(span.projection?.buffer || this.imageUpload, span.projection ? 0 : span.offset, this.x, span.start * this.cfg.hidden * 4, span.len * this.cfg.hidden * 4);
        } else if (op.kind === "copy-ple") enc.copyBufferToBuffer(this.hostPleUpload, this.hostPleOffsets[op.layer], op.dst, 0, T * this.cfg.pleDim * 4);
        else if (op.kind === "copy-input") enc.copyBufferToBuffer(op.src, 0, op.dst, 0, T * this.cfg.hidden * 4);
        else if (op.kind === "copy-kv-prefix") {
          const pair = this.prefixKV?.get(op.layer);
          if (!pair) throw new Error(`Gemma 4 missing captured prefix K/V for layer ${op.layer}`);
          const bytes = T * op.width * 4;
          enc.copyBufferToBuffer(this.k, 0, pair.k, 0, bytes);
          enc.copyBufferToBuffer(this.v, 0, pair.v, 0, bytes);
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
          const bytes = (op.total ? this.prefixLength + T : T) * op.width * 4;
          enc.copyBufferToBuffer(op.srcK, 0, op.dstK, 0, bytes);
          enc.copyBufferToBuffer(op.srcV, 0, op.dstV, 0, bytes);
        }
        pass = prof ? null : enc.beginComputePass();
        continue;
      }
      if (prof) pass = prof.pass(enc, op.label);
      pass.setPipeline(op.pipeline || this.mm.mm[1]);
      pass.setBindGroup(0, op.group);
      if (op.kind === "mm") {
        const kernel = this.gpuKernel === "auto" ? selectedMatmulKernel(this.tuning.selection, op.N, op.K, T) : this.gpuKernel;
        encodeMatmul(pass, this.tuning.pipelines[kernel] || this.tuning.pipelines.generic, op, T);
      } else op.dispatch(pass);
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
    const sharedPrefixTokens = sharedImagePrefix(sequences);
    this.lastPrefixReuse = { enabled: false, reusedTokens: 0, totalTokens: sequences.reduce((sum, sequence) => sum + (sequence.ids?.length || 0), 0), sharedPrefixTokens };
    if (!sharedPrefixTokens || !this.prefixCaptureOps || this.profiler) return null;
    this.lastProfile = null;
    try {
      return await withGpuErrors(this.runtime, () => this.forwardSharedRows(sequences, sharedPrefixTokens), "running Gemma 4 shared-prefix inference");
    } finally {
      this.imageSpans = [];
      this.hostPleUpload = null;
      this.hostPleOffsets = null;
    }
  }

  async forwardSharedRows(sequences, sharedPrefixTokens) {
    const cfg = this.cfg;
    const q = this.device.queue;
    const owned = [];
    const plePins = [];
    const prefixKV = new Map();
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
      await this.ensureRope(maxTokens);
      this.ensure(maxTokens);
      this.ensureTail(1);
      if (!this.prefixCaptureOps || !this.prefixSuffixOps || !this.tailOps) return null;
      for (let layer = 0; layer < cfg.sharedStart; layer++) {
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
      const prefixIds = sequences[0].ids.subarray(0, sharedPrefixTokens);
      const prefixPleIds = prefixIds.slice();
      for (const span of imageSpans) prefixPleIds.fill(0, span.start, span.start + span.len);
      const parameterWords = 4 + 3 * sharedPrefixTokens + sequences.reduce((sum, sequence) => sum + 11 + 3 * (sequence.ids.length - sharedPrefixTokens), 0);
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
      this.T = sharedPrefixTokens;
      this.R = sharedPrefixTokens;
      const prefixPositions = Uint32Array.from({ length: sharedPrefixTokens }, (_, i) => i);
      stage(this.globals, new Uint32Array([sharedPrefixTokens, sharedPrefixTokens, sharedPrefixTokens, 0]));
      stage(this.ids, prefixIds);
      stage(this.pleIds, prefixPleIds);
      stage(this.pos, prefixPositions);
      const prefixPleStarted = performance.now();
      const prefixPle = await this.hostPleRows(prefixIds, prefixPleIds, owned, plePins, cfg.sharedStart);
      pleCacheHits += Number(prefixPle?.cacheHit || false);
      pleMs += performance.now() - prefixPleStarted;
      this.encode(encoder, this.prefixCaptureOps, sharedPrefixTokens, null);

      const hiddenBytes = cfg.hidden * 4;
      const outputBytes = sequences.length * hiddenBytes;
      const flags = imageSpans.filter((span) => span.projection).map((span) => span.projection.status);
      const readBytes = outputBytes + flags.length * 4;
      this.checkBuffer(readBytes, "shared-prefix output rows and projection status");
      output = this.device.createBuffer({ label: "Gemma 4 shared-prefix output rows", size: Math.max(16, outputBytes), usage: U.COPY_SRC | U.COPY_DST });
      outputRead = this.device.createBuffer({ label: "Gemma 4 shared-prefix readback", size: Math.max(16, readBytes), usage: U.MAP_READ | U.COPY_DST });
      owned.push(output, outputRead);
      this.imageSpans = [];
      for (let index = 0; index < sequences.length; index++) {
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
        stage(this.tail.pos, new Uint32Array([selectedPosition]));
        stage(this.tail.ids, new Uint32Array([pleIds[length - 1]]));
        const pleStarted = performance.now();
        const tailIds = new Uint32Array([pleIds[length - 1]]);
        const suffixPle = await this.hostPleRows(suffix, pleIds, owned, plePins, cfg.sharedStart, tailIds);
        pleCacheHits += Number(suffixPle?.cacheHit || false);
        pleMs += performance.now() - pleStarted;
        this.encode(encoder, this.prefixSuffixOps, length, null);
        this.T = 1;
        this.encode(encoder, this.tailOps, 1, null);
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
      this.lastPrefixReuse = { enabled: true, reusedTokens: (sequences.length - 1) * sharedPrefixTokens, totalTokens, sharedPrefixTokens };
      this.lastPle = { ms: pleMs, cacheHit: pleCacheHits > 0, cacheHits: pleCacheHits };
      return result;
    } finally {
      if (submitted && !completed) await q.onSubmittedWorkDone().catch(() => {});
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
    } finally {
      this.imageSpans = [];
    }
  }

  async forwardRows(inputIds, rows, imageSpans) {
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
    if (useTail) {
      q.writeBuffer(this.tail.globals, 0, new Uint32Array([selected.length, selected.length, T, 0]));
      q.writeBuffer(this.tail.pos, 0, new Uint32Array(selected));
      q.writeBuffer(this.tail.ids, 0, Uint32Array.from(selected, (row) => pleIds[row]));
    }
    const pleOwned = [];
    const plePins = [];
    try {
      if (this.cfg.hostPle === true) {
        const pleStarted = performance.now();
        const prefixLayers = useTail ? this.cfg.sharedStart : this.cfg.layers;
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
      const runOps = useTail ? this.prefixOps : this.ops;
      const plans = (longChunks ? chunks(runOps, YIELD_CHUNKS) : [runOps]).map((ops, i) => ({ ops, rows: T, flushBefore: i > 0 }));
      if (useTail) plans.push({ ops: this.tailOps, rows: selected.length, flushBefore: false });
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
      if (profiler) this.lastProfile = await profiler.collect();
      await this.readback.mapAsync(GPUMapMode.READ, 0, readBytes);
      const out = mappedImageRows(this.readback, bytes, flags.length);
      return out;
    } finally {
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
