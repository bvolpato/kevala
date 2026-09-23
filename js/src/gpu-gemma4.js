// Gemma 4 E2B/E4B text trunk on WebGPU.
//
// E2B and E4B are dense models. Their distinctive GPU requirements are the packed
// per-layer embeddings (PLE), 256-wide local heads, 512-wide proportional-RoPE
// global heads, and shared KV states in the final layers. This implementation keeps
// all large tensors and the embedding lookup on the device. The CPU receives only
// the normalized rows requested by the scorer.

import { GpuWeights, pipeline, matmulPipelines, encodeMatmul, SPLIT_SCRATCH, YIELD_TOKENS, YIELD_CHUNKS, chunks, endChunk } from "./gpu.js";
import { calibrateMatmul, selectedMatmulKernel } from "./gpu-tuning.js";

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

function textConfig(header) {
  const c = header?.config || {};
  return c.text_config || c.textConfig || c;
}

/** Converts either the flat native config or HF's nested text_config to GPU fields. */
export function gemma4Config(header) {
  const c = textConfig(header);
  const layers = Number(c.num_hidden_layers ?? c.layers ?? c.num_layers);
  const rawLayerTypes = c.layer_types ?? c.layerTypes;
  const layerTypes = Array.isArray(rawLayerTypes) ? rawLayerTypes.map((t) => String(t === "global_attention" ? "full_attention" : t)) : null;
  const full = layerTypes?.map((t) => t === "full_attention");
  const hidden = Number(c.hidden_size ?? c.hidden);
  const localHeadDim = Number(c.head_dim ?? 256);
  const globalHeadDim = Number(c.global_head_dim ?? c.globalHeadDim ?? 512);
  const kvHeads = Number(c.num_key_value_heads ?? c.kv_heads);
  const globalKvHeads = kvHeads; // Separate K/V projections use the same head count in both layer types.
  const rope = c.rope_parameters || c.ropeParameters || {};
  const localRope = rope.sliding_attention || rope.sliding || {};
  const globalRope = rope.full_attention || rope.full || {};
  const layersShared = Number(c.num_kv_shared_layers ?? c.kv_shared_layers ?? 0);
  const sharedStart = Math.max(0, layers - layersShared);
  const boolField = (key) => {
    if (!Object.hasOwn(c, key)) return false;
    if (typeof c[key] !== "boolean") throw new Error(`Gemma 4 GPU config ${key} must be boolean`);
    return c[key];
  };
  if (Object.hasOwn(c, "hidden_activation") && c.hidden_activation !== "gelu_pytorch_tanh") throw new Error("Gemma 4 GPU config supports only hidden_activation=gelu_pytorch_tanh");
  const attentionBias = boolField("attention_bias");
  const attentionKEqV = boolField("attention_k_eq_v");
  const enableMoe = boolField("enable_moe_block");
  if (attentionBias) throw new Error("Gemma 4 GPU config does not support attention_bias=true");
  if (attentionKEqV) throw new Error("Gemma 4 GPU config does not support attention_k_eq_v=true");
  if (enableMoe) throw new Error("Gemma 4 GPU config does not support enable_moe_block=true");
  if (!Number.isSafeInteger(layers) || layers <= 0 || !Number.isSafeInteger(hidden) || hidden <= 0 || !Number.isSafeInteger(Number(c.intermediate_size ?? c.intermediate)) || Number(c.intermediate_size ?? c.intermediate) <= 0) throw new Error("Gemma 4 GPU config has invalid model dimensions");
  if (!Number.isSafeInteger(Number(c.num_attention_heads ?? c.heads)) || Number(c.num_attention_heads ?? c.heads) <= 0 || !Number.isSafeInteger(kvHeads) || kvHeads <= 0 || !Number.isSafeInteger(globalKvHeads) || globalKvHeads <= 0) throw new Error("Gemma 4 GPU config has invalid attention head count");
  if (Number(c.num_attention_heads ?? c.heads) % kvHeads || Number(c.num_attention_heads ?? c.heads) % globalKvHeads) throw new Error("Gemma 4 GPU config requires attention heads divisible by KV heads");
  if (!Number.isSafeInteger(localHeadDim) || localHeadDim <= 0 || localHeadDim > 256 || localHeadDim % 2 || !Number.isSafeInteger(globalHeadDim) || globalHeadDim <= 0 || globalHeadDim > 512 || globalHeadDim % 2) throw new Error("Gemma 4 GPU config supports even local head dimensions up to 256 and global dimensions up to 512");
  if (!Number.isSafeInteger(layersShared) || layersShared < 0 || layersShared >= layers) throw new Error("Gemma 4 GPU config has invalid KV sharing count");
  if (!Number.isSafeInteger(Number(c.max_position_embeddings ?? c.max_position ?? 131072)) || Number(c.max_position_embeddings ?? c.max_position ?? 131072) <= 0) throw new Error("Gemma 4 GPU config has invalid position limit");
  const pleDim = Number(c.hidden_size_per_layer_input ?? c.ple_dim ?? 256);
  const vocab = Number(c.vocab_size ?? c.vocab ?? 262144);
  const pleVocab = Number(c.vocab_size_per_layer_input ?? c.ple_vocab ?? vocab);
  if (!Number.isSafeInteger(pleDim) || pleDim <= 0 || !Number.isSafeInteger(vocab) || vocab <= 0 || !Number.isSafeInteger(pleVocab) || pleVocab <= 0) throw new Error("Gemma 4 GPU config has invalid PLE or vocabulary dimensions");
  const intermediate = Number(c.intermediate_size ?? c.intermediate);
  if (hidden % 4 || intermediate % 4 || pleDim % 4) throw new Error("Gemma 4 GPU config requires hidden, intermediate, and PLE dimensions divisible by four");
  if (c.use_bidirectional_attention != null) throw new Error("Gemma 4 GPU config does not support use_bidirectional_attention for direct scoring");
  if (!layerTypes || layerTypes.length !== layers) throw new Error("Gemma 4 GPU config requires layer_types for every decoder layer");
  if (layerTypes.some((t) => t !== "sliding_attention" && t !== "full_attention")) throw new Error("Gemma 4 GPU config has invalid layer_types");
  if (layerTypes.at(-1) !== "full_attention") throw new Error("Gemma 4 GPU config requires the final decoder layer to be full_attention");
  const prefixTypes = new Set(layerTypes.slice(0, sharedStart));
  for (const type of new Set(layerTypes.slice(sharedStart))) {
    if (!prefixTypes.has(type)) throw new Error(`Gemma 4 GPU config has no non-shared KV source for ${type}`);
  }
  const window = Number(c.sliding_window ?? c.window ?? 512);
  if (!Number.isSafeInteger(window) || window <= 0) throw new Error("Gemma 4 GPU config has invalid sliding window");
  const localRopeType = typeof localRope.rope_type === "string" ? localRope.rope_type : "default";
  const globalRopeType = typeof globalRope.rope_type === "string" ? globalRope.rope_type : "proportional";
  if (localRopeType !== "default" || globalRopeType !== "proportional") throw new Error(`Gemma 4 GPU config supports only sliding rope_type=default and full rope_type=proportional (got ${localRopeType}/${globalRopeType})`);
  const globalFraction = Number(c.global_partial_rotary ?? c.globalPartialRotary ?? globalRope.partial_rotary_factor ?? globalRope.partialRotaryFactor ?? 0.25);
  if (!Number.isFinite(globalFraction) || globalFraction < 0 || globalFraction > 1 || Math.floor((globalHeadDim * globalFraction) / 2) * 2 > 256) throw new Error("Gemma 4 GPU config has unsupported global proportional RoPE");
  return {
    hidden,
    layers,
    intermediate,
    heads: Number(c.num_attention_heads ?? c.heads),
    kvHeads,
    globalKvHeads,
    localHeadDim,
    globalHeadDim,
    maxHeadDim: Math.max(localHeadDim, globalHeadDim),
    maxQ: Number(c.num_attention_heads ?? c.heads) * Math.max(localHeadDim, globalHeadDim),
    maxKV: Math.max(kvHeads, globalKvHeads) * Math.max(localHeadDim, globalHeadDim),
    layerTypes,
    full,
    sharedStart,
    kvSharedLayers: layersShared,
    window,
    maxPosition: Number(c.max_position_embeddings ?? c.max_position ?? 131072),
    vocab,
    pleDim,
    pleVocab,
    eps: Number(c.rms_norm_eps ?? c.eps ?? 1e-6),
    embeddingScale: Number(c.embedding_scale ?? bf16Round(Math.sqrt(hidden))),
    pleEmbeddingScale: Number(c.ple_embedding_scale ?? bf16Round(Math.sqrt(pleDim))),
    pleInputScale: Number(c.ple_input_scale ?? Math.SQRT1_2),
    pleProjectionScale: Number(c.ple_projection_scale ?? 1 / Math.sqrt(hidden)),
    useDoubleWideMlp: !!(c.use_double_wide_mlp ?? c.useDoubleWideMlp),
    ropeLocalTheta: Number(localRope.rope_theta ?? localRope.ropeTheta ?? c.rope_theta ?? 10000),
    ropeGlobalTheta: Number(globalRope.rope_theta ?? globalRope.ropeTheta ?? c.rope_theta ?? 1000000),
    ropeGlobalFraction: globalFraction,
    causal: 1,
  };
}

/** Dense Gemma 4 text trunk. The caller owns CPU tokenization and readout labels. */
export class GpuGemma4 {
  constructor(gpu, layout, cfg) {
    U = GPUBufferUsage;
    this.device = gpu.device;
    this.wgsl = gpu.wgsl;
    this.name = gpu.name;
    this.gpuKernel = gpu.kernel || "auto";
    this.cfg = cfg;
    this.weights = new GpuWeights(gpu.device, layout);
    this.tensors = this.weights.tensors;
    for (const t of this.weights.header.tensors) {
      if (t.dtype === "q8" && Number(t.block) !== 32) throw new Error(`Gemma 4 WebGPU requires block-32 q8 tensors; ${t.name} declares block ${t.block}`);
      if (t.dtype === "q8" && (!Array.isArray(t.shape) || t.shape.length !== 2 || Number(t.shape[1]) % 32)) throw new Error(`Gemma 4 WebGPU requires q8 matrix columns divisible by 32; ${t.name} has shape ${JSON.stringify(t.shape)}`);
      this.checkBuffer(t.size, t.name);
      if (t.scales_size) this.checkBuffer(t.scales_size, `${t.name}.scales`);
    }
    this.capacity = 0;
    this.owned = [];
    this.uniforms = [];
    this.ropeMax = 0;
    this.shared = new Map();
    this.tailCapacity = 0;
    this.tail = null;
  }

  write(dst, bytes) {
    this.weights.write(dst, bytes);
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
    const b = this.device.createBuffer({ label, size: Math.max(16, floats * 4), usage: U.STORAGE | extra });
    this.owned.push(b);
    return b;
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
    const built = await Promise.all(GEMMA4_KERNELS.map(async ([key, name]) => [key, await pipeline(d, this.wgsl(name), name)]));
    this.p = Object.fromEntries(built);
    this.tuning = await calibrateMatmul(d, this.wgsl, this.weights, { kernel: this.gpuKernel, pipelines: { generic } });
    this.mm = this.tuning.pipelines.generic;
    this.globals = d.createBuffer({ label: "Gemma 4 globals", size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.owned.push(this.globals);
    // A zero bias and fallback scale buffer. Normal Gemma projection tensors are Q8;
    // the fallback keeps bind groups valid if a future converter emits an unscaled bias.
    this.zeros = this.staticBuffer(new Float32Array(16384), "Gemma 4 zero constants");
    this.owned.push(this.zeros);
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
    this.tailBuffers = [];
    this.tailCapacity = 0;
    this.tail = null;
    this.ops = null;
    this.prefixOps = null;
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
    const KV = Math.max(cfg.kvHeads, cfg.globalKvHeads) * cfg.globalHeadDim;
    const maxI = cfg.intermediate * (cfg.useDoubleWideMlp ? 2 : 1);
    const P = cfg.pleDim;
    this.x = make(cap * D, U.COPY_DST | U.COPY_SRC, "Gemma 4 hidden states");
    this.input = make(cap * D, U.COPY_DST, "Gemma 4 input embedding");
    this.h = make(cap * D, 0, "Gemma 4 normalized states");
    this.branch = make(cap * D, 0, "Gemma 4 residual branch");
    this.q = make(cap * Q, 0, "Gemma 4 queries");
    this.k = make(cap * KV, U.COPY_SRC, "Gemma 4 keys");
    this.v = make(cap * KV, U.COPY_SRC, "Gemma 4 values");
    this.ctx = make(cap * Q, 0, "Gemma 4 attention context");
    this.gate = make(cap * maxI, 0, "Gemma 4 MLP gate");
    this.up = make(cap * maxI, 0, "Gemma 4 MLP up");
    this.act = make(cap * maxI, 0, "Gemma 4 MLP activation");
    this.pleToken = make(cap * P, 0, "Gemma 4 PLE token embedding");
    this.pleProj = make(cap * P, 0, "Gemma 4 PLE projection");
    this.ple = make(cap * P, 0, "Gemma 4 PLE input");
    this.pleGate = make(cap * P, 0, "Gemma 4 PLE gate");
    this.pleAct = make(cap * P, 0, "Gemma 4 PLE activation");
    this.ids = make(cap, U.COPY_DST, "Gemma 4 token ids");
    this.pos = make(cap, U.COPY_DST, "Gemma 4 token positions");
    this.rows = make(cap, U.COPY_DST, "Gemma 4 output rows");
    this.gathered = make(cap * D, U.COPY_SRC, "Gemma 4 gathered rows");
    this.checkBuffer(cap * D * 4, "Gemma 4 readback");
    this.readback = d.createBuffer({ label: "Gemma 4 readback", size: Math.max(16, cap * D * 4), usage: U.MAP_READ | U.COPY_DST });
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
    this.owned.push(globals);
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
      pleToken: make(cap * P, 0, "Gemma 4 compact PLE token embedding"),
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
    const appendLayer = (ops, target, i, compact) => {
      const type = cfg.full[i] ? "full_attention" : "sliding_attention";
      const headDim = cfg.full[i] ? cfg.globalHeadDim : cfg.localHeadDim;
      const kvHeads = cfg.full[i] ? cfg.globalKvHeads : cfg.kvHeads;
      const shared = compact || i >= cfg.sharedStart;
      const kvWidth = kvHeads * headDim;
      const rotary = cfg.full[i] ? Math.floor((cfg.globalHeadDim * cfg.ropeGlobalFraction) / 2) * 2 : cfg.localHeadDim;
      const n = (s) => `l.${i}.${s}`;
      const store = cfg.kvSharedLayers > 0 && !compact && !shared && cfg.layerTypes.slice(0, cfg.sharedStart).findIndex((t) => t === cfg.layerTypes[i]) >= 0 && cfg.layerTypes.slice(0, cfg.sharedStart).map(String).lastIndexOf(cfg.layerTypes[i]) === i;
      // PLE projection is computed from the original scaled token embedding, then
      // combined immediately before this layer consumes it.
      ops.push(mm(target, `ple.${i}.proj`, target.input, target.pleProj, 0, "mm.ple.proj"));
      const pleEmbed = this.weight(`ple.${i}.embed`);
      ops.push(custom(target, "EMBED", [cfg.pleDim, Math.ceil(cfg.pleDim / 32), f32(cfg.pleEmbeddingScale), 0], [target.ids, pleEmbed.buf, pleEmbed.sbuf || this.zeros, target.pleToken], "ple.embed", (pass) => pass.dispatchWorkgroups(this.T)));
      ops.push(custom(target, "PLE", [cfg.pleDim, cfg.hidden, f32(cfg.eps), f32(cfg.pleInputScale), f32(cfg.pleProjectionScale)], [target.pleToken, target.pleProj, this.weight("ple.norm").buf, target.ple], "ple.combine", (pass) => pass.dispatchWorkgroups(this.T)));
      ops.push(rms(target, n("attn_norm"), target.x, target.h, cfg.hidden, 1, "rms.attn"));
      ops.push(mm(target, n("q"), target.h, target.q, 0, "mm.q"));
      if (!shared) {
        ops.push(mm(target, n("k"), target.h, target.k, 0, "mm.k"));
        ops.push(mm(target, n("v"), target.h, target.v, 0, "mm.v"));
      }
      const qn = this.weight(n("qn")).buf;
      const kn = shared ? this.zeros : this.weight(n("kn")).buf;
      // A compact tail computes Q for selected rows only. K/V are already in the
      // full-length shared pair, so the unused K/V groups must remain disabled.
      ops.push(custom(target, "QKV", [cfg.heads, kvHeads, headDim, rotary, cfg.full[i] ? 1 : 0, this.ropeMax * 128, shared ? 0 : 1, 0, f32(cfg.eps)], [target.q, target.k || this.k, target.v || this.v, qn, kn, this.rope, target.pos], "qkv", (pass) => pass.dispatchWorkgroups(this.T, cfg.heads + (shared ? 0 : 2 * kvHeads))));
      if (store) {
        const pair = this.shared.get(type);
        ops.push({ kind: "copy-kv", label: `kv.share.${type}`, srcK: target.k, srcV: target.v, dstK: pair.k, dstV: pair.v, width: kvWidth });
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
      pos: this.pos,
    };
    const ops = [];
    const embed = this.weight("embed");
    const embedWidth = Number(embed.info.shape[1]);
    ops.push(custom(full, "EMBED", [embedWidth, Math.ceil(embedWidth / 32), f32(cfg.embeddingScale), 0], [full.ids, embed.buf, embed.sbuf || this.zeros, full.x], "embed", (pass) => pass.dispatchWorkgroups(this.T)));
    ops.push({ kind: "copy-input", label: "embed.copy", src: this.x, dst: this.input });
    let prefixOps = null;
    for (let i = 0; i < cfg.layers; i++) {
      if (i === cfg.sharedStart) prefixOps = ops.slice();
      appendLayer(ops, full, i, false);
    }
    if (!prefixOps) prefixOps = ops.slice();
    ops.push(rms(full, "norm", full.x, full.h, cfg.hidden, 1, "rms.final"));
    ops.push(custom(full, "GATHER", [cfg.hidden, 0, 0, 0], [full.h, this.rows, this.gathered], "gather", (pass) => pass.dispatchWorkgroups(this.R)));
    this.ops = ops;
    this.prefixOps = prefixOps;
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
      if (op.kind === "copy-input" || op.kind === "copy-kv") {
        pass?.end();
        if (op.kind === "copy-input") enc.copyBufferToBuffer(op.src, 0, op.dst, 0, T * this.cfg.hidden * 4);
        else {
          const bytes = T * op.width * 4;
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

  async forward(inputIds, rows = null) {
    if (!this.initialized) throw new Error("Gemma 4 GPU trunk is not initialized");
    const ids = inputIds instanceof Uint32Array ? inputIds : Uint32Array.from(inputIds);
    const T = ids.length;
    if (!T) throw new Error("Gemma 4 GPU forward needs at least one token");
    if (ids.some((id) => id >= this.cfg.vocab || id >= this.cfg.pleVocab)) throw new Error("Gemma 4 token id is outside an embedding vocabulary");
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
    q.writeBuffer(this.pos, 0, positions);
    q.writeBuffer(this.rows, 0, new Uint32Array(selected));
    const useTail = !allRows && !!this.tailOps && !!this.prefixOps;
    if (useTail) {
      q.writeBuffer(this.tail.globals, 0, new Uint32Array([selected.length, selected.length, T, 0]));
      q.writeBuffer(this.tail.pos, 0, new Uint32Array(selected));
      q.writeBuffer(this.tail.ids, 0, Uint32Array.from(selected, (row) => ids[row]));
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
    enc.copyBufferToBuffer(useTail ? this.tail.h : this.gathered, 0, this.readback, 0, bytes);
    q.submit([enc.finish()]);
    if (profiler) this.lastProfile = await profiler.collect();
    await this.readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(this.readback.getMappedRange(0, bytes).slice(0));
    this.readback.unmap();
    return out;
  }
}
