// The transformer trunk on WebGPU: 28 ModernBERT layers and 2 decision-head layers, int8 weights
// widened inside the matmul tiles. The coordinator (WebAssembly) embeds tokens before and scores
// markers after; everything in between runs on the GPU.
//
// The kernels themselves are WGSL sources in the Rust crate (crates/kevala/src/wgsl), specialized
// by the WebAssembly binary: `gpu.wgsl(kernel, spec)` returns the source to compile. This file only
// creates buffers, pipelines and bind groups, and records the dispatches.

// Narrow matmuls at short lengths launch too few tiles to fill a GPU (a 1024-wide output at 64
// tokens is 16 workgroups), so K is split across up to 8 workgroups whose partial tiles a second
// pass sums. The kernel (wgsl/splits.wgsl) and the host compute the same split count.
const SPLIT_TARGET = 128;
/** Host side of `mm_splits`. */
export function mmSplits(T, N, K, target = SPLIT_TARGET, bm = 64, bn = 64) {
  const tiles = Math.ceil(N / bn) * Math.ceil(T / bm);
  if (tiles >= 96) return 1;
  return Math.max(1, Math.min(8, Math.ceil(target / tiles), Math.floor(K / 128)));
}

/** Rows per thread for an input of T tokens: a short input gets a shorter tile (16 R rows). */
export const rowsPerThread = (T) => (T >= 64 ? 4 : Math.max(1, Math.ceil(T / 16)));

/**
 * Encodes a matmul op (bind groups `group` and `reduce`), splitting K when it helps. The kernel
 * variant covers 16 R rows and 64 J columns per workgroup.
 */
export function dispatchMatmul(pass, pMatmul, pReduce, op, T, target = SPLIT_TARGET, R = 4, J = 1) {
  const bm = 16 * R;
  const bn = 64 * J;
  const splits = mmSplits(T, op.N, op.K, target, bm, bn);
  pass.setPipeline(pMatmul);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / bn), Math.ceil(T / bm), splits);
  if (splits > 1) {
    const n = Math.ceil((T * op.N) / 256);
    pass.setPipeline(pReduce);
    pass.setBindGroup(0, op.reduce);
    pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
  }
}

/** Encodes a matmul with the kernel variant for T. `pipes` comes from `matmulPipelines`. */
export function encodeMatmul(pass, pipes, op, T) {
  const R = rowsPerThread(T);
  dispatchMatmul(pass, pipes.mm[R], pipes.reduce[R], op, T, SPLIT_TARGET, R);
}

/** The matmul bind group layout, explicit so every kernel variant shares one bind group. */
export function matmulLayout(device) {
  const e = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  return device.createBindGroupLayout({
    label: "matmul",
    entries: [e(0, "uniform"), e(1, "uniform"), e(2, "read-only-storage"), e(3, "read-only-storage"), e(4, "read-only-storage"), e(5, "read-only-storage"), e(6, "storage"), e(7, "storage")],
  });
}

/** The split-K reduce layout, explicit for the same reason. */
export function reduceLayout(device) {
  const e = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  return device.createBindGroupLayout({ label: "reduce", entries: [e(0, "uniform"), e(1, "uniform"), e(2, "read-only-storage"), e(3, "read-only-storage"), e(4, "storage")] });
}

/**
 * Builds the matmul variants (1 to 4 rows per thread) and their split-K reduces from the
 * binary's kernels; resolves to { layout, reduceLayout, mm: [, R1..R4], reduce: [, R1..R4] }.
 * With `shader-f16` the tiles live in workgroup memory as f16, which halves the traffic that
 * limits this kernel (1.6-1.75x faster on Apple GPUs); products and sums stay f32, and the
 * rounding (about 3e-4 relative) is far below the int8 weight quantization.
 */
export async function matmulPipelines(device, wgsl) {
  const layout = matmulLayout(device);
  const rlayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const rpl = device.createPipelineLayout({ bindGroupLayouts: [rlayout] });
  const f16 = device.features.has("shader-f16");
  const mm = [];
  const reduce = [];
  await Promise.all(
    [1, 2, 3, 4].flatMap((rows) => [
      pipeline(device, wgsl("matmul", { f16, rows }), `matmul_r${rows}`, pl).then((p) => (mm[rows] = p)),
      pipeline(device, wgsl("reduce", { rows }), `reduce_r${rows}`, rpl).then((p) => (reduce[rows] = p)),
    ]),
  );
  return { layout, reduceLayout: rlayout, mm, reduce, f16 };
}

/** Floats of scratch the split-K partials need at most (tiles < 96, at most 8 splits). */
export const SPLIT_SCRATCH = 8 * 96 * 64 * 64;

function parseSubpackHeader(prefix) {
  const dv = new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength);
  const n = dv.getUint32(8, true);
  return JSON.parse(new TextDecoder().decode(prefix.subarray(16, 16 + n)));
}

/** Detects a usable GPU. Returns null when WebGPU is missing or the adapter is too small. */
// Compilers differ on the directive WGSL requires for subgroups (naga, Firefox's, rejects it in
// 2026), so the subgroup kernels are used only when this compiles.
const SUBGROUP_PROBE = "enable subgroups;\n@compute @workgroup_size(32) fn main() { _ = subgroupAdd(1u); }\n";

async function compiles(device, code) {
  const info = await device.createShaderModule({ code }).getCompilationInfo?.();
  return !info?.messages.some((m) => m.type === "error");
}

/**
 * A WebGPU device with the optional features and larger limits the adapter offers, or an error
 * saying why there is none. `baseline` asks for no optional feature and the default limits, the
 * way the weakest WebGPU device runs (for testing those kernel paths on a strong one).
 */
export async function requestDevice({ baseline = false, powerPreference = "high-performance" } = {}) {
  const where = typeof window === "undefined" ? "worker" : "page";
  if (typeof navigator === "undefined" || !navigator.gpu) throw new Error(`this browser has no WebGPU in a ${where} (navigator.gpu is missing)`);
  const adapter = (await navigator.gpu.requestAdapter({ powerPreference })) || (await navigator.gpu.requestAdapter());
  if (!adapter) throw new Error("the browser offers no WebGPU adapter: WebGPU may be turned off, or the GPU or its driver blocklisted");
  if (adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) throw new Error("the browser selected a software WebGPU adapter instead of a hardware GPU");
  const want = baseline
    ? {}
    : {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
        maxComputeWorkgroupStorageSize: Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768),
      };
  // timestamps only feed the optional profiler (`Kevala.load({ profile: true })`)
  const optional = baseline ? [] : ["timestamp-query", "shader-f16", "subgroups"];
  const requiredFeatures = optional.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredLimits: want, requiredFeatures });
  const gpu = { device, adapter, powerPreference, lost: null };
  device.lost.then((info) => (gpu.lost = info));
  const info = adapter.info || {};
  let subgroups;
  try {
    subgroups = device.features.has("subgroups") && (await compiles(device, SUBGROUP_PROBE));
  } catch (e) {
    device.destroy();
    throw e;
  }
  // some kernels map one lane to each of 32 keys: they need subgroups of exactly 32 lanes, and
  // the attention one more workgroup memory than the 16 KB every device has
  const subgroup32 = subgroups && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32 && device.limits.maxComputeWorkgroupStorageSize >= 24576;
  // others add across groups of 4 lanes, so any subgroup of at least 4 will do
  const subgroup4 = subgroups && info.subgroupMinSize >= 4;
  return Object.assign(gpu, { subgroup32, subgroup4, name: [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "WebGPU" });
}

/** Preserve allocation errors before a later write or readback reports an invalid buffer. */
export async function withGpuErrors(gpu, operation, phase) {
  const d = gpu.device;
  d.pushErrorScope("validation");
  d.pushErrorScope("internal");
  d.pushErrorScope("out-of-memory");
  let value, failure;
  try {
    value = await operation();
  } catch (e) {
    failure = e;
  }
  const scopes = await Promise.allSettled([d.popErrorScope(), d.popErrorScope(), d.popErrorScope()]);
  const [oom, internal, invalid] = scopes.map((s) => (s.status === "fulfilled" ? s.value : null));
  const scoped = oom || internal || invalid;
  let message;
  if (oom) message = `out of GPU memory while ${phase}: ${oom.message}`;
  else if (gpu.lost) message = `device lost while ${phase}: ${gpu.lost.message || gpu.lost.reason}`;
  else if (scoped) message = `${phase}: ${scoped.message}`;
  if (message) throw Object.assign(new Error(`WebGPU: ${message}`), { code: "WEBGPU_INIT" });
  if (failure) throw failure;
  const rejected = scopes.find((s) => s.status === "rejected");
  if (rejected) throw Object.assign(new Error(`WebGPU: ${phase}: ${rejected.reason?.message || rejected.reason}`), { code: "WEBGPU_INIT" });
  return value;
}

let U;


/**
 * Trunk weights streamed straight into GPU buffers, one buffer per tensor (and one for q8
 * scales), as the pack goes by.
 */
export class GpuWeights {
  constructor(device, layout) {
    U = GPUBufferUsage;
    this.device = device;
    this.header = parseSubpackHeader(layout.prefix);
    this.tensors = new Map();
    // destination ranges -> tensor parts, for routing streamed fragments
    this.parts = [];
    for (const t of this.header.tensors) {
      const e = { info: t, data: null, scales: null, filled: 0, need: t.size + (t.scales_size || 0), buf: null, sbuf: null, host: null };
      this.tensors.set(t.name, e);
      this.parts.push({ lo: t.offset, hi: t.offset + t.size, e, part: "data" });
      if (t.dtype === "q8") this.parts.push({ lo: t.scales_offset, hi: t.scales_offset + t.scales_size, e, part: "scales" });
    }
    this.parts.sort((a, b) => a.lo - b.lo);
    this.cursor = 0;
    this.keepHost = new Set();
  }

  /** Receives a fragment of the trunk sub-pack. */
  write(dst, bytes) {
    let off = 0;
    while (off < bytes.byteLength) {
      const at = dst + off;
      while (this.cursor < this.parts.length && this.parts[this.cursor].hi <= at) this.cursor++;
      const part = this.parts[this.cursor];
      if (!part || part.lo > at) {
        // padding or the header: skip to the next part
        const next = part ? part.lo : Infinity;
        off += Math.min(bytes.byteLength - off, next - at);
        continue;
      }
      const n = Math.min(bytes.byteLength - off, part.hi - at);
      const e = part.e;
      const key = part.part;
      if (!e[key]) e[key] = new Uint8Array(part.hi - part.lo);
      e[key].set(bytes.subarray(off, off + n), at - part.lo);
      e.filled += n;
      off += n;
      if (e.filled === e.need) this.upload(e);
    }
  }

  upload(e) {
    const mk = (bytes, label) => {
      const size = Math.max(16, Math.ceil(bytes.byteLength / 16) * 16);
      const b = this.device.createBuffer({ label, size, usage: U.STORAGE | U.COPY_DST });
      this.device.queue.writeBuffer(b, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength & ~3);
      return b;
    };
    e.buf = mk(e.data, e.info.name);
    if (e.scales) e.sbuf = mk(e.scales, `${e.info.name}.scales`);
    // small f32 tensors some kernels combine on the host (Kev's gate projections)
    if (this.keepHost.has(e.info.name)) e.host = new Float32Array(e.data.buffer, e.data.byteOffset, e.data.byteLength / 4);
    e.data = e.scales = null;
  }

  get(name) {
    const e = this.tensors.get(name);
    if (!e) throw new Error(`GPU trunk has no tensor ${name}`);
    return e;
  }

  missing() {
    return [...this.tensors.values()].filter((e) => !e.buf).map((e) => e.info.name);
  }
}

/**
 * Times every dispatch in its own compute pass and totals the milliseconds by label. Slower than
 * a normal pass (one pass per dispatch), so only for finding where the time goes.
 */
export class Profiler {
  constructor(device, max = 2048) {
    this.device = device;
    this.max = max;
    this.qs = device.createQuerySet({ type: "timestamp", count: 2 * max });
    this.resolved = device.createBuffer({ size: 16 * max, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.read = device.createBuffer({ size: 16 * max, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.labels = [];
  }

  pass(enc, label) {
    const i = this.labels.length;
    if (i >= this.max) return enc.beginComputePass();
    this.labels.push(label);
    return enc.beginComputePass({ timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
  }

  finish(enc) {
    const n = this.labels.length;
    if (!n) return;
    enc.resolveQuerySet(this.qs, 0, 2 * n, this.resolved, 0);
    enc.copyBufferToBuffer(this.resolved, 0, this.read, 0, 16 * n);
  }

  async collect() {
    const n = this.labels.length;
    const out = {};
    if (n) {
      await this.read.mapAsync(GPUMapMode.READ, 0, 16 * n);
      const t = new BigInt64Array(this.read.getMappedRange(0, 16 * n));
      this.labels.forEach((l, i) => (out[l] = (out[l] || 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6));
      this.read.unmap();
    }
    this.labels = [];
    for (const k of Object.keys(out)) out[k] = Math.round(out[k] * 1000) / 1000;
    return out;
  }
}

/**
 * Passes over more than this many tokens are submitted in several command buffers, waiting for
 * the queue to drain in between: one long submit starves the page's compositor (frames stall for
 * hundreds of milliseconds while the GPU is busy).
 */
export const YIELD_TOKENS = 256;
export const YIELD_CHUNKS = 6;

/**
 * How long passes are submitted: "await" waits for each chunk to finish before sending the next,
 * "split" sends the chunks as separate command buffers back to back, "none" sends one.
 */
export const SUBMIT = { mode: "await" };

/** Ends a chunk of a long pass according to SUBMIT.mode. */
export async function endChunk(device, enc) {
  if (SUBMIT.mode === "none") return enc;
  device.queue.submit([enc.finish()]);
  if (SUBMIT.mode === "await") await device.queue.onSubmittedWorkDone();
  return device.createCommandEncoder();
}

/** Splits `ops` into `n` contiguous chunks. */
export function chunks(ops, n) {
  const out = [];
  const size = Math.ceil(ops.length / n);
  for (let i = 0; i < ops.length; i += size) out.push(ops.slice(i, i + size));
  return out;
}

/** Compiles a compute pipeline, turning WGSL errors into exceptions. */
export async function pipeline(device, code, label, layout = "auto") {
  const m = device.createShaderModule({ code, label });
  const info = await m.getCompilationInfo?.();
  const errs = (info?.messages || []).filter((x) => x.type === "error");
  if (errs.length) throw Object.assign(new Error(`${label}: ${errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; ")}`), { code: "WEBGPU_INIT" });
  try {
    return await device.createComputePipelineAsync({ layout, compute: { module: m, entryPoint: "main" }, label });
  } catch (e) {
    throw Object.assign(new Error(`${label}: ${e.message}`), { code: "WEBGPU_INIT" });
  }
}

export class GpuTrunk {
  /**
   * `layout` is the single-shard trunk layout from `kevala_layouts`; tensors arrive through
   * `write(dst, bytes)` as the pack streams by.
   */
  constructor(gpu, layout, cfg) {
    U = GPUBufferUsage;
    this.device = gpu.device;
    this.name = gpu.name;
    this.subgroup32 = !!gpu.subgroup32;
    this.wgsl = gpu.wgsl;
    this.cfg = cfg;
    this.weights = new GpuWeights(gpu.device, layout);
    this.tensors = this.weights.tensors;
    this.capacity = 0;
  }

  write(dst, bytes) {
    this.weights.write(dst, bytes);
  }

  missing() {
    return this.weights.missing();
  }

  /** Builds pipelines and static buffers once all weights are uploaded. */
  async init(finalNorm, typeEmb) {
    const miss = this.missing();
    if (miss.length) throw new Error(`GPU trunk is missing ${miss.length} tensors (${miss[0]}...)`);
    const d = this.device;
    const cfg = this.cfg;
    const pipe = (code, label) => pipeline(d, code, label);
    const kernel = (name) => pipe(this.wgsl(name), name);
    [this.mm, this.pNorm, this.pRope, this.pAttn, this.pGeglu, this.pGather] = await Promise.all([
      matmulPipelines(d, this.wgsl),
      kernel("norm"),
      kernel("rope"),
      kernel(this.subgroup32 ? "attention_subgroup" : "attention"),
      kernel("geglu"),
      kernel("gather"),
    ]);
    this.globals = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const f32buf = (arr, usage = U.STORAGE) => {
      const b = d.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | U.COPY_DST });
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    this.finalNorm = f32buf(finalNorm);
    this.typeEmb = f32buf(typeEmb);
    this.zeros = f32buf(new Float32Array(4096));
    // rotary tables: [table][pos][32] of (cos, sin), same math as the CPU path
    const maxPos = 512;
    const cs = new Float32Array(2 * maxPos * 32 * 2);
    [cfg.rope_global, cfg.rope_local].forEach((theta, ti) => {
      for (let i = 0; i < 32; i++) {
        const inv = Math.fround(1 / Math.fround(Math.pow(theta, (2 * i) / 64)));
        for (let pos = 0; pos < maxPos; pos++) {
          const a = Math.fround(pos * inv);
          const o = ((ti * maxPos + pos) * 32 + i) * 2;
          cs[o] = Math.cos(a);
          cs[o + 1] = Math.sin(a);
        }
      }
    });
    this.rope = f32buf(cs);
    this.uniforms = [];
    this.ensure(64);
  }

  uniform(values) {
    const b = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    const a = new ArrayBuffer(16);
    const u = new Uint32Array(a);
    const f = new Float32Array(a);
    values.forEach((v, i) => {
      if (typeof v === "object") f[i] = v.f;
      else u[i] = v;
    });
    this.device.queue.writeBuffer(b, 0, a);
    this.uniforms.push(b);
    return b;
  }

  /** Grows activation buffers to hold `tokens` and rebuilds the bind groups. */
  ensure(tokens) {
    if (tokens <= this.capacity) return;
    const cap = Math.max(64, 1 << Math.ceil(Math.log2(tokens)));
    const d = this.device;
    for (const b of [this.x, this.h, this.qkv, this.ctx, this.up, this.act, this.tok, this.blocks, this.rowsBuf, this.gathered, this.readback, this.part]) b?.destroy();
    for (const b of this.uniforms) b.destroy();
    this.uniforms = [];
    const D = this.cfg.hidden;
    const buf = (floats, extra = 0) => d.createBuffer({ size: floats * 4, usage: U.STORAGE | extra });
    this.x = buf(cap * D, U.COPY_DST | U.COPY_SRC);
    this.h = buf(cap * D, U.COPY_SRC);
    this.qkv = buf(cap * 3 * D);
    this.ctx = buf(cap * D);
    this.up = buf(cap * 2 * this.cfg.intermediate);
    this.act = buf(cap * Math.max(this.cfg.intermediate, this.cfg.head_ff));
    this.tok = buf(cap * 4, U.COPY_DST);
    // attention query blocks: at most one per token
    this.blocks = buf(cap * 4, U.COPY_DST);
    this.rowsBuf = buf(cap, U.COPY_DST);
    this.gathered = buf(cap * D, U.COPY_SRC);
    this.readback = d.createBuffer({ size: cap * D * 4, usage: U.MAP_READ | U.COPY_DST });
    this.part = buf(SPLIT_SCRATCH);
    this.capacity = cap;
    this.buildGroups();
  }

  buildGroups() {
    const d = this.device;
    const cfg = this.cfg;
    const D = cfg.hidden;
    const T = (n) => this.tensors.get(n);
    const bg = (pipeline, entries) =>
      d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const mm = (w, bias, input, output, mode) => {
      const e = T(w);
      const [N, K] = e.info.shape;
      const b = bias ? T(bias).buf : this.zeros;
      const u = this.uniform([N, K, mode, bias ? 1 : 0]);
      return {
        kind: "mm",
        label: "mm." + w.split(".").pop(),
        N,
        K,
        group: d.createBindGroup({ layout: this.mm.layout, entries: [this.globals, u, input, e.buf, e.sbuf, b, output, this.part].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
        reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: [this.globals, u, this.part, b, output].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
      };
    };
    const norm = (w, b, input, output, typed = false) => ({
      kind: "norm",
      group: bg(this.pNorm, [this.globals, this.uniform([D, b ? 1 : 0, typed ? 1 : 0, { f: typed ? cfg.norm_eps : b ? cfg.head_norm_eps : cfg.norm_eps }]), input, w, b || this.zeros, output, this.tok, this.typeEmb]),
    });
    const ops = [];
    for (let i = 0; i < cfg.layers; i++) {
      const global = i % cfg.global_every === 0;
      const n = (s) => `enc.${i}.${s}`;
      // layer 0 has no attention norm: its input is the normalized embedding itself
      const attnIn = i === 0 ? this.x : this.h;
      if (i > 0) ops.push(norm(T(n("attn_norm")).buf, null, this.x, this.h));
      ops.push(mm(n("wqkv"), null, attnIn, this.qkv, 0));
      ops.push({ kind: "rope", group: bg(this.pRope, [this.globals, this.uniform([cfg.heads, 3 * D, global ? 0 : 1, 0]), this.qkv, this.tok, this.rope]) });
      ops.push({ kind: "attn", group: bg(this.pAttn, [this.globals, this.uniform([D, 3 * D, global ? 0 : cfg.window, 0]), this.qkv, this.blocks, this.ctx]) });
      ops.push(mm(n("wo"), null, this.ctx, this.x, 1));
      ops.push(norm(T(n("mlp_norm")).buf, null, this.x, this.h));
      ops.push(mm(n("wi"), null, this.h, this.up, 0));
      ops.push({ kind: "geglu", I: cfg.intermediate, group: bg(this.pGeglu, [this.globals, this.uniform([cfg.intermediate, 0, 0, 0]), this.up, this.act]) });
      ops.push(mm(n("wo2"), null, this.act, this.x, 1));
    }
    // bridge: final encoder norm plus the question-type embedding, in place
    ops.push(norm(this.finalNorm, null, this.x, this.h, true));
    ops.push({ kind: "copy" });
    for (let i = 0; i < cfg.head_layers; i++) {
      const n = (s) => `head.${i}.${s}`;
      ops.push(norm(T(n("norm1.w")).buf, T(n("norm1.b")).buf, this.x, this.h));
      ops.push(mm(n("in_proj"), n("in_proj.b"), this.h, this.qkv, 0));
      ops.push({ kind: "attn", group: bg(this.pAttn, [this.globals, this.uniform([D, 3 * D, 0, 0]), this.qkv, this.blocks, this.ctx]) });
      ops.push(mm(n("out_proj"), n("out_proj.b"), this.ctx, this.x, 1));
      ops.push(norm(T(n("norm2.w")).buf, T(n("norm2.b")).buf, this.x, this.h));
      ops.push(mm(n("lin1"), n("lin1.b"), this.h, this.act, 2));
      ops.push(mm(n("lin2"), n("lin2.b"), this.act, this.x, 1));
    }
    ops.push({ kind: "gather", group: bg(this.pGather, [this.globals, this.x, this.rowsBuf, this.gathered]) });
    this.ops = ops;
  }

  /**
   * Runs the trunk. `x` is the embedded batch (`tokens * hidden` f32), `segs` the segment table,
   * `rows` the token indices whose final states the scorer needs. Resolves to those rows.
   */
  async forward(x, segs, rows) {
    const d = this.device;
    const D = this.cfg.hidden;
    const tokens = x.length / D;
    this.ensure(Math.max(tokens, rows.length));
    const tok = new Uint32Array(tokens * 4);
    for (const s of segs) {
      for (let p = 0; p < s.len; p++) {
        const t = s.start + p;
        tok.set([s.start, s.len, p, s.qtype], t * 4);
      }
    }
    const q = d.queue;
    // the first passes run inside error scopes so a validation failure surfaces instead of
    // silently producing garbage
    // every pass runs in error scopes: a rejected command buffer must be an error, not the
    // previous pass's rows read back
    const checking = true;
    if (checking) {
      d.pushErrorScope("validation");
      d.pushErrorScope("out-of-memory");
    }
    const blocks = [];
    for (const sg of segs) for (let q0 = 0; q0 < sg.len; q0 += 16) blocks.push(sg.start, sg.len, q0, 0);
    const nblocks = blocks.length / 4;
    q.writeBuffer(this.globals, 0, new Uint32Array([tokens, rows.length, nblocks, 0]));
    q.writeBuffer(this.blocks, 0, new Uint32Array(blocks));
    q.writeBuffer(this.x, 0, x);
    q.writeBuffer(this.tok, 0, tok);
    q.writeBuffer(this.rowsBuf, 0, new Uint32Array(rows));
    const prof = this.profiler;
    const heads = this.cfg.heads;
    const groups = tokens > YIELD_TOKENS && !prof ? chunks(this.ops, YIELD_CHUNKS) : [this.ops];
    let enc = d.createCommandEncoder();
    for (let gi = 0; gi < groups.length; gi++) {
    // let the page render between chunks of a long pass
    if (gi) enc = await endChunk(d, enc);
    let pass = prof ? null : enc.beginComputePass();
    for (const op of groups[gi]) {
      if (op.kind === "copy") {
        pass?.end();
        enc.copyBufferToBuffer(this.h, 0, this.x, 0, tokens * D * 4);
        pass = prof ? null : enc.beginComputePass();
        continue;
      }
      if (prof) pass = prof.pass(enc, op.label || op.kind);
      pass.setBindGroup(0, op.group);
      switch (op.kind) {
        case "mm":
          encodeMatmul(pass, this.mm, op, tokens);
          break;
        case "norm":
          pass.setPipeline(this.pNorm);
          pass.dispatchWorkgroups(tokens);
          break;
        case "rope":
          pass.setPipeline(this.pRope);
          pass.dispatchWorkgroups(tokens, 2 * heads);
          break;
        case "attn":
          pass.setPipeline(this.pAttn);
          pass.dispatchWorkgroups(nblocks, heads);
          break;
        case "geglu": {
          pass.setPipeline(this.pGeglu);
          const n = Math.ceil((tokens * op.I) / 256);
          pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
          break;
        }
        case "gather":
          pass.setPipeline(this.pGather);
          pass.dispatchWorkgroups(rows.length);
          break;
      }
      if (prof) {
        pass.end();
        pass = null;
      }
    }
    pass?.end();
    }
    prof?.finish(enc);
    const bytes = rows.length * D * 4;
    enc.copyBufferToBuffer(this.gathered, 0, this.readback, 0, bytes);
    q.submit([enc.finish()]);
    if (prof) this.lastProfile = await prof.collect();
    if (checking) {
      const [oom, invalid] = [await d.popErrorScope(), await d.popErrorScope()];
      if (oom || invalid) throw Object.assign(new Error(`WebGPU: ${(oom || invalid).message}`), { code: "WEBGPU_INIT" });
    }
    await this.readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(this.readback.getMappedRange(0, bytes).slice(0));
    this.readback.unmap();
    return out;
  }
}
