// What every WebGPU trunk shares: the device, streamed weight buffers, compiled pipelines, the
// int8 matmul with its split-K reduce, per-kernel profiling, and long-pass submission. The trunks
// themselves are gpu-laya.js, gpu-kev.js and gpu-gemma4.js.
//
// The kernels are WGSL sources in the Rust crate (crates/kevala/src/wgsl), specialized by the
// WebAssembly binary: `gpu.wgsl(kernel, spec)` returns the source to compile. The JavaScript side
// only creates buffers, pipelines and bind groups, and records the dispatches.

import { parsePackHeader } from "./pack-layout.js";

// Narrow matmuls at short lengths launch too few tiles to fill a GPU (a 1024-wide output at 64
// tokens is 16 workgroups), so K is split across up to 8 workgroups whose partial tiles a second
// pass sums. The kernel (wgsl/splits.wgsl) and the host compute the same split count.
const SPLIT_TARGET = 128;
/** Host side of `mm_splits`. */
export function mmSplits(T, N, K, target = SPLIT_TARGET, bm = 64, bn = 64) {
  const tiles = Math.ceil(N / bn) * Math.ceil(T / bm);
  if (tiles >= Math.ceil((target * 3) / 4)) return 1;
  return Math.max(1, Math.min(8, Math.ceil(target / tiles), Math.floor(K / 128)));
}

/** Select a 16 R row tile without changing the workgroup grid or split-K partition. */
export function rowsPerThread(T, config = null) {
  if (config?.groups === 1 && (config.f16 === false || config.kernel !== "matmul_wide") && T >= 64 && Math.ceil(T / 48) === Math.ceil(T / 64)) return 3;
  return T >= 64 ? 4 : Math.max(1, Math.ceil(T / 16));
}

export function matmulConfig(device) {
  return { f16: device.features.has("shader-f16"), groups: 1, splitTarget: 256 };
}

/**
 * Encodes a matmul op (bind groups `group` and `reduce`), splitting K when it helps. The kernel
 * variant covers 16 R rows and 64 J columns per workgroup. `activeT` can trim output
 * dispatches while T retains the shader's split rule and partial-buffer stride.
 */
export function dispatchMatmul(pass, pMatmul, pReduce, op, T, target = SPLIT_TARGET, R = 4, J = 1, activeT = T) {
  const bm = 16 * R;
  const bn = 64 * J;
  const splits = mmSplits(T, op.N, op.K, target, bm, bn);
  pass.setPipeline(pMatmul);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / bn), Math.ceil(activeT / bm), splits);
  if (splits > 1) {
    const n = Math.ceil((activeT * op.N) / 256);
    pass.setPipeline(pReduce);
    pass.setBindGroup(0, op.reduce);
    pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
  }
}

/** Encodes a matmul with the kernel variant for T. `pipes` comes from `matmulPipelines`. */
export function encodeMatmul(pass, pipes, op, T) {
  const R = rowsPerThread(T, pipes);
  // BM56 trims padding only when the R4 grid and K partition stay unchanged. Prefer R3 first.
  const row56 = R === 4 && pipes.f16 === false && pipes.groups === 1 && Math.ceil(T / 56) === Math.ceil(T / 64);
  dispatchMatmul(pass, (row56 && pipes.mm56) || pipes.mm[R], pipes.reduce[R], op, T, pipes.splitTarget, R, pipes.groups);
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
 * binary's kernels, plus an optional FP32 generic BM56 pipeline paired with the R4 reducer.
 * With `shader-f16` the tiles live in workgroup memory as f16, which halves the traffic that
 * limits this kernel (1.6-1.75x faster on Apple GPUs); products and sums stay f32, and the
 * rounding (about 3e-4 relative) is far below the int8 weight quantization.
 */
export async function matmulPipelines(device, wgsl, kernel = "matmul") {
  const layout = matmulLayout(device);
  const rlayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const rpl = device.createPipelineLayout({ bindGroupLayouts: [rlayout] });
  const config = matmulConfig(device);
  const mm = [];
  const reduce = [];
  const [mm56] = await Promise.all([
    kernel === "matmul" && !config.f16 && config.groups === 1
      ? pipeline(device, wgsl(kernel, { ...config, rows: 4, row56: true }), "matmul_row56", pl)
      : null,
    ...[1, 2, 3, 4].flatMap((rows) => [
      pipeline(device, wgsl(kernel, { ...config, rows }), `${kernel}_r${rows}`, pl).then((p) => (mm[rows] = p)),
      pipeline(device, wgsl("reduce", { ...config, rows }), `reduce_r${rows}`, rpl).then((p) => (reduce[rows] = p)),
    ]),
  ]);
  return { layout, reduceLayout: rlayout, mm, mm56, reduce, kernel, ...config };
}

/** Conservative scratch bound for split targets up to 256 and 64 x 64 tiles. */
export const SPLIT_SCRATCH = 8 * 96 * 64 * 64;

/** Workgroup memory of attention_tile.wgsl: Q, K, V and probability tiles and the block info, and
 * without subgroups the table its threads exchange row statistics through. */
const ATTENTION_TILE_BYTES = 1088 * 8 + 544 * 8 + 512 * 8 + 2112 * 4 + 16;
const ATTENTION_TILE_SHARED_BYTES = ATTENTION_TILE_BYTES + 1088 * 4;

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
 * way the weakest WebGPU device runs (for testing those kernel paths on a strong one). `features`
 * limits the optional features asked for (for example ["shader-f16", "timestamp-query"] runs the
 * kernel paths of a GPU with f16 but no subgroups).
 */
export async function requestDevice({ baseline = false, features = null, powerPreference = "high-performance" } = {}) {
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
  // Timestamps feed profiling and kernel selection. The baseline keeps them while
  // omitting f16, so it still uses the generic matrix kernel.
  const optional = baseline ? ["timestamp-query"] : features || ["timestamp-query", "shader-f16", "subgroups"];
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
  // the tiled attention reduces across 16 lanes and keeps f16 tiles in 25 KB of workgroup memory
  const f16 = device.features.has("shader-f16");
  const room = device.limits.maxComputeWorkgroupStorageSize;
  const attentionTile = subgroups && info.subgroupMinSize >= 16 && f16 && room >= ATTENTION_TILE_BYTES;
  // without such subgroups the same kernel trades row statistics through workgroup memory
  const attentionTileShared = !attentionTile && f16 && room >= ATTENTION_TILE_SHARED_BYTES;
  return Object.assign(gpu, { subgroup32, subgroup4, attentionTile, attentionTileShared, name: [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "WebGPU" });
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

/** `GPUBufferUsage`, which exists only where WebGPU does. */
export function bufferUsage() {
  if (!globalThis.GPUBufferUsage) throw new Error("WebGPU buffer usage constants are unavailable");
  return globalThis.GPUBufferUsage;
}

/** A labelled buffer of at least the 16 bytes a binding needs. */
export function scratchBuffer(device, size, flags, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage: flags });
}

/**
 * Runs `work` inside error scopes and rethrows whatever the device reported, so a calibration
 * that fails cannot poison the scopes of the load around it.
 */
export async function inErrorScopes(device, work) {
  if (!device.pushErrorScope || !device.popErrorScope) return work();
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let value;
  let failure;
  try {
    value = await work();
  } catch (error) {
    failure = error;
  }
  const scopes = await Promise.allSettled([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  const scoped = scopes.map((s) => (s.status === "fulfilled" ? s.value : s.reason)).find(Boolean);
  if (failure || scoped) throw failure || scoped;
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
    this.header = parsePackHeader(layout.prefix);
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
