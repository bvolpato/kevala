// Qwen3.5's text trunk on WebGPU, shared by Kev and SemIf scoring.
//
// A pass runs in two stages inside one command buffer. Stage 1 runs every request's state and
// keeps what the question branches need: each DeltaNet layer's recurrent state and conv tail,
// each attention layer's keys and values. Stage 2 runs every question branch, starting from its
// state's carry. Only the pointer rows come back to the CPU.
//
// The kernels are WGSL sources in the Rust crate (crates/kevala/src/wgsl/kev_*.wgsl), specialized
// by the WebAssembly binary through `gpu.wgsl`; this file builds pipelines and records dispatches.

import { GpuWeights, pipeline, encodeMatmul, SPLIT_SCRATCH, YIELD_TOKENS, YIELD_CHUNKS, chunks, endChunk } from "./gpu.js";
import { calibrateMatmul, selectedMatmulKernel } from "./gpu-tuning.js";
import { calibrateRecurrence } from "./gpu-recur-tuning.js";

let U;

// the cross-request state cache (see KevModel in crates/kevala/src/kev.rs for the CPU twin)
const CACHE_SLOTS = 4;
const SLOT_ROWS = 1024;
const CACHE_MIN = 32;
const EXTEND_MIN = 16;

/** Kev's own kernels, exported for the kernel benchmarks. */

/** The tiled attention's block table for segments of these lengths: blocks of up to 8 tokens. */
function attnBlocks(lengths, cap) {
  const table = new Uint32Array(cap * 4);
  let n = 0;
  lengths.forEach((len, seg) => {
    for (let r0 = 0; r0 < len; r0 += 8) table.set([seg, r0, Math.min(8, len - r0), 0], 4 * (1 + n++));
  });
  table[0] = n;
  return table;
}

export class GpuKev {
  constructor(gpu, layout, cfg) {
    U = GPUBufferUsage;
    this.device = gpu.device;
    this.subgroup32 = !!gpu.subgroup32;
    this.subgroup4 = !!gpu.subgroup4;
    this.subgroupMinSize = this.subgroup4 ? (gpu.adapter?.info?.subgroupMinSize || 4) : 0;
    // Each workgroup shares keys and values across 8 tokens and 4 query heads.
    this.attnTileFp32 = this.subgroup32 && !this.device.features.has("shader-f16") && cfg.heads === 4 * cfg.kv_heads;
    this.attnTile = (!!gpu.attentionTile && cfg.heads === 4 * cfg.kv_heads) || this.attnTileFp32;
    this.wgsl = gpu.wgsl;
    this.name = gpu.name;
    this.gpuKernel = gpu.kernel || "auto";
    this.cfg = cfg;
    this.dims = {
      linear: 2 * cfg.lin_key_heads * 128 + cfg.lin_heads * 128,
      linearOut: cfg.lin_heads * 128,
      linearProj: 2 * cfg.lin_key_heads * 128 + 2 * cfg.lin_heads * 128,
      attentionOut: cfg.heads * 256,
      attentionProj: 2 * (cfg.heads + cfg.kv_heads) * 256,
      kv: 2 * cfg.kv_heads * 256,
    };
    this.weights = new GpuWeights(gpu.device, layout);
    for (const t of this.weights.header.tensors) {
      this.checkBuffer(t.size, t.name);
      if (t.scales_size) this.checkBuffer(t.scales_size, `${t.name}.scales`);
    }
    for (let i = 0; i < cfg.layers; i++) {
      if (!cfg.full[i]) for (const n of ["a", "b"]) this.weights.keepHost.add(`L.${i}.${n}`);
    }
    this.cap = { T1: 0, T2: 0, P: 0, S: 0, R: 0 };
    this.stats = { hits: 0, extensions: 0, misses: 0, tokensSaved: 0 };
    this.stateCacheEnabled = true;
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
      throw Object.assign(new Error(`Qwen WebGPU buffer ${label} needs ${Math.ceil(size / 1048576)} MiB; this device allows ${Math.floor(limit / 1048576)} MiB per buffer. Reduce the batch size or use the native runtime.`), { code: "WEBGPU_INIT" });
    }
  }

  async init() {
    const miss = this.missing();
    if (miss.length) throw new Error(`GPU trunk is missing ${miss.length} tensors (${miss[0]}...)`);
    const d = this.device;
    this.recurTiles = 2;
    const kernels = {
      RMS: ["kev_rms"],
      GATES: ["kev_gates"],
      CONV: ["kev_conv"],
      SAVE_TAIL: ["kev_save_tail"],
      RECUR: ["kev_recur_lanes", { subgroups: this.subgroup4 }],
      GNORM: ["kev_gnorm"],
      APREP: ["kev_aprep"],
      SAVE_KV: ["kev_save_kv"],
      KEYS: ["kev_attention_keys"],
      ATTN: this.attnTile ? ["kev_attention_tile", { f16: !this.attnTileFp32 }] : ["kev_attention", { subgroups: this.subgroup32 }],
      SILUMUL: ["kev_silumul"],
      GATHER: ["kev_gather"],
    };
    const built = await Promise.all(Object.entries(kernels).map(([key, [name, spec]]) => pipeline(d, this.wgsl(name, { ...spec, kev: this.cfg }), name).then((p) => [key, p])));
    this.p = Object.fromEntries(built);
    this.tuning = await calibrateMatmul(d, this.wgsl, this.weights, { kernel: this.gpuKernel });
    this.mm = this.tuning.pipelines.generic;
    const recurrence = await calibrateRecurrence(d, [
      { lanes: 4, pipeline: this.p.RECUR },
      ...[8, 16].map((lanes) => ({
        lanes,
        pipeline: () => pipeline(d, this.wgsl(`kev_recur_lanes${lanes}`, { subgroups: this.subgroupMinSize >= lanes, kev: this.cfg }), `kev_recur_lanes${lanes}`),
      })),
    ], this.cfg);
    this.p.RECUR = recurrence.pipeline || this.p.RECUR;
    this.recurTiles = recurrence.zTiles;
    this.tuning.diagnostics.recurrence = { ...recurrence.diagnostics, lanes: recurrence.choice };
    const cfg = this.cfg;
    const W = (n) => this.weights.get(n);
    const f32buf = (arr) => {
      this.checkBuffer(arr.byteLength, "model constants");
      const b = d.createBuffer({ label: "Qwen model constants", size: Math.max(16, arr.byteLength), usage: U.STORAGE | U.COPY_DST });
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    // Stack a and b, then interleave their vec4 weights so adjacent gate lanes read together.
    this.ab = [];
    for (let i = 0; i < cfg.layers; i++) {
      if (cfg.full[i]) {
        this.ab.push(null);
        continue;
      }
      const a = W(`L.${i}.a`).host;
      const b = W(`L.${i}.b`).host;
      const both = new Float32Array(a.length + b.length);
      both.set(a);
      both.set(b, a.length);
      const interleaved = new Float32Array(both.length);
      for (let k = 0; k < cfg.hidden / 4; k++) {
        for (let j = 0; j < 2 * cfg.lin_heads; j++) {
          for (let c = 0; c < 4; c++) interleaved[(k * 2 * cfg.lin_heads + j) * 4 + c] = both[j * cfg.hidden + k * 4 + c];
        }
      }
      this.ab.push(f32buf(interleaved));
    }
    // Rotary table [pos][rotary/2] of (cos, sin), f32 inverse frequencies and angles as in PyTorch.
    const maxPos = cfg.max_state + cfg.max_branch;
    const half = cfg.rotary / 2;
    const cs = new Float32Array(maxPos * cfg.rotary);
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(1 / Math.fround(Math.pow(cfg.rope_theta, (2 * i) / cfg.rotary)));
      for (let pos = 0; pos < maxPos; pos++) {
        const a = Math.fround(pos * inv);
        cs[(pos * half + i) * 2] = Math.cos(a);
        cs[(pos * half + i) * 2 + 1] = Math.sin(a);
      }
    }
    this.rope = f32buf(cs);
    this.zeros = f32buf(new Float32Array(16));
  }

  uni(values) {
    const a = new ArrayBuffer(16);
    const u = new Uint32Array(a);
    const f = new Float32Array(a);
    values.forEach((v, i) => (typeof v === "object" ? (f[i] = v.f) : (u[i] = v)));
    const b = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.uniforms.push(b);
    this.device.queue.writeBuffer(b, 0, a);
    return b;
  }

  resetDynamicBuffers() {
    for (const b of [...(this.owned || []), ...(this.uniforms || []), ...(this.carryBufs || [])]) b.destroy();
    this.owned = [];
    this.uniforms = [];
    this.carryBufs = [];
    this.cap = { T1: 0, T2: 0, P: 0, S: 0, R: 0 };
    this.carrySize = { slots: 0, rows: 0 };
    this.carry = null;
    this.kvStage = null;
    this.ops = null;
    this.lru = [];
  }

  /** Sizes every buffer for a batch and rebuilds the bind groups when it grows. */
  ensure(T1, T2, P, S, R) {
    const c = this.cap;
    if (T1 <= c.T1 && T2 <= c.T2 && P <= c.P && S <= c.S && R <= c.R) return;
    const up = (n, min) => Math.max(min, 2 ** Math.ceil(Math.log2(Math.max(1, n))));
    const cap = { T1: up(T1, 64), T2: up(T2, 64), P: up(P, 1), S: up(S, 4), R: up(R, 16) };
    const T = Math.max(cap.T1, cap.T2);
    this.checkBuffer(T * Math.max(this.cfg.hidden, this.dims.linearProj, this.dims.attentionProj, 2 * this.cfg.intermediate) * 4, "batch scratch");
    this.checkBuffer(cap.R * this.cfg.hidden * 4, "scorer rows");
    this.checkBuffer(cap.S * 32, "segments");
    this.cap = cap;
    for (const b of this.owned || []) b.destroy();
    this.owned = [];
    for (const b of this.uniforms || []) b.destroy();
    this.uniforms = [];
    const d = this.device;
    const cfg = this.cfg;
    const buf = (floats, extra = 0, label = "batch scratch") => {
      this.checkBuffer(floats * 4, label);
      const b = d.createBuffer({ label: `Qwen ${label}`, size: Math.max(16, floats * 4), usage: U.STORAGE | extra });
      this.owned.push(b);
      return b;
    };
    const D = cfg.hidden;
    this.x = buf(T * D, U.COPY_DST | U.COPY_SRC);
    this.x2 = buf(this.cap.T2 * D, U.COPY_DST | U.COPY_SRC);
    this.h = buf(T * D);
    this.proj = buf(T * Math.max(this.dims.linearProj, this.dims.attentionProj), 0, "QKV projection");
    this.conv = buf(T * Math.max(this.dims.linear, this.dims.kv / 2), 0, "convolution");
    this.core = buf(T * Math.max(this.dims.linearOut, this.dims.attentionOut), 0, "attention output");
    this.gates = buf(T * 2 * cfg.lin_heads);
    this.up = buf(T * 2 * cfg.intermediate, 0, "MLP projection");
    this.act = buf(T * cfg.intermediate);
    this.tok = buf(T * 4, U.COPY_DST);
    this.tok2 = buf(this.cap.T2 * 4, U.COPY_DST | U.COPY_SRC);
    this.segs = buf(this.cap.S * 8, U.COPY_DST);
    this.segs2 = buf(this.cap.S * 8, U.COPY_DST | U.COPY_SRC);
    // tiled attention blocks per stage: a count, then (segment, first position, tokens) per block
    this.blockCap = Math.ceil(Math.max(T, this.cap.T2) / 8) + this.cap.S + 1;
    this.ablocks = buf(this.blockCap * 4, U.COPY_DST);
    this.ablocks2 = buf(this.blockCap * 4, U.COPY_DST | U.COPY_SRC);
    this.rows = buf(this.cap.R, U.COPY_DST);
    this.gathered = buf(this.cap.R * D, U.COPY_SRC);
    this.part = buf(SPLIT_SCRATCH);
    this.readback = d.createBuffer({ size: this.cap.R * D * 4, usage: U.MAP_READ | U.COPY_DST });
    this.owned.push(this.readback);
    this.g = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.owned.push(this.g);
    this.g2 = d.createBuffer({ size: 16, usage: U.COPY_DST | U.COPY_SRC });
    this.owned.push(this.g2);
    if (!this.carry) this.ensureCarries(CACHE_SLOTS + 1, CACHE_SLOTS * SLOT_ROWS + 1024);
    this.build();
  }

  /**
   * Carry storage: CACHE_SLOTS slots that outlive a pass (the cross-request state cache, with
   * SLOT_ROWS rows of keys and values each) plus scratch slots and rows for states this pass
   * does not cache. Growing it drops the cache and rebuilds the bind groups.
   */
  ensureCarries(slots, rows) {
    const c = this.carrySize || { slots: 0, rows: 0 };
    if (slots <= c.slots && rows <= c.rows && this.carry) return;
    const up = (n) => 2 ** Math.ceil(Math.log2(Math.max(1, n)));
    const size = { slots: Math.max(c.slots, up(slots)), rows: Math.max(c.rows, up(rows)) };
    this.checkBuffer(size.rows * this.dims.kv * 4, "KV cache");
    this.checkBuffer(size.slots * this.cfg.lin_heads * 16384 * 4, "recurrent state");
    this.checkBuffer(size.slots * 3 * this.dims.linear * 4, "convolution tail");
    this.carrySize = size;
    for (const b of this.carryBufs || []) b.destroy();
    this.carryBufs = [];
    const buf = (floats, label) => {
      this.checkBuffer(floats * 4, label);
      const b = this.device.createBuffer({ label: `Qwen ${label}`, size: Math.max(16, floats * 4), usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
      this.carryBufs.push(b);
      return b;
    };
    this.kvStage = buf(SLOT_ROWS * this.dims.kv, "KV staging");
    this.carry = [];
    for (let i = 0; i < this.cfg.layers; i++) {
      this.carry.push(this.cfg.full[i] ? { kv: buf(this.carrySize.rows * this.dims.kv, "KV cache") } : { state: buf(this.carrySize.slots * this.cfg.lin_heads * 16384, "recurrent state"), tail: buf(this.carrySize.slots * 3 * this.dims.linear, "convolution tail") });
    }
    this.lru = [];
    if (this.cap.T1) this.build();
  }

  /**
   * Plans stage 1 against the state cache. Each request's state is an exact hit (no work), an
   * extension of a cached state (only the new tokens run, from its carry), or a miss.
   */
  plan(batch, ids) {
    const slotRows = SLOT_ROWS;
    const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
    const starts = (a, p) => p.length < a.length && p.every((v, i) => v === a[i]);
    // Newly planned states run concurrently. Only retained states from a completed pass
    // can supply a prefix carry; fresh exact duplicates share their planned destination.
    const ready = new Set(this.lru);
    const pinned = new Set();
    const carries = [];
    const segs = [];
    const copies = [];
    let scratchSlot = CACHE_SLOTS;
    let scratchRow = CACHE_SLOTS * slotRows;
    const fresh = new Map();
    batch.states.forEach(([start, len], r) => {
      if (!len) return carries.push(null);
      const mine = ids.subarray(start, start + len);
      const key = mine.join(",");
      if (fresh.has(key)) return carries.push(fresh.get(key));
      const hit = this.lru.find((e) => same(e.ids, mine));
      if (hit) {
        this.lru = [...this.lru.filter((e) => e !== hit), hit];
        pinned.add(hit.slot);
        this.stats.hits++;
        this.stats.tokensSaved += len;
        return carries.push(hit);
      }
      const ext = this.lru.filter((e) => ready.has(e) && e.len >= EXTEND_MIN && starts(mine, e.ids)).sort((a, b) => b.len - a.len)[0];
      const from = ext ? ext.len : 0;
      if (ext) {
        pinned.add(ext.slot);
        this.stats.extensions++;
        this.stats.tokensSaved += from;
      } else this.stats.misses++;
      // destination: a cache slot when the state is cacheable, else scratch
      let dst;
      let kvBase;
      let entry = null;
      if (len >= CACHE_MIN && len <= slotRows) {
        let victim = null;
        if (this.lru.length < CACHE_SLOTS) {
          const used = new Set(this.lru.map((e) => e.slot));
          victim = { slot: [...Array(CACHE_SLOTS).keys()].find((k) => !used.has(k) && !pinned.has(k)) };
        } else {
          victim = this.lru.find((e) => !pinned.has(e.slot));
          if (victim) this.lru = this.lru.filter((e) => e !== victim);
        }
        if (victim && victim.slot !== undefined) {
          dst = victim.slot;
          kvBase = dst * slotRows;
          entry = { ids: mine.slice(), slot: dst, kvBase, len };
          this.lru.push(entry);
          pinned.add(dst);
        }
      }
      if (dst === undefined) {
        dst = scratchSlot++;
        kvBase = scratchRow;
        scratchRow += len;
      }
      if (ext) copies.push({ from: ext.kvBase, to: kvBase, rows: from });
      segs.push({ src: start + from, len: len - from, parent: ext ? ext.slot : 0xffffffff, pstart: ext ? ext.kvBase : 0, plen: from, dst, kvdst: kvBase + from, pos0: from });
      const carry = entry || { slot: dst, kvBase, len };
      fresh.set(key, carry);
      carries.push(carry);
    });
    return { carries, segs, copies, slots: scratchSlot, rows: scratchRow };
  }

  build() {
    const d = this.device;
    const cfg = this.cfg;
    const W = (n) => this.weights.get(n);
    const bg = (p, list) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const mm = (w, input, output, mode) => {
      const e = W(w);
      const [N, K] = e.info.shape;
      const u = this.uni([N, K, mode, 0]);
      return {
        k: "mm",
        label: "mm." + w.split(".").pop(),
        N,
        K,
        group: d.createBindGroup({ layout: this.mm.layout, entries: [this.g, u, input, e.buf, e.sbuf, this.zeros, output, this.part].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
        reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: [this.g, u, this.part, this.zeros, output].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
      };
    };
    const rms = (w, input, output) => ({ k: "rms", group: bg(this.p.RMS, [this.g, this.uni([cfg.hidden, 0, 0, { f: cfg.eps }]), input, w, output]) });
    const one = [];
    const two = [];
    let stateComplete = false;
    const both = (op) => {
      if (!stateComplete) one.push(op);
      two.push(op);
    };
    for (let i = 0; i < cfg.layers; i++) {
      const n = (s) => `L.${i}.${s}`;
      both(rms(W(n("in_norm")).buf, this.x, this.h));
      if (cfg.full[i]) {
        const kv = this.carry[i].kv;
        both(mm(n("qkv"), this.h, this.proj, 0));
        both({ k: "aprep", group: bg(this.p.APREP, [this.g, this.uni([{ f: cfg.eps }, 0, 0, 0]), this.proj, W(n("q_norm")).buf, W(n("k_norm")).buf, this.tok, this.rope]) });
        if (!this.attnTile) both({ k: "keys", group: bg(this.p.KEYS, [this.g, this.proj, this.conv]) });
        one.push({ k: "savekv", group: bg(this.p.SAVE_KV, [this.g, this.proj, kv, this.tok, this.segs]) });
        // Stage 1 only supplies carries. Its final hidden rows have no consumer.
        if (i === cfg.layers - 1) stateComplete = true;
        const attn = this.attnTile ? [this.g, this.proj, kv, this.ablocks, this.segs, this.core] : [this.g, this.proj, kv, this.tok, this.segs, this.core, this.conv];
        both({ k: "attn", group: bg(this.p.ATTN, attn) });
        both(mm(n("o"), this.core, this.x, 1));
      } else {
        const { state, tail } = this.carry[i];
        both(mm(n("qkvz"), this.h, this.proj, 0));
        both({ k: "gates", group: bg(this.p.GATES, [this.g, this.h, this.ab[i], W(n("dt_bias")).buf, W(n("neg_a")).buf, this.gates]) });
        both({ k: "conv", group: bg(this.p.CONV, [this.g, this.proj, W(n("conv")).buf, this.tok, this.segs, tail, this.conv]) });
        one.push({ k: "savetail", group: bg(this.p.SAVE_TAIL, [this.g, this.proj, this.segs, tail]) });
        both({ k: "recur", group: bg(this.p.RECUR, [this.g, this.conv, this.gates, this.segs, state, this.core]) });
        if (i === cfg.layers - 1) stateComplete = true;
        both({ k: "gnorm", group: bg(this.p.GNORM, [this.g, this.uni([{ f: cfg.eps }, 0, 0, 0]), this.proj, W(n("gnorm")).buf, this.core]) });
        both(mm(n("out"), this.core, this.x, 1));
      }
      both(rms(W(n("post_norm")).buf, this.x, this.h));
      both(mm(n("gate_up"), this.h, this.up, 0));
      both({ k: "silumul", I: cfg.intermediate, group: bg(this.p.SILUMUL, [this.g, this.uni([cfg.intermediate, 0, 0, 0]), this.up, this.act]) });
      both(mm(n("down"), this.act, this.x, 1));
    }
    two.push({ k: "gather", group: bg(this.p.GATHER, [this.g, this.x, this.rows, this.gathered]) });
    this.ops = [one, two];
  }

  encode(enc, ops, T, S) {
    const prof = this.profiler;
    let pass = prof ? null : enc.beginComputePass();
    for (const op of ops) {
      if (prof) pass = prof.pass(enc, op.label || op.k);
      pass.setBindGroup(0, op.group);
      switch (op.k) {
        case "mm": {
          const kernel = this.gpuKernel === "auto" ? selectedMatmulKernel(this.tuning.selection, op.N, op.K, T) : this.gpuKernel;
          encodeMatmul(pass, this.tuning.pipelines[kernel], op, T);
          break;
        }
        case "rms":
          pass.setPipeline(this.p.RMS);
          pass.dispatchWorkgroups(T);
          break;
        case "gates":
          pass.setPipeline(this.p.GATES);
          pass.dispatchWorkgroups(T);
          break;
        case "conv":
          pass.setPipeline(this.p.CONV);
          pass.dispatchWorkgroups(T, this.dims.linear / 256);
          break;
        case "savetail":
          pass.setPipeline(this.p.SAVE_TAIL);
          pass.dispatchWorkgroups(S, (3 * this.dims.linear) / 256);
          break;
        case "recur":
          pass.setPipeline(this.p.RECUR);
          pass.dispatchWorkgroups(S, this.cfg.lin_heads, this.recurTiles);
          break;
        case "gnorm":
          pass.setPipeline(this.p.GNORM);
          pass.dispatchWorkgroups(T, this.cfg.lin_heads);
          break;
        case "aprep":
          pass.setPipeline(this.p.APREP);
          pass.dispatchWorkgroups(T, this.cfg.heads + this.cfg.kv_heads);
          break;
        case "savekv":
          pass.setPipeline(this.p.SAVE_KV);
          pass.dispatchWorkgroups(T, this.dims.kv / 256);
          break;
        case "keys":
          pass.setPipeline(this.p.KEYS);
          pass.dispatchWorkgroups(Math.ceil(T / 16), this.dims.kv / 32);
          break;
        case "attn":
          pass.setPipeline(this.p.ATTN);
          // tiled: at most one partial block per segment beyond T / 8 full ones; spares return at once
          if (this.attnTile) pass.dispatchWorkgroups(Math.ceil(T / 8) + S, this.cfg.kv_heads);
          else pass.dispatchWorkgroups(T, this.cfg.heads);
          break;
        case "silumul": {
          pass.setPipeline(this.p.SILUMUL);
          const n = Math.ceil((T * op.I) / 256);
          pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
          break;
        }
        case "gather":
          pass.setPipeline(this.p.GATHER);
          pass.dispatchWorkgroups(this.R);
          break;
      }
      if (prof) {
        pass.end();
        pass = null;
      }
    }
    pass?.end();
  }

  /**
   * Runs both stages. `x1`/`x2` are the embedded states and branches, `batch` the table from
   * `kevala_kev_prepare`. Resolves to the pointer rows (`rows * hidden` f32).
   */
  async forward(x1, x2, batch, ids) {
    try {
      return await this.forwardPass(x1, x2, batch, ids);
    } catch (e) {
      // A partial or rejected pass may leave both new and retained carries invalid.
      this.lru = [];
      throw e;
    }
  }

  async forwardPass(x1, x2, batch, ids) {
    const d = this.device;
    const D = this.cfg.hidden;
    const { T2, branches, rows } = batch;
    const B = branches.length;
    this.R = rows.length;
    if (this.stateCacheEnabled === false) this.lru = [];
    // size carries for the worst case first: growing them drops the cache, and the plan must
    // only reference slots that will still hold their data
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    let plan, P, T1, setupError;
    try {
      this.ensureCarries(CACHE_SLOTS + batch.states.length, CACHE_SLOTS * SLOT_ROWS + batch.T1);
      plan = this.plan(batch, ids);
      P = plan.segs.length;
      T1 = plan.segs.reduce((a, g) => a + g.len, 0);
      this.ensure(T1, T2, Math.max(P, 1), Math.max(P, B), rows.length);
    } catch (e) {
      setupError = e;
    }
    const oom = await d.popErrorScope();
    const setup = await d.popErrorScope();
    if (setupError || oom || setup) {
      this.resetDynamicBuffers();
      throw setupError || Object.assign(new Error(`WebGPU setup: ${(oom || setup).message}`), { code: "WEBGPU_INIT" });
    }
    const q = d.queue;
    // stage 1: the planned segments, their embedded rows gathered from the full state rows
    const seg1 = new Uint32Array(this.cap.S * 8);
    const tok1 = new Uint32Array(Math.max(1, T1) * 4);
    const x1c = new Float32Array(Math.max(1, T1) * D);
    let at = 0;
    plan.segs.forEach((g, si) => {
      seg1.set([at, g.len, g.parent, g.pstart, g.plen, g.dst, g.kvdst, 0], si * 8);
      x1c.set(x1.subarray(g.src * D, (g.src + g.len) * D), at * D);
      for (let r = 0; r < g.len; r++) tok1.set([si, r, g.pos0 + r, 0], (at + r) * 4);
      at += g.len;
    });
    // stage 2: every branch continues its request's carry
    const seg2 = new Uint32Array(this.cap.S * 8);
    const tok2 = new Uint32Array(T2 * 4);
    branches.forEach(([start, len, req], b) => {
      const c = plan.carries[req];
      // an empty state (a folded single-question row) has no carry to continue
      seg2.set([start, len, c ? c.slot : 0xffffffff, c ? c.kvBase : 0, c ? c.len : 0, 0, 0, 0], b * 8);
      for (let r = 0; r < len; r++) tok2.set([b, r, (c ? c.len : 0) + r, 0], (start + r) * 4);
    });
    // every pass runs in error scopes: a rejected command buffer must be an error, not the
    // previous pass's rows read back
    const checking = true;
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    q.writeBuffer(this.g, 0, new Uint32Array([T1, rows.length, P, 1]));
    q.writeBuffer(this.g2, 0, new Uint32Array([T2, rows.length, B, 2]));
    q.writeBuffer(this.x, 0, x1c);
    q.writeBuffer(this.x2, 0, x2);
    q.writeBuffer(this.tok, 0, tok1);
    q.writeBuffer(this.tok2, 0, tok2);
    q.writeBuffer(this.segs, 0, seg1);
    q.writeBuffer(this.segs2, 0, seg2);
    if (this.attnTile) {
      q.writeBuffer(this.ablocks, 0, attnBlocks(plan.segs.map((g) => g.len), this.blockCap));
      q.writeBuffer(this.ablocks2, 0, attnBlocks(branches.map(([, len]) => len), this.blockCap));
    }
    q.writeBuffer(this.rows, 0, new Uint32Array(rows));
    let enc = d.createCommandEncoder();
    // long passes go out in chunks so the page keeps rendering (see YIELD_TOKENS)
    const yieldable = !this.profiler && T1 + T2 > YIELD_TOKENS;
    const flush = async () => {
      enc = await endChunk(d, enc);
    };
    // an extension starts from its parent's keys and values, copied next to its own (through a
    // staging buffer: WebGPU copies cannot read and write the same buffer)
    for (const cp of plan.copies) {
      if (cp.from === cp.to || !cp.rows) continue;
      for (const c of this.carry) {
        if (!c.kv) continue;
        const stride = this.dims.kv * 4;
        enc.copyBufferToBuffer(c.kv, cp.from * stride, this.kvStage, 0, cp.rows * stride);
        enc.copyBufferToBuffer(this.kvStage, 0, c.kv, cp.to * stride, cp.rows * stride);
      }
    }
    if (T1 > 0) {
      const parts = yieldable ? chunks(this.ops[0], Math.max(1, Math.round((YIELD_CHUNKS * T1) / (T1 + T2)))) : [this.ops[0]];
      for (let i = 0; i < parts.length; i++) {
        if (i) await flush();
        this.encode(enc, parts[i], T1, P);
      }
      if (yieldable) await flush();
    }
    // switch the shared buffers over to stage 2
    enc.copyBufferToBuffer(this.x2, 0, this.x, 0, T2 * D * 4);
    enc.copyBufferToBuffer(this.tok2, 0, this.tok, 0, T2 * 16);
    enc.copyBufferToBuffer(this.segs2, 0, this.segs, 0, this.cap.S * 32);
    if (this.attnTile) enc.copyBufferToBuffer(this.ablocks2, 0, this.ablocks, 0, this.blockCap * 16);
    enc.copyBufferToBuffer(this.g2, 0, this.g, 0, 16);
    const parts2 = yieldable ? chunks(this.ops[1], Math.max(1, Math.round((YIELD_CHUNKS * T2) / (T1 + T2)))) : [this.ops[1]];
    for (let i = 0; i < parts2.length; i++) {
      if (i) await flush();
      this.encode(enc, parts2[i], T2, B);
    }
    this.profiler?.finish(enc);
    const bytes = rows.length * D * 4;
    enc.copyBufferToBuffer(this.gathered, 0, this.readback, 0, bytes);
    q.submit([enc.finish()]);
    if (this.profiler) this.lastProfile = await this.profiler.collect();
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

/** Parses the table `kevala_kev_prepare` writes. */
export function parseKevBatch(u) {
  let i = 0;
  const R = u[i++];
  const T1 = u[i++];
  const T2 = u[i++];
  const n = u[i++];
  const states = [];
  for (let r = 0; r < R; r++) states.push([u[i++], u[i++]]);
  const B = u[i++];
  const branches = [];
  for (let b = 0; b < B; b++) branches.push([u[i++], u[i++], u[i++]]);
  const rows = Array.from(u.subarray(i, i + n));
  return { R, T1, T2, states, branches, rows };
}

export function kevConfig(header) {
  const c = header.config;
  if (c.head_dim !== 256 || c.linear_key_head_dim !== 128 || c.linear_value_head_dim !== 128 || c.linear_conv_kernel_dim !== 4) {
    throw new Error("Qwen WebGPU requires 256-wide attention heads, 128-wide linear heads, and a four-token convolution");
  }
  return {
    hidden: c.hidden_size,
    layers: c.layer_types.length,
    full: c.layer_types.map((t) => t === "full_attention"),
    intermediate: c.intermediate_size,
    eps: c.rms_norm_eps,
    rope_theta: c.rope_theta,
    rotary: c.rotary_dim,
    heads: c.num_attention_heads,
    kv_heads: c.num_key_value_heads,
    lin_key_heads: c.linear_num_key_heads,
    lin_heads: c.linear_num_value_heads,
    max_state: c.max_state,
    max_branch: c.max_branch,
  };
}
