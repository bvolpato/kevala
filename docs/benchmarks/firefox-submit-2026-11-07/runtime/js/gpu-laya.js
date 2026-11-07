// Laya's trunk on WebGPU: 28 ModernBERT layers and 2 decision-head layers, int8 weights widened
// inside the matmul tiles. The coordinator (WebAssembly) embeds tokens before and scores markers
// after; everything in between runs on the GPU.

import { GpuWeights, SPLIT_SCRATCH, YIELD_CHUNKS, YIELD_TOKENS, chunks, dispatchMatmul, encodeMatmul, endChunk, matmulPipelines, pipeline, rowsPerThread } from "./gpu.js";
import { calibrateMatmul, selectedMatmulKernel } from "./gpu-tuning.js";

let U;

export class GpuTrunk {
  /**
   * `layout` is the single-shard trunk layout from `kevala_layouts`; tensors arrive through
   * `write(dst, bytes)` as the pack streams by.
   */
  constructor(gpu, layout, cfg) {
    U = GPUBufferUsage;
    this.device = gpu.device;
    this.name = gpu.name;
    this.gpuKernel = gpu.kernel || "auto";
    this.subgroup32 = !!gpu.subgroup32;
    // queries per attention workgroup: 64 for the tiled kernel, 16 for the others
    const tile = gpu.attentionTile || gpu.attentionTileShared;
    const tileFp32 = !tile && this.subgroup32 && this.device.limits.maxComputeWorkgroupStorageSize >= 30224 &&
      !!globalThis.navigator?.gpu?.wgslLanguageFeatures?.has("subgroup_id");
    this.attnKernel = tile ? ["attention_tile", { subgroups: !!gpu.attentionTile }] : tileFp32 ?
      ["attention_tile_f32", { subgroups: true }] : [this.subgroup32 ? "attention_subgroup" : "attention"];
    this.attnQueries = tile || tileFp32 ? 64 : 16;
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
    const kernel = (name, spec) => pipe(this.wgsl(name, spec), name);
    const [genericMm, pNorm, pRope, pAttn, pGeglu, pGather] = await Promise.all([
      matmulPipelines(d, this.wgsl),
      kernel("norm", { subgroups: this.subgroup32 }),
      kernel("rope"),
      kernel(...this.attnKernel),
      kernel("geglu"),
      kernel("gather"),
    ]);
    this.tuning = await calibrateMatmul(d, this.wgsl, this.weights, { kernel: this.gpuKernel, pipelines: { generic: genericMm } });
    this.mm = this.tuning.pipelines.generic;
    this.pNorm = pNorm;
    this.pRope = pRope;
    this.pAttn = pAttn;
    this.pGeglu = pGeglu;
    this.pGather = pGather;
    this.globals = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    this.compactGlobals = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
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
    const mm = (w, bias, input, output, mode, globals = this.globals) => {
      const e = T(w);
      const [N, K] = e.info.shape;
      const b = bias ? T(bias).buf : this.zeros;
      const u = this.uniform([N, K, mode, bias ? 1 : 0]);
      return {
        kind: "mm",
        label: "mm." + w.split(".").pop(),
        N,
        K,
        group: d.createBindGroup({ layout: this.mm.layout, entries: [globals, u, input, e.buf, e.sbuf, b, output, this.part].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
        reduce: d.createBindGroup({ layout: this.mm.reduceLayout, entries: [globals, u, this.part, b, output].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
      };
    };
    const norm = (w, b, input, output, typed = false, globals = this.globals) => ({
      kind: "norm",
      group: bg(this.pNorm, [globals, this.uniform([D, b ? 1 : 0, typed ? 1 : 0, { f: typed ? cfg.norm_eps : b ? cfg.head_norm_eps : cfg.norm_eps }]), input, w, b || this.zeros, output, this.tok, this.typeEmb]),
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
    this.compactHeadOps = null;
    for (let i = 0; i < cfg.head_layers; i++) {
      const n = (s) => `head.${i}.${s}`;
      ops.push(norm(T(n("norm1.w")).buf, T(n("norm1.b")).buf, this.x, this.h));
      ops.push(mm(n("in_proj"), n("in_proj.b"), this.h, this.qkv, 0));
      ops.push({ kind: "attn", group: bg(this.pAttn, [this.globals, this.uniform([D, 3 * D, 0, 0]), this.qkv, this.blocks, this.ctx]) });
      ops.push(mm(n("out_proj"), n("out_proj.b"), this.ctx, this.x, 1));
      if (i === cfg.head_layers - 1) {
        this.headTailAt = ops.length;
        // Gather before the final row-local norm and FFN. The residual is already compact,
        // so the final projection writes the scorer's rows without another gather.
        this.compactHeadOps = [
          { kind: "gather", label: "gather.head.compact", group: bg(this.pGather, [this.globals, this.x, this.rowsBuf, this.gathered]) },
          { ...norm(T(n("norm2.w")).buf, T(n("norm2.b")).buf, this.gathered, this.h, false, this.compactGlobals), compact: true, label: "norm.head.compact" },
          { ...mm(n("lin1"), n("lin1.b"), this.h, this.act, 2, this.compactGlobals), compact: true, label: "mm.head.compact.lin1" },
          { ...mm(n("lin2"), n("lin2.b"), this.act, this.gathered, 1, this.compactGlobals), compact: true, label: "mm.head.compact.lin2" },
        ];
      }
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
    // every pass runs in error scopes: a rejected command buffer must be an error, not the
    // previous pass's rows read back
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    const blocks = [];
    for (const sg of segs) for (let q0 = 0; q0 < sg.len; q0 += this.attnQueries) blocks.push(sg.start, sg.len, q0, 0);
    const nblocks = blocks.length / 4;
    q.writeBuffer(this.globals, 0, new Uint32Array([tokens, rows.length, nblocks, 0]));
    const compactHead = this.compactHeadOps && this.mm.f16 === false && this.mm.groups === 1 &&
      (this.gpuKernel === "auto" || this.gpuKernel === "generic") && rows.length > 0 && rows.length <= 16 && rows.length < tokens;
    // Preserve the original number of M tiles inside mm_splits, using existing R1 shaders.
    // Both shaders use virtualT for PART strides; dispatch only the selected physical rows.
    const virtualT = Math.min(tokens, Math.ceil(tokens / (16 * rowsPerThread(tokens, this.mm))) * 16);
    if (compactHead) q.writeBuffer(this.compactGlobals, 0, new Uint32Array([virtualT, rows.length, 0, 0]));
    q.writeBuffer(this.blocks, 0, new Uint32Array(blocks));
    q.writeBuffer(this.x, 0, x);
    q.writeBuffer(this.tok, 0, tok);
    q.writeBuffer(this.rowsBuf, 0, new Uint32Array(rows));
    const prof = this.profiler;
    const heads = this.cfg.heads;
    const ops = compactHead ? [...this.ops.slice(0, this.headTailAt), ...this.compactHeadOps] : this.ops;
    const groups = tokens > YIELD_TOKENS && !prof ? chunks(ops, YIELD_CHUNKS) : [ops];
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
        case "mm": {
          if (op.compact) {
            dispatchMatmul(pass, this.mm.mm[1], this.mm.reduce[1], op, virtualT, this.mm.splitTarget, 1, 1, rows.length);
            break;
          }
          const kernel = this.gpuKernel === "auto" ? selectedMatmulKernel(this.tuning.selection, op.N, op.K, tokens) : this.gpuKernel;
          encodeMatmul(pass, this.tuning.pipelines[kernel], op, tokens);
          break;
        }
        case "norm":
          pass.setPipeline(this.pNorm);
          pass.dispatchWorkgroups(op.compact ? rows.length : tokens);
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
    const [oom, invalid] = [await d.popErrorScope(), await d.popErrorScope()];
    if (oom || invalid) throw Object.assign(new Error(`WebGPU: ${(oom || invalid).message}`), { code: "WEBGPU_INIT" });
    await this.readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(this.readback.getMappedRange(0, bytes).slice(0));
    this.readback.unmap();
    return out;
  }
}
