import { GpuKev } from "./gpu-kev.js";
import { inErrorScopes, pipeline, Profiler } from "./gpu.js";

const LIMIT = 256 * 1024 * 1024;
const GUARD = 0x3f100001;
const POISON = 0x7fc01234;
const EXPORTS = [
  "kevala_clef_head_layout", "kevala_clef_head_begin", "kevala_clef_head_status",
  "kevala_clef_head_memory_ready", "kevala_clef_head_finish", "kevala_clef_head_cancel",
];
const stale = () => Object.assign(new Error("Clef head request expired"), { code: "CLEF_HEAD_STALE" });
const require = (ok, message) => { if (!ok) throw new Error(`Clef head: ${message}`); };
const budget = (ok, message) => { if (!ok) throw Object.assign(new Error(`Clef head: ${message}`), { code: "CLEF_HEAD_INELIGIBLE" }); };
const positive = (n) => Number.isSafeInteger(n) && n > 0 && n <= 0xffffffff;
const REQUEST_FALLBACKS = new Set([
  "sequence-input-range", "memory-input-range", "normalized-memory-input-range",
  "record-count-outside-budget", "empty-record", "size-overflow", "scratch-budget",
  "allocation-failed", "hidden-or-lexical-shape",
]);

/** Record-local odd rows use the CPU dot4 reduction, including each odd record's last row. */
export function clefTailRows(tokens, segments) {
  require(positive(tokens) && Array.isArray(segments) && segments.length > 0, "invalid record geometry");
  const mask = new Uint32Array(tokens), rows = [];
  let end = 0;
  for (const count of segments) {
    require(positive(count) && end + count <= tokens, "invalid record length");
    end += count;
    if (count % 2) { mask[end - 1] = 1; rows.push(end - 1); }
  }
  require(end === tokens, "record lengths do not cover input");
  return { mask, rows: Uint32Array.from(rows) };
}

function geometry(layout) {
  require(layout?.version === 1 && layout.status === "ready" && layout.staticEligible === true, "ineligible static layout");
  const { hidden, width, projections } = layout;
  require(positive(hidden) && hidden <= 4096 && positive(width) && width <= 4096, "unsupported dimensions");
  require(Array.isArray(projections) && projections.length >= 1 && projections.length <= 33, "invalid projection count");
  projections.forEach((p, index) => {
    require(p.index === index && positive(p.n) && positive(p.k) && p.n % 16 === 0 && p.k % 32 === 0, "unsupported tiled projection");
    require(p.n === (index ? 2 * width : width) && p.k === (index ? width : hidden), "inconsistent projection geometry");
    require(index ? ["routing", "decoder"].includes(p.kind) : p.kind === "memory", "invalid projection kind");
    require(p.weights?.length === p.n * p.k && (p.bias === null || p.bias?.length === p.n), "invalid static descriptor");
  });
  return layout;
}

/** Owns only the staged head. The original hidden and lexical allocations stay with the caller. */
export class ClefHead {
  constructor(trunk, engine) {
    this.trunk = trunk;
    this.engine = engine;
    this.coord = engine.coord;
    this.device = trunk.device;
    this.generation = 0;
    this.active = null;
    this.disposed = false;
    this.validated = false;
    this.staticBuffers = [];
    this.scratchBuffers = [];
    this.diagnostics = { requested: engine.clefHead, selected: "cpu", validated: false, reason: engine.clefHead === "auto" ? "awaiting-first-eligible-request" : "cpu-mode", bytes: 0 };
    for (const name of EXPORTS) require(typeof this.coord.x[name] === "function", `WASM is missing ${name}; rebuild all WASM flavors`);
    this.device.lost?.then((info) => {
      this.disabled = `device-lost: ${info.message || info.reason}`;
      this.diagnostics.selected = "cpu";
      this.diagnostics.reason = this.disabled;
      this.destroyBuffers();
    });
  }

  call(name, ...args) {
    this.coord.call(() => this.coord.check(this.coord.x[name](...args)));
    return JSON.parse(this.coord.outText());
  }

  request() {
    require(!this.disposed, "model is disposed");
    this.engine.signal?.throwIfAborted();
    if (this.active) {
      this.call("kevala_clef_head_cancel", this.active.id);
      this.release(this.scratchBuffers);
      this.scratch = null;
    }
    this.active = null;
    return ++this.generation;
  }

  assert(lease) {
    if (this.disposed || lease !== this.generation || this.engine.signal?.aborted) throw stale();
  }

  status(lease, id, phase) {
    this.assert(lease);
    const result = this.call("kevala_clef_head_status", id);
    if (result.version !== 1 || result.status !== "ready" || result.id !== id || result.phase !== phase) throw stale();
    return result;
  }

  view(descriptor, expected) {
    require(descriptor && descriptor.length === expected && Number.isSafeInteger(descriptor.ptr)
      && descriptor.ptr > 0 && descriptor.ptr <= 0xffffffff && descriptor.ptr % 4 === 0
      && descriptor.ptr + expected * 4 <= this.coord.memory.byteLength, "invalid or expired WASM descriptor");
    return this.coord.u32(descriptor.ptr, expected);
  }

  cpu(lease, hp, hiddenN, lp, lexicalN) {
    this.assert(lease);
    return this.call("kevala_clef_finish", hp, hiddenN, lp, lexicalN);
  }

  release(buffers) { for (const b of buffers.splice(0)) b.destroy(); }

  destroyBuffers() {
    this.release(this.scratchBuffers);
    this.release(this.staticBuffers);
    this.scratch = null;
    this.weights = null;
    this.diagnostics.bytes = 0;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    ++this.generation;
    if (this.active) this.call("kevala_clef_head_cancel", this.active.id);
    this.active = null;
    this.destroyBuffers();
  }

  make(bytes, usage, label, owned) {
    const size = Math.max(16, bytes);
    require(Number.isSafeInteger(size) && size <= this.device.limits.maxBufferSize
      && (!(usage & GPUBufferUsage.STORAGE) || size <= this.device.limits.maxStorageBufferBindingSize), `${label} exceeds device limits`);
    const buffer = this.device.createBuffer({ size, usage, label: `clef.head.${label}` });
    owned.push(buffer);
    return buffer;
  }

  async setup(lease, stage) {
    if (this.weights) return;
    const { device } = this;
    const layout = geometry(this.call("kevala_clef_head_layout"));
    const e = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, ...(binding === 0 ? { minBindingSize: 32 } : {}) } });
    this.bindLayout = device.createBindGroupLayout({ label: "clef.head", entries: [e(0, "uniform"), e(1, "read-only-storage"), e(2, "read-only-storage"), e(3, "read-only-storage"), e(4, "read-only-storage"), e(5, "storage")] });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] });
    const bulk = this.trunk.wgsl("clef_head_projection"), tail = this.trunk.wgsl("clef_head_tail");
    this.pipelines = await inErrorScopes(device, () => Promise.all([
      pipeline(device, bulk, "clef_head_projection", pl), pipeline(device, tail, "clef_head_tail", pl),
    ]));
    this.status(lease, stage.id, "memory");
    const U = GPUBufferUsage;
    const bytes = layout.projections.reduce((sum, p) => sum + (p.weights.length + p.n) * 4, 0);
    require(bytes <= LIMIT, "static weights exceed the 256 MiB head budget");
    this.layout = layout;
    this.weights = layout.projections.map((p) => {
      const w = this.make(p.weights.length * 4, U.STORAGE | U.COPY_DST, `${p.index}.weights`, this.staticBuffers);
      const bias = this.make(p.n * 4, U.STORAGE | U.COPY_DST, `${p.index}.bias`, this.staticBuffers);
      device.queue.writeBuffer(w, 0, this.view(p.weights, p.n * p.k));
      device.queue.writeBuffer(bias, 0, p.bias ? this.view(p.bias, p.n) : new Uint32Array(p.n));
      return { ...p, w, bias };
    });
    this.diagnostics = { requested: this.engine.clefHead, selected: "cpu", validated: this.validated, reason: null, bytes, arithmetic: "fma", tile: 0 };
  }

  group(resources) {
    return this.device.createBindGroup({ layout: this.bindLayout, entries: resources.map((resource, binding) => ({ binding, resource: resource.buffer ? resource : { buffer: resource } })) });
  }

  prepare(stage) {
    const { tokens, segments } = stage, { hidden, width } = this.layout;
    budget(positive(tokens) && tokens <= 16384, "token count exceeds the head budget");
    require(stage.version === 1 && stage.phase === "memory" && stage.tile === 0 && positive(stage.id), "invalid memory phase");
    require(stage.s?.length === tokens * hidden && stage.memory?.length === tokens * width, "invalid memory shape");
    const key = `${tokens}:${segments.join(",")}`;
    if (this.scratch?.key === key) return this.scratch;
    const U = GPUBufferUsage, device = this.device;
    const alignment = Math.max(256, device.limits.minStorageBufferOffsetAlignment || 256);
    const guardBytes = Math.ceil(256 / alignment) * alignment;
    const outputBytes = (p) => tokens * p.n * 4 + 2 * guardBytes;
    const routing = this.weights.filter((p) => p.kind === "routing").length;
    const kvBytes = this.weights.slice(1).reduce((sum, p) => sum + outputBytes(p), 0);
    const readBytes = Math.max(outputBytes(this.weights[0]), kvBytes);
    const poisonBytes = Math.max(...this.weights.map((p) => tokens * p.n * 4));
    const planned = this.staticBuffers.reduce((sum, b) => sum + b.size, 0) + tokens * (hidden + routing * width) * 4
      + this.weights.reduce((sum, p) => sum + outputBytes(p) + 32, 0) + readBytes + poisonBytes + Math.max(16, tokens * 4) + Math.max(16, segments.length * 4)
      + (this.trunk.profiler ? this.weights.length * 64 : 0);
    budget(planned <= LIMIT, "scratch exceeds the 256 MiB head budget");
    const tails = clefTailRows(tokens, segments);
    budget(this.weights.every((p) => Math.ceil(tails.rows.length * p.n / 64) <= device.limits.maxComputeWorkgroupsPerDimension), "odd-tail dispatch exceeds device limits");
    this.release(this.scratchBuffers);
    this.scratch = null;
    const make = (bytes, usage, label) => this.make(bytes, usage, label, this.scratchBuffers);
    const x = make(tokens * hidden * 4, U.STORAGE | U.COPY_DST, "S");
    const mask = make(tails.mask.byteLength, U.STORAGE | U.COPY_DST, "odd-mask");
    const compact = make(tails.rows.byteLength, U.STORAGE | U.COPY_DST, "odd-rows");
    const poison = make(poisonBytes, U.COPY_SRC | U.COPY_DST, "poison");
    const read = make(readBytes, U.MAP_READ | U.COPY_DST, "readback");
    device.queue.writeBuffer(mask, 0, tails.mask);
    if (tails.rows.length) device.queue.writeBuffer(compact, 0, tails.rows);
    device.queue.writeBuffer(poison, 0, new Uint32Array(poisonBytes / 4).fill(POISON));
    const guards = Uint32Array.from({ length: guardBytes / 4 }, (_, i) => GUARD + i % 4);
    const ops = [];
    let offset = 0;
    for (const p of this.weights) {
      const bytes = outputBytes(p), count = tokens * p.n;
      const y = make(bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, `${p.index}.output`);
      const params = make(32, U.UNIFORM | U.COPY_DST, `${p.index}.params`);
      const input = p.index === 0 ? { buffer: x } : p.kind === "routing"
        ? { buffer: make(tokens * width * 4, U.STORAGE | U.COPY_DST, `${p.index}.normalized-memory`) }
        : { buffer: ops[0].y, offset: guardBytes, size: tokens * width * 4 };
      device.queue.writeBuffer(params, 0, new Uint32Array([tokens, p.n, p.k, guardBytes / 4, tails.rows.length, 0, 0, 0]));
      device.queue.writeBuffer(y, 0, guards);
      device.queue.writeBuffer(y, guardBytes + count * 4, guards);
      const resources = [params, input, p.w, p.bias, mask, y];
      const group = this.group(resources);
      resources[4] = compact;
      const tailGroup = this.group(resources);
      const readOffset = p.index ? offset : 0;
      if (p.index) offset += bytes;
      ops.push({ ...p, input, y, group, tailGroup, count, bytes, readOffset });
    }
    this.diagnostics.bytes = this.staticBuffers.reduce((sum, b) => sum + b.size, 0) + this.scratchBuffers.reduce((sum, b) => sum + b.size, 0);
    return this.scratch = { key, tokens, segments, guardBytes, tailCount: tails.rows.length, x, poison, read, ops };
  }

  async wave(lease, stage, ops, upload = () => {}) {
    const s = this.scratch;
    let prof, failure;
    try {
      await inErrorScopes(this.device, async () => {
        this.status(lease, stage.id, stage.phase);
        // Head waves own their queries, including when an expired request skips collection.
        if (this.trunk.profiler && ops.length) prof = new Profiler(this.device, ops.length * 2);
        upload();
        const encoder = this.device.createCommandEncoder();
        for (const op of ops) {
          encoder.copyBufferToBuffer(s.poison, 0, op.y, s.guardBytes, op.count * 4);
          const label = op.index === 0 ? "memory" : `${op.kind}.${op.index}`;
          const pass = prof?.pass(encoder, "clef.head." + label) || encoder.beginComputePass();
          pass.setPipeline(this.pipelines[0]);
          pass.setBindGroup(0, op.group);
          pass.dispatchWorkgroups(op.n / 16, Math.ceil(s.tokens / 16));
          pass.end();
          if (s.tailCount) {
            const tail = prof?.pass(encoder, "clef.head." + label + ".tail") || encoder.beginComputePass();
            tail.setPipeline(this.pipelines[1]);
            tail.setBindGroup(0, op.tailGroup);
            tail.dispatchWorkgroups(Math.ceil(s.tailCount * op.n / 64));
            tail.end();
          }
          encoder.copyBufferToBuffer(op.y, 0, s.read, op.readOffset, op.bytes);
        }
        prof?.finish(encoder);
        this.device.queue.submit([encoder.finish()]);
        await s.read.mapAsync(GPUMapMode.READ);
      });
      this.assert(lease);
      if (prof) {
        const profile = await prof.collect();
        this.assert(lease);
        for (const [name, ms] of Object.entries(profile)) this.lastProfile[name] = (this.lastProfile[name] || 0) + ms;
      }
      this.status(lease, stage.id, stage.phase);
      const bits = new Uint32Array(s.read.getMappedRange());
      for (const op of ops) {
        const base = op.readOffset / 4, guard = s.guardBytes / 4;
        for (let i = 0; i < guard; i++) require(bits[base + i] === GUARD + i % 4
          && bits[base + guard + op.count + i] === GUARD + i % 4, "output canary overwritten");
      }
      return bits;
    } catch (error) {
      failure = error;
      try { s.read.unmap(); } catch { /* Keep the original GPU or request error. */ }
      throw error;
    } finally {
      // release() also destroys a profiler whose pending labels never reached collect().
      try { prof?.release(); } catch (error) { if (!failure) throw error; }
    }
  }

  copy(bits, ops, targets) {
    const s = this.scratch;
    for (let i = 0; i < ops.length; i++) {
      const op = ops[i], base = (op.readOffset + s.guardBytes) / 4;
      this.view(targets[i], op.count).set(bits.subarray(base, base + op.count));
    }
  }

  fallback(reason) {
    this.active = null; // A Rust fallback already dropped this stage, preserving prepared records.
    this.diagnostics.reason = reason;
    if (!REQUEST_FALLBACKS.has(reason)) {
      this.disabled = reason;
      this.destroyBuffers();
    }
  }

  /** Any matching failed stage is canceled before the unchanged whole CPU finish. */
  async finish(lease, hp, hiddenN, lp, lexicalN) {
    this.assert(lease);
    this.lastProfile = {};
    this.diagnostics.requested = this.engine.clefHead;
    this.diagnostics.selected = "cpu";
    let cpuStarted = false;
    const cpu = () => { cpuStarted = true; return this.cpu(lease, hp, hiddenN, lp, lexicalN); };
    const bypass = this.engine.clefHead !== "auto" ? "cpu-mode"
      : this.disabled || (this.engine.flavor !== "relaxed" ? "unsupported-coordinator-flavor"
        : this.coord.tile !== 0 ? "unproved-cpu-tile" : null);
    if (bypass) { this.diagnostics.reason = bypass; return cpu(); }
    let stage, consumed = false;
    try {
      stage = this.call("kevala_clef_head_begin", hp, hiddenN, lp, lexicalN, this.validated ? 0 : 1);
      if (stage.status === "fallback") { this.fallback(stage.reason); return cpu(); }
      require(stage.status === "ready" && positive(stage.id), "invalid begin response");
      this.active = { id: stage.id, phase: "memory", lease };
      if (stage.tile !== 0) { this.diagnostics.reason = "unproved-cpu-tile"; this.cancel(lease); return cpu(); }
      await inErrorScopes(this.device, async () => {
        await this.setup(lease, stage);
        this.status(lease, stage.id, "memory");
        this.prepare(stage);
      });
      this.status(lease, stage.id, "memory");
      const s = this.scratch;
      s.memoryPtr = stage.memory.ptr;
      let bits = await this.wave(lease, stage, [s.ops[0]], () => {
        this.device.queue.writeBuffer(s.x, 0, this.view(stage.s, s.tokens * this.layout.hidden));
      });
      try {
        this.status(lease, stage.id, "memory");
        this.copy(bits, [s.ops[0]], [stage.memory]);
        stage = this.call("kevala_clef_head_memory_ready", stage.id);
      } finally { s.read.unmap(); }
      if (stage.status === "stale") throw stale();
      if (stage.status === "fallback") { this.fallback(stage.reason); return cpu(); }
      require(stage.status === "ready" && stage.phase === "kv" && stage.id === this.active.id
        && stage.tokens === s.tokens && JSON.stringify(stage.segments) === JSON.stringify(s.segments), "invalid KV phase");
      this.active.phase = "kv";
      const ops = s.ops.slice(1);
      require(Array.isArray(stage.projections) && stage.projections.length === ops.length, "invalid KV count");
      stage.projections.forEach((p, i) => {
        const op = ops[i];
        require(p.index === op.index && p.input?.length === s.tokens * op.k
          && p.inputSource === (op.kind === "routing" ? "normalized-memory" : "memory"), "invalid KV input");
        require(op.kind !== "decoder" || p.input.ptr === s.memoryPtr, "decoder input does not alias resident memory");
      });
      bits = await this.wave(lease, stage, ops, () => {
        stage.projections.forEach((p, i) => {
          if (ops[i].kind === "routing") this.device.queue.writeBuffer(ops[i].input.buffer, 0, this.view(p.input, s.tokens * ops[i].k));
        });
      });
      let result;
      try {
        this.status(lease, stage.id, "kv");
        this.copy(bits, ops, stage.projections.map((p) => p.output));
        result = this.call("kevala_clef_head_finish", stage.id);
      } finally { s.read.unmap(); }
      if (result?.status === "stale") throw stale();
      this.active = null;
      if (result?.status === "fallback") { this.fallback(result.reason); return cpu(); }
      consumed = true;
      require(Array.isArray(result), "invalid finish response");
      this.validated = true;
      this.diagnostics.selected = "gpu";
      this.diagnostics.validated = true;
      this.diagnostics.reason = null;
      return result;
    } catch (error) {
      this.assert(lease);
      if (error.code === "CLEF_HEAD_STALE" || consumed || cpuStarted || !stage) throw error;
      if (this.active) this.cancel(lease);
      if (error.code === "CLEF_HEAD_INELIGIBLE") {
        this.diagnostics.reason = error.message;
        return cpu();
      }
      this.disabled = error.message || String(error);
      this.diagnostics.reason = this.disabled;
      this.destroyBuffers();
      return cpu();
    }
  }

  cancel(lease) {
    const active = this.active;
    if (!active) return;
    this.status(lease, active.id, active.phase);
    const result = this.call("kevala_clef_head_cancel", active.id);
    if (result.status !== "canceled" || result.id !== active.id) throw stale();
    this.active = null;
  }
}

export class GpuClef extends GpuKev {
  constructor(gpu, layout, cfg, engine) {
    super(gpu, layout, cfg);
    this.head = new ClefHead(this, engine);
    this.stateCacheEnabled = false;
  }

  dispose() { this.head.dispose(); }
}
