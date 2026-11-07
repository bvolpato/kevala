import assert from "node:assert/strict";
import test from "node:test";
import { ClefHead, clefTailRows } from "../js/src/gpu-clef-head.js";
import { requiredExports, assertModule } from "../scripts/lib/wasm-abi.mjs";
import { f32Fma, projectionReference } from "../dev/clef-head-check.js";
import clef from "../js/src/archs/clef.js";

function fixture(options = {}) {
  let memory = new ArrayBuffer(65536), out, active = null, allocation = 24576;
  const events = [], validates = [], memoryBits = Uint32Array.from({ length: 96 }, (_, i) => i % 7 ? 0x3f000001 + i : 0x80000000);
  const kvBits = Uint32Array.from({ length: 192 }, (_, i) => i % 5 ? 0xbf000001 + i : 0);
  const descriptor = (ptr, length) => ({ ptr, length });
  const stage = () => ({ version: 1, status: "ready", phase: "memory", tokens: 3, segments: [1, 2], tile: options.tile ?? 0,
    id: 11, s: descriptor(256, 96), memory: descriptor(4096, 96), reference: descriptor(8192, 96) });
  const result = (value) => { out = value; return 0; };
  const x = {
    kevala_clef_prepare: () => result({ ids: [1, 2, 3], option_spans: [[0, 1]] }),
    kevala_free() {},
    kevala_clef_head_layout: () => result({}),
    kevala_clef_head_begin: (_hp, _hn, _lp, _ln, validate) => {
      events.push("begin"); validates.push(validate);
      if (options.beginFallback) return result({ version: 1, status: "fallback", reason: "sequence-input-range" });
      active = "memory"; return result(stage());
    },
    kevala_clef_head_status: (id) => {
      events.push(`status:${active}`);
      if (options.growOnStatus) { const next = new ArrayBuffer(memory.byteLength + 64); new Uint8Array(next).set(new Uint8Array(memory)); memory = next; }
      return result(active ? { version: 1, status: "ready", phase: active, id } : { version: 1, status: "stale", id });
    },
    kevala_clef_head_memory_ready: (id) => {
      events.push("memory-ready");
      assert.deepEqual(new Uint32Array(memory, 4096, 96), memoryBits);
      if (options.memoryFallback) { active = null; return result({ version: 1, status: "fallback", reason: "memory-reference-bits" }); }
      active = "kv";
      return result({ version: 1, status: "ready", phase: "kv", tokens: 3, segments: [1, 2], id,
        projections: [{ index: 1, inputSource: "memory", input: descriptor(4096, 96), output: descriptor(12288, 192), reference: descriptor(16384, 192) }] });
    },
    kevala_clef_head_finish: () => {
      events.push("finish"); assert.deepEqual(new Uint32Array(memory, 12288, 192), kvBits); active = null;
      return result([{ answer: "gpu" }]);
    },
    kevala_clef_head_cancel: () => { events.push("cancel"); const old = active; active = null; return result({ version: 1, status: old ? "canceled" : "stale", id: 11 }); },
    kevala_clef_finish: () => { events.push("cpu"); active = null; if (options.cpuError) throw new Error("original CPU finish failed"); return result([{ answer: "cpu" }]); },
  };
  const coord = { x, tile: options.coordTile ?? 0, get memory() { return memory; },
    call: (fn) => fn(), check: (status) => assert.equal(status, 0), outText: () => JSON.stringify(out),
    u32: (ptr, length) => new Uint32Array(memory, ptr, length),
    withInput: (bytes, fn) => fn(0, bytes.length),
    put(bytes) { const ptr = allocation; allocation += Math.ceil(bytes.length / 16) * 16; new Uint8Array(memory, ptr, bytes.length).set(bytes); return [ptr, bytes.length]; } };
  const device = { queue: { writeBuffer() {} }, lost: new Promise(() => {}) };
  const engine = { coord, clefHead: options.mode ?? "auto", flavor: options.flavor ?? "relaxed" };
  const head = new ClefHead({ device }, engine);
  head.setup = async () => {
    if (head.weights) return;
    events.push("upload-weights");
    head.weights = [];
    head.layout = { hidden: 32, width: 32 };
    head.diagnostics.bytes = 64;
  };
  head.prepare = () => head.scratch = { tokens: 3, segments: [1, 2], guardBytes: 0, x: {}, read: { unmap() {} },
    ops: [{ index: 0, count: 96, readOffset: 0 }, { index: 1, kind: "decoder", k: 32, count: 192, readOffset: 0 }] };
  head.wave = async (_lease, phase) => {
    events.push(`wave:${phase.phase}`);
    await options.beforeWave?.(head, phase);
    if (options.gpuError) throw new Error("GPU map failed");
    if (options.expire) active = null;
    return phase.phase === "memory" ? memoryBits : kvBits;
  };
  return { head, engine, events, validates, coord, expire: () => active = null };
}

function model(f) {
  f.engine.header = { config: { hidden_size: 32 } };
  f.engine.gpu = f.head.trunk;
  Object.assign(f.engine.gpu, { head: f.head, gpuKernel: "auto", forwardRows: async () => new Float32Array(96), weightRows: async (_name, ids) => new Float32Array(ids.length * 32) });
  return f.engine;
}

const warmup = { state: "warm up", questions: { q: { type: "noul", instructions: "Is this a warm up?" } } };

test("record-local odd tails preserve even, odd and multiple record boundaries", () => {
  for (const segments of [[2], [1], [1, 2, 3, 4, 5], [158], [251], [659]]) {
    const tokens = segments.reduce((a, b) => a + b, 0), { mask, rows } = clefTailRows(tokens, segments);
    const expected = []; let end = 0;
    for (const count of segments) { end += count; if (count % 2) expected.push(end - 1); }
    assert.deepEqual([...rows], expected);
    assert.deepEqual([...mask], Array.from({ length: tokens }, (_, i) => Number(expected.includes(i))));
  }
  assert.throws(() => clefTailRows(4, [1, 2]), /do not cover/);
  assert.throws(() => clefTailRows(4, [0, 4]), /invalid record length/);
});

test("the first actual stage validates every supplied bit, then later stages use validate0", async () => {
  const f = fixture({ growOnStatus: true });
  for (let i = 0; i < 2; i++) assert.deepEqual(await f.head.finish(f.head.request(), 32, 96, 64, 32), [{ answer: "gpu" }]);
  assert.deepEqual(f.validates, [1, 0]);
  assert.equal(f.head.validated, true);
  assert.equal(f.events.includes("cpu"), false);
  assert.equal(f.events.includes("cancel"), false);
});

test("auto initialization validates and uploads the head before the next user request", async () => {
  const f = fixture(), engine = model(f);
  assert.deepEqual((await clef.run(engine, [warmup])).responses, [{ answer: "gpu" }]);
  assert.deepEqual(f.validates, [1]);
  assert.equal(f.head.diagnostics.selected, "gpu");
  assert.equal(f.head.diagnostics.validated, true);
  assert.equal(f.head.diagnostics.reason, null);
  assert.equal(f.head.diagnostics.bytes, 64);
  assert.deepEqual((await clef.run(engine, [{ ...warmup, state: "real request" }])).responses, [{ answer: "gpu" }]);
  assert.deepEqual(f.validates, [1, 0]);
  assert.equal(f.events.filter((event) => event === "upload-weights").length, 1);
});

test("an ineligible initialization request leaves validate1 for the first eligible user request", async () => {
  const options = { beginFallback: true }, f = fixture(options), engine = model(f);
  assert.deepEqual((await clef.run(engine, [warmup])).responses, [{ answer: "cpu" }]);
  assert.equal(f.head.diagnostics.selected, "cpu");
  assert.equal(f.head.diagnostics.reason, "sequence-input-range");
  assert.equal(f.head.validated, false);
  options.beginFallback = false;
  assert.deepEqual((await clef.run(engine, [{ ...warmup, state: "real request" }])).responses, [{ answer: "gpu" }]);
  assert.deepEqual(f.validates, [1, 1]);
});

test("explicit CPU mode keeps initialization and user requests on the original finish", async () => {
  const f = fixture({ mode: "cpu" }), engine = model(f);
  for (const state of ["warm up", "real request"]) assert.deepEqual((await clef.run(engine, [{ ...warmup, state }])).responses, [{ answer: "cpu" }]);
  assert.deepEqual(f.validates, []);
  assert.deepEqual(f.events, ["cpu", "cpu"]);
  assert.equal(f.head.diagnostics.selected, "cpu");
  assert.equal(f.head.diagnostics.reason, "cpu-mode");
  assert.equal(f.head.validated, false);
});

test("CPU mode, nonrelaxed flavors and unproved tile1 retain the original whole finish", async () => {
  for (const options of [{ mode: "cpu" }, { flavor: "base" }, { flavor: "simd" }, { coordTile: 1 }]) {
    const f = fixture(options);
    assert.deepEqual(await f.head.finish(f.head.request(), 32, 96, 64, 32), [{ answer: "cpu" }]);
    assert.deepEqual(f.events, ["cpu"]);
    assert.equal(f.head.validated, false);
    assert.equal(f.head.diagnostics.selected, "cpu");
    assert.equal(f.head.diagnostics.reason, options.mode === "cpu" ? "cpu-mode" : options.flavor ? "unsupported-coordinator-flavor" : "unproved-cpu-tile");
  }
  const f = fixture({ tile: 1 });
  await f.head.finish(f.head.request(), 32, 96, 64, 32);
  assert.deepEqual(f.events.filter((e) => !e.startsWith("status:")), ["begin", "cancel", "cpu"]);
  assert.equal(f.head.diagnostics.reason, "unproved-cpu-tile");
});

test("a GPU failure cancels only the matching stage before CPU fallback", async () => {
  const f = fixture({ gpuError: true });
  assert.deepEqual(await f.head.finish(f.head.request(), 32, 96, 64, 32), [{ answer: "cpu" }]);
  assert.deepEqual(f.events.slice(-3), ["status:memory", "cancel", "cpu"]);
  assert.equal(f.head.validated, false);
});

test("Rust eligibility and reference fallbacks preserve records without canceling an expired id", async () => {
  for (const options of [{ beginFallback: true }, { memoryFallback: true }]) {
    const f = fixture(options);
    assert.deepEqual(await f.head.finish(f.head.request(), 32, 96, 64, 32), [{ answer: "cpu" }]);
    assert.equal(f.events.includes("cancel"), false);
    assert.equal(f.head.validated, false);
  }
});

test("finite input range fallback keeps the head reusable; numeric failure destroys its buffers", () => {
  const f = fixture();
  let destroyed = 0;
  f.head.staticBuffers.push({ destroy() { destroyed++; } });
  f.head.fallback("memory-input-range");
  assert.equal(f.head.disabled, undefined);
  assert.equal(destroyed, 0);
  f.head.fallback("memory-reference-bits");
  assert.equal(f.head.disabled, "memory-reference-bits");
  assert.equal(destroyed, 1);
});

test("an error from the original CPU fallback is returned without a second finish", async () => {
  const f = fixture({ beginFallback: true, cpuError: true });
  await assert.rejects(f.head.finish(f.head.request(), 32, 96, 64, 32), /original CPU finish failed/);
  assert.equal(f.events.filter((event) => event === "cpu").length, 1);
});

test("malformed enormous token descriptors fail before allocating record masks", () => {
  const f = fixture();
  f.head.layout = { hidden: 32, width: 32 };
  assert.throws(() => ClefHead.prototype.prepare.call(f.head, { tokens: 0xffffffff, segments: [0xffffffff] }), { code: "CLEF_HEAD_INELIGIBLE" });
});

test("the independent F32 FMA reference rounds once and retains nearest-even and zero signs", () => {
  const ulp = 2 ** -23;
  assert.equal(f32Fma(1 + ulp, 1 - ulp, -1), -(2 ** -46));
  assert.equal(Math.fround(Math.fround((1 + ulp) * (1 - ulp)) - 1), 0);
  assert.ok(Object.is(f32Fma(-0, 1, -0), -0));
  assert.ok(Object.is(f32Fma(-0, 1, 0), 0));
  assert.equal(f32Fma(2 ** -149, 0.5, 0), 0);
  assert.equal(f32Fma(3 * 2 ** -149, 0.5, 0), 2 * 2 ** -149);
  assert.throws(() => f32Fma(Infinity, 1, 0), /finite operands/);
});

test("the numerical reference preserves separate bulk and record-tail reduction orders", () => {
  const k = 32, x = new Float32Array(3 * k).fill(1), w = new Float32Array(k);
  w.set([2 ** 24, 0, 0, 0, 1, 0, 0, 0, -(2 ** 24), 0, 0, 0, 1]);
  for (const fused of [false, true]) {
    const result = projectionReference(x, w, null, { t: 3, n: 1, k, segments: [2, 1] }, fused);
    assert.deepEqual([...result], [1, 1, 2]);
  }
});

test("stale status after map aborts before target writes and never takes CPU fallback", async () => {
  const f = fixture({ expire: true });
  await assert.rejects(f.head.finish(f.head.request(), 32, 96, 64, 32), { code: "CLEF_HEAD_STALE" });
  assert.deepEqual([...f.coord.u32(4096, 96)], Array(96).fill(0));
  assert.equal(f.events.includes("cpu"), false);
});

test("disposal or a superseding prepare during an await cannot write old stage pointers", async () => {
  for (const invalidate of [(head) => head.dispose(), (head) => head.request()]) {
    const f = fixture({ beforeWave: invalidate });
    await assert.rejects(f.head.finish(f.head.request(), 32, 96, 64, 32), { code: "CLEF_HEAD_STALE" });
    assert.deepEqual([...f.coord.u32(4096, 96)], Array(96).fill(0));
    assert.equal(f.events.includes("cpu"), false);
  }
});

test("real head waves own and release profiler queries on encoding, mapping, validation, stale and collection failures", async (t) => {
  for (const [name, value] of Object.entries({ GPUBufferUsage: { QUERY_RESOLVE: 1, COPY_SRC: 2, MAP_READ: 4, COPY_DST: 8 }, GPUMapMode: { READ: 1 } })) {
    const old = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    t.after(() => old ? Object.defineProperty(globalThis, name, old) : delete globalThis[name]);
  }
  for (const scenario of ["encode", "map", "validation", "stale", "collect", "success"]) {
    const f = fixture(), head = f.head, lease = head.request(), owned = [], queries = [];
    f.coord.x.kevala_clef_head_begin(32, 96, 64, 32, 1);
    let unmapped = 0, mapped = false;
    const buffer = (size, kind) => ({
      bytes: new Uint8Array(size), destroyed: false,
      async mapAsync() {
        if (kind === "output" && scenario === "map" || kind === "query-read" && scenario === "collect") throw new Error(`primary ${scenario}`);
        if (kind === "output") { mapped = true; if (scenario === "stale") head.generation++; }
      },
      getMappedRange: () => new ArrayBuffer(size),
      unmap() { if (kind === "output") { mapped = false; unmapped++; } },
      destroy() { this.destroyed = true; },
    });
    const read = buffer(136, "output"), poison = buffer(128, "poison"), y = buffer(136, "output-y");
    new Uint32Array(y.bytes.buffer).set([0x3f100001]);
    new Uint32Array(y.bytes.buffer)[33] = 0x3f100001;
    read.getMappedRange = () => read.bytes.buffer;
    const scopes = [];
    head.device = {
      createQuerySet() { const q = { destroyed: false, destroy() { this.destroyed = true; } }; queries.push(q); return q; },
      createBuffer({ size }) { const b = buffer(size, owned.length ? "query-read" : "query-resolve"); b.getMappedRange = () => b.bytes.buffer; owned.push(b); return b; },
      pushErrorScope: (kind) => scopes.push(kind),
      popErrorScope: async () => scopes.pop() === "validation" && scenario === "validation" ? new Error("primary validation") : null,
      createCommandEncoder: () => ({
        copyBufferToBuffer: (src, offset, dst, at, size) => dst.bytes.set(src.bytes.subarray(offset, offset + size), at),
        beginComputePass: () => ({
          setPipeline() {}, setBindGroup() {}, end() {},
          dispatchWorkgroups() { if (scenario === "encode") throw new Error("primary encode"); new Uint32Array(y.bytes.buffer).fill(0x3f800000, 1, 33); },
        }),
        resolveQuerySet(_qs, _start, count, dst) { const values = new BigInt64Array(dst.bytes.buffer); for (let i = 0; i < count; i++) values[i] = BigInt(1 + i * 1000); },
        finish() {},
      }),
      queue: { submit() {} },
    };
    const shared = { labels: ["trunk.pending"], pass() { throw new Error("head used trunk queries"); } };
    head.trunk.profiler = shared;
    head.scratch = { read, poison, guardBytes: 4, tokens: 1, tailCount: 1 };
    head.pipelines = [{}, {}];
    head.lastProfile = {};
    const work = ClefHead.prototype.wave.call(head, lease, { id: 11, phase: "memory" }, [{ index: 0, n: 32, count: 32, readOffset: 0, bytes: 136, y, group: {}, tailGroup: {} }]);
    if (scenario === "success") {
      const result = await work;
      assert.equal(mapped, true, "successful wave leaves head readback mapped for its caller");
      assert.equal(result.length, 34);
      read.unmap();
    } else {
      await assert.rejects(work, scenario === "stale" ? { code: "CLEF_HEAD_STALE" } : new RegExp(`primary ${scenario}`));
      assert.equal(mapped, false);
      assert.ok(unmapped > 0);
      assert.deepEqual([...f.coord.u32(4096, 96)], Array(96).fill(0));
    }
    assert.deepEqual(shared.labels, ["trunk.pending"]);
    assert.equal(owned.length, 2);
    assert.ok([...owned, ...queries].every((item) => item.destroyed), `${scenario}: owned profile resources leaked`);
    assert.deepEqual(scopes, []);
  }
});

test("the package ABI checker rejects an older binary lacking staged head exports", async () => {
  const required = await requiredExports();
  const staged = ["layout", "begin", "status", "memory_ready", "finish", "cancel"].map((name) => `kevala_clef_head_${name}`);
  assert.equal(staged.every((name) => required.includes(name)), true);
  assert.throws(() => assertModule(Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0), required, "stale"), /kevala_clef_head_status/);
});
