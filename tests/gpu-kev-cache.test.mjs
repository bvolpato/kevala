import assert from "node:assert/strict";
import test from "node:test";
import { GpuKev } from "../js/src/gpu-kev.js";

const NO_PARENT = 0xffffffff;
const tokens = (first, length = 32) => Array.from({ length }, (_, i) => first + i);

function cache(states = []) {
  const gpu = Object.create(GpuKev.prototype);
  gpu.lru = states.map((ids, slot) => ({ ids: Uint32Array.from(ids), slot, kvBase: slot * 1024, len: ids.length }));
  gpu.stats = { hits: 0, extensions: 0, misses: 0, tokensSaved: 0 };
  return gpu;
}

function plan(gpu, states) {
  let offset = 0;
  const batch = { states: states.map((ids) => {
    const row = [offset, ids.length];
    offset += ids.length;
    return row;
  }) };
  return gpu.plan(batch, Uint32Array.from(states.flat()));
}

function batch(states) {
  let offset = 0;
  const ranges = states.map((ids) => {
    const range = [offset, ids.length];
    offset += ids.length;
    return range;
  });
  return {
    T1: offset,
    T2: states.length,
    states: ranges,
    branches: states.map((_, index) => [index, 1, index]),
    rows: states.map((_, index) => index),
  };
}

function fakeForward(gpu) {
  gpu.cfg = { hidden: 1 };
  gpu.cap = { S: 4 };
  gpu.carry = [];
  gpu.ops = [[], []];
  gpu.ensureCarries = () => {};
  gpu.ensure = () => {};
  gpu.encode = () => {};
  gpu.device = {
    pushErrorScope() {},
    popErrorScope: async () => null,
    queue: { writeBuffer() {}, submit() {} },
    createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish: () => ({}) }),
  };
  gpu.readback = {
    mapAsync: async () => {},
    getMappedRange: () => new ArrayBuffer(16),
    unmap() {},
  };
}

function tracedForward(t, gpu) {
  const old = Object.getOwnPropertyDescriptor(globalThis, "GPUMapMode");
  Object.defineProperty(globalThis, "GPUMapMode", { configurable: true, value: { READ: 1 } });
  t.after(() => old ? Object.defineProperty(globalThis, "GPUMapMode", old) : delete globalThis.GPUMapMode);
  fakeForward(gpu);
  const trace = { scopes: [], writes: [], copies: [], submits: 0 };
  for (const name of ["g", "g2", "x", "x2", "tok", "tok2", "segs", "segs2", "ablocks", "ablocks2", "rows", "gathered", "readback"]) {
    if (name !== "readback") gpu[name] = name;
  }
  gpu.attnTile = true;
  gpu.blockCap = 8;
  gpu.device.pushErrorScope = (kind) => trace.scopes.push(kind);
  gpu.device.popErrorScope = async () => { assert.ok(trace.scopes.pop()); return null; };
  gpu.device.queue.writeBuffer = (buffer, offset, data) => trace.writes.push({ buffer, offset, data: data.slice() });
  gpu.device.queue.submit = () => trace.submits++;
  gpu.device.createCommandEncoder = () => ({ copyBufferToBuffer: (...args) => trace.copies.push(args), finish: () => ({}) });
  gpu.readback.getMappedRange = (_offset, bytes) => new ArrayBuffer(bytes);
  return trace;
}

test("an exact state hit skips the provider and state uploads while switching every shared branch buffer", async (t) => {
  const prefix = tokens(1), gpu = cache([prefix]);
  const trace = tracedForward(t, gpu), b = batch([prefix]);
  const result = await gpu.forward(() => { throw new Error("a cache hit embedded the state"); }, new Float32Array([7]), b, Uint32Array.from(prefix));
  assert.equal(result.length, 1);
  assert.deepEqual(gpu.stats, { hits: 1, extensions: 0, misses: 0, tokensSaved: prefix.length });
  assert.deepEqual(trace.writes.map(({ buffer }) => buffer), ["g2", "x2", "tok2", "segs2", "ablocks2", "rows"]);
  assert.deepEqual(Array.from(trace.writes.find(({ buffer }) => buffer === "tok2").data), [0, 0, prefix.length, 0]);
  assert.deepEqual(trace.copies.slice(0, 5).map(([src, , dst]) => [src, dst]), [["x2", "x"], ["tok2", "tok"], ["segs2", "segs"], ["ablocks2", "ablocks"], ["g2", "g"]]);
  assert.equal(trace.submits, 1);
  assert.deepEqual(trace.scopes, []);
});

for (const extension of [false, true]) {
  test(`a state ${extension ? "extension" : "miss"} calls its provider once and gathers the planned rows`, async (t) => {
    const prefix = tokens(1), state = [...prefix, 100, 101], gpu = cache(extension ? [prefix] : []);
    const trace = tracedForward(t, gpu), b = batch([state]);
    const values = Float32Array.from(state, (_, i) => i + 0.25);
    let calls = 0;
    await gpu.forward(() => {
      calls++;
      assert.deepEqual(trace.scopes, [], "provider failures must occur outside GPU error scopes");
      assert.equal(trace.writes.length, 0);
      return values;
    }, new Float32Array([7]), b, Uint32Array.from(state));
    assert.equal(calls, 1);
    const start = extension ? prefix.length : 0;
    assert.deepEqual(trace.writes.find(({ buffer }) => buffer === "x").data, values.slice(start));
    assert.deepEqual(Array.from(trace.writes.find(({ buffer }) => buffer === "g").data), [state.length - start, 1, 1, 1]);
    assert.equal(gpu.stats[extension ? "extensions" : "misses"], 1);
    assert.deepEqual(trace.scopes, []);
  });
}

test("a throwing state provider invalidates the planned carry without leaking error scopes or submitting work", async (t) => {
  const prefix = tokens(1), state = [...prefix, 100], gpu = cache([prefix]);
  const trace = tracedForward(t, gpu), b = batch([state]), failure = new Error("state embedding failed");
  await assert.rejects(gpu.forward(() => { throw failure; }, new Float32Array([7]), b, Uint32Array.from(state)), (error) => error === failure);
  assert.deepEqual(gpu.lru, []);
  assert.deepEqual(trace.scopes, []);
  assert.deepEqual(trace.writes, []);
  assert.equal(trace.submits, 0);
  await gpu.forward(() => new Float32Array(state.length), new Float32Array([7]), b, Uint32Array.from(state));
  assert.equal(trace.writes.find(({ buffer }) => buffer === "x").data.length, state.length, "retry must recompute the complete state");
  assert.equal(trace.submits, 1);
  assert.deepEqual(trace.scopes, []);
});

test("fresh prefixes run independently while exact duplicates share their destination", () => {
  const gpu = cache();
  const prefix = tokens(1);
  const longer = [...prefix, 100, 101];
  const result = plan(gpu, [prefix, longer, prefix]);
  assert.equal(result.segs.length, 2);
  assert.deepEqual(result.segs.map((s) => [s.parent, s.len, s.plen]), [[NO_PARENT, 32, 0], [NO_PARENT, 34, 0]]);
  assert.deepEqual(result.copies, []);
  assert.equal(result.carries[0], result.carries[2]);
  assert.notEqual(result.carries[0].slot, result.carries[1].slot);
  assert.deepEqual(gpu.stats, { hits: 0, extensions: 0, misses: 2, tokensSaved: 0 });
});

test("nested extensions both start from the completed prefix, never a fresh extension", () => {
  const prefix = tokens(1);
  const gpu = cache([prefix, tokens(100), tokens(200), tokens(300)]);
  const shorter = [...prefix, 500, 501];
  const longer = [...shorter, 502, 503];
  const result = plan(gpu, [shorter, longer, prefix]);
  assert.deepEqual(result.segs.map((s) => [s.parent, s.plen, s.len]), [[0, 32, 2], [0, 32, 4]]);
  assert.deepEqual(result.copies, [{ from: 0, to: 1024, rows: 32 }, { from: 0, to: 2048, rows: 32 }]);
  assert.equal(result.carries[2].slot, 0);
  assert.ok(result.segs.every((s) => s.dst !== 0));
  assert.deepEqual(gpu.stats, { hits: 1, extensions: 2, misses: 0, tokensSaved: 96 });
});

test("an evicted prefix cannot become a parent after its slot has been reassigned", () => {
  const a = tokens(1);
  const c = tokens(200);
  const gpu = cache([a, tokens(100), c, tokens(300)]);
  const result = plan(gpu, [tokens(500), [...a, 600], [...c, 700]]);
  assert.deepEqual(result.segs.map((s) => [s.parent, s.dst, s.plen]), [[NO_PARENT, 0, 0], [NO_PARENT, 1, 0], [2, 3, 32]]);
  assert.deepEqual(result.copies, [{ from: 2048, to: 3072, rows: 32 }]);
  assert.deepEqual(gpu.stats, { hits: 0, extensions: 1, misses: 2, tokensSaved: 32 });
});

test("disabled state cache misses warm prefixes but shares duplicates within a pass", async (t) => {
  const prefix = tokens(1);
  const gpu = cache([prefix]);
  gpu.stateCacheEnabled = false;
  const trace = tracedForward(t, gpu);
  const allocations = [];
  gpu.ensureCarries = (slots, rows) => allocations.push([slots, rows]);
  const plans = [];
  const originalPlan = gpu.plan;
  gpu.plan = (...args) => {
    const result = originalPlan.apply(gpu, args);
    plans.push(result);
    return result;
  };

  const warmBatch = batch([prefix]);
  warmBatch.T2 = 2;
  warmBatch.branches = [[0, 1, 0], [1, 1, 0]];
  warmBatch.rows = [0, 1];
  await gpu.forward(new Float32Array(warmBatch.T1), new Float32Array(warmBatch.T2), warmBatch, Uint32Array.from(prefix));
  assert.deepEqual(plans[0].segs.map((s) => [s.parent, s.len, s.plen]), [[NO_PARENT, 32, 0]]);
  assert.deepEqual(plans[0].copies, []);
  assert.deepEqual(allocations[0], [1, prefix.length]);
  assert.deepEqual(Array.from(trace.writes.find(({ buffer }) => buffer === "segs2").data.slice(0, 16)), [0, 1, 0, 0, prefix.length, 0, 0, 0, 1, 1, 0, 0, prefix.length, 0, 0, 0]);
  assert.deepEqual(gpu.lru, []);

  const duplicateBatch = batch([prefix, prefix]);
  await gpu.forward(new Float32Array(duplicateBatch.T1), new Float32Array(duplicateBatch.T2), duplicateBatch, Uint32Array.from([...prefix, ...prefix]));
  assert.deepEqual(plans[1].segs.map((s) => [s.parent, s.len, s.plen]), [[NO_PARENT, 32, 0]]);
  assert.deepEqual(plans[1].copies, []);
  assert.equal(plans[1].carries[0], plans[1].carries[1]);
  assert.equal(plans[1].carries[0].slot, 0);
  assert.equal(plans[1].carries[0].kvBase, 0);
  assert.deepEqual(allocations[1], [2, duplicateBatch.T1]);
  assert.deepEqual(gpu.lru, []);
  assert.deepEqual(gpu.stats, { hits: 0, extensions: 0, misses: 2, tokensSaved: 0 });

  gpu.stateCacheEnabled = true;
  await gpu.forward(new Float32Array(warmBatch.T1), new Float32Array(warmBatch.T2), warmBatch, Uint32Array.from(prefix));
  await gpu.forward(() => { throw new Error("a re-enabled cache hit embedded the state"); }, new Float32Array(warmBatch.T2), warmBatch, Uint32Array.from(prefix));
  assert.equal(plans[2].segs.length, 1);
  assert.equal(plans[3].segs.length, 0);
  assert.deepEqual(allocations.slice(2), [[5, 4096 + prefix.length], [5, 4096 + prefix.length]]);
  assert.deepEqual(gpu.stats, { hits: 1, extensions: 0, misses: 3, tokensSaved: prefix.length });
});

test("uncached mixed states use bounded scratch carries without retaining prefix extensions", () => {
  const prefix = tokens(1), longer = [...prefix, 100, 101], long = tokens(200, 1025);
  const gpu = cache([prefix]);
  gpu.stateCacheEnabled = false;
  const result = plan(gpu, [[], prefix, longer, prefix, long]);
  assert.equal(result.carries[0], null);
  assert.equal(result.carries[1], result.carries[3]);
  assert.deepEqual(result.segs.map((s) => [s.parent, s.plen, s.len]), [[NO_PARENT, 0, prefix.length], [NO_PARENT, 0, longer.length], [NO_PARENT, 0, long.length]]);
  assert.deepEqual(result.copies, []);
  assert.equal(result.slots, 3);
  assert.equal(result.rows, prefix.length + longer.length + long.length);
  assert.deepEqual(result.segs.map((s) => s.kvdst), [0, prefix.length, prefix.length + longer.length]);
  for (const segment of result.segs) {
    assert.ok(segment.dst < result.slots);
    assert.ok(segment.kvdst + segment.len <= result.rows);
  }
  assert.deepEqual(gpu.lru, []);
  assert.equal(plan(gpu, [prefix]).segs.length, 1, "the following request must recompute the prefix");
  assert.deepEqual(gpu.lru, []);
});

test("initial uncached carries allocate only the requested batch capacity", (t) => {
  const old = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, MAP_READ: 16 } });
  t.after(() => old ? Object.defineProperty(globalThis, "GPUBufferUsage", old) : delete globalThis.GPUBufferUsage);
  const header = new TextEncoder().encode('{"tensors":[]}');
  const prefix = new Uint8Array(16 + header.length);
  prefix.set(new TextEncoder().encode("KVLA"));
  new DataView(prefix.buffer).setUint32(4, 1, true);
  new DataView(prefix.buffer).setUint32(8, header.length, true);
  prefix.set(header, 16);
  const cfg = { hidden: 4096, layers: 32, full: Array.from({ length: 32 }, (_, i) => i % 4 === 3), intermediate: 12288, heads: 16, kv_heads: 4, lin_key_heads: 16, lin_heads: 32 };
  const outcomes = [];
  for (const enabled of [true, false]) {
    const allocations = [];
    const device = {
      limits: { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 },
      features: new Set(),
      createBuffer: (descriptor) => {
        const buffer = { ...descriptor, destroyed: false, destroy() { this.destroyed = true; } };
        allocations.push(buffer);
        return buffer;
      },
    };
    const gpu = new GpuKev({ device }, { prefix }, cfg);
    gpu.stateCacheEnabled = enabled;
    gpu.build = () => {};
    gpu.ensure(512, 1, 1, 1, 1);
    outcomes.push({ size: gpu.carrySize, bytes: gpu.carryBufs.reduce((sum, buffer) => sum + buffer.size, 0) });
    const allocationCount = allocations.length;
    gpu.ensure(512, 1, 1, 1, 1);
    assert.equal(allocations.length, allocationCount, "the same request capacity must not reallocate buffers");
    if (!enabled) {
      const carryBuffers = gpu.carryBufs.slice();
      gpu.stateCacheEnabled = true;
      gpu.ensureCarries(5, 4096 + 512);
      assert.deepEqual(gpu.carrySize, { slots: 8, rows: 8192 });
      assert.ok(carryBuffers.every((buffer) => buffer.destroyed), "growing for an enabled cache must release the smaller carry buffers");
      assert.equal(gpu.carryBufs.reduce((sum, buffer) => sum + buffer.size, 0), 922 * 1024 ** 2);
      const grownCount = allocations.length;
      gpu.stateCacheEnabled = false;
      gpu.ensureCarries(1, 512);
      assert.equal(allocations.length, grownCount, "disabling an existing cache must reuse its capacity without allocation churn");
    }
  }
  assert.deepEqual(outcomes, [
    { size: { slots: 8, rows: 8192 }, bytes: 922 * 1024 ** 2 },
    { size: { slots: 1, rows: 512 }, bytes: 90.25 * 1024 ** 2 },
  ]);
});

test("a rejected GPU pass cannot leave a reusable carry for the next request", async () => {
  const prefix = tokens(1);
  const state = [...prefix, 100];
  const gpu = cache([prefix]);
  const failure = new Error("command submission failed");
  gpu.cfg = { hidden: 1 };
  gpu.cap = { S: 1 };
  gpu.carry = [];
  gpu.ops = [[], []];
  gpu.ensureCarries = gpu.ensure = gpu.encode = () => {};
  gpu.device = {
    pushErrorScope: () => {},
    popErrorScope: async () => null,
    queue: { writeBuffer: () => {}, submit: () => { throw failure; } },
    createCommandEncoder: () => ({ copyBufferToBuffer: () => {}, finish: () => ({}) }),
  };
  const batch = { T1: state.length, T2: 1, states: [[0, state.length]], branches: [[0, 1, 0]], rows: [0] };
  await assert.rejects(gpu.forward(new Float32Array(state.length), new Float32Array(1), batch, Uint32Array.from(state)), (e) => e === failure);
  assert.deepEqual(gpu.lru, []);
  const retry = plan(gpu, [state]);
  assert.equal(retry.segs[0].parent, NO_PARENT);
  assert.equal(retry.segs[0].len, state.length);
  assert.deepEqual(retry.copies, []);
});

test("a full-sequence pass with no prefix reserves only a minimal carry", async (t) => {
  const old = Object.getOwnPropertyDescriptor(globalThis, "GPUMapMode");
  Object.defineProperty(globalThis, "GPUMapMode", { configurable: true, value: { READ: 1 } });
  t.after(() => old ? Object.defineProperty(globalThis, "GPUMapMode", old) : delete globalThis.GPUMapMode);
  const gpu = cache();
  fakeForward(gpu);
  gpu.readback.getMappedRange = (_offset, bytes) => new ArrayBuffer(bytes);
  gpu.stateCacheEnabled = false;
  const allocations = [];
  gpu.ensureCarries = (slots, rows) => allocations.push([slots, rows]);
  const b = batch([[]]);
  const result = await gpu.forward(new Float32Array(), new Float32Array(1), b, new Uint32Array());
  assert.equal(result.length, 1);
  assert.deepEqual(allocations, [[1, 16]]);
  assert.deepEqual(gpu.lru, []);
});

for (const errorKind of ["out-of-memory", "validation"]) {
  test(`a setup ${errorKind} error rebuilds buffers for a smaller retry`, async (t) => {
    for (const [key, value] of Object.entries({ GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, MAP_READ: 16 }, GPUMapMode: { READ: 1 } })) {
      const old = Object.getOwnPropertyDescriptor(globalThis, key);
      Object.defineProperty(globalThis, key, { configurable: true, value });
      t.after(() => old ? Object.defineProperty(globalThis, key, old) : delete globalThis[key]);
    }
    const allocated = [];
    const scopes = [];
    let rejectSetup = true;
    const device = {
      limits: { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 },
      pushErrorScope: (kind) => scopes.push(kind),
      popErrorScope: async () => {
        if (scopes.pop() === errorKind && rejectSetup) {
          rejectSetup = false;
          return { message: "simulated allocation failure" };
        }
        return null;
      },
      queue: { writeBuffer: (b) => assert.equal(b.destroyed, false), submit: () => {} },
      createCommandEncoder: () => ({ copyBufferToBuffer: () => {}, finish: () => ({}) }),
      createBuffer: () => {
        const buffer = { destroyed: false, destroy() { this.destroyed = true; }, mapAsync: async () => {}, getMappedRange: (_offset, size) => new ArrayBuffer(size), unmap: () => {} };
        allocated.push(buffer);
        return buffer;
      },
    };
    // a GPU sub-pack with no tensors: magic, format version 1, header length, header
    const json = new TextEncoder().encode('{"tensors":[]}');
    const prefix = new Uint8Array(16 + json.length);
    prefix.set(new TextEncoder().encode("KVLA"));
    new DataView(prefix.buffer).setUint32(4, 1, true);
    new DataView(prefix.buffer).setUint32(8, json.length, true);
    prefix.set(json, 16);
    const cfg = { hidden: 1024, layers: 2, full: [true, false], intermediate: 3584, heads: 8, kv_heads: 2, lin_key_heads: 16, lin_heads: 16 };
    const gpu = new GpuKev({ device }, { prefix }, cfg);
    const weight = device.createBuffer();
    gpu.weights.tensors.set("retained-weight", { buf: weight });
    gpu.build = () => { gpu.uni([1, 0, 0, 0]); gpu.ops = [[], []]; };
    gpu.encode = () => {};
    const run = (length) => gpu.forward(new Float32Array(length * cfg.hidden), new Float32Array(cfg.hidden), { T1: length, T2: 1, states: [[0, length]], branches: [[0, 1, 0]], rows: [0] }, Uint32Array.from(tokens(1, length)));
    await assert.rejects(run(64), /simulated allocation failure/);
    const failed = allocated.slice(1);
    assert.ok(failed.length > 0);
    assert.ok(failed.every((b) => b.destroyed));
    assert.equal(weight.destroyed, false);
    assert.deepEqual(scopes, []);
    const count = allocated.length;
    assert.equal((await run(32)).length, cfg.hidden);
    assert.ok(allocated.length > count, "the smaller retry must allocate fresh buffers");
    assert.ok(allocated.slice(count).every((b) => !b.destroyed));
    assert.equal(weight.destroyed, false);
    assert.deepEqual(scopes, []);
  });
}
