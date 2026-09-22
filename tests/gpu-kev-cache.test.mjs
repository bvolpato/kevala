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
    const json = new TextEncoder().encode('{"tensors":[]}');
    const prefix = new Uint8Array(16 + json.length);
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
