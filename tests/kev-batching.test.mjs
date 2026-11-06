import assert from "node:assert/strict";
import test from "node:test";
import kev from "../js/src/archs/kev.js";

function fixture(t, { hidden = 2048, limit = 2 ** 30, failAt = 0 } = {}) {
  let clock = 0, requests = [], scores = [];
  const leaves = [];
  const failure = new Error("GPU batch failed");
  t.mock.method(performance, "now", () => clock);
  const coord = {
    withInput: (bytes, fn) => { requests = JSON.parse(new TextDecoder().decode(bytes)).requests; return fn(0, bytes.length); },
    check: (value) => value,
    call: (fn) => fn(),
    out: () => {
      const n = requests.length;
      return new Uint8Array(Uint32Array.from([
        n, n * 2, n, n,
        ...requests.flatMap((_, i) => [i * 2, 2]),
        n, ...requests.flatMap((_, i) => [i, 1, i]),
        ...requests.map((_, i) => i),
      ]).buffer);
    },
    f32: (_p, length) => new Float32Array(length),
    u32: () => Uint32Array.from(requests.flatMap((r) => [r.id, r.id])),
    put: (bytes) => { scores = Array.from(new Float32Array(bytes.buffer)); return [1, bytes.length]; },
    outText: () => JSON.stringify(requests.map((r, i) => ({ id: r.id, score: scores[i] }))),
    x: {
      kevala_kev_prepare: () => { clock += 2; },
      kevala_kev_embed: () => { clock++; return 0; },
      kevala_kev_ids: () => 0,
      kevala_kev_finish: () => { clock++; },
      kevala_free: () => {},
    },
  };
  const gpu = {
    device: { limits: { maxStorageBufferBindingSize: limit, maxBufferSize: limit } },
    dims: { linearProj: 8192, attentionProj: 5120 },
    stats: { tokensSeen: 0 },
    async forward(x1, x2, batch, ids) {
      const mine = requests.map((r) => r.id);
      leaves.push(mine);
      assert.equal(typeof x1, "function");
      assert.equal(x1().length, batch.T1 * hidden);
      assert.equal(x2.length, batch.T2 * hidden);
      assert.deepEqual(Array.from(ids), mine.flatMap((id) => [id, id]));
      if (leaves.length === failAt) throw failure;
      clock += 5;
      this.stats.tokensSeen += batch.T1 + batch.T2;
      this.lastProfile = { matmul: mine.reduce((sum, id) => sum + id, 0), attention: mine.length };
      return Float32Array.from(mine, (id) => id * 2);
    },
  };
  return { engine: { coord, gpu, header: { config: { hidden_size: hidden, intermediate_size: 3584 } } }, leaves, failure };
}

for (const [reason, options, maxBatch] of [
  ["large-model request cap", { hidden: 2048 }, 4],
  ["GPU scratch buffer limit", { hidden: 1024, limit: 4 * 8192 * 4 }, 2],
]) {
  test(`recursive batches preserve responses and aggregate measurements under the ${reason}`, async (t) => {
    const { engine, leaves } = fixture(t, options);
    const requests = Array.from({ length: 9 }, (_, i) => ({ id: i + 1 }));
    const result = await kev.run(engine, requests);
    assert.deepEqual(leaves.flat(), requests.map((r) => r.id));
    assert.ok(leaves.every((leaf) => leaf.length <= maxBatch));
    assert.deepEqual(result.responses, requests.map((r) => ({ id: r.id, score: r.id * 2 })));
    assert.equal(result.timing.prepare, leaves.length * 4);
    assert.equal(result.timing.forward, leaves.length * 5);
    assert.equal(result.timing.tokens, requests.length * 3);
    assert.deepEqual(result.timing.cache, { tokensSeen: requests.length * 3 });
    assert.deepEqual(engine.gpu.lastProfile, { matmul: 45, attention: 9 });
  });
}

test("a failed recursive batch rejects the operation after an earlier batch succeeded", async (t) => {
  const { engine, leaves, failure } = fixture(t, { failAt: 2 });
  const requests = Array.from({ length: 9 }, (_, i) => ({ id: i + 1 }));
  await assert.rejects(kev.run(engine, requests), (e) => e === failure);
  assert.equal(leaves.length, 2);
  assert.equal(engine.gpu.stats.tokensSeen, leaves[0].length * 3);
  assert.ok(leaves.flat().length < requests.length);
});

test("deferred state embedding preserves copied branches and ids across native scratch reuse and memory growth", async (t) => {
  const { engine } = fixture(t, { hidden: 1 });
  const c = engine.coord, memory = new WebAssembly.Memory({ initial: 1 }), stages = [];
  const embed = c.x.kevala_kev_embed;
  c.x.kevala_kev_embed = (stage) => {
    stages.push(stage);
    embed(stage);
    if (stage === 1) {
      memory.grow(1);
      new Float32Array(memory.buffer, 0, 2).set([3, 4]);
      new Uint32Array(memory.buffer, 64, 2).fill(999);
    } else {
      new Float32Array(memory.buffer, 0, 1).set([7]);
      new Uint32Array(memory.buffer, 64, 2).set([5, 5]);
    }
    return 0;
  };
  c.x.kevala_kev_ids = () => 64;
  c.f32 = (ptr, length) => new Float32Array(memory.buffer, ptr, length);
  c.u32 = (ptr, length) => new Uint32Array(memory.buffer, ptr, length);
  const forward = engine.gpu.forward;
  engine.gpu.forward = async (...args) => {
    const result = await forward.apply(engine.gpu, args);
    assert.deepEqual(Array.from(args[1]), [7]);
    assert.deepEqual(Array.from(args[3]), [5, 5]);
    return result;
  };
  const result = await kev.run(engine, [{ id: 5 }]);
  assert.deepEqual(stages, [2, 1]);
  assert.equal(result.timing.prepare, 4);
  assert.equal(result.timing.forward, 5);
  assert.equal(result.timing.total, 10);
});

test("a hit excludes unused state embedding from preparation and keeps complete request timing", async (t) => {
  const { engine } = fixture(t, { hidden: 1 });
  const stages = [], embed = engine.coord.x.kevala_kev_embed;
  engine.coord.x.kevala_kev_embed = (stage) => { stages.push(stage); return embed(stage); };
  const forward = engine.gpu.forward;
  engine.gpu.forward = (x1, ...args) => forward.call(engine.gpu, () => new Float32Array(2), ...args);
  const result = await kev.run(engine, [{ id: 5 }]);
  assert.deepEqual(stages, [2]);
  assert.equal(result.timing.prepare, 3);
  assert.equal(result.timing.forward, 5);
  assert.equal(result.timing.total, 9);
});
