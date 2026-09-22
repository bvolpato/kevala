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
      assert.equal(x1.length, batch.T1 * hidden);
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
