import assert from "node:assert/strict";
import test from "node:test";
import { cpuOptions, threadCandidates, selectThreadCount, validProfile, tuningKey, readTuning, writeTuning } from "../js/src/cpu-policy.js";
import { RemoteShard } from "../js/src/shard-client.js";

const options = cpuOptions();
test("CPU overrides are explicit and reject malformed counts", () => {
  assert.deepEqual(options, { cpuKernel: "auto", threads: "auto", retune: false });
  assert.equal(cpuOptions({ cpuKernel: "4x4", threads: 7 }).threads, 7);
  for (const threads of [0, 17, 1.5, NaN, Infinity, "4", null]) assert.throws(() => cpuOptions({ threads }), /threads/);
  assert.throws(() => cpuOptions({ cpuKernel: "simd" }), /cpuKernel/);
  assert.throws(() => cpuOptions({ retune: "true" }), /retune/);
});

test("thread candidates respect hardware and model limits including uneven partitions", () => {
  assert.deepEqual(threadCandidates(32, 16), [1, 2, 4, 8, 16]);
  assert.deepEqual(threadCandidates(6, 16), [1, 2, 4, 6]);
  assert.deepEqual(threadCandidates(32, 1), [1]);
  assert.deepEqual(threadCandidates(4, 3), [1, 2, 3]);
});

const row = (threads, a, b = a) => ({ threads, samples: [a, b] });
test("thread selection uses medians across shapes and avoids insignificant extra workers", () => {
  assert.equal(selectThreadCount([row(1, [40, 40, 40]), row(4, [11, 11, 90]), row(8, [10, 10, 10])]), 4);
  assert.equal(selectThreadCount([row(1, [40, 40, 40]), row(4, [12, 12, 90]), row(8, [10, 10, 10])]), 8);
  assert.equal(selectThreadCount([row(1, [10, 10, 10], [100, 100, 100]), row(2, [20, 20, 20], [20, 20, 20])]), 2);
  assert.throws(() => selectThreadCount([row(1, [0, 0, 0])]), /usable timings/);
});

test("cached profiles expire and must match available choices", () => {
  const now = 1000000000;
  const p = { revision: 1, created: now, threads: 4, kernel: "2x4", complete: true };
  assert.equal(validProfile(p, [1, 2, 4], options, now), true);
  for (const patch of [{ revision: 0 }, { complete: false }, { threads: 8 }, { kernel: "unknown" }, { created: now + 1 }, { created: now - 7 * 86400000 }]) {
    assert.equal(validProfile({ ...p, ...patch }, [1, 2, 4], options, now), false);
  }
  assert.equal(validProfile(p, [4], cpuOptions({ cpuKernel: "4x4" }), now), false);
});

test("cache keys isolate browser, model, WASM flavor, and overrides", async () => {
  const input = { flavor: "relaxed", base: "https://example.test/js/", config: { hidden_size: 1024 }, options, hardware: 16, userAgent: "browser-v1" };
  const key = await tuningKey(input);
  for (const patch of [{ flavor: "simd" }, { base: "https://example.test/custom/" }, { config: { hidden_size: 512 } }, { hardware: 8 }, { userAgent: "browser-v2" }, { options: cpuOptions({ threads: 4 }) }, { options: cpuOptions({ cpuKernel: "4x4" }) }]) {
    assert.notEqual(await tuningKey({ ...input, ...patch }), key);
  }
  assert.equal(await tuningKey({ ...input, options: cpuOptions({ retune: true }) }), key);
});

test("tuning survives unavailable persistent storage", async () => {
  const old = globalThis.caches;
  globalThis.caches = { open: async () => { throw new Error("storage disabled"); } };
  try {
    assert.equal(await readTuning("x", [1], options), null);
    await writeTuning("x", {});
  } finally {
    if (old === undefined) delete globalThis.caches;
    else globalThis.caches = old;
  }
});

test("shard requests release pending entries on success, timeout, abort, and close", async () => {
  const port = { postMessage() {}, close() {} };
  const r = new RemoteShard(port);
  const success = r.call({ type: "probe" });
  port.onmessage({ data: { seq: 1, type: "ok" } });
  assert.equal((await success).type, "ok");
  await assert.rejects(r.call({ type: "probe" }, [], { timeout: 1 }), /timed out/);
  const controller = new AbortController();
  const cancelled = r.call({ type: "probe" }, [], { signal: controller.signal });
  controller.abort(new Error("cancel test"));
  await assert.rejects(cancelled, /cancel test/);
  const closed = r.call({ type: "probe" });
  r.close();
  await assert.rejects(closed, /closed/);
  assert.equal(r.waits.size, 0);
  await assert.rejects(r.call({ type: "probe" }, [], { signal: controller.signal }), /cancel test/);
  assert.equal(r.waits.size, 0);
});

test("profile persistence round-trips and rejects a count from another configuration", async () => {
  const old = globalThis.caches;
  const entries = new Map();
  globalThis.caches = { open: async () => ({
    put: async (key, response) => entries.set(key, response),
    match: async (key) => entries.get(key)?.clone(),
  }) };
  try {
    const profile = { kernel: "2x4", threads: 4, complete: true };
    await writeTuning("test", profile);
    assert.equal((await readTuning("test", [1, 2, 4], options)).threads, 4);
    assert.equal(await readTuning("test", [1, 2], options), null);
  } finally {
    if (old === undefined) delete globalThis.caches;
    else globalThis.caches = old;
  }
});
