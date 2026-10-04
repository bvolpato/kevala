import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_BATCH_REQUESTS, requestQueue } from "../js/src/request-queue.js";

/** A queue whose passes are released by hand, so a test controls what arrives while one runs. */
function harness({ fails = () => null, limit } = {}) {
  const passes = [];
  const posted = [];
  const gates = [];
  const run = (requests) => new Promise((resolve, reject) => {
    passes.push(requests);
    gates.push(() => {
      const error = fails(requests);
      if (error) reject(error);
      else resolve({ responses: requests.map((r) => `answer:${r}`), timing: { forward: 1 } });
    });
  });
  const queue = requestQueue(run, (m) => posted.push(m), limit);
  const release = async () => {
    gates.shift()();
    for (let i = 0; i < 6; i++) await Promise.resolve();
  };
  /** Releases passes until the queue is idle. */
  const settle = async () => {
    for (let guard = 0; gates.length && guard < 100; guard++) await release();
    assert.equal(gates.length, 0, "the queue never went idle");
  };
  return { queue, passes, posted, release, settle };
}

test("messages that arrive during a pass share the next pass and keep their own responses", async () => {
  const h = harness();
  h.queue.push({ id: 1, requests: ["a"] });
  h.queue.push({ id: 2, requests: ["b", "c"] });
  h.queue.push({ id: 3, requests: ["d"] });
  assert.deepEqual(h.passes, [["a"]], "the first message starts at once");
  await h.settle();
  assert.deepEqual(h.passes, [["a"], ["b", "c", "d"]]);
  assert.deepEqual(h.posted, [
    { type: "result", id: 1, responses: ["answer:a"], timing: { forward: 1, batched: 1 } },
    { type: "result", id: 2, responses: ["answer:b", "answer:c"], timing: { forward: 1, batched: 3 } },
    { type: "result", id: 3, responses: ["answer:d"], timing: { forward: 1, batched: 3 } },
  ]);
});

test("a pass holds at most the request limit, and an oversized message still runs alone", async () => {
  assert.equal(MAX_BATCH_REQUESTS, 64);
  const h = harness({ limit: 4 });
  h.queue.push({ id: 0, requests: ["warm"] });
  h.queue.push({ id: 1, requests: ["a", "b", "c"] });
  h.queue.push({ id: 2, requests: ["d"] }); // fills the pass exactly
  h.queue.push({ id: 3, requests: ["e", "f"] });
  h.queue.push({ id: 4, requests: ["g", "h", "i"] }); // would make five
  h.queue.push({ id: 5, requests: ["j", "k", "l", "m", "n", "o"] }); // larger than the limit by itself
  h.queue.push({ id: 6, requests: ["p"] });
  await h.settle();
  assert.deepEqual(h.passes.map((p) => p.length), [1, 4, 2, 3, 6, 1]);
  assert.deepEqual(h.posted.map((m) => m.id), [0, 1, 2, 3, 4, 5, 6]);
});

test("one bad request fails only its own message, and the others are answered in arrival order", async () => {
  const bad = Object.assign(new Error("question \"q\": unknown type"), { code: "BAD_REQUEST" });
  const h = harness({ fails: (requests) => (requests.includes("bad") ? bad : null) });
  h.queue.push({ id: 0, requests: ["warm"] });
  h.queue.push({ id: 1, requests: ["a"] });
  h.queue.push({ id: 2, requests: ["bad"] });
  h.queue.push({ id: 3, requests: ["c", "d"] });
  await h.settle();
  // the shared pass fails, then every message runs alone, first to last
  assert.deepEqual(h.passes, [["warm"], ["a", "bad", "c", "d"], ["a"], ["bad"], ["c", "d"]]);
  assert.deepEqual(h.posted.map((m) => [m.type, m.id]), [["result", 0], ["result", 1], ["error", 2], ["result", 3]]);
  assert.deepEqual(h.posted[2], { type: "error", id: 2, message: "question \"q\": unknown type", code: "BAD_REQUEST" });
  assert.deepEqual(h.posted[3].responses, ["answer:c", "answer:d"]);
  assert.equal(h.posted[3].timing.batched, 2, "a retried message reports its own pass");

  // the queue keeps working after a failure
  h.queue.push({ id: 4, requests: ["e"] });
  await h.settle();
  assert.deepEqual(h.posted.at(-1), { type: "result", id: 4, responses: ["answer:e"], timing: { forward: 1, batched: 1 } });
});

test("a failing pass of one message reports the error with its code and does not retry", async () => {
  const h = harness({ fails: () => Object.assign(new Error("device lost"), { code: "WEBGPU_INIT" }) });
  h.queue.push({ id: 7, requests: ["a"] });
  await h.settle();
  assert.equal(h.passes.length, 1);
  assert.deepEqual(h.posted, [{ type: "error", id: 7, message: "device lost", code: "WEBGPU_INIT" }]);
});

test("clear drops the messages that have not started", async () => {
  const h = harness();
  h.queue.push({ id: 1, requests: ["a"] });
  h.queue.push({ id: 2, requests: ["b"] });
  h.queue.clear();
  await h.settle();
  assert.deepEqual(h.posted.map((m) => m.id), [1]);
  h.queue.push({ id: 3, requests: ["c"] });
  await h.settle();
  assert.deepEqual(h.posted.map((m) => m.id), [1, 3]);
});
