import assert from "node:assert/strict";
import test from "node:test";
import { loadCode, requestCode } from "../app/code.js";
import { MODELS } from "../js/src/source.js";

async function execute(code) {
  const calls = [];
  const engine = {
    decide: async (state, questions) => { calls.push({ state, questions }); return { answers: {} }; },
    decideMany: async (items) => { calls.push(items); return items.map(() => ({ answers: {} })); },
  };
  let options;
  const Kevala = { load: async (value) => { options = value; return engine; } };
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  await new AsyncFunction("Kevala", "console", code.replace(/^import[^\n]+\n/, ""))(Kevala, { log() {} });
  return { options, calls };
}

test("example setup loads every selected catalog model and explicit backend", async () => {
  for (const model of Object.keys(MODELS)) {
    assert.deepEqual((await execute(loadCode({ model, backend: "webgpu" }))).options, { model, backend: "webgpu" });
  }
  assert.deepEqual((await execute(loadCode({ model: "kev-4b", backend: "auto" }))).options, { model: "kev-4b" });
});

test("copied single requests preserve custom URLs and JSON values", async () => {
  const customUrl = 'https://example.com/model.kevala?label="custom"&revision=1';
  const state = JSON.parse('{"__proto__":{"value":"own property"},"message":"Quotes \\" and a newline\\n</script>"}');
  const questions = { "needs-review": { type: "noul", instructions: "Does the text request a review?" } };
  const result = await execute(requestCode({ questions, items: [{ state, questions }] }, {
    model: "custom", customUrl, backend: "wasm", from: "checkpoint",
  }));
  assert.deepEqual(result.options, { model: customUrl, backend: "wasm", from: "checkpoint" });
  assert.deepEqual(result.calls, [{ state, questions }]);
  assert.equal(Object.hasOwn(result.calls[0].state, "__proto__"), true);
});

test("copied batches retain state types, order, and shared questions", async () => {
  const states = ["A plain string", { value: 4 }, ["array", false, null]];
  const questions = { urgent: { type: "noul", instructions: "Is it urgent?" } };
  const items = states.map((state) => ({ state, questions }));
  const result = await execute(requestCode({ items, questions }, { model: "gemma-4-e4b" }));
  assert.deepEqual(result.options, { model: "gemma-4-e4b" });
  assert.deepEqual(result.calls, [items]);
});
