// The page-side protocol after a model has loaded: request routing, errors, profiling, disposal.
import assert from "node:assert/strict";
import { test } from "node:test";
import { Kevala } from "../js/src/index.js";
import { workers } from "./support/fake-worker.mjs";

const ready = { type: "ready", info: { backend: "wasm-simd", gpu: null } };
const QUESTIONS = { urgent: { type: "noul", instructions: "Is it urgent?" } };

async function loaded(t, onRequest) {
  const state = workers(t, [ready], { onRequest });
  const model = await Kevala.load({ backend: "wasm", cache: false });
  t.after(() => model.dispose());
  return { model, state };
}

test("options with a fixed set of values are rejected before a worker starts", async (t) => {
  const state = workers(t, []);
  await assert.rejects(Kevala.load({ backend: "gpu" }), /backend must be "auto", "webgpu", or "wasm"/);
  await assert.rejects(Kevala.load({ submit: "later" }), /submit must be "await", "split", or "none"/);
  await assert.rejects(Kevala.load({ precision: "int8" }), /precision must be "bf16", "f32", or "q8"/);
  await assert.rejects(Kevala.load({ stateCache: "no" }), /stateCache must be a boolean/);
  assert.equal(state.instances.length, 0);
});

test("valid backend and submit values reach the engine unchanged", async (t) => {
  const state = workers(t, [ready]);
  const model = await Kevala.load({ backend: "wasm", submit: "split", precision: "q8", cache: false });
  t.after(() => model.dispose());
  assert.equal(state.loads[0].backend, "wasm");
  assert.equal(state.loads[0].submit, "split");
  assert.equal(state.loads[0].precision, "q8");
});

test("image loading rejects invalid companion and token budget options before starting workers", async (t) => {
  const state = workers(t, []);
  await assert.rejects(Kevala.load({ vision: 42 }), /vision must be/);
  await assert.rejects(Kevala.load({ vision: "vision.kevala", backend: "wasm" }), /requires WebGPU/);
  await assert.rejects(Kevala.load({ imageTokens: 71 }), /imageTokens must be/);
  assert.equal(state.instances.length, 0);
});

test("image requests require loaded image capability and retain typed pixels in worker messages", async (t) => {
  const { model, state } = await loaded(t, (worker, message) => worker.reply({ type: "result", id: message.id, responses: [{ answers: {} }] }));
  const data = { width: 1, height: 1, data: new Uint8ClampedArray([10, 20, 30, 255]) };
  await assert.rejects(model.decide("camera", QUESTIONS, { parts: [{ type: "image", data }] }), /does not support images/);
  assert.equal(state.posted.length, 0);
  model.info.modalities = ["text", "image"];
  await model.decide("camera", QUESTIONS, { parts: [{ type: "image", data }] });
  assert.ok(state.posted[0].requests[0].parts[0].data.data instanceof Uint8ClampedArray);
  assert.deepEqual(Array.from(state.posted[0].requests[0].parts[0].data.data), [10, 20, 30, 255]);
  await model.decide("image URL", QUESTIONS, { parts: [{ type: "image", data: new URL("https://example.com/image.png") }] });
  assert.equal(state.posted[1].requests[0].parts[0].data, "https://example.com/image.png");
});

test("decide sends one normalized request and returns its response with the pass timing", async (t) => {
  const { model, state } = await loaded(t, (worker, m) => {
    worker.reply({ type: "result", id: m.id, responses: m.requests.map((r) => ({ answers: { echoed: r.state } })), timing: { forward: 4, batched: 1 } });
  });
  const parts = [{ type: "text", text: "attachment" }];
  const first = await model.decide("refund me", QUESTIONS, { parts });
  assert.deepEqual(state.posted[0], { type: "decide", id: state.posted[0].id, requests: [{ state: "refund me", parts, questions: QUESTIONS }] });
  assert.deepEqual(first, { answers: { echoed: "refund me" }, timing: { forward: 4, batched: 1 } });
  assert.deepEqual(model.lastTiming, { forward: 4, batched: 1 });

  // a missing state is the empty string, and a request without parts sends no `parts` key
  await model.decide(undefined, QUESTIONS);
  assert.deepEqual(state.posted[1].requests, [{ state: "", questions: QUESTIONS }]);
  assert.notEqual(state.posted[1].id, state.posted[0].id);
});

test("decideMany keeps item order and gives every response the shared timing", async (t) => {
  const { model, state } = await loaded(t, (worker, m) => {
    worker.reply({ type: "result", id: m.id, responses: m.requests.map((r) => ({ answers: r.state })), timing: { forward: 9, batched: 3 } });
  });
  const items = [{ state: "a", questions: QUESTIONS }, { state: { b: 1 }, questions: QUESTIONS }, { state: ["c"], questions: QUESTIONS }];
  const responses = await model.decideMany(items);
  assert.equal(state.posted.length, 1, "one message for the whole batch");
  assert.deepEqual(responses.map((r) => r.answers), ["a", { b: 1 }, ["c"]]);
  assert.ok(responses.every((r) => r.timing.batched === 3));
});

test("legacy images and videos reject instead of disappearing before typed-part validation", async (t) => {
  const { model, state } = await loaded(t, () => {});
  await assert.rejects(model.decide("text", QUESTIONS, { images: ["image.png"] }), /use typed content in "parts"/);
  await assert.rejects(model.decide("text", QUESTIONS, { videos: ["clip.mp4"] }), /use typed content in "parts"/);
  await assert.rejects(model.decideMany([{ state: "text", questions: QUESTIONS, images: ["image.png"] }]), /use typed content in "parts"/);
  await assert.rejects(model.decideMany([{ state: "text", questions: QUESTIONS, videos: ["clip.mp4"] }]), /use typed content in "parts"/);
  assert.equal(state.posted.filter((message) => message.type === "decide").length, 0);
});

test("replies are matched to requests by id, whatever order they arrive in", async (t) => {
  const held = [];
  const { model } = await loaded(t, (worker, m) => held.push([worker, m]));
  const slow = model.decide("slow", QUESTIONS);
  const fast = model.decide("fast", QUESTIONS);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(held.length, 2);
  for (const [worker, m] of held.reverse()) {
    worker.reply({ type: "result", id: m.id, responses: [{ answers: m.requests[0].state }], timing: {} });
  }
  assert.equal((await slow).answers, "slow");
  assert.equal((await fast).answers, "fast");
});

test("an engine error rejects only its request and keeps its code", async (t) => {
  const { model } = await loaded(t, (worker, m) => {
    if (m.requests[0].state === "bad") worker.reply({ type: "error", id: m.id, message: "questions must not be empty", code: "BAD_REQUEST" });
    else worker.reply({ type: "result", id: m.id, responses: [{ answers: "ok" }], timing: {} });
  });
  const [bad, good] = await Promise.allSettled([model.decide("bad", {}), model.decide("good", QUESTIONS)]);
  assert.equal(bad.status, "rejected");
  assert.equal(bad.reason.message, "questions must not be empty");
  assert.equal(bad.reason.code, "BAD_REQUEST");
  assert.equal(good.value.answers, "ok");
});

test("profile reports support without replacing the timing of the last request", async (t) => {
  const { model, state } = await loaded(t, (worker, m) => {
    if (m.type === "profile") worker.reply({ type: "result", id: m.id, supported: m.on, responses: [] });
    else worker.reply({ type: "result", id: m.id, responses: [{ answers: "ok" }], timing: { forward: 2 } });
  });
  await model.decide("x", QUESTIONS);
  assert.equal(await model.profile(true), true);
  assert.equal(await model.profile(false), false);
  assert.deepEqual(state.posted.slice(1).map((m) => [m.type, m.on]), [["profile", true], ["profile", false]]);
  assert.deepEqual(model.lastTiming, { forward: 2 });
});

test("dispose rejects requests in flight and every later call", async (t) => {
  const { model, state } = await loaded(t, () => {}); // the engine never answers
  const pending = model.decide("x", QUESTIONS);
  const reason = new Error("page is closing");
  model.dispose(reason);
  await assert.rejects(pending, (error) => error === reason);
  await assert.rejects(model.decide("y", QUESTIONS), /kevala was disposed/);
  await assert.rejects(model.profile(), /kevala was disposed/);
  assert.ok(state.instances.every((w) => w.terminated));
});
