import test from "node:test";
import assert from "node:assert/strict";
import { createLatestFrameScheduler, supportsImage } from "../app/views/reaction-scheduler.js";

function fakeClock() {
  let time = 0;
  let id = 0;
  const tasks = new Map();
  return {
    now: () => time,
    setTimer(fn, delay) {
      const taskId = ++id;
      tasks.set(taskId, { at: time + delay, fn });
      return taskId;
    },
    clearTimer(taskId) {
      tasks.delete(taskId);
    },
    async advance(ms) {
      const end = time + ms;
      while (true) {
        const next = [...tasks.entries()].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        tasks.delete(next[0]);
        time = next[1].at;
        next[1].fn();
        await flush();
      }
      time = end;
      await flush();
    },
  };
}

async function flush() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
}

test("image support requires the loaded model to advertise the modality", () => {
  assert.equal(supportsImage(null), false);
  assert.equal(supportsImage({ modalities: ["text"] }), false);
  assert.equal(supportsImage({ modalities: ["text", "image"] }), true);
});

test("latest-frame scheduler never overlaps inference and captures the newest frame after work finishes", async () => {
  const clock = fakeClock();
  let newestFrame = 0;
  let active = 0;
  let maxActive = 0;
  const captures = [];
  const resolvers = [];
  const released = [];
  const results = [];
  const scheduler = createLatestFrameScheduler({
    capture() {
      const frame = { id: newestFrame };
      captures.push(frame.id);
      return frame;
    },
    infer(frame) {
      active += 1;
      maxActive = Math.max(maxActive, active);
      return new Promise((resolve) => resolvers.push(() => {
        active -= 1;
        resolve(frame.id);
      }));
    },
    onResult: (result, metrics) => results.push({ result, metrics }),
    release: (frame) => released.push(frame.id),
    intervalMs: 200,
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });

  scheduler.start();
  await clock.advance(0);
  assert.deepEqual(captures, [0]);
  newestFrame = 8;
  await clock.advance(1000);
  assert.deepEqual(captures, [0], "camera updates while inference runs do not build a stale queue");

  resolvers.shift()();
  await flush();
  await clock.advance(0);
  assert.deepEqual(captures, [0, 8], "the next inference reads the latest camera frame");

  await clock.advance(250);
  resolvers.shift()();
  await flush();
  assert.equal(maxActive, 1);
  assert.deepEqual(released, [0, 8]);
  assert.equal(results.length, 2);
  assert.equal(results[1].metrics.latencyMs, 250);
  assert.ok(results[1].metrics.fps > 0);
  scheduler.stop();
});

test("fast inference respects the minimum interval between frame starts", async () => {
  const clock = fakeClock();
  let captures = 0;
  let finish;
  const scheduler = createLatestFrameScheduler({
    capture: () => ++captures,
    infer: () => new Promise((resolve) => { finish = resolve; }),
    intervalMs: 200,
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });
  scheduler.start();
  await clock.advance(0);
  await clock.advance(30);
  finish({});
  await flush();
  await clock.advance(169);
  assert.equal(captures, 1);
  await clock.advance(1);
  assert.equal(captures, 2);
  scheduler.stop();
});

test("pausing suppresses an in-flight result, releases its frame, and prevents another capture", async () => {
  const clock = fakeClock();
  let captures = 0;
  let resolveInference;
  const released = [];
  const results = [];
  const scheduler = createLatestFrameScheduler({
    capture: () => ({ id: ++captures }),
    infer: (frame) => new Promise((resolve) => { resolveInference = () => resolve(frame.id); }),
    onResult: (result) => results.push(result),
    release: (frame) => released.push(frame.id),
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });

  scheduler.start();
  await clock.advance(0);
  scheduler.pause();
  resolveInference();
  await flush();
  await clock.advance(1000);
  assert.equal(captures, 1);
  assert.deepEqual(results, []);
  assert.deepEqual(released, [1]);
  scheduler.stop();
});
