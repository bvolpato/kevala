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

function fakeMessageChannels() {
  const channels = [];
  class Port {
    queue = [];
    onmessage = null;
    peer = null;
    closed = false;
    start() {}
    postMessage(data) {
      if (this.closed || this.peer.closed) throw new Error("port is closed");
      this.peer.queue.push(data);
    }
    deliverNext() {
      const data = this.queue.shift();
      if (data !== undefined) this.onmessage?.({ data });
    }
    close() { this.closed = true; }
  }
  class Channel {
    constructor() {
      this.port1 = new Port();
      this.port2 = new Port();
      this.port1.peer = this.port2;
      this.port2.peer = this.port1;
      channels.push(this);
    }
  }
  return { channels, create: () => new Channel() };
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

test("native zero-delay work uses cancellable channel tasks across pause and resume", async () => {
  const messages = fakeMessageChannels();
  const channel = messages.channels;
  let captures = 0;
  let finish;
  const released = [];
  const scheduler = createLatestFrameScheduler({
    capture: () => ({ id: ++captures }),
    infer: (frame) => new Promise((resolve) => { finish = () => resolve(frame.id); }),
    release: (frame) => released.push(frame.id),
    intervalMs: 0,
    createMessageChannel: messages.create,
  });

  scheduler.start();
  assert.equal(captures, 0, "the first frame waits for the channel macrotask");
  scheduler.pause();
  scheduler.resume();
  assert.equal(channel.length, 1, "one channel is reused after resume");
  channel[0].port1.deliverNext();
  assert.equal(captures, 0, "the stale pre-pause delivery cannot start work");
  assert.deepEqual(channel[0].port1.queue.length, 1, "the stale delivery leaves the resumed task pending");
  channel[0].port1.deliverNext();
  assert.equal(captures, 1);
  assert.equal(scheduler.running, true);

  scheduler.pause();
  finish();
  await flush();
  assert.deepEqual(released, [1]);
  scheduler.resume();
  scheduler.stop();
  assert.equal(channel[0].port1.closed, true);
  assert.equal(channel[0].port2.closed, true);
});

test("an empty capture at zero interval retries on a bounded timer", async () => {
  const clock = fakeClock();
  let captures = 0;
  const scheduler = createLatestFrameScheduler({
    capture: () => { captures += 1; return null; },
    infer: () => assert.fail("an empty capture must not reach inference"),
    intervalMs: 0,
    now: clock.now,
    setTimer: clock.setTimer,
    clearTimer: clock.clearTimer,
  });

  scheduler.start();
  await clock.advance(0);
  assert.equal(captures, 1);
  await clock.advance(15);
  assert.equal(captures, 1, "the empty-frame retry does not immediately spin");
  await clock.advance(1);
  assert.equal(captures, 2, "the retry becomes eligible after the 16ms backoff");
  scheduler.stop();
});

test("missing MessageChannel falls back to the native timer", async () => {
  let captures = 0;
  const scheduler = createLatestFrameScheduler({
    capture: () => ++captures,
    infer: () => "ok",
    intervalMs: 1000,
    createMessageChannel: () => null,
  });
  scheduler.start();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(captures, 1);
  scheduler.stop();
});
