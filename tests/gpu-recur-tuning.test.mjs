import assert from "node:assert/strict";
import test from "node:test";

import { calibrateRecurrence, selectRecurrenceCandidate } from "../js/src/gpu-recur-tuning.js";

function point(tokens, pairs) {
  return { tokens, pairs };
}

test("recurrence tuning requires a stable five percent win at both token points", () => {
  const result = selectRecurrenceCandidate({
    candidates: [{ lanes: 8 }],
    points: [
      point(128, { 8: { baseline: [10, 10, 10], candidate: [9, 9.2, 9.3] } }),
      point(512, { 8: { baseline: [20, 20, 20], candidate: [18, 18.4, 18.6] } }),
    ],
  });
  assert.equal(result.choice, 8);
  assert.equal(result.zTiles, 4);
  assert.equal(result.candidates[0].reason, "stable-win-both-points");
  assert.ok(result.candidates[0].benefit > 0.07);

  const noisy = selectRecurrenceCandidate({
    candidates: [{ lanes: 8 }],
    points: [
      point(128, { 8: { baseline: [10, 10, 10], candidate: [9, 10.1, 9] } }),
      point(512, { 8: { baseline: [20, 20, 20], candidate: [18, 18.2, 18.4] } }),
    ],
  });
  assert.equal(noisy.choice, 4);
  assert.equal(noisy.zTiles, 2);
  assert.equal(noisy.candidates[0].reason, "point-regression");
});

test("a candidate that regresses at T512 is rejected despite a T128 win", () => {
  const result = selectRecurrenceCandidate({
    candidates: [{ lanes: 16 }],
    points: [
      point(128, { 16: { baseline: [10, 10, 10], candidate: [8, 8.2, 8.4] } }),
      point(512, { 16: { baseline: [20, 20, 20], candidate: [20.5, 20.2, 20.4] } }),
    ],
  });
  assert.equal(result.choice, 4);
  assert.equal(result.candidates[0].points[0].stable, true);
  assert.equal(result.candidates[0].points[1].stable, false);
  assert.equal(result.candidates[0].points[1].reason, "regression");
});

test("recurrence tuning picks the best geometric mean among stable candidates", () => {
  const result = selectRecurrenceCandidate({
    candidates: [{ lanes: 8 }, { lanes: 16 }],
    points: [
      point(128, {
        8: { baseline: [10, 10, 10], candidate: [8, 8.1, 8.2] },
        16: { baseline: [10, 10, 10], candidate: [8.5, 8.6, 8.7] },
      }),
      point(512, {
        8: { baseline: [20, 20, 20], candidate: [18, 18.1, 18.2] },
        16: { baseline: [20, 20, 20], candidate: [16, 16.2, 16.4] },
      }),
    ],
  });
  assert.equal(result.choice, 16);
  assert.equal(result.zTiles, 8);
  assert.ok(result.winner.benefit > result.candidates[0].benefit);
});

function fakeDevice(log) {
  const buffers = [];
  const device = {
    features: new Set(["timestamp-query"]),
    limits: { maxStorageBufferBindingSize: 64 * 1024 * 1024, maxBufferSize: 64 * 1024 * 1024, maxUniformBufferBindingSize: 65536 },
    pushErrorScope() {},
    async popErrorScope() { return null; },
    addEventListener(type, listener) { log.listeners.push([type, listener]); },
    removeEventListener(type, listener) { log.removed.push([type, listener]); },
    createBuffer(descriptor) {
      const buffer = {
        ...descriptor,
        destroyed: false,
        destroy() { this.destroyed = true; },
        async mapAsync() {},
        getMappedRange(_offset, size) {
          const out = new ArrayBuffer(size);
          new BigInt64Array(out).set(this.stamps?.subarray(0, size / 8) || []);
          return out;
        },
        unmap() {},
      };
      buffers.push(buffer);
      log.buffers.push(buffer);
      return buffer;
    },
    createBindGroup() { return {}; },
    createQuerySet(descriptor) {
      log.liveLabelsAtQuerySet.push(buffers.filter((buffer) => !buffer.destroyed).map((buffer) => buffer.label));
      const querySet = {
        ...descriptor,
        destroyed: false,
        stamps: new BigInt64Array(descriptor.count),
        destroy() { this.destroyed = true; },
      };
      log.querySets.push(querySet);
      return querySet;
    },
    createCommandEncoder() {
      const command = { passes: [], resolves: [], copies: [] };
      return {
        beginComputePass(options = {}) {
          const pass = { options, pipeline: null };
          command.passes.push(pass);
          log.passes.push(pass);
          return {
            setPipeline(pipeline) { pass.pipeline = pipeline; },
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        resolveQuerySet(...args) { command.resolves.push(args); log.resolves.push(args); },
        copyBufferToBuffer(...args) { command.copies.push(args); log.copies.push(args); },
        finish() { return command; },
      };
    },
    queue: {
      writeBuffer() {},
      submit(commands) {
        for (const command of commands) {
          let clock = 1000n;
          for (const pass of command.passes) {
            const writes = pass.options.timestampWrites;
            if (!writes) continue;
            const duration = pass.pipeline.lanes === 4 ? 1_000_000n : 800_000n;
            writes.querySet.stamps[writes.beginningOfPassWriteIndex] = clock;
            writes.querySet.stamps[writes.endOfPassWriteIndex] = clock + duration;
            clock += duration + 100n;
          }
          for (const [querySet, first, count, resolved] of command.resolves) resolved.stamps = querySet.stamps.slice(first, first + count);
          for (const [_source, _sourceOffset, destination] of command.copies) destination.stamps = _source.stamps;
        }
      },
      async onSubmittedWorkDone() {},
    },
  };
  log.buffers = buffers;
  return device;
}

test("an optional pipeline compilation failure preserves the compiled four-lane fallback", async () => {
  const scopes = [];
  const device = {
    features: new Set(["timestamp-query"]),
    pushErrorScope(kind) { scopes.push(kind); },
    async popErrorScope() { scopes.pop(); return null; },
  };
  const baseline = { getBindGroupLayout: () => ({}) };
  const result = await calibrateRecurrence(device, [
    { lanes: 4, pipeline: baseline },
    { lanes: 8, pipeline: async () => { throw new Error("optional compilation failed"); } },
  ], { lin_key_heads: 16, lin_heads: 16 });
  assert.equal(result.choice, 4);
  assert.equal(result.pipeline, baseline);
  assert.equal(result.diagnostics.reason, "benchmark-failed");
  assert.match(result.diagnostics.error, /optional compilation failed/);
  assert.deepEqual(scopes, []);
});

test("recurrence tuner measures both points with bounded query sets and sequential scratch", async (t) => {
  const previousUsage = globalThis.GPUBufferUsage;
  const previousMapMode = globalThis.GPUMapMode;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, QUERY_RESOLVE: 4, COPY_SRC: 8, MAP_READ: 16, UNIFORM: 32 };
  globalThis.GPUMapMode = { READ: 1 };
  t.after(() => {
    if (previousUsage === undefined) delete globalThis.GPUBufferUsage;
    else globalThis.GPUBufferUsage = previousUsage;
    if (previousMapMode === undefined) delete globalThis.GPUMapMode;
    else globalThis.GPUMapMode = previousMapMode;
  });

  const log = { buffers: [], listeners: [], removed: [], passes: [], querySets: [], resolves: [], copies: [], liveLabelsAtQuerySet: [] };
  const device = fakeDevice(log);
  const candidatePipeline = { lanes: 8, getBindGroupLayout: () => ({}) };
  const result = await calibrateRecurrence(device, [
    { lanes: 4, pipeline: { lanes: 4, getBindGroupLayout: () => ({}) } },
    { lanes: 8, pipeline: async (resolvedDevice) => { assert.equal(resolvedDevice, device); return candidatePipeline; } },
  ], { lin_key_heads: 16, lin_heads: 16 });

  assert.equal(result.choice, 8);
  assert.equal(result.zTiles, 4);
  assert.equal(result.pipeline, candidatePipeline);
  assert.doesNotThrow(() => JSON.stringify(result.diagnostics));
  assert.equal(log.querySets.length, 2, "one timestamp set per token point");
  assert.deepEqual(log.querySets.map((querySet) => querySet.count), [12, 12], "3 samples x 2 passes x 2 timestamps");
  assert.deepEqual(log.resolves.map((resolve) => resolve[2]), [12, 12]);
  assert.deepEqual(log.copies.map((copy) => copy[4]), [96, 96]);
  const timed = log.passes.filter((pass) => pass.options.timestampWrites);
  assert.equal(timed.length, 12);
  for (let pointIndex = 0; pointIndex < 2; pointIndex++) {
    const pointTimed = timed.slice(pointIndex * 6, (pointIndex + 1) * 6);
    assert.deepEqual(pointTimed.flatMap((pass) => [pass.options.timestampWrites.beginningOfPassWriteIndex, pass.options.timestampWrites.endOfPassWriteIndex]), [...Array(12).keys()]);
  }
  assert.equal(log.liveLabelsAtQuerySet[1].some((label) => label.includes("t128")), false, "T128 scratch is released before T512 allocation");
  assert.ok(log.querySets.every((querySet) => querySet.destroyed));
  assert.ok(log.buffers.every((buffer) => buffer.destroyed));
  assert.equal(log.listeners.length, log.removed.length);
});
