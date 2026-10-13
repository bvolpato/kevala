import assert from "node:assert/strict";
import test from "node:test";
import { Profiler } from "../js/src/gpu.js";

function profilerDevice() {
  const destroyed = [];
  const timestamps = BigInt64Array.from([0n, 1_000_000n, 2_000_000n, 5_000_000n]).buffer;
  const device = {
    createQuerySet: () => ({ destroy: () => destroyed.push("queries") }),
    createBuffer: () => ({
      destroy: () => destroyed.push("buffer"),
      mapAsync: async () => {},
      getMappedRange: () => timestamps,
      unmap: () => {},
    }),
  };
  return { device, destroyed };
}

test("profiler aggregates repeated kernel labels and defers destruction until collection", async (t) => {
  const original = globalThis.GPUBufferUsage;
  const originalMapMode = globalThis.GPUMapMode;
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { QUERY_RESOLVE: 1, COPY_SRC: 2, MAP_READ: 4, COPY_DST: 8 },
  });
  Object.defineProperty(globalThis, "GPUMapMode", { configurable: true, value: { READ: 1 } });
  t.after(() => {
    if (original === undefined) delete globalThis.GPUBufferUsage;
    else Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: original });
    if (originalMapMode === undefined) delete globalThis.GPUMapMode;
    else Object.defineProperty(globalThis, "GPUMapMode", { configurable: true, value: originalMapMode });
  });

  const { device, destroyed } = profilerDevice();
  const profiler = new Profiler(device, 4);
  const encoder = {
    beginComputePass: () => ({ end() {} }),
    resolveQuerySet() {},
    copyBufferToBuffer() {},
  };
  profiler.pass(encoder, "vision.mm.q").end();
  profiler.pass(encoder, "vision.mm.q").end();
  profiler.destroy();
  assert.deepEqual(destroyed, [], "resources stay alive while timestamps are pending");

  profiler.finish(encoder);
  assert.deepEqual(await profiler.collect(), { "vision.mm.q": 4 });
  assert.equal(destroyed.length, 3, "query set and buffers are released after collection");
  profiler.destroy();
  assert.equal(destroyed.length, 3, "destroy is idempotent");
});
