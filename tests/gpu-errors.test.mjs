import assert from "node:assert/strict";
import test from "node:test";
import { requestDevice, withGpuErrors } from "../js/src/gpu.js";

function scopedGpu(errors = {}, lost = null) {
  const stack = [];
  return {
    lost,
    stack,
    device: {
      pushErrorScope: (kind) => stack.push(kind),
      popErrorScope: async () => errors[stack.pop()] || null,
    },
  };
}

test("an allocation failure survives a later invalid-buffer exception", async () => {
  const gpu = scopedGpu({
    "out-of-memory": { message: "Not enough memory left." },
    validation: { message: "Buffer with '' label is invalid" },
  });
  await assert.rejects(withGpuErrors(gpu, () => { throw new Error("Buffer with '' label is invalid"); }, "loading weights"), (e) => {
    assert.equal(e.code, "WEBGPU_INIT");
    assert.match(e.message, /out of GPU memory while loading weights: Not enough memory left/);
    return true;
  });
  assert.deepEqual(gpu.stack, []);
});

test("a network or pack error without a GPU error keeps its identity", async () => {
  const gpu = scopedGpu();
  const failure = new Error("pack ended before its header");
  await assert.rejects(withGpuErrors(gpu, async () => { throw failure; }, "loading weights"), (e) => e === failure);
  assert.deepEqual(gpu.stack, []);
});

test("device loss replaces a downstream readback failure", async () => {
  const gpu = scopedGpu({}, { reason: "unknown", message: "Device was lost" });
  await assert.rejects(withGpuErrors(gpu, () => { throw new Error("Buffer is invalid"); }, "warming up"), (e) => {
    assert.equal(e.code, "WEBGPU_INIT");
    assert.match(e.message, /device lost while warming up: Device was lost/);
    return true;
  });
  assert.deepEqual(gpu.stack, []);
});

test("successful GPU work returns its result and closes every scope", async () => {
  const gpu = scopedGpu();
  assert.equal(await withGpuErrors(gpu, async () => 42, "warming up"), 42);
  assert.deepEqual(gpu.stack, []);
});

test("requestDevice honors the retry preference and rejects software adapters", async (t) => {
  const requested = [];
  const device = { features: new Set(), limits: {}, lost: new Promise(() => {}) };
  const adapter = {
    info: { vendor: "AMD", isFallbackAdapter: false },
    features: new Set(),
    limits: {},
    requestDevice: async () => device,
  };
  t.mock.getter(globalThis, "navigator", () => ({ gpu: {
    requestAdapter: async (options) => { requested.push(options); return adapter; },
  } }));
  const gpu = await requestDevice({ baseline: true, powerPreference: "low-power" });
  assert.equal(gpu.device, device);
  assert.equal(gpu.powerPreference, "low-power");
  assert.equal(gpu.name, "AMD");
  assert.deepEqual(requested, [{ powerPreference: "low-power" }]);
  adapter.info.isFallbackAdapter = true;
  await assert.rejects(requestDevice(), /software WebGPU adapter/);
});
