import assert from "node:assert/strict";
import test from "node:test";
import { requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";
import { GpuGemma4Vision } from "../js/src/gpu-gemma4-vision.js";

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

test("Gemma rejects failed inference allocations even when the GPU returns zero readback rows", async () => {
  const model = Object.create(GpuGemma4.prototype);
  model.runtime = scopedGpu({ "out-of-memory": { message: "vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY" } });
  model.forwardRows = async () => { model.imageSpans = [{ rows: new Float32Array(1536) }]; return new Float32Array(1536); };
  await assert.rejects(model.forward(Uint32Array.from([3, 4])), (error) => {
    assert.equal(error.code, "WEBGPU_INIT");
    assert.match(error.message, /out of GPU memory while running Gemma 4 inference/);
    assert.match(error.message, /VK_ERROR_OUT_OF_DEVICE_MEMORY/);
    return true;
  });
  assert.deepEqual(model.runtime.stack, []);
  assert.deepEqual(model.imageSpans, []);
});

test("Gemma returns completed rows when inference reports no GPU failure", async () => {
  const model = Object.create(GpuGemma4.prototype);
  model.runtime = scopedGpu();
  const rows = Float32Array.from([0.1, -0.2]);
  model.forwardRows = async () => rows;
  assert.equal(await model.forward(Uint32Array.from([3, 4])), rows);
  assert.deepEqual(model.runtime.stack, []);
});

test("vision allocation failures identify a fatal GPU error and release frame state", async () => {
  const model = Object.create(GpuGemma4Vision.prototype);
  model.runtime = scopedGpu({ "out-of-memory": { message: "Not enough memory left." } });
  model.device = model.runtime.device;
  model.cfg = { patch: 16, pool: 3, positionSize: 32, headDim: 4, theta: 10000 };
  model.ensure = () => { throw new Error("invalid buffer"); };
  let reset = false;
  model.reset = () => { reset = true; };
  await assert.rejects(model.forward({
    patches: new Float32Array(9 * 768).fill(0.5),
    positions: Int32Array.from({ length: 18 }, (_, i) => i % 2 ? Math.floor(i / 6) : Math.floor(i / 2) % 3),
    softTokens: 1,
  }), (error) => {
    assert.equal(error.code, "WEBGPU_INIT");
    assert.match(error.message, /out of GPU memory while running Gemma 4 image encoding/);
    return true;
  });
  assert.equal(reset, true);
  assert.equal(model.running, false);
  assert.equal(model.lastProfile, null);
  assert.deepEqual(model.runtime.stack, []);
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
