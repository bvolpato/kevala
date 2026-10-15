import assert from "node:assert/strict";
import test from "node:test";
import { GpuGemma4 } from "../js/src/gpu-gemma4.js";
import { gpuLayouts } from "../js/src/pack-layout.js";

test("in-flight row uploads survive a full cache and overflow uploads are released separately", async (t) => {
  const previous = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { COPY_SRC: 1, COPY_DST: 2, STORAGE: 4 };
  t.after(() => previous === undefined ? delete globalThis.GPUBufferUsage : globalThis.GPUBufferUsage = previous);
  const allocated = [];
  const device = {
    limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20 },
    createBuffer({ size }) {
      const buffer = { bytes: new Uint8Array(size), destroyed: false, destroy() { this.destroyed = true; } };
      allocated.push(buffer);
      return buffer;
    },
    queue: { writeBuffer(buffer, offset, values) { buffer.bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), offset); } },
  };
  const layout = gpuLayouts({ config: {}, tokenizer: { offset: 4096, size: 0 }, tensors: [] }, 4096, () => true)[1];
  const model = new GpuGemma4({ device }, layout, { hostPle: true, layers: 1, hidden: 32, pleDim: 32, pleEmbeddingScale: 1 });
  model.weights.gatherStoredRows = async (_name, ids, scale, target, offset) => {
    for (let row = 0; row < ids.length; row++) target.fill(ids[row] * scale, offset + row * 32, offset + (row + 1) * 32);
  };
  const owned = [], pins = [];
  for (let id = 1; id <= 9; id++) await model.hostPleRows(Uint32Array.of(id), Uint32Array.of(id), owned, pins, 1);
  assert.equal(model.pleCache.size, 8);
  assert.equal(model.pleCacheBytes, 8 * 32 * 4);
  assert.equal(owned.length, 1);
  assert.ok(allocated.every((buffer) => !buffer.destroyed));
  assert.equal(new Float32Array(owned[0].bytes.buffer)[0], 9);
  assert.equal((await model.hostPleRows(Uint32Array.of(1), Uint32Array.of(1), owned, pins, 1)).cacheHit, true);
  for (const entry of pins) entry.active--;
  for (const buffer of owned) buffer.destroy();
  const nextPins = [];
  await model.hostPleRows(Uint32Array.of(10), Uint32Array.of(10), [], nextPins, 1);
  assert.equal(model.pleCache.size, 8);
  assert.ok(allocated.some((buffer) => buffer !== owned[0] && buffer.destroyed));
  for (const entry of nextPins) entry.active--;
  model.dispose();
  assert.ok(allocated.every((buffer) => buffer.destroyed));
});
