import test from "node:test";
import assert from "node:assert/strict";
import { clearGemma4VisionImageData, consumeGemma4VisionImage, prepareGemma4GpuVisionImage } from "../js/src/gemma4-image.js";
import { createGemma4ImageGpu } from "../js/src/gpu-gemma4-image.js";

globalThis.GPUBufferUsage ||= { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 };

function fixture() {
  const buffers = [], groups = [], writes = [], scopes = [];
  const device = {
    limits: { maxBufferSize: 32 * 1024 * 1024, maxStorageBufferBindingSize: 32 * 1024 * 1024 },
    pushErrorScope(kind) { scopes.push(kind); }, async popErrorScope() { scopes.pop(); return null; },
    createShaderModule({ code }) { return { code, async getCompilationInfo() { return { messages: [] }; } }; },
    async createComputePipelineAsync({ label }) { return { label, getBindGroupLayout() { return { label }; } }; },
    createBuffer(info) { const buffer = { ...info, destroyed: false, destroy() { this.destroyed = true; } }; buffers.push(buffer); return buffer; },
    createBindGroup(info) { const group = { ...info }; groups.push(group); return group; },
    queue: { writeBuffer(buffer, _offset, data) { writes.push({ buffer, bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice() }); } },
  };
  return { runtime: { device }, device, buffers, groups, writes, scopes };
}

function rawFrame() {
  const input = { width: 32, height: 24, data: Uint8Array.from({ length: 32 * 24 * 4 }, (_, i) => (i * 37) & 255) };
  return consumeGemma4VisionImage(prepareGemma4GpuVisionImage(input, 16), { patch: 16, pool: 3 });
}

test("GPU image preparation caches bounded geometry, binds sparse layouts, and clears full sensitive buffers", async () => {
  const f = fixture(), helper = await createGemma4ImageGpu(f.runtime, (name) => name);
  const owned = rawFrame(), output = { size: owned.raw.width * owned.raw.height * 12 };
  try {
    const first = await helper.prepare(owned.raw, output);
    assert.ok(first);
    assert.ok(helper.diagnostics.scratchBytes <= 8 * 1024 * 1024);
    assert.equal(f.buffers.length, 8);
    assert.deepEqual(first.horizontalGroup.entries.map((entry) => entry.binding), [0, 1, 2, 3, 4]);
    assert.deepEqual(first.verticalGroup.entries.map((entry) => entry.binding), [0, 5, 6, 7, 8, 9]);
    const upload = f.writes.findLast((write) => write.buffer === first.geometry.rgba);
    assert.deepEqual(upload.bytes, owned.raw.rgba);
    owned.raw.rgba[0] ^= 255;
    const second = await helper.prepare(owned.raw, output);
    assert.strictEqual(second.geometry, first.geometry);
    assert.strictEqual(second.verticalGroup, first.verticalGroup);
    assert.deepEqual(f.writes.findLast((write) => write.buffer === first.geometry.rgba).bytes, owned.raw.rgba);
    const rebound = await helper.prepare(owned.raw, { size: output.size });
    assert.strictEqual(rebound.geometry, first.geometry);
    assert.notStrictEqual(rebound.verticalGroup, first.verticalGroup);
    assert.equal(f.buffers.length, 8);
    const events = [];
    helper.encode({ beginComputePass() { return { setPipeline(pipeline) { events.push(pipeline.label); }, setBindGroup(_index, group) { events.push(group); }, dispatchWorkgroups(count) { events.push(count); }, end() { events.push("end"); } }; } }, rebound);
    assert.equal(events.filter((entry) => entry === "end").length, 2);
    assert.equal(events[0], "gemma4_image_resize_horizontal");
    assert.equal(events[4], "gemma4_image_resize_vertical");
    const clears = [];
    helper.clear({ clearBuffer(...args) { clears.push(args); } });
    assert.deepEqual(clears, [[first.geometry.rgba], [first.geometry.horizontal]]);
    helper.reset();
    assert.ok(f.buffers.every((buffer) => buffer.destroyed));
    assert.equal(helper.geometry, null);
    assert.deepEqual(f.scopes, []);
  } finally { helper.reset(); clearGemma4VisionImageData(owned); }
});

test("cached raw uploads propagate queue failures to the enclosing vision error scope", async () => {
  const f = fixture(), helper = await createGemma4ImageGpu(f.runtime, (name) => name), owned = rawFrame();
  const output = { size: owned.raw.width * owned.raw.height * 12 };
  try {
    await helper.prepare(owned.raw, output);
    f.device.queue.writeBuffer = () => { throw new Error("raw upload failed"); };
    await assert.rejects(helper.prepare(owned.raw, output), /raw upload failed/);
    assert.ok(owned.raw.rgba.some((value) => value !== 0));
    assert.deepEqual(f.scopes, []);
  } finally { helper.reset(); clearGemma4VisionImageData(owned); }
});

test("unsupported GPU geometry keeps raw pixels available for CPU fallback", async () => {
  const f = fixture(), helper = await createGemma4ImageGpu(f.runtime, (name) => name), owned = rawFrame();
  try {
    assert.equal(await helper.prepare(owned.raw, { size: 4 }), null);
    assert.equal(await helper.prepare({ ...owned.raw, rgba: { byteLength: 4 * 1024 * 1024 + 4 } }, { size: 32 * 1024 * 1024 }), null);
    assert.ok(owned.raw.rgba.some((value) => value !== 0));
    assert.equal(f.buffers.length, 0);
    assert.equal(helper.diagnostics.lastFallback, "geometry-unsupported");
  } finally { helper.reset(); clearGemma4VisionImageData(owned); }
});

test("optional compilation and setup failures close scopes and release scratch", async () => {
  const unavailable = fixture();
  unavailable.device.createComputePipelineAsync = async () => { throw new Error("optional pipeline unavailable"); };
  const missing = await createGemma4ImageGpu(unavailable.runtime, (name) => name);
  assert.equal(missing.enabled, false);
  assert.equal(missing.diagnostics.reason, "pipeline-failed");
  assert.deepEqual(unavailable.scopes, []);
  const f = fixture(), helper = await createGemma4ImageGpu(f.runtime, (name) => name), owned = rawFrame();
  f.device.createBindGroup = () => { throw new Error("optional bind group unavailable"); };
  try {
    assert.equal(await helper.prepare(owned.raw, { size: 1024 * 1024 }), null);
    assert.equal(helper.enabled, false);
    assert.equal(helper.diagnostics.reason, "setup-failed");
    assert.ok(f.buffers.every((buffer) => buffer.destroyed));
    assert.ok(owned.raw.rgba.some((value) => value !== 0));
    assert.deepEqual(f.scopes, []);
  } finally { helper.reset(); clearGemma4VisionImageData(owned); }
});

test("device loss remains an error and does not become optional CPU fallback", async () => {
  const f = fixture(), helper = await createGemma4ImageGpu(f.runtime, (name) => name), owned = rawFrame();
  f.device.queue.writeBuffer = () => { f.runtime.lost = { message: "test device removed" }; throw new Error("device lost"); };
  try {
    await assert.rejects(helper.prepare(owned.raw, { size: 1024 * 1024 }), /device lost/);
    assert.ok(f.buffers.every((buffer) => buffer.destroyed));
    assert.deepEqual(f.scopes, []);
  } finally { helper.reset(); clearGemma4VisionImageData(owned); }
});
