import assert from "node:assert/strict";
import test from "node:test";
import { GpuKev } from "../js/src/gpu-kev.js";
import { GpuWeights } from "../js/src/gpu.js";
import { gpuLayouts, parsePackHeader } from "../js/src/pack-layout.js";

function device(t, limit = 128) {
  const previous = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2 };
  t.after(() => previous === undefined ? delete globalThis.GPUBufferUsage : globalThis.GPUBufferUsage = previous);
  return {
    limits: { maxBufferSize: limit, maxStorageBufferBindingSize: limit },
    createBuffer({ size }) { assert.ok(size <= limit); return { bytes: new Uint8Array(size) }; },
    queue: {
      writeBuffer(buffer, offset, data, start = 0, length = data.byteLength) {
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data, start, length) : new Uint8Array(data.buffer, data.byteOffset + start, length);
        assert.equal(offset % 4, 0);
        assert.equal(length % 4, 0);
        buffer.bytes.set(bytes, offset);
      },
    },
  };
}

function layout(tensor) {
  const header = { config: { arch: "kev" }, model: {}, tokenizer: { offset: 4096, size: 0 }, tensors: [tensor] };
  return gpuLayouts(header, 4096, () => true)[1];
}

test("Qwen GPU rejects unsupported Q8 projection blocks before allocating weights", (t) => {
  const d = device(t);
  let allocations = 0;
  d.createBuffer = () => { allocations++; throw new Error("unexpected GPU allocation"); };
  const config = { layers: 0, heads: 4, kv_heads: 1, lin_key_heads: 2, lin_heads: 2 };
  for (const block of [16, 64]) {
    const gpu = layout({ name: "L.0.qkv", dtype: "q8", shape: [1, 64], offset: 4096, size: 64, block, scales_offset: 4160, scales_size: 64 / block * 4 });
    assert.throws(() => new GpuKev({ device: d }, gpu, config), (error) =>
      error.code === "WEBGPU_INIT" && error.message.includes(`L.0.qkv has block size ${block}`));
  }
  const supported = layout({ name: "L.0.qkv", dtype: "q8", shape: [1, 64], offset: 4096, size: 64, block: 32, scales_offset: 4160, scales_size: 8 });
  assert.doesNotThrow(() => new GpuKev({ device: d }, supported, config));
  assert.equal(allocations, 0);
});

test("BF16 row tables stream across word and shard boundaries without a host-sized table", (t) => {
  const gpu = layout({ name: "emb", dtype: "bf16", shape: [5, 32], offset: 4096, size: 320 });
  const info = parsePackHeader(gpu.prefix).tensors[0];
  const weights = new GpuWeights(device(t), gpu, { rowTables: ["emb"] });
  const data = Uint8Array.from({ length: info.size }, (_, i) => i % 251);
  assert.deepEqual(weights.missing(), ["emb"]);
  for (let at = 0; at < data.length; at += 7) weights.write(info.offset + at, data.subarray(at, at + 7));
  const entry = weights.get("emb");
  assert.deepEqual(entry.shards.map(({ start, count }) => [start, count]), [[0, 2], [2, 2], [4, 1]]);
  for (const shard of entry.shards) assert.deepEqual(shard.buf.bytes.subarray(0, shard.count * 64), data.subarray(shard.start * 64, (shard.start + shard.count) * 64));
  assert.deepEqual(weights.missing(), []);
  assert.equal(entry.data, null);
  assert.ok(entry.shards.every((shard) => shard.sbuf === null));
});

test("Q8 row table scales follow the same row shards as packed weights", (t) => {
  const gpu = layout({ name: "emb", dtype: "q8", shape: [6, 32], offset: 4096, size: 192, block: 32, scales_offset: 4288, scales_size: 24 });
  const info = parsePackHeader(gpu.prefix).tensors[0];
  const weights = new GpuWeights(device(t), gpu, { rowTables: ["emb"] });
  const data = Uint8Array.from({ length: 192 }, (_, i) => i);
  const scales = new Float32Array([1, 2, 3, 4, 5, 6]);
  weights.write(info.offset, data);
  assert.deepEqual(weights.missing(), ["emb"]);
  const bytes = new Uint8Array(scales.buffer);
  for (let at = 0; at < bytes.length; at += 3) weights.write(info.scales_offset + at, bytes.subarray(at, at + 3));
  const entry = weights.get("emb");
  assert.deepEqual(entry.shards.map(({ start, count }) => [start, count]), [[0, 4], [4, 2]]);
  for (const shard of entry.shards) {
    assert.deepEqual(shard.buf.bytes.subarray(0, shard.count * 32), data.subarray(shard.start * 32, (shard.start + shard.count) * 32));
    assert.deepEqual(new Float32Array(shard.sbuf.bytes.buffer, 0, shard.count), scales.subarray(shard.start, shard.start + shard.count));
  }
  assert.deepEqual(weights.missing(), []);
});

test("an odd BF16 vector keeps its final element in a padded GPU write", (t) => {
  const gpu = layout({ name: "L.0.vector", dtype: "bf16", shape: [3], offset: 4096, size: 6 });
  const info = parsePackHeader(gpu.prefix).tensors[0];
  const weights = new GpuWeights(device(t), gpu);
  const bytes = new Uint8Array(new Uint16Array([0x3f80, 0xbf80, 0x4000]).buffer);
  weights.write(info.offset, bytes);
  assert.deepEqual(weights.get(info.name).buf.bytes.subarray(0, 6), bytes);
  assert.deepEqual(weights.missing(), []);
});

test("host Q8 tables retain streamed bytes without a GPU allocation and gather exact F32 rows", (t) => {
  const gpu = layout({ name: "ple.0.embed", dtype: "q8", shape: [3, 64], offset: 4096, size: 192, block: 32, scales_offset: 4288, scales_size: 24 });
  const info = parsePackHeader(gpu.prefix).tensors[0];
  const d = device(t, 128);
  d.createBuffer = () => { throw new Error("a host table must not allocate on the GPU"); };
  const weights = new GpuWeights(d, gpu, { hostTables: [info.name] });
  const signed = Int8Array.from({ length: info.size }, (_, i) => i % 255 - 128);
  const bytes = new Uint8Array(signed.buffer);
  const scales = new Float32Array([0.123456789, 0.321654987, -0.045678912, 0.009876543, 0.333333333, 0.777777777]);
  for (let at = 0; at < bytes.length; at += 7) weights.write(info.offset + at, bytes.subarray(at, at + 7));
  assert.deepEqual(weights.missing(), [info.name]);
  assert.throws(() => weights.gatherHostRows(info.name, [0]), /is incomplete/);
  const scaleBytes = new Uint8Array(scales.buffer);
  for (let at = 0; at < scaleBytes.length; at += 3) weights.write(info.scales_offset + at, scaleBytes.subarray(at, at + 3));
  const entry = weights.get(info.name);
  assert.deepEqual(weights.missing(), []);
  assert.equal(entry.buf, null);
  assert.equal(entry.sbuf, null);
  assert.equal(entry.data, null);
  assert.equal(entry.scales, null);
  assert.deepEqual(entry.host.data, signed);
  assert.deepEqual(entry.host.scales, scales);
  const ids = [2, 0, 2, 1];
  const factor = 1.123456789;
  const target = new Float32Array(5 + ids.length * 64 + 3).fill(100);
  assert.equal(weights.gatherHostRows(info.name, ids, factor, target, 5), target);
  for (let row = 0; row < ids.length; row++) for (let col = 0; col < 64; col++) {
    const expected = Math.fround(Math.fround(signed[ids[row] * 64 + col] * scales[ids[row] * 2 + Math.floor(col / 32)]) * Math.fround(factor));
    assert.equal(target[5 + row * 64 + col], expected);
  }
  assert.deepEqual([...target.subarray(0, 5)], [100, 100, 100, 100, 100]);
  assert.deepEqual([...target.subarray(-3)], [100, 100, 100]);
  for (const ids of [[-1], [3], [0.5], [NaN]]) assert.throws(() => weights.gatherHostRows(info.name, ids), /invalid row selection/);
  assert.throws(() => weights.gatherHostRows(info.name, [0], Infinity), /invalid row selection/);
  assert.throws(() => weights.gatherHostRows(info.name, [0], 1, new Float32Array(63)), /undersized F32 destination/);
});

test("host Q8 row tables reject unsupported storage modes before allocating", (t) => {
  const d = device(t);
  for (const tensor of [
    { name: "emb", dtype: "bf16", shape: [2, 32], offset: 4096, size: 128 },
    { name: "emb", dtype: "q8", shape: [2, 32], offset: 4096, size: 64, block: 16, scales_offset: 4160, scales_size: 16 },
  ]) assert.throws(() => new GpuWeights(d, layout(tensor), { hostTables: ["emb"] }), /requires a block-32 Q8 matrix/);
  const gpu = layout({ name: "emb", dtype: "q8", shape: [2, 32], offset: 4096, size: 64, block: 32, scales_offset: 4160, scales_size: 8 });
  assert.throws(() => new GpuWeights(d, gpu, { hostTables: ["emb"], rowTables: ["emb"] }), /cannot be both sharded and host-resident/);
});
