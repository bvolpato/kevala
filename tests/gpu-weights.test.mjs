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
