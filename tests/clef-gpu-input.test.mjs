import assert from "node:assert/strict";
import test from "node:test";
import { lexicalMeans } from "../js/src/archs/clef.js";
import { GpuKev } from "../js/src/gpu-kev.js";
import { gpuLayouts } from "../js/src/pack-layout.js";

function trunk(t) {
  const old = globalThis.GPUBufferUsage;
  globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 };
  t.after(() => old === undefined ? delete globalThis.GPUBufferUsage : globalThis.GPUBufferUsage = old);
  const copies = [];
  const device = {
    limits: { maxBufferSize: 1024 * 1024, maxStorageBufferBindingSize: 1024 * 1024, minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256 },
    features: new Set(),
    createBuffer: ({ size }) => ({ bytes: new Uint8Array(size), destroy() {} }),
    createBindGroup: ({ entries }) => entries.map(({ resource }) => resource),
    queue: { writeBuffer: (buffer, offset, values) => buffer.bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), offset) },
  };
  const header = { config: { arch: "kev" }, model: {}, tokenizer: { offset: 4096, size: 0 }, tensors: [] };
  const cfg = { hidden: 4, layers: 0, heads: 4, kv_heads: 1, lin_key_heads: 2, lin_heads: 2 };
  const gpu = new GpuKev({ device }, gpuLayouts(header, 4096, () => true)[1], cfg);
  gpu.cap.T2 = 8;
  gpu.owned = [];
  gpu.x2 = device.createBuffer({ size: 8 * 4 * 4 });
  gpu.zeros = device.createBuffer({ size: 16 });
  const table = { info: { name: "emb", dtype: "f32", shape: [6, 4] }, shards: [[0, 3], [3, 3]].map(([start, count]) => ({ start, count, buf: { bytes: new Uint8Array(Float32Array.from({ length: count * 4 }, (_, i) => start * 4 + i + 0.25).buffer) } })) };
  gpu.weights = { get: () => table };
  const pipeline = { getBindGroupLayout() {} };
  gpu.rowPipelines = new Map([["f32", pipeline]]);
  const encoder = {
    beginComputePass() {
      let bindings;
      return {
        setPipeline() {},
        setBindGroup: (_, group) => bindings = group,
        dispatchWorkgroups() {
          const [p, ids, weights, , out] = bindings;
          const [rows, width, , first] = new Uint32Array(p.buffer.bytes.buffer, p.offset, 4);
          const selected = new Uint32Array(ids.buffer.bytes.buffer, ids.offset, rows);
          for (let row = 0; row < rows; row++) {
            const source = (selected[row] - first) * width * 4;
            out.buffer.bytes.set(weights.buffer.bytes.subarray(source, source + width * 4), row * width * 4);
          }
        },
        end() {},
      };
    },
    copyBufferToBuffer(source, sourceAt, destination, destinationAt, bytes) {
      copies.push(bytes);
      destination.bytes.set(source.bytes.subarray(sourceAt, sourceAt + bytes), destinationAt);
    },
  };
  return { gpu, table, pipeline, encoder, copies };
}

test("resident embeddings preserve duplicate and out-of-order rows across shards without mapping", (t) => {
  const { gpu, table, pipeline, encoder, copies } = trunk(t);
  const input = { table, pipeline, ids: Uint32Array.of(4, 1, 3, 4, 0) };
  gpu.prepareRowInput(input);
  gpu.encodeRowInput(encoder, input);
  assert.deepEqual(new Float32Array(gpu.x2.bytes.buffer, 0, input.ids.length * 4), Float32Array.from(Array.from(input.ids).flatMap((id) => Array.from({ length: 4 }, (_, col) => id * 4 + col + 0.25))));
  assert.deepEqual(copies, [16, 16, 16, 32]);
  const buffers = gpu.rowInput.buffers;
  gpu.prepareRowInput(input);
  assert.equal(gpu.rowInput.buffers, buffers);
});

test("one participating shard gathers directly into the branch buffer", (t) => {
  const { gpu, table, pipeline, encoder, copies } = trunk(t);
  const input = { table, pipeline, ids: Uint32Array.of(5, 3, 5) };
  gpu.prepareRowInput(input);
  gpu.encodeRowInput(encoder, input);
  assert.deepEqual(Array.from(new Float32Array(gpu.x2.bytes.buffer, 0, 12)), [20.25, 21.25, 22.25, 23.25, 12.25, 13.25, 14.25, 15.25, 20.25, 21.25, 22.25, 23.25]);
  assert.deepEqual(copies, []);
});

test("resident row inputs validate token ids and the complete-branch contract", async (t) => {
  const { gpu } = trunk(t);
  gpu.forward = async (_, input) => input.ids;
  const batch = { T1: 0, T2: 2 };
  assert.deepEqual(await gpu.forwardRows("emb", Uint32Array.of(5, 0), batch), Uint32Array.of(5, 0));
  await assert.rejects(gpu.forwardRows("emb", [6, 0], batch), /out-of-range row/);
  await assert.rejects(gpu.forwardRows("emb", [0, 1], { T1: 1, T2: 2 }), /complete branch/);
  await assert.rejects(gpu.forwardRows("emb", [0], batch), /complete branch/);
});

test("lexical option means preserve f32 accumulation and cache immutable results per trunk", async () => {
  const rows = Float32Array.of(1e8, 0.1, -3, 0.3, 1, 0.2, 7, 0.1, -1e8, 0.3, 5, 0.2, 2, 0.4, 11, 0.9);
  let reads = 0;
  const gpu = { async weightRows(_, ids) { reads++; return Float32Array.from(Array.from(ids).flatMap((id) => Array.from(rows.subarray(id * 4, (id + 1) * 4)))); } };
  const ids = Uint32Array.of(0, 1, 2, 0, 3);
  const spans = [[0, 3], [3, 5], [0, 3]];
  const baseline = await lexicalMeans(gpu, ids, spans, 4, false);
  assert.deepEqual(baseline, Float32Array.of(0, 0.2, 3, 0.2, 5e7, 0.25, 4, 0.6, 0, 0.2, 3, 0.2));
  const first = await lexicalMeans(gpu, ids, spans, 4);
  assert.deepEqual(first, baseline);
  const original = first.slice();
  first.fill(999);
  assert.deepEqual(await lexicalMeans(gpu, ids, spans, 4), original);
  assert.equal(reads, 2);
  const other = { weightRows: gpu.weightRows };
  await lexicalMeans(other, ids, spans, 4);
  assert.equal(reads, 3);
});

test("lexical cache evicts old options and keeps recently used ones", async () => {
  let reads = 0;
  const gpu = { async weightRows(_, ids) { reads++; return Float32Array.from(ids); } };
  for (let id = 0; id < 256; id++) await lexicalMeans(gpu, Uint32Array.of(id), [[0, 1]], 1);
  await lexicalMeans(gpu, Uint32Array.of(0), [[0, 1]], 1);
  await lexicalMeans(gpu, Uint32Array.of(256), [[0, 1]], 1);
  await lexicalMeans(gpu, Uint32Array.of(0), [[0, 1]], 1);
  assert.equal(reads, 257);
  await lexicalMeans(gpu, Uint32Array.of(1), [[0, 1]], 1);
  assert.equal(reads, 258);
});
