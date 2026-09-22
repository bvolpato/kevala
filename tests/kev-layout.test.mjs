import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, open, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { assertWasmPackSize, kevGpuLayouts, packSize } from "../js/src/kev-layout.js";
import { PieceSink, upstreamKey } from "../js/src/source.js";
import { loadFile } from "../js/src/node.js";

const BIG = 2 ** 32;
const HEAD = 4096;

test("reconverted packs with the same checkpoint get distinct cache keys", () => {
  const spec = { repo: "Qwen/example", revision: "checkpoint", block: 32 };
  const first = upstreamKey({ ...spec, packSha256: "first" });
  const corrected = upstreamKey({ ...spec, packSha256: "corrected" });
  assert.notEqual(first, corrected);
  assert.equal(upstreamKey(spec), "https://kevala.cache/Qwen/example/checkpoint/q8-b32.kevala");
});

function fixture() {
  return {
    format: "kevala", version: 1, model: { name: "synthetic-kev" }, config: { arch: "kev", readout: "semif" },
    tokenizer: { offset: HEAD, size: 8 },
    tensors: [
      { name: "emb", dtype: "q8", shape: [2, 32], offset: HEAD + 64, size: 64, block: 32, scales_offset: HEAD + 128, scales_size: 8 },
      { name: "L.0.norm", dtype: "f32", shape: [4], offset: BIG + 64, size: 16 },
      { name: "L.0.qkv", dtype: "q8", shape: [2, 32], offset: BIG + 128, size: 64, block: 32, scales_offset: BIG + 192, scales_size: 8 },
      { name: "semif.labels", dtype: "f32", shape: [4], offset: BIG + 256, size: 16 },
    ],
  };
}

function header(layout) {
  const size = new DataView(layout.prefix.buffer).getUint32(8, true);
  return JSON.parse(new TextDecoder().decode(layout.prefix.subarray(16, 16 + size)));
}

test("Kev GPU layout rebases coordinator ranges above 4 GiB without truncating source offsets", () => {
  const source = fixture();
  const original = structuredClone(source);
  const [coord, gpu] = kevGpuLayouts(source, HEAD);
  assert.deepEqual(source, original);
  assert.deepEqual(header(coord).tensors.map((t) => t.name), ["emb", "semif.labels"]);
  assert.deepEqual(header(gpu).tensors.map((t) => t.name), ["L.0.norm", "L.0.qkv"]);
  assert.deepEqual(header(coord).model, source.model);
  assert.deepEqual(header(coord).config, source.config);
  assert.equal(header(gpu).tokenizer.size, 0);
  assert.ok(coord.total < 65536);
  assert.equal(coord.pieces.at(-5), BIG + 256);
  assert.equal(gpu.pieces[0], BIG + 64);
  assert.equal(gpu.pieces.at(-5), BIG + 192);
  assert.equal(packSize(source, HEAD), BIG + 272);

  const bytes = new Uint8Array(coord.total);
  const sink = new PieceSink(coord, (dst, chunk) => bytes.set(chunk, dst));
  for (let i = 0; i < coord.pieces.length; i += 5) {
    const [src, dst, size] = coord.pieces.slice(i, i + 3);
    const data = Uint8Array.from({ length: size }, (_, j) => i + j);
    sink.push(data.subarray(0, 3), src);
    sink.push(data.subarray(3), src + 3);
    assert.deepEqual(bytes.subarray(dst, dst + size), data);
  }
  assert.equal(sink.done, true);
});

test("GPU destination offsets retain precision beyond 4 GiB without allocating the tensors", () => {
  const source = fixture();
  source.tensors = Array.from({ length: 5 }, (_, i) => ({
    name: `L.${i}.matrix`, dtype: "f32", shape: [16384, 16384], offset: HEAD + 64 + i * 2 ** 30, size: 2 ** 30,
  }));
  const [, gpu] = kevGpuLayouts(source, HEAD);
  assert.ok(gpu.total > BIG);
  const last = header(gpu).tensors.at(-1);
  assert.ok(last.offset > BIG);
  assert.equal(gpu.pieces.at(-4), last.offset);
  const writes = [];
  const sink = new PieceSink(gpu, (dst, bytes) => writes.push([dst, bytes.byteLength]));
  sink.push(new Uint8Array([1, 2, 3, 4]), source.tensors.at(-1).offset);
  assert.deepEqual(writes.at(-1), [last.offset, 4]);
});

test("unordered tensor metadata streams in increasing destination order", () => {
  const source = fixture();
  source.tensors.reverse();
  const [, gpu] = kevGpuLayouts(source, HEAD);
  const writes = [];
  const sink = new PieceSink(gpu, (dst, bytes) => writes.push([dst, bytes.length]));
  sink.push(new Uint8Array(136), BIG + 64);
  assert.equal(sink.done, true);
  for (let i = 1; i < writes.length; i++) assert.ok(writes[i][0] >= writes[i - 1][0] + writes[i - 1][1]);
});

test("invalid tensor sizes and unsafe or overlapping ranges are rejected before allocation", () => {
  const cases = [
    [(h) => { h.tensors[1].offset = Number.MAX_SAFE_INTEGER + 1; }, /safe integer/],
    [(h) => { h.tensors[1].offset = Number.MAX_SAFE_INTEGER - 63; h.tensors[1].shape = [32]; h.tensors[1].size = 128; }, /safe integer/],
    [(h) => { h.tensors[1].shape = [Number.MAX_SAFE_INTEGER, 2]; }, /safe integer/],
    [(h) => { h.tensors[1].size--; }, /inconsistent size/],
    [(h) => { h.tensors[2].scales_size = 4; }, /inconsistent q8 layout/],
    [(h) => { h.tensors[2].block = 0; }, /inconsistent q8 layout/],
    [(h) => { h.tensors[1].offset++; }, /unaligned/],
    [(h) => { h.tensors[1].offset = HEAD; }, /overlaps/],
    [(h) => { h.tensors[1].name = "emb"; }, /unique/],
  ];
  for (const [mutate, error] of cases) {
    const source = fixture();
    mutate(source);
    assert.throws(() => kevGpuLayouts(source, HEAD), error);
  }
});

test("oversized CPU packs and coordinator subsets fail with actionable allocation errors", () => {
  assert.doesNotThrow(() => assertWasmPackSize(0x7fffffc0));
  assert.throws(() => assertWasmPackSize(2 ** 31), { code: "WASM_PACK_TOO_LARGE" });
  const source = fixture();
  source.tensors = [{ name: "emb", dtype: "f32", shape: [2 ** 29], offset: HEAD + 64, size: 2 ** 31 }];
  assert.throws(() => kevGpuLayouts(source, HEAD), /coordinator.*native Kevala CLI/);
});

test("Node rejects a large pack by file size before reading weights or compiling WebAssembly", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "kevala-large-pack-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const path = join(dir, "large.kevala");
  const file = await open(path, "w");
  try {
    await file.truncate(BIG + 64);
  } finally {
    await file.close();
  }
  await assert.rejects(loadFile(path), { code: "WASM_PACK_TOO_LARGE" });
});
