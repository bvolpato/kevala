import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, open, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { assertWasmPackSize, gpuLayouts, packSize, parsePackHeader, readPackHead } from "../js/src/pack-layout.js";
import { PieceSink } from "../js/src/source.js";
import { loadFile } from "../js/src/node.js";

const BIG = 2 ** 32;
const HEAD = 4096;

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

const header = (layout) => parsePackHeader(layout.prefix);

test("Kev GPU layout rebases coordinator ranges above 4 GiB without truncating source offsets", () => {
  const source = fixture();
  const original = structuredClone(source);
  const [coord, gpu] = gpuLayouts(source, HEAD);
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
  const [, gpu] = gpuLayouts(source, HEAD);
  assert.ok(gpu.total > BIG);
  const last = header(gpu).tensors.at(-1);
  assert.ok(last.offset > BIG);
  assert.equal(gpu.pieces.at(-4), last.offset);
  const writes = [];
  const sink = new PieceSink(gpu, (dst, bytes) => writes.push([dst, bytes.byteLength]));
  sink.push(new Uint8Array([1, 2, 3, 4]), source.tensors.at(-1).offset);
  assert.deepEqual(writes.at(-1), [last.offset, 4]);
});

test("Gemma embedding tables stay on the GPU while only label rows enter WebAssembly", () => {
  const source = fixture();
  source.config.arch = "gemma4";
  source.tensors = [
    { name: "embed", dtype: "q8", shape: [262144, 1536], offset: HEAD + 64, size: 402653184, block: 32, scales_offset: HEAD + 64 + 402653184, scales_size: 50331648 },
    { name: "ple.34.embed", dtype: "q8", shape: [262144, 256], offset: BIG + 64, size: 67108864, block: 32, scales_offset: BIG + 64 + 67108864, scales_size: 8388608 },
    { name: "readout.labels", dtype: "f32", shape: [16, 1536], offset: BIG + 134217728, size: 98304 },
  ];
  const [coord, gpu] = gpuLayouts(source, HEAD, (name) => name !== "readout.labels");
  assert.deepEqual(header(coord).tensors.map((t) => t.name), ["readout.labels"]);
  assert.deepEqual(header(gpu).tensors.map((t) => t.name), ["embed", "ple.34.embed"]);
  assert.ok(coord.total < 128 * 1024);
  assert.equal(coord.pieces.at(-5), BIG + 134217728);
  assert.equal(gpu.pieces.at(-5), BIG + 64 + 67108864);
});

test("unordered tensor metadata streams in increasing destination order", () => {
  const source = fixture();
  source.tensors.reverse();
  const [, gpu] = gpuLayouts(source, HEAD);
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
    assert.throws(() => gpuLayouts(source, HEAD), error);
  }
});

test("oversized CPU packs and coordinator subsets fail with actionable allocation errors", () => {
  assert.doesNotThrow(() => assertWasmPackSize(0x7fffffc0));
  assert.throws(() => assertWasmPackSize(2 ** 31), { code: "WASM_PACK_TOO_LARGE" });
  const source = fixture();
  source.tensors = [{ name: "emb", dtype: "f32", shape: [2 ** 29], offset: HEAD + 64, size: 2 ** 31 }];
  assert.throws(() => gpuLayouts(source, HEAD), /coordinator.*native Kevala CLI/);
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

/** A pack prefix: magic, version, header length, and the JSON header padded with spaces. */
function packBytes(header, { magic = "KVLA", version = 1, room = 96, tail = 8 } = {}) {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(16 + room + tail).fill(32, 16, 16 + room);
  bytes.set(new TextEncoder().encode(magic));
  const view = new DataView(bytes.buffer);
  view.setUint32(4, version, true);
  view.setUint32(8, room, true);
  bytes.set(json, 16);
  bytes.fill(0xee, 16 + room);
  return bytes;
}

test("a pack header parses from an offset view and names what is wrong with a bad one", () => {
  const header = { format: "kevala", config: { arch: "kev" }, tensors: [] };
  const bytes = packBytes(header);
  const shifted = new Uint8Array(bytes.byteLength + 5);
  shifted.set(bytes, 5);
  assert.deepEqual(parsePackHeader(shifted.subarray(5)), header);
  assert.throws(() => parsePackHeader(packBytes(header, { magic: "GGUF" })), /not a \.kevala pack/);
  assert.throws(() => parsePackHeader(bytes.subarray(0, 12)), /not a \.kevala pack/);
  assert.throws(() => parsePackHeader(packBytes(header, { version: 2 })), /unsupported \.kevala version 2, this build reads 1/);
  assert.throws(() => parsePackHeader(bytes.subarray(0, 60)), /pack ended before its header/);
});

test("the header is read across any chunking of the stream, and no consumed byte is lost", async () => {
  const header = { format: "kevala", config: { arch: "laya" }, tensors: [] };
  const bytes = packBytes(header);
  const end = 16 + 96;
  for (const size of [1, 7, 16, 17, end - 1, end, end + 3, bytes.byteLength]) {
    async function* stream() {
      for (let at = 0; at < bytes.byteLength; at += size) yield bytes.subarray(at, at + size);
    }
    const chunks = stream();
    const read = await readPackHead(chunks);
    assert.deepEqual(read.header, header, `chunks of ${size}`);
    assert.equal(read.headerBytes.byteLength, end);
    // `head` holds every byte taken from the stream; the rest is still in the stream
    const rest = [];
    for await (const chunk of chunks) rest.push(...chunk);
    assert.deepEqual([...read.head, ...rest], [...bytes], `chunks of ${size}`);
  }
  async function* truncated() {
    yield bytes.subarray(0, 40);
  }
  await assert.rejects(readPackHead(truncated()), /pack ended before its header/);
});

test("a strided piece gathers one column range from every row, whatever the chunk boundaries", () => {
  // 4 rows of 8 source bytes at offset 100; keep columns 2..5 of every row, as a shard's column slice does.
  const source = Uint8Array.from({ length: 200 }, (_, i) => i);
  const layout = { prefix: Uint8Array.of(9, 9), total: 2 + 12, pieces: Uint32Array.of(102, 2, 3, 4, 8) };
  const want = [9, 9, 102, 103, 104, 110, 111, 112, 118, 119, 120, 126, 127, 128];
  for (const size of [1, 2, 3, 5, 8, 11, 200]) {
    const out = new Uint8Array(layout.total);
    const sink = new PieceSink(layout, (dst, bytes) => out.set(bytes, dst));
    for (let at = 0; at < source.byteLength; at += size) sink.push(source.subarray(at, at + size), at);
    assert.deepEqual([...out], want, `chunks of ${size}`);
    assert.equal(sink.done, true);
  }
});
