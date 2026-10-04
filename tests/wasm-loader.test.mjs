import assert from "node:assert/strict";
import { test } from "node:test";
import { compile, parseLayouts, parseSegments, wasmFlavor } from "../js/src/wasm.js";

// The smallest valid module: the magic and version, no sections.
const EMPTY_MODULE = Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0);

function replaceFetch(t, fn) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "fetch");
  Object.defineProperty(globalThis, "fetch", { configurable: true, writable: true, value: fn });
  t.after(() => Object.defineProperty(globalThis, "fetch", previous));
}

test("a failed WebAssembly download is retried, and a compiled module is fetched once", async (t) => {
  const requests = [];
  let online = false;
  replaceFetch(t, async (url) => {
    requests.push(url);
    return online ? new Response(EMPTY_MODULE, { headers: { "content-type": "application/octet-stream" } }) : new Response("gone", { status: 503 });
  });
  const base = "https://example.test/kevala/";
  await assert.rejects(compile(base, "simd"), /kevala-simd\.wasm: HTTP 503/);
  online = true;
  const module = await compile(base, "simd");
  assert.ok(module instanceof WebAssembly.Module);
  assert.equal(await compile(base, "simd"), module);
  assert.deepEqual(requests, Array(2).fill("https://example.test/kevala/kevala-simd.wasm"));
  // another flavor is another file
  await compile(base, "base");
  assert.equal(requests.at(-1), "https://example.test/kevala/kevala-base.wasm");
});

test("an explicit flavor wins; otherwise the best build this runtime validates is chosen", () => {
  assert.equal(wasmFlavor("base"), "base");
  assert.equal(wasmFlavor("simd"), "simd");
  assert.ok(["relaxed", "simd", "base"].includes(wasmFlavor()));
});

test("layout blobs parse into a prefix and five-number pieces, one layout after another", () => {
  const u32 = (...values) => new Uint8Array(Uint32Array.from(values).buffer);
  const blob = new Uint8Array([
    ...u32(3), 7, 8, 9, ...u32(4096, 2), ...u32(64, 128, 16, 1, 0), ...u32(256, 512, 8, 4, 32),
    ...u32(0), ...u32(64, 0),
  ]);
  // parse from an offset view, as the bytes sit in WebAssembly memory
  const padded = new Uint8Array(blob.byteLength + 3);
  padded.set(blob, 3);
  const layouts = parseLayouts(padded.subarray(3));
  assert.equal(layouts.length, 2);
  assert.deepEqual([...layouts[0].prefix], [7, 8, 9]);
  assert.equal(layouts[0].total, 4096);
  assert.deepEqual([...layouts[0].pieces], [64, 128, 16, 1, 0, 256, 512, 8, 4, 32]);
  assert.deepEqual({ prefix: layouts[1].prefix.length, total: layouts[1].total, pieces: layouts[1].pieces.length }, { prefix: 0, total: 64, pieces: 0 });
});

test("segment tables parse into token counts, segments, and marker positions", () => {
  const table = Uint32Array.of(30, 2, /* segment */ 0, 18, 2, 2, 5, 9, /* segment */ 18, 12, 0, 3, 4, 6, 8);
  assert.deepEqual(parseSegments(table), {
    tokens: 30,
    segs: [
      { start: 0, len: 18, qtype: 2, markers: [5, 9] },
      { start: 18, len: 12, qtype: 0, markers: [4, 6, 8] },
    ],
  });
});
