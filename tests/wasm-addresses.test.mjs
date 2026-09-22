import assert from "node:assert/strict";
import { test } from "node:test";
import { Wasm } from "../js/src/wasm.js";

const HIGH = 2 ** 31;
const section = (id, data) => [id, data.length, ...data];
const exported = (name, kind, index) => [name.length, ...new TextEncoder().encode(name), kind, index];
const body = (instructions) => [instructions.length + 1, 0, ...instructions];

// A real wasm32 ABI returning 0x80000000 as an i32. Its memory reserves just over 2 GiB,
// while these tests touch only a few bytes above that boundary.
const module = new WebAssembly.Module(new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  ...section(1, [2, 0x60, 0, 1, 0x7f, 0x60, 2, 0x7f, 0x7f, 1, 0x7f]),
  ...section(3, [3, 0, 1, 0]),
  ...section(5, [1, 1, 0x81, 0x80, 2, 0x81, 0x80, 2]), // 32769 pages, fixed maximum
  ...section(6, [1, 0x7f, 1, 0x41, 0, 0x0b]),
  ...section(7, [
    8,
    ...exported("memory", 2, 0),
    ...exported("freed", 3, 0),
    ...exported("kevala_alloc", 0, 0),
    ...exported("kevala_free", 0, 1),
    ...exported("kevala_out_ptr", 0, 0),
    ...exported("kevala_out_len", 0, 2),
    ...exported("kevala_error_ptr", 0, 0),
    ...exported("kevala_error_len", 0, 2),
  ]),
  ...section(10, [
    3,
    ...body([0x41, 0x80, 0x80, 0x80, 0x80, 0x78, 0x0b]),
    ...body([0x20, 0, 0x24, 0, 0x20, 0, 0x0b]),
    ...body([0x41, 4, 0x0b]),
  ]),
]));
const wasm = new Wasm();
wasm.x = new WebAssembly.Instance(module).exports;

test("byte, f32, and u32 views accept signed wasm32 pointers above 2 GiB", () => {
  const pointer = wasm.x.kevala_alloc(8);
  assert.equal(pointer, -HIGH);
  wasm.bytes(pointer, 8).set([1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual([...new Uint8Array(wasm.memory, HIGH, 8)], [1, 2, 3, 4, 5, 6, 7, 8]);
  wasm.f32(pointer, 2).set([1.5, -2.25]);
  assert.deepEqual([...new Float32Array(wasm.memory, HIGH, 2)], [1.5, -2.25]);
  wasm.u32(pointer, 2).set([0x80000000, 0xffffffff]);
  assert.deepEqual([...new Uint32Array(wasm.memory, HIGH, 2)], [0x80000000, 0xffffffff]);
});

test("allocations, input copies, output text, and free preserve the high address", () => {
  assert.equal(wasm.alloc(4), HIGH);
  const [pointer, length] = wasm.put("test");
  assert.deepEqual([pointer, length], [HIGH, 4]);
  assert.equal(wasm.outText(), "test");
  assert.equal(wasm.error(), "test");
  assert.equal(wasm.withInput("next", (ptr, len) => {
    assert.deepEqual([ptr, len], [HIGH, 4]);
    return wasm.outText();
  }), "next");
  assert.equal(wasm.x.freed.value, -HIGH);
  wasm.x.freed.value = 0;
  assert.throws(() => wasm.withInput("fail", () => { throw new Error("callback failed"); }), /callback failed/);
  assert.equal(wasm.x.freed.value, -HIGH);
});
