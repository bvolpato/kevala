import test from "node:test";
import assert from "node:assert/strict";
import { gemma4ImageSize, prepareGemma4Image } from "../js/src/gemma4-image.js";
import { assertImageCount, imagePixels } from "../js/src/images.js";

test("Gemma image budget caps tokens while retaining aspect ratio and whole pooling cells", () => {
  assert.deepEqual(gemma4ImageSize(320, 240), { width: 432, height: 336 });
  assert.deepEqual(gemma4ImageSize(240, 240), { width: 384, height: 384 });
  assert.deepEqual(gemma4ImageSize(8192, 1), { width: 3360, height: 48 });
  assert.throws(() => gemma4ImageSize(1, 16777216), /8192px/);
  assert.throws(() => gemma4ImageSize(0, 10), /dimensions/);
  assert.throws(() => gemma4ImageSize(10, 10, 71), /token budget/);
});

test("Gemma preprocessing matches pinned Pillow RGB bicubic pixels and row-major patches", () => {
  const data = new Uint8ClampedArray(4 * 3 * 4);
  for (let y = 0; y < 3; y++) for (let x = 0; x < 4; x++) data.set([10 + 20 * x + 3 * y, 40 + 11 * x + 50 * y, 200 - 13 * x - 17 * y, 255], (y * 4 + x) * 4);
  const image = prepareGemma4Image({ width: 4, height: 3, data });
  assert.equal(image.width, 432);
  assert.equal(image.height, 336);
  assert.equal(image.softTokens, 63);
  assert.equal(image.patches.length, 630 * 768);
  assert.deepEqual(Array.from(image.patches.subarray(0, 12)), Array.from({ length: 4 }, () => [8, 33, 204].map((v) => Math.fround(v / 255))).flat());
  assert.deepEqual(Array.from(image.patches.subarray(3 * 768 + 18, 3 * 768 + 21)), [10, 34, 202].map((v) => Math.fround(v / 255)));
  assert.deepEqual(Array.from(image.positions.subarray(0, 4)), [0, 0, 1, 0]);
  assert.deepEqual(Array.from(image.positions.subarray(566 * 2, 568 * 2)), [26, 20, -1, -1]);
  assert.ok(image.patches.subarray(567 * 768).every((x) => x === 0));
});

test("image parts validate RGBA dimensions and request image counts", async () => {
  const pixels = { width: 2, height: 1, data: new Uint8ClampedArray(8) };
  assert.deepEqual(await imagePixels(pixels), pixels);
  await assert.rejects(imagePixels({ ...pixels, data: new Uint8Array(3) }), /RGBA/);
  await assert.rejects(imagePixels({ width: 8192, height: 8192, data: new Uint8Array() }), /megapixel/);
  assert.throws(() => assertImageCount({ parts: Array.from({ length: 5 }, () => ({ type: "image" })) }), /four images/);
});

test("disposing an image session cancels an active image URL download", async (t) => {
  const controller = new AbortController();
  let fetching;
  const started = new Promise((resolve) => { fetching = resolve; });
  t.mock.method(globalThis, "fetch", (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
    fetching();
  }));
  const decoding = imagePixels("https://example.invalid/frame.png", { signal: controller.signal });
  await started;
  controller.abort(new Error("image session disposed"));
  await assert.rejects(decoding, /image session disposed/);
});
