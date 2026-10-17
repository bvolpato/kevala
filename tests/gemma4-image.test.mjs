import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { clearGemma4VisionImageData, consumeGemma4VisionImage, disposeGemma4VisionImage, gemma4ImageSize, gemma4VisionPixelLut, materializeGemma4VisionPixels, prepareGemma4GpuVisionImage, prepareGemma4Image, prepareGemma4VisionImage } from "../js/src/gemma4-image.js";
import { assertImageCount, imagePixels } from "../js/src/images.js";

function imageDigest(values) {
  return createHash("sha256").update(new Uint8Array(values.buffer, values.byteOffset, values.byteLength)).digest("hex");
}

test("Gemma image budget caps tokens while retaining aspect ratio and whole pooling cells", () => {
  assert.deepEqual(gemma4ImageSize(320, 240), { width: 432, height: 336 });
  assert.deepEqual(gemma4ImageSize(240, 240), { width: 384, height: 384 });
  assert.deepEqual(gemma4ImageSize(8192, 1), { width: 3360, height: 48 });
  assert.throws(() => gemma4ImageSize(1, 16777216), /8192px/);
  assert.throws(() => gemma4ImageSize(0, 10), /dimensions/);
  assert.throws(() => gemma4ImageSize(10, 10, 0), /token budget/);
  assert.deepEqual(gemma4ImageSize(320, 240, 16), { width: 192, height: 144 });
  assert.deepEqual(gemma4ImageSize(320, 240, 35), { width: 288, height: 240 });
  assert.deepEqual(gemma4ImageSize(320, 240, 40), { width: 336, height: 240 });
  assert.throws(() => gemma4ImageSize(10, 10, 1.5), /integer from 1 to 1120/);
  assert.throws(() => gemma4ImageSize(10, 10, 1121), /integer from 1 to 1120/);
});

test("smaller image budgets produce whole pooling cells with finite zero padding", () => {
  const width = 320, height = 240;
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i += 4) data.set([90, 130, 180, 255], i);
  for (const [budget, expectedWidth, expectedHeight, expectedTokens] of [[16, 192, 144, 12], [35, 288, 240, 30], [40, 336, 240, 35], [70, 432, 336, 63]]) {
    const image = prepareGemma4Image({ width, height, data }, budget);
    const realPatches = expectedWidth / 16 * (expectedHeight / 16);
    assert.equal(image.width, expectedWidth);
    assert.equal(image.height, expectedHeight);
    assert.equal(image.softTokens, expectedTokens);
    assert.equal(image.patches.length, budget * 9 * 768);
    assert.ok(image.patches.every(Number.isFinite));
    assert.ok(image.positions.subarray(realPatches * 2).every((position) => position === -1));
    assert.ok(image.patches.subarray(realPatches * 768).every((pixel) => pixel === 0));
  }
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

test("optimized Gemma preprocessing preserves complete patches and positions bit-for-bit", () => {
  const width = 320, height = 240;
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    data[i] = (x * 37 + y * 19 + x * y * 3) & 255;
    data[i + 1] = (x * 5 + y * 61 + 17) & 255;
    data[i + 2] = (x * 13 + y * 11 + 101) & 255;
    data[i + 3] = (x * 7 + y * 29) & 255;
  }
  const source = { width, height, data };
  const expected = [
    [70, 432, 336, 63, "61bdf24a3cb3be9b9cbbc0500ad75f021886aaaa0ddf10eef51b3ef2a6ff5d6b", "fbf4527641984a1d3419f329d99b9c96fac8960bdbc3f5b1e9faf8a4cd48c68b"],
    [40, 336, 240, 35, "1abf5eaeb2fdacebe04beeef6e0fa9bfb99989a835f415b1572f0fcb0c3f6cb6", "19c6d39f8f2ced4fa72b7d19bc4c6c870203f3f58e3241c7a2e9cf23aa2506c0"],
    [16, 192, 144, 12, "7e71c59f96966d98e6bf33d051a2f4fc85dfa6da3647dfdb838d9c84effe5db3", "005dff2d66c9e257faca051bbedabda62a54ad90cf70f878b552f70cf8f9d0c0"],
  ];
  for (const [budget, expectedWidth, expectedHeight, softTokens, patches, positions] of expected) {
    const result = prepareGemma4Image(source, budget);
    assert.deepEqual([result.width, result.height, result.softTokens], [expectedWidth, expectedHeight, softTokens]);
    assert.equal(imageDigest(result.patches), patches, `patch output changed at budget ${budget}`);
    assert.equal(imageDigest(result.positions), positions, `position output changed at budget ${budget}`);
  }
});

test("reused resize geometry never reuses camera pixels or mutable patch output", () => {
  const pixels = (width, height, color) => {
    const data = new Uint8Array(width * height * 4);
    for (let i = 0; i < data.length; i += 4) data.set([...color, 255], i);
    return { width, height, data };
  };
  const first = prepareGemma4Image(pixels(4, 3, [255, 0, 0]));
  const second = prepareGemma4Image(pixels(4, 3, [0, 0, 255]));
  const secondSample = second.patches[0];

  assert.notDeepEqual(first.patches, second.patches);
  assert.notStrictEqual(first.patches, second.patches);
  first.patches.fill(0);
  assert.equal(second.patches[0], secondSample);
  const rotated = prepareGemma4Image(pixels(3, 4, [0, 255, 0]));
  assert.equal(rotated.width, first.height);
  assert.equal(rotated.height, first.width);
});

test("owned compact pixels preserve both normalization rounds for every U8 value", () => {
  const data = Uint8ClampedArray.from({ length: 48 * 48 * 4 }, (_, i) => (i * 17 + Math.floor(i / 4)) & 255);
  const source = { width: 48, height: 48, data };
  const reference = prepareGemma4Image(source, 1);
  const handle = prepareGemma4VisionImage(source, 1);
  const owned = consumeGemma4VisionImage(handle, { patch: 16, pool: 3 });
  try {
    const seen = new Set();
    for (let i = 0; i < owned.pixels.length; i++) {
      seen.add(Math.round(reference.patches[i] * 255));
      assert.equal(owned.pixels[i], Math.fround(2 * (reference.patches[i] - 0.5)));
    }
    assert.equal(seen.size, 256);
    assert.deepEqual(owned.positions, reference.positions);
  } finally { owned.pixels.fill(0); reference.patches.fill(0); }
});

test("compact preparation matches complete public pixels at camera budgets and capacity bounds", () => {
  const cfg = { patch: 16, pool: 3 };
  const source = (width, height) => ({ width, height, data: Uint8Array.from({ length: width * height * 4 }, (_, i) => (i * 37 + Math.floor(i / 11)) & 255) });
  const cases = [[320, 240, 16], [320, 240, 40], [320, 240, 70], [1, 1, 1], [3, 4, 140], [1, 1, 280], [1, 1, 560], [1, 1, 1120], [8192, 1, 1]];
  for (const [width, height, budget] of cases) {
    const input = source(width, height);
    const reference = prepareGemma4Image(input, budget);
    const handle = prepareGemma4VisionImage(input, budget);
    const owned = consumeGemma4VisionImage(handle, cfg);
    try {
      assert.equal(handle.width, reference.width);
      assert.equal(handle.height, reference.height);
      assert.equal(handle.softTokens, reference.softTokens);
      assert.equal(handle.paddedRows, reference.positions.length / 2);
      assert.equal(owned.pixels.length, reference.width * reference.height * 3);
      assert.deepEqual(owned.positions, reference.positions);
      for (let i = 0; i < owned.pixels.length; i++) {
        const expected = Math.fround(2 * (reference.patches[i] - 0.5));
        if (!Object.is(owned.pixels[i], expected)) assert.equal(owned.pixels[i], expected, `${width}x${height}, budget ${budget}, pixel ${i}`);
      }
    } finally { owned.pixels.fill(0); reference.patches.fill(0); }
  }
  assert.throws(() => prepareGemma4VisionImage(source(1, 1), 0), /token budget/);
  assert.throws(() => prepareGemma4VisionImage(source(1, 1), 1121), /token budget/);
  assert.throws(() => prepareGemma4VisionImage(source(1, 1), 1, { patch_size: 8 }), /16px patches/);
});

test("owned preparation transfers pixels once and disposal cannot clear a transferred frame", () => {
  const cfg = { patch: 16, pool: 3 };
  const input = { width: 1, height: 1, data: Uint8Array.of(255, 0, 127, 5) };
  const handle = prepareGemma4VisionImage(input, 1);
  assert.throws(() => { handle.softTokens = 99; }, TypeError);
  assert.equal(consumeGemma4VisionImage({ ...handle }, cfg), null);
  const owned = consumeGemma4VisionImage(handle, cfg);
  disposeGemma4VisionImage(handle);
  assert.equal(owned.pixels[0], 1);
  assert.throws(() => consumeGemma4VisionImage(handle, cfg), /already been consumed or disposed/);
  owned.pixels.fill(0);
  const pending = prepareGemma4VisionImage(input, 1);
  disposeGemma4VisionImage(pending);
  disposeGemma4VisionImage(pending);
  assert.throws(() => consumeGemma4VisionImage(pending, cfg), /already been consumed or disposed/);
  assert.deepEqual(input.data, Uint8Array.of(255, 0, 127, 5));
});

test("owned preparation validates the underlying byte view and snapshots mutable source metadata", () => {
  class MisreportedBytes extends Uint8Array { get length() { return 8; } }
  assert.throws(() => prepareGemma4VisionImage({ width: 2, height: 1, data: new MisreportedBytes(4) }, 1), /RGBA bytes/);
  let reads = 0;
  const source = { width: 1, height: 1, get data() { return ++reads === 1 ? Uint8Array.of(255, 0, 127, 1) : [NaN, NaN, NaN, NaN]; } };
  const handle = prepareGemma4VisionImage(source, 1);
  const owned = consumeGemma4VisionImage(handle, { patch: 16, pool: 3 });
  assert.ok(owned.pixels.every(Number.isFinite));
  assert.equal(owned.pixels[0], 1);
  owned.pixels.fill(0);
  const invalid = prepareGemma4VisionImage({ width: 1, height: 1, data: new Uint8Array(4) }, 1);
  const failure = new Error("configuration unavailable");
  assert.throws(() => consumeGemma4VisionImage(invalid, { get patch() { throw failure; } }), (error) => error === failure);
  assert.throws(() => consumeGemma4VisionImage(invalid, { patch: 16, pool: 3 }), /already been consumed or disposed/);
});

test("private GPU preparation snapshots RGBA and preserves exact CPU fallback pixels", () => {
  const cfg = { patch: 16, pool: 3 };
  for (const [width, height, budget] of [[32, 24, 16], [3, 5, 40], [1, 1, 1]]) {
    const data = Uint8Array.from({ length: width * height * 4 }, (_, i) => (i * 37 + Math.floor(i / 7)) & 255);
    const reference = prepareGemma4Image({ width, height, data }, budget);
    const handle = prepareGemma4GpuVisionImage({ width, height, data }, budget);
    assert.equal(consumeGemma4VisionImage({ ...handle }, cfg), null);
    const owned = consumeGemma4VisionImage(handle, cfg);
    assert.ok(owned.raw);
    assert.notStrictEqual(owned.raw.rgba.buffer, data.buffer);
    assert.deepEqual(owned.raw.rgba, data);
    data.fill(0);
    disposeGemma4VisionImage(handle);
    assert.ok(owned.raw.rgba.some((value) => value !== 0));
    const pixels = materializeGemma4VisionPixels(owned);
    assert.deepEqual(owned.positions, reference.positions);
    for (let i = 0; i < pixels.length; i++) assert.ok(Object.is(pixels[i], Math.fround(2 * (reference.patches[i] - 0.5))), `pixel ${i}`);
    assert.ok(owned.raw.rgba.every((value) => value === 0));
    assert.strictEqual(materializeGemma4VisionPixels(owned), pixels);
    clearGemma4VisionImageData(owned);
    assert.ok(pixels.every((value) => value === 0));
    assert.throws(() => consumeGemma4VisionImage(handle, cfg), /already been consumed or disposed/);
  }
  const lut = gemma4VisionPixelLut();
  for (let i = 0; i < 256; i++) assert.ok(Object.is(lut[i], Math.fround(2 * (Math.fround(i / 255) - 0.5))));
  lut.fill(0);
  assert.equal(gemma4VisionPixelLut()[255], 1);
});

test("internal GPU preparation can borrow worker RGBA and leaves wiping to the request owner", () => {
  const cfg = { patch: 16, pool: 3 };
  const width = 32, height = 24;
  const backing = new Uint8Array(width * height * 4 + 8);
  const data = new Uint8ClampedArray(backing.buffer, 4, width * height * 4);
  data.set(Uint8Array.from({ length: data.length }, (_, i) => (i * 19 + 7) & 255));
  const input = { width, height, data };
  const handle = prepareGemma4GpuVisionImage(input, 16, {}, { borrowInput: true });
  const owned = consumeGemma4VisionImage(handle, cfg);
  assert.strictEqual(owned.raw.rgba.buffer, data.buffer);
  assert.equal(owned.raw.rgba.byteOffset, data.byteOffset);
  assert.equal(owned.raw.borrowedInput, true);
  const before = new Uint8Array(data);
  clearGemma4VisionImageData(owned);
  assert.deepEqual(new Uint8Array(data), before);
  data.fill(0); // Simulate engine-worker's requestQueue release hook.
  assert.ok(data.every((value) => value === 0));

  const fallbackData = new Uint8Array(8192 * 4).fill(123);
  const fallback = prepareGemma4GpuVisionImage({ width: 8192, height: 1, data: fallbackData }, 1, {}, { borrowInput: true });
  const cpu = consumeGemma4VisionImage(fallback, cfg);
  assert.equal(cpu.raw, undefined);
  assert.ok(fallbackData.some((value) => value !== 0));
  clearGemma4VisionImageData(cpu);
  assert.ok(fallbackData.some((value) => value !== 0));
});

test("private GPU preparation falls back for excessive bicubic taps and disposes once", () => {
  const cfg = { patch: 16, pool: 3 };
  const source = { width: 8192, height: 1, data: new Uint8Array(8192 * 4).fill(123) };
  const owned = consumeGemma4VisionImage(prepareGemma4GpuVisionImage(source, 1), cfg);
  assert.ok(owned.pixels instanceof Float32Array);
  assert.equal(owned.raw, undefined);
  clearGemma4VisionImageData(owned);
  const pending = prepareGemma4GpuVisionImage({ width: 1, height: 1, data: Uint8Array.of(255, 1, 2, 3) }, 1);
  disposeGemma4VisionImage(pending);
  assert.throws(() => consumeGemma4VisionImage(pending, cfg), /already been consumed or disposed/);
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
