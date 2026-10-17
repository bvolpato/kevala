import test from "node:test";
import assert from "node:assert/strict";
import { imagePixels } from "../js/src/images.js";

const png = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGP4L+zOoDr1ZAMADpUDXbTx1gEAAAAASUVORK5CYII=";
const webp = "UklGRiIAAABXRUJQVlA4TBYAAAAvAQAAEA/wCcGb/P9IcABDGhWI6H8A";
const opaquePng = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGP4L+yuOvUkAAneAt1TVgZ6AAAAAElFTkSuQmCC";
const opaqueWebp = "UklGRj4AAABXRUJQVlA4IDIAAAAwAgCdASoCAAEAAMASJZACdLoB+AADIQb97AD+3hZSpSXf/gfl04Y8P/nEsdTu5jAAAA==";
const opaqueLosslessWebp = "UklGRiIAAABXRUJQVlA4TBYAAAAvAQAAAA/wCcGb/P9IcP5DCxWI6H8A";
const bytes = (base64) => new Uint8Array(Buffer.from(base64, "base64"));

function browserDecoder(t) {
  let calls = 0;
  const pixels = { width: 2, height: 1, data: new Uint8ClampedArray([255, 19, 71, 255, 37, 149, 201, 255]) };
  for (const [name, value] of Object.entries({
    createImageBitmap: async () => { calls++; return { width: 2, height: 1, close() {} }; },
    OffscreenCanvas: class {
      getContext() { return { drawImage() {}, getImageData: () => pixels, clearRect() {} }; }
    },
  })) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    t.after(() => original ? Object.defineProperty(globalThis, name, original) : delete globalThis[name]);
  }
  return { pixels, calls: () => calls };
}

function largeHeaders() {
  const pngHeader = bytes(png).slice(0, 33);
  const pv = new DataView(pngHeader.buffer);
  pv.setUint32(16, 65535); pv.setUint32(20, 65535); pngHeader[25] = 2;
  const jpegHeader = new Uint8Array([255, 216, 255, 192, 0, 11, 8, 255, 255, 255, 255, 1, 1, 17, 0]);
  const webpHeader = new Uint8Array(30);
  webpHeader.set(new TextEncoder().encode("RIFF"));
  const wv = new DataView(webpHeader.buffer); wv.setUint32(4, 22, true);
  webpHeader.set(new TextEncoder().encode("WEBPVP8X"), 8); wv.setUint32(16, 10, true);
  webpHeader.fill(255, 24, 30);
  return [pngHeader, jpegHeader, webpHeader];
}

test("encoded PNG, JPEG and WebP dimensions are rejected before bitmap allocation", async (t) => {
  const decoder = browserDecoder(t);
  for (const data of largeHeaders()) await assert.rejects(imagePixels(data), /image dimensions exceed/);
  assert.equal(decoder.calls(), 0);
});

test("transparent encoded PNG and WebP are refused before losing hidden or partial-alpha RGB", async (t) => {
  const decoder = browserDecoder(t);
  for (const data of [bytes(png), bytes(webp)]) {
    await assert.rejects(imagePixels(data), /alpha-bearing encoded images.*ImageData.*RGBA/);
  }
  assert.equal(decoder.calls(), 0);
  const rgba = { width: 2, height: 1, data: new Uint8ClampedArray([255, 19, 71, 0, 37, 149, 201, 128]) };
  assert.deepEqual(await imagePixels(rgba), rgba);
});

test("RGBA image decoding keeps its copy contract and exposes an explicit borrow path", async () => {
  const rgba = { width: 2, height: 1, data: new Uint8ClampedArray([1, 2, 3, 4, 5, 6, 7, 8]) };
  const copied = await imagePixels(rgba);
  assert.notStrictEqual(copied.data, rgba.data);
  copied.data[0] = 99;
  assert.equal(rgba.data[0], 1);

  const borrowed = await imagePixels(rgba, { borrowRGBA: true });
  assert.strictEqual(borrowed.data, rgba.data);
});

test("truncated and uninspectable encoded headers never reach a browser decoder", async (t) => {
  const decoder = browserDecoder(t);
  for (const data of [bytes(png).slice(0, 20), bytes(webp).slice(0, 19), new Uint8Array([255, 216, 255, 192]), new TextEncoder().encode('<svg width="100000" height="100000"/>')]) {
    await assert.rejects(imagePixels(data), /encoded image|image header/);
  }
  assert.equal(decoder.calls(), 0);
});

test("opaque PNG, JPEG and WebP still decode with inspected MIME types", async (t) => {
  const decoder = browserDecoder(t);
  // A one-component JPEG header through SOS, enough for the metadata inspector. The browser
  // decoder is stubbed here; pixel decoding remains the platform's responsibility.
  const jpeg = new Uint8Array([255, 216, 255, 192, 0, 11, 8, 0, 1, 0, 2, 1, 1, 17, 0, 255, 218, 0, 8, 1, 1, 0, 0, 63, 0, 255, 217]);
  const decode = globalThis.createImageBitmap;
  const types = [];
  globalThis.createImageBitmap = async (blob, options) => {
    types.push(blob.type);
    assert.deepEqual(options, { colorSpaceConversion: "none" });
    return decode(blob, options);
  };
  for (const data of [new Blob([bytes(opaquePng)]), jpeg, bytes(opaqueWebp)]) assert.deepEqual(await imagePixels(data), decoder.pixels);
  assert.deepEqual(types, ["image/png", "image/jpeg", "image/webp"]);
  assert.equal(decoder.calls(), 3);
  await assert.rejects(imagePixels(jpeg.slice(0, 20)), /truncated encoded image header/);
  assert.equal(decoder.calls(), 3);
});

test("PNG transparency metadata and malformed WebP sizes are rejected before decoding", async (t) => {
  const decoder = browserDecoder(t);
  const source = bytes(opaquePng);
  const transparent = new Uint8Array(source.length + 18);
  transparent.set(source.subarray(0, 33));
  new DataView(transparent.buffer).setUint32(33, 6);
  transparent.set(new TextEncoder().encode("tRNS"), 37);
  transparent.set(source.subarray(33), 51);
  await assert.rejects(imagePixels(transparent), /alpha-bearing encoded images/);
  const malformed = bytes(opaqueWebp);
  new DataView(malformed.buffer).setUint32(16, 0xffffffff, true);
  await assert.rejects(imagePixels(malformed), /truncated encoded image header/);
  assert.equal(decoder.calls(), 0);
});

test("opaque lossless WebP remains supported after checking decoded alpha", async (t) => {
  const decoder = browserDecoder(t);
  assert.deepEqual(await imagePixels(bytes(opaqueLosslessWebp)), decoder.pixels);
  assert.equal(decoder.calls(), 1);
});

test("a cleared WebP lossless alpha hint does not admit hidden or partial-alpha RGB through canvas", async (t) => {
  const decoder = browserDecoder(t);
  const concealedAlpha = bytes(webp);
  concealedAlpha[24] &= 239; // Clear alpha_is_used without changing the alpha-bearing bitstream.
  for (const alpha of [0, 128]) {
    decoder.pixels.data[3] = alpha;
    await assert.rejects(imagePixels(concealedAlpha), /decoded image contains transparency.*RGBA/);
  }
  assert.equal(decoder.calls(), 2);
});

test("decoded transparency is refused even when encoded metadata claimed opaque pixels", async (t) => {
  const decoder = browserDecoder(t);
  decoder.pixels.data[3] = 0;
  await assert.rejects(imagePixels(bytes(opaqueWebp)), /decoded image contains transparency.*RGBA/);
  assert.equal(decoder.calls(), 1);
});
