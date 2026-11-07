// Gemma 4's pinned PIL fallback: bicubic U8 RGB resize, row-major patches, then 1/255.
// https://github.com/huggingface/transformers/blob/2c4914fb939fe9de0d8e7a798af4684d552f18b4/src/transformers/models/gemma4/image_processing_pil_gemma4.py
const PRECISION = 2 ** 22;
const NORMALIZED_U8 = Float32Array.from({ length: 256 }, (_, value) => value / 255);
const VISION_U8 = Float32Array.from(NORMALIZED_U8, (value) => 2 * (value - 0.5));
const preparedImages = new WeakMap();
const preparedHandles = new WeakSet();
const COEFFICIENT_CACHE_LIMIT = 8;
const COEFFICIENT_WEIGHT_LIMIT = 65_536;
const coefficientCache = new Map();
const packedCoefficientCache = new WeakMap();
let cachedCoefficientWeights = 0;
const bicubic = (x) => {
  x = Math.abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
};

function coefficients(input, output) {
  const key = `${input}:${output}`;
  const cached = coefficientCache.get(key);
  if (cached) {
    coefficientCache.delete(key);
    coefficientCache.set(key, cached);
    return cached.values;
  }

  const scale = input / output;
  const filter = Math.max(scale, 1);
  const values = Array.from({ length: output }, (_, i) => {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - 2 * filter + 0.5));
    const end = Math.min(input, Math.trunc(center + 2 * filter + 0.5));
    const weights = Array.from({ length: end - start }, (_, j) => bicubic((start + j - center + 0.5) / filter));
    const sum = weights.reduce((a, b) => a + b, 0);
    return { start, weights: weights.map((w) => { w /= sum; return Math.trunc(w * PRECISION + (w < 0 ? -0.5 : 0.5)); }) };
  });
  const weightCount = values.reduce((total, item) => total + item.weights.length, 0);
  if (weightCount <= COEFFICIENT_WEIGHT_LIMIT) {
    while (coefficientCache.size >= COEFFICIENT_CACHE_LIMIT || cachedCoefficientWeights + weightCount > COEFFICIENT_WEIGHT_LIMIT) {
      const oldestKey = coefficientCache.keys().next().value;
      if (oldestKey === undefined) break;
      cachedCoefficientWeights -= coefficientCache.get(oldestKey).weightCount;
      coefficientCache.delete(oldestKey);
    }
    coefficientCache.set(key, { values, weightCount });
    cachedCoefficientWeights += weightCount;
  }
  return values;
}

function resizeRgbToPatches(image, width, height, patch, pooling, budget, normalized = false) {
  const { width: inputWidth, height: inputHeight, data } = image;
  const horizontal = new Uint8Array(width * inputHeight * 3);
  let patches;
  try {
    const xs = coefficients(image.width, width);
    const ys = coefficients(image.height, height);
    const byte = (sum) => Math.max(0, Math.min(255, Math.floor(sum / PRECISION)));
    const inputRowBytes = inputWidth * 4;
    const horizontalRowBytes = width * 3;
    for (let y = 0; y < inputHeight; y++) {
      const sourceRow = y * inputRowBytes;
      const targetRow = y * horizontalRowBytes;
      for (let x = 0; x < width; x++) {
        const { start, weights } = xs[x];
        const sourceStart = sourceRow + start * 4;
        const target = targetRow + x * 3;
        let red = PRECISION / 2;
        let green = PRECISION / 2;
        let blue = PRECISION / 2;
        for (let k = 0, at = sourceStart; k < weights.length; k++, at += 4) {
          const weight = weights[k];
          red += data[at] * weight;
          green += data[at + 1] * weight;
          blue += data[at + 2] * weight;
        }
        horizontal[target] = byte(red);
        horizontal[target + 1] = byte(green);
        horizontal[target + 2] = byte(blue);
      }
    }

    const cols = width / patch;
    const maxPatches = budget * pooling * pooling;
    const patchWidth = patch * patch * 3;
    patches = new Float32Array((normalized ? width * height / (patch * patch) : maxPatches) * patchWidth);
    const values = normalized ? VISION_U8 : NORMALIZED_U8;
    const positions = new Int32Array(maxPatches * 2).fill(-1);
    for (let y = 0; y < height; y++) {
      const { start, weights } = ys[y];
      const sourceStart = start * horizontalRowBytes;
      const patchY = Math.floor(y / patch);
      const patchPixelY = y - patchY * patch;
      for (let x = 0; x < width; x++) {
        const source = sourceStart + x * 3;
        let red = PRECISION / 2;
        let green = PRECISION / 2;
        let blue = PRECISION / 2;
        for (let k = 0, at = source; k < weights.length; k++, at += horizontalRowBytes) {
          const weight = weights[k];
          red += horizontal[at] * weight;
          green += horizontal[at + 1] * weight;
          blue += horizontal[at + 2] * weight;
        }
        const patchX = Math.floor(x / patch);
        const patchPixelX = x - patchX * patch;
        const patchIndex = patchY * cols + patchX;
        if (patchPixelY === 0 && patchPixelX === 0) {
          positions[patchIndex * 2] = patchX;
          positions[patchIndex * 2 + 1] = patchY;
        }
        const target = patchIndex * patchWidth + (patchPixelY * patch + patchPixelX) * 3;
        patches[target] = values[byte(red)];
        patches[target + 1] = values[byte(green)];
        patches[target + 2] = values[byte(blue)];
      }
    }
    return { patches, positions };
  } catch (error) {
    patches?.fill(0);
    throw error;
  } finally { horizontal.fill(0); }
}

function packedCoefficients(input, output) {
  const rows = coefficients(input, output), cached = packedCoefficientCache.get(rows);
  if (cached) return cached;
  const count = rows.reduce((sum, row) => sum + row.weights.length, 0);
  if (count > COEFFICIENT_WEIGHT_LIMIT || rows.some((row) => row.weights.length > 32 || PRECISION / 2 + 255 * row.weights.reduce((sum, weight) => sum + Math.abs(weight), 0) > 0x7fffffff)) return null;
  const metadata = new Uint32Array(output * 4), weights = new Int32Array(count);
  let offset = 0;
  rows.forEach((row, i) => { metadata.set([row.start, row.weights.length, offset, 0], i * 4); weights.set(row.weights, offset); offset += row.weights.length; });
  const result = { metadata, weights };
  packedCoefficientCache.set(rows, result);
  return result;
}

export function gemma4VisionPixelLut() { return VISION_U8.slice(); }

export function clearGemma4VisionImageData(prepared) {
  prepared?.pixels?.fill(0);
  if (prepared?.raw && !prepared.raw.borrowedInput) prepared.raw.rgba.fill(0);
}

/** Materialize an already-consumed owned frame when optional GPU setup is unavailable. */
export function materializeGemma4VisionPixels(prepared) {
  if (prepared.pixels) return prepared.pixels;
  const raw = prepared.raw;
  if (!raw) throw new Error("Gemma 4 vision prepared image has no owned pixels");
  const { patches } = resizeRgbToPatches({ width: raw.inputWidth, height: raw.inputHeight, data: raw.rgba }, raw.width, raw.height, prepared.patch, prepared.pooling, prepared.budget, true);
  prepared.pixels = patches;
  if (!raw.borrowedInput) raw.rgba.fill(0);
  return patches;
}

export function gemma4ImageSize(width, height, budget = 70, patch = 16, pooling = 3) {
  if (!Number.isSafeInteger(budget) || budget < 1 || budget > 1120) throw new Error("Gemma 4 image token budget must be an integer from 1 to 1120");
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 || width > 8192 || height > 8192 || width * height > 16 * 1024 * 1024) throw new Error("invalid image dimensions: maximum 8192px per side and 16 megapixels");
  const unit = patch * pooling;
  const scale = Math.sqrt(budget * unit * unit / (width * height));
  let w = Math.floor(width * scale / unit) * unit;
  let h = Math.floor(height * scale / unit) * unit;
  if (!w) { w = unit; h = Math.min(h, budget * unit); }
  if (!h) { h = unit; w = Math.min(w, budget * unit); }
  return { width: w, height: h };
}

function processorSize(image, budget, config) {
  const patch = config.patch_size ?? 16;
  const pooling = config.pooling_kernel_size ?? 3;
  if (patch !== 16 || pooling !== 3) throw new Error("this Gemma 4 image processor supports 16px patches and 3×3 pooling");
  if (!(image.data instanceof Uint8Array || image.data instanceof Uint8ClampedArray) || image.data.length !== image.width * image.height * 4) throw new Error("image data must contain width × height RGBA bytes");
  const { width, height } = gemma4ImageSize(image.width, image.height, budget, patch, pooling);
  return { width, height, patch, pooling };
}

export function prepareGemma4Image(image, budget = 70, config = {}) {
  const { width, height, patch, pooling } = processorSize(image, budget, config);
  const cols = width / patch;
  const rows = height / patch;
  const { patches, positions } = resizeRgbToPatches(image, width, height, patch, pooling, budget);
  return { patches, positions, softTokens: rows * cols / (pooling * pooling), width, height };
}

/** Internal owned pixels. Public preprocessing retains its padded [0, 1] contract. */
export function prepareGemma4VisionImage(image, budget = 70, config = {}) {
  const { width: inputWidth, height: inputHeight, data } = image;
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray)) throw new Error("image data must contain width × height RGBA bytes");
  const source = { width: inputWidth, height: inputHeight, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
  const { width, height, patch, pooling } = processorSize(source, budget, config);
  const { patches: pixels, positions } = resizeRgbToPatches(source, width, height, patch, pooling, budget, true);
  try {
    const softTokens = width * height / (patch * patch * pooling * pooling);
    const handle = Object.freeze({ width, height, softTokens, paddedRows: positions.length / 2 });
    preparedImages.set(handle, { pixels, positions, softTokens, patch, pooling });
    preparedHandles.add(handle);
    return handle;
  } catch (error) { pixels.fill(0); throw error; }
}

/** Internal raw snapshot. Unsupported geometries retain the owned CPU-pixel path. */
export function prepareGemma4GpuVisionImage(image, budget = 70, config = {}, { borrowInput = false } = {}) {
  const { width: inputWidth, height: inputHeight, data } = image;
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray)) throw new Error("image data must contain width × height RGBA bytes");
  const source = { width: inputWidth, height: inputHeight, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
  const { width, height, patch, pooling } = processorSize(source, budget, config);
  if (source.data.byteLength > 4 * 1024 * 1024) return prepareGemma4VisionImage(source, budget, config);
  const xs = packedCoefficients(inputWidth, width), ys = packedCoefficients(inputHeight, height);
  const extraBytes = Math.max(16, source.data.byteLength) + Math.ceil(width * inputHeight * 3 / 4) * 4 + (xs?.metadata.byteLength || 0) + (xs?.weights.byteLength || 0) + (ys?.metadata.byteLength || 0) + (ys?.weights.byteLength || 0) + 1056;
  if (!xs || !ys || extraBytes > 8 * 1024 * 1024) return prepareGemma4VisionImage(source, budget, config);
  const rgba = borrowInput ? source.data : new Uint8Array(source.data);
  try {
    const positions = new Int32Array(budget * pooling * pooling * 2).fill(-1), cols = width / patch;
    for (let row = 0; row < width * height / (patch * patch); row++) positions.set([row % cols, Math.floor(row / cols)], row * 2);
    const softTokens = width * height / (patch * patch * pooling * pooling);
    const handle = Object.freeze({ width, height, softTokens, paddedRows: positions.length / 2 });
    preparedImages.set(handle, { raw: { rgba, inputWidth, inputHeight, width, height, xs, ys, borrowedInput: borrowInput }, positions, softTokens, patch, pooling, budget });
    preparedHandles.add(handle);
    return handle;
  } catch (error) {
    if (!borrowInput) rgba.fill(0);
    throw error;
  }
}

export function consumeGemma4VisionImage(image, cfg, compact = true) {
  if (!preparedHandles.has(image)) return null;
  const prepared = preparedImages.get(image);
  preparedImages.delete(image);
  if (!prepared) throw new Error("Gemma 4 vision prepared image has already been consumed or disposed");
  try {
    if (!compact || cfg?.patch !== prepared.patch || cfg?.pool !== prepared.pooling) throw new Error("Gemma 4 vision prepared image requires matching compact patch and pooling configuration");
    return prepared;
  } catch (error) {
    clearGemma4VisionImageData(prepared);
    throw error;
  }
}

export function disposeGemma4VisionImage(image) {
  const prepared = preparedImages.get(image);
  preparedImages.delete(image);
  clearGemma4VisionImageData(prepared);
}
