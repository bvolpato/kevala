// Gemma 4's pinned PIL fallback: bicubic U8 RGB resize, row-major patches, then 1/255.
// https://github.com/huggingface/transformers/blob/2c4914fb939fe9de0d8e7a798af4684d552f18b4/src/transformers/models/gemma4/image_processing_pil_gemma4.py
const PRECISION = 2 ** 22;
const bicubic = (x) => {
  x = Math.abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
};

function coefficients(input, output) {
  const scale = input / output;
  const filter = Math.max(scale, 1);
  return Array.from({ length: output }, (_, i) => {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - 2 * filter + 0.5));
    const end = Math.min(input, Math.trunc(center + 2 * filter + 0.5));
    const weights = Array.from({ length: end - start }, (_, j) => bicubic((start + j - center + 0.5) / filter));
    const sum = weights.reduce((a, b) => a + b, 0);
    return { start, weights: weights.map((w) => { w /= sum; return Math.trunc(w * PRECISION + (w < 0 ? -0.5 : 0.5)); }) };
  });
}

function resizeRgb(image, width, height) {
  const horizontal = new Uint8Array(width * image.height * 3);
  const xs = coefficients(image.width, width);
  const ys = coefficients(image.height, height);
  const byte = (sum) => Math.max(0, Math.min(255, Math.floor(sum / PRECISION)));
  for (let y = 0; y < image.height; y++) for (let x = 0; x < width; x++) {
    const { start, weights } = xs[x];
    for (let c = 0; c < 3; c++) {
      let sum = PRECISION / 2;
      for (let k = 0; k < weights.length; k++) sum += image.data[(y * image.width + start + k) * 4 + c] * weights[k];
      horizontal[(y * width + x) * 3 + c] = byte(sum);
    }
  }
  const rgb = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const { start, weights } = ys[y];
    for (let c = 0; c < 3; c++) {
      let sum = PRECISION / 2;
      for (let k = 0; k < weights.length; k++) sum += horizontal[((start + k) * width + x) * 3 + c] * weights[k];
      rgb[(y * width + x) * 3 + c] = byte(sum);
    }
  }
  horizontal.fill(0);
  return rgb;
}

export function gemma4ImageSize(width, height, budget = 70, patch = 16, pooling = 3) {
  if (![70, 140, 280, 560, 1120].includes(budget)) throw new Error("Gemma 4 image token budget must be 70, 140, 280, 560, or 1120");
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 || width > 8192 || height > 8192 || width * height > 16 * 1024 * 1024) throw new Error("invalid image dimensions: maximum 8192px per side and 16 megapixels");
  const unit = patch * pooling;
  const scale = Math.sqrt(budget * unit * unit / (width * height));
  let w = Math.floor(width * scale / unit) * unit;
  let h = Math.floor(height * scale / unit) * unit;
  if (!w) { w = unit; h = Math.min(h, budget * unit); }
  if (!h) { h = unit; w = Math.min(w, budget * unit); }
  return { width: w, height: h };
}

export function prepareGemma4Image(image, budget = 70, config = {}) {
  const patch = config.patch_size ?? 16;
  const pooling = config.pooling_kernel_size ?? 3;
  if (patch !== 16 || pooling !== 3) throw new Error("this Gemma 4 image processor supports 16px patches and 3×3 pooling");
  if (!(image.data instanceof Uint8Array || image.data instanceof Uint8ClampedArray) || image.data.length !== image.width * image.height * 4) throw new Error("image data must contain width × height RGBA bytes");
  const { width, height } = gemma4ImageSize(image.width, image.height, budget, patch, pooling);
  const rgb = resizeRgb(image, width, height);
  const cols = width / patch;
  const rows = height / patch;
  const maxPatches = budget * pooling * pooling;
  const patchWidth = patch * patch * 3;
  const patches = new Float32Array(maxPatches * patchWidth);
  const positions = new Int32Array(maxPatches * 2).fill(-1);
  let index = 0;
  for (let py = 0; py < rows; py++) for (let px = 0; px < cols; px++) {
    const p = py * cols + px;
    positions[p * 2] = px;
    positions[p * 2 + 1] = py;
    for (let y = 0; y < patch; y++) for (let x = 0; x < patch; x++) for (let c = 0; c < 3; c++) patches[index++] = rgb[((py * patch + y) * width + px * patch + x) * 3 + c] / 255;
  }
  rgb.fill(0);
  return { patches, positions, softTokens: rows * cols / (pooling * pooling), width, height };
}
