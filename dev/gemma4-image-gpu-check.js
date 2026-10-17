// Exact DEV-only image-preprocessing probe. It does not load a model or touch production state.
import { prepareGemma4Image, prepareGemma4VisionImage, disposeGemma4VisionImage } from "../js/src/gemma4-image.js";
import { pipeline, requestDevice } from "../js/src/gpu.js";
import { visionImagePlan } from "../js/src/gpu-gemma4-vision.js";
import { kernelSource } from "./wgsl.js";

const U = globalThis.GPUBufferUsage;
const Q22 = 2 ** 22;
const HALF_Q22 = Q22 / 2;
const I32_MAX = 0x7fffffff;
const MAX_TAPS = 32;
const MAX_COEFFICIENTS = 65_536;
const MAX_CASE_BYTES = 16 * 1024 * 1024;
const PATCH = 16;
const POOL = 3;
const PATCH_WIDTH = 3 * PATCH * PATCH;
const TAIL = Float32Array.of(12345.25, -1731.5, 913.75, -0);
const IMAGE_CONFIG = { patch_size: PATCH, pooling_kernel_size: POOL };
const PLAN_CONFIG = { patch: PATCH, pool: POOL, positionSize: 8192, headDim: 64, theta: 10000 };
const search = new URLSearchParams(location.search);
const state = window.gpuBench = {
  status: "running", done: false, backend: "webgpu", check: "gemma4-image-gpu-check",
  phase: "initializing", completed: 0, cases: [],
};

const logNode = document.getElementById("log");
const runButton = document.getElementById("run");
const log = (value) => { logNode.textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const percentile = (values, p) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)];
};
const align4 = (size) => (size + 3) & ~3;

function bicubic(x) {
  x = Math.abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}

// Mirrors the Q22 coefficient arithmetic in js/src/gemma4-image.js.
function coefficientTable(input, output, axis) {
  const metadata = new Uint32Array(output * 4);
  const flattened = [];
  let maximumAbsoluteAccumulator = 0;
  let maximumTaps = 0;
  for (let i = 0; i < output; i++) {
    const scale = input / output;
    const filter = Math.max(scale, 1);
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - 2 * filter + 0.5));
    const end = Math.min(input, Math.trunc(center + 2 * filter + 0.5));
    const weights = Array.from({ length: end - start }, (_, j) => bicubic((start + j - center + 0.5) / filter));
    const sum = weights.reduce((a, b) => a + b, 0);
    const quantized = weights.map((weight) => {
      weight /= sum;
      return Math.trunc(weight * Q22 + (weight < 0 ? -0.5 : 0.5));
    });
    if (!quantized.length || quantized.length > MAX_TAPS) throw new Error(`${axis} row ${i} has unsupported tap count ${quantized.length}`);
    const sumAbs = quantized.reduce((total, weight) => total + Math.abs(weight), 0);
    const accumulatorBound = HALF_Q22 + 255 * sumAbs;
    if (!Number.isSafeInteger(accumulatorBound) || accumulatorBound > I32_MAX || quantized.some((weight) => !Number.isSafeInteger(weight) || Math.abs(weight) > I32_MAX)) {
      throw new Error(`${axis} row ${i} may overflow signed i32: absolute accumulator bound ${accumulatorBound}`);
    }
    maximumAbsoluteAccumulator = Math.max(maximumAbsoluteAccumulator, accumulatorBound);
    maximumTaps = Math.max(maximumTaps, quantized.length);
    metadata.set([start, quantized.length, flattened.length, 0], i * 4);
    flattened.push(...quantized);
  }
  if (flattened.length > MAX_COEFFICIENTS) throw new Error(`${axis} coefficient table has ${flattened.length} weights, above ${MAX_COEFFICIENTS}`);
  const weights = Int32Array.from(flattened);
  return { metadata, weights, maximumAbsoluteAccumulator, maximumTaps };
}

function seededFixture({ id, width, height, seed, pattern }) {
  const data = new Uint8Array(width * height * 4);
  let state = seed >>> 0;
  const randomByte = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state & 255;
  };
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const at = (y * width + x) * 4;
    if (pattern === "gradient") {
      data[at] = (x * 13 + y * 3 + randomByte()) & 255;
      data[at + 1] = (y * 17 + x * 5 + randomByte()) & 255;
      data[at + 2] = ((x ^ y) * 11 + randomByte()) & 255;
    } else if (pattern === "checker") {
      const base = ((x >> 3) + (y >> 3)) & 1 ? 255 : 0;
      data[at] = base ^ (randomByte() & 31);
      data[at + 1] = (base + x * 7 + randomByte()) & 255;
      data[at + 2] = (255 - base + y * 9 + randomByte()) & 255;
    } else {
      data[at] = randomByte();
      data[at + 1] = randomByte();
      data[at + 2] = randomByte();
    }
    data[at + 3] = (x + y) % 5 === 0 ? 0 : (x * 19 + y * 23 + seed) & 255;
  }
  if (pattern === "corners") {
    const points = [[0, 0, 255, 0, 0], [width - 1, 0, 0, 255, 0], [0, height - 1, 0, 0, 255], [width - 1, height - 1, 255, 255, 255]];
    for (const [x, y, red, green, blue] of points) {
      const at = (y * width + x) * 4;
      data.set([red, green, blue, 255], at);
    }
  }
  return { id, width, height, data };
}

function fixtureCases(includeEdges) {
  const base = [
    { id: "camera-a", width: 320, height: 240, seed: 0x4a17c3, pattern: "gradient" },
    { id: "camera-b", width: 320, height: 240, seed: 0x1d39b7, pattern: "checker" },
    { id: "wide-odd", width: 321, height: 239, seed: 0x6f24a1, pattern: "seeded" },
    { id: "portrait", width: 137, height: 301, seed: 0x2c9e51, pattern: "gradient" },
  ];
  if (includeEdges) base.push(
    { id: "single-pixel", width: 1, height: 1, seed: 0x12, pattern: "seeded" },
    { id: "tiny-odd-alpha", width: 17, height: 13, seed: 0x19a0, pattern: "checker" },
    { id: "corner-impulses", width: 33, height: 29, seed: 0x77a1, pattern: "corners" },
  );
  return base;
}

function verifyGeometry(prepared, plan) {
  const columns = prepared.width / PATCH;
  const rows = prepared.height / PATCH;
  const patches = columns * rows;
  if (!Number.isInteger(columns) || !Number.isInteger(rows) || columns % POOL || rows % POOL) throw new Error("processor output grid is not divisible into 3x3 pooling blocks");
  if (plan.rows !== patches || plan.softRows !== patches / (POOL * POOL) || plan.pixels.length !== patches * PATCH_WIDTH) throw new Error("visionImagePlan returned inconsistent compact image dimensions");
  if (prepared.positions.length / 2 < patches || plan.positions.length !== patches * 2 || plan.pools.length !== patches) throw new Error("visionImagePlan returned inconsistent position or pool arrays");
  for (let patch = 0; patch < prepared.positions.length / 2; patch++) {
    const expectedX = patch < patches ? patch % columns : -1;
    const expectedY = patch < patches ? Math.floor(patch / columns) : -1;
    if (prepared.positions[patch * 2] !== expectedX || prepared.positions[patch * 2 + 1] !== expectedY) throw new Error(`public processor patch geometry differs at patch ${patch}`);
    if (patch < patches && (plan.positions[patch * 2] !== expectedX || plan.positions[patch * 2 + 1] !== expectedY)) throw new Error(`visionImagePlan compact geometry differs at patch ${patch}`);
  }
  for (let poolY = 0; poolY < rows / POOL; poolY++) for (let poolX = 0; poolX < columns / POOL; poolX++) {
    const pool = poolY * (columns / POOL) + poolX;
    for (let dy = 0; dy < POOL; dy++) for (let dx = 0; dx < POOL; dx++) {
      const index = pool * POOL * POOL + dy * POOL + dx;
      const expected = (poolY * POOL + dy) * columns + poolX * POOL + dx;
      if (plan.pools[index] !== expected) throw new Error(`visionImagePlan pool packing differs at pool ${pool}, slot ${index % 9}`);
    }
  }
  return { width: prepared.width, height: prepared.height, patchColumns: columns, patchRows: rows, patches, softRows: plan.softRows, poolIndices: plan.pools.length };
}

function comparePixels(actual, expectedPixels, outputCount, id) {
  if (actual.length !== outputCount + TAIL.length || expectedPixels.length !== outputCount) {
    throw new Error(`${id} has an unexpected pixel or tail length`);
  }
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expectedPixels.buffer, expectedPixels.byteOffset, expectedPixels.length);
  const tailBits = new Uint32Array(TAIL.buffer, TAIL.byteOffset, TAIL.length);
  let differences = 0;
  let firstDifference = null;
  let finiteFailures = 0;
  let rangeFailures = 0;
  for (let index = 0; index < outputCount; index++) {
    if (!Number.isFinite(actual[index])) finiteFailures++;
    if (actual[index] < -1 || actual[index] > 1) rangeFailures++;
    if (actualBits[index] !== expectedBits[index]) {
      differences++;
      firstDifference ||= { index, actual: actual[index], expected: expectedPixels[index], actualBits: actualBits[index], expectedBits: expectedBits[index] };
    }
  }
  for (let tail = 0; tail < TAIL.length; tail++) {
    const index = outputCount + tail;
    if (actualBits[index] !== tailBits[tail]) {
      differences++;
      firstDifference ||= { index, actual: actual[index], expected: TAIL[tail], actualBits: actualBits[index], expectedBits: tailBits[tail], tail: true };
    }
  }
  const result = { id, bitExact: differences === 0, differences, firstDifference, finiteFailures, rangeFailures, checkedPixels: outputCount, checkedTailFloats: TAIL.length };
  if (differences || finiteFailures || rangeFailures) throw new Error(`${id} CPU/GPU pixel check failed: ${JSON.stringify(result)}`);
  return result;
}

function compareCpuPaths(actual, expected, id) {
  if (actual.length !== expected.length) throw new Error(`${id} private CPU path returned ${actual.length} pixels, expected ${expected.length}`);
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < actual.length; index++) {
    if (!Number.isFinite(actual[index]) || actualBits[index] !== expectedBits[index]) {
      throw new Error(`${id} private/public CPU image paths differ at pixel ${index}: ${actual[index]} vs ${expected[index]}`);
    }
  }
  return { bitExact: true, checkedPixels: actual.length };
}

function summaryTiming(samples) {
  if (!samples.length) return null;
  const horizontal = samples.map((sample) => sample.horizontalMs);
  const vertical = samples.map((sample) => sample.verticalMs);
  const combined = samples.map((sample) => sample.twoPassMs);
  return {
    method: "WebGPU timestamp-query around the horizontal and vertical compute passes only; upload, encoding, output copy, and mapping are outside intervals",
    samples: samples.length,
    horizontalMs: { p50: median(horizontal), p95: percentile(horizontal, 0.95), values: horizontal },
    verticalMs: { p50: median(vertical), p95: percentile(vertical, 0.95), values: vertical },
    twoPassMs: { p50: median(combined), p95: percentile(combined, 0.95), values: combined },
  };
}

async function loadKernels(device) {
  const response = await fetch(new URL("./gemma4-image-gpu-check.wgsl", import.meta.url), { cache: "no-store" });
  if (!response.ok) throw new Error(`image resize WGSL HTTP ${response.status}`);
  const code = await response.text();
  if (search.get("registered") === "1") {
    const source = await kernelSource();
    const common = code.slice(0, code.indexOf("@group(0) @binding(1)"));
    const expected = {
      horizontal: code.slice(0, code.indexOf("@group(0) @binding(5)")).replace("fn resize_horizontal(", "fn main("),
      vertical: (common + code.slice(code.indexOf("@group(0) @binding(5)"))).replace("fn resize_vertical_pack(", "fn main("),
    };
    const hashes = {};
    const compiled = {};
    for (const name of ["horizontal", "vertical"]) {
      const registered = source(`gemma4_image_resize_${name}`);
      if (registered.trim() !== expected[name].trim()) throw new Error(`registered ${name} source differs from the exact DEV prototype`);
      hashes[name] = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(registered)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
      compiled[name] = await pipeline(device, registered, `gemma4_image_resize_${name}`);
    }
    return { ...compiled, codeBytes: new TextEncoder().encode(code).byteLength, registered: true, hashes, wasmUrl: source.wasmUrl };
  }
  const module = device.createShaderModule({ label: "Gemma image Q22 bicubic probe", code });
  const info = await module.getCompilationInfo?.();
  const errors = (info?.messages || []).filter((message) => message.type === "error");
  if (errors.length) throw new Error(`image resize WGSL: ${errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join("; ")}`);
  const compile = async (entryPoint, label) => device.createComputePipelineAsync({
    layout: "auto", label, compute: { module, entryPoint },
  });
  const [horizontal, vertical] = await Promise.all([
    compile("resize_horizontal", "gemma4_image_resize_horizontal"),
    compile("resize_vertical_pack", "gemma4_image_resize_vertical_pack"),
  ]);
  return { codeBytes: new TextEncoder().encode(code).byteLength, horizontal, vertical };
}

function bind(device, pipelineValue, entries) {
  return device.createBindGroup({
    layout: pipelineValue.getBindGroupLayout(0),
    entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
  });
}

async function runCase(runtime, kernels, description, budget, timingEnabled) {
  const { device } = runtime;
  const image = seededFixture(description);
  const owned = [];
  let prepared;
  let plan;
  let outputReadbackMapped = false;
  let timingReadbackMapped = false;
  let querySet;
  let cpuPrepareMs = 0;
  let cpuPlanMs = 0;
  let privateCpuMs = 0;
  let privateWipeMs = 0;
  let cpuOracleAgreement;
  let uploadCpuMs = 0;
  let encodeCpuMs = 0;
  let submitCpuMs = 0;
  let allocatedGpuBytes = 0;
  let caseTables;
  let outputReadback;
  let timingReadback;
  try {
    const prepareStarted = performance.now();
    prepared = prepareGemma4Image(image, budget, IMAGE_CONFIG);
    cpuPrepareMs = performance.now() - prepareStarted;
    const planStarted = performance.now();
    plan = visionImagePlan(prepared, { ...PLAN_CONFIG }, true);
    cpuPlanMs = performance.now() - planStarted;
    const geometry = verifyGeometry(prepared, plan);
    let privatePrepared;
    let privatePlan;
    const privateStarted = performance.now();
    try {
      privatePrepared = prepareGemma4VisionImage(image, budget, IMAGE_CONFIG);
      privatePlan = visionImagePlan(privatePrepared, { ...PLAN_CONFIG }, true);
      privateCpuMs = performance.now() - privateStarted;
      cpuOracleAgreement = compareCpuPaths(privatePlan.pixels, plan.pixels, description.id);
    } finally {
      const wipeStarted = performance.now();
      disposeGemma4VisionImage(privatePrepared);
      privatePlan?.pixels.fill(0); privatePlan?.positions.fill(0); privatePlan?.rope.fill(0); privatePlan?.pools.fill(0);
      privateWipeMs = performance.now() - wipeStarted;
    }
    const x = coefficientTable(image.width, prepared.width, "horizontal");
    const y = coefficientTable(image.height, prepared.height, "vertical");
    const coefficientCount = x.weights.length + y.weights.length;
    if (coefficientCount > MAX_COEFFICIENTS) throw new Error(`combined coefficient table has ${coefficientCount} weights, above ${MAX_COEFFICIENTS}`);
    caseTables = { x, y, lut: Float32Array.from({ length: 256 }, (_, value) => Math.fround(2 * (Math.fround(value / 255) - 0.5))) };
    const outputCount = plan.pixels.length;
    const horizontalByteCount = prepared.width * image.height * 3;
    const outputBytes = (outputCount + TAIL.length) * 4;
    const readbackBytes = outputBytes;
    const timestampFeature = timingEnabled && device.features.has("timestamp-query");
    const sampleCount = timestampFeature ? 5 : 0;
    const querySetBytes = timestampFeature ? sampleCount * 4 * 8 : 0;
    const paramsData = new Uint32Array([
      image.width, image.height, prepared.width, prepared.height,
      outputCount, geometry.patchColumns, horizontalByteCount, 0,
    ]);
    const plannedGpuBytes = [
      image.data.byteLength,
      x.metadata.byteLength, x.weights.byteLength, y.metadata.byteLength, y.weights.byteLength,
      caseTables.lut.byteLength, paramsData.byteLength,
      outputBytes, readbackBytes,
      align4(horizontalByteCount),
      timestampFeature ? 2 * sampleCount * 4 * 8 + querySetBytes : 0,
    ].reduce((sum, value) => sum + value, 0);
    const geometryHostEstimate = 2 * (plan.positions.byteLength + plan.rope.byteLength + plan.pools.byteLength) + prepared.positions.byteLength;
    const hostBytes = image.data.byteLength + prepared.patches.byteLength + prepared.positions.byteLength
      + plan.pixels.byteLength + plan.positions.byteLength + plan.rope.byteLength + plan.pools.byteLength
      + 2 * (plan.pixels.byteLength + plan.positions.byteLength + plan.rope.byteLength + plan.pools.byteLength)
      + geometryHostEstimate + x.metadata.byteLength + x.weights.byteLength + y.metadata.byteLength + y.weights.byteLength
      + caseTables.lut.byteLength + paramsData.byteLength + TAIL.byteLength;
    const cpuHorizontalScratchBytes = prepared.width * image.height * 3;
    const cpuReferencePeakBytes = image.data.byteLength + prepared.patches.byteLength + prepared.positions.byteLength
      + cpuHorizontalScratchBytes + plan.pixels.byteLength + plan.positions.byteLength + 2 * plan.rope.byteLength
      + 2 * plan.pools.byteLength + x.metadata.byteLength + x.weights.byteLength + y.metadata.byteLength + y.weights.byteLength;
    const estimatedPeakBytes = Math.max(cpuReferencePeakBytes, hostBytes + plannedGpuBytes);
    if (estimatedPeakBytes > MAX_CASE_BYTES) throw new Error(`case requires ${estimatedPeakBytes} controlled bytes, above ${MAX_CASE_BYTES}`);
    const storageLimit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
    const make = (bytes, usage, label, storage = false) => {
      const size = Math.max(16, align4(bytes));
      if (!Number.isSafeInteger(size) || size > device.limits.maxBufferSize || storage && size > device.limits.maxStorageBufferBindingSize) {
        throw new Error(`${label} needs ${size} bytes, beyond this adapter's buffer limit ${storageLimit}`);
      }
      const buffer = device.createBuffer({ size, usage, label: `Gemma image probe ${label}` });
      owned.push(buffer);
      allocatedGpuBytes += size;
      return buffer;
    };
    const src = make(image.data.byteLength, U.STORAGE | U.COPY_DST, "source RGBA", true);
    const xMeta = make(x.metadata.byteLength, U.STORAGE | U.COPY_DST, "horizontal coefficient rows", true);
    const xWeights = make(x.weights.byteLength, U.STORAGE | U.COPY_DST, "horizontal Q22 weights", true);
    const yMeta = make(y.metadata.byteLength, U.STORAGE | U.COPY_DST, "vertical coefficient rows", true);
    const yWeights = make(y.weights.byteLength, U.STORAGE | U.COPY_DST, "vertical Q22 weights", true);
    const lut = make(caseTables.lut.byteLength, U.STORAGE | U.COPY_DST, "normalization LUT", true);
    const params = make(paramsData.byteLength, U.UNIFORM | U.COPY_DST, "dimensions");
    const horizontalBytes = align4(horizontalByteCount);
    const horizontal = make(horizontalBytes, U.STORAGE | U.COPY_DST, "packed RGB intermediate", true);
    const output = make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "compact normalized patches", true);
    outputReadback = make(readbackBytes, U.COPY_DST | U.MAP_READ, "pixel readback");
    let resolved;
    if (timestampFeature) {
      querySet = device.createQuerySet({ type: "timestamp", count: sampleCount * 4, label: `Gemma image ${description.id} timestamps` });
      resolved = make(sampleCount * 4 * 8, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp resolve");
      timingReadback = make(sampleCount * 4 * 8, U.COPY_DST | U.MAP_READ, "timestamp readback");
    }
    if (hostBytes + allocatedGpuBytes + querySetBytes > MAX_CASE_BYTES) {
      throw new Error(`case allocated ${hostBytes + allocatedGpuBytes + querySetBytes} controlled bytes, above ${MAX_CASE_BYTES}`);
    }
    const sourceWords = new Uint32Array(image.data.buffer, image.data.byteOffset, image.data.byteLength / 4);
    const littleEndian = new Uint8Array(Uint32Array.of(0x04030201).buffer)[0] === 1;
    if (!littleEndian) throw new Error("the exact RGBA storage view requires a little-endian host");
    const bytesBeforeUpload = performance.now();
    device.queue.writeBuffer(src, 0, sourceWords);
    device.queue.writeBuffer(xMeta, 0, x.metadata);
    device.queue.writeBuffer(xWeights, 0, x.weights);
    device.queue.writeBuffer(yMeta, 0, y.metadata);
    device.queue.writeBuffer(yWeights, 0, y.weights);
    device.queue.writeBuffer(lut, 0, caseTables.lut);
    device.queue.writeBuffer(params, 0, paramsData);
    device.queue.writeBuffer(output, outputCount * 4, TAIL);
    uploadCpuMs = performance.now() - bytesBeforeUpload;

    const groupsHorizontal = bind(device, kernels.horizontal, [[0, params], [1, src], [2, xMeta], [3, xWeights], [4, horizontal]]);
    const groupsVertical = bind(device, kernels.vertical, [[0, params], [5, yMeta], [6, yWeights], [7, horizontal], [8, lut], [9, output]]);
    const warmups = 2;
    const timedSamples = timestampFeature ? sampleCount : 1;
    const queryBytes = timestampFeature ? sampleCount * 4 * 8 : 0;
    const encodeStarted = performance.now();
    const encoder = device.createCommandEncoder({ label: `Gemma image ${description.id} budget ${budget}` });
    const dispatchHorizontal = (pass) => { pass.setPipeline(kernels.horizontal); pass.setBindGroup(0, groupsHorizontal); pass.dispatchWorkgroups(Math.ceil(Math.ceil(horizontalByteCount / 4) / 128)); };
    const dispatchVertical = (pass) => { pass.setPipeline(kernels.vertical); pass.setBindGroup(0, groupsVertical); pass.dispatchWorkgroups(Math.ceil(outputCount / 128)); };
    for (let i = 0; i < warmups; i++) {
      let pass = encoder.beginComputePass({ label: "Q22 horizontal warmup" });
      dispatchHorizontal(pass); pass.end();
      pass = encoder.beginComputePass({ label: "Q22 vertical warmup" });
      dispatchVertical(pass); pass.end();
    }
    for (let sample = 0; sample < timedSamples; sample++) {
      const horizontalQuery = sample * 4;
      let pass = encoder.beginComputePass(timestampFeature ? {
        label: "Q22 horizontal timed",
        timestampWrites: { querySet, beginningOfPassWriteIndex: horizontalQuery, endOfPassWriteIndex: horizontalQuery + 1 },
      } : { label: "Q22 horizontal exactness" });
      dispatchHorizontal(pass); pass.end();
      pass = encoder.beginComputePass(timestampFeature ? {
        label: "Q22 vertical timed",
        timestampWrites: { querySet, beginningOfPassWriteIndex: horizontalQuery + 2, endOfPassWriteIndex: horizontalQuery + 3 },
      } : { label: "Q22 vertical exactness" });
      dispatchVertical(pass); pass.end();
    }
    // These clears are outside both timed compute passes and keep raw pixels out of reusable buffers.
    encoder.clearBuffer(src);
    encoder.clearBuffer(horizontal);
    encoder.copyBufferToBuffer(output, 0, outputReadback, 0, readbackBytes);
    if (timestampFeature) {
      encoder.resolveQuerySet(querySet, 0, sampleCount * 4, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, timingReadback, 0, queryBytes);
    }
    const commandBuffer = encoder.finish();
    encodeCpuMs = performance.now() - encodeStarted;
    const submitStarted = performance.now();
    device.queue.submit([commandBuffer]);
    submitCpuMs = performance.now() - submitStarted;
    const mapPromises = [outputReadback.mapAsync(GPUMapMode.READ, 0, readbackBytes).then(() => { outputReadbackMapped = true; })];
    if (timestampFeature) mapPromises.push(timingReadback.mapAsync(GPUMapMode.READ, 0, queryBytes).then(() => { timingReadbackMapped = true; }));
    await Promise.all(mapPromises);
    const actual = new Float32Array(outputReadback.getMappedRange(0, readbackBytes));
    const correctness = comparePixels(actual, plan.pixels, outputCount, description.id);
    let timing = null;
    if (timestampFeature) {
      const ticks = new BigUint64Array(timingReadback.getMappedRange(0, queryBytes));
      const samples = [];
      for (let sample = 0; sample < sampleCount; sample++) {
        const at = sample * 4;
        if (ticks[at + 1] <= ticks[at] || ticks[at + 3] <= ticks[at + 2]) throw new Error(`${description.id} has a nonpositive GPU timestamp interval`);
        const horizontalMs = Number(ticks[at + 1] - ticks[at]) / 1e6;
        const verticalMs = Number(ticks[at + 3] - ticks[at + 2]) / 1e6;
        samples.push({ horizontalMs, verticalMs, twoPassMs: horizontalMs + verticalMs });
      }
      timing = summaryTiming(samples);
    }
    const result = {
      id: description.id, seed: description.seed, input: { width: image.width, height: image.height, format: "RGBA8" }, budget,
      geometry, coefficients: {
        count: coefficientCount, maxTaps: Math.max(x.maximumTaps, y.maximumTaps),
        maxAbsoluteAccumulator: Math.max(x.maximumAbsoluteAccumulator, y.maximumAbsoluteAccumulator),
        signedI32Limit: I32_MAX, overflowGuardPassed: true,
      },
      correctness, timing,
      cpu: {
        publicOraclePrepareMs: cpuPrepareMs, publicOraclePlanMs: cpuPlanMs,
        productionPrivatePreparePlanMs: privateCpuMs, productionPrivateWipeMs: privateWipeMs,
        productionPrivateTotalMs: privateCpuMs + privateWipeMs, privateMatchesPublicOracle: cpuOracleAgreement,
        uploadCallsMs: uploadCpuMs, encodeCpuMs, submitCpuMs,
      },
      resourceBounds: { estimatedPeakControlledBytes: estimatedPeakBytes, gpuBufferBytes: allocatedGpuBytes, caseLimitBytes: MAX_CASE_BYTES, outputBytes, horizontalBytes, sourceBytes: image.data.byteLength },
    };
    if (outputReadbackMapped) { outputReadback.unmap(); outputReadbackMapped = false; }
    if (timingReadbackMapped) { timingReadback.unmap(); timingReadbackMapped = false; }
    return result;
  } finally {
    if (outputReadbackMapped) outputReadback?.unmap();
    if (timingReadbackMapped) timingReadback?.unmap();
    querySet?.destroy();
    for (const buffer of owned) buffer.destroy();
    caseTables?.x.metadata.fill(0); caseTables?.x.weights.fill(0);
    caseTables?.y.metadata.fill(0); caseTables?.y.weights.fill(0); caseTables?.lut.fill(0);
    prepared?.patches.fill(0); prepared?.positions.fill(0);
    plan?.pixels.fill(0); plan?.positions.fill(0); plan?.rope.fill(0); plan?.pools.fill(0);
    image.data.fill(0);
  }
}

async function main() {
  if (runButton.disabled) return;
  runButton.disabled = true;
  Object.assign(window.gpuBench, { status: "running", done: false, phase: "initializing", completed: 0, cases: [] });
  const started = performance.now();
  let runtime;
  const cases = [];
  try {
    const timingRequested = search.get("timing") !== "0";
    runtime = await requestDevice({ features: timingRequested ? ["timestamp-query"] : [] });
    const { device, adapter } = runtime;
    const uncaptured = [];
    device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
    state.phase = "compile";
    const compileStarted = performance.now();
    const kernels = await loadKernels(device);
    const compileMs = performance.now() - compileStarted;
    const budgets = search.has("budget") ? [Number(search.get("budget"))] : [16, 40, 70];
    if (budgets.some((budget) => ![16, 40, 70].includes(budget))) throw new Error("budget must be 16, 40, or 70");
    const descriptions = fixtureCases(search.get("edges") === "1").filter((fixture) => !search.has("case") || fixture.id === search.get("case"));
    if (!descriptions.length) throw new Error(`no fixture named ${search.get("case")}`);
    state.phase = "preprocess";
    state.adapter = {
      name: runtime.name,
      vendor: adapter.info?.vendor || null,
      architecture: adapter.info?.architecture || null,
      device: adapter.info?.device || null,
      features: [...device.features].sort(),
      timestampQuery: device.features.has("timestamp-query"),
      limits: { maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage },
    };
    state.compile = { ok: true, ms: compileMs, sourceBytes: kernels.codeBytes, passes: ["resize_horizontal", "resize_vertical_pack"], registered: kernels.registered === true, hashes: kernels.hashes, wasmUrl: kernels.wasmUrl };
    state.timingMethod = device.features.has("timestamp-query") && timingRequested
      ? "GPU timestamps around only the two resize passes"
      : "unavailable: adapter has no timestamp-query feature; no wall-clock GPU estimate used";
    log({ adapter: state.adapter, compile: state.compile, timing: state.timingMethod });
    const ordered = budgets.flatMap((budget) => descriptions.map((fixture) => ({ fixture, budget })));
    for (const [index, item] of ordered.entries()) {
      state.phase = `case ${index + 1}/${ordered.length}`;
      const result = await runCase(runtime, kernels, item.fixture, item.budget, timingRequested);
      cases.push(result);
      state.completed = cases.length;
      state.cases = cases;
      log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    const checkedPixels = cases.reduce((sum, result) => sum + result.correctness.checkedPixels, 0);
    const maxCaseBytes = Math.max(...cases.map((result) => result.resourceBounds.estimatedPeakControlledBytes));
    const timed = cases.map((result) => result.timing?.twoPassMs?.p50).filter((value) => Number.isFinite(value) && value > 0);
    const elapsedMs = performance.now() - started;
    Object.assign(window.gpuBench, {
      status: "done", done: true, phase: "complete", elapsedMs,
      metricMs: timed.length === cases.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value), 0) / timed.length) : elapsedMs,
      method: timed.length === cases.length ? "geometric mean of per-case median GPU preprocessing milliseconds" : "numerical guard wall time; GPU timestamps unavailable",
      backend: "webgpu", check: "gemma4-image-gpu-check", adapter: state.adapter, compile: state.compile,
      timingMethod: state.timingMethod,
      resourceBounds: { maxCaseBytes, limitBytes: MAX_CASE_BYTES, sequentialCases: true },
      correctness: { ok: true, bitExact: true, tolerance: 0, checkedPixels, cases: cases.length }, cases,
    });
    log({ done: true, metricMs: window.gpuBench.metricMs, checkedPixels, cases: cases.length, maxCaseBytes });
  } catch (error) {
    Object.assign(window.gpuBench, { status: "error", done: true, error: String(error?.message || error), cases });
    log(error.stack || String(error));
  } finally {
    runtime?.device.destroy();
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", main);
if (search.get("autorun") === "1") main();
