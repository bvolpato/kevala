import { dispatchMatmul, matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, rowsPerThread, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const MAX_SCRATCH = 200 * MiB;
const CASE_DEADLINE_MS = 30_000;
const CHUNK_FLOATS = MiB / 4;
const SENTINEL = 12345.25;
const REPETITIONS = 4;
const SAMPLES = 3;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

async function beforeDeadline(operation, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("weight expansion guard deadline exceeded");
  let timer;
  try {
    return await Promise.race([operation, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("weight expansion guard deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

function exact(a, b) {
  const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const bBits = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (aBits[i] !== bBits[i]) {
      bitMismatches++;
      firstDifference ||= { index: i, q8: a[i], expanded: b[i], q8Bits: aBits[i], expandedBits: bBits[i] };
    }
  }
  return { ok: bitMismatches === 0 && nonfinite === 0, samples: a.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

function guarded(length) {
  const values = new Float32Array(length + 4).fill(NaN);
  values.fill(SENTINEL, length);
  return values;
}

function configuration(T, N, K, kind) {
  const kernel = kind === "wide" ? "matmul_wide" : "matmul";
  const rows = rowsPerThread(T, { f16: false, groups: 1, kernel });
  const row56 = kind === "generic" && rows === 4 && Math.ceil(T / 56) === Math.ceil(T / 64);
  const splitTarget = 256;
  return { kernel, rows, row56, physicalBM: row56 ? 56 : 16 * rows, splitTarget, splits: mmSplits(T, N, K, splitTarget, 16 * rows, 64) };
}

async function pipelines(device, source, cache, config) {
  const key = `${config.kernel}:${config.rows}:${config.row56}`;
  if (cache.has(key)) return cache.get(key);
  const spec = { f16: false, rows: config.rows, row56: config.row56, groups: 1, splitTarget: config.splitTarget };
  const code = { q8: source(config.kernel, { ...spec, weights: "q8" }), expanded: source(config.kernel, { ...spec, weights: "f32" }), reduce: source("reduce", spec) };
  const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const rpl = device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] });
  const [q8, expanded, reduce, hashes] = await Promise.all([
    pipeline(device, code.q8, "weight_expand_q8", pl),
    pipeline(device, code.expanded, "weight_expand_f32", pl),
    pipeline(device, code.reduce, "weight_expand_reduce", rpl),
    Promise.all(Object.entries(code).map(async ([name, value]) => [name, await hash(value)])).then(Object.fromEntries),
  ]);
  const result = { q8, expanded, reduce, layout, reductionLayout, hashes };
  cache.set(key, result);
  return result;
}

async function check(runtime, pipes, { T, N, K, shape }, config) {
  const { device } = runtime, U = GPUBufferUsage;
  const deadline = performance.now() + CASE_DEADLINE_MS;
  const outputLength = T * N, outputBytes = (outputLength + 4) * 4;
  const partLength = config.splits > 1 ? config.splits * outputLength : 0;
  const partBytes = partLength ? (partLength + 4) * 4 : 16;
  const weightBytes = N * K, scaleBytes = weightBytes / 8, expandedBytes = weightBytes * 4, inputBytes = T * K * 4;
  const readBytes = 2 * (outputBytes + (partLength ? partBytes : 0));
  const gpuScratchBytes = weightBytes + scaleBytes + expandedBytes + inputBytes + 2 * outputBytes + 2 * partBytes + readBytes + 3 * 16 + 256;
  // Include live host fixtures, output initialization, copied readback and bounded upload staging.
  const hostScratchEstimate = weightBytes + scaleBytes + inputBytes + MiB + outputBytes + partBytes + readBytes + 24 * MiB;
  const totalScratchEstimate = gpuScratchBytes + hostScratchEstimate;
  if (totalScratchEstimate > MAX_SCRATCH) throw new Error(`weight expansion scratch exceeds ${MAX_SCRATCH} bytes`);
  const largest = Math.max(expandedBytes, partBytes, outputBytes, readBytes);
  if (largest > device.limits.maxBufferSize || Math.max(expandedBytes, partBytes, outputBytes) > device.limits.maxStorageBufferBindingSize) throw new Error("weight expansion case exceeds device buffer limits");
  const owned = [];
  let queries;
  const make = (size, usage, label, data) => {
    const buffer = device.createBuffer({ size, usage, label: `weight-expand.${label}` });
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const drain = () => beforeDeadline(device.queue.onSubmittedWorkDone(), deadline);
  const map = async (read, bytes) => {
    if (!await mapBeforeDeadline(read, bytes, deadline)) { read.unmap(); throw new Error("weight expansion readback deadline exceeded"); }
  };
  try {
    const bytes = new Uint8Array(weightBytes), scales = new Float32Array(scaleBytes / 4);
    for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) & 255;
    for (let i = 0; i < scales.length; i++) scales[i] = i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037);
    const inputs = Float32Array.from({ length: T * K }, (_, i) => Math.fround(0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)) || 0.00390625);
    const w = make(weightBytes, U.STORAGE | U.COPY_DST, "q8");
    const s = make(scaleBytes, U.STORAGE | U.COPY_DST, "scales", scales);
    const expandedW = make(expandedBytes, U.STORAGE | U.COPY_DST, "f32");
    const x = make(inputBytes, U.STORAGE | U.COPY_DST, "input", inputs);
    for (let offset = 0; offset < bytes.length; offset += 4 * MiB) {
      device.queue.writeBuffer(w, offset, bytes.buffer, offset, Math.min(4 * MiB, bytes.length - offset));
      await drain();
    }
    const expandStarted = performance.now(), chunk = new Float32Array(CHUNK_FLOATS);
    let conversionMs = 0;
    for (let offset = 0; offset < bytes.length; offset += chunk.length) {
      const count = Math.min(chunk.length, bytes.length - offset), started = performance.now();
      for (let j = 0; j < count; j++) {
        const index = offset + j, byte = bytes[index];
        chunk[j] = Math.fround((byte < 128 ? byte : byte - 256) * scales[index >>> 5]);
      }
      conversionMs += performance.now() - started;
      device.queue.writeBuffer(expandedW, offset * 4, chunk.buffer, 0, count * 4);
      if ((offset / chunk.length) % 4 === 3) await drain();
    }
    await drain();
    const expansionAndUploadMs = performance.now() - expandStarted;
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals", new Uint32Array([T, 0, 0, 0]));
    const params = make(16, U.UNIFORM | U.COPY_DST, "params", new Uint32Array([N, K, 0, 0]));
    const unused = make(16, U.STORAGE | U.COPY_DST, "unused", new Float32Array(4).fill(NaN));
    const initial = guarded(outputLength);
    const q8Y = make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "q8-output", initial);
    const expandedY = make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "f32-output", initial);
    const partInitial = partLength ? guarded(partLength) : new Float32Array(4);
    const q8Part = make(partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "q8-partials", partInitial);
    const expandedPart = make(partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "f32-partials", partInitial);
    const read = make(readBytes, U.COPY_DST | U.MAP_READ, "readback");
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const op = (weights, weightScales, output, part) => ({ N, K,
      group: group(pipes.layout, [globals, params, x, weights, weightScales, unused, output, part]),
      reduce: group(pipes.reductionLayout, [globals, params, part, unused, output]),
    });
    const operations = { q8: op(w, s, q8Y, q8Part), expanded: op(expandedW, unused, expandedY, expandedPart) };
    const dispatch = (pass, kind) => dispatchMatmul(pass, pipes[kind], pipes.reduce, operations[kind], T, config.splitTarget, config.rows, 1);
    const encoder = device.createCommandEncoder();
    let offset = 0;
    for (const [kind, output, part] of [["q8", q8Y, q8Part], ["expanded", expandedY, expandedPart]]) {
      const pass = encoder.beginComputePass(); dispatch(pass, kind); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, offset, outputBytes); offset += outputBytes;
      if (partLength) { encoder.copyBufferToBuffer(part, 0, read, offset, partBytes); offset += partBytes; }
    }
    device.queue.submit([encoder.finish()]);
    await map(read, readBytes);
    const values = new Float32Array(read.getMappedRange(0, readBytes).slice(0)); read.unmap();
    const stride = (outputBytes + (partLength ? partBytes : 0)) / 4;
    const outputCheck = exact(values.subarray(0, outputLength), values.subarray(stride, stride + outputLength));
    const partialCheck = partLength ? exact(values.subarray(outputBytes / 4, outputBytes / 4 + partLength), values.subarray(stride + outputBytes / 4, stride + outputBytes / 4 + partLength)) : null;
    const tails = [outputLength, stride + outputLength];
    if (partLength) tails.push(outputBytes / 4 + partLength, stride + outputBytes / 4 + partLength);
    const tailGuard = tails.every((start) => values.subarray(start, start + 4).every((value) => value === SENTINEL));
    const ok = outputCheck.ok && (!partialCheck || partialCheck.ok) && tailGuard;
    let timing = null;
    if (ok && device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
      const warmup = device.createCommandEncoder(), warmPass = warmup.beginComputePass();
      for (const kind of ["q8", "expanded"]) for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(warmPass, kind);
      warmPass.end(); device.queue.submit([warmup.finish()]); await drain();
      const timed = device.createCommandEncoder(), order = [];
      for (let sample = 0; sample < SAMPLES; sample++) for (const kind of sample % 2 ? ["expanded", "q8"] : ["q8", "expanded"]) {
        const index = order.length * 2; order.push({ kind, sample });
        const pass = timed.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(pass, kind);
        pass.end();
      }
      const timestampBytes = SAMPLES * 4 * 8;
      timed.resolveQuerySet(queries, 0, SAMPLES * 4, resolved, 0);
      timed.copyBufferToBuffer(resolved, 0, read, 0, timestampBytes); device.queue.submit([timed.finish()]);
      await map(read, timestampBytes);
      const timestamps = new BigUint64Array(read.getMappedRange(0, timestampBytes));
      const samples = { q8: [], expanded: [] };
      order.forEach(({ kind }, i) => samples[kind].push(Number(timestamps[2 * i + 1] - timestamps[2 * i]) / 1e6 / REPETITIONS));
      read.unmap();
      const selected = selectMatmulKernel({ generic: samples.q8, wide: samples.expanded });
      const savedMs = selected.genericMs === null ? null : selected.genericMs - selected.wideMs;
      timing = { ...samples, samples: SAMPLES, repetitions: REPETITIONS, order, q8MedianMs: selected.genericMs, expandedMedianMs: selected.wideMs,
        benefit: selected.benefit, stable: selected.stable, expandedWins: selected.kernel === "wide", reason: selected.reason.replaceAll("generic", "q8").replaceAll("wide", "expanded"),
        breakEvenInvocations: savedMs > 0 ? Math.ceil(expansionAndUploadMs / savedMs) : null };
    }
    return { T, N, K, shape, f16: false, ...config, ok, outputCheck, partialCheck, tailGuard, timing,
      expansion: { conversionMs, expansionAndUploadMs, q8WeightBytes: weightBytes + scaleBytes, expandedWeightBytes: expandedBytes, extraWeightBytes: expandedBytes - weightBytes - scaleBytes, weightByteRatio: expandedBytes / (weightBytes + scaleBytes), chunkBytes: chunk.byteLength },
      memory: { gpuScratchBytes, hostScratchEstimate, totalScratchEstimate, limitBytes: MAX_SCRATCH }, shaderHashes: pipes.hashes };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const kind = params.get("kernel") || "generic";
  if (!["generic", "wide"].includes(kind)) throw new Error("kernel must be generic or wide");
  const shapes = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "gate12288", N: 12288, K: 1536 }, { shape: "down6144", N: 1536, K: 6144 }, { shape: "down12288", N: 1536, K: 12288 }];
  const tokens = params.has("tokens") ? [Number(params.get("tokens"))] : [89, 97];
  if (tokens.some((T) => ![89, 97].includes(T))) throw new Error("tokens must be 89 or 97");
  const selectedShapes = params.has("shape") ? shapes.filter(({ shape }) => shape === params.get("shape")) : shapes;
  if (!selectedShapes.length) throw new Error("shape must be gate6144, gate12288, down6144, or down12288");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const source = await kernelSource(), cache = new Map();
    const adapter = { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features], maxStorageBufferBindingSize: runtime.device.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize };
    metadata = { adapter, kernel: kind, wasmUrl: source.wasmUrl };
    log({ kernel: kind, adapter, wasmUrl: source.wasmUrl, f16: false, scratchLimitBytes: MAX_SCRATCH });
    for (const T of tokens) for (const shape of selectedShapes) {
      const input = { T, ...shape }, config = configuration(T, shape.N, shape.K, kind);
      log({ status: "checking", ...input, ...config });
      const result = await withGpuErrors(runtime, async () => check(runtime, await pipelines(runtime.device, source, cache, config), input, config), `weight expansion T${T} ${shape.N}x${shape.K}`);
      cases.push(result); log(result);
      if (!result.ok) throw new Error("Q8 versus expanded F32 bit-exact parity failed");
      if (result.timing && result.timing.q8MedianMs === null) throw new Error("weight expansion timestamps were invalid");
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", adapter, kernel: kind, wasmUrl: source.wasmUrl,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.expandedMedianMs), 0) / timed.length) : performance.now() - started,
      method: timed.length ? "geometric mean expanded F32 GPU timestamp milliseconds; one-time CPU expansion and upload excluded" : "untimed exactness guard wall milliseconds; timestamp-query unavailable",
      elapsedMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
  log(error.stack || String(error));
});
