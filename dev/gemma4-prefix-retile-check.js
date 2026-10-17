import { dispatchMatmul, matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, rowsPerThread, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const MAX_SCRATCH = 64 * MiB;
const DEADLINE_MS = 30_000;
const TARGET = 256;
const SAMPLES = 3;
const REPETITIONS = 4;
const SENTINEL = 12345.25;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

async function beforeDeadline(operation, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("prefix retile guard deadline exceeded");
  let timer;
  try {
    return await Promise.race([operation, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("prefix retile guard deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

function exact(expected, actual) {
  if (expected.length !== actual.length) return { ok: false, reason: "length-mismatch" };
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  const b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) {
      bitMismatches++;
      firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] };
    }
  }
  return { ok: bitMismatches === 0 && nonfinite === 0, samples: expected.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

function policy(P, C, kind, f16, balanced) {
  const D = P - C, kernel = kind === "wide" ? "matmul_wide" : "matmul";
  const originalRows = rowsPerThread(P, { f16, groups: 1, kernel });
  const rows = P === 89 ? 4 : 2;
  const row56 = P === 89 && balanced === 56;
  if (row56 && (f16 || kind !== "generic")) throw new Error("BM56 requires generic F32 tiles");
  const splitT = P === 66 ? 64 : P;
  const originalLogicalTiles = Math.ceil(P / (16 * originalRows));
  const candidateLogicalTiles = Math.ceil(splitT / (16 * rows));
  if (originalLogicalTiles !== candidateLogicalTiles) throw new Error("retile changed original logical row tiles");
  return { P, C, D, kind, kernel, f16, originalRows, originalBM: 16 * originalRows, rows, row56,
    physicalBM: row56 ? 56 : 16 * rows, splitT, originalLogicalTiles, candidateLogicalTiles,
    unchanged: originalRows === rows && !row56 };
}

async function pipelines(device, source, layouts, cache, config, rows, row56, compact) {
  const kernel = config.kernel + (compact ? "_gemma_source" : "");
  const reduceKernel = compact ? "reduce_gemma_source" : "reduce";
  const key = `${kernel}:${config.f16}:${rows}:${row56}`;
  if (cache.has(key)) return cache.get(key);
  const spec = { weights: "q8", f16: config.f16, rows, row56, groups: 1, splitTarget: TARGET };
  const mmCode = source(kernel, spec), reduceCode = source(reduceKernel, spec);
  const [mm, reduce, mmHash, reduceHash] = await Promise.all([
    pipeline(device, mmCode, `prefix_retile.${kernel}.r${rows}.${row56}`, layouts.pipeline),
    pipeline(device, reduceCode, `prefix_retile.${reduceKernel}.r${rows}`, layouts.reductionPipeline),
    hash(mmCode), hash(reduceCode),
  ]);
  const value = { mm, reduce, hashes: { mm: mmHash, reduce: reduceHash } };
  cache.set(key, value);
  return value;
}

function scratch({ P, D }, N, K, splits) {
  const fullOutputBytes = (P * N + 4) * 4, outputBytes = (D * N + 4) * 4;
  const fullPartLength = splits > 1 ? splits * P * N : 0;
  const partLength = splits > 1 ? splits * D * N : 0;
  const fullPartBytes = Math.max(16, (fullPartLength + 4) * 4), partBytes = Math.max(16, (partLength + 4) * 4);
  const readBytes = 3 * (outputBytes + partBytes);
  const weightBytes = N * K, scaleBytes = weightBytes / 8, inputBytes = P * K * 4;
  const poisonBytes = Math.max(fullOutputBytes, fullPartBytes);
  const gpuBytes = weightBytes + scaleBytes + inputBytes + fullOutputBytes + 2 * outputBytes + fullPartBytes + 2 * partBytes + readBytes + 5 * 16 + 256;
  const hostBytes = weightBytes + scaleBytes + inputBytes + poisonBytes + readBytes + 4 * MiB;
  return { fullOutputBytes, outputBytes, fullPartLength, partLength, fullPartBytes, partBytes, readBytes, weightBytes, scaleBytes, inputBytes, poisonBytes,
    gpuBytes, hostBytes, totalBytes: gpuBytes + hostBytes, limitBytes: MAX_SCRATCH };
}

async function check(runtime, layouts, pipes, config, { N, K, shape }) {
  const { device } = runtime, { P, C, D, originalRows, rows, physicalBM, splitT } = config;
  const U = GPUBufferUsage, deadline = performance.now() + DEADLINE_MS;
  const splits = mmSplits(P, N, K, TARGET, 16 * originalRows, 64);
  if (splits !== mmSplits(splitT, N, K, TARGET, 16 * rows, 64)) throw new Error("retile changed split-K partition");
  const memory = scratch(config, N, K, splits);
  if (memory.totalBytes > MAX_SCRATCH) throw new Error(`prefix retile scratch exceeds ${MAX_SCRATCH} bytes`);
  if (Math.max(memory.readBytes, memory.weightBytes, memory.inputBytes, memory.fullOutputBytes, memory.fullPartBytes) > device.limits.maxBufferSize || Math.max(memory.weightBytes, memory.inputBytes, memory.fullOutputBytes, memory.fullPartBytes) > device.limits.maxStorageBufferBindingSize) throw new Error("prefix retile case exceeds device buffer limits");
  const inputOffset = C * K * 4;
  if (inputOffset % device.limits.minStorageBufferOffsetAlignment) throw new Error("prefix retile input view is not storage aligned");
  const owned = [];
  let queries;
  const make = (size, usage, label, data) => {
    const buffer = device.createBuffer({ size, usage, label: `prefix-retile.${label}` }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const drain = () => beforeDeadline(device.queue.onSubmittedWorkDone(), deadline);
  const map = async (read, bytes) => {
    if (!await mapBeforeDeadline(read, bytes, deadline)) { read.unmap(); throw new Error("prefix retile readback deadline exceeded"); }
  };
  try {
    const weights = new Int8Array(N * K), scales = new Float32Array(N * K / 32);
    for (let i = 0; i < weights.length; i++) weights[i] = (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) % 256 - 128;
    for (let i = 0; i < scales.length; i++) scales[i] = i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037);
    const inputValues = Float32Array.from({ length: P * K }, (_, i) => Math.fround(0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)) || 0.00390625);
    const w = make(memory.weightBytes, U.STORAGE | U.COPY_DST, "weights");
    for (let offset = 0; offset < weights.byteLength; offset += 4 * MiB) {
      device.queue.writeBuffer(w, offset, weights.buffer, offset, Math.min(4 * MiB, weights.byteLength - offset)); await drain();
    }
    const s = make(memory.scaleBytes, U.STORAGE | U.COPY_DST, "scales", scales); await drain();
    const x = make(memory.inputBytes, U.STORAGE | U.COPY_DST, "input", inputValues); await drain();
    const fullGlobals = make(16, U.UNIFORM | U.COPY_DST, "full-globals", new Uint32Array([P, P, P, 0]));
    const compactGlobals = make(16, U.UNIFORM | U.COPY_DST, "compact-globals", new Uint32Array([D, P, P, 0]));
    const candidateGlobals = make(16, U.UNIFORM | U.COPY_DST, "candidate-globals", new Uint32Array([D, splitT, P, 0]));
    const params = make(16, U.UNIFORM | U.COPY_DST, "params", new Uint32Array([N, K, 0, 0]));
    const unused = make(16, U.STORAGE | U.COPY_DST, "unused", new Float32Array(4).fill(NaN));
    const poison = new Float32Array(memory.poisonBytes / 4);
    const storage = async (size, label) => {
      poison.fill(NaN, 0, size / 4); poison.fill(SENTINEL, size / 4 - 4, size / 4);
      const buffer = make(size, U.STORAGE | U.COPY_SRC | U.COPY_DST, label, poison.subarray(0, size / 4));
      await drain(); return buffer;
    };
    const fullY = await storage(memory.fullOutputBytes, "full-output"), fullPart = await storage(memory.fullPartBytes, "full-partials");
    const compactY = await storage(memory.outputBytes, "compact-output"), compactPart = await storage(memory.partBytes, "compact-partials");
    const candidateY = await storage(memory.outputBytes, "candidate-output"), candidatePart = await storage(memory.partBytes, "candidate-partials");
    const read = make(memory.readBytes, U.MAP_READ | U.COPY_DST, "readback");
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((resource, binding) => ({ binding, resource: resource.buffer ? resource : { buffer: resource } })) });
    const operation = (globals, input, output, part) => ({ N, K,
      group: group(layouts.mm, [globals, params, input, w, s, unused, output, part]),
      reduce: group(layouts.reduce, [globals, params, part, unused, output]),
    });
    const compactX = { buffer: x, offset: inputOffset, size: D * K * 4 };
    const fullOp = operation(fullGlobals, x, fullY, fullPart);
    const compactOp = operation(compactGlobals, compactX, compactY, compactPart);
    const candidateOp = operation(candidateGlobals, compactX, candidateY, candidatePart);
    const dispatch = (pass, kind) => {
      const original = kind === "full", candidate = kind === "candidate";
      const op = original ? fullOp : candidate ? candidateOp : compactOp;
      dispatchMatmul(pass, pipes[kind].mm, pipes[kind].reduce, op, original ? P : D, TARGET, candidate ? rows : originalRows, 1, original ? P : D, candidate ? splitT : P);
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (const kind of ["full", "compact", "candidate"]) dispatch(pass, kind);
    pass.end();
    const stride = memory.outputBytes + memory.partBytes;
    encoder.copyBufferToBuffer(fullY, C * N * 4, read, 0, memory.outputBytes);
    if (splits > 1) {
      for (let split = 0; split < splits; split++) encoder.copyBufferToBuffer(fullPart, (split * P * N + C * N) * 4, read, memory.outputBytes + split * D * N * 4, D * N * 4 + (split === splits - 1 ? 16 : 0));
    } else encoder.copyBufferToBuffer(fullPart, 0, read, memory.outputBytes, 16);
    for (const [index, output, part] of [[1, compactY, compactPart], [2, candidateY, candidatePart]]) {
      encoder.copyBufferToBuffer(output, 0, read, index * stride, memory.outputBytes);
      encoder.copyBufferToBuffer(part, 0, read, index * stride + memory.outputBytes, memory.partBytes);
    }
    device.queue.submit([encoder.finish()]); await map(read, memory.readBytes);
    const values = new Float32Array(read.getMappedRange(0, memory.readBytes).slice(0)); read.unmap();
    const output = (index) => values.subarray(index * stride / 4, index * stride / 4 + D * N);
    const partials = (index) => values.subarray((index * stride + memory.outputBytes) / 4, (index * stride + memory.outputBytes) / 4 + memory.partLength);
    const outputCheck = { compact: exact(output(0), output(1)), candidate: exact(output(0), output(2)) };
    const partialCheck = splits > 1 ? { compact: exact(partials(0), partials(1)), candidate: exact(partials(0), partials(2)) } : null;
    const tails = [0, 1, 2].flatMap((index) => [(index * stride + memory.outputBytes - 16) / 4, ((index + 1) * stride - 16) / 4]);
    const tailGuard = tails.every((start) => values.subarray(start, start + 4).every((value) => value === SENTINEL));
    const ok = Object.values(outputCheck).every((check) => check.ok) && (!partialCheck || Object.values(partialCheck).every((check) => check.ok)) && tailGuard;
    let timing = null;
    if (ok && device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
      const warmup = device.createCommandEncoder(), warmPass = warmup.beginComputePass();
      for (const kind of ["compact", "candidate"]) for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(warmPass, kind);
      warmPass.end(); device.queue.submit([warmup.finish()]); await drain();
      const timed = device.createCommandEncoder(), order = [];
      for (let sample = 0; sample < SAMPLES; sample++) for (const kind of sample % 2 ? ["candidate", "compact"] : ["compact", "candidate"]) {
        const index = order.length * 2; order.push({ kind, sample });
        const pass = timed.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(pass, kind);
        pass.end();
      }
      const timestampBytes = SAMPLES * 4 * 8;
      timed.resolveQuerySet(queries, 0, SAMPLES * 4, resolved, 0);
      timed.copyBufferToBuffer(resolved, 0, read, 0, timestampBytes); device.queue.submit([timed.finish()]);
      await map(read, timestampBytes);
      const stamps = new BigUint64Array(read.getMappedRange(0, timestampBytes)), samples = { compact: [], candidate: [] };
      order.forEach(({ kind }, i) => samples[kind].push(Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6 / REPETITIONS)); read.unmap();
      const choice = selectMatmulKernel({ generic: samples.compact, wide: samples.candidate });
      timing = { ...samples, samples: SAMPLES, repetitions: REPETITIONS, order, compactMedianMs: choice.genericMs, candidateMedianMs: choice.wideMs,
        benefit: choice.benefit, stable: choice.stable, candidateWins: choice.kernel === "wide", reason: choice.reason.replaceAll("generic", "compact").replaceAll("wide", "candidate"), fullOracleTimed: false };
    }
    return { ...config, N, K, shape, splitTarget: TARGET, splits, originalPhysicalTiles: Math.ceil(D / (16 * originalRows)), candidatePhysicalTiles: Math.ceil(D / physicalBM),
      splitBlocks: Math.ceil(K / 32 / splits), ok, outputCheck, partialCheck, tailGuard, timing, memory,
      shaderHashes: Object.fromEntries(Object.entries(pipes).map(([kind, value]) => [kind, value.hashes])) };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const kind = params.get("kernel") || "generic", precision = params.get("precision") || "auto";
  if (!["generic", "wide"].includes(kind)) throw new Error("kernel must be generic or wide");
  if (!["auto", "f32", "f16"].includes(precision)) throw new Error("precision must be auto, f32, or f16");
  const runtime = await requestDevice({ features: precision === "f32" ? ["timestamp-query"] : ["shader-f16", "timestamp-query"] });
  try {
    const { device } = runtime, f16 = device.features.has("shader-f16");
    if (precision === "f16" && !f16) throw new Error("shader-f16 is unavailable");
    const balanced = Number(params.get("balanced") || (!f16 && kind === "generic" ? 56 : 64));
    if (![56, 64].includes(balanced)) throw new Error("balanced must be 56 or 64");
    const scopes = [{ scope: "balanced", P: 89, C: 38 }, { scope: "fast", P: 66, C: 38 }].filter(({ scope }) => !params.has("scope") || scope === params.get("scope"));
    if (!scopes.length) throw new Error("scope must be balanced or fast");
    const shapes = [{ shape: "gate", N: 6144, K: 1536 }, { shape: "down", N: 1536, K: 6144 }, { shape: "q", N: 2048, K: 1536 }, { shape: "o", N: 1536, K: 2048 }, { shape: "k-ple", N: 256, K: 1536 }].filter(({ shape }) => !params.has("shape") || shape === params.get("shape"));
    if (!shapes.length) throw new Error("shape must be gate, down, q, o, or k-ple");
    const source = await kernelSource(), cache = new Map();
    const mm = matmulLayout(device), reduce = reduceLayout(device);
    const layouts = { mm, reduce, pipeline: device.createPipelineLayout({ bindGroupLayouts: [mm] }), reductionPipeline: device.createPipelineLayout({ bindGroupLayouts: [reduce] }) };
    const adapter = { name: runtime.name, isFallbackAdapter: false, features: [...device.features], maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize };
    metadata = { adapter, kernel: kind, precision: f16 ? "f16" : "f32", balanced, wasmUrl: source.wasmUrl };
    log({ ...metadata, scratchLimitBytes: MAX_SCRATCH });
    for (const { scope, P, C } of scopes) {
      const config = { scope, ...policy(P, C, kind, f16, balanced) };
      if (config.unchanged) { const result = { ...config, ok: true, skipped: true, reason: "physical-tile-unchanged" }; cases.push(result); log(result); continue; }
      for (const shape of shapes) {
        log({ status: "checking", ...config, ...shape });
        const result = await withGpuErrors(runtime, async () => {
          const [full, compact, candidate] = await Promise.all([
            pipelines(device, source, layouts, cache, config, config.originalRows, false, false),
            pipelines(device, source, layouts, cache, config, config.originalRows, false, true),
            pipelines(device, source, layouts, cache, config, config.rows, config.row56, true),
          ]);
          return check(runtime, layouts, { full, compact, candidate }, config, shape);
        }, `prefix retile ${scope} ${shape.N}x${shape.K}`);
        cases.push(result); log(result);
        if (!result.ok) throw new Error("compact prefix retile bit-exact parity failed");
        if (result.timing && result.timing.compactMedianMs === null) throw new Error("prefix retile timestamps were invalid");
      }
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMedianMs), 0) / timed.length) : performance.now() - started,
      method: timed.length ? "geometric mean candidate GPU timestamp milliseconds; full P oracle excluded; compact prefix baseline" : "untimed guard wall milliseconds; timestamp-query unavailable or unchanged tile",
      elapsedMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
  log(error.stack || String(error));
});
