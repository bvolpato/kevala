import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const LIMIT = 64 * MiB;
const CHUNK = MiB;
const TARGET = 256;
const TAIL = 4;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const DEADLINE_MS = 60_000;
const SAMPLES = 3;
const REPETITIONS = 4;
const SCOPES = [
  { scope: "fast", kernel: "matmul_gemma_source", reduceKernel: "reduce_gemma_source", T: 28, splitT: 64, logicalRows: 66, rows: 2, row56: false },
  { scope: "suffix", kernel: "matmul_gemma_batch", reduceKernel: "reduce_gemma_batch", T: 97, splitT: 1, logicalRows: 97, rows: 4, row56: true },
];
const SHAPES = [
  { shape: "gate", N: 6144, K: 1536 },
  { shape: "down", N: 1536, K: 6144 },
  { shape: "q", N: 2048, K: 1536 },
  { shape: "o", N: 1536, K: 2048 },
  { shape: "ragged", N: 65, K: 1056 },
];
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);

export function assertShapeConstantsOnly(dynamicCode, specializedCode, N, K) {
  const marker = "  let N = p.N;\n  let K = p.K;";
  if (dynamicCode.split(marker).length !== 2) throw new Error("dynamic matrix WGSL did not contain exactly one N/K uniform declaration pair");
  const expected = dynamicCode.replace(marker, `  let N = ${N}u;\n  let K = ${K}u;`);
  if (specializedCode !== expected) throw new Error("shape-specialized WGSL changed text outside the N/K declarations");
  return true;
}

async function hash(code) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))]
    .map((value) => value.toString(16).padStart(2, "0")).join("");
}

function plan(shape, scope) {
  const { N, K } = shape;
  const { T, splitT, rows } = scope;
  const splits = mmSplits(splitT, N, K, TARGET, rows * 16, 64);
  const outputCount = T * N;
  const partialCount = splits > 1 ? splits * outputCount : 0;
  const outputBytes = (outputCount + TAIL) * 4;
  const partBytes = (partialCount + TAIL) * 4;
  const inputBytes = T * K * 4;
  const biasBytes = N * 4;
  const immutableWeightBytes = N * K + N * K / 8;
  const timestampBytes = 256;
  const gpuTemporaryBytes = inputBytes + biasBytes + 32 + 2 * outputBytes + 2 * partBytes + 2 * CHUNK + timestampBytes;
  const hostTemporaryBytes = CHUNK + 2 * CHUNK + 4096;
  const temporaryBytes = gpuTemporaryBytes + hostTemporaryBytes;
  return { splits, outputCount, partialCount, outputBytes, partBytes, inputBytes, biasBytes,
    immutableWeightBytes, gpuTemporaryBytes, hostTemporaryBytes, temporaryBytes, limitBytes: LIMIT };
}

async function compileScope(device, source, scope, reduceCache, layouts) {
  const spec = { weights: "q8", f16: false, rows: scope.rows, row56: scope.row56, groups: 1, splitTarget: TARGET };
  const reduceCode = source(scope.reduceKernel, spec);
  const reducePipeline = reduceCache.get(scope.scope) || await pipeline(device, reduceCode, `shaped-matmul.${scope.scope}.reduce`, layouts.reduce);
  reduceCache.set(scope.scope, reducePipeline);
  return { spec, reducePipeline, reducerHash: await hash(reduceCode) };
}

async function compileCase(device, source, scope, shape, compiled, layouts) {
  const { N, K } = shape;
  const dynamicCode = source(scope.kernel, compiled.spec);
  const specializedCode = source(scope.kernel, { ...compiled.spec, n: N, k: K });
  assertShapeConstantsOnly(dynamicCode, specializedCode, N, K);
  const [baseline, candidate] = await Promise.all([
    pipeline(device, dynamicCode, `shaped-matmul.${scope.scope}.${shape.shape}.dynamic`, layouts.matmul),
    pipeline(device, specializedCode, `shaped-matmul.${scope.scope}.${shape.shape}.constants`, layouts.matmul),
  ]);
  return { baseline, candidate, reduce: compiled.reducePipeline,
    sourceHashes: { dynamic: await hash(dynamicCode), specialized: await hash(specializedCode), reducer: compiled.reducerHash } };
}

async function drain(device, deadline, label) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error(`${label} deadline exceeded`);
  let timer;
  try {
    await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} deadline exceeded`)), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

async function uploadGenerated(device, target, byteLength, bytesPerValue, valueAt, stage, deadline, label) {
  const stageBytes = new Uint8Array(stage);
  const stageI8 = new Int8Array(stage);
  const stageF32 = new Float32Array(stage);
  for (let offset = 0; offset < byteLength; offset += CHUNK) {
    const bytes = Math.min(CHUNK, byteLength - offset);
    const count = bytes / bytesPerValue;
    const values = bytesPerValue === 1 ? stageI8 : stageF32;
    for (let index = 0; index < count; index++) values[index] = valueAt(offset / bytesPerValue + index);
    device.queue.writeBuffer(target, offset, stageBytes, 0, bytes);
    await drain(device, deadline, label);
  }
}

async function fillFloats(device, target, count, valueAt, stage, deadline, label) {
  await uploadGenerated(device, target, count * 4, 4, valueAt, stage, deadline, label);
  device.queue.writeBuffer(target, count * 4, new Float32Array(TAIL).fill(SENTINEL));
  await drain(device, deadline, label);
}

async function compareBuffers(device, read, left, right, logicalCount, label, deadline) {
  const total = logicalCount + TAIL;
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  let zeros = 0, negativeZeros = 0, tailGuard = true;
  for (let offset = 0; offset < total; offset += CHUNK / 4) {
    const count = Math.min(CHUNK / 4, total - offset);
    const bytes = count * 4;
    const encoder = device.createCommandEncoder({ label: `${label}.readback` });
    encoder.copyBufferToBuffer(left, offset * 4, read, 0, bytes);
    encoder.copyBufferToBuffer(right, offset * 4, read, bytes, bytes);
    device.queue.submit([encoder.finish()]);
    if (!await mapBeforeDeadline(read, bytes * 2, deadline)) {
      read.unmap();
      throw new Error(`${label} readback deadline exceeded`);
    }
    try {
      const copied = read.getMappedRange(0, bytes * 2).slice(0);
      const values = new Float32Array(copied), bits = new Uint32Array(copied);
      for (let index = 0; index < count; index++) {
        const at = offset + index, second = count + index;
        if (at >= logicalCount) {
          if (bits[index] !== SENTINEL_BITS || bits[second] !== SENTINEL_BITS) tailGuard = false;
          if (bits[index] !== bits[second]) bitMismatches++;
          continue;
        }
        if (!Number.isFinite(values[index]) || !Number.isFinite(values[second])) nonfinite++;
        else maxAbs = Math.max(maxAbs, Math.abs(values[index] - values[second]));
        if (bits[index] !== bits[second]) {
          bitMismatches++;
          firstDifference ||= { index: at, baseline: values[index], candidate: values[second],
            baselineBits: `0x${bits[index].toString(16)}`, candidateBits: `0x${bits[second].toString(16)}` };
        }
        if (values[index] === 0) zeros++;
        if (bits[index] === 0x80000000) negativeZeros++;
      }
    } finally { read.unmap(); }
  }
  return { label, samples: logicalCount, bitMismatches, nonfinite, maxAbs, firstDifference,
    zeros, negativeZeros, tailGuard, ok: bitMismatches === 0 && nonfinite === 0 && tailGuard };
}

async function checkCase(runtime, scope, shape, pipes, layouts) {
  const { device } = runtime, { N, K } = shape, { T, splitT, logicalRows, rows } = scope;
  const memory = plan(shape, scope);
  if (memory.temporaryBytes > LIMIT) throw new Error(`temporary memory exceeds 64 MiB: ${memory.temporaryBytes}`);
  const deviceLimit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if ([N * K, N * K / 8, memory.inputBytes, memory.biasBytes, memory.outputBytes, memory.partBytes, 2 * CHUNK].some((size) => size > deviceLimit)) {
    throw new Error(`${scope.scope}/${shape.shape} buffer exceeds the device binding limit`);
  }
  const deadline = performance.now() + DEADLINE_MS;
  const U = GPUBufferUsage;
  const owned = [];
  let querySet = null;
  let gpuTemporaryBytes = 0;
  const make = (size, usage, label, immutable = false) => {
    const actualSize = Math.max(16, size);
    if (!immutable) gpuTemporaryBytes += actualSize;
    if (gpuTemporaryBytes > memory.gpuTemporaryBytes) throw new Error("actual temporary GPU buffers exceeded the planned bound");
    const buffer = device.createBuffer({ size: actualSize, usage, label: `shaped-matmul.${scope.scope}.${shape.shape}.${label}` });
    owned.push(buffer);
    return buffer;
  };
  try {
    const stage = new ArrayBuffer(CHUNK);
    const weights = make(N * K, U.STORAGE | U.COPY_DST, "weights", true);
    const scales = make(N * K / 8, U.STORAGE | U.COPY_DST, "scales", true);
    const input = make(memory.inputBytes, U.STORAGE | U.COPY_DST, "input");
    const bias = make(memory.biasBytes, U.STORAGE | U.COPY_DST, "bias");
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const baselineY = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "baseline-y");
    const candidateY = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "candidate-y");
    const baselinePart = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "baseline-partials");
    const candidatePart = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "candidate-partials");
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "readback");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");

    await uploadGenerated(device, weights, N * K, 1,
      (index) => (index * 31 + (index >>> 5) * 17 + (index >>> 10) * 7 + 19) % 256 - 128,
      stage, deadline, "weight upload");
    await uploadGenerated(device, scales, N * K / 8, 4,
      (index) => index % 257 === 0 ? -0 : (index % 2 ? -1 : 1) * (0.0002 + index % 11 * 0.000037),
      stage, deadline, "scale upload");
    await uploadGenerated(device, input, memory.inputBytes, 4,
      (index) => index % 67 === 0 ? (index % 2 ? -0 : 0) : 0.04 * Math.sin(index * 0.127 + Math.floor(index / K) * 0.13 + 0.3),
      stage, deadline, "input upload");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, splitT, logicalRows, 0]));

    const bind = (layout, buffers) => device.createBindGroup({
      layout,
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const baselineGroup = bind(layouts.matmul, [globals, params, input, weights, scales, bias, baselineY, baselinePart]);
    const candidateGroup = bind(layouts.matmul, [globals, params, input, weights, scales, bias, candidateY, candidatePart]);
    const baselineReduce = bind(layouts.reduce, [globals, params, baselinePart, bias, baselineY]);
    const candidateReduce = bind(layouts.reduce, [globals, params, candidatePart, bias, candidateY]);
    const splits = memory.splits;
    const dispatch = (pass, pipelineValue, group, reduceGroup) => {
      pass.setPipeline(pipelineValue);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / (rows * 16)), splits);
      if (splits > 1) {
        const work = Math.ceil(T * N / 256);
        pass.setPipeline(pipes.reduce);
        pass.setBindGroup(0, reduceGroup);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const fillPair = async (mode, hasBias) => {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias]));
      await uploadGenerated(device, bias, memory.biasBytes, 4,
        (index) => hasBias ? (index % 7 - 3) * 0.0137 : NaN,
        stage, deadline, "bias upload");
      const initialY = mode === 1
        ? (index) => (index % 23 - 11) * 0.0037
        : () => NaN;
      await fillFloats(device, baselineY, memory.outputCount, initialY, stage, deadline, "baseline output init");
      await fillFloats(device, candidateY, memory.outputCount, initialY, stage, deadline, "candidate output init");
      await fillFloats(device, baselinePart, memory.partialCount, () => NaN, stage, deadline, "baseline partial init");
      await fillFloats(device, candidatePart, memory.partialCount, () => NaN, stage, deadline, "candidate partial init");
    };

    const guards = [];
    for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
      await fillPair(mode, hasBias);
      const encoder = device.createCommandEncoder({ label: `shaped-matmul.${scope.scope}.${shape.shape}.guard` });
      const pass = encoder.beginComputePass();
      dispatch(pass, pipes.baseline, baselineGroup, baselineReduce);
      dispatch(pass, pipes.candidate, candidateGroup, candidateReduce);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await drain(device, deadline, "guard dispatch");
      const output = await compareBuffers(device, read, baselineY, candidateY, memory.outputCount,
        `${scope.scope}.${shape.shape}.mode${mode}.bias${hasBias}.Y`, deadline);
      const partial = await compareBuffers(device, read, baselinePart, candidatePart, memory.partialCount,
        `${scope.scope}.${shape.shape}.mode${mode}.bias${hasBias}.PART`, deadline);
      if (!output.ok || !partial.ok) throw new Error(`bit-exact guard failed: ${JSON.stringify({ output, partial })}`);
      guards.push({ mode, bias: Boolean(hasBias), output, partial });
    }

    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    await uploadGenerated(device, bias, memory.biasBytes, 4, () => NaN, stage, deadline, "timing bias init");
    const warm = device.createCommandEncoder({ label: `shaped-matmul.${scope.scope}.${shape.shape}.warmup` });
    const warmPass = warm.beginComputePass();
    dispatch(warmPass, pipes.baseline, baselineGroup, baselineReduce);
    dispatch(warmPass, pipes.candidate, candidateGroup, candidateReduce);
    warmPass.end();
    device.queue.submit([warm.finish()]);
    await drain(device, deadline, "timing warmup");

    const queryCount = SAMPLES * 2 * 2;
    querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `shaped-matmul.${scope.scope}.${shape.shape}.timestamps` });
    for (let sample = 0; sample < SAMPLES; sample++) {
      const encoder = device.createCommandEncoder({ label: `shaped-matmul.${scope.scope}.${shape.shape}.sample${sample + 1}` });
      const candidateFirst = sample % 2 === 1;
      const paths = candidateFirst ? ["candidate", "baseline"] : ["baseline", "candidate"];
      for (let pathIndex = 0; pathIndex < paths.length; pathIndex++) {
        const path = paths[pathIndex];
        const at = sample * 4 + pathIndex * 2;
        const pass = encoder.beginComputePass({ timestampWrites: {
          querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1,
        } });
        const isCandidate = path === "candidate";
        for (let repetition = 0; repetition < REPETITIONS; repetition++) {
          dispatch(pass, isCandidate ? pipes.candidate : pipes.baseline,
            isCandidate ? candidateGroup : baselineGroup,
            isCandidate ? candidateReduce : baselineReduce);
        }
        pass.end();
      }
      if (sample === SAMPLES - 1) {
        encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
        encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8);
      }
      device.queue.submit([encoder.finish()]);
      await drain(device, deadline, "timing sample");
    }
    if (!await mapBeforeDeadline(read, queryCount * 8, deadline)) {
      read.unmap();
      throw new Error("timestamp readback deadline exceeded");
    }
    const timestamps = new BigUint64Array(read.getMappedRange(0, queryCount * 8).slice(0));
    read.unmap();
    const baselineSamples = [], candidateSamples = [];
    for (let sample = 0; sample < SAMPLES; sample++) {
      const firstCandidate = sample % 2 === 1;
      const baselineIndex = sample * 4 + (firstCandidate ? 2 : 0);
      const candidateIndex = sample * 4 + (firstCandidate ? 0 : 2);
      const elapsed = (index) => {
        if (timestamps[index + 1] <= timestamps[index]) throw new Error(`nonpositive timestamp interval at ${index}`);
        return Number(timestamps[index + 1] - timestamps[index]) / 1e6 / REPETITIONS;
      };
      baselineSamples.push(elapsed(baselineIndex));
      candidateSamples.push(elapsed(candidateIndex));
    }
    const baselineMs = median(baselineSamples), candidateMs = median(candidateSamples);
    if (![...baselineSamples, ...candidateSamples].every((value) => Number.isFinite(value) && value > 0)) {
      throw new Error("nonpositive or nonfinite timestamp sample");
    }
    const timing = { method: "paired WebGPU timestamp-query, one pass per path, matmul plus unchanged reducer",
      samples: SAMPLES, repetitions: REPETITIONS, baselineSamples, candidateSamples,
      baselineMs, candidateMs, speedup: baselineMs / candidateMs };
    const sourceHashes = pipes.sourceHashes;
    return { scope: scope.scope, kernel: scope.kernel, ...shape, T, globals: [T, splitT, logicalRows, 0],
      rows, row56: scope.row56, f16: false, weights: "q8", splits, ok: true, guards, timing, sourceHashes,
      memory: { ...memory, actualGpuTemporaryBytes: gpuTemporaryBytes } };
  } finally {
    querySet?.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

async function main() {
  const started = performance.now();
  const params = new URLSearchParams(location.search);
  const scopes = SCOPES.filter((scope) => !params.has("scope") || scope.scope === params.get("scope"));
  const shapes = SHAPES.filter((shape) => !params.has("shape") || shape.shape === params.get("shape"));
  if (!scopes.length) throw new Error("scope must be fast or suffix");
  if (!shapes.length) throw new Error("shape must be gate, down, q, o, or ragged");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const source = await kernelSource(params.get("wasm") || undefined);
    const device = runtime.device;
    if (!device.features.has("timestamp-query")) throw new Error("timestamp-query is required for paired kernel timing");
    const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
    const layouts = {
      matmul: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      reduce: device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] }),
    };
    const reduceCache = new Map();
    metadata = { adapter: { name: runtime.name, features: [...device.features] }, wasmUrl: source.wasmUrl,
      temporaryLimitBytes: LIMIT, excludedImmutableWeightBytes: true };
    for (const scope of scopes) {
      const compiled = await withGpuErrors(runtime, () => compileScope(device, source, scope, reduceCache, layouts), `building ${scope.scope} reducer`);
      for (const shape of shapes) {
        const pipes = await withGpuErrors(runtime, () => compileCase(device, source, scope, shape, compiled, layouts),
          `building ${scope.scope}/${shape.shape} dynamic and shape-constant kernels`);
        log({ status: "checking", scope: scope.scope, ...shape, globals: [scope.T, scope.splitT, scope.logicalRows, 0],
          memory: plan(shape, scope) });
        const result = await withGpuErrors(runtime, () => checkCase(runtime, scope, shape, pipes, { matmul: layout, reduce: reductionLayout }),
          `${scope.scope}/${shape.shape} exactness and timing`);
        cases.push(result);
        window.gpuBench.completed = cases.length;
        log(result);
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }
    const baselineMetricMs = geometricMean(cases.map((item) => item.timing.baselineMs));
    const metricMs = geometricMean(cases.map((item) => item.timing.candidateMs));
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-shaped-matmul",
      ...metadata, metric: "geometric mean of candidate GPU timestamp milliseconds across cases", metricMs,
      baselineMetricMs, speedup: baselineMetricMs / metricMs, elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-shaped-matmul", completed: 0 };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
      correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || error);
  });
}
