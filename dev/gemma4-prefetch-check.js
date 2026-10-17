import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const LIMIT = 64 * MiB;
const CHUNK = MiB;
const TAIL = 4;
const T = 97;
const SPLIT_T = 1;
const ROWS = 4;
const VIRTUAL_BM = 64;
const BN = 64;
const TARGET = 256;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const DEADLINE_MS = 60_000;
const SAMPLES = 3;
const REPETITIONS = 4;
const SHAPES = [
  { shape: "gate", N: 6144, K: 1536, expectedSplits: 3 },
  { shape: "gate12", N: 12288, K: 1536, expectedSplits: 1 },
  { shape: "down", N: 1536, K: 6144, expectedSplits: 8 },
  { shape: "q", N: 2048, K: 1536, expectedSplits: 8 },
  { shape: "q4k", N: 4096, K: 1536, expectedSplits: 4 },
  { shape: "kv256", N: 256, K: 1536, expectedSplits: 8 },
  { shape: "kv512", N: 512, K: 1536, expectedSplits: 8 },
  { shape: "o", N: 1536, K: 2048, expectedSplits: 8 },
  { shape: "o4k", N: 1536, K: 4096, expectedSplits: 8 },
];
const DEFAULT_SHAPES = new Set(["gate", "gate12", "down"]);
const cases = [];
let metadata = {};

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))]
  .map((value) => value.toString(16).padStart(2, "0")).join("");

function count(text, marker) { return text.split(marker).length - 1; }

function indent(text, prefix) {
  return text.split("\n").map((line) => `${prefix}${line}`).join("\n");
}

export function prefetchQ8NextBlock(original) {
  const requirements = [
    ["@compute @workgroup_size(32, 8)", 1],
    ["var<workgroup> xs: array<vec4<f32>, 512>;", 1],
    ["var<workgroup> ws: array<vec4<f32>, 512>;", 1],
    ["for (var kb = kb0; kb < kb1; kb++) {", 1],
    ["if (lr < 56u && xrow < T)", 1],
    ["let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];", 1],
    ["let lo = select(v.xy, v.zw, (lq & 1u) == 1u);", 1],
    ["sx(lo.x) * s", 1],
    ["sx(lo.y) * s", 1],
    ["ws[r * 8u + ((lq * 2u) ^ sw)]", 1],
    ["ws[r * 8u + ((lq * 2u + 1u) ^ sw)]", 1],
  ];
  for (const [marker, expected] of requirements) {
    if (count(original, marker) !== expected) throw new Error(`Q8 prefetch transform marker mismatch (${count(original, marker)}): ${marker}`);
  }
  if (original.includes("enable f16;") || original.includes("prefetchQ8")) {
    throw new Error("Q8 prefetch guard requires the registered F32 kernel and untouched source");
  }
  for (let kq = 0; kq < 8; kq++) {
    if (count(original, `let kq = ${kq}u;`) !== 1) throw new Error(`Q8 prefetch requires the eight explicit K-dot steps; missing kq ${kq}`);
  }

  const loopMarker = "  for (var kb = kb0; kb < kb1; kb++) {\n";
  const loopAt = original.indexOf(loopMarker);
  const firstBarrierMarker = "    workgroupBarrier();";
  const firstBarrierAt = original.indexOf(firstBarrierMarker, loopAt + loopMarker.length);
  const secondBarrierAt = original.indexOf(firstBarrierMarker, firstBarrierAt + firstBarrierMarker.length);
  if (loopAt < 0 || firstBarrierAt < 0 || secondBarrierAt < 0) throw new Error("Q8 prefetch could not locate the unconditional tile barriers");
  if (count(original.slice(loopAt, secondBarrierAt + firstBarrierMarker.length), firstBarrierMarker) !== 2) {
    throw new Error("Q8 prefetch expects exactly two unconditional workgroup barriers around each tile's dot steps");
  }

  const loaderAt = loopAt + loopMarker.length;
  const loader = original.slice(loaderAt, firstBarrierAt);
  if (!loader.includes("xv0 = X[base];") || !loader.includes("xv1 = X[base + 1u];") ||
      !loader.includes("wv[2u * gg] = sx(lo.x) * s;") || !loader.includes("wv[2u * gg + 1u] = sx(lo.y) * s;")) {
    throw new Error("Q8 prefetch found a different production tile loader");
  }

  const dotAt = firstBarrierAt + firstBarrierMarker.length;
  const dotEnd = secondBarrierAt;
  const dotBlock = original.slice(dotAt, dotEnd);
  const steps = [...dotBlock.matchAll(/let kq = (\d+)u;/g)].map((match) => Number(match[1]));
  if (JSON.stringify(steps) !== JSON.stringify([0, 1, 2, 3, 4, 5, 6, 7])) throw new Error("Q8 prefetch dot order differs from the registered kernel");

  const privateValues = `  // Raw next-block values stay in private registers until the current shared tile is consumed.\n  var prefetchX0 = vec4<f32>(0.0);\n  var prefetchX1 = vec4<f32>(0.0);\n  var prefetchW0 = 0u;\n  var prefetchW1 = 0u;\n  var prefetchScale = 0.0;\n`;
  const prefetchLoad = `    // This bound is uniform for the workgroup and leaves the final block untouched.\n    if (kb + 1u < kb1) {\n      let prefetchKb = kb + 1u;\n      prefetchX0 = vec4<f32>(0.0);\n      prefetchX1 = vec4<f32>(0.0);\n      if (lr < 56u && xrow < T) {\n        let prefetchXBase = (xrow * K + prefetchKb * 32u) / 4u + lq * 2u;\n        prefetchX0 = X[prefetchXBase];\n        prefetchX1 = X[prefetchXBase + 1u];\n      }\n      prefetchW0 = 0u;\n      prefetchW1 = 0u;\n      prefetchScale = 0.0;\n      let prefetchWrow = n0 + lr;\n      if (prefetchWrow < N) {\n        let prefetchV = W[(prefetchWrow * K + prefetchKb * 32u) / 16u + lq / 2u];\n        let prefetchLo = select(prefetchV.xy, prefetchV.zw, (lq & 1u) == 1u);\n        prefetchW0 = prefetchLo.x;\n        prefetchW1 = prefetchLo.y;\n        prefetchScale = S[prefetchWrow * nb + prefetchKb];\n      }\n    }`;
  const prefetchStage = `\n    // The prior barrier ends current shared-tile reads before these writes begin.\n    if (kb + 1u < kb1) {\n      if (lr < 56u) {\n        xs[lr * 8u + lq * 2u] = prefetchX0;\n        xs[lr * 8u + lq * 2u + 1u] = prefetchX1;\n      }\n      let prefetchWv0 = sx(prefetchW0) * prefetchScale;\n      let prefetchWv1 = sx(prefetchW1) * prefetchScale;\n      ws[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(prefetchWv0);\n      ws[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(prefetchWv1);\n    }\n`;

  let code = original;
  code = code.slice(0, secondBarrierAt + firstBarrierMarker.length) + prefetchStage + code.slice(secondBarrierAt + firstBarrierMarker.length);

  const transformedLoopAt = code.indexOf(loopMarker);
  const transformedLoaderAt = transformedLoopAt + loopMarker.length;
  const transformedFirstBarrierAt = code.indexOf(firstBarrierMarker, transformedLoaderAt);
  const originalLoader = code.slice(transformedLoaderAt, transformedFirstBarrierAt).trimEnd();
  const wrappedLoader = `    if (kb == kb0) {\n${indent(originalLoader, "  ")}\n    }\n    workgroupBarrier();\n${prefetchLoad}\n`;
  code = code.slice(0, transformedLoaderAt) + wrappedLoader + code.slice(transformedFirstBarrierAt + firstBarrierMarker.length);

  const finalLoopAt = code.indexOf(loopMarker);
  if (finalLoopAt < 0) throw new Error("Q8 prefetch loop disappeared during transformation");
  code = code.slice(0, finalLoopAt) + privateValues + code.slice(finalLoopAt);

  const finalDotAt = code.indexOf("    let nx = lid.x & 7u;", finalLoopAt);
  const finalDotEnd = code.indexOf(firstBarrierMarker, finalDotAt);
  if (finalDotAt < 0 || finalDotEnd < 0 || code.slice(finalDotAt, finalDotEnd).trimStart() !== dotBlock.trimStart()) {
    throw new Error("Q8 prefetch changed the original unrolled dot-step source");
  }
  const loopBody = code.slice(finalLoopAt, code.indexOf("\n  }", finalLoopAt));
  if (count(loopBody, firstBarrierMarker) !== 2 || count(loopBody, "if (kb + 1u < kb1)") !== 2 ||
      count(loopBody, "if (kb == kb0)") !== 1) {
    throw new Error("Q8 prefetch barrier or uniform-bound guard shape changed");
  }
  return code;
}

function memoryPlan(shape, splits) {
  const outputCount = T * shape.N;
  const partialCount = splits > 1 ? splits * outputCount : 0;
  const inputBytes = T * shape.K * 4;
  const outputBytes = (outputCount + TAIL) * 4;
  const partBytes = (partialCount + TAIL) * 4;
  const immutableWeightBytes = shape.N * shape.K;
  const immutableScaleBytes = shape.N * shape.K / 8;
  const gpuTemporaryBuffers = {
    X: inputBytes,
    B: shape.N * 4,
    globals: 16,
    params: 16,
    "baseline-Y": outputBytes,
    "candidate-Y": outputBytes,
    "seed-Y-fresh": outputBytes,
    "seed-Y-accumulate": outputBytes,
    "baseline-PART": partBytes,
    "candidate-PART": partBytes,
    "seed-PART": partBytes,
    "staged-readback": 2 * CHUNK,
    "timestamp-resolve": 256,
  };
  const gpuTemporaryBytes = Object.values(gpuTemporaryBuffers).reduce((total, bytes) => total + bytes, 0);
  const hostStageBytes = CHUNK;
  const mappedHostCopyBytes = 2 * CHUNK;
  const queuedUploadAllowanceBytes = CHUNK;
  const temporaryBytes = gpuTemporaryBytes + hostStageBytes + mappedHostCopyBytes + queuedUploadAllowanceBytes;
  return { splits, outputCount, partialCount, inputBytes, outputBytes, partBytes, immutableWeightBytes,
    immutableScaleBytes, gpuTemporaryBuffers, gpuTemporaryBytes, hostStageBytes, mappedHostCopyBytes, queuedUploadAllowanceBytes,
    temporaryBytes, limitBytes: LIMIT, excludedImmutableWeightBytes: true };
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
    const countValues = bytes / bytesPerValue;
    const values = bytesPerValue === 1 ? stageI8 : stageF32;
    for (let index = 0; index < countValues; index++) values[index] = valueAt(offset / bytesPerValue + index);
    device.queue.writeBuffer(target, offset, stageBytes, 0, bytes);
    await drain(device, deadline, label);
  }
}

async function fillWithTail(device, target, logicalCount, valueAt, stage, deadline, label) {
  await uploadGenerated(device, target, logicalCount * 4, 4, valueAt, stage, deadline, label);
  device.queue.writeBuffer(target, logicalCount * 4, new Float32Array(TAIL).fill(SENTINEL));
  await drain(device, deadline, label);
}

async function compareBuffers(device, read, baseline, candidate, logicalCount, label, deadline) {
  const total = logicalCount + TAIL;
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  let zeros = 0, negativeZeros = 0, tailGuard = true;
  for (let offset = 0; offset < total; offset += CHUNK / 4) {
    const countValues = Math.min(CHUNK / 4, total - offset);
    const byteLength = countValues * 4;
    const encoder = device.createCommandEncoder({ label: `prefetch.${label}.readback` });
    encoder.copyBufferToBuffer(baseline, offset * 4, read, 0, byteLength);
    encoder.copyBufferToBuffer(candidate, offset * 4, read, byteLength, byteLength);
    device.queue.submit([encoder.finish()]);
    if (!await mapBeforeDeadline(read, byteLength * 2, deadline)) {
      read.unmap();
      throw new Error(`${label} readback deadline exceeded`);
    }
    try {
      const copied = read.getMappedRange(0, byteLength * 2).slice(0);
      const values = new Float32Array(copied);
      const bits = new Uint32Array(copied);
      for (let index = 0; index < countValues; index++) {
        const at = offset + index;
        const other = countValues + index;
        if (at >= logicalCount) {
          if (bits[index] !== SENTINEL_BITS || bits[other] !== SENTINEL_BITS) tailGuard = false;
          if (bits[index] !== bits[other]) bitMismatches++;
          continue;
        }
        if (!Number.isFinite(values[index]) || !Number.isFinite(values[other])) nonfinite++;
        else maxAbs = Math.max(maxAbs, Math.abs(values[index] - values[other]));
        if (bits[index] !== bits[other]) {
          bitMismatches++;
          firstDifference ||= { index: at, baseline: values[index], candidate: values[other],
            baselineBits: `0x${bits[index].toString(16)}`, candidateBits: `0x${bits[other].toString(16)}` };
        }
        if (values[index] === 0) zeros++;
        if (bits[index] === 0x80000000) negativeZeros++;
      }
    } finally { read.unmap(); }
  }
  return { label, samples: logicalCount, bitMismatches, nonfinite, maxAbs, firstDifference,
    zeros, negativeZeros, tailGuard, ok: bitMismatches === 0 && nonfinite === 0 && tailGuard };
}

async function compile(device, source) {
  const spec = { weights: "q8", f16: false, rows: ROWS, row56: true, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul_gemma_batch", spec);
  const candidateCode = prefetchQ8NextBlock(baselineCode);
  const reduceCode = source("reduce_gemma_batch", spec);
  const matmul = matmulLayout(device);
  const reduce = reduceLayout(device);
  const [baseline, candidate, reducer] = await Promise.all([
    pipeline(device, baselineCode, "gemma_prefetch_registered_q8_f32", device.createPipelineLayout({ bindGroupLayouts: [matmul] })),
    pipeline(device, candidateCode, "gemma_prefetch_next_k32_q8_f32", device.createPipelineLayout({ bindGroupLayouts: [matmul] })),
    pipeline(device, reduceCode, "gemma_prefetch_registered_reduce", device.createPipelineLayout({ bindGroupLayouts: [reduce] })),
  ]);
  return { baseline, candidate, reducer, matmulLayout: matmul, reduceLayout: reduce,
    hashes: { baseline: await hash(baselineCode), candidate: await hash(candidateCode), reducer: await hash(reduceCode) } };
}

async function checkCase(runtime, pipes, shape) {
  const { device } = runtime;
  const splits = mmSplits(SPLIT_T, shape.N, shape.K, TARGET, VIRTUAL_BM, BN);
  if (splits !== shape.expectedSplits) throw new Error(`${shape.shape} split policy changed: expected ${shape.expectedSplits}, got ${splits}`);
  const memory = memoryPlan(shape, splits);
  if (memory.temporaryBytes > LIMIT) throw new Error(`temporary memory exceeds 64 MiB: ${memory.temporaryBytes}`);
  const maxBuffer = device.limits.maxBufferSize;
  const maxBinding = device.limits.maxStorageBufferBindingSize;
  const bindingSizes = [shape.N * shape.K, shape.N * shape.K / 8, memory.inputBytes, shape.N * 4,
    memory.outputBytes, memory.partBytes, 2 * CHUNK];
  if (bindingSizes.some((size) => size > maxBuffer || size > maxBinding)) {
    throw new Error(`${shape.shape} buffer exceeds the device binding limit`);
  }

  const deadline = performance.now() + DEADLINE_MS;
  const U = GPUBufferUsage;
  const owned = [];
  const allocatedTemporaryLabels = new Set();
  let querySet = null;
  let actualGpuTemporaryBytes = 0;
  const make = (size, usage, label, immutable = false) => {
    const actualSize = Math.max(16, size);
    if (!immutable) {
      const plannedSize = memory.gpuTemporaryBuffers[label];
      if (plannedSize === undefined) throw new Error(`${label} temporary GPU buffer has no planned allocation`);
      if (actualSize !== plannedSize) throw new Error(`${label} temporary GPU buffer allocated ${actualSize} bytes; plan has ${plannedSize}`);
      if (allocatedTemporaryLabels.has(label)) throw new Error(`${label} temporary GPU buffer was allocated more than once`);
      allocatedTemporaryLabels.add(label);
      const nextBytes = actualGpuTemporaryBytes + actualSize;
      if (nextBytes > memory.gpuTemporaryBytes) {
        throw new Error(`${label} temporary GPU buffers total ${nextBytes} bytes; planned bound is ${memory.gpuTemporaryBytes}`);
      }
      actualGpuTemporaryBytes = nextBytes;
    }
    const buffer = device.createBuffer({ size: actualSize, usage, label: `gemma-prefetch.${shape.shape}.${label}` });
    owned.push(buffer);
    return buffer;
  };
  const bind = (layout, buffers) => device.createBindGroup({
    layout,
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const mapRead = async (read, bytes, label) => {
    if (!await mapBeforeDeadline(read, bytes, deadline)) {
      read.unmap();
      throw new Error(`${label} readback deadline exceeded`);
    }
  };

  try {
    const stage = new ArrayBuffer(CHUNK);
    const stageF32 = new Float32Array(stage);
    const weights = make(memory.immutableWeightBytes, U.STORAGE | U.COPY_DST, "immutable-W", true);
    const scales = make(memory.immutableScaleBytes, U.STORAGE | U.COPY_DST, "immutable-S", true);
    const input = make(memory.inputBytes, U.STORAGE | U.COPY_DST, "X");
    const bias = make(shape.N * 4, U.STORAGE | U.COPY_DST, "B");
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const baselineY = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "baseline-Y");
    const candidateY = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "candidate-Y");
    const baselinePart = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "baseline-PART");
    const candidatePart = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "candidate-PART");
    const freshYSeed = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "seed-Y-fresh");
    const accumulateYSeed = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "seed-Y-accumulate");
    const partSeed = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "seed-PART");
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-readback");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");
    if (allocatedTemporaryLabels.size !== Object.keys(memory.gpuTemporaryBuffers).length ||
        actualGpuTemporaryBytes !== memory.gpuTemporaryBytes) {
      throw new Error(`temporary GPU allocations total ${actualGpuTemporaryBytes} bytes across ${allocatedTemporaryLabels.size} buffers; plan has ${memory.gpuTemporaryBytes} bytes across ${Object.keys(memory.gpuTemporaryBuffers).length} buffers`);
    }

    await uploadGenerated(device, weights, memory.immutableWeightBytes, 1,
      (index) => (index * 31 + (index >>> 5) * 17 + (index >>> 10) * 7 + 19) % 256 - 128,
      stage, deadline, "weight upload");
    await uploadGenerated(device, scales, memory.immutableScaleBytes, 4,
      (index) => index % 257 === 0 ? -0 : (index % 2 ? -1 : 1) * (0.0002 + index % 11 * 0.000037),
      stage, deadline, "scale upload");
    await uploadGenerated(device, input, memory.inputBytes, 4,
      (index) => index % 67 === 0 ? (index % 2 ? -0 : 0) : 0.04 * Math.sin(index * 0.127 + Math.floor(index / shape.K) * 0.13 + 0.3),
      stage, deadline, "activation upload");
    await fillWithTail(device, freshYSeed, memory.outputCount, () => NaN, stage, deadline, "fresh-Y seed upload");
    await fillWithTail(device, accumulateYSeed, memory.outputCount,
      (index) => (index % 23 - 11) * 0.0037, stage, deadline, "accumulate-Y seed upload");
    await fillWithTail(device, partSeed, memory.partialCount, () => NaN, stage, deadline, "PART seed upload");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, SPLIT_T, SPLIT_T, 0]));

    const baselineGroup = bind(pipes.matmulLayout, [globals, params, input, weights, scales, bias, baselineY, baselinePart]);
    const candidateGroup = bind(pipes.matmulLayout, [globals, params, input, weights, scales, bias, candidateY, candidatePart]);
    const baselineReduce = bind(pipes.reduceLayout, [globals, params, baselinePart, bias, baselineY]);
    const candidateReduce = bind(pipes.reduceLayout, [globals, params, candidatePart, bias, candidateY]);
    const dispatch = (pass, candidate) => {
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, candidate ? candidateGroup : baselineGroup);
      pass.dispatchWorkgroups(Math.ceil(shape.N / BN), Math.ceil(T / VIRTUAL_BM), splits);
      if (splits > 1) {
        const work = Math.ceil(T * shape.N / 256);
        pass.setPipeline(pipes.reducer);
        pass.setBindGroup(0, candidate ? candidateReduce : baselineReduce);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const resetOutputs = (mode, encoder) => {
      const seedY = mode === 1 ? accumulateYSeed : freshYSeed;
      for (const target of [baselineY, candidateY]) encoder.copyBufferToBuffer(seedY, 0, target, 0, memory.outputBytes);
      for (const target of [baselinePart, candidatePart]) encoder.copyBufferToBuffer(partSeed, 0, target, 0, memory.partBytes);
    };
    const compareCurrent = async (mode, hasBias, phase) => {
      const output = await compareBuffers(device, read, baselineY, candidateY, memory.outputCount,
        `${shape.shape}.${phase}.mode${mode}.bias${hasBias}.Y`, deadline);
      const partial = await compareBuffers(device, read, baselinePart, candidatePart, memory.partialCount,
        `${shape.shape}.${phase}.mode${mode}.bias${hasBias}.PART`, deadline);
      if (!output.ok || !partial.ok) throw new Error(`bit-exact ${phase} guard failed: ${JSON.stringify({ output, partial })}`);
      return { mode, bias: Boolean(hasBias), output, partial };
    };

    const runGuards = async (phase) => {
      const guards = [];
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
        device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, mode, hasBias]));
        for (let index = 0; index < shape.N; index++) stageF32[index] = hasBias ? (index % 7 - 3) * 0.0137 : NaN;
        device.queue.writeBuffer(bias, 0, stage, 0, shape.N * 4);
        const encoder = device.createCommandEncoder({ label: `gemma-prefetch.${shape.shape}.${phase}.mode${mode}.bias${hasBias}` });
        resetOutputs(mode, encoder);
        const pass = encoder.beginComputePass();
        dispatch(pass, false);
        dispatch(pass, true);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await drain(device, deadline, `${phase} guard dispatch`);
        guards.push(await compareCurrent(mode, hasBias, phase));
      }
      return guards;
    };

    const before = await runGuards("before");
    device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 0]));
    stageF32.fill(NaN, 0, shape.N);
    device.queue.writeBuffer(bias, 0, stage, 0, shape.N * 4);
    const warm = device.createCommandEncoder({ label: `gemma-prefetch.${shape.shape}.warmup` });
    resetOutputs(0, warm);
    const warmPass = warm.beginComputePass();
    dispatch(warmPass, false);
    dispatch(warmPass, true);
    warmPass.end();
    device.queue.submit([warm.finish()]);
    await drain(device, deadline, "timing warmup");

    querySet = device.createQuerySet({ type: "timestamp", count: SAMPLES * 2 * 2, label: `gemma-prefetch.${shape.shape}.timestamps` });
    const timingEncoder = device.createCommandEncoder({ label: `gemma-prefetch.${shape.shape}.paired-timing` });
    const order = [];
    for (let sample = 0; sample < SAMPLES; sample++) {
      const paths = sample % 2 ? [true, false] : [false, true];
      for (const candidate of paths) {
        const at = order.length * 2;
        order.push(candidate);
        const pass = timingEncoder.beginComputePass({ timestampWrites: {
          querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1,
        } });
        for (let repetition = 0; repetition < REPETITIONS; repetition++) dispatch(pass, candidate);
        pass.end();
      }
    }
    timingEncoder.resolveQuerySet(querySet, 0, SAMPLES * 4, resolved, 0);
    timingEncoder.copyBufferToBuffer(resolved, 0, read, 0, SAMPLES * 4 * 8);
    device.queue.submit([timingEncoder.finish()]);
    await drain(device, deadline, "paired timing");
    await mapRead(read, SAMPLES * 4 * 8, "timestamp");
    const timestamps = new BigUint64Array(read.getMappedRange(0, SAMPLES * 4 * 8).slice(0));
    read.unmap();
    const baselineSamples = [], candidateSamples = [];
    for (let sample = 0; sample < order.length; sample++) {
      const elapsedNs = timestamps[2 * sample + 1] - timestamps[2 * sample];
      if (elapsedNs <= 0n) throw new Error(`nonpositive timestamp interval at ${sample}`);
      const milliseconds = Number(elapsedNs) / 1e6 / REPETITIONS;
      (order[sample] ? candidateSamples : baselineSamples).push(milliseconds);
    }
    if (![...baselineSamples, ...candidateSamples].every((value) => Number.isFinite(value) && value > 0)) {
      throw new Error("nonpositive or nonfinite paired GPU timestamp");
    }
    const timing = { method: "paired WebGPU timestamp-query; registered and transformed matmul plus unchanged reducer",
      samples: SAMPLES, repetitions: REPETITIONS, baselineSamples, candidateSamples,
      baselineMs: median(baselineSamples), candidateMs: median(candidateSamples),
      speedup: median(baselineSamples) / median(candidateSamples) };
    const after = await runGuards("after");
    return { ...shape, T, splitT: SPLIT_T, rows: ROWS, row56: true, f16: false, weights: "q8", splits,
      dispatch: { workgroups: [Math.ceil(shape.N / BN), Math.ceil(T / VIRTUAL_BM), splits], virtualBM: VIRTUAL_BM, BN },
      ok: true, guards: { before, after }, timing, sourceHashes: pipes.hashes,
      memory: { ...memory, actualGpuTemporaryBytes, actualTemporaryBytes: actualGpuTemporaryBytes + CHUNK + 2 * CHUNK + CHUNK,
        readbackChunkBytesPerSide: CHUNK, maxQueuedUploadBytes: CHUNK } };
  } finally {
    querySet?.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

async function main() {
  const started = performance.now();
  const params = new URLSearchParams(location.search);
  const requestedT = Number(params.get("T") ?? T);
  if (requestedT !== T) throw new Error("T must be 97 for the registered suffix row56 case");
  const selected = SHAPES.filter((shape) => params.has("shape")
    ? shape.shape === params.get("shape")
    : params.has("all") || DEFAULT_SHAPES.has(shape.shape));
  if (!selected.length) throw new Error("shape must be gate, gate12, down, q, q4k, kv256, kv512, o, or o4k");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const { device } = runtime;
    metadata = { adapter: { name: runtime.name, features: [...device.features] }, wasmUrl: null,
      T, splitT: SPLIT_T, rows: ROWS, row56: true, f16: false, weights: "q8", target: TARGET,
      temporaryLimitBytes: LIMIT, immutableQ8WeightsExcluded: true };
    if (!device.features.has("timestamp-query")) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-prefetch",
        ...metadata, reason: "timestamp-query is required for paired GPU timing; no wall-clock fallback was run" };
      log(window.gpuBench);
      return;
    }
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(device, source), "building registered and Q8-prefetch pipelines");
    metadata.sourceHashes = pipes.hashes;
    for (const shape of selected) {
      const splits = mmSplits(SPLIT_T, shape.N, shape.K, TARGET, VIRTUAL_BM, BN);
      const memory = memoryPlan(shape, splits);
      log({ status: "checking", ...shape, T, splitT: SPLIT_T, rows: ROWS, row56: true, splits, memory });
      const result = await withGpuErrors(runtime, () => checkCase(runtime, pipes, shape), `${shape.shape} exactness and paired timing`);
      cases.push(result);
      window.gpuBench.completed = cases.length;
      log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const baselineMetricMs = geometricMean(cases.map((item) => item.timing.baselineMs));
    const metricMs = geometricMean(cases.map((item) => item.timing.candidateMs));
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-prefetch",
      ...metadata, metric: "geometric mean candidate GPU timestamp milliseconds across selected Q8/F32 suffix shapes",
      metricMs, baselineMetricMs, speedup: baselineMetricMs / metricMs, elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0, guards: "all Y and PART values plus four-float tails, six mode/bias combinations before and after timing", cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-prefetch", completed: 0 };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
      correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || error);
  });
}
