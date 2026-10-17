import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024, CHUNK = MiB, LIMIT = 64 * MiB;
const T = 97, SPLIT_T = 1, ROWS = 4, LOGICAL_BM = 64, PHYSICAL_BM = 56, BN = 64, TARGET = 256, TAIL = 4;
const SENTINEL = 12345.25, SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const DEADLINE_MS = 60_000, SAMPLES = 3, REPETITIONS = 4;
const SHAPES = [
  { shape: "gate", N: 6144, K: 1536, expectedSplits: 3 },
  { shape: "down", N: 1536, K: 6144, expectedSplits: 8 },
  { shape: "wide", N: 12288, K: 1536, expectedSplits: 1 },
];
const cases = [];
let metadata = {};

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const count = (text, marker) => text.split(marker).length - 1;
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");

function replaceExact(code, marker, replacement, expected, label) {
  const found = count(code, marker);
  if (found !== expected) throw new Error(`T97 tail-mask marker mismatch for ${label}: expected ${expected}, found ${found}`);
  return code.replaceAll(marker, replacement);
}

function functionSource(code, name) {
  return code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0] ?? null;
}

function orderedDots(code) {
  return [...code.matchAll(/acc(\d+) \+= vec2<f32>\(dot\(a\1, b0\), dot\(a\1, b1\)\);/g)].map(([, acc]) => Number(acc));
}

export function maskDeadTailDots(row56) {
  if (row56.includes("enable f16;") || !row56.includes("@workgroup_size(32, 8)") ||
      (row56.match(/var acc\d+ = vec2<f32>\(0\.0\);/g) || []).length !== 7 ||
      !row56.includes("let m0 = wg.y * 56u;") || !row56.includes("var<workgroup> xs: array<vec4<f32>, 512>;")) {
    throw new Error("T97 tail mask requires the rendered Q8/F32 R4 row56 kernel");
  }
  const statements = {
    acc5: "      let a5 = vec4<f32>(xs[(lid.y + 40u) * 8u + kq]);\n      acc5 += vec2<f32>(dot(a5, b0), dot(a5, b1));",
    acc6: "      let a6 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);\n      acc6 += vec2<f32>(dot(a6, b0), dot(a6, b1));",
  };
  const guards = {
    acc5: "m0 + lid.y + 40u < T",
    acc6: "m0 + 48u < T",
  };
  let candidate = row56;
  for (const key of ["acc5", "acc6"]) {
    const guarded = `      if (${guards[key]}) {\n${statements[key].split("\n").map((line) => `  ${line}`).join("\n")}\n      }`;
    candidate = replaceExact(candidate, statements[key], guarded, 8, `${key} dot block`);
  }
  for (const key of ["acc5", "acc6"]) {
    const guarded = `      if (${guards[key]}) {\n${statements[key].split("\n").map((line) => `  ${line}`).join("\n")}\n      }`;
    if (count(candidate, guarded) !== 8) throw new Error(`T97 tail-mask did not guard all eight ${key} K32 steps`);
    candidate = replaceExact(candidate, guarded, statements[key], 8, `${key} transform reversal`);
  }
  if (candidate !== row56) throw new Error("T97 tail-mask transform changed source outside acc5/acc6 dot blocks");
  candidate = row56;
  for (const key of ["acc5", "acc6"]) {
    const guarded = `      if (${guards[key]}) {\n${statements[key].split("\n").map((line) => `  ${line}`).join("\n")}\n      }`;
    candidate = replaceExact(candidate, statements[key], guarded, 8, `${key} candidate dot block`);
  }
  if (orderedDots(candidate).length !== 56 || JSON.stringify(orderedDots(candidate)) !== JSON.stringify(orderedDots(row56))) {
    throw new Error("T97 tail-mask changed R4 K32 dot order");
  }
  if (count(candidate, "workgroupBarrier();") !== count(row56, "workgroupBarrier();")) throw new Error("T97 tail-mask changed workgroup barriers");
  for (const name of ["sx", "mm_splits", "mm_store"]) {
    if (!functionSource(row56, name) || functionSource(candidate, name) !== functionSource(row56, name)) {
      throw new Error(`T97 tail-mask changed shared ${name} semantics`);
    }
  }
  return candidate;
}

export function memoryPlan(shape, splits) {
  const outputCount = T * shape.N, partialCount = splits > 1 ? splits * outputCount : 0;
  const inputRows = Math.ceil(T / LOGICAL_BM) * LOGICAL_BM, inputBytes = inputRows * shape.K * 4;
  const outputBytes = (outputCount + TAIL) * 4, partBytes = (partialCount + TAIL) * 4;
  const immutableWeightBytes = shape.N * shape.K, immutableScaleBytes = shape.N * shape.K / 8;
  const gpuTemporaryBuffers = {
    X: inputBytes, B: shape.N * 4, globals: 16, params: 16,
    "baseline-Y": outputBytes, "candidate-Y": outputBytes,
    "baseline-PART": partBytes, "candidate-PART": partBytes,
    "seed-Y-fresh": outputBytes, "seed-Y-accumulate": outputBytes, "seed-PART": partBytes,
    "staged-readback": 2 * CHUNK, "timestamp-resolve": 256,
  };
  const gpuTemporaryBytes = Object.values(gpuTemporaryBuffers).reduce((total, bytes) => total + bytes, 0);
  const hostStageBytes = CHUNK, mappedHostCopyBytes = 2 * CHUNK, queuedUploadAllowanceBytes = CHUNK, fixedHostScratchBytes = 128;
  const temporaryBytes = gpuTemporaryBytes + hostStageBytes + mappedHostCopyBytes + queuedUploadAllowanceBytes + fixedHostScratchBytes;
  return { splits, outputCount, partialCount, inputRows, inputBytes, outputBytes, partBytes,
    immutableWeightBytes, immutableScaleBytes, gpuTemporaryBuffers, gpuTemporaryBytes, hostStageBytes,
    mappedHostCopyBytes, queuedUploadAllowanceBytes, fixedHostScratchBytes, temporaryBytes, limitBytes: LIMIT,
    excludedImmutableQ8WeightAndScaleBytes: true };
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
  const stageBytes = new Uint8Array(stage), stageI8 = new Int8Array(stage), stageF32 = new Float32Array(stage);
  for (let offset = 0; offset < byteLength; offset += CHUNK) {
    const bytes = Math.min(CHUNK, byteLength - offset), values = bytesPerValue === 1 ? stageI8 : stageF32;
    for (let index = 0; index < bytes / bytesPerValue; index++) values[index] = valueAt(offset / bytesPerValue + index);
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
  let zeros = 0, negativeZeros = 0;
  for (let offset = 0; offset < total; offset += CHUNK / 4) {
    const countValues = Math.min(CHUNK / 4, total - offset), byteLength = countValues * 4;
    const encoder = device.createCommandEncoder({ label: `tail-mask.${label}.readback` });
    encoder.copyBufferToBuffer(baseline, offset * 4, read, 0, byteLength);
    encoder.copyBufferToBuffer(candidate, offset * 4, read, byteLength, byteLength);
    device.queue.submit([encoder.finish()]);
    if (!await mapBeforeDeadline(read, byteLength * 2, deadline)) { read.unmap(); throw new Error(`${label} readback deadline exceeded`); }
    try {
      const copy = read.getMappedRange(0, byteLength * 2).slice(0), values = new Float32Array(copy), bits = new Uint32Array(copy);
      for (let index = 0; index < countValues; index++) {
        const at = offset + index, other = countValues + index;
        if (at >= logicalCount) {
          if (bits[index] !== SENTINEL_BITS || bits[other] !== SENTINEL_BITS) throw new Error(`${label} four-float tail overwrite at ${at}`);
          continue;
        }
        if (!Number.isFinite(values[index]) || !Number.isFinite(values[other])) throw new Error(`${label} nonfinite active value at ${at}`);
        if (bits[index] !== bits[other]) throw new Error(`${label} bit mismatch at ${at}: 0x${bits[index].toString(16)} vs 0x${bits[other].toString(16)}`);
        if (values[index] === 0) zeros++;
        if (bits[index] === 0x80000000) negativeZeros++;
      }
    } finally { read.unmap(); }
  }
  return { ok: true, samples: logicalCount, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros,
    signedZeroChecked: true, fourFloatTailChecked: true };
}

function assertGridCoverage() {
  const coverage = new Uint8Array(T);
  for (let group = 0; group < Math.ceil(T / PHYSICAL_BM); group++) {
    for (let row = group * PHYSICAL_BM; row < Math.min(T, (group + 1) * PHYSICAL_BM); row++) coverage[row]++;
  }
  if (coverage.some((count) => count !== 1)) throw new Error("BM56 grid does not cover T=97 exactly once");
  return Math.ceil(T / PHYSICAL_BM);
}

async function compile(device, source) {
  const spec = { weights: "q8", f16: false, rows: ROWS, row56: true, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul_gemma_batch", spec), candidateCode = maskDeadTailDots(baselineCode);
  const reduceCode = source("reduce_gemma_batch", spec), layout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const matmulPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const reducePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] });
  const [baseline, candidate, reducer] = await Promise.all([
    pipeline(device, baselineCode, "gemma_matmul_q8_f32_row56", matmulPipelineLayout),
    pipeline(device, candidateCode, "gemma_matmul_q8_f32_t97_dead_tail_mask", matmulPipelineLayout),
    pipeline(device, reduceCode, "gemma_matmul_q8_f32_row56_reduce", reducePipelineLayout),
  ]);
  return { baseline, candidate, reducer, layout, reductionLayout,
    hashes: { baseline: await hash(baselineCode), candidate: await hash(candidateCode), reducer: await hash(reduceCode) } };
}

async function checkCase(runtime, pipes, shape) {
  const { device } = runtime, splits = mmSplits(SPLIT_T, shape.N, shape.K, TARGET, LOGICAL_BM, BN);
  if (splits !== shape.expectedSplits) throw new Error(`${shape.shape} logical BM64 split policy changed: expected ${shape.expectedSplits}, got ${splits}`);
  const memory = memoryPlan(shape, splits);
  if (memory.temporaryBytes > LIMIT) throw new Error(`${shape.shape} temporary allocations use ${memory.temporaryBytes} bytes; 64 MiB limit is ${LIMIT}`);
  if (device.limits.maxComputeWorkgroupStorageSize < 16_384) return { ...shape, T, splits, ok: true, skipped: true, reason: "row56 needs 16,384 bytes of workgroup storage", memory };
  const bindingSizes = [memory.immutableWeightBytes, memory.immutableScaleBytes, memory.inputBytes, shape.N * 4,
    memory.outputBytes, memory.partBytes, 2 * CHUNK];
  if (bindingSizes.some((size) => size > device.limits.maxBufferSize || size > device.limits.maxStorageBufferBindingSize)) {
    throw new Error(`${shape.shape} buffer exceeds the device binding limit`);
  }

  const deadline = performance.now() + DEADLINE_MS, U = GPUBufferUsage, owned = [], allocated = new Set();
  let actualGpuTemporaryBytes = 0, querySet;
  const make = (size, usage, label, immutable = false) => {
    const actualSize = Math.max(16, size);
    if (!immutable) {
      const planned = memory.gpuTemporaryBuffers[label];
      if (planned === undefined) throw new Error(`${label} temporary GPU buffer has no planned allocation`);
      if (actualSize !== planned) throw new Error(`${label} temporary GPU buffer allocated ${actualSize} bytes; plan has ${planned}`);
      if (allocated.has(label)) throw new Error(`${label} temporary GPU buffer allocated twice`);
      if (actualGpuTemporaryBytes + actualSize > memory.gpuTemporaryBytes) throw new Error(`${label} temporary GPU buffers exceeded the planned bound`);
      allocated.add(label);
      actualGpuTemporaryBytes += actualSize;
    }
    const buffer = device.createBuffer({ size: actualSize, usage, label: `tail-mask.${shape.shape}.${label}` });
    owned.push(buffer);
    return buffer;
  };
  const bind = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const stage = new ArrayBuffer(CHUNK);
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
    const freshSeed = make(memory.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "seed-Y-fresh");
    const accumulateSeed = make(memory.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "seed-Y-accumulate");
    const partSeed = make(memory.partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "seed-PART");
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-readback");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");
    if (allocated.size !== Object.keys(memory.gpuTemporaryBuffers).length || actualGpuTemporaryBytes !== memory.gpuTemporaryBytes) {
      throw new Error(`temporary GPU allocations total ${actualGpuTemporaryBytes} bytes across ${allocated.size} buffers; plan has ${memory.gpuTemporaryBytes} bytes across ${Object.keys(memory.gpuTemporaryBuffers).length} buffers`);
    }

    await uploadGenerated(device, weights, memory.immutableWeightBytes, 1,
      (index) => (index * 31 + (index >>> 5) * 17 + (index >>> 10) * 7 + 19) % 256 - 128, stage, deadline, "weight upload");
    await uploadGenerated(device, scales, memory.immutableScaleBytes, 4,
      (index) => index % 257 === 0 ? -0 : (index % 2 ? -1 : 1) * (0.0002 + index % 11 * 0.000037), stage, deadline, "scale upload");
    await uploadGenerated(device, input, memory.inputBytes, 4, (index) => {
      const row = Math.floor(index / shape.K), k = index % shape.K;
      if (row >= T) return NaN;
      if (index % 67 === 0) return index % 2 ? -0 : 0;
      return 0.04 * Math.sin(k * 0.127 + row * 0.13 + 0.3);
    }, stage, deadline, "NaN-padded activation upload");
    await fillWithTail(device, freshSeed, memory.outputCount, () => NaN, stage, deadline, "fresh Y seed upload");
    await fillWithTail(device, accumulateSeed, memory.outputCount, (index) => (index % 23 - 11) * 0.0037, stage, deadline, "accumulate Y seed upload");
    await fillWithTail(device, partSeed, memory.partialCount, () => NaN, stage, deadline, "PART seed upload");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, SPLIT_T, SPLIT_T, 0]));

    const baselineGroup = bind(pipes.layout, [globals, params, input, weights, scales, bias, baselineY, baselinePart]);
    const candidateGroup = bind(pipes.layout, [globals, params, input, weights, scales, bias, candidateY, candidatePart]);
    const baselineReduceGroup = bind(pipes.reductionLayout, [globals, params, baselinePart, bias, baselineY]);
    const candidateReduceGroup = bind(pipes.reductionLayout, [globals, params, candidatePart, bias, candidateY]);
    const work = Math.ceil(T * shape.N / 256), rowGroups = assertGridCoverage();
    const dispatch = (pass, candidate) => {
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, candidate ? candidateGroup : baselineGroup);
      pass.dispatchWorkgroups(Math.ceil(shape.N / BN), rowGroups, splits);
      if (splits > 1) {
        pass.setPipeline(pipes.reducer);
        pass.setBindGroup(0, candidate ? candidateReduceGroup : baselineReduceGroup);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const reset = (mode, encoder) => {
      const seedY = mode === 1 ? accumulateSeed : freshSeed;
      for (const output of [baselineY, candidateY]) encoder.copyBufferToBuffer(seedY, 0, output, 0, memory.outputBytes);
      for (const part of [baselinePart, candidatePart]) encoder.copyBufferToBuffer(partSeed, 0, part, 0, memory.partBytes);
    };
    const comparePair = async (phase, mode, hasBias) => {
      const output = await compareBuffers(device, read, baselineY, candidateY, memory.outputCount, `${phase}.mode${mode}.bias${hasBias}.Y`, deadline);
      const partial = await compareBuffers(device, read, baselinePart, candidatePart, memory.partialCount, `${phase}.mode${mode}.bias${hasBias}.PART`, deadline);
      return { mode, bias: Boolean(hasBias), output, partial };
    };
    const runPair = async (phase, mode, hasBias) => {
      device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, mode, hasBias]));
      await uploadGenerated(device, bias, shape.N * 4, 4,
        (index) => hasBias ? (index % 7 - 3) * 0.0137 : NaN, stage, deadline, `${phase} bias upload`);
      const encoder = device.createCommandEncoder({ label: `tail-mask.${shape.shape}.${phase}.mode${mode}.bias${hasBias}` });
      reset(mode, encoder);
      const pass = encoder.beginComputePass();
      dispatch(pass, false); dispatch(pass, true); pass.end();
      device.queue.submit([encoder.finish()]);
      await drain(device, deadline, `${phase} dispatch`);
      const checked = await comparePair(phase, mode, hasBias);
      if (!checked.output.ok || !checked.partial.ok) throw new Error(`${phase} mode${mode}/bias${hasBias} exactness failed`);
      return checked;
    };
    const runGuards = async (phase) => {
      const results = [];
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) results.push(await runPair(phase, mode, hasBias));
      return results;
    };

    const before = await runGuards("before");
    device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 0]));
    await uploadGenerated(device, bias, shape.N * 4, 4, () => NaN, stage, deadline, "warmup bias upload");
    const warm = device.createCommandEncoder({ label: `tail-mask.${shape.shape}.warmup` });
    reset(0, warm);
    const warmPass = warm.beginComputePass(); dispatch(warmPass, false); dispatch(warmPass, true); warmPass.end();
    device.queue.submit([warm.finish()]);
    await drain(device, deadline, "timing warmup");

    let timing = null;
    if (device.features.has("timestamp-query")) {
      querySet = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4, label: `tail-mask.${shape.shape}.timestamps` });
      const encoder = device.createCommandEncoder({ label: `tail-mask.${shape.shape}.paired-timing` }), order = [];
      let queryIndex = 0;
      for (let sample = 0; sample < SAMPLES; sample++) for (const candidate of sample % 2 ? [true, false] : [false, true]) {
        reset(0, encoder);
        const at = queryIndex; order.push(candidate);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
        for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(pass, candidate);
        pass.end(); queryIndex += 2;
      }
      encoder.resolveQuerySet(querySet, 0, SAMPLES * 4, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, read, 0, SAMPLES * 4 * 8);
      device.queue.submit([encoder.finish()]);
      if (!await mapBeforeDeadline(read, SAMPLES * 4 * 8, deadline)) { read.unmap(); throw new Error("paired timing readback deadline exceeded"); }
      try {
        const stamps = new BigUint64Array(read.getMappedRange(0, SAMPLES * 4 * 8).slice(0));
        const samples = { baseline: [], candidate: [] };
        order.forEach((candidate, index) => samples[candidate ? "candidate" : "baseline"].push(Number(stamps[index * 2 + 1] - stamps[index * 2]) / 1e6 / REPETITIONS));
        const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
        if (![baselineMs, candidateMs].every((value) => Number.isFinite(value) && value > 0)) throw new Error("invalid paired GPU timestamps");
        timing = { method: "paired GPU timestamp-query, matmul plus identical reducer", samples: SAMPLES, repetitions: REPETITIONS,
          ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs };
      } finally { read.unmap(); }
    }
    const after = await runGuards("after");
    return { ...shape, T, splitT: SPLIT_T, rows: ROWS, f16: false, weights: "q8", baselineRow56: true,
      baselinePhysicalBM: PHYSICAL_BM, candidatePhysicalBM: PHYSICAL_BM, logicalBM: LOGICAL_BM, BN, splits,
      dispatch: { workgroupsX: Math.ceil(shape.N / BN), workgroupsY: rowGroups, splitK: splits },
      deadTailMask: { acc6Condition: "m0 + 48u < T", acc5Condition: "m0 + lid.y + 40u < T", guardedK32Steps: { acc5: 8, acc6: 8 } },
      ok: true, guards: { before, after, inputPadding: "NaN outside T=97 active rows", allYAndPARTFinite: true, fourFloatTails: true },
      timing, sourceHashes: pipes.hashes,
      memory: { ...memory, actualGpuTemporaryBytes, actualTemporaryBytes: actualGpuTemporaryBytes + CHUNK + 2 * CHUNK + CHUNK + memory.fixedHostScratchBytes,
        readbackChunkBytesPerSide: CHUNK, maxQueuedUploadBytes: CHUNK } };
  } finally { querySet?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  if (Number(params.get("T") ?? T) !== T) throw new Error("this DEV tail-mask probe is restricted to T=97");
  const selected = SHAPES.filter(({ shape }) => params.has("shape") ? shape === params.get("shape") : params.get("all") === "1" || shape === "gate");
  if (!selected.length) throw new Error("shape must be gate, down, or wide");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const { device } = runtime;
    metadata = { adapter: { name: runtime.name, features: [...device.features], maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize },
      T, splitT: SPLIT_T, rows: ROWS, weights: "q8", f16: false, baselineRow56: true, baselinePhysicalBM: PHYSICAL_BM,
      candidatePhysicalBM: PHYSICAL_BM, logicalBM: LOGICAL_BM, BN, temporaryLimitBytes: LIMIT, immutableQ8WeightsAndScalesExcluded: true,
      requiredFeatures: ["timestamp-query"] };
    if (!device.features.has("timestamp-query")) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-matmul-tail-mask", ...metadata,
        reason: "timestamp-query is unavailable; no wall-clock fallback was run" };
      log(window.gpuBench);
      return;
    }
    if (device.limits.maxComputeWorkgroupStorageSize < 16_384) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-matmul-tail-mask", ...metadata,
        reason: "row56 requires 16,384 bytes of workgroup storage" };
      log(window.gpuBench);
      return;
    }
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(device, source), "building T97 row56 and dead-tail-mask pipelines");
    metadata.sourceHashes = pipes.hashes;
    for (const shape of selected) {
      const splits = mmSplits(SPLIT_T, shape.N, shape.K, TARGET, LOGICAL_BM, BN), memory = memoryPlan(shape, splits);
      if (splits !== shape.expectedSplits) throw new Error(`${shape.shape} expected ${shape.expectedSplits} split-K partitions, got ${splits}`);
      if (memory.temporaryBytes > LIMIT) throw new Error(`${shape.shape} planned temporary memory exceeds 64 MiB`);
      log({ status: "checking", ...shape, T, splitT: SPLIT_T, baselinePhysicalBM: PHYSICAL_BM, candidatePhysicalBM: PHYSICAL_BM,
        logicalBM: LOGICAL_BM, splits, memory });
      const result = await withGpuErrors(runtime, () => checkCase(runtime, pipes, shape), `${shape.shape} exactness and paired timing`);
      cases.push(result); window.gpuBench.completed = cases.length; log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const timed = cases.filter((item) => item.timing?.candidateMs > 0);
    const baselineMetricMs = timed.length ? geometricMean(timed.map((item) => item.timing.baselineMs)) : null;
    const metricMs = timed.length ? geometricMean(timed.map((item) => item.timing.candidateMs)) : null;
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matmul-tail-mask", ...metadata,
      metric: "geometric mean candidate GPU timestamp milliseconds across selected Q8/F32 T=97 shapes", metricMs, baselineMetricMs,
      speedup: metricMs ? baselineMetricMs / metricMs : null, elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0,
        guards: "all Y and PART values plus four-float tails, finite active outputs, NaN activation padding, and six mode/bias combinations before and after timing", cases } };
    log(window.gpuBench);
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matmul-tail-mask", completed: 0 };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", check: "gemma4-matmul-tail-mask",
      ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || String(error));
  });
}
