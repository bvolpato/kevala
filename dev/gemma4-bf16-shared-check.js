import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const LIMIT = 64 * MiB;
const CHUNK = MiB;
const T = 144;
const ROWS = 3;
const BN = 64;
const TARGET = 256;
const TAIL = 4;
const SAMPLES = 3;
const REPETITIONS = 4;
const DEADLINE_MS = 60_000;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const SHAPES = [
  { shape: "gate-up", N: 3072, K: 768, primary: true, actualVision: true },
  { shape: "down", N: 768, K: 3072, primary: true, actualVision: true },
  { shape: "attention", N: 768, K: 768, primary: true, actualVision: true },
  { shape: "projection", N: 1536, K: 768, primary: false, actualVision: true },
  { shape: "ragged", N: 770, K: 768, actualVision: false },
];
const WEIGHT_FIXTURES = ["normal", "tiny-subnormal", "large-finite"];
const cases = [];
let metadata = {};

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const count = (text, marker) => text.split(marker).length - 1;
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");

function replaceExact(code, marker, replacement, expected, label) {
  const found = count(code, marker);
  if (found !== expected) throw new Error(`BF16 shared-tile marker mismatch for ${label}: expected ${expected}, found ${found}`);
  return code.replaceAll(marker, replacement);
}

function functionSource(code, name) {
  return code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0] ?? null;
}

function orderedDots(code) {
  return [...code.matchAll(/acc(\d+) \+= vec2<f32>\(dot\(a\d+, b0\), dot\(a\d+, b1\)\);/g)].map(([, acc]) => Number(acc));
}

export function derivePackedBf16Shared(baseline) {
  const required = [
    "@compute @workgroup_size(32, 8)",
    "let m0 = wg.y * 48u;",
    "var<workgroup> xs: array<vec4<f32>, 512>;",
    "var<workgroup> ws: array<vec4<f32>, 512>;",
    "fn bf16(w: vec2<u32>) -> vec4<f32>",
  ];
  if (baseline.includes("enable f16;") || required.some((marker) => !baseline.includes(marker)) ||
      count(baseline, "let sw = lr & 7u;") !== 1 || count(baseline, "let nx = lid.x & 7u;") !== 1 ||
      orderedDots(baseline).length !== 48 || count(baseline, "workgroupBarrier();") !== 2) {
    throw new Error("packed BF16 shared variant requires registered generic R3/BM48, BN64, F32 tiles, and eight unrolled K32 steps");
  }
  const changes = [
    ["var<workgroup> ws: array<vec4<f32>, 512>;", "var<workgroup> ws: array<vec2<u32>, 512>;", 1, "shared weight tile type"],
    ["let sw = lr & 7u;", "let sw = (lr >> 1u) & 7u;", 1, "producer row swizzle"],
    ["var wv: array<vec4<f32>, 2>;", "var wv: array<vec2<u32>, 2>;", 1, "private raw-word staging"],
    ["wv[2u * gg] = vec4<f32>(0.0);", "wv[2u * gg] = vec2<u32>(0u);", 1, "zero-column low word"],
    ["wv[2u * gg + 1u] = vec4<f32>(0.0);", "wv[2u * gg + 1u] = vec2<u32>(0u);", 1, "zero-column high word"],
    ["wv[2u * gg] = bf16(v.xy);", "wv[2u * gg] = v.xy;", 1, "raw low BF16 pair"],
    ["wv[2u * gg + 1u] = bf16(v.zw);", "wv[2u * gg + 1u] = v.zw;", 1, "raw high BF16 pair"],
    ["ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(wv[2u * gg]);", "ws[r * 8u + ((lq * 2u) ^ sw)] = wv[2u * gg];", 1, "shared low-word store"],
    ["ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(wv[2u * gg + 1u]);", "ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = wv[2u * gg + 1u];", 1, "shared high-word store"],
    ["let nx = lid.x & 7u;", "let nx = (lid.x >> 1u) & 7u;", 1, "consumer row swizzle"],
    ["let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);", "let b0 = bf16(ws[c * 8u + (kq ^ nx)]);", 8, "decode first shared read"],
    ["let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);", "let b1 = bf16(ws[(c + 32u) * 8u + (kq ^ nx)]);", 8, "decode second shared read"],
  ];
  let candidate = baseline;
  for (const [marker, replacement, expected, label] of changes) candidate = replaceExact(candidate, marker, replacement, expected, label);
  let restored = candidate;
  for (const [marker, replacement, expected, label] of [...changes].reverse()) {
    restored = replaceExact(restored, replacement, marker, expected, `${label} reversal`);
  }
  if (restored !== baseline) throw new Error("packed BF16 transform changed source outside its declared tile, swizzle, store, and decode sites");
  if (orderedDots(candidate).length !== 48 || JSON.stringify(orderedDots(candidate)) !== JSON.stringify(orderedDots(baseline))) {
    throw new Error("packed BF16 transform changed the eight-step dot or accumulator order");
  }
  for (const name of ["bf16", "mm_splits", "mm_store"]) {
    if (!functionSource(baseline, name) || functionSource(candidate, name) !== functionSource(baseline, name)) {
      throw new Error(`packed BF16 transform changed shared ${name} semantics`);
    }
  }
  if (count(candidate, "workgroupBarrier();") !== count(baseline, "workgroupBarrier();")) {
    throw new Error("packed BF16 transform changed workgroup barrier placement");
  }
  return candidate;
}

export function assertPackedMapping() {
  const slots = new Set();
  for (let row = 0; row < 64; row++) for (let q = 0; q < 8; q++) {
    slots.add(row * 8 + (q ^ ((row >> 1) & 7)));
  }
  if (slots.size !== 512) throw new Error("packed shared writer swizzle is not a 64x8 bijection");
  for (let col = 0; col < 32; col++) {
    const consumerSwizzle = (col >> 1) & 7;
    if (consumerSwizzle !== (((col + 32) >> 1) & 7)) throw new Error("paired output columns do not share the same packed-word swizzle");
    for (let kq = 0; kq < 8; kq++) {
      const slot = kq ^ consumerSwizzle;
      if ((slot ^ ((col >> 1) & 7)) !== kq ||
          (slot ^ (((col + 32) >> 1) & 7)) !== kq) throw new Error("packed shared consumer does not invert the writer swizzle");
    }
  }
  for (let word = 0; word < 0x10000; word++) {
    const highWord = word ^ 0xffff;
    const packed = (word | (highWord << 16)) >>> 0;
    const lowBits = (packed << 16) >>> 0;
    const highBits = (packed & 0xffff0000) >>> 0;
    if (lowBits !== ((word << 16) >>> 0) || highBits !== ((highWord << 16) >>> 0)) {
      throw new Error(`BF16 raw-word decode differs at 0x${word.toString(16)}`);
    }
  }
  const banks = new Set();
  for (let col = 0; col < 32; col++) {
    const slot = 3 ^ ((col >> 1) & 7);
    const firstBank = (col * 8 * 2 + slot * 2) & 31;
    banks.add(firstBank);
    banks.add((firstBank + 1) & 31);
  }
  if (banks.size !== 32) throw new Error(`packed 64-bit reads cover ${banks.size} shared banks across 32 lanes, expected 32`);
  return { writerSwizzle: "(lr >> 1) & 7", readerSwizzle: "(lid.x >> 1) & 7", bijectiveSlots: 512,
    pairedColumnsShareMapping: true, bf16WordsVerified: 65536, banksCoveredForFixedKStep: banks.size };
}

function bf16Word(kind, index) {
  const sign = ((index * 13 + (index >>> 4)) & 1) ? 0x8000 : 0;
  if (index % 257 === 0) return sign;
  if (kind === "tiny-subnormal") return sign | (1 + ((index * 29 + 7) % 0x7f));
  if (kind === "large-finite") return sign | ((133 + (index % 5)) << 7) | ((index * 37 + 11) & 0x7f);
  return sign | ((126 + (index % 3)) << 7) | ((index * 37 + 11) & 0x7f);
}

function inputValue(index, K) {
  if (index % 97 === 0) return index % 194 === 0 ? 0 : -0;
  const row = Math.floor(index / K), k = index % K;
  return 0.0125 * Math.sin(k * 0.071 + row * 0.17 + 0.23);
}

function seedValue(index, N) {
  if (index % 193 === 0) return index % 386 === 0 ? 0 : -0;
  const row = Math.floor(index / N), col = index % N;
  return 0.013 * Math.cos(row * 0.11 + col * 0.037);
}

function biasValue(index, enabled) {
  if (!enabled) return index % 2 ? -0 : 0;
  if (index % 101 === 0) return index % 202 === 0 ? 0 : -0;
  return (index % 13 - 6) * 0.007;
}

export function memoryPlan(shape, splits, sourceBytes) {
  const outputCount = T * shape.N;
  const partialCount = splits > 1 ? splits * outputCount : 0;
  if (splits <= 1) throw new Error("vision BF16 probe expects split-K PART planes for every selected shape");
  const outputBytes = (outputCount + TAIL) * 4;
  const partBytes = (partialCount + TAIL) * 4;
  const fixtureWeightBytes = shape.N * shape.K * 2;
  const gpuTemporaryBuffers = {
    X: T * shape.K * 4,
    B: shape.N * 4,
    S: 16,
    globals: 16,
    params: 16,
    "baseline-Y": outputBytes,
    "candidate-Y": outputBytes,
    "seed-Y": outputBytes,
    "baseline-PART": partBytes,
    "candidate-PART": partBytes,
    "staged-readback": 2 * CHUNK,
    "timestamp-resolve": 256,
  };
  const gpuBufferBytes = Object.values(gpuTemporaryBuffers).reduce((total, bytes) => total + bytes, 0);
  const timestampQueryBytes = SAMPLES * 2 * 2 * 8;
  const hostStageBytes = CHUNK;
  const mappedHostCopyBytes = 2 * CHUNK;
  const queuedUploadAllowanceBytes = CHUNK;
  const fixedHostScratchBytes = 2 * sourceBytes + 64 * 1024;
  const temporaryBytes = gpuBufferBytes + timestampQueryBytes + hostStageBytes + mappedHostCopyBytes + queuedUploadAllowanceBytes + fixedHostScratchBytes;
  return { outputCount, partialCount, outputBytes, partBytes, fixtureWeightBytes, gpuTemporaryBuffers,
    gpuBufferBytes, timestampQueryBytes, hostStageBytes, mappedHostCopyBytes, queuedUploadAllowanceBytes,
    shaderWorkingBytes: sourceBytes, fixedHostScratchBytes, temporaryBytes, limitBytes: LIMIT,
    excludedImmutableFixtureWeightBytes: true };
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
  const bytes = new Uint8Array(stage), view = new DataView(stage);
  for (let offset = 0; offset < byteLength; offset += CHUNK) {
    const length = Math.min(CHUNK, byteLength - offset);
    if (length % bytesPerValue || length % 4) throw new Error(`${label} chunk is not aligned to its value width and WebGPU's four-byte write alignment`);
    const first = offset / bytesPerValue;
    for (let index = 0; index < length / bytesPerValue; index++) {
      if (bytesPerValue === 2) view.setUint16(index * 2, valueAt(first + index), true);
      else if (bytesPerValue === 4) view.setFloat32(index * 4, valueAt(first + index), true);
      else throw new Error(`unsupported upload width ${bytesPerValue}`);
    }
    device.queue.writeBuffer(target, offset, bytes, 0, length);
    await drain(device, deadline, label);
  }
}

async function uploadTail(device, target, logicalCount, deadline, label) {
  device.queue.writeBuffer(target, logicalCount * 4, new Float32Array(TAIL).fill(SENTINEL));
  await drain(device, deadline, label);
}

async function compareBuffers(device, read, baseline, candidate, logicalCount, label, deadline) {
  const total = logicalCount + TAIL;
  let zeros = 0, negativeZeros = 0;
  for (let offset = 0; offset < total; offset += CHUNK / 4) {
    const valuesCount = Math.min(CHUNK / 4, total - offset), byteLength = valuesCount * 4;
    const encoder = device.createCommandEncoder({ label: `bf16-shared.${label}.readback` });
    encoder.copyBufferToBuffer(baseline, offset * 4, read, 0, byteLength);
    encoder.copyBufferToBuffer(candidate, offset * 4, read, byteLength, byteLength);
    device.queue.submit([encoder.finish()]);
    if (!await mapBeforeDeadline(read, byteLength * 2, deadline)) {
      read.unmap();
      throw new Error(`${label} readback deadline exceeded`);
    }
    try {
      const copy = read.getMappedRange(0, byteLength * 2).slice(0);
      const values = new Float32Array(copy), bits = new Uint32Array(copy);
      for (let index = 0; index < valuesCount; index++) {
        const at = offset + index, other = valuesCount + index;
        if (at >= logicalCount) {
          if (bits[index] !== SENTINEL_BITS || bits[other] !== SENTINEL_BITS) throw new Error(`${label} tail sentinel changed at ${at}`);
          continue;
        }
        if (!Number.isFinite(values[index]) || !Number.isFinite(values[other])) throw new Error(`${label} has a nonfinite active value at ${at}`);
        if (bits[index] !== bits[other]) throw new Error(`${label} bit mismatch at ${at}: 0x${bits[index].toString(16)} vs 0x${bits[other].toString(16)}`);
        if (values[index] === 0) zeros++;
        if (bits[index] === 0x80000000) negativeZeros++;
      }
    } finally { read.unmap(); }
  }
  return { ok: true, samples: logicalCount, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros, tailSentinels: TAIL };
}

async function compile(device, source) {
  const spec = { weights: "bf16", f16: false, rows: ROWS, row56: false, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul", spec);
  const candidateCode = derivePackedBf16Shared(baselineCode);
  const reduceCode = source("reduce", spec);
  const matmul = matmulLayout(device), reduce = reduceLayout(device);
  const matmulPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [matmul] });
  const reducePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [reduce] });
  const [baseline, candidate, reducer] = await Promise.all([
    pipeline(device, baselineCode, "vision_bf16_registered_r3_bm48", matmulPipelineLayout),
    pipeline(device, candidateCode, "vision_bf16_packed_shared_r3_bm48", matmulPipelineLayout),
    pipeline(device, reduceCode, "vision_bf16_registered_reduce_r3", reducePipelineLayout),
  ]);
  return { baseline, candidate, reducer, matmulLayout: matmul, reduceLayout: reduce,
    sourceBytes: new TextEncoder().encode(baselineCode).byteLength + new TextEncoder().encode(candidateCode).byteLength + new TextEncoder().encode(reduceCode).byteLength,
    hashes: { baseline: await hash(baselineCode), candidate: await hash(candidateCode), reducer: await hash(reduceCode) } };
}

async function checkShape(runtime, pipes, shape, { full }) {
  const { device } = runtime;
  const splits = mmSplits(T, shape.N, shape.K, TARGET, ROWS * 16, BN);
  const memory = memoryPlan(shape, splits, pipes.sourceBytes);
  if (memory.temporaryBytes > LIMIT) throw new Error(`${shape.shape} temporary memory ${memory.temporaryBytes} exceeds the 64 MiB bound ${LIMIT}`);
  const allBufferSizes = [memory.fixtureWeightBytes, ...Object.values(memory.gpuTemporaryBuffers)];
  const storageSizes = [memory.fixtureWeightBytes, memory.gpuTemporaryBuffers.X, memory.gpuTemporaryBuffers.B,
    memory.gpuTemporaryBuffers.S, memory.outputBytes, memory.partBytes];
  if (allBufferSizes.some((bytes) => bytes > device.limits.maxBufferSize)) throw new Error(`${shape.shape} buffer exceeds maxBufferSize`);
  if (storageSizes.some((bytes) => bytes > device.limits.maxStorageBufferBindingSize)) {
    throw new Error(`${shape.shape} storage binding exceeds maxStorageBufferBindingSize`);
  }
  const deadline = performance.now() + (full ? 180_000 : DEADLINE_MS);
  const U = GPUBufferUsage, owned = [], allocated = new Set();
  let actualGpuBufferBytes = 0, querySet = null;
  const make = (size, usage, label, immutableFixture = false) => {
    const actualSize = Math.max(16, size);
    if (!immutableFixture) {
      const planned = memory.gpuTemporaryBuffers[label];
      if (planned === undefined) throw new Error(`${label} GPU temporary has no planned allocation`);
      if (actualSize !== planned) throw new Error(`${label} allocated ${actualSize} bytes; plan is ${planned}`);
      if (allocated.has(label)) throw new Error(`${label} GPU temporary was allocated more than once`);
      allocated.add(label);
      actualGpuBufferBytes += actualSize;
      if (actualGpuBufferBytes > memory.gpuBufferBytes) throw new Error("actual GPU temporary buffers exceeded the planned bound");
    }
    const buffer = device.createBuffer({ size: actualSize, usage, label: `bf16-shared.${shape.shape}.${label}` });
    owned.push(buffer);
    return buffer;
  };
  const bind = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const stage = new ArrayBuffer(CHUNK);
    const weights = make(memory.fixtureWeightBytes, U.STORAGE | U.COPY_DST, "fixture-BF16-W", true);
    const input = make(memory.gpuTemporaryBuffers.X, U.STORAGE | U.COPY_DST, "X");
    const bias = make(memory.gpuTemporaryBuffers.B, U.STORAGE | U.COPY_DST, "B");
    const scales = make(memory.gpuTemporaryBuffers.S, U.STORAGE | U.COPY_DST, "S");
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const baselineY = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "baseline-Y");
    const candidateY = make(memory.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "candidate-Y");
    const seedY = make(memory.outputBytes, U.COPY_DST | U.COPY_SRC, "seed-Y");
    const baselinePart = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "baseline-PART");
    const candidatePart = make(memory.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "candidate-PART");
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-readback");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");
    if (allocated.size !== Object.keys(memory.gpuTemporaryBuffers).length || actualGpuBufferBytes !== memory.gpuBufferBytes) {
      throw new Error(`GPU temporary total ${actualGpuBufferBytes} bytes across ${allocated.size} buffers; planned ${memory.gpuBufferBytes} bytes across ${Object.keys(memory.gpuTemporaryBuffers).length}`);
    }

    await uploadGenerated(device, input, T * shape.K * 4, 4, (index) => inputValue(index, shape.K), stage, deadline, "X upload");
    await uploadGenerated(device, scales, 16, 4, () => 0, stage, deadline, "BF16 dummy-scale upload");
    await uploadGenerated(device, seedY, memory.outputCount * 4, 4, (index) => seedValue(index, shape.N), stage, deadline, "seed-Y upload");
    await uploadTail(device, seedY, memory.outputCount, deadline, "seed-Y tail init");
    await uploadGenerated(device, baselinePart, memory.partialCount * 4, 4, () => NaN, stage, deadline, "baseline PART NaN init");
    await uploadTail(device, baselinePart, memory.partialCount, deadline, "baseline PART tail init");
    await uploadGenerated(device, candidatePart, memory.partialCount * 4, 4, () => NaN, stage, deadline, "candidate PART NaN init");
    await uploadTail(device, candidatePart, memory.partialCount, deadline, "candidate PART tail init");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));

    const baselineGroup = bind(pipes.matmulLayout, [globals, params, input, weights, scales, bias, baselineY, baselinePart]);
    const candidateGroup = bind(pipes.matmulLayout, [globals, params, input, weights, scales, bias, candidateY, candidatePart]);
    const baselineReduce = bind(pipes.reduceLayout, [globals, params, baselinePart, bias, baselineY]);
    const candidateReduce = bind(pipes.reduceLayout, [globals, params, candidatePart, bias, candidateY]);
    const reduceWork = Math.ceil(T * shape.N / 256);
    const dispatch = (pass, useCandidate) => {
      pass.setPipeline(useCandidate ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, useCandidate ? candidateGroup : baselineGroup);
      pass.dispatchWorkgroups(Math.ceil(shape.N / BN), Math.ceil(T / (ROWS * 16)), splits);
      if (splits > 1) {
        pass.setPipeline(pipes.reducer);
        pass.setBindGroup(0, useCandidate ? candidateReduce : baselineReduce);
        pass.dispatchWorkgroups(Math.min(reduceWork, 65535), Math.ceil(reduceWork / 65535));
      }
    };
    const resetY = (encoder) => {
      encoder.copyBufferToBuffer(seedY, 0, baselineY, 0, memory.outputBytes);
      encoder.copyBufferToBuffer(seedY, 0, candidateY, 0, memory.outputBytes);
    };
    const uploadBias = async (enabled) => {
      await uploadGenerated(device, bias, shape.N * 4, 4, (index) => biasValue(index, enabled), stage, deadline, "bias upload");
    };
    const runPair = async (fixture, phase, mode, hasBias) => {
      device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, mode, hasBias]));
      const encoder = device.createCommandEncoder({ label: `bf16-shared.${shape.shape}.${fixture}.${phase}.m${mode}.b${hasBias}` });
      resetY(encoder);
      const pass = encoder.beginComputePass();
      dispatch(pass, false);
      dispatch(pass, true);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await drain(device, deadline, `${phase} exact dispatch`);
      const output = await compareBuffers(device, read, baselineY, candidateY, memory.outputCount,
        `${shape.shape}.${fixture}.${phase}.mode${mode}.bias${hasBias}.Y`, deadline);
      const partial = await compareBuffers(device, read, baselinePart, candidatePart, memory.partialCount,
        `${shape.shape}.${fixture}.${phase}.mode${mode}.bias${hasBias}.PART`, deadline);
      if (!output.ok || !partial.ok) throw new Error(`${phase} mode${mode}/bias${hasBias} exactness failed`);
      return { mode, bias: Boolean(hasBias), output, partial };
    };
    const guards = { before: [], after: [] };
    const fixtures = full ? WEIGHT_FIXTURES : ["normal"];
    const modes = full ? [0, 1, 2] : [0];
    const biasModes = full ? [0, 1] : [0];
    for (const fixture of fixtures) {
      await uploadGenerated(device, weights, memory.fixtureWeightBytes, 2, (index) => bf16Word(fixture, index), stage, deadline, `${fixture} BF16 fixture upload`);
      for (const phase of ["before", "after"]) {
        for (const hasBias of biasModes) {
          await uploadBias(hasBias === 1);
          for (const mode of modes) guards[phase].push({ fixture, ...(await runPair(fixture, phase, mode, hasBias)) });
        }
        if (phase === "before") {
          device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 1]));
          await uploadBias(true);
          const queryCount = SAMPLES * 2 * 2;
          querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `bf16-shared.${shape.shape}.${fixture}.timestamps` });
          const timingEncoder = device.createCommandEncoder({ label: `bf16-shared.${shape.shape}.${fixture}.paired-timing` });
          const order = [];
          for (let sample = 0; sample < SAMPLES; sample++) {
            for (const useCandidate of (sample % 2 ? [true, false] : [false, true])) {
              resetY(timingEncoder);
              const index = order.length * 2;
              const timingPass = timingEncoder.beginComputePass({ timestampWrites: {
                querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1,
              } });
              for (let repetition = 0; repetition < REPETITIONS; repetition++) dispatch(timingPass, useCandidate);
              timingPass.end();
              order.push(useCandidate);
            }
          }
          timingEncoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
          timingEncoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8);
          device.queue.submit([timingEncoder.finish()]);
          if (!await mapBeforeDeadline(read, queryCount * 8, deadline)) { read.unmap(); throw new Error("paired GPU timestamp readback deadline exceeded"); }
          let timing;
          try {
            const stamps = new BigUint64Array(read.getMappedRange(0, queryCount * 8).slice(0));
            const samples = { baseline: [], candidate: [] };
            order.forEach((useCandidate, index) => {
              const ns = stamps[index * 2 + 1] - stamps[index * 2];
              if (ns <= 0n) throw new Error(`nonpositive timestamp interval at query ${index}`);
              samples[useCandidate ? "candidate" : "baseline"].push(Number(ns) / 1e6 / REPETITIONS);
            });
            const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
            if (![...samples.baseline, ...samples.candidate].every((value) => Number.isFinite(value) && value > 0)) {
              throw new Error("nonpositive or nonfinite paired GPU timestamp");
            }
            timing = { method: "alternating WebGPU timestamp-query, matmul plus unchanged split-K reducer",
              samples: SAMPLES, repetitions: REPETITIONS, ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs };
          } finally { read.unmap(); querySet.destroy(); querySet = null; }
          guards.timing ||= {};
          guards.timing[fixture] = timing;
        }
      }
    }
    const peakControlledTemporaryBytes = actualGpuBufferBytes + memory.timestampQueryBytes + memory.hostStageBytes + memory.mappedHostCopyBytes + memory.queuedUploadAllowanceBytes + memory.fixedHostScratchBytes;
    if (peakControlledTemporaryBytes !== memory.temporaryBytes || peakControlledTemporaryBytes > LIMIT) {
      throw new Error(`${shape.shape} peak controlled temporary memory ${peakControlledTemporaryBytes} differs from plan ${memory.temporaryBytes} or exceeds ${LIMIT}`);
    }
    return { ...shape, T, splitT: T, rows: ROWS, physicalBM: ROWS * 16, row56: false, BN, groups: 1,
      weights: "bf16", f16: false, target: TARGET, splits, dispatch: { workgroupsX: Math.ceil(shape.N / BN), workgroupsY: Math.ceil(T / (ROWS * 16)), splitK: splits },
      sharedTile: { baselineWeightBytes: 512 * 16, candidateWeightBytes: 512 * 8, savedBytesPerWorkgroup: 4096,
        xsBytesUnchanged: 512 * 16, baselineTotalSharedBytes: 16_384, candidateTotalSharedBytes: 12_288,
        unchangedDecode: "bf16(vec2<u32>) bit expansion; F32 products and ordered vec2 accumulator adds" },
      ok: true, guards, sourceHashes: pipes.hashes,
      memory: { ...memory, splits, actualGpuBufferBytes, peakControlledTemporaryBytes } };
  } finally {
    querySet?.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

async function main() {
  const started = performance.now();
  const params = new URLSearchParams(location.search);
  const selectedShape = params.get("shape");
  let shapes = params.has("all") ? SHAPES.filter((shape) => shape.actualVision)
    : selectedShape === "all" ? SHAPES.filter((shape) => shape.primary)
      : selectedShape ? SHAPES.filter((shape) => shape.shape === selectedShape)
        : SHAPES.filter((shape) => shape.primary);
  if (params.has("ragged") && !shapes.some((shape) => shape.shape === "ragged")) {
    shapes = [...shapes, SHAPES.find((shape) => shape.shape === "ragged")];
  }
  if (!shapes.length || shapes.some((shape) => !shape)) throw new Error("shape must be gate-up, down, attention, projection, or all");
  const full = params.has("full");
  const mapping = assertPackedMapping();
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const device = runtime.device;
    metadata = { adapter: { name: runtime.name, features: [...device.features] }, requiredFeature: "timestamp-query",
      geometry: { T, rows: ROWS, physicalBM: ROWS * 16, row56: false, BN, groups: 1, splitT: T, target: TARGET },
      selection: shapes.map(({ shape }) => shape), fullGuardMatrix: full,
      guardScope: full ? "three BF16 fixtures x all modes x both bias settings" : "normal BF16 fixture x mode 0 x bias disabled",
      mapping, temporaryLimitBytes: LIMIT,
      excludedWeightData: "one immutable BF16 fixture buffer shared by both paths, separately reported per shape" };
    if (!device.features.has("timestamp-query")) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-bf16-shared",
        ...metadata, reason: "timestamp-query is unavailable; no wall-clock fallback was run" };
      log(window.gpuBench);
      return;
    }
    const source = await kernelSource(params.get("wasm") || undefined);
    const pipes = await withGpuErrors(runtime, () => compile(device, source), "compile registered and packed BF16 vision matmul plus unchanged reducer");
    metadata = { ...metadata, wasmUrl: source.wasmUrl, sourceHashes: pipes.hashes,
      sourceTransform: "only BF16 shared tile storage, private raw pair staging, matching writer/reader swizzle, and shared-read decode" };
    for (const shape of shapes) {
      const result = await withGpuErrors(runtime, () => checkShape(runtime, pipes, shape, { full }), `check BF16 shared-tile ${shape.shape}`);
      cases.push(result);
      window.gpuBench.completed++;
      log({ status: "shape-done", ...result });
    }
    const timed = cases.flatMap((item) => (full ? WEIGHT_FIXTURES : ["normal"]).map((fixture) => item.guards.timing[fixture]));
    const baselineMetricMs = geometricMean(timed.map((item) => item.baselineMs));
    const metricMs = geometricMean(timed.map((item) => item.candidateMs));
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-bf16-shared", ...metadata,
      metric: "geometric mean candidate GPU timestamp milliseconds across selected shapes and BF16 fixtures",
      metricMs, baselineMetricMs, speedup: baselineMetricMs / metricMs, elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0,
        guards: full ? "all Y values and all split-K PART planes are bit-exact and finite; signed-zero/zero input, normal/tiny-subnormal/large-finite BF16 fixtures, six mode/bias combinations, selected ragged output columns, and four-float tails before and after timing"
          : "all Y values and all split-K PART planes are bit-exact and finite for the normal BF16 fixture, mode 0, bias disabled, and four-float tails before and after timing" }, cases };
    log(window.gpuBench);
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-bf16-shared", completed: 0 };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", check: "gemma4-bf16-shared",
      ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || String(error));
  });
}
