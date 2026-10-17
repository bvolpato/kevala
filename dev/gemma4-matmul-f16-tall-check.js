import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const CHUNK = MiB;
const LIMIT = 64 * MiB;
const T = 97;
const SPLIT_T = 1;
const ROWS = 4;
const LOGICAL_BM = 64;
const BASELINE_BM = 64;
const CANDIDATE_BM = 104;
const BN = 64;
const TARGET = 256;
const TAIL = 4;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const DEADLINE_MS = 60_000;
const SAMPLES = 3;
const REPETITIONS = 4;
const SHAPES = [
  { shape: "down", N: 1536, K: 6144, expectedSplits: 8 },
  { shape: "q2048", N: 2048, K: 1536, expectedSplits: 8 },
];
const cases = [];
let metadata = {};

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const count = (text, marker) => text.split(marker).length - 1;
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");

function replaceExact(code, marker, replacement, expected = 1, label = marker) {
  const found = count(code, marker);
  if (found !== expected) throw new Error(`BM104 source marker mismatch for ${label}: expected ${expected}, found ${found}`);
  return code.replaceAll(marker, replacement);
}

function functionSource(code, name) {
  return code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0] ?? null;
}

function orderedDotSteps(code) {
  return [...code.matchAll(/^\s+acc([0-9]+) \+= vec2<f32>\(dot\(a\1, b0\), dot\(a\1, b1\)\);$/gm)]
    .map(([, accumulator]) => Number(accumulator));
}

export function physicalBm104(generic) {
  if (!generic.includes("enable f16;") || !generic.includes("@workgroup_size(16, 16)") ||
      !generic.includes("var acc: array<vec4<f32>, 4>;") ||
      !generic.includes("let m0 = wg.y * 64u;") ||
      !generic.includes("var<workgroup> xs: array<vec4<f16>, 512>;") ||
      count(generic, "for (var kq = 0u; kq < 8u; kq++) {") !== 1 ||
      count(generic, "for (var i = 0u; i < 4u; i++) {") !== 2) {
    throw new Error("BM104 requires the rendered Q8/F16 R4 baseline kernel");
  }

  let code = generic;
  code = replaceExact(code,
    "// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), with Q8, BF16, or F32 stored weights.",
    "// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), with Q8 stored weights.",
    1, "Q8-only source header");
  code = replaceExact(code,
    "// A workgroup of 16 x 16 threads covers 64 rows (4 per thread) and 64 columns\n// (1 groups of 64); thread (x, y) owns rows y + 16 i and columns x + 16 j + 64 g, so\n// output stores coalesce. Each step stages one quantization block of K (32 values) for X and W\n// in workgroup memory as f16; products and sums stay f32. The weight tile is XOR-swizzled\n// so neighbouring threads read different banks, and every thread writes whole vectors\n// (sub-vector writes from several threads race on some GPUs).",
    "// A workgroup of 32 x 8 threads covers 104 rows and 64 columns. Each thread owns rows\n// y + 8 i and columns x and x + 32, with thirteen named vec2 FP32 accumulators. It keeps\n// the Q8/F16 K32 staging, swizzle, casts, and barriers from the registered single-matmul kernel.",
    1, "tile description");
  code = replaceExact(code, "@compute @workgroup_size(16, 16)", "@compute @workgroup_size(32, 8)", 1, "workgroup geometry");
  code = replaceExact(code, "var<workgroup> xs: array<vec4<f16>, 512>;", "var<workgroup> xs: array<vec4<f16>, 832>;", 1, "activation tile");
  code = replaceExact(code, "let m0 = wg.y * 64u;", "let m0 = wg.y * 104u;", 1, "physical row origin");
  code = replaceExact(code, "  var acc: array<vec4<f32>, 4>; // [i][g]: row y + 16 i, columns x + 16 j + 64 g for j < 4",
    Array.from({ length: 13 }, (_, i) => `  var acc${i} = vec2<f32>(0.0);`).join("\n"), 1, "accumulators");
  code = replaceExact(code,
    "  var xv1 = vec4<f32>(0.0);\n  var wv: array<vec4<f32>, 2>;",
    "  var xv1 = vec4<f32>(0.0);\n  var xv2 = vec4<f32>(0.0);\n  var xv3 = vec4<f32>(0.0);\n  var wv: array<vec4<f32>, 2>;",
    1, "second activation-row registers");
  code = replaceExact(code, "    xv1 = vec4<f32>(0.0);\n",
    "    xv1 = vec4<f32>(0.0);\n    xv2 = vec4<f32>(0.0);\n    xv3 = vec4<f32>(0.0);\n", 1, "activation register reset");
  code = replaceExact(code,
    "    if (lr < 64u && xrow < T) {\n      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;\n      xv0 = X[base];\n      xv1 = X[base + 1u];\n    }",
    `    if (lr < 64u && xrow < T) {\n      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;\n      xv0 = X[base];\n      xv1 = X[base + 1u];\n    }\n    let xrow2 = xrow + 64u;\n    if (lr + 64u < ${CANDIDATE_BM}u && xrow2 < T) {\n      let base2 = (xrow2 * K + kb * 32u) / 4u + lq * 2u;\n      xv2 = X[base2];\n      xv3 = X[base2 + 1u];\n    }`,
    1, "second activation-row load");
  code = replaceExact(code,
    "    if (lr < 64u) {\n      xs[lr * 8u + lq * 2u] = vec4<f16>(xv0);\n      xs[lr * 8u + lq * 2u + 1u] = vec4<f16>(xv1);\n    }",
    `    if (lr < 64u) {\n      xs[lr * 8u + lq * 2u] = vec4<f16>(xv0);\n      xs[lr * 8u + lq * 2u + 1u] = vec4<f16>(xv1);\n    }\n    if (lr + 64u < ${CANDIDATE_BM}u) {\n      xs[(lr + 64u) * 8u + lq * 2u] = vec4<f16>(xv2);\n      xs[(lr + 64u) * 8u + lq * 2u + 1u] = vec4<f16>(xv3);\n    }`,
    1, "second activation-row shared store");
  code = replaceExact(code, "// loaders: 4 threads per row, 8 values each; X rows 0..64, W rows lr + 64 g",
    "// loaders: 4 threads per row, 8 values each; X rows 0..104, W rows lr + 64 g", 1, "loader description");

  const dotStartMarker = "    let nx = lid.x & 7u;\n";
  if (count(code, dotStartMarker) !== 1) throw new Error("F16 BM104 dot schedule marker mismatch");
  const dotStart = code.indexOf(dotStartMarker);
  const dotEnd = code.indexOf("    workgroupBarrier();", dotStart);
  if (dotEnd < 0) throw new Error("F16 BM104 trailing K32 barrier is missing");
  const dots = Array.from({ length: 8 }, (_, kq) => {
    const rows = Array.from({ length: 13 }, (_, i) => {
      const row = 8 * i;
      return `    let a${i} = vec4<f32>(xs[(lid.y + ${row}u) * 8u + ${kq}u]);\n    acc${i} += vec2<f32>(dot(a${i}, b0), dot(a${i}, b1));`;
    }).join("\n");
    return `    {\n      let kq = ${kq}u;\n      let b0 = vec4<f32>(ws[lid.x * 8u + (kq ^ nx)]);\n      let b1 = vec4<f32>(ws[(lid.x + 32u) * 8u + (kq ^ nx)]);\n${rows}\n    }`;
  }).join("\n");
  code = `${code.slice(0, dotStart)}    let nx = lid.x & 7u;\n${dots}\n${code.slice(dotEnd)}`;

  const storeStartMarker = "  for (var gg = 0u; gg < 1u; gg++) {\n    for (var j = 0u; j < 4u; j++) {\n      let col = n0 + lid.x + 16u * j + 64u * gg;";
  if (count(code, storeStartMarker) !== 1) throw new Error("F16 BM104 store schedule marker mismatch");
  const storeStart = code.indexOf(storeStartMarker);
  const storeEnd = code.lastIndexOf("\n}");
  const storeValue = (acc, component, row) => `      mm_store(acc${acc}.${component}, row + ${row}u, col, bias, T, N, splits, wg.z);`;
  const outputBlock = (component, columnOffset) => {
    const stores = Array.from({ length: 13 }, (_, i) => storeValue(i, component, i * 8)).join("\n");
    return `  {\n    let col = n0 + lid.x + ${columnOffset}u;\n    if (col < N) {\n      var bias = 0.0;\n      if (p.bias == 1u) { bias = B[col]; }\n      let row = m0 + lid.y;\n${stores}\n    }\n  }`;
  };
  const storeFunction = `fn mm_store(value: f32, row: u32, col: u32, bias: f32, T: u32, N: u32, splits: u32, z: u32) {\n  if (row >= T) { return; }\n  let o = row * N + col;\n  if (splits > 1u) { PART[z * T * N + o] = value; return; }\n  var v = value + bias;\n  if (p.mode == 1u) { v += Y[o]; } else if (p.mode == 2u) { v = max(v, 0.0); }\n  Y[o] = v;\n}\n\n`;
  code = `${code.slice(0, storeStart)}${outputBlock("x", 0)}\n${outputBlock("y", 32)}${code.slice(storeEnd)}`;
  code = replaceExact(code, "@compute @workgroup_size(32, 8)", `${storeFunction}@compute @workgroup_size(32, 8)`, 1, "F16 store helper");

  if (!code.includes("enable f16;") || !code.includes("@compute @workgroup_size(32, 8)") ||
      !code.includes("var<workgroup> xs: array<vec4<f16>, 832>;") ||
      (code.match(/var acc\d+ = vec2<f32>\(0\.0\);/g) || []).length !== 13 ||
      count(code, "let xrow2 = xrow + 64u;") !== 1 ||
      count(code, "workgroupBarrier();") !== count(generic, "workgroupBarrier();")) {
    throw new Error("F16 BM104 generated tile shape or barriers are invalid");
  }
  const candidateDots = orderedDotSteps(code);
  const expectedDots = Array.from({ length: 8 }, () => Array.from({ length: 13 }, (_, i) => i)).flat();
  if (candidateDots.length !== 104 || JSON.stringify(candidateDots) !== JSON.stringify(expectedDots)) {
    throw new Error("F16 BM104 changed ascending Kq dot/add order or omitted accumulator rows");
  }
  for (const name of ["sx", "mm_splits"]) {
    if (!functionSource(generic, name) || functionSource(generic, name) !== functionSource(code, name)) {
      throw new Error(`F16 BM104 changed the original ${name} function`);
    }
  }
  for (const marker of [
    "let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];",
    "let s = S[wrow * nb + kb];",
    "let lo = select(v.xy, v.zw, (lq & 1u) == 1u);",
    "wv[2u * gg] = sx(lo.x) * s;",
    "wv[2u * gg + 1u] = sx(lo.y) * s;",
    "ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f16>(wv[2u * gg]);",
    "ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f16>(wv[2u * gg + 1u]);",
  ]) {
    if (count(code, marker) !== count(generic, marker)) throw new Error(`F16 BM104 changed Q8 decode, F16 casts, or weight swizzle: ${marker}`);
  }
  if (!code.includes("xs[(lr + 64u) * 8u + lq * 2u] = vec4<f16>(xv2);") ||
      !code.includes("xs[(lr + 64u) * 8u + lq * 2u + 1u] = vec4<f16>(xv3);")) {
    throw new Error("F16 BM104 omitted the second activation tile stores");
  }
  return code;
}

export function memoryPlan(shape, splits) {
  const outputCount = T * shape.N;
  const partialCount = splits > 1 ? splits * outputCount : 0;
  const inputRows = Math.ceil(T / LOGICAL_BM) * LOGICAL_BM;
  const inputBytes = inputRows * shape.K * 4;
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
  return { splits, outputCount, partialCount, inputRows, inputBytes, outputBytes, partBytes,
    immutableWeightBytes, immutableScaleBytes, gpuTemporaryBuffers, gpuTemporaryBytes, hostStageBytes,
    mappedHostCopyBytes, queuedUploadAllowanceBytes, temporaryBytes, limitBytes: LIMIT,
    excludedImmutableQ8WeightAndScaleBytes: true };
}

function assertGridCoverage(T, tile, label) {
  const groups = Math.ceil(T / tile);
  const coverage = new Uint8Array(T);
  for (let group = 0; group < groups; group++) {
    for (let row = group * tile; row < Math.min(T, (group + 1) * tile); row++) coverage[row]++;
  }
  if (coverage.some((count) => count !== 1)) throw new Error(`${label} physical BM${tile} grid does not cover T=${T} exactly once`);
  return groups;
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
    const encoder = device.createCommandEncoder({ label: `bm104.${label}.readback` });
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
  const baselineSpec = { weights: "q8", f16: true, rows: ROWS, row56: false, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul_gemma_batch", baselineSpec);
  const referenceCandidateCode = physicalBm104(baselineCode);
  const candidateCode = source("matmul_tall", baselineSpec);
  const baselineReduceCode = source("reduce_gemma_batch", baselineSpec);
  const referenceCandidateHash = await hash(referenceCandidateCode);
  const registeredCandidateHash = await hash(candidateCode);
  if (registeredCandidateHash !== referenceCandidateHash || candidateCode !== referenceCandidateCode) {
    throw new Error(`registered matmul_tall differs from the verified BM104 reference (registered ${registeredCandidateHash}, reference ${referenceCandidateHash})`);
  }
  if (!baselineCode.includes("enable f16;") || !candidateCode.includes("enable f16;")) {
    throw new Error("BM104 baseline and candidate must both require Q8/F16");
  }
  if (!baselineCode.includes("@compute @workgroup_size(16, 16)") ||
      !baselineCode.includes("var acc: array<vec4<f32>, 4>;") ||
      !baselineCode.includes("let m0 = wg.y * 64u;") ||
      !candidateCode.includes("@compute @workgroup_size(32, 8)") ||
      (candidateCode.match(/var acc\d+ = vec2<f32>\(0\.0\);/g) || []).length !== 13) {
    throw new Error("rendered F16 baseline or transformed R4 source no longer matches the BM104 probe contract");
  }
  for (const name of ["sx", "mm_splits"]) {
    if (functionSource(baselineCode, name) !== functionSource(candidateCode, name)) {
      throw new Error(`F16 baseline/candidate differ in shared ${name} semantics`);
    }
  }
  const matmul = matmulLayout(device);
  const reduce = reduceLayout(device);
  const [baseline, candidate, reducer] = await Promise.all([
    pipeline(device, baselineCode, "gemma_matmul_registered_q8_f16_bm64", device.createPipelineLayout({ bindGroupLayouts: [matmul] })),
    pipeline(device, candidateCode, "gemma_matmul_registered_q8_f16_bm104", device.createPipelineLayout({ bindGroupLayouts: [matmul] })),
    pipeline(device, baselineReduceCode, "gemma_matmul_registered_f16_reduce", device.createPipelineLayout({ bindGroupLayouts: [reduce] })),
  ]);
  return { baseline, candidate, reducer, matmulLayout: matmul, reduceLayout: reduce,
    hashes: { baseline: await hash(baselineCode), candidateReference: referenceCandidateHash,
      candidate: registeredCandidateHash, reducer: await hash(baselineReduceCode) } };
}

async function checkCase(runtime, pipes, shape) {
  const { device } = runtime;
  const splits = mmSplits(SPLIT_T, shape.N, shape.K, TARGET, LOGICAL_BM, BN);
  if (splits !== shape.expectedSplits) throw new Error(`${shape.shape} virtual BM64 split policy changed: expected ${shape.expectedSplits}, got ${splits}`);
  const memory = memoryPlan(shape, splits);
  if (memory.temporaryBytes > LIMIT) throw new Error(`${shape.shape} temporary allocations use ${memory.temporaryBytes} bytes; 64 MiB limit is ${LIMIT}`);
  if (device.limits.maxComputeWorkgroupStorageSize < 10_752) {
    return { ...shape, T, splits, ok: true, skipped: true, reason: "F16 BM104 requires 10,752 bytes of workgroup storage", memory };
  }
  const maxBuffer = device.limits.maxBufferSize;
  const maxBinding = device.limits.maxStorageBufferBindingSize;
  const bindingSizes = [memory.immutableWeightBytes, memory.immutableScaleBytes, memory.inputBytes,
    shape.N * 4, memory.outputBytes, memory.partBytes, 2 * CHUNK];
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
    const buffer = device.createBuffer({ size: actualSize, usage, label: `gemma-matmul-f16-bm104.${shape.shape}.${label}` });
    owned.push(buffer);
    return buffer;
  };
  const bind = (layout, buffers) => device.createBindGroup({
    layout,
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });

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
      (index) => {
        const row = Math.floor(index / shape.K), k = index % shape.K;
        if (row >= T) return NaN;
        if (index % 67 === 0) return index % 2 ? -0 : 0;
        return 0.04 * Math.sin(k * 0.127 + row * 0.13 + 0.3);
      }, stage, deadline, "activation upload");
    await fillWithTail(device, freshYSeed, memory.outputCount, () => NaN, stage, deadline, "fresh-Y seed upload");
    await fillWithTail(device, accumulateYSeed, memory.outputCount,
      (index) => (index % 23 - 11) * 0.0037, stage, deadline, "accumulate-Y seed upload");
    await fillWithTail(device, partSeed, memory.partialCount, () => NaN, stage, deadline, "PART seed upload");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, SPLIT_T, SPLIT_T, 0]));

    const baselineGroup = bind(pipes.matmulLayout, [globals, params, input, weights, scales, bias, baselineY, baselinePart]);
    const candidateGroup = bind(pipes.matmulLayout, [globals, params, input, weights, scales, bias, candidateY, candidatePart]);
    const baselineReduce = bind(pipes.reduceLayout, [globals, params, baselinePart, bias, baselineY]);
    const candidateReduce = bind(pipes.reduceLayout, [globals, params, candidatePart, bias, candidateY]);
    const work = Math.ceil(T * shape.N / 256);
    const rowGroups = { baseline: assertGridCoverage(T, BASELINE_BM, "baseline"), candidate: assertGridCoverage(T, CANDIDATE_BM, "candidate") };
    const dispatch = (pass, candidate) => {
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, candidate ? candidateGroup : baselineGroup);
      pass.dispatchWorkgroups(Math.ceil(shape.N / BN), rowGroups[candidate ? "candidate" : "baseline"], splits);
      if (splits > 1) {
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
    const runPair = async (mode, hasBias, phase) => {
      device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, mode, hasBias]));
      for (let index = 0; index < shape.N; index++) stageF32[index] = hasBias ? (index % 7 - 3) * 0.0137 : NaN;
      device.queue.writeBuffer(bias, 0, stage, 0, shape.N * 4);
      const encoder = device.createCommandEncoder({ label: `gemma-matmul-f16-bm104.${shape.shape}.${phase}.mode${mode}.bias${hasBias}` });
      resetOutputs(mode, encoder);
      const pass = encoder.beginComputePass();
      dispatch(pass, false);
      dispatch(pass, true);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await drain(device, deadline, `${phase} dispatch`);
      return compareCurrent(mode, hasBias, phase);
    };
    const runGuards = async (phase) => {
      const guards = [];
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) guards.push(await runPair(mode, hasBias, phase));
      return guards;
    };

    const before = await runGuards("before");
    device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 0]));
    stageF32.fill(NaN, 0, shape.N);
    device.queue.writeBuffer(bias, 0, stage, 0, shape.N * 4);
    const warm = device.createCommandEncoder({ label: `gemma-matmul-f16-bm104.${shape.shape}.warmup` });
    resetOutputs(0, warm);
    const warmPass = warm.beginComputePass();
    dispatch(warmPass, false);
    dispatch(warmPass, true);
    warmPass.end();
    device.queue.submit([warm.finish()]);
    await drain(device, deadline, "timing warmup");

    const timing = { method: "paired WebGPU timestamp-query; registered Q8/F16 BM64 and registered BM104 matmul plus the same reducer",
      samples: SAMPLES, repetitions: REPETITIONS, baselineSamples: [], candidateSamples: [] };
    if (device.features.has("timestamp-query")) {
      querySet = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4, label: `gemma-matmul-f16-bm104.${shape.shape}.timestamps` });
      const timingEncoder = device.createCommandEncoder({ label: `gemma-matmul-f16-bm104.${shape.shape}.paired-timing` });
      const order = [];
      let queryIndex = 0;
      for (let sample = 0; sample < SAMPLES; sample++) {
        const variants = sample % 2 ? [true, false] : [false, true];
        for (const candidate of variants) {
          resetOutputs(0, timingEncoder);
          const at = queryIndex;
          order.push(candidate);
          const pass = timingEncoder.beginComputePass({ timestampWrites: {
            querySet, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1,
          } });
          for (let repetition = 0; repetition < REPETITIONS; repetition++) dispatch(pass, candidate);
          pass.end();
          queryIndex += 2;
        }
      }
      timingEncoder.resolveQuerySet(querySet, 0, SAMPLES * 4, resolved, 0);
      timingEncoder.copyBufferToBuffer(resolved, 0, read, 0, SAMPLES * 4 * 8);
      device.queue.submit([timingEncoder.finish()]);
      if (!await mapBeforeDeadline(read, SAMPLES * 4 * 8, deadline)) {
        read.unmap();
        throw new Error("paired timing readback deadline exceeded");
      }
      let timestamps;
      try { timestamps = new BigInt64Array(read.getMappedRange(0, SAMPLES * 4 * 8).slice(0)); }
      finally { read.unmap(); }
      for (let sample = 0; sample < order.length; sample++) {
        const elapsedNs = timestamps[2 * sample + 1] - timestamps[2 * sample];
        if (elapsedNs <= 0n) throw new Error(`nonpositive timestamp interval at ${sample}`);
        const milliseconds = Number(elapsedNs) / 1e6 / REPETITIONS;
        (order[sample] ? timing.candidateSamples : timing.baselineSamples).push(milliseconds);
      }
      if (![...timing.baselineSamples, ...timing.candidateSamples].every((value) => Number.isFinite(value) && value > 0)) {
        throw new Error("nonpositive or nonfinite paired GPU timestamp");
      }
      timing.baselineMs = median(timing.baselineSamples);
      timing.candidateMs = median(timing.candidateSamples);
      timing.speedup = timing.baselineMs / timing.candidateMs;
    } else {
      timing.skipped = true;
      timing.reason = "timestamp-query is unavailable; no wall-clock fallback was run";
    }
    const after = await runGuards("after");
    return { ...shape, T, splitT: SPLIT_T, rows: ROWS, baselineRow56: false, baselinePhysicalBM: BASELINE_BM,
      candidateRow56: false, candidatePhysicalBM: CANDIDATE_BM, logicalBM: LOGICAL_BM, f16: true, weights: "q8", splits,
      dispatch: { workgroupsX: Math.ceil(shape.N / BN), baselineY: rowGroups.baseline, candidateY: rowGroups.candidate,
        splitK: splits, logicalBM: LOGICAL_BM, BN },
      workgroupStorageBytes: { activation: 832 * 8, weights: 512 * 8, total: 10_752 },
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
  if (Number(params.get("T") ?? T) !== T) throw new Error("this F16 BM104 probe is restricted to T=97");
  if (Math.ceil(T / BASELINE_BM) !== 2 || Math.ceil(T / CANDIDATE_BM) !== 1) {
    throw new Error("T=97 no longer compares two BM64 groups with one BM104 group");
  }
  const selected = SHAPES.filter((shape) => params.has("shape")
    ? shape.shape === params.get("shape")
    : params.has("all") || ["down", "q2048"].includes(shape.shape));
  if (!selected.length) throw new Error("shape must be down or q2048");
  const runtime = await requestDevice({ features: ["shader-f16", "timestamp-query"] });
  try {
    const { device } = runtime;
    metadata = { adapter: { name: runtime.name, features: [...device.features], maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize },
      wasmUrl: null, T, splitT: SPLIT_T, rows: ROWS, baselineRow56: false, baselinePhysicalBM: BASELINE_BM,
      candidatePhysicalBM: CANDIDATE_BM, logicalBM: LOGICAL_BM, f16: true, weights: "q8", target: TARGET,
      requiredFeature: "shader-f16", temporaryLimitBytes: LIMIT, immutableQ8WeightsAndScalesExcluded: true };
    if (!device.features.has("shader-f16")) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-matmul-f16-bm104",
        ...metadata, reason: "shader-f16 is unavailable; this probe has no F32 fallback" };
      log(window.gpuBench);
      return;
    }
    if (!device.features.has("timestamp-query")) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-matmul-f16-bm104",
        ...metadata, reason: "timestamp-query is required for paired GPU timing; no wall-clock fallback was run" };
      log(window.gpuBench);
      return;
    }
    if (device.limits.maxComputeWorkgroupStorageSize < 10_752) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-matmul-f16-bm104",
        ...metadata, reason: "F16 BM104 needs 10,752 bytes of workgroup storage" };
      log(window.gpuBench);
      return;
    }
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(device, source), "building registered Q8/F16 BM64 and BM104 pipelines");
    metadata.sourceHashes = pipes.hashes;
    for (const shape of selected) {
      const splits = mmSplits(SPLIT_T, shape.N, shape.K, TARGET, LOGICAL_BM, BN);
      const memory = memoryPlan(shape, splits);
      log({ status: "checking", ...shape, T, splitT: SPLIT_T, logicalBM: LOGICAL_BM,
        baselinePhysicalBM: BASELINE_BM, candidatePhysicalBM: CANDIDATE_BM, splits, memory });
      const result = await withGpuErrors(runtime, () => checkCase(runtime, pipes, shape), `${shape.shape} exactness and paired timing`);
      cases.push(result);
      window.gpuBench.completed = cases.length;
      log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const timed = cases.filter((item) => item.timing?.candidateMs > 0);
    const baselineMetricMs = timed.length ? geometricMean(timed.map((item) => item.timing.baselineMs)) : null;
    const metricMs = timed.length ? geometricMean(timed.map((item) => item.timing.candidateMs)) : null;
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matmul-f16-bm104",
      ...metadata, metric: "geometric mean candidate GPU timestamp milliseconds across selected Q8/F16 T=97 shapes",
      metricMs, baselineMetricMs, speedup: metricMs ? baselineMetricMs / metricMs : null,
      elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0,
        guards: "all Y values and PART planes plus four-float tails, NaN activation padding, and six mode/bias combinations before and after timing", cases } };
    log(window.gpuBench);
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matmul-f16-bm104", completed: 0 };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", check: "gemma4-matmul-f16-bm104",
      ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || String(error));
  });
}
