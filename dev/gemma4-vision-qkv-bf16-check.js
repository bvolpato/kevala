// DEV-only fused Q/K/V BF16 matmul probe. The three exact reducers stay unchanged.
import { matmulLayout, mmSplits, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const LIMIT = 64 * MiB;
const CHUNK = MiB;
const TAIL = 4;
const T_DEFAULT = 108;
const N = 768;
const K = 768;
const HEADS = 12;
const DIM = 64;
const ROWS = 4;
const BM_VIRTUAL = 64;
const BM_PHYSICAL = 56;
const BN = 64;
const TARGET = 256;
const SAMPLES = 3;
const REPETITIONS = 4;
const DEADLINE_MS = 120_000;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(Float32Array.of(SENTINEL).buffer)[0];
const SPEC = { weights: "bf16", f16: false, rows: ROWS, row56: true, groups: 1, splitTarget: TARGET };
const WEIGHT_FIXTURES = ["normal", "tiny-subnormal", "large-finite"];
const INPUT_FIXTURES = ["normal", "signed-zero", "cancellation"];
const PROJECTIONS = ["q", "k", "v"];
const cases = [];
let metadata = {};

const log = (value) => { document.getElementById("log").textContent += JSON.stringify(value) + "\n"; };
const count = (text, marker) => text.split(marker).length - 1;
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b), mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))]
  .map((byte) => byte.toString(16).padStart(2, "0")).join("");

function replaceExact(code, marker, replacement, expected, label) {
  const found = count(code, marker);
  if (found !== expected) throw new Error("QKV source marker mismatch for " + label + ": expected " + expected + ", found " + found);
  return code.replaceAll(marker, replacement);
}

function functionSource(code, name) {
  return code.match(new RegExp("fn " + name + "\\([\\s\\S]*?\\n\\}"))?.[0] ?? null;
}

function orderedDots(code, prefix = "") {
  const pattern = new RegExp("^\\s*" + prefix + "acc(\\d+) \\+= vec2<f32>\\(dot\\(a(\\d+), b0" + prefix +
    "\\), dot\\(a\\2, b1" + prefix + "\\)\\);$", "gm");
  return [...code.matchAll(pattern)].map(([, accumulator, row]) => accumulator + ":" + row);
}

export function deriveTriplePackedQkv(baseline) {
  const required = [
    "@compute @workgroup_size(32, 8)",
    "let m0 = wg.y * 56u;",
    "var<workgroup> xs: array<vec4<f32>, 512>;",
    "var<workgroup> ws: array<vec4<f32>, 512>;",
    "let kb0 = wg.z * per;",
    "fn bf16(w: vec2<u32>) -> vec4<f32>",
  ];
  if (baseline.includes("enable f16;") || required.some((marker) => !baseline.includes(marker)) ||
      count(baseline, "let sw = lr & 7u;") !== 1 || count(baseline, "let nx = lid.x & 7u;") !== 1 ||
      orderedDots(baseline).length !== 56 || count(baseline, "mm_store(") !== 15 || count(baseline, "workgroupBarrier();") !== 2) {
    throw new Error("triple QKV requires registered BF16/F32 R4 BM56/virtual64 BN64 with eight K32 steps");
  }

  let code = replaceExact(baseline,
    "// Y[T, N] = X[T, K] . W[N, K]^T (+ bias), with Q8, BF16, or F32 stored weights.",
    "// DEV-only triple BF16 vision Q/K/V path; registered FP32 arithmetic and split order.", 1, "specialized probe header");
  const bindings = "@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;\n" +
    "@group(0) @binding(4) var<storage, read> S: array<f32>;\n" +
    "@group(0) @binding(5) var<storage, read> B: array<f32>;\n" +
    "@group(0) @binding(6) var<storage, read_write> Y: array<f32>;\n" +
    "@group(0) @binding(7) var<storage, read_write> PART: array<f32>;";
  const tripleBindings = "@group(0) @binding(3) var<storage, read> WQ: array<vec4<u32>>;\n" +
    "@group(0) @binding(4) var<storage, read> WK: array<vec4<u32>>;\n" +
    "@group(0) @binding(5) var<storage, read> WV: array<vec4<u32>>;\n" +
    "@group(0) @binding(6) var<storage, read_write> PART: array<f32>;";
  code = replaceExact(code, bindings, tripleBindings, 1, "Q/K/V storage bindings");
  code = replaceExact(code, "var<workgroup> ws: array<vec4<f32>, 512>;",
    "var<workgroup> wsQ: array<vec2<u32>, 512>;\nvar<workgroup> wsK: array<vec2<u32>, 512>;\nvar<workgroup> wsV: array<vec2<u32>, 512>;",
    1, "three raw shared BF16 tiles");
  code = replaceExact(code, "let sw = lr & 7u;", "let sw = (lr >> 1u) & 7u;", 1, "packed shared producer swizzle");
  code = replaceExact(code, "let nx = lid.x & 7u;", "let nx = (lid.x >> 1u) & 7u;", 1, "packed shared consumer swizzle");
  code = replaceExact(code, "var wv: array<vec4<f32>, 2>;",
    "var wqv: array<vec2<u32>, 2>;\n  var wkv: array<vec2<u32>, 2>;\n  var wvv: array<vec2<u32>, 2>;",
    1, "private packed-word staging");

  const floatLoad = "      wv[2u * gg] = vec4<f32>(0.0);\n" +
    "      wv[2u * gg + 1u] = vec4<f32>(0.0);\n" +
    "      if (wrow < N) {\n" +
    "        let v = W[(wrow * K + kb * 32u) / 8u + lq];\n" +
    "        wv[2u * gg] = bf16(v.xy);\n" +
    "        wv[2u * gg + 1u] = bf16(v.zw);\n" +
    "      }";
  const packedLoad = "      wqv[2u * gg] = vec2<u32>(0u);\n" +
    "      wqv[2u * gg + 1u] = vec2<u32>(0u);\n" +
    "      wkv[2u * gg] = vec2<u32>(0u);\n" +
    "      wkv[2u * gg + 1u] = vec2<u32>(0u);\n" +
    "      wvv[2u * gg] = vec2<u32>(0u);\n" +
    "      wvv[2u * gg + 1u] = vec2<u32>(0u);\n" +
    "      if (wrow < N) {\n" +
    "        let wi = (wrow * K + kb * 32u) / 8u + lq;\n" +
    "        let q = WQ[wi];\n" +
    "        let k = WK[wi];\n" +
    "        let v = WV[wi];\n" +
    "        wqv[2u * gg] = q.xy;\n" +
    "        wqv[2u * gg + 1u] = q.zw;\n" +
    "        wkv[2u * gg] = k.xy;\n" +
    "        wkv[2u * gg + 1u] = k.zw;\n" +
    "        wvv[2u * gg] = v.xy;\n" +
    "        wvv[2u * gg + 1u] = v.zw;\n" +
    "      }";
  code = replaceExact(code, floatLoad, packedLoad, 1, "three BF16 weight loads");

  const sharedLoad = "      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(wv[2u * gg]);\n" +
    "      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(wv[2u * gg + 1u]);";
  const sharedTriple = "      wsQ[r * 8u + ((lq * 2u) ^ sw)] = wqv[2u * gg];\n" +
    "      wsQ[r * 8u + ((lq * 2u + 1u) ^ sw)] = wqv[2u * gg + 1u];\n" +
    "      wsK[r * 8u + ((lq * 2u) ^ sw)] = wkv[2u * gg];\n" +
    "      wsK[r * 8u + ((lq * 2u + 1u) ^ sw)] = wkv[2u * gg + 1u];\n" +
    "      wsV[r * 8u + ((lq * 2u) ^ sw)] = wvv[2u * gg];\n" +
    "      wsV[r * 8u + ((lq * 2u + 1u) ^ sw)] = wvv[2u * gg + 1u];";
  code = replaceExact(code, sharedLoad, sharedTriple, 1, "three shared BF16 tile stores");

  for (const name of ["b0", "b1"]) {
    const col = name === "b0" ? "c" : "(c + 32u)";
    const old = "let " + name + " = vec4<f32>(ws[" + col + " * 8u + (kq ^ nx)]);";
    const replacement = ["q", "k", "v"].map((projection) =>
      "let " + name + projection + " = bf16(ws" + projection.toUpperCase() + "[" + col + " * 8u + (kq ^ nx)]);"
    ).join("\n      ");
    code = replaceExact(code, old, replacement, 8, name + " raw BF16 reads");
  }

  for (let index = 0; index < 7; index++) {
    code = replaceExact(code, "var acc" + index + " = vec2<f32>(0.0);",
      "var qacc" + index + " = vec2<f32>(0.0);\n  var kacc" + index + " = vec2<f32>(0.0);\n  var vacc" + index + " = vec2<f32>(0.0);",
      1, "accumulator family " + index);
  }
  const dotPattern = /^(\s*)acc(\d+) \+= vec2<f32>\(dot\(a(\d+), b0\), dot\(a\3, b1\)\);$/gm;
  let dotCount = 0;
  code = code.replace(dotPattern, (_line, indent, index, row) => {
    dotCount++;
    return ["q", "k", "v"].map((projection) => indent + projection + "acc" + index +
      " += vec2<f32>(dot(a" + row + ", b0" + projection + "), dot(a" + row + ", b1" + projection + "));").join("\n");
  });
  if (dotCount !== 56) throw new Error("expected 56 registered dot/add statements, found " + dotCount);

  const oldStore = functionSource(code, "mm_store");
  const newStore = "fn qkv_store(value: f32, row: u32, col: u32, T: u32, N: u32, split: u32, splits: u32, projection: u32) {\n" +
    "  if (row >= T) { return; }\n" +
    "  PART[(projection * splits + split) * T * N + row * N + col] = value;\n}";
  if (!oldStore) throw new Error("registered partial store helper was not found");
  code = replaceExact(code, oldStore, newStore, 1, "partial-only Q/K/V store helper");
  code = replaceExact(code, "      var bias = 0.0;\n      if (p.bias == 1u) { bias = B[col]; }\n", "", 2, "unbiased vision epilogue");
  const storePattern = /^(\s*)mm_store\((acc\d+)\.([xy]), (row \+ \d+u), col, bias, T, N, splits, wg\.z\);$/gm;
  let storeCount = 0;
  code = code.replace(storePattern, (_line, indent, index, lane, row) => {
    storeCount++;
    return ["q", "k", "v"].map((projection) => indent + "qkv_store(" + projection + index + "." + lane +
      ", " + row + ", col, T, N, wg.z, splits, " + (projection === "q" ? "0u" : projection === "k" ? "1u" : "2u") + ");").join("\n");
  });
  if (storeCount !== 14) throw new Error("expected 14 registered BM56 store statements, found " + storeCount);
  if (/\b(?:S|B|Y)\s*\[/.test(code) || /@binding\([45]\) var<storage, read> S:/.test(code) || code.includes("enable f16;")) {
    throw new Error("triple QKV source retained a generic scale, bias, Y, or F16 path");
  }
  if (!code.includes("if (lr < 56u && xrow < T) {") || !code.includes("let m0 = wg.y * 56u;")) {
    throw new Error("triple QKV changed the registered padded-X row guard or BM56 output grid");
  }
  if (count(code, "workgroupBarrier();") !== 2 || !code.includes("workgroupBarrier();\n    let nx = (lid.x >> 1u) & 7u;") ||
      !code.includes("PART[(projection * splits + split) * T * N + row * N + col] = value;")) {
    throw new Error("triple QKV changed the two registered barriers or partial output mapping");
  }
  if ([ "q", "k", "v" ].some((projection) => JSON.stringify(orderedDots(code, projection)) !== JSON.stringify(orderedDots(baseline)))) {
    throw new Error("triple QKV changed a projection's K32 dot or accumulator add order");
  }
  if (count(code, "qkv_store(") !== 43) throw new Error("triple QKV must retain exactly 42 output stores and one helper");
  if (functionSource(code, "mm_store") || code.includes("mm_store(")) throw new Error("generic Y/bias store path remains in the partial-only triple QKV shader");
  for (const name of ["bf16", "mm_splits"]) {
    if (!functionSource(baseline, name) || functionSource(code, name) !== functionSource(baseline, name)) {
      throw new Error("triple QKV changed the registered " + name + " helper");
    }
  }
  if (count(code, "var<workgroup> xs: array<vec4<f32>, 512>;") !== 1 ||
      count(code, "var<workgroup> wsQ: array<vec2<u32>, 512>;") !== 1 ||
      count(code, "var<workgroup> wsK: array<vec2<u32>, 512>;") !== 1 ||
      count(code, "var<workgroup> wsV: array<vec2<u32>, 512>;") !== 1) {
    throw new Error("triple QKV shared tile size or type changed");
  }
  return code;
}

export function assertQkvMapping(T, splits) {
  const columns = new Set(), rows = new Set(), rowWrites = new Uint8Array(T), splitBlocks = new Set(), projections = new Set();
  for (let x = 0; x < Math.ceil(N / BN); x++) for (let lane = 0; lane < 32; lane++) {
    columns.add(x * BN + lane);
    columns.add(x * BN + lane + 32);
  }
  for (let y = 0; y < Math.ceil(T / BM_VIRTUAL); y++) for (let lane = 0; lane < 8; lane++) for (let slot = 0; slot < 7; slot++) {
    const row = y * BM_PHYSICAL + lane + slot * 8;
    if (row < T) { rows.add(row); rowWrites[row]++; }
  }
  if (rowWrites.some((writes) => writes !== 1)) throw new Error("BM56 output mapping does not write each active row exactly once");
  const blocks = K / 32, per = Math.ceil(blocks / splits);
  for (let split = 0; split < splits; split++) for (let block = split * per; block < Math.min(blocks, split * per + per); block++) {
    if (splitBlocks.has(block)) throw new Error("K32 block " + block + " is covered by overlapping split ranges");
    splitBlocks.add(block);
  }
  for (let split = 0; split < splits; split++) projections.add("q:" + split).add("k:" + split).add("v:" + split);
  if (columns.size !== N || rows.size !== T || splitBlocks.size !== blocks || projections.size !== splits * 3) {
    throw new Error("QKV mapping proof failed: columns=" + columns.size + "/" + N + ", rows=" + rows.size + "/" + T +
      ", K32=" + splitBlocks.size + "/" + blocks + ", projection/splits=" + projections.size + "/" + splits * 3);
  }
  const slots = new Set();
  for (let row = 0; row < 64; row++) for (let q = 0; q < 8; q++) slots.add(row * 8 + (q ^ ((row >> 1) & 7)));
  if (slots.size !== 512) throw new Error("raw BF16 shared writer mapping is not bijective");
  for (let col = 0; col < 32; col++) if (((col >> 1) & 7) !== (((col + 32) >> 1) & 7)) {
    throw new Error("paired output columns use different raw-BF16 swizzles");
  }
  const banks = new Set();
  for (let col = 0; col < 32; col++) {
    const slot = 3 ^ ((col >> 1) & 7), first = (col * 16 + slot * 2) & 31;
    banks.add(first); banks.add((first + 1) & 31);
  }
  if (banks.size !== 32) throw new Error("packed BF16 shared reads do not cover all 32 banks at a fixed K32 step");
  return { columns: columns.size, rows: rows.size, k32Blocks: splitBlocks.size, projectionSplits: projections.size,
    writerSlots: slots.size, producerSwizzle: "(lr >> 1) & 7", consumerSwizzle: "(lid.x >> 1) & 7",
    physicalWorkgroups: Math.ceil(N / BN) * Math.ceil(T / BM_VIRTUAL) * splits,
    baselinePhysicalWorkgroups: 3 * Math.ceil(N / BN) * Math.ceil(T / BM_VIRTUAL) * splits,
    projectionsComputedPerWorkgroup: 3, perSplitK32Blocks: per, packedReadBanks: banks.size,
    splitOrder: "increasing split index then increasing K32 block" };
}

function bf16Word(pattern, projection, index) {
  const at = index + projection * 7919;
  const sign = ((Math.floor(at / 11) + projection) & 1) ? 0x8000 : 0;
  if (pattern === "tiny-subnormal") return sign | (1 + ((at * 29 + 7) % 0x7f));
  if (pattern === "large-finite") return sign | ((134 + (at % 4)) << 7) | ((at * 37 + 11) & 0x7f);
  return sign | ((122 + (at % 5)) << 7) | ((at * 37 + 11) & 0x7f);
}

function inputValue(pattern, index) {
  const row = Math.floor(index / K), col = index % K;
  if (pattern === "signed-zero") return (index & 1) ? -0 : 0;
  if (pattern === "cancellation") return Math.fround((col & 1 ? -1 : 1) * (0.25 + (col % 7) * 0.015625) + ((row + col) % 5) * 0.00390625);
  return Math.fround(Math.sin(col * 0.017 + row * 0.13 + 0.23) * 0.37 + Math.cos(col * 0.031 - row * 0.07) * 0.19);
}

function normWeight(projection, index) {
  if (projection === 2) return NaN;
  const sign = ((index + projection) & 1) ? -1 : 1;
  return Math.fround(sign * (0.71 + ((index * (projection ? 7 : 11) + 3) % 23) * 0.037));
}

function ropeValue(row, col) {
  const quarter = DIM / 4, axis = Math.floor(col / (DIM / 2)), part = col % (DIM / 2), pair = Math.floor(part / 2);
  const position = axis === 0 ? row % 12 : Math.floor(row / 12);
  const inverse = Math.fround(1 / Math.fround(Math.pow(10000, pair / quarter)));
  const angle = Math.fround(position * inverse);
  return Math.fround(part % 2 === 0 ? Math.cos(angle) : Math.sin(angle));
}

export function qkvMemoryPlan(T, N, K, sourceBytes = 0) {
  if (![T, N, K].every((value) => Number.isSafeInteger(value) && value > 0) || N % 64 || K % 32 || T < 64) {
    throw new Error("QKV probe requires T>=64, N divisible by64, and K divisible by32");
  }
  const inputRows = Math.ceil(T / BM_PHYSICAL) * BM_PHYSICAL;
  if (Math.ceil(T / BM_PHYSICAL) !== Math.ceil(T / BM_VIRTUAL)) throw new Error("BM56 would change the registered virtual BM64 grid");
  const splits = mmSplits(T, N, K, TARGET, BM_PHYSICAL, BN), outputFloats = T * N, partialFloats = splits * outputFloats;
  if (splits < 2) throw new Error("QKV dispatch-reuse probe requires the registered split-K path");
  const outputBytes = (outputFloats + TAIL) * 4, partBytes = (partialFloats + TAIL) * 4;
  const gpuBuffers = {
    globals: 16, params: 16, X: (inputRows * K + TAIL) * 4, S: 16, B: (N + TAIL) * 4,
    baselinePartQ: partBytes, baselinePartK: partBytes, baselinePartV: partBytes,
    candidatePart: (3 * partialFloats + TAIL) * 4,
    baselineYQ: outputBytes, baselineYK: outputBytes, baselineYV: outputBytes,
    candidateYQ: outputBytes, candidateYK: outputBytes, candidateYV: outputBytes,
    normQ: (DIM + TAIL) * 4, normK: (DIM + TAIL) * 4, normV: (DIM + TAIL) * 4,
    rope: (T * DIM + TAIL) * 4,
    reduceParamsQ: 48, reduceParamsK: 48, reduceParamsV: 48,
    readback: 2 * CHUNK, timestampResolve: 256,
  };
  const gpuTemporaryBytes = Object.values(gpuBuffers).reduce((sum, bytes) => sum + bytes, 0);
  const immutableWeightBytesEach = N * K * 2 + 16;
  const immutableWeightBytes = 3 * immutableWeightBytesEach;
  const stageBytes = CHUNK, mappedCopyBytes = 2 * CHUNK, queuedUploadAllowanceBytes = CHUNK;
  const fixedHostBytes = sourceBytes + 65536, querySetBytes = 12 * 8;
  const controlledTemporaryBytes = gpuTemporaryBytes + stageBytes + mappedCopyBytes + queuedUploadAllowanceBytes + fixedHostBytes + querySetBytes;
  return { T, N, K, splits, inputRows, outputFloats, partialFloats, outputBytes, partBytes, gpuBuffers, gpuTemporaryBytes,
    immutableWeightBytesEach, immutableWeightBytes, stageBytes, mappedCopyBytes, queuedUploadAllowanceBytes,
    fixedHostBytes, querySetBytes, controlledTemporaryBytes, limitBytes: LIMIT };
}

async function drain(device, deadline, label) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error(label + " deadline exceeded");
  let timer;
  try {
    await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(label + " deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

async function fillBuffer(device, target, logicalBytes, valueAt, stage, deadline, label) {
  const bytes = new Uint8Array(stage), floats = new Float32Array(stage);
  for (let offset = 0; offset < logicalBytes; offset += CHUNK) {
    const length = Math.min(CHUNK, logicalBytes - offset);
    if (length % 4) throw new Error(label + " upload chunk is not F32-aligned");
    const first = offset / 4;
    for (let i = 0; i < length / 4; i++) floats[i] = valueAt(first + i);
    device.queue.writeBuffer(target, offset, bytes, 0, length);
    await drain(device, deadline, label);
  }
}

async function fillWeights(device, target, weightBytes, projection, pattern, stage, deadline) {
  const bytes = new Uint8Array(stage), words = new Uint16Array(stage);
  for (let offset = 0; offset < weightBytes; offset += CHUNK) {
    const length = Math.min(CHUNK, weightBytes - offset);
    if (length % 2) throw new Error("BF16 weight upload chunk is not word-aligned");
    const first = offset / 2;
    for (let i = 0; i < length / 2; i++) words[i] = bf16Word(pattern, projection, first + i);
    device.queue.writeBuffer(target, offset, bytes, 0, length);
    await drain(device, deadline, "BF16 fixture upload");
  }
  device.queue.writeBuffer(target, weightBytes, new Uint32Array([0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef]));
  await drain(device, deadline, "BF16 fixture tail");
}

async function uploadFloatTail(device, buffer, logicalFloats, deadline, label) {
  device.queue.writeBuffer(buffer, logicalFloats * 4, new Float32Array(TAIL).fill(SENTINEL));
  await drain(device, deadline, label);
}

async function compareRanges(device, read, left, leftOffset, right, rightOffset, logicalFloats, label, deadline) {
  const floatsPerChunk = CHUNK / 4;
  let zeros = 0, negativeZeros = 0;
  for (let offset = 0; offset < logicalFloats; offset += floatsPerChunk) {
    const countFloats = Math.min(floatsPerChunk, logicalFloats - offset), byteLength = countFloats * 4;
    const encoder = device.createCommandEncoder({ label: "qkv." + label + ".readback" });
    encoder.copyBufferToBuffer(left, leftOffset + offset * 4, read, 0, byteLength);
    encoder.copyBufferToBuffer(right, rightOffset + offset * 4, read, byteLength, byteLength);
    device.queue.submit([encoder.finish()]);
    if (!await mapBeforeDeadline(read, byteLength * 2, deadline)) { read.unmap(); throw new Error(label + " readback deadline exceeded"); }
    try {
      const copy = read.getMappedRange(0, byteLength * 2).slice(0), values = new Float32Array(copy), bits = new Uint32Array(copy);
      for (let i = 0; i < countFloats; i++) {
        if (!Number.isFinite(values[i]) || !Number.isFinite(values[countFloats + i])) throw new Error(label + " nonfinite active value at " + (offset + i));
        if (bits[i] !== bits[countFloats + i]) throw new Error(label + " raw-bit mismatch at " + (offset + i) + ": 0x" + bits[i].toString(16) + " vs 0x" + bits[countFloats + i].toString(16));
        if (values[i] === 0) zeros++;
        if (bits[i] === 0x80000000) negativeZeros++;
      }
    } finally { read.unmap(); }
  }
  return { ok: true, checkedFloats: logicalFloats, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros };
}

async function assertTails(device, read, entries, deadline) {
  const encoder = device.createCommandEncoder({ label: "qkv sentinel tails" });
  entries.forEach(({ buffer, offset }, index) => encoder.copyBufferToBuffer(buffer, offset, read, index * 16, 16));
  device.queue.submit([encoder.finish()]);
  const byteLength = entries.length * 16;
  if (!await mapBeforeDeadline(read, byteLength, deadline)) { read.unmap(); throw new Error("QKV tail sentinel readback deadline exceeded"); }
  try {
    const bits = new Uint32Array(read.getMappedRange(0, byteLength).slice(0));
    for (let index = 0; index < entries.length * TAIL; index++) if (bits[index] !== SENTINEL_BITS) {
      throw new Error(entries[Math.floor(index / TAIL)].label + " tail sentinel changed at " + (index % TAIL));
    }
  } finally { read.unmap(); }
  return { checkedBuffers: entries.length, floatsPerTail: TAIL, sentinel: SENTINEL };
}

async function compileProbe(device, source, registered = false) {
  const baselineCode = source("matmul", SPEC), derived = deriveTriplePackedQkv(baselineCode);
  const candidateCode = registered ? source("gemma4_vision_qkv", SPEC) : derived;
  if (registered && candidateCode.trim() !== derived.trim()) throw new Error("registered vision QKV differs from the exact probe");
  const reducerCode = source("gemma4_vision_reduce_norm_rope");
  const ordinaryLayout = matmulLayout(device);
  const candidateLayout = device.createBindGroupLayout({ label: "DEV triple BF16 QKV", entries:
    ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"].map((type, binding) =>
      ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const reductionLayout = device.createBindGroupLayout({ label: "registered vision reduce norm RoPE", entries:
    ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"].map((type, binding) =>
      ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pipelineLayout = (layout) => device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const [baseline, candidate, reducer] = await Promise.all([
    pipeline(device, baselineCode, "registered vision BF16 R4 row56 matmul", pipelineLayout(ordinaryLayout)),
    pipeline(device, candidateCode, "DEV fused triple vision BF16 matmul", pipelineLayout(candidateLayout)),
    pipeline(device, reducerCode, "registered unchanged vision reduce norm RoPE", pipelineLayout(reductionLayout)),
  ]);
  const codes = { baseline: baselineCode, candidate: candidateCode, reducer: reducerCode };
  const sourceBytes = Object.values(codes).reduce((sum, code) => sum + new TextEncoder().encode(code).byteLength, 0);
  const sourceHashes = Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)])));
  return { baseline, candidate, reducer, ordinaryLayout, candidateLayout, reductionLayout, sourceBytes, sourceHashes, registered };
}

function workgroupCounts(T, splits) {
  const grid = { x: Math.ceil(N / BN), y: Math.ceil(T / BM_VIRTUAL), z: splits };
  return { grid, candidate: grid.x * grid.y * grid.z, baseline: 3 * grid.x * grid.y * grid.z };
}

async function checkCase(runtime, pipes, T, inputFixture, weightFixture, includeTiming) {
  const d = runtime.device, U = GPUBufferUsage, plan = qkvMemoryPlan(T, N, K, pipes.sourceBytes), owned = [], allocated = new Map();
  if (plan.controlledTemporaryBytes > LIMIT) throw new Error("QKV controlled temporary bytes exceed64MiB: " + plan.controlledTemporaryBytes);
  const storageLimit = Math.min(d.limits.maxBufferSize, d.limits.maxStorageBufferBindingSize);
  for (const [label, bytes] of Object.entries(plan.gpuBuffers)) if (bytes > storageLimit) throw new Error("QKV temporary " + label + " exceeds adapter buffer limits");
  if (plan.immutableWeightBytesEach > storageLimit) throw new Error("QKV immutable fixture weight exceeds adapter buffer limits");
  const deadline = performance.now() + DEADLINE_MS, stage = new ArrayBuffer(CHUNK);
  let gpuTemporaryBytes = 0, queries = null;
  const make = (label, usage, immutable = false, bytesOverride = null) => {
    const bytes = bytesOverride ?? (immutable ? plan.immutableWeightBytesEach : plan.gpuBuffers[label]);
    if (!bytes || (!immutable && allocated.has(label))) throw new Error("unplanned or duplicate QKV buffer " + label);
    const buffer = d.createBuffer({ size: bytes, usage, label: "vision-qkv." + label }); owned.push(buffer);
    if (immutable) return buffer;
    allocated.set(label, bytes); gpuTemporaryBytes += bytes;
    if (gpuTemporaryBytes > plan.gpuTemporaryBytes) throw new Error("QKV actual temporary allocations exceed plan");
    return buffer;
  };
  const storage = (label, immutable = false) => make(label, U.STORAGE | U.COPY_DST | U.COPY_SRC, immutable);
  const uniform = (label, values, bytes = null) => {
    const buffer = make(label, U.UNIFORM | U.COPY_DST, false, bytes ?? values.byteLength);
    d.queue.writeBuffer(buffer, 0, values); return buffer;
  };
  const bind = (layout, buffers, ranges = []) => d.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding,
    resource: ranges[binding] ? { buffer, offset: ranges[binding].offset, size: ranges[binding].size } : { buffer } })) });
  try {
    const globals = uniform("globals", Uint32Array.of(T, ROWS, plan.splits, 0));
    const params = uniform("params", Uint32Array.of(N, K, 0, 0));
    const X = storage("X");
    const W = [storage("immutableWQ", true), storage("immutableWK", true), storage("immutableWV", true)];
    const S = storage("S"), B = storage("B");
    const baselineParts = [storage("baselinePartQ"), storage("baselinePartK"), storage("baselinePartV")];
    const candidatePart = storage("candidatePart");
    const baselineY = [storage("baselineYQ"), storage("baselineYK"), storage("baselineYV")];
    const candidateY = [storage("candidateYQ"), storage("candidateYK"), storage("candidateYV")];
    const normW = [storage("normQ"), storage("normK"), storage("normV")];
    const rope = storage("rope");
    const reducerParams = [uniform("reduceParamsQ", new Uint32Array(12), 48), uniform("reduceParamsK", new Uint32Array(12), 48), uniform("reduceParamsV", new Uint32Array(12), 48)];
    const readback = make("readback", U.MAP_READ | U.COPY_DST);
    const timestampResolve = make("timestampResolve", U.QUERY_RESOLVE | U.COPY_SRC);
    if (gpuTemporaryBytes !== plan.gpuTemporaryBytes || allocated.size !== Object.keys(plan.gpuBuffers).length) {
      throw new Error("QKV GPU allocation differs from the complete planned inventory");
    }
    if (plan.splits !== 6 && T === T_DEFAULT) throw new Error("actual T108 QKV must preserve S=6, got " + plan.splits);

    await fillBuffer(d, X, plan.inputRows * K * 4, (index) => {
      const row = Math.floor(index / K);
      return row < T ? inputValue(inputFixture, index) : NaN;
    }, stage, deadline, "QKV F32 input and NaN padded rows upload");
    await uploadFloatTail(d, X, plan.inputRows * K, deadline, "QKV input tail");
    for (let projection = 0; projection < 3; projection++) {
      await fillWeights(d, W[projection], N * K * 2, projection, weightFixture, stage, deadline);
      await fillBuffer(d, normW[projection], DIM * 4, (index) => normWeight(projection, index), stage, deadline, "QKV norm weight upload");
      await uploadFloatTail(d, normW[projection], DIM, deadline, "QKV norm weight tail");
      const lo = [-0.63, -0.47, -0.81][projection], hi = [0.79, 0.93, 0.59][projection];
      const values = new ArrayBuffer(48), u = new Uint32Array(values), f = new Float32Array(values);
      u[0] = T; u[1] = HEADS; u[2] = DIM; u[3] = Number(projection !== 2);
      f[4] = 1e-6; f[5] = lo; f[6] = hi; u[7] = Number(projection === 2); u[8] = plan.splits; u[9] = 0; u[10] = 0; u[11] = 0;
      d.queue.writeBuffer(reducerParams[projection], 0, values);
    }
    await fillBuffer(d, rope, T * DIM * 4, (index) => ropeValue(Math.floor(index / DIM), index % DIM), stage, deadline, "QKV RoPE upload");
    await uploadFloatTail(d, rope, T * DIM, deadline, "QKV RoPE tail");
    // BF16 paths must ignore disabled scale/bias storage; NaN makes an accidental read visible.
    await fillBuffer(d, S, 16, () => NaN, stage, deadline, "disabled QKV scales");
    await fillBuffer(d, B, N * 4, () => NaN, stage, deadline, "disabled QKV bias");
    await uploadFloatTail(d, B, N, deadline, "QKV bias tail");

    const baselineGroups = [0, 1, 2].map((projection) => bind(pipes.ordinaryLayout,
      [globals, params, X, W[projection], S, B, baselineY[projection], baselineParts[projection]]));
    const tripleGroup = bind(pipes.candidateLayout, [globals, params, X, ...W, candidatePart]);
    const segmentBytes = plan.partialFloats * 4;
    if (segmentBytes % 256) throw new Error("candidate reducer segment offset is not aligned to WebGPU storage binding alignment");
    const reducerGroups = [0, 1, 2].map((projection) => {
      const baseline = bind(pipes.reductionLayout, [reducerParams[projection], baselineParts[projection], normW[projection], rope, baselineY[projection]]);
      const candidate = bind(pipes.reductionLayout,
        [reducerParams[projection], candidatePart, normW[projection], rope, candidateY[projection]],
        [null, { offset: projection * segmentBytes, size: segmentBytes }, null, null, null]);
      return { baseline, candidate };
    });
    const dispatchBaseline = (pass) => {
      pass.setPipeline(pipes.baseline);
      for (const group of baselineGroups) { pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(N / BN), Math.ceil(T / BM_PHYSICAL), plan.splits); }
      pass.setPipeline(pipes.reducer);
      for (const groups of reducerGroups) { pass.setBindGroup(0, groups.baseline); pass.dispatchWorkgroups(T * HEADS); }
    };
    const dispatchCandidate = (pass) => {
      pass.setPipeline(pipes.candidate); pass.setBindGroup(0, tripleGroup);
      pass.dispatchWorkgroups(Math.ceil(N / BN), Math.ceil(T / BM_PHYSICAL), plan.splits);
      pass.setPipeline(pipes.reducer);
      for (const groups of reducerGroups) { pass.setBindGroup(0, groups.candidate); pass.dispatchWorkgroups(T * HEADS); }
    };
    const poison = async (phase) => {
      for (const [index, buffer] of baselineParts.entries()) {
        await fillBuffer(d, buffer, plan.partialFloats * 4, () => NaN, stage, deadline, phase + " baseline PART poison");
        await uploadFloatTail(d, buffer, plan.partialFloats, deadline, phase + " baseline PART tail");
      }
      await fillBuffer(d, candidatePart, 3 * plan.partialFloats * 4, () => NaN, stage, deadline, phase + " candidate PART poison");
      await uploadFloatTail(d, candidatePart, 3 * plan.partialFloats, deadline, phase + " candidate PART tail");
      for (const buffer of [...baselineY, ...candidateY]) {
        await fillBuffer(d, buffer, plan.outputFloats * 4, () => NaN, stage, deadline, phase + " Y poison");
        await uploadFloatTail(d, buffer, plan.outputFloats, deadline, phase + " Y tail");
      }
    };
    const verify = async (phase) => {
      await poison(phase);
      const encoder = d.createCommandEncoder({ label: "vision QKV " + phase });
      const pass = encoder.beginComputePass(); dispatchBaseline(pass); dispatchCandidate(pass); pass.end(); d.queue.submit([encoder.finish()]);
      await drain(d, deadline, phase + " QKV matmul and reducers");
      const parts = [], outputs = [];
      for (let projection = 0; projection < 3; projection++) {
        parts.push(await compareRanges(d, readback, baselineParts[projection], 0, candidatePart, projection * segmentBytes,
          plan.partialFloats, phase + " " + PROJECTIONS[projection] + " PART", deadline));
        outputs.push(await compareRanges(d, readback, baselineY[projection], 0, candidateY[projection], 0,
          plan.outputFloats, phase + " " + PROJECTIONS[projection] + " normalized-RoPE", deadline));
      }
      const tails = await assertTails(d, readback, [
        ...baselineParts.map((buffer, projection) => ({ buffer, offset: plan.partialFloats * 4, label: "baseline " + PROJECTIONS[projection] + " PART" })),
        { buffer: candidatePart, offset: 3 * plan.partialFloats * 4, label: "candidate triple PART" },
        ...baselineY.map((buffer, projection) => ({ buffer, offset: plan.outputFloats * 4, label: "baseline " + PROJECTIONS[projection] + " Y" })),
        ...candidateY.map((buffer, projection) => ({ buffer, offset: plan.outputFloats * 4, label: "candidate " + PROJECTIONS[projection] + " Y" })),
      ], deadline);
      return { phase, partialPlanes: parts, normalizedRopeOutputs: outputs, tails, allOutputsFinite: true, exact: true };
    };

    const before = await verify("before");
    let timing = null;
    if (includeTiming) {
      if (!d.features.has("timestamp-query")) throw new Error("timestamp-query is unavailable; no timing result");
      // Warm each complete path, then alternate paired timestamp intervals, with no reset/upload work in scope.
      const warm = d.createCommandEncoder({ label: "vision QKV warmup" }), warmPass = warm.beginComputePass();
      dispatchBaseline(warmPass); dispatchCandidate(warmPass); warmPass.end(); d.queue.submit([warm.finish()]);
      queries = d.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
      const encoder = d.createCommandEncoder({ label: "vision QKV paired timing" }), order = [];
      for (let sample = 0; sample < SAMPLES; sample++) for (const candidate of sample % 2 ? [true, false] : [false, true]) {
        const index = order.length * 2;
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repetition = 0; repetition < REPETITIONS; repetition++) candidate ? dispatchCandidate(pass) : dispatchBaseline(pass);
        pass.end(); order.push(candidate);
      }
      encoder.resolveQuerySet(queries, 0, SAMPLES * 4, timestampResolve, 0);
      encoder.copyBufferToBuffer(timestampResolve, 0, readback, 0, SAMPLES * 4 * 8); d.queue.submit([encoder.finish()]);
      if (!await mapBeforeDeadline(readback, SAMPLES * 4 * 8, deadline)) { readback.unmap(); throw new Error("QKV timestamp readback deadline exceeded"); }
      const samples = { baseline: [], candidate: [] };
      try {
        const ticks = new BigUint64Array(readback.getMappedRange(0, SAMPLES * 4 * 8).slice(0));
        order.forEach((candidate, index) => {
          const duration = ticks[index * 2 + 1] - ticks[index * 2];
          if (duration <= 0n) throw new Error("QKV timing produced a nonpositive timestamp interval");
          samples[candidate ? "candidate" : "baseline"].push(Number(duration) / 1e6 / REPETITIONS);
        });
      } finally { readback.unmap(); queries.destroy(); queries = null; }
      const baselineMedianMs = median(samples.baseline), candidateMedianMs = median(samples.candidate);
      timing = { scope: "three registered BF16 matmuls + three unchanged reducers vs one triple BF16 matmul + the same three reducers",
        dispatches: { baseline: 6, candidate: 4 }, workgroupsPerMatmul: workgroupCounts(T, plan.splits), samples: SAMPLES,
        repetitions: REPETITIONS, samplesMs: samples, baselineMedianMs, candidateMedianMs,
        speedup: baselineMedianMs / candidateMedianMs };
    }
    const after = await verify("after");
    return { T, N, K, splits: plan.splits, inputRowsWithNaNPadding: plan.inputRows, inputFixture, weightFixture, spec: SPEC,
      reducerBounds: { q: [-0.63, 0.79], k: [-0.47, 0.93], v: [-0.81, 0.59] }, dispatch: workgroupCounts(T, plan.splits),
      before, timing, after, sourceHashes: pipes.sourceHashes, precision: "BF16 stored words; original F32 decode, dot, accumulation, and split reduction; no shader-f16",
      sharedBytes: 20 * 1024, memory: { ...plan, actualGpuTemporaryBytes: gpuTemporaryBytes,
        totalGpuBytesIncludingImmutableFixtureWeights: gpuTemporaryBytes + plan.immutableWeightBytes }, ok: true };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), query = new URLSearchParams(location.search), full = query.has("full");
  const runtime = await requestDevice({ features: ["timestamp-query"] }), d = runtime.device, errors = [];
  d.addEventListener("uncapturederror", (event) => errors.push(event.error?.message || String(event.error)));
  try {
    const splits = mmSplits(T_DEFAULT, N, K, TARGET, BM_PHYSICAL, BN);
    const mapping = assertQkvMapping(T_DEFAULT, splits), grid = workgroupCounts(T_DEFAULT, splits);
    metadata = { adapter: { name: runtime.name, features: [...d.features], fallback: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter) },
      actualProfile: { T: T_DEFAULT, N, K, rowsPerThread: ROWS, physicalBM: BM_PHYSICAL, virtualBM: BM_VIRTUAL, BN, splits, splitTarget: TARGET },
      mapping, dispatchReduction: "baseline 3 matmul + 3 unchanged reduce_norm_rope dispatches; candidate one triple matmul + the same 3 reducers",
      candidateWGSL: "Each candidate workgroup computes Q/K/V for one unchanged logical split. Shared X tile and exact K32 ranges are read once; PART planes are stored projection-major then split-major.",
      temporaryLimitBytes: LIMIT, immutableWeights: "Three BF16 fixture weights are separately reported and excluded from the temporary cap; all X, partials, outputs, staging, query, and readback allocations count." };
    if (!d.features.has("timestamp-query") || d.limits.maxComputeWorkgroupStorageSize < 20 * 1024 ||
        d.limits.maxStorageBuffersPerShaderStage < 6 || d.limits.maxBindingsPerBindGroup < 8) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-vision-qkv-bf16", ...metadata,
        reason: "Requires timestamp-query, 20KiB workgroup storage, six storage bindings, and eight bindings per bind group; skipped without a timing claim." };
      log(window.gpuBench); return;
    }
    const source = await kernelSource(query.get("wasm") || undefined);
    const pipes = await withGpuErrors(runtime, () => compileProbe(d, source, query.get("registered") === "1"), "compile triple QKV BF16 probe");
    metadata = { ...metadata, sourceHashes: pipes.sourceHashes, sourceBytes: pipes.sourceBytes, wasmUrl: source.wasmUrl, registered: pipes.registered };
    const fixtures = full ? [
      { input: "normal", weight: "normal" }, { input: "signed-zero", weight: "normal" },
      { input: "cancellation", weight: "normal" }, { input: "normal", weight: "tiny-subnormal" },
      { input: "normal", weight: "large-finite" },
    ] : [{ input: "normal", weight: "normal" }];
    for (const [index, fixture] of fixtures.entries()) {
      const result = await withGpuErrors(runtime,
        () => checkCase(runtime, pipes, T_DEFAULT, fixture.input, fixture.weight, index === 0),
        "check triple QKV T108 " + fixture.input + "/" + fixture.weight);
      cases.push(result); log(result);
    }
    if (errors.length) throw new Error("uncaptured GPU errors: " + errors.join("; "));
    const actual = cases[0];
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-vision-qkv-bf16", ...metadata,
      metricMs: actual.timing?.candidateMedianMs, baselineMetricMs: actual.timing?.baselineMedianMs,
      metric: "T108 vision attention Q/K/V matmul plus unchanged exact reducers GPU time; not end-to-end image latency or camera FPS",
      elapsedMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0,
        fullRawPartialPlanes: true, fullNormalizedQkvAndRopeOutputs: true, beforeAndAfterTiming: true, finiteActiveValues: true,
        poisonedPartialsAndFourFloatTails: true, fixtureCoverage: full ? WEIGHT_FIXTURES.concat(INPUT_FIXTURES) : ["actual T108/N768/K768 normal fixtures"] }, cases };
    log(window.gpuBench);
  } finally { d.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-vision-qkv-bf16" };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", check: "gemma4-vision-qkv-bf16", ...metadata,
      correctness: { ok: false, bitExact: true, tolerance: 0 }, cases, error: String(error?.message || error) };
    log(error.stack || String(error));
  });
}
