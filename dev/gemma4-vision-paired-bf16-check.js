// DEV-only paired BF16 gate/up. Registered R3 dot/split order and vision GELU remain exact.
import { matmulLayout, reduceLayout, mmSplits, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const CHUNK = 1024 * 1024, LIMIT = 64 * CHUNK, TAIL = 4, SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(Float32Array.of(SENTINEL).buffer)[0];
const SPEC = { weights: "bf16", f16: false, rows: 3, row56: false, groups: 1, splitTarget: 256 };
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const count = (code, marker) => code.split(marker).length - 1;
const median = (values) => { const ordered = [...values].sort((a, b) => a - b), m = ordered.length >> 1; return ordered.length % 2 ? ordered[m] : (ordered[m - 1] + ordered[m]) / 2; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

// Frozen pure lossless packing derivation copied from gemma4-bf16-shared-check.js.

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

export function derivePairedBf16(original) {
  const packed = derivePackedBf16Shared(original);
  const originalStore = functionSource(packed, "mm_store");
  if (!originalStore) throw new Error("registered BF16 store function changed");
  let code = packed;
  const replace = (before, after, expected = 1) => { code = replaceExact(code, before, after, expected, "paired gate/up"); };
  const oldBindings = "@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;\n@group(0) @binding(4) var<storage, read> S: array<f32>;\n@group(0) @binding(5) var<storage, read> B: array<f32>;\n@group(0) @binding(6) var<storage, read_write> Y: array<f32>;\n@group(0) @binding(7) var<storage, read_write> PART: array<f32>;";
  replace(oldBindings, "@group(0) @binding(3) var<storage, read> WG: array<vec4<u32>>;\n@group(0) @binding(4) var<storage, read> WU: array<vec4<u32>>;\n@group(0) @binding(5) var<storage, read> B: array<f32>;\n@group(0) @binding(6) var<storage, read_write> YG: array<f32>;\n@group(0) @binding(7) var<storage, read_write> YU: array<f32>;\n@group(0) @binding(8) var<storage, read_write> PG: array<f32>;\n@group(0) @binding(9) var<storage, read_write> PU: array<f32>;");
  const gateStore = originalStore.replace(/\bmm_store\b/g, "mm_store_g").replace(/\bPART\b/g, "PG").replace(/\bY\b/g, "YG");
  const upStore = originalStore.replace(/\bmm_store\b/g, "mm_store_u").replace(/\bPART\b/g, "PU").replace(/\bY\b/g, "YU");
  replace(originalStore, gateStore + "\n\n" + upStore);
  code = code.replace(/\bws\b/g, "gs").replace(/\bwv\b/g, "gv");
  replace("var<workgroup> gs: array<vec2<u32>, 512>;", "var<workgroup> gs: array<vec2<u32>, 512>;\nvar<workgroup> us: array<vec2<u32>, 512>;");
  replace("var gv: array<vec2<u32>, 2>;", "var gv: array<vec2<u32>, 2>;\n  var uv: array<vec2<u32>, 2>;");
  for (let i = 0; i < 6; i++) replace(`var acc${i} = vec2<f32>(0.0);`, `var acc${i} = vec2<f32>(0.0);\n  var upacc${i} = vec2<f32>(0.0);`);
  for (const at of ["2u * gg", "2u * gg + 1u"]) replace(`gv[${at}] = vec2<u32>(0u);`, `gv[${at}] = vec2<u32>(0u);\n      uv[${at}] = vec2<u32>(0u);`);
  replace("let v = W[(wrow * K + kb * 32u) / 8u + lq];", "let v = WG[(wrow * K + kb * 32u) / 8u + lq];\n        let u = WU[(wrow * K + kb * 32u) / 8u + lq];");
  replace("gv[2u * gg] = v.xy;", "gv[2u * gg] = v.xy;\n        uv[2u * gg] = u.xy;");
  replace("gv[2u * gg + 1u] = v.zw;", "gv[2u * gg + 1u] = v.zw;\n        uv[2u * gg + 1u] = u.zw;");
  for (const lane of ["lq * 2u", "lq * 2u + 1u"]) {
    const at = lane === "lq * 2u" ? "2u * gg" : "2u * gg + 1u";
    const store = `gs[r * 8u + ((${lane}) ^ sw)] = gv[${at}];`;
    replace(store, store + `\n      us[r * 8u + ((${lane}) ^ sw)] = uv[${at}];`);
  }
  for (const [name, col] of [["b0", "c"], ["b1", "(c + 32u)"]]) {
    const load = `let ${name} = bf16(gs[${col} * 8u + (kq ^ nx)]);`;
    replace(load, load + `\n      let ${name === "b0" ? "u0" : "u1"} = bf16(us[${col} * 8u + (kq ^ nx)]);`, 8);
  }
  code = code.replace(/acc(\d+) \+= vec2<f32>\(dot\(a(\d+), b0\), dot\(a\2, b1\)\);/g, (line, acc, a) => `${line}\n      upacc${acc} += vec2<f32>(dot(a${a}, u0), dot(a${a}, u1));`);
  replace("      if (p.bias == 1u) { bias = B[col]; }", "      if (p.bias == 1u) { bias = B[col]; }\n      var upBias = 0.0;\n      if (p.bias == 1u) { upBias = B[N + col]; }", 2);
  let stores = 0;
  code = code.replace(/mm_store\((acc\d+\.[xy]), ([^;]+)\);/g, (_all, acc, args) => {
    stores++;
    return `mm_store_g(${acc}, ${args});\n      mm_store_u(up${acc}, ${args.replace(", bias,", ", upBias,")});`;
  });
  if (stores !== 12 || count(code, "workgroupBarrier();") !== 2 || code.includes("enable f16;") || code.includes("<f16>")) throw new Error("paired BF16 store/barrier/precision invariant failed");
  const dots = (s, up = false) => [...s.matchAll(up ? /upacc\d+ \+= vec2<f32>\([^;]+;/g : /\bacc\d+ \+= vec2<f32>\([^;]+;/g)].map(([line]) => line.replaceAll("upacc", "acc").replaceAll("u0", "b0").replaceAll("u1", "b1"));
  if (JSON.stringify(dots(code)) !== JSON.stringify(dots(original)) || JSON.stringify(dots(code, true)) !== JSON.stringify(dots(original))) throw new Error("paired BF16 changed either matrix's dot/add sequence");
  for (const name of ["bf16", "mm_splits"]) if (functionSource(code, name) !== functionSource(original, name)) throw new Error(`paired BF16 changed ${name}`);
  const restoredGate = functionSource(code, "mm_store_g").replace(/\bmm_store_g\b/g, "mm_store").replace(/\bPG\b/g, "PART").replace(/\bYG\b/g, "Y");
  const restoredUp = functionSource(code, "mm_store_u").replace(/\bmm_store_u\b/g, "mm_store").replace(/\bPU\b/g, "PART").replace(/\bYU\b/g, "Y");
  if (restoredGate !== originalStore || restoredUp !== originalStore || !code.includes("let per = (nb + splits - 1u) / splits;")) throw new Error("paired BF16 changed ordered split partition or scalar epilogue");
  return code;
}

export function pairedMappingProof() {
  const packed = assertPackedMapping(), xs = new Set(), gs = new Set(), us = new Set();
  for (let li = 0; li < 256; li++) {
    const lr = Math.floor(li / 4), lq = li % 4, sw = (lr >> 1) & 7;
    for (const q of [lq * 2, lq * 2 + 1]) { if (lr < 48) xs.add(lr * 8 + q); gs.add(lr * 8 + (q ^ sw)); us.add(lr * 8 + (q ^ sw)); }
  }
  if (xs.size !== 48 * 8 || gs.size !== 512 || us.size !== 512) throw new Error("paired BF16 producer coverage failed");
  let outputs = 0;
  for (let ly = 0; ly < 8; ly++) for (let lx = 0; lx < 32; lx++) for (let row = 0; row < 6; row++) for (const col of [lx, lx + 32]) {
    for (let kq = 0; kq < 8; kq++) {
      if (!xs.has((ly + row * 8) * 8 + kq) || !gs.has(col * 8 + (kq ^ ((lx >> 1) & 7))) || !us.has(col * 8 + (kq ^ ((lx >> 1) & 7)))) throw new Error("paired BF16 consumer reads an uninitialized shared slot");
    }
    outputs++;
  }
  if (outputs !== 48 * 64) throw new Error("paired BF16 output coverage failed");
  return { ...packed, sharedBytes: 16384, inputTileBytes: 8192, gateTileBytes: 4096, upTileBytes: 4096, outputsPerMatrix: outputs, activeXSlots: xs.size, gateSlots: gs.size, upSlots: us.size, orderedKq: [0, 1, 2, 3, 4, 5, 6, 7] };
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

export function pairedMemoryPlan(T, N, K, sourceBytes = 0) {
  const splits = mmSplits(T, N, K, 256, 48, 64), count = T * N, partialCount = splits * count;
  if (![T, N, K].every((v) => Number.isSafeInteger(v) && v > 0) || N % 4 || K % 32 || splits < 2) throw new Error("paired vision BF16 probe requires N multiple4, K multiple32, and original split count>1");
  const outputBytes = (count + TAIL) * 4, partBytes = (partialCount + TAIL) * 4;
  const gpuTemporaryBuffers = { X: (T * K + TAIL) * 4, BG: (N + TAIL) * 4, BU: (N + TAIL) * 4, Bpair: (2 * N + TAIL) * 4, S: 16,
    globals: 16, params: 16, geluParams: 48, seedY: outputBytes, oldYG: outputBytes, oldYU: outputBytes, newYG: outputBytes, newYU: outputBytes,
    oldAct: outputBytes, newAct: outputBytes, oldPG: partBytes, oldPU: partBytes, newPG: partBytes, newPU: partBytes,
    readback: 2 * CHUNK, timestamps: 256 };
  const gpuTemporaryBytes = Object.values(gpuTemporaryBuffers).reduce((sum, value) => sum + value, 0);
  const hostStageBytes = CHUNK, hostReadbackBytes = 2 * CHUNK, uploadReserveBytes = CHUNK, fixedHostBytes = 2 * sourceBytes + 65536, querySetBytes = 12 * 8;
  return { T, N, K, splits, count, partialCount, outputBytes, partBytes, gpuTemporaryBuffers, gpuTemporaryBytes, hostStageBytes, hostReadbackBytes, uploadReserveBytes, fixedHostBytes, querySetBytes,
    controlledTemporaryBytes: gpuTemporaryBytes + hostStageBytes + hostReadbackBytes + uploadReserveBytes + fixedHostBytes + querySetBytes,
    excludedImmutableGpuWeightBytes: 2 * N * K * 2, limitBytes: LIMIT };
}

function bf16Word(pattern, index, up) {
  const at = index + (up ? 71 : 19), sign = (at + (at >>> 4)) & 1 ? 0x8000 : 0;
  if (index % 257 === 0 || pattern === "zero") return sign;
  if (pattern === "tiny") return sign | (1 + (at * 29 + 7) % 0x7f);
  if (pattern === "large") return sign | ((133 + at % 5) << 7) | ((at * 37 + 11) & 0x7f);
  return sign | ((123 + at % 3) << 7) | ((at * 37 + 11) & 0x7f);
}

async function compileProbe(device, source) {
  const baselineCode = source("matmul", SPEC), pairedCode = derivePairedBf16(baselineCode), reducerCode = source("reduce", SPEC), geluCode = source("gemma4_vision_reduce_gelu");
  const ordinaryLayout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const pairedLayout = device.createBindGroupLayout({ entries: ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage", "storage", "storage"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pLayout = (layout) => device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const kernels = {
    baseline: await pipeline(device, baselineCode, "registered BF16 vision R3", pLayout(ordinaryLayout)),
    candidate: await pipeline(device, pairedCode, "paired packed BF16 vision R3", pLayout(pairedLayout)),
    reducer: await pipeline(device, reducerCode, "registered unchanged R3 reducer", pLayout(reductionLayout)),
    gelu: await pipeline(device, geluCode, "registered unchanged vision reduce GELU"),
  };
  const codes = { baseline: baselineCode, candidate: pairedCode, reducer: reducerCode, visionReduceGelu: geluCode };
  return { ...kernels, ordinaryLayout, pairedLayout, reductionLayout, sourceBytes: Object.values(codes).reduce((sum, code) => sum + new TextEncoder().encode(code).byteLength, 0), sourceHashes: Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([key, code]) => [key, await hash(code)]))) };
}

async function checkCase(runtime, pipes, c, full) {
  const d = runtime.device, U = GPUBufferUsage, plan = pairedMemoryPlan(c.T, c.N, c.K, pipes.sourceBytes), owned = [], allocated = new Set();
  if (plan.controlledTemporaryBytes > LIMIT) throw new Error(`paired BF16 controlled temporary memory exceeds64MiB: ${plan.controlledTemporaryBytes}`);
  const weightsBytes = c.N * c.K * 2, storageLimit = Math.min(d.limits.maxBufferSize, d.limits.maxStorageBufferBindingSize);
  if ([weightsBytes, ...Object.values(plan.gpuTemporaryBuffers)].some((v) => v > storageLimit)) throw new Error("paired BF16 buffer exceeds adapter limits");
  const deadline = performance.now() + (full ? 180000 : 60000), stage = new ArrayBuffer(CHUNK);
  let gpuBytes = 0, queries;
  const make = (label, usage, immutableWeight = false) => {
    const bytes = immutableWeight ? weightsBytes : plan.gpuTemporaryBuffers[label];
    if (!bytes || (!immutableWeight && allocated.has(label))) throw new Error(`invalid planned buffer ${label}`);
    const buffer = d.createBuffer({ size: bytes, usage, label: `vision-paired-bf16.${label}` }); owned.push(buffer);
    if (!immutableWeight) { allocated.add(label); gpuBytes += bytes; if (gpuBytes > plan.gpuTemporaryBytes) throw new Error("GPU temporary accounting exceeded plan"); }
    return buffer;
  };
  const storage = (label, immutable = false) => make(label, U.STORAGE | U.COPY_DST | U.COPY_SRC, immutable);
  const bind = (layout, buffers) => d.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const uniform = (label, values) => { const buffer = make(label, U.UNIFORM | U.COPY_DST); d.queue.writeBuffer(buffer, 0, values); return buffer; };
  const generated = (target, bytes, width, value, label) => uploadGenerated(d, target, bytes, width, value, stage, deadline, label);
  const filled = async (target, logical, value, label) => { await generated(target, logical * 4, 4, () => value, label); await uploadTail(d, target, logical, deadline, label); };
  try {
    const WG = storage("WG", true), WU = storage("WU", true), X = storage("X"), BG = storage("BG"), BU = storage("BU"), Bpair = storage("Bpair"), S = storage("S");
    const globals = uniform("globals", Uint32Array.of(c.T, 0, 0, 0)), params = uniform("params", Uint32Array.of(c.N, c.K, 0, 0));
    const geluData = new ArrayBuffer(48), gi = new Uint32Array(geluData), gf = new Float32Array(geluData);
    gi[0] = plan.count; gi[3] = Number(c.clipped); gi[8] = plan.splits;
    [gf[1], gf[2], gf[4], gf[5], gf[6], gf[7]] = c.clipped ? [-0.0317, 0.0713, -0.1731, 0.413, -0.317, 0.531] : [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity];
    const geluParams = uniform("geluParams", gi);
    const seedY = make("seedY", U.COPY_SRC | U.COPY_DST), oldYG = storage("oldYG"), oldYU = storage("oldYU"), newYG = storage("newYG"), newYU = storage("newYU"), oldAct = storage("oldAct"), newAct = storage("newAct");
    const oldPG = storage("oldPG"), oldPU = storage("oldPU"), newPG = storage("newPG"), newPU = storage("newPU");
    const read = make("readback", U.MAP_READ | U.COPY_DST), resolved = make("timestamps", U.QUERY_RESOLVE | U.COPY_SRC);
    if (gpuBytes !== plan.gpuTemporaryBytes || allocated.size !== Object.keys(plan.gpuTemporaryBuffers).length) throw new Error("GPU temporary accounting differs from the complete allocation plan");
    await generated(WG, weightsBytes, 2, (i) => bf16Word(c.pattern, i, false), "gate BF16 upload");
    await generated(WU, weightsBytes, 2, (i) => bf16Word(c.pattern, i, true), "up BF16 upload");
    await generated(X, c.T * c.K * 4, 4, (i) => c.pattern === "zero" ? i % 2 ? -0 : 0 : c.pattern === "cancellation" ? (i % 8 < 4 ? 64 : -64) + (i % 4 + 1) * 0.0625 : 0.0125 * Math.sin((i % c.K) * 0.071 + Math.floor(i / c.K) * 0.17 + 0.23), "input F32 upload");
    await uploadTail(d, X, c.T * c.K, deadline, "input tail");
    await generated(S, 16, 4, () => NaN, "disabled scale poison");
    await generated(seedY, plan.count * 4, 4, (i) => i % 193 === 0 ? i % 386 ? -0 : 0 : 0.013 * Math.cos(Math.floor(i / c.N) * 0.11 + (i % c.N) * 0.037), "raw Y seed upload");
    await uploadTail(d, seedY, plan.count, deadline, "raw Y seed tail");
    const ordinary = [bind(pipes.ordinaryLayout, [globals, params, X, WG, S, BG, oldYG, oldPG]), bind(pipes.ordinaryLayout, [globals, params, X, WU, S, BU, oldYU, oldPU])];
    const paired = bind(pipes.pairedLayout, [globals, params, X, WG, WU, Bpair, newYG, newYU, newPG, newPU]);
    const reduce = [[oldPG, BG, oldYG], [oldPU, BU, oldYU], [newPG, BG, newYG], [newPU, BU, newYU]].map((buffers) => bind(pipes.reductionLayout, [globals, params, ...buffers]));
    const acts = [[oldPG, oldPU, oldAct], [newPG, newPU, newAct]].map((buffers) => bind(pipes.gelu.getBindGroupLayout(0), [geluParams, ...buffers]));
    const gemm = (pass, candidate) => {
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
      for (const group of candidate ? [paired] : ordinary) { pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(c.N / 64), Math.ceil(c.T / 48), plan.splits); }
    };
    const activation = (pass, candidate) => { pass.setPipeline(pipes.gelu); pass.setBindGroup(0, acts[Number(candidate)]); pass.dispatchWorkgroups(Math.ceil(plan.count / 1024)); };
    const composite = (pass, candidate) => { gemm(pass, candidate); activation(pass, candidate); };
    const rawReduce = (pass) => { pass.setPipeline(pipes.reducer); for (const group of reduce) { pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(plan.count / 256)); } };
    const setBias = async (enabled) => {
      const value = (i, up) => !enabled ? i % 2 ? -0 : 0 : i % 101 === 0 ? i % 202 ? -0 : 0 : ((i * (up ? 7 : 11) + (up ? 3 : 5)) % 17 - 8) * 0.00737;
      await generated(BG, c.N * 4, 4, (i) => value(i, false), "gate bias upload"); await uploadTail(d, BG, c.N, deadline, "gate bias tail");
      await generated(BU, c.N * 4, 4, (i) => value(i, true), "up bias upload"); await uploadTail(d, BU, c.N, deadline, "up bias tail");
      await generated(Bpair, 2 * c.N * 4, 4, (i) => value(i % c.N, i >= c.N), "paired independent biases upload"); await uploadTail(d, Bpair, 2 * c.N, deadline, "paired bias tail");
    };
    const verify = async (phase, mode, bias) => {
      await setBias(bias);
      d.queue.writeBuffer(params, 0, Uint32Array.of(c.N, c.K, mode, Number(bias)));
      // Poison every active PART word and explicitly initialize all four allocation tails.
      for (const buffer of [oldPG, oldPU, newPG, newPU]) await filled(buffer, plan.partialCount, NaN, `${phase} PART poison`);
      for (const buffer of [oldAct, newAct]) await filled(buffer, plan.count, NaN, `${phase} activation poison`);
      const encoder = d.createCommandEncoder();
      for (const output of [oldYG, oldYU, newYG, newYU]) encoder.copyBufferToBuffer(seedY, 0, output, 0, plan.outputBytes);
      const pass = encoder.beginComputePass(); composite(pass, false); composite(pass, true); rawReduce(pass); pass.end(); d.queue.submit([encoder.finish()]);
      await drain(d, deadline, `${phase} compute`);
      const checks = {};
      for (const [label, a, b, size] of [["gatePART", oldPG, newPG, plan.partialCount], ["upPART", oldPU, newPU, plan.partialCount], ["gateY", oldYG, newYG, plan.count], ["upY", oldYU, newYU, plan.count], ["visionGelu", oldAct, newAct, plan.count]]) checks[label] = await compareBuffers(d, read, a, b, size, `${phase}.${label}.mode${mode}.bias${bias}`, deadline);
      return { phase, mode, bias, checks, poisonedActiveParts: true, allPartAndOutputTailsInitialized: true };
    };
    const modes = full ? [0, 1, 2] : [0], guards = { before: [], after: [] };
    for (const phase of ["before", "after"]) {
      for (const bias of [false, true]) for (const mode of modes) guards[phase].push(await verify(phase, mode, bias));
      if (phase === "before") {
        await setBias(false); d.queue.writeBuffer(params, 0, Uint32Array.of(c.N, c.K, 0, 0));
        const warm = d.createCommandEncoder(), warmPass = warm.beginComputePass(); for (let i = 0; i < 4; i++) { composite(warmPass, false); composite(warmPass, true); } warmPass.end(); d.queue.submit([warm.finish()]);
        const samples = { baseline: [], candidate: [] }, order = [], queryCount = 12;
        queries = d.createQuerySet({ type: "timestamp", count: queryCount });
        const encoder = d.createCommandEncoder();
        for (let sample = 0; sample < 3; sample++) for (const candidate of sample % 2 ? [true, false] : [false, true]) {
          const index = order.length * 2, pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
          for (let rep = 0; rep < 4; rep++) composite(pass, candidate); pass.end(); order.push(candidate);
        }
        encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8); d.queue.submit([encoder.finish()]);
        if (!await mapBeforeDeadline(read, queryCount * 8, deadline)) { read.unmap(); throw new Error("paired BF16 timestamp deadline exceeded"); }
        try {
          const times = new BigUint64Array(read.getMappedRange(0, queryCount * 8).slice(0));
          order.forEach((candidate, i) => { const ns = times[i * 2 + 1] - times[i * 2]; if (ns <= 0n) throw new Error("nonpositive paired BF16 timestamp"); samples[candidate ? "candidate" : "baseline"].push(Number(ns) / 1e6 / 4); });
        } finally { read.unmap(); queries.destroy(); queries = null; }
        const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
        guards.timing = { method: "alternating composite WebGPU timestamps", scope: "separate gate+up BF16 GEMMs + unchanged vision reducer/GELU vs paired GEMM + same reducer/GELU", dispatches: { baseline: 3, candidate: 2 }, samples: 3, repetitions: 4, ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs, stableFivePercentWin: samples.candidate.every((v, i) => v <= samples.baseline[i] * 0.95) };
      }
    }
    return { ...c, splits: plan.splits, spec: SPEC, dispatch: { x: Math.ceil(c.N / 64), y: Math.ceil(c.T / 48), z: plan.splits }, guards,
      precision: "BF16 stored words expanded exactly to F32; F32 dot/products/accumulation; no F16", sharedBytes: 16384, memory: { ...plan, actualGpuTemporaryBytes: gpuBytes, totalGpuBytesIncludingWeights: gpuBytes + plan.excludedImmutableGpuWeightBytes }, sourceHashes: pipes.sourceHashes, ok: true };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), query = new URLSearchParams(location.search), full = query.has("full"), mapping = pairedMappingProof();
  const runtime = await requestDevice({ features: ["timestamp-query"] }), d = runtime.device, errors = [];
  d.addEventListener("uncapturederror", (event) => errors.push(event.error?.message || String(event.error)));
  try {
    metadata = { adapter: { name: runtime.name, features: [...d.features], fallback: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter) }, mapping, temporaryLimitBytes: LIMIT, immutableWeights: "two BF16 GPU fixtures shared by both paths, excluded from temporary cap and separately reported; generated through1MiB staging" };
    if (!d.features.has("timestamp-query") || d.limits.maxComputeWorkgroupStorageSize < 16384 || d.limits.maxStorageBuffersPerShaderStage < 8) { window.gpuBench = { status: "skipped", done: true, ...metadata, reason: "timestamp-query,16KiB shared storage,and8 storage bindings required; no wall-clock metric" }; log(window.gpuBench); return; }
    const source = await kernelSource(query.get("wasm") || undefined), pipes = await withGpuErrors(runtime, () => compileProbe(d, source), "compile paired BF16 vision probe");
    metadata = { ...metadata, sourceHashes: pipes.sourceHashes, sourceBytes: pipes.sourceBytes, wasmUrl: source.wasmUrl };
    const inputs = [{ T: 144, N: 3072, K: 768, pattern: "normal", clipped: true, actualVision: true }];
    if (full) inputs.push({ T: 143, N: 3076, K: 768, pattern: "normal", clipped: false }, { T: 17, N: 260, K: 768, pattern: "cancellation", clipped: false }, { T: 17, N: 260, K: 768, pattern: "zero", clipped: true }, { T: 144, N: 3072, K: 768, pattern: "large", clipped: false }, { T: 17, N: 260, K: 768, pattern: "tiny", clipped: false });
    for (const input of inputs) { const result = await withGpuErrors(runtime, () => checkCase(runtime, pipes, input, full), `checking paired BF16 ${input.pattern} T${input.T}N${input.N}`); cases.push(result); log(result); }
    if (errors.length) throw new Error(`uncaptured GPU errors: ${errors.join("; ")}`);
    const actual = cases[0].guards.timing;
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-vision-paired-bf16", ...metadata, metricMs: actual.candidateMs, baselineMetricMs: actual.baselineMs, metric: "actual T144,N3072,K768 paired composite GPU milliseconds; not camera FPS", elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0, fullYAndPART: true, beforeAndAfterTimings: true, guardScope: full ? "all modes/biases, ragged rows/columns, signed zero/cancellation/large/tiny BF16" : "actual vision dimensions, independent gate/up weights, both bias settings, original mode0" }, cases };
    log(window.gpuBench);
  } finally { d.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-vision-paired-bf16" };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", check: "gemma4-vision-paired-bf16", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0 }, cases, error: String(error?.message || error) }; log(error.stack || String(error)); });
}
