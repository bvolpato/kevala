import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "../js/src/gpu-gemma4-batch-mlp.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024, CHUNK = MiB, LIMIT = 64 * MiB;
const T = 97, TARGET = 256, BM64 = 64, BN64 = 64, TAIL = 4;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const DEADLINE_MS = 60_000;
const SHAPES = [
  { shape: "gate", N: 6144, K: 1536, expectedSplits: 3 },
  { shape: "down", N: 1536, K: 6144, expectedSplits: 8 },
  { shape: "q", N: 2048, K: 1536, expectedSplits: 8 },
];
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const count = (text, marker) => text.split(marker).length - 1;
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
const functionSource = (code, name) => code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0] ?? null;

function replaceExact(code, marker, replacement, expected, label) {
  const found = count(code, marker);
  if (found !== expected) throw new Error(`square SIMT marker mismatch for ${label}: expected ${expected}, found ${found}`);
  return code.replaceAll(marker, replacement);
}

function declarations(names, type) {
  return names.map((name) => `  var ${name} = ${type}(0.0);`).join("\n");
}

function fourByFourCompute() {
  const lines = ["    let nx = lid.x & 7u;", "    let c = lid.x;"];
  for (let kq = 0; kq < 8; kq++) {
    lines.push("    {", `      let kq = ${kq}u;`);
    for (let col = 0; col < 4; col++) lines.push(`      let b${col} = vec4<f32>(ws[(c + ${16 * col}u) * 8u + (kq ^ nx)]);`);
    for (let row = 0; row < 3; row++) {
      lines.push(`      let a${row} = vec4<f32>(xs[(lid.y + ${16 * row}u) * 8u + kq]);`);
      lines.push(`      acc${row} += vec4<f32>(dot(a${row}, b0), dot(a${row}, b1), dot(a${row}, b2), dot(a${row}, b3));`);
    }
    lines.push("      if (lid.y < 8u) {");
    lines.push("        let a3 = vec4<f32>(xs[(lid.y + 48u) * 8u + kq]);");
    lines.push("        acc3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));");
    lines.push("      }");
    lines.push("    }");
  }
  return lines.join("\n");
}

function fourByFourStores() {
  const lines = ["  let row = m0 + lid.y;"];
  for (let col = 0; col < 4; col++) {
    lines.push("  {");
    lines.push(`    let col = n0 + lid.x + ${16 * col}u;`);
    lines.push("    if (col < N) {");
    lines.push("      var bias = 0.0;");
    lines.push("      if (p.bias == 1u) { bias = B[col]; }");
    for (let row = 0; row < 3; row++) lines.push(`      mm_store(acc${row}.${"xyzw"[col]}, row + ${16 * row}u, col, bias, T, N, splits, wg.z);`);
    lines.push("      if (lid.y < 8u) { mm_store(acc3." + "xyzw"[col] + ", row + 48u, col, bias, T, N, splits, wg.z); }");
    lines.push("    }");
    lines.push("  }");
  }
  return lines.join("\n");
}

export function square4x4Simt(original) {
  if (original.includes("enable f16;") || !original.includes("@workgroup_size(32, 8)") ||
      !original.includes("let m0 = wg.y * 56u;") || !original.includes("var<workgroup> xs: array<vec4<f32>, 512>;") ||
      (original.match(/^  var acc\d+ = vec2<f32>\(0\.0\);$/gm) || []).length !== 7) {
    throw new Error("4x4 SIMT requires the rendered Q8/F32 R4 BM56 batch kernel");
  }
  let code = original.replace("@workgroup_size(32, 8)", "@workgroup_size(16, 16)");
  const oldDeclarations = code.match(/^  var acc\d+ = vec2<f32>\(0\.0\);$/gm);
  if (oldDeclarations?.length !== 7) throw new Error("4x4 SIMT accumulator declaration mismatch");
  code = code.replace(/^  var acc\d+ = vec2<f32>\(0\.0\);\n/gm, "");
  code = replaceExact(code, "  let kb1 = min(nb, kb0 + per);\n", `  let kb1 = min(nb, kb0 + per);\n${declarations(["acc0", "acc1", "acc2", "acc3"], "vec4<f32>")}\n`, 1, "4x4 accumulators");
  const computeStart = code.indexOf("    let nx = lid.x & 7u;");
  const computeEnd = code.indexOf("\n    workgroupBarrier();\n  }\n\n  let row = m0 + lid.y;", computeStart);
  if (computeStart < 0 || computeEnd < computeStart) throw new Error("4x4 SIMT compute block marker mismatch");
  code = code.slice(0, computeStart) + fourByFourCompute() + code.slice(computeEnd);
  const storeStart = code.indexOf("  let row = m0 + lid.y;", computeStart);
  const mainEnd = code.lastIndexOf("\n}");
  if (storeStart < 0 || mainEnd < storeStart) throw new Error("4x4 SIMT store block marker mismatch");
  code = code.slice(0, storeStart) + fourByFourStores() + code.slice(mainEnd);
  for (const name of ["sx", "mm_splits", "mm_store"]) {
    if (!functionSource(original, name) || functionSource(code, name) !== functionSource(original, name)) throw new Error(`4x4 SIMT changed ${name} semantics`);
  }
  if (!code.includes("let m0 = wg.y * 56u;") || !code.includes("let n0 = wg.x * 64u;") || count(code, "acc0 += vec4<f32>") !== 8 || count(code, "acc1 += vec4<f32>") !== 8 || count(code, "acc2 += vec4<f32>") !== 8 || count(code, "acc3 += vec4<f32>") !== 8) {
    throw new Error("4x4 SIMT changed BM56/BN64 geometry or the eight ordered K32 updates");
  }
  return code;
}

function eightByEightLoader() {
  return `  let lr = li / 4u;
  let lq = li % 4u;
  let xrow0 = m0 + lr;
  let xrow1 = m0 + lr + 64u;
  let sw = lr & 7u;
  var xv0 = vec4<f32>(0.0);
  var xv1 = vec4<f32>(0.0);
  var xv2 = vec4<f32>(0.0);
  var xv3 = vec4<f32>(0.0);
  var wv: array<vec4<f32>, 4>;
  for (var kb = kb0; kb < kb1; kb++) {
    xv0 = vec4<f32>(0.0); xv1 = vec4<f32>(0.0);
    xv2 = vec4<f32>(0.0); xv3 = vec4<f32>(0.0);
    if (xrow0 < T) {
      let base0 = (xrow0 * K + kb * 32u) / 4u + lq * 2u;
      xv0 = X[base0]; xv1 = X[base0 + 1u];
    }
    if (xrow1 < T) {
      let base1 = (xrow1 * K + kb * 32u) / 4u + lq * 2u;
      xv2 = X[base1]; xv3 = X[base1 + 1u];
    }
    for (var gg = 0u; gg < 1u; gg++) {
      let wrow0 = n0 + lr + 128u * gg;
      let wrow1 = n0 + lr + 64u + 128u * gg;
      wv[0u] = vec4<f32>(0.0); wv[1u] = vec4<f32>(0.0);
      wv[2u] = vec4<f32>(0.0); wv[3u] = vec4<f32>(0.0);
      if (wrow0 < N) {
        let v0 = W[(wrow0 * K + kb * 32u) / 16u + lq / 2u];
        let s0 = S[wrow0 * nb + kb];
        let lo0 = select(v0.xy, v0.zw, (lq & 1u) == 1u);
        wv[0u] = sx(lo0.x) * s0; wv[1u] = sx(lo0.y) * s0;
      }
      if (wrow1 < N) {
        let v1 = W[(wrow1 * K + kb * 32u) / 16u + lq / 2u];
        let s1 = S[wrow1 * nb + kb];
        let lo1 = select(v1.xy, v1.zw, (lq & 1u) == 1u);
        wv[2u] = sx(lo1.x) * s1; wv[3u] = sx(lo1.y) * s1;
      }
      xs[lr * 8u + lq * 2u] = vec4<f32>(xv0);
      xs[lr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);
      xs[(lr + 64u) * 8u + lq * 2u] = vec4<f32>(xv2);
      xs[(lr + 64u) * 8u + lq * 2u + 1u] = vec4<f32>(xv3);
      let r0 = lr + 128u * gg;
      let r1 = lr + 64u + 128u * gg;
      ws[r0 * 8u + ((lq * 2u) ^ sw)] = wv[0u];
      ws[r0 * 8u + ((lq * 2u + 1u) ^ sw)] = wv[1u];
      ws[r1 * 8u + ((lq * 2u) ^ sw)] = wv[2u];
      ws[r1 * 8u + ((lq * 2u + 1u) ^ sw)] = wv[3u];
    }
`;
}

function eightByEightCompute() {
  const lines = ["    let nx = lid.x & 7u;", "    let c = lid.x;"];
  for (let kq = 0; kq < 8; kq++) {
    lines.push("    {", `      let kq = ${kq}u;`);
    for (let col = 0; col < 8; col++) lines.push(`      let b${col} = vec4<f32>(ws[(c + ${16 * col}u) * 8u + (kq ^ nx)]);`);
    for (let row = 0; row < 8; row++) {
      lines.push(`      let a${row} = vec4<f32>(xs[(lid.y + ${16 * row}u) * 8u + kq]);`);
      lines.push(`      acc${row}a += vec4<f32>(dot(a${row}, b0), dot(a${row}, b1), dot(a${row}, b2), dot(a${row}, b3));`);
      lines.push(`      acc${row}b += vec4<f32>(dot(a${row}, b4), dot(a${row}, b5), dot(a${row}, b6), dot(a${row}, b7));`);
    }
    lines.push("    }");
  }
  return lines.join("\n");
}

function eightByEightStores() {
  const lines = ["  let row = m0 + lid.y;"];
  for (let col = 0; col < 8; col++) {
    lines.push("  {");
    lines.push(`    let col = n0 + lid.x + ${16 * col}u;`);
    lines.push("    if (col < N) {");
    lines.push("      var bias = 0.0;");
    lines.push("      if (p.bias == 1u) { bias = B[col]; }");
    const accSuffix = col < 4 ? "a" : "b", component = "xyzw"[col % 4];
    for (let row = 0; row < 8; row++) lines.push(`      mm_store(acc${row}${accSuffix}.${component}, row + ${16 * row}u, col, bias, T, N, splits, wg.z);`);
    lines.push("    }");
    lines.push("  }");
  }
  return lines.join("\n");
}

export function square8x8Simt(original) {
  if (original.includes("enable f16;") || !original.includes("@workgroup_size(32, 8)") ||
      !original.includes("let m0 = wg.y * 56u;") || !original.includes("let n0 = wg.x * 64u;") ||
      !original.includes("var<workgroup> xs: array<vec4<f32>, 512>;") ||
      (original.match(/^  var acc\d+ = vec2<f32>\(0\.0\);$/gm) || []).length !== 7) {
    throw new Error("8x8 SIMT requires the rendered Q8/F32 R4 BM56 batch kernel");
  }
  let code = original.replace("@workgroup_size(32, 8)", "@workgroup_size(16, 16)");
  code = replaceExact(code, "var<workgroup> xs: array<vec4<f32>, 512>;", "var<workgroup> xs: array<vec4<f32>, 1024>;", 1, "8x8 X tile");
  code = replaceExact(code, "var<workgroup> ws: array<vec4<f32>, 512>;", "var<workgroup> ws: array<vec4<f32>, 1024>;", 1, "8x8 W tile");
  code = replaceExact(code, "let m0 = wg.y * 56u;", "let m0 = wg.y * 128u;", 1, "8x8 physical BM");
  code = replaceExact(code, "let n0 = wg.x * 64u;", "let n0 = wg.x * 128u;", 1, "8x8 physical BN");
  code = code.replace(/^  var acc\d+ = vec2<f32>\(0\.0\);\n/gm, "");
  const anchor = "  let kb1 = min(nb, kb0 + per);\n";
  const names = Array.from({ length: 8 }, (_, row) => [`acc${row}a`, `acc${row}b`]).flat();
  code = replaceExact(code, anchor, `${anchor}${declarations(names, "vec4<f32>")}\n`, 1, "8x8 accumulators");
  const loaderStart = code.indexOf("  let lr = li / 4u;");
  const loaderEnd = code.indexOf("\n    workgroupBarrier();\n    let nx = lid.x & 7u;", loaderStart);
  if (loaderStart < 0 || loaderEnd < loaderStart) throw new Error("8x8 SIMT loader block marker mismatch");
  code = code.slice(0, loaderStart) + eightByEightLoader().trimEnd() + code.slice(loaderEnd);
  const computeStart = code.indexOf("    let nx = lid.x & 7u;");
  const computeEnd = code.indexOf("\n    workgroupBarrier();\n  }\n\n  let row = m0 + lid.y;", computeStart);
  if (computeStart < 0 || computeEnd < computeStart) throw new Error("8x8 SIMT compute block marker mismatch");
  code = code.slice(0, computeStart) + eightByEightCompute() + code.slice(computeEnd);
  const storeStart = code.indexOf("  let row = m0 + lid.y;", computeStart);
  const mainEnd = code.lastIndexOf("\n}");
  if (storeStart < 0 || mainEnd < storeStart) throw new Error("8x8 SIMT store block marker mismatch");
  code = code.slice(0, storeStart) + eightByEightStores() + code.slice(mainEnd);
  for (const name of ["sx", "mm_splits", "mm_store"]) {
    if (!functionSource(original, name) || functionSource(code, name) !== functionSource(original, name)) throw new Error(`8x8 SIMT changed ${name} semantics`);
  }
  if (!code.includes("let m0 = wg.y * 128u;") || !code.includes("let n0 = wg.x * 128u;") ||
      count(code, "array<vec4<f32>, 1024>") !== 2 || names.some((name) => !code.includes(`var ${name} = vec4<f32>(0.0);`))) {
    throw new Error("8x8 SIMT changed its 128x128 tile or named accumulator set");
  }
  for (let row = 0; row < 8; row++) {
    if (count(code, `acc${row}a += vec4<f32>`) !== 8 || count(code, `acc${row}b += vec4<f32>`) !== 8) throw new Error(`8x8 SIMT changed row ${row} K32 update count`);
  }
  return code;
}

function geometry(variant) {
  if (variant === "4x4") return { physicalBM: 56, physicalBN: 64, sharedBytes: 16 * 1024, makeCandidate: square4x4Simt };
  if (variant === "8x8") return { physicalBM: 128, physicalBN: 128, sharedBytes: 32 * 1024, makeCandidate: square8x8Simt };
  throw new Error("variant must be 4x4 or 8x8");
}

export function memoryPlan({ N, K }, variant, splits) {
  const g = geometry(variant), active = T * N, partial = splits * active;
  const inputRows = 128, xBytes = inputRows * K * 4, yBytes = (active + TAIL) * 4, partBytes = (partial + TAIL) * 4;
  const hostWeightsBytes = N * K, hostScalesBytes = N * K / 32 * 4;
  const gpuTemporaryBuffers = {
    X: xBytes, bias: N * 4, globals: 16, params: 16,
    "baseline-Y": yBytes, "candidate-Y": yBytes,
    "baseline-PART": partBytes, "candidate-PART": partBytes,
    "Y-validation": 16, "PART-validation": 16, status: 16,
    "mapped-readback": 256, timestamps: 256,
  };
  const gpuTemporaryBytes = Object.values(gpuTemporaryBuffers).reduce((sum, bytes) => sum + bytes, 0);
  const hostStageBytes = CHUNK, queuedUploadAllowanceBytes = CHUNK, mappedHostCopyBytes = 256;
  const controlledScratchBytes = gpuTemporaryBytes + hostStageBytes + queuedUploadAllowanceBytes + mappedHostCopyBytes;
  return { splits, active, partial, inputRows, xBytes, yBytes, partBytes, gpuTemporaryBuffers, gpuTemporaryBytes,
    hostStageBytes, queuedUploadAllowanceBytes, mappedHostCopyBytes, controlledScratchBytes, limitBytes: LIMIT,
    separatelyReportedQ8WeightBytes: hostWeightsBytes, separatelyReportedScaleBytes: hostScalesBytes,
    candidateWorkgroupStorageBytes: g.sharedBytes, baselineWorkgroupStorageBytes: 16 * 1024,
    paddedOutputFloats: TAIL };
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

async function uploadGenerated(device, buffer, byteLength, bytesPerValue, valueAt, stage, deadline, label) {
  const stageI8 = new Int8Array(stage), stageF32 = new Float32Array(stage), raw = new Uint8Array(stage);
  for (let offset = 0; offset < byteLength; offset += CHUNK) {
    const bytes = Math.min(CHUNK, byteLength - offset), target = bytesPerValue === 1 ? stageI8 : stageF32;
    for (let i = 0; i < bytes / bytesPerValue; i++) target[i] = valueAt(offset / bytesPerValue + i);
    device.queue.writeBuffer(buffer, offset, raw, 0, bytes);
    await drain(device, deadline, label);
  }
}

async function compile(device, source, variant) {
  const g = geometry(variant), spec = { weights: "q8", f16: false, rows: 4, row56: true, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul_gemma_batch", spec), candidateCode = g.makeCandidate(baselineCode), reduceCode = source("reduce_gemma_batch", spec);
  const mmLayout = matmulLayout(device), rLayout = reduceLayout(device);
  const [baseline, candidate, reduce, validator] = await Promise.all([
    pipeline(device, baselineCode, "square-simt.baseline.r4-bm56-bn64", device.createPipelineLayout({ bindGroupLayouts: [mmLayout] })),
    pipeline(device, candidateCode, `square-simt.candidate.${variant}`, device.createPipelineLayout({ bindGroupLayouts: [mmLayout] })),
    pipeline(device, reduceCode, "square-simt.baseline.reducer", device.createPipelineLayout({ bindGroupLayouts: [rLayout] })),
    createGemma4BatchMlpValidator(device),
  ]);
  return { baseline, candidate, reduce, validator, mmLayout, rLayout,
    hashes: { baseline: await hash(baselineCode), candidate: await hash(candidateCode), reducer: await hash(reduceCode) } };
}

async function checkCase(runtime, pipes, shape, variant) {
  const { device } = runtime, g = geometry(variant), splits = mmSplits(1, shape.N, shape.K, TARGET, BM64, BN64);
  const memory = memoryPlan(shape, variant, splits);
  if (splits !== shape.expectedSplits) throw new Error(`${shape.shape} split policy changed: expected ${shape.expectedSplits}, received ${splits}`);
  if (memory.controlledScratchBytes > LIMIT) throw new Error(`${shape.shape} controlled scratch exceeds 64 MiB`);
  if (device.limits.maxComputeWorkgroupStorageSize < g.sharedBytes) return { ...shape, variant, splits, ok: true, skipped: true, reason: `needs-${g.sharedBytes}-bytes-workgroup-storage`, memory };
  if (device.limits.maxComputeWorkgroupSizeX < 16 || device.limits.maxComputeWorkgroupSizeY < 16 || device.limits.maxComputeInvocationsPerWorkgroup < 256) {
    return { ...shape, variant, splits, ok: true, skipped: true, reason: "device-workgroup-limits", memory };
  }
  const maxBinding = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if ([shape.N * shape.K, memory.xBytes, shape.N * 4, memory.yBytes, memory.partBytes].some((size) => size > maxBinding)) throw new Error(`${shape.shape} exceeds a device buffer limit`);
  const deadline = performance.now() + DEADLINE_MS, U = GPUBufferUsage, owned = [];
  let read, querySet, mapped = false;
  const allocated = { gpuTemporaryBytes: 0, weightsBytes: 0, scalesBytes: 0 };
  const make = (size, usage, label, category = "gpuTemporaryBytes") => {
    allocated[category] += size;
    if (allocated.gpuTemporaryBytes > memory.gpuTemporaryBytes) throw new Error("square SIMT GPU scratch exceeded its memory plan");
    if (allocated.weightsBytes > memory.separatelyReportedQ8WeightBytes || allocated.scalesBytes > memory.separatelyReportedScaleBytes) throw new Error("square SIMT immutable weights exceeded their memory plan");
    const buffer = device.createBuffer({ label: `square-simt.${label}`, size: Math.max(16, size), usage }); owned.push(buffer); return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const work = (pass, length) => { const n = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535)); };
  try {
    const weights = make(shape.N * shape.K, U.STORAGE | U.COPY_DST, "q8-weights", "weightsBytes");
    const scales = make(shape.N * shape.K / 32 * 4, U.STORAGE | U.COPY_DST, "q8-scales", "scalesBytes");
    const x = make(memory.xBytes, U.STORAGE | U.COPY_DST, "input");
    const bias = make(shape.N * 4, U.STORAGE | U.COPY_DST, "bias");
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const outputs = [make(memory.yBytes, U.STORAGE | U.COPY_DST, "baseline-output"), make(memory.yBytes, U.STORAGE | U.COPY_DST, "candidate-output")];
    const partials = [make(memory.partBytes, U.STORAGE | U.COPY_DST, "baseline-partials"), make(memory.partBytes, U.STORAGE | U.COPY_DST, "candidate-partials")];
    const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "validation-status");
    const yParams = make(16, U.UNIFORM | U.COPY_DST, "output-validation-params");
    const partParams = make(16, U.UNIFORM | U.COPY_DST, "partial-validation-params");
    read = make(256, U.MAP_READ | U.COPY_DST, "mapped-readback");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");
    if (allocated.gpuTemporaryBytes > memory.gpuTemporaryBytes) throw new Error("square SIMT allocated more GPU scratch than planned");
    const stage = new ArrayBuffer(CHUNK);
    await uploadGenerated(device, weights, shape.N * shape.K, 1,
      (i) => (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) % 256 - 128, stage, deadline, "weight upload");
    await uploadGenerated(device, scales, shape.N * shape.K / 32 * 4, 4,
      (i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037), stage, deadline, "scale upload");
    await uploadGenerated(device, x, memory.xBytes, 4, (i) => {
      const row = Math.floor(i / shape.K), k = i % shape.K;
      if (row >= T) return NaN;
      if (i % 127 === 0) return i % 2 ? -0 : 0;
      return Math.fround(0.04 * Math.sin(k * 0.127 + row * 0.13 + 0.3));
    }, stage, deadline, "input upload");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, 1, T, 0]));

    const active = T * shape.N, partialActive = splits * active;
    const checks = [
      { length: active + TAIL, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params: yParams, baseline: outputs[0], candidate: outputs[1], status }) },
      { length: partialActive + TAIL, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params: partParams, baseline: partials[0], candidate: partials[1], status }) },
    ];
    const ops = outputs.map((output, index) => ({
      group: group(pipes.mmLayout, [globals, params, x, weights, scales, bias, output, partials[index]]),
      reduce: group(pipes.rLayout, [globals, params, partials[index], bias, output]),
    }));
    const dispatch = (pass, candidate) => {
      const index = Number(candidate), pipelineValue = candidate ? pipes.candidate : pipes.baseline;
      pass.setPipeline(pipelineValue); pass.setBindGroup(0, ops[index].group);
      pass.dispatchWorkgroups(Math.ceil(shape.N / (candidate ? g.physicalBN : BN64)), Math.ceil(T / (candidate ? g.physicalBM : 56)), splits);
      if (splits > 1) {
        pass.setPipeline(pipes.reduce); pass.setBindGroup(0, ops[index].reduce);
        work(pass, active);
      }
    };
    const setMode = async (mode, hasBias) => {
      device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, mode, hasBias]));
      const poison = mode === 1 ? 0x3d000000 : 0x7fc00000;
      device.queue.writeBuffer(yParams, 0, new Uint32Array([memory.yBytes / 4, active, SENTINEL_BITS, poison]));
      device.queue.writeBuffer(partParams, 0, new Uint32Array([memory.partBytes / 4, partialActive, SENTINEL_BITS, 0x7fc00000]));
      await uploadGenerated(device, bias, shape.N * 4, 4,
        (i) => hasBias ? i % 257 === 0 ? -0 : Math.fround(0.003 * Math.sin(i * 0.17 + 0.3)) : NaN,
        stage, deadline, "bias upload");
    };
    const verify = async (mode, hasBias) => {
      await setMode(mode, hasBias);
      const encoder = device.createCommandEncoder({ label: `square-simt.${variant}.${shape.shape}.mode-${mode}.bias-${hasBias}` });
      encoder.clearBuffer(status, 0, 16);
      const fill = encoder.beginComputePass(); fill.setPipeline(pipes.validator.fill);
      for (const item of checks) for (const binding of item.fill) { fill.setBindGroup(0, binding); work(fill, item.length); }
      fill.end();
      const compute = encoder.beginComputePass(); dispatch(compute, false); dispatch(compute, true); compute.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(pipes.validator.compare);
      for (const item of checks) { compare.setBindGroup(0, item.compare); work(compare, item.length); }
      compare.end(); encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      if (!await mapBeforeDeadline(read, 16, deadline)) throw new Error("square SIMT parity deadline exceeded");
      mapped = true;
      const counts = new Uint32Array(read.getMappedRange(0, 16).slice(0)); read.unmap(); mapped = false;
      const result = { mode, hasBias: Boolean(hasBias), ok: counts.every((value) => value === 0), bitMismatches: counts[0], activeNonfinite: counts[1], tailMismatches: counts[2], reserved: counts[3] };
      if (!result.ok) throw new Error(`square SIMT output/PART/tail parity failed: ${JSON.stringify(result)}`);
      return result;
    };
    const guard = async () => {
      const results = [];
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) results.push(await verify(mode, hasBias));
      return results;
    };
    const before = await guard();
    let timing = null;
    if (device.features.has("timestamp-query")) {
      await setMode(0, 0);
      const querySet = device.createQuerySet({ type: "timestamp", count: 12 });
      try {
        const warmEncoder = device.createCommandEncoder(), warm = warmEncoder.beginComputePass();
        dispatch(warm, false); dispatch(warm, true); warm.end(); device.queue.submit([warmEncoder.finish()]);
        const encoder = device.createCommandEncoder({ label: `square-simt.${variant}.${shape.shape}.timing` });
        const order = [];
        for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
          const index = order.length * 2; order.push(candidate);
          const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
          for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, candidate);
          pass.end();
        }
        encoder.resolveQuerySet(querySet, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96);
        device.queue.submit([encoder.finish()]);
        if (!await mapBeforeDeadline(read, 96, deadline)) throw new Error("square SIMT timing deadline exceeded");
        mapped = true;
        const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
        const samples = { baseline: [], candidate: [] };
        order.forEach((candidate, index) => samples[candidate ? "candidate" : "baseline"].push(Number(stamps[2 * index + 1] - stamps[2 * index]) / 1e6 / 4));
        if ([...samples.baseline, ...samples.candidate].some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("invalid square SIMT timestamps");
        const choice = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
        timing = { ...samples, ...choice, candidateMs: choice.wideMs, samples: 3, repetitions: 4, includes: "Q8/F32 matmul plus original split reducer", allocationsAndUploadsTimed: false, mode: 0, hasBias: false };
      } finally { querySet.destroy(); }
      const after = await guard();
      const ok = before.every((result, i) => result.ok && after[i]?.ok);
      return { ...shape, variant, T, f16: false, splitT: 1, splits, ok, before, after, timing, memory,
        geometry: { baseline: { workgroup: [32, 8], physicalBM: 56, physicalBN: 64, accumulatorsPerThread: 14, sharedBytes: 16384 }, candidate: { workgroup: [16, 16], physicalBM: g.physicalBM, physicalBN: g.physicalBN, sharedBytes: g.sharedBytes, ...(variant === "4x4" ? { accumulatorsPerThread: 16, rowMask: "lane-y < 8 for the fourth 16-row group" } : { accumulatorsPerThread: 64 }) },
        splitGeometry: { virtualBM: BM64, virtualBN: BN64, splitTarget: TARGET, unchangedSplitPartitions: true }, inputs: { activeRows: T, baselinePaddedRows: Math.ceil(T / 56) * 56 - T, candidatePaddingRows: Math.ceil(T / g.physicalBM) * g.physicalBM - T, NaNInputPadding: true }, exactChecks: { outputsAndAllSplitPartials: true, allModesAndBiasBranches: true, activeFinite: true, signedZeroBitwise: true, fourFloatOutputAndPartialTails: true } } };
    }
    const after = await guard();
    const ok = before.every((result, i) => result.ok && after[i]?.ok);
    return { ...shape, variant, T, f16: false, splitT: 1, splits, ok, before, after, timing: null, memory,
      geometry: { baseline: { workgroup: [32, 8], physicalBM: 56, physicalBN: 64, accumulatorsPerThread: 14, sharedBytes: 16384 }, candidate: { workgroup: [16, 16], physicalBM: g.physicalBM, physicalBN: g.physicalBN, sharedBytes: g.sharedBytes }, splitGeometry: { virtualBM: BM64, virtualBN: BN64, splitTarget: TARGET, unchangedSplitPartitions: true } },
      reason: "timestamp-query-unavailable" };
  } finally { if (mapped) read?.unmap(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  const variant = params.get("variant") || "4x4", g = geometry(variant);
  const shapes = SHAPES.filter(({ shape }) => !params.has("shape") || params.get("shape") === shape);
  if (!shapes.length) throw new Error("shape must be gate, down, or q");
  if (params.has("shape") && !SHAPES.some(({ shape }) => shape === params.get("shape"))) throw new Error("shape must be gate, down, or q");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const { device } = runtime;
    const limits = device.limits;
    metadata = { adapter: { name: runtime.name, features: [...device.features], maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize }, variant,
      candidateWorkgroup: [16, 16], candidateSharedBytes: g.sharedBytes, f16: false, limitBytes: LIMIT,
      requiredWorkgroup: { maxComputeWorkgroupSizeX: 32, maxComputeWorkgroupSizeY: 16, maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupStorageSize: g.sharedBytes }, immutableQ8WeightsSeparateFromScratch: true };
    const unsupported = limits.maxComputeWorkgroupStorageSize < g.sharedBytes ? `candidate requires ${g.sharedBytes} bytes of workgroup storage` :
      limits.maxComputeWorkgroupSizeX < 32 || limits.maxComputeWorkgroupSizeY < 16 || limits.maxComputeInvocationsPerWorkgroup < 256 ?
        "candidate or baseline exceeds the adapter's workgroup invocation limits" : null;
    if (unsupported) {
      const skippedCases = shapes.map((shape) => ({ ...shape, variant, skipped: true, reason: unsupported }));
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", check: "gemma4-matmul-square-simt", ...metadata,
        skipped: true, reason: unsupported, elapsedMs: performance.now() - started, cases: skippedCases };
      log(window.gpuBench);
      return;
    }
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(device, source, variant), `compiling square SIMT ${variant} pipelines`);
    metadata.shaderHashes = pipes.hashes;
    for (const shape of shapes) {
      log({ status: "checking", shape: shape.shape, variant });
      const result = await withGpuErrors(runtime, () => checkCase(runtime, pipes, shape, variant), `square SIMT ${variant} ${shape.shape}`);
      cases.push(result); window.gpuBench.completed = cases.length; log(result);
      if (!result.ok) throw new Error(`${variant} ${shape.shape} failed full output/PART exactness`);
    }
    const timed = cases.filter((item) => item.timing);
    const metricMs = timed.length ? Math.exp(timed.reduce((sum, item) => sum + Math.log(item.timing.candidateMs), 0) / timed.length) : performance.now() - started;
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matmul-square-simt", ...metadata,
      metricMs, elapsedMs: performance.now() - started,
      method: timed.length ? "candidate GPU timestamp milliseconds, complete Q8/F32 matmul and unchanged split reducer; alternating 3 pairs x 4 repetitions" : "untimed exact guard wall milliseconds; timestamp-query unavailable",
      correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matmul-square-simt", completed: 0 };
  main().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
    log(error.stack || String(error));
  });
}
