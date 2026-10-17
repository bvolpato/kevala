import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const LIMIT = 64 * 1024 * 1024, SENTINEL = 12345.25, TARGET = 256;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function widerTile(original) {
  if (!original.includes("@workgroup_size(32, 8)") || original.includes("enable f16;")) throw new Error("BN128 requires the unrolled F32 generic kernel");
  let code = original;
  const replace = (from, to, count = 1) => {
    if (code.split(from).length - 1 !== count) throw new Error(`BN128 transform marker mismatch: ${from}`);
    code = code.replaceAll(from, to);
  };
  replace("var<workgroup> ws: array<vec4<f32>, 512>;", "var<workgroup> ws: array<vec4<f32>, 1024>;");
  replace("var wv: array<vec4<f32>, 2>;", "var wv: array<vec4<f32>, 4>;");
  replace("let n0 = wg.x * 64u;", "let n0 = wg.x * 128u;");
  replace("for (var gg = 0u; gg < 1u; gg++)", "for (var gg = 0u; gg < 2u; gg++)", 2);
  const accumulators = [...code.matchAll(/var acc(\d) = vec2<f32>\(0\.0\);/g)];
  if (![4, 7, 8].includes(accumulators.length)) throw new Error("BN128 unexpected accumulator row count");
  for (const [, i] of accumulators) {
    replace(`var acc${i} = vec2<f32>(0.0);`, `var acc${i} = vec4<f32>(0.0);`);
    replace(`acc${i} += vec2<f32>(dot(a${i}, b0), dot(a${i}, b1));`, `acc${i} += vec4<f32>(dot(a${i}, b0), dot(a${i}, b1), dot(a${i}, b2), dot(a${i}, b3));`, 8);
  }
  const b1 = "let b1 = vec4<f32>(ws[(c + 32u) * 8u + (kq ^ nx)]);";
  replace(b1, `${b1}\n      let b2 = vec4<f32>(ws[(c + 64u) * 8u + (kq ^ nx)]);\n      let b3 = vec4<f32>(ws[(c + 96u) * 8u + (kq ^ nx)]);`, 8);
  const firstMarker = "  {\n    let col = n0 + lid.x + 0u;", secondMarker = "  {\n    let col = n0 + lid.x + 32u;";
  if (code.split(firstMarker).length !== 2 || code.split(secondMarker).length !== 2) throw new Error("BN128 store marker count mismatch");
  const first = code.indexOf(firstMarker), second = code.indexOf(secondMarker, first), end = code.lastIndexOf("\n}");
  if (first < 0 || second < first || end < second) throw new Error("BN128 store blocks were not recognized");
  const block = code.slice(first, second);
  if ((block.match(/mm_store\(acc\d\.x,/g) || []).length !== accumulators.length || block.split("if (col < N)").length !== 2) throw new Error("BN128 guarded store block mismatch");
  const extra = (offset, component) => block.replace("let col = n0 + lid.x + 0u;", `let col = n0 + lid.x + ${offset}u;`).replace(/mm_store\(acc(\d)\.x,/g, `mm_store(acc$1.${component},`);
  return code.slice(0, end) + "\n" + extra(64, "z") + extra(96, "w") + code.slice(end);
}

async function drainBeforeDeadline(device, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("BN128 upload deadline exceeded");
  let timer;
  try {
    await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("BN128 upload deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

function exact(a, b) {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (x[i] !== y[i]) { bitMismatches++; firstDifference ||= { index: i, baselineBits: x[i], candidateBits: y[i], baseline: a[i], candidate: b[i] }; }
  }
  return { ok: a.length === b.length && !bitMismatches && !nonfinite, samples: a.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

async function compile(device, source, cache, config) {
  const key = `${config.policy}:${config.rows}:${config.row56}`;
  if (cache.has(key)) return cache.get(key);
  const spec = { weights: "q8", f16: false, rows: config.rows, row56: config.row56, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul" + config.policy, spec), candidateCode = widerTile(baselineCode), reduceCode = source("reduce" + config.policy, spec);
  const mmLayout = matmulLayout(device), rLayout = reduceLayout(device), pl = device.createPipelineLayout({ bindGroupLayouts: [mmLayout] });
  const [baseline, candidate, reduce, baselineHash, candidateHash, reduceHash] = await Promise.all([
    pipeline(device, baselineCode, "matmul_bn64", pl), pipeline(device, candidateCode, "matmul_bn128", pl),
    pipeline(device, reduceCode, "matmul_bn64_reduce", device.createPipelineLayout({ bindGroupLayouts: [rLayout] })), hash(baselineCode), hash(candidateCode), hash(reduceCode),
  ]);
  const value = { baseline, candidate, reduce, mmLayout, rLayout, shaderHashes: { baseline: baselineHash, candidate: candidateHash, reduce: reduceHash } };
  cache.set(key, value);
  return value;
}

async function check(runtime, source, cache, c) {
  const { T, N, K, rows, row56, splitT } = c;
  const splits = mmSplits(splitT, N, K, TARGET, rows * 16, 64), length = T * N;
  const partialLength = splits > 1 ? splits * length : 0, yBytes = (length + 4) * 4, partBytes = (partialLength + 4) * 4, stride = yBytes + partBytes;
  const scratchBytes = 6 * stride + 2 * T * K * 4 + 2048;
  if (scratchBytes > LIMIT) return { ...c, ok: true, skipped: true, reason: "temporary-memory-limit", scratchBytes };
  const { device } = runtime, U = GPUBufferUsage, deadline = performance.now() + 30_000;
  if (device.limits.maxComputeWorkgroupStorageSize < 24 * 1024) return { ...c, ok: true, skipped: true, reason: "requires-24-KiB-workgroup-storage" };
  if (Math.max(2 * stride, N * K, T * K * 4) > device.limits.maxBufferSize || Math.max(partBytes, yBytes, N * K, T * K * 4) > device.limits.maxStorageBufferBindingSize) throw new Error("BN128 case exceeds device buffer limits");
  const pipes = await compile(device, source, cache, c), owned = [];
  let query, read, mapped = false;
  const make = (size, usage, data) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const weights = new Int8Array(N * K), scales = new Float32Array(N * K / 32);
    for (let i = 0; i < weights.length; i++) weights[i] = (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) % 256 - 128;
    for (let i = 0; i < scales.length; i++) scales[i] = i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037);
    const w = make(weights.byteLength, U.STORAGE | U.COPY_DST, weights), s = make(scales.byteLength, U.STORAGE | U.COPY_DST, scales);
    const x = make(T * K * 4, U.STORAGE | U.COPY_DST, Float32Array.from({ length: T * K }, (_, i) => Math.fround(0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)) || 0.00390625));
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, splitT, splitT, 0])), params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 0, 0]));
    const zero = make(16, U.STORAGE);
    const poisoned = async (size) => {
      const data = new Float32Array(size / 4).fill(NaN); data.fill(SENTINEL, data.length - 4);
      const buffer = make(size, U.STORAGE | U.COPY_DST | U.COPY_SRC, data);
      await drainBeforeDeadline(device, deadline);
      return buffer;
    };
    const ys = [await poisoned(yBytes), await poisoned(yBytes)], parts = [await poisoned(partBytes), await poisoned(partBytes)];
    const ops = ys.map((y, index) => ({ group: group(pipes.mmLayout, [globals, params, x, w, s, zero, y, parts[index]]), reduce: group(pipes.rLayout, [globals, params, parts[index], zero, y]) }));
    const dispatch = (pass, candidate) => {
      const index = candidate ? 1 : 0;
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline); pass.setBindGroup(0, ops[index].group);
      pass.dispatchWorkgroups(Math.ceil(N / (candidate ? 128 : 64)), Math.ceil(T / (row56 ? 56 : rows * 16)), splits);
      if (splits > 1) {
        const work = Math.ceil(length / 256);
        pass.setPipeline(pipes.reduce); pass.setBindGroup(0, ops[index].reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    read = make(Math.max(256, 2 * stride), U.MAP_READ | U.COPY_DST);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end();
    for (let i = 0; i < 2; i++) { encoder.copyBufferToBuffer(ys[i], 0, read, i * stride, yBytes); encoder.copyBufferToBuffer(parts[i], 0, read, i * stride + yBytes, partBytes); }
    device.queue.submit([encoder.finish()]);
    const withinParityDeadline = await mapBeforeDeadline(read, 2 * stride, deadline); mapped = true;
    if (!withinParityDeadline) throw new Error("BN128 parity deadline exceeded");
    const data = new Float32Array(read.getMappedRange(0, 2 * stride).slice(0)); read.unmap(); mapped = false;
    const outputCheck = exact(data.subarray(0, length), data.subarray(stride / 4, stride / 4 + length));
    const partialCheck = partialLength ? exact(data.subarray(yBytes / 4, yBytes / 4 + partialLength), data.subarray((stride + yBytes) / 4, (stride + yBytes) / 4 + partialLength)) : null;
    const tailGuard = [(yBytes - 16) / 4, (stride - 16) / 4, (stride + yBytes - 16) / 4, (2 * stride - 16) / 4].every((start) => data.subarray(start, start + 4).every((value) => value === SENTINEL));
    const ok = outputCheck.ok && (!partialCheck || partialCheck.ok) && tailGuard;
    let timing = null;
    if (ok && device.features.has("timestamp-query")) {
      query = device.createQuerySet({ type: "timestamp", count: 12 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC), encoder = device.createCommandEncoder(), warmup = encoder.beginComputePass();
      dispatch(warmup, false); dispatch(warmup, true); warmup.end();
      const indices = { baseline: [], candidate: [] };
      let index = 0;
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
        indices[candidate ? "candidate" : "baseline"].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, candidate);
        pass.end(); index += 2;
      }
      encoder.resolveQuerySet(query, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
      const withinTimingDeadline = await mapBeforeDeadline(read, 96, deadline); mapped = true;
      if (!withinTimingDeadline) throw new Error("BN128 timing deadline exceeded");
      const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
      const samples = Object.fromEntries(Object.entries(indices).map(([key, values]) => [key, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4)]));
      const choice = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
      timing = { ...samples, samples: 3, repetitions: 4, baselineMs: choice.genericMs, candidateMs: choice.wideMs, benefit: choice.benefit, stable: choice.stable, selected: choice.kernel === "wide", reason: choice.reason };
    }
    return { ...c, f16: false, splits, virtualColumns: 64, baselineColumns: 64, candidateColumns: 128, ok, outputCheck, partialCheck, tailGuard, timing,
      scratchBytes, sharedWeightBytes: weights.byteLength + scales.byteLength, workgroupStorageBytes: { baseline: 16384, candidate: 24576 }, shaderHashes: pipes.shaderHashes };
  } finally { if (mapped) read.unmap(); query?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const configurations = [
    { scope: "batch", T: 97, splitT: 1, rows: 4, row56: true, policy: "_gemma_batch" },
    { scope: "balanced", T: 51, splitT: 128, rows: 4, row56: true, policy: "_gemma_source" },
    { scope: "fast", T: 28, splitT: 64, rows: 2, row56: false, policy: "_gemma_source" },
  ].filter(({ scope }) => !params.has("scope") || params.get("scope") === scope);
  const shapes = [{ shape: "ragged", N: 65, K: 1056 }, { shape: "gate", N: 6144, K: 1536 }, { shape: "down", N: 1536, K: 6144 }, { shape: "q", N: 2048, K: 1536 }, { shape: "o", N: 1536, K: 2048 }].filter(({ shape }) => !params.has("shape") || params.get("shape") === shape);
  if (!configurations.length) throw new Error("scope must be batch, balanced, or fast");
  if (!shapes.length) throw new Error("shape must be ragged, gate, down, q, or o");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const source = await kernelSource(), cache = new Map();
    metadata = { precision: "f32", wasmUrl: source.wasmUrl, adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features], maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize } };
    log(metadata);
    for (const config of configurations) for (const shape of shapes) {
      log({ status: "checking", ...config, ...shape });
      const result = await withGpuErrors(runtime, () => check(runtime, source, cache, { ...config, ...shape }), "BN128 exact matmul guard"); cases.push(result); log(result);
      if (!result.ok) throw new Error("BN128 matmul bit-exact parity failed");
      if (result.timing && result.timing.baselineMs === null) throw new Error("BN128 timestamps were invalid");
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMs), 0) / timed.length) : performance.now() - started,
      method: timed.length ? "GPU timestamps, physical BN128 versus BN64 with original virtual BN64 split partition" : "untimed guard wall milliseconds; timestamp-query or 24 KiB workgroup storage unavailable",
      elapsedMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
  log(error.stack || String(error));
});
