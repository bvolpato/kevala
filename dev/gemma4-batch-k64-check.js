import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const T = 97, ROWS = 4, BM = 56, TARGET = 256, CHUNK = 1024 * 1024, LIMIT = 64 * CHUNK, SENTINEL = 12345.25;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function batchK64(k64, batchReduce) {
  const splitFunction = (code) => {
    const matches = [...code.matchAll(/fn mm_splits\(T: u32, N: u32, K: u32\) -> u32 \{[\s\S]*?\n\}/g)];
    if (matches.length !== 1) throw new Error("K64 batch transform requires exactly one mm_splits function");
    return matches[0][0];
  };
  const original = splitFunction(k64), independent = splitFunction(batchReduce);
  if (!original.includes("((T + 63u) / 64u)") || !independent.includes("let tiles = ((N + 63u) / 64u);")) throw new Error("K64 batch transform split-policy marker mismatch");
  if (!k64.includes("@workgroup_size(32, 8)") || !k64.includes("let m0 = wg.y * 56u;") || k64.includes("enable f16;")) throw new Error("K64 batch guard requires F32 R4/BM56");
  if (!k64.includes("if (block + 1u < kb1) {")) throw new Error("K64 odd K32 tail guard is missing");
  for (let kq = 0; kq < 16; kq++) if (k64.split(`accumulate(${kq}u,`).length !== 2) throw new Error(`K64 ordered dot marker mismatch at ${kq}`);
  return k64.replace(original, independent);
}

async function drain(device, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("K64 batch upload deadline exceeded");
  let timer;
  try {
    await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("K64 batch upload deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

function exact(a, b, offset = 0) {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (x[i] !== y[i]) { bitMismatches++; firstDifference ||= { index: offset + i, baselineBits: x[i], candidateBits: y[i], baseline: a[i], candidate: b[i] }; }
  }
  return { ok: a.length === b.length && !bitMismatches && !nonfinite, samples: a.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

async function compile(device, source) {
  const spec = { weights: "q8", f16: false, rows: ROWS, row56: true, groups: 1, splitTarget: TARGET };
  const baselineCode = source("matmul_gemma_batch", spec), reduceCode = source("reduce_gemma_batch", spec), originalK64 = source("matmul_k64", spec);
  const candidateCode = batchK64(originalK64, reduceCode), mmLayout = matmulLayout(device), rLayout = reduceLayout(device);
  const layout = device.createPipelineLayout({ bindGroupLayouts: [mmLayout] });
  const [baseline, candidate, reduce, baselineHash, candidateHash, reduceHash, originalK64Hash] = await Promise.all([
    pipeline(device, baselineCode, "batch_k32_control", layout), pipeline(device, candidateCode, "batch_k64_candidate", layout),
    pipeline(device, reduceCode, "batch_k64_reduce", device.createPipelineLayout({ bindGroupLayouts: [rLayout] })),
    hash(baselineCode), hash(candidateCode), hash(reduceCode), hash(originalK64),
  ]);
  return { baseline, candidate, reduce, mmLayout, rLayout, shaderHashes: { baseline: baselineHash, candidate: candidateHash, reduce: reduceHash, originalK64: originalK64Hash } };
}

async function check(runtime, pipes, c) {
  const { N, K, fixture = "mixed" } = c, { device } = runtime, U = GPUBufferUsage;
  const splits = mmSplits(1, N, K, TARGET, ROWS * 16, 64), length = T * N, partLength = splits > 1 ? splits * length : 0;
  const yBytes = (length + 4) * 4, partBytes = (Math.max(4, partLength) + 4) * 4, sharedWeightBytes = N * K * 9 / 8;
  // Include host-side weight upload arrays; immutable GPU weights are reported separately.
  const scratchBytes = 2 * (yBytes + partBytes) + 2 * T * K * 4 + sharedWeightBytes + 8 * CHUNK + 4096;
  if (scratchBytes > LIMIT) return { ...c, ok: true, skipped: true, reason: "temporary-memory-limit", scratchBytes, sharedWeightBytes };
  if (Math.max(yBytes, partBytes, N * K, T * K * 4) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("K64 batch case exceeds device buffer limits");
  const deadline = performance.now() + 30_000, owned = [];
  let read, mapped = false, query;
  const make = (size, usage, data) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const weights = Int8Array.from({ length: N * K }, (_, i) => (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) % 256 - 128);
    const scales = Float32Array.from({ length: N * K / 32 }, (_, i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
    const w = make(weights.byteLength, U.STORAGE | U.COPY_DST, weights), s = make(scales.byteLength, U.STORAGE | U.COPY_DST, scales);
    await drain(device, deadline);
    const xValues = Float32Array.from({ length: T * K }, (_, i) => fixture === "zero" ? (i & 1 ? -0 : 0) : fixture === "cancellation" ? (i & 1 ? -0.03125 : 0.03125) : Math.fround(0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)) || 0.00390625);
    const x = make(xValues.byteLength, U.STORAGE | U.COPY_DST, xValues), zero = make(16, U.STORAGE);
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 1, 1, 0])), params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 0, 0]));
    const poison = new Float32Array(CHUNK / 4).fill(NaN);
    const guarded = async (size) => {
      const buffer = make(size, U.STORAGE | U.COPY_SRC | U.COPY_DST);
      for (let offset = 0; offset < size - 16; offset += CHUNK) {
        device.queue.writeBuffer(buffer, offset, poison.buffer, 0, Math.min(CHUNK, size - 16 - offset));
        await drain(device, deadline);
      }
      device.queue.writeBuffer(buffer, size - 16, new Float32Array(4).fill(SENTINEL));
      return buffer;
    };
    const ys = [await guarded(yBytes), await guarded(yBytes)], parts = [await guarded(partBytes), await guarded(partBytes)];
    const ops = ys.map((y, i) => ({ group: group(pipes.mmLayout, [globals, params, x, w, s, zero, y, parts[i]]), reduce: group(pipes.rLayout, [globals, params, parts[i], zero, y]) }));
    const dispatch = (pass, candidate) => {
      const i = candidate ? 1 : 0;
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline); pass.setBindGroup(0, ops[i].group);
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / BM), splits);
      if (splits > 1) {
        const work = Math.ceil(length / 256);
        pass.setPipeline(pipes.reduce); pass.setBindGroup(0, ops[i].reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); device.queue.submit([encoder.finish()]);
    read = make(CHUNK, U.COPY_DST | U.MAP_READ);
    const snapshot = async (buffer, offset, bytes) => {
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, offset, read, 0, bytes); device.queue.submit([encoder.finish()]);
      const withinDeadline = await mapBeforeDeadline(read, bytes, deadline); mapped = true;
      try {
        if (!withinDeadline) throw new Error("K64 batch parity deadline exceeded");
        return new Float32Array(read.getMappedRange(0, bytes).slice(0));
      } finally { read.unmap(); mapped = false; }
    };
    const compare = async (buffers, count) => {
      const result = { ok: true, samples: 0, bitMismatches: 0, nonfinite: 0, maxAbs: 0, firstDifference: null, signedZeroChecked: true };
      for (let offset = 0; offset < count * 4; offset += CHUNK) {
        const bytes = Math.min(CHUNK, count * 4 - offset), a = await snapshot(buffers[0], offset, bytes), b = await snapshot(buffers[1], offset, bytes), check = exact(a, b, offset / 4);
        result.ok &&= check.ok; result.samples += check.samples; result.bitMismatches += check.bitMismatches; result.nonfinite += check.nonfinite; result.maxAbs = Math.max(result.maxAbs, check.maxAbs); result.firstDifference ||= check.firstDifference;
      }
      return result;
    };
    const outputCheck = await compare(ys, length), partialCheck = partLength ? await compare(parts, partLength) : null;
    let tailGuard = true, unusedPartUntouched = true;
    for (const [buffers, bytes] of [[ys, yBytes], [parts, partBytes]]) for (const buffer of buffers) tailGuard &&= (await snapshot(buffer, bytes - 16, 16)).every((value) => value === SENTINEL);
    if (!partLength) for (const buffer of parts) unusedPartUntouched &&= (await snapshot(buffer, 0, 16)).every(Number.isNaN);
    const ok = outputCheck.ok && (!partialCheck || partialCheck.ok) && tailGuard && unusedPartUntouched;
    let timing = null;
    if (ok && fixture === "mixed" && device.features.has("timestamp-query")) {
      query = device.createQuerySet({ type: "timestamp", count: 12 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC), encoder = device.createCommandEncoder(), warmup = encoder.beginComputePass();
      dispatch(warmup, false); dispatch(warmup, true); warmup.end();
      const indices = { baseline: [], candidate: [] }; let index = 0;
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
        indices[candidate ? "candidate" : "baseline"].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, candidate);
        pass.end(); index += 2;
      }
      encoder.resolveQuerySet(query, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
      const withinDeadline = await mapBeforeDeadline(read, 96, deadline); mapped = true;
      if (!withinDeadline) throw new Error("K64 batch timing deadline exceeded");
      const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
      const samples = Object.fromEntries(Object.entries(indices).map(([key, values]) => [key, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4)]));
      const choice = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
      timing = { ...samples, samples: 3, repetitions: 4, baselineMs: choice.genericMs, candidateMs: choice.wideMs, benefit: choice.benefit, stable: choice.stable, selected: choice.kernel === "wide", reason: choice.reason };
    }
    const nb = K / 32, per = Math.ceil(nb / splits), splitK32Blocks = Array.from({ length: splits }, (_, z) => Math.max(0, Math.min(nb, (z + 1) * per) - z * per));
    return { ...c, T, splitT: 1, rows: ROWS, physicalBM: BM, virtualBM: 64, f16: false, splits, splitK32Blocks, oddK32TailExercised: splitK32Blocks.some((count) => count % 2),
      ok, outputCheck, partialCheck, tailGuard, unusedPartUntouched, timing, scratchBytes, scratchLimitBytes: LIMIT, sharedWeightBytes, hostWeightUploadBytes: sharedWeightBytes, readbackChunkBytes: CHUNK,
      dispatches: splits > 1 ? 2 : 1, workgroupStorageBytes: { baseline: 16384, candidate: 32768 }, shaderHashes: pipes.shaderHashes };
  } finally { if (mapped) read.unmap(); query?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const shapes = [
    { shape: "gate", N: 6144, K: 1536 }, { shape: "down", N: 1536, K: 6144 },
    { shape: "q", N: 2048, K: 1536 }, { shape: "o", N: 1536, K: 2048 },
    { shape: "gate-unsplit", N: 12288, K: 1536 }, { shape: "ragged", N: 65, K: 1056 },
    { shape: "ragged-unsplit", N: 12289, K: 1056 },
    { shape: "zero", N: 17, K: 1056, fixture: "zero" }, { shape: "cancellation", N: 65, K: 1056, fixture: "cancellation" },
  ].filter(({ shape }) => !params.has("shape") || params.get("shape") === shape);
  if (!shapes.length) throw new Error("unknown K64 batch shape");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const source = await kernelSource();
    metadata = { precision: "f32", wasmUrl: source.wasmUrl, adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features], maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize } };
    log(metadata);
    if (runtime.device.limits.maxComputeWorkgroupStorageSize < 32768) throw new Error("K64 F32 requires 32 KiB workgroup storage");
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source), "compiling suffix K64 guard");
    for (const shape of shapes) {
      log({ status: "checking", T, ...shape });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, shape), "suffix K64 exact guard"); cases.push(result); log(result);
      if (!result.ok) throw new Error("K64 batch bit-exact parity failed");
      if (result.timing && result.timing.baselineMs === null) throw new Error("K64 batch timestamps were invalid");
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata, elapsedMs: performance.now() - started,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMs), 0) / timed.length) : performance.now() - started,
      method: "GPU timestamps, F32 suffix K64 versus K32; identical R4/BM56 and logical splitT1 partitions", correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
  log(error.stack || String(error));
});
