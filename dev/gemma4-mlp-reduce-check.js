import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, rowsPerThread, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const MAX_TEMPORARY = 64 * MiB;
const CHUNK_BYTES = MiB;
const DEADLINE_MS = 30_000;
const SENTINEL = 12345.25;
const SAMPLES = 3;
const REPETITIONS = 4;
const TARGET = 256;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

async function beforeDeadline(operation, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("MLP reducer guard deadline exceeded");
  let timer;
  try {
    return await Promise.race([operation, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("MLP reducer guard deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

function exact(expected, actual) {
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  const b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) {
      bitMismatches++;
      firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] };
    }
  }
  return { ok: expected.length === actual.length && bitMismatches === 0 && nonfinite === 0, samples: expected.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

function memoryFor(T, N, K, splits) {
  const plane = splits * T * N * 4, output = T * N * 4, input = T * K * 4;
  const fixed = input + CHUNK_BYTES * 2 + 1024; // Input, small readback, poison chunk and uniforms.
  const staging = 4 * CHUNK_BYTES;
  const peaks = {
    baselinePart: fixed + plane + 16 + 2 * (plane + 16) + staging,
    candidatePart: fixed + 2 * plane + 16 + plane + 16 + CHUNK_BYTES + staging,
    baselineOutput: fixed + plane + 16 + 3 * (output + 16) + 2 * (output + 16) + staging,
    candidateOutput: fixed + 2 * plane + 16 + output + 16 + output + 16 + CHUNK_BYTES + staging,
  };
  return { temporaryLimitBytes: MAX_TEMPORARY, temporaryPeakBytes: Math.max(...Object.values(peaks)), peaks,
    sharedWeightBytes: 2 * N * K * 9 / 8, hostWeightFixturePeakBytes: N * K * 9 / 8,
    weightAccounting: "immutable resident weights are separate from the temporary CPU/GPU scratch cap", chunkBytes: CHUNK_BYTES };
}

async function pipelines(device, source, config) {
  const spec = { weights: "q8", f16: config.f16, rows: config.rows, row56: config.row56, groups: 1, splitTarget: TARGET };
  const code = { mm: source(config.kernel + "_gemma_batch", spec), reduce: source("reduce_gemma_batch", spec), gelu: source("gemma4_gelu"), reduceGelu: source("reduce_gelu", spec) };
  const mmLayout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const [mm, reduce, gelu, reduceGelu, hashes] = await Promise.all([
    pipeline(device, code.mm, "mlp_reduce_guard.mm", device.createPipelineLayout({ bindGroupLayouts: [mmLayout] })),
    pipeline(device, code.reduce, "mlp_reduce_guard.reduce", device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] })),
    pipeline(device, code.gelu, "mlp_reduce_guard.gelu"),
    pipeline(device, code.reduceGelu, "mlp_reduce_guard.reduce_gelu"),
    Promise.all(Object.entries(code).map(async ([name, value]) => [name, await hash(value)])).then(Object.fromEntries),
  ]);
  return { mm, reduce, gelu, reduceGelu, mmLayout, reductionLayout, hashes };
}

async function check(runtime, source, { T, N, K }, kind, f16) {
  const kernel = kind === "wide" ? "matmul_wide" : "matmul";
  const rows = rowsPerThread(T, { f16, groups: 1, kernel });
  const row56 = kind === "generic" && !f16 && rows === 4 && Math.ceil(T / 56) === Math.ceil(T / 64);
  const physicalBM = row56 ? 56 : 16 * rows;
  const splits = mmSplits(1, N, K, TARGET, 16 * rows, 64);
  const config = { T, N, K, kind, kernel, f16, rows, row56, physicalBM, splitTokens: 1, splitTarget: TARGET, splits };
  if (splits === 1) return { ...config, ok: true, skipped: true, reason: "unsplit-projection-keeps-original-MLP" };
  const memory = memoryFor(T, N, K, splits);
  if (memory.temporaryPeakBytes > MAX_TEMPORARY) throw new Error("MLP reducer temporary scratch exceeds 64 MiB");
  const { device } = runtime, U = GPUBufferUsage, deadline = performance.now() + DEADLINE_MS;
  const planeBytes = splits * T * N * 4, outputBytes = T * N * 4;
  if (planeBytes % device.limits.minStorageBufferOffsetAlignment) throw new Error("MLP partial plane is not storage aligned");
  if (Math.max(2 * planeBytes + 16, N * K, T * K * 4) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("MLP reducer case exceeds device buffer limits");
  const owned = [];
  let queries;
  const make = (size, usage, label, data, target = owned) => {
    const buffer = device.createBuffer({ size, usage, label: `mlp-reduce.${label}` }); target.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const drain = () => beforeDeadline(device.queue.onSubmittedWorkDone(), deadline);
  try {
    const pipes = await pipelines(device, source, config);
    const weightsFor = (offset) => {
      const values = new Int8Array(N * K), scales = new Float32Array(N * K / 32);
      for (let i = 0; i < values.length; i++) values[i] = (i * 31 + (i >>> 5) * 17 + offset) % 256 - 128;
      for (let i = 0; i < scales.length; i++) scales[i] = i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + (i + offset) % 11 * 0.000037);
      return { values: make(values.byteLength, U.STORAGE | U.COPY_DST, `weights-${offset}`, values), scales: make(scales.byteLength, U.STORAGE | U.COPY_DST, `scales-${offset}`, scales) };
    };
    const gateWeight = weightsFor(0); await drain();
    const upWeight = weightsFor(19); await drain();
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals", new Uint32Array([T, 0, 0, 0]));
    const params = make(16, U.UNIFORM | U.COPY_DST, "params", new Uint32Array([N, K, 0, 0]));
    const pairParams = make(16, U.UNIFORM | U.COPY_DST, "pair-params", new Uint32Array([N, K, 1, 0]));
    const geluParams = make(16, U.UNIFORM | U.COPY_DST, "gelu-params", new Uint32Array([N, 0, 0, 0]));
    const zero = make(16, U.STORAGE, "zero");
    const input = make(T * K * 4, U.STORAGE | U.COPY_DST, "input", Float32Array.from({ length: T * K }, (_, i) => Math.fround(0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)) || 0.00390625));
    const smallRead = make(CHUNK_BYTES, U.COPY_DST | U.MAP_READ, "small-read");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    queries = device.createQuerySet({ type: "timestamp", count: 2 });
    const poison = new Float32Array(CHUNK_BYTES / 4).fill(NaN), tail = new Float32Array(4).fill(SENTINEL);
    const poisonBuffer = async (buffer, length) => {
      for (let offset = 0; offset < length; offset += CHUNK_BYTES) {
        device.queue.writeBuffer(buffer, offset, poison.buffer, 0, Math.min(CHUNK_BYTES, length - offset));
        if ((offset / CHUNK_BYTES) % 4 === 3) await drain();
      }
      device.queue.writeBuffer(buffer, length, tail); await drain();
    };
    const snapshot = async (buffer, offset, size) => {
      const read = size <= CHUNK_BYTES ? smallRead : device.createBuffer({ size, usage: U.COPY_DST | U.MAP_READ, label: "mlp-reduce.large-read" });
      let mapped = false;
      try {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, offset, read, 0, size); device.queue.submit([encoder.finish()]);
        const withinDeadline = await mapBeforeDeadline(read, size, deadline); mapped = true;
        if (!withinDeadline) throw new Error("MLP reducer readback deadline exceeded");
        return new Float32Array(read.getMappedRange(0, size).slice(0));
      } finally { if (mapped) read.unmap(); if (read !== smallRead) read.destroy(); }
    };
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((resource, binding) => ({ binding, resource: resource.buffer ? resource : { buffer: resource } })) });
    const arena = async (candidate, partialOnly = false) => {
      const buffers = [];
      try {
        const create = (size, label) => make(size, U.STORAGE | U.COPY_SRC | U.COPY_DST, label, null, buffers);
        const part = create((candidate ? 2 : 1) * planeBytes + 16, "partials");
        await poisonBuffer(part, (candidate ? 2 : 1) * planeBytes);
        const output = create(partialOnly ? 16 : outputBytes + 16, "output");
        await poisonBuffer(output, partialOnly ? 0 : outputBytes);
        const gate = candidate || partialOnly ? output : create(outputBytes + 16, "gate");
        const up = candidate || partialOnly ? output : create(outputBytes + 16, "up");
        if (!candidate && !partialOnly) { await poisonBuffer(gate, outputBytes); await poisonBuffer(up, outputBytes); }
        const op = (weights, y, partView) => ({
          group: group(pipes.mmLayout, [globals, params, input, weights.values, weights.scales, zero, y, partView]),
          reduce: group(pipes.reductionLayout, [globals, params, partView, zero, y]),
        });
        const gatePart = candidate ? { buffer: part, offset: 0, size: planeBytes } : part;
        const upPart = candidate ? { buffer: part, offset: planeBytes, size: planeBytes + 16 } : part;
        const ops = [op(gateWeight, gate, gatePart), op(upWeight, up, upPart)];
        const encodeMM = (pass, projection) => {
          pass.setPipeline(pipes.mm); pass.setBindGroup(0, ops[projection].group);
          pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / physicalBM), splits);
        };
        const geluGroup = !candidate && !partialOnly ? group(pipes.gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, output]) : null;
        const reduceGeluGroup = candidate && !partialOnly ? group(pipes.reduceGelu.getBindGroupLayout(0), [globals, pairParams, part, output]) : null;
        const dispatch = (pass) => {
          for (let projection = 0; projection < 2; projection++) {
            encodeMM(pass, projection);
            if (!candidate) {
              pass.setPipeline(pipes.reduce); pass.setBindGroup(0, ops[projection].reduce);
              const work = Math.ceil(T * N / 256); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
            }
          }
          pass.setPipeline(candidate ? pipes.reduceGelu : pipes.gelu); pass.setBindGroup(0, candidate ? reduceGeluGroup : geluGroup);
          const work = Math.ceil(T * N / 256); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
        };
        return { part, output, gate, up, encodeMM, dispatch, buffers, destroy() { for (const buffer of buffers) buffer.destroy(); } };
      } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
    };
    const submit = (encode) => { const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); encode(pass); pass.end(); device.queue.submit([encoder.finish()]); };
    const tailOK = (values) => values.subarray(values.length - 4).every((value) => value === SENTINEL);
    const compareChunks = async (expected, buffer, offset, length) => {
      const result = { ok: true, samples: 0, bitMismatches: 0, nonfinite: 0, maxAbs: 0, firstDifference: null, signedZeroChecked: true };
      for (let start = 0; start < length; start += CHUNK_BYTES) {
        const size = Math.min(CHUNK_BYTES, length - start), values = await snapshot(buffer, offset + start, size);
        const check = exact(expected.subarray(start / 4, (start + size) / 4), values);
        result.ok &&= check.ok; result.samples += check.samples; result.bitMismatches += check.bitMismatches; result.nonfinite += check.nonfinite; result.maxAbs = Math.max(result.maxAbs, check.maxAbs);
        if (!result.firstDifference && check.firstDifference) result.firstDifference = { ...check.firstDifference, index: start / 4 + check.firstDifference.index };
      }
      return result;
    };
    const partialChecks = [];
    for (let projection = 0; projection < 2; projection++) {
      let baseline, candidate, expected;
      try {
        baseline = await arena(false, true); submit((pass) => baseline.encodeMM(pass, projection));
        expected = await snapshot(baseline.part, 0, planeBytes + 16);
        const baselineTail = tailOK(expected) && tailOK(await snapshot(baseline.output, 0, 16)); baseline.destroy(); baseline = null;
        candidate = await arena(true, true); submit((pass) => candidate.encodeMM(pass, projection));
        const check = await compareChunks(expected, candidate.part, projection * planeBytes, planeBytes);
        const candidateTail = tailOK(await snapshot(candidate.part, 2 * planeBytes, 16));
        const unusedOutputTail = tailOK(await snapshot(candidate.output, 0, 16));
        partialChecks.push({ projection: projection ? "up" : "gate", ...check, tailGuard: baselineTail && candidateTail && unusedOutputTail });
      } finally { baseline?.destroy(); candidate?.destroy(); }
    }
    let baseline, candidate, outputCheck, outputTail;
    try {
      baseline = await arena(false); submit(baseline.dispatch);
      const expected = await snapshot(baseline.output, 0, outputBytes + 16);
      const baselineTail = tailOK(expected) && tailOK(await snapshot(baseline.gate, outputBytes, 16)) && tailOK(await snapshot(baseline.up, outputBytes, 16));
      const baselinePartTail = tailOK(await snapshot(baseline.part, planeBytes, 16));
      baseline.destroy(); baseline = null;
      candidate = await arena(true); submit(candidate.dispatch);
      outputCheck = await compareChunks(expected, candidate.output, 0, outputBytes);
      outputTail = baselineTail && baselinePartTail && tailOK(await snapshot(candidate.output, outputBytes, 16)) && tailOK(await snapshot(candidate.part, 2 * planeBytes, 16));
    } finally { baseline?.destroy(); candidate?.destroy(); }
    const ok = outputCheck.ok && outputTail && partialChecks.every((check) => check.ok && check.tailGuard);
    let timing = null;
    if (ok && device.features.has("timestamp-query")) {
      const samples = { baseline: [], candidate: [] }, order = [];
      for (let sample = 0; sample < SAMPLES; sample++) for (const fused of sample % 2 ? [true, false] : [false, true]) {
        const current = await arena(fused);
        try {
          submit(current.dispatch); await drain();
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
          for (let repeat = 0; repeat < REPETITIONS; repeat++) current.dispatch(pass);
          pass.end(); encoder.resolveQuerySet(queries, 0, 2, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, smallRead, 0, 16); device.queue.submit([encoder.finish()]);
          const withinDeadline = await mapBeforeDeadline(smallRead, 16, deadline);
          if (!withinDeadline) { smallRead.unmap(); throw new Error("MLP reducer timing deadline exceeded"); }
          const stamps = new BigInt64Array(smallRead.getMappedRange(0, 16)), ms = Number(stamps[1] - stamps[0]) / 1e6 / REPETITIONS; smallRead.unmap();
          const name = fused ? "candidate" : "baseline"; samples[name].push(ms); order.push({ sample, kind: name });
        } finally { current.destroy(); }
      }
      const choice = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
      timing = { ...samples, order, samples: SAMPLES, repetitions: REPETITIONS, baselineMedianMs: choice.genericMs, candidateMedianMs: choice.wideMs,
        benefit: choice.benefit, stable: choice.stable, candidateWins: choice.kernel === "wide", reason: choice.reason.replaceAll("generic", "baseline").replaceAll("wide", "candidate"), allocationsAndUploadsTimed: false };
    }
    return { ...config, ok, outputCheck, partialChecks, tailGuard: outputTail, timing, baselineDispatches: 5, candidateDispatches: 3, memory, shaderHashes: pipes.hashes };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const kind = params.get("kernel") || "generic", precision = params.get("precision") || "auto";
  if (!["generic", "wide"].includes(kind)) throw new Error("kernel must be generic or wide");
  if (!["auto", "f32", "f16"].includes(precision)) throw new Error("precision must be auto, f32, or f16");
  const tokens = params.has("tokens") ? [Number(params.get("tokens"))] : [97, 241];
  if (tokens.some((T) => ![97, 241].includes(T))) throw new Error("tokens must be 97 or 241");
  const runtime = await requestDevice({ features: precision === "f32" ? ["timestamp-query"] : ["shader-f16", "timestamp-query"] });
  try {
    const f16 = runtime.device.features.has("shader-f16");
    if (precision === "f16" && !f16) throw new Error("shader-f16 is unavailable");
    if (!runtime.device.features.has("timestamp-query")) throw new Error("timestamp-query is required for this probe");
    const source = await kernelSource();
    metadata = { kernel: kind, precision: f16 ? "f16" : "f32", wasmUrl: source.wasmUrl, adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features] } };
    log(metadata);
    for (const T of tokens) for (const N of [6144, 12288]) {
      log({ status: "checking", T, N, K: 1536 });
      const result = await withGpuErrors(runtime, () => check(runtime, source, { T, N, K: 1536 }, kind, f16), `MLP reducer T${T} ${N}x1536`);
      cases.push(result); log(result);
      if (!result.ok) throw new Error("MLP reducer fusion bit-exact parity failed");
      if (result.timing && result.timing.baselineMedianMs === null) throw new Error("MLP reducer timestamps were invalid");
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMedianMs), 0) / timed.length) : performance.now() - started,
      method: "GPU timestamp geometric mean; ordinary gate/up matmuls with fused split reducers and GELU; phased temporary arenas exclude allocation/upload time", elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
  log(error.stack || String(error));
});
