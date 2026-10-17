import { dispatchMatmul, matmulLayout, mmSplits, pipeline, Profiler, reduceLayout, requestDevice, rowsPerThread, withGpuErrors } from "../js/src/gpu.js";
import { encodePairedMatmulGelu, pairedMatmulGeluBindGroup, pairedMatmulGeluPipelines } from "./gemma4-matmul-gelu.js";
import { kernelSource } from "./wgsl.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const SENTINEL = 12345.25;
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function exact(a, b, label) {
  if (a.length !== b.length) throw new Error(`${label}: length mismatch`);
  const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const bBits = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let maxAbs = 0;
  let bitMismatches = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) throw new Error(`${label}: nonfinite value at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (aBits[i] !== bBits[i]) bitMismatches++;
  }
  return { ok: bitMismatches === 0, samples: a.length, bitMismatches, maxAbs };
}

async function check(runtime, source, { T, N, K, splitT = T, saturated = false }, kind, f16, candidateRowsOverride = null) {
  const { device } = runtime;
  const U = GPUBufferUsage;
  const owned = [];
  let profiler;
  const make = (size, usage, data, label = "paired prefill guard") => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage, label });
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  try {
    const batch = splitT === 1 && T !== 1;
    const rowProbe = candidateRowsOverride !== null;
    if (rowProbe && (!batch || T !== 97)) throw new Error("candidateRows requires an independent T97 batch");
    const kernel = kind === "wide" ? "matmul_wide" : kind === "k64" ? "matmul_k64" : "matmul";
    if (batch && kind === "k64") throw new Error("K64 batch baseline is not registered");
    const rows = rowsPerThread(T, { f16, groups: 1, kernel });
    const candidateRows = candidateRowsOverride ?? rowsPerThread(T, { f16, groups: 1, kernel: "matmul" });
    const row56 = !f16 && rows === 4 && Math.ceil(T / 56) === Math.ceil(T / 64);
    const candidateRow56 = !f16 && candidateRows === 4 && Math.ceil(T / 56) === Math.ceil(T / 64);
    const target = 256;
    const baselineSplits = mmSplits(splitT, N, K, target, 16 * rows, 64);
    const candidateSplits = mmSplits(splitT, N, K, target, 16 * candidateRows, 64);
    if (baselineSplits !== candidateSplits) throw new Error("paired guard requires identical split-K partitions");
    const splits = baselineSplits;
    const chunkBytes = 1024 * 1024, deadline = performance.now() + 30_000;
    const scratchBytes = T * K * 8 + T * N * 16 + (splits > 1 ? 4 * splits * T * N * 4 : 128) + 8 * chunkBytes + 2048;
    if (rowProbe && scratchBytes > 64 * 1024 * 1024) throw new Error("candidateRows temporary scratch exceeds 64 MiB");
    const spec = { weights: "q8", f16, rows, row56, groups: 1, splitTarget: target };
    const mmCode = source(batch ? `${kernel}_gemma_batch` : kernel, spec);
    const reduceCode = source(batch ? "reduce_gemma_batch" : "reduce", spec);
    const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
    const [mm, reduce, gelu, candidate] = await Promise.all([
      pipeline(device, mmCode, "paired_guard_control", device.createPipelineLayout({ bindGroupLayouts: [layout] })),
      pipeline(device, reduceCode, "paired_guard_reduce", device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] })),
      pipeline(device, source("gemma4_gelu"), "paired_guard_gelu"),
      pairedMatmulGeluPipelines(device, source, { f16, rows: candidateRows, row56: candidateRow56, splitTarget: target }),
    ]);
    const weightsFor = (offset) => {
      const values = Int8Array.from({ length: N * K }, (_, i) => (i * 31 + Math.floor(i / K) * 17 + offset) % 256 - 128);
      const scales = Float32Array.from({ length: N * K / 32 }, (_, i) => (i % 2 ? -1 : 1) * (0.0002 + i % 7 * 0.000037));
      if (saturated) for (let row = 0; row < Math.min(2, N); row++) {
        values.fill(offset ? 17 : row ? -128 : 127, row * K, (row + 1) * K);
        scales.fill(0.05, row * K / 32, (row + 1) * K / 32);
      }
      return { values: make(values.byteLength, U.STORAGE | U.COPY_DST, values), scales: make(scales.byteLength, U.STORAGE | U.COPY_DST, scales) };
    };
    const gateWeights = weightsFor(0), upWeights = weightsFor(19);
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 0, 0, 0]));
    const params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 0, 0]));
    const pairParams = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, splitT, 0]));
    const geluParams = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, 0, 0, 0]));
    const xValues = Float32Array.from({ length: T * K }, (_, i) => saturated ? 0.32 * (1 + Math.floor(i / K) / 8) : Math.fround(0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)) || 0.00390625);
    const x = make(xValues.byteLength, U.STORAGE | U.COPY_DST, xValues);
    const zero = make(16, U.STORAGE);
    const gate = make(T * N * 4, U.STORAGE), up = make(T * N * 4, U.STORAGE);
    const partLength = splits > 1 ? splits * T * N : 0;
    const yBytes = (T * N + 4) * 4, gatePartBytes = (Math.max(4, partLength) + 4) * 4, pairPartBytes = (Math.max(4, partLength * 2) + 4) * 4;
    const poison = rowProbe ? new Float32Array(chunkBytes / 4).fill(NaN) : null;
    const beforeDeadline = async (operation, cancel = () => {}) => {
      const remaining = deadline - performance.now();
      if (remaining <= 0) { cancel(); throw new Error("candidateRows deadline exceeded"); }
      let timer;
      try {
        return await Promise.race([operation, new Promise((_, reject) => {
          timer = setTimeout(() => { cancel(); reject(new Error("candidateRows deadline exceeded")); }, remaining);
        })]);
      } finally { clearTimeout(timer); }
    };
    const guardedBuffer = async (length) => {
      const size = (length + 4) * 4, usage = U.STORAGE | U.COPY_SRC | U.COPY_DST;
      if (!rowProbe) {
        const values = new Float32Array(length + 4).fill(NaN); values.fill(SENTINEL, length);
        return make(size, usage, values);
      }
      const buffer = make(size, usage);
      for (let offset = 0; offset < length * 4; offset += chunkBytes) {
        if (performance.now() >= deadline) throw new Error("candidateRows upload deadline exceeded");
        device.queue.writeBuffer(buffer, offset, poison.buffer, 0, Math.min(chunkBytes, length * 4 - offset));
        await beforeDeadline(device.queue.onSubmittedWorkDone());
      }
      device.queue.writeBuffer(buffer, length * 4, new Float32Array(4).fill(SENTINEL));
      return buffer;
    };
    const baselineY = await guardedBuffer(T * N), candidateY = await guardedBuffer(T * N);
    const gatePart = await guardedBuffer(Math.max(4, partLength)), upPart = await guardedBuffer(Math.max(4, partLength)), pairPart = await guardedBuffer(Math.max(4, partLength * 2));
    const group = (l, buffers) => device.createBindGroup({ layout: l, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const opFor = (weights, output, partials) => ({
      N, K,
      group: group(layout, [globals, params, x, weights.values, weights.scales, zero, output, partials]),
      reduce: group(reductionLayout, [globals, params, partials, zero, output]),
    });
    const gateOp = opFor(gateWeights, gate, gatePart), upOp = opFor(upWeights, up, upPart);
    const geluGroup = group(gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, baselineY]);
    const fusedOp = { N, K, ...pairedMatmulGeluBindGroup(device, candidate, { globals, params: pairParams, input: x, gate: gateWeights.values, gateScales: gateWeights.scales, up: upWeights.values, upScales: upWeights.scales, output: candidateY, partials: pairPart }) };
    const dispatch = (pass, fused) => {
      if (fused) return encodePairedMatmulGelu(pass, candidate, fusedOp, T, splitT);
      for (const op of [gateOp, upOp]) dispatchMatmul(pass, mm, reduce, op, T, target, rows, 1, T, splitT);
      pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup);
      const work = Math.ceil(T * N / 256);
      pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
    };
    const toRead = [[baselineY, yBytes], [candidateY, yBytes], [gatePart, gatePartBytes], [upPart, gatePartBytes], [pairPart, pairPartBytes]];
    const readBytes = toRead.reduce((sum, [, size]) => sum + size, 0);
    const read = make(rowProbe ? chunkBytes : readBytes, U.COPY_DST | U.MAP_READ);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, false); dispatch(pass, true); pass.end();
    let offset = 0;
    if (!rowProbe) for (const [buffer, size] of toRead) { encoder.copyBufferToBuffer(buffer, 0, read, offset, size); offset += size; }
    device.queue.submit([encoder.finish()]);
    let outputCheck, partialChecks, chunks;
    if (rowProbe) {
      const snapshot = async (buffer, start, size) => {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, start, read, 0, size); device.queue.submit([encoder.finish()]);
        let mapped = false;
        try {
          const withinDeadline = await mapBeforeDeadline(read, size, deadline); mapped = true;
          if (!withinDeadline) throw new Error("candidateRows parity deadline exceeded");
          return new Float32Array(read.getMappedRange(0, size).slice(0));
        } finally { if (mapped) read.unmap(); }
      };
      const compare = async (left, right, rightOffset, length, label) => {
        const result = { ok: true, samples: 0, bitMismatches: 0, maxAbs: 0 };
        for (let start = 0; start < length * 4; start += chunkBytes) {
          const size = Math.min(chunkBytes, length * 4 - start);
          const a = await snapshot(left, start, size), b = await snapshot(right, rightOffset + start, size), check = exact(a, b, label);
          result.ok &&= check.ok; result.samples += check.samples; result.bitMismatches += check.bitMismatches; result.maxAbs = Math.max(result.maxAbs, check.maxAbs);
        }
        return result;
      };
      outputCheck = await compare(baselineY, candidateY, 0, T * N, "paired output");
      partialChecks = partLength ? [await compare(gatePart, pairPart, 0, partLength, "gate partials"), await compare(upPart, pairPart, partLength * 4, partLength, "up partials")] : [];
      for (const [buffer, size] of toRead) if (!(await snapshot(buffer, size - 16, 16)).every((value) => value === SENTINEL)) throw new Error("paired output or partial tail overwrite");
    } else {
      await read.mapAsync(GPUMapMode.READ);
      const readValues = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
      offset = 0;
      chunks = toRead.map(([, size]) => { const values = readValues.subarray(offset / 4, (offset + size) / 4); offset += size; return values; });
      outputCheck = exact(chunks[0].subarray(0, T * N), chunks[1].subarray(0, T * N), "paired output");
      partialChecks = partLength ? [exact(chunks[2].subarray(0, partLength), chunks[4].subarray(0, partLength), "gate partials"), exact(chunks[3].subarray(0, partLength), chunks[4].subarray(partLength, partLength * 2), "up partials")] : [];
      for (const values of chunks) for (let i = values.length - 4; i < values.length; i++) if (values[i] !== SENTINEL) throw new Error("paired output or partial tail overwrite");
    }
    const ok = outputCheck.ok && partialChecks.every((value) => value.ok);
    if (!ok) return { T, N, K, splitT, f16, kind, rows, candidateRows, row56, candidateRow56, splits, ok, outputCheck, partialChecks };
    if (saturated && chunks) for (let row = 0; row < T; row++) if (chunks[0][row * N] <= 10 || chunks[0][row * N + 1] !== 0) throw new Error("GELU saturation cases were not exercised");
    let timing = null;
    if (N >= 6144 && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 4);
      const samples = { baseline: [], candidate: [] }, repetitions = 4;
      for (let sample = 0; sample < (rowProbe ? 3 : 7); sample++) {
        const encoder = device.createCommandEncoder();
        for (const fused of sample % 2 ? [true, false] : [false, true]) {
          const pass = profiler.pass(encoder, fused ? "candidate" : "baseline");
          for (let i = 0; i < repetitions; i++) dispatch(pass, fused);
          pass.end();
        }
        profiler.finish(encoder); device.queue.submit([encoder.finish()]);
        const measured = rowProbe ? await beforeDeadline(profiler.collect(), () => profiler.read.unmap()) : await profiler.collect();
        if (rowProbe || sample >= 2) for (const label of ["baseline", "candidate"]) samples[label].push(measured[label] / repetitions);
      }
      timing = { ...samples, baselineMedianMs: median(samples.baseline), candidateMedianMs: median(samples.candidate) };
    }
    return {
      T, N, K, splitT, f16, kind, rows, candidateRows, row56, candidateRow56, splits, saturated, ok, outputCheck, partialChecks, tailGuard: true,
      baselineDispatches: splits > 1 ? 5 : 3, candidateDispatches: splits > 1 ? 2 : 1, sharedBytes: candidate.sharedBytes, timing,
      ...(rowProbe ? { scratchBytes, scratchLimitBytes: 64 * 1024 * 1024, sharedWeightBytes: 2 * N * K * 9 / 8, readbackChunkBytes: chunkBytes } : {}),
      shaderHashes: { baseline: await hash(mmCode), baselineReduce: await hash(reduceCode), candidate: await hash(candidate.mmCode), candidateReduce: await hash(candidate.reduceCode) },
    };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const f32 = params.get("precision") === "f32";
  const runtime = await requestDevice({ ...(f32 ? { features: ["timestamp-query"] } : {}) });
  const f16 = runtime.device.features.has("shader-f16");
  const kind = params.get("kernel") || (f16 ? "wide" : "generic");
  try {
    if (!["generic", "wide", "k64"].includes(kind)) throw new Error("kernel must be generic, wide, or k64");
    if (kind === "wide" && !f16) throw new Error("wide baseline requires shader-f16");
    const source = await kernelSource(), cases = [];
    const batchOnly = params.get("batch") === "1";
    const candidateRows = params.has("candidateRows") ? Number(params.get("candidateRows")) : null;
    if (candidateRows !== null && (!batchOnly || ![1, 2, 3, 4].includes(candidateRows))) throw new Error("candidateRows requires batch=1 and rows 1..4");
    const tokens = params.has("T") ? [Number(params.get("T"))] : candidateRows !== null ? [97] : [97, 241];
    if (tokens.some((T) => ![97, 241].includes(T)) || candidateRows !== null && tokens.some((T) => T !== 97)) throw new Error("candidateRows supports T97 only");
    const widths = params.has("N") ? [Number(params.get("N"))] : [6144, 12288];
    if (widths.some((N) => ![6144, 12288].includes(N))) throw new Error("N must be 6144 or 12288");
    if (batchOnly && kind === "k64") throw new Error("batch MLP guard requires a generic or wide baseline");
    const casesToRun = batchOnly ? [
      ...tokens.flatMap((T) => widths.map((N) => ({ T, N, K: 1536, splitT: 1 }))),
    ] : params.get("quick") === "1" ? [] : [
      { T: 1, N: 1, K: 32 },
      { T: 42, N: 17, K: 96, saturated: true },
      { T: 55, N: 17, K: 1056 },
      { T: 97, N: 65, K: 1536 },
      ...(kind === "k64" ? [] : [{ T: 5, N: 6144, K: 1536, splitT: 1 }]),
    ];
    if (!batchOnly) for (const T of [42, 55, 89, 97]) casesToRun.push({ T, N: 6144, K: 1536 });
    for (const input of casesToRun) {
      const result = await withGpuErrors(runtime, () => check(runtime, source, input, kind, f16, candidateRows), `paired prefill T${input.T} ${input.N}x${input.K}`);
      cases.push(result); log(result);
      if (!result.ok) throw Object.assign(new Error("paired prefill exact parity failed"), { cases });
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", metricMs: performance.now() - started,
      adapter: { name: runtime.name, isFallbackAdapter: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter), features: [...runtime.device.features], maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize },
      correctness: { ok: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", correctness: { ok: false, cases: error.cases || [] }, error: String(error?.message || error) }; log(error.stack || error); });
