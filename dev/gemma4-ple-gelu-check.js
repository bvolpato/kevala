import { mmSplits, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const search = new URLSearchParams(globalThis.location?.search || "");
const BUDGET = 32 * 1024 * 1024;
const TAIL = new Float32Array([12345.25, -1731.5, 913.75, -0]);
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");
const median = (values) => { const sorted = values.toSorted((a, b) => a - b), mid = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2; };

function replaceOne(source, before, after) {
  if (source.split(before).length !== 2) throw new Error(`PLE source marker is missing or ambiguous: ${before}`);
  return source.replace(before, after);
}

export function fusedSource(reduce, gelu) {
  const marker = "fn gelu(x: f32) -> f32 {", start = gelu.indexOf(marker);
  if (start < 0 || gelu.indexOf(marker, start + marker.length) >= 0) throw new Error("registered GELU function marker is missing or ambiguous");
  let end = start + marker.length, depth = 1;
  for (; end < gelu.length && depth; end++) { if (gelu[end] === "{") depth++; else if (gelu[end] === "}") depth--; }
  if (depth) throw new Error("registered GELU function has no closing brace");
  const unchangedGelu = gelu.slice(start, end);
  let code = replaceOne(reduce, "@compute @workgroup_size(256)\n", `@group(0) @binding(5) var<storage, read> PLE: array<f32>;\n@group(0) @binding(6) var<storage, read_write> A: array<f32>;\nvar<workgroup> RAW: array<f32, 256>;\n\n${unchangedGelu}\n\n@compute @workgroup_size(256)\n`);
  code = replaceOne(code, "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {", "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {");
  code = replaceOne(code, "  if (o >= T * N) { return; }", "  if (o < T * N) {");
  code = replaceOne(code, "  Y[o] = v;\n}", "    RAW[lane] = v;\n  }\n  workgroupBarrier();\n  if (o >= T * N) { return; }\n  let x = RAW[lane];\n  A[o] = gelu(x) * PLE[o];\n}");
  return { code, unchangedGelu, workgroupSize: 256, sharedBytes: 1024 };
}

export function probeCases(timingOnly = false) {
  const primary = [
    { T: 28, N: 256, K: 1536, logicalT: 28, policy: "source", rows: 2, splitTarget: 256, splits: 8, pattern: "signed", mode: 0, bias: 0, timing: true },
    { T: 97, N: 256, K: 1536, logicalT: 1, policy: "batch", rows: 4, splitTarget: 256, splits: 8, pattern: "signed", mode: 0, bias: 0, timing: true },
  ];
  if (timingOnly) return primary;
  const out = [...primary];
  for (const policy of ["source", "batch"]) for (const splits of [1, 2, 3, 5, 8]) {
    const T = policy === "source" ? 17 : 43, logicalT = policy === "source" ? 79 : 1, rows = policy === "source" ? 4 : 3, N = 257;
    const tiles = Math.ceil(N / 64) * Math.ceil(logicalT / (16 * rows));
    out.push({ T, N, K: 1536, logicalT, policy, rows, splitTarget: tiles * splits, splits, pattern: "signed", mode: 0, bias: 0 });
  }
  for (const N of [1, 17, 255, 256, 257]) for (const pattern of ["zero", "cancellation", "endpoints"]) {
    const T = 3, rows = 1, logicalT = 1, splits = pattern === "endpoints" ? 1 : 8;
    out.push({ T, N, K: 1536, logicalT, policy: "batch", rows, splitTarget: Math.ceil(N / 64) * splits, splits, pattern, mode: 0, bias: 0 });
  }
  // The runtime proposal uses mode0/bias0. Keep the original epilogue branches live
  // and check them too, so source derivation cannot silently change their order.
  for (const mode of [0, 1, 2]) for (const bias of [0, 1]) out.push({ T: 3, N: 257, K: 1536, logicalT: 1, policy: "batch", rows: 1, splitTarget: 40, splits: 8, pattern: "signed", mode, bias });
  for (const logicalT of [70, 117, 241]) for (const rows of [1, 2, 3, 4]) out.push({ T: 17, N: 513, K: 1536, logicalT, policy: "source", rows, splitTarget: 256,
    splits: mmSplits(logicalT, 513, 1536, 256, 16 * rows, 64), pattern: "signed", mode: 0, bias: 0 });
  return out;
}

export function resourceSize(c) {
  const count = c.T * c.N, outputBytes = (count + TAIL.length) * 4, partBytes = (c.splits * count + TAIL.length) * 4, biasBytes = Math.max(16, c.N * 4);
  // Resident GPU fixtures/readback + CPU fixtures/read copies + upload reserves.
  const gpuBytes = partBytes + 11 * outputBytes + biasBytes + 64;
  const hostBytes = partBytes + 11 * outputBytes + biasBytes;
  const uploadReserveBytes = partBytes + 6 * outputBytes + biasBytes + 64;
  const estimate = gpuBytes + hostBytes + uploadReserveBytes + 4096;
  if (![c.T, c.N, c.K, c.logicalT, c.rows, c.splitTarget, c.splits].every(Number.isSafeInteger) || c.T < 1 || c.T > 256 || c.N < 1 || c.N > 1024 || c.logicalT < 1 || c.logicalT > 256 || c.splits < 1 || c.splits > 8 || estimate > BUDGET) throw new Error(`invalid or oversized PLE probe case: ${estimate} bytes`);
  return { count, outputBytes, partBytes, biasBytes, gpuBytes, hostBytes, uploadReserveBytes, estimatedBytes: estimate };
}

export function fixture(c, size) {
  const part = new Float32Array(size.partBytes / 4).fill(NaN), ple = new Float32Array(size.outputBytes / 4).fill(NaN), seed = new Float32Array(size.outputBytes / 4).fill(NaN), bias = new Float32Array(size.biasBytes / 4).fill(c.bias ? 0 : NaN);
  const endpoints = [-12, -10, -9.999999, -0, 0, 9.999999, 10, 12];
  const cancellation = [1048576, 0.015625, -1048576, 0.00737, -0.25, 0.1731, -0.0317, 0.413];
  for (let s = 0; s < c.splits; s++) for (let i = 0; i < size.count; i++) {
    const at = s * size.count + i;
    if (c.pattern === "zero") part[at] = (i + s) % 2 ? -0 : 0;
    else if (c.pattern === "endpoints") part[at] = s === 0 ? endpoints[i % endpoints.length] : (s + i) % 2 ? -0 : 0;
    else if (c.pattern === "cancellation") part[at] = ((i % 13) - 6) / 8 * cancellation[s];
    else part[at] = (1.337 + s * 0.193) * (s % 2 ? -1 : 1) * (Math.sin(i * 0.017 + s * 0.137) * 0.731 + Math.cos(i * 0.031 - s * 0.193) * 0.417);
  }
  for (let i = 0; i < size.count; i++) {
    ple[i] = i % 13 === 0 ? -0 : i % 17 === 0 ? 0 : (i % 2 ? -1 : 1) * (0.03125 + (i % 23) * 0.07137);
    seed[i] = c.mode === 1 ? (i % 2 ? -1 : 1) * (0.00317 + (i % 19) * 0.01731) : NaN;
  }
  if (c.bias) for (let i = 0; i < c.N; i++) bias[i] = (i % 2 ? -1 : 1) * (0.00173 + (i % 11) * 0.0137);
  seed.set(TAIL, size.count);
  return { part, ple, seed, bias };
}

export function exact(label, expected, actual) {
  if (expected.length !== actual.length) throw new Error(`${label}: length mismatch`);
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length), b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let differences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`${label}: nonfinite active output at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) { differences++; firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] }; }
  }
  const result = { label, bitExact: differences === 0, differences, maxAbs, firstDifference, checkedValues: expected.length, signedZeroChecked: true };
  if (differences) throw new Error(JSON.stringify(result));
  return result;
}

async function checkCase(gpu, kernels, c) {
  const size = resourceSize(c), input = fixture(c, size), device = gpu.device, U = GPUBufferUsage;
  if (mmSplits(c.logicalT, c.N, c.K, c.splitTarget, 16 * c.rows, 64) !== c.splits) throw new Error("baseline host split count differs from fixture");
  const owned = []; let allocatedBytes = 0, queries, read, timingRead, mapped = false, timingMapped = false;
  const make = (size, usage, data, label) => {
    const bytes = Math.max(16, size); allocatedBytes += bytes;
    if (allocatedBytes > sizeLimit) throw new Error("PLE GPU allocation budget exceeded");
    const buffer = device.createBuffer({ size: bytes, usage, label }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const sizeLimit = size.gpuBytes + 4096;
  const bind = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const params = (data) => make(16, U.UNIFORM | U.COPY_DST, new Uint32Array(data), "PLE probe parameters");
  try {
    const globals = params([c.T, c.logicalT, 0, 0]), mm = params([c.N, c.K, c.mode, c.bias]), gelu = params([c.N, 1, 0, 0]);
    const partials = make(size.partBytes, U.STORAGE | U.COPY_DST, input.part, "PLE gate partials and poisoned tail");
    const ple = make(size.outputBytes, U.STORAGE | U.COPY_DST, input.ple, "independent PLE values and poisoned tail");
    const bias = make(size.biasBytes, U.STORAGE | U.COPY_DST, input.bias, "PLE bias or disabled NaN bias");
    const poison = new Float32Array(size.outputBytes / 4).fill(NaN); poison.set(TAIL, size.count);
    const raw = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, input.seed, "original reduced PLE gate");
    const candidateSeed = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, input.seed, "candidate residual seed, disabled NaNs for mode0");
    const baseline = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, "original GELU output");
    const candidate = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, "fused GELU output");
    read = make(4 * size.outputBytes, U.MAP_READ | U.COPY_DST, null, "PLE probe readback");
    const reduceBind = bind(kernels.reduce, [globals, mm, partials, bias, raw]);
    const geluBind = bind(kernels.gelu, [globals, gelu, raw, ple, baseline]);
    const fusedBind = bind(kernels.fused, [globals, mm, partials, bias, candidateSeed, ple, candidate]);
    const dispatch = (pass, next) => {
      if (next) { pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(Math.ceil(size.count / 256)); }
      else {
        pass.setPipeline(kernels.reduce); pass.setBindGroup(0, reduceBind); pass.dispatchWorkgroups(Math.ceil(size.count / 256));
        pass.setPipeline(kernels.gelu); pass.setBindGroup(0, geluBind); pass.dispatchWorkgroups(Math.ceil(size.count / 256));
      }
    };
    const copy = (encoder) => [raw, baseline, candidate, candidateSeed].forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, read, i * size.outputBytes, size.outputBytes));
    const check = async () => {
      await read.mapAsync(GPUMapMode.READ); mapped = true;
      const actual = new Float32Array(read.getMappedRange().slice(0)); read.unmap(); mapped = false;
      const stride = size.outputBytes / 4;
      for (let i = 0; i < 4; i++) exact(`output${i} tail`, TAIL, actual.subarray(i * stride + size.count, (i + 1) * stride));
      const rawValues = actual.subarray(0, size.count);
      if (!rawValues.every(Number.isFinite)) throw new Error("original reduction produced a nonfinite active value");
      if (c.mode === 1) exact("candidate residual seed is unchanged", input.seed.subarray(0, size.count), actual.subarray(3 * stride, 3 * stride + size.count));
      else if (!actual.subarray(3 * stride, 3 * stride + size.count).every(Number.isNaN)) throw new Error("disabled candidate residual seed was written");
      return { parity: exact("registered reduce+Gemma GELU versus fused", actual.subarray(stride, stride + size.count), actual.subarray(2 * stride, 2 * stride + size.count)), finiteRawValues: true, outputTailGuard: true, disabledInputsPoisoned: c.bias === 0 && c.mode !== 1 };
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); copy(encoder); device.queue.submit([encoder.finish()]);
    const correctness = await check(), timings = { baseline: [], candidate: [] }; let repeatedCorrectness = null;
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: 4 });
      const resolved = make(32, U.QUERY_RESOLVE | U.COPY_SRC, null, "PLE timestamps");
      timingRead = make(32, U.MAP_READ | U.COPY_DST, null, "PLE timestamp readback");
      const repetitions = 4;
      for (let pair = 0; pair < 8; pair++) {
        const encoder = device.createCommandEncoder(), order = pair % 2 ? [true, false] : [false, true];
        for (let i = 0; i < order.length; i++) {
          const next = order[i];
          const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
          for (let repeat = 0; repeat < repetitions; repeat++) dispatch(pass, next);
          pass.end();
        }
        encoder.resolveQuerySet(queries, 0, 4, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, timingRead, 0, 32); device.queue.submit([encoder.finish()]);
        await timingRead.mapAsync(GPUMapMode.READ); timingMapped = true;
        const ticks = new BigUint64Array(timingRead.getMappedRange());
        const values = order.map((next, i) => ({ name: next ? "candidate" : "baseline", ms: ticks[2 * i + 1] > ticks[2 * i] ? Number(ticks[2 * i + 1] - ticks[2 * i]) / 1e6 / repetitions : null }));
        timingRead.unmap(); timingMapped = false;
        if (values.some(({ ms }) => !Number.isFinite(ms) || ms <= 0)) throw new Error("GPU timestamps are missing or nonpositive");
        if (pair >= 2) for (const { name, ms } of values) {
          timings[name].push(ms);
        }
      }
      const encoder = device.createCommandEncoder(); copy(encoder); device.queue.submit([encoder.finish()]); repeatedCorrectness = await check();
    }
    return { ...c, source: kernels.metadata, correctness, repeatedCorrectness, timings,
      baselineGpuP50Ms: timings.baseline.length ? median(timings.baseline) : null, candidateGpuP50Ms: timings.candidate.length ? median(timings.candidate) : null,
      ownedGpuBufferBytes: allocatedBytes, resources: size, ok: true };
  } finally { if (mapped) read?.unmap(); if (timingMapped) timingRead?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), state = window.gpuBench = { status: "running", done: false, backend: "webgpu", kernel: "dev.ple_reduce_gelu", cases: [] };
  let gpu;
  try {
    const f16Feature = search.get("f16") === "1";
    gpu = await requestDevice({ features: ["timestamp-query", ...(f16Feature ? ["shader-f16"] : [])] });
    const info = gpu.adapter.info || {}, uncaptured = [];
    gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
    state.adapter = { name: gpu.name, isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features] };
    if (f16Feature && !gpu.device.features.has("shader-f16")) throw new Error("requested shader-f16 feature is unavailable");
    const source = await kernelSource(), geluSource = source("gemma4_gelu"), gelu = await withGpuErrors(gpu, () => pipeline(gpu.device, geluSource, "ple_gelu_original"), "compiling original PLE GELU");
    const cache = new Map(), geluSha256 = await hash(geluSource);
    for (const c of probeCases(search.get("cases") === "timing")) {
      const kernel = `reduce_gemma_${c.policy}`, spec = { weights: "q8", f16: f16Feature, rows: c.rows, groups: 1, splitTarget: c.splitTarget }, key = JSON.stringify({ kernel, spec });
      if (!cache.has(key)) {
        const original = source(kernel, spec), fused = fusedSource(original, geluSource);
        const registeredKernel = kernel.replace(/^reduce/, "reduce_ple");
        if (search.get("source") !== "derived") {
          const registered = source(registeredKernel, spec);
          if (registered !== fused.code) throw new Error("registered PLE reducer differs from the exact dev-derived candidate");
          fused.code = registered;
        }
        const metadata = { kernel, spec, baselineReduceSha256: await hash(original), baselineGeluSha256: geluSha256, fusedSha256: await hash(fused.code), fusedBytes: new TextEncoder().encode(fused.code).byteLength,
          registeredKernel: search.get("source") === "derived" ? null : registeredKernel,
          workgroupSize: fused.workgroupSize, sharedBytes: fused.sharedBytes, reducerAppliesTo: ["generic", "wide", "k64", "bn32"], unchangedGeluSha256: await hash(fused.unchangedGelu) };
        cache.set(key, await withGpuErrors(gpu, async () => ({ gelu, metadata, reduce: await pipeline(gpu.device, original, "ple_reduce_original"), fused: await pipeline(gpu.device, fused.code, "ple_reduce_gelu_probe") }), "compiling exact PLE reducer/GELU candidate"));
      }
      const result = await withGpuErrors(gpu, () => checkCase(gpu, cache.get(key), c), "checking exact PLE reducer/GELU candidate"); state.cases.push(result); log(result);
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU errors: ${uncaptured.join("; ")}`);
    const measured = state.cases.map((c) => c.candidateGpuP50Ms).filter(Number.isFinite), resources = state.cases.map((c) => c.resources);
    Object.assign(state, { status: "done", done: true, metricMs: measured.length ? median(measured) : null, elapsedMs: performance.now() - started,
      arithmetic: "Unchanged registered scalar split reduction, shared F32 materialization and barrier, unchanged registered GELU saturation/expression, then PLE multiplication. No GEMM, weights or precision policy changes.",
      method: "One compute pass per path. Alternating paired GPU timestamps, two warm pairs then six measured pairs of four repetitions. Intervals include only reduce+GELU or the candidate; no model, copies or CPU time. Full-bit guards run before and after timing.",
      integrationScope: "Only the final reducer of an existing split projection. Synthetic split1 coverage does not authorize replacing an unsplit matmul, which has a different epilogue/storage contract. Tail matvec is outside this probe.",
      resourceBounds: { budgetBytes: BUDGET, maxEstimatedCaseBytes: Math.max(...resources.map((r) => r.estimatedBytes)), maxGpuBufferBytes: Math.max(...state.cases.map((c) => c.ownedGpuBufferBytes)), sharedBytesPerWorkgroup: 1024, sequentialCases: true, modelWeightsBytes: 0 },
      compile: { ok: true, wasmUrl: source.wasmUrl, variants: [...cache.values()].map((p) => p.metadata) },
      correctness: { ok: true, bitExact: true, absoluteTolerance: 0, relativeTolerance: 0, fullOutputs: true, signedZeroChecked: true, finiteActiveOutputs: true, poisonedInputsAndOutputTails: true, cases: state.cases.length } });
  } catch (error) { Object.assign(state, { status: "error", done: true, error: String(error?.stack || error), elapsedMs: performance.now() - started }); log(state.error); }
  finally { gpu?.device.destroy(); }
}

if (typeof window !== "undefined") main();
