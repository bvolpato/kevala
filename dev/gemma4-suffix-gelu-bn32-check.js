import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024, CHUNK = MiB, LIMIT = 64 * MiB, SENTINEL = 12345.25;
const SPLIT_TOKENS = 1;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export function validateRows(T) {
  if (!Number.isSafeInteger(T) || T < 1 || T > 256) throw new RangeError("suffix fusion requires 1..256 active rows");
}

export function baselinePolicy(T, f16 = false) {
  validateRows(T);
  const row56 = !f16 && Math.ceil(T / 56) === Math.ceil(T / 64);
  return { rows: 4, row56, BM: row56 ? 56 : 64, BN: 64, splitT: 1,
    paddedRows: Math.max(Math.ceil(T / (row56 ? 56 : 64)) * (row56 ? 56 : 64), Math.ceil(T / 32) * 32) };
}

export function pairedF16Bn32(original) {
  let code = original;
  const replace = (from, to, count = 1) => {
    if (code.split(from).length - 1 !== count) throw new Error(`paired F16 BN32 marker mismatch: ${from}`);
    code = code.replaceAll(from, to);
  };
  if (!code.includes("enable f16;") || (code.match(/var acc\d = vec4<f32>\(0\.0\);/g) || []).length !== 4) throw new Error("paired F16 BN32 requires Q8 F16 R2");
  replace("@workgroup_size(32, 8)", "@workgroup_size(16, 8)");
  for (const tile of ["xs", "gs", "us"]) replace(`var<workgroup> ${tile}: array<vec4<f16>, 512>;`, `var<workgroup> ${tile}: array<vec4<f16>, 256>;`);
  replace("let n0 = wg.x * 64u;", "let n0 = wg.x * 32u;");
  replace("gs[(c + 32u) * 8u", "gs[(c + 16u) * 8u");
  replace("us[(c + 32u) * 8u", "us[(c + 16u) * 8u");
  replace("n0 + lid.x + 32u", "n0 + lid.x + 16u", 4);
  const body = (text, name) => text.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0];
  for (const name of ["sx", "gelu", "store_pair", "mm_splits"]) if (!body(original, name) || body(code, name) !== body(original, name)) throw new Error(`paired F16 BN32 changed ${name}`);
  const dots = (text) => [...text.matchAll(/\*acc\d \+= vec4<f32>\([^;]+;/g)].map(([statement]) => statement);
  if (JSON.stringify(dots(code)) !== JSON.stringify(dots(original))) throw new Error("paired F16 BN32 changed ordered gate/up dots");
  return code;
}

export function memoryPlan(T, N, K, f16 = false) {
  validateRows(T);
  const baseline = baselinePolicy(T, f16);
  const splits = mmSplits(SPLIT_TOKENS, N, K, 256, 64, 64), candidateSplits = mmSplits(SPLIT_TOKENS, N, K, 256, 32, 64);
  if (splits !== candidateSplits) throw new Error("suffix row tile changed the original split count");
  const length = T * N, capacity = baseline.paddedRows * N;
  const partialLength = splits > 1 ? splits * length : 0;
  const yBytes = (capacity + 4) * 4, partBytes = (partialLength + 4) * 4, pairPartBytes = (2 * partialLength + 4) * 4;
  const weights = 2 * N * K * 9 / 8, xBytes = baseline.paddedRows * K * 4;
  const gpuTemporaryBytes = xBytes + (f16 ? 2 : 4) * yBytes + (f16 ? 2 * pairPartBytes : 2 * partBytes + pairPartBytes) + 2 * CHUNK + 80 + 256;
  return { baseline, splits, candidateSplits, length, capacity, partialLength, yBytes, partBytes, pairPartBytes, xBytes, immutableWeightBytes: weights,
    gpuTemporaryBytes, gpuBytes: gpuTemporaryBytes + weights, hostStageBytes: CHUNK, mappedReadBytes: 2 * CHUNK, uploadAllowanceBytes: CHUNK,
    controlledTemporaryBytes: gpuTemporaryBytes + 4 * CHUNK + 128, limitBytes: LIMIT };
}

async function compile(device, source, T, f16) {
  const config = baselinePolicy(T, f16), spec = { weights: "q8", f16, rows: 4, row56: config.row56, groups: 1, splitTarget: 256 };
  const baseline = source(f16 ? "matmul_gelu" : "matmul_gemma_batch", spec), baselineReduce = source(f16 ? "reduce_gelu" : "reduce_gemma_batch", spec);
  const pairSpec = { ...spec, rows: 2, row56: false };
  const candidate = f16 ? pairedF16Bn32(source("matmul_gelu", pairSpec)) : source("matmul_gelu_bn32", pairSpec), candidateReduce = source("reduce_gelu", pairSpec);
  const geluCode = source("gemma4_gelu");
  const gelu = (code) => code.match(/fn gelu\([\s\S]*?\n\}/)?.[0];
  if (!gelu(geluCode) || gelu(candidate) !== gelu(geluCode) || gelu(candidateReduce) !== gelu(geluCode)) throw new Error("registered GELU arithmetic differs");
  const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const customLayout = (label, types) => device.createBindGroupLayout({ label, entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pairLayout = customLayout("suffix paired BN32", ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"]);
  const pairReduceLayout = customLayout("suffix paired BN32 reduce", ["uniform", "uniform", "read-only-storage", "storage"]);
  const pl = (group) => device.createPipelineLayout({ bindGroupLayouts: [group] });
  return { f16, layout, reductionLayout, pairLayout, pairReduceLayout,
    baseline: await pipeline(device, baseline, f16 ? "suffix-gelu.f16-r4-paired" : "suffix-gelu.r4-batch", pl(f16 ? pairLayout : layout)),
    baselineReduce: await pipeline(device, baselineReduce, "suffix-gelu.baseline-reduce", pl(f16 ? pairReduceLayout : reductionLayout)),
    gelu: f16 ? null : await pipeline(device, geluCode, "suffix-gelu.registered-gelu"),
    candidate: await pipeline(device, candidate, "suffix-gelu.paired-bn32", pl(pairLayout)),
    candidateReduce: await pipeline(device, candidateReduce, "suffix-gelu.paired-reduce", pl(pairReduceLayout)) };
}

async function check(runtime, pipes, input) {
  const { T, N, K, pattern = "normal" } = input, { device } = runtime, { f16 } = pipes, U = GPUBufferUsage, plan = memoryPlan(T, N, K, f16);
  if (plan.controlledTemporaryBytes > LIMIT) return { ...input, ok: true, skipped: true, reason: "controlled-temporary-memory-limit", memory: plan };
  if (Math.max(N * K, plan.yBytes, plan.pairPartBytes, plan.xBytes, 2 * CHUNK) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("suffix GELU probe exceeds device buffer limits");
  const owned = [], deadline = performance.now() + 30_000;
  let gpuBytes = 0, queries;
  const make = (size, usage, label) => {
    gpuBytes += size;
    if (gpuBytes > plan.gpuBytes) throw new Error("suffix GELU probe exceeded planned GPU allocation");
    const buffer = device.createBuffer({ size, usage, label: `suffix-gelu.${label}` }); owned.push(buffer); return buffer;
  };
  const map = async (read, size) => { if (!await mapBeforeDeadline(read, size, deadline)) { read.unmap(); throw new Error("suffix GELU readback deadline exceeded"); } };
  const drain = async () => {
    const remaining = deadline - performance.now(); if (remaining <= 0) throw new Error("suffix GELU upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("suffix GELU upload deadline exceeded")), remaining); })]); }
    finally { clearTimeout(timer); }
  };
  try {
    const stage = new Float32Array(CHUNK / 4), bytes = new Int8Array(stage.buffer);
    const generated = async (buffer, size, width, valueAt) => {
      for (let offset = 0; offset < size; offset += CHUNK) {
        const count = Math.min(CHUNK, size - offset), target = width === 1 ? bytes : stage;
        for (let i = 0; i < count / width; i++) target[i] = valueAt(offset / width + i);
        device.queue.writeBuffer(buffer, offset, stage.buffer, 0, count); await drain();
      }
    };
    const weightsFor = async (up) => {
      const w = make(N * K, U.STORAGE | U.COPY_DST, up ? "up-W" : "gate-W"), s = make(N * K / 8, U.STORAGE | U.COPY_DST, "scales");
      await generated(w, N * K, 1, (i) => {
        const col = Math.floor(i / K), k = i % K;
        if (pattern === "saturated" && col < 2) return up ? 17 : col ? -128 : 127;
        if (pattern === "cancellation") return (k % 2 ? -1 : 1) * (1 + (col * 13 + (k & ~1) * 7) % 127);
        return (i * 31 + (i >>> 5) * 17 + (up ? 37 : 19)) % 256 - 128;
      });
      await generated(s, N * K / 8, 4, (i) => pattern === "saturated" && i < 2 * K / 32 ? 0.05 : i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
      return { w, s };
    };
    const gateW = await weightsFor(false), upW = await weightsFor(true), x = make(plan.xBytes, U.STORAGE | U.COPY_DST, "X");
    await generated(x, plan.xBytes, 4, (i) => {
      const row = Math.floor(i / K), k = i % K;
      if (row >= T) return NaN;
      if (pattern === "zeros") return i % 2 ? -0 : 0;
      if (pattern === "saturated") return 0.32 * (1 + row / 32);
      return 0.04 * Math.sin((pattern === "cancellation" ? k & ~1 : k) * 0.127 + row * 0.13 + 0.3);
    });
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const pairParams = make(16, U.UNIFORM | U.COPY_DST, "pair-params"), geluParams = make(16, U.UNIFORM | U.COPY_DST, "gelu-params");
    const zero = make(16, U.STORAGE, "disabled-bias");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, SPLIT_TOKENS, T, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(pairParams, 0, new Uint32Array([N, K, SPLIT_TOKENS, 0]));
    device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
    const outputs = Array.from({ length: f16 ? 2 : 4 }, () => make(plan.yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y"));
    const parts = f16 ? [] : [0, 1].map(() => make(plan.partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "PART"));
    const baselinePart = f16 ? make(plan.pairPartBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-paired-PART") : null;
    const pairPart = make(plan.pairPartBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "paired-PART");
    const baselineOutput = outputs[f16 ? 0 : 2], candidateOutput = outputs.at(-1);
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-read"), resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const baseline = f16 ? [] : [gateW, upW].map((weight, i) => ({
      group: group(pipes.layout, [globals, params, x, weight.w, weight.s, zero, outputs[i], parts[i]]),
      reduce: group(pipes.reductionLayout, [globals, params, parts[i], zero, outputs[i]]) }));
    const geluGroup = f16 ? null : group(pipes.gelu.getBindGroupLayout(0), [globals, geluParams, outputs[0], outputs[1], baselineOutput]);
    const baselineGroup = f16 ? group(pipes.pairLayout, [globals, pairParams, x, gateW.w, gateW.s, upW.w, upW.s, baselineOutput, baselinePart]) : null;
    const baselineReduce = f16 ? group(pipes.pairReduceLayout, [globals, pairParams, baselinePart, baselineOutput]) : null;
    const pairGroup = group(pipes.pairLayout, [globals, pairParams, x, gateW.w, gateW.s, upW.w, upW.s, candidateOutput, pairPart]);
    const pairReduce = group(pipes.pairReduceLayout, [globals, pairParams, pairPart, candidateOutput]);
    const work = Math.ceil(plan.length / 256);
    const dispatch = (pass, fused) => {
      if (fused) {
        pass.setPipeline(pipes.candidate); pass.setBindGroup(0, pairGroup); pass.dispatchWorkgroups(Math.ceil(N / 32), Math.ceil(T / 32), plan.splits);
        if (plan.splits > 1) { pass.setPipeline(pipes.candidateReduce); pass.setBindGroup(0, pairReduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535)); }
      } else if (f16) {
        pass.setPipeline(pipes.baseline); pass.setBindGroup(0, baselineGroup); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 64), plan.splits);
        if (plan.splits > 1) { pass.setPipeline(pipes.baselineReduce); pass.setBindGroup(0, baselineReduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535)); }
      } else {
        for (const op of baseline) {
          pass.setPipeline(pipes.baseline); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / plan.baseline.BM), plan.splits);
          if (plan.splits > 1) { pass.setPipeline(pipes.baselineReduce); pass.setBindGroup(0, op.reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535)); }
        }
        pass.setPipeline(pipes.gelu); pass.setBindGroup(0, geluGroup); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const compare = async (left, right, offsetRight, length, offsetLeft = 0) => {
      let zeros = 0, negativeZeros = 0;
      for (let offset = 0; offset < length; offset += CHUNK / 4) {
        const count = Math.min(CHUNK / 4, length - offset), size = count * 4, encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(left, offsetLeft + offset * 4, read, 0, size); encoder.copyBufferToBuffer(right, offsetRight + offset * 4, read, size, size); device.queue.submit([encoder.finish()]);
        await map(read, size * 2);
        try {
          const range = read.getMappedRange(0, size * 2), values = new Float32Array(range), bits = new Uint32Array(range);
          for (let i = 0; i < count; i++) {
            if (!Number.isFinite(values[i]) || !Number.isFinite(values[count + i])) throw new Error(`suffix GELU nonfinite at ${offset + i}`);
            if (bits[i] !== bits[count + i]) throw new Error(`suffix GELU bit mismatch at ${offset + i}: 0x${bits[i].toString(16)} vs 0x${bits[count + i].toString(16)}`);
            if (values[i] === 0) zeros++; if (bits[i] === 0x80000000) negativeZeros++;
          }
        } finally { read.unmap(); }
      }
      return { samples: length, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros, signedZeroChecked: true };
    };
    const tail = async (buffer, active, capacity) => {
      for (let offset = active; offset < capacity; offset += CHUNK / 4) {
        const size = Math.min(CHUNK / 4, capacity - offset) * 4, encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, offset * 4, read, 0, size); device.queue.submit([encoder.finish()]); await map(read, size);
        try { if (new Float32Array(read.getMappedRange(0, size)).some((value) => value !== SENTINEL)) throw new Error(`suffix GELU padding/tail overwrite after ${active}`); }
        finally { read.unmap(); }
      }
    };
    const verify = async () => {
      for (const buffer of outputs) await generated(buffer, plan.yBytes, 4, (i) => i < plan.length ? NaN : SENTINEL);
      for (const buffer of parts) await generated(buffer, plan.partBytes, 4, (i) => i < plan.partialLength ? NaN : SENTINEL);
      if (baselinePart) await generated(baselinePart, plan.pairPartBytes, 4, (i) => i < 2 * plan.partialLength ? NaN : SENTINEL);
      await generated(pairPart, plan.pairPartBytes, 4, (i) => i < 2 * plan.partialLength ? NaN : SENTINEL);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); device.queue.submit([encoder.finish()]);
      const output = await compare(baselineOutput, candidateOutput, 0, plan.length);
      const gatePartials = await compare(f16 ? baselinePart : parts[0], pairPart, 0, plan.partialLength);
      const upPartials = await compare(f16 ? baselinePart : parts[1], pairPart, plan.partialLength * 4, plan.partialLength, f16 ? plan.partialLength * 4 : 0);
      for (const buffer of outputs) await tail(buffer, plan.length, plan.capacity + 4);
      for (const buffer of parts) await tail(buffer, plan.partialLength, plan.partialLength + 4);
      if (baselinePart) await tail(baselinePart, 2 * plan.partialLength, 2 * plan.partialLength + 4);
      await tail(pairPart, 2 * plan.partialLength, 2 * plan.partialLength + 4);
      if (pattern === "saturated") {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(baselineOutput, 0, read, 0, 8); device.queue.submit([encoder.finish()]); await map(read, 8);
        try { const values = new Float32Array(read.getMappedRange(0, 8)); if (values[0] <= 10 || values[1] !== 0) throw new Error("GELU saturation branches were not exercised"); }
        finally { read.unmap(); }
      }
      return { output, gatePartials, upPartials, paddingAndTailGuard: true, inputPadding: "NaN outside active rows" };
    };
    const numericalCheck = await verify();
    let timing = null;
    if (device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: 12 });
      const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(), order = [];
      dispatch(warm, false); dispatch(warm, true); warm.end();
      for (let pair = 0; pair < 3; pair++) for (const fused of pair % 2 ? [true, false] : [false, true]) {
        const at = order.length * 2; order.push(fused);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, fused); pass.end();
      }
      encoder.resolveQuerySet(queries, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]); await map(read, 96);
      try {
        const stamps = new BigUint64Array(read.getMappedRange(0, 96)), samples = { baseline: [], candidate: [] };
        order.forEach((fused, i) => samples[fused ? "candidate" : "baseline"].push(Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6 / 4));
        const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
        if (![baselineMs, candidateMs].every((value) => Number.isFinite(value) && value > 0)) throw new Error("invalid suffix GELU timestamps");
        timing = { samples: 3, repetitions: 4, ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs };
      } finally { read.unmap(); }
    }
    const afterTimingCheck = await verify();
    return { ...input, f16, candidateRows: 2, baselineRows: 4, baselineBM: plan.baseline.BM, baselineRow56: plan.baseline.row56, baselinePaired: f16,
      physicalBN: 32, virtualBN: 64, globals: [T, 1, T, 0], pairSplitT: 1, splits: plan.splits,
      ok: true, numericalCheck, afterTimingCheck, timing, baselineDispatches: f16 ? plan.splits > 1 ? 2 : 1 : plan.splits > 1 ? 5 : 3, candidateDispatches: plan.splits > 1 ? 2 : 1,
      sharedBytes: f16 ? 6144 : 12288, baselineSharedBytes: f16 ? 12288 : 16384, memory: { ...plan, actualGpuBytes: gpuBytes, readbackChunkBytesPerSide: CHUNK } };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search), f16 = params.get("f16") === "1";
  let inputs = [{ shape: "gate6144", T: 97, N: 6144, K: 1536 }, { shape: "ragged", T: 97, N: 65, K: 1056 },
    { shape: "cancellation", T: 97, N: 65, K: 1056, pattern: "cancellation" }, { shape: "zeros", T: 97, N: 65, K: 1056, pattern: "zeros" },
    { shape: "saturated", T: 97, N: 65, K: 32, pattern: "saturated" }];
  if (params.get("boundaries") === "1") for (const T of [65, 96, 112, 128, 241]) {
    inputs.push({ shape: "gate6144", T, N: 6144, K: 1536 }, { shape: "ragged", T, N: 65, K: 1056 });
  }
  if (f16 || params.get("wide") === "1") for (const T of params.get("boundaries") === "1" ? [65, 96, 97, 112, 128, 241] : [97]) inputs.push({ shape: "gate12288", T, N: 12288, K: 1536 });
  inputs = inputs.filter(({ shape, T }) => (!params.has("shape") || shape === params.get("shape")) && (!params.has("T") || T === Number(params.get("T"))));
  if (!inputs.length) throw new Error("no matching suffix fusion case; shape=gate6144/ragged/cancellation/zeros/saturated/gate12288; optional boundaries=1");
  metadata = { f16, requiredFeatures: f16 ? ["shader-f16"] : [] };
  const runtime = await requestDevice({ features: f16 ? ["shader-f16", "timestamp-query"] : ["timestamp-query"] });
  try {
    metadata.adapter = { name: runtime.name, features: [...runtime.device.features] };
    if (f16 && !runtime.device.features.has("shader-f16")) throw new Error("F16 suffix fusion requires shader-f16; F32 fallback is disabled");
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata = { ...metadata, wasmUrl: source.wasmUrl, controlledTemporaryLimitBytes: LIMIT, immutableWeightsExcludedFromLimit: true,
      baselineKernel: f16 ? "matmul_gelu" : "matmul_gemma_batch", candidateKernel: f16 ? "dev BN32 transform of matmul_gelu" : "matmul_gelu_bn32", baselinePaired: f16, registered: !f16 };
    const pipelines = new Map();
    for (const input of inputs) {
      const key = baselinePolicy(input.T, f16).row56;
      if (!pipelines.has(key)) pipelines.set(key, await withGpuErrors(runtime, () => compile(runtime.device, source, input.T, f16), "building suffix BN32 GELU fusion"));
      log({ status: "checking", ...input, f16, memory: memoryPlan(input.T, input.N, input.K, f16) });
      const result = await withGpuErrors(runtime, () => check(runtime, pipelines.get(key), input), `suffix BN32 GELU ${input.shape} T${input.T}`);
      cases.push(result); window.gpuBench.completed = cases.length; log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const executed = cases.filter((value) => !value.skipped), timed = executed.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-suffix-gelu-bn32", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMs), 0) / timed.length) : performance.now() - started,
      method: timed.length ? `geometric mean candidate GPU timestamp milliseconds; ${f16 ? "current paired F16 R4/BN64" : "current R4 batch"} baseline` : "untimed guard wall milliseconds", elapsedMs: performance.now() - started,
      skipped: executed.length === 0, correctness: { ok: true, bitExact: executed.length > 0, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-suffix-gelu-bn32", completed: 0 };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
    correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
