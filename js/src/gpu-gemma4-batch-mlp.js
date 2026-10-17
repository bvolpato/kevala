// Independent suffixes retain their original splitT=1 partition while sharing
// gate/up input tiles. Selection is calibrated for the exact physical row count.
import { bufferUsage, encodeMatmul, inErrorScopes, mmSplits, pipeline, scratchBuffer } from "./gpu.js";
import { compareGemma4MlpOutputs } from "./gpu-gemma4-mlp.js";
import { mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";
import { selectedMatmulKernel } from "./gpu-tuning.js";

const ROWS = 4;
const TARGET = 256;
const SAMPLES = 3;
const REPETITIONS = 4;
const DEADLINE_MS = 750;
const MAX_SCRATCH = 64 * 1024 * 1024;
const SENTINEL = 12345.25;
const SENTINEL_BITS = new Uint32Array(new Float32Array([SENTINEL]).buffer)[0];
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

export function gemma4BatchMlpEligible(tokens) {
  return Number.isSafeInteger(tokens) && tokens >= 97 && tokens <= 256;
}

export function gemma4BatchMlpTallEligible(tokens) {
  return Number.isSafeInteger(tokens) && tokens >= 97 && tokens <= 104;
}

export function gemma4BatchMlpPartBytes(tokens, N, K) {
  const splits = mmSplits(1, N, K, TARGET, 64, 64);
  return splits > 1 ? 2 * splits * tokens * N * 4 : 16;
}

async function batchMlpPipelines(device, source) {
  const layoutFor = (label, types) => device.createBindGroupLayout({ label, entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const layout = layoutFor("Gemma paired batch MLP", ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"]);
  const reduceLayout = layoutFor("Gemma paired batch MLP reduce", ["uniform", "uniform", "read-only-storage", "storage"]);
  const spec = { weights: "q8", f16: true, rows: ROWS, groups: 1, splitTarget: TARGET };
  const [matmul, reduce] = await Promise.all([
    pipeline(device, source("matmul_gelu", spec), "gemma4_batch_mlp_r4", device.createPipelineLayout({ bindGroupLayouts: [layout] })),
    pipeline(device, source("reduce_gelu", spec), "gemma4_batch_mlp_reduce_r4", device.createPipelineLayout({ bindGroupLayouts: [reduceLayout] })),
  ]);
  return { matmul, reduce, layout, reduceLayout, ...spec };
}

export function gemma4BatchMlpBindGroups(device, pipes, { globals, params, input, gate, gateScales, up, upScales, output, partials }) {
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  return {
    group: group(pipes.layout, [globals, params, input, gate, gateScales, up, upScales, output, partials]),
    reduce: group(pipes.reduceLayout, [globals, params, partials, output]),
  };
}

export function encodeGemma4BatchMlp(pass, pipes, op, tokens) {
  if (!gemma4BatchMlpEligible(tokens) || op.N < 6144 || op.batched !== true || op.sourcePolicy === true) throw new Error("Gemma paired batch MLP requires independent R4 suffix rows");
  const splits = mmSplits(1, op.N, op.K, TARGET, 64, 64);
  const tall = gemma4BatchMlpTallEligible(tokens) && pipes.tall?.enabled === true && pipes.tall.tokens === tokens && pipes.tall.selection.has(`${op.N}:${op.K}`) && pipes.tall.pipeline;
  pass.setPipeline(tall || pipes.matmul); pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / 64), Math.ceil(tokens / (tall ? 104 : 64)), splits);
  if (splits > 1) {
    const work = Math.ceil(tokens * op.N / 256);
    pass.setPipeline(pipes.reduce); pass.setBindGroup(0, op.reduce);
    pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
  }
}

function pairedShapes(weights, eligibleShapes) {
  const tensors = weights?.tensors;
  if (!(tensors instanceof Map)) return [];
  const shapes = [], seen = new Set();
  for (const [name, gate] of tensors) {
    if (!/^l\.\d+\.gate$/.test(name) || gate.info?.dtype !== "q8" || !Array.isArray(gate.info.shape) || gate.info.shape.length !== 2) continue;
    const [N, K] = gate.info.shape.map(Number), up = tensors.get(name.replace(/\.gate$/, ".up"));
    const key = `${N}:${K}`;
    if (eligibleShapes && !eligibleShapes.has(key)) continue;
    if (seen.has(key) || !Number.isSafeInteger(N) || !Number.isSafeInteger(K) || N < 6144 || K <= 0 || K % 32 !== 0) continue;
    if (up?.info?.dtype !== "q8" || !Array.isArray(up.info.shape) || up.info.shape.length !== 2 || Number(up.info.shape[0]) !== N || Number(up.info.shape[1]) !== K) continue;
    if (!gate.buf || !gate.sbuf || !up.buf || !up.sbuf) continue;
    seen.add(key); shapes.push({ N, K, key, name, gate, up });
  }
  return shapes.sort((a, b) => a.N * a.K - b.N * b.K).slice(0, 2);
}

function inputFor(tokens, shape) {
  let state = (0x424d4c50 ^ tokens ^ shape.N ^ Math.imul(shape.K, 31)) >>> 0;
  const values = new Float32Array(tokens * shape.K);
  for (let index = 0; index < values.length; index++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    values[index] = Math.fround((((value ^ (value >>> 14)) >>> 0) / 0x100000000 - 0.5) * 0.08) || 0.00390625;
  }
  return values;
}

function deadlineError() {
  return Object.assign(new Error("Gemma batch MLP calibration deadline exceeded"), { code: "GEMMA4_BATCH_MLP_DEADLINE" });
}

export async function createGemma4BatchMlpValidator(device) {
  const params = "struct P { len: u32, active_len: u32, sentinel: u32, poison: u32 }\n@group(0) @binding(0) var<uniform> p: P;\n";
  const fillCode = params + `@group(0) @binding(1) var<storage, read_write> values: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= p.len) { return; }
  values[i] = select(p.sentinel, p.poison, i < p.active_len);
}`;
  const compareCode = params + `@group(0) @binding(1) var<storage, read> baseline: array<u32>;
@group(0) @binding(2) var<storage, read> candidate: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= p.len) { return; }
  let a = baseline[i];
  let b = candidate[i];
  if (a != b) { atomicAdd(&status[0], 1u); }
  if (i < p.active_len) {
    if ((a & 0x7f800000u) == 0x7f800000u) { atomicAdd(&status[1], 1u); }
    if ((b & 0x7f800000u) == 0x7f800000u) { atomicAdd(&status[1], 1u); }
  } else {
    if (a != p.sentinel) { atomicAdd(&status[2], 1u); }
    if (b != p.sentinel) { atomicAdd(&status[2], 1u); }
  }
}`;
  const layout = (label, types) => device.createBindGroupLayout({ label, entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const fillLayout = layout("Gemma batch MLP validation fill", ["uniform", "storage"]);
  const compareLayout = layout("Gemma batch MLP validation compare", ["uniform", "read-only-storage", "read-only-storage", "storage"]);
  const results = await Promise.allSettled([
    pipeline(device, fillCode, "gemma4_batch_mlp_validation_fill", device.createPipelineLayout({ bindGroupLayouts: [fillLayout] })),
    pipeline(device, compareCode, "gemma4_batch_mlp_validation_compare", device.createPipelineLayout({ bindGroupLayouts: [compareLayout] })),
  ]);
  const rejected = results.find((entry) => entry.status === "rejected");
  if (rejected) throw rejected.reason;
  return { fill: results[0].value, compare: results[1].value, fillLayout, compareLayout };
}

export function gemma4BatchMlpValidationGroups(device, validator, { params, baseline, candidate, status }) {
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  return { fill: [baseline, candidate].map((buffer) => group(validator.fillLayout, [params, buffer])), compare: group(validator.compareLayout, [params, baseline, candidate, status]) };
}

async function measureShape(device, shape, tuning, batchPipelines, candidate, gelu, tokens, deadline, { baselinePaired = null } = {}) {
  const selected = selectedMatmulKernel(tuning.selection, shape.N, shape.K, 1);
  const baselineKernel = baselinePaired ? "paired-batch-mlp" : batchPipelines[selected] ? selected : "generic";
  const baseline = baselinePaired || batchPipelines[baselineKernel];
  const { N, K } = shape;
  const base = { key: shape.key, name: shape.name, N, K, tokens, splitTokens: 1, rows: ROWS, baselineKernel, selected: false };
  if (baselinePaired ? !baseline.matmul || !baseline.reduce : !baseline?.mm?.[ROWS] || !baseline.reduce?.[ROWS]) return { ...base, reason: "missing-baseline-pipeline" };
  if (baseline.f16 !== true || baseline.groups !== 1 || baseline.splitTarget !== TARGET) return { ...base, reason: "unsupported-baseline" };
  const yBytes = tokens * N * 4;
  const paddedRows = baselinePaired ? Math.max(Math.ceil(tokens / 64) * 64, Math.ceil(tokens / 104) * 104) : tokens;
  const partBytes = gemma4BatchMlpPartBytes(tokens, N, K);
  const bytes = { x: paddedRows * K * 4, y: paddedRows * N * 4 + (baselinePaired ? 16 : 0), read: baselinePaired ? 256 : Math.max(256, yBytes), part: partBytes + (baselinePaired ? 16 : 0), resolved: 256 };
  // Gate, up and one shared activation output plus one sequential readback.
  const gpuScratchBytes = bytes.x + bytes.y * (baselinePaired ? 2 : 3) + bytes.read + bytes.part * (baselinePaired ? 2 : 1) + bytes.resolved + 80 + (baselinePaired ? 64 : 0);
  const scratchBytes = gpuScratchBytes + (baselinePaired ? 2 * tokens * K * 4 + bytes.read + 512 : 0);
  const maxBuffer = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  if (!Number.isSafeInteger(scratchBytes) || scratchBytes > MAX_SCRATCH || Math.max(...Object.values(bytes)) > maxBuffer) return { ...base, reason: "scratch-limit", scratchBytes };
  const owned = [];
  let queries, read, mapped = false;
  try {
    const U = bufferUsage();
    const make = (size, usage, label) => (owned.push(scratchBuffer(device, size, usage, `${shape.name}.batch-mlp-tune.${label}`)), owned.at(-1));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const fusedParams = make(16, U.UNIFORM | U.COPY_DST, "fused-params");
    const geluParams = make(16, U.UNIFORM | U.COPY_DST, "gelu-params");
    const x = make(bytes.x, U.STORAGE | U.COPY_DST, "x");
    const zero = make(16, U.STORAGE, "zero");
    const part = make(bytes.part, U.STORAGE | (baselinePaired ? U.COPY_SRC | U.COPY_DST : 0), "part");
    const gate = baselinePaired ? null : make(yBytes, U.STORAGE, "gate"), up = baselinePaired ? null : make(yBytes, U.STORAGE, "up");
    const output = make(bytes.y, U.STORAGE | U.COPY_SRC | U.COPY_DST, "output");
    const candidateOutput = baselinePaired ? make(bytes.y, U.STORAGE | U.COPY_SRC | U.COPY_DST, "candidate-output") : output;
    const candidatePart = baselinePaired ? make(bytes.part, U.STORAGE | U.COPY_SRC | U.COPY_DST, "candidate-part") : part;
    read = make(bytes.read, U.MAP_READ | U.COPY_DST, "read");
    const resolved = make(bytes.resolved, U.QUERY_RESOLVE | U.COPY_SRC, "resolved");
    device.queue.writeBuffer(globals, 0, new Uint32Array([tokens, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(fusedParams, 0, new Uint32Array([N, K, 1, 0]));
    device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
    if (!baselinePaired) device.queue.writeBuffer(x, 0, inputFor(tokens, shape));
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const opFor = (weights, y) => ({ N, K,
      group: group(baseline.layout, [globals, params, x, weights.buf, weights.sbuf, zero, y, part]),
      reduce: group(baseline.reduceLayout, [globals, params, part, zero, y]),
    });
    const gateOp = baselinePaired ? null : opFor(shape.gate, gate), upOp = baselinePaired ? null : opFor(shape.up, up);
    const geluGroup = baselinePaired ? null : group(gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, output]);
    const candidateOp = { N, K, batched: true, ...gemma4BatchMlpBindGroups(device, candidate, { globals, params: fusedParams, input: x, gate: shape.gate.buf, gateScales: shape.gate.sbuf, up: shape.up.buf, upScales: shape.up.sbuf, output: candidateOutput, partials: candidatePart }) };
    const baselineOp = baselinePaired ? { N, K, batched: true, ...gemma4BatchMlpBindGroups(device, baselinePaired, { globals, params: fusedParams, input: x, gate: shape.gate.buf, gateScales: shape.gate.sbuf, up: shape.up.buf, upScales: shape.up.sbuf, output, partials: part }) } : null;
    const dispatch = (pass, fused) => {
      if (fused) return encodeGemma4BatchMlp(pass, candidate, candidateOp, tokens);
      if (baselinePaired) return encodeGemma4BatchMlp(pass, baselinePaired, baselineOp, tokens);
      encodeMatmul(pass, baseline, gateOp, tokens, 1); encodeMatmul(pass, baseline, upOp, tokens, 1);
      pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup);
      const work = Math.ceil(tokens * N / 256);
      pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
    };
    const snapshot = async (fused) => {
      if (now() >= deadline) throw deadlineError();
      device.queue.writeBuffer(output, 0, new Float32Array(tokens * N).fill(NaN));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(); dispatch(pass, fused); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, yBytes);
      device.queue.submit([encoder.finish()]);
      const withinDeadline = await mapBeforeDeadline(read, yBytes, deadline);
      mapped = true;
      if (!withinDeadline) throw deadlineError();
      const values = new Float32Array(read.getMappedRange(0, yBytes).slice(0));
      read.unmap(); mapped = false;
      return values;
    };
    const validator = baselinePaired ? candidate.tall.validator : null;
    const partFloats = mmSplits(1, N, K, TARGET, 64, 64) > 1 ? partBytes / 4 : 0;
    let validationGroups, status;
    const workgroups = (pass, length) => {
      const work = Math.ceil(length / 256);
      pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
    };
    if (baselinePaired) {
      const paramsFor = (length, active, label) => {
        const params = make(16, U.UNIFORM | U.COPY_DST, label);
        device.queue.writeBuffer(params, 0, new Uint32Array([length, active, SENTINEL_BITS, 0x7fc00000]));
        return params;
      };
      status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "validation-status");
      validationGroups = [
        { length: bytes.y / 4, ...gemma4BatchMlpValidationGroups(device, validator, { params: paramsFor(bytes.y / 4, tokens * N, "validation-y-params"), baseline: output, candidate: candidateOutput, status }) },
        { length: bytes.part / 4, ...gemma4BatchMlpValidationGroups(device, validator, { params: paramsFor(bytes.part / 4, partFloats, "validation-part-params"), baseline: part, candidate: candidatePart, status }) },
      ];
      const inputFill = group(validator.fillLayout, [paramsFor(bytes.x / 4, bytes.x / 4, "validation-x-params"), x]);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(validator.fill); pass.setBindGroup(0, inputFill); workgroups(pass, bytes.x / 4); pass.end();
      device.queue.submit([encoder.finish()]);
      device.queue.writeBuffer(x, 0, inputFor(tokens, shape));
    }
    const verifyPaired = async () => {
      if (now() >= deadline) throw deadlineError();
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(validator.fill);
      for (const groups of validationGroups) for (const group of groups.fill) { fill.setBindGroup(0, group); workgroups(fill, groups.length); }
      fill.end();
      const compute = encoder.beginComputePass(); dispatch(compute, false); dispatch(compute, true); compute.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(validator.compare);
      for (const groups of validationGroups) { compare.setBindGroup(0, groups.compare); workgroups(compare, groups.length); }
      compare.end(); encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const withinDeadline = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      if (!withinDeadline) throw deadlineError();
      try {
        const counts = new Uint32Array(read.getMappedRange(0, 16));
        return { ok: counts.every((value) => value === 0), length: tokens * N, partialValues: partFloats, bitMismatches: counts[0], nonFinite: counts[1], tailMismatches: counts[2], paddingAndTailGuard: counts[2] === 0, signedZeroChecked: true, validation: "gpu-bitwise", readbackBytes: 16 };
      } finally { read.unmap(); mapped = false; }
    };
    const numericalCheck = baselinePaired ? await verifyPaired() : compareGemma4MlpOutputs(await snapshot(false), await snapshot(true));
    if (!numericalCheck.ok) return { ...base, numericalCheck, reason: "numerical-guard-failed" };
    const queryCount = SAMPLES * 4;
    queries = device.createQuerySet({ type: "timestamp", count: queryCount });
    const encoder = device.createCommandEncoder();
    const warmup = encoder.beginComputePass(); dispatch(warmup, false); dispatch(warmup, true); warmup.end();
    const indices = { baseline: [], candidate: [] };
    let index = 0;
    for (let sample = 0; sample < SAMPLES; sample++) {
      if (now() >= deadline) throw deadlineError();
      for (const fused of sample % 2 ? [true, false] : [false, true]) {
        indices[fused ? "candidate" : "baseline"].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repetition = 0; repetition < REPETITIONS; repetition++) dispatch(pass, fused);
        pass.end(); index += 2;
      }
    }
    encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8);
    device.queue.submit([encoder.finish()]);
    const withinTimingDeadline = await mapBeforeDeadline(read, queryCount * 8, deadline);
    mapped = true;
    if (!withinTimingDeadline) throw deadlineError();
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0));
    read.unmap(); mapped = false;
    const times = Object.fromEntries(Object.entries(indices).map(([label, values]) => [label, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS)]));
    const afterTimingCheck = baselinePaired ? await verifyPaired() : null;
    if (afterTimingCheck && !afterTimingCheck.ok) return { ...base, numericalCheck, afterTimingCheck, reason: "numerical-guard-failed" };
    const choice = selectMatvecKernel({ ...times, numericalCheck });
    return { ...base, ...choice, ...times, scratchBytes, ...(afterTimingCheck ? { afterTimingCheck } : {}), kernel: choice.selected ? baselinePaired ? "paired-batch-mlp-tall" : "paired-batch-mlp" : baselineKernel, reason: choice.reason === "stable-matvec-win" ? baselinePaired ? "stable-paired-batch-mlp-tall-win" : "stable-paired-batch-mlp-win" : choice.reason };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    queries?.destroy();
  }
}

export async function calibrateGemma4BatchMlp(device, source, weights, tuning, { tokens, batchPipelines, gelu, kernel = "auto", pipelines = null, eligibleShapes = null, tall = true } = {}) {
  if (!["auto", "generic", "wide"].includes(kernel)) throw new Error('kernel must be "auto", "generic", or "wide"');
  const started = now(), selection = new Set(), tallSelection = new Set();
  const cachedTallPipeline = pipelines?.tall?.pipeline;
  const cachedValidator = pipelines?.tall?.validator;
  if (pipelines?.tall) pipelines = { ...pipelines, tall: null };
  const tallDiagnostics = { enabled: false, method: "baseline", tokens, physicalRows: 104, virtualRows: 64, points: [], truncated: false, reason: "not-calibrated" };
  const diagnostics = { revision: 2, method: "baseline", kernel, tokens, splitTokens: 1, rows: ROWS, samples: SAMPLES, repetitions: REPETITIONS, threshold: 0.05, points: [], truncated: false, tall: tallDiagnostics };
  const result = (reason) => ({ tokens, pipelines, selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  const abort = (reason) => {
    selection.clear();
    for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.kernel = point.baselineKernel; point.reason = "aborted-after-calibration-failure"; }
    return result(reason);
  };
  if (kernel !== "auto") return result("kernel-override");
  if (!gemma4BatchMlpEligible(tokens)) return result("unsupported-physical-tokens");
  if (!device.features?.has?.("shader-f16")) return result("no-shader-f16");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  if (!batchPipelines?.generic || !tuning || !gelu) return result("missing-baseline-pipeline");
  const shapes = pairedShapes(weights, eligibleShapes);
  if (!shapes.length && eligibleShapes) return result("no-eligible-shapes");
  if (!shapes.length) return result("no-q8-pairs");
  diagnostics.method = "paired-timestamp-query";
  try { pipelines ||= await inErrorScopes(device, () => batchMlpPipelines(device, source)); }
  catch (error) { diagnostics.error = String(error?.message || error); return result("pipeline-failed"); }
  const deadline = started + DEADLINE_MS;
  for (const shape of shapes) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    try {
      const point = await inErrorScopes(device, () => measureShape(device, shape, tuning, batchPipelines, pipelines, gelu, tokens, deadline));
      diagnostics.points.push(point);
      if (point.selected) selection.add(shape.key);
      if (["numerical-guard-failed", "missing-baseline-pipeline", "unsupported-baseline", "insufficient-samples"].includes(point.reason)) return abort(point.reason === "insufficient-samples" ? "timing-failed" : point.reason);
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (error?.code === "MATVEC_DEADLINE" || error?.code === "GEMMA4_BATCH_MLP_DEADLINE") { diagnostics.truncated = true; return result("deadline"); }
      return abort("calibration-failed");
    }
  }
  const baseReason = selection.size ? "calibrated" : "no-stable-win";
  if (!tall || !gemma4BatchMlpTallEligible(tokens) || !selection.size) {
    tallDiagnostics.reason = !tall ? "disabled" : !gemma4BatchMlpTallEligible(tokens) ? "unsupported-physical-tokens" : "no-paired-selections";
    return result(baseReason);
  }
  if (device.limits?.maxComputeWorkgroupStorageSize < 14848) { tallDiagnostics.reason = "workgroup-storage-limit"; return result(baseReason); }
  if (now() >= deadline) { tallDiagnostics.reason = "deadline"; tallDiagnostics.truncated = true; return result(baseReason); }
  const baselinePaired = pipelines;
  let tallPipeline = cachedTallPipeline;
  let validator = cachedValidator;
  try {
    tallPipeline ||= await inErrorScopes(device, () => pipeline(device, source("matmul_gelu_tall", { weights: "q8", f16: true, rows: ROWS, row56: false, groups: 1, splitTarget: TARGET }), "gemma4_batch_mlp_tall", device.createPipelineLayout({ bindGroupLayouts: [pipelines.layout] })));
    validator ||= await inErrorScopes(device, () => createGemma4BatchMlpValidator(device));
  } catch (error) {
    tallDiagnostics.reason = "pipeline-failed"; tallDiagnostics.error = String(error?.message || error); return result(baseReason);
  }
  const tallState = { enabled: true, tokens, pipeline: tallPipeline, validator, selection: tallSelection };
  pipelines = { ...pipelines, tall: tallState };
  tallDiagnostics.method = "paired-timestamp-query";
  const finishTall = (reason, failed = false) => {
    if (failed) {
      tallSelection.clear();
      for (const point of tallDiagnostics.points) if (point.selected) { point.selected = false; point.kernel = point.baselineKernel; point.reason = "aborted-after-tall-calibration-failure"; }
    }
    tallState.enabled = tallSelection.size > 0;
    tallDiagnostics.enabled = tallState.enabled; tallDiagnostics.reason = reason;
    return result(baseReason);
  };
  for (const shape of shapes.filter((shape) => selection.has(shape.key)).sort((a, b) => b.N - a.N)) {
    if (now() >= deadline) { tallDiagnostics.truncated = true; return finishTall("deadline"); }
    const probe = { ...baselinePaired, tall: { ...tallState, selection: new Set([shape.key]) } };
    try {
      const point = await inErrorScopes(device, () => measureShape(device, shape, tuning, batchPipelines, probe, gelu, tokens, deadline, { baselinePaired }));
      tallDiagnostics.points.push(point);
      if (point.selected) tallSelection.add(shape.key);
      if (["numerical-guard-failed", "missing-baseline-pipeline", "unsupported-baseline", "insufficient-samples"].includes(point.reason)) return finishTall(point.reason === "insufficient-samples" ? "timing-failed" : point.reason, true);
    } catch (error) {
      tallDiagnostics.error = String(error?.message || error);
      if (error?.code === "MATVEC_DEADLINE" || error?.code === "GEMMA4_BATCH_MLP_DEADLINE") { tallDiagnostics.truncated = true; return finishTall("deadline"); }
      return finishTall("calibration-failed", true);
    }
  }
  return finishTall(tallSelection.size ? "calibrated" : "no-stable-win");
}
