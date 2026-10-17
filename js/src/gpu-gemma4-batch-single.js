// Exact physical BM104 with the existing independent-suffix splitT=1 policy.
import { bufferUsage, encodeMatmul, inErrorScopes, mmSplits, pipeline, rowsPerThread, scratchBuffer } from "./gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "./gpu-gemma4-batch-mlp.js";
import { mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";
import { selectedMatmulKernel } from "./gpu-tuning.js";

const MAX_SCRATCH = 64 * 1024 * 1024;
const SENTINEL = new Uint32Array(new Float32Array([12345.25]).buffer)[0];
const now = () => performance.now();

export function gemma4BatchSingleEligible(tokens, op = null) {
  if (!Number.isSafeInteger(tokens) || tokens < 97 || tokens > 104) return false;
  if (!op) return true;
  return op.kind === "mm" && op.batched === true && op.sourcePolicy !== true && op.dtype === "q8" && op.mode === 0 && op.bias === 0 &&
    (op.label === "mm.q" && op.N === 2048 && op.K === 1536 || op.label === "mm.down" && op.N === 1536 && op.K === 6144);
}

export function gemma4BatchSingleMemory(tokens, N, K) {
  if (!gemma4BatchSingleEligible(tokens) || ![[2048, 1536], [1536, 6144]].some(([n, k]) => n === N && k === K)) return null;
  const splits = mmSplits(1, N, K, 256, 64, 64), padded = Math.max(Math.ceil(tokens / 64) * 64, Math.ceil(tokens / 104) * 104);
  const x = padded * K * 4, y = padded * N * 4 + 16, part = (splits > 1 ? splits * tokens * N * 4 : 0) + 16;
  // Uniforms, zero bias, status, timestamp resolve, and one reusable status/timestamp readback.
  const gpuBytes = x + 2 * y + 2 * part + 5 * 16 + 16 + 16 + 256 + 256;
  // Seeded input plus a conservative upload copy, uniform uploads, and mapped timestamp copies.
  const hostBytes = 2 * tokens * K * 4 + 1024;
  return { x, y, part, splits, gpuBytes, hostBytes, totalBytes: gpuBytes + hostBytes };
}

export function encodeGemma4BatchSingle(pass, pipelines, baseline, op, tokens) {
  if (!gemma4BatchSingleEligible(tokens, op) || baseline?.f16 !== true || baseline.groups !== 1 || baseline.splitTarget !== 256 || rowsPerThread(tokens, baseline) !== 4) throw new Error("Gemma single batch matmul requires independent F16 R4 Q/down suffix rows");
  pass.setPipeline(pipelines.pipeline); pass.setBindGroup(0, op.group);
  const splits = mmSplits(1, op.N, op.K, 256, 64, 64);
  pass.dispatchWorkgroups(Math.ceil(op.N / 64), Math.ceil(tokens / 104), splits);
  if (splits > 1) {
    const work = Math.ceil(tokens * op.N / 256);
    pass.setPipeline(baseline.reduce[4]); pass.setBindGroup(0, op.reduce);
    pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
  }
}

function shapesFor(weights, eligibleShapes) {
  const result = [], seen = new Set();
  for (const [name, tensor] of weights?.tensors || []) {
    if (!/^l\.\d+\.(?:q|down)$/.test(name) || tensor.info?.dtype !== "q8" || !tensor.buf || !tensor.sbuf || !Array.isArray(tensor.info.shape) || tensor.info.shape.length !== 2) continue;
    const [N, K] = tensor.info.shape.map(Number), key = `${N}:${K}`;
    const op = { kind: "mm", label: name.endsWith(".q") ? "mm.q" : "mm.down", N, K, batched: true, dtype: "q8", mode: 0, bias: 0 };
    if (!gemma4BatchSingleEligible(97, op) || seen.has(key) || eligibleShapes && !eligibleShapes.has(key)) continue;
    seen.add(key); result.push({ ...op, key, name, tensor });
  }
  return result.sort((a, b) => b.K - a.K).slice(0, 2);
}

function inputFor(tokens, shape) {
  let seed = (0x4253494e ^ tokens ^ shape.N ^ shape.K) >>> 0;
  const input = new Float32Array(tokens * shape.K);
  for (let i = 0; i < input.length; i++) {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    input[i] = Math.fround(((seed >>> 0) / 0x100000000 - 0.5) * 0.08) || 0.00390625;
  }
  return input;
}

async function measure(device, shape, baseline, candidate, tokens, deadline) {
  const memory = gemma4BatchSingleMemory(tokens, shape.N, shape.K);
  const base = { key: shape.key, name: shape.name, N: shape.N, K: shape.K, tokens, splitTokens: 1, selected: false, scratchBytes: memory.totalBytes };
  const limit = Math.min(device.limits?.maxBufferSize || Infinity, device.limits?.maxStorageBufferBindingSize || Infinity);
  if (memory.totalBytes > MAX_SCRATCH || Math.max(memory.x, memory.y, memory.part) > limit) return { ...base, reason: "scratch-limit" };
  const owned = [];
  let read, queries, mapped = false;
  const checkDeadline = () => { if (now() >= deadline) throw Object.assign(new Error("Gemma single batch calibration deadline exceeded"), { code: "MATVEC_DEADLINE" }); };
  try {
    const U = bufferUsage();
    const make = (size, usage, label) => (owned.push(scratchBuffer(device, size, usage, `Gemma single batch ${label}`)), owned.at(-1));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const x = make(memory.x, U.STORAGE | U.COPY_DST, "input"), zero = make(16, U.STORAGE, "zero");
    const outputs = [0, 1].map(() => make(memory.y, U.STORAGE | U.COPY_SRC | U.COPY_DST, "output"));
    const partials = [0, 1].map(() => make(memory.part, U.STORAGE | U.COPY_SRC | U.COPY_DST, "partial"));
    const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "status");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    read = make(256, U.MAP_READ | U.COPY_DST, "readback");
    device.queue.writeBuffer(globals, 0, new Uint32Array([tokens, 1, tokens, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 0]));
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const ops = outputs.map((output, i) => ({ ...shape,
      group: group(baseline.layout, [globals, params, x, shape.tensor.buf, shape.tensor.sbuf, zero, output, partials[i]]),
      reduce: group(baseline.reduceLayout, [globals, params, partials[i], zero, output]),
    }));
    const validationParams = (length, active) => {
      const p = make(16, U.UNIFORM | U.COPY_DST, "validation params");
      device.queue.writeBuffer(p, 0, new Uint32Array([length, active, SENTINEL, 0x7fc00000]));
      return p;
    };
    const validator = candidate.validator;
    const checks = [
      { length: memory.y / 4, ...gemma4BatchMlpValidationGroups(device, validator, { params: validationParams(memory.y / 4, tokens * shape.N), baseline: outputs[0], candidate: outputs[1], status }) },
      { length: memory.part / 4, ...gemma4BatchMlpValidationGroups(device, validator, { params: validationParams(memory.part / 4, memory.splits > 1 ? memory.splits * tokens * shape.N : 0), baseline: partials[0], candidate: partials[1], status }) },
    ];
    const work = (pass, length) => { const count = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(count, 65535), Math.ceil(count / 65535)); };
    const fillInput = group(validator.fillLayout, [validationParams(memory.x / 4, memory.x / 4), x]);
    const fillEncoder = device.createCommandEncoder(), fillPass = fillEncoder.beginComputePass();
    fillPass.setPipeline(validator.fill); fillPass.setBindGroup(0, fillInput); work(fillPass, memory.x / 4); fillPass.end();
    device.queue.submit([fillEncoder.finish()]);
    device.queue.writeBuffer(x, 0, inputFor(tokens, shape));
    const dispatch = (pass, which) => which ? encodeGemma4BatchSingle(pass, candidate, baseline, ops[1], tokens) : encodeMatmul(pass, baseline, ops[0], tokens, 1);
    const verify = async () => {
      checkDeadline();
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(validator.fill);
      for (const item of checks) for (const binding of item.fill) { fill.setBindGroup(0, binding); work(fill, item.length); }
      fill.end();
      const compute = encoder.beginComputePass(); dispatch(compute, 0); dispatch(compute, 1); compute.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(validator.compare);
      for (const item of checks) { compare.setBindGroup(0, item.compare); work(compare, item.length); }
      compare.end(); encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      if (!within) checkDeadline();
      try {
        const counts = new Uint32Array(read.getMappedRange(0, 16));
        return { ok: counts.every((value) => value === 0), bitMismatches: counts[0], nonFinite: counts[1], tailMismatches: counts[2], values: tokens * shape.N, partialValues: memory.splits * tokens * shape.N, signedZeroChecked: true, validation: "gpu-bitwise", readbackBytes: 16 };
      } finally { read.unmap(); mapped = false; }
    };
    const numericalCheck = await verify();
    if (!numericalCheck.ok) return { ...base, reason: "numerical-guard-failed", numericalCheck };
    queries = device.createQuerySet({ type: "timestamp", count: 12 });
    const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass();
    dispatch(warm, 0); dispatch(warm, 1); warm.end();
    const indices = [[], []]; let index = 0;
    for (let pair = 0; pair < 3; pair++) for (const which of pair % 2 ? [1, 0] : [0, 1]) {
      checkDeadline(); indices[which].push(index);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
      for (let repetition = 0; repetition < 4; repetition++) dispatch(pass, which);
      pass.end(); index += 2;
    }
    encoder.resolveQuerySet(queries, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, 96, deadline); mapped = true;
    if (!within) checkDeadline();
    const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
    const [baselineTimes, candidateTimes] = indices.map((items) => items.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4));
    const afterTimingCheck = await verify();
    if (!afterTimingCheck.ok) return { ...base, reason: "numerical-guard-failed", numericalCheck, afterTimingCheck };
    const choice = selectMatvecKernel({ baseline: baselineTimes, candidate: candidateTimes, numericalCheck });
    return { ...base, ...choice, reason: choice.selected ? "stable-batch-single-win" : choice.reason, baseline: baselineTimes, candidate: candidateTimes, afterTimingCheck };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    queries?.destroy();
  }
}

export async function calibrateGemma4BatchSingle(device, source, weights, tuning, { tokens, batchPipelines, eligibleShapes, kernel = "auto", pipelines = null } = {}) {
  const started = now(), selection = new Set();
  const diagnostics = { tokens, splitTokens: 1, physicalRows: 104, virtualRows: 64, samples: 3, repetitions: 4, threshold: 0.05, points: [], truncated: false };
  const result = (reason) => ({ tokens, pipelines, selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  if (kernel !== "auto") return result("kernel-override");
  if (!gemma4BatchSingleEligible(tokens)) return result("unsupported-physical-tokens");
  if (!device.features?.has?.("shader-f16")) return result("no-shader-f16");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  const shapes = shapesFor(weights, eligibleShapes).map((shape) => {
    const selected = selectedMatmulKernel(tuning?.selection, shape.N, shape.K, 1);
    return { ...shape, baselineKernel: batchPipelines?.[selected] ? selected : "generic", baseline: batchPipelines?.[selected] || batchPipelines?.generic };
  }).filter(({ baseline }) => baseline?.f16 === true && baseline.groups === 1 && baseline.splitTarget === 256 && rowsPerThread(tokens, baseline) === 4 && baseline.mm?.[4] && baseline.reduce?.[4]);
  if (!shapes.length) return result("no-eligible-shapes");
  const deadline = started + 750;
  try {
    if (!pipelines) pipelines = await inErrorScopes(device, async () => ({
      pipeline: await pipeline(device, source("matmul_tall", { weights: "q8", f16: true, rows: 4, row56: false, groups: 1, splitTarget: 256 }), "gemma4_batch_single_tall", device.createPipelineLayout({ bindGroupLayouts: [batchPipelines.generic.layout] })),
      validator: await createGemma4BatchMlpValidator(device),
    }));
  } catch (error) { diagnostics.error = String(error?.message || error); return result("pipeline-failed"); }
  for (const shape of shapes) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    try {
      const point = await inErrorScopes(device, () => measure(device, shape, shape.baseline, pipelines, tokens, deadline));
      diagnostics.points.push({ ...point, baselineKernel: shape.baselineKernel });
      if (point.selected) selection.add(shape.key);
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (error?.code === "MATVEC_DEADLINE") { diagnostics.truncated = true; return result("deadline"); }
      return result("calibration-failed");
    }
  }
  return result(selection.size ? "calibrated" : "no-stable-win");
}
