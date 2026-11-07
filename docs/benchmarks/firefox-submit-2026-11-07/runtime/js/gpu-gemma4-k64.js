// The K64 prefill candidate is restricted to F16 R3 tiles. Selection is local to
// this model load and requires exact outputs plus stable paired timing wins.

import { bufferUsage, encodeMatmul, inErrorScopes, mmSplits, pipeline, rowsPerThread, scratchBuffer } from "./gpu.js";
import { compareMatvecOutputs, mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";
import { matmulShapes, selectedMatmulKernel } from "./gpu-tuning.js";

export const GEMMA4_K64_TUNING_REVISION = 1;

const TOKENS = 96;
const ROWS = 3;
const SAMPLES = 3;
const REPETITIONS = 4;
const DEADLINE_MS = 1000;
const MAX_SCRATCH = 64 * 1024 * 1024;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

/** R3 and R4 retain the same workgroup count and split-K partition in this range. */
export function gemma4K64Eligible(tokens) {
  return Number.isSafeInteger(tokens) && tokens >= 65 && tokens <= 96;
}

function deadlineError() {
  return Object.assign(new Error("Gemma 4 K64 calibration deadline exceeded"), { code: "GEMMA4_K64_DEADLINE" });
}

function finish(diagnostics, pipelines = null, selection = new Set()) {
  diagnostics.elapsedMs = now() - diagnostics.startedAt;
  delete diagnostics.startedAt;
  return { pipelines, selection, diagnostics };
}

function inputFor(shape) {
  let state = (0x4b363447 ^ shape.N ^ Math.imul(shape.K, 31)) >>> 0;
  const values = new Float32Array(TOKENS * shape.K);
  for (let index = 0; index < values.length; index++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    values[index] = Math.fround((((value ^ (value >>> 14)) >>> 0) / 0x100000000 - 0.5) * 0.08) || 0.00390625;
  }
  return values;
}

function representativeShapes(weights) {
  const shapes = matmulShapes(weights, 24).filter((shape) => ["gate", "up", "down"].includes(shape.suffix));
  // E2B's 12288-wide final block runs after selected-row trimming. Prioritize
  // the 6144-wide projections used by the shared image and question prefix.
  const priority = (shape) => shape.N === 6144 && shape.K === 1536 ? 0 : shape.N === 1536 && shape.K === 6144 ? 1 : shape.N === 12288 || shape.K === 12288 ? 3 : 2;
  shapes.sort((a, b) => priority(a) - priority(b) || b.N * b.K - a.N * a.K);
  return shapes.slice(0, 4);
}

async function candidatePipelines(device, wgsl, baseline) {
  const spec = { weights: "q8", f16: true, rows: ROWS, groups: 1, splitTarget: 256 };
  const layout = baseline.layout;
  const reduceLayout = baseline.reduceLayout;
  const mmLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const reductionLayout = device.createPipelineLayout({ bindGroupLayouts: [reduceLayout] });
  const [mmPipeline, reducePipeline] = await Promise.all([
    pipeline(device, wgsl("matmul_k64", spec), "gemma4_k64_r3", mmLayout),
    pipeline(device, wgsl("reduce", spec), "gemma4_k64_reduce_r3", reductionLayout),
  ]);
  const mm = [];
  const reduce = [];
  mm[ROWS] = mmPipeline;
  reduce[ROWS] = reducePipeline;
  return { ...spec, kernel: "matmul_k64", layout, reduceLayout, mm, reduce };
}

function operation(device, pipes, shape, globals, params, x, bias, y, part) {
  const resources = [globals, params, x, shape.buf, shape.sbuf, bias, y, part];
  return {
    N: shape.N,
    K: shape.K,
    group: device.createBindGroup({ layout: pipes.layout, entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) }),
    reduce: device.createBindGroup({ layout: pipes.reduceLayout, entries: [globals, params, part, bias, y].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
  };
}

function encodePass(encoder, pipes, op, querySet = null, queryIndex = 0, repetitions = 1) {
  const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } } : {});
  for (let repetition = 0; repetition < repetitions; repetition++) encodeMatmul(pass, pipes, op, TOKENS);
  pass.end();
}

async function measureShape(device, shape, tuning, candidate, deadline) {
  const baselineKernel = selectedMatmulKernel(tuning.selection, shape.N, shape.K, TOKENS);
  const baseline = tuning.pipelines[baselineKernel] || tuning.pipelines.generic;
  const base = { shape: `${shape.N}:${shape.K}`, key: `${shape.N}:${shape.K}`, name: shape.name, baselineKernel, tokens: TOKENS, selected: false };
  if (!baseline) return { ...base, reason: "missing-baseline-pipeline" };
  if (baseline.f16 !== true) return { ...base, reason: "baseline-not-f16" };
  const splitCount = Math.max(...[baseline, candidate].map((pipes) => mmSplits(TOKENS, shape.N, shape.K, pipes.splitTarget, 16 * rowsPerThread(TOKENS, pipes), 64 * pipes.groups)));
  const bytes = {
    x: TOKENS * shape.K * 4,
    y: TOKENS * shape.N * 4,
    bias: shape.N * 4,
    part: splitCount > 1 ? splitCount * TOKENS * shape.N * 4 : 16,
    read: Math.max(256, TOKENS * shape.N * 8),
    resolved: 256,
    uniforms: 32,
  };
  const scratchBytes = bytes.x + bytes.y * 2 + bytes.bias + bytes.part + bytes.read + bytes.resolved + bytes.uniforms;
  const maxBuffer = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  if (scratchBytes > MAX_SCRATCH || Math.max(...Object.values(bytes)) > maxBuffer) return { ...base, reason: "scratch-limit", scratchBytes };

  const owned = [];
  let querySet;
  let read;
  let mapped = false;
  try {
    const U = bufferUsage();
    const make = (size, flags, label) => (owned.push(scratchBuffer(device, size, flags, `${shape.name}.k64-tune.${label}`)), owned.at(-1));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const x = make(bytes.x, U.STORAGE | U.COPY_DST, "x");
    const bias = make(bytes.bias, U.STORAGE | U.COPY_DST, "bias");
    const baselineY = make(bytes.y, U.STORAGE | U.COPY_SRC, "baseline-y");
    const candidateY = make(bytes.y, U.STORAGE | U.COPY_SRC, "candidate-y");
    const part = make(bytes.part, U.STORAGE, "part");
    read = make(bytes.read, U.MAP_READ | U.COPY_DST, "read");
    const resolved = make(bytes.resolved, U.QUERY_RESOLVE | U.COPY_SRC, "resolved");
    device.queue.writeBuffer(globals, 0, new Uint32Array([TOKENS, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 0]));
    device.queue.writeBuffer(x, 0, inputFor(shape));
    device.queue.writeBuffer(bias, 0, new Float32Array(shape.N));
    if (now() >= deadline) throw deadlineError();
    const baselineOp = operation(device, baseline, shape, globals, params, x, bias, baselineY, part);
    const candidateOp = operation(device, candidate, shape, globals, params, x, bias, candidateY, part);
    const checkEncoder = device.createCommandEncoder({ label: `${shape.name}.k64-tune.parity` });
    encodePass(checkEncoder, baseline, baselineOp);
    encodePass(checkEncoder, candidate, candidateOp);
    checkEncoder.copyBufferToBuffer(baselineY, 0, read, 0, bytes.y);
    checkEncoder.copyBufferToBuffer(candidateY, 0, read, bytes.y, bytes.y);
    device.queue.submit([checkEncoder.finish()]);
    const withinParityDeadline = await mapBeforeDeadline(read, bytes.y * 2, deadline);
    mapped = true;
    if (!withinParityDeadline) throw deadlineError();
    const outputs = new Float32Array(read.getMappedRange(0, bytes.y * 2).slice(0));
    read.unmap();
    mapped = false;
    const outputLength = TOKENS * shape.N;
    const numericalCheck = compareMatvecOutputs(outputs.subarray(0, outputLength), outputs.subarray(outputLength), { f16: true });
    if (!numericalCheck.ok) return { ...base, numericalCheck, reason: "numerical-guard-failed" };

    const queryCount = SAMPLES * 2 * 2;
    querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `${shape.name}.k64-tune.timestamps` });
    const encoder = device.createCommandEncoder({ label: `${shape.name}.k64-tune.timings` });
    encodePass(encoder, baseline, baselineOp);
    encodePass(encoder, candidate, candidateOp);
    const baselineIndices = [];
    const candidateIndices = [];
    let queryIndex = 0;
    for (let sample = 0; sample < SAMPLES; sample++) {
      if (now() >= deadline) throw deadlineError();
      for (const kind of sample % 2 === 0 ? ["baseline", "candidate"] : ["candidate", "baseline"]) {
        const isBaseline = kind === "baseline";
        encodePass(encoder, isBaseline ? baseline : candidate, isBaseline ? baselineOp : candidateOp, querySet, queryIndex, REPETITIONS);
        (isBaseline ? baselineIndices : candidateIndices).push(queryIndex);
        queryIndex += 2;
      }
    }
    encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8);
    device.queue.submit([encoder.finish()]);
    const withinTimingDeadline = await mapBeforeDeadline(read, queryCount * 8, deadline);
    mapped = true;
    if (!withinTimingDeadline) throw deadlineError();
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0));
    read.unmap();
    mapped = false;
    const timings = (indices) => indices.map((index) => Number(stamps[index + 1] - stamps[index]) / 1e6 / REPETITIONS);
    const baselineMs = timings(baselineIndices);
    const candidateMs = timings(candidateIndices);
    const choice = selectMatvecKernel({ baseline: baselineMs, candidate: candidateMs, numericalCheck });
    return {
      ...base, ...choice, kernel: choice.selected ? "k64" : baselineKernel,
      reason: choice.reason === "stable-matvec-win" ? "stable-k64-win" : choice.reason,
      baseline: baselineMs, candidate: candidateMs, scratchBytes,
    };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    querySet?.destroy();
  }
}

/** Calibrate only the F16 R3 prefill range against actual loaded Q8 projections. */
export async function calibrateGemma4K64(device, wgsl, weights, tuning, { kernel = "auto" } = {}) {
  if (!["auto", "generic", "wide"].includes(kernel)) throw new Error('kernel must be "auto", "generic", or "wide"');
  const startedAt = now();
  const diagnostics = { revision: GEMMA4_K64_TUNING_REVISION, method: "baseline", kernel, tokens: TOKENS, tokenRange: [65, 96], samples: SAMPLES, repetitions: REPETITIONS, threshold: 0.05, points: [], truncated: false, startedAt };
  const fallback = (reason) => { diagnostics.reason = reason; return finish(diagnostics); };
  if (kernel !== "auto") return fallback("kernel-override");
  if (!device.features?.has?.("shader-f16")) return fallback("no-shader-f16");
  if (!device.features?.has?.("timestamp-query")) return fallback("no-timestamp-query");
  if (!tuning?.pipelines?.generic) return fallback("missing-baseline-pipeline");
  const shapes = representativeShapes(weights);
  if (!shapes.length) return fallback("no-q8-prefix-projections");
  diagnostics.method = "paired-timestamp-query";
  const deadline = startedAt + DEADLINE_MS;
  let pipelines;
  try {
    pipelines = await inErrorScopes(device, () => candidatePipelines(device, wgsl, tuning.pipelines.generic));
  } catch (error) {
    diagnostics.error = String(error?.message || error);
    return fallback("pipeline-failed");
  }
  const selection = new Set();
  for (const shape of shapes) {
    if (now() >= deadline) {
      diagnostics.truncated = true;
      diagnostics.reason = "deadline";
      break;
    }
    let point;
    try {
      point = await inErrorScopes(device, () => measureShape(device, shape, tuning, pipelines, deadline));
    } catch (error) {
      if (error?.code === "MATVEC_DEADLINE" || error?.code === "GEMMA4_K64_DEADLINE") {
        diagnostics.truncated = true;
        diagnostics.reason = "deadline";
        break;
      }
      selection.clear();
      diagnostics.reason = "calibration-failed";
      diagnostics.error = String(error?.message || error);
      for (const previous of diagnostics.points) { previous.selected = false; previous.kernel = previous.baselineKernel; previous.reason = "aborted-after-calibration-failure"; }
      break;
    }
    diagnostics.points.push(point);
    if (point.reason === "numerical-guard-failed" || point.reason === "missing-baseline-pipeline" || point.reason === "baseline-not-f16" || point.reason === "insufficient-samples") {
      selection.clear();
      diagnostics.reason = point.reason === "insufficient-samples" ? "timing-failed" : point.reason;
      for (const previous of diagnostics.points) { previous.selected = false; previous.kernel = previous.baselineKernel; if (previous !== point) previous.reason = "aborted-after-calibration-failure"; }
      break;
    }
    if (point.selected) selection.add(point.key);
  }
  return finish(diagnostics, pipelines, selection);
}
