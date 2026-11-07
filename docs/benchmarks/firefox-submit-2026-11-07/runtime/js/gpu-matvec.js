// The single-row path is calibrated against the selected matmul kernel using the
// model's loaded projection buffers. It keeps no device-specific results beyond this load.

import { bufferUsage, encodeMatmul, encodeMatvec, inErrorScopes, matvecPipelines, mmSplits, rowsPerThread, scratchBuffer } from "./gpu.js";
import { matmulShapes, selectMatmulKernel, selectedMatmulKernel } from "./gpu-tuning.js";

export const MATVEC_TUNING_REVISION = 2;

const SAMPLES = 3;
const REPETITIONS = 4;
const WALL_REPETITIONS = 64;
const THRESHOLD = 0.05;
const DEADLINE_MS = 1500;
const MAX_SCRATCH = 64 * 1024 * 1024;
const MAX_CLOCK_RESOLUTION_MS = 2;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

export function matvecTimingMode(features, clockResolutionMs) {
  if (features?.has?.("timestamp-query")) return { method: "paired-timestamp-query", repetitions: REPETITIONS };
  if (Number.isFinite(clockResolutionMs) && clockResolutionMs > 0 && clockResolutionMs <= MAX_CLOCK_RESOLUTION_MS) {
    return { method: "paired-wall-clock", repetitions: WALL_REPETITIONS };
  }
  return { method: "baseline", repetitions: 0, reason: "no-wall-clock-resolution" };
}

function clockResolutionMs() {
  let previous = now();
  let resolution = Infinity;
  const deadline = Date.now() + 4;
  for (let index = 0; index < 262144 && Date.now() <= deadline; index++) {
    const current = now();
    if (current > previous) {
      resolution = Math.min(resolution, current - previous);
      if (resolution <= MAX_CLOCK_RESOLUTION_MS) break;
    }
    previous = current;
  }
  return Number.isFinite(resolution) ? resolution : null;
}
const validSamples = (values) => Array.isArray(values) && values.length === SAMPLES && values.every((value) => Number.isFinite(value) && value > 0);

function fallback(reason, diagnostics, extra = {}) {
  diagnostics.reason = reason;
  diagnostics.elapsedMs = now() - diagnostics.startedAt;
  delete diagnostics.startedAt;
  return { pipelines: null, selection: new Set(), diagnostics: { ...diagnostics, ...extra } };
}

/** Ordered F16 must match exactly; the F32 reduction has a bounded rounding allowance. */
export function compareMatvecOutputs(baseline, candidate, { f16 = false } = {}) {
  const maxAbsTolerance = f16 ? 0 : 0.0005;
  const relativeTolerance = f16 ? 0 : 0.0001;
  let maxAbs = 0;
  let worstIndex = null;
  let nonFinite = 0;
  let violations = 0;
  const lengthMismatch = baseline.length !== candidate.length;
  for (let index = 0; index < Math.min(baseline.length, candidate.length); index++) {
    const expected = baseline[index];
    const actual = candidate[index];
    if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
      nonFinite++;
      continue;
    }
    const difference = Math.abs(actual - expected);
    const tolerance = f16
      ? maxAbsTolerance + relativeTolerance * Math.abs(expected)
      : maxAbsTolerance + relativeTolerance * Math.max(1, Math.abs(expected));
    if (difference > maxAbs) {
      maxAbs = difference;
      worstIndex = index;
    }
    if (difference > tolerance) violations++;
  }
  return {
    ok: !lengthMismatch && nonFinite === 0 && violations === 0,
    length: candidate.length,
    expectedLength: baseline.length,
    lengthMismatch,
    nonFinite,
    violations,
    maxAbs: Number.isFinite(maxAbs) ? maxAbs : null,
    worstIndex,
    tolerance: { absolute: maxAbsTolerance, relative: relativeTolerance, f16 },
  };
}

/** The new path must have a finite parity guard and stable paired wins before it can be selected. */
export function selectMatvecKernel({ baseline, candidate, numericalCheck }, { threshold = THRESHOLD } = {}) {
  if (numericalCheck?.ok !== true) return { kernel: "matmul", reason: "numerical-guard-failed", selected: false, numericalCheck: numericalCheck || null };
  if (!validSamples(baseline) || !validSamples(candidate) || baseline.length !== candidate.length) {
    return { kernel: "matmul", reason: "insufficient-samples", selected: false, numericalCheck };
  }
  const picked = selectMatmulKernel({ generic: baseline, wide: candidate }, { threshold });
  return {
    ...picked,
    kernel: picked.kernel === "wide" ? "matvec" : "matmul",
    reason: picked.reason === "stable-wide-win" ? "stable-matvec-win" : picked.reason,
    selected: picked.kernel === "wide",
    numericalCheck,
  };
}

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

function inputFor(shape) {
  const next = random(0x4b455641 ^ shape.N ^ Math.imul(shape.K, 31));
  const x = new Float32Array(shape.K);
  for (let index = 0; index < x.length; index++) {
    const value = Math.fround((next() - 0.5) * 0.08);
    x[index] = value || 0.00390625;
  }
  return x;
}

function timedPass(encoder, kind, pipes, op, querySet, index) {
  const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
  for (let repetition = 0; repetition < REPETITIONS; repetition++) {
    // T is fixed to one for this tuner; the direct kernel never uses split-K scratch.
    if (kind === "matvec") encodeMatvec(pass, pipes, op, 1);
    else encodeMatmul(pass, pipes, op, 1);
  }
  pass.end();
}

function repeatedPass(encoder, kind, pipes, op, repetitions) {
  const pass = encoder.beginComputePass();
  for (let repetition = 0; repetition < repetitions; repetition++) {
    if (kind === "matvec") encodeMatvec(pass, pipes, op, 1);
    else encodeMatmul(pass, pipes, op, 1);
  }
  pass.end();
}

function untimedPass(encoder, kind, pipes, op) {
  const pass = encoder.beginComputePass();
  if (kind === "matvec") encodeMatvec(pass, pipes, op, 1);
  else encodeMatmul(pass, pipes, op, 1);
  pass.end();
}

function opFor(device, layout, reduceLayout, shape, globals, params, x, bias, y, part) {
  const resources = [globals, params, x, shape.buf, shape.sbuf || bias, bias, y, part];
  const group = device.createBindGroup({ layout, entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const reduce = reduceLayout ? device.createBindGroup({
    layout: reduceLayout,
    entries: [globals, params, part, bias, y].map((buffer, binding) => ({ binding, resource: { buffer } })),
  }) : null;
  return { N: shape.N, K: shape.K, group, reduce };
}

function deadlineError() {
  return Object.assign(new Error("matvec calibration deadline exceeded"), { code: "MATVEC_DEADLINE" });
}

export async function mapBeforeDeadline(read, size, deadline) {
  const remaining = deadline - now();
  if (remaining <= 0) throw deadlineError();
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(deadlineError());
      read.unmap();
    }, remaining);
  });
  try { await Promise.race([read.mapAsync(GPUMapMode.READ, 0, size), timeout]); }
  finally { clearTimeout(timer); }
  return now() < deadline;
}

async function wallClockSample(device, kind, pipes, op, output, read, outputBytes, repetitions, deadline) {
  if (now() >= deadline) throw deadlineError();
  const encoder = device.createCommandEncoder({ label: `${op.N}x${op.K}.matvec-tune.wall-clock` });
  repeatedPass(encoder, kind, pipes, op, repetitions);
  encoder.copyBufferToBuffer(output, 0, read, 0, outputBytes);
  const command = encoder.finish();
  const started = now();
  device.queue.submit([command]);
  let mapped = false;
  try {
    const withinDeadline = await mapBeforeDeadline(read, outputBytes, deadline);
    mapped = true;
    const finished = now();
    const values = new Float32Array(read.getMappedRange(0, outputBytes).slice(0));
    read.unmap();
    mapped = false;
    if (values.some((value) => !Number.isFinite(value))) throw Object.assign(new Error("matvec wall-clock output contains a non-finite value"), { code: "MATVEC_NUMERICAL" });
    if (!withinDeadline) throw deadlineError();
    return (finished - started) / repetitions;
  } finally {
    if (mapped) read.unmap();
  }
}

async function calibrateShape(device, shape, baselinePipelines, candidatePipelines, deadline, method) {
  const usage = bufferUsage();
  const N = shape.N;
  const K = shape.K;
  const baselineKernel = selectedMatmulKernel(baselinePipelines.selection, N, K, 1);
  const baselinePipe = baselinePipelines.pipelines?.[baselineKernel] || baselinePipelines.pipelines?.generic;
  if (!baselinePipe) return { shape: `${N}:${K}`, name: shape.name, reason: "missing-baseline-pipeline" };
  const rows = rowsPerThread(1, baselinePipe);
  const splitCount = mmSplits(1, N, K, baselinePipe.splitTarget, 16 * rows, 64 * baselinePipe.groups);
  const bytes = {
    x: K * 4,
    y: N * 4,
    bias: N * 4,
    part: Math.max(16, splitCount * N * 4),
    uniforms: 32,
    read: Math.max(256, N * 8),
    resolved: method === "paired-timestamp-query" ? 256 : 0,
  };
  const maxBinding = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  const largest = Math.max(bytes.x, bytes.y, bytes.bias, bytes.part, bytes.read, bytes.resolved, 16);
  const scratchTotal = bytes.x + bytes.y * 2 + bytes.bias + bytes.part + bytes.uniforms + bytes.read + bytes.resolved;
  if (largest > maxBinding || scratchTotal > MAX_SCRATCH) return { shape: `${N}:${K}`, name: shape.name, baselineKernel, reason: "scratch-limit", scratchBytes: scratchTotal };

  const owned = [];
  let querySet;
  let readBuffer;
  let mapped = false;
  try {
    const make = (size, flags, label) => (owned.push(scratchBuffer(device, size, flags, label)), owned.at(-1));
    const globals = make(16, usage.UNIFORM | usage.COPY_DST, `${shape.name}.matvec-tune.globals`);
    const params = make(16, usage.UNIFORM | usage.COPY_DST, `${shape.name}.matvec-tune.params`);
    const x = make(bytes.x, usage.STORAGE | usage.COPY_DST, `${shape.name}.matvec-tune.x`);
    const bias = make(bytes.bias, usage.STORAGE | usage.COPY_DST, `${shape.name}.matvec-tune.bias`);
    const baselineY = make(bytes.y, usage.STORAGE | usage.COPY_SRC, `${shape.name}.matvec-tune.baseline-y`);
    const candidateY = make(bytes.y, usage.STORAGE | usage.COPY_SRC, `${shape.name}.matvec-tune.candidate-y`);
    const part = make(bytes.part, usage.STORAGE, `${shape.name}.matvec-tune.part`);
    const read = readBuffer = make(bytes.read, usage.MAP_READ | usage.COPY_DST, `${shape.name}.matvec-tune.read`);
    const resolved = method === "paired-timestamp-query" ? make(bytes.resolved, usage.QUERY_RESOLVE | usage.COPY_SRC, `${shape.name}.matvec-tune.resolved`) : null;
    const xValues = inputFor(shape);
    device.queue.writeBuffer(globals, 0, new Uint32Array([1, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(x, 0, xValues);
    device.queue.writeBuffer(bias, 0, new Float32Array(N));
    const baselineOp = opFor(device, baselinePipe.layout, baselinePipe.reduceLayout, shape, globals, params, x, bias, baselineY, part);
    const candidateOp = opFor(device, candidatePipelines.layout, null, shape, globals, params, x, bias, candidateY, part);
    const checkEncoder = device.createCommandEncoder({ label: `${shape.name}.matvec-tune.parity` });
    untimedPass(checkEncoder, "matmul", baselinePipe, baselineOp);
    untimedPass(checkEncoder, "matvec", candidatePipelines, candidateOp);
    checkEncoder.copyBufferToBuffer(baselineY, 0, read, 0, bytes.y);
    checkEncoder.copyBufferToBuffer(candidateY, 0, read, bytes.y, bytes.y);
    device.queue.submit([checkEncoder.finish()]);
    const readBytes = bytes.y * 2;
    const withinDeadline = await mapBeforeDeadline(read, readBytes, deadline);
    mapped = true;
    if (!withinDeadline) throw deadlineError();
    const check = new Float32Array(read.getMappedRange(0, readBytes).slice(0));
    read.unmap();
    mapped = false;
    const baselineOutput = check.slice(0, N);
    const candidateOutput = check.slice(N);
    const numericalCheck = compareMatvecOutputs(baselineOutput, candidateOutput, { f16: baselinePipe.f16 === true });
    if (!numericalCheck.ok) return { shape: `${N}:${K}`, name: shape.name, baselineKernel, numericalCheck, reason: "numerical-guard-failed" };

    let baselineMs = [];
    let candidateMs = [];
    if (method === "paired-timestamp-query") {
      querySet = device.createQuerySet({ type: "timestamp", count: SAMPLES * 2 * 2, label: `${shape.name}.matvec-tune.timestamps` });
      const encoder = device.createCommandEncoder({ label: `${shape.name}.matvec-tune.timestamps` });
      untimedPass(encoder, "matmul", baselinePipe, baselineOp);
      untimedPass(encoder, "matvec", candidatePipelines, candidateOp);
      const baselineIndices = [];
      const candidateIndices = [];
      let queryIndex = 0;
      for (let sample = 0; sample < SAMPLES; sample++) {
        if (now() >= deadline) throw deadlineError();
        const order = sample % 2 === 0 ? ["matmul", "matvec"] : ["matvec", "matmul"];
        for (const kind of order) {
          const index = queryIndex;
          if (kind === "matmul") timedPass(encoder, kind, baselinePipe, baselineOp, querySet, index);
          else timedPass(encoder, kind, candidatePipelines, candidateOp, querySet, index);
          (kind === "matmul" ? baselineIndices : candidateIndices).push(index);
          queryIndex += 2;
        }
      }
      encoder.resolveQuerySet(querySet, 0, queryIndex, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, read, 0, queryIndex * 8);
      device.queue.submit([encoder.finish()]);
      const withinTimingDeadline = await mapBeforeDeadline(read, queryIndex * 8, deadline);
      mapped = true;
      if (!withinTimingDeadline) throw deadlineError();
      const stamps = new BigInt64Array(read.getMappedRange(0, queryIndex * 8).slice(0));
      read.unmap();
      mapped = false;
      const ms = (indices) => indices.map((index) => Number(stamps[index + 1] - stamps[index]) / 1e6 / REPETITIONS);
      baselineMs = ms(baselineIndices);
      candidateMs = ms(candidateIndices);
    } else {
      for (let sample = 0; sample < SAMPLES; sample++) {
        if (now() >= deadline) throw deadlineError();
        const order = sample % 2 === 0 ? ["matmul", "matvec"] : ["matvec", "matmul"];
        for (const kind of order) {
          if (kind === "matmul") baselineMs.push(await wallClockSample(device, kind, baselinePipe, baselineOp, baselineY, read, bytes.y, WALL_REPETITIONS, deadline));
          else candidateMs.push(await wallClockSample(device, kind, candidatePipelines, candidateOp, candidateY, read, bytes.y, WALL_REPETITIONS, deadline));
        }
      }
    }
    return {
      shape: `${N}:${K}`, name: shape.name, key: `${N}:${K}`, baselineKernel,
      baseline: baselineMs, candidate: candidateMs, numericalCheck,
    };
  } finally {
    if (mapped) readBuffer?.unmap();
    for (const buffer of owned) buffer.destroy();
    querySet?.destroy();
  }
}

/** Calibrate one precision's T=1 matvec path against the loaded model's selected matmul. */
export async function calibrateMatvec(device, wgsl, weights, tuning, { kernel = "auto", dtype = "q8" } = {}) {
  if (!["auto", "generic", "wide"].includes(kernel)) throw new Error('kernel must be "auto", "generic", or "wide"');
  if (!["q8", "bf16", "f32"].includes(dtype)) throw new Error(`unsupported GPU matrix dtype ${dtype}`);
  const startedAt = now();
  const diagnostics = {
    revision: MATVEC_TUNING_REVISION,
    dtype,
    method: kernel === "auto" ? "baseline" : "override",
    kernel,
    tokens: 1,
    samples: SAMPLES,
    repetitions: 0,
    threshold: THRESHOLD,
    points: [],
    truncated: false,
    startedAt,
  };
  if (kernel !== "auto") return fallback("kernel-override", diagnostics, { baseline: kernel });
  const hasTimestamps = device.features?.has?.("timestamp-query") === true;
  const clockResolution = hasTimestamps ? null : clockResolutionMs();
  const timing = matvecTimingMode(device.features, clockResolution);
  diagnostics.method = timing.method;
  diagnostics.repetitions = timing.repetitions;
  if (!hasTimestamps) diagnostics.clockResolutionMs = clockResolution;
  const shapes = matmulShapes(weights, 12, dtype);
  if (!shapes.length) return fallback(`no-${dtype}-projection-shapes`, diagnostics);
  if (timing.method === "baseline") return fallback(timing.reason, diagnostics);
  if (!tuning?.pipelines?.generic) return fallback("missing-baseline-pipeline", diagnostics);

  const selection = new Set();
  let pipelines;
  try {
    pipelines = await inErrorScopes(device, () => matvecPipelines(device, wgsl, dtype, tuning.pipelines.generic.layout));
  } catch (error) {
    return fallback("pipeline-failed", diagnostics, { error: String(error?.message || error) });
  }
  const deadline = startedAt + DEADLINE_MS;
  if (now() >= deadline) {
    diagnostics.truncated = true;
    diagnostics.reason = "deadline";
    diagnostics.elapsedMs = now() - startedAt;
    delete diagnostics.startedAt;
    return { pipelines, selection, diagnostics };
  }

  for (const shape of shapes) {
    if (now() >= deadline) {
      diagnostics.truncated = true;
      diagnostics.reason = "deadline";
      break;
    }
    let point;
    try {
      point = await inErrorScopes(device, () => calibrateShape(device, shape, tuning, pipelines, deadline, timing.method));
    } catch (error) {
      if (error?.code === "MATVEC_DEADLINE") {
        diagnostics.truncated = true;
        diagnostics.reason = "deadline";
        break;
      }
      selection.clear();
      diagnostics.reason = "calibration-failed";
      diagnostics.error = String(error?.message || error);
      for (const previous of diagnostics.points) {
        previous.selected = false;
        previous.reason = "aborted-after-calibration-failure";
      }
      break;
    }
    if (point.reason === "numerical-guard-failed") {
      selection.clear();
      point.selected = false;
      diagnostics.points.push(point);
      diagnostics.reason = "numerical-guard-failed";
      for (const previous of diagnostics.points.slice(0, -1)) {
        previous.selected = false;
        previous.reason = "aborted-after-parity-failure";
      }
      break;
    }
    if (point.baseline && point.candidate) {
      if (!validSamples(point.baseline) || !validSamples(point.candidate)) {
        selection.clear();
        point.selected = false;
        point.reason = "invalid-samples";
        diagnostics.points.push(point);
        diagnostics.reason = "timing-failed";
        for (const previous of diagnostics.points.slice(0, -1)) {
          previous.selected = false;
          previous.reason = "aborted-after-timing-failure";
        }
        break;
      }
      const choice = selectMatvecKernel({ baseline: point.baseline, candidate: point.candidate, numericalCheck: point.numericalCheck });
      Object.assign(point, choice);
      if (choice.selected) selection.add(point.key);
    } else point.selected = false;
    diagnostics.points.push(point);
    if (point.reason === "missing-baseline-pipeline") {
      selection.clear();
      diagnostics.reason = point.reason;
      for (const previous of diagnostics.points) previous.selected = false;
      break;
    }
  }
  diagnostics.elapsedMs = now() - startedAt;
  delete diagnostics.startedAt;
  return { pipelines, selection, diagnostics };
}
