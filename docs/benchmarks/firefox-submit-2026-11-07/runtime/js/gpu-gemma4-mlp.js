// Gemma's F16 R3 prefix MLP can share input tiles across gate/up and apply GELU
// in the same dispatch. Each model load verifies its weights against the current
// effective matmul selection before enabling a shape.
import { bufferUsage, encodeMatmul, inErrorScopes, mmSplits, pipeline, scratchBuffer } from "./gpu.js";
import { compareMatvecOutputs, mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";
import { selectedMatmulKernel } from "./gpu-tuning.js";

export const GEMMA4_MLP_TUNING_REVISION = 1;
const TOKENS = 96;
const ROWS = 3;
const SPLIT_TARGET = 256;
const SAMPLES = 3;
const REPETITIONS = 4;
const DEADLINE_MS = 750;
const MAX_SCRATCH = 32 * 1024 * 1024;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

export function gemma4MlpEligible(tokens) {
  return Number.isSafeInteger(tokens) && tokens >= 65 && tokens <= 96;
}

async function mlpPipelines(device, source) {
  const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"];
  const layout = device.createBindGroupLayout({ label: "Gemma paired prefix MLP", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const spec = { weights: "q8", f16: true, rows: ROWS, groups: 1, splitTarget: SPLIT_TARGET };
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  return { pipeline: await pipeline(device, source("matmul_gelu", spec), "gemma4_mlp_r3", pl), layout, ...spec };
}

export function gemma4MlpBindGroup(device, pipes, { globals, params, input, gate, gateScales, up, upScales, output, partials }) {
  return device.createBindGroup({ layout: pipes.layout, entries: [globals, params, input, gate, gateScales, up, upScales, output, partials].map((buffer, binding) => ({ binding, resource: { buffer } })) });
}

/** Only selected unsplit shapes in this nonbatched prefix range may use this pipeline. */
export function encodeGemma4Mlp(pass, pipes, op, T) {
  if (!gemma4MlpEligible(T) || op.N < 6144) throw new Error("Gemma paired prefix MLP requires T65..96 and N>=6144");
  pass.setPipeline(pipes.pipeline);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / 64), Math.ceil(T / 48), 1);
}

export function compareGemma4MlpOutputs(baseline, candidate) {
  const check = compareMatvecOutputs(baseline, candidate, { f16: true });
  const baselineBits = new Uint32Array(baseline.buffer, baseline.byteOffset, baseline.length);
  const candidateBits = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let bitMismatches = 0;
  for (let index = 0; index < Math.min(baselineBits.length, candidateBits.length); index++) if (baselineBits[index] !== candidateBits[index]) bitMismatches++;
  return { ...check, bitMismatches, ok: check.ok && bitMismatches === 0 };
}

function pairedShapes(weights) {
  const tensors = weights?.tensors;
  if (!(tensors instanceof Map)) return [];
  const seen = new Set(), shapes = [];
  for (const [name, gate] of tensors) {
    if (!/^l\.\d+\.gate$/.test(name) || gate.info?.dtype !== "q8" || !Array.isArray(gate.info.shape) || gate.info.shape.length !== 2) continue;
    const up = tensors.get(name.replace(/\.gate$/, ".up"));
    const [N, K] = gate.info.shape.map(Number);
    const key = `${N}:${K}`;
    if (seen.has(key) || !Number.isSafeInteger(N) || !Number.isSafeInteger(K) || N < 6144 || K <= 0 || K % 32 !== 0) continue;
    if (up?.info?.dtype !== "q8" || !Array.isArray(up.info.shape) || up.info.shape.length !== 2 || Number(up.info.shape[0]) !== N || Number(up.info.shape[1]) !== K) continue;
    if (!gate.buf || !gate.sbuf || !up.buf || !up.sbuf || mmSplits(TOKENS, N, K, SPLIT_TARGET, 48, 64) !== 1) continue;
    shapes.push({ N, K, key, name, gate, up });
    seen.add(key);
  }
  return shapes.sort((a, b) => a.N * a.K - b.N * b.K).slice(0, 2);
}

function effectiveBaseline(shape, tuning, k64) {
  if (k64?.pipelines && k64.selection?.has(shape.key)) return { kind: "k64", pipes: k64.pipelines };
  const selected = selectedMatmulKernel(tuning.selection, shape.N, shape.K, TOKENS);
  const kind = tuning.pipelines[selected] ? selected : "generic";
  return { kind, pipes: tuning.pipelines[kind] };
}

function inputFor(shape) {
  let state = (0x4d4c5034 ^ shape.N ^ Math.imul(shape.K, 31)) >>> 0;
  const values = new Float32Array(TOKENS * shape.K);
  for (let index = 0; index < values.length; index++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    values[index] = Math.fround((((value ^ (value >>> 14)) >>> 0) / 0x100000000 - 0.5) * 0.08) || 0.00390625;
  }
  return values;
}

function deadlineError() {
  return Object.assign(new Error("Gemma paired prefix MLP calibration deadline exceeded"), { code: "GEMMA4_MLP_DEADLINE" });
}

async function measureShape(device, shape, tuning, k64, candidate, gelu, deadline) {
  const baseline = effectiveBaseline(shape, tuning, k64);
  const { N, K } = shape;
  const base = { key: shape.key, name: shape.name, N, K, tokens: TOKENS, baselineKernel: baseline.kind, selected: false };
  if (!baseline.pipes) return { ...base, reason: "missing-baseline-pipeline" };
  if (baseline.pipes.f16 !== true) return { ...base, reason: "baseline-not-f16" };
  const baselineRows = baseline.kind === "wide" ? 4 : ROWS;
  if (mmSplits(TOKENS, N, K, baseline.pipes.splitTarget, 16 * baselineRows, 64 * baseline.pipes.groups) !== 1) return { ...base, reason: "baseline-not-unsplit" };
  const yBytes = TOKENS * N * 4;
  const bytes = { x: TOKENS * K * 4, y: yBytes, read: Math.max(256, yBytes * 2), resolved: 256 };
  const scratchBytes = bytes.x + bytes.y * 4 + bytes.read + bytes.resolved + 80;
  const maxBuffer = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  if (scratchBytes > MAX_SCRATCH || Math.max(...Object.values(bytes)) > maxBuffer) return { ...base, reason: "scratch-limit", scratchBytes };
  const owned = [];
  let querySet, read, mapped = false;
  try {
    const U = bufferUsage();
    const make = (size, usage, label) => (owned.push(scratchBuffer(device, size, usage, `${shape.name}.mlp-tune.${label}`)), owned.at(-1));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const geluParams = make(16, U.UNIFORM | U.COPY_DST, "gelu-params");
    const x = make(bytes.x, U.STORAGE | U.COPY_DST, "x");
    const zero = make(16, U.STORAGE, "zero");
    const part = make(16, U.STORAGE, "part");
    const gate = make(yBytes, U.STORAGE, "gate"), up = make(yBytes, U.STORAGE, "up");
    const baselineY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-y");
    const candidateY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "candidate-y");
    read = make(bytes.read, U.MAP_READ | U.COPY_DST, "read");
    const resolved = make(bytes.resolved, U.QUERY_RESOLVE | U.COPY_SRC, "resolved");
    device.queue.writeBuffer(globals, 0, new Uint32Array([TOKENS, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
    device.queue.writeBuffer(x, 0, inputFor(shape));
    const poison = new Float32Array(TOKENS * N).fill(NaN);
    device.queue.writeBuffer(baselineY, 0, poison);
    device.queue.writeBuffer(candidateY, 0, poison);
    if (now() >= deadline) throw deadlineError();
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const opFor = (weights, output) => ({ N, K,
      group: group(baseline.pipes.layout, [globals, params, x, weights.buf, weights.sbuf, zero, output, part]),
      reduce: group(baseline.pipes.reduceLayout, [globals, params, part, zero, output]),
    });
    const gateOp = opFor(shape.gate, gate), upOp = opFor(shape.up, up);
    const geluGroup = group(gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, baselineY]);
    const candidateOp = { N, group: gemma4MlpBindGroup(device, candidate, { globals, params, input: x, gate: shape.gate.buf, gateScales: shape.gate.sbuf, up: shape.up.buf, upScales: shape.up.sbuf, output: candidateY, partials: part }) };
    const dispatch = (pass, fused) => {
      if (fused) return encodeGemma4Mlp(pass, candidate, candidateOp, TOKENS);
      encodeMatmul(pass, baseline.pipes, gateOp, TOKENS);
      encodeMatmul(pass, baseline.pipes, upOp, TOKENS);
      pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup);
      const work = Math.ceil(TOKENS * N / 256);
      pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
    };
    const checkEncoder = device.createCommandEncoder();
    const checkPass = checkEncoder.beginComputePass();
    dispatch(checkPass, false); dispatch(checkPass, true); checkPass.end();
    checkEncoder.copyBufferToBuffer(baselineY, 0, read, 0, yBytes);
    checkEncoder.copyBufferToBuffer(candidateY, 0, read, yBytes, yBytes);
    device.queue.submit([checkEncoder.finish()]);
    const withinParityDeadline = await mapBeforeDeadline(read, yBytes * 2, deadline);
    mapped = true;
    if (!withinParityDeadline) throw deadlineError();
    const outputs = new Float32Array(read.getMappedRange(0, yBytes * 2).slice(0));
    read.unmap(); mapped = false;
    const length = TOKENS * N;
    const numericalCheck = compareGemma4MlpOutputs(outputs.subarray(0, length), outputs.subarray(length));
    if (!numericalCheck.ok) return { ...base, numericalCheck, reason: "numerical-guard-failed" };
    const queryCount = SAMPLES * 4;
    querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `${shape.name}.mlp-tune.timestamps` });
    const encoder = device.createCommandEncoder();
    const warmup = encoder.beginComputePass(); dispatch(warmup, false); dispatch(warmup, true); warmup.end();
    const indices = { baseline: [], candidate: [] };
    let index = 0;
    for (let sample = 0; sample < SAMPLES; sample++) {
      if (now() >= deadline) throw deadlineError();
      for (const fused of sample % 2 ? [true, false] : [false, true]) {
        indices[fused ? "candidate" : "baseline"].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repetition = 0; repetition < REPETITIONS; repetition++) dispatch(pass, fused);
        pass.end(); index += 2;
      }
    }
    encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8);
    device.queue.submit([encoder.finish()]);
    const withinTimingDeadline = await mapBeforeDeadline(read, queryCount * 8, deadline);
    mapped = true;
    if (!withinTimingDeadline) throw deadlineError();
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0));
    read.unmap(); mapped = false;
    const times = Object.fromEntries(Object.entries(indices).map(([label, values]) => [label, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS)]));
    const choice = selectMatvecKernel({ ...times, numericalCheck });
    return { ...base, ...choice, ...times, scratchBytes, kernel: choice.selected ? "paired-mlp" : baseline.kind, reason: choice.reason === "stable-matvec-win" ? "stable-paired-mlp-win" : choice.reason };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    querySet?.destroy();
  }
}

/** Only exact, faster F16 R3 prefix projections are selected for this loaded model. */
export async function calibrateGemma4Mlp(device, source, weights, tuning, { k64 = null, gelu, kernel = "auto", enabled = true } = {}) {
  if (!["auto", "generic", "wide"].includes(kernel)) throw new Error('kernel must be "auto", "generic", or "wide"');
  const started = now(), selection = new Set();
  const diagnostics = { revision: GEMMA4_MLP_TUNING_REVISION, method: "baseline", kernel, tokens: TOKENS, tokenRange: [65, 96], samples: SAMPLES, repetitions: REPETITIONS, threshold: 0.05, points: [], truncated: false };
  const result = (reason, pipelines = null) => ({ pipelines, selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  let pipelines;
  const abort = (reason) => {
    selection.clear();
    for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.kernel = point.baselineKernel; point.reason = "aborted-after-calibration-failure"; }
    return result(reason, pipelines);
  };
  if (!enabled) return result("disabled");
  if (kernel !== "auto") return result("kernel-override");
  if (!device.features?.has?.("shader-f16")) return result("no-shader-f16");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  if (!tuning?.pipelines?.generic || !gelu) return result("missing-baseline-pipeline");
  const shapes = pairedShapes(weights);
  if (!shapes.length) return result("no-q8-prefix-pairs");
  diagnostics.method = "paired-timestamp-query";
  try { pipelines = await inErrorScopes(device, () => mlpPipelines(device, source)); }
  catch (error) { diagnostics.error = String(error?.message || error); return result("pipeline-failed"); }
  const deadline = started + DEADLINE_MS;
  for (const shape of shapes) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline", pipelines); }
    try {
      const point = await inErrorScopes(device, () => measureShape(device, shape, tuning, k64, pipelines, gelu, deadline));
      diagnostics.points.push(point);
      if (point.selected) selection.add(shape.key);
      if (["numerical-guard-failed", "missing-baseline-pipeline", "baseline-not-f16", "baseline-not-unsplit", "insufficient-samples"].includes(point.reason)) return abort(point.reason === "insufficient-samples" ? "timing-failed" : point.reason);
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (error?.code === "MATVEC_DEADLINE" || error?.code === "GEMMA4_MLP_DEADLINE") { diagnostics.truncated = true; return result("deadline", pipelines); }
      return abort("calibration-failed");
    }
  }
  return result(selection.size ? "calibrated" : "no-stable-win", pipelines);
}
