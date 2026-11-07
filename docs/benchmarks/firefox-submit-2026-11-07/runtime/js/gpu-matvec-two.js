// Share Q8 weight loads across two independent rows, only after exact loaded-weight calibration.
import { bufferUsage, encodeMatvec, inErrorScopes, pipeline, scratchBuffer } from "./gpu.js";
import { encodeMatvecGelu } from "./gpu-matvec-gelu.js";
import { mapBeforeDeadline } from "./gpu-matvec.js";
import { selectMatmulKernel } from "./gpu-tuning.js";

const SAMPLES = 3;
const REPETITIONS = 4;
const THRESHOLD = 0.05;
const DEADLINE_MS = 1000;
const MAX_SCRATCH = 32 * 1024 * 1024;
const TAIL = new Float32Array([12345.25, -1731.5, 913.75, -0]);
const RANK = new Map(["down", "q", "o", "ple_gate", "ple_out", "gate", "up", "k", "v"].map((name, rank) => [name, rank]));
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

/** Require every finite F32 bit, including signed zero, to agree. */
export function compareMatvecTwoOutputs(baseline, candidate) {
  const expected = baseline instanceof Float32Array ? baseline : Float32Array.from(baseline);
  const actual = candidate instanceof Float32Array ? candidate : Float32Array.from(candidate);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let nonFinite = 0, bitDifferences = 0, maxAbs = 0, firstDifference = null;
  for (let index = 0; index < Math.min(expected.length, actual.length); index++) {
    if (!Number.isFinite(expected[index]) || !Number.isFinite(actual[index])) nonFinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(expected[index] - actual[index]));
    if (expectedBits[index] !== actualBits[index]) {
      bitDifferences++;
      firstDifference ||= { index, expected: expected[index], actual: actual[index], expectedBits: expectedBits[index], actualBits: actualBits[index] };
    }
  }
  const lengthMismatch = expected.length !== actual.length;
  return { ok: !lengthMismatch && nonFinite === 0 && bitDifferences === 0, bitExact: !lengthMismatch && nonFinite === 0 && bitDifferences === 0,
    length: actual.length, expectedLength: expected.length, lengthMismatch, nonFinite, bitDifferences, maxAbs, firstDifference,
    tolerance: { absolute: 0, relative: 0, exact: true } };
}

/** Three alternating pairs must show a stable >=5% win after the exact output guard. */
export function selectMatvecTwoKernel({ baseline, candidate, numericalCheck }) {
  if (numericalCheck?.ok !== true) return { selected: false, reason: "numerical-guard-failed", numericalCheck: numericalCheck || null };
  const valid = (samples) => Array.isArray(samples) && samples.length === SAMPLES && samples.every((value) => Number.isFinite(value) && value > 0);
  if (!valid(baseline) || !valid(candidate)) return { selected: false, reason: "invalid-samples", numericalCheck };
  const choice = selectMatmulKernel({ generic: baseline, wide: candidate }, { threshold: THRESHOLD });
  return { selected: choice.kernel === "wide", reason: choice.kernel === "wide" ? "stable-two-row-win" : choice.reason,
    baselineMs: choice.genericMs, candidateMs: choice.wideMs, benefit: choice.benefit, stable: choice.stable, numericalCheck };
}

export function encodeMatvecTwo(pass, pipes, op, T) {
  if (T !== 2) throw new RangeError("two-row matvec requires T=2");
  const subgroup = pipes.f16 === false && !op.sourcePolicy && pipes.subgroup?.enabled !== false && pipes.subgroup?.selection.has(`${op.N}:${op.K}`) ? pipes.subgroup : null;
  pass.setPipeline(subgroup?.pipeline || pipes.pipeline);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / pipes.columns), 1);
}

function projection(name, value) {
  const match = /^l\.(\d+)\.(.+)$/.exec(name);
  const info = value?.info, shape = info?.shape;
  if (!match || !RANK.has(match[2]) || info?.dtype !== "q8" || info.block !== 32 || !Array.isArray(shape) || shape.length !== 2 || !value.buf || !value.sbuf) return null;
  const [N, K] = shape.map(Number);
  if (!Number.isSafeInteger(N) || !Number.isSafeInteger(K) || !Number.isSafeInteger(N * K) || N <= 0 || K <= 0 || K % 32) return null;
  return { ...value, name, layer: Number(match[1]), suffix: match[2], N, K, key: `${N}:${K}` };
}

function shapesFor(weights, selected, sharedStart) {
  if (!(weights?.tensors instanceof Map)) return { plain: [], gelu: [] };
  const entries = [...weights.tensors].map(([name, value]) => projection(name, value)).filter((shape) => shape && selected?.has(shape.key));
  entries.sort((a, b) => Number(b.layer >= sharedStart) - Number(a.layer >= sharedStart) || RANK.get(a.suffix) - RANK.get(b.suffix) || a.layer - b.layer || b.N * b.K - a.N * a.K);
  const plain = [], gelu = [], plainSeen = new Set(), pairSeen = new Set();
  for (const shape of entries) {
    if (plain.length < 8 && !plainSeen.has(shape.key)) { plain.push(shape); plainSeen.add(shape.key); }
    if (shape.suffix !== "gate" || pairSeen.has(shape.key)) continue;
    const upName = shape.name.replace(/\.gate$/, ".up");
    const up = projection(upName, weights.tensors.get(upName));
    if (!up || up.N !== shape.N || up.K !== shape.K) continue;
    gelu.push({ ...shape, gate: shape, up }); pairSeen.add(shape.key);
  }
  gelu.sort((a, b) => Number(b.layer >= sharedStart) - Number(a.layer >= sharedStart) || b.N * b.K - a.N * a.K || a.layer - b.layer);
  return { plain, gelu: gelu.slice(0, 2) };
}

function pairLayout(device) {
  const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
  return device.createBindGroupLayout({ label: "Gemma two-row paired matvec", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
}

async function makePipelines(device, wgsl, baseline, pairs, needed, subgroups = false) {
  const f16 = baseline.f16;
  const spec = { weights: "q8", f16, rows: 1, groups: 1, splitTarget: 256, ...(subgroups ? { subgroups: true } : {}) };
  const result = { plain: null, gelu: null };
  for (const kind of ["plain", "gelu"]) {
    if (!needed[kind].length) continue;
    const layout = kind === "plain" ? baseline.layout : pairs?.layout || pairLayout(device);
    const name = kind === "plain" ? "matvec_two" : "matvec_gelu_two";
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    result[kind] = { pipeline: await pipeline(device, wgsl(name, spec), name + (subgroups ? "_subgroup" : ""), pl), layout, weights: "q8", f16, columns: f16 ? 16 : 4 };
  }
  return result;
}

// Surface GPU scope failures ahead of a simultaneous deadline, which may preserve earlier points.
async function checked(device, work) {
  const outcome = await inErrorScopes(device, async () => {
    try { return { value: await work() }; }
    catch (error) { return { error }; }
  });
  if (outcome.error) throw outcome.error;
  return outcome.value;
}

function deadlineError() { return Object.assign(new Error("two-row matvec calibration deadline"), { code: "MATVEC_DEADLINE" }); }

function inputFor(K) {
  const values = new Float32Array(2 * K + TAIL.length).fill(NaN);
  for (let row = 0; row < 2; row++) for (let i = 0; i < K; i++) values[row * K + i] = 0.037 * Math.sin(i * 0.127 + row * 1.731 + 0.3) + 0.011 * Math.cos(i * 0.071 - row * 0.417);
  return values;
}

async function measureShape(device, kind, shape, candidates, baseline, paired, gelu, deadline, referenceTwo = null) {
  const { N, K, key, name } = shape;
  const candidate = candidates[kind];
  const oldTwo = referenceTwo?.[kind];
  const oldPair = kind === "gelu" && paired?.selection?.has(key) ? paired.pipelines : null;
  const baselineKernel = oldTwo ? kind === "plain" ? "matvec-two" : "matvec-gelu-two" : kind === "plain" ? "matvec" : oldPair ? "matvec-gelu" : "matvec+matvec+gelu";
  if (oldPair && oldPair.f16 !== baseline.f16) return { kind, key, name, selected: false, reason: "baseline-precision-mismatch" };
  const U = bufferUsage(), floats = 2 * N, yBytes = (floats + TAIL.length) * 4, xBytes = (2 * K + TAIL.length) * 4;
  const separatePair = kind === "gelu" && !oldPair && !oldTwo;
  const readBytes = Math.max(256, yBytes * 2 + (separatePair ? TAIL.byteLength * 2 : 0));
  const scratchBytes = xBytes + yBytes * (separatePair ? 4 : 2) + readBytes + 256 + 80;
  const limit = Math.min(device.limits?.maxBufferSize || Infinity, device.limits?.maxStorageBufferBindingSize || Infinity);
  if (scratchBytes > MAX_SCRATCH || Math.max(xBytes, yBytes, readBytes) > limit || Math.ceil(N / candidate.columns) > (device.limits?.maxComputeWorkgroupsPerDimension || 65535)) {
    return { kind, key, name, selected: false, reason: "scratch-limit", scratchBytes };
  }
  const owned = [];
  let read, queries, mapped = false;
  try {
    const make = (size, usage) => { const value = scratchBuffer(device, size, usage, `${name}.two-row-tune`); owned.push(value); return value; };
    const globals = make(16, U.UNIFORM | U.COPY_DST), params = make(16, U.UNIFORM | U.COPY_DST), geluParams = make(16, U.UNIFORM | U.COPY_DST);
    const x = make(xBytes, U.STORAGE | U.COPY_DST), zero = make(16, U.STORAGE | U.COPY_DST), part = make(16, U.STORAGE);
    const oldY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST), newY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const gate = separatePair ? make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST) : null;
    const up = separatePair ? make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST) : null;
    const outputs = [oldY, newY, ...(separatePair ? [gate, up] : [])];
    const poisoned = new Float32Array(floats + TAIL.length).fill(NaN); poisoned.set(TAIL, floats);
    device.queue.writeBuffer(globals, 0, new Uint32Array([2, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
    device.queue.writeBuffer(x, 0, inputFor(K));
    device.queue.writeBuffer(zero, 0, new Float32Array(4).fill(NaN));
    for (const output of outputs) device.queue.writeBuffer(output, 0, poisoned);
    const bind = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const plainOp = (weight, output, layout) => ({ N, K, group: bind(layout, [globals, params, x, weight.buf, weight.sbuf, zero, output, part]) });
    const pairOp = (pipes, output) => ({ N, K, group: bind(pipes.layout, [globals, params, x, shape.gate.buf, shape.gate.sbuf, shape.up.buf, shape.up.sbuf, output]) });
    const newOp = kind === "plain" ? plainOp(shape, newY, candidate.layout) : pairOp(candidate, newY);
    const oldOp = kind === "plain" ? plainOp(shape, oldY, (oldTwo || baseline).layout) : oldTwo ? pairOp(oldTwo, oldY) : oldPair ? pairOp(oldPair, oldY) : null;
    const gateOp = separatePair ? plainOp(shape.gate, gate, baseline.layout) : null, upOp = separatePair ? plainOp(shape.up, up, baseline.layout) : null;
    const geluGroup = separatePair ? bind(gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, oldY]) : null;
    const dispatch = (pass, next) => {
      if (next) encodeMatvecTwo(pass, candidate, newOp, 2);
      else if (oldTwo) encodeMatvecTwo(pass, oldTwo, oldOp, 2);
      else if (kind === "plain") encodeMatvec(pass, baseline, oldOp, 2);
      else if (oldPair) encodeMatvecGelu(pass, oldPair, oldOp, 2);
      else {
        encodeMatvec(pass, baseline, gateOp, 2); encodeMatvec(pass, baseline, upOp, 2);
        pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup); pass.dispatchWorkgroups(Math.ceil(floats / 256));
      }
    };
    read = make(readBytes, U.MAP_READ | U.COPY_DST);
    const copyOutputs = (encoder) => {
      encoder.copyBufferToBuffer(oldY, 0, read, 0, yBytes); encoder.copyBufferToBuffer(newY, 0, read, yBytes, yBytes);
      if (separatePair) for (const [i, output] of [gate, up].entries()) encoder.copyBufferToBuffer(output, floats * 4, read, yBytes * 2 + i * TAIL.byteLength, TAIL.byteLength);
    };
    const readOutputs = async () => {
      const withinDeadline = await mapBeforeDeadline(read, readBytes, deadline);
      mapped = true;
      if (!withinDeadline) throw deadlineError();
      const values = new Float32Array(read.getMappedRange(0, readBytes).slice(0)); read.unmap(); mapped = false;
      const stride = yBytes / 4;
      const numericalCheck = compareMatvecTwoOutputs(values.subarray(0, floats), values.subarray(stride, stride + floats));
      const tailChecks = [compareMatvecTwoOutputs(TAIL, values.subarray(floats, stride)), compareMatvecTwoOutputs(TAIL, values.subarray(stride + floats, stride * 2))];
      if (separatePair) for (let i = 0; i < 2; i++) tailChecks.push(compareMatvecTwoOutputs(TAIL, values.subarray(stride * 2 + i * TAIL.length, stride * 2 + (i + 1) * TAIL.length)));
      const outputTailGuard = tailChecks.every((value) => value.ok);
      return { ...numericalCheck, ok: numericalCheck.ok && outputTailGuard, bitExact: numericalCheck.bitExact && outputTailGuard, outputTailGuard, tailChecks };
    };
    const checkEncoder = device.createCommandEncoder();
    const checkPass = checkEncoder.beginComputePass(); dispatch(checkPass, false); dispatch(checkPass, true); checkPass.end();
    copyOutputs(checkEncoder); device.queue.submit([checkEncoder.finish()]);
    const numericalCheck = await readOutputs();
    if (!numericalCheck.ok) return { kind, key, name, baselineKernel, selected: false, numericalCheck, reason: "numerical-guard-failed", scratchBytes };
    if (now() >= deadline) throw deadlineError();
    queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC), encoder = device.createCommandEncoder();
    const warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); warm.end();
    const indices = { baseline: [], candidate: [] };
    let queryIndex = 0;
    for (let sample = 0; sample < SAMPLES; sample++) for (const next of sample % 2 ? [true, false] : [false, true]) {
      if (now() >= deadline) throw deadlineError();
      indices[next ? "candidate" : "baseline"].push(queryIndex);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } });
      for (let repetition = 0; repetition < REPETITIONS; repetition++) dispatch(pass, next);
      pass.end(); queryIndex += 2;
    }
    encoder.resolveQuerySet(queries, 0, queryIndex, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, queryIndex * 8);
    device.queue.submit([encoder.finish()]);
    const withinDeadline = await mapBeforeDeadline(read, queryIndex * 8, deadline);
    mapped = true;
    if (!withinDeadline) throw deadlineError();
    const stamps = new BigInt64Array(read.getMappedRange(0, queryIndex * 8).slice(0)); read.unmap(); mapped = false;
    const times = Object.fromEntries(Object.entries(indices).map(([label, list]) => [label, list.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS)]));
    // Check again after repeated dispatches so a stateful mode or tail write cannot be selected.
    const finalEncoder = device.createCommandEncoder(); copyOutputs(finalEncoder); device.queue.submit([finalEncoder.finish()]);
    const repeatedCheck = await readOutputs();
    if (!repeatedCheck.ok) return { kind, key, name, baselineKernel, selected: false, numericalCheck: repeatedCheck, reason: "numerical-guard-failed", scratchBytes };
    return { kind, key, name, baselineKernel, ...times, ...selectMatvecTwoKernel({ ...times, numericalCheck }), repeatedCheck, scratchBytes };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    queries?.destroy();
  }
}

async function refineSubgroups(device, wgsl, shapes, pipelines, selection, baseline, paired, gelu, deadline, refinement) {
  const { diagnostics } = refinement;
  const needed = Object.fromEntries(["plain", "gelu"].map((kind) => [kind, shapes[kind].filter((shape) => selection[kind].has(shape.key))]));
  const result = (reason) => { diagnostics.reason = reason; };
  const abort = (reason, error) => {
    for (const kind of ["plain", "gelu"]) refinement.selection[kind].clear();
    for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.reason = "aborted-after-refinement-failure"; }
    if (error) diagnostics.error = String(error?.message || error);
    result(reason);
  };
  if (!needed.plain.length && !needed.gelu.length) return result("no-selected-two-row-shapes");
  if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
  try { refinement.pipelines = await checked(device, () => makePipelines(device, wgsl, pipelines.plain || baseline, pipelines.gelu, needed, true)); }
  catch (error) { return abort("pipeline-failed", error); }
  const work = [...needed.gelu.map((shape) => ["gelu", shape]), ...needed.plain.map((shape) => ["plain", shape])];
  for (const [kind, shape] of work) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    let point;
    try { point = await checked(device, () => measureShape(device, kind, shape, refinement.pipelines, baseline, paired, gelu, deadline, pipelines)); }
    catch (error) {
      if (error?.code === "MATVEC_DEADLINE") { diagnostics.truncated = true; return result("deadline"); }
      return abort("calibration-failed", error);
    }
    diagnostics.points.push(point);
    if (point.reason === "numerical-guard-failed") return abort("numerical-guard-failed");
    if (point.reason === "invalid-samples") return abort("timing-failed");
    if (point.selected) refinement.selection[kind].add(shape.key);
  }
  return result(refinement.selection.plain.size || refinement.selection.gelu.size ? "calibrated" : "no-stable-win");
}

/** Calibrate two-row plain and paired paths against the currently effective loaded-model kernels. */
export async function calibrateMatvecTwo(device, wgsl, weights, { matvec, matvecGelu, gelu, sharedStart = Infinity, enabled = true, subgroup32 = false } = {}) {
  const started = now(), selection = { plain: new Set(), gelu: new Set() };
  const diagnostics = { revision: 1, method: "paired-timestamp-query", tokens: 2, samples: SAMPLES, repetitions: REPETITIONS, threshold: THRESHOLD, points: [], truncated: false };
  const subgroups = { pipelines: { plain: null, gelu: null }, selection: { plain: new Set(), gelu: new Set() }, diagnostics: {
    method: "paired-timestamp-query", samples: SAMPLES, repetitions: REPETITIONS, threshold: THRESHOLD, points: [], truncated: false, reason: subgroup32 ? "not-attempted" : "not-fixed32",
  } };
  let pipelines = { plain: null, gelu: null };
  const result = (reason) => {
    for (const kind of ["plain", "gelu"]) if (pipelines[kind] && subgroups.pipelines[kind]) pipelines[kind].subgroup = {
      pipeline: subgroups.pipelines[kind].pipeline, selection: subgroups.selection[kind], enabled: true,
    };
    return { pipelines, selection, subgroups, diagnostics: { ...diagnostics, subgroups: subgroups.diagnostics, reason, elapsedMs: now() - started } };
  };
  const abort = (reason, error) => {
    selection.plain.clear(); selection.gelu.clear();
    for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.reason = "aborted-after-calibration-failure"; }
    if (error) diagnostics.error = String(error?.message || error);
    return result(reason);
  };
  if (!enabled) return result("disabled");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  const baseline = matvec?.pipelines;
  if (!baseline?.pipeline || !baseline.layout || ![true, false].includes(baseline.f16) || baseline.weights !== "q8" || !matvec.selection?.size) return result("missing-baseline-pipeline");
  if (baseline.f16 && !device.features.has("shader-f16")) return result("baseline-precision-unavailable");
  if (baseline.f16) subgroups.diagnostics.reason = "f16-baseline";
  else if (!device.features.has("subgroups")) subgroups.diagnostics.reason = "no-subgroups-feature";
  const shapes = shapesFor(weights, matvec.selection, sharedStart);
  if (!shapes.plain.length && !shapes.gelu.length) return result("no-selected-q8-shapes");
  if (shapes.gelu.some((shape) => !matvecGelu?.selection?.has(shape.key)) && !gelu) return result("missing-gelu-baseline");
  for (const shape of shapes.gelu) if (matvecGelu?.selection?.has(shape.key) && (!matvecGelu.pipelines?.pipeline || !matvecGelu.pipelines.layout || matvecGelu.pipelines.f16 !== baseline.f16)) return result("missing-paired-baseline");
  const deadline = started + DEADLINE_MS;
  try { pipelines = await checked(device, () => makePipelines(device, wgsl, baseline, matvecGelu?.pipelines, shapes)); }
  catch (error) { return abort("pipeline-failed", error); }
  const work = [
    ...shapes.gelu.slice(0, 1).map((shape) => ["gelu", shape]),
    ...shapes.plain.slice(0, 3).map((shape) => ["plain", shape]),
    ...shapes.gelu.slice(1).map((shape) => ["gelu", shape]),
    ...shapes.plain.slice(3).map((shape) => ["plain", shape]),
  ];
  for (const [kind, shape] of work) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    let point;
    try { point = await checked(device, () => measureShape(device, kind, shape, pipelines, baseline, matvecGelu, gelu, deadline)); }
    catch (error) {
      if (error?.code === "MATVEC_DEADLINE") { diagnostics.truncated = true; return result("deadline"); }
      return abort("calibration-failed", error);
    }
    diagnostics.points.push(point);
    if (point.reason === "numerical-guard-failed") return abort("numerical-guard-failed");
    if (point.reason === "invalid-samples") return abort("timing-failed");
    if (point.selected) selection[kind].add(shape.key);
  }
  if (subgroup32 && !baseline.f16 && device.features.has("subgroups")) await refineSubgroups(device, wgsl, shapes, pipelines, selection, baseline, matvecGelu, gelu, deadline, subgroups);
  return result(selection.plain.size || selection.gelu.size ? "calibrated" : "no-stable-win");
}
