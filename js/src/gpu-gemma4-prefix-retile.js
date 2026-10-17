// Compact prefix rows may use a smaller tile while retaining the original
// kernel family, tile precision and logical split-K partition.
import { bufferUsage, inErrorScopes, mmSplits, rowsPerThread, scratchBuffer } from "./gpu.js";
import { mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";
import { selectedMatmulKernel } from "./gpu-tuning.js";
import { gemma4K64Eligible } from "./gpu-gemma4-k64.js";
import { gemma4MlpEligible } from "./gpu-gemma4-mlp.js";

const SAMPLES = 3;
const REPETITIONS = 4;
const DEADLINE_MS = 750;
const MAX_SCRATCH = 32 * 1024 * 1024;
const SENTINEL = 12345.25;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

/** The surrogate changes only matmul globals, never attention or gather globals. */
export function gemma4PrefixRetileConfig(plan, pipes, kernel = "generic") {
  const P = plan?.logicalRows, C = plan?.staticRows, D = plan?.activeRows;
  if (![P, C, D].every(Number.isSafeInteger) || P < 2 || P > 256 || C < 1 || D !== P - C || D < 1 || D > 64) return null;
  if (!["generic", "wide"].includes(kernel) || pipes?.groups !== 1 || typeof pipes.f16 !== "boolean" || !Number.isSafeInteger(pipes.splitTarget) || pipes.splitTarget < 1) return null;
  const originalRows = rowsPerThread(P, pipes);
  const originalRow56 = originalRows === 4 && !pipes.f16 && Math.ceil(P / 56) === Math.ceil(P / 64) && !!pipes.mm56;
  const rows = D <= 32 ? 2 : 4;
  const row56 = rows === 4 && kernel === "generic" && !pipes.f16 && D <= 56 && !!pipes.mm56;
  const originalBM = originalRow56 ? 56 : 16 * originalRows, physicalBM = row56 ? 56 : 16 * rows;
  if (Math.ceil(D / physicalBM) * physicalBM >= Math.ceil(D / originalBM) * originalBM) return null;
  if (!(row56 ? pipes.mm56 : pipes.mm?.[rows]) || !pipes.reduce?.[rows] || !(originalRow56 ? pipes.mm56 : pipes.mm?.[originalRows]) || !pipes.reduce?.[originalRows]) return null;
  const splitTokens = Math.ceil(P / (16 * originalRows)) * 16 * rows;
  return { kernel, rows, row56, splitTokens };
}

export function compareGemma4PrefixRetileOutputs(baseline, candidate) {
  const lengthMismatch = baseline.length !== candidate.length;
  const a = new Uint32Array(baseline.buffer, baseline.byteOffset, baseline.length);
  const b = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let bitMismatches = 0, nonFinite = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (!Number.isFinite(baseline[index]) || !Number.isFinite(candidate[index])) nonFinite++;
    if (a[index] !== b[index]) bitMismatches++;
  }
  return { ok: !lengthMismatch && bitMismatches === 0 && nonFinite === 0, bitExact: true, length: candidate.length, expectedLength: baseline.length, lengthMismatch, bitMismatches, nonFinite };
}

function shapesFor(weights, plan) {
  if (!(weights?.tensors instanceof Map)) return [];
  const ranks = { down: 0, q: 1, o: 2, gate: 3, up: 3, k: 4, v: 4, ple_gate: 5, ple_out: 5, proj: 5 };
  const layerLimit = plan.layerWidths?.length;
  const shapes = [];
  for (const [name, entry] of weights.tensors) {
    const match = /^(?:l\.(\d+)\.(down|q|o|gate|up|k|v|ple_gate|ple_out)|ple\.(\d+)\.(proj))$/.exec(name);
    if (!match || entry.info?.dtype !== "q8" || !Array.isArray(entry.info.shape) || entry.info.shape.length !== 2 || !entry.buf || !entry.sbuf) continue;
    const layer = Number(match[1] ?? match[3]), suffix = match[2] || match[4];
    if (layerLimit && layer >= layerLimit) continue;
    const [N, K] = entry.info.shape.map(Number);
    if (!Number.isSafeInteger(N) || !Number.isSafeInteger(K) || N < 1 || K < 32 || K % 32) continue;
    shapes.push({ ...entry, N, K, key: `${N}:${K}`, name, layer, rank: ranks[suffix] });
  }
  shapes.sort((a, b) => a.rank - b.rank || a.layer - b.layer || a.N * a.K - b.N * b.K);
  const seen = new Set();
  return shapes.filter(({ key }) => { if (seen.has(key)) return false; seen.add(key); return true; });
}

function inputFor(tokens, { N, K }) {
  let state = (0x50525434 ^ tokens ^ N ^ Math.imul(K, 31)) >>> 0;
  const values = new Float32Array(tokens * K);
  for (let index = 0; index < values.length; index++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    values[index] = Math.fround((((value ^ (value >>> 14)) >>> 0) / 0x100000000 - 0.5) * 0.08) || 0.00390625;
  }
  return values;
}

function deadlineError() {
  return Object.assign(new Error("Gemma prefix retile calibration deadline exceeded"), { code: "GEMMA4_PREFIX_RETILE_DEADLINE" });
}

async function measureShape(device, shape, plan, pipes, config, deadline) {
  const { N, K } = shape, P = plan.logicalRows, D = plan.activeRows;
  const originalRows = rowsPerThread(P, pipes);
  const originalRow56 = originalRows === 4 && !pipes.f16 && Math.ceil(P / 56) === Math.ceil(P / 64) && !!pipes.mm56;
  const splits = mmSplits(P, N, K, pipes.splitTarget, 16 * originalRows, 64);
  const base = { key: shape.key, name: shape.name, N, K, tokens: D, logicalRows: P, originalRows, originalRow56, f16: pipes.f16, splits, ...config, selected: false };
  if (splits !== mmSplits(config.splitTokens, N, K, pipes.splitTarget, 16 * config.rows, 64)) return { ...base, reason: "split-policy-mismatch" };
  const outputBytes = (D * N + 4) * 4, partLength = splits > 1 ? splits * D * N : 0, partBytes = (partLength + 4) * 4;
  const bytes = { input: D * K * 4, output: outputBytes, part: partBytes, read: Math.max(256, 2 * (outputBytes + partBytes)), resolved: 256 };
  const scratchBytes = bytes.input + 2 * outputBytes + 2 * partBytes + bytes.read + bytes.resolved + 64;
  const maxBuffer = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  if (!Number.isSafeInteger(scratchBytes) || scratchBytes > MAX_SCRATCH || Math.max(...Object.values(bytes)) > maxBuffer) return { ...base, reason: "scratch-limit", scratchBytes };
  const owned = [];
  let queries, read, mapped = false;
  try {
    const U = bufferUsage();
    const make = (size, usage, label) => (owned.push(scratchBuffer(device, size, usage, `${shape.name}.prefix-retile.${label}`)), owned.at(-1));
    const baselineGlobals = make(16, U.UNIFORM | U.COPY_DST, "baseline-globals");
    const candidateGlobals = make(16, U.UNIFORM | U.COPY_DST, "candidate-globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params"), zero = make(16, U.STORAGE, "zero");
    const input = make(bytes.input, U.STORAGE | U.COPY_DST, "input");
    const guarded = (length) => { const values = new Float32Array(length + 4).fill(NaN); values.fill(SENTINEL, length); return values; };
    const baselineY = make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-y"), candidateY = make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "candidate-y");
    const baselinePart = make(partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-part"), candidatePart = make(partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "candidate-part");
    read = make(bytes.read, U.MAP_READ | U.COPY_DST, "read");
    const resolved = make(bytes.resolved, U.QUERY_RESOLVE | U.COPY_SRC, "resolved");
    device.queue.writeBuffer(baselineGlobals, 0, new Uint32Array([D, P, P, 0]));
    device.queue.writeBuffer(candidateGlobals, 0, new Uint32Array([D, config.splitTokens, P, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(input, 0, inputFor(D, shape));
    const outputPoison = guarded(D * N), partPoison = guarded(partLength);
    for (const buffer of [baselineY, candidateY]) device.queue.writeBuffer(buffer, 0, outputPoison);
    for (const buffer of [baselinePart, candidatePart]) device.queue.writeBuffer(buffer, 0, partPoison);
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const operation = (globals, output, part) => ({
      group: group(pipes.layout, [globals, params, input, shape.buf, shape.sbuf, zero, output, part]),
      reduce: group(pipes.reduceLayout, [globals, params, part, zero, output]),
    });
    const baselineOp = operation(baselineGlobals, baselineY, baselinePart), candidateOp = operation(candidateGlobals, candidateY, candidatePart);
    const dispatch = (pass, candidate) => {
      const rows = candidate ? config.rows : originalRows, row56 = candidate ? config.row56 : originalRow56;
      const op = candidate ? candidateOp : baselineOp;
      pass.setPipeline(row56 ? pipes.mm56 : pipes.mm[rows]); pass.setBindGroup(0, op.group);
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(D / (row56 ? 56 : 16 * rows)), splits);
      if (splits > 1) {
        const work = Math.ceil(D * N / 256);
        pass.setPipeline(pipes.reduce[rows]); pass.setBindGroup(0, op.reduce);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    if (now() >= deadline) throw deadlineError();
    const checkEncoder = device.createCommandEncoder(), checkPass = checkEncoder.beginComputePass();
    dispatch(checkPass, false); dispatch(checkPass, true); checkPass.end();
    const stride = outputBytes + partBytes;
    for (const [index, output, part] of [[0, baselineY, baselinePart], [1, candidateY, candidatePart]]) {
      checkEncoder.copyBufferToBuffer(output, 0, read, index * stride, outputBytes);
      checkEncoder.copyBufferToBuffer(part, 0, read, index * stride + outputBytes, partBytes);
    }
    device.queue.submit([checkEncoder.finish()]);
    const withinParityDeadline = await mapBeforeDeadline(read, 2 * stride, deadline); mapped = true;
    if (!withinParityDeadline) throw deadlineError();
    const values = new Float32Array(read.getMappedRange(0, 2 * stride).slice(0)); read.unmap(); mapped = false;
    const outputCheck = compareGemma4PrefixRetileOutputs(values.subarray(0, D * N), values.subarray(stride / 4, stride / 4 + D * N));
    const partialCheck = partLength ? compareGemma4PrefixRetileOutputs(values.subarray(outputBytes / 4, outputBytes / 4 + partLength), values.subarray((stride + outputBytes) / 4, (stride + outputBytes) / 4 + partLength)) : null;
    const tails = [(outputBytes - 16) / 4, (stride - 16) / 4, (stride + outputBytes - 16) / 4, (2 * stride - 16) / 4];
    const tailGuard = tails.every((start) => values.subarray(start, start + 4).every((value) => value === SENTINEL));
    const numericalCheck = { ok: outputCheck.ok && (!partialCheck || partialCheck.ok) && tailGuard, outputCheck, partialCheck, tailGuard, bitExact: true };
    if (!numericalCheck.ok) return { ...base, numericalCheck, scratchBytes, reason: "numerical-guard-failed" };
    const queryCount = SAMPLES * 4;
    queries = device.createQuerySet({ type: "timestamp", count: queryCount });
    const encoder = device.createCommandEncoder(), warmup = encoder.beginComputePass();
    dispatch(warmup, false); dispatch(warmup, true); warmup.end();
    const indices = { baseline: [], candidate: [] };
    let index = 0;
    for (let sample = 0; sample < SAMPLES; sample++) {
      if (now() >= deadline) throw deadlineError();
      for (const candidate of sample % 2 ? [true, false] : [false, true]) {
        indices[candidate ? "candidate" : "baseline"].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(pass, candidate);
        pass.end(); index += 2;
      }
    }
    encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8); device.queue.submit([encoder.finish()]);
    const withinTimingDeadline = await mapBeforeDeadline(read, queryCount * 8, deadline); mapped = true;
    if (!withinTimingDeadline) throw deadlineError();
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0)); read.unmap(); mapped = false;
    const times = Object.fromEntries(Object.entries(indices).map(([label, values]) => [label, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS)]));
    const choice = selectMatvecKernel({ ...times, numericalCheck });
    return { ...base, numericalCheck, ...times, scratchBytes, selected: choice.selected, stable: choice.stable, benefit: choice.benefit, baselineMs: choice.genericMs, candidateMs: choice.wideMs,
      reason: choice.reason === "stable-matvec-win" ? "stable-prefix-retile-win" : choice.reason };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    queries?.destroy();
  }
}

export async function calibrateGemma4PrefixRetile(device, weights, tuning, { plan, sourcePipelines, k64 = null, mlp = null, enabled = true } = {}) {
  const started = now(), selection = new Map();
  const diagnostics = { revision: 1, method: "baseline", logicalRows: plan?.logicalRows ?? null, staticRows: plan?.staticRows ?? null, activeRows: plan?.activeRows ?? null,
    samples: SAMPLES, repetitions: REPETITIONS, threshold: 0.05, points: [], truncated: false };
  const result = (reason) => ({ selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  const abort = (reason) => {
    selection.clear();
    for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.reason = "aborted-after-calibration-failure"; }
    return result(reason);
  };
  if (!enabled) return result("disabled");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  if (!tuning?.pipelines?.generic || !sourcePipelines?.generic) return result("missing-baseline-pipeline");
  if (tuning.diagnostics?.method === "override") return result("kernel-override");
  const candidates = shapesFor(weights, plan || {}).flatMap((shape) => {
    if ((gemma4K64Eligible(plan?.logicalRows) && k64?.selection?.has(shape.key)) || (gemma4MlpEligible(plan?.logicalRows) && mlp?.selection?.has(shape.key))) return [];
    const selected = selectedMatmulKernel(tuning.selection, shape.N, shape.K, plan?.logicalRows);
    const kernel = sourcePipelines[selected] ? selected : "generic", pipes = sourcePipelines[kernel], original = tuning.pipelines[kernel];
    if (!original || pipes.f16 !== original.f16 || pipes.groups !== original.groups || pipes.splitTarget !== original.splitTarget) return [];
    const config = gemma4PrefixRetileConfig(plan, pipes, kernel);
    return config ? [{ shape, pipes, config }] : [];
  }).slice(0, 8);
  if (!candidates.length) return result("no-eligible-shapes");
  diagnostics.method = "paired-timestamp-query";
  const deadline = started + DEADLINE_MS;
  for (const { shape, pipes, config } of candidates) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    try {
      const point = await inErrorScopes(device, () => measureShape(device, shape, plan, pipes, config, deadline));
      diagnostics.points.push(point);
      if (point.selected) selection.set(shape.key, config);
      if (["numerical-guard-failed", "split-policy-mismatch", "insufficient-samples"].includes(point.reason)) return abort(point.reason === "insufficient-samples" ? "timing-failed" : point.reason);
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (["MATVEC_DEADLINE", "GEMMA4_PREFIX_RETILE_DEADLINE"].includes(error?.code)) { diagnostics.truncated = true; return result("deadline"); }
      return abort("calibration-failed");
    }
  }
  return result(selection.size ? "calibrated" : "no-stable-win");
}
