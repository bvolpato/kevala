// F32 Q8 R2 prefix projections may halve physical BN while retaining virtual BN64 splits.
import { bufferUsage, matmulLayout, mmSplits, pipeline, scratchBuffer, withGpuErrors } from "./gpu.js";
import { mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";

const TARGET = 256, MAX_SCRATCH = 32 * 1024 * 1024, DEADLINE_MS = 1000;
const SAMPLES = 3, REPETITIONS = 4, SENTINEL = 12345.25;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());
const positive = (value) => Number.isSafeInteger(value) && value > 0;

export function compareGemma4Bn32Outputs(baseline, candidate) {
  const a = new Uint32Array(baseline.buffer, baseline.byteOffset, baseline.length);
  const b = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let bitMismatches = 0, nonFinite = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (!Number.isFinite(baseline[i]) || !Number.isFinite(candidate[i])) nonFinite++;
    if (a[i] !== b[i]) bitMismatches++;
  }
  return { ok: a.length === b.length && bitMismatches === 0 && nonFinite === 0, bitExact: true,
    length: b.length, expectedLength: a.length, bitMismatches, nonFinite, signedZeroChecked: true };
}

function shapesFor(weights, plan, retile) {
  const ranks = { gate: 0, up: 0, down: 1, o: 2, q: 3, k: 4, v: 4, proj: 5, ple_gate: 5, ple_out: 5 };
  const shapes = [], seen = new Set();
  for (const [name, entry] of weights?.tensors || []) {
    const match = /^(?:l\.(\d+)\.(gate|up|down|o|q|k|v|ple_gate|ple_out)|ple\.(\d+)\.(proj))$/.exec(name);
    if (!match || entry.info?.dtype !== "q8" || entry.info.block != null && entry.info.block !== 32 || !entry.buf || !entry.sbuf || entry.info.shape?.length !== 2) continue;
    const layer = Number(match[1] ?? match[3]), suffix = match[2] || match[4];
    if (layer >= plan.layerWidths.length) continue;
    const [N, K] = entry.info.shape.map(Number), key = `${N}:${K}`, config = retile.selection.get(key);
    if (!positive(N) || !positive(K) || K % 32 || config?.kernel !== "generic" || config.rows !== 2 || config.row56 !== false || !positive(config.splitTokens)) continue;
    shapes.push({ ...entry, N, K, key, name, config, rank: ranks[suffix], layer });
  }
  shapes.sort((a, b) => a.rank - b.rank || a.layer - b.layer || a.N * a.K - b.N * b.K);
  return shapes.filter(({ key }) => { if (seen.has(key)) return false; seen.add(key); return true; }).slice(0, 8);
}

function inputFor(tokens, { N, K }) {
  let state = (0x424e3332 ^ tokens ^ N ^ Math.imul(K, 31)) >>> 0;
  return Float32Array.from({ length: tokens * K }, () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return Math.fround((((value ^ (value >>> 14)) >>> 0) / 0x100000000 - 0.5) * 0.08) || 0.00390625;
  });
}

function deadlineError() {
  return Object.assign(new Error("Gemma BN32 calibration deadline exceeded"), { code: "GEMMA4_BN32_DEADLINE" });
}

async function measure(device, shape, plan, baseline, candidate, deadline) {
  const { N, K, config } = shape, T = plan.activeRows;
  const splits = mmSplits(config.splitTokens, N, K, TARGET, 32, 64), length = T * N;
  const partialLength = splits > 1 ? splits * length : 0;
  const yBytes = (length + 4) * 4, partBytes = (partialLength + 4) * 4, stride = yBytes + partBytes;
  const bytes = { x: T * K * 4, y: yBytes, part: partBytes, read: Math.max(256, 2 * stride), resolved: 256 };
  const gpuBytes = bytes.x + 2 * stride + bytes.read + bytes.resolved + 48;
  // Include fixture arrays, mapped readback, and queued upload snapshots in the cap.
  const scratchBytes = gpuBytes + 2 * bytes.x + 3 * stride + bytes.read;
  const base = { key: shape.key, name: shape.name, N, K, tokens: T, logicalRows: plan.logicalRows,
    splitTokens: config.splitTokens, splits, rows: 2, baselineBN: 64, candidateBN: 32, f16: false, selected: false, scratchBytes };
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (!Number.isSafeInteger(scratchBytes) || scratchBytes > MAX_SCRATCH || Math.max(...Object.values(bytes)) > limit) return { ...base, reason: "scratch-limit" };
  const owned = [];
  let queries, read, mapped = false;
  try {
    const U = bufferUsage();
    const make = (size, usage, label) => (owned.push(scratchBuffer(device, size, usage, `${shape.name}.bn32.${label}`)), owned.at(-1));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const zero = make(16, U.STORAGE, "zero"), x = make(bytes.x, U.STORAGE | U.COPY_DST, "X");
    const outputs = [0, 1].map(() => make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y"));
    const parts = [0, 1].map(() => make(partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "PART"));
    read = make(bytes.read, U.MAP_READ | U.COPY_DST, "read");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, config.splitTokens, plan.logicalRows, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(x, 0, inputFor(T, shape));
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const ops = outputs.map((output, index) => ({
      group: group(baseline.layout, [globals, params, x, shape.buf, shape.sbuf, zero, output, parts[index]]),
      reduce: group(baseline.reduceLayout, [globals, params, parts[index], zero, output]),
    }));
    const dispatch = (pass, narrow) => {
      const index = Number(narrow);
      pass.setPipeline(narrow ? candidate : baseline.mm[2]); pass.setBindGroup(0, ops[index].group);
      pass.dispatchWorkgroups(Math.ceil(N / (narrow ? 32 : 64)), Math.ceil(T / 32), splits);
      if (splits > 1) {
        const work = Math.ceil(length / 256);
        pass.setPipeline(baseline.reduce[2]); pass.setBindGroup(0, ops[index].reduce);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const poison = (count) => { const values = new Float32Array(count + 4).fill(NaN); values.fill(SENTINEL, count); return values; };
    const yInitial = poison(length), partInitial = poison(partialLength);
    const verify = async () => {
      if (now() >= deadline) throw deadlineError();
      for (let i = 0; i < 2; i++) { device.queue.writeBuffer(outputs[i], 0, yInitial); device.queue.writeBuffer(parts[i], 0, partInitial); }
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      dispatch(pass, false); dispatch(pass, true); pass.end();
      for (let i = 0; i < 2; i++) { encoder.copyBufferToBuffer(outputs[i], 0, read, i * stride, yBytes); encoder.copyBufferToBuffer(parts[i], 0, read, i * stride + yBytes, partBytes); }
      device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 2 * stride, deadline); mapped = true;
      if (!within) throw deadlineError();
      try {
        const values = new Float32Array(read.getMappedRange(0, 2 * stride));
        const output = compareGemma4Bn32Outputs(values.subarray(0, length), values.subarray(stride / 4, stride / 4 + length));
        const partial = partialLength ? compareGemma4Bn32Outputs(values.subarray(yBytes / 4, yBytes / 4 + partialLength), values.subarray((stride + yBytes) / 4, (stride + yBytes) / 4 + partialLength)) : null;
        const tails = [length, yBytes / 4 + partialLength, stride / 4 + length, (stride + yBytes) / 4 + partialLength];
        const tailGuard = tails.every((at) => values.subarray(at, at + 4).every((value) => value === SENTINEL));
        return { ok: output.ok && (!partial || partial.ok) && tailGuard, bitExact: true, output, partial, tailGuard };
      } finally { read.unmap(); mapped = false; }
    };
    const numericalCheck = await verify();
    if (!numericalCheck.ok) return { ...base, numericalCheck, reason: "numerical-guard-failed" };
    queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
    const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass();
    dispatch(warm, false); dispatch(warm, true); warm.end();
    const indices = { baseline: [], candidate: [] };
    let at = 0;
    for (let sample = 0; sample < SAMPLES; sample++) for (const narrow of sample % 2 ? [true, false] : [false, true]) {
      if (now() >= deadline) throw deadlineError();
      indices[narrow ? "candidate" : "baseline"].push(at);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
      for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(pass, narrow);
      pass.end(); at += 2;
    }
    encoder.resolveQuerySet(queries, 0, SAMPLES * 4, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, SAMPLES * 32); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, SAMPLES * 32, deadline); mapped = true;
    if (!within) throw deadlineError();
    let times;
    try {
      const stamps = new BigInt64Array(read.getMappedRange(0, SAMPLES * 32));
      times = Object.fromEntries(Object.entries(indices).map(([label, values]) => [label, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS)]));
    } finally { read.unmap(); mapped = false; }
    const afterTimingCheck = await verify();
    if (!afterTimingCheck.ok) return { ...base, numericalCheck, afterTimingCheck, ...times, reason: "numerical-guard-failed" };
    const choice = selectMatvecKernel({ ...times, numericalCheck: afterTimingCheck });
    return { ...base, numericalCheck, afterTimingCheck, ...times, selected: choice.selected, stable: choice.stable,
      benefit: choice.benefit, baselineMs: choice.genericMs, candidateMs: choice.wideMs,
      reason: choice.reason === "stable-matvec-win" ? "stable-bn32-win" : choice.reason };
  } finally {
    if (mapped) read?.unmap();
    queries?.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

export async function calibrateGemma4Bn32(device, source, weights, { plan, retile, sourcePipelines, enabled = true } = {}) {
  const started = now(), selection = new Set();
  let candidate = null;
  const diagnostics = { revision: 1, method: "baseline", logicalRows: plan?.logicalRows ?? null, activeRows: plan?.activeRows ?? null,
    rows: 2, f16: false, baselineBN: 64, candidateBN: 32, samples: SAMPLES, repetitions: REPETITIONS, threshold: 0.05, points: [], truncated: false };
  const result = (reason) => ({ pipeline: candidate, selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  const abort = (reason) => {
    selection.clear();
    for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.reason = "aborted-after-calibration-failure"; }
    return result(reason);
  };
  if (!enabled) return result("disabled");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  if (!positive(plan?.activeRows) || plan.activeRows > 32 || !positive(plan.staticRows) || !positive(plan.logicalRows) || plan.logicalRows > 256 || plan.logicalRows !== plan.staticRows + plan.activeRows || !plan.layerWidths?.length) return result("unsupported-prefix-plan");
  const baseline = sourcePipelines?.generic;
  if (!baseline?.mm?.[2] || !baseline.reduce?.[2] || !baseline.layout || !baseline.reduceLayout) return result("missing-baseline-pipeline");
  if (baseline.f16 !== false || baseline.groups !== 1 || baseline.splitTarget !== TARGET) return result("unsupported-baseline");
  if (!(retile?.selection instanceof Map)) return result("no-eligible-shapes");
  const shapes = shapesFor(weights, plan, retile);
  if (!shapes.length) return result("no-eligible-shapes");
  diagnostics.method = "paired-timestamp-query";
  // GPU errors take precedence over an overlapping deadline, so prior selections cannot survive one.
  const scoped = (work) => device.pushErrorScope ? withGpuErrors({ device }, work, "calibrating Gemma BN32") : work();
  try {
    candidate = await scoped(() => pipeline(device, source("matmul_bn32_gemma_source", { weights: "q8", f16: false, rows: 2, row56: false, groups: 1, splitTarget: TARGET }),
      "gemma4_bn32_r2", device.createPipelineLayout({ bindGroupLayouts: [matmulLayout(device)] })));
  } catch (error) { diagnostics.error = String(error?.message || error); return abort("pipeline-failed"); }
  const deadline = started + DEADLINE_MS;
  for (const shape of shapes) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    try {
      const point = await scoped(() => measure(device, shape, plan, baseline, candidate, deadline));
      diagnostics.points.push(point);
      if (point.selected) selection.add(shape.key);
      if (["numerical-guard-failed", "insufficient-samples"].includes(point.reason)) return abort(point.reason === "insufficient-samples" ? "timing-failed" : point.reason);
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (["MATVEC_DEADLINE", "GEMMA4_BN32_DEADLINE"].includes(error?.code)) { diagnostics.truncated = true; return result("deadline"); }
      return abort("calibration-failed");
    }
  }
  return result(selection.size ? "calibrated" : "no-stable-win");
}
