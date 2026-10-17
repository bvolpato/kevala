// Fast prefix gate/up/GELU fusion is calibrated against already selected BN32 projections.
import { bufferUsage, matmulLayout, mmSplits, pipeline, reduceLayout, scratchBuffer, withGpuErrors } from "./gpu.js";
import { mapBeforeDeadline, selectMatvecKernel } from "./gpu-matvec.js";
import { compareGemma4Bn32Outputs } from "./gpu-gemma4-bn32.js";

const TARGET = 256, MAX_SCRATCH = 32 * 1024 * 1024, DEADLINE_MS = 750;
const SAMPLES = 3, REPETITIONS = 4, SENTINEL = 12345.25;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());
const positive = (n) => Number.isSafeInteger(n) && n > 0;

function snapshotLayout(tokens, N, splits) {
  const length = tokens * N, capacity = 32 * N, partialLength = splits > 1 ? splits * length : 0;
  const y = capacity + 4, part = partialLength + 4, pair = 2 * partialLength + 4;
  return { length, capacity, partialLength, y, part, pair, floats: 4 * y + 2 * part + pair };
}

export function comparePrefixGeluSnapshot(values, { tokens, N, splits }) {
  const s = snapshotLayout(tokens, N, splits);
  if (values.length !== s.floats) return { ok: false, reason: "snapshot-length-mismatch" };
  const ys = Array.from({ length: 4 }, (_, i) => values.subarray(i * s.y, (i + 1) * s.y));
  const at = 4 * s.y;
  const gate = values.subarray(at, at + s.part), up = values.subarray(at + s.part, at + 2 * s.part), pair = values.subarray(at + 2 * s.part);
  const activation = compareGemma4Bn32Outputs(ys[2].subarray(0, s.length), ys[3].subarray(0, s.length));
  const gatePartials = compareGemma4Bn32Outputs(gate.subarray(0, s.partialLength), pair.subarray(0, s.partialLength));
  const upPartials = compareGemma4Bn32Outputs(up.subarray(0, s.partialLength), pair.subarray(s.partialLength, 2 * s.partialLength));
  const finiteProjections = ys.slice(0, 2).every((y) => y.subarray(0, s.length).every(Number.isFinite));
  const paddingGuard = ys.every((y) => y.subarray(s.length).every((value) => value === SENTINEL));
  const partialTailGuard = gate.subarray(s.partialLength).every((value) => value === SENTINEL)
    && up.subarray(s.partialLength).every((value) => value === SENTINEL)
    && pair.subarray(2 * s.partialLength).every((value) => value === SENTINEL);
  return { ok: activation.ok && gatePartials.ok && upPartials.ok && finiteProjections && paddingGuard && partialTailGuard,
    bitExact: true, activation, gatePartials, upPartials, finiteProjections, paddingGuard, partialTailGuard };
}

function shapesFor(weights, plan, retile, bn32) {
  const shapes = [], seen = new Set();
  for (const [name, gate] of weights?.tensors || []) {
    const match = /^l\.(\d+)\.gate$/.exec(name), up = weights.tensors.get(name.replace(/\.gate$/, ".up"));
    if (!match || Number(match[1]) >= plan.layerWidths.length || gate.info?.dtype !== "q8" || up?.info?.dtype !== "q8"
      || gate.info.block !== 32 || up.info.block !== 32 || gate.info.shape?.length !== 2 || up.info.shape?.length !== 2 || !gate.buf || !gate.sbuf || !up.buf || !up.sbuf) continue;
    const [N, K] = gate.info.shape.map(Number), key = `${N}:${K}`, config = retile.selection.get(key);
    if (!positive(N) || !positive(K) || K % 32 || Number(up.info.shape[0]) !== N || Number(up.info.shape[1]) !== K || seen.has(key) || !bn32.selection.has(key)
      || config?.kernel !== "generic" || config.rows !== 2 || config.row56 !== false || !positive(config.splitTokens)) continue;
    seen.add(key); shapes.push({ N, K, key, name, gate, up, config });
  }
  return shapes.sort((a, b) => Number(a.name.split(".")[1]) - Number(b.name.split(".")[1]) || a.N * a.K - b.N * b.K).slice(0, 2);
}

function deadlineError() { return Object.assign(new Error("Gemma prefix GELU calibration deadline exceeded"), { code: "GEMMA4_PREFIX_GELU_DEADLINE" }); }

async function compile(device, source) {
  const layoutFor = (label, types) => device.createBindGroupLayout({ label, entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const layout = layoutFor("Gemma prefix GELU", ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"]);
  const reductionLayout = layoutFor("Gemma prefix GELU reduce", ["uniform", "uniform", "read-only-storage", "storage"]);
  const baselineLayout = matmulLayout(device), baselineReduceLayout = reduceLayout(device);
  const pl = (l) => device.createPipelineLayout({ bindGroupLayouts: [l] });
  const spec = { weights: "q8", f16: false, rows: 2, row56: false, groups: 1, splitTarget: TARGET };
  const builds = await Promise.allSettled([
    pipeline(device, source("matmul_gelu_bn32", spec), "gemma4_prefix_gelu_bn32", pl(layout)),
    pipeline(device, source("reduce_gelu", spec), "gemma4_prefix_gelu_reduce", pl(reductionLayout)),
    pipeline(device, source("reduce_gemma_source", spec), "gemma4_prefix_gelu_baseline_reduce", pl(baselineReduceLayout)),
  ]);
  const failure = builds.find((build) => build.status === "rejected");
  if (failure) throw failure.reason;
  const [matmul, reduce, baselineReduce] = builds.map((build) => build.value);
  return { pipelines: { matmul, reduce, layout, reduceLayout: reductionLayout }, baselineLayout, baselineReduceLayout, baselineReduce };
}

async function measure(device, shape, plan, bundle, bn32, gelu, deadline) {
  const { N, K, config } = shape, tokens = plan.activeRows, splits = mmSplits(config.splitTokens, N, K, TARGET, 32, 64);
  const s = snapshotLayout(tokens, N, splits), bytes = { x: tokens * K * 4, y: s.y * 4, part: s.part * 4, pair: s.pair * 4, read: Math.max(256, s.floats * 4) };
  const gpuBytes = bytes.x + s.floats * 4 + bytes.read + 256 + 80;
  const hostFixtures = bytes.x + bytes.y + bytes.part + bytes.pair;
  const queuedUploads = bytes.x + s.floats * 4 + 64;
  const scratchBytes = gpuBytes + hostFixtures + bytes.read + queuedUploads;
  const base = { key: shape.key, name: shape.name, N, K, tokens, logicalRows: plan.logicalRows, splitTokens: config.splitTokens, splits,
    f16: false, rows: 2, physicalBN: 32, virtualBN: 64, scratchBytes, selected: false };
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (!Number.isSafeInteger(scratchBytes) || scratchBytes > MAX_SCRATCH || Math.max(...Object.values(bytes)) > limit) return { ...base, reason: "scratch-limit" };
  const owned = [];
  let queries, read, mapped = false;
  try {
    const U = bufferUsage(), pipes = bundle.pipelines;
    const make = (size, usage, label) => (owned.push(scratchBuffer(device, size, usage, `${shape.name}.prefix-gelu.${label}`)), owned.at(-1));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const pairParams = make(16, U.UNIFORM | U.COPY_DST, "pair-params"), geluParams = make(16, U.UNIFORM | U.COPY_DST, "gelu-params");
    const zero = make(16, U.STORAGE, "zero"), x = make(bytes.x, U.STORAGE | U.COPY_DST, "X");
    const outputs = Array.from({ length: 4 }, () => make(bytes.y, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y"));
    const parts = [0, 1].map(() => make(bytes.part, U.STORAGE | U.COPY_SRC | U.COPY_DST, "PART"));
    const pairPart = make(bytes.pair, U.STORAGE | U.COPY_SRC | U.COPY_DST, "pair-PART");
    read = make(bytes.read, U.MAP_READ | U.COPY_DST, "read"); const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    device.queue.writeBuffer(globals, 0, new Uint32Array([tokens, config.splitTokens, plan.logicalRows, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(pairParams, 0, new Uint32Array([N, K, config.splitTokens, 0]));
    device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
    device.queue.writeBuffer(x, 0, Float32Array.from({ length: tokens * K }, (_, i) => 0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3)));
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const baseline = [shape.gate, shape.up].map((weight, i) => ({ group: group(bundle.baselineLayout, [globals, params, x, weight.buf, weight.sbuf, zero, outputs[i], parts[i]]),
      reduce: group(bundle.baselineReduceLayout, [globals, params, parts[i], zero, outputs[i]]) }));
    const geluGroup = group(gelu.getBindGroupLayout(0), [globals, geluParams, outputs[0], outputs[1], outputs[2]]);
    const pairGroup = group(pipes.layout, [globals, pairParams, x, shape.gate.buf, shape.gate.sbuf, shape.up.buf, shape.up.sbuf, outputs[3], pairPart]);
    const pairReduce = group(pipes.reduceLayout, [globals, pairParams, pairPart, outputs[3]]), work = Math.ceil(s.length / 256);
    const dispatch = (pass, fused) => {
      if (fused) {
        pass.setPipeline(pipes.matmul); pass.setBindGroup(0, pairGroup); pass.dispatchWorkgroups(Math.ceil(N / 32), 1, splits);
        if (splits > 1) { pass.setPipeline(pipes.reduce); pass.setBindGroup(0, pairReduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535)); }
      } else {
        for (const op of baseline) {
          pass.setPipeline(bn32.pipeline); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(Math.ceil(N / 32), 1, splits);
          if (splits > 1) { pass.setPipeline(bundle.baselineReduce); pass.setBindGroup(0, op.reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535)); }
        }
        pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const poison = (capacity, active) => { const values = new Float32Array(capacity).fill(SENTINEL); values.fill(NaN, 0, active); return values; };
    const yInitial = poison(s.y, s.length), partInitial = poison(s.part, s.partialLength), pairInitial = poison(s.pair, 2 * s.partialLength);
    const verify = async () => {
      if (now() >= deadline) throw deadlineError();
      for (const output of outputs) device.queue.writeBuffer(output, 0, yInitial);
      for (const part of parts) device.queue.writeBuffer(part, 0, partInitial);
      device.queue.writeBuffer(pairPart, 0, pairInitial);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end();
      let offset = 0;
      for (const [buffer, size] of [...outputs.map((b) => [b, bytes.y]), ...parts.map((b) => [b, bytes.part]), [pairPart, bytes.pair]]) { encoder.copyBufferToBuffer(buffer, 0, read, offset, size); offset += size; }
      device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, s.floats * 4, deadline); mapped = true;
      if (!within) throw deadlineError();
      try { return comparePrefixGeluSnapshot(new Float32Array(read.getMappedRange(0, s.floats * 4)), { tokens, N, splits }); }
      finally { read.unmap(); mapped = false; }
    };
    const numericalCheck = await verify();
    if (!numericalCheck.ok) return { ...base, numericalCheck, reason: "numerical-guard-failed" };
    queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
    const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(), indices = { baseline: [], candidate: [] };
    dispatch(warm, false); dispatch(warm, true); warm.end(); let at = 0;
    for (let sample = 0; sample < SAMPLES; sample++) for (const fused of sample % 2 ? [true, false] : [false, true]) {
      if (now() >= deadline) throw deadlineError();
      indices[fused ? "candidate" : "baseline"].push(at);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
      for (let repeat = 0; repeat < REPETITIONS; repeat++) dispatch(pass, fused); pass.end(); at += 2;
    }
    encoder.resolveQuerySet(queries, 0, SAMPLES * 4, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, SAMPLES * 32); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, SAMPLES * 32, deadline); mapped = true; if (!within) throw deadlineError();
    let times;
    try { const stamps = new BigInt64Array(read.getMappedRange(0, SAMPLES * 32)); times = Object.fromEntries(Object.entries(indices).map(([label, values]) => [label, values.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS)])); }
    finally { read.unmap(); mapped = false; }
    const afterTimingCheck = await verify();
    if (!afterTimingCheck.ok) return { ...base, numericalCheck, afterTimingCheck, ...times, reason: "numerical-guard-failed" };
    const choice = selectMatvecKernel({ ...times, numericalCheck: afterTimingCheck });
    return { ...base, numericalCheck, afterTimingCheck, ...times, selected: choice.selected, stable: choice.stable, benefit: choice.benefit,
      baselineMs: choice.genericMs, candidateMs: choice.wideMs, reason: choice.reason === "stable-matvec-win" ? "stable-prefix-gelu-win" : choice.reason };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

export async function calibrateGemma4PrefixGelu(device, source, weights, { plan, retile, bn32, gelu, enabled = true } = {}) {
  const started = now(), selection = new Set(); let bundle = null;
  const diagnostics = { revision: 1, method: "baseline", activeRows: plan?.activeRows ?? null, logicalRows: plan?.logicalRows ?? null,
    f16: false, rows: 2, physicalBN: 32, virtualBN: 64, samples: SAMPLES, repetitions: REPETITIONS, threshold: 0.05, points: [], truncated: false };
  const result = (reason) => ({ pipelines: bundle?.pipelines || null, selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  const abort = (reason) => { selection.clear(); for (const point of diagnostics.points) if (point.selected) { point.selected = false; point.reason = "aborted-after-calibration-failure"; } return result(reason); };
  if (!enabled) return result("disabled");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  if (!positive(plan?.activeRows) || plan.activeRows > 32 || !positive(plan.staticRows) || plan.logicalRows !== plan.staticRows + plan.activeRows || plan.logicalRows > 256 || !plan.layerWidths?.length) return result("unsupported-prefix-plan");
  if (!bn32?.pipeline || !(bn32.selection instanceof Set) || bn32.diagnostics?.f16 !== false || bn32.diagnostics.rows !== 2 || bn32.diagnostics.candidateBN !== 32) return result("unsupported-bn32-baseline");
  if (!(retile?.selection instanceof Map) || !gelu) return result("missing-prefix-baseline");
  const shapes = shapesFor(weights, plan, retile, bn32); if (!shapes.length) return result("no-eligible-pairs");
  diagnostics.method = "paired-timestamp-query";
  const scoped = (work) => device.pushErrorScope ? withGpuErrors({ device }, work, "calibrating Gemma prefix GELU") : work();
  try { bundle = await scoped(() => compile(device, source)); }
  catch (error) { diagnostics.error = String(error?.message || error); return abort("pipeline-failed"); }
  const deadline = started + DEADLINE_MS;
  for (const shape of shapes) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline"); }
    try {
      const point = await scoped(() => measure(device, shape, plan, bundle, bn32, gelu, deadline)); diagnostics.points.push(point);
      if (point.selected) selection.add(shape.key);
      if (["numerical-guard-failed", "insufficient-samples"].includes(point.reason)) return abort(point.reason === "insufficient-samples" ? "timing-failed" : point.reason);
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (["MATVEC_DEADLINE", "GEMMA4_PREFIX_GELU_DEADLINE"].includes(error?.code)) { diagnostics.truncated = true; return result("deadline"); }
      return abort("calibration-failed");
    }
  }
  return result(selection.size ? "calibrated" : "no-stable-win");
}
