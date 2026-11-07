// Same-device exactness and timing gate for the narrow BF16 vision Q/K/V dispatch.
import { mmSplits, pipeline, rowsPerThread, withGpuErrors } from "./gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "./gpu-gemma4-batch-mlp.js";
import { mapBeforeDeadline } from "./gpu-matvec.js";

const T = 108, N = 768, K = 768, SPLITS = 6, HEADS = 12, DIM = 64;
const DEADLINE_MS = 1000, MAX_TEMPORARY = 16 * 1024 * 1024;
const SENTINEL = 0x4640e500, POISON = 0x7fc00000;
const SPEC = { weights: "bf16", f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
const PREFIX = "vision_tower.encoder.layers.0.self_attn.";
const now = () => performance.now();
const resource = (value) => value?.buffer ? value : { buffer: value };
const sameView = (a, b) => { a = resource(a); b = resource(b); return a.buffer === b.buffer && (a.offset || 0) === (b.offset || 0) && (a.size ?? a.buffer?.size - (a.offset || 0)) === (b.size ?? b.buffer?.size - (b.offset || 0)); };
const scalar = (value) => value && typeof value === "object" ? value.f : value;
const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];

export function visionQkvPlan(rows, width, inputWidth, pipes, capacity, alignment = 256) {
  if (rows !== T || width !== N || inputWidth !== K || pipes?.weights !== "bf16" || pipes.f16 !== false || pipes.groups !== 1 || pipes.splitTarget !== 256 || rowsPerThread(rows, pipes) !== 4 || !Number.isSafeInteger(capacity) || !Number.isSafeInteger(alignment) || alignment <= 0) return null;
  const splits = mmSplits(rows, width, inputWidth, 256, 64, 64), partBytes = splits * rows * width * 4;
  if (splits !== SPLITS || partBytes % alignment || 3 * partBytes > capacity) return null;
  return { rows, width, inputWidth, splits, partBytes, offsets: [0, partBytes, 2 * partBytes], requiredBytes: 3 * partBytes, work: [12, 2, SPLITS] };
}

export function selectVisionQkv({ before, after, baseline, candidate }) {
  if (![before, after].every((guard) => guard?.ok === true && guard.bitMismatches === 0 && guard.nonFinite === 0 && guard.tailMismatches === 0)) return { selected: false, reason: "numerical-guard-failed" };
  if (!Array.isArray(baseline) || !Array.isArray(candidate) || baseline.length !== 3 || candidate.length !== 3 || ![...baseline, ...candidate].every((value) => Number.isFinite(value) && value > 0)) return { selected: false, reason: "insufficient-timing-samples" };
  const baselineMs = median(baseline), candidateMs = median(candidate);
  const selected = candidate.every((value, index) => value <= baseline[index] * 0.95);
  return { selected, reason: selected ? "stable-qkv-win" : "no-stable-qkv-win", baselineMs, candidateMs, benefit: 1 - candidateMs / baselineMs };
}

function normMatches(item, name, weighted, skipRope) {
  const norm = item?.qkvNorm;
  return item?.label === `vision.reduce_norm_rope.${name}` && norm?.buffers?.length === 4 && Array.isArray(norm.values) && norm.values.length === 12 &&
    [[0, T], [1, HEADS], [2, DIM], [3, weighted], [7, skipRope], [8, SPLITS], [9, 0], [10, 0]].every(([at, value]) => scalar(norm.values[at]) === value);
}

/** Rebind only exact adjacent Q/K/V partial producers and their existing reducers. */
export function fuseVisionQkvOps(device, context, ops) {
  if (!context?.selected || !context.pipeline || !context.layout) return 0;
  let count = 0;
  for (let index = 0; index + 5 < ops.length; index++) {
    const [q, qr, k, kr, v, vr] = ops.slice(index, index + 6), projections = [q, k, v], reducers = [qr, kr, vr];
    if (![q, k, v].every((item, i) => item?.label === `vision.mm.${["q", "k", "v"][i]}` && item.partialOnly === true && item.qkvMatrix?.dtype === "bf16" && item.qkvMatrix.mode === 0 && item.qkvMatrix.bias === 0 && item.tokens === T) || !normMatches(qr, "q", 1, 0) || !normMatches(kr, "k", 1, 0) || !normMatches(vr, "v", 0, 1)) continue;
    const matrices = projections.map((item) => item.qkvMatrix.buffers);
    if (matrices.some((buffers) => buffers?.length !== 8) || !projections.every((item) => item.pipes === q.pipes) || ![1, 2].every((i) => [0, 2, 7].every((binding) => sameView(matrices[0][binding], matrices[i][binding])))) continue;
    const part = resource(matrices[0][7]), offset = part.offset || 0, capacity = part.size ?? part.buffer.size - offset;
    const plan = visionQkvPlan(T, q.op.N, q.op.K, q.pipes, capacity, device.limits.minStorageBufferOffsetAlignment);
    if (!plan || offset % device.limits.minStorageBufferOffsetAlignment || offset + plan.requiredBytes > part.buffer.size || plan.requiredBytes > device.limits.maxStorageBufferBindingSize || !projections.every((item) => item.op.N === N && item.op.K === K) || !reducers.every((item) => sameView(item.qkvNorm.buffers[0], part))) continue;
    if (matrices.some((buffers) => [2, 3].some((binding) => resource(buffers[binding]).buffer === part.buffer))) continue;
    const bind = (layout, values) => device.createBindGroup({ layout, entries: values.map((value, binding) => ({ binding, resource: resource(value) })) });
    const group = bind(context.layout, [matrices[0][0], matrices[0][1], matrices[0][2], ...matrices.map((buffers) => buffers[3]), { buffer: part.buffer, offset, size: plan.requiredBytes }]);
    const normOps = reducers.map((item, projection) => ({ ...item, group: bind(item.p.getBindGroupLayout(0), [item.qkvNorm.params, { buffer: part.buffer, offset: offset + plan.offsets[projection], size: plan.partBytes }, ...item.qkvNorm.buffers.slice(1)]) }));
    ops.splice(index, 6, { p: context.pipeline, group, groups: plan.work, label: "vision.mm.qkv", qkv: true }, ...normOps);
    count++; index += 3;
  }
  return count;
}

function calibrationInput(bounds) {
  const values = new Float32Array(T * K);
  for (let i = 0; i < values.length; i++) {
    const value = i % 97 === 0 ? i % 194 ? -0 : 0 : 0.0125 * Math.sin((i % K) * 0.071 + Math.floor(i / K) * 0.17 + 0.23);
    values[i] = Math.max(bounds[0], Math.min(bounds[1], value));
  }
  return values;
}

function calibrationRope(theta) {
  const rope = new Float32Array(T * DIM), quarter = DIM / 4;
  for (let row = 0; row < T; row++) for (let axis = 0; axis < 2; axis++) for (let pair = 0; pair < quarter; pair++) {
    const inverse = Math.fround(1 / Math.fround(Math.pow(theta, pair / quarter))), coordinate = axis ? Math.floor(row / 12) : row % 12;
    const angle = Math.fround(coordinate * inverse), at = row * DIM + (axis * quarter + pair) * 2;
    rope[at] = Math.cos(angle); rope[at + 1] = Math.sin(angle);
  }
  return rope;
}

export async function calibrateVisionQkv(runtime, source, { cfg, weights, clips, matmul, norm, enabled = true }) {
  const device = runtime.device, started = now(), deadline = started + DEADLINE_MS;
  const diagnostics = { selected: false, reason: "disabled", method: "actual-weight-bitwise-paired-timestamps", rows: T, N, K, splits: SPLITS, samples: 3, repetitions: 4, threshold: 0.05, deadlineMs: DEADLINE_MS, deadlineScope: "best-effort compilation/calibration cutoff; in-flight compilation is not cancelled", temporaryLimitBytes: MAX_TEMPORARY, sharedBytes: 20480, copiedWeightBytes: 0 };
  const context = { selected: false, pipeline: null, layout: null, diagnostics };
  const finish = (reason) => {
    if (!context.selected) { context.pipeline = null; context.layout = null; }
    diagnostics.reason = reason; diagnostics.selected = context.selected; diagnostics.available = !!context.pipeline; diagnostics.elapsedMs = now() - started;
    return context;
  };
  if (!enabled) return finish("disabled");
  if (!device.features.has("timestamp-query")) return finish("timestamp-query-unavailable");
  if (device.limits.maxComputeWorkgroupStorageSize < 20480 || device.limits.maxStorageBuffersPerShaderStage < 5) return finish("device-limits");
  if (cfg.hidden !== K || cfg.heads !== HEADS || cfg.kvHeads !== HEADS || cfg.headDim !== DIM || !Number.isFinite(cfg.theta)) return finish("unsupported-config");
  if (!visionQkvPlan(T, N, K, matmul, 12 * 1024 * 1024, device.limits.minStorageBufferOffsetAlignment) || !matmul.mm56 || !norm) return finish("unsupported-baseline");
  const owned = [];
  let input, rope, queries, mapped = false, readback, allocated = 0;
  const checkDeadline = () => { if (now() >= deadline) throw new Error("vision-qkv-deadline"); };
  try {
    const matrices = ["q", "k", "v"].map((name) => weights.get(PREFIX + name + "_proj.linear"));
    if (matrices.some((entry) => entry.info.dtype !== "bf16" || entry.info.shape.length !== 2 || entry.info.shape[0] !== N || entry.info.shape[1] !== K || !entry.buf)) return finish("unsupported-weights");
    const bounds = ["q", "k", "v"].map((name) => clips.get(PREFIX + name + "_proj"));
    if (bounds.some((value) => !Array.isArray(value) || value.length !== 4 || value.some(Number.isNaN)) || !bounds.every((value) => Object.is(value[0], bounds[0][0]) && Object.is(value[1], bounds[0][1]))) return finish("input-clips-differ");
    const qn = weights.get(PREFIX + "q_norm"), kn = weights.get(PREFIX + "k_norm");
    if ([qn, kn].some((entry) => entry.info.dtype !== "f32" || entry.info.shape.length !== 1 || entry.info.shape[0] !== DIM || !entry.buf)) return finish("unsupported-norm-weights");
    await withGpuErrors(runtime, async () => {
      const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
      context.layout = device.createBindGroupLayout({ label: "Gemma vision QKV", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
      context.pipeline = await pipeline(device, source("gemma4_vision_qkv", SPEC), "gemma4_vision_qkv", device.createPipelineLayout({ bindGroupLayouts: [context.layout] }));
      const validator = await createGemma4BatchMlpValidator(device);
      checkDeadline();
      const U = GPUBufferUsage, outputCount = 3 * T * N, partialCount = outputCount * SPLITS, segmentBytes = T * N * SPLITS * 4, outputSegment = T * N * 4;
      input = calibrationInput(bounds[0]); rope = calibrationRope(cfg.theta);
      const hostBytes = 2 * (input.byteLength + rope.byteLength) + 65536;
      const make = (bytes, usage, label, data) => {
        bytes = Math.max(16, bytes); allocated += bytes;
        if (allocated + hostBytes > MAX_TEMPORARY || bytes > device.limits.maxBufferSize || ((usage & U.STORAGE) && bytes > device.limits.maxStorageBufferBindingSize)) throw new Error("vision-qkv-scratch-limit");
        const buffer = device.createBuffer({ size: bytes, usage, label: `Gemma QKV calibration ${label}` }); owned.push(buffer);
        if (data) device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const uniform = (values, label) => {
        const data = new ArrayBuffer(Math.max(16, values.length * 4)), ints = new Uint32Array(data), floats = new Float32Array(data);
        values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
        return make(data.byteLength, U.UNIFORM | U.COPY_DST, label, data);
      };
      const bind = (layout, values) => device.createBindGroup({ layout, entries: values.map((value, binding) => ({ binding, resource: resource(value) })) });
      const globals = uniform([T, 0, 0, 0], "globals"), params = uniform([N, K, 0, 0], "matrix params");
      const X = make(input.byteLength, U.STORAGE | U.COPY_DST, "input", input), CS = make(rope.byteLength, U.STORAGE | U.COPY_DST, "RoPE", rope);
      const zero = make(16, U.STORAGE, "disabled weight"), unusedY = make(16, U.STORAGE, "unused matrix Y");
      const parts = [0, 1].map((i) => make((partialCount + 4) * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, `partials${i}`));
      const outputs = [0, 1].map((i) => make((outputCount + 4) * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST, `norm output${i}`));
      const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "validation status");
      readback = make(128, U.MAP_READ | U.COPY_DST, "readback");
      const resolved = make(96, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
      const view = (buffer, offset, size) => ({ buffer, offset, size });
      const normParams = bounds.map((value, i) => uniform([T, HEADS, DIM, Number(i < 2), { f: cfg.eps }, { f: value[2] }, { f: value[3] }, Number(i === 2), SPLITS, 0, 0, 0], `norm params${i}`));
      const baselineGroups = (captureAllParts) => matrices.map((entry, i) => bind(matmul.layout, [globals, params, X, entry.buf, zero, zero, unusedY, view(parts[0], captureAllParts ? i * segmentBytes : 0, segmentBytes)]));
      const normGroups = (candidate, captureAllParts = true) => matrices.map((_entry, i) => bind(norm.getBindGroupLayout(0), [normParams[i], view(parts[Number(candidate)], candidate || captureAllParts ? i * segmentBytes : 0, segmentBytes), i === 0 ? qn.buf : i === 1 ? kn.buf : zero, CS, view(outputs[Number(candidate)], i * outputSegment, outputSegment)]));
      const oldCapture = baselineGroups(true), oldTimed = baselineGroups(false), oldNormCapture = normGroups(false), oldNormTimed = normGroups(false, false), newNorm = normGroups(true);
      const candidateGroup = bind(context.layout, [globals, params, X, ...matrices.map((entry) => entry.buf), view(parts[1], 0, partialCount * 4)]);
      const dispatch = (pass, candidate, captureAllParts = false) => {
        if (candidate) { pass.setPipeline(context.pipeline); pass.setBindGroup(0, candidateGroup); pass.dispatchWorkgroups(12, 2, SPLITS); }
        for (let i = 0; i < 3; i++) {
          if (!candidate) { pass.setPipeline(matmul.mm56); pass.setBindGroup(0, (captureAllParts ? oldCapture : oldTimed)[i]); pass.dispatchWorkgroups(12, 2, SPLITS); }
          pass.setPipeline(norm); pass.setBindGroup(0, (candidate ? newNorm : captureAllParts ? oldNormCapture : oldNormTimed)[i]); pass.dispatchWorkgroups(T * HEADS);
        }
      };
      const groups = [[parts, partialCount], [outputs, outputCount]].map(([buffers, active]) => {
        const p = uniform([active + 4, active, SENTINEL, POISON], "guard params");
        return { length: active + 4, ...gemma4BatchMlpValidationGroups(device, validator, { params: p, baseline: buffers[0], candidate: buffers[1], status }) };
      });
      Object.assign(diagnostics, { temporaryBytes: allocated + hostBytes, gpuTemporaryBytes: allocated, hostAndUploadReserveBytes: hostBytes });
      const work = (pass, length) => pass.dispatchWorkgroups(Math.min(65535, Math.ceil(length / 256)), Math.ceil(length / (65535 * 256)));
      const verify = async () => {
        checkDeadline();
        const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
        const fill = encoder.beginComputePass(); fill.setPipeline(validator.fill);
        for (const item of groups) for (const group of item.fill) { fill.setBindGroup(0, group); work(fill, item.length); } fill.end();
        const compute = encoder.beginComputePass(); dispatch(compute, false, true); dispatch(compute, true); compute.end();
        const compare = encoder.beginComputePass(); compare.setPipeline(validator.compare);
        for (const item of groups) { compare.setBindGroup(0, item.compare); work(compare, item.length); } compare.end();
        encoder.copyBufferToBuffer(status, 0, readback, 0, 16); device.queue.submit([encoder.finish()]);
        if (!await mapBeforeDeadline(readback, 16, deadline)) { readback.unmap(); throw new Error("vision-qkv-deadline"); }
        mapped = true;
        try {
          const counts = new Uint32Array(readback.getMappedRange(0, 16));
          return { ok: counts.every((value) => value === 0), bitMismatches: counts[0], nonFinite: counts[1], tailMismatches: counts[2], partialValues: partialCount, normalizedValues: outputCount, signedZeroChecked: true, readbackBytes: 16 };
        } finally { readback.unmap(); mapped = false; }
      };
      const before = diagnostics.before = await verify();
      if (!before.ok) { diagnostics.reason = "numerical-guard-failed"; return; }
      checkDeadline();
      queries = device.createQuerySet({ type: "timestamp", count: 12 });
      const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); warm.end();
      const order = [];
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
        const at = order.length * 2, pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, candidate); pass.end(); order.push(candidate);
      }
      encoder.resolveQuerySet(queries, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, readback, 0, 96); device.queue.submit([encoder.finish()]);
      if (!await mapBeforeDeadline(readback, 96, deadline)) { readback.unmap(); throw new Error("vision-qkv-deadline"); }
      mapped = true;
      const baseline = diagnostics.baseline = [], candidate = diagnostics.candidate = [];
      try {
        const stamps = new BigUint64Array(readback.getMappedRange(0, 96));
        order.forEach((next, i) => (next ? candidate : baseline).push(Number(stamps[i * 2 + 1] - stamps[i * 2]) / 1e6 / 4));
      } finally { readback.unmap(); mapped = false; }
      const after = diagnostics.after = await verify();
      Object.assign(diagnostics, selectVisionQkv({ before, after, baseline, candidate }), { temporaryBytes: allocated + hostBytes, gpuTemporaryBytes: allocated, hostAndUploadReserveBytes: hostBytes, measuredScope: "original reusable-PART Q/K/V GEMMs+norms vs joint GEMM+same norms" });
    }, "calibrating Gemma vision QKV");
    if (now() >= deadline) return finish("deadline");
    context.selected = diagnostics.selected === true;
    return finish(diagnostics.reason);
  } catch (error) {
    if (runtime.lost || /device[\s-]*(?:lost|loss)/i.test(String(error?.message || error))) throw error;
    diagnostics.selected = false; diagnostics.error = String(error?.message || error);
    return finish(diagnostics.error.includes("vision-qkv-deadline") ? "deadline" : diagnostics.error.includes("scratch-limit") ? "scratch-limit" : "calibration-failed");
  } finally {
    if (mapped) readback.unmap(); queries?.destroy();
    for (const buffer of owned) buffer.destroy();
    input?.fill(0); rope?.fill(0);
  }
}
