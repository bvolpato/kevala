// Independent suffix gate/up workgroups share one dispatch grid, retaining BM56 arithmetic.
import { bufferUsage, inErrorScopes, mmSplits, pipeline, rowsPerThread } from './gpu.js';
import { createGemma4BatchMlpValidator } from './gpu-gemma4-batch-mlp.js';
import { mapBeforeDeadline } from './gpu-matvec.js';

const TOKENS = 97, N = 6144, K = 1536, KEY = `${N}:${K}`, LIMIT = 64 * 1024 * 1024;
const SPEC = { weights: 'q8', f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
const SENTINEL = new Uint32Array(new Float32Array([12345.25]).buffer)[0];
const expired = () => Object.assign(new Error('Gemma grid MLP calibration deadline exceeded'), { code: 'GRID_MLP_DEADLINE' });

export function gemma4GridMlpBaseline(tokens, baseline) {
  return tokens === TOKENS && baseline?.f16 === false && baseline.groups === 1 && baseline.splitTarget === 256 &&
    baseline.kernel === 'matmul' && rowsPerThread(tokens, baseline) === 4 && !!baseline.mm56 && !!baseline.reduce?.[4];
}

function matchingWeights(shape) {
  return ['gate', 'up'].every((name) => {
    const weight = shape?.[name];
    return weight?.info?.dtype === 'q8' && Array.isArray(weight.info.shape) && weight.info.shape.length === 2 &&
      Number(weight.info.shape[0]) === N && Number(weight.info.shape[1]) === K && !!weight.buf && !!weight.sbuf;
  });
}

export function gemma4GridMlpEligible(tokens, gate, up, gelu, baseline) {
  return gemma4GridMlpBaseline(tokens, baseline) && gate?.kind === 'mm' && gate.label === 'mm.gate' && gate.dtype === 'q8' &&
    gate.batched === true && gate.sourcePolicy !== true && gate.N === N && gate.K === K && gate.mode === 0 && gate.bias === 0 &&
    up?.kind === 'mm' && up.label === 'mm.up' && up.dtype === 'q8' && up.batched === true && up.sourcePolicy !== true &&
    up.N === N && up.K === K && up.mode === 0 && up.bias === 0 && gelu?.kind === 'GELU' && gelu.label === 'gelu' &&
    gate.mlpPartners?.[0] === up && gate.mlpPartners?.[1] === gelu && matchingWeights(gate.gridMlpSource);
}

export function gemma4GridMlpTiming(baseline, candidate) {
  const valid = (values) => Array.isArray(values) && values.length === 3 && values.every((v) => Number.isFinite(v) && v > 0);
  if (!valid(baseline) || !valid(candidate)) return { selected: false, reason: 'invalid-timestamps' };
  const median = (values) => [...values].sort((a, b) => a - b)[1];
  const baselineMs = median(baseline), candidateMs = median(candidate), gains = baseline.map((v, i) => (v - candidate[i]) / v);
  const selected = gains.every((gain) => gain >= 0.05);
  return { selected, reason: selected ? 'stable-grid-win' : 'below-threshold', baselineMs, candidateMs, benefit: (baselineMs - candidateMs) / baselineMs, pairedBenefits: gains };
}

export function selectGemma4GridMlp(point) {
  const memory = gemma4GridMlpMemory();
  const valid = (guard) => guard?.ok === true && guard.bitMismatches === 0 && guard.nonfinite === 0 && guard.tailMismatches === 0 &&
    guard.outputWords === memory.active && guard.partialWords === 2 * memory.partial && guard.rawBits === true;
  if (!valid(point.before) || !valid(point.after)) return { selected: false, reason: 'numerical-guard-failed' };
  return gemma4GridMlpTiming(point.baseline, point.candidate);
}

export function gemma4GridMlpMemory() {
  const splits = mmSplits(1, N, K, 256, 64, 64), active = TOKENS * N, partial = splits * active;
  const x = 128 * K * 4, y = (128 * N + 4) * 4, part = (partial + 4) * 4, pair = (2 * partial + 4) * 4;
  const retainedBytes = pair + 2 * 16;
  const gpuBytes = x + 4 * y + 2 * part + 16 + 9 * 16 + 16 + 256 + 256;
  const hostBytes = 2 * x + 2048;
  return { x, y, part, pair, splits, active, partial, retainedBytes, gpuBytes, hostBytes, totalBytes: retainedBytes + gpuBytes + hostBytes, limitBytes: LIMIT };
}

const group = (device, layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });

export function gemma4GridMlpBindGroups(device, context, op) {
  const r = op.gridMlpSource;
  if (!context?.selection?.has(KEY) || !r || !matchingWeights(r)) throw new Error('Gemma grid MLP bindings require a selected matching Q8 pair');
  return { matmul: group(device, context.layout, [r.globals, op.params, r.input, r.gate.buf, r.gate.sbuf, r.up.buf, r.up.sbuf, context.unused, context.part]),
    reduce: group(device, context.reduce.getBindGroupLayout(0), [r.globals, context.params, context.part, r.output]) };
}

export function encodeGemma4GridMlp(pass, context, op, tokens, device, baseline) {
  if (!context?.selection?.has(KEY) || context.splits !== 3 ||
      !gemma4GridMlpEligible(tokens, op, op.mlpPartners?.[0], op.mlpPartners?.[1], baseline)) {
    throw new Error('Gemma grid MLP dispatch requires its calibrated T97 shape and baseline');
  }
  let bindings = context.groups.get(op);
  if (!bindings) { bindings = gemma4GridMlpBindGroups(device, context, op); context.groups.set(op, bindings); }
  pass.setPipeline(context.pipeline); pass.setBindGroup(0, bindings.matmul); pass.dispatchWorkgroups(N / 64, 2, 2 * context.splits);
  const work = Math.ceil(tokens * N / 256);
  pass.setPipeline(context.reduce); pass.setBindGroup(0, bindings.reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
}

export async function calibrateGemma4GridMlp(device, source, shape, { tokens, baseline, gelu, automatic = true, pairedSelected = false, runtime = null } = {}) {
  const started = performance.now(), deadline = started + 1000;
  const diagnostics = { enabled: false, available: false, tokens, shape: KEY, reason: 'disabled',
    deadlineScope: 'best-effort 1000 ms including compilation; compilation cannot be interrupted' };
  const fallback = () => ({ tokens, selection: new Set(), diagnostics, destroy() {} });
  if (!automatic) { diagnostics.reason = 'kernel-override'; return fallback(); }
  if (!gemma4GridMlpBaseline(tokens, baseline) || !matchingWeights(shape) || pairedSelected) { diagnostics.reason = 'ineligible-baseline'; return fallback(); }
  if (!device.features.has('timestamp-query')) { diagnostics.reason = 'no-timestamp-query'; return fallback(); }
  const memory = gemma4GridMlpMemory(), limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (memory.totalBytes > LIMIT || Math.max(memory.x, memory.y, memory.part, memory.pair) > limit || device.limits.maxComputeWorkgroupStorageSize < 16384) { diagnostics.reason = 'scratch-limit'; return fallback(); }
  let context = null, selected = false;
  try {
    await inErrorScopes(device, async () => {
      const layout = device.createBindGroupLayout({ label: 'Gemma grid MLP', entries: ['uniform', 'uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'storage'].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
      const compute = await pipeline(device, source('gemma4_mlp_grid', SPEC), 'gemma4_mlp_grid', device.createPipelineLayout({ bindGroupLayouts: [layout] }));
      const reduce = await pipeline(device, source('reduce_gelu', SPEC), 'Gemma grid MLP ordered reducer');
      const validator = await createGemma4BatchMlpValidator(device);
      if (performance.now() >= deadline) throw expired();
      const U = bufferUsage(), buffers = [];
      const make = (size, usage, label) => { const buffer = device.createBuffer({ label, size, usage }); buffers.push(buffer); return buffer; };
      try {
        const part = make(memory.pair, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'Gemma grid MLP partials');
        const params = make(16, U.UNIFORM | U.COPY_DST, 'Gemma grid MLP split params');
        const unused = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'Gemma grid MLP unused output');
        device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 1, 0]));
        context = { tokens, pipeline: compute, reduce, layout, part, params, unused, validator, splits: memory.splits, selection: new Set(), groups: new WeakMap(), diagnostics,
          destroy() { for (const buffer of buffers) buffer.destroy(); buffers.length = 0; this.groups = new WeakMap(); this.selection.clear(); } };
      } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
      const point = await measure(device, context, shape, baseline, gelu, deadline);
      Object.assign(diagnostics, { reason: point.reason, point, elapsedMs: performance.now() - started, retainedBytes: memory.retainedBytes });
      if (point.selected) { context.selection.add(KEY); diagnostics.enabled = diagnostics.available = selected = true; }
    });
  } catch (error) {
    selected = false; context?.selection.clear();
    const textLoss = /device[\s-]*(lost|loss)/i.test(String(error?.message || error));
    let lost = runtime?.lost, timer;
    if (!lost && !textLoss && device.lost) {
      try {
        lost = await Promise.race([device.lost, new Promise((resolve) => { timer = setTimeout(() => resolve(null), 0); })]);
      } finally { clearTimeout(timer); }
      lost ||= runtime?.lost;
    }
    if (lost || textLoss) {
      throw Object.assign(new Error(`Gemma grid MLP calibration device lost: ${lost?.message || lost?.reason || error.message}`, { cause: error }), { code: 'WEBGPU_DEVICE_LOST' });
    }
    diagnostics.enabled = diagnostics.available = false;
    diagnostics.reason = error.code === 'GRID_MLP_DEADLINE' || error.code === 'MATVEC_DEADLINE' ? 'deadline' : 'calibration-failed';
    diagnostics.error = String(error?.message || error); diagnostics.elapsedMs = performance.now() - started;
  } finally { if (!selected) context?.destroy(); }
  return selected ? context : fallback();
}

async function measure(device, context, shape, baseline, gelu, deadline) {
  const T = TOKENS;
  const splits = mmSplits(1, N, K, 256, 64, 64), active = T * N, partial = splits * active;
  if (splits !== 3 || context.splits !== splits) throw new Error("Gemma grid MLP split geometry changed");
  const bytes = gemma4GridMlpMemory();
  const gpuBytes = bytes.gpuBytes, hostBytes = bytes.hostBytes, scratchBytes = bytes.totalBytes;
  const owned = [];
  let read, queries, mapped = false, allocated = 0;
  const checkDeadline = () => { if (performance.now() >= deadline) throw expired(); };
  try {
    checkDeadline();
    const validator = context.validator;
    checkDeadline();
    const U = bufferUsage();
    const make = (size, usage, label) => {
      allocated += size;
      if (allocated > gpuBytes) throw new Error('Gemma grid MLP calibration exceeded its declared GPU scratch');
      const buffer = device.createBuffer({ label: `Gemma grid MLP calibration ${label}`, size, usage }); owned.push(buffer); return buffer;
    };
    const uniform = (values) => { const b = make(16, U.UNIFORM | U.COPY_DST, 'params'); device.queue.writeBuffer(b, 0, new Uint32Array(values)); return b; };
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });
    const globals = uniform([T, 1, T, 0]), params = uniform([N, K, 0, 0]), pairParams = uniform([N, K, 1, 0]), geluParams = uniform([N, 0, 0, 0]);
    const zero = make(16, U.STORAGE | U.COPY_DST, 'disabled bias'); device.queue.writeBuffer(zero, 0, new Float32Array(4).fill(NaN));
    const x = make(bytes.x, U.STORAGE | U.COPY_DST, 'input'), values = new Float32Array(bytes.x / 4).fill(NaN);
    let seed = 0x47524944;
    for (let i = 0; i < T * K; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      values[i] = i % 257 === 0 ? -0 : i % 263 === 0 ? 0 : Math.fround((seed / 4294967296 - 0.5) * 0.08);
    }
    device.queue.writeBuffer(x, 0, values); checkDeadline();
    const y = Array.from({ length: 4 }, () => make(bytes.y, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'output'));
    const part = make(bytes.part, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'single reused baseline PART');
    const captured = make(bytes.part, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'untimed gate PART capture');
    const candidatePart = context.part, unused = context.unused;
    const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, 'validation status');
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, 'timestamps'); read = make(256, U.MAP_READ | U.COPY_DST, '16-byte status and timestamp readback');
    const yp = uniform([bytes.y / 4, active, SENTINEL, 0x7fc00000]);
    const pp = uniform([bytes.part / 4, partial, SENTINEL, 0x7fc00000]);
    const cp = uniform([bytes.pair / 4, 2 * partial, SENTINEL, 0x7fc00000]);
    const gp = uniform([partial, partial, SENTINEL, 0x7fc00000]);
    const up = uniform([4, 0, SENTINEL, 0x7fc00000]);
    const r = shape;
    const normal = [r.gate, r.up].map((weight, i) => ({
      group: group(baseline.layout, [globals, params, x, weight.buf, weight.sbuf, zero, y[i], part]),
      reduce: group(baseline.reduceLayout, [globals, params, part, zero, y[i]]),
    }));
    const geluGroup = group(gelu.getBindGroupLayout(0), [globals, geluParams, y[0], y[1], y[2]]);
    const candidate = group(context.layout, [globals, params, x, r.gate.buf, r.gate.sbuf, r.up.buf, r.up.sbuf, unused, candidatePart]);
    const candidateReduce = group(context.reduce.getBindGroupLayout(0), [globals, pairParams, candidatePart, y[3]]);
    const work = (pass, length) => { const n = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535)); };
    const projection = (pass, i) => {
      pass.setPipeline(baseline.mm56); pass.setBindGroup(0, normal[i].group); pass.dispatchWorkgroups(N / 64, 2, splits);
      pass.setPipeline(baseline.reduce[4]); pass.setBindGroup(0, normal[i].reduce); work(pass, active);
    };
    const activation = (pass) => { pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup); work(pass, active); };
    const dispatch = (pass, fused) => {
      if (fused) {
        pass.setPipeline(context.pipeline); pass.setBindGroup(0, candidate); pass.dispatchWorkgroups(N / 64, 2, 2 * splits);
        pass.setPipeline(context.reduce); pass.setBindGroup(0, candidateReduce); work(pass, active);
      } else { projection(pass, 0); projection(pass, 1); activation(pass); }
    };
    const compare = (a, b, p, length) => ({ length, group: group(validator.compareLayout, [p, a, b, status]) });
    const checks = [compare(y[2], y[3], yp, bytes.y / 4), ...y.slice(0, 2).map((b) => compare(b, b, yp, bytes.y / 4)),
      compare(part, part, pp, bytes.part / 4), compare(captured, captured, pp, bytes.part / 4), compare(candidatePart, candidatePart, cp, bytes.pair / 4), compare(unused, unused, up, 4),
      compare({ buffer: captured, size: partial * 4 }, { buffer: candidatePart, size: partial * 4 }, gp, partial),
      compare(part, { buffer: candidatePart, offset: partial * 4, size: bytes.part }, pp, bytes.part / 4)];
    const fills = [...y.map((b) => [b, yp, bytes.y / 4]), [part, pp, bytes.part / 4], [captured, pp, bytes.part / 4], [candidatePart, cp, bytes.pair / 4], [unused, up, 4]].map(([b, p, length]) => ({ length, group: group(validator.fillLayout, [p, b]) }));
    const verify = async () => {
      checkDeadline();
      const encoder = device.createCommandEncoder({ label: 'Gemma grid MLP exact loaded-weight check' }); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(validator.fill);
      for (const item of fills) { fill.setBindGroup(0, item.group); work(fill, item.length); } fill.end();
      const gate = encoder.beginComputePass(); projection(gate, 0); gate.end();
      encoder.copyBufferToBuffer(part, 0, captured, 0, bytes.part); // Capture is outside every timed region.
      const rest = encoder.beginComputePass(); projection(rest, 1); activation(rest); dispatch(rest, true); rest.end();
      const validation = encoder.beginComputePass(); validation.setPipeline(validator.compare);
      for (const item of checks) { validation.setBindGroup(0, item.group); work(validation, item.length); } validation.end();
      encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      try {
        if (!within) throw expired();
        const counts = [...new Uint32Array(read.getMappedRange(0, 16))];
        const result = { ok: counts.every((v) => v === 0), bitMismatches: counts[0], nonfinite: counts[1], tailMismatches: counts[2], outputWords: active, partialWords: 2 * partial, signedZeroChecked: true, inputPaddingNaN: true, rawBits: true, reusedBaselinePart: true, gateCaptureOutsideTiming: true };
        if (!result.ok) throw new Error(`Gemma grid MLP loaded-weight exactness failed: ${JSON.stringify(result)}`);
        return result;
      } finally { read.unmap(); mapped = false; }
    };
    const before = await verify(); checkDeadline();
    queries = device.createQuerySet({ type: 'timestamp', count: 12 });
    const encoder = device.createCommandEncoder({ label: 'Gemma grid MLP reused-PART paired timing' });
    const warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); warm.end();
    const indices = [[], []]; let index = 0;
    for (let pair = 0; pair < 3; pair++) for (const fused of pair % 2 ? [true, false] : [false, true]) {
      indices[Number(fused)].push(index);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
      for (let rep = 0; rep < 4; rep++) dispatch(pass, fused); pass.end(); index += 2;
    }
    encoder.resolveQuerySet(queries, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, 96, deadline); mapped = true;
    if (!within) throw expired();
    const timestamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
    const [baselineTimes, candidateTimes] = indices.map((values) => values.map((i) => Number(timestamps[i + 1] - timestamps[i]) / 1e6 / 4));
    if ([...baselineTimes, ...candidateTimes].some((v) => !Number.isFinite(v) || v <= 0)) throw new Error('Gemma grid MLP loaded-weight timestamps invalid');
    const after = await verify(), choice = selectGemma4GridMlp({ baseline: baselineTimes, candidate: candidateTimes, before, after });
    return { ...choice, before, after,
      gateName: r.gate.info.name || null, upName: r.up.info.name || null, baselinePartBuffers: 1, gateCaptureTimed: false, baselineDispatches: 5, candidateDispatches: 2,
      timing: { baseline: baselineTimes, candidate: candidateTimes, ...choice, pairs: 3, repetitions: 4, allocationsUploadsAndCaptureTimed: false },
      memory: { gpuTemporaryBytes: gpuBytes, actualGpuTemporaryBytes: allocated, hostTemporaryBytes: hostBytes, controlledTemporaryBytes: scratchBytes, limitBytes: LIMIT, loadedWeightsCopied: false, retainedContextBytes: bytes.retainedBytes },
    };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}
