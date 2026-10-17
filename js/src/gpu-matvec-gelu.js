// Gemma's short independent-row MLP gate/up projections and GELU share matvec arithmetic.
import { bufferUsage, encodeMatvec, inErrorScopes, pipeline, scratchBuffer } from "./gpu.js";
import { compareMatvecOutputs, mapBeforeDeadline, matvecTimingMode, selectMatvecKernel } from "./gpu-matvec.js";

const SAMPLES = 3;
const MAX_SHAPES = 2;
const DEADLINE_MS = 750;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

export async function matvecGeluPipelines(device, wgsl, { f16 = device.features.has("shader-f16") } = {}) {
  const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
  const layout = device.createBindGroupLayout({
    label: "Gemma paired matvec GELU",
    entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
  });
  const spec = { weights: "q8", f16, rows: 1, groups: 1, splitTarget: 256 };
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  return { pipeline: await pipeline(device, wgsl("matvec_gelu", spec), "matvec_gelu", pl), layout, weights: "q8", f16, columns: f16 ? 16 : 4 };
}

export function matvecGeluBindGroup(device, pipes, { globals, params, input, gate, gateScales, up, upScales, output }) {
  return device.createBindGroup({
    layout: pipes.layout,
    entries: [globals, params, input, gate, gateScales, up, upScales, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

export function encodeMatvecGelu(pass, pipes, op, T = 1) {
  pass.setPipeline(pipes.pipeline);
  pass.setBindGroup(0, op.group);
  pass.dispatchWorkgroups(Math.ceil(op.N / pipes.columns), T);
}

export function compareMatvecGeluOutputs(baseline, candidate) {
  const check = compareMatvecOutputs(baseline, candidate, { f16: true });
  return { ...check, tolerance: { absolute: 0, relative: 0, exact: true } };
}

function pairedShapes(weights, selection) {
  const tensors = weights?.tensors;
  if (!(tensors instanceof Map)) return [];
  const seen = new Set();
  const shapes = [];
  for (const [name, gate] of tensors) {
    if (!/^l\.\d+\.gate$/.test(name) || gate.info?.dtype !== "q8") continue;
    const up = tensors.get(name.replace(/\.gate$/, ".up"));
    const [N, K] = gate.info.shape;
    const key = `${N}:${K}`;
    if (!selection?.has(key) || seen.has(key) || up?.info?.dtype !== "q8" || up.info.shape[0] !== N || up.info.shape[1] !== K) continue;
    if (!gate.buf || !gate.sbuf || !up.buf || !up.sbuf) continue;
    shapes.push({ key, N, K, gate, up, name });
    seen.add(key);
    if (shapes.length === MAX_SHAPES) break;
  }
  return shapes;
}

function resolutionMs() {
  let previous = now(), resolution = Infinity;
  const deadline = Date.now() + 4;
  for (let i = 0; i < 262144 && Date.now() <= deadline; i++) {
    const current = now();
    if (current > previous) { resolution = Math.min(resolution, current - previous); if (resolution <= 2) break; }
    previous = current;
  }
  return Number.isFinite(resolution) ? resolution : null;
}

async function measureShape(device, shape, baseline, candidate, gelu, timing, deadline) {
  const U = bufferUsage();
  const { N, K } = shape;
  const yBytes = N * 4;
  const readBytes = Math.max(256, yBytes * 2);
  const largest = Math.max(K * 4, N * 4, readBytes);
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (largest > limit || K * 4 + N * 24 + readBytes > 16 * 1024 * 1024) return { key: shape.key, reason: "scratch-limit", selected: false };
  const owned = [];
  let queries, read, mapped = false;
  try {
    const make = (size, usage) => {
      const value = scratchBuffer(device, size, usage, "Gemma paired matvec calibration");
      owned.push(value); return value;
    };
    const globals = make(16, U.UNIFORM | U.COPY_DST);
    const params = make(16, U.UNIFORM | U.COPY_DST);
    const geluParams = make(16, U.UNIFORM | U.COPY_DST);
    const x = make(K * 4, U.STORAGE | U.COPY_DST);
    const zero = make(16, U.STORAGE);
    const part = make(16, U.STORAGE);
    const gate = make(yBytes, U.STORAGE), up = make(yBytes, U.STORAGE);
    const baselineY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const candidateY = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    read = make(readBytes, U.MAP_READ | U.COPY_DST);
    device.queue.writeBuffer(globals, 0, new Uint32Array([1, 0, 0, 0]));
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
    device.queue.writeBuffer(geluParams, 0, new Uint32Array([N, 0, 0, 0]));
    device.queue.writeBuffer(x, 0, Float32Array.from({ length: K }, (_, i) => 0.04 * Math.sin(i * 0.127 + 0.3)));
    const poison = new Float32Array(N).fill(NaN);
    device.queue.writeBuffer(baselineY, 0, poison);
    device.queue.writeBuffer(candidateY, 0, poison);
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const gateOp = { N, group: group(baseline.layout, [globals, params, x, shape.gate.buf, shape.gate.sbuf, zero, gate, part]) };
    const upOp = { N, group: group(baseline.layout, [globals, params, x, shape.up.buf, shape.up.sbuf, zero, up, part]) };
    const geluGroup = group(gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, baselineY]);
    const fusedOp = { N, group: matvecGeluBindGroup(device, candidate, { globals, params, input: x, gate: shape.gate.buf, gateScales: shape.gate.sbuf, up: shape.up.buf, upScales: shape.up.sbuf, output: candidateY }) };
    const dispatch = (pass, fused) => {
      if (fused) { encodeMatvecGelu(pass, candidate, fusedOp); return; }
      encodeMatvec(pass, baseline, gateOp, 1);
      encodeMatvec(pass, baseline, upOp, 1);
      pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup); pass.dispatchWorkgroups(Math.ceil(N / 256));
    };
    const checkEncoder = device.createCommandEncoder();
    const checkPass = checkEncoder.beginComputePass();
    dispatch(checkPass, false); dispatch(checkPass, true); checkPass.end();
    checkEncoder.copyBufferToBuffer(baselineY, 0, read, 0, yBytes);
    checkEncoder.copyBufferToBuffer(candidateY, 0, read, yBytes, yBytes);
    device.queue.submit([checkEncoder.finish()]);
    const withinDeadline = await mapBeforeDeadline(read, yBytes * 2, deadline);
    mapped = true;
    if (!withinDeadline) throw Object.assign(new Error("paired matvec calibration deadline"), { code: "MATVEC_DEADLINE" });
    const values = new Float32Array(read.getMappedRange(0, yBytes * 2).slice(0));
    read.unmap(); mapped = false;
    // Fusion must match the already-selected matvec plus GELU exactly in both precision modes.
    const numericalCheck = compareMatvecGeluOutputs(values.subarray(0, N), values.subarray(N));
    if (!numericalCheck.ok) return { key: shape.key, numericalCheck, selected: false, reason: "numerical-guard-failed" };
    const times = { baseline: [], candidate: [] };
    if (timing.method === "paired-timestamp-query") {
      queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC);
      const encoder = device.createCommandEncoder();
      const warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); warm.end();
      const indices = { baseline: [], candidate: [] };
      let index = 0;
      for (let sample = 0; sample < SAMPLES; sample++) for (const fused of sample % 2 ? [true, false] : [false, true]) {
        const label = fused ? "candidate" : "baseline";
        indices[label].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let i = 0; i < timing.repetitions; i++) dispatch(pass, fused);
        pass.end(); index += 2;
      }
      encoder.resolveQuerySet(queries, 0, index, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, read, 0, index * 8);
      device.queue.submit([encoder.finish()]);
      const withinTimingDeadline = await mapBeforeDeadline(read, index * 8, deadline);
      mapped = true;
      if (!withinTimingDeadline) throw Object.assign(new Error("paired matvec calibration deadline"), { code: "MATVEC_DEADLINE" });
      const stamps = new BigInt64Array(read.getMappedRange(0, index * 8).slice(0));
      read.unmap(); mapped = false;
      for (const label of ["baseline", "candidate"]) times[label] = indices[label].map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / timing.repetitions);
    } else {
      for (let sample = 0; sample < SAMPLES; sample++) for (const fused of sample % 2 ? [true, false] : [false, true]) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (let i = 0; i < timing.repetitions; i++) dispatch(pass, fused);
        pass.end();
        encoder.copyBufferToBuffer(fused ? candidateY : baselineY, 0, read, 0, yBytes);
        const command = encoder.finish(), started = now();
        device.queue.submit([command]);
        const withinWallDeadline = await mapBeforeDeadline(read, yBytes, deadline);
        mapped = true;
        const elapsed = now() - started;
        read.unmap(); mapped = false;
        if (!withinWallDeadline) throw Object.assign(new Error("paired matvec calibration deadline"), { code: "MATVEC_DEADLINE" });
        times[fused ? "candidate" : "baseline"].push(elapsed / timing.repetitions);
      }
    }
    const selected = selectMatvecKernel({ ...times, numericalCheck });
    return { key: shape.key, ...times, ...selected, kernel: selected.selected ? "matvec-gelu" : "matvec" };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    queries?.destroy();
  }
}

/** Calibrate at most two loaded MLP shapes, requiring exact outputs and stable paired wins. */
export async function calibrateMatvecGelu(device, wgsl, weights, matvec, { gelu, enabled = true } = {}) {
  const started = now(), selection = new Set();
  const diagnostics = { revision: 1, method: "baseline", samples: SAMPLES, tokens: 1, points: [], truncated: false };
  const result = (reason, pipelines = null) => ({ pipelines, selection, diagnostics: { ...diagnostics, reason, elapsedMs: now() - started } });
  const abort = (reason) => {
    selection.clear();
    for (const point of diagnostics.points) if (point.selected) {
      point.selected = false; point.kernel = "matvec"; point.reason = "aborted-after-calibration-failure";
    }
    return result(reason, pipelines);
  };
  if (!enabled) return result("disabled");
  if (!matvec?.pipelines || !gelu) return result("missing-baseline-pipeline");
  const shapes = pairedShapes(weights, matvec.selection);
  if (!shapes.length) return result("no-selected-matvec-pairs");
  const timing = matvecTimingMode(device.features, device.features.has("timestamp-query") ? null : resolutionMs());
  diagnostics.method = timing.method; diagnostics.repetitions = timing.repetitions;
  if (!timing.repetitions) return result(timing.reason);
  let pipelines;
  try { pipelines = await inErrorScopes(device, () => matvecGeluPipelines(device, wgsl, { f16: matvec.pipelines.f16 })); }
  catch (error) { diagnostics.error = String(error?.message || error); return result("pipeline-failed"); }
  const deadline = started + DEADLINE_MS;
  for (const shape of shapes) {
    if (now() >= deadline) { diagnostics.truncated = true; return result("deadline", pipelines); }
    try {
      const point = await inErrorScopes(device, () => measureShape(device, shape, matvec.pipelines, pipelines, gelu, timing, deadline));
      diagnostics.points.push(point);
      if (point.selected) selection.add(shape.key);
      if (point.reason === "numerical-guard-failed") return abort("numerical-guard-failed");
    } catch (error) {
      diagnostics.error = String(error?.message || error);
      if (error?.code === "MATVEC_DEADLINE") { diagnostics.truncated = true; return result("deadline", pipelines); }
      return abort("calibration-failed");
    }
  }
  return result(selection.size ? "calibrated" : "no-stable-win", pipelines);
}
