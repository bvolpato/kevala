// Optional automatic selection for the fused Q8 paired-projection kernel.
//
// This module is deliberately separate from the model runtimes. Callers provide the actual
// loaded Q8 tensor entries and an optional generic matmul pipeline bundle. The result exposes a
// fused pipeline only when at least one eligible token bucket wins calibration.

import { dispatchMatmul, matmulLayout, mmSplits, pipeline, reduceLayout, rowsPerThread } from "./gpu.js";
import { selectMatmulKernel } from "./gpu-tuning.js";

export const GATED_TUNING_REVISION = 1;
export const GATED_TOKENS = [192, 512];
export const GATED_SAMPLES = 3;
export const GATED_REPETITIONS = 2;
export const GATED_THRESHOLD = 0.05;
export const GATED_MAX_SHAPES = 2;
export const GATED_SHARED_BYTES = 24576;
export const GATED_DEADLINE_MS = 1500;
export const GATED_MAX_SCRATCH = 64 * 1024 * 1024;
export const GATED_SPLIT_TARGET = 256;

const ACTIVATIONS = new Set([0, 1, 2]);
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

function tokenBucket(tokens) {
  return tokens < 256 ? "mid" : "long";
}

function validInteger(value, minimum = 1) {
  return Number.isSafeInteger(value) && value >= minimum;
}

function shapeOf(entry) {
  const shape = entry?.info?.shape;
  if (!Array.isArray(shape) || shape.length !== 2) return null;
  const [n, k] = shape.map(Number);
  return validInteger(n) && validInteger(k) ? { n, k } : null;
}

function sameStorage(left, right) {
  return !!left && !!right && left.buf === right.buf && left.sbuf === right.sbuf;
}

function shapeDescriptor(descriptor) {
  const gate = descriptor?.gate;
  const up = descriptor?.up;
  const packed = descriptor?.packed === true;
  const activation = Number(descriptor?.activation);
  if (!gate || !up) return { ok: false, reason: "missing-projection" };
  if (!ACTIVATIONS.has(activation) || !Number.isInteger(activation)) return { ok: false, reason: "invalid-activation" };
  if (gate.info?.dtype !== "q8" || up.info?.dtype !== "q8") return { ok: false, reason: "q8-required" };
  if (Number(gate.info?.block) !== 32 || Number(up.info?.block) !== 32) return { ok: false, reason: "q8-block32-required" };
  if (!gate.buf || !gate.sbuf || !up.buf || !up.sbuf) return { ok: false, reason: "weights-not-loaded" };
  const gateShape = shapeOf(gate);
  const upShape = shapeOf(up);
  if (!gateShape || !upShape || gateShape.k !== upShape.k) return { ok: false, reason: "shape-mismatch" };
  if (gateShape.k % 32 !== 0) return { ok: false, reason: "k-not-q8-block-aligned" };
  if (packed) {
    if (!sameStorage(gate, up)) return { ok: false, reason: "packed-buffers-differ" };
    if (gateShape.n !== upShape.n || gateShape.n % 2 !== 0) return { ok: false, reason: "packed-halves-mismatch" };
  } else if (gateShape.n !== upShape.n) {
    return { ok: false, reason: "shape-mismatch" };
  }
  // The existing activation shaders implement these three runtime contracts. A packed tanh pair
  // and an unpacked erf/SiLU pair would need an additional baseline activation shader, so leave
  // those combinations on the generic path until such a shader exists.
  if (packed && activation === 0) return { ok: false, reason: "packed-tanh-baseline-unavailable" };
  if (!packed && activation !== 0) return { ok: false, reason: "unpacked-non-gelu-baseline-unavailable" };
  const n = packed ? gateShape.n / 2 : gateShape.n;
  return {
    ok: true,
    N: n,
    K: gateShape.k,
    packedN: packed ? gateShape.n : n,
    upOffset: packed ? n : 0,
    activation,
    packed,
    gate,
    up,
  };
}

function shapeOnly(value) {
  if (value?.gate || value?.up) return shapeDescriptor(value);
  const N = Number(value?.N);
  const K = Number(value?.K);
  const activation = Number(value?.activation);
  const packed = value?.packed === true;
  if (!validInteger(N) || !validInteger(K) || K % 32 !== 0 || !ACTIVATIONS.has(activation) || (packed && activation === 0) || (!packed && activation !== 0)) return { ok: false, reason: "invalid-shape" };
  return { ok: true, N, K, packedN: packed ? 2 * N : N, upOffset: packed ? N : 0, activation, packed };
}

function splitInfo(info, tokens, splitTarget) {
  if (tokens == null) return { ok: true };
  if (!validInteger(tokens, 64)) return { ok: false, reason: "short-input" };
  const fusedSplits = mmSplits(tokens, info.N, info.K, splitTarget, 64, 64);
  const baselineRows = rowsPerThread(tokens);
  const baselineSplits = mmSplits(tokens, info.packedN, info.K, splitTarget, 16 * baselineRows, 64);
  if (fusedSplits !== 1) return { ok: false, reason: "fused-split-k", fusedSplits, baselineSplits };
  if (baselineSplits !== 1) return { ok: false, reason: "baseline-split-k", fusedSplits, baselineSplits };
  return { ok: true, fusedSplits, baselineSplits };
}

/** Pure descriptor and shape guard. `tokens` adds the R4 and unsplit-K runtime guard. */
export function canFuse(descriptor, { tokens = null, splitTarget = GATED_SPLIT_TARGET } = {}) {
  const info = shapeOnly(descriptor);
  return info.ok && splitInfo(info, tokens, splitTarget).ok;
}

/** Explain the same pure contract used by `canFuse`, for diagnostics and tests. */
export function fusionEligibility(descriptor, { tokens = null, splitTarget = GATED_SPLIT_TARGET } = {}) {
  const info = shapeOnly(descriptor);
  if (!info.ok) return info;
  return { ...info, ...splitInfo(info, tokens, splitTarget) };
}

/** Stable map key for one logical pair shape and a mid/long token bucket. */
export function selectionKey(shape, tokens) {
  const info = shapeOnly(shape);
  if (!info.ok || !validInteger(tokens)) return null;
  return `${info.N}:${info.K}:a${info.activation}:p${info.packed ? 1 : 0}:${tokenBucket(tokens)}`;
}

function selectionPoint(point, threshold) {
  if (!point || point.eligible === false) return { tokens: point?.tokens ?? null, eligible: false, choice: "generic", stable: false, reason: point?.reason || "ineligible", baseline: point?.baseline || null, fused: point?.fused || null };
  const selected = selectMatmulKernel({ generic: point.baseline, wide: point.fused }, { threshold });
  return { ...selected, choice: selected.kernel === "wide" ? "fused" : "generic", tokens: point.tokens, eligible: true, baseline: point.baseline, fused: point.fused };
}

/** Pure paired decision. Each eligible token bucket is selected independently. */
export function selectGateUpCandidate(measurements, { threshold = GATED_THRESHOLD } = {}) {
  const input = Array.isArray(measurements?.points) ? measurements.points : [];
  const results = GATED_TOKENS.map((tokens) => selectionPoint(input.find((point) => point?.tokens === tokens), threshold));
  const winners = results.filter((point) => point.choice === "fused");
  const ratios = winners.map((point) => 1 - point.benefit);
  const normalizedRatio = ratios.length ? Math.exp(ratios.reduce((sum, ratio) => sum + Math.log(ratio), 0) / ratios.length) : null;
  const stable = winners.length > 0;
  const allEligible = results.every((point) => point.eligible);
  const reason = !stable ? results.some((point) => point.reason === "regression") ? "point-regression" : results.some((point) => point.reason === "insufficient-samples") ? "insufficient-samples" : "no-stable-win" : winners.length === GATED_TOKENS.length ? "stable-win-both-points" : allEligible ? "stable-win-eligible-points" : "stable-win-with-ineligible-point";
  return {
    choice: stable ? "fused" : "generic",
    stable,
    benefit: normalizedRatio == null ? null : 1 - normalizedRatio,
    normalizedRatio,
    points: results,
    reason,
  };
}

function usage() {
  if (!globalThis.GPUBufferUsage) throw new Error("WebGPU buffer usage constants are unavailable");
  return globalThis.GPUBufferUsage;
}

function buffer(device, size, flags, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage: flags });
}

function deviceLimit(device) {
  return Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
}

function activationSpec(activation, packed) {
  if (packed && activation === 1) return { kernel: "geglu", bindings: 4 };
  if (packed && activation === 2) return { kernel: "kev_silumul", bindings: 4 };
  if (!packed && activation === 0) return { kernel: "gemma4_gelu", bindings: 5 };
  return null;
}

function layoutForActivation(device, bindings) {
  const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  return device.createBindGroupLayout({
    label: "gated activation",
    entries: [entry(0, "uniform"), entry(1, "uniform"), ...Array.from({ length: bindings - 2 }, (_, i) => entry(i + 2, i === bindings - 3 ? "storage" : "read-only-storage"))],
  });
}

async function buildF32Matmul(device, wgsl, splitTarget) {
  const layout = matmulLayout(device);
  const reduceLayoutObject = reduceLayout(device);
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const reducePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [reduceLayoutObject] });
  const spec = { f16: false, groups: 1, rows: 4, splitTarget };
  const [mm, reduce] = await Promise.all([
    pipeline(device, wgsl("matmul", spec), "gated.generic.r4", pipelineLayout),
    pipeline(device, wgsl("reduce", spec), "gated.reduce.r4", reducePipelineLayout),
  ]);
  const mms = [];
  const reduces = [];
  mms[4] = mm;
  reduces[4] = reduce;
  return { layout, reduceLayout: reduceLayoutObject, mm: mms, reduce: reduces, splitTarget, groups: 1 };
}

async function buildFused(device, wgsl, splitTarget) {
  const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  const layout = device.createBindGroupLayout({
    label: "gated fused matmul",
    entries: [
      entry(0, "uniform"),
      entry(1, "uniform"),
      ...Array.from({ length: 5 }, (_, index) => entry(index + 2, "read-only-storage")),
      entry(7, "storage"),
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipelineObject = await pipeline(device, wgsl("matmul_gate_up", { f16: false, groups: 1, rows: 4, splitTarget }), "gated.fused.r4", pipelineLayout);
  return { pipeline: pipelineObject, layout };
}

async function buildActivation(device, wgsl, activation, packed) {
  const spec = activationSpec(activation, packed);
  if (!spec) throw new Error(`no baseline activation shader for activation ${activation} packed=${packed}`);
  const layout = layoutForActivation(device, spec.bindings);
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipelineObject = await pipeline(device, wgsl(spec.kernel), `gated.${spec.kernel}` , pipelineLayout);
  return { pipeline: pipelineObject, layout };
}

function makeGroup(device, layout, resources) {
  return device.createBindGroup({ layout, entries: resources.map((resource, binding) => ({ binding, resource: { buffer: resource } })) });
}

function makeMatmulOp(device, pipes, shape, global, params, input, output, bias, part) {
  const resources = [global, params, input, shape.buf, shape.sbuf, bias, output, part];
  const group = makeGroup(device, pipes.layout, resources);
  const reduce = makeGroup(device, pipes.reduceLayout, [global, params, part, bias, output]);
  return { N: shape.N, K: shape.K, group, reduce, pMatmul: pipes.mm[4], pReduce: pipes.reduce[4], splitTarget: pipes.splitTarget, groups: 1, rows: 4 };
}

function dispatchActivation(pass, group, T, N, pipelineObject) {
  pass.setPipeline(pipelineObject);
  pass.setBindGroup(0, group);
  const work = Math.ceil((T * N) / 256);
  pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
}

function beginTimed(encoder, querySet, index) {
  return encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
}

function beginUntimed(encoder) {
  return encoder.beginComputePass();
}

function encodeBaseline(pass, state, repetitions) {
  for (let i = 0; i < repetitions; i++) {
    dispatchMatmul(pass, state.generic.mm[4], state.generic.reduce[4], state.gateOp, state.T, state.splitTarget, 4, 1);
    if (!state.packed) dispatchMatmul(pass, state.generic.mm[4], state.generic.reduce[4], state.upOp, state.T, state.splitTarget, 4, 1);
    dispatchActivation(pass, state.activationGroup, state.T, state.N, state.activationPipeline);
  }
}

function encodeFused(pass, state, repetitions) {
  pass.setPipeline(state.fusedPipeline);
  pass.setBindGroup(0, state.fusedGroup);
  for (let i = 0; i < repetitions; i++) pass.dispatchWorkgroups(Math.ceil(state.N / 64), Math.ceil(state.T / 64));
}

function scratchSizes(info, tokens) {
  const maxT = Math.max(...tokens);
  const baselineN = info.packedN;
  const inputBytes = maxT * info.K * 4;
  const baselineBytes = maxT * baselineN * 4;
  const outputBytes = maxT * info.N * 4;
  const partBytes = 16;
  const queryBytes = tokens.length * GATED_SAMPLES * 2 * 2 * 8;
  const readBytes = Math.max(256, queryBytes);
  const buffers = [
    inputBytes,
    ...(info.packed ? [baselineBytes] : [outputBytes, outputBytes]),
    outputBytes,
    partBytes,
    baselineN * 4,
    ...Array(3 + tokens.length).fill(16),
    queryBytes,
    readBytes,
  ];
  return { maxT, baselineN, inputBytes, baselineBytes, outputBytes, partBytes, queryBytes, readBytes, buffers };
}

/** Estimate scratch for the actual allocations made for a set of measured token buckets. */
export function gatedScratchUsage(shape, tokens) {
  const info = shapeOnly(shape);
  if (!info.ok || !Array.isArray(tokens) || !tokens.length || tokens.some((token) => !GATED_TOKENS.includes(token))) return null;
  const { buffers } = scratchSizes(info, tokens);
  return { totalBytes: buffers.reduce((sum, size) => sum + size, 0), maxBufferBytes: Math.max(...buffers) };
}

function scratchFailure(device, info, tokens, maxScratch) {
  if (!tokens.length) return null;
  const scratch = gatedScratchUsage(info, tokens);
  if (!scratch || scratch.totalBytes > maxScratch) return "scratch-limit";
  if (scratch.maxBufferBytes > deviceLimit(device)) return "buffer-limit";
  return null;
}

async function measureShape(device, info, generic, fusedPipeline, fusedLayout, activationPipeline, activationLayout, options) {
  const started = now();
  const points = options.points.map(({ tokens }) => ({ tokens, rows: rowsPerThread(tokens) }));
  const eligibility = new Map(options.eligibility.map((point) => [point.tokens, point]));
  const pairs = GATED_TOKENS.map((tokens) => {
    const point = eligibility.get(tokens);
    return { tokens, eligible: !!point?.ok, reason: point?.reason, baseline: [], fused: [] };
  });
  if (now() >= options.deadline) return { points: pairs, elapsedMs: now() - started };
  const U = usage();
  const { baselineN, inputBytes, baselineBytes, outputBytes, partBytes, queryBytes, readBytes, buffers } = scratchSizes(info, points.map(({ tokens }) => tokens));
  if (buffers.reduce((sum, size) => sum + size, 0) > options.maxScratch || Math.max(...buffers) > deviceLimit(device)) throw new Error("gated tuning scratch exceeds device limits");

  const owned = [];
  let querySet;
  const make = (size, flags, label) => {
    const item = buffer(device, size, flags, label);
    owned.push(item);
    return item;
  };
  try {
    const X = make(inputBytes, U.STORAGE | U.COPY_DST, "gated.tune.X");
    const gateOutput = info.packed ? null : make(outputBytes, U.STORAGE, "gated.tune.gate");
    const upOutput = info.packed ? null : make(outputBytes, U.STORAGE, "gated.tune.up");
    const packedOutput = info.packed ? make(baselineBytes, U.STORAGE, "gated.tune.packed") : null;
    const fusedOutput = make(outputBytes, U.STORAGE, "gated.tune.fused");
    const bias = make(baselineN * 4, U.STORAGE | U.COPY_DST, "gated.tune.bias");
    const part = make(partBytes, U.STORAGE, "gated.tune.partials");
    const params = make(16, U.UNIFORM | U.COPY_DST, "gated.tune.params");
    const pairParams = make(16, U.UNIFORM | U.COPY_DST, "gated.tune.pair-params");
    const activationParams = make(16, U.UNIFORM | U.COPY_DST, "gated.tune.activation-params");
    const globals = points.map(({ tokens }) => make(16, U.UNIFORM | U.COPY_DST, `gated.tune.g${tokens}`));
    const read = make(readBytes, U.MAP_READ | U.COPY_DST, "gated.tune.read");
    const resolved = make(queryBytes, U.QUERY_RESOLVE | U.COPY_SRC, "gated.tune.resolved");
    const values = new Float32Array(inputBytes / 4);
    for (let i = 0; i < values.length; i++) {
      const magnitude = 0.125 + ((i * 37) % 876) / 1000;
      values[i] = Math.fround(i % 2 === 0 ? magnitude : -magnitude);
    }
    device.queue.writeBuffer(X, 0, values);
    device.queue.writeBuffer(bias, 0, new Float32Array(baselineN));
    device.queue.writeBuffer(params, 0, new Uint32Array([baselineN, info.K, 0, 0]));
    device.queue.writeBuffer(pairParams, 0, new Uint32Array([info.N, info.K, info.upOffset, info.activation]));
    device.queue.writeBuffer(activationParams, 0, new Uint32Array([info.N, 0, 0, 0]));
    for (let i = 0; i < points.length; i++) device.queue.writeBuffer(globals[i], 0, new Uint32Array([points[i].tokens, 0, 0, 0]));

    const states = points.map(({ tokens }, point) => {
      const gateOp = makeMatmulOp(device, generic, { ...info.gate, N: baselineN, K: info.K }, globals[point], params, X, info.packed ? packedOutput : gateOutput, bias, part);
      const upOp = info.packed ? null : makeMatmulOp(device, generic, { ...info.up, N: info.N, K: info.K }, globals[point], params, X, upOutput, bias, part);
      const activationResources = info.packed ? [globals[point], activationParams, packedOutput, fusedOutput] : [globals[point], activationParams, gateOutput, upOutput, fusedOutput];
      const activationGroup = makeGroup(device, activationLayout, activationResources);
      const fusedGroup = makeGroup(device, fusedLayout, [globals[point], pairParams, X, info.gate.buf, info.gate.sbuf, info.packed ? info.gate.buf : info.up.buf, info.packed ? info.gate.sbuf : info.up.sbuf, fusedOutput]);
      return {
        T: tokens,
        N: info.N,
        packed: info.packed,
        splitTarget: options.splitTarget,
        generic,
        gateOp,
        upOp,
        activation: activationGroup,
        activationGroup,
        activationPipeline,
        fusedPipeline,
        fusedGroup,
      };
    });

    if (now() < options.deadline) {
      const warmup = device.createCommandEncoder({ label: "gated.tune.warmup" });
      for (const state of states) {
        let pass = beginUntimed(warmup);
        encodeBaseline(pass, state, 1);
        pass.end();
        pass = beginUntimed(warmup);
        encodeFused(pass, state, 1);
        pass.end();
      }
      device.queue.submit([warmup.finish()]);
      await device.queue.onSubmittedWorkDone();
    } else {
      return { points: pairs, elapsedMs: now() - started };
    }

    querySet = device.createQuerySet({ type: "timestamp", count: queryBytes / 8, label: "gated.tune.timestamps" });
    const encoder = device.createCommandEncoder({ label: "gated.tune.measure" });
    const meta = [];
    let queryIndex = 0;
    for (let point = 0; point < states.length; point++) {
      for (let sample = 0; sample < GATED_SAMPLES; sample++) {
        if (now() >= options.deadline) break;
        const baselineFirst = (point + sample) % 2 === 0;
        for (const kind of (baselineFirst ? ["baseline", "fused"] : ["fused", "baseline"])) {
          const state = states[point];
          const pass = beginTimed(encoder, querySet, queryIndex);
          if (kind === "baseline") encodeBaseline(pass, state, GATED_REPETITIONS);
          else encodeFused(pass, state, GATED_REPETITIONS);
          pass.end();
          meta.push({ tokens: state.T, kind, queryIndex });
          queryIndex += 2;
        }
      }
      if (now() >= options.deadline) break;
    }
    if (queryIndex > 0) {
      const usedQueryBytes = queryIndex * 8;
      encoder.resolveQuerySet(querySet, 0, queryIndex, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, read, 0, usedQueryBytes);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ, 0, usedQueryBytes);
      const stamps = new BigInt64Array(read.getMappedRange(0, usedQueryBytes).slice(0));
      read.unmap();
      for (const item of meta) {
        const pair = pairs.find(({ tokens }) => tokens === item.tokens);
        pair[item.kind].push(Number(stamps[item.queryIndex + 1] - stamps[item.queryIndex]) / 1e6 / GATED_REPETITIONS);
      }
    }
    return { points: pairs, elapsedMs: now() - started };
  } finally {
    querySet?.destroy();
    for (const item of owned) item.destroy();
  }
}

async function isolated(device, work) {
  let value;
  let failure;
  let depth = 0;
  let scoped;
  const uncaptured = [];
  const onUncaptured = (event) => uncaptured.push(event.error?.message || String(event.error));
  try {
    device.addEventListener?.("uncapturederror", onUncaptured);
    for (const type of ["validation", "internal", "out-of-memory"]) {
      device.pushErrorScope(type);
      depth++;
    }
    value = await work();
  } catch (error) {
    failure = error;
  } finally {
    while (depth > 0) {
      depth--;
      try {
        const error = await device.popErrorScope();
        if (error && !scoped) scoped = error;
      } catch (error) {
        if (!failure) failure = error;
      }
    }
    device.removeEventListener?.("uncapturederror", onUncaptured);
  }
  if (failure || scoped || uncaptured.length) throw failure || scoped || new Error(uncaptured.join("; "));
  return value;
}

function defaultSelection(descriptors) {
  const selection = new Map();
  for (const descriptor of descriptors) {
    const info = shapeOnly(descriptor);
    if (!info.ok) continue;
    for (const tokens of GATED_TOKENS) selection.set(selectionKey(info, tokens), "generic");
  }
  return selection;
}

function fallback(reason, descriptors, options, extra = {}) {
  const generic = options.pipelines?.generic || null;
  return {
    pipeline: null,
    pipelines: { generic, fused: null },
    selection: defaultSelection(descriptors),
    diagnostics: {
      revision: GATED_TUNING_REVISION,
      method: "timestamp-query",
      tokens: [...GATED_TOKENS],
      samples: GATED_SAMPLES,
      repetitions: GATED_REPETITIONS,
      threshold: GATED_THRESHOLD,
      maxShapes: GATED_MAX_SHAPES,
      sharedBytes: GATED_SHARED_BYTES,
      reason,
      ...extra,
    },
  };
}

function normalizeDescriptors(value) {
  return Array.isArray(value) ? value : value ? [value] : [];
}

/**
 * Compile and automatically measure the F32 fused kernel for one or more loaded Q8 pair
 * descriptors. The descriptor is `{ gate, up, activation: 0|1|2, packed: boolean }`; `gate` and
 * `up` are entries from the runtime's GpuWeights map, including their GPU buffers and scales.
 */
export async function calibrateGateUp(device, wgsl, descriptorOrDescriptors, options = {}) {
  const descriptors = normalizeDescriptors(descriptorOrDescriptors);
  const base = {
    revision: GATED_TUNING_REVISION,
    method: "timestamp-query",
    tokens: [...GATED_TOKENS],
    samples: GATED_SAMPLES,
    repetitions: GATED_REPETITIONS,
    threshold: GATED_THRESHOLD,
    maxShapes: GATED_MAX_SHAPES,
    sharedBytes: GATED_SHARED_BYTES,
  };
  const automatic = options.kernel === undefined || options.kernel === "auto";
  if (!automatic) return fallback("automatic-only", descriptors, options, base);
  if (device?.features?.has?.("shader-f16")) return fallback("f16-existing-path", descriptors, options, base);
  if (!device?.features?.has?.("timestamp-query")) return fallback("no-timestamp-query", descriptors, options, base);
  if ((device.limits?.maxComputeWorkgroupStorageSize || 0) < GATED_SHARED_BYTES) return fallback("insufficient-workgroup-storage", descriptors, options, base);
  if (typeof wgsl !== "function") return fallback("missing-wgsl-source", descriptors, options, base);

  const suppliedGeneric = options.pipelines?.generic || null;
  const splitTarget = validInteger(suppliedGeneric?.splitTarget) ? suppliedGeneric.splitTarget : GATED_SPLIT_TARGET;
  const normalized = [];
  const rejected = [];
  const seen = new Set();
  for (const descriptor of descriptors) {
    const info = shapeDescriptor(descriptor);
    if (!info.ok) {
      rejected.push({ reason: info.reason });
      continue;
    }
    const key = `${info.N}:${info.K}:a${info.activation}:p${info.packed ? 1 : 0}`;
    if (seen.has(key) || normalized.length >= GATED_MAX_SHAPES) continue;
    seen.add(key);
    const eligibility = GATED_TOKENS.map((tokens) => {
      const point = { tokens, ...splitInfo(info, tokens, splitTarget) };
      if (point.ok) {
        const reason = scratchFailure(device, info, [tokens], GATED_MAX_SCRATCH);
        if (reason) return { ...point, ok: false, reason };
      }
      return point;
    });
    let eligiblePoints = eligibility.filter((point) => point.ok);
    while (eligiblePoints.length > 1) {
      const reason = scratchFailure(device, info, eligiblePoints.map((point) => point.tokens), GATED_MAX_SCRATCH);
      if (!reason) break;
      const largest = Math.max(...eligiblePoints.map((point) => point.tokens));
      const excluded = eligibility.find((point) => point.tokens === largest);
      excluded.ok = false;
      excluded.reason = reason;
      eligiblePoints = eligibility.filter((point) => point.ok);
    }
    if (!eligiblePoints.length) {
      rejected.push({ shape: key, reason: eligibility.find((point) => !point.ok)?.reason, eligibility });
      continue;
    }
    normalized.push({ descriptor, info, key, points: eligiblePoints, eligibility });
  }
  const selection = defaultSelection(descriptors);
  if (!normalized.length) return fallback("no-eligible-shapes", descriptors, options, { ...base, splitTarget, rejected });

  const started = now();
  const deadline = started + GATED_DEADLINE_MS;
  let generic = suppliedGeneric;
  let fused = null;
  const records = [];
  try {
    await isolated(device, async () => {
      if (!generic) generic = await buildF32Matmul(device, wgsl, splitTarget);
      if (now() >= deadline) return;
      const built = await buildFused(device, wgsl, splitTarget);
      fused = built.pipeline;
      const fusedLayout = built.layout;
      if (generic?.f16 === true) throw new Error("gated tuning requires an F32 generic baseline");
      const activationCache = new Map();
      for (const item of normalized) {
        if (now() >= deadline) break;
        const activationKey = `${item.info.activation}:${item.info.packed ? 1 : 0}`;
        if (!activationCache.has(activationKey)) activationCache.set(activationKey, await buildActivation(device, wgsl, item.info.activation, item.info.packed));
        if (now() >= deadline) break;
        const activation = activationCache.get(activationKey);
        const measured = await measureShape(device, item.info, generic, fused, fusedLayout, activation.pipeline, activation.layout, {
          splitTarget,
          maxScratch: GATED_MAX_SCRATCH,
          deadline,
          points: item.points,
          eligibility: item.eligibility,
        });
        const decision = selectGateUpCandidate({ points: measured.points });
        const record = { shape: item.key, name: item.descriptor.gate.info.name, packed: item.info.packed, activation: item.info.activation, ...decision, elapsedMs: measured.elapsedMs };
        records.push(record);
        for (const point of decision.points) {
          if (point.choice === "fused") selection.set(selectionKey(item.info, point.tokens), "fused");
        }
      }
    });
    const anyFused = records.some((record) => record.choice === "fused");
    return {
      pipeline: anyFused ? fused : null,
      pipelines: { generic, fused },
      selection,
      diagnostics: { ...base, reason: anyFused ? "stable-win" : "no-stable-win", splitTarget, elapsedMs: now() - started, truncated: now() >= deadline, rejected, records },
    };
  } catch (error) {
    const fallbackOptions = { ...options, pipelines: { ...options.pipelines, generic: generic || suppliedGeneric } };
    return fallback("benchmark-failed", descriptors, fallbackOptions, { ...base, splitTarget, elapsedMs: now() - started, rejected, records, error: String(error?.message || error) });
  }
}
