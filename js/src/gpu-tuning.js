// Load-time selector for the generic and wide matmul kernels. There is intentionally no
// persistent cache: Firefox may expose an anonymous adapter, so a profile can belong to another
// physical GPU on the next load.

import { dispatchMatmul, matmulPipelines, mmSplits, rowsPerThread } from "./gpu.js";

export const GPU_TUNING_REVISION = 1;
export const MATMUL_TOKENS = [16, 32, 48, 128, 512];
export const MATMUL_KERNELS = ["generic", "wide"];

const MAX_SHAPES = 6;
const SAMPLES = 3;
const REPETITIONS = 4;
const THRESHOLD = 0.05;
const DEADLINE_MS = 1500;
const MAX_SCRATCH = 128 * 1024 * 1024;
const RANK = new Map([
  ["gate_up", 0], ["qkvz", 1], ["down", 2], ["qkv", 3], ["out", 4], ["o", 5],
  ["gate", 0], ["up", 0], ["q", 1], ["k", 3], ["v", 3], ["ple_gate", 6], ["ple_out", 7], ["proj", 8],
  ["wi", 0], ["wqkv", 1], ["wo2", 2], ["wo", 3], ["in_proj", 4], ["out_proj", 5], ["lin1", 6], ["lin2", 7],
]);
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

export function matmulTokenBucket(tokens) {
  return tokens <= 16 ? "r1" : tokens <= 32 ? "r2" : tokens < 49 ? "r3" : tokens < 256 ? "mid" : "long";
}

export function matmulTuningKey(N, K, tokens) {
  return `${N}:${K}:${matmulTokenBucket(tokens)}`;
}

export function selectedMatmulKernel(selection, N, K, tokens) {
  return selection?.get(matmulTuningKey(N, K, tokens)) || "generic";
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Wide needs a stable >=5% paired win. Any incomplete or noisy measurement stays generic. */
export function selectMatmulKernel({ generic, wide }, { threshold = THRESHOLD } = {}) {
  const valid = (v) => Array.isArray(v) && v.length >= SAMPLES && v.every((x) => Number.isFinite(x) && x > 0);
  if (!valid(generic) || !valid(wide) || generic.length !== wide.length) return { kernel: "generic", reason: "insufficient-samples", genericMs: null, wideMs: null, benefit: null, stable: false };
  const genericMs = median(generic);
  const wideMs = median(wide);
  const benefit = (genericMs - wideMs) / genericMs;
  const need = Math.ceil(generic.length * 2 / 3);
  const wideWins = generic.reduce((n, x, i) => n + ((x - wide[i]) / x >= threshold ? 1 : 0), 0);
  const genericWins = wide.reduce((n, x, i) => n + ((x - generic[i]) / x >= threshold ? 1 : 0), 0);
  const stable = generic.every((x, i) => wide[i] < x) && wideWins >= need;
  const kernel = stable && benefit >= threshold ? "wide" : "generic";
  const reason = kernel === "wide" ? "stable-wide-win" : genericWins >= need ? "stable-generic-win" : stable ? "below-threshold" : "noisy";
  return { kernel, reason, genericMs, wideMs, benefit, stable };
}

/** Return distinct loaded Q8 projection shapes in calibration priority order. */
export function matmulShapes(weights, maxShapes = MAX_SHAPES) {
  const entries = weights?.tensors instanceof Map ? [...weights.tensors.values()] : [];
  const candidates = entries.flatMap((entry) => {
    const info = entry.info;
    const shape = info?.shape;
    const suffix = String(info?.name || "").split(".").at(-1);
    if (info?.dtype !== "q8" || !Array.isArray(shape) || shape.length !== 2 || !RANK.has(suffix) || !entry.buf || !entry.sbuf) return [];
    const [N, K] = shape.map(Number);
    return Number.isSafeInteger(N) && Number.isSafeInteger(K) && N > 0 && K > 0 && K % 32 === 0 ? [{ ...entry, name: info.name, suffix, N, K }] : [];
  });
  candidates.sort((a, b) => RANK.get(a.suffix) - RANK.get(b.suffix) || b.N * b.K - a.N * a.K);
  const seen = new Set();
  return candidates.filter((shape) => {
    const key = `${shape.N}:${shape.K}`;
    if (seen.has(key) || seen.size >= maxShapes) return false;
    seen.add(key);
    return true;
  });
}

function usage() {
  if (!globalThis.GPUBufferUsage) throw new Error("WebGPU buffer usage constants are unavailable");
  return globalThis.GPUBufferUsage;
}

function buffer(device, size, flags, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage: flags });
}

async function pipelinesFor(device, wgsl, kind, supplied) {
  if (supplied?.[kind]) return supplied[kind];
  return matmulPipelines(device, wgsl, kind === "wide" ? "matmul_wide" : "matmul");
}

function operation(device, pipes, shape, globals, params, x, y, bias, part, rows) {
  const resources = [globals, params, x, shape.buf, shape.sbuf, bias, y, part];
  return {
    N: shape.N,
    K: shape.K,
    group: device.createBindGroup({ layout: pipes.layout, entries: resources.map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
    reduce: device.createBindGroup({ layout: pipes.reduceLayout, entries: [globals, params, part, bias, y].map((b, i) => ({ binding: i, resource: { buffer: b } })) }),
    pMatmul: pipes.mm[rows],
    pReduce: pipes.reduce[rows],
    splitTarget: pipes.splitTarget,
    groups: pipes.groups,
    rows,
  };
}

function encodePass(encoder, op, tokens, querySet, queryIndex, repetitions = 1) {
  const pass = querySet ? encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } }) : encoder.beginComputePass();
  for (let i = 0; i < repetitions; i++) dispatchMatmul(pass, op.pMatmul, op.pReduce, op, tokens, op.splitTarget, op.rows, op.groups);
  pass.end();
}

async function measureShape(device, shape, pipes) {
  const U = usage();
  const maxT = Math.max(...MATMUL_TOKENS);
  const points = MATMUL_TOKENS.map((tokens) => ({ tokens, rows: rowsPerThread(tokens) }));
  const xBytes = maxT * shape.K * 4;
  const yBytes = maxT * shape.N * 4;
  const partBytes = Math.max(...points.flatMap(({ tokens, rows }) => MATMUL_KERNELS.map((kind) => {
    const p = pipes[kind];
    const splits = mmSplits(tokens, shape.N, shape.K, p.splitTarget, 16 * rows, 64 * p.groups);
    return splits > 1 ? splits * tokens * shape.N * 4 : 16;
  })));
  const scratchBytes = xBytes + yBytes + partBytes + shape.N * 4;
  const limit = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  const base = (tokens) => ({ N: shape.N, K: shape.K, tokens, key: matmulTuningKey(shape.N, shape.K, tokens), kernel: "generic", reason: "scratch-limit" });
  if (scratchBytes > MAX_SCRATCH || Math.max(xBytes, yBytes, partBytes, shape.N * 4) > limit) return { points: points.map(({ tokens }) => base(tokens)), elapsedMs: 0 };

  const owned = [];
  let querySet;
  try {
    const make = (size, flags, label) => (owned.push(buffer(device, size, flags, label)), owned.at(-1));
    const x = make(xBytes, U.STORAGE | U.COPY_DST, `${shape.name}.tune.x`);
    const y = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, `${shape.name}.tune.y`);
    const bias = make(shape.N * 4, U.STORAGE | U.COPY_DST, `${shape.name}.tune.bias`);
    const part = make(partBytes, U.STORAGE, `${shape.name}.tune.part`);
    const params = make(16, U.UNIFORM | U.COPY_DST, `${shape.name}.tune.params`);
    const globals = points.map(({ tokens }) => make(16, U.UNIFORM | U.COPY_DST, `${shape.name}.tune.g${tokens}`));
    device.queue.writeBuffer(x, 0, new Float32Array(xBytes / 4));
    device.queue.writeBuffer(bias, 0, new Float32Array(shape.N));
    device.queue.writeBuffer(params, 0, new Uint32Array([shape.N, shape.K, 0, 0]));
    const operations = points.map(({ tokens, rows }, point) => {
      device.queue.writeBuffer(globals[point], 0, new Uint32Array([tokens, 0, 0, 0]));
      return MATMUL_KERNELS.map((kind) => operation(device, pipes[kind], shape, globals[point], params, x, y, bias, part, rows));
    });
    const queryCount = points.length * SAMPLES * MATMUL_KERNELS.length * 2;
    querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `${shape.name}.tune.timestamps` });
    const resolved = make(queryCount * 8, U.QUERY_RESOLVE | U.COPY_SRC, `${shape.name}.tune.resolved`);
    const read = make(Math.max(256, queryCount * 8), U.MAP_READ | U.COPY_DST, `${shape.name}.tune.read`);
    const encoder = device.createCommandEncoder({ label: `${shape.name}.tune` });
    for (let point = 0; point < operations.length; point++) for (const candidate of operations[point]) encodePass(encoder, candidate, points[point].tokens, null, 0);
    let queryIndex = 0;
    const meta = [];
    for (let point = 0; point < points.length; point++) for (let sample = 0; sample < SAMPLES; sample++) {
      const order = sample % 2 === 0 ? [0, 1] : [1, 0];
      for (const index of order) {
        encodePass(encoder, operations[point][index], points[point].tokens, querySet, queryIndex, REPETITIONS);
        meta.push({ point, index, queryIndex });
        queryIndex += 2;
      }
    }
    encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8);
    const started = now();
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ, 0, queryCount * 8);
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0));
    read.unmap();
    const timings = points.map(() => ({ generic: [], wide: [] }));
    for (const { point, index, queryIndex: i } of meta) timings[point][MATMUL_KERNELS[index]].push(Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS);
    return { points: points.map(({ tokens }, point) => ({ ...base(tokens), ...selectMatmulKernel(timings[point]) })), elapsedMs: now() - started };
  } finally {
    for (const item of owned) item.destroy();
    querySet?.destroy();
  }
}

async function isolated(device, work) {
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let value;
  let failure;
  try {
    value = await work();
  } catch (error) {
    failure = error;
  }
  const scopes = await Promise.allSettled([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  const scoped = scopes.map((s) => s.status === "fulfilled" ? s.value : s.reason).find(Boolean);
  if (failure || scoped) throw failure || scoped;
  return value;
}

/** Build and, when possible, measure both candidates on up to six real loaded projection shapes. */
export async function calibrateMatmul(device, wgsl, weights, options = {}) {
  const kernel = options.kernel || "auto";
  if (!["auto", ...MATMUL_KERNELS].includes(kernel)) throw new Error('kernel must be "auto", "generic", or "wide"');
  if (kernel === "wide" && !device.features?.has?.("shader-f16")) throw new Error('wide matmul requires the "shader-f16" feature');
  const shapes = matmulShapes(weights);
  const selection = new Map();
  const defaults = () => { for (const shape of shapes) for (const T of MATMUL_TOKENS) selection.set(matmulTuningKey(shape.N, shape.K, T), kernel === "wide" ? "wide" : "generic"); };
  defaults();
  const diagnostics = { revision: GPU_TUNING_REVISION, method: kernel === "auto" ? "generic" : "override", points: [], elapsedMs: 0, truncated: false };
  const started = now();
  const generic = await pipelinesFor(device, wgsl, "generic", options.pipelines);
  if (kernel === "generic") return { pipelines: { generic, wide: options.pipelines?.wide || null }, selection, diagnostics };
  if (kernel === "auto" && (!device.features?.has?.("shader-f16") || !device.features?.has?.("timestamp-query"))) {
    diagnostics.reason = !device.features?.has?.("shader-f16") ? "no-shader-f16" : "no-timestamp-query";
    diagnostics.elapsedMs = now() - started;
    return { pipelines: { generic, wide: options.pipelines?.wide || null }, selection, diagnostics };
  }
  let wide;
  try {
    wide = await isolated(device, () => pipelinesFor(device, wgsl, "wide", options.pipelines));
  } catch (error) {
    if (kernel === "wide") throw error;
    diagnostics.reason = "wide-pipeline-failed";
    diagnostics.error = String(error?.message || error);
    diagnostics.elapsedMs = now() - started;
    return { pipelines: { generic, wide: null }, selection, diagnostics };
  }
  if (kernel === "wide") return { pipelines: { generic, wide }, selection, diagnostics };
  const pipes = { generic, wide };
  diagnostics.method = "timestamp-query";
  const deadline = started + DEADLINE_MS;
  for (const shape of shapes) {
    if (now() >= deadline && diagnostics.points.length) {
      diagnostics.truncated = true;
      break;
    }
    const record = { shape: `${shape.N}x${shape.K}`, name: shape.name, points: [] };
    try {
      const result = await isolated(device, () => measureShape(device, shape, pipes));
      record.points = result.points.map((point) => ({ ...point, elapsedMs: result.elapsedMs }));
      for (const point of result.points) selection.set(point.key, point.kernel);
    } catch (error) {
      selection.clear();
      defaults();
      diagnostics.reason = "calibration-failed";
      for (const prior of [...diagnostics.points, record]) for (const point of prior.points) point.kernel = "generic";
      record.reason = String(error?.message || error);
      diagnostics.truncated = true;
    }
    diagnostics.points.push(record);
    if (diagnostics.truncated || now() >= deadline) diagnostics.truncated = true;
    if (diagnostics.truncated) break;
  }
  diagnostics.elapsedMs = now() - started;
  return { pipelines: pipes, selection, diagnostics };
}
