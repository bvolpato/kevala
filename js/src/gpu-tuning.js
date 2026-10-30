// Load-time selector for the generic and wide matmul kernels. There is intentionally no
// persistent cache: Firefox may expose an anonymous adapter, so a profile can belong to another
// physical GPU on the next load.

import { bufferUsage, dispatchMatmul, inErrorScopes, matmulPipelines, mmSplits, rowsPerThread, scratchBuffer } from "./gpu.js";
import { median } from "./stats.js";

export const GPU_TUNING_REVISION = 2;
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

/** Return distinct loaded projection shapes of one precision in calibration priority order. */
export function matmulShapes(weights, maxShapes = MAX_SHAPES, dtype = "q8") {
  const entries = weights?.tensors instanceof Map ? [...weights.tensors.values()] : [];
  const candidates = entries.flatMap((entry) => {
    const info = entry.info;
    const shape = info?.shape;
    const suffix = String(info?.name || "").split(".").at(-1);
    if (info?.dtype !== dtype || !Array.isArray(shape) || shape.length !== 2 || !RANK.has(suffix) || !entry.buf || dtype === "q8" && !entry.sbuf) return [];
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

async function pipelinesFor(device, wgsl, kind, supplied, dtype) {
  if (supplied?.[kind]) return supplied[kind];
  return matmulPipelines(device, wgsl, kind === "wide" ? "matmul_wide" : "matmul", dtype);
}

function operation(device, pipes, shape, globals, params, x, y, bias, part, rows) {
  const resources = [globals, params, x, shape.buf, shape.sbuf || bias, bias, y, part];
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

const deadlineError = () => Object.assign(new Error("matmul calibration deadline exceeded"), { code: "MATMUL_DEADLINE" });

async function mapWithinDeadline(read, bytes, deadline) {
  const remaining = deadline - now();
  if (remaining <= 0) throw deadlineError();
  let timer;
  try {
    await Promise.race([
      read.mapAsync(GPUMapMode.READ, 0, bytes),
      new Promise((_, reject) => { timer = setTimeout(() => {
        reject(deadlineError());
        read.unmap();
      }, remaining); }),
    ]);
    if (now() >= deadline) {
      read.unmap();
      throw deadlineError();
    }
  } finally { clearTimeout(timer); }
}

export function compareMatmulOutputs(baseline, candidate) {
  if (baseline.length !== candidate.length) return { ok: false, reason: "length-mismatch" };
  let maxAbs = 0;
  for (let index = 0; index < baseline.length; index++) {
    const expected = baseline[index], actual = candidate[index];
    const difference = Math.abs(actual - expected);
    if (!Number.isFinite(expected) || !Number.isFinite(actual) || difference > 0.0005 + 0.0001 * Math.max(1, Math.abs(expected))) {
      return { ok: false, reason: "numerical-guard-failed", index, maxAbs: Number.isFinite(difference) ? difference : null };
    }
    maxAbs = Math.max(maxAbs, difference);
  }
  return { ok: true, maxAbs };
}

async function measureShape(device, shape, pipes, dtype, deadline) {
  const U = bufferUsage();
  const maxT = Math.max(...MATMUL_TOKENS);
  const points = MATMUL_TOKENS.map((tokens) => ({ tokens, rows: rowsPerThread(tokens) }));
  const xBytes = maxT * shape.K * 4;
  const yBytes = maxT * shape.N * 4;
  const partBytes = Math.max(...points.flatMap(({ tokens, rows }) => MATMUL_KERNELS.map((kind) => {
    const p = pipes[kind];
    const splits = mmSplits(tokens, shape.N, shape.K, p.splitTarget, 16 * rows, 64 * p.groups);
    return splits > 1 ? splits * tokens * shape.N * 4 : 16;
  })));
  const guardBytes = dtype === "q8" ? 0 : points.reduce((sum, { tokens }) => sum + tokens * shape.N * 8, 0);
  const scratchBytes = xBytes + yBytes + partBytes + shape.N * 4 + guardBytes;
  const limit = Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
  const base = (tokens) => ({ N: shape.N, K: shape.K, tokens, key: matmulTuningKey(shape.N, shape.K, tokens), kernel: "generic", reason: "scratch-limit" });
  if (scratchBytes > MAX_SCRATCH || Math.max(xBytes, yBytes, partBytes, shape.N * 4, guardBytes) > limit) return { points: points.map(({ tokens }) => base(tokens)), elapsedMs: 0 };

  const owned = [];
  let querySet;
  try {
    const make = (size, flags, label) => (owned.push(scratchBuffer(device, size, flags, label)), owned.at(-1));
    const x = make(xBytes, U.STORAGE | U.COPY_DST, `${shape.name}.tune.x`);
    const y = make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, `${shape.name}.tune.y`);
    const bias = make(shape.N * 4, U.STORAGE | U.COPY_DST, `${shape.name}.tune.bias`);
    const part = make(partBytes, U.STORAGE, `${shape.name}.tune.part`);
    const params = make(16, U.UNIFORM | U.COPY_DST, `${shape.name}.tune.params`);
    const globals = points.map(({ tokens }) => make(16, U.UNIFORM | U.COPY_DST, `${shape.name}.tune.g${tokens}`));
    const input = new Float32Array(xBytes / 4);
    if (guardBytes) for (let index = 0; index < input.length; index++) input[index] = (((Math.imul(index + 1, 1103515245) >>> 16) % 257) - 128) / 4096;
    device.queue.writeBuffer(x, 0, input);
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
    const guardRead = guardBytes ? make(guardBytes, U.MAP_READ | U.COPY_DST, `${shape.name}.tune.guard`) : null;
    const encoder = device.createCommandEncoder({ label: `${shape.name}.tune` });
    let guardOffset = 0;
    for (let point = 0; point < operations.length; point++) for (const candidate of operations[point]) {
      encodePass(encoder, candidate, points[point].tokens, null, 0);
      if (guardRead) {
        const bytes = points[point].tokens * shape.N * 4;
        encoder.copyBufferToBuffer(y, 0, guardRead, guardOffset, bytes);
        guardOffset += bytes;
      }
    }
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
    await mapWithinDeadline(read, queryCount * 8, deadline);
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0));
    read.unmap();
    const checks = [];
    if (guardRead) {
      await mapWithinDeadline(guardRead, guardBytes, deadline);
      const output = new Float32Array(guardRead.getMappedRange(0, guardBytes));
      let offset = 0;
      for (const { tokens } of points) {
        const length = tokens * shape.N;
        checks.push(compareMatmulOutputs(output.subarray(offset, offset + length), output.subarray(offset + length, offset + 2 * length)));
        offset += 2 * length;
      }
      guardRead.unmap();
    }
    const timings = points.map(() => ({ generic: [], wide: [] }));
    for (const { point, index, queryIndex: i } of meta) timings[point][MATMUL_KERNELS[index]].push(Number(stamps[i + 1] - stamps[i]) / 1e6 / REPETITIONS);
    return { points: points.map(({ tokens }, point) => ({ ...base(tokens),
      ...(checks[point]?.ok === false ? { reason: "numerical-guard-failed" } : selectMatmulKernel(timings[point])),
      ...(checks[point] ? { numericalCheck: checks[point] } : {}),
    })), elapsedMs: now() - started };
  } finally {
    for (const item of owned) item.destroy();
    querySet?.destroy();
  }
}

/** Build and, when possible, measure both candidates on up to six real loaded projection shapes. */
export async function calibrateMatmul(device, wgsl, weights, options = {}) {
  const kernel = options.kernel || "auto";
  const dtype = options.dtype || "q8";
  if (!["q8", "bf16", "f32"].includes(dtype)) throw new Error(`unsupported GPU matrix dtype ${dtype}`);
  if (!["auto", ...MATMUL_KERNELS].includes(kernel)) throw new Error('kernel must be "auto", "generic", or "wide"');
  if (kernel === "wide" && dtype === "q8" && !device.features?.has?.("shader-f16")) throw new Error('wide matmul requires the "shader-f16" feature');
  const shapes = matmulShapes(weights, MAX_SHAPES, dtype);
  const selection = new Map();
  const defaults = () => { for (const shape of shapes) for (const T of MATMUL_TOKENS) selection.set(matmulTuningKey(shape.N, shape.K, T), kernel === "wide" ? "wide" : "generic"); };
  defaults();
  const diagnostics = { revision: GPU_TUNING_REVISION, dtype, method: kernel === "auto" ? "generic" : "override", points: [], elapsedMs: 0, truncated: false };
  const started = now();
  const generic = await pipelinesFor(device, wgsl, "generic", options.pipelines, dtype);
  if (kernel === "generic") return { pipelines: { generic, wide: options.pipelines?.wide || null }, selection, diagnostics };
  if (kernel === "auto" && (dtype === "q8" && !device.features?.has?.("shader-f16") || !device.features?.has?.("timestamp-query"))) {
    diagnostics.reason = dtype === "q8" && !device.features?.has?.("shader-f16") ? "no-shader-f16" : "no-timestamp-query";
    diagnostics.elapsedMs = now() - started;
    return { pipelines: { generic, wide: options.pipelines?.wide || null }, selection, diagnostics };
  }
  let wide;
  try {
    wide = await inErrorScopes(device, () => pipelinesFor(device, wgsl, "wide", options.pipelines, dtype));
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
    if (now() >= deadline) {
      diagnostics.truncated = true;
      diagnostics.reason = "deadline";
      break;
    }
    const record = { shape: `${shape.N}x${shape.K}`, name: shape.name, points: [] };
    try {
      const result = await inErrorScopes(device, () => measureShape(device, shape, pipes, dtype, deadline));
      record.points = result.points.map((point) => ({ ...point, elapsedMs: result.elapsedMs }));
      for (const point of result.points) selection.set(point.key, point.kernel);
    } catch (error) {
      if (error?.code === "MATMUL_DEADLINE") {
        diagnostics.reason = record.reason = "deadline";
      } else {
        selection.clear();
        defaults();
        diagnostics.reason = "calibration-failed";
        for (const prior of diagnostics.points) for (const point of prior.points) {
          point.kernel = "generic";
          point.reason = "aborted-after-calibration-failure";
        }
        record.reason = String(error?.message || error);
      }
      diagnostics.truncated = true;
    }
    diagnostics.points.push(record);
    if (!diagnostics.truncated && now() >= deadline) {
      diagnostics.truncated = true;
      diagnostics.reason = "deadline";
    }
    if (diagnostics.truncated) break;
  }
  diagnostics.elapsedMs = now() - started;
  return { pipelines: pipes, selection, diagnostics };
}
