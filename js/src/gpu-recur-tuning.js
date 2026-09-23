// Select the recurrent lane width for the current WebGPU adapter. The workloads are deliberately
// fixed uncached 128- and 512-token segments, enough to expose occupancy without depending on model
// inputs or mutating a cached state.

export const RECUR_TUNING_REVISION = 2;

const TOKENS = [128, 512];
const SAMPLES = 3;
const REPETITIONS = 4;
const THRESHOLD = 0.05;
const MAX_VARIANTS = 3;
const MAX_SCRATCH = 64 * 1024 * 1024;
const NO_STATE = 0xffffffff;
const HEAD_DIM = 128;
const SUPPORTED_LANES = [4, 8, 16];
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function validSamples(values) {
  return Array.isArray(values) && values.length === SAMPLES && values.every((value) => Number.isFinite(value) && value > 0);
}

function zTiles(lanes) {
  return lanes / 2;
}

function pointResult(candidate, point, threshold) {
  const baselineMs = point?.pairs?.[candidate.lanes]?.baseline;
  const candidateMs = point?.pairs?.[candidate.lanes]?.candidate;
  if (!validSamples(baselineMs) || !validSamples(candidateMs)) {
    return { tokens: point?.tokens ?? null, reason: "insufficient-samples", stable: false };
  }
  const baselineMedian = median(baselineMs);
  const candidateMedian = median(candidateMs);
  const benefits = baselineMs.map((value, index) => (value - candidateMs[index]) / value);
  const wins = benefits.filter((benefit) => benefit >= threshold).length;
  const normalizedRatio = candidateMedian / baselineMedian;
  const benefit = 1 - normalizedRatio;
  const allFaster = baselineMs.every((value, index) => candidateMs[index] < value);
  const stable = allFaster && wins >= Math.ceil(SAMPLES * 2 / 3) && benefit >= threshold;
  const reason = stable ? "stable-win" : !allFaster ? "regression" : wins < Math.ceil(SAMPLES * 2 / 3) ? "noisy" : "below-threshold";
  return {
    tokens: point.tokens,
    baselineMs,
    candidateMs,
    baselineMedianMs: baselineMedian,
    candidateMedianMs: candidateMedian,
    benefits,
    benefit,
    normalizedRatio,
    wins,
    stable,
    reason,
  };
}

/** Pick a lane width from paired T128 and T512 timings. Four lanes is always the fallback. */
export function selectRecurrenceCandidate(measurements, { threshold = THRESHOLD } = {}) {
  const candidates = (Array.isArray(measurements?.candidates) ? measurements.candidates : []).filter((candidate) => SUPPORTED_LANES.includes(candidate?.lanes) && candidate.lanes !== 4);
  const points = Array.isArray(measurements?.points) ? measurements.points : [];
  const expectedPoints = points.length === TOKENS.length && points.every((point, index) => point?.tokens === TOKENS[index]);
  const records = [];
  let winner = null;
  for (const candidate of candidates) {
    const pointResults = points.map((point) => pointResult(candidate, point, threshold));
    const valid = expectedPoints && pointResults.every((point) => Number.isFinite(point.normalizedRatio));
    const stable = valid && pointResults.every((point) => point.stable);
    const normalizedRatio = valid ? Math.exp(pointResults.reduce((sum, point) => sum + Math.log(point.normalizedRatio), 0) / pointResults.length) : null;
    const benefit = normalizedRatio === null ? null : 1 - normalizedRatio;
    const reason = stable ? "stable-win-both-points" : pointResults.some((point) => point.reason === "regression") ? "point-regression" : pointResults.some((point) => point.reason === "insufficient-samples") ? "insufficient-samples" : pointResults.some((point) => point.reason === "below-threshold") ? "below-threshold" : "noisy";
    const record = {
      lanes: candidate.lanes,
      zTiles: zTiles(candidate.lanes),
      points: pointResults,
      normalizedRatio,
      benefit,
      stable,
      reason,
    };
    records.push(record);
    if (stable && (!winner || normalizedRatio < winner.normalizedRatio)) winner = record;
  }
  return {
    choice: winner?.lanes || 4,
    zTiles: winner ? zTiles(winner.lanes) : 2,
    winner,
    candidates: records,
  };
}

function fallback(reason, extra = {}) {
  return {
    choice: 4,
    zTiles: 2,
    diagnostics: {
      revision: RECUR_TUNING_REVISION,
      method: "timestamp-query",
      tokens: [...TOKENS],
      samples: SAMPLES,
      repetitions: REPETITIONS,
      threshold: THRESHOLD,
      reason,
      ...extra,
    },
  };
}

function usage() {
  if (!globalThis.GPUBufferUsage) throw new Error("WebGPU buffer usage constants are unavailable");
  return globalThis.GPUBufferUsage;
}

function buffer(device, size, flags, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage: flags });
}

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
}

function configValues(config) {
  const linKeyHeads = Number(config?.lin_key_heads);
  const linHeads = Number(config?.lin_heads);
  if (!Number.isInteger(linKeyHeads) || !Number.isInteger(linHeads) || linKeyHeads <= 0 || linHeads <= 0 || linKeyHeads > 128 || linHeads > 128 || linHeads % linKeyHeads) return null;
  return { linKeyHeads, linHeads };
}

function makeInput(linKeyHeads, linHeads, tokens) {
  const qk = linKeyHeads * HEAD_DIM;
  const out = linHeads * HEAD_DIM;
  const dim = 2 * qk + out;
  const C = new Float32Array(tokens * dim);
  const AB = new Float32Array(tokens * 2 * linHeads);
  const next = random(0x4b455641 ^ (linKeyHeads << 8) ^ linHeads ^ tokens);
  for (let i = 0; i < C.length; i++) C[i] = Math.fround((next() - 0.5) * 0.08);
  for (let t = 0; t < tokens; t++) for (let h = 0; h < linHeads; h++) {
    const base = t * 2 * linHeads + h;
    AB[base] = Math.fround(0.9 + next() * 0.1);
    AB[base + linHeads] = Math.fround(next());
  }
  return { C, AB, qk, out, dim };
}

async function scoped(device, work) {
  if (!device.pushErrorScope || !device.popErrorScope) return work();
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
  const error = scopes.map((result) => result.status === "fulfilled" ? result.value : result.reason).find(Boolean);
  if (failure || error) throw failure || error;
  return value;
}

function encodePass(encoder, candidate, bindGroup, linHeads, querySet, queryIndex, repetitions) {
  const pass = querySet
    ? encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } })
    : encoder.beginComputePass();
  pass.setPipeline(candidate.pipeline);
  pass.setBindGroup(0, bindGroup);
  for (let i = 0; i < repetitions; i++) pass.dispatchWorkgroups(1, linHeads, zTiles(candidate.lanes));
  pass.end();
}

function deviceLimit(device) {
  return Math.min(device.limits?.maxStorageBufferBindingSize || Infinity, device.limits?.maxBufferSize || Infinity);
}

async function resolveCandidatePipelines(device, candidates, onResolved) {
  const resolved = [];
  for (const candidate of candidates) {
    const pipeline = typeof candidate.pipeline === "function" ? await candidate.pipeline(device) : candidate.pipeline;
    if (!pipeline || typeof pipeline.getBindGroupLayout !== "function") throw new Error(`recurrence pipeline for ${candidate.lanes} lanes is unavailable`);
    const value = { ...candidate, pipeline };
    resolved.push(value);
    onResolved?.(value);
  }
  return resolved;
}

async function benchmark(device, candidates, config, tokens) {
  const U = usage();
  const { linKeyHeads, linHeads } = config;
  const input = makeInput(linKeyHeads, linHeads, tokens);
  const segment = new Uint32Array([0, tokens, NO_STATE, 0, 0, 0, 0, 0]);
  const globals = new Uint32Array([tokens, 0, 1, 0]);
  const cBytes = input.C.byteLength;
  const abBytes = input.AB.byteLength;
  const coreBytes = tokens * input.out * 4;
  const contenders = candidates.filter((candidate) => candidate.lanes !== 4);
  // Each paired sample has two passes, and each timestamped pass writes begin and end.
  const queryCount = contenders.length * SAMPLES * 4;
  const queryBytes = queryCount * 8;
  const readBytes = Math.max(256, queryBytes);
  const scratchBytes = cBytes + abBytes + segment.byteLength + 16 + coreBytes + 16 + queryBytes + readBytes;
  const limit = deviceLimit(device);
  const uniformLimit = device.limits?.maxUniformBufferBindingSize || Infinity;
  if (scratchBytes > MAX_SCRATCH || Math.max(cBytes, abBytes, segment.byteLength, 16, coreBytes, queryBytes, readBytes) > limit || 16 > uniformLimit) throw new Error("recurrence tuning scratch exceeds device limits");
  const owned = [];
  const uncaptured = [];
  const onUncaptured = (event) => uncaptured.push(event.error?.message || String(event.error));
  let querySet;
  let read;
  device.addEventListener?.("uncapturederror", onUncaptured);
  try {
    const make = (size, flags, label) => (owned.push(buffer(device, size, flags, label)), owned.at(-1));
    const prefix = `recur.tune.t${tokens}`;
    const C = make(cBytes, U.STORAGE | U.COPY_DST, `${prefix}.C`);
    const AB = make(abBytes, U.STORAGE | U.COPY_DST, `${prefix}.AB`);
    const segs = make(segment.byteLength, U.STORAGE | U.COPY_DST, `${prefix}.segs`);
    const state = make(16, U.STORAGE | U.COPY_DST, `${prefix}.state`);
    const core = make(coreBytes, U.STORAGE | U.COPY_DST, `${prefix}.core`);
    const global = make(16, U.UNIFORM | U.COPY_DST, `${prefix}.globals`);
    device.queue.writeBuffer(C, 0, input.C);
    device.queue.writeBuffer(AB, 0, input.AB);
    device.queue.writeBuffer(segs, 0, segment);
    device.queue.writeBuffer(state, 0, new Uint32Array(4));
    device.queue.writeBuffer(global, 0, globals);
    const bindGroups = candidates.map((candidate) => ({
      candidate,
      bind: device.createBindGroup({
        layout: candidate.pipeline.getBindGroupLayout(0),
        entries: [global, C, AB, segs, state, core].map((resource, binding) => ({ binding, resource: { buffer: resource } })),
      }),
    }));
    const warmup = device.createCommandEncoder({ label: "recur.tune.warmup" });
    for (const { candidate, bind } of bindGroups) encodePass(warmup, candidate, bind, linHeads, null, 0, 1);
    device.queue.submit([warmup.finish()]);
    await device.queue.onSubmittedWorkDone();

    querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `${prefix}.timestamps` });
    const resolved = make(queryBytes, U.QUERY_RESOLVE | U.COPY_SRC, `${prefix}.resolved`);
    read = make(readBytes, U.MAP_READ | U.COPY_DST, `${prefix}.read`);
    const encoder = device.createCommandEncoder({ label: `${prefix}.measure` });
    const byLanes = new Map(bindGroups.map(({ candidate, bind }) => [candidate.lanes, { candidate, bind }]));
    const pairs = Object.fromEntries(contenders.map((candidate) => [candidate.lanes, { baseline: [], candidate: [] }]));
    const meta = [];
    let queryIndex = 0;
    for (let sample = 0; sample < SAMPLES; sample++) for (let index = 0; index < contenders.length; index++) {
      const contender = contenders[index];
      const baselineFirst = (sample + index) % 2 === 0;
      const order = baselineFirst ? [4, contender.lanes] : [contender.lanes, 4];
      for (const lanes of order) {
        const entry = byLanes.get(lanes);
        encodePass(encoder, entry.candidate, entry.bind, linHeads, querySet, queryIndex, REPETITIONS);
        meta.push({ lanes: contender.lanes, sample, kind: lanes === 4 ? "baseline" : "candidate", queryIndex });
        queryIndex += 2;
      }
    }
    encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, read, 0, queryBytes);
    const started = now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await read.mapAsync(GPUMapMode.READ, 0, queryBytes);
    const stamps = new BigInt64Array(read.getMappedRange(0, queryBytes).slice(0));
    read.unmap();
    if (uncaptured.length) throw new Error(uncaptured.join("; "));
    for (const item of meta) pairs[item.lanes][item.kind].push(Number(stamps[item.queryIndex + 1] - stamps[item.queryIndex]) / 1e6 / REPETITIONS);
    if (queryIndex !== queryCount) throw new Error(`recurrence tuning wrote ${queryIndex} timestamps, expected ${queryCount}`);
    return { tokens, pairs, elapsedMs: now() - started };
  } finally {
    device.removeEventListener?.("uncapturederror", onUncaptured);
    querySet?.destroy();
    for (const item of owned) item.destroy();
  }
}

/** Benchmark up to two non-baseline lane candidates and choose a stable >=5% win at both points. */
export async function calibrateRecurrence(device, candidates, config) {
  const base = { revision: RECUR_TUNING_REVISION, method: "timestamp-query", tokens: [...TOKENS], samples: SAMPLES, repetitions: REPETITIONS, threshold: THRESHOLD };
  const list = Array.isArray(candidates) ? candidates : [];
  const byLane = new Map();
  for (const candidate of list) {
    if (!candidate || !SUPPORTED_LANES.includes(candidate.lanes) || !candidate.pipeline || byLane.has(candidate.lanes)) continue;
    byLane.set(candidate.lanes, { lanes: candidate.lanes, pipeline: candidate.pipeline });
  }
  const baseline = byLane.get(4);
  const fallbackPipeline = typeof baseline?.pipeline === "function" ? undefined : baseline?.pipeline;
  if (!device?.features?.has?.("timestamp-query")) return { ...fallback("no-timestamp-query", { ...base }), pipeline: fallbackPipeline };
  const values = configValues(config);
  if (!values) return { ...fallback("invalid-config", { ...base }), pipeline: fallbackPipeline };
  if (!baseline) return fallback("missing-4-lane-pipeline", { ...base, candidates: [...byLane.keys()] });
  const variants = [baseline, ...[...byLane.values()].filter((candidate) => candidate.lanes !== 4)].slice(0, MAX_VARIANTS);
  const contenders = variants.filter((candidate) => candidate.lanes !== 4);
  if (!contenders.length) return { choice: 4, zTiles: 2, pipeline: fallbackPipeline, diagnostics: { ...base, reason: "no-contender", candidates: [4] } };
  const started = now();
  let resolvedFallback = fallbackPipeline;
  try {
    const measured = await scoped(device, async () => {
      const resolved = await resolveCandidatePipelines(device, variants, (candidate) => {
        if (candidate.lanes === 4) resolvedFallback = candidate.pipeline;
      });
      const points = [];
      for (const tokens of TOKENS) points.push(await benchmark(device, resolved, values, tokens));
      return { points, resolved };
    });
    const selected = selectRecurrenceCandidate({ candidates: contenders, points: measured.points });
    const selectedPipeline = measured.resolved.find((candidate) => candidate.lanes === selected.choice)?.pipeline || resolvedFallback;
    return {
      choice: selected.choice,
      zTiles: selected.zTiles,
      pipeline: selectedPipeline,
      diagnostics: {
        ...base,
        reason: selected.winner ? "stable-win-both-points" : "no-stable-win-both-points",
        config: values,
        candidates: [4, ...contenders.map((candidate) => candidate.lanes)],
        elapsedMs: now() - started,
        variants: selected.candidates,
        benchmarkElapsedMs: measured.points.reduce((sum, point) => sum + point.elapsedMs, 0),
      },
    };
  } catch (error) {
    return { ...fallback("benchmark-failed", { ...base, elapsedMs: now() - started, error: String(error?.message || error) }), pipeline: resolvedFallback };
  }
}
