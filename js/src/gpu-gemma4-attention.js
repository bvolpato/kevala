// Load-time exactness and timing gate for Gemma's fixed-32 subgroup attention.
// Selection belongs to this loaded device and model configuration only.

import { bufferUsage, inErrorScopes, pipeline, scratchBuffer } from "./gpu.js";
import { mapBeforeDeadline } from "./gpu-matvec.js";
import { selectMatmulKernel } from "./gpu-tuning.js";

export const GEMMA4_ATTENTION_TUNING_REVISION = 1;

const SAMPLES = 3;
const REPETITIONS = 2;
const DEADLINE_MS = 750;
const MAX_SCRATCH_BYTES = 32 * 1024 * 1024;
const PREFIX_TOKENS = 89;
const BRANCH_A_TOKENS = 42;
const BRANCH_B_TOKENS = 55;
const SUFFIX_TOKENS = BRANCH_A_TOKENS + BRANCH_B_TOKENS;
const PREFIX_LENGTH = PREFIX_TOKENS;
const TOTAL_KV_TOKENS = PREFIX_TOKENS + SUFFIX_TOKENS;
const SENTINEL_BITS = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7fc0dead, 0x3eaaaaab, 0x00800000, 0xff800000];
const OUTPUT_POISON = 0x7fc0beef;
const now = () => (typeof performance === "undefined" ? Date.now() : performance.now());

function finish(diagnostics, candidatePipeline = null, layout = null, selection = new Set()) {
  diagnostics.elapsedMs = now() - diagnostics.startedAt;
  delete diagnostics.startedAt;
  return { pipeline: candidatePipeline, layout, selection, diagnostics };
}

function errorDetails(error) {
  return {
    message: String(error?.message || error || "unknown WebGPU error"),
    ...(typeof error?.code === "string" ? { code: error.code } : {}),
  };
}

function requestedHeadDims(cfg) {
  const values = Array.isArray(cfg?.headDims)
    ? cfg.headDims
    : [cfg?.localHeadDim, cfg?.globalHeadDim];
  const dims = [...new Set(values.map(Number))].sort((a, b) => a - b);
  return dims;
}

function signedValues(length, seed, scale) {
  const values = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < length; index++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    const unit = ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
    values[index] = Math.fround((unit * 2 - 1) * scale) || (unit < 0.5 ? -0.0009765625 : 0.0009765625);
  }
  return values;
}

function makeData(headDim, heads, kvHeads) {
  const prefixQueryWidth = heads * headDim;
  const kvWidth = kvHeads * headDim;
  const suffixQueryWidth = SUFFIX_TOKENS * prefixQueryWidth;
  return {
    prefixQ: signedValues(PREFIX_TOKENS * prefixQueryWidth, 0x47415454 ^ headDim, 0.035),
    suffixQ: signedValues(suffixQueryWidth, 0x53554658 ^ headDim, 0.035),
    keys: signedValues(TOTAL_KV_TOKENS * kvWidth, 0x4b455953 ^ headDim, 0.045),
    values: signedValues(TOTAL_KV_TOKENS * kvWidth, 0x56414c53 ^ headDim, 0.055),
  };
}

function positionsFor(kind) {
  if (kind === "prefix") {
    const positions = new Uint32Array(2 * PREFIX_TOKENS);
    for (let token = 0; token < PREFIX_TOKENS; token++) positions[token] = token;
    return { tokens: PREFIX_TOKENS, rows: PREFIX_TOKENS, total: PREFIX_TOKENS, branchPrefix: 0, batch: 0, positions };
  }
  if (kind === "suffix") {
    const positions = new Uint32Array(2 * SUFFIX_TOKENS);
    for (let token = 0; token < BRANCH_A_TOKENS; token++) positions[token] = PREFIX_LENGTH + token;
    for (let token = 0; token < BRANCH_B_TOKENS; token++) {
      const row = BRANCH_A_TOKENS + token;
      positions[row] = PREFIX_LENGTH + token;
      positions[SUFFIX_TOKENS + row] = BRANCH_A_TOKENS;
    }
    return { tokens: SUFFIX_TOKENS, rows: 2, total: TOTAL_KV_TOKENS, branchPrefix: PREFIX_LENGTH, batch: 1, positions };
  }
  const positions = new Uint32Array([PREFIX_LENGTH + BRANCH_A_TOKENS - 1, PREFIX_LENGTH + BRANCH_B_TOKENS - 1, 0, BRANCH_A_TOKENS]);
  return { tokens: 2, rows: 2, total: TOTAL_KV_TOKENS, branchPrefix: PREFIX_LENGTH, batch: 1, positions };
}

function queryFor(kind, data, heads, headDim) {
  if (kind === "prefix") return data.prefixQ;
  if (kind === "suffix") return data.suffixQ;
  const width = heads * headDim;
  const output = new Float32Array(2 * width);
  output.set(data.suffixQ.subarray((BRANCH_A_TOKENS - 1) * width, BRANCH_A_TOKENS * width), 0);
  output.set(data.suffixQ.subarray((SUFFIX_TOKENS - 1) * width, SUFFIX_TOKENS * width), width);
  return output;
}

function exactOutputCheck(baselineBuffer, candidateBuffer, outputBytes, activeFloats) {
  const baselineBits = new Uint32Array(baselineBuffer, 0, outputBytes / 4);
  const candidateBits = new Uint32Array(candidateBuffer, 0, outputBytes / 4);
  const baseline = new Float32Array(baselineBuffer, 0, activeFloats);
  const candidate = new Float32Array(candidateBuffer, 0, activeFloats);
  for (let index = 0; index < activeFloats; index++) {
    if (!Number.isFinite(baseline[index]) || !Number.isFinite(candidate[index])) {
      return {
        ok: false,
        reason: "non-finite-output",
        index,
        baseline: baseline[index],
        candidate: candidate[index],
      };
    }
    if (baselineBits[index] !== candidateBits[index]) {
      return {
        ok: false,
        reason: "output-bits-mismatch",
        index,
        baseline: baseline[index],
        candidate: candidate[index],
        baselineBits: `0x${baselineBits[index].toString(16).padStart(8, "0")}`,
        candidateBits: `0x${candidateBits[index].toString(16).padStart(8, "0")}`,
      };
    }
  }
  const sentinelStart = activeFloats;
  for (let index = 0; index < SENTINEL_BITS.length; index++) {
    const at = sentinelStart + index;
    if (baselineBits[at] !== SENTINEL_BITS[index] || candidateBits[at] !== SENTINEL_BITS[index]) {
      return {
        ok: false,
        reason: "output-sentinel-overwritten",
        index,
        expectedBits: `0x${SENTINEL_BITS[index].toString(16).padStart(8, "0")}`,
        baselineBits: `0x${baselineBits[at].toString(16).padStart(8, "0")}`,
        candidateBits: `0x${candidateBits[at].toString(16).padStart(8, "0")}`,
      };
    }
  }
  return { ok: true, reason: "exact-output-bits", activeFloats, sentinels: SENTINEL_BITS.length };
}

function dispatch(encoder, computePipeline, bindGroup, tokens, heads, querySet = null, queryIndex = 0, repetitions = 1) {
  const pass = encoder.beginComputePass(querySet
    ? { timestampWrites: { querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } }
    : {});
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, bindGroup);
  for (let repetition = 0; repetition < repetitions; repetition++) pass.dispatchWorkgroups(tokens, heads);
  pass.end();
}

function deadlineError() {
  return Object.assign(new Error("Gemma 4 subgroup attention calibration deadline exceeded"), { code: "GEMMA4_ATTN_DEADLINE" });
}

async function measureCase(runtime, baselinePipeline, candidatePipeline, cfg, headDim, kind, data, deadline) {
  const device = runtime.device;
  const { tokens, rows, total, branchPrefix, batch, positions } = positionsFor(kind);
  const heads = Number(cfg.heads);
  const kvHeads = Number(cfg.kvHeads);
  const query = queryFor(kind, data, heads, headDim);
  const queryBytes = query.byteLength;
  const kvBytes = data.keys.byteLength;
  const positionBytes = positions.byteLength;
  const activeFloats = tokens * heads * headDim;
  const outputBytes = (activeFloats + SENTINEL_BITS.length) * Float32Array.BYTES_PER_ELEMENT;
  const queryCount = SAMPLES * 2 * 2;
  const resolvedBytes = queryCount * 8;
  // Reuse one MAP_READ buffer for the two exactness snapshots. Keeping both snapshots in a
  // single mapped buffer would push the 512-wide, two-branch prefill fixture over 32 MiB.
  const readBytes = Math.max(256, outputBytes, resolvedBytes);
  const storageSizes = [queryBytes, kvBytes, kvBytes, positionBytes, outputBytes, outputBytes];
  const scratchBytes = 16 + 32 + queryBytes + kvBytes * 2 + positionBytes + outputBytes * 2 + readBytes + resolvedBytes;
  const limits = device.limits || {};
  const maxBufferSize = limits.maxBufferSize || Infinity;
  const maxStorage = Math.min(maxBufferSize, limits.maxStorageBufferBindingSize || Infinity);
  const base = { key: `${headDim}:${kind}`, headDim, kind, tokens, rows, total, activeFloats, selected: false, scratchBytes };
  if (!Number.isSafeInteger(scratchBytes) || scratchBytes > MAX_SCRATCH_BYTES || storageSizes.some((size) => size > maxStorage) || readBytes > maxBufferSize || resolvedBytes > maxBufferSize) {
    return { ...base, reason: "scratch-limit", maxScratchBytes: MAX_SCRATCH_BYTES };
  }
  if (now() >= deadline) throw deadlineError();

  const owned = [];
  let querySet = null;
  let read = null;
  let mapped = false;
  try {
    const usage = bufferUsage();
    const make = (size, flags, label) => {
      const buffer = scratchBuffer(device, size, flags, `gemma4-attention.${headDim}.${kind}.${label}`);
      owned.push(buffer);
      return buffer;
    };
    const globals = make(16, usage.UNIFORM | usage.COPY_DST, "globals");
    const params = make(32, usage.UNIFORM | usage.COPY_DST, "params");
    const q = make(queryBytes, usage.STORAGE | usage.COPY_DST, "q");
    const k = make(kvBytes, usage.STORAGE | usage.COPY_DST, "k");
    const v = make(kvBytes, usage.STORAGE | usage.COPY_DST, "v");
    const pos = make(positionBytes, usage.STORAGE | usage.COPY_DST, "pos");
    const baselineOutput = make(outputBytes, usage.STORAGE | usage.COPY_DST | usage.COPY_SRC, "baseline-output");
    const candidateOutput = make(outputBytes, usage.STORAGE | usage.COPY_DST | usage.COPY_SRC, "candidate-output");
    read = make(readBytes, usage.MAP_READ | usage.COPY_DST, "readback");
    const resolved = make(resolvedBytes, usage.QUERY_RESOLVE | usage.COPY_SRC, "timestamps");

    device.queue.writeBuffer(globals, 0, new Uint32Array([tokens, rows, total, branchPrefix]));
    device.queue.writeBuffer(params, 0, new Uint32Array([heads, kvHeads, headDim, headDim === Number(cfg.localHeadDim) ? Number(cfg.window) : 0, 1, batch, 0, 0]));
    device.queue.writeBuffer(q, 0, query);
    device.queue.writeBuffer(k, 0, data.keys);
    device.queue.writeBuffer(v, 0, data.values);
    device.queue.writeBuffer(pos, 0, positions);
    const poisoned = new Uint32Array(activeFloats + SENTINEL_BITS.length).fill(OUTPUT_POISON);
    poisoned.set(SENTINEL_BITS, activeFloats);
    device.queue.writeBuffer(baselineOutput, 0, poisoned);
    device.queue.writeBuffer(candidateOutput, 0, poisoned);

    const bindGroup = (computePipeline, output) => device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [globals, params, q, k, v, output, pos].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const baselineGroup = bindGroup(baselinePipeline, baselineOutput);
    const candidateGroup = bindGroup(candidatePipeline, candidateOutput);

    const baselineParityEncoder = device.createCommandEncoder({ label: `gemma4-attention.${headDim}.${kind}.baseline-parity` });
    dispatch(baselineParityEncoder, baselinePipeline, baselineGroup, tokens, heads);
    baselineParityEncoder.copyBufferToBuffer(baselineOutput, 0, read, 0, outputBytes);
    device.queue.submit([baselineParityEncoder.finish()]);
    if (!await mapBeforeDeadline(read, outputBytes, deadline)) throw deadlineError();
    mapped = true;
    const baselineSnapshot = read.getMappedRange(0, outputBytes).slice(0);
    read.unmap();
    mapped = false;

    const candidateParityEncoder = device.createCommandEncoder({ label: `gemma4-attention.${headDim}.${kind}.candidate-parity` });
    dispatch(candidateParityEncoder, candidatePipeline, candidateGroup, tokens, heads);
    candidateParityEncoder.copyBufferToBuffer(candidateOutput, 0, read, 0, outputBytes);
    device.queue.submit([candidateParityEncoder.finish()]);
    if (!await mapBeforeDeadline(read, outputBytes, deadline)) throw deadlineError();
    mapped = true;
    const candidateSnapshot = read.getMappedRange(0, outputBytes).slice(0);
    const parity = exactOutputCheck(baselineSnapshot, candidateSnapshot, outputBytes, activeFloats);
    read.unmap();
    mapped = false;
    if (!parity.ok) return { ...base, numericalCheck: parity, reason: "numerical-guard-failed" };

    if (now() >= deadline) throw deadlineError();
    const warmup = device.createCommandEncoder({ label: `gemma4-attention.${headDim}.${kind}.warmup` });
    dispatch(warmup, baselinePipeline, baselineGroup, tokens, heads);
    dispatch(warmup, candidatePipeline, candidateGroup, tokens, heads);
    device.queue.submit([warmup.finish()]);

    querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `gemma4-attention.${headDim}.${kind}.queries` });
    const timing = device.createCommandEncoder({ label: `gemma4-attention.${headDim}.${kind}.timing` });
    const sampleIndices = { baseline: [], candidate: [] };
    let queryIndex = 0;
    for (let sample = 0; sample < SAMPLES; sample++) {
      if (now() >= deadline) throw deadlineError();
      const order = sample % 2 === 0 ? ["baseline", "candidate"] : ["candidate", "baseline"];
      for (const which of order) {
        const isBaseline = which === "baseline";
        sampleIndices[which].push(queryIndex);
        dispatch(
          timing,
          isBaseline ? baselinePipeline : candidatePipeline,
          isBaseline ? baselineGroup : candidateGroup,
          tokens,
          heads,
          querySet,
          queryIndex,
          REPETITIONS,
        );
        queryIndex += 2;
      }
    }
    timing.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
    timing.copyBufferToBuffer(resolved, 0, read, 0, resolvedBytes);
    device.queue.submit([timing.finish()]);
    if (!await mapBeforeDeadline(read, resolvedBytes, deadline)) throw deadlineError();
    mapped = true;
    const stamps = new BigUint64Array(read.getMappedRange(0, resolvedBytes).slice(0));
    read.unmap();
    mapped = false;
    const timings = (indices) => indices.map((index) => Number(stamps[index + 1] - stamps[index]) / 1e6 / REPETITIONS);
    const baseline = timings(sampleIndices.baseline);
    const candidate = timings(sampleIndices.candidate);
    const picked = selectMatmulKernel({ generic: baseline, wide: candidate });
    return {
      ...base,
      baselineMs: baseline,
      candidateMs: candidate,
      numericalCheck: parity,
      selected: picked.kernel === "wide",
      stable: picked.stable,
      benefit: picked.benefit,
      reason: picked.reason === "stable-wide-win" ? "stable-subgroup-win" : picked.reason,
    };
  } finally {
    if (mapped) read?.unmap();
    for (const buffer of owned) buffer.destroy();
    querySet?.destroy();
  }
}

/**
 * Exact and paired-timed gate for the fixed-32 subgroup attention kernel. The returned selection
 * is conservative: prefix use requires exact wins for both shared-prefix and branched-suffix
 * cases; tail use is calibrated independently.
 */
export async function calibrateGemma4Attention(runtime, source, cfg, baselinePipeline, { enabled = true } = {}) {
  const startedAt = now();
  const selection = new Set();
  const diagnostics = {
    revision: GEMMA4_ATTENTION_TUNING_REVISION,
    method: "baseline",
    samples: SAMPLES,
    repetitions: REPETITIONS,
    threshold: 0.05,
    deadlineMs: DEADLINE_MS,
    maxScratchBytes: MAX_SCRATCH_BYTES,
    points: [],
    truncated: false,
    startedAt,
  };
  const result = (reason, candidatePipeline = null, layout = null) => {
    diagnostics.reason = reason;
    return finish(diagnostics, candidatePipeline, layout, selection);
  };
  if (!enabled) return result("disabled");
  const device = runtime?.device;
  if (!device || !baselinePipeline?.getBindGroupLayout || typeof source !== "function") return result("missing-runtime-or-baseline");
  const info = runtime.adapter?.info || {};
  diagnostics.subgroupSize = { min: info.subgroupMinSize ?? null, max: info.subgroupMaxSize ?? null };
  if (runtime.subgroup4 !== true || info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32) return result("requires-fixed-32-subgroups");
  if (!device.features?.has?.("timestamp-query")) return result("no-timestamp-query");
  const heads = Number(cfg?.heads);
  const kvHeads = Number(cfg?.kvHeads);
  const window = Number(cfg?.window);
  const dims = requestedHeadDims(cfg);
  diagnostics.headDims = dims;
  if (!Number.isSafeInteger(heads) || heads < 1 || !Number.isSafeInteger(kvHeads) || kvHeads < 1 || heads % kvHeads !== 0 || !Number.isSafeInteger(window) || window < 1 || !dims.length) return result("invalid-attention-config");
  const unsupported = dims.filter((dim) => dim !== 256 && dim !== 512);
  diagnostics.unsupportedHeadDims = unsupported;
  const supported = dims.filter((dim) => dim === 256 || dim === 512);
  if (!supported.length) return result("unsupported-head-dim");

  let candidatePipeline = null;
  let candidateLayout = null;
  try {
    if (now() >= startedAt + DEADLINE_MS) throw deadlineError();
    candidatePipeline = await inErrorScopes(device, async () => {
      candidateLayout = device.createBindGroupLayout({
        label: "Gemma 4 subgroup attention calibration",
        entries: ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]
          .map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [candidateLayout] });
      return pipeline(device, source("gemma4_attention_subgroup", {}), "gemma4_attention_subgroup.calibration", layout);
    });
  } catch (error) {
    diagnostics.error = errorDetails(error);
    if (error?.code === "GEMMA4_ATTN_DEADLINE") {
      diagnostics.truncated = true;
      return result("deadline");
    }
    return result("pipeline-failed");
  }

  diagnostics.method = "paired-timestamp-query";
  const deadline = startedAt + DEADLINE_MS;
  let fatalFailure = false;
  for (const headDim of supported) {
    let data;
    try {
      data = makeData(headDim, heads, kvHeads);
    } catch (error) {
      diagnostics.error = errorDetails(error);
      diagnostics.reason = "calibration-failed";
      fatalFailure = true;
      break;
    }
    const points = new Map();
    for (const kind of ["prefix", "suffix", "tail"]) {
      if (now() >= deadline) {
        diagnostics.truncated = true;
        diagnostics.reason = "deadline";
        break;
      }
      let point;
      try {
        point = await inErrorScopes(device, () => measureCase(runtime, baselinePipeline, candidatePipeline, cfg, headDim, kind, data, deadline));
      } catch (error) {
        diagnostics.error = errorDetails(error);
        if (error?.code === "MATVEC_DEADLINE" || error?.code === "GEMMA4_ATTN_DEADLINE") {
          diagnostics.truncated = true;
          diagnostics.reason = "deadline";
          break;
        }
        fatalFailure = true;
        diagnostics.reason = "calibration-failed";
        break;
      }
      points.set(kind, point);
      diagnostics.points.push(point);
      if (point.reason === "numerical-guard-failed") {
        fatalFailure = true;
        diagnostics.reason = "numerical-guard-failed";
        break;
      }
    }
    if (fatalFailure) break;
    const prefix = points.get("prefix");
    const suffix = points.get("suffix");
    if (prefix?.selected && suffix?.selected) selection.add(`${headDim}:prefill`);
    const tail = points.get("tail");
    if (tail?.selected) selection.add(`${headDim}:tail`);
    if (diagnostics.truncated) break;
  }
  if (fatalFailure) selection.clear();
  if (!diagnostics.reason) diagnostics.reason = selection.size ? "calibrated" : unsupported.length ? "unsupported-head-dim-or-no-stable-win" : "no-stable-win";
  return finish(diagnostics, candidatePipeline, candidateLayout, selection);
}
