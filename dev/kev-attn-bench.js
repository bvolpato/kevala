// Focused numerical and timing guard for the FP32 query-tiled Kev attention experiment. Each case
// runs the existing subgroup fallback and the new block-table tile kernel on identical inputs,
// checks both against an independent CPU oracle, then compares every output element.

import { kernelSource } from "./wgsl.js";
import { pipeline, requestDevice } from "../js/src/gpu.js";

const query = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => (logNode.textContent += `${line}\n`);
const state = { status: "running", backend: "webgpu", kernel: "kev_attention_tile", subgroups: true };
window.gpuBench = state;

const DIM = 256;
const KEV_SPEC = { hidden: 1024, heads: 8, kv_heads: 2, lin_key_heads: 16, lin_heads: 16, rotary: 64 };
const VALUE_SCALE = 0.35;
const GATE_SCALE = 8;
const TOLERANCE = { absolute: 2e-4, relative: 2e-4 };
let U;
let activeDevice = null;

const PLANS = [
  {
    label: "empty-dispatch",
    heads: 8,
    kvHeads: 2,
    segments: [],
  },
  {
    label: "fresh-boundaries",
    heads: 8,
    kvHeads: 2,
    segments: [127, 128, 129, 255, 256, 257].map((len) => ({ label: `fresh-${len}`, len })),
  },
  {
    label: "cached-extensions",
    heads: 8,
    kvHeads: 2,
    prefixes: [127, 128, 255, 256].map((len) => ({ id: `prefix-${len}`, len })),
    // Reusing a prefix in several independent segments models repeated cache extensions.
    segments: [
      { label: "repeat-127-a", len: 1, parent: "prefix-127" },
      { label: "repeat-127-b", len: 2, parent: "prefix-127" },
      { label: "extend-128", len: 1, parent: "prefix-128" },
      { label: "repeat-255-a", len: 1, parent: "prefix-255" },
      { label: "repeat-255-b", len: 2, parent: "prefix-255" },
      { label: "extend-256", len: 1, parent: "prefix-256" },
    ],
  },
  {
    label: "cached-tile-boundaries",
    heads: 8,
    kvHeads: 2,
    prefixes: [7, 8, 9, 15, 16, 17].map((len) => ({ id: `prefix-${len}`, len })),
    segments: [7, 8, 9, 15, 16, 17].map((len) => ({ label: `extend-${len}`, len: 1, parent: `prefix-${len}` })),
  },
  {
    label: "fresh-tile-boundaries",
    heads: 8,
    kvHeads: 2,
    segments: [1, 7, 8, 9, 15, 16, 17].map((len) => ({ label: `fresh-${len}`, len })),
  },
  ...[31, 124, 532, 105, 198, 606].map((len) => ({
    label: `actual-${len}`,
    heads: 8,
    kvHeads: 2,
    segments: [{ label: `actual-${len}`, len }],
  })),
  {
    label: "mixed-gqa-cache",
    heads: 16,
    kvHeads: 4,
    prefixes: [7, 16, 31].map((len) => ({ id: `prefix-${len}`, len })),
    segments: [7, 16, 31].map((len) => ({ label: `extend-${len}`, len: 1, parent: `prefix-${len}` })),
  },
];

function intParam(name, fallback, min, max) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  return Number.isInteger(value) && value >= min && value <= max ? value : fallback;
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b);
    s ^= s >>> 16;
    return (s >>> 0) / 0x100000000;
  };
}

function selectedPlans() {
  const only = query.get("only");
  if (!only || only === "all") return PLANS;
  const selected = PLANS.filter((plan) => plan.label === only);
  if (!selected.length) throw new Error(`unknown plan ${only}; expected ${PLANS.map((plan) => plan.label).join(", ")}`);
  return selected;
}

function makeArray(random, length, scale) {
  const values = new Float32Array(length);
  for (let i = 0; i < values.length; i++) values[i] = Math.fround((random() * 2 - 1) * scale);
  return values;
}

function makeCase(plan, seed) {
  const random = rng(seed);
  const heads = plan.heads;
  const kvHeads = plan.kvHeads;
  const attnQ = heads * DIM;
  const attnK = kvHeads * DIM;
  const attnKV = 2 * attnK;
  const attnWidth = 2 * attnQ + attnKV;
  const prefixes = new Map();
  let kvRows = 3;
  for (const [index, prefix] of (plan.prefixes || []).entries()) {
    const pstart = kvRows;
    kvRows += prefix.len + 3;
    prefixes.set(prefix.id, {
      id: prefix.id,
      index,
      len: prefix.len,
      pstart,
      k: makeArray(random, prefix.len * attnK, VALUE_SCALE),
      v: makeArray(random, prefix.len * attnK, VALUE_SCALE),
    });
  }

  const segments = [];
  let T = 0;
  for (const [index, segment] of plan.segments.entries()) {
    const prefix = segment.parent ? prefixes.get(segment.parent) : null;
    if (segment.parent && !prefix) throw new Error(`missing prefix ${segment.parent}`);
    segments.push({
      index,
      label: segment.label,
      start: T,
      len: segment.len,
      parent: prefix ? prefix.index + 1 : 0xffffffff,
      parentId: prefix?.id || null,
      pstart: prefix?.pstart || 0,
      plen: prefix?.len || 0,
    });
    T += segment.len;
  }

  const q = makeArray(random, T * attnQ, VALUE_SCALE);
  const gate = makeArray(random, T * attnQ, GATE_SCALE);
  if (query.get("gateTails") === "1") {
    const tails = [-3.4028234663852886e38, -1000, -100, -20, -10, -1, 0, 1, 10, 20, 100, 1000, 3.4028234663852886e38];
    for (let index = 0; index < gate.length; index++) gate[index] = tails[index % tails.length];
  }
  const currentK = makeArray(random, T * attnK, VALUE_SCALE);
  const currentV = makeArray(random, T * attnK, VALUE_SCALE);
  const proj = new Float32Array(T * attnWidth);
  const keys = new Float32Array(T * attnK);
  const tok = new Uint32Array(T * 4);
  const segs = new Uint32Array(segments.length * 8);
  const blockCount = segments.reduce((sum, segment) => sum + Math.ceil(segment.len / 8), 0);
  const blocks = new Uint32Array((1 + blockCount) * 4);
  blocks[0] = blockCount;
  const kv = new Float32Array(kvRows * attnKV);

  for (let t = 0; t < T; t++) {
    const row = t * attnWidth;
    for (let h = 0; h < heads; h++) {
      const qBase = t * attnQ + h * DIM;
      const qOut = row + h * 2 * DIM;
      const gateOut = qOut + DIM;
      for (let d = 0; d < DIM; d++) {
        proj[qOut + d] = q[qBase + d];
        proj[gateOut + d] = gate[qBase + d];
      }
    }
    for (let h = 0; h < kvHeads; h++) {
      const source = t * attnK + h * DIM;
      const keyOut = row + 2 * attnQ + h * DIM;
      const valueOut = keyOut + attnK;
      for (let d = 0; d < DIM; d++) {
        proj[keyOut + d] = currentK[source + d];
        proj[valueOut + d] = currentV[source + d];
        keys[(h * DIM + d) * T + t] = currentK[source + d];
      }
    }
  }

  let block = 1;
  for (const segment of segments) {
    for (let r = 0; r < segment.len; r++) {
      const t = segment.start + r;
      tok.set([segment.index, r, 0, 0], t * 4);
    }
    segs.set([segment.start, segment.len, segment.parent, segment.pstart, segment.plen, 0, 0, 0], segment.index * 8);
    for (let r0 = 0; r0 < segment.len; r0 += 8) {
      blocks.set([segment.index, r0, Math.min(8, segment.len - r0), 0], block * 4);
      block++;
    }
  }
  for (const prefix of prefixes.values()) {
    for (let j = 0; j < prefix.len; j++) {
      const cacheRow = prefix.pstart + j;
      for (let h = 0; h < kvHeads; h++) {
        const source = j * attnK + h * DIM;
        const keyOut = cacheRow * attnKV + h * DIM;
        const valueOut = keyOut + attnK;
        for (let d = 0; d < DIM; d++) {
          kv[keyOut + d] = prefix.k[source + d];
          kv[valueOut + d] = prefix.v[source + d];
        }
      }
    }
  }

  return {
    label: plan.label,
    T,
    heads,
    kvHeads,
    dim: DIM,
    attnQ,
    attnK,
    attnKV,
    attnWidth,
    segments,
    prefixes,
    q,
    gate,
    currentK,
    currentV,
    proj,
    keys,
    tok,
    blocks,
    segs,
    kv,
  };
}

function sampleRows(c, segment) {
  const positions = c.T <= 300
    ? Array.from({ length: segment.len }, (_, i) => i)
    : [0, 1, Math.floor(segment.len / 2), Math.max(0, segment.len - 2), segment.len - 1];
  return [...new Set(positions)].filter((position) => position >= 0 && position < segment.len).map((position) => segment.start + position);
}

// The oracle reads the generated logical tensors directly. It does not reuse the shader's
// transposed indexing, workgroup reduction, or online-softmax tile structure.
function cpuReference(c) {
  const checks = [];
  const group = c.heads / c.kvHeads;
  let checkedRows = 0;
  for (const segment of c.segments) {
    const rows = sampleRows(c, segment);
    checkedRows += rows.length;
    for (const t of rows) {
      const relative = t - segment.start;
      const n = segment.plen + relative + 1;
      for (let h = 0; h < c.heads; h++) {
        const kvHead = Math.floor(h / group);
        const qBase = t * c.attnQ + h * DIM;
        const scores = new Float64Array(n);
        let maxScore = -Infinity;
        for (let j = 0; j < n; j++) {
          const keyBase = j < segment.plen
            ? j * c.attnK + kvHead * DIM
            : (segment.start + j - segment.plen) * c.attnK + kvHead * DIM;
          const key = j < segment.plen ? c.prefixes.get(segment.parentId).k : c.currentK;
          let score = 0;
          for (let d = 0; d < DIM; d++) score += c.q[qBase + d] * key[keyBase + d];
          score /= 16;
          scores[j] = score;
          maxScore = Math.max(maxScore, score);
        }
        const weights = new Float64Array(n);
        let denominator = 0;
        for (let j = 0; j < n; j++) {
          weights[j] = Math.exp(scores[j] - maxScore);
          denominator += weights[j];
        }
        for (let d = 0; d < DIM; d++) {
          let value = 0;
          for (let j = 0; j < n; j++) {
            const valueBase = j < segment.plen
              ? j * c.attnK + kvHead * DIM
              : (segment.start + j - segment.plen) * c.attnK + kvHead * DIM;
            const values = j < segment.plen ? c.prefixes.get(segment.parentId).v : c.currentV;
            value += (weights[j] / denominator) * values[valueBase + d];
          }
          const gate = c.gate[qBase + d];
          checks.push({ index: t * c.attnQ + h * DIM + d, expected: value / (1 + Math.exp(-gate)) });
        }
      }
    }
  }
  return { checks, checkedRows, checkedQueryHeads: checkedRows * c.heads };
}

function compareCpu(values, reference) {
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) return { ok: false, samples: reference.checks.length, message: `non-finite output at index ${index}` };
  }
  for (const { index, expected } of reference.checks) {
    const actual = values[index];
    const abs = Math.abs(actual - expected);
    const relative = abs / Math.max(1, Math.abs(expected));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index, expected, actual };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > TOLERANCE.absolute + TOLERANCE.relative * Math.max(1, Math.abs(expected))) {
      return { ok: false, samples: reference.checks.length, maxAbs, maxRelative, worst, message: `CPU mismatch at output index ${index}` };
    }
  }
  return { ok: true, samples: reference.checks.length, maxAbs, maxRelative, worst, tolerance: TOLERANCE };
}

function compareArrays(actual, expected) {
  if (actual.length !== expected.length) return { ok: false, message: `output lengths differ: ${actual.length} vs ${expected.length}` };
  let maxAbs = 0;
  let maxRelative = 0;
  let worst = null;
  for (let index = 0; index < actual.length; index++) {
    if (!Number.isFinite(actual[index])) return { ok: false, message: `non-finite candidate output at index ${index}` };
    if (!Number.isFinite(expected[index])) return { ok: false, message: `non-finite baseline output at index ${index}` };
    const abs = Math.abs(actual[index] - expected[index]);
    const relative = abs / Math.max(1, Math.abs(expected[index]));
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = { index, expected: expected[index], actual: actual[index] };
    }
    maxRelative = Math.max(maxRelative, relative);
    if (abs > TOLERANCE.absolute + TOLERANCE.relative * Math.max(1, Math.abs(expected[index]))) {
      return { ok: false, maxAbs, maxRelative, worst, message: `paired output mismatch at index ${index}` };
    }
  }
  return { ok: true, maxAbs, maxRelative, worst, tolerance: TOLERANCE };
}

function makeBuffer(device, bytes, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, bytes), usage });
}

function adapterInfo(gpu) {
  const info = gpu.adapter.info || {};
  return {
    name: gpu.name,
    vendor: info.vendor || null,
    architecture: info.architecture || null,
    device: info.device || null,
    description: info.description || null,
    isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter),
    features: [...gpu.device.features].sort(),
    limits: {
      maxBufferSize: gpu.device.limits.maxBufferSize,
      maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: gpu.device.limits.maxComputeWorkgroupStorageSize,
    },
  };
}

function makeTimer(device) {
  if (!device.features.has("timestamp-query")) return { timestamp: false, method: "performance.now() wall clock" };
  return {
    timestamp: true,
    method: "WebGPU timestamp-query",
    querySet: device.createQuerySet({ type: "timestamp", count: 2, label: "kev-attn.timestamps" }),
    resolved: makeBuffer(device, 16, U.QUERY_RESOLVE | U.COPY_SRC, "kev-attn.timestamp-resolved"),
    read: makeBuffer(device, 16, U.MAP_READ | U.COPY_DST, "kev-attn.timestamp-read"),
  };
}

async function timedDispatch(device, timer, run, reps) {
  const encoder = device.createCommandEncoder({ label: `${run.label}.dispatch` });
  const pass = timer.timestamp
    ? encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })
    : encoder.beginComputePass();
  pass.setPipeline(run.pipeline);
  pass.setBindGroup(0, run.group);
  for (let i = 0; i < reps; i++) {
    if (run.dispatchX && run.dispatchY) pass.dispatchWorkgroups(run.dispatchX, run.dispatchY, 1);
  }
  pass.end();
  if (timer.timestamp) {
    encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolved, 0);
    encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16);
  }
  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  if (!timer.timestamp) return (performance.now() - started) / reps;
  await timer.read.mapAsync(GPUMapMode.READ, 0, 16);
  const timestamps = new BigInt64Array(timer.read.getMappedRange(0, 16).slice(0));
  timer.read.unmap();
  return Number(timestamps[1] - timestamps[0]) / 1e6 / reps;
}

async function readOutput(device, output, bytes) {
  if (bytes === 0) return new Float32Array();
  const readback = makeBuffer(device, bytes, U.MAP_READ | U.COPY_DST, "kev-attn.readback");
  try {
    const encoder = device.createCommandEncoder({ label: "kev-attn.readback" });
    encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const copy = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
    readback.unmap();
    return copy;
  } finally {
    readback.destroy();
  }
}

function estimateWork(c) {
  let work = 0;
  for (const segment of c.segments) {
    for (let r = 0; r < segment.len; r++) work += (segment.plen + r + 1) * c.heads * DIM;
  }
  return work;
}

function bindLayout(device, kind) {
  const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  const entries = [
    entry(0, "uniform"),
    entry(1, "read-only-storage"),
    entry(2, "read-only-storage"),
    entry(3, "read-only-storage"),
    entry(4, "read-only-storage"),
    entry(5, "storage"),
  ];
  if (kind === "fallback") entries.push(entry(6, "read-only-storage"));
  return device.createBindGroupLayout({
    label: `kev-attn-${kind}`,
    entries,
  });
}

function makeRun(device, layout, pipelineForCase, c, kind) {
  const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${c.label}.globals`);
  const proj = makeBuffer(device, c.proj.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.proj`);
  const kv = makeBuffer(device, c.kv.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.kv`);
  const segs = makeBuffer(device, c.segs.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.segs`);
  const output = makeBuffer(device, c.T * c.attnQ * 4, U.STORAGE | U.COPY_SRC, `${c.label}.out`);
  const blocks = kind === "tile" ? makeBuffer(device, c.blocks.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.blocks`) : null;
  const tok = kind === "fallback" ? makeBuffer(device, c.tok.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.tok`) : null;
  const keys = kind === "fallback" ? makeBuffer(device, c.keys.byteLength, U.STORAGE | U.COPY_DST, `${c.label}.keys`) : null;
  const writes = [[globals, new Uint32Array([c.T, 0, 0, 0])], [proj, c.proj], [kv, c.kv], [segs, c.segs]];
  if (blocks) writes.push([blocks, c.blocks]);
  if (tok) writes.push([tok, c.tok]);
  if (keys) writes.push([keys, c.keys]);
  for (const [buffer, data] of writes) {
    if (!data.byteLength) continue;
    device.queue.writeBuffer(buffer, 0, data);
  }
  const buffers = [globals, proj, kv, kind === "tile" ? blocks : tok, segs, output, ...(kind === "fallback" ? [keys] : [])];
  const entries = buffers.map((buffer, binding) => ({ binding, resource: { buffer } }));
  const group = device.createBindGroup({
    layout,
    entries,
  });
  return {
    label: `${c.label}.${kind}`,
    pipeline: pipelineForCase,
    group,
    output,
    dispatchX: kind === "tile" ? Math.ceil(c.T / 8) + c.segments.length : c.T,
    dispatchY: kind === "tile" ? c.kvHeads : c.heads,
    buffers,
  };
}

async function main() {
  const plans = selectedPlans();
  const seed = intParam("seed", 0x4b455641, 1, 0xffffffff);
  const warmups = intParam("warmups", 1, 0, 10);
  const samples = intParam("samples", 3, 1, 20);
  const gpu = await requestDevice();
  if (!gpu.subgroup32) throw new Error("FP32 Kev tile guard requires a hardware device with 32-lane subgroups and subgroup support");
  const device = gpu.device;
  U = GPUBufferUsage;
  activeDevice = gpu;
  const source = await kernelSource();
  const fallbackLayout = bindLayout(device, "fallback");
  const tileLayout = bindLayout(device, "tile");
  const fallbackPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fallbackLayout] });
  const tilePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [tileLayout] });
  const pipelines = new Map();
  const getPipelines = async (c) => {
    const key = `${c.heads}/${c.kvHeads}`;
    if (!pipelines.has(key)) {
      const kev = { ...KEV_SPEC, heads: c.heads, kv_heads: c.kvHeads };
      pipelines.set(key, Promise.all([
        pipeline(device, source("kev_attention", { subgroups: true, kev }), `kev_attention_${key}`, fallbackPipelineLayout),
        pipeline(device, source("kev_attention_tile", { f16: false, subgroups: true, kev }), `kev_attention_tile_${key}`, tilePipelineLayout),
      ]));
    }
    return pipelines.get(key);
  };
  const timer = makeTimer(device);
  const correctness = { ok: true, tolerance: TOLERANCE, cases: [] };
  const cases = [];
  const tileMedians = [];
  const fallbackMedians = [];
  log(`device: ${gpu.name}; timer: ${timer.method}; plans ${plans.map((plan) => plan.label).join(", ")}`);

  for (let caseIndex = 0; caseIndex < plans.length; caseIndex++) {
    const c = makeCase(plans[caseIndex], (seed + Math.imul(caseIndex, 0x9e3779b9)) >>> 0);
    const reference = cpuReference(c);
    const [fallbackPipeline, tilePipeline] = await getPipelines(c);
    const fallback = makeRun(device, fallbackLayout, fallbackPipeline, c, "fallback");
    const tile = makeRun(device, tileLayout, tilePipeline, c, "tile");
    const work = estimateWork(c);
    const reps = Math.max(1, Math.min(16, Math.floor(3e7 / Math.max(1, work))));
    for (let i = 0; i < warmups; i++) {
      await timedDispatch(device, timer, fallback, reps);
      await timedDispatch(device, timer, tile, reps);
    }
    const fallbackTimings = [];
    const tileTimings = [];
    for (let i = 0; i < samples; i++) {
      if (i % 2 === 0) {
        fallbackTimings.push(await timedDispatch(device, timer, fallback, reps));
        tileTimings.push(await timedDispatch(device, timer, tile, reps));
      } else {
        tileTimings.push(await timedDispatch(device, timer, tile, reps));
        fallbackTimings.push(await timedDispatch(device, timer, fallback, reps));
      }
    }
    const bytes = c.T * c.attnQ * 4;
    const fallbackValues = await readOutput(device, fallback.output, bytes);
    const tileValues = await readOutput(device, tile.output, bytes);
    const fallbackCpu = compareCpu(fallbackValues, reference);
    const tileCpu = compareCpu(tileValues, reference);
    const paired = compareArrays(tileValues, fallbackValues);
    if (!fallbackCpu.ok || !tileCpu.ok || !paired.ok) correctness.ok = false;
    correctness.cases.push({
      case: c.label,
      checkedRows: reference.checkedRows,
      checkedQueryHeads: reference.checkedQueryHeads,
      fallback: fallbackCpu,
      tile: tileCpu,
      paired,
    });
    const fallbackSorted = [...fallbackTimings].sort((a, b) => a - b);
    const tileSorted = [...tileTimings].sort((a, b) => a - b);
    const fallbackMedianMs = fallbackSorted[Math.floor(fallbackSorted.length / 2)] || 0;
    const tileMedianMs = tileSorted[Math.floor(tileSorted.length / 2)] || 0;
    if (c.T > 0) {
      fallbackMedians.push(fallbackMedianMs);
      tileMedians.push(tileMedianMs);
    }
    cases.push({
      case: c.label,
      T: c.T,
      heads: c.heads,
      kvHeads: c.kvHeads,
      headDim: DIM,
      segments: c.segments.map(({ label, len, parentId, plen }) => ({ label, len, parent: parentId, cachedPrefix: plen })),
      work,
      reps,
      fallbackSamples: fallbackTimings,
      tileSamples: tileTimings,
      fallbackMedianMs,
      tileMedianMs,
      speedup: fallbackMedianMs > 0 && tileMedianMs > 0 ? fallbackMedianMs / tileMedianMs : null,
      correctness: { fallback: fallbackCpu, tile: tileCpu, paired },
    });
    log(`${c.label}: fallback=${fallbackMedianMs.toFixed(4)} ms tile=${tileMedianMs.toFixed(4)} ms speedup=${cases.at(-1).speedup?.toFixed(3) || "n/a"}, ${fallbackCpu.ok && tileCpu.ok && paired.ok ? "correct" : "FAILED"} tileCpuMaxAbs=${tileCpu.maxAbs?.toExponential(2) || "n/a"} pairMaxAbs=${paired.maxAbs?.toExponential(2) || "n/a"}`);
    for (const buffer of [...fallback.buffers, ...tile.buffers]) buffer.destroy();
  }

  const geometricMean = (values) => values.length ? Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length) : 0;
  const metricMs = geometricMean(tileMedians);
  const result = {
    status: correctness.ok ? "done" : "error",
    done: true,
    backend: "webgpu",
    kernel: "kev_attention_tile",
    baselineKernel: "kev_attention",
    subgroups: true,
    metricMs,
    baselineMetricMs: geometricMean(fallbackMedians),
    metric: "geometric mean of per-plan median FP32 tiled Kev attention milliseconds",
    method: timer.method,
    seed,
    gateTails: query.get("gateTails") === "1",
    warmups,
    samples,
    cases,
    adapter: adapterInfo(gpu),
    correctness,
  };
  if (!correctness.ok) result.error = "Kev attention CPU or paired-output parity failed";
  window.gpuBench = result;
  log(`tileMetricMs=${metricMs.toFixed(4)}; fallbackMetricMs=${result.baselineMetricMs.toFixed(4)}; correctness=${correctness.ok ? "ok" : "FAILED"}`);
  timer.querySet?.destroy();
  timer.resolved?.destroy();
  timer.read?.destroy();
}

main().catch((error) => {
  const message = `${error?.message || error}\n${error?.stack || ""}`;
  window.gpuBench = { ...state, status: "error", done: true, error: message };
  log(`ERROR ${message}`);
}).finally(() => {
  activeDevice?.device.destroy();
  activeDevice = null;
});
