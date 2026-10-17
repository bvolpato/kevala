// DEV-only ordered O split reduction followed by the unchanged registered double RMS.
import { mmSplits, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "../js/src/gpu-gemma4-batch-mlp.js";
import { kernelSource } from "./wgsl.js";

const query = new URLSearchParams(location.search), U = GPUBufferUsage;
const CAP = 64 * 1024 * 1024, MAX_D = 1536, SENTINEL = 0x4640e500, POISON = 0x7fc00000;
const state = { status: "running", backend: "webgpu", kernel: "gemma4_reduce_attn_rms_probe" };
window.gpuBench = state;
const logNode = document.getElementById("log");
const log = (value) => { logNode.textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

function replaceOnce(source, old, replacement) {
  if (source.split(old).length !== 2) throw new Error(`candidate source marker must occur once: ${old}`);
  return source.replace(old, replacement);
}

export function deriveReduceAttnRmsSource(baseline) {
  let code = replaceOnce(baseline,
    "@group(0) @binding(2) var<storage, read> X: array<f32>;",
    "@group(0) @binding(2) var<storage, read> PART: array<f32>;");
  code = replaceOnce(code, "var<workgroup> sums: array<f32, 256>;",
    `var<workgroup> sums: array<f32, 256>;\nvar<workgroup> raw: array<f32, ${MAX_D}>;`);
  const rawRead = "X[base + d]";
  if (code.split(rawRead).length !== 3) throw new Error("expected exactly the two original branch reads in double RMS");
  code = code.replaceAll(rawRead, "raw[d]");
  code = replaceOnce(code, "  if (t >= g.T) { return; }", `  if (t >= g.T || p.D == 0u || p.D > ${MAX_D}u || p._a == 0u || p._a > 8u || p.mode > 2u) { return; }`);
  const prefix = `\n  for (var d = lane; d < p.D; d += 256u) {
    let o = base + d;
    var v = 0.0;
    for (var s = 0u; s < p._a; s++) { v += PART[s * g.T * p.D + o]; }
    raw[d] = v;
  }
  // Materialize the original reducer's F32 output before either RMS reads it.
  workgroupBarrier();`;
  code = replaceOnce(code, "  let base = t * p.D;", `  let base = t * p.D;${prefix}`);
  // Reverse every intentional change, protecting the norm trees, branches and storage boundary.
  let recovered = replaceOnce(code, `  let base = t * p.D;${prefix}`, "  let base = t * p.D;");
  recovered = replaceOnce(recovered, `  if (t >= g.T || p.D == 0u || p.D > ${MAX_D}u || p._a == 0u || p._a > 8u || p.mode > 2u) { return; }`, "  if (t >= g.T) { return; }");
  recovered = recovered.replaceAll("raw[d]", rawRead);
  recovered = replaceOnce(recovered, `var<workgroup> sums: array<f32, 256>;\nvar<workgroup> raw: array<f32, ${MAX_D}>;`, "var<workgroup> sums: array<f32, 256>;");
  recovered = replaceOnce(recovered, "@group(0) @binding(2) var<storage, read> PART: array<f32>;", "@group(0) @binding(2) var<storage, read> X: array<f32>;");
  if (recovered !== baseline) throw new Error("candidate changed the registered double-RMS body outside the ordered reduction and raw loads");
  return { code, roundTripExact: true, originalRawReads: 2, sharedBytes: (MAX_D + 256) * 4 };
}

function cases() {
  const out = [];
  for (const [T, policy, rows, splitTokens, logicalRows] of [[28, "source", 2, 64, 66], [97, "batch", 4, 1, 97]]) {
    for (const K of [2048, 4096]) out.push({ T, D: 1536, K, policy, rows, splitTokens, logicalRows, target: 256, pattern: "signed", mode: 2, timing: true });
  }
  if (query.get("cases") === "primary" || query.get("cases") === "timing") return out;
  for (let splits = 1; splits <= 8; splits++) out.push({ T: 17, D: 1536, K: 2048, policy: "batch", rows: 4, splitTokens: 1, target: 24 * splits, pattern: "cancellation", mode: 2, expectedSplits: splits });
  for (const D of [1, 17, 255, 257, 511, 1535]) out.push({ T: 3, D, K: 2048, policy: "batch", rows: 4, splitTokens: 1, target: Math.ceil(D / 64) * 3, pattern: "signed", mode: 2, expectedSplits: 3 });
  for (const T of [28, 97]) for (const pattern of ["zero", "large"]) out.push({ T, D: 1536, K: 2048, policy: "batch", rows: 4, splitTokens: 1, target: 256, pattern, mode: 2 });
  for (const mode of [0, 1]) out.push({ T: 17, D: 1536, K: 2048, policy: "batch", rows: 4, splitTokens: 1, target: 256, pattern: "signed", mode, weights: mode === 0 ? "poison-post" : "negative" });
  for (const weights of ["zero-post", "zero-next", "negative"]) out.push({ T: 17, D: 1536, K: 4096, policy: "batch", rows: 4, splitTokens: 1, target: 256, pattern: "signed", mode: 2, weights });
  return out;
}

function fixture(c) {
  const splits = mmSplits(c.splitTokens, c.D, c.K, c.target, c.rows * 16, 64), count = c.T * c.D;
  if (c.expectedSplits && splits !== c.expectedSplits) throw new Error("explicit and registered baseline split policies disagree");
  const bytes = (count + 4) * 4, partBytes = (splits * count + 4) * 4;
  // GPU buffers plus host fixtures, upload copies, readbacks and query allowance. Cases run serially.
  const estimateBytes = 3 * partBytes + 14 * bytes + 8 * c.D * 4 + 65536;
  if (estimateBytes > CAP) throw new Error(`case exceeds ${CAP} bytes: ${estimateBytes}`);
  const part = new Float32Array(partBytes / 4).fill(NaN), raw = new Float32Array(count + 4);
  const cancellation = [1048576, 0.015625, -1048576, 0.00737, -0.25, 0.1731, -0.0317, 0.413];
  for (let s = 0; s < splits; s++) for (let i = 0; i < count; i++) {
    let value;
    if (c.pattern === "zero") value = i % 2 ? -0 : 0;
    else if (c.pattern === "cancellation") value = cancellation[s] * (((i % 13) - 6) / 8);
    else value = (Math.sin(i * .017 + s * .137) * .731 + Math.cos(i * .031 - s * .193) * .417) * (s % 2 ? -1 : 1) * (1.337 + s * .193) * (c.pattern === "large" ? 1e12 : 1);
    part[s * count + i] = value;
    raw[i] = Math.fround(raw[i] + part[s * count + i]);
  }
  const seed = new Float32Array(count + 4), poison = new Float32Array(count + 4).fill(NaN);
  for (let i = 0; i < count; i++) seed[i] = c.pattern === "zero" ? i % 3 ? 0 : -0 : .473 * Math.sin(i * .137 - .417);
  for (const values of [raw, seed, poison]) new Uint32Array(values.buffer).fill(SENTINEL, count);
  const post = Float32Array.from({ length: c.D }, (_, i) => c.weights === "poison-post" ? NaN : c.weights === "zero-post" ? 0 : (c.weights === "negative" ? -1 : 1) * (.731 + 1.317 * Math.sin(i * .217 - .3)));
  const next = Float32Array.from({ length: c.D }, (_, i) => c.weights === "zero-next" ? (i % 2 ? -0 : 0) : .947 * Math.cos(i * .173 + .193) - .317);
  return { splits, count, bytes, partBytes, estimateBytes, part, raw, seed, poison, post, next };
}

async function checkCase(gpu, kernels, source, c, deadline) {
  const { device } = gpu, f = fixture(c), owned = [];
  let gpuBytes = 0, queries, mapped = null;
  const make = (bytes, usage, data, label) => {
    bytes = Math.max(16, bytes); gpuBytes += bytes;
    if (gpuBytes > CAP || bytes > device.limits.maxBufferSize || (usage & U.STORAGE) && bytes > device.limits.maxStorageBufferBindingSize) throw new Error("GPU scratch limit exceeded");
    const buffer = device.createBuffer({ size: bytes, usage, label }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (values) => {
    const data = new ArrayBuffer(16), ints = new Uint32Array(data), floats = new Float32Array(data);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(16, U.UNIFORM | U.COPY_DST, data, "probe params");
  };
  const bind = (p, values) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: values.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const read = async (buffer, bytes) => {
    if (!await mapBeforeDeadline(buffer, bytes, deadline)) { buffer.unmap(); throw new Error("probe readback deadline exceeded"); }
    mapped = buffer;
    try { return buffer.getMappedRange(0, bytes).slice(0); }
    finally { buffer.unmap(); mapped = null; }
  };
  try {
    const reduceName = c.policy === "source" ? "reduce_gemma_source" : "reduce_gemma_batch";
    const reduceCode = source(reduceName, { weights: "q8", f16: false, rows: c.rows, groups: 1, splitTarget: c.target });
    const reduce = await pipeline(device, reduceCode, `${reduceName}_r${c.rows}_target${c.target}`);
    const globals = uniform([c.T, c.splitTokens, c.logicalRows || c.T, 0]);
    const pReduce = uniform([c.D, c.K, 0, 0]), pRms = uniform([c.D, { f: 1e-6 }, c.mode, 0]), pFused = uniform([c.D, { f: 1e-6 }, c.mode, f.splits]);
    const PART = make(f.partBytes, U.STORAGE | U.COPY_DST, f.part, "readonly partials with poison tail");
    const Wpost = make(f.post.byteLength, U.STORAGE | U.COPY_DST, f.post, "post norm weights"), Wnext = make(f.next.byteLength, U.STORAGE | U.COPY_DST, f.next, "next norm weights");
    const bias = make(16, U.STORAGE | U.COPY_DST, new Float32Array(4).fill(NaN), "disabled bias");
    const seed = make(f.bytes, U.COPY_SRC | U.COPY_DST, f.seed, "immutable residual seed");
    const raw = make(f.bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, f.poison, "baseline reduced branch");
    const rawReference = make(f.bytes, U.STORAGE | U.COPY_DST, f.raw, "CPU ascending F32 reduction");
    const oldY = make(f.bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, f.seed, "baseline residual"), newY = make(f.bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, f.seed, "candidate residual");
    const oldH = make(f.bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, f.poison, "baseline next norm"), newH = make(f.bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, f.poison, "candidate next norm");
    const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, null, "bitwise guard status"), statusRead = make(16, U.MAP_READ | U.COPY_DST, null, "tiny guard readback");
    const reduceGroup = bind(reduce, [globals, pReduce, PART, bias, raw]);
    const oldGroup = bind(kernels.rms, [globals, pRms, raw, Wpost, oldY, Wnext, oldH]);
    const newGroup = bind(kernels.fused, [globals, pFused, PART, Wpost, newY, Wnext, newH]);
    const dispatch = (pass, candidate) => {
      if (!candidate) { pass.setPipeline(reduce); pass.setBindGroup(0, reduceGroup); pass.dispatchWorkgroups(Math.ceil(f.count / 256)); }
      pass.setPipeline(candidate ? kernels.fused : kernels.rms); pass.setBindGroup(0, candidate ? newGroup : oldGroup); pass.dispatchWorkgroups(c.T);
    };
    const guardParams = uniform([f.count + 4, f.count, SENTINEL, POISON]);
    const comparisons = [[rawReference, raw], [oldY, newY], [oldH, newH]].map(([baseline, candidate]) => gemma4BatchMlpValidationGroups(device, kernels.validator, { params: guardParams, baseline, candidate, status }));
    const verify = async () => {
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      encoder.copyBufferToBuffer(seed, 0, oldY, 0, f.bytes); encoder.copyBufferToBuffer(seed, 0, newY, 0, f.bytes);
      const fill = encoder.beginComputePass(); fill.setPipeline(kernels.validator.fill);
      // Poison all active raw/H outputs so untouched values cannot pass an equality check.
      for (const group of [comparisons[0].fill[1], ...comparisons[2].fill]) { fill.setBindGroup(0, group); fill.dispatchWorkgroups(Math.ceil((f.count + 4) / 256)); } fill.end();
      const pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(kernels.validator.compare);
      for (const group of comparisons) { compare.setBindGroup(0, group.compare); compare.dispatchWorkgroups(Math.ceil((f.count + 4) / 256)); } compare.end();
      encoder.copyBufferToBuffer(status, 0, statusRead, 0, 16); device.queue.submit([encoder.finish()]);
      const counts = new Uint32Array(await read(statusRead, 16));
      const result = { ok: counts.every((value) => value === 0), bitDifferences: counts[0], nonFinite: counts[1], tailDifferences: counts[2], checkedRawValues: f.count, checkedYValues: f.count, checkedHValues: f.count, signedZeroChecked: true, activePoisonAndTailGuard: true, readbackBytes: 16 };
      if (!result.ok) throw new Error(`strict double-RMS guard failed: ${JSON.stringify({ c, result })}`);
      return result;
    };
    const before = await verify(), timings = { baseline: [], candidate: [], pairs: [], repetitions: 4, pairsRequested: 3, residualResetExcluded: true };
    if (c.timing && query.get("timing") !== "0" && device.features.has("timestamp-query")) {
      const queryCount = 48;
      queries = device.createQuerySet({ type: "timestamp", count: queryCount });
      const resolved = make(queryCount * 8, U.QUERY_RESOLVE | U.COPY_SRC, null, "timestamps"), timingRead = make(queryCount * 8, U.COPY_DST | U.MAP_READ, null, "timestamp readback");
      const encoder = device.createCommandEncoder(), records = [];
      for (const candidate of [false, true]) {
        encoder.copyBufferToBuffer(seed, 0, candidate ? newY : oldY, 0, f.bytes);
        const pass = encoder.beginComputePass(); dispatch(pass, candidate); pass.end();
      }
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) for (let repetition = 0; repetition < 4; repetition++) {
        encoder.copyBufferToBuffer(seed, 0, candidate ? newY : oldY, 0, f.bytes);
        const index = records.length * 2, pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        dispatch(pass, candidate); pass.end(); records.push({ pair, candidate, index });
      }
      encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, timingRead, 0, queryCount * 8); device.queue.submit([encoder.finish()]);
      const stamps = new BigUint64Array(await read(timingRead, queryCount * 8));
      for (let pair = 0; pair < 3; pair++) {
        const values = { baseline: [], candidate: [] };
        for (const item of records.filter((record) => record.pair === pair)) {
          if (stamps[item.index + 1] <= stamps[item.index]) throw new Error("nonpositive timestamp interval");
          values[item.candidate ? "candidate" : "baseline"].push(Number(stamps[item.index + 1] - stamps[item.index]) / 1e6);
        }
        const baseline = mean(values.baseline), candidate = mean(values.candidate);
        timings.baseline.push(baseline); timings.candidate.push(candidate); timings.pairs.push({ pair, ...values, baselineMeanMs: baseline, candidateMeanMs: candidate, benefit: 1 - candidate / baseline });
      }
      timings.baselineMedianMs = median(timings.baseline); timings.candidateMedianMs = median(timings.candidate);
      timings.benefit = 1 - timings.candidateMedianMs / timings.baselineMedianMs;
      timings.stableFivePercentWin = timings.candidate.every((value, i) => value <= timings.baseline[i] * .95);
    }
    const after = await verify();
    return { ...c, splits: f.splits, arithmetic: "synthetic F32 PART only; no matmul or precision change", before, after, timings, ownedGpuBufferBytes: gpuBytes, estimatedHostGpuBytes: f.estimateBytes, ok: true };
  } finally {
    mapped?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy();
    for (const value of [f.part, f.raw, f.seed, f.poison, f.post, f.next]) value.fill(0);
  }
}

async function main() {
  const started = performance.now(), deadline = started + 120000, subgroups = query.get("subgroups") === "1", f16Feature = query.get("f16") === "1";
  const gpu = await requestDevice({ features: ["timestamp-query", ...(subgroups ? ["subgroups"] : []), ...(f16Feature ? ["shader-f16"] : [])] });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const info = gpu.adapter.info || {};
    if (subgroups && (!gpu.device.features.has("subgroups") || info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32)) {
      window.gpuBench = { ...state, status: "done", done: true, skipped: "fixed32 subgroups unavailable" }; return;
    }
    if (f16Feature && !gpu.device.features.has("shader-f16")) { window.gpuBench = { ...state, status: "done", done: true, skipped: "shader-f16 unavailable" }; return; }
    const source = await kernelSource(), baseline = source("gemma4_rms_residual_norm", { subgroups }), candidate = deriveReduceAttnRmsSource(baseline);
    const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");
    const kernels = await withGpuErrors(gpu, async () => ({
      rms: await pipeline(gpu.device, baseline, "gemma4_rms_residual_norm"), fused: await pipeline(gpu.device, candidate.code, "gemma4_reduce_attn_rms_probe"), validator: await createGemma4BatchMlpValidator(gpu.device),
    }), "compiling ordered O reduction/double RMS probe");
    const results = [];
    for (const c of cases()) {
      if (performance.now() >= deadline) throw new Error("probe deadline exceeded");
      const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, source, c, deadline), "checking ordered O reduction/double RMS");
      results.push(result); log(result);
    }
    if (uncaptured.length) throw new Error(`uncaptured GPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started, subgroups, f16Feature,
      adapter: { name: gpu.name, features: [...gpu.device.features], subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize },
      compile: { ok: true, derivationRoundTripExact: candidate.roundTripExact, originalRawReads: candidate.originalRawReads, baselineSha256: await hash(baseline), candidateSha256: await hash(candidate.code), workgroupSize: 256, sharedBytes: candidate.sharedBytes, storageBindings: 5, unchangedStoredResidualBoundary: true },
      resourceBounds: { budgetBytes: CAP, maxCaseEstimateBytes: Math.max(...results.map((value) => value.estimatedHostGpuBytes)), sequentialCases: true, perCaseDeadlineMs: null, overallDeadlineMs: 120000, fullOutputsComparedOnGpu: true },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code }; log(error.stack || error); });
