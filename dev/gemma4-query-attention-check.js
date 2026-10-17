// DEV-only query RMSNorm/RoPE + attention. No model, production kernels, or runtime changes.
import { pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const search = new URLSearchParams(location.search), U = GPUBufferUsage;
const state = { status: "running", backend: "webgpu", kernel: "gemma4_query_attention_probe" };
window.gpuBench = state;
const log = (value) => { document.getElementById("log").textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const CAP = 64 * 1024 * 1024, EPS = 1e-6;
const TAIL = Float32Array.of(12345.25, -1731.5, 913.75, -0);
const f32 = Math.fround;
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

function replaceOnce(source, before, after) {
  if (source.split(before).length !== 2) throw new Error(`registered source drift: expected one ${before}`);
  return source.replace(before, after);
}

function deriveQueryAttentionSource(attention, qkv, subgroup) {
  const oldParams = "_c: u32 }";
  const newParams = "_c: u32, rotary_dim: u32, table: u32, table_stride: u32, eps: f32 }";
  const oldLoad = "  for (var d = lane; d < H; d += 256u) { qtile[d] = Q[qbase + d]; }";
  const write = "    if (kind == 0u) { Q[base + d] = value; }\n    else if (kind == 1u) { K[base + d] = value; }\n    else { V[base + d] = value; }";
  const begin = qkv.indexOf("  var sum = 0.0;"), end = qkv.lastIndexOf("\n}");
  const ropeBegin = qkv.indexOf("fn rope_index("), ropeEnd = qkv.indexOf("@compute", ropeBegin);
  if (begin < 0 || end <= begin || ropeBegin < 0 || ropeEnd <= ropeBegin) throw new Error("registered QKV source markers changed");
  let norm = replaceOnce(qkv.slice(begin, end), write, "    qtile[d] = value;");
  norm = replaceOnce(norm, "learned = KN[d];", "learned = QN[d];"); // The K branch is unreachable for kind=0.
  const sumName = subgroup ? "query_sums" : "partial";
  norm = norm.replace(/\bsums\b/g, sumName);
  const prologue = `  let kind = 0u;\n  let half = H / 2u;\n${norm}`;
  const declarations = "@group(0) @binding(7) var<storage, read> QN: array<f32>;\n@group(0) @binding(8) var<storage, read> CS: array<vec2<f32>>;\nvar<workgroup> normed: array<f32, 512>;\n" + (subgroup ? "var<workgroup> query_sums: array<f32, 256>;\n" : "") + qkv.slice(ropeBegin, ropeEnd);
  let code = replaceOnce(attention, oldParams, newParams);
  code = replaceOnce(code, "var<workgroup> qtile:", declarations + "var<workgroup> qtile:");
  code = replaceOnce(code, oldLoad, prologue);
  let restored = replaceOnce(code, newParams, oldParams);
  restored = replaceOnce(restored, declarations, "");
  restored = replaceOnce(restored, prologue, oldLoad);
  if (restored !== attention) throw new Error("candidate changed attention outside params, new bindings, and query prologue");
  const afterLoad = attention.slice(attention.indexOf(oldLoad) + oldLoad.length);
  if (!code.endsWith(afterLoad)) throw new Error("candidate did not retain the complete attention suffix byte-for-byte");
  return { code, exactAttentionSuffix: true, originalNormTree: true, normBodyHashInput: norm, workgroupSize: 256, storageBindings: 7, sharedBytes: 5648 };
}

function branched(dim, prefix = 66, lengths = [42, 55], tail = true, pattern = "signed", label = "actual-tail") {
  const positions = [], offsets = [];
  let offset = 0;
  for (const length of lengths) {
    for (const row of tail ? [length - 1] : Array.from({ length }, (_, i) => i)) { positions.push(prefix + row); offsets.push(offset); }
    offset += length;
  }
  return { label, dim, prefix, lengths, T: positions.length, S: prefix + offset, heads: 8, kvHeads: 1,
    window: dim === 256 ? 512 : 0, causal: 1, batch: true, rotary: dim === 256 ? 256 : 128, table: dim === 256 ? 0 : 1,
    positions: Uint32Array.from(positions), offsets: Uint32Array.from(offsets), pattern, timing: pattern === "signed" };
}

function cases() {
  const primary = [branched(256), branched(512)];
  if (search.get("cases") === "primary") return primary;
  const out = [...primary];
  for (const dim of [256, 512]) {
    out.push(branched(dim, 89, [12], false, "signed", "representative-T12"), branched(dim, 89, [42, 55], false, "signed", "representative-T97"));
    for (const pattern of ["zero", "large", "cancellation"]) out.push({ ...branched(dim, 117, [42, 55], true, pattern, `${pattern}-tail`), window: 65 });
    out.push({ ...branched(dim, 66, [1], true, "signed", "single-query-no-rotation"), rotary: 0, timing: false });
  }
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function fixture(c, seed) {
  let randomState = seed >>> 0;
  const random = () => { randomState = Math.imul(randomState ^ (randomState >>> 16), 0x45d9f3b); randomState = Math.imul(randomState ^ (randomState >>> 16), 0x45d9f3b); randomState ^= randomState >>> 16; return (randomState >>> 0) / 0x100000000; };
  const qCount = c.T * c.heads * c.dim, kvCount = c.S * c.kvHeads * c.dim;
  const q = new Float32Array(qCount + TAIL.length), k = new Float32Array(kvCount + TAIL.length).fill(NaN), v = k.slice();
  for (let i = 0; i < qCount; i++) q[i] = c.pattern === "zero" ? i % 2 ? -0 : 0 : c.pattern === "large" ? (i % 2 ? -1 : 1) * (1e8 + (i % 13) * 8192) : c.pattern === "cancellation" ? (i % 2 ? -1 : 1) * (0.1731 + (i % 13) * 0.00737) : (random() - 0.5) * 0.5;
  q.set(TAIL, qCount);
  for (let i = 0; i < kvCount; i++) {
    k[i] = c.pattern === "zero" ? i % 3 ? 0 : -0 : c.pattern === "cancellation" ? (i % 2 ? -1 : 1) * 0.0713 : (random() - 0.5) * 0.5;
    v[i] = c.pattern === "zero" ? i % 5 ? -0 : 0 : c.pattern === "cancellation" ? (Math.floor(i / c.dim) % 2 ? -1 : 1) * (0.317 + (i % 19) * 0.00437) : (random() - 0.5) * 0.5;
  }
  const qn = Float32Array.from({ length: c.dim + TAIL.length }, (_, i) => i >= c.dim ? TAIL[i - c.dim] : (((i * 11 + seed) % 19) - 9) * 0.1837);
  const positions = new Uint32Array(c.T * 2 + TAIL.length).fill(0xffffffff); positions.set(c.positions); positions.set(c.offsets, c.T);
  const ropeRows = 2 ** Math.ceil(Math.log2(Math.max(64, ...c.positions) + 1)), tableStride = ropeRows * 128;
  const cs = new Float32Array(2 * tableStride * 2 + TAIL.length);
  for (const [table, dim, rotary, theta] of [[0, 256, 256, 10000], [1, 512, 128, 1000000]]) for (let pair = 0; pair < 128; pair++) {
    const inv = pair < rotary / 2 ? f32(1 / f32(Math.pow(theta, 2 * pair / dim))) : 0;
    for (let position = 0; position < ropeRows; position++) { const at = ((table * ropeRows + position) * 128 + pair) * 2, angle = f32(position * inv); cs[at] = Math.cos(angle); cs[at + 1] = Math.sin(angle); }
  }
  cs.set(TAIL, cs.length - TAIL.length);
  for (let t = 0; t < c.T; t++) {
    const end = c.causal ? c.positions[t] + 1 : c.S;
    const physicalLast = end - 1 + (c.batch && end - 1 >= c.prefix ? c.offsets[t] : 0);
    if (physicalLast >= c.S) throw new Error("fixture branch exceeds physical K/V buffer");
  }
  return { q, k, v, qn, positions, cs, tableStride, qCount, kvCount };
}

function strictBits(label, expected, actual, active = 0) {
  if (expected.byteLength !== actual.byteLength) throw new Error(`${label} length mismatch`);
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.byteLength / 4), b = new Uint32Array(actual.buffer, actual.byteOffset, actual.byteLength / 4);
  let differences = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (i < active && (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i]))) throw new Error(`${label} nonfinite active value ${i}`);
    if (a[i] !== b[i]) { differences++; firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] }; }
  }
  if (differences) throw new Error(`${label}: ${JSON.stringify({ differences, firstDifference })}`);
  return { ok: true, bitExact: true, checkedWords: a.length, activeFinite: active, signedZeroBits: true };
}

function checkNormalizedQuery(c, input, actual) {
  let maxAbs = 0;
  const sums = new Float32Array(256), normed = new Float32Array(c.dim);
  for (let t = 0; t < c.T; t++) for (let head = 0; head < c.heads; head++) {
    const base = (t * c.heads + head) * c.dim;
    for (let lane = 0; lane < 256; lane++) { let sum = 0; for (let d = lane; d < c.dim; d += 256) sum = f32(sum + f32(input.q[base + d] * input.q[base + d])); sums[lane] = sum; }
    for (let stride = 128; stride; stride >>= 1) for (let lane = 0; lane < stride; lane++) sums[lane] = f32(sums[lane] + sums[lane + stride]);
    const inv = f32(1 / Math.sqrt(f32(f32(sums[0] / c.dim) + f32(EPS))));
    for (let d = 0; d < c.dim; d++) normed[d] = f32(f32(input.q[base + d] * inv) * input.qn[d]);
    for (let d = 0; d < c.dim; d++) {
      const pair = d % (c.dim / 2), at = (c.table * input.tableStride + c.positions[t] * 128 + pair) * 2;
      const a = normed[pair], b = normed[pair + c.dim / 2], cosine = input.cs[at], sine = input.cs[at + 1];
      const value = pair < c.rotary / 2 ? d < c.dim / 2 ? f32(f32(a * cosine) - f32(b * sine)) : f32(f32(b * cosine) + f32(a * sine)) : normed[d];
      const error = Math.abs(actual[base + d] - value); maxAbs = Math.max(maxAbs, error);
      if (!Number.isFinite(actual[base + d]) || error > 1e-5 + 1e-5 * Math.abs(value)) throw new Error(`baseline query norm CPU reference mismatch at ${base + d}: ${actual[base + d]} vs ${value}`);
    }
  }
  strictBits("normalized query tail", TAIL, actual.subarray(input.qCount));
  return { ok: true, checkedValues: input.qCount, maxAbs, cpuTolerance: { absolute: 1e-5, relative: 1e-5 }, candidateTolerance: { absolute: 0, relative: 0 } };
}

function median(values) { const ordered = [...values].sort((a, b) => a - b), middle = ordered.length >> 1; return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2; }

async function checkCase(gpu, kernels, c, seed) {
  const d = gpu.device, input = fixture(c, seed), owned = [];
  const qBytes = input.q.byteLength, maxReadBytes = Math.max(...Object.values(input).filter(ArrayBuffer.isView).map((v) => v.byteLength));
  const poison = new Float32Array(input.q.length).fill(NaN); poison.set(TAIL, input.qCount);
  const hostFixtureBytes = [...Object.values(input).filter(ArrayBuffer.isView), poison, c.positions, c.offsets].reduce((sum, array) => sum + array.byteLength, 0);
  // Include mapped/sliced snapshots, CPU references, query buffers, and a full upload reserve.
  const hostReserveBytes = hostFixtureBytes * 2 + maxReadBytes * 6 + 16384;
  let gpuBytes = 0, mapped = false, readback;
  const make = (bytes, usage, data, label) => {
    bytes = Math.max(16, bytes); gpuBytes += bytes;
    if (gpuBytes + hostReserveBytes > CAP) throw new Error(`controlled host+GPU memory exceeds ${CAP}`);
    const buffer = d.createBuffer({ size: bytes, usage, label }); owned.push(buffer);
    if (data) d.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const storage = (data, label) => make(data.byteLength, U.STORAGE | U.COPY_SRC | U.COPY_DST, data, label);
  const bind = (kernel, values) => d.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: values.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const uniform = (data, label) => make(data.byteLength, U.UNIFORM | U.COPY_DST, data, label);
  let querySet;
  try {
    const globals = uniform(Uint32Array.of(c.T, 0, c.S, c.prefix), "query attention globals");
    const paramsRaw = new ArrayBuffer(48), paramsInts = new Uint32Array(paramsRaw), paramsFloats = new Float32Array(paramsRaw);
    paramsInts.set([c.heads, c.kvHeads, c.dim, c.window, c.causal, Number(c.batch), 0, 0, c.rotary, c.table, input.tableStride]); paramsFloats[11] = EPS;
    const params = uniform(new Uint32Array(paramsRaw), "query attention params");
    const normRaw = new ArrayBuffer(48), normInts = new Uint32Array(normRaw), normFloats = new Float32Array(normRaw);
    normInts.set([c.heads, c.kvHeads, c.dim, c.rotary, c.table, input.tableStride, 0, 0]); normFloats[8] = EPS;
    const normParams = uniform(new Uint32Array(normRaw), "query-only QKV params");
    const Q = storage(input.q, "immutable raw query"), oldQ = storage(input.q, "baseline normalized query");
    const K = storage(input.k, "immutable K with NaN tail"), V = storage(input.v, "immutable V with NaN tail"), QN = storage(input.qn, "query norm learned weights"), CS = storage(input.cs, "RoPE tables"), POS = storage(input.positions, "logical positions and offsets");
    const disabledPoison = Float32Array.of(NaN, NaN, NaN, NaN);
    const dummyK = storage(disabledPoison, "disabled QKV K poison"), dummyV = storage(disabledPoison, "disabled QKV V poison"), dummyKN = storage(disabledPoison, "disabled QKV KN poison");
    const outputs = [storage(poison, "baseline output poison"), storage(poison, "candidate output poison")];
    const normGroup = bind(kernels.norm, [globals, normParams, oldQ, dummyK, dummyV, QN, dummyKN, CS, POS]);
    const oldGroup = bind(kernels.baseline, [globals, params, oldQ, K, V, outputs[0], POS]);
    const newGroup = bind(kernels.candidate, [globals, params, Q, K, V, outputs[1], POS, QN, CS]);
    readback = make(maxReadBytes, U.COPY_DST | U.MAP_READ, null, "staged full-buffer guard readback");
    const read = async (buffer, bytes, type = Float32Array) => {
      const encoder = d.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, readback, 0, bytes); d.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ, 0, bytes); mapped = true;
      try { return new type(readback.getMappedRange(0, bytes).slice(0)); } finally { readback.unmap(); mapped = false; }
    };
    const dispatch = (pass, kernel, group) => { pass.setPipeline(kernel); pass.setBindGroup(0, group); pass.dispatchWorkgroups(c.T, c.heads); };
    const mode = (encoder, candidate, timestamp = null) => {
      if (!candidate) encoder.copyBufferToBuffer(Q, 0, oldQ, 0, qBytes);
      const pass = encoder.beginComputePass(timestamp !== null ? { timestampWrites: { querySet, beginningOfPassWriteIndex: timestamp, endOfPassWriteIndex: timestamp + 1 } } : undefined);
      if (!candidate) dispatch(pass, kernels.norm, normGroup);
      dispatch(pass, candidate ? kernels.candidate : kernels.baseline, candidate ? newGroup : oldGroup); pass.end();
    };
    const verify = async (label) => {
      d.queue.writeBuffer(outputs[0], 0, poison); d.queue.writeBuffer(outputs[1], 0, poison);
      const encoder = d.createCommandEncoder(); mode(encoder, false); mode(encoder, true); d.queue.submit([encoder.finish()]);
      const oldOutput = await read(outputs[0], qBytes), newOutput = await read(outputs[1], qBytes);
      const parity = strictBits(`${label} attention`, oldOutput, newOutput, input.qCount);
      strictBits(`${label} baseline output tail`, TAIL, oldOutput.subarray(input.qCount)); strictBits(`${label} candidate output tail`, TAIL, newOutput.subarray(input.qCount));
      const normalizedQuery = checkNormalizedQuery(c, input, await read(oldQ, qBytes));
      const immutable = [];
      for (const [name, buffer, expected] of [["rawQ", Q, input.q], ["K", K, input.k], ["V", V, input.v], ["QN", QN, input.qn], ["CS", CS, input.cs], ["POS", POS, input.positions], ["disabledK", dummyK, disabledPoison], ["disabledV", dummyV, disabledPoison], ["disabledKN", dummyKN, disabledPoison]]) immutable.push({ name, ...strictBits(`${label} immutable ${name}`, expected, await read(buffer, expected.byteLength, expected.constructor)) });
      return { parity, normalizedQuery, immutable, outputTailGuard: true };
    };
    const before = await verify("before timing"), records = [];
    const timingAvailable = c.timing && search.get("timing") !== "0" && d.features.has("timestamp-query");
    if (timingAvailable) {
      const warm = d.createCommandEncoder(); for (let i = 0; i < 4; i++) { mode(warm, false); mode(warm, true); } d.queue.submit([warm.finish()]);
      const count = 16, timingBytes = count * 8;
      querySet = d.createQuerySet({ type: "timestamp", count });
      const resolved = make(timingBytes, U.QUERY_RESOLVE | U.COPY_SRC, null, "query attention resolved timestamps");
      const timingRead = make(timingBytes, U.COPY_DST | U.MAP_READ, null, "query attention timestamps readback");
      for (let pair = 0; pair < 3; pair++) {
        const encoder = d.createCommandEncoder(), samples = [];
        let index = 0;
        for (let repeat = 0; repeat < 4; repeat++) for (const candidate of (pair + repeat) % 2 ? [true, false] : [false, true]) { mode(encoder, candidate, index); samples.push({ candidate, index }); index += 2; }
        encoder.resolveQuerySet(querySet, 0, count, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, timingRead, 0, timingBytes); d.queue.submit([encoder.finish()]);
        await timingRead.mapAsync(GPUMapMode.READ); const times = new BigUint64Array(timingRead.getMappedRange().slice(0)); timingRead.unmap();
        const baseline = [], candidate = [];
        for (const sample of samples) { const ns = times[sample.index + 1] - times[sample.index]; if (ns <= 0n) throw new Error("nonpositive GPU timestamp interval"); (sample.candidate ? candidate : baseline).push(Number(ns) / 1e6); }
        records.push({ pair, baselineMs: median(baseline), candidateMs: median(candidate), baselineSamplesMs: baseline, candidateSamplesMs: candidate });
      }
    }
    const after = await verify("after timing");
    const baselineMs = records.map((r) => r.baselineMs), candidateMs = records.map((r) => r.candidateMs);
    const timing = { available: Boolean(timingAvailable), scope: "composite GPU compute pass: baseline Q-only QKV + original attention vs fused attention; artificial raw-Q reset copy excluded", alternatingPairs: records.length, repetitionsPerMode: 4, records,
      ...(records.length ? { baselineMedianMs: median(baselineMs), candidateMedianMs: median(candidateMs), stableFivePercentWin: records.every((r) => r.candidateMs <= r.baselineMs * 0.95) } : { reason: !c.timing ? "guard-only-case" : "timestamps-unavailable-or-disabled" }) };
    const { positions: ignoredPositions, offsets: ignoredOffsets, ...summary } = c;
    return { ...summary, variant: kernels.variant, positionRange: [Math.min(...c.positions), Math.max(...c.positions)], branchOffsets: [...new Set(c.offsets)], before, after, timing,
      resources: { gpuBufferBytes: gpuBytes, hostFixtureBytes, hostAndUploadReserveBytes: hostReserveBytes, controlledPeakBytes: gpuBytes + hostReserveBytes, capBytes: CAP, learnedWeightsBytes: input.qn.byteLength, rotaryTableBytes: input.cs.byteLength, querySetQueries: querySet ? 16 : 0, sequentialCases: true }, ok: true };
  } finally { if (mapped) readback.unmap(); querySet?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), gpu = await requestDevice({ features: ["subgroups", "timestamp-query"] });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const info = gpu.adapter.info || {}, subgroupId = Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
    const subgroupAvailable = gpu.device.features.has("subgroups") && subgroupId && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
    const option = search.get("subgroups") || "auto";
    if (option === "only" && !subgroupAvailable) { window.gpuBench = { ...state, status: "skipped", done: true, reason: "fixed32-subgroup-id-unavailable" }; log(window.gpuBench); return; }
    const source = await kernelSource(), normSource = source("gemma4_qkv"), compilation = [], results = [];
    const norm = await withGpuErrors(gpu, () => pipeline(gpu.device, normSource, "query-attention baseline query-only QKV"), "compiling query-only QKV baseline");
    const variants = option === "only" ? ["subgroup"] : ["portable", ...(option !== "0" && subgroupAvailable ? ["subgroup"] : [])];
    for (const variant of variants) {
      const original = source(variant === "subgroup" ? "gemma4_attention_subgroup" : "gemma4_attention"), candidate = deriveQueryAttentionSource(original, normSource, variant === "subgroup");
      const kernels = await withGpuErrors(gpu, async () => ({ variant, norm,
        baseline: await pipeline(gpu.device, original, `${variant} query attention baseline`), candidate: await pipeline(gpu.device, candidate.code, `${variant} fused query attention probe`) }), `compiling ${variant} query attention probe`);
      compilation.push({ variant, baselineSha256: await hash(original), candidateSha256: await hash(candidate.code), qkvSha256: await hash(normSource), normBodySha256: await hash(candidate.normBodyHashInput), workgroupSize: candidate.workgroupSize, sharedBytes: candidate.sharedBytes, storageBindings: candidate.storageBindings, exactAttentionSuffix: candidate.exactAttentionSuffix, originalNormTree: candidate.originalNormTree });
      for (const [index, c] of cases().entries()) { const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, c, (0x4b455641 + Math.imul(index, 0x9e3779b9)) >>> 0), `checking ${variant} ${c.label} T${c.T} D${c.dim}`); results.push(result); log(result); }
    }
    if (uncaptured.length) throw new Error(`uncaptured GPU errors: ${uncaptured.join("; ")}`);
    const timed = results.filter((r) => r.timing.available), elapsedMs = performance.now() - started;
    window.gpuBench = { ...state, status: "done", done: true, elapsedMs,
      ...(timed.length ? { metricMs: Math.exp(timed.reduce((sum, r) => sum + Math.log(r.timing.candidateMedianMs), 0) / timed.length), method: "geometric mean candidate composite GPU milliseconds; no FPS inference" } : { method: "numerical guard only; no GPU timing metric" }),
      adapter: { name: gpu.name, features: [...gpu.device.features], subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, subgroupId },
      subgroups: { available: subgroupAvailable, requested: option, skipped: option !== "0" && !subgroupAvailable }, compile: { ok: true, source: "registered production attention/QKV with reversible query-prologue replacement", variants: compilation },
      resourceBounds: { capBytes: CAP, maxControlledPeakBytes: Math.max(...results.map((r) => r.resources.controlledPeakBytes)), sequentialCases: true }, correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code }; log(error.stack || String(error)); });
