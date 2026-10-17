import { matmulLayout, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const LIMIT = 64 * 1024 * 1024, SENTINEL = 12345.25, T = 2;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function activeHelper(helper, width, name = "ordered_active_sum32") {
  const signature = "fn ordered_sum32(prefix: f32, subgroupLane: u32) -> f32";
  const result = "return subgroupBroadcastFirst(aggregate);";
  if (helper.split(signature).length !== 2 || helper.split(result).length !== 2) throw new Error("active subgroup helper marker mismatch");
  return helper.replace(signature, `fn ${name}(prefix: vec${width}<f32>, subgroupLane: u32) -> vec${width}<f32>`).replace(result, "return aggregate;");
}

function subgroupReduction(original, helper, kind, strategy = "all") {
  if (original.includes("enable f16;") || !original.includes("@workgroup_size(256)")) throw new Error("two-row subgroup guard requires the F32 WG256 kernel");
  let code = original;
  const replace = (from, to) => {
    if (code.split(from).length !== 2) throw new Error(`two-row subgroup transform marker mismatch: ${from}`);
    code = code.replace(from, to);
  };
  const partial = kind === "plain" ? "pair_partials" : "partials", width = kind === "plain" ? 2 : 4;
  replace("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {",
    "fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) subgroupLane: u32) {");
  const oldTree = `  for (var stride = 32u; stride > 0u; stride /= 2u) {\n    if (lane < stride) { ${partial}[li] += ${partial}[li + stride]; }\n    workgroupBarrier();\n  }`;
  const prefix = `  var prefix = ${partial}[li];\n  if (lane < 32u) { prefix += ${partial}[li + 32u]; }`;
  if (strategy === "active") replace(oldTree, `${prefix}\n  var reduced = vec${width}<f32>(0.0);\n  let lowerHalfSubgroup = subgroupBroadcastFirst(lane) < 32u;\n  if (lowerHalfSubgroup) { reduced = ordered_active_sum32(prefix, subgroupLane); }`);
  else if (strategy === "vector") replace(oldTree, `${prefix}\n  let reduced = ordered_vector_sum32(prefix, subgroupLane);`);
  else {
    const components = "xyzw".slice(0, width).split("").map((component) => `ordered_sum32(prefix.${component}, subgroupLane)`);
    replace(oldTree, `${prefix}\n  // Every subgroup, including each unused upper half, executes every ordered tree.\n  let reduced = vec${width}<f32>(${components.join(", ")});`);
  }
  if (kind === "plain") replace("var value = pair_partials[li][row];", "var value = reduced[row];");
  else replace("let result = partials[li];", "let result = reduced;");
  if ((original.match(/workgroupBarrier\(\)/g) || []).length !== 2 || (code.match(/workgroupBarrier\(\)/g) || []).length !== 1) throw new Error("two-row subgroup barrier marker mismatch");
  return `enable subgroups;\n${helper}\n\n${code}`;
}

function exact(a, b) {
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    if (x[i] !== y[i]) { bitMismatches++; firstDifference ||= { index: i, baselineBits: x[i], candidateBits: y[i], baseline: a[i], candidate: b[i] }; }
  }
  return { ok: a.length === b.length && !bitMismatches && !nonfinite, samples: a.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

async function drain(device, deadline) {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error("two-row subgroup upload deadline exceeded");
  let timer;
  try {
    await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("two-row subgroup upload deadline exceeded")), remaining);
    })]);
  } finally { clearTimeout(timer); }
}

async function compile(device, source, kind, strategy, registered) {
  const spec = { weights: "q8", f16: false, rows: 1, groups: 1, splitTarget: 256 };
  const baselineCode = source(kind === "plain" ? "matvec_two" : "matvec_gelu_two", spec);
  const norm = source("gemma4_rms", { ...spec, subgroups: true });
  const helpers = [...norm.matchAll(/fn ordered_sum32\(prefix: f32, subgroupLane: u32\) -> f32 \{[\s\S]*?\n\}/g)];
  if (helpers.length !== 1 || !helpers[0][0].includes("subgroupShuffleDown(aggregate, step)") || !helpers[0][0].includes("subgroupBroadcastFirst(aggregate)")) throw new Error("rendered ordered_sum32 helper was not recognized");
  const productionHelper = helpers[0][0], helper = strategy === "all" ? productionHelper
    : activeHelper(productionHelper, kind === "plain" ? 2 : 4, strategy === "vector" ? "ordered_vector_sum32" : "ordered_active_sum32");
  const candidateCode = registered ? source(kind === "plain" ? "matvec_two" : "matvec_gelu_two", { ...spec, subgroups: true }) : subgroupReduction(baselineCode, helper, kind, strategy);
  if (registered && (!candidateCode.includes("enable subgroups;") || !candidateCode.includes("ordered_vector_sum32") || (candidateCode.match(/workgroupBarrier\(\)/g) || []).length !== 1)) throw new Error("registered subgroup kernel is unavailable; rebuild the WASM first");
  const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
  const layout = kind === "plain" ? matmulLayout(device) : device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const [baselineHash, candidateHash, helperHash, productionHelperHash] = await Promise.all([hash(baselineCode), hash(candidateCode), hash(helper), hash(productionHelper)]);
  const shaderHashes = { baseline: baselineHash, candidate: candidateHash, orderedSum32: helperHash, productionOrderedSum32: productionHelperHash };
  (metadata.compilations ||= {})[kind] = { strategy, registered, shaderHashes };
  const [baseline, candidate] = await Promise.all([
    pipeline(device, baselineCode, `matvec_two_${kind}_shared`, pl), pipeline(device, candidateCode, `matvec_two_${kind}_subgroup_${strategy}`, pl),
  ]);
  return { kind, strategy, registered, layout, baseline, candidate, shaderHashes };
}

async function check(runtime, pipes, c) {
  const { device } = runtime, U = GPUBufferUsage, { N, K, mode = 0, bias = 0, fixture = "normal" } = c;
  const count = T * N, xBytes = T * K * 4, yBytes = (count + 4) * 4, biasBytes = Math.max(16, N * 4), readBytes = Math.max(256, 2 * yBytes + 32);
  const sharedWeightBytes = (pipes.kind === "gelu" ? 2 : 1) * N * K * 9 / 8;
  const uploadStagingBytes = N * K * 9 / 8;
  // Include all host upload weights, F32 input/output copies, and mapped/readback staging.
  const scratchBytes = sharedWeightBytes + uploadStagingBytes + 3 * xBytes + 5 * yBytes + 3 * readBytes + 3 * biasBytes + 16384;
  if (scratchBytes > LIMIT) throw new Error("two-row subgroup temporary scratch exceeds 64 MiB");
  if (Math.max(N * K, xBytes, yBytes, biasBytes, readBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("two-row subgroup case exceeds device buffer limits");
  const deadline = performance.now() + 30000, owned = [];
  let read, mapped = false, query;
  const make = (size, flags, data) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage: flags }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  try {
    const weightsFor = async (offset) => {
      const weights = Int8Array.from({ length: N * K }, (_, i) => (i * 31 + Math.floor(i / K) * 17 + offset) % 256 - 128);
      const scales = Float32Array.from({ length: N * K / 32 }, (_, i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 7 * 0.000037));
      if (fixture === "saturation") for (let col = 0; col < Math.min(2, N); col++) { weights.fill(offset ? 17 : col ? -128 : 127, col * K, (col + 1) * K); scales.fill(0.05, col * K / 32, (col + 1) * K / 32); }
      if (fixture === "cancellation") { weights.fill(offset ? -3 : 1); scales.fill(offset ? 0.0021 : 0.0037); }
      const result = { values: make(weights.byteLength, U.STORAGE | U.COPY_DST, weights), scales: make(scales.byteLength, U.STORAGE | U.COPY_DST, scales) };
      await drain(device, deadline); return result;
    };
    const gate = await weightsFor(0), up = pipes.kind === "gelu" ? await weightsFor(19) : null;
    const xValues = Float32Array.from({ length: T * K }, (_, i) => {
      const row = Math.floor(i / K), col = i % K;
      if (fixture === "zeros") return row ? -0 : 0;
      if (fixture === "saturation") return 0.32 + row * 0.04;
      if (fixture === "cancellation") return Math.fround((col % 8 < 4 ? 1024 : -1024) + (col % 4 + 1) * 0.03125 + row * 0.015625);
      return Math.fround(0.04 * Math.sin(col * 0.127 + row * 0.71 + 0.3)) || 0.00390625;
    });
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 0, 0, 0])), params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, mode, bias]));
    const x = make(xBytes, U.STORAGE | U.COPY_DST, xValues);
    const biasValues = Float32Array.from({ length: biasBytes / 4 }, (_, i) => !bias ? NaN : fixture === "zeros" ? (i & 1 ? -0 : 0) : Math.fround(0.003 * Math.sin(i + 0.1)));
    const b = make(biasBytes, U.STORAGE | U.COPY_DST, biasValues);
    const initial = new Float32Array(count + 4).fill(NaN); initial.fill(SENTINEL, count);
    if (pipes.kind === "plain" && mode === 1) for (let i = 0; i < count; i++) initial[i] = fixture === "zeros" ? (i & 1 ? -0 : 0) : Math.fround(0.007 * Math.cos(i * 0.41));
    const ys = [0, 1].map(() => make(yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, initial));
    const partValues = new Float32Array(8).fill(NaN); partValues.fill(SENTINEL, 4);
    const part = make(32, U.STORAGE | U.COPY_SRC | U.COPY_DST, partValues);
    const groups = ys.map((y) => device.createBindGroup({ layout: pipes.layout, entries: (pipes.kind === "plain"
      ? [globals, params, x, gate.values, gate.scales, b, y, part]
      : [globals, params, x, gate.values, gate.scales, up.values, up.scales, y]).map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    const dispatch = (pass, candidate) => { pass.setPipeline(candidate ? pipes.candidate : pipes.baseline); pass.setBindGroup(0, groups[candidate ? 1 : 0]); pass.dispatchWorkgroups(Math.ceil(N / 4), 1); };
    read = make(readBytes, U.MAP_READ | U.COPY_DST);
    const copy = (encoder) => { for (let i = 0; i < 2; i++) encoder.copyBufferToBuffer(ys[i], 0, read, i * yBytes, yBytes); encoder.copyBufferToBuffer(part, 0, read, 2 * yBytes, 32); };
    const values = async () => {
      const within = await mapBeforeDeadline(read, readBytes, deadline); mapped = true;
      if (!within) throw new Error("two-row subgroup parity deadline exceeded");
      const result = new Float32Array(read.getMappedRange(0, readBytes).slice(0)); read.unmap(); mapped = false; return result;
    };
    const checkValues = (data) => {
      const stride = yBytes / 4, outputCheck = exact(data.subarray(0, count), data.subarray(stride, stride + count));
      const tailGuard = [count, stride + count].every((at) => data.subarray(at, at + 4).every((value) => value === SENTINEL));
      const unusedPartUntouched = data.subarray(stride * 2, stride * 2 + 4).every(Number.isNaN) && data.subarray(stride * 2 + 4, stride * 2 + 8).every((value) => value === SENTINEL);
      let fixtureCheck = true;
      if (pipes.kind === "gelu" && fixture === "saturation") for (let row = 0; row < T; row++) fixtureCheck &&= data[row * N] > 10 && data[row * N + 1] === 0;
      return { ok: outputCheck.ok && tailGuard && unusedPartUntouched && fixtureCheck, outputCheck, tailGuard, unusedPartUntouched, fixtureCheck };
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); copy(encoder); device.queue.submit([encoder.finish()]);
    const numericalCheck = checkValues(await values());
    let timing = null, repeatedCheck = null;
    if (numericalCheck.ok && fixture === "normal" && N >= 256) {
      query = device.createQuerySet({ type: "timestamp", count: 12 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC), encoder = device.createCommandEncoder(), warmup = encoder.beginComputePass(); dispatch(warmup, false); dispatch(warmup, true); warmup.end();
      const indices = { baseline: [], candidate: [] }; let index = 0;
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
        indices[candidate ? "candidate" : "baseline"].push(index);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let rep = 0; rep < 4; rep++) dispatch(pass, candidate); pass.end(); index += 2;
      }
      encoder.resolveQuerySet(query, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 96, deadline); mapped = true;
      if (!within) throw new Error("two-row subgroup timing deadline exceeded");
      const stamps = new BigUint64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
      const samples = Object.fromEntries(Object.entries(indices).map(([key, positions]) => [key, positions.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4)]));
      const selection = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
      timing = { ...samples, pairs: 3, repetitions: 4, baselineMs: selection.genericMs, candidateMs: selection.wideMs, benefit: selection.benefit, stable: selection.stable, selected: selection.kernel === "wide", reason: selection.reason };
      const final = device.createCommandEncoder(); copy(final); device.queue.submit([final.finish()]); repeatedCheck = checkValues(await values());
    }
    return { ...c, T, kind: pipes.kind, strategy: pipes.strategy, registered: pipes.registered, mode, bias, fixture, ok: numericalCheck.ok && (!repeatedCheck || repeatedCheck.ok), numericalCheck, repeatedCheck, timing,
      scratchBytes, scratchLimitBytes: LIMIT, hostWeightUploadBytes: sharedWeightBytes, uploadStagingBytes, immutableGpuWeightBytes: sharedWeightBytes, barrierExecutions: { baseline: 7, candidate: 1 }, shaderHashes: pipes.shaderHashes };
  } finally { if (mapped) read?.unmap(); query?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now(), kind = params.get("kind");
  if (kind && !["plain", "gelu"].includes(kind)) throw new Error("kind must be plain or gelu");
  const strategy = params.get("strategy") || "all";
  if (!["all", "active", "vector"].includes(strategy)) throw new Error("strategy must be all, active, or vector");
  const registered = params.get("registered") === "1";
  if (registered && strategy !== "vector") throw new Error("registered=1 requires strategy=vector");
  const actual = [{ shape: "gate", N: 6144, K: 1536 }, { shape: "gate-wide", N: 12288, K: 1536 }, { shape: "down", N: 1536, K: 6144 }, { shape: "q", N: 2048, K: 1536 }, { shape: "o", N: 1536, K: 2048 }];
  const inputs = [];
  for (const next of kind ? [kind] : ["plain", "gelu"]) {
    inputs.push(...actual.filter(({ shape }) => next === "plain" || shape.startsWith("gate")).map((shape) => ({ ...shape, kind: next })));
    if (!params.has("shape") && params.get("quick") !== "1") {
      if (next === "plain") {
        for (const K of [32, 1056]) for (const mode of [0, 1, 2]) for (const bias of [0, 1]) inputs.push({ kind: next, shape: "ragged", N: 17, K, mode, bias });
        for (const mode of [0, 1, 2]) for (const bias of [0, 1]) inputs.push({ kind: next, shape: "zeros", N: 17, K: 32, mode, bias, fixture: "zeros" });
        inputs.push({ kind: next, shape: "cancellation", N: 17, K: 1056, fixture: "cancellation" });
      } else for (const K of [32, 1056]) for (const fixture of ["normal", "zeros", "saturation", "cancellation"]) inputs.push({ kind: next, shape: "ragged", N: 17, K, fixture });
    }
  }
  const selected = inputs.filter((c) => !params.has("shape") || c.shape === params.get("shape"));
  if (!selected.length) throw new Error("unknown or unsupported shape filter");
  const runtime = await requestDevice({ features: ["timestamp-query", "subgroups"] });
  try {
    metadata = { precision: "f32", strategy, registered, subgroup32: runtime.subgroup32, adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features], subgroupMinSize: runtime.adapter.info?.subgroupMinSize, subgroupMaxSize: runtime.adapter.info?.subgroupMaxSize } };
    if (!runtime.subgroup32 || !runtime.device.features.has("timestamp-query") || runtime.device.features.has("shader-f16")) throw new Error("two-row subgroup guard requires fixed32 subgroups and timestamp-query without shader-f16");
    const source = await kernelSource(), pipes = new Map(); metadata.wasmUrl = source.wasmUrl;
    for (const input of selected) {
      if (!pipes.has(input.kind)) {
        try { pipes.set(input.kind, await withGpuErrors(runtime, () => compile(runtime.device, source, input.kind, strategy, registered), "compiling two-row subgroup guard")); }
        catch (error) {
          if (strategy !== "all" && error.code === "WEBGPU_INIT") error.variantSkipped = `${strategy}-strategy-compilation-rejected`;
          throw error;
        }
      }
      log({ status: "checking", ...input });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes.get(input.kind), input), "two-row subgroup exact guard"); cases.push(result); log(result);
      if (!result.ok) throw new Error("two-row subgroup bit-exact parity failed");
      if (result.timing && result.timing.baselineMs === null) throw new Error("two-row subgroup timestamps were invalid");
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMs), 0) / timed.length) : performance.now() - started,
      method: "F32 registered two-row matvec versus identical lane64 dot with exact fixed32 subgroup reduction; GPU timestamps, 3 paired samples x4 reps",
      elapsedMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().catch((error) => { window.gpuBench = { status: error.variantSkipped ? "skipped" : "error", done: true, backend: "webgpu", ...metadata,
  ...(error.variantSkipped ? { skipped: true, skipReason: error.variantSkipped } : {}), correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || String(error)); });
