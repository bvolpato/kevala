import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "../js/src/gpu-gemma4-batch-mlp.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const T = typeof location === 'undefined' ? 97 : Number(new URLSearchParams(location.search).get('T') || 97);
const CHUNK = 1024 * 1024, LIMIT = 64 * CHUNK;
const SENTINEL = new Uint32Array(new Float32Array([12345.25]).buffer)[0];
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const body = (code, name) => code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0];
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

export function gridMatmulCode(original) {
  let code = original;
  const replace = (from, to) => {
    if (code.split(from).length !== 2) throw new Error('grid source marker mismatch: ' + from);
    code = code.replace(from, to);
  };
  if (code.includes('enable f16;') || !code.includes('@workgroup_size(32, 8)') || !code.includes('let m0 = wg.y * 56u;') || (code.match(/var acc\d = vec2<f32>\(0\.0\);/g) || []).length !== 7) throw new Error('grid requires registered Q8/F32 R4 BM56 batch source');
  const bindings = '@group(0) @binding(3) var<storage, read> W: array<vec4<u32>>;\n@group(0) @binding(4) var<storage, read> S: array<f32>;\n@group(0) @binding(5) var<storage, read> B: array<f32>;\n@group(0) @binding(6) var<storage, read_write> Y: array<f32>;\n@group(0) @binding(7) var<storage, read_write> PART: array<f32>;';
  replace(bindings, '@group(0) @binding(3) var<storage, read> WG: array<vec4<u32>>;\n@group(0) @binding(4) var<storage, read> SG: array<f32>;\n@group(0) @binding(5) var<storage, read> WU: array<vec4<u32>>;\n@group(0) @binding(6) var<storage, read> SU: array<f32>;\n@group(0) @binding(7) var<storage, read_write> Y: array<f32>;\n@group(0) @binding(8) var<storage, read_write> PART: array<f32>;');
  const store = body(code, 'mm_store');
  if (!store?.includes('PART[z * T * N + o] = value;')) throw new Error('grid original PART store changed');
  replace(store, 'fn mm_store(value: f32, row: u32, col: u32, bias: f32, T: u32, N: u32, splits: u32, z: u32) {\n  if (row >= T) { return; }\n  let o = row * N + col;\n  PART[z * T * N + o] = value;\n}');
  replace('  let kb0 = wg.z * per;', '  let kb0 = (wg.z % splits) * per;');
  const loads = '        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];\n        let s = S[wrow * nb + kb];';
  replace(loads, '        let at = (wrow * K + kb * 32u) / 16u + lq / 2u;\n        let scaleAt = wrow * nb + kb;\n        var v = vec4<u32>(0u);\n        var s = 0.0;\n        if (wg.z < splits) { v = WG[at]; s = SG[scaleAt]; }\n        else { v = WU[at]; s = SU[scaleAt]; }');
  if ((code.match(/      if \(p.bias == 1u\) \{ bias = B\[col\]; \}/g) || []).length !== 2) throw new Error('grid original bias sites changed');
  code = code.replaceAll('      if (p.bias == 1u) { bias = B[col]; }\n', '');
  for (const name of ['sx', 'mm_splits']) if (body(code, name) !== body(original, name)) throw new Error('grid changed ' + name);
  const orderedDots = (text) => [...text.matchAll(/acc\d \+= vec2<f32>\(dot\(a\d, b0\), dot\(a\d, b1\)\);/g)].map(([line]) => line);
  if (JSON.stringify(orderedDots(code)) !== JSON.stringify(orderedDots(original)) || orderedDots(code).length !== 56 || (code.match(/workgroupBarrier\(\)/g) || []).length !== 2) throw new Error('grid changed original ordered dot arithmetic or barriers');
  return code;
}

export function gridProof(N = 6144, K = 1536) {
  const splits = mmSplits(1, N, K, 256, 64, 64), nb = K / 32, per = Math.ceil(nb / splits);
  const mapping = Array.from({ length: 2 * splits }, (_, z) => ({ z, projection: Math.floor(z / splits), split: z % splits, kb0: z % splits * per, kb1: Math.min(nb, (z % splits + 1) * per), partialPlane: z }));
  for (const projection of [0, 1]) {
    const order = mapping.filter((item) => item.projection === projection).flatMap(({ kb0, kb1 }) => Array.from({ length: Math.max(0, kb1 - kb0) }, (_, i) => kb0 + i));
    if (order.length !== nb || order.some((kb, i) => kb !== i)) throw new Error('grid changed original split partition coverage');
  }
  return { splits, mapping, physicalBM: 56, physicalBN: 64, virtualBM: 64, virtualBN: 64, accumulators: '7 vec2', sharedBytes: 16384, orderedDotsPerK32: 8, perWorkgroupOneProjection: true, noAdditionalInputOrWeightCopies: true };
}

export function memoryPlan(N, K) {
  const splits = mmSplits(1, N, K, 256, 64, 64), active = T * N, partial = splits * active;
  const xBytes = 128 * K * 4, yBytes = (128 * N + 4) * 4, partBytes = (partial + 4) * 4, gridPartBytes = (2 * partial + 4) * 4;
  const gpuTemporaryBytes = xBytes + 4 * yBytes + 2 * partBytes + gridPartBytes + 16 + 10 * 16 + 16 + 16 + 256 + 256;
  return { splits, active, partial, xBytes, yBytes, partBytes, gridPartBytes, gpuTemporaryBytes, controlledTemporaryBytes: gpuTemporaryBytes + 2 * CHUNK + 1024, limitBytes: LIMIT, immutableWeightBytes: 2 * N * K * 9 / 8, hostStageBytes: CHUNK, uploadAllowanceBytes: CHUNK };
}

const OFFSET_COMPARE = 'struct P { length: u32, offset: u32, _a: u32, _b: u32 }\n@group(0) @binding(0) var<uniform> p: P;\n@group(0) @binding(1) var<storage, read> A: array<u32>;\n@group(0) @binding(2) var<storage, read> B: array<u32>;\n@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x + gid.y * 65535u * 256u;\n  if (i >= p.length) { return; }\n  if (A[i] != B[p.offset + i]) { atomicAdd(&status[0], 1u); }\n}\n';

async function compile(device, source, registered = false) {
  const spec = { weights: 'q8', f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
  const codes = { baseline: source('matmul_gemma_batch', spec), reduce: source('reduce_gemma_batch', spec), gelu: source('gemma4_gelu'), reduceGelu: source('reduce_gelu', spec) };
  const derived = gridMatmulCode(codes.baseline);
  codes.candidate = registered ? source('gemma4_mlp_grid', spec) : derived;
  if (registered && codes.candidate.trim() !== derived.trim()) throw new Error('registered grid MLP differs from the exact probe');
  if (body(codes.reduceGelu, 'gelu') !== body(codes.gelu, 'gelu')) throw new Error('grid GELU reducer differs from registered GELU');
  const layout = matmulLayout(device), reduction = reduceLayout(device);
  const pairLayout = device.createBindGroupLayout({ entries: ['uniform', 'uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'storage'].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pl = (value) => device.createPipelineLayout({ bindGroupLayouts: [value] });
  return { layout, reduction, pairLayout,
    baseline: await pipeline(device, codes.baseline, 'mlp-grid.current-bm56', pl(layout)), reduce: await pipeline(device, codes.reduce, 'mlp-grid.current-reduce', pl(reduction)), gelu: await pipeline(device, codes.gelu, 'mlp-grid.current-gelu'),
    candidate: await pipeline(device, codes.candidate, 'mlp-grid.stacked-projections', pl(pairLayout)), reduceGelu: await pipeline(device, codes.reduceGelu, 'mlp-grid.registered-reduce-gelu'), offsetCompare: await pipeline(device, OFFSET_COMPARE, 'mlp-grid.partial-plane-compare'), validator: await createGemma4BatchMlpValidator(device),
    hashes: Object.fromEntries(await Promise.all([...Object.entries(codes), ['offsetCompare', OFFSET_COMPARE]].map(async ([name, code]) => [name, await hash(code)]))) };
}

async function check(runtime, pipes, input) {
  const { device } = runtime, U = GPUBufferUsage, { N, K, pattern = "normal" } = input, plan = memoryPlan(N, K);
  if (plan.splits === 1) throw new Error("grid probe is limited to split projections; unsplit retains the original path");
  if (plan.controlledTemporaryBytes > LIMIT) throw new Error("MLP grid controlled scratch exceeds 64 MiB");
  if (Math.max(N * K, plan.xBytes, plan.yBytes, plan.partBytes, plan.gridPartBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("MLP grid buffer exceeds device limits");
  const owned = [], deadline = performance.now() + 30000;
  let allocated = 0, read, queries, mapped = false;
  const make = (size, usage, label) => {
    allocated += size; if (allocated > plan.gpuTemporaryBytes + plan.immutableWeightBytes) throw new Error("MLP grid allocation exceeds declared memory");
    const buffer = device.createBuffer({ size, usage, label: `mlp-grid.${label}` }); owned.push(buffer); return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });
  const work = (pass, length) => { const count = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(count, 65535), Math.ceil(count / 65535)); };
  const drain = async () => {
    const remaining = deadline - performance.now(); if (remaining <= 0) throw new Error("MLP grid upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("MLP grid upload deadline exceeded")), remaining); })]); }
    finally { clearTimeout(timer); }
  };
  try {
    const stage = new Float32Array(CHUNK / 4), bytes = new Int8Array(stage.buffer);
    const generated = async (buffer, size, width, valueAt) => {
      for (let offset = 0; offset < size; offset += CHUNK) {
        const count = Math.min(CHUNK, size - offset), target = width === 1 ? bytes : stage;
        for (let i = 0; i < count / width; i++) target[i] = valueAt(offset / width + i);
        device.queue.writeBuffer(buffer, offset, stage.buffer, 0, count); await drain();
      }
    };
    const weightsFor = async (up) => {
      const w = make(N * K, U.STORAGE | U.COPY_DST, "immutable-W"), s = make(N * K / 8, U.STORAGE | U.COPY_DST, "immutable-S");
      await generated(w, N * K, 1, (i) => {
        const col = Math.floor(i / K), k = i % K;
        if (pattern === "saturation" && col < 2) return up ? 17 : col ? -128 : 127;
        if (pattern === "cancellation") return 1 + (col * 13 + (k & ~7) * 7 + (up ? 37 : 19)) % 127;
        return (i * 31 + (i >>> 5) * 17 + (up ? 37 : 19)) % 256 - 128;
      });
      await generated(s, N * K / 8, 4, (i) => pattern === "saturation" && i < 2 * K / 32 ? 0.05 : i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
      return { w, s };
    };
    const gateW = await weightsFor(false), upW = await weightsFor(true), x = make(plan.xBytes, U.STORAGE | U.COPY_DST, "X");
    await generated(x, plan.xBytes, 4, (i) => {
      const row = Math.floor(i / K), k = i % K;
      if (row >= T) return NaN;
      if (pattern === "zeros") return i % 2 ? -0 : 0;
      if (pattern === "saturation") return 0.32 * (1 + row / 32);
      if (pattern === "cancellation") return (k % 8 < 4 ? 64 : -64) + (k % 4 + 1) * 0.0625 + (row % 3) * 0.03125;
      return Math.fround(0.04 * Math.sin(k * 0.127 + row * 0.13 + 0.3));
    });
    const uniform = (values) => { const b = make(16, U.UNIFORM | U.COPY_DST, "params"); device.queue.writeBuffer(b, 0, new Uint32Array(values)); return b; };
    const globals = uniform([T, 1, T, 0]), params = uniform([N, K, 0, 0]), pairParams = uniform([N, K, 1, 0]), geluParams = uniform([N, 0, 0, 0]);
    const zero = make(16, U.STORAGE | U.COPY_DST, "disabled-bias"); device.queue.writeBuffer(zero, 0, new Float32Array(4).fill(NaN));
    const outputs = Array.from({ length: 4 }, () => make(plan.yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y"));
    const parts = [0, 1].map(() => make(plan.partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-PART"));
    const gridPart = make(plan.gridPartBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "grid-PART");
    const unusedPart = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "unused-candidate-output"), status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "status");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps"); read = make(256, U.MAP_READ | U.COPY_DST, "readback");
    const yp = uniform([plan.yBytes / 4, plan.active, SENTINEL, 0x7fc00000]), pp = uniform([plan.partBytes / 4, plan.partial, SENTINEL, 0x7fc00000]), unused = uniform([4, 0, SENTINEL, 0x7fc00000]);
    const gp = uniform([plan.gridPartBytes / 4, 2 * plan.partial, SENTINEL, 0x7fc00000]);
    const offsets = [0, 1].map((projection) => uniform([plan.partial, projection * plan.partial, 0, 0]));
    const fills = [...outputs.map((buffer) => ({ length: plan.yBytes / 4, group: group(pipes.validator.fillLayout, [yp, buffer]) })), ...parts.map((buffer) => ({ length: plan.partBytes / 4, group: group(pipes.validator.fillLayout, [pp, buffer]) })), { length: plan.gridPartBytes / 4, group: group(pipes.validator.fillLayout, [gp, gridPart]) }, { length: 4, group: group(pipes.validator.fillLayout, [unused, unusedPart]) }];
    const compare = (baseline, candidate, params, length) => ({ length, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params, baseline, candidate, status }) });
    const checks = [compare(outputs[2], outputs[3], yp, plan.yBytes / 4), ...outputs.slice(0, 2).map((y) => compare(y, y, yp, plan.yBytes / 4)), ...parts.map((p) => compare(p, p, pp, plan.partBytes / 4)), compare(gridPart, gridPart, gp, plan.gridPartBytes / 4), compare(unusedPart, unusedPart, unused, 4), ...parts.map((part, projection) => ({ length: plan.partial, pipeline: pipes.offsetCompare, compare: group(pipes.offsetCompare.getBindGroupLayout(0), [offsets[projection], part, gridPart, status]) }))];
    const baseline = [gateW, upW].map((weight, i) => ({ group: group(pipes.layout, [globals, params, x, weight.w, weight.s, zero, outputs[i], parts[i]]), reduce: group(pipes.reduction, [globals, params, parts[i], zero, outputs[i]]) }));
    const geluGroup = group(pipes.gelu.getBindGroupLayout(0), [globals, geluParams, outputs[0], outputs[1], outputs[2]]);
    const candidateGroup = group(pipes.pairLayout, [globals, params, x, gateW.w, gateW.s, upW.w, upW.s, unusedPart, gridPart]);
    const candidateReduce = group(pipes.reduceGelu.getBindGroupLayout(0), [globals, pairParams, gridPart, outputs[3]]);
    const planeBytes = plan.partial * 4;
    const control = planeBytes % device.limits.minStorageBufferOffsetAlignment === 0 ? [gateW, upW].map((weight, projection) => group(pipes.layout, [globals, params, x, weight.w, weight.s, zero, unusedPart, { buffer: gridPart, offset: projection * planeBytes, size: planeBytes }])) : null;
    const dispatch = (pass, candidate) => {
      if (candidate === "control") {
        if (!control) throw new Error("fused reducer control requires aligned partial planes");
        pass.setPipeline(pipes.baseline);
        for (const group of control) { pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), plan.splits); }
        pass.setPipeline(pipes.reduceGelu); pass.setBindGroup(0, candidateReduce); work(pass, plan.active);
        return;
      }
      if (candidate) {
        pass.setPipeline(pipes.candidate); pass.setBindGroup(0, candidateGroup); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), 2 * plan.splits);
        pass.setPipeline(pipes.reduceGelu); pass.setBindGroup(0, candidateReduce); work(pass, plan.active);
      }
      else {
        for (const op of baseline) {
          pass.setPipeline(pipes.baseline); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), plan.splits);
          if (plan.splits > 1) { pass.setPipeline(pipes.reduce); pass.setBindGroup(0, op.reduce); work(pass, plan.active); }
        }
        pass.setPipeline(pipes.gelu); pass.setBindGroup(0, geluGroup); work(pass, plan.active);
      }
    };
    const verify = async (candidate = true) => {
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(pipes.validator.fill);
      for (const item of fills) { fill.setBindGroup(0, item.group); work(fill, item.length); } fill.end();
      const compute = encoder.beginComputePass(); dispatch(compute, false); dispatch(compute, candidate); compute.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(pipes.validator.compare);
      for (const item of checks) { compare.setPipeline(item.pipeline || pipes.validator.compare); compare.setBindGroup(0, item.compare); work(compare, item.length); } compare.end();
      encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      try {
        if (!within) throw new Error("MLP grid validation deadline exceeded");
        const values = new Uint32Array(read.getMappedRange(0, 16));
        const result = { ok: values.every((value) => value === 0), bitMismatches: values[0], nonFinite: values[1], tailMismatches: values[2], outputValues: plan.active, signedZeroChecked: true, paddingAndTailsChecked: true, partialValues: 2 * plan.partial, bothProjectionPartialPlanesChecked: true, unusedCandidateOutputUntouched: true };
        if (!result.ok) throw new Error(`MLP grid raw-bit guard failed: ${JSON.stringify(result)}`);
        return result;
      } finally { read.unmap(); mapped = false; }
    };
    const before = await verify(), controlBefore = control ? await verify("control") : null;
    const queryCount = control ? 18 : 12;
    queries = device.createQuerySet({ type: "timestamp", count: queryCount });
    const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); if (control) dispatch(warm, "control"); warm.end();
    const indices = [[], [], []]; let index = 0;
    for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? control ? [true, "control", false] : [true, false] : control ? [false, "control", true] : [false, true]) {
      indices[candidate === "control" ? 2 : Number(candidate)].push(index);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
      for (let rep = 0; rep < 4; rep++) dispatch(pass, candidate); pass.end(); index += 2;
    }
    encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, queryCount * 8, deadline); mapped = true;
    if (!within) throw new Error("MLP grid timestamp deadline exceeded");
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0)); read.unmap(); mapped = false;
    const [baselineTimes, candidateTimes, controlTimes] = indices.map((items) => items.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4));
    if ([...baselineTimes, ...candidateTimes, ...controlTimes].some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("MLP grid invalid timestamps");
    const after = await verify(), controlAfter = control ? await verify("control") : null, choice = selectMatmulKernel({ generic: baselineTimes, wide: candidateTimes });
    return { ...input, T, ok: true, f16: false, proof: gridProof(N, K), before, after, controlBefore, controlAfter, fusedReducerControl: control ? "two registered GEMMs plus registered reduce_gelu" : "unavailable: unaligned partial plane", memory: plan, dispatches: { baseline: plan.splits > 1 ? 5 : 3, candidate: 2, fusedReducerControl: control ? 3 : null }, timing: { fusedReducerControl: controlTimes, gridVersusFusedReducer: control ? selectMatmulKernel({ generic: controlTimes, wide: candidateTimes }) : null, baseline: baselineTimes, candidate: candidateTimes, ...choice, candidateMs: choice.wideMs, pairs: 3, repetitions: 4, allocationsAndUploadsTimed: false } };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  if (!Number.isSafeInteger(T) || T < 97 || T > 112) throw new Error("MLP grid probe supports T97 to T112 only");
  const all = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "ragged", N: 65, K: 1056 }, { shape: "zeros", N: 65, K: 1056, pattern: "zeros" }, { shape: "cancellation", N: 65, K: 1056, pattern: "cancellation" }, { shape: "saturation", N: 17, K: 1056, pattern: "saturation" }];
  const inputs = all.filter(({ shape }) => params.has("shape") ? shape === params.get("shape") : params.get("guards") === "1" ? true : shape === "gate6144");
  if (!inputs.length) throw new Error("unknown MLP grid shape");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    metadata = { f16: false, adapter: { name: runtime.name, features: [...runtime.device.features] }, temporaryLimitBytes: LIMIT, immutableWeightsExcludedFromTemporaryLimit: true };
    if (!runtime.device.features.has("timestamp-query")) throw new Error("MLP grid probe requires timestamp-query");
    const source = await kernelSource(params.get("wasm") || undefined); metadata.wasmUrl = source.wasmUrl;
    const registered = params.get("registered") === "1";
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source, registered), "compiling MLP grid"); metadata.shaderHashes = pipes.hashes; metadata.registered = registered;
    for (const input of inputs) { log({ status: "checking", ...input }); const result = await withGpuErrors(runtime, () => check(runtime, pipes, input), "checking DEV MLP grid"); cases.push(result); log(result); }
    const metricMs = Math.exp(cases.reduce((sum, item) => sum + Math.log(item.timing.candidateMs), 0) / cases.length);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-mlp-grid", ...metadata, metricMs, elapsedMs: performance.now() - started, method: "candidate GPU timestamp milliseconds; one grid-stacked BM56 GEMM and registered fused reducer/GELU versus separate gate/up, reducers and GELU", correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-mlp-grid" };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
