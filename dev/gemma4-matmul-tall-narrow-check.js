import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "../js/src/gpu-gemma4-batch-mlp.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const T = 97, CHUNK = 1024 * 1024, LIMIT = 64 * CHUNK;
const SENTINEL = new Uint32Array(new Float32Array([12345.25]).buffer)[0];
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");
const body = (code, name) => code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0];

export function tallNarrowCode(original) {
  let code = original;
  const replace = (from, to) => {
    if (code.split(from).length !== 2) throw new Error('tall narrow source marker mismatch: ' + from);
    code = code.replace(from, to);
  };
  if (code.includes('enable f16;') || !code.includes('@workgroup_size(32, 8)') || (code.match(/var acc\d+ = vec2<f32>\(0\.0\);/g) || []).length !== 7) throw new Error('tall narrow requires registered Q8/F32 R4 BM56');
  replace('var<workgroup> xs: array<vec4<f32>, 512>;', 'var<workgroup> xs: array<vec4<f32>, 896>;');
  replace('var<workgroup> ws: array<vec4<f32>, 512>;', 'var<workgroup> ws: array<vec4<f32>, 256>;');
  replace('  let m0 = wg.y * 56u;', '  let m0 = wg.y * 112u;');
  replace('  let n0 = wg.x * 64u;', '  let n0 = wg.x * 32u;');
  const declarations = code.match(/  var acc0 = vec2<f32>\(0\.0\);[\s\S]*?  var acc6 = vec2<f32>\(0\.0\);/)?.[0];
  if (!declarations) throw new Error('tall narrow accumulator declarations missing');
  replace(declarations, Array.from({ length: 14 }, (_, i) => '  var acc' + i + ' = 0.0;').join('\n'));
  replace('  let xrow = m0 + lr;\n', '');
  replace('  var xv0 = vec4<f32>(0.0);\n  var xv1 = vec4<f32>(0.0);\n', '');
  const oldLoad = '    xv0 = vec4<f32>(0.0);\n    xv1 = vec4<f32>(0.0);\n    if (lr < 56u && xrow < T) {\n      let base = (xrow * K + kb * 32u) / 4u + lq * 2u;\n      xv0 = X[base];\n      xv1 = X[base + 1u];\n    }';
  replace(oldLoad, '    for (var xr = lr; xr < 112u; xr += 64u) {\n      var xv0 = vec4<f32>(0.0);\n      var xv1 = vec4<f32>(0.0);\n      let xrow = m0 + xr;\n      if (xrow < T) {\n        let base = (xrow * K + kb * 32u) / 4u + lq * 2u;\n        xv0 = X[base];\n        xv1 = X[base + 1u];\n      }\n      xs[xr * 8u + lq * 2u] = vec4<f32>(xv0);\n      xs[xr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);\n    }');
  replace('    if (lr < 56u) {\n      xs[lr * 8u + lq * 2u] = vec4<f32>(xv0);\n      xs[lr * 8u + lq * 2u + 1u] = vec4<f32>(xv1);\n    }\n', '');
  replace('      if (wrow < N) {', '      if (lr < 32u && wrow < N) {');
  const stores = '    for (var gg = 0u; gg < 1u; gg++) {\n      let r = lr + 64u * gg;\n      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(wv[2u * gg]);\n      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(wv[2u * gg + 1u]);\n    }';
  replace(stores, '    if (lr < 32u) {\n' + stores.split('\n').map((line) => '  ' + line).join('\n') + '\n    }');
  let blocks = 0;
  code = code.replace(/    \{\n      let kq = (\d)u;[\s\S]*?      acc6 \+= vec2<f32>\(dot\(a6, b0\), dot\(a6, b1\)\);\n    \}/g, (block, kq) => {
    if (Number(kq) !== blocks++ || (block.match(/acc\d \+= vec2<f32>\(dot\(a\d, b0\), dot\(a\d, b1\)\);/g) || []).length !== 7) throw new Error('tall narrow K dot order changed');
    return '    {\n      let kq = ' + kq + 'u;\n      let b0 = vec4<f32>(ws[c * 8u + (kq ^ nx)]);\n' + Array.from({ length: 14 }, (_, i) => '      let a' + i + ' = vec4<f32>(xs[(lid.y + ' + i * 8 + 'u) * 8u + kq]);\n      acc' + i + ' += dot(a' + i + ', b0);').join('\n') + '\n    }';
  });
  if (blocks !== 8) throw new Error('tall narrow expected eight original ascending K/4 blocks');
  const output = code.slice(code.indexOf('  let row = m0 + lid.y;'));
  if ((output.match(/mm_store\(acc\d\.[xy],/g) || []).length !== 14) throw new Error('tall narrow original output stores changed');
  const replacement = '  let row = m0 + lid.y;\n  let col = n0 + lid.x;\n  if (col < N) {\n    var bias = 0.0;\n    if (p.bias == 1u) { bias = B[col]; }\n' + Array.from({ length: 14 }, (_, i) => '    mm_store(acc' + i + ', row + ' + i * 8 + 'u, col, bias, T, N, splits, wg.z);').join('\n') + '\n  }\n}\n';
  replace(output, replacement);
  for (const name of ['sx', 'mm_store', 'mm_splits']) if (body(code, name) !== body(original, name)) throw new Error('tall narrow changed ' + name);
  if ((code.match(/var acc\d+ = 0\.0;/g) || []).length !== 14 || (code.match(/acc\d+ \+= dot\(/g) || []).length !== 112 || (code.match(/workgroupBarrier\(\)/g) || []).length !== 2) throw new Error('tall narrow accumulator or barrier geometry changed');
  code = code.replace('covers 56 rows and 64 columns, retaining a virtual\n// 64-row grid and split rule. Each thread owns seven rows and two columns.', 'covers 112 rows and 32 columns, retaining the original virtual\n// BM64/BN64 split rule. Each thread owns fourteen rows and one column.');
  code = code.replace('X rows 0..56, W rows lr + 64 g', 'X rows lr and lr+64 below 112, W rows lr below 32');
  code = code.replace('reusing two weight vectors across eight rows.', 'reusing one weight vector across fourteen rows.');
  return code;
}

export function tallNarrowProof() {
  const xs = new Set(), ws = new Set(), outputs = new Set();
  for (let li = 0; li < 256; li++) {
    const lr = Math.floor(li / 4), lq = li % 4;
    for (let xr = lr; xr < 112; xr += 64) for (let pair = 0; pair < 2; pair++) xs.add(xr * 8 + lq * 2 + pair);
    if (lr < 32) for (let pair = 0; pair < 2; pair++) ws.add(lr * 8 + ((lq * 2 + pair) ^ (lr & 7)));
    const x = li % 32, y = Math.floor(li / 32);
    for (let i = 0; i < 14; i++) outputs.add((y + 8 * i) * 32 + x);
  }
  if (xs.size !== 896 || Math.max(...xs) !== 895 || ws.size !== 256 || Math.max(...ws) !== 255 || outputs.size !== 112 * 32 || Math.max(...outputs) !== 112 * 32 - 1) throw new Error('tall narrow tile misses or aliases shared/input/output slots');
  return { physicalBM: 112, physicalBN: 32, virtualBM: 64, virtualBN: 64, threads: 256, scalarAccumulators: 14, sharedBytes: 18432, inputVec4Slots: xs.size, weightVec4Slots: ws.size, outputSlots: outputs.size, eachTileOutputCoveredOnce: true, kqOrder: [0, 1, 2, 3, 4, 5, 6, 7] };
}

export function memoryPlan(N, K) {
  const splits = mmSplits(1, N, K, 256, 64, 64);
  const active = T * N, partial = splits > 1 ? splits * active : 0;
  const xBytes = 128 * K * 4, yBytes = (128 * N + 4) * 4, partBytes = (partial + 4) * 4, biasBytes = N * 4;
  const gpuTemporaryBytes = xBytes + 2 * yBytes + 2 * partBytes + biasBytes + 2 * 16 + 2 * 16 + 16 + 256 + 256;
  return { splits, active, partial, xBytes, yBytes, partBytes, biasBytes, gpuTemporaryBytes, controlledTemporaryBytes: gpuTemporaryBytes + 2 * CHUNK + 1024, hostStageBytes: CHUNK, uploadAllowanceBytes: CHUNK, limitBytes: LIMIT, immutableWeightBytes: N * K * 9 / 8 };
}

async function compile(device, source) {
  if (device.limits.maxComputeWorkgroupStorageSize < 18432) throw new Error('tall narrow requires at least 18432 bytes of workgroup storage');
  const spec = { weights: 'q8', f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
  const codes = { baseline: source('matmul_gemma_batch', spec), reduce: source('reduce_gemma_batch', spec) };
  codes.candidate = tallNarrowCode(codes.baseline);
  const layout = matmulLayout(device), reduction = reduceLayout(device), pl = (value) => device.createPipelineLayout({ bindGroupLayouts: [value] });
  return { layout, reduction,
    baseline: await pipeline(device, codes.baseline, 'tall-narrow.registered-bm56-bn64', pl(layout)),
    candidate: await pipeline(device, codes.candidate, 'tall-narrow.bm112-bn32', pl(layout)),
    reduce: await pipeline(device, codes.reduce, 'tall-narrow.unchanged-reduce', pl(reduction)),
    validator: await createGemma4BatchMlpValidator(device), hashes: Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)]))) };
}

async function check(runtime, pipes, input) {
  const { device } = runtime, U = GPUBufferUsage, { N, K, pattern = "normal" } = input, plan = memoryPlan(N, K);
  if (plan.controlledTemporaryBytes > LIMIT) throw new Error("tall narrow scratch exceeds 64 MiB");
  if (Math.max(N * K, plan.xBytes, plan.yBytes, plan.partBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("tall narrow buffers exceed device limits");
  const owned = [], deadline = performance.now() + 30000;
  let allocated = 0, read, queries, mapped = false;
  const make = (size, usage, label) => {
    allocated += size;
    if (allocated > plan.gpuTemporaryBytes + plan.immutableWeightBytes) throw new Error("tall narrow allocation exceeds memory plan");
    const buffer = device.createBuffer({ size, usage, label: `tall-narrow.${label}` }); owned.push(buffer); return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const work = (pass, length) => { const n = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535)); };
  const drain = async () => {
    const remaining = deadline - performance.now(); if (remaining <= 0) throw new Error("tall narrow upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("tall narrow upload deadline exceeded")), remaining); })]); }
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
    const weights = make(N * K, U.STORAGE | U.COPY_DST, "immutable-W"), scales = make(N * K / 8, U.STORAGE | U.COPY_DST, "immutable-S");
    await generated(weights, N * K, 1, (i) => pattern === "cancellation" ? 1 + (Math.floor(i / K) * 13 + ((i % K) & ~7) * 7) % 127 : (i * 31 + (i >>> 5) * 17 + 19) % 256 - 128);
    await generated(scales, N * K / 8, 4, (i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
    const x = make(plan.xBytes, U.STORAGE | U.COPY_DST, "X"), bias = make(plan.biasBytes, U.STORAGE | U.COPY_DST, "bias");
    await generated(x, plan.xBytes, 4, (i) => {
      const row = Math.floor(i / K), k = i % K;
      if (row >= T) return NaN;
      if (pattern === "zeros") return i % 2 ? -0 : 0;
      if (pattern === "cancellation") return (k % 8 < 4 ? 64 : -64) + (k % 4 + 1) * 0.0625 + (row % 3) * 0.03125;
      return Math.fround(0.04 * Math.sin(k * 0.127 + row * 0.13 + 0.3));
    });
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), params = make(16, U.UNIFORM | U.COPY_DST, "params");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, 1, T, 0]));
    const outputs = [0, 1].map(() => make(plan.yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y"));
    const parts = [0, 1].map(() => make(plan.partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "PART"));
    const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "status"), resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    read = make(256, U.MAP_READ | U.COPY_DST, "readback");
    const yp = make(16, U.UNIFORM | U.COPY_DST, "Y-validation"), pp = make(16, U.UNIFORM | U.COPY_DST, "PART-validation");
    device.queue.writeBuffer(pp, 0, new Uint32Array([plan.partBytes / 4, plan.partial, SENTINEL, 0x7fc00000]));
    const checks = [
      { length: plan.yBytes / 4, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params: yp, baseline: outputs[0], candidate: outputs[1], status }) },
      { length: plan.partBytes / 4, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params: pp, baseline: parts[0], candidate: parts[1], status }) },
    ];
    const ops = outputs.map((output, i) => ({ group: group(pipes.layout, [globals, params, x, weights, scales, bias, output, parts[i]]), reduce: group(pipes.reduction, [globals, params, parts[i], bias, output]) }));
    const dispatch = (pass, candidate) => {
      const op = ops[Number(candidate)];
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline); pass.setBindGroup(0, op.group);
      pass.dispatchWorkgroups(Math.ceil(N / (candidate ? 32 : 64)), candidate ? 1 : 2, plan.splits);
      if (plan.splits > 1) { pass.setPipeline(pipes.reduce); pass.setBindGroup(0, op.reduce); work(pass, plan.active); }
    };
    const setMode = async (mode, hasBias) => {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias]));
      device.queue.writeBuffer(yp, 0, new Uint32Array([plan.yBytes / 4, plan.active, SENTINEL, mode === 1 ? 0x3d000000 : 0x7fc00000]));
      await generated(bias, plan.biasBytes, 4, (i) => hasBias ? i % 257 === 0 ? -0 : 0.003 * Math.sin(i * 0.17 + 0.3) : NaN);
    };
    const verify = async (mode, hasBias) => {
      await setMode(mode, hasBias);
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(pipes.validator.fill);
      for (const item of checks) for (const binding of item.fill) { fill.setBindGroup(0, binding); work(fill, item.length); } fill.end();
      const compute = encoder.beginComputePass(); dispatch(compute, false); dispatch(compute, true); compute.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(pipes.validator.compare);
      for (const item of checks) { compare.setBindGroup(0, item.compare); work(compare, item.length); } compare.end();
      encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      try {
        if (!within) throw new Error("tall narrow guard deadline exceeded");
        const counts = new Uint32Array(read.getMappedRange(0, 16));
        const result = { mode, hasBias, ok: counts.every((count) => count === 0), bitMismatches: counts[0], nonFinite: counts[1], tailMismatches: counts[2] };
        if (!result.ok) throw new Error(`tall narrow raw-bit guard failed: ${JSON.stringify(result)}`);
        return result;
      } finally { read.unmap(); mapped = false; }
    };
    const guardAllModes = async () => {
      const results = [];
      for (const mode of [0, 1, 2]) for (const bias of [0, 1]) results.push(await verify(mode, bias));
      return results;
    };
    const before = await guardAllModes(); await setMode(0, 0);
    queries = device.createQuerySet({ type: "timestamp", count: 12 });
    const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(); dispatch(warm, false); dispatch(warm, true); warm.end();
    const indices = [[], []]; let index = 0;
    for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
      indices[Number(candidate)].push(index);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
      for (let rep = 0; rep < 4; rep++) dispatch(pass, candidate); pass.end(); index += 2;
    }
    encoder.resolveQuerySet(queries, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, 96, deadline); mapped = true;
    if (!within) throw new Error("tall narrow timestamps exceeded deadline");
    const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
    const [baseline, candidate] = indices.map((items) => items.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4));
    if ([...baseline, ...candidate].some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("invalid tall narrow timestamps");
    const after = await guardAllModes(), choice = selectMatmulKernel({ generic: baseline, wide: candidate });
    return { ...input, T, f16: false, splitT: 1, splits: plan.splits, ok: true, before, after, outputValues: plan.active, partialValues: plan.partial, signedZeroChecked: true, inputPaddingNaN: true, paddingAndTailsChecked: true, memory: plan,
      dispatches: { baseline: plan.splits > 1 ? 2 : 1, candidate: plan.splits > 1 ? 2 : 1 }, timing: { baseline, candidate, ...choice, candidateMs: choice.wideMs, pairs: 3, repetitions: 4, mode: 0, hasBias: false, allocationsAndUploadsTimed: false } };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  if (params.has("T") && Number(params.get("T")) !== T) throw new Error("tall narrow probe supports T97 only");
  const all = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "down", N: 1536, K: 6144 }, { shape: "q2048", N: 2048, K: 1536 }, { shape: "ragged", N: 65, K: 1056 }, { shape: "zeros", N: 65, K: 1056, pattern: "zeros" }, { shape: "zeros-unsplit", N: 17, K: 32, pattern: "zeros" }, { shape: "cancellation", N: 65, K: 1056, pattern: "cancellation" }];
  const inputs = all.filter(({ shape }) => params.has("shape") ? shape === params.get("shape") : params.get("guards") === "1" ? true : params.get("all") === "1" ? ["gate6144", "down", "q2048"].includes(shape) : shape === "gate6144");
  if (!inputs.length) throw new Error("unknown tall narrow shape");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    metadata = { f16: false, adapter: { name: runtime.name, features: [...runtime.device.features] }, coverage: tallNarrowProof(), temporaryLimitBytes: LIMIT, immutableWeightsExcludedFromTemporaryLimit: true };
    if (!runtime.device.features.has("timestamp-query")) throw new Error("tall narrow probe requires timestamp-query");
    const source = await kernelSource(params.get("wasm") || undefined); metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source), "compiling tall narrow DEV matmuls"); metadata.shaderHashes = pipes.hashes;
    for (const input of inputs) { log({ status: "checking", ...input }); const result = await withGpuErrors(runtime, () => check(runtime, pipes, input), "checking tall narrow DEV matmuls"); cases.push(result); log(result); }
    const metricMs = Math.exp(cases.reduce((sum, item) => sum + Math.log(item.timing.candidateMs), 0) / cases.length);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matmul-tall-narrow", ...metadata, metricMs, elapsedMs: performance.now() - started, method: "candidate GPU timestamp milliseconds, BM112/BN32 plus unchanged reducer versus registered BM56/BN64 plus the same reducer", correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matmul-tall-narrow" };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
