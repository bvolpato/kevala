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
const body = (code, name) => code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0];
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

export function narrowSerialCode(original) {
  let code = original;
  const replace = (from, to) => {
    if (code.split(from).length !== 2) throw new Error(`narrow serial transform marker mismatch: ${from}`);
    code = code.replace(from, to);
  };
  if (code.includes("enable f16;") || !code.includes("let m0 = wg.y * 56u;") || (code.match(/var acc\d = vec4<f32>\(0\.0\);/g) || []).length !== 7) throw new Error("narrow serial requires registered Q8/F32 R4 BM56 paired source");
  for (const name of ["gs", "us"]) replace(`var<workgroup> ${name}: array<vec4<f32>, 512>;`, `var<workgroup> ${name}: array<vec4<f32>, 256>;`);
  replace("  let n0 = wg.x * 64u;", "  let n0 = wg.x * 32u;");
  const oldAccumulate = body(code, "accumulate");
  if (!oldAccumulate) throw new Error("narrow serial accumulator function missing");
  let accumulate = oldAccumulate.replaceAll("ptr<function, vec4<f32>>", "ptr<function, vec2<f32>>");
  for (const name of ["g1", "u1"]) {
    const line = accumulate.match(new RegExp(`^  let ${name} = [^;]+;\\n`, "m"))?.[0];
    if (!line) throw new Error(`narrow serial ${name} load missing`);
    accumulate = accumulate.replace(line, "");
  }
  let dots = 0;
  accumulate = accumulate.replace(/\*acc(\d) \+= vec4<f32>\(dot\(a\1, g0\), dot\(a\1, g1\), dot\(a\1, u0\), dot\(a\1, u1\)\);/g, (_, i) => { dots++; return `*acc${i} += vec2<f32>(dot(a${i}, g0), dot(a${i}, u0));`; });
  if (dots !== 7 || /\b(?:g1|u1)\b/.test(accumulate)) throw new Error("narrow serial changed the independent gate/up dot mapping");
  replace(oldAccumulate, accumulate);
  const oldStore = body(code, "store_pair");
  if (!oldStore?.includes("PART[(splits + z) * T * N + o] = value.y;")) throw new Error("narrow serial original partial-plane layout changed");
  replace(oldStore, `fn store_pair(value: vec2<f32>, row: u32, col: u32, T: u32, N: u32, splits: u32, z: u32) {
  if (row >= T || col >= N) { return; }
  var gate = value.x;
  var up = value.y;
  if (splits == 1u) { gate += 0.0; up += 0.0; }
  Y[row * N + col] = gelu(gate) * up;
}`);
  replace("    if (wrow < N) {", "    if (lr < 32u && wrow < N) {");
  const weightStores = [
    "    gs[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(gv0);",
    "    gs[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(gv1);",
    "    us[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(uv0);",
    "    us[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(uv1);",
  ].join("\n");
  replace(weightStores, "    if (lr < 32u) {\n" + weightStores.split("\n").map((line) => "  " + line).join("\n") + "\n    }");
  const partition = "  let kb0 = wg.z * per;\n  let kb1 = min(nb, kb0 + per);\n", row = "  let row = m0 + lid.y;";
  const begin = code.indexOf(partition), end = code.indexOf(row, begin);
  if (begin < 0 || end < begin || code.split(partition).length !== 2) throw new Error("narrow serial partition boundaries changed");
  let inner = code.slice(begin + partition.length, end).replace(/var acc(\d) = vec4<f32>\(0\.0\);/g, "var acc$1 = vec2<f32>(0.0);");
  const calls = [...inner.matchAll(/accumulate\((\d)u,/g)].map((match) => Number(match[1]));
  if (calls.length !== 8 || calls.some((value, i) => value !== i) || (inner.match(/workgroupBarrier\(\)/g) || []).length !== 2) throw new Error("narrow serial original K32 dot order or barriers changed");
  const finals = Array.from({ length: 7 }, (_, i) => `  var final${i} = vec2<f32>(0.0);`).join("\n");
  const sums = Array.from({ length: 7 }, (_, i) => `    if (splits == 1u) { final${i} = acc${i}; } else { final${i} += acc${i}; }`).join("\n");
  code = code.slice(0, begin) + `${finals}\n  for (var split = 0u; split < splits; split++) {\n    let kb0 = split * per;\n    let kb1 = min(nb, kb0 + per);\n${inner}${sums}\n  }\n` + code.slice(end);
  const secondStores = [...code.matchAll(/^  store_pair\(acc\d\.yw[^\n]*\n/gm)];
  if (secondStores.length !== 7) throw new Error("narrow serial second-column stores changed");
  for (const [line] of secondStores) code = code.replace(line, "");
  code = code.replace(/store_pair\(acc(\d)\.xz,/g, "store_pair(final$1,");
  if (/PART\s*\[/.test(code) || (code.match(/store_pair\(final\d,/g) || []).length !== 7) throw new Error("narrow serial still accesses partials or has incorrect output stores");
  for (const name of ["sx", "gelu", "mm_splits"]) if (body(code, name) !== body(original, name)) throw new Error(`narrow serial changed ${name}`);
  return code;
}

export function serialProof(N = 6144, K = 1536) {
  const splits = mmSplits(1, N, K, 256, 64, 64), nb = K / 32, per = Math.ceil(nb / splits);
  const partitions = Array.from({ length: splits }, (_, split) => [split * per, Math.min(nb, (split + 1) * per)]);
  const order = partitions.flatMap(([start, end]) => Array.from({ length: Math.max(0, end - start) }, (_, i) => start + i));
  if (order.length !== nb || order.some((value, i) => value !== i)) throw new Error("narrow serial changed the quantization-block order");
  const slots = new Set();
  for (let row = 0; row < 32; row++) for (let kq = 0; kq < 8; kq++) slots.add(row * 8 + (kq ^ (row & 7)));
  if (slots.size !== 256 || Math.max(...slots) !== 255) throw new Error("narrow serial weight tile aliases");
  return { splits, partitions, perSplitKqOrder: [0, 1, 2, 3, 4, 5, 6, 7], partialOrder: Array.from({ length: splits }, (_, i) => i), physicalBM: 56, physicalBN: 32, virtualBM: 64, virtualBN: 64, sharedBytes: 16384, partialAccumulators: 7, finalAccumulators: 7, candidatePartAccesses: 0 };
}

export function memoryPlan(N, K) {
  const splits = mmSplits(1, N, K, 256, 64, 64), active = T * N, partial = splits > 1 ? splits * active : 0;
  const xBytes = 128 * K * 4, yBytes = (128 * N + 4) * 4, partBytes = (partial + 4) * 4;
  const gpuTemporaryBytes = xBytes + 4 * yBytes + 2 * partBytes + 16 + 7 * 16 + 16 + 16 + 256 + 256;
  return { splits, active, partial, xBytes, yBytes, partBytes, gpuTemporaryBytes, controlledTemporaryBytes: gpuTemporaryBytes + 2 * CHUNK + 1024, limitBytes: LIMIT, immutableWeightBytes: 2 * N * K * 9 / 8, hostStageBytes: CHUNK, uploadAllowanceBytes: CHUNK };
}

async function compile(device, source) {
  const spec = { weights: "q8", f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
  const codes = { baseline: source("matmul_gemma_batch", spec), reduce: source("reduce_gemma_batch", spec), gelu: source("gemma4_gelu"), originalPaired: source("matmul_gelu", spec) };
  codes.candidate = narrowSerialCode(codes.originalPaired);
  if (body(codes.candidate, "gelu") !== body(codes.gelu, "gelu")) throw new Error("narrow serial GELU differs from the registered baseline");
  const layout = matmulLayout(device), reduction = reduceLayout(device);
  const pairLayout = device.createBindGroupLayout({ entries: ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pl = (value) => device.createPipelineLayout({ bindGroupLayouts: [value] });
  return { layout, reduction, pairLayout,
    baseline: await pipeline(device, codes.baseline, "narrow-serial.current-bm56", pl(layout)), reduce: await pipeline(device, codes.reduce, "narrow-serial.current-reduce", pl(reduction)), gelu: await pipeline(device, codes.gelu, "narrow-serial.current-gelu"),
    candidate: await pipeline(device, codes.candidate, "narrow-serial.bn32-paired", pl(pairLayout)), validator: await createGemma4BatchMlpValidator(device),
    hashes: Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)]))) };
}

async function check(runtime, pipes, input) {
  const { device } = runtime, U = GPUBufferUsage, { N, K, pattern = "normal" } = input, plan = memoryPlan(N, K);
  if (plan.controlledTemporaryBytes > LIMIT) throw new Error("narrow serial controlled scratch exceeds 64 MiB");
  if (Math.max(N * K, plan.xBytes, plan.yBytes, plan.partBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("narrow serial buffer exceeds device limits");
  const owned = [], deadline = performance.now() + 30000;
  let allocated = 0, read, queries, mapped = false;
  const make = (size, usage, label) => {
    allocated += size; if (allocated > plan.gpuTemporaryBytes + plan.immutableWeightBytes) throw new Error("narrow serial allocation exceeds declared memory");
    const buffer = device.createBuffer({ size, usage, label: `narrow-serial.${label}` }); owned.push(buffer); return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const work = (pass, length) => { const count = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(count, 65535), Math.ceil(count / 65535)); };
  const drain = async () => {
    const remaining = deadline - performance.now(); if (remaining <= 0) throw new Error("narrow serial upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("narrow serial upload deadline exceeded")), remaining); })]); }
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
    const unusedPart = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "unused-candidate-PART"), status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "status");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps"); read = make(256, U.MAP_READ | U.COPY_DST, "readback");
    const yp = uniform([plan.yBytes / 4, plan.active, SENTINEL, 0x7fc00000]), pp = uniform([plan.partBytes / 4, plan.partial, SENTINEL, 0x7fc00000]), unused = uniform([4, 0, SENTINEL, 0x7fc00000]);
    const fills = [...outputs.map((buffer) => ({ length: plan.yBytes / 4, group: group(pipes.validator.fillLayout, [yp, buffer]) })), ...parts.map((buffer) => ({ length: plan.partBytes / 4, group: group(pipes.validator.fillLayout, [pp, buffer]) })), { length: 4, group: group(pipes.validator.fillLayout, [unused, unusedPart]) }];
    const compare = (baseline, candidate, params, length) => ({ length, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params, baseline, candidate, status }) });
    const checks = [compare(outputs[2], outputs[3], yp, plan.yBytes / 4), ...outputs.slice(0, 2).map((y) => compare(y, y, yp, plan.yBytes / 4)), ...parts.map((p) => compare(p, p, pp, plan.partBytes / 4)), compare(unusedPart, unusedPart, unused, 4)];
    const baseline = [gateW, upW].map((weight, i) => ({ group: group(pipes.layout, [globals, params, x, weight.w, weight.s, zero, outputs[i], parts[i]]), reduce: group(pipes.reduction, [globals, params, parts[i], zero, outputs[i]]) }));
    const geluGroup = group(pipes.gelu.getBindGroupLayout(0), [globals, geluParams, outputs[0], outputs[1], outputs[2]]);
    const candidateGroup = group(pipes.pairLayout, [globals, pairParams, x, gateW.w, gateW.s, upW.w, upW.s, outputs[3], unusedPart]);
    const dispatch = (pass, candidate) => {
      if (candidate) { pass.setPipeline(pipes.candidate); pass.setBindGroup(0, candidateGroup); pass.dispatchWorkgroups(Math.ceil(N / 32), Math.ceil(T / 56), 1); }
      else {
        for (const op of baseline) {
          pass.setPipeline(pipes.baseline); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), plan.splits);
          if (plan.splits > 1) { pass.setPipeline(pipes.reduce); pass.setBindGroup(0, op.reduce); work(pass, plan.active); }
        }
        pass.setPipeline(pipes.gelu); pass.setBindGroup(0, geluGroup); work(pass, plan.active);
      }
    };
    const verify = async () => {
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(pipes.validator.fill);
      for (const item of fills) { fill.setBindGroup(0, item.group); work(fill, item.length); } fill.end();
      const compute = encoder.beginComputePass(); dispatch(compute, false); dispatch(compute, true); compute.end();
      const compare = encoder.beginComputePass(); compare.setPipeline(pipes.validator.compare);
      for (const item of checks) { compare.setBindGroup(0, item.compare); work(compare, item.length); } compare.end();
      encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      try {
        if (!within) throw new Error("narrow serial validation deadline exceeded");
        const values = new Uint32Array(read.getMappedRange(0, 16));
        const result = { ok: values.every((value) => value === 0), bitMismatches: values[0], nonFinite: values[1], tailMismatches: values[2], outputValues: plan.active, signedZeroChecked: true, paddingAndTailsChecked: true, unusedCandidatePartUntouched: true };
        if (!result.ok) throw new Error(`narrow serial raw-bit guard failed: ${JSON.stringify(result)}`);
        return result;
      } finally { read.unmap(); mapped = false; }
    };
    const before = await verify();
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
    if (!within) throw new Error("narrow serial timestamp deadline exceeded");
    const stamps = new BigInt64Array(read.getMappedRange(0, 96).slice(0)); read.unmap(); mapped = false;
    const [baselineTimes, candidateTimes] = indices.map((items) => items.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4));
    if ([...baselineTimes, ...candidateTimes].some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("narrow serial invalid timestamps");
    const after = await verify(), choice = selectMatmulKernel({ generic: baselineTimes, wide: candidateTimes });
    return { ...input, T, ok: true, f16: false, proof: serialProof(N, K), before, after, memory: plan, dispatches: { baseline: plan.splits > 1 ? 5 : 3, candidate: 1 }, timing: { baseline: baselineTimes, candidate: candidateTimes, ...choice, candidateMs: choice.wideMs, pairs: 3, repetitions: 4, allocationsAndUploadsTimed: false } };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  if (params.has("T") && Number(params.get("T")) !== T) throw new Error("narrow serial probe supports T97 only");
  const all = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "gate12288", N: 12288, K: 1536 }, { shape: "ragged", N: 65, K: 1056 }, { shape: "zeros", N: 65, K: 1056, pattern: "zeros" }, { shape: "zeros-unsplit", N: 17, K: 32, pattern: "zeros" }, { shape: "cancellation", N: 65, K: 1056, pattern: "cancellation" }, { shape: "saturation", N: 17, K: 1056, pattern: "saturation" }];
  const inputs = all.filter(({ shape }) => params.has("shape") ? shape === params.get("shape") : params.get("guards") === "1" ? true : shape === "gate6144");
  if (!inputs.length) throw new Error("unknown narrow serial shape");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    metadata = { f16: false, adapter: { name: runtime.name, features: [...runtime.device.features] }, temporaryLimitBytes: LIMIT, immutableWeightsExcludedFromTemporaryLimit: true };
    if (!runtime.device.features.has("timestamp-query")) throw new Error("narrow serial probe requires timestamp-query");
    const source = await kernelSource(params.get("wasm") || undefined); metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source), "compiling DEV narrow serial MLP"); metadata.shaderHashes = pipes.hashes;
    for (const input of inputs) { log({ status: "checking", ...input }); const result = await withGpuErrors(runtime, () => check(runtime, pipes, input), "checking DEV narrow serial MLP"); cases.push(result); log(result); }
    const metricMs = Math.exp(cases.reduce((sum, item) => sum + Math.log(item.timing.candidateMs), 0) / cases.length);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-mlp-narrow-serial", ...metadata, metricMs, elapsedMs: performance.now() - started, method: "candidate GPU timestamp milliseconds; one paired BN32 serial-split dispatch versus registered separate gate/up, reducers and GELU", correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-mlp-narrow-serial" };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
