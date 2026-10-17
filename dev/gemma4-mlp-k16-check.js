import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, rowsPerThread, withGpuErrors } from "../js/src/gpu.js";
import { createGemma4BatchMlpValidator, gemma4BatchMlpValidationGroups } from "../js/src/gpu-gemma4-batch-mlp.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const CHUNK = 1024 * 1024, LIMIT = 64 * CHUNK, SENTINEL = new Uint32Array(new Float32Array([12345.25]).buffer)[0];
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const body = (code, name) => code.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0];
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

export function pairedK16Code(original) {
  let code = original;
  const replace = (from, to) => {
    if (code.split(from).length !== 2) throw new Error(`K16 transform marker mismatch: ${from}`);
    code = code.replace(from, to);
  };
  if (code.includes("enable f16;") || (code.match(/var acc\d+ = vec4<f32>\(0\.0\);/g) || []).length !== 7 || !code.includes("let m0 = wg.y * 56u;")) throw new Error("K16 requires rendered Q8/F32 R4 BM56");
  for (const name of ["xs", "gs", "us"]) replace(`var<workgroup> ${name}: array<vec4<f32>, 512>;`, `var<workgroup> ${name}: array<vec4<f32>, 256>;`);
  const accumulate = body(code, "accumulate");
  if (!accumulate || (accumulate.match(/ \* 8u/g) || []).length !== 11) throw new Error("K16 ordered accumulator layout changed");
  replace(accumulate, accumulate.replace("let nx = c & 7u;", "let nx = (c >> 1u) & 3u;").replaceAll(" * 8u", " * 4u"));
  replace("  let sw = lr & 7u;", "  let sw = (lr >> 1u) & 3u;");
  const start = code.indexOf("  for (var kb = kb0; kb < kb1; kb++) {"), end = code.indexOf("  let row = m0 + lid.y;", start);
  if (start < 0 || end < start) throw new Error("K16 K32 loop bounds changed");
  const oldLoop = code.slice(start, end);
  const calls = [...oldLoop.matchAll(/accumulate\((\d)u,[\s\S]*?\);/g)];
  if (calls.length !== 8 || calls.some((call, index) => Number(call[1]) !== index) || (oldLoop.match(/workgroupBarrier\(\)/g) || []).length !== 2) throw new Error("K16 original dot sequence changed");
  const nextLoop = `  for (var kb = kb0; kb < kb1; kb++) {
    for (var kh = 0u; kh < 2u; kh++) {
      var xv = vec4<f32>(0.0);
      if (lr < 56u && xrow < T) {
        let base = (xrow * K + kb * 32u) / 4u + kh * 4u + lq;
        xv = X[base];
      }
      var gv = vec4<f32>(0.0);
      var uv = vec4<f32>(0.0);
      if (wrow < N) {
        let at = (wrow * K + kb * 32u) / 16u + kh;
        let gate = WG[at];
        let up = WU[at];
        let scaleG = SG[wrow * nb + kb];
        let scaleU = SU[wrow * nb + kb];
        gv = sx(gate[lq]) * scaleG;
        uv = sx(up[lq]) * scaleU;
      }
      if (lr < 56u) { xs[lr * 4u + lq] = vec4<f32>(xv); }
      gs[lr * 4u + (lq ^ sw)] = vec4<f32>(gv);
      us[lr * 4u + (lq ^ sw)] = vec4<f32>(uv);
      workgroupBarrier();
${calls.slice(0, 4).map(([call]) => "      " + call.replaceAll("\n", "\n  ")).join("\n")}
      workgroupBarrier();
    }
  }
`;
  code = code.slice(0, start) + nextLoop + code.slice(end);
  for (const name of ["sx", "gelu", "store_pair", "mm_splits"]) if (!body(original, name) || body(code, name) !== body(original, name)) throw new Error(`K16 changed ${name} arithmetic`);
  const dots = (text) => [...text.matchAll(/\*acc\d+ \+= vec4<f32>\([^;]+;/g)].map(([statement]) => statement);
  if (JSON.stringify(dots(code)) !== JSON.stringify(dots(original))) throw new Error("K16 changed ordered dot/add expressions");
  if (code.slice(0, code.indexOf("  for (var kb = kb0;")).includes("let per = (nb + splits - 1u) / splits;") === false) throw new Error("K16 changed original quantization-block split policy");
  return code;
}

export function k16LayoutProof() {
  const slots = new Set(), rows = new Set(), sequence = [];
  for (let half = 0; half < 2; half++) for (let q = 0; q < 4; q++) sequence.push(half * 4 + q);
  for (let row = 0; row < 64; row++) for (let q = 0; q < 4; q++) {
    const index = row * 4 + (q ^ ((row >> 1) & 3));
    if (index < row * 4 || index >= row * 4 + 4 || slots.has(index)) throw new Error("K16 weight tile aliases or escapes its row");
    slots.add(index);
  }
  for (let y = 0; y < 8; y++) for (let i = 0; i < 7; i++) rows.add(y + i * 8);
  if (rows.size !== 56 || Math.max(...rows) !== 55 || slots.size !== 256 || sequence.some((value, index) => value !== index)) throw new Error("K16 row or dot coverage failed");
  return { weightSlots: slots.size, physicalRows: rows.size, logicalDotOrder: sequence, sharedBytes: 3 * 256 * 16, swizzle: "(row>>1)&3" };
}

export function memoryPlan(T, N, K, plainVariant = false) {
  if (!Number.isSafeInteger(T) || T < 1 || rowsPerThread(T, { f16: false, groups: 1, kernel: "matmul" }) !== 4 || Math.ceil(T / 56) !== Math.ceil(T / 64)) throw new Error("K16 probe requires a matching current R4 BM56 grid");
  const splits = mmSplits(1, N, K, 256, 64, 64), paddedRows = Math.ceil(T / 64) * 64;
  const active = T * N, partial = splits > 1 ? splits * active : 0;
  const xBytes = paddedRows * K * 4, yBytes = (paddedRows * N + 4) * 4;
  const partBytes = (partial + 4) * 4, pairBytes = (2 * partial + 4) * 4;
  const gpuTemporaryBytes = xBytes + (plainVariant ? 6 : 5) * yBytes + 2 * partBytes + 2 * pairBytes + 656 + (plainVariant ? 32 : 0);
  const controlledTemporaryBytes = gpuTemporaryBytes + 2 * CHUNK + 1024;
  return { splits, active, partial, paddedRows, xBytes, yBytes, partBytes, pairBytes, gpuTemporaryBytes, hostStageBytes: CHUNK, uploadAllowanceBytes: CHUNK, controlledTemporaryBytes, limitBytes: LIMIT, immutableWeightBytes: 2 * N * K * 9 / 8 };
}

async function compile(device, source, options) {
  const spec = { weights: "q8", f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
  const codes = { ordinary: source("matmul_gemma_batch", spec), reduce: source("reduce_gemma_batch", spec), gelu: source("gemma4_gelu"), paired: source("matmul_gelu", spec), pairReduce: source("reduce_gelu", spec) };
  codes.candidate = options.derive(codes.paired, codes.ordinary, source);
  if (options.plainVariant) codes.ordinaryVariant = options.plainVariant(codes.ordinary, source);
  if (body(codes.gelu, "gelu") !== body(codes.paired, "gelu") || body(codes.pairReduce, "gelu") !== body(codes.paired, "gelu")) throw new Error("registered GELU arithmetic differs");
  const custom = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const layouts = { ordinary: matmulLayout(device), reduce: reduceLayout(device), paired: custom(["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"]), pairReduce: custom(["uniform", "uniform", "read-only-storage", "storage"]) };
  layouts.candidate = layouts.paired;
  layouts.ordinaryVariant = layouts.ordinary;
  const compiled = {};
  for (const [name, code] of Object.entries(codes)) compiled[name] = await pipeline(device, code, `mlp-k16.${name}`, layouts[name] ? device.createPipelineLayout({ bindGroupLayouts: [layouts[name]] }) : "auto");
  let partCompare;
  if (options.plainVariant) {
    const code = `struct P { len: u32, valid: u32, offset: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ordinary: array<u32>;
@group(0) @binding(2) var<storage, read> paired: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= p.len) { return; }
  let at = select(p.offset + i, 2u * p.valid + i - p.valid, i >= p.valid);
  if (ordinary[i] != paired[at]) { atomicAdd(&status[0], 1u); }
}`;
    layouts.partCompare = custom(["uniform", "read-only-storage", "read-only-storage", "storage"]);
    partCompare = await pipeline(device, code, "mlp-probe.compare-separate-parts", device.createPipelineLayout({ bindGroupLayouts: [layouts.partCompare] }));
  }
  return { ...compiled, layouts, partCompare, validator: await createGemma4BatchMlpValidator(device), hashes: Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)]))) };
}

async function check(runtime, pipes, input, options) {
  const { device } = runtime, U = GPUBufferUsage, { T, N, K, pattern = "normal" } = input, plan = memoryPlan(T, N, K, !!pipes.ordinaryVariant);
  if (plan.controlledTemporaryBytes > LIMIT) return { ...input, ok: true, skipped: true, reason: "temporary-memory-limit", memory: plan };
  if (Math.max(N * K, plan.xBytes, plan.yBytes, plan.pairBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("K16 probe exceeds device buffer limits");
  const owned = [], deadline = performance.now() + 30000;
  let allocated = 0, read, queries, mapped = false;
  const make = (size, usage, label) => {
    allocated += size;
    if (allocated > plan.gpuTemporaryBytes + plan.immutableWeightBytes) throw new Error("K16 allocation exceeds declared memory plan");
    const buffer = device.createBuffer({ size, usage, label: `mlp-k16.${label}` }); owned.push(buffer); return buffer;
  };
  const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const work = (pass, length) => { const count = Math.ceil(length / 256); pass.dispatchWorkgroups(Math.min(count, 65535), Math.ceil(count / 65535)); };
  const drain = async () => {
    const remaining = deadline - performance.now(); if (remaining <= 0) throw new Error("K16 upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("K16 upload deadline exceeded")), remaining); })]); }
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
    const weightFor = async (up) => {
      const w = make(N * K, U.STORAGE | U.COPY_DST, "weight"), s = make(N * K / 8, U.STORAGE | U.COPY_DST, "scale");
      await generated(w, N * K, 1, (i) => {
        const col = Math.floor(i / K), k = i % K;
        if (pattern === "saturation" && col < 2) return up ? 17 : col ? -128 : 127;
        if (pattern === "cancellation") return 1 + (col * 13 + (k & ~7) * 7 + (up ? 37 : 19)) % 127;
        return (i * 31 + (i >>> 5) * 17 + (up ? 37 : 19)) % 256 - 128;
      });
      await generated(s, N * K / 8, 4, (i) => pattern === "saturation" && i < 2 * K / 32 ? 0.05 : i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
      return { w, s };
    };
    const gateW = await weightFor(false), upW = await weightFor(true), x = make(plan.xBytes, U.STORAGE | U.COPY_DST, "input");
    await generated(x, plan.xBytes, 4, (i) => {
      const row = Math.floor(i / K), k = i % K;
      if (row >= T) return NaN;
      if (pattern === "zeros") return i % 2 ? -0 : 0;
      if (pattern === "saturation") return 0.32 * (1 + row / 32);
      if (pattern === "cancellation") return (k % 8 < 4 ? 64 : -64) + (k % 4 + 1) * 0.0625 + (row % 3) * 0.03125;
      return Math.fround(0.04 * Math.sin(k * 0.127 + row * 0.13 + 0.3));
    });
    const uniform = (values, label) => { const buffer = make(16, U.UNIFORM | U.COPY_DST, label); device.queue.writeBuffer(buffer, 0, new Uint32Array(values)); return buffer; };
    const globals = uniform([T, 1, T, 0], "globals"), params = uniform([N, K, 0, 0], "ordinary-params"), pairParams = uniform([N, K, 1, 0], "pair-params"), geluParams = uniform([N, 0, 0, 0], "gelu-params");
    const zero = make(16, U.STORAGE | U.COPY_DST, "disabled-bias"); device.queue.writeBuffer(zero, 0, new Float32Array(4).fill(NaN));
    const outputs = Array.from({ length: pipes.ordinaryVariant ? 6 : 5 }, () => make(plan.yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "output"));
    const parts = [0, 1].map(() => make(plan.partBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "ordinary-part"));
    const pairedParts = [0, 1].map(() => make(plan.pairBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "paired-part"));
    const status = make(16, U.STORAGE | U.COPY_SRC | U.COPY_DST, "status"), resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    read = make(256, U.MAP_READ | U.COPY_DST, "readback");
    const checkParams = (size, active) => uniform([size / 4, active, SENTINEL, 0x7fc00000], "validation-params");
    const yp = checkParams(plan.yBytes, plan.active), pp = checkParams(plan.partBytes, plan.partial), pairp = checkParams(plan.pairBytes, plan.partial * 2);
    const buffers = [...outputs.map((buffer) => ({ buffer, params: yp, length: plan.yBytes / 4 })), ...parts.map((buffer) => ({ buffer, params: pp, length: plan.partBytes / 4 })), ...pairedParts.map((buffer) => ({ buffer, params: pairp, length: plan.pairBytes / 4 }))];
    const fills = buffers.map((item) => ({ ...item, group: group(pipes.validator.fillLayout, [item.params, item.buffer]) }));
    const compare = (a, b, p, length) => ({ length, ...gemma4BatchMlpValidationGroups(device, pipes.validator, { params: p, baseline: a, candidate: b, status }) });
    const comparisons = [compare(outputs[2], outputs[3], yp, plan.yBytes / 4), compare(outputs[2], outputs[4], yp, plan.yBytes / 4), compare(pairedParts[0], pairedParts[1], pairp, plan.pairBytes / 4), ...parts.map((p) => compare(p, p, pp, plan.partBytes / 4)), ...outputs.slice(0, 2).map((y) => compare(y, y, yp, plan.yBytes / 4))];
    const partComparisons = pipes.ordinaryVariant ? parts.map((part, i) => {
      const p = uniform([plan.partBytes / 4, plan.partial, i * plan.partial, 0], "partial-offset-params");
      return group(pipes.layouts.partCompare, [p, part, pairedParts[0], status]);
    }) : [];
    const ordinary = [gateW, upW].map((weight, i) => ({ group: group(pipes.layouts.ordinary, [globals, params, x, weight.w, weight.s, zero, outputs[i], parts[i]]), reduce: group(pipes.layouts.reduce, [globals, params, parts[i], zero, outputs[i]]) }));
    const geluGroup = group(pipes.gelu.getBindGroupLayout(0), [globals, geluParams, outputs[0], outputs[1], outputs[2]]);
    const pairs = [0, 1].map((i) => ({ group: group(pipes.layouts.paired, [globals, pairParams, x, gateW.w, gateW.s, upW.w, upW.s, outputs[3 + i], pairedParts[i]]), reduce: group(pipes.layouts.pairReduce, [globals, pairParams, pairedParts[i], outputs[3 + i]]) }));
    const dispatch = (pass, kind) => {
      if (kind === "ordinary" || kind === "ordinaryVariant") {
        for (const op of ordinary) {
          pass.setPipeline(kind === "ordinary" ? pipes.ordinary : pipes.ordinaryVariant); pass.setBindGroup(0, op.group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), plan.splits);
          if (plan.splits > 1) { pass.setPipeline(pipes.reduce); pass.setBindGroup(0, op.reduce); work(pass, plan.active); }
        }
        pass.setPipeline(pipes.gelu); pass.setBindGroup(0, geluGroup); work(pass, plan.active);
      } else {
        const i = kind === "paired" ? 0 : 1;
        pass.setPipeline(i ? pipes.candidate : pipes.paired); pass.setBindGroup(0, pairs[i].group); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), plan.splits);
        if (plan.splits > 1) { pass.setPipeline(pipes.pairReduce); pass.setBindGroup(0, pairs[i].reduce); work(pass, plan.active); }
      }
    };
    const verify = async () => {
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(status);
      const fill = encoder.beginComputePass(); fill.setPipeline(pipes.validator.fill);
      for (const item of fills) { fill.setBindGroup(0, item.group); work(fill, item.length); } fill.end();
      const compute = encoder.beginComputePass(); for (const kind of ["ordinary", "paired", "candidate"]) dispatch(compute, kind); compute.end();
      const check = encoder.beginComputePass(); check.setPipeline(pipes.validator.compare);
      for (const item of comparisons) { check.setBindGroup(0, item.compare); work(check, item.length); } check.end();
      const compareParts = () => {
        const pass = encoder.beginComputePass(); pass.setPipeline(pipes.partCompare);
        for (const binding of partComparisons) { pass.setBindGroup(0, binding); work(pass, plan.partBytes / 4); } pass.end();
      };
      if (pipes.ordinaryVariant) {
        compareParts();
        encoder.copyBufferToBuffer(outputs[2], 0, outputs[5], 0, plan.yBytes);
        const reset = encoder.beginComputePass(); reset.setPipeline(pipes.validator.fill);
        for (const item of fills.filter(({ buffer }) => outputs.slice(0, 3).includes(buffer) || parts.includes(buffer))) { reset.setBindGroup(0, item.group); work(reset, item.length); } reset.end();
        const compute = encoder.beginComputePass(); dispatch(compute, "ordinaryVariant"); compute.end();
        const check = encoder.beginComputePass(); check.setPipeline(pipes.validator.compare);
        const extra = [compare(outputs[5], outputs[2], yp, plan.yBytes / 4), ...comparisons.slice(3)];
        for (const item of extra) { check.setBindGroup(0, item.compare); work(check, item.length); } check.end();
        compareParts();
      }
      encoder.copyBufferToBuffer(status, 0, read, 0, 16); device.queue.submit([encoder.finish()]);
      const within = await mapBeforeDeadline(read, 16, deadline); mapped = true;
      try {
        if (!within) throw new Error("K16 validation deadline exceeded");
        const counts = new Uint32Array(read.getMappedRange(0, 16));
        return { ok: counts.every((value) => value === 0), bitMismatches: counts[0], nonFinite: counts[1], tailMismatches: counts[2], activationValuesPerReference: plan.active, pairedPartialValues: 2 * plan.partial, ordinaryPartialReferenceChecked: !!pipes.ordinaryVariant, signedZeroChecked: true, paddingAndTailsChecked: true, readbackBytes: 16 };
      } finally { read.unmap(); mapped = false; }
    };
    const before = await verify(); if (!before.ok) throw new Error(`K16 initial raw-bit guard failed: ${JSON.stringify(before)}`);
    const samples = { ordinary: [], paired: [], candidate: [], ...(pipes.ordinaryVariant ? { ordinaryVariant: [] } : {}) };
    const kinds = Object.keys(samples), queryCount = kinds.length * 3 * 2;
    queries = device.createQuerySet({ type: "timestamp", count: queryCount });
    const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass();
    for (const kind of Object.keys(samples)) dispatch(warm, kind); warm.end();
    const indices = Object.fromEntries(kinds.map((kind) => [kind, []])); let index = 0;
    for (let pair = 0; pair < 3; pair++) for (const kind of pair % 2 ? [...kinds].reverse() : kinds) {
      indices[kind].push(index);
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
      for (let repetition = 0; repetition < 4; repetition++) dispatch(pass, kind); pass.end(); index += 2;
    }
    encoder.resolveQuerySet(queries, 0, queryCount, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, queryCount * 8); device.queue.submit([encoder.finish()]);
    const within = await mapBeforeDeadline(read, queryCount * 8, deadline); mapped = true;
    if (!within) throw new Error("K16 timing deadline exceeded");
    const stamps = new BigInt64Array(read.getMappedRange(0, queryCount * 8).slice(0)); read.unmap(); mapped = false;
    for (const [kind, points] of Object.entries(indices)) samples[kind] = points.map((i) => Number(stamps[i + 1] - stamps[i]) / 1e6 / 4);
    if (Object.values(samples).some((values) => values.some((value) => !Number.isFinite(value) || value <= 0))) throw new Error("invalid K16 timestamps");
    const after = await verify(); if (!after.ok) throw new Error(`K16 post-timing raw-bit guard failed: ${JSON.stringify(after)}`);
    const choices = { [`ordinaryVs${options.candidateName}`]: selectMatmulKernel({ generic: samples.ordinary, wide: samples.candidate }), [`pairedK32Vs${options.candidateName}`]: selectMatmulKernel({ generic: samples.paired, wide: samples.candidate }), ordinaryVsPairedK32: selectMatmulKernel({ generic: samples.ordinary, wide: samples.paired }), ...(pipes.ordinaryVariant ? { ordinaryVsOrdinaryDecode: selectMatmulKernel({ generic: samples.ordinary, wide: samples.ordinaryVariant }) } : {}) };
    return { ...input, ok: true, f16: false, splitT: 1, splits: plan.splits, physicalBM: 56, virtualBM: 64, sharedBytes: { ordinary: 16384, pairedK32: 24576, [`candidate${options.candidateName}`]: options.sharedBytes }, before, after, memory: plan,
      dispatches: { ordinary: plan.splits > 1 ? 5 : 3, paired: plan.splits > 1 ? 2 : 1, candidate: plan.splits > 1 ? 2 : 1 }, timing: { samples, choices, pairs: 3, repetitions: 4, allocationsAndUploadsTimed: false, candidateMs: choices[`ordinaryVs${options.candidateName}`].wideMs } };
  } finally { if (mapped) read?.unmap(); queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

export async function runMlpProbe(options = {}) {
  options = { derive: pairedK16Code, proof: k16LayoutProof, candidateName: "K16", checkName: "gemma4-mlp-k16", sharedBytes: 12288, ...options };
  const params = new URLSearchParams(location.search), started = performance.now(), T = Number(params.get("T") || 97);
  const all = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "gate12288", N: 12288, K: 1536 }, { shape: "ragged", N: 65, K: 1056 }, { shape: "zeros", N: 65, K: 1056, pattern: "zeros" }, { shape: "zeros-unsplit", N: 17, K: 32, pattern: "zeros" }, { shape: "cancellation", N: 65, K: 1056, pattern: "cancellation" }, { shape: "saturation", N: 17, K: 1056, pattern: "saturation" }];
  const inputs = all.filter(({ shape }) => params.has("shape") ? shape === params.get("shape") : params.get("guards") === "1" ? shape !== "gate12288" : shape === "gate6144");
  if (!inputs.length) throw new Error("unknown K16 shape");
  for (const shape of inputs) memoryPlan(T, shape.N, shape.K);
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    metadata = { f16: false, adapter: { name: runtime.name, features: [...runtime.device.features], maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize }, layoutProof: options.proof(), temporaryLimitBytes: LIMIT, immutableWeightsExcludedFromTemporaryLimit: true, ...options.metadata };
    if (!runtime.device.features.has("timestamp-query")) throw new Error("K16 probe requires timestamp-query");
    if (runtime.device.limits.maxComputeWorkgroupStorageSize < 24576) throw new Error(`registered paired K32 diagnostic control requires 24 KiB shared storage; candidate needs ${options.sharedBytes / 1024} KiB`);
    const source = await kernelSource(params.get("wasm") || undefined); metadata.wasmUrl = source.wasmUrl;
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source, options), "compiling DEV paired MLP probe"); metadata.shaderHashes = pipes.hashes;
    for (const shape of inputs) {
      log({ status: "checking", T, ...shape });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, { T, ...shape }, options), "checking DEV paired MLP probe"); cases.push(result); log(result);
    }
    const timed = cases.filter((item) => !item.skipped), metricMs = timed.length ? Math.exp(timed.reduce((sum, item) => sum + Math.log(item.timing.candidateMs), 0) / timed.length) : null;
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: options.checkName, ...metadata, metricMs, elapsedMs: performance.now() - started,
      method: options.plainVariant ? "candidate GPU timestamp milliseconds; four-way full-MLP comparison of current ordinary F32 BM56, registered paired K32, paired Marlin decode and ordinary Marlin decode" : "candidate GPU timestamp milliseconds; three-way full-MLP comparison with current ordinary F32 BM56 and registered paired K32", correctness: { ok: timed.length > 0, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

export function startMlpProbe(options = {}) {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: options.checkName || "gemma4-mlp-k16" };
  runMlpProbe(options).catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}

if (typeof window !== "undefined" && document.querySelector('script[src="./gemma4-mlp-k16-check.js"]')) startMlpProbe();
