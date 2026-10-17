import { mmSplits, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024, CHUNK = MiB, LIMIT = 64 * MiB, SENTINEL = 12345.25;
const SPLIT_TOKENS = 1;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export function validateRows(T) {
  if (!Number.isSafeInteger(T) || T < 1 || T > 256) throw new RangeError("suffix tall requires 1..256 active rows");
}

export function validateTile(tile) {
  if (tile !== 104 && tile !== 112) throw new RangeError("suffix tall tile must be 104 or 112");
}

export function pairedF16Tall(original, tile = 112, { swizzle = "default" } = {}) {
  validateTile(tile);
  if (swizzle !== "default" && swizzle !== "half-bank") throw new RangeError("suffix tall swizzle must be default or half-bank");
  if (swizzle === "half-bank" && tile !== 104) throw new RangeError("half-bank swizzle requires tile=104");
  let code = original;
  const replace = (from, to, count = 1) => {
    if (code.split(from).length - 1 !== count) throw new Error(`paired F16 tall marker mismatch: ${from}`);
    code = code.replaceAll(from, to);
  };
  if (!code.includes("enable f16;") || (code.match(/var acc\d = vec4<f32>\(0\.0\);/g) || []).length !== 8) throw new Error("paired F16 tall requires Q8 F16 R4");
  const extra = Array.from({ length: tile / 8 - 8 }, (_, i) => i + 8);
  replace("var<workgroup> xs: array<vec4<f16>, 512>;", `var<workgroup> xs: array<vec4<f16>, ${tile * 8}>;`);
  replace("let m0 = wg.y * 64u;", `let m0 = wg.y * ${tile}u;`);
  const parameter = "  acc7: ptr<function, vec4<f32>>,\n";
  replace(parameter, parameter + extra.map((i) => `  acc${i}: ptr<function, vec4<f32>>,\n`).join(""));
  const accumulator = "  var acc7 = vec4<f32>(0.0);\n";
  replace(accumulator, accumulator + extra.map((i) => `  var acc${i} = vec4<f32>(0.0);\n`).join(""));
  const dot = "  let a7 = vec4<f32>(xs[(y + 56u) * 8u + kq]);\n  *acc7 += vec4<f32>(dot(a7, g0), dot(a7, g1), dot(a7, u0), dot(a7, u1));\n";
  replace(dot, dot + extra.map((i) => `  let a${i} = vec4<f32>(xs[(y + ${8 * i}u) * 8u + kq]);\n  *acc${i} += vec4<f32>(dot(a${i}, g0), dot(a${i}, g1), dot(a${i}, u0), dot(a${i}, u1));\n`).join(""));
  replace("      &acc7,\n", `      &acc7,\n${extra.map((i) => `      &acc${i},\n`).join("")}`, 8);
  const stores = "  store_pair(acc7.xz, row + 56u, n0 + lid.x, T, N, splits, wg.z);\n  store_pair(acc7.yw, row + 56u, n0 + lid.x + 32u, T, N, splits, wg.z);\n";
  replace(stores, stores + extra.map((i) => `  store_pair(acc${i}.xz, row + ${8 * i}u, n0 + lid.x, T, N, splits, wg.z);\n  store_pair(acc${i}.yw, row + ${8 * i}u, n0 + lid.x + 32u, T, N, splits, wg.z);\n`).join(""));
  const firstInput = "    if (lr < 64u) {\n      xs[lr * 8u + lq * 2u] = vec4<f16>(xv0);\n      xs[lr * 8u + lq * 2u + 1u] = vec4<f16>(xv1);\n    }\n";
  replace(firstInput, firstInput + `    var xv2 = vec4<f32>(0.0);
    var xv3 = vec4<f32>(0.0);
    let xrow2 = xrow + 64u;
    if (lr + 64u < ${tile}u && xrow2 < T) {
      let base2 = (xrow2 * K + kb * 32u) / 4u + lq * 2u;
      xv2 = X[base2];
      xv3 = X[base2 + 1u];
    }
    if (lr + 64u < ${tile}u) {
      xs[(lr + 64u) * 8u + lq * 2u] = vec4<f16>(xv2);
      xs[(lr + 64u) * 8u + lq * 2u + 1u] = vec4<f16>(xv3);
    }
`);
  if (swizzle === "half-bank") {
    const loads = [
      "  let g0 = vec4<f32>(gs[c * 8u + (kq ^ nx)]);",
      "  let g1 = vec4<f32>(gs[(c + 32u) * 8u + (kq ^ nx)]);",
      "  let u0 = vec4<f32>(us[c * 8u + (kq ^ nx)]);",
      "  let u1 = vec4<f32>(us[(c + 32u) * 8u + (kq ^ nx)]);",
    ];
    const stores = [
      "    gs[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f16>(gv0);",
      "    gs[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f16>(gv1);",
      "    us[lr * 8u + ((lq * 2u) ^ sw)] = vec4<f16>(uv0);",
      "    us[lr * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f16>(uv1);",
    ];
    const requireOnce = (line, label) => {
      if (code.split(line).length - 1 !== 1) throw new Error(`paired F16 half-bank ${label} marker mismatch: ${line}`);
    };
    requireOnce("  let nx = c & 7u;", "loader row map");
    requireOnce("  let sw = lr & 7u;", "writer row map");
    for (const line of [...loads, ...stores]) requireOnce(line, "G/U address");
    code = code.replace("  let nx = c & 7u;", "  let nx = (c >> 1u) & 7u;")
      .replace("  let sw = lr & 7u;", "  let sw = (lr >> 1u) & 7u;");
    if (code.split("  let nx = (c >> 1u) & 7u;").length - 1 !== 1 || code.split("  let sw = (lr >> 1u) & 7u;").length - 1 !== 1
      || code.includes("let nx = c & 7u;") || code.includes("let sw = lr & 7u;")) throw new Error("paired F16 half-bank row map was not applied exactly");
    for (const line of [...loads, ...stores]) requireOnce(line, "G/U address after row-map change");
  }
  const body = (text, name) => text.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n\\}`))?.[0];
  for (const name of ["sx", "gelu", "store_pair", "mm_splits"]) if (!body(original, name) || body(code, name) !== body(original, name)) throw new Error(`paired F16 tall changed ${name}`);
  const dots = (text) => [...text.matchAll(/\*acc\d+ \+= vec4<f32>\([^;]+;/g)].map(([statement]) => statement);
  if (dots(code).length !== tile / 8 || JSON.stringify(dots(code).slice(0, 8)) !== JSON.stringify(dots(original))) throw new Error("paired F16 tall changed ordered gate/up dots");
  return code;
}

export function memoryPlan(T, N, K, tile = 112) {
  validateRows(T);
  validateTile(tile);
  const splits = mmSplits(SPLIT_TOKENS, N, K, 256, 64, 64), candidateSplits = mmSplits(SPLIT_TOKENS, N, K, 256, 64, 64);
  const paddedRows = Math.max(Math.ceil(T / 64) * 64, Math.ceil(T / tile) * tile), length = T * N, capacity = paddedRows * N;
  const partialLength = splits > 1 ? splits * length : 0;
  const yBytes = (capacity + 4) * 4, pairPartBytes = (2 * partialLength + 4) * 4;
  const immutableWeightBytes = 2 * N * K * 9 / 8, xBytes = paddedRows * K * 4;
  const gpuTemporaryBytes = xBytes + 2 * yBytes + 2 * pairPartBytes + 2 * CHUNK + 32 + 256;
  return { splits, candidateSplits, paddedRows, length, capacity, partialLength, yBytes, pairPartBytes, xBytes, immutableWeightBytes,
    gpuTemporaryBytes, gpuBytes: gpuTemporaryBytes + immutableWeightBytes, hostStageBytes: CHUNK, mappedReadBytes: 2 * CHUNK, uploadAllowanceBytes: CHUNK,
    controlledTemporaryBytes: gpuTemporaryBytes + 4 * CHUNK + 128, limitBytes: LIMIT };
}

async function compile(device, source, tile, { baselineMode = "current", swizzle = "default", registered = false } = {}) {
  const spec = { weights: "q8", f16: true, rows: 4, row56: false, groups: 1, splitTarget: 256 };
  const base = source("matmul_gelu", spec), baseline = baselineMode === "tall" ? source("matmul_gelu_tall", spec) : base;
  const derived = pairedF16Tall(base, tile, { swizzle }), reduce = source("reduce_gelu", spec);
  const candidate = baselineMode === "tall" || swizzle === "half-bank" ? derived : registered ? source("matmul_gelu_tall", spec) : derived;
  if (baselineMode === "tall" && tile !== 104) throw new Error("registered tall baseline supports tile=104 only");
  if (baselineMode === "tall" && swizzle === "default" && candidate !== baseline) throw new Error("registered tall baseline differs from the validated dev derivation");
  if (baselineMode !== "tall" && candidate !== derived) throw new Error("registered tall kernel differs from the validated dev derivation");
  const customLayout = (label, types) => device.createBindGroupLayout({ label, entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pairLayout = customLayout("suffix tall paired", ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage"]);
  const pairReduceLayout = customLayout("suffix tall paired reduce", ["uniform", "uniform", "read-only-storage", "storage"]);
  const pl = (group) => device.createPipelineLayout({ bindGroupLayouts: [group] });
  const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");
  const sourceHashes = { baseline: await hash(baseline), candidate: await hash(candidate), reduce: await hash(reduce) };
  const baselineTile = baselineMode === "tall" ? 104 : 64;
  return { tile, baselineTile, baselineKernel: baselineMode === "tall" ? "matmul_gelu_tall" : "matmul_gelu",
    candidateKernel: baselineMode === "tall" || swizzle === "half-bank" ? `dev BM${tile} ${swizzle} transform of matmul_gelu` : registered ? "matmul_gelu_tall" : `dev BM${tile} transform of matmul_gelu`,
    swizzle, pairLayout, pairReduceLayout, sourceHashes, sourceSpec: spec,
    baseline: await pipeline(device, baseline, baselineMode === "tall" ? "suffix-tall.baseline-registered-bm104" : "suffix-tall.f16-r4-paired", pl(pairLayout)),
    candidate: await pipeline(device, candidate, `suffix-tall.f16-bm${tile}-paired`, pl(pairLayout)),
    reduce: await pipeline(device, reduce, "suffix-tall.paired-reduce", pl(pairReduceLayout)) };
}

async function check(runtime, pipes, input) {
  const { T, N, K, pattern = "normal" } = input, { device } = runtime, { tile, baselineTile = 64 } = pipes, U = GPUBufferUsage, plan = memoryPlan(T, N, K, tile);
  if (plan.controlledTemporaryBytes > LIMIT) return { ...input, ok: true, skipped: true, reason: "controlled-temporary-memory-limit", memory: plan };
  if (Math.max(N * K, plan.yBytes, plan.pairPartBytes, plan.xBytes, 2 * CHUNK) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("suffix tall probe exceeds device buffer limits");
  const owned = [], deadline = performance.now() + 30_000;
  let gpuBytes = 0, queries;
  const make = (size, usage, label) => {
    gpuBytes += size;
    if (gpuBytes > plan.gpuBytes) throw new Error("suffix tall probe exceeded planned GPU allocation");
    const buffer = device.createBuffer({ size, usage, label: `suffix-tall.${label}` }); owned.push(buffer); return buffer;
  };
  const map = async (read, size) => { if (!await mapBeforeDeadline(read, size, deadline)) { read.unmap(); throw new Error("suffix tall readback deadline exceeded"); } };
  const drain = async () => {
    const remaining = deadline - performance.now(); if (remaining <= 0) throw new Error("suffix tall upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("suffix tall upload deadline exceeded")), remaining); })]); }
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
      const w = make(N * K, U.STORAGE | U.COPY_DST, up ? "up-W" : "gate-W"), s = make(N * K / 8, U.STORAGE | U.COPY_DST, "scales");
      await generated(w, N * K, 1, (i) => {
        const col = Math.floor(i / K), k = i % K;
        if (pattern === "saturated" && col < 2) return up ? 17 : col ? -128 : 127;
        if (pattern === "cancellation") return (k % 2 ? -1 : 1) * (1 + (col * 13 + (k & ~1) * 7) % 127);
        return (i * 31 + (i >>> 5) * 17 + (up ? 37 : 19)) % 256 - 128;
      });
      await generated(s, N * K / 8, 4, (i) => pattern === "saturated" && i < 2 * K / 32 ? 0.05 : i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
      return { w, s };
    };
    const gateW = await weightsFor(false), upW = await weightsFor(true), x = make(plan.xBytes, U.STORAGE | U.COPY_DST, "X");
    await generated(x, plan.xBytes, 4, (i) => {
      const row = Math.floor(i / K), k = i % K;
      if (row >= T) return NaN;
      if (pattern === "zeros") return i % 2 ? -0 : 0;
      if (pattern === "saturated") return 0.32 * (1 + row / 32);
      return 0.04 * Math.sin((pattern === "cancellation" ? k & ~1 : k) * 0.127 + row * 0.13 + 0.3);
    });
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), pairParams = make(16, U.UNIFORM | U.COPY_DST, "pair-params");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, SPLIT_TOKENS, T, 0]));
    device.queue.writeBuffer(pairParams, 0, new Uint32Array([N, K, SPLIT_TOKENS, 0]));
    const outputs = Array.from({ length: 2 }, () => make(plan.yBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "Y"));
    const baselinePart = make(plan.pairPartBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "baseline-paired-PART");
    const pairPart = make(plan.pairPartBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "paired-PART");
    const baselineOutput = outputs[0], candidateOutput = outputs.at(-1);
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-read"), resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const baselineGroup = group(pipes.pairLayout, [globals, pairParams, x, gateW.w, gateW.s, upW.w, upW.s, baselineOutput, baselinePart]);
    const baselineReduce = group(pipes.pairReduceLayout, [globals, pairParams, baselinePart, baselineOutput]);
    const pairGroup = group(pipes.pairLayout, [globals, pairParams, x, gateW.w, gateW.s, upW.w, upW.s, candidateOutput, pairPart]);
    const pairReduce = group(pipes.pairReduceLayout, [globals, pairParams, pairPart, candidateOutput]);
    const work = Math.ceil(plan.length / 256);
    const dispatch = (pass, candidate) => {
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, candidate ? pairGroup : baselineGroup);
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / (candidate ? tile : baselineTile)), plan.splits);
      if (plan.splits > 1) {
        pass.setPipeline(pipes.reduce); pass.setBindGroup(0, candidate ? pairReduce : baselineReduce);
        pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const compare = async (left, right, offsetRight, length, offsetLeft = 0) => {
      let zeros = 0, negativeZeros = 0;
      for (let offset = 0; offset < length; offset += CHUNK / 4) {
        const count = Math.min(CHUNK / 4, length - offset), size = count * 4, encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(left, offsetLeft + offset * 4, read, 0, size); encoder.copyBufferToBuffer(right, offsetRight + offset * 4, read, size, size); device.queue.submit([encoder.finish()]);
        await map(read, size * 2);
        try {
          const range = read.getMappedRange(0, size * 2), values = new Float32Array(range), bits = new Uint32Array(range);
          for (let i = 0; i < count; i++) {
            if (!Number.isFinite(values[i]) || !Number.isFinite(values[count + i])) throw new Error(`suffix tall nonfinite at ${offset + i}`);
            if (bits[i] !== bits[count + i]) throw new Error(`suffix tall bit mismatch at ${offset + i}: 0x${bits[i].toString(16)} vs 0x${bits[count + i].toString(16)}`);
            if (values[i] === 0) zeros++; if (bits[i] === 0x80000000) negativeZeros++;
          }
        } finally { read.unmap(); }
      }
      return { samples: length, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros, signedZeroChecked: true };
    };
    const tail = async (buffer, active, capacity) => {
      for (let offset = active; offset < capacity; offset += CHUNK / 4) {
        const size = Math.min(CHUNK / 4, capacity - offset) * 4, encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, offset * 4, read, 0, size); device.queue.submit([encoder.finish()]); await map(read, size);
        try { if (new Float32Array(read.getMappedRange(0, size)).some((value) => value !== SENTINEL)) throw new Error(`suffix tall padding/tail overwrite after ${active}`); }
        finally { read.unmap(); }
      }
    };
    const verify = async () => {
      for (const buffer of outputs) await generated(buffer, plan.yBytes, 4, (i) => i < plan.length ? NaN : SENTINEL);
      await generated(baselinePart, plan.pairPartBytes, 4, (i) => i < 2 * plan.partialLength ? NaN : SENTINEL);
      await generated(pairPart, plan.pairPartBytes, 4, (i) => i < 2 * plan.partialLength ? NaN : SENTINEL);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); device.queue.submit([encoder.finish()]);
      const output = await compare(baselineOutput, candidateOutput, 0, plan.length);
      const gatePartials = await compare(baselinePart, pairPart, 0, plan.partialLength);
      const upPartials = await compare(baselinePart, pairPart, plan.partialLength * 4, plan.partialLength, plan.partialLength * 4);
      for (const buffer of outputs) await tail(buffer, plan.length, plan.capacity + 4);
      await tail(baselinePart, 2 * plan.partialLength, 2 * plan.partialLength + 4);
      await tail(pairPart, 2 * plan.partialLength, 2 * plan.partialLength + 4);
      if (pattern === "saturated") {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(baselineOutput, 0, read, 0, 8); device.queue.submit([encoder.finish()]); await map(read, 8);
        try { const values = new Float32Array(read.getMappedRange(0, 8)); if (values[0] <= 10 || values[1] !== 0) throw new Error("GELU saturation branches were not exercised"); }
        finally { read.unmap(); }
      }
      return { output, gatePartials, upPartials, paddingAndTailGuard: true, inputPadding: "NaN outside active rows" };
    };
    const numericalCheck = await verify();
    let timing = null;
    if (device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: 12 });
      const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(), order = [];
      dispatch(warm, false); dispatch(warm, true); warm.end();
      for (let pair = 0; pair < 3; pair++) for (const fused of pair % 2 ? [true, false] : [false, true]) {
        const at = order.length * 2; order.push(fused);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, fused); pass.end();
      }
      encoder.resolveQuerySet(queries, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]); await map(read, 96);
      try {
        const stamps = new BigUint64Array(read.getMappedRange(0, 96)), samples = { baseline: [], candidate: [] };
        order.forEach((fused, i) => samples[fused ? "candidate" : "baseline"].push(Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6 / 4));
        const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
        if (![baselineMs, candidateMs].every((value) => Number.isFinite(value) && value > 0)) throw new Error("invalid suffix tall timestamps");
        timing = { samples: 3, repetitions: 4, ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs };
      } finally { read.unmap(); }
    }
    const afterTimingCheck = await verify();
    return { ...input, f16: true, candidateRows: tile / 16, candidateAccumulators: tile / 8, baselineRows: 4, candidateBM: tile, baselineBM: baselineTile, baselinePaired: true,
      eligibleForRuntime: tile === 104 && T >= 97 && T <= 104 && baselineTile === 64 && pipes.swizzle === "default",
      baselineKernel: pipes.baselineKernel, candidateKernel: pipes.candidateKernel, weightSwizzle: pipes.swizzle,
      physicalBN: 64, virtualBM: 64, virtualBN: 64, globals: [T, 1, T, 0], pairSplitT: 1, splits: plan.splits,
      ok: true, numericalCheck, afterTimingCheck, timing, baselineDispatches: plan.splits > 1 ? 2 : 1, candidateDispatches: plan.splits > 1 ? 2 : 1,
      sharedBytes: tile * 64 + 8192, baselineSharedBytes: baselineTile === 104 ? 104 * 64 + 8192 : 12288,
      memory: { ...plan, actualGpuBytes: gpuBytes, readbackChunkBytesPerSide: CHUNK } };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search), tile = Number(params.get("tile") ?? 112);
  const baselineMode = params.get("baseline") ?? "current", swizzle = params.get("swizzle") ?? "default", registered = params.get("registered") === "1";
  validateTile(tile);
  if (registered && tile !== 104) throw new Error("registered suffix tall kernel supports tile=104 only; tile=112 is dev-only");
  if (baselineMode !== "current" && baselineMode !== "tall") throw new Error("baseline must be current or tall");
  if (swizzle !== "default" && swizzle !== "half-bank") throw new Error("swizzle must be default or half-bank");
  if ((baselineMode === "tall" || swizzle === "half-bank") && tile !== 104) throw new Error("tall baseline and half-bank swizzle require tile=104");
  if (baselineMode === "tall" && registered) throw new Error("baseline=tall already uses the registered candidate as baseline; omit registered=1");
  if (swizzle === "half-bank" && baselineMode !== "tall") throw new Error("swizzle=half-bank requires baseline=tall");
  if (params.get("f16") === "0") throw new Error("suffix tall currently supports F16 only");
  let inputs = [{ shape: "gate6144", T: 97, N: 6144, K: 1536 }, { shape: "gate12288", T: 97, N: 12288, K: 1536 },
    { shape: "ragged", T: 97, N: 65, K: 1056 }, { shape: "cancellation", T: 97, N: 65, K: 1056, pattern: "cancellation" },
    { shape: "zeros", T: 97, N: 65, K: 1056, pattern: "zeros" }, { shape: "saturated", T: 97, N: 65, K: 32, pattern: "saturated" }];
  if (params.get("boundaries") === "1") for (const T of [...new Set([1, 64, 65, 96, tile, tile + 1, 112, 113, 128, 224, 241])]) inputs.push({ shape: "gate6144", T, N: 6144, K: 1536 }, { shape: "ragged", T, N: 65, K: 1056 });
  inputs = inputs.filter(({ shape, T }) => (!params.has("shape") || shape === params.get("shape")) && (!params.has("T") || T === Number(params.get("T"))));
  if (!inputs.length) throw new Error("no matching suffix tall case; shape=gate6144/gate12288/ragged/cancellation/zeros/saturated; optional boundaries=1");
  metadata = { f16: true, tile, candidateEligibility: { minRows: 97, maxRows: tile }, requiredFeatures: ["shader-f16"] };
  const runtime = await requestDevice({ features: ["shader-f16", "timestamp-query"] });
  try {
    metadata.adapter = { name: runtime.name, features: [...runtime.device.features] };
    if (!runtime.device.features.has("shader-f16")) throw new Error("suffix tall requires shader-f16; F32 fallback is disabled");
    if (runtime.device.limits.maxComputeWorkgroupStorageSize < tile * 64 + 8192) throw new Error(`suffix tall requires ${tile * 64 + 8192} bytes of workgroup storage`);
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata = { ...metadata, wasmUrl: source.wasmUrl, controlledTemporaryLimitBytes: LIMIT, immutableWeightsExcludedFromLimit: true,
      baselineKernel: baselineMode === "tall" ? "matmul_gelu_tall" : "matmul_gelu",
      candidateKernel: baselineMode === "tall" || swizzle === "half-bank" ? `dev BM${tile} ${swizzle} transform of matmul_gelu` : registered ? "matmul_gelu_tall" : `dev BM${tile} transform of matmul_gelu`,
      baselinePaired: true, registered };
    if (baselineMode !== "current" || swizzle !== "default") Object.assign(metadata, { baselineMode, baselineBM: baselineMode === "tall" ? 104 : 64, weightSwizzle: swizzle });
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source, tile, { baselineMode, swizzle, registered }), "building suffix tall fusion");
    metadata.sourceHashes = pipes.sourceHashes; metadata.sourceSpec = pipes.sourceSpec;
    for (const input of inputs) {
      log({ status: "checking", ...input, f16: true, tile, memory: memoryPlan(input.T, input.N, input.K, tile) });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, input), `suffix tall ${input.shape} T${input.T}`);
      cases.push(result); window.gpuBench.completed = cases.length; log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const executed = cases.filter((value) => !value.skipped), timed = executed.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-suffix-tall", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMs), 0) / timed.length) : performance.now() - started,
      method: timed.length ? pipes.baselineTile === 64 && pipes.swizzle === "default"
        ? "geometric mean candidate GPU timestamp milliseconds; current paired F16 R4/BN64 baseline"
        : `geometric mean candidate GPU timestamp milliseconds; ${pipes.baselineKernel} paired F16 BM104 baseline`
        : "untimed guard wall milliseconds", elapsedMs: performance.now() - started,
      skipped: executed.length === 0, correctness: { ok: true, bitExact: executed.length > 0, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-suffix-tall", completed: 0 };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
    correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
