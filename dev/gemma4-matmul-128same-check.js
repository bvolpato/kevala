import { matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024, LIMIT = 64 * MiB, CHUNK = MiB, T = 97, SENTINEL = 12345.25;
const SHAPES = [{ shape: "gate", N: 6144, K: 1536 }, { shape: "down", N: 1536, K: 6144 },
  { shape: "q", N: 2048, K: 1536 }, { shape: "o", N: 1536, K: 2048 }, { shape: "ragged", N: 65, K: 1056 }];
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

const Q8_LOADER = `
    for (var band = 0u; band < 2u; band++) {
      let r = lr + 32u * band;
      let xrowCandidate = m0 + r;
      var x0 = vec4<f32>(0.0);
      var x1 = vec4<f32>(0.0);
      if (r < 56u && xrowCandidate < T) {
        let base = (xrowCandidate * K + kb * 32u) / 4u + lq * 2u;
        x0 = X[base];
        x1 = X[base + 1u];
      }
      if (r < 56u) {
        xs[r * 8u + lq * 2u] = vec4<f32>(x0);
        xs[r * 8u + lq * 2u + 1u] = vec4<f32>(x1);
      }
      let wrow = n0 + r;
      wv[2u * band] = vec4<f32>(0.0);
      wv[2u * band + 1u] = vec4<f32>(0.0);
      if (wrow < N) {
        let v = W[(wrow * K + kb * 32u) / 16u + lq / 2u];
        let s = S[wrow * nb + kb];
        let lo = select(v.xy, v.zw, (lq & 1u) == 1u);
        wv[2u * band] = sx(lo.x) * s;
        wv[2u * band + 1u] = sx(lo.y) * s;
      }
      ws[r * 8u + ((lq * 2u) ^ sw)] = vec4<f32>(wv[2u * band]);
      ws[r * 8u + ((lq * 2u + 1u) ^ sw)] = vec4<f32>(wv[2u * band + 1u]);
    }
    workgroupBarrier();`;

export function wg128SameCode(original) {
  if (original.includes("enable f16;") || (original.match(/var acc\d = vec2<f32>\(0\.0\);/g) || []).length !== 7) throw new Error("WG128/BM56 requires the production F32 R4 BM56 kernel");
  let code = original;
  const replaceOnce = (from, to) => {
    if (code.split(from).length - 1 !== 1) throw new Error("WG128/BM56 transform marker mismatch: " + from);
    code = code.replace(from, to);
  };
  replaceOnce("@workgroup_size(32, 8)", "@workgroup_size(32, 4)");
  replaceOnce("var wv: array<vec4<f32>, 2>;", "var wv: array<vec4<f32>, 4>;");

  const declarations = [...code.matchAll(/  var acc\d+ = vec2<f32>\(0\.0\);/g)];
  const declStart = declarations[0]?.index ?? -1;
  const lastDecl = declarations.at(-1);
  if (declStart < 0 || declarations.length !== 7 || !lastDecl) throw new Error("WG128/BM56 accumulator declarations changed");
  const declEnd = lastDecl.index + lastDecl[0].length;
  code = code.slice(0, declStart) + Array.from({ length: 14 }, (_, i) => "  var acc" + i + " = vec2<f32>(0.0);").join("\n") + code.slice(declEnd);

  const loop = code.indexOf("  for (var kb = kb0; kb < kb1; kb++) {");
  const loopEnd = code.indexOf("    workgroupBarrier();", loop);
  if (loop < 0 || loopEnd < 0) throw new Error("WG128/BM56 production K loop was not found");
  const loadStart = code.indexOf("\n", loop) + 1;
  const loadEnd = loopEnd + "    workgroupBarrier();".length;
  if (loadStart <= loop || loadEnd <= loadStart) throw new Error("WG128/BM56 production loader bounds are invalid");
  code = code.slice(0, loadStart) + Q8_LOADER + code.slice(loadEnd);

  const dotStart = code.indexOf("    {\n      let kq = 0u;");
  const dotEnd = code.indexOf("    workgroupBarrier();", dotStart);
  if (dotStart < 0 || dotEnd < 0) throw new Error("WG128/BM56 explicit K-dot region was not found");
  const dotRegion = code.slice(dotStart, dotEnd);
  const blockPattern = /    \{\n      let kq = (\d+)u;\n[\s\S]*?\n    \}/g;
  let blocks = 0;
  const expandedDots = dotRegion.replace(blockPattern, (block, kq) => {
    if (Number(kq) !== blocks++) throw new Error("WG128/BM56 K-dot block order changed");
    const pairPattern = /      let a(\d+) = vec4<f32>\(xs\[\(lid\.y \+ (\d+)u\) \* 8u \+ kq\]\);\n      acc\1 \+= vec2<f32>\(dot\(a\1, b0\), dot\(a\1, b1\)\);/g;
    const pairs = [...block.matchAll(pairPattern)];
    if (pairs.length !== 7 || pairs.some((pair, i) => Number(pair[1]) !== i || Number(pair[2]) !== i * 8)) throw new Error("WG128/BM56 source row/accumulator order changed");
    const begin = pairs[0].index;
    const end = pairs.at(-1).index + pairs.at(-1)[0].length;
    const replacement = Array.from({ length: 14 }, (_, i) =>
      "      let a" + i + " = vec4<f32>(xs[(lid.y + " + i * 4 + "u) * 8u + kq]);\n" +
      "      acc" + i + " += vec2<f32>(dot(a" + i + ", b0), dot(a" + i + ", b1));").join("\n");
    return block.slice(0, begin) + replacement + block.slice(end);
  });
  if (blocks !== 8) throw new Error("WG128/BM56 expected the eight original ascending K/4 blocks");
  code = code.slice(0, dotStart) + expandedDots + code.slice(dotEnd);

  for (const component of ["x", "y"]) {
    const first = "      mm_store(acc0." + component + ", row + 0u, col, bias, T, N, splits, wg.z);";
    const last = "      mm_store(acc6." + component + ", row + 48u, col, bias, T, N, splits, wg.z);";
    const begin = code.indexOf(first), lastStart = code.indexOf(last, begin);
    if (begin < 0 || lastStart < begin) throw new Error("WG128/BM56 output store group changed: " + component);
    const end = lastStart + last.length;
    const region = code.slice(begin, end);
    if ((region.match(/mm_store\(acc\d+\./g) || []).length !== 7) throw new Error("WG128/BM56 expected seven baseline output stores");
    const stores = Array.from({ length: 14 }, (_, i) =>
      "      mm_store(acc" + i + "." + component + ", row + " + i * 4 + "u, col, bias, T, N, splits, wg.z);").join("\n");
    code = code.slice(0, begin) + stores + code.slice(end);
  }

  const dots = (text) => [...text.matchAll(/acc\d+ \+= vec2<f32>\(dot\(a\d+, b0\), dot\(a\d+, b1\)\);/g)].map((match) => match[0]);
  if (dots(code).length !== 112) throw new Error("WG128/BM56 changed the 8-by-14 K-dot accumulation count");
  const split = (text) => text.match(/fn mm_splits\([\s\S]*?\n\}/)?.[0];
  if (!split(original) || split(code) !== split(original)) throw new Error("WG128/BM56 changed the production BM64/BN64 split rule");
  for (const name of ["sx", "mm_store"]) {
    const body = (text) => text.match(new RegExp("fn " + name + "\\([\\s\\S]*?\\n\\}"))?.[0];
    if (body(original) !== body(code)) throw new Error("WG128/BM56 changed production " + name + " semantics");
  }
  return code;
}

export function wg128SameProof() {
  const xs = new Set(), ws = new Set(), outputs = new Set();
  for (let li = 0; li < 128; li++) {
    const lr = Math.floor(li / 4), lq = li % 4;
    for (const row of [lr, lr + 32]) {
      if (row < 56) for (let pair = 0; pair < 2; pair++) xs.add(row * 8 + lq * 2 + pair);
      for (let pair = 0; pair < 2; pair++) ws.add(row * 8 + ((lq * 2 + pair) ^ (row & 7)));
    }
    const x = li % 32, y = Math.floor(li / 32);
    for (let i = 0; i < 14; i++) for (const col of [x, x + 32]) outputs.add((y + 4 * i) * 64 + col);
  }
  if (xs.size !== 56 * 8 || Math.max(...xs) !== 56 * 8 - 1 || ws.size !== 64 * 8 || Math.max(...ws) !== 64 * 8 - 1 || outputs.size !== 56 * 64 || Math.max(...outputs) !== 56 * 64 - 1) throw new Error("WG128/BM56 tile misses or aliases shared/input/output slots");
  return { workgroup: [32, 4], threads: 128, physicalBM: 56, virtualBM: 64, BN: 64, vectorInputSlots: xs.size, vectorWeightSlots: ws.size, outputSlots: outputs.size, eachTileOutputCoveredOnce: true, kqOrder: [0, 1, 2, 3, 4, 5, 6, 7], vec2AccumulatorsPerThread: 14, scalarAccumulatorsPerThread: 28, sharedBytes: 16384 };
}

export function temporaryPlan(N, K) {
  const splits = mmSplits(1, N, K, 256, 64, 64), length = T * N;
  const partialLength = splits > 1 ? splits * length : 0;
  const outputBytes = (length + 4) * 4, partBytes = (partialLength + 4) * 4;
  const gpuBytes = T * K * 4 + 2 * outputBytes + 2 * partBytes + N * 4 + 2 * CHUNK + 32 + 256;
  return { splits, length, partialLength, outputBytes, partBytes, gpuBytes,
    hostStageBytes: CHUNK, mappedReadBytes: 2 * CHUNK, uploadAllowanceBytes: CHUNK,
    temporaryBytes: gpuBytes + 4 * CHUNK + 128, limitBytes: LIMIT, immutableWeightBytes: N * K * 9 / 8 };
}

async function compile(device, source) {
  const spec = { weights: "q8", f16: false, rows: 4, row56: true, groups: 1, splitTarget: 256 };
  const baselineCode = source("matmul_gemma_batch", spec), reduceCode = source("reduce_gemma_batch", spec);
  const candidateCode = wg128SameCode(baselineCode);
  const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  return { layout, reductionLayout,
    baseline: await pipeline(device, baselineCode, "matmul-wg256-bm56", pl),
    candidate: await pipeline(device, candidateCode, "matmul-wg128-bm56", pl),
    reduce: await pipeline(device, reduceCode, "matmul-bm56-reduce", device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] })) };
}

async function check(runtime, pipes, shape) {
  const { N, K } = shape, { device } = runtime, U = GPUBufferUsage, plan = temporaryPlan(N, K);
  if (plan.temporaryBytes > LIMIT) throw new Error("WG128 temporary memory exceeds 64 MiB");
  if (Math.max(N * K, plan.outputBytes, plan.partBytes, T * K * 4, 2 * CHUNK) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("WG128 buffer exceeds device limits");
  const owned = [], deadline = performance.now() + 30_000;
  let gpuTemporaryBytes = 0, query;
  const make = (size, usage, label, immutable = false) => {
    if (!immutable) gpuTemporaryBytes += size;
    if (gpuTemporaryBytes > plan.gpuBytes) throw new Error("WG128 exceeded planned temporary GPU allocation");
    const buffer = device.createBuffer({ size, usage, label: `wg128.${label}` }); owned.push(buffer); return buffer;
  };
  const map = async (read, size) => {
    if (!await mapBeforeDeadline(read, size, deadline)) { read.unmap(); throw new Error("WG128 readback deadline exceeded"); }
  };
  const drain = async () => {
    const remaining = deadline - performance.now();
    if (remaining <= 0) throw new Error("WG128 upload deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("WG128 upload deadline exceeded")), remaining); })]); }
    finally { clearTimeout(timer); }
  };
  try {
    const stage = new Float32Array(CHUNK / 4), byteStage = new Int8Array(stage.buffer);
    const generated = async (buffer, size, bytes, valueAt) => {
      for (let offset = 0; offset < size; offset += CHUNK) {
        const count = Math.min(CHUNK, size - offset), target = bytes === 1 ? byteStage : stage;
        for (let i = 0; i < count / bytes; i++) target[i] = valueAt(offset / bytes + i);
        device.queue.writeBuffer(buffer, offset, stage.buffer, 0, count); await drain();
      }
    };
    const w = make(N * K, U.STORAGE | U.COPY_DST, "W", true), s = make(N * K / 8, U.STORAGE | U.COPY_DST, "S", true);
    const x = make(T * K * 4, U.STORAGE | U.COPY_DST, "X");
    await generated(w, N * K, 1, (i) => (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) % 256 - 128);
    await generated(s, N * K / 8, 4, (i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
    await generated(x, T * K * 4, 4, (i) => i % 67 === 0 ? (i % 2 ? -0 : 0) : 0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals"), params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const bias = make(N * 4, U.STORAGE | U.COPY_DST, "B");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, 1, 1, 0]));
    const outputs = [0, 1].map(() => make(plan.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "Y"));
    const parts = [0, 1].map(() => make(plan.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "PART"));
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-read"), resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const ops = outputs.map((output, index) => ({ group: group(pipes.layout, [globals, params, x, w, s, bias, output, parts[index]]), reduce: group(pipes.reductionLayout, [globals, params, parts[index], bias, output]) }));
    const dispatch = (pass, candidate) => {
      const index = Number(candidate);
      pass.setPipeline(candidate ? pipes.candidate : pipes.baseline); pass.setBindGroup(0, ops[index].group);
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 56), plan.splits);
      if (plan.splits > 1) {
        const work = Math.ceil(plan.length / 256);
        pass.setPipeline(pipes.reduce); pass.setBindGroup(0, ops[index].reduce); pass.dispatchWorkgroups(Math.min(work, 65535), Math.ceil(work / 65535));
      }
    };
    const initialize = async (buffer, length, mode = 0) => {
      await generated(buffer, length * 4, 4, (i) => mode === 1 ? Math.sin(i * 0.13) * 0.03 : NaN);
      device.queue.writeBuffer(buffer, length * 4, new Float32Array(4).fill(SENTINEL));
    };
    const compare = async (buffers, length) => {
      let zeros = 0, negativeZeros = 0;
      for (let offset = 0; offset < length + 4; offset += CHUNK / 4) {
        const count = Math.min(CHUNK / 4, length + 4 - offset), bytes = count * 4, encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffers[0], offset * 4, read, 0, bytes); encoder.copyBufferToBuffer(buffers[1], offset * 4, read, bytes, bytes); device.queue.submit([encoder.finish()]);
        await map(read, bytes * 2);
        try {
          const range = read.getMappedRange(0, bytes * 2), values = new Float32Array(range), bits = new Uint32Array(range);
          for (let i = 0; i < count; i++) {
            const at = offset + i;
            if (at >= length) { if (values[i] !== SENTINEL || values[count + i] !== SENTINEL) throw new Error(`WG128 tail overwrite at ${at}`); }
            else {
              if (!Number.isFinite(values[i]) || !Number.isFinite(values[count + i])) throw new Error(`WG128 nonfinite output/partial at ${at}`);
              if (bits[i] !== bits[count + i]) throw new Error(`WG128 bit mismatch at ${at}: 0x${bits[i].toString(16)} vs 0x${bits[count + i].toString(16)}`);
              if (values[i] === 0) zeros++;
              if (bits[i] === 0x80000000) negativeZeros++;
            }
          }
        } finally { read.unmap(); }
      }
      return { samples: length, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros, tailGuard: true, signedZeroChecked: true };
    };
    const runGuards = async () => {
      const guards = [];
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
        device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias]));
        await generated(bias, N * 4, 4, (i) => hasBias ? (i % 7 - 3) * 0.0137 : NaN);
        for (let index = 0; index < 2; index++) { await initialize(outputs[index], plan.length, mode); await initialize(parts[index], plan.partialLength); }
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); device.queue.submit([encoder.finish()]);
        guards.push({ mode, bias: hasBias, output: await compare(outputs, plan.length), partial: await compare(parts, plan.partialLength) });
      }
      return guards;
    };
    const before = await runGuards();
    let timing = null;
    if (device.features.has("timestamp-query")) {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
      query = device.createQuerySet({ type: "timestamp", count: 12 });
      const encoder = device.createCommandEncoder(), warm = encoder.beginComputePass(), order = [];
      dispatch(warm, false); dispatch(warm, true); warm.end();
      for (let pair = 0; pair < 3; pair++) for (const candidate of pair % 2 ? [true, false] : [false, true]) {
        const at = order.length * 2; order.push(candidate);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: at, endOfPassWriteIndex: at + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, candidate); pass.end();
      }
      encoder.resolveQuerySet(query, 0, 12, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 96); device.queue.submit([encoder.finish()]);
      await map(read, 96);
      try {
        const stamps = new BigUint64Array(read.getMappedRange(0, 96)), samples = { baseline: [], candidate: [] };
        order.forEach((candidate, i) => samples[candidate ? "candidate" : "baseline"].push(Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6 / 4));
        const baselineMs = median(samples.baseline), candidateMs = median(samples.candidate);
        if (![baselineMs, candidateMs].every((value) => Number.isFinite(value) && value > 0)) throw new Error("invalid WG128 timestamps");
        timing = { samples: 3, repetitions: 4, ...samples, baselineMs, candidateMs, speedup: baselineMs / candidateMs };
      } finally { read.unmap(); }
    }
    const after = await runGuards();
    return { ...shape, T, splitT: 1, f16: false, ok: true, guards: { before, after }, timing, memory: { ...plan, actualGpuTemporaryBytes: gpuTemporaryBytes },
      geometry: { baseline: { workgroup: [32, 8], physicalBM: 56, virtualBM: 64, BN: 64, sharedBytes: 16384, vec2AccumulatorsPerThread: 7 }, candidate: { workgroup: [32, 4], physicalBM: 56, virtualBM: 64, BN: 64, sharedBytes: 16384, vec2AccumulatorsPerThread: 14 }, paddedRows: 56 } };
  } finally { query?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  const shapes = SHAPES.filter(({ shape }) => !params.has("shape") || shape === params.get("shape"));
  if (!shapes.length) throw new Error("shape must be gate, down, q, o, or ragged");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata = { adapter: { name: runtime.name, features: [...runtime.device.features] }, wasmUrl: source.wasmUrl, temporaryLimitBytes: LIMIT, coverage: wg128SameProof(), immutableWeightsExcludedFromTemporaryLimit: true };
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source), "building WG128/BM56 probe");
    for (const shape of shapes) {
      log({ status: "checking", ...shape, memory: temporaryPlan(shape.N, shape.K) });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, shape), "WG128/BM56 " + shape.shape);
      cases.push(result); window.gpuBench.completed = cases.length; log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matmul-128same", ...metadata,
      metricMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matmul-128same", completed: 0 };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
    correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
