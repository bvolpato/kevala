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

export function halfWorkgroup(original, loader) {
  if (original.includes("enable f16;") || (original.match(/var acc\d = vec2<f32>\(0\.0\);/g) || []).length !== 7) throw new Error("WG128 needs the production F32 R4 BM56 kernel");
  let code = original;
  const replace = (from, to, count = 1) => {
    if (code.split(from).length - 1 !== count) throw new Error(`WG128 transform marker mismatch: ${from}`);
    code = code.replaceAll(from, to);
  };
  replace("@workgroup_size(32, 8)", "@workgroup_size(32, 4)");
  replace("var<workgroup> xs: array<vec4<f32>, 512>;", "var<workgroup> xs: array<vec4<f32>, 256>;");
  replace("var wv: array<vec4<f32>, 2>;", "var wv: array<vec4<f32>, 4>;");
  replace("let m0 = wg.y * 56u;", "let m0 = wg.y * 28u;");
  const begin = code.indexOf("    for (var gg = 0u; gg < 1u; gg++) {", code.indexOf("for (var kb = kb0;"));
  const end = code.indexOf("    workgroupBarrier();", begin);
  if (begin < 0 || end <= begin) throw new Error("WG128 production loader was not found");
  const oldLoader = code.slice(begin, end);
  if (!oldLoader.includes("sx(lo.x) * s") || !oldLoader.includes("let r = lr + 64u * gg;") || (oldLoader.match(/gg < 1u/g) || []).length !== 2) throw new Error("WG128 production loader changed");
  code = code.slice(0, begin) + loader + "\n" + code.slice(end);
  replace("if (lr < 56u && xrow < T)", "if (lr < 28u && xrow < T)");
  // Replace only row addresses. Every K-dot statement and named accumulator stays intact.
  code = code.replace(/\(lid\.y \+ (\d+)u\) \* 8u/g, (match, offset) => {
    if (Number(offset) % 8 || Number(offset) > 48) throw new Error("WG128 unexpected input row offset");
    return `(lid.y + ${Number(offset) / 2}u) * 8u`;
  });
  code = code.replace(/(mm_store\(acc\d\.[xy], row \+ )(\d+)u/g, (match, prefix, offset) => {
    if (Number(offset) % 8 || Number(offset) > 48) throw new Error("WG128 unexpected output row offset");
    return `${prefix}${Number(offset) / 2}u`;
  });
  const dots = (text) => [...text.matchAll(/acc\d \+= vec2<f32>\(dot\(a\d, b0\), dot\(a\d, b1\)\);/g)].map((match) => match[0]);
  if (JSON.stringify(dots(code)) !== JSON.stringify(dots(original)) || dots(code).length !== 56) throw new Error("WG128 changed the K-dot accumulation order");
  const split = (text) => text.match(/fn mm_splits\([\s\S]*?\n\}/)?.[0];
  if (!split(original) || split(code) !== split(original)) throw new Error("WG128 changed the production split rule");
  return code;
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
  const response = await fetch(new URL("./gemma4-matmul-128threads-check.wgsl", import.meta.url), { cache: "no-store" });
  if (!response.ok) throw new Error(`WG128 loader HTTP ${response.status}`);
  const candidateCode = halfWorkgroup(baselineCode, await response.text());
  const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  return { layout, reductionLayout,
    baseline: await pipeline(device, baselineCode, "matmul-wg256-bm56", pl),
    candidate: await pipeline(device, candidateCode, "matmul-wg128-bm28", pl),
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
      pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / (candidate ? 28 : 56)), plan.splits);
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
    const guards = [];
    for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias]));
      await generated(bias, N * 4, 4, (i) => hasBias ? (i % 7 - 3) * 0.0137 : NaN);
      for (let index = 0; index < 2; index++) { await initialize(outputs[index], plan.length, mode); await initialize(parts[index], plan.partialLength); }
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); dispatch(pass, false); dispatch(pass, true); pass.end(); device.queue.submit([encoder.finish()]);
      guards.push({ mode, bias: hasBias, output: await compare(outputs, plan.length), partial: await compare(parts, plan.partialLength) });
    }
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
    return { ...shape, T, splitT: 1, f16: false, ok: true, guards, timing, memory: { ...plan, actualGpuTemporaryBytes: gpuTemporaryBytes },
      geometry: { baseline: { workgroup: [32, 8], BM: 56, BN: 64, sharedBytes: 16384 }, candidate: { workgroup: [32, 4], BM: 28, BN: 64, sharedBytes: 12288 }, paddedRows: 112, accumulatorsPerThread: 14 } };
  } finally { query?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  const shapes = SHAPES.filter(({ shape }) => !params.has("shape") || shape === params.get("shape"));
  if (!shapes.length) throw new Error("shape must be gate, down, q, o, or ragged");
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata = { adapter: { name: runtime.name, features: [...runtime.device.features] }, wasmUrl: source.wasmUrl, temporaryLimitBytes: LIMIT };
    const pipes = await withGpuErrors(runtime, () => compile(runtime.device, source), "building WG128/BM28 probe");
    for (const shape of shapes) {
      log({ status: "checking", ...shape, memory: temporaryPlan(shape.N, shape.K) });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, shape), `WG128/BM28 ${shape.shape}`);
      cases.push(result); window.gpuBench.completed = cases.length; log(result);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-matmul-128threads", ...metadata,
      metricMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-matmul-128threads", completed: 0 };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
    correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
