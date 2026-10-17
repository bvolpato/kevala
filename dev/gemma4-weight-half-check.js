import { dispatchMatmul, matmulLayout, mmSplits, pipeline, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { kernelSource } from "./wgsl.js";

const MiB = 1024 * 1024;
const LIMIT = 160 * MiB;
const ALLOWANCE = 24 * MiB;
const CHUNK = MiB;
const SENTINEL = 12345.25;
const HALF_TAIL = 0x3c003c00;
const CASE_MS = 30_000;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

export function halfWeightSource(q8) {
  const declaration = "var<storage, read> W: array<vec4<u32>>;";
  const halfDeclaration = "var<storage, read> W: array<vec4<f16>>;";
  const load = /let v = W\[\(wrow \* K \+ kb \* 32u\) \/ 16u \+ lq \/ 2u\];\s*let s = S\[wrow \* nb \+ kb\];\s*let lo = select\(v\.xy, v\.zw, \(lq & 1u\) == 1u\);\s*wv\[2u \* gg\] = sx\(lo\.x\) \* s;\s*wv\[2u \* gg \+ 1u\] = sx\(lo\.y\) \* s;/g;
  const matches = [...q8.matchAll(load)];
  if (!q8.includes("enable f16;") || matches.length !== 1 || q8.split(declaration).length !== 2) throw new Error("unexpected production Q8/F16 weight loader");
  const halfLoad = "let half_base = (wrow * K + kb * 32u) / 4u + lq * 2u;\n        wv[2u * gg] = vec4<f32>(W[half_base]);\n        wv[2u * gg + 1u] = vec4<f32>(W[half_base + 1u]);";
  const candidate = q8.replace(declaration, halfDeclaration).replace(matches[0][0], halfLoad);
  if (candidate.replace(halfDeclaration, declaration).replace(halfLoad, matches[0][0]) !== q8) throw new Error("half-weight transform changed arithmetic outside its loader");
  return candidate;
}

export function memoryPlan(T, N, K, splitT) {
  const splits = mmSplits(splitT, N, K, 256, 64, 64);
  const weightBytes = N * K, scaleBytes = weightBytes / 8, halfBytes = weightBytes * 2 + 32;
  const inputBytes = T * K * 4, outputLength = T * N;
  const outputBytes = (outputLength + 4) * 4;
  const partLength = splits > 1 ? splits * outputLength : 0;
  const partBytes = partLength ? (partLength + 4) * 4 : 16;
  const gpuBytes = weightBytes + scaleBytes + halfBytes + inputBytes + 2 * outputBytes + 2 * partBytes + 2 * CHUNK + N * 4 + 3 * 16 + 256;
  const hostFixtureBytes = 0;
  const controlledBytes = gpuBytes + hostFixtureBytes + CHUNK + 2 * CHUNK + CHUNK + 128;
  return { splits, weightBytes, scaleBytes, halfBytes, inputBytes, outputLength, outputBytes, partLength, partBytes,
    gpuBytes, hostFixtureBytes, hostStageBytes: CHUNK, mappedReadBytes: 2 * CHUNK, queuedUploadAllowanceBytes: CHUNK,
    controlledBytes, allowanceBytes: ALLOWANCE, estimatedTotalBytes: controlledBytes + ALLOWANCE, limitBytes: LIMIT };
}

function decoderSource(q8) {
  const sx = q8.match(/fn sx\(w: u32\) -> vec4<f32> \{[\s\S]*?\n\}/)?.[0];
  if (!sx) throw new Error("production sx function is missing");
  return `enable f16;
struct P { count: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> H: array<vec4<f16>>;
${sx}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let at = gid.x;
  if (at >= p.count) { return; }
  let packed = W[at / 4u];
  let decoded = sx(packed[at & 3u]) * S[at / 8u];
  H[at] = vec4<f16>(decoded);
}`;
}

async function pipelines(device, source, T) {
  const batched = T === 97;
  const spec = { f16: true, weights: "q8", rows: 4, row56: false, groups: 1, splitTarget: 256 };
  const q8 = source(batched ? "matmul_gemma_batch" : "matmul", spec);
  const half = halfWeightSource(q8);
  const reduction = source(batched ? "reduce_gemma_batch" : "reduce", spec);
  const layout = matmulLayout(device), reductionLayout = reduceLayout(device);
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const rpl = device.createPipelineLayout({ bindGroupLayouts: [reductionLayout] });
  return { layout, reductionLayout,
    q8: await pipeline(device, q8, "half-cache.q8", pl),
    half: await pipeline(device, half, "half-cache.f16", pl),
    reduce: await pipeline(device, reduction, "half-cache.reduce", rpl),
    decode: await pipeline(device, decoderSource(q8), "half-cache.decode"),
    policy: { rows: 4, groups: 1, splitTarget: 256, T, splitT: batched ? 1 : T, kernel: batched ? "matmul_gemma_batch" : "matmul", onlyWeightLoaderChanged: true } };
}

async function check(runtime, pipes, shape) {
  const { T, N, K } = shape, { device } = runtime, U = GPUBufferUsage;
  const plan = memoryPlan(T, N, K, pipes.policy.splitT);
  if (plan.estimatedTotalBytes > LIMIT) throw new Error(`half-weight probe needs ${plan.estimatedTotalBytes} bytes, limit ${LIMIT}`);
  const largest = Math.max(plan.halfBytes, plan.partBytes, plan.outputBytes, plan.inputBytes, 2 * CHUNK);
  if (largest > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("half-weight probe exceeds device buffer limits");
  const owned = [], deadline = performance.now() + CASE_MS;
  let querySet, gpuBytes = 0;
  const make = (size, usage, label) => {
    gpuBytes += size;
    if (gpuBytes > plan.gpuBytes) throw new Error("half-weight probe exceeded planned GPU allocation");
    const buffer = device.createBuffer({ size, usage, label: `weight-half.${label}` }); owned.push(buffer); return buffer;
  };
  const map = async (read, bytes) => {
    if (!await mapBeforeDeadline(read, bytes, deadline)) { read.unmap(); throw new Error("half-weight probe deadline exceeded"); }
  };
  const drain = async () => {
    if (performance.now() >= deadline) throw new Error("half-weight probe deadline exceeded");
    let timer;
    try { await Promise.race([device.queue.onSubmittedWorkDone(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("half-weight probe deadline exceeded")), Math.max(1, deadline - performance.now())); })]); }
    finally { clearTimeout(timer); }
  };
  try {
    const stage = new Float32Array(CHUNK / 4), byteStage = new Uint8Array(stage.buffer);
    const upload = async (buffer, bytes, byteValues, valueAt) => {
      for (let offset = 0; offset < bytes; offset += CHUNK) {
        const count = Math.min(CHUNK, bytes - offset), values = byteValues ? byteStage : stage;
        const stride = byteValues ? 1 : 4;
        for (let i = 0; i < count / stride; i++) values[i] = valueAt(offset / stride + i);
        device.queue.writeBuffer(buffer, offset, stage.buffer, 0, count); await drain();
      }
    };
    const w = make(plan.weightBytes, U.STORAGE | U.COPY_DST, "q8");
    const s = make(plan.scaleBytes, U.STORAGE | U.COPY_DST, "scales");
    const h = make(plan.halfBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "f16");
    const x = make(plan.inputBytes, U.STORAGE | U.COPY_DST, "X");
    await upload(w, plan.weightBytes, true, (i) => (i * 31 + (i >>> 5) * 17 + (i >>> 10) * 7 + 19) & 255);
    await upload(s, plan.scaleBytes, false, (i) => i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 11 * 0.000037));
    await upload(x, plan.inputBytes, false, (i) => i % 67 === 0 ? (i % 2 ? -0 : 0) : 0.04 * Math.sin(i * 0.127 + Math.floor(i / K) * 0.13 + 0.3));
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals");
    const params = make(16, U.UNIFORM | U.COPY_DST, "params");
    const decodeParams = make(16, U.UNIFORM | U.COPY_DST, "decode-params");
    const bias = make(N * 4, U.STORAGE | U.COPY_DST, "bias");
    const outputs = [0, 1].map(() => make(plan.outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "Y"));
    const parts = [0, 1].map(() => make(plan.partBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, "PART"));
    const read = make(2 * CHUNK, U.MAP_READ | U.COPY_DST, "staged-read");
    const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamp-resolve");
    device.queue.writeBuffer(globals, 0, new Uint32Array([T, T, T, 0]));
    device.queue.writeBuffer(decodeParams, 0, new Uint32Array([N * K / 4, 0, 0, 0]));
    device.queue.writeBuffer(h, N * K * 2, new Uint32Array(8).fill(HALF_TAIL));
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const decodeGroup = group(pipes.decode.getBindGroupLayout(0), [decodeParams, w, s, h]);
    const decodeEncoder = device.createCommandEncoder(), decodePass = decodeEncoder.beginComputePass();
    decodePass.setPipeline(pipes.decode); decodePass.setBindGroup(0, decodeGroup); decodePass.dispatchWorkgroups(Math.ceil(N * K / (4 * 256))); decodePass.end();
    decodeEncoder.copyBufferToBuffer(h, N * K * 2, read, 0, 32);
    const decodeStarted = performance.now(); device.queue.submit([decodeEncoder.finish()]);
    await map(read, 32);
    const decodeWallMs = performance.now() - decodeStarted;
    try { if (new Uint32Array(read.getMappedRange(0, 32)).some((value) => value !== HALF_TAIL)) throw new Error("decoded half-weight tail overwrite"); }
    finally { read.unmap(); }
    const ops = [w, h].map((weight, index) => ({ N, K,
      group: group(pipes.layout, [globals, params, x, weight, s, bias, outputs[index], parts[index]]),
      reduce: group(pipes.reductionLayout, [globals, params, parts[index], bias, outputs[index]]) }));
    const dispatch = (pass, kind) => dispatchMatmul(pass, kind ? pipes.half : pipes.q8, pipes.reduce, ops[kind], T, 256, 4, 1, T, pipes.policy.splitT);
    const initialize = async (buffer, length, mode = 0) => {
      for (let offset = 0; offset < length; offset += stage.length) {
        const count = Math.min(stage.length, length - offset);
        for (let i = 0; i < count; i++) stage[i] = mode === 1 ? Math.sin((offset + i) * 0.13) * 0.03 : NaN;
        device.queue.writeBuffer(buffer, offset * 4, stage.buffer, 0, count * 4); await drain();
      }
      device.queue.writeBuffer(buffer, length * 4, new Float32Array(4).fill(SENTINEL));
    };
    const compare = async (buffers, length) => {
      let zeros = 0, negativeZeros = 0;
      for (let offset = 0; offset < length + 4; offset += CHUNK / 4) {
        const count = Math.min(CHUNK / 4, length + 4 - offset), bytes = count * 4;
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffers[0], offset * 4, read, 0, bytes);
        encoder.copyBufferToBuffer(buffers[1], offset * 4, read, bytes, bytes); device.queue.submit([encoder.finish()]);
        await map(read, bytes * 2);
        try {
          const range = read.getMappedRange(0, bytes * 2), values = new Float32Array(range), bits = new Uint32Array(range);
          for (let i = 0; i < count; i++) {
            const at = offset + i;
            if (at >= length) { if (values[i] !== SENTINEL || values[count + i] !== SENTINEL) throw new Error(`tail overwrite at ${at}`); }
            else {
              if (!Number.isFinite(values[i]) || !Number.isFinite(values[count + i])) throw new Error(`nonfinite output or partial at ${at}`);
              if (bits[i] !== bits[count + i]) throw new Error(`F16 cached-weight bit mismatch at ${at}: 0x${bits[i].toString(16)} vs 0x${bits[count + i].toString(16)}`);
              if (values[i] === 0) zeros++;
              if (bits[i] === 0x80000000) negativeZeros++;
            }
          }
        } finally { read.unmap(); }
      }
      return { samples: length, bitMismatches: 0, nonfinite: 0, zeros, negativeZeros, tailGuard: true, signedZeroBitsChecked: true };
    };
    const guards = [];
    for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias]));
      for (let n = 0; n < N; n++) stage[n] = hasBias ? (n % 7 - 3) * 0.0137 : NaN;
      device.queue.writeBuffer(bias, 0, stage.buffer, 0, N * 4);
      for (let kind = 0; kind < 2; kind++) {
        await initialize(outputs[kind], plan.outputLength, mode);
        if (plan.partLength) await initialize(parts[kind], plan.partLength);
        else device.queue.writeBuffer(parts[kind], 0, new Float32Array(4).fill(SENTINEL));
      }
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      dispatch(pass, 0); dispatch(pass, 1); pass.end(); device.queue.submit([encoder.finish()]);
      const output = await compare(outputs, plan.outputLength);
      const partial = plan.partLength ? await compare(parts, plan.partLength) : await compare(parts, 0);
      guards.push({ mode, bias: hasBias, output, partial });
    }
    let timing = null;
    if (device.features.has("timestamp-query")) {
      device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0]));
      querySet = device.createQuerySet({ type: "timestamp", count: 14 });
      const encoder = device.createCommandEncoder(), order = [];
      const warm = encoder.beginComputePass(); for (const kind of [0, 1]) for (let repeat = 0; repeat < 4; repeat++) dispatch(warm, kind); warm.end();
      const decodePass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 12, endOfPassWriteIndex: 13 } });
      decodePass.setPipeline(pipes.decode); decodePass.setBindGroup(0, decodeGroup); decodePass.dispatchWorkgroups(Math.ceil(N * K / (4 * 256))); decodePass.end();
      for (let sample = 0; sample < 3; sample++) for (const kind of sample % 2 ? [1, 0] : [0, 1]) {
        const index = order.length * 2; order.push(kind);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let repeat = 0; repeat < 4; repeat++) dispatch(pass, kind); pass.end();
      }
      encoder.resolveQuerySet(querySet, 0, 14, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, 112); device.queue.submit([encoder.finish()]);
      await map(read, 112);
      try {
        const stamps = new BigUint64Array(read.getMappedRange(0, 112)), samples = { q8: [], half: [] };
        order.forEach((kind, index) => samples[kind ? "half" : "q8"].push(Number(stamps[index * 2 + 1] - stamps[index * 2]) / 1e6 / 4));
        const q8Ms = median(samples.q8), halfMs = median(samples.half), decodeGpuMs = Number(stamps[13] - stamps[12]) / 1e6;
        if (![q8Ms, halfMs, decodeGpuMs].every((value) => Number.isFinite(value) && value > 0)) throw new Error("invalid half-weight timestamps");
        timing = { samples: 3, repetitions: 4, ...samples, q8Ms, halfMs, speedup: q8Ms / halfMs, decodeGpuMs,
          breakEvenInvocations: q8Ms > halfMs ? Math.ceil(decodeGpuMs / (q8Ms - halfMs)) : null };
      } finally { read.unmap(); }
    }
    return { ...shape, ...pipes.policy, splits: plan.splits, ok: true, guards, timing,
      expansion: { decodeWallMs, halfWeightBytes: N * K * 2, sourceWeightBytes: plan.weightBytes + plan.scaleBytes, decodedTailGuard: true },
      memory: { ...plan, actualGpuBytes: gpuBytes, readbackChunkBytesPerSide: CHUNK } };
  } finally { querySet?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  const shapes = [{ shape: "gate6144", N: 6144, K: 1536 }, { shape: "down6144", N: 1536, K: 6144 },
    { shape: "gate12288", N: 12288, K: 1536 }, { shape: "down12288", N: 1536, K: 12288 }];
  const selected = params.has("shape") ? shapes.filter(({ shape }) => shape === params.get("shape")) : params.has("all") ? shapes : shapes.slice(0, 2);
  const tokens = params.has("tokens") ? [Number(params.get("tokens"))] : [97, 89];
  if (!selected.length || tokens.some((T) => ![89, 97].includes(T))) throw new Error("shape must be gate6144/down6144/gate12288/down12288; tokens must be 89 or 97");
  const runtime = await requestDevice({ features: ["shader-f16", "timestamp-query"] });
  try {
    metadata = { adapter: { name: runtime.name, features: [...runtime.device.features] }, limitBytes: LIMIT };
    if (!runtime.device.features.has("shader-f16")) {
      window.gpuBench = { status: "skipped", done: true, backend: "webgpu", ...metadata, reason: "shader-f16 is required; no F32 fallback was run" }; log(window.gpuBench); return;
    }
    const source = await kernelSource(params.get("wasm") || undefined);
    metadata.wasmUrl = source.wasmUrl;
    for (const T of tokens) {
      const pipes = await withGpuErrors(runtime, () => pipelines(runtime.device, source, T), `building half-weight T${T}`);
      for (const shape of selected) {
        log({ status: "checking", T, ...shape, policy: pipes.policy, memory: memoryPlan(T, shape.N, shape.K, pipes.policy.splitT) });
        const result = await withGpuErrors(runtime, () => check(runtime, pipes, { T, ...shape }), `half-weight T${T} ${shape.shape}`);
        cases.push(result); window.gpuBench.completed = cases.length; log(result);
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "gemma4-weight-half", ...metadata,
      metricMs: performance.now() - started, correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-weight-half", completed: 0 };
  main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata,
    correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) }; log(error.stack || error); });
}
