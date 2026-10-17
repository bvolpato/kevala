import { encodeMatvec, matvecPipelines, pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { calibrateMatvecGelu, encodeMatvecGelu, matvecGeluBindGroup, matvecGeluPipelines } from "../js/src/gpu-matvec-gelu.js";
import { kernelSource } from "./wgsl.js";

const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const SENTINEL = 12345.25;
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];

async function check(runtime, source, N, K, T, saturated = false) {
  const { device } = runtime, U = GPUBufferUsage, owned = [];
  const make = (size, usage, data) => {
    const value = device.createBuffer({ size: Math.max(16, size), usage });
    owned.push(value); if (data) device.queue.writeBuffer(value, 0, data); return value;
  };
  let profiler;
  try {
    const baseline = await matvecPipelines(device, source);
    const candidate = await matvecGeluPipelines(device, source, { f16: baseline.f16 });
    const gelu = await pipeline(device, source("gemma4_gelu"), "gemma4_gelu");
    const data = (offset) => {
      const values = Int8Array.from({ length: N * K }, (_, i) => (i * 31 + Math.floor(i / K) * 17 + offset) % 256 - 128);
      const scales = Float32Array.from({ length: N * K / 32 }, (_, i) => (i % 2 ? -1 : 1) * (0.0002 + i % 7 * 0.000037));
      if (saturated) for (let row = 0; row < 2; row++) {
        values.fill(offset ? 17 : row ? -128 : 127, row * K, (row + 1) * K);
        scales.fill(0.05, row * K / 32, (row + 1) * K / 32);
      }
      return { info: { shape: [N, K], dtype: "q8" }, buf: make(values.byteLength, U.STORAGE | U.COPY_DST, values), sbuf: make(scales.byteLength, U.STORAGE | U.COPY_DST, scales) };
    };
    const gateWeights = data(0), upWeights = data(19);
    const globals = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 0, 0, 0]));
    const params = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, K, 0, 0]));
    const geluParams = make(16, U.UNIFORM | U.COPY_DST, new Uint32Array([N, 0, 0, 0]));
    const x = make(T * K * 4, U.STORAGE | U.COPY_DST, Float32Array.from({ length: T * K }, (_, i) => saturated ? 0.32 * (1 + Math.floor(i / K) / 8) : 0.04 * Math.sin(i * 0.127 + 0.3)));
    const zero = make(16, U.STORAGE), gate = make(T * N * 4, U.STORAGE), up = make(T * N * 4, U.STORAGE);
    const part = make(16, U.STORAGE);
    const initial = new Float32Array(T * N + 4).fill(NaN); initial.fill(SENTINEL, T * N);
    const outputBytes = initial.byteLength;
    const outputs = [0, 1].map(() => make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, initial));
    const read = make(outputBytes * 2, U.COPY_DST | U.MAP_READ);
    const group = (layout, buffers) => device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const op = (weights, output) => ({ N, group: group(baseline.layout, [globals, params, x, weights.buf, weights.sbuf, zero, output, part]) });
    const gateOp = op(gateWeights, gate), upOp = op(upWeights, up);
    const geluGroup = group(gelu.getBindGroupLayout(0), [globals, geluParams, gate, up, outputs[0]]);
    const fused = { N, group: matvecGeluBindGroup(device, candidate, { globals, params, input: x, gate: gateWeights.buf, gateScales: gateWeights.sbuf, up: upWeights.buf, upScales: upWeights.sbuf, output: outputs[1] }) };
    const dispatch = (pass, fusion) => {
      if (fusion) { encodeMatvecGelu(pass, candidate, fused, T); return; }
      encodeMatvec(pass, baseline, gateOp, T); encodeMatvec(pass, baseline, upOp, T);
      pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup); pass.dispatchWorkgroups(Math.ceil(T * N / 256));
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, false); dispatch(pass, true); pass.end();
    outputs.forEach((value, i) => encoder.copyBufferToBuffer(value, 0, read, i * outputBytes, outputBytes));
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
    const first = actual.subarray(0, T * N + 4), second = actual.subarray(T * N + 4);
    let maxAbs = 0;
    for (let i = 0; i < T * N; i++) {
      if (!Number.isFinite(first[i]) || !Number.isFinite(second[i])) throw new Error(`nonfinite paired output ${i}`);
      maxAbs = Math.max(maxAbs, Math.abs(first[i] - second[i]));
    }
    if (maxAbs !== 0) throw new Error(`paired T${T} ${N}x${K} exact parity failed: ${maxAbs}`);
    for (let i = T * N; i < T * N + 4; i++) if (first[i] !== SENTINEL || second[i] !== SENTINEL) throw new Error(`paired tail overwrite at ${i}`);
    if (saturated) for (let row = 0; row < T; row++) if (first[row * N] <= 10 || first[row * N + 1] !== 0) throw new Error("positive/negative GELU saturation cases were not exercised");
    let timing = null, calibration = null;
    if (N >= 6144 && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 4);
      const samples = { baseline: [], candidate: [] }, repetitions = 4;
      for (let sample = 0; sample < 7; sample++) {
        const encoder = device.createCommandEncoder();
        for (const fusion of sample % 2 ? [true, false] : [false, true]) {
          const pass = profiler.pass(encoder, fusion ? "candidate" : "baseline");
          for (let i = 0; i < repetitions; i++) dispatch(pass, fusion);
          pass.end();
        }
        profiler.finish(encoder); device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const label of ["baseline", "candidate"]) samples[label].push(measured[label] / repetitions);
      }
      timing = { ...samples, baselineMedianMs: median(samples.baseline), candidateMedianMs: median(samples.candidate) };
      if (T === 1) {
        const modelWeights = { tensors: new Map([["l.0.gate", gateWeights], ["l.0.up", upWeights]]) };
        calibration = (await calibrateMatvecGelu(device, source, modelWeights, { pipelines: baseline, selection: new Set([`${N}:${K}`]) }, { gelu })).diagnostics;
        if (calibration.points.some((point) => point.numericalCheck?.ok !== true)) throw new Error("paired calibration numerical guard failed");
      }
    }
    return { T, N, K, saturated, f16: baseline.f16, maxAbs, tailGuard: true, timing, calibration, ok: true };
  } finally { profiler?.destroy(); for (const value of owned) value.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  const runtime = await requestDevice({ baseline: params.has("baseline") });
  try {
    const source = await kernelSource(), cases = [];
    for (const T of [1, 2, 5]) for (const [N, K, saturated] of [[1, 32], [17, 96, true], [17, 1056], [6144, 672], [6144, 1536], [12288, 1536]]) {
      const result = await withGpuErrors(runtime, () => check(runtime, source, N, K, T, saturated), `paired matvec T${T} ${N}x${K}`);
      cases.push(result); log(result);
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", metricMs: performance.now() - started,
      adapter: { name: runtime.name, isFallbackAdapter: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter), features: [...runtime.device.features] },
      correctness: { ok: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", error: String(error?.message || error) }; log(error.stack || error); });
