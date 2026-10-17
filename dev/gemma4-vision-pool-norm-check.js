// Compare the complete output bits with the original pool + norm dispatches.
import { pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const SENTINEL = 12345.25;
const WIDTHS = [32, 256, 768, 1024];
const PATTERNS = ["signed", "cancellation", "extreme", "tiny", "zero"];
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function inputFor(rows, width, patches, standardize, pattern) {
  const inputRows = rows * patches + 5;
  const input = new Float32Array(inputRows * width);
  const indices = new Uint32Array(rows * patches);
  const bias = new Float32Array(standardize ? width : 4).fill(NaN);
  const scale = new Float32Array(standardize ? width : 4).fill(NaN);
  for (let row = 0; row < inputRows; row++) for (let col = 0; col < width; col++) {
    const sign = (row + col) % 2 ? -1 : 1;
    let value;
    if (pattern === "cancellation") {
      const patch = row % patches;
      value = patch < 2 ? (patch ? -1 : 1) * 1e8 * (1 + col % 7 / 16) : sign * (0.000137 + col % 11 * 0.000031);
    } else if (pattern === "extreme") value = sign * (1e37 + col % 7 * 1e34);
    else if (pattern === "tiny") value = sign * (2 ** -120) * (1 + col % 7 / 16);
    else if (pattern === "zero") value = sign < 0 ? -0 : 0;
    else value = sign * (0.00371 + (row * 13 + col * 17) % 97 * 0.000237);
    input[row * width + col] = value;
  }
  for (let row = 0; row < rows; row++) for (let patch = 0; patch < patches; patch++) {
    // Reverse alternating rows; reuse a patch in the final row to exercise the
    // explicit index table rather than an assumed contiguous pooling layout.
    const at = row % 2 ? patches - patch - 1 : patch;
    indices[row * patches + patch] = row * patches + (row === rows - 1 && patch === patches - 1 ? 0 : at);
  }
  if (standardize) for (let col = 0; col < width; col++) {
    bias[col] = (col % 2 ? -1 : 1) * (0.1237 + col % 7 * 0.0371);
    scale[col] = (col % 3 ? 1 : -1) * (pattern === "extreme" ? 1e-37 : 0.317 + col % 11 * 0.0213);
  }
  return { input, indices, bias, scale, eps: pattern === "tiny" ? 3.71e-5 : 1e-6 };
}

function compare(a, b) {
  const aBits = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const bBits = new Uint32Array(b.buffer, b.byteOffset, b.length);
  let bitMismatches = 0, maxAbs = 0, firstDifference = null;
  for (let index = 0; index < a.length; index++) {
    if (!Number.isFinite(a[index]) || !Number.isFinite(b[index])) throw new Error(`pool norm nonfinite output at ${index}`);
    maxAbs = Math.max(maxAbs, Math.abs(a[index] - b[index]));
    if (aBits[index] !== bBits[index]) {
      bitMismatches++;
      firstDifference ??= { index, baseline: a[index], candidate: b[index], baselineBits: aBits[index], candidateBits: bBits[index] };
    }
  }
  return { ok: bitMismatches === 0, bitExact: bitMismatches === 0, samples: a.length, bitMismatches, maxAbs, firstDifference };
}

async function check(runtime, kernels, config, timed) {
  const { device } = runtime, U = GPUBufferUsage;
  const { rows, width, patches, standardize, pattern } = config;
  const input = inputFor(rows, width, patches, standardize, pattern);
  const owned = [];
  let profiler;
  const make = (size, usage, values) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage, label: "vision pool norm guard" });
    owned.push(buffer); if (values) device.queue.writeBuffer(buffer, 0, values); return buffer;
  };
  const uniform = (values) => {
    const bytes = new ArrayBuffer(32), ints = new Uint32Array(bytes), floats = new Float32Array(bytes);
    values.forEach((value, index) => { if (typeof value === "object") floats[index] = value.f; else ints[index] = value; });
    return make(bytes.byteLength, U.UNIFORM | U.COPY_DST, bytes);
  };
  try {
    const n = rows * width;
    const initial = new Float32Array(n + 4).fill(NaN); initial.fill(SENTINEL, n);
    const bytes = initial.byteLength;
    const x = make(input.input.byteLength, U.STORAGE | U.COPY_DST, input.input);
    const indices = make(input.indices.byteLength, U.STORAGE | U.COPY_DST, input.indices);
    const bias = make(input.bias.byteLength, U.STORAGE | U.COPY_DST, input.bias);
    const scale = make(input.scale.byteLength, U.STORAGE | U.COPY_DST, input.scale);
    const unused = make(16, U.STORAGE | U.COPY_DST, new Float32Array(4).fill(NaN));
    const pooled = make(bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, initial);
    const baseline = make(bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, initial);
    const candidate = make(bytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, initial);
    const poolParams = uniform([rows, width, patches, Number(standardize)]);
    const normParams = uniform([rows, width, 0, 0, { f: input.eps }, { f: -Infinity }, { f: Infinity }, 0]);
    const fusedParams = uniform([rows, width, patches, Number(standardize), { f: input.eps }, { f: -Infinity }, { f: Infinity }, 0]);
    const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const poolGroup = group(kernels.pool, [poolParams, x, indices, bias, scale, pooled]);
    const normGroup = group(kernels.norm, [normParams, pooled, unused, unused, baseline]);
    const fusedGroup = group(kernels.fused, [fusedParams, x, indices, bias, scale, candidate]);
    const dispatch = (pass, fused, extraRow = false) => {
      if (fused) {
        pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedGroup); pass.dispatchWorkgroups(rows + Number(extraRow));
      } else {
        pass.setPipeline(kernels.pool); pass.setBindGroup(0, poolGroup); pass.dispatchWorkgroups(Math.ceil(rows * width / 1024));
        pass.setPipeline(kernels.norm); pass.setBindGroup(0, normGroup); pass.dispatchWorkgroups(rows);
      }
    };
    const read = make(bytes * 2 + 16, U.MAP_READ | U.COPY_DST);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, false); dispatch(pass, true, true); pass.end();
    encoder.copyBufferToBuffer(baseline, 0, read, 0, bytes);
    encoder.copyBufferToBuffer(candidate, 0, read, bytes, bytes);
    encoder.copyBufferToBuffer(pooled, n * 4, read, bytes * 2, 16);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
    const numerical = compare(actual.subarray(0, n), actual.subarray(n + 4, n + 4 + n));
    for (const start of [n, n + 4 + n, (n + 4) * 2]) for (let index = start; index < start + 4; index++) if (actual[index] !== SENTINEL) throw new Error("pool norm output tail overwritten");
    let timing = null;
    if (numerical.ok && timed && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 4);
      const samples = { baseline: [], candidate: [] }, repetitions = 16;
      for (let sample = 0; sample < 7; sample++) {
        const encoder = device.createCommandEncoder();
        for (const fused of sample % 2 ? [true, false] : [false, true]) {
          const pass = profiler.pass(encoder, fused ? "candidate" : "baseline");
          for (let repetition = 0; repetition < repetitions; repetition++) dispatch(pass, fused);
          pass.end();
        }
        profiler.finish(encoder); device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const label of ["baseline", "candidate"]) samples[label].push(measured[label] / repetitions);
      }
      timing = { ...samples, baselineMedianMs: median(samples.baseline), candidateMedianMs: median(samples.candidate) };
    }
    return { ...config, eps: input.eps, ...numerical, tailGuard: true, extraWorkgroupGuard: true, timing };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), params = new URLSearchParams(location.search);
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  try {
    if (runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter) throw new Error("pool norm guard requires a hardware adapter");
    const source = await kernelSource();
    const codes = { pool: source("gemma4_vision_pool"), norm: source("gemma4_vision_norm"), fused: source("gemma4_vision_pool_norm") };
    const kernels = Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await pipeline(runtime.device, code, `vision_pool_norm_${name}`)])));
    const cases = [];
    const configurations = [];
    if (params.get("quick") !== "1") for (const width of WIDTHS) for (const standardize of [false, true]) for (const pattern of PATTERNS) for (const patches of [4, 9]) for (const rows of [1, 7]) configurations.push({ rows, width, patches, standardize, pattern });
    for (const rows of [40, 70]) for (const standardize of [false, true]) configurations.push({ rows, width: 768, patches: 9, standardize, pattern: "signed" });
    for (const config of configurations) {
      const result = await withGpuErrors(runtime, () => check(runtime, kernels, config, config.width === 768 && config.rows >= 40), `pool norm ${JSON.stringify(config)}`);
      cases.push(result); log(result);
      if (!result.ok) throw Object.assign(new Error("vision pool norm bit-exact parity failed"), { cases });
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", metricMs: performance.now() - started,
      adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features] },
      shaderHashes: Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)]))),
      correctness: { ok: cases.every((result) => result.ok), tolerance: 0, bitExact: true, cases } };
  } finally { runtime.device.destroy(); }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu" };
main().catch((error) => { window.gpuBench = { status: "error", done: true, backend: "webgpu", error: error.message, correctness: { ok: false, cases: error.cases || [] } }; log(error.stack || error); });
