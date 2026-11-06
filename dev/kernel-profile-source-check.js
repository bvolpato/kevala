import { matmulLayout, mmSplits, pipeline, Profiler, reduceLayout, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";
import { pleReference, reduceReference, SOURCE_PROFILE_CASES, sourceFixture, sourcePartials } from "./kernel-profile-source-fixtures.js";

const log = (item) => { document.getElementById("log").textContent += `${JSON.stringify(item)}\n`; };
const U = GPUBufferUsage, SEED = 20261106, TAIL = 4, SENTINEL = 12345.25;
const SPEC = { weights: "q8", f16: false, rows: 4, groups: 1, splitTarget: 256 };
const MATRIX_KERNELS = ["matmul_gemma_source", "matmul_wide_gemma_source", "matmul_k64_gemma_source"];
const MAX_GPU_BYTES = 128 * 1024 * 1024;
const median = (values) => values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)];
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((value) => value.toString(16).padStart(2, "0")).join("");

function resources(device) {
  const owned = []; let bytes = 0;
  const make = (size, usage, data) => {
    size = Math.max(16, size);
    bytes += size;
    if (bytes > MAX_GPU_BYTES || size > device.limits.maxBufferSize || usage & U.STORAGE && size > device.limits.maxStorageBufferBindingSize) throw new Error(`source fixture exceeds its GPU memory bound: ${bytes} bytes`);
    const buffer = device.createBuffer({ size, usage: usage | U.COPY_DST | (usage & U.STORAGE ? U.COPY_SRC : 0) });
    owned.push(buffer); if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const initialized = (count, initial = null) => {
    const values = new Float32Array(count + TAIL).fill(NaN);
    if (initial) values.set(initial);
    values.fill(SENTINEL, count);
    return values;
  };
  const group = (compute, buffers) => device.createBindGroup({ layout: compute.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const read = async (buffers) => {
    const readback = make(buffers.reduce((sum, buffer) => sum + buffer.size, 0), U.MAP_READ);
    const encoder = device.createCommandEncoder(); let offset = 0;
    for (const buffer of buffers) { encoder.copyBufferToBuffer(buffer, 0, readback, offset, buffer.size); offset += buffer.size; }
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    try {
      let at = 0; const mapped = readback.getMappedRange();
      return buffers.map((buffer) => { const values = new Float32Array(mapped.slice(at, at + buffer.size)); at += buffer.size; return values; });
    } finally { readback.unmap(); readback.destroy(); bytes -= readback.size; }
  };
  return { make, initialized, group, read, get bytes() { return bytes; }, destroy() { for (const buffer of owned) buffer.destroy(); } };
}

function compare(actual, expected, label, tolerance = 0.0005) {
  if (actual.length !== expected.length + TAIL) throw new Error(`${label}: output size differs`);
  let maxAbs = 0, maxRelative = 0, bitDifferences = 0;
  const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length), b = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < expected.length; index++) {
    const delta = Math.abs(actual[index] - expected[index]);
    if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index]) || delta > tolerance && delta / Math.max(1, Math.abs(expected[index])) > 0.0001) throw new Error(`${label}: numerical mismatch at ${index}: ${actual[index]} versus ${expected[index]}`);
    maxAbs = Math.max(maxAbs, delta); maxRelative = Math.max(maxRelative, delta / Math.max(1, Math.abs(expected[index])));
    if (a[index] !== b[index]) bitDifferences++;
  }
  for (let index = expected.length; index < actual.length; index++) if (actual[index] !== SENTINEL) throw new Error(`${label}: output tail overwritten at ${index}`);
  return { values: expected.length, maxAbs, maxRelative, bitDifferences, tailGuard: true };
}

async function timing(device, paths, reset) {
  const profiler = new Profiler(device, 16), samples = Object.fromEntries(Object.keys(paths).map((name) => [name, []])), names = Object.keys(paths);
  try {
    for (let round = 0; round < 12; round++) {
      reset();
      const encoder = device.createCommandEncoder();
      for (let index = 0; index < names.length; index++) {
        const name = names[(index + round) % names.length], pass = profiler.pass(encoder, name);
        for (let repeat = 0; repeat < 4; repeat++) paths[name](pass);
        pass.end();
      }
      profiler.finish(encoder); device.queue.submit([encoder.finish()]);
      const result = await profiler.collect();
      if (round >= 3) for (const name of names) {
        const value = result[name] / 4;
        if (!Number.isFinite(value) || value <= 0) throw new Error(`${name}: invalid GPU timestamp`);
        samples[name].push(value);
      }
    }
    return { method: "WebGPU timestamp-query; rotating path order; three warmups and nine samples; four repetitions", ...samples,
      mediansMs: Object.fromEntries(names.map((name) => [name, median(samples[name])])) };
  } finally { profiler.destroy(); }
}

async function checkMatrix(runtime, source, shape) {
  const { device } = runtime, r = resources(device), { T, R, N, K } = shape, count = T * N;
  const splits = mmSplits(R, N, K, 256, 64, 64), partialCount = splits > 1 ? splits * count : 0;
  const data = sourceFixture(shape, SEED), cpuPartials = sourcePartials(data, splits), guards = [];
  try {
    const codes = Object.fromEntries(MATRIX_KERNELS.map((kernel) => [kernel, source(kernel, SPEC)]));
    const reduceCode = source("reduce_gemma_source", SPEC), layout = device.createPipelineLayout({ bindGroupLayouts: [matmulLayout(device)] });
    const computes = await Promise.all(MATRIX_KERNELS.map((name) => pipeline(device, codes[name], name, layout)));
    const reducer = await pipeline(device, reduceCode, "reduce_gemma_source", device.createPipelineLayout({ bindGroupLayouts: [reduceLayout(device)] }));
    const globals = r.make(16, U.UNIFORM, new Uint32Array([T, R, 0, 0])), params = r.make(16, U.UNIFORM);
    const input = r.make(data.input.byteLength, U.STORAGE, data.input), weights = r.make(data.weights.byteLength, U.STORAGE, data.weights), scales = r.make(data.scales.byteLength, U.STORAGE, data.scales), bias = r.make(data.bias.byteLength, U.STORAGE);
    const outputs = computes.map(() => r.make((count + TAIL) * 4, U.STORAGE)), partials = computes.map(() => r.make((partialCount + TAIL) * 4, U.STORAGE));
    const groups = computes.map((compute, index) => r.group(compute, [globals, params, input, weights, scales, bias, outputs[index], partials[index]]));
    const reducers = computes.map((_, index) => r.group(reducer, [globals, params, partials[index], bias, outputs[index]]));
    const dispatch = (index, pass) => {
      pass.setPipeline(computes[index]); pass.setBindGroup(0, groups[index]); pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(T / 64), splits);
      if (splits > 1) { pass.setPipeline(reducer); pass.setBindGroup(0, reducers[index]); pass.dispatchWorkgroups(Math.ceil(count / 256)); }
    };
    const reset = (mode = 0) => {
      outputs.forEach((buffer) => device.queue.writeBuffer(buffer, 0, r.initialized(count, mode === 1 ? data.residual : null)));
      partials.forEach((buffer) => device.queue.writeBuffer(buffer, 0, r.initialized(partialCount)));
    };
    const check = async () => {
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
        device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias]));
        device.queue.writeBuffer(bias, 0, hasBias ? data.bias : new Float32Array(N).fill(NaN)); reset(mode);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        computes.forEach((_, index) => dispatch(index, pass)); pass.end(); device.queue.submit([encoder.finish()]);
        const actual = await r.read([...outputs, ...partials]);
        const expected = reduceReference(cpuPartials, { T, N, splits, mode, bias: hasBias ? data.bias : null, residual: data.residual });
        const checks = MATRIX_KERNELS.map((kernel, index) => ({ kernel, output: compare(actual[index], expected, `${kernel} CPU output`),
          pairedOutput: compare(actual[index], actual[0].subarray(0, count), `${kernel} paired output`),
          partials: compare(actual[index + computes.length], partialCount ? cpuPartials : new Float32Array(0), `${kernel} CPU partials`) }));
        guards.push({ mode, bias: hasBias, checks });
      }
    };
    await check();
    device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0])); device.queue.writeBuffer(bias, 0, new Float32Array(N).fill(NaN));
    const measured = await timing(device, Object.fromEntries(MATRIX_KERNELS.map((name, index) => [name, (pass) => dispatch(index, pass)])), reset);
    await check();
    return { kind: "source-projection", ...shape, splits, kernels: [...MATRIX_KERNELS, "reduce_gemma_source"], seed: SEED,
      scope: "registered source-policy matmul plus its unchanged reducer; not isolated kernel or model latency", correctness: { ok: true, cpuAbsoluteTolerance: 0.0005, cpuRelativeTolerance: 0.0001, fullOutputs: true, fullPartials: true, beforeAndAfterTiming: true, guards }, timing: measured,
      sourceHashes: Object.fromEntries(await Promise.all([...Object.entries(codes), ["reduce_gemma_source", reduceCode]].map(async ([name, code]) => [name, await hash(code)]))), gpuBytes: r.bytes };
  } finally { r.destroy(); }
}

async function checkPle(runtime, source, shape) {
  const { device } = runtime, r = resources(device), { T, N, K } = shape, count = T * N, splits = mmSplits(T, N, K, 256, 64, 64), guards = [];
  try {
    const kernels = ["reduce", "reduce_ple", "gemma4_gelu"], codes = Object.fromEntries(kernels.map((name) => [name, source(name, SPEC)]));
    const [reduce, fused, gelu] = await Promise.all(kernels.map((name) => pipeline(device, codes[name], name)));
    const globals = r.make(16, U.UNIFORM, new Uint32Array([T, 0, 0, 0])), params = r.make(16, U.UNIFORM), geluParams = r.make(16, U.UNIFORM, new Uint32Array([N, 0, 0, 0]));
    const parts = Float32Array.from({ length: splits * count }, (_, index) => ((index * 31 + SEED) % 17 - 8) / 64);
    const b = Float32Array.from({ length: N }, (_, index) => (index % 7 - 3) / 128), residual = Float32Array.from({ length: count }, (_, index) => (index % 11 - 5) / 256), ple = Float32Array.from({ length: count }, (_, index) => (index % 17 - 8) / 128);
    const partials = r.make(parts.byteLength, U.STORAGE, parts), bias = r.make(b.byteLength, U.STORAGE), p = r.make(ple.byteLength, U.STORAGE, ple);
    const y = [0, 1].map(() => r.make((count + TAIL) * 4, U.STORAGE)), a = [0, 1].map(() => r.make((count + TAIL) * 4, U.STORAGE));
    const reduceGroup = r.group(reduce, [globals, params, partials, bias, y[0]]), fusedGroup = r.group(fused, [globals, params, partials, bias, y[1], p, a[1]]), geluGroup = r.group(gelu, [globals, geluParams, y[0], p, a[0]]);
    const baseline = (pass) => { pass.setPipeline(reduce); pass.setBindGroup(0, reduceGroup); pass.dispatchWorkgroups(Math.ceil(count / 256)); pass.setPipeline(gelu); pass.setBindGroup(0, geluGroup); pass.dispatchWorkgroups(Math.ceil(count / 256)); };
    const candidate = (pass) => { pass.setPipeline(fused); pass.setBindGroup(0, fusedGroup); pass.dispatchWorkgroups(Math.ceil(count / 256)); };
    const reset = (mode = 0) => { y.forEach((buffer) => device.queue.writeBuffer(buffer, 0, r.initialized(count, mode === 1 ? residual : null))); a.forEach((buffer) => device.queue.writeBuffer(buffer, 0, r.initialized(count))); };
    const check = async () => {
      for (const mode of [0, 1, 2]) for (const hasBias of [0, 1]) {
        device.queue.writeBuffer(params, 0, new Uint32Array([N, K, mode, hasBias])); device.queue.writeBuffer(bias, 0, hasBias ? b : new Float32Array(N).fill(NaN)); reset(mode);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); baseline(pass); candidate(pass); pass.end(); device.queue.submit([encoder.finish()]);
        const actual = await r.read(a), values = reduceReference(parts, { T, N, splits, mode, bias: hasBias ? b : null, residual }), expected = pleReference(values, ple);
        const cpu = actual.map((value, index) => compare(value, expected, `PLE ${index} CPU`, 0.00005)), paired = compare(actual[1], actual[0].subarray(0, count), "PLE paired", 0);
        if (paired.bitDifferences) throw new Error(`PLE candidate changed ${paired.bitDifferences} output bits`);
        guards.push({ mode, bias: hasBias, cpu, paired });
      }
    };
    await check(); device.queue.writeBuffer(params, 0, new Uint32Array([N, K, 0, 0])); device.queue.writeBuffer(bias, 0, new Float32Array(N).fill(NaN));
    const measured = await timing(device, { baseline, candidate }, reset); await check();
    return { kind: "generic-ple", ...shape, splits, kernels, scope: "standalone reducer plus unchanged gated GELU versus fused PLE reducer", timing: measured,
      correctness: { ok: true, bitExact: true, cpuAbsoluteTolerance: 0.00005, fullOutputs: true, beforeAndAfterTiming: true, guards },
      sourceHashes: Object.fromEntries(await Promise.all(Object.entries(codes).map(async ([name, code]) => [name, await hash(code)]))), gpuBytes: r.bytes };
  } finally { r.destroy(); }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", cases: [] };
async function main() {
  const runtime = await requestDevice({ features: ["timestamp-query"] });
  const cases = [];
  try {
    if (!runtime.device.features.has("timestamp-query")) throw new Error("timestamp-query is required for source kernel profiling");
    const source = await kernelSource(), errors = [];
    runtime.device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
    for (const shape of SOURCE_PROFILE_CASES) {
      const result = await withGpuErrors(runtime, () => checkMatrix(runtime, source, shape), "source projection reference and timestamps");
      cases.push(result); window.gpuBench.cases = cases; log({ kind: result.kind, shape, timing: result.timing });
    }
    for (const shape of [{ T: 3, N: 17, K: 128 }, { T: 17, N: 257, K: 1024 }]) {
      const result = await withGpuErrors(runtime, () => checkPle(runtime, source, shape), "generic PLE reference and timestamps");
      cases.push(result); window.gpuBench.cases = cases; log({ kind: result.kind, shape, timing: result.timing });
    }
    if (errors.length) throw new Error(errors.join("; "));
    const medians = cases.flatMap((item) => Object.values(item.timing.mediansMs));
    window.gpuBench = { status: "done", done: true, backend: "webgpu", method: "WebGPU timestamp-query; paired complete operations", metricMs: Math.exp(medians.reduce((sum, value) => sum + Math.log(value), 0) / medians.length),
      adapter: { name: runtime.name, isFallbackAdapter: Boolean(runtime.adapter.isFallbackAdapter || runtime.adapter.info?.isFallbackAdapter), features: [...runtime.device.features] },
      seed: SEED, wasmUrl: source.wasmUrl, correctness: { ok: true, cases, scope: "full-output CPU and registered-control comparisons; tails; mode/bias branches; before and after timing" }, cases };
  } finally { runtime.device.destroy(); }
}
main().catch((error) => { window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.stack || error), correctness: { ok: false } }; log(error.stack || String(error)); });
