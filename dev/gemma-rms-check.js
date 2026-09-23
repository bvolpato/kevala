// Compare the existing Gemma RMS mode-1 plus residual mode-0 sequence with the
// fused RMS mode-2 operation. Every case compares complete output arrays. The
// bit comparison is diagnostic because WGSL permits arithmetic fusion; the
// numerical guard uses the existing Gemma tail-check absolute tolerance.

import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const query = new URLSearchParams(location.search);
const U = GPUBufferUsage;
const EPS = 1e-6;
const ABS_TOLERANCE = 2e-5;
const WIDTHS = [32, 257, 1536, 2048, 2560];
const TOKENS = [1, 8, 65];
const PATTERNS = ["signed", "zero", "high"];
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };

function numberParam(name, fallback, min, max) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  return Number.isInteger(value) && value >= min && value <= max ? value : fallback;
}

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
}

const f32 = Math.fround;

function makeInput(T, D, pattern, seed) {
  const random = rng(seed);
  const x = new Float32Array(T * D);
  const residual = new Float32Array(T * D);
  const weights = new Float32Array(D);
  for (let i = 0; i < x.length; i++) {
    const signed = Math.floor(random() * 17) - 8;
    const base = (Math.floor(random() * 17) - 8);
    if (pattern === "zero") x[i] = 0;
    else if (pattern === "high") x[i] = f32(signed * 1024);
    else x[i] = f32(signed / 16);
    residual[i] = pattern === "high" ? f32(base * 16) : f32(base / 32);
  }
  for (let d = 0; d < D; d++) {
    const signed = Math.floor(random() * 13) - 6;
    weights[d] = pattern === "high" ? f32(signed * 8) : f32(signed / 8);
  }
  return { T, D, pattern, x, residual, weights };
}

function cpuReference(input) {
  const { T, D, x, residual, weights } = input;
  const output = new Float64Array(x.length);
  for (let t = 0; t < T; t++) {
    const base = t * D;
    let sum = 0;
    for (let d = 0; d < D; d++) {
      const value = x[base + d];
      sum += value * value;
    }
    const inv = 1 / Math.sqrt(sum / D + EPS);
    for (let d = 0; d < D; d++) output[base + d] = residual[base + d] + x[base + d] * inv * weights[d];
  }
  return output;
}

function makeBuffer(device, bytes, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, bytes), usage });
}

function uniform(device, values, label) {
  const raw = new ArrayBuffer(16);
  const ints = new Uint32Array(raw);
  const floats = new Float32Array(raw);
  values.forEach((value, index) => {
    if (typeof value === "object") floats[index] = value.f;
    else ints[index] = value;
  });
  const buffer = makeBuffer(device, raw.byteLength, U.UNIFORM | U.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, raw);
  return buffer;
}

function createBindGroup(device, pipelineObject, buffers, label) {
  return device.createBindGroup({
    label,
    layout: pipelineObject.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

function makeResources(device, rms, residualKernel, input, caseLabel) {
  const { T, D, x, residual, weights } = input;
  const owned = [];
  const storage = (bytes, usage, label) => {
    const buffer = makeBuffer(device, bytes, U.STORAGE | usage, `${caseLabel}.${label}`);
    owned.push(buffer);
    return buffer;
  };
  const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${caseLabel}.globals`);
  device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
  owned.push(globals);
  const rmsMode1 = uniform(device, [D, { f: EPS }, 1, 0], `${caseLabel}.rms.mode1`);
  const rmsMode2 = uniform(device, [D, { f: EPS }, 2, 0], `${caseLabel}.rms.mode2`);
  const residualMode0 = uniform(device, [D, 0, 0, 0], `${caseLabel}.residual.mode0`);
  owned.push(rmsMode1, rmsMode2, residualMode0);
  const inputBuffer = storage(x.byteLength, U.COPY_DST, "input");
  const weightsBuffer = storage(weights.byteLength, U.COPY_DST, "weights");
  const oldInput = storage(residual.byteLength, U.COPY_DST | U.COPY_SRC, "old-input");
  const weighted = storage(residual.byteLength, U.COPY_DST, "weighted");
  const fusedOutput = storage(residual.byteLength, U.COPY_DST | U.COPY_SRC, "fused-output");
  const zeros = storage(16, U.COPY_DST, "zero-scale");
  device.queue.writeBuffer(inputBuffer, 0, x);
  device.queue.writeBuffer(weightsBuffer, 0, weights);
  device.queue.writeBuffer(oldInput, 0, residual);
  device.queue.writeBuffer(fusedOutput, 0, residual);
  device.queue.writeBuffer(zeros, 0, new Float32Array([0]));
  return {
    T,
    D,
    inputBuffer,
    weightsBuffer,
    oldInput,
    weighted,
    fusedOutput,
    oldRms: createBindGroup(device, rms, [globals, rmsMode1, inputBuffer, weightsBuffer, weighted], `${caseLabel}.old-rms`),
    fusedRms: createBindGroup(device, rms, [globals, rmsMode2, inputBuffer, weightsBuffer, fusedOutput], `${caseLabel}.fused-rms`),
    residual: createBindGroup(device, residualKernel, [globals, residualMode0, oldInput, weighted, zeros], `${caseLabel}.residual`),
    owned,
    destroy() { for (const buffer of owned) buffer.destroy(); },
  };
}

function appendDispatch(encoder, pipelineObject, group, T) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipelineObject);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(T);
  pass.end();
}

async function checked(device, operation, phase, uncaptured) {
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let value;
  let failure;
  try {
    value = await operation();
  } catch (error) {
    failure = error;
  }
  const scopes = await Promise.allSettled([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  const errors = scopes.flatMap((item) => item.status === "rejected" ? [item.reason] : item.value ? [item.value] : []);
  if (errors.length) throw new Error(`WebGPU ${phase}: ${errors.map((error) => error.message).join("; ")}`);
  if (failure) throw failure;
  if (uncaptured.length) throw new Error(`WebGPU ${phase}: uncaptured error: ${uncaptured.splice(0).map((item) => item.message).join("; ")}`);
  return value;
}

async function readBuffer(device, source, bytes, label, uncaptured) {
  const readback = makeBuffer(device, bytes, U.MAP_READ | U.COPY_DST, `${label}.readback`);
  try {
    return await checked(device, async () => {
      const encoder = device.createCommandEncoder({ label: `${label}.readback` });
      encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ, 0, bytes);
      const values = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
      readback.unmap();
      return values;
    }, `${label} readback`, uncaptured);
  } finally {
    readback.destroy();
  }
}

async function runCorrectness(device, rms, residualKernel, input, uncaptured) {
  const label = `${input.pattern}-T${input.T}-D${input.D}`;
  const resources = makeResources(device, rms, residualKernel, input, label);
  try {
    await checked(device, async () => {
      const encoder = device.createCommandEncoder({ label });
      appendDispatch(encoder, rms, resources.oldRms, input.T);
      appendDispatch(encoder, residualKernel, resources.residual, input.T);
      appendDispatch(encoder, rms, resources.fusedRms, input.T);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }, `${label} dispatch`, uncaptured);
    // Keep readbacks sequential because WebGPU error scopes are device-global.
    const oldOutput = await readBuffer(device, resources.oldInput, input.x.byteLength, `${label}.old`, uncaptured);
    const fusedOutput = await readBuffer(device, resources.fusedOutput, input.x.byteLength, `${label}.fused`, uncaptured);
    const result = { oldOutput, fusedOutput };
    const expected = cpuReference(input);
    const oldBits = new Uint32Array(result.oldOutput.buffer, result.oldOutput.byteOffset, result.oldOutput.length);
    const fusedBits = new Uint32Array(result.fusedOutput.buffer, result.fusedOutput.byteOffset, result.fusedOutput.length);
    let maxAbs = 0;
    let maxRelative = 0;
    let cpuMaxAbs = 0;
    let cpuMaxRelative = 0;
    let bitDifferences = 0;
    let firstBitDifference = null;
    let finite = true;
    for (let i = 0; i < result.oldOutput.length; i++) {
      const oldValue = result.oldOutput[i];
      const fusedValue = result.fusedOutput[i];
      if (!Number.isFinite(oldValue) || !Number.isFinite(fusedValue) || !Number.isFinite(expected[i])) finite = false;
      const abs = Math.abs(oldValue - fusedValue);
      maxAbs = Math.max(maxAbs, abs);
      maxRelative = Math.max(maxRelative, abs / Math.max(1, Math.abs(oldValue)));
      const cpuAbs = Math.max(Math.abs(oldValue - expected[i]), Math.abs(fusedValue - expected[i]));
      cpuMaxAbs = Math.max(cpuMaxAbs, cpuAbs);
      cpuMaxRelative = Math.max(cpuMaxRelative, cpuAbs / Math.max(1, Math.abs(expected[i])));
      if (oldBits[i] !== fusedBits[i]) {
        bitDifferences++;
        if (!firstBitDifference) firstBitDifference = { index: i, old: oldBits[i], fused: fusedBits[i] };
      }
    }
    return {
      label,
      T: input.T,
      D: input.D,
      pattern: input.pattern,
      outputsCompared: result.oldOutput.length,
      maxAbs,
      maxRelative,
      bitDifferences,
      bitExact: bitDifferences === 0,
      firstBitDifference,
      cpuReference: { maxAbs: cpuMaxAbs, maxRelative: cpuMaxRelative, ok: finite && cpuMaxAbs <= ABS_TOLERANCE },
      finite,
      ok: finite && maxAbs <= ABS_TOLERANCE && cpuMaxAbs <= ABS_TOLERANCE,
      tolerance: { absolute: ABS_TOLERANCE },
    };
  } finally {
    resources.destroy();
  }
}

function makeTimer(device) {
  if (!device.features.has("timestamp-query")) return { timestamp: false, method: "performance.now() wall clock" };
  return {
    timestamp: true,
    method: "WebGPU timestamp-query",
    querySet: device.createQuerySet({ type: "timestamp", count: 2, label: "gemma-rms.timestamps" }),
    resolved: makeBuffer(device, 16, U.QUERY_RESOLVE | U.COPY_SRC, "gemma-rms.timestamp-resolved"),
    read: makeBuffer(device, 256, U.MAP_READ | U.COPY_DST, "gemma-rms.timestamp-read"),
  };
}

async function timed(device, timer, resources, rms, residualKernel, separate, reps, uncaptured, phase) {
  return checked(device, async () => {
    const encoder = device.createCommandEncoder({ label: phase });
    const pass = timer.timestamp
      ? encoder.beginComputePass({ timestampWrites: { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } })
      : encoder.beginComputePass();
    for (let i = 0; i < reps; i++) {
      pass.setPipeline(rms);
      pass.setBindGroup(0, separate ? resources.oldRms : resources.fusedRms);
      pass.dispatchWorkgroups(resources.T);
      if (separate) {
        pass.setPipeline(residualKernel);
        pass.setBindGroup(0, resources.residual);
        pass.dispatchWorkgroups(resources.T);
      }
    }
    pass.end();
    if (timer.timestamp) {
      encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolved, 0);
      encoder.copyBufferToBuffer(timer.resolved, 0, timer.read, 0, 16);
    }
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (!timer.timestamp) return Math.max(1e-6, (performance.now() - started) / reps);
    await timer.read.mapAsync(GPUMapMode.READ, 0, 16);
    const timestamps = new BigInt64Array(timer.read.getMappedRange(0, 16).slice(0));
    timer.read.unmap();
    return Math.max(1e-6, Number(timestamps[1] - timestamps[0]) / 1e6 / reps);
  }, phase, uncaptured);
}

async function resetTimingInputs(device, resources, input) {
  device.queue.writeBuffer(resources.inputBuffer, 0, input.x);
  device.queue.writeBuffer(resources.oldInput, 0, input.residual);
  device.queue.writeBuffer(resources.fusedOutput, 0, input.residual);
  await device.queue.onSubmittedWorkDone();
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function runPerformance(device, rms, residualKernel, input, timer, warmups, samples, uncaptured) {
  const label = `${input.pattern}-T${input.T}-D${input.D}`;
  const resources = makeResources(device, rms, residualKernel, input, `perf.${label}`);
  const reps = Math.max(1, Math.min(32, Math.floor(2e6 / Math.max(1, input.T * input.D))));
  const separate = [];
  const fused = [];
  try {
    for (let i = 0; i < warmups; i++) {
      await resetTimingInputs(device, resources, input);
      await timed(device, timer, resources, rms, residualKernel, true, reps, uncaptured, `${label}.separate.warmup${i}`);
      await resetTimingInputs(device, resources, input);
      await timed(device, timer, resources, rms, residualKernel, false, reps, uncaptured, `${label}.fused.warmup${i}`);
    }
    for (let i = 0; i < samples; i++) {
      await resetTimingInputs(device, resources, input);
      separate.push(await timed(device, timer, resources, rms, residualKernel, true, reps, uncaptured, `${label}.separate.sample${i}`));
      await resetTimingInputs(device, resources, input);
      fused.push(await timed(device, timer, resources, rms, residualKernel, false, reps, uncaptured, `${label}.fused.sample${i}`));
    }
    const separateMedian = median(separate);
    const fusedMedian = median(fused);
    return { label, T: input.T, D: input.D, pattern: input.pattern, reps, separate, fused, separateMedian, fusedMedian, speedup: separateMedian / fusedMedian };
  } finally {
    resources.destroy();
  }
}

function adapterInfo(gpu) {
  const info = gpu.adapter.info || {};
  return {
    name: gpu.name,
    vendor: info.vendor || null,
    architecture: info.architecture || null,
    device: info.device || null,
    description: info.description || null,
    isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter),
    features: [...gpu.device.features].sort(),
    limits: {
      maxBufferSize: gpu.device.limits.maxBufferSize,
      maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: gpu.device.limits.maxComputeWorkgroupStorageSize,
    },
    powerPreference: gpu.powerPreference,
  };
}

async function main() {
  const seed = numberParam("seed", 0x4b455641, 1, 0xffffffff);
  const warmups = numberParam("warmups", 2, 0, 10);
  const samples = numberParam("samples", 5, 1, 20);
  const gpu = await requestDevice({ features: ["timestamp-query"] });
  if (gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter) throw new Error("software WebGPU adapter is not accepted");
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push({ message: event.error?.message || String(event.error) }));
  const source = await kernelSource();
  const rms = await pipeline(gpu.device, source("gemma4_rms"), "gemma4_rms");
  const residualKernel = await pipeline(gpu.device, source("gemma4_residual"), "gemma4_residual");
  const correctness = { ok: true, cases: [], maxAbs: 0, cpuMaxAbs: 0, bitDifferences: 0, tolerance: { absolute: ABS_TOLERANCE } };
  log(`device: ${gpu.name}; adapter=${adapterInfo(gpu).vendor || "unknown"}; timestamp=${gpu.device.features.has("timestamp-query")}`);
  for (let patternIndex = 0; patternIndex < PATTERNS.length; patternIndex++) {
    for (const D of WIDTHS) for (const T of TOKENS) {
      const input = makeInput(T, D, PATTERNS[patternIndex], (seed + patternIndex * 0x9e3779b9 + D * 17 + T) >>> 0);
      const result = await runCorrectness(gpu.device, rms, residualKernel, input, uncaptured);
      correctness.cases.push(result);
      correctness.ok &&= result.ok;
      correctness.maxAbs = Math.max(correctness.maxAbs, result.maxAbs);
      correctness.cpuMaxAbs = Math.max(correctness.cpuMaxAbs, result.cpuReference.maxAbs);
      correctness.bitDifferences += result.bitDifferences;
      log(`${result.label}: ${result.ok ? "ok" : "FAILED"}; maxAbs=${result.maxAbs.toExponential(3)}; cpu=${result.cpuReference.maxAbs.toExponential(3)}; bitDiff=${result.bitDifferences}`);
    }
  }
  if (!correctness.ok) throw new Error(`Gemma RMS residual correctness failed; maxAbs=${correctness.maxAbs}`);
  const timer = makeTimer(gpu.device);
  const performanceCases = [];
  for (const D of [257, 1536, 2048, 2560]) {
    const input = makeInput(65, D, "signed", (seed + D * 31) >>> 0);
    const result = await runPerformance(gpu.device, rms, residualKernel, input, timer, warmups, samples, uncaptured);
    performanceCases.push(result);
    log(`perf T${result.T} D${result.D}: separate=${result.separateMedian.toFixed(4)} ms fused=${result.fusedMedian.toFixed(4)} ms speedup=${result.speedup.toFixed(3)}x`);
  }
  const metricMs = Math.exp(performanceCases.reduce((sum, item) => sum + Math.log(item.fusedMedian), 0) / performanceCases.length);
  const result = {
    status: "done",
    done: true,
    backend: "webgpu",
    kernel: "gemma4_rms+gemma4_residual",
    metricMs,
    metric: "geometric mean of fused Gemma RMS mode-2 milliseconds for T=65 representative widths",
    method: timer.method,
    seed,
    eps: EPS,
    warmups,
    samples,
    widths: WIDTHS,
    tokens: TOKENS,
    patterns: PATTERNS,
    correctness,
    performance: { cases: performanceCases, separateVsFused: true },
    adapter: adapterInfo(gpu),
  };
  window.gpuBench = result;
  log(`done: ${correctness.cases.length} full-array cases; maxAbs=${correctness.maxAbs}; cpuMaxAbs=${correctness.cpuMaxAbs}; bitDifferences=${correctness.bitDifferences}; metricMs=${metricMs.toFixed(4)}`);
  timer.querySet?.destroy();
  timer.resolved?.destroy();
  timer.read?.destroy();
  gpu.device.destroy();
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma-rms-residual" };
main().catch((error) => {
  const message = `${error?.message || error}\n${error?.stack || ""}`;
  window.gpuBench = { status: "error", done: true, backend: "webgpu", error: message, correctness: { ok: false, errors: [message] } };
  log(`ERROR ${message}`);
});
