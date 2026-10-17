// Exact-output guard for the two adjacent Gemma RMS fusion kernels.
import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const U = GPUBufferUsage;
const EPS = 1e-6;
const WIDTHS = [32, 257, 1536, 2048, 2560];
const TOKENS = [1, 2, 5, 42, 89];
const PATTERNS = ["signed", "large", "nonpower"];
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state = Math.imul(state ^ (state >>> 16), 0x45d9f3b);
    state ^= state >>> 16;
    return (state >>> 0) / 0x100000000;
  };
}

function makeInput(T, D, pattern, seed) {
  const random = rng(seed);
  const x = new Float32Array(T * D);
  const residual = new Float32Array(T * D);
  const wpost = new Float32Array(D);
  const wnext = new Float32Array(D);
  for (let i = 0; i < x.length; i++) {
    const a = Math.floor(random() * 17) - 8;
    const b = Math.floor(random() * 17) - 8;
    x[i] = pattern === "large" ? a * 128 : pattern === "nonpower" ? a * 0.00437 : a / 16;
    residual[i] = pattern === "large" ? b * 4 : pattern === "nonpower" ? b * 0.0713 : b / 32;
  }
  for (let d = 0; d < D; d++) {
    const a = Math.floor(random() * 13) - 6;
    const b = Math.floor(random() * 13) - 6;
    wpost[d] = pattern === "large" ? a * 4 : pattern === "nonpower" ? a * 0.1837 : a / 8;
    wnext[d] = pattern === "large" ? b * 4 : pattern === "nonpower" ? b * 0.2719 : b / 8;
  }
  const scalar = new Float32Array([pattern === "large" ? -3.25 : pattern === "nonpower" ? -1.337 : -0.625]);
  const eps = pattern === "nonpower" ? 3.71e-5 : EPS;
  return { T, D, pattern, x, residual, wpost, wnext, scalar, eps };
}

function makeBuffer(device, bytes, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, bytes), usage });
}

function storage(device, values, label, extra = U.COPY_DST | U.COPY_SRC, minBytes = 16) {
  const bytes = Math.max(minBytes, values?.byteLength || 0, 16);
  const buffer = makeBuffer(device, bytes, U.STORAGE | extra, label);
  if (values) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
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

function group(device, kernel, buffers, label) {
  return device.createBindGroup({
    label,
    layout: kernel.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

function dispatch(encoder, kernel, bindGroup, T) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(kernel);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(T);
  pass.end();
}

async function readBuffer(device, source, length, label) {
  const bytes = length * 4;
  const readback = makeBuffer(device, bytes, U.MAP_READ | U.COPY_DST, `${label}.readback`);
  try {
    const encoder = device.createCommandEncoder({ label: `${label}.copy` });
    encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ, 0, bytes);
    const result = new Float32Array(readback.getMappedRange(0, bytes).slice(0));
    readback.unmap();
    return result;
  } finally {
    readback.destroy();
  }
}

function compare(label, oldValues, fusedValues) {
  const oldBits = new Uint32Array(oldValues.buffer, oldValues.byteOffset, oldValues.length);
  const fusedBits = new Uint32Array(fusedValues.buffer, fusedValues.byteOffset, fusedValues.length);
  let bitDifferences = 0;
  let firstDifference = null;
  let maxAbs = 0;
  let finite = true;
  for (let i = 0; i < oldValues.length; i++) {
    if (!Number.isFinite(oldValues[i]) || !Number.isFinite(fusedValues[i])) finite = false;
    maxAbs = Math.max(maxAbs, Math.abs(oldValues[i] - fusedValues[i]));
    if (oldBits[i] !== fusedBits[i]) {
      bitDifferences++;
      if (firstDifference === null) {
        firstDifference = {
          index: i,
          old: oldValues[i],
          fused: fusedValues[i],
          oldBits: `0x${oldBits[i].toString(16).padStart(8, "0")}`,
          fusedBits: `0x${fusedBits[i].toString(16).padStart(8, "0")}`,
        };
      }
    }
  }
  return { label, valuesCompared: oldValues.length, finite, bitExact: bitDifferences === 0, bitDifferences, maxAbs, firstDifference, ok: finite && bitDifferences === 0 };
}

function makeGlobals(device, T, label) {
  const globals = makeBuffer(device, 16, U.UNIFORM | U.COPY_DST, `${label}.globals`);
  device.queue.writeBuffer(globals, 0, new Uint32Array([T, 0, 0, 0]));
  return globals;
}

async function checkResidualNorm(device, kernels, input, tag) {
  const { T, D, x, residual, wpost, wnext, eps } = input;
  const label = `${tag}-T${T}-D${D}`;
  const owned = [];
  const keep = (buffer) => (owned.push(buffer), buffer);
  try {
    const globals = keep(makeGlobals(device, T, label));
    const pRms1 = keep(uniform(device, [D, { f: eps }, 1, 0], `${label}.rms-mode1`));
    const pRms2 = keep(uniform(device, [D, { f: eps }, 2, 0], `${label}.rms-mode2`));
    const pFused = keep(uniform(device, [D, { f: eps }, 2, 0], `${label}.fused`));
    const xBuf = keep(storage(device, x, `${label}.x`));
    const wpBuf = keep(storage(device, wpost, `${label}.wpost`));
    const wnBuf = keep(storage(device, wnext, `${label}.wnext`));
    const oldY = keep(storage(device, residual, `${label}.old-y`));
    const newY = keep(storage(device, residual, `${label}.new-y`));
    const outputBytes = T * D * Float32Array.BYTES_PER_ELEMENT;
    const oldH = keep(storage(device, null, `${label}.old-h`, U.COPY_DST | U.COPY_SRC, outputBytes));
    const newH = keep(storage(device, null, `${label}.new-h`, U.COPY_DST | U.COPY_SRC, outputBytes));
    const first = group(device, kernels.rms, [globals, pRms2, xBuf, wpBuf, oldY], `${label}.old-rms-mode2`);
    const second = group(device, kernels.rms, [globals, pRms1, oldY, wnBuf, oldH], `${label}.old-rms-next`);
    const fused = group(device, kernels.residualNorm, [globals, pFused, xBuf, wpBuf, newY, wnBuf, newH], `${label}.fused`);

    const encoder = device.createCommandEncoder({ label });
    dispatch(encoder, kernels.rms, first, T);
    dispatch(encoder, kernels.rms, second, T);
    dispatch(encoder, kernels.residualNorm, fused, T);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const [oldResidual, fusedResidual, oldOutput, fusedOutput] = await Promise.all([
      readBuffer(device, oldY, T * D, `${label}.old-y`),
      readBuffer(device, newY, T * D, `${label}.new-y`),
      readBuffer(device, oldH, T * D, `${label}.old-h`),
      readBuffer(device, newH, T * D, `${label}.new-h`),
    ]);
    return [
      compare(`${label}.residual`, oldResidual, fusedResidual),
      compare(`${label}.next-rms`, oldOutput, fusedOutput),
    ];
  } finally {
    for (const buffer of owned) buffer.destroy();
  }
}

async function checkRmsScale(device, kernels, input, tag) {
  const { T, D, x, residual, wpost, scalar, eps } = input;
  const label = `${tag}-T${T}-D${D}`;
  const owned = [];
  const keep = (buffer) => (owned.push(buffer), buffer);
  try {
    const globals = keep(makeGlobals(device, T, label));
    const pRms2 = keep(uniform(device, [D, { f: eps }, 2, 0], `${label}.rms-mode2`));
    const pResidualScale = keep(uniform(device, [D, 1, 0, 0], `${label}.residual-scale`));
    const pFused = keep(uniform(device, [D, { f: eps }, 2, 0], `${label}.fused`));
    const xBuf = keep(storage(device, x, `${label}.x`));
    const wBuf = keep(storage(device, wpost, `${label}.weight`));
    const oldY = keep(storage(device, residual, `${label}.old-y`));
    const newY = keep(storage(device, residual, `${label}.new-y`));
    const scalarBuf = keep(storage(device, scalar, `${label}.scalar`));
    const unused = keep(storage(device, new Float32Array([0]), `${label}.unused`));
    const oldRms = group(device, kernels.rms, [globals, pRms2, xBuf, wBuf, oldY], `${label}.old-rms`);
    const oldScale = group(device, kernels.residual, [globals, pResidualScale, oldY, unused, scalarBuf], `${label}.old-scale`);
    const fused = group(device, kernels.rmsScale, [globals, pFused, xBuf, wBuf, newY, scalarBuf], `${label}.fused`);

    const encoder = device.createCommandEncoder({ label });
    dispatch(encoder, kernels.rms, oldRms, T);
    dispatch(encoder, kernels.residual, oldScale, T);
    dispatch(encoder, kernels.rmsScale, fused, T);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return [compare(label, await readBuffer(device, oldY, T * D, `${label}.old`), await readBuffer(device, newY, T * D, `${label}.new`))];
  } finally {
    for (const buffer of owned) buffer.destroy();
  }
}

async function main() {
  const started = performance.now();
  const subgroups = new URLSearchParams(location.search).get("subgroups") === "1";
  window.gpuBench = { ...window.gpuBench, status: "running", backend: "webgpu", check: "gemma4-rms-fusions", subgroups, cases: 0, results: [] };
  const gpu = await requestDevice({ features: subgroups ? ["subgroups"] : [] });
  if (subgroups && !gpu.subgroup32) throw new Error("RMS subgroup path requires exactly 32-lane subgroups");
  if (gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter) throw new Error("software WebGPU adapter is not accepted");
  const source = await kernelSource();
  const kernels = {
    rms: await pipeline(gpu.device, source("gemma4_rms", { subgroups }), "gemma4_rms"),
    residual: await pipeline(gpu.device, source("gemma4_residual"), "gemma4_residual"),
    residualNorm: await pipeline(gpu.device, source("gemma4_rms_residual_norm", { subgroups }), "gemma4_rms_residual_norm"),
    rmsScale: await pipeline(gpu.device, source("gemma4_rms_scale", { subgroups }), "gemma4_rms_scale"),
  };
  const results = [];
  let caseIndex = 0;
  log(`device: ${gpu.name}; adapter=${gpu.adapter.info?.vendor || "unknown"}; subgroups=${subgroups}`);
  for (const pattern of PATTERNS) for (const D of WIDTHS) for (const T of TOKENS) {
    const input = makeInput(T, D, pattern, (0x4b455641 + caseIndex * 0x9e3779b9) >>> 0);
    results.push(...await checkResidualNorm(gpu.device, kernels, input, `residual-norm-${pattern}`));
    results.push(...await checkRmsScale(gpu.device, kernels, input, `rms-scale-${pattern}`));
    const recent = results.slice(-3);
    window.gpuBench = { ...window.gpuBench, status: "running", cases: results.length, results };
    for (const result of recent) log(`${result.label}: ${result.ok ? "bit-exact" : "FAILED"}; ${JSON.stringify(result)}`);
    const failed = recent.find((result) => !result.ok);
    if (failed) {
      const failure = { ...window.gpuBench, status: "error", done: true, failure: failed, error: `Gemma RMS fusion mismatch in ${failed.label}` };
      window.gpuBench = failure;
      throw new Error(`${failure.error}: ${JSON.stringify(failed)}`);
    }
    caseIndex++;
  }
  const result = {
    status: "done",
    done: true,
    backend: "webgpu",
    adapter: { name: gpu.name, isFallbackAdapter: Boolean(gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features] },
    check: "gemma4-rms-fusions",
    subgroups,
    cases: results.length,
    bitExact: results.every((item) => item.bitExact),
    finite: results.every((item) => item.finite),
    correctness: { ok: results.every((item) => item.ok), tolerance: 0 },
    metricMs: performance.now() - started,
    results,
  };
  window.gpuBench = result;
  log(`done: ${results.length} complete-output checks; all finite and bit-exact`);
  gpu.device.destroy();
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-rms-fusions" };
main().catch((error) => {
  const message = `${error?.message || error}\n${error?.stack || ""}`;
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, backend: "webgpu", error: message };
  log(`ERROR ${message}`);
});
