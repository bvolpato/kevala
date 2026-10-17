// Bounded exact-output probe: existing decoder split reducer + RMS versus fused WGSL.
import { mmSplits, pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const state = { status: "running", backend: "webgpu", kernel: "gemma4_reduce_rms" };
window.gpuBench = state;
const search = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const U = GPUBufferUsage;
const SENTINEL = 12345.25;
const BUDGET_BYTES = 64 * 1024 * 1024;

function cases() {
  const out = [];
  for (const T of [97, 241]) for (const pattern of ["signed", "zero", "cancellation"]) out.push({ T, D: 1536, splits: 8, pattern, timing: pattern === "signed" });
  for (let splits = 1; splits <= 8; splits++) out.push({ T: 17, D: 1536, splits, pattern: "signed" });
  for (const D of [1, 17, 255, 257, 511, 1535, 1537, 2048]) for (const T of [1, 17]) out.push({ T, D, splits: T === 1 ? 1 : 8, pattern: "signed" });
  for (const D of [1, 257, 1536, 2048]) out.push({ T: 17, D, splits: 8, pattern: "zero" });
  for (const weights of ["negative", "zero"]) out.push({ T: 17, D: 1536, splits: 8, pattern: "signed", weights });
  for (const mode of [0, 1]) out.push({ T: 17, D: 1536, splits: 8, pattern: "signed", mode, ...(mode === 0 ? { weights: "poison" } : {}) });
  for (const invalid of [{ D: 0 }, { D: 2049 }, { splits: 0 }, { splits: 9 }, { mode: 3 }]) out.push({ T: 1, D: 1536, splits: 8, pattern: "signed", invalid });
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function sizes(c) {
  const floats = c.T * c.D, matrixBytes = floats * 4, outputBytes = matrixBytes + 16;
  const partBytes = (floats * c.splits + 4) * 4;
  // Include GPU buffers, live CPU fixture/readback arrays, and one full PART upload reserve.
  const estimateBytes = 3 * partBytes + 2 * c.D * 4 + matrixBytes + 15 * outputBytes + 2048;
  if (estimateBytes > BUDGET_BYTES) throw new Error(`case estimate ${estimateBytes} exceeds ${BUDGET_BYTES}`);
  return { floats, matrixBytes, outputBytes, partBytes, estimateBytes };
}

function fixture(c, size) {
  const part = new Float32Array(size.partBytes / 4).fill(NaN);
  const cancellation = [1048576, 0.015625, -1048576, 0.00737, -0.25, 0.1731, -0.0317, 0.413];
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.floats; i++) {
    const scale = c.pattern === "cancellation" ? cancellation[split] : (split % 2 ? -1 : 1) * (1.337 + split * 0.193);
    const value = c.pattern === "zero" ? i % 2 ? -0 : 0 : c.pattern === "cancellation" ? ((i % 13) - 6) / 8 : Math.sin(i * 0.017 + split * 0.137) * 0.731 + Math.cos(i * 0.031 - split * 0.193) * 0.417;
    part[split * size.floats + i] = value * scale;
  }
  const raw = new Float32Array(size.floats);
  for (let split = 0; split < c.splits; split++) for (let i = 0; i < size.floats; i++) raw[i] = Math.fround(raw[i] + part[split * size.floats + i]);
  const weights = Float32Array.from({ length: c.D }, (_, i) => c.weights === "poison" ? NaN : c.weights === "zero" ? 0 : c.weights === "negative" ? -0.731 - 0.317 * Math.abs(Math.sin(i * 0.217)) : i % 17 === 16 ? 0 : 0.731 + 1.317 * Math.sin(i * 0.217 - 0.3));
  const residual = new Float32Array(size.outputBytes / 4);
  for (let i = 0; i < size.floats; i++) residual[i] = c.pattern === "zero" ? i % 3 ? 0 : -0 : 0.473 * Math.sin(i * 0.137 - 0.417);
  residual.fill(SENTINEL, size.floats);
  return { part, raw, weights, residual, eps: 1e-6 };
}

function candidateSource(raw, baseline, subgroups) {
  const start = baseline.indexOf("struct P {");
  if (start < 0) throw new Error("rendered RMS parameters were not found");
  const prelude = baseline.slice(0, start);
  if (!prelude.includes("struct Globals {") || !prelude.includes("var<uniform> g:") || subgroups && !prelude.includes("fn ordered_sum32(")) throw new Error("rendered RMS prelude does not supply the required globals/reduction helper");
  const out = [], stack = [];
  let enabled = true;
  for (const line of raw.split("\n")) {
    const directive = line.trim();
    if (directive === "//#if SUBGROUPS") {
      stack.push({ parent: enabled, alternate: false }); enabled &&= subgroups;
    } else if (directive === "//#else") {
      const top = stack.at(-1);
      if (!top || top.alternate) throw new Error("invalid candidate conditional");
      top.alternate = true; enabled = top.parent && !subgroups;
    } else if (directive === "//#endif") {
      const top = stack.pop();
      if (!top) throw new Error("unmatched candidate endif");
      enabled = top.parent;
    } else if (directive === "//#include common" || directive === "//#include ordered_sum") {
      // Reuse these exact definitions from the rendered baseline, including its subgroup helper.
    } else if (directive.startsWith("//#")) throw new Error(`unsupported candidate directive ${directive}`);
    else if (enabled) out.push(line);
  }
  if (stack.length) throw new Error("unterminated candidate conditional");
  return prelude + out.join("\n");
}

function compare(label, expected, actual) {
  const oldBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  const newBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let bitDifferences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`${label} nonfinite value at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (oldBits[i] !== newBits[i]) {
      bitDifferences++;
      firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: oldBits[i], actualBits: newBits[i] };
    }
  }
  return { label, ok: bitDifferences === 0, bitExact: bitDifferences === 0, bitDifferences, maxAbs, firstDifference, checkedOutputs: expected.length };
}

async function checkCase(gpu, kernels, getReduce, c) {
  const { device } = gpu;
  const size = sizes(c), input = fixture(c, size);
  const target = c.timing ? 256 : Math.ceil(c.D / 64) * c.splits;
  if (mmSplits(1, c.D, 6144, target, 64, 64) !== c.splits) throw new Error("baseline split count does not match explicit candidate count");
  const reduce = await getReduce(target);
  const owned = [];
  let allocatedBytes = 0, profiler;
  const make = (bytes, usage, data, label) => {
    const count = Math.max(16, bytes);
    if (allocatedBytes + count > BUDGET_BYTES) throw new Error("GPU buffer budget exceeded");
    const buffer = device.createBuffer({ label, size: count, usage });
    allocatedBytes += count; owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (values) => {
    const data = new ArrayBuffer(16), ints = new Uint32Array(data), floats = new Float32Array(data);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(16, U.UNIFORM | U.COPY_DST, data);
  };
  const storage = (data, label) => make(data.byteLength, U.STORAGE | U.COPY_DST, data, label);
  const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  try {
    const globals = uniform([c.T, 0, 0, 0]);
    const pReduce = uniform([c.D, 6144, 0, 0]);
    const pRms = uniform([c.D, { f: input.eps }, c.mode ?? 2, 0]);
    const pFused = uniform([c.invalid?.D ?? c.D, c.invalid?.splits ?? c.splits, { f: input.eps }, c.invalid?.mode ?? c.mode ?? 2]);
    const PART = storage(input.part, "partial planes with poisoned tail"), W = storage(input.weights, "signed norm weights");
    const bias = storage(new Float32Array(4).fill(NaN), "disabled bias");
    const poison = new Float32Array(size.outputBytes / 4).fill(NaN);
    poison.fill(SENTINEL, size.floats);
    const raw = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, "baseline branch");
    const oldY = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, input.residual, "baseline residual");
    const fusedY = make(size.outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, input.residual, "fused residual");
    const outputs = [raw, oldY, fusedY];
    const readback = make(size.outputBytes * outputs.length, U.COPY_DST | U.MAP_READ);
    const reduceBind = group(reduce, [globals, pReduce, PART, bias, raw]);
    const rmsBind = group(kernels.rms, [globals, pRms, raw, W, oldY]);
    const fusedBind = group(kernels.fused, [globals, pFused, PART, W, fusedY]);
    const dispatch = (pass, name) => {
      if (name === "baseline") {
        pass.setPipeline(reduce); pass.setBindGroup(0, reduceBind); pass.dispatchWorkgroups(Math.ceil(size.floats / 256));
        pass.setPipeline(kernels.rms); pass.setBindGroup(0, rmsBind); pass.dispatchWorkgroups(c.T);
      } else {
        pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(c.T);
      }
    };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch(pass, "baseline"); dispatch(pass, "candidate"); pass.end();
    outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * size.outputBytes, size.outputBytes));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const values = outputs.map((_, i) => {
      const base = i * poison.length;
      for (let j = size.floats; j < poison.length; j++) if (actual[base + j] !== SENTINEL) throw new Error(`output ${i} overwrote sentinel ${j - size.floats}`);
      return actual.subarray(base, base + size.floats);
    });
    const rawCheck = compare("original reducer vs scalar F32 reference", input.raw, values[0]);
    if (!rawCheck.ok) throw new Error(`baseline reducer mismatch: ${JSON.stringify(rawCheck)}`);
    const parity = c.invalid ? compare("rejected candidate leaves residual unchanged", input.residual.subarray(0, size.floats), values[2]) : compare("baseline vs candidate residual", values[1], values[2]);
    if (!parity.ok) throw new Error(`reduce/RMS ${JSON.stringify(c)} mismatch: ${JSON.stringify(parity)}`);
    const timings = { baseline: [], candidate: [] };
    let repeatedParity = null;
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 2);
      const repeats = 16;
      for (let sample = 0; sample < 8; sample++) {
        // Reset both in-place residuals before each pair. Each path then performs equal additions.
        device.queue.writeBuffer(oldY, 0, input.residual); device.queue.writeBuffer(fusedY, 0, input.residual);
        const encoder = device.createCommandEncoder();
        for (const name of sample % 2 ? ["candidate", "baseline"] : ["baseline", "candidate"]) {
          const pass = profiler.pass(encoder, name);
          for (let n = 0; n < repeats; n++) dispatch(pass, name);
          pass.end();
        }
        profiler.finish(encoder); device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const name of ["baseline", "candidate"]) timings[name].push(measured[name] / repeats);
      }
      const encoder = device.createCommandEncoder();
      outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * size.outputBytes, size.outputBytes));
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const repeated = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const stride = size.outputBytes / 4;
      repeatedParity = compare("residual after repeated timed additions", repeated.subarray(stride, stride + size.floats), repeated.subarray(stride * 2, stride * 2 + size.floats));
      if (!repeatedParity.ok) throw new Error(`repeated residual additions differ: ${JSON.stringify(repeatedParity)}`);
      for (let i = 0; i < outputs.length; i++) for (let j = size.floats; j < stride; j++) if (repeated[i * stride + j] !== SENTINEL) throw new Error(`timed output ${i} overwrote sentinel`);
    }
    return { ...c, K: 6144, logicalSplitT: 1, splitTarget: target, rawCheck, parity, repeatedParity, outputTailGuard: true, ownedGpuBufferBytes: allocatedBytes, estimatedCaseBufferBytes: size.estimateBytes, timings, ok: true };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now(), subgroups = search.get("subgroups") === "1", f16Feature = search.get("f16") === "1";
  const features = ["timestamp-query", ...(subgroups ? ["subgroups"] : []), ...(f16Feature ? ["shader-f16"] : [])];
  const gpu = await requestDevice({ features });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const info = gpu.adapter.info || {};
    if (subgroups && (!gpu.device.features.has("subgroups") || info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32)) throw new Error("subgroup RMS probe requires fixed32-lane subgroups");
    if (f16Feature && !gpu.device.features.has("shader-f16")) throw new Error("requested shader-f16 feature is unavailable");
    const source = await kernelSource();
    const baseline = source("gemma4_rms", { subgroups });
    const url = new URL("./gemma4-reduce-rms.wgsl", import.meta.url);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`candidate WGSL could not load: HTTP${response.status} ${url.href}`);
    const code = candidateSource(await response.text(), baseline, subgroups);
    const kernels = await withGpuErrors(gpu, async () => ({
      rms: await pipeline(gpu.device, baseline, "gemma4_rms"),
      fused: await pipeline(gpu.device, code, "gemma4_reduce_rms_probe"),
    }), "compiling reducer/RMS probe");
    const reducers = new Map();
    const getReduce = (splitTarget) => {
      if (!reducers.has(splitTarget)) reducers.set(splitTarget, pipeline(gpu.device, source("reduce_gemma_batch", { rows: 4, groups: 1, f16: false, splitTarget }), `reduce_batch_target${splitTarget}`));
      return reducers.get(splitTarget);
    };
    const results = [];
    for (const c of cases()) {
      const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, getReduce, c), "checking decoder reducer/RMS");
      results.push(result); log(JSON.stringify(result));
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started, subgroups, f16Feature, arithmetic: "F32 PART reduction and RMS; synthetic PART does not execute a matmul",
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(info.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features], subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, workgroupStorageBytes: gpu.device.limits.maxComputeWorkgroupStorageSize },
      compile: { ok: true, source: "raw candidate WGSL with current WASM-rendered RMS common/ordered_sum prelude", candidateUrl: url.href, workgroupSize: 256, sharedBytes: 9216 },
      resourceBounds: { maxCaseBufferEstimateBytes: Math.max(...results.map((c) => c.estimatedCaseBufferBytes)), budgetBytes: BUDGET_BYTES, sequentialCases: true },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code };
  log(error.stack || error);
});
