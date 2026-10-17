// Require exact residual and next-norm outputs against two original vision norm dispatches.
import { pipeline, Profiler, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const state = { status: "running", backend: "webgpu", kernel: "gemma4_vision_residual_norm" };
window.gpuBench = state;
const search = new URLSearchParams(location.search);
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const U = GPUBufferUsage;
const SENTINEL = 12345.25;

function cases() {
  const out = [];
  for (const width of [32, 64, 256, 768, 1024]) {
    for (let flags = 0; flags < 8; flags++) for (const clampAfter of [false, true]) {
      out.push({ rows: (flags + Number(clampAfter)) % 2 ? 17 : 1, width, weighted: Boolean(flags & 1), residual: Boolean(flags & 2), weightedNext: Boolean(flags & 4), clampAfter, clampAfterNext: !clampAfter,
        bounds: [-0.75, 1.25], boundsNext: [-0.1875, 0.3125], pattern: "signed" });
    }
    for (const [bounds, boundsNext, clampAfter, clampAfterNext] of [
      [[-Infinity, Infinity], [-Infinity, Infinity], false, false],
      [[-Infinity, Infinity], [-Infinity, Infinity], true, true],
      [[-0.75, Infinity], [-Infinity, 0.3125], false, true],
      [[-Infinity, 1.25], [-0.1875, Infinity], true, false],
    ]) out.push({ rows: 17, width, weighted: true, residual: true, weightedNext: true, bounds, boundsNext, clampAfter, clampAfterNext, pattern: "signed" });
  }
  for (const width of [768, 1024]) for (const rows of [1, 17]) out.push({ rows, width, weighted: true, residual: true, weightedNext: true, bounds: [-Infinity, Infinity], boundsNext: [-Infinity, Infinity], clampAfter: false, clampAfterNext: false, pattern: "zero" });
  for (const rows of [315, 567]) for (const clippedNext of [false, true]) out.push({ rows, width: 768, weighted: true, residual: true, weightedNext: true,
    bounds: [-0.75, 1.25], boundsNext: clippedNext ? [-0.1875, 0.3125] : [-Infinity, Infinity], clampAfter: false, clampAfterNext: clippedNext, pattern: "signed", timing: true });
  return search.get("cases") === "timing" ? out.filter((c) => c.timing) : out;
}

function fixture(c) {
  const zero = c.pattern === "zero";
  const x = Float32Array.from({ length: c.rows * c.width }, (_, i) => zero ? i % 2 ? -0 : 0 : 13.37 * Math.sin(i * 0.173 + 0.127) + 7.19 * Math.cos(i * 0.031));
  const residual = Float32Array.from({ length: x.length }, (_, i) => !c.residual ? NaN : zero ? i % 3 ? 0 : -0 : 0.473 * Math.sin(i * 0.137 - 0.417));
  const post = Float32Array.from({ length: c.width }, (_, i) => !c.weighted ? NaN : i % 17 === 0 ? 0 : 0.731 + 1.317 * Math.sin(i * 0.217 - 0.3));
  const next = Float32Array.from({ length: c.width }, (_, i) => !c.weightedNext ? NaN : i % 19 === 0 ? 0 : 0.913 - 1.173 * Math.cos(i * 0.193 + 0.2));
  return { x, residual, post, next, eps: 3.71e-5, epsNext: 1.337e-6 };
}

function compare(label, oldValues, candidate) {
  const oldBits = new Uint32Array(oldValues.buffer, oldValues.byteOffset, oldValues.length);
  const newBits = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let bitDifferences = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < oldValues.length; i++) {
    if (!Number.isFinite(oldValues[i]) || !Number.isFinite(candidate[i])) throw new Error(`${label} nonfinite output at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(oldValues[i] - candidate[i]));
    if (oldBits[i] !== newBits[i]) {
      bitDifferences++;
      firstDifference ||= { index: i, old: oldValues[i], candidate: candidate[i], oldBits: oldBits[i], candidateBits: newBits[i] };
    }
  }
  return { label, ok: bitDifferences === 0, bitExact: bitDifferences === 0, bitDifferences, maxAbs, firstDifference, checkedOutputs: oldValues.length };
}

async function checkCase(gpu, kernels, c) {
  const { device } = gpu;
  const input = fixture(c);
  const owned = [];
  let profiler;
  const make = (size, usage, data, label) => {
    const buffer = device.createBuffer({ label, size: Math.max(16, size), usage });
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const uniform = (values) => {
    const data = new ArrayBuffer(Math.max(16, values.length * 4));
    const ints = new Uint32Array(data), floats = new Float32Array(data);
    values.forEach((value, i) => typeof value === "object" ? floats[i] = value.f : ints[i] = value);
    return make(data.byteLength, U.UNIFORM | U.COPY_DST, data);
  };
  const storage = (data, label) => make(data.byteLength, U.STORAGE | U.COPY_DST, data, label);
  const group = (kernel, buffers) => device.createBindGroup({ layout: kernel.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const f32 = (f) => ({ f });
  try {
    const params = [c.rows, c.width, Number(c.weighted), Number(c.residual), f32(input.eps), ...c.bounds.map(f32), Number(c.clampAfter)];
    const pPost = uniform(params);
    const pNext = uniform([c.rows, c.width, Number(c.weightedNext), 0, f32(input.epsNext), ...c.boundsNext.map(f32), Number(c.clampAfterNext)]);
    const pFused = uniform([...params, Number(c.weightedNext), Number(c.clampAfterNext), 0, 0, f32(input.epsNext), ...c.boundsNext.map(f32), 0]);
    const X = storage(input.x, "projected input"), Wpost = storage(input.post, "post weights"), R = storage(input.residual, "residual input"), Wnext = storage(input.next, "next weights");
    const outputFloats = input.x.length + 4, outputBytes = outputFloats * 4;
    const poison = new Float32Array(outputFloats).fill(NaN);
    poison.fill(SENTINEL, input.x.length);
    const outputs = ["old residual", "old normalized", "fused residual", "fused normalized"].map((label) => make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, poison, label));
    const postBind = group(kernels.norm, [pPost, X, Wpost, R, outputs[0]]);
    const nextBind = group(kernels.norm, [pNext, outputs[0], Wnext, R, outputs[1]]);
    const fusedBind = group(kernels.fused, [pFused, X, Wpost, R, Wnext, outputs[2], outputs[3]]);
    const readback = make(outputBytes * outputs.length, U.COPY_DST | U.MAP_READ);
    const dispatch = (pass, name) => {
      if (name === "old") {
        pass.setPipeline(kernels.norm); pass.setBindGroup(0, postBind); pass.dispatchWorkgroups(c.rows);
        pass.setBindGroup(0, nextBind); pass.dispatchWorkgroups(c.rows);
      } else {
        pass.setPipeline(kernels.fused); pass.setBindGroup(0, fusedBind); pass.dispatchWorkgroups(c.rows);
      }
    };
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch(pass, "old"); dispatch(pass, "fused"); pass.end();
    outputs.forEach((buffer, i) => encoder.copyBufferToBuffer(buffer, 0, readback, i * outputBytes, outputBytes));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const values = outputs.map((_, i) => {
      const base = i * outputFloats;
      for (let j = input.x.length; j < outputFloats; j++) if (actual[base + j] !== SENTINEL) throw new Error(`output ${i} overwrote sentinel ${j - input.x.length}`);
      return actual.subarray(base, base + input.x.length);
    });
    const residual = compare("residual", values[0], values[2]);
    const normalized = compare("normalized", values[1], values[3]);
    if (!residual.ok || !normalized.ok) throw new Error(`vision residual/norm ${JSON.stringify(c)} differs: ${JSON.stringify({ residual, normalized })}`);
    const timings = { old: [], fused: [] };
    if (c.timing && search.get("timing") !== "0" && device.features.has("timestamp-query")) {
      profiler = new Profiler(device, 2);
      const repeats = 16;
      for (let sample = 0; sample < 8; sample++) {
        const encoder = device.createCommandEncoder();
        for (const name of sample % 2 ? ["fused", "old"] : ["old", "fused"]) {
          const pass = profiler.pass(encoder, name);
          for (let n = 0; n < repeats; n++) dispatch(pass, name);
          pass.end();
        }
        profiler.finish(encoder);
        device.queue.submit([encoder.finish()]);
        const measured = await profiler.collect();
        if (sample >= 2) for (const name of ["old", "fused"]) timings[name].push(measured[name] / repeats);
      }
    }
    const jsonBound = (value) => Number.isFinite(value) ? value : String(value);
    return { ...c, bounds: c.bounds.map(jsonBound), boundsNext: c.boundsNext.map(jsonBound), residual, normalized, outputTailGuard: true, disabledInputsPoisoned: true, timings, ok: true };
  } finally { profiler?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const started = performance.now();
  const gpu = await requestDevice({ baseline: true });
  const uncaptured = [];
  gpu.device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    const source = await kernelSource();
    const kernels = await withGpuErrors(gpu, async () => ({
      norm: await pipeline(gpu.device, source("gemma4_vision_norm"), "gemma4_vision_norm"),
      fused: await pipeline(gpu.device, source("gemma4_vision_residual_norm"), "gemma4_vision_residual_norm"),
    }), "compiling vision residual/norm guard");
    const results = [];
    for (const c of cases()) {
      const result = await withGpuErrors(gpu, () => checkCase(gpu, kernels, c), "checking vision residual/norm");
      results.push(result); log(JSON.stringify(result));
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    window.gpuBench = { ...state, status: "done", done: true, metricMs: performance.now() - started,
      adapter: { name: gpu.name, isFallbackAdapter: Boolean(gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter), features: [...gpu.device.features], workgroupStorageBytes: gpu.device.limits.maxComputeWorkgroupStorageSize },
      compile: { ok: true, source: "current kevala-base.wasm rendered production WGSL", workgroupSize: 256, sharedBytes: 5120 },
      correctness: { ok: true, bitExact: true, tolerance: { absolute: 0, relative: 0 }, cases: results } };
  } finally { gpu.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { ...state, status: "error", done: true, error: String(error?.message || error), code: error?.code };
  log(error.stack || error);
});
