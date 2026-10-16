// Tiny fixture by default. To check the extracted actual E2B companion, add
// ?fixture=../tmp/golden-gemma4-vision-e2b.json&pack=../tmp/gemma-4-e2b-vision-bf16.kevala
// Add &profile=1 for warmed wall-clock and per-kernel GPU timing samples.
import { Profiler, requestDevice } from "../js/src/gpu.js";
import { GpuGemma4Vision, gpuVisionProjection } from "../js/src/gpu-gemma4-vision.js";
import { gpuLayouts, readPackHead } from "../js/src/pack-layout.js";
import { PieceSink } from "../js/src/source.js";
import { kernelSource } from "./wgsl.js";

const log = (value) => { document.getElementById("log").textContent += `${value}\n`; };
const bytes = (data) => Uint8Array.from(atob(data), (ch) => ch.charCodeAt(0));
const floats = (data) => new Float32Array(bytes(data).buffer);
const median = (values) => {
  const sorted = values.toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

function compare(actual, expected, tolerance, id) {
  if (actual.length !== expected.length) throw new Error(`output length ${actual.length} != ${expected.length}`);
  let maxAbs = 0, error2 = 0, reference2 = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) throw new Error(`nonfinite output at ${i}`);
    const delta = actual[i] - expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(delta));
    error2 += delta * delta;
    reference2 += expected[i] * expected[i];
  }
  const relativeL2 = Math.sqrt(error2 / reference2);
  if (maxAbs > tolerance) throw new Error(`vision ${id} maxAbs ${maxAbs} exceeds ${tolerance}`);
  return { maxAbs, relativeL2 };
}

async function sample(model, image, expected, tolerance, id, profile) {
  model.lastProfile = null;
  const started = performance.now();
  const actual = await model.forward(image);
  const wallMs = performance.now() - started;
  const correctness = compare(actual, expected, tolerance, id);
  const gpu = model.lastProfile || null;
  if (profile && (!gpu || !Object.keys(gpu).length)) throw new Error(`vision ${id} did not return GPU timestamp samples`);
  return { wallMs, ...correctness, ...(profile ? { gpu } : {}) };
}

async function projectionCheck(runtime, item) {
  const { descriptor, expected, cpu, id } = item;
  const projection = gpuVisionProjection(descriptor, runtime.device, expected.byteLength);
  if (!projection || gpuVisionProjection(descriptor, {}, expected.byteLength) || gpuVisionProjection(descriptor, runtime.device, expected.byteLength + 4)) throw new Error("GPU projection ownership or size validation failed");
  const readback = runtime.device.createBuffer({ size: expected.byteLength + 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = runtime.device.createCommandEncoder();
    encoder.copyBufferToBuffer(projection.buffer, 0, readback, 0, expected.byteLength);
    encoder.copyBufferToBuffer(projection.status, 0, readback, expected.byteLength, 4);
    runtime.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const status = new Uint32Array(mapped, expected.byteLength, 1)[0];
    const actual = new Float32Array(mapped.slice(0, expected.byteLength));
    readback.unmap();
    if (status !== 0) throw new Error(`finite GPU projection ${id} was flagged`);
    const correctness = compare(actual, expected, item.tolerance, id);
    const cpuDelta = compare(actual, cpu, 0, `${id} GPU versus CPU handoff`).maxAbs;
    descriptor.destroy();
    if (gpuVisionProjection(descriptor, runtime.device, expected.byteLength)) throw new Error("released GPU projection remained valid");
    return { id, ...correctness, cpuDelta, status };
  } finally { readback.destroy(); }
}

async function finiteCheck(model) {
  const device = model.device, usage = GPUBufferUsage;
  const owned = [];
  const buffer = (flags) => {
    const value = device.createBuffer({ size: 16, usage: flags });
    owned.push(value);
    return value;
  };
  try {
    const params = buffer(usage.UNIFORM | usage.COPY_DST);
    const input = buffer(usage.STORAGE | usage.COPY_DST);
    const status = buffer(usage.STORAGE | usage.COPY_SRC | usage.COPY_DST);
    const readback = buffer(usage.MAP_READ | usage.COPY_DST);
    const compute = model.p.finite;
    const group = device.createBindGroup({ layout: compute.getBindGroupLayout(0), entries: [params, input, status].map((value, binding) => ({ binding, resource: { buffer: value } })) });
    device.queue.writeBuffer(params, 0, new Uint32Array([4, 0, 0, 0]));
    const cases = [];
    for (const [id, value, expected] of [["max-finite", 3.4028234663852886e38, 0], ["negative-zero", -0, 0], ["nan", NaN, 1], ["positive-infinity", Infinity, 1], ["negative-infinity", -Infinity, 1]]) {
      device.queue.writeBuffer(input, 0, new Float32Array([1, value, -1, 0]));
      device.queue.writeBuffer(status, 0, new Uint32Array(4));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(compute);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(status, 0, readback, 0, 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Uint32Array(readback.getMappedRange())[0];
      readback.unmap();
      if (actual !== expected) throw new Error(`projection finite check ${id}: ${actual} != ${expected}`);
      cases.push({ id, status: actual });
    }
    return cases;
  } finally { for (const value of owned) value.destroy(); }
}

function tinyPack(fixture) {
  let offset = 0;
  const data = [];
  const tensors = fixture.tensors.map((tensor) => {
    const value = bytes(tensor.data);
    data.push(value);
    const info = { name: tensor.name, shape: tensor.shape, dtype: tensor.dtype, offset, size: value.byteLength };
    offset += value.byteLength;
    return info;
  });
  const header = { config: fixture.config, tensors };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(16 + json.byteLength);
  prefix.set([0x4b, 0x56, 0x4c, 0x41]);
  new DataView(prefix.buffer).setUint32(4, 1, true);
  new DataView(prefix.buffer).setUint32(8, json.byteLength, true);
  prefix.set(json, 16);
  const packed = new Uint8Array(offset);
  offset = 0;
  for (const value of data) { packed.set(value, offset); offset += value.byteLength; }
  return { header, prefix, packed };
}

async function fromPack(runtime, url, options) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`companion HTTP ${response.status}`);
  const reader = response.body.getReader();
  const iterator = { next: () => reader.read() };
  const { head, header, headerBytes } = await readPackHead(iterator);
  const layout = gpuLayouts(header, headerBytes.byteLength, () => true)[1];
  const model = new GpuGemma4Vision(runtime, layout, header, options);
  try {
    const sink = new PieceSink(layout, (dst, chunk) => model.write(dst, chunk));
    let at = head.byteLength;
    sink.push(head, 0);
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      sink.push(value, at);
      at += value.byteLength;
    }
    if (!sink.done) throw new Error("truncated vision companion");
    await model.init();
    return model;
  } catch (error) {
    model.destroy();
    throw error;
  } finally { reader.releaseLock(); }
}

async function main() {
  const params = new URLSearchParams(location.search);
  const fixtureUrl = params.get("fixture") || "../tests/fixtures/gemma4-vision-tiny.json";
  const response = await fetch(fixtureUrl);
  if (!response.ok) throw new Error(`fixture HTTP ${response.status}`);
  const fixture = await response.json();
  const profile = params.get("profile") === "1";
  const checkGpuOutput = params.get("gpu") === "1";
  if (checkGpuOutput && profile) throw new Error("GPU projection handoff checks require profile=0");
  const runtime = await requestDevice({ features: profile ? ["timestamp-query"] : [] });
  if (profile && !runtime.device.features.has("timestamp-query")) throw new Error("WebGPU timestamp-query is unavailable on this adapter");
  runtime.wgsl = await kernelSource(params.get("wasm") || undefined);
  if (params.has("attention")) {
    const response = await fetch(params.get("attention"));
    if (!response.ok) throw new Error(`attention control HTTP ${response.status}`);
    const attention = await response.text();
    const source = runtime.wgsl;
    runtime.wgsl = Object.assign((kernel, spec) => kernel === "gemma4_vision_attention" ? attention : source(kernel, spec), { wasmUrl: source.wasmUrl });
  }
  const options = { compactPatches: params.get("padded") !== "1", submitMode: params.get("submit") || "split", fusedAttention: params.get("fused") !== "0" && !params.has("attention") };
  let model;
  let profiler;
  const heldProjections = [];
  try {
    if (params.has("pack")) model = await fromPack(runtime, params.get("pack"), options);
    else {
      const packed = tinyPack(fixture);
      model = new GpuGemma4Vision(runtime, { prefix: packed.prefix }, packed.header, options);
      // Odd fragments protect the streamed BF16 and F32 scalar loading contract.
      for (let at = 0; at < packed.packed.length; at += 307) model.write(at, packed.packed.subarray(at, at + 307));
      await model.init();
    }
    const cases = [];
    const tolerance = params.has("pack") ? 1e-4 : 3e-5;
    for (const test of fixture.cases) {
      const image = { patches: floats(test.patches), positions: new Int32Array(bytes(test.positions).buffer), softTokens: test.softTokens };
      const expected = floats(test.projected);
      if (!profile) {
        const result = await sample(model, image, expected, tolerance, test.id, false);
        cases.push({ id: test.id, patches: image.positions.length / 2, activePatches: model.shape[0], softTokens: image.softTokens, attentionMode: model.attentionMode, elapsedMs: result.wallMs, maxAbs: result.maxAbs, relativeL2: result.relativeL2 });
        log(`${test.id}: maxAbs=${result.maxAbs}, relativeL2=${result.relativeL2}, ${result.wallMs.toFixed(2)} ms`);
        if (checkGpuOutput) heldProjections.push({ descriptor: await model.forward(image, { gpuOutput: true }), cpu: await model.forward(image), expected, tolerance, id: test.id });
        continue;
      }

      const warmups = [];
      for (let i = 0; i < 5; i++) warmups.push(await sample(model, image, expected, tolerance, test.id, false));
      const baseline = [];
      for (let i = 0; i < 5; i++) baseline.push(await sample(model, image, expected, tolerance, test.id, false));
      profiler = new Profiler(runtime.device);
      model.profiler = profiler;
      const profileWarmup = await sample(model, image, expected, tolerance, test.id, true);
      const profiled = [];
      for (let i = 0; i < 5; i++) profiled.push(await sample(model, image, expected, tolerance, test.id, true));
      const kernelNames = [...new Set(profiled.flatMap((run) => Object.keys(run.gpu)))].sort();
      const kernelMedianMs = Object.fromEntries(kernelNames.map((name) => [name, median(profiled.map((run) => run.gpu[name] || 0))]));
      const baselineMedianMs = median(baseline.map((run) => run.wallMs));
      const profiledMedianMs = median(profiled.map((run) => run.wallMs));
      cases.push({
        id: test.id,
        patches: image.positions.length / 2,
        activePatches: model.shape[0],
        softTokens: image.softTokens,
        attentionMode: model.attentionMode,
        correctness: profiled.map(({ maxAbs, relativeL2 }) => ({ maxAbs, relativeL2 })),
        baseline: { warmupRuns: warmups.length, samples: baseline.map(({ wallMs }) => ({ wallMs })), medianWallMs: baselineMedianMs },
        profile: {
          warmupRuns: 1,
          warmupWallMs: profileWarmup.wallMs,
          samples: profiled.map(({ wallMs, gpu }) => ({ wallMs, gpu })),
          medianWallMs: profiledMedianMs,
          medianKernelMs: Object.values(kernelMedianMs).reduce((sum, ms) => sum + ms, 0),
          medianTotalKernelMs: median(profiled.map((run) => Object.values(run.gpu).reduce((sum, ms) => sum + ms, 0))),
          kernelMedianMs,
        },
      });
      log(`${test.id}: baseline p50=${baselineMedianMs.toFixed(2)} ms; profiled p50=${profiledMedianMs.toFixed(2)} ms; parity maxAbs=${Math.max(...profiled.map((run) => run.maxAbs))}`);
      for (const [name, ms] of Object.entries(kernelMedianMs).sort((a, b) => b[1] - a[1])) log(`  ${name}: ${ms.toFixed(3)} ms`);
    }
    const gpuOutput = checkGpuOutput ? { cases: [], finite: await finiteCheck(model) } : null;
    if (gpuOutput) for (const item of heldProjections) gpuOutput.cases.push(await projectionCheck(runtime, item));
    window.gpuBench = {
      status: "done", done: true, backend: "webgpu",
      adapter: {
        name: runtime.name,
        vendor: runtime.adapter.info?.vendor || null,
        architecture: runtime.adapter.info?.architecture || null,
        device: runtime.adapter.info?.device || null,
        isFallbackAdapter: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter),
        timestampQuery: runtime.device.features.has("timestamp-query"),
      },
      metricMs: cases.reduce((sum, item) => sum + (profile ? item.baseline.medianWallMs : item.elapsedMs), 0) / cases.length,
      profile,
      options,
      ...(gpuOutput ? { gpuOutput } : {}),
      wasm: runtime.wgsl.wasmUrl,
      correctness: { ok: true, tolerance, cases, sourceCommit: fixture.sourceCommit, compute: fixture.compute },
    };
  } finally { for (const item of heldProjections) item.descriptor.destroy(); profiler?.destroy(); model?.destroy(); runtime.device.destroy(); }
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
  log(error.stack || error);
});
