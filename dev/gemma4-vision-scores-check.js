import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const log = (value) => { document.getElementById("log").textContent += `${value}\n`; };
const f32 = Math.fround;

function expectedScores(q, k, positions, rows, heads, kvHeads, dim) {
  const result = new Float32Array(rows * rows * heads);
  for (let head = 0; head < heads; head++) for (let row = 0; row < rows; row++) for (let key = 0; key < rows; key++) {
    const at = (head * rows + row) * rows + key;
    if (positions[key * 2] < 0) { result[at] = -1e30; continue; }
    const qi = (row * heads + head) * dim;
    const ki = (key * kvHeads + Math.floor(head / (heads / kvHeads))) * dim;
    let score = 0;
    for (let col = 0; col < dim; col += 4) {
      let dot = 0;
      for (let lane = 0; lane < 4; lane++) dot = f32(dot + f32(q[qi + col + lane] * k[ki + col + lane]));
      score = f32(score + dot);
    }
    result[at] = score;
  }
  return result;
}

function compare(actual, expected, tolerance) {
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i])) throw new Error(`nonfinite score at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
  }
  if (maxAbs > tolerance) throw new Error(`score maxAbs ${maxAbs} exceeds ${tolerance}`);
  return maxAbs;
}

async function check(device, source, rows, heads, kvHeads, dim, mask) {
  const owned = [];
  const buffer = (size, usage, values) => {
    const result = device.createBuffer({ size: Math.max(16, size), usage });
    owned.push(result);
    if (values) device.queue.writeBuffer(result, 0, values);
    return result;
  };
  try {
    const usage = GPUBufferUsage;
    const q = Float32Array.from({ length: rows * heads * dim }, (_, i) => Math.sin(i * 0.013 + 0.4) * 0.3);
    const k = Float32Array.from({ length: rows * kvHeads * dim }, (_, i) => Math.cos(i * 0.019 - 0.7) * 0.2);
    const positions = Int32Array.from({ length: rows * 2 }, (_, i) => mask && Math.floor(i / 2) % 5 === 0 ? -1 : Math.floor(i / 2));
    const params = buffer(16, usage.UNIFORM | usage.COPY_DST, new Uint32Array([rows, heads, kvHeads, dim]));
    const inputs = [q, k, positions].map((values) => buffer(values.byteLength, usage.STORAGE | usage.COPY_DST, values));
    const expected = expectedScores(q, k, positions, rows, heads, kvHeads, dim);
    const output = buffer(expected.byteLength, usage.STORAGE | usage.COPY_SRC | usage.COPY_DST);
    const readback = buffer(expected.byteLength, usage.MAP_READ | usage.COPY_DST);
    const types = ["uniform", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
    const layout = device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const group = device.createBindGroup({ layout, entries: [params, ...inputs, output].map((item, binding) => ({ binding, resource: { buffer: item } })) });
    const results = [];
    for (const tiled of dim === 64 ? [false, true] : [false]) {
      const kernel = tiled ? "gemma4_vision_scores_tiled" : "gemma4_vision_scores";
      const compute = await pipeline(device, source(kernel), kernel, pipelineLayout);
      const groups = tiled ? Math.ceil(rows / 16) ** 2 * heads : Math.ceil(rows * rows * heads / 256);
      device.queue.writeBuffer(output, 0, new Float32Array(expected.length).fill(NaN));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(compute);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.min(65535, groups), Math.ceil(groups / 65535));
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, expected.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0, expected.byteLength));
      readback.unmap();
      results.push({ kernel, actual, maxAbs: compare(actual, expected, 1e-5) });
    }
    const delta = results.length === 2 ? compare(results[1].actual, results[0].actual, 1e-6) : null;
    return { rows, heads, kvHeads, dim, mask, maxAbs: Math.max(...results.map((result) => result.maxAbs)), ...(delta === null ? {} : { tiledVsLinearMaxAbs: delta, bitExact: delta === 0 }) };
  } finally { for (const item of owned) item.destroy(); }
}

async function main() {
  const runtime = await requestDevice({ baseline: true, features: [] });
  try {
    const source = await kernelSource(new URLSearchParams(location.search).get("wasm") || undefined);
    const cases = [];
    const started = performance.now();
    for (const shape of [[1, 1, 1, 64, false], [17, 4, 2, 64, true], [33, 4, 1, 64, false], [48, 12, 12, 64, true], [17, 4, 2, 32, true], [17, 4, 1, 256, true]]) {
      const result = await check(runtime.device, source, ...shape);
      cases.push(result);
      log(JSON.stringify(result));
    }
    window.gpuBench = {
      status: "done", done: true, backend: "webgpu", metricMs: performance.now() - started,
      adapter: { name: runtime.name, isFallbackAdapter: Boolean(runtime.adapter.info?.isFallbackAdapter || runtime.adapter.isFallbackAdapter) },
      correctness: { ok: true, cases }, wasm: source.wasmUrl,
    };
  } finally { runtime.device.destroy(); }
}

window.gpuBench = { status: "running", backend: "webgpu" };
main().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: error.message || String(error) };
  log(error.stack || error);
});
