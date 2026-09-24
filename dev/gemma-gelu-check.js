import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const logNode = document.getElementById("log");
window.gpuBench = { status: "running", backend: "webgpu" };

async function run() {
  const runtime = await requestDevice();
  const device = runtime.device;
  const owned = [];
  try {
    const source = await kernelSource(new URLSearchParams(location.search).get("wasm") || undefined);
    const shader = await pipeline(device, source("gemma4_gelu"), "gemma4_gelu");
    const values = [-3.4028234663852886e38, -1e12, -1e4, -1000, -100, -20, -10, -5, -1e-6, 0, 1e-6, 5, 10, 20, 100, 1000, 1e4, 1e12, 3.4028234663852886e38];
    for (let index = -48; index <= 48; index++) values.push(index / 4);
    const input = new Float32Array(values);
    const make = (data, usage, label) => {
      const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, label });
      device.queue.writeBuffer(buffer, 0, data);
      owned.push(buffer);
      return buffer;
    };
    const globals = make(new Uint32Array([1, 0, 0, 0]), GPUBufferUsage.UNIFORM, "GELU globals");
    const params = make(new Uint32Array([input.length, 0, 0, 0]), GPUBufferUsage.UNIFORM, "GELU params");
    const inputs = make(input, GPUBufferUsage.STORAGE, "GELU inputs");
    const other = make(new Float32Array(input.length), GPUBufferUsage.STORAGE, "GELU other input");
    const output = make(new Float32Array(input.length), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "GELU output");
    const readback = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    owned.push(readback);
    const group = device.createBindGroup({ layout: shader.getBindGroupLayout(0), entries: [globals, params, inputs, other, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const cases = [];
    let maxAbs = 0;
    const started = performance.now();
    for (const scale of [1, -1, 0]) {
      device.queue.writeBuffer(other, 0, new Float32Array(input.length).fill(scale));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(shader);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(input.length / 256));
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, input.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      for (let index = 0; index < input.length; index++) {
        const value = input[index];
        const expected = Math.fround(0.5 * value * (1 + Math.tanh(0.7978845608028654 * (value + 0.044715 * value ** 3))) * scale);
        const difference = Math.abs(actual[index] - expected);
        if (!Number.isFinite(actual[index]) || difference > 1e-5 + 2e-5 * Math.abs(expected)) throw new Error(`GELU(${value}) * ${scale}: ${actual[index]}, expected ${expected}`);
        maxAbs = Math.max(maxAbs, difference);
      }
      cases.push({ scale, values: input.length, finite: true });
    }
    window.gpuBench = {
      status: "done", done: true, backend: "webgpu",
      wasmUrl: source.wasmUrl,
      method: "performance.now wall-clock numerical guard",
      metricMs: (performance.now() - started) / cases.length,
      adapter: { name: runtime.name, vendor: runtime.adapter.info?.vendor ?? null, isFallbackAdapter: false },
      correctness: { ok: true, cases, maxAbs, absoluteTolerance: 1e-5, relativeTolerance: 2e-5 },
    };
    logNode.textContent = `GELU finite and accurate for ${input.length * cases.length} values; maxAbs=${maxAbs}`;
  } finally {
    for (const buffer of owned) buffer.destroy();
    device.destroy();
  }
}

run().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
  logNode.textContent = String(error?.stack || error);
});
