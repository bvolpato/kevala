import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const logNode = document.getElementById("log");
window.gpuBench = { status: "running", backend: "webgpu" };
const configuration = { hidden: 256, heads: 2, kv_heads: 1, lin_key_heads: 2, lin_heads: 2, rotary: 64 };
const values = new Float32Array([-3.4028234663852886e38, -1e20, -1000, -100, -30, -20, -18, -16, -10, -5, -1, -1e-6, 0, 1e-6, 1, 5, 10, 16, 20, 30, 100, 1000, 1e20, 3.4028234663852886e38]);
const uniform = (data) => ({ data, usage: GPUBufferUsage.UNIFORM });
const globals = (tokens) => uniform(new Uint32Array([tokens, 0, 1, 0]));
const sigmoid = (value) => {
  const magnitude = Math.exp(-Math.abs(value));
  return value >= 0 ? 1 / (1 + magnitude) : magnitude / (1 + magnitude);
};

async function run() {
  const runtime = await requestDevice();
  const device = runtime.device;
  const source = await kernelSource(new URLSearchParams(location.search).get("wasm") || undefined);
  const cases = [];
  const errors = [];
  let checkedValues = 0;
  const started = performance.now();
  const check = async (name, kernel, spec, inputs, expected, dispatch) => {
    const owned = [];
    try {
      const shader = await pipeline(device, source(kernel, spec), name);
      const buffers = inputs.map((input) => {
        const data = input.data ?? input;
        const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: (input.usage ?? GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(buffer, 0, data);
        owned.push(buffer);
        return buffer;
      });
      const seeded = new Float32Array(expected.length + 8).fill(37);
      const output = device.createBuffer({ size: seeded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(output, 0, seeded);
      owned.push(output);
      buffers.push(output);
      const readback = device.createBuffer({ size: seeded.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      owned.push(readback);
      const group = device.createBindGroup({ layout: shader.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(shader);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...dispatch);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, seeded.byteLength);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let maxRelative = 0;
      for (let index = 0; index < expected.length; index++) {
        const difference = Math.abs(actual[index] - expected[index]);
        if (!Number.isFinite(actual[index]) || difference > 1e-5 + 2e-5 * Math.abs(expected[index])) throw new Error(`${name}[${index}]: ${actual[index]}, expected ${expected[index]}`);
        maxRelative = Math.max(maxRelative, difference / Math.max(1, Math.abs(expected[index])));
      }
      for (let index = expected.length; index < actual.length; index++) if (actual[index] !== 37) throw new Error(`${name}: output guard ${index - expected.length} overwritten with ${actual[index]}`);
      checkedValues += expected.length;
      cases.push({ name, kernel, values: expected.length, maxRelative, guard: true });
    } catch (error) {
      errors.push(String(error?.message || error));
    } finally {
      for (const buffer of owned) buffer.destroy();
    }
  };
  try {
    const paired = new Float32Array(2 * values.length);
    paired.set(values);
    paired.fill(1, values.length);
    await check("silu tails", "kev_silumul", { kev: configuration }, [globals(1), uniform(new Uint32Array([values.length, 0, 0, 0])), paired], Float32Array.from(values, (value) => value * sigmoid(value)), [1]);
    const erf = (value) => {
      const magnitude = Math.abs(value);
      const factor = 1 / (1 + 0.3275911 * magnitude);
      const polynomial = (((((1.061405429 * factor - 1.453152027) * factor) + 1.421413741) * factor - 0.284496736) * factor + 0.254829592) * factor;
      return Math.sign(value) * (1 - polynomial * Math.exp(-magnitude * magnitude));
    };
    await check("erf GELU tails", "geglu", {}, [globals(1), uniform(new Uint32Array([values.length, 0, 0, 0])), paired], Float32Array.from(values, (value) => 0.5 * value * (1 + erf(value / Math.SQRT2))), [1]);
    const gates = new Float32Array([-1000, -100, -30, -20, -18, -16, -10, -1, 0, 1, 10, 20, 100, 1000]);
    for (const decay of [-1, -1e8]) {
      const hidden = new Float32Array(gates.length * configuration.hidden);
      gates.forEach((value, index) => { hidden[index * configuration.hidden] = value; });
      const weights = new Float32Array(configuration.hidden * 2 * configuration.lin_heads);
      for (let head = 0; head < 2 * configuration.lin_heads; head++) weights[head * 4] = 1;
      const expected = new Float32Array(gates.length * 2 * configuration.lin_heads);
      gates.forEach((value, token) => {
        const softplus = value > 20 ? value : Math.log1p(Math.exp(value));
        for (let head = 0; head < configuration.lin_heads; head++) {
          expected[token * 2 * configuration.lin_heads + head] = Math.exp(decay * softplus);
          expected[token * 2 * configuration.lin_heads + configuration.lin_heads + head] = sigmoid(value);
        }
      });
      await check(`gates decay=${decay}`, "kev_gates", { kev: configuration }, [globals(gates.length), hidden, weights, new Float32Array(configuration.lin_heads), new Float32Array(configuration.lin_heads).fill(decay)], expected, [gates.length]);
    }
    for (const heads of [2, 4, 16]) {
      const config = { ...configuration, lin_key_heads: heads, lin_heads: heads };
      const keyWidth = heads * 128;
      const coreWidth = 3 * keyWidth;
      const projectionWidth = 4 * keyWidth;
      const tokens = 2;
      const projections = new Float32Array(tokens * projectionWidth);
      const expected = new Float32Array(tokens * coreWidth);
      for (let token = 0; token < tokens; token++) {
        for (let channel = 0; channel < coreWidth; channel++) projections[token * projectionWidth + channel] = channel < 2 * keyWidth ? ((channel + 7 * token) % 17 - 8) / 7 : values[channel % values.length];
        for (let head = 0; head < 3 * heads; head++) {
          const activated = Array.from({ length: 128 }, (_, channel) => {
            const value = projections[token * projectionWidth + head * 128 + channel];
            return value * sigmoid(value);
          });
          const inverse = head < 2 * heads ? 1 / Math.sqrt(activated.reduce((sum, value) => sum + value * value, 0) + 1e-6) : 1;
          activated.forEach((value, channel) => { expected[token * coreWidth + head * 128 + channel] = value * inverse * (head < heads ? 1 / Math.sqrt(128) : 1); });
        }
      }
      const weights = new Float32Array(coreWidth * 4);
      for (let channel = 0; channel < coreWidth; channel++) weights[channel * 4 + 3] = 1;
      const tokenInfo = new Uint32Array([0, 0, 0, 0, 0, 1, 0, 0]);
      const segments = new Uint32Array([0, tokens, 0xffffffff, 0, 0, 0, 0, 0]);
      await check(`conv heads=${heads}`, "kev_conv", { kev: config }, [globals(tokens), projections, weights, tokenInfo, segments, new Float32Array(4)], expected, [tokens, Math.ceil(coreWidth / 256)]);
    }
    const coreWidth = configuration.lin_heads * 128;
    const projectionWidth = 4 * coreWidth;
    const projections = new Float32Array(projectionWidth);
    for (let index = 0; index < coreWidth; index++) projections[3 * coreWidth + index] = values[index % values.length];
    const expected = Float32Array.from({ length: coreWidth }, (_, index) => values[index % values.length] * sigmoid(values[index % values.length]) * (37 / Math.sqrt(37 * 37 + 1e-5)));
    const parameters = uniform(new Float32Array([1e-5, 0, 0, 0]));
    await check("gated RMS tails", "kev_gnorm", { kev: configuration }, [globals(1), parameters, projections, new Float32Array(128).fill(1)], expected, [1, configuration.lin_heads]);
    window.gpuBench = { status: errors.length ? "error" : "done", done: true, backend: "webgpu", wasmUrl: source.wasmUrl, method: "GPU numerical guard wall clock", metricMs: (performance.now() - started) / Math.max(1, cases.length), adapter: { name: runtime.name, vendor: runtime.adapter.info?.vendor ?? null, isFallbackAdapter: false }, correctness: { ok: !errors.length, cases, checkedValues, errors }, ...(errors.length ? { error: errors.join("; ") } : {}) };
    logNode.textContent = JSON.stringify(window.gpuBench, null, 2);
  } finally {
    device.destroy();
  }
}

run().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", error: String(error?.message || error) };
  logNode.textContent = String(error?.stack || error);
});
