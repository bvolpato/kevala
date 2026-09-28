import { matmulConfig, mmSplits, pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const query = new URLSearchParams(location.search);
const usage = GPUBufferUsage;
const log = (line) => { document.getElementById("log").textContent += `${line}\n`; };
const median = (values) => [...values].sort((left, right) => left - right)[Math.floor(values.length / 2)];
const families = {
  laya: ["matmul", "matmul_wide", "reduce", "norm", "rope", "attention", "attention_subgroup", "attention_tile", "attention_tile_f32", "geglu", "gather"],
  kev: ["kev_rms", "kev_gather", "kev_gates", "kev_conv", "kev_save_tail", "kev_qknorm", "kev_recur_lanes", "kev_recur_lanes8", "kev_recur_lanes16", "kev_gnorm", "kev_aprep", "kev_save_kv", "kev_attention_keys", "kev_attention", "kev_attention_tile", "kev_silumul"],
  gemma: ["gemma4_embed", "gemma4_rms", "gemma4_qkv", "gemma4_attention", "gemma4_gelu", "gemma4_ple", "gemma4_residual", "gemma4_gather"],
};
const inventory = Object.values(families).flat();

function integer(name, fallback, minimum, maximum) {
  const value = query.has(name) ? Number(query.get(name)) : fallback;
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be an integer from ${minimum} to ${maximum}`);
  return value;
}

function uniform(values) {
  const data = new ArrayBuffer(Math.max(16, Math.ceil(values.length / 4) * 16));
  const integers = new Uint32Array(data);
  const floats = new Float32Array(data);
  values.forEach((value, index) => {
    if (typeof value === "object") floats[index] = value.f;
    else integers[index] = value;
  });
  return new Uint8Array(data);
}

function plan(kernel, tokens, gpu) {
  const kevHidden = integer("kevHidden", 1024, 1024, 4096);
  if (![1024, 2048, 2560, 4096].includes(kevHidden)) throw new Error("kevHidden must be 1024, 2048, 2560, or 4096");
  const hidden = kernel === "kev_rms" ? kevHidden : 1024;
  const linear = 6144;
  const heads = 8;
  const width = 1536;
  const intermediate = 3584;
  const rows = Math.min(tokens, 4);
  const config = matmulConfig(gpu.device);
  const tiled = kernel.includes("tile");
  const blockRows = kernel.startsWith("kev_") ? 8 : tiled ? 64 : 16;
  const blocks = Math.ceil(tokens / blockRows);
  const grids = {
    matmul: [16, Math.ceil(tokens / 64), mmSplits(tokens, hidden, hidden, config.splitTarget)],
    matmul_wide: [16, Math.ceil(tokens / 64), mmSplits(tokens, hidden, hidden, config.splitTarget)],
    reduce: [Math.ceil(tokens * hidden / 256)],
    norm: [tokens], rope: [tokens, 32],
    attention: [blocks, 16], attention_subgroup: [blocks, 16], attention_tile: [blocks, 16], attention_tile_f32: [blocks, 16],
    geglu: [Math.ceil(tokens * 2624 / 256)], gather: [rows],
    kev_rms: [tokens], kev_gather: [rows], kev_gates: [tokens], kev_conv: [tokens, linear / 256],
    kev_save_tail: [1, 3 * linear / 256], kev_qknorm: [tokens, 32],
    kev_recur_lanes: [1, 16, 2], kev_recur_lanes8: [1, 16, 4], kev_recur_lanes16: [1, 16, 8],
    kev_gnorm: [tokens, 16], kev_aprep: [tokens, heads + 2], kev_save_kv: [tokens, 4],
    kev_attention_keys: [Math.ceil(tokens / 16), 32], kev_attention: [tokens, heads], kev_attention_tile: [blocks, 2],
    kev_silumul: [Math.ceil(tokens * intermediate / 256)],
    gemma4_embed: [tokens], gemma4_rms: [tokens], gemma4_qkv: [tokens, 12], gemma4_attention: [tokens, heads],
    gemma4_gelu: [Math.ceil(tokens * 6144 / 256)], gemma4_ple: [tokens], gemma4_residual: [tokens], gemma4_gather: [rows],
  };
  const params = {
    matmul: [hidden, hidden, 0, 0], matmul_wide: [hidden, hidden, 0, 0], reduce: [hidden, hidden, 0, 0],
    norm: [hidden, 1, 1, { f: 1e-5 }], rope: [16, 3072, 0, 0],
    attention: [hidden, 3072, 0, 0], attention_subgroup: [hidden, 3072, 0, 0],
    attention_tile: [hidden, 3072, 0, 0], attention_tile_f32: [hidden, 3072, 0, 0],
    geglu: [2624, 0, 0, 0], kev_rms: [hidden, 0, 0, { f: 1e-6 }],
    kev_gnorm: [{ f: 1e-6 }, 0, 0, 0], kev_aprep: [{ f: 1e-6 }, 0, 0, 0], kev_silumul: [intermediate, 0, 0, 0],
    gemma4_embed: [width, width / 32, { f: Math.sqrt(width) }, 0],
    gemma4_rms: [width, { f: 1e-6 }, 1, 0],
    gemma4_qkv: [heads, 2, 256, 256, 0, 512 * 128, 1, 0, { f: 1e-6 }, 0, 0, 0],
    gemma4_attention: [heads, 2, 256, 0, 1, 0, 0, 0],
    gemma4_gelu: [6144, 0, 0, 0], gemma4_ple: [width, 256, { f: 1e-6 }, { f: 0.5 }, { f: 0.125 }, 0, 0, 0],
    gemma4_residual: [width, 1, 0, 0], gemma4_gather: [width, 0, 0, 0],
  };
  const outputs = {
    matmul: ["PART", mmSplits(tokens, hidden, hidden, config.splitTarget) * tokens * hidden],
    matmul_wide: ["PART", mmSplits(tokens, hidden, hidden, config.splitTarget) * tokens * hidden],
    reduce: ["Y", tokens * hidden], norm: ["Y", tokens * hidden], rope: ["QKV", tokens * 3072],
    attention: ["CTX", tokens * hidden], attention_subgroup: ["CTX", tokens * hidden], attention_tile: ["CTX", tokens * hidden], attention_tile_f32: ["CTX", tokens * hidden],
    geglu: ["A", tokens * 2624], gather: ["O", rows * hidden],
    kev_rms: ["Y", tokens * hidden], kev_gather: ["O", rows * hidden], kev_gates: ["AB", tokens * 32],
    kev_conv: ["C", tokens * linear], kev_save_tail: ["TAIL", 3 * linear], kev_qknorm: ["C", tokens * linear],
    kev_recur_lanes: ["CORE", tokens * 2048], kev_recur_lanes8: ["CORE", tokens * 2048], kev_recur_lanes16: ["CORE", tokens * 2048],
    kev_gnorm: ["CORE", tokens * 2048], kev_aprep: ["PROJ", tokens * 5120], kev_save_kv: ["KV", tokens * 1024],
    kev_attention_keys: ["KEYS", tokens * 512], kev_attention: ["OUT", tokens * 2048], kev_attention_tile: ["OUT", tokens * 2048],
    kev_silumul: ["A", tokens * intermediate], gemma4_embed: ["Y", tokens * width], gemma4_rms: ["Y", tokens * width],
    gemma4_qkv: ["Q", tokens * heads * 256], gemma4_attention: ["O", tokens * heads * 256],
    gemma4_gelu: ["A", tokens * 6144], gemma4_ple: ["Y", tokens * width], gemma4_residual: ["X", tokens * width], gemma4_gather: ["O", rows * width],
  };
  const segments = new Uint32Array([0, tokens, 0xffffffff, 0, 0, 0, 0, 0]);
  const tokenData = new Uint32Array(tokens * 4);
  for (let token = 0; token < tokens; token++) tokenData.set(kernel.startsWith("kev_") ? [0, token, token, 0] : [0, tokens, token, 0], token * 4);
  const header = kernel === "kev_attention_tile" ? 1 : 0;
  const blockData = new Uint32Array((blocks + header) * 4);
  if (header) blockData[0] = blocks;
  for (let block = 0; block < blocks; block++) blockData.set(kernel.startsWith("kev_") ? [0, block * blockRows, Math.min(blockRows, tokens - block * blockRows), 0] : [0, tokens, block * blockRows, 0], (block + header) * 4);
  const cosine = new Float32Array(2 * 512 * 128 * 2);
  for (let index = 0; index < cosine.length; index += 2) cosine[index] = 1;
  const gates = new Float32Array(tokens * 32);
  for (let token = 0; token < tokens; token++) gates.fill(0.99, token * 32, token * 32 + 16).fill(0.5, token * 32 + 16, (token + 1) * 32);
  return {
    grid: grids[kernel], params: params[kernel], output: outputs[kernel],
    globals: [tokens, rows, kernel.startsWith("attention") ? blocks : kernel === "gemma4_attention" ? tokens : 1, 0],
    spec: { ...config, f16: kernel === "attention_tile_f32" ? false : config.f16, subgroups: gpu.subgroup32, ...(kernel === "kev_rms" ? { kev: { hidden } } : {}) },
    overrides: {
      g: uniform([tokens, rows, kernel.startsWith("attention") ? blocks : kernel === "gemma4_attention" ? tokens : 1, 0]),
      p: uniform(params[kernel] || []), tok: tokenData, segs: segments, blocks: blockData, BLOCKS: blockData,
      CS: cosine, AB: gates, rows: Uint32Array.from({ length: rows }, (_, row) => tokens - row - 1),
      ids: Uint32Array.from({ length: tokens }, (_, token) => token % 4), pos: Uint32Array.from({ length: tokens }, (_, token) => token), POS: Uint32Array.from({ length: tokens }, (_, token) => token),
      DT: new Float32Array(16).fill(0.125), NA: new Float32Array(16).fill(-1),
    },
  };
}

async function checked(device, work) {
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let result;
  let failure;
  try { result = await work(); } catch (error) { failure = error; }
  const errors = await Promise.all([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  if (failure || errors.some(Boolean)) throw failure || errors.find(Boolean);
  return result;
}

async function hash(code) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function cpuError(kernel, tokens, input, output) {
  let maxAbs = 0;
  const check = (index, expected) => { maxAbs = Math.max(maxAbs, Math.abs(output[index] - expected)); };
  if (["norm", "kev_rms", "gemma4_rms", "gemma4_ple"].includes(kernel)) {
    const width = kernel.startsWith("gemma4") ? 1536 : kernel === "kev_rms" ? integer("kevHidden", 1024, 1024, 4096) : 1024;
    const projectionScale = kernel === "gemma4_ple" ? 0.125 : 1;
    for (let token = 0; token < tokens; token++) {
      const base = token * width;
      let mean = 0;
      if (kernel === "norm") {
        for (let dimension = 0; dimension < width; dimension++) mean += input[base + dimension];
        mean /= width;
      }
      let variance = 0;
      for (let dimension = 0; dimension < width; dimension++) variance += (input[base + dimension] * projectionScale - mean) ** 2;
      const inverse = 1 / Math.sqrt(variance / width + (kernel === "norm" ? 1e-5 : 1e-6));
      for (let dimension = 0; dimension < width; dimension++) {
        let expected = (input[base + dimension] * projectionScale - mean) * inverse * input[dimension];
        if (kernel === "norm") expected += input[dimension] * 2;
        if (kernel === "gemma4_ple") expected = (input[base + dimension] + expected) * 0.5;
        check(base + dimension, expected);
      }
    }
  } else if (kernel === "kev_aprep") {
    const expected = input.slice(0, tokens * 5120);
    for (let token = 0; token < tokens; token++) for (let head = 0; head < 10; head++) {
      const base = token * 5120 + (head < 8 ? head * 512 : 4096 + (head - 8) * 256);
      let sum = 0;
      for (let dimension = 0; dimension < 256; dimension++) sum += input[base + dimension] ** 2;
      const inverse = 1 / Math.sqrt(sum / 256 + 1e-6);
      for (let dimension = 0; dimension < 256; dimension++) expected[base + dimension] = input[base + dimension] * inverse * input[dimension];
    }
    for (let index = 0; index < expected.length; index++) check(index, expected[index]);
  } else return null;
  if (maxAbs > 1e-5) throw new Error(`${kernel}: CPU mismatch ${maxAbs}`);
  return maxAbs;
}

async function main() {
  const tokens = (query.get("tokens") || "47,128,512").split(",").map(Number);
  if (tokens.some((value) => !Number.isSafeInteger(value) || value < 1 || value > 512)) throw new Error("tokens must be comma-separated integers from 1 to 512");
  const kernels = (query.get("kernels") || inventory.join(",")).split(",");
  if (new Set(kernels).size !== kernels.length || kernels.some((kernel) => !inventory.includes(kernel))) throw new Error("unknown or duplicate kernel");
  const samples = integer("samples", 7, 3, 20);
  const warmups = integer("warmups", 2, 1, 10);
  const gpu = await requestDevice({ baseline: query.get("baseline") === "1" });
  const device = gpu.device;
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error.message));
  if (!device.features.has("timestamp-query")) throw new Error("timestamp-query is required");
  const sources = [{ label: "current", source: await kernelSource(query.get("wasm") || undefined) }];
  if (query.has("baselineWasm")) sources.unshift({ label: "baseline", source: await kernelSource(query.get("baselineWasm")) });
  const queries = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: usage.QUERY_RESOLVE | usage.COPY_SRC });
  const readback = device.createBuffer({ size: 16, usage: usage.COPY_DST | usage.MAP_READ });
  const cases = [];
  try {
    for (const kernel of kernels) for (const length of tokens) {
      const fixture = plan(kernel, length, gpu);
      const bytes = Math.max(4 * 1024 * 1024, length * 8192 * 4, fixture.output[1] * 4);
      const floats = new Float32Array(bytes / 4);
      let seed = integer("seed", 0, 0, 0xffffffff);
      for (let index = 0; index < floats.length; index++) {
        if (query.has("seed")) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; floats[index] = (seed / 0x100000000 - 0.5) * 4; }
        else floats[index] = ((index % 17) - 8) / 64;
      }
      const resources = new Map();
      const owned = [];
      try {
        const runs = await checked(device, async () => {
          const runs = [];
          for (const { label, source } of sources) {
            const code = source(kernel, fixture.spec);
            const compute = await pipeline(device, code, kernel);
            const bindings = [...code.matchAll(/@binding\((\d+)\)\s+var<([^>]+)>\s+(\w+):/g)];
            const entries = bindings.map((match) => {
              const [, binding, space, name] = match;
              const key = `${space}:${name}`;
              if (!resources.has(key)) {
                const inplace = ["rope", "kev_qknorm", "kev_gnorm", "kev_aprep", "gemma4_qkv", "gemma4_residual"].includes(kernel);
                const initial = name === fixture.output[0] && !inplace
                  ? new Float32Array(bytes / 4).fill(NaN)
                  : fixture.overrides[name] || floats;
                const size = space === "uniform" ? initial.byteLength : bytes;
                const buffer = device.createBuffer({ size, label: `${kernel}.${name}`, usage: (space === "uniform" ? usage.UNIFORM : usage.STORAGE) | usage.COPY_DST | usage.COPY_SRC });
                const resource = { name, buffer, initial, mutable: space.includes("read_write") };
                resources.set(key, resource);
                owned.push(buffer);
                device.queue.writeBuffer(buffer, 0, initial);
              }
              return { binding: Number(binding), resource: { buffer: resources.get(key).buffer } };
            });
            const group = device.createBindGroup({ layout: compute.getBindGroupLayout(0), entries });
            runs.push({ label, compute, group, samples: [], shaderSha256: await hash(code) });
          }
          return runs;
        });
        const repetitions = kernel.includes("attention") || kernel.includes("recur") || kernel.startsWith("matmul") ? 16 : 128;
        const reset = () => { for (const resource of resources.values()) if (resource.mutable) device.queue.writeBuffer(resource.buffer, 0, resource.initial); };
        const encode = (encoder, run, repeats, timestamp) => {
          const pass = encoder.beginComputePass(timestamp ? { timestampWrites: { querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
          pass.setPipeline(run.compute);
          pass.setBindGroup(0, run.group);
          for (let repeat = 0; repeat < repeats; repeat++) pass.dispatchWorkgroups(...fixture.grid);
          pass.end();
        };
        const measure = async (run) => checked(device, async () => {
          reset();
          const encoder = device.createCommandEncoder();
          encode(encoder, run, repetitions, true);
          encoder.resolveQuerySet(queries, 0, 2, resolved, 0);
          encoder.copyBufferToBuffer(resolved, 0, readback, 0, 16);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const stamps = new BigUint64Array(readback.getMappedRange().slice(0));
          readback.unmap();
          const duration = Number(stamps[1] - stamps[0]) / 1e6 / repetitions;
          if (!Number.isFinite(duration) || duration <= 0) throw new Error(`${kernel}: invalid timestamp duration`);
          return duration;
        });
        for (let round = 0; round < warmups + samples; round++) for (let index = 0; index < runs.length; index++) {
          const run = runs[(index + round) % runs.length];
          const duration = await measure(run);
          if (round >= warmups) run.samples.push(duration);
        }
        const outputs = [];
        let cpuMaxAbs = null;
        for (const run of runs) {
          reset();
          const output = [...resources.values()].find((resource) => resource.name === fixture.output[0]);
          if (!output) throw new Error(`${kernel}: output binding ${fixture.output[0]} not found`);
          const outputReadback = device.createBuffer({ size: fixture.output[1] * 4, usage: usage.COPY_DST | usage.MAP_READ });
          owned.push(outputReadback);
          await checked(device, async () => {
            const encoder = device.createCommandEncoder();
            encode(encoder, run, 1, false);
            encoder.copyBufferToBuffer(output.buffer, 0, outputReadback, 0, outputReadback.size);
            device.queue.submit([encoder.finish()]);
            await outputReadback.mapAsync(GPUMapMode.READ);
          });
          const values = new Float32Array(outputReadback.getMappedRange().slice(0));
          outputReadback.unmap();
          if (values.some((value) => !Number.isFinite(value))) throw new Error(`${kernel}: non-finite output`);
          const referenceError = cpuError(kernel, length, floats, values);
          if (referenceError !== null) cpuMaxAbs = Math.max(cpuMaxAbs || 0, referenceError);
          outputs.push(values);
        }
        if (kernel === "kev_qknorm") {
          cpuMaxAbs = 0;
          for (const values of outputs) for (let token = 0; token < length; token++) for (let head = 0; head < 32; head++) {
            const base = token * 6144 + head * 128;
            let sum = 0;
            for (let dimension = 0; dimension < 128; dimension++) sum += floats[base + dimension] ** 2;
            const scale = (head < 16 ? 1 / Math.sqrt(128) : 1) / Math.sqrt(sum + 1e-6);
            for (let dimension = 0; dimension < 128; dimension++) cpuMaxAbs = Math.max(cpuMaxAbs, Math.abs(values[base + dimension] - floats[base + dimension] * scale));
          }
          if (cpuMaxAbs > 1e-5) throw new Error(`${kernel}: CPU mismatch ${cpuMaxAbs}`);
        }
        let bitDifferences = 0;
        let maxAbs = 0;
        if (outputs.length === 2) {
          const baselineBits = new Uint32Array(outputs[0].buffer);
          const currentBits = new Uint32Array(outputs[1].buffer);
          for (let index = 0; index < baselineBits.length; index++) {
            if (baselineBits[index] !== currentBits[index]) bitDifferences++;
            maxAbs = Math.max(maxAbs, Math.abs(outputs[0][index] - outputs[1][index]));
          }
        }
        if (query.get("exact") === "1" && bitDifferences) throw new Error(`${kernel}: ${bitDifferences} paired output bits differ`);
        cases.push({ kernel, tokens: length, grid: fixture.grid, spec: fixture.spec, repetitions, timings: runs.map(({ label, samples, shaderSha256 }) => ({ label, samples, medianMs: median(samples), shaderSha256 })), finiteValues: fixture.output[1], cpuMaxAbs, bitDifferences, maxAbs });
        log(`${kernel} T=${length}: ${runs.map((run) => `${run.label}=${median(run.samples).toFixed(6)}ms`).join(" ")} diff=${maxAbs}`);
      } finally {
        for (const buffer of owned) buffer.destroy();
      }
    }
    if (uncaptured.length) throw new Error(uncaptured.join("\n"));
    window.gpuBench = {
      status: "done", backend: "webgpu", method: "WebGPU timestamp-query", samples, warmups, kernels, tokens,
      fixtureSeed: query.has("seed") ? integer("seed", 0, 0, 0xffffffff) : null, exactComparisonRequired: query.get("exact") === "1",
      metricMs: Math.exp(cases.reduce((sum, item) => sum + Math.log(item.timings.at(-1).medianMs), 0) / cases.length),
      metric: "geometric mean of isolated dispatch batch medians; not model latency",
      adapter: { name: gpu.name, features: [...device.features], info: { vendor: gpu.adapter.info.vendor, architecture: gpu.adapter.info.architecture, subgroupMinSize: gpu.adapter.info.subgroupMinSize, subgroupMaxSize: gpu.adapter.info.subgroupMaxSize } },
      sources: sources.map(({ label, source }) => ({ label, wasmUrl: source.wasmUrl })),
      correctness: { ok: true, scope: "finite-output smoke and complete overwrite of out-of-place outputs initialized to NaN; CPU-reference checks for listed kernels; independent numerical guards remain required", cpuKernels: [...new Set(cases.filter((item) => item.cpuMaxAbs !== null).map((item) => item.kernel))], finiteValues: cases.reduce((sum, item) => sum + item.finiteValues, 0) },
      caveats: ["Synthetic throughput fixtures, not model accuracy or serving latency.", "Mutable outputs reset before each batch; repeated in-place dispatches within a batch can change values.", "Matmul and reduce are timed separately, not as a complete projection."],
      cases,
    };
  } finally {
    queries.destroy(); resolved.destroy(); readback.destroy(); device.destroy();
  }
}

window.gpuBench = { status: "running" };
main().catch((error) => { log(error.stack || String(error)); window.gpuBench = { status: "error", error: error.message }; });
