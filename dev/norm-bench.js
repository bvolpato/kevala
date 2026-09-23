// Paired full-output and independent f64 guards for Laya norm and Qwen/Gemma RMS.
// Both variants use the production renderer; only the subgroup flag differs.
import { pipeline, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const query = new URLSearchParams(location.search);
const log = (line) => { document.getElementById("log").textContent += `${line}\n`; };
const EPS = Math.fround(1e-6);
const CPU_ABS = 2e-5;
const SENTINEL = -3141.5;
const VARIANTS = ["shared", "shuffle"];
const state = { status: "running", done: false, backend: "webgpu", kernel: "norm+kev_rms+gemma4_rms", correctness: { ok: false, cases: [] } };
window.gpuBench = state;
let activeDevice;
let U;

const PERFORMANCE_CASES = [
  ...[47, 140, 512].map((T) => ({ kernel: "norm", T, D: 1024 })),
  ...[31, 124, 532].map((T) => ({ kernel: "kev_rms", T, D: 1024 })),
  ...[105, 198, 606].map((T) => ({ kernel: "kev_rms", T, D: 2048 })),
  ...[1536, 2560].flatMap((D) => [99, 193, 601].flatMap((T) => [1, 2].map((mode) => ({ kernel: "gemma4_rms", T, D, mode })))),
];

function intParam(name, fallback, min, max) {
  if (!query.has(name)) return fallback;
  const value = Number(query.get(name));
  if (!Number.isInteger(value) || value < min || value > max) throw new Error(`invalid ${name}`);
  return value;
}

function makeInput(c, seed) {
  let rng = seed >>> 0;
  const random = () => {
    rng = Math.imul(rng ^ (rng >>> 16), 0x45d9f3b);
    rng = Math.imul(rng ^ (rng >>> 16), 0x45d9f3b);
    rng ^= rng >>> 16;
    return (rng >>> 0) / 0x100000000;
  };
  const x = new Float32Array(c.T * c.D);
  const weights = new Float32Array(c.D);
  const bias = new Float32Array(c.D);
  const types = new Float32Array(3 * c.D);
  const tokens = new Uint32Array(Math.max(4, c.T * 4));
  for (let i = 0; i < x.length; i++) {
    const value = (random() - 0.5) * 2;
    const sign = i % 2 === 0 ? 1 : -1;
    switch (c.pattern) {
      case "zero": x[i] = 0; break;
      case "constant": x[i] = 7; break;
      case "high": x[i] = value * 1e10; break;
      case "tiny": x[i] = value * 1e-12; break;
      case "cancellation": x[i] = sign * (65536 + (i % 7) / 16); break;
      case "outlier": x[i] = i % c.D === 0 ? sign * 1e6 : value / 16; break;
      default: x[i] = value;
    }
  }
  for (let i = 0; i < c.D; i++) {
    weights[i] = (random() - 0.5) * 2;
    bias[i] = (random() - 0.5) / 8;
  }
  for (let i = 0; i < types.length; i++) types[i] = (random() - 0.5) / 8;
  for (let t = 0; t < c.T; t++) tokens[t * 4 + 3] = t % 3;
  const residual = new Float32Array(x.length);
  for (let i = 0; i < residual.length; i++) residual[i] = (random() - 0.5) / 4;
  return { x, weights, bias, types, tokens, residual };
}

function cpuReference(c, input) {
  const output = new Float64Array(c.T * c.D);
  for (let t = 0; t < c.T; t++) {
    const base = t * c.D;
    let mean = 0;
    if (c.kernel === "norm") {
      for (let d = 0; d < c.D; d++) mean += input.x[base + d];
      mean /= c.D;
    }
    let variance = 0;
    for (let d = 0; d < c.D; d++) {
      const value = input.x[base + d] - mean;
      variance += value * value;
    }
    const inv = 1 / Math.sqrt(variance / c.D + EPS);
    for (let d = 0; d < c.D; d++) {
      const factor = c.kernel === "gemma4_rms" && c.mode === 0 ? 1 : input.weights[d];
      let value = (input.x[base + d] - mean) * inv * factor;
      if (c.kernel === "gemma4_rms" && c.mode === 2) value = input.residual[base + d] + value;
      if (c.kernel === "norm" && c.bias) value += input.bias[d];
      if (c.kernel === "norm" && c.typed) value += input.types[(t % 3) * c.D + d];
      output[base + d] = value;
    }
  }
  return output;
}

function caseLabel(c) {
  return `${c.kernel}-${c.T}x${c.D}-${c.pattern || "normal"}-b${c.bias || 0}t${c.typed || 0}m${c.mode || 0}`;
}

function makeResources(device, pipes, c, input) {
  const owned = [];
  const buffer = (array, usage) => {
    const b = device.createBuffer({ size: Math.max(16, array.byteLength), usage: usage | U.COPY_DST });
    if (array.byteLength) device.queue.writeBuffer(b, 0, array);
    owned.push(b);
    return b;
  };
  const globals = buffer(new Uint32Array([c.T, 0, 0, 0]), U.UNIFORM);
  const raw = new ArrayBuffer(16);
  new Uint32Array(raw).set([c.D, c.bias || 0, c.typed || 0, 0]);
  new Float32Array(raw)[3] = EPS;
  if (c.kernel === "gemma4_rms") {
    new Uint32Array(raw).set([c.D, 0, c.mode || 0, 0]);
    new Float32Array(raw)[1] = EPS;
  }
  const params = buffer(new Uint32Array(raw), U.UNIFORM);
  const x = buffer(input.x, U.STORAGE);
  const weights = buffer(input.weights, U.STORAGE);
  const bias = buffer(input.bias, U.STORAGE);
  const types = buffer(input.types, U.STORAGE);
  const tokens = buffer(input.tokens, U.STORAGE);
  const initial = new Float32Array((c.T + 2) * c.D).fill(SENTINEL);
  if (c.kernel === "gemma4_rms" && c.mode === 2) initial.set(input.residual);
  const outputs = VARIANTS.map(() => buffer(initial, U.STORAGE | U.COPY_SRC));
  const groups = pipes.map((pipe, i) => {
    const buffers = c.kernel === "norm"
      ? [globals, params, x, weights, bias, outputs[i], tokens, types]
      : [globals, params, x, weights, outputs[i]];
    return device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) });
  });
  return { outputs, groups, bytes: initial.byteLength,
    resetOutput(v) { if (c.kernel === "gemma4_rms" && c.mode === 2) device.queue.writeBuffer(outputs[v], 0, initial); },
    destroy() { for (const b of owned) b.destroy(); } };
}

async function checked(device, operation) {
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  let value, failure;
  try { value = await operation(); } catch (error) { failure = error; }
  const errors = await Promise.all([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
  if (errors.some(Boolean)) throw new Error(errors.filter(Boolean).map((e) => e.message).join("; "));
  if (failure) throw failure;
  return value;
}

async function readOutput(device, output, bytes) {
  const read = device.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, read, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    return values;
  } finally { read.destroy(); }
}

function compare(c, input, outputs) {
  const expected = cpuReference(c, input);
  const bits = outputs.map((a) => new Uint32Array(a.buffer, a.byteOffset, a.length));
  const cpuMaxAbs = [0, 0];
  let maxAbs = 0, bitDifferences = 0, tailErrors = 0, finite = true;
  for (let i = 0; i < outputs[0].length; i++) {
    if (bits[0][i] !== bits[1][i]) bitDifferences++;
    maxAbs = Math.max(maxAbs, Math.abs(outputs[0][i] - outputs[1][i]));
    for (let v = 0; v < 2; v++) {
      finite &&= Number.isFinite(outputs[v][i]);
      if (i < expected.length) cpuMaxAbs[v] = Math.max(cpuMaxAbs[v], Math.abs(outputs[v][i] - expected[i]));
      else if (outputs[v][i] !== SENTINEL) tailErrors++;
    }
  }
  return { label: caseLabel(c), ...c, outputsCompared: expected.length, tailOutputsCompared: 2 * c.D,
    maxAbs, bitDifferences, tailErrors, cpuMaxAbs, finite,
    ok: finite && bitDifferences === 0 && tailErrors === 0 && cpuMaxAbs.every((e) => e <= CPU_ABS) };
}

function timer(device) {
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: U.QUERY_RESOLVE | U.COPY_SRC });
  const read = device.createBuffer({ size: 16, usage: U.MAP_READ | U.COPY_DST });
  return { querySet, resolved, read, destroy() { querySet.destroy(); resolved.destroy(); read.destroy(); } };
}

async function timed(device, clock, pipe, group, T, reps) {
  return checked(device, async () => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: { querySet: clock.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    for (let i = 0; i < reps; i++) pass.dispatchWorkgroups(T);
    pass.end();
    encoder.resolveQuerySet(clock.querySet, 0, 2, clock.resolved, 0);
    encoder.copyBufferToBuffer(clock.resolved, 0, clock.read, 0, 16);
    device.queue.submit([encoder.finish()]);
    await clock.read.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(clock.read.getMappedRange().slice(0));
    clock.read.unmap();
    const ms = Number(stamps[1] - stamps[0]) / 1e6 / reps;
    if (!(ms > 0 && Number.isFinite(ms))) throw new Error(`invalid timestamp ${ms}`);
    return ms;
  });
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const geomean = (xs) => Math.exp(xs.reduce((sum, x) => sum + Math.log(x), 0) / xs.length);
const hash = async (text) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)))].map((b) => b.toString(16).padStart(2, "0")).join("");

async function main() {
  U = GPUBufferUsage;
  const warmups = intParam("warmups", 3, 0, 30);
  const samples = intParam("samples", 11, 1, 100);
  const reps = intParam("reps", 32, 1, 256);
  const seed = intParam("seed", 0x4b455641, 1, 0xffffffff);
  const reverse = query.get("reverse") === "1";
  const forceSharedTail = query.get("forceSharedTail") === "1";
  const kernels = (query.get("kernels") || "norm,kev_rms,gemma4_rms").split(",");
  if (kernels.some((name) => !["norm", "kev_rms", "gemma4_rms"].includes(name))) throw new Error("invalid kernels");
  const performanceCases = PERFORMANCE_CASES.filter((c) => kernels.includes(c.kernel));
  state.kernel = kernels.join("+");
  if (!navigator.gpu?.wgslLanguageFeatures?.has("subgroup_id")) throw new Error("subgroup_id language feature required");
  const gpu = await requestDevice({ features: ["timestamp-query", "subgroups"] });
  const device = activeDevice = gpu.device;
  if (!gpu.subgroup32 || !device.features.has("timestamp-query")) throw new Error("fixed 32-lane subgroups and timestamp-query required");
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  const source = await kernelSource();
  const cache = new Map();
  const shaderHashes = {};
  const getPipes = async (c) => {
    const key = `${c.kernel}/${c.D}`;
    if (!cache.has(key)) {
      const pipes = [];
      for (const [i, variant] of VARIANTS.entries()) {
        const spec = { subgroups: i === 1 };
        if (c.kernel === "kev_rms") spec.kev = { hidden: c.D };
        let code = source(c.kernel, spec);
        if (i === 1 && forceSharedTail) {
          if (!code.includes("subgroup_count == 8u")) throw new Error("missing subgroup-count fallback guard");
          code = code.replaceAll("subgroup_count == 8u", "false");
        }
        shaderHashes[`${key}/${variant}`] = await hash(code);
        pipes.push(await pipeline(device, code, `${key}/${variant}`));
      }
      cache.set(key, pipes);
    }
    return cache.get(key);
  };
  const correctnessCases = [];
  for (const pattern of ["normal", "zero", "constant", "high", "tiny", "cancellation", "outlier"]) {
    for (let mode = 0; mode < 4; mode++) correctnessCases.push({ kernel: "norm", T: 3, D: 1024, bias: mode & 1, typed: mode >> 1, pattern });
    for (const D of [1024, 2048, 2560, 4096]) correctnessCases.push({ kernel: "kev_rms", T: 3, D, pattern });
    for (const D of [32, 257, 1536, 2560]) for (const mode of [0, 1, 2]) correctnessCases.push({ kernel: "gemma4_rms", T: 3, D, mode, pattern });
  }
  for (const T of [0, 1, 31, 65]) {
    correctnessCases.push({ kernel: "norm", T, D: 1024, bias: 1, typed: 1, pattern: "normal" });
    correctnessCases.push({ kernel: "kev_rms", T, D: T === 0 ? 256 : T === 1 ? 512 : 1536, pattern: "normal" });
    for (const mode of [0, 1, 2]) correctnessCases.push({ kernel: "gemma4_rms", T, D: T <= 1 ? 31 : 255, mode, pattern: "normal" });
  }
  correctnessCases.push(...PERFORMANCE_CASES.map((c) => ({ ...c, pattern: "normal" })));
  state.adapter = { name: gpu.name, vendor: gpu.adapter.info.vendor, architecture: gpu.adapter.info.architecture,
    isFallbackAdapter: !!gpu.adapter.info.isFallbackAdapter, subgroupMinSize: gpu.adapter.info.subgroupMinSize,
    subgroupMaxSize: gpu.adapter.info.subgroupMaxSize, features: [...device.features].sort() };
  state.provenance = { sourceBaseCommit: "fb2dc51362895c53d1393ce3fd9c5c7a72ec64ca", wasmUrl: source.wasmUrl,
    page: location.href, userAgent: navigator.userAgent, timestamp: new Date().toISOString(), shaderHashes, forceSharedTail,
    wgslLanguageFeatures: [...navigator.gpu.wgslLanguageFeatures].sort(),
    reduction: "Existing 128/64 shared stages; eight subgroups use stride32 in registers and XOR16/8/4/2/1; other subgroup counts retain shared32..1; shared scalar broadcast" };
  state.correctness.tolerance = { pairedBitExact: true, cpuAbsolute: CPU_ABS };
  log(`device=${gpu.name}; fixed subgroup32; ${correctnessCases.filter((c) => kernels.includes(c.kernel)).length} full-array cases`);
  for (const c of correctnessCases.filter((item) => kernels.includes(item.kernel))) {
    const input = makeInput(c, seed + c.T + c.D);
    const pipes = await getPipes(c);
    const resources = makeResources(device, pipes, c, input);
    try {
      await checked(device, async () => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (let v = 0; v < 2; v++) {
          pass.setPipeline(pipes[v]);
          pass.setBindGroup(0, resources.groups[v]);
          pass.dispatchWorkgroups(c.T + 2);
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
      });
      const outputs = [];
      for (const output of resources.outputs) outputs.push(await readOutput(device, output, resources.bytes));
      const result = compare(c, input, outputs);
      state.correctness.cases.push(result);
      log(`${result.label}: ${result.ok ? "ok" : "FAILED"}; bits=${result.bitDifferences}; CPU=${result.cpuMaxAbs.map((e) => e.toExponential(3))}`);
    } finally { resources.destroy(); }
  }
  state.correctness.ok = state.correctness.cases.every((c) => c.ok);
  if (!state.correctness.ok) throw new Error("normalization correctness guard failed");
  const clock = timer(device);
  const cases = [];
  try {
    for (const c of performanceCases) {
      const pipes = await getPipes(c);
      const resources = makeResources(device, pipes, c, makeInput(c, seed + c.T + c.D));
      const times = [[], []];
      try {
        for (let i = 0; i < warmups + samples; i++) {
          const order = (i + Number(reverse)) % 2 ? [1, 0] : [0, 1];
          for (const v of order) {
            resources.resetOutput(v);
            const ms = await timed(device, clock, pipes[v], resources.groups[v], c.T, reps);
            if (i >= warmups) times[v].push(ms);
          }
        }
        const medians = times.map(median);
        cases.push({ ...c, shared: times[0], shuffle: times[1], sharedMs: medians[0], shuffleMs: medians[1], speedup: medians[0] / medians[1] });
        log(`${caseLabel(c)}: shared=${medians[0].toFixed(6)} shuffle=${medians[1].toFixed(6)} ms speed=${(medians[0] / medians[1]).toFixed(3)}x`);
      } finally { resources.destroy(); }
    }
  } finally { clock.destroy(); }
  if (uncaptured.length) throw new Error(uncaptured.join("; "));
  Object.assign(state, { status: "done", done: true, metricMs: geomean(cases.map((c) => c.shuffleMs)),
    metric: `geometric mean of ${cases.length} normalization cases, milliseconds`, method: "alternated WebGPU timestamp-query",
    warmups, samples, reps, reverse, seed, performance: { cases, sharedMs: geomean(cases.map((c) => c.sharedMs)), speedup: geomean(cases.map((c) => c.speedup)) } });
  log(`done: ${state.correctness.cases.length} guards; metricMs=${state.metricMs}`);
}

main().catch((error) => {
  state.status = "error";
  state.done = true;
  state.error = `${error?.message || error}\n${error?.stack || ""}`;
  state.correctness.ok = false;
  log(state.error);
}).finally(() => activeDevice?.destroy());
