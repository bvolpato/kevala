import { ClefHead, clefTailRows } from "../js/src/gpu-clef-head.js";
import { Profiler, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const scalar = new Float32Array(1), word = new Uint32Array(scalar.buffer);
const bits = (value) => { scalar[0] = value; return word[0]; };
const value = (raw) => { word[0] = raw; return scalar[0]; };
const check = (ok, message) => { if (!ok) throw new Error(message); };
const round = Math.fround;

/** Exact finite F32 FMA reference with one nearest-even rounding, independent of WGSL. */
export function f32Fma(a, b, c) {
  const unpack = (number) => {
    const raw = bits(number), exponent = raw >>> 23 & 255;
    check(exponent !== 255, "FMA reference requires finite operands");
    return { m: BigInt((raw & 0x7fffff) | (exponent ? 0x800000 : 0)) * (raw >>> 31 ? -1n : 1n), e: exponent ? exponent - 150 : -149, negative: !!(raw >>> 31) };
  };
  const x = unpack(a), y = unpack(b), z = unpack(c), pe = x.e + y.e, e = Math.min(pe, z.e);
  let m = (x.m * y.m << BigInt(pe - e)) + (z.m << BigInt(z.e - e));
  if (!m) return !x.m || !y.m ? ((x.negative !== y.negative) && z.negative ? -0 : 0) : 0;
  const negative = m < 0n;
  if (negative) m = -m;
  const target = Math.max(-149, e + m.toString(2).length - 24), shift = target - e;
  if (shift > 0) {
    const s = BigInt(shift), half = 1n << s - 1n, remainder = m & ((1n << s) - 1n);
    m >>= s;
    if (remainder > half || remainder === half && (m & 1n)) m++;
  } else m <<= BigInt(-shift);
  return round((negative ? -1 : 1) * Number(m) * 2 ** target);
}

export function projectionReference(x, w, bias, { t, n, k, segments }, fused) {
  const odd = clefTailRows(t, segments).mask, out = new Float32Array(t * n);
  const mac = fused ? f32Fma : (a, b, c) => round(c + round(a * b));
  for (let row = 0; row < t; row++) for (let col = 0; col < n; col++) {
    const low = [0, 0, 0, 0], high = [0, 0, 0, 0];
    for (let q = 0; q < k; q += 4) for (let lane = 0; lane < 4; lane++) {
      const acc = odd[row] && q % 8 ? high : low;
      acc[lane] = mac(x[row * k + q + lane], w[col * k + q + lane], acc[lane]);
    }
    if (odd[row]) for (let lane = 0; lane < 4; lane++) low[lane] = round(low[lane] + high[lane]);
    out[row * n + col] = round(round(round(low[0] + low[2]) + round(low[1] + low[3])) + (bias?.[col] ?? 0));
  }
  return out;
}

function input(length, seed, pattern) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  for (let i = 0; i < length; i++) {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    out[i] = pattern === "zero" ? (i % 2 ? -0 : 0) : value((state & 0x807fffff) | ((123 + (i % 4)) << 23));
  }
  return out;
}

async function runCase(runtime, source, config) {
  const { device } = runtime, { t, n, k, pattern } = config;
  const x = input(t * k, 0x173451 + k, pattern), w = input(n * k, 0x729431 + t, "normal");
  if (pattern === "cancellation") for (let row = 0; row < t; row++) for (let q = 8; q < k; q++) x[row * k + q] = x[row * k + q % 8];
  if (pattern === "cancellation") for (let col = 0; col < n; col++) for (let q = 8; q < k; q++) w[col * k + q] = (Math.floor(q / 8) % 2 ? -1 : 1) * w[col * k + q % 8];
  const bias = config.bias ? input(n, 0x938173, "normal") : null;
  const reference = projectionReference(x, w, bias, config, true), expected = new Uint32Array(reference.buffer);
  const heap = new ArrayBuffer((x.length + w.length + n + t * n * 2 + 256) * 4);
  let at = 64, json;
  const put = (array) => { const p = { ptr: at, length: array.length }; new Uint32Array(heap, at, array.length).set(new Uint32Array(array.buffer, array.byteOffset, array.length)); at += array.byteLength; return p; };
  const xd = put(x), wd = put(w), bd = bias ? put(bias) : null, output = put(new Float32Array(t * n));
  const layout = { version: 1, status: "ready", staticEligible: true, hidden: k, width: n, projections: [{ index: 0, kind: "memory", n, k, weights: wd, bias: bd }] };
  const coord = { memory: heap, u32: (p, length) => new Uint32Array(heap, p, length), call: (fn) => fn(), check: (status) => check(status === 0, "ABI status"), outText: () => JSON.stringify(json), tile: 0, x: {} };
  for (const name of ["layout", "begin", "status", "memory_ready", "finish", "cancel"]) coord.x[`kevala_clef_head_${name}`] = (id) => {
    json = name === "layout" ? layout : { version: 1, status: "ready", phase: "memory", id }; return 0;
  };
  const profiler = device.features.has("timestamp-query") ? new Profiler(device, 4) : null;
  const trunk = { device, wgsl: source, profiler }, engine = { coord, clefHead: "auto", flavor: "relaxed" };
  const head = new ClefHead(trunk, engine), lease = head.request();
  const stage = { version: 1, status: "ready", phase: "memory", id: 1, tokens: t, segments: config.segments, tile: 0, s: xd, memory: output };
  const timings = { bulkSamplesMs: [], tailSamplesMs: [] };
  try {
    await head.setup(lease, stage);
    const s = head.prepare(stage);
    for (let sample = -3; sample < 7; sample++) {
      head.lastProfile = {};
      const actual = await head.wave(lease, stage, s.ops, () => device.queue.writeBuffer(s.x, 0, x));
      try { head.copy(actual, s.ops, [output]); } finally { s.read.unmap(); }
      const received = coord.u32(output.ptr, output.length);
      let mismatches = 0, nonfinite = 0;
      for (let i = 0; i < received.length; i++) { mismatches += Number(received[i] !== expected[i]); nonfinite += Number((received[i] & 0x7f800000) === 0x7f800000); }
      check(mismatches === 0 && nonfinite === 0, `${engine.flavor} ${JSON.stringify(config)}: ${mismatches} bit differences, ${nonfinite} nonfinite values`);
      if (sample >= 0 && profiler) {
        timings.bulkSamplesMs.push(head.lastProfile["clef.head.memory"]);
        if (s.tailCount) timings.tailSamplesMs.push(head.lastProfile["clef.head.memory.tail"]);
      }
    }
    return { ...config, flavor: engine.flavor, ok: true, checkedValues: reference.length * 10, bitDifferences: 0, nonfinite: 0, canaryDifferences: 0,
      oddTailRows: [...clefTailRows(t, config.segments).rows], kernels: ["clef_head_projection", ...(s.tailCount ? ["clef_head_tail"] : [])], timings };
  } finally { head.dispose(); profiler?.destroy(); }
}

async function run() {
  const started = performance.now();
  const runtime = await requestDevice(), results = [], source = await kernelSource();
  try {
    const cases = [
      { t: 1, n: 32, k: 32, segments: [1], pattern: "normal", bias: false },
      { t: 16, n: 32, k: 64, segments: [16], pattern: "normal", bias: true },
      { t: 31, n: 32, k: 96, segments: [1, 2, 3, 4, 21], pattern: "cancellation", bias: true },
      { t: 47, n: 32, k: 64, segments: [2, 3, 42], pattern: "zero", bias: false },
    ];
    for (const config of cases) {
      results.push(await runCase(runtime, source, config));
      document.getElementById("log").textContent = JSON.stringify(results, null, 2);
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", metricMs: performance.now() - started,
      metric: "whole-guard elapsed wall milliseconds; not GPU kernel or model time",
      method: "whole-guard elapsed wall clock including CPU reference, compilation, dispatch/readback and checks; optional per-kernel WebGPU timestamp-query arrays are separate",
      adapter: { name: runtime.name, isFallbackAdapter: false }, cases: results,
      coordinatorFlavors: { staged: ["relaxed"], originalCpu: ["simd", "base"] },
      correctness: { ok: true, checkedValues: results.reduce((sum, item) => sum + item.checkedValues, 0) },
      scope: "relaxed coordinator only; explicit FMA tile0 four-lane bulk and compact record-local dot4 tails; independent exact finite F32 FMA reference; full output bits, finite values and canaries before/after and every sample; no full-model speed claim" };
  } finally { runtime.device.destroy(); }
}

if (typeof document !== "undefined") {
  window.gpuBench = { status: "running", backend: "webgpu" };
  run().catch((error) => {
    window.gpuBench = { status: "error", done: true, backend: "webgpu", error: String(error?.stack || error) };
    document.getElementById("log").textContent = window.gpuBench.error;
  });
}
