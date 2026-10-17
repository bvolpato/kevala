import { mmSplits, pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { mapBeforeDeadline } from "../js/src/gpu-matvec.js";
import { selectMatmulKernel } from "../js/src/gpu-tuning.js";
import { kernelSource } from "./wgsl.js";

const SENTINEL = 12345.25;
const MAX_SCRATCH = 64 * 1024 * 1024;
const SAMPLES = 3;
const REPETITIONS = 4;
const cases = [];
let metadata = {};
const log = (value) => { document.getElementById("log").textContent += `${JSON.stringify(value)}\n`; };
const hash = async (code) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code)))].map((b) => b.toString(16).padStart(2, "0")).join("");

function exact(expected, actual) {
  const a = new Uint32Array(expected.buffer, expected.byteOffset, expected.length), b = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let bitMismatches = 0, nonfinite = 0, maxAbs = 0, firstDifference = null;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) nonfinite++;
    else maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
    if (a[i] !== b[i]) { bitMismatches++; firstDifference ||= { index: i, expected: expected[i], actual: actual[i], expectedBits: a[i], actualBits: b[i] }; }
  }
  return { ok: expected.length === actual.length && bitMismatches === 0 && nonfinite === 0, samples: expected.length, bitMismatches, nonfinite, maxAbs, firstDifference, signedZeroChecked: true };
}

function inputs(T, K, variant, f16 = false) {
  return Float32Array.from({ length: T * K }, (_, i) => {
    const row = Math.floor(i / K), column = i % K;
    if (variant === "zeros") return row % 2 ? -0 : 0;
    if (variant === "saturation") return 0.32 + row * 0.04;
    if (variant === "cancellation") return f16
      ? Math.fround((column % 8 < 4 ? 64 : -64) + (column % 4 + 1) * 0.0625 + row * 0.03125)
      : Math.fround((column % 8 < 4 ? 1024 : -1024) + (column % 4 + 1) * 0.03125 + row * 0.015625);
    return Math.fround(0.04 * Math.sin(column * 0.127 + row * 0.71 + 0.3)) || 0.00390625;
  });
}

async function compile(device, source, f16) {
  const spec = { weights: "q8", f16, rows: 1, groups: 1, splitTarget: 256 };
  const baselineCode = source("matvec_gelu", spec), marker = f16 ? "var<workgroup> inputs:" : "var<workgroup> partials:";
  const index = baselineCode.indexOf(marker);
  if (index < 0 || baselineCode.indexOf(marker, index + marker.length) >= 0 || baselineCode.includes("enable f16;") !== f16) throw new Error("rendered matvec_gelu precision prelude was not recognized");
  let splitCode = "";
  if (f16) {
    const splitStart = baselineCode.indexOf("fn mm_splits(", index), splitEnd = baselineCode.indexOf("@compute", splitStart);
    if (splitStart < index || splitEnd < splitStart) throw new Error("rendered F16 matvec_gelu split policy was not recognized");
    splitCode = baselineCode.slice(splitStart, splitEnd);
  }
  const response = await fetch(new URL(f16 ? "./gemma4-matvec-rows-f16.wgsl" : "./gemma4-matvec-rows.wgsl", import.meta.url));
  if (!response.ok) throw new Error(`two-row WGSL fetch failed: ${response.status}`);
  const fragment = await response.text(), candidateCode = baselineCode.slice(0, index) + splitCode + fragment;
  const types = ["uniform", "uniform", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage"];
  const layout = device.createBindGroupLayout({ label: "two-row matvec guard", entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const [baseline, candidate, baselineHash, candidateHash, fragmentHash] = await Promise.all([
    pipeline(device, baselineCode, "matvec_rows.baseline", pl), pipeline(device, candidateCode, "matvec_rows.candidate", pl), hash(baselineCode), hash(candidateCode), hash(fragment),
  ]);
  return { baseline, candidate, layout, f16, columns: f16 ? 16 : 4, shaderHashes: { baseline: baselineHash, candidate: candidateHash, fragment: fragmentHash } };
}

async function check(runtime, pipes, { N, K, T = 2, variant = "normal" }) {
  const { device } = runtime, U = GPUBufferUsage, owned = [];
  const deadline = performance.now() + 30_000, supported = T === 2;
  const count = T * N, outputBytes = (count + 4) * 4, outputCount = supported ? 2 : 3, readBytes = Math.max(256, outputCount * outputBytes);
  const inputBytes = T * K * 4;
  const temporaryBytes = inputBytes * 2 + outputCount * outputBytes + readBytes * 2 + outputBytes + 512;
  if (temporaryBytes > MAX_SCRATCH) throw new Error("two-row matvec scratch exceeds 64 MiB");
  if (Math.max(N * K, inputBytes, outputBytes, readBytes) > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) throw new Error("two-row matvec case exceeds device buffer limits");
  let queries;
  const make = (size, usage, label, data) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage, label: `matvec-rows.${label}` }); owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  try {
    const weightsFor = (offset) => {
      const values = new Int8Array(N * K), scales = new Float32Array(N * K / 32);
      for (let i = 0; i < values.length; i++) values[i] = (i * 31 + Math.floor(i / K) * 17 + offset) % 256 - 128;
      for (let i = 0; i < scales.length; i++) scales[i] = i % 257 === 0 ? -0 : (i % 2 ? -1 : 1) * (0.0002 + i % 7 * 0.000037);
      if (variant === "saturation") for (let row = 0; row < Math.min(2, N); row++) {
        values.fill(offset ? 17 : row ? -128 : 127, row * K, (row + 1) * K);
        scales.fill(0.05, row * K / 32, (row + 1) * K / 32);
      }
      if (variant === "cancellation") {
        values.fill(offset ? -3 : 1);
        for (let i = 0; i < scales.length; i++) scales[i] = offset ? 0.0021 : 0.0037;
      }
      return { values: make(values.byteLength, U.STORAGE | U.COPY_DST, `weights-${offset}`, values), scales: make(scales.byteLength, U.STORAGE | U.COPY_DST, `scales-${offset}`, scales) };
    };
    const gate = weightsFor(0), up = weightsFor(19);
    const globals = make(16, U.UNIFORM | U.COPY_DST, "globals", new Uint32Array([T, 0, 0, 0]));
    const params = make(16, U.UNIFORM | U.COPY_DST, "params", new Uint32Array([N, K, 0, 0]));
    const x = make(inputBytes, U.STORAGE | U.COPY_DST, "input", inputs(T, K, variant, pipes.f16));
    const initial = new Float32Array(count + 4).fill(NaN); initial.fill(SENTINEL, count);
    const output = [0, 1].map((index) => make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, `output-${index}`, initial));
    if (!supported) output.push(make(outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, "unsupported-output", new Float32Array(count + 4).fill(SENTINEL)));
    const read = make(readBytes, U.MAP_READ | U.COPY_DST, "readback");
    const groups = output.map((y) => device.createBindGroup({ layout: pipes.layout, entries: [globals, params, x, gate.values, gate.scales, up.values, up.scales, y].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    const dispatch = (pass, candidate) => {
      pass.setPipeline(candidate && supported ? pipes.candidate : pipes.baseline);
      pass.setBindGroup(0, groups[candidate ? 1 : 0]);
      pass.dispatchWorkgroups(Math.ceil(N / pipes.columns), candidate && supported ? 1 : T);
    };
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, false); dispatch(pass, true);
    if (!supported) { pass.setPipeline(pipes.candidate); pass.setBindGroup(0, groups[2]); pass.dispatchWorkgroups(Math.ceil(N / pipes.columns), 1); }
    pass.end(); output.forEach((buffer, index) => encoder.copyBufferToBuffer(buffer, 0, read, index * outputBytes, outputBytes)); device.queue.submit([encoder.finish()]);
    let mapped = false, actual;
    try {
      const withinDeadline = await mapBeforeDeadline(read, outputCount * outputBytes, deadline); mapped = true;
      if (!withinDeadline) throw new Error("two-row matvec parity deadline exceeded");
      actual = new Float32Array(read.getMappedRange(0, outputCount * outputBytes).slice(0));
    } finally { if (mapped) read.unmap(); }
    const first = actual.subarray(0, count), second = actual.subarray(outputBytes / 4, outputBytes / 4 + count);
    const outputCheck = exact(first, second);
    const tailGuard = output.every((_, index) => actual.subarray(index * outputBytes / 4 + count, (index + 1) * outputBytes / 4).every((value) => value === SENTINEL));
    const unsupportedNoWrite = supported || actual.subarray(2 * outputBytes / 4).every((value) => value === SENTINEL);
    let fixtureCheck = true;
    if (variant === "zeros") fixtureCheck = first.every((value) => value === 0);
    if (variant === "saturation") fixtureCheck = Array.from({ length: T }, (_, row) => first[row * N] > 10 && first[row * N + 1] === 0).every(Boolean);
    const ok = outputCheck.ok && tailGuard && unsupportedNoWrite && fixtureCheck;
    let timing = null;
    if (ok && supported && N >= 6144 && device.features.has("timestamp-query")) {
      queries = device.createQuerySet({ type: "timestamp", count: SAMPLES * 4 });
      const resolved = make(256, U.QUERY_RESOLVE | U.COPY_SRC, "timestamps");
      const encoder = device.createCommandEncoder(), warmup = encoder.beginComputePass();
      for (const candidate of [false, true]) for (let i = 0; i < REPETITIONS; i++) dispatch(warmup, candidate);
      warmup.end();
      const order = [];
      for (let sample = 0; sample < SAMPLES; sample++) for (const candidate of sample % 2 ? [true, false] : [false, true]) {
        const index = order.length * 2; order.push({ sample, kind: candidate ? "candidate" : "baseline" });
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } });
        for (let i = 0; i < REPETITIONS; i++) dispatch(pass, candidate);
        pass.end();
      }
      const timestampBytes = SAMPLES * 4 * 8;
      encoder.resolveQuerySet(queries, 0, SAMPLES * 4, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, read, 0, timestampBytes); device.queue.submit([encoder.finish()]);
      const withinDeadline = await mapBeforeDeadline(read, timestampBytes, deadline);
      if (!withinDeadline) { read.unmap(); throw new Error("two-row matvec timing deadline exceeded"); }
      const stamps = new BigInt64Array(read.getMappedRange(0, timestampBytes)), samples = { baseline: [], candidate: [] };
      order.forEach(({ kind }, i) => samples[kind].push(Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6 / REPETITIONS)); read.unmap();
      const choice = selectMatmulKernel({ generic: samples.baseline, wide: samples.candidate });
      timing = { ...samples, order, samples: SAMPLES, repetitions: REPETITIONS, baselineMedianMs: choice.genericMs, candidateMedianMs: choice.wideMs,
        benefit: choice.benefit, stable: choice.stable, candidateWins: choice.kernel === "wide", reason: choice.reason.replaceAll("generic", "baseline").replaceAll("wide", "candidate") };
    }
    return { T, N, K, variant, f16: pipes.f16, splits: pipes.f16 ? mmSplits(1, N, K, 256, 16, 64) : 1, supported, fallback: !supported, unsupportedNoWrite, ok, outputCheck, fixtureCheck, tailGuard, timing,
      baselineWorkgroups: Math.ceil(N / pipes.columns) * T, candidateWorkgroups: supported ? Math.ceil(N / pipes.columns) : null,
      memory: { temporaryBytes, limitBytes: MAX_SCRATCH, sharedWeightBytes: 2 * N * K * 9 / 8, sharedBytesPerWorkgroup: pipes.f16 ? { baseline: 4224, candidate: 4352 } : { baseline: 2048, candidate: 4096 } }, shaderHashes: pipes.shaderHashes };
  } finally { queries?.destroy(); for (const buffer of owned) buffer.destroy(); }
}

async function main() {
  const params = new URLSearchParams(location.search), started = performance.now();
  if (params.has("precision") && !["f32", "f16"].includes(params.get("precision"))) throw new Error("precision must be f32 or f16");
  const f16 = params.get("f16") === "1" || params.get("precision") === "f16";
  const runtime = await requestDevice({ features: f16 ? ["timestamp-query", "shader-f16"] : ["timestamp-query"] });
  try {
    if (f16 && !runtime.device.features.has("shader-f16")) throw new Error("shader-f16 is unavailable");
    const source = await kernelSource(), pipes = await withGpuErrors(runtime, () => compile(runtime.device, source, f16), "two-row matvec pipeline compilation");
    metadata = { precision: f16 ? "f16" : "f32", wasmUrl: source.wasmUrl, shaderHashes: pipes.shaderHashes, adapter: { name: runtime.name, isFallbackAdapter: false, features: [...runtime.device.features] } };
    log(metadata);
    const inputs = params.get("quick") === "1" ? [] : [
      ...[32, 1056].flatMap((K) => ["normal", "zeros", "saturation", "cancellation"].map((variant) => ({ N: 17, K, variant }))),
      ...[1, 5].map((T) => ({ T, N: 17, K: 32 })),
    ];
    inputs.push(...[6144, 12288].map((N) => ({ N, K: 1536 })));
    for (const input of inputs) {
      log({ status: "checking", ...input });
      const result = await withGpuErrors(runtime, () => check(runtime, pipes, input), `two-row matvec ${input.N}x${input.K}`);
      cases.push(result); log(result);
      if (!result.ok) throw new Error("two-row shared-weight matvec bit-exact parity failed");
      if (result.timing && result.timing.baselineMedianMs === null) throw new Error("two-row matvec timestamps were invalid");
    }
    const timed = cases.filter((value) => value.timing);
    window.gpuBench = { status: "done", done: true, backend: "webgpu", ...metadata,
      metricMs: timed.length ? Math.exp(timed.reduce((sum, value) => sum + Math.log(value.timing.candidateMedianMs), 0) / timed.length) : performance.now() - started,
      method: timed.length ? "geometric mean T2 shared-weight candidate GPU timestamp milliseconds" : "untimed exactness guard wall milliseconds; timestamp-query unavailable", elapsedMs: performance.now() - started,
      correctness: { ok: true, bitExact: true, tolerance: 0, cases } };
  } finally { runtime.device.destroy(); }
}

main().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", ...metadata, correctness: { ok: false, bitExact: true, tolerance: 0, cases }, error: String(error?.message || error) };
  log(error.stack || String(error));
});
