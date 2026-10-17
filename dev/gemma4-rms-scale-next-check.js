// Dev-only exactness and paired-timestamp guard for Gemma RMS scale followed by weighted RMS.
// The candidate is derived from the registered rendered shaders at runtime; this page loads no model.
import { pipeline, requestDevice, withGpuErrors } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const U = GPUBufferUsage;
const TAIL = 4;
const SENTINEL = 12345.25;
const MEMORY_CAP = 32 * 1024 * 1024;
const TOKENS = [2, 28, 97];
const EPS = 3.71e-5;
const query = new URLSearchParams(location.search);
const useSubgroups = query.get("subgroups") === "1";
const logNode = document.getElementById("log");
const log = (value) => { logNode.textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const f32 = Math.fround;

function buffer(device, size, usage, data, label, owned, accounting) {
  const actualSize = Math.max(16, size);
  const value = device.createBuffer({ label, size: actualSize, usage });
  owned.push(value);
  accounting.bytes += actualSize;
  if (data) device.queue.writeBuffer(value, 0, data);
  return value;
}

function uniform(device, values, label, owned, accounting) {
  const raw = new ArrayBuffer(16);
  const ints = new Uint32Array(raw);
  const floats = new Float32Array(raw);
  values.forEach((value, index) => {
    if (typeof value === "object") floats[index] = value.f;
    else ints[index] = value;
  });
  return buffer(device, raw.byteLength, U.UNIFORM | U.COPY_DST, raw, label, owned, accounting);
}

function storage(device, values, label, owned, accounting, extra = U.COPY_DST | U.COPY_SRC) {
  return buffer(device, values.byteLength, U.STORAGE | extra, values, label, owned, accounting);
}

function bind(device, kernel, buffers, label) {
  return device.createBindGroup({
    label,
    layout: kernel.getBindGroupLayout(0),
    entries: buffers.map((value, binding) => ({ binding, resource: { buffer: value } })),
  });
}

function dispatch(encoder, kernel, group, T, timestamp = null) {
  const options = timestamp ? { timestampWrites: {
    querySet: timestamp.querySet,
    beginningOfPassWriteIndex: timestamp.start,
    endOfPassWriteIndex: timestamp.end,
  } } : undefined;
  const pass = encoder.beginComputePass(options);
  pass.setPipeline(kernel);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(T);
  pass.end();
}

function renderNextBlock(residualNormSource) {
  const start = residualNormSource.indexOf("  var sum2 = 0.0;");
  const end = residualNormSource.lastIndexOf("\n}");
  if (start < 0 || end <= start) throw new Error("registered residual-norm WGSL did not contain the second RMS block");
  return residualNormSource.slice(start, end);
}

function deriveCandidate(scaleSource, residualNormSource) {
  const binding5 = "@group(0) @binding(5) var<storage, read> S: array<f32>;";
  const bindingText = `${binding5}\n@group(0) @binding(6) var<storage, read> Wnext: array<f32>;\n@group(0) @binding(7) var<storage, read_write> H: array<f32>;`;
  const occurrences = scaleSource.split(binding5).length - 1;
  if (occurrences !== 1) throw new Error(`rendered RMS scale WGSL expected one binding 5 declaration, found ${occurrences}`);
  const withBindings = scaleSource.replace(binding5, bindingText);
  const mainEnd = withBindings.lastIndexOf("\n}");
  if (mainEnd < 0) throw new Error("rendered RMS scale WGSL has no main closing brace");
  const secondRmsBlock = renderNextBlock(residualNormSource);
  const appended = `\n  storageBarrier();\n${secondRmsBlock}`;
  const code = `${withBindings.slice(0, mainEnd)}${appended}${withBindings.slice(mainEnd)}`;
  const expected = `${scaleSource.slice(0, scaleSource.indexOf(binding5) + binding5.length)}\n@group(0) @binding(6) var<storage, read> Wnext: array<f32>;\n@group(0) @binding(7) var<storage, read_write> H: array<f32>;${scaleSource.slice(scaleSource.indexOf(binding5) + binding5.length, scaleSource.lastIndexOf("\n}"))}${appended}${scaleSource.slice(scaleSource.lastIndexOf("\n}"))}`;
  if (code !== expected) throw new Error("candidate derivation changed the registered scale source outside its bindings and appended RMS block");
  return { code, appendedBlock: secondRmsBlock, exactBasePreserved: true };
}

function fixture(T, D, mode, seed, scalarOverride = undefined) {
  const count = T * D;
  const X = new Float32Array(count);
  const residual = new Float32Array(count);
  const Wpost = new Float32Array(D);
  const Wnext = new Float32Array(D);
  const pattern = mode === 0 ? "zero" : mode === 1 ? "signed" : "nonpower";
  for (let i = 0; i < count; i++) {
    const t = Math.floor(i / D), d = i - t * D;
    if (pattern === "zero") {
      X[i] = d % 2 ? -0 : 0;
      residual[i] = d % 3 ? 0 : -0;
    } else if (pattern === "signed") {
      X[i] = f32((((d * 13 + t * 7 + seed) % 17) - 8) / 16);
      residual[i] = f32((((d * 5 + t * 11 + seed) % 19) - 9) / 32);
    } else {
      X[i] = f32((((d * 17 + t * 3 + seed) % 17) - 8) * 0.00437);
      residual[i] = f32((((d * 7 + t * 13 + seed) % 19) - 9) * 0.0713);
    }
  }
  for (let d = 0; d < D; d++) {
    Wpost[d] = f32((((d * 7 + seed * 3) % 17) - 8) * 0.1837);
    Wnext[d] = f32((((d * 11 + seed * 5 + 3) % 19) - 9) * 0.2719);
  }
  if (Wpost.every((value, index) => value === Wnext[index])) Wnext[0] = f32(Wnext[0] + 0.2719);
  const independentWeights = Wpost.some((value, index) => value !== Wnext[index]);
  if (!independentWeights) throw new Error(`failed to generate independent norm weights for D=${D}`);
  const scalar = scalarOverride === undefined ? (mode === 0 ? -0 : mode === 1 ? -1.337 : 0.625) : scalarOverride;
  return { T, D, count, mode, pattern, X, residual, Wpost, Wnext, independentWeights, scalar, eps: mode === 2 ? EPS : 1e-6 };
}

function compareBits(label, baseline, candidate, logicalCount) {
  if (baseline.length !== candidate.length) throw new Error(`${label}: output length mismatch`);
  const a = new Uint32Array(baseline.buffer, baseline.byteOffset, baseline.length);
  const b = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let differences = 0;
  let firstDifference = null;
  let finite = true;
  let maxAbs = 0;
  for (let i = 0; i < baseline.length; i++) {
    if (!Number.isFinite(baseline[i]) || !Number.isFinite(candidate[i])) finite = false;
    maxAbs = Math.max(maxAbs, Math.abs(baseline[i] - candidate[i]));
    if (a[i] !== b[i]) {
      differences++;
      if (firstDifference === null) firstDifference = {
        index: i,
        baseline: baseline[i], candidate: candidate[i],
        baselineBits: `0x${a[i].toString(16).padStart(8, "0")}`,
        candidateBits: `0x${b[i].toString(16).padStart(8, "0")}`,
      };
    }
  }
  const baselineTail = baseline.slice(logicalCount);
  const candidateTail = candidate.slice(logicalCount);
  const tailGuard = baselineTail.length === TAIL && candidateTail.length === TAIL
    && [...baselineTail, ...candidateTail].every((value) => value === SENTINEL);
  return {
    label, samples: logicalCount, comparedIncludingTail: baseline.length,
    finite, bitExact: differences === 0, bitDifferences: differences,
    maxAbs, firstDifference, tailGuard,
    ok: finite && differences === 0 && tailGuard,
  };
}

async function readFloats(device, source, bytes, readback, label) {
  const encoder = device.createCommandEncoder({ label: `${label}.readback` });
  encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try { return new Float32Array(readback.getMappedRange(0, bytes).slice(0)); }
  finally { readback.unmap(); }
}

async function compareBuffers(device, left, right, byteLength, readback, label, logicalCount) {
  const a = await readFloats(device, left, byteLength, readback, `${label}.baseline`);
  const b = await readFloats(device, right, byteLength, readback, `${label}.candidate`);
  return compareBits(label, a, b, logicalCount);
}

function timestampDispatch(pass, kernel, group, T) {
  pass.setPipeline(kernel);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(T);
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = ordered.length >> 1;
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

async function timePair(device, kernels, groups, fixtureData, buffers, byteLength, T) {
  const repetitions = 4;
  const sampleCount = 3;
  const records = [];
  for (let sample = 0; sample < sampleCount; sample++) {
    const queryCount = repetitions * 2 * 2;
    const querySet = device.createQuerySet({ type: "timestamp", count: queryCount, label: `rms-next.T${T}.sample${sample}.timestamps` });
    const resolved = device.createBuffer({ size: queryCount * 8, usage: U.QUERY_RESOLVE | U.COPY_SRC, label: `rms-next.T${T}.sample${sample}.resolved` });
    const readback = device.createBuffer({ size: queryCount * 8, usage: U.MAP_READ | U.COPY_DST, label: `rms-next.T${T}.sample${sample}.readback` });
    const encoder = device.createCommandEncoder({ label: `rms-next.T${T}.sample${sample}` });
    const sampleRecords = [];
    let q = 0;
    const runCandidate = () => {
      encoder.copyBufferToBuffer(buffers.seed, 0, buffers.newY, 0, byteLength);
      const start = q;
      q += 2;
      const pass = encoder.beginComputePass({ timestampWrites: {
        querySet, beginningOfPassWriteIndex: start, endOfPassWriteIndex: start + 1,
      } });
      timestampDispatch(pass, kernels.candidate, groups.candidate, T);
      pass.end();
      sampleRecords.push({ mode: "candidate", start, end: start + 1 });
    };
    const runBaseline = () => {
      encoder.copyBufferToBuffer(buffers.seed, 0, buffers.oldY, 0, byteLength);
      const start = q;
      q += 2;
      const pass = encoder.beginComputePass({ timestampWrites: {
        querySet, beginningOfPassWriteIndex: start, endOfPassWriteIndex: start + 1,
      } });
      timestampDispatch(pass, kernels.rmsScale, groups.oldScale, T);
      timestampDispatch(pass, kernels.rms, groups.oldNext, T);
      pass.end();
      sampleRecords.push({ mode: "baseline", start, end: start + 1 });
    };
    for (let repetition = 0; repetition < repetitions; repetition++) {
      const candidateFirst = (sample + repetition) % 2 === 1;
      if (candidateFirst) { runCandidate(); runBaseline(); }
      else { runBaseline(); runCandidate(); }
    }
    if (q !== queryCount) throw new Error(`timestamp query count mismatch: ${q} != ${queryCount}`);
    encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, readback, 0, queryCount * 8);
    device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ, 0, queryCount * 8);
      const timestamps = new BigUint64Array(readback.getMappedRange(0, queryCount * 8).slice(0));
      const duration = (start, end) => {
        if (timestamps[end] <= timestamps[start]) throw new Error(`nonpositive timestamp interval ${start}:${end}`);
        return Number(timestamps[end] - timestamps[start]) / 1e6;
      };
      const baselineMs = [];
      const candidateMs = [];
      for (const item of sampleRecords) {
        if (item.mode === "baseline") baselineMs.push(duration(item.start, item.end));
        else candidateMs.push(duration(item.start, item.end));
      }
      records.push({
        sample: sample + 1,
        baselineMs,
        candidateMs,
        baselineMedianMs: median(baselineMs),
        candidateMedianMs: median(candidateMs),
        speedup: median(baselineMs) / median(candidateMs),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      querySet.destroy(); resolved.destroy(); readback.destroy();
    }
  }
  const outputBytes = byteLength + TAIL * Float32Array.BYTES_PER_ELEMENT;
  const yCheck = await compareBuffers(device, buffers.oldY, buffers.newY, outputBytes, buffers.readback, `rms-next.T${T}.post-timing.Y`, byteLength / Float32Array.BYTES_PER_ELEMENT);
  const hCheck = await compareBuffers(device, buffers.oldH, buffers.newH, outputBytes, buffers.readback, `rms-next.T${T}.post-timing.H`, byteLength / Float32Array.BYTES_PER_ELEMENT);
  if (!yCheck.ok || !hCheck.ok) throw new Error(`post-timing exactness check failed for T=${T}: ${JSON.stringify({ Y: yCheck, H: hCheck })}`);
  return {
    method: "WebGPU timestamp-query; one compute pass per path; three paired samples; four repetitions per path per sample",
    baseline: "registered gemma4_rms_scale mode + registered gemma4_rms mode 1",
    candidate: "derived single-pass RMS scale + next weighted RMS",
    T, D: fixtureData.D, repetitionsPerSample: repetitions, pairedSamples: records,
    postTimingCheck: { Y: yCheck, H: hCheck, ok: true },
    baselineMedianMs: median(records.map((record) => record.baselineMedianMs)),
    candidateMedianMs: median(records.map((record) => record.candidateMedianMs)),
    speedup: median(records.map((record) => record.baselineMedianMs)) / median(records.map((record) => record.candidateMedianMs)),
  };
}

async function checkCase(device, kernels, input) {
  const { T, D, count, mode, pattern } = input;
  const outputFloats = count + TAIL;
  const outputBytes = outputFloats * Float32Array.BYTES_PER_ELEMENT;
  const logicalBytes = count * Float32Array.BYTES_PER_ELEMENT;
  const memory = { bytes: 0 };
  const timed = mode === 2 && D === 1536 && TOKENS.includes(T) && input.scalar === 0.625;
  const timingResourcesBytes = timed ? 2048 : 0;
  const fixtureBytes = input.X.byteLength + input.residual.byteLength + input.Wpost.byteLength + input.Wnext.byteLength;
  const uploadStagingBytes = input.X.byteLength + input.Wpost.byteLength + input.Wnext.byteLength + 4 + 5 * outputBytes;
  const hostBytes = fixtureBytes + 5 * outputBytes + uploadStagingBytes + (timed ? 4096 : 512);
  const expectedGpuBytes = Math.max(16, count * 4) + 2 * Math.max(16, D * 4) + 16 + 6 * outputBytes + 48 + timingResourcesBytes;
  const combinedBytes = expectedGpuBytes + hostBytes;
  if (combinedBytes > MEMORY_CAP) {
    throw new Error(`case T${T} D${D} exceeds combined 32 MiB host+GPU bound: GPU=${expectedGpuBytes}, host=${hostBytes}`);
  }
  const owned = [];
  const scalarTag = Object.is(input.scalar, -0) ? "negative-zero" : input.scalar < 0 ? "negative" : "positive";
  const label = `rms-scale-next.mode${mode}.${pattern}.${scalarTag}.T${T}.D${D}`;
  try {
    const globals = buffer(device, 16, U.UNIFORM | U.COPY_DST, new Uint32Array([T, 0, 0, 0]), `${label}.globals`, owned, memory);
    const pScale = uniform(device, [D, { f: input.eps }, mode, 0], `${label}.p-scale`, owned, memory);
    const pNext = uniform(device, [D, { f: input.eps }, 1, 0], `${label}.p-next`, owned, memory);
    const X = storage(device, input.X, `${label}.x`, owned, memory);
    const Wpost = storage(device, input.Wpost, `${label}.wpost`, owned, memory);
    const Wnext = storage(device, input.Wnext, `${label}.wnext`, owned, memory);
    const scalar = storage(device, new Float32Array([input.scalar]), `${label}.scalar`, owned, memory);
    const paddedResidual = new Float32Array(outputFloats).fill(SENTINEL);
    paddedResidual.set(input.residual);
    const seedBuffer = buffer(device, outputBytes, U.STORAGE | U.COPY_SRC | U.COPY_DST, paddedResidual, `${label}.seed`, owned, memory);
    const oldY = buffer(device, outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, paddedResidual, `${label}.old-y`, owned, memory);
    const newY = buffer(device, outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, paddedResidual, `${label}.new-y`, owned, memory);
    const oldHSeed = new Float32Array(outputFloats).fill(SENTINEL);
    const oldH = buffer(device, outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, oldHSeed, `${label}.old-h`, owned, memory);
    const newH = buffer(device, outputBytes, U.STORAGE | U.COPY_DST | U.COPY_SRC, oldHSeed, `${label}.new-h`, owned, memory);
    const readback = buffer(device, outputBytes, U.MAP_READ | U.COPY_DST, null, `${label}.readback`, owned, memory);
    const oldScale = bind(device, kernels.rmsScale, [globals, pScale, X, Wpost, oldY, scalar], `${label}.baseline-scale-bind`);
    const oldNext = bind(device, kernels.rms, [globals, pNext, oldY, Wnext, oldH], `${label}.baseline-next-bind`);
    const candidateGroup = bind(device, kernels.candidate, [globals, pScale, X, Wpost, newY, scalar, Wnext, newH], `${label}.candidate-bind`);
    const encoder = device.createCommandEncoder({ label });
    dispatch(encoder, kernels.rmsScale, oldScale, T);
    dispatch(encoder, kernels.rms, oldNext, T);
    dispatch(encoder, kernels.candidate, candidateGroup, T);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const yCheck = await compareBuffers(device, oldY, newY, outputBytes, readback, `${label}.Y`, count);
    const hCheck = await compareBuffers(device, oldH, newH, outputBytes, readback, `${label}.H`, count);
    const result = {
      T, D, mode, pattern, scalar: Object.is(input.scalar, -0) ? "-0" : input.scalar,
      scalarBits: `0x${new Uint32Array(new Float32Array([input.scalar]).buffer)[0].toString(16).padStart(8, "0")}`,
      eps: input.eps,
      independentWeights: input.independentWeights,
      outputTailFloats: TAIL,
      estimatedPeakGpuBytes: expectedGpuBytes,
      accountedPeakGpuBytes: memory.bytes + timingResourcesBytes,
      estimatedPeakHostBytes: hostBytes,
      estimatedPeakCombinedBytes: combinedBytes,
      memoryCapBytes: MEMORY_CAP,
      Y: yCheck, H: hCheck, ok: yCheck.ok && hCheck.ok,
    };
    if (timed) {
      result.timing = await timePair(device, kernels, { oldScale, oldNext, candidate: candidateGroup }, input,
        { seed: seedBuffer, oldY, newY, oldH, newH, readback }, logicalBytes, T);
    }
    return result;
  } finally {
    for (const value of owned) value.destroy();
  }
}

function shapes() {
  return [
    { D: 1536, T: [2, 28, 97], actual: true },
    { D: 1, T: [1, 2, 97] },
    { D: 257, T: [1, 2, 97] },
    { D: 1537, T: [2, 28, 97] },
    { D: 2560, T: [2, 28, 97] },
    { D: 3072, T: [2, 28, 97], wider: true },
  ];
}

async function sha256(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function main() {
  const started = performance.now();
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-rms-scale-next", subgroups: useSubgroups };
  const features = ["timestamp-query", ...(useSubgroups ? ["subgroups"] : [])];
  const gpu = await requestDevice({ features });
  const device = gpu.device;
  const uncaptured = [];
  device.addEventListener("uncapturederror", (event) => uncaptured.push(event.error?.message || String(event.error)));
  try {
    if (!device.features.has("timestamp-query")) throw new Error("paired RMS timing requires timestamp-query");
    if (useSubgroups && !gpu.subgroup32) throw new Error("subgroup path requires fixed 32-lane subgroups and 32 KiB workgroup storage");
    const source = await kernelSource();
    const scaleSource = source("gemma4_rms_scale", { subgroups: useSubgroups });
    const residualNormSource = source("gemma4_rms_residual_norm", { subgroups: useSubgroups });
    const nextSource = source("gemma4_rms", { subgroups: useSubgroups });
    const derived = deriveCandidate(scaleSource, residualNormSource);
    const registered = query.get("registered") === "1";
    const candidateSource = registered ? source("gemma4_rms_scale_next", { subgroups: useSubgroups }) : derived.code;
    const normalized = (code) => code.replace(/\/\/[^\n]*/g, "").replace(/\s+/g, "");
    if (registered && normalized(candidateSource) !== normalized(derived.code)) throw new Error("registered RMS scale/next differs from the exact derived shader");
    const hashes = {
      registeredRmsScale: await sha256(scaleSource),
      registeredRms: await sha256(nextSource),
      registeredResidualNorm: await sha256(residualNormSource),
      candidate: await sha256(candidateSource),
    };
    window.gpuBench = {
      ...window.gpuBench,
      status: "running", adapter: { name: gpu.name, features: [...device.features], subgroup32: gpu.subgroup32 },
      shaderDerivation: { ...hashes, registered, baseWasUnchanged: derived.exactBasePreserved, candidateFromRegisteredRenderedSources: true, bindingsAdded: [6, 7], storageBarrierBeforeSecondNorm: true },
    };
    log({ device: gpu.name, features: [...device.features], subgroup32: gpu.subgroup32, shaderDerivation: window.gpuBench.shaderDerivation });
    const kernels = await withGpuErrors(gpu, async () => ({
      rmsScale: await pipeline(device, scaleSource, "rms-scale-next.baseline-scale"),
      rms: await pipeline(device, nextSource, "rms-scale-next.baseline-next"),
      candidate: await pipeline(device, candidateSource, "rms-scale-next.candidate"),
    }), "compiling RMS scale-next guard pipelines");
    const results = [];
    let caseNumber = 0;
    for (const { D, T: tokenCounts, ...shapeFlags } of shapes()) {
      for (const T of tokenCounts) for (const mode of [0, 1, 2]) {
        const scalarCases = mode === 2 && D === 1536 ? [0.625, -0, -1.337] : [undefined];
        for (const scalar of scalarCases) {
          const input = fixture(T, D, mode, (0x4b455641 + caseNumber * 0x9e3779b9) >>> 0, scalar);
          const result = await withGpuErrors(gpu, () => checkCase(device, kernels, input), `checking RMS scale-next T${T} D${D} mode${mode}`);
          result.shape = shapeFlags;
          results.push(result);
          window.gpuBench = { ...window.gpuBench, status: "running", cases: results.length, results };
          log(result);
          if (!result.ok) throw Object.assign(new Error(`RMS scale-next bit-exactness failed for T${T} D${D} mode${mode}`), { failure: result });
          caseNumber++;
        }
      }
    }
    if (uncaptured.length) throw new Error(`uncaptured WebGPU error: ${uncaptured.join("; ")}`);
    const exact = results.every((result) => result.Y.bitExact && result.H.bitExact && result.Y.tailGuard && result.H.tailGuard);
    const underMemoryCap = results.every((result) => result.estimatedPeakCombinedBytes <= MEMORY_CAP && result.accountedPeakGpuBytes <= MEMORY_CAP);
    window.gpuBench = {
      ...window.gpuBench, status: "done", done: true, metricMs: performance.now() - started,
      check: "gemma4-rms-scale-next", subgroups: useSubgroups,
      adapter: { name: gpu.name, features: [...device.features], subgroup32: gpu.subgroup32, isFallbackAdapter: Boolean(gpu.adapter.info?.isFallbackAdapter || gpu.adapter.isFallbackAdapter) },
      candidate: { derivedFromRegisteredRenderedWGSL: true, exactBasePreserved: derived.exactBasePreserved, appendedSecondRmsBlock: true, addedBindings: [6, 7], hashes },
      correctness: { ok: exact && underMemoryCap, bitExact: exact, signedZeroComparedAsBits: true, tailGuards: true, cases: results.length, memoryCapBytes: MEMORY_CAP, underMemoryCap, results },
      timing: results.filter((result) => result.timing).map((result) => result.timing),
    };
    log({ done: true, cases: results.length, bitExact: exact, underMemoryCap, timings: window.gpuBench.timing });
  } finally {
    device.destroy();
  }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-rms-scale-next", subgroups: useSubgroups };
main().catch((error) => {
  const state = { ...window.gpuBench, status: "error", done: true, backend: "webgpu", error: String(error?.message || error) };
  if (error?.failure) state.failure = error.failure;
  if (error?.code) state.code = error.code;
  window.gpuBench = state;
  log(error?.stack || error);
});
