import { encodeMatmul, encodeMatvec, matmulPipelines, matvecPipelines, mmSplits, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";

const DTYPES = ["q8", "bf16", "f32"];
const KS = [32, 96, 256, 1536, 6144];
const NS = [1, 17, 256, 1536, 6144];
const MODES = [0, 1, 2];
const BIASES = [0, 1];
const SENTINEL = 12345.25;
const NONFINITE = Number.NaN;
const logNode = document.getElementById("log");
const log = (line) => { logNode.textContent += `${line}\n`; };
const u = GPUBufferUsage;
const perfShapes = new Set(["6144x1536", "1536x6144", "1536x1536", "12288x1536", "1536x12288"]);
const decoderShapes = [[12288, 1536], [1536, 12288]];
const splitTailShapes = [[17, 1056], [6144, 672]];
const floatBits = new Float32Array(1);
const wordBits = new Uint32Array(floatBits.buffer);

function buffer(device, size, usage, label, values = null) {
  const result = device.createBuffer({ size: Math.max(16, size), usage, label });
  if (values) device.queue.writeBuffer(result, 0, values);
  return result;
}

function bf16Lookup() {
  const values = new Float32Array(65536);
  const view = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < values.length; i++) {
    view.setUint32(0, i << 16, true);
    values[i] = view.getFloat32(0, true);
  }
  return values;
}

function weightData(dtype, N, K, bf16Values, { q8Scale = "signed-powers-of-two", exponentRange = false } = {}) {
  const size = N * K;
  if (dtype === "q8") {
    const values = new Int8Array(size);
    const blocks = K / 32;
    const scales = new Float32Array(N * blocks);
    for (let n = 0; n < N; n++) {
      for (let block = 0; block < blocks; block++) {
        const sign = (n + block) % 2 === 0 ? 1 : -1;
        scales[n * blocks + block] = q8Scale === "signed-non-power-of-two"
          ? sign * (0.000031 + ((n * 7 + block * 11) % 17) * 0.000037)
          : sign * 2 ** (-16 + ((n * 3 + block * 5) % 7));
        for (let lane = 0; lane < 32; lane++) {
          const k = block * 32 + lane;
          if (n === 0) {
            if ((lane & 1) === 0) values[n * K + k] = 7 + ((block * 13 + lane * 9) % 89);
            else values[n * K + k] = -values[n * K + k - 1];
          } else {
            const q = (n * 37 + k * 53 + (k >>> 5) * 19) % 255 - 127;
            values[n * K + k] = (n * K + k) % 97 === 0 ? -128 : q;
          }
        }
      }
    }
    values[0] = -128;
    values[1] = 127;
    return { gpu: values, scales, values, blocks, bf16Values, q8Scale, valueAt: (n, k) => values[n * K + k] * scales[n * blocks + (k >>> 5)] };
  }

  const values = dtype === "bf16" ? new Uint16Array(size) : new Float32Array(size);
  const blocks = K / 32;
  for (let n = 0; n < N; n++) for (let k = 0; k < K; k++) {
    const paired = k & ~1;
    let value;
    if (exponentRange) {
      value = k === 0 ? 2 ** 20 : k === 1 ? 2 ** -30 : 0;
    } else if (n === 0) {
      const positive = 0.003 + ((paired * 7 + 11) % 31) / 4000;
      value = (k & 1) === 0 ? positive : -positive;
    } else {
      value = Math.sin((n + 1) * 0.103 + (k + 1) * 0.019) * 0.0125;
    }
    if (dtype === "bf16") {
      floatBits[0] = value;
      const f32bits = wordBits[0];
      const upper = f32bits >>> 16;
      values[n * K + k] = (f32bits + 0x7fff + (upper & 1)) >>> 16;
    } else values[n * K + k] = value;
  }
  const valueAt = dtype === "bf16"
    ? (n, k) => bf16Values[values[n * K + k]]
    : (n, k) => values[n * K + k];
  return { gpu: values, scales: new Float32Array(4), values, blocks, bf16Values, exponentRange, valueAt };
}

function inputValues(K, { exponentRange = false } = {}) {
  const values = new Float32Array(K);
  if (exponentRange) {
    values[0] = 2 ** -20;
    values[1] = 2 ** 30;
    return values;
  }
  for (let k = 0; k < K; k += 2) {
    const value = k === 0 ? 0 : Math.sin((k + 1) * 0.071) * 0.35;
    values[k] = value;
    values[k + 1] = value;
  }
  return values;
}

function cpuBase(x, weights, N, K) {
  const out = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < K; k++) sum += x[k] * weights.valueAt(n, k);
    out[n] = sum;
  }
  return out;
}

function biasValues(N) {
  return Float32Array.from({ length: N }, (_, n) => (n & 1) ? 0.1 : -0.1);
}

function oldValues(N) {
  return Float32Array.from({ length: N }, (_, n) => ((n % 11) - 5) * 0.1);
}

function expectedValues(base, bias, old, mode, hasBias) {
  return Float64Array.from(base, (value, n) => {
    let result = value + (hasBias ? bias[n] : 0);
    if (mode === 1) result += old[n];
    else if (mode === 2) result = Math.max(result, 0);
    return result;
  });
}

function compare(actual, expected, N, label, f16 = false) {
  let maxAbs = 0;
  let maxRel = 0;
  let refPeak = 0;
  for (let i = 0; i < N; i++) {
    if (!Number.isFinite(actual[i])) throw new Error(`${label}: nonfinite output at ${i}: ${actual[i]}`);
    const abs = Math.abs(actual[i] - expected[i]);
    const rel = abs / Math.max(1e-7, Math.abs(expected[i]));
    maxAbs = Math.max(maxAbs, abs);
    maxRel = Math.max(maxRel, rel);
    refPeak = Math.max(refPeak, Math.abs(expected[i]));
  }
  const tolerance = f16 ? 3e-2 + 2e-3 * Math.max(1, refPeak) : 5e-4 + 1e-4 * Math.max(1, refPeak);
  if (maxAbs > tolerance) throw new Error(`${label}: maxAbs ${maxAbs} exceeds ${tolerance} (maxRel ${maxRel})`);
  for (let i = N; i < N + 4; i++) if (actual[i] !== SENTINEL) throw new Error(`${label}: tail store at ${i}, expected sentinel ${SENTINEL}, got ${actual[i]}`);
  return { maxAbs, maxRel, tolerance };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function waitSubmitted(device) {
  await device.queue.onSubmittedWorkDone();
}

async function timestampPair(device, direct, tiled, buffers, N, K) {
  const timestamps = device.features.has("timestamp-query");
  const repeats = Math.max(4, Math.min(24, Math.floor(120_000_000 / (N * K))));
  for (let i = 0; i < 2; i++) {
    const warmup = device.createCommandEncoder();
    for (const dispatch of [direct, tiled]) {
      const pass = warmup.beginComputePass();
      dispatch(pass);
      pass.end();
    }
    device.queue.submit([warmup.finish()]);
    await waitSubmitted(device);
  }
  const directMs = [];
  const tiledMs = [];
  if (timestamps) {
    const querySet = device.createQuerySet({ type: "timestamp", count: 4 });
    const resolved = buffer(device, 32, u.QUERY_RESOLVE | u.COPY_SRC, "matvec.timestamp-resolve");
    const readback = buffer(device, 32, u.MAP_READ | u.COPY_DST, "matvec.timestamp-readback");
    try {
      for (let sample = 0; sample < 5; sample++) {
        const encoder = device.createCommandEncoder();
        for (const [dispatch, start, end] of [[direct, 0, 1], [tiled, 2, 3]]) {
          const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: start, endOfPassWriteIndex: end } });
          for (let i = 0; i < repeats; i++) dispatch(pass);
          pass.end();
        }
        encoder.resolveQuerySet(querySet, 0, 4, resolved, 0);
        encoder.copyBufferToBuffer(resolved, 0, readback, 0, 32);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result = new BigUint64Array(readback.getMappedRange().slice(0));
        readback.unmap();
        directMs.push(Number(result[1] - result[0]) / 1e6 / repeats);
        tiledMs.push(Number(result[3] - result[2]) / 1e6 / repeats);
      }
    } finally {
      querySet.destroy();
      resolved.destroy();
      readback.destroy();
    }
    return { method: "GPU timestamp", directMs: median(directMs), matmulMs: median(tiledMs), speedup: median(tiledMs) / median(directMs), samples: 5, repetitions: repeats };
  }

  const wall = async (dispatch, output) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let i = 0; i < repeats; i++) dispatch(pass);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, buffers.readback, 0, buffers.outputBytes);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await buffers.readback.mapAsync(GPUMapMode.READ);
    buffers.readback.unmap();
    return (performance.now() - start) / repeats;
  };
  for (let sample = 0; sample < 3; sample++) {
    directMs.push(await wall(direct, buffers.directOutput));
    tiledMs.push(await wall(tiled, buffers.matmulOutput));
  }
  return { method: "submit and readback wall time", directMs: median(directMs), matmulMs: median(tiledMs), speedup: median(tiledMs) / median(directMs), samples: 3, repetitions: repeats };
}

async function checkShape(device, dtype, N, K, pipes, lut, options = {}) {
  const owned = [];
  const keep = (item) => (owned.push(item), item);
  const T = 1;
  const weights = weightData(dtype, N, K, lut, options);
  const x = inputValues(K, options);
  const bias = biasValues(N);
  const old = oldValues(N);
  const base = cpuBase(x, weights, N, K);
  const bytes = (N + 4) * 4;
  const slots = MODES.flatMap((mode) => BIASES.map((hasBias) => ({ mode, hasBias })));
  const shared = {
    globals: keep(buffer(device, 16, u.UNIFORM | u.COPY_DST, "matvec.globals", new Uint32Array([T, 1, 1, 0]))),
    input: keep(buffer(device, x.byteLength, u.STORAGE | u.COPY_DST, "matvec.input", x)),
    weights: keep(buffer(device, weights.gpu.byteLength, u.STORAGE | u.COPY_DST, `${dtype}.weights`, weights.gpu)),
    scales: keep(buffer(device, weights.scales.byteLength, u.STORAGE | u.COPY_DST, `${dtype}.scales`, weights.scales)),
    bias: keep(buffer(device, bias.byteLength, u.STORAGE | u.COPY_DST, "matvec.bias", bias)),
    nanBias: keep(buffer(device, Math.max(16, N * 4), u.STORAGE | u.COPY_DST, "matvec.disabled-bias-NaN", new Float32Array(Math.max(4, N)).fill(NONFINITE))),
  };
  const prepared = { matvec: [], matmul: [] };
  const splitCount = mmSplits(T, N, K, pipes.matmul.splitTarget, 16, 64);

  try {
    for (const { mode, hasBias } of slots) {
      const p = keep(buffer(device, 16, u.UNIFORM | u.COPY_DST, `params.${mode}.${hasBias}`, new Uint32Array([N, K, mode, hasBias])));
      for (const method of ["matvec", "matmul"]) {
        const outputValues = new Float32Array(N + 4);
        outputValues.fill(mode === 1 ? 0 : NONFINITE, 0, N);
        if (mode === 1) outputValues.set(old);
        outputValues.fill(SENTINEL, N);
        const output = keep(buffer(device, bytes, u.STORAGE | u.COPY_SRC | u.COPY_DST, `${method}.Y.${mode}.${hasBias}`, outputValues));
        const part = keep(buffer(device, Math.max(16, splitCount * N * 4), u.STORAGE | u.COPY_DST, `${method}.PART.${mode}.${hasBias}`));
        const layout = method === "matvec" ? pipes.matvec.layout : pipes.matmul.layout;
        const group = keep(device.createBindGroup({ layout, entries: [
          { binding: 0, resource: { buffer: shared.globals } },
          { binding: 1, resource: { buffer: p } },
          { binding: 2, resource: { buffer: shared.input } },
          { binding: 3, resource: { buffer: shared.weights } },
          { binding: 4, resource: { buffer: shared.scales } },
          { binding: 5, resource: { buffer: hasBias ? shared.bias : shared.nanBias } },
          { binding: 6, resource: { buffer: output } },
          { binding: 7, resource: { buffer: part } },
        ] }));
        const reduce = method === "matmul" ? keep(device.createBindGroup({ layout: pipes.matmul.reduceLayout, entries: [
          { binding: 0, resource: { buffer: shared.globals } },
          { binding: 1, resource: { buffer: p } },
          { binding: 2, resource: { buffer: part } },
          { binding: 3, resource: { buffer: hasBias ? shared.bias : shared.nanBias } },
          { binding: 4, resource: { buffer: output } },
        ] })) : null;
        prepared[method].push({ mode, hasBias, output, group, op: { N, K, group, reduce } });
      }
    }

    const readback = keep(buffer(device, slots.length * 2 * bytes, u.MAP_READ | u.COPY_DST, "matvec.output-readback"));
    const encoder = device.createCommandEncoder();
    for (const method of ["matmul", "matvec"]) {
      const pass = encoder.beginComputePass();
      for (const item of prepared[method]) {
        if (method === "matmul") encodeMatmul(pass, pipes.matmul, item.op, T);
        else encodeMatvec(pass, pipes.matvec, item.op, T);
      }
      pass.end();
      for (const [i, item] of prepared[method].entries()) {
        const slot = method === "matmul" ? i : slots.length + i;
        encoder.copyBufferToBuffer(item.output, 0, readback, slot * bytes, bytes);
      }
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const results = [];
    for (const [i, slot] of slots.entries()) {
      const expected = expectedValues(base, bias, old, slot.mode, slot.hasBias);
      const matmulAt = i * (N + 4);
      const matvecAt = (slots.length + i) * (N + 4);
      const matmulResult = actual.subarray(matmulAt, matmulAt + N + 4);
      const matvecResult = actual.subarray(matvecAt, matvecAt + N + 4);
      const matmulError = compare(matmulResult, expected, N, `${dtype} ${N}x${K} matmul mode=${slot.mode} bias=${slot.hasBias}`, pipes.matmul.f16);
      const matvecError = compare(matvecResult, expected, N, `${dtype} ${N}x${K} matvec mode=${slot.mode} bias=${slot.hasBias}`);
      let pairMaxAbs = 0;
      for (let n = 0; n < N; n++) pairMaxAbs = Math.max(pairMaxAbs, Math.abs(matvecResult[n] - matmulResult[n]));
      const pairTolerance = pipes.matvec.f16 ? 0 : matvecError.tolerance + matmulError.tolerance;
      if (pairMaxAbs > pairTolerance) throw new Error(`${dtype} ${N}x${K} direct-vs-matmul maxAbs ${pairMaxAbs} exceeds ${pairTolerance}`);
      results.push({ mode: slot.mode, bias: Boolean(slot.hasBias), matmulMaxAbs: matmulError.maxAbs, matvecMaxAbs: matvecError.maxAbs, pairMaxAbs });
    }
    const cancellationResidual = options.exponentRange ? null : Math.abs(base[0]);
    if (cancellationResidual !== null && cancellationResidual > 1e-12) throw new Error(`${dtype} ${N}x${K} cancellation row residual ${cancellationResidual}`);

    let timing = null;
    if (perfShapes.has(`${N}x${K}`)) {
      const directItem = prepared.matvec[0];
      const matmulItem = prepared.matmul[0];
      const direct = (pass) => encodeMatvec(pass, pipes.matvec, directItem.op, T);
      const tiled = (pass) => encodeMatmul(pass, pipes.matmul, matmulItem.op, T);
      timing = await timestampPair(device, direct, tiled, { directOutput: directItem.output, matmulOutput: matmulItem.output, readback, outputBytes: bytes }, N, K);
    }
    return { dtype, N, K, checked: results.length, maxMatvecAbs: Math.max(...results.map((r) => r.matvecMaxAbs)), maxMatmulAbs: Math.max(...results.map((r) => r.matmulMaxAbs)), maxPairAbs: Math.max(...results.map((r) => r.pairMaxAbs)), q8IncludesMinus128: dtype === "q8" ? weights.values.includes(-128) : undefined, q8ScaleStyle: dtype === "q8" ? weights.q8Scale : undefined, exponentRange: Boolean(options.exponentRange), cancellationResidual, timing };
  } finally {
    for (const item of owned) item.destroy?.();
  }
}

async function main() {
  const started = performance.now();
  const runtime = await requestDevice();
  let pipelineCount = 0;
  try {
    const source = await kernelSource(new URLSearchParams(location.search).get("wasm") || undefined);
    const lut = bf16Lookup();
    const adapter = { name: runtime.name, timestampQuery: runtime.device.features.has("timestamp-query"), shaderF16: runtime.device.features.has("shader-f16") };
    window.gpuBench.adapter = adapter;
    window.gpuBench.wasm = source.wasmUrl;
    log(`Adapter: ${JSON.stringify(adapter)}`);
    log(`Building production matvec and matmul pipelines for ${DTYPES.join(", ")}.`);
    const pipes = {};
    for (const dtype of DTYPES) {
      pipes[dtype] = {
        matvec: await matvecPipelines(runtime.device, source, dtype),
        matmul: await matmulPipelines(runtime.device, source, "matmul", dtype),
      };
      pipelineCount += 1;
    }
    const cases = [];
    for (const dtype of DTYPES) for (const K of KS) for (const N of NS) {
      const result = await checkShape(runtime.device, dtype, N, K, pipes[dtype], lut);
      cases.push(result);
      window.gpuBench.completed = cases.length;
      log(`${dtype} ${N}x${K}: ${result.checked} mode/bias combinations; max abs matvec ${result.maxMatvecAbs}; matmul ${result.maxMatmulAbs}; pair ${result.maxPairAbs}${result.timing ? `; ${JSON.stringify(result.timing)}` : ""}`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    for (const dtype of DTYPES) for (const [N, K] of decoderShapes) {
      const result = await checkShape(runtime.device, dtype, N, K, pipes[dtype], lut);
      cases.push(result);
      window.gpuBench.completed = cases.length;
      log(`E2B MLP ${dtype} ${N}x${K}: ${result.checked} mode/bias combinations; max abs matvec ${result.maxMatvecAbs}; matmul ${result.maxMatmulAbs}; pair ${result.maxPairAbs}; ${JSON.stringify(result.timing)}`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    for (const dtype of ["bf16", "f32"]) {
      const result = await checkShape(runtime.device, dtype, 17, 32, pipes[dtype], lut, { exponentRange: true });
      cases.push(result);
      window.gpuBench.completed = cases.length;
      log(`${dtype} exponent range 17x32: 2^20×2^-20 and 2^-30×2^30 products; max abs matvec ${result.maxMatvecAbs}; matmul ${result.maxMatmulAbs}.`);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    for (const [N, K] of splitTailShapes) {
      const result = await checkShape(runtime.device, "q8", N, K, pipes.q8, lut);
      cases.push(result);
      window.gpuBench.completed = cases.length;
      log(`Q8 odd split boundaries ${N}x${K}: max abs matvec ${result.maxMatvecAbs}; pair ${result.maxPairAbs}.`);
    }
    const scaledQ8 = await checkShape(runtime.device, "q8", 17, 96, pipes.q8, lut, { q8Scale: "signed-non-power-of-two" });
    cases.push(scaledQ8);
    window.gpuBench.completed = cases.length;
    log(`Q8 signed non-power-of-two scales 17x96: max abs matvec ${scaledQ8.maxMatvecAbs}; matmul ${scaledQ8.maxMatmulAbs}.`);
    window.gpuBench = {
      status: "done", done: true, backend: "webgpu", check: "gemma4-direct-matvec",
      metricMs: performance.now() - started, pipelineDtypes: DTYPES, pipelineSets: pipelineCount,
      cases: cases.length, dimensions: { N: NS, K: KS, decoderShapes, splitTailShapes, exponentRange: [17, 32], signedNonPowerQ8: [17, 96] }, modes: MODES, bias: BIASES,
      correctness: { ok: true, cases }, adapter, wasm: source.wasmUrl,
    };
    log(`Complete in ${window.gpuBench.metricMs.toFixed(1)} ms. All output, bias-disabled NaN, cancellation, dtype, and tail-sentinel checks passed.`);
  } finally {
    runtime.device.destroy();
  }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "gemma4-direct-matvec", completed: 0 };
main().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error) };
  log(error.stack || error);
});
