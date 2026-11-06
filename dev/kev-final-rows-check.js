const query = new URLSearchParams(location.search);
const log = (value) => { document.getElementById("log").textContent += `${typeof value === "string" ? value : JSON.stringify(value)}\n`; };
const median = (values) => { const sorted = values.toSorted((a, b) => a - b), middle = sorted.length >> 1; return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2; };
const integer = (name, fallback, minimum, maximum) => {
  const value = Number(query.get(name) ?? fallback);
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw new Error(`${name} must be ${minimum}..${maximum}`);
  return value;
};

export function hiddenParity(expected, actual) {
  if (expected.length !== actual.length) throw new Error(`hidden row lengths differ: ${expected.length} != ${actual.length}`);
  const left = new Uint32Array(expected.buffer, expected.byteOffset, expected.length), right = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  let differences = 0, maxAbs = 0;
  for (let i = 0; i < left.length; i++) {
    if (!Number.isFinite(expected[i]) || !Number.isFinite(actual[i])) throw new Error(`nonfinite hidden value ${i}`);
    if (left[i] !== right[i]) differences++;
    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
  }
  if (differences) throw new Error(`compacted hidden rows differ in ${differences}/${left.length} words; maxAbs=${maxAbs}`);
  return { checkedWords: left.length, bitDifferences: differences, maxAbs };
}

async function hash(bytes) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function seededRows(tokens, width, seed) {
  let state = seed >>> 0;
  return Float32Array.from({ length: tokens * width }, () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return ((state / 2 ** 32) - 0.5) * 0.5; });
}

async function checkTail(gpu, Profiler, pipeline, tokens, rows, samples, warmups) {
  gpu.R = rows.length;
  gpu.ensure(0, tokens, 1, 1, rows.length);
  const compact = await gpu.prepareFinalRows(tokens, rows);
  const preparation = { ...gpu.lastFinalRows };
  if (rows.length < tokens && !compact && gpu.lastFinalRows?.reason !== "same-row-tiles") throw new Error(`final-row path did not activate: ${gpu.lastFinalRows?.reason}`);
  if (rows.length === tokens && compact) throw new Error("all-row identity path was compacted");
  const input = seededRows(tokens, gpu.cfg.hidden, 20261106 + tokens), bytes = rows.length * gpu.cfg.hidden * 4;
  const full = gpu.ops[1].slice(gpu.finalRows.prefix.length);
  const poison = await pipeline(gpu.device, `
    @group(0) @binding(0) var<storage, read_write> values: array<u32>;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      let i = id.x + id.y * 65535u * 256u;
      if (i < arrayLength(&values)) { values[i] = 0x7fc00000u; }
    }`, "Qwen final-row poison");
  const poisoned = [gpu.x, gpu.h, gpu.up, gpu.act, gpu.gathered].map((buffer) => ({ buffer,
    group: gpu.device.createBindGroup({ layout: poison.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] }) }));
  const reset = async () => {
    gpu.device.pushErrorScope("validation");
    gpu.device.pushErrorScope("out-of-memory");
    try {
      const q = gpu.device.queue, enc = gpu.device.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(poison);
      for (const { buffer, group } of poisoned) {
        const workgroups = Math.ceil(buffer.size / (4 * 256));
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.min(workgroups, 65535), Math.ceil(workgroups / 65535));
      }
      pass.end(); q.submit([enc.finish()]);
      q.writeBuffer(gpu.x, 0, input);
      q.writeBuffer(gpu.g, 0, new Uint32Array([tokens, rows.length, 1, 2]));
      q.writeBuffer(gpu.finalG, 0, new Uint32Array([rows.length, tokens, 1, 2]));
      q.writeBuffer(gpu.rows, 0, new Uint32Array(rows));
    } finally {
      const errors = [await gpu.device.popErrorScope(), await gpu.device.popErrorScope()].filter(Boolean);
      if (errors.length) throw new Error(`poison/reset failed: ${errors.map((error) => error.message).join("; ")}`);
    }
  };
  await reset();
  const gathered = gpu.device.createCommandEncoder();
  gpu.encode(gathered, [gpu.finalRows.gather], tokens, 1);
  gathered.copyBufferToBuffer(gpu.gathered, 0, gpu.readback, 0, bytes);
  gpu.device.queue.submit([gathered.finish()]);
  await gpu.readback.mapAsync(GPUMapMode.READ, 0, bytes);
  const gatheredValues = new Float32Array(gpu.readback.getMappedRange(0, bytes).slice(0));
  gpu.readback.unmap();
  const expectedGather = new Float32Array(rows.length * gpu.cfg.hidden);
  rows.forEach((row, index) => expectedGather.set(input.subarray(row * gpu.cfg.hidden, (row + 1) * gpu.cfg.hidden), index * gpu.cfg.hidden));
  const gatherReference = hiddenParity(expectedGather, gatheredValues);
  const run = async (candidate) => {
    const q = gpu.device.queue;
    await reset();
    const profiler = new Profiler(gpu.device), prior = gpu.profiler;
    gpu.profiler = profiler;
    try {
      const enc = gpu.device.createCommandEncoder();
      gpu.device.pushErrorScope("validation");
      gpu.device.pushErrorScope("out-of-memory");
      try {
        if (candidate && compact) {
          gpu.encode(enc, [gpu.finalRows.gather], tokens, 1);
          gpu.encode(enc, gpu.finalRows.tail, rows.length, 1, tokens);
        } else gpu.encode(enc, full, tokens, 1);
        profiler.finish(enc);
        enc.copyBufferToBuffer(gpu.gathered, 0, gpu.readback, 0, bytes);
        q.submit([enc.finish()]);
        const timings = await profiler.collect();
        const values = Object.values(timings);
        if (!values.length || values.some((value) => !Number.isFinite(value) || value < 0) || values.reduce((a, b) => a + b, 0) <= 0) throw new Error("tail GPU timestamp samples are invalid");
        await gpu.readback.mapAsync(GPUMapMode.READ, 0, bytes);
        const output = new Float32Array(gpu.readback.getMappedRange(0, bytes).slice(0));
        gpu.readback.unmap();
        return { output, gpu: timings, totalMs: values.reduce((a, b) => a + b, 0) };
      } finally {
        const errors = [await gpu.device.popErrorScope(), await gpu.device.popErrorScope()].filter(Boolean);
        if (errors.length) throw new Error(errors.map((error) => error.message).join("; "));
      }
    } finally { gpu.profiler = prior; profiler.destroy(); }
  };
  const measured = [], checked = [];
  for (let sample = -warmups; sample < samples; sample++) {
    const pair = {};
    for (const candidate of sample % 2 ? [true, false] : [false, true]) pair[candidate ? "candidate" : "baseline"] = await run(candidate);
    const parity = hiddenParity(pair.baseline.output, pair.candidate.output);
    checked.push(parity);
    if (sample >= 0) measured.push({ baseline: pair.baseline.totalMs, candidate: pair.candidate.totalMs, baselineKernels: pair.baseline.gpu, candidateKernels: pair.candidate.gpu });
  }
  return { tokens, rows, sourceRows: tokens, activeRows: rows.length, compact, plan: preparation,
    poisonedBuffers: poisoned.length, gatherReference,
    checkedWords: checked.reduce((sum, value) => sum + value.checkedWords, 0), bitDifferences: 0, maxAbs: 0,
    inputSha256: await hash(input), timing: { method: "alternating paired WebGPU timestamp-query; final gather + RMS + gate/up + SiLU + down projection, excluding upload, copies and readback", samples: measured,
      baselineMedianMs: median(measured.map((value) => value.baseline)), candidateMedianMs: median(measured.map((value) => value.candidate)) } };
}

const baseState = "A customer reports a recurring invoice, an unrecognised charge, an address change and a request for a refund. ";
const questions = {
  refund: { type: "noul", instructions: "Does the customer request a refund?" },
  category: { type: "choice", instructions: "Choose the main issue.", criteria: { billing: "A billing dispute", delivery: "A missing shipment", access: "An account access problem" } },
};

function loadedCases() {
  const short = { state: "Please refund the duplicate payment.", questions: { refund: questions.refund } };
  const long = { state: baseState.repeat(5), questions };
  return [
    { id: "short-single", requests: [short] },
    { id: "long-multiple-questions", requests: [long] },
    { id: "batch-with-duplicate-state", requests: [long, { ...long, questions: { refund: questions.refund } }, { ...short, state: "The parcel was delivered yesterday." }] },
    { id: "cache-hit", prime: [long], requests: [long], cache: "hits" },
    { id: "cache-extension", prime: [long], requests: [{ ...long, state: `${long.state}The account holder confirmed the dispute by telephone.` }], cache: "extensions" },
  ];
}

async function checkLoaded(model, gpu, capture, item, samples, warmups, readout) {
  const measured = [], checked = [], cache = [];
  const run = async (candidate) => {
    gpu.finalRowsEnabled = candidate;
    gpu.stateCacheEnabled = true;
    gpu.lru = [];
    if (item.prime) await model.decideMany(item.prime);
    capture.length = 0;
    const before = { ...gpu.stats }, started = performance.now();
    const output = await model.decideMany(item.requests);
    const wallMs = performance.now() - started;
    const responses = output.map(({ timing: ignored, ...response }) => response);
    return { wallMs, responses, hidden: capture.map((value) => value.slice()), cache: { hits: gpu.stats.hits - before.hits, extensions: gpu.stats.extensions - before.extensions, misses: gpu.stats.misses - before.misses }, plan: { ...gpu.lastFinalRows } };
  };
  for (let sample = -warmups; sample < samples; sample++) {
    const pair = {};
    for (const candidate of sample % 2 ? [true, false] : [false, true]) pair[candidate ? "candidate" : "baseline"] = await run(candidate);
    if (pair.baseline.hidden.length !== pair.candidate.hidden.length || !pair.baseline.hidden.length) throw new Error(`${item.id}: hidden forward count differs or is empty`);
    for (let i = 0; i < pair.baseline.hidden.length; i++) checked.push(hiddenParity(pair.baseline.hidden[i], pair.candidate.hidden[i]));
    if (JSON.stringify(pair.baseline.responses) !== JSON.stringify(pair.candidate.responses)) throw new Error(`${item.id}: native answers or raw probabilities changed`);
    if (item.cache === "hits" && readout !== "clef" && (!pair.baseline.cache.hits || !pair.candidate.cache.hits)) throw new Error(`${item.id}: cache-hit fixture did not exercise a retained carry`);
    cache.push({ baseline: pair.baseline.cache, candidate: pair.candidate.cache });
    if (sample >= 0) measured.push({ baselineMs: pair.baseline.wallMs, candidateMs: pair.candidate.wallMs, plan: pair.candidate.plan });
  }
  return { id: item.id, requests: item.requests.length, checkedWords: checked.reduce((sum, value) => sum + value.checkedWords, 0), bitDifferences: 0, nativeResponsesEqual: true, cache,
    ...(item.cache === "extensions" && !cache.some((pair) => pair.baseline.extensions && pair.candidate.extensions) ? { limitation: "The native tokenizer did not produce a reusable extension for this fixture; extension coverage is not established." } : {}),
    timing: { method: "alternating unprofiled completed model-request wall milliseconds; cache priming excluded", samples: measured,
      baselineMedianMs: median(measured.map((value) => value.baselineMs)), candidateMedianMs: median(measured.map((value) => value.candidateMs)) } };
}

async function main() {
  const samples = integer("samples", 3, 3, 15), warmups = integer("warmups", 1, 1, 5);
  const scope = query.get("scope") || "all";
  if (!["all", "tail", "loaded"].includes(scope)) throw new Error("scope must be all, tail or loaded");
  const runtime = new URL(query.get("runtime") || "../js/src/index.js", document.baseURI);
  if (runtime.origin !== location.origin) throw new Error("runtime must be same-origin");
  const { Kevala, MODELS } = await import(runtime.href), { archPlugin } = await import(new URL("archs/index.js", runtime));
  const { Profiler, pipeline } = await import(new URL("gpu.js", runtime));
  const name = query.get("model") || "kev-0.8b", precision = query.get("precision") || "q8";
  if (!["kev", "clef"].includes(MODELS[name]?.arch)) throw new Error("model must use the Qwen trunk");
  const pack = new URL(query.get("packUrl") || `../tmp/${name}-${precision}.kevala`, document.baseURI);
  if (pack.origin !== location.origin) throw new Error("packUrl must be same-origin");
  const plugin = archPlugin(MODELS[name].arch), original = plugin.createGpu;
  let gpu, hardware, header, model;
  const capture = [], results = [];
  plugin.createGpu = function(adapter, layout, value) {
    hardware = adapter; header = value;
    gpu = original.call(this, adapter, layout, value);
    const forward = gpu.forward;
    gpu.forward = async function(...args) { const rows = await forward.apply(this, args); capture.push(rows.slice()); return rows; };
    return gpu;
  };
  try {
    model = await Kevala.load({ model: { name, url: pack.href }, backend: "webgpu", onPage: true, precision, cache: false, stateCache: false, profile: false });
    plugin.createGpu = original;
    if (!gpu?.prepareFinalRows || !hardware.device.features.has("timestamp-query")) throw new Error("current final-row runtime and hardware timestamps are required");
    if (scope !== "loaded") {
      const tokens = (query.get("tokens") || "1,7,47,65,97,129,161").split(",").map(Number);
      if (tokens.some((value) => !Number.isSafeInteger(value) || value < 1 || value > 1024)) throw new Error("tokens must be integers 1..1024");
      for (const T of tokens) {
        const rows = T === 1 ? [0] : T === 7 ? [6, 1, 6] : [T - 1, 0, Math.floor(T / 2)];
        const result = await checkTail(gpu, Profiler, pipeline, T, rows, samples, warmups);
        results.push({ scope: "tail", ...result }); log(results.at(-1));
      }
      if (tokens.includes(161)) {
        const result = await checkTail(gpu, Profiler, pipeline, 161, Array.from({ length: 57 }, (_, i) => 160 - i), samples, warmups);
        results.push({ scope: "tail", ...result }); log(results.at(-1));
      }
    }
    if (scope !== "tail") for (const item of loadedCases()) {
      const result = await checkLoaded(model, gpu, capture, item, samples, warmups, header.config.readout);
      results.push({ scope: "loaded", ...result }); log(results.at(-1));
    }
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "kev-final-rows", model: name, precision, runtime: runtime.href, pack: model.info.pack, modelInfo: model.info.model,
      adapter: { name: hardware.name, isFallbackAdapter: Boolean(hardware.adapter.isFallbackAdapter || hardware.adapter.info?.isFallbackAdapter), features: [...hardware.device.features] },
      metricMs: Math.exp(results.reduce((sum, value) => sum + Math.log(value.timing.candidateMedianMs), 0) / results.length),
      method: scope === "tail" ? "paired GPU timestamps of complete final-MLP fixtures" : "mixed numerical guard fixtures; inspect each case timing method, not a model speed metric",
      sourcePipelineCompileMs: gpu.finalRowCompileMs, sourcePipelineVariants: [...gpu.finalRowPipelines.keys()],
      correctness: { ok: true, bitExact: true, checkedWords: results.reduce((sum, value) => sum + value.checkedWords, 0), cases: results }, samples, warmups };
  } finally { plugin.createGpu = original; model?.dispose(); }
}

if (typeof window !== "undefined") {
  window.gpuBench = { status: "running", done: false, backend: "webgpu", check: "kev-final-rows" };
  main().catch((error) => { window.gpuBench = { ...window.gpuBench, status: "error", done: true, error: String(error?.message || error), correctness: { ok: false } }; log(error.stack || error); });
}
