import { GpuKev } from "../js/src/gpu-kev.js";
import { gpuLayouts } from "../js/src/pack-layout.js";
import { inErrorScopes, requestDevice } from "../js/src/gpu.js";
import { kernelSource } from "./wgsl.js";
import { Kevala } from "../js/src/index.js";
import clef from "../js/src/archs/clef.js";

const query = new URLSearchParams(location.search);
const log = (message) => document.getElementById("log").textContent += `${message}\n`;
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function exact(actual, expected, label) {
  if (actual.length !== expected.length) throw new Error(`${label}: output lengths differ`);
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Object.is(actual[i], expected[i])) throw new Error(`${label}: output ${i} differs: ${actual[i]} versus ${expected[i]}`);
    maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
  }
  return maxAbs;
}

async function checkRows() {
  const runtime = await requestDevice();
  runtime.wgsl = await kernelSource();
  const device = runtime.device;
  const u = GPUBufferUsage;
  const cases = [];
  const owned = [];
  const make = (data, usage) => {
    const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | u.COPY_DST });
    owned.push(buffer);
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  try {
    for (const dtype of ["q8", "bf16", "f32"]) {
      const config = { hidden: 32, layers: 0, heads: 4, kv_heads: 1, lin_key_heads: 2, lin_heads: 2 };
      const header = { config: { arch: "kev" }, model: {}, tokenizer: { offset: 4096, size: 0 }, tensors: [] };
      const gpu = new GpuKev(runtime, gpuLayouts(header, 4096, () => true)[1], config);
      gpu.cap.T2 = 8;
      gpu.owned = [];
      gpu.zeros = make(new Float32Array(4), u.STORAGE);
      gpu.x2 = make(new Float32Array(8 * 32), u.STORAGE | u.COPY_SRC);
      gpu.rowPipelines = new Map();
      const table = { info: { name: "emb", dtype, shape: [6, 32], block: 32 }, shards: [] };
      for (const start of [0, 3]) {
        const values = Float32Array.from({ length: 3 * 32 }, (_, i) => (start * 32 + i - 80) * 0.125);
        let data;
        if (dtype === "q8") data = Int8Array.from(values, (value) => value * 8);
        else if (dtype === "bf16") data = Uint16Array.from(new Uint32Array(values.buffer), (word) => word >>> 16);
        else {
          values[0] = 2 ** 20;
          values[values.length - 1] = 2 ** -30;
          data = values;
        }
        table.shards.push({ start, count: 3, buf: make(data, u.STORAGE), sbuf: dtype === "q8" ? make(new Float32Array(3).fill(0.125), u.STORAGE) : null });
      }
      gpu.weights.tensors.set("emb", table);
      const pipeline = await gpu.rowPipeline(table);
      for (const ids of [Uint32Array.of(4, 1, 3, 4, 0), Uint32Array.of(5, 3, 5), Uint32Array.of(0, 1, 2), Uint32Array.of(3, 4, 5)]) {
        const baseline = await gpu.weightRows("emb", ids);
        const input = { table, ids, pipeline };
        const values = await inErrorScopes(device, async () => {
          gpu.prepareRowInput(input);
          const bytes = ids.length * 32 * 4;
          const read = make(new Uint8Array(bytes), u.MAP_READ);
          const encoder = device.createCommandEncoder();
          gpu.encodeRowInput(encoder, input);
          encoder.copyBufferToBuffer(gpu.x2, 0, read, 0, bytes);
          device.queue.submit([encoder.finish()]);
          await read.mapAsync(GPUMapMode.READ, 0, bytes);
          const result = new Float32Array(read.getMappedRange(0, bytes).slice(0));
          read.unmap();
          return result;
        });
        const maxAbs = exact(values, baseline, `${dtype} rows ${ids}`);
        cases.push({ dtype, ids: [...ids], values: values.length, maxAbs });
      }
      for (const buffer of gpu.owned) buffer.destroy();
      log(`${dtype}: direct and sharded resident rows match the readback path exactly`);
    }
    const info = runtime.adapter.info || {};
    return { adapter: { name: runtime.name, vendor: info.vendor || null, architecture: info.architecture || null, device: info.device || null, description: info.description || null, isFallbackAdapter: false }, correctness: { ok: true, cases } };
  } finally {
    for (const buffer of owned) buffer.destroy();
    device.destroy();
  }
}

async function checkModel(pack) {
  const runs = Number(query.get("runs") || 3);
  if (!Number.isInteger(runs) || runs < 1 || runs > 20) throw new Error("runs must be an integer from 1 to 20");
  const requests = [
    "The package arrived damaged. Please refund the duplicate payment today.",
    "The customer received an invoice with two charges for a single order. They called support yesterday and have not received a reply. They need the duplicate payment refunded before the end of the week.",
    "This is a routine announcement about our updated opening hours. No payment or account changes are required. The support team will answer general questions during the normal business day.",
  ];
  const questions = {
    urgent: { type: "noul", instructions: "Does the customer need urgent action?" },
    team: { type: "choice", instructions: "Which team should handle this?", criteria: { billing: "charges, payments, refunds", shipping: "delivery, damaged goods", other: "general information" } },
  };
  const previous = clef.run;
  let resident = false;
  let hidden;
  let headInputs = [];
  let captureHead = false;
  let reads = { emb: 0, lexical: 0 };
  clef.run = async (engine, batch) => {
    engine.gpu.rowInputEnabled = resident;
    captureHead = false;
    headInputs = [];
    if (!engine.gpu.clefCheckInstrumented) {
      const oldRows = engine.gpu.weightRows.bind(engine.gpu);
      engine.gpu.weightRows = (...args) => {
        reads[args[0] === "emb" ? "emb" : "lexical"]++;
        return oldRows(...args);
      };
      const oldForward = engine.gpu.forward.bind(engine.gpu);
      engine.gpu.forward = async (...args) => {
        const result = await oldForward(...args);
        hidden = result.slice();
        captureHead = true;
        return result;
      };
      const oldPut = engine.coord.put.bind(engine.coord);
      engine.coord.put = (data) => {
        if (captureHead) headInputs.push(data.slice());
        return oldPut(data);
      };
      engine.gpu.clefCheckInstrumented = true;
    }
    return previous(engine, batch);
  };
  let model;
  try {
    model = await Kevala.load({ model: new URL(pack, location.href).href, backend: "webgpu", onPage: true, cache: false, gpuKernel: "auto", onProgress: (progress) => { if (progress.message) log(progress.message); } });
    const decide = async (state, mode) => {
      resident = mode;
      reads = { emb: 0, lexical: 0 };
      window.gpuBench.stage = mode ? "resident decision" : "baseline decision";
      const before = performance.now();
      const result = await model.decide(state, questions);
      if (!result.answers || Object.keys(result.answers).length !== Object.keys(questions).length) throw new Error("Clef did not return every requested answer");
      if (headInputs.length !== 2) throw new Error(`expected hidden and lexical inputs to the Clef head, received ${headInputs.length}`);
      return { ms: performance.now() - before, answers: result.answers, hidden, headInputs, reads: { ...reads } };
    };
    await decide(requests[0], false);
    await decide(requests[0], true);
    const cases = [];
    const samples = requests.map(() => ({ baseline: [], resident: [] }));
    for (let sample = 0; sample < runs; sample++) for (const [index, state] of requests.entries()) {
      const pair = {};
      for (const mode of sample % 2 ? [true, false] : [false, true]) pair[mode ? "resident" : "baseline"] = await decide(state, mode);
      exact(pair.resident.hidden, pair.baseline.hidden, `request ${index} full hidden`);
      exact(pair.resident.headInputs[0], pair.baseline.headInputs[0], `request ${index} head hidden bytes`);
      exact(new Float32Array(pair.resident.headInputs[1].buffer), new Float32Array(pair.baseline.headInputs[1].buffer), `request ${index} head lexical means`);
      if (JSON.stringify(pair.resident.answers) !== JSON.stringify(pair.baseline.answers)) throw new Error(`request ${index}: public answers differ`);
      if (pair.resident.reads.emb || pair.resident.reads.lexical) throw new Error(`warm resident request performed row readbacks: ${JSON.stringify(pair.resident.reads)}`);
      for (const mode of ["baseline", "resident"]) samples[index][mode].push(pair[mode].ms);
      cases.push({ request: index, sample, baselineMs: pair.baseline.ms, residentMs: pair.resident.ms, baselineReads: pair.baseline.reads, residentReads: pair.resident.reads, hiddenValues: pair.resident.hidden.length, lexicalValues: pair.resident.headInputs[1].byteLength / 4, answers: pair.resident.answers, maxAbs: 0 });
      log(`request ${index} pair ${sample}: exact hidden, head inputs and public answers; ${pair.baseline.ms.toFixed(2)} -> ${pair.resident.ms.toFixed(2)} ms`);
    }
    const timings = samples.map((record, request) => ({ request, baselineMs: median(record.baseline), residentMs: median(record.resident), speedup: median(record.baseline) / median(record.resident) }));
    return { pack, info: model.info, correctness: { ok: true, cases }, timings, metricMs: Math.exp(timings.reduce((sum, item) => sum + Math.log(item.residentMs), 0) / timings.length) };
  } finally {
    model?.dispose();
    clef.run = previous;
  }
}

window.gpuBench = { status: "running", done: false, backend: "webgpu" };
async function run() {
  const started = performance.now();
  const rows = await checkRows();
  Object.assign(window.gpuBench, { adapter: rows.adapter, correctness: rows.correctness, stage: "model load" });
  const model = query.has("model") ? await checkModel(query.get("model")) : null;
  window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "clef-resident-rows", method: model ? "awaited decide paired same loaded model" : "GPU full-row exactness", metricMs: model?.metricMs || performance.now() - started, adapter: rows.adapter, correctness: rows.correctness, model };
  log(JSON.stringify(window.gpuBench));
}
run().catch((error) => {
  window.gpuBench = { ...window.gpuBench, status: "error", done: true, backend: "webgpu", error: String(error?.message || error), stack: String(error?.stack || error) };
  log(error.stack || error);
});
