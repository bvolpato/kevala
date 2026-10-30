import { Kevala } from "../js/src/index.js";
import { GpuKev } from "../js/src/gpu-kev.js";
import { median } from "../js/src/stats.js";

const query = new URLSearchParams(location.search);
const log = (line) => { document.getElementById("log").textContent += `${line}\n`; };
window.gpuBench = { status: "running", done: false, backend: "webgpu" };

function compare(actual, expected, absolute, relative, label) {
  if (actual.length !== expected.length) throw new Error(`${label}: output length mismatch`);
  let maxAbs = 0;
  for (let index = 0; index < actual.length; index++) {
    const delta = Math.abs(actual[index] - expected[index]);
    if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index]) || delta > absolute + relative * Math.max(1, Math.abs(expected[index]))) {
      throw new Error(`${label}: mismatch at ${index}: ${actual[index]} vs ${expected[index]}`);
    }
    maxAbs = Math.max(maxAbs, delta);
  }
  return maxAbs;
}

async function run() {
  let trunk;
  let model;
  const originalInit = GpuKev.prototype.init;
  GpuKev.prototype.init = async function() {
    await originalInit.call(this);
    trunk = this;
  };
  try {
    const pack = new URL(query.get("pack") || "../tmp/kev-0.8b-bf16.kevala", location.href).href;
    model = await Kevala.load({ model: pack, backend: "webgpu", onPage: true, cache: false, stateCache: false,
      onProgress: (progress) => { if (progress.message) log(progress.message); } });
    GpuKev.prototype.init = originalInit;
    if (!trunk) throw new Error("Qwen trunk was not captured on the page");
    const calibration = Object.fromEntries([...trunk.precisionTuning].map(([dtype, tuning]) => [dtype, {
      matmul: tuning.diagnostics, matvecSelection: [...trunk.precisionMatvec.get(dtype).selection],
    }]));
    log(`Loaded ${trunk.cfg.hidden}-wide Qwen. Calibration: ${JSON.stringify(calibration)}`);
    const forward = trunk.forward;
    const encode = trunk.encode;
    let hidden;
    let firstInput;
    let captureBatch = false;
    let batchInfo;
    trunk.encode = function(encoder, operations, tokens, segments) {
      if (captureBatch) batchInfo.encoded.push({
        stage: operations === this.ops[0] ? "state" : operations === this.ops[1] ? "branch" : "chunk",
        tokens, segments,
      });
      return encode.call(this, encoder, operations, tokens, segments);
    };
    trunk.forward = async function(...args) {
      if (!firstInput && args[1] instanceof Float32Array && args[1].length >= this.cfg.hidden) firstInput = args[1].slice(0, this.cfg.hidden);
      if (captureBatch) batchInfo = { preparedT1: args[2].T1, preparedT2: args[2].T2, encoded: [] };
      hidden = await forward.apply(this, args);
      return hidden;
    };
    const questions = {
      team: { type: "choice", instructions: "Which team should handle this?", criteria: {
        billing: "invoices, payments, refunds", technical: "bugs, outages", other: "anything else",
      } },
    };
    const cases = [];
    const performanceCases = [];
    for (const repeats of [1, 5, 20]) {
      const state = `We were billed twice. Please refund the duplicate payment. ${"The customer expects help with their invoice. ".repeat(repeats)}`;
      const times = { generic: [], auto: [] };
      const batches = {};
      for (let pair = 0; pair < 5; pair++) {
        captureBatch = pair === 0;
        const results = {};
        const order = pair % 2 ? ["auto", "generic"] : ["generic", "auto"];
        for (const kind of order) {
          trunk.gpuKernel = kind;
          const started = performance.now();
          const response = await model.decide(state, questions);
          const elapsed = performance.now() - started;
          results[kind] = { response, hidden: hidden.slice(), elapsed };
          if (captureBatch) batches[kind] = batchInfo;
          if (pair >= 2) times[kind].push(elapsed);
        }
        const probability = (response) => Object.keys(response.raw_probabilities).sort().flatMap((key) => response.raw_probabilities[key]);
        const row = {
          repeats, pair, warmup: pair < 2,
          hiddenMaxAbs: compare(results.auto.hidden, results.generic.hidden, 0.001, 0.0001, "full hidden rows"),
          probabilityMaxAbs: compare(probability(results.auto.response), probability(results.generic.response), 0.0001, 0, "raw probabilities"),
          genericMs: results.generic.elapsed, autoMs: results.auto.elapsed,
        };
        cases.push(row);
        log(JSON.stringify(row));
      }
      const genericMs = median(times.generic), autoMs = median(times.auto);
      performanceCases.push({ repeats, batches, genericMs, autoMs, speedup: genericMs / autoMs, samples: times });
    }
    captureBatch = false;
    if (!firstInput) throw new Error("No loaded-model embeddings were captured");
    const batch = { T1: 0, T2: 1, R: 1, states: [[0, 0]], branches: [[0, 1, 0]], rows: [0] };
    const single = {};
    for (const kind of ["generic", "auto"]) {
      trunk.gpuKernel = kind;
      single[kind] = await forward.call(trunk, new Float32Array(), firstInput, batch, new Uint32Array());
    }
    const singleTokenMaxAbs = compare(single.auto, single.generic, 0.001, 0.0001, "single-token full hidden row");
    const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
    const genericMs = geometricMean(performanceCases.map((item) => item.genericMs));
    const autoMs = geometricMean(performanceCases.map((item) => item.autoMs));
    window.gpuBench = { status: "done", done: true, backend: "webgpu", check: "loaded-qwen-projections",
      method: "geometric mean of paired per-shape completed-decision medians", metricMs: autoMs,
      adapter: { name: trunk.name }, pack, calibration,
      correctness: { ok: true, cases, singleTokenMaxAbs },
      performance: { genericMs, autoMs, speedup: genericMs / autoMs, cases: performanceCases, warmupPairs: 2, measuredPairs: 3 },
    };
  } finally {
    GpuKev.prototype.init = originalInit;
    model?.dispose();
  }
}

run().catch((error) => {
  window.gpuBench = { status: "error", done: true, backend: "webgpu", error: String(error?.message || error) };
  log(error?.stack || String(error));
});
