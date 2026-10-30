import { Kevala, MODELS } from "../js/src/index.js";
import { archPlugin } from "../js/src/archs/index.js";
import { packPrecision } from "../js/src/pack-layout.js";
import { median } from "../js/src/stats.js";

const query = new URLSearchParams(location.search);
const button = document.getElementById("run");
const log = (line) => { document.getElementById("log").textContent += `${line}\n`; };
window.gpuBench = { status: "idle", done: false, backend: "webgpu" };

const QUESTIONS = {
  team: { type: "choice", instructions: "Which team should handle this?", criteria: {
    billing: "invoices, payments, refunds", technical: "software bugs and outages", other: "anything else",
  } },
  urgent: { type: "noul", instructions: "Does the customer request urgent help?" },
  frustration: { type: "score", instructions: "How frustrated is the customer?", criteria: ["Calm", "Concerned", "Very angry"] },
};
const PARAGRAPHS = [
  "A customer reports two charges for the same monthly invoice. They ask the billing team to investigate and refund the duplicate payment today.",
  "The first payment appears on Monday and the second on Tuesday. Both bank entries have the same amount and account reference. The customer says they authorized only one payment.",
  "Their subscription remains active and they can sign in normally. They have not reported a software error or a service outage. Their concern is the extra money taken from their account.",
  "A support agent has checked the invoice history and found one invoice for the month. The agent has asked the payments team to compare the invoice with the bank transaction records.",
  "The customer has already sent a copy of the two bank entries. They want a written explanation and a clear date for the refund. They are concerned that another duplicate payment could happen next month.",
  "The support team plans to reply after reconciling the payments. The account should remain available during the investigation. The requested action is to resolve the duplicate charge and explain the outcome to the customer.",
];
const RAW_TOLERANCE = 5e-4;
const SUM_TOLERANCE = 5e-4;

function require(condition, message) { if (!condition) throw new Error(message); }
function integer(name, fallback, min, max) {
  const value = query.has(name) ? Number(query.get(name)) : fallback;
  require(Number.isSafeInteger(value) && value >= min && value <= max, `${name} must be an integer from ${min} to ${max}`);
  return value;
}
function number(value, min, max, label) {
  require(typeof value === "number" && Number.isFinite(value) && value >= min && value <= max, `${label} must be finite and in [${min}, ${max}], got ${value}`);
}
function distribution(value, keys, tolerance, label) {
  require(value && typeof value === "object", `${label}: missing distribution`);
  if (Array.isArray(value)) require(value.length === keys.length, `${label}: wrong option count`);
  else require(Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key)), `${label}: wrong option keys`);
  const values = keys.map((key, index) => Array.isArray(value) ? value[index] : value[key]);
  values.forEach((value, index) => number(value, 0, 1, `${label}[${index}]`));
  const sum = values.reduce((total, value) => total + value, 0);
  require(Math.abs(sum - 1) <= tolerance, `${label}: probabilities sum to ${sum}, tolerance ${tolerance}`);
  return values;
}
function validate(response, unit, label) {
  require(response && typeof response.model === "string" && response.model.length > 0, `${label}: missing model identity`);
  require(response.answers && Object.keys(response.answers).length === 3, `${label}: wrong answer count`);
  const probabilities = {};
  for (const [id, question] of Object.entries(QUESTIONS)) {
    const answer = response.answers[id];
    require(answer?.type === question.type, `${label}.${id}: expected ${question.type} answer`);
    const keys = question.type === "choice" ? Object.keys(question.criteria) : question.type === "score" ? ["0", "1", "2"] : ["false", "true"];
    if (question.type === "noul") number(answer.noul, 0, 1, `${label}.${id}.noul`);
    else {
      distribution(answer.probabilities, keys, Math.max(SUM_TOLERANCE, keys.length * unit / 2 + 1e-8), `${label}.${id}.probabilities`);
      if (question.type === "choice") require(keys.includes(answer.choice), `${label}.${id}: invalid chosen option ${answer.choice}`);
      else {
        number(answer.score, 0, keys.length - 1, `${label}.${id}.score`);
        require(answer.legend && keys.every((key) => answer.legend[key] === question.criteria[Number(key)]), `${label}.${id}: incorrect score legend`);
      }
    }
    if (answer.confidence !== undefined) number(answer.confidence, 0, 1, `${label}.${id}.confidence`);
    if (answer.action?.act_probability !== undefined) number(answer.action.act_probability, 0, 1, `${label}.${id}.act_probability`);
    const raw = response.raw_probabilities?.[id];
    probabilities[id] = raw !== undefined
      ? distribution(raw, keys, SUM_TOLERANCE, `${label}.${id}.raw_probabilities`)
      : question.type === "noul" ? [1 - answer.noul, answer.noul] : keys.map((key) => answer.probabilities[key]);
  }
  require(Number.isSafeInteger(response.usage?.input_tokens) && response.usage.input_tokens > 0, `${label}: invalid input token usage`);
  require(Number.isSafeInteger(response.usage?.output_tokens) && response.usage.output_tokens >= 0, `${label}: invalid output token usage`);
  return { probabilities, answers: response.answers, usage: response.usage, raw: response.raw_probabilities !== undefined };
}
function parity(actual, expected, unit, label, probabilityTolerance = RAW_TOLERANCE) {
  let maxProbabilityDelta = 0;
  const nearTieChoices = [];
  for (const id of Object.keys(QUESTIONS)) {
    const tolerance = actual.raw && expected.raw ? probabilityTolerance : Math.max(probabilityTolerance, unit);
    actual.probabilities[id].forEach((value, index) => {
      const delta = Math.abs(value - expected.probabilities[id][index]);
      maxProbabilityDelta = Math.max(maxProbabilityDelta, delta);
      if (delta > tolerance + 1e-8) throw Object.assign(new Error(`${label}.${id}: probability delta ${delta} exceeds ${tolerance}`), {
        comparison: { label, question: id, optionIndex: index, actual: value, expected: expected.probabilities[id][index], delta, tolerance,
          actualProbabilities: actual.probabilities[id], expectedProbabilities: expected.probabilities[id], actualAnswer: actual.answers[id], expectedAnswer: expected.answers[id] },
      });
    });
    for (const field of ["noul", "score", "confidence"]) if (expected.answers[id][field] !== undefined) {
      require(Math.abs(actual.answers[id][field] - expected.answers[id][field]) <= 2 * RAW_TOLERANCE + unit + 1e-8, `${label}.${id}.${field}: scalar answer changed`);
    }
    if (QUESTIONS[id].type === "choice" && actual.answers[id].choice !== expected.answers[id].choice) {
      const keys = Object.keys(QUESTIONS[id].criteria);
      const gap = expected.probabilities[id][keys.indexOf(expected.answers[id].choice)] - expected.probabilities[id][keys.indexOf(actual.answers[id].choice)];
      const tieTolerance = actual.raw && expected.raw ? RAW_TOLERANCE : Math.max(RAW_TOLERANCE, unit);
      require(gap <= 2 * tieTolerance + unit, `${label}.${id}: chosen option changed outside a near tie`);
      nearTieChoices.push({ id, gap });
    }
  }
  require(actual.usage.input_tokens === expected.usage.input_tokens, `${label}: input token usage changed`);
  return { maxProbabilityDelta, nearTieChoices };
}

async function run() {
  if (button.disabled) return;
  button.disabled = true;
  document.getElementById("log").textContent = "";
  const result = window.gpuBench = { status: "running", done: false, backend: "webgpu", stage: "parameters", startedAt: new Date().toISOString() };
  let model;
  let runtime;
  let header;
  let timer;
  const hooks = [];
  const started = performance.now();
  const controller = new AbortController();
  const cancelled = new Promise((_, reject) => controller.signal.addEventListener("abort", () => reject(controller.signal.reason), { once: true }));
  const wait = (promise) => Promise.race([promise, cancelled]);
  const stage = (name) => { result.stage = name; log(name); };
  try {
    const id = query.get("model");
    require(id === null || Object.hasOwn(MODELS, id), `unknown catalog model ${id}`);
    const pack = query.has("pack") ? new URL(query.get("pack"), location.href).href : null;
    const selected = pack ? id ? { name: id, url: pack } : pack : id || "laya";
    const timed = query.get("timing") === "1" || query.has("samples");
    const samples = integer("samples", timed ? 5 : 1, 1, 10);
    const warmups = integer("warmups", timed ? 2 : 0, 0, 3);
    const timeoutSeconds = integer("timeout", 600, 30, 1800);
    const paragraphCounts = query.has("paragraphCounts") ? query.get("paragraphCounts").split(",").map(Number) : [1, 3, 6];
    require(paragraphCounts.length === 3 && paragraphCounts.every((value, index) => Number.isSafeInteger(value) && value >= 1 && value <= 6 && (index === 0 || value > paragraphCounts[index - 1])),
      "paragraphCounts must contain exactly three unique ascending integers from 1 through 6");
    const inputs = paragraphCounts.map((paragraphs, index) => ({ label: ["short", "medium", "long"][index], state: PARAGRAPHS.slice(0, paragraphs).join(" ") }));
    const batchOverride = query.has("batchTolerance") ? Number(query.get("batchTolerance")) : null;
    require(!query.has("batchTolerance") || query.get("batchTolerance").trim() !== "", "batchTolerance must be a number from 0 through 0.01");
    if (batchOverride !== null) number(batchOverride, 0, 0.01, "batchTolerance");
    const features = query.has("features") ? query.get("features").split(",").filter((value) => value && value !== "none") : null;
    require(features === null || features.every((feature) => ["timestamp-query", "shader-f16", "subgroups"].includes(feature)), "features must list timestamp-query, shader-f16, subgroups, or none");
    result.request = { model: id || (pack ? null : "laya"), pack, samples, warmups, features, timeoutSeconds, paragraphCounts, batchTolerance: batchOverride };
    timer = setTimeout(() => {
      const error = Object.assign(new Error(`model smoke check exceeded ${timeoutSeconds} seconds`), { code: "SMOKE_TIMEOUT" });
      controller.abort(error);
      model?.dispose(error);
    }, timeoutSeconds * 1000);
    for (const name of ["laya", "kev", "gemma4", "clef"]) {
      const plugin = archPlugin(name);
      const create = plugin.createGpu;
      hooks.push(() => { plugin.createGpu = create; });
      plugin.createGpu = function(gpu, ...args) { runtime = gpu; header = args[1]; return create.call(this, gpu, ...args); };
    }
    stage("load");
    const loadStart = performance.now();
    model = await wait(Kevala.load({ model: selected, backend: "webgpu", onPage: true, cache: false, stateCache: false,
      signal: controller.signal, ...(features === null ? {} : { gpuFeatures: features }),
      ...(query.get("baseline") === "1" ? { gpuBaseline: true } : {}),
      ...(query.has("precision") ? { precision: query.get("precision") } : {}),
      onProgress: (progress) => { result.progress = progress; },
    }));
    hooks.splice(0).reverse().forEach((restore) => restore());
    result.loadMs = performance.now() - loadStart;
    result.modelInfo = model.info;
    require(model.info.backend === "webgpu" && runtime, "model did not load the requested WebGPU backend");
    const catalogId = id || (pack ? null : "laya");
    if (catalogId) {
      const spec = MODELS[catalogId];
      require(model.info.arch === spec.arch, `${catalogId}: loaded architecture ${model.info.arch} differs from catalog ${spec.arch}`);
      require(model.info.model?.name === (spec.name || catalogId), `${catalogId}: loaded model header name ${model.info.model?.name} differs from catalog`);
      const expected = { source: `https://huggingface.co/${spec.repo}`, revision: spec.revision,
        ...(spec.base ? { base: `https://huggingface.co/${spec.base.repo}`, base_revision: spec.base.revision } : {}) };
      const checked = ["name", "arch"];
      for (const [key, value] of Object.entries(expected)) if (model.info.model[key] !== undefined && value !== undefined) {
        require(model.info.model[key] === value, `${catalogId}: model header ${key} ${model.info.model[key]} differs from catalog ${value}`);
        checked.push(key);
      }
      result.identity = { catalogId, checked, contentHashVerified: false };
    } else result.identity = { catalogId: null, checked: [], scope: "pack header only; catalog identity was not requested", contentHashVerified: false };
    const info = runtime.adapter.info || {};
    result.adapter = { available: true, name: runtime.name, vendor: info.vendor || null, architecture: info.architecture || null,
      device: info.device || null, description: info.description || null,
      isFallbackAdapter: Boolean(info.isFallbackAdapter || runtime.adapter.isFallbackAdapter),
      features: [...runtime.device.features].sort(), limits: { maxBufferSize: runtime.device.limits.maxBufferSize,
        maxStorageBufferBindingSize: runtime.device.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: runtime.device.limits.maxComputeWorkgroupStorageSize } };
    require(!result.adapter.isFallbackAdapter, "model used a software WebGPU adapter");
    const config = model.info.config;
    const unit = model.info.arch === "gemma4" || model.info.arch === "kev" && config.response_version !== 1 && config.readout !== "h2o-lightning" ? 0.01 : 0.0001;
    const batchTolerance = batchOverride ?? Math.max(RAW_TOLERANCE, unit / 2);
    result.request.batchTolerance = batchTolerance;
    result.tolerances = { rawProbability: RAW_TOLERANCE, batchProbability: batchTolerance, probabilitySum: SUM_TOLERANCE, roundedAnswerUnit: unit,
      source: "strict same-model smoke guards with family response rounding; batchTolerance is independently configurable", accuracyEvaluated: false };
    result.batchScope = { precision: packPrecision(header), probabilityTolerance: batchTolerance,
      scope: "independent records in one loaded model; token layout and split-K reduction can differ from single requests",
      toleranceSource: batchOverride !== null ? "explicit query override" : "API compatibility bound: max(5e-4, half the rounded response unit)",
      numericalEquivalenceClaimed: false,
      baselineComparison: { performed: false, evidence: "retain baseline revision and before/after raw-vector artifacts with the run; this harness does not compare code revisions" },
      accuracyEvaluated: false };
    const checks = result.correctness = { ok: false, cases: [], repeat: null, batch: [] };
    const references = [];
    const latencies = [];
    for (const input of inputs) {
      stage(`decide:${input.label}`);
      const times = [];
      let reference;
      for (let iteration = 0; iteration < warmups + samples; iteration++) {
        const before = performance.now();
        const response = await wait(model.decide(input.state, QUESTIONS));
        const elapsed = performance.now() - before;
        const checked = validate(response, unit, input.label);
        if (reference) parity(checked, reference, unit, `${input.label}:repeat`);
        else reference = checked;
        if (iteration >= warmups) times.push(elapsed);
      }
      references.push(reference);
      const record = { label: input.label, medianMs: median(times), samplesMs: times, usage: reference.usage, answers: reference.answers, probabilities: reference.probabilities };
      checks.cases.push(record);
      latencies.push(record.medianMs);
      log(JSON.stringify(record));
    }
    stage("repeat:short");
    const repeated = validate(await wait(model.decide(inputs[0].state, QUESTIONS)), unit, "repeated short");
    checks.repeat = parity(repeated, references[0], unit, "repeated short");
    stage("batch:short+medium");
    const batch = await wait(model.decideMany(inputs.slice(0, 2).map((input) => ({ state: input.state, questions: QUESTIONS }))));
    require(Array.isArray(batch) && batch.length === 2, "batch returned the wrong response count");
    checks.batchResponses = batch.map((response, index) => validate(response, unit, `batch:${index}`));
    checks.batchResponses.forEach((response, index) => checks.batch.push({ label: inputs[index].label,
      ...parity(response, references[index], unit, `batch:${index}`, batchTolerance) }));
    require(!runtime.lost, `WebGPU device was lost: ${runtime.lost?.message}`);
    checks.ok = true;
    Object.assign(result, { status: "done", done: true, stage: "complete", method: "geometric mean of completed-decision wall medians; cache disabled",
      metricMs: Math.exp(latencies.reduce((sum, value) => sum + Math.log(Math.max(0.001, value)), 0) / latencies.length), elapsedMs: performance.now() - started });
  } catch (error) {
    Object.assign(result, { status: "error", done: true, error: String(error?.message || error), errorCode: error?.code || null,
      errorStack: String(error?.stack || error), ...(error?.comparison ? { comparison: error.comparison } : {}), elapsedMs: performance.now() - started });
    log(result.errorStack);
  } finally {
    clearTimeout(timer);
    hooks.reverse().forEach((restore) => restore());
    model?.dispose();
    runtime?.device.destroy();
    result.disposed = true;
    button.disabled = false;
  }
}

button.addEventListener("click", run);
if (query.get("auto") === "1") run();
