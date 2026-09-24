// End-to-end decision benchmark. The page runs one typed choice request at a time so quality
// and latency have independent accounting. The canonical cases live in benchmarks/decisions;
// this adapter accepts the stable dataset exports used by the offline metrics harness.
import { Kevala, MODELS } from "../js/src/index.js";
import { DEFAULT_PERMUTATIONS, evaluateDecisions, expandPermutations, summarizeLatencies } from "../benchmarks/decisions/metrics.js";

const logNode = document.getElementById("log");
const summaryNode = document.getElementById("summary");
const log = (line) => { logNode.textContent += `${line}\n`; };
const fail = (message) => { throw new Error(message); };

window.decisionBench = { status: "idle", done: false, errors: [] };

const PROBABILITY_SUM_TOLERANCE = 0.0005;
const OPTION_LETTERS = "ABCDEFGHIJKLMNOP";

function params() {
  const out = new URLSearchParams(location.search);
  const hash = location.hash.replace(/^#/, "");
  for (const part of hash.split("&")) {
    if (!part || !part.includes("=")) continue;
    const [key, ...rest] = part.split("=");
    if (key) out.set(decodeURIComponent(key), decodeURIComponent(rest.join("=")));
  }
  return out;
}

function integerParam(query, name, fallback, minimum, maximum) {
  const raw = query.get(name);
  const value = raw === null || raw === "" ? fallback : Number(raw);
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    fail(`${name} must be an integer from ${minimum} to ${maximum}`);
  }
  return value;
}

async function loadDataset(name) {
  const files = name === "kevala"
    ? ["kevala-authored36.jsonl"]
    : name === "semif"
      ? ["semif-authored144.jsonl", "semif-perturbations108.jsonl"]
      : ["kevala-authored36.jsonl", "semif-authored144.jsonl", "semif-perturbations108.jsonl"];
  const manifestResponse = await fetch("../benchmarks/decisions/manifest.json", { cache: "no-store" });
  if (!manifestResponse.ok) throw new Error(`manifest.json: HTTP ${manifestResponse.status}`);
  const manifestBytes = new Uint8Array(await manifestResponse.arrayBuffer());
  let manifest;
  try {
    manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  } catch (error) {
    throw new Error(`manifest.json: invalid JSON: ${error.message}`);
  }
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) fail("manifest.json: expected an object");
  const manifestSha256 = await bytesHash(manifestBytes);
  const manifestEntry = (file) => {
    if (manifest.kevala?.path === file) return manifest.kevala;
    return manifest.semif?.fixtures?.find((entry) => entry?.path === file) || null;
  };
  const manifestCountKey = (file) => file === "kevala-authored36.jsonl"
    ? "kevala_authored"
    : file === "semif-authored144.jsonl"
      ? "semif_authored"
      : file === "semif-perturbations108.jsonl"
        ? "semif_perturbations"
        : null;
  const fileHashes = {};
  const load = async (file) => {
    const response = await fetch(`../benchmarks/decisions/${file}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const rows = new TextDecoder().decode(bytes).split(/\r?\n/).filter(Boolean).map((line, index) => {
      try { return JSON.parse(line); } catch (error) { throw new Error(`${file}:${index + 1}: invalid JSON: ${error.message}`); }
    });
    const sha256 = await bytesHash(bytes);
    const expected = manifestEntry(file);
    if (!expected || typeof expected !== "object") throw new Error(`${file}: missing manifest entry`);
    if (expected.rows !== rows.length) throw new Error(`${file}: manifest rows ${expected.rows} != fetched ${rows.length}`);
    const countKey = manifestCountKey(file);
    if (!countKey || manifest.counts?.[countKey] !== rows.length) throw new Error(`${file}: manifest count does not match fetched ${rows.length}`);
    if (expected.sha256 !== sha256) throw new Error(`${file}: manifest SHA-256 ${expected.sha256} != fetched ${sha256}`);
    fileHashes[file] = { rows: rows.length, sha256 };
    return rows;
  };
  const cases = (await Promise.all(files.map(load))).flat();
  if (!cases.length) fail(`canonical ${name} decision dataset is empty`);
  return { cases, files, manifest, manifestSha256, fileHashes };
}

function scalarText(value, fallback = "") {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return fallback;
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function optionValue(value, index) {
  if (typeof value === "string") return { id: value, text: value };
  if (!value || typeof value !== "object") return { id: String(index), text: String(value) };
  const id = scalarText(value.id ?? value.key ?? value.name ?? value.value, String(index));
  const text = scalarText(value.text ?? value.description ?? value.label ?? value.value ?? value.name, id);
  return { id, text };
}

function integerLikeOptionId(id) {
  return /^(?:0|[1-9]\d*)$/.test(id);
}

function normalizeCase(row, index) {
  if (!row || typeof row !== "object") fail(`dataset case ${index} is not an object`);
  const id = scalarText(row.id ?? row.caseId, `case-${index}`);
  const state = row.state ?? row.evidence ?? row.input;
  const criterion = row.criterion ?? row.instructions ?? row.question;
  const source = row.options ?? row.criteria;
  const options = Array.isArray(source)
    ? source.map(optionValue)
    : source && typeof source === "object"
      ? Object.entries(source).map(([key, value]) => ({ id: key, text: scalarText(value?.text ?? value?.description ?? value, key) }))
      : [];
  const expectedValue = row.gold_id ?? row.expectedId ?? row.expected ?? row.answer ?? row.target;
  const expected = typeof expectedValue === "object" && expectedValue !== null
    ? scalarText(expectedValue.id ?? expectedValue.key ?? expectedValue.option_id, "")
    : Number.isInteger(row.label) && row.label >= 0 && row.label < options.length
      ? options[row.label].id
      : scalarText(expectedValue, "");
  const tags = row.dataset ?? row.datasets ?? row.benchmark ?? row.family ?? row.suite ?? row.track ?? [];
  const suites = Array.isArray(tags) ? tags.map(String) : [String(tags)].filter(Boolean);
  const seen = new Set();
  if (state === undefined) fail(`${id}: dataset case has no state/evidence`);
  if (!criterion) fail(`${id}: dataset case has no criterion/instructions`);
  if (options.length < 2) fail(`${id}: dataset case needs at least two options`);
  if (options.some((option) => !option.id || seen.has(option.id) || !seen.add(option.id))) fail(`${id}: option IDs must be unique and nonempty`);
  if (options.some((option) => integerLikeOptionId(option.id))) fail(`${id}: option IDs must be non-integer text IDs`);
  return {
    ...row,
    id,
    state,
    criterion: String(criterion),
    options,
    gold_id: expected,
    suites,
    source: row,
  };
}

function questionFor(item, options) {
  return {
    q: {
      type: "choice",
      instructions: item.criterion,
      criteria: Object.fromEntries(options.map((option) => [option.id, option.text])),
    },
  };
}

function answerFor(response, options) {
  const answer = response?.answers?.q?.choice;
  if (typeof answer === "string") {
    const direct = options.find((option) => option.id === answer || option.text === answer);
    if (direct) return direct.id;
    // A SemIf-style implementation may expose its A..P readout label. Map it back to the
    // semantic option ID before comparing predictions across option permutations.
    const label = answer.match(/^[A-P]$/)?.[0];
    if (label) return options["ABCDEFGHIJKLMNOP".indexOf(label)]?.id ?? null;
  }
  return null;
}

function probabilitiesFor(response, options) {
  const raw = response?.raw_probabilities?.q;
  const fallback = response?.answers?.q?.probabilities;
  const values = raw !== undefined ? raw : fallback;
  if (!values || typeof values !== "object") return null;
  let keys;
  if (Array.isArray(values)) {
    if (values.length !== options.length) return null;
    keys = options.map((option) => option.id);
  } else {
    const actualKeys = Object.keys(values);
    const candidates = [
      options.map((option) => option.id),
      options.map((_, index) => String(index)),
      options.map((_, index) => OPTION_LETTERS[index]),
    ];
    keys = candidates.find((candidate) => candidate.length === actualKeys.length && candidate.every((key) => Object.hasOwn(values, key)) && actualKeys.every((key) => candidate.includes(key)));
    if (!keys) return null;
  }
  const probs = {};
  let sum = 0;
  for (let index = 0; index < options.length; index++) {
    const option = options[index];
    const value = Array.isArray(values) ? values[index] : values[keys[index]];
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) return null;
    probs[option.id] = value;
    sum += value;
  }
  if (Math.abs(sum - 1) > PROBABILITY_SUM_TOLERANCE) return null;
  return probs;
}

async function clockInfo() {
  const samples = [];
  let previous = performance.now();
  for (let i = 0; i < 64; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
    const current = performance.now();
    if (current > previous) samples.push(current - previous);
    previous = current;
  }
  return {
    source: "performance.now",
    minDeltaMs: samples.length ? Math.min(...samples) : null,
    quantized: samples.length ? Math.min(...samples) >= 50 : true,
    fine: samples.length > 0 && Math.min(...samples) < 50,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
  };
}

async function gpuInfo() {
  if (!navigator.gpu) return { available: false, isFallbackAdapter: null, name: null };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { available: false, isFallbackAdapter: null, name: null };
  const info = adapter.info || {};
  const fallback = Boolean(info.isFallbackAdapter || adapter.isFallbackAdapter);
  return {
    available: true,
    name: [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "WebGPU",
    vendor: info.vendor || null,
    architecture: info.architecture || null,
    device: info.device || null,
    description: info.description || null,
    isFallbackAdapter: fallback,
    features: [...adapter.features].sort(),
    limits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
  };
}

async function bytesHash(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function metadataHash(value) {
  const encoded = new TextEncoder().encode(JSON.stringify(value));
  return bytesHash(encoded);
}

function validateModelInfo(info, modelName, spec) {
  const header = info?.model;
  if (!header || typeof header !== "object") fail(`${modelName}: loaded pack has no model header`);
  const expected = {
    name: spec.name || modelName,
    revision: spec.revision,
    source: spec.repo ? `https://huggingface.co/${spec.repo}` : null,
  };
  for (const key of ["name", "revision", "source"]) {
    if (!expected[key] || header[key] !== expected[key]) fail(`${modelName}: model header ${key} ${JSON.stringify(header[key])} != catalog ${JSON.stringify(expected[key])}`);
  }
  if (info.arch !== spec.arch) fail(`${modelName}: loaded architecture ${JSON.stringify(info.arch)} != catalog ${JSON.stringify(spec.arch)}`);
}

async function run() {
  const query = params();
  const modelName = query.get("model") || "laya";
  const backend = query.get("backend") || "webgpu";
  const pack = query.get("pack") || "hosted";
  const datasetName = query.get("dataset") || "all";
  const permutations = integerParam(query, "permutations", 3, 1, 3);
  const warmups = integerParam(query, "warmups", 5, 1, 32);
  const batch = integerParam(query, "batch", 1, 1, 1);
  const profile = query.get("profile") === "1";
  if (!["webgpu", "wasm"].includes(backend)) fail(`backend must be webgpu or wasm, got ${backend}`);
  if (!["local", "hosted"].includes(pack)) fail(`pack must be local or hosted, got ${pack}`);
  if (!["all", "kevala", "semif"].includes(datasetName)) fail(`dataset must be all, kevala, or semif, got ${datasetName}`);
  if (!modelName || !MODELS[modelName]) fail(`unknown model ${modelName}`);
  const source = pack === "local" ? `../tmp/${modelName}-q8.kevala` : modelName;
  const dataset = await loadDataset(datasetName);
  const baseCases = dataset.cases.map(normalizeCase);
  if (!baseCases.length) fail(`dataset ${datasetName} selected no canonical cases`);
  const permutationNames = DEFAULT_PERMUTATIONS.slice(0, permutations);
  const cases = expandPermutations(baseCases, permutationNames);
  const spec = MODELS[modelName];
  const errors = [];
  const rows = [];
  let model = null;
  let profileRow = null;
  const clock = await clockInfo();
  const started = performance.now();
  window.decisionBench = { status: "running", done: false, errors, model: modelName, backend, dataset: datasetName };
  summaryNode.textContent = `loading ${modelName} (${backend})`;
  try {
    const beforeLoad = performance.now();
    model = await Kevala.load({ model: source, backend, cache: false, profile: false, stateCache: false });
    const loadWallMs = performance.now() - beforeLoad;
    const info = model.info || {};
    validateModelInfo(info, modelName, spec);
    const warmupStart = performance.now();
    for (let index = 0; index < warmups; index++) {
      await model.decide("kevala decision benchmark warmup", { warmup: { type: "choice", instructions: "Choose the warmup option.", criteria: { ready: "ready", idle: "idle" } } });
    }
    const warmupDecisionMs = performance.now() - warmupStart;
    if (profile) {
      const supported = await model.profile(true);
      if (supported) {
        const item = cases[0];
        const profileStart = performance.now();
        const response = await model.decide(item.state, questionFor(item, item.options));
        profileRow = { caseId: item.id, wallMs: Math.max(0.001, performance.now() - profileStart), timing: response.timing || null };
      } else {
        profileRow = { supported: false };
      }
      await model.profile(false);
    }
    const decisionStart = performance.now();
    const predictions = [];
    for (const item of cases) {
      const rowStart = performance.now();
      try {
        const response = await model.decide(item.state, questionFor(item, item.options));
        const wallMs = Math.max(0.001, performance.now() - rowStart);
        const prediction = answerFor(response, item.options);
        const rawProbabilities = probabilitiesFor(response, item.options);
        const invalid = !prediction || !rawProbabilities;
        predictions.push({ id: item.id, option_id: prediction, status: invalid ? "invalid" : "ok" });
        rows.push({
          caseId: item.id,
          baseId: item.base_id || item.id,
          permutation: item.permutation || "identity",
          options: item.options.map(({ id }) => id),
          expected: item.gold_id,
          prediction,
          correct: !invalid && prediction === item.gold_id,
          invalid,
          wallMs,
          timing: response.timing || null,
          rawProbabilities,
        });
        log(`${item.id}: ${prediction || "invalid"} ${wallMs.toFixed(1)} ms`);
      } catch (error) {
        const message = String(error?.message || error);
        errors.push({ caseId: item.id, permutation: item.permutation || "identity", message });
        predictions.push({ id: item.id, status: "error" });
        rows.push({ caseId: item.id, baseId: item.base_id || item.id, permutation: item.permutation || "identity", invalid: true, wallMs: Math.max(0.001, performance.now() - rowStart), error: message });
      }
    }
    const metricQuality = evaluateDecisions(cases, predictions);
    const expectedRows = cases.length;
    const invalidCount = metricQuality.invalid + metricQuality.missing;
    const correctCount = metricQuality.correct;
    const wall = rows.map((row) => row.wallMs);
    const revision = info.model?.revision || spec.revision || null;
    // The catalog value identifies the expected published pack. This page does not read the
    // complete pack bytes, so it must not describe the value as a verified download hash.
    const catalogHash = spec.packSha256 || null;
    const hash = catalogHash || await metadataHash({ model: modelName, revision, config: info.config || null });
    const quality = { ...metricQuality, total: expectedRows, valid: expectedRows - invalidCount, correct: correctCount, accuracy: expectedRows ? correctCount / expectedRows : 0, invalidCount };
    const result = {
      status: "done",
      done: true,
      model: modelName,
      modelRevision: revision,
      hash,
      hashScope: catalogHash ? "expected-catalog-pack-sha256" : "pack-header-metadata-sha256",
      catalogHash,
      catalogHashScope: catalogHash ? "expected-catalog-pack-sha256" : null,
      modelInfo: info.model || null,
      modelArch: info.arch || null,
      backend: info.backend || backend,
      threads: info.threads || null,
      gpuInfo: info.backend === "webgpu" ? await gpuInfo() : { available: false, isFallbackAdapter: null, name: null },
      loadMs: Number(info.loadMs ?? loadWallMs),
      warmupMs: Number(info.warmupMs ?? warmupDecisionMs),
      warmupDecisionMs,
      timingAllRows: rows,
      rawProbs: rows.map(({ caseId, permutation, rawProbabilities }) => ({ caseId, permutation, probabilities: rawProbabilities })),
      latency: summarizeLatencies(wall),
      quality,
      errors,
      profile: profileRow,
      stateCache: info.stateCache ?? null,
      metadata: {
        dataset: datasetName,
        datasetModule: dataset.manifest?.version || null,
        datasetFiles: dataset.files,
        datasetManifest: dataset.manifest,
        datasetManifestSha256: dataset.manifestSha256,
        datasetFileHashes: dataset.fileHashes,
        metricsApi: "benchmarks/decisions/metrics.js:evaluateDecisions",
        permutations: permutationNames,
        batch,
        warmups,
        profileDuringMeasurements: false,
        stateCache: info.stateCache ?? null,
        clock,
        modelInfo: info.model || null,
        modelArch: info.arch || null,
        measuredDecisionMs: performance.now() - decisionStart,
        elapsedMs: performance.now() - started,
      },
    };
    window.decisionBench = result;
    summaryNode.textContent = `${modelName}: ${quality.correct}/${quality.total} correct; p50 ${result.latency.p50Ms.toFixed(1)} ms`;
  } catch (error) {
    const message = String(error?.message || error);
    errors.push({ phase: "setup", message });
    window.decisionBench = { status: "error", done: true, model: modelName, backend, errors, error: message };
    summaryNode.textContent = `ERROR ${message}`;
    log(`ERROR ${message}`);
  } finally {
    model?.dispose();
  }
}

run().catch((error) => {
  const message = String(error?.message || error);
  window.decisionBench = { status: "error", done: true, errors: [{ phase: "startup", message }], error: message };
  summaryNode.textContent = `ERROR ${message}`;
  log(`ERROR ${message}`);
});
