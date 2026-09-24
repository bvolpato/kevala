#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_PERMUTATIONS,
  decisionDataset,
  evaluateDecisions,
  expandPermutations,
  goldOptionId,
} from "../benchmarks/decisions/metrics.js";
import { MODELS } from "../js/src/source.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BASE_MODEL_FILES = Object.freeze({
  laya: "decision-laya.json",
  "kev-0.8b": "decision-kev-0.8b.json",
  "kev-4b": "decision-kev-4b.json",
  "kev-9b": "decision-kev-9b.json",
  "semif-qwen3.5-0.8b": "decision-semif-qwen3.5-0.8b.json",
  "semif-qwen3.5-2b": "decision-semif-qwen3.5-2b.json",
  "semif-qwen3.5-4b": "decision-semif-qwen3.5-4b.json",
  "gemma-4-e2b": "decision-gemma-4-e2b.json",
  "gemma-4-e4b": "decision-gemma-4-e4b.json",
});
const OPTIONAL_MODEL_FILES = Object.freeze({
  "bruv1-0.8b": "decision-bruv1-0.8b.json",
  "bruv1-4b": "decision-bruv1-4b.json",
});
const MODEL_FILES = Object.freeze({ ...BASE_MODEL_FILES, ...OPTIONAL_MODEL_FILES });
const MODEL_ALIASES = Object.freeze({
  "gemma-4-e2b-it": "gemma-4-e2b",
  "gemma-4-e4b-it": "gemma-4-e4b",
  "semif-0.8b": "semif-qwen3.5-0.8b",
  "semif-2b": "semif-qwen3.5-2b",
  "semif-4b": "semif-qwen3.5-4b",
});
const DEFAULT_REFERENCES = ["gemma-base-vs-it.json"];
const DEFAULT_PROBABILITY_TOLERANCE = 1e-3;
const ARGMAX_TOLERANCE = 1e-3;
const TYPED_OPTION_RENDERING = Object.freeze({
  name: "kev-typed-choice-v1",
  choiceDescription: "${id}: ${text}",
  emptyDescription: "${id}",
});
const DATASET_FILES = [
  "kevala-authored36.jsonl",
  "semif-authored144.jsonl",
  "semif-perturbations108.jsonl",
];

function parseArgs(argv) {
  const args = { results: resolve(ROOT, "tmp"), fixtures: resolve(ROOT, "benchmarks/decisions"), references: [], optionalModels: [], output: null, markdown: null, checkSummary: null, probabilityTolerance: DEFAULT_PROBABILITY_TOLERANCE };
  for (let index = 0; index < argv.length; index++) {
    const value = argv[index];
    const next = () => {
      if (index + 1 >= argv.length) throw new Error(`${value} needs a value`);
      return argv[++index];
    };
    if (value === "--results") args.results = resolvePath(next());
    else if (value === "--fixtures") args.fixtures = resolvePath(next());
    else if (value === "--reference") args.references.push(resolvePath(next()));
    else if (value === "--include-model") {
      const model = next();
      if (!Object.hasOwn(OPTIONAL_MODEL_FILES, model)) throw new Error(`unknown optional model ${model}`);
      if (!args.optionalModels.includes(model)) args.optionalModels.push(model);
    }
    else if (value === "--output") args.output = resolvePath(next());
    else if (value === "--markdown") args.markdown = resolvePath(next());
    else if (value === "--check-summary") args.checkSummary = resolvePath(next());
    else if (value === "--probability-tolerance") args.probabilityTolerance = Number(next());
    else if (value === "--help" || value === "-h") {
      console.log("Usage: pnpm exec node scripts/report-decisions.mjs [--results DIR] [--include-model NAME ...] [--reference FILE ...] [--output FILE] [--markdown FILE] [--check-summary FILE]");
      process.exit(0);
    } else throw new Error(`unknown argument ${value}`);
  }
  if (!Number.isFinite(args.probabilityTolerance) || args.probabilityTolerance <= 0) throw new Error("--probability-tolerance must be positive");
  if (!args.references.length) args.references = DEFAULT_REFERENCES.map((name) => resolve(args.results, name));
  return args;
}

function resolvePath(value) {
  return isAbsolute(value) ? value : resolve(process.cwd(), value);
}

async function exists(path) {
  try {
    await readFile(path);
    return true;
  } catch {
    return false;
  }
}

async function sha256(path) {
  const digest = createHash("sha256");
  digest.update(await readFile(path));
  return digest.digest("hex");
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function readJsonl(path) {
  return (await readFile(path, "utf8")).split(/\r?\n/).filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line);
    } catch (error) {
      throw new Error(`${path}:${index + 1}: ${error.message}`);
    }
  });
}

function idOf(row) {
  return row?.caseId ?? row?.case_id ?? row?.id ?? null;
}

function referenceIdOf(row) {
  return row?.id ?? row?.case_id ?? row?.caseId ?? null;
}

function optionIds(options) {
  if (!Array.isArray(options)) return null;
  const ids = options.map((option) => typeof option === "string" ? option : option?.id ?? option?.option_id ?? null);
  return ids.every((id) => typeof id === "string" && id) ? ids : null;
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

function addIssue(issues, message) {
  if (!issues.includes(message)) issues.push(message);
}

function sameValue(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function typedDescription(option) {
  if (!option || typeof option.id !== "string") return null;
  if (typeof option.description !== "string" || !option.description) return option.id;
  const prefix = `${option.id}: `;
  return option.description.startsWith(prefix) ? option.description : `${prefix}${option.description}`;
}

function sourcePathOf(value) {
  if (!value || typeof value !== "object") return null;
  return value.relative_path ?? value.relativePath ?? value.path ?? value.name ?? value.file ?? null;
}

function canonicalFileForPath(pathValue, expectedFiles) {
  if (typeof pathValue !== "string" || !pathValue) return null;
  const normalized = pathValue.replaceAll("\\", "/");
  const matches = expectedFiles.filter((file) => {
    const relativePath = file.path.replaceAll("\\", "/");
    return normalized === relativePath
      || normalized.endsWith(`/${relativePath}`)
      || normalized === basename(relativePath);
  });
  return matches.length === 1 ? matches[0] : null;
}

function datasetProvenanceRoots(data) {
  return [
    data.dataset,
    data.datasetProvenance,
    data.dataset_provenance,
    data.metadata?.dataset,
    data.metadata?.datasetProvenance,
    data.metadata?.dataset_provenance,
    data.metadata?.datasetFiles,
    data.metadata?.dataset_files,
    data.datasetFiles,
    data.dataset_files,
  ].filter((value) => value != null);
}

function datasetFileEntries(value, entries = [], seen = new Set(), depth = 0) {
  if (value == null || depth > 4) return entries;
  if (Array.isArray(value)) {
    for (const item of value) datasetFileEntries(item, entries, seen, depth + 1);
    return entries;
  }
  if (typeof value !== "object") return entries;
  const path = sourcePathOf(value);
  if (path || Object.hasOwn(value, "sha256")) {
    const key = `${path ?? ""}\u0000${value.sha256 ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      entries.push({ path, sha256: value.sha256 ?? null, verification: value.verification ?? null });
    }
  }
  for (const key of ["source_file", "sourceFile", "files", "datasetFiles", "dataset_files", "datasets"]) {
    if (value[key] != null) datasetFileEntries(value[key], entries, seen, depth + 1);
  }
  if (!path && !Object.hasOwn(value, "sha256") && depth < 3) {
    for (const child of Object.values(value)) {
      if (child && typeof child === "object") datasetFileEntries(child, entries, seen, depth + 1);
    }
  }
  return entries;
}

function capturedDatasetMetadata(data) {
  const metadata = data.metadata ?? {};
  return {
    fileHashes: metadata.datasetFileHashes ?? data.datasetFileHashes ?? null,
    manifest: metadata.datasetManifest ?? data.datasetManifest ?? null,
    manifestSha256: metadata.datasetManifestSha256 ?? data.datasetManifestSha256 ?? null,
  };
}

function postRunProvenance(data) {
  const values = [data.datasetProvenance, data.dataset_provenance, data.metadata?.datasetProvenance, data.metadata?.dataset_provenance];
  return values.some((value) => {
    if (!value || typeof value !== "object") return false;
    return ["kind", "type", "mode", "status", "capture", "verification"].some((key) => {
      const text = value[key];
      return typeof text === "string" && /post[-_]run/.test(text.toLowerCase());
    });
  });
}

function manifestFileEntry(manifest, expectedFile) {
  if (!manifest || typeof manifest !== "object") return null;
  const candidates = [manifest.kevala, ...(Array.isArray(manifest.semif?.fixtures) ? manifest.semif.fixtures : [])].filter(Boolean);
  return candidates.find((entry) => canonicalFileForPath(entry.path ?? entry.relative_path, [expectedFile])) ?? null;
}

function explicitDatasetHashes(data, expected, issues) {
  const captured = capturedDatasetMetadata(data);
  const hasSchema = captured.fileHashes != null || captured.manifest != null || captured.manifestSha256 != null;
  if (!hasSchema) return { hasSchema: false, complete: false, hashes: [], manifest: null };

  let complete = true;
  const hashes = [];
  if (!captured.fileHashes || typeof captured.fileHashes !== "object" || Array.isArray(captured.fileHashes)) {
    addIssue(issues, "datasetFileHashes is missing or not an object");
    complete = false;
  } else {
    for (const expectedFile of expected.files) {
      const key = Object.keys(captured.fileHashes).find((candidate) => canonicalFileForPath(candidate, [expectedFile]));
      const actual = key == null ? null : captured.fileHashes[key];
      const digest = typeof actual === "string" ? actual : actual?.sha256;
      if (!actual) {
        addIssue(issues, `datasetFileHashes is missing ${expectedFile.path}`);
        complete = false;
        continue;
      }
      if (typeof digest !== "string" || !/^[a-f0-9]{64}$/i.test(digest)) {
        addIssue(issues, `datasetFileHashes has an invalid SHA-256 for ${expectedFile.path}`);
        complete = false;
        continue;
      }
      const matchesCanonical = digest.toLowerCase() === expectedFile.sha256;
      if (!matchesCanonical) addIssue(issues, `dataset source hash mismatch for ${expectedFile.path}`);
      if (actual?.rows !== undefined && actual.rows !== expectedFile.rows) addIssue(issues, `dataset row count mismatch for ${expectedFile.path}`);
      hashes.push({ path: expectedFile.path, dataset: expectedFile.dataset, sha256: digest.toLowerCase(), matchesCanonical });
    }
  }

  if (!captured.manifest || typeof captured.manifest !== "object" || Array.isArray(captured.manifest)) {
    addIssue(issues, "datasetManifest is missing or not an object");
    complete = false;
  } else {
    for (const expectedFile of expected.files) {
      const entry = manifestFileEntry(captured.manifest, expectedFile);
      if (!entry) {
        addIssue(issues, `datasetManifest is missing ${expectedFile.path}`);
        complete = false;
        continue;
      }
      if (entry.sha256 !== expectedFile.sha256) addIssue(issues, `datasetManifest SHA-256 mismatch for ${expectedFile.path}`);
      if (entry.rows !== expectedFile.rows) addIssue(issues, `datasetManifest row count mismatch for ${expectedFile.path}`);
    }
  }

  if (typeof captured.manifestSha256 !== "string" || !/^[a-f0-9]{64}$/i.test(captured.manifestSha256)) {
    addIssue(issues, "datasetManifestSha256 is missing or invalid");
    complete = false;
  } else if (!expected.manifest || captured.manifestSha256.toLowerCase() !== expected.manifest.sha256) {
    addIssue(issues, "datasetManifestSha256 does not match the canonical manifest");
  }
  return { hasSchema: true, complete, hashes, manifest: captured.manifest };
}

function validateDatasetProvenance(data, expected, issues) {
  const entries = [];
  const seen = new Set();
  for (const value of datasetProvenanceRoots(data)) datasetFileEntries(value, entries, seen);
  const genericHashes = [];
  const unverified = [];
  for (const entry of entries) {
    const file = canonicalFileForPath(entry.path, expected.files);
    if (entry.sha256 == null) {
      if (entry.path) unverified.push(entry.path);
      continue;
    }
    if (typeof entry.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(entry.sha256)) {
      addIssue(issues, `dataset source hash for ${entry.path ?? "unknown file"} is not a SHA-256 digest`);
      continue;
    }
    if (!file) {
      addIssue(issues, `dataset source hash has unknown canonical file ${entry.path ?? "unknown file"}`);
      continue;
    }
    if (entry.sha256.toLowerCase() !== file.sha256) addIssue(issues, `dataset source hash mismatch for ${file.path}`);
    genericHashes.push({ path: file.path, dataset: file.dataset, sha256: entry.sha256.toLowerCase(), matchesCanonical: entry.sha256.toLowerCase() === file.sha256 });
  }
  const explicitIssues = [];
  const explicit = explicitDatasetHashes(data, expected, explicitIssues);
  for (const issue of explicitIssues) addIssue(issues, issue);
  const hashes = explicit.hasSchema ? explicit.hashes : genericHashes;
  const postRun = postRunProvenance(data);
  const complete = explicit.hasSchema && explicit.complete && hashes.length === expected.files.length && hashes.every((entry) => entry.matchesCanonical) && explicitIssues.length === 0;
  return {
    status: issues.length ? "invalid" : complete ? (postRun ? "post-run" : "verified") : postRun ? "post-run" : "unverified",
    hashesPresent: hashes.length > 0,
    capturedAtRuntime: complete && !postRun,
    verificationScope: "canonical fixture SHA-256",
    hashes,
    unverifiedPaths: [...new Set(unverified)],
    verification: datasetProvenanceRoots(data).find((value) => value && typeof value === "object" && value.verification)?.verification ?? null,
    postRunVerification: data.metadata?.postRunVerification ?? data.postRunVerification ?? null,
    note: postRun ? "Fixture hashes were supplied as explicitly labeled post-run provenance." : complete ? "Fixture hashes and the dataset manifest were emitted by the benchmark harness." : "The artifact has no complete captured fixture provenance; scores do not claim runtime fixture provenance.",
  };
}

function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1))];
}

function latency(rows, field) {
  const values = rows.map((row) => Number(row?.[field])).filter((value) => Number.isFinite(value) && value >= 0);
  return {
    count: values.length,
    p50Ms: quantile(values, 0.5),
    p95Ms: quantile(values, 0.95),
    minMs: values.length ? Math.min(...values) : null,
    maxMs: values.length ? Math.max(...values) : null,
  };
}

function modelKey(value) {
  if (typeof value !== "string") return null;
  const name = value.toLowerCase();
  return MODEL_ALIASES[name] ?? (Object.hasOwn(MODEL_FILES, name) ? name : null);
}

async function findResult(resultsDir, model) {
  const direct = resolve(resultsDir, MODEL_FILES[model]);
  if (await exists(direct)) return direct;
  let files = [];
  try {
    files = (await readdir(resultsDir)).filter((name) => /^decision-.*\.json$/.test(name));
  } catch {
    return null;
  }
  for (const name of files) {
    const path = resolve(resultsDir, name);
    try {
      const data = await readJson(path);
      if (modelKey(data.model) === model) return path;
    } catch {
      // The caller reports a missing result; malformed discovered files are not guessed.
    }
  }
  return null;
}

async function loadCanonical(fixturesDir) {
  const all = [];
  const files = [];
  const byDataset = new Map();
  for (const name of DATASET_FILES) {
    const path = resolve(fixturesDir, name);
    const rows = await readJsonl(path);
    const expanded = expandPermutations(rows);
    const dataset = decisionDataset(rows[0]);
    for (const row of expanded) {
      all.push(row);
      if (!byDataset.has(dataset)) byDataset.set(dataset, []);
      byDataset.get(dataset).push(row);
    }
    files.push({ path: relative(ROOT, path), sha256: await sha256(path), rows: rows.length, expandedRows: expanded.length, dataset });
  }
  const manifestPath = resolve(fixturesDir, "manifest.json");
  const manifest = await exists(manifestPath)
    ? { path: relative(ROOT, manifestPath), sha256: await sha256(manifestPath) }
    : null;
  return { all, byId: new Map(all.map((row) => [row.id, row])), byDataset, files, manifest };
}

function probabilities(value, expectedOptions, issues, label, tolerance) {
  const ids = expectedOptions;
  const object = Array.isArray(value)
    ? Object.fromEntries(value.map((number, index) => [ids[index], number]))
    : value && typeof value === "object" ? value : null;
  if (!object || Object.keys(object).length !== ids.length || ids.some((id) => !Object.hasOwn(object, id)) || Object.keys(object).some((id) => !ids.includes(id))) {
    addIssue(issues, `${label}: probability keys do not match option order`);
    return null;
  }
  const values = ids.map((id) => object[id]);
  if (values.some((number) => typeof number !== "number" || !Number.isFinite(number))) {
    addIssue(issues, `${label}: probability must contain only finite JSON numbers`);
    return null;
  }
  if (values.some((number) => number < 0 || number > 1)) {
    addIssue(issues, `${label}: probability is outside [0, 1]`);
    return null;
  }
  const sum = values.reduce((total, number) => total + number, 0);
  if (Math.abs(sum - 1) > tolerance) addIssue(issues, `${label}: probability sum ${sum} exceeds tolerance ${tolerance}`);
  return Object.fromEntries(ids.map((id, index) => [id, values[index]]));
}

function validatePrediction(prediction, normalized, expectedOptions, issues, label, tolerance = ARGMAX_TOLERANCE) {
  if (prediction == null) return;
  if (!expectedOptions.includes(prediction)) {
    addIssue(issues, `${label}: prediction is not one of the option IDs`);
    return;
  }
  if (!normalized) return;
  const chosen = normalized[prediction];
  const maximum = Math.max(...expectedOptions.map((id) => normalized[id]));
  if (maximum - chosen > tolerance) addIssue(issues, `${label}: prediction is not a probability argmax within tolerance ${tolerance}`);
}

function probabilitiesAgree(left, right, expectedOptions, issues, label, tolerance) {
  if (!left || !right) return;
  for (const id of expectedOptions) {
    if (Math.abs(left[id] - right[id]) > tolerance) {
      addIssue(issues, `${label}: inline and raw probability values differ for ${id}`);
      return;
    }
  }
}

function validatePackVerification(data, model, issues) {
  const expected = MODELS[model] ?? {};
  const recorded = data.postRunVerification?.pack
    ?? data.metadata?.postRunVerification?.pack
    ?? null;
  const verification = {
    status: "invalid",
    source: "postRunVerification.pack",
    expectedCatalogHash: expected.packSha256 ?? null,
    expectedCatalogBytes: expected.pack ?? null,
    recordedFullPackSha256: recorded?.sha256 ?? null,
    recordedFullPackBytes: recorded?.bytes ?? null,
    verifiedUtc: data.postRunVerification?.verifiedUtc ?? data.metadata?.postRunVerification?.verifiedUtc ?? null,
    method: recorded?.method ?? null,
    originalOutputSha256: data.postRunVerification?.originalOutputSha256
      ?? data.metadata?.postRunVerification?.originalOutputSha256
      ?? null,
  };
  let valid = true;
  if (!expected.packSha256 || !Number.isSafeInteger(expected.pack)) {
    addIssue(issues, `MODELS[${model}] is missing expected full-pack catalog metadata`);
    valid = false;
  }
  if (!recorded || typeof recorded !== "object") {
    addIssue(issues, `${model}: missing postRunVerification.pack full-pack evidence`);
    return verification;
  }
  if (typeof recorded.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(recorded.sha256)) {
    addIssue(issues, `${model}: postRunVerification.pack full-pack sha256 is missing or invalid`);
    valid = false;
  } else if (expected.packSha256 && recorded.sha256.toLowerCase() !== expected.packSha256.toLowerCase()) {
    addIssue(issues, `${model}: postRunVerification.pack full-pack sha256 mismatch with MODELS catalog hash`);
    valid = false;
  }
  if (!Number.isSafeInteger(recorded.bytes) || recorded.bytes <= 0) {
    addIssue(issues, `${model}: postRunVerification.pack full-pack byte size is missing or invalid`);
    valid = false;
  } else if (Number.isSafeInteger(expected.pack) && recorded.bytes !== expected.pack) {
    addIssue(issues, `${model}: postRunVerification.pack full-pack byte size mismatch with MODELS catalog size`);
    valid = false;
  }
  verification.status = valid ? "verified" : "invalid";
  return verification;
}

function validateRawResult(data, expected, tolerance, model) {
  const issues = [];
  const runtimeValidation = validateModelMetadata(data, model, issues);
  const packVerification = validatePackVerification(data, model, issues);
  const provenanceIssues = [];
  const datasetProvenance = validateDatasetProvenance(data, expected, provenanceIssues);
  for (const issue of provenanceIssues) addIssue(issues, issue);
  if (data.status !== "done" || data.done !== true) addIssue(issues, "result is not complete");
  const timingRows = Array.isArray(data.timingAllRows) ? data.timingAllRows : [];
  const probabilityRows = Array.isArray(data.rawProbs) ? data.rawProbs : [];
  if (timingRows.length !== expected.all.length) addIssue(issues, `timing rows ${timingRows.length} != canonical ${expected.all.length}`);
  if (probabilityRows.length !== expected.all.length) addIssue(issues, `probability rows ${probabilityRows.length} != canonical ${expected.all.length}`);

  const timingById = new Map();
  for (const raw of timingRows) {
    const id = idOf(raw);
    if (!id) addIssue(issues, "timing row has no case ID");
    else if (timingById.has(id)) addIssue(issues, `duplicate timing case ID ${id}`);
    else timingById.set(id, raw);
  }
  const probabilityById = new Map();
  for (const raw of probabilityRows) {
    const id = idOf(raw);
    if (!id) addIssue(issues, "probability row has no case ID");
    else if (probabilityById.has(id)) addIssue(issues, `duplicate probability case ID ${id}`);
    else probabilityById.set(id, raw);
  }
  const predictions = [];
  const probabilitiesById = new Map();
  for (const row of expected.all) {
    const raw = timingById.get(row.id);
    const probability = probabilityById.get(row.id);
    if (!raw) {
      addIssue(issues, `missing timing case ID ${row.id}`);
      predictions.push({ id: row.id, status: "missing" });
      continue;
    }
    if (!sameArray(optionIds(raw.options), row.options.map((option) => option.id))) addIssue(issues, `option order mismatch for ${row.id}`);
    if ((raw.expected ?? raw.gold_id) !== goldOptionId(row)) addIssue(issues, `gold mismatch for ${row.id}`);
    if ((raw.permutation ?? null) !== (row.permutation ?? null)) addIssue(issues, `permutation mismatch for ${row.id}`);
    if (raw.baseId !== undefined && raw.baseId !== row.base_id) addIssue(issues, `base ID mismatch for ${row.id}`);
    const optionIdsForRow = row.options.map((option) => option.id);
    const normalized = probabilities(probability?.probabilities, optionIdsForRow, issues, row.id, tolerance);
    if (normalized) probabilitiesById.set(row.id, normalized);
    if ((probability?.permutation ?? null) !== (row.permutation ?? null)) addIssue(issues, `probability permutation mismatch for ${row.id}`);
    const inline = raw.rawProbabilities === undefined
      ? null
      : probabilities(raw.rawProbabilities, optionIdsForRow, issues, `${row.id} inline`, tolerance);
    probabilitiesAgree(normalized, inline, optionIdsForRow, issues, row.id, tolerance);
    const prediction = raw.prediction ?? raw.option_id ?? raw.predicted_id ?? null;
    validatePrediction(prediction, normalized ?? inline, optionIdsForRow, issues, row.id);
    predictions.push({ id: row.id, option_id: prediction, status: raw.error ? "error" : raw.invalid || prediction == null ? "invalid" : "ok" });
  }
  for (const id of timingById.keys()) if (!expected.byId.has(id)) addIssue(issues, `unexpected timing case ID ${id}`);
  for (const id of probabilityById.keys()) if (!expected.byId.has(id)) addIssue(issues, `unexpected probability case ID ${id}`);
  return { issues, timingRows, probabilityRows, predictions, probabilitiesById, timingById, probabilityById, runtimeValidation, packVerification, datasetProvenance };
}

function runtimeReport(data) {
  return {
    backend: data.backend ?? null,
    threads: data.threads ?? null,
    gpu: data.gpuInfo ?? null,
    runner: data.runner ?? null,
    cachePolicy: data.metadata?.stateCache ?? data.cachePolicy ?? data.stateCache ?? null,
    environment: data.environment ?? null,
  };
}

function packReport(data, verification) {
  return {
    model: data.model ?? null,
    modelRevision: data.modelRevision ?? null,
    catalogHash: data.catalogHash ?? data.hash ?? null,
    catalogHashScope: data.catalogHashScope ?? data.hashScope ?? null,
    verification,
    modelInfo: data.metadata?.modelInfo ?? null,
    datasetModule: data.metadata?.datasetModule ?? null,
    datasetFiles: data.metadata?.datasetFiles ?? null,
  };
}

function validateModelMetadata(data, model, issues) {
  const cachePolicy = data.cachePolicy ?? data.metadata?.stateCache ?? data.stateCache ?? null;
  if (!cachePolicy || cachePolicy.enabled !== false) addIssue(issues, "state cache policy must explicitly report enabled:false");

  const clock = data.metadata?.clock ?? data.clock ?? null;
  if (!clock || clock.quantized !== false) addIssue(issues, "benchmark clock must explicitly report quantized:false");

  if (String(data.backend ?? "").toLowerCase() !== "webgpu") addIssue(issues, `backend ${data.backend ?? "<missing>"} is not WebGPU`);
  const gpu = data.gpuInfo;
  if (!gpu || gpu.available !== true || gpu.isFallbackAdapter !== false) addIssue(issues, "GPU evidence must report an available non-fallback hardware adapter");

  const modelInfo = data.metadata?.modelInfo;
  if (!modelInfo || typeof modelInfo !== "object") {
    addIssue(issues, "model metadata is missing");
  } else {
    if (modelInfo.name !== model) addIssue(issues, `model metadata name ${modelInfo.name ?? "<missing>"} does not match ${model}`);
    if (typeof modelInfo.revision !== "string" || !modelInfo.revision) addIssue(issues, "model metadata revision is missing");
    const expectedRevision = MODELS[model]?.revision ?? null;
    if (expectedRevision && modelInfo.revision !== expectedRevision) addIssue(issues, `model metadata revision ${modelInfo.revision} does not match MODELS[${model}]`);
    const expectedSource = MODELS[model]?.repo ? `https://huggingface.co/${MODELS[model].repo}` : null;
    if (modelInfo.source !== undefined && expectedSource && modelInfo.source !== expectedSource) addIssue(issues, `model metadata source ${modelInfo.source} does not match MODELS[${model}]`);
  }
  const modelArch = data.modelArch ?? data.metadata?.modelArch;
  const expectedArch = MODELS[model]?.arch ?? null;
  if (modelArch !== undefined && expectedArch && modelArch !== expectedArch) addIssue(issues, `model architecture ${modelArch} does not match MODELS[${model}]`);
  return {
    backend: data.backend ?? null,
    hardwareAdapter: gpu ?? null,
    clock: clock ?? null,
    cachePolicy,
    modelInfo: modelInfo ?? null,
    modelArch: modelArch ?? null,
    expectedArch,
    expectedSource: MODELS[model]?.repo ? `https://huggingface.co/${MODELS[model].repo}` : null,
    expectedRevision: MODELS[model]?.revision ?? null,
  };
}

async function processModel(model, path, expected, tolerance) {
  if (!path) return { report: { status: "missing", file: null }, internal: null };
  let data;
  try {
    data = await readJson(path);
  } catch (error) {
    return { report: { status: "invalid", file: { path: relative(ROOT, path) }, errors: [error.message] }, internal: null };
  }
  const checked = validateRawResult(data, expected, tolerance, model);
  if (data.model && modelKey(data.model) !== model) addIssue(checked.issues, `result model ${data.model} does not match ${model}`);
  const validation = {
    ok: checked.issues.length === 0,
    issues: checked.issues,
    expectedRows: expected.all.length,
    timingRows: checked.timingRows.length,
    probabilityRows: checked.probabilityRows.length,
    probabilityTolerance: tolerance,
  };
  const report = {
    status: validation.ok ? "ok" : "invalid",
    file: { path: relative(ROOT, path), sha256: await sha256(path) },
    validation,
    runtime: { ...runtimeReport(data), validation: checked.runtimeValidation },
    pack: packReport(data, checked.packVerification),
    runtimeErrors: Array.isArray(data.errors) ? data.errors : [],
    datasetProvenance: checked.datasetProvenance,
    latency: latency(checked.timingRows, "wallMs"),
    reportedLatency: data.latency ?? null,
    cachePolicy: checked.runtimeValidation.cachePolicy,
    model: model,
  };
  if (validation.ok) report.metrics = evaluateDecisions(expected.all, checked.predictions);
  else report.metrics = null;
  return {
    report,
    internal: validation.ok ? { probabilitiesById: checked.probabilitiesById, predictions: checked.predictions, timingById: checked.timingById } : null,
  };
}

function referenceDataset(reference) {
  const source = reference.dataset?.source_file?.relative_path
    ?? reference.dataset?.source_file?.path
    ?? reference.dataset?.source_file?.name
    ?? reference.dataset?.source
    ?? "";
  if (source.includes("kevala-authored36")) return "kevala-authored36";
  if (source.includes("semif-perturbations108")) return "semif-perturbations108";
  if (source.includes("semif-authored144")) return "semif-authored144";
  return null;
}

function validateReferenceDataset(reference, expected, dataset) {
  const issues = [];
  const sourceFile = reference.dataset?.source_file;
  const expectedFile = dataset ? expected.files.find((file) => file.dataset === dataset) : null;
  if (!dataset || !expectedFile) addIssue(issues, "reference dataset source is not a known canonical fixture");
  if (!sourceFile || typeof sourceFile !== "object") {
    addIssue(issues, "reference dataset must include dataset.source_file metadata");
  } else {
    const matchedFile = canonicalFileForPath(sourcePathOf(sourceFile), expected.files);
    if (!matchedFile || matchedFile.dataset !== dataset) addIssue(issues, "reference dataset.source_file path does not identify its canonical fixture");
    if (typeof sourceFile.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(sourceFile.sha256)) {
      addIssue(issues, "reference dataset.source_file.sha256 is missing or invalid");
    } else if (expectedFile && sourceFile.sha256.toLowerCase() !== expectedFile.sha256) {
      addIssue(issues, `reference dataset source hash mismatch for ${expectedFile.path}`);
    }
    if (sourceFile.rows !== undefined && sourceFile.rows !== expectedFile?.rows) addIssue(issues, "reference dataset source row count does not match the canonical fixture");
  }
  return {
    status: issues.length ? "invalid" : "verified",
    ok: issues.length === 0,
    issues,
    dataset,
    expectedFile: expectedFile ?? null,
    sourceFile: sourceFile ?? null,
  };
}

function optionRenderingFor(reference, variant) {
  return variant.option_rendering
    ?? reference.option_rendering
    ?? reference.protocols?.[variant.protocol]?.option_rendering
    ?? null;
}

function validateOptionRendering(reference, variant, name, issues) {
  const rendering = optionRenderingFor(reference, variant);
  const isIt = String(variant.weight_variant ?? "").toLowerCase() === "it"
    || String(variant.protocol ?? "").toLowerCase().includes("gemma-it");
  if (!isIt) return rendering;
  if (!rendering || typeof rendering !== "object") {
    addIssue(issues, `${name}: missing typed option ID prefix option_rendering contract`);
    return rendering;
  }
  if (rendering.name !== TYPED_OPTION_RENDERING.name
      || rendering.choice_description !== TYPED_OPTION_RENDERING.choiceDescription
      || rendering.empty_description !== TYPED_OPTION_RENDERING.emptyDescription) {
    addIssue(issues, `${name}: option_rendering is not kev-typed-choice-v1`);
  }
  return rendering;
}

function validateReferenceVariant(name, variant, expectedRows, tolerance, reference, sourceIssues = []) {
  const issues = [...sourceIssues];
  const optionRendering = validateOptionRendering(reference, variant, name, issues);
  const rows = Array.isArray(variant.rows) ? variant.rows : [];
  if (rows.length !== expectedRows.length) addIssue(issues, `rows ${rows.length} != canonical ${expectedRows.length}`);
  const byId = new Map();
  const probabilitiesById = new Map();
  const predictions = [];
  for (const raw of rows) {
    const id = referenceIdOf(raw);
    if (!id) {
      addIssue(issues, `${name}: row has no case ID`);
      continue;
    }
    if (byId.has(id)) addIssue(issues, `${name}: duplicate case ID ${id}`);
    byId.set(id, raw);
  }
  for (const expected of expectedRows) {
    const raw = byId.get(expected.id);
    if (!raw) {
      addIssue(issues, `${name}: missing case ID ${expected.id}`);
      predictions.push({ id: expected.id, status: "missing" });
      continue;
    }
    const rawOptions = optionIds(raw.options);
    const expectedOptionIds = expected.options.map((option) => option.id);
    const expectedOptionDescriptions = expected.options.map(typedDescription);
    if (!sameArray(rawOptions, expectedOptionIds)) addIssue(issues, `${name}: option order mismatch for ${expected.id}`);
    const rawDescriptions = Array.isArray(raw.options) ? raw.options.map((option) => option?.description ?? null) : null;
    if (!sameArray(rawDescriptions, expectedOptionDescriptions)) addIssue(issues, `${name}: typed option descriptions mismatch for ${expected.id}`);
    if (!sameValue(raw.state, expected.state)) addIssue(issues, `${name}: state mismatch for ${expected.id}`);
    if (raw.question !== expected.question) addIssue(issues, `${name}: question mismatch for ${expected.id}`);
    if ((raw.gold_option_id ?? raw.gold_id) !== goldOptionId(expected)) addIssue(issues, `${name}: gold mismatch for ${expected.id}`);
    if ((raw.permutation ?? null) !== (expected.permutation ?? null)) addIssue(issues, `${name}: permutation mismatch for ${expected.id}`);
    const expectedIndex = expected.options.findIndex((option) => option.id === goldOptionId(expected));
    if (raw.gold_option_index !== undefined && raw.gold_option_index !== expectedIndex) addIssue(issues, `${name}: gold index mismatch for ${expected.id}`);
    const normalized = probabilities(raw.probabilities, expectedOptionIds, issues, `${name}:${expected.id}`, tolerance);
    if (normalized) probabilitiesById.set(expected.id, normalized);
    const chosen = raw.chosen_option_id ?? null;
    if (chosen == null) addIssue(issues, `${name}: missing chosen option for ${expected.id}`);
    validatePrediction(chosen, normalized, expectedOptionIds, issues, `${name}:${expected.id}`);
    predictions.push({ id: expected.id, option_id: chosen, status: chosen == null ? "invalid" : "ok" });
  }
  for (const id of byId.keys()) if (!expectedRows.some((row) => row.id === id)) addIssue(issues, `${name}: unexpected case ID ${id}`);
  const forwardRows = rows.map((row) => ({ value: row.timing_ms?.forward_median }));
  const report = {
    status: issues.length ? "invalid" : "ok",
    variant: name,
    family: variant.family ?? null,
    weightVariant: variant.weight_variant ?? null,
    source: variant.source ?? null,
    revision: variant.revision ?? null,
    checkpoint: variant.checkpoint ?? null,
    protocol: variant.protocol ?? null,
    dtype: variant.dtype ?? null,
    device: variant.device ?? null,
    quantization: variant.quantization ?? null,
    optionRendering,
    validation: { ok: issues.length === 0, issues, expectedRows: expectedRows.length, observedRows: rows.length, probabilityTolerance: tolerance },
    metrics: issues.length ? null : evaluateDecisions(expectedRows, predictions),
    latency: latency(forwardRows, "value"),
    sourceSummary: variant.summary ?? null,
  };
  return { report, internal: issues.length ? null : { probabilitiesById, predictions } };
}

async function processReference(path, expected, tolerance) {
  let data;
  try {
    data = await readJson(path);
  } catch (error) {
    return { report: { status: "invalid", file: { path: relative(ROOT, path) }, errors: [error.message] }, variants: new Map() };
  }
  const dataset = referenceDataset(data);
  const datasetValidation = validateReferenceDataset(data, expected, dataset);
  const expectedRows = dataset && expected.byDataset.has(dataset) ? expected.byDataset.get(dataset) : [];
  const variants = new Map();
  const reports = {};
  for (const [name, variant] of Object.entries(data.variants ?? {})) {
    const checked = validateReferenceVariant(name, variant, expectedRows, tolerance, data, datasetValidation.issues);
    variants.set(name, checked);
    reports[name] = checked.report;
  }
  return {
    report: {
      status: datasetValidation.ok && Object.keys(reports).length > 0 && Object.values(reports).every((item) => item.status === "ok") ? "ok" : "invalid",
      file: { path: relative(ROOT, path), sha256: await sha256(path) },
      dataset,
      datasetMetadata: data.dataset ?? null,
      datasetDigest: data.dataset_digest ?? null,
      datasetValidation,
      schemaVersion: data.schema_version ?? null,
      environment: data.environment ?? null,
      referenceHelper: data.reference_helper ?? null,
      sources: data.sources ?? null,
      variants: reports,
      comparison: data.comparison ?? null,
    },
    variants,
  };
}

function firstDifference(left, right, path = "$") {
  if (Object.is(left, right)) return null;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right)) return `${path}: array/object mismatch`;
    if (left.length !== right.length) return `${path}.length: ${left.length} != ${right.length}`;
    for (let index = 0; index < left.length; index++) {
      const difference = firstDifference(left[index], right[index], `${path}[${index}]`);
      if (difference) return difference;
    }
    return null;
  }
  if (left && typeof left === "object" || right && typeof right === "object") {
    if (!left || !right || typeof left !== "object" || typeof right !== "object") return `${path}: ${JSON.stringify(left)} != ${JSON.stringify(right)}`;
    const keys = [...new Set([...Object.keys(left), ...Object.keys(right)])].sort();
    for (const key of keys) {
      if (!Object.hasOwn(left, key)) return `${path}.${key}: missing from regenerated report`;
      if (!Object.hasOwn(right, key)) return `${path}.${key}: missing from committed summary`;
      const difference = firstDifference(left[key], right[key], `${path}.${key}`);
      if (difference) return difference;
    }
    return null;
  }
  return `${path}: ${JSON.stringify(left)} != ${JSON.stringify(right)}`;
}

function summaryComparable(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const comparable = { ...value };
  delete comparable.generatedUtc;
  return comparable;
}

async function checkSummary(path, report) {
  let committed;
  try {
    committed = await readJson(path);
  } catch (error) {
    console.error(`Decision summary check failed: could not read ${path}: ${error.message}`);
    return false;
  }
  const difference = firstDifference(summaryComparable(report), summaryComparable(committed));
  if (difference) {
    console.error(`Decision summary mismatch: ${path}`);
    console.error(`First difference: ${difference}`);
    return false;
  }
  return true;
}

function compareQ8WithReference(modelData, referenceData, expectedRows, model, referenceName) {
  const unavailable = (reason) => ({ status: "unavailable", model, reference: referenceName, reason, quantizationOnly: false });
  if (!modelData?.report || modelData.report.status !== "ok") return unavailable("Q8 result is missing or invalid");
  const reference = referenceData?.get(referenceName);
  if (!reference || reference.report.status !== "ok") return unavailable("BF16 reference is missing or invalid");
  const ids = expectedRows.map((row) => row.id);
  let matched = 0;
  let q8Correct = 0;
  let bf16Correct = 0;
  let absoluteSum = 0;
  let absoluteMax = 0;
  let probabilityCount = 0;
  for (const row of expectedRows) {
    const q8 = modelData.internal.probabilitiesById.get(row.id);
    const bf16 = reference.internal.probabilitiesById.get(row.id);
    if (!q8 || !bf16) return unavailable(`missing probabilities for ${row.id}`);
    const options = row.options.map((option) => option.id);
    for (const option of options) {
      const difference = Math.abs(q8[option] - bf16[option]);
      absoluteSum += difference;
      absoluteMax = Math.max(absoluteMax, difference);
      probabilityCount++;
    }
    const qPrediction = modelData.internal.predictions.find((prediction) => prediction.id === row.id)?.option_id ?? null;
    const bPrediction = reference.internal.predictions.find((prediction) => prediction.id === row.id)?.option_id ?? null;
    if (qPrediction === bPrediction) matched++;
    if (qPrediction === goldOptionId(row)) q8Correct++;
    if (bPrediction === goldOptionId(row)) bf16Correct++;
  }
  return {
    status: "ok",
    model,
    reference: referenceName,
    cases: ids.length,
    options: { identicalCaseCount: ids.length, sameOptionOrder: true, comparedGoldAndOptionIds: true },
    decisionAgreement: { matched, total: ids.length, rate: matched / ids.length },
    gold: {
      q8Correct,
      bf16Correct,
      total: ids.length,
      q8Accuracy: q8Correct / ids.length,
      bf16Accuracy: bf16Correct / ids.length,
    },
    probabilityDifference: {
      maxAbsProbabilityDiff: absoluteMax,
      meanAbsProbabilityDiff: absoluteSum / probabilityCount,
      comparedValues: probabilityCount,
    },
    quantizationOnly: false,
    interpretation: "This compares browser Q8 output with a BF16 reference; weight quantization and runtime/device arithmetic both contribute to the difference.",
  };
}

function markdown(report) {
  const out = ["# Decision benchmark report", "", `Generated: ${report.generatedUtc}`, "", "## Model results", "", "| Model | Status | Rows | Accuracy | Coverage | Invalid | p50 ms | p95 ms |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"];
  for (const [model, item] of Object.entries(report.models)) {
    const metrics = item.metrics;
    out.push(`| ${model} | ${item.status} | ${metrics?.total ?? "—"} | ${metrics?.accuracy?.toFixed?.(4) ?? "—"} | ${metrics?.coverage?.toFixed?.(4) ?? "—"} | ${metrics ? metrics.invalid + metrics.missing : "—"} | ${item.latency?.p50Ms?.toFixed?.(2) ?? "—"} | ${item.latency?.p95Ms?.toFixed?.(2) ?? "—"} |`);
  }
  out.push("", "## Per-dataset results", "", "| Model | Dataset | Rows | Accuracy | Coverage | Invalid | CI 95% | Flips |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |");
  for (const [model, item] of Object.entries(report.models)) for (const [dataset, metrics] of Object.entries(item.metrics?.perDataset ?? {})) {
    const ci = metrics.groupBootstrap.confidence95.map((value) => value == null ? "—" : value.toFixed(4)).join("–");
    out.push(`| ${model} | ${dataset} | ${metrics.total} | ${metrics.accuracy.toFixed(4)} | ${metrics.coverage.toFixed(4)} | ${metrics.invalid + metrics.missing} | ${ci} | ${metrics.semanticPermutationFlip.rate == null ? "—" : metrics.semanticPermutationFlip.rate.toFixed(4)} |`);
  }
  out.push("", "## Per-family results", "", "| Model | Dataset | Family | Rows | Accuracy | Coverage | Invalid |", "| --- | --- | --- | ---: | ---: | ---: | ---: |");
  for (const [model, item] of Object.entries(report.models)) for (const [dataset, metrics] of Object.entries(item.metrics?.perDataset ?? {})) for (const [family, familyMetrics] of Object.entries(metrics.perFamily)) {
    out.push(`| ${model} | ${dataset} | ${family} | ${familyMetrics.total} | ${familyMetrics.accuracy.toFixed(4)} | ${familyMetrics.coverage.toFixed(4)} | ${familyMetrics.invalid + familyMetrics.missing} |`);
  }
  out.push("", "## Missing inputs", "", ...(report.missing.length ? report.missing.map((item) => `- ${item}`) : ["- None"]), "", "## Gemma Q8 versus BF16 references", "", "| Model | Reference | Cases | Agreement | Max abs diff | Mean abs diff | Status |", "| --- | --- | ---: | ---: | ---: | ---: | --- |");
  for (const item of report.comparisons) out.push(`| ${item.model} | ${item.reference} | ${item.cases ?? "—"} | ${item.decisionAgreement?.rate?.toFixed?.(4) ?? "—"} | ${item.probabilityDifference?.maxAbsProbabilityDiff?.toFixed?.(6) ?? "—"} | ${item.probabilityDifference?.meanAbsProbabilityDiff?.toFixed?.(6) ?? "—"} | ${item.status} |`);
  return `${out.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const canonical = await loadCanonical(args.fixtures);
  const report = {
    schema: "kevala-decision-report-v1",
    generatedUtc: new Date().toISOString(),
    probabilitySumTolerance: args.probabilityTolerance,
    canonical: { permutations: [...DEFAULT_PERMUTATIONS], expandedRows: canonical.all.length, files: canonical.files, manifest: canonical.manifest },
    models: {},
    references: {},
    comparisons: [],
    missing: [],
  };
  const modelData = new Map();
  for (const model of [...Object.keys(BASE_MODEL_FILES), ...args.optionalModels]) {
    const path = await findResult(args.results, model);
    if (!path) report.missing.push(`model result: ${MODEL_FILES[model]}`);
    const checked = await processModel(model, path, canonical, args.probabilityTolerance);
    report.models[model] = checked.report;
    modelData.set(model, checked);
  }
  const referenceData = new Map();
  for (const path of args.references) {
    if (!(await exists(path))) {
      report.missing.push(`reference: ${relative(ROOT, path)}`);
      continue;
    }
    const checked = await processReference(path, canonical, args.probabilityTolerance);
    report.references[relative(ROOT, path)] = checked.report;
    for (const [name, variant] of checked.variants) referenceData.set(name, variant);
  }
  for (const variant of ["e2b-base", "e2b-it", "e4b-base", "e4b-it"]) {
    if (!referenceData.has(variant)) report.missing.push(`reference variant: ${variant}`);
  }
  const kevalaRows = canonical.byDataset.get("kevala-authored36") ?? [];
  report.comparisons.push(compareQ8WithReference(modelData.get("gemma-4-e2b"), referenceData, kevalaRows, "gemma-4-e2b", "e2b-it"));
  report.comparisons.push(compareQ8WithReference(modelData.get("gemma-4-e4b"), referenceData, kevalaRows, "gemma-4-e4b", "e4b-it"));
  const text = JSON.stringify(report, null, 2) + "\n";
  if (args.output) {
    await mkdir(dirname(args.output), { recursive: true });
    await writeFile(args.output, text);
  } else {
    process.stdout.write(text);
  }
  if (args.markdown) {
    await mkdir(dirname(args.markdown), { recursive: true });
    await writeFile(args.markdown, markdown(report));
  }
  const summaryMatches = args.checkSummary ? await checkSummary(args.checkSummary, report) : true;
  if (!summaryMatches
      || report.missing.length
      || Object.values(report.models).some((item) => item.status !== "ok")
      || Object.values(report.references).some((item) => item.status !== "ok")
      || report.comparisons.some((item) => item.status !== "ok")) {
    console.error("Decision report contains missing or invalid inputs; inspect the saved validation results.");
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exitCode = 1;
});
