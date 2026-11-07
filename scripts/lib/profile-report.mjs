import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, realpath, rename, stat, writeFile } from "node:fs/promises";
import { basename, dirname, extname, isAbsolute, relative, resolve, sep } from "node:path";
import { gzipSync } from "node:zlib";
import { MODELS } from "../../js/src/source.js";
import { KERNEL_HARNESSES, kernelCoverage, kernelHarnessSkipReason, parseKernelRegistry, validateKernelResult } from "./kernel-profile-catalog.mjs";
import { classifyProfileFailure, summarizeProfileRuns, validateProfileBrowser, validateProfileRun } from "./model-profile.mjs";

const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const check = (condition, message) => { if (!condition) throw new Error(message); };
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const sha = (value) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const finite = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const unique = (values) => [...new Set(values)].sort();
const pick = (value, keys) => Object.fromEntries(keys.filter((key) => value?.[key] !== undefined).map((key) => [key, value[key]]));
const modelKey = (row) => `${row.id}/${row.precision}/${row.browser}`;
const ATTRIBUTION_FAILURE = "Kernel labels changed within a request shape";
const isAttributionCorrection = (correction) => correction.status === "measured" && correction.reasonCode === "heterogeneous-kernel-attribution";
const CAPABILITY_FAILURE = "wide baseline requires shader-f16: wide baseline requires shader-f16";
const CAPABILITY_CORRECTION = { id: "paired-projections-wide", previousStatus: "failed", status: "skipped", reasonCode: "missing-capability", reason: "Missing capabilities: shaderF16" };

export function stableJson(value, space = 0) {
  return JSON.stringify(value, (_, item) => object(item) ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]])) : item, space);
}

export function normalizeRuntimeFiles(files, label = "runtime") {
  check(Array.isArray(files) && files.length > 0, `${label}: missing runtime asset hashes`);
  const normalized = files.map((file) => {
    check(object(file) && typeof file.path === "string" && sha(file.sha256), `${label}: invalid runtime asset`);
    const path = file.path.replaceAll("\\", "/").replace(/^js\/src\//, "");
    check(path && !isAbsolute(path) && !path.split("/").some((part) => !part || part === "." || part === "..") && /\.(js|wasm)$/.test(path), `${label}: invalid runtime asset path`);
    return { path, sha256: file.sha256 };
  }).sort((a, b) => a.path.localeCompare(b.path));
  check(new Set(normalized.map((file) => file.path)).size === normalized.length, `${label}: duplicate runtime asset path`);
  return normalized;
}

function sharedProtocol(summaries, kind) {
  let reference;
  for (const summary of summaries) {
    check(object(summary.protocol), `${kind}: missing profiling protocol`);
    const protocol = { ...summary.protocol };
    // Selection and process budgets do not change the measurement protocol.
    for (const key of kind === "models" ? ["models", "precisions", "browsers", "kernelShare"] : ["timeoutSeconds", "budgetSeconds"]) delete protocol[key];
    if (reference) check(stableJson(reference) === stableJson(protocol), `${kind}: mixed profiling protocols`);
    else reference = protocol;
  }
  return reference || null;
}

function validateMetric(metric, label) {
  check(object(metric) && finite(metric.medianMs) && finite(metric.p95Ms), `${label}: invalid median or p95`);
  check(Array.isArray(metric.samples) && metric.samples.length > 0 && metric.samples.every(finite), `${label}: invalid or missing raw timing samples`);
  check(Array.isArray(metric.blockMediansMs) && metric.blockMediansMs.length > 0 && metric.blockMediansMs.every(finite), `${label}: invalid or missing block medians`);
  return pick(metric, ["medianMs", "p95Ms", "meanMs", "samples", "blockMediansMs"]);
}

const REQUEST_STAGE_MS_FIELDS = ["reportedPrepareMs", "reportedForwardMs", "reportedTotalMs", "postForwardResidualMs", "outerResidualMs"];
const REQUEST_STAGE_FRACTION_FIELDS = ["prepareOfRuntimeTotal", "forwardOfRuntimeTotal", "postForwardResidualOfRuntimeTotal", "postForwardResidualOfWall", "runtimeTotalOfWall", "outerResidualOfWall"];

function requestStageMedian(values) {
  const sorted = [...values].sort((a, b) => a - b), middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function validateRequestStageDistribution(metric, valuesByBlock, field, fractional, label) {
  const values = valuesByBlock.flat();
  const median = requestStageMedian(values), mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const sorted = [...values].sort((a, b) => a - b), p95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
  const sampleKey = "samples", blockKey = fractional ? "blockMedians" : "blockMediansMs";
  const medianKey = fractional ? "median" : "medianMs", meanKey = fractional ? "mean" : "meanMs", p95Key = fractional ? "p95" : "p95Ms";
  check(object(metric) && Array.isArray(metric[sampleKey]) && metric[sampleKey].length === values.length
    && metric[sampleKey].every((value, index) => Number.isFinite(value) && value === values[index]), label + ": invalid " + field + " samples");
  check(Array.isArray(metric[blockKey]) && metric[blockKey].length === valuesByBlock.length
    && metric[blockKey].every((value, index) => Number.isFinite(value) && value === requestStageMedian(valuesByBlock[index])), label + ": invalid " + field + " block medians");
  check(metric[medianKey] === median && metric[meanKey] === mean && metric[p95Key] === p95, label + ": invalid " + field + " distribution");
  return pick(metric, [medianKey, meanKey, p95Key, sampleKey, blockKey]);
}

function validateRequestStageLane(lane, label, identities) {
  check(object(lane) && Array.isArray(lane.blocks) && lane.blocks.length >= 2, label + ": invalid request stage blocks");
  const blockIds = new Set(), sampleCount = lane.blocks[0]?.samples?.length;
  check(Number.isSafeInteger(sampleCount) && sampleCount > 0, label + ": missing request stage samples");
  const blocks = lane.blocks.map((block) => {
    check(object(block) && Number.isSafeInteger(block.block) && block.block >= 0 && !blockIds.has(block.block), label + ": invalid or duplicate block identity");
    blockIds.add(block.block);
    check(Array.isArray(block.samples) && block.samples.length === sampleCount, label + ": inconsistent request stage sample count");
    const samples = block.samples.map((sample, sampleIndex) => {
      check(object(sample) && sample.sampleIndex === sampleIndex && /^[a-f0-9]{64}$/.test(sample.requestHash)
        && Number.isSafeInteger(sample.tokens) && sample.tokens > 0, label + ": invalid request stage sample identity");
      for (const field of REQUEST_STAGE_MS_FIELDS) check(Number.isFinite(sample[field]), label + ": invalid request stage value " + field);
      for (const field of REQUEST_STAGE_FRACTION_FIELDS) check(Number.isFinite(sample[field]), label + ": invalid request stage fraction " + field);
      check(sample.reportedPrepareMs >= 0 && sample.reportedForwardMs >= 0 && sample.reportedTotalMs > 0
        && Number.isFinite(sample.wallMs) && sample.wallMs > 0,
        label + ": invalid reported request stage timing");
      const postForwardResidual = sample.reportedTotalMs - sample.reportedPrepareMs - sample.reportedForwardMs;
      const outerResidual = sample.wallMs - sample.reportedTotalMs;
      check(sample.postForwardResidualMs === postForwardResidual && sample.outerResidualMs === outerResidual
        && sample.prepareOfRuntimeTotal === sample.reportedPrepareMs / sample.reportedTotalMs
        && sample.forwardOfRuntimeTotal === sample.reportedForwardMs / sample.reportedTotalMs
        && sample.postForwardResidualOfRuntimeTotal === postForwardResidual / sample.reportedTotalMs
        && sample.postForwardResidualOfWall === postForwardResidual / sample.wallMs
        && sample.runtimeTotalOfWall === sample.reportedTotalMs / sample.wallMs
        && sample.outerResidualOfWall === outerResidual / sample.wallMs, label + ": inconsistent samplewise request stage derivation");
      const known = identities.get(sample.sampleIndex);
      if (known) check(known.requestHash === sample.requestHash && known.tokens === sample.tokens, label + ": request stage identity differs between blocks or lanes");
      else identities.set(sample.sampleIndex, { requestHash: sample.requestHash, tokens: sample.tokens });
      return { sampleIndex, requestHash: sample.requestHash, tokens: sample.tokens,
        ...pick(sample, ["reportedPrepareMs", "reportedForwardMs", "reportedTotalMs", "wallMs", "postForwardResidualMs", "outerResidualMs", ...REQUEST_STAGE_FRACTION_FIELDS]) };
    });
    return { block: block.block, samples };
  });
  const distributions = {};
  for (const field of REQUEST_STAGE_MS_FIELDS) {
    distributions[field] = validateRequestStageDistribution(lane.distributions?.[field], blocks.map((block) => block.samples.map((sample) => sample[field])), field, false, label);
  }
  for (const field of REQUEST_STAGE_FRACTION_FIELDS) {
    distributions[field] = validateRequestStageDistribution(lane.distributions?.[field], blocks.map((block) => block.samples.map((sample) => sample[field])), field, true, label);
  }
  check(object(lane.distributions) && Object.keys(lane.distributions).length === REQUEST_STAGE_MS_FIELDS.length + REQUEST_STAGE_FRACTION_FIELDS.length,
    label + ": unexpected request stage distributions");
  return { blocks, distributions };
}

function validateRequestStages(stages, label) {
  check(object(stages) && typeof stages.attribution === "string" && stages.attribution.length > 0, label + ": invalid request stage attribution");
  const identities = new Map();
  const wall = validateRequestStageLane(stages.wall, label + " wall", identities);
  const gpuProfiler = validateRequestStageLane(stages.gpuProfiler, label + " GPU profiler", identities);
  check(stableJson(wall.blocks.map(({ block }) => block).sort((a, b) => a - b))
    === stableJson(gpuProfiler.blocks.map(({ block }) => block).sort((a, b) => a - b)), label + ": request stage lane blocks differ");
  check(wall.blocks[0].samples.length === gpuProfiler.blocks[0].samples.length, label + ": request stage lane sample counts differ");
  check(identities.size > 0, label + ": missing request stage identities");
  return { attribution: stages.attribution, wall, gpuProfiler };
}

function tuningDiagnostics(tuning) {
  const calibrators = [];
  let omitted = false;
  const visit = (value, path, depth) => {
    if (!object(value)) return;
    if (typeof value.method === "string" || typeof value.truncated === "boolean" || Array.isArray(value.points)) {
      if (calibrators.length === 32) { omitted = true; return; }
      const record = { path, ...pick(value, ["method", "reason", "truncated", "elapsedMs", "revision", "dtype", "shapeLimit", "availableShapes", "scheduledShapes"]) };
      if (Array.isArray(value.points)) {
        const points = value.points.flatMap((item) => Array.isArray(item.points) ? item.points : [item]).filter(object);
        const projections = points.filter((item) => typeof item.kernel === "string" && typeof item.tokens === "number");
        record.recordedPoints = points.length;
        if (projections.length || value.points.length === 0) {
          record.projectionPoints = projections.length;
          record.measuredProjectionPoints = projections.filter((item) => finite(item.genericMs) && [item.wideMs, item.tallMs, item.candidateMedianMs].some(finite)).length;
          record.selectedCandidates = Object.fromEntries(unique(projections.map((item) => item.kernel)).map((kernel) => [kernel, projections.filter((item) => item.kernel === kernel).length]));
          record.selectionReasons = Object.fromEntries(unique(projections.map((item) => item.reason).filter((reason) => typeof reason === "string")).map((reason) => [reason, projections.filter((item) => item.reason === reason).length]));
        }
      }
      calibrators.push(record);
    }
    if (depth === 4) { if (Object.values(value).some(object)) omitted = true; return; }
    for (const key of Object.keys(value).sort()) if (!["points", "trials", "numericalCheck"].includes(key) && object(value[key])) visit(value[key], `${path}.${key}`, depth + 1);
  };
  visit(tuning, "gpuTuning", 0);
  return { calibrators, ...(omitted ? { summaryTruncated: true } : {}) };
}

function publicModel(row, catalog) {
  check(object(row) && Object.hasOwn(catalog, row.id), `unknown current model ${row?.id}`);
  check(["q8", "bf16", "f32"].includes(row.precision) && typeof row.browser === "string" && row.browser.length > 0, `${row.id}: missing precision or browser`);
  check(["measured", "blocked", "unsupported", "failed", "pending"].includes(row.status), `${modelKey(row)}: invalid model status`);
  const result = { ...pick(row, ["id", "precision", "browser", "status", "reason", "reasonCode", "cacheMode", "correctness", "adapter"]),
    name: catalog[row.id].label || row.name || row.id, arch: catalog[row.id].arch };
  if (row.pack) result.pack = pick(row.pack, ["model", "arch", "precision", "responseModel", "answerUnit", "bytes", "sha256", "headerBytes", "headerSha256", "verification", "catalogHashVerified"]);
  if (row.runner) result.runner = pick(row.runner, ["actualThreads", "browser", "browserRequested", "browserVersion", "browserContext", "chromiumArgs", "expectedBackend", "headless", "targetVisibility", "timerPrivacy", "vkDriverFiles"]);
  if (Array.isArray(row.tuning)) result.tuningSummary = row.tuning.map((record) => ({ ...pick(record, ["block", "profile"]), ...tuningDiagnostics(record.gpuTuning) }));
  if (row.status === "pending") {
    result.status = "blocked";
    result.reason = row.reason || "The source capture ended before this model row was measured.";
    result.reasonCode = "incomplete-capture";
  }
  if (row.status === "measured") {
    check(sha(row.pack?.sha256) && Number.isSafeInteger(row.pack?.bytes) && row.pack.bytes > 0, `${modelKey(row)}: measured model lacks a verified pack identity`);
    check(Array.isArray(row.cases) && row.cases.length > 0, `${modelKey(row)}: measured model lacks cases`);
    result.cases = row.cases.map((item) => {
      check(typeof item.name === "string" && item.name.length > 0, `${modelKey(row)}: unnamed case`);
      const wall = validateMetric(item.wall, `${modelKey(row)} wall`);
      const gpu = validateMetric(item.gpu, `${modelKey(row)} GPU`);
      check(Array.isArray(item.gpu.kernels) && item.gpu.kernels.every((kernel) => typeof kernel.name === "string" && finite(kernel.medianMs) && finite(kernel.share)), `${modelKey(row)}: invalid per-kernel medians`);
      gpu.kernels = item.gpu.kernels.map((kernel) => {
        if (kernel.dispatchedSamples !== undefined || kernel.totalSamples !== undefined) check(Number.isSafeInteger(kernel.dispatchedSamples) && kernel.dispatchedSamples > 0 && Number.isSafeInteger(kernel.totalSamples) && kernel.totalSamples >= kernel.dispatchedSamples, `${modelKey(row)}: invalid kernel dispatch sample counts`);
        return pick(kernel, ["name", "medianMs", "share", "dispatchedSamples", "totalSamples"]);
      });
      if (item.gpu.kernelAttribution !== undefined) {
        check(typeof item.gpu.kernelAttribution === "string" && item.gpu.kernelAttribution.length > 0, `${modelKey(row)}: invalid kernel attribution`);
        gpu.kernelAttribution = item.gpu.kernelAttribution;
      }
      const result = { ...pick(item, ["name", "tokens", "tokenCounts", "tokenRange"]), wall, gpu };
      if (item.requestStages !== undefined) result.requestStages = validateRequestStages(item.requestStages, modelKey(row) + " " + item.name);
      return result;
    });
    check(new Set(result.cases.map((item) => item.name)).size === result.cases.length, `${modelKey(row)}: duplicated cases`);
  } else {
    check(!row.cases?.length, `${modelKey(row)}: unavailable model contains completed case timings`);
    result.reason ||= "No measurement was recorded.";
  }
  return result;
}

function captureIdentity(summary, kind) {
  check(summary.schemaVersion === 1, `${kind}: unsupported capture schema`);
  check(["complete", "partial", "failed"].includes(summary.status), `${kind}: capture is not terminal`);
  if (kind === "kernels") {
    check(summary.executionComplete === true, "kernels: capture execution is incomplete");
    check(summary.sourceChanged !== true, "kernels: source changed during capture");
  }
  const source = kind === "models" ? summary.runtime : summary.source || summary.runtime;
  check(object(source) && typeof source.commit === "string" && /^[a-f0-9]{40,64}$/.test(source.commit), `${kind}: missing source base commit`);
  check(typeof source.workingTreeDirty === "boolean", `${kind}: missing dirty-source state`);
  const fingerprint = kind === "models" ? source.sha256 : source.fingerprint;
  const harness = kind === "models" ? source.harnessSha256 : source.harnesses?.fingerprint;
  check(sha(fingerprint) && sha(harness), `${kind}: missing runtime or harness fingerprint`);
  return { kind, commit: source.commit, workingTreeDirty: source.workingTreeDirty, fingerprint, harness,
    files: normalizeRuntimeFiles(kind === "models" ? source.assets : source.files, kind),
    registrySha256: source.registrySha256,
    ...(kind === "kernels" ? { browser: summary.browser } : {}) };
}

function declaredCorrections(summary) {
  if (!summary.validatedSource) {
    check(!summary.corrections?.length, "model corrections require a validatedSource manifest");
    return [];
  }
  check(object(summary.validatedSource) && typeof summary.validatedSource.path === "string" && sha(summary.validatedSource.sha256), "validatedSource requires a relative JSON path and SHA-256");
  check(Array.isArray(summary.corrections), "validated model manifest requires a corrections array");
  check(summary.validation === undefined || object(summary.validation), "model validation metadata must be an object");
  const keys = new Set();
  for (const correction of summary.corrections) {
    check(object(correction) && correction.previousStatus === "failed" && (["blocked", "unsupported"].includes(correction.status) || isAttributionCorrection(correction)) && typeof correction.reasonCode === "string" && correction.reasonCode && typeof correction.reason === "string" && correction.reason, "corrections require an unavailable classification or a verified heterogeneous-kernel-attribution correction");
    const key = modelKey(correction), row = summary.models.find((item) => modelKey(item) === key);
    check(!keys.has(key) && row && row.status === correction.status && row.reasonCode === correction.reasonCode && row.reason === correction.reason, `correction does not match its derived model row ${key}`);
    keys.add(key);
  }
  return summary.corrections;
}

async function rederiveAttribution(capture, row) {
  const { protocol, runtime } = capture.summary, spec = MODELS[row.id];
  check(spec && spec.arch === row.arch && Number.isSafeInteger(protocol.blocks) && protocol.blocks >= 2 && Array.isArray(row.raw) && row.raw.length === protocol.blocks * 2, `attribution correction requires all raw wall/profile blocks ${modelKey(row)}`);
  const runs = [], keys = new Set();
  for (const source of row.raw) {
    const match = /^(\d+)-(gpu|wall)\.json$/.exec(basename(source));
    check(match && Number(match[1]) < protocol.blocks, `attribution correction has an invalid raw block ${source}`);
    const block = Number(match[1]), profile = match[2] === "gpu", key = `${block}/${profile}`;
    check(!keys.has(key), `attribution correction duplicated raw block ${key}`);
    keys.add(key);
    const result = JSON.parse(await evidenceFile(capture, source));
    validateProfileRun(result, { id: row.id, arch: row.arch, precision: row.precision, browser: row.browser, spec, samples: protocol.samples, warmups: protocol.warmups,
      profile, shapes: protocol.shapes, runtime: runtime.entry, packBytes: row.pack?.bytes, responseModel: row.pack?.responseModel, cacheMode: row.cacheMode || protocol.cacheMode });
    runs.push({ block, profile, result });
  }
  runs.sort((a, b) => a.block - b.block || (a.block % 2 ? Number(b.profile) - Number(a.profile) : Number(a.profile) - Number(b.profile)));
  const measured = summarizeProfileRuns(runs, { shapes: protocol.shapes, probabilityTolerance: protocol.probabilityTolerance, answerUnit: row.pack.answerUnit });
  check(measured.cases.some((item) => item.gpu.kernelAttribution), `attribution correction has no heterogeneous kernel labels ${modelKey(row)}`);
  const calculated = { ...measured, adapter: runs[0].result.adapter, runner: runs[0].result.runner,
    tuning: runs.map((run) => ({ block: run.block, profile: run.profile, gpuTuning: run.result.gpuTuning })) };
  for (const key of ["cases", "correctness", "adapter", "runner", "tuning"]) check(stableJson(row[key]) === stableJson(calculated[key]), `attribution correction differs from recalculated raw ${key} ${modelKey(row)}`);
}

async function verifyUnavailableCorrection(capture, original, derived, correction) {
  const classified = classifyProfileFailure(original.reason, { stage: "browser" }), key = modelKey(original);
  check(classified.status === correction.status && classified.reasonCode === correction.reasonCode && correction.reason === original.reason, `correction differs from the original browser failure classification ${key}`);
  const previous = original.raw || [], added = (derived.raw || []).filter((source) => !previous.includes(source));
  check(added.length > 0 && new Set(derived.raw).size === derived.raw.length, `correction lacks distinct added failure evidence ${key}`);
  const digests = new Set();
  for (const source of previous) digests.add(hash(await evidenceFile(capture, source)));
  for (const source of added) {
    const bytes = await evidenceFile(capture, source), digest = hash(bytes), raw = JSON.parse(bytes);
    check(!digests.has(digest), `duplicate raw failure evidence ${key}`);
    digests.add(digest);
    check(raw.done === true && raw.status === "error" && raw.error === original.reason
      && (raw.cases === undefined || Array.isArray(raw.cases) && raw.cases.length === 0)
      && raw.correctness?.ok !== true && (raw.correctness?.cases === undefined || Array.isArray(raw.correctness.cases) && raw.correctness.cases.length === 0), `correction requires a completed failed browser page without successful cases ${key}`);
    validateProfileBrowser(raw.runner, original.browser);
  }
}

async function verifyModelDerivation(capture, original) {
  const summary = capture.summary;
  const corrections = declaredCorrections(summary);
  captureIdentity(original, "models");
  check(!original.validatedSource, "nested validated model manifests are not supported");
  for (const key of ["runtime", "protocol", "environment", "generatedAt"]) check(stableJson(summary[key]) === stableJson(original[key]), `validated model manifest changed captured ${key}`);
  check(Array.isArray(original.models) && original.models.length === summary.models.length, "validated model manifest changed the captured row inventory");
  const changed = new Map(corrections.map((correction) => [modelKey(correction), correction]));
  for (const row of original.models) {
    const key = modelKey(row), derived = summary.models.find((item) => modelKey(item) === key), correction = changed.get(key);
    check(derived, `validated model manifest omitted ${key}`);
    const without = (value, fields) => Object.fromEntries(Object.entries(value).filter(([name]) => !fields.includes(name)));
    const attribution = correction && isAttributionCorrection(correction);
    const mutable = correction ? ["raw", "status", "reason", "reasonCode", ...(attribution ? ["cases", "correctness", "adapter", "runner", "tuning"] : [])] : ["raw"];
    check(stableJson(without(row, mutable)) === stableJson(without(derived, mutable)), `validated model manifest changed captured row evidence ${key}`);
    check(Array.isArray(derived.raw || []) && (row.raw || []).every((source) => derived.raw?.includes(source)), `validated model manifest removed raw evidence ${key}`);
    if (correction) {
      check(row.status === correction.previousStatus, `correction previous status differs ${key}`);
      if (attribution) {
        check(row.reason === ATTRIBUTION_FAILURE, `attribution correction requires the exact original label-change failure ${key}`);
        await rederiveAttribution(capture, derived);
      } else await verifyUnavailableCorrection(capture, row, derived, correction);
    }
  }
  const status = summary.models.some((row) => row.status === "failed") ? "failed" : summary.models.every((row) => row.status === "measured") ? "complete" : "partial";
  check(summary.status === status, "validated model manifest status does not match its corrected rows");
}

function declaredKernelCorrection(summary) {
  if (!summary.validatedSource) {
    check(!summary.corrections?.length, "kernel corrections require a validatedSource manifest");
    return null;
  }
  check(object(summary.validatedSource) && typeof summary.validatedSource.path === "string" && sha(summary.validatedSource.sha256), "kernel validatedSource requires a relative JSON path and SHA-256");
  check(Array.isArray(summary.corrections) && summary.corrections.length === 1 && Object.entries(CAPABILITY_CORRECTION).every(([key, value]) => summary.corrections[0]?.[key] === value), "kernel derivation permits only the paired-projections-wide shaderF16 capability correction");
  check(summary.validation === undefined || object(summary.validation), "kernel validation metadata must be an object");
  return summary.corrections[0];
}

async function rederiveKernelCapability(capture, registry) {
  const original = capture.summary;
  captureIdentity(original, "kernels");
  check(!original.validatedSource && original.status === "failed" && original.sourceChanged === false && original.browser === "chromium" && original.capabilities?.shaderF16 === false, "kernel capability correction requires an unchanged failed Chromium capture without shaderF16");
  const failures = original.runs.filter((run) => run.status === "failed"), harness = KERNEL_HARNESSES.find((item) => item.id === CAPABILITY_CORRECTION.id);
  check(failures.length === 1 && failures[0].id === harness.id && failures[0].reason === CAPABILITY_FAILURE && kernelHarnessSkipReason(harness, original.capabilities) === CAPABILITY_CORRECTION.reason, "kernel capability correction requires the sole exact wide-baseline capability failure");
  const runs = [], outputRuns = structuredClone(original.runs);
  check(new Set(original.runs.map((run) => run.id)).size === original.runs.length, "kernel capability correction has duplicate run IDs");
  for (const run of original.runs) {
    const raw = JSON.parse(await evidenceFile(capture, run.source)), definition = KERNEL_HARNESSES.find((item) => item.id === run.id);
    check(definition, `unknown captured kernel harness ${run.id}`);
    if (run.id === harness.id) {
      check(raw.status === "failed" && raw.reason === CAPABILITY_FAILURE && raw.pageResult?.error === "wide baseline requires shader-f16" && raw.pageResult.status === "error" && raw.pageResult.correctness?.ok === false && raw.pageResult.correctness.cases?.length === 0, "kernel capability correction raw failure differs from the known missing shader-f16 error");
      const corrected = outputRuns.find((item) => item.id === run.id);
      corrected.status = CAPABILITY_CORRECTION.status;
      corrected.reason = CAPABILITY_CORRECTION.reason;
      runs.push(corrected);
    } else {
      if (run.status === "passed") {
        validateKernelResult(definition, raw, registry);
        check(stableJson(run.adapter || null) === stableJson(raw.adapter || null) && stableJson(run.runner || null) === stableJson(raw.runner || null), `kernel raw adapter or runner changed ${run.id}`);
      }
      if (run.id === "compile") check(run.status === "passed" && raw.capabilities?.shaderF16 === false, "kernel capability correction requires raw compile evidence without shaderF16");
      runs.push({ ...run, ...(run.status === "passed" ? { result: raw } : {}) });
    }
  }
  check(runs.some((run) => run.id === "compile"), "kernel capability correction lacks its compilation probe");
  const coverage = kernelCoverage(registry, runs);
  return { ...structuredClone(original), ...coverage, status: coverage.coverageComplete ? "complete" : "partial", runs: outputRuns };
}

export async function deriveKernelCapabilityManifest({ sourcePath, registry, validation }) {
  const capture = await readCapture(resolve(sourcePath), "kernels"), derived = await rederiveKernelCapability(capture, registry);
  return { ...derived, validatedSource: { path: basename(sourcePath), sha256: hash(capture.bytes) }, corrections: [structuredClone(CAPABILITY_CORRECTION)], ...(validation ? { validation } : {}) };
}

async function verifyKernelDerivation(capture, bytes, registry) {
  declaredKernelCorrection(capture.summary);
  const original = JSON.parse(bytes), calculated = await rederiveKernelCapability({ ...capture, summary: original }, registry);
  const supplied = Object.fromEntries(Object.entries(capture.summary).filter(([key]) => !["validatedSource", "corrections", "validation"].includes(key)));
  check(stableJson(supplied) === stableJson(calculated), "derived kernel manifest differs from revalidated raw coverage or changed captured metadata");
}

function mergeModels(summaries, catalog, browsers) {
  const rows = new Map();
  const originals = new Map();
  const precisions = new Set(["q8"]);
  for (const summary of summaries) {
    check(Array.isArray(summary.models), "models: missing result rows");
    const requested = summary.protocol.precisions ?? unique(summary.models.map((row) => row.precision));
    check(requested === "all" || Array.isArray(requested) && requested.every((value) => ["q8", "bf16", "f32"].includes(value)), "models: invalid requested precisions");
    for (const precision of requested === "all" ? ["q8", "bf16", "f32"] : requested) precisions.add(precision);
    for (const row of summary.models) {
      const result = publicModel(row, catalog), key = modelKey(result);
      precisions.add(row.precision);
      if (rows.has(key)) check(originals.get(key) === stableJson(row), `contradictory model row ${key}`);
      else { rows.set(key, result); originals.set(key, stableJson(row)); }
    }
  }
  for (const [id, spec] of Object.entries(catalog)) for (const precision of [...precisions].sort()) for (const browser of browsers) {
    const key = modelKey({ id, precision, browser });
    if (rows.has(key)) continue;
    const supported = (spec.precisions || [spec.defaultPrecision || "q8"]).includes(precision);
    rows.set(key, { id, name: spec.label || id, arch: spec.arch, precision, browser,
      status: supported ? "blocked" : "unsupported", reasonCode: supported ? "missing-capture" : "catalog-precision",
      reason: supported ? "No capture was supplied for this current catalog model, precision, and browser." : `Catalog does not support ${precision}.` });
  }
  const order = Object.keys(catalog);
  const precisionOrder = ["q8", "bf16", "f32"];
  return [...rows.values()].sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id) || precisionOrder.indexOf(a.precision) - precisionOrder.indexOf(b.precision) || a.browser.localeCompare(b.browser));
}

const gate = (rows, field) => new Set(rows.map((row) => row[field])).size === 1 ? rows[0][field] : "mixed";
function mergeKernels(summaries, registry, browsers) {
  check(Array.isArray(registry) && registry.length > 0 && new Set(registry).size === registry.length, "missing or duplicated current kernel registry");
  const supplied = new Map();
  for (const summary of summaries) {
    check(typeof summary.browser === "string" && summary.browser.length > 0, "kernels: missing browser");
    check(summary.registered === registry.length, "kernels: registered count differs from the current registry");
    check(Array.isArray(summary.kernels), "kernels: missing result rows");
    const names = new Set();
    for (const row of summary.kernels) {
      check(registry.includes(row.name) && !names.has(row.name), `kernels: unknown or duplicated kernel ${row.name}`);
      names.add(row.name);
      check(["measured", "unsupported", "failed", "validated", "timed", "compile-only", "unmeasured"].includes(row.status), `${row.name}: invalid kernel status`);
      check(["passed", "unsupported", "not-run"].includes(row.compileStatus) && ["passed", "finite-only", "not-run"].includes(row.numericalStatus) && ["isolated", "profiled", "composite", "not-run"].includes(row.timingStatus), `${row.name}: invalid kernel gate state`);
      check(row.status !== "measured" || row.compileStatus === "passed" && row.numericalStatus === "passed" && row.timingStatus !== "not-run", `${row.name}: compilation alone cannot establish a measured kernel`);
      const key = `${summary.browser}/${row.name}`;
      if (supplied.has(key)) check(stableJson(supplied.get(key)) === stableJson(row), `contradictory kernel row ${key}`);
      else supplied.set(key, row);
    }
  }
  return registry.map((name) => {
    const records = browsers.map((browser) => ({ browser, ...structuredClone(supplied.get(`${browser}/${name}`) || {
      status: "unmeasured", compileStatus: "not-run", numericalStatus: "not-run", timingStatus: "not-run",
      reason: "No kernel capture row was supplied for this browser.", harnesses: [], evidence: [],
    }) }));
    return { name, status: records.some((row) => row.status === "failed") ? "failed" : gate(records, "status"),
      compileStatus: gate(records, "compileStatus"), numericalStatus: gate(records, "numericalStatus"), timingStatus: gate(records, "timingStatus"),
      reason: records.map((row) => `${row.browser}: ${row.reason || row.status}`).join(" "),
      harnesses: unique(records.flatMap((row) => row.harnesses || [])),
      browsers: records.map((row) => {
        const result = pick(row, ["browser", "status", "compileStatus", "numericalStatus", "timingStatus", "reason", "harnesses", "compilation", "isolated", "evidence", "skipped"]);
        if (Array.isArray(row.evidence)) result.evidence = row.evidence.map(({ measurements, ...entry }) => ({ ...entry,
          ...(Array.isArray(measurements) ? { measurementRecords: measurements.length } : {}) }));
        return result;
      }) };
  });
}

function environmentSummary(models, summaries) {
  const environments = summaries.map((summary) => summary.environment).filter(Boolean);
  const hardwareNames = models.flatMap((row) => {
    const adapter = row.adapter || {};
    return [adapter.description, adapter.name, adapter.info?.description].filter((name) => name && name !== "WebGPU");
  });
  const hostGpus = unique(summaries.flatMap((summary) => summary.models.flatMap((row) => row.before?.nvidia?.rows?.map((gpu) => gpu[1]) || [])));
  const browsers = unique(models.map((row) => row.browser)).flatMap((browser) => {
    const versions = unique(models.filter((row) => row.browser === browser).map((row) => row.runner?.browserVersion).filter(Boolean));
    return versions.length ? versions.map((version) => `${browser} ${version}`) : [browser];
  });
  return { gpu: unique(hardwareNames).join("; ") || "Adapter details are recorded per model and raw capture.",
    ...(hostGpus.length ? { hostGpus } : {}),
    browser: browsers.join("; "),
    os: unique(environments.map((env) => [env.platform, env.release].filter(Boolean).join(" "))).join("; ") || "See raw capture metadata.",
    captures: environments };
}

export function assembleProfileReport({ modelSummaries = [], kernelSummaries = [], catalog = MODELS, registry, currentRuntimeFiles, registrySha256, improvements = [], limitations = [] }) {
  check(modelSummaries.length + kernelSummaries.length > 0, "supply at least one capture summary");
  const identities = [...modelSummaries.map((summary) => captureIdentity(summary, "models")), ...kernelSummaries.map((summary) => captureIdentity(summary, "kernels"))];
  const files = identities[0].files;
  for (const identity of identities) {
    check(stableJson(identity.files) === stableJson(files), "mixed runtime source asset hashes");
    check(identity.commit === identities[0].commit, "mixed source base commits");
    if (identity.kind === "kernels" && registrySha256) check(identity.registrySha256 === registrySha256, "kernel capture does not match the current registry source");
    const peers = identities.filter((item) => item.kind === identity.kind);
    check(peers.every((item) => item.fingerprint === identity.fingerprint && item.harness === identity.harness), `${identity.kind}: mixed runtime or harness fingerprints`);
  }
  if (currentRuntimeFiles) check(stableJson(normalizeRuntimeFiles(currentRuntimeFiles)) === stableJson(files), "capture runtime differs from the current runtime assets");
  const browsers = unique([...modelSummaries.flatMap((summary) => [...(summary.protocol?.browsers || []), ...(summary.models || []).map((row) => row.browser)]), ...kernelSummaries.map((summary) => summary.browser)]);
  check(browsers.length > 0 && browsers.every((browser) => typeof browser === "string" && browser.length > 0), "no capture browsers were reported");
  const modelProtocol = sharedProtocol(modelSummaries, "models"), kernelProtocol = sharedProtocol(kernelSummaries, "kernels");
  const models = mergeModels(modelSummaries, catalog, browsers), rows = mergeKernels(kernelSummaries, registry, browsers);
  check(Array.isArray(improvements) && improvements.every(object), "improvements must be an array of objects");
  for (const improvement of improvements) for (const key of ["beforeMs", "afterMs", "changePercent"]) {
    const value = improvement[key];
    check(value === undefined || value === null || typeof value === "number" && Number.isFinite(value) && (key === "changePercent" || value >= 0), `invalid improvement ${key}`);
  }
  check(Array.isArray(limitations) && limitations.every((item) => typeof item === "string"), "limitations must be an array of strings");
  const sources = [...modelSummaries, ...kernelSummaries];
  const timestamps = sources.map((summary) => summary.completedAt || summary.generatedAt);
  check(timestamps.every((value) => typeof value === "string" && Number.isFinite(Date.parse(value))), "capture completion timestamp is missing or invalid");
  const failed = sources.some((summary) => summary.status === "failed") || models.some((row) => row.status === "failed") || rows.some((row) => row.status === "failed");
  const complete = modelSummaries.length > 0 && kernelSummaries.length > 0 && models.every((row) => row.status === "measured") && rows.every((row) => row.status === "measured");
  const automaticLimits = ["Results apply to the recorded adapters, browsers, runtime bytes, precisions, and fixtures. They do not establish a universally fastest kernel.",
    "Unprofiled wall latency and instrumented GPU timings come from separate requests. Instrumentation changes dispatch passes and may change optional paths; do not subtract these metrics.",
    "Numerical checks establish agreement for the recorded fixtures, not model accuracy."];
  for (const summary of modelSummaries) for (const correction of declaredCorrections(summary)) automaticLimits.push(`Validated model-manifest correction: ${modelKey(correction)} changed from ${correction.previousStatus} to ${correction.status} (${correction.reasonCode}). ${isAttributionCorrection(correction) ? "Timings were rederived after raw numerical and protocol validation; the original failed manifest and raw blocks are archived." : "The original manifest and raw failure evidence are archived."}`);
  for (const summary of kernelSummaries) if (declaredKernelCorrection(summary)) automaticLimits.push("Validated kernel-manifest correction: paired-projections-wide changed from failed to skipped because captured shaderF16 support was absent. Passed raw fixtures were revalidated and coverage was recomputed without contributions from that skipped run; the original failure is archived.");
  if (models.some((row) => row.status !== "measured")) automaticLimits.push("Some catalog model, precision, or browser rows remain unavailable. Their named states and reasons are retained.");
  if (rows.some((row) => row.status !== "measured")) automaticLimits.push("Kernel coverage is incomplete or differs by browser. Compilation, numerical checks, isolated timings, and composite timings remain separate.");
  if (models.some((row) => row.tuningSummary?.some((record) => record.calibrators.some((calibrator) => calibrator.truncated === true)))) automaticLimits.push("Some tuning calibrations were truncated. A valid model capture does not establish that every tuner completed or that its selected kernel is optimal.");
  return { schemaVersion: 1, status: failed ? "failed" : complete ? "complete" : "partial",
    generatedAt: new Date(Math.max(...timestamps.map(Date.parse))).toISOString(),
    runtime: { commit: identities[0].commit, workingTreeDirty: identities.some((identity) => identity.workingTreeDirty),
      sourceFingerprint: hash(files.map((file) => `${file.path}:${file.sha256}\n`).join("")), assets: files,
      captures: identities.map(({ files: ignored, ...identity }) => identity).sort((a, b) => stableJson(a).localeCompare(stableJson(b))) },
    protocol: { models: modelProtocol, kernels: kernelProtocol, browsers },
    environment: environmentSummary(models, modelSummaries), models, kernels: { registered: registry.length, rows },
    improvements, limitations: [...new Set([...automaticLimits, ...limitations])], artifacts: [] };
}

async function readCapture(path, kind) {
  const bytes = await readFile(path);
  return { kind, path: resolve(path), bytes, summary: JSON.parse(bytes), id: `${kind}-${hash(bytes).slice(0, 16)}` };
}

async function evidenceFile(capture, source) {
  check(typeof source === "string" && source.endsWith(".json") && !source.includes("\\") && !isAbsolute(source) && !source.split("/").some((part) => !part || part === "." || part === ".."), `${capture.id}: raw evidence must be a relative JSON file inside its capture directory`);
  const folder = await realpath(dirname(capture.path)), path = await realpath(resolve(folder, source));
  const within = relative(folder, path);
  check(within && within !== ".." && !within.startsWith(`..${sep}`) && !isAbsolute(within), `${capture.id}: raw evidence leaves its capture directory`);
  check((await stat(path)).isFile(), `${capture.id}: raw evidence is not a regular file`);
  const bytes = await readFile(path);
  JSON.parse(bytes);
  return bytes;
}

async function currentSources(root) {
  const folder = resolve(root, "js/src");
  const paths = (await readdir(folder, { recursive: true })).filter((path) => [".js", ".wasm"].includes(extname(path))).sort();
  return Promise.all(paths.map(async (path) => ({ path, sha256: hash(await readFile(resolve(folder, path))) })));
}

export async function writeProfileReport({ modelPaths = [], kernelPaths = [], evidencePaths = [], output, root, improvements = [], limitations = [] }) {
  check(typeof output === "string" && typeof root === "string", "output and repository root are required");
  const destination = resolve(root, output), relativeOutput = relative(root, destination).split(sep).join("/");
  check(relativeOutput.startsWith("docs/") && !relativeOutput.split("/").includes("..") && destination.endsWith(".json"), "public report output must be a JSON file under repository docs/");
  const captures = await Promise.all([...modelPaths.map((path) => readCapture(resolve(root, path), "models")), ...kernelPaths.map((path) => readCapture(resolve(root, path), "kernels"))]);
  captures.sort((a, b) => a.id.localeCompare(b.id));
  const additional = await Promise.all(evidencePaths.map(async (path) => {
    check(extname(path) === ".json", "additional evidence must be a JSON file");
    const bytes = await readFile(resolve(root, path)), parsed = JSON.parse(bytes);
    return { bytes, label: typeof parsed?.label === "string" && parsed.label.trim() ? parsed.label : basename(path) };
  }));
  additional.sort((a, b) => hash(a.bytes).localeCompare(hash(b.bytes)) || a.label.localeCompare(b.label));
  const registrySource = await readFile(resolve(root, "crates/kevala/src/gpu.rs"), "utf8"), registry = parseKernelRegistry(registrySource);
  for (const capture of captures) if (capture.summary.validatedSource) {
    const bytes = await evidenceFile(capture, capture.summary.validatedSource.path);
    check(hash(bytes) === capture.summary.validatedSource.sha256, `${capture.id}: validatedSource SHA-256 mismatch`);
    if (capture.kind === "models") await verifyModelDerivation(capture, JSON.parse(bytes));
    else await verifyKernelDerivation(capture, bytes, registry);
    capture.original = bytes;
  }
  const report = assembleProfileReport({ modelSummaries: captures.filter((capture) => capture.kind === "models").map((capture) => capture.summary),
    kernelSummaries: captures.filter((capture) => capture.kind === "kernels").map((capture) => capture.summary),
    registry, registrySha256: hash(registrySource), currentRuntimeFiles: await currentSources(root), improvements, limitations });
  const evidence = [], archives = new Map(), sourceArchives = [], links = new Map(), provenance = [];
  const rawPrefix = `${relative(root, resolve(dirname(destination), "raw")).split(sep).join("/")}/`;
  const archive = (bytes) => {
    const digest = hash(bytes), gzip = gzipSync(bytes, { level: 9 }), filename = `${digest}.json.gz`;
    archives.set(filename, gzip);
    return { url: `${rawPrefix}${filename}`, sha256: digest, bytes: bytes.length, gzipSha256: hash(gzip), gzipBytes: gzip.length };
  };
  for (const capture of captures) {
    const source = archive(capture.bytes);
    sourceArchives.push({ label: `${capture.kind === "models" ? "Model" : "Kernel"} capture ${capture.id.slice(-16)}`, ...source });
    if (capture.original) {
      const original = archive(capture.original);
      sourceArchives.push({ label: `Original ${capture.kind === "models" ? "model" : "kernel"} capture ${capture.summary.validatedSource.sha256.slice(0, 16)}`, ...original });
      provenance.push({ kind: capture.kind === "models" ? "validated-model-manifest" : "validated-kernel-manifest", capture: source, original, corrections: capture.summary.corrections, ...(capture.summary.validation ? { validation: capture.summary.validation } : {}) });
    }
    const references = unique(capture.kind === "models" ? capture.summary.models.flatMap((row) => row.raw || []) : capture.summary.runs.map((run) => run.source).filter(Boolean));
    for (const reference of references) {
      const record = archive(await evidenceFile(capture, reference));
      evidence.push({ capture: capture.id, source: reference, ...record });
      links.set(`${capture.id}/${reference}`, record);
    }
    if (capture.kind === "models") for (const input of capture.summary.models) {
      const row = report.models.find((item) => modelKey(item) === modelKey(input));
      if (input.raw?.length) row.evidence = input.raw.map((reference) => links.get(`${capture.id}/${reference}`));
    }
    else for (const row of report.kernels.rows) {
      const browser = row.browsers.find((item) => item.browser === capture.summary.browser);
      for (const item of browser.evidence || []) if (item.source) {
        const reference = links.get(`${capture.id}/${item.source}`);
        check(reference, `${capture.id}: kernel evidence points to an unarchived run`);
        delete item.source;
        item.artifact = reference;
      }
    }
  }
  const additionalEvidence = additional.map((item) => ({ kind: "additional-evidence", label: item.label, ...archive(item.bytes) }));
  const index = { schemaVersion: 1, sources: sourceArchives, files: evidence.sort((a, b) => `${a.capture}/${a.source}`.localeCompare(`${b.capture}/${b.source}`)), ...(additionalEvidence.length ? { additionalEvidence } : {}) };
  if (provenance.length) report.provenance = provenance;
  report.artifacts = [...sourceArchives, ...additionalEvidence, { label: "Raw evidence index", ...archive(Buffer.from(`${stableJson(index, 2)}\n`)) }];
  await mkdir(resolve(dirname(destination), "raw"), { recursive: true });
  for (const [filename, bytes] of [...archives].sort(([a], [b]) => a.localeCompare(b))) await writeFile(resolve(dirname(destination), "raw", filename), bytes);
  await writeFile(`${destination}.tmp`, `${stableJson(report, 2)}\n`);
  await rename(`${destination}.tmp`, destination);
  return report;
}
