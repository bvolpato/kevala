import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { gunzipSync } from "node:zlib";
import { MODELS } from "../js/src/source.js";
import { CLEF_HEAD_ATTRIBUTION, SHAPES, summarizeProfileRuns } from "../scripts/lib/model-profile.mjs";
import { assembleProfileReport, deriveKernelCapabilityManifest, normalizeRuntimeFiles, writeProfileReport } from "../scripts/lib/profile-report.mjs";
import { kernelCoverage } from "../scripts/lib/kernel-profile-catalog.mjs";
import { reportOptions } from "../scripts/report-kernel-profile.mjs";

const hash = (value) => createHash("sha256").update(value).digest("hex");
const digest = (letter) => letter.repeat(64);
const catalog = { first: { label: "First model", arch: "one" }, second: { label: "Second model", arch: "two", precisions: ["q8", "bf16"] } };
const registry = ["matmul", "reduce"];
const runtimeFiles = [{ path: "index.js", sha256: hash("export {};\n") }];
const protocol = { samples: 2, warmups: 1, blocks: 2, shapes: [0], cacheMode: "uncached", models: ["first"], precisions: ["q8"], browsers: ["firefox"] };
const metric = (value) => ({ medianMs: value, p95Ms: value + 1, samples: [value - 1, value + 1], blockMediansMs: [value - 1, value + 1] });
const measuredModel = (browser = "firefox") => ({ id: "first", name: "First", arch: "one", precision: "q8", browser, status: "measured",
  pack: { bytes: 42, sha256: digest("d"), headerSha256: digest("e"), file: { path: "/private/model.kevala" } },
  adapter: { name: "Test GPU", isFallbackAdapter: false, features: ["timestamp-query"] }, runner: { browser, browserVersion: "123", binary: "/private/browser" },
  cases: [{ name: "short", tokens: 40, wall: metric(12), gpu: { ...metric(4), kernels: [{ name: "matmul", medianMs: 4, share: 1 }] } }],
  tuning: [{ large: "tuning" }], before: { privateTelemetry: true }, raw: [] });
const measuredKernel = (name) => ({ name, status: "measured", compileStatus: "passed", numericalStatus: "passed", timingStatus: "isolated", reason: "CPU reference and timing passed.", harnesses: ["isolated"],
  isolated: [{ tokens: 1, timings: [{ label: "current", medianMs: 0.01, samples: [0.009, 0.011] }] }], evidence: [] });
const requestStageMsFields = ["reportedPrepareMs", "reportedForwardMs", "reportedTotalMs", "postForwardResidualMs", "outerResidualMs"];
const requestStageFractionFields = ["prepareOfRuntimeTotal", "forwardOfRuntimeTotal", "postForwardResidualOfRuntimeTotal", "postForwardResidualOfWall", "runtimeTotalOfWall", "outerResidualOfWall"];
function testMedian(values) {
  const sorted = [...values].sort((a, b) => a - b), middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function requestStageTestSample(sampleIndex, block, lane) {
  const reportedPrepareMs = [1.1, 1.8][sampleIndex], reportedForwardMs = [8.2, 11.9][sampleIndex];
  const reportedTotalMs = [9.2, 15.1][sampleIndex], wallMs = [14.2, 18.3][sampleIndex] + block * 2 + lane;
  const postForwardResidualMs = reportedTotalMs - reportedPrepareMs - reportedForwardMs, outerResidualMs = wallMs - reportedTotalMs;
  return { sampleIndex, requestHash: digest(sampleIndex ? "b" : "a"), tokens: sampleIndex ? 42 : 40,
    reportedPrepareMs, reportedForwardMs, reportedTotalMs, wallMs, postForwardResidualMs, outerResidualMs,
    prepareOfRuntimeTotal: reportedPrepareMs / reportedTotalMs, forwardOfRuntimeTotal: reportedForwardMs / reportedTotalMs,
    postForwardResidualOfRuntimeTotal: postForwardResidualMs / reportedTotalMs,
    postForwardResidualOfWall: postForwardResidualMs / wallMs, runtimeTotalOfWall: reportedTotalMs / wallMs,
    outerResidualOfWall: outerResidualMs / wallMs };
}
function requestStageTestLane(lane) {
  const blocks = [0, 1].map((block) => ({ block, samples: [0, 1].map((sampleIndex) => requestStageTestSample(sampleIndex, block, lane)) }));
  const distributions = {};
  for (const [fields, fractional] of [[requestStageMsFields, false], [requestStageFractionFields, true]]) for (const field of fields) {
    const valuesByBlock = blocks.map(({ samples }) => samples.map((sample) => sample[field])), values = valuesByBlock.flat();
    const sorted = [...values].sort((a, b) => a - b), mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    distributions[field] = fractional
      ? { median: testMedian(values), mean, p95: sorted[Math.ceil(sorted.length * 0.95) - 1], samples: values, blockMedians: valuesByBlock.map(testMedian) }
      : { medianMs: testMedian(values), meanMs: mean, p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1], samples: values, blockMediansMs: valuesByBlock.map(testMedian) };
  }
  return { blocks, distributions };
}
function requestStagesFixture() {
  return { attribution: "Test timer boundaries; residuals include intervening CPU, copy, and GPU wait work.",
    wall: requestStageTestLane(0), gpuProfiler: requestStageTestLane(1) };
}
function clefHeadSnapshot(requestIndex, selected = "gpu") {
  return { ...(requestIndex === undefined ? {} : { requestIndex }), requested: "auto", selected, validated: true,
    reason: selected === "gpu" ? null : "hidden-or-lexical-shape", flavor: "relaxed", coordTile: 0, headDiagnosticsTile: 0 };
}
function clefHeadLane(profile) {
  return [0, 1].map((block) => ({ block, afterWarmup: clefHeadSnapshot(),
    samples: [0, 1].map((sampleIndex) => ({ sampleIndex, requestHash: digest(sampleIndex ? "b" : "a"),
      requests: [clefHeadSnapshot(0, profile ? "gpu" : "cpu"), clefHeadSnapshot(1, "cpu")] })) }));
}
function clefHeadFixture() { return { attribution: CLEF_HEAD_ATTRIBUTION, wall: clefHeadLane(false), gpuProfiler: clefHeadLane(true) }; }
function models(browser = "firefox") {
  return { schemaVersion: 1, status: "complete", generatedAt: "2026-11-06T01:00:00.000Z", runtime: { commit: "a".repeat(40), workingTreeDirty: true,
    sha256: digest("b"), harnessSha256: digest("c"), assets: structuredClone(runtimeFiles), entry: "http://localhost/private/index.js" },
  protocol: { ...structuredClone(protocol), browsers: [browser] }, environment: { platform: "linux", release: "test", cpu: "Test CPU" }, models: [measuredModel(browser)] };
}
function kernels(browser = "firefox") {
  return { schemaVersion: 1, status: "complete", executionComplete: true, sourceChanged: false, completedAt: "2026-11-06T02:00:00.000Z", browser,
    source: { commit: "a".repeat(40), workingTreeDirty: false, fingerprint: digest("f"), files: runtimeFiles.map((file) => ({ ...file, path: `js/src/${file.path}` })), harnesses: { fingerprint: digest("1") } },
    protocol: { isolatedSamples: 7, isolatedWarmups: 3, isolatedTokens: [1, 47, 128, 512], timeoutSeconds: 300 }, registered: registry.length,
    kernels: registry.map(measuredKernel), runs: [] };
}
const assemble = (options = {}) => assembleProfileReport({ catalog, registry, modelSummaries: [models()], kernelSummaries: [kernels()], ...options });

test("assembly keeps browser gates, separate timings, sample arrays, pack hashes and dirty source identity", () => {
  const chrome = kernels("chromium");
  chrome.kernels[0] = { name: "matmul", status: "unsupported", compileStatus: "unsupported", numericalStatus: "not-run", timingStatus: "not-run", reason: "Missing shader-f16", harnesses: ["compile"], evidence: [] };
  const report = assemble({ modelSummaries: [models(), models("chromium")], kernelSummaries: [kernels(), chrome] });
  assert.equal(report.status, "partial");
  assert.equal(report.generatedAt, "2026-11-06T02:00:00.000Z");
  assert.equal(report.runtime.commit, "a".repeat(40));
  assert.equal(report.runtime.workingTreeDirty, true);
  assert.equal(report.runtime.captures.find((capture) => capture.kind === "kernels").workingTreeDirty, false);
  assert.deepEqual(report.models[0].cases[0].wall.samples, [11, 13]);
  assert.deepEqual(report.models[0].cases[0].gpu.samples, [3, 5]);
  assert.equal(report.models[0].pack.sha256, digest("d"));
  assert.equal(report.models[0].pack.file, undefined);
  assert.equal(report.models[0].before, undefined);
  assert.equal(report.models[0].tuning, undefined);
  assert.equal(report.models[0].runner.binary, undefined);
  const row = report.kernels.rows[0];
  assert.equal(row.status, "mixed");
  assert.equal(row.compileStatus, "mixed");
  assert.equal(row.browsers.find((item) => item.browser === "chromium").status, "unsupported");
  assert.deepEqual(row.browsers.find((item) => item.browser === "firefox").isolated[0].timings[0].samples, [0.009, 0.011]);
});

test("missing models, precision captures and kernel browser rows stay named and unavailable", () => {
  const input = models();
  input.protocol.precisions = ["q8", "bf16"];
  input.models.push({ id: "second", precision: "bf16", browser: "firefox", status: "blocked", reason: "Missing pack" });
  const chrome = kernels("chromium");
  chrome.kernels = [];
  const report = assemble({ modelSummaries: [input], kernelSummaries: [kernels(), chrome] });
  const missing = report.models.find((row) => row.id === "second" && row.precision === "q8" && row.browser === "chromium");
  assert.equal(missing.status, "blocked");
  assert.equal(missing.reasonCode, "missing-capture");
  assert.equal(missing.cases, undefined);
  const unsupported = report.models.find((row) => row.id === "first" && row.precision === "bf16");
  assert.equal(unsupported.status, "unsupported");
  assert.equal(unsupported.cases, undefined);
  assert.equal(report.kernels.rows.length, registry.length);
  assert.equal(report.kernels.rows[0].browsers.find((row) => row.browser === "chromium").timingStatus, "not-run");
  assert.equal(report.status, "partial");
});

test("Q8 rows lead each model and known browser versions suppress unversioned duplicates", () => {
  const firefox = models(), chromium = models("chromium");
  firefox.protocol.precisions = chromium.protocol.precisions = ["q8", "bf16", "f32"];
  const report = assemble({ modelSummaries: [firefox, chromium], kernelSummaries: [kernels(), kernels("chromium")] });
  assert.deepEqual(report.models.filter((row) => row.id === "first").map((row) => [row.precision, row.browser]), [
    ["q8", "chromium"], ["q8", "firefox"], ["bf16", "chromium"], ["bf16", "firefox"], ["f32", "chromium"], ["f32", "firefox"],
  ]);
  assert.equal(report.environment.browser, "chromium 123; firefox 123");
  assert.equal(report.models.filter((row) => row.status === "measured").length, 2);
  const missing = assemble({ kernelSummaries: [kernels("chromium")] });
  assert.equal(missing.environment.browser, "chromium; firefox 123");
});

test("tuning summaries preserve deadlines, shape coverage and measured candidate counts without inventing absent diagnostics", () => {
  const input = models();
  input.models[0].tuning = [{ block: 0, profile: false, gpuTuning: { method: "timestamp-query", truncated: true, reason: "deadline", elapsedMs: 2000,
    shapeLimit: 12, availableShapes: 10, scheduledShapes: 10, points: [{ name: "matrix", points: [
      { tokens: 16, kernel: "wide", genericMs: 0.2, wideMs: 0.1, reason: "stable-wide-win" },
      { tokens: 128, kernel: "generic", genericMs: 0.2, wideMs: null, elapsedMs: 100, reason: "scratch-limit" },
    ] }], recurrence: { method: "generic", reason: "unsupported" } } }];
  const report = assemble({ modelSummaries: [input] });
  const tuning = report.models[0].tuningSummary[0];
  assert.equal(tuning.block, 0);
  assert.equal(tuning.profile, false);
  const matrix = tuning.calibrators.find((item) => item.path === "gpuTuning");
  assert.equal(matrix.truncated, true);
  assert.equal(matrix.reason, "deadline");
  assert.equal(matrix.shapeLimit, 12);
  assert.equal(matrix.availableShapes, 10);
  assert.equal(matrix.measuredProjectionPoints, 1);
  assert.deepEqual(matrix.selectedCandidates, { generic: 1, wide: 1 });
  assert.equal(tuning.calibrators.find((item) => item.path === "gpuTuning.recurrence").truncated, undefined);
  assert.equal(report.models[0].status, "measured");
  assert.ok(report.limitations.some((reason) => reason.includes("Some tuning calibrations were truncated")));
});

test("compilation alone cannot become numerical or GPU timing coverage", () => {
  const input = kernels();
  input.kernels[0] = { ...input.kernels[0], status: "compile-only", numericalStatus: "not-run", timingStatus: "not-run", isolated: undefined };
  const report = assemble({ kernelSummaries: [input] });
  assert.equal(report.kernels.rows[0].status, "compile-only");
  assert.equal(report.kernels.rows[0].numericalStatus, "not-run");
  input.kernels[0].status = "measured";
  assert.throws(() => assemble({ kernelSummaries: [input] }), /compilation alone cannot establish a measured kernel/);
});

test("assembly refuses mixed source bytes, fingerprints, harnesses and model sampling protocols", () => {
  for (const field of ["sha256", "harnessSha256"]) {
    const second = models("chromium"); second.runtime[field] = digest("2");
    assert.throws(() => assemble({ modelSummaries: [models(), second] }), /mixed runtime or harness fingerprints/);
  }
  const changed = kernels(); changed.source.files[0].sha256 = digest("2");
  assert.throws(() => assemble({ kernelSummaries: [changed] }), /mixed runtime source asset hashes/);
  const protocolChange = models("chromium"); protocolChange.protocol.samples++;
  assert.throws(() => assemble({ modelSummaries: [models(), protocolChange] }), /mixed profiling protocols/);
  assert.throws(() => assemble({ currentRuntimeFiles: [{ path: "index.js", sha256: digest("2") }] }), /differs from the current runtime assets/);
  const registryChange = kernels(); registryChange.source.registrySha256 = digest("2");
  assert.throws(() => assemble({ kernelSummaries: [registryChange], registrySha256: digest("3") }), /current registry source/);
});

test("captures without optional selection metadata use explicit row identities and keep the same measurement protocol", () => {
  const original = models();
  for (const key of ["models", "precisions", "browsers"]) delete original.protocol[key];
  const newer = models("chromium");
  newer.protocol.kernelShare = "per-label median divided by the sum of label medians; not a share of wall latency";
  const report = assemble({ modelSummaries: [original, newer] });
  assert.equal(report.models.filter((row) => row.id === "first" && row.status === "measured").length, 2);
  assert.deepEqual(report.protocol.browsers, ["chromium", "firefox"]);
});

test("assembly refuses contradictory duplicate rows, unfinished captures and fake unavailable timings", () => {
  const conflict = models(); conflict.models[0].cases[0].wall.medianMs++;
  assert.throws(() => assemble({ modelSummaries: [models(), conflict] }), /contradictory model row/);
  const pending = models(); pending.status = "running";
  assert.throws(() => assemble({ modelSummaries: [pending] }), /capture is not terminal/);
  const unfinished = kernels(); unfinished.executionComplete = false;
  assert.throws(() => assemble({ kernelSummaries: [unfinished] }), /execution is incomplete/);
  const stale = kernels(); stale.sourceChanged = true;
  assert.throws(() => assemble({ kernelSummaries: [stale] }), /source changed during capture/);
  const unavailable = models(); unavailable.models[0].status = "unsupported";
  assert.throws(() => assemble({ modelSummaries: [unavailable] }), /unavailable model contains completed case timings/);
  const invalid = models(); invalid.models[0].cases[0].gpu.samples[0] = null;
  assert.throws(() => assemble({ modelSummaries: [invalid] }), /invalid or missing raw timing samples/);
});

test("normalization compares source maps across runtime directory prefixes without hiding duplicate paths", () => {
  assert.deepEqual(normalizeRuntimeFiles([{ path: "js/src/index.js", sha256: digest("a") }]), [{ path: "index.js", sha256: digest("a") }]);
  assert.throws(() => normalizeRuntimeFiles([{ path: "../index.js", sha256: digest("a") }]), /invalid runtime asset path/);
  assert.throws(() => normalizeRuntimeFiles([{ path: "index.js", sha256: digest("a") }, { path: "js/src/index.js", sha256: digest("b") }]), /duplicate runtime asset path/);
});

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "kevala-profile-report-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const registrySource = 'pub const KERNELS: &[&str] = &["matmul", "reduce"];\n';
  const files = { "js/src/index.js": "export {};\n", "crates/kevala/src/gpu.rs": registrySource };
  for (const [path, value] of Object.entries(files)) { await mkdir(dirname(join(root, path)), { recursive: true }); await writeFile(join(root, path), value); }
  const model = models();
  model.models[0].id = "laya";
  model.protocol.models = ["laya"];
  model.models[0].raw = ["runs/model.json"];
  const kernel = kernels();
  kernel.source.registrySha256 = hash(registrySource);
  kernel.runs = [{ id: "isolated", source: "runs/kernel.json" }];
  kernel.kernels[0].evidence = [{ harness: "isolated", source: "runs/kernel.json", status: "passed" }];
  const bytes = Buffer.from('{"status":"done","samples":[1,2,3]}\n');
  await mkdir(join(root, "capture/runs"), { recursive: true });
  await writeFile(join(root, "capture/runs/model.json"), bytes);
  await writeFile(join(root, "capture/runs/kernel.json"), bytes);
  const paths = { root, output: "docs/campaign/summary.json", modelPaths: ["capture/models.json"], kernelPaths: ["capture/kernels.json"] };
  const persist = async () => { await writeFile(join(root, "capture/models.json"), JSON.stringify(model)); await writeFile(join(root, "capture/kernels.json"), JSON.stringify(kernel)); };
  await persist();
  return { ...paths, model, kernel, bytes, persist };
}

test("public evidence archives are deterministic, content hashed, linked, and preserve exact source JSON", async (t) => {
  const input = await fixture(t);
  const report = await writeProfileReport(input);
  const first = await readFile(join(input.root, input.output));
  const files = await readdir(join(input.root, "docs/campaign/raw"));
  const compressed = await Promise.all(files.map((path) => readFile(join(input.root, "docs/campaign/raw", path))));
  const model = report.models.find((row) => row.id === "laya");
  const archived = await readFile(join(input.root, model.evidence[0].url));
  assert.deepEqual(gunzipSync(archived), input.bytes);
  assert.equal(hash(archived), model.evidence[0].gzipSha256);
  const indexArtifact = report.artifacts.find((item) => item.label === "Raw evidence index");
  const index = JSON.parse(gunzipSync(await readFile(join(input.root, indexArtifact.url))));
  assert.equal(index.files.length, 2);
  assert.deepEqual(gunzipSync(await readFile(join(input.root, index.sources.find((source) => source.label.startsWith("Model")).url))), await readFile(join(input.root, "capture/models.json")));
  assert.ok(report.kernels.rows[0].browsers[0].evidence[0].artifact.url.startsWith("docs/campaign/raw/"));
  await writeProfileReport(input);
  assert.deepEqual(await readFile(join(input.root, input.output)), first);
  assert.deepEqual(await Promise.all(files.map((path) => readFile(join(input.root, "docs/campaign/raw", path)))), compressed);
});

test("public report preserves complete request stages and rejects partial stage samples", async (t) => {
  const input = await fixture(t);
  input.model.models[0].cases[0].requestStages = requestStagesFixture();
  await input.persist();
  const report = await writeProfileReport(input);
  const requestStages = report.models.find((row) => row.id === "laya").cases[0].requestStages;
  assert.deepEqual(requestStages, input.model.models[0].cases[0].requestStages);
  assert.ok(requestStages.wall.blocks[0].samples[0].postForwardResidualMs < 0);

  const malformed = await fixture(t);
  malformed.model.models[0].cases[0].requestStages = requestStagesFixture();
  delete malformed.model.models[0].cases[0].requestStages.wall.blocks[0].samples[0].reportedForwardMs;
  await malformed.persist();
  await assert.rejects(writeProfileReport(malformed), /invalid request stage value reportedForwardMs/);
});

test("public report passes through bounded Clef head traces and rejects incomplete ordering", async (t) => {
  const input = await fixture(t);
  input.model.models[0].id = "clef-flash";
  input.model.protocol.models = ["clef-flash"];
  await input.persist();
  const oldReport = assembleProfileReport({ modelSummaries: [input.model], kernelSummaries: [input.kernel], registry });
  assert.equal(oldReport.models.find((row) => row.id === "clef-flash").cases[0].clefHead, undefined);

  input.model.models[0].cases[0].clefHead = clefHeadFixture();
  await input.persist();
  const report = await writeProfileReport(input);
  const clefHead = report.models.find((row) => row.id === "clef-flash").cases[0].clefHead;
  assert.deepEqual(clefHead, input.model.models[0].cases[0].clefHead);
  assert.ok(clefHead.attribution.includes("persistent head diagnostics"));
  assert.equal(clefHead.wall[0].afterWarmup.selected, "gpu");
  assert.equal(clefHead.wall[0].samples[0].requests[0].coordTile, 0);
  assert.equal(clefHead.wall[0].samples[0].requests[0].reason, "hidden-or-lexical-shape");
  assert.equal(clefHead.wall[0].samples[0].requests[0].headDiagnosticsTile, 0);
  assert.equal(clefHead.gpuProfiler[0].samples[0].requests[0].headDiagnosticsTile, 0);

  const malformed = await fixture(t);
  malformed.model.models[0].id = "clef-flash";
  malformed.model.protocol.models = ["clef-flash"];
  malformed.model.models[0].cases[0].clefHead = clefHeadFixture();
  malformed.model.models[0].cases[0].clefHead.wall[0].samples[0].requests[1].requestIndex = 0;
  await malformed.persist();
  await assert.rejects(writeProfileReport(malformed), /Clef head request order differs/);

  const wrongMode = await fixture(t);
  wrongMode.model.models[0].id = "clef-flash";
  wrongMode.model.protocol.models = ["clef-flash"];
  wrongMode.model.models[0].cases[0].clefHead = clefHeadFixture();
  wrongMode.model.models[0].cases[0].clefHead.gpuProfiler[0].samples[0].requests[0].requested = "cpu";
  await wrongMode.persist();
  await assert.rejects(writeProfileReport(wrongMode), /requested mode changed after warmup/);

  const wrongArch = await fixture(t);
  wrongArch.model.models[0].cases[0].clefHead = clefHeadFixture();
  await wrongArch.persist();
  await assert.rejects(writeProfileReport(wrongArch), /only valid for the Clef architecture/);
});

test("public kernel evidence stays bounded while exact measurements, isolated samples and model timings remain available", async (t) => {
  const input = await fixture(t), records = Array.from({ length: 4 }, (_, index) => ({ path: `result.timings.${index}`, rawSamples: true,
    values: { candidateMedianMs: 0.01 }, samples: { candidate: Array(5000).fill(0.01) } }));
  input.kernel.kernels[0].evidence[0].measurements = records;
  const original = structuredClone(input.kernel);
  const preview = assembleProfileReport({ modelSummaries: [input.model], kernelSummaries: [input.kernel], registry });
  assert.deepEqual(input.kernel, original);
  assert.equal(preview.kernels.rows[0].browsers[0].evidence[0].measurementRecords, 4);
  await input.persist();
  const source = await readFile(join(input.root, "capture/kernels.json")), report = await writeProfileReport(input);
  const row = report.kernels.rows[0].browsers[0];
  assert.equal(row.evidence[0].measurementRecords, 4);
  assert.equal(row.evidence[0].measurements, undefined);
  assert.equal(row.status, "measured");
  assert.deepEqual(row.isolated, original.kernels[0].isolated);
  assert.deepEqual(report.models.find((model) => model.id === "laya").cases[0].wall.samples, input.model.models[0].cases[0].wall.samples);
  assert.deepEqual(report.models.find((model) => model.id === "laya").cases[0].gpu.samples, input.model.models[0].cases[0].gpu.samples);
  assert.equal(report.models.find((model) => model.id === "laya").cases[0].requestStages, undefined);
  const archive = report.artifacts.find((item) => item.label.startsWith("Kernel capture"));
  assert.deepEqual(gunzipSync(await readFile(join(input.root, archive.url))), source);
  assert.deepEqual(JSON.parse(source).kernels[0].evidence[0].measurements, records);
  assert.deepEqual(gunzipSync(await readFile(join(input.root, row.evidence[0].artifact.url))), input.bytes);
  assert.ok(Buffer.byteLength(JSON.stringify(report)) < source.length / 4);
});

test("raw evidence cannot traverse or symlink outside its own capture directory", async (t) => {
  const input = await fixture(t);
  input.model.models[0].raw = ["../outside.json"];
  await input.persist();
  await assert.rejects(writeProfileReport(input), /relative JSON file inside its capture directory/);
  await writeFile(join(input.root, "outside.json"), "{}");
  await symlink(join(input.root, "outside.json"), join(input.root, "capture/runs/linked.json"));
  input.model.models[0].raw = ["runs/linked.json"];
  await input.persist();
  await assert.rejects(writeProfileReport(input), /raw evidence leaves its capture directory/);
  assert.deepEqual(await readdir(join(input.root, "docs")).catch(() => []), []);
});

async function derivedFixture(t) {
  const input = await fixture(t), row = input.model.models[0];
  input.model.status = "failed";
  row.status = "failed";
  row.reason = "GPU device lost: out of memory";
  row.reasonCode = "measurement-failed";
  row.raw = [];
  delete row.cases;
  await input.persist();
  const original = await readFile(join(input.root, "capture/models.json"));
  await writeFile(join(input.root, "capture/summary.json"), original);
  input.model.status = "partial";
  row.status = "blocked";
  row.reasonCode = "gpu-memory-limit";
  row.raw = ["runs/model.json"];
  await writeFile(join(input.root, "capture/runs/model.json"), JSON.stringify({ done: true, status: "error", error: row.reason,
    runner: { browser: "firefox", browserRequested: "firefox", browserVersion: "123" } }));
  input.model.validatedSource = { path: "summary.json", sha256: hash(original) };
  input.model.corrections = [{ id: row.id, precision: row.precision, browser: row.browser, previousStatus: "failed", status: "blocked", reasonCode: row.reasonCode, reason: row.reason }];
  input.model.validation = { validatedAt: "2026-11-06T03:00:00.000Z", method: "Raw failures inspected; measured rows independently revalidated." };
  await input.persist();
  return { ...input, original };
}

test("validated manifests archive unchanged originals and openly retain failed-to-blocked corrections", async (t) => {
  const input = await derivedFixture(t);
  const report = await writeProfileReport(input);
  assert.equal(report.models.find((row) => row.id === "laya").status, "blocked");
  assert.equal(report.models.find((row) => row.id === "laya").cases, undefined);
  assert.equal(report.status, "partial");
  assert.ok(report.limitations.some((reason) => reason.includes("laya/q8/firefox changed from failed to blocked (gpu-memory-limit)")));
  const provenance = report.provenance[0];
  assert.deepEqual(provenance.corrections, input.model.corrections);
  assert.equal(provenance.validation.validatedAt, "2026-11-06T03:00:00.000Z");
  assert.deepEqual(gunzipSync(await readFile(join(input.root, provenance.original.url))), input.original);
  assert.deepEqual(await readFile(join(input.root, "capture/summary.json")), input.original);
  assert.ok(report.artifacts.some((artifact) => artifact.label.startsWith("Original model capture")));
});

test("validated manifests reject altered source hashes, capture evidence and unlisted reclassifications", async (t) => {
  const input = await derivedFixture(t);
  const correctHash = input.model.validatedSource.sha256;
  input.model.validatedSource.sha256 = digest("2");
  await input.persist();
  await assert.rejects(writeProfileReport(input), /validatedSource SHA-256 mismatch/);
  input.model.validatedSource.sha256 = correctHash;
  input.model.models[0].pack.bytes++;
  await input.persist();
  await assert.rejects(writeProfileReport(input), /changed captured row evidence laya\/q8\/firefox/);
  input.model.models[0].pack.bytes--;
  input.model.corrections = [];
  await input.persist();
  await assert.rejects(writeProfileReport(input), /changed captured row evidence laya\/q8\/firefox/);
  assert.deepEqual(await readdir(join(input.root, "docs")).catch(() => []), []);
});

test("resource corrections cannot rename a numerical failure and attach successful output as a waiver", async (t) => {
  const input = await derivedFixture(t), original = JSON.parse(input.original);
  original.models[0].reason = "Profile changed an answer probability by 0.02, exceeding 0.0005";
  const bytes = Buffer.from(JSON.stringify(original));
  await writeFile(join(input.root, "capture/summary.json"), bytes);
  input.model.validatedSource.sha256 = hash(bytes);
  input.model.models[0].reason = input.model.corrections[0].reason = "GPUOutOfMemoryError";
  await writeFile(join(input.root, "capture/runs/model.json"), input.bytes);
  await input.persist();
  await assert.rejects(writeProfileReport(input), /original browser failure classification/);
  assert.deepEqual(await readdir(join(input.root, "docs")).catch(() => []), []);
});

test("resource corrections require an unchanged error and a distinct failed page with matching browser identity", async (t) => {
  const input = await derivedFixture(t), path = join(input.root, "capture/runs/model.json"), errorBytes = await readFile(path);
  await writeFile(path, input.bytes);
  await assert.rejects(writeProfileReport(input), /completed failed browser page/);
  await writeFile(path, JSON.stringify({ ...JSON.parse(errorBytes), cases: [{ samples: [{ wallMs: 12 }] }] }));
  await assert.rejects(writeProfileReport(input), /without successful cases/);
  const errorPage = JSON.parse(errorBytes);
  errorPage.runner.browserRequested = "chromium";
  await writeFile(path, JSON.stringify(errorPage));
  await assert.rejects(writeProfileReport(input), /browser/i);
  await writeFile(path, errorBytes);
  input.model.models[0].reason = input.model.corrections[0].reason = "GPUOutOfMemoryError";
  await input.persist();
  await assert.rejects(writeProfileReport(input), /original browser failure classification/);
  input.model.models[0].reason = input.model.corrections[0].reason = JSON.parse(input.original).models[0].reason;
  input.model.models[0].raw.push("runs/duplicate-error.json");
  await writeFile(join(input.root, "capture/runs/duplicate-error.json"), errorBytes);
  await input.persist();
  await assert.rejects(writeProfileReport(input), /duplicate raw failure evidence/);
});

async function attributionFixture(t) {
  const input = await fixture(t), id = "gemma-4-e4b", spec = MODELS[id];
  const row = { id, name: id, arch: spec.arch, precision: "q8", browser: "chromium", status: "failed", reasonCode: "measurement-failed",
    reason: "Kernel labels changed within a request shape", cacheMode: "uncached",
    pack: { bytes: 42, sha256: digest("d"), responseModel: id, answerUnit: 0.0001 }, raw: [] };
  input.model.status = "failed";
  input.model.models = [row];
  input.model.protocol = { ...protocol, models: [id], browsers: ["chromium"], probabilityTolerance: 0.0005 };
  const runs = [];
  for (const block of [0, 1]) for (const profile of block % 2 ? [true, false] : [false, true]) {
    const samples = [0, 1].map((index) => ({ tokens: 10 + index, requestHash: hash(`request-${index}`), wallMs: 12 + index,
      outputs: [{ model: id, answers: { q: { type: "noul", noul: 0.7 } }, raw_probabilities: { q: { false: 0.3, true: 0.7 } } }],
      ...(profile ? { gpu: { projection: 1 + index * 0.1, ...(block === 0 ? { "mm.gate.up.gelu": 0.2 } : {}) } } : {}) }));
    const result = { done: true, backend: "webgpu", model: id, arch: spec.arch, precision: "q8", modelInfo: { name: id, source: `https://huggingface.co/${spec.repo}`, revision: spec.revision },
      runtime: input.model.runtime.entry, runs: 2, warmups: 1, profile, unique: true, context: "page", cache: { packEnabled: false, state: { enabled: false } }, pack: { cached: false, bytes: 42 },
      adapter: measuredModel().adapter, runner: { browser: "chrome", browserRequested: "chromium", browserVersion: "123" }, gpuTuning: { method: "timestamp-query", truncated: false, points: [] },
      cases: [{ name: SHAPES[0], tokens: 11, samples, wallMs: samples.map((sample) => sample.wallMs) }] };
    const path = `runs/${block}-${profile ? "gpu" : "wall"}.json`;
    await writeFile(join(input.root, "capture", path), JSON.stringify(result));
    row.raw.push(path);
    runs.push({ block, profile, result });
  }
  await input.persist();
  const original = await readFile(join(input.root, "capture/models.json"));
  await writeFile(join(input.root, "capture/summary.json"), original);
  const calculated = summarizeProfileRuns(runs, { shapes: [0], probabilityTolerance: 0.0005, answerUnit: row.pack.answerUnit });
  Object.assign(row, calculated, { status: "measured", reasonCode: "heterogeneous-kernel-attribution", reason: "Valid raw blocks rederived with per-request heterogeneous kernel contributions.",
    adapter: runs[0].result.adapter, runner: runs[0].result.runner, tuning: runs.map((run) => ({ block: run.block, profile: run.profile, gpuTuning: run.result.gpuTuning })) });
  input.model.status = "complete";
  input.model.validatedSource = { path: "summary.json", sha256: hash(original) };
  input.model.corrections = [{ id, precision: row.precision, browser: row.browser, previousStatus: "failed", status: row.status, reasonCode: row.reasonCode, reason: row.reason }];
  await input.persist();
  return { ...input, original, row };
}

test("attribution-only corrections are revalidated from every raw block and retain dispatch presence metadata", async (t) => {
  const input = await attributionFixture(t), report = await writeProfileReport(input);
  const row = report.models.find((item) => item.id === "gemma-4-e4b" && item.browser === "chromium");
  assert.equal(row.status, "measured");
  assert.ok(row.cases[0].gpu.kernelAttribution.includes("not dispatched"));
  const kernel = row.cases[0].gpu.kernels.find((item) => item.name === "mm.gate.up.gelu");
  assert.equal(kernel.dispatchedSamples, 2);
  assert.equal(kernel.totalSamples, 4);
  assert.equal(kernel.medianMs, 0.1);
  assert.ok(report.limitations.some((item) => item.includes("Timings were rederived after raw numerical and protocol validation")));
  assert.deepEqual(gunzipSync(await readFile(join(input.root, report.provenance[0].original.url))), input.original);
});

test("attribution correction cannot promote a numerical failure or supply fabricated measurements", async (t) => {
  const input = await attributionFixture(t);
  input.row.cases[0].wall.medianMs++;
  await input.persist();
  await assert.rejects(writeProfileReport(input), /differs from recalculated raw cases/);
  input.row.cases[0].wall.medianMs--;
  const rawPath = join(input.root, "capture/runs/1-gpu.json"), rawBytes = await readFile(rawPath), result = JSON.parse(rawBytes);
  result.cases[0].samples[0].outputs[0].raw_probabilities.q = { false: 0.1, true: 0.9 };
  await writeFile(rawPath, JSON.stringify(result));
  await input.persist();
  await assert.rejects(writeProfileReport(input), /Profile changed an answer probability/);
  await writeFile(rawPath, rawBytes);
  const original = JSON.parse(input.original);
  original.models[0].reason = "Profile changed an answer probability by 0.2";
  const changed = Buffer.from(JSON.stringify(original));
  await writeFile(join(input.root, "capture/summary.json"), changed);
  input.model.validatedSource.sha256 = hash(changed);
  await input.persist();
  await assert.rejects(writeProfileReport(input), /requires the exact original label-change failure/);
  assert.deepEqual(await readdir(join(input.root, "docs")).catch(() => []), []);
});

test("attribution correction accepts the Chrome product alias but rejects a mismatched browser product or selector", async (t) => {
  const input = await attributionFixture(t), path = join(input.root, "capture/runs/1-wall.json"), raw = JSON.parse(await readFile(path));
  for (const runner of [{ browser: "firefox", browserRequested: "chromium" }, { browser: "chrome", browserRequested: "firefox" }]) {
    await writeFile(path, JSON.stringify({ ...raw, runner: { ...raw.runner, ...runner } }));
    await assert.rejects(writeProfileReport(input), /browser/i);
  }
  assert.deepEqual(await readdir(join(input.root, "docs")).catch(() => []), []);
});

async function kernelCapabilityFixture(t) {
  const input = await fixture(t), reason = "wide baseline requires shader-f16: wide baseline requires shader-f16";
  const compiled = { done: true, status: "done", backend: "webgpu", capabilities: { shaderF16: false }, correctness: { ok: true, registeredKernels: registry.length,
    compiled: registry.map((kernel) => ({ kernel, spec: {} })), skipped: [], failures: [] } };
  const rawFailure = { status: "failed", reason, pageResult: { status: "error", error: "wide baseline requires shader-f16", correctness: { ok: false, cases: [] } } };
  const runs = [{ id: "compile", source: "runs/compile.json", status: "passed", result: compiled, adapter: null, runner: null },
    { id: "paired-projections-wide", source: "runs/paired-projections-wide.json", status: "failed", reason, adapter: null, runner: null }];
  await writeFile(join(input.root, "capture/runs/compile.json"), JSON.stringify(compiled));
  await writeFile(join(input.root, "capture/runs/paired-projections-wide.json"), JSON.stringify(rawFailure));
  Object.assign(input.kernel, kernelCoverage(registry, runs), { status: "failed", browser: "chromium", capabilities: { shaderF16: false }, runs: runs.map(({ result, ...run }) => run) });
  await input.persist();
  const sourcePath = join(input.root, "capture/kernels.json"), original = await readFile(sourcePath);
  const derived = await deriveKernelCapabilityManifest({ sourcePath, registry, validation: { method: "Recorded capability and exact raw error checked; passed fixtures revalidated." } });
  await writeFile(join(input.root, "capture/kernels-validated.json"), JSON.stringify(derived));
  return { ...input, sourcePath, original, derived, kernelPaths: ["capture/kernels-validated.json"] };
}

test("kernel capability derivation archives the original and removes only the known unsupported run's failure contribution", async (t) => {
  const input = await kernelCapabilityFixture(t), report = await writeProfileReport(input);
  assert.equal(input.derived.runs.find((run) => run.id === "paired-projections-wide").status, "skipped");
  assert.deepEqual(input.derived.source, input.kernel.source);
  assert.equal(input.derived.counts.failed, 0);
  const reduce = report.kernels.rows.find((row) => row.name === "reduce").browsers.find((row) => row.browser === "chromium");
  assert.equal(reduce.status, "compile-only");
  assert.equal(reduce.numericalStatus, "not-run");
  assert.equal(reduce.timingStatus, "not-run");
  assert.ok(report.limitations.some((item) => item.includes("paired-projections-wide changed from failed to skipped")));
  const provenance = report.provenance.find((item) => item.kind === "validated-kernel-manifest");
  assert.deepEqual(gunzipSync(await readFile(join(input.root, provenance.original.url))), input.original);
  assert.deepEqual(await readFile(input.sourcePath), input.original);
});

test("kernel capability derivation rejects edited coverage, passed failures, different errors and present capabilities", async (t) => {
  const input = await kernelCapabilityFixture(t), path = join(input.root, "capture/kernels-validated.json");
  input.derived.counts.numericallyValidated++;
  await writeFile(path, JSON.stringify(input.derived));
  await assert.rejects(writeProfileReport(input), /differs from revalidated raw coverage/);
  const rawPath = join(input.root, "capture/runs/compile.json"), raw = JSON.parse(await readFile(rawPath));
  raw.correctness.failures.push({ kernel: "matmul", error: "numerical failure" });
  await writeFile(rawPath, JSON.stringify(raw));
  await assert.rejects(deriveKernelCapabilityManifest({ sourcePath: input.sourcePath, registry }), /incomplete kernel compilation ledger/);
  raw.correctness.failures = [];
  await writeFile(rawPath, JSON.stringify(raw));
  const original = JSON.parse(input.original);
  original.capabilities.shaderF16 = true;
  await writeFile(input.sourcePath, JSON.stringify(original));
  await assert.rejects(deriveKernelCapabilityManifest({ sourcePath: input.sourcePath, registry }), /without shaderF16/);
  original.capabilities.shaderF16 = false;
  original.runs.find((run) => run.id === "paired-projections-wide").reason = "Numerical mismatch";
  await writeFile(input.sourcePath, JSON.stringify(original));
  await assert.rejects(deriveKernelCapabilityManifest({ sourcePath: input.sourcePath, registry }), /sole exact wide-baseline capability failure/);
});

test("additional evidence archives exact JSON without affecting rows, coverage or performance claims", async (t) => {
  const input = await fixture(t), baseline = await writeProfileReport(input);
  const bytes = Buffer.from('{ "label": "Gemma control evidence", "status": "improved", "models": [{"id":"invented","status":"measured"}] }\n');
  await writeFile(join(input.root, "capture/control.json"), bytes);
  await writeFile(join(input.root, "capture/parity.json"), "[1, 2, 3]\n");
  const report = await writeProfileReport({ ...input, evidencePaths: ["capture/control.json", "capture/parity.json"] });
  assert.deepEqual(report.models, baseline.models);
  assert.deepEqual(report.kernels, baseline.kernels);
  assert.deepEqual(report.improvements, baseline.improvements);
  assert.equal(report.status, baseline.status);
  const artifact = report.artifacts.find((item) => item.kind === "additional-evidence" && item.label === "Gemma control evidence");
  const compressed = await readFile(join(input.root, artifact.url));
  assert.deepEqual(gunzipSync(compressed), bytes);
  assert.equal(artifact.sha256, hash(bytes));
  assert.equal(artifact.bytes, bytes.length);
  assert.equal(artifact.gzipSha256, hash(compressed));
  assert.ok(report.artifacts.some((item) => item.kind === "additional-evidence" && item.label === "parity.json"));
  const first = await readFile(join(input.root, input.output));
  await writeProfileReport({ ...input, evidencePaths: ["capture/parity.json", "capture/control.json"] });
  assert.deepEqual(await readFile(join(input.root, input.output)), first);
});

test("CLI accepts repeated capture inputs and requires a public destination", () => {
  const options = reportOptions(["--models=one.json", "--models=two.json", "--kernels=k.json", "--evidence=control.json", "--evidence=parity.json", "--output=docs/report.json"]);
  assert.deepEqual(options.models, ["one.json", "two.json"]);
  assert.deepEqual(options.kernels, ["k.json"]);
  assert.deepEqual(options.evidence, ["control.json", "parity.json"]);
  assert.throws(() => reportOptions(["--models=one.json"]), /supply --output/);
});
