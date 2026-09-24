import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execFile = promisify(execFileCallback);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const reporter = resolve(root, "scripts/report-decisions.mjs");
const archivedResults = resolve(root, "benchmarks/results/decisions-linux-2026-09-22");
const committedSummary = resolve(archivedResults, "summary.json");
const referenceName = "gemma-base-vs-it.json";

async function runReporter(results, output, checkSummary = null, extraArgs = []) {
  const args = [
    reporter,
    "--results", results,
    "--output", output,
  ];
  if (!extraArgs.includes("--no-references")) args.push("--reference", resolve(results, referenceName));
  if (checkSummary) args.push("--check-summary", checkSummary);
  args.push(...extraArgs);
  try {
    const result = await execFile(process.execPath, args, { cwd: root, maxBuffer: 32 * 1024 * 1024 });
    return { code: 0, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    return { code: error.code ?? 1, stdout: error.stdout ?? "", stderr: error.stderr ?? "" };
  }
}

test("selected additional models require a measured result", async (t) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  t.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  await rm(resolve(results, "decision-bruv1-0.8b.json"));
  const result = await runReporter(results, output, null, ["--include-model", "bruv1-0.8b"]);
  assert.notEqual(result.code, 0);
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.missing.length, 1);
  assert.match(report.missing[0], /decision-bruv1-0\.8b\.json$/);
  assert.equal(report.models["bruv1-0.8b"].status, "missing");
});

async function copiedResults() {
  const directory = await mkdtemp(resolve(tmpdir(), "kevala-report-decisions-"));
  await cp(archivedResults, directory, { recursive: true });
  return directory;
}

async function temporaryOutput() {
  const directory = await mkdtemp(resolve(tmpdir(), "kevala-report-output-"));
  return resolve(directory, "summary.json");
}

async function readResult(directory, model) {
  const path = resolve(directory, `decision-${model}.json`);
  return { path, value: JSON.parse(await readFile(path, "utf8")) };
}

test("reporter accepts matching full-pack evidence and committed summary", async (t) => {
  const output = await temporaryOutput();
  t.after(async () => rm(dirname(output), { recursive: true, force: true }));
  const result = await runReporter(archivedResults, output, committedSummary);
  assert.equal(result.code, 0, result.stderr);
  const report = JSON.parse(await readFile(output, "utf8"));
  for (const [model, { metrics }] of Object.entries(report.models)) {
    for (const [dataset, metric] of Object.entries({ all: metrics, ...metrics.perDataset })) {
      assert.equal(metric.accuracy, metric.correct / metric.total, `${model}/${dataset} accuracy`);
      assert.equal(metric.groupBootstrap.estimate, metric.accuracy, `${model}/${dataset} bootstrap estimate`);
    }
  }
  const verification = report.models["kev-0.8b"].pack.verification;
  assert.equal(verification.status, "verified");
  assert.equal(verification.expectedCatalogHash, "8c4859f55d38b29bb5a024791b781f9ed978bd90f88ce9bce16219bd4461eca2");
  assert.equal(verification.recordedFullPackSha256, verification.expectedCatalogHash);
  assert.equal(verification.recordedFullPackBytes, 857259584);
  const raw = await readResult(archivedResults, "kev-0.8b");
  assert.equal(report.models["kev-0.8b"].latency.meanMs, raw.value.timingAllRows.reduce((sum, row) => sum + row.wallMs, 0) / raw.value.timingAllRows.length);
});

test("README generation checks computed scores and preserves unrelated text", async (context) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  const readme = resolve(dirname(output), "README.md");
  context.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const campaignFile = resolve(results, "campaign.json");
  const campaign = JSON.parse(await readFile(campaignFile, "utf8"));
  campaign.reportPath = "benchmarks/results/new-campaign/report.md";
  await writeFile(campaignFile, `${JSON.stringify(campaign)}\n`);
  const source = "Keep this introduction.\n<!-- decision-benchmark:start -->\nstale\n<!-- decision-benchmark:end -->\nKeep this ending.\n";
  await writeFile(readme, source);
  const before = await runReporter(results, output, null, ["--check-readme", readme]);
  assert.notEqual(before.code, 0);
  assert.match(before.stderr, /README decision table is stale/);
  const regenerated = await runReporter(results, output, null, ["--readme", readme]);
  assert.equal(regenerated.code, 0, regenerated.stderr);
  const actual = await readFile(readme, "utf8");
  assert.ok(actual.startsWith("Keep this introduction.\n"));
  assert.ok(actual.endsWith("\nKeep this ending.\n"));
  assert.match(actual, /89\.8% \(97\/108\)/);
  assert.match(actual, /\| `kev-0\.8b` .* \| 100\.1 \| 100\.3 \|/);
  assert.ok(actual.includes(`[the full report](${campaign.reportPath})`));
  const checked = await runReporter(results, output, null, ["--check-readme", readme]);
  assert.equal(checked.code, 0, checked.stderr);
});

test("README publication rejects mixed browser campaigns and leaves the file unchanged", async (context) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  const readme = resolve(dirname(output), "README.md");
  context.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const source = "<!-- decision-benchmark:start -->\noriginal\n<!-- decision-benchmark:end -->\n";
  await writeFile(readme, source);
  const result = await readResult(results, "kev-0.8b");
  result.value.runner.browserVersion = "different-version";
  await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
  const reported = await runReporter(results, output, null, ["--readme", readme]);
  assert.notEqual(reported.code, 0);
  assert.match(reported.stderr, /browser does not match campaign.json/);
  assert.equal(await readFile(readme, "utf8"), source);
});

test("repeated runs pool actual wall samples and retain independent source groups", async (context) => {
  const repeated = await copiedResults();
  const output = await temporaryOutput();
  context.after(async () => {
    await rm(repeated, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const original = JSON.parse(await readFile(committedSummary, "utf8"));
  for (const model of Object.keys(original.models)) {
    const result = await readResult(repeated, model);
    result.value.timingAllRows.forEach((row) => { row.wallMs *= 2; });
    await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
  }
  const reported = await runReporter(archivedResults, output, null, ["--no-references", "--repeat-results", repeated]);
  assert.equal(reported.code, 0, reported.stderr);
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.canonical.passes, 2);
  for (const [model, result] of Object.entries(report.models)) {
    assert.equal(result.metrics.total, 1728);
    assert.equal(result.metrics.correct, original.models[model].metrics.correct * 2);
    assert.equal(result.metrics.groupBootstrap.groups, 72);
    assert.equal(result.metrics.semanticPermutationFlip.groups, original.models[model].metrics.semanticPermutationFlip.groups * 2);
    assert.equal(result.metrics.semanticPermutationFlip.rate, original.models[model].metrics.semanticPermutationFlip.rate);
    assert.equal(result.metrics.perDataset["kevala-authored36"].groupBootstrap.groups, 36);
    assert.equal(result.latency.count, 1728);
    assert.ok(Math.abs(result.latency.meanMs - original.models[model].latency.meanMs * 1.5) < 1e-8);
    assert.equal(result.runs.length, 2);
  }
  assert.deepEqual(report.references, {});
  assert.deepEqual(report.comparisons, []);
});

test("README publication rejects a hardware caption that contradicts the raw environment", async (context) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  const readme = resolve(dirname(output), "README.md");
  context.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const source = "<!-- decision-benchmark:start -->\noriginal\n<!-- decision-benchmark:end -->\n";
  await writeFile(readme, source);
  const campaign = JSON.parse(await readFile(resolve(results, "campaign.json"), "utf8"));
  for (const model of Object.keys(JSON.parse(await readFile(committedSummary, "utf8")).models)) {
    const result = await readResult(results, model);
    result.value.environment = { hardware: { ...campaign.hardware, gpu: "A different GPU" } };
    await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
  }
  const reported = await runReporter(results, output, null, ["--readme", readme]);
  assert.notEqual(reported.code, 0);
  assert.match(reported.stderr, /hardware does not match campaign.json/);
  assert.equal(await readFile(readme, "utf8"), source);
});

test("copied artifacts cannot masquerade as independent repeated measurements", async (context) => {
  const repeated = await copiedResults();
  const output = await temporaryOutput();
  context.after(async () => {
    await rm(repeated, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const reported = await runReporter(archivedResults, output, null, ["--no-references", "--repeat-results", repeated]);
  assert.notEqual(reported.code, 0);
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.models.laya.status, "invalid");
  assert.match(report.models.laya.validation.issues.join("\n"), /duplicate artifact bytes/);
});

test("reporter rejects a wrong full-pack digest", async (t) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  t.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const result = await readResult(results, "kev-0.8b");
  result.value.postRunVerification.pack.sha256 = "0".repeat(64);
  await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
  const reportResult = await runReporter(results, output);
  assert.notEqual(reportResult.code, 0);
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.models["kev-0.8b"].pack.verification.status, "invalid");
  assert.match(report.models["kev-0.8b"].validation.issues.join("\n"), /full-pack sha256 mismatch/);
});

test("reporter rejects missing full-pack evidence", async (t) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  t.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const result = await readResult(results, "kev-0.8b");
  delete result.value.postRunVerification.pack.sha256;
  const original = result.value.postRunVerification.pack;
  await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
  const reportResult = await runReporter(results, output);
  assert.notEqual(reportResult.code, 0);
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.models["kev-0.8b"].pack.verification.recordedFullPackSha256, null);
  assert.match(report.models["kev-0.8b"].validation.issues.join("\n"), /full-pack sha256 is missing or invalid/);
  assert.equal(original.bytes, 857259584);
});

test("reporter rejects a corrupted full-pack byte size", async (t) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  t.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const result = await readResult(results, "kev-0.8b");
  result.value.postRunVerification.pack.bytes += 1;
  await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
  const reportResult = await runReporter(results, output);
  assert.notEqual(reportResult.code, 0);
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.models["kev-0.8b"].pack.verification.status, "invalid");
  assert.match(report.models["kev-0.8b"].validation.issues.join("\n"), /full-pack byte size mismatch/);
});

test("reporter rejects missing, coerced, and nonpositive wall timings", async (context) => {
  const results = await copiedResults();
  const output = await temporaryOutput();
  context.after(async () => {
    await rm(results, { recursive: true, force: true });
    await rm(dirname(output), { recursive: true, force: true });
  });
  const result = await readResult(results, "kev-0.8b");
  for (const value of [null, undefined, "100", 0, -1]) {
    result.value.timingAllRows[0].wallMs = value;
    await writeFile(result.path, `${JSON.stringify(result.value)}\n`);
    const reported = await runReporter(results, output);
    assert.notEqual(reported.code, 0, `wallMs=${JSON.stringify(value)} must fail validation`);
    const report = JSON.parse(await readFile(output, "utf8"));
    assert.equal(report.models["kev-0.8b"].latency, null);
    assert.match(report.models["kev-0.8b"].validation.issues.join("\n"), /wallMs must be a finite positive number/);
  }
});

test("summary check rejects altered metrics", async (t) => {
  const directory = await mkdtemp(resolve(tmpdir(), "kevala-report-summary-"));
  const output = resolve(directory, "generated.json");
  const alteredSummary = resolve(directory, "altered-summary.json");
  t.after(() => rm(directory, { recursive: true, force: true }));
  const summary = JSON.parse(await readFile(committedSummary, "utf8"));
  summary.models.laya.metrics.accuracy += 0.01;
  await writeFile(alteredSummary, `${JSON.stringify(summary)}\n`);
  const result = await runReporter(archivedResults, output, alteredSummary);
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /Decision summary mismatch/);
  assert.match(result.stderr, /models\.laya\.metrics\.accuracy/);
});

test("summary check ignores only generatedUtc", async (t) => {
  const directory = await mkdtemp(resolve(tmpdir(), "kevala-report-summary-"));
  const output = resolve(directory, "generated.json");
  const timestampOnlySummary = resolve(directory, "timestamp-only-summary.json");
  t.after(() => rm(directory, { recursive: true, force: true }));
  const summary = JSON.parse(await readFile(committedSummary, "utf8"));
  summary.generatedUtc = "2000-01-01T00:00:00.000Z";
  await writeFile(timestampOnlySummary, `${JSON.stringify(summary)}\n`);
  const result = await runReporter(archivedResults, output, timestampOnlySummary);
  assert.equal(result.code, 0, result.stderr);
});
