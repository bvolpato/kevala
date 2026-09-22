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

async function runReporter(results, output, checkSummary = null) {
  const args = [
    reporter,
    "--results", results,
    "--reference", resolve(results, referenceName),
    "--output", output,
  ];
  if (checkSummary) args.push("--check-summary", checkSummary);
  try {
    const result = await execFile(process.execPath, args, { cwd: root, maxBuffer: 32 * 1024 * 1024 });
    return { code: 0, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    return { code: error.code ?? 1, stdout: error.stdout ?? "", stderr: error.stderr ?? "" };
  }
}

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
  const verification = report.models["kev-0.8b"].pack.verification;
  assert.equal(verification.status, "verified");
  assert.equal(verification.expectedCatalogHash, "8c4859f55d38b29bb5a024791b781f9ed978bd90f88ce9bce16219bd4461eca2");
  assert.equal(verification.recordedFullPackSha256, verification.expectedCatalogHash);
  assert.equal(verification.recordedFullPackBytes, 857259584);
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
