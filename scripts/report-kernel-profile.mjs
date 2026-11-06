#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { writeProfileReport } from "./lib/profile-report.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const HELP = `Assemble a public report from terminal captures of the current runtime.

  node scripts/report-kernel-profile.mjs --models=tmp/models/summary.json --kernels=tmp/kernels/summary.json --output=docs/benchmarks/kernel-profile-2026-11-06/summary.json

Repeat --models and --kernels for additional browsers. Repeat --evidence for
extra JSON archives that do not affect reported rows or status. --improvements
and --limitations accept JSON arrays. Runtime bytes and measurement protocols must
agree; unavailable rows remain explicit. Raw JSON is copied as deterministic
gzip evidence under the report's sibling raw/ directory. No packs or logs are
published. Exit status 0 means assembly succeeded, including partial reports.
`;

export function reportOptions(args) {
  const { values } = parseArgs({ args, strict: true, allowPositionals: false, options: {
    models: { type: "string", multiple: true }, kernels: { type: "string", multiple: true }, evidence: { type: "string", multiple: true }, output: { type: "string" },
    improvements: { type: "string" }, limitations: { type: "string" }, help: { type: "boolean" },
  } });
  if (!values.help && (!values.output || !values.models?.length && !values.kernels?.length)) throw new Error("supply --output and at least one --models or --kernels summary");
  return values;
}

async function main() {
  const options = reportOptions(process.argv.slice(2));
  if (options.help) { console.log(HELP); return; }
  const json = async (path) => path ? JSON.parse(await readFile(resolve(ROOT, path), "utf8")) : [];
  const report = await writeProfileReport({ root: ROOT, output: options.output, modelPaths: options.models, kernelPaths: options.kernels,
    evidencePaths: options.evidence,
    improvements: await json(options.improvements), limitations: await json(options.limitations) });
  console.log(JSON.stringify({ status: report.status, models: report.models.length, kernels: report.kernels.registered, output: options.output }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
