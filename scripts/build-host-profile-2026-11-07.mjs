#!/usr/bin/env node
import { readFile, open, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { KERNEL_HARNESSES, kernelCoverage, parseKernelRegistry, validateKernelResult } from "./lib/kernel-profile-catalog.mjs";
import { stableJson, writeProfileReport } from "./lib/profile-report.mjs";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const CAPTURE_ROOT = process.env.KEVALA_PROFILE_CAPTURE_ROOT || "/tmp/kevala-profile-4e27753-20261107/tmp";
const CAMPAIGN_ROOT = process.env.KEVALA_PROFILE_CAMPAIGN_ROOT
  || "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn";
const PUBLICATION_ROOT = process.env.KEVALA_PROFILE_PUBLICATION_ROOT
  || "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/current-profile-publication";
const q8 = resolve(CAPTURE_ROOT, "profile-ci73-all11-q8-chromium/summary.json");
const bf16 = resolve(CAPTURE_ROOT, "profile-ci73-bf16-available-chromium/summary.json");
const kernels = resolve(CAPTURE_ROOT, "kernel-ci73-chromium/summary.json");
const recheckDir = resolve(CAPTURE_ROOT, "kernel-ci73-chromium-core-reference-recheck");
const recheckPath = resolve(recheckDir, "summary.json");
const recheckEvidencePath = resolve(PUBLICATION_ROOT, "core-reference-recheck-evidence-v1.json");
const visionEvidencePath = resolve(PUBLICATION_ROOT, "gemma4-vision-encoder-evidence-v1.json");
const imageEvidencePath = resolve(PUBLICATION_ROOT, "gemma4-image-parity-evidence-v1.json");
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function pinnedFile(path, bytes, expectedSha256, label) {
  const contents = await readFile(path);
  if (contents.length !== bytes || sha256(contents) !== expectedSha256) throw new Error(`${label} differs from the independently reviewed bytes`);
  return contents;
}

async function createOrVerifyImmutable(path, contents) {
  await mkdir(dirname(path), { recursive: true });
  let handle;
  try {
    handle = await open(path, "wx");
    await handle.writeFile(contents);
    await handle.sync();
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const existing = await readFile(path);
    if (!existing.equals(contents)) throw new Error(`${path} already exists with different bytes`);
  } finally {
    await handle?.close();
  }
}

const initialKernelBytes = await readFile(kernels);
const initialKernel = JSON.parse(initialKernelBytes);
const recheckBytes = await pinnedFile(recheckPath, 69821,
  "e6cf094ea18a6ff82d6635fe5208b6a74c5d42b07e524fff3d139096cf36856a", "recheck summary");
const recheck = JSON.parse(recheckBytes);
if (recheck.status !== "partial" || recheck.executionComplete !== true || recheck.sourceChanged !== false
  || !Array.isArray(recheck.runs) || recheck.runs.length !== 2
  || !["compile", "core-reference"].every((id) => recheck.runs.some((run) => run.id === id && run.status === "passed"))) {
  throw new Error("controlled core-reference recheck does not match the expected separate successful capture");
}
const expectedRawFiles = {
  "runs/compile.json": { bytes: 78215, sha256: "514fff23f56467a6b4d6742525630bab16acc455617b4f91ad00d401f13e988b" },
  "runs/core-reference.json": { bytes: 901, sha256: "cc00ed9f18a87cede0fd348d1da8d9730dbc69563747c983bcc65bbe79fc8902" },
};
const initialSource = initialKernel.source;
const recheckSource = recheck.source;
for (const field of ["commit", "workingTreeDirty", "registrySha256", "fingerprint"]) {
  if (initialSource[field] !== recheckSource[field]) throw new Error(`recheck ${field} differs from the initial kernel capture`);
}
if (stableJson(initialSource.files) !== stableJson(recheckSource.files)
  || initialSource.baseline?.sha256 !== recheckSource.baseline?.sha256
  || recheckSource.harnesses?.fingerprint !== "9fa1bb0cb50eb7a0de61a72b22cd0b8ebc62daafc2680abf8d90fdd1f399f56c"
  || recheckSource.harnesses?.files?.length !== 14) {
  throw new Error("recheck runtime or harness identity differs from the reviewed capture");
}
const initialHarnessFiles = new Map(initialSource.harnesses.files.map((file) => [file.path, stableJson(file)]));
if (recheckSource.harnesses.files.some((file) => initialHarnessFiles.get(file.path) !== stableJson(file))) {
  throw new Error("recheck harness files are not an exact subset of the initial capture");
}
const registryBytes = await readFile(resolve(ROOT, "crates/kevala/src/gpu.rs"));
if (sha256(registryBytes) !== recheckSource.registrySha256) throw new Error("current kernel registry differs from both reviewed captures");
const registry = parseKernelRegistry(registryBytes.toString("utf8"));
const recheckRuns = await Promise.all(recheck.runs.map(async (run) => {
  const descriptor = expectedRawFiles[run.source];
  if (!descriptor || run.status !== "passed") throw new Error(`unexpected recheck raw run ${run.id}`);
  const bytes = await pinnedFile(resolve(recheckDir, run.source), descriptor.bytes, descriptor.sha256, run.source);
  const result = JSON.parse(bytes);
  const harness = KERNEL_HARNESSES.find((item) => item.id === run.id);
  if (!harness) throw new Error(`unregistered recheck harness ${run.id}`);
  validateKernelResult(harness, result, registry);
  return { ...run, result, file: { path: run.source, bytes: bytes.length, sha256: sha256(bytes) } };
}));
if (stableJson(recheckRuns.map(({ id, source }) => ({ id, source }))) !== stableJson([
  { id: "compile", source: "runs/compile.json" }, { id: "core-reference", source: "runs/core-reference.json" },
])) throw new Error("recheck does not contain exactly the reviewed compile and core-reference runs");
const coverage = kernelCoverage(registry, recheckRuns);
const coreValidated = coverage.kernels.filter((row) => row.numericalStatus === "passed").map((row) => row.name);
const compileResult = recheckRuns.find((run) => run.id === "compile").result;
const coreResult = recheckRuns.find((run) => run.id === "core-reference").result;
const coreFields = ["matmul_k96", "matmul_k1024", "norm", "attention_w0", "attention_w64", "geglu", "rope"];
const recheckKernelCapture = {
  path: kernels,
  bytes: initialKernelBytes.length,
  sha256: sha256(initialKernelBytes),
  sourceIdentity: {
    commit: initialSource.commit,
    runtimeFingerprint: initialSource.fingerprint,
    registrySha256: initialSource.registrySha256,
    runtimeFiles: initialSource.files,
    initialHarnessFingerprint: initialSource.harnesses.fingerprint,
    recheckHarnessFingerprint: recheckSource.harnesses.fingerprint,
    recheckHarnessFiles: recheckSource.harnesses.files,
  },
};
const evidence = {
  schemaVersion: 1,
  label: "Separate controlled kernel startup recheck",
  note: "Compile and core-reference passed in a fresh browser run. This evidence does not alter the initial kernel capture status or its failed rows.",
  independentlyReviewedSummary: { path: recheckPath, bytes: recheckBytes.length, sha256: sha256(recheckBytes) },
  initialKernelCapture: recheckKernelCapture,
  identityChecks: { runtimeAndRegistryMatch: true, recheckHarnessFilesMatchInitialSubset: true },
  summary: recheck,
  runs: recheckRuns.map(({ result, ...run }) => ({ ...run, result })),
  validation: {
    registeredKernels: registry.length,
    compiledSpecializations: compileResult.correctness.compiled.length,
    skippedSpecializations: compileResult.correctness.skipped.length,
    compileFailures: compileResult.correctness.failures.length,
    coreReferenceKernels: coreValidated,
    coreReferenceKernelCount: coreValidated.length,
    maxCoreReferenceError: Math.max(...coreFields.map((field) => coreResult[field])),
    failedKernelRows: coverage.kernels.filter((row) => row.status === "failed").map((row) => row.name),
    timedKernelRows: coverage.kernels.filter((row) => row.timingStatus !== "not-run").map((row) => row.name),
  },
};
await createOrVerifyImmutable(recheckEvidencePath, Buffer.from(`${JSON.stringify(evidence, null, 2)}\n`));

const visionInputs = [
  { path: "vision-ci73-v1/before.json", bytes: 20300, sha256: "753486e510b7514b1ee16347b128864cce99a18e829f85c968994381b9fb8a33" },
  { path: "vision-ci73-v1/after.json", bytes: 20299, sha256: "3ff84c97de950fbf41ea06cc2340869fdf8132e369230304efd74dfc8aefc3cf" },
  { path: "vision-ci73-v1/encoder.json", bytes: 9093, sha256: "2710485325e3f0d64eb9d29c617b237217ce39b9755d87cee4f6e36a5e07960c" },
];
const imageInputs = [
  { path: "vision-image-ci73-v1/before.json", bytes: 20307, sha256: "5a119214747ea7d13e6fb301512b2248ecd704a5cdc7e85b65054818b67630b5" },
  { path: "vision-image-ci73-v1/after.json", bytes: 20305, sha256: "477b88941e34e574f59e374c1f09c91925fdfc2bc9e11bacf51bd5951141e9a3" },
  { path: "vision-image-ci73-v1/image.json", bytes: 142156, sha256: "6d48ea3b7622a8fec845f2ccc1f6c27597d190983c15038fada389bec32ffbe0" },
];
async function pinnedRecords(descriptors) {
  return Promise.all(descriptors.map(async (descriptor) => {
    const path = resolve(CAMPAIGN_ROOT, descriptor.path);
    const bytes = await pinnedFile(path, descriptor.bytes, descriptor.sha256, descriptor.path);
    return { path, bytes: bytes.length, sha256: sha256(bytes), record: JSON.parse(bytes) };
  }));
}
const [encoderRecords, imageRecords] = await Promise.all([pinnedRecords(visionInputs), pinnedRecords(imageInputs)]);
const encoderHelperPath = resolve(CAMPAIGN_ROOT, "vision-ci73-v1/inputs.mjs");
const imageHelperPath = resolve(CAMPAIGN_ROOT, "vision-image-ci73-v1/inputs.mjs");
const encoderHelper = await pinnedFile(encoderHelperPath, 4421,
  "d05c0498d2d015238fbce9bd64a3e2d8a5b66274903935409d3de24e42998b5d", "vision encoder input helper");
const imageHelper = await pinnedFile(imageHelperPath, 4421,
  "d05c0498d2d015238fbce9bd64a3e2d8a5b66274903935409d3de24e42998b5d", "image parity input helper");
if (!encoderHelper.equals(imageHelper)) throw new Error("vision input helper copies differ");
const helperSource = encoderHelper.toString("utf8");
const helperCopies = [
  { path: encoderHelperPath, bytes: encoderHelper.length, sha256: sha256(encoderHelper) },
  { path: imageHelperPath, bytes: imageHelper.length, sha256: sha256(imageHelper) },
];
const encoderEvidence = {
  schemaVersion: 1,
  label: "Separate Gemma 4 E2B vision encoder reference check",
  note: "This is encoder evidence, not a text latency row or natural-image quality test.",
  captures: encoderRecords,
  inputHelper: { sha256: sha256(encoderHelper), copies: helperCopies, source: helperSource },
  auditedFindings: {
    pack: "CI73 current runtime; fixture and encoder pack hashes are pinned in the before/after capture records",
    fixtureSha256: "ba35047fb3ecb45b8d975a7b8d302ef3a6fe9772af5095dc3af8727c42d1da59",
    reference: { implementation: "Transformers F32", commit: "2c4914fb939fe9de0d8e7a798af4684d552f18b4" },
    patches: 630,
    activePatches: 576,
    softTokens: 64,
    sampling: { warmups: 5, wallSamples: 5, diagnosticWarmups: 1, diagnosticSamples: 5 },
    wallMedianMs: 24.6,
    maxAbs: 0.000020653,
    maxAbsLimit: 0.0001,
    separateGpuTimestampMedianMs: 14.834,
    adapterFallbackProvenance: "The legacy Boolean-OR field does not prove an explicitly observed false value.",
  },
};
const imageEvidence = {
  schemaVersion: 1,
  label: "Separate Gemma 4 E2B generated-image and prefix parity check",
  note: "This evidence checks generated red/blue image questions and prefix reuse. It does not claim natural-image, face, expression, or other-model quality.",
  captures: imageRecords,
  inputHelper: { sha256: sha256(imageHelper), copies: helperCopies, source: helperSource },
  auditedFindings: {
    answers: { red: "red", blue: "blue" },
    prefixParityMaxDelta: [1.34e-14, 1.63e-15],
    prefixParityLimit: 0.0005,
    phaseDisposed: true,
    initialWallSamplesMs: [329.5, 66.1],
    wallLatencyEstimate: "Not reported; the two samples differ too much for a stable estimate.",
    separateProfiledRequestMs: 152.9,
    profiledRequestNote: "Prefix reuse is disabled for the profiled request; this is not paired with the wall samples.",
    adapterFallbackProvenance: "The legacy Boolean-OR field does not prove an explicitly observed false value.",
  },
};
await Promise.all([
  createOrVerifyImmutable(visionEvidencePath, Buffer.from(`${JSON.stringify(encoderEvidence, null, 2)}\n`)),
  createOrVerifyImmutable(imageEvidencePath, Buffer.from(`${JSON.stringify(imageEvidence, null, 2)}\n`)),
]);

const limitations = [
  "Measured on an RTX 5070 Ti with driver 595.71.05 and a Ryzen 9 9950X3D host in Chrome 156. These results describe this host and browser capture, not a universal hardware ranking.",
  "Model rows are uncached resident WebGPU requests for short, medium, and long single-question fixtures. Each row reports the actual input token counts. Loading, calibration, fixture hashing, and warmup requests are excluded.",
  "GPU timestamps come from separate instrumented requests. Per-kernel GPU shares describe GPU kernel medians and are not shares of end-to-end wall latency.",
  "BF16 coverage includes only local packs that passed the explicit 9 GB policy. Kev-9B BF16 was blocked because its pack is 15,902,057,920 bytes; this was not an observed allocation failure. Clef-flash BF16 was blocked because no local pack was available. No weights were downloaded.",
  "The initial kernel capture failed when requestDevice returned 'A valid external Instance reference no longer exists.' The error happened before core-reference math and maps to six kernel rows. A separate controlled compile/core-reference retry passed; both records are preserved, with no correction to the initial failure.",
  "The legacy browser fallback field can report false when both source properties are missing. The current profile records that provenance limit; false is not treated as a confirmed non-fallback adapter in those legacy rows.",
  "Numerical checks cover the recorded fixtures and establish output agreement only. They do not evaluate model accuracy. Q8 and BF16 were captured separately, so these runs do not establish a paired precision speedup.",
  "Some tuning summaries were truncated. The report does not establish that every tuning candidate was tested or that all kernels are optimal.",
];
const report = await writeProfileReport({
  root: ROOT,
  output: "docs/benchmarks/kernel-profile-2026-11-07/summary.json",
  modelPaths: [q8, bf16],
  kernelPaths: [kernels],
  evidencePaths: [recheckEvidencePath, visionEvidencePath, imageEvidencePath],
  limitations,
});
console.log(JSON.stringify({ status: report.status, measuredModels: report.models.filter((row) => row.status === "measured").length,
  modelRows: report.models.length, kernels: report.kernels.registered, output: "docs/benchmarks/kernel-profile-2026-11-07/summary.json" }));
