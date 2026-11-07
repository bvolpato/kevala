import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createLeaseGuard } from "./fd-lease.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const canonicalLock = "/home/bruno/.cache/kevala/gpu-profile.lock";
const maxInputBytes = 1572864;
const maxProjectionBytes = 1048576;
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const args = process.argv.slice(2);
assert.equal(args.shift(), "--prepare", "use --prepare");
const roots = {};
while (args.length) {
  const flag = args.shift();
  assert(["--clef-root", "--campaign-root", "--f16-root"].includes(flag), `unknown argument ${flag}`);
  const value = args.shift();
  assert(value && !value.startsWith("--"), `missing ${flag} value`);
  const key = flag.slice(2, -5);
  roots[key] = resolve(value);
}
for (const key of ["clef", "campaign", "f16"]) {
  assert(roots[key], `missing --${key}-root`);
  const stat = await lstat(roots[key]);
  assert(stat.isDirectory() && !stat.isSymbolicLink(), `invalid ${key} root`);
  roots[key] = await realpath(roots[key]);
}

const specs = {
  clefResult: ["clef", "tmp/clef-route-20261107-r1/result.json", "2e64adef39e6a4cd03b772d01474343e3f44df04f5be8a3a4c2ef45dd98aa932"],
  clefController: ["clef", "tmp/clef-route-20261107-r1/controller.json", "a3323956bc7f4f9be293aac0d4ad292da87dbb55c91b7dc607aa1a8c86f6f188"],
  clefRelease: ["clef", "tmp/clef-route-20261107-r1/external-release-v1.json", "e4544ac91edd872317946707a3e73118589565d2db05797894b8c84292491399"],
  clefLease0: ["clef", "tmp/clef-route-20261107-r1/lease-000-clef-route-controller-bind-before-inputs.json"],
  clefLease1: ["clef", "tmp/clef-route-20261107-r1/lease-001-before-source-and-runtime-inputs.json"],
  clefLease2: ["clef", "tmp/clef-route-20261107-r1/lease-002-before-browser-load-of-runtime-wasm-and-model-pack.json"],
  clefLease3: ["clef", "tmp/clef-route-20261107-r1/lease-003-before-pack-identity-and-link.json"],
  clefLease4: ["clef", "tmp/clef-route-20261107-r1/lease-004-before-browser-controller.json"],
  clefLease5: ["clef", "tmp/clef-route-20261107-r1/lease-005-after-owned-process-server-and-link-cleanup-and-scope-proof.json"],
  clefPresReceipt: ["campaign", "clef-route-r1-source-preservation-v1/receipt.json", "aa1d9d49dd15d63eaea013862cfc587700fed468c5313b633b33d15c7c08e95a"],
  clefPresRelease: ["campaign", "clef-route-r1-source-preservation-v1/release.json", "803e98354d02d2c0071615744ebdd0ebecade993828dc82bad5e8a017b1cfe5f"],
  clefPresLease0: ["campaign", "clef-route-r1-source-preservation-v1/lease-0.json"],
  clefPresLease1: ["campaign", "clef-route-r1-source-preservation-v1/lease-1.json"],
  clefCorrectedReceipt: ["campaign", "clef-route-r1-corrected-validation-v1/receipt.json", "c2c22511db87d993e631f7ab8558ee5b4450ff4d4fda3304913064f74f4d5249"],
  clefCorrectedLease0: ["campaign", "clef-route-r1-corrected-validation-v1/lease-00-clef-r1-offline-validation-before-inputs.json"],
  clefCorrectedLease1: ["campaign", "clef-route-r1-corrected-validation-v1/lease-01-clef-r1-offline-validation-after-result-gates.json"],
  clefSmoke: ["campaign", "clef-route-static-browser-smoke-v1.json", "7952adafc6cfca19b98a5d816881eb0aeb38cf004fa76023319502da039efac0"],
  clefPreflight: ["campaign", "clef-route-source-preflight-v1/plan.json", "260e437231f8c795d7c0afefef03c3239ca52b76eaeb31700079eabf8d8b8378"],
  f16R1Result: ["f16", "tmp/capture-20261107-r1/baseline/result.json", "739e057deb7797296fd1917619b5fda61a5dfadc39365282767435c528ee9c3e"],
  f16R1Controller: ["f16", "tmp/capture-20261107-r1/controller.json", "2e6c90362229c8eecb7ce4a960034b4db4b7f957cace20f2016cde6acada3cda"],
  f16R1Release: ["f16", "tmp/capture-20261107-r1/external-release-v1.json", "42f5b20e752708d45e147a3557d5e1a0e941b9cc3b56d5ef8a5d9a6aecfd8e3f"],
  f16R1PresReceipt: ["campaign", "f16-feature-r1-source-preservation-v1/receipt.json", "34fe385005d3f7e86da51a5f9166ed63d153320aac61b908e25f1cb49ecad693"],
  f16R1PresRelease: ["campaign", "f16-feature-r1-source-preservation-v1/release.json", "2922320f940e4624148edfa0b30e362fe82795b45fb8208123c10d5b14cb1180"],
  f16R1PresLease0: ["campaign", "f16-feature-r1-source-preservation-v1/lease-0.json"],
  f16R1PresLease1: ["campaign", "f16-feature-r1-source-preservation-v1/lease-1.json"],
  f16R1Lease0: ["f16", "tmp/capture-20261107-r1/lease-000-feature-controller-bind-inherited-fd.json"],
  f16R1Lease1: ["f16", "tmp/capture-20261107-r1/lease-001-before-browser-binary-identity-read.json"],
  f16R1Lease2: ["f16", "tmp/capture-20261107-r1/lease-002-before-baseline-browser.json"],
  f16R1Lease3: ["f16", "tmp/capture-20261107-r1/lease-003-after-owned-processes-stdio-and-server-closed.json"],
  f16R2Baseline: ["f16", "tmp/capture-20261107-r2/baseline/result.json", "bae48d2bade445bb96d67731ceceace0896da0a5c25a4f0264906408ce87d557"],
  f16R2Toggle: ["f16", "tmp/capture-20261107-r2/f16-toggle/result.json", "f9da9f0c2fb9f3cc11cacd0a668b4a500fe147b185c788ad88d4454c194539b7"],
  f16R2Controller: ["f16", "tmp/capture-20261107-r2/controller.json", "fdefaddf6b8cf0f572a26ec2b14b2efbc17d1f9b2fae282186b5a803ff8636fc"],
  f16R2Release: ["f16", "tmp/capture-20261107-r2/external-release-v1.json", "8362a6ee20e1faa05e6c813e33a3c9eb68e4e3316573ef6b177a3901612a5f4d"],
  f16R2Lease0: ["f16", "tmp/capture-20261107-r2/lease-000-feature-controller-bind-inherited-fd.json"],
  f16R2Lease1: ["f16", "tmp/capture-20261107-r2/lease-001-before-browser-binary-identity-read.json"],
  f16R2Lease2: ["f16", "tmp/capture-20261107-r2/lease-002-before-baseline-browser.json"],
  f16R2Lease3: ["f16", "tmp/capture-20261107-r2/lease-003-after-baseline-children-and-stdio-closed.json"],
  f16R2Lease4: ["f16", "tmp/capture-20261107-r2/lease-004-before-f16-toggle-browser.json"],
  f16R2Lease5: ["f16", "tmp/capture-20261107-r2/lease-005-after-f16-toggle-children-and-stdio-closed.json"],
  f16R2Lease6: ["f16", "tmp/capture-20261107-r2/lease-006-after-owned-processes-stdio-and-server-closed.json"],
  f16R2Plan: ["campaign", "f16-feature-source-preflight-r2-v1/parsed-plan.json", "70c31b39e324bd0b024394f96a01abedd7ee2d14e3b20fe335a92b69f5cf6150"],
  f16R2Preflight: ["campaign", "f16-feature-source-preflight-r2-v1/parsed-source-preflight.json", "bb641498cc88cb894fb6ad83b6fb9ef4b77be1dc1d34fd92e1f4986e18b3a63b"],
  driverReceipt: ["campaign", "host-capability-metadata-20261107-v1/receipt.json", "0f231beb6b8f9b17b90fdc10a41cd581f4f7633529e7bdae052e8dfd01d61d9c"],
  driverRelease: ["campaign", "host-capability-metadata-20261107-v1/release.json", "d5f2e5f195115f7d6fb1eb8bce79b2125178679c85f1ad186a9ba5a0edd2d7b3"],
  driverLease0: ["campaign", "host-capability-metadata-20261107-v1/lease-0.json"],
  driverLease1: ["campaign", "host-capability-metadata-20261107-v1/lease-1.json"],
};

const inputs = {};
async function readInput(id) {
  if (inputs[id]) return inputs[id];
  const spec = specs[id];
  assert(spec, `unknown input ${id}`);
  const [rootName, relative, expectedSha256] = spec;
  const path = resolve(roots[rootName], relative);
  const originalStat = await lstat(path);
  assert(originalStat.isFile() && !originalStat.isSymbolicLink(), `invalid input path: ${id}`);
  const canonical = await realpath(path);
  assert(canonical.startsWith(roots[rootName] + sep), `input escaped ${rootName} root: ${relative}`);
  const stat = await lstat(canonical);
  assert(stat.isFile() && !stat.isSymbolicLink() && stat.size <= maxInputBytes, `invalid or oversized input: ${id}`);
  const bytes = await readFile(canonical);
  assert(bytes.length <= maxInputBytes);
  const sha256 = hash(bytes);
  if (expectedSha256) assert.equal(sha256, expectedSha256, `${id} source hash changed`);
  const input = { id, root: rootName, path: relative, bytes: bytes.length, sha256 };
  try { input.value = JSON.parse(bytes); } catch {}
  inputs[id] = input;
  return input;
}
const value = async (id) => (await readInput(id)).value;
const pick = (object, fields) => Object.fromEntries(fields.map((field) => [field, object[field]]));

const leaseSummaries = [];
const leaseGuard = createLeaseGuard({ canonicalPath: canonicalLock, recordEvidence: async (evidence) => {
  const binding = evidence.binding || evidence.proof?.binding;
  const observation = evidence.observation || {};
  const before = observation.before || {};
  const after = observation.after || {};
  const proof = evidence.proof || {};
  leaseSummaries.push({
    label: evidence.label,
    passed: evidence.passed === true,
    exactInheritedFdOwnershipProved: proof.exactInheritedFdOwnershipProved === true,
    ownerStableBeforeAfter: Boolean(before.process && after.process && before.process.pid === after.process.pid
      && before.process.startTicks === after.process.startTicks && before.process.pid === binding?.pid),
    descriptorStableBeforeAfter: Boolean(before.fd && after.fd && before.fd.fd === after.fd.fd
      && before.fd.device === after.fd.device && before.fd.inode === after.fd.inode
      && before.fdTarget === after.fdTarget && before.fdTarget === binding?.canonicalPath),
    canonicalDevice: String(binding?.device ?? before.canonical?.device ?? ""),
    canonicalInode: String(binding?.inode ?? before.canonical?.inode ?? ""),
    fdNumber: binding?.fd ?? before.fd?.fd ?? null,
    primaryFdLockRowsBefore: proof.primaryFdinfoRowsBefore?.length ?? 0,
    primaryFdLockRowsAfter: proof.primaryFdinfoRowsAfter?.length ?? 0,
  });
  return null;
} });
const lockBefore = await leaseGuard.bind("clef-evidence-preparation-admission");
assert.equal(lockBefore.exactInheritedFdOwnershipProved, true);

const ids = Object.keys(specs);
for (const id of ids) await readInput(id);
const data = Object.fromEntries(Object.entries(inputs).map(([id, item]) => [id, item.value]));
for (const id of ids) assert(data[id], `input ${id} must be JSON`);

async function readSource(rootName, relative, expectedBytes, expectedSha256) {
  const path = resolve(roots[rootName], relative);
  const originalStat = await lstat(path);
  assert(originalStat.isFile() && !originalStat.isSymbolicLink(), `invalid source path: ${relative}`);
  const canonical = await realpath(path);
  assert(canonical.startsWith(roots[rootName] + sep), `source escaped ${rootName} root: ${relative}`);
  const stat = await lstat(canonical);
  assert(stat.isFile() && !stat.isSymbolicLink() && stat.size <= maxInputBytes, `invalid or oversized source: ${relative}`);
  const bytes = await readFile(canonical);
  assert(bytes.length <= maxInputBytes);
  if (expectedBytes !== undefined) assert.equal(bytes.length, expectedBytes, relative);
  const sha256 = hash(bytes);
  if (expectedSha256) assert.equal(sha256, expectedSha256, relative);
  return { root: rootName, path: relative, bytes, sha256 };
}
const sourceCopies = [];
const sourceCopyByPath = new Map();
async function copySource(source, archivePath) {
  const prior = sourceCopyByPath.get(archivePath);
  if (prior) {
    assert.equal(prior.sha256, source.sha256, `shared source mismatch: ${archivePath}`);
    return prior;
  }
  const target = join(directory, archivePath);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, source.bytes, { flag: "wx", mode: 0o644 });
  const copied = await readFile(target);
  assert(copied.equals(source.bytes), `copied source differs: ${archivePath}`);
  const record = { archivePath, sourceRoot: source.root, sourcePath: source.path, bytes: copied.length, sha256: hash(copied) };
  sourceCopies.push(record);
  sourceCopyByPath.set(archivePath, record);
  return record;
}

const clefPreservation = data.clefPresReceipt;
assert.equal(clefPreservation.status, "passed");
assert.equal(clefPreservation.all61SourcePinsMatch, true);
for (const copy of clefPreservation.copies) {
  const source = await readSource("campaign", `clef-route-r1-source-preservation-v1/source/${copy.path}`, copy.bytes, copy.sha256);
  await copySource(source, `source/clef-route-r1-original/${copy.path}`);
}
const correctedValidator = data.clefCorrectedReceipt.validator;
const correctedSource = await readSource("clef", "scripts/private/clef-route-diagnostic.mjs", correctedValidator.bytes, correctedValidator.sha256);
await copySource(correctedSource, "source/clef-route-r1-corrected-offline/clef-route-diagnostic.mjs");

const f16R1Preservation = data.f16R1PresReceipt;
for (const copy of f16R1Preservation.sources || []) {
  const source = await readSource("campaign", `f16-feature-r1-source-preservation-v1/source/${copy.path}`, copy.bytes, copy.sha256);
  const name = basename(copy.path);
  const archivePath = name === "f16-feature-probe.py"
    ? "source/f16-feature-common/f16-feature-probe.py"
    : `source/f16-feature-r1/${name}`;
  await copySource(source, archivePath);
}
const f16SourceIdentity = data.f16R2Plan.sourceIdentity;
for (const [relative, pin] of Object.entries(f16SourceIdentity)) {
  const source = await readSource("f16", relative, pin.bytes, pin.sha256);
  const name = basename(relative);
  const archivePath = name === "f16-feature-probe.py"
    ? "source/f16-feature-common/f16-feature-probe.py"
    : `source/f16-feature-r2/${name}`;
  await copySource(source, archivePath);
}

const outputFiles = [];
async function storeProjection(archivePath, projection, sourceIds, selectedFields, transformation = "field-whitelisted-json") {
  const bytes = Buffer.from(JSON.stringify(projection) + "\n");
  assert(bytes.length <= maxProjectionBytes, `projection exceeds cap: ${archivePath}`);
  const target = join(directory, archivePath);
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes, { flag: "wx", mode: 0o644 });
  const saved = await readFile(target);
  assert(saved.equals(bytes), `projection readback mismatch: ${archivePath}`);
  outputFiles.push({ archivePath, bytes: bytes.length, sha256: hash(bytes) });
  projections.push({ archivePath, sourceIds, selectedFields, transformation, bytes: bytes.length, sha256: hash(bytes) });
}
const projections = [];
const sourceSha = (id) => inputs[id].sha256;
const routeResult = data.clefResult;
await storeProjection("evidence/clef-route-r1/result.json", {
  sourceSha256: sourceSha("clefResult"),
  model: routeResult.model,
  modelInfo: routeResult.modelInfo,
  arch: routeResult.arch,
  backend: routeResult.backend,
  flavor: routeResult.flavor,
  selectedFlavor: routeResult.selectedFlavor,
  precision: routeResult.precision,
  adapter: routeResult.adapter,
  cache: routeResult.cache,
  pack: routeResult.pack,
  protocol: routeResult.protocol,
  warmups: routeResult.warmups,
  runs: routeResult.runs,
  unique: routeResult.unique,
  profile: routeResult.profile,
  clefHeadAfterWarmup: routeResult.clefHeadAfterWarmup,
  clefHeadInstrumentation: routeResult.clefHeadInstrumentation,
  cases: routeResult.cases,
}, ["clefResult"], ["model", "modelInfo", "arch", "backend", "flavor", "selectedFlavor", "precision", "adapter", "cache", "pack", "protocol", "warmups", "runs", "unique", "profile", "clefHeadAfterWarmup", "clefHeadInstrumentation", "cases (complete records: wall vectors, all samples/routes/hashes/timing maps, and cases[].last timing objects)"]);

function ownFdLease(id) {
  const raw = data[id];
  const binding = raw.binding || raw.proof?.binding;
  const observation = raw.observation || {};
  const before = observation.before || {};
  const after = observation.after || {};
  const proof = raw.proof || {};
  const exactRow = (row) => Boolean(row && binding && row.pid === binding.pid && row.kind === "FLOCK"
    && row.classification === "ADVISORY" && row.mode === "WRITE" && row.start === "0" && row.end === "EOF" && !row.blocked);
  const beforeRows = proof.primaryFdinfoRowsBefore || [];
  const afterRows = proof.primaryFdinfoRowsAfter || [];
  return {
    sourceSha256: sourceSha(id),
    sequence: raw.sequence,
    label: raw.label,
    at: raw.at,
    passed: raw.passed === true,
    exactInheritedFdOwnershipProved: proof.exactInheritedFdOwnershipProved === true,
    ownerStableBeforeAfter: Boolean(before.process && after.process && before.process.pid === after.process.pid
      && before.process.startTicks === after.process.startTicks && before.process.pid === binding?.pid),
    descriptorStableBeforeAfter: Boolean(before.fd && after.fd && before.fd.fd === after.fd.fd
      && before.fd.device === after.fd.device && before.fd.inode === after.fd.inode
      && before.fdTarget === after.fdTarget && before.fdTarget === binding?.canonicalPath),
    exactOwnFlockRowBefore: beforeRows.length === 1 && exactRow(beforeRows[0]),
    exactOwnFlockRowAfter: afterRows.length === 1 && exactRow(afterRows[0]),
    canonicalDevice: String(binding?.device ?? before.canonical?.device ?? ""),
    canonicalInode: String(binding?.inode ?? before.canonical?.inode ?? ""),
    fdNumber: binding?.fd ?? before.fd?.fd ?? null,
  };
}
function receiptLeaseSummary(proof) {
  const binding = proof?.binding || {};
  const before = proof?.primaryFdinfoRowsBefore || [];
  const after = proof?.primaryFdinfoRowsAfter || [];
  const exact = (row) => Boolean(row && row.kind === "FLOCK" && row.classification === "ADVISORY"
    && row.mode === "WRITE" && row.start === "0" && row.end === "EOF" && !row.blocked);
  return {
    exactInheritedFdOwnershipProved: proof?.exactInheritedFdOwnershipProved === true,
    exactOwnFlockRowsBefore: before.length === 1 && exact(before[0]),
    exactOwnFlockRowsAfter: after.length === 1 && exact(after[0]),
    canonicalDevice: String(binding.device ?? ""),
    canonicalInode: String(binding.inode ?? ""),
    fdNumber: binding.fd ?? null,
  };
}
function leaseRefs(receipt) {
  return (receipt?.leaseRefs || []).map((ref) => ({ file: basename(ref.path || "lease.json"), sha256: ref.sha256 || ref.hash || null }));
}
function releaseSummary(id, extra = {}) {
  const raw = data[id];
  const scopeText = typeof raw.scope === "string" ? raw.scope : raw.scope?.stdout || "";
  const processes = Array.isArray(raw.processes) ? raw.processes : [];
  return {
    sourceSha256: sourceSha(id),
    timestamp: raw.timestamp,
    controllerStatus: raw.controllerStatus ?? null,
    releaseVerified: raw.releaseVerified ?? raw.passed ?? false,
    knownProcessCount: processes.length,
    allKnownProcessesAbsent: processes.length > 0 && processes.every((entry) => entry.absent === true),
    scopeUnit: typeof raw.scope === "object" ? raw.scope.unit ?? null : null,
    scopeCollected: scopeText.includes("LoadState=not-found") && scopeText.includes("ActiveState=inactive") && scopeText.includes("SubState=dead"),
    lock: raw.lock ? { device: raw.lock.device, inode: raw.lock.inode, freeProbeExit: raw.lock.freeProbeExit } : null,
    runnerStdioClosed: raw.runnerStdioClosed ?? null,
    ...extra,
  };
}

const originalController = data.clefController;
await storeProjection("evidence/clef-route-r1/validation.json", {
  originalController: {
    sourceSha256: sourceSha("clefController"),
    status: originalController.status,
    error: originalController.error,
    stopReason: originalController.stopReason,
    runnerExit: originalController.runnerExit,
    result: originalController.result,
  },
  correctedOfflineValidation: {
    sourceSha256: sourceSha("clefCorrectedReceipt"),
    status: data.clefCorrectedReceipt.status,
    originalControllerFailureStatusPreserved: data.clefCorrectedReceipt.originalControllerFailureStatusPreserved,
    originalRawFilesChanged: data.clefCorrectedReceipt.originalRawFilesChanged,
    validator: pick(data.clefCorrectedReceipt.validator, ["bytes", "sha256", "mode", "exit", "stdoutBytes", "stdoutSha256", "summary"]),
    originalController: pick(data.clefCorrectedReceipt.originalController, ["bytes", "sha256", "status", "originalOuterError", "originalValidatorError"]),
    originalResult: pick(data.clefCorrectedReceipt.originalResult, ["bytes", "sha256", "done", "model", "backend", "sampleCount"]),
    independentAudit: pick(data.clefCorrectedReceipt.independentAudit, ["status", "sampleCount", "allRoutes", "flavor", "coordTile", "validated", "reason", "eventCountPerSample", "zeroErrorsDropsAndFallbacks", "fdReferencesVerified", "physicalRunFdReferences", "preservationFdReferences"]),
  },
}, ["clefController", "clefCorrectedReceipt"], ["controller.status/error/stopReason/runnerExit/result.status/result.error", "corrected receipt status/originalControllerFailureStatusPreserved/originalRawFilesChanged", "validator identity and offline summary", "original raw identities and failure text", "independent audit route/timing/FD summary"]);

const clefPresLeaseRows = ["clefPresLease0", "clefPresLease1"].map(ownFdLease);
const correctedLeaseRows = ["clefCorrectedLease0", "clefCorrectedLease1"].map(ownFdLease);
const physicalClefLeaseIds = ["clefLease0", "clefLease1", "clefLease2", "clefLease3", "clefLease4", "clefLease5"];
const clefPhysicalLeaseRows = physicalClefLeaseIds.map(ownFdLease);
await storeProjection("evidence/clef-route-r1/leases.json", {
  physicalRunChecks: clefPhysicalLeaseRows,
  sourcePreservationChecks: clefPresLeaseRows,
  correctedOfflineValidationChecks: correctedLeaseRows,
  note: "Selected own-FD summaries are not replayable raw proofs. Full descriptor inventories and global lock-table text are excluded.",
}, [...physicalClefLeaseIds, "clefPresLease0", "clefPresLease1", "clefCorrectedLease0", "clefCorrectedLease1"], ["sequence", "label", "timestamp", "sourceSha256", "passed", "exactInheritedFdOwnershipProved", "owner/descriptor stability booleans", "canonical device/inode", "exact own-FD lock-row booleans"]);

function preservationSummary(id, lease0Id, lease1Id) {
  const receipt = data[id];
  return {
    sourceSha256: sourceSha(id),
    status: receipt.status,
    note: receipt.note,
    all61SourcePinsMatch: receipt.all61SourcePinsMatch ?? null,
    copies: (receipt.sources || receipt.copies || []).map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 })),
    leaseRefs: leaseRefs(receipt),
    before: receiptLeaseSummary(receipt.before),
    after: receiptLeaseSummary(receipt.after),
    leaseProofSummaries: [ownFdLease(lease0Id), ownFdLease(lease1Id)],
  };
}
const clefRelease = releaseSummary("clefRelease", {
  originalControllerStatus: data.clefRelease.controllerStatus,
  originalControllerError: data.clefRelease.controllerError,
  runnerExit: data.clefRelease.runnerExit,
  rawDone: data.clefRelease.rawDone,
  removedOwnedLinks: Array.isArray(originalController.removedOwnedLinks) ? originalController.removedOwnedLinks.length : null,
  serverClosed: originalController.serverClosed ?? null,
});
const clefPresRelease = releaseSummary("clefPresRelease");
await storeProjection("evidence/clef-route-r1/source-release.json", {
  captureRelease: clefRelease,
  sourcePreservation: preservationSummary("clefPresReceipt", "clefPresLease0", "clefPresLease1"),
  sourcePreservationRelease: clefPresRelease,
}, ["clefRelease", "clefPresReceipt", "clefPresRelease", "clefPresLease0", "clefPresLease1"], ["release verified/controller status/known process count and absence summary/collected scope/canonical lock and free probe", "61-pin audit/copies/source lease refs/before-after own-FD summaries", "source preservation release summary"]);

const sourceIdentity = data.clefPreflight.sourceIdentity;
await storeProjection("evidence/clef-route-r1/source-identities.json", {
  sourceSha256: sourceSha("clefPreflight"),
  baseCommit: data.clefPreflight.baseCommit,
  sourceIdentity,
}, ["clefPreflight"], ["baseCommit", "sourceIdentity (all 61 path, byte-count, SHA-256 entries)"]);

const smoke = data.clefSmoke;
await storeProjection("evidence/clef-route-r1/static-browser-smoke.json", {
  sourceSha256: sourceSha("clefSmoke"),
  timestamp: smoke.timestamp,
  pageSha256: smoke.pageSha256,
  smokePageSha256: smoke.smokePageSha256,
  result: smoke.result,
  scope: smoke.scope,
}, ["clefSmoke"], ["timestamp", "pageSha256", "smokePageSha256", "result", "scope"]);

const f16R1Controller = data.f16R1Controller;
const f16R1LaneSummaries = (f16R1Controller.lanes || []).map((lane) => ({
  lane: lane.lane,
  exitCode: lane.exit?.code ?? null,
  exitSignal: lane.exit?.signal ?? null,
  result: lane.result ? { bytes: lane.result.bytes, sha256: lane.result.sha256, status: lane.result.status } : null,
}));
await storeProjection("evidence/f16-feature-r1/failure.json", {
  controller: {
    sourceSha256: sourceSha("f16R1Controller"),
    status: f16R1Controller.status,
    capabilityOnly: f16R1Controller.capabilityOnly,
    productionPromotion: f16R1Controller.productionPromotion,
    error: f16R1Controller.error,
    lanes: f16R1LaneSummaries,
  },
  baseline: {
    sourceSha256: sourceSha("f16R1Result"),
    status: data.f16R1Result.status,
    done: data.f16R1Result.done,
    lane: data.f16R1Result.lane,
    page: data.f16R1Result.page,
  },
  release: releaseSummary("f16R1Release", { controllerError: data.f16R1Release.controllerError, baselineError: data.f16R1Release.baselineError }),
}, ["f16R1Controller", "f16R1Result", "f16R1Release"], ["controller status/capability-only flags/error and bounded lane result identities", "baseline status/done/lane/complete page object", "release verification summary"]);

function f16Preservation(id, releaseId, lease0Id, lease1Id) {
  const receipt = data[id];
  return {
    sourceSha256: sourceSha(id),
    status: receipt.status,
    copies: (receipt.sources || receipt.copies || []).map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 })),
    leaseRefs: leaseRefs(receipt),
    before: receiptLeaseSummary(receipt.before),
    after: receiptLeaseSummary(receipt.after),
    leaseProofSummaries: [ownFdLease(lease0Id), ownFdLease(lease1Id)],
    release: releaseSummary(releaseId),
  };
}
await storeProjection("evidence/f16-feature-r1/source-release.json", f16Preservation("f16R1PresReceipt", "f16R1PresRelease", "f16R1PresLease0", "f16R1PresLease1"), ["f16R1PresReceipt", "f16R1PresRelease", "f16R1PresLease0", "f16R1PresLease1"], ["source preservation status/copies/source lease refs and own-FD proof summaries", "release summary"]);

const f16R2Controller = data.f16R2Controller;
const f16R2LeaseIds = ["f16R2Lease0", "f16R2Lease1", "f16R2Lease2", "f16R2Lease3", "f16R2Lease4", "f16R2Lease5", "f16R2Lease6"];
const f16R2LaneSummaries = (f16R2Controller.lanes || []).map((lane) => ({
  lane: lane.lane,
  exitCode: lane.exit?.code ?? null,
  exitSignal: lane.exit?.signal ?? null,
  result: lane.result ? { bytes: lane.result.bytes, sha256: lane.result.sha256, status: lane.result.status } : null,
  runner: pick((lane.lane === "baseline" ? data.f16R2Baseline : data.f16R2Toggle).runner, [
    "browserVersion", "chromiumArgs", "existingBrowserAttached", "sharedWrapperSha256", "sessionBusHelperSha256",
  ]),
}));
await storeProjection("evidence/f16-feature-r2/capabilities.json", {
  controller: {
    sourceSha256: sourceSha("f16R2Controller"),
    status: f16R2Controller.status,
    capabilityOnly: f16R2Controller.capabilityOnly,
    productionPromotion: f16R2Controller.productionPromotion,
    lanes: f16R2LaneSummaries,
    browserExecutable: pick(f16R2Controller.browserExecutable, ["bytes", "sha256", "version"]),
  },
  baseline: { sourceSha256: sourceSha("f16R2Baseline"), status: data.f16R2Baseline.status, done: data.f16R2Baseline.done, lane: data.f16R2Baseline.lane, page: data.f16R2Baseline.page },
  toggle: { sourceSha256: sourceSha("f16R2Toggle"), status: data.f16R2Toggle.status, done: data.f16R2Toggle.done, lane: data.f16R2Toggle.lane, page: data.f16R2Toggle.page },
  release: releaseSummary("f16R2Release"),
  sourcePlan: {
    sourceSha256: sourceSha("f16R2Plan"),
    capabilityOnly: data.f16R2Plan.capabilityOnly,
    productionPromotion: data.f16R2Plan.productionPromotion,
    browser: pick(data.f16R2Plan.browser, ["version", "sha256"]),
    sourceIdentity: data.f16R2Plan.sourceIdentity,
    sourcePreflightSha256: sourceSha("f16R2Preflight"),
  },
  leaseProofs: f16R2LeaseIds.map(ownFdLease),
}, ["f16R2Controller", "f16R2Baseline", "f16R2Toggle", "f16R2Release", "f16R2Plan", "f16R2Preflight", ...f16R2LeaseIds], ["controller status/capability-only/no-promotion flags, bounded lane result identities, and actual browser executable bytes/hash/version identity", "complete baseline and toggle page capability objects", "per-lane browser version/arguments/attachment flag and shared wrapper/session-bus helper hashes", "release summary", "Chrome version/hash pin and source identity", "all seven bounded own-FD lease summaries"]);

const driver = data.driverReceipt;
await storeProjection("evidence/host/driver-metadata.json", {
  sourceSha256: sourceSha("driverReceipt"),
  startedAt: driver.startedAt,
  scope: driver.scope,
  query: {
    command: driver.query.command,
    exit: driver.query.exit,
    stdout: driver.query.stdout,
    stderr: driver.query.stderr,
    error: driver.query.error,
  },
  release: releaseSummary("driverRelease"),
  leaseProofs: [ownFdLease("driverLease0"), ownFdLease("driverLease1")],
}, ["driverReceipt", "driverRelease", "driverLease0", "driverLease1"], ["read-only nvidia-smi command/exit/stdout/stderr", "bounded release summary", "two bounded own-FD proof summaries"]);

await mkdir(join(directory, "evidence"), { recursive: true });
const toolBytes = await readFile(fileURLToPath(import.meta.url));
const provenance = {
  schemaVersion: 1,
  baseCommit: "81a27608f24e0b2252318515e272387c48dec7d8",
  preparedAt: new Date().toISOString(),
  preparationLock: {
    canonicalPath: canonicalLock,
    device: lockBefore.binding.device,
    inode: lockBefore.binding.inode,
    fd: lockBefore.binding.fd,
    exactInheritedFdOwnershipProved: lockBefore.exactInheritedFdOwnershipProved,
    checks: leaseSummaries,
    rawGlobalLockTablesStored: false,
  },
  rawInputs: Object.values(inputs).map(({ id, root, path, bytes, sha256 }) => ({ id, root, path, bytes, sha256 })),
  sourceCopies,
  toolSources: [
    { path: "prepare-evidence.mjs", bytes: toolBytes.length, sha256: hash(toolBytes) },
    { path: "fd-lease.mjs", bytes: (await lstat(join(directory, "fd-lease.mjs"))).size, sha256: hash(await readFile(join(directory, "fd-lease.mjs"))) },
    { path: "lease-parse.mjs", bytes: (await lstat(join(directory, "lease-parse.mjs"))).size, sha256: hash(await readFile(join(directory, "lease-parse.mjs"))) },
  ],
  projections,
  sourceReferences: [{ kind: "immutable public repository snapshot", commit: "81a27608f24e0b2252318515e272387c48dec7d8", note: "Unchanged production modules are not copied into the evidence bundle." }],
  exclusions: ["model weights and pack payloads", "compiled WASM", "complete browser/controller captures", "full PID/TID inventories", "global /proc/locks text", "browser diagnostic logs", "Clef decision vectors, answer text, logits and raw probabilities, which are absent from the saved route capture"],
};
const provenanceBytes = Buffer.from(JSON.stringify(provenance, null, 2) + "\n");
assert(provenanceBytes.length <= maxProjectionBytes, "provenance exceeds one MiB");
const provenancePath = join(directory, "provenance.json");
await writeFile(provenancePath, provenanceBytes, { flag: "wx", mode: 0o644 });
const provenanceSaved = await readFile(provenancePath);
assert(provenanceSaved.equals(provenanceBytes));

const lockAfter = await leaseGuard.check("clef-evidence-preparation-complete");
assert.deepEqual(lockBefore.binding, lockAfter.binding);
assert.equal(lockAfter.exactInheritedFdOwnershipProved, true);
console.log(JSON.stringify({
  status: "prepared",
  rawInputs: Object.keys(inputs).length,
  projections: outputFiles.length,
  copiedSources: sourceCopies.length,
  projectedBytes: outputFiles.reduce((sum, item) => sum + item.bytes, provenanceSaved.length),
  provenanceBytes: provenanceSaved.length,
  ownFdProved: true,
  postCheck: true,
  canonicalDevice: lockAfter.binding.device,
  canonicalInode: lockAfter.binding.inode,
  files: [...outputFiles, { archivePath: "provenance.json", bytes: provenanceSaved.length, sha256: hash(provenanceSaved) }],
}, null, 2));
