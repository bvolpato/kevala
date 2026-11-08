import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, realpath } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createLeaseGuard } from "./fd-lease.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const docsRoot = resolve(directory, "../../..");
const canonicalLock = "/home/bruno/.cache/kevala/gpu-profile.lock";
const maxInputBytes = 1572864;
const maxProjectionBytes = 1048576;
const immutableSourceCommit = "7a6de58b74d2ad9e452e8ca317d84f4b83abfe86";
const repositoryUrl = "https://github.com/bvolpato/kevala";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const jsonBytes = (value) => Buffer.from(JSON.stringify(value) + "\n");
const prettyBytes = (value) => Buffer.from(JSON.stringify(value, null, 2) + "\n");

assert.deepEqual(process.argv.slice(2), ["--prepare"], "use --prepare");

const roots = {
  clef: "/tmp/kevala-clef-route-diagnostic-v1",
  matrix: "/tmp/kevala-subgroup-matrix-probe",
  integer: "/tmp/kevala-integer-matrix-gate",
  campaign: "/home/bruno/githubworkspace/kevala",
};

const inputs = [
  { id: "clefResult", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/result.json", bytes: 1026534, sha256: "8bc21b0a6217f83f58e1e062a28dbd8438ad90fb1b16d884976faf0f02dc1c05" },
  { id: "clefController", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/controller.json", bytes: 49809, sha256: "c0f796de1abab95daa217357f5413e499d6537c7372bb4682df5063de4dc76c9" },
  { id: "clefRelease", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/external-release-observed-union-v2.json", bytes: 1684, sha256: "eed71a1e3f335db93a69368aa0d8fbbc25a6a2ae7f393dbec8bfb39d6346bc12" },
  { id: "clefLease000", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-000-clef-kernel-profile-controller-bind-before-inputs.json", bytes: 107022, sha256: "29882828cfb490f9f02dff233e4ba28633c1f890e0f59ca5fc5d0ab7563bd5fc" },
  { id: "clefLease001", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-001-before-source-and-runtime-inputs.json", bytes: 54661, sha256: "2e5c91be119834e5b28f0e48df2168a0a6670ec4e6432d4fe8e181db25fd2c56" },
  { id: "clefLease002", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-002-before-browser-load-of-runtime-wasm-and-model-pack.json", bytes: 54681, sha256: "1f03cac76fde8f960be227c5cf3e3aa6dea3b1fdcec26a5f1b83f9f351c1445a" },
  { id: "clefLease003", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-003-before-pack-identity-and-link.json", bytes: 54660, sha256: "59eaae9a87dff2f20523fcf19fc3e0edd7df7a3efe99ef72b5cfa1386bf5080b" },
  { id: "clefLease004", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-004-before-browser-controller.json", bytes: 54656, sha256: "c365df66cabe93a8dfa313658053e064de3e89078600e1cf91c24c2bacea5167" },
  { id: "clefLease005", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-005-after-browser-controller.json", bytes: 54602, sha256: "2062af11c09fe9c407d92b04ccc0bb9d37f734c931217a1568b4a17f0d62225f" },
  { id: "clefLease006", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-006-before-post-run-source-and-runtime-identity-check.json", bytes: 54627, sha256: "16e4751f0da4167352ad0398dd0bb0df03c2b7fb0cd2041929818f2df03311fe" },
  { id: "clefLease007", root: "clef", path: "tmp/clef-kernel-profile-20261107-r1/lease-007-after-owned-process-server-and-link-cleanup-and-scope-proof.json", bytes: 54637, sha256: "3788e0cf398e5ef6f047fae1d7493f66c34285f11b89b0088fca35b1fb557f08" },
  { id: "clefStaticSmoke", root: "campaign", path: "tmp/sota-20261106-kdDTGn/campaign/clef-kernel-profile-static-browser-smoke-v2.json", bytes: 1759, sha256: "e0fb065f5763afdef577cf10ed1bbef3e3040aefe0e516b539203feffc061f95" },
  { id: "matrixResult", root: "matrix", path: "tmp/capture-20261107-r1/baseline/result.json", bytes: 1050925, sha256: "b87b82d845546238c587eaeed0678ae16cafcb306e4325cefe33143d3229d84d" },
  { id: "matrixController", root: "matrix", path: "tmp/capture-20261107-r1/controller.json", bytes: 30392, sha256: "8fd1b42daa625e1f0538e66a5cf14661da82e795fab664730555596c3b6e56ff" },
  { id: "matrixRelease", root: "matrix", path: "tmp/capture-20261107-r1/external-release-observed-union-v2.json", bytes: 1577, sha256: "731fe31cd3860dee8b1c60c0c31c0d06b44c49208f6db429e19480de2598b178" },
  { id: "matrixLease000", root: "matrix", path: "tmp/capture-20261107-r1/lease-000-feature-controller-bind-inherited-fd.json", bytes: 112119, sha256: "82b95d84b52955149c82da941c3def141f8b79da33d4fb2088b6c70bfe897ca7" },
  { id: "matrixLease001", root: "matrix", path: "tmp/capture-20261107-r1/lease-001-before-browser-binary-identity-read.json", bytes: 57219, sha256: "3828e119dfadfec5a6f147fa12adec2eb39a7ace9e7216c38db6bea92a41a2c0" },
  { id: "matrixLease002", root: "matrix", path: "tmp/capture-20261107-r1/lease-002-before-baseline-browser.json", bytes: 57210, sha256: "503178a964f50952927fe6eb521dcb5bc9bf6c31bdfb399a7db2d0f0f7e71fd9" },
  { id: "matrixLease003", root: "matrix", path: "tmp/capture-20261107-r1/lease-003-after-baseline-children-and-stdio-closed.json", bytes: 57250, sha256: "2cd599b3c26c60b00bfbba69e5e178393826ae5e6c89bb4821d857d4b45652bf" },
  { id: "matrixLease004", root: "matrix", path: "tmp/capture-20261107-r1/lease-004-after-owned-processes-stdio-and-server-closed.json", bytes: 57256, sha256: "854f1c0904e793337f2384cd736071f75cb5e8177f3fa9248cc6ed11477e16aa" },
  { id: "integerResult", root: "integer", path: "tmp/capture-20261107-r1/baseline/result.json", bytes: 1090039, sha256: "f92d5e879be733ec3ec130ecd6ce1d4e4073d00c3f05c29c8de9e532b64932a0" },
  { id: "integerController", root: "integer", path: "tmp/capture-20261107-r1/controller.json", bytes: 32442, sha256: "6be2cf1d41d8de0d1fab77594738821a4648ec732636172e64ab5ce79274e809" },
  { id: "integerRelease", root: "integer", path: "tmp/capture-20261107-r1/external-release-observed-union-v1.json", bytes: 1801, sha256: "5e34768b806a347a51062494df03840ac5f71cb5da668f9917ce80bb1b4811bb" },
  { id: "integerLease000", root: "integer", path: "tmp/capture-20261107-r1/lease-000-feature-controller-bind-inherited-fd.json", bytes: 110362, sha256: "d6471a8c91a4326abe7fc114f5b946984610bf838b7c98b251e0e19f6b2d07e0" },
  { id: "integerLease001", root: "integer", path: "tmp/capture-20261107-r1/lease-001-before-browser-binary-identity-read.json", bytes: 56340, sha256: "88bc1275875592201b5f7dfa35e9edd434aa46375608ca72bf385696b8208fcc" },
  { id: "integerLease002", root: "integer", path: "tmp/capture-20261107-r1/lease-002-before-baseline-browser.json", bytes: 56255, sha256: "e83547c31e316884781884843c01b760871b8ac14046497c487c47fac8fd2dc4" },
  { id: "integerLease003", root: "integer", path: "tmp/capture-20261107-r1/lease-003-after-baseline-children-and-stdio-closed.json", bytes: 56290, sha256: "67de199db60b1a901564b4497a9925a532b454a95e8a0aa6e7d428222c97f049" },
  { id: "integerLease004", root: "integer", path: "tmp/capture-20261107-r1/lease-004-after-owned-processes-stdio-and-server-closed.json", bytes: 56372, sha256: "d96ad8c112db22795338456b9d7ccb5271ca9654cf425285840b6eea62a2f1a6" },
  { id: "integerStaticSmoke", root: "campaign", path: "tmp/sota-20261106-kdDTGn/campaign/integer-matrix-gate-static-browser-smoke-v1/browser.json", bytes: 1204, sha256: "1a1bed08abd1c61ede0f5fd4ac657fb941c8042085eec80d3b22e80a89aefcde" },
];

const sourceCopies = [
  { id: "clefProfileHtml", root: "clef", originPath: "bench-clef-kernel-profile.html", archivePath: "source/clef-kernel-profile-r1/bench-clef-kernel-profile.html", bytes: 20828, sha256: "928e79272dc69bc50def73cffb0368f4bb2d0a5b04dd7144db702902d2c42dc9" },
  { id: "clefProfileScript", root: "clef", originPath: "scripts/private/clef-kernel-profile.mjs", archivePath: "source/clef-kernel-profile-r1/clef-kernel-profile.mjs", bytes: 71752, sha256: "cd978379ddca009376344c39842b092aa4cd175abbdeda052c9abe8535d7f9a4" },
  { id: "matrixProbeHtml", root: "matrix", originPath: "scripts/private/subgroup-matrix-probe.html", archivePath: "source/subgroup-matrix-probe-r1/subgroup-matrix-probe.html", bytes: 8520, sha256: "24df77e2c8de8609f01021064a0b8b34655a6f3e62652955b9aa1e6479754d37" },
  { id: "matrixProbeScript", root: "matrix", originPath: "scripts/private/subgroup-matrix-probe.mjs", archivePath: "source/subgroup-matrix-probe-r1/subgroup-matrix-probe.mjs", bytes: 26476, sha256: "0f97b431f8f16e8afd9197eba0d9e2e8f14199376fe704240f8651b2ba9dbf80" },
  { id: "matrixProbeRunner", root: "matrix", originPath: "scripts/private/subgroup-matrix-probe.py", archivePath: "source/subgroup-matrix-probe-r1/subgroup-matrix-probe.py", bytes: 11242, sha256: "a109005fa7ffcd7b7a33e52ae224d73138bf31a4d0e468d1ce04da38431a0795" },
  { id: "matrixProbeTest", root: "matrix", originPath: "scripts/private/subgroup-matrix-probe.test.mjs", archivePath: "source/subgroup-matrix-probe-r1/subgroup-matrix-probe.test.mjs", bytes: 7373, sha256: "fca439c24ad4ad71b14a13614e4797a37df40cecc93df04e275d28cce7fad1f1" },
  { id: "integerGateHelper", root: "integer", originPath: "scripts/private/integer-matrix-gate-helper.js", archivePath: "source/integer-matrix-gate-r1/integer-matrix-gate-helper.js", bytes: 12993, sha256: "70e56d7d3ba2053fb46f84573f95647366e694f86950c3f3f1ee1bedc71460eb" },
  { id: "integerGateHtml", root: "integer", originPath: "scripts/private/integer-matrix-gate.html", archivePath: "source/integer-matrix-gate-r1/integer-matrix-gate.html", bytes: 15651, sha256: "aba79fa9a419448edec639fb53f8ed4979f887d5950cd1fb8d8c18caf652a18b" },
  { id: "integerGateScript", root: "integer", originPath: "scripts/private/integer-matrix-gate.mjs", archivePath: "source/integer-matrix-gate-r1/integer-matrix-gate.mjs", bytes: 28731, sha256: "0a54825e0d7cae52d206d8c9dabd6ada471e8120363d52e47badc050d653a70c" },
  { id: "integerGateRunner", root: "integer", originPath: "scripts/private/integer-matrix-gate.py", archivePath: "source/integer-matrix-gate-r1/integer-matrix-gate.py", bytes: 12229, sha256: "35081d3bc7e2ee29f7453214dafb1a101675f481dc46a7870d01f55f605e1b25" },
  { id: "integerGateTest", root: "integer", originPath: "scripts/private/integer-matrix-gate.test.mjs", archivePath: "source/integer-matrix-gate-r1/integer-matrix-gate.test.mjs", bytes: 9305, sha256: "4bbf7ce574a2c4e2b829cf121bf6b8960a7bf690c86ee8dbd1327bf56af4a873" },
];

const packageSourceByCapturePath = new Map(sourceCopies.map((source) => [source.originPath, source]));
const sharedArchiveSources = new Map([
  ["bench-clef-route.html", "docs/benchmarks/clef-route-2026-11-07/source/clef-route-r1-original/bench-clef-route.html"],
  ["bench-clef-route-smoke.html", "docs/benchmarks/clef-route-2026-11-07/source/clef-route-r1-original/bench-clef-route-smoke.html"],
  ["scripts/private/clef-route-browser.py", "docs/benchmarks/clef-route-2026-11-07/source/clef-route-r1-original/scripts/private/clef-route-browser.py"],
  ["scripts/private/q8-monitor-cadence.mjs", "docs/benchmarks/clef-route-2026-11-07/source/clef-route-r1-original/scripts/private/q8-monitor-cadence.mjs"],
]);
const hashOnlySourcePaths = new Map([
  ["bench.html", "No byte-identical public copy was found at the frozen merge commit."],
  ["scripts/private/clef-route-diagnostic.mjs", "The public corrected offline validator has a different SHA-256 from this capture's original source pin."],
]);

const leaseGuard = createLeaseGuard({ canonicalPath: canonicalLock, recordEvidence: async (value) => value });
const leaseBefore = await leaseGuard.bind("clef-gpu-profile-evidence-preparation");
assert.equal(leaseBefore.exactInheritedFdOwnershipProved, true);

const rootPaths = {};
for (const [name, path] of Object.entries(roots)) rootPaths[name] = await realpath(path);

const inputRecords = new Map();
async function readInput(id) {
  const cached = inputRecords.get(id);
  if (cached) return cached;
  const spec = inputs.find((entry) => entry.id === id);
  assert(spec, `unknown raw input: ${id}`);
  const path = resolve(rootPaths[spec.root], spec.path);
  const original = await lstat(path);
  assert(original.isFile() && !original.isSymbolicLink() && original.size <= maxInputBytes, `invalid or oversized raw input: ${id}`);
  const canonical = await realpath(path);
  assert(canonical.startsWith(rootPaths[spec.root] + sep), `raw input escaped its capture root: ${id}`);
  const info = await lstat(canonical);
  assert(info.isFile() && !info.isSymbolicLink());
  const bytes = await readFile(canonical);
  assert.equal(bytes.length, spec.bytes, `${id} byte count changed`);
  assert.equal(hash(bytes), spec.sha256, `${id} SHA-256 changed`);
  const value = JSON.parse(bytes);
  const record = { spec, bytes, value };
  inputRecords.set(id, record);
  return record;
}

const copyRecords = new Map();
async function readSourceCopy(id) {
  const cached = copyRecords.get(id);
  if (cached) return cached;
  const spec = sourceCopies.find((entry) => entry.id === id);
  assert(spec, `unknown source copy: ${id}`);
  const path = resolve(directory, spec.archivePath);
  const original = await lstat(path);
  assert(original.isFile() && !original.isSymbolicLink() && original.size <= maxInputBytes, `invalid or oversized source copy: ${id}`);
  const canonical = await realpath(path);
  assert(canonical.startsWith(directory + sep), `source copy escaped package: ${id}`);
  const bytes = await readFile(canonical);
  assert.equal(bytes.length, spec.bytes, `${id} byte count changed`);
  assert.equal(hash(bytes), spec.sha256, `${id} SHA-256 changed`);
  copyRecords.set(id, { spec, bytes });
  return copyRecords.get(id);
}

const value = async (id) => (await readInput(id)).value;
const rawPin = (id) => {
  const record = inputRecords.get(id);
  assert(record, `raw input not loaded: ${id}`);
  return { id, captureRoot: record.spec.root, path: record.spec.path, bytes: record.bytes.length, sha256: hash(record.bytes) };
};
const selected = (object, keys) => Object.fromEntries(keys.filter((key) => Object.hasOwn(object, key)).map((key) => [key, object[key]]));

const clefResult = await value("clefResult");
const clefController = await value("clefController");
const matrixResult = await value("matrixResult");
const matrixController = await value("matrixController");
const integerResult = await value("integerResult");
const integerController = await value("integerController");

assert.equal(clefController.status, "complete");
assert.deepEqual(clefController.runnerExit, { code: 0, signal: null });
assert.equal(clefController.request.profile, true);
assert.equal(clefController.request.requestedHead, "auto");
assert.equal(clefController.request.cache, false);
assert.equal(clefController.request.stateCache, false);
assert.equal(clefController.request.warmups, 3);
assert.equal(clefController.request.runs, 9);
assert.equal(clefResult.done, true);
assert.equal(clefResult.profile, true);
assert.equal(clefResult.warmups, 3);
assert.equal(clefResult.runs, 9);
assert.equal(clefResult.cases.length, 3);
assert.equal(String(clefController.pack.size), "10586832640");

const routeRecords = [];
const sampleCount = clefResult.cases.reduce((total, current) => total + current.samples.length, 0);
assert.equal(sampleCount, 27);
for (const item of clefResult.cases) {
  assert.equal(item.samples.length, 9, `${item.name} sample count`);
  assert.equal(item.wallMs.length, 9, `${item.name} wall vector length`);
  assert(item.last && typeof item.last === "object");
  const tokenValues = item.samples.map((sample) => sample.tokens);
  assert(tokenValues.every(Number.isInteger));
  const range = [Math.min(...tokenValues), Math.max(...tokenValues)];
  assert.deepEqual(range, item.name.includes("~40") ? [158, 159]
    : item.name.includes("~128") ? [251, 252] : [659, 660], `${item.name} token range`);
  for (const sample of item.samples) {
    assert(/^[a-f0-9]{64}$/.test(sample.requestHash), "missing request hash");
    assert(Array.isArray(sample.outputs) && sample.outputs.length === 1);
    assert.deepEqual(Object.keys(sample.outputs[0]).sort(), ["answers", "model", "usage"]);
    assert(Array.isArray(sample.clefHeadRequests) && sample.clefHeadRequests.length === 1);
    const route = sample.clefHeadRequests[0];
    assert.equal(route.requested, "auto");
    assert.equal(route.selected, "gpu");
    assert.equal(route.validated, true);
    assert.equal(route.reason, null);
    assert.equal(route.flavor, "relaxed");
    assert.equal(route.coordTile, 0);
    assert.equal(route.headDiagnosticsTile, 0);
    assert(sample.gpu && Object.keys(sample.gpu).length > 0);
    const labelValues = Object.values(sample.gpu);
    assert(labelValues.every((number) => Number.isFinite(number) && number >= 0));
    assert(labelValues.reduce((sum, number) => sum + number, 0) > 0);
    assert(sample.clefHeadTiming && typeof sample.clefHeadTiming === "object");
    routeRecords.push(route);
  }
}
assert.equal(routeRecords.length, 27);

assert.equal(matrixController.status, "complete");
assert.equal(matrixController.capabilityAndCompilationOnly, true);
assert.equal(matrixController.productionPromotion, false);
assert.equal(matrixResult.status, "complete");
assert.equal(matrixResult.done, true);
assert.equal(matrixResult.page.compilation.status, "skipped");
assert.equal(matrixResult.page.compilation.reason, "no-advertised-f32-f32-configuration");
assert.equal(matrixResult.page.adapter.subgroupMatrixConfigs.exposed, true);
assert.equal(matrixResult.page.adapter.subgroupMatrixConfigs.entries.length, 4);
assert.deepEqual(matrixResult.page.adapter.subgroupMatrixConfigs.entries.map(({ componentType, resultComponentType, M, N, K }) => [componentType, resultComponentType, M, N, K]), [
  ["u8", "u32", 16, 16, 32],
  ["i8", "i32", 16, 16, 32],
  ["u8", "u32", 16, 8, 32],
  ["i8", "i32", 16, 8, 32],
]);
assert(matrixResult.page.adapter.features.includes("chromium-experimental-subgroup-matrix"));
assert.equal(matrixResult.page.device.status, "created");
assert.equal(matrixResult.page.device.destroyed, true);
assert(matrixResult.page.device.features.includes("chromium-experimental-subgroup-matrix"));
assert.equal(matrixResult.page.deviceRequest.shaderF16Requested, false);
assert.equal(matrixResult.page.adapterRequests.highPerformanceReturnedNull, false);
assert.equal(matrixResult.page.adapterRequests.defaultFallbackRequested, false);

assert.equal(integerController.status, "complete");
assert.equal(integerController.gateOnly, true);
assert.equal(integerController.productionPromotion, false);
assert.equal(integerResult.status, "complete");
assert.equal(integerResult.done, true);
assert.equal(integerResult.runner.browserVersion, "156.0.8078.4");
assert.equal(integerResult.runner.chromiumArgs.length, 5);
assert.equal(integerResult.runner.existingBrowserAttached, false);
assert.equal(integerResult.page.deviceRequest.shaderF16Requested, false);
assert.equal(integerResult.page.adapterRequests.highPerformanceReturnedNull, false);
assert.equal(integerResult.page.adapterRequests.defaultFallbackRequested, false);
const gate = integerResult.page.gate;
assert.equal(gate.status, "passed");
assert.equal(gate.scope, "one-signed-i8-16x16x32-operation");
assert.equal(gate.selectedConfig.componentType, "i8");
assert.equal(gate.selectedConfig.resultComponentType, "i32");
assert.deepEqual([gate.selectedConfig.M, gate.selectedConfig.N, gate.selectedConfig.K], [16, 16, 32]);
assert.deepEqual([gate.packing.leftOffsetWords, gate.packing.rightOffsetWords, gate.packing.backingRowStrideBytes], [72, 88, 1024]);
assert.equal(gate.outputs.length, 256);
assert.equal(gate.reference.expected.length, 256);
assert.deepEqual(gate.outputs, gate.reference.expected);
assert.equal(gate.correctnessPassed, true);
assert.equal(gate.compared, 256);
assert.equal(gate.different, 0);
assert.deepEqual(gate.mismatches, []);
assert.equal(gate.positiveOutputCount, 106);
assert.equal(gate.outputs.filter((number) => number > 0).length, 106);
assert.equal(gate.outputs.filter((number) => number < 0).length, 150);
assert.equal(gate.outputs.filter((number) => number === 0).length, 0);
assert.equal(gate.canaries.compared, 128);
assert.equal(gate.canaries.changed, 0);
assert.equal(gate.dispatches, 1);
assert.equal(gate.bufferObjectsDestroyed, 4);
assert.equal(gate.compilation.status, "pipeline-created");
assert.equal(gate.compilation.diagnostics.totalMessages, 0);
assert.equal(gate.compilation.moduleValidationError, null);
assert.equal(gate.compilation.pipelineValidationError, null);
assert.equal(gate.executionErrors.validation, null);
assert.equal(gate.executionErrors.outOfMemory, null);
assert.equal(integerResult.page.device.destroyed, true);
const outputBytes = Buffer.alloc(gate.outputs.length * 4);
gate.outputs.forEach((number, index) => {
  assert(Number.isInteger(number) && number >= -2147483648 && number <= 2147483647);
  outputBytes.writeInt32LE(number, index * 4);
});
assert.equal(hash(outputBytes), "498117f912a5f1a4a3a2f6cd8bda7bacd741f38d389ea754e842b8fa463a58b3");

for (const source of sourceCopies) await readSourceCopy(source.id);
function ownFdLease(id) {
  const raw = inputRecords.get(id)?.value;
  assert(raw, `lease input not loaded: ${id}`);
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
    sourceSha256: rawPin(id).sha256,
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

async function inputRun(ids) {
  const records = [];
  for (const id of ids) records.push(await readInput(id));
  return records;
}
await inputRun(["clefRelease", "clefStaticSmoke", ...inputs.filter((item) => item.id.startsWith("clefLease")).map((item) => item.id),
  "matrixRelease", ...inputs.filter((item) => item.id.startsWith("matrixLease")).map((item) => item.id),
  "integerRelease", "integerStaticSmoke", ...inputs.filter((item) => item.id.startsWith("integerLease")).map((item) => item.id)]);

function releaseProjection(id) {
  const raw = inputRecords.get(id).value;
  const scopeText = raw.scope?.stdout || "";
  const processEntries = Array.isArray(raw.processes) ? raw.processes : [];
  const collected = ["LoadState=not-found", "ActiveState=inactive", "SubState=dead"].every((marker) => scopeText.includes(marker));
  assert.equal(raw.passed, true, `${id} release pass status`);
  assert(processEntries.length > 0, `${id} process proof missing`);
  assert(processEntries.every((entry) => entry.absent === true), `${id} process absence proof failed`);
  assert(collected, `${id} scope collection proof missing`);
  assert.equal(raw.lock?.freeProbeExit, 0, `${id} free-lock probe`);
  return {
    sourceSha256: rawPin(id).sha256,
    sourceBytes: rawPin(id).bytes,
    capturedAt: raw.capturedAt,
    passed: true,
    observedProcessStartIdentityCount: processEntries.length,
    allObservedIdentitiesAbsent: true,
    scopeCollected: true,
    lock: { device: raw.lock.dev, inode: raw.lock.ino, freeProbeExit: raw.lock.freeProbeExit },
  };
}

function leaseProjection(ids, note) {
  const rows = ids.map(ownFdLease);
  assert(rows.every((row) => row.passed && row.exactInheritedFdOwnershipProved
    && row.ownerStableBeforeAfter && row.descriptorStableBeforeAfter
    && row.exactOwnFlockRowBefore && row.exactOwnFlockRowAfter));
  return { checks: rows, note };
}

const matrixSourcePins = Object.entries(matrixController.sourceIdentity).sort(([a], [b]) => a.localeCompare(b));
const integerSourcePins = Object.entries(integerController.sourceIdentity).sort(([a], [b]) => a.localeCompare(b));
function sourceCopyProjection(source) {
  const record = copyRecords.get(source.id);
  assert(record);
  return {
    id: source.id,
    originRoot: source.root,
    originPath: source.originPath,
    archivePath: source.archivePath,
    bytes: record.bytes.length,
    sha256: hash(record.bytes),
  };
}
const copiesById = new Map(sourceCopies.map((source) => [source.id, source]));
for (const [originPath, pin] of Object.entries(clefController.sourceIdentity)) {
  const copy = sourceCopies.find((source) => source.root === "clef" && source.originPath === originPath);
  if (copy) assert.deepEqual({ bytes: pin.bytes, sha256: pin.sha256 }, { bytes: copy.bytes, sha256: copy.sha256 });
}
for (const [originPath, pin] of matrixSourcePins) {
  const copy = sourceCopies.find((source) => source.root === "matrix" && source.originPath === originPath);
  assert(copy, `matrix source identity has no package copy: ${originPath}`);
  assert.deepEqual({ bytes: pin.bytes, sha256: pin.sha256 }, { bytes: copy.bytes, sha256: copy.sha256 });
}
for (const [originPath, pin] of integerSourcePins) {
  const copy = sourceCopies.find((source) => source.root === "integer" && source.originPath === originPath);
  assert(copy, `integer-gate source identity has no package copy: ${originPath}`);
  assert.deepEqual({ bytes: pin.bytes, sha256: pin.sha256 }, { bytes: copy.bytes, sha256: copy.sha256 });
}

const currentHead = execFileSync("git", ["rev-parse", "HEAD"], { cwd: docsRoot, encoding: "utf8" }).trim();
assert.equal(currentHead, immutableSourceCommit, "prepare from the frozen shared-source merge commit");
execFileSync("git", ["cat-file", "-e", `${immutableSourceCommit}^{commit}`], { cwd: docsRoot, stdio: "ignore" });
const packageSourceRows = [];
const immutableSourceRows = [];
const hashOnlySourceRows = [];
for (const [path, pin] of Object.entries(clefController.sourceIdentity).sort(([a], [b]) => a.localeCompare(b))) {
  const sourceCopy = packageSourceByCapturePath.get(path);
  if (sourceCopy && sourceCopy.root === "clef") {
    const copy = copyRecords.get(sourceCopy.id);
    assert.deepEqual({ bytes: copy.bytes.length, sha256: hash(copy.bytes) }, { bytes: pin.bytes, sha256: pin.sha256 });
    packageSourceRows.push({ path, bytes: pin.bytes, sha256: pin.sha256,
      publication: { status: "included-in-this-package", archivePath: sourceCopy.archivePath } });
    continue;
  }
  const archivedPath = sharedArchiveSources.get(path);
  if (archivedPath) {
    const extension = path.slice(path.lastIndexOf("."));
    assert([".js", ".mjs", ".html", ".py"].includes(extension));
    const bytes = execFileSync("git", ["show", `${immutableSourceCommit}:${archivedPath}`], { cwd: docsRoot, maxBuffer: 2097152, stdio: ["ignore", "pipe", "ignore"] });
    assert.deepEqual({ bytes: bytes.length, sha256: hash(bytes) }, { bytes: pin.bytes, sha256: pin.sha256 });
    immutableSourceRows.push({ path, bytes: pin.bytes, sha256: pin.sha256,
      publication: { status: "immutable-archived-source-copy", commit: immutableSourceCommit, path: archivedPath,
        url: `${repositoryUrl}/blob/${immutableSourceCommit}/${archivedPath}` } });
    continue;
  }
  if (hashOnlySourcePaths.has(path)) {
    hashOnlySourceRows.push({ path, bytes: pin.bytes, sha256: pin.sha256,
      publication: { status: "hash-only", note: hashOnlySourcePaths.get(path) } });
    continue;
  }
  const directRepoPath = path.startsWith("js/src/") || path.startsWith("scripts/lib/")
    || ["dev/adapter-metadata.js", "scripts/serve.mjs", "app/model-policy.js"].includes(path);
  assert(directRepoPath, `unmapped captured source pin: ${path}`);
  const extension = path.slice(path.lastIndexOf("."));
  assert([".js", ".mjs", ".html", ".py"].includes(extension));
  const bytes = execFileSync("git", ["show", `${immutableSourceCommit}:${path}`], { cwd: docsRoot, maxBuffer: 2097152, stdio: ["ignore", "pipe", "ignore"] });
  assert.deepEqual({ bytes: bytes.length, sha256: hash(bytes) }, { bytes: pin.bytes, sha256: pin.sha256 });
  immutableSourceRows.push({ path, bytes: pin.bytes, sha256: pin.sha256,
    publication: { status: "immutable-repository-source", commit: immutableSourceCommit, path,
      url: `${repositoryUrl}/blob/${immutableSourceCommit}/${path}` } });
}
assert.equal(packageSourceRows.length, 2);
assert.equal(immutableSourceRows.length, 60);
assert.equal(hashOnlySourceRows.length, 2);
assert.equal(Object.keys(clefController.sourceIdentity).length, 64);
assert.equal(Object.keys(clefController.sourceIdentity).filter((path) => path.startsWith("js/src/") && path.endsWith(".js")).length, 50);

const profileSourceIdentity = {
  sourceSha256: rawPin("clefController").sha256,
  sourceBytes: rawPin("clefController").bytes,
  capturedPinCount: 64,
  runtimeJsPinCount: 50,
  immutableCommit: immutableSourceCommit,
  immutableByteMatchedPins: immutableSourceRows,
  packageSourcePins: packageSourceRows,
  hashOnlyPins: hashOnlySourceRows,
};
const sourceIdentitiesProjection = {
  clefProfile: profileSourceIdentity,
  subgroupMatrix: {
    sourceSha256: rawPin("matrixController").sha256,
    sourceBytes: rawPin("matrixController").bytes,
    pins: matrixSourcePins.map(([path, pin]) => ({ path, bytes: pin.bytes, sha256: pin.sha256,
      publication: { status: "included-in-this-package", archivePath: sourceCopies.find((source) => source.root === "matrix" && source.originPath === path).archivePath } })),
  },
  integerPackingGate: {
    sourceSha256: rawPin("integerController").sha256,
    sourceBytes: rawPin("integerController").bytes,
    pins: integerSourcePins.map(([path, pin]) => ({ path, bytes: pin.bytes, sha256: pin.sha256,
      publication: { status: "included-in-this-package", archivePath: sourceCopies.find((source) => source.root === "integer" && source.originPath === path).archivePath } })),
  },
  producerFiles: sourceCopies.map(sourceCopyProjection),
  note: "Source identities are exact path, byte-count, and SHA-256 pins. The archive is not a complete portable reproduction closure.",
};

const profileResultProjection = {
  sourceSha256: rawPin("clefResult").sha256,
  sourceBytes: rawPin("clefResult").bytes,
  model: clefResult.model,
  modelInfo: clefResult.modelInfo,
  arch: clefResult.arch,
  backend: clefResult.backend,
  flavor: clefResult.flavor,
  selectedFlavor: clefResult.selectedFlavor,
  precision: clefResult.precision,
  adapter: clefResult.adapter,
  cache: clefResult.cache,
  protocol: clefResult.protocol,
  warmups: clefResult.warmups,
  runs: clefResult.runs,
  unique: clefResult.unique,
  profile: clefResult.profile,
  context: clefResult.context,
  gpu: clefResult.gpu,
  gpuTuning: clefResult.gpuTuning,
  hardwareConcurrency: clefResult.hardwareConcurrency,
  threads: clefResult.threads,
  crossOriginIsolated: clefResult.crossOriginIsolated,
  clefHeadAfterWarmup: clefResult.clefHeadAfterWarmup,
  clefHeadInstrumentation: clefResult.clefHeadInstrumentation,
  pack: { sizeBytes: String(clefController.pack.size), freshWholeContentHashComputed: false },
  cases: clefResult.cases,
};
const profileControllerProjection = {
  sourceSha256: rawPin("clefController").sha256,
  sourceBytes: rawPin("clefController").bytes,
  status: clefController.status,
  runnerExit: clefController.runnerExit,
  browser: clefController.browser,
  cpuPolicy: clefController.cpuPolicy,
  request: clefController.request,
  packSizeBytes: String(clefController.pack.size),
  sourcePinCount: Object.keys(clefController.sourceIdentity).length,
  runtimeJsPinCount: Object.keys(clefController.sourceIdentity).filter((path) => path.startsWith("js/src/") && path.endsWith(".js")).length,
};
const matrixResultProjection = {
  sourceSha256: rawPin("matrixResult").sha256,
  sourceBytes: rawPin("matrixResult").bytes,
  status: matrixResult.status,
  done: matrixResult.done,
  lane: matrixResult.lane,
  scope: matrixResult.scope,
  page: matrixResult.page,
};
const matrixControllerProjection = {
  sourceSha256: rawPin("matrixController").sha256,
  sourceBytes: rawPin("matrixController").bytes,
  status: matrixController.status,
  capabilityAndCompilationOnly: matrixController.capabilityAndCompilationOnly,
  productionPromotion: matrixController.productionPromotion,
  lanes: matrixController.lanes.map((lane) => ({ lane: lane.lane, result: selected(lane.result || {}, ["bytes", "sha256", "status"]) })),
  sourceIdentity: Object.fromEntries(matrixSourcePins),
};
const integerResultProjection = {
  sourceSha256: rawPin("integerResult").sha256,
  sourceBytes: rawPin("integerResult").bytes,
  status: integerResult.status,
  done: integerResult.done,
  lane: integerResult.lane,
  scope: integerResult.scope,
  page: integerResult.page,
  runner: selected(integerResult.runner, ["browserVersion", "chromiumArgs", "existingBrowserAttached"]),
};
const integerControllerProjection = {
  sourceSha256: rawPin("integerController").sha256,
  sourceBytes: rawPin("integerController").bytes,
  status: integerController.status,
  gateOnly: integerController.gateOnly,
  productionPromotion: integerController.productionPromotion,
  lanes: integerController.lanes.map((lane) => ({ lane: lane.lane, result: selected(lane.result || {}, ["bytes", "sha256", "status"]) })),
  sourceIdentity: Object.fromEntries(integerSourcePins),
};

const clefSmoke = (await value("clefStaticSmoke"));
assert.deepEqual(clefSmoke.sources, {
  "bench-clef-kernel-profile.html": sourceCopies.find((source) => source.id === "clefProfileHtml").sha256,
  "scripts/private/clef-kernel-profile.mjs": sourceCopies.find((source) => source.id === "clefProfileScript").sha256,
});
const clefStaticProjection = {
  sourceSha256: rawPin("clefStaticSmoke").sha256,
  sourceBytes: rawPin("clefStaticSmoke").bytes,
  capturedAt: clefSmoke.capturedAt,
  limitation: clefSmoke.limitation,
  sources: clefSmoke.sources,
  fullGraph: clefSmoke.fullGraph,
  forcedMissingPage: clefSmoke.forcedMissingPage,
  foreignRuntime: clefSmoke.foreignRuntime,
  idle: clefSmoke.idle,
};
for (const check of [clefSmoke.fullGraph, clefSmoke.forcedMissingPage, clefSmoke.foreignRuntime, clefSmoke.idle]) {
  assert.equal(check.actualGpuCalls.adapterRequests, 0);
  assert.equal(check.actualGpuCalls.deviceRequests, 0);
  assert.equal(check.forbiddenPayloadRequests, 0);
}
const integerSmoke = await value("integerStaticSmoke");
assert.equal(integerSmoke.gpuEntrypointsBlocked, true);
assert.deepEqual(integerSmoke.sources, {
  "scripts/private/integer-matrix-gate.html": { bytes: 15651, sha256: "aba79fa9a419448edec639fb53f8ed4979f887d5950cd1fb8d8c18caf652a18b" },
  "scripts/private/integer-matrix-gate-helper.js": { bytes: 12993, sha256: "70e56d7d3ba2053fb46f84573f95647366e694f86950c3f3f1ee1bedc71460eb" },
});
assert.equal(integerSmoke.result.idle.gpuCalls.adapterRequests, 0);
assert.equal(integerSmoke.result.idle.gpuCalls.deviceRequests, 0);
assert.equal(integerSmoke.result.idle.forbiddenPayloadRequests, 0);
assert.equal(integerSmoke.result.forcedMissing.gpuCalls.adapterRequests, 0);
assert.equal(integerSmoke.result.forcedMissing.gpuCalls.deviceRequests, 0);
assert.equal(integerSmoke.result.forcedMissing.forbiddenPayloadRequests, 0);
const integerStaticProjection = {
  sourceSha256: rawPin("integerStaticSmoke").sha256,
  sourceBytes: rawPin("integerStaticSmoke").bytes,
  capturedAt: integerSmoke.at,
  gpuEntrypointsBlocked: integerSmoke.gpuEntrypointsBlocked,
  sources: integerSmoke.sources,
  result: integerSmoke.result,
};

const leaseSets = {
  clef: inputs.filter((item) => item.id.startsWith("clefLease")).map((item) => item.id),
  matrix: inputs.filter((item) => item.id.startsWith("matrixLease")).map((item) => item.id),
  integer: inputs.filter((item) => item.id.startsWith("integerLease")).map((item) => item.id),
};
const evidence = new Map([
  ["evidence/clef-profile/result.json", profileResultProjection],
  ["evidence/clef-profile/controller.json", profileControllerProjection],
  ["evidence/clef-profile/source-identities.json", sourceIdentitiesProjection],
  ["evidence/clef-profile/release.json", releaseProjection("clefRelease")],
  ["evidence/clef-profile/static-browser-smoke.json", clefStaticProjection],
  ["evidence/clef-profile/leases.json", leaseProjection(leaseSets.clef, "Selected own-FD proofs only. No full descriptor inventory, process identity, or global lock table is included.")],
  ["evidence/subgroup-matrix/result.json", matrixResultProjection],
  ["evidence/subgroup-matrix/controller.json", matrixControllerProjection],
  ["evidence/subgroup-matrix/release.json", releaseProjection("matrixRelease")],
  ["evidence/subgroup-matrix/leases.json", leaseProjection(leaseSets.matrix, "Selected own-FD proofs only. No full descriptor inventory, process identity, or global lock table is included.")],
  ["evidence/integer-packing-gate/result.json", integerResultProjection],
  ["evidence/integer-packing-gate/controller.json", integerControllerProjection],
  ["evidence/integer-packing-gate/release.json", releaseProjection("integerRelease")],
  ["evidence/integer-packing-gate/static-browser-smoke.json", integerStaticProjection],
  ["evidence/integer-packing-gate/leases.json", leaseProjection(leaseSets.integer, "Selected own-FD proofs only. No full descriptor inventory, process identity, or global lock table is included.")],
]);

const outputPlans = [];
for (const [relative, projection] of evidence) {
  const bytes = jsonBytes(projection);
  assert(bytes.length <= maxProjectionBytes, `projection exceeds 1 MiB: ${relative}`);
  outputPlans.push({ path: join(directory, relative), relative, bytes });
}
const generatedProjectionPins = outputPlans.map(({ relative, bytes }) => ({ path: relative, bytes: bytes.length, sha256: hash(bytes) }));
const provenance = {
  schemaVersion: 1,
  captureDate: "2026-11-07",
  publicationBaseCommit: immutableSourceCommit,
  rawInputs: inputs.map((spec) => {
    const record = inputRecords.get(spec.id);
    assert(record, `raw input was not included in the frozen inventory: ${spec.id}`);
    return { id: spec.id, captureRoot: spec.root, path: spec.path, bytes: record.bytes.length, sha256: hash(record.bytes) };
  }),
  sourceCopies: sourceCopies.map(sourceCopyProjection),
  clefSourcePinInventory: {
    capturedCount: 64,
    runtimeJsCount: 50,
    immutableReferenceCommit: immutableSourceCommit,
    immutableByteMatchedCount: immutableSourceRows.length,
    packageCopyCount: packageSourceRows.length,
    hashOnlyCount: hashOnlySourceRows.length,
    hashOnlyPaths: hashOnlySourceRows.map((entry) => entry.path),
    inventoryPath: "evidence/clef-profile/source-identities.json",
  },
  generatedProjectionPins,
  exclusions: [
    "model weights and pack payloads",
    "compiled WASM and copied runtime trees",
    "full raw controllers and runner logs",
    "trace files and browser logs",
    "full process and thread inventories",
    "global /proc/locks text",
  ],
  limits: { maxRawInputBytes: maxInputBytes, maxDecodedBytes: 1048576, maxCompressedBytes: 262144 },
};

const writeNew = async (path, bytes) => {
  const file = await open(path, "wx", 0o644);
  try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
  const parent = await open(dirname(path), "r");
  try { await parent.sync(); } finally { await parent.close(); }
  const saved = await readFile(path);
  assert(saved.equals(bytes), `write verification failed: ${path}`);
};
for (const output of outputPlans) {
  await mkdir(dirname(output.path), { recursive: true });
}
const provenancePath = join(directory, "provenance.json");
const provenanceBytes = prettyBytes(provenance);
assert(provenanceBytes.length <= maxProjectionBytes);
await writeNew(provenancePath, provenanceBytes);
for (const output of outputPlans) await writeNew(output.path, output.bytes);

const packageFiles = [
  { id: "README.md", archivePath: "README.md", sourcePath: "docs/benchmarks/clef-gpu-profile-2026-11-07/README.md" },
  { id: "archive-publication.mjs", archivePath: "archive-publication.mjs", sourcePath: "docs/benchmarks/clef-gpu-profile-2026-11-07/archive-publication.mjs" },
  { id: "fd-lease.mjs", archivePath: "fd-lease.mjs", sourcePath: "docs/benchmarks/clef-gpu-profile-2026-11-07/fd-lease.mjs" },
  { id: "lease-parse.mjs", archivePath: "lease-parse.mjs", sourcePath: "docs/benchmarks/clef-gpu-profile-2026-11-07/lease-parse.mjs" },
  { id: "prepare-evidence.mjs", archivePath: "prepare-evidence.mjs", sourcePath: "docs/benchmarks/clef-gpu-profile-2026-11-07/prepare-evidence.mjs" },
  { id: "provenance.json", archivePath: "provenance.json", sourcePath: "docs/benchmarks/clef-gpu-profile-2026-11-07/provenance.json" },
  { id: "report", archivePath: "report/clef-gpu-kernel-profile.md", sourcePath: "docs/clef-gpu-kernel-profile.md" },
  ...outputPlans.map(({ relative }) => ({ id: relative, archivePath: relative, sourcePath: `docs/benchmarks/clef-gpu-profile-2026-11-07/${relative}` })),
  ...sourceCopies.map((source) => ({ id: source.id, archivePath: source.archivePath, sourcePath: `docs/benchmarks/clef-gpu-profile-2026-11-07/${source.archivePath}` })),
];
const sourcePins = [];
const planEntries = [];
for (const file of packageFiles) {
  const path = resolve(docsRoot, file.sourcePath);
  const original = await lstat(path);
  assert(original.isFile() && !original.isSymbolicLink() && original.size <= maxInputBytes, `invalid package source: ${file.sourcePath}`);
  const canonical = await realpath(path);
  assert(canonical.startsWith(docsRoot + sep), `package source escaped documentation root: ${file.sourcePath}`);
  const bytes = await readFile(canonical);
  const pin = { id: file.id, path: file.sourcePath, bytes: bytes.length, sha256: hash(bytes) };
  sourcePins.push(pin);
  planEntries.push({ archivePath: file.archivePath, method: "copy-exact", sourceId: file.id });
}
planEntries.push({ archivePath: "archive-plan.json", method: "copy-plan" });
sourcePins.sort((a, b) => Buffer.compare(Buffer.from(a.path), Buffer.from(b.path)));
planEntries.sort((a, b) => Buffer.compare(Buffer.from(a.archivePath), Buffer.from(b.archivePath)));

const plan = {
  schemaVersion: 1,
  name: "Clef GPU profile, subgroup-matrix metadata, and integer packing gate evidence",
  baseCommit: immutableSourceCommit,
  archive: {
    path: "evidence.json.gz",
    manifestPath: "manifest.json",
    maxSourceBytes: maxInputBytes,
    maxDecodedBytes: 1048576,
    maxCompressedBytes: 262144,
    estimatedDecodedBytes: 0,
    sizeEstimateMethod: "Exact source bytes and base64 JSON framing, including the pinned plan bytes. Plan mode verifies the estimate without creating gzip.",
    maxEntries: 64,
    entryOrder: "UTF-8 bytewise ascending by archivePath",
    derivedJson: "Evidence projections use compact JSON plus one trailing newline. The archive payload, manifest, and plan use two-space JSON indentation plus one trailing newline.",
    gzipMtimeSeconds: 0,
    overwrite: false,
    requiredLock: canonicalLock,
    publisherPattern: "Fixed inventory with pinned helpers, nonblocking inherited-FD proof, and bounded archive verification.",
  },
  lanes: [
    { id: "clef-profile-r1", reportSection: "Clef GPU kernel profile", evidencePrefix: "evidence/clef-profile/" },
    { id: "subgroup-matrix-probe-r1", reportSection: "Subgroup-matrix metadata probe", evidencePrefix: "evidence/subgroup-matrix/" },
    { id: "integer-packing-gate-r1", reportSection: "Integer packing gate", evidencePrefix: "evidence/integer-packing-gate/" },
  ],
  sourcePins,
  entries: planEntries,
  excludedSources: [
    "model weights and pack payloads",
    "compiled WASM and copied runtime trees",
    "full raw controllers and runner logs",
    "trace files and browser logs",
    "full process and thread inventories",
    "global /proc/locks text",
  ],
  otherExclusions: [
    "Clef raw output records contain answers, model, and usage only; no raw probability array is present.",
    "The subgroup-matrix metadata probe skipped compilation and has no shader or dispatch result.",
    "The integer gate covers one signed i8 M16 N16 K32 packing and workgroup-store operation only; it is not full-operator correctness or latency evidence.",
    "Two Clef controller source pins are explicitly hash-only; this package is not a complete portable reproduction closure.",
  ],
};

async function absent(path) {
  try { await lstat(path); throw Error(`refusing existing output: ${path}`); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
}
await absent(join(directory, "archive-plan.json"));
const fileCache = new Map();
for (const pin of sourcePins) {
  const bytes = await readFile(resolve(docsRoot, pin.path));
  fileCache.set(pin.id, bytes);
}
let settled = false;
for (let attempt = 0; attempt < 8; attempt++) {
  const planBytes = prettyBytes(plan);
  const bundleFiles = [];
  for (const entry of plan.entries) {
    const bytes = entry.method === "copy-plan" ? planBytes : fileCache.get(entry.sourceId);
    assert(bytes, `missing pinned source for ${entry.archivePath}`);
    bundleFiles.push({ path: entry.archivePath, bytes: bytes.length, sha256: hash(bytes), encoding: "base64", data: bytes.toString("base64") });
  }
  bundleFiles.sort((a, b) => Buffer.compare(Buffer.from(a.path), Buffer.from(b.path)));
  const decodedBytes = prettyBytes({ schemaVersion: 1, planSha256: hash(planBytes), files: bundleFiles }).length;
  if (plan.archive.estimatedDecodedBytes === decodedBytes) { settled = true; break; }
  plan.archive.estimatedDecodedBytes = decodedBytes;
}
assert(settled, "could not settle the self-inclusive decoded-size estimate");
const planBytes = prettyBytes(plan);
await writeNew(join(directory, "archive-plan.json"), planBytes);

const leaseAfter = await leaseGuard.check("clef-gpu-profile-evidence-preparation-complete");
assert.deepEqual(leaseBefore.binding, leaseAfter.binding);
assert.equal(leaseAfter.exactInheritedFdOwnershipProved, true);
console.log(JSON.stringify({
  status: "prepared-and-bounded",
  rawInputsVerified: inputs.length,
  sourceCopiesVerified: sourceCopies.length,
  clefRouteRecords: routeRecords.length,
  projectionsWritten: outputPlans.length,
  sourcePins: { total: Object.keys(clefController.sourceIdentity).length, immutableByteMatched: immutableSourceRows.length,
    packageCopies: packageSourceRows.length, hashOnly: hashOnlySourceRows.length },
  archivePlanEntries: plan.entries.length,
  estimatedDecodedBytes: plan.archive.estimatedDecodedBytes,
  decodedLimitBytes: plan.archive.maxDecodedBytes,
  archiveOrGzipCreated: false,
  exactInheritedFdOwnershipProved: true,
  postCheck: true,
}));
