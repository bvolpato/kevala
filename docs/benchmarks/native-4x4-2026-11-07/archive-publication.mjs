import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { lstatSync, readFileSync, realpathSync } from "node:fs";
import { lstat, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const output = dirname(fileURLToPath(import.meta.url));
const repo = resolve(output, "../../..");
const lockPath = "/home/bruno/.cache/kevala/gpu-profile.lock";
const evidencePath = "/home/bruno/githubworkspace/kevala/tmp/sota-20261106-kdDTGn/native-4x4";
const baseCommit = "dd6c94a25cb707112ff7f146ceff756d1157cfd0";
const candidateHash = "a1d101ed08726cafb2d7a301a39edf789adfc203044852fd8c7b4cf7cf12b5da";
const validationPath = resolve(evidencePath, "production-auto4-release-v1-r1");
const validationReceiptHash = "057a37da8fe56854ba569cb876ce7253a6dfc5dc4d842d1dc0e9a685ff2d8df6";
const validationInventoryHash = "d30cb21e6d815c0fdd3f8afabeb633f0e606a8a56410c6299ecb45ddf299a1c0";
const executionAuditInventoryHash = "9371bab71d9370b04e3d6ea41cb529b94f98363ae8bd0ca572dec0f73b716d67";
const validationProtocolFiles = [
  ["production-auto4-release-v1-r1/plan.json", "fbb68dca7c23ab543eb81d353f2cdb5ec3aaaf57ce7886a955d9f3031adaf78f"],
  ["production-auto4-release-v1-r1/run.mjs", "83ba94735ae46de03e509fd78fbb464c492d261bc126437b37f55aea4b15b748"],
  ["production-auto4-release-v1-r1/source-index.json", "04bcaa369e43869ae8e488ded342a425ddfbc065be1f636484b656e8fe2ee801"],
  ["launch-core-v2-r1.mjs", "9f2e0423e0561cb487a80e0e936454e7285653113950608c2136fa8cd1db28b1"],
  ["prepare-production-auto4-v1-r1.mjs", "3ef259a30a91f2631cc4dc1b9be9224ada17cbd604adbb6516fc25f1a2279954"],
  ["publication-history/archive-publication-pre-lease-fix.mjs", "28af2c4544e8d29aa959613ef2365cf86dfc3a2b4f5075b0745334a7883ea7f4"],
];
const maxFileBytes = 32 * 1024 * 1024;
const allowed = /\.(?:asm|json|jsonl|js|lock|log|md|mjs|patch|rs|sh|stderr|stdout|toml|wgsl|txt)$/i;
const forbidden = /\.(?:wasm|kevala|safetensors|gguf|bin|pt|npy|npz|f32|f16|o|a|so|rlib|rmeta|gz|zst|zip)$/i;
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const state = { mode: null, evidence: evidencePath, validation: null, release: null, io: false };

function parseArgs() {
  const args = process.argv.slice(2);
  state.mode = args.shift();
  if (!["--plan", "--archive", "--verify"].includes(state.mode)) throw Error("use --plan, --archive --allow-io, or --verify --allow-io");
  while (args.length) {
    const key = args.shift();
    if (key === "--allow-io") state.io = true;
    else if (key === "--evidence") state.evidence = args.shift();
    else if (key === "--validation") state.validation = args.shift();
    else if (key === "--release") state.release = args.shift();
    else throw Error(`unknown argument: ${key}`);
  }
  if (!isAbsolute(state.evidence)) throw Error("--evidence must be an absolute directory");
  if (state.validation && !isAbsolute(state.validation)) throw Error("--validation must be an absolute directory");
  if (state.validation && resolve(state.validation) !== validationPath) throw Error("--validation must name the frozen AUTO4 production packet");
  if ((state.mode !== "--plan") && !state.io) throw Error(`${state.mode} requires --allow-io and the parent's I/O window`);
  if (state.mode === "--verify" && state.validation) throw Error("--verify does not accept --validation");
}

function linuxDevice(dev) {
  const major = Number((dev >> 8n & 0xfffn) | (dev >> 32n & 0xfffff000n));
  const minor = Number((dev & 0xffn) | (dev >> 12n & 0xffffff00n));
  return { major: BigInt(major), minor: BigInt(minor) };
}

function assertLease() {
  if (realpathSync(lockPath) !== lockPath) throw Error("canonical profile lock path must resolve to itself");
  const lock = lstatSync(lockPath, { bigint: true });
  if (!lock.isFile() || lock.isSymbolicLink()) throw Error("canonical profile lock path must be a regular file");
  const expected = linuxDevice(lock.dev);
  const rows = readFileSync("/proc/locks", "utf8").split("\n").map(line => ({ rawLine: line, fields: line.trim().split(/\s+/) }));
  const matches = rows.filter(({ fields }) => {
    if (fields[1] !== "FLOCK" || fields[2] !== "ADVISORY" || fields[3] !== "WRITE" || fields[4] !== String(process.pid) || fields[6] !== "0" || fields[7] !== "EOF") return false;
    const identity = /^([0-9a-f]+):([0-9a-f]+):(\d+)$/i.exec(fields[5] || "");
    return identity !== null && BigInt(`0x${identity[1]}`) === expected.major && BigInt(`0x${identity[2]}`) === expected.minor && BigInt(identity[3]) === lock.ino;
  });
  if (matches.length !== 1) throw Error("archive/verify requires this process to hold the full-range canonical profile flock");
  return { pid: process.pid, path: lockPath, deviceMajor: expected.major.toString(), deviceMinor: expected.minor.toString(), inode: lock.ino.toString(), kind: "FLOCK", advisory: "ADVISORY", mode: "WRITE", start: "0", end: "EOF", rawLine: matches[0].rawLine };
}

function sameLease(left, right) {
  return ["pid", "path", "deviceMajor", "deviceMinor", "inode", "kind", "advisory", "mode", "start", "end"].every(key => left[key] === right[key]);
}

function safeRelative(path) {
  if (typeof path !== "string" || !path || isAbsolute(path) || path.split(/[\\/]/).some(part => !part || part === "." || part === "..")) throw Error(`unsafe path: ${path}`);
  return path;
}

async function checkedPath(root, path) {
  safeRelative(path);
  let current = root;
  const parts = path.split(/[\\/]/);
  for (let index = 0; index < parts.length; index++) {
    current = resolve(current, parts[index]);
    const info = await lstat(current);
    if (info.isSymbolicLink() || (index < parts.length - 1 && !info.isDirectory()) || (index === parts.length - 1 && !info.isFile())) throw Error(`not a regular file path: ${path}`);
  }
  const info = await lstat(current);
  if (info.size > maxFileBytes) throw Error(`file exceeds 32 MiB: ${path}`);
  return { absolute: current, bytes: info.size };
}

async function fileAt(root, path, group, expected = null, expectedBytes = null, read = true) {
  if (!allowed.test(path) || forbidden.test(path)) throw Error(`disallowed archive input: ${path}`);
  const info = await checkedPath(root, path);
  if (expectedBytes !== null && info.bytes !== expectedBytes) throw Error(`byte count differs from inventory: ${path}`);
  if (!read) return { path, group, bytes: info.bytes, sha256: expected, source: resolve(root, path) };
  const data = await readFile(info.absolute);
  const digest = hash(data);
  if (expected && digest !== expected) throw Error(`hash differs from inventory: ${path}`);
  return { path, group, bytes: data.length, sha256: digest, source: resolve(root, path), data };
}

async function jsonAt(root, path) {
  const info = await checkedPath(root, path);
  return JSON.parse(await readFile(info.absolute, "utf8"));
}

const sourceFiles = [
  "frozen-source-v2.json", "source-inventory-v2.json", "strict-codegen-plan-v2.json",
  "actual-default-model-plan-v2.json", "launch-plan-v2-r1.json", "gate-launch-index-v2-r1.json",
  "gates-release-v2-r1.json", "scoped-candidate-v2.patch", "prototype-to-auto-v2.patch",
  "run-gates-v2-r1.mjs", "inspect-stage-assembly-v2-r1.mjs", "prepare-gate-launch-v2-r1.mjs",
  "parity-v2/src/main.rs",
];
const microFiles = [
  "micro-inventory-v1-r2.json", "micro-summary-inventory-v1-r2.json", "micro-summary-v1-r2.json",
  "micro-v1-r2/experiment.command.json", "micro-v1-r2/experiment.stdout", "micro-v1-r2/receipt.json",
  "micro-v1-r2/identical-control.command.json", "micro-v1-r2/identical-control.stdout",
  "micro-v1-r2/identical-libraries-before.json", "micro-v1-r2/identical-libraries-after-control.json",
  "micro-v1-r2/identical-control.stderr", "run-micro-v1-r2.mjs", "gate-and-micro-plan-v1.json",
];
const gateInventoryPath = "gates-inventory-v2-r1.json";
const modelPrefix = "paired-native-auto2-v2-r1";
const modelExplicit = [
  `${modelPrefix}/plan.json`, `${modelPrefix}/source-index.json`, `${modelPrefix}/run-inventory-v1.json`,
  `${modelPrefix}/exec-cli.mjs`, `${modelPrefix}/metadata.mjs`, `${modelPrefix}/run.mjs`,
  `${modelPrefix}/lease-release-v1.json`, `${modelPrefix}/runs-v1/receipt.json`,
  `${modelPrefix}/runs-v1/journal.jsonl`,
];

function put(groups, entry) {
  if (!groups.has(entry.group)) groups.set(entry.group, []);
  if (groups.get(entry.group).some(row => row.path === entry.path)) return;
  groups.get(entry.group).push(entry);
}

async function collect({ read, includePublication = true }) {
  const rootInfo = await lstat(state.evidence);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) throw Error("evidence root must be a real directory");
  const groups = new Map(), omitted = [], sourceInventories = {};
  const add = async (root, path, group, expected = null, bytes = null) => put(groups, await fileAt(root, path, group, expected, bytes, read));

  const sourceInventory = await jsonAt(state.evidence, "source-inventory-v2.json");
  const sourceInventoryHash = hash(await readFile(resolve(state.evidence, "source-inventory-v2.json")));
  if (sourceInventoryHash !== "ac67f24366b2cf7d836d1980d3c95b590d42c5959c440589b7b4836c8bea2841") throw Error("frozen source inventory hash differs");
  for (const path of sourceFiles) {
    const expected = sourceInventory.files?.[path];
    await add(state.evidence, path, "source-and-protocol", expected?.sha256 ?? null, expected?.bytes ?? null);
  }
  put(groups, await fileAt(state.evidence, "source-inventory-v2.json", "source-and-protocol", sourceInventoryHash));
  const kernel = await fileAt(repo, "crates/kevala/src/kernels.rs", "source-and-protocol", candidateHash, null, read);
  kernel.path = "candidate/crates/kevala/src/kernels.rs";
  put(groups, kernel);
  const patch = execFileSync("git", ["diff", "--binary", baseCommit, "--", "crates/kevala/src/kernels.rs"], { cwd: repo });
  if (!patch.length || !patch.toString("utf8").startsWith("diff --git a/crates/kevala/src/kernels.rs")) throw Error("cannot reconstruct the candidate kernel patch from base commit");
  put(groups, { path: "candidate/native-auto4.patch", group: "source-and-protocol", bytes: patch.length, sha256: hash(patch), source: "git diff from base", data: read ? patch : undefined });
  const candidatePath = "frozen-v2/candidate/crates/kevala/src/kernels.rs";
  if (sourceInventory.files?.[candidatePath]?.sha256 !== candidateHash) throw Error("frozen source inventory does not identify the AUTO4 kernel");
  for (const [path, expected] of validationProtocolFiles) await add(state.evidence, path, "production-validation-protocol", expected);
  const validationPlan = await jsonAt(state.evidence, "production-auto4-release-v1-r1/plan.json");
  const validationSourceIndex = await jsonAt(state.evidence, "production-auto4-release-v1-r1/source-index.json");
  if (validationPlan.productionBase !== baseCommit || validationPlan.kernelSha256 !== candidateHash || validationPlan.outputInventory !== resolve(validationPath, "result-inventory-v1.json") || validationPlan.inputPins?.["paired-native-auto2-v2-r1/runs-v1/receipt.json"] !== "b3f25efcfaab04b78fc5f8bf0f40b6ab4597ae2c124e7dc62f3bae6e2211365c" || validationSourceIndex.releaseSourceFingerprintSha256 !== "099f782102253276a610540a18730f3b631e01a20dd1326b2adc30b0a026e796") throw Error("production validation plan/index do not bind the measured AUTO4 source and prior evidence");
  if (validationSourceIndex.files?.["../prepare-production-auto4-v1-r1.mjs"]?.sha256 !== "3ef259a30a91f2631cc4dc1b9be9224ada17cbd604adbb6516fc25f1a2279954") throw Error("production source index does not bind its preparer");
  for (const [path, desc] of Object.entries(validationSourceIndex.files || {})) {
    if (!path.startsWith("../")) await add(validationPath, path, "production-validation-protocol", desc.sha256, desc.bytes);
  }

  const microInventory = await jsonAt(state.evidence, "micro-inventory-v1-r2.json");
  const microIncluded = new Set(microFiles.filter(path => path.startsWith("micro-v1-r2/")));
  for (const [path, desc] of Object.entries(microInventory.files || {})) {
    if (microIncluded.has(path)) await add(state.evidence, path, "prototype-micro-screen", desc.sha256, desc.bytes);
    else omitted.push({ group: "prototype-micro-screen", path, bytes: desc.bytes, sha256: desc.sha256, reason: "not needed in compact archive; original inventory retains size and SHA-256" });
  }
  const microSummaryInventory = await jsonAt(state.evidence, "micro-summary-inventory-v1-r2.json");
  for (const [path, desc] of Object.entries(microSummaryInventory.files || {})) await add(state.evidence, path, "prototype-micro-screen", desc.sha256, desc.bytes);
  for (const path of ["run-micro-v1-r2.mjs", "gate-and-micro-plan-v1.json"]) await add(state.evidence, path, "prototype-micro-screen");

  for (const path of [gateInventoryPath]) await add(state.evidence, path, "strict-parity-and-codegen");
  const gateInventory = await jsonAt(state.evidence, gateInventoryPath);
  for (const [path, desc] of Object.entries(gateInventory.files || {}).sort(([a], [b]) => a.localeCompare(b))) {
    await add(state.evidence, path, "strict-parity-and-codegen", desc.sha256, desc.bytes);
  }

  const modelSourceIndex = await jsonAt(state.evidence, `${modelPrefix}/source-index.json`);
  for (const path of modelExplicit) {
    const relativePath = path.slice(modelPrefix.length + 1);
    const desc = modelSourceIndex.files?.[relativePath];
    await add(state.evidence, path, "actual-default-model-run", desc?.sha256 ?? null, desc?.bytes ?? null);
  }
  const runIndex = await jsonAt(state.evidence, `${modelPrefix}/run-inventory-v1.json`);
  for (const [path, desc] of Object.entries(runIndex.files || {}).sort(([a], [b]) => a.localeCompare(b))) {
    await add(state.evidence, `${modelPrefix}/${path}`, "actual-default-model-run", desc.sha256, desc.bytes);
  }

  let validation = null;
  if (state.validation) validation = await collectValidation(read, omitted, groups);
  if (includePublication) {
    for (const path of ["docs/native-4x4-performance.md", "docs/benchmarks/native-4x4-2026-11-07/README.md", "docs/benchmarks/native-4x4-2026-11-07/archive-publication.mjs"]) {
      try { await add(repo, path, "publication-source"); }
      catch (error) { if (error.code !== "ENOENT") throw error; }
    }
  }
  for (const rows of groups.values()) rows.sort((a, b) => a.path.localeCompare(b.path));
  return { groups, omitted, validation };
}

async function collectValidation(read, omitted, groups) {
  const rootInfo = await lstat(state.validation);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) throw Error("validation evidence root must be a real directory");
  const indexPath = "result-inventory-v1.json";
  const indexBytes = await readFile(resolve(state.validation, indexPath));
  if (hash(indexBytes) !== validationInventoryHash) throw Error("production result inventory differs from the frozen validation run");
  const index = JSON.parse(indexBytes.toString("utf8"));
  if (!index.files || Object.keys(index.files).length !== 55) throw Error("production result inventory must bind the complete 55-file packet");
  const receiptDescriptor = index.files["results-v1/receipt.json"];
  if (receiptDescriptor?.sha256 !== validationReceiptHash) throw Error("production receipt is absent from its result inventory");
  const receipt = await jsonAt(state.validation, "results-v1/receipt.json");
  const receiptBytes = await readFile(resolve(state.validation, "results-v1/receipt.json"));
  if (hash(receiptBytes) !== validationReceiptHash || receipt.allPassed !== true) throw Error("production receipt is not the independently audited passing receipt");
  const auditIndexPath = "execution-audit-inventory-v1.json";
  const auditIndexBytes = await readFile(resolve(state.validation, auditIndexPath));
  const auditIndexHash = hash(auditIndexBytes);
  if (auditIndexHash !== executionAuditInventoryHash) throw Error("production execution audit inventory differs from the frozen run");
  const auditIndex = JSON.parse(auditIndexBytes.toString("utf8"));
  const expectedAuditFiles = {
    "execution-envelope-v1.json": "c3a353ed49acc510ea4622e108f7008ef6de5cb0b6a02c5498f1b5dbf8d69e60",
    "initial-scope-audit-v1.json": "f3c512a3bc1c96933c513f1aaaf80f8a96af0ecaf7058d4e140645085c753f34",
    "final-execution-audit-v1.json": "7c15a11bcf4a9af5a0bbb59962aac38642f086285740ded130d5ec58e3d2baf4",
    "result-inventory-v1.json": validationInventoryHash,
    "results-v1/receipt.json": validationReceiptHash,
  };
  if (Object.keys(auditIndex.files || {}).length !== Object.keys(expectedAuditFiles).length || Object.entries(expectedAuditFiles).some(([path, sha]) => auditIndex.files?.[path]?.sha256 !== sha)) throw Error("execution audit index does not bind the exact validation and scope records");
  const outputs = new Map();
  const inventory = await fileAt(state.validation, indexPath, "production-validation-results", validationInventoryHash, indexBytes.length, read);
  put(groups, inventory);
  for (const [path, desc] of Object.entries(index.files).sort(([a], [b]) => a.localeCompare(b))) {
    if (!allowed.test(path) || forbidden.test(path)) {
      omitted.push({ group: "production-validation", path, bytes: desc.bytes, sha256: desc.sha256, reason: "binary/build payload excluded; exact identity retained in inventory" });
      continue;
    }
    const entry = await fileAt(state.validation, path, "production-validation-results", desc.sha256, desc.bytes, read);
    entry.path = path;
    if (entry.data) outputs.set(path, entry.data);
    put(groups, entry);
  }
  put(groups, await fileAt(state.validation, "results-v1/receipt.json", "production-validation-results", validationReceiptHash, receiptDescriptor.bytes, read));
  const auditInventoryEntry = await fileAt(state.validation, auditIndexPath, "production-validation-execution", executionAuditInventoryHash, auditIndexBytes.length, read);
  put(groups, auditInventoryEntry);
  for (const [path, expectedHash] of Object.entries(expectedAuditFiles)) {
    if ([indexPath, "results-v1/receipt.json"].includes(path)) continue;
    const desc = auditIndex.files[path];
    if (desc?.sha256 !== expectedHash) throw Error(`execution audit inventory entry differs: ${path}`);
    put(groups, await fileAt(state.validation, path, "production-validation-execution", expectedHash, desc.bytes, read));
  }
  return { status: "passed", inventorySha256: validationInventoryHash, receiptSha256: validationReceiptHash, executionAuditInventorySha256: auditIndexHash, finalExecutionAuditSha256: expectedAuditFiles["final-execution-audit-v1.json"], outputFiles: Object.keys(index.files).length };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function deriveModelStats(files) {
  const get = path => files.get(`actual-default-model-run/${path}`);
  const receiptBytes = get("paired-native-auto2-v2-r1/runs-v1/receipt.json");
  const inventoryBytes = get("paired-native-auto2-v2-r1/run-inventory-v1.json");
  if (!receiptBytes || !inventoryBytes) throw Error("paired model receipt or inventory is missing");
  const receipt = JSON.parse(receiptBytes.toString("utf8"));
  const inventory = JSON.parse(inventoryBytes.toString("utf8"));
  if (hash(receiptBytes) !== "b3f25efcfaab04b78fc5f8bf0f40b6ab4597ae2c124e7dc62f3bae6e2211365c" || !receipt.complete || receipt.observationsRecorded !== 84 || receipt.discardedObservations !== 12 || receipt.measuredObservations !== 72 || receipt.allCompletedResponsesExact !== true || receipt.retries !== 0) throw Error("actual-default model receipt does not match the frozen run");
  if (Object.keys(inventory.files).length !== 513 || receipt.records.length !== 84) throw Error("model observation or raw artifact count differs");
  if (receipt.records.reduce((sum, row) => sum + row.numericValuesCompared, 0) !== 420) throw Error("numeric comparison count differs");
  const responseFor = row => {
    const marker = `${modelPrefix}/`;
    const start = row.stdout.lastIndexOf(marker);
    if (start < 0) throw Error(`response path leaves paired run: ${row.id}`);
    const path = `${modelPrefix}/${row.stdout.slice(start + marker.length)}`;
    const bytes = get(path);
    if (!bytes || hash(bytes) !== row.stdoutSha256) throw Error(`archived raw response hash differs: ${row.id}`);
    return bytes;
  };
  for (const model of ["laya-q8", "kev-0.8b-q8"]) {
    const all = receipt.records.filter(row => row.model === model);
    const control = all.find(row => row.variant === "baseline");
    const expected = responseFor(control);
    for (const row of all) {
      if (row.exactRawResponseBytesMatch !== true || row.exactFullResponseMatch !== true || row.numericValuesCompared !== 5 || !responseFor(row).equals(expected)) throw Error(`raw response parity differs for ${row.id}`);
    }
  }
  const results = [];
  for (const model of ["laya-q8", "kev-0.8b-q8"]) {
    const rows = receipt.records.filter(row => row.model === model && row.phase === "measured");
    if (rows.length !== 36 || rows.some(row => row.exactRawResponseBytesMatch !== true || row.exactFullResponseMatch !== true || row.numericValuesCompared !== 5)) throw Error(`response evidence differs for ${model}`);
    const baselines = rows.filter(row => row.variant === "baseline").map(row => row.inferenceRequestMs);
    const candidates = rows.filter(row => row.variant === "candidate").map(row => row.inferenceRequestMs);
    if (baselines.length !== 18 || candidates.length !== 18) throw Error(`model arm counts differ for ${model}`);
    const quartets = [];
    for (let index = 0; index < 9; index++) {
      const quartet = rows.filter(row => row.index === index).sort((a, b) => a.position - b.position);
      const order = index % 2 ? "BAAB" : "ABBA";
      if (quartet.length !== 4 || quartet.map(row => row.variant === "baseline" ? "A" : "B").join("") !== order) throw Error(`paired schedule differs for ${model} quartet ${index}`);
      const base = quartet.filter(row => row.variant === "baseline").map(row => row.inferenceRequestMs);
      const candidate = quartet.filter(row => row.variant === "candidate").map(row => row.inferenceRequestMs);
      quartets.push((base[0] + base[1]) / (candidate[0] + candidate[1]));
    }
    const ratio = median(quartets);
    const expected = model === "laya-q8" ? 1.108545280408641 : 1.0772229441247445;
    if (ratio !== expected) throw Error(`paired ratio differs for ${model}`);
    results.push({ model, baselineIndependentMedianMs: median(baselines), candidateIndependentMedianMs: median(candidates), pairedMedianRatio: ratio, pairedRatioRange: [Math.min(...quartets), Math.max(...quartets)], wins: quartets.filter(value => value > 1).length, pairedLatencyReductionPercent: 100 * (1 - 1 / ratio), quartets });
  }
  return { scope: "recomputed from the preserved raw paired record", observations: 84, discarded: 12, measured: 72, responseBytesAndNumericValuesExact: true, models: results };
}

async function plan() {
  const result = await collect({ read: false });
  const files = [...result.groups.values()].flat();
  console.log(JSON.stringify({ mode: "metadata-only", baseCommit, candidateKernelSha256: candidateHash, groups: [...result.groups].map(([name, rows]) => ({ name, files: rows.length, sourceBytes: rows.reduce((sum, row) => sum + row.bytes, 0) })), totalFiles: files.length, totalSourceBytes: files.reduce((sum, row) => sum + row.bytes, 0), omitted: result.omitted, validation: result.validation }, null, 2));
}

function makeEnvelope(name, rows) {
  const files = rows.map(({ path, bytes, sha256, data }) => ({ path, bytes, sha256, encoding: "base64", data: data.toString("base64") }));
  const decoded = Buffer.from(JSON.stringify({ schemaVersion: 1, name, files }) + "\n");
  const compressed = gzipSync(decoded, { level: 9, mtime: 0 });
  return { decoded, compressed, files: rows.map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 })) };
}

async function archive() {
  const initialLease = assertLease();
  const result = await collect({ read: true });
  const fileMap = new Map([...result.groups.values()].flat().map(row => [`${row.group}/${row.path}`, row.data]));
  const modelStats = deriveModelStats(fileMap);
  const generated = new Date().toISOString().replace(/[:.]/g, "-");
  const validationId = result.validation ? result.validation.inventorySha256.slice(0, 12) : hash(Buffer.from("validation-pending")).slice(0, 12);
  const releaseId = `auto4-${validationId}-${generated}`;
  const releaseRoot = resolve(output, "releases", releaseId);
  const tempRoot = resolve(output, "releases", `.${releaseId}.partial`);
  for (const path of [releaseRoot, tempRoot]) {
    try { await lstat(path); throw Error(`refusing to overwrite previous archive: ${path}`); } catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  const envelopes = [];
  for (const [name, rows] of [...result.groups].sort(([a], [b]) => a.localeCompare(b))) {
    const envelope = makeEnvelope(name, rows), path = `raw/${name}.json.gz`;
    envelopes.push({ path, bytes: envelope.compressed.length, sha256: hash(envelope.compressed), decodedBytes: envelope.decoded.length, decodedSha256: hash(envelope.decoded), files: envelope.files, data: envelope.compressed });
  }
  const derived = Buffer.from(JSON.stringify(modelStats, null, 2) + "\n");
  const summary = Buffer.from(JSON.stringify({ schemaVersion: 1, releaseId, baseCommit, candidateKernelSha256: candidateHash, productionValidation: result.validation?.status === "passed" ? "passed" : "pending", modelStats, exclusions: ["model weights", "measured activation and hidden-state arrays", "compiled binaries and build trees", "credentials", "broad micro-screen disassembly retained by inventory hashes"] }, null, 2) + "\n");
  const files = [...result.groups.values()].flat();
  const manifest = { schemaVersion: 1, releaseId, generatedAt: generated, format: "deterministic gzip JSON envelopes containing exact base64 source bytes", baseCommit, candidateKernelSha256: candidateHash, initialLease, finalLease: null, productionValidation: result.validation, fileCount: files.length, sourceBytes: files.reduce((sum, row) => sum + row.bytes, 0), reports: [{ path: "summary.json", bytes: summary.length, sha256: hash(summary) }, { path: "derived-statistics.json", bytes: derived.length, sha256: hash(derived) }], omitted: result.omitted, envelopes: envelopes.map(({ data, ...entry }) => entry) };
  const parentLease = assertLease();
  if (!sameLease(parentLease, initialLease)) throw Error("canonical lease identity changed during archive preparation");
  manifest.finalLease = parentLease;
  await mkdir(resolve(output, "releases"), { recursive: true });
  const stage = resolve(output, "releases", `.${releaseId}.partial`);
  await mkdir(stage, { recursive: false });
  try {
    await mkdir(resolve(stage, "raw"));
    for (const item of envelopes) await writeFile(resolve(stage, item.path), item.data, { flag: "wx" });
    await writeFile(resolve(stage, "summary.json"), summary, { flag: "wx" });
    await writeFile(resolve(stage, "derived-statistics.json"), derived, { flag: "wx" });
    await writeFile(resolve(stage, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
    assertLease();
    await rename(stage, releaseRoot);
  } catch (error) {
    await rm(stage, { recursive: true, force: true });
    throw error;
  }
  assertLease();
  console.log(JSON.stringify({ archived: true, release: releaseRoot, groups: envelopes.length, files: files.length, sourceBytes: manifest.sourceBytes, modelStats: modelStats.models }, null, 2));
}

async function verify() {
  const initialLease = assertLease();
  const releases = resolve(output, "releases");
  let releaseRoot;
  if (state.release) {
    if (!/^auto4-[a-f0-9]{12}-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z$/.test(state.release)) throw Error("invalid --release id");
    releaseRoot = resolve(releases, state.release);
  } else {
    const children = (await import("node:fs/promises")).readdir(releases, { withFileTypes: true });
    const dirs = (await children).filter(entry => entry.isDirectory() && /^auto4-[a-f0-9]{12}-/.test(entry.name)).map(entry => entry.name).sort();
    if (dirs.length !== 1) throw Error("specify --release when zero or multiple immutable releases exist");
    releaseRoot = resolve(releases, dirs[0]);
  }
  const manifest = JSON.parse(await readFile(resolve(releaseRoot, "manifest.json"), "utf8"));
  const fileMap = new Map();
  let total = 0;
  for (const envelope of manifest.envelopes) {
    const compressed = await readFile(resolve(releaseRoot, envelope.path));
    if (compressed.length !== envelope.bytes || hash(compressed) !== envelope.sha256) throw Error(`compressed archive differs: ${envelope.path}`);
    const decoded = gunzipSync(compressed);
    if (decoded.length !== envelope.decodedBytes || hash(decoded) !== envelope.decodedSha256) throw Error(`decompressed archive differs: ${envelope.path}`);
    const payload = JSON.parse(decoded.toString("utf8"));
    if (payload.name !== envelope.path.slice("raw/".length, -".json.gz".length) || payload.files.length !== envelope.files.length) throw Error(`envelope identity/count differs: ${envelope.path}`);
    for (let index = 0; index < payload.files.length; index++) {
      const entry = payload.files[index], expected = envelope.files[index];
      const bytes = Buffer.from(entry.data, "base64");
      if (entry.path !== expected.path || entry.bytes !== expected.bytes || entry.sha256 !== expected.sha256 || bytes.length !== entry.bytes || hash(bytes) !== entry.sha256) throw Error(`embedded file differs: ${entry.path}`);
      const path = `${payload.name}/${entry.path}`;
      if (fileMap.has(path)) throw Error(`duplicate archived path: ${path}`);
      fileMap.set(path, bytes);
      total += bytes.length;
    }
  }
  if (total !== manifest.sourceBytes || fileMap.size !== manifest.fileCount) throw Error("manifest file or source byte count differs");
  const candidate = fileMap.get("source-and-protocol/candidate/crates/kevala/src/kernels.rs");
  if (!candidate || hash(candidate) !== candidateHash || manifest.candidateKernelSha256 !== candidateHash || manifest.baseCommit !== baseCommit) throw Error("archived kernel or base identity differs");
  for (const report of manifest.reports) {
    const bytes = await readFile(resolve(releaseRoot, report.path));
    if (bytes.length !== report.bytes || hash(bytes) !== report.sha256) throw Error(`report differs: ${report.path}`);
  }
  const modelStats = deriveModelStats(fileMap);
  const derived = await readFile(resolve(releaseRoot, "derived-statistics.json"));
  if (hash(derived) !== manifest.reports.find(row => row.path === "derived-statistics.json")?.sha256 || JSON.stringify(JSON.parse(derived)) !== JSON.stringify(modelStats)) throw Error("derived paired statistics differ from archived raw records");
  const finalLease = assertLease();
  if (!sameLease(finalLease, initialLease)) throw Error("canonical lease identity changed during verification");
  console.log(JSON.stringify({ verified: true, release: manifest.releaseId, envelopes: manifest.envelopes.length, files: fileMap.size, sourceBytes: total }, null, 2));
}

parseArgs();
if (state.mode === "--plan") await plan();
else if (state.mode === "--archive") await archive();
else await verify();
