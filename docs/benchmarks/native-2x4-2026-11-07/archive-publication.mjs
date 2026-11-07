import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { lstat, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";
import { isDeepStrictEqual } from "node:util";

const output = dirname(fileURLToPath(import.meta.url));
const repo = resolve(output, "../../..");
const args = process.argv.slice(2), mode = args.shift(), options = { evidence: null };
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const maxFileBytes = 32 * 1024 ** 2;
const profileLockPath = "/home/bruno/.cache/kevala/gpu-profile.lock";
const allowed = /\.(?:asm|json|jsonl|lock|md|mjs|patch|rs|stderr|stdout|toml|wgsl)$/;
const forbidden = /\.(?:wasm|kevala|safetensors|gguf|bin|pt|npy|npz|f32|f16|o|a|so|rlib|rmeta|gz|zst|zip)$/i;
const forbiddenPart = part => /^(?:target(?:-|$)|cargo-home(?:-|$)|node_modules$|\.git$|\.cache$|cache$|caches$|__pycache__$|packs$)$/i.test(part);
const omittedDisassembly = path => path === "micro-v2/assembly-baseline.stdout" || path === "micro-v2/assembly-candidate.stdout";
const inventorySpecs = [
  ["source-inventory-v1.json", "sources-v1", "b7a4d1c77c23139c51b889ff9809848ad6f04efec400d5129bb48876a5b07bad"],
  ["source-inventory-v2.json", "sources-v2", "bae30a4d40acff2e336a747a3617981412edb5911053773c4980f884daec8620"],
  ["final-inventory-v2.json", "final-v2", "66de1bfccda25611df4a80be7a5a81a98278d85700363f8ab8be93bb1ca10382"],
  ["gates-inventory-v1.json", "gates-v1", "155298d548589375289803549d105eabef03bf6c0474152cef0014a9821831e0"],
  ["gates-inventory-v2.json", "gates-v2", "37eec6c9dbd856bb35e4c3ed5ee84f4538c98d3b2b79e38a736c344b4239f26e"],
  ["micro-inventory-v2.json", "micro-v2", "f2f95eabf60928763dec13e46466873ba6775366f8300380765b15c640b85e97"],
  ["paired-native-v2/final-inventory.json", "paired-native-v2", "d346c96b770a0fbfecc7261a3d8cde3647f94d9d4924147d2ebd4d5077f7cd98", "paired-native-v2/"],
  ["production-validation-inventory-v2.json", "production-validation-v2", "ddb3d3d8fe8ebd4167d6fc61329641a400cf8f2c5bb944af42772df7528f1137"],
  ["numerical-artifact-binding-inventory-v2-r1.json", "artifact-binding-v2-r1", "3732451842f13e3e64498f23cf6e268fe0cfd20e4e8561ebcb45239fe8d9b73d"],
];
const pinned = [
  ["numerical-binding-amendment-v2-r1.json", "ce296ce68b0b50bff616142c1130a8d17fedbac01204e2c9e494dbd497611423", "artifact-binding-v2-r1"],
  ["numerical-binding-followup-review-r1.json", "e9305f61924a01ce6ea234e4fe060d1d4fe21327ff4d3767ea2b6717d9a6c52d", "artifact-binding-v2-r1"],
];

function parseArgs() {
  if (!["--plan", "--archive", "--verify"].includes(mode)) throw Error("use --plan, --archive --allow-io --evidence PATH, or --verify --allow-io");
  while (args.length) {
    const key = args.shift();
    if (key === "--allow-io") options.allowIo = true;
    else if (key === "--evidence") options.evidence = args.shift();
    else throw Error(`unknown argument: ${key}`);
  }
  if (mode !== "--verify" && (!options.evidence || !isAbsolute(options.evidence))) throw Error("--evidence must be an absolute path");
  if ((mode === "--archive" || mode === "--verify") && !options.allowIo) throw Error(`${mode} requires --allow-io and the parent's I/O window grant`);
}
function safeRelative(path) {
  if (typeof path !== "string" || !path || isAbsolute(path) || path.split(/[\\/]/).some(part => !part || part === "." || part === ".." || forbiddenPart(part))) throw Error(`unsafe input path: ${path}`);
  return path;
}
function archivePath(path) { return path.split(sep).join("/"); }
function assertProfileLock() {
  const inode = statSync(profileLockPath, { bigint: true }).ino.toString();
  const line = readFileSync("/proc/locks", "utf8").split("\n").find(value => {
    const fields = value.trim().split(/\s+/);
    return fields[1] === "FLOCK" && fields[3] === "WRITE" && fields[4] === String(process.pid) && fields[5]?.split(":").at(-1) === inode;
  });
  if (!line) throw Error("requires current process to hold the canonical GPU profile flock");
}
function evidenceRelative(path) {
  const root = resolve(options.evidence), full = isAbsolute(path) ? resolve(path) : resolve(root, path);
  const rel = relative(root, full);
  if (!rel || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw Error(`response path escapes evidence root: ${path}`);
  return archivePath(rel);
}
async function checkedStat(base, relativePath) {
  const safe = safeRelative(relativePath), parts = safe.split(/[\\/]/);
  let current = base;
  for (let i = 0; i < parts.length; i++) {
    current = resolve(current, parts[i]);
    const info = await lstat(current);
    if (info.isSymbolicLink() || (i < parts.length - 1 && !info.isDirectory()) || (i === parts.length - 1 && !info.isFile())) throw Error(`not a plain file path: ${relativePath}`);
  }
  const info = await lstat(current);
  if (info.size > maxFileBytes) throw Error(`input exceeds 32 MiB: ${relativePath}`);
  return { absolute: current, bytes: info.size };
}
function inspectFixture(path, bytes) {
  if (!path.includes("/tests/fixtures/")) return null;
  const root = JSON.parse(bytes.toString("utf8"));
  const fixture = path.slice(path.lastIndexOf("/tests/fixtures/") + "/tests/fixtures/".length);
  const topLevelKeys = root && typeof root === "object" && !Array.isArray(root) ? Object.keys(root).sort() : [];
  let arrays = 0, numericArrays = 0, maxArrayLength = 0, maxNumericArrayLength = 0;
  const tensorLikeFields = new Set();
  const visit = (value, at) => {
    if (Array.isArray(value)) {
      arrays++;
      maxArrayLength = Math.max(maxArrayLength, value.length);
      if (value.length && value.every(item => typeof item === "number" && Number.isFinite(item))) {
        numericArrays++;
        maxNumericArrayLength = Math.max(maxNumericArrayLength, value.length);
      }
      value.forEach((item, index) => visit(item, `${at}[${index}]`));
    } else if (value && typeof value === "object") {
      for (const [key, child] of Object.entries(value)) {
        if (/(?:weight|activation|hidden|tensor|kv.?cache|embedding|patch|values?)/i.test(key)) tensorLikeFields.add(key);
        visit(child, at ? `${at}.${key}` : key);
      }
    }
  };
  visit(root, "");
  return { fixture, sha256: hash(bytes), topLevelKeys, arrays, numericArrays, maxArrayLength, maxNumericArrayLength, tensorLikeFields: [...tensorLikeFields].sort() };
}

async function collect(readPayloads) {
  const baseInfo = await lstat(options.evidence);
  if (baseInfo.isSymbolicLink() || !baseInfo.isDirectory()) throw Error("evidence root must be a real directory");
  const groups = new Map(), selected = new Map(), inventoryDescriptors = new Map(), omitted = [], missing = [], fixtureAudit = new Map();
  const add = (group, path, bytes, expectedSha = null, expectedBytes = null) => {
    safeRelative(path);
    if (selected.has(path)) return;
    if (!allowed.test(path) || forbidden.test(path)) throw Error(`unapproved input type: ${path}`);
    if (expectedBytes !== null && bytes.length !== expectedBytes) throw Error(`byte count differs from inventory: ${path}`);
    const digest = hash(bytes);
    if (expectedSha && digest !== expectedSha) throw Error(`hash differs from inventory: ${path}`);
    const fixture = inspectFixture(path, bytes);
    if (fixture) {
      const key = `${fixture.fixture}:${fixture.sha256}`;
      const prior = fixtureAudit.get(key);
      if (prior) prior.paths.push(path);
      else fixtureAudit.set(key, { ...fixture, paths: [path] });
    }
    const entry = { group, path: `native-2x4/${archivePath(path)}`, sourcePath: path, bytes: bytes.length, sha256: digest, data: bytes };
    selected.set(path, entry);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(entry);
  };
  for (const [inventoryPath, group, expectedInventorySha, prefix = ""] of inventorySpecs) {
    const inventoryStat = await checkedStat(options.evidence, inventoryPath);
    const inventoryBytes = readPayloads ? await readFile(inventoryStat.absolute) : null;
    if (readPayloads && hash(inventoryBytes) !== expectedInventorySha) throw Error(`inventory hash changed: ${inventoryPath}`);
    let inventory;
    try { inventory = JSON.parse((inventoryBytes ?? await readFile(inventoryStat.absolute)).toString("utf8")); }
    catch (error) { throw Error(`invalid inventory ${inventoryPath}: ${error.message}`); }
    if (!inventory.files || typeof inventory.files !== "object") throw Error(`invalid files map: ${inventoryPath}`);
    if (readPayloads) add(group, inventoryPath, inventoryBytes, expectedInventorySha, inventoryStat.bytes);
    else selected.set(inventoryPath, { group, sourcePath: inventoryPath, bytes: inventoryStat.bytes, sha256: expectedInventorySha, metadataOnly: true });
    for (const [path, descriptor] of Object.entries(inventory.files).sort(([a], [b]) => a.localeCompare(b))) {
      const sourcePath = `${prefix}${path}`;
      safeRelative(sourcePath);
      const priorDescriptor = inventoryDescriptors.get(sourcePath);
      if (priorDescriptor && (priorDescriptor.bytes !== descriptor.bytes || priorDescriptor.sha256 !== descriptor.sha256)) throw Error(`inventories disagree: ${sourcePath}`);
      inventoryDescriptors.set(sourcePath, descriptor);
      if (omittedDisassembly(sourcePath)) {
        omitted.push({ path: `native-2x4/${sourcePath}`, bytes: descriptor.bytes, sha256: descriptor.sha256, reason: "complete broad microbenchmark disassembly omitted; scoped function extracts are retained" });
        continue;
      }
      if (!allowed.test(sourcePath) || forbidden.test(sourcePath)) {
        omitted.push({ path: `native-2x4/${sourcePath}`, bytes: descriptor.bytes, sha256: descriptor.sha256, reason: "binary, cache, or non-source payload excluded by fixed allowlist" });
        continue;
      }
      try {
        const info = await checkedStat(options.evidence, sourcePath);
        if (info.bytes !== descriptor.bytes) throw Error(`byte count differs from inventory: ${sourcePath}`);
        if (readPayloads) add(group, sourcePath, await readFile(info.absolute), descriptor.sha256, descriptor.bytes);
        else selected.set(sourcePath, { group, sourcePath, bytes: descriptor.bytes, sha256: descriptor.sha256, metadataOnly: true });
      } catch (error) {
        if (error.code === "ENOENT") missing.push({ path: `native-2x4/${sourcePath}`, expected: descriptor });
        else throw error;
      }
    }
  }
  for (const [path, expectedSha, group] of pinned) {
    try {
      const info = await checkedStat(options.evidence, path);
      if (readPayloads) add(group, path, await readFile(info.absolute), expectedSha, info.bytes);
      else selected.set(path, { group, sourcePath: path, bytes: info.bytes, sha256: expectedSha, metadataOnly: true });
    } catch (error) {
      if (error.code === "ENOENT") missing.push({ path: `native-2x4/${path}`, expectedSha256: expectedSha });
      else throw error;
    }
  }
  for (const [path, group] of [["docs/native-2x4-performance.md", "publication-source"], ["docs/benchmarks/native-2x4-2026-11-07/README.md", "publication-source"], ["docs/benchmarks/native-2x4-2026-11-07/archive-publication.mjs", "publication-source"]]) {
    const info = await checkedStat(repo, path);
    if (readPayloads) add(group, `publication/${path.slice("docs/".length)}`, await readFile(info.absolute), null, info.bytes);
    else selected.set(`publication/${path.slice("docs/".length)}`, { group, sourcePath: path, bytes: info.bytes, sha256: null, metadataOnly: true });
  }
  for (const entries of groups.values()) entries.sort((a, b) => a.path.localeCompare(b.path));
  return { groups, selected, omitted, missing, fixtureAudit: [...fixtureAudit.values()].sort((a, b) => a.fixture.localeCompare(b.fixture) || a.sha256.localeCompare(b.sha256)) };
}

const median = values => {
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
function validateCorrection(files) {
  const get = path => {
    const bytes = files.get(`native-2x4/${path}`);
    if (!bytes) throw Error(`required validation input missing: ${path}`);
    return bytes;
  };
  const originalReceiptBytes = get("production-validation-v2/receipt.json");
  const correctionReceiptBytes = get("numerical-artifact-binding-v2-r1/receipt.json");
  const finalReceiptBytes = get("final-receipt-v2.json");
  const amendmentBytes = get("numerical-binding-amendment-v2-r1.json");
  const originalInventory = JSON.parse(get("production-validation-inventory-v2.json"));
  const finalInventory = JSON.parse(get("final-inventory-v2.json"));
  const correctionInventory = JSON.parse(get("numerical-artifact-binding-inventory-v2-r1.json"));
  const auditBytes = get("numerical-binding-audit-v2.json");
  const originalReceiptSha256 = hash(originalReceiptBytes), correctiveReceiptSha256 = hash(correctionReceiptBytes), finalReceiptSha256 = hash(finalReceiptBytes), amendmentSha256 = hash(amendmentBytes);
  const amendment = JSON.parse(amendmentBytes), receipt = JSON.parse(correctionReceiptBytes);
  if (originalInventory.files?.["production-validation-v2/receipt.json"]?.sha256 !== originalReceiptSha256) throw Error("original validation receipt differs from its inventory");
  if (finalInventory.files?.["final-receipt-v2.json"]?.sha256 !== finalReceiptSha256) throw Error("prior final receipt differs from its inventory");
  if (correctionInventory.files?.["numerical-artifact-binding-v2-r1/receipt.json"]?.sha256 !== correctiveReceiptSha256) throw Error("corrective receipt differs from its inventory");
  if (amendment.originalFinalReceipt?.sha256 !== finalReceiptSha256 || amendment.audit?.sha256 !== hash(auditBytes) || amendment.correctiveReceipt?.sha256 !== correctiveReceiptSha256) throw Error("binding amendment does not identify the included receipts and audit");
  if (amendment.exactGeneratedArtifactBindingEstablished !== true || amendment.inputVerificationComplete !== true || amendment.inputVerificationFailures?.length !== 0 || amendment.timedRustSourceUnchanged !== true || amendment.previousReceiptsPreserved !== true || amendment.failure !== null) throw Error("binding amendment does not record a complete correction");
  if (receipt.originalValidationReceiptSha256 !== originalReceiptSha256 || receipt.previousFinalReceiptSha256 !== finalReceiptSha256 || receipt.complete !== true || receipt.exactGeneratedArtifactBindingEstablished !== true || receipt.overrideRemovedFromActualSpawnEnv !== true || receipt.timedRustEdited !== false || receipt.guardScriptEdited !== false || receipt.timedRustSourceUnchanged !== true || receipt.previousReceiptsPreserved !== true || receipt.microOrModelTimingsRun !== false || receipt.retries !== 0) throw Error("corrective receipt does not close the recorded artifact-binding gap");
  const flavors = ["base", "relaxed", "simd"], commands = receipt.commands ?? [];
  if (commands.length !== 3 || new Set(commands.map(row => row.flavor)).size !== 3 || flavors.some(flavor => !commands.some(row => row.flavor === flavor))) throw Error("corrective run is not exactly the three required flavors");
  for (const command of commands) {
    const artifactIndex = command.args.indexOf("--artifact"), requestedPath = command.args[artifactIndex + 1];
    const before = receipt.artifactsBefore?.[command.flavor], after = receipt.artifactsAfter?.[command.flavor], report = receipt.reports?.[command.flavor];
    if (command.status !== 0 || command.environment?.KEVALA_CPU_BENCH_WASM !== null || command.environment?.overrideHasOwnAtSpawn !== false || artifactIndex < 0 || !requestedPath || requestedPath !== before?.resolvedPath || requestedPath !== after?.resolvedPath || before?.sha256 !== after?.sha256 || before?.bytes !== after?.bytes || report?.artifactBefore?.sha256 !== before?.sha256 || report?.artifactAfter?.sha256 !== after?.sha256 || !(report?.passed > 0)) throw Error(`corrective artifact binding differs: ${command.flavor}`);
  }
  return { originalReceiptSha256, originalInventorySha256: hash(get("production-validation-inventory-v2.json")), correctiveReceiptSha256, correctionInventorySha256: hash(get("numerical-artifact-binding-inventory-v2-r1.json")), amendmentSha256, previousFinalReceiptSha256: finalReceiptSha256, correctionAmendmentAgreementVerified: true };
}
function deriveStatistics(files) {
  const get = path => files.get(path);
  const finalValidation = validateCorrection(files);
  const paired = JSON.parse(get("native-2x4/paired-native-v2/runs-v1/receipt.json").toString("utf8"));
  const published = JSON.parse(get("native-2x4/paired-native-v2/final-summary.json").toString("utf8"));
  const records = paired.records;
  if (records.length !== 84 || records.filter(row => row.phase === "discard").length !== 12 || records.filter(row => row.phase === "measured").length !== 72) throw Error("paired CLI observation counts differ");
  const pairedRows = [];
  for (const model of [...new Set(records.map(row => row.model))].sort()) {
    const measured = records.filter(row => row.phase === "measured" && row.model === model);
    const baseline = measured.filter(row => row.variant === "baseline").map(row => row.inferenceRequestMs);
    const candidate = measured.filter(row => row.variant === "candidate").map(row => row.inferenceRequestMs);
    if (measured.length !== 36 || baseline.length !== 18 || candidate.length !== 18) throw Error(`paired arm count differs: ${model}`);
    const quartets = [];
    for (let index = 0; index < 9; index++) {
      const rows = measured.filter(row => row.index === index).sort((a, b) => a.position - b.position);
      const order = index % 2 ? "BAAB" : "ABBA";
      if (rows.length !== 4 || rows.map(row => row.variant === "baseline" ? "A" : "B").join("") !== order) throw Error(`paired order differs: ${model} ${index}`);
      const a = rows.filter(row => row.variant === "baseline").map(row => row.inferenceRequestMs);
      const b = rows.filter(row => row.variant === "candidate").map(row => row.inferenceRequestMs);
      quartets.push({ index, order, baselineRequestMs: a, candidateRequestMs: b, pairedRatio: a.reduce((sum, value) => sum + value, 0) / b.reduce((sum, value) => sum + value, 0) });
    }
    const ratios = quartets.map(row => row.pairedRatio), publicRow = published.summaries.find(row => row.model === model);
    const baseMedian = median(baseline), candidateMedian = median(candidate), pairedMedianRatio = median(ratios);
    if (!publicRow || baseMedian !== publicRow.baselineIndependentMedianRequestMs || candidateMedian !== publicRow.candidateIndependentMedianRequestMs || pairedMedianRatio !== publicRow.pairedMedianRatio) throw Error(`paired derivation differs from preserved summary: ${model}`);
    const controlRecord = records.find(row => row.model === model && row.variant === "baseline");
    const control = get(`native-2x4/${evidenceRelative(controlRecord.stdout)}`);
    if (!control) throw Error(`paired response missing: ${model}`);
    for (const row of records.filter(row => row.model === model)) {
      const responsePath = `native-2x4/${evidenceRelative(row.stdout)}`;
      if (!get(responsePath)?.equals(control)) throw Error(`raw response bytes differ from same-model control: ${model} ${row.id}`);
    }
    pairedRows.push({ model, baselineIndependentMedianRequestMs: baseMedian, candidateIndependentMedianRequestMs: candidateMedian, pairedMedianRatio, pairedRatioRange: [Math.min(...ratios), Math.max(...ratios)], candidateFasterQuartets: ratios.filter(value => value > 1).length, quartets, all84ModelResponsesByteExact: true });
  }
  const micro = JSON.parse(get("native-2x4/micro-v2/receipt.json").toString("utf8"));
  const rawMicro = get("native-2x4/micro-v2/paired-screen.stdout").toString("utf8").trim().split("\n").map(JSON.parse);
  const microRows = rawMicro.map(row => {
    if (row.roundsNs.length !== 9 || !Number.isSafeInteger(row.repetitions) || row.repetitions <= 0) throw Error(`micro rounds differ: ${row.name}`);
    const baseline = [], candidate = [], ratios = [];
    for (let index = 0; index < 9; index++) {
      const sample = row.roundsNs[index], a = index % 2 ? [1, 2] : [0, 3], b = index % 2 ? [0, 3] : [1, 2];
      baseline.push(...a.map(i => sample[i])); candidate.push(...b.map(i => sample[i]));
      ratios.push((sample[a[0]] + sample[a[1]]) / (sample[b[0]] + sample[b[1]]));
    }
    const expected = micro.rows.find(item => item.name === row.name), pairedMedianRatio = median(ratios);
    const baselineConventionalMedianNsPerCall = median(baseline) / row.repetitions, candidateConventionalMedianNsPerCall = median(candidate) / row.repetitions;
    if (!expected || expected.pairedMedianRatio !== pairedMedianRatio || expected.baselineConventionalMedianNsPerCall !== baselineConventionalMedianNsPerCall || expected.candidateConventionalMedianNsPerCall !== candidateConventionalMedianNsPerCall) throw Error(`micro derivation differs from preserved receipt: ${row.name}`);
    return { name: row.name, repetitions: row.repetitions, baselineConventionalMedianNsPerCall, candidateConventionalMedianNsPerCall, pairedMedianRatio, pairedRatioRange: [Math.min(...ratios), Math.max(...ratios)], candidateFasterQuartets: ratios.filter(value => value > 1).length, pairedRatios: ratios };
  });
  return { schemaVersion: 1, scope: "recomputed from preserved raw paired observations; original summaries and records are unchanged", finalValidation, pairedCli: pairedRows, microScreen: microRows };
}

async function plan() {
  const result = await collect(false);
  const fixturePaths = [...new Set([...result.selected.values()].map(row => row.sourcePath).filter(path => path?.includes("/tests/fixtures/")).map(path => path.slice(path.lastIndexOf("/tests/fixtures/") + "/tests/fixtures/".length)))].sort();
  console.log(JSON.stringify({ mode: "metadata-only", evidence: options.evidence, selectedFiles: result.selected.size, selectedBytes: [...result.selected.values()].reduce((sum, row) => sum + row.bytes, 0), missing: result.missing, omittedDisassembly: result.omitted, fixturePathsToInspect: fixturePaths, groups: [...new Set([...result.selected.values()].map(row => row.group))].sort() }, null, 2));
}
async function archive() {
  assertProfileLock();
  const result = await collect(true);
  if (result.missing.length) throw Error(`missing inventoried inputs: ${result.missing.map(row => row.path).join(", ")}`);
  const fileBytes = new Map([...result.selected.values()].map(row => [row.path, row.data]));
  const derived = deriveStatistics(fileBytes);
  const derivedBytes = Buffer.from(JSON.stringify(derived, null, 2) + "\n");
  const summary = { schemaVersion: 1, archiveSet: "native-2x4-2026-11-07", sourceBase: "97778af0ed881c876313fcefa5a17399ceb10f2d", candidateKernelSha256: "d7bfece0d5fdbaa4e22c6681ff32720e4bbe074378d83fd61f814b0dc280623f", pairedResponses: 84, discardedObservations: 12, measuredObservations: 72, microTelemetryIncluded: true, publicFixtureAudit: result.fixtureAudit, excludedPayloadPolicy: ["model weights", "measured activation and hidden-state buffers; tracked public source fixtures are listed separately", "compiled binaries", "build output and caches", "credentials", "two full microbenchmark disassemblies retained by exact hash descriptor"], omittedDisassembly: result.omitted };
  const summaryBytes = Buffer.from(JSON.stringify(summary, null, 2) + "\n");
  const archives = [], compressed = [];
  for (const [name, rows] of [...result.groups].sort(([a], [b]) => a.localeCompare(b))) {
    const files = rows.map(({ path, bytes, sha256, data }) => ({ path, bytes, sha256, encoding: "base64", data: data.toString("base64") }));
    const decoded = Buffer.from(JSON.stringify({ schemaVersion: 1, name, files }) + "\n");
    const bytes = gzipSync(decoded, { level: 9, mtime: 0 }), path = `raw/${name}.json.gz`;
    compressed.push([path, bytes]);
    archives.push({ path, bytes: bytes.length, sha256: hash(bytes), decodedBytes: decoded.length, decodedSha256: hash(decoded), files: rows.map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 })) });
  }
  const reports = [["summary.json", summaryBytes], ["derived-statistics.json", derivedBytes]].map(([path, bytes]) => ({ path, bytes: bytes.length, sha256: hash(bytes) }));
  const manifest = { schemaVersion: 1, archiveSet: "native-2x4-2026-11-07", format: "deterministic gzip JSON with exact base64 file bytes", evidenceRoot: resolve(options.evidence), commonBase: "97778af0ed881c876313fcefa5a17399ceb10f2d", candidateKernelSha256: "d7bfece0d5fdbaa4e22c6681ff32720e4bbe074378d83fd61f814b0dc280623f", finalValidation: { ...derived.finalValidation, independentArtifactBindingAuditPassed: true }, fileCount: result.selected.size, reports, omitted: result.omitted, archives };
  for (const path of [resolve(output, "raw"), resolve(output, "manifest.json"), resolve(output, "summary.json"), resolve(output, "derived-statistics.json")]) {
    try { await lstat(path); throw Error(`refusing existing output: ${path}`); } catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  assertProfileLock();
  await mkdir(resolve(output, "raw"), { recursive: false });
  for (const [path, bytes] of compressed) await writeFile(resolve(output, path), bytes, { flag: "wx" });
  await writeFile(resolve(output, "summary.json"), summaryBytes, { flag: "wx" });
  await writeFile(resolve(output, "derived-statistics.json"), derivedBytes, { flag: "wx" });
  await writeFile(resolve(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  assertProfileLock();
  console.log(JSON.stringify({ archived: true, groups: archives.length, files: result.selected.size, sourceBytes: [...result.selected.values()].reduce((sum, row) => sum + row.bytes, 0) }));
}
async function verify() {
  assertProfileLock();
  const manifest = JSON.parse(await readFile(resolve(output, "manifest.json"), "utf8")), files = new Map();
  if (typeof manifest.evidenceRoot !== "string" || !isAbsolute(manifest.evidenceRoot)) throw Error("manifest must identify the original evidence root");
  options.evidence = manifest.evidenceRoot;
  if (!Array.isArray(manifest.archives) || !Array.isArray(manifest.reports) || !Number.isSafeInteger(manifest.fileCount)) throw Error("invalid archive manifest shape");
  const declaredArchivePaths = new Set(), declaredFilePaths = new Set();
  let declaredCount = 0;
  for (const archive of manifest.archives) {
    if (!/^raw\/[a-z0-9-]+\.json\.gz$/.test(archive.path) || declaredArchivePaths.has(archive.path)) throw Error(`invalid or duplicate archive path: ${archive.path}`);
    declaredArchivePaths.add(archive.path);
    if (!Array.isArray(archive.files)) throw Error(`invalid file list: ${archive.path}`);
    const local = new Set();
    for (const entry of archive.files) {
      if (typeof entry.path !== "string" || local.has(entry.path) || declaredFilePaths.has(entry.path)) throw Error(`duplicate or invalid declared file path: ${entry.path}`);
      local.add(entry.path);
      declaredFilePaths.add(entry.path);
      declaredCount++;
    }
  }
  if (declaredCount !== manifest.fileCount) throw Error("manifest file count differs from archive envelopes");
  const reportPaths = new Set();
  for (const report of manifest.reports) {
    if (!new Set(["summary.json", "derived-statistics.json"]).has(report.path) || reportPaths.has(report.path)) throw Error(`invalid or duplicate report path: ${report.path}`);
    reportPaths.add(report.path);
    const bytes = await readFile(resolve(output, report.path));
    if (bytes.length !== report.bytes || hash(bytes) !== report.sha256) throw Error(`report differs: ${report.path}`);
  }
  if (reportPaths.size !== 2) throw Error("manifest must bind both reports");
  const observedFilePaths = new Set();
  for (const archive of manifest.archives) {
    const compressed = await readFile(resolve(output, archive.path));
    if (compressed.length !== archive.bytes || hash(compressed) !== archive.sha256) throw Error(`compressed bytes differ: ${archive.path}`);
    const decoded = gunzipSync(compressed);
    if (decoded.length !== archive.decodedBytes || hash(decoded) !== archive.decodedSha256) throw Error(`decoded bytes differ: ${archive.path}`);
    const envelope = JSON.parse(decoded);
    if (envelope.name !== archive.path.slice("raw/".length, -".json.gz".length) || !Array.isArray(envelope.files) || envelope.files.length !== archive.files.length) throw Error(`envelope count or identity differs: ${archive.path}`);
    const expectedFiles = new Map(archive.files.map(entry => [entry.path, entry]));
    if (expectedFiles.size !== archive.files.length) throw Error(`duplicate declared path: ${archive.path}`);
    const envelopePaths = new Set();
    for (const row of envelope.files) {
      if (typeof row.path !== "string" || envelopePaths.has(row.path) || observedFilePaths.has(row.path)) throw Error(`duplicate envelope path: ${row.path}`);
      envelopePaths.add(row.path);
      observedFilePaths.add(row.path);
      const data = Buffer.from(row.data, "base64"), expected = expectedFiles.get(row.path);
      if (!expected || row.encoding !== "base64" || data.length !== row.bytes || hash(data) !== row.sha256 || row.bytes !== expected.bytes || row.sha256 !== expected.sha256) throw Error(`entry differs: ${row.path}`);
      files.set(row.path, data);
    }
    if (envelopePaths.size !== expectedFiles.size || [...expectedFiles.keys()].some(path => !envelopePaths.has(path))) throw Error(`envelope file set differs: ${archive.path}`);
  }
  if (files.size !== manifest.fileCount || observedFilePaths.size !== declaredFilePaths.size || [...declaredFilePaths].some(path => !observedFilePaths.has(path))) throw Error("global archive file set differs from manifest");
  const derived = deriveStatistics(files), derivedBytes = await readFile(resolve(output, "derived-statistics.json"));
  if (!isDeepStrictEqual(derived, JSON.parse(derivedBytes))) throw Error("derived statistics differ from archived raw inputs");
  assertProfileLock();
  console.log(JSON.stringify({ verified: true, archives: manifest.archives.length, files: files.size }));
}

parseArgs();
if (mode === "--plan") await plan();
else if (mode === "--archive") await archive();
else await verify();
