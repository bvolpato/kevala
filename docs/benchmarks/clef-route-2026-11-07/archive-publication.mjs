import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, realpath } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const directory = dirname(fileURLToPath(import.meta.url));
const docsRoot = resolve(directory, "../../..");
const planPath = join(directory, "archive-plan.json");
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const encode = (value) => Buffer.from(JSON.stringify(value, null, 2) + "\n");
const order = (a, b) => Buffer.compare(Buffer.from(a.path), Buffer.from(b.path));
const safePath = (path) => typeof path === "string" && !path.includes("\\") && path.split("/").every((part) => part && part !== "." && part !== "..");
const args = process.argv.slice(2), mode = args.shift();
assert(["--plan", "--archive", "--verify"].includes(mode), "use --plan, --archive, or --verify");
const options = { campaignRoot: docsRoot };
while (args.length) {
  const key = args.shift();
  if (key === "--allow-io") options.allow = true;
  else if (["--campaign-root", "--receipt"].includes(key)) {
    const value = args.shift();
    assert(value && !value.startsWith("--"), `missing ${key} value`);
    options[key === "--receipt" ? "receipt" : "campaignRoot"] = resolve(value);
  } else throw Error(`unknown argument ${key}`);
}
async function plain(path, maximum) {
  const stat = await lstat(path);
  assert(stat.isFile() && !stat.isSymbolicLink() && stat.size <= maximum, `invalid or oversized file: ${path}`);
  const bytes = await readFile(path);
  assert(bytes.length <= maximum);
  return bytes;
}
async function absent(path) {
  try { await lstat(path); throw Error(`refusing existing output: ${path}`); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
}
async function durable(path, bytes) {
  const file = await open(path, "wx", 0o600);
  try { await file.writeFile(bytes); await file.sync(); } finally { await file.close(); }
  const parent = await open(dirname(path), "r");
  try { await parent.sync(); } finally { await parent.close(); }
}
const planBytes = await plain(planPath, 1048576), plan = JSON.parse(planBytes);
assert.equal(plan.schemaVersion, 1);
const limits = plan.archive;
assert(Number.isSafeInteger(limits.maxSourceBytes) && limits.maxSourceBytes > 0 && limits.maxSourceBytes <= 2097152);
assert(limits.maxDecodedBytes > 0 && limits.maxDecodedBytes <= 1048576);
assert(limits.maxCompressedBytes > 0 && limits.maxCompressedBytes <= 262144);
assert(plan.entries.length > 0 && plan.entries.length <= limits.maxEntries && limits.maxEntries <= 64);
const pins = new Map();
for (const pin of plan.sourcePins) {
  assert(typeof pin.id === "string" && !pins.has(pin.id));
  assert(safePath(pin.path) && /^(docs|tmp)\//.test(pin.path));
  assert(Number.isSafeInteger(pin.bytes) && pin.bytes >= 0 && pin.bytes <= limits.maxSourceBytes && /^[a-f0-9]{64}$/.test(pin.sha256));
  pins.set(pin.id, pin);
}
const paths = new Set();
for (const entry of plan.entries) {
  assert(safePath(entry.archivePath) && !paths.has(entry.archivePath));
  paths.add(entry.archivePath);
  assert(["copy-exact", "copy-plan", "json-select"].includes(entry.method));
  if (entry.method !== "copy-plan") assert(pins.has(entry.sourceId));
}
assert.equal(plan.entries.filter((entry) => entry.method === "copy-plan").length, 1);
const planIdentity = { path: "archive-plan.json", bytes: planBytes.length, sha256: hash(planBytes) };
if (mode === "--plan") {
  for (const [name, expected] of [
    ["fd-lease.mjs", "cf3ad8822a1386ec435723bb478b6a2ef1d753a0b9896c23e1b27d6e6d4b2915"],
    ["lease-parse.mjs", "6efc5046dc7b769f4c59ea256661581087b6e3ee1a366898b7852192697b5a31"],
  ]) assert.equal(hash(await plain(join(directory, name), 16384)), expected, name);
  const { createLeaseGuard } = await import(pathToFileURL(join(directory, "fd-lease.mjs")));
  const guard = createLeaseGuard({ canonicalPath: limits.requiredLock, recordEvidence: async (value) => value });
  const leaseBefore = await guard.bind("clef-route-publication-plan-admission");
  assert.equal(leaseBefore.exactInheritedFdOwnershipProved, true);
  const cache = new Map();
  async function source(id) {
    if (cache.has(id)) return cache.get(id);
    const pin = pins.get(id), root = await realpath(pin.path.startsWith("tmp/") ? options.campaignRoot : docsRoot);
    const path = resolve(root, pin.path), canonical = await realpath(path);
    assert(canonical.startsWith(root + sep), `source escaped its root: ${pin.path}`);
    const bytes = await plain(canonical, limits.maxSourceBytes);
    assert.equal(bytes.length, pin.bytes, pin.path); assert.equal(hash(bytes), pin.sha256, pin.path);
    cache.set(id, bytes); return bytes;
  }
  const files = [];
  for (const entry of plan.entries) {
    const bytes = entry.method === "copy-plan" ? planBytes : await source(entry.sourceId);
    assert.equal(entry.method === "copy-plan" || entry.method === "copy-exact", true, "plan mode accepts only fixed byte copies");
    files.push({ path: entry.archivePath, bytes: bytes.length, sha256: hash(bytes), encoding: "base64", data: bytes.toString("base64") });
  }
  files.sort(order);
  const decodedBytes = encode({ schemaVersion: 1, planSha256: planIdentity.sha256, files }).length;
  assert(decodedBytes <= limits.maxDecodedBytes, "decoded archive exceeds cap");
  assert.equal(decodedBytes, plan.archive.estimatedDecodedBytes, "plan's decoded-size estimate is stale");
  const leaseAfter = await guard.check("clef-route-publication-plan-complete");
  assert.deepEqual(leaseBefore.binding, leaseAfter.binding);
  assert.equal(leaseAfter.exactInheritedFdOwnershipProved, true);
  console.log(JSON.stringify({ status: "source-pins-and-size-verified", plan: planIdentity, entries: plan.entries.length,
    decodedBytes, compressedBytesMeasured: false, pinnedSourcesVerified: cache.size, exactInheritedFdOwnershipProved: true,
    postCheck: true, canonicalDevice: leaseAfter.binding.device, canonicalInode: leaseAfter.binding.inode,
    archiveReadOrCreated: false, gzipCreated: false }));
} else {
  assert(options.allow, "archive I/O requires --allow-io and the external canonical flock");
  const receiptPath = options.receipt || join(tmpdir(), `kevala-clef-route-diagnostic-${mode.slice(2)}-${process.pid}-${Date.now()}.json`);
  await absent(receiptPath);
  const evidenceDirectory = receiptPath + ".leases";
  await mkdir(evidenceDirectory, { mode: 0o700 });
  const state = { schemaVersion: 1, mode, pid: process.pid, startedAt: new Date().toISOString(), status: "running",
    plan: planIdentity, evidenceDirectory, leaseEvidence: [], privateSourcesRead: false, artifactIo: "not started" };
  let guard;
  try {
    for (const [name, expected] of [
      ["fd-lease.mjs", "cf3ad8822a1386ec435723bb478b6a2ef1d753a0b9896c23e1b27d6e6d4b2915"],
      ["lease-parse.mjs", "6efc5046dc7b769f4c59ea256661581087b6e3ee1a366898b7852192697b5a31"],
    ]) assert.equal(hash(await plain(join(directory, name), 16384)), expected, name);
    const { createLeaseGuard } = await import(pathToFileURL(join(directory, "fd-lease.mjs")));
    guard = createLeaseGuard({ canonicalPath: limits.requiredLock, recordEvidence: async (value) => {
      const bytes = encode(value), path = join(evidenceDirectory, `lease-${String(value.sequence).padStart(3, "0")}.json`);
      await durable(path, bytes);
      const reference = { path, bytes: bytes.length, sha256: hash(bytes) };
      state.leaseEvidence.push(reference);
      return reference;
    } });
    state.leaseBefore = await guard.bind("clef-route-publication-admission");
    const archivePath = join(directory, "evidence.json.gz"), manifestPath = join(directory, "manifest.json");
    if (mode === "--archive") {
      await absent(archivePath); await absent(manifestPath);
      const cache = new Map();
      async function source(id) {
        if (cache.has(id)) return cache.get(id);
        const pin = pins.get(id), root = await realpath(pin.path.startsWith("tmp/") ? options.campaignRoot : docsRoot);
        const path = resolve(root, pin.path), canonical = await realpath(path);
        assert(canonical.startsWith(root + sep), `source escaped its root: ${pin.path}`);
        state.privateSourcesRead ||= pin.path.startsWith("tmp/");
        const bytes = await plain(path, limits.maxSourceBytes);
        assert.equal(bytes.length, pin.bytes, pin.path); assert.equal(hash(bytes), pin.sha256, pin.path);
        cache.set(id, bytes); return bytes;
      }
      const files = [];
      for (const entry of plan.entries) {
        let bytes;
        if (entry.method === "copy-plan") bytes = planBytes;
        else {
          bytes = await source(entry.sourceId);
          if (entry.method === "json-select") {
            const original = JSON.parse(bytes), selected = {};
            for (const key of entry.select) {
              const parts = key.split(".");
              assert(parts.every((part) => part && !["__proto__", "prototype", "constructor"].includes(part)));
              let input = original, output = selected;
              for (const [index, part] of parts.entries()) {
                assert(input && Object.hasOwn(input, part), `missing selected field ${entry.sourceId}:${key}`);
                input = input[part];
                if (index === parts.length - 1) output[part] = input;
                else output = output[part] ||= {};
              }
            }
            for (const key of entry.drop || []) {
              const parts = key.split(".");
              assert(parts.every((part) => part && !["__proto__", "prototype", "constructor"].includes(part)));
              let output = selected;
              for (const part of parts.slice(0, -1)) output = output?.[part];
              if (output) delete output[parts.at(-1)];
            }
            if (entry.addSourceSha256) selected.sourceSha256 = pins.get(entry.sourceId).sha256;
            bytes = encode(selected);
          }
        }
        files.push({ path: entry.archivePath, bytes: bytes.length, sha256: hash(bytes), encoding: "base64", data: bytes.toString("base64") });
      }
      files.sort(order);
      const decoded = encode({ schemaVersion: 1, planSha256: planIdentity.sha256, files });
      assert(decoded.length <= limits.maxDecodedBytes, "decoded archive exceeds cap");
      const compressed = gzipSync(decoded, { level: 9, mtime: 0 });
      assert(compressed.length <= limits.maxCompressedBytes, "compressed archive exceeds cap");
      const manifest = { schemaVersion: 1, plan: planIdentity, archive: { path: "evidence.json.gz", bytes: compressed.length,
        sha256: hash(compressed), decodedBytes: decoded.length, decodedSha256: hash(decoded) },
      files: files.map(({ data, encoding, ...descriptor }) => descriptor),
      exclusions: { sources: plan.excludedSources, other: plan.otherExclusions } };
      await guard.check("clef-route-publication-before-artifact-writes");
      state.artifactIo = "possibly partial";
      await durable(archivePath, compressed); await durable(manifestPath, encode(manifest));
      state.archive = manifest.archive; state.entries = files.length; state.artifactIo = "created archive and manifest";
    } else {
      state.artifactIo = "possibly partial artifact reads";
      const manifest = JSON.parse(await plain(manifestPath, limits.maxDecodedBytes));
      assert.equal(manifest.schemaVersion, 1); assert.deepEqual(manifest.plan, planIdentity);
      assert.equal(manifest.archive.path, "evidence.json.gz");
      const compressed = await plain(archivePath, limits.maxCompressedBytes);
      assert.equal(compressed.length, manifest.archive.bytes); assert.equal(hash(compressed), manifest.archive.sha256);
      assert.equal(compressed.readUInt32LE(4), 0, "gzip timestamp must be zero");
      const decoded = gunzipSync(compressed, { maxOutputLength: limits.maxDecodedBytes });
      state.evidenceDecompressionPerformed = true;
      assert.equal(decoded.length, manifest.archive.decodedBytes); assert.equal(hash(decoded), manifest.archive.decodedSha256);
      const payload = JSON.parse(decoded);
      assert.equal(payload.schemaVersion, 1); assert.equal(payload.planSha256, planIdentity.sha256);
      assert.equal(payload.files.length, plan.entries.length);
      assert.deepEqual(payload.files, payload.files.slice().sort(order));
      const actualPaths = new Set(), descriptors = [];
      for (const { encoding, data, ...descriptor } of payload.files) {
        assert(safePath(descriptor.path) && !actualPaths.has(descriptor.path)); actualPaths.add(descriptor.path);
        assert.equal(encoding, "base64"); const bytes = Buffer.from(data, "base64");
        assert.equal(bytes.toString("base64"), data); assert.equal(bytes.length, descriptor.bytes); assert.equal(hash(bytes), descriptor.sha256);
        const entry = plan.entries.find((item) => item.archivePath === descriptor.path);
        assert(entry);
        if (entry.method === "copy-plan") assert(bytes.equals(planBytes));
        else if (entry.method === "copy-exact") {
          const pin = pins.get(entry.sourceId); assert.equal(bytes.length, pin.bytes); assert.equal(hash(bytes), pin.sha256);
        } else {
          const selected = JSON.parse(bytes);
          if (entry.addSourceSha256) assert.equal(selected.sourceSha256, pins.get(entry.sourceId).sha256);
        }
        descriptors.push(descriptor);
      }
      assert.deepEqual(descriptors, manifest.files);
      assert.deepEqual(manifest.exclusions, { sources: plan.excludedSources, other: plan.otherExclusions });
      state.archive = manifest.archive; state.entries = payload.files.length;
      state.artifactIo = "archive and manifest read and verified";
    }
    assert((await plain(planPath, 1048576)).equals(planBytes), "publication plan changed during execution");
    state.leaseAfter = await guard.check("clef-route-publication-complete");
    assert.deepEqual(state.leaseBefore.binding, state.leaseAfter.binding);
    state.status = "passed";
  } catch (error) {
    state.status = "failed"; state.error = String(error.stack || error); state.errorLeaseEvidenceRef = error.leaseEvidenceRef ?? null;
    if (mode === "--verify" && !state.evidenceDecompressionPerformed) state.evidenceDecompressionPerformed = "unknown; may have occurred";
    if (guard) try { state.leaseAfterFailure = await guard.check("clef-route-publication-failure"); } catch (leaseError) { state.leaseFailure = String(leaseError); }
    process.exitCode = 1;
  }
  state.completedAt = new Date().toISOString();
  await durable(receiptPath, encode(state));
  console.log(JSON.stringify({ status: state.status, mode, receiptPath, pid: process.pid, entries: state.entries, archive: state.archive,
    privateSourcesRead: state.privateSourcesRead, error: state.error }));
}
